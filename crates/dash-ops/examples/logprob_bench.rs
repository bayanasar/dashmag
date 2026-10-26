//! Time and peak memory of the chosen-token log-probability at a 248320-token vocabulary and a
//! hidden size of 1024, bf16 inputs, on the first CUDA device.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example logprob_bench \
//!       -- [n ...] [--vocab-chunk C] [--row-chunk R] [--reps K]
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example logprob_bench \
//!       -- --matmul
//!
//! For each n (default 1200 and 3072), first memory, over the first forward + backward passes
//! the device runs: the most bytes the allocator has in use above what it held before (the
//! inputs and the weight), sampled from a second thread, beside the count of the buffers the
//! operation holds together at its peak, and how far the reservation grew. Then time: forward +
//! backward (`hidden` requires a gradient, the upstream gradient is random), and the forward
//! alone without autodiff, each the median of K timed repetitions (default 10) after 3 warm-up
//! ones, with a device sync around each. The masked token is one near the end of the vocabulary.
//!
//! `--matmul` times the two matmuls of one chunk alone, at several row counts and chunk widths,
//! on the CubeCL backend directly (so under fusion too); with autotune, the kernels it picks.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("logprob_bench measures the CUDA path: build it with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    bench::main();
}

#[cfg(feature = "cuda")]
mod bench {
    use burn::backend::{Backend, DType};
    use burn::tensor::{Device, Distribution, Int, Tensor};
    use burn_cubecl::{CubeBackend, CubeDevice};
    use dash_ops::logprob::{LogprobOptions, chosen_logprob};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Instant;

    const V: usize = 248_320;
    const D: usize = 1024;
    /// A token near the end of the vocabulary, taken out of the softmax.
    const MASKED: u32 = 248_044;

    pub fn main() {
        let mut ns = Vec::new();
        let mut options = LogprobOptions::default();
        let mut reps = 10;
        let mut args = std::env::args().skip(1);
        if std::env::args().any(|a| a == "--matmul") {
            matmul_sweep();
            return;
        }
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .expect("a value")
                    .parse::<usize>()
                    .expect("a number")
            };
            match arg.as_str() {
                "--vocab-chunk" => options.vocab_chunk = value(),
                "--row-chunk" => options.row_chunk = value(),
                "--reps" => reps = value(),
                n => ns.push(n.parse().expect("n")),
            }
        }
        if ns.is_empty() {
            ns = vec![1200, 3072];
        }
        println!("V {V}, D {D}, bf16 inputs, {options:?}");
        for n in ns {
            run(n, options, reps);
        }
    }

    /// The two matmuls of one chunk, bf16 operands to an f32 result, at several row counts and
    /// chunk widths: `[m, D] x [D, c]` (the logits, the weight's rows as columns) and
    /// `[m, c] x [c, D]` (the gradient's).
    fn matmul_sweep() {
        use burn::backend::ops::FloatTensorOps;
        use burn::backend::{FloatDType, Shape};
        use burn_cubecl::kernel::matmul::{MatmulStrategy, matmul};
        // The CubeCL backend's own device and tensors, not the dispatch ones: under fusion those
        // wrap the backend, and the matmul here is the one the operation calls inside it.
        let device = CubeDevice::cuda(0).expect("a CUDA device");
        let random = |shape: [usize; 2]| {
            CubeBackend::float_random(
                Shape::new(shape),
                Distribution::Normal(0.0, 1.0),
                &device,
                FloatDType::BF16,
            )
        };
        for (m, c) in [
            (1200, 4096),
            (1200, 8192),
            (1200, 16384),
            (1200, 32768),
            (2048, 8192),
            (3072, 4096),
            (3072, 8192),
            (3072, 16384),
            (3584, 8192),
            (4096, 4096),
        ] {
            let a = random([m, D]);
            let w = random([c, D]);
            let p = random([m, c]);
            let sync = || <CubeBackend as Backend>::sync(&device).unwrap();
            let time = |f: &dyn Fn()| {
                for _ in 0..3 {
                    f();
                }
                sync();
                let start = Instant::now();
                for _ in 0..10 {
                    f();
                }
                sync();
                start.elapsed().as_secs_f64() * 1e3 / 10.0
            };
            let logits = time(&|| {
                let wt = CubeBackend::float_transpose(w.clone());
                let _ = matmul(a.clone(), wt, None, MatmulStrategy::default(), DType::F32).unwrap();
            });
            let grad = time(&|| {
                let _ = matmul(
                    p.clone(),
                    w.clone(),
                    None,
                    MatmulStrategy::default(),
                    DType::F32,
                )
                .unwrap();
            });
            let flop = 2.0 * m as f64 * c as f64 * D as f64;
            println!(
                "m {m:5} c {c:5}: [m, {D}] x [{D}, c]^T {logits:6.2} ms ({:5.1} TFLOPS)   [m, c] x [c, {D}] {grad:6.2} ms ({:5.1} TFLOPS)",
                flop / logits / 1e9,
                flop / grad / 1e9
            );
        }
    }

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / (1u64 << 20) as f64
    }

    fn run(n: usize, options: LogprobOptions, reps: usize) {
        let device = Device::cuda(0);
        device.seed(n as u64);
        // Made in bf16 directly: no f32 temporary to leave the allocator a larger peak than the
        // operation's own.
        let bf16 = (&device, DType::BF16);
        let weight = Tensor::<2>::random([V, D], Distribution::Normal(0.0, 0.02), bf16);
        let hidden = Tensor::<2>::random([n, D], Distribution::Normal(0.0, 1.0), bf16);
        let targets = Tensor::<1, Int>::random([n], Distribution::Uniform(0.0, V as f64), &device);
        let upstream = Tensor::<1>::random([n], Distribution::Normal(0.0, 1.0), &device);
        device.sync().unwrap();

        let train_step = || {
            let h = hidden.clone().autodiff().require_grad();
            let out = chosen_logprob(
                h.clone(),
                weight.clone(),
                targets.clone(),
                &[MASKED],
                options,
            );
            let grads = (out.logprob * upstream.clone()).sum().backward();
            let grad = h.grad(&grads).expect("a gradient");
            (grad, out.entropy)
        };
        let forward = || {
            chosen_logprob(
                hidden.clone(),
                weight.clone(),
                targets.clone(),
                &[MASKED],
                options,
            )
        };

        // Memory, over the first forward + backward passes the device runs: how far the
        // allocator's reservation grew (what the device had to provide, page by page), and the
        // most bytes in use above the inputs, sampled from a second thread over three passes (a
        // lower bound: a pass queues its chunks within a few milliseconds of host time, and the
        // sampler sees only some moments of them). The sampler reads the CubeCL backend's
        // allocator itself: under fusion the operation runs on the fusion server's thread, and
        // asking the dispatch device would wait for the whole of it. Beside it, the count of the
        // buffers the operation holds together at its peak: a chunk of logits, the softmax
        // weights in two bf16 halves, `o`, the gradient and the outputs.
        let cube_device = CubeDevice::cuda(0).expect("a CUDA device");
        let usage = move || {
            <CubeBackend as Backend>::memory_pool_usage(&cube_device)
                .expect("the allocator reports usage")
        };
        device.memory_cleanup();
        device.sync().unwrap();
        let before = usage();
        let peak = Arc::new(AtomicU64::new(before.bytes_in_use));
        let done = Arc::new(AtomicBool::new(false));
        let sampler = {
            let (usage, peak, done) = (usage.clone(), peak.clone(), done.clone());
            std::thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    peak.fetch_max(usage().bytes_in_use, Ordering::Relaxed);
                    std::thread::yield_now();
                }
            })
        };
        for _ in 0..3 {
            let kept = train_step();
            device.sync().unwrap();
            drop(kept);
        }
        done.store(true, Ordering::Relaxed);
        sampler.join().unwrap();
        let after = usage();
        let rows = n.div_ceil(n.div_ceil(options.row_chunk));
        let chunk = options.vocab_chunk_for(rows);
        // f32 logits and two bf16 halves per chunk; f32 `o` and state per block; the f32
        // gradient and the two outputs over all rows.
        let counted = (rows * chunk * (4 + 2 + 2) + rows * (D + 8) * 4 + n * (D + 2) * 4) as u64;
        println!(
            "n {n:5}  inputs {:5.0} MiB in use, {:5.0} MiB reserved; reservation grew {:4.0} MiB; \
             peak in use above the inputs: sampled +{:4.0} MiB, counted +{:4.0} MiB",
            mib(before.bytes_in_use),
            mib(before.bytes_reserved),
            mib(after.bytes_reserved) - mib(before.bytes_reserved),
            mib(peak.load(Ordering::Relaxed)) - mib(before.bytes_in_use),
            mib(counted),
        );

        let time = |f: &dyn Fn()| {
            for _ in 0..3 {
                f();
                device.sync().unwrap();
            }
            let mut times: Vec<f64> = (0..reps)
                .map(|_| {
                    device.sync().unwrap();
                    let start = Instant::now();
                    f();
                    device.sync().unwrap();
                    start.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (times[times.len() / 2], times[0])
        };
        let (train_median, train_min) = time(&|| {
            let _ = train_step();
        });
        let (fwd_median, fwd_min) = time(&|| {
            let _ = forward();
        });
        println!(
            "n {n:5}  fwd+bwd median {train_median:7.1} ms (min {train_min:7.1})  \
             fwd median {fwd_median:7.1} ms (min {fwd_min:7.1})  vocab chunk {}",
            options.vocab_chunk_for(n.div_ceil(n.div_ceil(options.row_chunk)))
        );
    }
}
