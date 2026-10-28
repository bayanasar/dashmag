//! Time and peak memory of the fused linear cross-entropy at a 248320-token vocabulary and a
//! hidden size of 1024 (Qwen3.5-0.8B's output head), bf16 inputs, on the first CUDA device.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example loss_bench \
//!       -- [n ...] [--vocab-chunk C] [--row-chunk R] [--reps K] [--options]
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example loss_bench \
//!       -- --matmul [m c ...]
//!
//! For each n (default 3072 and 8192), three steps, each first for memory, then for time:
//!
//! - `train`: forward + backward of the mean loss with `hidden` and `weight` both requiring a
//!   gradient (the second pass over the vocabulary, the weight's `[V, D]` bf16 gradient);
//! - `frozen`: the same with a frozen weight (the forward pass accumulates the gradient of
//!   `hidden`, no second pass);
//! - `forward`: the forward pass alone, without autodiff.
//!
//! Memory: the most bytes the allocator has in use above what it held before (the inputs and the
//! weight) over three passes after a first one (with autotune, the first tunes the matmuls),
//! sampled from a second thread (a lower bound: a pass queues its chunks within milliseconds of
//! host time, and the sampler sees only some moments of them), and how far the allocator's
//! reservation grew over the four. Time: the median of K timed repetitions
//! (default 10) after 3 warm-up ones, with a device sync around each. The options are torch's
//! defaults (the mean, no smoothing, no masked id); `--options` turns on label smoothing 0.1, a
//! z-loss of 1e-4, a masked id and the entropy.
//!
//! `--matmul` times the matmuls of one chunk alone at several row counts `m` and chunk widths
//! `c`, on the CubeCL backend directly (so under fusion too): the logits `[m, D] x [D, c]`, the
//! gradient of `hidden` `[m, c] x [c, D]`, and the weight's `[c, m] x [m, D]` with the left
//! operand a transposed view, as the backward pass runs them.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("loss_bench measures the CUDA path: build it with --features cuda");
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
    use dash_ops::logprob::{LogprobOptions, VocabChunk};
    use dash_ops::loss::{CrossEntropyOptions, linear_cross_entropy};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Instant;

    const V: usize = 248_320;
    const D: usize = 1024;
    /// A token near the end of the vocabulary, taken out of the softmax with `--options`.
    const MASKED: u32 = 248_044;

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "--matmul") {
            let sizes: Vec<usize> = args.iter().filter_map(|a| a.parse().ok()).collect();
            matmul_sweep(&sizes);
            return;
        }
        let mut ns = Vec::new();
        let mut chunks = LogprobOptions::default();
        let mut options = CrossEntropyOptions::default();
        let mut reps = 10;
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .expect("a value")
                    .parse::<usize>()
                    .expect("a number")
            };
            match arg.as_str() {
                "--vocab-chunk" => chunks.vocab_chunk = value(),
                "--row-chunk" => chunks.row_chunk = value(),
                "--reps" => reps = value(),
                "--options" => {
                    options.label_smoothing = 0.1;
                    options.z_loss = 1e-4;
                    options.masked = vec![MASKED];
                    options.entropy = true;
                }
                n => ns.push(n.parse().expect("n")),
            }
        }
        if ns.is_empty() {
            ns = vec![3072, 8192];
        }
        options.chunks = chunks;
        println!("V {V}, D {D}, bf16 inputs, {options:?}");
        for n in ns {
            run(n, &options, reps);
        }
    }

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / (1u64 << 20) as f64
    }

    /// The most bytes in use above what the allocator held before, over three runs of `step`
    /// after an unsampled one (the first, which with autotune tunes the matmuls with buffers of
    /// its own), and how far the allocator's reservation grew over all four.
    fn memory(device: &Device, step: &dyn Fn()) -> (f64, f64, f64) {
        // The CubeCL backend's allocator itself: under fusion the operation runs on the fusion
        // server's thread, and asking the dispatch device would wait for the whole of it.
        let cube_device = CubeDevice::cuda(0).expect("a CUDA device");
        let usage = move || {
            <CubeBackend as Backend>::memory_pool_usage(&cube_device)
                .expect("the allocator reports usage")
        };
        device.memory_cleanup();
        device.sync().unwrap();
        let reserved = usage().bytes_reserved;
        step();
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
            step();
            device.sync().unwrap();
        }
        done.store(true, Ordering::Relaxed);
        sampler.join().unwrap();
        let after = usage();
        (
            mib(before.bytes_in_use),
            mib(peak.load(Ordering::Relaxed)) - mib(before.bytes_in_use),
            mib(after.bytes_reserved) - mib(reserved),
        )
    }

    /// The median and the fastest of `reps` timed runs after three warm-up ones, in ms.
    fn time(device: &Device, reps: usize, step: &dyn Fn()) -> (f64, f64) {
        for _ in 0..3 {
            step();
            device.sync().unwrap();
        }
        let mut times: Vec<f64> = (0..reps)
            .map(|_| {
                device.sync().unwrap();
                let start = Instant::now();
                step();
                device.sync().unwrap();
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (times[times.len() / 2], times[0])
    }

    fn run(n: usize, options: &CrossEntropyOptions, reps: usize) {
        let device = Device::cuda(0);
        device.seed(n as u64);
        // Made in bf16 directly: no f32 temporary to leave the allocator a larger peak than the
        // operation's own.
        let bf16 = (&device, DType::BF16);
        let weight = Tensor::<2>::random([V, D], Distribution::Normal(0.0, 0.02), bf16);
        let hidden = Tensor::<2>::random([n, D], Distribution::Normal(0.0, 1.0), bf16);
        let targets = Tensor::<1, Int>::random([n], Distribution::Uniform(0.0, V as f64), &device);
        device.sync().unwrap();

        let train = || {
            let h = hidden.clone().autodiff().require_grad();
            let w = weight.clone().autodiff().require_grad();
            let out = linear_cross_entropy(h.clone(), w.clone(), targets.clone(), options);
            let grads = out.loss.backward();
            let kept = (h.grad(&grads), w.grad(&grads));
            assert!(kept.0.is_some() && kept.1.is_some(), "both gradients");
        };
        let frozen = || {
            let h = hidden.clone().autodiff().require_grad();
            let w = weight.clone().autodiff();
            let out = linear_cross_entropy(h.clone(), w, targets.clone(), options);
            let grads = out.loss.backward();
            assert!(h.grad(&grads).is_some(), "a gradient");
        };
        let forward = || {
            let out =
                linear_cross_entropy(hidden.clone(), weight.clone(), targets.clone(), options);
            drop(out);
        };
        let rows = n.div_ceil(n.div_ceil(options.chunks.row_chunk));
        println!(
            "n {n:5}  row blocks of {rows}, vocab chunk {}",
            options.chunks.vocab_chunk_for(rows)
        );
        for (label, step) in [
            ("train", &train as &dyn Fn()),
            ("frozen", &frozen),
            ("forward", &forward),
        ] {
            let (inputs, peak, grew) = memory(&device, step);
            let (median, min) = time(&device, reps, step);
            println!(
                "n {n:5}  {label:7} median {median:7.1} ms (min {min:7.1})  peak in use +{peak:5.0} MiB \
                 above the inputs' {inputs:4.0} MiB; reservation grew {grew:5.0} MiB"
            );
        }
    }

    /// The matmuls of one chunk, bf16 operands to an f32 result, at several row counts and chunk
    /// widths (pairs `m c` on the command line, or a default set).
    fn matmul_sweep(sizes: &[usize]) {
        use burn::backend::ops::FloatTensorOps;
        use burn::backend::{FloatDType, Shape};
        use burn_cubecl::kernel::matmul::{MatmulStrategy, matmul};
        let device = CubeDevice::cuda(0).expect("a CUDA device");
        let random = |shape: [usize; 2]| {
            CubeBackend::float_random(
                Shape::new(shape),
                Distribution::Normal(0.0, 1.0),
                &device,
                FloatDType::BF16,
            )
        };
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
        let shapes: Vec<(usize, usize)> = match sizes.len() >= 2 {
            true => sizes.chunks_exact(2).map(|p| (p[0], p[1])).collect(),
            false => vec![
                (2048, 4096),
                (3072, 2048),
                (3072, 4096),
                (3072, 8192),
                (4096, 4096),
                (8192, 2048),
                (8192, 4096),
                (8192, 8192),
            ],
        };
        for (m, c) in shapes {
            let a = random([m, D]);
            let w = random([c, D]);
            let g = random([m, c]);
            let tflops = |ms: f64| 2.0 * m as f64 * c as f64 * D as f64 / ms / 1e9;
            let gemm = |lhs, rhs| {
                let _ = matmul(lhs, rhs, None, MatmulStrategy::default(), DType::F32).unwrap();
            };
            let logits = time(&|| gemm(a.clone(), CubeBackend::float_transpose(w.clone())));
            let grad_hidden = time(&|| gemm(g.clone(), w.clone()));
            let grad_weight = time(&|| gemm(CubeBackend::float_transpose(g.clone()), a.clone()));
            println!(
                "m {m:5} c {c:5}: logits {logits:6.2} ms ({:5.1} TFLOPS)  grad hidden {grad_hidden:6.2} \
                 ms ({:5.1})  grad weight {grad_weight:6.2} ms ({:5.1})",
                tflops(logits),
                tflops(grad_hidden),
                tflops(grad_weight),
            );
        }
    }
}
