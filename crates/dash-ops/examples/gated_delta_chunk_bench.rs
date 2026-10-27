//! Time and memory of the chunked gated delta rule on CUDA: `T` tokens as one sequence, as `S`
//! packed sequences of `T / S` tokens (`cu_seqlens`), and as `S` calls of one sequence each; `H`
//! heads with `Dk = Dv = D`, bf16 inputs, queries and keys normalised in the kernel. By default
//! `T = 6144`, `S = 4`, `H = 16`, `D = 128` (Qwen3.5-0.8B's Gated DeltaNet shapes).
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda \
//!     --example gated_delta_chunk_bench -- [T] [H] [D] [bf16|f32] [S] [one|packed|calls|all] \
//!     [reps] [forward|both|all]
//!
//! The sixth argument runs one of the three layouts alone, the eighth one of the two passes (for
//! a profiler, `nsys profile`, whose kernel times over the `3 + reps` passes give the device's
//! time); the seventh sets the timed repetitions (20). The host-side timings include Burn's
//! dispatch of every operation, so they rise with the load on the host's CPU; the kernel times do
//! not.
//!
//! Forward alone runs without autodiff; forward + backward runs through Autodiff with the
//! gradient of the output given (`sum(o * do)`), as a training step does. Each timing is a mean
//! over repetitions after warm-up, with the device synchronised after every repetition. Memory is
//! the growth of the CUDA memory pool over one forward + backward, from a pool emptied of unused
//! pages, with the inputs and `do` already allocated.
//!
//! Without the `cuda` feature it only says so.

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("gated_delta_chunk_bench times the CUDA kernels: run it with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    bench::main();
}

#[cfg(feature = "cuda")]
mod bench {
    use burn::tensor::{DType, Device, Distribution, Int, Tensor, TensorData};
    use dash_ops::gated_delta::chunk::{ChunkGatedDeltaOptions, chunk_gated_delta_rule};
    use std::time::Instant;

    /// One call's inputs: `[1, T, H, D]` tokens, and the packed boundaries if any.
    struct Inputs {
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        g: Tensor<3>,
        beta: Tensor<3>,
        d_out: Tensor<4>,
        cu_seqlens: Option<Tensor<1, Int>>,
    }

    fn inputs(
        device: &Device,
        [t, h, d]: [usize; 3],
        dtype: DType,
        grad: bool,
        sequences: Option<usize>,
    ) -> Inputs {
        let normal = Distribution::Normal(0.0, 1.0);
        let leaf4 = |shape: [usize; 4]| {
            let x = Tensor::<4>::random(shape, normal, device).cast(dtype);
            if grad { x.require_grad() } else { x }
        };
        let (q, k, v) = (
            leaf4([1, t, h, d]),
            leaf4([1, t, h, d]),
            leaf4([1, t, h, d]),
        );
        // g = -softplus(x) / 2 and beta = sigmoid(x), as the model forms them.
        let x = Tensor::<3>::random([1, t, h], normal, device);
        let g = (x.exp() + 1.0).log() * (-0.5);
        let beta =
            burn::tensor::activation::sigmoid(Tensor::<3>::random([1, t, h], normal, device));
        let (g, beta) = if grad {
            (g.require_grad(), beta.cast(dtype).require_grad())
        } else {
            (g, beta.cast(dtype))
        };
        let d_out = Tensor::<4>::random([1, t, h, d], normal, device);
        let cu_seqlens = sequences.map(|s| {
            let offsets: Vec<i64> = (0..=s).map(|i| (i * t / s) as i64).collect();
            Tensor::from_data(TensorData::new(offsets, [s + 1]), device)
        });
        Inputs {
            q,
            k,
            v,
            g,
            beta,
            d_out,
            cu_seqlens,
        }
    }

    fn forward(x: &Inputs) -> Tensor<4> {
        let (out, _) = chunk_gated_delta_rule(
            x.q.clone(),
            x.k.clone(),
            x.v.clone(),
            x.g.clone(),
            x.beta.clone(),
            None,
            x.cu_seqlens.clone(),
            ChunkGatedDeltaOptions::default().with_qk_l2norm(true),
        );
        out
    }

    /// Forward and backward over every call of `calls`, one loss over all their outputs.
    fn forward_backward(calls: &[Inputs]) {
        let loss = calls
            .iter()
            .map(|x| (forward(x).cast(DType::F32) * x.d_out.clone()).sum())
            .reduce(|a, b| a + b)
            .unwrap();
        let grads = loss.backward();
        // Drop the gradients at once, as an optimizer step would consume them.
        drop(calls[0].q.grad(&grads));
    }

    fn time(device: &Device, warmup: usize, reps: usize, mut f: impl FnMut()) -> f64 {
        for _ in 0..warmup {
            f();
            device.sync().unwrap();
        }
        let start = Instant::now();
        for _ in 0..reps {
            f();
            device.sync().unwrap();
        }
        start.elapsed().as_secs_f64() * 1e3 / reps as f64
    }

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    /// The calls of one layout: the tokens as one sequence, packed, or one call per sequence.
    fn calls(
        device: &Device,
        layout: &str,
        [t, h, d, s]: [usize; 4],
        dtype: DType,
        grad: bool,
    ) -> Vec<Inputs> {
        match layout {
            "one sequence" => vec![inputs(device, [t, h, d], dtype, grad, None)],
            "packed" => vec![inputs(device, [t, h, d], dtype, grad, Some(s))],
            _ => (0..s)
                .map(|_| inputs(device, [t / s, h, d], dtype, grad, None))
                .collect(),
        }
    }

    pub(super) fn main() {
        let args: Vec<String> = std::env::args().collect();
        let arg = |i: usize, default: usize| args.get(i).map_or(default, |s| s.parse().unwrap());
        let (t, h, d) = (arg(1, 6144), arg(2, 16), arg(3, 128));
        let dtype = match args.get(4).map(String::as_str) {
            None | Some("bf16") => DType::BF16,
            Some("f32") => DType::F32,
            Some(other) => panic!("dtype {other}: bf16 or f32"),
        };
        let s = arg(5, 4);
        let only = args.get(6).map(String::as_str).filter(|x| *x != "all");
        let reps = arg(7, 20);
        let pass = args.get(8).map(String::as_str).unwrap_or("all");
        let device = Device::cuda(0);
        println!(
            "T {t} H {h} Dk = Dv {d} {dtype:?}, l2norm in kernel; {s} sequences of {} tokens",
            t / s
        );
        let shape = [t, h, d, s];
        for (layout, name) in [
            ("one sequence", "one"),
            ("packed", "packed"),
            ("one call per sequence", "calls"),
        ] {
            if only.is_some_and(|only| only != name) {
                continue;
            }
            let plain = calls(&device, layout, shape, dtype, false);
            let fwd = if pass == "both" {
                f64::NAN
            } else {
                time(&device, 3, reps, || {
                    for x in &plain {
                        drop(forward(x));
                    }
                })
            };
            drop(plain);
            if pass == "forward" {
                println!("{layout:22} forward {fwd:7.3} ms");
                continue;
            }

            let ad = device.clone().autodiff();
            let x = calls(&ad, layout, shape, dtype, true);
            let both = time(&ad, 3, reps, || forward_backward(&x));

            device.sync().unwrap();
            device.memory_cleanup();
            let before = device
                .memory_pool_usage()
                .expect("the CUDA pool reports its usage");
            forward_backward(&x);
            device.sync().unwrap();
            let after = device.memory_pool_usage().unwrap();
            println!(
                "{layout:22} forward {fwd:7.3} ms, forward + backward {both:7.3} ms, pool +{:.1} MiB",
                mib(after.bytes_reserved.saturating_sub(before.bytes_reserved))
            );
        }
    }
}
