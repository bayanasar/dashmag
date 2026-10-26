//! Time and memory of the chunked gated delta rule on CUDA, at Qwen3.5-0.8B's Gated DeltaNet
//! shapes by default: one sequence of 6144 tokens, 16 heads, Dk = Dv = 128, bf16 inputs, queries
//! and keys normalised in the kernel.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda \
//!     --example gated_delta_chunk_bench -- [T] [H] [D] [bf16|f32] [B]
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
    use burn::tensor::{DType, Device, Distribution, Tensor};
    use dash_ops::gated_delta::chunk::chunk_gated_delta_rule;
    use std::time::Instant;

    struct Inputs {
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        g: Tensor<3>,
        beta: Tensor<3>,
        d_out: Tensor<4>,
    }

    fn inputs(device: &Device, [b, t, h, d]: [usize; 4], dtype: DType, grad: bool) -> Inputs {
        let normal = Distribution::Normal(0.0, 1.0);
        let leaf4 = |shape: [usize; 4]| {
            let x = Tensor::<4>::random(shape, normal, device).cast(dtype);
            if grad { x.require_grad() } else { x }
        };
        let (q, k, v) = (
            leaf4([b, t, h, d]),
            leaf4([b, t, h, d]),
            leaf4([b, t, h, d]),
        );
        // g = -softplus(x) / 2 and beta = sigmoid(x), as the model forms them.
        let x = Tensor::<3>::random([b, t, h], normal, device);
        let g = (x.exp() + 1.0).log() * (-0.5);
        let beta =
            burn::tensor::activation::sigmoid(Tensor::<3>::random([b, t, h], normal, device));
        let (g, beta) = if grad {
            (g.require_grad(), beta.cast(dtype).require_grad())
        } else {
            (g, beta.cast(dtype))
        };
        let d_out = Tensor::<4>::random([b, t, h, d], normal, device);
        Inputs {
            q,
            k,
            v,
            g,
            beta,
            d_out,
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
            false,
            true,
        );
        out
    }

    fn forward_backward(x: &Inputs) {
        let out = forward(x);
        let grads = (out.cast(DType::F32) * x.d_out.clone()).sum().backward();
        // Drop the gradients at once, as an optimizer step would consume them.
        drop(x.q.grad(&grads));
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

    pub(super) fn main() {
        let args: Vec<String> = std::env::args().collect();
        let arg = |i: usize, default: usize| args.get(i).map_or(default, |s| s.parse().unwrap());
        let (t, h, d) = (arg(1, 6144), arg(2, 16), arg(3, 128));
        let dtype = match args.get(4).map(String::as_str) {
            None | Some("bf16") => DType::BF16,
            Some("f32") => DType::F32,
            Some(other) => panic!("dtype {other}: bf16 or f32"),
        };
        let b = arg(5, 1);
        let device = Device::cuda(0);
        println!("B {b} T {t} H {h} Dk = Dv {d} {dtype:?}, l2norm in kernel");

        let plain = inputs(&device, [b, t, h, d], dtype, false);
        let fwd = time(&device, 3, 20, || drop(forward(&plain)));
        println!("forward            {fwd:8.3} ms");
        drop(plain);

        let ad = device.clone().autodiff();
        let x = inputs(&ad, [b, t, h, d], dtype, true);
        let both = time(&ad, 3, 20, || forward_backward(&x));
        println!("forward + backward {both:8.3} ms");

        device.sync().unwrap();
        device.memory_cleanup();
        let before = device
            .memory_pool_usage()
            .expect("the CUDA pool reports its usage");
        forward_backward(&x);
        device.sync().unwrap();
        let after = device.memory_pool_usage().unwrap();
        println!(
            "pool reserved     {:8.1} MiB before, {:8.1} MiB after one forward + backward: +{:.1} MiB",
            mib(before.bytes_reserved),
            mib(after.bytes_reserved),
            mib(after.bytes_reserved.saturating_sub(before.bytes_reserved))
        );
    }
}
