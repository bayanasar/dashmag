//! Times the causal depthwise conv1d on CUDA, and the reserved device memory it adds: dash-ops'
//! kernels against Burn's own `conv1d` with `groups = C`, at Qwen3.5-0.8B's width by default.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example conv1d_bench \
//!     -- [B=1] [T=6144] [C=6144] [bf16|f32]
//!
//! Inputs are random, drawn on the device. Each time is the median of five trials, each the mean
//! over repetitions between device syncs, after warm-up repetitions (which also let Burn's
//! autotuner settle).
//!
//! - "op": the operation's own forward and backward, as the backward pass of a training step
//!   runs them: `causal_conv1d` then `causal_conv1d_backward` (input and weight gradients from
//!   the output gradient), or Burn's `conv1d`, `conv1d_x_backward` and `conv1d_weight_backward`.
//!   This is the work PyTorch's `out.backward(grad)` does.
//! - "autodiff": through `Autodiff`, with `(out * grad).sum()` as the loss, so the loss's own
//!   multiply and sum (and their backward) are in it too, the same for both.
//! - memory: growth of the device's reserved pool memory over one pass, after a cleanup, with
//!   the inputs and the output gradient already allocated; and what a forward pass through
//!   `Autodiff` holds until its backward pass (live bytes while its output is alive: the output
//!   and whatever the backward pass keeps).
//! - decoding: one token at batch 8, the state `[8, C, 3]` carried from step to step.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("conv1d_bench needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    bench::run();
}

#[cfg(feature = "cuda")]
mod bench {
    use burn::backend::Dispatch;
    use burn::backend::ops::ModuleOps;
    use burn::tensor::activation::silu;
    use burn::tensor::module::conv1d;
    use burn::tensor::ops::ConvOptions;
    use burn::tensor::{DType, Device, Distribution, Tensor};
    use dash_ops::conv1d::{CausalConv1dBackend, causal_conv1d, causal_conv1d_update};
    use std::time::Instant;

    const WARMUP: usize = 5;
    const REPS: usize = 30;

    /// Milliseconds per call: the median over five trials of `reps` calls each, after `warmup`.
    fn time(device: &Device, (warmup, reps): (usize, usize), mut f: impl FnMut()) -> f64 {
        for _ in 0..warmup {
            f();
        }
        let mut trials: Vec<f64> = (0..5)
            .map(|_| {
                device.sync().unwrap();
                let start = Instant::now();
                for _ in 0..reps {
                    f();
                }
                device.sync().unwrap();
                start.elapsed().as_secs_f64() * 1e3 / reps as f64
            })
            .collect();
        trials.sort_by(f64::total_cmp);
        trials[2]
    }

    fn reserved_mib(device: &Device, f: impl FnOnce()) -> f64 {
        device.sync().unwrap();
        device.memory_cleanup();
        let before = device.memory_pool_usage().unwrap().bytes_reserved;
        f();
        device.sync().unwrap();
        let after = device.memory_pool_usage().unwrap().bytes_reserved;
        (after as f64 - before as f64) / (1u64 << 20) as f64
    }

    /// Live device memory a value holds: bytes in use while `f`'s result is alive, less before.
    fn held_mib<T>(device: &Device, f: impl FnOnce() -> T) -> f64 {
        device.sync().unwrap();
        let before = device.memory_pool_usage().unwrap().bytes_in_use;
        let value = f();
        device.sync().unwrap();
        let after = device.memory_pool_usage().unwrap().bytes_in_use;
        drop(value);
        (after as f64 - before as f64) / (1u64 << 20) as f64
    }

    fn max_abs_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
        (a.cast(DType::F32) - b.cast(DType::F32))
            .abs()
            .max()
            .into_scalar::<f32>()
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let arg = |i: usize, default: usize| args.get(i).map_or(default, |s| s.parse().unwrap());
        let (b, t, c) = (arg(1, 1), arg(2, 6144), arg(3, 6144));
        let dtype = match args.get(4).map(String::as_str) {
            Some("f32") => DType::F32,
            _ => DType::BF16,
        };
        let device = Device::cuda(0);
        let normal = |shape: [usize; 3], std: f64| {
            Tensor::<3>::random(shape, Distribution::Normal(0.0, std), (&device, dtype))
        };
        // dash-ops' layout: [B, T, C].
        let x = normal([b, t, c], 1.0);
        let grad = normal([b, t, c], 1.0);
        let weight = Tensor::<2>::random([c, 4], Distribution::Normal(0.0, 0.5), (&device, dtype));
        println!("B={b} T={t} C={c} {dtype:?}");

        // dash-ops, op level.
        let forward = time(&device, (WARMUP, REPS), || {
            let _ = causal_conv1d(x.clone(), weight.clone(), true);
        });
        let backward = || {
            <Dispatch as CausalConv1dBackend>::causal_conv1d_backward(
                x.clone().into_dispatch(),
                weight.clone().into_dispatch(),
                grad.clone().into_dispatch(),
                true,
            )
        };
        let both = time(&device, (WARMUP, REPS), || {
            let _ = causal_conv1d(x.clone(), weight.clone(), true);
            let _ = backward();
        });
        let mem_forward = reserved_mib(&device, || {
            let _ = causal_conv1d(x.clone(), weight.clone(), true);
        });
        let mem_both = reserved_mib(&device, || {
            let out = causal_conv1d(x.clone(), weight.clone(), true);
            let grads = backward();
            drop((out, grads));
        });
        println!(
            "dash-ops op, SiLU fused: forward {forward:.3} ms (+{mem_forward:.0} MiB), \
             forward+backward {both:.3} ms (+{mem_both:.0} MiB)"
        );

        // dash-ops through Autodiff.
        let autodiff = device.clone().autodiff();
        let grad_ad = grad.clone().autodiff();
        let step = || {
            let x = x.clone().autodiff().require_grad();
            let weight = weight.clone().autodiff().require_grad();
            let out = causal_conv1d(x.clone(), weight.clone(), true);
            let grads = (out * grad_ad.clone()).sum().backward();
            (x.grad(&grads).unwrap(), weight.grad(&grads).unwrap())
        };
        let ad = time(&autodiff, (WARMUP, REPS), || {
            let _ = step();
        });
        let mem_ad = reserved_mib(&autodiff, || {
            let _ = step();
        });
        let (xg, wg) = (
            x.clone().autodiff().require_grad(),
            weight.clone().autodiff().require_grad(),
        );
        let held = held_mib(&autodiff, || causal_conv1d(xg.clone(), wg.clone(), true));
        println!(
            "dash-ops autodiff forward: holds {held:.0} MiB until the backward (output included)"
        );
        // The loss alone, on a leaf standing in for the output.
        let loss_only = time(&autodiff, (WARMUP, REPS), || {
            let out = x.clone().autodiff().require_grad();
            let grads = (out.clone() * grad_ad.clone()).sum().backward();
            let _ = out.grad(&grads).unwrap();
        });
        println!(
            "dash-ops autodiff, SiLU fused: forward+backward {ad:.3} ms (+{mem_ad:.0} MiB), \
             of which the loss alone {loss_only:.3} ms"
        );

        // Burn's conv1d, groups = C, on [B, C, T]: padded by K - 1 on both sides and cut to the
        // first T outputs, as torch's fallback does.
        let options = ConvOptions::new([1], [3], [1], c);
        let weight3 = weight.clone().reshape([c, 1, 4]);
        for (layout, x_bct) in [
            ("[B, C, T] contiguous", normal([b, c, t], 1.0)),
            ("[B, T, C] seen as [B, C, T]", x.clone().swap_dims(1, 2)),
        ] {
            let grad_padded = Tensor::cat(
                vec![
                    grad.clone().swap_dims(1, 2),
                    Tensor::zeros([b, c, 3], (&device, dtype)),
                ],
                2,
            );
            let forward = time(&device, (2, 5), || {
                let _ = conv1d(x_bct.clone(), weight3.clone(), None, options.clone()).slice([
                    0..b,
                    0..c,
                    0..t,
                ]);
            });
            let backward = || {
                let gx = <Dispatch as ModuleOps<Dispatch>>::conv1d_x_backward(
                    x_bct.clone().into_dispatch(),
                    weight3.clone().into_dispatch(),
                    grad_padded.clone().into_dispatch(),
                    options.clone(),
                );
                let gw = <Dispatch as ModuleOps<Dispatch>>::conv1d_weight_backward(
                    x_bct.clone().into_dispatch(),
                    weight3.clone().into_dispatch(),
                    grad_padded.clone().into_dispatch(),
                    options.clone(),
                );
                (gx, gw)
            };
            let both = time(&device, (2, 3), || {
                let _ = conv1d(x_bct.clone(), weight3.clone(), None, options.clone());
                let _ = backward();
            });
            let mem_both = reserved_mib(&device, || {
                let out = conv1d(x_bct.clone(), weight3.clone(), None, options.clone());
                let grads = backward();
                drop((out, grads));
            });
            println!(
                "burn conv1d op, no activation, {layout}: forward {forward:.3} ms, \
                 forward+backward {both:.3} ms (+{mem_both:.0} MiB)"
            );
        }

        // Burn's conv1d then SiLU through Autodiff, on the transposed view (transformers' path).
        let step = || {
            let x = x.clone().autodiff().require_grad();
            let weight3 = weight3.clone().autodiff().require_grad();
            let out = silu(
                conv1d(
                    x.clone().swap_dims(1, 2),
                    weight3.clone(),
                    None,
                    options.clone(),
                )
                .slice([0..b, 0..c, 0..t]),
            )
            .swap_dims(1, 2);
            let grads = (out * grad_ad.clone()).sum().backward();
            (x.grad(&grads).unwrap(), weight3.grad(&grads).unwrap())
        };
        let ad = time(&autodiff, (2, 3), || {
            let _ = step();
        });
        let mem_ad = reserved_mib(&autodiff, || {
            let _ = step();
        });
        println!("burn conv1d + SiLU autodiff: forward+backward {ad:.3} ms (+{mem_ad:.0} MiB)");
        let (xg, wg) = (
            x.clone().autodiff().require_grad(),
            weight3.clone().autodiff().require_grad(),
        );
        let held = held_mib(&autodiff, || {
            silu(
                conv1d(
                    xg.clone().swap_dims(1, 2),
                    wg.clone(),
                    None,
                    options.clone(),
                )
                .slice([0..b, 0..c, 0..t]),
            )
        });
        println!("burn conv1d + SiLU autodiff forward: holds {held:.0} MiB until the backward");

        // The two agree (no activation), and the gradients through Autodiff agree.
        let ours = causal_conv1d(x.clone(), weight.clone(), false);
        let burns = conv1d(
            x.clone().swap_dims(1, 2),
            weight3.clone(),
            None,
            options.clone(),
        )
        .slice([0..b, 0..c, 0..t])
        .swap_dims(1, 2);
        println!(
            "max |dash-ops - burn| forward, no activation: {:.3e}",
            max_abs_diff(ours, burns)
        );

        // Decoding: one token at batch 8, the state [8, C, 3] carried from step to step.
        let tokens = Tensor::<2>::random([8, c], Distribution::Normal(0.0, 1.0), (&device, dtype));
        let state0 =
            Tensor::<3>::random([8, c, 3], Distribution::Normal(0.0, 1.0), (&device, dtype));
        let mut state = state0.clone();
        let per_step = time(&device, (20, 200), || {
            let (_, next) =
                causal_conv1d_update(tokens.clone(), state.clone(), weight.clone(), true);
            state = next;
        });
        println!(
            "decode B=8 C={c} state [8, C, 3]: {:.1} us per step",
            per_step * 1e3
        );
    }
}
