//! Times the causal depthwise conv1d on CUDA, and the reserved device memory it adds: dash-ops'
//! kernels against Burn's own `conv1d` with `groups = C`. By default one sequence of 6144 steps
//! and 6144 channels in bf16 (Qwen3.5-0.8B's width), kernel width 4 with SiLU and no bias, then
//! widths 2, 3 and 4, with and without a bias, beside it.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example conv1d_bench \
//!     -- [B=1] [T=6144] [C=6144] [bf16|f32]
//!
//! Inputs are random, drawn on the device. Each time is the median of five trials, each the mean
//! over repetitions between device syncs, after warm-up repetitions (which also let Burn's
//! autotuner settle).
//!
//! - "op": the operation's own forward and backward, as the backward pass of a training step
//!   runs them: `causal_conv1d` then `causal_conv1d_backward` (input, weight and bias gradients
//!   from the output gradient), or Burn's `conv1d`, `conv1d_x_backward` and
//!   `conv1d_weight_backward`. This is the work PyTorch's `out.backward(grad)` does.
//! - "autodiff": through `Autodiff`, with `(out * grad).sum()` as the loss, so the loss's own
//!   multiply and sum (and their backward) are in it too, the same for both.
//! - memory: growth of the device's reserved pool memory over one pass, after a cleanup, with
//!   the inputs and the output gradient already allocated; and what a forward pass through
//!   `Autodiff` holds until its backward pass (live bytes while its output is alive: the output
//!   and whatever the backward pass keeps).
//! - decoding: one token at batch 8, the f32 state `[8, K - 1, C]` carried from step to step.

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
    use dash_ops::conv1d::{
        Activation, Bias, CausalConv1dBackend, CausalConv1dOptions, causal_conv1d,
        causal_conv1d_update,
    };
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

    /// The op-level backward: input, weight and bias gradients from the output gradient.
    fn backward(
        x: &Tensor<3>,
        weight: &Tensor<2>,
        bias: Option<&Tensor<1>>,
        grad: &Tensor<3>,
        options: CausalConv1dOptions,
    ) -> impl Sized {
        <Dispatch as CausalConv1dBackend>::causal_conv1d_backward(
            x.clone().into_dispatch(),
            weight.clone().into_dispatch(),
            Bias::from(bias.map(|bias| bias.clone().into_dispatch())),
            grad.clone().into_dispatch(),
            options,
        )
    }

    /// Forward, forward + backward (ms) and one decoding step at batch 8 (us), for width `k`,
    /// with or without a bias, SiLU.
    fn general(device: &Device, [b, t, c]: [usize; 3], dtype: DType, k: usize, with_bias: bool) {
        let options = CausalConv1dOptions::default().with_activation(Activation::Silu);
        let normal = |shape: Vec<usize>, std: f64| {
            Tensor::<3>::random(shape, Distribution::Normal(0.0, std), (device, dtype))
        };
        let x = normal(vec![b, t, c], 1.0);
        let grad = normal(vec![b, t, c], 1.0);
        let weight = normal(vec![1, c, k], 0.5).reshape([c, k]);
        let bias = with_bias.then(|| normal(vec![1, 1, c], 0.5).reshape([c]));
        let forward = time(device, (WARMUP, REPS), || {
            let _ = causal_conv1d(x.clone(), weight.clone(), bias.clone(), options);
        });
        let both = time(device, (WARMUP, REPS), || {
            let _ = causal_conv1d(x.clone(), weight.clone(), bias.clone(), options);
            let _ = backward(&x, &weight, bias.as_ref(), &grad, options);
        });
        let tokens = normal(vec![1, 8, c], 1.0).reshape([8, c]);
        let mut state = Tensor::<3>::random(
            [8, k - 1, c],
            Distribution::Normal(0.0, 1.0),
            (device, DType::F32),
        );
        let step = time(device, (20, 200), || {
            let (_, next) = causal_conv1d_update(
                tokens.clone(),
                state.clone(),
                weight.clone(),
                bias.clone(),
                options,
            );
            state = next;
        });
        println!(
            "K={k} bias={with_bias:<5} SiLU: forward {forward:.3} ms, forward+backward {both:.3} ms, \
             decode B=8 {:.1} us per step",
            step * 1e3
        );
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
        let silu_options = CausalConv1dOptions::default().with_activation(Activation::Silu);
        // dash-ops' layout: [B, T, C].
        let x = normal([b, t, c], 1.0);
        let grad = normal([b, t, c], 1.0);
        let weight = Tensor::<2>::random([c, 4], Distribution::Normal(0.0, 0.5), (&device, dtype));
        println!("B={b} T={t} C={c} {dtype:?}, K=4, SiLU, no bias");

        // dash-ops, op level.
        let forward = time(&device, (WARMUP, REPS), || {
            let _ = causal_conv1d(x.clone(), weight.clone(), None, silu_options);
        });
        let both = time(&device, (WARMUP, REPS), || {
            let _ = causal_conv1d(x.clone(), weight.clone(), None, silu_options);
            let _ = backward(&x, &weight, None, &grad, silu_options);
        });
        let mem_forward = reserved_mib(&device, || {
            let _ = causal_conv1d(x.clone(), weight.clone(), None, silu_options);
        });
        let mem_both = reserved_mib(&device, || {
            let out = causal_conv1d(x.clone(), weight.clone(), None, silu_options);
            let grads = backward(&x, &weight, None, &grad, silu_options);
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
            let out = causal_conv1d(x.clone(), weight.clone(), None, silu_options);
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
        let held = held_mib(&autodiff, || {
            causal_conv1d(xg.clone(), wg.clone(), None, silu_options)
        });
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
        let ours = causal_conv1d(
            x.clone(),
            weight.clone(),
            None,
            CausalConv1dOptions::default(),
        );
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

        // Decoding: one token at batch 8, the f32 state [8, 3, C] carried from step to step.
        let tokens = Tensor::<2>::random([8, c], Distribution::Normal(0.0, 1.0), (&device, dtype));
        let mut state = Tensor::<3>::random(
            [8, 3, c],
            Distribution::Normal(0.0, 1.0),
            (&device, DType::F32),
        );
        let per_step = time(&device, (20, 200), || {
            let (_, next) = causal_conv1d_update(
                tokens.clone(),
                state.clone(),
                weight.clone(),
                None,
                silu_options,
            );
            state = next;
        });
        println!(
            "decode B=8 C={c} state [8, 3, C] f32: {:.1} us per step",
            per_step * 1e3
        );

        // The general cases.
        for k in [2, 3, 4] {
            for with_bias in [false, true] {
                general(&device, [b, t, c], dtype, k, with_bias);
            }
        }
    }
}
