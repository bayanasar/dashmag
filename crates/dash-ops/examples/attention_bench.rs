//! Time and device memory of `dash_ops::attention` on a CUDA GPU, at Qwen3.5-0.8B's
//! full-attention shapes: 8 query heads, 2 key-value heads, head dimension 256, bf16.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example attention_bench
//!   ... --example attention_bench -- 8192        # another sequence length
//!   ... --example attention_bench -- 6144 burn   # also time Burn's own attention (needs ~5 GB)
//!
//! Training: one sequence of T tokens, causal; the forward (with its log-sum-exp), the backward
//! (`delta`, `dQ`, `dK`/`dV`), and both through `Autodiff` with `sum(o * dO)` as the loss.
//! Decoding: batch 8, one query each, against a 6144-token cache, per layer. Beside them, Burn's
//! own `attention`, which takes no grouped heads (the key-value heads are expanded for it), and,
//! for decoding, its flash kernel asked for by name. Each time is the median of 20 runs after 3
//! warm-ups, with the device synchronised around each run, so a time includes one launch and
//! synchronisation; decoding is also timed over 24 calls between synchronisations, as a step
//! over the layers runs. Memory is the growth of the bytes CubeCL's pools hold in use, read
//! after synchronising (under Burn's fusion nothing has run before), and the largest single
//! allocation the pools that hold tensors served.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("attention_bench needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    bench::run();
}

#[cfg(feature = "cuda")]
mod bench {
    use burn::backend::{DType, ops::AttentionModuleOptions};
    use burn::tensor::{Device, Distribution, Tensor, module};
    use dash_ops::attention::{
        attention_decode, causal_attention, causal_attention_backward, causal_attention_with_lse,
    };
    use std::time::Instant;

    fn median_ms(device: &Device, mut f: impl FnMut()) -> f64 {
        for _ in 0..3 {
            f();
        }
        device.sync().unwrap();
        let mut times: Vec<f64> = (0..20)
            .map(|_| {
                let start = Instant::now();
                f();
                device.sync().unwrap();
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        times.sort_by(f64::total_cmp);
        times[times.len() / 2]
    }

    /// Bytes in use in CubeCL's pools on device 0, and the largest allocation the pools that
    /// hold tensors (dynamic and persistent) ever served. The dedicated pool is left out: with
    /// Burn's `autotune` on, CubeCL's throughput probes reserve a gigabyte there, which no
    /// operation measured here allocates. Under Burn's fusion, synchronise the device first,
    /// or the operations have not run yet.
    fn memory() -> (u64, u64) {
        let client = cubecl::Device::cuda(0).expect("CUDA device 0").client();
        let report = client.memory_report(cubecl::MemoryScope::Device);
        let mut largest = 0;
        for stream in &report.streams {
            let pools = &stream.pools;
            for pool in pools.dynamic.iter().chain([&pools.persistent]) {
                largest = largest.max(pool.largest_alloc);
            }
        }
        (report.usage().bytes_in_use, largest)
    }

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / (1u64 << 20) as f64
    }

    fn random(shape: [usize; 4], device: &Device) -> Tensor<4> {
        Tensor::<4>::random(shape, Distribution::Normal(0.0, 1.0), device).cast(DType::BF16)
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let t: usize = args
            .get(1)
            .map_or(6144, |a| a.parse().expect("sequence length"));
        let burn_too = args.iter().any(|a| a == "burn");
        let device = Device::cuda(0);
        let (b, hq, hkv, d) = (1, 8, 2, 256);
        let scale = (d as f32).powf(-0.5);
        println!("training: B {b}, Hq {hq}, Hkv {hkv}, T {t}, D {d}, bf16, causal");

        let q = random([b, hq, t, d], &device);
        let k = random([b, hkv, t, d], &device);
        let v = random([b, hkv, t, d], &device);
        let grad = random([b, hq, t, d], &device);
        device.sync().unwrap();

        let forward = median_ms(&device, || {
            let _ = causal_attention(q.clone(), k.clone(), v.clone(), scale);
        });
        let (o, lse) = causal_attention_with_lse(q.clone(), k.clone(), v.clone(), scale);
        let backward = median_ms(&device, || {
            let _ = causal_attention_backward(
                q.clone(),
                k.clone(),
                v.clone(),
                o.clone(),
                lse.clone(),
                grad.clone(),
                scale,
            );
        });
        let both = median_ms(&device, || {
            let (o, lse) = causal_attention_with_lse(q.clone(), k.clone(), v.clone(), scale);
            let _ = causal_attention_backward(
                q.clone(),
                k.clone(),
                v.clone(),
                o,
                lse,
                grad.clone(),
                scale,
            );
        });
        drop((o, lse));
        println!(
            "  forward {forward:.2} ms, backward {backward:.2} ms, forward + backward {both:.2} ms"
        );

        // Memory of one forward and backward, everything they return kept alive.
        device.sync().unwrap();
        let (before, _) = memory();
        let (o, lse) = causal_attention_with_lse(q.clone(), k.clone(), v.clone(), scale);
        device.sync().unwrap();
        let (after_forward, _) = memory();
        let grads = causal_attention_backward(
            q.clone(),
            k.clone(),
            v.clone(),
            o.clone(),
            lse.clone(),
            grad.clone(),
            scale,
        );
        device.sync().unwrap();
        let (after_backward, largest) = memory();
        println!(
            "  memory over the inputs: +{:.1} MiB after the forward (output, log-sum-exp), \
             +{:.1} MiB after the backward (and dQ, dK, dV); largest allocation so far {:.1} MiB",
            mib(after_forward - before),
            mib(after_backward - before),
            mib(largest)
        );
        drop((o, lse, grads));

        // Through Autodiff, as training runs it: the loss sum(o * dO) adds an elementwise product,
        // a sum and their backward.
        let autodiff = device.clone().autodiff();
        let to_ad =
            |x: &Tensor<4>| Tensor::<4>::from_data(x.to_data(), &autodiff).cast(DType::BF16);
        let (qa, ka, va, ga) = (to_ad(&q), to_ad(&k), to_ad(&v), to_ad(&grad));
        let autodiff_ms = median_ms(&autodiff, || {
            let (q, k, v) = (
                qa.clone().require_grad(),
                ka.clone().require_grad(),
                va.clone().require_grad(),
            );
            let o = causal_attention(q.clone(), k.clone(), v.clone(), scale);
            let grads = (o * ga.clone()).sum().backward();
            let _ = (q.grad(&grads), k.grad(&grads), v.grad(&grads));
        });
        autodiff.sync().unwrap();
        let (before, _) = memory();
        let (qr, kr, vr) = (
            qa.clone().require_grad(),
            ka.clone().require_grad(),
            va.clone().require_grad(),
        );
        let o = causal_attention(qr.clone(), kr.clone(), vr.clone(), scale);
        autodiff.sync().unwrap();
        let (after_forward, _) = memory();
        let grads = (o * ga.clone()).sum().backward();
        let kept = (qr.grad(&grads), kr.grad(&grads), vr.grad(&grads));
        autodiff.sync().unwrap();
        let (after_backward, _) = memory();
        println!(
            "  through Autodiff: forward + backward {autodiff_ms:.2} ms; +{:.1} MiB after the \
             forward, +{:.1} MiB after the backward",
            mib(after_forward - before),
            mib(after_backward - before)
        );
        drop((kept, grads));

        if burn_too {
            burn_attention(&device, &q, &k, &v, &grad, hq / hkv);
        }

        decode(&device, scale);
    }

    /// Burn's own `attention` at the training shape, the key-value heads expanded (it takes no
    /// grouped heads): the forward on CUDA, and the forward and backward through `Autodiff`,
    /// whose backward is the unfused fallback.
    fn burn_attention(
        device: &Device,
        q: &Tensor<4>,
        k: &Tensor<4>,
        v: &Tensor<4>,
        grad: &Tensor<4>,
        group: usize,
    ) {
        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: true,
        };
        let expand = |x: &Tensor<4>| {
            let [b, h, t, d] = x.dims();
            x.clone()
                .reshape([b, h, 1, t, d])
                .expand([b, h, group, t, d])
                .reshape([b, h * group, t, d])
        };
        let (ke, ve) = (expand(k), expand(v));
        let forward = median_ms(device, || {
            let _ = module::attention(q.clone(), ke.clone(), ve.clone(), None, None, options);
        });
        let autodiff = device.clone().autodiff();
        let to_ad =
            |x: &Tensor<4>| Tensor::<4>::from_data(x.to_data(), &autodiff).cast(DType::BF16);
        let (qa, ka, va, ga) = (to_ad(q), to_ad(k), to_ad(v), to_ad(grad));
        let expand_ad = |x: &Tensor<4>| {
            let [b, h, t, d] = x.dims();
            x.clone()
                .reshape([b, h, 1, t, d])
                .expand([b, h, group, t, d])
                .reshape([b, h * group, t, d])
        };
        autodiff.sync().unwrap();
        let (before, _) = memory();
        let both = median_ms(&autodiff, || {
            let (q, k, v) = (
                qa.clone().require_grad(),
                ka.clone().require_grad(),
                va.clone().require_grad(),
            );
            let o = module::attention(q.clone(), expand_ad(&k), expand_ad(&v), None, None, options);
            let grads = (o * ga.clone()).sum().backward();
            let _ = (q.grad(&grads), k.grad(&grads), v.grad(&grads));
        });
        autodiff.sync().unwrap();
        let (_, largest) = memory();
        println!(
            "  Burn's attention (key-value heads expanded): forward {forward:.2} ms; through \
             Autodiff forward + backward {both:.2} ms, largest allocation {:.1} MiB (in use \
             before: {:.1} MiB)",
            mib(largest),
            mib(before)
        );
    }

    /// One query per sequence against a cache, batch 8, per layer.
    fn decode(device: &Device, scale: f32) {
        let (b, hq, hkv, s, d) = (8, 8, 2, 6144, 256);
        println!("decoding: B {b}, Hq {hq}, Hkv {hkv}, cache {s}, D {d}, bf16");
        let q = random([b, hq, 1, d], device);
        let k = random([b, hkv, s, d], device);
        let v = random([b, hkv, s, d], device);
        device.sync().unwrap();
        let ms = median_ms(device, || {
            let _ = attention_decode(q.clone(), k.clone(), v.clone(), scale);
        });
        // A decoding step runs every layer before it synchronises: 24 calls per run.
        let layers = 24;
        let step = median_ms(device, || {
            for _ in 0..layers {
                let _ = attention_decode(q.clone(), k.clone(), v.clone(), scale);
            }
        }) / layers as f64;
        let bytes = (2 * b * hkv * s * d * 2) as f64;
        println!(
            "  attention_decode {:.1} us per layer with a synchronisation each, {:.1} us per layer \
             over {layers} calls between synchronisations ({:.0} GB/s of keys and values read)",
            ms * 1e3,
            step * 1e3,
            bytes / (step * 1e-3) / 1e9
        );

        // Burn's attention takes no grouped heads: the cache expanded to 8 heads first.
        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: true,
        };
        let expand = |x: &Tensor<4>| {
            x.clone()
                .reshape([b, hkv, 1, s, d])
                .expand([b, hkv, hq / hkv, s, d])
                .reshape([b, hq, s, d])
        };
        let (ke, ve) = (expand(&k), expand(&v));
        let expanded = median_ms(device, || {
            let _ = module::attention(q.clone(), ke.clone(), ve.clone(), None, None, options);
        });
        let with_expand = median_ms(device, || {
            let _ = module::attention(q.clone(), expand(&k), expand(&v), None, None, options);
        });
        println!(
            "  Burn's attention on the expanded cache {:.1} us, {:.1} us with the expansion",
            expanded * 1e3,
            with_expand * 1e3
        );
        flash_unit_decode(device, [b, hq, s, d]);
    }

    /// Burn's flash attention asked for by name through burn-cubecl, on the expanded cache: the
    /// default strategy of a Burn built without `autotune`, as this workspace builds it, is the
    /// unfused fallback, and the accelerated flash routine cannot take one query (its stage's
    /// query count must divide the problem's), which leaves the unit routine.
    fn flash_unit_decode(device: &Device, [b, h, s, d]: [usize; 4]) {
        use burn::backend::{Shape, bf16};
        use burn_cubecl::kernel::attention::{AttentionStrategy, attention};
        use burn_cubecl::tensor::CubeTensor;

        let cube = cubecl::Device::cuda(0).expect("CUDA device 0");
        let client = cube.client();
        let make = |shape: [usize; 4]| {
            let n: usize = shape.iter().product();
            let bytes: Vec<u8> = (0..n)
                .flat_map(|i| {
                    bf16::from_f32((i % 97) as f32 / 48.0 - 1.0)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect();
            let handle = client.create_from_slice(&bytes);
            CubeTensor::new_contiguous(
                client.clone(),
                cube.clone(),
                Shape::new(shape),
                handle,
                DType::BF16,
            )
        };
        let (q, k, v) = (make([b, h, 1, d]), make([b, h, s, d]), make([b, h, s, d]));
        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: true,
        };
        let run = || {
            attention(
                q.clone(),
                k.clone(),
                v.clone(),
                None,
                None,
                options,
                AttentionStrategy::FlashUnit,
            )
        };
        match run() {
            Ok(_) => {
                let ms = median_ms(device, || {
                    let _ = run();
                });
                println!(
                    "  Burn's flash attention (unit routine) on the expanded cache {:.1} us",
                    ms * 1e3
                );
            }
            Err(error) => {
                println!("  Burn's flash attention (unit routine) refuses the shape: {error:?}")
            }
        }
    }
}
