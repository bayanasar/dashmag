//! Time and device memory of `dash_ops::attention` on a CUDA GPU, at Qwen3.5-0.8B's
//! full-attention head shapes (8 query heads, 2 key-value heads, head dimension 256) in bf16,
//! for each kind of mask.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example attention_bench
//!   ... --example attention_bench -- 8192        # another sequence length
//!   ... --example attention_bench -- 6144 burn   # also time Burn's own attention (needs ~5 GB)
//!   ... --example attention_bench -- 6144 packed # only what starts with "packed" (or
//!                                                # "causal", "window", "no mask", "decoding")
//!
//! Training: T tokens, queries `[1, T, 8, 256]`, keys and values `[1, T, 2, 256]`, with each mask
//! in turn: causal; none (every query sees every key); four packed sequences of `T / 4`
//! tokens, causal within each; causal within a sliding window of 1024 keys. For each, the forward
//! (with its log-sum-exp), the backward (`delta`, the five products, the gradients rounded from
//! their f32 accumulators) and both; for causal also both through `Autodiff` with `sum(o * dO)`
//! as the loss, and the memory. Decoding: batch 8, one query each (`[8, 8, 256]`), against a
//! 6144-token cache (`[8, 6144, 2, 256]`), per layer: the whole cache, mixed cache lengths, and a
//! window of 1024. Beside them, Burn's own `attention`, on inputs of its own in its layout,
//! `[B, H, T, D]`, which takes no grouped heads (the key-value heads are expanded for it), and,
//! for decoding, its flash kernel asked for by name. Each time is the median of 20 runs after 3
//! warm-ups, with the device synchronised around each run, so a time includes one launch and
//! synchronisation; decoding is also timed over 24 calls between synchronisations, as a step over
//! the layers runs. Memory is the growth of the bytes CubeCL's pools hold in use, read after
//! synchronising (under Burn's fusion nothing has run before), and the largest single allocation
//! the pools that hold tensors served.
//!
//! The times are wall clock; on a card other work shares, compare device times instead: run the
//! benchmark under `nsys profile -t cuda` and take each kernel's median from
//! `nsys stats -r cuda_gpu_kern_sum` (the forward is `forward_kernel`; the backward
//! `delta_kernel`, `backward_kernel` and `finish_kernel`).

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
    use burn::tensor::{Device, Distribution, Int, Tensor, TensorData, module};
    use dash_ops::attention::{
        AttentionMask, AttentionOptions, attention, attention_backward, attention_decode,
        attention_with_lse,
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

    fn random<const N: usize>(shape: [usize; N], device: &Device) -> Tensor<N> {
        Tensor::<N>::random(shape, Distribution::Normal(0.0, 1.0), device).cast(DType::BF16)
    }

    fn ints(values: Vec<i32>, device: &Device) -> Tensor<1, Int> {
        let n = values.len();
        Tensor::<1, Int>::from_data(TensorData::new(values, [n]), device)
    }

    /// One mask to time: its name, the mask (rebuilt for each call, its tensors cloned) and the
    /// options.
    struct Kind {
        name: &'static str,
        mask: AttentionMask,
        options: AttentionOptions,
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let t: usize = args
            .get(1)
            .map_or(6144, |a| a.parse().expect("sequence length"));
        let burn_too = args.iter().any(|a| a == "burn");
        // The other arguments pick what runs: what names start with one of them.
        let filters: Vec<&str> = args[1..]
            .iter()
            .map(String::as_str)
            .filter(|a| a.parse::<usize>().is_err() && *a != "burn")
            .collect();
        let picked = |name: &str| filters.is_empty() || filters.iter().any(|f| name.starts_with(f));
        let device = Device::cuda(0);
        let (b, hq, hkv, d) = (1, 8, 2, 256);
        println!("training: B {b}, Hq {hq}, Hkv {hkv}, T {t}, D {d}, bf16");

        let q = random([b, t, hq, d], &device);
        let k = random([b, t, hkv, d], &device);
        let v = random([b, t, hkv, d], &device);
        let grad = random([b, t, hq, d], &device);
        device.sync().unwrap();

        let quarter = (t / 4) as i32;
        let cu = ints((0..=4).map(|i| i * quarter).collect(), &device);
        let kinds = [
            Kind {
                name: "causal",
                mask: AttentionMask::None,
                options: AttentionOptions::causal(),
            },
            Kind {
                name: "no mask",
                mask: AttentionMask::None,
                options: AttentionOptions::default(),
            },
            Kind {
                name: "packed sequences, 4 of T / 4, causal",
                mask: AttentionMask::Packed {
                    cu_seqlens_q: cu.clone(),
                    cu_seqlens_k: cu,
                },
                options: AttentionOptions::causal(),
            },
            Kind {
                name: "window of 1024, causal",
                mask: AttentionMask::None,
                options: AttentionOptions::causal().with_window(1024, 0),
            },
        ];
        for kind in kinds.iter().filter(|kind| picked(kind.name)) {
            let (mask, options) = (|| kind.mask.clone(), kind.options);
            let forward = median_ms(&device, || {
                let _ = attention(q.clone(), k.clone(), v.clone(), mask(), options);
            });
            let (o, lse) = attention_with_lse(q.clone(), k.clone(), v.clone(), mask(), options);
            let backward = median_ms(&device, || {
                let _ = attention_backward(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    o.clone(),
                    lse.clone(),
                    grad.clone(),
                    mask(),
                    options,
                );
            });
            let both = median_ms(&device, || {
                let (o, lse) = attention_with_lse(q.clone(), k.clone(), v.clone(), mask(), options);
                let _ = attention_backward(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    o,
                    lse,
                    grad.clone(),
                    mask(),
                    options,
                );
            });
            drop((o, lse));
            println!(
                "  {}: forward {forward:.2} ms, backward {backward:.2} ms, forward + backward \
                 {both:.2} ms",
                kind.name
            );
        }
        let causal = AttentionOptions::causal();
        if !picked("causal") {
            drop((q, k, v, grad));
            if picked("decoding") {
                decode(&device);
            }
            return;
        }

        // Memory of one causal forward and backward, everything they return kept alive.
        device.sync().unwrap();
        let (before, _) = memory();
        let (o, lse) =
            attention_with_lse(q.clone(), k.clone(), v.clone(), AttentionMask::None, causal);
        device.sync().unwrap();
        let (after_forward, _) = memory();
        let grads = attention_backward(
            q.clone(),
            k.clone(),
            v.clone(),
            o.clone(),
            lse.clone(),
            grad.clone(),
            AttentionMask::None,
            causal,
        );
        device.sync().unwrap();
        let (after_backward, largest) = memory();
        println!(
            "  causal, memory over the inputs: +{:.1} MiB after the forward (output, \
             log-sum-exp), +{:.1} MiB after the backward (and dQ, dK, dV); largest allocation so \
             far {:.1} MiB",
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
            let o = attention(q.clone(), k.clone(), v.clone(), AttentionMask::None, causal);
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
        let o = attention(
            qr.clone(),
            kr.clone(),
            vr.clone(),
            AttentionMask::None,
            causal,
        );
        autodiff.sync().unwrap();
        let (after_forward, _) = memory();
        let grads = (o * ga.clone()).sum().backward();
        let kept = (qr.grad(&grads), kr.grad(&grads), vr.grad(&grads));
        autodiff.sync().unwrap();
        let (after_backward, _) = memory();
        println!(
            "  causal through Autodiff: forward + backward {autodiff_ms:.2} ms; +{:.1} MiB after \
             the forward, +{:.1} MiB after the backward",
            mib(after_forward - before),
            mib(after_backward - before)
        );
        drop((kept, grads));

        drop((q, k, v, grad));
        if burn_too {
            burn_attention(&device, [b, hq, hkv, t, d]);
        }
        if picked("decoding") {
            decode(&device);
        }
    }

    /// Burn's own `attention` at the training shape, on inputs of its own in its layout,
    /// `[B, H, T, D]`, the key-value heads expanded (it takes no grouped heads): the forward on
    /// CUDA, and the forward and backward through `Autodiff`, whose backward is the unfused
    /// fallback.
    fn burn_attention(device: &Device, [b, hq, hkv, t, d]: [usize; 5]) {
        let group = hq / hkv;
        let q = &random([b, hq, t, d], device);
        let k = &random([b, hkv, t, d], device);
        let v = &random([b, hkv, t, d], device);
        let grad = &random([b, hq, t, d], device);
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

    /// One query per sequence against a cache, batch 8, per layer: the whole cache, mixed cache
    /// lengths, a window of 1024.
    fn decode(device: &Device) {
        let (b, hq, hkv, s, d) = (8, 8, 2, 6144, 256);
        println!("decoding: B {b}, Hq {hq}, Hkv {hkv}, cache {s}, D {d}, bf16");
        let q = random([b, hq, d], device);
        let k = random([b, s, hkv, d], device);
        let v = random([b, s, hkv, d], device);
        device.sync().unwrap();
        // One length per sequence.
        let mixed = vec![6144, 5000, 4096, 3000, 2048, 1024, 512, 100];
        let held: i32 = mixed.iter().sum();
        let lengths = ints(mixed.clone(), device);
        let full = AttentionOptions::default();
        let window = AttentionOptions::default().with_window(1024, 0);
        let cases: [(&str, Option<&Tensor<1, Int>>, AttentionOptions, usize); 3] = [
            ("whole cache", None, full, b * s),
            ("mixed lengths", Some(&lengths), full, held as usize),
            ("window 1024", None, window, b * 1025),
        ];
        for (name, lengths, options, keys) in cases {
            let call = || {
                let _ =
                    attention_decode(q.clone(), k.clone(), v.clone(), lengths.cloned(), options);
            };
            let ms = median_ms(device, call);
            // A decoding step runs every layer before it synchronises: 24 calls per run.
            let layers = 24;
            let step = median_ms(device, || {
                for _ in 0..layers {
                    call();
                }
            }) / layers as f64;
            let bytes = (2 * keys * hkv * d * 2) as f64;
            println!(
                "  {name}: {:.1} us per layer with a synchronisation each, {:.1} us per layer \
                 over {layers} calls between synchronisations ({:.0} GB/s of keys and values \
                 read)",
                ms * 1e3,
                step * 1e3,
                bytes / (step * 1e-3) / 1e9
            );
        }

        // Burn's attention takes no grouped heads and its own layout: a query and a cache of its
        // own, `[B, H, T, D]`, the cache expanded to 8 heads first.
        drop((q, k, v));
        let q = random([b, hq, 1, d], device);
        let k = random([b, hkv, s, d], device);
        let v = random([b, hkv, s, d], device);
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
