//! Attention: `dash_conformance`'s `attention` tests through dash-mlx's adapter (`tests/common`),
//! on the GPU, through its tile kernels and through the reference built from MLX's operations:
//! dash-ops' ten records against torch (causal with `Tq != Tk`, no mask, causal within a window, a
//! window alone, key padding with and without causality, packed sequences, packed cross attention
//! with a window, grouped heads throughout), the output, the log-sum-exp and the three gradients
//! of `sum(o * do)`; the same without gradients; decoding against attention with one query. The
//! adapter asserts that the blocked backward runs where gradients are asked for.
//!
//! The bounds are the caller's. f32 takes dash-ops' under Burn's autotune ("a few roundings"), since
//! MLX's matmuls and reductions sum in their own order; bf16 its kernels', but for the log-sum-exp
//! of the mask records, computed in f32 from the same inputs in either dtype, which takes the f32
//! bound.
//!
//! What is dash-mlx's own stays here: a query before every key, decoding in bf16 and its gradients,
//! two runs to the same bits, and several tiles against plain attention in float64.
//!
//! f32 needs `MLX_ENABLE_TF32=0`:
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test attention -- --nocapture
#![cfg(target_os = "macos")]

mod common;

use common::{Mlx, Reference, host, require_ieee_f32_matmul};
use dash_conformance::attention::{self as conformance, Bounds, Record};
use dash_conformance::{Adapter, DType};
use dash_mlx::attention::{
    AttentionMask, AttentionOptions, Window, attention_decode, attention_with_lse, probe,
};
use mlx_rs::error::Result;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::transforms::fallible_vjp;
use mlx_rs::{Array, Dtype};
use std::sync::{Mutex, MutexGuard};

/// One test at a time: they read the probe counters, and share the GPU's default stream.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const F32: Bounds = conformance::AUTOTUNE_F32;
const BF16: Bounds = Bounds {
    masks: [3.2e-3, 2e-7, 4.3e-3, 4.4e-3, 3.6e-3],
    ..conformance::BF16
};

/// `x` to three significant figures, rounded up: the form the README and PROGRESS quote.
fn up3(x: f64) -> String {
    if x <= 0.0 {
        return "0".into();
    }
    let mut e = x.log10().floor() as i32;
    let mut m = (x / 10f64.powi(e - 2) * (1.0 - 1e-12)).ceil();
    if m >= 1000.0 {
        m /= 10.0;
        e += 1;
    }
    format!("{:.2}e{e}", m / 100.0)
}

fn values(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
    a.as_slice::<f32>().to_vec()
}

/// Every record in `dtype`, the query gradient summed either way (the same pass here), through the
/// kernels and the reference; the largest errors over the two, printed as the README quotes them.
fn matches_torch(dtype: DType, bounds: Bounds) {
    let adapters: [(&dyn Adapter, &str); 2] = [(&Mlx, "kernels"), (&Reference, "reference")];
    for (adapter, path) in adapters {
        for deterministic in [true, false] {
            let label = format!("{path} {dtype:?} deterministic {deterministic}");
            let worst =
                conformance::matches_torch(adapter, &label, dtype, deterministic, bounds).all_ran();
            let largest: [String; 5] =
                std::array::from_fn(|i| up3(worst[0][i].max(worst[1][i]) as f64));
            println!(
                "{label}: largest o {}, lse {}, dq {}, dk {}, dv {}",
                largest[0], largest[1], largest[2], largest[3], largest[4]
            );
        }
    }
}

#[test]
fn matches_torch_in_f32() {
    let _serial = serial();
    require_ieee_f32_matmul();
    matches_torch(DType::F32, F32);
}

#[test]
fn matches_torch_in_bf16() {
    let _serial = serial();
    matches_torch(DType::BF16, BF16);
}

#[test]
fn without_gradients() {
    let _serial = serial();
    require_ieee_f32_matmul();
    conformance::without_gradients(&Mlx, "kernels", F32).all_ran();
    conformance::without_gradients(&Reference, "reference", F32).all_ran();
}

#[test]
fn decoding_is_attention_with_one_query() {
    let _serial = serial();
    require_ieee_f32_matmul();
    // dash-ops' bounds are 0 and 1e-6: there decoding and attention run the same kernels. Here a
    // kernel of its own decodes and sums in its own order (measured 2.98e-7 and 3.58e-7), so both
    // take the second, a few roundings.
    conformance::decoding_is_attention_with_one_query(&Mlx, "kernels", [1e-6, 1e-6]).all_ran();
}

/// A query that sees no key: zeros out, `-inf` log-sum-exp, no gradient from it.
#[test]
fn a_query_that_sees_no_key() -> Result<()> {
    let _serial = serial();
    // Causal with Tq > Tk: the first Tq - Tk queries sit before every key.
    let normal = |shape: &[i32], seed: u64| -> Result<Array> {
        mlx_rs::random::normal::<f32>(shape, None, None, &mlx_rs::random::key(seed)?)
    };
    let (q, k, v) = (
        normal(&[1, 6, 2, 64], 1)?,
        normal(&[1, 4, 1, 64], 2)?,
        normal(&[1, 4, 1, 64], 3)?,
    );
    let (o, lse) =
        attention_with_lse(&q, &k, &v, &AttentionMask::None, AttentionOptions::causal())?;
    let (o, lse) = (values(&o), values(&lse));
    assert!(o[..2 * 2 * 64].iter().all(|&x| x == 0.0), "zeros out");
    assert!(
        lse[..2 * 2].iter().all(|&x| x == f32::NEG_INFINITY),
        "-inf log-sum-exp"
    );
    assert!(lse[2 * 2..].iter().all(|x| x.is_finite()));
    let f = |a: &[Array]| -> Result<Vec<Array>> {
        Ok(vec![
            attention_with_lse(
                &a[0],
                &a[1],
                &a[2],
                &AttentionMask::None,
                AttentionOptions::causal(),
            )?
            .0,
        ])
    };
    let (_, grads) = fallible_vjp(f, &[q.clone(), k, v], &[normal(&[1, 6, 2, 64], 4)?])?;
    let dq = values(&grads[0]);
    assert!(
        dq[..2 * 2 * 64].iter().all(|&x| x == 0.0),
        "no query gradient"
    );
    assert!(dq[2 * 2 * 64..].iter().all(|x| x.is_finite()));
    Ok(())
}

/// What the shared body leaves out of decoding. In bf16, on inputs rounded to bf16, the f32 result
/// rounded once: within one bf16 unit of each value of the training path with one query. A full
/// cache, per-sequence lengths (one sequence holding no key) and a window's left side; and its
/// gradients to the query and the cache.
#[test]
fn decoding_in_bf16_and_its_gradients() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    let normal = |shape: &[i32], seed: u64| -> Result<Array> {
        mlx_rs::random::normal::<f32>(shape, None, None, &mlx_rs::random::key(seed)?)
    };
    let (b, s, hq, hkv, d) = (3, 37, 8, 2, 256);
    // Inputs rounded to bf16, as dash-ops' records are, so that a bf16 run reads them exactly.
    let exact = |x: Array| x.as_dtype(Dtype::Bfloat16)?.as_dtype(Dtype::Float32);
    let q = exact(normal(&[b, hq, d], 5)?)?;
    let k = exact(normal(&[b, s, hkv, d], 6)?)?;
    let v = exact(normal(&[b, s, hkv, d], 7)?)?;
    let lengths = [37i32, 20, 0];
    for (window, label) in [(None, "no window"), (Some(9usize), "window 9")] {
        let options = match window {
            Some(left) => AttentionOptions::default().with_window(left, 0),
            None => AttentionOptions::default(),
        };
        let cast = |x: &Array| x.as_dtype(Dtype::Bfloat16).unwrap();
        let got = attention_decode(
            &cast(&q),
            &cast(&k),
            &cast(&v),
            Some(&Array::from_slice(&lengths, &[b])),
            options,
        )?;
        // The training path, one sequence at a time, the query after its last key.
        for (row, &len) in lengths.iter().enumerate() {
            let got_row = values(&got.try_index((row as i32, .., ..))?);
            if len == 0 {
                assert!(got_row.iter().all(|&x| x == 0.0), "{label}: no key, zeros");
                continue;
            }
            let slice = |x: &Array| {
                x.try_index((row as i32..row as i32 + 1, 0..len, .., ..))
                    .unwrap()
            };
            let one = q
                .try_index((row as i32..row as i32 + 1, .., ..))?
                .expand_dims(1)?;
            let want = dash_mlx::attention::attention(
                &one,
                &slice(&k),
                &slice(&v),
                &AttentionMask::None,
                options.with_causal(true),
            )?;
            // Computed in f32 and rounded once: within one bf16 unit of each f32 value.
            for (i, (g, w)) in got_row.iter().zip(values(&want)).enumerate() {
                let unit = w.abs().max(f32::MIN_POSITIVE).log2().floor().exp2() / 128.0;
                assert!(
                    (g - w).abs() <= unit,
                    "decode {label} bf16 {row}[{i}]: {g} against {w}"
                );
            }
        }
    }
    // Gradients flow to the query and the cache.
    let f = |a: &[Array]| -> Result<Vec<Array>> {
        Ok(vec![attention_decode(
            &a[0],
            &a[1],
            &a[2],
            Some(&Array::from_slice(&lengths, &[b])),
            AttentionOptions::default(),
        )?])
    };
    let (_, grads) = fallible_vjp(
        f,
        &[q.clone(), k.clone(), v.clone()],
        &[normal(&[b, hq, d], 8)?],
    )?;
    for (g, name) in grads.iter().zip(["query", "key", "value"]) {
        let x = values(g);
        assert!(x.iter().all(|v| v.is_finite()), "{name} gradient finite");
        assert!(x.iter().any(|&v| v != 0.0), "{name} gradient reaches it");
    }
    // Keys past a sequence's length get none.
    let dk = values(&grads[1].try_index((1, 20.., .., ..))?);
    assert!(dk.iter().all(|&x| x == 0.0), "no gradient past the length");
    Ok(())
}

/// By default a run repeats bit for bit: the forward and backward passes, twice, in f32 and bf16.
#[test]
fn repeats_bit_for_bit() {
    let _serial = serial();
    for dtype in [DType::F32, DType::BF16] {
        for name in ["gqa8x2_d256_t40", "padding_causal_b2_gqa2x1_d64_t26"] {
            let record = Record::load(name);
            let bits = || {
                let out = Mlx.attention(&record.call(dtype, true, true)).unwrap();
                let grads = out.grads.unwrap();
                [out.out, out.lse, grads.q, grads.k, grads.v]
                    .map(|x| x.values.iter().map(|f| f.to_bits()).collect::<Vec<_>>())
            };
            assert!(bits() == bits(), "{dtype:?} {name}: two runs differ");
        }
    }
}

/// Attention as plain MLX operations, differentiated by MLX: the scores, the band as a
/// mask, the softmax and the value product, `[o, lse]` in the operation's layout.
fn plain(q: &Array, k: &Array, v: &Array, options: AttentionOptions) -> Result<Vec<Array>> {
    let (b, tq, hq, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (tk, hkv) = (k.shape()[1], k.shape()[2]);
    let g = hq / hkv;
    let qg = q
        .reshape(&[b, tq, hkv, g, d])?
        .transpose_axes(&[0, 2, 3, 1, 4])?;
    let heads = |x: &Array| x.swap_axes(1, 2)?.expand_dims(2);
    let (kh, vh) = (heads(k)?, heads(v)?);
    let scale = options.scale.unwrap_or(1.0 / (d as f32).sqrt());
    let scores = qg
        .multiply(Array::from_f32(scale))?
        .matmul(kh.swap_axes(-1, -2)?)?;
    let offset = tk - tq;
    let mut keep = vec![true; (tq * tk) as usize];
    for i in 0..tq {
        let pos = (i + offset) as i64;
        for j in 0..tk {
            let j64 = j as i64;
            let mut sees = !(options.causal && j64 > pos);
            if let Some(w) = options.window {
                sees &= j64 >= pos - w.left.min(1 << 30) as i64;
                sees &= j64 <= pos + w.right.min(1 << 30) as i64;
            }
            keep[(i * tk + j) as usize] = sees;
        }
    }
    let keep = Array::from_slice(&keep, &[tq, tk]);
    let scores = mlx_rs::ops::select(&keep, &scores, Array::from_f32(f32::NEG_INFINITY))?;
    let lse = scores.logsumexp_axis(-1, None)?;
    let p = mlx_rs::ops::softmax_axis(&scores, -1, None)?;
    let o = p.matmul(&vh)?;
    Ok(vec![
        o.reshape(&[b, hq, tq, d])?.swap_axes(1, 2)?,
        lse.reshape(&[b, hq, tq])?.swap_axes(1, 2)?,
    ])
}

/// The blocked passes against plain attention in float64 (MLX's operations on the CPU, differentiated
/// by MLX) at sizes the records do not reach: several tiles of queries a block of keys, the last
/// tiles partial, the band cutting through tiles; f32 with dash-ops' f32 bounds, the output, the
/// log-sum-exp and the three gradients of `sum(o * do)`.
#[test]
fn tiles_match_plain_attention_at_scale() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    let normal = |shape: &[i32], seed: u64| -> Result<Array> {
        mlx_rs::random::normal::<f32>(shape, None, None, &mlx_rs::random::key(seed)?)
    };
    let causal = AttentionOptions {
        causal: true,
        ..Default::default()
    };
    let cases = [
        ("causal, T 1300", 1300, 1300, causal),
        (
            "causal, window 300, Tq 700, Tk 1300",
            700,
            1300,
            AttentionOptions {
                window: Some(Window {
                    left: 300,
                    right: usize::MAX,
                }),
                ..causal
            },
        ),
        (
            "no mask, Tq 1300, Tk 1100",
            1300,
            1100,
            AttentionOptions::default(),
        ),
    ];
    let mut worst = [0.0f32; 5];
    for (label, tq, tk, options) in cases {
        let (hq, hkv, d) = (4, 2, 64);
        let q = normal(&[1, tq, hq, d], 1)?;
        let k = normal(&[1, tk, hkv, d], 2)?;
        let v = normal(&[1, tk, hkv, d], 3)?;
        let grad = normal(&[1, tq, hq, d], 4)?;
        let primals = [q, k, v];
        let both = |f: &dyn Fn(&[Array]) -> Result<Vec<Array>>| -> Result<[Array; 5]> {
            let mut lse = None;
            let (out, grads) = fallible_vjp(
                |a: &[Array]| {
                    let out = f(a)?;
                    lse = Some(out[1].clone());
                    Ok(vec![out[0].clone()])
                },
                &primals,
                std::slice::from_ref(&grad),
            )?;
            Ok([
                out[0].clone(),
                lse.expect("ran"),
                grads[0].clone(),
                grads[1].clone(),
                grads[2].clone(),
            ])
        };
        let tiles = probe::tiles();
        let got = both(&|a: &[Array]| {
            let (o, lse) = attention_with_lse(&a[0], &a[1], &a[2], &AttentionMask::None, options)?;
            Ok(vec![o, lse])
        })?;
        assert!(probe::tiles() > tiles);
        // The reference in float64, on the CPU (MLX's GPU has no float64), rounded to f32.
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        let want = (|| -> Result<[Array; 5]> {
            let wide = |x: &Array| x.as_dtype(Dtype::Float64);
            let primals = [wide(&primals[0])?, wide(&primals[1])?, wide(&primals[2])?];
            let grad = wide(&grad)?;
            let mut lse = None;
            let (out, grads) = fallible_vjp(
                |a: &[Array]| {
                    let out = plain(&a[0], &a[1], &a[2], options)?;
                    lse = Some(out[1].clone());
                    Ok(vec![out[0].clone()])
                },
                &primals,
                std::slice::from_ref(&grad),
            )?;
            let narrow = |x: &Array| -> Result<Array> {
                let x = x.as_dtype(Dtype::Float32)?;
                mlx_rs::transforms::eval([&x])?;
                Ok(x)
            };
            Ok([
                narrow(&out[0])?,
                narrow(&lse.expect("ran"))?,
                narrow(&grads[0])?,
                narrow(&grads[1])?,
                narrow(&grads[2])?,
            ])
        })();
        mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
        let errors = conformance::check(
            label,
            got.each_ref().map(host).each_ref(),
            want?.each_ref().map(host).each_ref(),
            F32.masks,
        );
        for (w, e) in worst.iter_mut().zip(errors) {
            *w = w.max(e);
        }
    }
    println!(
        "tiles against plain attention in float64: largest o {}, lse {}, dq {}, dk {}, dv {}",
        up3(worst[0] as f64),
        up3(worst[1] as f64),
        up3(worst[2] as f64),
        up3(worst[3] as f64),
        up3(worst[4] as f64)
    );
    Ok(())
}
