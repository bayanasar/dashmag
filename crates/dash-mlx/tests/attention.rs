//! Attention against dash-ops' fixtures, with dash-ops' bounds: the two causal records of its
//! `attention` tests and the eight of its `attention_masks` tests (no mask with `Tq != Tk`, causal
//! with `Tq < Tk`, causal within a window, a window alone, key padding with and without causality,
//! packed sequences, packed cross attention with a window, grouped heads throughout). Each record
//! holds inputs rounded to bf16 and torch's output, log-sum-exp and gradients of
//! `sum(o * do)` from them in float64-exact f32, in torch's `[B, H, T, D]` order, swapped here
//! into `[B, T, H, D]`. Runs in f32 and bf16; then several tiles against plain attention in
//! float64, a query that sees no key, decoding against the training path, and two runs to the
//! same bits.
//!
//! Errors are the largest absolute difference and that over the largest reference magnitude,
//! printed to three figures and rounded up; where the reference log-sum-exp is `-inf` the result
//! must be `-inf` too. f32 needs `MLX_ENABLE_TF32=0`:
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test attention -- --nocapture
#![cfg(target_os = "macos")]

use dash_mlx::attention::{
    AttentionMask, AttentionOptions, Window, attention_decode, attention_with_lse,
    attention_with_lse_reference, probe,
};
use mlx_rs::error::Result;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::transforms::fallible_vjp;
use mlx_rs::{Array, Dtype};
use safetensors::{Dtype as StDtype, SafeTensors};
use std::sync::{Mutex, MutexGuard};

/// One test at a time: they read the probe counters, and share the GPU's default stream.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../dash-ops/tests/fixtures/attention"
);

/// dash-ops' `attention` records (causal) and its `attention_masks` records.
const CAUSAL_RECORDS: [&str; 2] = ["gqa8x2_d256_t40", "gqa6x2_d64_tq24_tk56"];
const MASK_RECORDS: [&str; 8] = [
    "none_b2_gqa2x1_d64_tq19_tk33",
    "causal_b2_gqa4x2_d64_tq17_tk41",
    "causal_window9_gqa3x1_d64_t40",
    "window6x4_b2_gqa2x1_d64_tq24_tk37",
    "padding_b3_gqa2x1_d64_t29",
    "padding_causal_b2_gqa2x1_d64_t26",
    "packed_causal_gqa2x1_d64",
    "packed_cross_causal_window5_gqa2x1_d64",
];

/// dash-ops' bounds, in the order o, lse, dq, dk, dv. f32: those it gives its f32 runs when the
/// kernels doing the sums differ from the ones its tighter bounds were measured on (with
/// autotune: "a few roundings"), since MLX's matmuls and reductions sum in their own order
/// (`attention.rs`, `attention_masks.rs`); bf16: its kernels'.
const CAUSAL_F32: [f64; 5] = [1e-6, 2e-7, 1e-6, 1e-6, 1e-6];
const MASK_F32: [f64; 5] = [1e-6, 2e-7, 1e-6, 1e-6, 1e-6];
/// The log-sum-exp is computed in f32 from the same inputs in either dtype, so in bf16 its bound is
/// the f32 one's where that is the larger.
const CAUSAL_BF16: [f64; 5] = [3.4e-3, 2.5e-7, 4.7e-3, 4e-3, 4.2e-3];
const MASK_BF16: [f64; 5] = [3.2e-3, 2e-7, 4.3e-3, 4.4e-3, 3.6e-3];

fn require_ieee_f32_matmul() {
    assert_eq!(
        std::env::var("MLX_ENABLE_TF32").as_deref(),
        Ok("0"),
        "run with MLX_ENABLE_TF32=0: MLX's f32 matmuls run in TF32 otherwise, about 1e-3 off"
    );
}

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

/// A record: its inputs, mask and options, and torch's `[o, lse, dq, dk, dv]`.
struct Record {
    name: &'static str,
    q: Array,
    k: Array,
    v: Array,
    grad: Array,
    mask: AttentionMask,
    options: AttentionOptions,
    expected: [Array; 5],
}

/// Tensor `name`, as f32, heads and time swapped into `[B, T, H, ...]`.
fn read(tensors: &SafeTensors, name: &str) -> Array {
    let view = tensors.tensor(name).expect("tensor in record");
    let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
    let values: Vec<f32> = match view.dtype() {
        StDtype::F32 => view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        StDtype::BF16 => view
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
            .collect(),
        other => panic!("unexpected dtype {other:?}"),
    };
    Array::from_slice(&values, &shape).swap_axes(1, 2).unwrap()
}

fn record(name: &'static str) -> Record {
    let path = format!("{FIXTURES}/{name}.safetensors");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let (_, metadata) = SafeTensors::read_metadata(&bytes).expect("metadata");
    let metadata = metadata.metadata().clone().expect("metadata");
    let meta = |key: &str| metadata.get(key).cloned();
    let tensors = SafeTensors::deserialize(&bytes).expect("safetensors");
    let q = read(&tensors, "q");
    let scale = meta("scale").expect("scale").parse::<f32>().unwrap();
    let d = q.shape()[3];
    // The causal records carry no `kind`: causal, no mask.
    let causal = meta("causal").is_none_or(|c| c == "true");
    let mut options = AttentionOptions::default().with_causal(causal);
    if scale != (1.0 / (d as f64).sqrt()) as f32 {
        options = options.with_scale(scale);
    }
    let reach = |key: &str| meta(key).map_or(-1, |x| x.parse::<i64>().unwrap());
    let (left, right) = (reach("window_left"), reach("window_right"));
    if left >= 0 || right >= 0 {
        let side = |r: i64| if r < 0 { usize::MAX } else { r as usize };
        options = options.with_window(side(left), side(right));
    }
    let mask = match meta("kind").as_deref() {
        None | Some("none") => AttentionMask::None,
        Some("padding") => {
            let view = tensors.tensor("key_padding").expect("key_padding");
            let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
            let values: Vec<bool> = view.data().iter().map(|&x| x != 0).collect();
            AttentionMask::KeyPadding(Array::from_slice(&values, &shape))
        }
        Some("packed") => {
            let offsets = |key: &str| {
                let view = tensors.tensor(key).expect("offsets");
                let values: Vec<i64> = view
                    .data()
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_le_bytes(*b))
                    .collect();
                Array::from_slice(&values, &[values.len() as i32])
            };
            AttentionMask::Packed {
                cu_seqlens_q: offsets("cu_seqlens_q"),
                cu_seqlens_k: offsets("cu_seqlens_k"),
            }
        }
        Some(other) => panic!("unknown mask kind {other}"),
    };
    Record {
        name,
        q,
        k: read(&tensors, "k"),
        v: read(&tensors, "v"),
        grad: read(&tensors, "do"),
        mask,
        options,
        expected: ["o", "lse", "dq", "dk", "dv"].map(|n| read(&tensors, n)),
    }
}

fn values(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
    a.as_slice::<f32>().to_vec()
}

type Op = fn(&Array, &Array, &Array, &AttentionMask, AttentionOptions) -> Result<(Array, Array)>;

/// The tile kernels and the reference built from MLX's operations.
const PATHS: [(&str, Op); 2] = [
    ("kernels", attention_with_lse),
    ("reference", attention_with_lse_reference),
];

/// The forward with its log-sum-exp, then the backward of `sum(o * grad)`, in `dtype`:
/// `[o, lse, dq, dk, dv]`.
fn run(r: &Record, dtype: Dtype) -> Result<[Array; 5]> {
    run_on(attention_with_lse, r, dtype)
}

fn run_on(op: Op, r: &Record, dtype: Dtype) -> Result<[Array; 5]> {
    let primals = [&r.q, &r.k, &r.v].map(|x| x.as_dtype(dtype).unwrap());
    let grad = r.grad.as_dtype(dtype)?;
    let mut lse = None;
    let f = |a: &[Array]| -> Result<Vec<Array>> {
        let (o, l) = op(&a[0], &a[1], &a[2], &r.mask, r.options)?;
        lse = Some(l);
        Ok(vec![o])
    };
    let (outputs, grads) = fallible_vjp(f, &primals, std::slice::from_ref(&grad))?;
    assert_eq!(outputs[0].dtype(), dtype);
    let lse = lse.expect("ran");
    assert_eq!(lse.dtype(), Dtype::Float32);
    for g in &grads {
        assert_eq!(g.dtype(), dtype);
    }
    Ok([
        outputs[0].clone(),
        lse,
        grads[0].clone(),
        grads[1].clone(),
        grads[2].clone(),
    ])
}

/// The largest absolute difference and that over the largest reference magnitude, over the
/// entries where the reference is finite; where it is `-inf` the result must be `-inf` too.
fn errors(what: &str, actual: &Array, expected: &Array) -> (f64, f64) {
    assert_eq!(actual.shape(), expected.shape(), "{what}: shapes");
    let (a, e) = (values(actual), values(expected));
    let (mut max_abs, mut max_ref) = (0.0f64, 0.0f64);
    for (i, (a, e)) in a.iter().zip(&e).enumerate() {
        if *e == f32::NEG_INFINITY {
            assert_eq!(
                *a,
                f32::NEG_INFINITY,
                "{what}[{i}]: {a} where -inf is expected"
            );
            continue;
        }
        assert!(a.is_finite(), "{what}[{i}]: non-finite {a}");
        max_abs = max_abs.max((*a as f64 - *e as f64).abs());
        max_ref = max_ref.max((*e as f64).abs());
    }
    (max_abs, max_abs / max_ref.max(f64::MIN_POSITIVE))
}

fn check(
    label: &str,
    actual: &[Array; 5],
    expected: &[Array; 5],
    bounds: [f64; 5],
    worst: &mut [f64; 5],
) {
    for (i, name) in ["o", "lse", "dq", "dk", "dv"].iter().enumerate() {
        let (abs, rel) = errors(&format!("{label} {name}"), &actual[i], &expected[i]);
        println!(
            "{label:52} {name:3} max abs {abs:9.3e}  relative {}",
            up3(rel)
        );
        assert!(
            rel <= bounds[i],
            "{label} {name}: {rel:.3e} above {:.1e}",
            bounds[i]
        );
        worst[i] = worst[i].max(rel);
    }
}

#[test]
fn matches_torch_for_every_mask() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    for ((path, op), (dtype, causal_bounds, mask_bounds)) in PATHS.into_iter().flat_map(|p| {
        [
            (p, (Dtype::Float32, CAUSAL_F32, MASK_F32)),
            (p, (Dtype::Bfloat16, CAUSAL_BF16, MASK_BF16)),
        ]
    }) {
        let mut worst = [0.0; 5];
        for (names, bounds) in [
            (&CAUSAL_RECORDS[..], causal_bounds),
            (&MASK_RECORDS[..], mask_bounds),
        ] {
            for &name in names {
                let r = record(name);
                let runs = probe::backward_runs();
                let actual = run_on(op, &r, dtype)?;
                assert!(
                    probe::backward_runs() > runs,
                    "{name}: the blocked backward did not run"
                );
                check(
                    &format!("{path} {dtype:?} {}", r.name),
                    &actual,
                    &r.expected,
                    bounds,
                    &mut worst,
                );
            }
        }
        println!(
            "{path} {dtype:?}: largest o {}, lse {}, dq {}, dk {}, dv {}",
            up3(worst[0]),
            up3(worst[1]),
            up3(worst[2]),
            up3(worst[3]),
            up3(worst[4])
        );
    }
    Ok(())
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

/// Decoding against the training path with one query: in f32 within f32's rounding; in bf16, on
/// inputs rounded to bf16, the f32 result rounded once (within one bf16 unit of each value). A
/// full cache, per-sequence lengths (one sequence holding no key) and a window's left side; and its
/// gradients to the query and the cache.
#[test]
fn decoding_is_attention_with_one_query() -> Result<()> {
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
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let cast = |x: &Array| x.as_dtype(dtype).unwrap();
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
                    assert!(
                        got_row.iter().all(|&x| x == 0.0),
                        "{label} {dtype:?}: no key, zeros"
                    );
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
                let want = want.reshape(&[hq, d])?;
                let (abs, rel) = errors("decode", &Array::from_slice(&got_row, &[hq, d]), &want);
                println!(
                    "decode {label} {dtype:?} sequence {row}: max abs {abs:9.3e}  relative {}",
                    up3(rel)
                );
                if dtype == Dtype::Float32 {
                    assert!(rel <= 3e-7, "decode {label} f32 {row}: {rel:.3e}");
                    continue;
                }
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
fn repeats_bit_for_bit() -> Result<()> {
    let _serial = serial();
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for name in ["gqa8x2_d256_t40", "padding_causal_b2_gqa2x1_d64_t26"] {
            let r = record(name);
            let bits = |x: [Array; 5]| {
                x.map(|a| values(&a).iter().map(|f| f.to_bits()).collect::<Vec<_>>())
            };
            assert!(
                bits(run(&r, dtype)?) == bits(run(&r, dtype)?),
                "{dtype:?} {name}: two runs differ"
            );
        }
    }
    Ok(())
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
    let mut worst = [0.0; 5];
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
        check(label, &got, &want?, MASK_F32, &mut worst);
    }
    println!(
        "tiles against plain attention in float64: largest o {}, lse {}, dq {}, dk {}, dv {}",
        up3(worst[0]),
        up3(worst[1]),
        up3(worst[2]),
        up3(worst[3]),
        up3(worst[4])
    );
    Ok(())
}
