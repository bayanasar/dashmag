//! The chosen-token log-probability and entropy against dash-ops' fixtures, with dash-ops' bounds.
//!
//! `fixtures/logprob/cases.safetensors` (dash-ops') holds three cases, inputs rounded to bf16 so
//! that a bf16 run casts them without loss, and PyTorch's float64 results stored as f32: the
//! log-probabilities, the entropies, and the gradient of `sum_i g[i] logprob[i]` for `hidden`,
//! with and without the masked token ids. `basic` has two masked ids at the two ends of the
//! vocabulary; `peaked` logits up to 49, targets with p = 0.99 and a masked token that is most
//! rows' argmax; `wide` D = 256. In each the last row's target is masked, so its log-probability
//! is `-inf`. Each case runs in f32 and bf16, with dash-ops' small chunks (several chunks a row
//! block, the last partial; uneven row blocks) and with the defaults.
//!
//! Errors are `max |got - want|` and that over `max |want|`, per output, printed to three figures
//! and rounded up. f32 needs `MLX_ENABLE_TF32=0`:
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test logprob -- --nocapture
#![cfg(target_os = "macos")]

use dash_mlx::logprob::{
    ChosenLogprob, LogprobOptions, chosen_logprob, chosen_logprob_reference, probe,
};
use mlx_rs::error::Result;
use mlx_rs::transforms::fallible_vjp;
use mlx_rs::{Array, Dtype};
use safetensors::SafeTensors;
use std::sync::{Mutex, MutexGuard};

/// One test at a time: they read the probe counters, and share the GPU's default stream.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../dash-ops/tests/fixtures/logprob"
);
const CASES: [&str; 3] = ["basic", "peaked", "wide"];

/// dash-ops' small chunks: every case crosses several, the last partial, and the row blocks
/// split the rows unevenly.
const SMALL_CHUNKS: LogprobOptions = LogprobOptions {
    vocab_chunk: 128,
    row_chunk: 10,
};

/// dash-ops' bounds, `[logprob, entropy, gradient]`: f32 throughout; and from bf16 inputs, the
/// gradient rounded to bf16 on the way out (half an ulp is 2^-9 of the value).
const F32: [f64; 3] = [2e-7, 2e-6, 1e-6];
const BF16: [f64; 3] = [2e-7, 2e-6, 4e-3];

/// MLX runs f32 products in TF32 unless told not to; the f32 bounds assume IEEE f32.
fn require_ieee_f32_matmul() {
    assert_eq!(
        std::env::var("MLX_ENABLE_TF32").as_deref(),
        Ok("0"),
        "run with MLX_ENABLE_TF32=0: MLX's f32 matmuls run in TF32 otherwise, about 1e-3 off"
    );
}

struct Case {
    n: i32,
    v: i32,
    d: i32,
    hidden: Vec<f32>,
    weight: Vec<f32>,
    targets: Vec<i64>,
    upstream: Vec<f32>,
    masked: Vec<u32>,
    tensors: Vec<u8>,
    name: &'static str,
}

impl Case {
    fn load(name: &'static str) -> Self {
        let tensors = std::fs::read(format!("{FIXTURES}/cases.safetensors")).unwrap();
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{FIXTURES}/meta.json")).unwrap(),
        )
        .unwrap();
        let meta = &meta[name];
        let dim = |key: &str| meta[key].as_i64().unwrap() as i32;
        let mut case = Self {
            n: dim("n"),
            v: dim("v"),
            d: dim("d"),
            hidden: Vec::new(),
            weight: Vec::new(),
            targets: Vec::new(),
            upstream: Vec::new(),
            masked: meta["masked"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap() as u32)
                .collect(),
            tensors,
            name,
        };
        case.hidden = case.f32s("hidden");
        case.weight = case.f32s("weight");
        case.upstream = case.f32s("upstream");
        let file = SafeTensors::deserialize(&case.tensors).unwrap();
        let view = file.tensor(&format!("{name}.targets")).unwrap();
        assert_eq!(view.dtype(), safetensors::Dtype::I64);
        case.targets = view
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect();
        case
    }

    fn f32s(&self, key: &str) -> Vec<f32> {
        let file = SafeTensors::deserialize(&self.tensors).unwrap();
        let view = file.tensor(&format!("{}.{key}", self.name)).unwrap();
        assert_eq!(view.dtype(), safetensors::Dtype::F32, "{key}");
        view.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// PyTorch's `[logprob, entropy, grad_hidden]`, with the masked ids or without.
    fn reference(&self, masked: bool) -> [Vec<f32>; 3] {
        let suffix = if masked { "" } else { "_unmasked" };
        ["logprob", "entropy", "grad_hidden"].map(|key| self.f32s(&format!("{key}{suffix}")))
    }
}

fn values(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
    a.as_slice::<f32>().to_vec()
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

/// `(max |got - want|, that / max |want|)` over the finite entries; `-inf` must match `-inf`.
fn error(what: &str, got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut max_abs = 0.0f64;
    let mut scale = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.is_infinite() {
            assert_eq!(g, w, "{what}[{i}]: want {w}, got {g}");
            continue;
        }
        assert!(g.is_finite(), "{what}[{i}]: want {w}, got {g}");
        max_abs = max_abs.max((g as f64 - w as f64).abs());
        scale = scale.max((w as f64).abs());
    }
    (max_abs, max_abs / scale)
}

type Op = fn(&Array, &Array, &Array, &[u32], LogprobOptions) -> Result<ChosenLogprob>;

/// The chunk kernels and the reference built from MLX's operations.
const PATHS: [(&str, Op); 2] = [
    ("kernels", chosen_logprob),
    ("reference", chosen_logprob_reference),
];

/// The operation on `case` in `dtype`: the log-probabilities, the entropies and the gradient of
/// `sum_i upstream[i] logprob[i]` for `hidden`, as f32.
fn run(
    case: &Case,
    dtype: Dtype,
    masked: &[u32],
    options: LogprobOptions,
) -> Result<[Vec<f32>; 3]> {
    run_on(chosen_logprob, case, dtype, masked, options)
}

fn run_on(
    op: Op,
    case: &Case,
    dtype: Dtype,
    masked: &[u32],
    options: LogprobOptions,
) -> Result<[Vec<f32>; 3]> {
    let hidden = Array::from_slice(&case.hidden, &[case.n, case.d]).as_dtype(dtype)?;
    let weight = Array::from_slice(&case.weight, &[case.v, case.d]).as_dtype(dtype)?;
    let targets = Array::from_slice(&case.targets, &[case.n]);
    let upstream = Array::from_slice(&case.upstream, &[case.n]);
    let mut entropy = None;
    let f = |a: &[Array]| -> Result<Vec<Array>> {
        let out = op(&a[0], &weight, &targets, masked, options)?;
        entropy = Some(out.entropy);
        Ok(vec![out.logprob])
    };
    let (outputs, grads) = fallible_vjp(f, &[hidden], &[upstream])?;
    assert_eq!(outputs[0].dtype(), Dtype::Float32);
    assert_eq!(
        grads[0].dtype(),
        dtype,
        "the gradient comes back in hidden's dtype"
    );
    let entropy = entropy.expect("ran");
    assert_eq!(entropy.dtype(), Dtype::Float32);
    Ok([values(&outputs[0]), values(&entropy), values(&grads[0])])
}

fn check(
    label: &str,
    got: &[Vec<f32>; 3],
    want: &[Vec<f32>; 3],
    bounds: [f64; 3],
    worst: &mut [f64; 3],
) {
    for (i, what) in ["logprob", "entropy", "grad_hidden"].iter().enumerate() {
        let (abs, rel) = error(what, &got[i], &want[i]);
        println!(
            "{label:32} {what:12} max abs {abs:9.3e}  relative {}",
            up3(rel)
        );
        assert!(
            rel <= bounds[i],
            "{label} {what}: relative error {rel:e} above {:e} (max abs {abs:e})",
            bounds[i]
        );
        worst[i] = worst[i].max(rel);
    }
}

#[test]
fn matches_pytorch() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    for ((path, op), (dtype, bounds)) in PATHS
        .into_iter()
        .flat_map(|p| [(p, (Dtype::Float32, F32)), (p, (Dtype::Bfloat16, BF16))])
    {
        let mut worst = [0.0; 3];
        for name in CASES {
            let case = Case::load(name);
            for (options, chunks) in [
                (SMALL_CHUNKS, "small chunks"),
                (LogprobOptions::default(), "defaults"),
            ] {
                for masked in [true, false] {
                    let ids: &[u32] = if masked { &case.masked } else { &[] };
                    let label = format!(
                        "{path} {dtype:?} {name} {chunks}{}",
                        if masked { ", masked" } else { "" }
                    );
                    let (forward, backward) = (probe::forward_passes(), probe::backward_passes());
                    let got = run_on(op, &case, dtype, ids, options)?;
                    // One pass over the vocabulary each way per block of rows.
                    let n = case.n as usize;
                    let blocks = n.div_ceil(options.row_chunk);
                    assert_eq!(probe::forward_passes() - forward, blocks, "{label}");
                    assert_eq!(probe::backward_passes() - backward, blocks, "{label}");
                    check(&label, &got, &case.reference(masked), bounds, &mut worst);
                }
            }
        }
        println!(
            "{path} {dtype:?}: largest logprob {}, entropy {}, grad_hidden {}",
            up3(worst[0]),
            up3(worst[1]),
            up3(worst[2])
        );
    }
    Ok(())
}

/// A forward pass alone walks the vocabulary once per block of rows and never for the gradient.
#[test]
fn a_pass_without_a_gradient_takes_one_walk() -> Result<()> {
    let _serial = serial();
    let case = Case::load("basic");
    let hidden = Array::from_slice(&case.hidden, &[case.n, case.d]);
    let weight = Array::from_slice(&case.weight, &[case.v, case.d]);
    let targets = Array::from_slice(&case.targets, &[case.n]);
    let (forward, backward) = (probe::forward_passes(), probe::backward_passes());
    let out = chosen_logprob(&hidden, &weight, &targets, &case.masked, SMALL_CHUNKS)?;
    mlx_rs::transforms::eval([&out.logprob, &out.entropy])?;
    assert_eq!(probe::forward_passes() - forward, 4);
    assert_eq!(probe::backward_passes() - backward, 0);
    Ok(())
}

/// The contract's errors: shapes, masked ids, every token masked, a zero row chunk.
#[test]
fn refuses_what_the_contract_rules_out() {
    let _serial = serial();
    let hidden = Array::from_slice(&[0.0f32; 8], &[2, 4]);
    let weight = Array::from_slice(&[0.0f32; 12], &[3, 4]);
    let targets = Array::from_slice(&[0i64, 1], &[2]);
    let options = LogprobOptions::default();
    let narrow = Array::from_slice(&[0.0f32; 6], &[3, 2]);
    assert!(chosen_logprob(&hidden, &narrow, &targets, &[], options).is_err());
    assert!(chosen_logprob(&hidden, &weight, &targets, &[3], options).is_err());
    assert!(chosen_logprob(&hidden, &weight, &targets, &[0, 1, 2], options).is_err());
    let zero = LogprobOptions {
        row_chunk: 0,
        ..options
    };
    assert!(chosen_logprob(&hidden, &weight, &targets, &[], zero).is_err());
    let floats = Array::from_slice(&[0.0f32, 1.0], &[2]);
    assert!(chosen_logprob(&hidden, &weight, &floats, &[], options).is_err());
}

/// By default a run repeats bit for bit: the forward and backward passes, run twice, give the
/// same bits, in f32 and bf16, with masked ids and several chunks and row blocks.
#[test]
fn repeats_bit_for_bit() -> Result<()> {
    let _serial = serial();
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let case = Case::load("peaked");
        let first = run(&case, dtype, &case.masked, SMALL_CHUNKS)?;
        let second = run(&case, dtype, &case.masked, SMALL_CHUNKS)?;
        let bits = |x: &[Vec<f32>; 3]| {
            x.clone()
                .map(|v| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>())
        };
        assert!(bits(&first) == bits(&second), "{dtype:?}: two runs differ");
    }
    Ok(())
}

/// The kernels against the reference at a size the fixtures do not reach: D 1024, 20,000 tokens in
/// chunks of 3072 (the last partial), two blocks of rows, masked ids including a row's target and
/// a run of tokens across a chunk's edge, targets out of range; f32.
#[test]
fn kernels_match_the_reference_at_scale() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    let (n, d, v) = (700, 1024, 20_000);
    let normal = |shape: &[i32], seed: u64| -> Result<Array> {
        mlx_rs::random::normal::<f32>(shape, None, None, &mlx_rs::random::key(seed)?)
    };
    let hidden = normal(&[n, d], 7)?;
    let weight = normal(&[v, d], 8)?.multiply(Array::from_f32(0.05))?;
    let upstream = normal(&[n], 9)?;
    let mut ids: Vec<i64> = (0..n as i64).map(|i| (i * 7919) % v as i64).collect();
    ids[3] = -1;
    ids[4] = v as i64;
    let masked = vec![ids[5] as u32, 0, 3070, 3071, 3072, 3073, (v - 1) as u32];
    let targets = Array::from_slice(&ids, &[n]);
    let options = LogprobOptions {
        vocab_chunk: 0,
        row_chunk: 400,
    };
    let run = |op: Op| -> Result<[Vec<f32>; 3]> {
        let mut entropy = None;
        let f = |a: &[Array]| -> Result<Vec<Array>> {
            let out = op(&a[0], &weight, &targets, &masked, options)?;
            entropy = Some(out.entropy);
            Ok(vec![out.logprob])
        };
        let (outputs, grads) = fallible_vjp(
            f,
            std::slice::from_ref(&hidden),
            std::slice::from_ref(&upstream),
        )?;
        Ok([
            values(&outputs[0]),
            values(&entropy.expect("ran")),
            values(&grads[0]),
        ])
    };
    let kernels = run(chosen_logprob)?;
    let reference = run(chosen_logprob_reference)?;
    for (i, what) in ["logprob", "entropy", "grad_hidden"].iter().enumerate() {
        let (abs, rel) = error(what, &kernels[i], &reference[i]);
        println!(
            "kernels against the reference {what:12} max abs {abs:9.3e}  relative {}",
            up3(rel)
        );
        assert!(rel <= 1e-6, "{what}: {rel:e}");
    }
    for row in [3, 4, 5] {
        assert_eq!(kernels[0][row], f32::NEG_INFINITY, "row {row}");
    }
    Ok(())
}
