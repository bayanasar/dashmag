//! The fused linear cross-entropy against dash-ops' fixtures, with dash-ops' bounds.
//!
//! `fixtures/loss/cases.safetensors` (dash-ops') holds three cases, their inputs as f32 values,
//! and for two input precisions (the values as they are, and rounded to bf16) PyTorch's float64
//! results stored as f32: the loss, each row's log-probability and entropy, and the gradients for
//! `hidden` and `weight` of `loss + sum_i g_lp[i] logprob[i] + sum_i g_ent[i] entropy[i]` (with
//! `sum_i g_loss[i] loss[i]` for the loss when it is not reduced). Four variants of the options:
//! `mean` is torch's default; `smooth` takes label smoothing, a z-loss and masked ids, and records
//! each term's gradient alone too; `sum` also masks the last counted row's target (its loss is
//! `+inf`); `none` keeps every row's loss and ignores another id than `-100`, which makes the
//! `-100` targets out of range (`+inf`). `peaked` has logits up to 38 and rows whose target has
//! p = 0.98, `wide` D = 256.
//!
//! As dash-ops' tests: the vocabulary in chunks of 64 tokens (the last partial in every case) and
//! the rows in blocks of 10 or fewer; both inputs taking a gradient, each term alone, a frozen
//! weight with and without the entropy, a frozen `hidden`, no gradient at all, other chunkings,
//! f16 inputs against f32 ones of the same values; the kernels and the reference built from MLX's
//! operations, each held to the fixtures in f32 and bf16.
//!
//! Errors are `max |got - want|` and that over `max |want|`, per output, printed to three figures
//! and rounded up. f32 needs `MLX_ENABLE_TF32=0`:
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test loss -- --nocapture
#![cfg(target_os = "macos")]

use dash_mlx::logprob::{LogprobOptions, chosen_logprob};
use dash_mlx::loss::{
    CrossEntropy, CrossEntropyOptions, Reduction, linear_cross_entropy,
    linear_cross_entropy_reference, probe,
};
use mlx_rs::error::Result;
use mlx_rs::ops::{ones, zeros};
use mlx_rs::transforms::{eval, fallible_vjp};
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
    "/../dash-ops/tests/fixtures/loss"
);
const CASES: [&str; 3] = ["basic", "peaked", "wide"];

/// dash-ops' small chunks: every case crosses several, the last partial, and the row blocks
/// split the rows unevenly.
const SMALL_CHUNKS: LogprobOptions = LogprobOptions {
    vocab_chunk: 64,
    row_chunk: 10,
};

/// dash-ops' bounds for its f32 reference path, relative, `[loss, logprob, entropy,
/// grad_hidden, grad_weight]`: f32 throughout, at f32's noise floor for `peaked`.
const F32: Tolerance = [2e-7, 2e-7, 1e-6, 4e-6, 4e-6];
/// From bf16 inputs, computed in f32, the gradients rounded to bf16 on the way out (half an ulp
/// is 2^-9 of the value, at most 2^-8 relative to a power of two).
const BF16: Tolerance = [2e-7, 2e-7, 5e-7, 3e-3, 4e-3];
/// f16 inputs against f32 inputs of the same values, both computed in f32: the gradients'
/// rounding to f16 on the way out.
const F32_VS_F16: Tolerance = [2e-7, 2e-7, 1e-6, 1e-3, 1e-3];

type Tolerance = [f64; 5];

type Op = fn(&Array, &Array, &Array, &CrossEntropyOptions) -> Result<CrossEntropy>;

/// The kernels and the reference built from MLX's operations.
const PATHS: [(&str, Op); 2] = [
    ("kernels", linear_cross_entropy),
    ("reference", linear_cross_entropy_reference),
];

/// MLX runs f32 products in TF32 unless told not to; the f32 bounds assume IEEE f32.
fn require_ieee_f32_matmul() {
    assert_eq!(
        std::env::var("MLX_ENABLE_TF32").as_deref(),
        Ok("0"),
        "run with MLX_ENABLE_TF32=0: MLX's f32 matmuls run in TF32 otherwise, about 1e-3 off"
    );
}

/// The values the inputs take: the fixtures' f32 values, or those rounded to bf16.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Precision {
    F32,
    Bf16,
}

impl Precision {
    fn key(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::Bf16 => "bf16",
        }
    }

    fn round(self, values: &[f32]) -> Vec<f32> {
        match self {
            Precision::F32 => values.to_vec(),
            Precision::Bf16 => values
                .iter()
                .map(|&x| half::bf16::from_f32(x).to_f32())
                .collect(),
        }
    }
}

struct Case {
    name: &'static str,
    n: i32,
    v: i32,
    d: i32,
    hidden: Vec<f32>,
    weight: Vec<f32>,
    targets: Vec<i64>,
    g_loss: Vec<f32>,
    g_lp: Vec<f32>,
    g_ent: Vec<f32>,
    /// The variants the case has, with their options.
    variants: Vec<(String, CrossEntropyOptions)>,
    tensors: Vec<u8>,
}

/// What a run returns, or the reference, all as f32; an empty field was not computed.
#[derive(Clone, Debug)]
struct Outputs {
    loss: Vec<f32>,
    logprob: Vec<f32>,
    entropy: Vec<f32>,
    grad_hidden: Vec<f32>,
    grad_weight: Vec<f32>,
}

fn f32s(bytes: &[u8], key: &str) -> Vec<f32> {
    let tensors = SafeTensors::deserialize(bytes).unwrap();
    let view = tensors.tensor(key).unwrap_or_else(|_| panic!("{key}"));
    assert_eq!(view.dtype(), safetensors::Dtype::F32, "{key}");
    view.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
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
        let targets = {
            let file = SafeTensors::deserialize(&tensors).unwrap();
            let view = file.tensor(&format!("{name}.targets")).unwrap();
            assert_eq!(view.dtype(), safetensors::Dtype::I64);
            view.data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect()
        };
        let variants = meta["variants"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(variant, entry)| {
                let options = CrossEntropyOptions {
                    reduction: match entry["reduction"].as_str().unwrap() {
                        "mean" => Reduction::Mean,
                        "sum" => Reduction::Sum,
                        _ => Reduction::None,
                    },
                    ignore_index: entry["ignore_index"].as_i64().unwrap(),
                    label_smoothing: entry["eps"].as_f64().unwrap() as f32,
                    z_loss: entry["lam"].as_f64().unwrap() as f32,
                    masked: entry["masked"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|id| id.as_u64().unwrap() as u32)
                        .collect(),
                    entropy: true,
                    chunks: SMALL_CHUNKS,
                };
                (variant.clone(), options)
            })
            .collect();
        let key = |k: &str| f32s(&tensors, &format!("{name}.{k}"));
        Self {
            name,
            n: dim("n"),
            v: dim("v"),
            d: dim("d"),
            hidden: key("hidden"),
            weight: key("weight"),
            targets,
            g_loss: key("g_loss"),
            g_lp: key("g_lp"),
            g_ent: key("g_ent"),
            variants,
            tensors,
        }
    }

    fn options(&self, variant: &str) -> CrossEntropyOptions {
        self.variants
            .iter()
            .find(|(name, _)| name == variant)
            .map(|(_, options)| options.clone())
            .unwrap_or_else(|| panic!("{} has no variant {variant}", self.name))
    }

    /// The reference for `precision` and `variant`; with `term`, the gradients of that term of
    /// the objective alone (`loss`, `lp` or `ent`, recorded for `smooth`).
    fn want(&self, precision: Precision, variant: &str, term: Option<&str>) -> Outputs {
        let key = |k: &str| {
            f32s(
                &self.tensors,
                &format!("{}.{}.{variant}.{k}", self.name, precision.key()),
            )
        };
        let grad = |k: &str| match term {
            Some(term) => key(&format!("{term}_only.{k}")),
            None => key(k),
        };
        Outputs {
            loss: key("loss"),
            logprob: key("logprob"),
            entropy: key("entropy"),
            grad_hidden: grad("grad_hidden"),
            grad_weight: grad("grad_weight"),
        }
    }
}

/// Which inputs take a gradient, and which terms the objective sums.
#[derive(Clone, Copy, Debug)]
struct Setup {
    hidden: bool,
    weight: bool,
    loss: bool,
    logprob: bool,
    entropy: bool,
}

const ALL: Setup = Setup {
    hidden: true,
    weight: true,
    loss: true,
    logprob: true,
    entropy: true,
};

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

/// The operation on `case`, inputs at `precision` cast to `dtype`, and the gradients of the
/// objective the generator used for the inputs `setup` names, by a VJP over the outputs.
fn run(
    op: Op,
    case: &Case,
    precision: Precision,
    dtype: Dtype,
    options: &CrossEntropyOptions,
    setup: Setup,
) -> Result<Outputs> {
    let (n, v, d) = (case.n, case.v, case.d);
    let hidden = Array::from_slice(&precision.round(&case.hidden), &[n, d]).as_dtype(dtype)?;
    let weight = Array::from_slice(&precision.round(&case.weight), &[v, d]).as_dtype(dtype)?;
    let targets = Array::from_slice(&case.targets, &[n]);
    let check_dtypes = |out: &CrossEntropy| {
        for x in [Some(&out.loss), Some(&out.logprob), out.entropy.as_ref()]
            .into_iter()
            .flatten()
        {
            assert_eq!(x.dtype(), Dtype::Float32, "the outputs are f32");
        }
    };
    if !(setup.hidden || setup.weight) {
        let out = op(&hidden, &weight, &targets, options)?;
        check_dtypes(&out);
        return Ok(Outputs {
            loss: values(&out.loss),
            logprob: values(&out.logprob),
            entropy: out.entropy.as_ref().map(values).unwrap_or_default(),
            grad_hidden: vec![],
            grad_weight: vec![],
        });
    }
    let mut primals = Vec::new();
    if setup.hidden {
        primals.push(hidden.clone());
    }
    if setup.weight {
        primals.push(weight.clone());
    }
    let f = |p: &[Array]| -> Result<Vec<Array>> {
        let mut p = p.iter();
        let h = match setup.hidden {
            true => p.next().unwrap(),
            false => &hidden,
        };
        let w = match setup.weight {
            true => p.next().unwrap(),
            false => &weight,
        };
        let out = op(h, w, &targets, options)?;
        check_dtypes(&out);
        let mut outputs = vec![out.loss, out.logprob];
        outputs.extend(out.entropy);
        Ok(outputs)
    };
    let upstream = |on: bool, g: &[f32]| -> Result<Array> {
        match on {
            true => Ok(Array::from_slice(g, &[n])),
            false => zeros::<f32>(&[n]),
        }
    };
    let mut cotangents = vec![
        match (setup.loss, options.reduction) {
            (true, Reduction::None) => Array::from_slice(&case.g_loss, &[n]),
            (true, _) => ones::<f32>(&[1])?,
            (false, Reduction::None) => zeros::<f32>(&[n])?,
            (false, _) => zeros::<f32>(&[1])?,
        },
        upstream(setup.logprob, &case.g_lp)?,
    ];
    if options.entropy {
        cotangents.push(upstream(setup.entropy, &case.g_ent)?);
    }
    let (outputs, grads) = fallible_vjp(f, &primals, &cotangents)?;
    let mut grads = grads.into_iter();
    let mut grad = |on: bool| -> Vec<f32> {
        match on {
            true => {
                let g = grads.next().unwrap();
                assert_eq!(
                    g.dtype(),
                    dtype,
                    "a gradient comes back in its input's dtype"
                );
                values(&g)
            }
            false => vec![],
        }
    };
    let grad_hidden = grad(setup.hidden);
    let grad_weight = grad(setup.weight);
    Ok(Outputs {
        loss: values(&outputs[0]),
        logprob: values(&outputs[1]),
        entropy: outputs.get(2).map(values).unwrap_or_default(),
        grad_hidden,
        grad_weight,
    })
}

/// `(max |got - want|, that / max |want|)` over the finite entries; an infinity must match.
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
    (
        max_abs,
        if scale > 0.0 {
            max_abs / scale
        } else {
            max_abs
        },
    )
}

/// Asserts each output's relative error is within its bound and prints it; the largest of each
/// output goes into `worst`. An empty `got` is not checked (not computed in that run).
fn check(label: &str, got: &Outputs, want: &Outputs, tolerance: Tolerance, worst: &mut [f64; 5]) {
    for (i, (what, got, want)) in [
        ("loss", &got.loss, &want.loss),
        ("logprob", &got.logprob, &want.logprob),
        ("entropy", &got.entropy, &want.entropy),
        ("grad_hidden", &got.grad_hidden, &want.grad_hidden),
        ("grad_weight", &got.grad_weight, &want.grad_weight),
    ]
    .into_iter()
    .enumerate()
    {
        if got.is_empty() {
            continue;
        }
        let (abs, rel) = error(what, got, want);
        println!(
            "{label:40} {what:12} max abs {abs:9.3e}  relative {}",
            up3(rel)
        );
        assert!(
            rel <= tolerance[i],
            "{label} {what}: relative error {rel:e} above {:e} (max abs {abs:e})",
            tolerance[i]
        );
        worst[i] = worst[i].max(rel);
    }
}

fn report(label: &str, worst: &[f64; 5]) {
    println!(
        "{label}: largest loss {}, logprob {}, entropy {}, grad_hidden {}, grad_weight {}",
        up3(worst[0]),
        up3(worst[1]),
        up3(worst[2]),
        up3(worst[3]),
        up3(worst[4])
    );
}

/// The sum of two references' gradients (their outputs are the same).
fn sum_of(a: Outputs, b: &Outputs) -> Outputs {
    let add = |x: Vec<f32>, y: &[f32]| x.iter().zip(y).map(|(x, y)| x + y).collect();
    Outputs {
        grad_hidden: add(a.grad_hidden, &b.grad_hidden),
        grad_weight: add(a.grad_weight, &b.grad_weight),
        ..a
    }
}

/// Each path, f32 and bf16, the bounds of each.
fn paths() -> impl Iterator<Item = (&'static str, Op, Precision, Dtype, Tolerance)> {
    PATHS.into_iter().flat_map(|(path, op)| {
        [
            (path, op, Precision::F32, Dtype::Float32, F32),
            (path, op, Precision::Bf16, Dtype::Bfloat16, BF16),
        ]
    })
}

/// Every case and variant, both inputs taking a gradient: one pass over the vocabulary each way
/// per block of rows.
#[test]
fn matches_pytorch() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    for (path, op, precision, dtype, tolerance) in paths() {
        let mut worst = [0.0; 5];
        for name in CASES {
            let case = Case::load(name);
            for (variant, options) in &case.variants {
                let (forward, gradient) = (probe::forward_passes(), probe::gradient_passes());
                let got = run(op, &case, precision, dtype, options, ALL)?;
                let blocks = (case.n as usize).div_ceil(options.chunks.row_chunk);
                assert_eq!(probe::forward_passes() - forward, blocks);
                assert_eq!(probe::gradient_passes() - gradient, blocks);
                check(
                    &format!("{path} {dtype:?} {name} {variant}"),
                    &got,
                    &case.want(precision, variant, None),
                    tolerance,
                    &mut worst,
                );
            }
        }
        report(&format!("{path} {dtype:?}"), &worst);
    }
    Ok(())
}

/// Each term of the objective alone: the loss's, the log-probability's and the entropy's own
/// gradients, for both inputs.
#[test]
fn each_output_has_its_gradient() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    for (path, op, precision, dtype, tolerance) in paths() {
        let mut worst = [0.0; 5];
        for name in ["basic", "peaked"] {
            let case = Case::load(name);
            let options = case.options("smooth");
            for (term, setup) in [
                (
                    "loss",
                    Setup {
                        logprob: false,
                        entropy: false,
                        ..ALL
                    },
                ),
                (
                    "lp",
                    Setup {
                        loss: false,
                        entropy: false,
                        ..ALL
                    },
                ),
                (
                    "ent",
                    Setup {
                        loss: false,
                        logprob: false,
                        ..ALL
                    },
                ),
            ] {
                let got = run(op, &case, precision, dtype, &options, setup)?;
                check(
                    &format!("{path} {dtype:?} {name} {term} alone"),
                    &got,
                    &case.want(precision, "smooth", Some(term)),
                    tolerance,
                    &mut worst,
                );
            }
        }
        report(&format!("{path} {dtype:?} each term"), &worst);
    }
    Ok(())
}

/// A weight that takes no gradient, with and without the entropy: the gradient of `hidden`
/// alone, the pass over the vocabulary the same.
#[test]
fn frozen_weight() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    for (path, op, precision, dtype, tolerance) in paths() {
        let mut worst = [0.0; 5];
        for name in CASES {
            let case = Case::load(name);
            // Without the entropy: the loss and the log-probability's terms.
            let mut options = case.options("smooth");
            options.entropy = false;
            let setup = Setup {
                weight: false,
                entropy: false,
                ..ALL
            };
            let got = run(op, &case, precision, dtype, &options, setup)?;
            assert!(got.entropy.is_empty(), "no entropy unless asked for");
            // `wide` records no term alone: its outputs are checked below.
            if name != "wide" {
                let want = sum_of(
                    case.want(precision, "smooth", Some("loss")),
                    &case.want(precision, "smooth", Some("lp")),
                );
                check(
                    &format!("{path} {dtype:?} {name} frozen, no entropy"),
                    &got,
                    &Outputs {
                        entropy: vec![],
                        ..want
                    },
                    tolerance,
                    &mut worst,
                );
            }
            // With the entropy, every variant.
            for (variant, options) in &case.variants {
                let setup = Setup {
                    weight: false,
                    ..ALL
                };
                let got = run(op, &case, precision, dtype, options, setup)?;
                assert!(got.grad_weight.is_empty());
                check(
                    &format!("{path} {dtype:?} {name} {variant} frozen"),
                    &got,
                    &case.want(precision, variant, None),
                    tolerance,
                    &mut worst,
                );
            }
        }
        report(&format!("{path} {dtype:?} frozen weight"), &worst);
    }
    Ok(())
}

/// `hidden` that takes no gradient, a weight that does: a linear probe on frozen features.
#[test]
fn hidden_frozen() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    for (path, op, precision, dtype, tolerance) in paths() {
        let mut worst = [0.0; 5];
        let case = Case::load("basic");
        for (variant, options) in &case.variants {
            let setup = Setup {
                hidden: false,
                ..ALL
            };
            let got = run(op, &case, precision, dtype, options, setup)?;
            assert!(got.grad_hidden.is_empty());
            check(
                &format!("{path} {dtype:?} basic {variant} hidden frozen"),
                &got,
                &case.want(precision, variant, None),
                tolerance,
                &mut worst,
            );
        }
        report(&format!("{path} {dtype:?} hidden frozen"), &worst);
    }
    Ok(())
}

/// The chunking is an implementation detail: one chunk, tiny chunks, uneven row blocks and the
/// defaults agree with the fixtures.
#[test]
fn chunking_does_not_change_the_answer() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    let case = Case::load("peaked");
    let mut worst = [0.0; 5];
    for chunks in [
        LogprobOptions {
            vocab_chunk: case.v as usize,
            row_chunk: case.n as usize,
        },
        LogprobOptions {
            vocab_chunk: 7,
            row_chunk: 5,
        },
        LogprobOptions {
            vocab_chunk: 1000,
            row_chunk: 1,
        },
        LogprobOptions::default(),
    ] {
        for variant in ["smooth", "none"] {
            let options = CrossEntropyOptions {
                chunks,
                ..case.options(variant)
            };
            for (path, op) in PATHS {
                let got = run(op, &case, Precision::F32, Dtype::Float32, &options, ALL)?;
                check(
                    &format!("{path} {variant} {chunks:?}"),
                    &got,
                    &case.want(Precision::F32, variant, None),
                    F32,
                    &mut worst,
                );
            }
        }
    }
    report("chunkings", &worst);
    Ok(())
}

/// Without a gradient the same values come out, and the vocabulary is walked once per block.
#[test]
fn a_pass_without_a_gradient_takes_one_walk() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    let case = Case::load("basic");
    let mut worst = [0.0; 5];
    for (variant, options) in &case.variants {
        let setup = Setup {
            hidden: false,
            weight: false,
            ..ALL
        };
        let (forward, gradient) = (probe::forward_passes(), probe::gradient_passes());
        let got = run(
            linear_cross_entropy,
            &case,
            Precision::F32,
            Dtype::Float32,
            options,
            setup,
        )?;
        assert_eq!(probe::forward_passes() - forward, 4);
        assert_eq!(probe::gradient_passes() - gradient, 0);
        check(
            &format!("untracked {variant}"),
            &got,
            &case.want(Precision::F32, variant, None),
            F32,
            &mut worst,
        );
    }
    Ok(())
}

/// The loss's log-probability is `chosen_logprob`'s, its gradient too (frozen weight), and
/// without smoothing or z-loss the loss is minus the mean log-probability over counted rows.
#[test]
fn agrees_with_chosen_logprob() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    let case = Case::load("basic");
    let (n, v, d) = (case.n, case.v, case.d);
    let masked = case.options("sum").masked;
    let hidden = Array::from_slice(&Precision::Bf16.round(&case.hidden), &[n, d]);
    let weight = Array::from_slice(&Precision::Bf16.round(&case.weight), &[v, d]);
    let targets = Array::from_slice(&case.targets, &[n]);
    let g = Array::from_slice(&case.g_lp, &[n]);
    // Rows ignored by the loss give it 0; chosen_logprob has -inf there (out of range).
    let counted: Vec<f32> = case
        .targets
        .iter()
        .map(|&t| if t == -100 { 0.0 } else { 1.0 })
        .collect();
    let counted = Array::from_slice(&counted, &[n]);

    let options = CrossEntropyOptions {
        masked: masked.clone(),
        chunks: SMALL_CHUNKS,
        ..Default::default()
    };
    let mut loss = None;
    let f = |p: &[Array]| -> Result<Vec<Array>> {
        let out = linear_cross_entropy(&p[0], &weight, &targets, &options)?;
        loss = Some(out.loss);
        Ok(vec![out.logprob])
    };
    let (out, grad) = fallible_vjp(f, std::slice::from_ref(&hidden), std::slice::from_ref(&g))?;
    let (loss_lp, loss_grad) = (values(&out[0]), values(&grad[0]));
    let loss_value = values(&loss.expect("ran"))[0];

    let g = g.multiply(&counted)?;
    let f = |p: &[Array]| -> Result<Vec<Array>> {
        Ok(vec![
            chosen_logprob(&p[0], &weight, &targets, &masked, SMALL_CHUNKS)?.logprob,
        ])
    };
    let (out, grad) = fallible_vjp(f, std::slice::from_ref(&hidden), std::slice::from_ref(&g))?;
    let chosen_grad = values(&grad[0]);
    let chosen_lp: Vec<f32> = values(&out[0])
        .into_iter()
        .zip(&case.targets)
        .map(|(lp, &t)| if t == -100 { 0.0 } else { lp })
        .collect();
    let (_, rel) = error("logprob", &loss_lp, &chosen_lp);
    assert!(rel < 1e-6, "logprob: {rel:e}");
    let (_, rel) = error("grad", &loss_grad, &chosen_grad);
    assert!(rel < 1e-6, "grad: {rel:e}");
    // The masked target makes the mean +inf; it is -inf in both log-probabilities.
    assert_eq!(loss_value, f32::INFINITY);
    assert_eq!(loss_lp[n as usize - 1], f32::NEG_INFINITY);
    Ok(())
}

/// No rows: empty outputs, a sum of 0, zero gradients of the inputs' shapes.
#[test]
fn empty_batch() -> Result<()> {
    let _serial = serial();
    let hidden = zeros::<f32>(&[0, 3])?;
    let weight = ones::<f32>(&[5, 3])?;
    let targets = zeros::<i64>(&[0])?;
    let options = CrossEntropyOptions {
        reduction: Reduction::Sum,
        entropy: true,
        ..Default::default()
    };
    let mut shapes = None;
    let f = |p: &[Array]| -> Result<Vec<Array>> {
        let out = linear_cross_entropy(&p[0], &p[1], &targets, &options)?;
        shapes = Some((
            out.logprob.shape().to_vec(),
            out.entropy.as_ref().map(|e| e.shape().to_vec()),
        ));
        Ok(vec![out.loss])
    };
    let (out, grads) = fallible_vjp(f, &[hidden, weight], &[ones::<f32>(&[1])?])?;
    assert_eq!(shapes, Some((vec![0], Some(vec![0]))));
    assert_eq!(values(&out[0]), vec![0.0]);
    assert_eq!(grads[0].shape(), [0, 3]);
    assert_eq!(grads[1].shape(), [5, 3]);
    assert!(values(&grads[1]).iter().all(|&g| g == 0.0));
    Ok(())
}

/// Every row ignored: the mean is NaN (torch's 0 / 0), and every gradient 0 although the mean's
/// own gradient is not finite: an ignored row receives none, whatever its upstream holds.
#[test]
fn all_ignored() -> Result<()> {
    let _serial = serial();
    let case = Case::load("basic");
    let (n, v, d) = (case.n, case.v, case.d);
    let hidden = Array::from_slice(&case.hidden, &[n, d]);
    let weight = Array::from_slice(&case.weight, &[v, d]);
    let targets = Array::from_slice(&vec![-100i64; n as usize], &[n]);
    let options = CrossEntropyOptions {
        label_smoothing: 0.1,
        z_loss: 1e-2,
        chunks: SMALL_CHUNKS,
        ..Default::default()
    };
    let f = |p: &[Array]| -> Result<Vec<Array>> {
        let out = linear_cross_entropy(&p[0], &p[1], &targets, &options)?;
        Ok(vec![out.loss, out.logprob])
    };
    // The loss's cotangent is 0, but the mean's division by zero counted rows still makes its
    // gradient non-finite before the rows are reached.
    let (out, grads) = fallible_vjp(
        f,
        &[hidden, weight],
        &[zeros::<f32>(&[1])?, ones::<f32>(&[n])?],
    )?;
    assert!(values(&out[0])[0].is_nan());
    assert!(values(&out[1]).iter().all(|&x| x == 0.0));
    for grad in &grads {
        assert!(values(grad).iter().all(|&g| g == 0.0));
    }
    Ok(())
}

/// f16 inputs are computed in f32: the same as f32 inputs of the same values, but for the
/// gradients' rounding to f16 on the way out.
#[test]
fn f16_inputs() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    let mut case = Case::load("basic");
    let f16 =
        |x: &[f32]| -> Vec<f32> { x.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect() };
    case.hidden = f16(&case.hidden);
    case.weight = f16(&case.weight);
    let options = case.options("smooth");
    for (path, op) in PATHS {
        let want = run(op, &case, Precision::F32, Dtype::Float32, &options, ALL)?;
        let got = run(op, &case, Precision::F32, Dtype::Float16, &options, ALL)?;
        let want = Outputs {
            grad_hidden: f16(&want.grad_hidden),
            grad_weight: f16(&want.grad_weight),
            ..want
        };
        let mut worst = [0.0; 5];
        check(
            &format!("{path} f16 against f32"),
            &got,
            &want,
            F32_VS_F16,
            &mut worst,
        );
    }
    Ok(())
}

/// The contract's errors: dash-ops panics on each.
#[test]
fn refuses_what_the_contract_rules_out() {
    let _serial = serial();
    let hidden = Array::from_slice(&[0.0f32; 6], &[2, 3]);
    let weight = Array::from_slice(&[0.0f32; 15], &[5, 3]);
    let targets = Array::from_slice(&[0i64, 1], &[2]);
    let run = |hidden: &Array, weight: &Array, options: CrossEntropyOptions| {
        linear_cross_entropy(hidden, weight, &targets, &options).map(|_| ())
    };
    let refused = |result: Result<()>, what: &str| {
        let message = result.expect_err(what).what().to_string();
        assert!(message.contains(what), "{what}: {message}");
    };
    refused(
        run(
            &hidden,
            &weight,
            CrossEntropyOptions {
                label_smoothing: 1.5,
                ..Default::default()
            },
        ),
        "label_smoothing",
    );
    refused(
        run(
            &hidden,
            &weight,
            CrossEntropyOptions {
                z_loss: -1.0,
                ..Default::default()
            },
        ),
        "z_loss",
    );
    refused(
        run(
            &hidden,
            &weight,
            CrossEntropyOptions {
                z_loss: f32::INFINITY,
                ..Default::default()
            },
        ),
        "z_loss",
    );
    refused(
        run(
            &hidden,
            &weight,
            CrossEntropyOptions {
                masked: vec![1, 5],
                ..Default::default()
            },
        ),
        "outside the vocabulary",
    );
    refused(
        run(
            &hidden,
            &weight,
            CrossEntropyOptions {
                masked: vec![0, 1, 2, 3, 4],
                ..Default::default()
            },
        ),
        "every token",
    );
    refused(
        run(
            &hidden,
            &weight,
            CrossEntropyOptions {
                chunks: LogprobOptions {
                    vocab_chunk: 0,
                    row_chunk: 0,
                },
                ..Default::default()
            },
        ),
        "row_chunk",
    );
    let empty = zeros::<f32>(&[0, 3]).unwrap();
    refused(
        run(&hidden, &empty, CrossEntropyOptions::default()),
        "vocabulary is empty",
    );
    let narrow = Array::from_slice(&[0.0f32; 10], &[5, 2]);
    refused(
        run(&hidden, &narrow, CrossEntropyOptions::default()),
        "weight must be [V, D]",
    );
}

/// By default a run repeats bit for bit: the loss, its outputs and both gradients, run twice, in
/// f32 and bf16, with masked ids, label smoothing, a z-loss, the entropy, several chunks and row
/// blocks.
#[test]
fn repeats_bit_for_bit() -> Result<()> {
    let _serial = serial();
    let case = Case::load("peaked");
    let options = case.options("smooth");
    for (precision, dtype) in [
        (Precision::F32, Dtype::Float32),
        (Precision::Bf16, Dtype::Bfloat16),
    ] {
        let once = || -> Result<Vec<Vec<u32>>> {
            let out = run(linear_cross_entropy, &case, precision, dtype, &options, ALL)?;
            Ok([
                out.loss,
                out.logprob,
                out.entropy,
                out.grad_hidden,
                out.grad_weight,
            ]
            .map(|v| v.iter().map(|f| f.to_bits()).collect())
            .to_vec())
        };
        assert!(once()? == once()?, "{dtype:?}: two runs differ");
    }
    Ok(())
}

/// The kernels against the reference at a size the fixtures do not reach: D 1024, 20,000 tokens
/// in chunks of 3072 (the last partial), two blocks of rows, masked ids including a row's target
/// and a run of tokens across a chunk's edge, targets out of range and an ignored row, label
/// smoothing, a z-loss and the entropy, each row's loss; f32.
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
    let g_loss = normal(&[n], 11)?;
    let g_lp = normal(&[n], 9)?;
    let g_ent = normal(&[n], 10)?;
    let mut ids: Vec<i64> = (0..n as i64).map(|i| (i * 7919) % v as i64).collect();
    ids[3] = -1;
    ids[4] = v as i64;
    ids[6] = -100;
    let masked = vec![ids[5] as u32, 0, 3070, 3071, 3072, 3073, (v - 1) as u32];
    let targets = Array::from_slice(&ids, &[n]);
    // Each row's loss: the mean is +inf with targets masked or out of range.
    let options = CrossEntropyOptions {
        reduction: Reduction::None,
        label_smoothing: 0.1,
        z_loss: 1e-3,
        masked,
        entropy: true,
        chunks: LogprobOptions {
            vocab_chunk: 0,
            row_chunk: 400,
        },
        ..Default::default()
    };
    let run = |op: Op| -> Result<Vec<Vec<f32>>> {
        let f = |p: &[Array]| -> Result<Vec<Array>> {
            let out = op(&p[0], &p[1], &targets, &options)?;
            Ok(vec![out.loss, out.logprob, out.entropy.expect("asked for")])
        };
        let (outputs, grads) = fallible_vjp(
            f,
            &[hidden.clone(), weight.clone()],
            &[g_loss.clone(), g_lp.clone(), g_ent.clone()],
        )?;
        eval(outputs.iter().chain(&grads))?;
        Ok(outputs.iter().chain(&grads).map(values).collect())
    };
    let kernels = run(linear_cross_entropy)?;
    let reference = run(linear_cross_entropy_reference)?;
    for (i, what) in ["loss", "logprob", "entropy", "grad_hidden", "grad_weight"]
        .iter()
        .enumerate()
    {
        let (abs, rel) = error(what, &kernels[i], &reference[i]);
        println!(
            "kernels against the reference {what:12} max abs {abs:9.3e}  relative {}",
            up3(rel)
        );
        assert!(rel <= 1e-6, "{what}: {rel:e}");
    }
    for row in [3, 4, 5] {
        assert_eq!(kernels[1][row], f32::NEG_INFINITY, "row {row}");
    }
    assert_eq!(kernels[1][6], 0.0, "an ignored row's log-probability is 0");
    Ok(())
}

/// The backward pass knows whether the weight's gradient is asked for, which only changes the
/// order MLX evaluates in: with the weight differentiated (alone or with `hidden`) every pass
/// interleaves its two products, with a frozen weight none does, and the gradient of `hidden` is
/// the same to the bit either way.
#[test]
fn the_weight_gradient_is_seen_when_asked_for() -> Result<()> {
    let _serial = serial();
    let case = Case::load("peaked");
    let options = case.options("smooth");
    let blocks = (case.n as usize).div_ceil(options.chunks.row_chunk);
    let frozen = Setup {
        weight: false,
        ..ALL
    };
    let probe_only = Setup {
        hidden: false,
        ..ALL
    };
    for (path, op) in PATHS {
        let mut grad_hidden = Vec::new();
        for (setup, interleaved) in [(frozen, 0), (ALL, blocks), (probe_only, blocks)] {
            let (passes, before) = (probe::gradient_passes(), probe::interleaved_passes());
            let got = run(op, &case, Precision::Bf16, Dtype::Bfloat16, &options, setup)?;
            assert_eq!(
                probe::gradient_passes() - passes,
                blocks,
                "{path} {setup:?}"
            );
            assert_eq!(
                probe::interleaved_passes() - before,
                interleaved,
                "{path} {setup:?}"
            );
            if setup.hidden {
                grad_hidden.push(
                    got.grad_hidden
                        .iter()
                        .map(|g| g.to_bits())
                        .collect::<Vec<_>>(),
                );
            }
        }
        assert!(
            grad_hidden[0] == grad_hidden[1],
            "{path}: the order changed a bit"
        );
    }
    Ok(())
}
