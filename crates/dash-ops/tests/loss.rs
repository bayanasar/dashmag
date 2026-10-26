//! The fused linear cross-entropy against PyTorch.
//!
//! `fixtures/loss/cases.safetensors` holds three cases, their inputs as f32 values, and for two
//! input precisions (the values as they are, and rounded to bf16 so that a bf16 test casts them
//! without loss) the reference computed by PyTorch in float64 and stored as f32: the loss, each
//! row's log-probability and entropy, and the gradients for `hidden` and `weight` of
//! `loss + sum_i g_lp[i] logprob[i] + sum_i g_ent[i] entropy[i]` (with `sum_i g_loss[i] loss[i]`
//! for the loss when it is not reduced). Four variants of the options: `mean` is torch's
//! default; `smooth` takes label smoothing, a z-loss and masked ids, and records the gradient of
//! each of the three terms alone too; `sum` also masks the last counted row's target (its loss
//! is `+inf`); `none` keeps every row's loss and ignores another id than `-100`, which makes the
//! `-100` targets out of range (`+inf`). Each case ignores rows; `basic` masks ids at both ends of
//! its 300-token vocabulary, `peaked` has logits up to 38 and rows whose target has p = 0.98,
//! `wide` has D = 256. The generator checked every variant against `F.cross_entropy`.
//!
//! The tests chunk the vocabulary in 64 tokens (the last chunk partial in every case: one token
//! in `peaked`) and the rows in blocks of 10 or fewer. On Flex the reference path is held to the
//! fixtures in f32 and bf16; on CUDA the kernel path, in bf16 (the tensor-core path) and f32, and
//! also to the reference path at a hidden width of 1024 and with several large row blocks. Each
//! path is taken: both inputs requiring a gradient, a frozen weight with and without the entropy
//! (the second case accumulates the gradient of `hidden` in the forward pass), a frozen `hidden`;
//! f16 inputs, computed in f32, against f32 inputs of the same values. The probe counters show
//! which code ran: the CUDA kernels, the written-out backward, and whether it made a second pass
//! over the vocabulary.
//!
//! Errors are stated as `max |got - want|` and that over `max |want|` ("relative"), per output;
//! each tolerance sits just above the largest error measured over the five builds.
//!
//!   cargo test --release -p dash-ops --test loss
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test loss
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda,fusion --test loss

use burn::backend::{DType, bf16};
use burn::tensor::{Device, Int, Tensor, TensorData};
use dash_ops::logprob::{LogprobOptions, chosen_logprob};
use dash_ops::loss::{CrossEntropyOptions, Reduction, linear_cross_entropy, probe};
use safetensors::SafeTensors;
use std::sync::Mutex;

/// The probe counters are global; tests that read them take turns.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/loss");
const CASES: [&str; 3] = ["basic", "peaked", "wide"];

/// Chunks small enough that every case crosses several, the last one partial, and row blocks
/// that split the rows unevenly.
const SMALL_CHUNKS: LogprobOptions = LogprobOptions {
    vocab_chunk: 64,
    row_chunk: 10,
};

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
            Precision::Bf16 => values.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
        }
    }
}

struct Case {
    name: &'static str,
    n: usize,
    v: usize,
    d: usize,
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

/// What a run returns, or the reference, all as f32.
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
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn load(name: &'static str) -> Case {
    let tensors = std::fs::read(format!("{FIXTURES}/cases.safetensors")).unwrap();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FIXTURES}/meta.json")).unwrap())
            .unwrap();
    let meta = &meta[name];
    let dim = |key: &str| meta[key].as_u64().unwrap() as usize;
    let targets = {
        let st = SafeTensors::deserialize(&tensors).unwrap();
        let view = st.tensor(&format!("{name}.targets")).unwrap();
        assert_eq!(view.dtype(), safetensors::Dtype::I64);
        view.data()
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
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
    Case {
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

impl Case {
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

/// Which inputs require a gradient, and which terms the objective sums.
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

fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.to_data().convert::<f32>().try_to_vec::<f32>().unwrap()
}

/// The loss on `device` (an autodiff device unless nothing requires a gradient), with inputs at
/// `precision` cast to `dtype`, and the gradients of the objective the generator used.
fn run(
    device: &Device,
    case: &Case,
    precision: Precision,
    dtype: DType,
    options: &CrossEntropyOptions,
    setup: Setup,
) -> Outputs {
    let (n, v, d) = (case.n, case.v, case.d);
    let mut hidden = Tensor::<2>::from_data(
        TensorData::new(precision.round(&case.hidden), [n, d]),
        device,
    )
    .cast(dtype);
    let mut weight = Tensor::<2>::from_data(
        TensorData::new(precision.round(&case.weight), [v, d]),
        device,
    )
    .cast(dtype);
    if setup.hidden {
        hidden = hidden.require_grad();
    }
    if setup.weight {
        weight = weight.require_grad();
    }
    let targets = Tensor::<1, Int>::from_data(TensorData::new(case.targets.clone(), [n]), device);
    let upstream = |g: &[f32]| Tensor::<1>::from_data(TensorData::new(g.to_vec(), [n]), device);
    let out = linear_cross_entropy(hidden.clone(), weight.clone(), targets, options);
    for x in [Some(&out.loss), Some(&out.logprob), out.entropy.as_ref()]
        .into_iter()
        .flatten()
    {
        assert_eq!(x.dtype(), DType::F32, "the outputs are f32");
    }
    let entropy = out.entropy.clone().map(values).unwrap_or_default();
    let mut outputs = Outputs {
        loss: values(out.loss.clone()),
        logprob: values(out.logprob.clone()),
        entropy,
        grad_hidden: vec![],
        grad_weight: vec![],
    };
    if !(setup.hidden || setup.weight) {
        return outputs;
    }
    let mut objective: Option<Tensor<1>> = None;
    let mut add = |term: Tensor<1>| {
        objective = Some(match objective.take() {
            Some(sum) => sum + term,
            None => term,
        })
    };
    if setup.loss {
        add(match options.reduction {
            Reduction::None => (out.loss * upstream(&case.g_loss)).sum(),
            _ => out.loss,
        });
    }
    if setup.logprob {
        add((out.logprob * upstream(&case.g_lp)).sum());
    }
    if setup.entropy {
        let entropy = out.entropy.expect("the entropy was asked for");
        add((entropy * upstream(&case.g_ent)).sum());
    }
    let grads = objective.expect("a term").backward();
    if setup.hidden {
        let grad = hidden.grad(&grads).expect("hidden has a gradient");
        assert_eq!(
            grad.dtype(),
            dtype,
            "the gradient comes back in hidden's dtype"
        );
        outputs.grad_hidden = values(grad);
    }
    if setup.weight {
        let grad = weight.grad(&grads).expect("weight has a gradient");
        assert_eq!(
            grad.dtype(),
            dtype,
            "the gradient comes back in weight's dtype"
        );
        outputs.grad_weight = values(grad);
    }
    outputs
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

/// The tolerances of one build and dtype, relative: loss, logprob, entropy, grad_hidden,
/// grad_weight.
type Tolerance = [f64; 5];

/// Asserts each output's relative error is within its tolerance, and prints the measured ones;
/// an empty `got` is not checked (not computed in that run).
fn check(label: &str, got: &Outputs, want: &Outputs, tolerance: Tolerance) {
    for ((what, got, want), tolerance) in [
        ("loss", &got.loss, &want.loss),
        ("logprob", &got.logprob, &want.logprob),
        ("entropy", &got.entropy, &want.entropy),
        ("grad_hidden", &got.grad_hidden, &want.grad_hidden),
        ("grad_weight", &got.grad_weight, &want.grad_weight),
    ]
    .into_iter()
    .zip(tolerance)
    {
        if got.is_empty() {
            continue;
        }
        let (abs, rel) = error(what, got, want);
        println!(
            "{label:34} {what:12} max abs {abs:9.3e}  relative {rel:9.3e}  (tolerance {tolerance:.0e})"
        );
        assert!(
            rel <= tolerance,
            "{label} {what}: relative error {rel:e} above {tolerance:e} (max abs {abs:e})"
        );
    }
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

/// The reference path (Burn's tensor operations in f32) against float64 PyTorch. Measured:
/// 1.0e-7, 9.3e-8, 7.4e-7, 2.8e-6, 3.6e-6. The largest, the entropy and the gradients of
/// `peaked` (logits up to 38), are at f32's noise floor: PyTorch's own f32 lands 9e-7 and 3e-6
/// from float64 there.
const FLEX_F32: Tolerance = [2e-7, 2e-7, 1e-6, 4e-6, 4e-6];
/// bf16 inputs on Flex are widened to f32 inside; the gradients are rounded to bf16 on the way
/// out (half an ulp is 2^-9 of the value, at most 2^-8 relative to a power of two). Measured:
/// 1.5e-7, 9.3e-8, 4.3e-7, 2.5e-3, 3.4e-3.
const FLEX_BF16: Tolerance = [2e-7, 2e-7, 5e-7, 3e-3, 4e-3];

/// Every case and variant on `device`, track both inputs: `(label, got, want)` checked against
/// `tolerance`, with the probes showing the backward pass ran and made its second pass.
fn all_variants(
    device: &Device,
    label: &str,
    precision: Precision,
    dtype: DType,
    tolerance: Tolerance,
) {
    for name in CASES {
        let case = load(name);
        for (variant, options) in &case.variants {
            let (runs, passes) = (probe::backward_runs(), probe::gradient_passes());
            let got = run(device, &case, precision, dtype, options, ALL);
            assert!(
                probe::backward_runs() > runs,
                "the written-out backward did not run"
            );
            assert!(
                probe::gradient_passes() > passes,
                "a trained weight takes the second pass"
            );
            check(
                &format!("{label} {name} {variant}"),
                &got,
                &case.want(precision, variant, None),
                tolerance,
            );
        }
    }
}

#[test]
fn matches_pytorch_on_flex() {
    let _serial = serial();
    all_variants(
        &Device::flex().autodiff(),
        "flex f32",
        Precision::F32,
        DType::F32,
        FLEX_F32,
    );
}

#[test]
fn bf16_inputs_on_flex() {
    let _serial = serial();
    all_variants(
        &Device::flex().autodiff(),
        "flex bf16",
        Precision::Bf16,
        DType::BF16,
        FLEX_BF16,
    );
}

/// Each term of the objective alone: the loss's, the log-probability's and the entropy's own
/// gradients, for both inputs.
fn each_term(
    device: &Device,
    label: &str,
    precision: Precision,
    dtype: DType,
    tolerance: Tolerance,
) {
    for name in ["basic", "peaked"] {
        let case = load(name);
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
            let got = run(device, &case, precision, dtype, &options, setup);
            check(
                &format!("{label} {name} {term} alone"),
                &got,
                &case.want(precision, "smooth", Some(term)),
                tolerance,
            );
        }
    }
}

#[test]
fn each_output_has_its_gradient_on_flex() {
    let _serial = serial();
    each_term(
        &Device::flex().autodiff(),
        "flex f32",
        Precision::F32,
        DType::F32,
        FLEX_F32,
    );
}

/// A weight that requires no gradient: without the entropy the forward pass accumulates the
/// gradient of `hidden` and the backward pass makes no second pass over the vocabulary; with it,
/// the backward pass walks the vocabulary again for `hidden` alone.
fn frozen_weight(
    device: &Device,
    label: &str,
    precision: Precision,
    dtype: DType,
    tolerance: Tolerance,
) {
    for name in CASES {
        let case = load(name);
        // Without the entropy: the loss and the log-probability's terms.
        let mut options = case.options("smooth");
        options.entropy = false;
        let setup = Setup {
            weight: false,
            entropy: false,
            ..ALL
        };
        let (runs, passes) = (probe::backward_runs(), probe::gradient_passes());
        let got = run(device, &case, precision, dtype, &options, setup);
        assert!(
            probe::backward_runs() > runs,
            "the written-out backward did not run"
        );
        assert_eq!(
            probe::gradient_passes(),
            passes,
            "a frozen weight without the entropy takes no second pass"
        );
        let want = match name {
            // `wide` records no term alone: compare the outputs only.
            "wide" => None,
            _ => Some(sum_of(
                case.want(precision, "smooth", Some("loss")),
                &case.want(precision, "smooth", Some("lp")),
            )),
        };
        if let Some(want) = want {
            let got = Outputs {
                entropy: vec![],
                ..got
            };
            check(&format!("{label} {name} frozen"), &got, &want, tolerance);
        }
        // With the entropy, every variant: the second pass, for `hidden` alone.
        for (variant, options) in &case.variants {
            let setup = Setup {
                weight: false,
                ..ALL
            };
            let passes = probe::gradient_passes();
            let got = run(device, &case, precision, dtype, options, setup);
            assert!(
                probe::gradient_passes() > passes,
                "the entropy's gradient takes the second pass"
            );
            check(
                &format!("{label} {name} {variant} frozen"),
                &got,
                &case.want(precision, variant, None),
                tolerance,
            );
        }
    }
}

#[test]
fn frozen_weight_on_flex() {
    let _serial = serial();
    frozen_weight(
        &Device::flex().autodiff(),
        "flex f32",
        Precision::F32,
        DType::F32,
        FLEX_F32,
    );
}

/// `hidden` that requires no gradient, a weight that does: a linear probe on frozen features.
fn hidden_frozen(
    device: &Device,
    label: &str,
    precision: Precision,
    dtype: DType,
    tolerance: Tolerance,
) {
    let case = load("basic");
    for (variant, options) in &case.variants {
        let setup = Setup {
            hidden: false,
            ..ALL
        };
        let got = run(device, &case, precision, dtype, options, setup);
        assert!(got.grad_hidden.is_empty());
        check(
            &format!("{label} basic {variant} hidden frozen"),
            &got,
            &case.want(precision, variant, None),
            tolerance,
        );
    }
}

#[test]
fn hidden_frozen_on_flex() {
    let _serial = serial();
    hidden_frozen(
        &Device::flex().autodiff(),
        "flex f32",
        Precision::F32,
        DType::F32,
        FLEX_F32,
    );
}

/// The chunking is an implementation detail: one chunk, tiny chunks and uneven row blocks agree.
#[test]
fn chunking_does_not_change_the_answer_on_flex() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    let case = load("peaked");
    for chunks in [
        LogprobOptions {
            vocab_chunk: case.v,
            row_chunk: case.n,
        },
        LogprobOptions {
            vocab_chunk: 7,
            row_chunk: 5,
        },
        LogprobOptions {
            vocab_chunk: 1000,
            row_chunk: 1,
        },
    ] {
        for variant in ["smooth", "none"] {
            let options = CrossEntropyOptions {
                chunks,
                ..case.options(variant)
            };
            let got = run(&device, &case, Precision::F32, DType::F32, &options, ALL);
            check(
                &format!("flex {variant} {chunks:?}"),
                &got,
                &case.want(Precision::F32, variant, None),
                FLEX_F32,
            );
        }
    }
}

/// Without autodiff the same values come out, and nothing is recorded.
#[test]
fn untracked_on_flex() {
    let _serial = serial();
    let case = load("basic");
    let runs = probe::backward_runs();
    for (variant, options) in &case.variants {
        let setup = Setup {
            hidden: false,
            weight: false,
            ..ALL
        };
        let got = run(
            &Device::flex(),
            &case,
            Precision::F32,
            DType::F32,
            options,
            setup,
        );
        check(
            &format!("flex untracked {variant}"),
            &got,
            &case.want(Precision::F32, variant, None),
            FLEX_F32,
        );
    }
    assert_eq!(probe::backward_runs(), runs);
}

/// The loss's log-probability is `chosen_logprob`'s, its gradient too (frozen weight), and
/// without smoothing or z-loss the loss is minus the mean log-probability over counted rows.
fn agrees_with_chosen_logprob(device: &Device, dtype: DType) {
    let case = load("basic");
    let (n, v, d) = (case.n, case.v, case.d);
    let masked = case.options("sum").masked;
    let inputs = || {
        (
            Tensor::<2>::from_data(
                TensorData::new(Precision::Bf16.round(&case.hidden), [n, d]),
                device,
            )
            .cast(dtype)
            .require_grad(),
            Tensor::<2>::from_data(
                TensorData::new(Precision::Bf16.round(&case.weight), [v, d]),
                device,
            )
            .cast(dtype),
            Tensor::<1, Int>::from_data(TensorData::new(case.targets.clone(), [n]), device),
        )
    };
    let g = Tensor::<1>::from_data(TensorData::new(case.g_lp.clone(), [n]), device);
    // Rows ignored by the loss give it 0; chosen_logprob has -inf there (out of range).
    let counted: Vec<f32> = case
        .targets
        .iter()
        .map(|&t| if t == -100 { 0.0 } else { 1.0 })
        .collect();
    let counted = Tensor::<1>::from_data(TensorData::new(counted, [n]), device);

    let (h, w, t) = inputs();
    let options = CrossEntropyOptions {
        masked: masked.clone(),
        chunks: SMALL_CHUNKS,
        ..Default::default()
    };
    let out = linear_cross_entropy(h.clone(), w, t, &options);
    let grads = (out.logprob.clone() * g.clone()).sum().backward();
    let loss_grad = values(h.grad(&grads).unwrap());
    let loss_lp = values(out.logprob);
    let loss_value = values(out.loss)[0];

    let (h, w, t) = inputs();
    let chosen = chosen_logprob(h.clone(), w, t, &masked, SMALL_CHUNKS);
    let grads = (chosen.logprob.clone() * g * counted).sum().backward();
    let chosen_grad = values(h.grad(&grads).unwrap());
    let chosen_lp: Vec<f32> = values(chosen.logprob)
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
    assert_eq!(loss_lp[n - 1], f32::NEG_INFINITY);
}

#[test]
fn agrees_with_chosen_logprob_on_flex() {
    let _serial = serial();
    agrees_with_chosen_logprob(&Device::flex().autodiff(), DType::F32);
}

/// No rows: empty outputs, a NaN mean (torch's), zero gradients of the right shapes.
fn empty_batch(device: &Device) {
    let hidden = Tensor::<2>::zeros([0, 3], device).require_grad();
    let weight = Tensor::<2>::ones([5, 3], device).require_grad();
    let targets = Tensor::<1, Int>::zeros([0], device);
    let options = CrossEntropyOptions {
        reduction: Reduction::Sum,
        entropy: true,
        ..Default::default()
    };
    let out = linear_cross_entropy(hidden.clone(), weight.clone(), targets, &options);
    assert_eq!(out.logprob.dims(), [0]);
    assert_eq!(out.entropy.as_ref().map(|e| e.dims()), Some([0]));
    assert_eq!(values(out.loss.clone()), vec![0.0]);
    let grads = out.loss.backward();
    assert_eq!(hidden.grad(&grads).map(|g| g.dims()), Some([0, 3]));
    let grad_weight = weight.grad(&grads).expect("weight has a gradient");
    assert_eq!(grad_weight.dims(), [5, 3]);
    assert!(values(grad_weight).iter().all(|&g| g == 0.0));
}

/// Every row ignored: the mean is NaN (torch's 0 / 0), and every gradient 0 although the mean's
/// own gradient is NaN: an ignored row receives none, whatever its upstream holds.
fn all_ignored(device: &Device) {
    let case = load("basic");
    let (n, v, d) = (case.n, case.v, case.d);
    let hidden =
        Tensor::<2>::from_data(TensorData::new(case.hidden.clone(), [n, d]), device).require_grad();
    let weight =
        Tensor::<2>::from_data(TensorData::new(case.weight.clone(), [v, d]), device).require_grad();
    let targets = Tensor::<1, Int>::full([n], -100, device);
    let options = CrossEntropyOptions {
        label_smoothing: 0.1,
        z_loss: 1e-2,
        chunks: SMALL_CHUNKS,
        ..Default::default()
    };
    let out = linear_cross_entropy(hidden.clone(), weight.clone(), targets, &options);
    assert!(values(out.loss.clone())[0].is_nan());
    assert!(values(out.logprob.clone()).iter().all(|&x| x == 0.0));
    let grads = (out.logprob.sum() + out.loss.clone() * 0.0).backward();
    for grad in [
        values(hidden.grad(&grads).unwrap()),
        values(weight.grad(&grads).unwrap()),
    ] {
        assert!(grad.iter().all(|&g| g == 0.0));
    }
}

#[test]
fn empty_batch_on_flex() {
    empty_batch(&Device::flex().autodiff());
}

#[test]
fn all_ignored_on_flex() {
    let _serial = serial();
    all_ignored(&Device::flex().autodiff());
}

/// f16 inputs are computed in f32: the same as f32 inputs of the same values, but for the
/// gradients' rounding to f16 on the way out.
fn f16_inputs(device: &Device, label: &str) {
    let mut case = load("basic");
    let f16 = |x: &[f32]| -> Vec<f32> {
        x.iter()
            .map(|&x| burn::backend::f16::from_f32(x).to_f32())
            .collect()
    };
    case.hidden = f16(&case.hidden);
    case.weight = f16(&case.weight);
    let options = case.options("smooth");
    let want = run(device, &case, Precision::F32, DType::F32, &options, ALL);
    let got = run(device, &case, Precision::F32, DType::F16, &options, ALL);
    let want = Outputs {
        grad_hidden: f16(&want.grad_hidden),
        grad_weight: f16(&want.grad_weight),
        ..want
    };
    check(label, &got, &want, F32_VS_F16);
}

/// f16 against f32 on the same values, both computed in f32: measured 0 on every build. The
/// slack is for autotune choosing another kernel for one of the two runs, which can move an f32
/// output by an ulp and round a gradient to the next f16 (an ulp, at most 2^-10 of the largest).
const F32_VS_F16: Tolerance = [2e-7, 2e-7, 1e-6, 1e-3, 1e-3];

#[test]
fn f16_inputs_on_flex() {
    let _serial = serial();
    f16_inputs(&Device::flex().autodiff(), "flex f16 vs f32");
}

#[test]
#[should_panic(expected = "label_smoothing")]
fn a_smoothing_above_one_is_refused() {
    let device = Device::flex();
    let _ = linear_cross_entropy(
        Tensor::<2>::zeros([2, 3], &device),
        Tensor::<2>::zeros([5, 3], &device),
        Tensor::<1, Int>::from_ints([0, 1], &device),
        &CrossEntropyOptions {
            label_smoothing: 1.5,
            ..Default::default()
        },
    );
}

#[test]
#[should_panic(expected = "outside the vocabulary")]
fn a_masked_id_outside_the_vocabulary_is_refused() {
    let device = Device::flex();
    let _ = linear_cross_entropy(
        Tensor::<2>::zeros([2, 3], &device),
        Tensor::<2>::zeros([5, 3], &device),
        Tensor::<1, Int>::from_ints([0, 1], &device),
        &CrossEntropyOptions {
            masked: vec![5],
            ..Default::default()
        },
    );
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// bf16 inputs: the tensor-core path, the gradient of the logits rounded to bf16 for the
    /// matmuls of the second pass, the gradients rounded to bf16 on the way out. Measured:
    /// 1.4e-7, 1.4e-7, 2.6e-7, 4.4e-3, 4.9e-3.
    const CUDA_BF16: Tolerance = [2e-7, 2e-7, 3e-7, 5e-3, 5e-3];
    /// f32 inputs: f32 matmuls (IEEE, `CUBECL_CUDA_TF32=0`), nothing rounded. Measured: 1.4e-7,
    /// 9.3e-8, 7.4e-7, 2.7e-6, 3.6e-6.
    const CUDA_F32: Tolerance = [2e-7, 2e-7, 1e-6, 4e-6, 4e-6];
    /// bf16 on CUDA against f32 on Flex on the same values, both gradients rounded to bf16, the
    /// CUDA path's also through the logits' gradient in bf16. Measured: 2.0e-7, 4.5e-7, 5.8e-7,
    /// 4.6e-3, 4.5e-3 (the largest at D = 1024 and in 1800 rows).
    const CUDA_VS_FLEX: Tolerance = [3e-7, 5e-7, 1e-6, 5e-3, 5e-3];

    fn device() -> Device {
        Device::cuda(0).autodiff()
    }

    /// The kernel ran (the counter grew) around `f`.
    fn launches<T>(f: impl FnOnce() -> T) -> T {
        let before = probe::kernel_launches();
        let out = f();
        assert!(
            probe::kernel_launches() > before,
            "the CUDA kernels did not run"
        );
        out
    }

    #[test]
    fn matches_pytorch_on_cuda_bf16() {
        let _serial = serial();
        launches(|| {
            all_variants(
                &device(),
                "cuda bf16",
                Precision::Bf16,
                DType::BF16,
                CUDA_BF16,
            )
        });
    }

    #[test]
    fn matches_pytorch_on_cuda_f32() {
        let _serial = serial();
        launches(|| all_variants(&device(), "cuda f32", Precision::F32, DType::F32, CUDA_F32));
    }

    #[test]
    fn each_output_has_its_gradient_on_cuda() {
        let _serial = serial();
        launches(|| {
            each_term(
                &device(),
                "cuda bf16",
                Precision::Bf16,
                DType::BF16,
                CUDA_BF16,
            )
        });
        launches(|| each_term(&device(), "cuda f32", Precision::F32, DType::F32, CUDA_F32));
    }

    #[test]
    fn frozen_weight_on_cuda() {
        let _serial = serial();
        launches(|| {
            frozen_weight(
                &device(),
                "cuda bf16",
                Precision::Bf16,
                DType::BF16,
                CUDA_BF16,
            )
        });
        launches(|| frozen_weight(&device(), "cuda f32", Precision::F32, DType::F32, CUDA_F32));
    }

    #[test]
    fn hidden_frozen_on_cuda() {
        let _serial = serial();
        launches(|| {
            hidden_frozen(
                &device(),
                "cuda bf16",
                Precision::Bf16,
                DType::BF16,
                CUDA_BF16,
            )
        });
    }

    #[test]
    fn agrees_with_chosen_logprob_on_cuda() {
        let _serial = serial();
        agrees_with_chosen_logprob(&device(), DType::BF16);
    }

    #[test]
    fn empty_batch_on_cuda() {
        empty_batch(&device());
    }

    #[test]
    fn all_ignored_on_cuda() {
        let _serial = serial();
        all_ignored(&device());
    }

    /// Random inputs, rounded to bf16 so that both devices see one value.
    fn random_case(n: usize, v: usize, d: usize) -> Case {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64 ^ (n * v * d) as u64;
        let mut normal = |scale: f32| {
            // Box-Muller on a xorshift.
            let mut uniform = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (uniform(), uniform());
            let x = (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos();
            bf16::from_f32(x as f32 * scale).to_f32()
        };
        let mut targets: Vec<i64> = (0..n as i64).map(|i| (i * 7919 + 13) % v as i64).collect();
        for t in targets.iter_mut().step_by(17) {
            *t = -100;
        }
        Case {
            name: "random",
            n,
            v,
            d,
            hidden: (0..n * d).map(|_| normal(1.0)).collect(),
            weight: (0..v * d)
                .map(|_| normal(1.6 / (d as f32).sqrt()))
                .collect(),
            targets,
            g_loss: (0..n).map(|_| normal(1.0 / n as f32)).collect(),
            g_lp: (0..n).map(|_| normal(1.0 / n as f32)).collect(),
            g_ent: (0..n).map(|_| normal(1.0 / n as f32)).collect(),
            variants: vec![],
            tensors: vec![],
        }
    }

    /// The CUDA path with bf16 operands against the reference path on Flex in f32, on the same
    /// values, every option on.
    fn agrees_with_flex(label: &str, case: &Case, chunks: LogprobOptions) {
        let options = CrossEntropyOptions {
            reduction: Reduction::Sum,
            label_smoothing: 0.1,
            z_loss: 1e-3,
            masked: vec![17, case.v as u32 / 2, case.v as u32 - 1],
            entropy: true,
            chunks,
            ..Default::default()
        };
        let flex = Device::flex().autodiff();
        let want = run(&flex, case, Precision::F32, DType::F32, &options, ALL);
        let got = launches(|| run(&device(), case, Precision::F32, DType::BF16, &options, ALL));
        // Flex rounds its f32 gradients to bf16 too, for a like-for-like comparison.
        let want = Outputs {
            grad_hidden: Precision::Bf16.round(&want.grad_hidden),
            grad_weight: Precision::Bf16.round(&want.grad_weight),
            ..want
        };
        check(label, &got, &want, CUDA_VS_FLEX);
    }

    /// At a hidden width of 1024: several row blocks, chunks and a partial last chunk, then the
    /// defaults (one row block, chunks of 4096 with the weight's gradient written in place).
    #[test]
    fn agrees_with_flex_at_full_width() {
        let _serial = serial();
        let case = random_case(100, 5000, 1024);
        let small = LogprobOptions {
            vocab_chunk: 1024,
            row_chunk: 40,
        };
        agrees_with_flex("cuda bf16 vs flex, D 1024", &case, small);
        agrees_with_flex(
            "cuda bf16 defaults vs flex",
            &case,
            LogprobOptions::default(),
        );
    }

    /// Large row blocks: 1800 rows in one block and in two.
    #[test]
    fn agrees_with_flex_with_large_row_blocks() {
        let _serial = serial();
        let case = random_case(1800, 700, 64);
        let options = |row_chunk| LogprobOptions {
            vocab_chunk: 256,
            row_chunk,
        };
        agrees_with_flex("cuda bf16 vs flex, 1800 rows", &case, options(1800));
        agrees_with_flex("cuda bf16 vs flex, 2 x 900 rows", &case, options(1000));
    }

    #[test]
    fn f16_inputs_on_cuda() {
        let _serial = serial();
        launches(|| f16_inputs(&device(), "cuda f16 vs f32"));
    }
}
