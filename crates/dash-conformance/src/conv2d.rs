//! 2-D convolution against torch, from one recording, `conv2d/cases.safetensors`
//! (`tools/record_conv2d_batch_norm.py`): torch's `F.conv2d` and its gradients evaluated in float64
//! from float32 inputs, and kept in float64, over fourteen cases, among them strides of 1 and 2 and
//! `(2, 1)`, paddings of 0 to 3 (past the kernel too), dilations of 1 and 2 and `(2, 1)`, two
//! groups, depthwise kernels with multipliers of 1 and 2, 1 x 1 and 7 x 7 kernels, and each with
//! and without a bias.
//!
//! Each element of the output and of each gradient is held to a count of roundings
//! ([`dash_contract::conv2d`], Bounds): its error over `2^-24` times its magnitude, the sum of the
//! absolute values of its terms, which the recording keeps beside each reference (the same
//! convolution and gradients evaluated on the inputs' absolute values). An element of `n` terms is
//! allowed `n + 1`: what f32 sums of `n` products give in any order, the bias's addition with them.
//! An element with no terms (an output whose window holds only padding, without a bias) is held to
//! 0 exactly. The cases run in f32, the one dtype the contract's implementations take yet.
//!
//! - [`against_torch`]: every case alone and with its gradients.

use crate::fixtures::File;
use crate::{Adapter, Conv2d, DType, HostTensor, Outcome, Tally, ran};
use dash_contract::conv2d::Conv2dOptions;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

/// f32's unit roundoff, the size of one rounding relative to a value.
pub const UNIT: f64 = 1.0 / (1u64 << 24) as f64;

/// The recording.
const FILE: &str = "conv2d/cases.safetensors";

/// One recorded case.
#[derive(Clone, Debug)]
pub struct Case {
    file: Arc<File>,
    /// The case's name in the recording.
    pub name: String,
    /// Its stride, padding, dilation and groups.
    pub options: Conv2dOptions,
    /// Whether it has a bias.
    pub bias: bool,
    /// The most terms an element sums, per part: `out`, `grad_x`, `grad_weight`, `grad_bias`.
    pub terms: BTreeMap<String, usize>,
}

/// Every recorded case.
pub fn cases() -> Vec<Case> {
    let file = Arc::new(File::read(FILE));
    let cases: BTreeMap<String, Value> =
        serde_json::from_str(&file.metadata()["cases"]).expect("cases JSON");
    cases
        .into_iter()
        .map(|(name, case)| {
            let pair = |key: &str| -> [usize; 2] {
                let v = case[key].as_array().expect("a pair");
                [
                    v[0].as_u64().unwrap() as usize,
                    v[1].as_u64().unwrap() as usize,
                ]
            };
            let options = Conv2dOptions::default()
                .with_stride(pair("stride"))
                .with_padding(pair("padding"))
                .with_dilation(pair("dilation"))
                .with_groups(case["groups"].as_u64().unwrap() as usize);
            let terms = case["terms"]
                .as_object()
                .expect("terms")
                .iter()
                .map(|(part, n)| (part.clone(), n.as_u64().unwrap() as usize))
                .collect();
            Case {
                file: file.clone(),
                name,
                options,
                bias: case["bias"].as_bool().unwrap(),
                terms,
            }
        })
        .collect()
}

impl Case {
    fn part(&self, part: &str) -> String {
        format!("{}.{part}", self.name)
    }

    /// The input `part` (`x`, `weight`, `bias`, `d_out`), f32.
    pub fn input(&self, part: &str) -> HostTensor {
        self.file.tensor(&self.part(part))
    }

    /// The bias, if the case has one.
    pub fn bias(&self) -> Option<HostTensor> {
        self.bias.then(|| self.input("bias"))
    }

    /// The reference for `part` (`out`, `grad_x`, `grad_weight`, `grad_bias`), and each element's
    /// magnitude: the shape, then the two in f64.
    pub fn reference(&self, part: &str) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
        let (shape, values) = self.file.wide(&self.part(part));
        let (_, magnitude) = self.file.wide(&self.part(&format!("{part}.magnitude")));
        (shape, values, magnitude)
    }

    /// The roundings an element of `part` is allowed: its terms, and one more.
    pub fn allowed(&self, part: &str) -> f64 {
        (self.terms[part] + 1) as f64
    }
}

/// The largest count of roundings measured per part, over the cases compared.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Roundings(pub BTreeMap<String, f64>);

impl Roundings {
    /// The largest count measured for `part`, 0 where nothing was compared.
    pub fn of(&self, part: &str) -> f64 {
        self.0.get(part).copied().unwrap_or(0.0)
    }
}

/// `got`'s errors against `want`, each counted in roundings of its magnitude: the largest count,
/// and the largest the case allows. Asserts every element under `allowed` roundings.
///
/// # Panics
///
/// If the shapes or dtypes differ from the reference's, a value is not finite, or an element errs
/// by more than it is allowed.
pub fn count(label: &str, case: &Case, part: &str, got: &HostTensor) -> f64 {
    let (shape, want, magnitude) = case.reference(part);
    assert_eq!(got.shape, shape, "{label} {} {part}: shape", case.name);
    assert_eq!(got.dtype, DType::F32, "{label} {} {part}: dtype", case.name);
    let allowed = case.allowed(part);
    // Each rounding is at most `UNIT` relative, so `k` of them at most `k u / (1 - k u)`.
    let gamma = allowed * UNIT / (1.0 - allowed * UNIT);
    let mut worst = 0f64;
    for (i, ((&g, &w), &m)) in got.values.iter().zip(&want).zip(&magnitude).enumerate() {
        assert!(g.is_finite(), "{label} {} {part}[{i}]: {g}", case.name);
        let err = (g as f64 - w).abs();
        let roundings = if m == 0.0 {
            assert_eq!(
                err, 0.0,
                "{label} {} {part}[{i}]: no terms, off by {err:e}",
                case.name
            );
            0.0
        } else {
            err / (UNIT * m)
        };
        assert!(
            err <= gamma * m,
            "{label} {} {part}[{i}]: {g} against {w}, off by {roundings:.2} roundings of its \
             magnitude {m:.3e}, {allowed} allowed",
            case.name
        );
        worst = worst.max(roundings);
    }
    println!(
        "{label} {} {part}: at most {worst:.2} roundings ({allowed} allowed)",
        case.name
    );
    worst
}

/// Every case through the adapter, alone and with the gradients of `x`, the weight and the bias,
/// in f32: each element of the output, from either call, and of each gradient within its count of
/// roundings. The outcome's value is the largest count measured per part.
pub fn against_torch(adapter: &dyn Adapter, label: &str) -> Outcome<Roundings> {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let mut worst = Roundings::default();
    if !adapter.has(DType::F32) {
        return tally.outcome(worst);
    }
    let mut note = |part: &str, roundings: f64| {
        let w = worst.0.entry(part.to_string()).or_default();
        *w = w.max(roundings);
    };
    for case in cases() {
        let alone = Conv2d {
            x: case.input("x"),
            weight: case.input("weight"),
            bias: case.bias(),
            options: case.options,
            grad: None,
        };
        let Some(out) = ran(adapter, &case.name, adapter.conv2d(&alone)) else {
            continue;
        };
        assert!(
            out.grads.is_none(),
            "{label} {}: gradients not asked for",
            case.name
        );
        note("out", count(label, &case, "out", &out.out));
        let with = Conv2d {
            grad: Some(case.input("d_out")),
            ..alone
        };
        let Some(result) = ran(adapter, &case.name, adapter.conv2d(&with)) else {
            continue;
        };
        note("out", count(label, &case, "out", &result.out));
        let grads = result.grads.expect("the gradients asked for");
        note("grad_x", count(label, &case, "grad_x", &grads.x));
        note(
            "grad_weight",
            count(label, &case, "grad_weight", &grads.weight),
        );
        match (&grads.bias, case.bias) {
            (Some(bias), true) => note("grad_bias", count(label, &case, "grad_bias", bias)),
            (None, false) => {}
            (got, _) => panic!(
                "{label} {}: a bias gradient {} for a case {} a bias",
                case.name,
                if got.is_some() { "returned" } else { "missing" },
                if case.bias { "with" } else { "without" }
            ),
        }
    }
    for (part, roundings) in &worst.0 {
        println!("{label} worst {part}: {roundings:.2} roundings");
    }
    tally.outcome(worst)
}
