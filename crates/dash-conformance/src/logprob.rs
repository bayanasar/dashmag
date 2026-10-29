//! The chosen-token log-probability and the entropy against PyTorch, from
//! `logprob/cases.safetensors`: three cases, their inputs rounded to bf16 so that a bf16 call
//! takes them without loss, and their reference, computed by PyTorch in float64 and stored as
//! f32: the log-probabilities, the entropies and the gradient of `sum_i g[i] logprob[i]` for
//! `hidden`, with and without the masked token ids (`logprob/meta.json` has the shapes and the
//! ids). `basic` has two masked ids at the two ends of the vocabulary; `peaked` has logits up to
//! 49, rows whose target has p = 0.99 and a masked token that is the argmax of the most rows;
//! `wide` has D = 256. In each, the last row's target is a masked token, so its log-probability
//! is `-inf`.
//!
//! Errors are stated as `max |got - want|` and that over `max |want|` ("relative"), per output
//! ([`errors`]), and `-inf` must match `-inf`; each bound ([`bounds`]) sits just above the largest
//! error measured. Every body runs its cases in f32 and in bf16, each where the device has it.
//!
//! - [`matches_pytorch`]: every case with its masked ids, in chunks small enough that each
//!   crosses several ([`SMALL_CHUNKS`]): calls of [`Adapter::chosen_logprob`] with the gradient;
//! - [`masked_tokens_leave_the_softmax`]: every case without the mask, against PyTorch without
//!   it, and the mask moving each row's answer by what it moves PyTorch's;
//! - [`chunking_does_not_change_the_answer`]: `peaked` in one chunk, in tiny chunks, in uneven
//!   row blocks and in the implementation's own chunks;
//! - [`without_gradients`]: a call that asks for no gradient gives the same values;
//! - [`empty_batch`]: no rows at all give empty outputs and an empty gradient.

use crate::fixtures::{self, File};
use crate::{Adapter, ChosenLogprob, DType, HostTensor, Outcome, Tally, ran};
use dash_contract::logprob::LogprobOptions;
use serde_json::Value;

/// The recorded cases.
pub const CASES: [&str; 3] = ["basic", "peaked", "wide"];

/// Chunks small enough that every case crosses several, the last one partial, and row blocks
/// that split the rows unevenly.
pub const SMALL_CHUNKS: LogprobOptions = LogprobOptions {
    vocab_chunk: 128,
    row_chunk: 10,
};

/// The outputs, in the order the bodies compare them.
const PARTS: [&str; 3] = ["logprob", "entropy", "grad_hidden"];

/// One case: its inputs and PyTorch's results. A test of an implementation's own may build one
/// of its own inputs, its results left empty.
#[derive(Clone, Debug)]
pub struct Case {
    /// The case's name in the recording.
    pub name: String,
    /// The rows.
    pub n: usize,
    /// The vocabulary.
    pub v: usize,
    /// The hidden width.
    pub d: usize,
    /// `[n, D]`, bf16 values.
    pub hidden: Vec<f32>,
    /// `[V, D]`, bf16 values.
    pub weight: Vec<f32>,
    /// The target ids, `[n]`.
    pub targets: Vec<i64>,
    /// The cotangent `g` of the log-probabilities, `[n]`.
    pub upstream: Vec<f32>,
    /// The masked token ids.
    pub masked: Vec<u32>,
    /// PyTorch's log-probabilities, entropies and gradient, with the masked ids.
    pub masked_reference: [Vec<f32>; 3],
    /// The same without them.
    pub unmasked_reference: [Vec<f32>; 3],
}

impl Case {
    /// The recorded case `name`, one of [`CASES`].
    ///
    /// # Panics
    ///
    /// If the recording has no such case.
    pub fn load(name: &str) -> Self {
        let file = File::read("logprob/cases.safetensors");
        let path = fixtures::path("logprob/meta.json");
        let meta =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let meta: Value = serde_json::from_str(&meta).expect("meta JSON");
        let meta = &meta[name];
        let dim = |key: &str| meta[key].as_u64().expect("a size") as usize;
        let f32s = |part: &str| {
            let tensor = file.tensor(&format!("{name}.{part}"));
            assert_eq!(tensor.dtype, DType::F32, "{name}.{part}");
            tensor.values
        };
        let outputs = |suffix: &str| PARTS.map(|part| f32s(&format!("{part}{suffix}")));
        Self {
            name: name.to_string(),
            n: dim("n"),
            v: dim("v"),
            d: dim("d"),
            hidden: f32s("hidden"),
            weight: f32s("weight"),
            targets: file.ints(&format!("{name}.targets")),
            upstream: f32s("upstream"),
            masked: meta["masked"]
                .as_array()
                .expect("masked ids")
                .iter()
                .map(|id| id.as_u64().expect("an id") as u32)
                .collect(),
            masked_reference: outputs(""),
            unmasked_reference: outputs("_unmasked"),
        }
    }

    /// The call on the case in `dtype`, with the ids `masked` and `options`, asking for the
    /// gradient.
    pub fn call(&self, dtype: DType, masked: &[u32], options: LogprobOptions) -> ChosenLogprob {
        ChosenLogprob {
            hidden: HostTensor::new(dtype, vec![self.n, self.d], self.hidden.clone()),
            weight: HostTensor::new(dtype, vec![self.v, self.d], self.weight.clone()),
            targets: self.targets.clone(),
            masked: masked.to_vec(),
            options,
            grad: Some(HostTensor::f32(vec![self.n], self.upstream.clone())),
        }
    }
}

/// The bounds `[logprob, entropy, grad_hidden]` on the relative errors against PyTorch of a call
/// in `dtype`, the same on Flex and on every CubeCL runtime. In f32 the largest measured are
/// 8.8e-8, 1.1e-6 and 7.0e-7; the entropy of `peaked` (logits up to 49) is at f32's noise floor,
/// where PyTorch's own f32 lands 9e-7 from float64. From bf16 inputs, 1.8e-7, 6.0e-7 and 3.3e-3:
/// the gradient is rounded to bf16 on the way out (half an ulp is 2^-9 of the value, at most 2^-8
/// relative to a power of two).
///
/// # Panics
///
/// For f16, which the operation does not take.
pub fn bounds(dtype: DType) -> [f64; 3] {
    match dtype {
        DType::F32 => [2e-7, 2e-6, 1e-6],
        DType::BF16 => [2e-7, 2e-6, 4e-3],
        DType::F16 => panic!("the log-probability takes bf16 or f32"),
    }
}

/// `(max |got - want|, that / max |want|)` over the finite entries of `want`, in f64.
///
/// # Panics
///
/// If the lengths differ, an entry of `got` is not finite where `want`'s is, or `want` is `-inf`
/// (or `inf`) where `got` is not the same.
pub fn errors(what: &str, got: &[f32], want: &[f32]) -> (f64, f64) {
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

/// Asserts that the relative error of `got` against `want` is within `bound`, once it has
/// printed it.
fn check_part(label: &str, what: &str, got: &[f32], want: &[f32], bound: f64) {
    let (abs, rel) = errors(what, got, want);
    println!(
        "{label:28} {what:12} max abs {abs:9.3e}  relative {rel:9.3e}  (tolerance {bound:.0e})"
    );
    assert!(
        rel <= bound,
        "{label} {what}: relative error {rel:e} above {bound:e} (max abs {abs:e})"
    );
}

/// The same for each of the log-probabilities, the entropies and the gradient.
pub fn check(label: &str, got: [&[f32]; 3], want: [&[f32]; 3], bounds: [f64; 3]) {
    for (what, ((got, want), bound)) in PARTS.into_iter().zip(got.into_iter().zip(want).zip(bounds))
    {
        check_part(label, what, got, want, bound);
    }
}

/// The three as slices.
fn parts(values: &[Vec<f32>; 3]) -> [&[f32]; 3] {
    [&values[0], &values[1], &values[2]]
}

/// The dtypes the operation takes that `adapter`'s device has, saying which it has not.
fn dtypes(adapter: &dyn Adapter) -> Vec<DType> {
    [DType::F32, DType::BF16]
        .into_iter()
        .filter(|&dtype| {
            let has = adapter.has(dtype);
            if !has {
                println!("skipped: {} has no {dtype:?}", adapter.describe());
            }
            has
        })
        .collect()
}

/// `dtype` as the labels print it.
fn dtype_name(dtype: DType) -> String {
    format!("{dtype:?}").to_lowercase()
}

/// The log-probabilities, the entropies and the gradient that `adapter` gives for `case` in
/// `dtype` with `masked` and `options`, or `None` where it does not run the call.
fn run(
    adapter: &dyn Adapter,
    label: &str,
    case: &Case,
    dtype: DType,
    masked: &[u32],
    options: LogprobOptions,
) -> Option<[Vec<f32>; 3]> {
    let out = ran(
        adapter,
        label,
        adapter.chosen_logprob(&case.call(dtype, masked, options)),
    )?;
    assert_eq!(
        (out.logprob.dtype, out.entropy.dtype),
        (DType::F32, DType::F32),
        "{label}: the log-probabilities and entropies are f32"
    );
    assert_eq!(
        (&out.logprob.shape, &out.entropy.shape),
        (&vec![case.n], &vec![case.n])
    );
    let grad = out.grad_hidden.expect("the gradient asked for");
    assert_eq!(
        grad.dtype, dtype,
        "{label}: the gradient comes back in hidden's dtype"
    );
    assert_eq!(grad.shape, [case.n, case.d]);
    Some([out.logprob.values, out.entropy.values, grad.values])
}

/// Every case with its masked ids in [`SMALL_CHUNKS`], against PyTorch.
pub fn matches_pytorch(adapter: &dyn Adapter, label: &str) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    for dtype in dtypes(adapter) {
        for name in CASES {
            let case = Case::load(name);
            let label = format!("{label} {} {name}", dtype_name(dtype));
            let Some(got) = run(adapter, &label, &case, dtype, &case.masked, SMALL_CHUNKS) else {
                continue;
            };
            check(
                &label,
                parts(&got),
                parts(&case.masked_reference),
                bounds(dtype),
            );
        }
    }
    tally.outcome(())
}

/// Every case without its mask, against PyTorch without it; and masking a token changes every
/// row's answer, since it leaves the log-sum-exp.
pub fn masked_tokens_leave_the_softmax(adapter: &dyn Adapter, label: &str) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    for dtype in dtypes(adapter) {
        for name in CASES {
            let case = Case::load(name);
            let label = format!("{label} {} unmasked {name}", dtype_name(dtype));
            let Some(masked) = run(adapter, &label, &case, dtype, &case.masked, SMALL_CHUNKS)
            else {
                continue;
            };
            let Some(unmasked) = run(adapter, &label, &case, dtype, &[], SMALL_CHUNKS) else {
                continue;
            };
            check(
                &label,
                parts(&unmasked),
                parts(&case.unmasked_reference),
                bounds(dtype),
            );
            assert_masking_matters(&case, &masked, &unmasked);
        }
    }
    tally.outcome(())
}

/// The mask moves each row's log-probability by what it moves PyTorch's, some by far more than
/// any bound here (in `peaked` the masked token is most rows' argmax; elsewhere its probability
/// is too small to show); the last row's target is masked: `-inf` with the mask only.
fn assert_masking_matters(case: &Case, masked: &[Vec<f32>; 3], unmasked: &[Vec<f32>; 3]) {
    let (name, n) = (&case.name, case.n);
    let mut largest = 0.0f32;
    for i in 0..n - 1 {
        let want = case.masked_reference[0][i] - case.unmasked_reference[0][i];
        let got = masked[0][i] - unmasked[0][i];
        assert!(
            (got - want).abs() <= 2e-5 + 1e-3 * want.abs(),
            "{name} row {i}: the mask moved logprob by {got:e}, PyTorch's by {want:e}"
        );
        largest = largest.max(got.abs());
    }
    assert!(
        largest > 5e-3,
        "{name}: the mask moved no logprob by more than {largest:e}"
    );
    assert_eq!(
        masked[0][n - 1],
        f32::NEG_INFINITY,
        "{name}: a masked target"
    );
    assert!(
        unmasked[0][n - 1].is_finite(),
        "{name}: the same target unmasked"
    );
    let (_, rel) = errors("grad", &masked[2], &unmasked[2]);
    assert!(
        rel > 1e-3,
        "{name}: the mask moved the gradient by only {rel:e}"
    );
}

/// The chunking sets what is held at once and the order of the sums, not the answer: `peaked`
/// in one chunk of the whole vocabulary and one block of rows, in chunks of 7 tokens and blocks of
/// 5 rows, in chunks larger than the vocabulary and blocks of one row, and in the
/// implementation's own chunks, against PyTorch.
pub fn chunking_does_not_change_the_answer(adapter: &dyn Adapter, label: &str) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let case = Case::load("peaked");
    for dtype in dtypes(adapter) {
        for options in [
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
            LogprobOptions::default(),
        ] {
            let label = format!("{label} {} {options:?}", dtype_name(dtype));
            let Some(got) = run(adapter, &label, &case, dtype, &case.masked, options) else {
                continue;
            };
            check(
                &label,
                parts(&got),
                parts(&case.masked_reference),
                bounds(dtype),
            );
        }
    }
    tally.outcome(())
}

/// A call that asks for no gradient gives the same log-probabilities and entropies, and no
/// gradient: `basic`, against PyTorch.
pub fn without_gradients(adapter: &dyn Adapter, label: &str) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let case = Case::load("basic");
    for dtype in dtypes(adapter) {
        let label = format!("{label} {} no gradient basic", dtype_name(dtype));
        let call = ChosenLogprob {
            grad: None,
            ..case.call(dtype, &case.masked, SMALL_CHUNKS)
        };
        let Some(out) = ran(adapter, &label, adapter.chosen_logprob(&call)) else {
            continue;
        };
        assert!(
            out.grad_hidden.is_none(),
            "{label}: a gradient not asked for"
        );
        let [logprob, entropy, _] = bounds(dtype);
        let [want_logprob, want_entropy, _] = &case.masked_reference;
        check_part(
            &label,
            "logprob",
            &out.logprob.values,
            want_logprob,
            logprob,
        );
        check_part(
            &label,
            "entropy",
            &out.entropy.values,
            want_entropy,
            entropy,
        );
    }
    tally.outcome(())
}

/// An empty batch, no rows at all, gives empty outputs and an empty gradient.
pub fn empty_batch(adapter: &dyn Adapter) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let call = ChosenLogprob {
        hidden: HostTensor::f32(vec![0, 3], vec![]),
        weight: HostTensor::f32(vec![5, 3], vec![1.0; 15]),
        targets: vec![],
        masked: vec![1],
        options: LogprobOptions::default(),
        grad: Some(HostTensor::f32(vec![0], vec![])),
    };
    if let Some(out) = ran(adapter, "an empty batch", adapter.chosen_logprob(&call)) {
        assert_eq!(out.logprob.shape, [0]);
        assert_eq!(out.entropy.shape, [0]);
        let grad = out.grad_hidden.expect("the gradient asked for");
        assert_eq!(grad.shape, [0, 3]);
    }
    tally.outcome(())
}
