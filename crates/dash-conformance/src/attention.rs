//! Attention against torch, from ten records under `attention/`, and decoding against attention
//! with one query.
//!
//! Each record holds inputs drawn from a normal distribution and rounded to bf16, so that the same
//! inputs feed f32, bf16 and f16 calls, and torch's output, log-sum-exp and gradients of
//! `sum(o * do)` computed from them, in torch's order, the heads before the time (`[B, H, T, D]`),
//! which [`Record::load`] swaps into the contract's `[B, T, H, D]`:
//!
//! - [`CAUSAL`]: causal attention with grouped heads, one record with `D = 256` and one with
//!   `Tq < Tk`, from torch's math `scaled_dot_product_attention` in f32;
//! - [`MASKS`]: no mask with `Tq != Tk`, causality with `Tq < Tk`, causality within a window, a
//!   window alone, key padding with and without causality, packed sequences, and packed cross
//!   attention within a causal window, with grouped heads throughout, from torch in float64 with
//!   an explicit boolean mask (packed sequences recorded as one call per sequence). A query that
//!   sees no key is recorded as the contract defines it: zeros, a log-sum-exp of `-inf`, a zero
//!   query gradient.
//!
//! Errors are the largest absolute difference and that over the largest reference magnitude
//! ([`errors`]); a log-sum-exp of `-inf` must be `-inf` exactly, and its query's output and query
//! gradient exactly zero. The bounds are the caller's, one [`Bounds`] per dtype: those measured for
//! dash-ops differ between its reference on Flex ([`FLEX_F32`]), its reference on CubeCL's
//! runtimes ([`CUBE_F32`], [`AUTOTUNE_F32`] under Burn's autotune) and its kernels ([`BF16`],
//! [`F16`]).
//!
//! - [`matches_torch`]: every record in one dtype, the query gradient summed as
//!   [`AttentionOptions::deterministic`] asks: calls of [`Adapter::attention`] with gradients;
//! - [`without_gradients`]: a call that asks for no gradient gives the same output and
//!   log-sum-exp;
//! - [`decoding_is_attention_with_one_query`]: [`Adapter::attention_decode`] against
//!   [`Adapter::attention`] with one query after every key, without and with a window, and with
//!   per-sequence cache lengths against attention over each sequence's own keys.

use crate::fixtures::File;
use crate::{
    Adapter, Attention, AttentionDecode, DType, HostTensor, KeyPadding, Outcome, Tally, ran,
};
use dash_contract::attention::{AttentionMask, AttentionOptions};

/// The causal records.
pub const CAUSAL: [&str; 2] = ["gqa8x2_d256_t40", "gqa6x2_d64_tq24_tk56"];

/// The records of the other masks (the generator's comments say what each holds).
pub const MASKS: [&str; 8] = [
    "none_b2_gqa2x1_d64_tq19_tk33",
    "causal_b2_gqa4x2_d64_tq17_tk41",
    "causal_window9_gqa3x1_d64_t40",
    "window6x4_b2_gqa2x1_d64_tq24_tk37",
    "padding_b3_gqa2x1_d64_t29",
    "padding_causal_b2_gqa2x1_d64_t26",
    "packed_causal_gqa2x1_d64",
    "packed_cross_causal_window5_gqa2x1_d64",
];

/// The results, in the order every bound and error lists them.
const PARTS: [&str; 5] = ["o", "lse", "dq", "dk", "dv"];

/// The bounds on the relative errors against torch of a run in one dtype, each in the order o,
/// lse, dq, dk, dv: on the [`CAUSAL`] records and on the [`MASKS`] records, measured apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    /// On the causal records.
    pub causal: [f32; 5],
    /// On the records of the other masks.
    pub masks: [f32; 5],
}

/// dash-ops' reference on Flex in f32, which agrees with torch to f32's rounding: just above the
/// largest error measured over every record.
pub const FLEX_F32: Bounds = Bounds {
    causal: [3e-7, 1.1e-7, 4.5e-7, 3.5e-7, 3.5e-7],
    masks: [3.2e-7, 1.2e-7, 3.2e-7, 4.1e-7, 3.5e-7],
};

/// dash-ops' reference on a CubeCL device in f32, its products and reductions Burn's CubeCL
/// kernels, which the runtime picks differently: just above the largest error over the CUDA builds
/// with and without fusion (an RTX 5070 Ti, CUDA 13), lavapipe and OpenGL over D3D12 through wgpu,
/// and CubeCL's CPU runtime. On the causal records dq is 4.2e-7 relative on CUDA and 8.1e-7 on
/// lavapipe, dk 3.3e-7 and 4.4e-7; on OpenGL the log-sum-exp of the mask records reaches 1.33e-7.
pub const CUBE_F32: Bounds = Bounds {
    causal: [3.2e-7, 1.1e-7, 8.5e-7, 4.6e-7, 3.8e-7],
    masks: [5.5e-7, 1.4e-7, 4.3e-7, 4.1e-7, 3.4e-7],
};

/// The same under Burn's autotune, whose kernels change from run to run (the value gradient of
/// the mask records reached 4.0e-7 in one of two runs): a few roundings.
pub const AUTOTUNE_F32: Bounds = Bounds {
    causal: [1e-6, 2e-7, 1e-6, 1e-6, 1e-6],
    masks: [1e-6, 2e-7, 1e-6, 1e-6, 1e-6],
};

/// dash-ops' kernels in bf16, the query gradient summed over the key blocks in a fixed order or
/// with atomics in whatever order the kernels run: just above the largest of both over repeated
/// runs of each CUDA build, where the largest errors came out the same in every run. The error is
/// the output's own rounding (8 significant bits) plus the probabilities' rounding to bf16 before
/// the value product; the log-sum-exp is accumulated in f32 throughout.
pub const BF16: Bounds = Bounds {
    causal: [3.4e-3, 2.5e-7, 4.7e-3, 4e-3, 4.2e-3],
    masks: [3.2e-3, 1.2e-7, 4.3e-3, 4.4e-3, 3.6e-3],
};

/// The same in f16.
pub const F16: Bounds = Bounds {
    causal: [4.6e-4, 1.7e-7, 4.7e-4, 3.6e-4, 4e-4],
    masks: [3.9e-4, 1.2e-7, 7.4e-4, 6.9e-4, 6.4e-4],
};

/// One record: its inputs, mask and options, and torch's results.
#[derive(Clone, Debug)]
pub struct Record {
    /// The record's name.
    pub name: &'static str,
    /// Whether it is one of the [`CAUSAL`] records.
    pub causal: bool,
    /// The queries, `[B, Tq, Hq, D]`, bf16 values as f32.
    pub q: HostTensor,
    /// The keys, `[B, Tk, Hkv, D]`.
    pub k: HostTensor,
    /// The values, `[B, Tk, Hkv, D]`.
    pub v: HostTensor,
    /// The output's cotangent, `[B, Tq, Hq, D]`.
    pub grad: HostTensor,
    /// The mask.
    pub mask: AttentionMask<KeyPadding, Vec<i64>>,
    /// The options: the causality, the window, and the scale where it is not `1 / sqrt(D)`.
    pub options: AttentionOptions,
    /// torch's output, log-sum-exp and the gradients of the queries, keys and values, f32.
    pub expected: [HostTensor; 5],
}

impl Record {
    /// The record `name`, one of [`CAUSAL`] or [`MASKS`].
    ///
    /// # Panics
    ///
    /// If there is no such record.
    pub fn load(name: &'static str) -> Self {
        let file = File::read(&format!("attention/{name}.safetensors"));
        let meta = |key: &str| file.metadata().get(key).cloned();
        let read = |part: &str| swap_1_2(&file.tensor(part).with_dtype(DType::F32));
        let q = read("q");
        // The records' scale is `1 / sqrt(D)` but where it is not: leave it to the default, so
        // that the default is held to torch too.
        let scale: f32 = meta("scale").expect("scale").parse().expect("scale");
        let d = q.shape[3];
        // The causal records carry no `kind`: causal, no mask.
        let causal = meta("causal").is_none_or(|c| c == "true");
        let mut options = AttentionOptions::default().with_causal(causal);
        if scale != (1.0 / (d as f64).sqrt()) as f32 {
            options = options.with_scale(scale);
        }
        let reach = |key: &str| meta(key).map_or(-1, |r| r.parse::<i64>().expect(key));
        let (left, right) = (reach("window_left"), reach("window_right"));
        if left >= 0 || right >= 0 {
            let side = |r: i64| if r < 0 { usize::MAX } else { r as usize };
            options = options.with_window(side(left), side(right));
        }
        let mask = match meta("kind").as_deref() {
            None | Some("none") => AttentionMask::None,
            Some("padding") => {
                let (shape, values) = file.bools("key_padding");
                AttentionMask::KeyPadding(KeyPadding { shape, values })
            }
            Some("packed") => AttentionMask::Packed {
                cu_seqlens_q: file.ints("cu_seqlens_q"),
                cu_seqlens_k: file.ints("cu_seqlens_k"),
            },
            Some(other) => panic!("{name}: unknown mask kind {other}"),
        };
        Self {
            name,
            causal: CAUSAL.contains(&name),
            q,
            k: read("k"),
            v: read("v"),
            grad: read("do"),
            mask,
            options,
            expected: PARTS.map(read),
        }
    }

    /// The call on the record in `dtype`, the query gradient summed as `deterministic` says, with
    /// gradients or without.
    pub fn call(&self, dtype: DType, deterministic: bool, grad: bool) -> Attention {
        Attention {
            q: self.q.clone().with_dtype(dtype),
            k: self.k.clone().with_dtype(dtype),
            v: self.v.clone().with_dtype(dtype),
            mask: self.mask.clone(),
            options: self.options.with_deterministic(deterministic),
            grad: grad.then(|| self.grad.clone()),
        }
    }
}

/// Every record, the causal ones first.
pub fn records() -> Vec<Record> {
    CAUSAL.into_iter().chain(MASKS).map(Record::load).collect()
}

/// `tensor` with dimensions 1 and 2 swapped, the values moved with them: `[B, H, T, ...]` to
/// `[B, T, H, ...]` and back.
pub fn swap_1_2(tensor: &HostTensor) -> HostTensor {
    let shape = &tensor.shape;
    let [b, n1, n2] = [shape[0], shape[1], shape[2]];
    let inner: usize = shape[3..].iter().product();
    let mut values = Vec::with_capacity(tensor.values.len());
    for i in 0..b {
        for j in 0..n2 {
            for k in 0..n1 {
                let at = ((i * n1 + k) * n2 + j) * inner;
                values.extend_from_slice(&tensor.values[at..at + inner]);
            }
        }
    }
    let mut shape = shape.clone();
    shape.swap(1, 2);
    HostTensor::new(tensor.dtype, shape, values)
}

/// The largest absolute difference, and that over the largest reference magnitude, over the
/// entries where the reference is finite.
///
/// # Panics
///
/// If the shapes differ, where the reference is `-inf` (the log-sum-exp of a query that sees no
/// key) and `actual` is not, or where `actual` is not finite and the reference is.
pub fn errors(actual: &HostTensor, expected: &HostTensor) -> (f32, f32) {
    assert_eq!(actual.shape, expected.shape, "shapes differ");
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    for (i, (a, e)) in actual.values.iter().zip(&expected.values).enumerate() {
        if *e == f32::NEG_INFINITY {
            assert_eq!(
                *a,
                f32::NEG_INFINITY,
                "entry {i}: {a} where -inf is expected"
            );
            continue;
        }
        assert!(a.is_finite(), "entry {i}: non-finite value {a}");
        max_abs = max_abs.max((a - e).abs());
        max_ref = max_ref.max(e.abs());
    }
    (max_abs, max_abs / max_ref.max(f32::MIN_POSITIVE))
}

/// The queries that see no key, by the reference log-sum-exp `expected`: their outputs `o` and,
/// where given, query gradients `dq` must be exactly zero. Returns how many there are.
fn check_empty_rows(
    label: &str,
    o: &HostTensor,
    dq: Option<&HostTensor>,
    expected: &HostTensor,
) -> usize {
    let d = o.shape[3];
    let mut empty = 0;
    for (row, &lse) in expected.values.iter().enumerate() {
        if lse != f32::NEG_INFINITY {
            continue;
        }
        empty += 1;
        let range = row * d..(row + 1) * d;
        assert!(
            o.values[range.clone()].iter().all(|&x| x == 0.0),
            "{label}: query {row} sees no key, its output is not zero"
        );
        if let Some(dq) = dq {
            assert!(
                dq.values[range].iter().all(|&x| x == 0.0),
                "{label}: query {row} sees no key, its gradient is not zero"
            );
        }
    }
    empty
}

/// Compares `actual` with `expected`, both in the order o, lse, dq, dk, dv: each relative error
/// must stay within `bounds`, the queries that see no key giving zeros. Prints the errors; returns
/// the largest relative error of each.
///
/// # Panics
///
/// If a relative error exceeds its bound, once it has printed it.
pub fn check(
    label: &str,
    actual: [&HostTensor; 5],
    expected: [&HostTensor; 5],
    bounds: [f32; 5],
) -> [f32; 5] {
    let empty = check_empty_rows(label, actual[0], Some(actual[2]), expected[1]);
    let mut worst = [0.0f32; 5];
    for (i, ((name, (a, e)), bound)) in PARTS
        .into_iter()
        .zip(actual.into_iter().zip(expected))
        .zip(bounds)
        .enumerate()
    {
        let (abs, rel) = errors(a, e);
        println!(
            "{label} {name}: max abs {abs:.3e}, relative {rel:.3e} (tolerance {bound:.1e}){}",
            if i == 1 && empty > 0 {
                format!(", {empty} queries seeing no key")
            } else {
                String::new()
            }
        );
        assert!(
            rel <= bound,
            "{label} {name}: relative error {rel:.3e} > {bound:.1e}"
        );
        worst[i] = rel;
    }
    worst
}

/// What one call gave with gradients: `[o, lse, dq, dk, dv]`, the dtypes the contract's.
fn run(adapter: &dyn Adapter, label: &str, call: &Attention) -> Option<[HostTensor; 5]> {
    let dtype = call.q.dtype;
    let out = ran(adapter, label, adapter.attention(call))?;
    let grads = out.grads.expect("the gradients asked for");
    let results = [out.out, out.lse, grads.q, grads.k, grads.v];
    let dtypes = results.each_ref().map(|result| result.dtype);
    assert_eq!(
        dtypes,
        [dtype, DType::F32, dtype, dtype, dtype],
        "{label}: the output and gradients in the inputs' dtype, the log-sum-exp in f32"
    );
    Some(results)
}

/// Whether `adapter`'s device has `dtype`, saying so where it has not.
fn has(adapter: &dyn Adapter, dtype: DType) -> bool {
    let has = adapter.has(dtype);
    if !has {
        println!("skipped: {} has no {dtype:?}", adapter.describe());
    }
    has
}

/// Every record in `dtype`, with gradients, the query gradient summed as `deterministic` says,
/// against torch within `bounds`; each line starts `<label> <record>`. Returns the largest
/// relative errors, `[o, lse, dq, dk, dv]`, over the causal records and over the others.
///
/// # Panics
///
/// If a result exceeds its bound, a query that sees no key gives anything but zeros, or no record
/// that was compared has such a query.
pub fn matches_torch(
    adapter: &dyn Adapter,
    label: &str,
    dtype: DType,
    deterministic: bool,
    bounds: Bounds,
) -> Outcome<[[f32; 5]; 2]> {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let mut worst = [[0.0f32; 5]; 2];
    if !has(adapter, dtype) {
        return tally.outcome(worst);
    }
    let mut empty = 0;
    let mut compared = 0;
    for record in records() {
        let label = format!("{label} {}", record.name);
        let call = record.call(dtype, deterministic, true);
        let Some(actual) = run(adapter, &label, &call) else {
            continue;
        };
        let (set, bounds) = match record.causal {
            true => (0, bounds.causal),
            false => (1, bounds.masks),
        };
        let errors = check(
            &label,
            actual.each_ref(),
            record.expected.each_ref(),
            bounds,
        );
        for (w, e) in worst[set].iter_mut().zip(errors) {
            *w = w.max(e);
        }
        empty += check_empty_rows(&label, &actual[0], Some(&actual[2]), &record.expected[1]);
        compared += 1;
    }
    assert!(
        compared == 0 || empty > 0,
        "no record has a query that sees no key"
    );
    tally.outcome(worst)
}

/// A call without gradients gives the output and log-sum-exp of the gradient run: two records with
/// queries that see no key (key padding with causality, packed sequences) in f32, against torch
/// within the mask records' bounds.
pub fn without_gradients(adapter: &dyn Adapter, label: &str, bounds: Bounds) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    for name in [
        "padding_causal_b2_gqa2x1_d64_t26",
        "packed_causal_gqa2x1_d64",
    ] {
        let record = Record::load(name);
        let label = format!("{label} no gradient {name}");
        let call = record.call(DType::F32, true, false);
        let Some(out) = ran(adapter, &label, adapter.attention(&call)) else {
            continue;
        };
        assert!(out.grads.is_none(), "{label}: gradients not asked for");
        check_empty_rows(&label, &out.out, None, &record.expected[1]);
        for (i, actual) in [out.out, out.lse].iter().enumerate() {
            let (abs, rel) = errors(actual, &record.expected[i]);
            let bound = bounds.masks[i];
            println!(
                "{label} {}: max abs {abs:.3e}, relative {rel:.3e} (tolerance {bound:.1e})",
                PARTS[i]
            );
            assert!(
                rel <= bound,
                "{label} {}: relative error {rel:.3e} > {bound:.1e}",
                PARTS[i]
            );
        }
    }
    tally.outcome(())
}

/// Deterministic values in about [-2, 2] of `shape`, rounded to bf16 so that every dtype reads
/// the same, in f32.
pub fn seeded(shape: &[usize], seed: u64) -> HostTensor {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let values = (0..shape.iter().product::<usize>())
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let uniform = (state >> 40) as f32 / (1u64 << 24) as f32;
            round_to_bf16((uniform - 0.5) * 4.0)
        })
        .collect();
    HostTensor::f32(shape.to_vec(), values)
}

/// `x` rounded to the nearest bf16, ties to even (`x` finite).
fn round_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// Rows `range` of dimension 1 of `[B, T, ...]` for batch element `b` alone: `[1, len, ...]`.
fn slice(tensor: &HostTensor, b: usize, range: std::ops::Range<usize>) -> HostTensor {
    let t = tensor.shape[1];
    let inner: usize = tensor.shape[2..].iter().product();
    let at = (b * t + range.start) * inner;
    let mut shape = tensor.shape.clone();
    (shape[0], shape[1]) = (1, range.len());
    let values = tensor.values[at..at + range.len() * inner].to_vec();
    HostTensor::new(tensor.dtype, shape, values)
}

/// The largest absolute difference between `a` and `b`.
fn max_difference(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "lengths differ");
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

/// Decoding in f32 is attention with one query after every key: without cache lengths, with the
/// default options and with a window of 7 keys before the query, within `bounds[0]` of
/// [`Adapter::attention`]'s causal output; with the lengths 50, 17 and 1 of three sequences and a
/// window of 9, each sequence's output within `bounds[1]` of attention with one query over its
/// own keys. The bounds are absolute differences: dash-ops' reference on Flex decodes to the bit
/// without lengths, and within 1e-6 with them.
///
/// # Panics
///
/// If a difference exceeds its bound.
pub fn decoding_is_attention_with_one_query(
    adapter: &dyn Adapter,
    label: &str,
    bounds: [f32; 2],
) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let [b, s, hq, hkv, d] = [3, 50, 4, 2, 64];
    let q = seeded(&[b, 1, hq, d], 400);
    let k = seeded(&[b, s, hkv, d], 401);
    let v = seeded(&[b, s, hkv, d], 402);
    let step_q = HostTensor::f32(vec![b, hq, d], q.values.clone());
    let attention = |q: HostTensor, k: HostTensor, v: HostTensor, options| Attention {
        q,
        k,
        v,
        mask: AttentionMask::None,
        options,
        grad: None,
    };
    for options in [
        AttentionOptions::default(),
        AttentionOptions::default().with_window(7, 0),
    ] {
        let case = format!("{label} decode {options:?}");
        let decode = AttentionDecode {
            q: step_q.clone(),
            k: k.clone(),
            v: v.clone(),
            lengths: None,
            options,
        };
        let Some(step) = ran(adapter, &case, adapter.attention_decode(&decode)) else {
            continue;
        };
        assert_eq!(step.shape, [b, hq, d], "decoding returns [B, Hq, D]");
        let call = attention(q.clone(), k.clone(), v.clone(), options.with_causal(true));
        let Some(full) = ran(adapter, &case, adapter.attention(&call)) else {
            continue;
        };
        let difference = max_difference(&step.values, &full.out.values);
        println!("{case}: decode against attention, max abs {difference:.3e}");
        assert!(
            difference <= bounds[0],
            "{case}: decode and attention differ by {difference}"
        );
    }

    // Sequence `b` holding `lengths[b]` keys: attention over the first `lengths[b]` keys alone.
    let lengths = [50, 17, 1];
    let options = AttentionOptions::default().with_window(9, 0);
    let case = format!("{label} decode, lengths {lengths:?}, window 9");
    let decode = AttentionDecode {
        q: step_q,
        k: k.clone(),
        v: v.clone(),
        lengths: Some(lengths.to_vec()),
        options,
    };
    if let Some(step) = ran(adapter, &case, adapter.attention_decode(&decode)) {
        for (bi, len) in lengths.into_iter().enumerate() {
            let len = len as usize;
            let call = attention(
                slice(&q, bi, 0..1),
                slice(&k, bi, 0..len),
                slice(&v, bi, 0..len),
                options,
            );
            let Some(full) = ran(adapter, &case, adapter.attention(&call)) else {
                continue;
            };
            let got = &step.values[bi * hq * d..(bi + 1) * hq * d];
            let difference = max_difference(got, &full.out.values);
            println!("{case}: sequence {bi} of {len} keys, max abs {difference:.3e}");
            assert!(
                difference <= bounds[1],
                "{case}: sequence {bi} of {len} keys: decode differs by {difference}"
            );
        }
    }
    tally.outcome(())
}
