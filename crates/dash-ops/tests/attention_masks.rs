//! Attention with every mask (`dash_ops::attention`): no mask with `Tq != Tk`, causality within a
//! sliding window, a window without causality, key padding with and without causality, and
//! packed variable-length sequences, with grouped key-value heads and queries that see no key.
//!
//! - The reference and its written-out backward pass against torch's
//!   `scaled_dot_product_attention` in float64 with an explicit boolean mask (records in
//!   `tests/fixtures/attention/`; packed sequences recorded as one call per sequence).
//! - On a CubeCL device (`tests/common`): the kernels in bf16 and f16 against the same records,
//!   and against the reference on Flex at shapes spanning several tiles and sequences, sequence
//!   boundaries and window edges inside tiles; decoding with per-sequence cache lengths and a
//!   window. Where the device lacks what the kernels ask the reference runs, and each test
//!   asserts which ran.
//! - Causal attention over a sequence in two segments, as training against a key-value cache
//!   runs it: the second segment's queries against both segments' keys (`Tq < Tk`) give the
//!   whole sequence's rows, and the two backward passes' key and value gradients add up to the
//!   whole sequence's, on Flex and on a CubeCL device.
//!
//!   cargo test --release -p dash-ops --test attention_masks
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test attention_masks
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test attention_masks
//!   DASH_OPS_DEVICE=cpu cargo test --release -p dash-ops --features cpu --test attention_masks
//!
//! The records hold inputs drawn from a normal distribution and rounded to bf16, so that the
//! same inputs feed f32, bf16 and f16 runs, in torch's order (`[B, H, T, D]`), which [`read`]
//! swaps into the module's `[B, T, H, D]`. A query that sees no key is recorded as the module
//! defines it: zeros, a log-sum-exp of `-inf`, a zero query gradient. Errors are the largest
//! absolute difference and that over the largest reference magnitude; log-sum-exps of `-inf`
//! must be `-inf` exactly, and their rows' outputs and query gradients exactly zero.

mod common;

use burn::backend::{DType, FloatDType, bf16};
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use dash_ops::attention::{
    AttentionMask, AttentionOptions, attention_decode, attention_with_lse, probe,
};
use safetensors::{Dtype, SafeTensors};
use std::sync::{Mutex, MutexGuard};

/// The probe counters are global: tests that read them run one at a time.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A host tensor: values and shape.
#[derive(Clone, Debug)]
struct Host {
    values: Vec<f32>,
    shape: Vec<usize>,
}

impl Host {
    fn on<const D: usize>(&self, device: &Device, dtype: DType) -> Tensor<D> {
        let data = TensorData::new(self.values.clone(), self.shape.clone());
        Tensor::<D>::from_data(data, device).cast(dtype)
    }

    fn from_tensor<const D: usize>(tensor: Tensor<D>) -> Self {
        let shape = tensor.dims().to_vec();
        let values = tensor
            .cast(FloatDType::F32)
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32 data");
        Self { values, shape }
    }

    /// Positions `range` of the time, dimension 1 of `[B, T, ...]`.
    fn rows(&self, range: std::ops::Range<usize>) -> Self {
        let [b, t] = [self.shape[0], self.shape[1]];
        let inner: usize = self.shape[2..].iter().product();
        let mut values = Vec::with_capacity(b * range.len() * inner);
        for i in 0..b {
            let at = (i * t + range.start) * inner;
            values.extend_from_slice(&self.values[at..at + range.len() * inner]);
        }
        let mut shape = self.shape.clone();
        shape[1] = range.len();
        Self { values, shape }
    }

    /// `self` and then `other` along the time, dimension 1 of `[B, T, ...]`.
    fn then(&self, other: &Self) -> Self {
        let [b, t1, t2] = [self.shape[0], self.shape[1], other.shape[1]];
        let inner: usize = self.shape[2..].iter().product();
        let mut values = Vec::with_capacity(self.values.len() + other.values.len());
        for i in 0..b {
            values.extend_from_slice(&self.values[i * t1 * inner..(i + 1) * t1 * inner]);
            values.extend_from_slice(&other.values[i * t2 * inner..(i + 1) * t2 * inner]);
        }
        let mut shape = self.shape.clone();
        shape[1] = t1 + t2;
        Self { values, shape }
    }

    /// `self` with `head` added to its first positions along the time, in f32.
    fn plus_head(&self, head: &Self) -> Self {
        let [b, t, th] = [self.shape[0], self.shape[1], head.shape[1]];
        let inner: usize = self.shape[2..].iter().product();
        let mut values = self.values.clone();
        for i in 0..b {
            for j in 0..th * inner {
                values[i * t * inner + j] += head.values[i * th * inner + j];
            }
        }
        Self {
            values,
            shape: self.shape.clone(),
        }
    }

    /// Dimensions 1 and 2 swapped, the values moved with them: `[B, H, T, ...]` to
    /// `[B, T, H, ...]` and back.
    fn swap_1_2(&self) -> Self {
        let [b, n1, n2] = [self.shape[0], self.shape[1], self.shape[2]];
        let inner: usize = self.shape[3..].iter().product();
        let mut values = Vec::with_capacity(self.values.len());
        for i in 0..b {
            for j in 0..n2 {
                for k in 0..n1 {
                    let at = ((i * n1 + k) * n2 + j) * inner;
                    values.extend_from_slice(&self.values[at..at + inner]);
                }
            }
        }
        let mut shape = self.shape.clone();
        shape.swap(1, 2);
        Self { values, shape }
    }
}

/// The mask of a call, on the host.
#[derive(Clone, Debug)]
enum Mask {
    None,
    /// `[B, Tk]`, `true` where a key is left out.
    KeyPadding(Vec<bool>, [usize; 2]),
    /// The queries' and the keys' offsets.
    Packed(Vec<i32>, Vec<i32>),
}

impl Mask {
    fn on(&self, device: &Device) -> AttentionMask {
        match self {
            Mask::None => AttentionMask::None,
            Mask::KeyPadding(values, shape) => {
                AttentionMask::KeyPadding(Tensor::<2, Bool>::from_data(
                    TensorData::new(values.clone(), shape.to_vec()),
                    device,
                ))
            }
            Mask::Packed(q, k) => {
                let offsets = |cu: &Vec<i32>| {
                    Tensor::<1, Int>::from_data(TensorData::new(cu.clone(), [cu.len()]), device)
                };
                AttentionMask::Packed {
                    cu_seqlens_q: offsets(q),
                    cu_seqlens_k: offsets(k),
                }
            }
        }
    }
}

/// One call: inputs, the mask and options, and (for a record) torch's results.
#[derive(Clone)]
struct Case {
    name: String,
    q: Host,
    k: Host,
    v: Host,
    grad: Host,
    mask: Mask,
    options: AttentionOptions,
}

impl Case {
    /// `[B, Tq, Tk, Hq, Hkv, D]`.
    #[cfg(feature = "cubecl")]
    fn dims(&self) -> [usize; 6] {
        let (q, k) = (&self.q.shape, &self.k.shape);
        [q[0], q[1], k[1], q[2], k[2], q[3]]
    }
}

/// What a run produces: the output, the log-sum-exp and the three input gradients.
#[derive(Debug)]
struct Results {
    o: Host,
    lse: Host,
    dq: Host,
    dk: Host,
    dv: Host,
}

/// Tensor `name` of a record, its heads and time swapped into the module's layout.
fn read(tensors: &SafeTensors, name: &str) -> Host {
    let view = tensors.tensor(name).expect("tensor in record");
    let bytes = view.data();
    let values = match view.dtype() {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        other => panic!("unexpected dtype {other:?}"),
    };
    Host {
        values,
        shape: view.shape().to_vec(),
    }
    .swap_1_2()
}

/// The record `name`: its inputs, mask, options and torch's results.
fn record(name: &str) -> (Case, Results) {
    let path = format!(
        "{}/tests/fixtures/attention/{name}.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let (_, metadata) = SafeTensors::read_metadata(&bytes).expect("metadata");
    let metadata = metadata.metadata().clone().expect("metadata");
    let meta = |key: &str| {
        metadata
            .get(key)
            .unwrap_or_else(|| panic!("{key} in metadata"))
    };
    let tensors = SafeTensors::deserialize(&bytes).expect("safetensors");
    let reach = |key: &str| meta(key).parse::<i64>().expect(key);
    let q = read(&tensors, "q");
    // The records' scale is `1 / sqrt(D)`: leave it to the default, so that the default is
    // held to torch too.
    let scale = meta("scale").parse::<f32>().expect("scale");
    let d = q.shape[3];
    let mut options = AttentionOptions::default().with_causal(meta("causal") == "true");
    if scale != (1.0 / (d as f64).sqrt()) as f32 {
        options = options.with_scale(scale);
    }
    let (left, right) = (reach("window_left"), reach("window_right"));
    if left >= 0 || right >= 0 {
        let side = |r: i64| if r < 0 { usize::MAX } else { r as usize };
        options = options.with_window(side(left), side(right));
    }
    let mask = match meta("kind").as_str() {
        "none" => Mask::None,
        "padding" => {
            let view = tensors.tensor("key_padding").expect("key_padding");
            assert_eq!(view.dtype(), Dtype::U8);
            let shape = view.shape();
            Mask::KeyPadding(
                view.data().iter().map(|&x| x != 0).collect(),
                [shape[0], shape[1]],
            )
        }
        "packed" => {
            let offsets = |key: &str| {
                let view = tensors.tensor(key).expect("offsets");
                assert_eq!(view.dtype(), Dtype::I64);
                view.data()
                    .chunks_exact(8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as i32)
                    .collect()
            };
            Mask::Packed(offsets("cu_seqlens_q"), offsets("cu_seqlens_k"))
        }
        other => panic!("unknown mask kind {other}"),
    };
    let case = Case {
        name: name.to_string(),
        q,
        k: read(&tensors, "k"),
        v: read(&tensors, "v"),
        grad: read(&tensors, "do"),
        mask,
        options,
    };
    let expected = Results {
        o: read(&tensors, "o"),
        lse: read(&tensors, "lse"),
        dq: read(&tensors, "dq"),
        dk: read(&tensors, "dk"),
        dv: read(&tensors, "dv"),
    };
    (case, expected)
}

/// The records, one per mask kind (the generator's comments say what each holds).
const RECORDS: [&str; 8] = [
    "none_b2_gqa2x1_d64_tq19_tk33",
    "causal_b2_gqa4x2_d64_tq17_tk41",
    "causal_window9_gqa3x1_d64_t40",
    "window6x4_b2_gqa2x1_d64_tq24_tk37",
    "padding_b3_gqa2x1_d64_t29",
    "padding_causal_b2_gqa2x1_d64_t26",
    "packed_causal_gqa2x1_d64",
    "packed_cross_causal_window5_gqa2x1_d64",
];

/// The forward with its log-sum-exp, then the backward of `sum(o * grad)`, on `device` (an
/// autodiff device) with inputs and the output gradient in `dtype`.
fn run(device: &Device, case: &Case, dtype: DType) -> Results {
    let q = case.q.on::<4>(device, dtype).detach().require_grad();
    let k = case.k.on::<4>(device, dtype).detach().require_grad();
    let v = case.v.on::<4>(device, dtype).detach().require_grad();
    let grad = case.grad.on::<4>(device, dtype);
    let (o, lse) = attention_with_lse(
        q.clone(),
        k.clone(),
        v.clone(),
        case.mask.on(device),
        case.options,
    );
    let grads = (o.clone() * grad).sum().backward();
    Results {
        o: Host::from_tensor(o),
        lse: Host::from_tensor(lse),
        dq: Host::from_tensor(q.grad(&grads).expect("query gradient")),
        dk: Host::from_tensor(k.grad(&grads).expect("key gradient")),
        dv: Host::from_tensor(v.grad(&grads).expect("value gradient")),
    }
}

/// The largest absolute difference, and that over the largest reference magnitude, over the
/// entries where the reference is finite; where it is `-inf` (the log-sum-exp of a query that
/// sees no key) the result must be `-inf` too.
fn errors(actual: &Host, expected: &Host) -> (f32, f32) {
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

/// The queries that see no key, by their log-sum-exp: their outputs and query gradients must be
/// exactly zero. Returns how many there are.
fn check_empty_rows(label: &str, actual: &Results, expected: &Results) -> usize {
    let d = actual.o.shape[3];
    let mut empty = 0;
    for (row, &lse) in expected.lse.values.iter().enumerate() {
        if lse != f32::NEG_INFINITY {
            continue;
        }
        empty += 1;
        let range = row * d..(row + 1) * d;
        assert!(
            actual.o.values[range.clone()].iter().all(|&x| x == 0.0),
            "{label}: query {row} sees no key, its output is not zero"
        );
        assert!(
            actual.dq.values[range].iter().all(|&x| x == 0.0),
            "{label}: query {row} sees no key, its gradient is not zero"
        );
    }
    empty
}

/// Compares every result with the reference: each relative error must stay under `tolerance`
/// (one per result, in the order o, lse, dq, dk, dv). Prints the errors; returns the largest
/// relative error of each result.
fn check(label: &str, actual: &Results, expected: &Results, tolerance: [f32; 5]) -> [f32; 5] {
    let pairs = [
        ("o", &actual.o, &expected.o),
        ("lse", &actual.lse, &expected.lse),
        ("dq", &actual.dq, &expected.dq),
        ("dk", &actual.dk, &expected.dk),
        ("dv", &actual.dv, &expected.dv),
    ];
    let empty = check_empty_rows(label, actual, expected);
    let mut worst = [0.0f32; 5];
    for (i, ((name, a, e), tol)) in pairs.into_iter().zip(tolerance).enumerate() {
        let (abs, rel) = errors(a, e);
        println!(
            "{label} {name}: max abs {abs:.3e}, relative {rel:.3e} (tolerance {tol:.1e}){}",
            if i == 1 && empty > 0 {
                format!(", {empty} queries seeing no key")
            } else {
                String::new()
            }
        );
        assert!(
            rel <= tol,
            "{label} {name}: relative error {rel:.3e} > {tol:.1e}"
        );
        worst[i] = rel;
    }
    worst
}

/// Five relative errors, for the output.
#[cfg(feature = "cubecl")]
fn format_errors(errors: [f32; 5]) -> String {
    errors.map(|e| format!("{e:.3e}")).join(", ")
}

/// Tolerances on the error relative to the largest reference magnitude, in the order o, lse,
/// dq, dk, dv, each just above the largest error measured over every record and every build
/// that uses it (an RTX 5070 Ti, CUDA 13; lavapipe and OpenGL over D3D12 through wgpu; CubeCL's
/// CPU runtime). In f32 the reference agrees with torch's float64 to f32 rounding.
const F32_TOLERANCE: [f32; 5] = [3.2e-7, 1.2e-7, 3.2e-7, 4.1e-7, 3.5e-7];
/// The reference run on a CubeCL device in f32, against torch: its products and reductions are
/// Burn's CubeCL kernels; on OpenGL its log-sum-exp reaches 1.33e-7. With autotune those kernels
/// change from run to run (the value gradient reached 4.0e-7 in one of two runs), so the bounds are
/// then a few roundings, as for checkpointing.
#[cfg(feature = "cubecl")]
const CUBE_F32_TOLERANCE: [f32; 5] = if cfg!(feature = "autotune") {
    [1e-6, 2e-7, 1e-6, 1e-6, 1e-6]
} else {
    [5.5e-7, 1.4e-7, 4.3e-7, 4.1e-7, 3.4e-7]
};
/// In bf16 the error is the output's own rounding (8 significant bits) plus the probabilities'
/// rounding to bf16 before the value product; the log-sum-exp is accumulated in f32 throughout.
/// The query gradient is summed over the key blocks in a fixed order, or with atomics in whatever
/// order the kernels run: its bounds here and below are just above the largest of both over
/// repeated runs of each CUDA build, where the largest errors came out the same in every run.
#[cfg(feature = "cubecl")]
const BF16_TOLERANCE: [f32; 5] = [3.2e-3, 1.2e-7, 4.3e-3, 4.4e-3, 3.6e-3];
#[cfg(feature = "cubecl")]
const F16_TOLERANCE: [f32; 5] = [3.9e-4, 1.2e-7, 7.4e-4, 6.9e-4, 6.4e-4];
/// Decoding in bf16 and in f16 against the f32 reference.
#[cfg(feature = "cubecl")]
const DECODE_BF16_TOLERANCE: f32 = 2.2e-3;
#[cfg(feature = "cubecl")]
const DECODE_F16_TOLERANCE: f32 = 2.6e-4;
/// The kernels against the f32 reference at shapes of several tiles, in bf16 and in f16, the
/// backward's products split and whole: in bf16, dq reaches 7.31e-3 at the queries that see few
/// keys (the start of a causal sequence), whose `delta = rowsum(dO * O)` is formed from the output
/// rounded to bf16, as in FlashAttention-2; the same queries run alone through the plain causal
/// kernels give the same bits.
#[cfg(feature = "cubecl")]
const TILED_BF16_TOLERANCE: [f32; 5] = [3.3e-3, 4.4e-7, 7.7e-3, 4.6e-3, 3.9e-3];
#[cfg(feature = "cubecl")]
const TILED_F16_TOLERANCE: [f32; 5] = [4.1e-4, 4.4e-7, 7.3e-4, 6.8e-4, 4.9e-4];
/// A head dimension the kernels do not take: the reference on the device in bf16 and in f16
/// against the f32 reference, the error the outputs' and gradients' rounding.
#[cfg(feature = "cubecl")]
const REFERENCE_BF16_TOLERANCE: [f32; 5] = [2.1e-3, 1e-7, 2.7e-3, 2.9e-3, 1.7e-3];
#[cfg(feature = "cubecl")]
const REFERENCE_F16_TOLERANCE: [f32; 5] = [2.6e-4, 1e-7, 3.6e-4, 3.7e-4, 3.2e-4];
/// Two causal segments composed by the kernels against the whole sequence, by the kernels and by
/// the f32 reference: each segment's gradients are rounded to the dtype before they are added.
#[cfg(feature = "cubecl")]
const SEGMENTS_BF16_TOLERANCE: [f32; 5] = [3.2e-3, 2.3e-7, 4.7e-3, 4.7e-3, 4.9e-3];
#[cfg(feature = "cubecl")]
const SEGMENTS_F16_TOLERANCE: [f32; 5] = [5.2e-4, 2.3e-7, 5.3e-4, 7.1e-4, 6.4e-4];
/// Two causal segments composed against the whole sequence, both in f32 on Flex.
#[cfg(feature = "flex")]
const SEGMENTS_F32_TOLERANCE: [f32; 5] = [1.9e-7, 9e-8, 2.6e-7, 6.1e-7, 7.6e-7];

#[cfg(feature = "flex")]
#[test]
fn reference_and_its_backward_match_torch_for_every_mask_on_flex() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    let mut empty = 0;
    for name in RECORDS {
        let (case, expected) = record(name);
        let runs = probe::backward_runs();
        let actual = run(&device, &case, DType::F32);
        assert!(
            probe::backward_runs() > runs,
            "the written-out backward did not run"
        );
        check(
            &format!("flex f32 {name}"),
            &actual,
            &expected,
            F32_TOLERANCE,
        );
        empty += check_empty_rows(name, &actual, &expected);
    }
    assert!(empty > 0, "no record has a query that sees no key");
}

/// Without autodiff, and through Autodiff without a gradient, the output and log-sum-exp are
/// those of the gradient run.
#[cfg(feature = "flex")]
#[test]
fn forward_without_gradients_on_flex() {
    for name in [
        "padding_causal_b2_gqa2x1_d64_t26",
        "packed_causal_gqa2x1_d64",
    ] {
        let (case, expected) = record(name);
        for device in [Device::flex(), Device::flex().autodiff()] {
            let (o, lse) = attention_with_lse(
                case.q.on::<4>(&device, DType::F32),
                case.k.on::<4>(&device, DType::F32),
                case.v.on::<4>(&device, DType::F32),
                case.mask.on(&device),
                case.options,
            );
            let (_, rel) = errors(&Host::from_tensor(o), &expected.o);
            assert!(rel <= F32_TOLERANCE[0], "{name}: o {rel:.3e}");
            let (_, rel) = errors(&Host::from_tensor(lse), &expected.lse);
            assert!(rel <= F32_TOLERANCE[1], "{name}: lse {rel:.3e}");
        }
    }
}

/// Deterministic values in about [-2, 2], rounded to bf16 so that every dtype reads the same.
fn host(shape: &[usize], seed: u64) -> Host {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let values = (0..shape.iter().product::<usize>())
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let uniform = (state >> 40) as f32 / (1u64 << 24) as f32;
            bf16::from_f32((uniform - 0.5) * 4.0).to_f32()
        })
        .collect();
    Host {
        values,
        shape: shape.to_vec(),
    }
}

/// Deterministic booleans, `true` with probability `p`.
#[cfg(feature = "cubecl")]
fn coins(n: usize, p: f32, seed: u64) -> Vec<bool> {
    let mut state = seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) < p
        })
        .collect()
}

/// A case of seeded inputs, `[B, Hq, Hkv, Tq, Tk, D]`, with `mask` and `options`.
fn seeded(
    name: &str,
    [b, hq, hkv, tq, tk, d]: [usize; 6],
    mask: Mask,
    options: AttentionOptions,
    seed: u64,
) -> Case {
    Case {
        name: name.to_string(),
        q: host(&[b, tq, hq, d], seed),
        k: host(&[b, tk, hkv, d], seed + 1),
        v: host(&[b, tk, hkv, d], seed + 2),
        grad: host(&[b, tq, hq, d], seed + 3),
        mask,
        options,
    }
}

/// Offsets of sequences of `lengths`.
#[cfg(feature = "cubecl")]
fn offsets(lengths: &[usize]) -> Vec<i32> {
    let mut cu = vec![0];
    for len in lengths {
        cu.push(cu.last().unwrap() + *len as i32);
    }
    cu
}

/// Shapes spanning several query and key tiles (128 and 64 queries, 64 or 32 keys, 32-key
/// blocks), with partial last tiles, sequence boundaries and window edges inside tiles, every
/// mask kind and their combinations with causality and a window, and queries that see no key.
#[cfg(feature = "cubecl")]
fn tiled_cases() -> Vec<Case> {
    let causal = AttentionOptions::causal();
    let full = AttentionOptions::default();
    let mut padding = coins(3 * 200, 0.3, 1);
    // The third sequence is all padding: none of its queries sees a key.
    padding[400..].fill(true);
    // Left padding of 37 and 100 keys, and scattered padding past it.
    let mut left_padding = coins(2 * 170, 0.2, 2);
    left_padding[..37].fill(true);
    left_padding[170..270].fill(true);
    let self_lengths = [130, 1, 0, 257, 64, 77];
    let (q_lengths, k_lengths) = ([50, 3, 140, 0, 31], [70, 99, 140, 9, 20]);
    vec![
        seeded(
            "none, Tq < Tk",
            [2, 8, 2, 200, 333, 128],
            Mask::None,
            full,
            10,
        ),
        seeded(
            "causal, window 100",
            [1, 8, 2, 300, 300, 256],
            Mask::None,
            causal.with_window(100, 0),
            20,
        ),
        seeded(
            "window 70 x 45, Tq < Tk",
            [1, 6, 2, 150, 260, 64],
            Mask::None,
            full.with_window(70, 45),
            30,
        ),
        seeded(
            "causal, Tq > Tk",
            [1, 4, 1, 230, 90, 64],
            Mask::None,
            causal,
            40,
        ),
        // Tk - Tq = 30: the diagonal crosses the 32-key blocks of dK/dV, the halves of dQ and the
        // tiles of the forward one key from where their decision to mask flips.
        seeded(
            "causal, Tk - Tq = 30",
            [1, 4, 2, 98, 128, 64],
            Mask::None,
            causal,
            45,
        ),
        seeded(
            "causal, Tk - Tq = 30",
            [1, 8, 2, 66, 96, 256],
            Mask::None,
            causal,
            46,
        ),
        // The queries that see a key block end one past a 32-query step (window 33), and start
        // one before one (a window reaching one key ahead): the edges of dK/dV's query range.
        seeded(
            "causal, window 33",
            [1, 4, 2, 160, 160, 64],
            Mask::None,
            causal.with_window(33, 0),
            47,
        ),
        seeded(
            "window 20 x 1",
            [1, 4, 2, 160, 160, 64],
            Mask::None,
            full.with_window(20, 1),
            48,
        ),
        seeded(
            "key padding",
            [3, 4, 2, 200, 200, 128],
            Mask::KeyPadding(padding, [3, 200]),
            full,
            50,
        ),
        seeded(
            "key padding, causal, window 50",
            [2, 8, 2, 170, 170, 256],
            Mask::KeyPadding(left_padding, [2, 170]),
            causal.with_window(50, 0),
            60,
        ),
        seeded(
            "packed, causal",
            [1, 8, 2, 529, 529, 256],
            Mask::Packed(offsets(&self_lengths), offsets(&self_lengths)),
            causal,
            70,
        ),
        seeded(
            "packed, not causal",
            [1, 4, 4, 529, 529, 64],
            Mask::Packed(offsets(&self_lengths), offsets(&self_lengths)),
            full,
            80,
        ),
        seeded(
            "packed cross, causal, window 40",
            [1, 6, 2, 224, 338, 128],
            Mask::Packed(offsets(&q_lengths), offsets(&k_lengths)),
            causal.with_window(40, 0),
            90,
        ),
    ]
}

/// One causal sequence run whole and in two segments, as training against a key-value cache runs
/// it: segment 1 alone (queries, keys and values `0..T1`), then segment 2's queries `T1..T`
/// against the keys and values of both (`Tq = T - T1 < Tk = T`, bottom-right aligned, so that
/// query `T1 + i` sees the keys up to `T1 + i`), each with its own backward pass. Returns the whole
/// sequence's results and the segments' composed: their outputs, log-sum-exps and query
/// gradients end to end, and segment 2's key and value gradients with segment 1's added to their
/// first `T1` positions.
fn segments(device: &Device, dtype: DType, case: &Case, t1: usize) -> (Results, Results) {
    let t = case.q.shape[1];
    let whole = run(device, case, dtype);
    let part = |q: std::ops::Range<usize>, kv: std::ops::Range<usize>| Case {
        name: format!("{} {q:?} against {kv:?}", case.name),
        q: case.q.rows(q.clone()),
        k: case.k.rows(kv.clone()),
        v: case.v.rows(kv),
        grad: case.grad.rows(q),
        mask: Mask::None,
        options: case.options,
    };
    let first = run(device, &part(0..t1, 0..t1), dtype);
    let second = run(device, &part(t1..t, 0..t), dtype);
    let composed = Results {
        o: first.o.then(&second.o),
        lse: first.lse.then(&second.lse),
        dq: first.dq.then(&second.dq),
        dk: second.dk.plus_head(&first.dk),
        dv: second.dv.plus_head(&first.dv),
    };
    (whole, composed)
}

/// Causal sequences `[B, Hq, Hkv, T, D]` and where they are cut, `T1`: within one tile, and
/// across several with the cut inside a tile or at its edge.
fn segment_cases() -> [(Case, usize); 3] {
    let causal = AttentionOptions::causal();
    let case = |name: &str, [b, hq, hkv, t, d]: [usize; 5], seed| {
        seeded(name, [b, hq, hkv, t, t, d], Mask::None, causal, seed)
    };
    [
        (case("59 = 23 + 36", [2, 4, 2, 59, 64], 500), 23),
        (case("263 = 150 + 113", [1, 8, 2, 263, 256], 510), 150),
        (case("264 = 64 + 200", [1, 6, 3, 264, 128], 520), 64),
    ]
}

/// The two segments composed against the whole sequence, in f32 on Flex: the same to f32
/// rounding (the products and sums run over the same keys in other orders).
#[cfg(feature = "flex")]
#[test]
fn causal_segments_compose_into_the_whole_sequence_on_flex() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    for (case, t1) in segment_cases() {
        let (whole, composed) = segments(&device, DType::F32, &case, t1);
        check(
            &format!("flex f32 segments {}", case.name),
            &composed,
            &whole,
            SEGMENTS_F32_TOLERANCE,
        );
    }
}

/// The same checks on a CubeCL device, any runtime: the one `DASH_OPS_DEVICE` names. Where the
/// device lacks what the kernels ask, the reference runs, and the tests assert that too.
#[cfg(all(feature = "cubecl", feature = "flex"))]
mod cube {
    use super::*;
    use dash_ops::attention::{decode_kernels_take, kernels_take};

    fn device() -> Device {
        common::cube_device().autodiff()
    }

    /// The kernel launches a forward and backward pass over `dims` (`[B, Tq, Tk, Hq, Hkv, D]`)
    /// in `dtype` makes on `device`: one forward and three backward kernels where the kernels
    /// take them, none where the reference runs; [`launches_kernels`] adds the fourth of a
    /// deterministic backward whose products split. Under CUDA the kernels take every bf16 and
    /// f16 call with a head dimension of 64, 128 or 256, whatever the mask.
    fn expected_launches(device: &Device, dtype: DType, dims: [usize; 6]) -> usize {
        let [forward, backward] = kernels_take(device, dtype, dims);
        let takes = matches!(dtype, DType::BF16 | DType::F16) && matches!(dims[5], 64 | 128 | 256);
        if common::runtime_of(device) == Some("cuda") {
            assert_eq!(
                [forward, backward],
                [takes, takes],
                "CUDA turned the kernels away for {dims:?} {dtype:?}"
            );
        }
        usize::from(forward) + 3 * usize::from(backward)
    }

    /// Whether `device` stores `dtype`; the tests in a dtype it does not skip, saying so.
    fn has(device: &Device, dtype: DType) -> bool {
        let has = device.supports_dtype(dtype);
        if !has {
            println!("skipped: {} has no {dtype:?}", common::describe(device));
        }
        has
    }

    /// Asserts that `f` launched exactly `expected` kernels, and one more for each backward pass
    /// whose products split and whose query gradient is the fixed-order one (its key and value
    /// gradients are rounded from accumulators by a kernel the atomic way runs anyway): under
    /// fusion the kernels run when their results are read, so `f` reads them. The passes in one
    /// `f` take their query gradient the same way.
    fn launches_kernels<R>(expected: usize, f: impl FnOnce() -> R) -> R {
        let before = probe::kernel_launches();
        let (splits, atomics) = (probe::split_runs(), probe::atomic_runs());
        let result = f();
        let launched = probe::kernel_launches() - before;
        let split = probe::split_runs() - splits;
        let expected = expected
            + if probe::atomic_runs() == atomics {
                split
            } else {
                0
            };
        assert_eq!(
            launched, expected,
            "the kernels made {launched} launches, expected {expected}"
        );
        result
    }

    #[test]
    fn kernels_match_torch_for_every_mask_in_bf16_and_f16() {
        let _serial = serial();
        let device = device();
        println!("{}", common::describe(&device));
        for (dtype, label, tolerance) in [
            (DType::BF16, "bf16", BF16_TOLERANCE),
            (DType::F16, "f16", F16_TOLERANCE),
        ] {
            if !has(&device, dtype) {
                continue;
            }
            let mut worst = [0.0f32; 5];
            for name in RECORDS {
                let (mut case, expected) = record(name);
                for deterministic in [true, false] {
                    case.options = case.options.with_deterministic(deterministic);
                    let launches = expected_launches(&device, dtype, case.dims());
                    let backward = kernels_take(&device, dtype, case.dims())[1];
                    let (runs, atomics) = (probe::backward_runs(), probe::atomic_runs());
                    let actual = launches_kernels(launches, || run(&device, &case, dtype));
                    assert!(
                        probe::backward_runs() > runs,
                        "the written-out backward did not run"
                    );
                    assert_eq!(
                        probe::atomic_runs() - atomics,
                        usize::from(backward && !deterministic),
                        "the query gradient was not summed the way asked"
                    );
                    let errors = check(
                        &format!("cube {label} deterministic {deterministic} {name}"),
                        &actual,
                        &expected,
                        tolerance,
                    );
                    for (w, e) in worst.iter_mut().zip(errors) {
                        *w = w.max(e);
                    }
                }
            }
            println!(
                "cube {label}, largest over the records (o, lse, dq, dk, dv): {}",
                format_errors(worst)
            );
        }
    }

    #[test]
    fn f32_runs_the_reference_for_every_mask() {
        let _serial = serial();
        let device = device();
        for name in RECORDS {
            let (case, expected) = record(name);
            assert_eq!(expected_launches(&device, DType::F32, case.dims()), 0);
            let actual = launches_kernels(0, || run(&device, &case, DType::F32));
            check(
                &format!("cube f32 {name}"),
                &actual,
                &expected,
                CUBE_F32_TOLERANCE,
            );
        }
    }

    /// Makes the backward's products split each key block's sweep between two cubes, or keep it
    /// whole, while it lives; then the call's size decides again.
    struct ForcedSplit;

    impl ForcedSplit {
        fn new(split: bool) -> Self {
            probe::force_split(Some(split));
            Self
        }
    }

    impl Drop for ForcedSplit {
        fn drop(&mut self) {
            probe::force_split(None);
        }
    }

    /// The kernels against the reference on Flex from the same inputs (rounded to bf16, so that
    /// f16 holds them too), over [`tiled_cases`], in bf16 and in f16, the backward's products
    /// both split between two cubes per key block and whole, its query gradient both in a fixed
    /// order and with atomics.
    #[test]
    fn kernels_match_the_reference_across_tiles_and_sequences() {
        let _serial = serial();
        let device = device();
        let cases = tiled_cases();
        let expected: Vec<Results> = cases
            .iter()
            .map(|case| run(&Device::flex().autodiff(), case, DType::F32))
            .collect();
        for (split, deterministic) in [(false, true), (true, true), (false, false), (true, false)] {
            let _forced = ForcedSplit::new(split);
            let products = match (split, deterministic) {
                (false, true) => "whole",
                (true, true) => "split",
                (false, false) => "whole atomic",
                (true, false) => "split atomic",
            };
            for (dtype, label, tolerance) in [
                (DType::BF16, "bf16", TILED_BF16_TOLERANCE),
                (DType::F16, "f16", TILED_F16_TOLERANCE),
            ] {
                if !has(&device, dtype) {
                    continue;
                }
                let mut worst = [0.0f32; 5];
                for (case, expected) in cases.iter().zip(&expected) {
                    let case = &Case {
                        options: case.options.with_deterministic(deterministic),
                        ..case.clone()
                    };
                    let launches = expected_launches(&device, dtype, case.dims());
                    let backward = kernels_take(&device, dtype, case.dims())[1];
                    let splits = probe::split_runs();
                    let actual = launches_kernels(launches, || run(&device, case, dtype));
                    assert_eq!(
                        probe::split_runs() - splits,
                        usize::from(split && backward),
                        "the products did not run {products}"
                    );
                    let name = format!("cube {label} {products} {} {:?}", case.name, case.dims());
                    let errors = check(&name, &actual, expected, tolerance);
                    for (w, e) in worst.iter_mut().zip(errors) {
                        *w = w.max(e);
                    }
                }
                println!(
                    "{label} {products}, largest over the cases (o, lse, dq, dk, dv): {}",
                    format_errors(worst)
                );
            }
        }
    }

    /// By default the backward gives the same bits from run to run, at a size where 32 key blocks
    /// add to the last queries' gradient, the products split or not: the query gradient is summed
    /// over the key blocks in a fixed order; whole, each key block's sums run in one cube in a
    /// fixed order; split, each key and value gradient is the sum of two halves' sums added to
    /// zero, whose order does not show. With atomics the output and the key and value gradients
    /// are still those bits; the query gradient is summed in whatever order the cubes run, so it
    /// may differ from run to run and from the fixed order's: by a unit in the dtype's last place
    /// where its value is large, and by the f32 sum's reordering, a few millionths of the largest
    /// value, where terms cancel.
    #[test]
    fn the_backward_repeats_bit_for_bit_unless_atomic() {
        let _serial = serial();
        let device = device();
        if !has(&device, DType::BF16) {
            return;
        }
        let causal = AttentionOptions::causal();
        let fixed = seeded(
            "causal",
            [1, 8, 2, 1000, 1000, 128],
            Mask::None,
            causal,
            600,
        );
        let atomic = Case {
            options: causal.with_deterministic(false),
            ..fixed.clone()
        };
        let bits = |h: &Host| h.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for split in [false, true] {
            let _forced = ForcedSplit::new(split);
            let first = run(&device, &fixed, DType::BF16);
            for _ in 0..4 {
                let again = run(&device, &fixed, DType::BF16);
                for (name, a, b) in [
                    ("o", &again.o, &first.o),
                    ("lse", &again.lse, &first.lse),
                    ("dq", &again.dq, &first.dq),
                    ("dk", &again.dk, &first.dk),
                    ("dv", &again.dv, &first.dv),
                ] {
                    assert_eq!(bits(a), bits(b), "{name} differs between runs");
                }
            }
            let largest = first.dq.values.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let (mut differing, mut compared, mut worst) = (0, 0, 0.0f32);
            let mut first_atomic: Option<Vec<u32>> = None;
            let mut moved = false;
            for _ in 0..5 {
                let again = run(&device, &atomic, DType::BF16);
                assert_eq!(
                    bits(&again.dk),
                    bits(&first.dk),
                    "atomic dk is not the fixed order's"
                );
                assert_eq!(
                    bits(&again.dv),
                    bits(&first.dv),
                    "atomic dv is not the fixed order's"
                );
                assert_eq!(
                    bits(&again.o),
                    bits(&first.o),
                    "atomic o is not the fixed order's"
                );
                for (a, b) in again.dq.values.iter().zip(&first.dq.values) {
                    compared += 1;
                    if a.to_bits() != b.to_bits() {
                        differing += 1;
                        let diff = (a - b).abs();
                        worst = worst.max(diff / largest);
                        assert!(
                            diff <= a.abs().max(b.abs()) / 128.0 + 1e-5 * largest,
                            "dq: {a} atomic and {b} in a fixed order"
                        );
                    }
                }
                let now = bits(&again.dq);
                match &first_atomic {
                    None => first_atomic = Some(now),
                    Some(earlier) => moved |= *earlier != now,
                }
            }
            println!(
                "split {split}: o, lse, dq, dk and dv the same bits over 5 runs; with atomics o, \
                 dk and dv those bits again, and dq differs from the fixed order's in {differing} \
                 of {compared} elements over 5 runs, by at most {worst:.2e} of its largest value \
                 (moved between atomic runs: {moved})"
            );
        }
    }

    /// A head dimension the kernels do not take runs the reference, every mask included.
    #[test]
    fn other_head_dimensions_run_the_reference() {
        let _serial = serial();
        let device = device();
        let lengths = [30, 0, 41];
        let case = seeded(
            "packed, D = 96",
            [1, 4, 2, 71, 71, 96],
            Mask::Packed(offsets(&lengths), offsets(&lengths)),
            AttentionOptions::causal(),
            100,
        );
        let expected = run(&Device::flex().autodiff(), &case, DType::F32);
        for (dtype, label, tolerance) in [
            (DType::BF16, "bf16", REFERENCE_BF16_TOLERANCE),
            (DType::F16, "f16", REFERENCE_F16_TOLERANCE),
        ] {
            if !has(&device, dtype) {
                continue;
            }
            assert_eq!(expected_launches(&device, dtype, case.dims()), 0);
            let actual = launches_kernels(0, || run(&device, &case, dtype));
            check(
                &format!("cube {label} D = 96"),
                &actual,
                &expected,
                tolerance,
            );
        }
    }

    /// The two segments composed against the whole sequence, both run by the kernels in bf16 and
    /// in f16, and against the f32 reference on Flex: within the bounds of the dtype (each
    /// segment's gradients are rounded to it before they are added).
    #[test]
    fn causal_segments_compose_into_the_whole_sequence() {
        let _serial = serial();
        let device = device();
        for (dtype, label, tolerance) in [
            (DType::BF16, "bf16", SEGMENTS_BF16_TOLERANCE),
            (DType::F16, "f16", SEGMENTS_F16_TOLERANCE),
        ] {
            if !has(&device, dtype) {
                continue;
            }
            for (case, t1) in segment_cases() {
                let [b, t, _, hq, hkv, d] = case.dims();
                let launches = expected_launches(&device, dtype, [b, t1, t1, hq, hkv, d])
                    + expected_launches(&device, dtype, [b, t - t1, t, hq, hkv, d])
                    + expected_launches(&device, dtype, case.dims());
                let (whole, composed) =
                    launches_kernels(launches, || segments(&device, dtype, &case, t1));
                let reference = run(&Device::flex().autodiff(), &case, DType::F32);
                let name = format!("cube {label} segments {}", case.name);
                check(
                    &format!("{name} against the whole sequence's kernels"),
                    &composed,
                    &whole,
                    tolerance,
                );
                check(
                    &format!("{name} against the whole sequence's f32 reference"),
                    &composed,
                    &reference,
                    tolerance,
                );
            }
        }
    }

    /// Decoding with mixed cache lengths (a sequence holding none, one holding one key, others
    /// cut inside a split), with and without a window, against the reference on Flex.
    #[test]
    fn decode_with_cache_lengths_and_a_window_matches_the_reference() {
        let _serial = serial();
        let device = common::cube_device();
        for (dtype, tolerance) in [
            (DType::BF16, DECODE_BF16_TOLERANCE),
            (DType::F16, DECODE_F16_TOLERANCE),
        ] {
            if has(&device, dtype) {
                decode_cases(&device, dtype, tolerance);
            }
        }
    }

    /// [`decode_with_cache_lengths_and_a_window_matches_the_reference`] in `dtype`.
    fn decode_cases(device: &Device, dtype: DType, tolerance: f32) {
        let flex = Device::flex();
        let cases: [([usize; 5], &[i32], Option<usize>); 4] = [
            ([6, 1000, 8, 2, 256], &[1000, 0, 1, 517, 999, 64], None),
            ([6, 1000, 8, 2, 256], &[1000, 0, 1, 517, 999, 64], Some(100)),
            ([3, 300, 6, 3, 64], &[300, 299, 2], Some(0)),
            ([2, 77, 4, 1, 128], &[77, 40], Some(1000)),
        ];
        for (i, ([b, s, hq, hkv, d], lengths, left)) in cases.into_iter().enumerate() {
            let seed = 300 + 10 * i as u64;
            let q = host(&[b, hq, d], seed);
            let k = host(&[b, s, hkv, d], seed + 1);
            let v = host(&[b, s, hkv, d], seed + 2);
            let mut options = AttentionOptions::default();
            if let Some(left) = left {
                options = options.with_window(left, 0);
            }
            let lengths_on = |device: &Device| {
                Tensor::<1, Int>::from_data(TensorData::new(lengths.to_vec(), [b]), device)
            };
            let expected = Host::from_tensor(attention_decode(
                q.on::<3>(&flex, DType::F32),
                k.on::<4>(&flex, DType::F32),
                v.on::<4>(&flex, DType::F32),
                Some(lengths_on(&flex)),
                options,
            ));
            let takes = decode_kernels_take(device, dtype, [b, s, hq, hkv, d]);
            if common::runtime_of(device) == Some("cuda") {
                assert!(takes, "CUDA turned the decoding kernels away");
            }
            let actual = launches_kernels(2 * usize::from(takes), || {
                Host::from_tensor(attention_decode(
                    q.on::<3>(device, dtype),
                    k.on::<4>(device, dtype),
                    v.on::<4>(device, dtype),
                    Some(lengths_on(device)),
                    options,
                ))
            });
            // The sequence that holds no key decodes to zeros.
            for (bi, &len) in lengths.iter().enumerate() {
                if len == 0 {
                    let row = &actual.values[bi * hq * d..(bi + 1) * hq * d];
                    assert!(row.iter().all(|&x| x == 0.0), "an empty cache gave {row:?}");
                }
            }
            let (abs, rel) = errors(&actual, &expected);
            println!(
                "decode {dtype:?} [B, S, Hq, Hkv, D] {:?}, lengths {lengths:?}, window {left:?}: \
                 max abs {abs:.3e}, relative {rel:.3e}",
                [b, s, hq, hkv, d]
            );
            assert!(rel <= tolerance, "decode {i}: relative error {rel:.3e}");
        }
    }
}

/// Decoding without lengths is attention with one query after every key, the window included;
/// with lengths, attention over each sequence's own keys.
#[cfg(feature = "flex")]
#[test]
fn decode_is_attention_with_one_query_on_flex() {
    let device = Device::flex();
    let [b, s, hq, hkv, d] = [3, 50, 4, 2, 64];
    let q = host(&[b, 1, hq, d], 400);
    let k = host(&[b, s, hkv, d], 401);
    let v = host(&[b, s, hkv, d], 402);
    for options in [
        AttentionOptions::default(),
        AttentionOptions::default().with_window(7, 0),
    ] {
        let step = Host::from_tensor(attention_decode(
            Tensor::<4>::from_data(TensorData::new(q.values.clone(), [b, 1, hq, d]), &device)
                .reshape([b, hq, d]),
            k.on::<4>(&device, DType::F32),
            v.on::<4>(&device, DType::F32),
            None,
            options,
        ));
        let (full, _) = attention_with_lse(
            q.on::<4>(&device, DType::F32),
            k.on::<4>(&device, DType::F32),
            v.on::<4>(&device, DType::F32),
            AttentionMask::None,
            options.with_causal(true),
        );
        let full = Host::from_tensor(full.reshape([b, hq, d]));
        let (abs, _) = errors(&step, &full);
        assert!(
            abs == 0.0,
            "{options:?}: decode and attention differ by {abs}"
        );
    }

    // Sequence b holding `lengths[b]` keys: attention over the first `lengths[b]` keys alone.
    let lengths = [50, 17, 1];
    let lengths_t = Tensor::<1, Int>::from_data(TensorData::new(lengths.to_vec(), [b]), &device);
    let options = AttentionOptions::default().with_window(9, 0);
    let step = Host::from_tensor(attention_decode(
        q.on::<4>(&device, DType::F32).reshape([b, hq, d]),
        k.on::<4>(&device, DType::F32),
        v.on::<4>(&device, DType::F32),
        Some(lengths_t),
        options,
    ));
    for (bi, len) in lengths.into_iter().enumerate() {
        let one = |h: &Host, t: usize, heads: usize| {
            h.on::<4>(&device, DType::F32)
                .slice([bi..bi + 1, 0..t, 0..heads, 0..d])
        };
        let (full, _) = attention_with_lse(
            one(&q, 1, hq),
            one(&k, len as usize, hkv),
            one(&v, len as usize, hkv),
            AttentionMask::None,
            options,
        );
        let full = Host::from_tensor(full.reshape([hq * d]));
        let got = &step.values[bi * hq * d..(bi + 1) * hq * d];
        let diff = got
            .iter()
            .zip(&full.values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff <= 1e-6,
            "sequence {bi} of {len} keys: decode differs by {diff}"
        );
    }
}

/// A decoding step with cache lengths that requires gradients runs the reference through
/// autodiff: its gradients are those of attention with one query over each sequence's own keys
/// (and zero for the keys past a sequence's length).
#[cfg(feature = "flex")]
#[test]
fn decode_with_lengths_takes_gradients_on_flex() {
    let device = Device::flex().autodiff();
    let [b, s, hq, hkv, d] = [3, 20, 4, 2, 64];
    let q = host(&[b, 1, hq, d], 410);
    let k = host(&[b, s, hkv, d], 411);
    let v = host(&[b, s, hkv, d], 412);
    let grad = host(&[b, 1, hq, d], 413);
    let lengths = [20, 9, 1];
    let options = AttentionOptions::default().with_window(5, 0);

    let step_q = q
        .on::<4>(&device, DType::F32)
        .reshape([b, hq, d])
        .detach()
        .require_grad();
    let step_k = k.on::<4>(&device, DType::F32).require_grad();
    let step_v = v.on::<4>(&device, DType::F32).require_grad();
    let lengths_t = Tensor::<1, Int>::from_data(TensorData::new(lengths.to_vec(), [b]), &device);
    let step = attention_decode(
        step_q.clone(),
        step_k.clone(),
        step_v.clone(),
        Some(lengths_t),
        options,
    );
    let grads = (step * grad.on::<4>(&device, DType::F32).reshape([b, hq, d]))
        .sum()
        .backward();
    let got = [
        Host::from_tensor(step_q.grad(&grads).expect("query gradient")),
        Host::from_tensor(step_k.grad(&grads).expect("key gradient")),
        Host::from_tensor(step_v.grad(&grads).expect("value gradient")),
    ];

    for (bi, len) in lengths.into_iter().enumerate() {
        let len = len as usize;
        let one = |h: &Host, t: usize, heads: usize| {
            h.on::<4>(&device, DType::F32)
                .slice([bi..bi + 1, 0..t, 0..heads, 0..d])
                .detach()
                .require_grad()
        };
        let (fq, fk, fv) = (one(&q, 1, hq), one(&k, len, hkv), one(&v, len, hkv));
        let (out, _) = attention_with_lse(
            fq.clone(),
            fk.clone(),
            fv.clone(),
            AttentionMask::None,
            options,
        );
        let g = grad
            .on::<4>(&device, DType::F32)
            .slice([bi..bi + 1, 0..1, 0..hq, 0..d]);
        let grads = (out * g).sum().backward();
        let want = [
            Host::from_tensor(fq.grad(&grads).unwrap()),
            Host::from_tensor(fk.grad(&grads).unwrap()),
            Host::from_tensor(fv.grad(&grads).unwrap()),
        ];
        let rows = [hq * d, s * hkv * d, s * hkv * d];
        for (name, ((got, want), row)) in ["dq", "dk", "dv"]
            .iter()
            .zip(got.iter().zip(&want).zip(rows))
        {
            let got = &got.values[bi * row..(bi + 1) * row];
            let n = want.values.len();
            let diff = got[..n]
                .iter()
                .zip(&want.values)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                diff <= 1e-6,
                "sequence {bi} of {len} keys: {name} differs by {diff}"
            );
            assert!(
                got[n..].iter().all(|&x| x == 0.0),
                "sequence {bi} of {len} keys: {name} past its length is not zero"
            );
        }
    }
}
