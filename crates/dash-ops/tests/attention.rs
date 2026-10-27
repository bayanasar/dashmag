//! Causal attention with grouped key-value heads (`dash_ops::attention`): the reference and its
//! written-out backward pass against torch's math SDPA in fp32, the CubeCL kernels against the
//! same records in bf16 and f16 and against the reference at shapes with several blocks, and
//! decoding against the reference. `tests/attention_masks.rs` covers the other masks.
//!
//! With a CubeCL runtime's feature the kernel tests run on that runtime's device
//! (`tests/common`): the kernels where the device has what they ask, the reference elsewhere,
//! each test asserting which ran.
//!
//!   cargo test -p dash-ops --test attention
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test attention
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test attention
//!   DASH_OPS_DEVICE=cpu cargo test --release -p dash-ops --features cpu --test attention
//!
//! The records (`tests/fixtures/attention/*.safetensors`) hold inputs drawn from a normal
//! distribution and rounded to bf16, so that the same inputs feed f32 and bf16 runs, and
//! torch's output, log-sum-exp and gradients computed from them in fp32. They are in torch's
//! order, the heads before the time (`[B, H, T, D]`, `[B, H, T]`); [`record`] swaps the two into
//! the module's `[B, T, H, D]` and `[B, T, H]`. Errors are reported as the largest absolute
//! difference and that difference over the largest reference magnitude.

mod common;

use burn::backend::{DType, FloatDType, bf16};
use burn::tensor::{Device, Tensor, TensorData};
#[cfg(feature = "cuda")]
use dash_ops::attention::causal_attention_backward;
use dash_ops::attention::{
    AttentionMask, AttentionOptions, attention_decode, attention_with_lse, causal_attention,
    causal_attention_with_lse, probe,
};
use safetensors::{Dtype, SafeTensors};
use std::sync::Mutex;

/// The probe counters are global: tests that read them run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// A host tensor: values and shape.
#[derive(Clone)]
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

    /// The same values in another shape of as many elements (row-major order kept).
    fn reshaped(&self, shape: &[usize]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), self.values.len());
        Self {
            values: self.values.clone(),
            shape: shape.to_vec(),
        }
    }
}

/// One record: inputs, the scale, and torch's results.
struct Record {
    name: &'static str,
    q: Host,
    k: Host,
    v: Host,
    grad: Host,
    scale: f32,
    expected: Results,
}

/// What a run produces: the output, the log-sum-exp and the three input gradients.
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

fn record(name: &'static str) -> Record {
    let path = format!(
        "{}/tests/fixtures/attention/{name}.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let (_, metadata) = SafeTensors::read_metadata(&bytes).expect("metadata");
    let scale = metadata
        .metadata()
        .as_ref()
        .and_then(|m| m.get("scale"))
        .expect("scale in metadata")
        .parse::<f32>()
        .expect("scale");
    let tensors = SafeTensors::deserialize(&bytes).expect("safetensors");
    Record {
        name,
        q: read(&tensors, "q"),
        k: read(&tensors, "k"),
        v: read(&tensors, "v"),
        grad: read(&tensors, "do"),
        scale,
        expected: Results {
            o: read(&tensors, "o"),
            lse: read(&tensors, "lse"),
            dq: read(&tensors, "dq"),
            dk: read(&tensors, "dk"),
            dv: read(&tensors, "dv"),
        },
    }
}

/// The causal forward with its log-sum-exp, then the backward of `sum(o * grad)`, on `device` (an
/// autodiff device) with inputs and the output gradient in `dtype`, the backward deterministic or
/// not ([`AttentionOptions::deterministic`]).
#[allow(clippy::too_many_arguments)]
fn run(
    device: &Device,
    q: &Host,
    k: &Host,
    v: &Host,
    grad: &Host,
    scale: f32,
    dtype: DType,
    deterministic: bool,
) -> Results {
    let q = q.on::<4>(device, dtype).detach().require_grad();
    let k = k.on::<4>(device, dtype).detach().require_grad();
    let v = v.on::<4>(device, dtype).detach().require_grad();
    let grad = grad.on::<4>(device, dtype);
    let options = AttentionOptions::causal()
        .with_scale(scale)
        .with_deterministic(deterministic);
    let (o, lse) = attention_with_lse(
        q.clone(),
        k.clone(),
        v.clone(),
        AttentionMask::None,
        options,
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

/// The largest absolute difference, and that over the largest reference magnitude.
fn errors(actual: &Host, expected: &Host) -> (f32, f32) {
    assert_eq!(actual.shape, expected.shape, "shapes differ");
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    for (a, e) in actual.values.iter().zip(&expected.values) {
        assert!(a.is_finite(), "non-finite value {a}");
        max_abs = max_abs.max((a - e).abs());
        max_ref = max_ref.max(e.abs());
    }
    (max_abs, max_abs / max_ref.max(f32::MIN_POSITIVE))
}

/// Compares every result with the reference: each relative error must stay under `tolerance`
/// (one per result, in the order o, lse, dq, dk, dv). Prints the errors.
fn check(label: &str, actual: &Results, expected: &Results, tolerance: [f32; 5]) {
    let pairs = [
        ("o", &actual.o, &expected.o),
        ("lse", &actual.lse, &expected.lse),
        ("dq", &actual.dq, &expected.dq),
        ("dk", &actual.dk, &expected.dk),
        ("dv", &actual.dv, &expected.dv),
    ];
    for ((name, a, e), tol) in pairs.into_iter().zip(tolerance) {
        let (abs, rel) = errors(a, e);
        println!("{label} {name}: max abs {abs:.3e}, relative {rel:.3e} (tolerance {tol:.1e})");
        assert!(
            rel <= tol,
            "{label} {name}: relative error {rel:.3e} > {tol:.1e}"
        );
    }
}

/// Tolerances on the error relative to the largest reference magnitude, in the order o, lse,
/// dq, dk, dv, each just above the largest error measured over every case and every build that
/// uses it: Flex; `cuda`, with `fusion`, `autotune` or both (an RTX 5070 Ti, CUDA 13); `wgpu` on
/// lavapipe and `cpu`, where the reference runs. In f32 the reference agrees with torch to
/// rounding. In bf16 the error is the output's own rounding (8 significant bits) plus the
/// probabilities' rounding to bf16 before the value product; the log-sum-exp is accumulated in
/// f32 throughout.
const F32_TOLERANCE: [f32; 5] = [3e-7, 1.1e-7, 4.5e-7, 3.5e-7, 3.5e-7];
/// The reference run on a CubeCL device in f32, against torch: just above the largest error
/// over the four CUDA builds (`cuda`, with `fusion`, each with and without Burn's `autotune`),
/// lavapipe and CubeCL's CPU runtime. Its products and reductions are Burn's CubeCL kernels,
/// which autotune and the runtime pick differently: dq is 4.2e-7 relative on CUDA and 8.1e-7 on
/// lavapipe, dk 3.3e-7 and 4.4e-7; the rest moves less. Flex takes no part in those features, so
/// `F32_TOLERANCE` holds for every build. With autotune Burn's kernels change from run to run, so
/// the bounds are then a few roundings, as in `attention_masks.rs`.
#[cfg(feature = "cubecl")]
const CUBE_F32_TOLERANCE: [f32; 5] = if cfg!(feature = "autotune") {
    [1e-6, 2e-7, 1e-6, 1e-6, 1e-6]
} else {
    [3.2e-7, 1.1e-7, 8.5e-7, 4.6e-7, 3.8e-7]
};
/// The kernels in bf16 and f16, the query gradient summed in a fixed order or with atomics: just
/// above the largest of both over repeated runs of each CUDA build. With atomics the query
/// gradient's sum comes in whatever order the kernels run; its largest errors are where few key
/// blocks add up, and the same in every run.
#[cfg(feature = "cubecl")]
const BF16_TOLERANCE: [f32; 5] = [3.4e-3, 2.5e-7, 4.7e-3, 4e-3, 4.2e-3];
#[cfg(feature = "cubecl")]
const F16_TOLERANCE: [f32; 5] = [4.6e-4, 1.7e-7, 4.7e-4, 3.6e-4, 4e-4];
/// Decoding in bf16 against the f32 reference.
#[cfg(feature = "cubecl")]
const DECODE_BF16_TOLERANCE: f32 = 2.9e-3;

const RECORDS: [&str; 2] = ["gqa8x2_d256_t40", "gqa6x2_d64_tq24_tk56"];

#[cfg(feature = "flex")]
#[test]
fn reference_and_its_backward_match_torch_on_flex() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::flex().autodiff();
    for name in RECORDS {
        let r = record(name);
        let runs = probe::backward_runs();
        let actual = run(
            &device,
            &r.q,
            &r.k,
            &r.v,
            &r.grad,
            r.scale,
            DType::F32,
            true,
        );
        assert!(
            probe::backward_runs() > runs,
            "the written-out backward did not run"
        );
        check(
            &format!("flex f32 {}", r.name),
            &actual,
            &r.expected,
            F32_TOLERANCE,
        );
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

/// The queries `[B, Tq, Hq, D]`, keys and values `[B, Tk, Hkv, D]`, the output gradient and the
/// scale, from a seed, for `dims` `[B, Hq, Hkv, Tq, Tk, D]`. The values are drawn in torch's
/// order, the heads before the time, as the records hold them.
fn inputs(dims: [usize; 6], seed: u64) -> (Host, Host, Host, Host, f32) {
    let [b, hq, hkv, tq, tk, d] = dims;
    (
        host(&[b, hq, tq, d], seed).swap_1_2(),
        host(&[b, hkv, tk, d], seed + 1).swap_1_2(),
        host(&[b, hkv, tk, d], seed + 2).swap_1_2(),
        host(&[b, hq, tq, d], seed + 3).swap_1_2(),
        (d as f32).powf(-0.5),
    )
}

/// A decoding step's query, `[B, Hq, D]`, from the `[B, 1, Hq, D]` queries of [`inputs`].
fn step_query(q: &Host) -> Host {
    let [b, tq, hq, d] = [q.shape[0], q.shape[1], q.shape[2], q.shape[3]];
    assert_eq!(tq, 1, "one query per sequence");
    q.reshaped(&[b, hq, d])
}

/// Decoding with the whole cache and the scale given: no lengths, no window.
fn decode(q: Tensor<3>, k: Tensor<4>, v: Tensor<4>, scale: f32) -> Tensor<3> {
    attention_decode(q, k, v, None, AttentionOptions::default().with_scale(scale))
}

#[cfg(feature = "flex")]
#[test]
fn decode_is_attention_with_one_query_on_flex() {
    let device = Device::flex();
    let (q, k, v, _, scale) = inputs([2, 4, 2, 1, 37, 64], 7);
    let decoded = decode(
        step_query(&q).on::<3>(&device, DType::F32),
        k.on::<4>(&device, DType::F32),
        v.on::<4>(&device, DType::F32),
        scale,
    );
    let full = causal_attention(
        q.on::<4>(&device, DType::F32),
        k.on::<4>(&device, DType::F32),
        v.on::<4>(&device, DType::F32),
        scale,
    );
    let decoded = Host::from_tensor(decoded);
    assert_eq!(decoded.shape, [2, 4, 64], "decoding returns [B, Hq, D]");
    let full = Host::from_tensor(full).reshaped(&decoded.shape);
    let (abs, _) = errors(&decoded, &full);
    assert!(
        abs == 0.0,
        "decode and attention with Tq = 1 differ by {abs}"
    );
}

/// A decoding step that requires gradients takes the training path: its query's gradient is
/// that of attention with one query.
#[cfg(feature = "flex")]
#[test]
fn decode_with_gradients_is_attention_with_one_query_on_flex() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::flex().autodiff();
    let (q, k, v, grad, scale) = inputs([2, 4, 2, 1, 37, 64], 8);
    let leaf = |h: &Host| h.on::<4>(&device, DType::F32).require_grad();

    let (full_q, full_k, full_v) = (leaf(&q), leaf(&k), leaf(&v));
    let full = causal_attention(full_q.clone(), full_k.clone(), full_v.clone(), scale);
    let full_grads = (full * grad.on::<4>(&device, DType::F32)).sum().backward();

    let step_q = step_query(&q).on::<3>(&device, DType::F32).require_grad();
    let (step_k, step_v) = (leaf(&k), leaf(&v));
    let runs = probe::backward_runs();
    let step = decode(step_q.clone(), step_k.clone(), step_v.clone(), scale);
    let step_grads = (step * step_query(&grad).on::<3>(&device, DType::F32))
        .sum()
        .backward();
    assert!(
        probe::backward_runs() > runs,
        "the written-out backward did not run"
    );
    let pairs = [
        (
            Host::from_tensor(step_q.grad(&step_grads).unwrap()),
            Host::from_tensor(full_q.grad(&full_grads).unwrap()).reshaped(&[2, 4, 64]),
        ),
        (
            Host::from_tensor(step_k.grad(&step_grads).unwrap()),
            Host::from_tensor(full_k.grad(&full_grads).unwrap()),
        ),
        (
            Host::from_tensor(step_v.grad(&step_grads).unwrap()),
            Host::from_tensor(full_v.grad(&full_grads).unwrap()),
        ),
    ];
    for (name, (step, full)) in ["dq", "dk", "dv"].into_iter().zip(&pairs) {
        let (abs, _) = errors(step, full);
        assert!(abs == 0.0, "{name}: decode and attention differ by {abs}");
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

    /// The `[B, Tq, Tk, Hq, Hkv, D]` of inputs from [`inputs`] for `dims` `[B, Hq, Hkv, Tq, Tk, D]`.
    fn call_dims([b, hq, hkv, tq, tk, d]: [usize; 6]) -> [usize; 6] {
        [b, tq, tk, hq, hkv, d]
    }

    /// The kernel launches a forward and backward pass over `dims` (`[B, Tq, Tk, Hq, Hkv, D]`) in
    /// `dtype` makes on `device`: one forward and three backward kernels where the kernels take
    /// them, none where the reference runs; [`launches_kernels`] adds the fourth of a
    /// deterministic backward whose products split. Under CUDA the kernels take every bf16 and f16 call
    /// with a head dimension of 64, 128 or 256: cards of compute capability 8.0 and later have
    /// all they ask.
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
    fn kernels_match_torch_in_bf16_and_f16() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let device = device();
        println!("{}", common::describe(&device));
        for name in RECORDS {
            let r = record(name);
            let [_, tq, hq, d] = [r.q.shape[0], r.q.shape[1], r.q.shape[2], r.q.shape[3]];
            let [b, tk, hkv, _] = [r.k.shape[0], r.k.shape[1], r.k.shape[2], r.k.shape[3]];
            for (dtype, label, tolerance) in [
                (DType::BF16, "bf16", BF16_TOLERANCE),
                (DType::F16, "f16", F16_TOLERANCE),
            ] {
                if !has(&device, dtype) {
                    continue;
                }
                for deterministic in [true, false] {
                    let expected = expected_launches(&device, dtype, [b, tq, tk, hq, hkv, d]);
                    let backward = kernels_take(&device, dtype, [b, tq, tk, hq, hkv, d])[1];
                    let (runs, copies) = (probe::backward_runs(), probe::input_copies());
                    let atomics = probe::atomic_runs();
                    let actual = launches_kernels(expected, || {
                        run(
                            &device,
                            &r.q,
                            &r.k,
                            &r.v,
                            &r.grad,
                            r.scale,
                            dtype,
                            deterministic,
                        )
                    });
                    assert!(
                        probe::backward_runs() > runs,
                        "the written-out backward did not run"
                    );
                    assert_eq!(
                        probe::input_copies(),
                        copies,
                        "contiguous inputs were copied"
                    );
                    assert_eq!(
                        probe::atomic_runs() - atomics,
                        usize::from(backward && !deterministic),
                        "the query gradient was not summed the way asked"
                    );
                    check(
                        &format!("cube {label} deterministic {deterministic} {}", r.name),
                        &actual,
                        &r.expected,
                        tolerance,
                    );
                }
            }
        }
    }

    #[test]
    fn f32_runs_the_reference() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let device = device();
        for name in RECORDS {
            let r = record(name);
            let dims = [
                r.q.shape[0],
                r.q.shape[1],
                r.k.shape[1],
                r.q.shape[2],
                r.k.shape[2],
                r.q.shape[3],
            ];
            assert_eq!(expected_launches(&device, DType::F32, dims), 0);
            let actual = launches_kernels(0, || {
                run(
                    &device,
                    &r.q,
                    &r.k,
                    &r.v,
                    &r.grad,
                    r.scale,
                    DType::F32,
                    true,
                )
            });
            check(
                &format!("cube f32 {}", r.name),
                &actual,
                &r.expected,
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

    /// Shapes with several query and key blocks, partial last blocks, groups of 1, 3 and 4,
    /// Tk > Tq and a single query, against the reference on Flex from the same bf16 inputs, the
    /// backward's products both split between two cubes per key block and whole.
    #[test]
    fn kernels_match_the_reference_across_blocks() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let device = device();
        if !has(&device, DType::BF16) {
            return;
        }
        let shapes = [
            [2, 8, 2, 200, 200, 256],
            [1, 4, 4, 130, 130, 128],
            [1, 6, 2, 70, 150, 64],
            [1, 8, 2, 1, 33, 256],
        ];
        for (split, deterministic) in [(false, true), (true, true), (false, false), (true, false)] {
            let _forced = ForcedSplit::new(split);
            for (i, dims) in shapes.into_iter().enumerate() {
                let (q, k, v, grad, scale) = inputs(dims, 100 + 10 * i as u64);
                let expected = run(
                    &Device::flex().autodiff(),
                    &q,
                    &k,
                    &v,
                    &grad,
                    scale,
                    DType::F32,
                    true,
                );
                let launches = expected_launches(&device, DType::BF16, call_dims(dims));
                let backward = kernels_take(&device, DType::BF16, call_dims(dims))[1];
                let (copies, splits) = (probe::input_copies(), probe::split_runs());
                let atomics = probe::atomic_runs();
                let actual = launches_kernels(launches, || {
                    run(
                        &device,
                        &q,
                        &k,
                        &v,
                        &grad,
                        scale,
                        DType::BF16,
                        deterministic,
                    )
                });
                assert_eq!(probe::input_copies(), copies);
                assert_eq!(probe::split_runs() - splits, usize::from(split && backward));
                assert_eq!(
                    probe::atomic_runs() - atomics,
                    usize::from(backward && !deterministic)
                );
                check(
                    &format!("cube bf16 split {split} deterministic {deterministic} {dims:?}"),
                    &actual,
                    &expected,
                    BF16_TOLERANCE,
                );
            }
        }
    }

    /// The forward and backward through autodiff, every input given as a view of other storage
    /// through `layout`, against the reference on Flex from contiguous inputs. Returns the
    /// inputs the kernels copied, and whether they ran.
    fn run_with_layout(
        dims: [usize; 6],
        seed: u64,
        layout: impl Fn(&Host, &Device) -> Tensor<4>,
    ) -> (usize, bool) {
        let (q, k, v, grad, scale) = inputs(dims, seed);
        let expected = run(
            &Device::flex().autodiff(),
            &q,
            &k,
            &v,
            &grad,
            scale,
            DType::F32,
            true,
        );
        let device = device();
        let leaf = |h: &Host| layout(h, &device).detach().require_grad();
        let (qt, kt, vt) = (leaf(&q), leaf(&k), leaf(&v));
        let launches = expected_launches(&device, DType::BF16, call_dims(dims));
        let copies = probe::input_copies();
        let actual = launches_kernels(launches, || {
            let (o, lse) = causal_attention_with_lse(qt.clone(), kt.clone(), vt.clone(), scale);
            let grads = (o.clone() * grad.on::<4>(&device, DType::BF16))
                .sum()
                .backward();
            Results {
                o: Host::from_tensor(o),
                lse: Host::from_tensor(lse),
                dq: Host::from_tensor(qt.grad(&grads).unwrap()),
                dk: Host::from_tensor(kt.grad(&grads).unwrap()),
                dv: Host::from_tensor(vt.grad(&grads).unwrap()),
            }
        });
        check(
            &format!("cube bf16 strided {dims:?}"),
            &actual,
            &expected,
            BF16_TOLERANCE,
        );
        (probe::input_copies() - copies, launches > 0)
    }

    /// Inputs stored in torch's order, `[B, H, T, D]`, and seen as `[B, T, H, D]` through
    /// `swap_dims(1, 2)` are read in place: nothing is copied on the way in. Inputs whose last
    /// dimension is not contiguous are copied, and give the same results.
    #[test]
    fn kernels_read_head_major_inputs_in_place() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        if !has(&device(), DType::BF16) {
            return;
        }
        let dims = [2, 8, 2, 96, 96, 256];
        let head_major = |h: &Host, device: &Device| {
            let t = h.on::<4>(device, DType::BF16).swap_dims(1, 2);
            Tensor::<4>::from_data(t.into_data(), device)
                .cast(DType::BF16)
                .swap_dims(1, 2)
        };
        let (copies, _) = run_with_layout(dims, 5, head_major);
        assert_eq!(copies, 0, "inputs the kernels read in place were copied");

        let features_first = |h: &Host, device: &Device| {
            let t = h.on::<4>(device, DType::BF16).swap_dims(2, 3);
            Tensor::<4>::from_data(t.into_data(), device)
                .cast(DType::BF16)
                .swap_dims(2, 3)
        };
        let (copies, kernels) = run_with_layout(dims, 6, features_first);
        assert!(
            copies >= 3 || !kernels,
            "inputs with a strided last dimension were not copied"
        );
    }

    /// Decoding against the reference, including a cache read through a slice of a longer one.
    #[test]
    fn decode_matches_the_reference() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let device = common::cube_device();
        if !has(&device, DType::BF16) {
            return;
        }
        let launches = |[b, hq, hkv, _, s, d]: [usize; 6]| {
            let takes = decode_kernels_take(&device, DType::BF16, [b, s, hq, hkv, d]);
            if common::runtime_of(&device) == Some("cuda") {
                assert!(takes, "CUDA turned the decoding kernels away");
            }
            2 * usize::from(takes)
        };
        for (i, dims) in [
            [3, 8, 2, 1, 1000, 256],
            [2, 4, 1, 1, 77, 128],
            [1, 6, 3, 1, 300, 64],
            [8, 8, 2, 1, 6144, 256],
        ]
        .into_iter()
        .enumerate()
        {
            let (q, k, v, _, scale) = inputs(dims, 200 + 10 * i as u64);
            let q = step_query(&q);
            let flex = Device::flex();
            let expected = Host::from_tensor(decode(
                q.on::<3>(&flex, DType::F32),
                k.on::<4>(&flex, DType::F32),
                v.on::<4>(&flex, DType::F32),
                scale,
            ));
            let copies = probe::input_copies();
            let actual = launches_kernels(launches(dims), || {
                Host::from_tensor(decode(
                    q.on::<3>(&device, DType::BF16),
                    k.on::<4>(&device, DType::BF16),
                    v.on::<4>(&device, DType::BF16),
                    scale,
                ))
            });
            assert_eq!(probe::input_copies(), copies);
            let (abs, rel) = errors(&actual, &expected);
            println!("decode bf16 {dims:?}: max abs {abs:.3e}, relative {rel:.3e}");
            assert!(
                rel <= DECODE_BF16_TOLERANCE,
                "decode {dims:?}: relative error {rel:.3e}"
            );
        }

        // On an autodiff device, a step that takes no gradient runs the same kernels.
        let dims = [3, 8, 2, 1, 1000, 256];
        let (q, k, v, _, scale) = inputs(dims, 200);
        let q = step_query(&q);
        let step = |device: &Device| {
            Host::from_tensor(decode(
                q.on::<3>(device, DType::BF16),
                k.on::<4>(device, DType::BF16),
                v.on::<4>(device, DType::BF16),
                scale,
            ))
        };
        let plain = step(&device);
        let autodiff = launches_kernels(launches(dims), || step(&device.clone().autodiff()));
        let (abs, _) = errors(&autodiff, &plain);
        assert!(
            abs == 0.0,
            "decoding on an autodiff device differs by {abs}"
        );

        // A cache of 512 positions of which the first 300 are filled: read as a slice, in place.
        let dims = [2, 8, 2, 1, 512, 256];
        let (q, k, v, _, scale) = inputs(dims, 300);
        let q = step_query(&q);
        let flex = Device::flex();
        let slice = |h: &Host, device: &Device, dtype| {
            h.on::<4>(device, dtype).slice([0..2, 0..300, 0..2, 0..256])
        };
        let expected = Host::from_tensor(decode(
            q.on::<3>(&flex, DType::F32),
            slice(&k, &flex, DType::F32),
            slice(&v, &flex, DType::F32),
            scale,
        ));
        let copies = probe::input_copies();
        let actual = launches_kernels(launches([2, 8, 2, 1, 300, 256]), || {
            Host::from_tensor(decode(
                q.on::<3>(&device, DType::BF16),
                slice(&k, &device, DType::BF16),
                slice(&v, &device, DType::BF16),
                scale,
            ))
        });
        assert_eq!(probe::input_copies(), copies, "the sliced cache was copied");
        let (abs, rel) = errors(&actual, &expected);
        println!("decode bf16 sliced cache: max abs {abs:.3e}, relative {rel:.3e}");
        assert!(
            rel <= DECODE_BF16_TOLERANCE,
            "sliced cache: relative error {rel:.3e}"
        );
    }
}

/// Memory on CUDA, where CubeCL's pools report what they hold.
#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// What CubeCL's pools on device 0 hold.
    struct Memory {
        /// Bytes in use.
        in_use: u64,
        /// The largest single allocation the pools that hold tensors (the dynamic and the
        /// persistent ones) ever served.
        largest_tensor: u64,
        /// The largest the dedicated pool ever served. CubeCL reserves there what belongs to
        /// no workload: with Burn's `autotune` on, the first autotuned operation (a loss's
        /// reduction, a matmul) has its throughput probes reserve two 512 MiB windows there
        /// for the memory curve its roofline bounds need. Reported, not asserted on.
        largest_dedicated: u64,
    }

    fn memory() -> Memory {
        let client = cubecl::Device::cuda(0).expect("CUDA device 0").client();
        let report = client.memory_report(cubecl::MemoryScope::Device);
        let (mut largest_tensor, mut largest_dedicated) = (0, 0);
        for stream in &report.streams {
            let pools = &stream.pools;
            for pool in pools.dynamic.iter().chain([&pools.persistent]) {
                largest_tensor = largest_tensor.max(pool.largest_alloc);
            }
            largest_dedicated = largest_dedicated.max(pools.dedicated.largest_alloc);
        }
        Memory {
            in_use: report.usage().bytes_in_use,
            largest_tensor,
            largest_dedicated,
        }
    }

    /// At T = 8192 one layer's scores would be 8 x 8192 x 8192: 2 GiB in f32, 1 GiB in bf16.
    /// No allocation of the pools that hold tensors may come near that over the forward and
    /// backward: the largest are the inputs' f32 staging copies, 64 MiB.
    ///
    /// The two are called directly on the CUDA device rather than through autodiff, so that
    /// what runs between the measurements is this module's kernels alone (no loss, no
    /// autotuned Burn operation), and each step is synchronised before memory is read, which
    /// Burn's fusion needs for it to have run at all. On CUDA only: it reads CUDA's pools.
    #[test]
    fn forward_and_backward_keep_no_t_by_t() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const LIMIT: u64 = 256 << 20;
        let device = Device::cuda(0);
        let (q, k, v, grad, scale) = inputs([1, 8, 2, 8192, 8192, 256], 9);
        let q = q.on::<4>(&device, DType::BF16);
        let k = k.on::<4>(&device, DType::BF16);
        let v = v.on::<4>(&device, DType::BF16);
        let grad = grad.on::<4>(&device, DType::BF16);
        device.sync().unwrap();
        let start = memory();
        assert!(
            start.largest_tensor < LIMIT,
            "{} bytes were allocated before the forward: the bound below would prove nothing",
            start.largest_tensor
        );

        let (launches, copies) = (probe::kernel_launches(), probe::input_copies());
        let splits = probe::split_runs();
        let (o, lse) = causal_attention_with_lse(q.clone(), k.clone(), v.clone(), scale);
        device.sync().unwrap();
        let forward = memory();
        let (dq, dk, dv) = causal_attention_backward(
            q.clone(),
            k.clone(),
            v.clone(),
            o.clone(),
            lse.clone(),
            grad.clone(),
            scale,
        );
        device.sync().unwrap();
        let backward = memory();
        // The products split here (512 key blocks on 70 multiprocessors leave the last round
        // light), and a deterministic backward whose products split runs a fourth kernel.
        let split = probe::split_runs() - splits;
        assert_eq!(
            probe::kernel_launches(),
            launches + 4 + split,
            "the forward and the backward's kernels ran, and no other path"
        );
        assert_eq!(probe::input_copies(), copies, "an input was copied");
        for gradient in [dq, dk, dv] {
            let sample = Host::from_tensor(gradient.slice([0..1, 0..1, 0..1, 0..8]));
            assert!(sample.values.iter().all(|x| x.is_finite()));
        }

        let mib = |bytes: u64| bytes as f64 / (1u64 << 20) as f64;
        println!(
            "T 8192: {:.1} MiB more in use after the forward, {:.1} MiB after the backward; \
             largest tensor allocation {:.1} MiB (dedicated pool: {:.1} MiB)",
            mib(forward.in_use - start.in_use),
            mib(backward.in_use - start.in_use),
            mib(backward.largest_tensor),
            mib(backward.largest_dedicated)
        );
        assert!(
            backward.largest_tensor < LIMIT,
            "a {}-byte allocation: T x T was materialised",
            backward.largest_tensor
        );
    }
}
