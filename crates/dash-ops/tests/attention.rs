//! Causal attention with grouped key-value heads (`dash_ops::attention`): the reference and its
//! written-out backward pass against torch's math SDPA in fp32, the CUDA kernels against the
//! same records in bf16 and f16 and against the reference at shapes with several blocks, and
//! decoding against the reference.
//!
//!   cargo test -p dash-ops --test attention
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test attention
//!
//! The records (`tests/fixtures/attention/*.safetensors`) hold inputs drawn from a normal
//! distribution and rounded to bf16, so that the same inputs feed f32 and bf16 runs, and
//! torch's output, log-sum-exp and gradients computed from them in fp32. They are in torch's
//! order, the heads before the time (`[B, H, T, D]`, `[B, H, T]`); [`record`] swaps the two into
//! the module's `[B, T, H, D]` and `[B, T, H]`. Errors are reported as the largest absolute
//! difference and that difference over the largest reference magnitude.

use burn::backend::{DType, FloatDType, bf16};
use burn::tensor::{Device, Tensor, TensorData};
#[cfg(feature = "cuda")]
use dash_ops::attention::causal_attention_backward;
use dash_ops::attention::{attention_decode, causal_attention, causal_attention_with_lse, probe};
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

/// The forward with its log-sum-exp, then the backward of `sum(o * grad)`, on `device` (an
/// autodiff device) with inputs and the output gradient in `dtype`.
fn run(
    device: &Device,
    q: &Host,
    k: &Host,
    v: &Host,
    grad: &Host,
    scale: f32,
    dtype: DType,
) -> Results {
    let q = q.on::<4>(device, dtype).detach().require_grad();
    let k = k.on::<4>(device, dtype).detach().require_grad();
    let v = v.on::<4>(device, dtype).detach().require_grad();
    let grad = grad.on::<4>(device, dtype);
    let (o, lse) = causal_attention_with_lse(q.clone(), k.clone(), v.clone(), scale);
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
/// uses it (an RTX 5070 Ti, CUDA 13). In f32 the reference agrees with torch to rounding. In
/// bf16 the error is the output's own rounding (8 significant bits) plus the probabilities'
/// rounding to bf16 before the value product; the log-sum-exp is accumulated in f32
/// throughout.
const F32_TOLERANCE: [f32; 5] = [4e-7, 2e-7, 6e-7, 5e-7, 5e-7];
/// The reference run on CUDA in f32, against torch: just above the largest error over the four
/// CUDA builds (`cuda`, with `fusion`, each with and without Burn's `autotune`). Its products
/// and reductions are Burn's CubeCL kernels, which autotune picks differently: dq is 4.2e-7
/// relative without it and 7.1e-7 with it, dk 3.3e-7 and 3.7e-7; the rest moves less. Flex
/// takes no part in those features, so `F32_TOLERANCE` holds for every build.
#[cfg(feature = "cuda")]
const CUDA_F32_TOLERANCE: [f32; 5] = [3e-7, 2e-7, 9e-7, 5e-7, 3e-7];
#[cfg(feature = "cuda")]
const BF16_TOLERANCE: [f32; 5] = [4e-3, 5e-7, 5.5e-3, 4.5e-3, 5e-3];
#[cfg(feature = "cuda")]
const F16_TOLERANCE: [f32; 5] = [5.5e-4, 5e-7, 5.5e-4, 4.5e-4, 5e-4];
/// Decoding in bf16 against the f32 reference.
#[cfg(feature = "cuda")]
const DECODE_BF16_TOLERANCE: f32 = 3.5e-3;

const RECORDS: [&str; 2] = ["gqa8x2_d256_t40", "gqa6x2_d64_tq24_tk56"];

#[test]
fn reference_and_its_backward_match_torch_on_flex() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::flex().autodiff();
    for name in RECORDS {
        let r = record(name);
        let runs = probe::backward_runs();
        let actual = run(&device, &r.q, &r.k, &r.v, &r.grad, r.scale, DType::F32);
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

#[test]
fn decode_is_attention_with_one_query_on_flex() {
    let device = Device::flex();
    let (q, k, v, _, scale) = inputs([2, 4, 2, 1, 37, 64], 7);
    let decoded = attention_decode(
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
    let step = attention_decode(step_q.clone(), step_k.clone(), step_v.clone(), scale);
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

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    fn cuda() -> Device {
        Device::cuda(0).autodiff()
    }

    #[test]
    fn kernels_match_torch_in_bf16_and_f16() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        for name in RECORDS {
            let r = record(name);
            for (dtype, label, tolerance) in [
                (DType::BF16, "bf16", BF16_TOLERANCE),
                (DType::F16, "f16", F16_TOLERANCE),
            ] {
                let (launches, runs, copies) = (
                    probe::kernel_launches(),
                    probe::backward_runs(),
                    probe::input_copies(),
                );
                let actual = run(&cuda(), &r.q, &r.k, &r.v, &r.grad, r.scale, dtype);
                // The forward, then delta, dQ, and dK with dV.
                assert!(
                    probe::kernel_launches() >= launches + 4,
                    "the CUDA kernels did not run"
                );
                assert!(
                    probe::backward_runs() > runs,
                    "the written-out backward did not run"
                );
                assert_eq!(
                    probe::input_copies(),
                    copies,
                    "contiguous inputs were copied"
                );
                check(
                    &format!("cuda {label} {}", r.name),
                    &actual,
                    &r.expected,
                    tolerance,
                );
            }
        }
    }

    #[test]
    fn f32_runs_the_reference_on_cuda() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let r = record(RECORDS[0]);
        let launches = probe::kernel_launches();
        let actual = run(&cuda(), &r.q, &r.k, &r.v, &r.grad, r.scale, DType::F32);
        assert_eq!(
            probe::kernel_launches(),
            launches,
            "f32 should take the reference"
        );
        check(
            &format!("cuda f32 {}", r.name),
            &actual,
            &r.expected,
            CUDA_F32_TOLERANCE,
        );
    }

    /// Shapes with several query and key blocks, partial last blocks, groups of 1, 3 and 4,
    /// Tk > Tq and a single query, against the reference on Flex from the same bf16 inputs.
    #[test]
    fn kernels_match_the_reference_across_blocks() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let shapes = [
            [2, 8, 2, 200, 200, 256],
            [1, 4, 4, 130, 130, 128],
            [1, 6, 2, 70, 150, 64],
            [1, 8, 2, 1, 33, 256],
        ];
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
            );
            let (launches, copies) = (probe::kernel_launches(), probe::input_copies());
            let actual = run(&cuda(), &q, &k, &v, &grad, scale, DType::BF16);
            assert!(probe::kernel_launches() >= launches + 4);
            assert_eq!(probe::input_copies(), copies);
            check(
                &format!("cuda bf16 {dims:?}"),
                &actual,
                &expected,
                BF16_TOLERANCE,
            );
        }
    }

    /// The forward and backward through autodiff, every input given as a view of other storage
    /// through `layout`, against the reference on Flex from contiguous inputs. Returns the
    /// inputs the kernels copied.
    fn run_with_layout(
        dims: [usize; 6],
        seed: u64,
        layout: impl Fn(&Host, &Device) -> Tensor<4>,
    ) -> usize {
        let (q, k, v, grad, scale) = inputs(dims, seed);
        let expected = run(
            &Device::flex().autodiff(),
            &q,
            &k,
            &v,
            &grad,
            scale,
            DType::F32,
        );
        let device = cuda();
        let leaf = |h: &Host| layout(h, &device).detach().require_grad();
        let (qt, kt, vt) = (leaf(&q), leaf(&k), leaf(&v));
        let (launches, copies) = (probe::kernel_launches(), probe::input_copies());
        let (o, lse) = causal_attention_with_lse(qt.clone(), kt.clone(), vt.clone(), scale);
        let grads = (o.clone() * grad.on::<4>(&device, DType::BF16))
            .sum()
            .backward();
        let actual = Results {
            o: Host::from_tensor(o),
            lse: Host::from_tensor(lse),
            dq: Host::from_tensor(qt.grad(&grads).unwrap()),
            dk: Host::from_tensor(kt.grad(&grads).unwrap()),
            dv: Host::from_tensor(vt.grad(&grads).unwrap()),
        };
        assert!(probe::kernel_launches() >= launches + 4);
        check(
            &format!("cuda bf16 strided {dims:?}"),
            &actual,
            &expected,
            BF16_TOLERANCE,
        );
        probe::input_copies() - copies
    }

    /// Inputs stored in torch's order, `[B, H, T, D]`, and seen as `[B, T, H, D]` through
    /// `swap_dims(1, 2)` are read in place: nothing is copied on the way in. Inputs whose last
    /// dimension is not contiguous are copied, and give the same results.
    #[test]
    fn kernels_read_head_major_inputs_in_place() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dims = [2, 8, 2, 96, 96, 256];
        let head_major = |h: &Host, device: &Device| {
            let t = h.on::<4>(device, DType::BF16).swap_dims(1, 2);
            Tensor::<4>::from_data(t.into_data(), device)
                .cast(DType::BF16)
                .swap_dims(1, 2)
        };
        let copies = run_with_layout(dims, 5, head_major);
        assert_eq!(copies, 0, "inputs the kernels read in place were copied");

        let features_first = |h: &Host, device: &Device| {
            let t = h.on::<4>(device, DType::BF16).swap_dims(2, 3);
            Tensor::<4>::from_data(t.into_data(), device)
                .cast(DType::BF16)
                .swap_dims(2, 3)
        };
        let copies = run_with_layout(dims, 6, features_first);
        assert!(
            copies >= 3,
            "inputs with a strided last dimension were not copied"
        );
    }

    /// Decoding against the reference, including a cache read through a slice of a longer one.
    #[test]
    fn decode_matches_the_reference() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let device = Device::cuda(0);
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
            let expected = Host::from_tensor(attention_decode(
                q.on::<3>(&flex, DType::F32),
                k.on::<4>(&flex, DType::F32),
                v.on::<4>(&flex, DType::F32),
                scale,
            ));
            let (launches, copies) = (probe::kernel_launches(), probe::input_copies());
            let actual = Host::from_tensor(attention_decode(
                q.on::<3>(&device, DType::BF16),
                k.on::<4>(&device, DType::BF16),
                v.on::<4>(&device, DType::BF16),
                scale,
            ));
            assert!(probe::kernel_launches() >= launches + 2);
            assert_eq!(probe::input_copies(), copies);
            let (abs, rel) = errors(&actual, &expected);
            println!("decode bf16 {dims:?}: max abs {abs:.3e}, relative {rel:.3e}");
            assert!(
                rel <= DECODE_BF16_TOLERANCE,
                "decode {dims:?}: relative error {rel:.3e}"
            );
        }

        // On an autodiff device, a step that takes no gradient runs the same kernels.
        let (q, k, v, _, scale) = inputs([3, 8, 2, 1, 1000, 256], 200);
        let q = step_query(&q);
        let step = |device: &Device| {
            Host::from_tensor(attention_decode(
                q.on::<3>(device, DType::BF16),
                k.on::<4>(device, DType::BF16),
                v.on::<4>(device, DType::BF16),
                scale,
            ))
        };
        let plain = step(&device);
        let launches = probe::kernel_launches();
        let autodiff = step(&device.clone().autodiff());
        assert!(
            probe::kernel_launches() >= launches + 2,
            "the autodiff device did not run the decoding kernels"
        );
        let (abs, _) = errors(&autodiff, &plain);
        assert!(
            abs == 0.0,
            "decoding on an autodiff device differs by {abs}"
        );

        // A cache of 512 positions of which the first 300 are filled: read as a slice, in place.
        let (q, k, v, _, scale) = inputs([2, 8, 2, 1, 512, 256], 300);
        let q = step_query(&q);
        let flex = Device::flex();
        let slice = |h: &Host, device: &Device, dtype| {
            h.on::<4>(device, dtype).slice([0..2, 0..300, 0..2, 0..256])
        };
        let expected = Host::from_tensor(attention_decode(
            q.on::<3>(&flex, DType::F32),
            slice(&k, &flex, DType::F32),
            slice(&v, &flex, DType::F32),
            scale,
        ));
        let copies = probe::input_copies();
        let actual = Host::from_tensor(attention_decode(
            q.on::<3>(&device, DType::BF16),
            slice(&k, &device, DType::BF16),
            slice(&v, &device, DType::BF16),
            scale,
        ));
        assert_eq!(probe::input_copies(), copies, "the sliced cache was copied");
        let (abs, rel) = errors(&actual, &expected);
        println!("decode bf16 sliced cache: max abs {abs:.3e}, relative {rel:.3e}");
        assert!(
            rel <= DECODE_BF16_TOLERANCE,
            "sliced cache: relative error {rel:.3e}"
        );
    }

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
    /// Burn's fusion needs for it to have run at all.
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
        assert_eq!(
            probe::kernel_launches(),
            launches + 4,
            "the forward and the three backward kernels ran, and no other path"
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
