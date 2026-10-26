//! The causal depthwise conv1d against transformers 5.15's torch fallbacks, `causal_conv1d_fn`
//! and `causal_conv1d_update`, evaluated in float64 from float32 inputs and from inputs rounded
//! to bfloat16, and recorded as float32 (`tests/fixtures/conv1d/conv1d.safetensors`; the
//! bfloat16 cases also hold torch's own bfloat16 results on CUDA, for comparison).
//!
//! Errors are stated as the largest absolute difference and as that difference over the largest
//! magnitude of the reference, per output and gradient.
//!
//!   cargo test --release -p dash-ops --test conv1d -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test conv1d -- --nocapture

use burn::tensor::{DType, Device, Tensor, TensorData};
use dash_ops::conv1d::{causal_conv1d, causal_conv1d_state, causal_conv1d_update, counters};
use safetensors::SafeTensors;
use serde_json::Value;
use std::sync::Mutex;
use std::sync::atomic::Ordering;

/// The tests read process-wide launch counters, so they run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const FIXTURES: &[u8] = include_bytes!("fixtures/conv1d/conv1d.safetensors");

fn fixtures() -> SafeTensors<'static> {
    SafeTensors::deserialize(FIXTURES).expect("conv1d fixtures")
}

fn cases() -> serde_json::Map<String, Value> {
    let (_, metadata) = SafeTensors::read_metadata(FIXTURES).expect("conv1d fixtures");
    let cases = metadata.metadata().as_ref().expect("metadata")["cases"].clone();
    serde_json::from_str::<Value>(&cases)
        .expect("cases JSON")
        .as_object()
        .expect("cases")
        .clone()
}

/// A recorded tensor: its values and shape.
fn recorded(name: &str) -> (Vec<f32>, Vec<usize>) {
    let fixtures = fixtures();
    let view = fixtures
        .tensor(name)
        .unwrap_or_else(|_| panic!("no fixture {name}"));
    let values = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (values, view.shape().to_vec())
}

fn load<const D: usize>(name: &str, device: &Device, dtype: DType) -> Tensor<D> {
    let (values, shape) = recorded(name);
    Tensor::from_data(
        TensorData::new(values, shape).convert_dtype(dtype),
        (device, dtype),
    )
}

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
}

/// The largest absolute error and that error over the reference's largest magnitude.
fn errors(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len(), "lengths differ");
    let mut abs = 0.0f64;
    let mut scale = 0.0f64;
    for (a, e) in actual.iter().zip(expected) {
        assert!(a.is_finite(), "non-finite output {a}");
        abs = abs.max((*a as f64 - *e as f64).abs());
        scale = scale.max((*e as f64).abs());
    }
    (abs, abs / scale.max(f64::MIN_POSITIVE))
}

/// Asserts `actual` against the recorded `name` within `rel` (error over the largest magnitude)
/// and prints the measured errors.
fn check(label: &str, actual: Vec<f32>, name: &str, rel: f64) {
    let (expected, _) = recorded(name);
    let (abs, measured) = errors(&actual, &expected);
    println!("{label} {name}: max abs {abs:.3e}, rel {measured:.3e} (bound {rel:.1e})");
    assert!(
        measured <= rel,
        "{label} {name}: error {measured:.3e} over the largest magnitude exceeds {rel:.1e}"
    );
}

/// Bounds on the error over the largest magnitude, each just above the largest error measured
/// on Flex and on CUDA (an RTX 5070 Ti) over these cases.
///
/// f32: output and input gradient up to 1.3e-7, about an ulp of the largest value, as torch's
/// own f32 on CUDA (up to 1.6e-7 on the same cases). The weight gradient sums B x T products
/// per weight: 2.3e-7 on CUDA, whose kernel sums runs of 64 steps and then the runs, and 4.5e-7
/// on Flex, which sums in one pass.
///
/// bf16: output 2.0e-3, input gradient 2.7e-3, weight gradient 2.2e-3: the f32 result rounded
/// once to bf16, within its 2^-8 bound. torch's bf16 path on CUDA also rounds between its
/// steps (the convolution's output before SiLU, SiLU's gradient before the convolution's
/// backward), and reaches 2.8e-3, 4.3e-3 and 3.1e-3 on the same case.
const F32_REL: f64 = 2e-7;
const F32_WEIGHT_REL: f64 = 6e-7;
const BF16_REL: f64 = 3e-3;

fn bound(case: &Value) -> (DType, f64, f64) {
    match case["bf16"].as_bool().unwrap() {
        true => (DType::BF16, BF16_REL, BF16_REL),
        false => (DType::F32, F32_REL, F32_WEIGHT_REL),
    }
}

/// Every sequence case: the output and both gradients through `Autodiff`, whose written-out
/// backward pass must run.
fn sequences(device: &Device, label: &str) {
    for (name, case) in cases()
        .iter()
        .filter(|(_, case)| case["kind"] == "sequence")
    {
        let (dtype, rel, rel_weight) = bound(case);
        let silu = case["silu"].as_bool().unwrap();
        let x = load::<3>(&format!("{name}.x"), device, dtype).require_grad();
        let weight = load::<2>(&format!("{name}.weight"), device, dtype).require_grad();
        let grad_out = load::<3>(&format!("{name}.grad_out"), device, dtype);

        let out = causal_conv1d(x.clone(), weight.clone(), silu);
        assert_eq!(out.dtype(), dtype);
        check(label, values(out.clone()), &format!("{name}.out"), rel);

        let runs = counters::BACKWARD_RUNS.load(Ordering::SeqCst);
        let grads = (out * grad_out).sum().backward();
        assert!(
            counters::BACKWARD_RUNS.load(Ordering::SeqCst) > runs,
            "{name}: the written-out backward pass did not run"
        );
        let grad_x = x.grad(&grads).expect("x gradient");
        let grad_weight = weight.grad(&grads).expect("weight gradient");
        assert_eq!((grad_x.dtype(), grad_weight.dtype()), (dtype, dtype));
        check(label, values(grad_x), &format!("{name}.grad_x"), rel);
        check(
            label,
            values(grad_weight),
            &format!("{name}.grad_weight"),
            rel_weight,
        );

        if case["bf16"].as_bool().unwrap() {
            for part in ["out", "grad_x", "grad_weight"] {
                let (torch, _) = recorded(&format!("{name}.torch_{part}"));
                let (truth, _) = recorded(&format!("{name}.{part}"));
                let (abs, measured) = errors(&torch, &truth);
                println!(
                    "  torch bf16 on CUDA {name}.{part}: max abs {abs:.3e}, rel {measured:.3e}"
                );
            }
        }
    }
}

/// Every decoding case, on a device with gradients (where the step is recorded as a constant)
/// and on one without.
fn steps(device: &Device, label: &str) {
    for (name, case) in cases().iter().filter(|(_, case)| case["kind"] == "step") {
        let (dtype, rel, _) = bound(case);
        let silu = case["silu"].as_bool().unwrap();
        for device in [device.clone(), device.clone().autodiff()] {
            let x = load::<2>(&format!("{name}.x"), &device, dtype);
            let state = load::<3>(&format!("{name}.state"), &device, dtype);
            let weight = load::<2>(&format!("{name}.weight"), &device, dtype);
            let (out, next) = causal_conv1d_update(x, state, weight, silu);
            check(label, values(out), &format!("{name}.out"), rel);
            // The state only moves values: exact in any dtype.
            check(label, values(next), &format!("{name}.new_state"), 0.0);
        }
        if case["bf16"].as_bool().unwrap() {
            let (torch, _) = recorded(&format!("{name}.torch_out"));
            let (truth, _) = recorded(&format!("{name}.out"));
            let (abs, measured) = errors(&torch, &truth);
            println!("  torch bf16 on CUDA {name}.out: max abs {abs:.3e}, rel {measured:.3e}");
        }
    }
}

/// The state a prefix leaves, stepped through the rest of the sequence one token at a time,
/// gives the sequence's own outputs: the decoding step and the convolution agree.
fn prefill_then_steps(device: &Device, label: &str) {
    for state_len in [3, 4] {
        let x = load::<3>("long.x", device, DType::F32);
        let weight = load::<2>("long.weight", device, DType::F32);
        let [b, t, c] = x.dims();
        let full = values(causal_conv1d(x.clone(), weight.clone(), true));
        for prefix in [0, 1, 2, 100] {
            let mut state =
                causal_conv1d_state(x.clone().slice([0..b, 0..prefix, 0..c]), state_len);
            let mut worst = 0.0f64;
            for step in prefix..prefix + 20 {
                let token = x
                    .clone()
                    .slice([0..b, step..step + 1, 0..c])
                    .reshape([b, c]);
                let (out, next) = causal_conv1d_update(token, state, weight.clone(), true);
                state = next;
                let out = values(out);
                for bi in 0..b {
                    for ci in 0..c {
                        let expected = full[(bi * t + step) * c + ci] as f64;
                        worst = worst.max((out[bi * c + ci] as f64 - expected).abs());
                    }
                }
            }
            println!(
                "{label} prefill {prefix} then 20 steps, state {state_len}: max abs {worst:.3e}"
            );
            assert!(
                worst <= 1e-6,
                "{label}: steps after a prefill of {prefix} disagree by {worst:.3e}"
            );
        }
    }
}

#[test]
fn sequence_on_flex_matches_transformers() {
    let _serial = serial();
    sequences(&Device::flex().autodiff(), "flex");
}

#[test]
fn step_on_flex_matches_transformers() {
    let _serial = serial();
    steps(&Device::flex(), "flex");
}

#[test]
fn prefill_then_steps_on_flex_equals_the_sequence() {
    let _serial = serial();
    prefill_then_steps(&Device::flex(), "flex");
}

/// A gradient asked of a decoding step is answered (by the reference, which Burn
/// differentiates), not dropped.
#[test]
fn a_tracked_step_is_differentiable() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    let x = load::<2>("step_s3.x", &device, DType::F32);
    let state = load::<3>("step_s3.state", &device, DType::F32).require_grad();
    let weight = load::<2>("step_s3.weight", &device, DType::F32);
    let (out, _) = causal_conv1d_update(x, state.clone(), weight.clone(), false);
    let grads = out.sum().backward();
    // Without the activation, d out[b, c] / d state[b, c, j] is weight[c, j] for the last three
    // steps of the window (taps 0, 1, 2).
    let grad = values(state.grad(&grads).expect("state gradient"));
    let weight = values(weight);
    let [b, c, s] = [8, 24, 3];
    for bi in 0..b {
        for ci in 0..c {
            for j in 0..s {
                assert_eq!(grad[(bi * c + ci) * s + j], weight[ci * 4 + j]);
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn cuda() -> Device {
        Device::cuda(0)
    }

    fn ran(counter: &AtomicUsize, before: usize, what: &str) {
        assert!(
            counter.load(Ordering::SeqCst) > before,
            "the {what} kernel did not run"
        );
    }

    #[test]
    fn sequence_on_cuda_runs_the_kernels() {
        let _serial = serial();
        let (forward, backward) = (
            counters::FORWARD_KERNEL.load(Ordering::SeqCst),
            counters::BACKWARD_KERNEL.load(Ordering::SeqCst),
        );
        sequences(&cuda().autodiff(), "cuda");
        ran(&counters::FORWARD_KERNEL, forward, "forward");
        ran(&counters::BACKWARD_KERNEL, backward, "backward");
    }

    #[test]
    fn step_on_cuda_runs_the_kernel() {
        let _serial = serial();
        let before = counters::UPDATE_KERNEL.load(Ordering::SeqCst);
        steps(&cuda(), "cuda");
        ran(&counters::UPDATE_KERNEL, before, "update");
    }

    #[test]
    fn prefill_then_steps_on_cuda_equals_the_sequence() {
        let _serial = serial();
        prefill_then_steps(&cuda(), "cuda");
    }

    /// The kernels against the reference on Flex at a size the fixtures do not reach: many
    /// runs per sequence, a ragged last run, hundreds of channel vectors, two batches.
    #[test]
    fn kernels_match_the_reference_at_scale() {
        let _serial = serial();
        let [b, t, c] = [2, 1000, 1544];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut uniform = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
                })
                .collect()
        };
        let (x, weight, grad) = (
            uniform(b * t * c, 3.0),
            uniform(c * 4, 0.5),
            uniform(b * t * c, 1.0),
        );
        let run = |device: &Device| {
            let x = Tensor::<3>::from_data(TensorData::new(x.clone(), [b, t, c]), device)
                .require_grad();
            let weight = Tensor::<2>::from_data(TensorData::new(weight.clone(), [c, 4]), device)
                .require_grad();
            let grad = Tensor::<3>::from_data(TensorData::new(grad.clone(), [b, t, c]), device);
            let out = causal_conv1d(x.clone(), weight.clone(), true);
            let grads = (out.clone() * grad).sum().backward();
            [
                values(out),
                values(x.grad(&grads).unwrap()),
                values(weight.grad(&grads).unwrap()),
            ]
        };
        let forward = counters::FORWARD_KERNEL.load(Ordering::SeqCst);
        let actual = run(&cuda().autodiff());
        ran(&counters::FORWARD_KERNEL, forward, "forward");
        let expected = run(&Device::flex().autodiff());
        // Both sides err at f32 rounding; the weight gradient's 2000-term sums most (1.1e-6).
        let bounds = [("out", 3e-7), ("grad_x", 3e-7), ("grad_weight", 2e-6)];
        for ((part, bound), (a, e)) in bounds.iter().zip(actual.iter().zip(&expected)) {
            let (abs, rel) = errors(a, e);
            println!(
                "cuda against flex at [{b}, {t}, {c}] {part}: max abs {abs:.3e}, rel {rel:.3e}"
            );
            assert!(rel <= *bound, "{part}: {rel:.3e} exceeds {bound:.1e}");
        }
    }
}
