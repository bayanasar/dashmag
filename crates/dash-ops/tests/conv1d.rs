//! The causal depthwise conv1d against torch: `dash_conformance::conv1d`'s tests (every kernel
//! width with and without a bias and SiLU, decoding steps from a recorded state, a prefill
//! followed by single steps) through the Burn adapter, on Flex and on a CubeCL device, any
//! runtime (`DASH_OPS_DEVICE`). The adapter asserts that the kernels ran exactly where
//! `kernels_take` says they take a case (always under CUDA); cases in a dtype the device does not
//! have are left out.
//!
//! What is dash-ops' own stays here: the decoding steps and the prefill from Burn's views (the
//! recorded transformers state transposed in place, the prompt and each token sliced out of the
//! sequence), which the adapter's contiguous inputs do not reach; a tracked step's gradients,
//! through the reference Burn differentiates; the kernels against Flex's reference at a size the
//! fixtures do not reach; and widths the kernels do not take.
//!
//!   cargo test --release -p dash-ops --test conv1d -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test conv1d -- --nocapture
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test conv1d

mod common;

use burn::tensor::{DType, Device, Tensor};
use common::adapter::{self, Burn, Watch, burn_dtype, launched};
use dash_conformance::conv1d::{self, Case, Worst, torch_error};
use dash_conformance::errors;
use dash_ops::conv1d::{
    Activation, CausalConv1dOptions, causal_conv1d, causal_conv1d_update, probe,
};
use std::sync::Mutex;

/// The tests read process-wide launch counters, so they run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
}

/// Whether `device` is a CubeCL device but a CUDA one, held to the wider of the bounds.
fn other_runtime(device: &Device) -> bool {
    common::runtime_of(device).is_some() && !common::on_cuda(device)
}

/// Whether the kernels take a case of kernel width `k` in `dtype` for `x` `[B, T, C]` on
/// `device`; under CUDA they take every width they hold.
fn takes(device: &Device, dtype: DType, k: usize, shape: [usize; 3]) -> bool {
    adapter::conv1d_kernels_take(device, dtype, k, shape, None)
}

/// The cases in a dtype `device` has, the sequences or the decoding steps.
fn cases_on(device: &Device, step: bool) -> Vec<Case> {
    conv1d::cases()
        .into_iter()
        .filter(|case| case.step == step && device.supports_dtype(burn_dtype(case.dtype)))
        .collect()
}

/// The input `part` of `case` on `device`, in the case's dtype.
fn load<const D: usize>(case: &Case, part: &str, device: &Device) -> Tensor<D> {
    adapter::tensor(&case.input(part), device)
}

fn bias(case: &Case, device: &Device) -> Option<Tensor<1>> {
    case.bias().map(|bias| adapter::tensor(&bias, device))
}

/// The decoding state, `[B, W, C]` f32: transformers' `[B, C, W]` seen transposed, a view whose
/// channels are not adjacent, as a caller holding transformers' cache hands it over.
fn recorded_state(case: &Case, device: &Device) -> Tensor<3> {
    let state = case.input("state").with_dtype(dash_conformance::DType::F32);
    let state = adapter::tensor::<3>(&state, device);
    match case.state_transposed() {
        true => state.swap_dims(1, 2),
        false => state,
    }
}

/// Every decoding case from a recorded state, on a device with gradients (where the step is
/// taken as a constant) and on one without, the state as [`recorded_state`] gives it; where the
/// kernels take the case, through the step kernel.
fn steps(device: &Device, label: &str) {
    let mut worst = Worst::new(other_runtime(device));
    for case in cases_on(device, true) {
        for device in [device.clone(), device.clone().autodiff()] {
            let x = load::<2>(&case, "x", &device);
            let [b, c] = x.dims();
            let dtype = burn_dtype(case.dtype);
            let kernels = takes(&device, dtype, case.kernel_width(), [b, 1, c]);
            let weight = load::<2>(&case, "weight", &device);
            let launches = probe::update_launches();
            let (out, next) = causal_conv1d_update(
                x,
                weight,
                bias(&case, &device),
                recorded_state(&case, &device),
                case.options.into(),
            );
            assert_eq!((out.dtype(), next.dtype()), (dtype, DType::F32));
            let expected = case.recorded("out");
            worst.check_against(label, &case, "step", "out", &values(out), &expected);
            launched(kernels, probe::update_launches, launches, &case.name);
            // The state only moves values: exact.
            let next = match case.state_transposed() {
                true => next.swap_dims(1, 2),
                false => next,
            };
            let expected = case.recorded("new_state");
            assert_eq!(values(next), expected, "{label} {}: next state", case.name);
        }
        torch_error(&case, "out");
    }
    worst.print(label);
}

/// Every sequence case as a prefill of `p` steps, then single steps to its end, the prompt and
/// each token sliced out of the sequence on the device: the outputs are the recorded sequence's,
/// and the final state holds its last inputs; where the kernels take the case, through the step
/// kernel. Returns the largest difference between the steps' outputs and the sequence's own,
/// computed on `device`.
fn prefill_then_steps(device: &Device, label: &str) -> f64 {
    let mut worst = Worst::new(other_runtime(device));
    let mut against_sequence = 0.0f64;
    for case in cases_on(device, false) {
        let k = case.kernel_width();
        let x = load::<3>(&case, "x", device);
        let weight = load::<2>(&case, "weight", device);
        let bias = bias(&case, device);
        let [b, t, c] = x.dims();
        let kernels = takes(device, burn_dtype(case.dtype), k, [b, 1, c]);
        let sequence = values(
            causal_conv1d(
                x.clone(),
                weight.clone(),
                bias.clone(),
                None,
                None,
                case.options,
            )
            .0,
        );
        let expected = case.recorded("out");
        // The least state that works, transformers' (`S = K`), and a longer one, each after a
        // prefill of a different length: none, one step, and half the sequence.
        for (prefix, state_len) in [(0, k - 1), (1.min(t), k), (t / 2, k + 2)] {
            // The prompt's outputs and the state it leaves, from one call (none for no prompt).
            let prompt = x.clone().slice([0..b, 0..prefix, 0..c]);
            let (out, state) = causal_conv1d(
                prompt,
                weight.clone(),
                bias.clone(),
                None,
                None,
                case.options.with_final_state_len(state_len),
            );
            let mut outputs = Vec::new();
            if prefix > 0 {
                outputs.push(out);
            }
            let mut state = state.expect("the final state");
            let launches = probe::update_launches();
            for step in prefix..t {
                let token = x
                    .clone()
                    .slice([0..b, step..step + 1, 0..c])
                    .reshape([b, c]);
                let (out, next) = causal_conv1d_update(
                    token,
                    weight.clone(),
                    bias.clone(),
                    state,
                    case.options.into(),
                );
                outputs.push(out.reshape([b, 1, c]));
                state = next;
            }
            let decoded = values(Tensor::cat(outputs, 1));
            if prefix < t {
                launched(kernels, probe::update_launches, launches, &case.name);
            }
            let what = format!("prefill {prefix}, state {state_len}");
            worst.check_against(label, &case, "step", &what, &decoded, &expected);
            let (abs, _) = errors(&decoded, &sequence);
            against_sequence = against_sequence.max(abs);

            // The final state is the last `state_len` inputs, zeros before the sequence. The
            // recorded inputs are values of the case's dtype, which the f32 state holds exactly.
            let inputs = case.recorded("x");
            let final_state = values(state);
            for bi in 0..b {
                for s in 0..state_len {
                    for ci in 0..c {
                        let step = (t + s).checked_sub(state_len);
                        let want = step.map_or(0.0, |step| inputs[(bi * t + step) * c + ci]);
                        let got = final_state[(bi * state_len + s) * c + ci];
                        assert_eq!(got, want, "{label} {} {what}: final state", case.name);
                    }
                }
            }
        }
    }
    worst.print(label);
    println!(
        "{label} steps against the sequence on the same device: max abs {against_sequence:.3e}"
    );
    against_sequence
}

#[cfg(feature = "flex")]
#[test]
fn sequence_on_flex_matches_torch() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    let flex = Burn::new(device.clone()).watching(Watch::Launches);
    conv1d::sequences(&flex, "flex", other_runtime(&device)).all_ran();
}

/// On a device with gradients (where the step is taken as a constant) and on one without.
#[cfg(feature = "flex")]
#[test]
fn step_on_flex_matches_torch() {
    let _serial = serial();
    let device = Device::flex();
    let (plain, autodiff) = (
        Burn::new(device.clone()),
        Burn::new(device.clone().autodiff()),
    );
    conv1d::steps(&[&plain, &autodiff], "flex", other_runtime(&device)).all_ran();
}

#[cfg(feature = "flex")]
#[test]
fn step_on_flex_from_views_matches_torch() {
    let _serial = serial();
    steps(&Device::flex(), "flex");
}

#[cfg(feature = "flex")]
#[test]
fn prefill_then_steps_on_flex_equals_the_sequence() {
    let _serial = serial();
    let device = Device::flex();
    let flex = Burn::new(device.clone());
    conv1d::prefill_then_steps(&flex, "flex", other_runtime(&device)).all_ran();
}

#[cfg(feature = "flex")]
#[test]
fn prefill_then_steps_on_flex_from_views_equals_the_sequence() {
    let _serial = serial();
    prefill_then_steps(&Device::flex(), "flex");
}

#[test]
fn options_default_to_no_activation() {
    assert_eq!(CausalConv1dOptions::default().activation, Activation::None);
}

/// A gradient asked of a decoding step is answered (by the reference, which Burn
/// differentiates), not dropped: for every input, here with a bias and no activation.
#[cfg(feature = "flex")]
#[test]
fn a_tracked_step_is_differentiable() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    let case = conv1d::cases()
        .into_iter()
        .find(|case| case.name == "step_k4_bias_linear_f32")
        .expect("case");
    let x = load::<2>(&case, "x", &device).require_grad();
    let state = recorded_state(&case, &device).require_grad();
    let weight = load::<2>(&case, "weight", &device).require_grad();
    let bias = load::<1>(&case, "bias", &device).require_grad();
    let [b, s, c] = state.dims();
    let k = case.kernel_width();
    let (out, _) = causal_conv1d_update(
        x.clone(),
        weight.clone(),
        Some(bias.clone()),
        state.clone(),
        case.options.into(),
    );
    let grads = out.sum().backward();
    // d out[b, c] / d state[b, s, c] is weight[c, j] for the last K - 1 steps of the state (taps
    // 0 .. K - 2), zero before; d / d x[b, c] is weight[c, K - 1]; d / d bias[c] is 1 per batch.
    let w = values(weight.clone());
    let grad_state = values(state.grad(&grads).expect("state gradient"));
    let grad_x = values(x.grad(&grads).expect("x gradient"));
    let grad_bias = values(bias.grad(&grads).expect("bias gradient"));
    for bi in 0..b {
        for ci in 0..c {
            for si in 0..s {
                let want = match (si + k).checked_sub(s + 1) {
                    Some(j) => w[ci * k + j],
                    None => 0.0,
                };
                assert_eq!(grad_state[(bi * s + si) * c + ci], want);
            }
            assert_eq!(grad_x[bi * c + ci], w[ci * k + k - 1]);
        }
    }
    assert!(grad_bias.iter().all(|&g| g == b as f32));
    // The weight's gradient: d out[b, c] / d weight[c, j] summed over the batch.
    let window = values(Tensor::cat(
        vec![
            recorded_state(&case, &device).slice([0..b, s + 1 - k..s, 0..c]),
            load::<2>(&case, "x", &device).reshape([b, 1, c]),
        ],
        1,
    ));
    let grad_weight = values(weight.grad(&grads).expect("weight gradient"));
    for ci in 0..c {
        for j in 0..k {
            let want: f32 = (0..b).map(|bi| window[(bi * k + j) * c + ci]).sum();
            let got = grad_weight[ci * k + j];
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "{got} {want}"
            );
        }
    }
}

/// The same checks on a CubeCL device, any runtime: the one `DASH_OPS_DEVICE` names.
#[cfg(feature = "cubecl")]
mod cube {
    use super::*;
    use burn::tensor::TensorData;

    fn device() -> Device {
        common::cube_device()
    }

    #[test]
    fn sequence_on_cubecl_matches_torch() {
        let _serial = serial();
        println!("{}", common::describe(&device()));
        let cube = Burn::new(device().autodiff()).watching(Watch::Launches);
        conv1d::sequences(&cube, "cubecl", other_runtime(&device())).all_ran();
    }

    #[test]
    fn step_on_cubecl_matches_torch() {
        let _serial = serial();
        let (plain, autodiff) = (Burn::new(device()), Burn::new(device().autodiff()));
        conv1d::steps(&[&plain, &autodiff], "cubecl", other_runtime(&device())).all_ran();
    }

    /// The step kernel reads transformers' state through its strides, or a copy of it.
    #[test]
    fn step_on_cubecl_from_views_matches_torch() {
        let _serial = serial();
        steps(&device(), "cubecl");
    }

    /// Where the kernels run the steps' outputs are the sequence's to the bit: the forward and
    /// step kernels share their arithmetic.
    #[test]
    fn prefill_then_steps_on_cubecl_equals_the_sequence() {
        let _serial = serial();
        let cube = Burn::new(device());
        let difference =
            conv1d::prefill_then_steps(&cube, "cubecl", other_runtime(&device())).all_ran();
        if takes(&device(), DType::F32, 4, [1, 1, 1]) {
            assert_eq!(difference, 0.0);
        }
    }

    /// The same from views, the forward and step kernels reading the prompt and each token
    /// through the sequence's strides.
    #[test]
    fn prefill_then_steps_on_cubecl_from_views_equals_the_sequence() {
        let _serial = serial();
        let difference = prefill_then_steps(&device(), "cubecl");
        if takes(&device(), DType::F32, 4, [1, 1, 1]) {
            assert_eq!(difference, 0.0);
        }
    }

    /// A seeded uniform draw in `[-scale, scale)`.
    fn uniform(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// Input values, f32: `x` and `grad` `[B, T, C]`, `weight` `[C, K]`, `bias` `[C]`.
    #[derive(Clone, Copy)]
    struct Inputs<'a> {
        x: &'a [f32],
        weight: &'a [f32],
        bias: Option<&'a [f32]>,
        grad: &'a [f32],
    }

    /// The output and the three gradients through `Autodiff`, from f32 values given in
    /// `dtype`; when `transposed`, `x` handed over as a `[B, C, T]` tensor seen as `[B, T, C]`
    /// and `weight` as `[K, C]` seen as `[C, K]`.
    fn run(
        device: &Device,
        dtype: DType,
        [b, t, c, k]: [usize; 4],
        Inputs {
            x,
            weight,
            bias,
            grad,
        }: Inputs,
        options: CausalConv1dOptions,
        transposed: bool,
    ) -> Vec<Vec<f32>> {
        let tensor = |values: &[f32], shape: Vec<usize>| {
            TensorData::new(values.to_vec(), shape).convert_dtype(dtype)
        };
        let x = match transposed {
            false => Tensor::<3>::from_data(tensor(x, vec![b, t, c]), (device, dtype)),
            true => {
                // The same values, laid out [B, C, T].
                let mut bct = vec![0.0; x.len()];
                for bi in 0..b {
                    for ti in 0..t {
                        for ci in 0..c {
                            bct[(bi * c + ci) * t + ti] = x[(bi * t + ti) * c + ci];
                        }
                    }
                }
                Tensor::<3>::from_data(tensor(&bct, vec![b, c, t]), (device, dtype)).swap_dims(1, 2)
            }
        }
        .require_grad();
        let weight = match transposed {
            false => Tensor::<2>::from_data(tensor(weight, vec![c, k]), (device, dtype)),
            true => {
                let mut kc = vec![0.0; weight.len()];
                for ci in 0..c {
                    for j in 0..k {
                        kc[j * c + ci] = weight[ci * k + j];
                    }
                }
                Tensor::<2>::from_data(tensor(&kc, vec![k, c]), (device, dtype)).swap_dims(0, 1)
            }
        }
        .require_grad();
        let bias = bias.map(|bias| {
            Tensor::<1>::from_data(tensor(bias, vec![c]), (device, dtype)).require_grad()
        });
        let grad = Tensor::<3>::from_data(tensor(grad, vec![b, t, c]), (device, dtype));
        let (out, _) = causal_conv1d(x.clone(), weight.clone(), bias.clone(), None, None, options);
        let grads = (out.clone() * grad).sum().backward();
        let mut parts = vec![
            values(out),
            values(x.grad(&grads).unwrap()),
            values(weight.grad(&grads).unwrap()),
        ];
        parts.extend(bias.map(|bias| values(bias.grad(&grads).unwrap())));
        parts
    }

    /// The kernels against the reference on Flex at a size the fixtures do not reach: many
    /// runs per sequence, a ragged last run, hundreds of channel vectors, two batches; for
    /// several widths, with and without a bias and SiLU, and an input read through strides. On a
    /// device the kernels do not take, its reference against Flex's.
    #[cfg(feature = "flex")]
    #[test]
    fn kernels_match_the_reference_at_scale() {
        let _serial = serial();
        let [b, t, c] = [2, 1000, 1544];
        let silu = CausalConv1dOptions::default().with_activation(Activation::Silu);
        let none = CausalConv1dOptions::default();
        let configs = [
            (4, false, silu, DType::F32, false),
            (4, true, silu, DType::F32, true),
            (1, true, silu, DType::F32, false),
            (2, true, silu, DType::F32, false),
            (3, true, none, DType::F32, false),
            (5, false, none, DType::F32, false),
            (8, true, silu, DType::F32, false),
            (3, true, silu, DType::BF16, false),
        ];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for (k, has_bias, options, dtype, transposed) in configs {
            if !common::has(&device(), dtype) {
                continue;
            }
            let kernels = takes(&device(), dtype, k, [b, t, c]);
            let x = uniform(&mut seed, b * t * c, 3.0);
            let weight = uniform(&mut seed, c * k, 0.5);
            let bias = uniform(&mut seed, c, 0.5);
            let grad = uniform(&mut seed, b * t * c, 1.0);
            let inputs = Inputs {
                x: &x,
                weight: &weight,
                bias: has_bias.then_some(&bias[..]),
                grad: &grad,
            };
            let (forward, backward) = (probe::forward_launches(), probe::backward_launches());
            let shape = [b, t, c, k];
            let actual = run(
                &device().autodiff(),
                dtype,
                shape,
                inputs,
                options,
                transposed,
            );
            launched(kernels, probe::forward_launches, forward, "forward");
            launched(kernels, probe::backward_launches, backward, "backward");
            let expected = run(
                &Device::flex().autodiff(),
                dtype,
                shape,
                inputs,
                options,
                false,
            );
            // Both sides err at f32 rounding, the weight and bias gradients' 2000-term sums most
            // (largest measured: 2.3e-7, 2.2e-7, 1.3e-6, 1.1e-6). In bf16 each side rounds its
            // own f32 result, at most an ulp apart (3.7e-3, 3.0e-3, 3.7e-5, 5.2e-4). The CubeCL
            // CPU runtime's reference sums those gradients in other orders still (2.2e-6, 2.4e-6;
            // in bf16 1.5e-4 for the weight's).
            let bounds = match (dtype, other_runtime(&device())) {
                (DType::F32, false) => [2.4e-7, 2.3e-7, 1.4e-6, 1.2e-6],
                (DType::F32, true) => [2.4e-7, 2.3e-7, 2.3e-6, 2.4e-6],
                (_, false) => [3.8e-3, 3.1e-3, 3.8e-5, 5.2e-4],
                (_, true) => [3.8e-3, 3.1e-3, 1.5e-4, 5.2e-4],
            };
            let parts = ["out", "grad_x", "grad_weight", "grad_bias"];
            for ((part, bound), (a, e)) in
                parts.iter().zip(bounds).zip(actual.iter().zip(&expected))
            {
                let (abs, rel) = errors(a, e);
                println!(
                    "cubecl against flex at [{b}, {t}, {c}], K = {k}, bias {has_bias}, \
                     {:?}, {dtype:?}, transposed {transposed}, {part}: max abs {abs:.3e}, \
                     rel {rel:.3e}",
                    options.activation
                );
                assert!(rel <= bound, "{part}: {rel:.3e} exceeds {bound:.1e}");
            }
        }
    }

    /// A width the kernels do not take runs the reference, on CUDA too.
    #[cfg(feature = "flex")]
    #[test]
    fn wider_kernels_run_the_reference() {
        let _serial = serial();
        let [b, t, c, k] = [2, 70, 6, dash_ops::conv1d::MAX_KERNEL_WIDTH + 1];
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let x = uniform(&mut seed, b * t * c, 3.0);
        let weight = uniform(&mut seed, c * k, 0.5);
        let bias = uniform(&mut seed, c, 0.5);
        let grad = uniform(&mut seed, b * t * c, 1.0);
        let inputs = Inputs {
            x: &x,
            weight: &weight,
            bias: Some(&bias),
            grad: &grad,
        };
        let silu = CausalConv1dOptions::default().with_activation(Activation::Silu);
        let launches = probe::kernel_launches();
        let runs = probe::backward_runs();
        assert!(!takes(&device(), DType::F32, k, [b, t, c]));
        let actual = run(
            &device().autodiff(),
            DType::F32,
            [b, t, c, k],
            inputs,
            silu,
            false,
        );
        assert_eq!(probe::kernel_launches(), launches, "a kernel ran");
        assert!(
            probe::backward_runs() > runs,
            "the backward pass did not run"
        );
        let expected = run(
            &Device::flex().autodiff(),
            DType::F32,
            [b, t, c, k],
            inputs,
            silu,
            false,
        );
        // The reference on two devices: the same operations, summed in different orders (largest
        // measured: 5.7e-8, 1.6e-7, 2.8e-7, 1.7e-7); on lavapipe, with its own exponential, and
        // on the CubeCL CPU runtime, 1.2e-7, 1.6e-7, 4.3e-7, 2.2e-7.
        let parts = ["out", "grad_x", "grad_weight", "grad_bias"];
        let bounds = match other_runtime(&device()) {
            false => [6e-8, 1.7e-7, 2.9e-7, 1.8e-7],
            true => [1.2e-7, 1.7e-7, 4.3e-7, 2.3e-7],
        };
        for ((part, bound), (a, e)) in parts.iter().zip(bounds).zip(actual.iter().zip(&expected)) {
            let (abs, rel) = errors(a, e);
            println!("cubecl against flex, K = {k}, {part}: max abs {abs:.3e}, rel {rel:.3e}");
            assert!(rel <= bound, "{part}: {rel:.3e} exceeds {bound:.1e}");
        }
    }
}
