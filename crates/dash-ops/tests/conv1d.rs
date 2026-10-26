//! The causal depthwise conv1d against torch, from two recordings, each evaluated in float64
//! from float32 inputs or from inputs rounded to bfloat16 (or float16) and stored as float32:
//!
//! - `tests/fixtures/conv1d/general.safetensors`: torch's `conv1d` with `groups = C` on the input
//!   left-padded by `K - 1` steps, then SiLU where asked: every kernel width from 1 to 8, with
//!   and without a bias and SiLU, and decoding steps from a random state;
//! - `tests/fixtures/conv1d/conv1d.safetensors`: transformers 5.15's torch fallbacks
//!   `causal_conv1d_fn` and `causal_conv1d_update` (kernel width 4 and 3, no bias), whose
//!   decoding state is `[B, C, S]` in the inputs' dtype: the tests transpose it to `[B, S, C]`
//!   and widen it to f32.
//!
//! The reduced-precision cases also hold torch's own results in that dtype on CUDA, for
//! comparison. Every sequence case also runs as a prefill followed by single decoding steps,
//! against the same recorded outputs.
//!
//! Errors are stated as the largest absolute difference and as that difference over the largest
//! magnitude of the reference, per output and gradient.
//!
//!   cargo test --release -p dash-ops --test conv1d -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test conv1d -- --nocapture

use burn::tensor::{DType, Device, Tensor, TensorData};
use dash_ops::conv1d::{
    Activation, CausalConv1dOptions, causal_conv1d, causal_conv1d_state, causal_conv1d_update,
    probe,
};
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// The tests read process-wide launch counters, so they run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A recording: its tensors, and the cases its metadata describes.
#[derive(Clone, Copy)]
struct Fixtures(&'static [u8]);

const GENERAL: Fixtures = Fixtures(include_bytes!("fixtures/conv1d/general.safetensors"));
const TRANSFORMERS: Fixtures = Fixtures(include_bytes!("fixtures/conv1d/conv1d.safetensors"));

impl Fixtures {
    fn cases(self) -> Vec<Case> {
        let (_, metadata) = SafeTensors::read_metadata(self.0).expect("conv1d fixtures");
        let cases = metadata.metadata().as_ref().expect("metadata")["cases"].clone();
        let cases: BTreeMap<String, Value> = serde_json::from_str(&cases).expect("cases JSON");
        cases
            .into_iter()
            .map(|(name, case)| Case::new(self, name, &case))
            .collect()
    }

    fn has(self, name: &str) -> bool {
        let tensors = SafeTensors::deserialize(self.0).expect("conv1d fixtures");
        tensors.tensor(name).is_ok()
    }

    /// A recorded tensor: its values and shape.
    fn recorded(self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let tensors = SafeTensors::deserialize(self.0).expect("conv1d fixtures");
        let view = tensors
            .tensor(name)
            .unwrap_or_else(|_| panic!("no fixture {name}"));
        let values = view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        (values, view.shape().to_vec())
    }

    fn load<const D: usize>(self, name: &str, device: &Device, dtype: DType) -> Tensor<D> {
        let (values, shape) = self.recorded(name);
        Tensor::from_data(
            TensorData::new(values, shape).convert_dtype(dtype),
            (device, dtype),
        )
    }
}

/// One recorded case.
#[derive(Clone)]
struct Case {
    fixtures: Fixtures,
    name: String,
    step: bool,
    dtype: DType,
    bias: bool,
    options: CausalConv1dOptions,
    /// transformers' decoding state, `[B, C, S]` in the case's dtype.
    transposed_state: bool,
}

impl Case {
    fn new(fixtures: Fixtures, name: String, case: &Value) -> Self {
        let dtype = match (case.get("dtype"), case.get("bf16")) {
            (Some(dtype), _) => match dtype.as_str().unwrap() {
                "f32" => DType::F32,
                "bf16" => DType::BF16,
                "f16" => DType::F16,
                other => panic!("unknown dtype {other}"),
            },
            (None, Some(Value::Bool(true))) => DType::BF16,
            (None, _) => DType::F32,
        };
        let activation = match case["silu"].as_bool().unwrap() {
            true => Activation::Silu,
            false => Activation::None,
        };
        Self {
            fixtures,
            step: case["kind"] == "step",
            dtype,
            bias: case.get("bias").and_then(Value::as_bool).unwrap_or(false),
            options: CausalConv1dOptions::default().with_activation(activation),
            transposed_state: case.get("dtype").is_none(),
            name,
        }
    }

    fn part(&self, part: &str) -> String {
        format!("{}.{part}", self.name)
    }

    fn load<const D: usize>(&self, part: &str, device: &Device) -> Tensor<D> {
        self.fixtures.load(&self.part(part), device, self.dtype)
    }

    fn bias(&self, device: &Device) -> Option<Tensor<1>> {
        self.bias.then(|| self.load::<1>("bias", device))
    }

    /// The decoding state, `[B, S, C]` f32.
    fn state(&self, device: &Device) -> Tensor<3> {
        let state = self
            .fixtures
            .load::<3>(&self.part("state"), device, DType::F32);
        match self.transposed_state {
            true => state.swap_dims(1, 2),
            false => state,
        }
    }

    fn kernel_width(&self) -> usize {
        self.fixtures.recorded(&self.part("weight")).1[1]
    }
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

/// Bounds on the error over the largest magnitude, per dtype and part ("step": a decoding step,
/// from a recorded state or after a prefill), each just above the largest measured over these
/// cases on Flex and on CUDA (an RTX 5070 Ti), which are the same with and without fusion and
/// autotune. In the comments, the largest measured on Flex / CUDA, then torch's own on CUDA in
/// the same dtype over the same cases.
///
/// f32 is within about an ulp of the largest value for the output and the input gradient. The
/// weight and bias gradients sum `B x T` products per weight, and err more where those sums
/// cancel; the CUDA kernels sum runs of 64 steps, then the runs.
///
/// bf16 and f16 are the f32 result rounded once: half an ulp of the dtype at the largest
/// magnitude, which is 2^-9 to 2^-8 of it in bf16. torch rounds between its steps too (the
/// convolution's output before SiLU, SiLU's gradient before the convolution's backward), and errs
/// more.
fn bound(dtype: DType, part: &str) -> f64 {
    match (dtype, part) {
        (DType::F32, "out") => 1.4e-7,    // 1.36e-7 / 1.26e-7; torch 1.26e-7
        (DType::F32, "step") => 1.8e-7,   // 1.72e-7 / 1.56e-7; torch 1.39e-7
        (DType::F32, "grad_x") => 2.6e-7, // 2.58e-7 / 2.07e-7; torch 2.04e-7
        (DType::F32, "grad_weight") => 6.1e-7, // 4.53e-7 / 6.01e-7; torch 4.22e-7
        (DType::F32, "grad_bias") => 4.1e-7, // 2.71e-7 / 4.09e-7; torch 3.41e-7
        (DType::BF16, "out" | "step") => 3.7e-3, // 3.62e-3 on both; torch 5.05e-3, steps 3.65e-3
        (DType::BF16, "grad_x") => 3.2e-3, // 3.11e-3 on both; torch 4.26e-3
        (DType::BF16, "grad_weight") => 3.4e-3, // 3.33e-3 on both; torch 4.90e-3
        (DType::BF16, "grad_bias") => 3.4e-3, // 3.31e-3 on both; torch 3.31e-3
        (DType::F16, "out") => 2.7e-4,    // 2.67e-4 on both; torch 3.56e-4
        (DType::F16, "step") => 3.2e-4,   // 3.20e-4 on both; torch 5.44e-4
        (DType::F16, "grad_x") => 3.2e-4, // 3.14e-4 on both; torch 3.24e-4
        (DType::F16, "grad_weight") => 3.4e-4, // 3.38e-4 on both; torch 3.80e-4
        (DType::F16, "grad_bias") => 2.7e-4, // 2.67e-4 on both; torch 2.68e-4
        _ => panic!("no bound for {dtype:?} {part}"),
    }
}

/// The largest error over the largest magnitude seen per dtype and part, for the summary each
/// test prints.
#[derive(Default)]
struct Worst(BTreeMap<String, f64>);

impl Worst {
    /// Asserts `actual` against the recorded `part` of `case` and prints the error.
    fn check(&mut self, label: &str, case: &Case, part: &str, actual: Vec<f32>) {
        let (expected, _) = case.fixtures.recorded(&case.part(part));
        self.check_against(label, case, part, part, &actual, &expected);
    }

    fn check_against(
        &mut self,
        label: &str,
        case: &Case,
        part: &str,
        what: &str,
        actual: &[f32],
        expected: &[f32],
    ) {
        let (abs, rel) = errors(actual, expected);
        let bound = bound(case.dtype, part);
        println!(
            "{label} {} {what}: max abs {abs:.3e}, rel {rel:.3e} (bound {bound:.1e})",
            case.name
        );
        let key = format!("{:?} {part}", case.dtype);
        let worst = self.0.entry(key).or_default();
        *worst = worst.max(rel);
        assert!(
            rel <= bound,
            "{label} {} {what}: error {rel:.3e} over the largest magnitude exceeds {bound:.1e}",
            case.name
        );
    }

    fn print(&self, label: &str) {
        for (key, rel) in &self.0 {
            println!("{label} worst {key}: rel {rel:.3e}");
        }
    }
}

/// torch's own error on the reduced-precision cases, from its recorded results.
fn torch_error(case: &Case, part: &str) {
    let torch = format!("torch_{part}");
    if case.fixtures.has(&case.part(&torch)) {
        let (theirs, _) = case.fixtures.recorded(&case.part(&torch));
        let (truth, _) = case.fixtures.recorded(&case.part(part));
        let (abs, rel) = errors(&theirs, &truth);
        println!(
            "  torch {:?} on CUDA {} {part}: max abs {abs:.3e}, rel {rel:.3e}",
            case.dtype, case.name
        );
    }
}

/// Asserts that `count` grew past `before` when `kernels` (on CUDA), and did not otherwise.
fn launched(kernels: bool, count: fn() -> usize, before: usize, what: &str) {
    assert_eq!(
        count() > before,
        kernels,
        "{what}: the kernel ran {} times, expected {}",
        count() - before,
        if kernels { "some" } else { "none" }
    );
}

/// Every sequence case: the output and every gradient through `Autodiff`, whose written-out
/// backward pass must run, and on CUDA (`kernels`) the forward and backward kernels.
fn sequences(device: &Device, label: &str, kernels: bool) {
    let mut worst = Worst::default();
    for case in [GENERAL, TRANSFORMERS]
        .into_iter()
        .flat_map(Fixtures::cases)
        .filter(|case| !case.step)
    {
        let dtype = case.dtype;
        let x = case.load::<3>("x", device).require_grad();
        let weight = case.load::<2>("weight", device).require_grad();
        let bias = case.bias(device).map(Tensor::require_grad);
        let grad_out = case.load::<3>("grad_out", device);

        let forward = probe::forward_launches();
        let out = causal_conv1d(x.clone(), weight.clone(), bias.clone(), case.options);
        assert_eq!(out.dtype(), dtype);
        worst.check(label, &case, "out", values(out.clone()));
        launched(kernels, probe::forward_launches, forward, &case.name);

        let (runs, backward) = (probe::backward_runs(), probe::backward_launches());
        let grads = (out * grad_out).sum().backward();
        assert!(
            probe::backward_runs() > runs,
            "{}: the written-out backward pass did not run",
            case.name
        );
        let grad_x = x.grad(&grads).expect("x gradient");
        let grad_weight = weight.grad(&grads).expect("weight gradient");
        assert_eq!((grad_x.dtype(), grad_weight.dtype()), (dtype, dtype));
        worst.check(label, &case, "grad_x", values(grad_x));
        worst.check(label, &case, "grad_weight", values(grad_weight));
        // Read after the values: under fusion the kernels run when they are read.
        launched(kernels, probe::backward_launches, backward, &case.name);
        if let Some(bias) = bias {
            let grad_bias = bias.grad(&grads).expect("bias gradient");
            assert_eq!(grad_bias.dtype(), dtype);
            worst.check(label, &case, "grad_bias", values(grad_bias));
        }
        for part in ["out", "grad_x", "grad_weight", "grad_bias"] {
            torch_error(&case, part);
        }
    }
    worst.print(label);
}

/// Every decoding case from a recorded state, on a device with gradients (where the step is
/// taken as a constant) and on one without; on CUDA (`kernels`) through the step kernel.
fn steps(device: &Device, label: &str, kernels: bool) {
    let mut worst = Worst::default();
    for case in [GENERAL, TRANSFORMERS]
        .into_iter()
        .flat_map(Fixtures::cases)
        .filter(|case| case.step)
    {
        for device in [device.clone(), device.clone().autodiff()] {
            let x = case.load::<2>("x", &device);
            let weight = case.load::<2>("weight", &device);
            let launches = probe::update_launches();
            let (out, next) = causal_conv1d_update(
                x,
                case.state(&device),
                weight,
                case.bias(&device),
                case.options,
            );
            assert_eq!((out.dtype(), next.dtype()), (case.dtype, DType::F32));
            let (expected, _) = case.fixtures.recorded(&case.part("out"));
            worst.check_against(label, &case, "step", "out", &values(out), &expected);
            launched(kernels, probe::update_launches, launches, &case.name);
            // The state only moves values: exact.
            let next = match case.transposed_state {
                true => next.swap_dims(1, 2),
                false => next,
            };
            let (expected, _) = case.fixtures.recorded(&case.part("new_state"));
            assert_eq!(values(next), expected, "{label} {}: next state", case.name);
        }
        torch_error(&case, "out");
    }
    worst.print(label);
}

/// Every sequence case as a prefill of `p` steps, then single steps to its end: the outputs are
/// the recorded sequence's, and the final state holds its last inputs; on CUDA (`kernels`)
/// through the step kernel. Returns the largest difference between the steps' outputs and the
/// sequence's own, computed on `device`.
fn prefill_then_steps(device: &Device, label: &str, kernels: bool) -> f64 {
    let mut worst = Worst::default();
    let mut against_sequence = 0.0f64;
    for case in [GENERAL, TRANSFORMERS]
        .into_iter()
        .flat_map(Fixtures::cases)
        .filter(|case| !case.step)
    {
        let k = case.kernel_width();
        let x = case.load::<3>("x", device);
        let weight = case.load::<2>("weight", device);
        let bias = case.bias(device);
        let [b, t, c] = x.dims();
        let sequence = values(causal_conv1d(
            x.clone(),
            weight.clone(),
            bias.clone(),
            case.options,
        ));
        let (expected, _) = case.fixtures.recorded(&case.part("out"));
        // The least state that works, transformers' (`S = K`), and a longer one, each after a
        // prefill of a different length: none, one step, and half the sequence.
        for (prefix, state_len) in [(0, k - 1), (1.min(t), k), (t / 2, k + 2)] {
            let prompt = x.clone().slice([0..b, 0..prefix, 0..c]);
            let mut outputs = Vec::new();
            if prefix > 0 {
                outputs.push(causal_conv1d(
                    prompt.clone(),
                    weight.clone(),
                    bias.clone(),
                    case.options,
                ));
            }
            let mut state = causal_conv1d_state(prompt, state_len);
            let launches = probe::update_launches();
            for step in prefix..t {
                let token = x
                    .clone()
                    .slice([0..b, step..step + 1, 0..c])
                    .reshape([b, c]);
                let (out, next) =
                    causal_conv1d_update(token, state, weight.clone(), bias.clone(), case.options);
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
            let (inputs, _) = case.fixtures.recorded(&case.part("x"));
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

#[test]
fn sequence_on_flex_matches_torch() {
    let _serial = serial();
    sequences(&Device::flex().autodiff(), "flex", false);
}

#[test]
fn step_on_flex_matches_torch() {
    let _serial = serial();
    steps(&Device::flex(), "flex", false);
}

#[test]
fn prefill_then_steps_on_flex_equals_the_sequence() {
    let _serial = serial();
    prefill_then_steps(&Device::flex(), "flex", false);
}

#[test]
fn options_default_to_no_activation() {
    assert_eq!(CausalConv1dOptions::default().activation, Activation::None);
}

/// A gradient asked of a decoding step is answered (by the reference, which Burn
/// differentiates), not dropped: for every input, here with a bias and no activation.
#[test]
fn a_tracked_step_is_differentiable() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    let case = GENERAL
        .cases()
        .into_iter()
        .find(|case| case.name == "step_k4_bias_linear_f32")
        .expect("case");
    let x = case.load::<2>("x", &device).require_grad();
    let state = case.state(&device).require_grad();
    let weight = case.load::<2>("weight", &device).require_grad();
    let bias = case.load::<1>("bias", &device).require_grad();
    let [b, s, c] = state.dims();
    let k = case.kernel_width();
    let (out, _) = causal_conv1d_update(
        x.clone(),
        state.clone(),
        weight.clone(),
        Some(bias.clone()),
        case.options,
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
            case.state(&device).slice([0..b, s + 1 - k..s, 0..c]),
            case.load::<2>("x", &device).reshape([b, 1, c]),
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

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    fn cuda() -> Device {
        Device::cuda(0)
    }

    fn ran(count: impl Fn() -> usize, before: usize, what: &str) {
        assert!(count() > before, "the {what} kernel did not run");
    }

    #[test]
    fn sequence_on_cuda_runs_the_kernels() {
        let _serial = serial();
        sequences(&cuda().autodiff(), "cuda", true);
    }

    #[test]
    fn step_on_cuda_runs_the_kernel() {
        let _serial = serial();
        steps(&cuda(), "cuda", true);
    }

    /// On CUDA the steps' outputs are the sequence's to the bit: the forward and step kernels
    /// share their arithmetic.
    #[test]
    fn prefill_then_steps_on_cuda_equals_the_sequence() {
        let _serial = serial();
        let difference = prefill_then_steps(&cuda(), "cuda", true);
        assert_eq!(difference, 0.0);
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
        let out = causal_conv1d(x.clone(), weight.clone(), bias.clone(), options);
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
    /// several widths, with and without a bias and SiLU, and an input read through strides.
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
                &cuda().autodiff(),
                dtype,
                shape,
                inputs,
                options,
                transposed,
            );
            ran(probe::forward_launches, forward, "forward");
            ran(probe::backward_launches, backward, "backward");
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
            // own f32 result, at most an ulp apart (3.7e-3, 3.0e-3, 3.7e-5, 5.2e-4).
            let bounds = match dtype {
                DType::F32 => [2.4e-7, 2.3e-7, 1.4e-6, 1.2e-6],
                _ => [3.8e-3, 3.1e-3, 3.8e-5, 5.2e-4],
            };
            let parts = ["out", "grad_x", "grad_weight", "grad_bias"];
            for ((part, bound), (a, e)) in
                parts.iter().zip(bounds).zip(actual.iter().zip(&expected))
            {
                let (abs, rel) = errors(a, e);
                println!(
                    "cuda against flex at [{b}, {t}, {c}], K = {k}, bias {has_bias}, \
                     {:?}, {dtype:?}, transposed {transposed}, {part}: max abs {abs:.3e}, \
                     rel {rel:.3e}",
                    options.activation
                );
                assert!(rel <= bound, "{part}: {rel:.3e} exceeds {bound:.1e}");
            }
        }
    }

    /// A width the kernels do not take runs the reference, on CUDA too.
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
        let actual = run(
            &cuda().autodiff(),
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
        // measured: 5.7e-8, 1.6e-7, 2.8e-7, 1.7e-7).
        let parts = ["out", "grad_x", "grad_weight", "grad_bias"];
        let bounds = [6e-8, 1.7e-7, 2.9e-7, 1.8e-7];
        for ((part, bound), (a, e)) in parts.iter().zip(bounds).zip(actual.iter().zip(&expected)) {
            let (abs, rel) = errors(a, e);
            println!("cuda against flex, K = {k}, {part}: max abs {abs:.3e}, rel {rel:.3e}");
            assert!(rel <= bound, "{part}: {rel:.3e} exceeds {bound:.1e}");
        }
    }
}
