//! The causal depthwise conv1d against torch, from two recordings, each evaluated in float64
//! from float32 inputs or from inputs rounded to bfloat16 (or float16) and stored as float32:
//!
//! - `conv1d/general.safetensors`: torch's `conv1d` with `groups = C` on the input left-padded by
//!   `K - 1` steps, then SiLU where asked: every kernel width from 1 to 8, with and without a bias
//!   and SiLU, and decoding steps from a random state;
//! - `conv1d/conv1d.safetensors`: transformers 5.15's torch fallbacks `causal_conv1d_fn` and
//!   `causal_conv1d_update` (kernel width 4 and 3, no bias), whose decoding state is `[B, C, W]`
//!   in the inputs' dtype: the tests transpose it to `[B, W, C]` and widen it to f32.
//!
//! The reduced-precision cases also hold torch's own results in that dtype on CUDA, for
//! comparison. Every sequence case also runs as a prefill followed by single decoding steps,
//! against the same recorded outputs.
//!
//! Errors are stated as the largest absolute difference and as that difference over the largest
//! magnitude of the reference, per output and gradient ([`crate::errors`]). The cases in a dtype
//! the device does not have are left out, silently.
//!
//! - [`sequences`]: every sequence case, the output and every gradient;
//! - [`steps`]: every decoding case from a recorded state;
//! - [`prefill_then_steps`]: every sequence case as a prefill, then single steps to its end.

use crate::fixtures::File;
use crate::{
    Adapter, Conv1d, Conv1dCotangents, Conv1dStep, DType, HostTensor, Outcome, Tally, errors, ran,
};
use dash_contract::conv1d::{Activation, CausalConv1dOptions};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A recording: its tensors, and the cases its metadata describes.
#[derive(Clone, Debug)]
struct Fixtures(Arc<File>);

impl Fixtures {
    fn read(file: &str) -> Self {
        Self(Arc::new(File::read(file)))
    }

    fn cases(self) -> Vec<Case> {
        let cases = self.0.metadata()["cases"].clone();
        let cases: BTreeMap<String, Value> = serde_json::from_str(&cases).expect("cases JSON");
        cases
            .into_iter()
            .map(|(name, case)| Case::new(self.clone(), name, &case))
            .collect()
    }

    fn has(&self, name: &str) -> bool {
        self.0.has(name)
    }

    /// A recorded tensor: its values and shape.
    fn recorded(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let tensor = self.0.tensor(name);
        (tensor.values, tensor.shape)
    }
}

/// Every recorded case, the general recording's then transformers'.
pub fn cases() -> Vec<Case> {
    ["conv1d/general.safetensors", "conv1d/conv1d.safetensors"]
        .into_iter()
        .flat_map(|file| Fixtures::read(file).cases())
        .collect()
}

/// One recorded case.
#[derive(Clone, Debug)]
pub struct Case {
    fixtures: Fixtures,
    /// The case's name in its recording.
    pub name: String,
    /// Whether it is a decoding step rather than a sequence.
    pub step: bool,
    /// The dtype its inputs are in.
    pub dtype: DType,
    /// Whether it has a bias.
    pub bias: bool,
    /// Its options: the activation.
    pub options: CausalConv1dOptions,
    /// transformers' decoding state, `[B, C, W]` in the case's dtype.
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

    /// The recorded values of `part`.
    pub fn recorded(&self, part: &str) -> Vec<f32> {
        self.fixtures.recorded(&self.part(part)).0
    }

    /// The input `part` (`x`, `weight`, `bias`, `grad_out`), in the case's dtype.
    pub fn input(&self, part: &str) -> HostTensor {
        let (values, shape) = self.fixtures.recorded(&self.part(part));
        HostTensor::new(self.dtype, shape, values)
    }

    /// The bias, if the case has one.
    pub fn bias(&self) -> Option<HostTensor> {
        self.bias.then(|| self.input("bias"))
    }

    /// The decoding state, `[B, W, C]` f32.
    pub fn state(&self) -> HostTensor {
        let (values, shape) = self.fixtures.recorded(&self.part("state"));
        let state = HostTensor::f32(shape, values);
        match self.transposed_state {
            true => swap_last(&state),
            false => state,
        }
    }

    /// Whether the recording keeps the decoding state as transformers does, `[B, C, W]`, which
    /// [`Case::state`] gives transposed.
    pub fn state_transposed(&self) -> bool {
        self.transposed_state
    }

    /// A state `[B, W, C]` the way the recording keeps it: as it is, or transformers' `[B, C, W]`.
    fn recorded_layout(&self, state: &HostTensor) -> Vec<f32> {
        match self.transposed_state {
            true => swap_last(state).values,
            false => state.values.clone(),
        }
    }

    /// The kernel width `K`.
    pub fn kernel_width(&self) -> usize {
        self.fixtures.recorded(&self.part("weight")).1[1]
    }
}

/// A `[B, M, N]` tensor's values as `[B, N, M]`.
fn swap_last(tensor: &HostTensor) -> HostTensor {
    let [b, m, n] = tensor.shape[..] else {
        panic!("a state is [B, W, C], got {:?}", tensor.shape)
    };
    let mut values = vec![0.0; tensor.values.len()];
    for bi in 0..b {
        for i in 0..m {
            for j in 0..n {
                values[(bi * n + j) * m + i] = tensor.values[(bi * m + i) * n + j];
            }
        }
    }
    HostTensor::new(tensor.dtype, vec![b, n, m], values)
}

/// Steps `range` of each row of a `[B, T, C]` tensor.
fn steps_of(x: &HostTensor, range: std::ops::Range<usize>) -> HostTensor {
    let [b, t, c] = x.shape[..] else {
        panic!("x is [B, T, C], got {:?}", x.shape)
    };
    let mut values = Vec::with_capacity(b * range.len() * c);
    for row in 0..b {
        values.extend_from_slice(&x.values[(row * t + range.start) * c..(row * t + range.end) * c]);
    }
    HostTensor::new(x.dtype, vec![b, range.len(), c], values)
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
///
/// On the other CubeCL runtimes (`other`), measured on lavapipe through wgpu, where the kernels
/// run, and on the CubeCL CPU runtime, where the reference does, three f32 parts err a little
/// more: lavapipe's exponential (SiLU's) is its own, and the CPU runtime sums in other orders.
pub fn bound(dtype: DType, part: &str, other: bool) -> f64 {
    match (dtype, part) {
        (DType::F32, "out") if other => 1.8e-7, // 1.74e-7 lavapipe / 1.41e-7 CPU runtime
        (DType::F32, "step") if other => 2.2e-7, // 1.74e-7 / 2.18e-7
        (DType::F32, "grad_bias") if other => 5.5e-7, // 3.58e-7 / 5.47e-7
        (DType::F32, "out") => 1.4e-7,          // 1.36e-7 / 1.26e-7; torch 1.26e-7
        (DType::F32, "step") => 1.8e-7,         // 1.72e-7 / 1.56e-7; torch 1.39e-7
        (DType::F32, "grad_x") => 2.6e-7,       // 2.58e-7 / 2.07e-7; torch 2.04e-7
        (DType::F32, "grad_weight") => 6.1e-7,  // 4.53e-7 / 6.01e-7; torch 4.22e-7
        (DType::F32, "grad_bias") => 4.1e-7,    // 2.71e-7 / 4.09e-7; torch 3.41e-7
        (DType::BF16, "out" | "step") => 3.7e-3, // 3.62e-3 on both; torch 5.05e-3, steps 3.65e-3
        (DType::BF16, "grad_x") => 3.2e-3,      // 3.11e-3 on both; torch 4.26e-3
        (DType::BF16, "grad_weight") => 3.4e-3, // 3.33e-3 on both; torch 4.90e-3
        (DType::BF16, "grad_bias") => 3.4e-3,   // 3.31e-3 on both; torch 3.31e-3
        (DType::F16, "out") => 2.7e-4,          // 2.67e-4 on both; torch 3.56e-4
        (DType::F16, "step") => 3.2e-4,         // 3.20e-4 on both; torch 5.44e-4
        (DType::F16, "grad_x") => 3.2e-4,       // 3.14e-4 on both; torch 3.24e-4
        (DType::F16, "grad_weight") => 3.4e-4,  // 3.38e-4 on both; torch 3.80e-4
        (DType::F16, "grad_bias") => 2.7e-4,    // 2.67e-4 on both; torch 2.68e-4
        _ => panic!("no bound for {dtype:?} {part}"),
    }
}

/// The largest error over the largest magnitude seen per dtype and part, for the summary each
/// test prints, against the bounds of the device's kind (`other`: [`bound`]'s), for the bodies
/// here and for tests that run the cases themselves.
pub struct Worst(BTreeMap<String, f64>, bool);

impl Worst {
    /// None seen yet, against [`bound`]'s `other` bounds where `other`.
    pub fn new(other: bool) -> Self {
        Self(BTreeMap::new(), other)
    }

    /// Asserts `actual` against the recorded `part` of `case` and prints the error.
    fn check(&mut self, label: &str, case: &Case, part: &str, actual: &[f32]) {
        let expected = case.recorded(part);
        self.check_against(label, case, part, part, actual, &expected);
    }

    /// Asserts `actual` against `expected` under the bound of `case`'s dtype and `part`, and
    /// prints the error as `what`.
    pub fn check_against(
        &mut self,
        label: &str,
        case: &Case,
        part: &str,
        what: &str,
        actual: &[f32],
        expected: &[f32],
    ) {
        let (abs, rel) = errors(actual, expected);
        let bound = bound(case.dtype, part, self.1);
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

    /// Prints the largest error per dtype and part.
    pub fn print(&self, label: &str) {
        for (key, rel) in &self.0 {
            println!("{label} worst {key}: rel {rel:.3e}");
        }
    }
}

/// Prints torch's own error on the reduced-precision cases, from its recorded results.
pub fn torch_error(case: &Case, part: &str) {
    let torch = format!("torch_{part}");
    if case.fixtures.has(&case.part(&torch)) {
        let theirs = case.recorded(&torch);
        let truth = case.recorded(part);
        let (abs, rel) = errors(&theirs, &truth);
        println!(
            "  torch {:?} on CUDA {} {part}: max abs {abs:.3e}, rel {rel:.3e}",
            case.dtype, case.name
        );
    }
}

/// The cases in a dtype `adapter`'s device has, the sequences or the decoding steps.
fn cases_on(adapter: &dyn Adapter, step: bool) -> Vec<Case> {
    cases()
        .into_iter()
        .filter(|case| case.step == step && adapter.has(case.dtype))
        .collect()
}

/// Every sequence case: the output and every gradient.
pub fn sequences(adapter: &dyn Adapter, label: &str, other: bool) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let mut worst = Worst::new(other);
    for case in cases_on(adapter, false) {
        let dtype = case.dtype;
        let call = Conv1d {
            x: case.input("x"),
            weight: case.input("weight"),
            bias: case.bias(),
            initial_state: None,
            cu_seqlens: None,
            options: case.options,
            grad: Some(Conv1dCotangents {
                out: Some(case.input("grad_out").with_dtype(DType::F32)),
                final_state: None,
            }),
        };
        let Some(result) = ran(adapter, &case.name, adapter.causal_conv1d(&call)) else {
            continue;
        };
        assert_eq!(result.out.dtype, dtype);
        worst.check(label, &case, "out", &result.out.values);
        let grads = result.grads.expect("the gradients asked for");
        assert_eq!((grads.x.dtype, grads.weight.dtype), (dtype, dtype));
        worst.check(label, &case, "grad_x", &grads.x.values);
        worst.check(label, &case, "grad_weight", &grads.weight.values);
        if case.bias {
            let grad_bias = grads.bias.expect("bias gradient");
            assert_eq!(grad_bias.dtype, dtype);
            worst.check(label, &case, "grad_bias", &grad_bias.values);
        }
        for part in ["out", "grad_x", "grad_weight", "grad_bias"] {
            torch_error(&case, part);
        }
    }
    worst.print(label);
    tally.outcome(())
}

/// Every decoding case from a recorded state, through each of `adapters` in turn: an
/// implementation with more than one way to take a step passes one adapter for each (dash-ops:
/// on a device with gradients, where the step is taken as a constant, and on one without).
pub fn steps(adapters: &[&dyn Adapter], label: &str, other: bool) -> Outcome {
    let tally = Tally::default();
    let adapters: Vec<_> = adapters
        .iter()
        .map(|&adapter| tally.count(adapter))
        .collect();
    let mut worst = Worst::new(other);
    let Some(first) = adapters.first() else {
        return tally.outcome(());
    };
    for case in cases_on(first, true) {
        for adapter in &adapters {
            let call = Conv1dStep {
                x: case.input("x"),
                weight: case.input("weight"),
                bias: case.bias(),
                state: case.state(),
                options: case.options.into(),
            };
            let Some(step) = ran(adapter, &case.name, adapter.causal_conv1d_update(&call)) else {
                continue;
            };
            assert_eq!((step.out.dtype, step.state.dtype), (case.dtype, DType::F32));
            let expected = case.recorded("out");
            worst.check_against(label, &case, "step", "out", &step.out.values, &expected);
            // The state only moves values: exact.
            let next = case.recorded_layout(&step.state);
            let expected = case.recorded("new_state");
            assert_eq!(next, expected, "{label} {}: next state", case.name);
        }
        torch_error(&case, "out");
    }
    worst.print(label);
    tally.outcome(())
}

/// Every sequence case as a prefill of `p` steps, then single steps to its end: the outputs are
/// the recorded sequence's, and the final state holds its last inputs. Returns the largest
/// difference between the steps' outputs and the sequence's own, computed through the same
/// adapter.
pub fn prefill_then_steps(adapter: &dyn Adapter, label: &str, other: bool) -> Outcome<f64> {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let mut worst = Worst::new(other);
    let mut against_sequence = 0.0f64;
    'cases: for case in cases_on(adapter, false) {
        let k = case.kernel_width();
        let x = case.input("x");
        let [b, t, c] = x.shape[..] else {
            panic!("x is [B, T, C]")
        };
        let call = |x: HostTensor, options: CausalConv1dOptions| Conv1d {
            x,
            weight: case.input("weight"),
            bias: case.bias(),
            initial_state: None,
            cu_seqlens: None,
            options,
            grad: None,
        };
        let Some(sequence) = ran(
            adapter,
            &case.name,
            adapter.causal_conv1d(&call(x.clone(), case.options)),
        ) else {
            continue;
        };
        let sequence = sequence.out.values;
        let expected = case.recorded("out");
        // The least state that works, transformers' (`S = K`), and a longer one, each after a
        // prefill of a different length: none, one step, and half the sequence.
        for (prefix, state_len) in [(0, k - 1), (1.min(t), k), (t / 2, k + 2)] {
            // The prompt's outputs and the state it leaves, from one call (none for no prompt).
            let prompt = steps_of(&x, 0..prefix);
            let options = case.options.with_final_state_len(state_len);
            let Some(prefill) = ran(
                adapter,
                &case.name,
                adapter.causal_conv1d(&call(prompt, options)),
            ) else {
                continue 'cases;
            };
            let mut outputs = Vec::new();
            if prefix > 0 {
                outputs.push(prefill.out);
            }
            let mut state = prefill.final_state.expect("the final state");
            for step in prefix..t {
                let token = steps_of(&x, step..step + 1);
                let token = HostTensor::new(token.dtype, vec![b, c], token.values);
                let call = Conv1dStep {
                    x: token,
                    weight: case.input("weight"),
                    bias: case.bias(),
                    state,
                    options: case.options.into(),
                };
                let Some(next) = ran(adapter, &case.name, adapter.causal_conv1d_update(&call))
                else {
                    continue 'cases;
                };
                outputs.push(HostTensor::new(
                    next.out.dtype,
                    vec![b, 1, c],
                    next.out.values,
                ));
                state = next.state;
            }
            let decoded = cat_steps(&outputs, [b, c]);
            let what = format!("prefill {prefix}, state {state_len}");
            worst.check_against(label, &case, "step", &what, &decoded, &expected);
            let (abs, _) = errors(&decoded, &sequence);
            against_sequence = against_sequence.max(abs);

            // The final state is the last `state_len` inputs, zeros before the sequence. The
            // recorded inputs are values of the case's dtype, which the f32 state holds exactly.
            let inputs = case.recorded("x");
            let final_state = state.values;
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
    tally.outcome(against_sequence)
}

/// `[B, T_i, C]` tensors laid end to end along `T`.
fn cat_steps(parts: &[HostTensor], [b, c]: [usize; 2]) -> Vec<f32> {
    let mut values = Vec::with_capacity(parts.iter().map(|p| p.values.len()).sum());
    for row in 0..b {
        for part in parts {
            let t = part.shape[1];
            values.extend_from_slice(&part.values[row * t * c..(row + 1) * t * c]);
        }
    }
    values
}
