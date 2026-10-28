//! The causal depthwise conv1d from given states and over packed variable-length sequences,
//! against torch, and against itself.
//!
//! `conv1d_state/cases.safetensors` holds torch's results in float64 on the CPU, from float32
//! inputs or from inputs rounded to bfloat16, stored as float32: per sequence the initial state as
//! left context (`torch.cat([state, x])`, then `F.conv1d` with `groups = C` and no padding, then
//! SiLU where asked), the final state the last `W'` steps of `torch.cat([zeros, state, x])`;
//! packed cases as per-sequence calls, concatenated. The loss is
//! `sum(out * grad_out) + sum(final_state * grad_final_state)`, so every gradient (of `x`, the
//! weight, the bias and the initial states) includes what flows back from the final states.
//! Widths 2, 4 and 5; with and without a bias and SiLU; batches of sequences from given states or
//! from zeros, shorter than the window or longer than a run; packed sequences whose boundaries fall
//! inside dash-ops' kernels' 64-step runs and on them, with empty sequences first, last and between
//! others, and sequences shorter than `K - 1`.
//!
//! Then, without torch: a sequence split in two, the second part from the first's final state,
//! against the whole sequence; packed sequences against one call per sequence; empty sequences
//! passing their states through; either output alone in the loss.
//!
//! Errors are stated as the largest absolute difference and as that over the largest magnitude
//! of the reference, per output and gradient ([`crate::errors`]), each asserted just above the
//! largest measured over Flex, `cuda` with and without `fusion` and `autotune`, lavapipe and
//! OpenGL over D3D12 through `wgpu`, and CubeCL's CPU runtime (which runs the reference).
//!
//! - [`fixtures`]: every case against torch;
//! - [`splits`] with [`assert_splits`]: a sequence split in two against the whole;
//! - [`packed_against_calls`] with [`assert_packed`]: packed sequences against a call each;
//! - [`empty_sequences`], [`outputs_alone`].

use crate::fixtures::File;
use crate::{Adapter, Conv1d, Conv1dCotangents, DType, HostTensor, Outcome, Tally, errors, ran};
use dash_contract::conv1d::{Activation, CausalConv1dOptions};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The recording, read once.
fn recording() -> &'static File {
    static FILE: OnceLock<File> = OnceLock::new();
    FILE.get_or_init(|| File::read("conv1d_state/cases.safetensors"))
}

/// One recorded case.
#[derive(Clone, Debug)]
pub struct Case {
    /// The case's name in the recording.
    pub name: String,
    /// The dtype of `x`, the weight and the bias.
    pub dtype: DType,
    /// Whether it has a bias.
    pub bias: bool,
    /// The activation.
    pub activation: Activation,
    /// The initial states' length `W`, if the case has them.
    pub w: Option<usize>,
    /// The final states' length `W'`, if the case asks for them.
    pub w_out: Option<usize>,
    /// The packed sequences' boundaries.
    pub cu_seqlens: Option<Vec<i64>>,
    /// The kernel width.
    pub k: usize,
}

/// Every recorded case.
pub fn cases() -> Vec<Case> {
    let cases = recording().metadata()["cases"].clone();
    let cases: BTreeMap<String, Value> = serde_json::from_str(&cases).expect("cases JSON");
    cases
        .into_iter()
        .map(|(name, case)| {
            let cu_seqlens =
                (case["kind"] == "packed").then(|| recording().ints(&format!("{name}.cu_seqlens")));
            Case {
                dtype: match case["dtype"].as_str().unwrap() {
                    "f32" => DType::F32,
                    "bf16" => DType::BF16,
                    other => panic!("unknown dtype {other}"),
                },
                bias: case["bias"].as_bool().unwrap(),
                activation: match case["silu"].as_bool().unwrap() {
                    true => Activation::Silu,
                    false => Activation::None,
                },
                w: case["W"].as_u64().map(|w| w as usize),
                w_out: case["W_out"].as_u64().map(|w| w as usize),
                k: case["K"].as_u64().unwrap() as usize,
                cu_seqlens,
                name,
            }
        })
        .collect()
}

/// A recorded tensor: its values and shape.
fn recorded(name: &str) -> (Vec<f32>, Vec<usize>) {
    let tensor = recording().tensor(name);
    (tensor.values, tensor.shape)
}

/// Inputs of one call, as f32 values in a dtype: `x` `[B, T, C]`, `weight` `[C, K]`, `bias`
/// `[C]`, the initial states `[S, W, C]`, the output's gradient and the final states'.
#[derive(Clone, Debug, Default)]
pub struct Inputs {
    /// `[B, T, C, K]`.
    pub shape: [usize; 4],
    /// `x`.
    pub x: Vec<f32>,
    /// The weight.
    pub weight: Vec<f32>,
    /// The bias.
    pub bias: Option<Vec<f32>>,
    /// The initial states' length and values.
    pub initial: Option<(usize, Vec<f32>)>,
    /// The output's gradient.
    pub grad_out: Vec<f32>,
    /// The final states' length and their gradient.
    pub grad_final: Option<(usize, Vec<f32>)>,
    /// The packed sequences' boundaries.
    pub cu_seqlens: Option<Vec<i64>>,
}

impl Inputs {
    /// The number of sequences, `S`.
    pub fn sequences(&self) -> usize {
        self.cu_seqlens
            .as_ref()
            .map_or(self.shape[0], |cu| cu.len() - 1)
    }

    /// A recorded case's inputs.
    pub fn from_case(case: &Case) -> Self {
        let part = |part: &str| format!("{}.{part}", case.name);
        let (x, x_shape) = recorded(&part("x"));
        let [b, t, c] = [x_shape[0], x_shape[1], x_shape[2]];
        Self {
            shape: [b, t, c, case.k],
            x,
            weight: recorded(&part("weight")).0,
            bias: case.bias.then(|| recorded(&part("bias")).0),
            initial: case.w.map(|w| (w, recorded(&part("initial_state")).0)),
            grad_out: recorded(&part("grad_out")).0,
            grad_final: case
                .w_out
                .map(|w| (w, recorded(&part("grad_final_state")).0)),
            cu_seqlens: case.cu_seqlens.clone(),
        }
    }

    /// The call these inputs make in `dtype` (the states f32), with the final states asked for
    /// where there is a final states' gradient, and the gradients.
    fn call(&self, dtype: DType, activation: Activation) -> Conv1d {
        let [b, t, c, k] = self.shape;
        let s = self.sequences();
        let mut options = CausalConv1dOptions::default().with_activation(activation);
        if let Some((w, _)) = &self.grad_final {
            options = options.with_final_state_len(*w);
        }
        Conv1d {
            x: HostTensor::new(dtype, vec![b, t, c], self.x.clone()),
            weight: HostTensor::new(dtype, vec![c, k], self.weight.clone()),
            bias: self
                .bias
                .as_ref()
                .map(|bias| HostTensor::new(dtype, vec![c], bias.clone())),
            initial_state: self
                .initial
                .as_ref()
                .map(|(w, values)| HostTensor::f32(vec![s, *w, c], values.clone())),
            cu_seqlens: self.cu_seqlens.clone(),
            options,
            grad: Some(Conv1dCotangents {
                out: Some(HostTensor::f32(vec![b, t, c], self.grad_out.clone())),
                final_state: self
                    .grad_final
                    .as_ref()
                    .map(|(w, grad)| HostTensor::f32(vec![s, *w, c], grad.clone())),
            }),
        }
    }
}

/// The parts of one call of `inputs` in `dtype` (the states f32), keyed by the recording's names,
/// with their dtypes and shapes checked; `None` if the adapter does not run it.
pub fn run(
    adapter: &dyn Adapter,
    dtype: DType,
    inputs: &Inputs,
    activation: Activation,
) -> Option<BTreeMap<&'static str, Vec<f32>>> {
    let [b, t, c, k] = inputs.shape;
    let s = inputs.sequences();
    let what = format!("[{b}, {t}, {c}] K = {k}");
    let result = ran(
        adapter,
        &what,
        adapter.causal_conv1d(&inputs.call(dtype, activation)),
    )?;
    assert_eq!(result.out.dtype, dtype);
    let mut parts = BTreeMap::new();
    if let (Some(final_state), Some((w, _))) = (result.final_state, &inputs.grad_final) {
        assert_eq!(final_state.dtype, DType::F32);
        assert_eq!(final_state.shape, [s, *w, c]);
        parts.insert("final_state", final_state.values);
    }
    parts.insert("out", result.out.values);
    let grads = result.grads.expect("the gradients asked for");
    assert_eq!((grads.x.dtype, grads.weight.dtype), (dtype, dtype));
    parts.insert("grad_x", grads.x.values);
    parts.insert("grad_weight", grads.weight.values);
    if inputs.bias.is_some() {
        let grad = grads.bias.expect("bias gradient");
        assert_eq!(grad.dtype, dtype);
        parts.insert("grad_bias", grad.values);
    }
    if inputs.initial.is_some() {
        let grad = grads.initial_state.expect("initial state gradient");
        assert_eq!(grad.dtype, DType::F32);
        parts.insert("grad_initial_state", grad.values);
    }
    Some(parts)
}

/// Bounds on the error against torch over the largest magnitude, per dtype and part, each just
/// above the largest measured over every build and runtime (in the comments: Flex / the CUDA
/// kernels / lavapipe / OpenGL / the CPU runtime). The final states only move values: exact.
///
/// f32 is within an ulp or two of the largest value; the weight and bias gradients sum every
/// step of every sequence, and err most. bf16 is the f32 result rounded once, half an ulp of
/// bf16 at the largest magnitude, but for the initial states' gradient, which is f32.
pub fn bound(dtype: DType, part: &str) -> f64 {
    match (dtype, part) {
        (_, "final_state") => 0.0,
        (DType::F32, "out") => 1.8e-7, // 1.74e-7 / 1.39e-7 / 1.39e-7 / 1.39e-7 / 1.74e-7
        (DType::F32, "grad_x") => 1.7e-7, // 1.66e-7 / 1.62e-7 / 1.62e-7 / 1.62e-7 / 1.62e-7
        (DType::F32, "grad_weight") => 4.1e-7, // 3.99e-7 / 2.55e-7 / 3.15e-7 / 2.99e-7 / 3.85e-7
        (DType::F32, "grad_bias") => 4.5e-7, // 4.42e-7 / 3.97e-7 / 3.97e-7 / 3.68e-7 / 4.32e-7
        (DType::F32, "grad_initial_state") => 1.2e-7, // 1.10e-7 / 1.18e-7 / 1.18e-7 / 1.18e-7 / 1.10e-7
        (DType::BF16, "out") => 2.7e-3, // 2.70e-3 on each with bf16 (Flex, CUDA, CPU runtime)
        (DType::BF16, "grad_x") => 2.6e-3, // 2.53e-3 on each
        (DType::BF16, "grad_weight") => 2.5e-3, // 2.45e-3 on each
        (DType::BF16, "grad_bias") => 2.1e-3, // 2.02e-3 on each
        (DType::BF16, "grad_initial_state") => 1.5e-7, // 1.32e-7 / 1.32e-7 / 1.45e-7 (CPU runtime)
        _ => panic!("no bound for {dtype:?} {part}"),
    }
}

/// Every case against torch, returning the largest relative error per dtype and part. The cases
/// in a dtype the device does not have are skipped, saying so.
pub fn fixtures(adapter: &dyn Adapter, label: &str) -> Outcome<BTreeMap<String, f64>> {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let mut worst = BTreeMap::new();
    for case in cases() {
        if !adapter.has(case.dtype) {
            println!("skipped: {} has no {:?}", adapter.describe(), case.dtype);
            continue;
        }
        let inputs = Inputs::from_case(&case);
        let Some(parts) = run(adapter, case.dtype, &inputs, case.activation) else {
            continue;
        };
        for (part, actual) in &parts {
            let (expected, _) = recorded(&format!("{}.{part}", case.name));
            let (abs, rel) = errors(actual, &expected);
            let bound = bound(case.dtype, part);
            println!(
                "{label} {} {part}: max abs {abs:.3e}, rel {rel:.3e} (bound {bound:.1e})",
                case.name
            );
            let entry = worst
                .entry(format!("{:?} {part}", case.dtype))
                .or_insert(0.0f64);
            *entry = entry.max(rel);
            assert!(
                rel <= bound,
                "{label} {} {part}: {rel:.3e} exceeds {bound:.1e}",
                case.name
            );
        }
    }
    for (key, rel) in &worst {
        println!("{label} worst {key}: rel {rel:.3e}");
    }
    tally.outcome(worst)
}

/// A seeded uniform draw in `[-scale, scale)`.
pub fn uniform(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `values` of a `[N, T, C]` tensor, steps `range` of each row.
fn steps_of(values: &[f32], [n, t, c]: [usize; 3], range: std::ops::Range<usize>) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * range.len() * c);
    for row in 0..n {
        for step in range.clone() {
            out.extend_from_slice(&values[(row * t + step) * c..(row * t + step + 1) * c]);
        }
    }
    out
}

/// A sequence of `T` steps split at `split`, the second part from the first's final state,
/// against the whole sequence: outputs, the final state and every gradient. The state between
/// the parts holds `max(K - 1, W')` steps, all the second part's final state can reach back to.
/// The gradients compose as one autodiff graph through both parts composes them: the second
/// part's initial-state gradient is the first part's final-state gradient, and the first part's
/// weight and bias gradients are added to the second's. Returns the largest relative difference
/// per part, or `None` if the adapter does not run a call.
fn split_against_whole(
    adapter: &dyn Adapter,
    dtype: DType,
    whole: &Inputs,
    activation: Activation,
    split: usize,
) -> Option<BTreeMap<&'static str, f64>> {
    let [b, t, c, k] = whole.shape;
    let (w_out, _) = whole.grad_final.clone().expect("a final state's gradient");
    let w = (k - 1).max(w_out);
    let what = format!("[{b}, {t}, {c}] K = {k} split at {split}");
    let whole_run = run(adapter, dtype, whole, activation)?;

    // The second part from the first's final state, then the first from its initial state with
    // the gradient the second gave its initial state.
    let first_x = steps_of(&whole.x, [b, t, c], 0..split);
    let second = Inputs {
        shape: [b, t - split, c, k],
        x: steps_of(&whole.x, [b, t, c], split..t),
        grad_out: steps_of(&whole.grad_out, [b, t, c], split..t),
        initial: None,
        ..whole.clone()
    };
    let mut first = Inputs {
        shape: [b, split, c, k],
        x: first_x,
        grad_out: steps_of(&whole.grad_out, [b, t, c], 0..split),
        grad_final: None,
        ..whole.clone()
    };
    let mut first_call = first.call(dtype, activation);
    first_call.options = first_call.options.with_final_state_len(w);
    first_call.grad = None;
    let state = ran(adapter, &what, adapter.causal_conv1d(&first_call))?
        .final_state
        .expect("the final state");
    let second = Inputs {
        initial: Some((w, state.values)),
        ..second
    };
    let second_run = run(adapter, dtype, &second, activation)?;
    first.grad_final = Some((w, second_run["grad_initial_state"].clone()));
    let first_run = run(adapter, dtype, &first, activation)?;

    let mut parts: BTreeMap<&'static str, Vec<f32>> = BTreeMap::new();
    let both = |part: &str| {
        let mut values = Vec::new();
        let (a, z) = (&first_run[part], &second_run[part]);
        let (ta, tz) = (split, t - split);
        for row in 0..b {
            values.extend_from_slice(&a[row * ta * c..(row + 1) * ta * c]);
            values.extend_from_slice(&z[row * tz * c..(row + 1) * tz * c]);
        }
        values
    };
    parts.insert("out", both("out"));
    parts.insert("final_state", second_run["final_state"].clone());
    parts.insert("grad_x", both("grad_x"));
    let sum = |part: &str| -> Vec<f32> {
        first_run[part]
            .iter()
            .zip(&second_run[part])
            .map(|(a, z)| z + a)
            .collect()
    };
    parts.insert("grad_weight", sum("grad_weight"));
    if whole.bias.is_some() {
        parts.insert("grad_bias", sum("grad_bias"));
    }
    if whole.initial.is_some() {
        parts.insert(
            "grad_initial_state",
            first_run["grad_initial_state"].clone(),
        );
    }
    Some(
        whole_run
            .iter()
            .map(|(part, expected)| (*part, errors(&parts[part], expected).1))
            .collect(),
    )
}

/// Random unpacked inputs: `[B, T, C]`, width `k`, with or without a bias, initial states of `w`
/// steps (or none), and a final state's gradient of `w_out` steps (or none).
pub fn random_inputs(
    seed: &mut u64,
    [b, t, c, k]: [usize; 4],
    bias: bool,
    w: Option<usize>,
    w_out: Option<usize>,
    cu_seqlens: Option<Vec<i64>>,
) -> Inputs {
    let s = cu_seqlens.as_ref().map_or(b, |cu| cu.len() - 1);
    Inputs {
        shape: [b, t, c, k],
        x: uniform(seed, b * t * c, 3.0),
        weight: uniform(seed, c * k, 0.5),
        bias: bias.then(|| uniform(seed, c, 0.5)),
        initial: w.map(|w| (w, uniform(seed, s * w * c, 2.0))),
        grad_out: uniform(seed, b * t * c, 1.0),
        grad_final: w_out.map(|w| (w, uniform(seed, s * w * c, 1.0))),
        cu_seqlens,
    }
}

/// Splits of a sequence of 200 steps in two, at a step shorter than the window, inside a run, at
/// a run's edge, near the end; widths 2, 4 and 5; from a given state and from zeros.
pub fn splits(adapter: &dyn Adapter, label: &str) -> Outcome<BTreeMap<&'static str, f64>> {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let mut worst: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut seed = 0x51ed_2701_7a11_c0deu64;
    let silu = Activation::Silu;
    let configs = [
        (4, true, Some(3), 3, silu),
        (4, false, None, 3, silu),
        (5, true, Some(6), 6, Activation::None),
        (2, true, Some(1), 4, silu),
    ];
    for (k, bias, w, w_out, activation) in configs {
        let whole = random_inputs(&mut seed, [2, 200, 6, k], bias, w, Some(w_out), None);
        for split in [1, 2, 37, 64, 150, 199] {
            let Some(differences) =
                split_against_whole(adapter, DType::F32, &whole, activation, split)
            else {
                continue;
            };
            for (part, rel) in differences {
                println!(
                    "{label} split K = {k}, W = {w:?}, W' = {w_out}, at {split}: {part} rel {rel:.3e}"
                );
                let entry = worst.entry(part).or_insert(0.0);
                *entry = entry.max(rel);
            }
        }
    }
    for (part, rel) in &worst {
        println!("{label} split worst {part}: rel {rel:.3e}");
    }
    tally.outcome(worst)
}

/// The split's bounds: the outputs and final states exact (the same arithmetic on the same
/// values), the gradients at f32 rounding (the final state's gradient joins the input gradient
/// in another order, and the weight and bias gradients add two calls' sums), each just above the
/// largest measured (Flex / the CUDA kernels / lavapipe / OpenGL / the CPU runtime, whose
/// reference sums in other orders); `other` for the wider bounds of the runtimes but CUDA.
pub fn assert_splits(worst: &BTreeMap<&'static str, f64>, other: bool) {
    for (part, rel) in worst {
        let bound = match (*part, other) {
            ("out" | "final_state", _) => 0.0,
            ("grad_x", _) => 1.0e-7, // 5.82e-8 / 9.05e-8 / 5.82e-8 / 9.05e-8 / 5.82e-8
            ("grad_initial_state", _) => 1.1e-7, // 7.98e-8 / 1.01e-7 / 7.98e-8 / 1.01e-7 / 7.98e-8
            ("grad_weight", false) => 4.6e-7, // 4.48e-7 / 2.24e-7
            ("grad_weight", true) => 9.0e-7, // 2.36e-7 / 2.24e-7 / 8.96e-7
            ("grad_bias", false) => 7.6e-7, // 7.52e-7 / 2.91e-7
            ("grad_bias", true) => 1.3e-6, // 3.26e-7 / 3.11e-7 / 1.26e-6
            _ => panic!("no bound for {part}"),
        };
        assert!(*rel <= bound, "split {part}: {rel:.3e} exceeds {bound:.1e}");
    }
}

/// Sequence lengths that put boundaries everywhere: inside runs, on run edges, empty sequences
/// first, between others and last, sequences of one and two steps.
pub const LENGTHS: [&[usize]; 3] = [
    &[37, 27, 64, 0, 100, 3, 1, 0, 23],
    &[0, 70, 2, 64, 1, 50, 0],
    &[1, 1, 130, 5, 2, 63, 65, 64, 0],
];

/// The boundaries of sequences of `lengths`.
pub fn cumulative(lengths: &[usize]) -> Vec<i64> {
    let mut cu = vec![0i64];
    for n in lengths {
        cu.push(cu.last().unwrap() + *n as i64);
    }
    cu
}

/// Packed sequences against one call per sequence through the same adapter: the outputs, final
/// states, input and initial-state gradients exact where the same arithmetic runs; the weight and
/// bias gradients, summed over the sequences in another order, at f32 rounding.
pub fn packed_against_calls(
    adapter: &dyn Adapter,
    label: &str,
) -> Outcome<BTreeMap<&'static str, f64>> {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let mut worst: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut seed = 0x0dd5_eed5_1234_5678u64;
    let c = 6;
    let configs = [
        (4, true, Some(3), Some(3), Activation::Silu),
        (5, false, Some(6), Some(4), Activation::None),
        (2, true, None, None, Activation::Silu),
        (4, true, None, Some(5), Activation::Silu),
        (3, false, Some(2), None, Activation::Silu),
    ];
    'configs: for (lengths, (k, bias, w, w_out, activation)) in LENGTHS.iter().cycle().zip(configs)
    {
        let cu = cumulative(lengths);
        let t = *cu.last().unwrap() as usize;
        let packed = random_inputs(&mut seed, [1, t, c, k], bias, w, w_out, Some(cu.clone()));
        let Some(together) = run(adapter, DType::F32, &packed, activation) else {
            continue;
        };
        let mut apart: BTreeMap<&'static str, Vec<f32>> = BTreeMap::new();
        let mut weight_sum = vec![0.0f32; c * k];
        let mut bias_sum = vec![0.0f32; c];
        for (s, span) in cu.windows(2).enumerate() {
            let range = span[0] as usize..span[1] as usize;
            let state = |states: &Option<(usize, Vec<f32>)>| {
                states
                    .as_ref()
                    .map(|(w, values)| (*w, values[s * w * c..(s + 1) * w * c].to_vec()))
            };
            let one = Inputs {
                shape: [1, range.len(), c, k],
                x: steps_of(&packed.x, [1, t, c], range.clone()),
                grad_out: steps_of(&packed.grad_out, [1, t, c], range.clone()),
                initial: state(&packed.initial),
                grad_final: state(&packed.grad_final),
                cu_seqlens: None,
                ..packed.clone()
            };
            if range.is_empty() && one.initial.is_none() && one.grad_final.is_none() {
                continue;
            }
            let Some(parts) = run(adapter, DType::F32, &one, activation) else {
                continue 'configs;
            };
            for (part, values) in parts {
                match part {
                    "grad_weight" => weight_sum
                        .iter_mut()
                        .zip(&values)
                        .for_each(|(sum, v)| *sum += v),
                    "grad_bias" => bias_sum
                        .iter_mut()
                        .zip(&values)
                        .for_each(|(sum, v)| *sum += v),
                    _ => apart.entry(part).or_default().extend(values),
                }
            }
        }
        apart.insert("grad_weight", weight_sum);
        if bias {
            apart.insert("grad_bias", bias_sum);
        }
        for (part, expected) in &apart {
            let actual = &together[part];
            let (abs, rel) = errors(actual, expected);
            println!(
                "{label} packed {lengths:?} K = {k}, W = {w:?}, W' = {w_out:?}: {part} max abs \
                 {abs:.3e}, rel {rel:.3e}"
            );
            let entry = worst.entry(part).or_insert(0.0);
            *entry = entry.max(rel);
        }
    }
    for (part, rel) in &worst {
        println!("{label} packed worst {part}: rel {rel:.3e}");
    }
    tally.outcome(worst)
}

/// The packed bounds: exact but for the weight and bias gradients' sums, which dash-ops' kernels
/// take over runs of 64 steps of the packed sequences rather than of each (Flex and the CPU
/// runtime, running the reference one sequence at a time, sum them as the calls do: exact).
/// Measured: the CUDA kernels / lavapipe / OpenGL.
pub fn assert_packed(worst: &BTreeMap<&'static str, f64>) {
    for (part, rel) in worst {
        let bound = match *part {
            "grad_weight" => 2.3e-7, // 2.24e-7 / 1.71e-7 / 2.24e-7
            "grad_bias" => 3.3e-7,   // 2.44e-7 / 3.25e-7 / 1.63e-7
            _ => 0.0,
        };
        assert!(
            *rel <= bound,
            "packed {part}: {rel:.3e} exceeds {bound:.1e}"
        );
    }
}

/// An empty sequence has no outputs; its final state is the last `W'` steps of its initial state
/// (zeros in front where `W' > W`), and its initial state's gradient what the final state's
/// gradient gives those steps.
pub fn empty_sequences(adapter: &dyn Adapter) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let mut seed = 0x00e5_e9e5_eb1c_e000u64;
    for w_out in [2, 3, 5] {
        let cu = vec![0i64, 4, 4, 9];
        let inputs = random_inputs(
            &mut seed,
            [1, 9, 4, 4],
            true,
            Some(3),
            Some(w_out),
            Some(cu),
        );
        let Some(parts) = run(adapter, DType::F32, &inputs, Activation::Silu) else {
            continue;
        };
        let (w, initial) = inputs.initial.as_ref().unwrap();
        let (_, grad_final) = inputs.grad_final.as_ref().unwrap();
        let c = 4;
        let final_state = &parts["final_state"][w_out * c..2 * w_out * c];
        let grad_initial = &parts["grad_initial_state"][w * c..2 * w * c];
        for i in 0..w_out {
            for ch in 0..c {
                let step = (i + w).checked_sub(w_out);
                let want = step.map_or(0.0, |step| initial[(w + step) * c + ch]);
                assert_eq!(final_state[i * c + ch], want, "final state row {i}");
            }
        }
        for i in 0..*w {
            for ch in 0..c {
                let row = (i + w_out).checked_sub(*w);
                let want = row.map_or(0.0, |row| grad_final[(w_out + row) * c + ch]);
                assert_eq!(grad_initial[i * c + ch], want, "initial gradient row {i}");
            }
        }
    }
    tally.outcome(())
}

/// Either output alone in the loss. The final state alone: its gradient reaches the steps it holds
/// (the initial state's where the sequence is shorter), and nothing else. The output alone, the
/// final state asked for and dropped: the gradients of the call without one, to the bit.
pub fn outputs_alone(adapter: &dyn Adapter) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let mut seed = 0xa1_0e5a_1e00u64;
    let [b, c, k, w] = [2, 6, 4, 3];
    let silu = CausalConv1dOptions::default().with_activation(Activation::Silu);
    for (t, w_out) in [(70, 5), (2, 5)] {
        let x = uniform(&mut seed, b * t * c, 3.0);
        let weight = uniform(&mut seed, c * k, 0.5);
        let bias = uniform(&mut seed, c, 0.5);
        let initial = uniform(&mut seed, b * w * c, 2.0);
        let g = uniform(&mut seed, b * w_out * c, 1.0);
        let call = |options: CausalConv1dOptions, grad: Conv1dCotangents| Conv1d {
            x: HostTensor::f32(vec![b, t, c], x.clone()),
            weight: HostTensor::f32(vec![c, k], weight.clone()),
            bias: Some(HostTensor::f32(vec![c], bias.clone())),
            initial_state: Some(HostTensor::f32(vec![b, w, c], initial.clone())),
            cu_seqlens: None,
            options,
            grad: Some(grad),
        };

        let final_alone = Conv1dCotangents {
            out: None,
            final_state: Some(HostTensor::f32(vec![b, w_out, c], g.clone())),
        };
        let what = format!("[{b}, {t}, {c}] K = {k}, the final state alone");
        let Some(result) = ran(
            adapter,
            &what,
            adapter.causal_conv1d(&call(silu.with_final_state_len(w_out), final_alone)),
        ) else {
            continue;
        };
        let grads = result.grads.expect("the gradients asked for");
        let grad_x = &grads.x.values;
        let grad_initial = &grads.initial_state.as_ref().unwrap().values;
        // History step p of a sequence (the initial state's W, then its T inputs) is final state
        // row p - (W + T - W'), where that is one.
        for bi in 0..b {
            for p in 0..w + t {
                for ch in 0..c {
                    let row = (p + w_out).checked_sub(w + t);
                    let want = row.map_or(0.0, |row| g[(bi * w_out + row) * c + ch]);
                    let got = match p.checked_sub(w) {
                        Some(step) => grad_x[(bi * t + step) * c + ch],
                        None => grad_initial[(bi * w + p) * c + ch],
                    };
                    assert_eq!(got, want, "T = {t}: history step {p}");
                }
            }
        }
        for part in [Some(&grads.weight), grads.bias.as_ref()] {
            assert!(part.unwrap().values.iter().all(|&v| v == 0.0));
        }

        let grad_out = uniform(&mut seed, b * t * c, 1.0);
        let mut results = Vec::new();
        for options in [silu.with_final_state_len(w_out), silu] {
            let out_alone = Conv1dCotangents {
                out: Some(HostTensor::f32(vec![b, t, c], grad_out.clone())),
                final_state: None,
            };
            let what = format!("[{b}, {t}, {c}] K = {k}, the output alone");
            let Some(result) = ran(
                adapter,
                &what,
                adapter.causal_conv1d(&call(options, out_alone)),
            ) else {
                break;
            };
            let grads = result.grads.expect("the gradients asked for");
            results.push([
                grads.x.values,
                grads.weight.values,
                grads.bias.unwrap().values,
                grads.initial_state.unwrap().values,
            ]);
        }
        if let [with, without] = &results[..] {
            assert_eq!(
                with, without,
                "T = {t}: the dropped final state changed a gradient"
            );
        }
    }
    tally.outcome(())
}
