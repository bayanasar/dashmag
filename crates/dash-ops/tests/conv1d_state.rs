//! The causal depthwise conv1d from given states and over packed variable-length sequences,
//! against torch, and against itself.
//!
//! `tests/fixtures/conv1d_state/cases.safetensors` holds torch's results in float64 on the CPU,
//! from float32 inputs or from inputs rounded to bfloat16, stored as float32: per sequence the
//! initial state as left context (`torch.cat([state, x])`, then `F.conv1d` with `groups = C` and
//! no padding, then SiLU where asked), the final state the last `W'` steps of
//! `torch.cat([zeros, state, x])`; packed cases as per-sequence calls, concatenated. The loss is
//! `sum(out * grad_out) + sum(final_state * grad_final_state)`, so every gradient (of `x`, the
//! weight, the bias and the initial states) includes what flows back from the final states.
//! Widths 2, 4 and 5; with and without a bias and SiLU; batches of sequences from given states or
//! from zeros, shorter than the window or longer than a run; packed sequences whose boundaries fall
//! inside the kernels' 64-step runs and on them, with empty sequences first, last and between
//! others, and sequences shorter than `K - 1`.
//!
//! Then, on each device, without torch: a sequence split in two, the second part from the first's
//! final state, against the whole sequence; packed sequences against one call per sequence; empty
//! sequences passing their states through; either output alone in the loss; and the kernels
//! against Flex's reference at a size with many runs and boundaries. Every call asserts which
//! path ran (the kernels where `kernels_take` says they take the call, always under CUDA, and
//! then without reading `cu_seqlens` to the host).
//!
//! Errors are stated as the largest absolute difference and as that over the largest magnitude
//! of the reference, per output and gradient, each asserted just above the largest measured over
//! Flex, `cuda` with and without `fusion` and `autotune`, lavapipe and OpenGL over D3D12 through
//! `wgpu`, and CubeCL's CPU runtime (which runs the reference).
//!
//!   cargo test --release -p dash-ops --test conv1d_state -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test conv1d_state
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test conv1d_state
//!   DASH_OPS_DEVICE=cpu cargo test --release -p dash-ops --features cpu --test conv1d_state

mod common;

use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use dash_ops::conv1d::{Activation, CausalConv1dOptions, causal_conv1d, kernels_take, probe};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

/// The tests read process-wide counters, so they run one at a time.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const FIXTURES: &[u8] = include_bytes!("fixtures/conv1d_state/cases.safetensors");

/// One recorded case.
#[derive(Clone, Debug)]
struct Case {
    name: String,
    dtype: DType,
    bias: bool,
    activation: Activation,
    /// The initial states' length `W`, if the case has them.
    w: Option<usize>,
    /// The final states' length `W'`, if the case asks for them.
    w_out: Option<usize>,
    /// The packed sequences' boundaries.
    cu_seqlens: Option<Vec<i64>>,
    k: usize,
}

fn cases() -> Vec<Case> {
    let (_, metadata) = SafeTensors::read_metadata(FIXTURES).expect("conv1d_state fixtures");
    let cases = metadata.metadata().as_ref().expect("metadata")["cases"].clone();
    let cases: BTreeMap<String, Value> = serde_json::from_str(&cases).expect("cases JSON");
    let tensors = SafeTensors::deserialize(FIXTURES).expect("conv1d_state fixtures");
    cases
        .into_iter()
        .map(|(name, case)| {
            let cu_seqlens = (case["kind"] == "packed").then(|| {
                let view = tensors.tensor(&format!("{name}.cu_seqlens")).unwrap();
                assert_eq!(view.dtype(), Dtype::I64);
                view.data()
                    .chunks_exact(8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            });
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
    let tensors = SafeTensors::deserialize(FIXTURES).expect("conv1d_state fixtures");
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

/// `offsets` as `cu_seqlens` of `dtype` (`None`: the device's default int dtype).
fn offsets(offsets: &[i64], device: &Device, dtype: Option<DType>) -> Tensor<1, Int> {
    let data = TensorData::new(offsets.to_vec(), [offsets.len()]);
    match dtype {
        None => Tensor::from_data(data, device),
        Some(dtype) => Tensor::from_data(data.convert_dtype(dtype), (device, dtype)),
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

/// Inputs of one call, as f32 values in a dtype: `x` `[B, T, C]`, `weight` `[C, K]`, `bias`
/// `[C]`, the initial states `[S, W, C]`, the output's gradient and the final states'.
#[derive(Clone, Debug, Default)]
struct Inputs {
    shape: [usize; 4],
    x: Vec<f32>,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    initial: Option<(usize, Vec<f32>)>,
    grad_out: Vec<f32>,
    grad_final: Option<(usize, Vec<f32>)>,
    cu_seqlens: Option<Vec<i64>>,
}

impl Inputs {
    fn sequences(&self) -> usize {
        self.cu_seqlens
            .as_ref()
            .map_or(self.shape[0], |cu| cu.len() - 1)
    }

    fn from_case(case: &Case) -> Self {
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
}

/// The parts of one call through `Autodiff` on `device`, inputs in `dtype` (the states f32),
/// `cu_seqlens` in `int` (`None`: the device's default), keyed by [`PARTS`].
fn run(
    device: &Device,
    dtype: DType,
    inputs: &Inputs,
    activation: Activation,
    int: Option<DType>,
) -> BTreeMap<&'static str, Vec<f32>> {
    let device = device.clone().autodiff();
    let [b, t, c, k] = inputs.shape;
    let s = inputs.sequences();
    let tensor = |values: &[f32], shape: Vec<usize>, dtype: DType| {
        Tensor::<3>::from_data(
            TensorData::new(values.to_vec(), shape).convert_dtype(dtype),
            (&device, dtype),
        )
    };
    let x = tensor(&inputs.x, vec![b, t, c], dtype).require_grad();
    let weight = tensor(&inputs.weight, vec![1, c, k], dtype)
        .reshape([c, k])
        .detach()
        .require_grad();
    let bias = inputs.bias.as_ref().map(|bias| {
        tensor(bias, vec![1, 1, c], dtype)
            .reshape([c])
            .detach()
            .require_grad()
    });
    let initial = inputs
        .initial
        .as_ref()
        .map(|(w, values)| tensor(values, vec![s, *w, c], DType::F32).require_grad());
    let mut options = CausalConv1dOptions::default().with_activation(activation);
    if let Some((w, _)) = &inputs.grad_final {
        options = options.with_final_state_len(*w);
    }
    let cu = inputs
        .cu_seqlens
        .as_ref()
        .map(|cu| offsets(cu, &device, int));
    let (out, final_state) = causal_conv1d(
        x.clone(),
        weight.clone(),
        bias.clone(),
        initial.clone(),
        cu,
        options,
    );
    assert_eq!(out.dtype(), dtype);
    let grad_out = tensor(&inputs.grad_out, vec![b, t, c], dtype);
    // In f32, which the final states' part is in: d loss / d out is `grad_out` exactly.
    let mut loss = (out.clone() * grad_out).sum().cast(DType::F32);
    let mut parts = BTreeMap::new();
    if let (Some(final_state), Some((w, grad))) = (final_state, &inputs.grad_final) {
        assert_eq!(final_state.dtype(), DType::F32);
        assert_eq!(final_state.dims(), [s, *w, c]);
        let grad = tensor(grad, vec![s, *w, c], DType::F32);
        loss = loss + (final_state.clone() * grad).sum();
        parts.insert("final_state", values(final_state));
    }
    parts.insert("out", values(out));
    let grads = loss.backward();
    let grad_x = x.grad(&grads).expect("x gradient");
    let grad_weight = weight.grad(&grads).expect("weight gradient");
    assert_eq!((grad_x.dtype(), grad_weight.dtype()), (dtype, dtype));
    parts.insert("grad_x", values(grad_x));
    parts.insert("grad_weight", values(grad_weight));
    if let Some(bias) = bias {
        let grad = bias.grad(&grads).expect("bias gradient");
        assert_eq!(grad.dtype(), dtype);
        parts.insert("grad_bias", values(grad));
    }
    if let Some(initial) = initial {
        let grad = initial.grad(&grads).expect("initial state gradient");
        assert_eq!(grad.dtype(), DType::F32);
        parts.insert("grad_initial_state", values(grad));
    }
    parts
}

/// Whether the kernels take `inputs` on `device` (CUDA takes them all).
fn takes(device: &Device, dtype: DType, inputs: &Inputs, int: Option<DType>) -> bool {
    let [b, t, c, k] = inputs.shape;
    let packed = inputs.cu_seqlens.as_ref().map(|cu| {
        let int = int.unwrap_or_else(|| default_int(device));
        (cu.len() - 1, int)
    });
    let takes = kernels_take(device, dtype, k, [b, t, c], packed);
    if common::on_cuda(device) {
        assert!(takes, "CUDA turned the kernels away for {:?}", inputs.shape);
    }
    takes
}

/// The dtype `Tensor::from_data` gives int tensors on `device`.
fn default_int(device: &Device) -> DType {
    Tensor::<1, Int>::from_data(TensorData::new(vec![0i64], [1]), device).dtype()
}

/// What the counters showed of one call: whether the kernels ran (forward, backward, those
/// over sequences and those of states), and whether `cu_seqlens` was read to the host.
struct Counters([usize; 6]);

impl Counters {
    fn now() -> Self {
        Self([
            probe::forward_launches(),
            probe::backward_launches(),
            probe::sequence_launches(),
            probe::state_launches(),
            probe::host_reads(),
            probe::backward_runs(),
        ])
    }

    /// Asserts what ran since `self` for a call of `inputs` through `Autodiff`: the written-out
    /// backward always; where `kernels`, the forward and backward kernels, those over sequences
    /// for states or packed sequences, those of states for final or initial states, and no read
    /// of `cu_seqlens` to the host; elsewhere no kernel, and for packed sequences their read.
    fn check(&self, kernels: bool, inputs: &Inputs, what: &str) {
        let now = Counters::now().0;
        let grew = |i: usize| now[i] > self.0[i];
        assert!(grew(5), "{what}: the written-out backward did not run");
        let sequences =
            inputs.initial.is_some() || inputs.cu_seqlens.is_some() || inputs.grad_final.is_some();
        let states = inputs.initial.is_some() || inputs.grad_final.is_some();
        if kernels {
            assert!(grew(0) && grew(1), "{what}: the kernels did not run");
            assert_eq!(grew(2), sequences, "{what}: the kernels over sequences");
            assert_eq!(grew(3), states, "{what}: the kernels of states");
            assert!(!grew(4), "{what}: cu_seqlens read to the host");
        } else {
            assert!(!grew(0) && !grew(1) && !grew(3), "{what}: a kernel ran");
            assert_eq!(grew(4), inputs.cu_seqlens.is_some(), "{what}: host reads");
        }
    }
}

/// Whether `device` is a CubeCL device but a CUDA one (lavapipe, OpenGL, the CPU runtime): the
/// reference sums in other orders on the CPU runtime, and those bounds are a little wider.
#[cfg(feature = "cubecl")]
fn other_runtime(device: &Device) -> bool {
    common::runtime_of(device).is_some() && !common::on_cuda(device)
}

/// Bounds on the error against torch over the largest magnitude, per dtype and part, each just
/// above the largest measured over every build and runtime (in the comments: Flex / the CUDA
/// kernels / lavapipe / OpenGL / the CPU runtime). The final states only move values: exact.
///
/// f32 is within an ulp or two of the largest value; the weight and bias gradients sum every
/// step of every sequence, and err most. bf16 is the f32 result rounded once, half an ulp of
/// bf16 at the largest magnitude, but for the initial states' gradient, which is f32.
fn bound(dtype: DType, part: &str) -> f64 {
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

/// Every case against torch on `device` (with `cu_seqlens` in `int`), returning the largest
/// relative error per dtype and part.
fn fixtures(device: &Device, label: &str, int: Option<DType>) -> BTreeMap<String, f64> {
    let mut worst = BTreeMap::new();
    let (mut cases_run, mut kernels_ran) = (0, 0);
    for case in cases() {
        if !common::has(device, case.dtype) {
            continue;
        }
        let inputs = Inputs::from_case(&case);
        let kernels = takes(device, case.dtype, &inputs, int);
        cases_run += 1;
        kernels_ran += usize::from(kernels);
        let before = Counters::now();
        let parts = run(device, case.dtype, &inputs, case.activation, int);
        before.check(kernels, &inputs, &case.name);
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
    println!("{label}: the kernels took {kernels_ran} of {cases_run} cases");
    worst
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
/// through one autodiff graph, against the whole sequence: outputs, the final state and every
/// gradient. The state between the parts holds `max(K - 1, W')` steps, all the second part's
/// final state can reach back to. Returns the largest relative difference per part.
fn split_against_whole(
    device: &Device,
    dtype: DType,
    whole: &Inputs,
    activation: Activation,
    split: usize,
) -> BTreeMap<&'static str, f64> {
    let device = device.clone().autodiff();
    let [b, t, c, k] = whole.shape;
    let tensor = |values: &[f32], shape: Vec<usize>, dtype: DType| {
        Tensor::<3>::from_data(
            TensorData::new(values.to_vec(), shape).convert_dtype(dtype),
            (&device, dtype),
        )
    };
    let x = tensor(&whole.x, vec![b, t, c], dtype).require_grad();
    let weight = tensor(&whole.weight, vec![1, c, k], dtype)
        .reshape([c, k])
        .detach()
        .require_grad();
    let bias = whole.bias.as_ref().map(|bias| {
        tensor(bias, vec![1, 1, c], dtype)
            .reshape([c])
            .detach()
            .require_grad()
    });
    let initial = whole
        .initial
        .as_ref()
        .map(|(w, values)| tensor(values, vec![b, *w, c], DType::F32).require_grad());
    let (w_out, grad_final) = whole.grad_final.clone().expect("a final state's gradient");
    let w = (k - 1).max(w_out);
    let grad_out = tensor(&whole.grad_out, vec![b, t, c], dtype);
    let grad_final = tensor(&grad_final, vec![b, w_out, c], DType::F32);
    let options = CausalConv1dOptions::default().with_activation(activation);

    let mut results: Vec<BTreeMap<&'static str, Vec<f32>>> = Vec::new();
    for split in [None, Some(split)] {
        let input = x.clone();
        let (out, final_state) = match split {
            None => causal_conv1d(
                input,
                weight.clone(),
                bias.clone(),
                initial.clone(),
                None,
                options.with_final_state_len(w_out),
            ),
            Some(p) => {
                let first = input.clone().slice([0..b, 0..p, 0..c]);
                let second = input.slice([0..b, p..t, 0..c]);
                let (out1, state) = causal_conv1d(
                    first,
                    weight.clone(),
                    bias.clone(),
                    initial.clone(),
                    None,
                    options.with_final_state_len(w),
                );
                let (out2, final_state) = causal_conv1d(
                    second,
                    weight.clone(),
                    bias.clone(),
                    state,
                    None,
                    options.with_final_state_len(w_out),
                );
                (Tensor::cat(vec![out1, out2], 1), final_state)
            }
        };
        let final_state = final_state.expect("the final state");
        let loss = (out.clone() * grad_out.clone()).sum().cast(DType::F32)
            + (final_state.clone() * grad_final.clone()).sum();
        let mut parts = BTreeMap::new();
        parts.insert("out", values(out));
        parts.insert("final_state", values(final_state));
        let grads = loss.backward();
        parts.insert("grad_x", values(x.grad(&grads).unwrap()));
        parts.insert("grad_weight", values(weight.grad(&grads).unwrap()));
        if let Some(bias) = &bias {
            parts.insert("grad_bias", values(bias.grad(&grads).unwrap()));
        }
        if let Some(initial) = &initial {
            parts.insert("grad_initial_state", values(initial.grad(&grads).unwrap()));
        }
        results.push(parts);
    }
    let (whole, split) = (&results[0], &results[1]);
    whole
        .iter()
        .map(|(part, expected)| (*part, errors(&split[part], expected).1))
        .collect()
}

/// Random unpacked inputs: `[B, T, C]`, width `k`, with or without a bias, initial states of `w`
/// steps (or none), and a final state's gradient of `w_out` steps (or none).
#[allow(clippy::too_many_arguments)]
fn random_inputs(
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
fn splits(device: &Device, label: &str) -> BTreeMap<&'static str, f64> {
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
            let differences = split_against_whole(device, DType::F32, &whole, activation, split);
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
    worst
}

/// The split's bounds: the outputs and final states exact (the same arithmetic on the same
/// values), the gradients at f32 rounding (the final state's gradient joins the input gradient
/// in another order, and the weight and bias gradients add two calls' sums), each just above the
/// largest measured (Flex / the CUDA kernels / lavapipe / OpenGL / the CPU runtime, whose
/// reference sums in other orders).
fn assert_splits(worst: &BTreeMap<&'static str, f64>, other: bool) {
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
const LENGTHS: [&[usize]; 3] = [
    &[37, 27, 64, 0, 100, 3, 1, 0, 23],
    &[0, 70, 2, 64, 1, 50, 0],
    &[1, 1, 130, 5, 2, 63, 65, 64, 0],
];

fn cumulative(lengths: &[usize]) -> Vec<i64> {
    let mut cu = vec![0i64];
    for n in lengths {
        cu.push(cu.last().unwrap() + *n as i64);
    }
    cu
}

/// Packed sequences against one call per sequence on the same device: the outputs, final states,
/// input and initial-state gradients exact where the same arithmetic runs; the weight and bias
/// gradients, summed over the sequences in another order, at f32 rounding.
fn packed_against_calls(device: &Device, label: &str) -> BTreeMap<&'static str, f64> {
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
    for (lengths, (k, bias, w, w_out, activation)) in LENGTHS.iter().cycle().zip(configs) {
        let cu = cumulative(lengths);
        let t = *cu.last().unwrap() as usize;
        let packed = random_inputs(&mut seed, [1, t, c, k], bias, w, w_out, Some(cu.clone()));
        let together = run(device, DType::F32, &packed, activation, None);
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
            let parts = run(device, DType::F32, &one, activation, None);
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
    worst
}

/// The packed bounds: exact but for the weight and bias gradients' sums, which the kernels take
/// over runs of 64 steps of the packed sequences rather than of each (Flex and the CPU runtime,
/// running the reference one sequence at a time, sum them as the calls do: exact). Measured: the
/// CUDA kernels / lavapipe / OpenGL.
fn assert_packed(worst: &BTreeMap<&'static str, f64>) {
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

#[cfg(feature = "flex")]
#[test]
fn fixtures_on_flex_match_torch() {
    let _serial = serial();
    fixtures(&Device::flex(), "flex", Some(DType::I64));
}

#[cfg(feature = "flex")]
#[test]
fn a_split_sequence_on_flex_equals_the_whole() {
    let _serial = serial();
    assert_splits(&splits(&Device::flex(), "flex"), false);
}

#[cfg(feature = "flex")]
#[test]
fn packed_sequences_on_flex_equal_one_call_each() {
    let _serial = serial();
    assert_packed(&packed_against_calls(&Device::flex(), "flex"));
}

/// An empty sequence has no outputs; its final state is the last `W'` steps of its initial state
/// (zeros in front where `W' > W`), and its initial state's gradient what the final state's
/// gradient gives those steps.
fn empty_sequences(device: &Device) {
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
        let parts = run(device, DType::F32, &inputs, Activation::Silu, None);
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
}

/// Either output alone in the loss. The final state alone: its gradient reaches the steps it holds
/// (the initial state's where the sequence is shorter), and nothing else. The output alone, the
/// final state asked for and dropped: the gradients of the call without one, to the bit.
fn outputs_alone(device: &Device) {
    let device = device.clone().autodiff();
    let mut seed = 0xa1_0e5a_1e00u64;
    let [b, c, k, w] = [2, 6, 4, 3];
    let silu = CausalConv1dOptions::default().with_activation(Activation::Silu);
    for (t, w_out) in [(70, 5), (2, 5)] {
        let tensor = |values: Vec<f32>, shape: [usize; 3]| {
            Tensor::<3>::from_data(TensorData::new(values, shape), &device)
        };
        let x = tensor(uniform(&mut seed, b * t * c, 3.0), [b, t, c]).require_grad();
        let weight = tensor(uniform(&mut seed, c * k, 0.5), [1, c, k])
            .reshape([c, k])
            .detach()
            .require_grad();
        let bias = tensor(uniform(&mut seed, c, 0.5), [1, 1, c])
            .reshape([c])
            .detach()
            .require_grad();
        let initial = tensor(uniform(&mut seed, b * w * c, 2.0), [b, w, c]).require_grad();
        let g = uniform(&mut seed, b * w_out * c, 1.0);
        let call = |options: CausalConv1dOptions| {
            causal_conv1d(
                x.clone(),
                weight.clone(),
                Some(bias.clone()),
                Some(initial.clone()),
                None,
                options,
            )
        };

        let (_, final_state) = call(silu.with_final_state_len(w_out));
        let loss = (final_state.unwrap() * tensor(g.clone(), [b, w_out, c])).sum();
        let grads = loss.backward();
        let grad_x = values(x.grad(&grads).unwrap());
        let grad_initial = values(initial.grad(&grads).unwrap());
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
        for part in [
            weight.grad(&grads),
            bias.grad(&grads).map(|g| g.reshape([1, c])),
        ] {
            assert!(values(part.unwrap()).iter().all(|&v| v == 0.0));
        }

        let grad_out = tensor(uniform(&mut seed, b * t * c, 1.0), [b, t, c]);
        let mut results = Vec::new();
        for options in [silu.with_final_state_len(w_out), silu] {
            let (out, final_state) = call(options);
            drop(final_state);
            let grads = (out * grad_out.clone()).sum().backward();
            results.push([
                values(x.grad(&grads).unwrap()),
                values(weight.grad(&grads).unwrap()),
                values(bias.grad(&grads).unwrap()),
                values(initial.grad(&grads).unwrap()),
            ]);
        }
        assert_eq!(
            results[0], results[1],
            "T = {t}: the dropped final state changed a gradient"
        );
    }
}

#[cfg(feature = "flex")]
#[test]
fn empty_sequences_on_flex_pass_their_states_through() {
    let _serial = serial();
    empty_sequences(&Device::flex());
}

#[cfg(feature = "flex")]
#[test]
fn either_output_alone_on_flex_has_its_own_gradients() {
    let _serial = serial();
    outputs_alone(&Device::flex());
}

/// The same checks on a CubeCL device, any runtime: the one `DASH_OPS_DEVICE` names.
#[cfg(feature = "cubecl")]
mod cube {
    use super::*;

    fn device() -> Device {
        common::cube_device()
    }

    #[test]
    fn fixtures_on_cubecl_match_torch() {
        let _serial = serial();
        println!("{}", common::describe(&device()));
        // The device's own int dtype, and 64-bit offsets (fla's) where the device has them.
        fixtures(&device(), "cubecl", None);
        if device().supports_dtype(DType::I64) {
            fixtures(&device(), "cubecl i64", Some(DType::I64));
        } else {
            println!("skipped: 64-bit cu_seqlens, which the device does not have");
        }
    }

    #[test]
    fn empty_sequences_on_cubecl_pass_their_states_through() {
        let _serial = serial();
        empty_sequences(&device());
    }

    #[test]
    fn either_output_alone_on_cubecl_has_its_own_gradients() {
        let _serial = serial();
        outputs_alone(&device());
    }

    #[test]
    fn a_split_sequence_on_cubecl_equals_the_whole() {
        let _serial = serial();
        assert_splits(&splits(&device(), "cubecl"), other_runtime(&device()));
    }

    #[test]
    fn packed_sequences_on_cubecl_equal_one_call_each() {
        let _serial = serial();
        assert_packed(&packed_against_calls(&device(), "cubecl"));
    }

    /// The kernels against Flex's reference where the fixtures do not reach: 1000 steps packed
    /// into sequences whose boundaries fall anywhere, hundreds of channel vectors; for several
    /// widths, with and without a bias, SiLU, initial and final states; bf16 too.
    #[cfg(feature = "flex")]
    #[test]
    fn packed_kernels_match_the_reference_at_scale() {
        let _serial = serial();
        let mut seed = 0x7a1e_5ca1_e000_0001u64;
        let c = 1544;
        let silu = Activation::Silu;
        let configs = [
            (4, true, Some(3), Some(3), silu, DType::F32),
            (2, false, None, Some(2), Activation::None, DType::F32),
            (5, true, Some(8), None, silu, DType::F32),
            (8, true, Some(7), Some(9), silu, DType::F32),
            (4, true, Some(3), Some(3), silu, DType::BF16),
        ];
        for (k, bias, w, w_out, activation, dtype) in configs {
            if !common::has(&device(), dtype) {
                continue;
            }
            // Lengths from 0 to 99, as many as fill 1000 steps.
            let mut lengths = Vec::new();
            let mut total = 0;
            while total < 1000 {
                let n = (uniform(&mut seed, 1, 1.0)[0].abs() * 100.0) as usize % 100;
                let n = n.min(1000 - total);
                lengths.push(n);
                total += n;
            }
            let cu = cumulative(&lengths);
            let inputs = random_inputs(&mut seed, [1, 1000, c, k], bias, w, w_out, Some(cu));
            let kernels = takes(&device(), dtype, &inputs, None);
            let before = Counters::now();
            let actual = run(&device(), dtype, &inputs, activation, None);
            before.check(kernels, &inputs, "at scale");
            let expected = run(&Device::flex(), dtype, &inputs, activation, None);
            for (part, expected) in &expected {
                let (abs, rel) = errors(&actual[part], expected);
                // f32: rounding, the weight and bias gradients' long sums most; bf16, an ulp of
                // each side's own rounding (the initial states' gradient is f32). The largest
                // measured: on CUDA, then on lavapipe / OpenGL / the CPU runtime (whose reference
                // sums in other orders; neither of the others has bf16).
                let bound = match (dtype, *part, other_runtime(&device())) {
                    (_, "final_state", _) => 0.0,
                    (DType::F32, "out", _) => 2.6e-7, // 2.58e-7; 2.58e-7 / 2.58e-7 / 1.39e-7
                    (DType::F32, "grad_x", _) => 2.4e-7, // 2.35e-7; 1.56e-7 / 2.35e-7 / 1.56e-7
                    (DType::F32, "grad_weight", false) => 2.9e-7, // 2.81e-7
                    (DType::F32, "grad_weight", true) => 3.8e-7, // 2.81e-7 / 3.75e-7 / 2.11e-7
                    (DType::F32, "grad_bias", false) => 3.3e-7, // 3.27e-7
                    (DType::F32, "grad_bias", true) => 5.1e-7, // 5.03e-7 / 5.03e-7 / 2.01e-7
                    (_, "grad_initial_state", _) => 1.7e-7, // 1.69e-7 on each, f32 and bf16
                    (_, "out", false) => 1.5e-3,      // 1.49e-3
                    (_, "grad_x", false) => 1.1e-3,   // 1.07e-3
                    (_, "out", true) => 3.8e-4,       // 3.72e-4
                    (_, "grad_x", true) => 4.3e-3,    // 4.29e-3
                    (_, "grad_weight" | "grad_bias", false) => 7.0e-7, // 6.94e-7, 0
                    (_, "grad_weight" | "grad_bias", true) => 1.8e-4, // 1.78e-4, 0
                    _ => panic!("no bound for {dtype:?} {part}"),
                };
                println!(
                    "cubecl against flex, {} sequences, K = {k}, bias {bias}, W = {w:?}, \
                     W' = {w_out:?}, {dtype:?}, {part}: max abs {abs:.3e}, rel {rel:.3e}",
                    lengths.len()
                );
                assert!(rel <= bound, "{part}: {rel:.3e} exceeds {bound:.1e}");
            }
        }
    }
}
