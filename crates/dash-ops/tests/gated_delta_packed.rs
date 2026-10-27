//! The gated delta rule over packed variable-length sequences, with a query scale of its own, and
//! with heads 256 wide: the chunked rule and the step, held to fla 0.5.2 (packed, with
//! `cu_seqlens`) and to transformers' torch functions in float64 called once per sequence.
//!
//! The fixtures in `tests/fixtures/gated_delta_packed/` hold the inputs, the gradients fed back
//! (`d_out`, `d_state`) and the results, with the suffix `_f64` (transformers in float64, one call
//! per sequence, stored rounded to f32) and `_fla` (fla over the packed inputs, f32 in IEEE on
//! CUDA; for the step, no gradients; not for the 256-wide cases, the largest files, where fla
//! agrees with float64 to f32 rounding as on the others):
//!
//! - `chunk_f32`: six sequences of 37, 64, 0, 100, 5 and 1 tokens (boundaries inside chunks, a
//!   sequence one chunk long, an empty one, one of a single token), 2 heads, Dk = Dv = 16, queries
//!   and keys normalised, initial states and final-state gradients;
//! - `chunk_f32_scale`: 70 and 130 tokens, Dk 32, Dv 16, the scale 0.3, no normalisation, no
//!   initial state, no final-state gradient;
//! - `chunk_bf16`: 50, 90 and 20 tokens, Dk = Dv = 32, bf16 inputs (`g` in f32), against float64
//!   from the same values (fla's own bf16 results beside them);
//! - `chunk_wide_k`, `chunk_wide_v`: 40 and 25 tokens, Dk = 256 (Dv 24) and Dv = 256 (Dk 16);
//! - `step_f32`, `step_f32_scale`: the step over packed tokens, sequences of 1, 3, 0 and 2 tokens
//!   (the l2 normalisation), and of 2, 1 and 4 (the scale 0.7).
//!
//! Errors are the largest absolute difference and that over the largest absolute reference value
//! ("relative"); the relative one is asserted just above the largest measured over every build
//! (Flex; `cuda`; `cuda` with `fusion`, with `autotune`, with both) and on lavapipe and CubeCL's
//! CPU runtime.
//!
//!   cargo test --release -p dash-ops --test gated_delta_packed -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test gated_delta_packed
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test gated_delta_packed

mod common;

use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use dash_ops::gated_delta::chunk::{self, ChunkGatedDeltaOptions, chunk_gated_delta_rule};
use dash_ops::gated_delta::recurrent::{GatedDeltaStepOptions, gated_delta_step};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

/// One test at a time, so that the counters each test reads count only that test's work.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The chunked rule, f32, against transformers in float64 [measured: 1.23e-6].
const F32_VS_F64: f32 = 1.5e-6;
/// The chunked rule, f32, against fla's packed f32 path [measured: 2.03e-6]. fla computes its
/// gates in base 2 and in its own order, a few f32 roundings from the exact result itself.
const F32_VS_FLA: f32 = 2.5e-6;
/// bf16 inputs against float64 from the same values, where the reference or the f32 kernels run:
/// one bf16 rounding of each output and gradient, at most 2^-9 relative to its largest value
/// [measured: 2.58e-3]; fla's own bf16 path is further off (6.4e-3, printed beside it).
const BF16_VS_F64: f32 = 3e-3;
/// The same where the kernels on tensor cores run, with bf16 operands as fla's [measured: 5.58e-3,
/// the queries' gradient; fla's 6.36e-3].
#[cfg(feature = "cubecl")]
const BF16_TC_VS_F64: f32 = 6e-3;
/// The step, f32, its output and new states against transformers in float64 and fla [measured:
/// 2.3e-7].
const STEP_VS_REF: f32 = 3e-7;
/// The tracked step's gradients (the reference on the autodiff backend) against float64
/// [measured: 3.1e-7].
const STEP_GRAD: f32 = 4e-7;
/// Packed against per-sequence calls of the same operation on the same device: every chunk's
/// arithmetic is the same, so they agree to the bit [measured: 0 on every build].
const PACKED_VS_SEQUENCES: f32 = 0.0;

/// One recorded fixture: its tensors as f32 (the boundaries as i64) and its metadata.
struct Fixture {
    name: &'static str,
    meta: HashMap<String, serde_json::Value>,
    floats: HashMap<String, (Vec<usize>, Vec<f32>)>,
    cu_seqlens: Vec<i64>,
}

impl Fixture {
    fn load(name: &'static str) -> Self {
        let path = format!(
            "{}/tests/fixtures/gated_delta_packed/{name}.safetensors",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let (_, metadata) = SafeTensors::read_metadata(&bytes).unwrap();
        let meta = metadata
            .metadata()
            .as_ref()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::from_str(v).unwrap()))
            .collect();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let mut floats = HashMap::new();
        let mut cu_seqlens = Vec::new();
        for (key, view) in st.tensors() {
            match view.dtype() {
                Dtype::F32 => {
                    let data = view
                        .data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    floats.insert(key, (view.shape().to_vec(), data));
                }
                Dtype::I64 => {
                    assert_eq!(key, "cu_seqlens");
                    cu_seqlens = view
                        .data()
                        .chunks_exact(8)
                        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                }
                other => panic!("{key}: unexpected dtype {other:?}"),
            }
        }
        Self {
            name,
            meta,
            floats,
            cu_seqlens,
        }
    }

    fn flag(&self, key: &str) -> bool {
        self.meta[key].as_bool().unwrap()
    }

    fn scale(&self) -> Option<f32> {
        self.meta["scale"].as_f64().map(|x| x as f32)
    }

    fn has(&self, key: &str) -> bool {
        self.floats.contains_key(key)
    }

    fn get(&self, key: &str) -> &[f32] {
        &self
            .floats
            .get(key)
            .unwrap_or_else(|| panic!("{}: no {key}", self.name))
            .1
    }

    fn tensor<const D: usize>(&self, key: &str, device: &Device) -> Tensor<D> {
        let (shape, data) = &self.floats[key];
        Tensor::from_data(TensorData::new(data.clone(), shape.clone()), device)
    }

    fn cu_seqlens(&self, device: &Device) -> Tensor<1, Int> {
        cu_seqlens(&self.cu_seqlens, device)
    }

    /// The elements of one sequence's state, `H Dk Dv`.
    fn state_len(&self) -> usize {
        ["heads", "dk", "dv"]
            .iter()
            .map(|key| self.meta[*key].as_u64().unwrap() as usize)
            .product()
    }

    /// The sequences' token spans.
    fn spans(&self) -> Vec<std::ops::Range<usize>> {
        self.cu_seqlens
            .windows(2)
            .map(|w| w[0] as usize..w[1] as usize)
            .collect()
    }
}

fn cu_seqlens(offsets: &[i64], device: &Device) -> Tensor<1, Int> {
    Tensor::from_data(TensorData::new(offsets.to_vec(), [offsets.len()]), device)
}

/// What one run gave: outputs and gradients as f32 vectors.
type Results = HashMap<&'static str, Vec<f32>>;

fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.cast(DType::F32)
        .into_data()
        .try_into_vec::<f32>()
        .unwrap()
}

/// Max abs error, and max abs error relative to the reference's max abs value.
fn errors(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_err = 0f32;
    let mut scale = 0f32;
    for (a, e) in actual.iter().zip(expected) {
        assert!(a.is_finite(), "non-finite value {a}");
        max_err = max_err.max((a - e).abs());
        scale = scale.max(e.abs());
    }
    (max_err, max_err / scale.max(f32::MIN_POSITIVE))
}

/// Compares every result with the fixture's `<name><suffix>`, printing the errors, and asserts
/// the relative error is at most `tol`.
fn check(case: &Fixture, label: &str, res: &Results, suffix: &str, tol: f32) {
    let mut keys: Vec<_> = res
        .keys()
        .copied()
        .filter(|key| case.has(&format!("{key}{suffix}")))
        .collect();
    keys.sort();
    assert!(
        !keys.is_empty(),
        "{}: nothing to compare with {suffix}",
        case.name
    );
    let mut failed = Vec::new();
    for key in keys {
        let (abs, rel) = errors(&res[key], case.get(&format!("{key}{suffix}")));
        println!(
            "{:16} {label:12} {key:6} vs {:8} max abs {abs:.3e} rel {rel:.3e}",
            case.name,
            &suffix[1..]
        );
        if rel > tol {
            failed.push(format!("{key} {rel:.3e}"));
        }
    }
    assert!(
        failed.is_empty(),
        "{} {label}: relative errors above {tol:.1e} against {suffix}: {failed:?}",
        case.name
    );
}

/// The options a chunked fixture was recorded with, the final states returned.
fn chunk_options(case: &Fixture) -> ChunkGatedDeltaOptions {
    ChunkGatedDeltaOptions {
        qk_l2norm: case.flag("l2norm"),
        scale: case.scale(),
        output_final_state: true,
    }
}

/// The chunked rule over the fixture's packed sequences on `device` (an autodiff device), the
/// queries, keys, values and `beta` in `dtype` (`g` stays f32): forward and backward.
fn run_chunk(case: &Fixture, device: &Device, dtype: DType) -> Results {
    let leaf4 = |key: &str| case.tensor::<4>(key, device).cast(dtype).require_grad();
    let (q, k, v) = (leaf4("q"), leaf4("k"), leaf4("v"));
    let g = case.tensor::<3>("g", device).require_grad();
    let beta = case.tensor::<3>("beta", device).cast(dtype).require_grad();
    let h0 = case
        .flag("initial_state")
        .then(|| case.tensor::<4>("h0", device).require_grad());
    let runs = chunk::probe::backward_runs();
    let (out, state) = chunk_gated_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        h0.clone(),
        Some(case.cu_seqlens(device)),
        chunk_options(case),
    );
    let state = state.expect("the final states, asked for");
    assert_eq!(out.dtype(), dtype);
    assert_eq!(state.dtype(), DType::F32);
    let mut loss = (out.clone().cast(DType::F32) * case.tensor::<4>("d_out", device)).sum();
    if case.flag("final_state_grad") {
        loss = loss + (state.clone() * case.tensor::<4>("d_state", device)).sum();
    }
    let grads = loss.backward();
    assert!(
        chunk::probe::backward_runs() > runs,
        "the written-out backward did not run"
    );
    let mut res = Results::new();
    res.insert("out", to_vec(out));
    res.insert("state", to_vec(state));
    res.insert("dq", to_vec(q.grad(&grads).unwrap()));
    res.insert("dk", to_vec(k.grad(&grads).unwrap()));
    res.insert("dv", to_vec(v.grad(&grads).unwrap()));
    res.insert("dg", to_vec(g.grad(&grads).unwrap()));
    res.insert("dbeta", to_vec(beta.grad(&grads).unwrap()));
    if let Some(h0) = h0 {
        res.insert("dh0", to_vec(h0.grad(&grads).unwrap()));
    }
    res
}

/// The same inputs, each sequence a call of its own (a batch of one, unpacked), the results laid
/// out as the packed call's.
fn run_chunk_per_sequence(case: &Fixture, device: &Device) -> Results {
    let options = chunk_options(case);
    let mut parts: HashMap<&'static str, Vec<Vec<f32>>> = HashMap::new();
    let mut push = |key: &'static str, x: Vec<f32>| parts.entry(key).or_default().push(x);
    for (s, span) in case.spans().into_iter().enumerate() {
        let slice4 = |key: &str| {
            case.tensor::<4>(key, device)
                .narrow(1, span.start, span.len())
        };
        let slice3 = |key: &str| {
            case.tensor::<3>(key, device)
                .narrow(1, span.start, span.len())
        };
        let state_of = |key: &str| case.tensor::<4>(key, device).narrow(0, s, 1);
        let h0 = case
            .flag("initial_state")
            .then(|| state_of("h0").require_grad());
        if span.is_empty() {
            // Nothing to run: the state passes through, and so does its gradient.
            let zeros = || vec![0.0; case.state_len()];
            push("state", h0.map_or_else(zeros, to_vec));
            if case.flag("initial_state") {
                let d = if case.flag("final_state_grad") {
                    to_vec(state_of("d_state"))
                } else {
                    zeros()
                };
                push("dh0", d);
            }
            continue;
        }
        let (q, k, v) = (
            slice4("q").require_grad(),
            slice4("k").require_grad(),
            slice4("v").require_grad(),
        );
        let (g, beta) = (slice3("g").require_grad(), slice3("beta").require_grad());
        let (out, state) = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            h0.clone(),
            None,
            options,
        );
        let state = state.unwrap();
        let mut loss = (out.clone() * slice4("d_out")).sum();
        if case.flag("final_state_grad") {
            loss = loss + (state.clone() * state_of("d_state")).sum();
        }
        let grads = loss.backward();
        push("out", to_vec(out));
        push("state", to_vec(state));
        push("dq", to_vec(q.grad(&grads).unwrap()));
        push("dk", to_vec(k.grad(&grads).unwrap()));
        push("dv", to_vec(v.grad(&grads).unwrap()));
        push("dg", to_vec(g.grad(&grads).unwrap()));
        push("dbeta", to_vec(beta.grad(&grads).unwrap()));
        if let Some(h0) = h0 {
            push("dh0", to_vec(h0.grad(&grads).unwrap()));
        }
    }
    // Per sequence, token-major tensors are [1, len, H, D]: concatenated in order they are the
    // packed tensors; states [1, H, Dk, Dv] likewise.
    parts
        .into_iter()
        .map(|(key, xs)| (key, xs.concat()))
        .collect()
}

/// Packed against per-sequence calls on one device: the largest relative difference.
fn packed_against_sequences(case: &Fixture, device: &Device, label: &str) -> f32 {
    let packed = run_chunk(case, device, DType::F32);
    let apart = run_chunk_per_sequence(case, device);
    let mut keys: Vec<_> = packed.keys().copied().collect();
    keys.sort();
    let mut worst = 0f32;
    for key in keys {
        let (abs, rel) = errors(&packed[key], &apart[key]);
        println!(
            "{:16} {label:12} {key:6} packed vs per sequence max abs {abs:.3e} rel {rel:.3e}",
            case.name
        );
        worst = worst.max(rel);
    }
    worst
}

const CHUNK_F32: [&str; 4] = [
    "chunk_f32",
    "chunk_f32_scale",
    "chunk_wide_k",
    "chunk_wide_v",
];

fn check_chunk_f32(device: &Device, label: &str) {
    for name in CHUNK_F32 {
        let case = Fixture::load(name);
        let res = run_chunk(&case, device, DType::F32);
        check(&case, label, &res, "_f64", F32_VS_F64);
        if case.flag("fla") {
            check(&case, label, &res, "_fla", F32_VS_FLA);
        }
    }
}

fn check_chunk_bf16(device: &Device, label: &str, tol: f32) {
    let case = Fixture::load("chunk_bf16");
    let res = run_chunk(&case, device, DType::BF16);
    check(&case, label, &res, "_f64", tol);
    // For the record: fla's own bf16 path against the same float64 reference.
    let mut keys: Vec<_> = res.keys().copied().collect();
    keys.sort();
    for key in keys {
        let (abs, rel) = errors(
            case.get(&format!("{key}_fla")),
            case.get(&format!("{key}_f64")),
        );
        println!(
            "{:16} {:12} {key:6} vs f64      max abs {abs:.3e} rel {rel:.3e}",
            case.name, "fla-bf16"
        );
    }
}

/// The step over the fixture's packed tokens on `device`, without gradients: output and states.
fn run_step(case: &Fixture, device: &Device) -> Results {
    let options = GatedDeltaStepOptions {
        qk_l2norm: case.flag("l2norm"),
        scale: case.scale(),
    };
    let (out, state) = gated_delta_step(
        case.tensor("q", device),
        case.tensor("k", device),
        case.tensor("v", device),
        case.tensor("g", device),
        case.tensor("beta", device),
        case.tensor("h0", device),
        Some(case.cu_seqlens(device)),
        options,
    );
    assert_eq!(state.dtype(), DType::F32);
    Results::from([("out", to_vec(out)), ("state", to_vec(state))])
}

/// The step with tracked inputs (the reference on the autodiff backend): its gradients.
fn run_step_gradients(case: &Fixture, device: &Device) -> Results {
    let options = GatedDeltaStepOptions {
        qk_l2norm: case.flag("l2norm"),
        scale: case.scale(),
    };
    let leaf3 = |key: &str| case.tensor::<3>(key, device).require_grad();
    let leaf2 = |key: &str| case.tensor::<2>(key, device).require_grad();
    let (q, k, v) = (leaf3("q"), leaf3("k"), leaf3("v"));
    let (g, beta) = (leaf2("g"), leaf2("beta"));
    let h0 = case.tensor::<4>("h0", device).require_grad();
    let (out, state) = gated_delta_step(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        h0.clone(),
        Some(case.cu_seqlens(device)),
        options,
    );
    let loss = (out * case.tensor::<3>("d_out", device)).sum()
        + (state * case.tensor::<4>("d_state", device)).sum();
    let grads = loss.backward();
    Results::from([
        ("dq", to_vec(q.grad(&grads).unwrap())),
        ("dk", to_vec(k.grad(&grads).unwrap())),
        ("dv", to_vec(v.grad(&grads).unwrap())),
        ("dg", to_vec(g.grad(&grads).unwrap())),
        ("dbeta", to_vec(beta.grad(&grads).unwrap())),
        ("dh0", to_vec(h0.grad(&grads).unwrap())),
    ])
}

/// The step token by token, each sequence apart: what the packed step must give.
fn run_step_per_token(case: &Fixture, device: &Device) -> Results {
    let options = GatedDeltaStepOptions {
        qk_l2norm: case.flag("l2norm"),
        scale: case.scale(),
    };
    let row3 = |key: &str, t: usize| case.tensor::<3>(key, device).narrow(0, t, 1);
    let row2 = |key: &str, t: usize| case.tensor::<2>(key, device).narrow(0, t, 1);
    let (mut outs, mut states) = (Vec::new(), Vec::new());
    for (s, span) in case.spans().into_iter().enumerate() {
        let mut state = case.tensor::<4>("h0", device).narrow(0, s, 1);
        for t in span {
            let (out, next) = gated_delta_step(
                row3("q", t),
                row3("k", t),
                row3("v", t),
                row2("g", t),
                row2("beta", t),
                state,
                None,
                options,
            );
            outs.push(to_vec(out));
            state = next;
        }
        states.push(to_vec(state));
    }
    Results::from([("out", outs.concat()), ("state", states.concat())])
}

const STEPS: [&str; 2] = ["step_f32", "step_f32_scale"];

fn check_steps(device: &Device, label: &str) {
    for name in STEPS {
        let case = Fixture::load(name);
        let res = run_step(&case, device);
        check(&case, label, &res, "_f64", STEP_VS_REF);
        check(&case, label, &res, "_fla", STEP_VS_REF);
        let apart = run_step_per_token(&case, device);
        for key in ["out", "state"] {
            let (abs, rel) = errors(&res[key], &apart[key]);
            println!(
                "{name:16} {label:12} {key:6} packed vs token by token max abs {abs:.3e} rel {rel:.3e}"
            );
            assert!(rel <= PACKED_VS_SEQUENCES, "{name} {key}: {rel:.3e}");
        }
    }
}

fn check_step_gradients(device: &Device, label: &str) {
    for name in STEPS {
        let case = Fixture::load(name);
        let res = run_step_gradients(&case, &device.clone().autodiff());
        check(&case, label, &res, "_f64", STEP_GRAD);
    }
}

#[cfg(feature = "flex")]
mod flex {
    use super::*;

    #[test]
    fn packed_chunks_on_flex_match_fla_and_transformers() {
        let _serial = serial();
        check_chunk_f32(&Device::flex().autodiff(), "flex");
    }

    #[test]
    fn packed_bf16_chunks_on_flex() {
        let _serial = serial();
        check_chunk_bf16(&Device::flex().autodiff(), "flex-bf16", BF16_VS_F64);
    }

    #[test]
    fn packed_chunks_on_flex_are_per_sequence_calls() {
        let _serial = serial();
        let device = Device::flex().autodiff();
        for name in ["chunk_f32", "chunk_f32_scale"] {
            let worst = packed_against_sequences(&Fixture::load(name), &device, "flex");
            assert!(worst <= PACKED_VS_SEQUENCES, "{name}: {worst:.3e}");
        }
    }

    #[test]
    fn packed_steps_on_flex_match_fla_and_transformers() {
        let _serial = serial();
        check_steps(&Device::flex(), "flex");
    }

    #[test]
    fn tracked_packed_steps_on_flex_take_the_reference_gradients() {
        let _serial = serial();
        check_step_gradients(&Device::flex(), "flex");
    }

    /// The default scale is `1 / sqrt(Dk)`: giving it explicitly changes nothing.
    #[test]
    fn the_default_scale_is_one_over_the_root_of_dk() {
        let _serial = serial();
        let case = Fixture::load("chunk_f32");
        let device = Device::flex();
        let dk = case.meta["dk"].as_u64().unwrap() as f32;
        let run = |options: ChunkGatedDeltaOptions| {
            let (out, state) = chunk_gated_delta_rule(
                case.tensor("q", &device),
                case.tensor("k", &device),
                case.tensor("v", &device),
                case.tensor("g", &device),
                case.tensor("beta", &device),
                Some(case.tensor("h0", &device)),
                Some(case.cu_seqlens(&device)),
                options,
            );
            (to_vec(out), to_vec(state.unwrap()))
        };
        let default = ChunkGatedDeltaOptions::default()
            .with_qk_l2norm(true)
            .with_final_state(true);
        assert_eq!(run(default), run(default.with_scale(1.0 / dk.sqrt())));
    }
}

/// Seeded normal values (xorshift and Box-Muller), the same on every device.
#[cfg(feature = "cubecl")]
fn normal(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..len)
        .map(|_| {
            let (u1, u2) = (uniform(), uniform());
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

/// The same checks on a CubeCL device, any runtime: the one `DASH_OPS_DEVICE` names.
#[cfg(feature = "cubecl")]
mod cube {
    use super::*;
    use dash_ops::gated_delta::chunk::{
        chunk_gated_delta_rule_reference, chunk_gated_delta_rule_reference_backward, kernels_take,
        tensor_cores_take,
    };
    use dash_ops::gated_delta::recurrent::{self, gated_delta_step_reference, kernel_takes};

    /// The kernel launches a packed forward and backward pass over `[1, T, H, Dk, Dv]` in
    /// `dtype` make on `device`: four forward and, backward, seven on tensor cores or ten on the
    /// f32 kernels where the kernels take them (the chunks' layout first in each), none where the
    /// reference does. Under CUDA the kernels take heads up to their width.
    fn expected_launches(device: &Device, dtype: DType, dims: [usize; 5], seqs: usize) -> usize {
        let int = cu_seqlens(&[0, 1], device).dtype();
        let [forward, backward] = kernels_take(device, dtype, dims, Some((seqs, int)));
        if common::runtime_of(device) == Some("cuda")
            && dims[3].max(dims[4]) <= chunk::MAX_KERNEL_HEAD_DIM
        {
            assert!(
                forward && backward,
                "CUDA turned the kernels away for {dims:?}"
            );
        }
        let [_, tc_backward] = tensor_cores_take(device, dtype, dims, Some((seqs, int)));
        let backward_launches = if tc_backward { 7 } else { 10 };
        4 * usize::from(forward) + backward_launches * usize::from(backward)
    }

    /// How many of those run on tensor cores: three forward and six backward (all but the
    /// chunks' layout) where the kernels on tensor cores take them, which under CUDA they do for
    /// bf16 inputs with heads of 32, 64 or 128.
    fn expected_tc_launches(device: &Device, dtype: DType, dims: [usize; 5], seqs: usize) -> usize {
        let int = cu_seqlens(&[0, 1], device).dtype();
        let [forward, backward] = tensor_cores_take(device, dtype, dims, Some((seqs, int)));
        let narrow = |d: usize| matches!(d, 32 | 64 | 128);
        if common::runtime_of(device) == Some("cuda")
            && matches!(dtype, DType::BF16 | DType::F16)
            && narrow(dims[3])
            && narrow(dims[4])
        {
            assert!(
                forward && backward,
                "CUDA turned the tensor cores away for {dims:?}"
            );
        }
        3 * usize::from(forward) + 6 * usize::from(backward)
    }

    fn has(device: &Device, dtype: DType) -> bool {
        let has = device.supports_dtype(dtype);
        if !has {
            println!("skipped: {} has no {dtype:?}", common::describe(device));
        }
        has
    }

    /// Asserts that `f` launched exactly `expected` kernels, `tc` of them on tensor cores.
    fn launches<R>(expected: usize, tc: usize, f: impl FnOnce() -> R) -> R {
        let before = chunk::probe::kernel_launches();
        let before_tc = chunk::probe::tensor_core_launches();
        let result = f();
        let launched = chunk::probe::kernel_launches() - before;
        let launched_tc = chunk::probe::tensor_core_launches() - before_tc;
        assert_eq!(
            (launched, launched_tc),
            (expected, tc),
            "the kernels made {launched} launches ({launched_tc} on tensor cores), expected \
             {expected} ({tc})"
        );
        result
    }

    fn dims_of(case: &Fixture) -> ([usize; 5], usize) {
        let q = &case.floats["q"].0;
        let v = &case.floats["v"].0;
        ([q[0], q[1], q[2], q[3], v[3]], case.cu_seqlens.len() - 1)
    }

    #[test]
    fn packed_chunks_on_cubecl_match_fla_and_transformers() {
        let _serial = serial();
        let device = common::cube_device();
        println!("{}", common::describe(&device));
        for name in CHUNK_F32 {
            let case = Fixture::load(name);
            let (dims, seqs) = dims_of(&case);
            let expected = expected_launches(&device, DType::F32, dims, seqs);
            println!("{name}: {dims:?}, {seqs} sequences, {expected} kernel launches");
            let res = launches(expected, 0, || {
                run_chunk(&case, &device.clone().autodiff(), DType::F32)
            });
            check(&case, "cubecl", &res, "_f64", F32_VS_F64);
            if case.flag("fla") {
                check(&case, "cubecl", &res, "_fla", F32_VS_FLA);
            }
        }
    }

    #[test]
    fn packed_bf16_chunks_on_cubecl() {
        let _serial = serial();
        let device = common::cube_device();
        if !has(&device, DType::BF16) {
            return;
        }
        let case = Fixture::load("chunk_bf16");
        let (dims, seqs) = dims_of(&case);
        let expected = expected_launches(&device, DType::BF16, dims, seqs);
        let tc = expected_tc_launches(&device, DType::BF16, dims, seqs);
        let tol = if tc > 0 { BF16_TC_VS_F64 } else { BF16_VS_F64 };
        launches(expected, tc, || {
            check_chunk_bf16(&device.clone().autodiff(), "cubecl-bf16", tol)
        });
    }

    #[test]
    fn packed_chunks_on_cubecl_are_per_sequence_calls() {
        let _serial = serial();
        let device = common::cube_device().autodiff();
        for name in ["chunk_f32", "chunk_f32_scale", "chunk_wide_k"] {
            let worst = packed_against_sequences(&Fixture::load(name), &device, "cubecl");
            assert!(worst <= PACKED_VS_SEQUENCES, "{name}: {worst:.3e}");
        }
    }

    fn t4(x: &[f32], dims: [usize; 4], device: &Device) -> Tensor<4> {
        Tensor::from_data(TensorData::new(x.to_vec(), dims), device)
    }

    fn t3(x: &[f32], dims: [usize; 3], device: &Device) -> Tensor<3> {
        Tensor::from_data(TensorData::new(x.to_vec(), dims), device)
    }

    /// Packed sequences of `lens` tokens, `h` heads of `d`: the kernels (forward, and backward
    /// through Autodiff) against the reference (packed, its sequences one at a time) on the same
    /// device, from the same inputs rounded to `dtype`. Returns the largest relative error over the
    /// outputs and gradients, printing each.
    fn kernels_against_reference(lens: &[usize], h: usize, d: usize, dtype: DType) -> (f32, bool) {
        let device = common::cube_device();
        let ad = device.clone().autodiff();
        // Where no kernel takes the shape the reference runs (lavapipe, OpenGL, CubeCL's CPU
        // runtime): sequences of at most 48 tokens, boundaries still inside chunks, do there.
        let full: usize = lens.iter().sum();
        let int = cu_seqlens(&[0, 1], &device).dtype();
        let shorter: Vec<usize>;
        let lens = if kernels_take(&device, dtype, [1, full, h, d, d], Some((lens.len(), int)))
            == [false, false]
        {
            shorter = lens.iter().map(|&n| n.min(48)).collect();
            &shorter[..]
        } else {
            lens
        };
        let t: usize = lens.iter().sum();
        let seqs = lens.len();
        let mut offsets = vec![0i64];
        for &n in lens {
            offsets.push(offsets.last().unwrap() + n as i64);
        }
        let expected = expected_launches(&device, dtype, [1, t, h, d, d], seqs);
        let tc = expected_tc_launches(&device, dtype, [1, t, h, d, d], seqs);
        let x = |seed: u64, len: usize| normal(seed, len);
        let unit = |v: Vec<f32>| -> Vec<f32> {
            v.chunks(d)
                .flat_map(|r| {
                    let n = r.iter().map(|x| x * x).sum::<f32>().sqrt();
                    r.iter().map(move |x| x / n).collect::<Vec<_>>()
                })
                .collect()
        };
        let q = unit(x(1, t * h * d));
        let k = unit(x(2, t * h * d));
        let v = x(3, t * h * d);
        let g: Vec<f32> = x(4, t * h)
            .into_iter()
            .map(|y| -(1.0 + (y - 0.5).exp()).ln() * 0.4)
            .collect();
        let beta: Vec<f32> = x(5, t * h)
            .into_iter()
            .map(|y| 1.0 / (1.0 + (-y).exp()))
            .collect();
        let h0: Vec<f32> = x(6, seqs * h * d * d)
            .into_iter()
            .map(|y| y * 0.5)
            .collect();
        let d_out = x(7, t * h * d);
        let d_state = x(8, seqs * h * d * d);
        let tok4 = [1, t, h, d];
        let st4 = [seqs, h, d, d];
        let options = ChunkGatedDeltaOptions::default().with_final_state(true);

        let leaf4 = |v: &[f32]| t4(v, tok4, &ad).cast(dtype).require_grad();
        let leaf3 = |v: &[f32]| t3(v, [1, t, h], &ad).cast(dtype).require_grad();
        let (lq, lk, lv) = (leaf4(&q), leaf4(&k), leaf4(&v));
        let (lg, lb) = (leaf3(&g), leaf3(&beta));
        let lh0 = t4(&h0, st4, &ad).require_grad();
        let ours = launches(expected, tc, || {
            let (out, state) = chunk_gated_delta_rule(
                lq.clone(),
                lk.clone(),
                lv.clone(),
                lg.clone(),
                lb.clone(),
                Some(lh0.clone()),
                Some(cu_seqlens(&offsets, &ad)),
                options,
            );
            let state = state.unwrap();
            let loss = (out.clone().cast(DType::F32) * t4(&d_out, tok4, &ad)).sum()
                + (state.clone() * t4(&d_state, st4, &ad)).sum();
            let grads = loss.backward();
            [
                to_vec(out),
                to_vec(state),
                to_vec(lq.grad(&grads).unwrap()),
                to_vec(lk.grad(&grads).unwrap()),
                to_vec(lv.grad(&grads).unwrap()),
                to_vec(lg.grad(&grads).unwrap()),
                to_vec(lb.grad(&grads).unwrap()),
                to_vec(lh0.grad(&grads).unwrap()),
            ]
        });
        let r4 = |v: &[f32]| t4(v, tok4, &device).cast(dtype).cast(DType::F32);
        let r3 = |v: &[f32]| t3(v, [1, t, h], &device).cast(dtype).cast(DType::F32);
        let (out, state) = chunk_gated_delta_rule_reference(
            r4(&q),
            r4(&k),
            r4(&v),
            r3(&g),
            r3(&beta),
            Some(t4(&h0, st4, &device)),
            Some(cu_seqlens(&offsets, &device)),
            options,
        );
        let (dq, dk, dv, dg, dbeta, dh0) = chunk_gated_delta_rule_reference_backward(
            r4(&q),
            r4(&k),
            r4(&v),
            r3(&g),
            r3(&beta),
            Some(t4(&h0, st4, &device)),
            Some(cu_seqlens(&offsets, &device)),
            t4(&d_out, tok4, &device),
            t4(&d_state, st4, &device),
            options,
        );
        let theirs = [
            to_vec(out),
            to_vec(state),
            to_vec(dq),
            to_vec(dk),
            to_vec(dv),
            to_vec(dg),
            to_vec(dbeta),
            to_vec(dh0),
        ];
        let names = ["out", "state", "dq", "dk", "dv", "dg", "dbeta", "dh0"];
        let label = format!("{lens:?} H{h} D{d} {dtype:?}");
        let mut worst = 0f32;
        for ((name, a), e) in names.iter().zip(&ours).zip(&theirs) {
            let (abs, rel) = errors(a, e);
            println!("{label:32} {name:6} vs reference  max abs {abs:.3e} rel {rel:.3e}");
            worst = worst.max(rel);
        }
        (worst, tc > 0)
    }

    /// Qwen3.5-0.8B's head width, sequences ending inside chunks and one shorter than a chunk:
    /// the kernels within 8e-7 of the reference in f32 [measured: 6.3e-7].
    #[test]
    fn packed_kernels_match_the_reference() {
        let _serial = serial();
        let (worst, _) = kernels_against_reference(&[100, 37, 163], 3, 128, DType::F32);
        assert!(worst <= 8e-7, "relative error {worst:.3e}");
    }

    /// The same in bf16: the f32 kernels within a bf16 rounding of the f32 reference [measured:
    /// 3.07e-3], those on tensor cores, with bf16 operands, within 5e-3 [measured: 4.70e-3, the
    /// final states].
    #[test]
    fn packed_kernels_match_the_reference_in_bf16() {
        let _serial = serial();
        if !has(&common::cube_device(), DType::BF16) {
            return;
        }
        let (worst, tc) = kernels_against_reference(&[90, 110], 2, 128, DType::BF16);
        let tol = if tc { 5e-3 } else { 3.5e-3 };
        assert!(worst <= tol, "relative error {worst:.3e}");
    }

    /// Heads 256 wide, packed: the kernels where they take them, else the reference, against the
    /// reference [measured: 6.8e-7].
    #[test]
    fn wide_heads_match_the_reference() {
        let _serial = serial();
        let (worst, _) = kernels_against_reference(&[70, 30, 64], 2, 256, DType::F32);
        assert!(worst <= 8e-7, "relative error {worst:.3e}");
    }

    /// Whether the step's kernel takes the fixture's packed tokens on `device`. Under CUDA it
    /// takes every head it holds.
    fn step_takes(device: &Device, case: &Fixture) -> bool {
        let q = &case.floats["q"].0;
        let dv = case.floats["v"].0[2];
        let int = cu_seqlens(&[0, 1], device).dtype();
        let seqs = case.cu_seqlens.len() - 1;
        let takes = kernel_takes(
            device,
            [DType::F32; 3],
            [q[0], q[1], q[2], dv],
            Some((seqs, int)),
        );
        if common::runtime_of(device) == Some("cuda") {
            assert!(takes, "CUDA turned the step's kernel away");
        }
        takes
    }

    #[test]
    fn packed_steps_on_cubecl_match_fla_and_transformers() {
        let _serial = serial();
        let device = common::cube_device();
        for name in STEPS {
            let case = Fixture::load(name);
            let kernel = step_takes(&device, &case);
            println!("{name}: the step's kernel takes it: {kernel}");
            let before = recurrent::probe::kernel_launches();
            let res = run_step(&case, &device);
            let launched = recurrent::probe::kernel_launches() - before;
            assert_eq!(launched, usize::from(kernel), "{name}: {launched} launches");
            check(&case, "cubecl", &res, "_f64", STEP_VS_REF);
            check(&case, "cubecl", &res, "_fla", STEP_VS_REF);
        }
        check_steps(&device, "cubecl");
    }

    #[test]
    fn tracked_packed_steps_on_cubecl_take_the_reference_gradients() {
        let _serial = serial();
        let before = recurrent::probe::kernel_launches();
        check_step_gradients(&common::cube_device(), "cubecl");
        assert_eq!(
            recurrent::probe::kernel_launches(),
            before,
            "a tracked step ran the kernel"
        );
    }

    /// Decoding sizes, sequences bringing 0 to 4 tokens: the step's kernel against the
    /// reference, f32 and bf16, and the scale option [measured: output 4.7e-7 in f32, 1.2e-3 in
    /// bf16 (each rounds nearly equal f32 values to bf16 on its own); states 1.9e-7].
    #[test]
    fn packed_step_kernel_matches_the_reference() {
        let _serial = serial();
        let device = common::cube_device();
        let counts = [1usize, 2, 4, 1, 0, 3, 1, 2];
        let (seqs, h, d) = (counts.len(), 16, 128);
        let t: usize = counts.iter().sum();
        let mut offsets = vec![0i64];
        for &n in &counts {
            offsets.push(offsets.last().unwrap() + n as i64);
        }
        for (dtype, options, tol) in [
            (
                DType::F32,
                GatedDeltaStepOptions::default().with_qk_l2norm(true),
                [6e-7, 3e-7],
            ),
            (
                DType::F32,
                GatedDeltaStepOptions::default().with_scale(0.5),
                [6e-7, 3e-7],
            ),
            (
                DType::BF16,
                GatedDeltaStepOptions::default().with_qk_l2norm(true),
                [1.5e-3, 3e-7],
            ),
        ] {
            if !has(&device, dtype) {
                continue;
            }
            let tok = |x: Vec<f32>, w: usize| t3(&x, [t, h, w], &device).cast(dtype);
            // Without the normalisation in the step, the caller normalises, as models do.
            let unit = |x: Vec<f32>| -> Vec<f32> {
                if options.qk_l2norm {
                    return x;
                }
                x.chunks(d)
                    .flat_map(|r| {
                        let n = r.iter().map(|y| y * y).sum::<f32>().sqrt();
                        r.iter().map(move |y| y / n).collect::<Vec<_>>()
                    })
                    .collect()
            };
            let q = tok(unit(normal(11, t * h * d)), d);
            let k = tok(unit(normal(12, t * h * d)), d);
            let v = tok(normal(13, t * h * d), d);
            let g: Vec<f32> = normal(14, t * h)
                .into_iter()
                .map(|y| -(1.0 + (1.5 * y).exp()).ln())
                .collect();
            let g = Tensor::<2>::from_data(TensorData::new(g, [t, h]), &device);
            let beta: Vec<f32> = normal(15, t * h)
                .into_iter()
                .map(|y| 1.0 / (1.0 + (-y).exp()))
                .collect();
            let beta = Tensor::<2>::from_data(TensorData::new(beta, [t, h]), &device).cast(dtype);
            let state: Vec<f32> = normal(16, seqs * h * d * d)
                .into_iter()
                .map(|y| 0.1 * y)
                .collect();
            let state = t4(&state, [seqs, h, d, d], &device);
            let kernel = kernel_takes(
                &device,
                [dtype, DType::F32, dtype],
                [t, h, d, d],
                Some((seqs, cu_seqlens(&offsets, &device).dtype())),
            );
            if common::runtime_of(&device) == Some("cuda") {
                assert!(kernel, "CUDA turned the step's kernel away");
            }
            println!("packed step {dtype:?}: the step's kernel takes it: {kernel}");
            let (ref_out, ref_state) = gated_delta_step_reference(
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                beta.clone(),
                state.clone(),
                Some(cu_seqlens(&offsets, &device)),
                options,
            );
            let before = recurrent::probe::kernel_launches();
            let (out, new_state) = gated_delta_step(
                q,
                k,
                v,
                g,
                beta,
                state,
                Some(cu_seqlens(&offsets, &device)),
                options,
            );
            let (out, new_state) = (to_vec(out), to_vec(new_state));
            assert_eq!(
                recurrent::probe::kernel_launches() - before,
                usize::from(kernel)
            );
            for (what, a, e, tol) in [
                ("out", &out, to_vec(ref_out), tol[0]),
                ("state", &new_state, to_vec(ref_state), tol[1]),
            ] {
                let (abs, rel) = errors(a, &e);
                println!(
                    "packed step {dtype:?} {options:?} {what}: max abs {abs:.3e} rel {rel:.3e}"
                );
                assert!(
                    rel <= tol,
                    "{what}: relative error {rel:.3e} above {tol:.0e}"
                );
            }
        }
    }
}
