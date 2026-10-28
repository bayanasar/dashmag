//! The gated delta rule over packed variable-length sequences, with a query scale of its own, and
//! with heads 256 wide: the chunked rule and the step, held to fla 0.5.2 (packed, with
//! `cu_seqlens`) and to transformers' torch functions in float64 called once per sequence.
//!
//! The fixtures in `gated_delta_packed/` hold the inputs, the gradients fed back (`d_out`,
//! `d_state`) and the results, with the suffix `_f64` (transformers in float64, one call per
//! sequence, stored rounded to f32) and `_fla` (fla over the packed inputs, f32 in IEEE on CUDA;
//! for the step, no gradients; not for the 256-wide cases, the largest files, where fla agrees
//! with float64 to f32 rounding as on the others):
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
//! ("relative", [`crate::errors_f32`]); the relative one is asserted just above the largest
//! measured over every build (Flex; `cuda`; `cuda` with `fusion`, with `autotune`, with both) and
//! on lavapipe and CubeCL's CPU runtime.
//!
//! - [`check_chunk_f32`], [`check_chunk_bf16`]: the chunked rule against fla and transformers;
//! - [`packed_against_sequences`]: packed against one call per sequence;
//! - [`check_steps`]: the packed step against fla and transformers, and against itself token by
//!   token; [`check_step_gradients`]: the step's gradients, where the implementation gives them;
//! - [`the_default_scale_is_one_over_the_root_of_dk`];
//! - [`Fixture`], [`run_chunk`], [`run_step`] and [`check`], for tests that look at each run.

use crate::fixtures::File;
use crate::{
    Adapter, ChunkGatedDelta, ChunkGatedDeltaCotangents, DType, GatedDeltaStep,
    GatedDeltaStepCotangents, HostTensor, Outcome, Tally, errors_f32, ran,
};
use dash_contract::gated_delta::chunk::ChunkGatedDeltaOptions;
use dash_contract::gated_delta::recurrent::GatedDeltaStepOptions;
use std::collections::HashMap;

/// The chunked rule, f32, against transformers in float64 [measured: 1.23e-6].
pub const F32_VS_F64: f32 = 1.5e-6;
/// The chunked rule, f32, against fla's packed f32 path [measured: 2.03e-6]. fla computes its
/// gates in base 2 and in its own order, a few f32 roundings from the exact result itself.
pub const F32_VS_FLA: f32 = 2.5e-6;
/// bf16 inputs against float64 from the same values, where the reference or the f32 kernels run:
/// one bf16 rounding of each output and gradient, at most 2^-9 relative to its largest value
/// [measured: 2.58e-3]; fla's own bf16 path is further off (6.4e-3, printed beside it).
pub const BF16_VS_F64: f32 = 3e-3;
/// The same where the kernels on tensor cores run, with bf16 operands as fla's [measured: 5.58e-3,
/// the queries' gradient; fla's 6.36e-3].
pub const BF16_TC_VS_F64: f32 = 6e-3;
/// The step, f32, its output and new states against transformers in float64 and fla [measured:
/// 2.3e-7].
pub const STEP_VS_REF: f32 = 3e-7;
/// The tracked step's gradients (the reference on the autodiff backend) against float64
/// [measured: 3.1e-7].
pub const STEP_GRAD: f32 = 4e-7;
/// Packed against per-sequence calls of the same operation on the same device: every chunk's
/// arithmetic is the same, so they agree to the bit [measured: 0 on every build].
pub const PACKED_VS_SEQUENCES: f32 = 0.0;

/// One recorded fixture: its tensors as f32 (the boundaries as i64) and its metadata.
pub struct Fixture {
    /// The fixture's name: its file.
    pub name: &'static str,
    meta: HashMap<String, serde_json::Value>,
    file: File,
    /// The packed sequences' boundaries.
    pub cu_seqlens: Vec<i64>,
}

impl Fixture {
    /// Reads the fixture `name`.
    pub fn load(name: &'static str) -> Self {
        let file = File::read(&format!("gated_delta_packed/{name}.safetensors"));
        let meta = file
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::from_str(v).unwrap()))
            .collect();
        let cu_seqlens = file.ints("cu_seqlens");
        Self {
            name,
            meta,
            file,
            cu_seqlens,
        }
    }

    /// A flag of the fixture's metadata.
    pub fn flag(&self, key: &str) -> bool {
        self.meta[key].as_bool().unwrap()
    }

    /// The query scale it was recorded with, if one.
    pub fn scale(&self) -> Option<f32> {
        self.meta["scale"].as_f64().map(|x| x as f32)
    }

    /// A number of its metadata (`heads`, `dk`, `dv`).
    pub fn number(&self, key: &str) -> usize {
        self.meta[key].as_u64().unwrap() as usize
    }

    fn has(&self, key: &str) -> bool {
        self.file.has(key)
    }

    /// A recorded tensor's values.
    pub fn get(&self, key: &str) -> Vec<f32> {
        assert!(self.has(key), "{}: no {key}", self.name);
        self.file.tensor(key).values
    }

    /// A recorded tensor, f32.
    pub fn tensor(&self, key: &str) -> HostTensor {
        self.file.tensor(key)
    }

    /// The elements of one sequence's state, `H Dk Dv`.
    fn state_len(&self) -> usize {
        ["heads", "dk", "dv"]
            .iter()
            .map(|key| self.number(key))
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

/// What one run gave: outputs and gradients as f32 vectors.
pub type Results = HashMap<&'static str, Vec<f32>>;

/// Compares every result with the fixture's `<name><suffix>`, printing the errors, and asserts
/// the relative error is at most `tol`.
pub fn check(case: &Fixture, label: &str, res: &Results, suffix: &str, tol: f32) {
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
        let (abs, rel) = errors_f32(&res[key], &case.get(&format!("{key}{suffix}")));
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

/// The chunked rule over the fixture's packed sequences, the queries, keys, values and `beta` in
/// `dtype` (`g` stays f32): forward and backward. `None` if the adapter does not run it.
pub fn run_chunk(adapter: &dyn Adapter, case: &Fixture, dtype: DType) -> Option<Results> {
    let input = |key: &str| case.tensor(key).with_dtype(dtype);
    let call = ChunkGatedDelta {
        q: input("q"),
        k: input("k"),
        v: input("v"),
        g: case.tensor("g"),
        beta: input("beta"),
        initial_state: case.flag("initial_state").then(|| case.tensor("h0")),
        cu_seqlens: Some(case.cu_seqlens.clone()),
        options: chunk_options(case),
        grad: Some(ChunkGatedDeltaCotangents {
            out: case.tensor("d_out"),
            final_state: case
                .flag("final_state_grad")
                .then(|| case.tensor("d_state")),
        }),
    };
    let result = ran(adapter, case.name, adapter.chunk_gated_delta_rule(&call))?;
    let state = result.final_state.expect("the final states, asked for");
    assert_eq!(result.out.dtype, dtype);
    assert_eq!(state.dtype, DType::F32);
    let grads = result.grads.expect("the gradients asked for");
    let mut res = Results::new();
    res.insert("out", result.out.values);
    res.insert("state", state.values);
    res.insert("dq", grads.q.values);
    res.insert("dk", grads.k.values);
    res.insert("dv", grads.v.values);
    res.insert("dg", grads.g.values);
    res.insert("dbeta", grads.beta.values);
    if let Some(dh0) = grads.initial_state {
        res.insert("dh0", dh0.values);
    }
    Some(res)
}

/// Tokens `span` of a `[1, T, ...]` tensor, as `[1, span, ...]`.
fn tokens(x: &HostTensor, span: &std::ops::Range<usize>) -> HostTensor {
    let row: usize = x.shape[2..].iter().product();
    let mut shape = x.shape.clone();
    shape[1] = span.len();
    HostTensor::new(
        x.dtype,
        shape,
        x.values[span.start * row..span.end * row].to_vec(),
    )
}

/// Rows `rows` of a tensor, along its first dimension.
fn rows(x: &HostTensor, rows: std::ops::Range<usize>) -> HostTensor {
    let row: usize = x.shape[1..].iter().product();
    let mut shape = x.shape.clone();
    shape[0] = rows.len();
    HostTensor::new(
        x.dtype,
        shape,
        x.values[rows.start * row..rows.end * row].to_vec(),
    )
}

/// The same inputs, each sequence a call of its own (a batch of one, unpacked), the results laid
/// out as the packed call's.
fn run_chunk_per_sequence(adapter: &dyn Adapter, case: &Fixture) -> Option<Results> {
    let options = chunk_options(case);
    let mut parts: HashMap<&'static str, Vec<Vec<f32>>> = HashMap::new();
    let mut push = |key: &'static str, x: Vec<f32>| parts.entry(key).or_default().push(x);
    for (s, span) in case.spans().into_iter().enumerate() {
        let state_of = |key: &str| rows(&case.tensor(key), s..s + 1);
        let h0 = case.flag("initial_state").then(|| state_of("h0"));
        if span.is_empty() {
            // Nothing to run: the state passes through, and so does its gradient.
            let zeros = || vec![0.0; case.state_len()];
            push("state", h0.map_or_else(zeros, |h0| h0.values));
            if case.flag("initial_state") {
                let d = if case.flag("final_state_grad") {
                    state_of("d_state").values
                } else {
                    zeros()
                };
                push("dh0", d);
            }
            continue;
        }
        let slice = |key: &str| tokens(&case.tensor(key), &span);
        let call = ChunkGatedDelta {
            q: slice("q"),
            k: slice("k"),
            v: slice("v"),
            g: slice("g"),
            beta: slice("beta"),
            initial_state: h0,
            cu_seqlens: None,
            options,
            grad: Some(ChunkGatedDeltaCotangents {
                out: slice("d_out"),
                final_state: case.flag("final_state_grad").then(|| state_of("d_state")),
            }),
        };
        let result = ran(adapter, case.name, adapter.chunk_gated_delta_rule(&call))?;
        let grads = result.grads.expect("the gradients asked for");
        push("out", result.out.values);
        push("state", result.final_state.expect("the final state").values);
        push("dq", grads.q.values);
        push("dk", grads.k.values);
        push("dv", grads.v.values);
        push("dg", grads.g.values);
        push("dbeta", grads.beta.values);
        if let Some(dh0) = grads.initial_state {
            push("dh0", dh0.values);
        }
    }
    // Per sequence, token-major tensors are [1, len, H, D]: concatenated in order they are the
    // packed tensors; states [1, H, Dk, Dv] likewise.
    Some(
        parts
            .into_iter()
            .map(|(key, xs)| (key, xs.concat()))
            .collect(),
    )
}

/// Packed against per-sequence calls through the same adapter: the largest relative difference,
/// 0 where a call was skipped and nothing compared.
pub fn packed_against_sequences(
    adapter: &dyn Adapter,
    case: &Fixture,
    label: &str,
) -> Outcome<f32> {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let (Some(packed), Some(apart)) = (
        run_chunk(adapter, case, DType::F32),
        run_chunk_per_sequence(adapter, case),
    ) else {
        return tally.outcome(0.0);
    };
    let mut keys: Vec<_> = packed.keys().copied().collect();
    keys.sort();
    let mut worst = 0f32;
    for key in keys {
        let (abs, rel) = errors_f32(&packed[key], &apart[key]);
        println!(
            "{:16} {label:12} {key:6} packed vs per sequence max abs {abs:.3e} rel {rel:.3e}",
            case.name
        );
        worst = worst.max(rel);
    }
    tally.outcome(worst)
}

/// The chunked fixtures in f32.
pub const CHUNK_F32: [&str; 4] = [
    "chunk_f32",
    "chunk_f32_scale",
    "chunk_wide_k",
    "chunk_wide_v",
];

/// The f32 chunked fixtures against transformers in float64, and fla's where it was recorded.
pub fn check_chunk_f32(adapter: &dyn Adapter, label: &str) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    for name in CHUNK_F32 {
        let case = Fixture::load(name);
        let Some(res) = run_chunk(adapter, &case, DType::F32) else {
            continue;
        };
        check(&case, label, &res, "_f64", F32_VS_F64);
        if case.flag("fla") {
            check(&case, label, &res, "_fla", F32_VS_FLA);
        }
    }
    tally.outcome(())
}

/// The bf16 chunked fixture against float64, within `tol`, and fla's own bf16 path against it.
pub fn check_chunk_bf16(adapter: &dyn Adapter, label: &str, tol: f32) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let case = Fixture::load("chunk_bf16");
    let Some(res) = run_chunk(adapter, &case, DType::BF16) else {
        return tally.outcome(());
    };
    check(&case, label, &res, "_f64", tol);
    // For the record: fla's own bf16 path against the same float64 reference.
    let mut keys: Vec<_> = res.keys().copied().collect();
    keys.sort();
    for key in keys {
        let (abs, rel) = errors_f32(
            &case.get(&format!("{key}_fla")),
            &case.get(&format!("{key}_f64")),
        );
        println!(
            "{:16} {:12} {key:6} vs f64      max abs {abs:.3e} rel {rel:.3e}",
            case.name, "fla-bf16"
        );
    }
    tally.outcome(())
}

/// The options a step fixture was recorded with.
fn step_options(case: &Fixture) -> GatedDeltaStepOptions {
    GatedDeltaStepOptions {
        qk_l2norm: case.flag("l2norm"),
        scale: case.scale(),
    }
}

/// The step over the fixture's packed tokens, without gradients: output and states. `None` if
/// the adapter does not run it.
pub fn run_step(adapter: &dyn Adapter, case: &Fixture) -> Option<Results> {
    let call = GatedDeltaStep {
        q: case.tensor("q"),
        k: case.tensor("k"),
        v: case.tensor("v"),
        g: case.tensor("g"),
        beta: case.tensor("beta"),
        state: case.tensor("h0"),
        cu_seqlens: Some(case.cu_seqlens.clone()),
        options: step_options(case),
        grad: None,
    };
    let result = ran(adapter, case.name, adapter.gated_delta_step(&call))?;
    assert_eq!(result.state.dtype, DType::F32);
    Some(Results::from([
        ("out", result.out.values),
        ("state", result.state.values),
    ]))
}

/// The step's gradients through the implementation's autodiff, where it gives them.
fn run_step_gradients(adapter: &dyn Adapter, case: &Fixture) -> Option<Results> {
    let call = GatedDeltaStep {
        q: case.tensor("q"),
        k: case.tensor("k"),
        v: case.tensor("v"),
        g: case.tensor("g"),
        beta: case.tensor("beta"),
        state: case.tensor("h0"),
        cu_seqlens: Some(case.cu_seqlens.clone()),
        options: step_options(case),
        grad: Some(GatedDeltaStepCotangents {
            out: case.tensor("d_out"),
            state: case.tensor("d_state"),
        }),
    };
    let result = ran(adapter, case.name, adapter.gated_delta_step(&call))?;
    let grads = result.grads.expect("the gradients asked for");
    Some(Results::from([
        ("dq", grads.q.values),
        ("dk", grads.k.values),
        ("dv", grads.v.values),
        ("dg", grads.g.values),
        ("dbeta", grads.beta.values),
        ("dh0", grads.state.values),
    ]))
}

/// The step token by token, each sequence apart: what the packed step must give.
fn run_step_per_token(adapter: &dyn Adapter, case: &Fixture) -> Option<Results> {
    let (mut outs, mut states) = (Vec::new(), Vec::new());
    for (s, span) in case.spans().into_iter().enumerate() {
        let mut state = rows(&case.tensor("h0"), s..s + 1);
        for t in span {
            let row = |key: &str| rows(&case.tensor(key), t..t + 1);
            let call = GatedDeltaStep {
                q: row("q"),
                k: row("k"),
                v: row("v"),
                g: row("g"),
                beta: row("beta"),
                state,
                cu_seqlens: None,
                options: step_options(case),
                grad: None,
            };
            let result = ran(adapter, case.name, adapter.gated_delta_step(&call))?;
            outs.push(result.out.values);
            state = result.state;
        }
        states.push(state.values);
    }
    Some(Results::from([
        ("out", outs.concat()),
        ("state", states.concat()),
    ]))
}

/// The step fixtures.
pub const STEPS: [&str; 2] = ["step_f32", "step_f32_scale"];

/// The packed step against transformers in float64 and fla, and against itself token by token
/// through the same adapter (to the bit).
pub fn check_steps(adapter: &dyn Adapter, label: &str) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    for name in STEPS {
        let case = Fixture::load(name);
        let (Some(res), Some(apart)) =
            (run_step(adapter, &case), run_step_per_token(adapter, &case))
        else {
            continue;
        };
        check(&case, label, &res, "_f64", STEP_VS_REF);
        check(&case, label, &res, "_fla", STEP_VS_REF);
        for key in ["out", "state"] {
            let (abs, rel) = errors_f32(&res[key], &apart[key]);
            println!(
                "{name:16} {label:12} {key:6} packed vs token by token max abs {abs:.3e} rel {rel:.3e}"
            );
            assert!(rel <= PACKED_VS_SEQUENCES, "{name} {key}: {rel:.3e}");
        }
    }
    tally.outcome(())
}

/// The packed step's gradients against transformers in float64.
pub fn check_step_gradients(adapter: &dyn Adapter, label: &str) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    for name in STEPS {
        let case = Fixture::load(name);
        let Some(res) = run_step_gradients(adapter, &case) else {
            continue;
        };
        check(&case, label, &res, "_f64", STEP_GRAD);
    }
    tally.outcome(())
}

/// The default scale is `1 / sqrt(Dk)`: giving it explicitly changes nothing, to the bit.
pub fn the_default_scale_is_one_over_the_root_of_dk(adapter: &dyn Adapter) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let case = Fixture::load("chunk_f32");
    let dk = case.number("dk") as f32;
    let run = |options: ChunkGatedDeltaOptions| {
        let call = ChunkGatedDelta {
            q: case.tensor("q"),
            k: case.tensor("k"),
            v: case.tensor("v"),
            g: case.tensor("g"),
            beta: case.tensor("beta"),
            initial_state: Some(case.tensor("h0")),
            cu_seqlens: Some(case.cu_seqlens.clone()),
            options,
            grad: None,
        };
        let result = ran(adapter, case.name, adapter.chunk_gated_delta_rule(&call))?;
        Some((result.out.values, result.final_state.unwrap().values))
    };
    let default = ChunkGatedDeltaOptions::default()
        .with_qk_l2norm(true)
        .with_final_state(true);
    if let (Some(by_default), Some(given)) =
        (run(default), run(default.with_scale(1.0 / dk.sqrt())))
    {
        assert_eq!(by_default, given);
    }
    tally.outcome(())
}
