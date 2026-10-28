//! The chunked gated delta rule against transformers' `torch_chunk_gated_delta_rule`.
//!
//! The fixtures in `gated_delta_chunk/` hold inputs, the gradients fed back (`d_out`,
//! `d_state`), and what torch computed from them: the output, the final state and the gradient
//! of every input, at f32 (`out`, `dq`, ...), and with the function's f32 casts turned into f64
//! (`out_f64`, ...; stored rounded to f32). The bf16 case holds bf16 inputs (as f32 values),
//! torch's own bf16 results (`out_torch_bf16`, ...) and the f64 results from the same inputs.
//! Errors are the largest absolute difference and that over the largest absolute reference value
//! ([`crate::errors_f32`]).
//!
//! - [`check_f32_cases`]: the f32 cases against torch at f32 and at f64;
//! - [`check_bf16_case`]: the bf16 case against f64, beside torch's own bf16 path;
//! - [`Case`], [`run`] and [`check`]: one case, its run and its comparisons, for tests that look
//!   at each run themselves.

use crate::fixtures::File;
use crate::{
    Adapter, ChunkGatedDelta, ChunkGatedDeltaCotangents, DType, Outcome, Tally, errors_f32, ran,
};
use dash_contract::gated_delta::chunk::ChunkGatedDeltaOptions;
use std::collections::HashMap;

/// One recorded case.
pub struct Case {
    /// The case's name: its file in the fixtures.
    pub name: &'static str,
    file: File,
    meta: HashMap<String, serde_json::Value>,
}

impl Case {
    /// Reads the case `name`.
    pub fn load(name: &'static str) -> Self {
        let file = File::read(&format!("gated_delta_chunk/{name}.safetensors"));
        let meta = file
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::from_str(v).unwrap()))
            .collect();
        for name in file.names() {
            assert_eq!(file.tensor(name).dtype, DType::F32, "{name}");
        }
        Self { name, file, meta }
    }

    /// A flag of the case's metadata.
    pub fn flag(&self, key: &str) -> bool {
        self.meta[key].as_bool().unwrap()
    }

    /// A recorded tensor's values.
    pub fn get(&self, key: &str) -> Vec<f32> {
        self.file.tensor(key).values
    }

    /// A recorded tensor's shape.
    pub fn shape(&self, key: &str) -> Vec<usize> {
        self.file.tensor(key).shape
    }

    /// A recorded tensor, f32.
    pub fn tensor(&self, key: &str) -> crate::HostTensor {
        self.file.tensor(key)
    }
}

/// What one run of the operation gave: output, final state, and gradients, as f32 vectors.
pub type Results = HashMap<&'static str, Vec<f32>>;

/// Runs the case's forward and backward with inputs in `dtype` (the states f32), the final
/// state asked for where the case has its gradient; `None` if the adapter does not run it.
pub fn run(adapter: &dyn Adapter, case: &Case, dtype: DType) -> Option<Results> {
    let input = |key: &str| case.tensor(key).with_dtype(dtype);
    let state_grad = case.flag("final_state_grad");
    let options = ChunkGatedDeltaOptions {
        qk_l2norm: case.flag("l2norm"),
        output_final_state: state_grad,
        ..Default::default()
    };
    let call = ChunkGatedDelta {
        q: input("q"),
        k: input("k"),
        v: input("v"),
        g: input("g"),
        beta: input("beta"),
        initial_state: case.flag("initial_state").then(|| case.tensor("h0")),
        cu_seqlens: None,
        options,
        grad: Some(ChunkGatedDeltaCotangents {
            out: case.tensor("d_out"),
            final_state: state_grad.then(|| case.tensor("d_state")),
        }),
    };
    let result = ran(adapter, case.name, adapter.chunk_gated_delta_rule(&call))?;
    if let Some(state) = &result.final_state {
        assert_eq!(state.dtype, DType::F32);
    }
    assert_eq!(result.out.dtype, dtype);
    let grads = result.grads.expect("the gradients asked for");
    let mut res = Results::new();
    res.insert("out", result.out.values);
    if let Some(state) = result.final_state {
        res.insert("state", state.values);
    }
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

/// Compares every result with the reference `<name><suffix>`, printing the errors, and asserts
/// the relative error is at most `tol`.
pub fn check(case: &Case, label: &str, res: &Results, suffix: &str, tol: f32) {
    let mut keys: Vec<_> = res.keys().copied().collect();
    keys.sort();
    let mut failed = Vec::new();
    for key in keys {
        let (abs, rel) = errors_f32(&res[key], &case.get(&format!("{key}{suffix}")));
        println!(
            "{:20} {label:12} {key:6} vs {:13} max abs {abs:.3e} rel {rel:.3e}",
            case.name,
            format!("torch{suffix}")
        );
        if rel > tol {
            failed.push(format!("{key} {rel:.3e}"));
        }
    }
    assert!(
        failed.is_empty(),
        "{} {label}: relative errors above {tol:.1e}: {failed:?}",
        case.name
    );
}

/// The cases in f32.
pub const F32_CASES: [&str; 3] = ["f32_l2norm_state", "f32_plain", "f32_short"];

/// fp32 against torch at fp32 (and at f64): the requirement is 1e-5 relative; the measured errors
/// are at most 1.2e-6, on Flex, on CUDA, and through the reference on lavapipe, on OpenGL over
/// D3D12 and on CubeCL's CPU runtime (printed with --nocapture).
pub const F32_TOL: f32 = 2e-6;

/// bf16 inputs, against the f64 reference from the same inputs, where the reference or the f32
/// kernels run: they compute in f32 and round once, to the outputs' dtype, so the error is about
/// one bf16 rounding of the largest values (2^-9 relative): measured at most 2.5e-3. torch's own
/// bf16 path, which normalises queries and keys in bf16, is further off (up to 8.1e-3, printed
/// beside it).
pub const BF16_TOL: f32 = 3e-3;

/// The same where the kernels on tensor cores run, whose products take bf16 operands (the state,
/// `W`, `U`, `T`, `V'` and the backward's operands rounded to bf16, as fla's are): measured at
/// most 6.12e-3 (the output; the queries' gradient 5.34e-3), on CUDA in every build.
pub const BF16_TC_TOL: f32 = 6.5e-3;

/// Every f32 case against torch at f32 and at f64.
pub fn check_f32_cases(adapter: &dyn Adapter, label: &str) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    for name in F32_CASES {
        let case = Case::load(name);
        let Some(res) = run(adapter, &case, DType::F32) else {
            continue;
        };
        check(&case, label, &res, "", F32_TOL);
        check(&case, label, &res, "_f64", F32_TOL);
    }
    tally.outcome(())
}

/// The bf16 case against f64, within `tol`, and torch's own bf16 path against the same f64.
pub fn check_bf16_case(adapter: &dyn Adapter, label: &str, tol: f32) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let case = Case::load("bf16_l2norm_state");
    let Some(res) = run(adapter, &case, DType::BF16) else {
        return tally.outcome(());
    };
    check(&case, label, &res, "_f64", tol);
    // For the record: torch's own bf16 path against the same f64 reference.
    let mut keys: Vec<_> = res.keys().copied().collect();
    keys.sort();
    for key in keys {
        let (abs, rel) = errors_f32(
            &case.get(&format!("{key}_torch_bf16")),
            &case.get(&format!("{key}_f64")),
        );
        println!(
            "{:20} {:12} {key:6} vs torch_f64     max abs {abs:.3e} rel {rel:.3e}",
            case.name, "torch-bf16"
        );
    }
    tally.outcome(())
}
