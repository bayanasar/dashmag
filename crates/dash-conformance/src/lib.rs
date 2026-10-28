//! The operation-level tests of Dashmag's operations, written once for every implementation of
//! [`dash_contract`] and run through a thin adapter.
//!
//! An implementation's tests implement [`Adapter`] for a device and call the test bodies here.
//! Each body reads the values recorded from what the operation matches (PyTorch, transformers,
//! fla) under `crates/dash-ops/tests/fixtures`, runs its cases through the adapter, compares what
//! comes back with the recording, prints every comparison and asserts it under the bounds
//! measured when the cases were recorded. dash-ops runs them on Flex and on every CubeCL device;
//! what is about dash-ops alone (which of its kernels ran, its kernels against its reference on
//! the device, routing by what the device has, one autodiff graph through calls that feed each
//! other, Burn's views as inputs) stays in its own tests.
//!
//! - [`conv1d`]: causal depthwise conv1d against torch: every kernel width with and without a
//!   bias and SiLU, the decoding step, and a prefill followed by decoding steps. It calls
//!   [`Adapter::causal_conv1d`], with gradients and alone, and [`Adapter::causal_conv1d_update`].
//! - [`conv1d_state`]: conv1d from given states and over packed sequences against torch, final
//!   states and every gradient; and against itself: a sequence split in two, packed sequences
//!   against one call each, empty sequences, either output alone in the loss. It calls
//!   [`Adapter::causal_conv1d`], with gradients and alone.
//! - [`gated_delta_chunk`]: the chunked gated delta rule against transformers, in f32 and from
//!   bf16 inputs, with initial states and the final state's gradient: calls of
//!   [`Adapter::chunk_gated_delta_rule`] with gradients.
//! - [`gated_delta_packed`]: the chunked rule and the step over packed sequences against fla and
//!   transformers, a query scale of their own, 256-wide heads, packed against per-sequence calls:
//!   [`Adapter::chunk_gated_delta_rule`], with gradients and alone, and
//!   [`Adapter::gated_delta_step`], alone and, in `check_step_gradients`, with gradients.
//! - [`gated_delta_recurrent`]: the step against transformers, from bf16 inputs, its gradients,
//!   and decoding from the state the chunked rule leaves after a prompt: calls of
//!   [`Adapter::gated_delta_step`], alone and, in `check_gradients`, with gradients.
//!
//! # The adapter
//!
//! An [`Adapter`] is one device of an implementation. It has one method per operation, which
//! takes one call, runs it on the device and returns what it gave:
//!
//! - **Inputs.** Every tensor comes as a [`HostTensor`]: its shape, its values as f32 in
//!   row-major order, and the dtype ([`DType`]) the implementation is to give it, f32, bf16 or
//!   f16. The values are the dtype's own (a recording rounds an input before it stores it as
//!   f32), so that converting them loses nothing. The operation's floating inputs come in the dtype
//!   the case runs in, those the contract allows in f32 beside it (the gated delta rule's `g`
//!   and `beta`) in f32 where the recording has them so, and states in f32. Packed sequences'
//!   boundaries come as `i64` offsets, which the implementation takes in an int dtype of its own;
//!   the options are the contract's types.
//! - **Gradients.** A call with `grad` set asks for the gradients of the inputs, which the
//!   adapter computes through its implementation's autodiff. `grad` holds the cotangents, f32
//!   values the shape of the outputs they belong to: the gradients asked for are those of the
//!   loss `sum(out * d_out) + sum(final_state * d_final_state)`, a term left out where its
//!   cotangent is absent, and an output narrower than f32 takes its cotangent rounded to its own
//!   dtype. A call without `grad` runs the forward pass alone.
//! - **Outputs.** Every output and gradient comes back as a [`HostTensor`] of its values read
//!   to the host as f32, in the shape the contract gives it, with the dtype the implementation
//!   returned it in: the tests hold the dtypes to the contract's (an output in its input's dtype,
//!   a state in f32, each gradient in its input's dtype), and the values to the recordings.
//! - **What it does not take.** [`Adapter::has`] says whether the device stores and computes in
//!   a dtype; the tests skip the cases in a dtype it does not have (some silently, as dash-ops'
//!   tests always have, some printing `skipped: <device> has no <dtype>`). A call with an option
//!   or an input the implementation does not have returns [`Unsupported`] with the reason,
//!   rather than a result computed some other way: an implementation whose operations return a
//!   `Result` turns its refusal into it, one that panics checks before it calls. Any other failure
//!   is the test's: the adapter panics with it, and the test fails. The test prints
//!   `skipped: <device> does not run <case>: <reason>` and goes on with its next case; a body
//!   that needs every call of a case skips the whole case.
//!
//! A test that reads counters or global state of the implementation runs the bodies one at a
//! time; the bodies themselves keep no state between calls.
//!
//! # What ran
//!
//! A body goes on past a call the adapter does not run, and passes without it; so every body
//! returns an [`Outcome`]: how many calls the adapter ran, each call it did not run with its
//! reason, and what the body measured for its caller to bound. The caller asserts on it: an
//! implementation that runs every case, as dash-ops does, asserts [`Outcome::all_ran`]; one that
//! does not yet checks that what it skipped is what it means to skip. A body whose calls were all
//! skipped has compared nothing, and its outcome says so.
//!
//! # Running them
//!
//! Each body takes the adapter, the label its lines start with, and, where the bounds have two
//! columns, which applies: `other` holds a device to the wider bounds measured on CubeCL's other
//! runtimes than CUDA (lavapipe and OpenGL through wgpu, CubeCL's CPU runtime: their own
//! exponential, their own orders of summation), the narrower ones having been measured on Flex
//! and CUDA. An implementation measured elsewhere starts from the narrower ones. A body panics on
//! a comparison above its bound, once it has printed it.
//!
//! ```ignore
//! struct Mine(MyDevice);
//! impl dash_conformance::Adapter for Mine { /* the operations, on self.0 */ }
//!
//! #[test]
//! fn conv1d_matches_torch() {
//!     dash_conformance::conv1d::sequences(&Mine(MyDevice::default()), "mine", false).all_ran();
//! }
//! ```

use dash_contract::conv1d::{CausalConv1dOptions, CausalConv1dStepOptions};
use dash_contract::gated_delta::chunk::ChunkGatedDeltaOptions;
use dash_contract::gated_delta::recurrent::GatedDeltaStepOptions;
use std::cell::{Cell, RefCell};

pub use dash_contract::DType;

pub mod conv1d;
pub mod conv1d_state;
mod fixtures;
pub mod gated_delta_chunk;
pub mod gated_delta_packed;
pub mod gated_delta_recurrent;

/// A tensor on the host: its shape, its values as f32 in row-major order, and its dtype, the
/// dtype the implementation is to give it (an input) or gave it (an output). The values are
/// exactly the dtype's.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    /// The dtype the tensor is in.
    pub dtype: DType,
    /// The shape.
    pub shape: Vec<usize>,
    /// The values, row-major.
    pub values: Vec<f32>,
}

impl HostTensor {
    /// `values` of `shape`, in `dtype`.
    ///
    /// # Panics
    ///
    /// If `values` does not hold the shape's number of elements.
    pub fn new(dtype: DType, shape: Vec<usize>, values: Vec<f32>) -> Self {
        let elements: usize = shape.iter().product();
        assert_eq!(
            values.len(),
            elements,
            "a tensor {shape:?} holds {elements} values, not {}",
            values.len()
        );
        Self {
            dtype,
            shape,
            values,
        }
    }

    /// `values` of `shape`, in f32.
    pub fn f32(shape: Vec<usize>, values: Vec<f32>) -> Self {
        Self::new(DType::F32, shape, values)
    }

    /// The same values, to be given in `dtype`.
    pub fn with_dtype(self, dtype: DType) -> Self {
        Self { dtype, ..self }
    }
}

/// Why an adapter does not run a call: the option or input its implementation does not have.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported(pub String);

/// One device of an implementation of [`dash_contract`]'s operations, as the tests here drive
/// it. The [crate documentation](crate) says what each method receives and returns.
pub trait Adapter {
    /// The device, as the tests print it beside a skipped case: dash-ops gives its runtime, its
    /// adapter and the limits its kernels ask about.
    fn describe(&self) -> String;

    /// Whether the device stores and computes in `dtype`; the tests skip the cases in a dtype it
    /// does not have.
    fn has(&self, dtype: DType) -> bool;

    /// `causal_conv1d` ([`dash_contract::conv1d`]), with its gradients when [`Conv1d::grad`]
    /// asks for them.
    fn causal_conv1d(&self, call: &Conv1d) -> Result<Conv1dOut, Unsupported>;

    /// `causal_conv1d_update`, one decoding step, without gradients.
    fn causal_conv1d_update(&self, call: &Conv1dStep) -> Result<Conv1dStepOut, Unsupported>;

    /// `chunk_gated_delta_rule` ([`dash_contract::gated_delta::chunk`]), with its gradients when
    /// [`ChunkGatedDelta::grad`] asks for them.
    fn chunk_gated_delta_rule(
        &self,
        call: &ChunkGatedDelta,
    ) -> Result<ChunkGatedDeltaOut, Unsupported>;

    /// `gated_delta_step` ([`dash_contract::gated_delta::recurrent`]), with the gradients of the
    /// step's definition when [`GatedDeltaStep::grad`] asks for them; an implementation that
    /// gives a step no gradient returns [`Unsupported`] for those calls.
    fn gated_delta_step(&self, call: &GatedDeltaStep) -> Result<GatedDeltaStepOut, Unsupported>;
}

/// One call of `causal_conv1d`.
#[derive(Clone, Debug)]
pub struct Conv1d {
    /// `[B, T, C]`, in the case's dtype.
    pub x: HostTensor,
    /// `[C, K]`, in `x`'s dtype.
    pub weight: HostTensor,
    /// `[C]`, in `x`'s dtype, or none.
    pub bias: Option<HostTensor>,
    /// `[S, W, C]`, f32, or none (zeros).
    pub initial_state: Option<HostTensor>,
    /// The boundaries `[S + 1]` of packed sequences (`B = 1`), or none: a sequence per batch row.
    pub cu_seqlens: Option<Vec<i64>>,
    /// The activation and the final states asked for.
    pub options: CausalConv1dOptions,
    /// The cotangents, if the call asks for gradients.
    pub grad: Option<Conv1dCotangents>,
}

/// The cotangents of a [`Conv1d`] call: of the output, `[B, T, C]`, and of the final states,
/// `[S, W', C]`, each left out of the loss where it is `None`.
#[derive(Clone, Debug)]
pub struct Conv1dCotangents {
    /// The output's.
    pub out: Option<HostTensor>,
    /// The final states'.
    pub final_state: Option<HostTensor>,
}

/// What a [`Conv1d`] call gave.
#[derive(Clone, Debug)]
pub struct Conv1dOut {
    /// The output, `[B, T, C]`.
    pub out: HostTensor,
    /// The final states `[S, W', C]`, if asked for.
    pub final_state: Option<HostTensor>,
    /// The gradients, if asked for.
    pub grads: Option<Conv1dGrads>,
}

/// The gradients of a [`Conv1d`] call's inputs.
#[derive(Clone, Debug)]
pub struct Conv1dGrads {
    /// `x`'s.
    pub x: HostTensor,
    /// The weight's.
    pub weight: HostTensor,
    /// The bias's, if there is a bias.
    pub bias: Option<HostTensor>,
    /// The initial states', if there are initial states.
    pub initial_state: Option<HostTensor>,
}

/// One call of `causal_conv1d_update`.
#[derive(Clone, Debug)]
pub struct Conv1dStep {
    /// The new token, `[B, C]`, in the case's dtype.
    pub x: HostTensor,
    /// `[C, K]`, in `x`'s dtype.
    pub weight: HostTensor,
    /// `[C]`, in `x`'s dtype, or none.
    pub bias: Option<HostTensor>,
    /// `[B, W, C]`, f32.
    pub state: HostTensor,
    /// The activation.
    pub options: CausalConv1dStepOptions,
}

/// What a [`Conv1dStep`] call gave.
#[derive(Clone, Debug)]
pub struct Conv1dStepOut {
    /// The token's output, `[B, C]`.
    pub out: HostTensor,
    /// The next state, `[B, W, C]`.
    pub state: HostTensor,
}

/// One call of `chunk_gated_delta_rule`.
#[derive(Clone, Debug)]
pub struct ChunkGatedDelta {
    /// The queries, `[B, T, H, Dk]`, in the case's dtype.
    pub q: HostTensor,
    /// The keys, `[B, T, H, Dk]`.
    pub k: HostTensor,
    /// The values, `[B, T, H, Dv]`.
    pub v: HostTensor,
    /// The gate, `[B, T, H]`, in the case's dtype or f32.
    pub g: HostTensor,
    /// `beta`, `[B, T, H]`, in the case's dtype or f32.
    pub beta: HostTensor,
    /// `[S, H, Dk, Dv]`, f32, or none (zeros).
    pub initial_state: Option<HostTensor>,
    /// The boundaries `[S + 1]` of packed sequences (`B = 1`), or none.
    pub cu_seqlens: Option<Vec<i64>>,
    /// The normalisation, the scale and the final states asked for.
    pub options: ChunkGatedDeltaOptions,
    /// The cotangents, if the call asks for gradients.
    pub grad: Option<ChunkGatedDeltaCotangents>,
}

/// The cotangents of a [`ChunkGatedDelta`] call: of the output, `[B, T, H, Dv]`, and of the
/// final states, `[S, H, Dk, Dv]`, left out of the loss where it is `None`.
#[derive(Clone, Debug)]
pub struct ChunkGatedDeltaCotangents {
    /// The output's.
    pub out: HostTensor,
    /// The final states'.
    pub final_state: Option<HostTensor>,
}

/// What a [`ChunkGatedDelta`] call gave.
#[derive(Clone, Debug)]
pub struct ChunkGatedDeltaOut {
    /// The output, `[B, T, H, Dv]`.
    pub out: HostTensor,
    /// The final states, `[S, H, Dk, Dv]`, if asked for.
    pub final_state: Option<HostTensor>,
    /// The gradients, if asked for.
    pub grads: Option<ChunkGatedDeltaGrads>,
}

/// The gradients of a [`ChunkGatedDelta`] call's inputs.
#[derive(Clone, Debug)]
pub struct ChunkGatedDeltaGrads {
    /// The queries'.
    pub q: HostTensor,
    /// The keys'.
    pub k: HostTensor,
    /// The values'.
    pub v: HostTensor,
    /// The gate's.
    pub g: HostTensor,
    /// `beta`'s.
    pub beta: HostTensor,
    /// The initial states', if there are initial states.
    pub initial_state: Option<HostTensor>,
}

/// One call of `gated_delta_step`.
#[derive(Clone, Debug)]
pub struct GatedDeltaStep {
    /// The queries, `[B, H, Dk]` (packed, `[T, H, Dk]`), in the case's dtype.
    pub q: HostTensor,
    /// The keys, `[B, H, Dk]`.
    pub k: HostTensor,
    /// The values, `[B, H, Dv]`.
    pub v: HostTensor,
    /// The gate, `[B, H]`, in the case's dtype or f32.
    pub g: HostTensor,
    /// `beta`, `[B, H]`, in the case's dtype or f32.
    pub beta: HostTensor,
    /// `[B, H, Dk, Dv]` (packed, `[S, H, Dk, Dv]`), f32.
    pub state: HostTensor,
    /// The boundaries `[S + 1]` of packed tokens, or none: a token per sequence.
    pub cu_seqlens: Option<Vec<i64>>,
    /// The normalisation and the scale.
    pub options: GatedDeltaStepOptions,
    /// The cotangents, if the call asks for gradients.
    pub grad: Option<GatedDeltaStepCotangents>,
}

/// The cotangents of a [`GatedDeltaStep`] call: of the output and of the new states.
#[derive(Clone, Debug)]
pub struct GatedDeltaStepCotangents {
    /// The output's.
    pub out: HostTensor,
    /// The new states'.
    pub state: HostTensor,
}

/// What a [`GatedDeltaStep`] call gave.
#[derive(Clone, Debug)]
pub struct GatedDeltaStepOut {
    /// The output, `[B, H, Dv]` (packed, `[T, H, Dv]`).
    pub out: HostTensor,
    /// The new states, shaped as the states given.
    pub state: HostTensor,
    /// The gradients, if asked for.
    pub grads: Option<GatedDeltaStepGrads>,
}

/// The gradients of a [`GatedDeltaStep`] call's inputs.
#[derive(Clone, Debug)]
pub struct GatedDeltaStepGrads {
    /// The queries'.
    pub q: HostTensor,
    /// The keys'.
    pub k: HostTensor,
    /// The values'.
    pub v: HostTensor,
    /// The gate's.
    pub g: HostTensor,
    /// `beta`'s.
    pub beta: HostTensor,
    /// The states'.
    pub state: HostTensor,
}

/// What a test body ran: how many calls the adapter ran, each call it did not run, and what the
/// body measured for its caller to bound. A body goes on past a call the adapter does not run;
/// this is how its caller knows what was left out.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "a body passes over the calls the adapter does not run: assert what ran"]
pub struct Outcome<T = ()> {
    /// What the body measured for its caller to bound, where it leaves a bound to it (the largest
    /// differences of a comparison); over what was compared, nothing where nothing was.
    pub value: T,
    /// How many calls the adapter ran.
    pub ran: usize,
    /// The calls it did not run, each `<operation>: <reason>`, in order. A case that needed one
    /// of them was not compared.
    pub skipped: Vec<String>,
}

impl<T> Outcome<T> {
    /// The value, once it is asserted that the adapter ran every call, and some: for an
    /// implementation that runs every case.
    ///
    /// # Panics
    ///
    /// If a call was skipped, or none ran.
    #[track_caller]
    pub fn all_ran(self) -> T {
        assert!(
            self.skipped.is_empty(),
            "{} calls skipped, {} ran: {:?}",
            self.skipped.len(),
            self.ran,
            self.skipped
        );
        assert!(self.ran > 0, "no call ran");
        self.value
    }
}

/// What a body's adapters have run so far: each body counts its calls through one.
#[derive(Default)]
struct Tally {
    ran: Cell<usize>,
    skipped: RefCell<Vec<String>>,
}

impl Tally {
    /// `adapter`, its calls counted here.
    fn count<'a>(&'a self, adapter: &'a dyn Adapter) -> Counted<'a> {
        Counted {
            adapter,
            tally: self,
        }
    }

    /// What was counted, with what the body measured.
    fn outcome<T>(&self, value: T) -> Outcome<T> {
        Outcome {
            value,
            ran: self.ran.get(),
            skipped: self.skipped.take(),
        }
    }

    /// Counts the result of a call of `operation`.
    fn note<T>(&self, operation: &str, result: Result<T, Unsupported>) -> Result<T, Unsupported> {
        match &result {
            Ok(_) => self.ran.set(self.ran.get() + 1),
            Err(Unsupported(reason)) => self
                .skipped
                .borrow_mut()
                .push(format!("{operation}: {reason}")),
        }
        result
    }
}

/// An adapter whose calls a [`Tally`] counts.
struct Counted<'a> {
    adapter: &'a dyn Adapter,
    tally: &'a Tally,
}

impl Adapter for Counted<'_> {
    fn describe(&self) -> String {
        self.adapter.describe()
    }

    fn has(&self, dtype: DType) -> bool {
        self.adapter.has(dtype)
    }

    fn causal_conv1d(&self, call: &Conv1d) -> Result<Conv1dOut, Unsupported> {
        let result = self.adapter.causal_conv1d(call);
        self.tally.note("causal_conv1d", result)
    }

    fn causal_conv1d_update(&self, call: &Conv1dStep) -> Result<Conv1dStepOut, Unsupported> {
        let result = self.adapter.causal_conv1d_update(call);
        self.tally.note("causal_conv1d_update", result)
    }

    fn chunk_gated_delta_rule(
        &self,
        call: &ChunkGatedDelta,
    ) -> Result<ChunkGatedDeltaOut, Unsupported> {
        let result = self.adapter.chunk_gated_delta_rule(call);
        self.tally.note("chunk_gated_delta_rule", result)
    }

    fn gated_delta_step(&self, call: &GatedDeltaStep) -> Result<GatedDeltaStepOut, Unsupported> {
        let result = self.adapter.gated_delta_step(call);
        self.tally.note("gated_delta_step", result)
    }
}

/// What a call gave, or `None` after printing why the adapter did not run it: the test goes on
/// with its next case.
pub fn ran<T>(adapter: &dyn Adapter, case: &str, result: Result<T, Unsupported>) -> Option<T> {
    match result {
        Ok(out) => Some(out),
        Err(Unsupported(reason)) => {
            println!(
                "skipped: {} does not run {case}: {reason}",
                adapter.describe()
            );
            None
        }
    }
}

/// The error measure of conv1d's tests, in f64: the largest absolute difference and that
/// difference over the largest magnitude of the expected values.
///
/// # Panics
///
/// If the lengths differ or an actual value is not finite.
pub fn errors(actual: &[f32], expected: &[f32]) -> (f64, f64) {
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

/// The same measure as the gated delta rule's tests take it, in f32: the largest absolute
/// difference and that over the largest absolute expected value. (The two measures were taken in
/// different precisions when their bounds were measured; each operation keeps its own.)
///
/// # Panics
///
/// If the lengths differ or an actual value is not finite.
pub fn errors_f32(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len(), "lengths differ");
    let mut max_err = 0f32;
    let mut scale = 0f32;
    for (a, e) in actual.iter().zip(expected) {
        assert!(a.is_finite(), "non-finite value {a}");
        max_err = max_err.max((a - e).abs());
        scale = scale.max(e.abs());
    }
    (max_err, max_err / scale.max(f32::MIN_POSITIVE))
}
