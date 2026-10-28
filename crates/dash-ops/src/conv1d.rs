//! Causal depthwise 1-D convolution over sequences, of any small kernel width, with an optional
//! bias and activation; from a given state or from zeros, over a batch or packed variable-length
//! sequences; with its gradients, and the step a decoder takes.
//!
//! [`dash_contract::conv1d`] defines the operation, which every implementation of it shares: what
//! [`causal_conv1d`] and [`causal_conv1d_update`] compute and what they match (torch's
//! `F.conv1d`, the causal-conv1d package's `causal_conv1d_fn`, transformers' fallback), their
//! sequences, layouts, dtypes and states, and their options, whose types are re-exported here.
//! This module implements it on Burn.
//!
//! # Backends
//!
//! The trait's default bodies ([`CausalConv1dBackend`]) are the reference, built from Burn's
//! tensor operations in f32 for any width, packed sequences one at a time (their boundaries read
//! to the host); every device runs them. On a CubeCL device (any runtime), kernel widths 1 to
//! [`MAX_KERNEL_WIDTH`] run CubeCL kernels (others run the reference), which read `cu_seqlens`
//! on the device:
//!
//! - the forward pass: one kernel; each thread walks a run of steps of one or two channels,
//!   keeping the last `K - 1` inputs in registers, and where a sequence starts (anywhere, inside
//!   a thread's run too) fills them from that sequence's initial state; with final states, one
//!   more kernel gathers them;
//! - the backward pass: one kernel for the input gradient, the initial states' gradients and
//!   partial sums of the weight and bias gradients over runs of 64 steps, and one adding those
//!   partial sums up; with initial states, one more for the rows of their gradients that only
//!   the final states reach. The activation's derivative is fused, its pre-activation recomputed
//!   from `x`, `weight`, `bias` and the initial states rather than kept, so training holds only
//!   the inputs for this operation; the final states' gradient is added to the input gradient in
//!   f32, before its rounding;
//! - a decoding step: one kernel.
//!
//! The kernels ask of the device cubes of 256 units, and no shared memory or plane operations,
//! and with 64-bit `cu_seqlens`, 64-bit integers: CUDA and wgpu (lavapipe, and OpenGL with 32-bit
//! `cu_seqlens`) have that; CubeCL's CPU runtime, with as many units per cube as cores, runs the
//! reference. A kernel copies an input whose channels are not adjacent.
//!
//! On `Autodiff` the forward records one node, whose backward runs
//! [`CausalConv1dBackend::causal_conv1d_backward`]; with final states, two more that hand the
//! output's and the final states' gradients to it, so that one backward pass takes both.

use burn::backend::autodiff::NodeId;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, NodeGuard, Ops, OpsKind};
use burn::backend::tensor::FloatTensor;
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Dispatch, ExtensionType, TensorMetadata, backend_extension,
};
use burn::tensor::{DType, Device, FloatDType, Int, Shape, Tensor};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

#[cfg(feature = "flex")]
use burn::backend::Flex;

/// The packed sequences of a call, as the backend extension takes them: the gated delta rule's.
pub use crate::sequences::Sequences;
pub use dash_contract::conv1d::{Activation, CausalConv1dOptions, CausalConv1dStepOptions};

#[cfg(feature = "cubecl")]
mod cube;
mod reference;

/// The widest kernel the CubeCL kernels take; wider ones run the reference. Each channel a thread
/// carries holds `4 K` f32 values in registers in the backward pass (the taps, the last `K`
/// inputs and output gradients, and the weight gradient's sums), so the width is bounded by the
/// register file rather than by the algorithm.
pub const MAX_KERNEL_WIDTH: usize = 8;

/// A bias `[C]` or none, as the backend extension takes it (an `Option` of a tensor does not
/// cross Burn's dispatch boundary; this enum does). Also the bias gradient's type: it has the
/// variant of the bias it belongs to.
#[derive(ExtensionType, Clone, Debug)]
#[extension_type(fusion: cfg(feature = "fusion"))]
pub enum Bias<B: Backend> {
    /// No bias.
    None,
    /// One value per channel, `[C]`, in the dtype of `x`.
    Some(FloatTensor<B>),
}

impl<B: Backend> Bias<B> {
    /// The bias as an `Option`.
    pub fn into_option(self) -> Option<FloatTensor<B>> {
        match self {
            Bias::None => None,
            Bias::Some(bias) => Some(bias),
        }
    }
}

impl<B: Backend> From<Option<FloatTensor<B>>> for Bias<B> {
    fn from(bias: Option<FloatTensor<B>>) -> Self {
        match bias {
            None => Bias::None,
            Some(bias) => Bias::Some(bias),
        }
    }
}

/// Per-sequence states `[S, W, C]` (f32) or none, as the backend extension takes and returns
/// them: the initial states, the final states, and the gradients of both.
#[derive(ExtensionType, Clone, Debug)]
#[extension_type(fusion: cfg(feature = "fusion"))]
pub enum State<B: Backend> {
    /// No states: an initial state of zeros, no final state, or no gradient.
    None,
    /// One state per sequence, `[S, W, C]`, f32.
    Some(FloatTensor<B>),
}

impl<B: Backend> State<B> {
    /// The states as an `Option`.
    pub fn into_option(self) -> Option<FloatTensor<B>> {
        match self {
            State::None => None,
            State::Some(state) => Some(state),
        }
    }
}

impl<B: Backend> From<Option<FloatTensor<B>>> for State<B> {
    fn from(state: Option<FloatTensor<B>>) -> Self {
        match state {
            None => State::None,
            Some(state) => State::Some(state),
        }
    }
}

/// Counters a test reads to tell which code ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static FORWARD_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static UPDATE_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static STATE_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static SEQUENCE_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);
    pub(super) static HOST_READS: AtomicUsize = AtomicUsize::new(0);

    /// Launches of this module's CubeCL kernels, all of them (none where the reference runs).
    pub fn kernel_launches() -> usize {
        forward_launches() + backward_launches() + update_launches() + state_launches()
    }

    /// Runs of the backward pass written out for `Autodiff`.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }

    /// Launches of the forward kernels, one per forward pass (a final states' kernel counts among
    /// [`state_launches`]).
    pub fn forward_launches() -> usize {
        FORWARD_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of the backward kernels: two per backward pass, the input gradient with the
    /// weight and bias gradients' partial sums, then the sum of those.
    pub fn backward_launches() -> usize {
        BACKWARD_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of the decoding step's kernel.
    pub fn update_launches() -> usize {
        UPDATE_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of the kernels that gather the final states, and that write the rows of the
    /// initial states' gradients only the final states reach.
    pub fn state_launches() -> usize {
        STATE_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches, among the forward and backward ones, of the kernels that take states or packed
    /// sequences (the others take neither).
    pub fn sequence_launches() -> usize {
        SEQUENCE_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Reads of `cu_seqlens` to the host: the reference's, which splits packed sequences there.
    /// The kernels read them on the device.
    pub fn host_reads() -> usize {
        HOST_READS.load(Ordering::SeqCst)
    }
}

/// Whether the CubeCL kernels, rather than the reference, take [`causal_conv1d`], its backward
/// pass and [`causal_conv1d_update`] on `device` for `x` `[B, T, C]` (a step's `[B, C]` and its
/// state fit within it), a kernel of width `width` and inputs in `dtype`, with every state and,
/// `packed`, that many sequences whose `cu_seqlens` are of that dtype: widths 1 to
/// [`MAX_KERNEL_WIDTH`], on a device with what the kernels ask (module documentation). `false` on
/// any device not CubeCL's. For tests, as the [`probe`] counters are.
#[doc(hidden)]
pub fn kernels_take(
    device: &Device,
    dtype: DType,
    width: usize,
    shape: [usize; 3],
    packed: Option<(usize, DType)>,
) -> bool {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return cube::kernels_take(&client, dtype, width, shape, packed);
    }
    let _ = (device, dtype, width, shape, packed);
    false
}

#[cfg(feature = "fusion")]
mod fusion_meta {
    use super::{Activation, BiasMetadata, StateMetadata};
    use crate::sequences::SequencesMetadata;
    use burn::backend::fusion::custom::TensorSpec;
    use burn::tensor::{DType, Shape};

    pub(super) fn forward(
        x: &TensorSpec,
        _weight: &TensorSpec,
        _bias: &BiasMetadata,
        _initial_state: &StateMetadata,
        sequences: &SequencesMetadata,
        _activation: &Activation,
        final_state_len: &Option<usize>,
    ) -> (TensorSpec, StateMetadata) {
        let [b, _, c] = x.shape.dims::<3>();
        let final_state = match final_state_len {
            None => StateMetadata::None,
            Some(len) => {
                let s = match sequences {
                    SequencesMetadata::Batch => b,
                    SequencesMetadata::Packed(cu_seqlens) => cu_seqlens.shape[0] - 1,
                };
                StateMetadata::Some(TensorSpec::new(Shape::new([s, *len, c]), DType::F32))
            }
        };
        (x.clone(), final_state)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn backward(
        x: &TensorSpec,
        weight: &TensorSpec,
        bias: &BiasMetadata,
        initial_state: &StateMetadata,
        _sequences: &SequencesMetadata,
        _grad: &TensorSpec,
        _grad_final_state: &StateMetadata,
        _activation: &Activation,
    ) -> (TensorSpec, TensorSpec, BiasMetadata, StateMetadata) {
        (
            x.clone(),
            weight.clone(),
            bias.clone(),
            initial_state.clone(),
        )
    }
}

/// The causal depthwise convolution as a backend extension. Call it through [`causal_conv1d`]
/// and [`causal_conv1d_update`]; the default bodies are the reference, and
/// [`dash_contract::conv1d`] gives the definition, layouts and dtypes.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(feature = "cubecl"),
    Fusion: cfg(feature = "fusion"),
    Autodiff
)]
pub trait CausalConv1dBackend: Backend {
    /// The convolution over sequences: `x` `[B, T, C]`, `weight` `[C, K]`, the bias and the
    /// initial states `[S, W, C]` (f32, `W >= K - 1`; zeros where there are none) to the output
    /// `[B, T, C]` and, with `final_state_len` `Some(W')`, the final states `[S, W', C]` (f32).
    /// `S` is `B`, or with packed `sequences` their number.
    #[fusion(meta = fusion_meta::forward)]
    fn causal_conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        #[extension_type] bias: Bias<Self>,
        #[extension_type] initial_state: State<Self>,
        #[extension_type] sequences: Sequences<Self>,
        activation: Activation,
        final_state_len: Option<usize>,
    ) -> (FloatTensor<Self>, State<Self>) {
        reference::forward::<Self>(
            x,
            weight,
            bias,
            initial_state,
            sequences,
            activation,
            final_state_len,
        )
    }

    /// The gradients of [`causal_conv1d`](Self::causal_conv1d) with respect to `x`, `weight`, the
    /// bias and the initial states, given the gradient of its output, `grad` `[B, T, C]`, and of
    /// its final states (none where they have none). The bias and initial-state gradients have
    /// the variants of the bias and the initial states.
    #[fusion(meta = fusion_meta::backward)]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn causal_conv1d_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        #[extension_type] bias: Bias<Self>,
        #[extension_type] initial_state: State<Self>,
        #[extension_type] sequences: Sequences<Self>,
        grad: FloatTensor<Self>,
        #[extension_type] grad_final_state: State<Self>,
        activation: Activation,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        Bias<Self>,
        State<Self>,
    ) {
        reference::backward::<Self>(
            x,
            weight,
            bias,
            initial_state,
            sequences,
            grad,
            grad_final_state,
            activation,
        )
    }

    /// One decoding step: the new token `x` `[B, C]`, `weight` `[C, K]`, the bias and the f32
    /// `state` `[B, W, C]` (`W >= K - 1`) to the token's output `[B, C]` and the next state, f32
    /// `[B, W, C]`: the state without its oldest step, followed by `x`.
    #[fusion(meta = |x, _weight, _bias, state, _activation| (x.clone(), state.clone()))]
    fn causal_conv1d_update(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        #[extension_type] bias: Bias<Self>,
        state: FloatTensor<Self>,
        activation: Activation,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        reference::update::<Self>(x, weight, bias, state, activation)
    }
}

#[cfg(feature = "flex")]
impl CausalConv1dBackend for Flex {}

/// The node `Autodiff` records for the convolution, with `x`, `weight` and, when there are, the
/// bias and the initial states as parents (`N` = 2 to 4).
#[derive(Debug)]
struct CausalConv1dNode;

/// The gradients the nodes of the output and of the final states hand to the convolution's node,
/// when there are final states: an `Autodiff` node has one output, so the convolution's node
/// outputs a token, and two nodes on it output the output and the final states.
#[derive(Debug)]
struct Handover<B: Backend> {
    output: Option<FloatTensor<B>>,
    final_state: Option<FloatTensor<B>>,
}

type Shared<B> = Arc<Mutex<Handover<B>>>;

/// What the backward pass keeps: the inputs, through the checkpointer, the packed sequences' and
/// the activation; with final states, where the outputs' nodes leave their gradients, and the
/// output's shape and dtype for a gradient of zeros when none is left. The pre-activation is
/// recomputed from the inputs rather than kept.
#[derive(Clone, Debug)]
struct NodeState<B: Backend> {
    x: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
    initial_state: Option<NodeId>,
    sequences: Sequences<B>,
    activation: Activation,
    handover: Option<(Shared<B>, Shape, DType)>,
}

impl<B: Backend + CausalConv1dBackend, const N: usize> Backward<B, N> for CausalConv1dNode {
    type State = NodeState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, N>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        probe::BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
        let node_grad = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let mut retrieve = |id| checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(id);
        let x = retrieve(state.x);
        let weight = retrieve(state.weight);
        let bias = Bias::from(state.bias.map(&mut retrieve));
        let initial_state = State::from(state.initial_state.map(&mut retrieve));
        let (grad, grad_final_state) = match state.handover {
            None => (node_grad, State::None),
            Some((shared, shape, dtype)) => {
                let handed = std::mem::replace(
                    &mut *shared.lock().unwrap_or_else(|e| e.into_inner()),
                    Handover {
                        output: None,
                        final_state: None,
                    },
                );
                let grad = handed
                    .output
                    .unwrap_or_else(|| B::float_zeros(shape, &x.device(), dtype.into()));
                (grad, State::from(handed.final_state))
            }
        };
        // One pass yields every gradient; those of the weight and the bias are small partial sums
        // per channel and run, not worth a variant without them.
        let (grad_x, grad_weight, grad_bias, grad_initial_state) = B::causal_conv1d_backward(
            x,
            weight,
            bias,
            initial_state,
            state.sequences,
            grad,
            grad_final_state,
            state.activation,
        );
        let mut parents = ops.parents.into_iter();
        if let Some(node) = parents.next().flatten() {
            grads.register::<B>(node.id, grad_x);
        }
        if let Some(node) = parents.next().flatten() {
            grads.register::<B>(node.id, grad_weight);
        }
        // The bias, when there is one, is the third parent.
        if state.bias.is_some()
            && let (Some(node), Bias::Some(grad_bias)) = (parents.next().flatten(), grad_bias)
        {
            grads.register::<B>(node.id, grad_bias);
        }
        if let (Some(node), State::Some(grad)) = (parents.next().flatten(), grad_initial_state) {
            grads.register::<B>(node.id, grad);
        }
    }
}

/// Which of the convolution's two outputs a [`HandoverNode`] outputs.
#[derive(Clone, Copy, Debug)]
enum Slot {
    Output,
    FinalState,
}

/// A node on the convolution's token that outputs the output or the final states, and whose
/// backward leaves their gradient for the convolution's node.
#[derive(Debug)]
struct HandoverNode;

impl<B: Backend> Backward<B, 1> for HandoverNode {
    type State = (Shared<B>, Slot);

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let grad = grads.consume::<B>(&ops.node);
        let device = grad.device();
        let (shared, slot) = ops.state;
        {
            let mut handover = shared.lock().unwrap_or_else(|e| e.into_inner());
            let place = match slot {
                Slot::Output => &mut handover.output,
                Slot::FinalState => &mut handover.final_state,
            };
            *place = Some(grad);
        }
        // The token's gradient carries nothing; it only makes the convolution's node run.
        if let Some(node) = ops.parents.into_iter().next().flatten() {
            let token = B::float_zeros(Shape::new([1]), &device, FloatDType::F32);
            grads.register::<B>(node.id, token);
        }
    }
}

/// The inputs of one call on `Autodiff`.
struct Inputs<B: Backend, C: CheckpointStrategy> {
    x: FloatTensor<Autodiff<B, C>>,
    weight: FloatTensor<Autodiff<B, C>>,
    bias: Option<FloatTensor<Autodiff<B, C>>>,
    initial_state: Option<FloatTensor<Autodiff<B, C>>>,
}

/// The output and the final states of one call on `Autodiff`.
type Outputs<B, C> = (FloatTensor<Autodiff<B, C>>, State<Autodiff<B, C>>);

/// The forward on `Autodiff`: one node over `nodes` (those of `x`, `weight` and the bias and
/// initial states, if any), keeping the inputs when tracked; with final states, two nodes on it
/// for the two outputs.
fn record<B: Backend + CausalConv1dBackend, C: CheckpointStrategy, const N: usize>(
    nodes: [NodeGuard; N],
    inputs: Inputs<B, C>,
    sequences: Sequences<B>,
    activation: Activation,
    final_state_len: Option<usize>,
) -> Outputs<B, C> {
    let Inputs {
        x,
        weight,
        bias,
        initial_state,
    } = inputs;
    match CausalConv1dNode
        .prepare::<C>(nodes)
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            let mut state = NodeState {
                x: prep.checkpoint(&x),
                weight: prep.checkpoint(&weight),
                bias: bias.as_ref().map(|bias| prep.checkpoint(bias)),
                initial_state: initial_state.as_ref().map(|state| prep.checkpoint(state)),
                sequences: sequences.clone(),
                activation,
                handover: None,
            };
            let (out, final_state) = B::causal_conv1d(
                x.into_primitive(),
                weight.into_primitive(),
                Bias::from(bias.map(|bias| bias.into_primitive())),
                State::from(initial_state.map(|state| state.into_primitive())),
                sequences,
                activation,
                final_state_len,
            );
            let State::Some(final_state) = final_state else {
                return (prep.finish(state, out), State::None);
            };
            let shared = Arc::new(Mutex::new(Handover {
                output: None,
                final_state: None,
            }));
            state.handover = Some((shared.clone(), out.shape(), out.dtype()));
            let token = B::float_zeros(Shape::new([1]), &out.device(), FloatDType::F32);
            let token = prep.finish(state, token);
            let out = hand_over::<B, C>(&token, shared.clone(), Slot::Output, out);
            let final_state = hand_over::<B, C>(&token, shared, Slot::FinalState, final_state);
            (out, State::Some(final_state))
        }
        OpsKind::UnTracked(prep) => {
            let (out, final_state) = B::causal_conv1d(
                x.into_primitive(),
                weight.into_primitive(),
                Bias::from(bias.map(|bias| bias.into_primitive())),
                State::from(initial_state.map(|state| state.into_primitive())),
                sequences,
                activation,
                final_state_len,
            );
            let final_state = final_state
                .into_option()
                .map(<Autodiff<B, C> as AutodiffBackend>::from_inner);
            (prep.finish(out), State::from(final_state))
        }
    }
}

/// A node on the convolution's `token` that outputs `value`, one of the convolution's outputs.
fn hand_over<B: Backend, C: CheckpointStrategy>(
    token: &FloatTensor<Autodiff<B, C>>,
    shared: Shared<B>,
    slot: Slot,
    value: FloatTensor<B>,
) -> FloatTensor<Autodiff<B, C>> {
    match HandoverNode
        .prepare::<C>([token.node()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((shared, slot), value),
        OpsKind::UnTracked(prep) => prep.finish(value),
    }
}

/// The sequences of an `Autodiff` call on its inner backend: int tensors are not tracked.
fn inner_sequences<B: Backend, C: CheckpointStrategy>(
    sequences: Sequences<Autodiff<B, C>>,
) -> Sequences<B> {
    match sequences {
        Sequences::Batch => Sequences::Batch,
        Sequences::Packed(cu_seqlens) => Sequences::Packed(cu_seqlens),
    }
}

impl<B: Backend + CausalConv1dBackend, C: CheckpointStrategy> CausalConv1dBackend
    for Autodiff<B, C>
{
    fn causal_conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Bias<Self>,
        initial_state: State<Self>,
        sequences: Sequences<Self>,
        activation: Activation,
        final_state_len: Option<usize>,
    ) -> (FloatTensor<Self>, State<Self>) {
        let sequences = inner_sequences(sequences);
        let (bias, initial_state) = (bias.into_option(), initial_state.into_option());
        let mut nodes = vec![x.node(), weight.node()];
        nodes.extend(bias.as_ref().map(|bias| bias.node()));
        nodes.extend(initial_state.as_ref().map(|state| state.node()));
        let inputs = Inputs {
            x,
            weight,
            bias,
            initial_state,
        };
        let (activation, len) = (activation, final_state_len);
        let mut nodes = nodes.into_iter();
        let mut next = || nodes.next().expect("a node per input");
        match (inputs.bias.is_some(), inputs.initial_state.is_some()) {
            (false, false) => {
                record::<B, C, 2>([next(), next()], inputs, sequences, activation, len)
            }
            (true, false) | (false, true) => {
                record::<B, C, 3>([next(), next(), next()], inputs, sequences, activation, len)
            }
            (true, true) => record::<B, C, 4>(
                [next(), next(), next(), next()],
                inputs,
                sequences,
                activation,
                len,
            ),
        }
    }

    // `causal_conv1d_backward` keeps the default body: under `Autodiff` it is differentiable
    // through Burn's own operations, which is all a second derivative needs.

    fn causal_conv1d_update(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Bias<Self>,
        state: FloatTensor<Self>,
        activation: Activation,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Decoding needs no gradient, so the inner backend's results are taken as constants.
        // Should a gradient be asked for, the reference runs instead, which Burn differentiates.
        let bias_tracked = matches!(&bias, Bias::Some(bias) if bias.is_tracked());
        if x.is_tracked() || state.is_tracked() || weight.is_tracked() || bias_tracked {
            return reference::update::<Self>(x, weight, bias, state, activation);
        }
        let bias = Bias::from(bias.into_option().map(|bias| bias.into_primitive()));
        let (out, next) = B::causal_conv1d_update(
            x.into_primitive(),
            weight.into_primitive(),
            bias,
            state.into_primitive(),
            activation,
        );
        (
            <Self as AutodiffBackend>::from_inner(out),
            <Self as AutodiffBackend>::from_inner(next),
        )
    }
}

fn into_bias(bias: Option<Tensor<1>>) -> Bias<Dispatch> {
    Bias::from(bias.map(Tensor::into_dispatch))
}

/// The causal depthwise convolution of `x` `[B, T, C]` with `weight` `[C, K]`, plus `bias` `[C]`
/// if given, followed by `options.activation`, each sequence continuing from its initial state
/// if given; [`dash_contract::conv1d`] gives the exact definition.
///
/// - `initial_state`: `[S, W, C]`, f32, each sequence's last `W >= K - 1` inputs before its first
///   step, oldest first; zeros when `None`.
/// - `cu_seqlens`: `None` for one sequence per batch element (`S = B`); else `[N + 1]`, the
///   boundaries of `S = N` sequences packed along `T` of a batch of one (`B = 1`), from 0 to `T`
///   without decreasing (a sequence may be empty). On a device running the kernels they are read
///   there, not back to the host, and not checked.
/// - `options`: the activation; whether to return the final states, and their length `W'`.
///
/// Returns the output `[B, T, C]` in the dtype of `x` and, if asked, the final states
/// `[S, W', C]`, f32. Differentiable with respect to `x`, `weight`, `bias` and `initial_state`,
/// through the output and the final states.
///
/// # Panics
///
/// If the shapes disagree (`C` of `x`, `weight`, `bias` and the states; `S`), `K` is 0, `x`,
/// `weight` and `bias` do not share one dtype, the initial states are not f32 or shorter than
/// `K - 1` steps, packed sequences come with `B > 1`, or the activation is one this crate does
/// not implement; where the reference runs, if `cu_seqlens` does not go from 0 to `T` without
/// decreasing.
pub fn causal_conv1d(
    x: Tensor<3>,
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
    initial_state: Option<Tensor<3>>,
    cu_seqlens: Option<Tensor<1, Int>>,
    options: CausalConv1dOptions,
) -> (Tensor<3>, Option<Tensor<3>>) {
    let op = "causal_conv1d";
    let [b, _, c] = x.dims();
    let k = check_parameters(op, c, x.dtype(), &weight, bias.as_ref());
    check_activation(op, options.activation);
    let sequences = match &cu_seqlens {
        None => b,
        Some(cu_seqlens) => {
            assert_eq!(
                b, 1,
                "{op}: packed sequences lie along T of a batch of one, got B = {b}"
            );
            let [n] = cu_seqlens.dims();
            assert!(
                n >= 2,
                "{op}: cu_seqlens must hold N + 1 >= 2 offsets, got {n}"
            );
            n - 1
        }
    };
    if let Some(state) = &initial_state {
        check_state(op, "initial state", state, [sequences, c], k);
    }
    let final_state_len = options.output_final_state.then(|| {
        options
            .final_state_len
            .or(initial_state.as_ref().map(|state| state.dims()[1]))
            .unwrap_or(k - 1)
    });
    let (out, final_state) = <Dispatch as CausalConv1dBackend>::causal_conv1d(
        x.into_dispatch(),
        weight.into_dispatch(),
        into_bias(bias),
        State::from(initial_state.map(Tensor::into_dispatch)),
        Sequences::from(cu_seqlens.map(Tensor::into_dispatch)),
        options.activation,
        final_state_len,
    );
    (
        Tensor::from_dispatch(out),
        final_state.into_option().map(Tensor::from_dispatch),
    )
}

/// One decoding step: the new token `x` `[B, C]`, `weight` `[C, K]`, `bias` `[C]` if given and
/// the f32 `state` `[B, W, C]` (the last `W >= K - 1` inputs, oldest first), to the token's
/// output `[B, C]` in the dtype of `x` and the next state, f32 `[B, W, C]`. The output is the
/// last step of [`causal_conv1d`] over the token from `state`, the next state its final state.
///
/// # Panics
///
/// If the shapes disagree, the state is shorter than `K - 1` steps or not f32, `x`, `weight`
/// and `bias` do not share one dtype, or the activation is one this crate does not implement.
pub fn causal_conv1d_update(
    x: Tensor<2>,
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
    state: Tensor<3>,
    options: CausalConv1dStepOptions,
) -> (Tensor<2>, Tensor<3>) {
    let op = "causal_conv1d_update";
    let [b, c] = x.dims();
    let k = check_parameters(op, c, x.dtype(), &weight, bias.as_ref());
    check_activation(op, options.activation);
    check_state(op, "state", &state, [b, c], k);
    let (out, next) = <Dispatch as CausalConv1dBackend>::causal_conv1d_update(
        x.into_dispatch(),
        weight.into_dispatch(),
        into_bias(bias),
        state.into_dispatch(),
        options.activation,
    );
    (Tensor::from_dispatch(out), Tensor::from_dispatch(next))
}

/// Checks `weight` `[C, K]` and `bias` `[C]` against `C` channels and the dtype of `x`, and
/// returns `K`.
fn check_parameters(
    op: &str,
    c: usize,
    dtype: DType,
    weight: &Tensor<2>,
    bias: Option<&Tensor<1>>,
) -> usize {
    let [wc, k] = weight.dims();
    assert!(k >= 1, "{op}: the kernel width must be at least 1");
    assert_eq!(wc, c, "{op}: x has {c} channels, weight [C, K] has {wc}");
    assert_eq!(
        weight.dtype(),
        dtype,
        "{op}: x and weight must share one dtype"
    );
    if let Some(bias) = bias {
        let [bc] = bias.dims();
        assert_eq!(bc, c, "{op}: x has {c} channels, bias [C] has {bc}");
        assert_eq!(bias.dtype(), dtype, "{op}: x and bias must share one dtype");
    }
    k
}

/// Refuses an activation this crate does not implement: [`Activation`] is the contract's, which
/// may name more than these.
fn check_activation(op: &str, activation: Activation) {
    assert!(
        matches!(activation, Activation::None | Activation::Silu),
        "{op}: activation {activation:?} is not implemented"
    );
}

/// Checks a state `[S, W, C]` against `S` sequences, `C` channels and a kernel of width `k`.
fn check_state(op: &str, what: &str, state: &Tensor<3>, [s, c]: [usize; 2], k: usize) {
    let [ss, w, sc] = state.dims();
    assert!(
        ss == s && sc == c,
        "{op}: the {what} [{ss}, {w}, {sc}] must be [S, W, C] with S = {s} sequences and C = {c} \
         channels"
    );
    assert!(
        w + 1 >= k,
        "{op}: a {what} of {w} steps cannot feed a kernel of width {k}; it needs at least {}",
        k - 1
    );
    assert_eq!(state.dtype(), DType::F32, "{op}: the {what} is f32");
}
