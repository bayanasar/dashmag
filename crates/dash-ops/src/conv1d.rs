//! Causal depthwise 1-D convolution over a sequence, of any small kernel width, with an optional
//! bias and activation; its gradients; and the state a decoder carries from step to step.
//!
//! Per channel `c`, with `K` the kernel width and inputs before the first step taken as zero:
//!
//! ```text
//! y[b, t, c]   = bias[c] + sum_{j < K} weight[c, j] * x[b, t - (K - 1) + j, c]
//! out[b, t, c] = act(y[b, t, c])
//! ```
//!
//! `weight[c, K - 1]` multiplies the current step, `weight[c, 0]` the step `K - 1` before it.
//! Without a bias, `bias[c]` is zero; `act` is the identity or SiLU ([`Activation`]). This is
//! torch's `F.conv1d(F.pad(x, (K - 1, 0)), weight[:, None, :], bias, groups=C)` on `x` in
//! `[B, C, T]`, then the activation: what the causal-conv1d package's `causal_conv1d_fn`
//! computes, and transformers' fallback for it.
//!
//! # Layouts and dtypes
//!
//! - `x`, the output and their gradients: `[B, T, C]`. Inputs may have any strides (a kernel
//!   copies one whose channels are not adjacent); outputs are contiguous.
//! - `weight`: `[C, K]`, torch's `[C, 1, K]` without its middle axis.
//! - `bias`: `[C]`, optional.
//! - the decoding state: `[B, S, C]`, f32, the last `S >= K - 1` inputs, oldest first.
//! - one decoding step's input and output: `[B, C]`.
//!
//! `x`, `weight` and `bias` share one float dtype (f32, bf16 or f16). Arithmetic is in f32
//! whatever it is, and the output and each gradient are rounded once, to the dtype of the input
//! they belong to; the weight and bias gradients are reduced over `B x T` in f32 too.
//!
//! The decoding state is f32, as every state an operation carries from call to call: it holds
//! inputs, which f32 holds exactly in any of the three dtypes, so a prefill followed by single
//! steps reads the same values as the whole sequence, and models can keep every state of every
//! layer in one dtype. It costs `S x C x 4` bytes per sequence (72 KiB at `S = 3`,
//! `C = 6144`). transformers' cache keeps the state as `[B, C, S]` in the model's dtype:
//! `state.transpose(1, 2).float()` converts it.
//!
//! # Decoding
//!
//! [`causal_conv1d_state`] gives the state a prompt leaves; [`causal_conv1d_update`] takes one
//! token and the state to the token's output and the next state. A prompt's outputs from
//! [`causal_conv1d`], then each step's, are the outputs of [`causal_conv1d`] over the whole
//! sequence.
//!
//! # Backends
//!
//! The trait's default bodies ([`CausalConv1dBackend`]) are the reference, built from Burn's
//! tensor operations in f32 for any width; every device runs them. On CUDA, kernel widths 1 to
//! [`MAX_KERNEL_WIDTH`] run CubeCL kernels (others run the reference):
//!
//! - the forward pass: one kernel; each thread walks a run of steps of one or two channels,
//!   keeping the last `K - 1` inputs in registers;
//! - the backward pass: one kernel for the input gradient and partial sums of the weight and bias
//!   gradients over runs of 64 steps, and one adding those partial sums up. The activation's
//!   derivative is fused, its pre-activation recomputed from `x`, `weight` and `bias` rather than
//!   kept, so training holds only the inputs for this operation;
//! - a decoding step: one kernel.
//!
//! On `Autodiff` the forward records one node, whose backward runs
//! [`CausalConv1dBackend::causal_conv1d_backward`].

use burn::backend::autodiff::NodeId;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, NodeGuard, Ops, OpsKind};
use burn::backend::tensor::FloatTensor;
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Dispatch, ExtensionType, backend_extension,
};
use burn::tensor::{DType, Tensor};
use std::sync::atomic::Ordering;

#[cfg(feature = "flex")]
use burn::backend::Flex;

#[cfg(feature = "cuda")]
mod cube;
mod reference;

/// The widest kernel the CUDA kernels take; wider ones run the reference. Each channel a thread
/// carries holds `4 K` f32 values in registers in the backward pass (the taps, the last `K`
/// inputs and output gradients, and the weight gradient's sums), so the width is bounded by the
/// register file rather than by the algorithm.
pub const MAX_KERNEL_WIDTH: usize = 8;

/// The activation applied to the convolution's output (after the bias).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Activation {
    /// None: the convolution's output as it is.
    #[default]
    None,
    /// SiLU, also called swish: `y * sigmoid(y)`.
    Silu,
}

/// The arguments of the causal convolution that have a usual default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CausalConv1dOptions {
    /// The activation after the convolution and the bias. Default: none.
    pub activation: Activation,
}

impl CausalConv1dOptions {
    /// The options with `activation`.
    pub fn with_activation(self, activation: Activation) -> Self {
        Self { activation }
    }
}

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

/// Counters a test reads to tell which code ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static FORWARD_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static UPDATE_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);

    /// Launches of this module's CUDA kernels, all of them (zero without `--features cuda`).
    pub fn kernel_launches() -> usize {
        forward_launches() + backward_launches() + update_launches()
    }

    /// Runs of the backward pass written out for `Autodiff`.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }

    /// Launches of the forward kernel.
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
}

/// The causal depthwise convolution as a backend extension. The default bodies are the
/// reference; the [module](self) gives the definition, layouts and dtypes.
#[backend_extension(Flex: cfg(feature = "flex"), Cube: cfg(feature = "cuda"), Fusion: cfg(feature = "fusion"), Autodiff)]
pub trait CausalConv1dBackend: Backend {
    /// The convolution over a sequence: `x` `[B, T, C]`, `weight` `[C, K]` and the bias to the
    /// output `[B, T, C]`.
    #[fusion(dtype = x, shape = x)]
    fn causal_conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        #[extension_type] bias: Bias<Self>,
        options: CausalConv1dOptions,
    ) -> FloatTensor<Self> {
        reference::forward::<Self>(x, weight, bias, options)
    }

    /// The gradients of [`causal_conv1d`](Self::causal_conv1d) with respect to `x`, `weight`
    /// and the bias, given the gradient of its output, `grad` `[B, T, C]`. The bias gradient
    /// is [`Bias::None`] when there is no bias.
    #[fusion(meta = |x, weight, bias, _grad, _options| (x.clone(), weight.clone(), bias.clone()))]
    fn causal_conv1d_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        #[extension_type] bias: Bias<Self>,
        grad: FloatTensor<Self>,
        options: CausalConv1dOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, Bias<Self>) {
        reference::backward::<Self>(x, weight, bias, grad, options)
    }

    /// One decoding step: the new token `x` `[B, C]`, the f32 `state` `[B, S, C]` (`S >= K - 1`),
    /// `weight` `[C, K]` and the bias to the token's output `[B, C]` and the next state, f32
    /// `[B, S, C]`: the state without its oldest step, followed by `x`.
    #[fusion(meta = |x, state, _weight, _bias, _options| (x.clone(), state.clone()))]
    fn causal_conv1d_update(
        x: FloatTensor<Self>,
        state: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        #[extension_type] bias: Bias<Self>,
        options: CausalConv1dOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        reference::update::<Self>(x, state, weight, bias, options)
    }
}

#[cfg(feature = "flex")]
impl CausalConv1dBackend for Flex {}

/// The node `Autodiff` records for the convolution, with `x`, `weight` and, when there is one,
/// the bias as parents (`N` = 2 or 3).
#[derive(Debug)]
struct CausalConv1dNode;

/// What the backward pass keeps: the inputs, through the checkpointer, and the options. The
/// pre-activation is recomputed from the inputs rather than kept.
#[derive(Clone, Debug)]
struct NodeState {
    x: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
    options: CausalConv1dOptions,
}

impl<B: Backend + CausalConv1dBackend, const N: usize> Backward<B, N> for CausalConv1dNode {
    type State = NodeState;

    fn backward(
        self,
        ops: Ops<Self::State, N>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        probe::BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
        let grad = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let mut retrieve = |id| checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(id);
        let x = retrieve(state.x);
        let weight = retrieve(state.weight);
        let bias = Bias::from(state.bias.map(&mut retrieve));
        // One pass yields every gradient; those of the weight and the bias are small partial sums
        // per channel and run, not worth a variant without them.
        let (grad_x, grad_weight, grad_bias) =
            B::causal_conv1d_backward(x, weight, bias, grad, state.options);
        let mut parents = ops.parents.into_iter();
        if let Some(node) = parents.next().flatten() {
            grads.register::<B>(node.id, grad_x);
        }
        if let Some(node) = parents.next().flatten() {
            grads.register::<B>(node.id, grad_weight);
        }
        if let (Some(node), Bias::Some(grad_bias)) = (parents.next().flatten(), grad_bias) {
            grads.register::<B>(node.id, grad_bias);
        }
    }
}

/// The forward on `Autodiff`: one node over `nodes` (those of `x`, `weight` and the bias, if
/// any), keeping the inputs when tracked.
fn record<B: Backend + CausalConv1dBackend, C: CheckpointStrategy, const N: usize>(
    nodes: [NodeGuard; N],
    x: FloatTensor<Autodiff<B, C>>,
    weight: FloatTensor<Autodiff<B, C>>,
    bias: Option<FloatTensor<Autodiff<B, C>>>,
    options: CausalConv1dOptions,
) -> FloatTensor<Autodiff<B, C>> {
    match CausalConv1dNode
        .prepare::<C>(nodes)
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            let state = NodeState {
                x: prep.checkpoint(&x),
                weight: prep.checkpoint(&weight),
                bias: bias.as_ref().map(|bias| prep.checkpoint(bias)),
                options,
            };
            let bias = Bias::from(bias.map(|bias| bias.into_primitive()));
            let out = B::causal_conv1d(x.into_primitive(), weight.into_primitive(), bias, options);
            prep.finish(state, out)
        }
        OpsKind::UnTracked(prep) => {
            let bias = Bias::from(bias.map(|bias| bias.into_primitive()));
            prep.finish(B::causal_conv1d(
                x.into_primitive(),
                weight.into_primitive(),
                bias,
                options,
            ))
        }
    }
}

impl<B: Backend + CausalConv1dBackend, C: CheckpointStrategy> CausalConv1dBackend
    for Autodiff<B, C>
{
    fn causal_conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Bias<Self>,
        options: CausalConv1dOptions,
    ) -> FloatTensor<Self> {
        match bias {
            Bias::None => record::<B, C, 2>([x.node(), weight.node()], x, weight, None, options),
            Bias::Some(bias) => record::<B, C, 3>(
                [x.node(), weight.node(), bias.node()],
                x,
                weight,
                Some(bias),
                options,
            ),
        }
    }

    // `causal_conv1d_backward` keeps the default body: under `Autodiff` it is differentiable
    // through Burn's own operations, which is all a second derivative needs.

    fn causal_conv1d_update(
        x: FloatTensor<Self>,
        state: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Bias<Self>,
        options: CausalConv1dOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Decoding needs no gradient, so the inner backend's results are taken as constants.
        // Should a gradient be asked for, the reference runs instead, which Burn differentiates.
        let bias_tracked = matches!(&bias, Bias::Some(bias) if bias.is_tracked());
        if x.is_tracked() || state.is_tracked() || weight.is_tracked() || bias_tracked {
            return reference::update::<Self>(x, state, weight, bias, options);
        }
        let bias = Bias::from(bias.into_option().map(|bias| bias.into_primitive()));
        let (out, next) = B::causal_conv1d_update(
            x.into_primitive(),
            state.into_primitive(),
            weight.into_primitive(),
            bias,
            options,
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

/// The causal depthwise convolution of `x` `[B, T, C]` with `weight` `[C, K]`, plus `bias`
/// `[C]` if given, followed by `options.activation`: `[B, T, C]` in the dtype of `x`.
/// Differentiable with respect to `x`, `weight` and `bias`. The [module](self) gives the exact
/// definition.
///
/// # Panics
///
/// If the shapes disagree (`C` of `x`, `weight` and `bias`), `K` is 0, or `x`, `weight` and
/// `bias` do not share one dtype.
pub fn causal_conv1d(
    x: Tensor<3>,
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
    options: CausalConv1dOptions,
) -> Tensor<3> {
    let [_, _, c] = x.dims();
    check_parameters("causal_conv1d", c, x.dtype(), &weight, bias.as_ref());
    Tensor::from_dispatch(<Dispatch as CausalConv1dBackend>::causal_conv1d(
        x.into_dispatch(),
        weight.into_dispatch(),
        into_bias(bias),
        options,
    ))
}

/// One decoding step: the new token `x` `[B, C]`, the f32 `state` `[B, S, C]` (the last
/// `S >= K - 1` inputs, oldest first), `weight` `[C, K]` and `bias` `[C]` if given, to the
/// token's output `[B, C]` in the dtype of `x` and the next state, f32 `[B, S, C]`. The output
/// is the last step of [`causal_conv1d`] over the inputs the state holds followed by `x`.
///
/// # Panics
///
/// If the shapes disagree, the state is shorter than `K - 1` steps or not f32, or `x`, `weight`
/// and `bias` do not share one dtype.
pub fn causal_conv1d_update(
    x: Tensor<2>,
    state: Tensor<3>,
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
    options: CausalConv1dOptions,
) -> (Tensor<2>, Tensor<3>) {
    let [b, c] = x.dims();
    let [sb, s, sc] = state.dims();
    let k = check_parameters("causal_conv1d_update", c, x.dtype(), &weight, bias.as_ref());
    assert!(
        sb == b && sc == c,
        "causal_conv1d_update: x [{b}, {c}] and state [{sb}, {s}, {sc}] disagree on the batch or \
         the channels"
    );
    assert!(
        s + 1 >= k,
        "causal_conv1d_update: a state of {s} steps cannot feed a kernel of width {k}; it needs \
         at least {}",
        k - 1
    );
    assert_eq!(
        state.dtype(),
        DType::F32,
        "causal_conv1d_update: the state is f32"
    );
    let (out, next) = <Dispatch as CausalConv1dBackend>::causal_conv1d_update(
        x.into_dispatch(),
        state.into_dispatch(),
        weight.into_dispatch(),
        into_bias(bias),
        options,
    );
    (Tensor::from_dispatch(out), Tensor::from_dispatch(next))
}

/// The decoding state a prompt `x` `[B, T, C]` leaves: its last `state_len` inputs, oldest
/// first, as a new f32 tensor `[B, state_len, C]`, with zeros in front when the prompt is
/// shorter. [`causal_conv1d_update`] continues from it; a kernel of width `K` needs
/// `state_len >= K - 1`.
pub fn causal_conv1d_state(x: Tensor<3>, state_len: usize) -> Tensor<3> {
    let [b, t, c] = x.dims();
    let state = Tensor::<3>::zeros([b, state_len, c], (&x.device(), DType::F32));
    let n = t.min(state_len);
    if n == 0 {
        return state;
    }
    // Assigned into a new tensor rather than sliced, so the state does not hold on to the
    // prompt's memory.
    let tail = x.slice([0..b, t - n..t, 0..c]).cast(DType::F32);
    state.slice_assign([0..b, state_len - n..state_len, 0..c], tail)
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
