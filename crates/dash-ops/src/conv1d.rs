//! Causal depthwise 1-D convolution, with the state a decoder carries from token to token.
//!
//! Gated DeltaNet runs it over its query, key and value projections, concatenated (Qwen3.5-0.8B:
//! 6144 channels, kernel width 4), and follows it with SiLU. Per channel `c`, with `K` the
//! kernel width and inputs before the first step taken as zero:
//!
//! ```text
//! out[b, t, c] = act( sum_{j < K} weight[c, j] * x[b, t - (K - 1) + j, c] )
//! ```
//!
//! where `act` is SiLU (`silu = true`) or the identity. There is no bias: Qwen3.5 builds this
//! convolution with `bias=False`. This is transformers 5.15's `causal_conv1d_fn` (its torch
//! fallback: `F.conv1d(groups = C, padding = K - 1)` cut to the first `T` outputs, then the
//! activation) and `causal_conv1d_update` (`torch.cat` of the state and the new token, the same
//! convolution, the last output), in these layouts:
//!
//! - `x`, the output, and their gradients: `[B, T, C]`, which is what the projection produces
//!   and what the query, key and value split takes. transformers hands its function the same
//!   memory transposed to `[B, C, T]` and transposes the result back; `causal_conv1d(x, w,
//!   true)` here equals `causal_conv1d_fn(x.transpose(1, 2), w, None, "silu").transpose(1, 2)`.
//! - `weight`: `[C, K]`, the checkpoint's `conv1d.weight` (`[C, 1, K]`) without its middle
//!   axis, as transformers passes it. `weight[c, K - 1]` multiplies the current step.
//! - the decoding state: `[B, C, S]` with `S >= K - 1`, the last `S` inputs, oldest first, as
//!   transformers' cache holds it (its cache keeps `S = K`; `S = K - 1` is the least that works).
//! - one token: `[B, C]`.
//!
//! `x`, `weight` and the state share one float dtype. The CUDA kernels compute in f32 whatever
//! that dtype is and round once, on the way out; the weight gradient is reduced in f32 too.
//!
//! On CUDA ([`causal_conv1d`] with kernel width 4) three kernels run: the forward pass, one
//! backward pass that produces the input gradient and partial sums of the weight gradient over
//! runs of 64 steps, and a reduction of those partial sums. SiLU is fused into both passes; the backward
//! pass recomputes the pre-activation from `x` rather than keeping it, so training holds `x` and
//! the weight and nothing else for this operation. [`causal_conv1d_update`] is one kernel. Other
//! kernel widths, and every other device, run the reference: the trait's default bodies, built
//! from Burn's tensor operations.

use burn::backend::autodiff::NodeId;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Autodiff, Backend, Dispatch, backend_extension};
use burn::tensor::Tensor;

#[cfg(feature = "flex")]
use burn::backend::Flex;

#[cfg(feature = "cuda")]
mod cube;
mod reference;

/// The kernel width the CUDA kernels are written for; others run the reference.
pub const KERNEL_WIDTH: usize = 4;

/// How many times each CUDA kernel and the written-out backward pass ran, so that tests can tell
/// them from the reference, which computes the same values.
#[doc(hidden)]
pub mod counters {
    use std::sync::atomic::AtomicUsize;

    /// Launches of the forward kernel.
    pub static FORWARD_KERNEL: AtomicUsize = AtomicUsize::new(0);
    /// Launches of the backward kernel (input gradient and weight-gradient partial sums).
    pub static BACKWARD_KERNEL: AtomicUsize = AtomicUsize::new(0);
    /// Launches of the single-token update kernel.
    pub static UPDATE_KERNEL: AtomicUsize = AtomicUsize::new(0);
    /// Runs of the backward pass written out for `Autodiff`.
    pub static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);
}

/// The causal depthwise convolution as a backend extension. The default bodies are the
/// reference; see the [module](self) for layouts and semantics.
#[backend_extension(Flex: cfg(feature = "flex"), Cube: cfg(feature = "cuda"), Fusion: cfg(feature = "fusion"), Autodiff)]
pub trait CausalConv1dBackend: Backend {
    /// The convolution over a sequence: `x` `[B, T, C]` and `weight` `[C, K]` to `[B, T, C]`,
    /// followed by SiLU when `silu`.
    #[fusion(dtype = x, shape = x)]
    fn causal_conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        silu: bool,
    ) -> FloatTensor<Self> {
        reference::forward::<Self>(x, weight, silu)
    }

    /// The gradients of [`causal_conv1d`](Self::causal_conv1d) with respect to `x` and `weight`,
    /// given the gradient of its output, `grad` `[B, T, C]`.
    #[fusion(meta = |x, weight, _grad, _silu| (x.clone(), weight.clone()))]
    fn causal_conv1d_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        silu: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        reference::backward::<Self>(x, weight, grad, silu)
    }

    /// One decoding step: the new token `x` `[B, C]`, the state `[B, C, S]` and `weight`
    /// `[C, K]`, `S >= K - 1`, to the token's output `[B, C]` and the next state `[B, C, S]`
    /// (the state without its oldest input, followed by `x`).
    #[fusion(meta = |x, state, _weight, _silu| (x.clone(), state.clone()))]
    fn causal_conv1d_update(
        x: FloatTensor<Self>,
        state: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        silu: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        reference::update::<Self>(x, state, weight, silu)
    }
}

#[cfg(feature = "flex")]
impl CausalConv1dBackend for Flex {}

impl<B: Backend + CausalConv1dBackend, C: CheckpointStrategy> CausalConv1dBackend
    for Autodiff<B, C>
{
    fn causal_conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        silu: bool,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct CausalConv1d;

        impl<B: Backend + CausalConv1dBackend> Backward<B, 2> for CausalConv1d {
            // The input and the weight, through the checkpointer, and the activation flag. The
            // pre-activation is recomputed from them in the backward pass instead of being kept.
            type State = (NodeId, NodeId, bool);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                counters::BACKWARD_RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let [node_x, node_weight] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (x_state, weight_state, silu) = ops.state;
                let x = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(x_state);
                let weight =
                    checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(weight_state);
                // One pass yields both; the weight gradient's share of it is small (partial sums
                // per channel and tile), so it is not worth a second variant without it.
                let (grad_x, grad_weight) = B::causal_conv1d_backward(x, weight, grad, silu);
                if let Some(node) = node_x {
                    grads.register::<B>(node.id, grad_x);
                }
                if let Some(node) = node_weight {
                    grads.register::<B>(node.id, grad_weight);
                }
            }
        }

        match CausalConv1d
            .prepare::<C>([x.node(), weight.node()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let weight_state = prep.checkpoint(&weight);
                let out = B::causal_conv1d(x.into_primitive(), weight.into_primitive(), silu);
                prep.finish((x_state, weight_state, silu), out)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::causal_conv1d(
                x.into_primitive(),
                weight.into_primitive(),
                silu,
            )),
        }
    }

    // `causal_conv1d_backward` keeps the default body: under `Autodiff` it is differentiable
    // through Burn's own operations, which is all a second derivative needs.

    fn causal_conv1d_update(
        x: FloatTensor<Self>,
        state: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        silu: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Decoding needs no gradient, so the kernel's results are recorded as constants. Should a
        // gradient be asked for, the reference runs instead, which Burn differentiates.
        if x.is_tracked() || state.is_tracked() || weight.is_tracked() {
            return reference::update::<Self>(x, state, weight, silu);
        }

        #[derive(Debug)]
        struct Untracked;

        impl<B: Backend> Backward<B, 3> for Untracked {
            type State = ();

            fn backward(self, _: Ops<(), 3>, _: &mut Gradients, _: &mut Checkpointer) {
                unreachable!("an untracked decoding step has no backward pass");
            }
        }

        let nodes = || [x.node(), state.node(), weight.node()];
        let (out, next) = B::causal_conv1d_update(
            x.primitive().clone(),
            state.primitive().clone(),
            weight.primitive().clone(),
            silu,
        );
        let wrap = |tensor| match Untracked.prepare::<C>(nodes()).compute_bound().stateful() {
            OpsKind::UnTracked(prep) => prep.finish(tensor),
            OpsKind::Tracked(_) => unreachable!("no input of this step is tracked"),
        };
        (wrap(out), wrap(next))
    }
}

/// The causal depthwise convolution of `x` `[B, T, C]` with `weight` `[C, K]`, followed by
/// SiLU when `silu`: `[B, T, C]`. Differentiable with respect to both inputs. See the
/// [module](self) for the exact definition and its correspondence to transformers.
pub fn causal_conv1d(x: Tensor<3>, weight: Tensor<2>, silu: bool) -> Tensor<3> {
    check_sequence(&x.dims(), &weight.dims());
    assert_eq!(
        x.dtype(),
        weight.dtype(),
        "causal_conv1d: x and weight must share one dtype"
    );
    Tensor::from_dispatch(<Dispatch as CausalConv1dBackend>::causal_conv1d(
        x.into_dispatch(),
        weight.into_dispatch(),
        silu,
    ))
}

/// One decoding step: the new token `x` `[B, C]`, the `state` `[B, C, S]` (the last `S >= K - 1`
/// inputs, oldest first) and `weight` `[C, K]` give the token's output `[B, C]`, followed by
/// SiLU when `silu`, and the next state `[B, C, S]`. The output equals the last step of
/// [`causal_conv1d`] over the inputs the state holds followed by `x`.
pub fn causal_conv1d_update(
    x: Tensor<2>,
    state: Tensor<3>,
    weight: Tensor<2>,
    silu: bool,
) -> (Tensor<2>, Tensor<3>) {
    let [b, c] = x.dims();
    let [sb, sc, s] = state.dims();
    let [wc, k] = weight.dims();
    assert!(
        k >= 1,
        "causal_conv1d_update: the kernel width must be at least 1"
    );
    assert!(
        sb == b && sc == c && wc == c,
        "causal_conv1d_update: x [{b}, {c}], state [{sb}, {sc}, {s}] and weight [{wc}, {k}] \
         disagree on the batch or the channels"
    );
    assert!(
        s + 1 >= k,
        "causal_conv1d_update: a state of {s} steps cannot feed a kernel of width {k}; it needs \
         at least {}",
        k - 1
    );
    assert!(
        x.dtype() == state.dtype() && x.dtype() == weight.dtype(),
        "causal_conv1d_update: x, state and weight must share one dtype"
    );
    let (out, next) = <Dispatch as CausalConv1dBackend>::causal_conv1d_update(
        x.into_dispatch(),
        state.into_dispatch(),
        weight.into_dispatch(),
        silu,
    );
    (Tensor::from_dispatch(out), Tensor::from_dispatch(next))
}

/// The decoding state a prompt `x` `[B, T, C]` leaves: its last `state_len` inputs, oldest
/// first, `[B, C, state_len]`, with zeros in front when the prompt is shorter, as transformers'
/// cache records it after a prefill.
pub fn causal_conv1d_state(x: Tensor<3>, state_len: usize) -> Tensor<3> {
    let [b, t, c] = x.dims();
    let tail = if t >= state_len {
        x.slice([0..b, t - state_len..t, 0..c])
    } else {
        let zeros = Tensor::<3>::zeros([b, state_len - t, c], (&x.device(), x.dtype()));
        Tensor::cat(vec![zeros, x], 1)
    };
    tail.swap_dims(1, 2)
}

fn check_sequence(x: &[usize; 3], weight: &[usize; 2]) {
    let [_, _, c] = *x;
    let [wc, k] = *weight;
    assert!(k >= 1, "causal_conv1d: the kernel width must be at least 1");
    assert_eq!(
        c, wc,
        "causal_conv1d: x [B, T, C] has {c} channels, weight [C, K] has {wc}"
    );
}
