//! Causal attention whose backward pass does not materialise the T x T scores, and decoding
//! against a key-value cache.
//!
//! Shapes follow Burn's `attention` and PyTorch's `scaled_dot_product_attention`: queries
//! `[B, Hq, Tq, D]`, keys and values `[B, Hkv, Tk, D]`, output `[B, Hq, Tq, D]`. `Hq` is a
//! multiple of `Hkv`: query head `h` reads key-value head `h / (Hq / Hkv)`, as PyTorch's
//! `enable_gqa=True` does. `Tk >= Tq`, and the causal mask is aligned at the bottom right: query
//! `i` sees the keys `j <= i + (Tk - Tq)`. With `Tq == Tk` that is the usual lower triangle; a
//! chunk of queries continuing a cached prefix sees the whole prefix; a single query sees every
//! key. Any strides are accepted; the kernels read the last dimension as contiguous and copy an
//! input whose last dimension is not.
//!
//! - [`causal_attention`]: for training. On CUDA, bf16 and f16 inputs with a head dimension of
//!   64, 128 or 256 run a FlashAttention-2 style forward that also writes each query's
//!   log-sum-exp, and a backward of three kernels: `delta = rowsum(dO * O)`, `dQ` sweeping the
//!   key blocks of each query block, and `dK`, `dV` sweeping the query blocks, and the query
//!   heads of the group, of each key block. Both passes recompute the scores tile by tile from
//!   the log-sum-exp, so neither holds T x T, and grouped heads are handled inside the kernels:
//!   nothing is expanded. Tiles go through the tensor cores (`mma.sync` m16n8k16, f32
//!   accumulation, operands staged through swizzled shared memory with `ldmatrix`). No kernel
//!   uses atomics, so the gradients are the same bits from run to run.
//! - [`attention_decode`]: one query per sequence against a key-value cache, split over the
//!   cache (flash-decoding), each key and value row loaded once for all the query heads of its
//!   group, then the splits merged by their log-sum-exp.
//!
//! The reference is the default body of every method: the textbook formulation built from
//! Burn's tensor operations in f32, with the key-value heads expanded and the T x T scores
//! materialised. Flex runs it, the tests hold the kernels to it, and the CUDA backend falls back
//! to it for inputs the kernels do not take (f32, other head dimensions). On `Autodiff`, the
//! forward records one node whose backward calls
//! [`CausalAttentionBackend::causal_attention_backward`].

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Dispatch, Flex, backend_extension, tensor::FloatTensor,
};
use burn::tensor::Tensor;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "cuda")]
mod cube;
#[cfg(feature = "cuda")]
mod kernels;
mod reference;

/// How many times the written-out backward pass ran, so tests can tell it ran rather than the
/// autodiff of the reference.
pub static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);

/// How many CUDA kernels of this module were launched, so tests can tell a kernel ran rather
/// than the reference: the forward adds one, the backward three (`delta`, `dQ`, `dK` with
/// `dV`), decoding two (the splits, then their merge).
pub static KERNEL_LAUNCHES: AtomicUsize = AtomicUsize::new(0);

/// Causal attention with grouped key-value heads, its log-sum-exp, its backward pass, and
/// decoding against a key-value cache. The module documentation gives the shapes.
#[backend_extension(Flex, Cube: cfg(feature = "cuda"), Fusion: cfg(feature = "fusion"), Autodiff)]
pub trait CausalAttentionBackend: Backend {
    /// `softmax(scale * q k^T + causal mask) v`, `[B, Hq, Tq, D]` in the dtype of `query`.
    #[fusion(dtype = query, shape = query)]
    fn causal_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        scale: f32,
    ) -> FloatTensor<Self> {
        reference::forward::<Self>(query, key, value, scale).0
    }

    /// The forward pass and, beside it, the natural log-sum-exp of each query's scaled, masked
    /// scores: `[B, Hq, Tq]`, f32. On `Autodiff` the output records the same node as
    /// [`causal_attention`](Self::causal_attention); the log-sum-exp is a constant.
    #[fusion(meta = |query, _key, _value, _scale| (
        query.clone(),
        burn::backend::fusion::custom::TensorSpec::new(
            burn::backend::Shape::new([query.shape[0], query.shape[1], query.shape[2]]),
            burn::backend::DType::F32,
        ),
    ))]
    fn causal_attention_with_lse(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        reference::forward::<Self>(query, key, value, scale)
    }

    /// The gradients of `query`, `key` and `value`, in their dtypes, given the forward's
    /// `output` and `lse` and the gradient of the output. Not itself differentiable.
    #[fusion(meta = |query, key, value, _output, _lse, _grad_output, _scale| (
        query.clone(),
        key.clone(),
        value.clone(),
    ))]
    fn causal_attention_backward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        output: FloatTensor<Self>,
        lse: FloatTensor<Self>,
        grad_output: FloatTensor<Self>,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        reference::backward::<Self>(query, key, value, output, lse, grad_output, scale)
    }

    /// One query per sequence, `[B, Hq, 1, D]`, against a key-value cache `[B, Hkv, S, D]`: the
    /// query sees all `S` keys. Mathematically [`causal_attention`](Self::causal_attention)
    /// with `Tq = 1`; on CUDA a kernel shaped for it. On `Autodiff`, inputs that require
    /// gradients take the training path instead, so that the backward pass applies.
    #[fusion(dtype = query, shape = query)]
    fn attention_decode(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        scale: f32,
    ) -> FloatTensor<Self> {
        reference::forward::<Self>(query, key, value, scale).0
    }
}

impl CausalAttentionBackend for Flex {}

/// The node `Autodiff` records for the forward: its backward calls the inner backend's
/// written-out backward pass.
#[derive(Debug)]
struct CausalAttentionNode;

/// What the backward pass needs: the inputs, the forward's output and log-sum-exp, the scale.
type NodeState<B> = (
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    f32,
);

impl<B: Backend + CausalAttentionBackend> Backward<B, 3> for CausalAttentionNode {
    type State = NodeState<B>;

    fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients, _: &mut Checkpointer) {
        BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
        let [query_node, key_node, value_node] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let (query, key, value, output, lse, scale) = ops.state;
        let (grad_query, grad_key, grad_value) =
            B::causal_attention_backward(query, key, value, output, lse, grad, scale);
        if let Some(node) = query_node {
            grads.register::<B>(node.id, grad_query);
        }
        if let Some(node) = key_node {
            grads.register::<B>(node.id, grad_key);
        }
        if let Some(node) = value_node {
            grads.register::<B>(node.id, grad_value);
        }
    }
}

/// Which inner method the forward on `Autodiff` runs when nothing is tracked.
#[derive(Clone, Copy, PartialEq)]
enum Untracked {
    Plain,
    WithLse,
    Decode,
}

/// The forward on `Autodiff`: one node, keeping the inputs, the output and the log-sum-exp.
/// Returns the output and, when tracked or asked for, the log-sum-exp.
fn autodiff_forward<B: Backend + CausalAttentionBackend, C: CheckpointStrategy>(
    query: FloatTensor<Autodiff<B, C>>,
    key: FloatTensor<Autodiff<B, C>>,
    value: FloatTensor<Autodiff<B, C>>,
    scale: f32,
    untracked: Untracked,
) -> (FloatTensor<Autodiff<B, C>>, Option<FloatTensor<B>>) {
    let nodes = [query.node(), key.node(), value.node()];
    let (query, key, value) = (
        query.into_primitive(),
        key.into_primitive(),
        value.into_primitive(),
    );
    match CausalAttentionNode
        .prepare::<C>(nodes)
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            let (output, lse) =
                B::causal_attention_with_lse(query.clone(), key.clone(), value.clone(), scale);
            let state = (query, key, value, output.clone(), lse.clone(), scale);
            (prep.finish(state, output), Some(lse))
        }
        OpsKind::UnTracked(prep) => match untracked {
            Untracked::Plain => (
                prep.finish(B::causal_attention(query, key, value, scale)),
                None,
            ),
            Untracked::WithLse => {
                let (output, lse) = B::causal_attention_with_lse(query, key, value, scale);
                (prep.finish(output), Some(lse))
            }
            Untracked::Decode => (
                prep.finish(B::attention_decode(query, key, value, scale)),
                None,
            ),
        },
    }
}

impl<B: Backend + CausalAttentionBackend, C: CheckpointStrategy> CausalAttentionBackend
    for Autodiff<B, C>
{
    fn causal_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        scale: f32,
    ) -> FloatTensor<Self> {
        autodiff_forward::<B, C>(query, key, value, scale, Untracked::Plain).0
    }

    fn causal_attention_with_lse(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (output, lse) = autodiff_forward::<B, C>(query, key, value, scale, Untracked::WithLse);
        let lse = lse.expect("the forward with its log-sum-exp returns it");
        (output, <Self as AutodiffBackend>::from_inner(lse))
    }

    fn causal_attention_backward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        output: FloatTensor<Self>,
        lse: FloatTensor<Self>,
        grad_output: FloatTensor<Self>,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (grad_query, grad_key, grad_value) = B::causal_attention_backward(
            query.into_primitive(),
            key.into_primitive(),
            value.into_primitive(),
            output.into_primitive(),
            lse.into_primitive(),
            grad_output.into_primitive(),
            scale,
        );
        (
            <Self as AutodiffBackend>::from_inner(grad_query),
            <Self as AutodiffBackend>::from_inner(grad_key),
            <Self as AutodiffBackend>::from_inner(grad_value),
        )
    }

    fn attention_decode(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        scale: f32,
    ) -> FloatTensor<Self> {
        autodiff_forward::<B, C>(query, key, value, scale, Untracked::Decode).0
    }
}

/// Checks the shapes the module documentation gives and returns `(B, Hq, Hkv, Tq, Tk, D)`.
fn check_shapes(query: &[usize], key: &[usize], value: &[usize]) -> [usize; 6] {
    let [b, hq, tq, d] = query.try_into().expect("query is [B, Hq, Tq, D]");
    let [bk, hkv, tk, dk] = key.try_into().expect("key is [B, Hkv, Tk, D]");
    assert_eq!(key, value, "key and value must have one shape");
    assert_eq!(b, bk, "query and key must have one batch size");
    assert_eq!(d, dk, "query and key must have one head dimension");
    assert!(
        hkv > 0 && hq % hkv == 0,
        "the query heads ({hq}) must be a multiple of the key-value heads ({hkv})"
    );
    assert!(
        tk >= tq,
        "there must be at least as many keys ({tk}) as queries ({tq})"
    );
    [b, hq, hkv, tq, tk, d]
}

/// Causal attention for training: `query` `[B, Hq, Tq, D]`, `key` and `value` `[B, Hkv, Tk, D]`,
/// output `[B, Hq, Tq, D]` in the dtype of `query`; `scale` multiplies the scores, usually
/// `1 / sqrt(D)`. On an autodiff device, the backward pass is
/// [`CausalAttentionBackend::causal_attention_backward`].
pub fn causal_attention(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    scale: f32,
) -> Tensor<4> {
    check_shapes(&query.dims(), &key.dims(), &value.dims());
    Tensor::from_dispatch(<Dispatch as CausalAttentionBackend>::causal_attention(
        query.into_dispatch(),
        key.into_dispatch(),
        value.into_dispatch(),
        scale,
    ))
}

/// [`causal_attention`] with the natural log-sum-exp of each query's scaled scores beside the
/// output, `[B, Hq, Tq]` in f32.
pub fn causal_attention_with_lse(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    scale: f32,
) -> (Tensor<4>, Tensor<3>) {
    check_shapes(&query.dims(), &key.dims(), &value.dims());
    let (output, lse) = <Dispatch as CausalAttentionBackend>::causal_attention_with_lse(
        query.into_dispatch(),
        key.into_dispatch(),
        value.into_dispatch(),
        scale,
    );
    (Tensor::from_dispatch(output), Tensor::from_dispatch(lse))
}

/// The backward pass of [`causal_attention`] on its own: the gradients of `query`, `key` and
/// `value` given the forward's `output` and `lse` (from [`causal_attention_with_lse`]) and the
/// gradient of the output. An autodiff device calls this itself; it is here for callers that
/// run their own backward pass.
pub fn causal_attention_backward(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    output: Tensor<4>,
    lse: Tensor<3>,
    grad_output: Tensor<4>,
    scale: f32,
) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    check_shapes(&query.dims(), &key.dims(), &value.dims());
    let (grad_query, grad_key, grad_value) =
        <Dispatch as CausalAttentionBackend>::causal_attention_backward(
            query.into_dispatch(),
            key.into_dispatch(),
            value.into_dispatch(),
            output.into_dispatch(),
            lse.into_dispatch(),
            grad_output.into_dispatch(),
            scale,
        );
    (
        Tensor::from_dispatch(grad_query),
        Tensor::from_dispatch(grad_key),
        Tensor::from_dispatch(grad_value),
    )
}

/// Decoding: `query` `[B, Hq, 1, D]`, one token per sequence, against `key` and `value`
/// `[B, Hkv, S, D]`, the cache up to and including that token (a slice of a longer cache works:
/// any strides are read in place). Returns `[B, Hq, 1, D]`.
pub fn attention_decode(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    scale: f32,
) -> Tensor<4> {
    let [_, _, _, tq, _, _] = check_shapes(&query.dims(), &key.dims(), &value.dims());
    assert_eq!(tq, 1, "attention_decode takes one query per sequence");
    Tensor::from_dispatch(<Dispatch as CausalAttentionBackend>::attention_decode(
        query.into_dispatch(),
        key.into_dispatch(),
        value.into_dispatch(),
        scale,
    ))
}
