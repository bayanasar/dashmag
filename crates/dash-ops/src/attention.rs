//! Scaled dot-product attention with grouped key-value heads and the masks models train with:
//! none, causality, key padding, packed variable-length sequences and a sliding window. The
//! backward pass keeps no T x T scores, and decoding runs against a key-value cache.
//!
//! [`dash_contract::attention`] defines the operation, which every implementation of it shares:
//! what it computes, its shapes and dtypes, which keys a query sees under the options
//! ([`AttentionOptions`], [`Window`]) and the masks ([`AttentionMask`]: none, key padding, packed
//! sequences), what a query that sees no key returns, and decoding against a key-value cache.
//! This module implements it on Burn.
//!
//! # Functions
//!
//! - [`attention`], [`attention_with_lse`]: for training; on an autodiff device the backward pass
//!   is [`attention_backward`], which callers running their own backward pass can also call.
//! - [`attention_decode`]: one query per sequence against a key-value cache, with optional
//!   per-sequence cache lengths (`[B]`, Int: sequence `b` holds `lengths[b]` keys, the last of
//!   them the query's own) and the window's `left`. Decoding takes no gradient: on an autodiff
//!   device, a step with a tracked input and no lengths runs the training path with one query, so
//!   that its written-out backward applies, and one with lengths runs the reference on the
//!   autodiff backend.
//! - [`causal_attention`], [`causal_attention_with_lse`], [`causal_attention_backward`]: the
//!   above with no mask, causality on and the scale given.
//!
//! # Implementations
//!
//! The reference is the default body of every method of [`AttentionBackend`]: the textbook
//! formulation from Burn's tensor operations in f32, the key-value heads expanded and the
//! T x T scores and mask materialised. Flex runs it, the tests hold the kernels to it, and a
//! CubeCL device runs it where the kernels do not take the inputs or the device lacks what they
//! ask. On `Autodiff` the forward records one node whose backward calls
//! [`AttentionBackend::attention_backward`].
//!
//! The CubeCL kernels take bf16 and f16 with a head dimension of 64, 128 or 256, every mask
//! above, on a device with `mma.sync` m16n8k16 and `ldmatrix` for that dtype, planes of exactly
//! 32 units (the tensor-core fragments and the plane shuffles are laid out for them), 256 units
//! per cube, up to 97 KiB of shared memory per cube and, for the backward, atomic f32 additions
//! in global memory: CUDA cards of compute capability 8.0 and later, where they always run;
//! elsewhere the reference runs (`kernels_take` tells which).
//!
//! - Training: a FlashAttention-2 style forward that also writes each query's log-sum-exp, and a
//!   backward that by default gives the same bits on every run: `delta = rowsum(dO * O)`; the
//!   query gradient by a kernel of its own, each block of queries sweeping in order the key
//!   blocks it sees and computing the scores, the probabilities, their gradient `dP = dO V^T`,
//!   `dS = P (dP - delta)` and `dQ += dS K` tile by tile; and the products of each key block,
//!   sweeping its query blocks, and the query heads of the group, computing the scores, `P`,
//!   `dP` and `dS` once per tile and accumulating `dK` and `dV` in registers: seven T x T x D
//!   products in all. With [`AttentionOptions::deterministic`] off the backward does
//!   FlashAttention-2's five: there is no query gradient kernel, the products of each key block
//!   also add each tile's `dQ = dS K` to an f32 accumulator with atomic additions, and a last
//!   kernel rounds it to the inputs' dtype. Both passes recompute the scores tile by tile from
//!   the log-sum-exp, so neither holds T x T, and grouped heads are handled inside the kernels:
//!   nothing is expanded. Each query's visible keys are one interval (the mask kinds above all
//!   give one, apart from padding), so every block of queries sweeps only the key blocks some of
//!   its queries see, every block of keys only the query blocks that see some of its keys, and
//!   per-element masking runs only on tiles that cut an interval or carry padding: tiles wholly
//!   outside the causal triangle, the window or a sequence cost nothing. A padding mask is
//!   applied per element. Tiles go through the tensor cores (`mma.sync` m16n8k16, f32
//!   accumulation, operands staged through swizzled shared memory with `ldmatrix`). When the key
//!   blocks would fill the multiprocessors unevenly, two cubes share each key block, each
//!   sweeping half its query blocks, and add their `dK` and `dV` to f32 accumulators, which the
//!   last kernel rounds.
//!
//!   With atomics, the query gradient's sum over the key blocks comes in whatever order the cubes
//!   run, so its last bits vary from run to run: by a unit in the dtype's last place where it is
//!   large, and where its terms cancel by the f32 sum's reordering, a few millionths of its
//!   largest value. PyTorch's and flash-attn's flash backward sum it that way unless asked not to
//!   (flash-attn's `deterministic`). The output, the log-sum-exp and the key and value gradients
//!   are the same bits from run to run, and the same in both ways (split, each key and value
//!   gradient is two addends added to zero, in either order the same).
//! - Decoding: the cache is split (flash-decoding), counted back from each sequence's last key so
//!   that splits past its length or before its window cost nothing; each key and value row is
//!   loaded once for all the query heads of its group; then the splits are merged by their
//!   log-sum-exp. Groups of more than 16 query heads per key-value head run the reference.
//!
//! Inputs may have any strides. The kernels read an input in place when its last dimension is
//! contiguous, its other strides are whole numbers of 16-byte vectors and its first element is
//! 16-byte aligned. That holds for a contiguous tensor, for a slice of one along its other
//! dimensions (a cache longer than the keys used; the queries, keys and values of one fused
//! projection) and for a permutation of those dimensions (`[B, H, T, D]` storage seen through
//! `swap_dims(1, 2)`); the kernels copy any other input. They write the output, the log-sum-exp
//! and the gradients contiguous. `probe` counts the kernel launches, the runs of the written-out
//! backward, the inputs the kernels copied, the backward passes whose products split and those
//! that summed the query gradient with atomics.
//!
//! # Measured
//!
//! Against PyTorch 2.13's `scaled_dot_product_attention` in float64 with an explicit boolean
//! mask (the tests' records: every mask above, grouped heads, `Tq != Tk`, queries that see no
//! key; packed sequences one call per sequence), the reference in f32 agrees to within 4.06e-7 of
//! each output's and gradient's largest value on Flex (on CUDA 5.29e-7, 7.11e-7 with autotune;
//! 8.12e-7 on lavapipe), and the kernels to within 4.15e-3 in bf16 and 7.09e-4 in f16. Against
//! the f32 reference at shapes of several tiles and sequences, the bf16 kernels agree to within
//! 4.53e-3, but for the query gradient's 7.31e-3 at the queries that see few keys (the start of a
//! causal sequence), whose `delta = rowsum(dO * O)` comes from the output rounded to bf16, as in
//! FlashAttention-2; f16 to within 6.94e-4. Both ways of summing the query gradient are held to
//! the same bounds; with atomics its bits move from run to run by up to 5.24e-4 of its largest
//! value, and its error moves less. A causal sequence run in two segments, the second's queries
//! against both segments' keys, composes into the whole sequence's outputs and gradients to f32
//! rounding (7.23e-7).
//!
//! On an RTX 5070 Ti with 6144 tokens, 8 query and 2 key-value heads of 256 in bf16
//! (`examples/attention_bench.rs`, PyTorch 2.13 run in turn with it on an otherwise idle card),
//! forward and backward kernels take 9.17 ms of device time causal by default and 7.62 ms with
//! atomics, against FlashAttention's 7.30 ms with its grouped heads expanded and their gradients
//! summed: the forward 2.46 ms (1.94); the backward 6.70 ms, of which the query gradient's kernel
//! 2.73 ms and the other products 3.87 ms, or with atomics 5.16 ms, of which the products 4.88 ms
//! (FlashAttention's 5.37 ms, the grouped heads' expansion and sums 0.22 ms of it). They grow the
//! memory pool by 60 MiB (flash 204 MiB at its peak), and the backward holds its f32 accumulators
//! meanwhile: 24 MiB where the products split (the key and value gradients), 72 MiB with atomics
//! (the query gradient's too). With no mask they take 17.6 ms, 14.9 ms with atomics (14.7 ms); four
//! packed sequences of 1536, 2.59 ms and 2.33 ms (2.34 ms as a padded batch of four); causal within
//! a window of 1024, 3.07 ms and 2.69 ms (PyTorch has no fused window at this head width:
//! flex_attention 4.35 ms). Decoding a batch of 8 against 6144 cached keys takes 131 us a layer of
//! device time (130 us), 36 us with cache lengths from 100 to 6144 (3663 us through a padding mask)
//! and 15 us within a window of 1024 (26 us on the last 1025 keys); by wall clock over 24 calls
//! between synchronisations 146, 59 and 37 us (137, 3716 and 34 us), the last two set by the host's
//! dispatch rather than the device.

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::{BoolTensor, FloatTensor, IntTensor};
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Dispatch, ExtensionType, Shape, TensorMetadata,
    backend_extension,
};
use burn::tensor::{Bool, DType, Device, Int, Tensor};
use std::sync::atomic::Ordering;

#[cfg(feature = "flex")]
use burn::backend::Flex;

#[cfg(feature = "cubecl")]
mod cube;
#[cfg(feature = "cubecl")]
mod kernels;
mod reference;

/// Counters tests read to tell which path ran, and a switch they set to run each. Not part of
/// the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

    pub(super) static KERNEL_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);
    pub(super) static INPUT_COPIES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static SPLIT_RUNS: AtomicUsize = AtomicUsize::new(0);
    pub(super) static ATOMIC_RUNS: AtomicUsize = AtomicUsize::new(0);
    /// 0: the backward's products split by the call's size; 1: never; 2: always.
    static SPLIT: AtomicU8 = AtomicU8::new(0);

    /// Launches of this module's CubeCL kernels: the forward adds one; the backward three
    /// (`delta`, the query gradient, the other products) and a fourth when the products split
    /// (the key and value gradients' rounding from their accumulators), or with atomics three
    /// (`delta`, the products, the gradients' rounding); decoding two (the splits, then their
    /// merge). Zero without a CubeCL runtime's feature.
    pub fn kernel_launches() -> usize {
        KERNEL_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Runs of the written-out backward pass on `Autodiff`.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }

    /// Inputs the kernels copied because they could not read them in place (the module
    /// documentation says which they read in place).
    pub fn input_copies() -> usize {
        INPUT_COPIES.load(Ordering::SeqCst)
    }

    /// Backward passes whose products split each key block's sweep between two cubes.
    pub fn split_runs() -> usize {
        SPLIT_RUNS.load(Ordering::SeqCst)
    }

    /// Backward passes whose kernels summed the query gradient with atomics
    /// ([`AttentionOptions::deterministic`](super::AttentionOptions::deterministic) off).
    pub fn atomic_runs() -> usize {
        ATOMIC_RUNS.load(Ordering::SeqCst)
    }

    /// Makes the backward's products split each key block's sweep (`Some(true)`), keep it whole
    /// (`Some(false)`), or choose by the call's size (`None`, the default), so that tests run
    /// both paths whatever the device.
    pub fn force_split(split: Option<bool>) {
        let mode = match split {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        };
        SPLIT.store(mode, Ordering::SeqCst);
    }

    /// What [`force_split`] asked for.
    #[cfg(feature = "cubecl")]
    pub(super) fn forced_split() -> Option<bool> {
        match SPLIT.load(Ordering::SeqCst) {
            1 => Some(false),
            2 => Some(true),
            _ => None,
        }
    }
}

pub use dash_contract::attention::{AttentionOptions, Window};

/// [`dash_contract::attention::AttentionMask`] over Burn's tensors: the key padding mask a
/// `Tensor<2, Bool>`, the offsets of packed sequences `Tensor<1, Int>`.
pub type AttentionMask = dash_contract::attention::AttentionMask<Tensor<2, Bool>, Tensor<1, Int>>;

/// How far a query sees before and after its position, `usize::MAX` for no bound: the window,
/// and causality's bound after it.
pub(crate) trait Reach {
    fn reach(&self) -> (usize, usize);
}

impl Reach for AttentionOptions {
    fn reach(&self) -> (usize, usize) {
        let (left, right) = self
            .window
            .map_or((usize::MAX, usize::MAX), |w| (w.left, w.right));
        (left, if self.causal { 0 } else { right })
    }
}

/// The mask as the backend extension takes it.
fn mask_primitive(mask: AttentionMask) -> AttentionMaskPrimitive<Dispatch> {
    match mask {
        AttentionMask::None => AttentionMaskPrimitive::None,
        AttentionMask::KeyPadding(mask) => AttentionMaskPrimitive::KeyPadding(mask.into_dispatch()),
        AttentionMask::Packed {
            cu_seqlens_q,
            cu_seqlens_k,
        } => AttentionMaskPrimitive::Packed {
            cu_seqlens_q: cu_seqlens_q.into_dispatch(),
            cu_seqlens_k: cu_seqlens_k.into_dispatch(),
        },
    }
}

/// [`AttentionMask`] as the backend extension takes it, its tensors backend primitives (an
/// `Option` of a tensor does not cross Burn's dispatch boundary; this enum does).
#[derive(ExtensionType, Clone, Debug)]
#[extension_type(fusion: cfg(feature = "fusion"))]
pub enum AttentionMaskPrimitive<B: Backend> {
    /// Every key.
    None,
    /// `[B, Tk]`, `true` where a key is left out.
    KeyPadding(BoolTensor<B>),
    /// Packed variable-length sequences, `B = 1`.
    Packed {
        /// The queries' offsets, `[S + 1]`.
        cu_seqlens_q: IntTensor<B>,
        /// The keys' offsets, `[S + 1]`.
        cu_seqlens_k: IntTensor<B>,
    },
}

/// The cache lengths of a decoding step as the backend extension takes them.
#[derive(ExtensionType, Clone, Debug)]
#[extension_type(fusion: cfg(feature = "fusion"))]
pub enum CacheLengths<B: Backend> {
    /// Every sequence holds the whole cache, `S` keys.
    Full,
    /// `[B]`: sequence `b` holds `lengths[b]` keys, the first of the cache.
    Lengths(IntTensor<B>),
}

/// Attention and its backward pass, with the masks of [`dash_contract::attention`], and
/// decoding against a key-value cache. The default bodies are the reference.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(feature = "cubecl"),
    Fusion: cfg(feature = "fusion"),
    Autodiff
)]
pub trait AttentionBackend: Backend {
    /// The output, `[B, Tq, Hq, D]` in the dtype of `query`, and beside it the natural
    /// log-sum-exp of each query's scaled scores over the keys it sees, `[B, Tq, Hq]` in f32
    /// (`-inf` for a query that sees none). On `Autodiff` the output records one node; the
    /// log-sum-exp is a constant.
    #[fusion(meta = |query, _key, _value, _mask, _options| (
        query.clone(),
        burn::backend::fusion::custom::TensorSpec::new(
            burn::backend::Shape::new([query.shape[0], query.shape[1], query.shape[2]]),
            burn::backend::DType::F32,
        ),
    ))]
    fn attention_forward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        #[extension_type] mask: AttentionMaskPrimitive<Self>,
        options: AttentionOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        reference::forward::<Self>(query, key, value, mask, options)
    }

    /// The gradients of `query`, `key` and `value`, each in its input's shape and dtype, given
    /// the forward's `output` and `lse` and the gradient of the output. Not itself
    /// differentiable.
    #[fusion(meta = |query, key, value, _output, _lse, _grad_output, _mask, _options| (
        query.clone(),
        key.clone(),
        value.clone(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn attention_backward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        output: FloatTensor<Self>,
        lse: FloatTensor<Self>,
        grad_output: FloatTensor<Self>,
        #[extension_type] mask: AttentionMaskPrimitive<Self>,
        options: AttentionOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        reference::backward::<Self>(query, key, value, output, lse, grad_output, mask, options)
    }

    /// One query per sequence, `[B, Hq, D]`, against a key-value cache `[B, S, Hkv, D]`, each
    /// sequence's query after its last key; returns `[B, Hq, D]` in the dtype of `query`. Of the
    /// options, the scale and the window's `left` apply (no key comes after the query).
    #[fusion(dtype = query, shape = query)]
    fn attention_decode(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        #[extension_type] lengths: CacheLengths<Self>,
        options: AttentionOptions,
    ) -> FloatTensor<Self> {
        reference::decode::<Self>(query, key, value, lengths, options)
    }
}

#[cfg(feature = "flex")]
impl AttentionBackend for Flex {}

/// The mask of an autodiff backend as its inner backend takes it: the same tensors, which carry
/// no gradient.
fn inner_mask<B: Backend, C: CheckpointStrategy>(
    mask: AttentionMaskPrimitive<Autodiff<B, C>>,
) -> AttentionMaskPrimitive<B> {
    match mask {
        AttentionMaskPrimitive::None => AttentionMaskPrimitive::None,
        AttentionMaskPrimitive::KeyPadding(mask) => AttentionMaskPrimitive::KeyPadding(mask),
        AttentionMaskPrimitive::Packed {
            cu_seqlens_q,
            cu_seqlens_k,
        } => AttentionMaskPrimitive::Packed {
            cu_seqlens_q,
            cu_seqlens_k,
        },
    }
}

/// The node `Autodiff` records for the forward: its backward calls the inner backend's
/// written-out backward pass.
#[derive(Debug)]
struct AttentionNode;

/// What the backward pass needs: the inputs, the forward's output and log-sum-exp, the mask and
/// the options.
#[derive(Clone, Debug)]
struct NodeState<B: Backend> {
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    output: FloatTensor<B>,
    lse: FloatTensor<B>,
    mask: AttentionMaskPrimitive<B>,
    options: AttentionOptions,
}

impl<B: Backend + AttentionBackend> Backward<B, 3> for AttentionNode {
    type State = NodeState<B>;

    fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients, _: &mut Checkpointer) {
        probe::BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
        let [query_node, key_node, value_node] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let s = ops.state;
        let (grad_query, grad_key, grad_value) = B::attention_backward(
            s.query, s.key, s.value, s.output, s.lse, grad, s.mask, s.options,
        );
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

/// The forward on `Autodiff`: one node, keeping the inputs, the output, the log-sum-exp and the
/// mask. Returns the output and the log-sum-exp.
fn autodiff_forward<B: Backend + AttentionBackend, C: CheckpointStrategy>(
    query: FloatTensor<Autodiff<B, C>>,
    key: FloatTensor<Autodiff<B, C>>,
    value: FloatTensor<Autodiff<B, C>>,
    mask: AttentionMaskPrimitive<B>,
    options: AttentionOptions,
) -> (FloatTensor<Autodiff<B, C>>, FloatTensor<B>) {
    let nodes = [query.node(), key.node(), value.node()];
    let (query, key, value) = (
        query.into_primitive(),
        key.into_primitive(),
        value.into_primitive(),
    );
    match AttentionNode.prepare::<C>(nodes).compute_bound().stateful() {
        OpsKind::Tracked(prep) => {
            let (output, lse) = B::attention_forward(
                query.clone(),
                key.clone(),
                value.clone(),
                mask.clone(),
                options,
            );
            let state = NodeState {
                query,
                key,
                value,
                output: output.clone(),
                lse: lse.clone(),
                mask,
                options,
            };
            (prep.finish(state, output), lse)
        }
        OpsKind::UnTracked(prep) => {
            let (output, lse) = B::attention_forward(query, key, value, mask, options);
            (prep.finish(output), lse)
        }
    }
}

impl<B: Backend + AttentionBackend, C: CheckpointStrategy> AttentionBackend for Autodiff<B, C> {
    fn attention_forward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: AttentionMaskPrimitive<Self>,
        options: AttentionOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (output, lse) =
            autodiff_forward::<B, C>(query, key, value, inner_mask::<B, C>(mask), options);
        (output, <Self as AutodiffBackend>::from_inner(lse))
    }

    fn attention_backward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        output: FloatTensor<Self>,
        lse: FloatTensor<Self>,
        grad_output: FloatTensor<Self>,
        mask: AttentionMaskPrimitive<Self>,
        options: AttentionOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (grad_query, grad_key, grad_value) = B::attention_backward(
            query.into_primitive(),
            key.into_primitive(),
            value.into_primitive(),
            output.into_primitive(),
            lse.into_primitive(),
            grad_output.into_primitive(),
            inner_mask::<B, C>(mask),
            options,
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
        lengths: CacheLengths<Self>,
        options: AttentionOptions,
    ) -> FloatTensor<Self> {
        let tracked = [&query, &key, &value]
            .iter()
            .any(|input| input.is_tracked());
        match lengths {
            // The training forward with one query, which sits after every key: its node's
            // backward pass applies, and the reshapes around it carry the query's gradient back
            // to `[B, Hq, D]`. Causality changes nothing for it.
            CacheLengths::Full if tracked => {
                let [b, hq, d] = query.shape().dims::<3>();
                let query = Self::float_reshape(query, Shape::new([b, 1, hq, d]));
                let options = options.with_causal(false);
                let output = autodiff_forward::<B, C>(
                    query,
                    key,
                    value,
                    AttentionMaskPrimitive::None,
                    options,
                )
                .0;
                Self::float_reshape(output, Shape::new([b, hq, d]))
            }
            // Per-sequence lengths have no training path: the reference, recorded by autodiff.
            lengths if tracked => reference::decode::<Self>(query, key, value, lengths, options),
            lengths => {
                let lengths = match lengths {
                    CacheLengths::Full => CacheLengths::Full,
                    CacheLengths::Lengths(lengths) => CacheLengths::Lengths(lengths),
                };
                <Self as AutodiffBackend>::from_inner(B::attention_decode(
                    query.into_primitive(),
                    key.into_primitive(),
                    value.into_primitive(),
                    lengths,
                    options,
                ))
            }
        }
    }
}

/// The dimensions of one call, `[B, Tq, Tk, Hq, Hkv, D]`, checked against each other and against
/// the mask, as [`dash_contract::attention`] gives them.
///
/// # Panics
///
/// If the shapes do not fit together.
fn check_shapes(
    query: &[usize],
    key: &[usize],
    value: &[usize],
    mask: Option<MaskShape>,
) -> [usize; 6] {
    let [b, tq, hq, d] = query
        .try_into()
        .expect("attention: query is [B, Tq, Hq, D]");
    let [bk, tk, hkv, dk] = key.try_into().expect("attention: key is [B, Tk, Hkv, D]");
    assert_eq!(key, value, "attention: key and value must have one shape");
    assert_eq!(b, bk, "attention: query and key must have one batch size");
    assert_eq!(
        d, dk,
        "attention: query and key must have one head dimension"
    );
    assert!(
        hkv > 0 && hq % hkv == 0,
        "attention: the query heads ({hq}) must be a multiple of the key-value heads ({hkv})"
    );
    match mask {
        None => {}
        Some(MaskShape::KeyPadding(shape)) => assert_eq!(
            shape,
            [b, tk],
            "attention: the key padding mask is [B, Tk] = [{b}, {tk}]"
        ),
        Some(MaskShape::Packed(q, k)) => {
            assert_eq!(
                b, 1,
                "attention: packed sequences lie along the time, B = 1"
            );
            assert!(
                q.len() == 1 && k.len() == 1 && q == k && q[0] >= 1,
                "attention: cu_seqlens_q and cu_seqlens_k are [S + 1], one S, got {q:?}, {k:?}"
            );
        }
    }
    [b, tq, tk, hq, hkv, d]
}

/// The shapes of a mask's tensors, for [`check_shapes`].
enum MaskShape {
    KeyPadding(Vec<usize>),
    Packed(Vec<usize>, Vec<usize>),
}

impl MaskShape {
    fn of<B: Backend>(mask: &AttentionMaskPrimitive<B>) -> Option<Self> {
        match mask {
            AttentionMaskPrimitive::None => None,
            AttentionMaskPrimitive::KeyPadding(mask) => {
                Some(Self::KeyPadding(mask.shape().as_slice().to_vec()))
            }
            AttentionMaskPrimitive::Packed {
                cu_seqlens_q,
                cu_seqlens_k,
            } => Some(Self::Packed(
                cu_seqlens_q.shape().as_slice().to_vec(),
                cu_seqlens_k.shape().as_slice().to_vec(),
            )),
        }
    }
}

/// The dimensions of a decoding step, `[B, S, Hq, Hkv, D]`, checked as [`check_shapes`] does.
fn check_decode_shapes(
    query: &[usize],
    key: &[usize],
    value: &[usize],
    lengths: Option<&[usize]>,
) -> [usize; 5] {
    let [b, hq, d] = query
        .try_into()
        .expect("attention_decode: query is [B, Hq, D]");
    let [_, s, _, _] = key
        .try_into()
        .expect("attention_decode: key is [B, S, Hkv, D]");
    let [_, _, _, _, hkv, _] = check_shapes(&[b, 1, hq, d], key, value, None);
    if let Some(lengths) = lengths {
        assert_eq!(
            lengths,
            [b],
            "attention_decode: the cache lengths are [B] = [{b}]"
        );
    }
    [b, s, hq, hkv, d]
}

/// Attention for training: `query` `[B, Tq, Hq, D]`, `key` and `value` `[B, Tk, Hkv, D]`, the
/// keys each query sees given by `mask` and `options` ([`dash_contract::attention`]); returns
/// `[B, Tq, Hq, D]` in the dtype of `query`. On an autodiff device, the backward pass is
/// [`attention_backward`].
///
/// # Panics
///
/// If the shapes do not fit together ([`dash_contract::attention`]).
pub fn attention(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    mask: AttentionMask,
    options: AttentionOptions,
) -> Tensor<4> {
    attention_with_lse(query, key, value, mask, options).0
}

/// [`attention`] with the natural log-sum-exp of each query's scaled scores over the keys it
/// sees beside the output, `[B, Tq, Hq]` in f32, `-inf` for a query that sees no key.
pub fn attention_with_lse(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    mask: AttentionMask,
    options: AttentionOptions,
) -> (Tensor<4>, Tensor<3>) {
    let mask = mask_primitive(mask);
    check_shapes(
        &query.dims(),
        &key.dims(),
        &value.dims(),
        MaskShape::of(&mask),
    );
    let (output, lse) = <Dispatch as AttentionBackend>::attention_forward(
        query.into_dispatch(),
        key.into_dispatch(),
        value.into_dispatch(),
        mask,
        options,
    );
    (Tensor::from_dispatch(output), Tensor::from_dispatch(lse))
}

/// The backward pass of [`attention`] on its own: the gradients of `query`, `key` and `value`,
/// each in its input's shape and dtype, given the forward's `output` and `lse` (from
/// [`attention_with_lse`]), the gradient of the output, and the forward's mask and options. An
/// autodiff device calls this itself; it is here for callers that run their own backward pass.
#[allow(clippy::too_many_arguments)]
pub fn attention_backward(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    output: Tensor<4>,
    lse: Tensor<3>,
    grad_output: Tensor<4>,
    mask: AttentionMask,
    options: AttentionOptions,
) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    let mask = mask_primitive(mask);
    let [b, tq, _, hq, _, _] = check_shapes(
        &query.dims(),
        &key.dims(),
        &value.dims(),
        MaskShape::of(&mask),
    );
    assert_eq!(output.dims(), query.dims(), "attention_backward: output");
    assert_eq!(grad_output.dims(), query.dims(), "attention_backward: grad");
    assert_eq!(lse.dims(), [b, tq, hq], "attention_backward: lse");
    let (grad_query, grad_key, grad_value) = <Dispatch as AttentionBackend>::attention_backward(
        query.into_dispatch(),
        key.into_dispatch(),
        value.into_dispatch(),
        output.into_dispatch(),
        lse.into_dispatch(),
        grad_output.into_dispatch(),
        mask,
        options,
    );
    (
        Tensor::from_dispatch(grad_query),
        Tensor::from_dispatch(grad_key),
        Tensor::from_dispatch(grad_value),
    )
}

/// Decoding: `query` `[B, Hq, D]`, one token per sequence, against `key` and `value`
/// `[B, S, Hkv, D]`, the cache up to and including that token (a slice of a longer cache is read
/// in place). `cache_lengths`, `[B]` Int, gives how many keys of the cache each sequence holds,
/// `S` for all of them when `None`; each sequence's query sits after its last key. Of the
/// options, the scale and the window's `left` apply. Returns `[B, Hq, D]` in the dtype of
/// `query`; a sequence holding no key gets zeros.
pub fn attention_decode(
    query: Tensor<3>,
    key: Tensor<4>,
    value: Tensor<4>,
    cache_lengths: Option<Tensor<1, Int>>,
    options: AttentionOptions,
) -> Tensor<3> {
    check_decode_shapes(
        &query.dims(),
        &key.dims(),
        &value.dims(),
        cache_lengths
            .as_ref()
            .map(|l| l.dims())
            .as_ref()
            .map(|d| &d[..]),
    );
    let lengths = match cache_lengths {
        None => CacheLengths::Full,
        Some(lengths) => CacheLengths::Lengths(lengths.into_dispatch()),
    };
    Tensor::from_dispatch(<Dispatch as AttentionBackend>::attention_decode(
        query.into_dispatch(),
        key.into_dispatch(),
        value.into_dispatch(),
        lengths,
        options,
    ))
}

/// Causal attention for training with the scale given: [`attention`] with no mask and
/// [`AttentionOptions::causal`].
pub fn causal_attention(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    scale: f32,
) -> Tensor<4> {
    let options = AttentionOptions::causal().with_scale(scale);
    attention(query, key, value, AttentionMask::None, options)
}

/// [`causal_attention`] with each query's log-sum-exp: [`attention_with_lse`].
pub fn causal_attention_with_lse(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    scale: f32,
) -> (Tensor<4>, Tensor<3>) {
    let options = AttentionOptions::causal().with_scale(scale);
    attention_with_lse(query, key, value, AttentionMask::None, options)
}

/// The backward pass of [`causal_attention`]: [`attention_backward`].
pub fn causal_attention_backward(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    output: Tensor<4>,
    lse: Tensor<3>,
    grad_output: Tensor<4>,
    scale: f32,
) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    let options = AttentionOptions::causal().with_scale(scale);
    attention_backward(
        query,
        key,
        value,
        output,
        lse,
        grad_output,
        AttentionMask::None,
        options,
    )
}

/// Whether the CubeCL kernels take a forward and a backward pass over inputs of `dtype` and
/// dimensions `[B, Tq, Tk, Hq, Hkv, D]` on `device`, rather than the reference, whatever the mask
/// and options: bf16 or f16 with a head dimension of 64, 128 or 256, on a device with what the
/// kernels ask (module documentation). `[false, false]` on any device not CubeCL's. For tests,
/// as the [`probe`] counters are.
#[doc(hidden)]
pub fn kernels_take(device: &Device, dtype: DType, dims: [usize; 6]) -> [bool; 2] {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return [false, true].map(|backward| cube::kernels_take(&client, dtype, dims, backward));
    }
    let _ = (device, dtype, dims);
    [false, false]
}

/// Whether the CubeCL kernels take a decoding step over inputs of `dtype` and dimensions
/// `[B, S, Hq, Hkv, D]` on `device`, rather than the reference: as [`kernels_take`], with at
/// most 16 query heads per key-value head. For tests.
#[doc(hidden)]
pub fn decode_kernels_take(device: &Device, dtype: DType, dims: [usize; 5]) -> bool {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return cube::decode_kernels_take(&client, dtype, dims);
    }
    let _ = (device, dtype, dims);
    false
}
