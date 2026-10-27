//! Scaled dot-product attention with grouped key-value heads and the masks models train with:
//! none, causality, key padding, packed variable-length sequences and a sliding window. The
//! backward pass keeps no T x T scores, and decoding runs against a key-value cache.
//!
//! # The operation
//!
//! For each batch element, query head `h` and query `i`, over the keys `j` the query sees:
//!
//! ```text
//! out[i] = sum_j p[i, j] v[j],   p[i, j] = exp(s[i, j] - lse[i]),   s[i, j] = scale q[i] . k[j]
//! lse[i] = log sum_j exp(s[i, j])
//! ```
//!
//! with the keys and values of key-value head `h / (Hq / Hkv)`: `Hq` is a multiple of `Hkv`, as
//! with PyTorch's `enable_gqa=True`. `scale` defaults to `1 / sqrt(D)`.
//!
//! # Shapes and dtypes
//!
//! The crate's layout: queries `[B, Tq, Hq, D]`, keys and values `[B, Tk, Hkv, D]`, the output
//! `[B, Tq, Hq, D]` and the log-sum-exp `[B, Tq, Hq]`, the order a projection's `[B, T, H * D]`
//! output reshapes to without a copy (PyTorch's `scaled_dot_product_attention` takes the heads
//! before the time, `[B, H, T, D]`; flash-attn takes this order). `Tq` and `Tk` are independent.
//! A decoding step drops the time: one query per sequence, `[B, Hq, D]`, against the cache
//! `[B, S, Hkv, D]`, returns `[B, Hq, D]`. Queries, keys and values share one float dtype (f32,
//! bf16 or f16), which the output takes; the log-sum-exp is f32; everything is accumulated in
//! f32, and each gradient is rounded once, to its input's dtype.
//!
//! # Which keys a query sees
//!
//! Positions are aligned at the bottom right: query `i` sits at key position `i + Tk - Tq`, so
//! that with `Tq == Tk` query `i` is key `i`, a chunk of queries continuing a cached prefix
//! comes after the whole prefix, and a single query comes after every key. Around that position
//! [`AttentionOptions`] says how far a query sees, as flash-attn's `causal` and `window_size`:
//!
//! - `causal`: no key after the query's position;
//! - `window`, [`Window`] `{ left, right }`: no key more than `left` positions before it or
//!   `right` after it (with `causal`, none after it whatever `right` says).
//!
//! Beside that, the [`AttentionMask`] passed as an argument, since it is data:
//!
//! - [`AttentionMask::None`]: every key, for encoders and cross-attention;
//! - [`AttentionMask::KeyPadding`]: a boolean mask `[B, Tk]`, `true` where a key is left out
//!   (padding), as Burn's masks are;
//! - [`AttentionMask::Packed`]: variable-length sequences packed end to end along the time,
//!   `B = 1`, their boundaries the cumulative offsets `cu_seqlens_q` and `cu_seqlens_k`
//!   (`[S + 1]`, Int) of flash-attn's variable-length API. A query sees only the keys of its own
//!   sequence, and causality and the window apply within each sequence, aligned at its bottom
//!   right. Offsets start at 0 and do not decrease; positions past the last offset belong to no
//!   sequence: those queries see no key and those keys are seen by no query.
//!
//! A query that sees no key at all (every key padded, a causal query before the first key when
//! `Tk < Tq`, an empty window) returns zeros and a log-sum-exp of `-inf`, and takes no part in
//! any gradient: its own query gradient is zero. flash-attn does the same, and so does PyTorch
//! 2.13's `scaled_dot_product_attention` on the CPU (its math backend and its kernel alike).
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
//!   backward in FlashAttention-2's five products, in three kernels: `delta = rowsum(dO * O)`;
//!   the products, sweeping the query blocks, and the query heads of the group, of each key
//!   block, computing the scores, the probabilities, their gradient `dP = dO V^T` and
//!   `dS = P (dP - delta)` once per tile, accumulating `dK` and `dV` in registers and adding
//!   each tile's `dQ = dS K` to an f32 accumulator with atomic additions; then the gradients
//!   rounded once from f32 to their inputs' dtype. Both passes recompute the scores tile by tile
//!   from the log-sum-exp, so neither holds T x T, and grouped heads are handled inside the
//!   kernels: nothing is expanded. Each query's visible keys are one interval (the mask kinds
//!   above all give one, apart from padding), so every block of queries sweeps only the key
//!   blocks some of its queries see, every block of keys only the query blocks that see some of
//!   its keys, and per-element masking runs only on tiles that cut an interval or carry
//!   padding: tiles wholly outside the causal triangle, the window or a sequence cost nothing.
//!   A padding mask is applied per element. Tiles go through the tensor cores (`mma.sync`
//!   m16n8k16, f32 accumulation, operands staged through swizzled shared memory with
//!   `ldmatrix`). When the key blocks would fill the multiprocessors unevenly, two cubes share
//!   each key block, each sweeping half its query blocks, and add their `dK` and `dV` to f32
//!   accumulators too.
//!
//!   The query gradient's sum over the key blocks comes in whatever order the cubes run, so its
//!   last bits vary from run to run: by a unit in the dtype's last place where it is large, and
//!   where its terms cancel by the f32 sum's reordering, a few millionths of its largest value.
//!   The output, the log-sum-exp and the key and value gradients are the same bits from run to
//!   run (split, each key and value gradient is two addends added to zero, in either order the
//!   same). PyTorch's and flash-attn's flash backward sum the query gradient the same way. A
//!   fixed-order query gradient, a kernel of its own sweeping the key blocks of each query block
//!   (seven products in all), costs 28% more backward time on an RTX 5070 Ti and is not kept.
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
//! backward, the inputs the kernels copied and the backward passes whose products split.
//!
//! # Measured
//!
//! Against PyTorch 2.13's `scaled_dot_product_attention` in float64 with an explicit boolean
//! mask (the tests' records: every mask above, grouped heads, `Tq != Tk`, queries that see no
//! key; packed sequences one call per sequence), the reference in f32 agrees to within 4.1e-7 of
//! each output's and gradient's largest value, and the kernels to within 4.2e-3 in bf16 and
//! 7.1e-4 in f16. Against the f32 reference at shapes of several tiles and sequences, the bf16
//! kernels agree to within 4.6e-3, but for the query gradient's 7.3e-3 at the queries that see
//! few keys (the start of a causal sequence), whose `delta = rowsum(dO * O)` comes from the
//! output rounded to bf16, as in FlashAttention-2; f16 to within 7.0e-4. These are the largest
//! over repeated runs: where the query gradient's atomic sums change its bits, by up to 5.2e-4
//! of its largest value from one run to the next, its error moves less. A causal sequence run in
//! two segments, the second's queries against both segments' keys, composes into the whole
//! sequence's outputs and gradients to f32 rounding (7.6e-7).
//!
//! On an RTX 5070 Ti with 6144 tokens, 8 query and 2 key-value heads of 256 in bf16
//! (`examples/attention_bench.rs`, PyTorch 2.13 beside it, the two run in turn on a card other
//! work shared), forward and backward kernels take 8.7 ms of device time causal (flash attention
//! 8.9 ms): the forward 2.8 ms (2.4), the backward 5.9 ms (6.5, with grouped heads expanded and
//! summed), of which the products 5.6 ms. They grow the memory pool by 60 MiB (flash 204 MiB at
//! its peak), and the backward holds 72 MiB more meanwhile, its f32 accumulators. With no mask
//! they take 17.6 ms (17.8 ms); four packed sequences of 1536, 2.6 ms (2.8 ms as a padded batch
//! of four); causal within a window of 1024, 3.0 ms (PyTorch has no fused window at this head
//! width: flex_attention 7.3 ms, an explicit mask 49.4 ms of wall clock). Decoding
//! a batch of 8 against 6144 cached keys takes 147 us a layer (140 us), 52 us with cache lengths
//! from 100 to 6144 (3729 us through a padding mask) and 42 us within a window of 1024 (37 us on
//! the last 1025 keys).

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
    /// 0: the backward's products split by the call's size; 1: never; 2: always.
    static SPLIT: AtomicU8 = AtomicU8::new(0);

    /// Launches of this module's CubeCL kernels: the forward adds one, the backward three
    /// (`delta`, the products, the gradients' rounding from their accumulators), decoding two
    /// (the splits, then their merge). Zero without a CubeCL runtime's feature.
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

/// A sliding window around each query's position (after the bottom-right alignment): the query
/// sees no key more than `left` positions before it or `right` after it, as flash-attn's
/// `window_size = (left, right)`. `usize::MAX` leaves that side unbounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Window {
    /// How many keys before the query's position it sees, besides the one at it.
    pub left: usize,
    /// How many keys after the query's position it sees; with `causal`, none whatever this says.
    pub right: usize,
}

/// The arguments of attention that have a usual default.
///
/// The default is attention over every key, with no causality and no window, scaled by
/// `1 / sqrt(D)`: what PyTorch's `scaled_dot_product_attention` and flash-attn do by default,
/// and what encoders and cross-attention use. [`AttentionOptions::causal`] gives the decoder's.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AttentionOptions {
    /// Whether a query sees no key after its own position. Default: `false`.
    pub causal: bool,
    /// The factor on the scores; `None` (the default) for `1 / sqrt(D)`.
    pub scale: Option<f32>,
    /// A sliding window, or `None` (the default) for none.
    pub window: Option<Window>,
}

impl AttentionOptions {
    /// Causal attention, otherwise the defaults.
    pub fn causal() -> Self {
        Self {
            causal: true,
            ..Self::default()
        }
    }

    /// The options with causality on or off.
    pub fn with_causal(self, causal: bool) -> Self {
        Self { causal, ..self }
    }

    /// The options with the factor on the scores.
    pub fn with_scale(self, scale: f32) -> Self {
        Self {
            scale: Some(scale),
            ..self
        }
    }

    /// The options with a sliding window of `left` keys before each query's position and
    /// `right` after it.
    pub fn with_window(self, left: usize, right: usize) -> Self {
        Self {
            window: Some(Window { left, right }),
            ..self
        }
    }

    /// The factor on the scores for a head dimension of `d`.
    pub fn scale_for(&self, d: usize) -> f32 {
        self.scale.unwrap_or((1.0 / (d as f64).sqrt()) as f32)
    }

    /// How far a query sees before and after its position, `usize::MAX` for no bound: the
    /// window, and causality's bound after it.
    pub(crate) fn reach(&self) -> (usize, usize) {
        let (left, right) = self
            .window
            .map_or((usize::MAX, usize::MAX), |w| (w.left, w.right));
        (left, if self.causal { 0 } else { right })
    }
}

/// Which keys each query may see, beside the causality and the window of [`AttentionOptions`];
/// the [module documentation](self) gives the definitions.
// A `Tensor` is large (it holds any backend's handle), and a mask is built for a call and
// consumed by it: boxing the variants would only add an allocation.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Default)]
pub enum AttentionMask {
    /// Every key.
    #[default]
    None,
    /// `[B, Tk]`, `true` where a key is left out.
    KeyPadding(Tensor<2, Bool>),
    /// Packed variable-length sequences, `B = 1`: the cumulative offsets of the queries and of
    /// the keys, `[S + 1]` each, sequence `s` holding queries `cu_seqlens_q[s]..cu_seqlens_q[s + 1]`
    /// and keys `cu_seqlens_k[s]..cu_seqlens_k[s + 1]`.
    Packed {
        /// The queries' offsets, `[S + 1]`.
        cu_seqlens_q: Tensor<1, Int>,
        /// The keys' offsets, `[S + 1]`.
        cu_seqlens_k: Tensor<1, Int>,
    },
}

impl AttentionMask {
    /// The mask as the backend extension takes it.
    fn into_primitive(self) -> AttentionMaskPrimitive<Dispatch> {
        match self {
            AttentionMask::None => AttentionMaskPrimitive::None,
            AttentionMask::KeyPadding(mask) => {
                AttentionMaskPrimitive::KeyPadding(mask.into_dispatch())
            }
            AttentionMask::Packed {
                cu_seqlens_q,
                cu_seqlens_k,
            } => AttentionMaskPrimitive::Packed {
                cu_seqlens_q: cu_seqlens_q.into_dispatch(),
                cu_seqlens_k: cu_seqlens_k.into_dispatch(),
            },
        }
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

/// Attention and its backward pass, with the masks of the [module documentation](self), and
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
/// the mask, as the module documentation gives them.
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
/// keys each query sees given by `mask` and `options` (module documentation); returns
/// `[B, Tq, Hq, D]` in the dtype of `query`. On an autodiff device, the backward pass is
/// [`attention_backward`].
///
/// # Panics
///
/// If the shapes do not fit together (module documentation).
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
    let mask = mask.into_primitive();
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
    let mask = mask.into_primitive();
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
