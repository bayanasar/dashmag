//! The trait each side implements. The gateway ([`crate::Tensor`]) normalises axes and shapes,
//! broadcasts and promotes dtypes before it calls here, so a side sees only what the comments say:
//! non-negative axes, one shape, one dtype.

use crate::ops::{
    AttentionOptions, CausalConv1dOptions, CausalConv1dStepOptions, ChunkGatedDeltaOptions,
    GatedDeltaStepOptions, LogprobOptions, LowRankOptions, SampleOptions,
};
use crate::{DType, Result};

mod burn;
pub(crate) use self::burn::Burn as Selected;

/// Operations of two arrays of one shape and one floating dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
}

/// Comparisons of two arrays of one shape and one dtype, floating or integer, to booleans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Compare {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
}

/// Operations of two boolean arrays of one shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Logical {
    And,
    Or,
}

/// Elementwise operations of a floating array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unary {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Square,
    Sigmoid,
    Silu,
    Softplus,
}

/// Reductions of a floating array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reduce {
    Sum,
    Mean,
    Max,
    Min,
}

/// What the attention operations take for a mask, in the side's arrays.
pub(crate) enum Mask<A> {
    None,
    KeyPadding(A),
    Packed { cu_seqlens_q: A, cu_seqlens_k: A },
}

/// What the sampler returns, in the side's arrays.
pub(crate) struct Drawn<A> {
    pub tokens: A,
    pub logprob: A,
    pub tempered_logprob: A,
    pub sampling_logprob: A,
}

/// What a vector-Jacobian product returns: the outputs, and each primal's gradient.
pub(crate) type VjpOut<A> = (Vec<A>, Vec<A>);

/// The function a vector-Jacobian product runs.
pub(crate) type VjpFn<'a, A> = &'a mut dyn FnMut(&[A]) -> Result<Vec<A>>;

pub(crate) trait Arm {
    type Array: Clone;

    fn describe() -> String;
    fn clear_cache();

    /// An array from little-endian bytes of a floating `dtype`, or from `i32`s, or from booleans.
    fn from_bytes(bytes: &[u8], shape: &[usize], dtype: DType) -> Result<Self::Array>;
    fn from_i32(values: &[i32], shape: &[usize]) -> Result<Self::Array>;
    fn from_bool(values: &[bool], shape: &[usize]) -> Result<Self::Array>;
    fn full(shape: &[usize], value: f64, dtype: DType) -> Result<Self::Array>;

    fn shape(a: &Self::Array) -> Vec<usize>;
    fn dtype(a: &Self::Array) -> DType;
    /// The little-endian bytes of a floating array, in its dtype.
    fn to_bytes(a: &Self::Array) -> Result<Vec<u8>>;
    fn to_i32(a: &Self::Array) -> Result<Vec<i32>>;
    fn to_bool(a: &Self::Array) -> Result<Vec<bool>>;
    /// Asks for these arrays to be computed now, where the side computes lazily.
    fn eval(arrays: &[&Self::Array]) -> Result<()>;

    /// To any dtype of any kind, an integer to the width asked for.
    fn cast(a: &Self::Array, dtype: DType) -> Result<Self::Array>;
    fn stop_gradient(a: &Self::Array) -> Result<Self::Array>;

    fn binary(op: Binary, a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn compare(op: Compare, a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn logical(op: Logical, a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn logical_not(a: &Self::Array) -> Result<Self::Array>;
    fn unary(op: Unary, a: &Self::Array) -> Result<Self::Array>;
    /// Both of rank 2 or more, with one batch shape and one floating dtype.
    fn matmul(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    /// `a` where `condition`, else `b`: one shape, `a` and `b` one dtype, of any kind.
    fn select(condition: &Self::Array, a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;

    /// A new shape of the same number of elements, of rank 0 to 6.
    fn reshape(a: &Self::Array, shape: &[usize]) -> Result<Self::Array>;
    /// `axes` a permutation of the array's axes.
    fn permute(a: &Self::Array, axes: &[usize]) -> Result<Self::Array>;
    /// `shape` of the array's rank or more, the array's axes aligned at the end.
    fn broadcast_to(a: &Self::Array, shape: &[usize]) -> Result<Self::Array>;
    /// One dtype; one shape but for `axis`.
    fn concat(arrays: &[&Self::Array], axis: usize) -> Result<Self::Array>;
    /// `indices` integer, of rank 1.
    fn take(a: &Self::Array, indices: &Self::Array, axis: usize) -> Result<Self::Array>;
    /// `start..end` of `axis`.
    fn slice(a: &Self::Array, axis: usize, start: usize, end: usize) -> Result<Self::Array>;

    /// Over the whole array, to rank 0, or over `axis`, kept with a length of 1.
    fn reduce(op: Reduce, a: &Self::Array, axis: Option<usize>) -> Result<Self::Array>;
    /// `i32`, with `axis` kept with a length of 1.
    fn argmax(a: &Self::Array, axis: usize) -> Result<Self::Array>;
    fn sort(a: &Self::Array, axis: usize) -> Result<Self::Array>;

    fn vjp(
        f: VjpFn<'_, Self::Array>,
        primals: &[Self::Array],
        cotangents: &[Option<Self::Array>],
    ) -> Result<VjpOut<Self::Array>>;

    fn attention(
        query: &Self::Array,
        key: &Self::Array,
        value: &Self::Array,
        mask: Mask<&Self::Array>,
        options: AttentionOptions,
    ) -> Result<Self::Array>;
    fn attention_decode(
        query: &Self::Array,
        key: &Self::Array,
        value: &Self::Array,
        cache_lengths: Option<&Self::Array>,
        options: AttentionOptions,
    ) -> Result<Self::Array>;
    fn causal_conv1d(
        x: &Self::Array,
        weight: &Self::Array,
        bias: Option<&Self::Array>,
        initial_state: Option<&Self::Array>,
        cu_seqlens: Option<&Self::Array>,
        options: CausalConv1dOptions,
    ) -> Result<(Self::Array, Option<Self::Array>)>;
    fn causal_conv1d_update(
        x: &Self::Array,
        weight: &Self::Array,
        bias: Option<&Self::Array>,
        state: &Self::Array,
        options: CausalConv1dStepOptions,
    ) -> Result<(Self::Array, Self::Array)>;
    #[allow(clippy::too_many_arguments)]
    fn chunk_gated_delta_rule(
        q: &Self::Array,
        k: &Self::Array,
        v: &Self::Array,
        g: &Self::Array,
        beta: &Self::Array,
        initial_state: Option<&Self::Array>,
        cu_seqlens: Option<&Self::Array>,
        options: ChunkGatedDeltaOptions,
    ) -> Result<(Self::Array, Option<Self::Array>)>;
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_step(
        q: &Self::Array,
        k: &Self::Array,
        v: &Self::Array,
        g: &Self::Array,
        beta: &Self::Array,
        state: &Self::Array,
        cu_seqlens: Option<&Self::Array>,
        options: GatedDeltaStepOptions,
    ) -> Result<(Self::Array, Self::Array)>;
    /// The log-probabilities and the entropies, `[n]` each.
    fn chosen_logprob(
        hidden: &Self::Array,
        weight: &Self::Array,
        targets: &Self::Array,
        masked: &[u32],
        options: LogprobOptions,
    ) -> Result<(Self::Array, Self::Array)>;
    fn sample(logits: &Self::Array, options: &SampleOptions) -> Result<Drawn<Self::Array>>;
    /// `u` `[in, r]`, `v` `[r, out]`, `x` and `base` of rank 2 or more.
    fn low_rank(
        x: &Self::Array,
        u: &Self::Array,
        v: &Self::Array,
        base: Option<&Self::Array>,
        options: LowRankOptions,
    ) -> Result<Self::Array>;
}
