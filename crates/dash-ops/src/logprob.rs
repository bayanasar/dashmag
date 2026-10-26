//! The log-probability of chosen tokens under a large vocabulary, computed in chunks so that
//! the n x V logits are never held, with the entropy beside it.
//!
//! For rows `i` of `hidden` `[n, D]`, an output weight `[V, D]` (an output head, often tied to
//! the input embedding) and target ids `t[i]`:
//!
//! ```text
//! z[i, j]       = hidden[i] . weight[j]                        (the logits, never all held)
//! logprob[i]    = z[i, t[i]] - logsumexp_j z[i, j]
//! entropy[i]    = -sum_j p[i, j] log p[i, j],   p[i, j] = softmax_j z[i, j]
//! ```
//!
//! where `j` runs over the vocabulary minus the masked token ids: a masked token takes no part in
//! the log-sum-exp nor in the entropy, as if its logit were `-inf`. A target that is itself
//! masked (or outside `0..V`) has `logprob = -inf`.
//!
//! # How it is computed
//!
//! The vocabulary is walked in chunks of `C` tokens ([`LogprobOptions::vocab_chunk`]), each one
//! matmul `[n, D] x [D, C]`, with an online log-sum-exp in f32: a running maximum `m`, the sum
//! `s = sum_j exp(z - m)` rescaled whenever `m` grows, and beside them the entropy's
//! `a = sum_j exp(z - m) (z - m)` (`entropy = ln s - a / s`) and the target's logit. The rows are
//! walked in blocks of [`LogprobOptions::row_chunk`]. Only one `[rows, C]` chunk of logits lives
//! at a time.
//!
//! The gradient. Only `hidden` gets one (the weight is frozen, the entropy carries none), and
//! row `i` of it is the upstream gradient `g[i]`, a scalar, times
//!
//! ```text
//! d logprob[i] / d hidden[i] = weight[t[i]] - sum_j p[i, j] weight[j]
//!                            = (r[i] weight[t[i]] - o[i]) / s[i]
//! r[i] = sum_{j != t[i]} exp(z[i, j] - m[i]),   o[i] = sum_{j != t[i]} exp(z[i, j] - m[i]) weight[j]
//! ```
//!
//! which does not depend on `g`. So when the output is tracked, the same pass over the vocabulary
//! that computes the log-sum-exp also accumulates `r` and `o` (one more matmul per chunk,
//! `[rows, C] x [C, D]`, both rescaled like `s` as `m` grows), and the backward pass is `g[i]`
//! times a saved `[n, D]` f32 tensor: no second pass over the vocabulary, no logits recomputed.
//! Leaving the target's own term out of `r` and `o` keeps the sum free of cancellation when
//! `p[i, t[i]]` nears 1 and the gradient nears 0.
//!
//! # Backends
//!
//! The trait's default body ([`ChosenLogprobBackend`]) is the reference: the algorithm above in
//! Burn's tensor operations, in f32 throughout (bf16 inputs are widened chunk by chunk). On
//! the CUDA backend (`--features cuda`) the matmuls take bf16 operands straight to f32 results on
//! the tensor cores (each product of two bf16 numbers is exact in f32, so the logits are those of
//! an f32 matmul, up to summation order), and one CubeCL kernel per chunk does the masked online
//! log-sum-exp and writes the softmax weights for the gradient's matmul. Those weights are f32
//! values a bf16 matmul cannot take as they are, so they are split into a bf16 part and a bf16
//! remainder (16 significant bits in all) and both go through the matmul, which keeps the
//! gradient within about 1e-6 of an f32 computation at the cost of one more matmul.

mod reference;

#[cfg(feature = "cuda")]
mod cube;

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind, unary};
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, DType, Dispatch, FloatDType, Shape, TensorMetadata,
    backend_extension,
};
use burn::tensor::{Int, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};

/// How the vocabulary and the rows are chunked. The largest chunk of logits held at once is
/// `rows x vocab_chunk` f32 values, 96 MiB at most at the defaults; with the gradient, the CUDA
/// backend holds as much again for the softmax weights (in two bf16 parts) and three
/// `[rows, D]` f32 buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogprobOptions {
    /// Vocabulary tokens per chunk (`C`), or 0 (the default) to choose by the size of the row
    /// block: 4096 tokens below 2304 rows, 8192 from there. On an RTX 5070 Ti those are the faster
    /// of the two for Burn's default bf16 matmul kernel (at 1200 rows, forward and backward take
    /// 36.8 ms with 4096 and 44.8 ms with 8192; at 3072 rows, 80.7 ms and 78.9 ms).
    pub vocab_chunk: usize,
    /// The most rows per block, at least 1; the rows are split into equal blocks no larger.
    /// Every block makes its own pass over the vocabulary, reading the whole weight. The default,
    /// 3072, keeps every matmul under 4096 rows, from which the default kernel slows nine-fold
    /// on the gradient's `[rows, C] x [C, D]`.
    pub row_chunk: usize,
}

impl Default for LogprobOptions {
    fn default() -> Self {
        Self {
            vocab_chunk: 0,
            row_chunk: 3072,
        }
    }
}

impl LogprobOptions {
    /// The vocabulary chunk for a block of `rows` rows.
    pub fn vocab_chunk_for(&self, rows: usize) -> usize {
        match self.vocab_chunk {
            0 if rows < 2304 => 4096,
            0 => 8192,
            chunk => chunk,
        }
    }

    /// The row blocks of `n` rows: as few as `row_chunk` allows, of equal size but for the last.
    pub(crate) fn row_blocks(&self, n: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
        let block = n.div_ceil(n.div_ceil(self.row_chunk).max(1)).max(1);
        (0..n).step_by(block).map(move |r0| r0..(r0 + block).min(n))
    }
}

/// The chosen-token log-probability as a backend extension.
///
/// Inputs, for both operations:
///
/// - `hidden` `[n, D]`, bf16 or f32;
/// - `weight` `[V, D]`, bf16 or f32 (unless both are bf16, both are taken as f32);
/// - `targets` `[n]`, any int dtype;
/// - `masked`: token ids taken out of the softmax, in any order, each `< V`;
/// - `options`: the chunking.
///
/// Outputs are f32: `logprob` `[n]` and `entropy` `[n]`.
#[backend_extension(Flex, Cube: cfg(feature = "cuda"), Fusion: cfg(feature = "fusion"), Autodiff)]
pub trait ChosenLogprobBackend: Backend {
    /// `(logprob, entropy)`. On `Autodiff`, `logprob` carries the gradient for `hidden`.
    #[fusion(meta = fusion_meta::forward)]
    fn chosen_logprob(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        masked: Vec<u32>,
        options: LogprobOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let out = reference::forward::<Self>(hidden, weight, targets, &masked, options, false);
        (out.logprob, out.entropy)
    }

    /// `(logprob, entropy, grad)`, where `grad` `[n, D]` f32 holds row by row the gradient of
    /// `logprob[i]` with respect to `hidden[i]`: what the backward pass multiplies by the
    /// upstream gradient. Not tracked by autodiff itself.
    #[fusion(meta = fusion_meta::forward_with_grad)]
    fn chosen_logprob_with_grad(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        masked: Vec<u32>,
        options: LogprobOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let out = reference::forward::<Self>(hidden, weight, targets, &masked, options, true);
        (
            out.logprob,
            out.entropy,
            out.grad.expect("asked for the gradient"),
        )
    }
}

impl ChosenLogprobBackend for burn::backend::Flex {}

/// Output metadata for Burn's lazy fusion: every output is f32, `[n]` or `[n, D]`.
#[cfg(feature = "fusion")]
mod fusion_meta {
    use super::LogprobOptions;
    use burn::backend::fusion::custom::TensorSpec;
    use burn::backend::{DType, Shape};

    fn rows(hidden: &TensorSpec) -> TensorSpec {
        TensorSpec::new(Shape::new([hidden.shape[0]]), DType::F32)
    }

    pub(super) fn forward(
        hidden: &TensorSpec,
        _weight: &TensorSpec,
        _targets: &TensorSpec,
        _masked: &Vec<u32>,
        _options: &LogprobOptions,
    ) -> (TensorSpec, TensorSpec) {
        (rows(hidden), rows(hidden))
    }

    pub(super) fn forward_with_grad(
        hidden: &TensorSpec,
        _weight: &TensorSpec,
        _targets: &TensorSpec,
        _masked: &Vec<u32>,
        _options: &LogprobOptions,
    ) -> (TensorSpec, TensorSpec, TensorSpec) {
        let grad = TensorSpec::new(hidden.shape.clone(), DType::F32);
        (rows(hidden), rows(hidden), grad)
    }
}

/// Counters a test reads to tell which code ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use super::{AtomicUsize, Ordering};

    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);
    pub(super) static KERNEL_LAUNCHES: AtomicUsize = AtomicUsize::new(0);

    /// Runs of the written-out backward pass.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }

    /// Launches of the CUDA chunk kernel (zero without `--features cuda`).
    pub fn kernel_launches() -> usize {
        KERNEL_LAUNCHES.load(Ordering::SeqCst)
    }
}

impl<B: Backend + ChosenLogprobBackend, C: CheckpointStrategy> ChosenLogprobBackend
    for Autodiff<B, C>
{
    fn chosen_logprob(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        masked: Vec<u32>,
        options: LogprobOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        /// The backward pass: `grad_hidden[i] = g[i] * grad[i]`, cast to `hidden`'s dtype.
        #[derive(Debug)]
        struct ChosenLogprobBackward;

        impl<B: Backend> Backward<B, 1> for ChosenLogprobBackward {
            /// The `[n, D]` f32 gradient of each `logprob[i]` with respect to `hidden[i]`, and
            /// `hidden`'s dtype.
            type State = (FloatTensor<B>, DType);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                probe::BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
                let (grad, dtype) = ops.state;
                unary::<B, _>(ops.parents, ops.node, grads, |upstream| {
                    let n = upstream.shape()[0];
                    let upstream = match upstream.dtype() {
                        DType::F32 => upstream,
                        _ => B::float_cast(upstream, FloatDType::F32),
                    };
                    let upstream = B::float_reshape(upstream, Shape::new([n, 1]));
                    let grad = B::float_mul(grad, upstream);
                    match dtype {
                        DType::F32 => grad,
                        dtype => B::float_cast(grad, dtype.into()),
                    }
                });
            }
        }

        assert!(
            !weight.is_tracked(),
            "chosen_logprob takes the output weight as frozen and computes no gradient for it; \
             pass a weight that does not require one (detach it, or freeze its module)"
        );
        let (hidden, hidden_node) = hidden.into_parts();
        let weight = weight.into_primitive();
        let dtype = hidden.dtype();
        match ChosenLogprobBackward
            .prepare::<C>([hidden_node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (logprob, entropy, grad) =
                    B::chosen_logprob_with_grad(hidden, weight, targets, masked, options);
                (
                    prep.finish((grad, dtype), logprob),
                    <Self as AutodiffBackend>::from_inner(entropy),
                )
            }
            OpsKind::UnTracked(prep) => {
                let (logprob, entropy) =
                    B::chosen_logprob(hidden, weight, targets, masked, options);
                (
                    prep.finish(logprob),
                    <Self as AutodiffBackend>::from_inner(entropy),
                )
            }
        }
    }

    fn chosen_logprob_with_grad(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        masked: Vec<u32>,
        options: LogprobOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (logprob, entropy, grad) = B::chosen_logprob_with_grad(
            hidden.into_primitive(),
            weight.into_primitive(),
            targets,
            masked,
            options,
        );
        (
            <Self as AutodiffBackend>::from_inner(logprob),
            <Self as AutodiffBackend>::from_inner(entropy),
            <Self as AutodiffBackend>::from_inner(grad),
        )
    }
}

/// What [`chosen_logprob`] returns, both f32 `[n]`.
#[derive(Debug, Clone)]
pub struct ChosenLogprob {
    /// `log p(target)` per row; `-inf` where the target is masked. Carries the gradient for
    /// `hidden` on an autodiff device.
    pub logprob: Tensor<1>,
    /// The entropy of each row's distribution over the unmasked vocabulary. No gradient.
    pub entropy: Tensor<1>,
}

/// The log-probability of `targets` under `softmax(hidden @ weight^T)` with the `masked` token
/// ids removed, and the entropy of that distribution, without holding the n x V logits.
///
/// - `hidden` `[n, D]` (bf16 for the CUDA fast path, or f32); on an autodiff device it may
///   require a gradient, and gets one.
/// - `weight` `[V, D]`, the same dtype as `hidden` for the fast path; frozen: it must not
///   require a gradient (this panics if it does).
/// - `targets` `[n]` token ids.
/// - `masked`: token ids excluded from the softmax, in any order; each must be `< V`, and at
///   least one token must stay unmasked.
///
/// # Panics
///
/// On mismatched shapes, a masked id `>= V`, a zero `row_chunk`, or a weight that requires a
/// gradient.
pub fn chosen_logprob(
    hidden: Tensor<2>,
    weight: Tensor<2>,
    targets: Tensor<1, Int>,
    masked: &[u32],
    options: LogprobOptions,
) -> ChosenLogprob {
    let [n, d] = hidden.dims();
    let [v, dw] = weight.dims();
    let [nt] = targets.dims();
    assert_eq!(
        d, dw,
        "hidden [n, D] and weight [V, D] must share D: got {d} and {dw}"
    );
    assert_eq!(nt, n, "targets must have one id per row of hidden");
    assert!(
        options.row_chunk > 0,
        "row_chunk must be at least 1: {options:?}"
    );
    let mut masked = masked.to_vec();
    masked.sort_unstable();
    masked.dedup();
    if let Some(&last) = masked.last() {
        assert!(
            (last as usize) < v,
            "masked token id {last} is outside the vocabulary of {v}"
        );
        assert!(masked.len() < v, "every token of the vocabulary is masked");
    }
    let (logprob, entropy) = <Dispatch as ChosenLogprobBackend>::chosen_logprob(
        hidden.into_dispatch(),
        weight.into_dispatch(),
        targets.into_dispatch(),
        masked,
        options,
    );
    ChosenLogprob {
        logprob: Tensor::from_dispatch(logprob),
        entropy: Tensor::from_dispatch(entropy),
    }
}
