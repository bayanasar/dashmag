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
//! # Shapes and dtypes
//!
//! - `hidden` `[n, D]` and `weight` `[V, D]`, bf16 or f32: in bf16 where both are, else both
//!   taken as f32;
//! - `targets` `[n]`, any int dtype;
//! - `masked`: token ids taken out of the softmax, in any order, each `< V`, at least one token
//!   left unmasked;
//! - the outputs, f32: `logprob` `[n]` and `entropy` `[n]`.
//!
//! Only `hidden` gets a gradient, through `logprob`: the weight is frozen (a weight that requires
//! a gradient is refused), and the entropy carries none.
//!
//! The vocabulary is walked in chunks of `vocab_chunk` tokens and the rows in blocks of
//! `row_chunk` ([`LogprobOptions`]), one `[rows, vocab_chunk]` chunk of logits at a time, with an
//! online log-sum-exp in f32. The chunking sets what is held at once and the order of the sums,
//! not what is computed.

/// How the vocabulary and the rows are chunked. The largest chunk of logits held at once is
/// `rows x vocab_chunk` f32 values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogprobOptions {
    /// Vocabulary tokens per chunk (`C`), or 0 (the default) for the implementation's own choice,
    /// which may depend on the rows in a block and which each implementation documents: the chunk
    /// sets what is held at once and the order of the sums, and what is fastest differs between
    /// devices.
    pub vocab_chunk: usize,
    /// The most rows per block, at least 1; the rows are split into equal blocks no larger.
    /// Every block makes its own pass over the vocabulary, reading the whole weight. Default:
    /// 3072.
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

