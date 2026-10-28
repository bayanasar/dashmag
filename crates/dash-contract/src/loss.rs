//! Cross-entropy over a large vocabulary straight from the hidden states and the output weight,
//! without the n x V logits: `linear_cross_entropy` is torch's
//! `F.cross_entropy(F.linear(hidden, weight), targets)` with its reductions, ignore index and
//! label smoothing, plus a z-loss and masked vocabulary ids, returning beside the loss each row's
//! log-probability of its target and, if asked, its entropy. Gradients flow to `hidden` and,
//! when it requires one, to `weight`: an output head trains through it.
//!
//! # Definition
//!
//! For rows `i` of `hidden` `[n, D]`, an output weight `[V, D]` and targets `t = t[i]`, with `U`
//! the vocabulary less the masked ids and `K = |U|` its size:
//!
//! ```text
//! z[i, j]    = hidden[i] . weight[j]                          (the logits, never all held)
//! lse[i]     = log sum_{j in U} exp z[i, j]
//! logprob[i] = z[i, t] - lse[i]
//! entropy[i] = -sum_{j in U} p[i, j] log p[i, j],     p[i, j] = exp(z[i, j] - lse[i])
//! loss[i]    = (1 - eps) (lse[i] - z[i, t]) + eps (lse[i] - mean_{j in U} z[i, j]) + lambda lse[i]^2
//! ```
//!
//! - `eps` is [`CrossEntropyOptions::label_smoothing`], torch's: the target distribution is
//!   `1 - eps` on the target and `eps / K` on every class, so its second term is
//!   `eps / K sum_{j in U} -log p[i, j]`. A masked id is not a class: `K` counts the unmasked ids
//!   (torch's is the width of its logits, which has no masked ids).
//! - `lambda` is [`CrossEntropyOptions::z_loss`]: PaLM's auxiliary loss on the softmax's
//!   normaliser, `lambda log(Z)^2` with `log Z = lse` (PaLM uses `1e-4`); fused linear
//!   cross-entropy kernels call it the lse-square scale.
//! - Masked ids take no part in any sum: their logits are as if `-inf`. A target that is masked,
//!   or outside `0..V` without being the ignore index, has probability zero: `logprob = -inf` and
//!   `loss = +inf`, which is what torch computes with the masked logits at `-inf`. Its gradient
//!   stays finite: the target's logit is then a constant.
//! - A row whose target is [`CrossEntropyOptions::ignore_index`] (torch's `-100` by default) is
//!   ignored: its loss, log-probability and entropy are 0, it receives no gradient, and the mean
//!   does not count it.
//!
//! [`Reduction`] reduces the rows' losses as torch's `reduction` does: the mean over the counted
//! rows (NaN when every row is ignored, as torch), their sum, or none (`[n]`). The
//! log-probability and the entropy are per row; each takes a gradient as the loss does.
//!
//! # Shapes and dtypes
//!
//! `hidden` `[n, D]` and `weight` `[V, D]` share one float dtype (f32, bf16 or f16), and their
//! gradients come back in it; `targets` `[n]` are of any int dtype. The loss (`[1]`, or `[n]`
//! unreduced), the log-probability `[n]` and the entropy `[n]` are f32. From bf16 inputs the
//! logits' gradient may be rounded to bf16 before it is multiplied by the weight and the hidden
//! states, as torch's bf16 linear layer rounds the gradient of its output. The vocabulary and the
//! rows are chunked as [`crate::logprob`]'s ([`CrossEntropyOptions::chunks`]).

use crate::logprob::LogprobOptions;

/// How `linear_cross_entropy` reduces the rows' losses, as torch's `reduction`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reduction {
    /// The mean over the rows that are not ignored, `[1]`.
    #[default]
    Mean,
    /// The sum over the rows, `[1]`.
    Sum,
    /// Each row's loss, `[n]` (0 for an ignored row).
    None,
}

/// The options of `linear_cross_entropy`. The defaults are torch's `F.cross_entropy`'s: the
/// mean, ignore index `-100`, no label smoothing; and no z-loss, no masked id, no entropy.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossEntropyOptions {
    /// How the rows' losses are reduced.
    pub reduction: Reduction,
    /// The target id of rows to ignore.
    pub ignore_index: i64,
    /// `eps` in `0..=1`: torch's `label_smoothing`.
    pub label_smoothing: f32,
    /// `lambda >= 0`, the z-loss coefficient: `lambda lse^2` is added to each row's loss.
    pub z_loss: f32,
    /// Token ids taken out of the softmax, in any order, each `< V`.
    pub masked: Vec<u32>,
    /// Whether to return each row's entropy (with its gradient).
    pub entropy: bool,
    /// How the vocabulary and the rows are chunked.
    pub chunks: LogprobOptions,
}

impl Default for CrossEntropyOptions {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            ignore_index: -100,
            label_smoothing: 0.0,
            z_loss: 0.0,
            masked: Vec::new(),
            entropy: false,
            chunks: LogprobOptions::default(),
        }
    }
}
