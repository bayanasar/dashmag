//! The recurrent gated delta rule: one token at a time, for decoding.
//!
//! `gated_delta_step` advances the recurrent state of a Gated DeltaNet layer's sequences by a
//! token each, as one iteration of transformers' `torch_recurrent_gated_delta_rule` (and fla's
//! `fused_recurrent_gated_delta_rule`) does; or, over packed sequences, by each sequence's tokens
//! in turn. For each sequence and head, with `S` the `[Dk, Dv]` state and a token's `q`, `k`, `v`,
//! `g`, `beta`:
//!
//! ```text
//! q, k  <- q / sqrt(sum q^2 + 1e-6), k / sqrt(sum k^2 + 1e-6)   (with the l2 normalisation only)
//! q     <- scale q                                               (scale 1 / sqrt(Dk) by default)
//! S     <- exp(g) S
//! delta <- beta (v - S^T k)
//! S     <- S + k delta^T
//! out   <- S^T q
//! ```
//!
//! Shapes follow [`crate::gated_delta`]: `q`, `k` `[B, H, Dk]`, `v` `[B, H, Dv]`, `g` and
//! `beta` `[B, H]`, the state `[B, H, Dk, Dv]` in f32: batch row `b` is one token of sequence
//! `b`. With `cu_seqlens` `[S + 1]` the rows are instead `T` packed tokens of `S` sequences
//! (sequence `s` the rows `cu_seqlens[s]..cu_seqlens[s + 1]`, in order, possibly none), the state
//! `[S, H, Dk, Dv]`: a decoder serving sequences that bring different numbers of tokens to one
//! step (speculative tokens to verify, a prompt's tail) advances them all in one call, and one
//! token per sequence (`cu_seqlens = [0, 1, ..., S]`) is the batched step. Queries, keys and
//! values share one float dtype, which the output takes, as transformers casts its output back to
//! the queries' dtype; `g` and `beta` may each be in that dtype or in f32 (models compute `g` in
//! f32 and `beta` in their own dtype), and are read as f32 whatever their dtype. Everything is
//! computed in f32, the l2 normalisation included: on bf16 inputs this is what fla's kernels do,
//! while transformers' torch path normalises in the input dtype, so the two differ there by the
//! rounding of the normalised `q` and `k` to bf16.
//!
//! The step is forward only: decoding takes no gradient. An implementation that gives one where
//! an input requires it gives the gradients of the step as defined above, those torch's autograd
//! takes through `torch_recurrent_gated_delta_rule`.

/// The options of `gated_delta_step` that have a usual default; the defaults are fla's and
/// transformers': no normalisation, the scale `1 / sqrt(Dk)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GatedDeltaStepOptions {
    /// Normalise queries and keys to unit length first, `x / sqrt(sum(x^2) + 1e-6)`
    /// (transformers' and fla's `use_qk_l2norm_in_kernel`).
    pub qk_l2norm: bool,
    /// The queries' scale, applied after the normalisation; `None` for `1 / sqrt(Dk)` (fla's
    /// `scale=None`).
    pub scale: Option<f32>,
}

impl GatedDeltaStepOptions {
    /// The options with the l2 normalisation of queries and keys on or off.
    pub fn with_qk_l2norm(self, qk_l2norm: bool) -> Self {
        Self { qk_l2norm, ..self }
    }

    /// The options with the query scale `scale`.
    pub fn with_scale(self, scale: f32) -> Self {
        Self {
            scale: Some(scale),
            ..self
        }
    }
}

impl From<super::chunk::ChunkGatedDeltaOptions> for GatedDeltaStepOptions {
    /// The step that continues a chunked pass with these options.
    fn from(options: super::chunk::ChunkGatedDeltaOptions) -> Self {
        Self {
            qk_l2norm: options.qk_l2norm,
            scale: options.scale,
        }
    }
}
