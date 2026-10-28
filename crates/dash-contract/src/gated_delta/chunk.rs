//! The chunked gated delta rule, forward and backward, for training.
//!
//! `chunk_gated_delta_rule` computes the gated delta rule (the [parent module](super) states it)
//! over sequences of any length, as transformers' `torch_chunk_gated_delta_rule` and fla's
//! `chunk_gated_delta_rule` do: in chunks of 64 tokens, within a chunk through the WY
//! representation of the products of its Householder-like updates, and the state carried from
//! chunk to chunk.
//!
//! - Inputs: queries and keys `[B, T, H, Dk]`, values `[B, T, H, Dv]` in one float dtype (f32,
//!   bf16 or f16); `g`, the log of the decay (`<= 0`), and `beta`, both `[B, T, H]`, each in that
//!   dtype or in f32 (models compute `g` in f32), read as f32; an optional initial state per
//!   sequence, f32.
//! - Sequences: every batch element one sequence, or packed sequences, `B = 1` with
//!   `cu_seqlens` `[S + 1]` (the [parent module](super) gives the layout); the states are then
//!   `[S, H, Dk, Dv]`. Each packed sequence is chunked on its own, as fla does: its chunks start at
//!   its first token, the last one shorter, so no chunk holds two sequences and every chunk's
//!   arithmetic is what a call on that sequence alone does.
//! - Outputs: `o` `[B, T, H, Dv]` in the queries' dtype and, when asked, each sequence's final
//!   state `[B, H, Dk, Dv]` (packed, `[S, H, Dk, Dv]`) in f32, from which [`super::recurrent`]
//!   continues decoding.
//! - Options ([`ChunkGatedDeltaOptions`]): the l2 normalisation of queries and keys, the query
//!   scale (`1 / sqrt(Dk)` by default), and whether to return the final states.
//! - Arithmetic accumulates in f32 whatever the inputs' dtype; the outputs and gradients are
//!   rounded once, to their dtype. From f32 inputs everything is computed in IEEE f32, as
//!   transformers does. From bf16 and f16 inputs an implementation may take the products'
//!   operands in the input dtype, as fla does, rounding to it the intermediates that only feed
//!   products. (transformers normalises bf16 queries and keys in bf16 before casting; here, as in
//!   fla, they are normalised in f32.)
//!
//! Gradients flow to the queries, keys, values, `g`, `beta` and the initial states, and from both
//! the output and the final states.

/// The options of `chunk_gated_delta_rule` that have a usual default; the defaults are fla's
/// and transformers': no normalisation, the scale `1 / sqrt(Dk)`, no final state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChunkGatedDeltaOptions {
    /// Normalise queries and keys to unit length first, `x / sqrt(sum(x^2) + 1e-6)`
    /// (transformers' and fla's `use_qk_l2norm_in_kernel`).
    pub qk_l2norm: bool,
    /// The queries' scale, applied after the normalisation; `None` for `1 / sqrt(Dk)` (fla's
    /// `scale=None`).
    pub scale: Option<f32>,
    /// Return each sequence's state after its last token (fla's `output_final_state`).
    pub output_final_state: bool,
}

impl ChunkGatedDeltaOptions {
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

    /// The options returning the final states, or not.
    pub fn with_final_state(self, output_final_state: bool) -> Self {
        Self {
            output_final_state,
            ..self
        }
    }
}
