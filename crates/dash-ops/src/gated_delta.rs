//! The gated delta rule of Gated DeltaNet: chunked forward and backward for training
//! ([`chunk`]), and a recurrent step for decoding ([`recurrent`]).
//!
//! [`dash_contract::gated_delta`] defines the rule, which every implementation of it shares: the
//! recurrence and what it matches (transformers' torch functions, fla's kernels), the layout the
//! two modules share so that a decoder continues from a chunked pass's states, packed
//! variable-length sequences, and the l2 normalisation and scale of the queries and keys. This
//! module implements it on Burn.

pub use crate::sequences::Sequences;
pub(crate) use crate::sequences::read_cu_seqlens;

pub mod chunk;
pub mod recurrent;

/// The query scale an option asks for: `scale`, or `1 / sqrt(Dk)` without one.
pub(crate) fn query_scale(scale: Option<f32>, dk: usize) -> f32 {
    scale.unwrap_or_else(|| (1.0 / (dk as f64).sqrt()) as f32)
}
