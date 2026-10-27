//! The gated delta rule of Gated DeltaNet: chunked forward and backward for training
//! ([`chunk`]), and a recurrent step for decoding ([`recurrent`]).
//!
//! For each sequence and head, with `S` the `[Dk, Dv]` state, the rule reads one token at a time:
//!
//! ```text
//! S_t = exp(g_t) S_{t-1} + beta_t k_t (v_t - exp(g_t) S_{t-1}^T k_t)^T
//! o_t = S_t^T (scale q_t)
//! ```
//!
//! as transformers' `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule` and
//! fla's `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` compute it. Both
//! modules share one layout, so that a decoder continues from the states a chunked pass over its
//! prompts leaves:
//!
//! - queries and keys `[B, T, H, Dk]`, values `[B, T, H, Dv]`; a step's `[B, H, Dk]`,
//!   `[B, H, Dv]`;
//! - the gate `g`, the log of the decay, `<= 0`, and `beta`, both `[B, T, H]`; a step's `[B, H]`;
//! - the recurrent state `[B, H, Dk, Dv]`, always f32, one per sequence.
//!
//! Packed variable-length sequences lie end to end along `T` with `B = 1`, their boundaries
//! given as cumulative offsets `cu_seqlens` `[S + 1]` (Int, any int dtype): sequence `s` is
//! tokens `cu_seqlens[s]..cu_seqlens[s + 1]`, the offsets go from 0 to `T` without decreasing (a
//! sequence may be empty), and each sequence has its own initial and final state, `[S, H, Dk,
//! Dv]`. No token of one sequence reaches another: packed, the rule gives each sequence what a
//! call on that sequence alone gives. A step takes packed tokens as `[T, H, D]`, the packed layout
//! without its batch axis of one, each sequence's tokens in turn. These are fla's conventions
//! (`cu_seqlens` `[N + 1]`, int64 there, with a batch of one; `initial_state` and the final state
//! `[N, H, K, V]` in f32), so that fla's packed tensors pass as they are. fla also takes more value
//! heads than key heads; this rule takes as many of each.
//!
//! Queries and keys may first be normalised to unit length, `x / sqrt(sum(x^2) + 1e-6)`
//! (transformers' and fla's `use_qk_l2norm_in_kernel`); the queries are then scaled, by
//! `1 / sqrt(Dk)` unless an option says otherwise (fla's `scale`).

use burn::backend::tensor::IntTensor;
use burn::backend::{Backend, ExtensionType};

pub mod chunk;
pub mod recurrent;

/// The tokens of a call as sequences, as the backend extensions take them (an `Option` of a
/// tensor does not cross Burn's dispatch boundary; this enum does).
#[derive(ExtensionType, Clone, Debug)]
#[extension_type(fusion: cfg(feature = "fusion"))]
pub enum Sequences<B: Backend> {
    /// Every batch element is one sequence (a step: every row).
    Batch,
    /// Packed sequences: the boundaries `cu_seqlens` `[S + 1]`, any int dtype, over the tokens
    /// of a batch of one (a step: over its rows).
    Packed(IntTensor<B>),
}

impl<B: Backend> Sequences<B> {
    /// The boundaries, if packed.
    pub fn into_option(self) -> Option<IntTensor<B>> {
        match self {
            Sequences::Batch => None,
            Sequences::Packed(cu_seqlens) => Some(cu_seqlens),
        }
    }
}

impl<B: Backend> From<Option<IntTensor<B>>> for Sequences<B> {
    fn from(cu_seqlens: Option<IntTensor<B>>) -> Self {
        match cu_seqlens {
            None => Sequences::Batch,
            Some(cu_seqlens) => Sequences::Packed(cu_seqlens),
        }
    }
}

/// The query scale an option asks for: `scale`, or `1 / sqrt(Dk)` without one.
pub(crate) fn query_scale(scale: Option<f32>, dk: usize) -> f32 {
    scale.unwrap_or_else(|| (1.0 / (dk as f64).sqrt()) as f32)
}

/// The boundaries of `sequences` packed sequences over `tokens` tokens, read to the host, as
/// the references take them.
///
/// # Panics
///
/// If there are not `sequences + 1` of them, or they do not go from 0 to `tokens` without
/// decreasing.
pub(crate) fn read_cu_seqlens<B: Backend>(
    cu_seqlens: IntTensor<B>,
    sequences: usize,
    tokens: usize,
) -> Vec<usize> {
    let data = burn::tensor::try_read_sync(B::int_into_data(cu_seqlens))
        .expect("cu_seqlens: a synchronous read")
        .expect("cu_seqlens: read to the host");
    let cu: Vec<i64> = data.iter::<i64>().collect();
    assert_eq!(
        cu.len(),
        sequences + 1,
        "cu_seqlens must hold S + 1 = {} offsets for the {sequences} states, got {}",
        sequences + 1,
        cu.len()
    );
    assert!(
        cu[0] == 0 && cu.windows(2).all(|w| w[0] <= w[1]) && cu[sequences] == tokens as i64,
        "cu_seqlens must go from 0 to T = {tokens} without decreasing, got {cu:?}"
    );
    cu.into_iter().map(|x| x as usize).collect()
}
