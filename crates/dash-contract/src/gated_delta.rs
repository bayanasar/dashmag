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

pub mod chunk;
pub mod recurrent;
