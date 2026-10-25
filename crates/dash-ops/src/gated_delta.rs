//! The gated delta rule of Gated DeltaNet: chunked forward and backward for training
//! ([`chunk`]), and a recurrent single-token step for decoding ([`recurrent`]).
//!
//! Both follow transformers' `torch_chunk_gated_delta_rule` and
//! `torch_recurrent_gated_delta_rule` and share one layout, so that a decoder can continue from
//! the state a chunked pass over a prompt leaves:
//!
//! - queries and keys `[B, T, H, Dk]`, values `[B, T, H, Dv]` (one token: `[B, H, Dk]`,
//!   `[B, H, Dv]`);
//! - the gate `g`, the log of the decay, `<= 0`, and `beta`, both `[B, T, H]` (one token:
//!   `[B, H]`);
//! - the recurrent state `[B, H, Dk, Dv]`, always f32.

pub mod chunk;
pub mod recurrent;
