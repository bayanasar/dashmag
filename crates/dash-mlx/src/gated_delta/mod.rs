//! The gated delta rule of Gated DeltaNet: chunked forward and backward for training
//! ([`chunk`]) and a recurrent single-token step for decoding ([`recurrent`]), dash-ops'
//! `gated_delta` on MLX. A decoding step continues from the state a chunked pass leaves.
//!
//! Layouts as in dash-ops: queries and keys `[B, T, H, Dk]`, values `[B, T, H, Dv]`; the gate
//! `g`, the log of the decay, `<= 0`, and `beta`, both `[B, T, H]`; the recurrent state
//! `[B, H, Dk, Dv]`, always f32.

pub mod chunk;
pub mod recurrent;
mod scan;
