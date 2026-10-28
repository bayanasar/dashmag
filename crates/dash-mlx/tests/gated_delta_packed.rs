//! The gated delta rule over packed variable-length sequences, with a query scale of its own and
//! with heads 256 wide: `dash_conformance::gated_delta_packed`'s tests through dash-mlx's adapter
//! (`tests/common`), on the GPU, with dash-ops' bounds. The chunked rule and the step against fla
//! 0.5.2 over the packed inputs and transformers' torch functions in float64 called once per
//! sequence; packed calls against each sequence (or token) on its own, to the bit; the step's
//! gradients; the default scale. The adapter asserts which kernels ran.
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test gated_delta_packed -- --nocapture
#![cfg(target_os = "macos")]

mod common;

use common::{Mlx, require_ieee_f32_matmul};
use dash_conformance::gated_delta_packed::{
    self as packed, BF16_VS_F64, CHUNK_F32, Fixture, PACKED_VS_SEQUENCES,
};
use std::sync::{Mutex, MutexGuard};

/// One test at a time: they read the kernel counters, and share the GPU's default stream.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn packed_chunks_match_fla_and_transformers() {
    let _serial = serial();
    require_ieee_f32_matmul();
    packed::check_chunk_f32(&Mlx, "mlx").all_ran();
}

#[test]
fn packed_bf16_chunks() {
    let _serial = serial();
    packed::check_chunk_bf16(&Mlx, "mlx-bf16", BF16_VS_F64).all_ran();
}

#[test]
fn packed_chunks_are_per_sequence_calls() {
    let _serial = serial();
    for name in CHUNK_F32 {
        let worst = packed::packed_against_sequences(&Mlx, &Fixture::load(name), "mlx").all_ran();
        assert!(worst <= PACKED_VS_SEQUENCES, "{name}: {worst:.3e}");
    }
}

#[test]
fn packed_steps_match_fla_and_transformers() {
    let _serial = serial();
    require_ieee_f32_matmul();
    packed::check_steps(&Mlx, "mlx").all_ran();
}

#[test]
fn packed_step_gradients_match_transformers() {
    let _serial = serial();
    require_ieee_f32_matmul();
    packed::check_step_gradients(&Mlx, "mlx").all_ran();
}

#[test]
fn the_default_scale_is_one_over_the_root_of_dk() {
    let _serial = serial();
    packed::the_default_scale_is_one_over_the_root_of_dk(&Mlx).all_ran();
}
