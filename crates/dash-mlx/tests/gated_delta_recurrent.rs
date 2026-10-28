//! The recurrent gated delta rule's step: `dash_conformance::gated_delta_recurrent`'s tests
//! through dash-mlx's adapter (`tests/common`), on the GPU, with dash-ops' bounds: single steps
//! in f32 (Dk 16 / Dv 24 with and without the l2 norm, and Qwen3.5-0.8B's head, Dk = Dv = 128);
//! a step on bf16 inputs; gradients through one step; and 16 steps after a 100-token prompt, from
//! the state transformers' chunked rule left. The adapter asserts that the step's kernel ran.
//!
//! What is dash-mlx's own stays here: its chunked rule over that prompt leaves transformers'
//! state.
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test gated_delta_recurrent -- --nocapture
#![cfg(target_os = "macos")]

mod common;

use common::{Mlx, require_ieee_f32_matmul};
use dash_conformance::gated_delta_chunk::F32_TOL;
use dash_conformance::gated_delta_recurrent::{
    Fixture, assert_close, check_bf16_step, check_f32_steps, check_gradients,
    check_prefill_then_decode,
};
use dash_conformance::{Adapter, ChunkGatedDelta};
use dash_mlx::gated_delta::chunk::ChunkGatedDeltaOptions;
use std::sync::{Mutex, MutexGuard};

/// One test at a time: they read the kernel counters, and share the GPU's default stream.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn f32_steps_match_torch() {
    let _serial = serial();
    require_ieee_f32_matmul();
    check_f32_steps(&Mlx).all_ran();
}

#[test]
fn decoding_after_a_prompt_matches_torch() {
    let _serial = serial();
    require_ieee_f32_matmul();
    check_prefill_then_decode(&Mlx).all_ran();
}

/// The chunked rule over the prompt, normalised, leaves the state transformers' left, from which
/// the decoding above starts, within the chunked rule's own f32 bound.
#[test]
fn the_chunked_rule_over_the_prompt_leaves_torchs_state() {
    let _serial = serial();
    require_ieee_f32_matmul();
    let fixture = Fixture::open("prefill_decode.safetensors");
    let t = |name: &str| fixture.data(&format!("prefill_decode.{name}"));
    let call = ChunkGatedDelta {
        q: t("prompt_q"),
        k: t("prompt_k"),
        v: t("prompt_v"),
        g: t("prompt_g"),
        beta: t("prompt_beta"),
        initial_state: None,
        cu_seqlens: None,
        options: ChunkGatedDeltaOptions::default()
            .with_qk_l2norm(true)
            .with_final_state(true),
        grad: None,
    };
    let prefill = Mlx.chunk_gated_delta_rule(&call).expect("the prompt");
    let state = prefill.final_state.expect("asked for");
    assert_close(
        "chunked prefill state",
        &state,
        &t("prefill_state"),
        F32_TOL,
    );
}

#[test]
fn bf16_inputs() {
    let _serial = serial();
    check_bf16_step(&Mlx).all_ran();
}

#[test]
fn gradients_through_a_step_match_torch() {
    let _serial = serial();
    require_ieee_f32_matmul();
    check_gradients(&Mlx).all_ran();
}
