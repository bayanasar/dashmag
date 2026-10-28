//! dash-ops' and dash-lora's operations on MLX, for Apple silicon.
//!
//! On NVIDIA, dashmag's training operations sit on Burn: dash-ops and dash-lora add what Burn
//! lacks, as CubeCL kernels. On Apple silicon the backend is MLX instead, which already has what
//! Burn gives there: arrays, autodiff, matmul, a fused attention forward, rms_norm, RoPE, Metal.
//! This crate adds the rest, with dash-ops' and dash-lora's contracts (arguments, shapes, dtypes,
//! states in and out), so that an application can target either backend, and holds each operation
//! to their test fixtures.
//!
//! - [`conv1d`]: causal depthwise conv1d with its states, over a batch or packed sequences, and
//!   its decoding step.
//! - [`gated_delta`]: the gated delta rule, chunked for training and recurrent for decoding.
//! - [`logprob`]: the chosen tokens' log-probabilities and entropies over a large vocabulary,
//!   without the n x V logits.
//!
//! The operations compute in f32 inside. MLX runs f32 matrix products in TF32 on GPUs that have
//! it unless `MLX_ENABLE_TF32=0`; the f32 bounds the tests hold the operations to assume it is off.
//!
//! The operations build on three MLX functions: [`mlx_rs::transforms::checkpoint`] and
//! [`mlx_rs::transforms::custom_vjp`], from the workspace's mlx-rs (bayanasar/mlx-rs's `dashmag`
//! branch, which merges the pull request that adds them into upstream main), and
//! [`mlx_rs::fast::MetalKernel`], which mlx-rs 0.33.0 wraps. `tests/hooks.rs` holds each of them
//! to plain MLX operations.
//!
//! MLX runs on Apple silicon only; on other targets this crate is empty.

// Off macOS the crate is empty and mlx-rs is not a dependency, so the links above into mlx-rs
// and to this crate's modules do not resolve there.
#![cfg_attr(not(target_os = "macos"), allow(rustdoc::broken_intra_doc_links))]

#[cfg(target_os = "macos")]
pub mod conv1d;
#[cfg(target_os = "macos")]
pub mod gated_delta;
#[cfg(target_os = "macos")]
mod kernel;
#[cfg(target_os = "macos")]
pub mod logprob;
