//! dash-ops' and dash-lora's operations on MLX, for Apple silicon.
//!
//! On NVIDIA, dashmag's training operations sit on Burn: dash-ops and dash-lora add what Burn
//! lacks, as CubeCL kernels. On Apple silicon the backend is MLX instead, which already has what
//! Burn gives there: arrays, autodiff, matmul, a fused attention forward, rms_norm, RoPE, Metal.
//! This crate adds the rest, with dash-ops' and dash-lora's contracts (arguments, shapes, dtypes,
//! states in and out), so that an application can target either backend, and holds each operation
//! to their test fixtures.
//!
//! The operations build on three MLX functions. mlx-rs 0.33.0 wraps the third; the first two
//! come from the workspace's mlx-rs, bayanasar/mlx-rs's `dashmag` branch, which merges the pull
//! request that adds them (`feat/checkpoint-custom-vjp`) into upstream main:
//!
//! - [`mlx_rs::transforms::checkpoint`]: intermediate arrays computed again in the backward pass;
//! - [`mlx_rs::transforms::custom_vjp`]: a written-out backward pass;
//! - [`mlx_rs::fast::MetalKernel`]: a Metal kernel from source, compiled at run time.
//!
//! `tests/hooks.rs` holds each of them to plain MLX operations.
//!
//! MLX runs on Apple silicon only; on other targets this crate is empty.
