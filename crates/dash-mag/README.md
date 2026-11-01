# dash-mag

The one crate an application names: the gateway to dash, which runs on this machine, and to the
`mag-` crates, planned and not started, which will pool devices across machines. A model and an
algorithm are written once and run on every platform: on Burn and dash-ops (CUDA, Flex) on Linux,
and on MLX and dash-mlx on macOS, chosen at compile time.

Part of [Dashmag](https://github.com/bayanasar/dashmag). See the repository for the full picture.

> **Status:** pre-alpha. Both sides are written, and one training step of a small model with LoRA runs
> from one source on each (`tests/step.rs`). No application has moved onto it yet.

## On MLX

`src/arm/mlx.rs` is the side on macOS, over dash-mlx. Building it is dash-mlx's: a prebuilt MLX through
`MLX_SYS_PREBUILT`, or MLX built from source. f32 results assume `MLX_ENABLE_TF32=0`.

MLX's GPU has no f64. An array operation that reads or makes f64 runs on the CPU, and the operations
with Metal kernels refuse it. MLX's `exp` in f64 has only f32's accuracy, so `exp`, `sigmoid`, `silu`
and `softplus` take it from the host there; `log` and `sqrt` are exact.
