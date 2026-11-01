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

MLX's GPU has no f64. An array operation that reads or makes f64 runs on the CPU. The operations in
`ops` take dash-contract's dtypes, f32, bf16 and f16, and the gateway refuses f64 for them on every
side. MLX's `exp` in f64 has only f32's accuracy, so `exp`, `sigmoid`, `silu`
and `softplus` take it from the host there; `log` and `sqrt` are exact.

## Speed on MLX

One training step of the gate's model (80 tokens, three blocks) with clipping and AdamW, against the
same model and step written directly on dash-mlx, `examples/step_bench.rs`. On an Apple M6 (32 GB), MLX
0.32.4 prebuilt, `MLX_ENABLE_TF32=0`, built in release: each process the median of 50 steps after 5 of
warm-up, twelve processes of each in turn over two batches, the median of those and their range, in ms:

| | through dash-mag | dash-mlx directly |
|---|---|---|
| f32, the step | 2.367 (2.313-2.775) | 2.350 (2.293-2.529) |
| building the graph | 0.276 (0.274-0.280) | 0.264 (0.262-0.267) |
| evaluating it | 1.265 (1.227-1.658) | 1.353 (1.252-1.549) |
| clipping, AdamW and evaluating them | 0.815 (0.800-0.835) | 0.762 (0.745-0.778) |
| bf16, the step | 2.555 (2.458-2.720) | 2.443 (2.384-2.552) |
| building the graph | 0.283 (0.280-0.286) | 0.273 (0.271-0.278) |
| evaluating it | 1.488 (1.360-1.658) | 1.387 (1.348-1.481) |
| clipping, AdamW and evaluating them | 0.814 (0.792-0.827) | 0.772 (0.763-0.785) |

A second run of eight processes each gave 2.393 against 2.306 in f32, and 2.489 against 2.492 in bf16.

The gap is host time per operation. Both ways compute the same: the loss and the norm agree to the
printed digit. Evaluating shows no gap beyond its own spread. The gateway's checks and handles take an
add of one shape from 0.105 µs to 0.21 µs to build, and 0.275 µs beside a scalar; the update's 293
binary operations, 168 of them beside a scalar, come to 42 µs by that, where the gap is 37 to 53 µs.
Taking away the 168 `Broadcast` nodes the gateway adds for those scalars, or its `format!`, moved nothing
measurable.
