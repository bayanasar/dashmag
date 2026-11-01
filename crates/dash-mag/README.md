# dash-mag

The one crate an application names: the gateway to dash, which runs on this machine, and to the
`mag-` crates, planned and not started, which will pool devices across machines. A model and an
algorithm are written once and run on every platform: on Burn and dash-ops (CUDA, Flex) on Linux,
and on MLX and dash-mlx on macOS, chosen at compile time.

Part of [Dashmag](https://github.com/bayanasar/dashmag). See the repository for the full picture.

> **Status:** pre-alpha. Both sides are written, and one training step of a small model with LoRA runs
> from one source on each (`tests/step.rs`). One application's trainer runs on it on MLX, with the
> numbers it had; its run on CUDA is to come.

## On MLX

`src/arm/mlx.rs` is the side on macOS, over dash-mlx. Building it is dash-mlx's: a prebuilt MLX through
`MLX_SYS_PREBUILT`, or MLX built from source. f32 results assume `MLX_ENABLE_TF32=0`.

MLX's GPU has no f64. An array operation that reads or makes f64 runs on the CPU, and the operations
with Metal kernels refuse it. MLX's `exp` in f64 has only f32's accuracy, so `exp`, `sigmoid`, `silu`
and `softplus` take it from the host there; `log` and `sqrt` are exact.

`Tensor::rms_norm` and `Rope` are MLX's own `fast` functions there, so that a model's numbers are the ones
mlx-lm's stack gives. They differ from the f32 form and the tables Burn's side runs in low bits, which is
what the gate's bounds cover. A read waits for its array alone: `eval_async` then pipelines, as a decoder
wants, and `contiguous` is called only on data that needs it.

## Speed on MLX

One training step of the gate's model (80 tokens, three blocks) with clipping and AdamW, against the
same model and step written directly on dash-mlx, `examples/step_bench.rs`. On an Apple M6 (32 GB), MLX
0.32.4 prebuilt, `MLX_ENABLE_TF32=0`, built in release: each process the median of 50 steps after 5 of
warm-up, twelve processes of each in turn over two batches, the median of those and their range, in ms:

| | through dash-mag | dash-mlx directly |
|---|---|---|
| f32, the step | 2.324 (2.240-2.644) | 2.314 (2.194-2.359) |
| building the graph | 0.261 (0.259-0.267) | 0.255 (0.252-0.257) |
| evaluating it | 1.250 (1.175-1.576) | 1.276 (1.153-1.366) |
| clipping, AdamW and evaluating them | 0.814 (0.802-0.840) | 0.770 (0.749-0.782) |
| bf16, the step | 2.453 (2.365-2.556) | 2.408 (2.315-2.636) |
| building the graph | 0.269 (0.265-0.273) | 0.261 (0.259-0.270) |
| evaluating it | 1.370 (1.274-1.449) | 1.347 (1.288-1.475) |
| clipping, AdamW and evaluating them | 0.810 (0.802-0.824) | 0.776 (0.757-0.790) |

The two batches apart gave 2.405 and 2.292 against 2.272 and 2.325 in f32, and 2.481 and 2.395 against
2.447 and 2.370 in bf16.

The gap is host time per operation. Both ways compute the same: the loss and the norm agree to the
printed digit. Evaluating shows no gap beyond its own spread. The gateway's checks and handles take an
add of one shape from 0.105 µs to 0.21 µs to build, and 0.275 µs beside a scalar; the update's 293
binary operations, 168 of them beside a scalar, come to 42 µs by that, where the gap is 34 in bf16 and 44 in f32.
Taking away the 168 `Broadcast` nodes the gateway adds for those scalars, or its `format!`, moved nothing
measurable.
