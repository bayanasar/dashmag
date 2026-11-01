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

With a prebuilt MLX, an application's binaries find `libmlx.dylib` through an rpath only the application
can add. This crate publishes the directory to its build script, unset where there is nothing to add:

```rust,ignore
fn main() {
    println!("cargo:rerun-if-env-changed=DEP_DASH_MAG_LIB_DIR");
    if let Ok(dir) = std::env::var("DEP_DASH_MAG_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
}
```

MLX's GPU has no f64. An array operation that reads or makes f64 runs on the CPU. The operations in
`ops` take dash-contract's dtypes, f32, bf16 and f16, and the gateway refuses f64 for them on every
side. MLX's `exp` in f64 has only f32's accuracy, so `exp`, `sigmoid`, `silu`
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
| f32, the step | 2.345 (2.246-3.572) | 2.308 (2.211-2.618) |
| building the graph | 0.264 (0.258-0.266) | 0.256 (0.251-0.261) |
| evaluating it | 1.272 (1.176-2.459) | 1.289 (1.186-1.572) |
| clipping, AdamW and evaluating them | 0.816 (0.801-0.851) | 0.776 (0.766-0.811) |
| bf16, the step | 2.554 (2.397-2.748) | 2.402 (2.322-2.495) |
| building the graph | 0.271 (0.267-0.273) | 0.262 (0.259-0.266) |
| evaluating it | 1.443 (1.296-1.652) | 1.361 (1.301-1.448) |
| clipping, AdamW and evaluating them | 0.817 (0.807-0.849) | 0.778 (0.765-0.786) |

A second run of eight processes each gave 2.356 against 2.257 in f32, and 2.458 against 2.381 in bf16.
The M6 was not strictly idle: other applications were open, the load average was 2.1 to 2.8, and when
sampled no other process used more than 6.4% of a core.

Both ways compute the same: the loss and the norm agree to the printed digit. The update's gap is host
time per operation. The gateway's checks and handles take an add of one shape from 0.105 µs to 0.21 µs
to build, and 0.275 µs beside a scalar; the update's 293 binary operations, 168 of them beside a scalar,
come to 42 µs by that, where the table's gap is 39 µs in bf16 and 40 µs in f32. Taking away the 168
`Broadcast` nodes the gateway adds for those scalars, or its `format!`, moved nothing measurable.
Evaluating shows no gap beyond its own spread: 82 µs of the bf16 step's 152 lie there, within ranges
that overlap, and 11 µs of 77 in the second run.
