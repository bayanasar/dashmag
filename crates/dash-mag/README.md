# dash-mag

The one crate an application names: the gateway to dash, which runs on this machine, and to the
`mag-` crates, planned and not started, which will pool devices across machines. A model and an
algorithm are written once and run on every platform: on Burn and dash-ops (CUDA, ROCm, Flex) on Linux,
and on MLX and dash-mlx on macOS, chosen at compile time.

Part of [Dashmag](https://github.com/bayanasar/dashmag). See the repository for the full picture.

> **Status:** pre-alpha. Both sides are written, and one training step of a small model with LoRA runs
> from one source on each (`tests/step.rs`). No application has moved onto it yet.

## On CUDA

`--features cuda`. f32 results assume IEEE f32 products. cubecl-cuda as published runs f32 matmuls and
convolutions in TF32 on compute capability 8.0 and later, each input rounded to 10 mantissa bits, and
Burn has no setting for that yet. This repository patches it, but Cargo applies a `[patch]` only in the
workspace that writes it, so an application repeats it, at the revision of dash-mag it pins:

```toml
[patch.crates-io]
cubecl-cuda = { git = "https://github.com/bayanasar/dashmag", rev = "<that revision>" }
```

and runs with `CUBECL_CUDA_TF32=0`. A device that would still run them in TF32, because the patch is
missing or the variable unset, refuses the first tensor, with an error that says this.
`CUBECL_CUDA_TF32=1` accepts TF32. The check asks the device what the matmul kernels ask it, so a card
before 8.0 needs neither.

The numbers in `PROGRESS.md` are from this workspace's `Cargo.lock`. A new application resolves the rest
of CubeCL to its latest patch release beside the patched cubecl-cuda 0.11.1.

## On ROCm

`--features rocm`, built against ROCm 6.2 to 7.2 (found through `ROCM_PATH` or `HIP_PATH`). Under WSL2 the
runtime finds the card only with `HSA_ENABLE_DXG_DETECTION=1`. The same TF32 check runs; no patch is
needed. dash-ops' tests pass on a Radeon AI PRO R9700 (gfx1201) with ROCm 7.2.1; its speed there has not
been measured (`PROGRESS.md`).

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
mlx-lm's stack gives. `rms_norm` comes out in the dtype `x` and its weight promote to, on every side. They differ from the f32 form and the tables Burn's side runs in low bits, which is
what the gate's bounds cover. A read waits for its array alone: `eval_async` then pipelines, as a decoder
wants, and `contiguous` is called only on data that needs it.

## Speed on MLX

One training step of the gate's model (80 tokens, three blocks) with clipping and AdamW, against the
same model and step written directly on dash-mlx, `examples/step_bench.rs`. On an Apple M6 (32 GB), MLX
0.32.4 prebuilt, `MLX_ENABLE_TF32=0`, built in release: each process the median of 50 steps after 5 of
warm-up, twelve processes of each in turn over two batches, the median of those and their range, in ms:

| | through dash-mag | dash-mlx directly |
|---|---|---|
| f32, the step | 2.370 (2.248-3.627) | 2.254 (2.183-2.375) |
| building the graph | 0.261 (0.258-0.269) | 0.253 (0.250-0.256) |
| evaluating it | 1.328 (1.181-2.470) | 1.252 (1.151-1.368) |
| clipping, AdamW and evaluating them | 0.810 (0.797-0.880) | 0.768 (0.754-0.783) |
| bf16, the step | 2.452 (2.347-2.556) | 2.367 (2.334-2.487) |
| building the graph | 0.269 (0.266-0.275) | 0.261 (0.256-0.265) |
| evaluating it | 1.355 (1.269-1.440) | 1.357 (1.285-1.455) |
| clipping, AdamW and evaluating them | 0.810 (0.800-0.823) | 0.761 (0.748-0.775) |

Taken on 2026-11-20 UTC, the direct side on a local projection that computes what dash-mlx's removed
LoRA layer computed. A second run of eight processes each gave 2.318 against 2.256 in f32, and 2.436
against 2.376 in bf16. The M6 was not strictly idle: some applications were open, the load average was
1.30 to 1.48, and when sampled no other process used more than 5.8% of a core. f32's slowest gateway
process, 3.627 ms, was the first of all 48, its evaluation 2.470 ms against 1.181 to 1.372 for the other
eleven. The record of 2026-11-01, through that layer under a load of 2.1 to 2.8, gave 2.345 against
2.308 in f32 and 2.554 against 2.402 in bf16.

Both ways compute the same: in all 80 processes the first step's loss and norm are the same, gateway
and direct alike. The update's gap is host time per operation. The gateway's checks and handles take an
add of one shape from 0.105 µs to 0.21 µs to build, and 0.275 µs beside a scalar (2026-11-01); the
update's 293 binary operations, 168 of them beside a scalar, come to 42 µs by that, where the table's gap
is 42 µs in f32 and 49 µs in bf16 (41 and 34 in the second run). Taking away the 168 `Broadcast` nodes
the gateway adds for those scalars, or its `format!`, moved nothing measurable. Evaluating shows no gap
beyond its own spread in bf16 (1.355 against 1.357 ms).
