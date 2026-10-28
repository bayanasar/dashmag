# dash-mlx

dash-ops' and dash-lora's operations on MLX, for Apple silicon, with dash-contract's options and
held to dash-ops' and dash-lora's test fixtures, through dash-conformance's tests where they
exist. For now this crate is dashmag's path on Apple silicon: MLX already has arrays, autodiff,
matmul, a fused attention forward, rms_norm, RoPE and Metal, and this crate adds what dash-ops and
dash-lora add to Burn. dashmag's Burn side still targets Metal as well, through dash-ops' and
dash-lora's `metal` feature.

| module | operation | on Metal |
|---|---|---|
| `conv1d` | causal depthwise conv1d of any small width, with an optional bias and SiLU, from initial states or zeros and to final states, over a batch or packed variable-length sequences, with its gradients; the decoding step | widths 2 to 8: forward, backward and step kernels; others the reference |

The operations build on three MLX functions: `checkpoint` and `custom_vjp`, which mlx-rs does not
wrap yet and the workspace's mlx-rs adds (below), and `MetalKernel`, which mlx-rs 0.33.0 wraps.
`tests/hooks.rs` holds each to plain MLX operations.

Part of [Dashmag](https://github.com/bayanasar/dashmag). See the repository for the full picture.

> **Status:** pre-alpha. APIs will change.

## Measured

`tests/conv1d.rs` runs dash-conformance's `conv1d` and `conv1d_state` tests through an MLX
adapter (`tests/common`), which also checks that the kernels ran exactly where they take the call.
Against dash-ops' fixtures (`conv1d`, `general` and `conv1d_state`), as a share of the largest
reference value: the largest over every comparison, to three figures and rounded up, with the bound
for the fixture it was reached on, dash-ops' for the runtimes other than CUDA; every case also
within its own fixture's bound. f32 with `MLX_ENABLE_TF32=0`. Final and new states are exact; so
are a sequence split in two against the whole (outputs and final states) and packed sequences
against one call each (all but the weight and bias gradients' sums, within 1.72e-7).

| | output | input gradient | weight gradient | bias gradient | initial state's gradient | decoding step |
|---|---|---|---|---|---|---|
| f32 | 1.60e-7 (1.8e-7) | 2.07e-7 (2.6e-7) | 6.02e-7 (6.1e-7) | 3.41e-7 (5.5e-7) | 1.19e-7 (1.2e-7) | 1.82e-7 (2.2e-7) |
| bf16 | 3.63e-3 (3.7e-3) | 3.11e-3 (3.2e-3) | 3.34e-3 (3.4e-3) | 3.32e-3 (3.4e-3) | 1.33e-7 (1.5e-7) | 2.40e-3 (3.7e-3) |
| f16 | 2.68e-4 (2.7e-4) | 3.15e-4 (3.2e-4) | 3.38e-4 (3.4e-4) | 2.68e-4 (2.7e-4) | | 3.20e-4 (3.2e-4) |

Each kernel run twice on the same inputs gives the same bits, in f32 and bf16
(`kernels_repeat_bit_for_bit`).

Speed and memory on an Apple M6 (32 GB), MLX 0.32.3, bf16, against mlx-lm's operations (the
state's rows prepended, MLX's `conv1d`, SiLU), timed by the same example in the same process:
the median of 10 runs (20 for the decoding step) after warm-up, the range over two runs of the
example; memory is MLX's peak allocation above the live inputs, outputs included. Decoding steps
are timed as a model runs them, every layer's step in one evaluation, and given per layer.

| operation | shape | dash-mlx | mlx-lm's operations |
|---|---|---|---|
| conv1d, forward | 6144 steps, 6144 channels, width 4 | 1.42-1.43 ms, +72 MiB | 3.22-3.28 ms, +144 MiB |
| conv1d, forward + backward (input and state) | 6144 steps, 6144 channels, width 4 | 3.43-3.46 ms, +155 MiB | 23.0 ms, +576 MiB |
| conv1d, decoding step, 18 layers | batch 32, 6144 channels | 0.053 ms a layer | 0.064 ms a layer |

    cargo run --release -p dash-mlx --example conv1d_bench

## Building

MLX runs on Apple silicon only; on other targets the crate is empty. mlx-rs and mlx-sys come from
bayanasar/mlx-rs's `dashmag` branch, at the commit the workspace's `Cargo.toml` pins: upstream main
(oxiglade/mlx-rs) with three branches merged, each meant as an upstream pull request:
`feat/checkpoint-custom-vjp` adds `checkpoint` and `custom_vjp` to mlx-rs, `feat/mlx-sys-prebuilt`
lets mlx-sys link a prebuilt MLX, and `feat/mlx-c-0.8.0` moves mlx-c to 0.8.0 and MLX to 0.32.4.
When upstream releases them, the two crates go back to crates.io.

It needs the Command Line Tools and cmake, and MLX 0.32.4, the version mlx-c pins there, either

- built from source by mlx-sys, which needs Xcode with its Metal toolchain; or
- prebuilt: `MLX_SYS_PREBUILT` set to the prefix of an installed MLX 0.32.4, which needs neither.
  The `mlx/` directory of `mlx_metal-0.32.4-py3-none-macosx_26_0_arm64.whl` from PyPI (sha256
  `514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936`), a zip archive, is one:

```sh
unzip -q mlx_metal-0.32.4-py3-none-macosx_26_0_arm64.whl 'mlx/*' -d /tmp/mlx-metal
mv /tmp/mlx-metal/mlx ~/.local/opt/mlx-0.32.4
MLX_SYS_PREBUILT=~/.local/opt/mlx-0.32.4 cargo test -p dash-mlx
```

### Outside this workspace

A workspace that takes dash-mlx from this repository, as a git dependency, gets the same mlx-rs
and mlx-sys with it: they are git dependencies of this workspace, not a `[patch]`, which Cargo
applies only in the workspace that declares it. A binary that links dash-mlx against a prebuilt
MLX also needs an rpath to its `lib/`, which mlx-sys publishes as `DEP_MLX_LIB_DIR`; `build.rs`
adds it to this crate's own tests, examples and benchmarks, and a dependent's build script adds it
to its binaries the same way.

Licensed under the Apache License, Version 2.0.
