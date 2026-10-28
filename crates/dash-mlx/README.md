# dash-mlx

dash-ops' and dash-lora's operations on MLX, for Apple silicon, with dash-ops' and dash-lora's
contracts and held to their test fixtures. For now this crate is dashmag's path on Apple silicon:
MLX already has arrays, autodiff, matmul, a fused attention forward, rms_norm, RoPE and Metal, and
this crate adds what dash-ops and dash-lora add to Burn. dashmag's Burn side still targets Metal
as well, through dash-ops' and dash-lora's `metal` feature.

So far it holds the three MLX functions the operations build on: `checkpoint` and `custom_vjp`,
which mlx-rs does not wrap yet and the workspace's mlx-rs adds (below), and `MetalKernel`, which
mlx-rs 0.33.0 wraps. `tests/hooks.rs` holds each to plain MLX operations.

Part of [Dashmag](https://github.com/bayanasar/dashmag). See the repository for the full picture.

> **Status:** pre-alpha. APIs will change.

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
