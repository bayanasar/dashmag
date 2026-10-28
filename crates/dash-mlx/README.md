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
| `gated_delta::chunk` | the gated delta rule, chunked, forward and backward: the l2 normalisation of queries and keys and the query scale as options, from initial states and to final states, over a batch or packed variable-length sequences | the chunks' triangular inverses and the scan from chunk to chunk; the rest MLX's batched operations |
| `gated_delta::recurrent` | the gated delta rule a token at a time, for decoding: one token of each sequence, or several of each, packed | one kernel a call; its gradient the reference's |

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

The gated delta rule (`tests/gated_delta_chunk.rs`, `gated_delta_recurrent.rs` and
`gated_delta_packed.rs`: dash-conformance's tests of the same names through the adapter, which
also checks which kernels ran, on dash-ops' fixtures): the output, the final state and every
input's gradient, against transformers' torch rule in f32 or with its casts made f64, and against
fla where the fixture has it. The largest over every comparison, to three figures and rounded up,
with dash-ops' bounds; f32 with `MLX_ENABLE_TF32=0` (below).

| case | reference | largest error (bound) |
|---|---|---|
| chunked, f32 | transformers in f32; in f64 | 1.06e-6 (2e-6); 9.68e-7 (2e-6) |
| chunked, f32, packed, with a query scale and wide keys or values among the cases | torch in f64; fla | 1.47e-6 (1.5e-6); 2.34e-6 (2.5e-6) |
| chunked, bf16; packed | torch in f64 | 2.41e-3 (3e-3); 2.59e-3 (3e-3) |
| chunked, the kernels at 1000 and 6144 tokens, f32 | the same rule as MLX's operations | 3.93e-7 (2e-6) |
| a step, f32: output; state | transformers in f32 | 2.60e-7 (8e-7); 1.27e-7 (4e-7) |
| a step, f32: every gradient | transformers' autograd | 1.25e-6 (2e-6) |
| steps, f32, packed: output and state; gradients | torch in f64 | 1.40e-7 (3e-7); 2.48e-7 (4e-7) |
| steps, f32, packed: output and state | fla | 2.02e-7 (3e-7) |
| a step, bf16 inputs: output; state | torch's own bf16 | 6.95e-3 (8e-3); 3.44e-3 (4e-3) |
| a 100-token prompt chunked, then 16 steps: the prompt's state; the steps' output; their state | transformers in f32 | 1.33e-6 (2e-6); 2.04e-7 (8e-7); 1.34e-7 (4e-7) |

Packed sequences give each sequence's values, to the bit, as a call on it alone. Each kernel run
twice on the same inputs gives the same bits, in f32 and bf16, the chunked rule's forward and
backward and the step, over one sequence and packed (`kernels_repeat_bit_for_bit`).

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

mlx-lm has no backward pass for the gated delta rule (its kernel serves inference), so the chunked
rule's baselines are MLX's own operations: the same rule differentiated by MLX throughout, and the
same with the scan's and the inverses' backward passes written out (this crate's reference). Each
of the two kernels is also timed against its part as MLX's operations, the other part held. For
the forward pass and the decoding step the baseline is mlx-lm's kernel (`examples/common/mlx_lm.rs`,
its source under its MIT notice), which keeps the state as `[Dv, Dk]` where dash-ops keeps
`[Dk, Dv]`. 16 heads of 128, bf16 inputs, an f32 state, the normalisation inside; the range over
two runs of each example; TF32 at MLX's default.

| operation, 6144 tokens | forward | forward + backward |
|---|---|---|
| dash-mlx, both kernels | 24.8 ms, +843 MiB | 76.4-77.0 ms, +1661 MiB |
| the inverses' kernel; the scan as MLX's operations | 26.9-27.0 ms, +807 MiB | 79.4-79.9 ms, +1878-1901 MiB |
| the scan's kernels; the inverses as MLX's operations | 29.0 ms, +1021 MiB | 83.6-83.9 ms, +1661 MiB |
| MLX's operations, backward passes written out (the reference) | 31.8-32.0 ms, +817 MiB | 84.1-84.3 ms, +1878-1922 MiB |
| MLX's operations through autograd | 29.5-29.6 ms, +807 MiB | 474-475 ms, +2028 MiB |
| mlx-lm's kernel, a token at a time | 10.9-11.0 ms, +73 MiB | |
| dash-mlx's step, a token at a time | 9.49-9.50 ms, +25 MiB | |

| decoding step, per layer, 18 layers in one evaluation | dash-mlx | mlx-lm |
|---|---|---|
| batch 32, a token each, the normalisation inside | 0.467-0.468 ms | 0.460-0.462 ms (`rms_norm`, `exp` and its kernel) |
| the same, the kernels alone | 0.466 ms | 0.455 ms |
| batch 32, four tokens each, the normalisation inside | 0.624-0.636 ms | 0.505-0.506 ms |
| the same, the kernels alone | 0.620-0.633 ms | 0.491-0.492 ms |

    cargo run --release -p dash-mlx --example gated_delta_chunk_bench
    cargo run --release -p dash-mlx --example gated_delta_recurrent_bench

Each kernel is faster than its part as MLX's operations: the inverses' kernel saves 4.2 to 7.5 ms
of a forward and backward pass, the scan's 0.2 to 3.5 ms and 217 to 261 MiB of the peak. The decoding
step's memory is almost all the 18 new states (576 MiB of +578). Its kernel was chosen by
measurement on dash-ops' `[Dk, Dv]` state (module documentation): against dash-ops' arrangement in
MSL, 0.463-0.464 ms for one token and 0.737-0.756 ms for four, it takes 1% more time for one
token and 14 to 18% less for four; arrangements with no threadgroup memory between tokens were
slower still. It takes 1 to 2% more time than mlx-lm's kernel for one token and 23 to 26% more
for four: the state moves in and out once a call either way, but here the sums over a column
meet in threadgroup memory, four barriers a token, where in mlx-lm's `[Dv, Dk]` layout a lane's
rows lie together and its sums are `simd_sum`s. Run over 6144 tokens of one sequence, it takes
13 to 14% less time than mlx-lm's kernel and a third of its memory.

## f32 and TF32

MLX runs f32 matrix products in TF32 on GPUs that have it, unless `MLX_ENABLE_TF32=0`. On the M6
the products' error is then 5e-4 to 7.6e-4 of the largest value, against 1e-7 to 3.4e-7 with it
off (measured, products of 8 to 1024 on a side). The operations here compute in f32 inside, as
dash-ops does, and their f32 bounds assume IEEE f32: the delta rule's tests check the setting
and stop with a message if it is on. The chunked rule's speed does not depend on it (measured
both ways: 24.9 and 77.3 ms with it on, 24.9 and 76.1 ms off).

    MLX_ENABLE_TF32=0 cargo test -p dash-mlx

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
MLX_SYS_PREBUILT=~/.local/opt/mlx-0.32.4 MLX_ENABLE_TF32=0 cargo test -p dash-mlx
```

### Outside this workspace

A workspace that takes dash-mlx from this repository, as a git dependency, gets the same mlx-rs
and mlx-sys with it: they are git dependencies of this workspace, not a `[patch]`, which Cargo
applies only in the workspace that declares it. A binary that links dash-mlx against a prebuilt
MLX also needs an rpath to its `lib/`, which mlx-sys publishes as `DEP_MLX_LIB_DIR`; `build.rs`
adds it to this crate's own tests, examples and benchmarks, and a dependent's build script adds it
to its binaries the same way.

Licensed under the Apache License, Version 2.0.
