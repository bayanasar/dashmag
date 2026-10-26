# dash-ops

Training operations Burn 0.22 does not have built in. Each is a backend extension with its own
backward pass:

| module | operation | on CUDA |
|---|---|---|
| `gated_delta::chunk` | the gated delta rule, chunked, forward and backward | f32 kernels, no tensor cores yet |
| `gated_delta::recurrent` | the gated delta rule one token at a time, for decoding | one kernel a step |
| `attention` | causal attention with grouped key-value heads, a backward pass that keeps no T x T scores, and decoding against a key-value cache | FlashAttention-2 style kernels on the tensor cores |
| `logprob` | the log-probability of chosen tokens under a large vocabulary, with the entropy, without the n x V logits | chunked matmuls and a fused log-sum-exp kernel |
| `loss` | cross-entropy fused with the output layer over a large vocabulary (mean, sum or none; ignore index, label smoothing, z-loss, masked ids), with each row's log-probability and entropy, gradients for the hidden states and the output weight, without the n x V logits | `logprob`'s pass, and a second pass for the weight's gradient |
| `conv1d` | causal depthwise conv1d of any kernel width, with an optional bias and activation (none or SiLU), and its decoding state | forward, backward and step kernels for widths 1 to 8 |
| `checkpoint` | activation checkpointing: a stack of layers run outside autodiff, each layer run again in the backward pass (built on Burn's tensor API, not an extension) | the layers' own |

Each operation has three parts: a reference built from Burn's tensor operations, which runs on
every device and is what the tests hold the kernels to; CubeCL kernels on the CUDA backend; and
an implementation for `Autodiff` with a written-out backward pass, so that training records one
node instead of the reference's intermediate tensors. `tests/extension_pattern.rs` is the
smallest complete example of that shape. Shapes, dtypes and what each operation matches are in
its module's documentation.

Every operation follows one set of conventions (layout, `[B, T, H, D]` for tensors split into
heads; masks; dtypes; options; parts; tests), written out under "Conventions" in the crate
documentation (`src/lib.rs`).

Part of [Dashmag](https://github.com/bayanasar/dashmag) — a machine-learning library in Rust
on Burn. See the repository for the full picture.

## Features

- `flex` (default): the references, on the CPU.
- `cuda`: the CubeCL kernels.
- `fusion`: Burn's lazy fusion.
- `autotune`: Burn's autotune. Matmul-heavy work wants it: on an RTX 5070 Ti, Burn's bf16
  [6144x3584]x[3584x1024] runs at 6 TFLOPS without it and 75 with it.

`fusion` and `autotune` are off by default. They change which kernels every Burn crate in the
same build runs, and with `autotune` which kernel runs depends on timing: a process tunes each
new shape when it first meets it (the results do not persist), and threads that ask for a shape
while it is being tuned run an untuned candidate meanwhile. Results then differ in their last
bits from run to run, and between two passes of one computation in one process.
Before its first tuning, autotune measures the device's memory bandwidth with two 512 MiB
buffers of its own, returned to the driver afterwards; a process that has filled the card by
then leaves it no room.

## Measured

On one RTX 5070 Ti (sm_120) under WSL2, bf16, against PyTorch 2.13 on the same card, built
with `cuda` only unless a row says otherwise:

| operation | shape | dash-ops | PyTorch |
|---|---|---|---|
| gated delta rule, chunked, forward + backward | 6144 tokens, 16 heads of 128 | 9.2 ms, +687 MiB | 3.6 ms, +422 MiB (fla 0.5.2) |
| gated delta rule, one decoding step | batch 8, 16 heads of 128, 18 layers in turn | 22 us a layer | 34 us (fla 0.5.2) |
| attention, forward + backward | 6144 tokens, 8 query and 2 key-value heads of 256 | 9.2 ms, +60 MiB | 7.4 ms, +204 MiB (flash) |
| attention, decoding | batch 8, 6144 cached keys | 200 us a layer, synchronised after each call (144 us over 24 calls; 454 us each with `fusion`) | 201 us |
| log-probability, forward + backward | 3072 tokens, 248320 classes, width 1024 | 78.9 ms, +216 MiB (70.4 ms with `fusion` and `autotune`) | 86.4 ms, +8730 MiB |
| linear + cross-entropy, forward + backward with the weight's gradient | 3072 / 8192 tokens, 248320 classes, width 1024 | 102 / 293 ms, +641 / +731 MiB (94 / 261 ms with `autotune`) | logits in f32: 95 / 1179 ms, +11672 / +31072 MiB; in bf16: 64 / 1241 ms, +5824 / +15520 MiB (at 8192 both spill past the card's 16 GB into host memory) |
| conv1d, forward + backward | 6144 steps, 6144 channels, width 4, SiLU (widths 2 to 4, with or without a bias: 0.54 to 0.57 ms) | 0.56 ms | 2.63 ms (conv1d, then SiLU) |

The chunked gated delta rule's kernels compute in f32 without tensor cores, which is most of its
distance from fla; attention's backward does seven T x T x D products to FlashAttention-2's five,
to avoid atomics. Each example in `examples/` reproduces its row (`--features cuda`).
`matmul_bench` and `dispatch_bench` measure Burn itself: bf16 matmul against cuBLAS, and the
host-side cost of each unfused operation.

## Tests

    cargo test -p dash-ops                                     # the references, on the CPU
    CUBECL_CUDA_TF32=0 cargo test -p dash-ops --features cuda  # the kernels, against them

`CUBECL_CUDA_TF32=0` keeps f32 matmuls in IEEE f32 (the workspace's patched cubecl-cuda); the f32
tolerances assume it. The fixtures in `tests/fixtures/` were recorded once from PyTorch and
transformers on seeded inputs; nothing in the repository runs Python.
