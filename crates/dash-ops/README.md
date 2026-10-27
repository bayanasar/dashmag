# dash-ops

Training operations Burn 0.22 does not have built in. Each is a backend extension with its own
backward pass:

| module | operation | kernels |
|---|---|---|
| `gated_delta::chunk` | the gated delta rule, chunked, forward and backward, over a batch or packed variable-length sequences (`cu_seqlens`), heads up to 256 wide, with the l2 normalisation of queries and keys and the query scale as options | kernels on the tensor cores from bf16 and f16 inputs (heads of 32, 64 and 128; the products' operands in the input dtype, as fla's), f32 kernels otherwise (heads up to 256), on devices with what they ask (CUDA; not lavapipe, OpenGL or the CPU runtime) |
| `gated_delta::recurrent` | the gated delta rule a token at a time, for decoding: one token per sequence, or packed sequences' tokens in turn | one kernel a step |
| `attention` | attention with grouped key-value heads and the masks models train with (none, causal, key padding, packed variable-length sequences, a sliding window), a backward pass that keeps no T x T scores, and decoding against a key-value cache with per-sequence lengths | FlashAttention-2 style kernels on the tensor cores (bf16, f16; heads of 64, 128, 256) that skip the tiles a mask leaves out, on devices with the tensor cores' mma and ldmatrix (CUDA; not lavapipe, OpenGL or the CPU runtime) |
| `logprob` | the log-probability of chosen tokens under a large vocabulary, with the entropy, without the n x V logits | chunked matmuls and a fused log-sum-exp kernel, on devices with plane operations on planes of 8 or more units (CUDA; not lavapipe, OpenGL or the CPU runtime) |
| `loss` | cross-entropy fused with the output layer over a large vocabulary (mean, sum or none; ignore index, label smoothing, z-loss, masked ids), with each row's log-probability and entropy, gradients for the hidden states and the output weight, without the n x V logits | `logprob`'s pass, and a second pass for the weight's gradient whose kernel takes cubes of 256 units and 64-bit integers (CUDA, lavapipe; not OpenGL or the CPU runtime) |
| `conv1d` | causal depthwise conv1d of any kernel width, with an optional bias and activation (none or SiLU), from given initial states (their gradients too) and returning final states, over a batch or packed variable-length sequences (`cu_seqlens`), and its decoding step | forward, backward and step kernels for widths 1 to 8, reading `cu_seqlens` on the device, on devices with cubes of 256 units (CUDA, lavapipe, OpenGL; 64-bit `cu_seqlens` also need 64-bit integers, which OpenGL lacks; not the CPU runtime) |
| `checkpoint` | activation checkpointing: a stack of layers run outside autodiff, each layer run again in the backward pass (built on Burn's tensor API, not an extension) | the layers' own |

Each operation has three parts: a reference built from Burn's tensor operations, which runs on
every device and is what the tests hold the kernels to; CubeCL kernels, which run on any CubeCL
runtime whose device offers what they ask; and an implementation for `Autodiff` with a
written-out backward pass, so that training records one node instead of the reference's
intermediate tensors. `tests/extension_pattern.rs` is the smallest complete example of that
shape. Shapes, dtypes and what each operation matches are in its module's documentation.

Every operation follows one set of conventions (layout, `[B, T, H, D]` for tensors split into
heads; masks; dtypes; options; parts; tests), written out under "Conventions" in the crate
documentation (`src/lib.rs`).

Part of [Dashmag](https://github.com/bayanasar/dashmag) — a machine-learning library in Rust
on Burn. See the repository for the full picture.

## Features

- `flex` (default): the references, on the CPU.
- `cuda`, `rocm`, `wgpu`, `vulkan`, `metal`, `webgpu`, `cpu`: Burn's CubeCL runtimes, each with
  the CubeCL kernels. Burn 0.22 has one CubeCL backend for all of them, and a tensor's device
  names the runtime its kernels compile for: CUDA; HIP (`rocm`); WGSL on whichever graphics API
  wgpu finds (`wgpu`), SPIR-V on Vulkan, MSL on Metal, WGSL in the browser (`webgpu`); LLVM for
  the CPU (`cpu`). Before it launches, a kernel asks the device for what it needs (units per cube,
  shared memory, a plane width, the tensor cores' instructions, 64-bit integers, the dtypes it
  reads), and a device without it runs the operation's reference. Tested: `cuda` on an RTX 5070
  Ti, where every kernel runs; `wgpu` on the same card through OpenGL over D3D12 (Mesa's d3d12
  driver under WSL2) and on lavapipe (Vulkan on the CPU), both with 32 KiB of shared memory and
  no plane operations, where the gated delta rule's step and conv1d run their kernels (and on
  lavapipe, with 64-bit integers, the loss's second pass) and the chunked gated delta rule,
  attention and `logprob`'s pass (the loss's first) run their references; `cpu`, with as many
  units per cube as cores, where the references run. `vulkan` crashes lavapipe on any kernel with
  shared memory, Burn's own included (CubeCL's SPIR-V compiler declares shared memory wrongly
  where a device lacks explicit workgroup layouts); `rocm` builds but is untested, `metal` and
  `webgpu` are untested. The tests run the kernels on the device `DASH_OPS_DEVICE` names
  (`tests/common/mod.rs`).
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
| gated delta rule, chunked, forward + backward | 6144 tokens, 16 heads of 128: one sequence / four packed of 1536 | this operation's kernels 2.87 / 2.72 ms on the device (with the benchmark's loss 3.90 / 3.75; 4.4 / 4.6 ms on the host), +400 / +408 MiB; forward alone 0.89 / 0.86 ms on the device | fla's kernels 2.67 / 2.50 ms (with the loss 3.25 / 3.10), +398 / +398 MiB; forward 0.73 / 0.71 ms (fla 0.5.2) |
| gated delta rule, chunked, forward + backward | the same with 16 heads of 256 (the f32 kernels) | 21.8 / 22.5 ms on the device, +1548 / +1985 MiB; forward 6.1 / 6.4 ms | 7.5 / 6.8 ms; forward 2.2 / 1.8 ms (fla 0.5.2) |
| gated delta rule, one decoding step | batch 8, 16 heads of 128, 18 layers in turn | 21 us a layer | 34 us (fla 0.5.2) |
| attention, forward + backward | 6144 tokens, 8 query and 2 key-value heads of 256: causal; no mask; 4 packed sequences of 1536; causal within a window of 1024 | 9.5 ms, +60 MiB (kernels 9.0 ms); 18.4 ms; 2.9 ms; 3.4 ms | flash: 7.6 ms, +204 MiB; 15.0 ms; 2.5 ms as a padded batch of 4; the window through flex_attention 12.9 ms |
| attention, decoding | batch 8, 6144 cached keys: the whole cache; lengths from 100 to 6144; a window of 1024 | 147 us; 52 us; 42 us a layer, over 24 calls between synchronisations | 140 us; 3729 us (a padding mask); 37 us (the last 1025 keys) |
| log-probability, forward + backward | 3072 tokens, 248320 classes, width 1024 | 78.9 ms, +216 MiB (70.4 ms with `fusion` and `autotune`) | 86.4 ms, +8730 MiB |
| linear + cross-entropy, forward + backward with the weight's gradient | 3072 / 8192 tokens, 248320 classes, width 1024 | 102 / 293 ms, +641 / +731 MiB (94 / 261 ms with `autotune`) | logits in f32: 95 / 1179 ms, +11672 / +31072 MiB; in bf16: 64 / 1241 ms, +5824 / +15520 MiB (at 8192 both spill past the card's 16 GB into host memory) |
| conv1d, forward + backward | 6144 steps, 6144 channels, width 4, SiLU (widths 2 to 4, with or without a bias: 0.54 to 0.57 ms); from initial states / with final states too; as 4 packed sequences of 1536 / 4 calls on 1536 / a batch of 4 | 0.56 ms; 0.57 / 0.57 ms; 0.56 / 0.55 / 0.57 ms (on the device 0.55; 0.55 / 0.56; 0.55 / 0.51 / 0.55 ms) | 2.63 ms (conv1d, then SiLU); 2.76 / 3.14 ms (`torch.cat` of the state first); 4 calls 2.46 ms |

The chunked gated delta rule's device times are sums over `nsys`'s kernels, fla's measured
beside them under the same load on the card's host (the wall-clock times rise with it). At heads
of 128, forward and backward, its kernels on tensor cores take 1.07 times fla's time (1.09
packed): the state passings and the WY part are faster than fla's (0.26 ms forward, twice, and
0.40 backward, against 0.23 and 0.54; 0.28 against 0.37 with the keys' l2 normalisation), the
other per-chunk kernels slower. Its first kernel inverts `I + A` with 64 units in f32, on CUDA
cores (0.37 ms with the forward's other work, fla's 0.28 in three kernels, its inversion on
tensor cores in TF32), and the backward's kernel for all but the WY part holds 77 KiB of shared
memory, so that one cube fits on a multiprocessor and waits on memory between its phases more
than fla's pipelined kernels do (0.64 ms, against 0.49 for fla's and the queries' l2
normalisation, `dv` going to fla's WY kernel). The backward keeps the forward's `T` (8 KiB a
chunk of each head) as fla does. The loss around
it in the benchmark (a cast, a product and a sum over the output) takes 1.0 ms in Burn's kernels
and 0.6 in PyTorch's. Heads of 256 run the f32 kernels, three times fla's time. Packed sequences
cost about what one sequence of their length does: the state passings take each sequence's
chunks in turn, two of their cubes fitting on a multiprocessor on tensor cores (44 and 48 KiB of
shared memory), one in f32. Attention's backward does seven
T x T x D products to FlashAttention-2's five, to avoid atomics. Each example in `examples/`
reproduces its row (`--features cuda`).
`matmul_bench` and `dispatch_bench` measure Burn itself: bf16 matmul against cuBLAS, and the
host-side cost of each unfused operation.

## Tests

    cargo test -p dash-ops                                     # the references, on the CPU
    CUBECL_CUDA_TF32=0 cargo test -p dash-ops --features cuda  # the kernels, against them

`CUBECL_CUDA_TF32=0` keeps f32 matmuls in IEEE f32 (the workspace's patched cubecl-cuda); the f32
tolerances assume it. The fixtures in `tests/fixtures/` were recorded once from PyTorch and
transformers on seeded inputs; nothing in the repository runs Python.
