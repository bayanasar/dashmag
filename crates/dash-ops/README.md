# dash-ops

Operations Burn 0.22 does not have built in, for training and decoding. Each is a backend
extension with its own backward pass:

| module | operation | kernels |
|---|---|---|
| `gated_delta::chunk` | the gated delta rule, chunked, forward and backward, over a batch or packed variable-length sequences (`cu_seqlens`), heads up to 256 wide, with the l2 normalisation of queries and keys and the query scale as options | kernels on the tensor cores from bf16 and f16 inputs (heads of 32, 64 and 128; the products' operands in the input dtype, as fla's), f32 kernels otherwise (heads up to 256), on devices with what they ask (CUDA; not lavapipe, OpenGL or the CPU runtime) |
| `gated_delta::recurrent` | the gated delta rule a token at a time, for decoding: one token per sequence, or packed sequences' tokens in turn | one kernel a step |
| `attention` | attention with grouped key-value heads and the masks models train with (none, causal, key padding, packed variable-length sequences, a sliding window), a backward pass that keeps no T x T scores, and decoding against a key-value cache with per-sequence lengths | FlashAttention-2 style kernels on the tensor cores (bf16, f16; heads of 64, 128, 256) that skip the tiles a mask leaves out, on devices with the tensor cores' mma and ldmatrix (CUDA; not lavapipe, OpenGL or the CPU runtime) |
| `logprob` | the log-probability of chosen tokens under a large vocabulary, with the entropy, without the n x V logits | chunked matmuls and a fused log-sum-exp kernel, on devices with plane operations on planes of 8 or more units (CUDA; not lavapipe, OpenGL or the CPU runtime) |
| `loss` | cross-entropy fused with the output layer over a large vocabulary (mean, sum or none; ignore index, label smoothing, z-loss, masked ids), with each row's log-probability and entropy, gradients for the hidden states and the output weight, without the n x V logits | `logprob`'s pass, and a second pass for the weight's gradient whose kernel takes cubes of 256 units and 64-bit integers (CUDA, lavapipe; not OpenGL or the CPU runtime) |
| `conv1d` | causal depthwise conv1d of any kernel width, with an optional bias and activation (none or SiLU), from given initial states (their gradients too) and returning final states, over a batch or packed variable-length sequences (`cu_seqlens`), and its decoding step | forward, backward and step kernels for widths 1 to 8, reading `cu_seqlens` on the device, on devices with cubes of 256 units (CUDA, lavapipe, OpenGL; 64-bit `cu_seqlens` also need 64-bit integers, which OpenGL lacks; not the CPU runtime) |
| `lowrank` | low-rank products `base + scale · (x U) V` of ranks 1 to 64, as LoRA and the adapters built on it add to a frozen layer: f32 factors over f32, bf16 or f16 inputs, the sum with the base rounded once, gradients of the input, the factors and the base | four kernels (`x U`, `h V` with the base added, the factors' gradients over the rows, a sum of partial runs) that read the wide operand once and sum each element in order, on devices with cubes of 256 units and 17 KiB of shared memory (CUDA, lavapipe; not the CPU runtime) |
| `sample` | the next token of a decoding step from logits `[B, V]` (f32, bf16, f16): greedy, or drawn at a temperature with top-k, top-p and masked ids by the Gumbel-max trick, from Philox4x32-10 keyed by a seed and counted by an offset a step, the same bits on every run; with the token's log-probability under the model, under the model at the temperature (for RL trainers' importance ratios) and under the distribution drawn from | one kernel launch a call, a cube per row that finds the cuts with no sort and no host round trip, among candidates gathered into shared memory (radix passes over the row when they do not fit), on devices with plane operations (CUDA; not lavapipe, OpenGL or the CPU runtime) |
| `checkpoint` | activation checkpointing: a stack of layers run outside autodiff, each layer run again in the backward pass (built on Burn's tensor API, not an extension) | the layers' own |

Each operation has three parts: a reference built from Burn's tensor operations, which runs on
every device and is what the tests hold the kernels to; CubeCL kernels, which run on any CubeCL
runtime whose device offers what they ask; and an implementation for `Autodiff` with a
written-out backward pass, so that training records one node instead of the reference's
intermediate tensors. `tests/extension_pattern.rs` is the smallest complete example of that
shape. Each operation's definition, shapes, dtypes and options, and what it matches, are in
the module of the same name in [`dash-contract`](https://crates.io/crates/dash-contract),
written once for every implementation of these operations; this crate re-exports its option
types and documents its own kernels in each module.

Every operation follows one set of conventions (layout, `[B, T, H, D]` for tensors split into
heads; masks; dtypes; options; how a call outside an operation's contract is refused), written
out in `dash-contract`'s crate documentation; this crate's own (parts, tests) are under
"Conventions" in its crate documentation (`src/lib.rs`).

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
  attention, `logprob`'s pass (the loss's first) and sampling run their references; `cpu`, with as
  many units per cube as cores, where the references run. `vulkan` crashes lavapipe on any kernel
  with shared memory, Burn's own included (CubeCL's SPIR-V compiler declares shared memory wrongly
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

On one RTX 5070 Ti (sm_120) under WSL2 with nothing else on the card, bf16, against PyTorch 2.13
on the same card, the two run in turn (2026-10-27; sampling, the chunked gated delta rule at heads
of 128 and attention's forward at heads of 64, 2026-10-28), built with `cuda` only unless a row says otherwise:

| operation | shape | dash-ops | PyTorch |
|---|---|---|---|
| gated delta rule, chunked, forward + backward | 6144 tokens, 16 heads of 128: one sequence / four packed of 1536 | this operation's kernels 2.57 / 2.44 ms on the device (with the benchmark's loss 3.58 / 3.45), +400 / +408 MiB; forward alone 0.81 / 0.79 ms on the device | fla's kernels 2.54 / 2.37 ms (with the loss 3.11 / 2.94), +398 / +398 MiB; forward 0.69 / 0.70 ms (fla 0.5.2) |
| gated delta rule, chunked, forward + backward | the same with 16 heads of 256 (the f32 kernels) | 19.7 / 20.0 ms on the device, +1548 / +1985 MiB; forward 5.4 / 5.7 ms | 6.6 / 6.2 ms; forward 1.9 / 1.7 ms (fla 0.5.2) |
| gated delta rule, one decoding step | batch 8, 16 heads of 128, 18 layers in turn | 21.7 us a layer (bf16 inputs 21.5) | 30.5 us (bf16 31.1; fla 0.5.2) |
| attention, forward + backward | 6144 tokens, 8 query and 2 key-value heads of 256: causal; no mask; 4 packed sequences of 1536; causal within a window of 1024 | on the device, deterministic (the default): 9.17 ms, +60 MiB (and 24 MiB of f32 accumulators during the backward); 17.6 ms; 2.59 ms; 3.07 ms; with atomics: 7.62 ms (72 MiB of accumulators); 14.9 ms; 2.33 ms; 2.69 ms (wall clock 9.6; 18.1; 2.9; 3.4 ms and with atomics 8.0; 15.3; 2.7; 3.0 ms) | flash on the device: 7.30 ms, +204 MiB; 14.7 ms; 2.34 ms as a padded batch of 4; the window through flex_attention 4.35 ms |
| attention, forward | 6144 tokens, 8 query and 2 key-value heads of 64: causal; no mask; 4 packed sequences of 1536; causal within a window of 1024 | on the device 0.528; 0.902; 0.145; 0.163 ms (in the tiles of 128 queries that heads of 128 and 256 keep: 0.587; 1.09; 0.175; 0.199 ms) | flash through PyTorch's SDPA on the device: 0.557 ms (cuDNN's 0.510); 0.887 ms; 0.150 ms as a padded batch of 4; the window through flex_attention 0.169 ms |
| attention, decoding | batch 8, 6144 cached keys: the whole cache; lengths from 100 to 6144; a window of 1024 | on the device 131 us; 36 us; 15 us a layer (wall clock over 24 calls between synchronisations: 146; 59; 37 us, the last two bound by the host's dispatch) | on the device 130 us; 3663 us (a padding mask); 26 us (the last 1025 keys) (wall clock 137; 3716; 34 us) |
| sampling a token | batch 8 / 32, 248320 classes, bf16 logits peaked as a model's: greedy; temperature 0.7; with top-p 0.95, and on flat logits, where the cut keeps over half the vocabulary; top-k 50; temperature 0.7, top-k 50, top-p 0.95 | on the device 21.7 / 21.8 us; 136 / 136 us; 228 / 230 us, flat 460 / 462 us; 48.2 / 48.2 us; 49.4 / 49.4 us, measured before the tempered log-probability, which adds an exponential a token where top-k cuts at a temperature, and not yet measured again (wall clock under Nsight Systems 0.15 / 0.14; 0.27 / 0.27; 0.51 / 0.51, flat 0.75 / 0.74; 0.17 / 0.17; 0.17 / 0.17 ms) | sort, softmax, cumulative sum and multinomial, as Hugging Face's and vLLM's PyTorch samplers form them: on the device 67.9 / 152 us; 157 / 406 us; 720 / 2545 us, flat 706 / 2523 us; 131 / 338 us; 142 / 389 us (wall clock 0.12 / 0.21; 0.29 / 0.51; 0.96 / 2.85, flat 0.94 / 2.85; 0.42 / 0.50; 0.45 / 0.62 ms) |
| log-probability, forward + backward | 3072 tokens, 248320 classes, width 1024 | 78.9 ms, +216 MiB (70.4 ms with `fusion` and `autotune`) | 86.7 ms, +8730 MiB |
| linear + cross-entropy, forward + backward with the weight's gradient | 3072 / 8192 tokens, 248320 classes, width 1024 | 102 / 293 ms, +641 / +731 MiB (94 / 261 ms with `autotune`) | logits in f32: 94 / 1112 to 1137 ms, +11672 / +31072 MiB; in bf16: 64 / 2249 to 2271 ms, +5824 / +15520 MiB (at 8192 both spill past the card's 16 GB into host memory, and their times move with it) |
| low-rank product `scale · (x U) V`, r 16, bf16 x, f32 factors | 2048 tokens, `in -> out` 1024 -> 3584 / 3584 -> 1024 / 1024 -> 2048 / 1024 -> 256: forward; forward + backward | on the device 0.042 / 0.052 / 0.032 / 0.018 ms; 0.19 / 0.17 / 0.14 / 0.069 ms (Burn's own matmuls: 0.32 / 0.90 / 0.29 / 0.26; 1.6 / 1.8 / 1.2 / 0.84 ms) | PEFT's LoRA term: 0.088 / 0.108 / 0.048 / 0.027; 0.27 / 0.24 / 0.15 / 0.063 ms |
| conv1d, forward + backward | 6144 steps, 6144 channels, width 4, SiLU (widths 2 to 4, with or without a bias: 0.54 to 0.57 ms); from initial states / with final states too; as 4 packed sequences of 1536 / 4 calls on 1536 / a batch of 4 | 0.56 ms; 0.57 / 0.57 ms; 0.56 / 0.54 / 0.56 ms (on the device 0.55; 0.55 / 0.55; 0.55 / 0.51 / 0.55 ms) | 2.64 ms (conv1d, then SiLU); 2.75 / 3.13 ms (`torch.cat` of the state first); 4 calls 2.44 ms |

The device times are sums over `nsys`'s kernels; each row's two sides ran alternately, two or three
rounds each, and the figures are their medians. At heads of 128, forward and backward, the chunked
gated delta rule's kernels on tensor cores take 1.01 times fla's time (1.03 packed). Kernel by
kernel, over one sequence: the state passings 0.90 ms against fla's 0.92 (0.25 ms forward, twice,
and 0.41 backward, against 0.20 and 0.51); the WY part 0.26 against 0.26 and the local `dv` 0.11
against 0.12; the first kernel 0.33 against 0.20 in fla's three, the output 0.24 against 0.18,
recomputing `W` and `U` backward 0.18 against 0.16, and the kernel for all but the WY part 0.55
against 0.40; fla's l2 normalisation of the queries and keys, 0.12 ms forward and 0.18 backward,
runs in kernels of its own, where these fold it into theirs. The first kernel inverts `I + A` with
64 units in f32, on CUDA cores, where fla inverts on tensor cores in TF32, and the kernel for all
but the WY part holds 68.5 KiB of shared memory, so that one cube fits on a multiprocessor; how
much of either gap those account for has not been measured by switching them.
The backward keeps the forward's `T` (8 KiB a chunk of each head) as fla does. The loss around it in the benchmark (a cast, a product and a sum over the output) takes
1.01 ms in Burn's kernels and 0.57 in PyTorch's. Heads of 256 run the f32 kernels, three times fla's
time. Packed sequences cost about what one sequence of their length does: the state passings take
each sequence's chunks in turn, two of their cubes fitting on a multiprocessor on tensor cores (44
and 48 KiB of shared memory), one in f32.

Attention's backward by default gives the same bits on every run: a kernel of its own sums the
query gradient over the key blocks in a fixed order, seven T x T x D products in all, which take
1.30 times the backward's time with five. `AttentionOptions::deterministic` set to `false` runs the
five, adding the query gradient with f32 atomics as FlashAttention-2 does by default, and its last
bits then vary from run to run (the output and the key and value gradients do not). Attention's
decoding is timed over 24 calls between synchronisations; with cache lengths or a window its kernels
take 15 to 36 us of the device's time, and the wall clock, 37 to 59 us, is the host dispatching
them.

Sampling is timed over 200 calls with a synchronisation around each, back to back with PyTorch's
for each configuration over three rounds; on flat logits every mode's device time but top-p's is
within 1% of the peaked figure. The kernel
takes a cube a row, so a batch of 8 leaves most of the card idle; greedy's wall clock there,
0.15 ms against 21.7 us on the device, is the host's part of the call.
FlashInfer's fused sampling, which serving engines use on CUDA, has not been run beside it.

The low-rank product's row is by device time under Nsight Systems: the kernels and Burn's own
matmuls from `lowrank_bench`, PEFT's term the same way on PyTorch, from its timed passes but for
1024 -> 2048 forward and backward, where the card had lowered its clocks during them (0.19 ms) and
the passes run back to back before them give 0.15 (`dash-lora`'s README says why). Each example
in `examples/` reproduces its row (`--features cuda`).
`matmul_bench` and `dispatch_bench` measure Burn itself: bf16 matmul against cuBLAS, and the
host-side cost of each unfused operation.

## Tests

    cargo test -p dash-ops                                     # the references, on the CPU
    CUBECL_CUDA_TF32=0 cargo test -p dash-ops --features cuda  # the kernels, against them

`CUBECL_CUDA_TF32=0` keeps f32 matmuls in IEEE f32 (the workspace's patched cubecl-cuda); the f32
tolerances assume it. The fixtures in `tests/fixtures/` were recorded once from PyTorch and
transformers on seeded inputs; nothing in the repository runs Python.
