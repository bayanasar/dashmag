//! Operations Burn does not have built in, for training and decoding.
//!
//! Each operation is a backend extension (`burn::backend::backend_extension`): a trait whose
//! default body builds the operation from Burn's own tensor operations, which is the reference
//! every device can run and the tests hold everything else to; CubeCL kernels where the
//! reference is too slow or holds too much memory, run on any CubeCL runtime whose device offers
//! what they ask; and an implementation for `Autodiff` with the operation's own backward pass,
//! so that training does not record the reference's intermediate tensors.
//! `tests/extension_pattern.rs` is the smallest complete example of the three, and the shape
//! each module here follows.
//!
//! - [`gated_delta`]: the gated delta rule of Gated DeltaNet, chunked for training and
//!   recurrent for decoding, over a batch or packed variable-length sequences.
//! - [`attention`]: attention with grouped key-value heads and the masks models train with,
//!   whose backward pass keeps no T x T scores, and decoding against a key-value cache.
//! - [`logprob`]: the log-probability of chosen tokens under a large vocabulary, without the
//!   n x V logits.
//! - [`loss`]: cross-entropy fused with the output layer over a large vocabulary, with label
//!   smoothing, z-loss, an ignore index and masked ids, training the output weight too.
//! - [`conv1d`]: causal depthwise 1-D convolution, from a given state and over a batch or packed
//!   variable-length sequences, with the state a decoder carries.
//! - [`lowrank`]: low-rank products `base + scale · (x Aᵀ) Bᵀ` of small rank, as LoRA and the
//!   adapters built on it add to a frozen layer, with f32 factors over inputs of any float dtype.
//! - [`checkpoint`]: activation checkpointing, not an operation but built on Burn's tensor API:
//!   a stack of layers run outside autodiff, each layer run again in the backward pass.
//! - [`sequences`]: packed variable-length sequences, as the operations that take them share
//!   them.
//!
//! # Conventions
//!
//! Every operation follows these, and new code is held to them, so that one operation's output
//! feeds the next without a copy or a cast. Each module's documentation gives its shapes,
//! dtypes and options in their terms.
//!
//! ## Layout
//!
//! Sequences are batch-major, then time: `[B, T, ...]`, the order a linear layer's output
//! reshapes to without a copy. Tensors split into heads are `[B, T, H, D]`, tensors of channels
//! `[B, T, C]`; an operation over rows takes `[N, D]`, `N` any flattening of the leading
//! dimensions. A single decoding step drops `T`: `[B, H, D]`, `[B, C]`. State an operation
//! carries from call to call keeps `B` first and is f32: the gated delta rule's
//! `[B, H, Dk, Dv]`, a convolution's last `W` inputs `[B, W, C]`. Packed variable-length
//! sequences lie end to end along `T` with `B = 1`, the boundaries of `S` sequences given as
//! cumulative offsets `cu_seqlens` `[S + 1]` (Int), as flash-attn and fla take them
//! ([`sequences`]). Inputs may have any strides (a kernel may copy one it cannot read in place);
//! outputs are contiguous in the documented layout.
//!
//! ## Masks
//!
//! A boolean mask is `true` where an element is left out, as in Burn's `mask_fill` and its
//! padding masks; a list of indices names the elements left out (vocabulary ids). Causality is
//! an option, not a mask tensor, aligned at the bottom right when keys outnumber queries. A
//! left-out element takes no part in any sum or normalisation; what a row with nothing left
//! returns is stated by its operation.
//!
//! ## Dtypes
//!
//! The floating inputs of one call share one dtype, f32, bf16 or f16, apart from arguments
//! documented as f32 (states, and gates models compute in f32). Arithmetic accumulates in f32;
//! outputs and gradients come back in the dtype of the input they belong to, rounded once from
//! f32, unless the operation names an intermediate it rounds to the input's dtype for the tensor
//! cores. Statistics an operation returns per row (a log-sum-exp, log-probabilities, entropies)
//! are f32 whatever the inputs' dtype. f32 inputs are computed in IEEE f32: on CUDA with
//! `CUBECL_CUDA_TF32=0`, without which Burn 0.22 runs f32 matmuls in TF32.
//!
//! ## Options
//!
//! Arguments with a usual default go in an options struct implementing `Default`, like
//! [`logprob::LogprobOptions`]; a softmax scale defaults to `1 / sqrt(D)`.
//!
//! ## Parts
//!
//! The reference is the extension trait's default body, built from Burn's tensor operations in f32;
//! every device runs it, and the tests hold everything else to it. Kernels are CubeCL, written once
//! for every CubeCL runtime (`cuda`, `rocm`, `wgpu` and its `vulkan`, `metal` and `webgpu`, `cpu`);
//! where a kernel does not take an input (a dtype, a head width), or the device lacks what it asks
//! (shared memory, units per cube, a plane width, a dtype), the reference runs instead. On
//! `Autodiff` the forward records one node whose backward is written out, keeping what it needs
//! rather than the reference's intermediates. Each operation module has a `pub mod probe` with
//! `fn kernel_launches() -> usize` and, where the backward is written out,
//! `fn backward_runs() -> usize` (and finer counters where useful), so that tests can show which
//! path ran; the probes are for tests, not part of the API.
//!
//! ## Tests
//!
//! Values are recorded from the implementation the operation matches (PyTorch, transformers,
//! ...) on seeded inputs and committed under `tests/fixtures/<op>/`. Errors are stated as
//! `max |got - want|` and as that over `max |want|`, per output and gradient, each asserted just
//! above the largest measured over every feature set the tests run with: the default (Flex),
//! `cuda`, `cuda` with `fusion`, `cuda` with `autotune`, and `cuda` with both; on other CubeCL
//! runtimes where a machine offers them, `DASH_OPS_DEVICE` naming the device. Tests that read a
//! global counter or draw from Burn's process-global random generator serialise on a mutex;
//! inputs come from a seeded host generator, never from Burn's.

// `cubecl` only gathers what every CubeCL runtime needs; one of them names the devices.
#[cfg(all(
    feature = "cubecl",
    not(any(feature = "cuda", feature = "rocm", feature = "wgpu", feature = "cpu"))
))]
compile_error!(
    "turn on a CubeCL runtime: `cuda`, `rocm`, `wgpu` (or `vulkan`, `metal`, `webgpu`) or `cpu`"
);

pub mod attention;
pub mod checkpoint;
pub mod conv1d;
#[cfg(feature = "cubecl")]
pub(crate) mod cube;
pub mod gated_delta;
pub mod logprob;
pub mod loss;
pub mod lowrank;
mod reference;
pub mod sequences;
