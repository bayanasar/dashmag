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
//! - [`sample`]: sampling the next token on the device, with temperature, top-k, top-p and
//!   masked ids, from a counter-based generator, and the token's log-probabilities.
//! - [`checkpoint`]: activation checkpointing, not an operation but built on Burn's tensor API:
//!   a stack of layers run outside autodiff, each layer run again in the backward pass.
//! - [`sequences`]: packed variable-length sequences, as the operations that take them share
//!   them.
//!
//! # Conventions
//!
//! Every operation follows the conventions of [`dash_contract`], which every implementation of
//! these operations shares: the layout, masks, dtypes and options, and how a call outside an
//! operation's contract is refused (here, by a panic with the operation's name, listed under
//! `# Panics`). Each operation's contract (its definition, shapes, dtypes and options) is written
//! in the module of the same name there, whose option types this crate re-exports where they
//! have always been; each module here documents its implementation. f32 inputs are computed in
//! IEEE f32: on CUDA with `CUBECL_CUDA_TF32=0`, without which Burn 0.22 runs f32 matmuls in TF32.
//! New code is held to those conventions and to these:
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
pub mod sample;
pub mod sequences;
