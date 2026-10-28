//! The contract of Dashmag's operations, written once for every implementation of them.
//!
//! For each operation, its module here says what it computes and what it matches (PyTorch,
//! transformers, fla, flash-attn), the shapes, layouts and dtypes it takes and returns, what its
//! masks and states mean and what each of its options does; and it holds the options themselves,
//! as plain data with their defaults. `dash-ops` implements the operations on Burn, and
//! re-exports these types where it has always had them; an implementation on another framework
//! implements the same contract with the same types, and `dash-conformance` runs the same tests
//! on both. The crate depends on nothing, so that any implementation can take it.
//!
//! - [`gated_delta`]: the gated delta rule of Gated DeltaNet, chunked for training and
//!   recurrent for decoding, over a batch or packed variable-length sequences.
//! - [`attention`]: attention with grouped key-value heads and the masks models train with, and
//!   decoding against a key-value cache.
//! - [`logprob`]: the log-probability of chosen tokens under a large vocabulary, with the
//!   entropy.
//! - [`loss`]: cross-entropy fused with the output layer over a large vocabulary, with label
//!   smoothing, z-loss, an ignore index and masked ids.
//! - [`conv1d`]: causal depthwise 1-D convolution, from a given state and over a batch or packed
//!   variable-length sequences, with the state a decoder carries.
//! - [`lowrank`]: low-rank products `base + scale · (x U) V` of small rank, as LoRA and the
//!   adapters built on it add to a frozen layer, with f32 factors over inputs of any float dtype.
//! - [`lora`]: what LoRA adapters in PEFT's format share whatever reads them: the name of the
//!   adapter PEFT makes by default, and how an adapter is given to a model's layers.
//! - [`sample`]: sampling the next token on the device, with temperature, top-k, top-p and
//!   masked ids, from a counter-based generator, and the token's log-probabilities.
//! - [`clip`]: gradients clipped by their global norm, as `torch.nn.utils.clip_grad_norm_`
//!   clips them, which training steps take beside the operations.
//!
//! # Conventions
//!
//! Every operation follows these, and new operations are held to them, so that one operation's
//! output feeds the next without a copy or a cast. Each module's documentation gives its shapes,
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
//! cumulative offsets `cu_seqlens` `[S + 1]` (Int, any int dtype) that start at 0 and do not
//! decrease (a sequence may be empty), as flash-attn and fla take them. Where the last one falls
//! is the operation's to say: the gated delta rule's and conv1d's sequences hold every token,
//! their last offset `T`; attention's may end before `T`, the positions past the last offset
//! belonging to no sequence. Inputs may have any strides (an implementation may copy one it
//! cannot read in place); outputs are contiguous in the documented layout.
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
//! are f32 whatever the inputs' dtype. f32 inputs are computed in IEEE f32, not TF32.
//!
//! ## Options
//!
//! Arguments with a usual default go in an options struct implementing `Default`, like
//! [`logprob::LogprobOptions`]; a softmax scale defaults to `1 / sqrt(D)`.
//!
//! ## What an implementation does not take
//!
//! A call outside an operation's contract (shapes that do not fit together, floating inputs of
//! more than one dtype, a state of the wrong length or dtype, an option out of its range, such
//! as a label smoothing outside `0..=1`) is refused before anything is computed, and so is a
//! call with an option or an input the contract defines but the implementation does not have, at
//! all or yet: the refusal names the operation and the option or input, and what is wrong with it
//! (`causal_conv1d: activation Silu is not implemented`). It is reported, never ignored: no
//! implementation drops an option or computes something else in its place. How it is reported is
//! the implementation's convention: `dash-ops`, as Burn does, panics with that message, and lists
//! each case under the function's `# Panics`; an implementation whose operations return a
//! `Result` returns it as its error. Within the contract, `dash-ops` takes every input and every
//! option: where one of its paths does not take an input (a kernel without the dtype, the head
//! width or what it asks of the device), another does, its reference wherever its kernels do not
//! run. A dtype the device cannot store is the device's: the call fails as the framework fails it,
//! and tests skip that dtype, saying so.

pub mod attention;
pub mod clip;
pub mod conv1d;
pub mod gated_delta;
pub mod logprob;
pub mod lora;
pub mod loss;
pub mod lowrank;
pub mod sample;

/// The float dtypes of the conventions, for what implementations share beside the operations
/// (the conformance tests' calls, which give each tensor's dtype); each implementation computes in
/// its own framework's dtypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE single precision.
    F32,
    /// bfloat16: f32's exponent, eight bits of significand.
    BF16,
    /// IEEE half precision.
    F16,
}
