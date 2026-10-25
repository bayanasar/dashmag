//! Training operations Burn does not have built in.
//!
//! Each operation is a backend extension (`burn::backend::backend_extension`): a trait whose
//! default body builds the operation from Burn's own tensor operations, which is the reference
//! every device can run and the tests hold everything else to; a CubeCL kernel on the CUDA
//! backend where the reference is too slow or holds too much memory; and an implementation for
//! `Autodiff` with the operation's own backward pass, so that training does not record the
//! reference's intermediate tensors. `tests/extension_pattern.rs` is the smallest complete
//! example of the three, and the shape each module here follows.
//!
//! - [`gated_delta`]: the gated delta rule of Gated DeltaNet, chunked for training and
//!   recurrent for decoding.
//! - [`attention`]: causal attention whose backward pass does not materialise the T x T scores.
//! - [`logprob`]: the log-probability of chosen tokens under a large vocabulary, without the
//!   n x V logits.
//! - [`conv1d`]: causal depthwise 1-D convolution, with the state a decoder carries.

pub mod attention;
pub mod conv1d;
pub mod gated_delta;
pub mod logprob;
