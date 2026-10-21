//! Model architectures as Burn modules.
//!
//! A model is a Rust type here, not a file: the graph is the code, and only weights move
//! between machines (as safetensors, or a published PyTorch checkpoint read directly).
//! The same module is trained and run, so there is no exported format to prove equal to
//! the model it came from.
//!
//! Parameter paths follow the published PyTorch definitions — `features.3.block.1.0.weight`
//! is the same tensor in both — so published weights load through burn-store's PyTorch
//! reader without a conversion table. Getting the names right is not enough to get the
//! numbers right; each architecture's module docs list the constants that differ from
//! Burn's defaults and fail silently when missed.

pub mod mobilenet_v3;

pub use mobilenet_v3::MobileNetV3Small;
