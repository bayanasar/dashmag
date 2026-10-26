//! LoRA for Burn: a linear layer whose base weight is frozen at low precision (bf16) while the
//! low-rank adapters A and B are full precision (f32) and alone receive gradients. The adapter
//! is never folded into the base weight, in training or in inference: rounding the update into
//! bf16 would lose most of it. Adapters are read and written in PEFT's format
//! (`adapter_model.safetensors` and `adapter_config.json`), so they move between the two.
//!
//! - [`LoraLinear`] and [`Lora`] ([`linear`]): the layer, `y = x Wᵀ + scale · (x Aᵀ) Bᵀ`,
//!   with the precision of every part stated there.
//! - [`PeftAdapter`] and [`PeftConfig`] ([`peft`]): PEFT 0.20's files, read and written back
//!   unchanged; the caller says which of its layers each PEFT module path names.
//! - [`recompute`]: a stack of layers trained with each layer's forward pass run again in the
//!   backward pass, so that only the layers' inputs stay alive between the passes.

pub mod linear;
pub mod peft;
pub mod recompute;

pub use linear::{Lora, LoraLinear};
pub use peft::{LoraWeights, PeftAdapter, PeftConfig, PeftError};
