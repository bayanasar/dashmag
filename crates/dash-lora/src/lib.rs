//! LoRA for Burn: a linear layer whose base weight is frozen at low precision (bf16) while the
//! low-rank adapters A and B are full precision (f32) and alone receive gradients. The adapters
//! are never folded into the base weight, in training or in inference: rounding an update into
//! bf16 would lose most of it. Adapters are read and written in PEFT's format
//! (`adapter_model.safetensors` and `adapter_config.json`), so they move between the two.
//!
//! - [`LoraLinear`] and [`Lora`] ([`linear`]): the layer, `y = x Wᵀ + scale · (x Aᵀ) Bᵀ`,
//!   with the precision of every part stated there.
//! - [`AdapterChoice`] ([`adapters`]): several adapters over one copy of the base weights, by
//!   name, and the choice, made for a whole model at once without its code passing anything
//!   through its forward pass, of the one each layer uses, or of none. Each adapter trains or is
//!   frozen on its own: a KL penalty against the adapter training started from holds both over
//!   one base, the reference frozen.
//! - [`PeftAdapter`] and [`PeftConfig`] ([`peft`]): PEFT 0.20's files, read and written back
//!   unchanged, one directory per adapter; the caller says which of its layers each PEFT module
//!   path names. The format is [`dash_peft`]'s, re-exported; [`LoraLinear`] is one of its
//!   [`AdapterLayer`](peft::AdapterLayer)s.
//!
//! Activation checkpointing, which trains a stack of LoRA'd layers with each layer's forward pass
//! run again in the backward pass so that only the layers' inputs stay alive between the passes,
//! is [`dash_ops::checkpoint`], for any model; `examples/recompute_memory.rs` measures it on
//! LoRA'd MLP blocks.

pub mod adapters;
pub mod linear;
pub mod peft;

pub use adapters::{AdapterChoice, DEFAULT_ADAPTER};
pub use linear::{Lora, LoraLinear};
pub use peft::{ApplyOptions, LoraWeights, PeftAdapter, PeftConfig, PeftError};
