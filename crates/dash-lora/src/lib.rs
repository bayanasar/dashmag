//! LoRA for Burn: a linear layer whose base weight is frozen at low precision (bf16) while the
//! low-rank adapters A and B are full precision (f32) and alone receive gradients. The adapter
//! is never folded into the base weight, in training or in inference: rounding the update into
//! bf16 would lose most of it. Adapters are read and written in PEFT's format
//! (`adapter_model.safetensors` and `adapter_config.json`), so they move between the two.
