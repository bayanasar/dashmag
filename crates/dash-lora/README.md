# dash-lora

LoRA for Burn: a linear layer whose base weight is frozen at its own precision (bf16) while the
low-rank adapters A and B are f32 and alone receive gradients. The adapter is never folded into
the base weight, in training or inference: rounded into bf16, most of a trained update is lost
(83% of its elements round to zero). Adapters are read and written in PEFT's format
(`adapter_model.safetensors` and `adapter_config.json`); the mapping from PEFT's module paths to
the caller's layers is the caller's.

- `LoraLinear`, `Lora`: the layer; the base product at the weight's dtype, the low-rank product in
  f32, their sum rounded once, as PEFT computes it.
- `PeftAdapter`, `PeftConfig`: PEFT's files, read and written back byte for byte.
- `recompute`: a stack's forward pass run outside autodiff, each layer run again in the backward
  pass, as `torch.utils.checkpoint` around each block does. Over 6144 tokens through 24 LoRA'd MLP
  blocks at Qwen3.5-0.8B's shapes it peaks 0.7 GiB above the weights, against 7.1 GiB without it.

Part of [Dashmag](https://github.com/bayanasar/dashmag) — a machine-learning library in Rust
on Burn. See the repository for the full picture.

Features as in `dash-ops`: `flex` (default), `cuda`, and opt-in `autotune`.

    cargo test -p dash-lora
    CUBECL_CUDA_TF32=0 cargo test -p dash-lora --features cuda
