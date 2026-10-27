# dash-lora

LoRA for Burn: a linear layer whose base weight is frozen at its own precision (bf16) while the
low-rank adapters A and B are f32 and alone receive gradients. An adapter is never folded into
the base weight, in training or inference: rounded into bf16, most of a trained update is lost
(83% of its elements round to zero). A layer holds any number of adapters by name over its one
base weight, and each forward pass uses one of them, or none. Adapters are read and written in
PEFT's format (`adapter_model.safetensors` and `adapter_config.json`, a directory per adapter);
the mapping from PEFT's module paths to the caller's layers is the caller's.

- `LoraLinear`, `Lora`: the layer; the base product at the weight's dtype, the low-rank product in
  f32, their sum rounded once, as PEFT computes it. On a CubeCL device the low-rank products are
  `dash_ops::lowrank`'s kernels, which add the base and round in the same pass. Per layer at a
  language model's projection shapes (r 16, 2048 and 6144 tokens, `examples/lora_layer.rs`), on an
  otherwise idle RTX 5070 Ti against PEFT by device time: the low-rank products within 1.1x of
  PEFT's and mostly ahead in every build; a layer's forward and backward, or a step of one adapter
  over another's frozen output, with Burn's `fusion` and `autotune` 0.63x to 0.94x of PEFT's over
  6144 tokens and 0.78x to 1.12x over 2048; with Burn's default kernels 0.83x to 1.09x where
  Burn's default bf16 matrix product is near cuBLAS, and 2.2x to 9.1x where it is not (the base
  product of a 1024 -> 256 projection, and of every projection over 6144 tokens), which autotune
  fixes. These are the passes the example runs back to back before its timed ones: in the timed
  passes, each followed by a sync, the device is busy too little of the time, and an idle card
  lowers its clocks within two seconds of them (memory 13.8 to 7.0 GHz), PyTorch's passes losing
  more than these.
- `AdapterChoice`: `model.set_adapter(Some(name))` makes every layer of any Burn module use its
  adapter `name` from then on, and `set_adapter(None)` the base alone (PEFT's `set_adapter` and
  `disable_adapter()`), without the model's code passing anything through its forward pass. The
  choice is part of the module value: a clone choosing another adapter shares every tensor and
  leaves the original's choice as it was, and the recomputation of `dash_ops::checkpoint` runs the
  adapter its forward pass ran. Whether an adapter trains is its own, not the choice's: training
  against a frozen reference adapter, as a KL penalty does, holds both over one base, and the
  reference records nothing and gets no optimizer state. A layer with an adapter chosen computes
  bit for bit what a layer holding that adapter alone computes. Against PEFT's `set_adapter` and
  `disable_adapter()` on a two-block model holding two adapters, and a step of cross-entropy plus
  a KL penalty against the frozen one: in f32, logits within 3.5e-7 of their largest value and
  the trained adapter's gradients within 6.1e-7; with a bf16 base, logits bit-identical on Flex
  and one bf16 unit apart at most on CUDA, gradients within 2.9e-6 and 1.1e-3. A layer holding
  two adapters keeps its base once: at a 1024 -> 3584 projection with r 16, 7.56 MiB of an RTX
  5070 Ti where two layers take 14.56 MiB, as with PEFT (`examples/lora_layer.rs`).
- `PeftAdapter`, `PeftConfig`: PEFT's files, read and written back byte for byte;
  `apply_with` gives the caller's layers an adapter under a name, trainable or frozen (PEFT's
  `adapter_name` and `is_trainable`), and `from_layers_named` takes it back.

Activation checkpointing (a stack's forward pass run outside autodiff, each layer run again in the
backward pass, as `torch.utils.checkpoint` around each block does) lives in `dash-ops`, as
`dash_ops::checkpoint`, for any model. `examples/recompute_memory.rs` measures it here: over 6144
tokens through 24 LoRA'd MLP blocks at Qwen3.5-0.8B's shapes it peaks 0.69 GiB above the weights,
against 7.2 GiB without it.

Part of [Dashmag](https://github.com/bayanasar/dashmag) — a machine-learning library in Rust
on Burn. See the repository for the full picture.

Features as in `dash-ops`: `flex` (default); the CubeCL runtimes `cuda`, `rocm`, `wgpu` (and
`vulkan`, `metal`, `webgpu`) and `cpu`, whose device the tests run on beside Flex
(`DASH_OPS_DEVICE` naming one); and opt-in `autotune`.

    cargo test -p dash-lora
    CUBECL_CUDA_TF32=0 cargo test -p dash-lora --features cuda
    DASH_OPS_DEVICE=wgpu:vulkan cargo test -p dash-lora --features dash-ops/wgpu,dash-lora/wgpu
