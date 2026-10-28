# dash-peft

PEFT's LoRA adapter files, with no tensor library (its options, `ApplyOptions` and
`DEFAULT_ADAPTER`, are dash-contract's, re-exported): a directory holding
`adapter_model.safetensors` and `adapter_config.json`, as PEFT 0.20's `save_pretrained` writes
them, read and written back byte for byte. Tensors keep their dtype and bytes, tensors that are
not LoRA factors (`modules_to_save`, trainable tokens) are carried through, configuration fields
this crate does not interpret are kept, and the configuration is written as PEFT's
`json.dumps(indent=2, sort_keys=True)` prints it. Adapters whose forward pass is not plain LoRA
(DoRA, `lora_bias`, aLoRA and the other variants) are refused when read.

- `PeftConfig`: the configuration, with `r`, `lora_alpha`, rsLoRA, `rank_pattern` and
  `alpha_pattern`, and each module's scale as PEFT computes it.
- `PeftAdapter`: the files; each module's factors as host data (`LoraWeights`: dtype, shape and
  bytes), by PEFT's module path.
- `AdapterLayer`: what a backend's LoRA layer gives the adapter, so that `apply_with` gives the
  caller's layers an adapter under a name, trainable or frozen (PEFT's `adapter_name` and
  `is_trainable`), and `from_layers_named` reads it back out. The checks are made here, once for
  every backend: a module's shapes against its layer's, every module and tensor taken by some
  layer, each layer's scale and rank against the configuration.

`dash-lora` implements `AdapterLayer` for its Burn layer and re-exports this crate. The tests
(`tests/format.rs`) read and write back dash-lora's `peft_*` fixtures, recorded from PEFT 0.20, and
apply them to layers that keep host data.

Part of [Dashmag](https://github.com/bayanasar/dashmag). See the repository for the full picture.

> **Status:** pre-alpha. APIs will change.

Licensed under the Apache License, Version 2.0.
