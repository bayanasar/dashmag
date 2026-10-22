# dash-train

Train a multi-head classifier in Rust, on any Burn device, to the recipe a PyTorch run used — so that a model trained here differs from one trained there by the noise of training, not by a different recipe.

```text
dash-train cache     --trainset rows.jsonl --root DIR --meta model.json --out CACHE
dash-train train     --cache CACHE --meta model.json --init mobilenet_v3_small-047dcff4.pth --out RUN
dash-train calibrate --model RUN --cache CACHE --head HEAD --class CLASS --out RUN/calibrated
dash-train evaluate  --model RUN --cache CACHE --split val
```

- `cache` decodes every frame once and resizes it whole to the model's input through `dash-image`, the preprocessing inference uses. The arrays are NumPy `.npy`, so a cache written by Python reads here and the other way round.
- `train` reproduces, each part checked against values recorded from PyTorch: masked, class-weighted cross-entropy reduced as PyTorch's weighted mean; AdamW as `torch.optim.AdamW` computes it, beta1 set per step; `OneCycleLR` per parameter group (backbone and heads); the epoch kept by validation macro-F1, rounded as the recipe rounds it. It writes `model.safetensors` (PyTorch's names) and `model.json`, the description `dash --meta` reads.
- `calibrate` shifts one class's logit by the value best on validation and folds it into the head's bias.

What is not reproduced — random streams, the crop's resize filter, batch norm's running variance — is stated in the crate docs.

Flex, the pure-Rust CPU device, is the default build; `--features cuda`, `rocm` or `metal` add a GPU.

Part of [Dashmag](https://github.com/bayanasar/dashmag) — a runtime-agnostic
edge ML inference framework in Rust. See the repository for the full picture.

> **Status:** pre-alpha. APIs will change.

Licensed under the Apache License, Version 2.0.
