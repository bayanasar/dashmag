# dash-train

The modules training uses, in Rust on any Burn device. Each reproduces, as PyTorch computes it, its part of the recipe a PyTorch run used, and is checked against values recorded from PyTorch — so that a model trained with them differs from one trained by PyTorch by the noise of training, not by a different recipe:

- `loss`: masked, class-weighted cross-entropy over several heads, reduced as PyTorch's weighted mean.
- `optim`: AdamW as `torch.optim.AdamW` computes it, beta1 set per step; with the schedules attached, held to PyTorch over a whole schedule.
- `schedule`: `OneCycleLR` per parameter group, and beta1.
- `metrics`: macro-F1 per head, rounded as the recipe rounds it.
- `calibrate`: one class's logit shifted by the value best on validation, folded into its head's bias.

The crate still also holds a training loop for multi-head classifiers built on these modules, with its data handling, prediction and a command line. A training loop is the application's, not the library's: these parts are moving to the application that uses them. As they stand:

```text
dash-train cache     --trainset rows.jsonl --root DIR --meta model.json --out CACHE
dash-train train     --cache CACHE --meta model.json --init mobilenet_v3_small-047dcff4.pth --out RUN
dash-train calibrate --model RUN --cache CACHE --head HEAD --class CLASS --out RUN/calibrated
dash-train evaluate  --model RUN --cache CACHE --split val
dash-train predict   --model RUN --cache CACHE --split val --out predictions.jsonl
```

- `cache` decodes every frame once and resizes it whole to the model's input through `dash-image`, the preprocessing inference uses. The arrays are NumPy `.npy`, so a cache written by Python reads here and the other way round.
- `train` runs that recipe on the modules above: `OneCycleLR` per parameter group (backbone and heads), the epoch kept by validation macro-F1, rounded as the recipe rounds it. It writes `model.safetensors` (PyTorch's names) and `model.json`, the description `dash --meta` reads, for the best epoch, and the same two files for every epoch under `epochs/NN/`, so runs can also be compared at a fixed epoch.
- `calibrate` shifts one class's logit by the value best on validation and folds it into the head's bias.
- `predict` writes down what a model answers on each frame of one or more splits — `{"image", "predicted": {head: class}}`, a JSON line a frame — to be scored by whatever holds the truth. It writes every head's top-1, also where the description's `undefined_when` holds and `dash --meta` would print the output as undefined.

What is not reproduced — random streams, the crop's resize filter, batch norm's running variance — is stated in the crate docs.

Flex, the pure-Rust CPU device, is the default build; `--features cuda`, `rocm` or `metal` add a GPU.

Part of [Dashmag](https://github.com/bayanasar/dashmag) — a machine-learning library in Rust
on Burn. See the repository for the full picture.

> **Status:** pre-alpha. APIs will change.

Licensed under the Apache License, Version 2.0.
