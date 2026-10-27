# Dashmag

**A machine-learning library in Rust, on [Burn](https://burn.dev).**

Dashmag is a general machine-learning library in Rust. Burn beneath it supplies tensors, autodiff
and devices; Dashmag adds what a model's life needs around them — architectures, training, the
operations training needs that Burn does not have, inference on whatever hardware is at hand, and
exits to the compilers of accelerators Burn cannot drive. Together they are meant to do for Rust
what PyTorch does for Python. Each part is held to values recorded from the implementation it
replaces (PyTorch, torchvision, Pillow, transformers, PEFT), and each test says where its
recording comes from.

> **Status:** pre-alpha, moving fast. APIs will change.

## What is here

| Crate | What it is | Status |
|---|---|---|
| [`dash-models`](https://crates.io/crates/dash-models) | model architectures as Burn modules, named so that published PyTorch weights load unchanged; MobileNetV3-Small with any number of heads | torchvision's weights give torch's answer on Burn's CPU backend and on CUDA |
| [`dash-train`](https://crates.io/crates/dash-train) | training: masked, class-weighted multi-head cross-entropy, AdamW under a one-cycle schedule, calibration, prediction | reproduces a PyTorch recipe's arithmetic; trains on the CPU and on CUDA |
| [`dash-ops`](https://crates.io/crates/dash-ops) | training operations Burn lacks, each with a written-out backward pass and CUDA kernels: the gated delta rule, causal attention that keeps no T x T scores, chosen-token log-probabilities over a large vocabulary, causal depthwise conv1d | agrees with PyTorch and transformers on the CPU and on CUDA |
| [`dash-lora`](https://crates.io/crates/dash-lora) | LoRA with a frozen low-precision base and f32 adapters, several over one base, one chosen per forward for a whole model, read and written in PEFT's format | one layer bit-identical to PEFT; two adapters over one base held to PEFT's `set_adapter` |
| [`dash-image`](https://crates.io/crates/dash-image) | image preprocessing: JPEG decode, Pillow-exact bilinear resize, normalisation | byte-exact with Pillow's resize |
| [`dash-core`](https://crates.io/crates/dash-core) | the inference trait: tensors, capability negotiation, dispatch across runtimes | — |
| [`dash-burn`](https://crates.io/crates/dash-burn) | Burn behind that trait: the model built from code, weights loaded, run on Flex (pure-Rust CPU), CUDA, ROCm or Metal | runs on x86-64 (Flex) and on CUDA |
| [`dash-tract`](https://crates.io/crates/dash-tract) | pure-Rust CPU runtime ([tract](https://github.com/sonos/tract)) | runs on x86 and aarch64 |
| [`dash-iree`](https://crates.io/crates/dash-iree) | compiled artifacts via [IREE](https://iree.dev) (CPU, CUDA) | runs on CPU and CUDA GPUs |
| [`dash-tensorrt`](https://crates.io/crates/dash-tensorrt) | NVIDIA TensorRT (Jetson GPUs) | runs on Jetson |
| [`dash-edgetpu`](https://crates.io/crates/dash-edgetpu) | Coral Edge TPU | not yet run on hardware |
| [`dash-cli`](https://crates.io/crates/dash-cli) | `dash`, a command-line runner over every runtime above | — |

`PROGRESS.md` says what has been measured, on which hardware, and what has not run yet.

## How a model gets here

- **As code.** A model is a Burn module (`dash-models`), trained by `dash-train` and run by
  `dash-burn`, or by `dash --backend burn`, on any Burn device. Published PyTorch weights for a
  `dash-models` architecture load directly, and `dash-import` converts a PyTorch checkpoint into
  the safetensors file `dash-burn` reads.
- **As an artifact.** tract and TensorRT run an ONNX file, IREE a module compiled from one
  (`tools/compile_iree.sh`), the Edge TPU an int8 TFLite file. Today the ONNX files are exported
  from PyTorch (`tools/export_*.py`); writing them from a Burn module in Rust is planned, so that
  both ways start from the same model.

## Inference across runtimes

The inference seam is **whole-model inference** — load a model, tensors in, tensors out — so
runtimes of very different shapes sit behind one trait. Each runtime advertises its real
capabilities, and a registry selects the best admissible one per target, refusing rather than
guessing when nothing fits. Runtimes are feature-gated crates, so every target compiles only what
it can build, and a conformance test holds them to each other.

The IREE, TensorRT and Edge TPU runtimes are currently **spikes** that drive their vendor tool as a
subprocess; each crate's docs name the in-process FFI shim that replaces it. They are labelled as
such in the source rather than presented as finished.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Contributing

Pre-alpha and moving fast — issues and design discussion welcome. Please open an issue before large PRs.
