# Dashmag

**A machine-learning library in Rust, on [Burn](https://burn.dev).**

Dashmag is a general machine-learning library in Rust. Burn beneath it supplies tensors, autodiff
and devices; Dashmag adds the compute layer that applications build on, and inference. Together
they are meant to do for Rust what PyTorch does for Python.

The compute layer is the operations Burn does not have, in forward, backward and decoding-step
forms with kernels for Burn's backends; layers, such as LoRA; autodiff utilities, such as
activation checkpointing; and the modules training uses (losses, optimisers, learning-rate
schedules). Each is built for a family of models (transformers for language, convolutional
networks), not shaped to one, and each operation is benchmarked on its own, at the shapes its
workload gives it. Inference runs trained models, a whole model in one forward call, tensors in
and tensors out, on whatever hardware is at hand, with exits to the compilers of accelerators Burn
cannot drive. Each part is held to values recorded from the implementation it replaces (PyTorch,
torchvision, Pillow, transformers, PEFT), and each test says where its recording comes from.

What sits above both is the application's, written on top of Dashmag: modules and training code
shaped to its own model, its training loop and training algorithms (PPO, GRPO and their kind), the
generation loop that calls a model step by step, samples, keeps each sequence's state and decides
when to stop, and its task harness. For a generation loop Dashmag gives the decoding-step
operations, such as attention against a key-value cache and the gated delta rule's recurrent step.

> **Status:** pre-alpha, moving fast. APIs will change.

## What is here

| Crate | What it is | Status |
|---|---|---|
| [`dash-models`](https://crates.io/crates/dash-models) | model architectures as Burn modules, named so that published PyTorch weights load unchanged; MobileNetV3-Small with any number of heads | torchvision's weights give torch's answer on Burn's CPU backend and on CUDA |
| [`dash-train`](https://crates.io/crates/dash-train) | the modules training uses: masked, class-weighted multi-head cross-entropy, AdamW under a one-cycle schedule, calibration; the training loop, data handling, prediction and command line it still holds are moving to the application that uses them | reproduces a PyTorch recipe's arithmetic; trains on the CPU and on CUDA |
| [`dash-ops`](https://crates.io/crates/dash-ops) | operations Burn lacks, for training and decoding, each with a reference, a written-out backward pass and CubeCL kernels for any runtime whose device offers what they ask: attention with the masks models train with, keeping no T x T scores; the gated delta rule and causal depthwise conv1d, over packed sequences and from given states; chosen-token log-probabilities and cross-entropy fused with the output layer over a large vocabulary; the low-rank product LoRA's layers use; activation checkpointing | agrees with PyTorch, transformers and fla on the CPU, on CUDA, through wgpu and on CubeCL's CPU runtime |
| [`dash-contract`](https://crates.io/crates/dash-contract) | the contract of `dash-ops`' operations, written once for every implementation of them: what each computes and matches, its shapes, layouts, dtypes, masks and states, and its options as plain data, with no framework beneath it | — |
| [`dash-conformance`](crates/dash-conformance) | the operation-level tests of `dash-contract`'s operations, written once for every implementation of them and run through a thin adapter: the recorded fixtures, the error measures and the bounds; conv1d, the gated delta rule, the chosen-token log-probability and attention so far (not published) | dash-ops passes them on Flex, on CUDA, through wgpu and on CubeCL's CPU runtime |
| [`dash-lora`](https://crates.io/crates/dash-lora) | LoRA with a frozen low-precision base and f32 adapters, several over one base, one chosen per forward for a whole model, read and written in PEFT's format | one layer bit-identical to PEFT; two adapters over one base held to PEFT's `set_adapter` |
| [`dash-peft`](https://crates.io/crates/dash-peft) | PEFT's LoRA adapter files with no tensor library: read and written back byte for byte, and applied to a backend's layers and read out of them through one trait | dash-lora's Burn layer implements the trait; PEFT's files written back byte for byte |
| [`dash-mlx`](crates/dash-mlx) | dash-ops' and dash-lora's operations on MLX, with their contracts, for Apple silicon; so far the three MLX functions they build on, its checkpoint, custom VJP and Metal kernels, reachable from Rust | held to plain MLX operations; not published |
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

- **As code.** A model is a Burn module (`dash-models`), trained by the application that uses it,
  with Dashmag's operations and training modules, and run by `dash-burn`, or by
  `dash --backend burn`, on any Burn device. Published PyTorch weights for a
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
