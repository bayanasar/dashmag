# Dashmag

**One integration. Every target. Native speed where the hardware allows it, a pure-Rust floor everywhere else.**

Dashmag is a runtime-agnostic edge ML inference framework in Rust. Bring a trained model, get correct
outputs on every target behind a single API — with performance that scales to whatever the silicon exposes.

> **Status:** pre-alpha, moving fast. APIs will change.

## Why

Edge hardware is fragmented. Every accelerator speaks its own dialect, and most frameworks make you pick
one and rewrite when you outgrow it. Dashmag doesn't. It dispatches each model to the best available
runtime for the target and falls back to a pure-Rust CPU backend everywhere else. Write the integration
once; correctness never depends on the silicon underneath.

- **Runtime-agnostic** — one Rust API over compiled-native and pure-Rust backends.
- **Correct everywhere** — deterministic outputs on every target, anchored by a pure-Rust CPU
  backend that builds wherever Rust does. (A `no_std` build for MCU-class targets is a design
  goal, not yet implemented.)
- **Native speed where it counts** — accelerator backends light up when the hardware exposes a usable path.
- **Bring your own model** — export it from PyTorch; a build-time step lowers it to each backend's
  artifact format. The runtime API is identical regardless of which backend runs it.

## How it works

The abstraction seam is **whole-model inference** — load a model, tensors in, tensors out — so backends of
very different shapes sit behind one trait. Each backend advertises its real capabilities, and a registry
selects the best admissible one per target, refusing rather than guessing when nothing fits. Backends are
feature-gated crates, so every target compiles only what it can build.

| Crate | Backend | Status |
|---|---|---|
| [`dash-core`](https://crates.io/crates/dash-core) | traits, tensors, capability negotiation, dispatch | — |
| [`dash-tract`](https://crates.io/crates/dash-tract) | pure-Rust CPU floor ([tract](https://github.com/sonos/tract)) | runs on x86 and aarch64 |
| [`dash-iree`](https://crates.io/crates/dash-iree) | compiled artifacts via [IREE](https://iree.dev) (CPU, CUDA) | runs on CPU and CUDA GPUs |
| [`dash-tensorrt`](https://crates.io/crates/dash-tensorrt) | NVIDIA TensorRT (Jetson GPUs) | runs on Jetson |
| [`dash-edgetpu`](https://crates.io/crates/dash-edgetpu) | Coral Edge TPU | not yet run on hardware |

The IREE, TensorRT and Edge TPU backends are currently **spikes** that drive their vendor tool as a
subprocess; each crate's docs name the in-process FFI shim that replaces it. They are labelled as such
in the source rather than presented as finished.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Contributing

Pre-alpha and moving fast — issues and design discussion welcome. Please open an issue before large PRs.
