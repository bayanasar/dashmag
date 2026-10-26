# dash-burn

Burn backend for Dashmag. The model is built from code (`dash-models`), its weights are loaded from safetensors, and it runs on whichever Burn device the binary was built with: Flex (pure-Rust CPU), CUDA, ROCm or Metal — one runtime per device, behind the same `dash-core` trait as every other backend.

Part of [Dashmag](https://github.com/bayanasar/dashmag) — a machine-learning library in Rust
on Burn. See the repository for the full picture.

> **Status:** pre-alpha. APIs will change.

Licensed under the Apache License, Version 2.0.
