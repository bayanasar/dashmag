# dash-image

Image preprocessing for Dashmag models: JPEG decode, a bilinear resize that reproduces Pillow's byte for byte, and per-channel normalisation. One implementation, shared by everything that feeds a model.

Part of [Dashmag](https://github.com/bayanasar/dashmag) — a runtime-agnostic
edge ML inference framework in Rust. See the repository for the full picture.

> **Status:** pre-alpha. APIs will change.

Licensed under the Apache License, Version 2.0.
