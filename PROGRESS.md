# PROGRESS — dashmag

<!-- progress -->
updated: 2026-10-20
owner: bayanasar
stage: one model, correct on every target tried; the next step moves models and training onto Burn
<!-- /progress -->

## What this is
A runtime-agnostic inference framework: one `dash-core` trait, several backends (tract,
IREE, TensorRT, Edge TPU), and a cross-backend conformance test that holds them to each
other. The claim is "one model source, correct everywhere", with a pure-Rust CPU floor.

Where it is going: the model defined once in Rust on Burn, trained and run by dashmag, with
ONNX and int8 tflite written from Rust only as exits to NPU vendors' compilers.

## Sealed
On `main`, and re-runnable by anyone who checks it out.

- tract runs MobileNetV3-Small to class 258 @ logit 11.7283 on x86-64, Coral (Cortex-A53), Jetson Nano (A57), Jetson TX2, Orin Nano (A78AE) — `dash` from one static aarch64-musl binary on each board.
- IREE agrees with tract per logit (1e-3) on x86-64 CPU, Tesla T4, RTX 2060, Orin Nano CPU and GPU (sm_87) — conformance test, `DASH_IREE_REQUIRE_CUDA=1` on the board run.
- Five-output models agree across tract, IREE CPU/CUDA and TensorRT on the Orin Nano — five-head conformance test, at the earlier fixture widths (4, 6, 6, 4, 4).
- Five-output models at the deployed widths (3, 3, 3, 2, 2) agree across tract and IREE CPU on x86-64 — `cargo test -p dash-cli --features iree -- --ignored`, 2 of 2 conformance tests green on `main` (2026-10-20), fixtures from `tools/export_multihead.py`.
- `--image` resizes exactly as Pillow does — byte-for-byte on Pillow's decoded pixels (2 synthetic cases, a public test image, 30 real 1280x854 frames); decode differs from libjpeg-turbo by at most 2 levels after resize. The code is the `dash-image` crate; `cargo test -p dash-image -- --include-ignored` runs the synthetic and public-image cases.

## Not sealed
- five-head at the deployed widths on CUDA and on a board — missing; only the x86-64 CPU leg has run at these widths. From here this is measured on Burn, not IREE.
- `--image` on a board — missing; a decoder or resize difference on aarch64 would surface here first.
- Edge TPU with more than one output — never run (no int8 `.tflite` exists).
- tract 0.23 on the boards — the board results above ran tract 0.21. On x86-64, 0.23.9 returns the same bits as 0.21.17 for the MobileNetV3 and five-head fixtures; its aarch64 kernels have not run on a board.
- `no_std` floor — not built at all; the MCU claim is unproven.

## Decisions
- 2026-09-04 On NVIDIA silicon TensorRT is the speed tier and IREE the portability tier — because IREE's kernels run at 8-9% of SM compute against TensorRT's 63% on the same GPU, and closing that needs hand-written tuning specs per shape.
- 2026-10-13 Commits carry a GitHub noreply address and merges are fast-forward only — because a forge-made merge commit publishes the merging account's address, which a pull-request check cannot see.
- 2026-10-13 zune-jpeg is built without SIMD — so the host and every board decode through the same scalar code.
- 2026-10-20 Models are defined, trained and run in Rust on Burn; Python, torch.export, iree-turbine and IREE retire; ONNX and int8 tflite become exits written from Rust for TensorRT and the Edge TPU compiler — because one definition that both trains and runs has no format crossing to prove equal, and every board result so far had to be proven across one. The torch.export-on-a-board item is retired with it, not pending.
- 2026-10-20 tract moves from 0.21 to 0.23 — because tract 0.21.11 to 0.21.19 pin `half` to exactly 2.4.1 while Burn 0.22 needs 2.7.1 or later; one lockfile cannot hold both, and cargo's only other answer was to fall back to tract 0.21.10.

## Open
- Next: `dash-core` running on Burn (CPU, CUDA, ROCm, Metal). Image preprocessing is already one crate, `dash-image`, for training and inference to share; the workspace is on Rust 1.95 and edition 2024, which Burn 0.22 needs.
- IREE stays until Burn covers its targets, then is archived; tract stays only as an independent check over the exported ONNX.
- `main` is not protected on the forge; the no-direct-push rule is a convention.
