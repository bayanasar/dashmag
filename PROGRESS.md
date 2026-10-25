# PROGRESS — dashmag

<!-- progress -->
updated: 2026-10-24
owner: bayanasar
stage: one model, correct on every target tried; models run on Burn and train in Rust (`dash-train`), held to PyTorch's arithmetic and, on data this repository does not hold, no more than 3 standard errors below its results, one cell of ten unresolved
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
- Burn's MobileNetV3-Small (`dash-models`) is torchvision's: torchvision's published ImageNet checkpoint, read directly by burn-store, gives 258 @ 11.728306 on the recorded input on Flex (x86-64), and every one of the 1000 logits is within 2.6e-5 of tract on that input and within 2.2e-5 through `dash-image` — `cargo test -p dash-models --features pytorch -- --ignored`, with the checkpoint fetched as the test's docs say. Through the `dash-core` trait too: the same weights written to safetensors and loaded by `dash-burn` as an artifact, chosen by the registry, give the same 258 @ 11.728306 and 2.6e-5 against tract — `cargo test -p dash-burn -- --ignored`; without fixtures, `cargo test -p dash-burn` holds the artifact path to the module bit for bit.
- `dash --meta model.json` names every output, prints an output whose undefined condition holds as undefined instead of answering, and changes no logit bit — `cargo test -p dash-cli --test meta`, on the five-head fixture. `dash --backend burn` builds the model the description names, loads its safetensors, and returns the module's logits bit for bit — `cargo test -p dash-cli --features burn --test burn`; the same binary builds as a static aarch64-musl `dash`.
- `dash-train` reproduces a PyTorch recipe's arithmetic — `cargo test -p dash-train`, 21 green on x86-64 (2026-10-24): the masked, class-weighted cross-entropy and AdamW with beta1 changed per step within 1e-6 of values recorded from torch 2.13; OneCycleLR's rate for two parameter groups and beta1 within 1e-12 relative of torch's at all 1875 steps of a recorded run; the optimiser and both schedules together, as a run steps them, fed the same synthetic gradients as `torch.optim.AdamW` under OneCycleLR for all 1875 steps, end within 2.4e-7 of it, one f32 step, where decay after the update or beta1 a step late would show at 2.3e-5 and 1.4e-5; validation runs on the inference module, bit for bit the logits of the saved weights loaded fresh; every epoch's weights are kept beside the best.
- A seed fixes a `dash-train` run's weights on Flex, whatever the thread count — `cargo test -p dash-train --test epochs`: two epochs from scratch on a synthetic cache at 1 and at 4 threads, every tensor bit-identical, in batches of 16 so that the neck's matrix product splits across the pool as well as the convolutions. It holds for one run per process: Flex draws initial weights and dropout masks from one generator per process, and two runs at once interleave their draws. The safetensors files can still differ byte for byte, because Burn writes their metadata keys in hash order.
- `dash-train predict` writes down a model's answers: each head's top-1 class for every frame of a cache's split, in the cache's order, the heads in the model's order — `cargo test -p dash-train --test predict`, against the module's own predictions, and the writer's unit test.
- `--image` resizes exactly as Pillow does — byte-for-byte on Pillow's decoded pixels (2 synthetic cases, a public test image, 30 real 1280x854 frames); decode differs from libjpeg-turbo by at most 2 levels after resize. The code is the `dash-image` crate; `cargo test -p dash-image -- --include-ignored` runs the synthetic and public-image cases.

## Not sealed
- five-head at the deployed widths on CUDA and on a board — missing; only the x86-64 CPU leg has run at these widths. From here this is measured on Burn, not IREE.
- `--image` on a board — missing; a decoder or resize difference on aarch64 would surface here first.
- Edge TPU with more than one output — never run (no int8 `.tflite` exists).
- tract 0.23 on the boards — the board results above ran tract 0.21. On x86-64, 0.23.9 returns the same bits as 0.21.17 for the MobileNetV3 and five-head fixtures; its aarch64 kernels have not run on a board.
- Burn on a board, and on CUDA — missing. The binary the boards are to run — the static aarch64-musl `dash` with `--features burn`, Flex with its SIMD kernels and threads, tract beside it — passes R0 under qemu's user mode emulating a Cortex-A53 (2026-10-24): 258 @ 11.728302 on the recorded input and 13.128326 on `dog.jpg` through `dash-image`, every logit within 2.2e-5 of tract in the same binary. That checks the aarch64 code paths and the A53's instruction set, not a board. The CUDA, ROCm and Metal builds compile; none has run. Burn 0.22 runs f32 matmuls and convolutions in TF32 on CUDA GPUs of compute capability 8.0 and later; the workspace's patched cubecl-cuda turns that off with `CUBECL_CUDA_TF32=0`, which compiles and has not run.
- `dash-train predict` answers as PyTorch does — missing: a run anyone can repeat, on public data. On data this repository does not hold, fed the pixels PyTorch's pipeline decoded, all 128125 of its top-1 answers are PyTorch's: five heads on each of 25625 frame predictions, the 4015 frames of two evaluation sets under six models (five trained by `dash-train`, one by PyTorch) as trained and calibrated (2026-10-24) — so those models' scores no longer need PyTorch to compute.
- `no_std` floor — not built at all; the MCU claim is unproven.
- `dash-train` trains as well as PyTorch did — missing: a comparison anyone can re-run, on public data. On data this repository does not hold: `dash-train evaluate` and `calibrate` reproduce a PyTorch-trained model's validation report and calibration exactly (2026-10-21); five complete runs on Flex fall no more than 3 standard errors of the difference below five PyTorch runs of the same recipe, per head on two evaluation sets (2026-10-23); the test is one-sided, so it says dash-train is not worse by more than that, not that the two are equal. One of those ten cells passes only because the margin includes dash-train's own spread; it follows which epoch each run's selection kept, and a late drift on one head is unresolved at five runs a side. Those runs kept only their best epoch. Both trainers now keep every epoch, so new runs of each can be compared at a fixed epoch, where selection does not enter.

## Decisions
- 2026-09-04 On NVIDIA silicon TensorRT is the speed tier and IREE the portability tier — because IREE's kernels run at 8-9% of SM compute against TensorRT's 63% on the same GPU, and closing that needs hand-written tuning specs per shape.
- 2026-10-13 Commits carry a GitHub noreply address and merges are fast-forward only — because a forge-made merge commit publishes the merging account's address, which a pull-request check cannot see.
- 2026-10-13 zune-jpeg is built without SIMD — so the host and every board decode through the same scalar code.
- 2026-10-20 Models are defined, trained and run in Rust on Burn; Python, torch.export, iree-turbine and IREE retire; ONNX and int8 tflite become exits written from Rust for TensorRT and the Edge TPU compiler — because one definition that both trains and runs has no format crossing to prove equal, and every board result so far had to be proven across one. The torch.export-on-a-board item is retired with it, not pending.
- 2026-10-20 A Burn model's weights are stored as safetensors in PyTorch's convention (PyTorch's parameter names and layouts) — because such a file is also a PyTorch `state_dict` that any tool reads, and Burn's own record format stopped loading across 0.21 to 0.22.
- 2026-10-20 tract moves from 0.21 to 0.23 — because tract 0.21.11 to 0.21.19 pin `half` to exactly 2.4.1 while Burn 0.22 needs 2.7.1 or later; one lockfile cannot hold both, and cargo's only other answer was to fall back to tract 0.21.10.
- 2026-10-20 TF32 is turned off for comparisons by a patched cubecl-cuda (`patches/cubecl-cuda`, one hunk, opt-in through `CUBECL_CUDA_TF32=0`), not by widening tolerances for it — because every agreement recorded here is per logit at f32 precision, and TF32 rounds each matmul input to 10 mantissa bits. Upstream has the request (https://github.com/tracel-ai/burn/issues/6140); the patch goes when a Burn release has a setting.

- 2026-10-21 Training runs on Flex, the CPU device, until a GPU is free — because a full run takes about 95 minutes on ten cores, and the one other GPU in reach, an RDNA2 card under WSL2, has no ROCm: AMD's WSL support starts at RDNA3.
- 2026-10-24 `dash-models` names no Flex features; `dash-burn` and `dash-train` turn on its SIMD kernels and threads — because the crate that defines the model stays device-agnostic, and which kernels run is decided where the device is chosen.

## Open
- Depthwise convolution's backward on Flex is 1.35 s of a 3.0 s training step against 0.13 s for its forward (`cargo run --release -p dash-train --example depthwise_backward`): Flex has a depthwise path for the forward only. The input gradient goes through its transposed convolution, a GEMM per image and channel with an inner dimension of 1; the weight gradient through Burn's generic grouped path, a serial loop over every channel. A depthwise backward kernel in Flex would take most of it; not started.
- Next on Burn: the CUDA leg, waiting on a free GPU, run with `CUBECL_CUDA_TF32=0`; and Flex on an aarch64 board (it passes on an emulated Cortex-A53). Trained PyTorch weights come in through `dash-import` (checkpoint to `model.safetensors`, strict), and `tests/recorded.rs` holds an imported model to logits its old framework recorded. Image preprocessing is already one crate, `dash-image`, for training and inference to share; the workspace is on Rust 1.95 and edition 2024, which Burn 0.22 needs.
- IREE stays until Burn covers its targets, then is archived; tract stays only as an independent check over the exported ONNX.
- `main` is not protected on the forge; the no-direct-push rule is a convention.
