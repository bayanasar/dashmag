#!/usr/bin/env python3
"""Export a torchvision model to IREE .vmfb straight from `torch.export`.

This is the integration contract's real path — **no ONNX anywhere**:

    torch.export -> ExportedProgram -> iree-turbine -> .vmfb

    python tools/export_torch.py [out_dir]        # default: fixtures

Requires: torch, torchvision, iree-turbine, iree-compile.

Env, as in `tools/compile_iree.sh`:
  IREE_CPU_TRIPLE    llvm-cpu triple (default: this host's own, from rustc —
                     never left implicit); `aarch64-unknown-linux-gnu` for a board
  IREE_CUDA_TARGET   CUDA arch (default sm_75; the Orin Nano is sm_87)

Every module is named after its target — `mnv3_torch_<arch>_cpu.vmfb`,
`mnv3_torch_<sm>_cuda.vmfb` — for the reason `compile_iree.sh` gives: a .vmfb is
per-target machine code, and a name without its target cannot be told apart on
the way to a board. The `torch` in the name separates these from the ONNX-path
modules of the same model, which call their entry point `main_graph`.

Three things this path needs that are easy to trip over, all validated here:

1. **Fold BatchNorm into the preceding conv first.**
   `aten._native_batch_norm_legit_no_training` reaches iree-turbine's importer
   as an opaque `torch.operator` and fails to legalize ("explicitly marked
   illegal"). Decomposing it instead trips a torch functionalization assertion.
   Conv+BN folding is exact in eval mode and removes the op entirely.

2. **`run_decompositions()` after `torch.export`.**
   Without it the in-place `aten.hardswish_` (MobileNetV3 uses
   `nn.Hardswish(inplace=True)`) fails verification: an in-place op needs a
   mutable tensor type but the importer supplies `!torch.vtensor`.

3. **The entry point is `main`, not `main_graph`.**
   `iree-import-onnx` names it `main_graph`; iree-turbine names it `main`.
   Pass `--function main` to `dash` / `iree-run-module`.
"""
import os
import subprocess
import sys
import urllib.request

import torch
from iree.turbine.aot import export as turbine_export
from PIL import Image
from torch.fx.experimental.optimization import fuse
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

out = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
os.makedirs(out, exist_ok=True)
CUDA_TARGET = os.environ.get("IREE_CUDA_TARGET", "sm_75")


def host_triple():
    r = subprocess.run(["rustc", "-vV"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("host: "):
            return line[len("host: "):]
    raise SystemExit("cannot determine the host triple (no rustc); set IREE_CPU_TRIPLE")


TRIPLE = os.environ.get("IREE_CPU_TRIPLE") or host_triple()

w = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=w).eval()
open(f"{out}/imagenet_classes.txt", "w").write("\n".join(w.meta["categories"]) + "\n")

img = f"{out}/dog.jpg"
if not os.path.exists(img):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", img)
x = w.transforms()(Image.open(img).convert("RGB")).unsqueeze(0).contiguous()
x.numpy().astype("<f4").tofile(f"{out}/input_1x3x224x224.f32")

with torch.no_grad():
    ref = model(x)
top = int(ref.argmax(1))
print(f"torch reference: class {top} logit {ref[0, top].item():.4f} => {w.meta['categories'][top]}")

# (1) conv+bn folding — exact in eval mode
fused = fuse(model, inplace=False)
with torch.no_grad():
    drift = (fused(x) - ref).abs().max().item()
print(f"conv+bn folded, max drift {drift:.2e}")

# (2) export + decompose, then lower to IREE with no ONNX hop
ep = torch.export.export(fused, (x,)).run_decompositions()
mlir = f"{out}/mnv3_torch.mlir"
turbine_export(ep).save_mlir(mlir)
print(f"  wrote {os.path.basename(mlir)} (target-independent)")


def iree_compile(out_file, *flags):
    subprocess.run(["iree-compile", mlir, *flags, "-o", out_file], check=True)
    print(f"  wrote {os.path.basename(out_file)}")


cpu_vmfb = f"{out}/mnv3_torch_{TRIPLE.split('-')[0]}_cpu.vmfb"
iree_compile(cpu_vmfb, "--iree-hal-target-backends=llvm-cpu",
             f"--iree-llvmcpu-target-triple={TRIPLE}")
cuda_vmfb = f"{out}/mnv3_torch_{CUDA_TARGET}_cuda.vmfb"
try:
    iree_compile(cuda_vmfb, "--iree-hal-target-backends=cuda",
                 f"--iree-cuda-target={CUDA_TARGET}")
except subprocess.CalledProcessError:
    # No stale module may stand in for this one, and the run must not exit 0
    # without it — the same rule as export_multihead.py.
    if os.path.exists(cuda_vmfb):
        os.remove(cuda_vmfb)
    print(f"  cuda {CUDA_TARGET} FAILED, {os.path.basename(cuda_vmfb)} not written")
    sys.exit(1)

print(f"\nrun it (entry point is 'main', see note 3):\n"
      f"  dash --backend iree --device local-task --function main \\\n"
      f"       --model {cpu_vmfb} --input {out}/input_1x3x224x224.f32 \\\n"
      f"       --labels {out}/imagenet_classes.txt --out-shape 1,1000")
