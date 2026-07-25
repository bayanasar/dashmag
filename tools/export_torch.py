#!/usr/bin/env python3
"""Export a torchvision model to IREE .vmfb straight from `torch.export`.

This is the integration contract's real path — **no ONNX anywhere**:

    torch.export -> ExportedProgram -> iree-turbine -> .vmfb

    python tools/export_torch.py [out_dir]        # default: fixtures

Requires: torch, torchvision, iree-turbine.

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
import sys, os, urllib.request, torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.fx.experimental.optimization import fuse
from iree.turbine.aot import export as turbine_export
from PIL import Image

out = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
os.makedirs(out, exist_ok=True)

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
exported = turbine_export(ep)
for backend, name in (("llvm-cpu", "mnv3_torch_cpu.vmfb"), ("cuda", "mnv3_torch_cuda.vmfb")):
    try:
        exported.compile(save_to=f"{out}/{name}", target_backends=[backend])
        print(f"  {backend:9} -> {name}")
    except Exception as e:
        print(f"  {backend:9} FAILED: {str(e).splitlines()[0][:120]}")

print(f"\nrun it (entry point is 'main', see note 3):\n"
      f"  dash --backend iree --device local-task --function main \\\n"
      f"       --model {out}/mnv3_torch_cpu.vmfb --input {out}/input_1x3x224x224.f32 \\\n"
      f"       --labels {out}/imagenet_classes.txt --out-shape 1,1000")
