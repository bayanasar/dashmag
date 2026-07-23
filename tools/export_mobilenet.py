#!/usr/bin/env python3
"""Generate Phase-0 fixtures: MobileNetV3-Small (ImageNet) as tract-parseable
ONNX + a preprocessed Samoyed input + the ImageNet label list.

Requires torch + torchvision (CPU is fine). Usage:
    python tools/export_mobilenet.py [out_dir]   # default: ./fixtures

NOTE: uses the LEGACY opset-13 TorchScript ONNX exporter on purpose — torch's
newer dynamo exporter (opset 18) emits ops tract 0.21 cannot parse.
"""
import sys, os, urllib.request, torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

out = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
os.makedirs(out, exist_ok=True)

w = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=w).eval()
open(f"{out}/imagenet_classes.txt", "w").write("\n".join(w.meta["categories"]) + "\n")

img_path = f"{out}/dog.jpg"
if not os.path.exists(img_path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", img_path)

x = w.transforms()(Image.open(img_path).convert("RGB")).unsqueeze(0).contiguous()
x.numpy().astype("<f4").tofile(f"{out}/input_1x3x224x224.f32")
with torch.no_grad():
    top = int(model(x).argmax(1))
print(f"torch top-1: class {top} => {w.meta['categories'][top]}  (expect 258 Samoyed)")

torch.onnx.export(model, (x,), f"{out}/mnv3_legacy.onnx",
                  input_names=["input"], output_names=["logits"],
                  opset_version=13, dynamo=False)
print(f"wrote fixtures to {out}/")
