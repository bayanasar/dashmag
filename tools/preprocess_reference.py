#!/usr/bin/env python3
"""Write Pillow's side of the preprocessing comparison for one JPEG.

    python tools/preprocess_reference.py fixtures/dog.jpg [H W]     # default 224 224

Next to the image, for `<stem>`:

    <stem>.preprocess.json   the decoded size and the output size
    <stem>_rgb.u8            Pillow's decoded RGB, interleaved, row-major
    <stem>_HxW.u8            Pillow's `resize((W, H), Image.BILINEAR)` of it
    <stem>_HxW.f32           that, scaled to [0, 1], ImageNet-normalised, NCHW

`dash-cli`'s preprocessing tests read these: the resize is held to Pillow byte for
byte on Pillow's own decoded pixels, and the whole JPEG-to-tensor path is held to
the `.f32` within the decoder tolerance it states.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

path = sys.argv[1]
h, w = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (224, 224)
stem = os.path.splitext(path)[0]

img = Image.open(path).convert("RGB")
small = img.resize((w, h), Image.BILINEAR)
x = (np.asarray(small, dtype=np.float32) / 255.0 - MEAN) / STD

open(f"{stem}_rgb.u8", "wb").write(img.tobytes())
open(f"{stem}_{h}x{w}.u8", "wb").write(small.tobytes())
x.transpose(2, 0, 1).astype("<f4").tofile(f"{stem}_{h}x{w}.f32")
json.dump({"width": img.width, "height": img.height, "out": [h, w]},
          open(f"{stem}.preprocess.json", "w"))
print(f"{path}: {img.width}x{img.height} -> {w}x{h}")
