#!/usr/bin/env python3
"""Edge TPU runner for dash-edgetpu (UNPROVEN — no hardware run yet).

Runs an edgetpu-compiled int8 .tflite on the Coral TPU and writes each output
as dequantized little-endian f32.

    python3 edgetpu_run.py <model_edgetpu.tflite> <input.bin> <out_prefix>

Writes <out_prefix>.0.bin, <out_prefix>.1.bin, ... one per model output, so
multi-output models (detectors) do not silently lose tensors.

Requires tflite_runtime + libedgetpu on the Coral.
"""
import sys
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

if len(sys.argv) != 4:
    sys.exit(__doc__)
model, inp, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]

it = Interpreter(model_path=model,
                 experimental_delegates=[load_delegate("libedgetpu.so.1")])
it.allocate_tensors()

di = it.get_input_details()[0]
x = np.fromfile(inp, dtype=di["dtype"])
expected = int(np.prod(di["shape"]))
if x.size != expected:
    sys.exit(f"input has {x.size} elements, model expects {expected} {tuple(di['shape'])}")
it.set_tensor(di["index"], x.reshape(di["shape"]))
it.invoke()

for i, do in enumerate(it.get_output_details()):
    y = it.get_tensor(do["index"]).astype(np.float32).reshape(-1)
    scale, zero_point = do.get("quantization", (0.0, 0))
    if scale:  # dequantize back to real values
        y = (y - zero_point) * scale
    y.astype("<f4").tofile(f"{out_prefix}.{i}.bin")
