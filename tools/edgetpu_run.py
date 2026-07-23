#!/usr/bin/env python3
"""Edge TPU runner for dash-edgetpu (UNPROVEN starting point).
Runs an edgetpu-compiled int8 .tflite on the Coral TPU and writes dequantized
f32 output. Requires tflite_runtime + libedgetpu on the Coral.
    python3 edgetpu_run.py <model_edgetpu.tflite> <input.bin> <output.f32>
"""
import sys, numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

model, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
it = Interpreter(model_path=model,
                 experimental_delegates=[load_delegate("libedgetpu.so.1")])
it.allocate_tensors()
di, do = it.get_input_details()[0], it.get_output_details()[0]
x = np.fromfile(inp, dtype=di["dtype"]).reshape(di["shape"])
it.set_tensor(di["index"], x)
it.invoke()
y = it.get_tensor(do["index"]).astype(np.float32).reshape(-1)
scale, zp = do.get("quantization", (0.0, 0))
if scale:  # dequantize
    y = (y - zp) * scale
y.astype("<f4").tofile(outp)
