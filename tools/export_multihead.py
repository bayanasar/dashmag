#!/usr/bin/env python3
"""Export a five-headed toy model, because every hardware run so far had one output.

    python tools/export_multihead.py [out_dir]     # default: fixtures

Phase 1's model (alcuka/ai-nerekh#3) is one backbone with five independent
attribute heads, and until this fixture existed no test in the repo declared more
than one output — the cross-backend conformance test, the thing that stands behind
"one codebase, correct everywhere", had only ever compared output 0.

The weights of the heads are random and the model predicts nothing. That is the
point: what is being tested is whether an artifact with five outputs survives
torch.export, the IREE compiler, the runtime's buffer handling and the CLI, in
declaration order, on every target. The head widths are the real ones from
`nerekh/tools/calib/vocab.py` — 4, 6, 6, 4, 4 — so the shape is not a toy even
though the numbers are.

Writes, into `out_dir`:

    mh5.mlir                  the exported program, target-independent
    mh5_<arch>_cpu.vmfb       llvm-cpu module for this host's triple
    mh5_<cuda target>_cuda.vmfb
    mh5.onnx                  the same model for tract, which has no torch frontend
    mh5_input_1x3x64x64.f32   the input blob
    mh5_reference.json        torch's own outputs, to catch a fixture that changed

Naming follows `tools/compile_iree.sh`: a .vmfb is per-target machine code and no
artifact is written whose target is not in its name.
"""
import json
import os
import platform
import subprocess
import sys

import torch
from iree.turbine.aot import export as turbine_export

OUT = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
HEADS = {"time_of_day": 4, "weather": 6, "road_type": 6, "surface": 4, "traffic": 4}
SHAPE = (1, 3, 64, 64)  # small on purpose: this fixture tests plumbing, not compute
CUDA_TARGET = os.environ.get("IREE_CUDA_TARGET", "sm_75")


class FiveHeads(torch.nn.Module):
    """One trunk, five classifier heads. Deliberately tiny and deliberately not MNv3:
    a fixture that takes minutes to compile is a fixture nobody runs on a board."""

    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(32, n) for n in HEADS.values()]
        )

    def forward(self, x):
        f = self.trunk(x)
        # A tuple return is what makes this five outputs rather than one stacked
        # tensor: the heads have different widths and must stay separable.
        return tuple(h(f) for h in self.heads)


def host_triple():
    out = subprocess.run(["rustc", "-vV"], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("host: "):
            return line[len("host: "):]
    raise SystemExit("cannot determine the host triple (no rustc); set it by hand")


def iree_compile(mlir, out_file, *flags):
    subprocess.run(["iree-compile", mlir, *flags, "-o", out_file], check=True)
    print(f"  wrote {os.path.basename(out_file)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)  # a fixture that changes under you is worse than no fixture
    model = FiveHeads().eval()
    x = torch.randn(*SHAPE)
    x.numpy().astype("<f4").tofile(f"{OUT}/mh5_input_1x3x64x64.f32")

    with torch.no_grad():
        ref = model(x)
    print(f"{len(ref)} outputs: " + ", ".join(
        f"{name}{tuple(t.shape)}" for name, t in zip(HEADS, ref)))
    json.dump(
        {"heads": list(HEADS), "shapes": [list(t.shape) for t in ref],
         "argmax": [int(t.argmax()) for t in ref],
         "values": [[round(v, 6) for v in t.flatten().tolist()] for t in ref]},
        open(f"{OUT}/mh5_reference.json", "w"), indent=1)

    ep = torch.export.export(model, (x,)).run_decompositions()
    exported = turbine_export(ep)
    mlir = f"{OUT}/mh5.mlir"
    exported.save_mlir(mlir)
    print(f"  wrote {os.path.basename(mlir)}")

    triple = host_triple()
    iree_compile(mlir, f"{OUT}/mh5_{triple.split('-')[0]}_cpu.vmfb",
                 "--iree-hal-target-backends=llvm-cpu",
                 f"--iree-llvmcpu-target-triple={triple}")
    try:
        iree_compile(mlir, f"{OUT}/mh5_{CUDA_TARGET}_cuda.vmfb",
                     "--iree-hal-target-backends=cuda",
                     f"--iree-cuda-target={CUDA_TARGET}")
    except subprocess.CalledProcessError as e:
        print(f"  cuda {CUDA_TARGET} FAILED: {e}")

    # tract has no torch frontend, so this fixture reaches it the way the MobileNet
    # one does — through ONNX, which architecture.md quarantines rather than blesses.
    torch.onnx.export(model, (x,), f"{OUT}/mh5.onnx",
                      input_names=["input"], output_names=list(HEADS), dynamo=False)
    print("  wrote mh5.onnx")
    print("\nrun it (entry point is 'main' for turbine artifacts):\n"
          f"  dash --backend tract --model {OUT}/mh5.onnx \\\n"
          f"       --input {OUT}/mh5_input_1x3x64x64.f32 --shape 1,3,64,64 \\\n"
          + "".join(f"       --out-shape 1,{n} \\\n" for n in HEADS.values())
          + "       # one --out-shape per head, in declaration order")


if __name__ == "__main__":
    main()
