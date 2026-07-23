#!/usr/bin/env bash
# Compile the MobileNetV3-Small ONNX fixture to IREE .vmfb for CPU and CUDA.
# Requires iree-import-onnx + iree-compile (pip iree-base-compiler).
#   tools/compile_iree.sh [fixtures_dir]   ; IREE_CUDA_TARGET overrides sm_75
set -euo pipefail
DIR="${1:-fixtures}"
TARGET="${IREE_CUDA_TARGET:-sm_75}"
iree-import-onnx "$DIR/mnv3_legacy.onnx" -o "$DIR/mnv3.mlir"
iree-compile "$DIR/mnv3.mlir" --iree-hal-target-backends=llvm-cpu -o "$DIR/mnv3_cpu.vmfb"
iree-compile "$DIR/mnv3.mlir" --iree-hal-target-backends=cuda --iree-cuda-target="$TARGET" -o "$DIR/mnv3_cuda.vmfb"
echo "wrote $DIR/mnv3_cpu.vmfb and mnv3_cuda.vmfb (cuda target $TARGET)"
