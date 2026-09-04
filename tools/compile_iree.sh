#!/usr/bin/env bash
# Compile the MobileNetV3-Small ONNX fixture to IREE .vmfb for CPU and CUDA.
# Requires iree-import-onnx + iree-compile (pip iree-base-compiler).
#
#   tools/compile_iree.sh [fixtures_dir]
#
# Env:
#   IREE_CUDA_TARGET   CUDA arch          (default sm_75 — T4/RTX 2060 Turing;
#                                          the Orin Nano is sm_87)
#   IREE_CPU_TRIPLE    llvm-cpu triple    (default: the host's own)
#
# A .vmfb is per-target machine code, not a portable artifact: a host-triple CPU
# module will not run on aarch64, and an sm_75 module will not run on sm_87. Name
# the outputs after the target so the two cannot be confused on the way to a board.
set -euo pipefail
DIR="${1:-fixtures}"
TARGET="${IREE_CUDA_TARGET:-sm_75}"
TRIPLE="${IREE_CPU_TRIPLE:-}"

iree-import-onnx "$DIR/mnv3_legacy.onnx" -o "$DIR/mnv3.mlir"

if [ -n "$TRIPLE" ]; then
  CPU_OUT="$DIR/mnv3_${TRIPLE%%-*}_cpu.vmfb"
  iree-compile "$DIR/mnv3.mlir" --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple="$TRIPLE" -o "$CPU_OUT"
else
  CPU_OUT="$DIR/mnv3_cpu.vmfb"
  iree-compile "$DIR/mnv3.mlir" --iree-hal-target-backends=llvm-cpu -o "$CPU_OUT"
fi

CUDA_OUT="$DIR/mnv3_${TARGET}_cuda.vmfb"
iree-compile "$DIR/mnv3.mlir" --iree-hal-target-backends=cuda \
  --iree-cuda-target="$TARGET" -o "$CUDA_OUT"

echo "wrote $CPU_OUT (cpu triple ${TRIPLE:-host}) and $CUDA_OUT (cuda target $TARGET)"
