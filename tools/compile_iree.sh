#!/usr/bin/env bash
# Compile the MobileNetV3-Small ONNX fixture to IREE .vmfb for CPU and CUDA.
# Requires iree-import-onnx + iree-compile (pip iree-base-compiler).
#
#   tools/compile_iree.sh [fixtures_dir]
#
# Env:
#   IREE_CUDA_TARGET   CUDA arch          (default sm_75 — T4/RTX 2060 Turing;
#                                          the Orin Nano is sm_87)
#   IREE_CPU_TRIPLE    llvm-cpu triple    (default: this host's own, taken from
#                                          rustc or llvm-config — never left
#                                          implicit)
#
# A .vmfb is per-target machine code, not a portable artifact: a host-triple CPU
# module will not run on aarch64, and an sm_75 module will not run on sm_87. Every
# output is named after the target it was built for, the host default included —
# an unqualified mnv3_cpu.vmfb built on x86_64 and one built on aarch64 are
# different machine code under one name, and the file gives no way to tell them
# apart on the way to a board.
set -euo pipefail
DIR="${1:-fixtures}"
TARGET="${IREE_CUDA_TARGET:-sm_75}"

# Resolve the host triple rather than letting iree-compile pick one silently:
# the point of the naming rule is that no artifact exists whose target is
# unknown, and an implicit default is exactly such an artifact.
host_triple() {
  if command -v rustc >/dev/null 2>&1; then
    rustc -vV | sed -n 's/^host: //p'
  elif command -v llvm-config >/dev/null 2>&1; then
    llvm-config --host-target
  fi
}
TRIPLE="${IREE_CPU_TRIPLE:-$(host_triple)}"
if [ -z "$TRIPLE" ]; then
  echo "cannot determine the host llvm-cpu triple (no rustc, no llvm-config)." >&2
  echo "set IREE_CPU_TRIPLE=<triple> explicitly." >&2
  exit 1
fi

iree-import-onnx "$DIR/mnv3_legacy.onnx" -o "$DIR/mnv3.mlir"

CPU_OUT="$DIR/mnv3_${TRIPLE%%-*}_cpu.vmfb"
iree-compile "$DIR/mnv3.mlir" --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple="$TRIPLE" -o "$CPU_OUT"

CUDA_OUT="$DIR/mnv3_${TARGET}_cuda.vmfb"
iree-compile "$DIR/mnv3.mlir" --iree-hal-target-backends=cuda \
  --iree-cuda-target="$TARGET" -o "$CUDA_OUT"

echo "wrote $CPU_OUT (cpu triple $TRIPLE) and $CUDA_OUT (cuda target $TARGET)"
