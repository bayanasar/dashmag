# mlx-c, as Apple publishes it

[mlx-c](https://github.com/ml-explore/mlx-c) 0.8.0, commit
`a26f2d584a14ec619ae8db9a09be03becd47598a`, whose CMake configuration pins MLX 0.32.4. MIT-licensed
(`LICENSE`).

Only what the build reads is kept: `mlx/`, `CMakeLists.txt`, `mlx-c.pc.in`, `LICENSE` and
`ACKNOWLEDGMENTS.md`, each unchanged. `crates/dash-mlx/build.rs` compiles it against MLX and
generates the Rust bindings from its headers.

To move to another version, replace these files with that commit's, change `MLX_C_COMMIT` in
`build.rs` and the commit above, and run dash-mlx's tests with a prebuilt MLX and with MLX built
from source. An upgrade is its own change.

## A known problem in this version

mlx-c hands an exception's message to `vsnprintf` as the format (`mlx/c/error.cpp`,
[upstream's issue 138](https://github.com/ml-explore/mlx-c/issues/138)), and MLX's message for a
Metal kernel that fails to compile quotes the kernel's source. On an Apple M6 with MLX 0.32.4
(2026-11-01), a kernel failing to compile on a line holding `%d` reported that `%d` as a number, so
`%n` or `%s` on such a line, or `% n` or `% s` since a space is a flag, would write or read memory.
Only a failed compile reaches this; the kernels here compile. The next upgrade of mlx-c should check whether it is fixed.
