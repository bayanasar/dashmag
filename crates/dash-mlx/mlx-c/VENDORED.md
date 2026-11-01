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
