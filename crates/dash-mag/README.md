# dash-mag

The one crate an application names: the gateway to dash, which runs on this machine, and to the
`mag-` crates, planned and not started, which will pool devices across machines. A model and an
algorithm are written once and run on every platform: on Burn and dash-ops (CUDA, Flex) on Linux,
and on MLX and dash-mlx on macOS, chosen at compile time.

Part of [Dashmag](https://github.com/bayanasar/dashmag). See the repository for the full picture.

> **Status:** pre-alpha. The Burn side only; the macOS side is not written yet.
