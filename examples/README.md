# Deployment examples

How a host gives a mag pool its machines. Each one is a recipe, not a part of the `mag-` crates:
the same library runs under every host. A recipe here says only what has run.

| recipe | host | what runs today |
|---|---|---|
| [`machine-list`](machine-list) | you: a list of machines written by hand, the program started by hand or over ssh | planning: a pool file for three machines and `mag plan` on it, with Rust or in Docker on CPU |
| Slurm | one Slurm job per machine | not yet |
