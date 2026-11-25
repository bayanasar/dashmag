# A list of machines

The smallest host a mag pool can have: you. There is no scheduler and no queue in front of the
pool. You write down which machines are in it, and you start the program on each, by hand or
over ssh, as `mpirun` runs from a hostfile. The target is one job's graph run across three
unequal machines: one with a CUDA card, one with a ROCm card, and a Mac with MLX.

What runs today is the first step, planning. You write the pool file, and `mag plan` shows where
the pool would place a graph on it, and why. Nothing is started on any machine and nothing
crosses a network: the program each machine runs (`mag-mar`) is not built yet.

## The files

- [`pool.json`](pool.json): the pool, written by hand. It has three machines, named by role
  (`workstation`, `server`, `laptop`), five devices and no costs. The format is `mag-by`'s
  [pool file](../../crates/mag-by/schemas/mag-pool.schema.json).
- The graph is the workspace's example,
  [`crates/mag-by/graphs/example.json`](../../crates/mag-by/graphs/example.json): a labelling job
  whose frames are kept at `owner`, a two-block model that may be divided, a classifier pinned to
  CUDA, and a count that alone is released.
- [`plan.txt`](plan.txt): what `mag plan` prints for the two. A test in `mag-mortar` and the CI
  check that the command still prints it.

## Running it

From the repository's root, with Rust:

    cargo run --locked -q -p mag-mortar --features files --example mag -- plan \
        examples/machine-list/pool.json crates/mag-by/graphs/example.json

or in Docker, on CPU, with nothing else installed:

    docker build -f examples/machine-list/Dockerfile -t mag-machine-list .
    docker run --rm mag-machine-list

Both print [`plan.txt`](plan.txt) and exit 0. Exit status 3 means the job would wait, and the
plan says what removed every place for it.

One container is enough for this step. Only the machine that coordinates plans, and the other
two would have nothing to run. Three containers come with the member program, when there is
something to start on each.

## Reading the plan

    decode  copies on server/1, workstation/1
    b0      on laptop/0, with chain model
    b1      on laptop/0, with chain model
    head    on workstation/0
    count   on server/1

- `decode` and `count` ask for the reference's bits (`exact`). Only the CPUs are offered as
  `exact` here, so they run there, one copy of `decode` on each.
- `head` is pinned to CUDA, so it runs on the workstation's card.
- The model's two blocks stay together, on the laptop. That is a tie, not a judgement: no cost
  is written, so every cost is the same guess, the laptop and the server score the same, and the
  first device in name order wins. The plan says so with its band, ±100%. Time your units and
  write the costs, and the blocks go where the costs say.
- Everything is `owner`, so the guard on kept data removes nothing. A rented machine would be
  written with `"trust": "outside"`, as in the workspace's
  [example pool](../../crates/mag-by/pools/example.json), and its plan shows the guard.

## What you fill in for your own machines

No host fills in the pool file here, so every field is yours to find out. For each machine, write
a member. For each device on it that you give the pool, write an offer.

| field | what it is | how you find it |
|---|---|---|
| member `name` | a name for the machine, used in every device's name and in the plan | any name. A role (`workstation`) keeps your hostnames out of files you share. |
| member `trust` | how far you trust the machine: `owner`, `site` or `outside`. Data kept at a level never goes to a machine trusted less. | yours. Your own machines are `owner`. A machine on your site that is not yours is `site`. A rented one is `outside`. |
| member `give_back_seconds` | how soon the machine gives its devices back when you take it back | yours: how long you will wait. The member program is to honour it; it is not built. |
| offer `device` | the member and an index, one per device | number each machine's devices from 0. Here the card is 0 and the CPU is 1. |
| offer `backend` | `cuda`, `rocm`, `mlx` or `flex` (the CPU) | the build of dash you run on that device |
| offer `memory_bytes` | the memory the device gives the pool: what is free of anything else holding it, in bytes | for a card, its total less what other programs keep (`nvidia-smi --query-gpu=memory.total,memory.used --format=csv`, `rocm-smi --showmeminfo vram`). On a Mac the card and the CPU share one memory (`sysctl -n hw.memsize` gives the total), so give the pool only what you can spare, and offer the CPU and MLX together no more than that. |
| offer `tolerances` | the numerics classes the device's measured bounds meet: `exact` (the reference's bits), `f32`, `bf16`, `int8` | dash's tests on that device, for the operations your parts use. Write `exact` only where you have seen the reference's bits. No command reports this yet: measuring it is the member program's, and not built. |
| offer `dtypes` | the element types the device computes in | those your build of dash runs on it, for the dtypes your graph's edges carry |
| offer `costs` | seconds per step of each unit on the device, each with a `band` and `estimate` | time one step of each unit yourself, at the batch the memory allows, after a warm-up and on an idle device. `band` is how far a later measurement may fall from it, relative. A cost you leave out is taken as the slowest any other device measured for that unit, or 1 s, and flagged as an estimate with a band of 100%. This file leaves all of them out, because the example's units are sketches with no code to time. |
| `at`, `jobs`, `formats` | the pool's time, the jobs it holds, the file versions it reads | copy them from [`pool.json`](pool.json): `0.0`, none, and the list as it is. A pool a coordinator writes out fills them itself. |

A graph names no machine, so nothing in it changes with your machines, except a `pin` that names
a member.

## Next

- The pool's certificate authority, and a certificate for each member. You make them once, on
  the machine you keep the pool from, from the repository's root:

      cargo run --locked -q -p mag-mar --features issue --example certs -- \
          new pool-certs workstation server laptop

  This runs today. It writes `authority.pem` and `authority.key` into `pool-certs`, and for each
  member `MEMBER.pem` and `MEMBER.key`, and prints each member's certificate serial. Keep
  `authority.key` where you run this, and give it to no member. Copy each member's two files to
  that machine, readable only by the user the program runs as. For a machine you add later, or
  one that comes back after you took it out, run the same with `add` in place of `new`: the
  same authority signs a new certificate, and the pool needs no other word of it.
- A start file for each of the three members, named by `MAG_START`: the member, the pool, the
  authority's certificate (the text of `authority.pem`), its peers and their addresses, which
  member coordinates, the ledger, the door, the idle time, and the paths of the member's
  certificate and key, as `mag-by`'s [start file](../../crates/mag-by/starts/example.json) has
  them.
- The member program on three containers on CPU: two jobs submitted to the coordinator's door
  and placed against each other, and one machine taken back with SIGTERM. This needs `mag-mar`.
- The same on three real machines with CUDA, ROCm and MLX.
