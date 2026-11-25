# A list of machines

The smallest host a mag pool can have: you. There is no scheduler and no queue in front of the
pool. You write down which machines are in it, and you start the program on each, by hand or
over ssh, as `mpirun` runs from a hostfile. The target is one job's graph run across three
unequal machines: one with a CUDA card, one with a ROCm card, and a Mac with MLX.

What runs today is the first step, planning. You write the pool file, and `mag plan` shows where
the pool would place a graph on it, and why. The program each machine runs (`mag-mar`) starts,
joins the pool at the coordinator, which records the join in the pool's ledger, and gives its
machine back on SIGTERM; and at the coordinator's door you submit a job to the running pool,
which places it and writes it to the ledger, and take a device back by need, which the pool
chooses and its member gives back. A coordinator killed and started again takes up its ledger,
and the members join it again. Nothing runs on any machine yet: running the parts placed on
a member is not built.

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
| member `name` | a name for the machine, used in every device's name, in the plan and in the machine's certificate | a DNS name in lower case: letters `a` to `z`, digits and `-`, in labels joined by `.`, the last not all digits; the pool file refuses any other. A role (`workstation`) keeps your hostnames out of files you share. |
| member `trust` | how far you trust the machine: `owner`, `site` or `outside`. Data kept at a level never goes to a machine trusted less. | yours. Your own machines are `owner`. A machine on your site that is not yours is `site`. A rented one is `outside`. |
| member `give_back_seconds` | how soon the machine gives its devices back when you take it back | yours: how long you will wait. The member program holds to it: on SIGTERM it exits 0 within it, whatever the other machines do. |
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
  that machine, the key readable only by the user the program runs as (`chmod 600`): a member
  refuses a key file its group or others can read, write or run. For a machine you add later, or
  one that comes back after you took it out, run the same with `add` in place of `new`: the
  same authority signs a new certificate, and the pool needs no other word of it.

  The members' certificates last 365 days, unless you give `--days N` before the directory, and
  the authority ten years. A pool that outlives a member's certificate loses that member: it
  cannot start on the certificate, and the other members refuse every new channel to it or from
  it as not the pool's, though channels open when it ended carry on until they close. Before
  then, issue that member a new certificate from the same authority,

      cargo run --locked -q -p mag-mar --features issue --example certs -- \
          add pool-certs server

  copy its two new files over the old on that machine, and start its program again. The other
  members need nothing new, since they trust the authority and not one certificate. When the
  authority itself ends, make a new pool with `new`. `certs help` says the same.
- A start file for each of the three members, named by `MAG_START`: the member, where it
  answers the others (`listen`, such as `0.0.0.0:7340`; leave it out for a machine behind
  address translation, which only dials), the pool, the authority's certificate (the text of
  `authority.pem`), its peers and their addresses, which member coordinates, the ledger, the
  door, the idle time, and the paths of the member's certificate and key, as `mag-by`'s
  [start file](../../crates/mag-by/starts/example.json) has them. The coordinator needs a
  `listen` address, and every other member's start file gives it as the coordinator's address,
  since every member dials it. The door is a path on the coordinator's machine, such as
  `door.sock`, at most 101 bytes long, since a Unix socket's path is short; a relative one is
  taken from the directory you start the program in.
- The member program on each machine, from the repository's root:

      MAG_START=server.json cargo run --release -q -p mag-mar

  This runs today, as far as it goes. The coordinator begins the ledger at the start file's
  `ledger`, or takes up the one there (below). Each other member reads its start file, binds its
  `listen` address, dials the coordinator until it answers, waiting 250 ms and then twice as long
  each time, up to 2 s, and joins; it says `joined the pool` once the coordinator has the join in
  its ledger. The coordinator records one entry
  in the ledger for each device the member offers, from the offers its start file's pool gives
  for it, so write an offer for each device there: a member with none is recorded nowhere.
  `kill -TERM` on a member ends it with status 0 within its `give_back_seconds`, once the
  coordinator has written its leave to the ledger. A machine you did not list in the
  coordinator's start file, issued a certificate later with `certs add`, joins too, and is
  recorded at `outside` whatever its own start file says. The ledger is one JSON object per line,
  as `mag-by`'s [ledger schema](../../crates/mag-by/schemas/mag-ledger.schema.json) has it.
- The door, on the coordinator's machine. With the three members running, ask the pool with
  the coordinator's start file, from the repository's root, where you started the coordinator:

      MAG_START=workstation.json cargo run --release -q -p mag-mar -- door status
      MAG_START=workstation.json cargo run --release -q -p mag-mar -- door submit \
          crates/mag-by/graphs/example.json
      MAG_START=workstation.json cargo run --release -q -p mag-mar -- door end label-frames
      MAG_START=workstation.json cargo run --release -q -p mag-mar -- door take-back '{}' 10

  This runs today, as far as it goes: the pool places the job and writes it to its ledger, and
  nothing runs on any machine. Each command prints the coordinator's answer as JSON, and a line
  for you on standard error. Run by hand on one machine, the three members on loopback, the
  pool file above in each start file:
  - `status` says `3 members, 5 devices, 0 jobs; wants no device; would give back no member`,
    and prints the pool as one virtual device: the three members at `owner`, and the devices as
    their start files offer them, in name order: `laptop/0` (MLX), `server/0` (ROCm), `server/1`
    (CPU), `workstation/0` (CUDA) and `workstation/1` (CPU). Its `asks` are what the pool asks
    of you: `wants`, the kind of device it most wants while a job waits, and `gives_back`, the
    members that have held no part for the coordinator's start file's `idle_seconds`, never the
    coordinator. You poll it and decide; the pool never calls you.
  - `submit` says `` `label-frames` is placed `` and exits 0, once the job is in the ledger. It
    prints the loop's record, every candidate with why any was removed, which is long, and the
    job's arrangement: the parts where [`plan.txt`](plan.txt) puts them, `decode` in copies on
    `server/1` and `workstation/1`, the model's blocks on `laptop/0`, `head` on `workstation/0`
    and `count` on `server/1`. A job that waits exits 3. With `--stamp stamp.json` after the
    graph, the stamp file holds your stamp, such as `{"by": "you", "priority_ceiling": 1}`.
  - `status` then says `1 jobs`, and shows `label-frames` with that arrangement, at priority 2,
    having waited 0 s.
  - `end label-frames` ends it, in the ledger before it answers, and `status` shows no job again.
    Ending it a second time is refused, with exit 4: the pool holds no job of that name.
  - The same graph with each of the model's two blocks at 20 GiB (`memory_bytes`
    21474836480), named `label-large`, waits, with exit 3: the chain's 40 GiB fits no device.
    `status` then says `wants a device {"memory_bytes":42949672960,"trust":"owner",
    "tolerances":["bf16"],"dtypes":["bf16"]}`: a device with 40 GiB for the chain, of a member
    trusted at `owner`, since the frames are kept there, meeting `bf16` and computing in it.
  - With `idle_seconds` 10 in the coordinator's start file, `status` 10 s after the members
    joined says `would give back server, laptop`, longest idle first; with `label-frames`
    placed on every member, `would give back no member`; and 10 s after it ended, `laptop,
    server`, equal in time and so in name order.
  - With no coordinator running, each command fails at once with exit 5.

  Taking a device back by need, also run by hand on one machine with the three members on
  loopback, before any job:
  - `take-back '{}' 10` asks for any device within 10 s. The laptop gives a device back within
    2 s, the server within 30 and the coordinator within 5, and the coordinator's own devices
    come last, so the pool gives up `laptop/0`: it writes its leave to the ledger, tells the
    laptop, which offers it no longer, and says `laptop/0 is given back`, exit 0. The laptop,
    its only device gone, says it stays a member until its lease ends; `kill -TERM` still ends
    it.
  - `take-back '{}' 1` is refused, with exit 4: `the devices of the kind asked for (server/0,
    server/1, workstation/0, workstation/1) are each of a member that takes longer than the
    deadline to give a device back`. A kind no device is, such as
    `'{"backend":"cuda","memory_bytes":68719476736}'`, is refused as `no device in the pool is
    of the kind asked for`. Nothing refused is written.
  - With `label-frames` then submitted and placed (`decode` in copies on `server/1` and
    `workstation/1`, the blocks on `server/0`, `head` on `workstation/0`, `count` on
    `server/1`), `take-back '{"backend":"flex"}' 60` asks for a CPU. Of the two, the
    coordinator's comes last, so the pool gives up `server/1`, and the loop places the job
    again: `decode`'s one copy and `count` on `workstation/1`. `status` then says `3 members, 3
    devices, 1 jobs`.

  The door answers once the member says it gave the device back, or at the deadline, when the
  device is yours whatever the member says. Taking a machine back by name is not the door's:
  send its program SIGTERM.
- Stopping the coordinator, and starting it again. The ledger is the pool's state, and the
  coordinator takes up the ledger it finds at its start file's `ledger`. Of the same version of
  `mag-mortar`, it replays it, each record checked against the one logged, and goes on writing
  it; a ledger that does not replay is refused, exit 2, and left as it is. Written by another
  version, it is kept beside its place as `ledger.jsonl.1`, and a new ledger begins that follows
  it, with the jobs it held submitted again. A last line a crash cut short is dropped. Those three
  are held by `mag-mar`'s tests. While no coordinator runs, the members run on and dial it again,
  and each door command fails at once with exit 5. Run by hand on one machine, the three members on loopback, with `label-frames`
  placed:
  - `kill -KILL` on the coordinator. `status` fails at once with exit 5, `no coordinator
    answers at the door`. The server and the laptop each say `the control channel ended ...; it
    runs on, and dials the coordinator again`, dial again 0.25 s later, and, refused, wait 0.5,
    1 and 2 s before the next dials.
  - The coordinator started again, the same start file and the same command. It says `takes up
    the ledger ledger.jsonl, which this version wrote: its 4 entries replayed, each record as
    logged, and appended to`, and `awaits` the laptop for 5 s and the server for 30 s. Each
    joins again (`` `server` joined again with 2 devices, at trust owner: 2 held already, 0
    joined, 0 left ``), and says `joined the pool`. Nothing is written to the ledger, and
    `status` prints what it printed before the kill, `label-frames` where it was.
  - `kill -TERM` on the server, the laptop and then the coordinator: each exits 0, and the
    ledger holds the five devices' leaves. `kill -TERM` is how you stop the coordinator for
    good: it writes its own devices' leave. Killed, it writes nothing more, and the next one
    awaits its members.

  The coordinator waits for a member no longer than its grace: a member that has not joined
  again within its `give_back_seconds`, and at least 5 s, from when the coordinator took up the
  ledger or found its channel gone, has its devices' leave written then (held by the tests).

  Starting the coordinator on another machine, with the ledger copied to its start file's place
  and the other members' start files giving its address, is what you do when the coordinator's
  machine is gone; that has not been tried yet.
- Three containers on CPU: two jobs submitted to the coordinator's door and placed against each
  other, and one machine taken back with SIGTERM. This needs the members to run the parts placed
  on them, in `mag-mar`.
- The same on three real machines with CUDA, ROCm and MLX.
