# Dashmag

**A machine-learning library in Rust, on [Burn](https://burn.dev).**

Dashmag is a general machine-learning library in Rust. Burn beneath it supplies tensors, autodiff
and devices; Dashmag adds the compute layer that applications build on, and inference. Together
they are meant to do for Rust what PyTorch does for Python.

Dashmag is two families of crates. dash is the operations, the same on every platform: the
`dash-` crates are the library, on the hardware of one machine. mag is the pathway from one
algorithm to whatever devices are there, on every platform: the `mag-` crates, just begun, are to
make one pool of the machines a person has and place work across them. Three crates make it
([below](#mag-one-pool-from-the-machines-you-have)). dash + mag = Dashmag.

The compute layer is the operations Burn does not have, in forward, backward and decoding-step
forms with kernels for Burn's backends; layers, such as LoRA; autodiff utilities, such as
activation checkpointing; and the modules training uses (losses, optimisers, learning-rate
schedules). Each is built for a family of models (transformers for language, convolutional
networks), not shaped to one, and each operation is benchmarked on its own, at the shapes its
workload gives it. Inference runs trained models, a whole model in one forward call, tensors in
and tensors out, on whatever hardware is at hand, with exits to the compilers of accelerators Burn
cannot drive. Each part is held to values recorded from the implementation it replaces (PyTorch,
torchvision, Pillow, transformers, PEFT), and each test says where its recording comes from.

What sits above both is the application's, written on top of Dashmag: modules and training code
shaped to its own model, its training loop and training algorithms (PPO, GRPO and their kind), the
generation loop that calls a model step by step, samples, keeps each sequence's state and decides
when to stop, and its task harness. For a generation loop Dashmag gives the decoding-step
operations, such as attention against a key-value cache and the gated delta rule's recurrent step.

> **Status:** pre-alpha, moving fast. APIs will change.

## What is here

| Crate | What it is | Status |
|---|---|---|
| [`dash-models`](crates/dash-models) | model architectures as Burn modules, named so that published PyTorch weights load unchanged; MobileNetV3-Small with any number of heads | torchvision's weights give torch's answer on Burn's CPU backend and on CUDA |
| [`dash-train`](crates/dash-train) | a training loop for multi-head classifiers on Burn modules, with `dash-mag`'s training modules and the Burn-tensor loss and AdamW it still needs; the loop, data handling, prediction and command line are moving to the application that uses them | reproduces a PyTorch recipe's arithmetic; trains on the CPU and on CUDA |
| [`dash-ops`](crates/dash-ops) | operations Burn lacks, for training and decoding, each with a reference, a written-out backward pass and CubeCL kernels for any runtime whose device offers what they ask: attention with the masks models train with, keeping no T x T scores; the gated delta rule and causal depthwise conv1d, over packed sequences and from given states; chosen-token log-probabilities and cross-entropy fused with the output layer over a large vocabulary; the low-rank product LoRA's layers use; activation checkpointing; and 2-D convolution, Burn's own under the contract's checks | agrees with PyTorch, transformers and fla on the CPU, on CUDA, on ROCm, through wgpu and on CubeCL's CPU runtime; 2-D convolution with torch on Flex, CUDA and ROCm |
| [`dash-contract`](crates/dash-contract) | the contract of `dash-ops`' operations, written once for every implementation of them: what each computes and matches, its shapes, layouts, dtypes, masks and states, and its options as plain data, with no framework beneath it | — |
| [`dash-conformance`](crates/dash-conformance) | the operation-level tests of `dash-contract`'s operations, written once for every implementation of them and run through a thin adapter: the recorded fixtures, the error measures and the bounds; conv1d, 2-D convolution, the gated delta rule, the chosen-token log-probability and attention so far (not published) | dash-ops passes them on Flex, on CUDA, through wgpu and on CubeCL's CPU runtime; 2-D convolution's on Flex, CUDA and ROCm |
| [`dash-mlx`](crates/dash-mlx) | dash-ops' operations on MLX, with their contracts, for Apple silicon: attention, conv1d, 2-D convolution (MLX's own, from torch's layout), the gated delta rule, chosen-token log-probabilities, the fused linear cross-entropy, the low-rank product, recompute and sampling, with Metal kernels where MLX's operations fall short | conv1d, the gated delta rule, the log-probability and attention pass dash-conformance's tests; the others are held to dash-ops' tests by ports of them; 2-D convolution held to torch's on an Apple M5; not published |
| [`dash-mag`](crates/dash-mag) | the one crate an application names, the gateway to dash and, as they are built, to the `mag-` crates (it does not reach `mag-by` yet): a tensor, its array operations, a vector-Jacobian product, the operations' entry points, LoRA's layer and PEFT's adapter files (`dash_mag::peft`, host data only), rotary embeddings, AdamW and clipping, a multi-head classifier's loss, batch normalisation, `OneCycleLR`, macro-F1 and calibration, over `dash-ops` or `dash-mlx`, the side chosen at compile time, and the device by its index before the first tensor. | on Burn and on MLX: every entry point the same bits as `dash-ops` or `dash-mlx`, one training step from one source held to one recording; LoRA's layer and AdamW held to PEFT's and torch's recordings; PEFT's files written back byte for byte; AdamW under `OneCycleLR` and clipping held to torch's on Flex; 2-D convolution and batch normalisation held to torch's on Flex, CUDA, ROCm and MLX |
| [`mag-by`](crates/mag-by) | the words the `mag-` crates share: the graph of a job's parts, its checks with typed refusals, the level of trust each part's data is kept at, and the pool's words (offers, arrangements, records, a pool's description and its ledger, the host's stamp on a job, taking a machine back, and a member's start file); with the `graph` feature, the files in JSON (the graph, the pool, the ledger's lines and the start file; version 1 files still read) with their JSON Schemas, and DOT for a person to look at; `no_std` with no dependencies unless `graph` is on | its tests hold the checks, the levels of trust, each file's refusals and the kept schemas and DOT to the code; CI builds the default for a `no_std` target with an empty dependency tree |
| [`mag-mortar`](crates/mag-mortar) | the placement library of a pool: the loop, which on each event lists the ways to place the waiting jobs, removes those that break a constraint (memory, tolerance, dtypes, a pin, a job's need, the devices the host's stamp allows, the guard on the levels of trust), scores the rest from the devices' measured costs and the jobs' priorities, commits the best and records every candidate; `describe` and `plan` for an author; the pool's state as a ledger its host saves and replays; `mag plan`, a command over files. It does no I/O, and runs on offers given as data | its tests hold each constraint, the guard, priorities and the record, the plan against the submit, the replay of every prefix of a run, and the command's kept output; no member, network or probe yet |
| [`mag-mar`](crates/mag-mar) | the program every member of a pool is to run; so far its link to the other members: one channel per job, over TCP with TLS 1.3, both ends holding certificates from the pool's own authority and refused unless signed by it and naming the member expected; `mag-by`'s words in each message's header, and a tensor's bytes streamed in its body; the rate measured on each link; `certs`, the command that makes the authority and the members' certificates | its tests run two members over loopback, each refusal, two jobs' channels and a 64 MiB tensor byte for byte; a 4 GiB tensor measured over loopback; no member program, coordinator or door yet |
| [`dash-image`](crates/dash-image) | image preprocessing: JPEG decode, Pillow-exact bilinear resize, normalisation | byte-exact with Pillow's resize |
| [`dash-core`](https://crates.io/crates/dash-core) | the inference trait: tensors, capability negotiation, dispatch across runtimes | — |
| [`dash-burn`](crates/dash-burn) | Burn behind that trait: the model built from code, weights loaded, run on Flex (pure-Rust CPU), CUDA, ROCm or Metal | runs on x86-64 (Flex) and on CUDA |
| [`dash-tract`](https://crates.io/crates/dash-tract) | pure-Rust CPU runtime ([tract](https://github.com/sonos/tract)) | runs on x86 and aarch64 |
| [`dash-iree`](https://crates.io/crates/dash-iree) | compiled artifacts via [IREE](https://iree.dev) (CPU, CUDA) | runs on CPU and CUDA GPUs |
| [`dash-tensorrt`](https://crates.io/crates/dash-tensorrt) | NVIDIA TensorRT (Jetson GPUs) | runs on Jetson |
| [`dash-edgetpu`](https://crates.io/crates/dash-edgetpu) | Coral Edge TPU | not yet run on hardware |
| [`dash-cli`](https://crates.io/crates/dash-cli) | `dash`, a command-line runner over every runtime above | — |

`PROGRESS.md` says what has been measured, on which hardware, and what has not run yet.

## How a model gets here

- **As code.** A model is a Burn module (`dash-models`), trained by the application that uses it,
  with Dashmag's operations and training modules, and run by `dash-burn`, or by
  `dash --backend burn`, on any Burn device. Published PyTorch weights for a
  `dash-models` architecture load directly, and `dash-import` converts a PyTorch checkpoint into
  the safetensors file `dash-burn` reads.
- **As an artifact.** tract and TensorRT run an ONNX file, IREE a module compiled from one
  (`tools/compile_iree.sh`), the Edge TPU an int8 TFLite file. Today the ONNX files are exported
  from PyTorch (`tools/export_*.py`); writing them from a Burn module in Rust is planned, so that
  both ways start from the same model.

## Inference across runtimes

The inference seam is **whole-model inference** — load a model, tensors in, tensors out — so
runtimes of very different shapes sit behind one trait. Each runtime advertises its real
capabilities, and a registry selects the best admissible one per target, refusing rather than
guessing when nothing fits. Runtimes are feature-gated crates, so every target compiles only what
it can build, and a conformance test holds them to each other.

The IREE, TensorRT and Edge TPU runtimes are currently **spikes** that drive their vendor tool as a
subprocess; each crate's docs name the in-process FFI shim that replaces it. They are labelled as
such in the source rather than presented as finished.

## mag: one pool from the machines you have

People who train and run models at home or in a small lab usually have several unequal
machines: a workstation with one GPU, a second box with another vendor's card, a laptop with its
own accelerator, now and then a rented cloud GPU for a night. Each is too small for some job, and
most of them are idle most of the time. mag is to make them one pool. A cluster under Slurm is
the same problem with more machines.

The work is divided as it is between Slurm and an MPI library:

- **The author** of an algorithm, often an AI agent, writes each part's function with dash's
  operations, and a graph of the job, which names no device. The graph says what the parts are (a
  function, the weights it holds, the memory it needs, where it can stop), what flows between them
  and how large it is, which parts may run as many copies, where a model may be divided between
  devices, and what the job needs: a rate or a deadline, and how much it weighs against other
  jobs.
- **A host** gives the pool its machines and starts the same program on each: a person with a
  list of machines, Slurm on a cluster, or a home system. It decides which machines are in the
  pool, and when one is taken back. It never reads a graph.
- **The pool**, which is mag, places every job submitted to it against the others: it queues them,
  weighs their priorities, decides which device each part runs on, which job yields when a more
  urgent one arrives, and places again when the host takes a machine back. One of the pool's
  machines coordinates; the host names it and starts it like any other.
- **dash** runs the operations on each device.

mag does not decide which machines it gets, does not wake them, and does not take them back:
that is the host's, as it is Slurm's and not MPI's. `mag-mortar` is the placement library the
coordinator runs; `mag-mar` is to be the program on every machine. mag keeps no server of its
own, and is the same code under every host.

### What mag covers, and what it leaves to the host

The placement algorithm is not yours to write: it is `mag-mortar`'s. Which machines are in the
pool is not mag's: it is the host's. What you write is each part's function, with dash's
operations, and the graph; on a list of machines, the pool file as well. What runs today is the
planning, not the running. In this table "is to" marks what is not built.

| thing | whose | on a list of machines | under Slurm | in a home system |
|---|---|---|---|---|
| each part's function | yours to write, with dash's operations; the graph names it as a `Unit` (a name and a version), and it is to be built into the program each machine runs; how a unit's code reaches a machine is not built, nor decided | the same | the same | the same |
| the job's graph | yours to write; `mag-by` reads and checks it (`Graph::from_json`, `Graph::check`, [`schemas/mag-graph.schema.json`](crates/mag-by/schemas/mag-graph.schema.json)) | the same | the same | the same |
| which machines are in the pool | **the host's** | you write the pool file; the [list-of-machines recipe](examples/machine-list) has one for three machines, and says where each field comes from | the allocation, one Slurm job per machine; turning it into a pool description is not built | that system's |
| the pool's description: its members, the level each is trusted at (`outside`, `site` or `owner`), how soon it gives a device back, and each device's backend, memory, tolerance classes, dtypes and costs | the format is `mag-by`'s (`Description`, the pool file, [`schemas/mag-pool.schema.json`](crates/mag-by/schemas/mag-pool.schema.json)) | yours, by hand | is to come from the allocation; not built | is to come from that system; not built |
| the costs in an offer: seconds per step of each part's unit on each device | the format is `mag-by`'s (`Cost`) | you time one step of each unit on each device yourself, at the batch its memory allows, after warm-up and on an idle device, and write it with a band, how far a later measurement may fall from it; a cost you leave out is taken as the slowest any other device measured for that unit, or 1 s if none did, flagged `estimate` with a band of 100% | is to be measured by `mag-mar`; not built | is to be measured by `mag-mar`; not built |
| the placement loop: list the arrangements, remove those that break a constraint, score, commit, record | `mag-mortar`'s `Pool`, public; probing candidates, the time of edges, dividing a chain and moving a running job are not built | `mag plan` shows its decision on your two files | the same library | the same library |
| the queue of jobs and the priority between them | **the pool's**: `mag-mortar`'s, every job submitted to the pool placed against the others | the pool's | the pool's; Slurm queues only the machines' allocations | the pool's |
| which machines a job may use | the host's, given to mag as data: the pool it is handed, a member's `trust`, the level an input or weights are kept at (`keep`), the devices and priority ceiling of a job's stamp, a part's `pin` | the pool file is your policy: list only the machines you allow, and trust a rented one at `outside` | Slurm's accounts and partitions decide the allocation | that system's own authorization decides the pool before mag sees it |
| starting the program on each machine | **the host's**; the program is `mag-mar`'s, not built | you, by hand or over ssh | `sbatch` | that system's |
| taking a machine back | **the host's**, by sending its program a signal; the pool places again; not built | you | Slurm's preemption, with a grace time | that system's |
| coordinating: running the loop over time and keeping its ledger | one of the pool's machines, named by the host; `mag-mortar` does no I/O and hands its state over (`Pool::apply`, `Pool::ledger`, `Ledger::line`, `Pool::replay`); the program that does this is `mag-mar`'s, not built | the same | the same | the same |
| carrying a placement out on the machines | `mag-mar`'s; not built | the same | the same | the same |
| finding machines, waking a sleeping one, keeping it awake | **the host's** | yours: they must be on | the cluster's | that system's |

It is designed for:

- many copies of one worker (rollouts, batch inference, labelling), one on every device that can
  take it;
- a model that no single card holds, divided along a chain across two or three;
- a teacher and a student sharing the pool's memory;
- a long-running service in the pool that yields a card when a more urgent job arrives, and
  spreads back afterwards; and placing again when the host takes a machine back;
- data that must not leave: data is kept at a level of trust (`outside < site < owner`), and what
  is computed from it is never to be placed on a machine trusted below that level, unless a part of
  the graph releases it to a lower one.

**What exists today is the description and the loop that places it, not the pool.**
[`mag-by`](crates/mag-by) reads a graph file, refuses one that is wrong with a reason that names
what is wrong, works out the level of trust each part's data is kept at, and writes the graph as
Graphviz DOT:

![The example graph: a labelling job, its frames kept at owner](crates/mag-by/graphs/example.svg)

That is [`graphs/example.json`](crates/mag-by/graphs/example.json). Red is data kept at `owner` and
everything it reaches, here from the frames and from the classifier's weights; orange would be
`site`; the dashed edge is where a part releases its output, a count of labels, to `outside`. To
draw your own, with Graphviz installed:

    cargo run -p mag-by --features graph --example dot -- your-graph.json | dot -Tsvg > graph.svg

[`mag-mortar`](crates/mag-mortar) holds the placement loop, run on what each device offers given as
data: on every arrival, end, join or leave it lists the ways to place the jobs waiting, removes
those that break memory, a part's tolerance and dtypes, a pin, a job's need, the devices the job's
stamp allows, or the guard that keeps data off machines trusted below its level, scores the rest
from the devices' measured costs and the jobs' priorities, commits the best, and records every
candidate with why any was left out. Its tests hold it to the cases it is designed for: a job on the
fastest device that holds it, copies wherever they fit, a pinned job waiting for its backend, the
higher priority of two first, a stamped job only on its devices, and no data on a machine trusted
below its level.

To see it decide, plan the example graph on a pool of four devices described in a file,
[`pools/example.json`](crates/mag-by/pools/example.json), one of them a fast rented card:

    cargo run -p mag-mortar --features files --example mag -- plan \
        crates/mag-by/pools/example.json crates/mag-by/graphs/example.json

```text
label-frames would be placed: of 65 candidates, 58 were removed and 7 scored.
The best scores 2.000, band ±5%.

  decode  copies on server/0, workstation/0
  b0      on server/0, with chain model
  b1      on server/0, with chain model
  head    on workstation/0
  count   on workstation/0

Kept off a device, and why:
  laptop/0's measured bounds do not meet decode's tolerance class
  the guard: decode carries data kept at owner, and rented/0's member is trusted only at outside
  ...
```

The rented card is the fastest and gets nothing: every part reads the frames kept at owner. `head`
is pinned to CUDA, and the chain's two blocks stay together. The whole output is
[`plans/example.txt`](crates/mag-mortar/plans/example.txt); `--json` prints the loop's record.
The command plans and keeps nothing.

A pool keeps its state as a ledger of what it was told, with the record made each time. It writes
nothing itself: the coordinator is to append each entry as a line, and a pool comes back by
replaying them, each record checked against the one logged.

[`mag-mar`](crates/mag-mar) has its link: two members open a channel for a job over TCP with
TLS 1.3, each holding a certificate from the pool's own authority, and send `mag-by`'s words and
tensors on it. Not built yet: the program each machine is to run, offering its devices to the
pool, running the parts placed on it, and on the coordinator serving the pool; the recipes for a
list of machines and for Slurm; running anything; members on other machines; and trying
candidates on the job's own steps. Nothing here runs work yet.

The graph file is at version 2 (version 1 files still read) and will change. If your job cannot be
said in it, that is what we most want to hear: please open an issue.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Contributing

Pre-alpha and moving fast — issues and design discussion welcome. Please open an issue before large PRs.
