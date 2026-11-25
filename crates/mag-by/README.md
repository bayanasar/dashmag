# mag-by

The words the `mag-` crates share. mag is to place the work of one algorithm across the machines of
a pool: its author describes the work as a graph and names no device, a host gives the pool its
machines, and mag decides where each piece runs (`mag-mortar`, on offers given as data so far).
This crate holds the words that description and that decision are made of, so that every member
reads them the same way. It is `no_std` with `alloc`, and depends on nothing unless the `graph`
feature is on.

- `Graph`: one job. Its parts, the smallest pieces mag places (a function with the weights it
  holds, the memory it needs, and where it can stop); the inputs it reads; the edges between
  them, with their sizes; the chains along which a model may be divided between devices; and
  what the job needs and how much it weighs against other jobs.
- `Graph::check`: what every graph must satisfy before it is placed, refused with a typed
  `Refusal` that names what is wrong.
- `Trust`, a level ordered `outside < site < owner`. The host trusts each member at a level, and
  each input and blob is kept at one: the least trust a machine must have to hold it.
  `Graph::keeps` carries the levels to everything computed from them, the highest among what a
  part reads, unless a part releases an output to a lower level; nothing is to be placed on a
  member trusted below its level. The host gives the levels, and mag only compares them.
- The pool's words: a member and its devices, the offer each device makes, where a job's parts
  are, the events the placement loop runs on and the record of each run, with every candidate
  and why any was removed; a pool's `Description`, as an author reads it; and its `Ledger`, the
  changes a pool was told with the record of each, from which the pool is replayed, whose header
  names the version of `mag-mortar` that wrote it and the ledger it follows.
- The host's words: the `Stamp` the host puts on a submission (who submitted it, which mag never
  reads; the devices it may use; a ceiling on its priority), which a submission carries into the
  ledger; a `TakeBack`, by name (a device or a member) or by need (a `DeviceKind`: a backend, a
  least memory, or both), each with a deadline, which the host asks at the door and the pool
  learns as devices leaving; what the pool `Asks` of the host, the kind of device it most wants
  and the members it would give back; and the `Start` file each member's program begins from.

The `graph` feature, named for the first of them, now turns on all of the files, each read and
written as JSON with serde, each refusing a key repeated in one object (JSON would keep its last
value, and a schema cannot see it), an unknown field and a `null`, and each with its JSON Schema
kept in [`schemas`](schemas) and held to the types by a test. The graph file, the pool file and
the ledger are at version 2, where trust became a level; each still reads its version 1 to the
same meaning (`local_only` and `remote` as levels, a `release` as a release to `outside`), as
the files kept in [`tests/v1`](tests/v1) show:

- `Graph::from_json` and `Graph::to_json`: the graph file, a graph in
  JSON, as an agent writes, reads back and edits it (`format` `"mag-graph"`, `version` 2).
  [`graphs/example.json`](graphs/example.json) is one. `graph_schema` is the file's JSON
  Schema, kept at [`schemas/mag-graph.schema.json`](schemas/mag-graph.schema.json), and
  `Graph::to_dot` writes a graph as Graphviz DOT for a person to look at
  ([`graphs/example.dot`](graphs/example.dot), drawn as
  [`graphs/example.svg`](graphs/example.svg)). The SVG is rendered again with Graphviz
  (`dot -Tsvg`) in the commit that changes `example.dot`.
- `Description::from_json` and `Description::to_json`: the pool file, a pool's description as
  `describe` gives it (`"mag-pool"`, version 2), so that a pool can be written to a file and
  planned against elsewhere. [`pools/example.json`](pools/example.json) is four devices, one
  of them on a member trusted only at `outside`.
- `Ledger::header_line`, `Ledger::line` and `Ledger::from_lines`: the ledger as JSON Lines
  (`"mag-ledger"`, version 2), a header and then an entry a line, for the host of a pool to
  append and read back. Where the lines are kept is the host's. A last line cut short was never
  acknowledged, and is dropped and reported; a line cut short before it is refused. Every float
  reads back to its bits. A ledger of version 1 is written again whole before it is appended to.
- `Start::from_json` and `Start::to_json`: the start file (`"mag-start"`, version 1), which the
  environment variable `MAG_START` names: this member's name, the pool's description, the pool's
  authority certificate, the peers with an address where they have one, the coordinator, where
  the ledger goes, the door's local address, the idle time after which the pool offers a member
  back, and the paths of this member's certificate and private key files. The key itself is
  never in it. Each pool has its own certificate authority, held by the host, which signs every
  member's certificate and names the member in it; a link is taken only from a certificate the
  authority signed that names the member expected, so a member that joins later proves itself
  by its certificate. [`starts/example.json`](starts/example.json) is one; `mag-mar`'s `certs`
  command makes an authority and its members' certificates.
- The words of a link between two members, at `WIRE_VERSION` 1: `Open`, which each end says
  once on a channel for one job, and `Tensor`, the value of one edge at one step, whose bytes
  follow it on the channel. With the `graph` feature they are JSON, as the files' words are.

To print a graph file's DOT, or why the file is refused:

    cargo run -p mag-by --features graph --example dot -- crates/mag-by/graphs/example.json

The author decides what a part is and where the graph may be divided; mag never divides a part.
