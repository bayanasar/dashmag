# mag-by

The words the `mag-` crates share. mag is to place the work of one algorithm across the machines of
a pool: its author describes the work as a graph and names no device, and mag decides where each
piece runs (`mag-mortar`, on offers given as data so far). This crate holds the words that
description and that decision are made of, so that every member reads them the same way. It is
`no_std` with `alloc`, and depends on nothing unless the `graph` feature is on.

- `Graph`: one job. Its parts, the smallest pieces mag places (a function with the weights it
  holds, the memory it needs, and where it can stop); the inputs it reads; the edges between
  them, with their sizes; the chains along which a model may be divided between devices; and
  what the job needs and how much it weighs against other jobs.
- `Graph::check`: what every graph must satisfy before it is placed, refused with a typed
  `Refusal` that names what is wrong.
- `Graph::local_only`: which parts and edges carry data computed from an input or from weights
  marked local-only, so that none of it is to be placed on a member marked remote unless a part
  releases it.
- The pool's words: a member and its devices, the offer each device makes, where a job's parts
  are, the events the placement loop runs on and the record of each run, with every candidate
  and why any was removed; a pool's `Description`, as an author reads it; and its `Ledger`, the
  changes a pool was told with the record of each, from which the pool is replayed.

The `graph` feature, named for the first of them, now turns on all of the files, each read and
written as JSON with serde, each refusing a key repeated in one object (JSON would keep its last
value, and a schema cannot see it), an unknown field and a `null`, and each with its JSON Schema
kept in [`schemas`](schemas) and held to the types by a test:

- `Graph::from_json` and `Graph::to_json`: the graph file, a graph in
  JSON, as an agent writes, reads back and edits it (`format` `"mag-graph"`, `version` 1).
  [`graphs/example.json`](graphs/example.json) is one. `graph_schema` is the file's JSON
  Schema, kept at [`schemas/mag-graph.schema.json`](schemas/mag-graph.schema.json), and
  `Graph::to_dot` writes a graph as Graphviz DOT for a person to look at
  ([`graphs/example.dot`](graphs/example.dot), drawn as
  [`graphs/example.svg`](graphs/example.svg)). The SVG is rendered again with Graphviz
  (`dot -Tsvg`) in the commit that changes `example.dot`.
- `Description::from_json` and `Description::to_json`: the pool file, a pool's description as
  `describe` gives it (`"mag-pool"`, version 1), so that a pool can be written to a file and
  planned against elsewhere. [`pools/example.json`](pools/example.json) is four devices, one
  of them on a member marked remote.
- `Ledger::header_line`, `Ledger::line` and `Ledger::from_lines`: the ledger as JSON Lines
  (`"mag-ledger"`, version 1), a header and then an entry a line, for the host of a pool to
  append and read back. Where the lines are kept is the host's. A last line cut short was never
  acknowledged, and is dropped and reported; a line cut short before it is refused. Every float
  reads back to its bits.

To print a graph file's DOT, or why the file is refused:

    cargo run -p mag-by --features graph --example dot -- crates/mag-by/graphs/example.json

The author decides what a part is and where the graph may be divided; mag never divides a part.
