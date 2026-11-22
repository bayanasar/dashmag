# mag-by

The words the `mag-` crates share. mag places the work of one algorithm across the machines of
a pool: its author describes the work as a graph and names no device, and mag decides where each
piece runs. This crate holds the words that description is made of, so that every member reads
it the same way. It is `no_std` with `alloc`, and depends on nothing unless the `graph` feature
is on.

- `Graph`: one job. Its parts, the smallest pieces mag places (a function with the weights it
  holds, the memory it needs, and where it can stop); the inputs it reads; the edges between
  them, with their sizes; the chains along which a model may be divided between devices; and
  what the job needs and how much it weighs against other jobs.
- `Graph::check`: what every graph must satisfy before it is placed, refused with a typed
  `Refusal` that names what is wrong.
- `Graph::local_only`: which parts and edges carry data computed from an input that must stay
  local, so that none of it is placed on a member marked remote unless a part releases it.
- With the `graph` feature, `Graph::from_json` and `Graph::to_json`: the graph file, a graph in
  JSON, as an agent writes, reads back and edits it (`format` `"mag-graph"`, `version` 1).
  [`graphs/example.json`](graphs/example.json) is one. `graph_schema` is the file's JSON
  Schema, kept at [`schemas/mag-graph.schema.json`](schemas/mag-graph.schema.json), and
  `Graph::to_dot` writes a graph as Graphviz DOT for a person to look at
  ([`graphs/example.dot`](graphs/example.dot)).

The author decides what a part is and where the graph may be divided; mag never divides a part.
