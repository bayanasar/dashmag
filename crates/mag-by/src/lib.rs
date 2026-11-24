//! The words the `mag-` crates share.
//!
//! mag is to place the work of one algorithm across the machines of a pool. Its author, often an
//! agent, describes the work as a graph and names no device; mag is to decide where each piece
//! runs. That part is not built yet.
//! This crate holds the words that description is made of, so that every member of a pool,
//! down to a small processor with no operating system, reads it the same way. It is `no_std`
//! with `alloc`, and depends on nothing unless the `graph` feature is on.
//!
//! - [`Graph`]: one job. Its parts, the smallest pieces mag places; the inputs it reads; the
//!   edges between them, with their sizes; the chains along which a model may be divided between
//!   devices; and what the job needs and how much it weighs against other jobs.
//! - [`Graph::check`]: what every graph must satisfy before it is placed, refused with a typed
//!   [`Refusal`] that names what is wrong.
//! - [`Graph::local_only`]: which parts and edges carry data computed from an input that must
//!   stay local, so that none of it is placed on a member marked remote unless a part releases
//!   it.
//! - The pool's words: a [`Member`] and its devices, the [`Offer`] each device makes, the
//!   [`Arrangement`] of a job's parts, the [`Event`]s the placement loop runs on, and the
//!   [`Record`] of each run of it, with each [`Candidate`] and why any was removed; and the
//!   [`Description`] of a pool, as an author reads it before writing a graph.
//! - With the `graph` feature, `Graph::from_json` and `Graph::to_json`: the graph file, a
//!   graph in JSON, as an agent writes, reads back and edits it (`format` `"mag-graph"`,
//!   `version` 1); `graph_schema`, its JSON Schema; and `Graph::to_dot`, the graph as Graphviz
//!   DOT for a person to look at.
//!
//! The author decides what a part is and where the graph may be divided; mag never divides a
//! part.

#![cfg_attr(not(feature = "graph"), no_std)]

extern crate alloc;

#[cfg(feature = "graph")]
mod dot;
mod graph;
#[cfg(feature = "graph")]
mod json;
mod local_only;
mod pool;

pub use graph::{
    Backend, Blob, Chain, Dtype, Edge, GRAPH_FORMAT, GRAPH_VERSION, Graph, Input, Job, Mark, Need,
    Part, Pin, Refusal, Step, Tolerance, Unit,
};
#[cfg(feature = "graph")]
pub use json::{ReadError, graph_schema};
pub use local_only::LocalOnly;
pub use pool::{
    Arrangement, Candidate, Cost, Description, DeviceId, Event, Format, Held, Member, Move, Offer,
    Place, Record, Removal,
};
