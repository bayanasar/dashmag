//! The words the `mag-` crates share.
//!
//! mag is to place the work of one algorithm across the machines of a pool. Its author, often an
//! agent, describes the work as a graph and names no device; mag is to decide where each piece
//! runs. A host gives the pool its machines, and the pool places the jobs on them.
//! This crate holds the words that description is made of, so that every member of a pool,
//! down to a small processor with no operating system, reads it the same way. It is `no_std`
//! with `alloc`, and depends on nothing unless the `graph` feature is on.
//!
//! - [`Graph`]: one job. Its parts, the smallest pieces mag places; the inputs it reads; the
//!   edges between them, with their sizes; the chains along which a model may be divided between
//!   devices; and what the job needs and how much it weighs against other jobs.
//! - [`Graph::check`]: what every graph must satisfy before it is placed, refused with a typed
//!   [`Refusal`] that names what is wrong.
//! - [`Trust`], a level ordered `outside < site < owner`, and [`Graph::keeps`]: the level each
//!   part and edge is kept at, carried from the inputs and blobs it reads unless a part releases
//!   it, so that none of it is placed on a member trusted at a lower level.
//! - The pool's words: a [`Member`] and its devices, the [`Offer`] each device makes, the
//!   [`Arrangement`] of a job's parts, the [`Event`]s the placement loop runs on, and the
//!   [`Record`] of each run of it, with each [`Candidate`] and why any was removed; the
//!   [`Description`] of a pool, as an author reads it before writing a graph; and the pool's
//!   [`Ledger`]: what it began with, then each [`Change`] it was told, as an [`Entry`] with the
//!   record the loop made of it, from which the pool is restored by replaying it.
//! - The host's words: the [`Stamp`] the host puts on a submission, a [`TakeBack`] by name or by
//!   need, what the pool [`Asks`] of the host, and the [`Start`] file a member's program begins
//!   from, with the pool's authority certificate and its [`Peer`]s.
//! - The words of a link between two members, at [`WIRE_VERSION`]: the [`Open`] each end says
//!   once on a channel for a job, and a [`Tensor`] whose bytes follow it.
//! - With the `graph` feature, `Graph::from_json` and `Graph::to_json`: the graph file, a
//!   graph in JSON, as an agent writes, reads back and edits it (`format` `"mag-graph"`,
//!   `version` 2); `graph_schema`, its JSON Schema; `Graph::to_dot`, the graph as Graphviz DOT
//!   for a person to look at; `Ledger::header_line`, `Ledger::line` and `Ledger::from_lines`,
//!   the ledger as JSON Lines (`format` `"mag-ledger"`, `version` 2), for the host of a pool to
//!   append and read back, with `ledger_schema`, the schema of a line; `Description::from_json`
//!   and `Description::to_json`, the pool file (`format` `"mag-pool"`, `version` 2), with
//!   `pool_schema`; and `Start::from_json` and `Start::to_json`, the start file (`format`
//!   `"mag-start"`, `version` 1), with `start_schema`. Files of the graph's, the pool's and the
//!   ledger's version 1 are read too, to the same meaning.
//!
//! The author decides what a part is and where the graph may be divided; mag never divides a
//! part.

#![cfg_attr(not(feature = "graph"), no_std)]

extern crate alloc;

#[cfg(feature = "graph")]
mod dot;
mod graph;
mod host;
#[cfg(feature = "graph")]
mod json;
mod keep;
#[cfg(feature = "graph")]
mod ledger;
mod link;
mod pool;
#[cfg(feature = "graph")]
mod v1;

pub use graph::{
    Backend, Blob, Chain, Dtype, Edge, GRAPH_FORMAT, GRAPH_VERSION, Graph, Input, Job, Mark, Need,
    Part, Pin, Refusal, Step, Tolerance, Unit,
};
pub use host::{Asks, DeviceKind, Peer, START_FORMAT, START_VERSION, Stamp, Start, TakeBack};
#[cfg(feature = "graph")]
pub use json::{PoolFileError, ReadError, StartFileError, graph_schema, pool_schema, start_schema};
pub use keep::{Keeps, Trust};
#[cfg(feature = "graph")]
pub use ledger::{LedgerError, ReadLedger, ledger_schema};
pub use link::{Open, Tensor, WIRE_VERSION};
pub use pool::{
    Arrangement, Candidate, Change, Cost, Description, DeviceId, Entry, Event, Format, Held,
    LEDGER_FORMAT, LEDGER_VERSION, Ledger, Member, Move, Offer, POOL_FORMAT, POOL_VERSION, Place,
    Record, Removal,
};
