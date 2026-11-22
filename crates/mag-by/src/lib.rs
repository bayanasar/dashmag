//! The words the `mag-` crates share.
//!
//! mag places the work of one algorithm across the machines of a pool. Its author, often an
//! agent, describes the work as a graph and names no device; mag decides where each piece runs.
//! This crate holds the words that description is made of, so that every member of a pool,
//! down to a small processor with no operating system, reads it the same way. It is `no_std`
//! with `alloc`, and depends on nothing.
//!
//! - [`Graph`]: one job. Its parts, the smallest pieces mag places; the inputs it reads; the
//!   edges between them, with their sizes; the chains along which a model may be divided between
//!   devices; and what the job needs and how much it weighs against other jobs.
//! - [`Graph::check`]: what every graph must satisfy before it is placed, refused with a typed
//!   [`Refusal`] that names what is wrong.
//!
//! The author decides what a part is and where the graph may be divided; mag never divides a
//! part.

#![no_std]

extern crate alloc;

mod graph;

pub use graph::{
    Backend, Blob, Chain, Edge, Graph, Input, Job, Mark, Need, Part, Pin, Refusal, Step, Tolerance,
    Unit,
};
