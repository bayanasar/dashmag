//! The words of a link between two members: opening a channel for a job, and a tensor crossing
//! it.
//!
//! A member links to another with one channel for each job they share, so that one job's words
//! never reach another's. How a channel is carried (TCP with TLS, and how a word is framed) is
//! the transport's, in `mag-mar`; what is said on it is these words, and the others of this
//! crate.

use alloc::string::String;
use alloc::vec::Vec;

use crate::Dtype;
#[cfg(feature = "graph")]
use crate::graph::present;

/// The version of the words members send each other on a link. Its own, apart from each file's
/// version and from the crate's.
pub const WIRE_VERSION: u64 = 1;

/// The first word on a channel, said once by each end. The member that dials names the job the
/// channel is for; the other answers with the same job when it takes the channel. Each states
/// the wire version it speaks, and a channel whose two ends differ is closed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Open {
    pub wire: u64,
    pub job: String,
}

/// A tensor crossing a link: the value one edge of the channel's job carries at one step. Its
/// `bytes` follow it on the channel, the elements in row-major order over `shape`, each
/// little-endian.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Tensor {
    /// The edge's input or part, as the graph names it.
    pub from: String,
    /// The edge's part, or `None` for a result of the job.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub to: Option<String>,
    /// The step whose value it is, counted from 0.
    pub step: u64,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// How many bytes follow.
    pub bytes: u64,
}
