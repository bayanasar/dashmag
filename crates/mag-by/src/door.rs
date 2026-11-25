//! The words of the door: what a command run on the coordinator's machine asks the pool, and
//! what the pool answers.
//!
//! The door is a local socket on the coordinator's machine, which the start file names (`door`).
//! The host reaches it by a command run there, by hand on a list of machines, as a step of the
//! coordinator's job under a cluster's scheduler. Each request is one [`DoorRequest`], and the
//! coordinator answers it with one [`DoorAnswer`]. A change the request makes to the pool is
//! written to the pool's ledger before it is answered, so an answer is never given for a change
//! a restarted coordinator would not find.
//!
//! With the `graph` feature each word has serde, written as the files' words are: on the socket,
//! a request is one JSON object on one line, and so is its answer.

use alloc::string::String;

#[cfg(feature = "graph")]
use crate::graph::present;
use crate::{Arrangement, Asks, Description, Graph, Place, Record, Stamp};

/// What a command asks at the door.
///
/// Taking a device back by need is to be asked here too, and is not built: a coordinator that
/// does not read a request answers [`DoorRefusal::Unreadable`].
// One request is made for each command, so a graph held inline costs nothing worth a box.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(
    feature = "graph",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum DoorRequest {
    /// A job, with the host's stamp on it: the pool places it against the jobs it holds, or it
    /// waits.
    Submit { graph: Graph, stamp: Stamp },
    /// A job ends, by its name: it finishes, or is withdrawn.
    End { job: String },
    /// The pool as one virtual device, with its jobs.
    Status,
}

/// The coordinator's answer to one [`DoorRequest`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(
    feature = "graph",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum DoorAnswer {
    /// The job is in the pool, and in its ledger: the record of the loop's run on its arrival,
    /// and where its parts are, each on a device or waiting.
    Submitted {
        record: Record,
        arrangement: Arrangement,
    },
    /// The job's end is in the pool, and in its ledger: the record of the loop's run on it.
    Ended { record: Record },
    /// The pool now: its members, their devices' offers, and its jobs, where each is and how
    /// long each waiting one has waited, at the pool's time.
    Status {
        pool: Description,
        /// What the pool asks of its host: the kind of device it most wants, and the members it
        /// would give back. Left out until the pool can say; it is not built yet.
        #[cfg_attr(
            feature = "graph",
            serde(
                default,
                deserialize_with = "present",
                skip_serializing_if = "Option::is_none"
            )
        )]
        asks: Option<Asks>,
    },
    /// The request is refused, and nothing was written.
    Refused { refusal: DoorRefusal },
}

impl DoorAnswer {
    /// For a submission, whether the job is placed: every part on a device. `None` for any
    /// other answer.
    pub fn placed(&self) -> Option<bool> {
        match self {
            DoorAnswer::Submitted { arrangement, .. } => Some(
                !arrangement
                    .places
                    .iter()
                    .any(|p| matches!(p, Place::Waiting)),
            ),
            _ => None,
        }
    }
}

/// Why the door refused a request. Nothing it refuses is in the ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(
    feature = "graph",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum DoorRefusal {
    /// The request is not one this coordinator reads: not JSON, not a request, or one it does
    /// not take yet. Why, in words.
    Unreadable { error: String },
    /// The pool refused the change, as its placement library says: a graph that fails its
    /// check, a stamp it cannot take, a second job of one name. Why, in words.
    Pool { reason: String },
    /// The pool holds no job of this name, so it has nothing to end.
    NoJob { job: String },
}

impl core::fmt::Display for DoorRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DoorRefusal::Unreadable { error } => {
                write!(f, "the coordinator does not read the request: {error}")
            }
            DoorRefusal::Pool { reason } => write!(f, "the pool refuses it: {reason}"),
            DoorRefusal::NoJob { job } => write!(f, "the pool holds no job named `{job}`"),
        }
    }
}
