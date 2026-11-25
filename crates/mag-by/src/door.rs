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
use alloc::vec::Vec;

#[cfg(feature = "graph")]
use crate::graph::present;
use crate::{Arrangement, Asks, Description, DeviceId, Graph, Place, Record, Stamp, TakeBack};

/// What a command asks at the door.
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
    /// The host takes a device back: by need ([`TakeBack::Need`]), a device of a kind within a
    /// deadline, whichever the pool loses least by. Taking back by name is the host's SIGTERM to
    /// the member's program, not the door's: a coordinator answers a request by name
    /// [`DoorRefusal::ByName`].
    TakeBack(TakeBack),
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
        /// would give back. Left out by a coordinator that does not say.
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
    /// A device is taken back for the host's need: the device the pool chose, the record of the
    /// loop's run on its leave, which is in the pool's ledger, and whether it is given back.
    TakenBack {
        device: DeviceId,
        record: Record,
        /// Whether the device's member said it gave the device back before the deadline, or the
        /// device was the coordinator's own, which it stops offering itself. False when the
        /// deadline passed first, or the member could not be told; the device is the host's at
        /// the deadline either way, and the pool places nothing on it from its leave.
        given_back: bool,
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
    /// Nothing the pool holds meets the need: no device of the kind, or none whose member gives
    /// a device back within the deadline. `slow` lists the devices of the kind whose members take
    /// longer, and is empty when no device is of the kind.
    Unmet { slow: Vec<DeviceId> },
    /// A take-back by name, of a device or a member: the host's to do, by SIGTERM to the member's
    /// program, which gives all its devices back within its time to give back. The door takes
    /// back by need only.
    ByName,
}

impl core::fmt::Display for DoorRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DoorRefusal::Unreadable { error } => {
                write!(f, "the coordinator does not read the request: {error}")
            }
            DoorRefusal::Pool { reason } => write!(f, "the pool refuses it: {reason}"),
            DoorRefusal::NoJob { job } => write!(f, "the pool holds no job named `{job}`"),
            DoorRefusal::Unmet { slow } if slow.is_empty() => {
                write!(f, "no device in the pool is of the kind asked for")
            }
            DoorRefusal::Unmet { slow } => {
                write!(f, "the devices of the kind asked for (")?;
                for (i, d) in slow.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{d}")?;
                }
                write!(
                    f,
                    ") are each of a member that takes longer than the deadline to give a \
                     device back"
                )
            }
            DoorRefusal::ByName => write!(
                f,
                "the door takes a device back by need only; by name, send the member's program \
                 SIGTERM"
            ),
        }
    }
}
