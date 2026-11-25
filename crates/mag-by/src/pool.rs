//! The words of a pool: its members and devices, what each device offers, where a job's parts
//! are, what happens, the record of each run of the placement loop, the pool as an author sees
//! it, and the ledger of what the pool was told.
//!
//! `mag-mar` on each member reports offers and carries out placements; `mag-mortar` runs the loop
//! and keeps the record. Both speak these words, and a ledger keeps the record.
//!
//! With the `graph` feature each word has serde, written as the graph file's words are: snake
//! case, unknown fields refused, and an option present with a value or left out, never `null`.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

#[cfg(feature = "graph")]
use crate::graph::present;
use crate::{Backend, Dtype, Graph, Stamp, Tolerance, Trust, Unit};

/// The ledger's `format`.
pub const LEDGER_FORMAT: &str = "mag-ledger";

/// The ledger's `version` this crate writes. It reads version 1 too, to the same meaning.
///
/// Version 2 carries the graph file's version 2 in its submissions, each with the host's
/// [`Stamp`]; a member's `trust` where version 1 had `remote`; the levels in a guard's removal
/// and the new removal by a stamp; a record's `lowered` where version 1 had `marked_remote`; and
/// in its header the version of `mag-mortar` that wrote it and the ledger it follows.
pub const LEDGER_VERSION: u64 = 2;

/// The pool file's `format`: a pool's [`Description`].
pub const POOL_FORMAT: &str = "mag-pool";

/// The pool file's `version` this crate writes. It reads version 1 too, to the same meaning.
///
/// Version 2 gives each member its `trust` where version 1 had `remote`.
pub const POOL_VERSION: u64 = 2;

/// A machine in the pool, running `mag-mar`: one lease of the host's.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Member {
    /// A DNS name in lower case, as [`is_member_name`] takes it: the member's certificate
    /// carries it so.
    #[cfg_attr(
        feature = "graph",
        schemars(
            regex(
                pattern = r"^(?!(.*\.)?[0-9]+$)[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*$"
            ),
            length(min = 1, max = 253)
        )
    )]
    pub name: String,
    /// How far the host trusts it. Nothing kept at a higher level is placed on its devices.
    pub trust: Trust,
    /// Seconds within which the member gives a device back when the host takes it back.
    pub give_back_seconds: f64,
}

/// Whether `name` may name a member: a DNS name in lower case, since the member's certificate
/// carries its name as a DNS name, and DNS names are compared without regard to case. Labels of
/// `a` to `z`, `0` to `9` and `-`, each of 1 to 63 bytes and neither beginning nor ending with
/// `-`, joined by `.`, at most 253 bytes in all, the last label not all digits, so that no name
/// reads as an address. The pool file and the start file refuse any other.
pub fn is_member_name(name: &str) -> bool {
    let label = |l: &str| {
        let b = l.as_bytes();
        !b.is_empty()
            && b.len() <= 63
            && b[0] != b'-'
            && b[b.len() - 1] != b'-'
            && b.iter()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'-')
    };
    !name.is_empty()
        && name.len() <= 253
        && name.split('.').all(label)
        && name
            .rsplit('.')
            .next()
            .is_some_and(|last| !last.bytes().all(|c| c.is_ascii_digit()))
}

/// A device: its member, and its index on that member.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct DeviceId {
    pub member: String,
    pub index: u32,
}

/// `member/index`, as `workstation/0`.
impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.member, self.index)
    }
}

/// What a device gives the pool now.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Offer {
    pub device: DeviceId,
    pub backend: Backend,
    /// The memory it can give: what is free of anything else holding it.
    pub memory_bytes: u64,
    /// The tolerance classes its measured bounds meet.
    pub tolerances: Vec<Tolerance>,
    /// The dtypes it computes in.
    pub dtypes: Vec<Dtype>,
    /// Its time per step for each part kind it has run or estimated.
    pub costs: Vec<Cost>,
}

/// A part kind's time on a device, per step, at the batch the device's memory allows.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Cost {
    pub unit: Unit,
    pub seconds_per_step: f64,
    /// How far a measurement may be from the prediction, relative: taken from past differences
    /// between the two.
    pub band: f64,
    /// Not measured on this device: estimated from the part's shapes, and so to be probed, not
    /// trusted.
    pub estimate: bool,
}

/// Where a part is.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(rename_all = "snake_case"))]
pub enum Place {
    /// Not placed: its job waits.
    Waiting,
    /// On one device.
    On(DeviceId),
    /// A replicable part's copies, one on each device, each pulling items from one queue.
    Copies(Vec<DeviceId>),
}

/// Where each part of one job is, in the graph's order of parts.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Arrangement {
    pub job: String,
    pub places: Vec<Place>,
}

/// What the loop runs on.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(
    feature = "graph",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum Event {
    /// A job is submitted.
    Arrival { job: String },
    /// A job finishes, or is withdrawn.
    End { job: String },
    /// A device joins the pool.
    Join { device: DeviceId },
    /// A device leaves, or the host takes it back.
    Leave { device: DeviceId },
    /// A device's offer changes.
    OfferChanged { device: DeviceId },
    /// A measured cost leaves its predicted band.
    Drift { job: String, part: String },
    /// A job's deadline comes at risk.
    DeadlineAtRisk { job: String },
}

/// How a candidate changes the current arrangement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(rename_all = "snake_case"))]
pub enum Move {
    /// Waiting parts onto devices with memory free.
    Fit,
    /// A chain onto more devices, or more copies of a replicable part.
    Spread,
    /// Parts into the memory another job leaves free on a device.
    Share,
    /// A running job's parts off a device, onto fewer devices or a smaller variant.
    Squeeze,
    /// The arrival, or a job of lower priority, left in the queue.
    Wait,
}

/// Why a candidate was removed before it was scored, or why a device was given no copy of a
/// replicable part.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(
    feature = "graph",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum Removal {
    /// The device is not in the pool.
    NoDevice { device: DeviceId },
    /// The parts placed on a device need more memory than it offers, or a copy does not fit
    /// beside them.
    Memory { device: DeviceId },
    /// The device's bounds do not meet the part's tolerance class.
    Tolerance { part: String, device: DeviceId },
    /// The device does not compute in a dtype one of the part's edges carries.
    Dtype { part: String, device: DeviceId },
    /// The part is pinned elsewhere.
    Pin { part: String, device: DeviceId },
    /// The part carries data kept at a level above the trust of the device's member: `keep` is
    /// the part's level, `trust` the member's (`outside` for a member the pool does not know).
    Guard {
        part: String,
        device: DeviceId,
        keep: Trust,
        trust: Trust,
    },
    /// The device is not among those the job's stamp allows.
    Stamp { job: String, device: DeviceId },
    /// The job's need is not met: its rate is below the rate it declared.
    Need { job: String },
}

/// The reason in words, for a person reading a plan.
impl fmt::Display for Removal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Removal::NoDevice { device } => write!(f, "{device} is not in the pool"),
            Removal::Memory { device } => write!(
                f,
                "{device} has not the memory for the parts that would be placed on it"
            ),
            Removal::Tolerance { part, device } => write!(
                f,
                "{device}'s measured bounds do not meet {part}'s tolerance class"
            ),
            Removal::Dtype { part, device } => write!(
                f,
                "{device} does not compute in a dtype that {part}'s edges carry"
            ),
            Removal::Pin { part, device } => {
                write!(
                    f,
                    "{part} is pinned to a member or backend that {device} is not"
                )
            }
            Removal::Guard {
                part,
                device,
                keep,
                trust,
            } => write!(
                f,
                "the guard: {part} carries data kept at {keep}, and {device}'s member is trusted \
                 only at {trust}"
            ),
            Removal::Stamp { job, device } => {
                write!(f, "{device} is not among the devices {job}'s stamp allows")
            }
            Removal::Need { job } => {
                write!(f, "{job} would run below the rate it needs")
            }
        }
    }
}

/// One candidate arrangement of the jobs an event touches.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Candidate {
    pub arrangements: Vec<Arrangement>,
    pub moves: Vec<Move>,
    /// Why it was removed, if it was.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub removed: Option<Removal>,
    /// The devices given no copy of a replicable part, each with why: a candidate keeps the
    /// devices that may take a copy and have room for it, and says why it left out the rest.
    pub left_out: Vec<(String, Removal)>,
    /// Its predicted score, if it was not removed.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub score: Option<f64>,
    /// The score's band, relative.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub band: Option<f64>,
}

/// One run of the placement loop: the event, every candidate, and the choice.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Record {
    pub event: Event,
    pub candidates: Vec<Candidate>,
    /// The candidate committed, by index; `None` when nothing moved.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub chosen: Option<usize>,
    /// Whether the candidates were too many to list and were generated greedily instead.
    pub greedy: bool,
    /// Members whose trust this event lowered below an earlier report's: of two reports for one
    /// member, the lower trust stands.
    pub lowered: Vec<String>,
}

/// A file format a pool reads, at the version it reads.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Format {
    pub name: String,
    pub version: u64,
}

/// A job a pool holds, as the pool's description shows it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Held {
    /// Where its parts are. The job's name is the arrangement's.
    pub arrangement: Arrangement,
    pub priority: u32,
    /// Seconds it has waited since it arrived or last lost its place: 0 while it is placed.
    pub waited_seconds: f64,
}

/// The pool as an author sees it, as one virtual device: its members and what each device
/// offers, where each job it holds is and how long each waiting one has waited, and the file
/// formats it reads. An author reads it before writing a graph.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Description {
    /// The pool's time, in seconds: the latest it has been told of.
    pub at: f64,
    /// Its members, each with the lowest trust and the shortest time to give a device back it was
    /// reported with.
    pub members: Vec<Member>,
    /// What each device offers, with the costs measured or estimated on it, in the order of the
    /// devices' ids.
    pub offers: Vec<Offer>,
    /// The jobs it holds, in the order they arrived.
    pub jobs: Vec<Held>,
    /// The file formats it reads.
    pub formats: Vec<Format>,
}

/// What a pool is told: the only ways its state changes.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(
    feature = "graph",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum Change {
    /// A job arrives, with the host's stamp on it.
    Submit { graph: Graph, stamp: Stamp },
    /// A job finishes, or is withdrawn.
    End { job: String },
    /// A device joins, with its member's report if the member has one to make.
    Join {
        #[cfg_attr(
            feature = "graph",
            serde(
                default,
                deserialize_with = "present",
                skip_serializing_if = "Option::is_none"
            )
        )]
        member: Option<Member>,
        offer: Offer,
    },
    /// A device leaves, or the host takes it back.
    Leave { device: DeviceId },
}

/// One change a pool took, at the time it was told of it, with the record the loop made of it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Entry {
    /// Seconds, as the pool was told: finite.
    pub at: f64,
    pub change: Change,
    pub record: Record,
}

/// A pool's ledger: the members and offers it began with, then every change it took, in order.
///
/// The pool is deterministic, so this is all of its state: applying the changes again to a pool
/// begun the same way gives it back, and each entry's record checks that it went the same way. A
/// refused change is not in it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ledger {
    /// The pool's time when it began, in seconds: 0, or the time of the pool file it began from.
    pub at: f64,
    /// The version of `mag-mortar` that began it, which `mag-mortar` writes. Absent in a ledger
    /// of version 1, which did not record it.
    pub written_by: Option<String>,
    /// The name of the ledger this one follows, where a new coordinator began it from an earlier
    /// one; absent for a first ledger. What a name is, is the host's: the ledger's place, as a
    /// start file gives it.
    pub follows: Option<String>,
    pub members: Vec<Member>,
    /// In the order of the devices' ids.
    pub offers: Vec<Offer>,
    pub entries: Vec<Entry>,
}
