//! The words of a pool: its members and devices, what each device offers, where a job's parts
//! are, what happens, and the record of each run of the placement loop.
//!
//! `mag-mar` on each member reports offers and carries out placements; `mag-mortar` runs the loop
//! and keeps the record. Both speak these words, and a ledger keeps the record.

use alloc::string::String;
use alloc::vec::Vec;

use crate::{Backend, Dtype, Tolerance, Unit};

/// A machine in the pool, running `mag-mar`.
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    pub name: String,
    /// Outside the owner's own machines, as a rented GPU is. Nothing local-only is placed on a
    /// remote member's devices.
    pub remote: bool,
    /// Seconds within which the member gives a device back when its owner wants it.
    pub give_back_seconds: f64,
}

/// A device: its member, and its index on that member.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeviceId {
    pub member: String,
    pub index: u32,
}

/// What a device gives the pool now.
#[derive(Clone, Debug, PartialEq)]
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
pub struct Arrangement {
    pub job: String,
    pub places: Vec<Place>,
}

/// What the loop runs on.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A job is submitted.
    Arrival { job: String },
    /// A job finishes, or is withdrawn.
    End { job: String },
    /// A device joins the pool.
    Join { device: DeviceId },
    /// A device leaves, or its owner takes it back.
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
    /// The part carries local-only data and the device's member is remote.
    Guard { part: String, device: DeviceId },
    /// The job's need is not met: its rate is below the rate it declared.
    Need { job: String },
}

/// One candidate arrangement of the jobs an event touches.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub arrangements: Vec<Arrangement>,
    pub moves: Vec<Move>,
    /// Why it was removed, if it was.
    pub removed: Option<Removal>,
    /// The devices given no copy of a replicable part, each with why: a candidate keeps the
    /// devices that may take a copy and have room for it, and says why it left out the rest.
    pub left_out: Vec<(String, Removal)>,
    /// Its predicted score, if it was not removed.
    pub score: Option<f64>,
    /// The score's band, relative.
    pub band: Option<f64>,
}

/// One run of the placement loop: the event, every candidate, and the choice.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub event: Event,
    pub candidates: Vec<Candidate>,
    /// The candidate committed, by index; `None` when nothing moved.
    pub chosen: Option<usize>,
    /// Whether the candidates were too many to list and were generated greedily instead.
    pub greedy: bool,
    /// Members this event marked remote where an earlier report had them local: of two marks for
    /// one member, the stricter stands.
    pub marked_remote: Vec<String>,
}
