//! The words of a link between two members: opening a channel for a job, a tensor crossing it,
//! and a member's words to the coordinator on its control channel.
//!
//! A member links to another with one channel for each job they share, so that one job's words
//! never reach another's, and to the coordinator with one control channel besides, for no job.
//! How a channel is carried (TCP with TLS, and how a word is framed) is the transport's, in
//! `mag-mar`; what is said on it is these words, and the others of this crate.

use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "graph")]
use crate::graph::present;
use crate::{DeviceId, Dtype, Member, Offer};

/// The version of the words members send each other on a link. Its own, apart from each file's
/// version and from the crate's.
pub const WIRE_VERSION: u64 = 1;

/// The job a control channel is opened for: none, written as the empty name, which no job may
/// have ([`crate::Graph::check`] refuses it). A member opens one to the coordinator when its
/// lease begins, and says [`Control`] on it.
pub const CONTROL: &str = "";

/// The first word on a channel, said once by each end. The member that dials names the job the
/// channel is for, or [`CONTROL`]; the other answers with the same job when it takes the channel.
/// Each states the wire version it speaks, and a channel whose two ends differ is closed.
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

/// The words on a member's control channel to the coordinator ([`CONTROL`]): the member's,
/// [`Control::Join`], [`Control::Leave`] and [`Control::GaveBack`], and the coordinator's,
/// [`Control::GiveBack`]. Each end takes only the other's.
///
/// When its lease begins, a member dials the coordinator and joins; the host tells the
/// coordinator nothing. When the host ends the lease (SIGTERM), the member says it leaves, closes
/// its way of the channel, and waits for the coordinator to close the other way, the sign that
/// the leaving is recorded, until its `give_back_seconds` are up. When the host takes a device
/// back by need at the door, and the pool chooses one of this member's, the coordinator tells it
/// to give that device back, and the member says when it has.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(
    feature = "graph",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum Control {
    /// The member joins the pool: itself as its start file's pool describes it (its trust and its
    /// time to give a device back, as the host wrote them), and what each of its devices offers.
    /// Its name is the one its certificate carries, which the coordinator checks.
    Join { member: Member, offers: Vec<Offer> },
    /// The member leaves: its lease is ending, and its devices are given back.
    Leave,
    /// The coordinator's word: give this device back within `deadline_seconds`, as the host
    /// took a device back by need and the pool chose this one. Its leave is in the ledger before
    /// the word is said; the member stops offering it, and keeps its other devices and its lease.
    GiveBack {
        device: DeviceId,
        deadline_seconds: f64,
    },
    /// The member has given this device back, as the coordinator told it to: it offers it no
    /// longer.
    GaveBack { device: DeviceId },
}
