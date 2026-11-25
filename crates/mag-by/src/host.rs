//! The words between a pool and its host: the stamp the host puts on a job, taking a device
//! back, what the pool asks of the host, and the start file each member's program begins from.
//!
//! The host gives a pool its machines; the pool places the jobs on them. The host never reads a
//! graph, and the pool never touches a machine outside it. Every host speaks to the pool in these
//! words, a person with a list of machines as much as a cluster's scheduler.
//!
//! With the `graph` feature each word has serde, written as the graph file's words are.

use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "graph")]
use crate::graph::present;
use crate::{Backend, Description, DeviceId};

/// The start file's `format`: a [`Start`].
pub const START_FORMAT: &str = "mag-start";

/// The start file's `version` this crate reads and writes. A new format, at its first version.
pub const START_VERSION: u64 = 1;

/// The host's stamp on a submission: its one handle on a job, since priority between jobs is
/// decided inside the pool. Each field may be left out, and a stamp with none puts no bound on
/// the job: as a program that submits its own job to a pool of its own stamps it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Stamp {
    /// Who submitted the job, in the host's words. mag never reads it; the ledger keeps it.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub by: Option<String>,
    /// The devices the job may use. Left out, every device in the pool, as it is and as devices
    /// join. Given, at least one.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub devices: Option<Vec<DeviceId>>,
    /// The highest priority the job may have: a graph that asks for more is given this. At
    /// least 1.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub priority_ceiling: Option<u32>,
}

/// A kind of device: a backend, a least memory, both, or, with neither, any device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct DeviceKind {
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub backend: Option<Backend>,
    /// The least memory it offers.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub memory_bytes: Option<u64>,
}

/// The host taking a device back, at its door: by name, or by need, each within a deadline in
/// seconds.
///
/// A request, not a change the pool is told. What it does to the pool is that devices leave,
/// once their parts have stopped, and the pool is told that as [`crate::Change::Leave`]. By
/// name, the member is told to stop and gives its devices back within its `give_back_seconds`,
/// whatever the pool would rather. By need, the pool chooses which device of the kind costs it
/// least to lose, and that device's member then stops as for a name.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(
    feature = "graph",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum TakeBack {
    /// This device.
    Device {
        device: DeviceId,
        deadline_seconds: f64,
    },
    /// Every device of this member: its lease ends.
    Member {
        member: String,
        deadline_seconds: f64,
    },
    /// One device of this kind, whichever the pool loses least by.
    Need {
        kind: DeviceKind,
        deadline_seconds: f64,
    },
}

impl TakeBack {
    /// Seconds within which the host wants the device back.
    pub fn deadline_seconds(&self) -> f64 {
        match self {
            TakeBack::Device {
                deadline_seconds, ..
            }
            | TakeBack::Member {
                deadline_seconds, ..
            }
            | TakeBack::Need {
                deadline_seconds, ..
            } => *deadline_seconds,
        }
    }
}

/// What the pool asks of its host, in its status at the door. The host polls it and decides
/// whether to act; mag never calls the host.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Asks {
    /// While jobs wait, the kind of device whose addition would serve them best; left out when
    /// none waits, or none would.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub wants: Option<DeviceKind>,
    /// The members idle for longer than the start file's idle time, which the pool would give
    /// back.
    pub gives_back: Vec<String>,
}

/// A start file: what a member's program begins from, the same convention for every host. The
/// environment variable `MAG_START` names it.
///
/// Each pool has its own certificate authority, which the host holds. The host issues every
/// member a certificate signed by it and naming the member, and a link between two members is
/// taken only when the other end's certificate is signed by the pool's authority and names the
/// member expected. So a member that joins a running pool proves itself by its certificate, and
/// the host tells the coordinator nothing.
///
/// It holds no private key: it gives the paths of this member's certificate and key files,
/// which the host places and keeps readable by the program alone.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Start {
    /// This process's member, by its name in the pool.
    pub member: String,
    /// Where this member answers other members, as `host:port`, such as `0.0.0.0:7340`: the
    /// address it binds, which the others' start files give as its peer's address. Left out, it
    /// only dials, as a machine behind address translation does; a coordinator needs one, since
    /// every member dials it.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub listen: Option<String>,
    /// The pool as the host describes it, as a pool file does: its members, each with its trust
    /// and its time to give a device back, and any offers the host knows.
    pub pool: Description,
    /// The pool's authority certificate, in PEM: what every member's certificate is signed by.
    /// Public; its private key stays with the host.
    pub authority: String,
    /// The other members this one knows, each with an address where it has one.
    pub peers: Vec<Peer>,
    /// The member that coordinates: this one, with a `listen` address, or one of the peers,
    /// with an address to dial it at.
    pub coordinator: String,
    /// Where the coordinator writes the ledger: a path.
    pub ledger: String,
    /// The door's local address on the coordinator's machine: the path of a local socket.
    pub door: String,
    /// Seconds a member stays idle before the pool offers it back. Finite, and not below 0.
    pub idle_seconds: f64,
    /// The path of this member's certificate file, in PEM: the host's authority's signature on
    /// this member's name and public key.
    pub certificate_file: String,
    /// The path of this member's private key file, in PEM. The key itself is never in a start
    /// file.
    pub key_file: String,
}

/// Another member, as this one reaches it. It is known by the certificate it presents, signed by
/// the pool's authority and naming it, not by anything listed here.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(deny_unknown_fields))]
pub struct Peer {
    pub member: String,
    /// Where it is dialled, as `host:port`, if it can be: a machine behind address translation
    /// has none, and dials out instead.
    #[cfg_attr(
        feature = "graph",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub address: Option<String>,
}
