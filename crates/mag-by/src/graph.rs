//! A job's graph: the parts to place, the inputs they read, the edges between them, and what the
//! job needs.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// One submitted job.
///
/// Every name in a graph (inputs, parts and chains together) is unique, and edges and chains
/// refer to inputs and parts by name. [`Graph::check`] says whether a graph is well formed.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(deny_unknown_fields))]
pub struct Graph {
    /// The job's name, its weight and its need.
    pub job: Job,
    /// What the job reads from outside its parts.
    #[cfg_attr(feature = "json", serde(default))]
    pub inputs: Vec<Input>,
    /// The pieces mag places. At least one.
    pub parts: Vec<Part>,
    /// Ordered runs of parts that may be divided between devices at any part boundary.
    #[cfg_attr(feature = "json", serde(default))]
    pub chains: Vec<Chain>,
    /// What moves between inputs and parts, and out of the job.
    #[cfg_attr(feature = "json", serde(default))]
    pub edges: Vec<Edge>,
}

/// A job's name, how much it weighs against the other jobs in the pool, and what it needs.
///
/// Beyond its need a job wants to run as fast as it can: when jobs share the pool, an
/// arrangement is judged by the sum over jobs of priority times each job's rate.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(deny_unknown_fields))]
pub struct Job {
    pub name: String,
    /// The job's weight against other jobs. At least 1.
    pub priority: u32,
    #[cfg_attr(feature = "json", serde(default))]
    pub need: Need,
}

/// What an arrangement must give a job. Each part is optional, and a job with neither only
/// wants to run as fast as its priority allows.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(deny_unknown_fields))]
pub struct Need {
    /// The rate it must sustain, in its own steps per second (items, tokens or optimiser
    /// steps): a service's need. Finite and above 0.
    #[cfg_attr(
        feature = "json",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub rate_per_second: Option<f64>,
    /// Seconds from its submission by which it must finish. Finite and above 0.
    #[cfg_attr(
        feature = "json",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub deadline_seconds: Option<f64>,
}

/// Data a job reads from outside its parts.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(deny_unknown_fields))]
pub struct Input {
    pub name: String,
    pub bytes: u64,
    /// Never placed on a member marked remote, nor is anything computed from it unless a stage
    /// releases it. The graph file must say it either way: left out, it is refused, so that
    /// forgetting it never lets an input move.
    pub local_only: bool,
}

/// The smallest piece mag places: a unit with the weights it holds, the memory it needs, and
/// where it can stop.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(deny_unknown_fields))]
pub struct Part {
    pub name: String,
    /// The function the part runs, already built into each member's binary.
    pub unit: Unit,
    #[cfg_attr(feature = "json", serde(default))]
    pub mark: Mark,
    /// Where the part can stop with its state written out, to move or to make way.
    pub step: Step,
    /// The memory it needs on its device, weights and working memory together.
    pub memory_bytes: u64,
    /// The weights it holds, fetched once per member and kept.
    #[cfg_attr(feature = "json", serde(default))]
    pub weights: Vec<Blob>,
    /// The numerics it accepts, which decide the devices it may run on.
    pub tolerance: Tolerance,
    /// A member or a backend it must run on.
    #[cfg_attr(
        feature = "json",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub pin: Option<Pin>,
}

/// A named, versioned function.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(deny_unknown_fields))]
pub struct Unit {
    pub name: String,
    pub version: u32,
}

/// Data named by its content's hash.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(deny_unknown_fields))]
pub struct Blob {
    pub hash: String,
    pub bytes: u64,
}

/// How a part may be spread over devices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(rename_all = "snake_case"))]
pub enum Mark {
    /// One copy, on one device.
    #[default]
    Atomic,
    /// As many copies as there are devices to hold them, each pulling items from one queue.
    Replicable,
}

/// The point where a part can stop with its state written out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(rename_all = "snake_case"))]
pub enum Step {
    /// After each item, as a map stops.
    Item,
    /// After each token, as a decoder stops.
    Token,
    /// After each optimiser step, as a learner stops.
    OptimiserStep,
    /// After each round of a loop.
    Round,
}

/// The numerics a part accepts. A device takes the part only if its measured bounds for the
/// part's operations meet it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(rename_all = "snake_case"))]
pub enum Tolerance {
    /// The reference's bits.
    Exact,
    F32,
    Bf16,
    Int8,
}

/// Where a part must run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(rename_all = "snake_case"))]
pub enum Pin {
    /// On this member.
    Member(String),
    /// On a device of this backend, so that results stay comparable.
    Backend(Backend),
}

/// A backend a device runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(rename_all = "snake_case"))]
pub enum Backend {
    Cuda,
    Rocm,
    Mlx,
    /// The CPU.
    Flex,
}

/// An ordered run of parts, such as a model's blocks, that may sit on one device or be divided
/// between devices at any boundary between two of them.
///
/// Each part's output feeds the next, so an edge joins each part to the next. Dividing a model
/// between devices is placing a chain's parts on more than one; only the edge at a division
/// crosses between them.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(deny_unknown_fields))]
pub struct Chain {
    pub name: String,
    /// The parts in order. At least 2, none replicable, none in another chain.
    pub parts: Vec<String>,
}

/// Data that moves from an input or a part to a part, or out of the job.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(deny_unknown_fields))]
pub struct Edge {
    /// An input or a part.
    pub from: String,
    /// A part, or `None` for a result of the job.
    #[cfg_attr(
        feature = "json",
        serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )
    )]
    pub to: Option<String>,
    pub bytes: u64,
    /// Whether the producing part releases this output, so that it may go where its local-only
    /// inputs may not. Only a part releases; an input is never released.
    #[cfg_attr(feature = "json", serde(default))]
    pub release: bool,
}

/// Why a graph is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The job, an input, a part, a chain, a unit or a blob's hash has an empty name.
    EmptyName { what: &'static str },
    /// Two inputs, parts or chains share a name.
    DuplicateName { name: String },
    /// The graph has no part.
    NoParts,
    /// The job's priority is 0.
    ZeroPriority,
    /// A need is not finite or not above 0.
    BadNeed { field: &'static str },
    /// An edge leaves a name that is no input or part.
    UnknownSource { edge: usize, name: String },
    /// An edge enters a name that is no part.
    UnknownTarget { edge: usize, name: String },
    /// An edge enters an input.
    EdgeIntoInput { edge: usize, name: String },
    /// An edge leaving an input is released: only a part releases what it computes.
    ReleaseFromInput { edge: usize, name: String },
    /// A chain has fewer than 2 parts.
    ShortChain { chain: String },
    /// A chain names a part the graph does not have.
    UnknownChainPart { chain: String, part: String },
    /// A chain names a replicable part.
    ReplicableInChain { chain: String, part: String },
    /// A part appears more than once across the chains.
    RepeatedChainPart { part: String },
    /// No edge joins two consecutive parts of a chain.
    ChainGap {
        chain: String,
        from: String,
        to: String,
    },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::EmptyName { what } => write!(f, "a {what} has an empty name"),
            Refusal::DuplicateName { name } => {
                write!(f, "the name `{name}` is used more than once")
            }
            Refusal::NoParts => write!(f, "the graph has no part"),
            Refusal::ZeroPriority => write!(f, "the job's priority is 0; it must be at least 1"),
            Refusal::BadNeed { field } => {
                write!(f, "the need's `{field}` must be finite and above 0")
            }
            Refusal::UnknownSource { edge, name } => {
                write!(f, "edge {edge} leaves `{name}`, which is no input or part")
            }
            Refusal::UnknownTarget { edge, name } => {
                write!(f, "edge {edge} enters `{name}`, which is no part")
            }
            Refusal::EdgeIntoInput { edge, name } => {
                write!(f, "edge {edge} enters `{name}`, which is an input")
            }
            Refusal::ReleaseFromInput { edge, name } => write!(
                f,
                "edge {edge} leaves the input `{name}` and is released; only a part releases \
                 what it computes"
            ),
            Refusal::ShortChain { chain } => {
                write!(f, "the chain `{chain}` has fewer than 2 parts")
            }
            Refusal::UnknownChainPart { chain, part } => {
                write!(f, "the chain `{chain}` names `{part}`, which is no part")
            }
            Refusal::ReplicableInChain { chain, part } => {
                write!(f, "the chain `{chain}` names `{part}`, which is replicable")
            }
            Refusal::RepeatedChainPart { part } => {
                write!(f, "the part `{part}` appears more than once in the chains")
            }
            Refusal::ChainGap { chain, from, to } => write!(
                f,
                "in the chain `{chain}`, no edge goes from `{from}` to the next part, `{to}`"
            ),
        }
    }
}

impl core::error::Error for Refusal {}

/// An option in the graph file is present with a value, or left out: an explicit `null` is
/// refused, so that a value that cannot be written (a need that is not finite, written as
/// `null`) is not read back as no value at all.
#[cfg(feature = "json")]
fn present<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    T::deserialize(d).map(Some)
}

/// What a name in a graph refers to.
#[derive(Clone, Copy)]
enum Named {
    Input,
    Part(usize),
    Chain,
}

impl Graph {
    /// Whether the graph is well formed: names unique and not empty, at least one part, a
    /// priority of at least 1, a need that is finite and above 0, edges that leave an input or a
    /// part and enter a part (or leave the job), no input released, and chains of at least 2
    /// parts that exist, are not replicable, appear in one chain only and are joined by edges in
    /// order. The first problem found is returned.
    ///
    /// Cycles are allowed: a loop is one.
    pub fn check(&self) -> Result<(), Refusal> {
        let names = self.names()?;
        if self.parts.is_empty() {
            return Err(Refusal::NoParts);
        }
        if self.job.priority == 0 {
            return Err(Refusal::ZeroPriority);
        }
        for (field, value) in [
            ("rate_per_second", self.job.need.rate_per_second),
            ("deadline_seconds", self.job.need.deadline_seconds),
        ] {
            if let Some(v) = value
                && !(v.is_finite() && v > 0.0)
            {
                return Err(Refusal::BadNeed { field });
            }
        }
        for (i, edge) in self.edges.iter().enumerate() {
            match names.get(edge.from.as_str()) {
                Some(Named::Input) if edge.release => {
                    return Err(Refusal::ReleaseFromInput {
                        edge: i,
                        name: edge.from.clone(),
                    });
                }
                Some(Named::Input | Named::Part(_)) => {}
                Some(Named::Chain) | None => {
                    return Err(Refusal::UnknownSource {
                        edge: i,
                        name: edge.from.clone(),
                    });
                }
            }
            if let Some(to) = &edge.to {
                match names.get(to.as_str()) {
                    Some(Named::Part(_)) => {}
                    Some(Named::Input) => {
                        return Err(Refusal::EdgeIntoInput {
                            edge: i,
                            name: to.clone(),
                        });
                    }
                    Some(Named::Chain) | None => {
                        return Err(Refusal::UnknownTarget {
                            edge: i,
                            name: to.clone(),
                        });
                    }
                }
            }
        }
        self.check_chains(&names)
    }

    /// Every name, with what it refers to; refuses an empty or repeated one.
    fn names(&self) -> Result<BTreeMap<&str, Named>, Refusal> {
        if self.job.name.is_empty() {
            return Err(Refusal::EmptyName { what: "job" });
        }
        let mut names = BTreeMap::new();
        let all = self
            .inputs
            .iter()
            .map(|x| (x.name.as_str(), Named::Input, "input"))
            .chain(
                self.parts
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (x.name.as_str(), Named::Part(i), "part")),
            )
            .chain(
                self.chains
                    .iter()
                    .map(|x| (x.name.as_str(), Named::Chain, "chain")),
            );
        for (name, named, what) in all {
            if name.is_empty() {
                return Err(Refusal::EmptyName { what });
            }
            if names.insert(name, named).is_some() {
                return Err(Refusal::DuplicateName { name: name.into() });
            }
        }
        for part in &self.parts {
            if part.unit.name.is_empty() {
                return Err(Refusal::EmptyName { what: "unit" });
            }
            if part.weights.iter().any(|w| w.hash.is_empty()) {
                return Err(Refusal::EmptyName {
                    what: "blob's hash",
                });
            }
            if let Some(Pin::Member(member)) = &part.pin
                && member.is_empty()
            {
                return Err(Refusal::EmptyName {
                    what: "pinned member",
                });
            }
        }
        Ok(names)
    }

    fn check_chains(&self, names: &BTreeMap<&str, Named>) -> Result<(), Refusal> {
        let mut seen = BTreeMap::new();
        for chain in &self.chains {
            if chain.parts.len() < 2 {
                return Err(Refusal::ShortChain {
                    chain: chain.name.clone(),
                });
            }
            for part in &chain.parts {
                let Some(&Named::Part(i)) = names.get(part.as_str()) else {
                    return Err(Refusal::UnknownChainPart {
                        chain: chain.name.clone(),
                        part: part.clone(),
                    });
                };
                if self.parts[i].mark == Mark::Replicable {
                    return Err(Refusal::ReplicableInChain {
                        chain: chain.name.clone(),
                        part: part.clone(),
                    });
                }
                if seen.insert(part.as_str(), ()).is_some() {
                    return Err(Refusal::RepeatedChainPart { part: part.clone() });
                }
            }
            for pair in chain.parts.windows(2) {
                let joined = self
                    .edges
                    .iter()
                    .any(|e| e.from == pair[0] && e.to.as_deref() == Some(pair[1].as_str()));
                if !joined {
                    return Err(Refusal::ChainGap {
                        chain: chain.name.clone(),
                        from: pair[0].clone(),
                        to: pair[1].clone(),
                    });
                }
            }
        }
        Ok(())
    }
}
