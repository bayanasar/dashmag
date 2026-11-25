//! How far each part and edge of a graph must be kept: the least trust a machine must have to
//! hold what it carries.
//!
//! The host trusts each member of a pool at a [`Trust`] level, and mag only compares levels. An
//! input, and a blob a part holds as weights, is kept at a level: the least trust a machine must
//! have to hold it. What is computed from data is kept at the highest level among what it read,
//! however indirectly, unless the part that computed it releases that output to a lower level.
//! So a part is kept at the highest level among the blobs it holds and the edges into it; an edge
//! leaving an input at the input's level; and an edge leaving a part at the part's level, or,
//! where the part releases it, at the lower of that and the release's level. A release never
//! makes data stricter than it is. A part reached both through a release and directly keeps the
//! direct level. Loops make the graph cyclic, so levels are raised to a fixed point rather than
//! set in one pass.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::{Graph, Refusal};

/// How far a machine is trusted, and how far data must be kept, in order: `outside < site <
/// owner`. A member is trusted at a level; an input or a blob is kept at a level, and is placed
/// only on a member trusted at least as far.
///
/// The host gives each member its level, and mag judges no one: what a level means is the
/// host's. As a rule of thumb, `owner` is the owner's own machines, `site` a machine inside the
/// owner's site that is not the owner's, and `outside` anything else, a rented machine among
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
    feature = "graph",
    derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)
)]
#[cfg_attr(feature = "graph", serde(rename_all = "snake_case"))]
pub enum Trust {
    /// Trusted with what anyone may hold.
    Outside,
    /// Inside the owner's site, but not the owner's.
    Site,
    /// The owner's own.
    Owner,
}

/// The level's name, as the files write it: `outside`, `site`, `owner`.
impl fmt::Display for Trust {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Trust::Outside => "outside",
            Trust::Site => "site",
            Trust::Owner => "owner",
        })
    }
}

/// The level each part and edge of a graph is kept at, by their index in the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Keeps {
    /// One per part, in the graph's order.
    pub parts: Vec<Trust>,
    /// One per edge, in the graph's order.
    pub edges: Vec<Trust>,
}

impl Graph {
    /// The level each part and edge is kept at: the highest among what it reads, unless
    /// released.
    ///
    /// The graph is checked first, and a graph [`Graph::check`] refuses gets its refusal, not
    /// levels: on a malformed graph the levels would say less than the author meant, and a level
    /// that says too little lets data move.
    pub fn keeps(&self) -> Result<Keeps, Refusal> {
        self.check()?;
        Ok(self.levels())
    }

    /// The levels, on a graph that may not pass `check`. They fail closed: an edge whose source
    /// names nothing known is kept at `owner`, and so is what it feeds.
    pub(crate) fn levels(&self) -> Keeps {
        let inputs: BTreeMap<&str, Trust> = self
            .inputs
            .iter()
            .map(|i| (i.name.as_str(), i.keep))
            .collect();
        let parts: BTreeMap<&str, usize> = self
            .parts
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.as_str(), i))
            .collect();
        let mut keeps = Keeps {
            parts: self
                .parts
                .iter()
                .map(|p| {
                    p.weights
                        .iter()
                        .map(|w| w.keep)
                        .fold(Trust::Outside, Ord::max)
                })
                .collect(),
            edges: vec![Trust::Outside; self.edges.len()],
        };
        // Levels only ever rise, and there are three, so this ends after at most two passes per
        // level.
        loop {
            let mut changed = false;
            for (i, edge) in self.edges.iter().enumerate() {
                let from = match (
                    inputs.get(edge.from.as_str()),
                    parts.get(edge.from.as_str()),
                ) {
                    (Some(&keep), _) => keep,
                    (None, Some(&p)) => match edge.release {
                        Some(to) => keeps.parts[p].min(to),
                        None => keeps.parts[p],
                    },
                    (None, None) => Trust::Owner,
                };
                if from > keeps.edges[i] {
                    keeps.edges[i] = from;
                    changed = true;
                }
                if let Some(&p) = edge.to.as_deref().and_then(|to| parts.get(to))
                    && keeps.edges[i] > keeps.parts[p]
                {
                    keeps.parts[p] = keeps.edges[i];
                    changed = true;
                }
            }
            if !changed {
                return keeps;
            }
        }
    }
}
