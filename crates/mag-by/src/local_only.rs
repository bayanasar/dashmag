//! Which parts and edges carry data that must stay local.
//!
//! An input marked local-only never goes to a member marked remote, and neither does anything
//! computed from it, however indirectly, unless the part that computed it releases that output.
//! So a part is local-only when any edge into it is, and an edge is local-only when it leaves a
//! local-only input, or leaves a local-only part without a release. A part reached both through a
//! release and directly stays local-only. Loops make the graph cyclic, so the marks are taken to
//! a fixed point rather than in one pass.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::{Graph, Refusal};

/// Which parts and edges of a graph are local-only, by their index in the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalOnly {
    /// One per part, in the graph's order.
    pub parts: Vec<bool>,
    /// One per edge, in the graph's order.
    pub edges: Vec<bool>,
}

impl Graph {
    /// Which parts and edges carry data computed from a local-only input without a release.
    ///
    /// The graph is checked first, and a graph [`Graph::check`] refuses gets its refusal, not
    /// marks: on a malformed graph the marks would say less than the author meant, and a mark
    /// that says too little lets data move.
    pub fn local_only(&self) -> Result<LocalOnly, Refusal> {
        self.check()?;
        Ok(self.marks())
    }

    /// The marks, on a graph that may not pass `check`. They fail closed: an edge whose source
    /// names nothing known is taken as local-only, so what it feeds is too.
    pub(crate) fn marks(&self) -> LocalOnly {
        let inputs: BTreeMap<&str, bool> = self
            .inputs
            .iter()
            .map(|i| (i.name.as_str(), i.local_only))
            .collect();
        let parts: BTreeMap<&str, usize> = self
            .parts
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.as_str(), i))
            .collect();
        let mut marks = LocalOnly {
            parts: vec![false; self.parts.len()],
            edges: vec![false; self.edges.len()],
        };
        // Marks only ever turn on, so this ends after at most one pass per mark.
        loop {
            let mut changed = false;
            for (i, edge) in self.edges.iter().enumerate() {
                let from = match (
                    inputs.get(edge.from.as_str()),
                    parts.get(edge.from.as_str()),
                ) {
                    (Some(&local), _) => local,
                    (None, Some(&p)) => marks.parts[p] && !edge.release,
                    (None, None) => true,
                };
                if from && !marks.edges[i] {
                    marks.edges[i] = true;
                    changed = true;
                }
                if marks.edges[i]
                    && let Some(&p) = edge.to.as_deref().and_then(|to| parts.get(to))
                    && !marks.parts[p]
                {
                    marks.parts[p] = true;
                    changed = true;
                }
            }
            if !changed {
                return marks;
            }
        }
    }
}
