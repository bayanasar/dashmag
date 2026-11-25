//! A graph as Graphviz DOT, for a person to look at. One way only: DOT is never read back.
//!
//! Inputs are ellipses, parts boxes (a replicable part a stack of them), a chain's parts sit in
//! a cluster, and the job's results meet in one node. Each edge is labelled with its dtype and
//! size, the size in binary units (`1.12 GiB`). It is laid out top to bottom, so that a graph
//! shown in a column of text keeps its labels readable. An input, a part or an edge kept at
//! `owner` is red and one kept at `site` orange, an input labelled with its level, and a released
//! edge is dashed and labelled with the level it is released to, so that what may leave which
//! machines can be told apart at a glance.

use std::fmt::Write;
use std::string::String;

use crate::{Graph, Mark, Trust};

/// The colour of what is kept at a level, as a DOT attribute: none for `outside`.
fn level_colour(keep: Trust) -> &'static str {
    match keep {
        Trust::Outside => "",
        Trust::Site => ", color=orange",
        Trust::Owner => ", color=red",
    }
}

/// `s` as a quoted DOT identifier.
fn quoted(s: &str) -> String {
    let mut q = String::with_capacity(s.len() + 2);
    q.push('"');
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                q.push('\\');
                q.push(c);
            }
            '\n' => q.push_str("\\n"),
            c => q.push(c),
        }
    }
    q.push('"');
    q
}

fn step(s: crate::Step) -> &'static str {
    match s {
        crate::Step::Item => "item",
        crate::Step::Token => "token",
        crate::Step::OptimiserStep => "optimiser step",
        crate::Step::Round => "round",
    }
}

fn dtype(d: crate::Dtype) -> &'static str {
    use crate::Dtype::*;
    match d {
        F64 => "f64",
        F32 => "f32",
        Bf16 => "bf16",
        F16 => "f16",
        I64 => "i64",
        I32 => "i32",
        I8 => "i8",
        U8 => "u8",
        Bool => "bool",
        Bytes => "bytes",
    }
}

/// `bytes` in binary units, to three significant digits: `64 B`, `4 KiB`, `1.12 GiB`.
fn size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit + 1 < UNITS.len() {
        v /= 1024.0;
        unit += 1;
    }
    let mut n = if unit == 0 || v >= 100.0 {
        std::format!("{v:.0}")
    } else if v >= 10.0 {
        std::format!("{v:.1}")
    } else {
        std::format!("{v:.2}")
    };
    if n.contains('.') {
        while n.ends_with('0') {
            n.pop();
        }
        if n.ends_with('.') {
            n.pop();
        }
    }
    std::format!("{n} {}", UNITS[unit])
}

/// The node the job's results go to; a name no input or part can take, since it holds a space.
const RESULTS: &str = "job's results";

impl Graph {
    /// The graph as Graphviz DOT, for a person to look at (`dot -Tsvg`). Never read back. A
    /// graph that [`Graph::check`] refuses is drawn too, so that its author can see what is wrong;
    /// its levels fail closed, and an edge from a name nothing defines is drawn kept at `owner`.
    pub fn to_dot(&self) -> String {
        let keeps = self.levels();
        let mut d = String::new();
        let w = &mut d;
        let _ = writeln!(w, "digraph {} {{", quoted(&self.job.name));
        let _ = writeln!(w, "  rankdir=TB;");
        let _ = writeln!(w, "  node [fontname=\"monospace\"];");
        let _ = writeln!(w, "  edge [fontname=\"monospace\"];");
        for input in &self.inputs {
            let mut label = std::format!("{}\n{}", input.name, size(input.bytes));
            if input.keep > Trust::Outside {
                let _ = write!(label, "\nkept at {}", input.keep);
            }
            let colour = level_colour(input.keep);
            let _ = writeln!(
                w,
                "  {} [shape=ellipse, label={}{colour}];",
                quoted(&input.name),
                quoted(&label)
            );
        }
        let in_chain = |name: &str| {
            self.chains
                .iter()
                .any(|c| c.parts.iter().any(|p| p == name))
        };
        let part_line = |i: usize, indent: &str| {
            let p = &self.parts[i];
            let shape = match p.mark {
                Mark::Atomic => "box",
                Mark::Replicable => "box3d",
            };
            let mark = match p.mark {
                Mark::Atomic => "atomic",
                Mark::Replicable => "replicable",
            };
            let label = std::format!(
                "{}\n{} v{}\n{mark}, stops each {}",
                p.name,
                p.unit.name,
                p.unit.version,
                step(p.step)
            );
            let colour = level_colour(keeps.parts[i]);
            std::format!(
                "{indent}{} [shape={shape}, label={}{colour}];",
                quoted(&p.name),
                quoted(&label)
            )
        };
        for (i, p) in self.parts.iter().enumerate() {
            if !in_chain(&p.name) {
                let _ = writeln!(w, "{}", part_line(i, "  "));
            }
        }
        for chain in &self.chains {
            let _ = writeln!(
                w,
                "  subgraph {} {{",
                quoted(&std::format!("cluster {}", chain.name))
            );
            let _ = writeln!(
                w,
                "    label={};",
                quoted(&std::format!("chain {}", chain.name))
            );
            for name in &chain.parts {
                if let Some(i) = self.parts.iter().position(|p| &p.name == name) {
                    let _ = writeln!(w, "{}", part_line(i, "    "));
                }
            }
            let _ = writeln!(w, "  }}");
        }
        if self.edges.iter().any(|e| e.to.is_none()) {
            let _ = writeln!(
                w,
                "  {} [shape=doublecircle, label=\"results\"];",
                quoted(RESULTS)
            );
        }
        for (i, e) in self.edges.iter().enumerate() {
            let mut label = std::format!("{} {}", dtype(e.dtype), size(e.bytes));
            let mut style = String::new();
            if let Some(to) = e.release {
                let _ = write!(label, "\nreleased to {to}");
                style.push_str(", style=dashed");
            }
            style.push_str(level_colour(keeps.edges[i]));
            let _ = writeln!(
                w,
                "  {} -> {} [label={}{style}];",
                quoted(&e.from),
                quoted(e.to.as_deref().unwrap_or(RESULTS)),
                quoted(&label)
            );
        }
        let _ = writeln!(w, "}}");
        d
    }
}
