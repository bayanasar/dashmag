//! Local-only carried from inputs to everything computed from them, unless released.

use mag_by::{Edge, Graph, Input, Job, Mark, Need, Part, Step, Tolerance, Unit};

/// A graph of the given inputs (name, local-only) and parts, with edges `(from, to, release)`;
/// `to` empty for a result of the job.
fn graph(inputs: &[(&str, bool)], parts: &[&str], edges: &[(&str, &str, bool)]) -> Graph {
    let g = Graph {
        job: Job {
            name: "job".into(),
            priority: 1,
            need: Need::default(),
        },
        inputs: inputs
            .iter()
            .map(|&(name, local_only)| Input {
                name: name.into(),
                bytes: 1,
                local_only,
            })
            .collect(),
        parts: parts
            .iter()
            .map(|&name| Part {
                name: name.into(),
                unit: Unit {
                    name: "u".into(),
                    version: 1,
                },
                mark: Mark::Atomic,
                step: Step::Item,
                memory_bytes: 1,
                weights: vec![],
                tolerance: Tolerance::F32,
                pin: None,
            })
            .collect(),
        chains: vec![],
        edges: edges
            .iter()
            .map(|&(from, to, release)| Edge {
                from: from.into(),
                to: (!to.is_empty()).then(|| to.into()),
                bytes: 1,
                release,
            })
            .collect(),
    };
    g.check().unwrap();
    g
}

#[test]
fn everything_below_a_local_only_input() {
    let g = graph(
        &[("frames", true)],
        &["a", "b", "c"],
        &[
            ("frames", "a", false),
            ("a", "b", false),
            ("b", "c", false),
            ("c", "", false),
        ],
    );
    let m = g.local_only();
    assert_eq!(m.parts, [true, true, true]);
    assert_eq!(m.edges, [true, true, true, true]);
}

#[test]
fn nothing_when_no_input_is_local_only() {
    let g = graph(
        &[("prompts", false)],
        &["a", "b"],
        &[("prompts", "a", false), ("a", "b", false), ("b", "", false)],
    );
    let m = g.local_only();
    assert_eq!(m.parts, [false, false]);
    assert_eq!(m.edges, [false, false, false]);
}

#[test]
fn a_release_stops_it() {
    let g = graph(
        &[("frames", true)],
        &["detect", "count"],
        &[
            ("frames", "detect", false),
            ("detect", "count", true),
            ("count", "", false),
        ],
    );
    let m = g.local_only();
    assert_eq!(m.parts, [true, false]);
    assert_eq!(m.edges, [true, false, false]);
}

#[test]
fn a_release_on_one_of_two_outputs() {
    let g = graph(
        &[("frames", true)],
        &["detect", "count", "crop"],
        &[
            ("frames", "detect", false),
            ("detect", "count", true),
            ("detect", "crop", false),
        ],
    );
    let m = g.local_only();
    assert_eq!(m.parts, [true, false, true]);
    assert_eq!(m.edges, [true, false, true]);
}

#[test]
fn reached_through_a_release_and_directly_stays_local_only() {
    let g = graph(
        &[("frames", true)],
        &["detect", "merge"],
        &[
            ("frames", "detect", false),
            ("detect", "merge", true),
            ("frames", "merge", false),
        ],
    );
    let m = g.local_only();
    assert_eq!(m.parts, [true, true]);
    assert_eq!(m.edges, [true, false, true]);
}

#[test]
fn two_inputs_one_local_only() {
    let g = graph(
        &[("public", false), ("private", true)],
        &["p", "s", "merge"],
        &[
            ("public", "p", false),
            ("private", "s", false),
            ("p", "merge", false),
            ("s", "merge", false),
        ],
    );
    let m = g.local_only();
    assert_eq!(m.parts, [false, true, true]);
    assert_eq!(m.edges, [false, true, false, true]);
}

#[test]
fn around_a_cycle_whatever_the_edges_order() {
    // The input enters the loop last in the edges' order, so one pass in order would miss `a`
    // and `c`.
    let g = graph(
        &[("frames", true)],
        &["a", "b", "c"],
        &[
            ("b", "a", false),
            ("a", "c", false),
            ("a", "b", false),
            ("frames", "b", false),
        ],
    );
    let m = g.local_only();
    assert_eq!(m.parts, [true, true, true]);
    assert_eq!(m.edges, [true, true, true, true]);
}

#[test]
fn a_released_loop_edge_lets_nothing_back_out() {
    // A learner fed local-only data releases its weights to the rollouts; the rollouts' own
    // outputs are computed from released data only.
    let g = graph(
        &[("frames", true), ("prompts", false)],
        &["learner", "rollout"],
        &[
            ("frames", "learner", false),
            ("learner", "rollout", true),
            ("prompts", "rollout", false),
            ("rollout", "learner", false),
            ("rollout", "", false),
        ],
    );
    let m = g.local_only();
    assert_eq!(m.parts, [true, false]);
    assert_eq!(m.edges, [true, false, false, false, false]);
}
