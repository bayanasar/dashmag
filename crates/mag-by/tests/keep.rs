//! Levels carried from inputs and blobs to everything computed from them, unless released.

use mag_by::Trust::{Outside, Owner, Site};
use mag_by::{Dtype, Edge, Graph, Input, Job, Mark, Need, Part, Step, Tolerance, Trust, Unit};

/// A graph of the given inputs (name, level) and parts, with edges `(from, to, release)`; `to`
/// empty for a result of the job.
fn graph(inputs: &[(&str, Trust)], parts: &[&str], edges: &[(&str, &str, Option<Trust>)]) -> Graph {
    let g = Graph {
        job: Job {
            name: "job".into(),
            priority: 1,
            need: Need::default(),
            note: None,
        },
        inputs: inputs
            .iter()
            .map(|&(name, keep)| Input {
                name: name.into(),
                bytes: 1,
                keep,
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
                note: None,
            })
            .collect(),
        chains: vec![],
        edges: edges
            .iter()
            .map(|&(from, to, release)| Edge {
                from: from.into(),
                to: (!to.is_empty()).then(|| to.into()),
                dtype: Dtype::Bf16,
                bytes: 1,
                release,
                note: None,
            })
            .collect(),
        note: None,
    };
    g.check().unwrap();
    g
}

#[test]
fn the_levels_are_ordered_and_named() {
    assert!(Outside < Site && Site < Owner);
    assert_eq!(
        [Outside, Site, Owner].map(|t| t.to_string()),
        ["outside", "site", "owner"]
    );
}

#[test]
fn everything_below_an_input_kept_at_owner() {
    let g = graph(
        &[("frames", Owner)],
        &["a", "b", "c"],
        &[
            ("frames", "a", None),
            ("a", "b", None),
            ("b", "c", None),
            ("c", "", None),
        ],
    );
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Owner, Owner, Owner]);
    assert_eq!(k.edges, [Owner, Owner, Owner, Owner]);
}

#[test]
fn nothing_when_every_input_is_kept_at_outside() {
    let g = graph(
        &[("prompts", Outside)],
        &["a", "b"],
        &[("prompts", "a", None), ("a", "b", None), ("b", "", None)],
    );
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Outside, Outside]);
    assert_eq!(k.edges, [Outside, Outside, Outside]);
}

#[test]
fn a_release_to_outside_stops_it() {
    let g = graph(
        &[("frames", Owner)],
        &["detect", "count"],
        &[
            ("frames", "detect", None),
            ("detect", "count", Some(Outside)),
            ("count", "", None),
        ],
    );
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Owner, Outside]);
    assert_eq!(k.edges, [Owner, Outside, Outside]);
}

/// A part kept at `owner` releases one output to each level: each is kept at the level named,
/// and the part that reads it with it.
#[test]
fn a_release_releases_to_the_level_it_names() {
    let g = graph(
        &[("frames", Owner)],
        &["detect", "to_outside", "to_site", "to_owner"],
        &[
            ("frames", "detect", None),
            ("detect", "to_outside", Some(Outside)),
            ("detect", "to_site", Some(Site)),
            ("detect", "to_owner", Some(Owner)),
        ],
    );
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Owner, Outside, Site, Owner]);
    assert_eq!(k.edges, [Owner, Outside, Site, Owner]);
}

/// A release never keeps data higher than it is: a part kept at `site` releasing to `owner`
/// leaves its output at `site`, and one kept at `outside` leaves it at `outside`.
#[test]
fn a_release_never_raises_a_level() {
    let g = graph(
        &[("lab", Site), ("public", Outside)],
        &["s", "o", "after_s", "after_o"],
        &[
            ("lab", "s", None),
            ("public", "o", None),
            ("s", "after_s", Some(Owner)),
            ("o", "after_o", Some(Site)),
        ],
    );
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Site, Outside, Site, Outside]);
    assert_eq!(k.edges, [Site, Outside, Site, Outside]);
}

/// A part reading inputs of different levels is kept at the highest of them, and so is what it
/// computes, unless it releases it; each release to each level gives that level, or the part's
/// own where that is lower.
#[test]
fn a_part_reading_several_levels_takes_the_highest() {
    for (inputs, highest) in [
        ([Outside, Site], Site),
        ([Site, Owner], Owner),
        ([Outside, Owner], Owner),
        ([Site, Site], Site),
        ([Outside, Outside], Outside),
    ] {
        for release in [None, Some(Outside), Some(Site), Some(Owner)] {
            let g = graph(
                &[("x", inputs[0]), ("y", inputs[1])],
                &["merge", "next"],
                &[
                    ("x", "merge", None),
                    ("y", "merge", None),
                    ("merge", "next", release),
                ],
            );
            let k = g.keeps().unwrap();
            let after = release.map_or(highest, |to| to.min(highest));
            assert_eq!(k.parts, [highest, after], "{inputs:?}, {release:?}");
            assert_eq!(
                k.edges,
                [inputs[0], inputs[1], after],
                "{inputs:?}, {release:?}"
            );
        }
    }
}

#[test]
fn a_release_on_one_of_two_outputs() {
    let g = graph(
        &[("frames", Owner)],
        &["detect", "count", "crop"],
        &[
            ("frames", "detect", None),
            ("detect", "count", Some(Outside)),
            ("detect", "crop", None),
        ],
    );
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Owner, Outside, Owner]);
    assert_eq!(k.edges, [Owner, Outside, Owner]);
}

#[test]
fn reached_through_a_release_and_directly_keeps_the_direct_level() {
    let g = graph(
        &[("frames", Owner)],
        &["detect", "merge"],
        &[
            ("frames", "detect", None),
            ("detect", "merge", Some(Outside)),
            ("frames", "merge", None),
        ],
    );
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Owner, Owner]);
    assert_eq!(k.edges, [Owner, Outside, Owner]);
}

#[test]
fn two_inputs_one_kept_at_owner() {
    let g = graph(
        &[("public", Outside), ("private", Owner)],
        &["p", "s", "merge"],
        &[
            ("public", "p", None),
            ("private", "s", None),
            ("p", "merge", None),
            ("s", "merge", None),
        ],
    );
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Outside, Owner, Owner]);
    assert_eq!(k.edges, [Outside, Owner, Outside, Owner]);
}

#[test]
fn around_a_cycle_whatever_the_edges_order() {
    // The input enters the loop last in the edges' order, so one pass in order would miss `a`
    // and `c`. A second input at `site` enters first, so `a` and `c` rise twice.
    let g = graph(
        &[("frames", Owner), ("lab", Site)],
        &["a", "b", "c"],
        &[
            ("lab", "a", None),
            ("b", "a", None),
            ("a", "c", None),
            ("a", "b", None),
            ("frames", "b", None),
        ],
    );
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Owner, Owner, Owner]);
    assert_eq!(k.edges, [Site, Owner, Owner, Owner, Owner]);
}

#[test]
fn a_released_loop_edge_lets_nothing_back_out() {
    // A learner fed data kept at owner releases its weights to the rollouts; the rollouts' own
    // outputs are computed from released data only.
    let g = graph(
        &[("frames", Owner), ("prompts", Outside)],
        &["learner", "rollout"],
        &[
            ("frames", "learner", None),
            ("learner", "rollout", Some(Outside)),
            ("prompts", "rollout", None),
            ("rollout", "learner", None),
            ("rollout", "", None),
        ],
    );
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Owner, Outside]);
    assert_eq!(k.edges, [Owner, Outside, Outside, Outside, Outside]);
}

/// A graph whose input kept at owner was renamed while an edge still names the old name:
/// `check` refuses it, and so does `keeps`, rather than calling everything free to move.
#[test]
fn a_graph_that_fails_its_check_gets_no_levels() {
    let mut g = graph(
        &[("frames", Owner)],
        &["a", "b"],
        &[("frames", "a", None), ("a", "b", None)],
    );
    g.inputs[0].name = "framez".into();
    let refused = g.check().unwrap_err();
    assert_eq!(g.keeps(), Err(refused));
    // Drawn anyway, for its author to see, with the unknown edge and what it feeds kept at owner.
    let dot = g.to_dot();
    assert!(
        dot.contains("\"frames\" -> \"a\" [label=\"bf16 1 B\", color=red];"),
        "{dot}"
    );
    assert!(
        dot.contains("\"b\" [shape=box, label=\"b\\nu v1\\natomic, stops each item\", color=red];"),
        "{dot}"
    );
}

/// Weights trained in an earlier job on data kept at a level, served in this one: no input here
/// is kept above outside, but the part holding them is kept at their level, and so is what it
/// computes unless released.
#[test]
fn a_blob_s_level_is_the_level_of_the_part_that_holds_it() {
    let mut g = graph(
        &[("prompts", Outside)],
        &["adapter", "serve", "count"],
        &[
            ("prompts", "adapter", None),
            ("adapter", "serve", None),
            ("serve", "", None),
            ("adapter", "count", Some(Outside)),
        ],
    );
    let blob = |keep| mag_by::Blob {
        hash: "trained-at-home".into(),
        bytes: 1 << 20,
        keep,
    };
    g.parts[0].weights.push(blob(Owner));
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Owner, Owner, Outside]);
    assert_eq!(k.edges, [Outside, Owner, Owner, Outside]);
    // Two blobs: the part takes the higher.
    g.parts[0].weights[0].keep = Site;
    g.parts[0].weights.push(blob(Outside));
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Site, Site, Outside]);
    // The same weights at outside keep nothing.
    g.parts[0].weights[0].keep = Outside;
    let k = g.keeps().unwrap();
    assert_eq!(k.parts, [Outside, Outside, Outside]);
}
