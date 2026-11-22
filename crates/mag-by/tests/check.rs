//! Each refusal of `Graph::check`, triggered once, and graphs that pass.

use mag_by::{
    Backend, Blob, Chain, Dtype, Edge, Graph, Input, Job, Mark, Need, Part, Pin, Refusal, Step,
    Tolerance, Unit,
};

fn part(name: &str, mark: Mark) -> Part {
    Part {
        name: name.into(),
        unit: Unit {
            name: "block".into(),
            version: 1,
        },
        mark,
        step: Step::Item,
        memory_bytes: 1 << 20,
        weights: vec![Blob {
            hash: "h".into(),
            bytes: 4,
        }],
        tolerance: Tolerance::F32,
        pin: None,
        note: None,
    }
}

fn edge(from: &str, to: Option<&str>) -> Edge {
    Edge {
        from: from.into(),
        to: to.map(Into::into),
        dtype: Dtype::Bf16,
        bytes: 8,
        release: false,
        note: None,
    }
}

/// An input read by a replicable part, a chain of three blocks after it, and a result.
fn valid() -> Graph {
    Graph {
        job: Job {
            name: "job".into(),
            priority: 1,
            need: Need {
                rate_per_second: Some(20.0),
                deadline_seconds: None,
            },
            note: None,
        },
        inputs: vec![Input {
            name: "frames".into(),
            bytes: 1000,
            local_only: true,
        }],
        parts: vec![
            part("decode", Mark::Replicable),
            part("b0", Mark::Atomic),
            part("b1", Mark::Atomic),
            part("b2", Mark::Atomic),
        ],
        chains: vec![Chain {
            name: "model".into(),
            parts: vec!["b0".into(), "b1".into(), "b2".into()],
        }],
        edges: vec![
            edge("frames", Some("decode")),
            edge("decode", Some("b0")),
            edge("b0", Some("b1")),
            edge("b1", Some("b2")),
            edge("b2", None),
        ],
        note: None,
    }
}

fn refused(change: impl FnOnce(&mut Graph)) -> Refusal {
    let mut g = valid();
    change(&mut g);
    g.check().expect_err("the change should be refused")
}

#[test]
fn a_valid_graph_passes() {
    valid().check().unwrap();
}

#[test]
fn a_loop_and_pins_pass() {
    let mut g = valid();
    // The last block feeds the first again: a loop is a cycle.
    g.edges.push(edge("b2", Some("b0")));
    g.parts[0].pin = Some(Pin::Backend(Backend::Rocm));
    g.parts[1].pin = Some(Pin::Member("member-a".into()));
    g.edges[2].release = true;
    g.check().unwrap();
}

#[test]
fn empty_names() {
    for (what, change) in [
        (
            "job",
            (|g: &mut Graph| g.job.name.clear()) as fn(&mut Graph),
        ),
        ("input", |g| g.inputs[0].name.clear()),
        ("part", |g| g.parts[3].name.clear()),
        ("chain", |g| g.chains[0].name.clear()),
        ("unit", |g| g.parts[1].unit.name.clear()),
        ("blob's hash", |g| g.parts[2].weights[0].hash.clear()),
        ("pinned member", |g| {
            g.parts[0].pin = Some(Pin::Member(String::new()))
        }),
    ] {
        assert_eq!(refused(change), Refusal::EmptyName { what }, "{what}");
    }
}

#[test]
fn duplicate_names_across_inputs_parts_and_chains() {
    let name = |n: &str| Refusal::DuplicateName { name: n.into() };
    assert_eq!(refused(|g| g.parts[1].name = "b1".into()), name("b1"));
    assert_eq!(
        refused(|g| g.inputs[0].name = "decode".into()),
        name("decode")
    );
    assert_eq!(
        refused(|g| g.chains[0].name = "frames".into()),
        name("frames")
    );
}

#[test]
fn no_parts() {
    let mut g = valid();
    g.parts.clear();
    g.chains.clear();
    g.edges.clear();
    assert_eq!(g.check(), Err(Refusal::NoParts));
}

#[test]
fn zero_priority() {
    assert_eq!(refused(|g| g.job.priority = 0), Refusal::ZeroPriority);
}

#[test]
fn bad_needs() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            refused(|g| g.job.need.rate_per_second = Some(bad)),
            Refusal::BadNeed {
                field: "rate_per_second"
            }
        );
        assert_eq!(
            refused(|g| g.job.need.deadline_seconds = Some(bad)),
            Refusal::BadNeed {
                field: "deadline_seconds"
            }
        );
    }
}

#[test]
fn edges_with_unknown_ends() {
    assert_eq!(
        refused(|g| g.edges[1].from = "nothing".into()),
        Refusal::UnknownSource {
            edge: 1,
            name: "nothing".into()
        }
    );
    // A chain is a name, but not something data leaves or enters.
    assert_eq!(
        refused(|g| g.edges[1].from = "model".into()),
        Refusal::UnknownSource {
            edge: 1,
            name: "model".into()
        }
    );
    assert_eq!(
        refused(|g| g.edges[4].to = Some("model".into())),
        Refusal::UnknownTarget {
            edge: 4,
            name: "model".into()
        }
    );
    assert_eq!(
        refused(|g| g.edges[0].to = Some("frames".into())),
        Refusal::EdgeIntoInput {
            edge: 0,
            name: "frames".into()
        }
    );
}

#[test]
fn a_replicable_part_needs_a_source_of_items() {
    assert_eq!(
        refused(|g| {
            g.edges.remove(0);
        }),
        Refusal::ReplicableWithoutSource {
            part: "decode".into()
        }
    );
}

#[test]
fn an_input_is_never_released() {
    assert_eq!(
        refused(|g| g.edges[0].release = true),
        Refusal::ReleaseFromInput {
            edge: 0,
            name: "frames".into()
        }
    );
}

#[test]
fn chains() {
    let chain = || "model".to_string();
    assert_eq!(
        refused(|g| g.chains[0].parts.truncate(1)),
        Refusal::ShortChain { chain: chain() }
    );
    assert_eq!(
        refused(|g| g.chains[0].parts[1] = "frames".into()),
        Refusal::UnknownChainPart {
            chain: chain(),
            part: "frames".into()
        }
    );
    assert_eq!(
        refused(|g| g.chains[0].parts[0] = "decode".into()),
        Refusal::ReplicableInChain {
            chain: chain(),
            part: "decode".into()
        }
    );
    assert_eq!(
        refused(|g| g.chains.push(Chain {
            name: "again".into(),
            parts: vec!["b2".into(), "b0".into()],
        })),
        Refusal::RepeatedChainPart { part: "b2".into() }
    );
    assert_eq!(
        refused(|g| g.chains[0].parts.push("b0".into())),
        Refusal::RepeatedChainPart { part: "b0".into() }
    );
    // The edge between the second and third block is gone.
    assert_eq!(
        refused(|g| {
            g.edges.remove(3);
        }),
        Refusal::ChainGap {
            chain: chain(),
            from: "b1".into(),
            to: "b2".into()
        }
    );
}

#[test]
fn refusals_say_what_is_wrong() {
    let text = Refusal::ChainGap {
        chain: "model".into(),
        from: "b1".into(),
        to: "b2".into(),
    }
    .to_string();
    assert_eq!(
        text,
        "in the chain `model`, no edge goes from `b1` to the next part, `b2`"
    );
}
