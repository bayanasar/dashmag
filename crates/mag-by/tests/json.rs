//! The graph file: read, checked, and written back.

use mag_by::{
    Backend, Blob, Dtype, Edge, Graph, Input, Job, Mark, Need, Part, Pin, ReadError, Refusal, Step,
    Tolerance, Unit,
};

const EXAMPLE: &str = include_str!("../graphs/example.json");

/// The example with one change made to its JSON.
fn changed(change: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(EXAMPLE).unwrap();
    change(&mut v);
    serde_json::to_string(&v).unwrap()
}

#[test]
fn the_example_reads_and_writes_back_byte_for_byte() {
    let g = Graph::from_json(EXAMPLE).unwrap();
    assert_eq!(g.parts.len(), 5);
    assert_eq!(g.to_json(), EXAMPLE);
}

#[test]
fn the_examples_counts_are_released_and_its_labels_stay_local() {
    let g = Graph::from_json(EXAMPLE).unwrap();
    let m = g.local_only();
    let names: Vec<_> = g.parts.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["decode", "b0", "b1", "head", "count"]);
    assert_eq!(m.parts, [true, true, true, true, false]);
    // The labels leave the job unreleased; the count of them leaves released.
    assert_eq!(m.edges[6..], [true, false]);
}

#[test]
fn a_wrong_format_is_refused_first() {
    let other = changed(|v| v["format"] = "other".into());
    assert!(matches!(
        Graph::from_json(&other),
        Err(ReadError::Format { found: Some(f) }) if f == "\"other\""
    ));
    let missing = changed(|v| {
        v.as_object_mut().unwrap().remove("format");
    });
    assert!(matches!(
        Graph::from_json(&missing),
        Err(ReadError::Format { found: None })
    ));
    assert!(matches!(
        Graph::from_json("[]"),
        Err(ReadError::Format { found: None })
    ));
}

#[test]
fn a_newer_version_is_refused_as_newer_not_for_its_fields() {
    let newer = changed(|v| {
        v["version"] = 2.into();
        v["objective"] = "makespan".into();
    });
    let err = Graph::from_json(&newer).unwrap_err();
    assert!(matches!(&err, ReadError::Version { found: Some(f) } if f == "2"));
    assert_eq!(
        err.to_string(),
        "the version is 2; this reader reads version 1"
    );
    let missing = changed(|v| {
        v.as_object_mut().unwrap().remove("version");
    });
    assert!(matches!(
        Graph::from_json(&missing),
        Err(ReadError::Version { found: None })
    ));
}

#[test]
fn unknown_fields_are_refused() {
    for change in [
        (|v: &mut serde_json::Value| v["objective"] = "makespan".into()) as fn(&mut _),
        |v| v["job"]["deadline"] = 1.into(),
        |v| v["parts"][0]["memory"] = 1.into(),
        |v| v["edges"][0]["size"] = 1.into(),
        |v| v["parts"][0]["mark"] = "replicated".into(),
    ] {
        let err = Graph::from_json(&changed(change)).unwrap_err();
        assert!(matches!(err, ReadError::Json(_)), "{err}");
    }
}

#[test]
fn local_only_must_be_written() {
    let missing = changed(|v| {
        v["inputs"][0].as_object_mut().unwrap().remove("local_only");
    });
    let err = Graph::from_json(&missing).unwrap_err();
    assert!(err.to_string().contains("local_only"), "{err}");
}

#[test]
fn null_is_not_an_absent_option() {
    let null_pin = changed(|v| v["parts"][0]["pin"] = serde_json::Value::Null);
    assert!(matches!(
        Graph::from_json(&null_pin),
        Err(ReadError::Json(_))
    ));
    // A need that is not finite is written as null, and so cannot come back as no need.
    let mut g = Graph::from_json(EXAMPLE).unwrap();
    g.job.need.rate_per_second = Some(f64::NAN);
    assert!(g.to_json().contains("\"rate_per_second\": null"));
    assert!(matches!(
        Graph::from_json(&g.to_json()),
        Err(ReadError::Json(_))
    ));
}

#[test]
fn a_graph_that_fails_its_check_is_refused() {
    let duplicate = changed(|v| v["parts"][1]["name"] = "decode".into());
    assert!(matches!(
        Graph::from_json(&duplicate),
        Err(ReadError::Refused(Refusal::DuplicateName { name })) if name == "decode"
    ));
}

#[test]
fn left_out_fields_take_their_defaults() {
    let text = r#"{
        "format": "mag-graph",
        "version": 1,
        "job": { "name": "j", "priority": 1 },
        "parts": [{
            "name": "p",
            "unit": { "name": "u", "version": 3 },
            "step": "token",
            "memory_bytes": 10,
            "tolerance": "bf16"
        }]
    }"#;
    let g = Graph::from_json(text).unwrap();
    assert_eq!(g.job.need, Need::default());
    assert!(g.inputs.is_empty() && g.chains.is_empty() && g.edges.is_empty());
    let p = &g.parts[0];
    assert_eq!(p.mark, Mark::Atomic);
    assert!(p.weights.is_empty());
    assert_eq!(p.pin, None);
    // Written back, every field is present but the absent options.
    let written = g.to_json();
    assert!(written.contains("\"mark\": \"atomic\""));
    assert!(written.contains("\"weights\": []"));
    assert!(!written.contains("\"pin\""));
    assert!(!written.contains("\"need\": {\n      \""));
    assert_eq!(Graph::from_json(&written).unwrap(), g);
}

/// A graph using every variant and every field.
fn everything() -> Graph {
    let steps = [Step::Item, Step::Token, Step::OptimiserStep, Step::Round];
    let tolerances = [
        Tolerance::Exact,
        Tolerance::F32,
        Tolerance::Bf16,
        Tolerance::Int8,
    ];
    let pins = [
        None,
        Some(Pin::Member("member-b".into())),
        Some(Pin::Backend(Backend::Cuda)),
        Some(Pin::Backend(Backend::Rocm)),
        Some(Pin::Backend(Backend::Mlx)),
        Some(Pin::Backend(Backend::Flex)),
    ];
    let parts: Vec<Part> = pins
        .into_iter()
        .enumerate()
        .map(|(i, pin)| Part {
            name: format!("p{i}"),
            unit: Unit {
                name: format!("u{i}"),
                version: i as u32 + 1,
            },
            mark: if i == 5 {
                Mark::Replicable
            } else {
                Mark::Atomic
            },
            step: steps[i % 4],
            memory_bytes: u64::MAX - i as u64,
            weights: (0..i)
                .map(|w| Blob {
                    hash: format!("h{w}"),
                    bytes: w as u64,
                })
                .collect(),
            tolerance: tolerances[i % 4],
            pin,
            note: (i % 2 == 0).then(|| format!("part {i}: why it is here")),
        })
        .collect();
    Graph {
        job: Job {
            name: "everything".into(),
            priority: u32::MAX,
            need: Need {
                rate_per_second: Some(0.1),
                deadline_seconds: Some(1e300),
            },
            note: Some("the job's note, with \"quotes\" and a\nnewline".into()),
        },
        inputs: vec![
            Input {
                name: "local".into(),
                bytes: 0,
                local_only: true,
            },
            Input {
                name: "public".into(),
                bytes: 7,
                local_only: false,
            },
        ],
        chains: vec![mag_by::Chain {
            name: "c".into(),
            parts: vec!["p0".into(), "p1".into(), "p2".into()],
        }],
        edges: vec![
            Edge {
                from: "local".into(),
                to: Some("p0".into()),
                dtype: Dtype::F64,
                bytes: 1,
                release: false,
                note: None,
            },
            Edge {
                from: "p0".into(),
                to: Some("p1".into()),
                dtype: Dtype::F32,
                bytes: 2,
                release: true,
                note: Some("an edge's note".into()),
            },
            Edge {
                from: "p1".into(),
                to: Some("p2".into()),
                dtype: Dtype::Bf16,
                bytes: 3,
                release: false,
                note: None,
            },
            Edge {
                from: "public".into(),
                to: Some("p5".into()),
                dtype: Dtype::F16,
                bytes: 4,
                release: false,
                note: None,
            },
            Edge {
                from: "p5".into(),
                to: None,
                dtype: Dtype::I64,
                bytes: 5,
                release: true,
                note: None,
            },
        ],
        parts,
        note: Some("a graph-wide note".into()),
    }
}

#[test]
fn every_field_and_variant_round_trips() {
    let g = everything();
    g.check().unwrap();
    let text = g.to_json();
    let back = Graph::from_json(&text).unwrap();
    assert_eq!(back, g);
    assert_eq!(back.to_json(), text);
    for word in [
        "\"optimiser_step\"",
        "\"round\"",
        "\"int8\"",
        "\"replicable\"",
        "\"member\": \"member-b\"",
        "\"backend\": \"rocm\"",
        "\"backend\": \"flex\"",
        "\"rate_per_second\": 0.1",
        "\"dtype\": \"f64\"",
        "\"dtype\": \"i64\"",
        "\"note\": \"a graph-wide note\"",
    ] {
        assert!(text.contains(word), "{word} not in\n{text}");
    }
}

#[test]
fn every_dtype_round_trips() {
    for dtype in [
        Dtype::F64,
        Dtype::F32,
        Dtype::Bf16,
        Dtype::F16,
        Dtype::I64,
        Dtype::I32,
        Dtype::U8,
        Dtype::Bool,
        Dtype::Bytes,
    ] {
        let mut g = everything();
        g.edges[0].dtype = dtype;
        assert_eq!(Graph::from_json(&g.to_json()).unwrap(), g, "{dtype:?}");
    }
}

#[test]
fn an_edge_without_a_dtype_is_refused() {
    let missing = changed(|v| {
        v["edges"][0].as_object_mut().unwrap().remove("dtype");
    });
    let err = Graph::from_json(&missing).unwrap_err();
    assert!(err.to_string().contains("dtype"), "{err}");
}
