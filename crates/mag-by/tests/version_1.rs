//! Files of version 1, kept in `tests/v1` as version 1 wrote them, read to the same meaning in
//! version 2's words.

use mag_by::Trust::{Outside, Owner};
use mag_by::{
    Change, Description, Format, Graph, Ledger, LedgerError, PoolFileError, ReadError, Removal,
    Stamp,
};
use serde_json::Value;

const GRAPH_1: &str = include_str!("v1/graph.json");
const POOL_1: &str = include_str!("v1/pool.json");
const LEDGER_1: &str = include_str!("v1/ledger.jsonl");
const GRAPH_2: &str = include_str!("../graphs/example.json");
const POOL_2: &str = include_str!("../pools/example.json");

fn edited(text: &str, change: impl FnOnce(&mut Value)) -> String {
    let mut v: Value = serde_json::from_str(text).unwrap();
    change(&mut v);
    v.to_string()
}

/// The example graph as version 1 wrote it is the example as version 2 writes it: the frames and
/// the head's weights kept at owner, the rest at outside, and the count released to outside.
#[test]
fn a_graph_of_version_1_reads_as_the_same_graph() {
    let g = Graph::from_json(GRAPH_1).unwrap();
    assert_eq!(g, Graph::from_json(GRAPH_2).unwrap());
    assert_eq!(g.to_json(), GRAPH_2);
    assert_eq!(
        g.inputs.iter().map(|i| i.keep).collect::<Vec<_>>(),
        [Owner, Outside]
    );
    let blobs: Vec<_> = g
        .parts
        .iter()
        .flat_map(|p| p.weights.iter().map(|w| w.keep))
        .collect();
    assert_eq!(blobs, [Outside, Outside, Owner]);
    let releases: Vec<_> = g.edges.iter().map(|e| e.release).collect();
    assert_eq!(releases[..7], [None; 7]);
    assert_eq!(releases[7], Some(Outside));
}

#[test]
fn a_release_left_out_or_false_in_version_1_is_none() {
    let left_out = edited(GRAPH_1, |v| {
        v["edges"][0].as_object_mut().unwrap().remove("release");
    });
    assert_eq!(Graph::from_json(&left_out).unwrap().edges[0].release, None);
    let released = edited(GRAPH_1, |v| v["edges"][6]["release"] = true.into());
    assert_eq!(
        Graph::from_json(&released).unwrap().edges[6].release,
        Some(Outside)
    );
}

/// What version 1 refused it still refuses, in its own words; and version 2's words are unknown
/// to it.
#[test]
fn a_graph_of_version_1_is_refused_as_version_1_refused_it() {
    let missing = edited(GRAPH_1, |v| {
        v["inputs"][0].as_object_mut().unwrap().remove("local_only");
    });
    let err = Graph::from_json(&missing).unwrap_err();
    assert!(
        err.to_string().contains("missing field `local_only`"),
        "{err}"
    );
    let blob = edited(GRAPH_1, |v| {
        v["parts"][1]["weights"][0]
            .as_object_mut()
            .unwrap()
            .remove("local_only");
    });
    assert!(matches!(Graph::from_json(&blob), Err(ReadError::Json(_))));
    for bad in [
        edited(GRAPH_1, |v| v["inputs"][0]["local_only"] = "yes".into()),
        edited(GRAPH_1, |v| v["inputs"][0]["local_only"] = Value::Null),
        edited(GRAPH_1, |v| v["edges"][0]["release"] = "outside".into()),
        edited(GRAPH_1, |v| v["inputs"][0]["keep"] = "owner".into()),
        edited(GRAPH_1, |v| {
            let input = v["inputs"][0].as_object_mut().unwrap();
            input.remove("local_only");
            input.insert("keep".into(), "owner".into());
        }),
        edited(GRAPH_1, |v| {
            v["parts"][3]["weights"][0]["keep"] = "outside".into()
        }),
    ] {
        let err = Graph::from_json(&bad).unwrap_err();
        assert!(matches!(err, ReadError::Json(_)), "{err}");
    }
    // A repeated key is still found in the text, before the words are rewritten.
    let repeated = GRAPH_1.replacen(
        "\"local_only\": true",
        "\"local_only\": true, \"local_only\": false",
        1,
    );
    assert!(matches!(
        Graph::from_json(&repeated),
        Err(ReadError::RepeatedKey { at }) if at == "inputs[0].local_only"
    ));
}

/// The example pool as version 1 wrote it is the example as version 2 writes it but for the
/// formats that pool read.
#[test]
fn a_pool_file_of_version_1_reads_as_the_same_pool() {
    let one = Description::from_json(POOL_1).unwrap();
    let two = Description::from_json(POOL_2).unwrap();
    let trust: Vec<_> = one
        .members
        .iter()
        .map(|m| (m.name.as_str(), m.trust))
        .collect();
    assert_eq!(
        trust,
        [
            ("workstation", Owner),
            ("server", Owner),
            ("laptop", Owner),
            ("rented", Outside)
        ]
    );
    let read_by_version_1: Vec<Format> = ["mag-graph", "mag-ledger", "mag-pool"]
        .map(|name| Format {
            name: name.into(),
            version: 1,
        })
        .into();
    assert_eq!(one.formats, read_by_version_1);
    assert_eq!(
        Description {
            formats: two.formats.clone(),
            ..one
        },
        two
    );
    for bad in [
        edited(POOL_1, |v| {
            v["members"][0].as_object_mut().unwrap().remove("remote");
        }),
        edited(POOL_1, |v| v["members"][0]["trust"] = "owner".into()),
        edited(POOL_1, |v| v["members"][0]["remote"] = 1.into()),
    ] {
        let err = Description::from_json(&bad).unwrap_err();
        assert!(matches!(err, PoolFileError::Json(_)), "{err}");
    }
}

/// A ledger version 1 wrote: two jobs, one reading frames kept at owner and released to outside
/// after its first part, one holding weights kept at owner; a member that joins and then reports
/// itself remote; a device that leaves; a job that ends.
#[test]
fn a_ledger_of_version_1_reads_to_the_same_meaning() {
    let read = Ledger::from_lines(LEDGER_1).unwrap();
    assert_eq!((read.version, read.dropped), (1, None));
    let l = read.ledger;
    assert_eq!(
        (l.written_by.as_deref(), l.follows.as_deref()),
        (None, None)
    );
    let trust: Vec<_> = l
        .members
        .iter()
        .map(|m| (m.name.as_str(), m.trust))
        .collect();
    assert_eq!(trust, [("house", Owner), ("rented", Outside)]);
    assert_eq!(l.entries.len(), 6);
    let Change::Submit { graph, stamp } = &l.entries[0].change else {
        panic!("the first entry is a submission");
    };
    assert_eq!(stamp, &Stamp::default());
    assert_eq!(
        graph.inputs.iter().map(|i| i.keep).collect::<Vec<_>>(),
        [Owner, Outside]
    );
    assert_eq!(graph.edges[2].release, Some(Outside));
    assert_eq!(
        graph.edges.iter().filter(|e| e.release.is_some()).count(),
        1
    );
    let Change::Submit { graph, .. } = &l.entries[1].change else {
        panic!("the second entry is a submission");
    };
    assert_eq!(graph.parts[0].weights[0].keep, Owner);
    let reports: Vec<_> = l.entries[2..4]
        .iter()
        .map(|e| match &e.change {
            Change::Join {
                member: Some(m), ..
            } => m.trust,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(reports, [Owner, Outside]);
    assert_eq!(l.entries[3].record.lowered, ["lab"]);
    assert!(
        l.entries
            .iter()
            .all(|e| e.record.lowered.is_empty() || e.at == 3.0)
    );
    // Every guard version 1 recorded kept data at owner from a member trusted at outside.
    let guards: Vec<_> = l
        .entries
        .iter()
        .flat_map(|e| &e.record.candidates)
        .flat_map(|c| c.removed.iter().chain(c.left_out.iter().map(|(_, r)| r)))
        .filter_map(|r| match r {
            Removal::Guard { keep, trust, .. } => Some((*keep, *trust)),
            _ => None,
        })
        .collect();
    assert_eq!(guards.len(), 19);
    assert!(guards.iter().all(|&g| g == (Owner, Outside)));
    // Written again whole, it is version 2, and reads back as the same ledger.
    let again = l.to_lines();
    assert!(again.starts_with(r#"{"format":"mag-ledger","version":2,"#));
    assert!(!again.contains("local_only") && !again.contains("remote"));
    let back = Ledger::from_lines(&again).unwrap();
    assert_eq!((back.ledger, back.version), (l, 2));
}

/// The lines of a ledger of version 1 with one line's JSON changed.
fn ledger_edited(line: usize, change: impl FnOnce(&mut Value)) -> String {
    let mut lines: Vec<String> = LEDGER_1.lines().map(String::from).collect();
    lines[line - 1] = edited(&lines[line - 1], change);
    lines.join("\n") + "\n"
}

#[test]
fn a_ledger_of_version_1_with_words_of_version_2_is_refused_at_their_line() {
    let cases = [
        (ledger_edited(1, |v| v["written_by"] = "0.0.1".into()), 1),
        (ledger_edited(1, |v| v["follows"] = "x".into()), 1),
        (
            ledger_edited(1, |v| v["members"][0]["trust"] = "owner".into()),
            1,
        ),
        (
            ledger_edited(2, |v| {
                v["change"]["submit"]["stamp"] = Value::Object(Default::default())
            }),
            2,
        ),
        (
            ledger_edited(2, |v| v["record"]["lowered"] = Value::Array(vec![])),
            2,
        ),
        (
            ledger_edited(2, |v| {
                v["record"].as_object_mut().unwrap().remove("marked_remote");
            }),
            2,
        ),
        (
            ledger_edited(3, |v| {
                v["record"]["candidates"][2]["removed"]["guard"]["keep"] = "owner".into()
            }),
            3,
        ),
        (
            ledger_edited(3, |v| {
                v["record"]["candidates"][2]["removed"] = serde_json::json!({
                    "stamp": { "job": "served", "device": { "member": "house", "index": 0 } }
                })
            }),
            3,
        ),
        (
            ledger_edited(5, |v| {
                v["change"]["join"]["member"]["trust"] = "owner".into()
            }),
            5,
        ),
    ];
    for (text, line) in cases {
        let err = Ledger::from_lines(&text).unwrap_err();
        assert!(
            matches!(err, LedgerError::Json { line: l, .. } if l == line),
            "line {line}: {err}"
        );
    }
}
