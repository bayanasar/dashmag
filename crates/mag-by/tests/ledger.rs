//! The ledger's lines: written, read back, refused as the graph file is, and held to their kept
//! schema.

use std::path::Path;

use mag_by::{
    Arrangement, Backend, Candidate, Change, Cost, DeviceId, Dtype, Entry, Event, Graph, Ledger,
    LedgerError, Member, Move, Offer, Place, Record, Removal, Tolerance, Unit, ledger_schema,
};
use serde_json::Value;

const EXAMPLE: &str = include_str!("../graphs/example.json");

fn device(member: &str, index: u32) -> DeviceId {
    DeviceId {
        member: member.into(),
        index,
    }
}

fn member(name: &str, remote: bool) -> Member {
    Member {
        name: name.into(),
        remote,
        give_back_seconds: 5.0,
    }
}

fn offer(d: DeviceId) -> Offer {
    Offer {
        device: d,
        backend: Backend::Cuda,
        memory_bytes: 16 << 30,
        tolerances: vec![Tolerance::Bf16],
        dtypes: vec![Dtype::Bf16, Dtype::F32],
        costs: vec![Cost {
            unit: Unit {
                name: "u".into(),
                version: 1,
            },
            seconds_per_step: 0.01,
            band: 0.05,
            estimate: false,
        }],
    }
}

fn record(event: Event, candidates: Vec<Candidate>, chosen: Option<usize>) -> Record {
    Record {
        event,
        candidates,
        chosen,
        greedy: false,
        marked_remote: vec![],
    }
}

/// A ledger with each kind of change, and records with every option both present and left out.
fn sample() -> Ledger {
    let graph = Graph::from_json(EXAMPLE).unwrap();
    let job = graph.job.name.clone();
    let placed = Arrangement {
        job: job.clone(),
        places: vec![
            Place::On(device("a", 0)),
            Place::Copies(vec![device("a", 0), device("b", 0)]),
            Place::Waiting,
        ],
    };
    Ledger {
        at: 0.0,
        members: vec![member("a", false), member("b", false)],
        offers: vec![offer(device("a", 0))],
        entries: vec![
            Entry {
                at: 0.0,
                change: Change::Submit { graph },
                record: record(
                    Event::Arrival { job: job.clone() },
                    vec![
                        Candidate {
                            arrangements: vec![placed.clone()],
                            moves: vec![Move::Fit],
                            removed: Some(Removal::Guard {
                                part: "decode".into(),
                                device: device("b", 0),
                            }),
                            left_out: vec![],
                            score: None,
                            band: None,
                        },
                        Candidate {
                            arrangements: vec![placed],
                            moves: vec![Move::Wait],
                            removed: None,
                            left_out: vec![(
                                "b0".into(),
                                Removal::Memory {
                                    device: device("b", 0),
                                },
                            )],
                            score: Some(1.0 / 3.0),
                            band: Some(0.05),
                        },
                    ],
                    Some(1),
                ),
            },
            Entry {
                at: 1.5,
                change: Change::Join {
                    member: Some(member("b", true)),
                    offer: offer(device("b", 1)),
                },
                record: Record {
                    greedy: true,
                    marked_remote: vec!["b".into()],
                    ..record(
                        Event::Join {
                            device: device("b", 1),
                        },
                        vec![],
                        None,
                    )
                },
            },
            Entry {
                at: 2.0,
                change: Change::Join {
                    member: None,
                    offer: offer(device("a", 1)),
                },
                record: record(
                    Event::Join {
                        device: device("a", 1),
                    },
                    vec![],
                    None,
                ),
            },
            Entry {
                at: 3.0,
                change: Change::Leave {
                    device: device("a", 0),
                },
                record: record(
                    Event::Leave {
                        device: device("a", 0),
                    },
                    vec![],
                    None,
                ),
            },
            Entry {
                at: 4.25,
                change: Change::End { job: job.clone() },
                record: record(Event::End { job }, vec![], None),
            },
        ],
    }
}

/// The sample's lines with one line's JSON changed.
fn changed(line: usize, change: impl FnOnce(&mut Value)) -> String {
    let text = sample().to_lines();
    let mut lines: Vec<String> = text.lines().map(String::from).collect();
    let mut v: Value = serde_json::from_str(&lines[line - 1]).unwrap();
    change(&mut v);
    lines[line - 1] = v.to_string();
    lines.join("\n") + "\n"
}

#[test]
fn a_ledger_written_as_lines_reads_back_and_writes_back_byte_for_byte() {
    let ledger = sample();
    let text = ledger.to_lines();
    assert_eq!(text.lines().count(), 1 + ledger.entries.len());
    assert!(text.starts_with(r#"{"format":"mag-ledger","version":1,"at":0.0,"members":"#));
    let read = Ledger::from_lines(&text).unwrap();
    assert_eq!(read.dropped, None);
    assert_eq!(read.ledger, ledger);
    assert_eq!(read.ledger.to_lines(), text);
    // Options left out are not written as null; a join without a report has no member.
    assert!(!text.contains("null"));
    assert!(text.lines().nth(3).unwrap().contains(r#""join":{"offer""#));
    // A ledger begun and not yet told anything is its header alone.
    let empty = Ledger {
        entries: vec![],
        ..ledger
    };
    assert_eq!(
        Ledger::from_lines(&empty.header_line()).unwrap().ledger,
        empty
    );
}

#[test]
fn the_header_s_format_and_version_are_checked_first() {
    let other = changed(1, |v| v["format"] = "mag-graph".into());
    assert!(matches!(
        Ledger::from_lines(&other),
        Err(LedgerError::Format { found: Some(f) }) if f == "\"mag-graph\""
    ));
    let newer = changed(1, |v| {
        v["version"] = 2.into();
        v["checkpoints"] = Value::Array(vec![]);
    });
    let err = Ledger::from_lines(&newer).unwrap_err();
    assert!(matches!(&err, LedgerError::Version { found: Some(f) } if f == "2"));
    assert_eq!(
        err.to_string(),
        "the version is 2; this reader reads version 1"
    );
    assert!(matches!(
        Ledger::from_lines(""),
        Err(LedgerError::Format { found: None })
    ));
    assert!(matches!(
        Ledger::from_lines("[]\n"),
        Err(LedgerError::Format { found: None })
    ));
}

#[test]
fn unknown_fields_repeated_keys_and_null_are_refused_with_their_line() {
    let unknown = [
        changed(1, |v| v["owner"] = "x".into()),
        changed(2, |v| v["note"] = "x".into()),
        changed(2, |v| v["record"]["candidates"][0]["why"] = "x".into()),
        changed(3, |v| v["change"]["join"]["owner"] = "x".into()),
    ];
    for (text, line) in unknown.iter().zip([1, 2, 2, 3]) {
        let err = Ledger::from_lines(text).unwrap_err();
        assert!(
            matches!(err, LedgerError::Json { line: l, .. } if l == line),
            "{err}"
        );
    }

    let nulls = [
        (changed(2, |v| v["record"]["chosen"] = Value::Null), 2),
        (
            changed(2, |v| v["record"]["candidates"][1]["score"] = Value::Null),
            2,
        ),
        (
            changed(3, |v| v["change"]["join"]["member"] = Value::Null),
            3,
        ),
        (
            changed(4, |v| v["change"]["join"]["member"] = Value::Null),
            4,
        ),
    ];
    for (text, line) in nulls {
        let err = Ledger::from_lines(&text).unwrap_err();
        assert!(
            matches!(err, LedgerError::Json { line: l, .. } if l == line),
            "{err}"
        );
    }

    let text = sample().to_lines();
    let repeated = text.replacen(r#""remote":false,"#, r#""remote":false,"remote":true,"#, 1);
    assert!(matches!(
        Ledger::from_lines(&repeated),
        Err(LedgerError::RepeatedKey { line: 1, at }) if at == "members[0].remote"
    ));
    let repeated = text.replacen(r#""chosen":1,"#, r#""chosen":1,"chosen":0,"#, 1);
    assert!(matches!(
        Ledger::from_lines(&repeated),
        Err(LedgerError::RepeatedKey { line: 2, at }) if at == "record.chosen"
    ));
}

#[test]
fn a_line_that_is_not_json_is_refused_and_one_that_ends_early_is_torn() {
    let text = sample().to_lines();
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let garbage = format!(
        "{}x{}\n{}",
        lines[..2].concat(),
        lines[2],
        lines[3..].concat()
    );
    assert!(matches!(
        Ledger::from_lines(&garbage),
        Err(LedgerError::Json { line: 3, .. })
    ));
    // An empty line between two others ends before it begins.
    let empty = format!("{}\n{}", lines[..2].concat(), lines[2..].concat());
    assert!(matches!(
        Ledger::from_lines(&empty),
        Err(LedgerError::Torn { line: 3 })
    ));
}

fn kept() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("schemas/mag-ledger.schema.json")
}

/// Fails when the kept schema is not what the types generate. `MAG_BY_WRITE_SCHEMA=1` writes it.
#[test]
fn the_kept_schema_is_the_generated_one() {
    let generated = ledger_schema();
    if std::env::var_os("MAG_BY_WRITE_SCHEMA").is_some() {
        std::fs::write(kept(), &generated).unwrap();
    }
    let on_disk = std::fs::read_to_string(kept()).unwrap_or_default();
    assert!(
        on_disk == generated,
        "schemas/mag-ledger.schema.json differs from the types; run the tests with \
         MAG_BY_WRITE_SCHEMA=1 to write it"
    );
}

/// A change made to a line's JSON.
type Edit = fn(&mut Value);

fn validator() -> jsonschema::Validator {
    let schema: Value = serde_json::from_str(&ledger_schema()).unwrap();
    jsonschema::draft202012::new(&schema).expect("the schema is a valid draft 2020-12 schema")
}

#[test]
fn every_line_the_reader_takes_is_valid() {
    let v = validator();
    for line in sample().to_lines().lines() {
        let value: Value = serde_json::from_str(line).unwrap();
        assert!(v.is_valid(&value), "{line}");
    }
}

/// Each change the reader refuses for its form, the schema refuses too. (A repeated key, which
/// JSON allows, a schema cannot see.)
#[test]
fn lines_the_reader_refuses_for_their_form_are_invalid() {
    let v = validator();
    let changes: [(&str, usize, Edit); 8] = [
        ("another format", 1, |v| v["format"] = "mag-graph".into()),
        ("a newer version", 1, |v| v["version"] = 2.into()),
        ("an unknown field in the header", 1, |v| {
            v["owner"] = "x".into()
        }),
        ("an unknown field in an entry", 2, |v| {
            v["note"] = "x".into()
        }),
        ("a null choice", 2, |v| v["record"]["chosen"] = Value::Null),
        ("a null score", 2, |v| {
            v["record"]["candidates"][1]["score"] = Value::Null
        }),
        ("an unknown change", 3, |v| {
            v["change"] = serde_json::json!({ "squeeze": {} })
        }),
        ("an entry without its record", 4, |v| {
            v.as_object_mut().unwrap().remove("record");
        }),
    ];
    for (what, line, change) in changes {
        let text = changed(line, change);
        assert!(
            Ledger::from_lines(&text).is_err(),
            "the reader takes {what}"
        );
        let value: Value = serde_json::from_str(text.lines().nth(line - 1).unwrap()).unwrap();
        assert!(!v.is_valid(&value), "the schema takes {what}");
    }
}
