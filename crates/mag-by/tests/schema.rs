//! The graph file's JSON Schema: kept in the repository as generated, and checked by a validator
//! against files the reader takes and files it refuses.

use std::path::Path;

use mag_by::{Graph, graph_schema};
use serde_json::{Value, json};

const EXAMPLE: &str = include_str!("../graphs/example.json");

/// A change made to a file's JSON.
type Change = fn(&mut Value);

fn kept() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("schemas/mag-graph.schema.json")
}

/// Fails when the kept schema is not what the types generate. `MAG_BY_WRITE_SCHEMA=1` writes it.
#[test]
fn the_kept_schema_is_the_generated_one() {
    let generated = graph_schema();
    if std::env::var_os("MAG_BY_WRITE_SCHEMA").is_some() {
        std::fs::write(kept(), &generated).unwrap();
    }
    let on_disk = std::fs::read_to_string(kept()).unwrap_or_default();
    assert!(
        on_disk == generated,
        "schemas/mag-graph.schema.json differs from the types; run the tests with \
         MAG_BY_WRITE_SCHEMA=1 to write it"
    );
}

fn validator() -> jsonschema::Validator {
    let schema: Value = serde_json::from_str(&graph_schema()).unwrap();
    jsonschema::draft202012::new(&schema).expect("the schema is a valid draft 2020-12 schema")
}

fn example() -> Value {
    serde_json::from_str(EXAMPLE).unwrap()
}

#[test]
fn files_the_reader_takes_are_valid() {
    let v = validator();
    assert!(v.is_valid(&example()));
    // Fields with defaults left out.
    let minimal = json!({
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
    });
    assert!(Graph::from_json(&minimal.to_string()).is_ok());
    assert!(v.is_valid(&minimal));
}

/// Each change the reader refuses for its form, the schema refuses too. (What only `check`
/// refuses, such as a duplicate name, a schema cannot say.)
#[test]
fn files_the_reader_refuses_for_their_form_are_invalid() {
    let v = validator();
    let changes: [(&str, Change); 9] = [
        ("another format", |f| f["format"] = "other".into()),
        ("a newer version", |f| f["version"] = 2.into()),
        ("no format", |f| {
            f.as_object_mut().unwrap().remove("format");
        }),
        ("an unknown field", |f| f["objective"] = "makespan".into()),
        ("an unknown field in a part", |f| {
            f["parts"][0]["memory"] = 1.into()
        }),
        ("a null note", |f| f["parts"][0]["note"] = Value::Null),
        ("a null pin", |f| f["parts"][1]["pin"] = Value::Null),
        ("an edge without a dtype", |f| {
            f["edges"][0].as_object_mut().unwrap().remove("dtype");
        }),
        ("an unknown mark", |f| {
            f["parts"][0]["mark"] = "replicated".into()
        }),
    ];
    for (what, change) in changes {
        let mut f = example();
        change(&mut f);
        assert!(
            Graph::from_json(&f.to_string()).is_err(),
            "the reader takes {what}"
        );
        assert!(!v.is_valid(&f), "the schema takes {what}");
    }
}
