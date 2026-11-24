//! The pool file: a pool's description, read, refused as the graph file is, written back, and
//! held to its kept schema.

use std::path::Path;

use mag_by::{Description, DeviceId, PoolFileError, Removal, pool_schema};
use serde_json::{Value, json};

const EXAMPLE: &str = include_str!("../pools/example.json");

fn changed(change: impl FnOnce(&mut Value)) -> String {
    let mut v: Value = serde_json::from_str(EXAMPLE).unwrap();
    change(&mut v);
    serde_json::to_string(&v).unwrap()
}

#[test]
fn the_example_reads_and_writes_back_byte_for_byte() {
    let d = Description::from_json(EXAMPLE).unwrap();
    assert_eq!(d.members.len(), 4);
    assert_eq!(d.offers.len(), 4);
    assert!(d.jobs.is_empty());
    // One member is remote: the rented card, the fastest and the largest.
    let remote: Vec<_> = d.members.iter().filter(|m| m.remote).collect();
    assert_eq!(remote.len(), 1);
    assert_eq!(d.to_json(), EXAMPLE);
}

#[test]
fn a_pool_file_is_refused_as_a_graph_file_is() {
    let other = changed(|v| v["format"] = "mag-graph".into());
    assert!(matches!(
        Description::from_json(&other),
        Err(PoolFileError::Format { found: Some(f) }) if f == "\"mag-graph\""
    ));
    let newer = changed(|v| {
        v["version"] = 2.into();
        v["links"] = json!([]);
    });
    assert!(matches!(
        Description::from_json(&newer),
        Err(PoolFileError::Version { found: Some(f) }) if f == "2"
    ));
    for bad in [
        changed(|v| v["owner"] = "x".into()),
        changed(|v| v["offers"][0]["speed"] = 1.into()),
        changed(|v| v["members"][3]["remote"] = Value::Null),
        changed(|v| {
            v["members"][3].as_object_mut().unwrap().remove("remote");
        }),
    ] {
        assert!(matches!(
            Description::from_json(&bad),
            Err(PoolFileError::Json(_))
        ));
    }
    let repeated = EXAMPLE.replacen("\"remote\": true", "\"remote\": true, \"remote\": false", 1);
    let err = Description::from_json(&repeated).unwrap_err();
    assert!(
        matches!(&err, PoolFileError::RepeatedKey { at } if at == "members[3].remote"),
        "{err}"
    );
}

#[test]
fn a_reason_reads_as_words() {
    let device = DeviceId {
        member: "rented".into(),
        index: 0,
    };
    assert_eq!(device.to_string(), "rented/0");
    let guard = Removal::Guard {
        part: "decode".into(),
        device,
    };
    assert_eq!(
        guard.to_string(),
        "the guard: decode carries local-only data, and rented/0's member is remote"
    );
}

/// A change made to the file's JSON.
type Edit = fn(&mut Value);

fn kept() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("schemas/mag-pool.schema.json")
}

/// Fails when the kept schema is not what the types generate. `MAG_BY_WRITE_SCHEMA=1` writes it.
#[test]
fn the_kept_schema_is_the_generated_one() {
    let generated = pool_schema();
    if std::env::var_os("MAG_BY_WRITE_SCHEMA").is_some() {
        std::fs::write(kept(), &generated).unwrap();
    }
    let on_disk = std::fs::read_to_string(kept()).unwrap_or_default();
    assert!(
        on_disk == generated,
        "schemas/mag-pool.schema.json differs from the types; run the tests with \
         MAG_BY_WRITE_SCHEMA=1 to write it"
    );
}

#[test]
fn the_schema_takes_the_example_and_refuses_what_the_reader_refuses_for_its_form() {
    let schema: Value = serde_json::from_str(&pool_schema()).unwrap();
    let v = jsonschema::draft202012::new(&schema).expect("a valid draft 2020-12 schema");
    assert!(v.is_valid(&serde_json::from_str(EXAMPLE).unwrap()));
    let changes: [(&str, Edit); 5] = [
        ("another format", |v| v["format"] = "mag-graph".into()),
        ("a newer version", |v| v["version"] = 2.into()),
        ("an unknown field", |v| v["owner"] = "x".into()),
        ("a null mark", |v| v["members"][3]["remote"] = Value::Null),
        ("an unknown backend", |v| {
            v["offers"][0]["backend"] = "tpu".into()
        }),
    ];
    for (what, change) in changes {
        let text = changed(change);
        assert!(
            Description::from_json(&text).is_err(),
            "the reader takes {what}"
        );
        assert!(
            !v.is_valid(&serde_json::from_str(&text).unwrap()),
            "the schema takes {what}"
        );
    }
}

#[test]
fn a_member_listed_twice_or_a_device_offered_twice_is_refused() {
    let member = changed(|v| {
        let first = v["members"][0].clone();
        v["members"].as_array_mut().unwrap().push(first);
    });
    let err = Description::from_json(&member).unwrap_err();
    assert!(
        matches!(&err, PoolFileError::RepeatedMember { name } if name == "workstation"),
        "{err}"
    );
    let device = changed(|v| {
        let mut again = v["offers"][3].clone();
        again["memory_bytes"] = 1.into();
        v["offers"].as_array_mut().unwrap().push(again);
    });
    let err = Description::from_json(&device).unwrap_err();
    assert_eq!(err.to_string(), "device workstation/0 is offered twice");
}
