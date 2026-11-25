//! The host's words: the stamp, taking back, what the pool asks, and the start file, read,
//! refused, written back and held to its kept schema.

use std::path::Path;

use mag_by::{
    Asks, Backend, DeviceId, DeviceKind, Stamp, Start, StartFileError, TakeBack, Trust,
    start_schema,
};
use serde_json::{Value, json};

const EXAMPLE: &str = include_str!("../starts/example.json");

fn device(member: &str, index: u32) -> DeviceId {
    DeviceId {
        member: member.into(),
        index,
    }
}

/// Each word as JSON, and back.
fn round_trip<T>(value: &T, expected: Value)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    assert_eq!(serde_json::to_value(value).unwrap(), expected);
    assert_eq!(&serde_json::from_value::<T>(expected).unwrap(), value);
}

#[test]
fn a_stamp_bounds_nothing_unless_it_says_so() {
    round_trip(&Stamp::default(), json!({}));
    round_trip(
        &Stamp {
            by: Some("someone".into()),
            devices: Some(vec![device("a", 0)]),
            priority_ceiling: Some(3),
        },
        json!({
            "by": "someone",
            "devices": [{ "member": "a", "index": 0 }],
            "priority_ceiling": 3
        }),
    );
    for bad in [
        json!({ "by": null }),
        json!({ "ceiling": 3 }),
        json!({ "priority_ceiling": -1 }),
    ] {
        assert!(serde_json::from_value::<Stamp>(bad).is_err());
    }
}

#[test]
fn taking_back_by_name_and_by_need_each_with_a_deadline() {
    let device_back = TakeBack::Device {
        device: device("a", 1),
        deadline_seconds: 30.0,
    };
    round_trip(
        &device_back,
        json!({ "device": { "device": { "member": "a", "index": 1 }, "deadline_seconds": 30.0 } }),
    );
    let member_back = TakeBack::Member {
        member: "a".into(),
        deadline_seconds: 60.0,
    };
    round_trip(
        &member_back,
        json!({ "member": { "member": "a", "deadline_seconds": 60.0 } }),
    );
    let need = TakeBack::Need {
        kind: DeviceKind {
            backend: Some(Backend::Cuda),
            memory_bytes: Some(24 << 30),
        },
        deadline_seconds: 120.0,
    };
    round_trip(
        &need,
        json!({ "need": {
            "kind": { "backend": "cuda", "memory_bytes": 24u64 << 30 },
            "deadline_seconds": 120.0
        } }),
    );
    let any = TakeBack::Need {
        kind: DeviceKind::default(),
        deadline_seconds: 5.0,
    };
    round_trip(
        &any,
        json!({ "need": { "kind": {}, "deadline_seconds": 5.0 } }),
    );
    assert_eq!(
        [&device_back, &member_back, &need, &any].map(TakeBack::deadline_seconds),
        [30.0, 60.0, 120.0, 5.0]
    );
}

#[test]
fn the_pool_asks_for_a_kind_and_offers_members_back() {
    round_trip(&Asks::default(), json!({ "gives_back": [] }));
    round_trip(
        &Asks {
            wants: Some(DeviceKind {
                backend: Some(Backend::Rocm),
                memory_bytes: None,
            }),
            gives_back: vec!["laptop".into()],
        },
        json!({ "wants": { "backend": "rocm" }, "gives_back": ["laptop"] }),
    );
}

fn changed(change: impl FnOnce(&mut Value)) -> String {
    let mut v: Value = serde_json::from_str(EXAMPLE).unwrap();
    change(&mut v);
    serde_json::to_string(&v).unwrap()
}

#[test]
fn the_example_start_file_reads_and_writes_back_byte_for_byte() {
    let s = Start::from_json(EXAMPLE).unwrap();
    assert_eq!(s.member, "server");
    assert_eq!(s.coordinator, "workstation");
    assert_eq!(s.pool.members.len(), 4);
    assert_eq!(s.pool.members[3].trust, Trust::Outside);
    // The laptop is behind address translation: it has no address, and dials out.
    assert_eq!(s.peers[1].address, None);
    // The pool's authority, held once; this member's certificate and key, by their paths.
    assert!(s.authority.starts_with("-----BEGIN CERTIFICATE-----\n"));
    assert!(s.authority.ends_with("-----END CERTIFICATE-----\n"));
    assert_eq!(s.certificate_file, "/etc/mag/server.pem");
    assert_eq!(s.key_file, "/etc/mag/server.key");
    assert_eq!(s.to_json(), EXAMPLE);
}

#[test]
fn a_start_file_is_refused_for_its_form_as_every_file_is() {
    let other = changed(|v| v["format"] = "mag-pool".into());
    assert!(matches!(
        Start::from_json(&other),
        Err(StartFileError::Format { found: Some(f) }) if f == "\"mag-pool\""
    ));
    let newer = changed(|v| v["version"] = 2.into());
    let err = Start::from_json(&newer).unwrap_err();
    assert_eq!(
        err.to_string(),
        "the version is 2; this reader reads version 1"
    );
    for bad in [
        changed(|v| v["private_key"] = "secret".into()),
        changed(|v| v["peers"][0]["address"] = Value::Null),
        changed(|v| {
            v.as_object_mut().unwrap().remove("key_file");
        }),
        changed(|v| {
            v.as_object_mut().unwrap().remove("certificate_file");
        }),
        changed(|v| {
            v.as_object_mut().unwrap().remove("authority");
        }),
        // A peer is known by its certificate, not by a key listed for it.
        changed(|v| v["peers"][0]["public_key"] = "ed25519:AAAA".into()),
        changed(|v| v["pool"]["members"][0]["remote"] = false.into()),
    ] {
        let err = Start::from_json(&bad).unwrap_err();
        assert!(matches!(err, StartFileError::Json(_)), "{err}");
    }
    let repeated = EXAMPLE.replacen(
        "\"door\": \"/run/mag/door.sock\"",
        "\"door\": \"/run/mag/door.sock\", \"door\": \"/tmp/door\"",
        1,
    );
    assert!(matches!(
        Start::from_json(&repeated),
        Err(StartFileError::RepeatedKey { at }) if at == "door"
    ));
}

#[test]
fn a_start_file_that_does_not_hold_together_is_refused() {
    let cases: [(String, &str); 8] = [
        (
            changed(|v| v["member"] = "nobody".into()),
            "`member` names `nobody`, which is no member of the pool",
        ),
        (
            changed(|v| v["coordinator"] = "nobody".into()),
            "`coordinator` names `nobody`, which is no member of the pool",
        ),
        (
            changed(|v| v["peers"][2]["member"] = "nobody".into()),
            "`peers` names `nobody`, which is no member of the pool",
        ),
        (
            changed(|v| v["peers"][2]["member"] = "server".into()),
            "`server` is listed as a peer twice, or is this member itself",
        ),
        (
            changed(|v| v["peers"][2]["member"] = "laptop".into()),
            "`laptop` is listed as a peer twice, or is this member itself",
        ),
        (
            changed(|v| {
                v["peers"].as_array_mut().unwrap().remove(0);
            }),
            "the coordinator `workstation` is another member but not a peer, so it cannot be \
             reached",
        ),
        (
            changed(|v| v["idle_seconds"] = (-1.0).into()),
            "`idle_seconds` must be finite and not below 0",
        ),
        (
            changed(|v| {
                let first = v["pool"]["members"][0].clone();
                v["pool"]["members"].as_array_mut().unwrap().push(first);
            }),
            "the pool: member `workstation` is listed twice",
        ),
    ];
    for (text, reason) in cases {
        let err = Start::from_json(&text).unwrap_err();
        assert_eq!(err.to_string(), reason);
    }
    // The coordinator may be this member, with no peer for it; and an idle time of 0 is taken.
    let own = changed(|v| {
        v["coordinator"] = "server".into();
        v["peers"].as_array_mut().unwrap().remove(0);
        v["idle_seconds"] = 0.0.into();
    });
    assert!(Start::from_json(&own).is_ok());
}

fn kept() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("schemas/mag-start.schema.json")
}

/// Fails when the kept schema is not what the types generate. `MAG_BY_WRITE_SCHEMA=1` writes it.
#[test]
fn the_kept_schema_is_the_generated_one() {
    let generated = start_schema();
    if std::env::var_os("MAG_BY_WRITE_SCHEMA").is_some() {
        std::fs::write(kept(), &generated).unwrap();
    }
    let on_disk = std::fs::read_to_string(kept()).unwrap_or_default();
    assert!(
        on_disk == generated,
        "schemas/mag-start.schema.json differs from the types; run the tests with \
         MAG_BY_WRITE_SCHEMA=1 to write it"
    );
}

/// A change made to the file's JSON.
type Edit = fn(&mut Value);

#[test]
fn the_schema_takes_the_example_and_refuses_what_the_reader_refuses_for_its_form() {
    let schema: Value = serde_json::from_str(&start_schema()).unwrap();
    let v = jsonschema::draft202012::new(&schema).expect("a valid draft 2020-12 schema");
    assert!(v.is_valid(&serde_json::from_str(EXAMPLE).unwrap()));
    let changes: [(&str, Edit); 8] = [
        ("another format", |v| v["format"] = "mag-pool".into()),
        ("a newer version", |v| v["version"] = 2.into()),
        ("a private key", |v| v["private_key"] = "secret".into()),
        ("a null address", |v| v["peers"][0]["address"] = Value::Null),
        ("a peer with a key", |v| {
            v["peers"][0]["public_key"] = "ed25519:AAAA".into()
        }),
        ("no authority", |v| {
            v.as_object_mut().unwrap().remove("authority");
        }),
        ("no certificate file", |v| {
            v.as_object_mut().unwrap().remove("certificate_file");
        }),
        ("an unknown level", |v| {
            v["pool"]["members"][0]["trust"] = "anyone".into()
        }),
    ];
    for (what, change) in changes {
        let text = changed(change);
        assert!(Start::from_json(&text).is_err(), "the reader takes {what}");
        assert!(
            !v.is_valid(&serde_json::from_str(&text).unwrap()),
            "the schema takes {what}"
        );
    }
}
