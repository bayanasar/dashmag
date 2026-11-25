//! The words of the door, written as JSON and read back.

use mag_by::{
    Arrangement, Asks, Backend, Description, DeviceId, DeviceKind, DoorAnswer, DoorRefusal,
    DoorRequest, Event, Graph, Place, Record, Stamp, TakeBack,
};
use serde_json::{Value, json};

fn round_trip<T>(value: &T, expected: Value)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    assert_eq!(serde_json::to_value(value).unwrap(), expected);
    assert_eq!(&serde_json::from_value::<T>(expected).unwrap(), value);
}

fn example() -> Graph {
    Graph::from_json(include_str!("../graphs/example.json")).unwrap()
}

fn id(member: &str, index: u32) -> DeviceId {
    DeviceId {
        member: member.into(),
        index,
    }
}

fn record(job: &str) -> Record {
    Record {
        event: Event::Arrival { job: job.into() },
        candidates: vec![],
        chosen: None,
        greedy: false,
        lowered: vec![],
    }
}

#[test]
fn a_request_submits_a_graph_with_its_stamp_ends_a_job_or_asks_the_status() {
    let graph = example();
    let stamp = Stamp {
        by: Some("ops".into()),
        devices: None,
        priority_ceiling: Some(2),
    };
    let submit = DoorRequest::Submit {
        graph: graph.clone(),
        stamp: stamp.clone(),
    };
    round_trip(
        &submit,
        json!({ "submit": {
            "graph": serde_json::to_value(&graph).unwrap(),
            "stamp": { "by": "ops", "priority_ceiling": 2 }
        } }),
    );
    round_trip(
        &DoorRequest::End {
            job: "label-frames".into(),
        },
        json!({ "end": { "job": "label-frames" } }),
    );
    round_trip(&DoorRequest::Status, json!("status"));
    for bad in [
        json!({ "submit": { "graph": serde_json::to_value(&graph).unwrap() } }),
        json!({ "end": { "job": "label-frames", "by": "ops" } }),
        json!({ "end": {} }),
        json!({ "status": {} }),
        json!({ "take_back": { "need": {}, "deadline_seconds": 1.0 } }),
        json!({ "take_back": { "need": { "kind": {} } } }),
        json!({ "take_back": { "need": { "kind": { "cores": 8 }, "deadline_seconds": 1.0 } } }),
        json!({ "take_back": null }),
        json!("take_back"),
        json!("stop"),
    ] {
        assert!(
            serde_json::from_value::<DoorRequest>(bad.clone()).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn a_request_takes_a_device_back_by_need_and_may_name_one_which_the_door_refuses() {
    let need = DoorRequest::TakeBack(TakeBack::Need {
        kind: DeviceKind {
            backend: Some(Backend::Cuda),
            memory_bytes: Some(1 << 30),
            ..DeviceKind::default()
        },
        deadline_seconds: 30.0,
    });
    round_trip(
        &need,
        json!({ "take_back": { "need": {
            "kind": { "backend": "cuda", "memory_bytes": 1073741824u64 },
            "deadline_seconds": 30.0
        } } }),
    );
    round_trip(
        &DoorRequest::TakeBack(TakeBack::Need {
            kind: DeviceKind::default(),
            deadline_seconds: 0.0,
        }),
        json!({ "take_back": { "need": { "kind": {}, "deadline_seconds": 0.0 } } }),
    );
    // By name it is a word too, which the coordinator refuses.
    round_trip(
        &DoorRequest::TakeBack(TakeBack::Member {
            member: "laptop".into(),
            deadline_seconds: 5.0,
        }),
        json!({ "take_back": { "member": { "member": "laptop", "deadline_seconds": 5.0 } } }),
    );
}

#[test]
fn an_answer_carries_the_record_and_where_the_job_is_or_the_pool_or_a_typed_refusal() {
    let device = DeviceId {
        member: "server".into(),
        index: 0,
    };
    let placed = DoorAnswer::Submitted {
        record: record("tally"),
        arrangement: Arrangement {
            job: "tally".into(),
            places: vec![Place::On(device.clone())],
        },
    };
    round_trip(
        &placed,
        json!({ "submitted": {
            "record": serde_json::to_value(record("tally")).unwrap(),
            "arrangement": { "job": "tally", "places": [{ "on": { "member": "server",
                                                                 "index": 0 } }] }
        } }),
    );
    assert_eq!(placed.placed(), Some(true));
    let waits = DoorAnswer::Submitted {
        record: record("tally"),
        arrangement: Arrangement {
            job: "tally".into(),
            places: vec![Place::On(device), Place::Waiting],
        },
    };
    assert_eq!(waits.placed(), Some(false));

    let ended = DoorAnswer::Ended {
        record: record("tally"),
    };
    round_trip(
        &ended,
        json!({ "ended": { "record": serde_json::to_value(record("tally")).unwrap() } }),
    );
    assert_eq!(ended.placed(), None);

    let pool = Description {
        at: 12.5,
        members: vec![],
        offers: vec![],
        jobs: vec![],
        formats: vec![],
    };
    // The pool's asks have their place, and are left out by a coordinator that does not say them.
    let status = DoorAnswer::Status {
        pool: pool.clone(),
        asks: None,
    };
    round_trip(
        &status,
        json!({ "status": { "pool": serde_json::to_value(&pool).unwrap() } }),
    );
    round_trip(
        &DoorAnswer::Status {
            pool: pool.clone(),
            asks: Some(Asks {
                wants: None,
                gives_back: vec!["laptop".into()],
            }),
        },
        json!({ "status": {
            "pool": serde_json::to_value(&pool).unwrap(),
            "asks": { "gives_back": ["laptop"] }
        } }),
    );

    for (refusal, expected) in [
        (
            DoorRefusal::Unreadable {
                error: "not JSON".into(),
            },
            json!({ "unreadable": { "error": "not JSON" } }),
        ),
        (
            DoorRefusal::Pool {
                reason: "the pool already holds a job named `tally`".into(),
            },
            json!({ "pool": { "reason": "the pool already holds a job named `tally`" } }),
        ),
        (
            DoorRefusal::NoJob {
                job: "tally".into(),
            },
            json!({ "no_job": { "job": "tally" } }),
        ),
        (
            DoorRefusal::Unmet { slow: vec![] },
            json!({ "unmet": { "slow": [] } }),
        ),
        (
            DoorRefusal::Unmet {
                slow: vec![id("laptop", 0)],
            },
            json!({ "unmet": { "slow": [{ "member": "laptop", "index": 0 }] } }),
        ),
        (DoorRefusal::ByName, json!("by_name")),
    ] {
        round_trip(
            &DoorAnswer::Refused {
                refusal: refusal.clone(),
            },
            json!({ "refused": { "refusal": expected } }),
        );
    }
    assert_eq!(
        DoorRefusal::NoJob {
            job: "tally".into()
        }
        .to_string(),
        "the pool holds no job named `tally`"
    );
    assert_eq!(
        DoorRefusal::Unmet { slow: vec![] }.to_string(),
        "no device in the pool is of the kind asked for"
    );
    assert_eq!(
        DoorRefusal::Unmet {
            slow: vec![id("laptop", 0), id("server", 1)]
        }
        .to_string(),
        "the devices of the kind asked for (laptop/0, server/1) are each of a member that takes \
         longer than the deadline to give a device back"
    );
    assert!(DoorRefusal::ByName.to_string().contains("SIGTERM"));

    // A device taken back: the one chosen, the record of its leave, and whether its member said
    // it gave it back in time.
    let leave = Record {
        event: Event::Leave {
            device: id("laptop", 0),
        },
        ..record("tally")
    };
    for given_back in [true, false] {
        round_trip(
            &DoorAnswer::TakenBack {
                device: id("laptop", 0),
                record: leave.clone(),
                given_back,
            },
            json!({ "taken_back": {
                "device": { "member": "laptop", "index": 0 },
                "record": serde_json::to_value(&leave).unwrap(),
                "given_back": given_back
            } }),
        );
    }

    for bad in [
        json!({ "status": { "pool": serde_json::to_value(&pool).unwrap(), "asks": null } }),
        json!({ "ended": {} }),
        json!({ "refused": { "refusal": { "busy": {} } } }),
        json!({ "submitted": { "record": serde_json::to_value(record("tally")).unwrap() } }),
        json!({ "refused": { "refusal": { "unmet": {} } } }),
        json!({ "refused": { "refusal": { "by_name": {} } } }),
        json!({ "taken_back": {
            "device": { "member": "laptop", "index": 0 },
            "record": serde_json::to_value(record("tally")).unwrap()
        } }),
    ] {
        assert!(
            serde_json::from_value::<DoorAnswer>(bad.clone()).is_err(),
            "{bad}"
        );
    }
}
