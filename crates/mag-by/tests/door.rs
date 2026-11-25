//! The words of the door, written as JSON and read back.

use mag_by::{
    Arrangement, Asks, Description, DeviceId, DoorAnswer, DoorRefusal, DoorRequest, Event, Graph,
    Place, Record, Stamp,
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
        json!("stop"),
    ] {
        assert!(
            serde_json::from_value::<DoorRequest>(bad.clone()).is_err(),
            "{bad}"
        );
    }
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
    // The pool's asks have their place, and are left out until the pool can say them.
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

    for bad in [
        json!({ "status": { "pool": serde_json::to_value(&pool).unwrap(), "asks": null } }),
        json!({ "ended": {} }),
        json!({ "refused": { "refusal": { "busy": {} } } }),
        json!({ "submitted": { "record": serde_json::to_value(record("tally")).unwrap() } }),
    ] {
        assert!(
            serde_json::from_value::<DoorAnswer>(bad.clone()).is_err(),
            "{bad}"
        );
    }
}
