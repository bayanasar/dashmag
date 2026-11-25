//! The pool's words: each built once, as `mag-mar` and `mag-mortar` will build them.

use mag_by::{
    Arrangement, Backend, Candidate, Cost, DeviceId, Dtype, Event, Member, Move, Offer, Place,
    Record, Removal, Tolerance, Trust, Unit,
};

fn device(member: &str, index: u32) -> DeviceId {
    DeviceId {
        member: member.into(),
        index,
    }
}

#[test]
fn a_member_and_its_offer() {
    let m = Member {
        name: "member-a".into(),
        trust: Trust::Site,
        give_back_seconds: 5.0,
    };
    let offer = Offer {
        device: device(&m.name, 0),
        backend: Backend::Cuda,
        memory_bytes: 16 << 30,
        tolerances: vec![Tolerance::F32, Tolerance::Bf16],
        dtypes: vec![Dtype::F32, Dtype::Bf16, Dtype::I32],
        costs: vec![Cost {
            unit: Unit {
                name: "block".into(),
                version: 1,
            },
            seconds_per_step: 0.004,
            band: 0.05,
            estimate: false,
        }],
    };
    assert_eq!(offer.device.member, "member-a");
    assert!(device("a", 0) < device("a", 1) && device("a", 9) < device("b", 0));
}

#[test]
fn a_record_of_one_run() {
    let a = Arrangement {
        job: "job".into(),
        places: vec![
            Place::On(device("a", 0)),
            Place::Copies(vec![device("a", 0), device("b", 0)]),
            Place::Waiting,
        ],
    };
    let kept = Candidate {
        arrangements: vec![a.clone()],
        moves: vec![Move::Fit],
        removed: None,
        left_out: vec![(
            "r".into(),
            Removal::Guard {
                part: "r".into(),
                device: device("rented", 0),
                keep: Trust::Site,
                trust: Trust::Outside,
            },
        )],
        score: Some(1.0),
        band: Some(0.05),
    };
    let removed = Candidate {
        arrangements: vec![a],
        moves: vec![Move::Fit, Move::Spread],
        removed: Some(Removal::Guard {
            part: "decode".into(),
            device: device("rented", 0),
            keep: Trust::Owner,
            trust: Trust::Site,
        }),
        left_out: vec![],
        score: None,
        band: None,
    };
    let r = Record {
        event: Event::Arrival { job: "job".into() },
        candidates: vec![kept, removed],
        chosen: Some(0),
        greedy: false,
        lowered: vec!["member-b".into()],
    };
    assert_eq!(r.candidates.len(), 2);
    for e in [
        Event::End { job: "j".into() },
        Event::Join {
            device: device("a", 1),
        },
        Event::Leave {
            device: device("a", 1),
        },
        Event::OfferChanged {
            device: device("a", 1),
        },
        Event::Drift {
            job: "j".into(),
            part: "p".into(),
        },
        Event::DeadlineAtRisk { job: "j".into() },
    ] {
        assert_ne!(e, r.event);
    }
    for m in [Move::Share, Move::Squeeze, Move::Wait] {
        assert_ne!(m, Move::Fit);
    }
    // Each reason a candidate can be removed for.
    let removals = [
        Removal::NoDevice {
            device: device("gone", 0),
        },
        Removal::Memory {
            device: device("a", 0),
        },
        Removal::Tolerance {
            part: "p".into(),
            device: device("a", 0),
        },
        Removal::Dtype {
            part: "p".into(),
            device: device("a", 0),
        },
        Removal::Pin {
            part: "p".into(),
            device: device("a", 0),
        },
        Removal::Stamp {
            job: "j".into(),
            device: device("a", 0),
        },
        Removal::Need { job: "j".into() },
    ];
    assert_eq!(removals.len(), 7);
}
