//! Each constraint the loop removes a candidate for, and the estimate of a cost never measured:
//! each test names what it holds.

mod common;

use common::*;
use mag_by::{Backend, Dtype, Mark, Need, Pin, Place, Removal, Tolerance};
use mag_mortar::Pool;

fn two_cards() -> Pool {
    Pool::new(
        vec![member("a", false), member("b", false)],
        vec![
            offer(
                device("a", 0),
                Backend::Cuda,
                16,
                &[("u", 0.01), ("v", 0.01)],
            ),
            offer(
                device("b", 0),
                Backend::Rocm,
                16,
                &[("u", 0.02), ("v", 0.02)],
            ),
        ],
    )
}

#[test]
fn a_tolerance_no_device_meets_keeps_the_job_waiting_with_its_reason() {
    let mut pool = two_cards();
    let mut p = part("p", "u", Mark::Atomic, 1);
    p.tolerance = Tolerance::Int8;
    let r = pool
        .submit(job("int8", 1, vec![], vec![p], vec![]), 0.0)
        .unwrap();
    assert_eq!(r.chosen, None);
    assert_eq!(
        r.candidates[1].removed,
        Some(Removal::Tolerance {
            part: "p".into(),
            device: device("a", 0)
        })
    );
}

#[test]
fn a_dtype_a_device_does_not_compute_in_removes_it() {
    let mut pool = two_cards();
    let mut e = edge("p", None);
    e.dtype = Dtype::F64;
    let r = pool
        .submit(
            job(
                "f64",
                1,
                vec![],
                vec![part("p", "u", Mark::Atomic, 1)],
                vec![e],
            ),
            0.0,
        )
        .unwrap();
    assert_eq!(r.chosen, None);
    assert_eq!(
        r.candidates[1].removed,
        Some(Removal::Dtype {
            part: "p".into(),
            device: device("a", 0)
        })
    );
}

#[test]
fn a_rate_the_pool_cannot_sustain_keeps_the_job_waiting() {
    let mut pool = two_cards();
    let mut g = job(
        "fast",
        1,
        vec![],
        vec![part("p", "u", Mark::Atomic, 1)],
        vec![],
    );
    // 0.01 s a step on the faster card is 100 steps a second.
    g.job.need = Need {
        rate_per_second: Some(150.0),
        deadline_seconds: None,
    };
    let r = pool.submit(g, 0.0).unwrap();
    assert_eq!(r.chosen, None);
    assert_eq!(
        r.candidates[1].removed,
        Some(Removal::Need { job: "fast".into() })
    );
}

#[test]
fn an_arrival_that_would_slow_a_running_job_below_its_need_goes_elsewhere() {
    let mut pool = two_cards();
    let mut service = job(
        "service",
        1,
        vec![],
        vec![part("p", "u", Mark::Atomic, 1)],
        vec![],
    );
    // 100 steps a second alone on the faster card; it needs 80.
    service.job.need = Need {
        rate_per_second: Some(80.0),
        deadline_seconds: None,
    };
    pool.submit(service, 0.0).unwrap();
    assert_eq!(
        pool.arrangement("service").unwrap().places,
        [Place::On(device("a", 0))]
    );
    let r = pool
        .submit(
            job(
                "batch",
                1,
                vec![],
                vec![part("q", "v", Mark::Atomic, 1)],
                vec![],
            ),
            1.0,
        )
        .unwrap();
    // Beside the service the card would take 0.02 s a step: 50, below 80.
    assert_eq!(
        r.candidates[1].removed,
        Some(Removal::Need {
            job: "service".into()
        })
    );
    assert_eq!(
        pool.arrangement("batch").unwrap().places,
        [Place::On(device("b", 0))]
    );
}

#[test]
fn a_pin_to_a_member_holds_however_slow_that_member_is() {
    let mut pool = two_cards();
    let g = job(
        "here",
        1,
        vec![],
        vec![pinned(
            part("p", "u", Mark::Atomic, 1),
            Pin::Member("b".into()),
        )],
        vec![],
    );
    let r = pool.submit(g, 0.0).unwrap();
    assert_eq!(
        r.candidates[1].removed,
        Some(Removal::Pin {
            part: "p".into(),
            device: device("a", 0)
        })
    );
    assert_eq!(
        pool.arrangement("here").unwrap().places,
        [Place::On(device("b", 0))]
    );
}

#[test]
fn a_chain_s_parts_stay_on_one_device_where_dividing_them_would_score_higher() {
    // Divided, each card would take 0.01 s a step; together one takes 0.02. The chain stays
    // together: dividing it across devices is a later step's.
    let mut pool = Pool::new(
        vec![member("a", false), member("b", false)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("block", 0.01)]),
            offer(device("b", 0), Backend::Cuda, 16, &[("block", 0.01)]),
        ],
    );
    let mut g = job(
        "model",
        1,
        vec![],
        vec![
            part("b0", "block", Mark::Atomic, 4),
            part("b1", "block", Mark::Atomic, 4),
        ],
        vec![edge("b0", Some("b1"))],
    );
    g.chains.push(mag_by::Chain {
        name: "blocks".into(),
        parts: vec!["b0".into(), "b1".into()],
    });
    let r = pool.submit(g, 0.0).unwrap();
    // Waiting, then the chain on a, then on b: no candidate divides it.
    assert_eq!(r.candidates.len(), 3);
    let places = &pool.arrangement("model").unwrap().places;
    assert_eq!(places[0], places[1]);
}

#[test]
fn a_cost_never_measured_is_estimated_as_the_slowest_measured_and_flagged() {
    // `c` measured the part kind at 0.05 and `a` at 0.02; `b` never ran it. Its estimate is the
    // slowest, 0.05, with the wide band: the job goes to `a`.
    let mut pool = Pool::new(
        vec![member("a", false), member("b", false), member("c", false)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.02)]),
            offer(device("b", 0), Backend::Cuda, 16, &[]),
            offer(device("c", 0), Backend::Cuda, 16, &[("u", 0.05)]),
        ],
    );
    let r = pool
        .submit(
            job(
                "j",
                1,
                vec![],
                vec![part("p", "u", Mark::Atomic, 1)],
                vec![],
            ),
            0.0,
        )
        .unwrap();
    assert_eq!(r.candidates[2].band, Some(mag_mortar::ESTIMATE_BAND));
    assert_eq!(r.candidates[1].band, Some(0.05));
    assert_eq!(
        pool.arrangement("j").unwrap().places,
        [Place::On(device("a", 0))]
    );
}
