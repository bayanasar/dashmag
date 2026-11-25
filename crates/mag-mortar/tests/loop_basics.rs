//! The loop's steps on the smallest pools.

mod common;

use common::*;
use mag_by::{Backend, Event, Mark, Place};
use mag_mortar::Pool;

fn one_device() -> Pool {
    Pool::new(
        vec![member("member-a", Owner)],
        vec![offer(
            device("member-a", 0),
            Backend::Cuda,
            16,
            &[("u", 0.01)],
        )],
    )
}

#[test]
fn a_job_is_placed_on_the_only_device() {
    let mut pool = one_device();
    let g = job(
        "j",
        1,
        vec![],
        vec![part("p", "u", Mark::Atomic, 1)],
        vec![],
    );
    let r = pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert_eq!(r.event, Event::Arrival { job: "j".into() });
    // Waiting, then the one device.
    assert_eq!(r.candidates.len(), 2);
    assert_eq!(r.chosen, Some(1));
    assert_eq!(
        pool.arrangement("j").unwrap().places,
        [Place::On(device("member-a", 0))]
    );
}

#[test]
fn a_job_too_large_for_the_pool_waits_with_its_reason() {
    let mut pool = one_device();
    let g = job(
        "j",
        1,
        vec![],
        vec![part("p", "u", Mark::Atomic, 32)],
        vec![],
    );
    let r = pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert_eq!(r.chosen, None);
    assert!(matches!(
        r.candidates[1].removed,
        Some(mag_by::Removal::Memory { .. })
    ));
    assert_eq!(pool.arrangement("j").unwrap().places, [Place::Waiting]);
}

#[test]
fn a_graph_that_fails_its_check_is_refused_and_the_pool_is_unchanged() {
    let mut pool = one_device();
    let g = job(
        "j",
        0,
        vec![],
        vec![part("p", "u", Mark::Atomic, 1)],
        vec![],
    );
    assert!(pool.submit(g, Stamp::default(), 0.0).is_err());
    assert!(pool.records().len() == 0 && pool.arrangement("j").is_none());
}

#[test]
fn the_same_events_give_the_same_records() {
    let run = || {
        let mut pool = Pool::new(
            vec![member("a", Owner), member("b", Owner)],
            vec![
                offer(
                    device("a", 0),
                    Backend::Cuda,
                    16,
                    &[("u", 0.01), ("v", 0.02)],
                ),
                offer(
                    device("b", 0),
                    Backend::Rocm,
                    32,
                    &[("u", 0.01), ("v", 0.02)],
                ),
            ],
        );
        let g = job(
            "j",
            1,
            vec![],
            vec![
                part("p", "u", Mark::Atomic, 1),
                part("q", "v", Mark::Atomic, 1),
            ],
            vec![edge("p", Some("q"))],
        );
        pool.submit(g, Stamp::default(), 0.0).unwrap();
        pool.records().cloned().collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}
