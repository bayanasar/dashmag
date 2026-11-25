//! The pool's events beyond an arrival: a second job of one name, a member's trust, a device
//! leaving, and a job too large to list every placement of.

mod common;

use std::time::Instant;

use common::*;
use mag_by::{Backend, Event, Mark, Place, Removal};
use mag_mortar::{ChangeError, Pool};

#[test]
fn a_second_job_of_one_name_is_refused_and_the_first_is_kept() {
    let mut pool = Pool::new(
        vec![member("a", Owner)],
        vec![offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)])],
    );
    let g = || {
        job(
            "same",
            1,
            vec![],
            vec![part("p", "u", Mark::Atomic, 1)],
            vec![],
        )
    };
    pool.submit(g(), Stamp::default(), 0.0).unwrap();
    assert_eq!(
        pool.submit(g(), Stamp::default(), 1.0).unwrap_err(),
        ChangeError::Duplicate { job: "same".into() }
    );
    assert_eq!(pool.records().len(), 1);
    pool.end("same", 2.0).unwrap();
    assert!(pool.arrangement("same").is_none());
}

#[test]
fn a_later_report_of_a_lower_trust_stands_over_an_earlier_higher_one() {
    let mut pool = Pool::new(
        vec![member("m", Owner)],
        vec![offer(device("m", 0), Backend::Cuda, 16, &[("u", 0.05)])],
    );
    let r = pool
        .join(
            Some(member("m", Site)),
            offer(device("m", 1), Backend::Cuda, 16, &[("u", 0.01)]),
            0.0,
        )
        .unwrap();
    assert_eq!(r.lowered, ["m"]);
    // A higher report does not undo it.
    let r = pool
        .join(
            Some(member("m", Owner)),
            offer(device("m", 2), Backend::Cuda, 16, &[("u", 0.01)]),
            1.0,
        )
        .unwrap();
    assert!(r.lowered.is_empty());
    assert_eq!(pool.describe().members, [member("m", Site)]);
    // A lower one lowers it again.
    let r = pool
        .join(
            Some(member("m", Outside)),
            offer(device("m", 3), Backend::Cuda, 16, &[("u", 0.01)]),
            1.5,
        )
        .unwrap();
    assert_eq!(r.lowered, ["m"]);
    assert_eq!(pool.describe().members, [member("m", Outside)]);
    let g = job(
        "label",
        1,
        vec![input("frames", Owner)],
        vec![part("detect", "u", Mark::Atomic, 4)],
        vec![edge("frames", Some("detect"))],
    );
    let r = pool.submit(g, Stamp::default(), 2.0).unwrap();
    assert_eq!(r.chosen, None);
    assert!(
        r.candidates[1..]
            .iter()
            .all(|c| matches!(c.removed, Some(Removal::Guard { .. })))
    );
}

#[test]
fn a_device_that_leaves_takes_nothing_with_it() {
    let mut pool = Pool::new(
        vec![member("a", Owner), member("b", Owner)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Rocm, 16, &[("u", 0.02)]),
        ],
    );
    pool.submit(
        job(
            "j",
            1,
            vec![],
            vec![part("p", "u", Mark::Atomic, 4)],
            vec![],
        ),
        Stamp::default(),
        0.0,
    )
    .unwrap();
    assert_eq!(
        pool.arrangement("j").unwrap().places,
        [Place::On(device("a", 0))]
    );
    let r = pool.leave(&device("a", 0), 5.0).unwrap();
    assert_eq!(
        r.event,
        Event::Leave {
            device: device("a", 0)
        }
    );
    assert_eq!(
        pool.arrangement("j").unwrap().places,
        [Place::On(device("b", 0))]
    );
    // With the last device gone the job waits; it names no device that has left.
    pool.leave(&device("b", 0), 6.0).unwrap();
    assert_eq!(pool.arrangement("j").unwrap().places, [Place::Waiting]);
}

#[test]
fn past_the_bound_the_loop_places_greedily_and_quickly() {
    // Twelve atomic parts on four cards are 4^12 placements; listing them would take minutes.
    let mut pool = Pool::new(
        (0..4).map(|i| member(&format!("m{i}"), Owner)).collect(),
        (0..4)
            .map(|i| {
                offer(
                    device(&format!("m{i}"), 0),
                    Backend::Cuda,
                    16,
                    &[("u", 0.01 * (i + 1) as f64)],
                )
            })
            .collect(),
    );
    let parts = (0..12)
        .map(|i| part(&format!("p{i}"), "u", Mark::Atomic, 4))
        .collect();
    let start = Instant::now();
    let r = pool
        .submit(job("wide", 1, vec![], parts, vec![]), Stamp::default(), 0.0)
        .unwrap();
    assert!(r.greedy);
    assert!(start.elapsed().as_secs_f64() < 1.0, "{:?}", start.elapsed());
    // Every part placed, four to a card at most, as memory allows.
    let a = pool.arrangement("wide").unwrap();
    assert!(a.places.iter().all(|p| matches!(p, Place::On(_))));
}
