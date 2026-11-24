//! What an author asks a pool before submitting: its description, and the plan of a graph, which
//! leaves the pool as it was.

mod common;

use common::*;
use mag_by::{Backend, Format, Mark, Place, Refusal};
use mag_mortar::{Pool, SubmitError};

/// One device of 64 GiB on which every part takes a second a step.
fn one_device() -> Pool {
    Pool::new(
        vec![member("a", false)],
        vec![offer(device("a", 0), Backend::Cuda, 64, &[("u", 1.0)])],
    )
}

/// A job of one atomic part of `gib` GiB.
fn one_part(name: &str, priority: u32, gib: u64) -> mag_by::Graph {
    job(
        name,
        priority,
        vec![],
        vec![part("p", "u", Mark::Atomic, gib)],
        vec![],
    )
}

#[test]
fn a_plan_and_a_submit_of_one_graph_at_one_time_make_the_same_record() {
    let mut pool = Pool::new(
        vec![member("a", false), member("b", false)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Rocm, 16, &[("u", 0.02)]),
        ],
    );
    pool.submit(one_part("first", 1, 12), 0.0).unwrap();
    let g = job(
        "second",
        2,
        vec![],
        vec![
            part("p", "u", Mark::Atomic, 2),
            part("copies", "u", Mark::Replicable, 1),
        ],
        vec![edge("p", Some("copies"))],
    );
    let planned = pool.plan(&g, 7.0).unwrap();
    let submitted = pool.submit(g, 7.0).unwrap();
    assert_eq!(&planned, submitted);
    assert!(planned.chosen.is_some());
}

#[test]
fn a_plan_leaves_the_description_and_the_records_as_they_were() {
    // A second, small device, which only a small job fits.
    let mut pool = Pool::new(
        vec![member("a", false), member("b", false)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Cuda, 4, &[("u", 0.01)]),
        ],
    );
    pool.submit(one_part("running", 1, 12), 0.0).unwrap();
    pool.submit(one_part("waiting", 1, 12), 1.0).unwrap();
    let before = (pool.describe(), pool.records().to_vec());

    // A job that would be placed, one that would wait, and two that would be refused, each
    // planned later than the pool's time, so that a plan run on the pool would move it.
    let placed = pool.plan(&one_part("small", 1, 2), 30.0).unwrap();
    assert!(placed.chosen.is_some());
    let waits = pool.plan(&one_part("large", 1, 40), 30.0).unwrap();
    assert_eq!(waits.chosen, None);
    assert_eq!(
        pool.plan(&one_part("running", 1, 2), 30.0).unwrap_err(),
        SubmitError::Duplicate {
            job: "running".into()
        }
    );
    assert_eq!(
        pool.plan(&one_part("empty", 1, 0), 30.0).unwrap_err(),
        SubmitError::Graph(Refusal::ZeroMemory { part: "p".into() })
    );

    assert_eq!((pool.describe(), pool.records().to_vec()), before);
    assert_eq!(pool.describe().at, 1.0);
    assert!(pool.arrangement("small").is_none());
}

#[test]
fn a_plan_at_a_later_time_weighs_the_waiting_jobs_as_they_would_be_then() {
    let mut pool = one_device();
    // A job of priority 10 holds the device. Sharing it with a job of priority 1 would halve the
    // first job's rate for half of the second's: 5 + 0.5 is less than 10, so the second waits.
    pool.submit(one_part("first", 10, 1), 0.0).unwrap();
    let r = pool.submit(one_part("second", 1, 1), 0.0).unwrap();
    assert_eq!(r.chosen, None);

    // Planned now, a third job changes nothing. Planned 1000 s later, the second job has waited
    // 1000 s and weighs 1 + 1000/60: sharing now scores 5 + 8.8, and it is placed, not the third.
    let third = one_part("third", 1, 1);
    let now = pool.plan(&third, 0.0).unwrap();
    assert_eq!(now.chosen, None);
    let later = pool.plan(&third, 1000.0).unwrap();
    let chosen = &later.candidates[later.chosen.expect("the second job is placed")];
    let places: Vec<_> = chosen
        .arrangements
        .iter()
        .map(|a| (a.job.as_str(), a.places[0].clone()))
        .collect();
    assert_eq!(
        places,
        [
            ("second", Place::On(device("a", 0))),
            ("third", Place::Waiting)
        ]
    );

    // The pool's own time has not moved, and the second job has waited nothing yet.
    let d = pool.describe();
    assert_eq!(d.at, 0.0);
    assert_eq!(d.jobs[1].waited_seconds, 0.0);
    // Submitted at that time, the third job's arrival makes the record the plan showed.
    assert_eq!(pool.submit(third, 1000.0).unwrap(), &later);
}

#[test]
fn a_description_shows_where_each_job_is_how_long_each_waited_and_the_stricter_mark() {
    let mut pool = Pool::new(
        vec![member("a", false), member("b", false)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Rocm, 16, &[("u", 0.02)]),
        ],
    );
    pool.submit(one_part("one", 1, 12), 0.0).unwrap();
    pool.submit(one_part("two", 1, 12), 5.0).unwrap();
    let local = job(
        "local",
        1,
        vec![input("frames", true)],
        vec![part("p", "u", Mark::Atomic, 12)],
        vec![edge("frames", Some("p"))],
    );
    pool.submit(local, 10.0).unwrap();
    // `two` loses its device and waits.
    pool.leave(&device("b", 0), 20.0);
    // b comes back with another device, now reported as remote: `two` takes it, and `local`,
    // which reads local-only frames, may not.
    let r = pool.join(
        Some(member("b", true)),
        offer(device("b", 1), Backend::Rocm, 16, &[("u", 0.02)]),
        30.0,
    );
    assert_eq!(r.marked_remote, ["b"]);

    let d = pool.describe();
    assert_eq!(d.at, 30.0);
    assert_eq!(d.members, [member("a", false), member("b", true)]);
    let devices: Vec<_> = d.offers.iter().map(|o| o.device.clone()).collect();
    assert_eq!(devices, [device("a", 0), device("b", 1)]);
    let jobs: Vec<_> = d
        .jobs
        .iter()
        .map(|h| {
            (
                h.arrangement.job.as_str(),
                h.arrangement.places[0].clone(),
                h.waited_seconds,
            )
        })
        .collect();
    assert_eq!(
        jobs,
        [
            ("one", Place::On(device("a", 0)), 0.0),
            ("two", Place::On(device("b", 1)), 0.0),
            ("local", Place::Waiting, 20.0),
        ]
    );
    assert!(d.jobs.iter().all(|h| h.priority == 1));
    assert_eq!(
        d.formats,
        [Format {
            name: "mag-graph".into(),
            version: 1
        }]
    );
}

#[test]
fn a_job_that_waits_again_after_being_placed_counts_its_wait_from_zero() {
    let mut pool = Pool::new(
        vec![member("a", false), member("b", false)],
        vec![offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)])],
    );
    pool.submit(one_part("holder", 1, 12), 0.0).unwrap();
    pool.submit(one_part("late", 1, 12), 0.0).unwrap();
    // It waits 100 s, is placed when a device joins, and loses that device 10 s later.
    pool.join(
        None,
        offer(device("b", 0), Backend::Cuda, 16, &[("u", 0.01)]),
        100.0,
    );
    assert_eq!(pool.describe().jobs[1].waited_seconds, 0.0);
    pool.leave(&device("b", 0), 110.0);
    assert_eq!(pool.describe().jobs[1].waited_seconds, 0.0);
    // Its wait now runs from the leave, not from its arrival.
    pool.end("no such job", 130.0);
    assert_eq!(pool.describe().jobs[1].waited_seconds, 20.0);
}
