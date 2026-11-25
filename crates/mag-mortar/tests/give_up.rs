//! Which device the pool gives up when its host takes one back by need: of the devices of the
//! kind whose members give a device back in time, the one whose loss costs the pool least, the
//! coordinator's own last.

mod common;

use common::*;
use mag_by::{Backend, DeviceId, DeviceKind, Ledger, Mark, Place, Tolerance};
use mag_mortar::{GiveUp, Pool, Unmet};

const GIB: u64 = 1 << 30;

fn one_part(name: &str, priority: u32, unit: &str, memory_gib: u64) -> mag_by::Graph {
    job(
        name,
        priority,
        vec![],
        vec![part("p", unit, Mark::Atomic, memory_gib)],
        vec![],
    )
}

fn at_least(memory_gib: u64) -> DeviceKind {
    DeviceKind {
        memory_bytes: Some(memory_gib * GIB),
        ..DeviceKind::default()
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn ranked(g: &GiveUp) -> Vec<DeviceId> {
    g.ranked.iter().map(|(d, _)| d.clone()).collect()
}

/// Two fast cards of 16 GiB, `a/0` holding `low` and `b/0` holding `high`, of twice its
/// priority, and a slow card `c/0` of `c_gib`.
fn two_jobs(c_gib: u64) -> Pool {
    let mut pool = Pool::new(
        vec![member("a", Owner), member("b", Owner), member("c", Owner)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("c", 0), Backend::Cuda, c_gib, &[("u", 0.04)]),
        ],
    );
    pool.submit(one_part("low", 1, "u", 4), Stamp::default(), 0.0)
        .unwrap();
    pool.submit(one_part("high", 2, "u", 4), Stamp::default(), 0.0)
        .unwrap();
    assert_eq!(
        pool.arrangement("low").unwrap().places,
        [Place::On(device("a", 0))]
    );
    assert_eq!(
        pool.arrangement("high").unwrap().places,
        [Place::On(device("b", 0))]
    );
    pool
}

#[test]
fn the_device_whose_jobs_lose_least_is_given_up_their_jobs_placed_again_where_they_fit() {
    // `c/0` is not of the kind, but has room: the job on the device given up moves to it, at a
    // quarter of its rate. Losing `a/0` costs `low` 3/4 at priority 1; losing `b/0` costs
    // `high` as much at priority 2, or shares `a/0` with `low` for as much.
    let mut pool = two_jobs(8);
    let before = pool.describe();
    let chosen = pool.give_up("none", &at_least(16), 10.0, 0.0).unwrap();
    assert_eq!(chosen.device, device("a", 0));
    assert!(close(chosen.loss, 0.75), "{chosen:?}");
    assert_eq!(ranked(&chosen), [device("a", 0), device("b", 0)]);
    assert!(close(chosen.ranked[1].1, 1.5), "{chosen:?}");
    // Asking changes nothing.
    assert_eq!(pool.describe(), before);
    assert_eq!(pool.records().len(), 2);

    // Told of the leave, the loop places `low` where the choice weighed it.
    pool.leave(&chosen.device, 1.0).unwrap();
    assert_eq!(
        pool.arrangement("low").unwrap().places,
        [Place::On(device("c", 0))]
    );

    // With no room on `c/0`, `low` waits and loses all its rate; `high` still shares `a/0`.
    let pool = two_jobs(2);
    let chosen = pool.give_up("none", &at_least(16), 10.0, 0.0).unwrap();
    assert_eq!(chosen.device, device("a", 0));
    assert!(close(chosen.loss, 1.0), "{chosen:?}");
    assert!(close(chosen.ranked[1].1, 1.5), "{chosen:?}");
}

#[test]
fn losing_the_fastest_device_is_weighed_on_the_pools_scale_as_it_is() {
    // `fast` may use `a/0` or the slow `c/0`, `medium` `b/0` or `c/0`. Without `a/0`, `fast`'s
    // best rate would be `c/0`'s, and on that scale losing `a/0` would cost nothing; on the
    // pool's scale as it is, `fast` loses 3/4 of its rate and `medium` half of its.
    let mut pool = Pool::new(
        vec![member("a", Owner), member("b", Owner), member("c", Owner)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("f", 0.01)]),
            offer(device("b", 0), Backend::Cuda, 16, &[("m", 0.02)]),
            offer(
                device("c", 0),
                Backend::Cuda,
                8,
                &[("f", 0.04), ("m", 0.04)],
            ),
        ],
    );
    let stamped = |devices: [DeviceId; 2]| Stamp {
        devices: Some(devices.to_vec()),
        ..Stamp::default()
    };
    pool.submit(
        one_part("fast", 1, "f", 4),
        stamped([device("a", 0), device("c", 0)]),
        0.0,
    )
    .unwrap();
    pool.submit(
        one_part("medium", 1, "m", 4),
        stamped([device("b", 0), device("c", 0)]),
        0.0,
    )
    .unwrap();
    let chosen = pool.give_up("none", &at_least(16), 10.0, 0.0).unwrap();
    assert_eq!(chosen.device, device("b", 0), "{chosen:?}");
    assert!(close(chosen.loss, 0.5), "{chosen:?}");
    assert!(close(chosen.ranked[1].1, 0.75), "{chosen:?}");
}

#[test]
fn a_waiting_job_placed_in_the_room_a_leave_frees_counts_at_its_priority_grown_by_its_wait() {
    // `pair` holds both cards; `later` waits for room. Losing either card makes `pair` wait,
    // and `later` takes the other: at 0 the two weigh the same, and at 60, `later` having
    // waited 60 s at priority 1, the pool gains by it.
    let mut pool = Pool::new(
        vec![member("a", Owner), member("b", Owner)],
        vec![
            offer(device("a", 0), Backend::Cuda, 4, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Cuda, 4, &[("u", 0.01)]),
        ],
    );
    let pair = job(
        "pair",
        1,
        vec![],
        vec![
            part("p", "u", Mark::Atomic, 4),
            part("q", "u", Mark::Atomic, 4),
        ],
        vec![],
    );
    pool.submit(pair, Stamp::default(), 0.0).unwrap();
    pool.submit(one_part("later", 1, "u", 4), Stamp::default(), 0.0)
        .unwrap();
    assert!(
        pool.arrangement("later")
            .unwrap()
            .places
            .contains(&Place::Waiting)
    );
    let now = pool
        .give_up("none", &DeviceKind::default(), 10.0, 0.0)
        .unwrap();
    assert!(close(now.loss, 0.0), "{now:?}");
    let later = pool
        .give_up("none", &DeviceKind::default(), 10.0, 60.0)
        .unwrap();
    assert!(close(later.loss, -1.0), "{later:?}");
    // Of the two equal losses, the first device.
    assert_eq!(later.device, device("a", 0));
    // The pool's time is its own: asking at 60 moved nothing.
    assert_eq!(pool.describe().at, 0.0);
}

#[test]
fn only_the_devices_the_kind_admits_are_weighed() {
    // `a/0`, of a member at `site`, holds nothing; `b/0`, at `owner`, holds a job.
    let mut pool = Pool::new(
        vec![member("a", Site), member("b", Owner)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Rocm, 16, &[("u", 0.01)]),
        ],
    );
    pool.submit(
        job(
            "kept",
            1,
            vec![input("x", Owner)],
            vec![part("p", "u", Mark::Atomic, 4)],
            vec![edge("x", Some("p"))],
        ),
        Stamp::default(),
        0.0,
    )
    .unwrap();
    assert_eq!(
        pool.arrangement("kept").unwrap().places,
        [Place::On(device("b", 0))]
    );
    let any = pool
        .give_up("none", &DeviceKind::default(), 10.0, 0.0)
        .unwrap();
    assert_eq!(any.device, device("a", 0));
    assert!(close(any.loss, 0.0));
    assert_eq!(ranked(&any), [device("a", 0), device("b", 0)]);

    // By trust, backend and memory, only `b/0` is of each kind, and it is given up whatever it
    // costs.
    for kind in [
        DeviceKind {
            trust: Some(Owner),
            ..DeviceKind::default()
        },
        DeviceKind {
            backend: Some(Backend::Rocm),
            ..DeviceKind::default()
        },
    ] {
        let chosen = pool.give_up("none", &kind, 10.0, 0.0).unwrap();
        assert_eq!(ranked(&chosen), [device("b", 0)], "{kind:?}");
        assert!(close(chosen.loss, 1.0), "{chosen:?}");
    }
    // A tolerance class no device meets: nothing is of the kind.
    let exact = DeviceKind {
        tolerances: vec![Tolerance::Exact],
        ..DeviceKind::default()
    };
    assert_eq!(
        pool.give_up("none", &exact, 10.0, 0.0),
        Err(Unmet { slow: vec![] })
    );
    assert_eq!(
        pool.give_up("none", &at_least(17), 10.0, 0.0),
        Err(Unmet { slow: vec![] })
    );
}

#[test]
fn of_equal_losses_the_device_with_the_least_memory_then_the_first_by_id() {
    let pool = Pool::new(
        vec![member("a", Owner), member("b", Owner)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[]),
            offer(device("b", 1), Backend::Cuda, 8, &[]),
            offer(device("b", 0), Backend::Cuda, 8, &[]),
            offer(device("a", 1), Backend::Cuda, 24, &[]),
        ],
    );
    let chosen = pool
        .give_up("none", &DeviceKind::default(), 10.0, 0.0)
        .unwrap();
    assert_eq!(
        ranked(&chosen),
        [
            device("b", 0),
            device("b", 1),
            device("a", 0),
            device("a", 1)
        ]
    );
    assert_eq!(chosen.device, device("b", 0));
    assert!(chosen.ranked.iter().all(|&(_, loss)| loss == 0.0));
}

#[test]
fn nothing_meets_a_need_with_no_device_of_its_kind_or_none_whose_member_is_in_time() {
    // `common`'s members give a device back within 5 s; `x` the pool was never told of.
    let pool = Pool::new(
        vec![member("a", Owner)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[]),
            offer(device("x", 0), Backend::Rocm, 16, &[]),
        ],
    );
    let cuda = DeviceKind {
        backend: Some(Backend::Cuda),
        ..DeviceKind::default()
    };
    // Exactly at its time, a member is in time; a moment less, it is not.
    assert_eq!(
        pool.give_up("none", &cuda, 5.0, 0.0).unwrap().device,
        device("a", 0)
    );
    let late = pool.give_up("none", &cuda, 4.999, 0.0).unwrap_err();
    assert_eq!(
        late,
        Unmet {
            slow: vec![device("a", 0)]
        }
    );
    assert_eq!(
        late.to_string(),
        "the devices of the kind asked for (a/0) are each of a member that takes longer than \
         the deadline to give a device back"
    );
    // A member the pool does not know meets no deadline, and is never weighed.
    let rocm = DeviceKind {
        backend: Some(Backend::Rocm),
        ..DeviceKind::default()
    };
    assert_eq!(
        pool.give_up("none", &rocm, 1e9, 0.0),
        Err(Unmet {
            slow: vec![device("x", 0)]
        })
    );
    let any = pool
        .give_up("none", &DeviceKind::default(), 1e9, 0.0)
        .unwrap();
    assert_eq!(ranked(&any), [device("a", 0)]);
    // A deadline below 0, or no number, meets no member.
    for deadline in [-1.0, f64::NAN] {
        assert_eq!(
            pool.give_up("none", &cuda, deadline, 0.0),
            Err(Unmet {
                slow: vec![device("a", 0)]
            })
        );
    }
    let none = pool
        .give_up(
            "none",
            &DeviceKind {
                backend: Some(Backend::Mlx),
                ..DeviceKind::default()
            },
            10.0,
            0.0,
        )
        .unwrap_err();
    assert_eq!(none, Unmet { slow: vec![] });
    assert_eq!(
        none.to_string(),
        "no device in the pool is of the kind asked for"
    );
    // An empty pool meets nothing.
    assert_eq!(
        Pool::new(vec![], vec![]).give_up("none", &DeviceKind::default(), 10.0, 0.0),
        Err(Unmet { slow: vec![] })
    );
}

#[test]
fn the_coordinators_own_devices_come_last_and_are_given_up_when_no_other_meets_the_need() {
    // `b/1` has no room for a job.
    let mut pool = Pool::new(
        vec![member("a", Owner), member("b", Owner)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("b", 1), Backend::Rocm, 2, &[("u", 0.01)]),
        ],
    );
    pool.submit(one_part("j", 1, "u", 4), Stamp::default(), 0.0)
        .unwrap();
    pool.submit(one_part("k", 1, "u", 4), Stamp::default(), 0.0)
        .unwrap();
    // `j` on `a/0`, `k` on `b/0`; `b/1` holds nothing.
    let cuda = DeviceKind {
        backend: Some(Backend::Cuda),
        ..DeviceKind::default()
    };
    let held = |d: &DeviceId| {
        ["j", "k"]
            .iter()
            .any(|j| pool.arrangement(j).unwrap().places == [Place::On(d.clone())])
    };
    assert!(held(&device("a", 0)) && held(&device("b", 0)));
    // Coordinated by `a`: `b/0` is given up before `a/0`, though both lose as much: losing
    // either, the two jobs share the other.
    let chosen = pool.give_up("a", &cuda, 10.0, 0.0).unwrap();
    assert_eq!(ranked(&chosen), [device("b", 0), device("a", 0)]);
    assert!(close(chosen.loss, 1.0));
    // Coordinated by `b`: `a/0` before `b/0`, though both lose as much.
    let chosen = pool.give_up("b", &cuda, 10.0, 0.0).unwrap();
    assert_eq!(ranked(&chosen), [device("a", 0), device("b", 0)]);
    // Of any kind, `b/1`, holding nothing, before both, unless `b` coordinates.
    let any = DeviceKind::default();
    assert_eq!(
        ranked(&pool.give_up("a", &any, 10.0, 0.0).unwrap()),
        [device("b", 1), device("b", 0), device("a", 0)]
    );
    assert_eq!(
        ranked(&pool.give_up("b", &any, 10.0, 0.0).unwrap()),
        [device("a", 0), device("b", 1), device("b", 0)]
    );
    // When only the coordinator's own device is of the kind, it is given up.
    let rocm = DeviceKind {
        backend: Some(Backend::Rocm),
        ..DeviceKind::default()
    };
    assert_eq!(
        pool.give_up("b", &rocm, 10.0, 0.0).unwrap().device,
        device("b", 1)
    );
    pool.leave(&device("b", 1), 1.0).unwrap();
    assert_eq!(
        pool.give_up("b", &rocm, 10.0, 1.0),
        Err(Unmet { slow: vec![] })
    );
}

#[test]
fn the_choice_is_the_same_after_a_replay_in_memory_and_through_the_ledgers_lines() {
    let begin = || {
        Pool::new(
            vec![member("a", Owner), member("b", Site)],
            vec![
                offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)]),
                offer(device("b", 0), Backend::Cuda, 16, &[("u", 0.02)]),
            ],
        )
    };
    let mut pool = begin();
    pool.submit(one_part("j", 2, "u", 8), Stamp::default(), 10.0)
        .unwrap();
    pool.join(
        Some(member("c", Owner)),
        offer(device("c", 0), Backend::Cuda, 8, &[("u", 0.04)]),
        20.0,
    )
    .unwrap();
    pool.submit(one_part("k", 1, "u", 12), Stamp::default(), 30.0)
        .unwrap();
    pool.submit(one_part("w", 3, "u", 16), Stamp::default(), 40.0)
        .unwrap();
    pool.leave(&device("b", 0), 50.0).unwrap();
    pool.end("j", 60.0).unwrap();

    let chosen = |p: &Pool| {
        [
            (DeviceKind::default(), 70.0),
            (at_least(16), 200.0),
            (
                DeviceKind {
                    trust: Some(Owner),
                    ..DeviceKind::default()
                },
                1000.0,
            ),
        ]
        .map(|(kind, at)| p.give_up("a", &kind, 10.0, at))
    };
    let mine = chosen(&pool);
    assert!(mine.iter().all(Result::is_ok), "{mine:?}");
    assert_eq!(chosen(&Pool::replay(pool.ledger()).unwrap()), mine);
    let lines = Ledger::from_lines(&pool.ledger().to_lines())
        .unwrap()
        .ledger;
    assert_eq!(chosen(&Pool::replay(&lines).unwrap()), mine);

    // Every prefix replays to the choice the pool made at that point.
    let mut again = begin();
    for (n, e) in pool.ledger().entries.iter().enumerate() {
        again.apply(e.change.clone(), e.at).unwrap();
        let mut upto = pool.ledger().clone();
        upto.entries.truncate(n + 1);
        assert_eq!(
            chosen(&Pool::replay(&upto).unwrap()),
            chosen(&again),
            "entry {n}"
        );
    }
}
