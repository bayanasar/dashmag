//! What the pool asks of its host: the kind of device it most wants while jobs wait, and the
//! members it would give back once they have held no part for the idle time.

mod common;

use common::*;
use mag_by::{Backend, DeviceKind, Dtype, Ledger, Mark, Pin, Place, Tolerance};
use mag_mortar::Pool;

const GIB: u64 = 1 << 30;

/// The kind the test jobs' parts ask for: `common`'s parts are of tolerance bf16, and their edges
/// carry bf16.
fn kind(memory_gib: u64, edges: bool) -> DeviceKind {
    DeviceKind {
        memory_bytes: Some(memory_gib * GIB),
        tolerances: vec![Tolerance::Bf16],
        dtypes: if edges { vec![Dtype::Bf16] } else { vec![] },
        ..DeviceKind::default()
    }
}

fn one_part(name: &str, priority: u32, memory_gib: u64) -> mag_by::Graph {
    job(
        name,
        priority,
        vec![],
        vec![part("p", "u", Mark::Atomic, memory_gib)],
        vec![],
    )
}

fn waits(pool: &Pool, job: &str) -> bool {
    pool.arrangement(job)
        .unwrap()
        .places
        .iter()
        .any(|p| matches!(p, Place::Waiting))
}

#[test]
fn a_job_wanting_more_memory_asks_for_a_device_with_that_much_and_one_joining_places_it() {
    let mut pool = Pool::new(
        vec![member("a", Owner)],
        vec![offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)])],
    );
    assert_eq!(pool.wants(0.0), None);
    pool.submit(one_part("big", 1, 24), Stamp::default(), 0.0)
        .unwrap();
    assert!(waits(&pool, "big"));
    let wanted = pool.wants(1.0).unwrap();
    assert_eq!(wanted, kind(24, false));
    // Asking changes nothing.
    assert_eq!(pool.records().len(), 1);
    assert_eq!(pool.describe().at, 0.0);

    // A device of that kind, from a member the host trusts least, joins and takes it.
    let joining = offer(device("b", 0), Backend::Flex, 24, &[]);
    assert!(wanted.admits(&joining, Outside));
    pool.join(Some(member("b", Outside)), joining, 2.0).unwrap();
    assert!(!waits(&pool, "big"));
    assert_eq!(pool.wants(3.0), None);
}

#[test]
fn a_replicable_part_with_room_for_no_copy_asks_for_one_copy() {
    let mut pool = Pool::new(
        vec![member("a", Owner)],
        vec![offer(device("a", 0), Backend::Cuda, 4, &[("u", 0.01)])],
    );
    // A part of 1 GiB the pool holds, then copies of 6 GiB: one copy is asked for, not the job.
    let g = job(
        "map",
        1,
        vec![input("items", Outside)],
        vec![
            part("first", "u", Mark::Atomic, 1),
            part("copy", "u", Mark::Replicable, 6),
        ],
        vec![edge("items", Some("first")), edge("first", Some("copy"))],
    );
    pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert!(waits(&pool, "map"));
    assert_eq!(pool.wants(0.0), Some(kind(6, true)));
}

#[test]
fn a_pin_to_a_backend_the_pool_lacks_asks_for_that_backend() {
    let mut pool = Pool::new(
        vec![member("a", Owner)],
        vec![offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)])],
    );
    let g = job(
        "on-rocm",
        1,
        vec![],
        vec![pinned(
            part("p", "u", Mark::Atomic, 1),
            Pin::Backend(Backend::Rocm),
        )],
        vec![],
    );
    pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert!(waits(&pool, "on-rocm"));
    let wanted = pool.wants(0.0).unwrap();
    assert_eq!(
        wanted,
        DeviceKind {
            backend: Some(Backend::Rocm),
            ..kind(1, false)
        }
    );
    pool.join(
        Some(member("b", Owner)),
        offer(device("b", 0), Backend::Rocm, 1, &[]),
        1.0,
    )
    .unwrap();
    assert!(!waits(&pool, "on-rocm"));
}

#[test]
fn a_kind_that_names_no_backend_is_not_taken_to_meet_a_pin() {
    // Two jobs wait on a pool with no room: `any` asks for any device of 1 GiB, and `on-cpu`, of
    // a higher priority, for a CPU of 1 GiB. Tried first, as the smaller kind, any device would
    // be the CPU's gain too if it met the pin; it does not, so the CPU is asked for.
    let mut pool = Pool::new(
        vec![member("a", Owner)],
        vec![offer(device("a", 0), Backend::Cuda, 0, &[("u", 0.01)])],
    );
    pool.submit(one_part("any", 1, 1), Stamp::default(), 0.0)
        .unwrap();
    let g = job(
        "on-cpu",
        5,
        vec![],
        vec![pinned(
            part("p", "u", Mark::Atomic, 1),
            Pin::Backend(Backend::Flex),
        )],
        vec![],
    );
    pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert!(waits(&pool, "any") && waits(&pool, "on-cpu"));
    assert_eq!(
        pool.wants(0.0),
        Some(DeviceKind {
            backend: Some(Backend::Flex),
            ..kind(1, false)
        })
    );
}

#[test]
fn data_kept_above_every_members_trust_asks_for_a_member_trusted_that_far() {
    let mut pool = Pool::new(
        vec![member("a", Site), member("b", Outside)],
        vec![
            offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Cuda, 16, &[("u", 0.01)]),
        ],
    );
    let g = job(
        "label",
        1,
        vec![input("frames", Owner)],
        vec![part("detect", "u", Mark::Atomic, 1)],
        vec![edge("frames", Some("detect"))],
    );
    pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert!(waits(&pool, "label"));
    let wanted = pool.wants(0.0).unwrap();
    assert_eq!(
        wanted,
        DeviceKind {
            trust: Some(Owner),
            ..kind(1, true)
        }
    );
    // A device of the same offer at `site` is not of it; at `owner` it is, and takes the job.
    let joining = offer(device("c", 0), Backend::Cuda, 16, &[]);
    assert!(!wanted.admits(&joining, Site));
    assert!(wanted.admits(&joining, Owner));
    pool.join(Some(member("c", Owner)), joining, 1.0).unwrap();
    assert!(!waits(&pool, "label"));

    // Kept at `site`, the kind asks for that much and no more; a member already at `site` has
    // no room, so it is the trust a new one must have.
    let mut pool = Pool::new(
        vec![member("a", Site), member("b", Outside)],
        vec![
            offer(device("a", 0), Backend::Cuda, 0, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Cuda, 16, &[("u", 0.01)]),
        ],
    );
    let g = job(
        "label",
        1,
        vec![input("frames", Site)],
        vec![part("detect", "u", Mark::Atomic, 1)],
        vec![edge("frames", Some("detect"))],
    );
    pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert_eq!(
        pool.wants(0.0),
        Some(DeviceKind {
            trust: Some(Site),
            ..kind(1, true)
        })
    );
}

#[test]
fn a_tolerance_class_and_a_dtype_no_device_meets_are_asked_for() {
    let mut pool = Pool::new(
        vec![member("a", Owner)],
        vec![offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)])],
    );
    let mut exact = part("p", "u", Mark::Atomic, 1);
    exact.tolerance = Tolerance::Exact;
    let mut wide = edge("x", Some("p"));
    wide.dtype = Dtype::F64;
    let mut bytes = edge("p", None);
    bytes.dtype = Dtype::Bytes;
    let g = job(
        "exact",
        1,
        vec![input("x", Outside)],
        vec![exact],
        vec![wide, bytes],
    );
    pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert!(waits(&pool, "exact"));
    // The bytes edge asks nothing of a device.
    assert_eq!(
        pool.wants(0.0),
        Some(DeviceKind {
            memory_bytes: Some(GIB),
            tolerances: vec![Tolerance::Exact],
            dtypes: vec![Dtype::F64],
            ..DeviceKind::default()
        })
    );
}

#[test]
fn of_equal_gains_the_smaller_kind_is_asked_for_and_a_greater_gain_wins_over_size() {
    let empty = || {
        Pool::new(
            vec![member("a", Owner)],
            vec![offer(device("a", 0), Backend::Cuda, 2, &[("u", 0.01)])],
        )
    };
    // Two jobs alike but for their memory, arrived together: the smaller is asked for, whichever
    // arrived first.
    for order in [["eight", "four"], ["four", "eight"]] {
        let mut pool = empty();
        for name in order {
            let gib = if name == "eight" { 8 } else { 4 };
            pool.submit(one_part(name, 1, gib), Stamp::default(), 0.0)
                .unwrap();
        }
        assert_eq!(pool.wants(0.0), Some(kind(4, false)), "{order:?}");
    }

    // Of equal memory, the lower trust.
    let mut pool = empty();
    let kept = job(
        "kept",
        1,
        vec![input("x", Owner)],
        vec![part("p", "u", Mark::Atomic, 4)],
        vec![edge("x", Some("p"))],
    );
    let free = job(
        "free",
        1,
        vec![input("x", Outside)],
        vec![part("p", "u", Mark::Atomic, 4)],
        vec![edge("x", Some("p"))],
    );
    pool.submit(kept, Stamp::default(), 0.0).unwrap();
    pool.submit(free, Stamp::default(), 0.0).unwrap();
    assert_eq!(pool.wants(0.0), Some(kind(4, true)));

    // Of equal memory and trust, no backend before one named, and then the backends in order.
    let mut pool = empty();
    let on = |name: &str, b: Backend| {
        job(
            name,
            1,
            vec![],
            vec![pinned(part("p", "u", Mark::Atomic, 4), Pin::Backend(b))],
            vec![],
        )
    };
    pool.submit(on("mlx", Backend::Mlx), Stamp::default(), 0.0)
        .unwrap();
    pool.submit(on("rocm", Backend::Rocm), Stamp::default(), 0.0)
        .unwrap();
    assert_eq!(
        pool.wants(0.0),
        Some(DeviceKind {
            backend: Some(Backend::Rocm),
            ..kind(4, false)
        })
    );
    pool.submit(one_part("any", 1, 4), Stamp::default(), 0.0)
        .unwrap();
    assert_eq!(pool.wants(0.0), Some(kind(4, false)));

    // A higher priority wins over a smaller kind.
    let mut pool = empty();
    pool.submit(one_part("small", 1, 4), Stamp::default(), 0.0)
        .unwrap();
    pool.submit(one_part("urgent", 3, 8), Stamp::default(), 0.0)
        .unwrap();
    assert_eq!(pool.wants(0.0), Some(kind(8, false)));

    // So may a longer wait, at the time of asking. At 90 `old`, of priority 1, has waited 90 s,
    // which counts as 2.5, and `new`, of priority 2, has just come: 2. At 200 they count as 4.3
    // and 5.7.
    let mut pool = empty();
    pool.submit(one_part("old", 1, 8), Stamp::default(), 0.0)
        .unwrap();
    pool.submit(one_part("new", 2, 4), Stamp::default(), 90.0)
        .unwrap();
    assert_eq!(pool.wants(90.0), Some(kind(8, false)));
    assert_eq!(pool.wants(200.0), Some(kind(4, false)));
}

#[test]
fn a_kinds_device_takes_only_what_the_kind_names() {
    let full = || {
        Pool::new(
            vec![member("a", Owner)],
            vec![offer(device("a", 0), Backend::Cuda, 0, &[("u", 0.01)])],
        )
    };
    // `free` asks for any member, `kept`, of the higher priority, for one at `owner`. Tried
    // first, a device of a member at `outside` would be `kept`'s gain too if the guard let it
    // hold the part; it does not, so the member at `owner` is asked for.
    let mut pool = full();
    let kept = job(
        "kept",
        5,
        vec![input("x", Owner)],
        vec![part("p", "u", Mark::Atomic, 4)],
        vec![edge("x", Some("p"))],
    );
    let free = job(
        "free",
        1,
        vec![input("x", Outside)],
        vec![part("p", "u", Mark::Atomic, 4)],
        vec![edge("x", Some("p"))],
    );
    pool.submit(kept, Stamp::default(), 0.0).unwrap();
    pool.submit(free, Stamp::default(), 0.0).unwrap();
    assert_eq!(
        pool.wants(0.0),
        Some(DeviceKind {
            trust: Some(Owner),
            ..kind(4, true)
        })
    );

    // The same for a tolerance class: a device meeting `exact`, tried first, does not meet
    // `bf16` unless the kind names it.
    let mut pool = full();
    let mut exact = part("p", "u", Mark::Atomic, 4);
    exact.tolerance = Tolerance::Exact;
    pool.submit(
        job("exact", 1, vec![], vec![exact], vec![]),
        Stamp::default(),
        0.0,
    )
    .unwrap();
    pool.submit(one_part("bf16", 5, 4), Stamp::default(), 0.0)
        .unwrap();
    assert_eq!(pool.wants(0.0), Some(kind(4, false)));

    // And for a dtype: a device computing in f64 alone, tried first, does not compute in f32.
    let mut pool = full();
    let on = |name: &str, priority: u32, dtype: Dtype| {
        let mut e = edge("x", Some("p"));
        e.dtype = dtype;
        job(
            name,
            priority,
            vec![input("x", Outside)],
            vec![part("p", "u", Mark::Atomic, 4)],
            vec![e],
        )
    };
    pool.submit(on("f64", 1, Dtype::F64), Stamp::default(), 0.0)
        .unwrap();
    pool.submit(on("f32", 5, Dtype::F32), Stamp::default(), 0.0)
        .unwrap();
    assert_eq!(
        pool.wants(0.0),
        Some(DeviceKind {
            dtypes: vec![Dtype::F32],
            ..kind(4, false)
        })
    );
}

#[test]
fn a_job_the_pool_would_place_on_its_own_devices_asks_for_no_new_one() {
    // `low` waits beside `high`, which it would halve, and is stamped to that device alone. By
    // 600 s its wait has raised it past half of `high`, so the loop, run then, would place it
    // there: the gain is the pool's own, not a new device's.
    let mut pool = Pool::new(
        vec![member("a", Owner)],
        vec![offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)])],
    );
    pool.submit(one_part("high", 10, 1), Stamp::default(), 0.0)
        .unwrap();
    let only_a = Stamp {
        devices: Some(vec![device("a", 0)]),
        ..Stamp::default()
    };
    pool.submit(one_part("low", 1, 1), only_a, 0.0).unwrap();
    assert!(waits(&pool, "low"));
    assert_eq!(pool.wants(0.0), None);
    assert_eq!(pool.wants(600.0), None);
    // Told of a change at 600, the pool places it so.
    pool.join(None, offer(device("a", 1), Backend::Cuda, 0, &[]), 600.0)
        .unwrap();
    assert!(!waits(&pool, "low"));
}

#[test]
fn a_group_is_asked_for_rather_than_the_whole_job_when_the_pool_has_room_for_the_rest() {
    // Two parts of 4 GiB and one device of 4 GiB: one more device of 4 GiB places the job.
    let mut pool = Pool::new(
        vec![member("a", Owner)],
        vec![offer(device("a", 0), Backend::Cuda, 4, &[("u", 0.01)])],
    );
    let g = job(
        "two",
        1,
        vec![],
        vec![
            part("p", "u", Mark::Atomic, 4),
            part("q", "u", Mark::Atomic, 4),
        ],
        vec![edge("p", Some("q"))],
    );
    pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert!(waits(&pool, "two"));
    assert_eq!(pool.wants(0.0), Some(kind(4, true)));

    // With no device of its own able to take either part, a device of 4 GiB takes only one: the
    // whole job is asked for.
    let mut pool = Pool::new(
        vec![member("a", Owner)],
        vec![offer(device("a", 0), Backend::Cuda, 2, &[("u", 0.01)])],
    );
    let g = job(
        "two",
        1,
        vec![],
        vec![
            part("p", "u", Mark::Atomic, 4),
            part("q", "u", Mark::Atomic, 4),
        ],
        vec![edge("p", Some("q"))],
    );
    pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert_eq!(pool.wants(0.0), Some(kind(8, true)));
}

#[test]
fn nothing_is_wanted_while_nothing_waits_or_no_device_kind_would_serve() {
    let mut pool = Pool::new(
        vec![member("a", Owner)],
        vec![offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)])],
    );
    assert_eq!(pool.wants(0.0), None);
    pool.submit(one_part("fits", 1, 1), Stamp::default(), 0.0)
        .unwrap();
    assert!(!waits(&pool, "fits"));
    assert_eq!(pool.wants(0.0), None);
    assert_eq!(pool.asks("a", 0.0, 0.0).wants, None);

    // Pinned to a member the pool lacks: no kind of device is that member.
    let g = job(
        "on-b",
        1,
        vec![],
        vec![pinned(
            part("p", "u", Mark::Atomic, 1),
            Pin::Member("b".into()),
        )],
        vec![],
    );
    pool.submit(g, Stamp::default(), 1.0).unwrap();
    assert!(waits(&pool, "on-b"));
    assert_eq!(pool.wants(1.0), None);

    // Stamped to a device that has left: no new device is among those its stamp allows.
    let stamped = Stamp {
        devices: Some(vec![device("c", 0)]),
        ..Stamp::default()
    };
    pool.submit(one_part("on-c", 1, 32), stamped, 2.0).unwrap();
    assert!(waits(&pool, "on-c"));
    assert_eq!(pool.wants(2.0), None);
}

/// A pool of the coordinator `a` and members `b` and `c`, each with one device, begun at 0.
fn three() -> Pool {
    Pool::new(
        vec![member("a", Owner), member("b", Owner), member("c", Owner)],
        vec![
            offer(device("a", 0), Backend::Cuda, 1, &[("u", 0.01)]),
            offer(device("b", 0), Backend::Cuda, 16, &[("u", 0.01)]),
            offer(device("c", 0), Backend::Cuda, 16, &[("u", 0.02)]),
        ],
    )
}

#[test]
fn a_member_is_given_back_once_it_has_held_no_part_for_the_idle_time() {
    let mut pool = three();
    // Nothing held since the pool began: at the idle time, each member but the coordinator.
    assert_eq!(pool.gives_back("a", 50.0, 49.0), Vec::<String>::new());
    assert_eq!(pool.gives_back("a", 50.0, 50.0), ["b", "c"]);

    // `b` takes a job at 10, the fastest device that holds it, and holds it until it ends at 120.
    pool.submit(one_part("j", 1, 2), Stamp::default(), 10.0)
        .unwrap();
    assert_eq!(
        pool.arrangement("j").unwrap().places,
        [Place::On(device("b", 0))]
    );
    assert_eq!(pool.gives_back("a", 50.0, 100.0), ["c"]);
    pool.end("j", 120.0).unwrap();
    assert_eq!(pool.gives_back("a", 50.0, 169.0), ["c"]);
    // Longest idle first.
    assert_eq!(pool.gives_back("a", 50.0, 170.0), ["c", "b"]);
    // An idle time of 0 gives back every member holding nothing, at once.
    assert_eq!(pool.gives_back("a", 0.0, 120.0), ["c", "b"]);
    // A time earlier than the pool's is the pool's.
    assert_eq!(pool.gives_back("a", 0.0, 0.0), ["c", "b"]);
    assert_eq!(pool.asks("a", 50.0, 170.0).gives_back, ["c", "b"]);
}

#[test]
fn idle_time_is_counted_from_when_a_part_left_or_a_member_joined() {
    let mut pool = three();
    pool.submit(one_part("j", 1, 2), Stamp::default(), 0.0)
        .unwrap();
    // `b`'s device leaves at 40: the job goes to `c`, and `b`, with a second device that joined
    // at 30, has held nothing since 40.
    pool.join(None, offer(device("b", 1), Backend::Cuda, 1, &[]), 30.0)
        .unwrap();
    pool.leave(&device("b", 0), 40.0).unwrap();
    assert_eq!(
        pool.arrangement("j").unwrap().places,
        [Place::On(device("c", 0))]
    );
    assert_eq!(pool.gives_back("a", 100.0, 139.0), Vec::<String>::new());
    assert_eq!(pool.gives_back("a", 100.0, 140.0), ["b"]);

    // A member whose first device joins at 200 is idle from then, and a second device joining
    // does not start it again.
    pool.join(
        Some(member("d", Owner)),
        offer(device("d", 0), Backend::Cuda, 1, &[]),
        200.0,
    )
    .unwrap();
    pool.join(None, offer(device("d", 1), Backend::Cuda, 1, &[]), 250.0)
        .unwrap();
    assert_eq!(pool.gives_back("a", 100.0, 299.0), ["b"]);
    assert_eq!(pool.gives_back("a", 100.0, 300.0), ["b", "d"]);

    // A member with no device left is no member to give back; joining again starts it anew.
    pool.leave(&device("b", 1), 310.0).unwrap();
    assert_eq!(pool.gives_back("a", 100.0, 400.0), ["d"]);
    pool.join(None, offer(device("b", 0), Backend::Cuda, 1, &[]), 400.0)
        .unwrap();
    assert_eq!(pool.gives_back("a", 100.0, 499.0), ["d"]);
    assert_eq!(pool.gives_back("a", 100.0, 500.0), ["d", "b"]);
}

#[test]
fn the_coordinator_is_never_given_back() {
    let mut pool = three();
    assert_eq!(pool.gives_back("a", 0.0, 1e9), ["b", "c"]);
    assert_eq!(pool.gives_back("b", 0.0, 1e9), ["a", "c"]);
    // A coordinator holding nothing, whatever it offers and however long.
    pool.submit(one_part("j", 1, 2), Stamp::default(), 0.0)
        .unwrap();
    assert_eq!(pool.gives_back("c", 0.0, 1e9), ["a"]);
    assert!(!pool.asks("a", 0.0, 1e9).gives_back.contains(&"a".into()));
}

#[test]
fn the_asks_are_the_same_after_a_replay_in_memory_and_through_the_ledgers_lines() {
    let mut pool = three();
    pool.submit(one_part("j", 1, 2), Stamp::default(), 10.0)
        .unwrap();
    pool.submit(one_part("big", 1, 32), Stamp::default(), 20.0)
        .unwrap();
    pool.join(
        Some(member("d", Site)),
        offer(device("d", 0), Backend::Rocm, 8, &[]),
        30.0,
    )
    .unwrap();
    let g = job(
        "kept",
        2,
        vec![input("x", Owner)],
        vec![part("p", "u", Mark::Atomic, 8)],
        vec![edge("x", Some("p"))],
    );
    pool.submit(g, Stamp::default(), 40.0).unwrap();
    pool.end("j", 50.0).unwrap();
    pool.leave(&device("c", 0), 60.0).unwrap();
    pool.submit(one_part("small", 1, 1), Stamp::default(), 70.0)
        .unwrap();

    let asked = |p: &Pool| {
        [80.0, 200.0, 1000.0].map(|at| {
            (
                p.asks("a", 100.0, at),
                p.wants(at),
                p.gives_back("a", 100.0, at),
            )
        })
    };
    let mine = asked(&pool);
    // Something is asked of both kinds, so the comparison is not of nothing with nothing.
    assert!(mine[0].0.wants.is_some());
    assert!(!mine[2].0.gives_back.is_empty());
    assert_eq!(asked(&Pool::replay(pool.ledger()).unwrap()), mine);
    let lines = Ledger::from_lines(&pool.ledger().to_lines())
        .unwrap()
        .ledger;
    assert_eq!(asked(&Pool::replay(&lines).unwrap()), mine);

    // Every prefix replays to the asks the pool made at that point.
    let mut again = three();
    for (n, e) in pool.ledger().entries.iter().enumerate() {
        again.apply(e.change.clone(), e.at).unwrap();
        let mut upto = pool.ledger().clone();
        upto.entries.truncate(n + 1);
        assert_eq!(
            asked(&Pool::replay(&upto).unwrap()),
            asked(&again),
            "entry {n}"
        );
    }
}
