//! The guard over three levels of trust, and the host's stamp: the devices a job may use and the
//! ceiling on its priority.

mod common;

use common::*;
use mag_by::{Backend, Change, DeviceId, Ledger, Mark, Place, Removal};
use mag_mortar::{ChangeError, Pool, VERSION};

const LEVELS: [Trust; 3] = [Outside, Site, Owner];

/// One part reading one input kept at `keep`.
fn kept_at(keep: Trust) -> mag_by::Graph {
    job(
        "job",
        1,
        vec![input("data", keep)],
        vec![part("p", "u", Mark::Atomic, 1)],
        vec![edge("data", Some("p"))],
    )
}

/// At every pair of levels: a part kept at `keep` is placed on a member trusted at `trust`
/// exactly when `keep` is no higher, and is otherwise removed by the guard, which names both.
#[test]
fn the_guard_at_every_pair_of_levels() {
    for trust in LEVELS {
        for keep in LEVELS {
            let mut pool = Pool::new(
                vec![member("m", trust)],
                vec![offer(device("m", 0), Backend::Cuda, 16, &[("u", 0.01)])],
            );
            let r = pool
                .submit(kept_at(keep), Stamp::default(), 0.0)
                .unwrap()
                .clone();
            let placed = pool.arrangement("job").unwrap().places[0] == Place::On(device("m", 0));
            assert_eq!(placed, keep <= trust, "kept at {keep}, trusted at {trust}");
            let guard = Removal::Guard {
                part: "p".into(),
                device: device("m", 0),
                keep,
                trust,
            };
            let removed = r
                .candidates
                .iter()
                .any(|c| c.removed.as_ref() == Some(&guard));
            assert_eq!(removed, keep > trust, "kept at {keep}, trusted at {trust}");
        }
    }
}

/// One device per level, the faster the less trusted.
fn one_of_each() -> Pool {
    Pool::new(
        vec![
            member("owner", Owner),
            member("site", Site),
            member("outside", Outside),
        ],
        vec![
            offer(device("owner", 0), Backend::Cuda, 16, &[("u", 0.04)]),
            offer(device("site", 0), Backend::Cuda, 16, &[("u", 0.02)]),
            offer(device("outside", 0), Backend::Cuda, 16, &[("u", 0.01)]),
        ],
    )
}

/// A part reading inputs at `site` and at `owner` is kept at `owner`, and so is the part after
/// it unless the edge between them is released: the part after goes on the fastest device its
/// level allows, so a release to each level puts it on that level's device.
#[test]
fn a_part_after_inputs_of_two_levels_goes_where_the_release_allows() {
    for (release, lands) in [
        (None, "owner"),
        (Some(Outside), "outside"),
        (Some(Site), "site"),
        (Some(Owner), "owner"),
    ] {
        let mut after = edge("merge", Some("next"));
        after.release = release;
        let g = job(
            "job",
            1,
            vec![input("lab", Site), input("home", Owner)],
            vec![
                part("merge", "u", Mark::Atomic, 1),
                part("next", "u", Mark::Atomic, 1),
            ],
            vec![
                edge("lab", Some("merge")),
                edge("home", Some("merge")),
                after,
            ],
        );
        let mut pool = one_of_each();
        pool.submit(g, Stamp::default(), 0.0).unwrap();
        assert_eq!(
            pool.arrangement("job").unwrap().places,
            [Place::On(device("owner", 0)), Place::On(device(lands, 0))],
            "released to {release:?}"
        );
    }
    // With only the input at `site`, the part reading it may go on the site's device.
    let g = job(
        "job",
        1,
        vec![input("lab", Site)],
        vec![part("merge", "u", Mark::Atomic, 1)],
        vec![edge("lab", Some("merge"))],
    );
    let mut pool = one_of_each();
    pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert_eq!(
        pool.arrangement("job").unwrap().places,
        [Place::On(device("site", 0))]
    );
}

fn stamped(devices: &[DeviceId]) -> Stamp {
    Stamp {
        devices: Some(devices.to_vec()),
        ..Stamp::default()
    }
}

fn devices_used(pool: &Pool, job: &str) -> Vec<DeviceId> {
    let mut used: Vec<DeviceId> = pool
        .arrangement(job)
        .unwrap()
        .places
        .iter()
        .flat_map(|p| match p {
            Place::Waiting => vec![],
            Place::On(d) => vec![d.clone()],
            Place::Copies(ds) => ds.clone(),
        })
        .collect();
    used.sort();
    used.dedup();
    used
}

/// Four cards, the fastest last.
fn four_cards() -> Pool {
    Pool::new(
        (0..4).map(|i| member(&format!("m{i}"), Owner)).collect(),
        (0..4)
            .map(|i| {
                offer(
                    device(&format!("m{i}"), 0),
                    Backend::Cuda,
                    16,
                    &[("u", 0.04 - 0.01 * i as f64)],
                )
            })
            .collect(),
    )
}

/// A job stamped with two of four devices runs on those two only: its atomic part, its copies,
/// and, past the bound, every part the loop places greedily. A device left out is recorded with
/// the stamp as its reason.
#[test]
fn a_stamped_job_is_never_placed_outside_its_devices() {
    let allowed = [device("m0", 0), device("m1", 0)];
    let g = job(
        "job",
        1,
        vec![input("items", Outside)],
        vec![
            part("a", "u", Mark::Atomic, 1),
            part("r", "u", Mark::Replicable, 1),
        ],
        vec![edge("items", Some("r")), edge("r", Some("a"))],
    );
    let mut pool = four_cards();
    let r = pool.submit(g, stamped(&allowed), 0.0).unwrap().clone();
    assert_eq!(devices_used(&pool, "job"), allowed);
    let chosen = &r.candidates[r.chosen.unwrap()];
    for i in 2..4 {
        let d = device(&format!("m{i}"), 0);
        let reason = Removal::Stamp {
            job: "job".into(),
            device: d.clone(),
        };
        assert!(
            chosen.left_out.contains(&("r".into(), reason.clone())),
            "{d}"
        );
        assert!(
            r.candidates
                .iter()
                .any(|c| c.removed.as_ref() == Some(&reason))
        );
    }
    // Unstamped, the same job uses the faster devices.
    let mut free = four_cards();
    let g = pool.ledger().entries[0].change.clone();
    let Change::Submit { graph, .. } = g else {
        unreachable!()
    };
    free.submit(graph, Stamp::default(), 0.0).unwrap();
    assert!(devices_used(&free, "job").contains(&device("m3", 0)));

    // Twelve parts are past the bound: placed greedily, and still only on the two.
    let parts = (0..12)
        .map(|i| part(&format!("p{i}"), "u", Mark::Atomic, 2))
        .collect();
    let mut pool = four_cards();
    let r = pool
        .submit(
            job("wide", 1, vec![], parts, vec![]),
            stamped(&allowed),
            0.0,
        )
        .unwrap();
    assert!(r.greedy);
    assert!(
        pool.arrangement("wide")
            .unwrap()
            .places
            .iter()
            .all(|p| matches!(p, Place::On(_)))
    );
    assert_eq!(devices_used(&pool, "wide"), allowed);
}

/// A stamp naming a device the pool does not have yet: the job waits for it, and runs on it
/// when it joins.
#[test]
fn a_stamped_job_waits_for_its_device_to_join() {
    let mut pool = four_cards();
    let g = kept_at(Outside);
    pool.submit(g, stamped(&[device("late", 0)]), 0.0).unwrap();
    assert_eq!(pool.arrangement("job").unwrap().places, [Place::Waiting]);
    pool.join(
        Some(member("late", Owner)),
        offer(device("late", 0), Backend::Rocm, 16, &[("u", 0.5)]),
        1.0,
    )
    .unwrap();
    assert_eq!(
        pool.arrangement("job").unwrap().places,
        [Place::On(device("late", 0))]
    );
}

/// Each copy 17 GiB: one fits each of two cards, so two such jobs cannot run at once.
fn two_cards() -> Pool {
    Pool::new(
        vec![member("a", Owner), member("b", Owner)],
        vec![
            offer(device("a", 0), Backend::Cuda, 24, &[("r", 0.01)]),
            offer(device("b", 0), Backend::Rocm, 24, &[("r", 0.02)]),
        ],
    )
}

fn wants_everything(name: &str, priority: u32) -> mag_by::Graph {
    job(
        name,
        priority,
        vec![input("items", Outside)],
        vec![part("r", "r", Mark::Replicable, 17)],
        vec![edge("items", Some("r"))],
    )
}

fn ceiling(c: u32) -> Stamp {
    Stamp {
        priority_ceiling: Some(c),
        ..Stamp::default()
    }
}

/// A job asking for priority 5 under a ceiling of 1 weighs 1: a job of priority 2 that arrived
/// with it runs first. The description shows the priority held to the ceiling, and a ceiling
/// above a job's priority does not raise it.
#[test]
fn a_stamped_priority_ceiling_is_never_exceeded() {
    let mut pool = two_cards();
    pool.submit(wants_everything("blocker", 1), Stamp::default(), 0.0)
        .unwrap();
    pool.submit(wants_everything("asks-5", 5), ceiling(1), 0.0)
        .unwrap();
    pool.submit(wants_everything("plain-2", 2), Stamp::default(), 0.0)
        .unwrap();
    pool.submit(wants_everything("low", 1), ceiling(3), 0.0)
        .unwrap();
    let priorities: Vec<_> = pool
        .describe()
        .jobs
        .iter()
        .map(|h| (h.arrangement.job.clone(), h.priority))
        .collect();
    assert_eq!(
        priorities,
        [
            ("blocker".to_string(), 1),
            ("asks-5".to_string(), 1),
            ("plain-2".to_string(), 2),
            ("low".to_string(), 1)
        ]
    );
    pool.end("blocker", 0.0).unwrap();
    assert!(matches!(
        pool.arrangement("plain-2").unwrap().places[0],
        Place::Copies(_)
    ));
    assert_eq!(pool.arrangement("asks-5").unwrap().places, [Place::Waiting]);
}

/// A ceiling of 0, or a list of no devices, would leave a job that never runs: refused, and
/// the pool is unchanged and logs nothing.
#[test]
fn a_stamp_that_could_never_be_met_is_refused() {
    let mut pool = four_cards();
    let zero = pool.submit(kept_at(Outside), ceiling(0), 0.0).unwrap_err();
    assert_eq!(zero, ChangeError::ZeroCeiling);
    let none = pool
        .submit(kept_at(Outside), stamped(&[]), 0.0)
        .unwrap_err();
    assert_eq!(none, ChangeError::NoDevices);
    assert_eq!(
        none.to_string(),
        "the stamp names no device the job may use"
    );
    assert!(pool.arrangement("job").is_none());
    assert_eq!(pool.ledger().entries.len(), 0);
    // A plan of either is refused the same way.
    assert_eq!(
        pool.plan(&kept_at(Outside), &ceiling(0), 0.0).unwrap_err(),
        ChangeError::ZeroCeiling
    );
}

/// The ledger keeps each submission's stamp and names the version that wrote it; it replays,
/// as lines too, to the same records.
#[test]
fn the_ledger_keeps_the_stamp_and_its_writer() {
    let mut pool = four_cards();
    let stamp = Stamp {
        by: Some("someone".into()),
        devices: Some(vec![device("m1", 0), device("m2", 0)]),
        priority_ceiling: Some(1),
    };
    let small = job(
        "job",
        4,
        vec![],
        vec![part("p", "u", Mark::Atomic, 1)],
        vec![],
    );
    pool.submit(small, stamp.clone(), 0.0).unwrap();
    assert_eq!(
        pool.arrangement("job").unwrap().places,
        [Place::On(device("m2", 0))]
    );
    assert_eq!(pool.describe().jobs[0].priority, 1);
    let ledger = pool.ledger();
    assert_eq!(ledger.written_by.as_deref(), Some(VERSION));
    assert_eq!(ledger.follows, None);
    assert!(matches!(
        &ledger.entries[0].change,
        Change::Submit { stamp: s, .. } if s == &stamp
    ));
    let read = Ledger::from_lines(&ledger.to_lines()).unwrap().ledger;
    assert_eq!(&read, ledger);
    let back = Pool::replay(&read).unwrap();
    assert_eq!(back.describe(), pool.describe());
    assert_eq!(back.ledger(), ledger);
    // A replay keeps the header it was given: another writer, and a ledger it follows.
    let other = Ledger {
        written_by: Some("0.0.0".into()),
        follows: Some("the ledger before".into()),
        ..ledger.clone()
    };
    assert_eq!(Pool::replay(&other).unwrap().ledger(), &other);
}
