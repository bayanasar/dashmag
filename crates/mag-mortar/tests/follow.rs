//! A pool that follows a ledger another version of this crate wrote: begun from what that ledger
//! was told, its members, the devices it held and the jobs it held, and none of its decisions.

mod common;

use common::*;
use mag_by::{Backend, Change, Graph, Ledger, Mark, Member, Place};
use mag_mortar::{ChangeError, Pool, VERSION};

fn one_part(name: &str, gib: u64) -> Graph {
    job(
        name,
        1,
        vec![],
        vec![part("p", "u", Mark::Atomic, gib)],
        vec![],
    )
}

fn submit(graph: Graph) -> Change {
    Change::Submit {
        graph,
        stamp: Stamp::default(),
    }
}

fn cuda(member: &str, index: u32, gib: u64) -> mag_by::Offer {
    offer(device(member, index), Backend::Cuda, gib, &[("u", 0.01)])
}

/// A ledger as a pool of another version might leave it: members that joined, one of them
/// lowered, a device that left, three jobs, one of them ended, and the version that wrote it.
fn old_ledger() -> Ledger {
    let mut pool = Pool::new(vec![member("a", Owner)], vec![cuda("a", 0, 16)]);
    let at = |s: f64| 1000.0 + s;
    let joins = [
        (member("b", Owner), cuda("b", 0, 32)),
        (member("b", Owner), cuda("b", 1, 8)),
        (member("c", Owner), cuda("c", 0, 24)),
        // `b` again, at a lower trust: the lower stands.
        (member("b", Site), cuda("b", 0, 32)),
    ];
    for (i, (m, o)) in joins.into_iter().enumerate() {
        pool.apply(
            Change::Join {
                member: Some(m),
                offer: o,
            },
            at(i as f64),
        )
        .unwrap();
    }
    pool.apply(
        Change::Leave {
            device: device("c", 0),
        },
        at(10.0),
    )
    .unwrap();
    for (i, name) in ["first", "second", "third"].into_iter().enumerate() {
        pool.apply(submit(one_part(name, 4)), at(20.0 + i as f64))
            .unwrap();
    }
    pool.apply(
        Change::End {
            job: "second".into(),
        },
        at(30.0),
    )
    .unwrap();
    let mut ledger = pool.ledger().clone();
    ledger.written_by = Some("0.0.0-elsewhere".into());
    ledger
}

#[test]
fn a_pool_follows_an_old_ledger_from_its_members_devices_and_jobs_and_replays_by_itself() {
    let old = old_ledger();
    let (pool, refused) = Pool::follow(&old, "ledger.jsonl.1", 5000.0);
    assert_eq!(refused, []);
    let ledger = pool.ledger();

    // Its header: this version, the old ledger named, the time it began at, and the facts.
    assert_eq!(ledger.written_by.as_deref(), Some(VERSION));
    assert_eq!(ledger.follows.as_deref(), Some("ledger.jsonl.1"));
    assert_eq!(ledger.at, 5000.0);
    assert_eq!(
        ledger.members,
        [
            member("a", Owner),
            Member {
                trust: Site,
                ..member("b", Owner)
            },
            member("c", Owner),
        ]
    );
    let offered: Vec<_> = ledger.offers.iter().map(|o| o.device.clone()).collect();
    assert_eq!(offered, [device("a", 0), device("b", 0), device("b", 1)]);

    // Its entries: the jobs held, submitted again in the order they arrived, at its time; the
    // ended one not.
    let submitted: Vec<(&str, f64)> = ledger
        .entries
        .iter()
        .map(|e| match &e.change {
            Change::Submit { graph, .. } => (graph.job.name.as_str(), e.at),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(submitted, [("first", 5000.0), ("third", 5000.0)]);
    // Placed afresh, each on a device: the pool holds them, waiting from 0.
    let described = pool.describe();
    assert_eq!(described.jobs.len(), 2);
    for held in &described.jobs {
        assert!(
            held.arrangement
                .places
                .iter()
                .all(|p| matches!(p, Place::On(_))),
            "{held:?}"
        );
        assert_eq!(held.waited_seconds, 0.0);
    }

    // The new ledger replays by itself, to the same lines, its header too.
    let replayed = Pool::replay(ledger).unwrap();
    assert_eq!(replayed.ledger(), ledger);
    assert_eq!(replayed.ledger().to_lines(), ledger.to_lines());
}

#[test]
fn a_job_this_version_refuses_is_left_out_and_returned_with_why() {
    let mut old = old_ledger();
    // A job the old version took and this one refuses: a stamp whose ceiling is 0.
    old.entries.push(mag_by::Entry {
        at: 2000.0,
        change: Change::Submit {
            graph: one_part("zero", 1),
            stamp: Stamp {
                priority_ceiling: Some(0),
                ..Stamp::default()
            },
        },
        record: old.entries.last().unwrap().record.clone(),
    });
    let (pool, refused) = Pool::follow(&old, "old", 5000.0);
    assert_eq!(refused, [("zero".to_string(), ChangeError::ZeroCeiling)]);
    assert!(pool.arrangement("zero").is_none());
    assert!(pool.arrangement("first").is_some());
    assert!(pool.arrangement("third").is_some());
    assert_eq!(pool.ledger().entries.len(), 2);
}

#[test]
fn a_ledger_of_no_entries_is_followed_by_its_header_alone() {
    let pool = Pool::new(vec![member("a", Owner)], vec![cuda("a", 0, 16)]);
    let (followed, refused) = Pool::follow(pool.ledger(), "first", 7.0);
    assert_eq!(refused, []);
    assert_eq!(followed.ledger().members, pool.ledger().members);
    assert_eq!(followed.ledger().offers, pool.ledger().offers);
    assert_eq!(followed.ledger().entries, []);
    assert_eq!(followed.ledger().follows.as_deref(), Some("first"));
}
