//! The pool's state as its host keeps it: the ledger of what the pool was told, replayed to give
//! the pool back, in memory and through its lines.

mod common;

use common::*;
use mag_by::{
    Backend, Change, Description, Entry, Event, Graph, Ledger, LedgerError, Mark, Place, Record,
};
use mag_mortar::{ChangeError, Pool, ReplayError};

/// Copies of 17 GiB of a replicable part: one fits each of the cards of 24 and 32 GiB.
fn wants_everything(name: &str, priority: u32) -> Graph {
    job(
        name,
        priority,
        vec![input("items", false)],
        vec![part("r", "r", Mark::Replicable, 17)],
        vec![edge("items", Some("r"))],
    )
}

fn one_part(name: &str, gib: u64) -> Graph {
    job(
        name,
        1,
        vec![],
        vec![part("p", "u", Mark::Atomic, gib)],
        vec![],
    )
}

/// A job reading local-only frames, of one part of `gib` GiB.
fn local(name: &str, gib: u64) -> Graph {
    job(
        name,
        1,
        vec![input("frames", true)],
        vec![part("p", "u", Mark::Atomic, gib)],
        vec![edge("frames", Some("p"))],
    )
}

fn submit(graph: Graph) -> Change {
    Change::Submit { graph }
}

fn end(job: &str) -> Change {
    Change::End { job: job.into() }
}

fn three_cards() -> Pool {
    Pool::new(
        vec![member("a", false), member("b", false), member("c", false)],
        vec![
            offer(
                device("a", 0),
                Backend::Cuda,
                16,
                &[("u", 0.01), ("r", 0.01)],
            ),
            offer(
                device("b", 0),
                Backend::Rocm,
                32,
                &[("u", 0.02), ("r", 0.015)],
            ),
            offer(
                device("c", 0),
                Backend::Mlx,
                24,
                &[("u", 0.03), ("r", 0.03)],
            ),
        ],
    )
}

/// The pool's state as a test compares it.
type State = (Description, Vec<Record>);

fn state(pool: &Pool) -> State {
    (pool.describe(), pool.records().cloned().collect())
}

/// A run of 24 changes, two of them refused, and the state after each change taken (the first
/// before any). Jobs are placed, wait, are refused and end; a member is reported remote; a
/// device leaves and a job waits for it; and a job that waited passes a newer one of higher
/// priority.
fn scripted() -> (Pool, Vec<State>) {
    let mut pool = three_cards();
    let mut states = vec![state(&pool)];
    let mut take = |pool: &mut Pool, change: Change, at: f64| -> Result<Record, ChangeError> {
        let r = pool.apply(change, at).cloned();
        if r.is_ok() {
            states.push(state(pool));
        }
        r
    };
    take(&mut pool, submit(wants_everything("blocker", 1)), 0.0).unwrap();
    take(&mut pool, submit(wants_everything("old", 1)), 0.0).unwrap();
    // `frames` fills a's card of 16 GiB, the only one with room for it.
    take(&mut pool, submit(local("frames", 16)), 1.0).unwrap();
    assert_eq!(
        pool.arrangement("frames").unwrap().places,
        [Place::On(device("a", 0))]
    );
    take(&mut pool, submit(one_part("small", 2)), 2.0).unwrap();
    // Refused, and not logged: a second job of one name, and a time that is not finite.
    assert_eq!(
        take(&mut pool, submit(one_part("small", 1)), 3.0).unwrap_err(),
        ChangeError::Duplicate {
            job: "small".into()
        }
    );
    assert_eq!(
        take(&mut pool, end("small"), f64::NAN).unwrap_err(),
        ChangeError::Time
    );
    take(&mut pool, submit(wants_everything("new", 2)), 300.0).unwrap();
    // The blocker ends: `old` has waited 300 s and weighs 6 against `new`'s 2.
    take(&mut pool, end("blocker"), 300.0).unwrap();
    assert!(matches!(
        pool.arrangement("old").unwrap().places[0],
        Place::Copies(_)
    ));
    assert_eq!(pool.arrangement("new").unwrap().places, [Place::Waiting]);
    // A member joins remote with a large card: `new` takes it.
    let r = take(
        &mut pool,
        Change::Join {
            member: Some(member("d", true)),
            offer: offer(
                device("d", 0),
                Backend::Cuda,
                80,
                &[("u", 0.005), ("r", 0.005)],
            ),
        },
        310.0,
    )
    .unwrap();
    assert!(r.chosen.is_some());
    // c reports itself remote with a second device.
    let r = take(
        &mut pool,
        Change::Join {
            member: Some(member("c", true)),
            offer: offer(device("c", 1), Backend::Mlx, 24, &[("u", 0.03)]),
        },
        320.0,
    )
    .unwrap();
    assert_eq!(r.marked_remote, ["c"]);
    // a's card leaves: `frames` reads local-only data, b's card has no 16 GiB free, and c and d
    // are remote, so it waits.
    take(
        &mut pool,
        Change::Leave {
            device: device("a", 0),
        },
        330.0,
    )
    .unwrap();
    assert_eq!(pool.arrangement("frames").unwrap().places, [Place::Waiting]);
    take(&mut pool, end("small"), 340.0).unwrap();
    // a comes back with another card, and `frames` is placed on it.
    take(
        &mut pool,
        Change::Join {
            member: None,
            offer: offer(device("a", 1), Backend::Cuda, 16, &[("u", 0.01)]),
        },
        400.0,
    )
    .unwrap();
    assert_eq!(
        pool.arrangement("frames").unwrap().places,
        [Place::On(device("a", 1))]
    );
    for (i, gib) in [3, 5, 7, 9, 11, 13].into_iter().enumerate() {
        let at = 410.0 + 10.0 * i as f64;
        take(&mut pool, submit(one_part(&format!("j{i}"), gib)), at).unwrap();
    }
    take(&mut pool, end("old"), 480.0).unwrap();
    take(&mut pool, end("j1"), 490.0).unwrap();
    take(&mut pool, end("new"), 500.0).unwrap();
    take(&mut pool, end("frames"), 510.0).unwrap();
    take(&mut pool, submit(local("frames again", 4)), 520.0).unwrap();
    assert_eq!(pool.ledger().entries.len(), 22);
    assert_eq!(states.len(), 23);
    (pool, states)
}

/// The ledger with only its first `n` entries.
fn prefix(ledger: &Ledger, n: usize) -> Ledger {
    Ledger {
        at: ledger.at,
        members: ledger.members.clone(),
        offers: ledger.offers.clone(),
        entries: ledger.entries[..n].to_vec(),
    }
}

#[test]
fn replaying_every_prefix_of_a_run_gives_back_the_pool_as_it_was_then() {
    let (pool, states) = scripted();
    let ledger = pool.ledger();
    for (n, then) in states.iter().enumerate() {
        let part = prefix(ledger, n);
        let back = Pool::replay(&part).unwrap();
        assert_eq!(&state(&back), then, "after {n} entries");
        assert_eq!(back.ledger(), &part);
    }
}

#[test]
fn every_prefix_written_as_lines_reads_back_and_replays() {
    let (pool, states) = scripted();
    let ledger = pool.ledger();
    // The lines a host appends one by one are the whole ledger's lines.
    let mut appended = ledger.header_line();
    for e in &ledger.entries {
        appended.push_str(&Ledger::line(e));
    }
    assert_eq!(appended, ledger.to_lines());
    for (n, then) in states.iter().enumerate() {
        let part = prefix(ledger, n);
        let read = Ledger::from_lines(&part.to_lines()).unwrap();
        assert_eq!(read.dropped, None);
        assert_eq!(read.ledger, part, "after {n} entries");
        assert_eq!(&state(&Pool::replay(&read.ledger).unwrap()), then);
    }
}

#[test]
fn a_logged_record_altered_is_found_at_its_entry() {
    let (pool, _) = scripted();
    let ledger = pool.ledger();

    // A score changed in its last bit.
    let (at, c) = ledger
        .entries
        .iter()
        .enumerate()
        .find_map(|(i, e)| {
            e.record
                .candidates
                .iter()
                .position(|c| c.score.is_some())
                .map(|c| (i, c))
        })
        .unwrap();
    let mut altered = ledger.clone();
    let score = altered.entries[at].record.candidates[c]
        .score
        .as_mut()
        .unwrap();
    *score = f64::from_bits(score.to_bits() + 1);
    assert_eq!(
        Pool::replay(&altered).unwrap_err(),
        ReplayError::Differs { entry: at }
    );

    // A choice changed, later in the run.
    let at = ledger
        .entries
        .iter()
        .rposition(|e| e.record.chosen.is_some() && e.record.candidates.len() > 1)
        .unwrap();
    let mut altered = ledger.clone();
    let chosen = altered.entries[at].record.chosen.as_mut().unwrap();
    *chosen = if *chosen == 0 { 1 } else { 0 };
    assert_eq!(
        Pool::replay(&altered).unwrap_err(),
        ReplayError::Differs { entry: at }
    );

    // A change the pool now refuses: a job submitted twice.
    let mut altered = ledger.clone();
    let first = altered.entries[0].clone();
    altered.entries.insert(1, first);
    assert_eq!(
        Pool::replay(&altered).unwrap_err(),
        ReplayError::Refused {
            entry: 1,
            error: ChangeError::Duplicate {
                job: "blocker".into()
            }
        }
    );
}

#[test]
fn a_last_line_cut_short_is_dropped_and_said_so_and_one_before_it_is_refused() {
    let (pool, _) = scripted();
    let ledger = pool.ledger();
    let text = ledger.to_lines();
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let n = lines.len();

    // The last line cut in half: never acknowledged, so dropped.
    let last = lines[n - 1];
    let cut = format!("{}{}", lines[..n - 1].concat(), &last[..last.len() / 2]);
    let read = Ledger::from_lines(&cut).unwrap();
    assert_eq!(read.dropped, Some(n));
    assert_eq!(read.ledger, prefix(ledger, ledger.entries.len() - 1));

    // Whole but for its newline: the line is not written until its newline is.
    let read = Ledger::from_lines(text.strip_suffix('\n').unwrap()).unwrap();
    assert_eq!(read.dropped, Some(n));
    assert_eq!(read.ledger.entries.len(), ledger.entries.len() - 1);

    // The third line cut short, with the lines after it whole: the ledger is damaged.
    let third = lines[2];
    let torn = format!(
        "{}{}\n{}",
        lines[..2].concat(),
        &third[..third.len() / 2],
        lines[3..].concat()
    );
    assert!(matches!(
        Ledger::from_lines(&torn),
        Err(LedgerError::Torn { line: 3 })
    ));

    // A header cut short, with nothing after it.
    assert!(matches!(
        Ledger::from_lines(&lines[0][..10]),
        Err(LedgerError::Torn { line: 1 })
    ));
}

#[test]
fn every_f64_reads_back_to_the_same_bits() {
    let (pool, _) = scripted();
    let mut ledger = pool.ledger().clone();
    // Values with no short decimal form, and the ends of the range. The first two are ratios a
    // score can be that a reader without exact float parsing reads one bit off.
    let hard = [
        f64::from_bits(0x3fdc_c9f7_2451_7eac),
        f64::from_bits(0x3ff5_8375_9f22_9837),
        1.0 / 3.0,
        0.1 + 0.2,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
        -0.0,
        std::f64::consts::PI * 1e-300,
    ];
    let candidates = ledger
        .entries
        .iter_mut()
        .flat_map(|e| e.record.candidates.iter_mut());
    for (c, v) in candidates.zip(hard.iter().cycle()) {
        c.score = Some(*v);
        c.band = Some(-v);
    }
    ledger.entries[0].at = 1.0 / 7.0;
    let read = Ledger::from_lines(&ledger.to_lines()).unwrap().ledger;

    let bits = |l: &Ledger| -> Vec<u64> {
        let mut out = Vec::new();
        for e in &l.entries {
            out.push(e.at.to_bits());
            for c in &e.record.candidates {
                out.extend(c.score.map(f64::to_bits));
                out.extend(c.band.map(f64::to_bits));
            }
        }
        for o in &l.offers {
            out.extend(o.costs.iter().map(|c| c.seconds_per_step.to_bits()));
        }
        out
    };
    assert!(hard.iter().all(|v| bits(&ledger).contains(&v.to_bits())));
    assert_eq!(bits(&read), bits(&ledger));
}

#[test]
fn a_time_that_is_not_finite_is_refused_and_the_pool_is_unchanged() {
    let mut pool = three_cards();
    pool.apply(submit(one_part("a", 1)), 0.0).unwrap();
    let before = state(&pool);
    for at in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            pool.apply(end("a"), at).unwrap_err(),
            ChangeError::Time,
            "{at}"
        );
    }
    assert_eq!(state(&pool), before);
    assert_eq!(pool.ledger().entries.len(), 1);
    // Every entry logged is the change as given, at its time, with its record.
    let Entry { at, change, record } = &pool.ledger().entries[0];
    assert_eq!(*at, 0.0);
    assert!(matches!(change, Change::Submit { .. }));
    assert_eq!(record.event, Event::Arrival { job: "a".into() });
}
