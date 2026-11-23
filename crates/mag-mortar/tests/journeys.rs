//! M-1's placement lines, on pools of offers written by hand: each test names the journey or the
//! rule it checks.

mod common;

use common::*;
use mag_by::{Backend, Mark, Pin, Place, Removal};
use mag_mortar::Pool;

/// A fast CUDA card, a slower ROCm card and a slower Apple GPU, all on members of the house.
fn three_cards() -> Pool {
    Pool::new(
        vec![member("a", false), member("b", false), member("c", false)],
        vec![
            offer(
                device("a", 0),
                Backend::Cuda,
                16,
                &[("u", 0.010), ("r", 0.010)],
            ),
            offer(
                device("b", 0),
                Backend::Rocm,
                32,
                &[("u", 0.020), ("r", 0.015)],
            ),
            offer(
                device("c", 0),
                Backend::Mlx,
                24,
                &[("u", 0.030), ("r", 0.030)],
            ),
        ],
    )
}

#[test]
fn j1_an_atomic_job_lands_on_the_fastest_device_that_holds_it() {
    let mut pool = three_cards();
    let g = job(
        "small",
        1,
        vec![],
        vec![part("p", "u", Mark::Atomic, 8)],
        vec![],
    );
    pool.submit(g, 0.0).unwrap();
    assert_eq!(
        pool.arrangement("small").unwrap().places,
        [Place::On(device("a", 0))]
    );
    // Too large for the fastest card: the next fastest that holds it.
    let g = job(
        "large",
        1,
        vec![],
        vec![part("p", "u", Mark::Atomic, 20)],
        vec![],
    );
    pool.submit(g, 1.0).unwrap();
    assert_eq!(
        pool.arrangement("large").unwrap().places,
        [Place::On(device("b", 0))]
    );
}

#[test]
fn j1_a_replicable_job_runs_a_copy_on_every_device_that_holds_it() {
    let mut pool = three_cards();
    let g = job(
        "map",
        1,
        vec![input("items", false)],
        vec![part("r", "r", Mark::Replicable, 12)],
        vec![edge("items", Some("r"))],
    );
    pool.submit(g, 0.0).unwrap();
    assert_eq!(
        pool.arrangement("map").unwrap().places,
        [Place::Copies(vec![
            device("a", 0),
            device("b", 0),
            device("c", 0)
        ])]
    );
    // A copy needing more than the first card holds runs on the other two.
    let mut pool = three_cards();
    let g = job(
        "map",
        1,
        vec![input("items", false)],
        vec![part("r", "r", Mark::Replicable, 20)],
        vec![edge("items", Some("r"))],
    );
    pool.submit(g, 0.0).unwrap();
    assert_eq!(
        pool.arrangement("map").unwrap().places,
        [Place::Copies(vec![device("b", 0), device("c", 0)])]
    );
}

#[test]
fn j13_a_job_pinned_to_a_backend_no_device_offers_waits_until_one_joins() {
    let mut pool = Pool::new(
        vec![member("a", false)],
        vec![offer(device("a", 0), Backend::Cuda, 16, &[("u", 0.01)])],
    );
    let g = job(
        "pinned",
        1,
        vec![],
        vec![pinned(
            part("p", "u", Mark::Atomic, 4),
            Pin::Backend(Backend::Mlx),
        )],
        vec![],
    );
    let r = pool.submit(g, 0.0).unwrap();
    assert_eq!(r.chosen, None);
    assert_eq!(
        r.candidates[1].removed,
        Some(Removal::Pin {
            part: "p".into(),
            device: device("a", 0)
        })
    );
    assert_eq!(pool.arrangement("pinned").unwrap().places, [Place::Waiting]);
    let r = pool.join(
        Some(member("c", false)),
        offer(device("c", 0), Backend::Mlx, 24, &[("u", 0.03)]),
        5.0,
    );
    assert!(r.chosen.is_some());
    assert_eq!(
        pool.arrangement("pinned").unwrap().places,
        [Place::On(device("c", 0))]
    );
}

/// A job that wants every device: a copy on each, too large for two to share one.
fn wants_everything(name: &str, priority: u32) -> mag_by::Graph {
    job(
        name,
        priority,
        vec![input("items", false)],
        vec![part("r", "r", Mark::Replicable, 12)],
        vec![edge("items", Some("r"))],
    )
}

#[test]
fn j14_of_two_jobs_that_each_want_every_device_the_higher_priority_runs_first() {
    let mut pool = three_cards();
    pool.submit(wants_everything("blocker", 1), 0.0).unwrap();
    pool.submit(wants_everything("low", 1), 1.0).unwrap();
    pool.submit(wants_everything("high", 2), 2.0).unwrap();
    assert!(matches!(
        pool.arrangement("low").unwrap().places[0],
        Place::Waiting
    ));
    assert!(matches!(
        pool.arrangement("high").unwrap().places[0],
        Place::Waiting
    ));
    // The blocker ends: both wait, and the higher priority takes the devices.
    pool.end("blocker", 10.0);
    assert!(matches!(
        pool.arrangement("high").unwrap().places[0],
        Place::Copies(_)
    ));
    assert!(matches!(
        pool.arrangement("low").unwrap().places[0],
        Place::Waiting
    ));
    // It ends: the other starts at once, in the same run of the loop.
    let r = pool.end("high", 100.0);
    assert!(r.chosen.is_some());
    assert!(matches!(
        pool.arrangement("low").unwrap().places[0],
        Place::Copies(_)
    ));
}

#[test]
fn j14_a_waiting_job_s_priority_grows_until_it_passes_a_newer_higher_one() {
    let mut pool = three_cards();
    pool.submit(wants_everything("blocker", 1), 0.0).unwrap();
    pool.submit(wants_everything("old", 1), 0.0).unwrap();
    // Priority 2 but newly arrived, against priority 1 that has waited 300 s: 1 x (1 + 300/60).
    pool.submit(wants_everything("new", 2), 300.0).unwrap();
    pool.end("blocker", 300.0);
    assert!(matches!(
        pool.arrangement("old").unwrap().places[0],
        Place::Copies(_)
    ));
    assert!(matches!(
        pool.arrangement("new").unwrap().places[0],
        Place::Waiting
    ));
}

/// A local card, slower, and a rented one, faster, on a member marked remote.
fn house_and_rented() -> Pool {
    Pool::new(
        vec![member("house", false), member("rented", true)],
        vec![
            offer(
                device("house", 0),
                Backend::Cuda,
                16,
                &[("u", 0.05), ("c", 0.05)],
            ),
            offer(
                device("rented", 0),
                Backend::Cuda,
                80,
                &[("u", 0.01), ("c", 0.01)],
            ),
        ],
    )
}

#[test]
fn the_guard_keeps_local_only_data_off_a_remote_member_however_fast() {
    let mut pool = house_and_rented();
    let g = job(
        "label",
        1,
        vec![input("frames", true)],
        vec![
            part("detect", "u", Mark::Atomic, 4),
            part("count", "c", Mark::Atomic, 1),
        ],
        vec![
            edge("frames", Some("detect")),
            edge("detect", Some("count")),
            edge("count", None),
        ],
    );
    let guarded = pool.submit(g, 0.0).unwrap().candidates.iter().any(|c| {
        matches!(
            &c.removed,
            Some(Removal::Guard { part, device: d }) if part == "detect" && d == &device("rented", 0)
        )
    });
    assert!(guarded);
    let a = pool.arrangement("label").unwrap();
    assert_eq!(
        a.places,
        [Place::On(device("house", 0)), Place::On(device("house", 0))]
    );
}

#[test]
fn the_guard_holds_for_a_part_whose_weights_are_local_only() {
    let mut pool = house_and_rented();
    let mut adapter = part("adapter", "u", Mark::Atomic, 4);
    adapter.weights.push(local_weights());
    let g = job(
        "serve",
        1,
        vec![input("prompts", false)],
        vec![adapter],
        vec![edge("prompts", Some("adapter"))],
    );
    pool.submit(g, 0.0).unwrap();
    assert_eq!(
        pool.arrangement("serve").unwrap().places,
        [Place::On(device("house", 0))]
    );
}

#[test]
fn a_released_output_s_consumer_may_go_to_the_remote_member() {
    let mut pool = house_and_rented();
    let mut released = edge("detect", Some("count"));
    released.release = true;
    let g = job(
        "label",
        1,
        vec![input("frames", true)],
        vec![
            part("detect", "u", Mark::Atomic, 4),
            part("count", "c", Mark::Atomic, 1),
        ],
        vec![
            edge("frames", Some("detect")),
            released,
            edge("count", None),
        ],
    );
    pool.submit(g, 0.0).unwrap();
    assert_eq!(
        pool.arrangement("label").unwrap().places,
        [
            Place::On(device("house", 0)),
            Place::On(device("rented", 0))
        ]
    );
}

#[test]
fn work_others_wait_on_lands_on_the_fastest_device() {
    // Two parts fit a card. The hub feeds the other two; on the fast card it is a third of the
    // time it is on the slow one.
    let mut pool = Pool::new(
        vec![member("a", false), member("b", false)],
        vec![
            offer(
                device("a", 0),
                Backend::Cuda,
                16,
                &[("hub", 0.01), ("x", 0.05), ("y", 0.05)],
            ),
            offer(
                device("b", 0),
                Backend::Rocm,
                16,
                &[("hub", 0.03), ("x", 0.05), ("y", 0.05)],
            ),
        ],
    );
    let g = job(
        "fan",
        1,
        vec![],
        vec![
            part("hub", "hub", Mark::Atomic, 6),
            part("x", "x", Mark::Atomic, 6),
            part("y", "y", Mark::Atomic, 6),
        ],
        vec![edge("hub", Some("x")), edge("hub", Some("y"))],
    );
    pool.submit(g, 0.0).unwrap();
    assert_eq!(
        pool.arrangement("fan").unwrap().places[0],
        Place::On(device("a", 0))
    );
}

#[test]
fn a_model_that_takes_memory_lands_where_its_measured_rate_is_highest() {
    // Measured at the batch each card's memory allows: the larger card runs the large model
    // faster, though the smaller card is faster at everything else.
    let mut pool = Pool::new(
        vec![member("a", false), member("b", false)],
        vec![
            offer(
                device("a", 0),
                Backend::Cuda,
                16,
                &[("model", 0.040), ("u", 0.01)],
            ),
            offer(
                device("b", 0),
                Backend::Rocm,
                32,
                &[("model", 0.020), ("u", 0.02)],
            ),
        ],
    );
    let g = job(
        "big",
        1,
        vec![],
        vec![part("m", "model", Mark::Atomic, 14)],
        vec![],
    );
    pool.submit(g, 0.0).unwrap();
    assert_eq!(
        pool.arrangement("big").unwrap().places,
        [Place::On(device("b", 0))]
    );
}

#[test]
fn every_removed_candidate_has_a_reason_that_names_what_broke() {
    let mut pool = house_and_rented();
    let g = job(
        "label",
        1,
        vec![input("frames", true)],
        vec![part("detect", "u", Mark::Atomic, 40)],
        vec![edge("frames", Some("detect"))],
    );
    let r = pool.submit(g, 0.0).unwrap();
    // Waiting, the house's card (too small), the rented card (remote).
    assert_eq!(r.candidates.len(), 3);
    assert_eq!(r.candidates[0].removed, None);
    assert_eq!(
        r.candidates[1].removed,
        Some(Removal::Memory {
            device: device("house", 0)
        })
    );
    assert_eq!(
        r.candidates[2].removed,
        Some(Removal::Guard {
            part: "detect".into(),
            device: device("rented", 0)
        })
    );
    assert_eq!(r.chosen, None);
}
