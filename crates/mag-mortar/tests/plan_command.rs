//! The `mag plan` command over the example pool and graph: its kept output, its exit status, and
//! its reasons for refusing.

#[path = "../examples/mag.rs"]
mod example;

use std::path::{Path, PathBuf};

use mag_by::{Description, Graph, Record, Stamp};
use mag_mortar::{HoldsJobs, Pool};

fn by(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../mag-by")
        .join(path)
}

fn pool_file() -> PathBuf {
    by("pools/example.json")
}

fn graph_file() -> PathBuf {
    by("graphs/example.json")
}

fn plan(pool: &Path, graph: &Path, rest: &[&str]) -> example::Output {
    let mut args = vec![
        "plan".to_string(),
        pool.display().to_string(),
        graph.display().to_string(),
    ];
    args.extend(rest.iter().map(|s| s.to_string()));
    example::run(&args)
}

/// Fails when the example's plan is not the kept one. `MAG_WRITE_PLAN=1` writes it.
#[test]
fn the_example_s_plan_is_the_kept_one() {
    let out = plan(&pool_file(), &graph_file(), &[]);
    assert_eq!((out.code, out.stderr.as_str()), (0, ""));
    let kept = Path::new(env!("CARGO_MANIFEST_DIR")).join("plans/example.txt");
    if std::env::var_os("MAG_WRITE_PLAN").is_some() {
        std::fs::write(&kept, &out.stdout).unwrap();
    }
    assert!(
        std::fs::read_to_string(kept).unwrap_or_default() == out.stdout,
        "plans/example.txt differs; run the tests with MAG_WRITE_PLAN=1 to write it"
    );
}

#[test]
fn the_plan_keeps_the_frames_off_the_outside_card_head_on_cuda_and_the_chain_together() {
    let out = plan(&pool_file(), &graph_file(), &[]).stdout;
    // Every part reads the frames kept at owner: the rented card, the fastest, gets none of them,
    // and the guard is named for each.
    let (placed, kept_off) = out.split_once("Kept off").unwrap();
    assert!(!placed.contains("rented"));
    for part in ["decode", "b0", "head", "count"] {
        let line = format!(
            "the guard: {part} carries data kept at owner, and rented/0's member is trusted only \
             at outside"
        );
        assert!(kept_off.contains(&line), "no guard named for {part}");
    }
    // head is pinned to CUDA: on the workstation's card, kept off the others by its pin.
    assert!(out.contains("  head    on workstation/0\n"));
    assert!(out.contains("head is pinned to a member or backend that server/0 is not"));
    // The chain's parts on one device.
    assert!(out.contains("  b0      on server/0, with chain model\n"));
    assert!(out.contains("  b1      on server/0, with chain model\n"));
}

fn recipe(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/machine-list")
        .join(path)
}

/// Fails when the list-of-machines recipe's plan is not the kept one. `MAG_WRITE_PLAN=1` writes
/// it.
#[test]
fn the_machine_list_recipe_s_plan_is_the_kept_one() {
    let out = plan(&recipe("pool.json"), &graph_file(), &[]);
    assert_eq!((out.code, out.stderr.as_str()), (0, ""));
    let kept = recipe("plan.txt");
    if std::env::var_os("MAG_WRITE_PLAN").is_some() {
        std::fs::write(&kept, &out.stdout).unwrap();
    }
    assert!(
        std::fs::read_to_string(kept).unwrap_or_default() == out.stdout,
        "examples/machine-list/plan.txt differs; run the tests with MAG_WRITE_PLAN=1 to write it"
    );
}

#[test]
fn the_recipe_s_plan_says_it_is_a_guess_and_keeps_each_part_where_it_may_run() {
    let out = plan(&recipe("pool.json"), &graph_file(), &[]).stdout;
    // No cost is written, so every cost is an estimate and the band is the widest.
    assert!(out.contains("band \u{b1}100%"), "{out}");
    // The exact parts on the CPUs, the chain on one card, head on the CUDA card.
    for line in [
        "  decode  copies on server/1, workstation/1\n",
        "  count   on server/1\n",
        "  head    on workstation/0\n",
    ] {
        assert!(out.contains(line), "no line {line:?} in\n{out}");
    }
    let (placed, _) = out.split_once("Kept off").unwrap();
    let chain: Vec<&str> = placed
        .lines()
        .filter(|l| l.contains("with chain model"))
        .collect();
    assert_eq!(chain.len(), 2);
    assert!(
        chain
            .iter()
            .all(|l| l.ends_with("on laptop/0, with chain model"))
    );
}

#[test]
fn the_json_is_the_record_the_pool_plans() {
    let out = plan(&pool_file(), &graph_file(), &["--json"]);
    assert_eq!(out.code, 0);
    let printed: Record = serde_json::from_str(&out.stdout).unwrap();
    let d = Description::from_json(&std::fs::read_to_string(pool_file()).unwrap()).unwrap();
    let g = Graph::from_json(&std::fs::read_to_string(graph_file()).unwrap()).unwrap();
    let planned = Pool::from_description(&d)
        .unwrap()
        .plan(&g, &Stamp::default(), d.at)
        .unwrap();
    assert_eq!(printed, planned);
}

#[test]
fn a_pool_built_from_the_example_describes_itself_as_the_file_says() {
    let text = std::fs::read_to_string(pool_file()).unwrap();
    let pool = Pool::from_description(&Description::from_json(&text).unwrap()).unwrap();
    assert_eq!(pool.describe().to_json(), text);
}

#[test]
fn each_refusal_gives_its_reason_and_its_exit_status() {
    let dir = std::env::temp_dir().join(format!("mag-plan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let write = |name: &str, text: String| {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    };
    let pool_text = std::fs::read_to_string(pool_file()).unwrap();
    let graph_text = std::fs::read_to_string(graph_file()).unwrap();

    // A graph that fails its check: refused, 1.
    let bad = write(
        "bad.json",
        graph_text.replace("\"priority\": 2", "\"priority\": 0"),
    );
    let out = plan(&pool_file(), &bad, &[]);
    assert_eq!(out.code, 1);
    assert!(
        out.stderr
            .ends_with("the graph is refused: the job's priority is 0; it must be at least 1\n")
    );

    // A pool file of another format: refused, 1.
    let other = write(
        "other.json",
        pool_text.replace("\"mag-pool\"", "\"mag-graph\""),
    );
    let out = plan(&other, &graph_file(), &[]);
    assert_eq!(out.code, 1);
    assert!(
        out.stderr
            .ends_with("the format is \"mag-graph\", not \"mag-pool\"\n")
    );

    // A file that is not there: 2, with the system's reason.
    let out = plan(&dir.join("missing.json"), &graph_file(), &[]);
    assert_eq!(out.code, 2);
    assert!(out.stderr.contains("missing.json: "), "{}", out.stderr);

    // Usage: 2.
    assert_eq!(example::run(&["plan".to_string()]).code, 2);
    assert_eq!(plan(&pool_file(), &graph_file(), &["--yaml"]).code, 2);

    // A job no device can hold: it would wait, 3, with what removed every candidate.
    let huge = write(
        "huge.json",
        graph_text.replace(
            "\"memory_bytes\": 2147483648",
            "\"memory_bytes\": 2199023255552",
        ),
    );
    let out = plan(&pool_file(), &huge, &[]);
    assert_eq!(out.code, 3);
    assert!(out.stdout.starts_with("label-frames would wait"));
    assert!(out.stdout.contains("has not the memory"));

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_description_of_a_pool_that_holds_jobs_does_not_begin_a_pool() {
    let text = std::fs::read_to_string(pool_file()).unwrap();
    let d = Description::from_json(&text).unwrap();
    let mut pool = Pool::from_description(&d).unwrap();
    let g = Graph::from_json(&std::fs::read_to_string(graph_file()).unwrap()).unwrap();
    pool.submit(g, Stamp::default(), 0.0).unwrap();
    assert_eq!(
        Pool::from_description(&pool.describe()).unwrap_err(),
        HoldsJobs {
            jobs: vec!["label-frames".into()]
        }
    );
}

#[test]
fn a_pool_begun_from_a_description_keeps_its_time_and_replays_from_it() {
    let text = std::fs::read_to_string(pool_file()).unwrap();
    let mut d = Description::from_json(&text).unwrap();
    d.at = 250.0;
    let mut pool = Pool::from_description(&d).unwrap();
    assert_eq!(pool.describe().at, 250.0);
    assert_eq!(pool.ledger().at, 250.0);
    // A job told of at an earlier time waits from the pool's time, not from its own, and the
    // ledger replays to the same.
    let huge = std::fs::read_to_string(graph_file()).unwrap().replace(
        "\"memory_bytes\": 2147483648",
        "\"memory_bytes\": 2199023255552",
    );
    pool.submit(Graph::from_json(&huge).unwrap(), Stamp::default(), 100.0)
        .unwrap();
    pool.end("nothing", 310.0).unwrap();
    assert_eq!(pool.describe().jobs[0].waited_seconds, 60.0);
    let back = Pool::replay(pool.ledger()).unwrap();
    assert_eq!(back.describe(), pool.describe());
}
