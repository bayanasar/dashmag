//! Files of version 1 mean to the loop what they meant: a ledger version 1 wrote replays to the
//! records it logged, and the example files as version 1 wrote them plan as the kept plan says.

#[path = "../examples/mag.rs"]
mod example;

use std::path::{Path, PathBuf};

use mag_by::Ledger;
use mag_mortar::Pool;

fn v1(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../mag-by/tests/v1")
        .join(name)
}

/// Every record the loop made under version 1, guards and a lowered member among them, is the
/// record it makes now from the same changes read in version 2's words.
#[test]
fn a_ledger_of_version_1_replays_to_the_records_it_logged() {
    let text = std::fs::read_to_string(v1("ledger.jsonl")).unwrap();
    let read = Ledger::from_lines(&text).unwrap();
    assert_eq!(read.version, 1);
    let pool = Pool::replay(&read.ledger).unwrap();
    assert_eq!(pool.ledger(), &read.ledger);
    // Written again whole as version 2, it reads back and replays the same.
    let again = Ledger::from_lines(&pool.ledger().to_lines()).unwrap();
    assert_eq!(again.version, 2);
    assert_eq!(
        Pool::replay(&again.ledger).unwrap().describe(),
        pool.describe()
    );
}

#[test]
fn the_example_files_of_version_1_plan_as_the_kept_plan_says() {
    let args = [
        "plan".to_string(),
        v1("pool.json").display().to_string(),
        v1("graph.json").display().to_string(),
    ];
    let out = example::run(&args);
    assert_eq!((out.code, out.stderr.as_str()), (0, ""));
    let kept = Path::new(env!("CARGO_MANIFEST_DIR")).join("plans/example.txt");
    assert_eq!(out.stdout, std::fs::read_to_string(kept).unwrap());
}
