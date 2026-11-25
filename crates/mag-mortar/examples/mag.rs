//! `mag plan POOL.json GRAPH.json [--json]`: what the pool described in a file would do with a
//! graph, and why, without placing it.
//!
//! ```text
//! cargo run -p mag-mortar --features files --example mag -- plan \
//!     crates/mag-by/pools/example.json crates/mag-by/graphs/example.json
//! ```
//!
//! For a person it prints where each part would go, and for each device a part was kept off, why,
//! in the words of the reason the loop recorded; the score of the arrangement chosen and its band;
//! or that the job would wait, and what removed every candidate that placed it. `--json` prints
//! the loop's record instead. It plans and keeps nothing: nothing runs until a member does.
//!
//! Exit status: 0 the job would be placed, 3 it would wait, 1 a file or the graph is refused, 2
//! usage, or a file that cannot be read.

use std::fmt::Write as _;
use std::io::Write as _;

use mag_by::{Description, Graph, Place, Record, Removal, Stamp};
use mag_mortar::Pool;

/// What the command prints, and its exit status.
pub struct Output {
    pub stdout: String,
    pub stderr: String,
    pub code: i32,
}

const USAGE: &str = "usage: mag plan POOL.json GRAPH.json [--json]\n";

/// The command for its arguments, the program's name left out.
pub fn run(args: &[String]) -> Output {
    let (paths, json): (Vec<&String>, bool) = match args {
        [plan, pool, graph] if plan == "plan" => (vec![pool, graph], false),
        [plan, pool, graph, flag] if plan == "plan" && flag == "--json" => {
            (vec![pool, graph], true)
        }
        _ => return fail(2, USAGE.into()),
    };
    let mut texts = Vec::new();
    for path in &paths {
        match std::fs::read_to_string(path) {
            Ok(text) => texts.push(text),
            Err(e) => return fail(2, format!("{path}: {e}\n")),
        }
    }
    let description = match Description::from_json(&texts[0]) {
        Ok(d) => d,
        Err(e) => return fail(1, format!("{}: {e}\n", paths[0])),
    };
    let pool = match Pool::from_description(&description) {
        Ok(p) => p,
        Err(e) => return fail(1, format!("{}: {e}\n", paths[0])),
    };
    let graph = match Graph::from_json(&texts[1]) {
        Ok(g) => g,
        Err(e) => return fail(1, format!("{}: {e}\n", paths[1])),
    };
    let record = match pool.plan(&graph, &Stamp::default(), description.at) {
        Ok(r) => r,
        Err(e) => return fail(1, format!("{}: {e}\n", paths[1])),
    };
    let code = if record.chosen.is_some() { 0 } else { 3 };
    let stdout = if json {
        let mut text = serde_json::to_string_pretty(&record).expect("a record serialises");
        text.push('\n');
        text
    } else {
        explain(&graph, &record)
    };
    Output {
        stdout,
        stderr: String::new(),
        code,
    }
}

fn fail(code: i32, stderr: String) -> Output {
    Output {
        stdout: String::new(),
        stderr,
        code,
    }
}

/// The record, for a person.
fn explain(graph: &Graph, record: &Record) -> String {
    let job = &graph.job.name;
    let mut out = String::new();
    let kept_off = kept_off(graph, record);
    let all = record.candidates.len();
    let removed = record
        .candidates
        .iter()
        .filter(|c| c.removed.is_some())
        .count();
    match record.chosen {
        Some(c) => {
            let chosen = &record.candidates[c];
            let score = chosen.score.expect("a chosen candidate is scored");
            let band = chosen.band.unwrap_or(0.0);
            writeln!(
                out,
                "{job} would be placed: of {all} candidates, {removed} were removed and {} \
                 scored.\nThe best scores {score:.3}, band ±{:.0}%.",
                all - removed,
                band * 100.0
            )
            .unwrap();
            writeln!(out).unwrap();
            let width = graph.parts.iter().map(|p| p.name.len()).max().unwrap_or(0);
            let places = &chosen.arrangements[0].places;
            for (part, place) in graph.parts.iter().zip(places) {
                let at = match place {
                    Place::Waiting => "nowhere".to_string(),
                    Place::On(d) => format!("on {d}"),
                    Place::Copies(ds) => {
                        let ds: Vec<String> = ds.iter().map(|d| d.to_string()).collect();
                        format!("copies on {}", ds.join(", "))
                    }
                };
                let chain = graph
                    .chains
                    .iter()
                    .find(|c| c.parts.contains(&part.name))
                    .map(|c| format!(", with chain {}", c.name))
                    .unwrap_or_default();
                writeln!(out, "  {:width$}  {at}{chain}", part.name).unwrap();
            }
        }
        None => {
            writeln!(
                out,
                "{job} would wait: of {all} candidates, {removed} were removed.\nNone of the rest \
                 scores above leaving it waiting."
            )
            .unwrap();
        }
    }
    if !kept_off.is_empty() {
        writeln!(out).unwrap();
        writeln!(out, "Kept off a device, and why:").unwrap();
        for reason in kept_off {
            writeln!(out, "  {reason}").unwrap();
        }
    }
    out
}

/// Every reason a part was kept off a device, once each, in the order of the graph's parts and
/// then of the reasons' devices: from the candidates removed, and from the copies the chosen
/// candidate left out.
fn kept_off(graph: &Graph, record: &Record) -> Vec<Removal> {
    let mut reasons: Vec<Removal> = Vec::new();
    let chosen = record.chosen.map(|c| &record.candidates[c]);
    let removed = record.candidates.iter().filter_map(|c| c.removed.clone());
    let left_out = chosen
        .into_iter()
        .flat_map(|c| c.left_out.iter().map(|(_, r)| r.clone()));
    for r in removed.chain(left_out) {
        if !reasons.contains(&r) {
            reasons.push(r);
        }
    }
    let order = |r: &Removal| -> (usize, String) {
        let (part, device) = match r {
            Removal::Tolerance { part, device }
            | Removal::Dtype { part, device }
            | Removal::Pin { part, device }
            | Removal::Guard { part, device, .. } => (Some(part), Some(device)),
            Removal::NoDevice { device }
            | Removal::Memory { device }
            | Removal::Stamp { device, .. } => (None, Some(device)),
            Removal::Need { .. } => (None, None),
        };
        let p = part
            .and_then(|n| graph.parts.iter().position(|p| &p.name == n))
            .unwrap_or(graph.parts.len());
        (p, device.map(|d| d.to_string()).unwrap_or_default())
    };
    reasons.sort_by_key(order);
    reasons
}

// The test of this example includes this file as a module, to call `run`; there `main` is never
// called, hence the allowance.
#[allow(dead_code)]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = run(&args);
    // A reader that stops early, as `head` does, is not an error of the plan's.
    let _ = std::io::stdout().write_all(out.stdout.as_bytes());
    let _ = std::io::stderr().write_all(out.stderr.as_bytes());
    std::process::exit(out.code);
}
