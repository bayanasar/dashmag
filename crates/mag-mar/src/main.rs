//! `mag-mar`: the program a host starts on each member of a pool, with `MAG_START` naming its
//! start file. It joins the pool, and on SIGTERM gives its devices back and exits 0 within its
//! `give_back_seconds`. What it says goes to standard error, a line at a time.
//!
//! Exit status: 0 once its lease has ended; 2 when it cannot start: `MAG_START` unset, a start
//! file that cannot be read or is refused, credentials refused, an address it cannot bind, an
//! argument it does not take, or, on the coordinator, a door another coordinator serves or a
//! ledger it cannot begin; 3 when the coordinator stopped because a line of its ledger could not
//! be written.
//!
//! `mag-mar door submit GRAPH.json [--stamp STAMP.json] | end JOB | status`, run on the
//! coordinator's machine, asks the coordinator at the door of the start file `MAG_START` names,
//! and prints its answer as JSON (`mag_mar::door::command`). Exit status: 0 answered (the job
//! placed, ended, or the status given); 2 it could not ask; 3 the job is submitted and waits; 4
//! the request is refused; 5 no coordinator answered.

use std::process::ExitCode;

use mag_mar::door;
use mag_mar::member::{self, Notice};

const USAGE: &str = "usage: mag-mar, with MAG_START naming the member's start file\n       \
                     mag-mar door submit GRAPH.json [--stamp STAMP.json] | end JOB | status";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        None => {}
        Some((first, rest)) if first == "door" => return ExitCode::from(door::command(rest)),
        Some(_) => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    }
    // First, so that SIGTERM never finds the program without its handler.
    let notice = match Notice::sigterm() {
        Ok(notice) => notice,
        Err(e) => {
            eprintln!("mag-mar: cannot take SIGTERM: {e}");
            return ExitCode::from(2);
        }
    };
    let failed = |e: member::MemberError| {
        eprintln!("mag-mar: {e}");
        match e {
            member::MemberError::LedgerUnwritten { .. } => ExitCode::from(3),
            _ => ExitCode::from(2),
        }
    };
    let start = match member::read_start() {
        Ok(start) => start,
        Err(e) => return failed(e),
    };
    let offers = member::offers(&start);
    match member::run(&start, offers, notice) {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => failed(e),
    }
}
