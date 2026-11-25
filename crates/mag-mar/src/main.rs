//! `mag-mar`: the program a host starts on each member of a pool, with `MAG_START` naming its
//! start file. It joins the pool, and on SIGTERM gives its devices back and exits 0 within its
//! `give_back_seconds`. What it says goes to standard error, a line at a time.
//!
//! Exit status: 0 once its lease has ended; 2 when it cannot start: `MAG_START` unset, a start
//! file that cannot be read or is refused, credentials refused, an address it cannot bind, or, on
//! the coordinator, a ledger it cannot begin; 3 when the coordinator stopped because a line of
//! its ledger could not be written.

use std::process::ExitCode;

use mag_mar::member::{self, Notice};

fn main() -> ExitCode {
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
