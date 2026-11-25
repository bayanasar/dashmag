//! `certs`: the pool's certificate authority and its members' certificates, as the host makes
//! them for a list of machines.
//!
//!     certs new DIR MEMBER...   a new authority in DIR, and a certificate for each member
//!     certs add DIR MEMBER...   a certificate for each member from the authority in DIR: a
//!                               member that joins, or one whose lease ended and comes back
//!
//! `--days N` before DIR sets how long the members' certificates are valid (365 by default);
//! the authority is valid for ten years. In DIR it writes `authority.pem`, the certificate every
//! start file holds, and `authority.key`, the authority's key, which stays with the host; and
//! for each member `MEMBER.pem` and `MEMBER.key`, its certificate and key, which go to that
//! machine and are named by its start file. Keys are written readable by their owner alone. It
//! prints each member with its certificate's serial.
//!
//! Exit status: 0 when every certificate is written; 2 for a usage error, a member name a
//! certificate cannot carry, an authority that already exists for `new` or is missing for
//! `add`, or a file that cannot be written.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mag_mar::issue::Authority;

const AUTHORITY_DAYS: u32 = 3650;

fn usage() -> ExitCode {
    eprintln!("usage: certs new|add [--days N] DIR MEMBER...");
    ExitCode::from(2)
}

/// Writes `text` to `path`; a key readable by its owner alone.
fn write(path: &Path, text: &str, key: bool) -> Result<(), String> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    if key {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    #[cfg(not(unix))]
    let _ = key;
    let mut file = options
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(text.as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn run(new: bool, days: u32, dir: &Path, members: &[String]) -> Result<(), String> {
    let cert_path = dir.join("authority.pem");
    let key_path = dir.join("authority.key");
    let authority = if new {
        if cert_path.exists() || key_path.exists() {
            return Err(format!(
                "{} already holds an authority; `add` issues from it",
                dir.display()
            ));
        }
        for m in members {
            mag_mar::check_name(m).map_err(|e| e.to_string())?;
        }
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let authority = Authority::new(AUTHORITY_DAYS).map_err(|e| e.to_string())?;
        write(&key_path, &authority.key(), true)?;
        write(&cert_path, authority.certificate(), false)?;
        authority
    } else {
        let read = |p: &PathBuf| fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()));
        Authority::from_pem(&read(&cert_path)?, &read(&key_path)?).map_err(|e| e.to_string())?
    };
    for m in members {
        let issued = authority.issue(m, days).map_err(|e| e.to_string())?;
        write(&dir.join(format!("{m}.key")), &issued.key, true)?;
        write(&dir.join(format!("{m}.pem")), &issued.certificate, false)?;
        let serial: String = issued.serial.iter().map(|b| format!("{b:02x}")).collect();
        println!("{m}  serial {serial}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let new = match args.next().as_deref() {
        Some("new") => true,
        Some("add") => false,
        _ => return usage(),
    };
    let mut rest: Vec<String> = args.collect();
    let mut days = 365;
    if rest.first().map(String::as_str) == Some("--days") {
        match rest.get(1).and_then(|d| d.parse().ok()) {
            Some(d) if d > 0 => days = d,
            _ => return usage(),
        }
        rest.drain(..2);
    }
    let [dir, members @ ..] = rest.as_slice() else {
        return usage();
    };
    if members.is_empty() {
        return usage();
    }
    match run(new, days, Path::new(dir), members) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("certs: {e}");
            ExitCode::from(2)
        }
    }
}
