//! The door: a local socket on the coordinator's machine, at the start file's `door`, where a
//! command run there asks the pool to take a job, to end one, or for its status.
//!
//! - **The coordinator serves it** ([`Door::open`]). It is a Unix socket of mode 0600, so that
//!   only the coordinator's user can connect to it. It is bound inside a new directory of mode
//!   0700 beside the door's path, made 0600 there, and only then renamed to the door's path, so
//!   that no one else can reach it at any moment. Beside it, `DOOR.lock` (the door's path and
//!   `.lock`) is held locked while the coordinator serves the door; it is never removed.
//! - **One coordinator at a door.** A door whose lock another process holds, or at whose path a
//!   socket answers, is refused ([`DoorError::Held`]). A socket at the path that no one answers
//!   is a door a coordinator left when it stopped, and is replaced. Anything else at the path is
//!   refused and left as it is ([`DoorError::NotASocket`]): it is not a door.
//! - **One request a connection.** The command writes one [`DoorRequest`] as a line of JSON; the
//!   coordinator answers with one [`DoorAnswer`] as a line of JSON, and closes. A request that
//!   changes the pool is applied to it and written to its ledger, and synced, before the answer
//!   is written. A coordinator that stops before then closes the connection with no answer.
//! - **The command** ([`command`], `mag-mar door`) finds the door by the start file `MAG_START`
//!   names, asks, prints the answer and exits with a status that says what it was. With no
//!   coordinator at the door it fails at once.

use std::fmt;
use std::io;
use std::time::Duration;

use mag_by::{Asks, DoorAnswer, DoorRequest, Graph, Stamp};

/// The longest request line the coordinator reads, in bytes: a graph is a few kilobytes, and
/// even a large one far less than this.
pub const REQUEST_LIMIT: u64 = 64 << 20;

/// How long the coordinator waits for a request's line once a command has connected.
pub const REQUEST_WAIT: Duration = Duration::from_secs(10);

/// How long the command waits for the coordinator's answer once its request is written.
pub const ANSWER_WAIT: Duration = Duration::from_secs(60);

/// Why the coordinator cannot serve its door.
#[derive(Debug)]
pub enum DoorError {
    /// Another coordinator serves this door: it holds the door's lock, or a socket at its path
    /// answers.
    Held,
    /// Something other than a socket is at the door's path. It is left as it is.
    NotASocket,
    /// The door's socket, its lock or the directory it is bound in could not be made.
    Io(io::Error),
}

impl fmt::Display for DoorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DoorError::Held => write!(f, "another coordinator serves this door"),
            DoorError::NotASocket => write!(
                f,
                "something other than a socket is at the door's path, and is left as it is"
            ),
            DoorError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DoorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DoorError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for DoorError {
    fn from(e: io::Error) -> DoorError {
        DoorError::Io(e)
    }
}

/// Why the command had no answer from the door.
#[derive(Debug)]
pub enum AskError {
    /// No coordinator answers at the door: nothing is at its path, a socket is there that no one
    /// answers, or it cannot be reached. Nothing was asked.
    NoCoordinator(io::Error),
    /// The coordinator took the request and closed without answering: it stopped, perhaps on a
    /// ledger it could not write. Whether a change it asked for was taken is for status to say.
    NoAnswer,
    /// The request could not be written, or the answer read: why. As for `NoAnswer`, status
    /// says whether a change was taken.
    Io(io::Error),
    /// The answer is not an answer this command reads.
    Unreadable(serde_json::Error),
}

impl fmt::Display for AskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AskError::NoCoordinator(e) => write!(f, "no coordinator answers at the door ({e})"),
            AskError::NoAnswer => write!(
                f,
                "the coordinator closed the door without answering; ask its status to see whether \
                 the request was taken"
            ),
            AskError::Io(e) => write!(
                f,
                "the door failed while asking ({e}); ask its status to see whether the request \
                 was taken"
            ),
            AskError::Unreadable(e) => write!(f, "the coordinator's answer is not read ({e})"),
        }
    }
}

impl std::error::Error for AskError {}

#[cfg(unix)]
pub use unix::{Door, ask};

#[cfg(unix)]
pub(crate) use unix::serve;

#[cfg(unix)]
mod unix {
    use std::fs::{self, File, OpenOptions};
    use std::io::{self, BufRead, BufReader, Read, Write};
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, OpenOptionsExt};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};
    use std::thread;

    use mag_by::{DoorAnswer, DoorRefusal, DoorRequest};

    use super::{ANSWER_WAIT, AskError, DoorError, REQUEST_LIMIT, REQUEST_WAIT};
    use crate::member::say;

    /// The door, as the coordinator serves it: its socket, bound and of mode 0600, and its lock,
    /// held. Dropped, it removes its socket, if the socket at its path is still its own.
    pub struct Door {
        listener: UnixListener,
        path: PathBuf,
        /// The socket's file, by device and inode, as it was put at the path.
        file: (u64, u64),
        /// Held locked for as long as the door is served; unlocked as it is closed.
        _lock: File,
    }

    fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
        let mut name = path.as_os_str().to_owned();
        name.push(suffix);
        PathBuf::from(name)
    }

    impl Door {
        /// Serves the door at `path`: takes its lock, binds its socket where no one else can
        /// reach it, makes it 0600, and puts it at `path` in place of a door left by a
        /// coordinator that stopped. A door another coordinator serves is refused, and so is
        /// anything at `path` that is not a socket.
        pub fn open(path: &str) -> Result<Door, DoorError> {
            let path = PathBuf::from(path);
            let lock = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .mode(0o600)
                .open(with_suffix(&path, ".lock"))?;
            // SAFETY: flock(2) on a descriptor this function owns.
            if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                let e = io::Error::last_os_error();
                return Err(if e.kind() == io::ErrorKind::WouldBlock {
                    DoorError::Held
                } else {
                    DoorError::Io(e)
                });
            }
            match fs::symlink_metadata(&path) {
                Ok(m) if !m.file_type().is_socket() => return Err(DoorError::NotASocket),
                // A socket that answers is a door served without its lock, such as by a
                // coordinator whose lock file was removed: it is not taken from it.
                Ok(_) if UnixStream::connect(&path).is_ok() => return Err(DoorError::Held),
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
            // A new directory only this user can enter, the socket bound and made 0600 in it,
            // and only then moved to the door's path: no one else can reach it at any moment.
            // The lock is held, so a directory already there was left by a coordinator that
            // stopped while it opened its door.
            let staging = with_suffix(&path, ".new");
            let bound = staging.join("s");
            if fs::symlink_metadata(&staging).is_ok() {
                match fs::remove_file(&bound) {
                    Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
                    _ => fs::remove_dir(&staging)?,
                }
            }
            fs::DirBuilder::new().mode(0o700).create(&staging)?;
            let placed = (|| {
                let listener = UnixListener::bind(&bound)?;
                fs::set_permissions(&bound, fs::Permissions::from_mode(0o600))?;
                fs::rename(&bound, &path)?;
                let m = fs::symlink_metadata(&path)?;
                Ok::<_, io::Error>((listener, (m.dev(), m.ino())))
            })();
            let _ = fs::remove_file(&bound);
            fs::remove_dir(&staging)?;
            let (listener, file) = placed?;
            Ok(Door {
                listener,
                path,
                file,
                _lock: lock,
            })
        }

        /// Where it is.
        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for Door {
        fn drop(&mut self) {
            if let Ok(m) = fs::symlink_metadata(&self.path)
                && (m.dev(), m.ino()) == self.file
            {
                let _ = fs::remove_file(&self.path);
            }
        }
    }

    /// Answers the door on threads of its own, each connection on one: reads its request, has
    /// `answer` answer it, and writes the answer. `answer` returns `None` when the coordinator
    /// has stopped without answering, and the connection is then closed with no answer.
    pub(crate) fn serve<F>(door: &Door, name: String, answer: F) -> io::Result<()>
    where
        F: Fn(DoorRequest) -> Option<DoorAnswer> + Clone + Send + 'static,
    {
        let listener = door.listener.try_clone()?;
        thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let answer = answer.clone();
                        let name = name.clone();
                        thread::spawn(move || {
                            if let Err(e) = answer_one(stream, answer) {
                                say(&name, format_args!("the door: a request failed ({e})"));
                            }
                        });
                    }
                    Err(e) => {
                        say(&name, format_args!("the door takes no connection ({e})"));
                        thread::sleep(std::time::Duration::from_millis(100));
                    }
                }
            }
        });
        Ok(())
    }

    fn answer_one<F>(stream: UnixStream, answer: F) -> io::Result<()>
    where
        F: Fn(DoorRequest) -> Option<DoorAnswer>,
    {
        stream.set_read_timeout(Some(REQUEST_WAIT))?;
        let mut line = Vec::new();
        BufReader::new((&stream).take(REQUEST_LIMIT + 1)).read_until(b'\n', &mut line)?;
        let reply = if line.len() as u64 > REQUEST_LIMIT {
            Some(DoorAnswer::Refused {
                refusal: DoorRefusal::Unreadable {
                    error: format!("the request is longer than {REQUEST_LIMIT} bytes"),
                },
            })
        } else {
            match serde_json::from_slice::<DoorRequest>(&line) {
                Ok(request) => answer(request),
                Err(e) => Some(DoorAnswer::Refused {
                    refusal: DoorRefusal::Unreadable {
                        error: e.to_string(),
                    },
                }),
            }
        };
        let Some(reply) = reply else {
            return Ok(());
        };
        let mut text = serde_json::to_vec(&reply).expect("an answer always serialises");
        text.push(b'\n');
        (&stream).write_all(&text)?;
        // The answer's end, before the socket is closed: closed with bytes of the request still
        // unread, as after a request too long, it would end the command's reading as reset, and
        // the command would lose the answer it had.
        stream.shutdown(std::net::Shutdown::Write)
    }

    /// Asks the coordinator at the door at `path`, and returns its answer. Fails at once when no
    /// coordinator answers there, and waits at most [`ANSWER_WAIT`] for the answer.
    pub fn ask(path: &str, request: &DoorRequest) -> Result<DoorAnswer, AskError> {
        let mut stream = UnixStream::connect(path).map_err(AskError::NoCoordinator)?;
        let mut line = serde_json::to_vec(request).expect("a request always serialises");
        line.push(b'\n');
        stream.write_all(&line).map_err(AskError::Io)?;
        stream
            .shutdown(std::net::Shutdown::Write)
            .map_err(AskError::Io)?;
        stream
            .set_read_timeout(Some(ANSWER_WAIT))
            .map_err(AskError::Io)?;
        let mut text = Vec::new();
        stream.read_to_end(&mut text).map_err(AskError::Io)?;
        if text.is_empty() {
            return Err(AskError::NoAnswer);
        }
        serde_json::from_slice(&text).map_err(AskError::Unreadable)
    }
}

/// The command's exit status: answered, and the job placed, ended, or the status given.
pub const ANSWERED: u8 = 0;
/// The command could not ask: its arguments, `MAG_START`, the start file, or the graph or stamp
/// file.
pub const CANNOT_ASK: u8 = 2;
/// The job was submitted, is in the pool and its ledger, and waits.
pub const WAITS: u8 = 3;
/// The door refused the request; nothing was written.
pub const REFUSED: u8 = 4;
/// No coordinator answered: none at the door, or it closed without answering.
pub const NO_ANSWER: u8 = 5;

const USAGE: &str = "usage: mag-mar door submit GRAPH.json [--stamp STAMP.json]\n       \
                     mag-mar door end JOB\n       mag-mar door status";

/// `mag-mar door`, with the arguments after `door`: asks the coordinator at the door of the start
/// file `MAG_START` names, prints its answer as JSON on standard output and a line for a person
/// on standard error, and returns the exit status.
pub fn command(args: &[String]) -> u8 {
    let request = match request(args) {
        Ok(request) => request,
        Err(e) => {
            eprintln!("mag-mar door: {e}");
            return CANNOT_ASK;
        }
    };
    let start = match crate::member::read_start() {
        Ok(start) => start,
        Err(e) => {
            eprintln!("mag-mar door: {e}");
            return CANNOT_ASK;
        }
    };
    #[cfg(unix)]
    let answer = ask(&start.door, &request);
    #[cfg(not(unix))]
    let answer: Result<DoorAnswer, AskError> = Err(AskError::NoCoordinator(io::Error::new(
        io::ErrorKind::Unsupported,
        "the door is a Unix socket",
    )));
    let answer = match answer {
        Ok(answer) => answer,
        Err(e) => {
            eprintln!("mag-mar door: {}: {e}", start.door);
            return NO_ANSWER;
        }
    };
    let mut text = serde_json::to_string_pretty(&answer).expect("an answer always serialises");
    text.push('\n');
    print!("{text}");
    let (code, said) = match &answer {
        DoorAnswer::Submitted { arrangement, .. } => match answer.placed() {
            Some(true) => (ANSWERED, format!("`{}` is placed", arrangement.job)),
            _ => (WAITS, format!("`{}` waits", arrangement.job)),
        },
        DoorAnswer::Ended { .. } => (ANSWERED, "the job has ended".into()),
        DoorAnswer::Status { pool, asks } => (
            ANSWERED,
            format!(
                "{} members, {} devices, {} jobs{}",
                pool.members.len(),
                pool.offers.len(),
                pool.jobs.len(),
                asks.as_ref().map(asked).unwrap_or_default()
            ),
        ),
        DoorAnswer::Refused { refusal } => (REFUSED, format!("refused: {refusal}")),
    };
    eprintln!("mag-mar door: {said}");
    code
}

/// What the pool asks of its host, for a person: the kind of device it wants, as JSON, and the
/// members it would give back.
fn asked(asks: &Asks) -> String {
    let wants = match &asks.wants {
        Some(kind) => format!(
            "wants a device {}",
            serde_json::to_string(kind).expect("a kind always serialises")
        ),
        None => "wants no device".into(),
    };
    let back = if asks.gives_back.is_empty() {
        "would give back no member".into()
    } else {
        format!("would give back {}", asks.gives_back.join(", "))
    };
    format!("; {wants}; {back}")
}

/// The request the arguments ask, its files read.
fn request(args: &[String]) -> Result<DoorRequest, String> {
    let read = |path: &str| std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"));
    match args {
        [status] if status == "status" => Ok(DoorRequest::Status),
        [end, job] if end == "end" => Ok(DoorRequest::End { job: job.clone() }),
        [submit, graph, rest @ ..] if submit == "submit" => {
            let stamp = match rest {
                [] => Stamp::default(),
                [flag, path] if flag == "--stamp" => serde_json::from_str(&read(path)?)
                    .map_err(|e| format!("{path}: not a stamp ({e})"))?,
                _ => return Err(USAGE.into()),
            };
            let graph = Graph::from_json(&read(graph)?).map_err(|e| format!("{graph}: {e}"))?;
            Ok(DoorRequest::Submit { graph, stamp })
        }
        _ => Err(USAGE.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn the_arguments_name_one_request_and_anything_else_is_its_usage() {
        assert_eq!(request(&args(&["status"])), Ok(DoorRequest::Status));
        assert_eq!(
            request(&args(&["end", "tally"])),
            Ok(DoorRequest::End {
                job: "tally".into()
            })
        );
        for bad in [
            &[][..],
            &["status", "now"],
            &["end"],
            &["end", "a", "b"],
            &["submit"],
            &["stop"],
        ] {
            assert_eq!(request(&args(bad)), Err(USAGE.into()), "{bad:?}");
        }
        let graph = concat!(env!("CARGO_MANIFEST_DIR"), "/../mag-by/graphs/example.json");
        match request(&args(&["submit", graph])) {
            Ok(DoorRequest::Submit { graph, stamp }) => {
                assert_eq!(graph.job.name, "label-frames");
                assert_eq!(stamp, Stamp::default());
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            request(&args(&["submit", graph, "--stamp"])),
            Err(USAGE.into())
        );
        assert_eq!(
            request(&args(&["submit", graph, "--ceiling", graph])),
            Err(USAGE.into())
        );
        // A stamp file must be a stamp.
        let refused = request(&args(&["submit", graph, "--stamp", graph])).unwrap_err();
        assert!(refused.contains("not a stamp"), "{refused}");
    }
}
