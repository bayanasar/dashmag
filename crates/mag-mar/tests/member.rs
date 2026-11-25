//! The member program, run as the host runs it: a process of its own with `MAG_START` naming its
//! start file, ended by SIGTERM. A stand-in coordinator, in place of the coordinator's own
//! program, takes its control channel and records what it says.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use mag_by::{
    Backend, CONTROL, Control, Description, DeviceId, Dtype, Member, Offer, Peer, Start, Tolerance,
    Trust,
};
use mag_mar::issue::Authority;
use mag_mar::{ChannelEnd, Credentials, Link, Receiver, Sender, TlsLink};

/// The members' time to give their devices back, in seconds.
const GIVE_BACK: f64 = 2.0;

/// A pool's files in a directory of their own: the authority, and each member's certificate and
/// key, the key readable by its owner alone.
struct Pool {
    dir: PathBuf,
    authority: Authority,
}

impl Pool {
    fn new(test: &str) -> Pool {
        let dir =
            std::env::temp_dir().join(format!("mag-mar-member-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Pool {
            dir,
            authority: Authority::new(1).unwrap(),
        }
    }

    /// Issues `member` a certificate, and writes it and its key; their paths.
    fn issue(&self, member: &str, key_mode: u32) -> (String, String) {
        let issued = self.authority.issue(member, 1).unwrap();
        let (cert, key) = (
            self.dir.join(format!("{member}.pem")),
            self.dir.join(format!("{member}.key")),
        );
        std::fs::write(&cert, &issued.certificate).unwrap();
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, key_mode);
        let mut file = options.open(&key).unwrap();
        file.write_all(issued.key.as_bytes()).unwrap();
        std::fs::set_permissions(&key, std::os::unix::fs::PermissionsExt::from_mode(key_mode))
            .unwrap();
        let path = |p: &Path| p.to_str().unwrap().to_string();
        (path(&cert), path(&key))
    }

    fn credentials(&self, member: &str) -> Credentials {
        let issued = self.authority.issue(member, 1).unwrap();
        Credentials::from_pem(
            member,
            self.authority.certificate(),
            &issued.certificate,
            &issued.key,
        )
        .unwrap()
    }

    /// Writes `member`'s start file: it answers at `listen` if given, and dials the coordinator,
    /// `workstation`, at `coordinator` if given. Its path.
    fn start(&self, member: &str, listen: Option<&str>, coordinator: Option<&str>) -> PathBuf {
        let (certificate_file, key_file) = self.issue(member, 0o600);
        self.start_with(member, listen, coordinator, certificate_file, key_file)
    }

    fn start_with(
        &self,
        member: &str,
        listen: Option<&str>,
        coordinator: Option<&str>,
        certificate_file: String,
        key_file: String,
    ) -> PathBuf {
        let peers = ["workstation", "server", "laptop"]
            .into_iter()
            .filter(|m| *m != member)
            .map(|m| Peer {
                member: m.into(),
                address: (m == "workstation").then(|| coordinator.unwrap().into()),
            })
            .collect();
        let start = Start {
            member: member.into(),
            listen: listen.map(String::from),
            pool: description(),
            authority: self.authority.certificate().into(),
            peers,
            coordinator: "workstation".into(),
            ledger: self.dir.join("ledger.jsonl").to_str().unwrap().into(),
            door: self.dir.join("door.sock").to_str().unwrap().into(),
            idle_seconds: 900.0,
            certificate_file,
            key_file,
        };
        let path = self.dir.join(format!("{member}.json"));
        std::fs::write(&path, start.to_json()).unwrap();
        // What the member reads is what its reader takes.
        Start::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap();
        path
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn member(name: &str, trust: Trust, give_back_seconds: f64) -> Member {
    Member {
        name: name.into(),
        trust,
        give_back_seconds,
    }
}

fn offer(member: &str, index: u32, memory_bytes: u64) -> Offer {
    Offer {
        device: DeviceId {
            member: member.into(),
            index,
        },
        backend: Backend::Flex,
        memory_bytes,
        tolerances: vec![Tolerance::Exact],
        dtypes: vec![Dtype::F32],
        costs: vec![],
    }
}

/// The pool: the coordinator, and two members that give back within [`GIVE_BACK`], each
/// offering its CPU.
fn description() -> Description {
    Description {
        at: 0.0,
        members: vec![
            member("workstation", Trust::Owner, 30.0),
            member("server", Trust::Owner, GIVE_BACK),
            member("laptop", Trust::Site, GIVE_BACK),
        ],
        offers: vec![
            offer("workstation", 0, 8 << 30),
            offer("server", 0, 16 << 30),
            offer("laptop", 0, 4 << 30),
        ],
        jobs: vec![],
        formats: vec![],
    }
}

/// The member program, running, its standard error read a line at a time. Killed if dropped
/// still running.
struct Program {
    child: Child,
    lines: mpsc::Receiver<String>,
    said: Vec<String>,
}

impl Program {
    fn start(start: Option<&Path>) -> Program {
        let mut command = Command::new(env!("CARGO_BIN_EXE_mag-mar"));
        command.env_remove("MAG_START");
        if let Some(start) = start {
            command.env("MAG_START", start);
        }
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { return };
                if tx.send(line).is_err() {
                    return;
                }
            }
        });
        Program {
            child,
            lines,
            said: Vec::new(),
        }
    }

    /// Waits for a line that holds `what`, and returns it.
    fn says(&mut self, what: &str) -> String {
        let limit = Instant::now() + Duration::from_secs(20);
        if let Some(line) = self.said.iter().find(|l| l.contains(what)) {
            return line.clone();
        }
        loop {
            let left = limit.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) => {
                    self.said.push(line.clone());
                    if line.contains(what) {
                        return line;
                    }
                }
                Err(_) => panic!("no line with `{what}`; it said {:#?}", self.said),
            }
        }
    }

    /// Sends SIGTERM, and waits for the program to exit, failing it after `limit`: its status,
    /// and the time from the signal to its exit.
    fn terminate(&mut self, limit: Duration) -> (ExitStatus, Duration) {
        // SAFETY: kill(2) on the child this test started and has not waited for.
        assert_eq!(
            unsafe { libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM) },
            0
        );
        let sent = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                let took = sent.elapsed();
                self.said.extend(self.lines.try_iter());
                // The rest of its lines, once its standard error has closed.
                while let Ok(line) = self.lines.recv_timeout(Duration::from_secs(1)) {
                    self.said.push(line);
                }
                return (status, took);
            }
            if sent.elapsed() > limit {
                let _ = self.child.kill();
                panic!(
                    "still running {limit:?} after SIGTERM; it said {:#?}",
                    self.said
                );
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn said(&self, what: &str) -> bool {
        self.said.iter().any(|l| l.contains(what))
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// What the stand-in coordinator does once the member says it leaves.
#[derive(Clone, Copy, PartialEq)]
enum AfterLeave {
    /// Closes its way of the control channel: the leaving is recorded.
    Answer,
    /// Nothing more: it holds the channel open and says nothing, as a coordinator that stopped
    /// answering does.
    Stall,
}

/// What the stand-in coordinator saw on the control channel.
#[derive(Debug)]
enum Seen {
    /// The channel opened: who opened it, and for which job.
    Opened {
        member: String,
        job: String,
    },
    Word(Control),
}

/// A stand-in for the coordinator's program: `workstation`, answering on loopback, which takes
/// one control channel from one of `from` and reports each word on it. Its address, and what it
/// saw.
fn stand_in(
    pool: &Pool,
    from: &'static [&'static str],
    after: AfterLeave,
) -> (String, mpsc::Receiver<Seen>) {
    let link = TlsLink::bind(pool.credentials("workstation"), "127.0.0.1:0").unwrap();
    let address = link.local_addr().unwrap().to_string();
    let (tx, seen) = mpsc::channel();
    thread::spawn(move || {
        let mut channel = link.accept(from).unwrap();
        let _ = tx.send(Seen::Opened {
            member: channel.peer().member.clone(),
            job: channel.job().into(),
        });
        loop {
            let Ok(word) = channel.recv_word::<Control>() else {
                return;
            };
            let leave = word == Control::Leave;
            let _ = tx.send(Seen::Word(word));
            if leave {
                break;
            }
        }
        match after {
            AfterLeave::Answer => channel.close().unwrap(),
            AfterLeave::Stall => {
                // Held, unanswered, until the test ends.
                thread::sleep(Duration::from_secs(60));
                drop(channel);
            }
        }
        drop(link);
    });
    (address, seen)
}

fn next(seen: &mpsc::Receiver<Seen>) -> Seen {
    seen.recv_timeout(Duration::from_secs(20))
        .expect("the stand-in coordinator saw nothing more")
}

/// The members' time to give their devices back.
fn limit() -> Duration {
    Duration::from_secs_f64(GIVE_BACK)
}

#[test]
fn a_member_binds_joins_with_its_offers_and_leaves_on_sigterm_once_recorded() {
    let pool = Pool::new("joins");
    let (coordinator, seen) = stand_in(&pool, &["server"], AfterLeave::Answer);
    let start = pool.start("server", Some("127.0.0.1:0"), Some(&coordinator));
    let mut program = Program::start(Some(&start));

    // It binds its `listen` address, and answers there.
    let line = program.says("answers at ");
    let at = line.rsplit("answers at ").next().unwrap();
    TcpStream::connect(at).unwrap();

    // It opens a control channel, for no job, as the member its certificate names, and joins:
    // itself as the start file describes it, and its own device's offer, not the others'.
    match next(&seen) {
        Seen::Opened { member, job } => {
            assert_eq!(member, "server");
            assert_eq!(job, CONTROL);
        }
        other => panic!("{other:?}"),
    }
    match next(&seen) {
        Seen::Word(Control::Join { member: m, offers }) => {
            assert_eq!(m, member("server", Trust::Owner, GIVE_BACK));
            assert_eq!(offers, vec![offer("server", 0, 16 << 30)]);
        }
        other => panic!("{other:?}"),
    }
    program.says("joined the pool");

    // SIGTERM: it says it leaves, the coordinator records it and closes, and it exits 0 at once.
    let (status, took) = program.terminate(limit() * 3);
    assert!(matches!(next(&seen), Seen::Word(Control::Leave)));
    assert_eq!(status.code(), Some(0), "{:#?}", program.said);
    assert!(took < limit(), "{took:?}");
    assert!(
        program.said("left: the coordinator recorded it"),
        "{:#?}",
        program.said
    );
    assert!(!program.said("at its deadline"), "{:#?}", program.said);
}

#[test]
fn sigterm_ends_a_member_within_its_give_back_seconds_when_the_coordinator_stops_answering() {
    let pool = Pool::new("stalled");
    let (coordinator, seen) = stand_in(&pool, &["server"], AfterLeave::Stall);
    let start = pool.start("server", Some("127.0.0.1:0"), Some(&coordinator));
    let mut program = Program::start(Some(&start));
    assert!(matches!(next(&seen), Seen::Opened { .. }));
    assert!(matches!(next(&seen), Seen::Word(Control::Join { .. })));
    program.says("joined the pool");

    // The coordinator takes its leaving and never closes: the member waits for it until its
    // deadline, and no longer, and exits 0 within its give-back time.
    let (status, took) = program.terminate(limit() * 3);
    assert!(matches!(next(&seen), Seen::Word(Control::Leave)));
    assert_eq!(status.code(), Some(0), "{:#?}", program.said);
    assert!(took < limit(), "{took:?}");
    assert!(
        took >= limit() * 3 / 4,
        "it did not wait for the coordinator: {took:?}"
    );
    assert!(program.said("at its deadline"), "{:#?}", program.said);
    assert!(!program.said("left: the coordinator recorded it"));
}

#[test]
fn sigterm_ends_a_member_still_opening_its_channel_to_a_coordinator_that_never_answers() {
    let pool = Pool::new("opening");
    // A coordinator that takes the connection and then says nothing: the opening would wait
    // for its own deadline, 10 s, far past the member's give-back time.
    let silent = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = silent.local_addr().unwrap().to_string();
    let start = pool.start("server", None, Some(&address));
    let mut program = Program::start(Some(&start));
    let (_held, _) = silent.accept().unwrap();

    let (status, took) = program.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0), "{:#?}", program.said);
    assert!(took < limit(), "{took:?}");
    assert!(program.said("no one to tell"), "{:#?}", program.said);
}

#[test]
fn a_start_file_without_a_listen_address_dials_only() {
    let pool = Pool::new("dials");
    let (coordinator, seen) = stand_in(&pool, &["laptop"], AfterLeave::Answer);
    let start = pool.start("laptop", None, Some(&coordinator));
    let mut program = Program::start(Some(&start));
    program.says("dials only");
    match next(&seen) {
        Seen::Opened { member, job } => assert_eq!((member.as_str(), job.as_str()), ("laptop", "")),
        other => panic!("{other:?}"),
    }
    match next(&seen) {
        Seen::Word(Control::Join { member: m, offers }) => {
            assert_eq!(m, member("laptop", Trust::Site, GIVE_BACK));
            assert_eq!(offers, vec![offer("laptop", 0, 4 << 30)]);
        }
        other => panic!("{other:?}"),
    }
    program.says("joined the pool");
    let (status, _) = program.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0));
    assert!(!program.said("answers at"), "{:#?}", program.said);
}

#[test]
fn the_coordinators_member_binds_dials_no_one_and_gives_back_on_sigterm() {
    let pool = Pool::new("coordinator");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let mut program = Program::start(Some(&start));
    let line = program.says("answers at ");
    let at = line.rsplit("answers at ").next().unwrap();
    TcpStream::connect(at).unwrap();
    program.says("coordinates");
    let (status, took) = program.terminate(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0));
    assert!(took < Duration::from_secs(1), "{took:?}");
    assert!(!program.said("dial"), "{:#?}", program.said);
}

#[test]
fn a_member_does_not_start_without_its_start_file_or_with_a_key_others_can_read() {
    let mut program = Program::start(None);
    let status = program.child.wait().unwrap();
    assert_eq!(status.code(), Some(2));
    program.says("MAG_START must name a start file");

    let pool = Pool::new("refused");
    let (certificate_file, key_file) = pool.issue("server", 0o644);
    let start = pool.start_with(
        "server",
        None,
        Some("127.0.0.1:9"),
        certificate_file,
        key_file,
    );
    let mut program = Program::start(Some(&start));
    let status = program.child.wait().unwrap();
    assert_eq!(status.code(), Some(2));
    program.says("chmod 600");
}
