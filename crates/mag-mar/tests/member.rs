//! The member program, run as the host runs it: a process of its own with `MAG_START` naming its
//! start file, ended by SIGTERM. A member is run first against a stand-in coordinator, which takes
//! its control channel and records what it says, or fails to answer as a test needs; then against
//! the program itself as coordinator, whose ledger the tests read and replay; and last, the
//! coordinator's door, asked by `mag-mar door` as a process of its own.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use mag_by::{
    Asks, Backend, CONTROL, Change, Control, Description, DeviceId, DeviceKind, DoorAnswer,
    DoorRefusal, Dtype, Ledger, Member, Offer, Peer, Place, Record, Start, Tolerance, Trust,
};
use mag_mar::issue::Authority;
use mag_mar::{ChannelEnd, Credentials, Link, LinkError, Receiver, Sender, TlsLink};

/// The members' time to give their devices back, in seconds.
const GIVE_BACK: f64 = 2.0;

/// A time to give back that no load on the machine reaches, for the tests of a leave the
/// coordinator records: within [`GIVE_BACK`] a member under load can end at its deadline before
/// the coordinator's answer comes, which is also right, and the test would see the other ending.
const PATIENT: f64 = 60.0;

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
        self.start_in(member, listen, coordinator, description())
    }

    /// The same, with `pool` as the start file's pool.
    fn start_in(
        &self,
        member: &str,
        listen: Option<&str>,
        coordinator: Option<&str>,
        pool: Description,
    ) -> PathBuf {
        let (certificate_file, key_file) = self.issue(member, 0o600);
        self.start_with(
            member,
            listen,
            coordinator,
            pool,
            certificate_file,
            key_file,
        )
    }

    fn start_with(
        &self,
        member: &str,
        listen: Option<&str>,
        coordinator: Option<&str>,
        pool: Description,
        certificate_file: String,
        key_file: String,
    ) -> PathBuf {
        let peers = pool
            .members
            .iter()
            .map(|m| m.name.as_str())
            .filter(|m| *m != member)
            .map(|m| Peer {
                member: m.into(),
                address: (m == "workstation").then(|| coordinator.unwrap().into()),
            })
            .collect();
        let start = Start {
            member: member.into(),
            listen: listen.map(String::from),
            pool,
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
/// offering its CPU, and the server a second device.
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
            offer("server", 1, 24 << 30),
            offer("laptop", 0, 4 << 30),
        ],
        jobs: vec![],
        formats: vec![],
    }
}

/// The pool with members that give back within [`PATIENT`] in their own start files. The
/// coordinator, on [`description`], holds them to the shorter time, as it takes the shorter
/// of the two.
fn patient() -> Description {
    let mut pool = description();
    for member in &mut pool.members {
        if member.name != "workstation" {
            member.give_back_seconds = PATIENT;
        }
    }
    pool
}

/// An address on loopback at a port below the system's ephemeral ports, free when chosen. A
/// coordinator stopped and started again binds it again: no connection's own port and no bind
/// of port 0 can take it meanwhile, as they can a port from that range under load.
fn steady_address() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALLS: AtomicU64 = AtomicU64::new(0);
    // Linux's range begins at 32768 by default, macOS's at 49152.
    let low: u16 = std::fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range")
        .ok()
        .and_then(|r| r.split_whitespace().next()?.parse().ok())
        .unwrap_or(32768);
    const SPAN: u16 = 10_000;
    assert!(low > SPAN + 1024, "ephemeral ports begin at {low}");
    // A place of this process's own and this call's to begin from, so that copies of these
    // tests run at once seldom try the same ports.
    let seed = u64::from(std::process::id()) * 7919
        + CALLS.fetch_add(1, Ordering::Relaxed) * 104_729
        + u64::from(Instant::now().elapsed().subsec_nanos());
    (0..u64::from(SPAN))
        .map(|i| low - SPAN + ((seed + i) % u64::from(SPAN)) as u16)
        .find_map(|port| TcpListener::bind(("127.0.0.1", port)).ok())
        .expect("no free port below the ephemeral ones")
        .local_addr()
        .unwrap()
        .to_string()
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
        Program::spawn(Program::command(start))
    }

    fn command(start: Option<&Path>) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_mag-mar"));
        command.env_remove("MAG_START");
        if let Some(start) = start {
            command.env("MAG_START", start);
        }
        command
    }

    fn spawn(mut command: Command) -> Program {
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

    /// Its exit status, once it ends by itself within `limit`; killed, and the test failed, if
    /// it has not.
    fn ends_within(&mut self, limit: Duration) -> ExitStatus {
        self.exits(limit).0
    }

    /// Sends SIGTERM, and waits for the program to exit, failing it after `limit`: its status,
    /// and the time from the signal to its exit.
    fn terminate(&mut self, limit: Duration) -> (ExitStatus, Duration) {
        // SAFETY: kill(2) on the child this test started and has not waited for.
        assert_eq!(
            unsafe { libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM) },
            0
        );
        self.exits(limit)
    }

    /// Waits for the program to exit, failing it after `limit`: its status, and the time it took.
    fn exits(&mut self, limit: Duration) -> (ExitStatus, Duration) {
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
                    "still running {limit:?} after SIGTERM, or after it was to end; it said {:#?}",
                    self.said
                );
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn said(&self, what: &str) -> bool {
        self.said.iter().any(|l| l.contains(what))
    }

    /// Waits until it has said a line that holds `what` `times` times.
    fn says_times(&mut self, what: &str, times: usize) {
        let limit = Instant::now() + Duration::from_secs(20);
        while self.said.iter().filter(|l| l.contains(what)).count() < times {
            let left = limit.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) => self.said.push(line),
                Err(_) => panic!("not {times} lines with `{what}`; it said {:#?}", self.said),
            }
        }
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
/// one control channel from one of `from`, reports each word on it, and answers a join with
/// `Joined`. Its address, and what it saw.
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
            let (join, leave) = (matches!(word, Control::Join { .. }), word == Control::Leave);
            let _ = tx.send(Seen::Word(word));
            if join && channel.send_word(&Control::Joined).is_err() {
                return;
            }
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

/// Whether a member said it left, either way that is right: the coordinator recorded its leave,
/// or its deadline came first, which under load it can.
fn left(member: &Program) -> bool {
    member.said("left: the coordinator recorded it") || member.said("at its deadline")
}

fn next(seen: &mpsc::Receiver<Seen>) -> Seen {
    seen.recv_timeout(Duration::from_secs(20))
        .expect("the stand-in coordinator saw nothing more")
}

/// The members' time to give their devices back.
fn limit() -> Duration {
    Duration::from_secs_f64(GIVE_BACK)
}

/// A patient member's time to give its devices back.
fn patiently() -> Duration {
    Duration::from_secs_f64(PATIENT)
}

#[test]
fn a_member_binds_joins_with_its_offers_and_leaves_on_sigterm_once_recorded() {
    let pool = Pool::new("joins");
    let (coordinator, seen) = stand_in(&pool, &["server"], AfterLeave::Answer);
    let start = pool.start_in("server", Some("127.0.0.1:0"), Some(&coordinator), patient());
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
            assert_eq!(m, member("server", Trust::Owner, PATIENT));
            assert_eq!(
                offers,
                vec![offer("server", 0, 16 << 30), offer("server", 1, 24 << 30)]
            );
        }
        other => panic!("{other:?}"),
    }
    program.says("joined the pool");

    // SIGTERM: it says it leaves, the coordinator records it and closes, and it exits 0 then,
    // not at its deadline.
    let (status, took) = program.terminate(patiently());
    assert!(matches!(next(&seen), Seen::Word(Control::Leave)));
    assert_eq!(status.code(), Some(0), "{:#?}", program.said);
    assert!(took < patiently(), "{took:?}");
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
        description(),
        certificate_file,
        key_file,
    );
    let mut program = Program::start(Some(&start));
    let status = program.child.wait().unwrap();
    assert_eq!(status.code(), Some(2));
    program.says("chmod 600");
}

#[test]
fn a_member_does_not_start_with_a_give_back_time_it_cannot_wait() {
    let pool = Pool::new("give-back");
    let start = pool.start("server", None, Some("127.0.0.1:9"));
    let text = std::fs::read_to_string(&start).unwrap();
    // Below 0, the start file's reader refuses it. Longer than a member can wait, the reader
    // takes it and the member refuses it.
    for (seconds, says) in [
        (
            -1.0,
            "the pool: member `server`'s `give_back_seconds` is -1",
        ),
        (
            1e20,
            "`give_back_seconds` is 100000000000000000000; it must be finite, not below 0 and \
             below 2^64",
        ),
    ] {
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        v["pool"]["members"][1]["give_back_seconds"] = seconds.into();
        std::fs::write(&start, v.to_string()).unwrap();
        let mut program = Program::start(Some(&start));
        let status = program.ends_within(Duration::from_secs(10));
        assert_eq!(status.code(), Some(2), "{:#?}", program.said);
        program.says(says);
    }
}

/// The coordinator's ledger, read from its file, every line whole.
fn read_ledger(pool: &Pool) -> (Ledger, String) {
    let text = std::fs::read_to_string(pool.dir.join("ledger.jsonl")).unwrap();
    let read = Ledger::from_lines(&text).unwrap();
    assert_eq!(read.dropped, None, "{text}");
    (read.ledger, text)
}

/// Waits until the coordinator's ledger holds `entries` entries, and returns it.
fn ledger_of(pool: &Pool, entries: usize) -> Ledger {
    let limit = Instant::now() + Duration::from_secs(20);
    loop {
        let path = pool.dir.join("ledger.jsonl");
        if path.exists() {
            let (ledger, text) = read_ledger(pool);
            assert!(ledger.entries.len() <= entries, "{text}");
            if ledger.entries.len() == entries {
                return ledger;
            }
        }
        assert!(
            Instant::now() < limit,
            "the ledger never held {entries} entries"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn device(member: &str, index: u32) -> DeviceId {
    DeviceId {
        member: member.into(),
        index,
    }
}

/// The changes in a ledger, each a join as (its member, its device) or a leave as its device.
fn changes(ledger: &Ledger) -> Vec<(Option<Member>, DeviceId, bool)> {
    ledger
        .entries
        .iter()
        .map(|e| match &e.change {
            Change::Join { member, offer } => (member.clone(), offer.device.clone(), true),
            Change::Leave { device } => (None, device.clone(), false),
            other => panic!("{other:?}"),
        })
        .collect()
}

/// The coordinator, `workstation`, run as the program with `MAG_START`, from `command`; it and
/// the address it answers at.
fn coordinator_from(command: Command) -> (Program, String) {
    let mut program = Program::spawn(command);
    let line = program.says("answers at ");
    let at = line.rsplit("answers at ").next().unwrap().to_string();
    program.says("coordinates");
    (program, at)
}

fn coordinator(pool: &Pool) -> (Program, String) {
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    coordinator_from(Program::command(Some(&start)))
}

/// A member's program dialling the coordinator at `at`, with `pool` as its start file's pool,
/// once it has joined.
fn joined(pool: &Pool, name: &str, at: &str, description: Description) -> Program {
    let start = pool.start_in(name, None, Some(at), description);
    let mut program = Program::start(Some(&start));
    program.says("joined the pool");
    program
}

#[test]
fn the_coordinator_records_its_members_joins_in_its_ledger_one_entry_per_device() {
    let pool = Pool::new("records-joins");
    let (mut coordinating, at) = coordinator(&pool);
    let _server = joined(&pool, "server", &at, description());
    coordinating.says("`server` joined with 2 devices");
    let _laptop = joined(&pool, "laptop", &at, description());
    coordinating.says("`laptop` joined with 1 devices");

    let ledger = ledger_of(&pool, 3);
    // Begun with the coordinator's own member and devices, by this version of mag-mortar.
    assert_eq!(
        ledger.members,
        [member("workstation", Trust::Owner, 30.0)],
        "{ledger:?}"
    );
    assert_eq!(ledger.offers, [offer("workstation", 0, 8 << 30)]);
    assert_eq!(ledger.written_by.as_deref(), Some(mag_mortar::VERSION));
    assert_eq!(
        changes(&ledger),
        [
            (
                Some(member("server", Trust::Owner, GIVE_BACK)),
                device("server", 0),
                true
            ),
            (
                Some(member("server", Trust::Owner, GIVE_BACK)),
                device("server", 1),
                true
            ),
            (
                Some(member("laptop", Trust::Site, GIVE_BACK)),
                device("laptop", 0),
                true
            ),
        ]
    );
    // Each at the pool's time, seconds since 1970, after the header's.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    assert!(ledger.at <= ledger.entries[0].at && ledger.entries[2].at <= now);
    assert!(now - ledger.at < 60.0, "{}", ledger.at);
    let replayed = mag_mortar::Pool::replay(&ledger).unwrap();
    let offered: Vec<DeviceId> = replayed
        .describe()
        .offers
        .into_iter()
        .map(|o| o.device)
        .collect();
    assert_eq!(
        offered,
        [
            device("laptop", 0),
            device("server", 0),
            device("server", 1),
            device("workstation", 0)
        ]
    );
}

#[test]
fn a_member_leaving_on_sigterm_is_recorded_before_it_is_told_and_the_ledger_replays() {
    let pool = Pool::new("records-leave");
    let (mut coordinating, at) = coordinator(&pool);
    let mut server = joined(&pool, "server", &at, patient());
    let _laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 3);

    // SIGTERM: the leave of each of its devices is in the ledger, and the member is told, within
    // its time to give back.
    let (status, took) = server.terminate(patiently());
    assert_eq!(status.code(), Some(0), "{:#?}", server.said);
    assert!(took < patiently(), "{took:?}");
    assert!(
        server.said("left: the coordinator recorded it"),
        "{:#?}",
        server.said
    );
    assert!(!server.said("at its deadline"), "{:#?}", server.said);
    // Told only once written: the file holds the leave as the member exits.
    let (ledger, text) = read_ledger(&pool);
    assert_eq!(ledger.entries.len(), 5, "{text}");
    assert_eq!(
        changes(&ledger)[3..],
        [
            (None, device("server", 0), false),
            (None, device("server", 1), false)
        ]
    );
    coordinating.says("`server` left, its 2 devices recorded");

    // The same pool again from its ledger: the same records, and the same lines.
    let replayed = mag_mortar::Pool::replay(&ledger).unwrap();
    assert_eq!(replayed.ledger().to_lines(), text);
    let offered: Vec<DeviceId> = replayed
        .describe()
        .offers
        .into_iter()
        .map(|o| o.device)
        .collect();
    assert_eq!(offered, [device("laptop", 0), device("workstation", 0)]);

    // The coordinator's own notice: its own device's leave, and it exits 0.
    let (status, _) = coordinating.terminate(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "{:#?}", coordinating.said);
    let (ledger, text) = read_ledger(&pool);
    assert_eq!(
        changes(&ledger)[5..],
        [(None, device("workstation", 0), false)]
    );
    let replayed = mag_mortar::Pool::replay(&ledger).unwrap();
    assert_eq!(replayed.ledger().to_lines(), text);
}

/// Opens a control channel to the coordinator at `at` as `member`, for `job`, and says `word`.
/// A receive on it gives up after 10 s of silence, so that a test fails rather than hangs.
fn say_as(pool: &Pool, member: &str, at: &str, job: &str, word: &Control) -> mag_mar::TlsChannel {
    let mut channel = open_as(pool, member, at, job);
    channel.send_word(word).unwrap();
    channel
}

/// Opens a channel to the coordinator at `at` as `member`, for `job`, and says nothing. A receive
/// on it gives up after 10 s of silence.
fn open_as(pool: &Pool, member: &str, at: &str, job: &str) -> mag_mar::TlsChannel {
    TlsLink::new(pool.credentials(member))
        .unwrap()
        .with_silence(Some(Duration::from_secs(10)))
        .open("workstation", at, job)
        .unwrap()
}

/// The same, as a join the coordinator takes: it says `Joined` next.
fn join_as(pool: &Pool, member: &str, at: &str, word: &Control) -> mag_mar::TlsChannel {
    let mut channel = say_as(pool, member, at, CONTROL, word);
    assert_eq!(channel.recv_word::<Control>().unwrap(), Control::Joined);
    channel
}

/// What the channel hears next: a join refused for now ends it unclosed, so that the member dials
/// again.
fn cut_short(channel: &mut mag_mar::TlsChannel) {
    match channel.recv() {
        Err(LinkError::Truncated) => {}
        other => panic!("{other:?}"),
    }
}

/// What the channel hears next: a join refused for good ends it closed, without `Joined`, so that
/// the member dials no more.
fn closed(channel: &mut mag_mar::TlsChannel) {
    match channel.recv() {
        Err(LinkError::Closed) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_join_whose_names_are_not_its_certificates_or_whose_member_is_held_is_refused() {
    let pool = Pool::new("refused-join");
    let (mut coordinating, at) = coordinator(&pool);
    let join = |name: &str, devices: &[(&str, u32)]| Control::Join {
        member: member(name, Trust::Owner, GIVE_BACK),
        offers: devices.iter().map(|(m, i)| offer(m, *i, 1 << 30)).collect(),
    };

    // Refused for good, each closed without `Joined`. The server's certificate, joining as the
    // laptop.
    let mut c = say_as(
        &pool,
        "server",
        &at,
        CONTROL,
        &join("laptop", &[("laptop", 0)]),
    );
    closed(&mut c);
    coordinating.says("it joins as `laptop`, and its certificate names `server`");
    // As itself, offering the laptop's device beside its own.
    let mut c = say_as(
        &pool,
        "server",
        &at,
        CONTROL,
        &join("server", &[("server", 0), ("laptop", 0)]),
    );
    closed(&mut c);
    coordinating.says("it offers laptop/0");
    // As the coordinator itself.
    let mut c = say_as(
        &pool,
        "workstation",
        &at,
        CONTROL,
        &join("workstation", &[("workstation", 1)]),
    );
    closed(&mut c);
    coordinating.says("`workstation` is the coordinator");
    // On a channel for a job, not the control channel: cut short as it opens, before any join.
    // Nothing is said on it: the coordinator does not read it, and a word sent may meet the cut
    // first and fail.
    let mut c = open_as(&pool, "server", &at, "j");
    cut_short(&mut c);
    coordinating.says("no job is placed yet");
    assert_eq!(read_ledger(&pool).0.entries.len(), 0);

    // As itself, with its own device: taken, and told so.
    let mut held = join_as(&pool, "server", &at, &join("server", &[("server", 0)]));
    assert_eq!(read_ledger(&pool).0.entries.len(), 1);
    // Again, on a second channel while the first is live: the member is held, for now, since
    // the first may be gone without the coordinator knowing yet; cut short, to dial again.
    let mut c = say_as(
        &pool,
        "server",
        &at,
        CONTROL,
        &join("server", &[("server", 1)]),
    );
    cut_short(&mut c);
    coordinating.says("`server` is in the pool already, on another channel; it may dial again");
    // Again, on the same channel.
    let mut twice = join_as(&pool, "laptop", &at, &join("laptop", &[("laptop", 0)]));
    ledger_of(&pool, 2);
    twice.send_word(&join("laptop", &[("laptop", 1)])).unwrap();
    cut_short(&mut twice);
    coordinating.says("it joined already on this channel");
    drop(twice);
    // The laptop's channel gone, its device leaves at the end of its grace; the server's one
    // join stands.
    coordinating.says("`laptop` went without leaving");
    let ledger = ledger_of(&pool, 3);
    assert_eq!(
        changes(&ledger),
        [
            (
                Some(member("server", Trust::Owner, GIVE_BACK)),
                device("server", 0),
                true
            ),
            (
                Some(member("laptop", Trust::Site, GIVE_BACK)),
                device("laptop", 0),
                true
            ),
            (None, device("laptop", 0), false),
        ]
    );

    // The server leaves: recorded, then the coordinator closes, which the member reads as closed.
    held.send_word(&Control::Leave).unwrap();
    match held.recv() {
        Err(LinkError::Closed) => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(
        changes(&ledger_of(&pool, 4))[3],
        (None, device("server", 0), false)
    );
}

#[test]
fn an_unlisted_member_is_recorded_at_outside_and_a_listed_one_at_the_lower_trust() {
    let pool = Pool::new("trust");
    let (_coordinating, at) = coordinator(&pool);

    // A member that joins later: in no start file the coordinator has, with a certificate from
    // the same authority, saying it is trusted as an owner.
    let mut later = description();
    later
        .members
        .push(member("rented", Trust::Owner, GIVE_BACK));
    later.offers.push(offer("rented", 0, 80 << 30));
    let _rented = joined(&pool, "rented", &at, later);
    // The laptop, which the coordinator lists at `site`, saying it is an owner's.
    let mut raised = description();
    raised.members[2].trust = Trust::Owner;
    let _laptop = joined(&pool, "laptop", &at, raised);

    let ledger = ledger_of(&pool, 2);
    let mut joins = changes(&ledger);
    joins.sort_by(|a, b| a.1.cmp(&b.1));
    assert_eq!(
        joins,
        [
            (
                Some(member("laptop", Trust::Site, GIVE_BACK)),
                device("laptop", 0),
                true
            ),
            (
                Some(member("rented", Trust::Outside, GIVE_BACK)),
                device("rented", 0),
                true
            ),
        ]
    );
}

#[test]
fn a_member_with_no_offers_is_recorded_nowhere_and_its_leave_is_answered() {
    let pool = Pool::new("no-offers");
    let (mut coordinating, at) = coordinator(&pool);
    let mut bare = patient();
    bare.offers.retain(|o| o.device.member != "server");
    let mut server = joined(&pool, "server", &at, bare);
    coordinating.says("`server` joined with 0 devices");
    let (status, took) = server.terminate(patiently());
    assert_eq!(status.code(), Some(0));
    assert!(took < patiently(), "{took:?}");
    assert!(
        server.said("left: the coordinator recorded it"),
        "{:#?}",
        server.said
    );
    coordinating.says("`server` left, its 0 devices recorded");
    assert_eq!(read_ledger(&pool).0.entries.len(), 0);
}

#[test]
fn a_member_joins_at_once_while_peers_that_say_nothing_hold_connections_to_the_coordinator() {
    let pool = Pool::new("silent-peers");
    let (mut coordinating, at) = coordinator(&pool);
    // Taken one at a time, each would hold the coordinator for the opening's deadline, 10 s.
    let _silent: Vec<TcpStream> = (0..5).map(|_| TcpStream::connect(&at).unwrap()).collect();
    let started = Instant::now();
    let _laptop = joined(&pool, "laptop", &at, description());
    coordinating.says("`laptop` joined with 1 devices");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
}

#[test]
fn a_member_gone_without_leaving_is_recorded_as_left_at_the_end_of_its_grace() {
    let pool = Pool::new("gone");
    let (mut coordinating, at) = coordinator(&pool);
    let mut laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 1);
    // Timed from before the kill: the coordinator's grace begins once it notices, which is
    // after, while its line saying so can reach this test late.
    let gone = Instant::now();
    laptop.child.kill().unwrap();
    coordinating.says("`laptop` went without leaving");
    // Kept for its grace: its time to give back is 2 s, so the least grace, 5 s.
    assert_eq!(mag_mar::coordinator::LEAST_GRACE, Duration::from_secs(5));
    let ledger = ledger_of(&pool, 2);
    let took = gone.elapsed();
    assert!(took >= Duration::from_millis(4900), "{took:?}");
    assert!(took < Duration::from_secs(8), "{took:?}");
    assert_eq!(changes(&ledger)[1], (None, device("laptop", 0), false));
    coordinating.says("`laptop` did not join again within its grace");
}

#[test]
fn a_coordinator_does_not_take_up_or_overwrite_what_is_not_a_ledger_at_its_place() {
    let pool = Pool::new("ledger-there");
    let path = pool.dir.join("ledger.jsonl");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    for kept in [
        &b"kept\n"[..],
        b"",
        b"{\"format\": \"mag-ledger\"",
        b"\xff\xfe\n",
    ] {
        std::fs::write(&path, kept).unwrap();
        let mut program = Program::start(Some(&start));
        let (status, _) = program.exits(Duration::from_secs(10));
        assert_eq!(status.code(), Some(2), "{:#?}", program.said);
        program.says("is not taken up, and is left as it is");
        assert_eq!(std::fs::read(&path).unwrap(), kept);
        assert!(!pool.dir.join("ledger.jsonl.1").exists());
    }
}

/// A leave whose ledger line cannot be written is never acknowledged: the coordinator stops with
/// status 3, and the member, told nothing, gives back on its own. The line is made unwritable by
/// holding the coordinator's files to the size its ledger has reached (Linux's `prlimit`), with
/// SIGXFSZ ignored so that the write fails rather than the signal ending the program.
#[cfg(target_os = "linux")]
#[test]
fn a_leave_is_not_acknowledged_until_its_ledger_line_is_written() {
    use std::os::unix::process::CommandExt;

    let pool = Pool::new("unwritten");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let mut command = Program::command(Some(&start));
    // SAFETY: signal(2) is async-signal-safe, and SIG_IGN is kept across exec.
    unsafe {
        command.pre_exec(|| {
            libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
            Ok(())
        });
    }
    let (mut coordinating, at) = coordinator_from(command);
    let mut laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 1);
    let (_, text) = read_ledger(&pool);

    let limit_to = libc::rlimit {
        rlim_cur: text.len() as libc::rlim_t,
        rlim_max: text.len() as libc::rlim_t,
    };
    // SAFETY: prlimit(2) on the child this test started, with a limit that lives in this frame.
    let set = unsafe {
        libc::prlimit(
            coordinating.child.id() as libc::pid_t,
            libc::RLIMIT_FSIZE,
            &limit_to,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(set, 0, "{}", std::io::Error::last_os_error());

    let (status, took) = laptop.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0), "{:#?}", laptop.said);
    assert!(took < limit(), "{took:?}");
    assert!(
        !laptop.said("left: the coordinator recorded it"),
        "{:#?}",
        laptop.said
    );
    let (status, _) = coordinating.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(3), "{:#?}", coordinating.said);
    assert!(
        coordinating.said("nothing after its last line was acknowledged"),
        "{:#?}",
        coordinating.said
    );
    assert_eq!(read_ledger(&pool).1, text);
}

// The door: a command run on the coordinator's machine, `mag-mar door`, asking the coordinator.

/// A graph of one part, `sum`, that any of the test pool's devices can run; `pinned` pins it to
/// CUDA, which no device of the test pool offers, so that it waits.
fn graph_file(pool: &Pool, job: &str, pinned: bool) -> PathBuf {
    let mut graph = serde_json::json!({
        "format": "mag-graph",
        "version": 2,
        "job": { "name": job, "priority": 1 },
        "inputs": [{ "name": "numbers", "bytes": 4096, "keep": "owner" }],
        "parts": [{
            "name": "sum",
            "unit": { "name": "sum", "version": 1 },
            "mark": "atomic",
            "step": "item",
            "memory_bytes": 1 << 20,
            "weights": [],
            "tolerance": "exact"
        }],
        "chains": [],
        "edges": [
            { "from": "numbers", "to": "sum", "dtype": "f32", "bytes": 4096 },
            { "from": "sum", "dtype": "f32", "bytes": 4 }
        ]
    });
    if pinned {
        graph["parts"][0]["pin"] = serde_json::json!({ "backend": "cuda" });
    }
    let path = pool.dir.join(format!("{job}.json"));
    let text = serde_json::to_string_pretty(&graph).unwrap();
    // What the command reads is what the graph file's reader takes.
    mag_by::Graph::from_json(&text).unwrap();
    std::fs::write(&path, text).unwrap();
    path
}

/// What `mag-mar door ARGS` did, run with `MAG_START` naming `start`.
struct Asked {
    code: Option<i32>,
    answer: Option<DoorAnswer>,
    said: String,
    took: Duration,
}

fn door(start: &Path, args: &[&str]) -> Asked {
    let began = Instant::now();
    let out = Command::new(env!("CARGO_BIN_EXE_mag-mar"))
        .arg("door")
        .args(args)
        .env("MAG_START", start)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let said = String::from_utf8(out.stderr).unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    Asked {
        code: out.status.code(),
        answer: (!stdout.is_empty())
            .then(|| serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("{e}: {stdout}"))),
        said,
        took: began.elapsed(),
    }
}

fn status_at(start: &Path) -> Description {
    asks_at(start).0
}

/// The status, with what the pool asks, which the coordinator always says, and the line the
/// command said for a person.
fn asks_at(start: &Path) -> (Description, Asks, String) {
    match door(start, &["status"]) {
        Asked {
            code: Some(0),
            answer:
                Some(DoorAnswer::Status {
                    pool,
                    asks: Some(asks),
                }),
            said,
            ..
        } => (pool, asks, said),
        Asked { code, said, .. } => panic!("{code:?}: {said}"),
    }
}

fn path(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn a_job_submitted_at_the_door_is_written_before_it_is_answered_ended_and_replayed() {
    let pool = Pool::new("door");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let (mut coordinating, at) = coordinator_from(Program::command(Some(&start)));
    coordinating.says("serves the door at");
    let _server = joined(&pool, "server", &at, description());
    let _laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 3);

    // The door is a socket only the coordinator's user can reach.
    let door_path = pool.dir.join("door.sock");
    let meta = std::fs::symlink_metadata(&door_path).unwrap();
    assert!(std::os::unix::fs::FileTypeExt::is_socket(&meta.file_type()));
    assert_eq!(
        std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o777,
        0o600
    );
    assert!(!pool.dir.join("door.sock.new").exists());

    // Status before: the pool as one virtual device, its three members' four devices, no job.
    let before = status_at(&start);
    let devices =
        |d: &Description| -> Vec<DeviceId> { d.offers.iter().map(|o| o.device.clone()).collect() };
    assert_eq!(
        devices(&before),
        [
            device("laptop", 0),
            device("server", 0),
            device("server", 1),
            device("workstation", 0)
        ]
    );
    assert_eq!(before.members.len(), 3);
    assert!(before.jobs.is_empty());

    // Submitted with its stamp: placed, and in the ledger, synced, as the command has its answer.
    let graph = graph_file(&pool, "tally", false);
    let stamp = pool.dir.join("stamp.json");
    std::fs::write(&stamp, r#"{ "by": "ops", "priority_ceiling": 1 }"#).unwrap();
    let asked = door(&start, &["submit", path(&graph), "--stamp", path(&stamp)]);
    assert_eq!(asked.code, Some(0), "{}", asked.said);
    assert!(asked.said.contains("`tally` is placed"), "{}", asked.said);
    let Some(DoorAnswer::Submitted {
        record,
        arrangement,
    }) = asked.answer
    else {
        panic!("{:?}", asked.answer)
    };
    assert_eq!(arrangement.job, "tally");
    assert!(matches!(arrangement.places[..], [Place::On(_)]));
    assert!(record.chosen.is_some());
    let (ledger, _) = read_ledger(&pool);
    assert_eq!(ledger.entries.len(), 4);
    let submitted = &ledger.entries[3];
    match &submitted.change {
        Change::Submit { graph: g, stamp: s } => {
            assert_eq!(g.job.name, "tally");
            assert_eq!(s.by.as_deref(), Some("ops"));
            assert_eq!(s.priority_ceiling, Some(1));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(submitted.record, record);
    coordinating.says("the door: `tally` submitted, written, and placed");

    // Status after: the job, where the answer said.
    let during = status_at(&start);
    assert_eq!(during.jobs.len(), 1);
    assert_eq!(during.jobs[0].arrangement, arrangement);
    assert_eq!(during.jobs[0].priority, 1);

    // Ended: in the ledger as it is answered, and gone from the status.
    let asked = door(&start, &["end", "tally"]);
    assert_eq!(asked.code, Some(0), "{}", asked.said);
    let Some(DoorAnswer::Ended { record }) = asked.answer else {
        panic!("{:?}", asked.answer)
    };
    let (ledger, text) = read_ledger(&pool);
    assert_eq!(ledger.entries.len(), 5);
    assert_eq!(
        ledger.entries[4].change,
        Change::End {
            job: "tally".into()
        }
    );
    assert_eq!(ledger.entries[4].record, record);
    let after = status_at(&start);
    assert!(after.jobs.is_empty());
    assert_eq!(devices(&after), devices(&before));

    // The ledger replays to the same pool: the same lines, and the status the door gave.
    let replayed = mag_mortar::Pool::replay(&ledger).unwrap();
    assert_eq!(replayed.ledger().to_lines(), text);
    assert_eq!(replayed.describe(), after);
    let mut upto = ledger.clone();
    upto.entries.truncate(4);
    assert_eq!(mag_mortar::Pool::replay(&upto).unwrap().describe(), during);

    // On its notice the coordinator closes its door: the command then finds no one.
    let (status, _) = coordinating.terminate(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "{:#?}", coordinating.said);
    assert!(!door_path.exists());
    assert_eq!(door(&start, &["status"]).code, Some(5));
}

#[test]
fn the_door_refuses_what_the_pool_refuses_writes_nothing_for_it_and_says_when_a_job_waits() {
    let pool = Pool::new("door-refuses");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let (_coordinating, _) = coordinator_from(Program::command(Some(&start)));

    // A job no device can take: in the ledger, and waiting.
    let waits = graph_file(&pool, "on-cuda", true);
    let asked = door(&start, &["submit", path(&waits)]);
    assert_eq!(asked.code, Some(3), "{}", asked.said);
    assert!(asked.said.contains("`on-cuda` waits"), "{}", asked.said);
    assert_eq!(asked.answer.as_ref().unwrap().placed(), Some(false));
    assert_eq!(read_ledger(&pool).0.entries.len(), 1);

    let refused = |asked: Asked| match asked {
        Asked {
            code: Some(4),
            answer: Some(DoorAnswer::Refused { refusal }),
            ..
        } => refusal,
        Asked { code, said, .. } => panic!("{code:?}: {said}"),
    };
    // A second job of one name.
    assert!(matches!(
        refused(door(&start, &["submit", path(&waits)])),
        DoorRefusal::Pool { reason } if reason.contains("already holds a job named `on-cuda`")
    ));
    // A stamp the pool cannot take.
    let tally = graph_file(&pool, "tally", false);
    let stamp = pool.dir.join("stamp.json");
    std::fs::write(&stamp, r#"{ "priority_ceiling": 0 }"#).unwrap();
    assert!(matches!(
        refused(door(&start, &["submit", path(&tally), "--stamp", path(&stamp)])),
        DoorRefusal::Pool { reason } if reason.contains("ceiling is 0")
    ));
    // The end of a job the pool does not hold.
    assert_eq!(
        refused(door(&start, &["end", "tally"])),
        DoorRefusal::NoJob {
            job: "tally".into()
        }
    );
    // None of them written.
    assert_eq!(read_ledger(&pool).0.entries.len(), 1);

    // A line that is no request, sent to the socket by hand.
    let mut stream = std::os::unix::net::UnixStream::connect(pool.dir.join("door.sock")).unwrap();
    stream.write_all(b"{\"stop\": {}}\n").unwrap();
    let mut answer = String::new();
    std::io::Read::read_to_string(&mut stream, &mut answer).unwrap();
    assert!(matches!(
        serde_json::from_str::<DoorAnswer>(&answer).unwrap(),
        DoorAnswer::Refused {
            refusal: DoorRefusal::Unreadable { .. }
        }
    ));
    // A line longer than the door reads, refused once it has read that much. A mebibyte more
    // than that is still coming as the door answers: it reads that and lets it go before it
    // closes, or the answer would end in a reset.
    let mut stream = std::os::unix::net::UnixStream::connect(pool.dir.join("door.sock")).unwrap();
    let long = vec![b' '; mag_mar::door::REQUEST_LIMIT as usize + (1 << 20)];
    let writing = {
        let mut stream = stream.try_clone().unwrap();
        thread::spawn(move || {
            let _ = stream.write_all(&long);
        })
    };
    // Read once the writing has ended: by then the door has read the rest, or has closed on it
    // and the answer would end in a reset.
    writing.join().unwrap();
    let mut answer = String::new();
    std::io::Read::read_to_string(&mut stream, &mut answer).unwrap();
    match serde_json::from_str::<DoorAnswer>(&answer).unwrap() {
        DoorAnswer::Refused {
            refusal: DoorRefusal::Unreadable { error },
        } => assert!(error.contains("longer than"), "{error}"),
        other => panic!("{other:?}"),
    }
    // It reads what still comes after its answer, until the asker closes: a mebibyte more, more
    // than a socket holds unread, goes through.
    stream.write_all(&vec![b' '; 1 << 20]).unwrap();
    drop(stream);

    // The command's own refusals, before it asks: a graph file it cannot read, and its usage.
    let asked = door(&start, &["submit", path(&pool.dir.join("none.json"))]);
    assert_eq!(asked.code, Some(2));
    let asked = door(&start, &["stop"]);
    assert_eq!(asked.code, Some(2));
    assert!(asked.said.contains("usage"), "{}", asked.said);
    assert_eq!(read_ledger(&pool).0.entries.len(), 1);
}

#[test]
fn the_status_asks_for_the_kind_of_device_a_waiting_job_wants_and_gives_back_idle_members() {
    let pool = Pool::new("door-asks");
    // An idle time of 0: a member holding no part is offered back at once.
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let mut file = Start::from_json(&std::fs::read_to_string(&start).unwrap()).unwrap();
    file.idle_seconds = 0.0;
    std::fs::write(&start, file.to_json()).unwrap();
    let (_coordinating, at) = coordinator_from(Program::command(Some(&start)));
    let _server = joined(&pool, "server", &at, description());
    let _laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 3);

    // Before: nothing waits, so nothing is wanted; both members hold nothing, and are offered
    // back, the coordinator never.
    let (_, asks, said) = asks_at(&start);
    assert_eq!(asks.wants, None);
    let mut back = asks.gives_back.clone();
    back.sort();
    assert_eq!(back, ["laptop", "server"]);
    assert!(said.contains("wants no device"), "{said}");

    // A job pinned to CUDA, which no device offers, waits; the status then asks for a CUDA
    // device with its memory, a member trusted as far as its input is kept, its tolerance class
    // and its edges' dtype.
    let waiting = graph_file(&pool, "on-cuda", true);
    let asked = door(&start, &["submit", path(&waiting)]);
    assert_eq!(asked.code, Some(3), "{}", asked.said);
    let wanted = DeviceKind {
        backend: Some(Backend::Cuda),
        memory_bytes: Some(1 << 20),
        trust: Some(Trust::Owner),
        tolerances: vec![Tolerance::Exact],
        dtypes: vec![Dtype::F32],
    };
    let (status, asks, said) = asks_at(&start);
    assert_eq!(asks.wants.as_ref(), Some(&wanted));
    assert!(
        said.contains("wants a device {\"backend\":\"cuda\""),
        "{said}"
    );
    // A device of that kind, not one the pool has.
    assert!(!status.offers.iter().any(|o| wanted.admits(o, Trust::Owner)));

    // A job that is placed: the member holding it is no longer offered back.
    let placed = graph_file(&pool, "tally", false);
    let asked = door(&start, &["submit", path(&placed)]);
    assert_eq!(asked.code, Some(0), "{}", asked.said);
    let (status, asks, said) = asks_at(&start);
    let holder = &status.jobs[1].arrangement.places[0];
    // Kept at `owner`, so not on the laptop, trusted at `site`.
    assert_eq!(holder, &Place::On(device("server", 0)));
    assert_eq!(asks.gives_back, ["laptop"]);
    assert_eq!(asks.wants.as_ref(), Some(&wanted));
    assert!(said.ends_with("would give back laptop\n"), "{said}");

    // The pool the ledger replays to asks the same, at the status's time.
    let (ledger, _) = read_ledger(&pool);
    let replayed = mag_mortar::Pool::replay(&ledger).unwrap();
    assert_eq!(replayed.asks("workstation", 0.0, status.at), asks);
}

#[test]
fn submissions_while_members_join_are_written_in_the_order_the_pool_took_them() {
    let pool = Pool::new("door-busy");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let (_coordinating, at) = coordinator_from(Program::command(Some(&start)));
    let graphs: Vec<PathBuf> = (0..6)
        .map(|i| graph_file(&pool, &format!("job-{i}"), false))
        .collect();
    // Six commands at once, while two members join.
    let asking: Vec<_> = graphs
        .into_iter()
        .map(|g| {
            let start = start.clone();
            thread::spawn(move || door(&start, &["submit", path(&g)]))
        })
        .collect();
    let _server = joined(&pool, "server", &at, description());
    let _laptop = joined(&pool, "laptop", &at, description());
    // Each in the pool, placed or waiting as the loop chose, which is not this test's to say.
    for a in asking {
        let asked = a.join().unwrap();
        assert!(matches!(asked.code, Some(0 | 3)), "{}", asked.said);
    }
    // Six submissions and three joins, the joins written once the coordinator hears them.
    ledger_of(&pool, 9);
    let (ledger, text) = read_ledger(&pool);
    let submitted = ledger
        .entries
        .iter()
        .filter(|e| matches!(e.change, Change::Submit { .. }))
        .count();
    assert_eq!(submitted, 6);
    // Each line whole, and the whole replaying to the same records and lines.
    let replayed = mag_mortar::Pool::replay(&ledger).unwrap();
    assert_eq!(replayed.ledger().to_lines(), text);
    assert_eq!(replayed.describe().jobs.len(), 6);
}

#[test]
fn the_door_command_fails_at_once_when_no_coordinator_answers() {
    let pool = Pool::new("no-door");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    // Nothing at the door's path.
    let asked = door(&start, &["status"]);
    assert_eq!(asked.code, Some(5), "{}", asked.said);
    assert!(asked.said.contains("no coordinator answers at the door"));
    assert!(asked.took < Duration::from_secs(2), "{:?}", asked.took);
    assert!(asked.answer.is_none());
    // A socket there that no one answers, as a coordinator killed leaves it.
    let left = std::os::unix::net::UnixListener::bind(pool.dir.join("door.sock")).unwrap();
    drop(left);
    let graph = graph_file(&pool, "tally", false);
    let asked = door(&start, &["submit", path(&graph)]);
    assert_eq!(asked.code, Some(5), "{}", asked.said);
    assert!(asked.said.contains("no coordinator answers at the door"));
    assert!(asked.took < Duration::from_secs(2), "{:?}", asked.took);
    // No `MAG_START`.
    let out = Command::new(env!("CARGO_BIN_EXE_mag-mar"))
        .args(["door", "status"])
        .env_remove("MAG_START")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    // The program takes no other argument than `door`: it does not start as a member, here the
    // coordinator, which would run until its notice.
    let mut command = Program::command(Some(&start));
    command.arg("doors");
    let mut program = Program::spawn(command);
    let (status, _) = program.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(2), "{:#?}", program.said);
    program.says("usage");
}

/// Writes the start file at `start` again with its ledger at `ledger`, in the pool's directory.
fn with_ledger(start: &Path, ledger: &str, pool: &Pool) {
    let mut s = Start::from_json(&std::fs::read_to_string(start).unwrap()).unwrap();
    s.ledger = pool.dir.join(ledger).to_str().unwrap().into();
    std::fs::write(start, s.to_json()).unwrap();
}

#[test]
fn a_second_coordinator_at_a_door_is_refused_and_a_door_left_behind_is_replaced() {
    let pool = Pool::new("door-held");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let (mut first, _) = coordinator_from(Program::command(Some(&start)));

    // A second coordinator at the same door, with a ledger of its own: refused, its ledger not
    // begun, and the first's door still answering.
    let second = pool.dir.join("second.json");
    std::fs::copy(&start, &second).unwrap();
    with_ledger(&second, "second.jsonl", &pool);
    let mut refused = Program::start(Some(&second));
    let (status, _) = refused.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(2), "{:#?}", refused.said);
    refused.says("another coordinator serves this door");
    assert!(!pool.dir.join("second.jsonl").exists());
    assert_eq!(status_at(&start).members.len(), 1);
    // Its socket moved away, the first's door is still its own: it holds the lock.
    let door_path = pool.dir.join("door.sock");
    let away = pool.dir.join("away.sock");
    std::fs::rename(&door_path, &away).unwrap();
    let mut refused = Program::start(Some(&second));
    let (status, _) = refused.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(2), "{:#?}", refused.said);
    refused.says("another coordinator serves this door");
    assert!(!door_path.exists());
    std::fs::rename(&away, &door_path).unwrap();
    // Its lock removed, the first's door is still its own: the socket answers.
    std::fs::remove_file(pool.dir.join("door.sock.lock")).unwrap();
    let mut refused = Program::start(Some(&second));
    let (status, _) = refused.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(2), "{:#?}", refused.said);
    refused.says("another coordinator serves this door");
    assert!(!pool.dir.join("second.jsonl").exists());
    assert_eq!(status_at(&start).members.len(), 1);

    // The first killed, its door left behind, a socket no one answers: the next coordinator
    // replaces it.
    first.child.kill().unwrap();
    first.child.wait().unwrap();
    assert!(pool.dir.join("door.sock").exists());
    assert_eq!(door(&start, &["status"]).code, Some(5));
    let (mut third, _) = coordinator_from(Program::command(Some(&second)));
    third.says("serves the door");
    assert!(pool.dir.join("second.jsonl").exists());
    assert_eq!(status_at(&start).members.len(), 1);
    drop(third);

    // Anything at the door's path that is not a socket is not a door: refused, and left.
    let _ = std::fs::remove_file(&door_path);
    std::fs::write(&door_path, "kept\n").unwrap();
    with_ledger(&second, "fourth.jsonl", &pool);
    let mut fourth = Program::start(Some(&second));
    let (status, _) = fourth.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(2), "{:#?}", fourth.said);
    fourth.says("something other than a socket");
    assert_eq!(std::fs::read_to_string(&door_path).unwrap(), "kept\n");
    assert!(!pool.dir.join("fourth.jsonl").exists());
}

/// A submission whose ledger line cannot be written is never answered: the coordinator stops
/// with status 3, and the command finds the door closed without an answer. The line is made
/// unwritable as for a leave, by `prlimit`.
#[cfg(target_os = "linux")]
#[test]
fn a_submission_is_not_answered_until_its_ledger_line_is_written() {
    use std::os::unix::process::CommandExt;

    let pool = Pool::new("door-unwritten");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let mut command = Program::command(Some(&start));
    // SAFETY: signal(2) is async-signal-safe, and SIG_IGN is kept across exec.
    unsafe {
        command.pre_exec(|| {
            libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
            Ok(())
        });
    }
    let (mut coordinating, at) = coordinator_from(command);
    let _laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 1);
    let (_, text) = read_ledger(&pool);
    let limit_to = libc::rlimit {
        rlim_cur: text.len() as libc::rlim_t,
        rlim_max: text.len() as libc::rlim_t,
    };
    // SAFETY: prlimit(2) on the child this test started, with a limit that lives in this frame.
    let set = unsafe {
        libc::prlimit(
            coordinating.child.id() as libc::pid_t,
            libc::RLIMIT_FSIZE,
            &limit_to,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(set, 0, "{}", std::io::Error::last_os_error());

    let graph = graph_file(&pool, "tally", false);
    let asked = door(&start, &["submit", path(&graph)]);
    assert_eq!(asked.code, Some(5), "{}", asked.said);
    assert!(asked.answer.is_none());
    assert!(
        asked.said.contains("closed the door without answering"),
        "{}",
        asked.said
    );
    let (status, _) = coordinating.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(3), "{:#?}", coordinating.said);
    assert_eq!(read_ledger(&pool).1, text);
}

// Taking a device back by need, at the door.

/// What `mag-mar door take-back KIND SECONDS` was answered: the device taken back, the record
/// of its leave, and whether its member said it gave it back; with the line for a person and the
/// time it took.
fn take_back(
    start: &Path,
    kind: &str,
    seconds: &str,
) -> (DeviceId, Record, bool, String, Duration) {
    match door(start, &["take-back", kind, seconds]) {
        Asked {
            code: Some(0),
            answer:
                Some(DoorAnswer::TakenBack {
                    device,
                    record,
                    given_back,
                }),
            said,
            took,
        } => (device, record, given_back, said, took),
        Asked { code, said, .. } => panic!("{code:?}: {said}"),
    }
}

/// What the door refused, with exit 4.
fn refused_at(start: &Path, args: &[&str]) -> DoorRefusal {
    match door(start, args) {
        Asked {
            code: Some(4),
            answer: Some(DoorAnswer::Refused { refusal }),
            ..
        } => refusal,
        Asked { code, said, .. } => panic!("{code:?}: {said}"),
    }
}

/// The devices a pool's status offers.
fn offered(d: &Description) -> Vec<DeviceId> {
    d.offers.iter().map(|o| o.device.clone()).collect()
}

#[test]
fn a_device_taken_back_by_need_is_written_its_member_told_and_the_door_answered() {
    let pool = Pool::new("take-back");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let (mut coordinating, at) = coordinator_from(Program::command(Some(&start)));
    let mut server = joined(&pool, "server", &at, description());
    let mut laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 3);
    // A job kept at `owner`, so not on the laptop: on the first of the server's devices.
    let graph = graph_file(&pool, "tally", false);
    assert_eq!(door(&start, &["submit", path(&graph)]).code, Some(0));
    let (ledger, _) = read_ledger(&pool);
    assert_eq!(ledger.entries.len(), 4);
    assert_eq!(
        status_at(&start).jobs[0].arrangement.places,
        [Place::On(device("server", 0))]
    );

    // Any device within 10 s: of those that lose the pool nothing, the laptop's, which offers
    // least memory. Its leave is in the ledger, as the record the door answers with; the laptop
    // is told, and says it gave it back, before the deadline.
    let (taken, record, given_back, said, took) = take_back(&start, "{}", "10");
    assert_eq!(taken, device("laptop", 0));
    assert!(given_back, "{said}");
    assert!(said.contains("laptop/0 is given back"), "{said}");
    assert!(took < Duration::from_secs(5), "{took:?}");
    let (ledger, _) = read_ledger(&pool);
    assert_eq!(ledger.entries.len(), 5);
    assert_eq!(
        ledger.entries[4].change,
        Change::Leave {
            device: device("laptop", 0)
        }
    );
    assert_eq!(ledger.entries[4].record, record);
    coordinating.says("tells `laptop` to give laptop/0 back within 10 s");
    coordinating.says("`laptop` gave laptop/0 back");
    laptop.says("gives laptop/0 back at the coordinator's word, within 2 s");
    // Its only device: it stays a member, with none, until its lease ends.
    laptop.says("offers no device now, and stays a member until its lease ends");
    assert!(laptop.child.try_wait().unwrap().is_none());
    assert!(!offered(&status_at(&start)).contains(&device("laptop", 0)));

    // A device of 16 GiB or more: the server's two are of the kind, and the coordinator's is not.
    // Every cost is the same estimate, so losing the server's first, which holds the job, costs
    // nothing: the job is placed again, and of equal losses the smaller device is given up.
    let (taken, record, given_back, said, _) =
        take_back(&start, r#"{"memory_bytes":17179869184}"#, "10");
    assert_eq!(taken, device("server", 0));
    assert!(given_back, "{said}");
    assert!(record.chosen.is_some());
    let now = status_at(&start);
    assert_eq!(
        now.jobs[0].arrangement.places,
        [Place::On(device("server", 1))]
    );
    assert_eq!(
        offered(&now),
        [device("server", 1), device("workstation", 0)]
    );
    server.says("gives server/0 back at the coordinator's word");
    assert!(!server.said("offers no device now"), "{:#?}", server.said);

    // The ledger replays to the pool the door shows.
    let (ledger, text) = read_ledger(&pool);
    assert_eq!(ledger.entries.len(), 6);
    let replayed = mag_mortar::Pool::replay(&ledger).unwrap();
    assert_eq!(replayed.ledger().to_lines(), text);
    assert_eq!(offered(&replayed.describe()), offered(&now));

    // The laptop's lease ends with no device to record; the server's with the one it kept.
    let (status, _) = laptop.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0), "{:#?}", laptop.said);
    assert!(left(&laptop), "{:#?}", laptop.said);
    let (status, _) = server.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0), "{:#?}", server.said);
    let (ledger, _) = read_ledger(&pool);
    assert_eq!(
        ledger.entries[6..]
            .iter()
            .map(|e| e.change.clone())
            .collect::<Vec<_>>(),
        [Change::Leave {
            device: device("server", 1)
        }]
    );
}

#[test]
fn a_need_nothing_meets_and_a_take_back_by_name_are_refused_and_the_coordinators_own_is_last() {
    let pool = Pool::new("take-back-refused");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let (mut coordinating, _) = coordinator_from(Program::command(Some(&start)));
    coordinating.says("serves the door at");

    // No device of the kind.
    assert_eq!(
        refused_at(&start, &["take-back", r#"{"backend":"cuda"}"#, "60"]),
        DoorRefusal::Unmet { slow: vec![] }
    );
    // Of the kind, but the coordinator's member gives a device back within 30 s, not 1.
    let asked = door(&start, &["take-back", "{}", "1"]);
    assert_eq!(asked.code, Some(4), "{}", asked.said);
    assert!(
        asked
            .said
            .contains("(workstation/0) are each of a member that takes longer"),
        "{}",
        asked.said
    );
    assert_eq!(
        refused_at(&start, &["take-back", "{}", "1"]),
        DoorRefusal::Unmet {
            slow: vec![device("workstation", 0)]
        }
    );
    // By name, which the command does not say, sent to the socket by hand.
    for by_name in [
        r#"{"take_back":{"member":{"member":"workstation","deadline_seconds":60}}}"#,
        concat!(
            r#"{"take_back":{"device":{"device":{"member":"workstation","index":0},"#,
            r#""deadline_seconds":60}}}"#
        ),
    ] {
        let mut stream =
            std::os::unix::net::UnixStream::connect(pool.dir.join("door.sock")).unwrap();
        stream.write_all(format!("{by_name}\n").as_bytes()).unwrap();
        let mut answer = String::new();
        std::io::Read::read_to_string(&mut stream, &mut answer).unwrap();
        assert_eq!(
            serde_json::from_str::<DoorAnswer>(&answer).unwrap(),
            DoorAnswer::Refused {
                refusal: DoorRefusal::ByName
            }
        );
    }
    // A kind or a time the command cannot say.
    assert_eq!(door(&start, &["take-back", "cuda", "60"]).code, Some(2));
    assert_eq!(door(&start, &["take-back", "{}", "-1"]).code, Some(2));
    // Nothing written.
    assert!(read_ledger(&pool).0.entries.is_empty());

    // With no other member, the coordinator's own device is given up: it stops offering it.
    let (taken, record, given_back, said, _) = take_back(&start, "{}", "30");
    assert_eq!(taken, device("workstation", 0));
    assert!(given_back, "{said}");
    coordinating.says("gives its own workstation/0 back");
    let (ledger, _) = read_ledger(&pool);
    assert_eq!(ledger.entries.len(), 1);
    assert_eq!(ledger.entries[0].record, record);
    assert!(status_at(&start).offers.is_empty());
    // On its notice it records no second leave of it.
    let (status, _) = coordinating.terminate(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "{:#?}", coordinating.said);
    assert_eq!(read_ledger(&pool).0.entries.len(), 1);
}

#[test]
fn a_member_that_does_not_say_it_gave_a_device_back_is_answered_for_at_the_deadline() {
    let pool = Pool::new("take-back-late");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let (mut coordinating, at) = coordinator_from(Program::command(Some(&start)));
    // A member held by the test, which joins and then never answers.
    let mut server = join_as(
        &pool,
        "server",
        &at,
        &Control::Join {
            member: member("server", Trust::Owner, GIVE_BACK),
            offers: vec![offer("server", 0, 16 << 30)],
        },
    );
    ledger_of(&pool, 1);

    // Within its time to give back: the server's device, since the coordinator's own member
    // takes 30 s.
    let asking = {
        let start = start.clone();
        thread::spawn(move || take_back(&start, "{}", "2"))
    };
    // The member is told, once the leave is in the ledger.
    assert_eq!(
        server.recv_word::<Control>().unwrap(),
        Control::GiveBack {
            device: device("server", 0),
            deadline_seconds: 2.0
        }
    );
    let (ledger, _) = read_ledger(&pool);
    assert_eq!(
        ledger.entries[1].change,
        Change::Leave {
            device: device("server", 0)
        }
    );
    // It says nothing: the door is answered at the deadline, the device taken back but not
    // said to be given back.
    let (taken, record, given_back, said, took) = asking.join().unwrap();
    assert_eq!(taken, device("server", 0));
    assert!(!given_back);
    assert!(said.contains("it is yours now"), "{said}");
    assert!(took >= Duration::from_millis(1900), "{took:?}");
    assert!(took < Duration::from_secs(10), "{took:?}");
    assert_eq!(read_ledger(&pool).0.entries[1].record, record);
    coordinating.says("did not say it gave it back before the deadline");

    // Said late, it is only noted; its leave is not written again.
    server
        .send_word(&Control::GaveBack {
            device: device("server", 0),
        })
        .unwrap();
    coordinating.says("which no take-back waits for");
    server.send_word(&Control::Leave).unwrap();
    coordinating.says("`server` left, its 0 devices recorded");
    assert_eq!(read_ledger(&pool).0.entries.len(), 2);

    // A member told to give a device back that leaves instead gives it back with the rest: the
    // door is answered at once. Its devices are larger than the coordinator's, which, holding
    // nothing either, would be given up first but for being the coordinator's.
    let mut laptop = join_as(
        &pool,
        "laptop",
        &at,
        &Control::Join {
            member: member("laptop", Trust::Site, GIVE_BACK),
            offers: vec![offer("laptop", 0, 16 << 30), offer("laptop", 1, 16 << 30)],
        },
    );
    ledger_of(&pool, 4);
    let asking = {
        let start = start.clone();
        thread::spawn(move || take_back(&start, "{}", "30"))
    };
    assert!(matches!(
        laptop.recv_word::<Control>().unwrap(),
        Control::GiveBack { device: d, .. } if d == device("laptop", 0)
    ));
    laptop.send_word(&Control::Leave).unwrap();
    let (taken, _, given_back, said, took) = asking.join().unwrap();
    assert_eq!(taken, device("laptop", 0));
    assert!(given_back, "{said}");
    assert!(took < Duration::from_secs(10), "{took:?}");
    // Its other device's leave, and not the first's again.
    let (ledger, _) = read_ledger(&pool);
    assert_eq!(
        ledger.entries[4..]
            .iter()
            .map(|e| e.change.clone())
            .collect::<Vec<_>>(),
        [
            Change::Leave {
                device: device("laptop", 0)
            },
            Change::Leave {
                device: device("laptop", 1)
            }
        ]
    );
}

#[test]
fn a_member_told_to_give_back_a_device_it_does_not_offer_gives_back_nothing_for_it() {
    let pool = Pool::new("give-back-other");
    let link = TlsLink::bind(pool.credentials("workstation"), "127.0.0.1:0").unwrap();
    let address = link.local_addr().unwrap().to_string();
    let start = pool.start("server", None, Some(&address));
    let mut program = Program::start(Some(&start));
    let mut channel = link.accept(&["server"]).unwrap();
    assert!(matches!(
        channel.recv_word::<Control>().unwrap(),
        Control::Join { .. }
    ));
    channel.send_word(&Control::Joined).unwrap();
    // The laptop's device, then one of its own.
    for d in [device("laptop", 0), device("server", 1)] {
        channel
            .send_word(&Control::GiveBack {
                device: d,
                deadline_seconds: 10.0,
            })
            .unwrap();
    }
    // It says it gave back its own, and only that.
    assert_eq!(
        channel.recv_word::<Control>().unwrap(),
        Control::GaveBack {
            device: device("server", 1)
        }
    );
    program.says("is told to give laptop/0 back, which it does not offer");
    program.says("gives server/1 back at the coordinator's word");
    // It keeps its other device, and its lease.
    assert!(!program.said("offers no device now"), "{:#?}", program.said);
    let (status, _) = program.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0), "{:#?}", program.said);
    assert_eq!(channel.recv_word::<Control>().unwrap(), Control::Leave);
}

// A coordinator that stops, and one that takes up its ledger.

/// The coordinator started again at `at`, the address the members dial, on its start file with
/// that address: it and its start file. `at` is a [`steady_address`], which nothing else takes
/// while the coordinator is down.
fn again(pool: &Pool, at: &str) -> (Program, PathBuf) {
    let start = pool.start("workstation", Some(at), None);
    let (program, answers) = coordinator_from(Program::command(Some(&start)));
    assert_eq!(answers, at);
    (program, start)
}

/// Kills the coordinator: no notice, nothing more written.
fn kill(program: &mut Program) {
    program.child.kill().unwrap();
    program.child.wait().unwrap();
}

/// The ledger's place in the pool's directory.
fn ledger_path(pool: &Pool) -> PathBuf {
    pool.dir.join("ledger.jsonl")
}

#[test]
fn a_coordinator_killed_and_started_again_takes_up_its_ledger_and_its_members_join_again() {
    let pool = Pool::new("restart");
    let start = pool.start("workstation", Some(&steady_address()), None);
    let (mut first, at) = coordinator_from(Program::command(Some(&start)));
    let mut server = joined(&pool, "server", &at, description());
    let mut laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 3);
    let graph = graph_file(&pool, "sums", false);
    let asked = door(&start, &["submit", path(&graph)]);
    assert_eq!(asked.code, Some(0), "{}", asked.said);
    let before = status_at(&start);
    assert_eq!(before.jobs.len(), 1);
    assert_eq!(before.offers.len(), 4);
    let (_, text) = read_ledger(&pool);

    // Killed with its members joined and a job placed. The members run on, and dial again; the
    // door answers no one meanwhile.
    kill(&mut first);
    for member in [&mut server, &mut laptop] {
        member.says("the control channel ended");
        member.says("is not reached");
    }
    assert_eq!(door(&start, &["status"]).code, Some(5));
    assert_eq!(read_ledger(&pool).1, text);

    // Started again on the same ledger: replayed, each record as logged, and the members join
    // again, each with the devices the pool holds, so nothing is written.
    let (mut second, start) = again(&pool, &at);
    second.says("takes up the ledger");
    second.says("its 4 entries replayed");
    second.says("awaits `laptop`");
    second.says("awaits `server`");
    second.says("`server` joined again with 2 devices, at trust owner: 2 held already, 0 joined");
    second.says("`laptop` joined again with 1 devices, at trust site: 1 held already, 0 joined");
    server.says_times("joined the pool", 2);
    laptop.says_times("joined the pool", 2);
    assert_eq!(read_ledger(&pool).1, text);
    // The job still held where it was, and the door answering as before.
    assert_eq!(status_at(&start), before);

    // Appended to, as before: the server leaves on SIGTERM, then the coordinator on its own.
    let (status, _) = server.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0), "{:#?}", server.said);
    assert!(left(&server), "{:#?}", server.said);
    let (status, _) = second.terminate(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "{:#?}", second.said);
    let (ledger, text) = read_ledger(&pool);
    assert_eq!(ledger.entries.len(), 7, "{text}");
    let replayed = mag_mortar::Pool::replay(&ledger).unwrap();
    assert_eq!(replayed.ledger().to_lines(), text);
}

#[test]
fn a_last_ledger_line_cut_short_is_dropped_said_and_cut_away_before_the_ledger_goes_on() {
    let pool = Pool::new("torn");
    let start = pool.start("workstation", Some(&steady_address()), None);
    let (mut first, at) = coordinator_from(Program::command(Some(&start)));
    let mut laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 1);
    kill(&mut first);
    laptop.says("the control channel ended");

    // A line the coordinator was writing as it died: never acknowledged.
    let whole = std::fs::read(ledger_path(&pool)).unwrap();
    let mut torn = whole.clone();
    torn.extend_from_slice(b"{\"at\":17");
    std::fs::write(ledger_path(&pool), &torn).unwrap();

    let (mut second, start) = again(&pool, &at);
    second.says("last line, line 3, was cut short and so never acknowledged: dropped");
    second.says("its 1 entries replayed");
    second.says("`laptop` joined again");
    laptop.says_times("joined the pool", 2);
    assert_eq!(std::fs::read(ledger_path(&pool)).unwrap(), whole);
    // The next line follows the whole ones.
    let graph = graph_file(&pool, "sums", false);
    assert_eq!(door(&start, &["submit", path(&graph)]).code, Some(0));
    let (ledger, text) = read_ledger(&pool);
    assert_eq!(ledger.entries.len(), 2, "{text}");
    assert_eq!(
        mag_mortar::Pool::replay(&ledger)
            .unwrap()
            .ledger()
            .to_lines(),
        text
    );
}

#[test]
fn a_ledger_another_version_wrote_is_read_for_its_facts_and_a_new_one_follows_it() {
    let pool = Pool::new("other-version");
    let start = pool.start("workstation", Some(&steady_address()), None);
    let (mut first, at) = coordinator_from(Program::command(Some(&start)));
    let mut server = joined(&pool, "server", &at, description());
    let mut laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 3);
    for (job, code) in [("sums", 0), ("ended", 0)] {
        let graph = graph_file(&pool, job, false);
        assert_eq!(door(&start, &["submit", path(&graph)]).code, Some(code));
    }
    assert_eq!(door(&start, &["end", "ended"]).code, Some(0));
    let before = status_at(&start);
    ledger_of(&pool, 6);
    kill(&mut first);

    // Written, it says, by another version of mag-mortar.
    let text = std::fs::read_to_string(ledger_path(&pool)).unwrap();
    let (head, rest) = text.split_once('\n').unwrap();
    let mut header: serde_json::Value = serde_json::from_str(head).unwrap();
    header["written_by"] = "0.0.0-elsewhere".into();
    let faked = format!("{header}\n{rest}");
    std::fs::write(ledger_path(&pool), &faked).unwrap();

    let (mut second, start) = again(&pool, &at);
    let moved = pool.dir.join("ledger.jsonl.1");
    second.says("was written by mag-mortar 0.0.0-elsewhere");
    second.says(&format!("kept as {}", moved.display()));
    second.says("with 1 jobs it held submitted again");
    // The old one kept as it was; the new one in its place, following it.
    assert_eq!(std::fs::read_to_string(&moved).unwrap(), faked);
    let (ledger, text) = read_ledger(&pool);
    assert_eq!(ledger.written_by.as_deref(), Some(mag_mortar::VERSION));
    assert_eq!(ledger.follows.as_deref(), Some(moved.to_str().unwrap()));
    let names: Vec<&str> = ledger.members.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["workstation", "server", "laptop"]);
    assert_eq!(ledger.offers, before.offers);
    // The job it held submitted again, the ended one not; nothing else.
    match &ledger.entries[..] {
        [e] => match &e.change {
            Change::Submit { graph, .. } => assert_eq!(graph.job.name, "sums"),
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
    // The members join again, holding the devices the new ledger began with: nothing written.
    second.says("`server` joined again with 2 devices");
    second.says("`laptop` joined again with 1 devices");
    server.says_times("joined the pool", 2);
    laptop.says_times("joined the pool", 2);
    assert_eq!(read_ledger(&pool).1, text);
    // The job held, placed afresh where it was, and the new ledger replays by itself.
    let after = status_at(&start);
    assert_eq!(after.offers, before.offers);
    assert_eq!(after.jobs.len(), 1);
    assert_eq!(after.jobs[0].arrangement, before.jobs[0].arrangement);
    assert_eq!(
        mag_mortar::Pool::replay(&ledger)
            .unwrap()
            .ledger()
            .to_lines(),
        text
    );

    // Started again on the new ledger, by this version: replayed and appended to.
    kill(&mut second);
    let (mut third, _) = again(&pool, &at);
    third.says("its 1 entries replayed");
    assert!(!pool.dir.join("ledger.jsonl.2").exists());
}

#[test]
fn a_ledger_this_version_wrote_that_does_not_replay_is_refused_and_left_as_it_is() {
    let pool = Pool::new("no-replay");
    let start = pool.start("workstation", Some(&steady_address()), None);
    let (mut first, at) = coordinator_from(Program::command(Some(&start)));
    let graph = graph_file(&pool, "sums", false);
    assert_eq!(door(&start, &["submit", path(&graph)]).code, Some(0));
    ledger_of(&pool, 1);
    kill(&mut first);
    // The submission written twice: the pool refuses the second on replay.
    let text = std::fs::read_to_string(ledger_path(&pool)).unwrap();
    let submit = text.lines().nth(1).unwrap();
    let doubled = format!("{text}{submit}\n");
    std::fs::write(ledger_path(&pool), &doubled).unwrap();

    let start = pool.start("workstation", Some(&at), None);
    let mut second = Program::start(Some(&start));
    let (status, _) = second.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(2), "{:#?}", second.said);
    second.says("is not taken up, and is left as it is");
    second.says("entry 1 is refused on replay: the pool already holds a job named `sums`");
    assert_eq!(
        std::fs::read_to_string(ledger_path(&pool)).unwrap(),
        doubled
    );
}

#[test]
fn a_member_that_joins_again_within_its_grace_keeps_its_devices_and_one_gone_past_it_leaves() {
    let pool = Pool::new("grace");
    let (mut coordinating, at) = coordinator(&pool);
    let join = |devices: &[u32]| Control::Join {
        member: member("laptop", Trust::Site, GIVE_BACK),
        offers: devices
            .iter()
            .map(|&i| offer("laptop", i, 4 << 30))
            .collect(),
    };
    let laptop = join_as(&pool, "laptop", &at, &join(&[0]));
    assert_eq!(read_ledger(&pool).0.entries.len(), 1);

    // Its channel cut without a leave: kept for its grace.
    drop(laptop);
    coordinating.says("`laptop` went without leaving");
    // Back within it, as it was: taken back in, nothing written; and still nothing once the
    // grace it had would have ended.
    let laptop = join_as(&pool, "laptop", &at, &join(&[0]));
    coordinating.says("`laptop` joined again with 1 devices, at trust site: 1 held already");
    thread::sleep(mag_mar::coordinator::LEAST_GRACE + Duration::from_secs(1));
    assert_eq!(read_ledger(&pool).0.entries.len(), 1);

    // Back with another device in place of its first: the first leaves, the other joins.
    drop(laptop);
    coordinating.says_times("`laptop` went without leaving", 2);
    let laptop = join_as(&pool, "laptop", &at, &join(&[1]));
    coordinating.says("0 held already, 1 joined, 1 left");
    assert_eq!(
        changes(&read_ledger(&pool).0)[1..],
        [
            (None, device("laptop", 0), false),
            (
                Some(member("laptop", Trust::Site, GIVE_BACK)),
                device("laptop", 1),
                true
            ),
        ]
    );

    // Gone past its grace: its device's leave recorded at its end, and not before (timed from
    // before the cut, as the coordinator's grace begins after it).
    let gone = Instant::now();
    drop(laptop);
    coordinating.says_times("`laptop` went without leaving", 3);
    let ledger = ledger_of(&pool, 4);
    let took = gone.elapsed();
    assert!(took >= Duration::from_millis(4900), "{took:?}");
    assert_eq!(changes(&ledger)[3], (None, device("laptop", 1), false));
    coordinating.says("`laptop` did not join again within its grace: its 1 devices");
}

/// The next channel the server opens to `link`, failing the test after 20 s.
fn accepted(link: &std::sync::Arc<TlsLink>) -> mag_mar::TlsChannel {
    let (tx, opened) = mpsc::channel();
    let link = link.clone();
    thread::spawn(move || {
        let _ = tx.send(link.accept(&["server"]));
    });
    opened
        .recv_timeout(Duration::from_secs(20))
        .expect("no channel opened within 20 s")
        .unwrap()
}

#[test]
fn a_member_whose_channel_is_cut_dials_again_and_joins_again() {
    let pool = Pool::new("redial");
    let link =
        std::sync::Arc::new(TlsLink::bind(pool.credentials("workstation"), "127.0.0.1:0").unwrap());
    let address = link.local_addr().unwrap().to_string();
    let start = pool.start("server", None, Some(&address));
    let mut program = Program::start(Some(&start));
    let join = |channel: &mut mag_mar::TlsChannel| match channel.recv_word::<Control>().unwrap() {
        Control::Join { offers, .. } => offers,
        other => panic!("{other:?}"),
    };

    // Cut before `Joined`, as by a coordinator that died before its line was written: it dials
    // again, soon.
    let mut channel = accepted(&link);
    let offered = join(&mut channel);
    drop(channel);
    let cut = Instant::now();
    let mut channel = accepted(&link);
    assert!(
        cut.elapsed() < Duration::from_secs(2),
        "{:?}",
        cut.elapsed()
    );
    assert_eq!(join(&mut channel), offered);
    channel.send_word(&Control::Joined).unwrap();
    program.says("joined the pool");
    // Cut after `Joined`: the same.
    drop(channel);
    let mut channel = accepted(&link);
    assert_eq!(join(&mut channel), offered);
    channel.send_word(&Control::Joined).unwrap();
    program.says_times("joined the pool", 2);
    assert!(
        program.said("it runs on, and dials the coordinator again"),
        "{:#?}",
        program.said
    );

    // Leaving, it dials no more.
    // The program has exited by the time its leave is read, so the channel is not closed from
    // here: its other end may be gone already.
    let (status, _) = program.terminate(limit() * 3);
    assert_eq!(channel.recv_word::<Control>().unwrap(), Control::Leave);
    assert_eq!(status.code(), Some(0), "{:#?}", program.said);
}

#[test]
fn a_member_whose_join_is_refused_for_good_dials_no_more() {
    let pool = Pool::new("refused-for-good");
    let link = TlsLink::bind(pool.credentials("workstation"), "127.0.0.1:0").unwrap();
    let address = link.local_addr().unwrap().to_string();
    let start = pool.start("server", None, Some(&address));
    let mut program = Program::start(Some(&start));
    let mut channel = link.accept(&["server"]).unwrap();
    assert!(matches!(
        channel.recv_word::<Control>().unwrap(),
        Control::Join { .. }
    ));
    // Closed without `Joined`: the coordinator's refusal.
    channel.close().unwrap();
    program.says("the coordinator refused its join");
    // Nothing more comes, for more than twice the longest wait between dials.
    let (tx, dialled) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(link.accept(&["server"]).is_ok());
    });
    assert!(
        dialled
            .recv_timeout(mag_mar::member::REDIAL * 2 + Duration::from_secs(1))
            .is_err()
    );
    assert!(!program.said("joined the pool"), "{:#?}", program.said);
    // It keeps its lease until SIGTERM, with no one to tell.
    let (status, took) = program.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0), "{:#?}", program.said);
    assert!(took < limit(), "{took:?}");
    assert!(program.said("no one to tell"), "{:#?}", program.said);
}

#[test]
fn a_second_coordinator_on_a_ledger_another_one_writes_is_refused() {
    let pool = Pool::new("ledger-held");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let (_first, at) = coordinator_from(Program::command(Some(&start)));
    let _laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 1);
    let (_, text) = read_ledger(&pool);
    // Its own door and address, the same ledger.
    let second = pool.dir.join("second.json");
    let mut s = Start::from_json(&std::fs::read_to_string(&start).unwrap()).unwrap();
    s.door = pool.dir.join("second.sock").to_str().unwrap().into();
    std::fs::write(&second, s.to_json()).unwrap();
    let mut refused = Program::start(Some(&second));
    let (status, _) = refused.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(2), "{:#?}", refused.said);
    refused.says("another coordinator holds its lock, and writes it still");
    assert_eq!(read_ledger(&pool).1, text);
}

#[test]
fn a_coordinator_started_again_joins_its_own_devices_as_they_are_and_records_one_not_back_as_left()
{
    let pool = Pool::new("not-back");
    let start = pool.start("workstation", Some(&steady_address()), None);
    let (mut first, at) = coordinator_from(Program::command(Some(&start)));
    let mut laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 1);
    kill(&mut first);
    // The laptop does not come back.
    laptop.child.kill().unwrap();
    laptop.child.wait().unwrap();

    // Started again with a device of its own more, and its first offering more memory.
    let mut own = description();
    own.offers[0].memory_bytes = 12 << 30;
    own.offers.push(offer("workstation", 1, 2 << 30));
    let start = pool.start_in("workstation", Some(&at), None, own);
    let restarted = Instant::now();
    let (mut second, _) = coordinator_from(Program::command(Some(&start)));
    second.says("its own devices differ from the ledger's: 2 changes");
    second.says("awaits `laptop`");
    let workstation = || Some(member("workstation", Trust::Owner, 30.0));
    assert_eq!(
        changes(&ledger_of(&pool, 3))[1..],
        [
            (workstation(), device("workstation", 0), true),
            (workstation(), device("workstation", 1), true),
        ]
    );
    // The laptop, gone past its grace from the moment the coordinator took up the ledger.
    let ledger = ledger_of(&pool, 4);
    assert!(
        restarted.elapsed() >= Duration::from_millis(4900),
        "{:?}",
        restarted.elapsed()
    );
    assert_eq!(changes(&ledger)[3], (None, device("laptop", 0), false));
    second.says("`laptop` did not join again within its grace");
    // On its notice, the leave of both its own devices.
    let (status, _) = second.terminate(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "{:#?}", second.said);
    let (ledger, text) = read_ledger(&pool);
    assert_eq!(
        changes(&ledger)[4..],
        [
            (None, device("workstation", 0), false),
            (None, device("workstation", 1), false),
        ]
    );
    assert_eq!(
        mag_mortar::Pool::replay(&ledger)
            .unwrap()
            .ledger()
            .to_lines(),
        text
    );
}

/// A join whose ledger line cannot be written is never told `Joined`: the coordinator stops with
/// status 3, and the member, told nothing, dials again. The line is made unwritable as for a
/// leave, by `prlimit`.
#[cfg(target_os = "linux")]
#[test]
fn a_join_is_not_told_joined_until_its_ledger_line_is_written() {
    use std::os::unix::process::CommandExt;

    let pool = Pool::new("join-unwritten");
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let mut command = Program::command(Some(&start));
    // SAFETY: signal(2) is async-signal-safe, and SIG_IGN is kept across exec.
    unsafe {
        command.pre_exec(|| {
            libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
            Ok(())
        });
    }
    let (mut coordinating, at) = coordinator_from(command);
    let (_, text) = read_ledger(&pool);
    let limit_to = libc::rlimit {
        rlim_cur: text.len() as libc::rlim_t,
        rlim_max: text.len() as libc::rlim_t,
    };
    // SAFETY: prlimit(2) on the child this test started, with a limit that lives in this frame.
    let set = unsafe {
        libc::prlimit(
            coordinating.child.id() as libc::pid_t,
            libc::RLIMIT_FSIZE,
            &limit_to,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(set, 0, "{}", std::io::Error::last_os_error());

    let start = pool.start_in("laptop", None, Some(&at), description());
    let mut laptop = Program::start(Some(&start));
    laptop.says("asks to join the pool");
    let (status, _) = coordinating.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(3), "{:#?}", coordinating.said);
    laptop.says("the control channel ended");
    assert!(!laptop.said("joined the pool"), "{:#?}", laptop.said);
    assert_eq!(read_ledger(&pool).1, text);
}

#[test]
fn a_member_that_cannot_reach_the_coordinator_dials_again_less_and_less_often() {
    let pool = Pool::new("backoff");
    // An address no one answers at: a port bound, then let go, where no other socket of this
    // machine's is given it meanwhile.
    let address = steady_address();
    let start = pool.start("server", None, Some(&address));
    let mut program = Program::start(Some(&start));
    program.says_times("is not reached", 3);
    let (status, _) = program.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0), "{:#?}", program.said);
    let waits: Vec<&str> = program
        .said
        .iter()
        .filter_map(|l| l.split("dialling again in ").nth(1))
        .collect();
    assert_eq!(
        waits[..3],
        ["0.25 s", "0.5 s", "1 s"],
        "{:#?}",
        program.said
    );
}
