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
    DoorRefusal, Dtype, Ledger, Member, Offer, Peer, Place, Start, Tolerance, Trust,
};
use mag_mar::issue::Authority;
use mag_mar::{ChannelEnd, Credentials, Link, LinkError, Receiver, Sender, TlsLink};

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
            assert_eq!(
                offers,
                vec![offer("server", 0, 16 << 30), offer("server", 1, 24 << 30)]
            );
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
    let mut server = joined(&pool, "server", &at, description());
    let _laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 3);

    // SIGTERM: the leave of each of its devices is in the ledger, and the member is told, within
    // its time to give back.
    let (status, took) = server.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0), "{:#?}", server.said);
    assert!(took < limit(), "{took:?}");
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

/// What the channel hears next: a join refused ends it unclosed, never as an acknowledgement.
fn cut_short(channel: &mut mag_mar::TlsChannel) {
    match channel.recv() {
        Err(LinkError::Truncated) => {}
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

    // The server's certificate, joining as the laptop.
    let mut c = say_as(
        &pool,
        "server",
        &at,
        CONTROL,
        &join("laptop", &[("laptop", 0)]),
    );
    cut_short(&mut c);
    coordinating.says("it joins as `laptop`, and its certificate names `server`");
    // As itself, offering the laptop's device beside its own.
    let mut c = say_as(
        &pool,
        "server",
        &at,
        CONTROL,
        &join("server", &[("server", 0), ("laptop", 0)]),
    );
    cut_short(&mut c);
    coordinating.says("it offers laptop/0");
    // On a channel for a job, not the control channel. Nothing is said on it: the coordinator
    // does not read it, and a word sent may meet the cut first and fail.
    let mut c = open_as(&pool, "server", &at, "j");
    cut_short(&mut c);
    coordinating.says("no job is placed yet");
    // As the coordinator itself.
    let mut c = say_as(
        &pool,
        "workstation",
        &at,
        CONTROL,
        &join("workstation", &[("workstation", 1)]),
    );
    cut_short(&mut c);
    coordinating.says("`workstation` is in the pool already");
    assert_eq!(read_ledger(&pool).0.entries.len(), 0);

    // As itself, with its own device: taken.
    let mut held = say_as(
        &pool,
        "server",
        &at,
        CONTROL,
        &join("server", &[("server", 0)]),
    );
    ledger_of(&pool, 1);
    // Again, on a second channel while the first is live: the member is held.
    let mut c = say_as(
        &pool,
        "server",
        &at,
        CONTROL,
        &join("server", &[("server", 1)]),
    );
    cut_short(&mut c);
    coordinating.says("`server` is in the pool already");
    // Again, on the same channel.
    let mut twice = say_as(
        &pool,
        "laptop",
        &at,
        CONTROL,
        &join("laptop", &[("laptop", 0)]),
    );
    ledger_of(&pool, 2);
    twice.send_word(&join("laptop", &[("laptop", 1)])).unwrap();
    cut_short(&mut twice);
    coordinating.says("it joined already on this channel");
    drop(twice);
    // The laptop's channel gone, its device leaves; the server's one join stands.
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
    let mut bare = description();
    bare.offers.retain(|o| o.device.member != "server");
    let mut server = joined(&pool, "server", &at, bare);
    coordinating.says("`server` joined with 0 devices");
    let (status, took) = server.terminate(limit() * 3);
    assert_eq!(status.code(), Some(0));
    assert!(took < limit(), "{took:?}");
    assert!(
        server.said("left: the coordinator recorded it"),
        "{:#?}",
        server.said
    );
    coordinating.says("`server` left, its 0 devices recorded");
    assert_eq!(read_ledger(&pool).0.entries.len(), 0);
}

#[test]
fn a_member_gone_without_leaving_is_recorded_as_left() {
    let pool = Pool::new("gone");
    let (mut coordinating, at) = coordinator(&pool);
    let mut laptop = joined(&pool, "laptop", &at, description());
    ledger_of(&pool, 1);
    laptop.child.kill().unwrap();
    let ledger = ledger_of(&pool, 2);
    assert_eq!(changes(&ledger)[1], (None, device("laptop", 0), false));
    coordinating.says("`laptop` went without leaving");
}

#[test]
fn a_coordinator_does_not_overwrite_a_ledger_already_at_its_place() {
    let pool = Pool::new("ledger-there");
    let path = pool.dir.join("ledger.jsonl");
    std::fs::write(&path, "kept\n").unwrap();
    let start = pool.start("workstation", Some("127.0.0.1:0"), None);
    let mut program = Program::start(Some(&start));
    let (status, _) = program.exits(Duration::from_secs(10));
    assert_eq!(status.code(), Some(2));
    program.says("cannot be begun");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "kept\n");
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
    // A line longer than the door reads, refused once it has read that much.
    let mut stream = std::os::unix::net::UnixStream::connect(pool.dir.join("door.sock")).unwrap();
    let long = vec![b' '; mag_mar::door::REQUEST_LIMIT as usize + 2];
    let writing = {
        let mut stream = stream.try_clone().unwrap();
        thread::spawn(move || {
            let _ = stream.write_all(&long);
        })
    };
    let mut answer = String::new();
    std::io::Read::read_to_string(&mut stream, &mut answer).unwrap();
    writing.join().unwrap();
    match serde_json::from_str::<DoorAnswer>(&answer).unwrap() {
        DoorAnswer::Refused {
            refusal: DoorRefusal::Unreadable { error },
        } => assert!(error.contains("longer than"), "{error}"),
        other => panic!("{other:?}"),
    }

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
