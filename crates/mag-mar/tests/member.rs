//! The member program, run as the host runs it: a process of its own with `MAG_START` naming its
//! start file, ended by SIGTERM. A member is run first against a stand-in coordinator, which takes
//! its control channel and records what it says, or fails to answer as a test needs; then against
//! the program itself as coordinator, whose ledger the tests read and replay.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use mag_by::{
    Backend, CONTROL, Change, Control, Description, DeviceId, Dtype, Ledger, Member, Offer, Peer,
    Start, Tolerance, Trust,
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
