//! The coordinator: what the member the start file names as coordinator runs, in the member
//! program, in place of dialling one.
//!
//! It keeps the pool's state in a `mag_mortar::Pool`, and the pool's ledger in the file its start
//! file names (`ledger`), and takes its members' joins and leaves into both:
//!
//! 1. **It begins a ledger, or takes one up** ([`take_up`]). With nothing at the ledger's place,
//!    the pool begins with the coordinator's own member and devices, at the time it starts, and
//!    the ledger's header is written and synced before any member is taken. A ledger already
//!    there is one a coordinator left when it stopped, and this one takes it up:
//!    - written by this version of `mag-mortar`, the pool is replayed from it
//!      (`mag_mortar::Pool::replay`), each record checked against the one logged, and the ledger
//!      is appended to. A record that differs, or a change the pool refuses, is a defect: the
//!      coordinator does not start, and says which entry;
//!    - written by another version, or by one that did not say, it is read for its facts alone.
//!      It is copied beside its place, at the first free name with `.1`, `.2`, ..., added, and a
//!      new ledger takes its place, whose header names the copy as the ledger it follows
//!      (`mag_mortar::Pool::follow`), with the jobs it held submitted again;
//!    - a last line cut short was never acknowledged: it is dropped, and said, and the file is
//!      cut back to its whole lines before anything is appended. A ledger refused otherwise is
//!      left as it is, and the coordinator does not start.
//!
//!    Its own devices then join again as a member's do (4), and every other member the pool holds
//!    a device of is awaited for its grace (6).
//! 2. **It takes a control channel from any member of the pool** (`Link::accept_any`): whoever
//!    holds a certificate the pool's authority signed naming one member, since a member that joins
//!    later is named by nothing the coordinator was told. A channel for a job is refused: no job
//!    is placed yet.
//! 3. **It checks a join** ([`check_join`]): the member it names and every offer's member must be
//!    the member the certificate names, and that member may hold no other live control channel,
//!    nor be the coordinator itself. A join refused for good, as another member's or the
//!    coordinator's, is answered by closing the coordinator's way of the channel without
//!    `mag_by::Control::Joined`, so that the member dials no more. One refused for now, the
//!    member holding another channel the coordinator has not yet found ended, or a second join on
//!    one channel, is cut short unclosed, and the member dials again.
//! 4. **It records a join** ([`join_changes`]): a `mag_by::Change::Join` for each device offered
//!    that the pool does not hold as it is offered, the member as [`recorded_member`] makes it;
//!    a `mag_by::Change::Leave` for each device of the member the pool holds and the join does not
//!    offer; and nothing for a device the pool holds as offered. So a member that comes back is
//!    taken back in without its devices joining twice. The changes are applied to the pool, the
//!    new lines written to the ledger and synced, and only then the coordinator says
//!    `mag_by::Control::Joined`. A member with no offers is recorded nowhere, since the pool is
//!    told of devices, not of members: it holds its channel, is told it joined, and leaves as any
//!    other.
//! 5. **It records a leave**: on `mag_by::Control::Leave`, one `mag_by::Change::Leave` per device
//!    the member joined with, the lines written and synced, and only then the coordinator closes
//!    its way of the channel. That close is the acknowledgement the member waits for, within its
//!    `give_back_seconds`.
//! 6. **It awaits a member gone** for its [`grace`]: a member whose control channel ended without
//!    `Leave`, or whose devices the ledger it took up holds. A member that joins again within the
//!    grace keeps its devices in the pool, as in 4; at the grace's end, the leave of each of its
//!    devices the pool still holds is recorded, and nothing is acknowledged.
//! 7. **It serves the door** ([`crate::door`]), at the start file's `door`, opened before the
//!    ledger is begun or taken up, so that a coordinator refused its door leaves no ledger behind.
//!    A command there submits a job with its stamp (`mag_by::Change::Submit`), ends one
//!    (`mag_by::Change::End`), or asks the pool's status (`mag_mortar::Pool::describe`), with
//!    what the pool asks of its host at that moment (`mag_mortar::Pool::asks`): the kind of
//!    device it most wants while jobs wait, and the members it would give back after the start
//!    file's `idle_seconds`, never the coordinator's own. The host polls the status, and decides;
//!    the coordinator calls no one. A submission or an end is applied to the pool, and its line
//!    written to the ledger and synced, before it is answered; one the pool refuses is answered
//!    with why, and nothing is written. Ending a job the pool does not hold is refused by the
//!    pool (`mag_mortar::ChangeError::NoJob`), and so at the door, rather than written as an end
//!    of nothing.
//! 8. **It takes a device back by need**, for a take-back at the door: a kind of device within a
//!    deadline. The pool chooses the device ([`mag_mortar::Pool::give_up`]): of those of the kind
//!    whose members give a device back within the deadline, the one whose loss costs it least,
//!    the coordinator's own last. Its `mag_by::Change::Leave` is applied and written to the
//!    ledger and synced first, so nothing is placed on it from then on; then its member is told
//!    on its control channel (`mag_by::Control::GiveBack`), and the door is answered with the
//!    device once the member says it gave it back (`mag_by::Control::GaveBack`), or once the
//!    deadline passes, the device being the host's then whatever the member says. A device of
//!    the coordinator's own it stops offering itself. A need nothing meets is refused with the
//!    devices of the kind too slow to give one back, and nothing is written; a take-back by
//!    name is refused, since it is the host's SIGTERM.
//! 9. **On its own notice** it records its own devices' leave the same way, and returns. Its
//!    members' channels end without a close: their leave is not recorded by this coordinator,
//!    and a coordinator that takes up its ledger awaits them.
//!
//! **One thread keeps the pool.** The members' channels and the door's connections are each heard
//! on a thread of their own, and each hands what it hears to the one thread that holds the pool
//! and the ledger, as an event, and the door's waits for its answer. So the pool takes one change
//! at a time, in the order the events come, and the ledger's lines are written in that order,
//! each synced before the next change is taken: a submission while members join is one more
//! event between theirs.
//!
//! A ledger line that cannot be written stops the coordinator at once, with nothing after the
//! last line written acknowledged or answered.
//!
//! Not built yet: taking a job's channel; announcing a member to its peers; sending placements;
//! refusing a member's old certificate by its serial; and, once parts run, a member cut off from
//! the coordinator for its grace stopping them, so that the pool placing them again at the
//! grace's end never has a part live in two places.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mag_by::{
    CONTROL, Change, Control, Description, DeviceId, DoorAnswer, DoorRefusal, DoorRequest, Ledger,
    LedgerError, Member, Offer, Record, TakeBack, Trust,
};
use mag_mortar::{ChangeError, Pool, ReplayError};

use crate::member::{Ended, MemberError, Notice, give_back_by, say};
use crate::{Channel, TlsSender};
use crate::{ChannelEnd, Identity, Link, LinkError, Receiver, Sender, TlsLink, TlsReceiver};

/// The member a join is recorded as: the member the coordinator's start file lists under that
/// name, at the lower of the two trusts and the shorter of the two times to give a device back,
/// or, for a member the coordinator's start file does not list, the member as it says it is, but
/// trusted at [`Trust::Outside`], whatever it says.
///
/// Not settled: how far a member that joins after the coordinator began is to be trusted. A
/// member the coordinator does not list is known only by its own join, and a machine holding a
/// certificate the host meant for `outside` could say `owner`. One answer is that the host writes
/// the trust, and the time to give a device back, into the certificate it signs, and the
/// coordinator reads them from there and believes nothing the member says; the other is that
/// such a member is trusted at `outside` and no further. Until it is settled, this takes the
/// second, the one that can place nothing kept above `outside` where it should not go. Either
/// answer changes this function alone.
pub fn recorded_member(listed: &[Member], said: &Member) -> Member {
    match listed.iter().find(|m| m.name == said.name) {
        Some(known) => Member {
            name: said.name.clone(),
            trust: known.trust.min(said.trust),
            give_back_seconds: known.give_back_seconds.min(said.give_back_seconds),
        },
        None => Member {
            name: said.name.clone(),
            trust: Trust::Outside,
            give_back_seconds: said.give_back_seconds,
        },
    }
}

/// The least grace a member gone without leaving is given: more than twice the longest wait
/// between a member's dials ([`crate::member::REDIAL`]), so that a member waiting that long to
/// dial again as the coordinator returns still comes back within its grace, its opening too.
pub const LEAST_GRACE: Duration = Duration::from_secs(5);

/// How long the coordinator keeps a member's devices in the pool for it to come back, once its
/// control channel ended without `Leave`, or once the coordinator took up a ledger that holds
/// them: the member's `give_back_seconds`, as the pool holds them, and at least [`LEAST_GRACE`];
/// `None` for a time too long to count, which is awaited for ever.
///
/// The `give_back_seconds` are what the host already gives the member to let its devices go
/// after its notice, so the pool keeps a device it has lost touch with no longer than the host
/// would wait for it; and a member slow to give back is slow to come back too. A member that
/// comes back within its grace keeps its devices, so a coordinator restarted, or a channel cut
/// and dialled again, moves nothing.
///
/// What this leaves for when parts run: a member cut off from the coordinator runs its parts on,
/// and at the grace's end the pool places them again elsewhere. A part must never be live on two
/// devices at once, so the member must stop its parts at a step, and write their state, once it
/// has been cut off for as long, before the pool places them again; and the member's count must
/// not start later than the coordinator's.
pub fn grace(give_back_seconds: f64) -> Option<Duration> {
    Duration::try_from_secs_f64(give_back_seconds)
        .ok()
        .map(|d| d.max(LEAST_GRACE))
}

/// Why a join was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JoinRefused {
    /// The join names another member than the certificate.
    NotItself { said: String, certificate: String },
    /// An offer is for a device of another member than the certificate names.
    NotItsDevice {
        device: DeviceId,
        certificate: String,
    },
    /// The member holds another live control channel to the coordinator.
    Held(String),
    /// The member is the coordinator itself.
    Coordinator(String),
    /// The channel has joined already.
    Twice,
}

impl JoinRefused {
    /// Whether the refusal holds for good, whatever the member does: a join of another member's
    /// name or device, or of the coordinator's. Such a join is answered by closing the channel,
    /// and the member dials no more. Otherwise the channel is cut short, and the member dials
    /// again: a channel the coordinator holds of the same member may be one already gone, which
    /// it has not yet found ended.
    pub fn lasting(&self) -> bool {
        match self {
            JoinRefused::NotItself { .. }
            | JoinRefused::NotItsDevice { .. }
            | JoinRefused::Coordinator(_) => true,
            JoinRefused::Held(_) | JoinRefused::Twice => false,
        }
    }
}

impl fmt::Display for JoinRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JoinRefused::NotItself { said, certificate } => write!(
                f,
                "it joins as `{said}`, and its certificate names `{certificate}`"
            ),
            JoinRefused::NotItsDevice {
                device,
                certificate,
            } => write!(
                f,
                "it offers {device}, a device of another member than `{certificate}`, whom its \
                 certificate names"
            ),
            JoinRefused::Held(member) => {
                write!(f, "`{member}` is in the pool already, on another channel")
            }
            JoinRefused::Coordinator(member) => write!(f, "`{member}` is the coordinator"),
            JoinRefused::Twice => write!(f, "it joined already on this channel"),
        }
    }
}

/// Whether the member `certificate` names may join as `member`, with `offers`: the member and
/// every offer's device must be its own. Whether it is held already is the caller's to check.
pub fn check_join(certificate: &str, member: &Member, offers: &[Offer]) -> Result<(), JoinRefused> {
    if member.name != certificate {
        return Err(JoinRefused::NotItself {
            said: member.name.clone(),
            certificate: certificate.into(),
        });
    }
    match offers.iter().find(|o| o.device.member != certificate) {
        Some(o) => Err(JoinRefused::NotItsDevice {
            device: o.device.clone(),
            certificate: certificate.into(),
        }),
        None => Ok(()),
    }
}

/// What a join changes, `pool` holding what it holds: the member as recorded joins with
/// `offers`. A `Change::Leave` for each device of the member the pool holds that the join does not
/// offer, then a `Change::Join` for each device offered that the pool does not hold as offered;
/// for every device offered, if the member is recorded at a lower trust or a shorter time to give
/// a device back than the pool holds it at, so that the pool lowers it. A member coming back as
/// it was changes nothing, and its devices do not join twice.
pub fn join_changes(pool: &Description, member: &Member, offers: &[Offer]) -> Vec<Change> {
    let known = pool.members.iter().find(|m| m.name == member.name);
    let stricter = known
        .is_none_or(|k| member.trust < k.trust || member.give_back_seconds < k.give_back_seconds);
    let held = pool
        .offers
        .iter()
        .filter(|o| o.device.member == member.name);
    let gone = held
        .filter(|h| !offers.iter().any(|o| o.device == h.device))
        .map(|h| Change::Leave {
            device: h.device.clone(),
        });
    let new = offers
        .iter()
        .filter(|o| stricter || !pool.offers.contains(o))
        .map(|offer| Change::Join {
            member: Some(member.clone()),
            offer: offer.clone(),
        });
    gone.chain(new).collect()
}

/// The devices of `member` the pool holds.
fn held_by(pool: &Pool, member: &str) -> Vec<DeviceId> {
    pool.describe()
        .offers
        .into_iter()
        .map(|o| o.device)
        .filter(|d| d.member == member)
        .collect()
}

/// How the coordinator came by its ledger.
#[derive(Debug)]
pub enum Begun {
    /// Nothing was at its place: it began a ledger there.
    New,
    /// This version of `mag-mortar` wrote the ledger there: replayed, its `entries` each checked,
    /// and appended to.
    Replayed { entries: usize },
    /// Another version wrote the ledger there, `written_by`, or one that did not say: read for its
    /// facts, copied to `moved`, and a new ledger begun in its place following it, with each job
    /// it held submitted again but those `refused`, each with why.
    Followed {
        written_by: Option<String>,
        moved: String,
        jobs: usize,
        refused: Vec<(String, ChangeError)>,
    },
}

/// Why a ledger at the coordinator's place was not taken up. It is left as it is.
#[derive(Debug)]
pub enum TakeUpError {
    /// Another coordinator holds the ledger's lock: it writes the ledger still.
    Held,
    /// Reading it, cutting it back to its whole lines, or writing the ledger that follows it.
    Io(io::Error),
    /// Its whole lines are not UTF-8 text.
    NotText,
    /// Its lines are refused: not a ledger, a version this reader does not read, a line that is
    /// not one, or a line cut short that is not the last.
    Lines(LedgerError),
    /// This version wrote it, and it does not replay.
    Replay(ReplayError),
}

impl fmt::Display for TakeUpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TakeUpError::Held => write!(
                f,
                "another coordinator holds its lock, and writes it still (the ledger's path with \
                 `.lock` added)"
            ),
            TakeUpError::Io(e) => write!(f, "{e}"),
            TakeUpError::NotText => write!(f, "its lines are not UTF-8 text"),
            TakeUpError::Lines(e) => write!(f, "it is not read as a ledger: {e}"),
            TakeUpError::Replay(e) => write!(
                f,
                "this version of mag-mortar ({}) wrote it, and it does not replay: {e}; a \
                 ledger this version wrote must replay, so this is a defect, or the ledger was \
                 changed",
                mag_mortar::VERSION
            ),
        }
    }
}

impl std::error::Error for TakeUpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TakeUpError::Io(e) => Some(e),
            TakeUpError::Lines(e) => Some(e),
            TakeUpError::Replay(e) => Some(e),
            TakeUpError::NotText | TakeUpError::Held => None,
        }
    }
}

impl From<io::Error> for TakeUpError {
    fn from(e: io::Error) -> Self {
        TakeUpError::Io(e)
    }
}

/// A ledger begun or taken up, by [`take_up`].
pub struct TakenUp {
    pub pool: Pool,
    pub file: LedgerFile,
    pub begun: Begun,
    /// The number of a last line that was cut short, and dropped.
    pub dropped: Option<usize>,
}

/// The pool and the ledger file at `path`, at `at`: a new ledger begun with `me` and `offers` if
/// nothing is there, or the ledger there taken up, with the number of a last line cut short that
/// was dropped. See [`Begun`]. A ledger refused is left as it is; a ledger begun is refused if
/// something comes to its place meanwhile.
///
/// On Unix the ledger's lock, a file at `path` with `.lock` added, never removed, is taken first,
/// and held for as long as the ledger file is: a ledger whose lock another coordinator holds is
/// refused, so that two coordinators never write one ledger.
pub fn take_up(
    path: &str,
    me: Member,
    offers: Vec<Offer>,
    at: f64,
) -> Result<TakenUp, MemberError> {
    let failed = |error| MemberError::TakeUp {
        path: path.into(),
        error,
    };
    #[cfg(unix)]
    let lock = lock(path).map_err(failed)?;
    let mut taken = take_up_locked(path, me, offers, at)?;
    #[cfg(unix)]
    {
        taken.file._lock = Some(lock);
    }
    #[cfg(not(unix))]
    let _ = &mut taken;
    Ok(taken)
}

/// Takes the lock of the ledger at `path`, unless another process holds it.
#[cfg(unix)]
fn lock(path: &str) -> Result<File, TakeUpError> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;

    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(format!("{path}.lock"))?;
    // SAFETY: flock(2) on a descriptor this function owns.
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let e = io::Error::last_os_error();
        return Err(if e.kind() == io::ErrorKind::WouldBlock {
            TakeUpError::Held
        } else {
            TakeUpError::Io(e)
        });
    }
    Ok(lock)
}

fn take_up_locked(
    path: &str,
    me: Member,
    offers: Vec<Offer>,
    at: f64,
) -> Result<TakenUp, MemberError> {
    let failed = |error| MemberError::TakeUp {
        path: path.into(),
        error,
    };
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let pool = Pool::from_description(&Description {
                at,
                members: vec![me],
                offers,
                jobs: Vec::new(),
                formats: Vec::new(),
            })
            .expect("a description with no jobs");
            let file =
                LedgerFile::create(path, pool.ledger()).map_err(|error| MemberError::Ledger {
                    path: path.into(),
                    error,
                })?;
            return Ok(TakenUp {
                pool,
                file,
                begun: Begun::New,
                dropped: None,
            });
        }
        Err(e) => return Err(failed(e.into())),
    };
    // A last line cut short is the part after the last newline; only it may be other than text.
    let whole = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    let mut text = std::str::from_utf8(&bytes[..whole])
        .map_err(|_| failed(TakeUpError::NotText))?
        .to_string();
    text.push_str(&String::from_utf8_lossy(&bytes[whole..]));
    let read = Ledger::from_lines(&text).map_err(|e| failed(TakeUpError::Lines(e)))?;
    let old = read.ledger;
    if old.written_by.as_deref() == Some(mag_mortar::VERSION) {
        let pool = Pool::replay(&old).map_err(|e| failed(TakeUpError::Replay(e)))?;
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| failed(e.into()))?;
        if read.dropped.is_some() {
            file.set_len(whole as u64).map_err(|e| failed(e.into()))?;
            file.sync_all().map_err(|e| failed(e.into()))?;
        }
        let entries = old.entries.len();
        let file = LedgerFile {
            path: path.into(),
            file,
            written: entries,
            _lock: None,
        };
        return Ok(TakenUp {
            pool,
            file,
            begun: Begun::Replayed { entries },
            dropped: read.dropped,
        });
    }
    let moved = (1..)
        .map(|n| format!("{path}.{n}"))
        .find(|p| !Path::new(p).exists())
        .expect("a free name");
    let (pool, refused) = Pool::follow(&old, &moved, at);
    let jobs = pool.ledger().entries.len();
    let file = replace(path, &moved, pool.ledger()).map_err(|e| failed(e.into()))?;
    Ok(TakenUp {
        pool,
        file,
        begun: Begun::Followed {
            written_by: old.written_by,
            moved,
            jobs,
            refused,
        },
        dropped: read.dropped,
    })
}

/// Puts `ledger` in the place of the ledger at `path`, which it follows, as `moved`: the old one
/// copied to `moved` and synced, the new one written and synced beside it, at `path` with `.new`
/// added, then renamed into its place. Stopped at any point, the old ledger is still at `path`,
/// and a copy of it or a `.new` left over is harmless.
fn replace(path: &str, moved: &str, ledger: &Ledger) -> io::Result<LedgerFile> {
    std::fs::copy(path, moved)?;
    File::open(moved)?.sync_all()?;
    let new = format!("{path}.new");
    match std::fs::remove_file(&new) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    let mut file = LedgerFile::create(&new, ledger)?;
    std::fs::rename(&new, path)?;
    sync_directory(path)?;
    file.path = path.into();
    Ok(file)
}

/// Syncs the directory `path` is in, so that a name made or changed in it is found after a crash.
fn sync_directory(path: &str) -> io::Result<()> {
    #[cfg(unix)]
    if let Some(dir) = Path::new(path).parent() {
        let dir = if dir.as_os_str().is_empty() {
            Path::new(".")
        } else {
            dir
        };
        File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// The ledger's file: written as the pool takes each change, and synced before the change is
/// acknowledged.
pub struct LedgerFile {
    path: String,
    file: File,
    /// The ledger's entries written so far.
    written: usize,
    /// The ledger's lock, held while the file is, where [`take_up`] took it.
    _lock: Option<File>,
}

impl LedgerFile {
    /// Begins the ledger at `path`, a new file: its header, and any entries it holds, synced. A
    /// file already there is refused, and left as it is.
    pub fn create(path: &str, ledger: &Ledger) -> io::Result<LedgerFile> {
        let file = OpenOptions::new()
            .append(true)
            .create_new(true)
            .open(path)?;
        let mut this = LedgerFile {
            path: path.into(),
            file,
            written: 0,
            _lock: None,
        };
        this.file.write_all(ledger.header_line().as_bytes())?;
        this.write(ledger)?;
        // The file's name in its directory, so that the new file is found after a crash.
        sync_directory(path)?;
        Ok(this)
    }

    /// Writes the entries of `ledger` not yet written, each as its line, and syncs them. Returns
    /// once they are on the disk, as far as the system can say.
    pub fn write(&mut self, ledger: &Ledger) -> io::Result<()> {
        let mut text = String::new();
        for e in &ledger.entries[self.written..] {
            text.push_str(&Ledger::line(e));
        }
        self.file.write_all(text.as_bytes())?;
        self.file.sync_data()?;
        self.written = ledger.entries.len();
        Ok(())
    }

    /// Where it is.
    pub fn path(&self) -> &str {
        &self.path
    }
}

/// Seconds since 1970, the pool's time.
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// What the coordinator's one waiting thread is told.
enum Event {
    Notice,
    /// A control channel opened: the member at its other end, and its sending half.
    Opened {
        id: u64,
        peer: Identity,
        sending: TlsSender,
    },
    /// A word on a control channel.
    Said {
        id: u64,
        word: Control,
    },
    /// A control channel ended, as its receiving half found it.
    Ended {
        id: u64,
        error: LinkError,
    },
    /// A request at the door, and where its answer goes.
    Door {
        request: DoorRequest,
        reply: mpsc::Sender<DoorAnswer>,
    },
}

/// A device the coordinator told its member to give back, for a take-back at the door, and the
/// door's answer, which waits until the member says it has or the deadline passes.
struct Giving {
    /// The member's control channel.
    id: u64,
    device: DeviceId,
    /// The deadline, unless it is too far to be kept as an instant.
    by: Option<Instant>,
    /// The record of the device's leave, in the ledger already.
    record: Record,
    reply: mpsc::Sender<DoorAnswer>,
}

impl Giving {
    /// Answers the door: the device is taken back, given back or not.
    fn answer(self, name: &str, given_back: bool) {
        let answer = DoorAnswer::TakenBack {
            device: self.device,
            record: self.record,
            given_back,
        };
        say(name, format_args!("the door: {}", told(&answer)));
        let _ = self.reply.send(answer);
    }
}

/// What the coordinator does with a request at the door, once what it changes is in the ledger.
enum Answering {
    /// Answers it now.
    Now(DoorAnswer),
    /// Tells the member holding the device chosen for a take-back to give it back, or stops
    /// offering it if it is the coordinator's own; then answers.
    GiveBack {
        device: DeviceId,
        deadline_seconds: f64,
        record: Record,
    },
}

/// A member's control channel, as the coordinator holds it.
struct Attached {
    peer: Identity,
    /// Its sending half, until it is closed or dropped.
    sending: Option<TlsSender>,
    /// The devices it joined with, once it has joined: none for a member with no offers.
    joined: Option<Vec<DeviceId>>,
}

/// Takes channels on a thread of its own, and hands each control channel on.
fn take(link: TlsLink, name: String, events: mpsc::Sender<Event>) {
    thread::spawn(move || {
        let mut id = 0;
        loop {
            match link.accept_any() {
                Ok(channel) if channel.job() == CONTROL => {
                    id += 1;
                    let (sending, receiving) = channel.split();
                    let peer = sending.peer().clone();
                    if events.send(Event::Opened { id, peer, sending }).is_err() {
                        return;
                    }
                    hear(id, receiving, events.clone());
                }
                Ok(channel) => say(
                    &name,
                    format_args!(
                        "refuses a channel from `{}` for the job `{}`: no job is placed yet",
                        channel.peer().member,
                        channel.job()
                    ),
                ),
                Err(LinkError::Io(e)) => {
                    say(&name, format_args!("takes no channel ({e})"));
                    thread::sleep(Duration::from_millis(100));
                }
                Err(e) => say(&name, format_args!("a channel was not taken ({e})")),
            }
        }
    });
}

/// Hears a member on its control channel's receiving half, on a thread of its own.
fn hear(id: u64, mut receiving: TlsReceiver, events: mpsc::Sender<Event>) {
    thread::spawn(move || {
        loop {
            match receiving.recv_word::<Control>() {
                Ok(word) => {
                    if events.send(Event::Said { id, word }).is_err() {
                        return;
                    }
                }
                Err(error) => {
                    let _ = events.send(Event::Ended { id, error });
                    return;
                }
            }
        }
    });
}

/// The coordinator's pool, its ledger file, and what it says.
struct Keeper {
    name: String,
    pool: Pool,
    ledger: LedgerFile,
    /// The start file's idle time, after which the pool offers a member back.
    idle_seconds: f64,
}

impl Keeper {
    /// Applies `changes` to the pool, then writes and syncs their lines. Only once this returns
    /// may they be acknowledged.
    fn record(&mut self, changes: Vec<Change>) -> Result<(), MemberError> {
        if changes.is_empty() {
            return Ok(());
        }
        for change in changes {
            let what = describe(&change);
            if let Err(e) = self.pool.apply(change, now()) {
                say(&self.name, format_args!("the pool refused {what} ({e})"));
            }
        }
        self.ledger
            .write(self.pool.ledger())
            .map_err(|error| MemberError::LedgerUnwritten {
                path: self.ledger.path().into(),
                error,
            })
    }

    /// Applies one change to the pool, and if the pool takes it, writes and syncs its line: the
    /// record of the loop's run on it, or why the pool refused it, which writes nothing. Only once
    /// this returns may it be answered.
    fn take(&mut self, change: Change) -> Result<Result<Record, ChangeError>, MemberError> {
        let record = match self.pool.apply(change, now()) {
            Ok(record) => record.clone(),
            Err(e) => return Ok(Err(e)),
        };
        self.ledger
            .write(self.pool.ledger())
            .map_err(|error| MemberError::LedgerUnwritten {
                path: self.ledger.path().into(),
                error,
            })?;
        Ok(Ok(record))
    }

    /// The answer to a request at the door, once what it changes is in the ledger; for a
    /// take-back, the device chosen, whose leave is in the ledger, and which its member is to
    /// be told to give back before the door is answered.
    fn door(&mut self, request: DoorRequest) -> Result<Answering, MemberError> {
        let refused = |e: ChangeError| DoorAnswer::Refused {
            refusal: DoorRefusal::Pool {
                reason: e.to_string(),
            },
        };
        Ok(Answering::Now(match request {
            DoorRequest::Submit { graph, stamp } => {
                let job = graph.job.name.clone();
                match self.take(Change::Submit { graph, stamp })? {
                    Ok(record) => DoorAnswer::Submitted {
                        record,
                        arrangement: self
                            .pool
                            .arrangement(&job)
                            .expect("the pool holds a job it took")
                            .clone(),
                    },
                    Err(e) => refused(e),
                }
            }
            DoorRequest::End { job } => match self.take(Change::End { job })? {
                Ok(record) => DoorAnswer::Ended { record },
                Err(ChangeError::NoJob { job }) => DoorAnswer::Refused {
                    refusal: DoorRefusal::NoJob { job },
                },
                Err(e) => refused(e),
            },
            DoorRequest::Status => DoorAnswer::Status {
                pool: self.pool.describe(),
                asks: Some(self.pool.asks(&self.name, self.idle_seconds, now())),
            },
            DoorRequest::TakeBack(TakeBack::Need {
                kind,
                deadline_seconds,
            }) => {
                let device = match self
                    .pool
                    .give_up(&self.name, &kind, deadline_seconds, now())
                {
                    Ok(chosen) => chosen.device,
                    Err(unmet) => {
                        return Ok(Answering::Now(DoorAnswer::Refused {
                            refusal: DoorRefusal::Unmet { slow: unmet.slow },
                        }));
                    }
                };
                match self.take(Change::Leave {
                    device: device.clone(),
                })? {
                    Ok(record) => {
                        return Ok(Answering::GiveBack {
                            device,
                            deadline_seconds,
                            record,
                        });
                    }
                    Err(e) => refused(e),
                }
            }
            DoorRequest::TakeBack(TakeBack::Device { .. } | TakeBack::Member { .. }) => {
                DoorAnswer::Refused {
                    refusal: DoorRefusal::ByName,
                }
            }
        }))
    }
}

/// What a door's answer says, for the coordinator's own lines.
fn told(answer: &DoorAnswer) -> String {
    match answer {
        DoorAnswer::Submitted { arrangement, .. } => format!(
            "`{}` submitted, written, and {}",
            arrangement.job,
            if answer.placed() == Some(true) {
                "placed"
            } else {
                "waits"
            }
        ),
        DoorAnswer::Ended { record } => match &record.event {
            mag_by::Event::End { job } => format!("`{job}` ended, written"),
            other => format!("{other:?} written"),
        },
        DoorAnswer::Status { .. } => "status given".into(),
        DoorAnswer::TakenBack {
            device,
            given_back: true,
            ..
        } => format!("{device} taken back, written, and given back"),
        DoorAnswer::TakenBack { device, .. } => format!(
            "{device} taken back and written; its member did not say it gave it back before the \
             deadline, and it is the host's now"
        ),
        DoorAnswer::Refused { refusal } => format!("refused: {refusal}"),
    }
}

fn describe(change: &Change) -> String {
    match change {
        Change::Join { offer, .. } => format!("the join of {}", offer.device),
        Change::Leave { device } => format!("the leave of {device}"),
        Change::Submit { graph, .. } => format!("the job `{}`", graph.job.name),
        Change::End { job } => format!("the end of `{job}`"),
    }
}

/// When the grace of `member` ends, counted from now: see [`grace`]. A member the pool was not
/// told of, which only a device's offer names, is given the least.
fn grace_of(pool: &Description, member: &str) -> Option<Instant> {
    let give_back = pool
        .members
        .iter()
        .find(|m| m.name == member)
        .map_or(0.0, |m| m.give_back_seconds);
    grace(give_back).and_then(|g| Instant::now().checked_add(g))
}

/// How long until `by`, for the coordinator's lines.
fn within(by: Option<Instant>) -> String {
    match by {
        Some(by) => format!(
            "its grace of {:.1} s",
            by.saturating_duration_since(Instant::now()).as_secs_f64()
        ),
        None => "ever, its grace too long to count".into(),
    }
}

fn leaves(devices: &[DeviceId]) -> Vec<Change> {
    devices
        .iter()
        .map(|device| Change::Leave {
            device: device.clone(),
        })
        .collect()
}

/// Runs the coordinator on `link`, the coordinator's member being `me`, with `offers`, until its
/// notice. The ledger is begun before anything else, and a ledger that cannot be begun is an
/// error, as a start file that cannot be read is.
pub(crate) fn run(
    start: &mag_by::Start,
    me: Member,
    offers: Vec<Offer>,
    link: TlsLink,
    notice: Notice,
    give_back: Duration,
) -> Result<Ended, MemberError> {
    let name = me.name.clone();
    let mut mine: Vec<DeviceId> = offers.iter().map(|o| o.device.clone()).collect();
    // The door first: a coordinator refused its door begins no ledger.
    #[cfg(unix)]
    let door = crate::door::Door::open(&start.door).map_err(|error| MemberError::Door {
        path: start.door.clone(),
        error,
    })?;
    let TakenUp {
        pool,
        file: ledger,
        begun,
        dropped,
    } = take_up(&start.ledger, me.clone(), offers.clone(), now())?;
    if let Some(line) = dropped {
        say(
            &name,
            format_args!(
                "the ledger {}'s last line, line {line}, was cut short and so never \
                 acknowledged: dropped",
                start.ledger
            ),
        );
    }
    let path = &start.ledger;
    match &begun {
        Begun::New => say(
            &name,
            format_args!(
                "coordinates: the ledger is {path}, begun with {} devices of its own",
                mine.len()
            ),
        ),
        Begun::Replayed { entries } => say(
            &name,
            format_args!(
                "coordinates: takes up the ledger {path}, which this version wrote: its {entries} \
                 entries replayed, each record as logged, and appended to"
            ),
        ),
        Begun::Followed {
            written_by,
            moved,
            jobs,
            refused,
        } => {
            say(
                &name,
                format_args!(
                    "coordinates: the ledger {path} was written by {}, and this is mag-mortar {}: \
                     read for its facts, kept as {moved}, and a new ledger begun in its place \
                     that follows it, with {jobs} jobs it held submitted again",
                    match written_by {
                        Some(v) => format!("mag-mortar {v}"),
                        None => "a version that does not say".into(),
                    },
                    mag_mortar::VERSION
                ),
            );
            for (job, why) in refused {
                say(
                    &name,
                    format_args!("refuses the job `{job}` the old ledger held: {why}"),
                );
            }
        }
    }
    let mut keeper = Keeper {
        name: name.clone(),
        pool,
        ledger,
        idle_seconds: start.idle_seconds,
    };
    // Its own devices join again, as a member's do: nothing is written for a device the pool holds
    // as offered.
    let own = join_changes(&keeper.pool.describe(), &me, &offers);
    if !own.is_empty() {
        say(
            &name,
            format_args!(
                "its own devices differ from the ledger's: {} changes",
                own.len()
            ),
        );
    }
    keeper.record(own)?;
    // Every other member the pool holds a device of is awaited for its grace.
    let mut away: BTreeMap<String, Option<Instant>> = BTreeMap::new();
    let described = keeper.pool.describe();
    for m in described.offers.iter().map(|o| &o.device.member) {
        if *m != name && !away.contains_key(m) {
            let by = grace_of(&described, m);
            say(
                &name,
                format_args!(
                    "awaits `{m}`, whose devices the ledger holds, for {}",
                    within(by)
                ),
            );
            away.insert(m.clone(), by);
        }
    }

    let (events_tx, events) = mpsc::channel();
    {
        let events = events_tx.clone();
        give_back_by(notice, give_back, name.clone(), move || {
            let _ = events.send(Event::Notice);
        });
    }
    #[cfg(unix)]
    {
        let events = events_tx.clone();
        crate::door::serve(&door, name.clone(), move |request| {
            let (reply, answer) = mpsc::channel();
            events.send(Event::Door { request, reply }).ok()?;
            answer.recv().ok()
        })
        .map_err(|error| MemberError::Door {
            path: start.door.clone(),
            error: crate::door::DoorError::Io(error),
        })?;
        say(&name, format_args!("serves the door at {}", start.door));
    }
    take(link, name.clone(), events_tx);

    let mut members: BTreeMap<u64, Attached> = BTreeMap::new();
    let mut giving: Vec<Giving> = Vec::new();
    loop {
        // The next event, or the first deadline: of a device being given back, or of a member's
        // grace.
        let first = giving
            .iter()
            .filter_map(|g| g.by)
            .chain(away.values().flatten().copied())
            .min();
        let event = match first {
            None => events.recv().expect("the taking thread holds a sender"),
            Some(by) => match events.recv_timeout(by.saturating_duration_since(Instant::now())) {
                Ok(event) => event,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let now = Instant::now();
                    let (passed, waiting) = giving
                        .into_iter()
                        .partition(|g| g.by.is_some_and(|by| by <= now));
                    giving = waiting;
                    for g in passed {
                        g.answer(&name, false);
                    }
                    let gone: Vec<String> = away
                        .iter()
                        .filter(|(_, by)| by.is_some_and(|by| by <= now))
                        .map(|(m, _)| m.clone())
                        .collect();
                    for m in gone {
                        away.remove(&m);
                        // Those it still holds: one may have been taken back by need meanwhile.
                        let devices = held_by(&keeper.pool, &m);
                        keeper.record(leaves(&devices))?;
                        say(
                            &name,
                            format_args!(
                                "`{m}` did not join again within its grace: its {} devices \
                                 recorded as left",
                                devices.len()
                            ),
                        );
                    }
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    unreachable!("the taking thread holds a sender")
                }
            },
        };
        match event {
            Event::Notice => {
                say(&name, "has its notice: records its own devices' leave");
                keeper.record(leaves(&mine))?;
                return Ok(Ended::Coordinator);
            }
            Event::Opened { id, peer, sending } => {
                say(
                    &name,
                    format_args!(
                        "a control channel from `{}`, certificate {}",
                        peer.member,
                        peer.serial_hex()
                    ),
                );
                members.insert(
                    id,
                    Attached {
                        peer,
                        sending: Some(sending),
                        joined: None,
                    },
                );
            }
            Event::Said {
                id,
                word: Control::Join { member, offers },
            } => {
                let Some(m) = members.get(&id) else { continue };
                let who = m.peer.member.clone();
                let refused = if m.joined.is_some() {
                    Err(JoinRefused::Twice)
                } else {
                    check_join(&who, &member, &offers).and_then(|()| {
                        if who == name {
                            Err(JoinRefused::Coordinator(who.clone()))
                        } else if members.iter().any(|(other, o)| {
                            *other != id && o.joined.is_some() && o.peer.member == who
                        }) {
                            Err(JoinRefused::Held(who.clone()))
                        } else {
                            Ok(())
                        }
                    })
                };
                if let Err(why) = refused {
                    say(
                        &name,
                        format_args!(
                            "refuses the join on `{who}`'s channel: {why}; {}",
                            if why.lasting() {
                                "it is told so, and is to dial no more"
                            } else {
                                "it may dial again"
                            }
                        ),
                    );
                    if let Some(m) = members.get_mut(&id) {
                        // Refused for good: closed without `Joined`, which the member reads as
                        // its refusal. Refused for now: dropped unclosed, so the member finds the
                        // channel cut short, and dials again.
                        if let (Some(sending), true) = (m.sending.take(), why.lasting()) {
                            let _ = sending.close();
                        }
                        if m.joined.is_none() {
                            members.remove(&id);
                        }
                    }
                    continue;
                }
                let recorded = recorded_member(&start.pool.members, &member);
                let devices: Vec<DeviceId> = offers.iter().map(|o| o.device.clone()).collect();
                let back = away.remove(&who).is_some() || !held_by(&keeper.pool, &who).is_empty();
                let changes = join_changes(&keeper.pool.describe(), &recorded, &offers);
                let joins = changes
                    .iter()
                    .filter(|c| matches!(c, Change::Join { .. }))
                    .count();
                let left = changes.len() - joins;
                keeper.record(changes)?;
                if back {
                    say(
                        &name,
                        format_args!(
                            "`{who}` joined again with {} devices, at trust {}: {} held already, \
                             {joins} joined, {left} left",
                            devices.len(),
                            recorded.trust,
                            devices.len() - joins,
                        ),
                    );
                } else {
                    say(
                        &name,
                        format_args!(
                            "`{who}` joined with {} devices, at trust {}{}",
                            devices.len(),
                            recorded.trust,
                            if devices.is_empty() {
                                ": nothing to record"
                            } else {
                                ""
                            }
                        ),
                    );
                }
                if let Some(m) = members.get_mut(&id) {
                    m.joined = Some(devices);
                    // Written and synced first: only now is the member told it is in.
                    if let Some(Err(e)) = m.sending.as_mut().map(|s| s.send_word(&Control::Joined))
                    {
                        say(&name, format_args!("`{who}` was not told it joined ({e})"));
                    }
                }
            }
            Event::Said {
                id,
                word: Control::Leave,
            } => {
                let Some(m) = members.get_mut(&id) else {
                    continue;
                };
                let who = m.peer.member.clone();
                let devices = m.joined.take().unwrap_or_default();
                let sending = m.sending.take();
                // Written and synced first: the close below is the acknowledgement.
                keeper.record(leaves(&devices))?;
                // Leaving, it gives back every device, the one it was told to give back too.
                let (left, waiting) = giving.into_iter().partition(|g| g.id == id);
                giving = waiting;
                for g in left {
                    g.answer(&name, true);
                }
                match sending.map(Sender::close) {
                    Some(Ok(())) => say(
                        &name,
                        format_args!("`{who}` left, its {} devices recorded", devices.len()),
                    ),
                    Some(Err(e)) => say(
                        &name,
                        format_args!("`{who}` left, recorded, and was not told ({e})"),
                    ),
                    None => {}
                }
            }
            Event::Door { request, reply } => {
                // Written and synced first: the answer is sent, and a member told to give a
                // device back, only once `door` returns.
                let (device, deadline_seconds, record) = match keeper.door(request)? {
                    Answering::Now(answer) => {
                        say(&name, format_args!("the door: {}", told(&answer)));
                        let _ = reply.send(answer);
                        continue;
                    }
                    Answering::GiveBack {
                        device,
                        deadline_seconds,
                        record,
                    } => (device, deadline_seconds, record),
                };
                let by = Duration::try_from_secs_f64(deadline_seconds)
                    .ok()
                    .and_then(|d| Instant::now().checked_add(d));
                let holder = members
                    .iter_mut()
                    .find(|(_, m)| m.joined.as_ref().is_some_and(|d| d.contains(&device)));
                let mut g = Giving {
                    id: 0,
                    device: device.clone(),
                    by,
                    record,
                    reply,
                };
                let Some((&id, m)) = holder else {
                    // Its own: nothing runs on it, so it stops offering it at once, and does not
                    // record its leave again on its notice.
                    let own = mine.contains(&device);
                    mine.retain(|d| d != &device);
                    if own {
                        say(&name, format_args!("gives its own {device} back"));
                    } else {
                        say(
                            &name,
                            format_args!("holds no channel to {device}'s member, to tell it"),
                        );
                    }
                    g.answer(&name, own);
                    continue;
                };
                // Its leave is in the ledger: neither it nor this coordinator's notice records it
                // again.
                if let Some(joined) = m.joined.as_mut() {
                    joined.retain(|d| d != &device);
                }
                let word = Control::GiveBack {
                    device: device.clone(),
                    deadline_seconds,
                };
                match m.sending.as_mut().map(|s| s.send_word(&word)) {
                    Some(Ok(())) => {
                        say(
                            &name,
                            format_args!(
                                "tells `{}` to give {device} back within {deadline_seconds} s",
                                m.peer.member
                            ),
                        );
                        g.id = id;
                        giving.push(g);
                    }
                    told => {
                        say(
                            &name,
                            format_args!(
                                "could not tell `{}` to give {device} back ({})",
                                m.peer.member,
                                match told {
                                    Some(Err(e)) => e.to_string(),
                                    _ => "its channel is closed".into(),
                                }
                            ),
                        );
                        g.answer(&name, false);
                    }
                }
            }
            Event::Said {
                id,
                word: Control::GaveBack { device },
            } => {
                let who = members.get(&id).map(|m| m.peer.member.clone());
                match giving.iter().position(|g| g.id == id && g.device == device) {
                    Some(i) => {
                        say(
                            &name,
                            format_args!("`{}` gave {device} back", who.unwrap_or_default()),
                        );
                        giving.swap_remove(i).answer(&name, true);
                    }
                    None => say(
                        &name,
                        format_args!(
                            "`{}` says it gave {device} back, which no take-back waits for",
                            who.unwrap_or_default()
                        ),
                    ),
                }
            }
            Event::Said {
                id,
                word: word @ (Control::GiveBack { .. } | Control::Joined),
            } => {
                let who = members.get(&id).map(|m| m.peer.member.clone());
                say(
                    &name,
                    format_args!(
                        "`{}` says {word:?}, a coordinator's word: not taken",
                        who.unwrap_or_default()
                    ),
                );
            }
            Event::Ended { id, error } => {
                let Some(m) = members.remove(&id) else {
                    continue;
                };
                if m.joined.is_none() {
                    continue;
                }
                let who = m.peer.member;
                let devices = held_by(&keeper.pool, &who);
                if devices.is_empty() {
                    say(
                        &name,
                        format_args!("`{who}` went without leaving ({error}), holding no device"),
                    );
                    continue;
                }
                // Not recorded as left yet: for its grace, it may join again and keep them.
                let by = grace_of(&keeper.pool.describe(), &who);
                say(
                    &name,
                    format_args!(
                        "`{who}` went without leaving ({error}): its {} devices are kept for it \
                         to join again, for {}",
                        devices.len(),
                        within(by)
                    ),
                );
                away.insert(who, by);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mag_by::Backend;

    fn member(name: &str, trust: Trust, give_back_seconds: f64) -> Member {
        Member {
            name: name.into(),
            trust,
            give_back_seconds,
        }
    }

    fn offer(member: &str, index: u32) -> Offer {
        Offer {
            device: DeviceId {
                member: member.into(),
                index,
            },
            backend: Backend::Flex,
            memory_bytes: 1 << 30,
            tolerances: vec![],
            dtypes: vec![],
            costs: vec![],
        }
    }

    #[test]
    fn a_listed_member_is_recorded_at_the_stricter_of_the_two_and_an_unlisted_one_at_outside() {
        let listed = [
            member("server", Trust::Site, 30.0),
            member("laptop", Trust::Owner, 5.0),
        ];
        // Saying more than the listing gives: the listing stands.
        assert_eq!(
            recorded_member(&listed, &member("server", Trust::Owner, 60.0)),
            member("server", Trust::Site, 30.0)
        );
        // Saying less: what it says stands.
        assert_eq!(
            recorded_member(&listed, &member("laptop", Trust::Site, 2.0)),
            member("laptop", Trust::Site, 2.0)
        );
        // Unlisted: `outside` whatever it says, and its own time to give back.
        for trust in [Trust::Owner, Trust::Site, Trust::Outside] {
            assert_eq!(
                recorded_member(&listed, &member("rented", trust, 7.0)),
                member("rented", Trust::Outside, 7.0)
            );
        }
    }

    #[test]
    fn a_member_joining_again_changes_only_what_differs_from_what_the_pool_holds() {
        let laptop = member("laptop", Trust::Site, 2.0);
        let mut larger = offer("laptop", 1);
        larger.memory_bytes *= 2;
        let pool = Description {
            at: 0.0,
            members: vec![laptop.clone()],
            offers: vec![offer("laptop", 0), offer("laptop", 1)],
            jobs: vec![],
            formats: vec![],
        };
        let leave = |i| Change::Leave {
            device: DeviceId {
                member: "laptop".into(),
                index: i,
            },
        };
        let join = |m: &Member, o: &Offer| Change::Join {
            member: Some(m.clone()),
            offer: o.clone(),
        };
        // As it was: nothing.
        assert_eq!(
            join_changes(&pool, &laptop, &[offer("laptop", 0), offer("laptop", 1)]),
            []
        );
        // A device gone: its leave.
        assert_eq!(
            join_changes(&pool, &laptop, &[offer("laptop", 0)]),
            [leave(1)]
        );
        // A device offered otherwise, and one new: each joins; a leave comes first.
        let new = offer("laptop", 2);
        assert_eq!(
            join_changes(&pool, &laptop, &[larger.clone(), new.clone()]),
            [leave(0), join(&laptop, &larger), join(&laptop, &new)]
        );
        // Stricter than the pool holds it, in trust or in time to give back: every device joins,
        // to lower it. Laxer: nothing, since the pool keeps the stricter.
        for stricter in [
            member("laptop", Trust::Outside, 2.0),
            member("laptop", Trust::Site, 1.0),
        ] {
            assert_eq!(
                join_changes(&pool, &stricter, &[offer("laptop", 0)]),
                [leave(1), join(&stricter, &offer("laptop", 0))]
            );
        }
        let laxer = member("laptop", Trust::Owner, 9.0);
        assert_eq!(
            join_changes(&pool, &laxer, &[offer("laptop", 0)]),
            [leave(1)]
        );
        // A member the pool does not know: every device joins.
        let server = member("server", Trust::Owner, 2.0);
        assert_eq!(
            join_changes(&pool, &server, &[offer("server", 0)]),
            [join(&server, &offer("server", 0))]
        );
    }

    #[test]
    fn the_grace_is_the_time_to_give_back_and_at_least_the_least() {
        assert_eq!(LEAST_GRACE, Duration::from_secs(5));
        assert!(LEAST_GRACE > crate::member::REDIAL * 2);
        assert_eq!(grace(0.0), Some(LEAST_GRACE));
        assert_eq!(grace(2.0), Some(LEAST_GRACE));
        assert_eq!(grace(30.0), Some(Duration::from_secs(30)));
        assert_eq!(grace(1e20), None);
    }

    #[test]
    fn a_refusal_holds_for_good_only_for_another_members_name_or_device_or_the_coordinators() {
        let device = DeviceId {
            member: "laptop".into(),
            index: 0,
        };
        for (refused, lasting) in [
            (
                JoinRefused::NotItself {
                    said: "laptop".into(),
                    certificate: "server".into(),
                },
                true,
            ),
            (
                JoinRefused::NotItsDevice {
                    device,
                    certificate: "server".into(),
                },
                true,
            ),
            (JoinRefused::Coordinator("workstation".into()), true),
            (JoinRefused::Held("server".into()), false),
            (JoinRefused::Twice, false),
        ] {
            assert_eq!(refused.lasting(), lasting, "{refused:?}");
        }
    }

    #[test]
    fn a_join_must_name_the_certificates_member_and_offer_only_its_devices() {
        let server = member("server", Trust::Owner, 2.0);
        assert_eq!(
            check_join("server", &server, &[offer("server", 0), offer("server", 1)]),
            Ok(())
        );
        assert_eq!(check_join("server", &server, &[]), Ok(()));
        assert_eq!(
            check_join("laptop", &server, &[offer("server", 0)]),
            Err(JoinRefused::NotItself {
                said: "server".into(),
                certificate: "laptop".into()
            })
        );
        assert_eq!(
            check_join("server", &server, &[offer("server", 0), offer("laptop", 0)]),
            Err(JoinRefused::NotItsDevice {
                device: DeviceId {
                    member: "laptop".into(),
                    index: 0
                },
                certificate: "server".into()
            })
        );
    }
}
