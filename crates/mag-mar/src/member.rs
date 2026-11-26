//! The member program: what each member of a pool runs, from its start file to the end of its
//! lease.
//!
//! [`run`] is the program, and the crate's binary calls it:
//!
//! 1. The host starts it with `MAG_START` naming a start file ([`read_start`]). It loads the
//!    member's credentials from the files the start file names (`Credentials::from_start`).
//! 2. It binds the start file's `listen` address, where other members are to dial it; with none,
//!    it only dials, as a member behind address translation does.
//! 3. Unless it is the coordinator, it dials the coordinator and opens a control channel, for no
//!    job (`mag_by::CONTROL`), and joins: `mag_by::Control::Join`, with itself as its start file
//!    describes it and what each of its devices offers. The coordinator says
//!    `mag_by::Control::Joined` once the join is in its ledger. While the coordinator cannot be
//!    reached it dials again, after [`REDIAL_FIRST`] and then twice as long each time, up to
//!    [`REDIAL`]; the host tells the coordinator nothing, so the join is this member's to make.
//! 4. **It joins again.** When the control channel ends and the member is not leaving, as when
//!    the coordinator stops or the network between them fails, it runs on, and dials the
//!    coordinator again the same way and joins again, offering the devices it still holds. A
//!    coordinator that comes back, or another that takes up its ledger, takes it back in with
//!    them. A join the coordinator refuses for good, closing the channel without `Joined`, it
//!    does not make again: it stays out of the pool until its lease ends.
//! 5. When the coordinator tells it to give a device back (`mag_by::Control::GiveBack`), the
//!    host having taken a device back by need at the door, it stops offering that device at
//!    once, since no part runs on it yet, and says it has (`mag_by::Control::GaveBack`). It keeps
//!    its other devices and its lease: a member whose last device is taken stays a member with
//!    none until the host ends its lease, which is the host's to do by name.
//! 6. On notice, SIGTERM from the host ([`Notice`]), it says `mag_by::Control::Leave`, closes its
//!    way of the control channel, and waits for the coordinator to close the other way, the sign
//!    that the leaving is recorded. Then it returns, and the program exits 0. It dials no more.
//!
//! The member the start file names as coordinator dials no one: it runs the coordinator
//! ([`crate::coordinator`]), which takes the other members' joins and leaves into the pool and its
//! ledger, until its own notice.
//!
//! The member's `give_back_seconds` bound step 6 whatever any peer does. The notice sets a
//! deadline on a thread of its own, at a tenth of them before their end and never more than 1 s
//! before it ([`deadline`]), and at that deadline the process exits 0, wherever [`run`] is
//! waiting: on a coordinator that stopped answering, on a dial, on anything. A link's keepalive
//! cannot promise this, since TCP does not probe while data is unacknowledged.
//!
//! Not built yet: running the parts placed on it, channels to other members for their jobs,
//! giving back with the parts' state, reporting what it measured while the coordinator was gone,
//! and stopping its parts once it has been cut off from the coordinator for as long as the
//! coordinator keeps its devices for it.

use std::fmt;
use std::io;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use mag_by::{CONTROL, Control, Offer, Start, StartFileError};

use crate::{Channel, Credentials, Link, LinkError, Receiver, Sender, TlsChannel, TlsLink};
use crate::{TlsReceiver, TlsSender};

/// The environment variable that names a member's start file.
pub const MAG_START: &str = "MAG_START";

/// How long a member waits before it dials the coordinator again, after its control channel
/// ended or a dial failed. Each dial that fails doubles it, up to [`REDIAL`].
pub const REDIAL_FIRST: Duration = Duration::from_millis(250);

/// The longest a member waits between two dials to the coordinator: so that it comes back soon
/// after the coordinator does, well within the grace the coordinator gives it
/// ([`crate::coordinator::grace`]).
pub const REDIAL: Duration = Duration::from_secs(2);

/// When, after its notice, a member with `give_back` to give its devices back ends itself: a
/// tenth of `give_back` before its end, and never more than 1 s before it, so that the process
/// has ended by `give_back`.
pub fn deadline(give_back: Duration) -> Duration {
    give_back - (give_back / 10).min(Duration::from_secs(1))
}

/// How a member's lease ended, when [`run`] returns before its deadline.
#[derive(Debug)]
pub enum Ended {
    /// It said it leaves, and the coordinator closed the control channel: the leaving is
    /// recorded.
    Left,
    /// It tried to say it leaves, and the control channel failed: why. Its devices are given back
    /// all the same.
    Unheard(LinkError),
    /// The notice came while it had no control channel: before it joined, or after the channel
    /// ended. There was no one to tell.
    Alone,
    /// It is the coordinator: it took the members' joins and leaves into the pool and its ledger
    /// until its notice, and then recorded its own devices' leave.
    Coordinator,
}

/// Why a member could not start.
#[derive(Debug)]
pub enum MemberError {
    /// `MAG_START` is not set, or is not Unicode.
    NoStart,
    /// The start file could not be read: its path, and why.
    Read { path: String, error: io::Error },
    /// The start file was refused: its path, and why.
    Start { path: String, error: StartFileError },
    /// The member is not in its start file's pool, which a start file read from its file never
    /// allows.
    NotInPool(String),
    /// The member's `give_back_seconds` are not a time it can wait: not finite or below 0, which
    /// a start file read from its file never allows, or 2^64 s or more, which it does.
    GiveBack(f64),
    /// The coordinator is a peer with no address, which a start file read from its file never
    /// allows.
    NoCoordinatorAddress(String),
    /// The member's credentials, or binding its `listen` address.
    Link(LinkError),
    /// The coordinator could not begin its ledger at the start file's `ledger`: the path, and
    /// why. A file that comes there meanwhile is refused, not overwritten.
    Ledger { path: String, error: io::Error },
    /// The coordinator did not take up the ledger at the start file's `ledger`, left by one that
    /// stopped: the path, and why. It is left as it is.
    TakeUp {
        path: String,
        error: crate::coordinator::TakeUpError,
    },
    /// The coordinator could not write a line of its ledger, and stopped: the path, and why.
    /// Nothing after the last line written was acknowledged.
    LedgerUnwritten { path: String, error: io::Error },
    /// The coordinator could not serve its door at the start file's `door`: the path, and why.
    /// Another coordinator serving it is one reason.
    Door {
        path: String,
        error: crate::door::DoorError,
    },
}

impl fmt::Display for MemberError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemberError::NoStart => write!(f, "{MAG_START} must name a start file"),
            MemberError::Read { path, error } => write!(f, "{path}: {error}"),
            MemberError::Start { path, error } => write!(f, "{path}: {error}"),
            MemberError::NotInPool(name) => {
                write!(f, "`{name}` is not a member of its start file's pool")
            }
            MemberError::GiveBack(s) => write!(
                f,
                "`give_back_seconds` is {s}; it must be finite, not below 0 and below 2^64"
            ),
            MemberError::NoCoordinatorAddress(name) => {
                write!(f, "the coordinator `{name}` has no address to dial")
            }
            MemberError::Link(e) => write!(f, "{e}"),
            MemberError::Ledger { path, error } => {
                write!(f, "the ledger {path} cannot be begun ({error})")
            }
            MemberError::TakeUp { path, error } => write!(
                f,
                "the ledger {path} is not taken up, and is left as it is: {error}"
            ),
            MemberError::LedgerUnwritten { path, error } => write!(
                f,
                "a line of the ledger {path} was not written ({error}): the coordinator stops, \
                 and nothing after its last line was acknowledged"
            ),
            MemberError::Door { path, error } => {
                write!(f, "the door {path} cannot be served: {error}")
            }
        }
    }
}

impl std::error::Error for MemberError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MemberError::Read { error, .. } => Some(error),
            MemberError::Start { error, .. } => Some(error),
            MemberError::Link(e) => Some(e),
            MemberError::Ledger { error, .. } | MemberError::LedgerUnwritten { error, .. } => {
                Some(error)
            }
            MemberError::Door { error, .. } => Some(error),
            MemberError::TakeUp { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// The start file `MAG_START` names, read and checked.
pub fn read_start() -> Result<Start, MemberError> {
    let path = std::env::var(MAG_START).map_err(|_| MemberError::NoStart)?;
    let text = std::fs::read_to_string(&path).map_err(|error| MemberError::Read {
        path: path.clone(),
        error,
    })?;
    Start::from_json(&text).map_err(|error| MemberError::Start { path, error })
}

/// What the start file's pool says this member's devices offer. Until a member measures its own
/// devices, these are the offers it joins with.
pub fn offers(start: &Start) -> Vec<Offer> {
    let mine = |o: &&Offer| o.device.member == start.member;
    start.pool.offers.iter().filter(mine).cloned().collect()
}

/// The host's notice that a member's lease ends: SIGTERM.
pub struct Notice {
    #[cfg(unix)]
    pipe: std::fs::File,
}

impl Notice {
    /// Takes SIGTERM as the notice from now on, in place of its default, which ends the process
    /// at once. Made once, as the program starts; a second call fails. Unix only.
    pub fn sigterm() -> io::Result<Notice> {
        #[cfg(unix)]
        {
            Ok(Notice {
                pipe: signal::sigterm()?,
            })
        }
        #[cfg(not(unix))]
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "SIGTERM is a Unix signal",
        ))
    }

    /// Waits for the notice, and returns when it was seen. A notice that came before is seen at
    /// once.
    pub(crate) fn wait(self) -> Instant {
        #[cfg(unix)]
        {
            use std::io::Read;
            let mut pipe = self.pipe;
            let mut byte = [0];
            loop {
                match pipe.read(&mut byte) {
                    Ok(1) => return Instant::now(),
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    // The pipe's writing end is never closed: no notice can come.
                    _ => break,
                }
            }
        }
        loop {
            thread::park();
        }
    }
}

#[cfg(unix)]
mod signal {
    //! SIGTERM, written as one byte to a pipe by its handler, which does nothing else: writing to
    //! a pipe is safe in a signal handler, and a thread reads the byte at its leisure.

    use std::io;
    use std::os::fd::FromRawFd;
    use std::sync::atomic::{AtomicI32, Ordering};

    /// The pipe's writing end, once SIGTERM is taken.
    static WRITE: AtomicI32 = AtomicI32::new(-1);

    extern "C" fn on_term(_: libc::c_int) {
        let fd = WRITE.load(Ordering::Relaxed);
        if fd >= 0 {
            // SAFETY: write(2) is async-signal-safe; the buffer is one byte that lives in this
            // frame. A full pipe (non-blocking) or any failure is ignored: a byte is there
            // already. errno is not restored; nothing reads it across a SIGTERM.
            unsafe {
                libc::write(fd, [1u8].as_ptr().cast(), 1);
            }
        }
    }

    fn check(r: libc::c_int) -> io::Result<libc::c_int> {
        if r < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(r)
        }
    }

    pub(super) fn sigterm() -> io::Result<std::fs::File> {
        let mut fds = [-1; 2];
        // SAFETY: `fds` has room for the two descriptors pipe(2) writes.
        check(unsafe { libc::pipe(fds.as_mut_ptr()) })?;
        let [read, write] = fds;
        // SAFETY: fcntl(2) on two descriptors this function owns.
        unsafe {
            for fd in fds {
                check(libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC))?;
            }
            let flags = check(libc::fcntl(write, libc::F_GETFL))?;
            check(libc::fcntl(write, libc::F_SETFL, flags | libc::O_NONBLOCK))?;
        }
        // SAFETY: `read` is a new descriptor that nothing else owns.
        let reading = unsafe { std::fs::File::from_raw_fd(read) };
        if WRITE
            .compare_exchange(-1, write, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            // SAFETY: `write` is this call's own, and not published.
            unsafe { libc::close(write) };
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "SIGTERM is taken already",
            ));
        }
        // SAFETY: a zeroed sigaction is a valid one; the handler only loads an atomic and writes
        // to a pipe.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_term as extern "C" fn(libc::c_int) as libc::sighandler_t;
            action.sa_flags = libc::SA_RESTART;
            libc::sigemptyset(&mut action.sa_mask);
            check(libc::sigaction(
                libc::SIGTERM,
                &action,
                std::ptr::null_mut(),
            ))?;
        }
        Ok(reading)
    }
}

/// What the member's one waiting thread is told.
enum Event {
    /// The notice. Its deadline is the notice's own thread's.
    Notice,
    /// A control channel to the coordinator, open.
    Dialled(Box<TlsChannel>),
    /// A word from the coordinator.
    Heard(Control),
    /// The control channel ended, as its receiving half found it.
    Ended(LinkError),
}

pub(crate) fn say(member: &str, what: impl fmt::Display) {
    eprintln!("mag-mar {member}: {what}");
}

/// Waits for the notice on a thread of its own, calls `tell` when it comes, and at the
/// [`deadline`] it sets ends the process with status 0, whatever any other thread waits on: so
/// the process ends by `give_back`.
pub(crate) fn give_back_by(
    notice: Notice,
    give_back: Duration,
    name: String,
    tell: impl FnOnce() + Send + 'static,
) {
    thread::spawn(move || {
        let at = notice.wait();
        tell();
        let end = at + deadline(give_back);
        thread::sleep(end.saturating_duration_since(Instant::now()));
        say(
            &name,
            format_args!(
                "gives its devices back at its deadline, {:.2} s after the notice, without \
                 waiting longer for any peer",
                at.elapsed().as_secs_f64()
            ),
        );
        std::process::exit(0);
    });
}

/// The wait before the next dial, after one of `wait` that failed: twice as long, up to
/// [`REDIAL`].
fn longer(wait: Duration) -> Duration {
    (wait * 2).min(REDIAL)
}

/// Dials the coordinator on a thread of its own until a control channel opens, after `wait`.
/// After each dial that fails it waits twice as long as before, up to [`REDIAL`], or
/// [`REDIAL_FIRST`] after a first dial made at once.
fn dial(
    link: Arc<TlsLink>,
    member: String,
    coordinator: String,
    address: String,
    wait: Duration,
    events: mpsc::Sender<Event>,
) {
    thread::spawn(move || {
        let mut wait = wait;
        loop {
            thread::sleep(wait);
            match link.open(&coordinator, &address, CONTROL) {
                Ok(channel) => {
                    let _ = events.send(Event::Dialled(Box::new(channel)));
                    return;
                }
                Err(e) => {
                    wait = if wait.is_zero() {
                        REDIAL_FIRST
                    } else {
                        longer(wait)
                    };
                    say(
                        &member,
                        format_args!(
                            "the coordinator `{coordinator}` at {address} is not reached ({e}); \
                             dialling again in {} s",
                            wait.as_secs_f64()
                        ),
                    );
                }
            }
        }
    });
}

/// Hears the coordinator on the control channel's receiving half, on a thread of its own.
fn hear(mut receiving: TlsReceiver, events: mpsc::Sender<Event>) {
    thread::spawn(move || {
        loop {
            match receiving.recv_word::<Control>() {
                Ok(word) => {
                    if events.send(Event::Heard(word)).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = events.send(Event::Ended(e));
                    return;
                }
            }
        }
    });
}

/// Runs the member `start` describes until the host ends its lease, joining with `offers`.
///
/// Returns once the member has given back; the program then exits 0. If that has not happened
/// by the [`deadline`] the notice sets, the process exits 0 there, from the notice's own thread,
/// and this never returns. Fails only before the notice can come: on credentials, binding,
/// or a start file that does not hold together.
pub fn run(start: &Start, mut offers: Vec<Offer>, notice: Notice) -> Result<Ended, MemberError> {
    let name = start.member.clone();
    let me = start
        .pool
        .members
        .iter()
        .find(|m| m.name == name)
        .ok_or_else(|| MemberError::NotInPool(name.clone()))?
        .clone();
    let give_back = Duration::try_from_secs_f64(me.give_back_seconds)
        .map_err(|_| MemberError::GiveBack(me.give_back_seconds))?;
    let coordinator = (start.coordinator != name)
        .then(|| {
            start
                .peers
                .iter()
                .find(|p| p.member == start.coordinator)
                .and_then(|p| p.address.clone())
                .ok_or_else(|| MemberError::NoCoordinatorAddress(start.coordinator.clone()))
        })
        .transpose()?;
    let credentials = Credentials::from_start(start).map_err(MemberError::Link)?;
    let link = match &start.listen {
        Some(at) => TlsLink::bind(credentials, at.as_str()),
        None => TlsLink::new(credentials),
    }
    .map_err(MemberError::Link)?;
    match link.local_addr() {
        Some(at) => say(&name, format_args!("answers at {at}")),
        None => say(&name, "dials only"),
    }

    let Some(address) = coordinator else {
        let offers = offers
            .into_iter()
            .filter(|o| o.device.member == name)
            .collect();
        return crate::coordinator::run(start, me, offers, link, notice, give_back);
    };

    let (events_tx, events) = mpsc::channel();
    {
        let events = events_tx.clone();
        give_back_by(notice, give_back, name.clone(), move || {
            let _ = events.send(Event::Notice);
        });
    }
    let next = || events.recv().expect("this thread holds a sender");
    let link = Arc::new(link);
    let redial = |wait| {
        dial(
            link.clone(),
            name.clone(),
            start.coordinator.clone(),
            address.clone(),
            wait,
            events_tx.clone(),
        )
    };
    redial(Duration::ZERO);

    let mut control: Option<TlsSender> = None;
    // Whether the coordinator said the join on this control channel is in.
    let mut joined = false;
    loop {
        match next() {
            Event::Notice => {
                say(&name, "has its notice: giving its devices back");
                break;
            }
            Event::Dialled(channel) => {
                let (mut sending, receiving) = (*channel).split();
                let join = Control::Join {
                    member: me.clone(),
                    offers: offers.clone(),
                };
                match sending.send_word(&join) {
                    Ok(()) => {
                        say(
                            &name,
                            format_args!(
                                "asks to join the pool at the coordinator `{}`, offering {} \
                                 devices",
                                start.coordinator,
                                offers.len()
                            ),
                        );
                        hear(receiving, events_tx.clone());
                        control = Some(sending);
                        joined = false;
                    }
                    Err(e) => {
                        say(&name, format_args!("the join was not sent ({e})"));
                        redial(REDIAL_FIRST);
                    }
                }
            }
            Event::Heard(Control::Joined) => {
                joined = true;
                say(
                    &name,
                    format_args!(
                        "joined the pool at the coordinator `{}`, offering {} devices",
                        start.coordinator,
                        offers.len()
                    ),
                );
            }
            Event::Heard(Control::GiveBack {
                device,
                deadline_seconds,
            }) => {
                if !offers.iter().any(|o| o.device == device) {
                    say(
                        &name,
                        format_args!("is told to give {device} back, which it does not offer"),
                    );
                    continue;
                }
                // No part runs on it yet: it is given back by offering it no longer, at once,
                // well within the deadline and this member's own time to give back.
                offers.retain(|o| o.device != device);
                let within = deadline_seconds.min(me.give_back_seconds);
                let said = match control.as_mut() {
                    Some(sending) => sending.send_word(&Control::GaveBack {
                        device: device.clone(),
                    }),
                    None => Err(LinkError::Closed),
                };
                say(
                    &name,
                    format_args!(
                        "gives {device} back at the coordinator's word, within {within} s{}",
                        match &said {
                            Ok(()) => String::new(),
                            Err(e) => format!(", and could not say so ({e})"),
                        }
                    ),
                );
                if offers.is_empty() {
                    say(
                        &name,
                        "offers no device now, and stays a member until its lease ends",
                    );
                }
            }
            Event::Heard(word) => say(
                &name,
                format_args!("a word from the coordinator this member does not take: {word:?}"),
            ),
            // Closed before `Joined`: the coordinator refused the join for good.
            Event::Ended(LinkError::Closed) if !joined => {
                say(
                    &name,
                    "the coordinator refused its join, closing the control channel: it dials no \
                     more, and stays out of the pool until its lease ends",
                );
                control = None;
            }
            Event::Ended(e) => {
                say(
                    &name,
                    format_args!(
                        "the control channel ended ({e}); it runs on, and dials the coordinator \
                         again"
                    ),
                );
                control = None;
                joined = false;
                redial(REDIAL_FIRST);
            }
        }
    }

    let Some(mut sending) = control else {
        say(&name, "has no control channel: no one to tell it leaves");
        return Ok(Ended::Alone);
    };
    if let Err(e) = sending.send_word(&Control::Leave) {
        say(&name, format_args!("its leaving was not sent ({e})"));
        return Ok(Ended::Unheard(e));
    }
    // Then its own way closed. The coordinator may have read the leaving, recorded it and closed
    // both ways already, and the close then fails (not connected): whether the leaving was heard
    // is the coordinator's close, which the channel's receiving half reports below, either way.
    if let Err(e) = sending.close() {
        say(
            &name,
            format_args!("its own way of the control channel did not close cleanly ({e})"),
        );
    }
    // The coordinator closes its way once the leaving is recorded. Until then this waits, and
    // the deadline bounds it.
    loop {
        match next() {
            Event::Ended(LinkError::Closed) => {
                say(&name, "left: the coordinator recorded it");
                return Ok(Ended::Left);
            }
            Event::Ended(e) => {
                say(
                    &name,
                    format_args!("the control channel failed as it left ({e})"),
                );
                return Ok(Ended::Unheard(e));
            }
            Event::Dialled(_) | Event::Heard(_) | Event::Notice => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_member_waits_twice_as_long_after_each_dial_that_fails_up_to_the_longest() {
        let mut wait = REDIAL_FIRST;
        let mut waits = vec![wait];
        for _ in 0..5 {
            wait = longer(wait);
            waits.push(wait);
        }
        let ms = |m| Duration::from_millis(m);
        assert_eq!(
            waits,
            [ms(250), ms(500), ms(1000), ms(2000), ms(2000), ms(2000)]
        );
        assert_eq!(REDIAL, ms(2000));
    }
}
