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
//!    describes it and what each of its devices offers. While the coordinator cannot be reached
//!    it dials again every [`REDIAL`]; the host tells the coordinator nothing, so the join is
//!    this member's to make.
//! 4. On notice, SIGTERM from the host ([`Notice`]), it says `mag_by::Control::Leave`, closes its
//!    way of the control channel, and waits for the coordinator to close the other way, the sign
//!    that the leaving is recorded. Then it returns, and the program exits 0.
//!
//! The member's `give_back_seconds` bound step 4 whatever any peer does. The notice sets a
//! deadline on a thread of its own, at a tenth of them before their end and never more than 1 s
//! before it ([`deadline`]), and at that deadline the process exits 0, wherever [`run`] is
//! waiting: on a coordinator that stopped answering, on a dial, on anything. A link's keepalive
//! cannot promise this, since TCP does not probe while data is unacknowledged.
//!
//! Not built yet: running the parts placed on it, channels to other members for their jobs,
//! giving back with the parts' state, and what the coordinator's member does as coordinator.
//! The coordinator's member binds and waits for its notice; joining again after the
//! coordinator returns is the coordinator's step too.

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

/// How long a member waits before dialling the coordinator again after a dial failed.
pub const REDIAL: Duration = Duration::from_secs(1);

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
    /// It is the coordinator, whose part is not built yet: it bound its address and waited.
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
        }
    }
}

impl std::error::Error for MemberError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MemberError::Read { error, .. } => Some(error),
            MemberError::Start { error, .. } => Some(error),
            MemberError::Link(e) => Some(e),
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
    fn wait(self) -> Instant {
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
    /// A word from the coordinator, as its header: none is expected yet.
    Heard(Vec<u8>),
    /// The control channel ended, as its receiving half found it.
    Ended(LinkError),
}

fn say(member: &str, what: impl fmt::Display) {
    eprintln!("mag-mar {member}: {what}");
}

/// Dials the coordinator on a thread of its own until a control channel opens, after `wait`.
fn dial(
    link: Arc<TlsLink>,
    member: String,
    coordinator: String,
    address: String,
    wait: Duration,
    events: mpsc::Sender<Event>,
) {
    thread::spawn(move || {
        thread::sleep(wait);
        loop {
            match link.open(&coordinator, &address, CONTROL) {
                Ok(channel) => {
                    let _ = events.send(Event::Dialled(Box::new(channel)));
                    return;
                }
                Err(e) => say(
                    &member,
                    format_args!(
                        "the coordinator `{coordinator}` at {address} is not reached ({e}); \
                         dialling again in {} s",
                        REDIAL.as_secs_f64()
                    ),
                ),
            }
            thread::sleep(REDIAL);
        }
    });
}

/// Hears the coordinator on the control channel's receiving half, on a thread of its own.
fn hear(mut receiving: TlsReceiver, events: mpsc::Sender<Event>) {
    thread::spawn(move || {
        loop {
            match receiving.recv() {
                Ok(incoming) => {
                    if events.send(Event::Heard(incoming.header)).is_err() {
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
pub fn run(start: &Start, offers: Vec<Offer>, notice: Notice) -> Result<Ended, MemberError> {
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

    let (events_tx, events) = mpsc::channel();
    {
        // The notice, and the deadline it sets, on a thread of its own: whatever this thread
        // waits on, the process ends by `give_back`.
        let events = events_tx.clone();
        let name = name.clone();
        thread::spawn(move || {
            let at = notice.wait();
            let _ = events.send(Event::Notice);
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
    let next = || events.recv().expect("this thread holds a sender");

    let Some(address) = coordinator else {
        say(&name, "coordinates: waiting for its notice");
        while !matches!(next(), Event::Notice) {}
        return Ok(Ended::Coordinator);
    };
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
                                "joined the pool at the coordinator `{}`, offering {} devices",
                                start.coordinator,
                                offers.len()
                            ),
                        );
                        hear(receiving, events_tx.clone());
                        control = Some(sending);
                    }
                    Err(e) => {
                        say(&name, format_args!("the join was not sent ({e})"));
                        redial(REDIAL);
                    }
                }
            }
            Event::Heard(header) => say(
                &name,
                format_args!(
                    "a word from the coordinator this member does not take yet: {}",
                    String::from_utf8_lossy(&header)
                ),
            ),
            Event::Ended(e) => {
                say(&name, format_args!("the control channel ended ({e})"));
                control = None;
            }
        }
    }

    let Some(mut sending) = control else {
        say(&name, "has no control channel: no one to tell it leaves");
        return Ok(Ended::Alone);
    };
    if let Err(e) = sending
        .send_word(&Control::Leave)
        .and_then(|()| sending.close())
    {
        say(&name, format_args!("its leaving was not sent ({e})"));
        return Ok(Ended::Unheard(e));
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
