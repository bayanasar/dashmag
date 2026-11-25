//! The program every member of a mag pool runs: so far its link to the other members, and the
//! program's skeleton, which joins the pool and gives its devices back on notice.
//!
//! A host gives a pool its machines; on each, it starts this program with a start file
//! (`mag_by::Start`). The members then link to each other, and this crate holds that link:
//!
//! - [`Link`] and [`Channel`]: the one interface between members, small so that another
//!   transport can stand in for the first. A member opens a channel to a peer for one job, or
//!   accepts one a peer opens; on it, it sends and receives framed messages, each a header and a
//!   body that may be gigabytes long; it reads the channel's measured rate, for the placement
//!   loop's cost model; and it closes it. Each job between two members has its own channel, so
//!   that one job's words never reach another's. A channel splits into a [`Sender`] and a
//!   [`Receiver`], one for each thread, so that both ends can send at once and neither waits on
//!   the other.
//! - [`TlsLink`], the first implementation: TCP with TLS 1.3 (rustls), both ends authenticated.
//!   Each pool has its own certificate authority, held by the host, which issues each member a
//!   certificate naming it ([`Credentials`]). A channel is taken only when the other end's
//!   certificate is signed by the pool's authority and names the member expected; anything
//!   else is a typed [`Refused`]. Either end may dial, so a member behind address translation
//!   dials out.
//! - The messages are `mag-by`'s words, as JSON in the header: [`Sender::send_word`],
//!   [`Receiver::recv_word`], and for a tensor, [`Sender::send_tensor`] and
//!   [`Receiver::recv_tensor`], its bytes in the body, streamed from the caller's buffer and into
//!   the caller's buffer, never held twice.
//! - An open channel is watched: TCP keepalive finds a peer whose machine went away
//!   ([`Keepalive`]), and a receive may be given a limit on silence
//!   ([`TlsLink::with_silence`]). Dialling and the opening are bounded by one deadline. The
//!   connections an end answers open apart, each on a thread of its own, so that a peer that
//!   connects and says nothing holds no other's opening, and at most so many at once
//!   ([`TlsLink::with_answering`]).
//! - [`Identity`]: who the other end is, by the member its certificate names and the
//!   certificate's serial, which tells a certificate the host issued after a lease ended from
//!   the one it ended on.
//! - With the `issue` feature, `issue`: the pool's authority and its members' certificates,
//!   as the host makes them; the `certs` example is the command.
//!
//!
//! And the program on the link, [`member::run`], which the crate's binary calls: it reads the
//! start file `MAG_START` names, binds the member's own address if it has one, dials the
//! coordinator unless it is the coordinator, and joins over a control channel with its offers.
//! On SIGTERM it says it leaves and exits 0 within its `give_back_seconds`, whatever any peer
//! does. On the member that coordinates, it runs [`coordinator`]: it takes any member's control
//! channel ([`Link::accept_any`]), and records each join and leave in a `mag_mortar::Pool` and in
//! the pool's ledger, written before a join or a leave is acknowledged. A coordinator started on
//! the ledger of one that stopped takes it up, and its members, which ran on and dialled again,
//! join again within their grace. It serves the door, a local socket
//! where a command run on its machine ([`door::command`], `mag-mar door`) submits a job, ends one
//! or asks the pool's status, each change written to the ledger before it is answered. Running
//! the parts placed on a member is not built yet.

pub mod coordinator;
pub mod door;
mod frame;
#[cfg(feature = "issue")]
pub mod issue;
pub mod member;
mod tls;

use std::fmt;
use std::io;

use mag_by::Tensor;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub use frame::MAX_HEADER;
pub use tls::{
    ANSWERING, Credentials, KEEPALIVE, Keepalive, OPENING, TlsChannel, TlsLink, TlsReceiver,
    TlsSender,
};

/// One end of the links between members: the replaceable interface. A member holds one, and
/// opens or accepts a channel for each job it shares with another member.
pub trait Link {
    type Channel: Channel;

    /// Opens a channel to `member`, dialled at `address`, for `job`. It is open once the other
    /// end has taken it: authenticated as `member`, and answering for the same job.
    fn open(&self, member: &str, address: &str, job: &str) -> Result<Self::Channel, LinkError>;

    /// Takes the next channel another member opens to this one, if it is one of `expected`: the
    /// coordinator, or a peer the coordinator has announced. A channel from any other member is
    /// refused and closed, and the refusal returned; the caller accepts again. A certificate
    /// that names more than one of `expected` is refused too, since it could stand for either;
    /// one that names exactly one of them is taken as that member.
    ///
    /// Connections open apart from each other, and from the caller: a peer that stalls its
    /// opening holds no other's, so a member's channel opens while others wait, and `accept`
    /// returns whichever opening finishes first. An opening that does not finish in the time the
    /// implementation allows is given up, and that failure returned to a caller as a refusal is.
    /// An implementation holds only so many openings at once, and closes a connection past them
    /// at once, so that a flood of connections cannot exhaust it; the member closed so dials
    /// again. A call takes one channel: a member that must answer several peers accepts on a
    /// thread of its own and hands each channel on.
    fn accept(&self, expected: &[&str]) -> Result<Self::Channel, LinkError>;

    /// Takes the next channel any member of the pool opens to this one, and reports it as the
    /// channel's peer: whoever holds a certificate the pool's authority signed naming exactly one
    /// member. The coordinator takes its members' control channels so, since a member that joins
    /// after it began is named by nothing it was told, only by its certificate; it then tells a
    /// control channel from a job's by [`ChannelEnd::job`]. A certificate naming no member, or
    /// more than one, is refused and closed, [`Refused::NotOneMember`], and the refusal
    /// returned; the caller accepts again. Its connections open apart from each other, as
    /// [`Link::accept`]'s do.
    fn accept_any(&self) -> Result<Self::Channel, LinkError>;

    /// The rate measured on every channel to `member` so far, for the placement loop's cost
    /// model; `None` before any channel to it has moved a body.
    fn rate(&self, member: &str) -> Option<Rate>;
}

/// What a channel, or either half of one, says of itself.
pub trait ChannelEnd {
    /// The other end.
    fn peer(&self) -> &Identity;

    /// The job the channel is for.
    fn job(&self) -> &str;

    /// The rate measured so far: both ways on a whole channel, and on a half, its own way only.
    fn rate(&self) -> Rate;
}

/// The sending half of a channel, or a whole channel as it sends.
pub trait Sender: ChannelEnd {
    /// Sends one message: a header of at most [`MAX_HEADER`] bytes, and a body of any length,
    /// read from `body` as it is sent.
    fn send(&mut self, header: &[u8], body: &[u8]) -> Result<(), LinkError>;

    /// Closes the channel: the other end's next receive finds it [`LinkError::Closed`]. Closing
    /// the sending half closes the channel's one way; the other end may still send until it
    /// closes its own.
    fn close(self) -> Result<(), LinkError>
    where
        Self: Sized;

    /// Sends one of `mag-by`'s words, with no body.
    fn send_word<W: Serialize>(&mut self, word: &W) -> Result<(), LinkError>
    where
        Self: Sized,
    {
        let header = serde_json::to_vec(word).map_err(LinkError::Word)?;
        self.send(&header, &[])
    }

    /// Sends a tensor: the word, and its bytes as the body, which must be `tensor.bytes` long.
    fn send_tensor(&mut self, tensor: &Tensor, bytes: &[u8]) -> Result<(), LinkError>
    where
        Self: Sized,
    {
        if tensor.bytes != bytes.len() as u64 {
            return Err(LinkError::Body {
                expected: tensor.bytes,
                found: bytes.len() as u64,
            });
        }
        let header = serde_json::to_vec(tensor).map_err(LinkError::Word)?;
        self.send(&header, bytes)
    }
}

/// The receiving half of a channel, or a whole channel as it receives.
pub trait Receiver: ChannelEnd {
    /// Receives the next message's header and the length of its body. The body is then read
    /// with [`Receiver::read_body`], before the next message.
    fn recv(&mut self) -> Result<Incoming, LinkError>;

    /// Reads the body of the message last received into `into`, which must be its length.
    fn read_body(&mut self, into: &mut [u8]) -> Result<(), LinkError>;

    /// Receives one of `mag-by`'s words, sent with no body.
    fn recv_word<W: DeserializeOwned>(&mut self) -> Result<W, LinkError>
    where
        Self: Sized,
    {
        let incoming = self.recv()?;
        if incoming.body != 0 {
            return Err(LinkError::Body {
                expected: 0,
                found: incoming.body,
            });
        }
        serde_json::from_slice(&incoming.header).map_err(LinkError::Word)
    }

    /// Receives a tensor's word. Its bytes, `tensor.bytes` of them, are then read with
    /// [`Receiver::read_body`].
    fn recv_tensor(&mut self) -> Result<Tensor, LinkError>
    where
        Self: Sized,
    {
        let incoming = self.recv()?;
        let tensor: Tensor = serde_json::from_slice(&incoming.header).map_err(LinkError::Word)?;
        if tensor.bytes != incoming.body {
            return Err(LinkError::Body {
                expected: tensor.bytes,
                found: incoming.body,
            });
        }
        Ok(tensor)
    }
}

/// One job's channel between two members: it sends and it receives.
///
/// Used by one thread, a channel takes turns: while it sends a body, it does not read, so two
/// ends that both send bodies larger than the sockets' buffers, each before reading, would wait
/// on each other. Split, it cannot: each half goes to a thread of its own, and a half that
/// sends never waits for its own end to read, only for the other end's receiving half. A
/// member that sends and receives on one job at once, as a chain does with activations forward
/// and gradients back, splits the channel.
pub trait Channel: Sender + Receiver {
    type SendHalf: Sender + Send + 'static;
    type ReceiveHalf: Receiver + Send + 'static;

    /// The two halves, each to be used on its own thread. Dropping the sending half without
    /// closing it ends the channel's one way, and the other end finds it [`LinkError::Truncated`].
    fn split(self) -> (Self::SendHalf, Self::ReceiveHalf);
}

/// A message's header, and the length of the body that follows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Incoming {
    pub header: Vec<u8>,
    pub body: u64,
}

/// The other end of a channel, as its certificate shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// The member its certificate names.
    pub member: String,
    /// The certificate's serial number, as its bytes, big-endian. The host gives every
    /// certificate a new one, so it tells a certificate issued after a member's lease ended from
    /// the one it ended on. Refusing the old one is the coordinator's.
    pub serial: Vec<u8>,
}

impl Identity {
    /// The serial in hexadecimal, as tools print it.
    pub fn serial_hex(&self) -> String {
        self.serial.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// The bytes moved on a channel, or on all of a link's channels to one member, and the time
/// they took, each way. Only bodies are counted: from the first byte of a body to its last, as
/// read, and from the first byte written to the last one flushed, as sent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rate {
    pub sent: Measure,
    pub received: Measure,
}

/// Bytes, and the seconds they took.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Measure {
    pub bytes: u64,
    pub seconds: f64,
}

impl Measure {
    /// Bytes per second, once any time has been measured.
    pub fn bytes_per_second(&self) -> Option<f64> {
        (self.seconds > 0.0).then(|| self.bytes as f64 / self.seconds)
    }

    pub(crate) fn add(&mut self, bytes: u64, seconds: f64) {
        self.bytes += bytes;
        self.seconds += seconds;
    }
}

/// Why the other end of a channel was not taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// It presented no certificate.
    NoCertificate,
    /// Its certificate is not signed by the pool's authority, or is not valid now. Why, as the
    /// certificate check gives it.
    NotThePool(String),
    /// Its certificate is signed by the pool's authority, but names none of the members
    /// expected, or more than one of them.
    WrongMember {
        expected: Vec<String>,
        named: Vec<String>,
    },
    /// It speaks no protocol this end speaks: TLS below 1.3, or nothing in common. Why.
    Protocol(String),
    /// Its certificate is signed by the pool's authority, but names no member, or more than one,
    /// to an end that takes any member: the names it carries.
    NotOneMember { named: Vec<String> },
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::NoCertificate => write!(f, "it presented no certificate"),
            Refused::NotThePool(why) => {
                write!(f, "its certificate is not the pool authority's: {why}")
            }
            Refused::WrongMember { expected, named } => write!(
                f,
                "its certificate names {named:?}, not one of the members expected, {expected:?}"
            ),
            Refused::Protocol(why) => write!(f, "it speaks no protocol this end speaks: {why}"),
            Refused::NotOneMember { named } => {
                write!(f, "its certificate names {named:?}, not exactly one member")
            }
        }
    }
}

/// Why a link or a channel failed.
#[derive(Debug)]
pub enum LinkError {
    /// The other end was not taken.
    Refused(Refused),
    /// The other end did not take this one: it refused it, or closed the channel before it was
    /// open. Why, as far as this end can tell.
    RefusedBy(String),
    /// The two ends speak different versions of the words (`mag_by::WIRE_VERSION`).
    Wire {
        ours: u64,
        theirs: u64,
    },
    /// The opening of a channel went wrong: the other end answered for another job, or said
    /// something other than `mag_by::Open`.
    Opening(String),
    /// The other end closed the channel, between two messages.
    Closed,
    /// The channel ended inside a message, or without the other end closing it.
    Truncated,
    /// The other end went away without closing the channel: it reset the connection, or TCP
    /// keepalive found it gone. The channel is ended.
    Lost(io::Error),
    /// Nothing came from the other end for as long as this end allows ([`TlsLink::with_silence`]).
    /// The channel is ended both ways, so a send waiting on the same peer ends too.
    Silent {
        after: std::time::Duration,
    },
    /// The channel did not open in the time each opening is given ([`TlsLink::with_opening`]):
    /// dialling, the handshake and the opening words together. That time, and whether this end
    /// was dialling or answering. The channel is given up.
    OpeningTimedOut {
        after: std::time::Duration,
        side: Side,
    },
    /// A header longer than [`MAX_HEADER`], sent or received.
    HeaderTooLong {
        bytes: u64,
    },
    /// A body of another length than expected: a buffer for it that is not its length, a word
    /// that came with a body, or a tensor whose body is not its `bytes`.
    Body {
        expected: u64,
        found: u64,
    },
    /// A message received before the last one's body was read.
    BodyUnread {
        left: u64,
    },
    /// A word that is not JSON of the type asked for.
    Word(serde_json::Error),
    /// A member name that cannot be put in a certificate: it must be a DNS name in lower case.
    Name(String),
    /// This end's certificate, key or authority could not be read, or do not hold together.
    Credentials(String),
    /// The private key file may be read, written or run by its group or by others (Unix): its
    /// path, and its mode. It must be its owner's alone, as `certs` writes it.
    KeyExposed {
        path: String,
        mode: u32,
    },
    /// TLS failed for a reason not listed above.
    Tls(rustls::Error),
    Io(io::Error),
}

/// Which end of a channel: the one that dialled, or the one that answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Dialling,
    Answering,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Side::Dialling => "dialling",
            Side::Answering => "answering",
        })
    }
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinkError::Refused(r) => write!(f, "the other end was refused: {r}"),
            LinkError::RefusedBy(why) => write!(f, "the other end did not take this one: {why}"),
            LinkError::Wire { ours, theirs } => write!(
                f,
                "the other end speaks wire version {theirs}, this one {ours}"
            ),
            LinkError::Opening(why) => write!(f, "the channel did not open: {why}"),
            LinkError::Closed => write!(f, "the other end closed the channel"),
            LinkError::Truncated => write!(f, "the channel ended inside a message, or unclosed"),
            LinkError::Lost(e) => write!(f, "the other end went away without closing: {e}"),
            LinkError::Silent { after } => write!(
                f,
                "nothing came from the other end for {:.1} s; the channel is ended",
                after.as_secs_f64()
            ),
            LinkError::OpeningTimedOut { after, side } => write!(
                f,
                "the channel did not open within {:.1} s, while {side}",
                after.as_secs_f64()
            ),
            LinkError::HeaderTooLong { bytes } => write!(
                f,
                "a header of {bytes} bytes; at most {MAX_HEADER} are taken"
            ),
            LinkError::Body { expected, found } => {
                write!(f, "a body of {found} bytes where {expected} were expected")
            }
            LinkError::BodyUnread { left } => write!(
                f,
                "a message received with {left} bytes of the last one's body unread"
            ),
            LinkError::Word(e) => write!(f, "not the word expected: {e}"),
            LinkError::Name(name) => write!(
                f,
                "`{name}` cannot name a member in a certificate: it must be a DNS name in lower \
                 case"
            ),
            LinkError::Credentials(why) => write!(f, "this end's credentials: {why}"),
            LinkError::KeyExposed { path, mode } => write!(
                f,
                "the key file {path} has mode {mode:03o}: its group and others must have no \
                 access to it (chmod 600)"
            ),
            LinkError::Tls(e) => write!(f, "TLS: {e}"),
            LinkError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl From<io::Error> for LinkError {
    /// TLS failures reach this end as I/O errors that hold them; they are sorted as TLS
    /// failures.
    fn from(e: io::Error) -> LinkError {
        match e.get_ref().and_then(|e| e.downcast_ref::<rustls::Error>()) {
            Some(tls) => tls::from_tls(tls.clone()),
            None => LinkError::Io(e),
        }
    }
}

impl std::error::Error for LinkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LinkError::Word(e) => Some(e),
            LinkError::Tls(e) => Some(e),
            LinkError::Io(e) | LinkError::Lost(e) => Some(e),
            _ => None,
        }
    }
}

/// Refuses a member name that a certificate cannot carry exactly: `mag-by`'s rule,
/// [`mag_by::is_member_name`], a DNS name in lower case, which the pool file and the start file
/// hold too.
pub fn check_name(name: &str) -> Result<(), LinkError> {
    if mag_by::is_member_name(name) {
        Ok(())
    } else {
        Err(LinkError::Name(name.into()))
    }
}
