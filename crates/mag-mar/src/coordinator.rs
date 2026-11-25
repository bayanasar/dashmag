//! The coordinator: what the member the start file names as coordinator runs, in the member
//! program, in place of dialling one.
//!
//! It keeps the pool's state in a `mag_mortar::Pool`, and the pool's ledger in the file its start
//! file names (`ledger`), and takes its members' joins and leaves into both:
//!
//! 1. **It begins a new ledger.** The pool begins with the coordinator's own member and devices,
//!    at the time it starts, and the ledger's header is written and synced before any member is
//!    taken. A ledger already at that place is refused: taking one up again, after a coordinator
//!    stops, is not built yet.
//! 2. **It takes a control channel from any member of the pool** (`Link::accept_any`): whoever
//!    holds a certificate the pool's authority signed naming one member, since a member that joins
//!    later is named by nothing the coordinator was told. A channel for a job is refused: no job
//!    is placed yet.
//! 3. **It checks a join** ([`check_join`]): the member it names and every offer's member must be
//!    the member the certificate names, and that member may hold no other live control channel,
//!    nor be the coordinator itself. A join that fails is refused, and the channel ended without
//!    a close.
//! 4. **It records a join**: one `mag_by::Change::Join` per offer, the member as
//!    [`recorded_member`] makes it, applied to the pool; then the new lines are written to the
//!    ledger and synced. A member with no offers is recorded nowhere, since the pool is told of
//!    devices, not of members: it holds its channel and leaves as any other.
//! 5. **It records a leave**: on `mag_by::Control::Leave`, one `mag_by::Change::Leave` per device
//!    the member joined with, the lines written and synced, and only then the coordinator closes
//!    its way of the channel. That close is the acknowledgement the member waits for, within its
//!    `give_back_seconds`. A control channel that ends without `Leave`, the member gone, is
//!    recorded the same way, and nothing is acknowledged.
//! 6. **On its own notice** it records its own devices' leave the same way, and returns. Its
//!    members' channels end without a close: their leave is not recorded by this coordinator.
//!
//! A ledger line that cannot be written stops the coordinator at once, with nothing after the
//! last line written acknowledged.
//!
//! Not built yet: the door; taking a job's channel; announcing a member to its peers; sending
//! placements; refusing a member's old certificate by its serial; and a coordinator that takes up
//! the ledger of one that stopped, with its members joining again.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mag_by::{CONTROL, Change, Control, Description, DeviceId, Ledger, Member, Offer, Trust};
use mag_mortar::Pool;

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
    /// The member holds another live control channel to the coordinator, or is the coordinator
    /// itself.
    Held(String),
    /// The channel has joined already.
    Twice,
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
            JoinRefused::Held(member) => write!(
                f,
                "`{member}` is in the pool already, on another channel or as the coordinator"
            ),
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

/// The ledger's file: written as the pool takes each change, and synced before the change is
/// acknowledged.
pub struct LedgerFile {
    path: String,
    file: File,
    /// The ledger's entries written so far.
    written: usize,
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
        };
        this.file.write_all(ledger.header_line().as_bytes())?;
        this.write(ledger)?;
        // The file's name in its directory, so that the new file is found after a crash.
        #[cfg(unix)]
        if let Some(dir) = std::path::Path::new(path).parent() {
            let dir = if dir.as_os_str().is_empty() {
                std::path::Path::new(".")
            } else {
                dir
            };
            File::open(dir)?.sync_all()?;
        }
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
}

fn describe(change: &Change) -> String {
    match change {
        Change::Join { offer, .. } => format!("the join of {}", offer.device),
        Change::Leave { device } => format!("the leave of {device}"),
        Change::Submit { graph, .. } => format!("the job `{}`", graph.job.name),
        Change::End { job } => format!("the end of `{job}`"),
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
    let mine: Vec<DeviceId> = offers.iter().map(|o| o.device.clone()).collect();
    let pool = Pool::from_description(&Description {
        at: now(),
        members: vec![me],
        offers,
        jobs: Vec::new(),
        formats: Vec::new(),
    })
    .expect("a description with no jobs");
    let ledger =
        LedgerFile::create(&start.ledger, pool.ledger()).map_err(|error| MemberError::Ledger {
            path: start.ledger.clone(),
            error,
        })?;
    say(
        &name,
        format_args!(
            "coordinates: the ledger is {}, begun with {} devices of its own",
            start.ledger,
            mine.len()
        ),
    );
    let mut keeper = Keeper {
        name: name.clone(),
        pool,
        ledger,
    };

    let (events_tx, events) = mpsc::channel();
    {
        let events = events_tx.clone();
        give_back_by(notice, give_back, name.clone(), move || {
            let _ = events.send(Event::Notice);
        });
    }
    take(link, name.clone(), events_tx);

    let mut members: BTreeMap<u64, Attached> = BTreeMap::new();
    loop {
        match events.recv().expect("the taking thread holds a sender") {
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
                        let held = who == name
                            || members.iter().any(|(other, o)| {
                                *other != id && o.joined.is_some() && o.peer.member == who
                            });
                        if held {
                            Err(JoinRefused::Held(who.clone()))
                        } else {
                            Ok(())
                        }
                    })
                };
                if let Err(why) = refused {
                    say(
                        &name,
                        format_args!("refuses the join on `{who}`'s channel: {why}"),
                    );
                    // Its sending half dropped unclosed: the member finds the channel cut short,
                    // never closed as an acknowledgement is.
                    if let Some(m) = members.get_mut(&id) {
                        m.sending = None;
                        if m.joined.is_none() {
                            members.remove(&id);
                        }
                    }
                    continue;
                }
                let recorded = recorded_member(&start.pool.members, &member);
                let devices: Vec<DeviceId> = offers.iter().map(|o| o.device.clone()).collect();
                let changes = offers
                    .into_iter()
                    .map(|offer| Change::Join {
                        member: Some(recorded.clone()),
                        offer,
                    })
                    .collect();
                keeper.record(changes)?;
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
                if let Some(m) = members.get_mut(&id) {
                    m.joined = Some(devices);
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
            Event::Ended { id, error } => {
                let Some(m) = members.remove(&id) else {
                    continue;
                };
                if let Some(devices) = m.joined {
                    keeper.record(leaves(&devices))?;
                    say(
                        &name,
                        format_args!(
                            "`{}` went without leaving ({error}): its {} devices recorded as left",
                            m.peer.member,
                            devices.len()
                        ),
                    );
                }
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
