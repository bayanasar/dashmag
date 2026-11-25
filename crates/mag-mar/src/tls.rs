//! The first implementation of the link: TCP with TLS 1.3, both ends authenticated by
//! certificates from the pool's own authority.
//!
//! A member dials another as the TLS client, and the other answers as the server, but the two
//! check each other alike: the certificate the other end presents must be signed by the pool's
//! authority and must name the member expected. The dialling end names the member it dials, and
//! rustls checks the name in the handshake. The answering end takes any certificate the
//! authority signed in the handshake, then reads the member it names and closes the channel
//! unless that member is one the caller expects. Only TLS 1.3 is offered or taken. Sessions are
//! not resumed, so every channel shows its certificate afresh, and no server name is sent in
//! the clear.

use std::collections::{BTreeMap, VecDeque};
use std::io::{self, Read, Write};
use std::net::{
    IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::client::{Resumption, WebPkiServerVerifier};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime};
use rustls::server::{NoServerSessionStorage, WebPkiClientVerifier};
use rustls::{
    CertificateError, ClientConfig, ClientConnection, DigitallySignedStruct, RootCertStore,
    ServerConfig, ServerConnection, SignatureScheme, StreamOwned,
};

use crate::{
    Channel, ChannelEnd, Identity, Incoming, Link, LinkError, Measure, Rate, Receiver, Refused,
    Sender, Side, check_name, frame,
};

/// How long the opening of a channel may take, dialling, the handshake and the opening words
/// together, before the channel is given up: the default of [`TlsLink::with_opening`].
pub const OPENING: Duration = Duration::from_secs(10);

/// How many connections an end that answers holds at once while they open, before a caller of
/// `accept` takes them: the default of [`TlsLink::with_answering`]. A connection past it is
/// closed at once.
pub const ANSWERING: usize = 64;

/// A member's credentials on its links: the pool's authority certificate, the member's own
/// certificate, signed by it and naming the member, and the member's private key.
pub struct Credentials {
    member: String,
    authority: CertificateDer<'static>,
    certificate: CertificateDer<'static>,
    key: PrivateKeyDer<'static>,
    serial: Vec<u8>,
}

impl Clone for Credentials {
    fn clone(&self) -> Self {
        Credentials {
            member: self.member.clone(),
            authority: self.authority.clone(),
            certificate: self.certificate.clone(),
            key: self.key.clone_key(),
            serial: self.serial.clone(),
        }
    }
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("member", &self.member)
            .field("serial", &self.identity().serial_hex())
            .finish_non_exhaustive()
    }
}

impl Credentials {
    /// Credentials from PEM text: the pool's authority certificate, this member's certificate
    /// and its private key. Refuses a member name a certificate cannot carry, and a certificate
    /// the authority did not sign, that does not name `member`, or that is not valid now for
    /// both ends of a link.
    pub fn from_pem(
        member: &str,
        authority: &str,
        certificate: &str,
        key: &str,
    ) -> Result<Credentials, LinkError> {
        check_name(member)?;
        let bad =
            |what: &str, e: &dyn std::fmt::Display| LinkError::Credentials(format!("{what}: {e}"));
        let authority = CertificateDer::from_pem_slice(authority.as_bytes())
            .map_err(|e| bad("the authority certificate", &e))?;
        let certificate = CertificateDer::from_pem_slice(certificate.as_bytes())
            .map_err(|e| bad("the member's certificate", &e))?;
        let key = PrivateKeyDer::from_pem_slice(key.as_bytes())
            .map_err(|e| bad("the member's key", &e))?;
        let serial = {
            let anchor = webpki::anchor_from_trusted_cert(&authority)
                .map_err(|e| bad("the authority certificate", &e))?;
            let ee = webpki::EndEntityCert::try_from(&certificate)
                .map_err(|e| bad("the member's certificate", &e))?;
            for usage in [
                webpki::KeyUsage::server_auth(),
                webpki::KeyUsage::client_auth(),
            ] {
                ee.verify_for_usage(
                    webpki::ALL_VERIFICATION_ALGS,
                    std::slice::from_ref(&anchor),
                    &[],
                    UnixTime::now(),
                    usage,
                    None,
                    None,
                )
                .map_err(|e| bad("the member's certificate is not the authority's", &e))?;
            }
            let name = ServerName::try_from(member).map_err(|e| bad("the member", &e))?;
            ee.verify_is_valid_for_subject_name(&name)
                .map_err(|e| bad(&format!("the certificate does not name `{member}`"), &e))?;
            ee.serial().to_vec()
        };
        Ok(Credentials {
            member: member.into(),
            authority,
            certificate,
            key,
            serial,
        })
    }

    /// Credentials as a start file gives them: its member, the authority it holds, and the
    /// certificate and key files it names. On Unix, a key file its group or others may read,
    /// write or run is refused, [`LinkError::KeyExposed`], before it is read: a key others can
    /// read lets them be this member, and one they can write lets them choose its key.
    pub fn from_start(start: &mag_by::Start) -> Result<Credentials, LinkError> {
        let failed =
            |path: &str, e: io::Error| LinkError::Credentials(format!("reading {path}: {e}"));
        let read = |path: &str, key: bool| -> Result<String, LinkError> {
            let mut file = std::fs::File::open(path).map_err(|e| failed(path, e))?;
            #[cfg(unix)]
            if key {
                use std::os::unix::fs::PermissionsExt;
                let mode = file
                    .metadata()
                    .map_err(|e| failed(path, e))?
                    .permissions()
                    .mode();
                if mode & 0o077 != 0 {
                    return Err(LinkError::KeyExposed {
                        path: path.into(),
                        mode: mode & 0o7777,
                    });
                }
            }
            #[cfg(not(unix))]
            let _ = key;
            let mut text = String::new();
            file.read_to_string(&mut text)
                .map_err(|e| failed(path, e))?;
            Ok(text)
        };
        Credentials::from_pem(
            &start.member,
            &start.authority,
            &read(&start.certificate_file, false)?,
            &read(&start.key_file, true)?,
        )
    }

    /// The member these credentials are for.
    pub fn member(&self) -> &str {
        &self.member
    }

    /// This member as the other end of a link sees it.
    pub fn identity(&self) -> Identity {
        Identity {
            member: self.member.clone(),
            serial: self.serial.clone(),
        }
    }
}

fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

fn roots(authority: &CertificateDer<'static>) -> Result<RootCertStore, LinkError> {
    let mut roots = RootCertStore::empty();
    roots.add(authority.clone()).map_err(LinkError::Tls)?;
    Ok(roots)
}

/// rustls's own check of the certificate the answering end presents, which reports the names a
/// certificate of the wrong member gives as the names alone.
#[derive(Debug)]
struct Named(Arc<WebPkiServerVerifier>);

impl ServerCertVerifier for Named {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        self.0
            .verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now)
            .map_err(|e| match e {
                rustls::Error::InvalidCertificate(
                    CertificateError::NotValidForName
                    | CertificateError::NotValidForNameContext { .. },
                ) => rustls::Error::InvalidCertificate(CertificateError::NotValidForNameContext {
                    expected: server_name.to_owned(),
                    presented: webpki::EndEntityCert::try_from(end_entity)
                        .map(|ee| ee.valid_dns_names().map(String::from).collect())
                        .unwrap_or_default(),
                }),
                e => e,
            })
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.0.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.0.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.supported_verify_schemes()
    }
}

fn client_config(c: &Credentials) -> Result<ClientConfig, LinkError> {
    let verifier =
        WebPkiServerVerifier::builder_with_provider(Arc::new(roots(&c.authority)?), provider())
            .build()
            .map_err(|e| LinkError::Credentials(e.to_string()))?;
    let mut config = ClientConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(LinkError::Tls)?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(Named(verifier)))
        .with_client_auth_cert(vec![c.certificate.clone()], c.key.clone_key())
        .map_err(LinkError::Tls)?;
    config.resumption = Resumption::disabled();
    config.enable_sni = false;
    Ok(config)
}

fn server_config(c: &Credentials) -> Result<ServerConfig, LinkError> {
    let verifier =
        WebPkiClientVerifier::builder_with_provider(Arc::new(roots(&c.authority)?), provider())
            .build()
            .map_err(|e| LinkError::Credentials(e.to_string()))?;
    let mut config = ServerConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(LinkError::Tls)?
        .with_client_cert_verifier(verifier)
        .with_single_cert(vec![c.certificate.clone()], c.key.clone_key())
        .map_err(LinkError::Tls)?;
    config.session_storage = Arc::new(NoServerSessionStorage {});
    config.send_tls13_tickets = 0;
    Ok(config)
}

/// Sorts a TLS failure into a refusal of the other end, a refusal by it, or another failure.
pub(crate) fn from_tls(e: rustls::Error) -> LinkError {
    match e {
        rustls::Error::NoCertificatesPresented => LinkError::Refused(Refused::NoCertificate),
        rustls::Error::InvalidCertificate(CertificateError::NotValidForNameContext {
            expected,
            presented,
        }) => LinkError::Refused(Refused::WrongMember {
            expected: vec![expected.to_str().into_owned()],
            named: presented,
        }),
        rustls::Error::InvalidCertificate(why) => {
            LinkError::Refused(Refused::NotThePool(format!("{why:?}")))
        }
        rustls::Error::PeerIncompatible(why) => {
            LinkError::Refused(Refused::Protocol(format!("{why:?}")))
        }
        rustls::Error::AlertReceived(alert) => LinkError::RefusedBy(format!("{alert:?}")),
        other => LinkError::Tls(other),
    }
}

fn timed_out() -> io::Error {
    io::Error::new(
        io::ErrorKind::TimedOut,
        "the channel did not open in the time allowed",
    )
}

/// The socket under a channel while it opens. Each read and write may take only what is left of
/// one deadline for the whole opening, dialling included, so a peer that sends a byte now and
/// then cannot hold the opening, nor the thread it runs on, for longer than that.
struct Sock {
    tcp: TcpStream,
    deadline: Instant,
}

impl Sock {
    fn bound(&self) -> io::Result<()> {
        let left = self.deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(timed_out());
        }
        self.tcp.set_read_timeout(Some(left))?;
        self.tcp.set_write_timeout(Some(left))
    }
}

impl Read for Sock {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.bound()?;
        self.tcp.read(buf)
    }
}

impl Write for Sock {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.bound()?;
        self.tcp.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.tcp.flush()
    }
}

/// The stream under a channel while it opens, as the end that dialled or the end that answered
/// holds it.
enum Stream {
    Dialled(StreamOwned<ClientConnection, Sock>),
    Answered(StreamOwned<ServerConnection, Sock>),
}

impl Stream {
    fn handshake(&mut self) -> Result<(), LinkError> {
        match self {
            Stream::Dialled(s) => {
                while s.conn.is_handshaking() {
                    s.conn.complete_io(&mut s.sock)?;
                }
            }
            Stream::Answered(s) => {
                while s.conn.is_handshaking() {
                    s.conn.complete_io(&mut s.sock)?;
                }
            }
        }
        Ok(())
    }

    fn peer_certificate(&self) -> Option<&CertificateDer<'static>> {
        match self {
            Stream::Dialled(s) => s.conn.peer_certificates(),
            Stream::Answered(s) => s.conn.peer_certificates(),
        }
        .and_then(|chain| chain.first())
    }

    /// Tells the other end the channel is closed, and closes this end's writing half: a channel
    /// refused before it opened.
    fn close(&mut self) -> Result<(), LinkError> {
        match self {
            Stream::Dialled(s) => s.conn.send_close_notify(),
            Stream::Answered(s) => s.conn.send_close_notify(),
        }
        self.flush()?;
        let tcp = match self {
            Stream::Dialled(s) => &s.sock.tcp,
            Stream::Answered(s) => &s.sock.tcp,
        };
        tcp.shutdown(Shutdown::Write)?;
        Ok(())
    }

    fn into_parts(self) -> (rustls::Connection, TcpStream) {
        match self {
            Stream::Dialled(s) => (s.conn.into(), s.sock.tcp),
            Stream::Answered(s) => (s.conn.into(), s.sock.tcp),
        }
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Stream::Dialled(s) => s.read(buf),
            Stream::Answered(s) => s.read(buf),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Stream::Dialled(s) => s.write(buf),
            Stream::Answered(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Stream::Dialled(s) => s.flush(),
            Stream::Answered(s) => s.flush(),
        }
    }
}

/// Connects to `address` by `deadline`: each address it resolves to in turn, each given what is
/// left of the time, until one answers or the time is up. Resolving a name is the system
/// resolver's, bounded by its own timeouts; an address given as numbers resolves at once.
fn dial(address: &str, deadline: Instant) -> io::Result<TcpStream> {
    connect_any(address, address.to_socket_addrs()?, deadline)
}

/// Connects to the first of `addresses` that answers by `deadline`, `address` as they came.
fn connect_any(
    address: &str,
    addresses: impl IntoIterator<Item = SocketAddr>,
    deadline: Instant,
) -> io::Result<TcpStream> {
    let mut last = None;
    for at in addresses {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        match TcpStream::connect_timeout(&at, left) {
            Ok(tcp) => return Ok(tcp),
            Err(e) => last = Some(e),
        }
    }
    if Instant::now() >= deadline {
        return Err(timed_out());
    }
    Err(last.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("`{address}` resolves to no address"),
        )
    }))
}

/// How TCP watches the other end of a channel while nothing crosses it: after `idle` of quiet, a
/// probe every `interval`, and the other end taken for gone once `probes` of them go unanswered.
/// Then a receive waiting on the channel fails, [`LinkError::Lost`].
///
/// It finds a peer whose machine went away, switched off, cut from the network, or behind an
/// address translation that forgot the connection, while the channel is quiet: as when a
/// member waits for the next word. While this end has sent bytes the other has not acknowledged,
/// TCP probes nothing, and the system's own limit on retransmitting decides (on Linux about 15
/// minutes, `tcp_retries2`). A peer whose machine is up answers probes from its kernel, even
/// while its program is stopped. A limit on silence ([`TlsLink::with_silence`]) bounds both.
///
/// It sets no bound on unacknowledged data (Linux's `TCP_USER_TIMEOUT`): that bound also ends a
/// channel whose peer is alive and busy, not reading, for longer than it, which a member computing
/// a long step is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Keepalive {
    pub idle: Duration,
    pub interval: Duration,
    pub probes: u32,
}

impl Keepalive {
    /// How long after its last sign of life a peer that went away is found gone: `idle`, and
    /// `interval` for each probe.
    pub fn limit(&self) -> Duration {
        self.idle + self.interval * self.probes
    }
}

/// The default [`Keepalive`]: probes after 10 s of quiet, every 5 s, three of them, so that a
/// peer that went away is found within 25 s, inside most members' time to give a device back.
pub const KEEPALIVE: Keepalive = Keepalive {
    idle: Duration::from_secs(10),
    interval: Duration::from_secs(5),
    probes: 3,
};

/// Sets `keepalive` on a channel's socket, or turns it off.
fn watch(tcp: &TcpStream, keepalive: Option<Keepalive>) -> io::Result<()> {
    let sock = socket2::SockRef::from(tcp);
    let Some(k) = keepalive else {
        return sock.set_keepalive(false);
    };
    let params = socket2::TcpKeepalive::new().with_time(k.idle);
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        windows
    ))]
    let params = params.with_interval(k.interval).with_retries(k.probes);
    sock.set_tcp_keepalive(&params)
}

/// A failure of an opening that was given `opening` and was to end by `deadline`, at `side`.
/// While a channel opens, its socket has no timeout but what the deadline leaves, so a read or a
/// write that waits that out is the opening timing out ([`LinkError::OpeningTimedOut`]): Linux
/// reports it as `WouldBlock`, other systems as `TimedOut`. So is a `TimedOut` once the deadline
/// has passed: a dial given up at it, or the deadline found passed before the next read or
/// write. A `TimedOut` before it is the system's own, such as TCP keepalive finding the peer
/// gone, and stays as it came, as does every other failure: a peer that refused this end, or
/// dropped the connection.
fn ran_out(e: LinkError, opening: Duration, deadline: Instant, side: Side) -> LinkError {
    match &e {
        LinkError::Io(io)
            if io.kind() == io::ErrorKind::WouldBlock
                || (io.kind() == io::ErrorKind::TimedOut && Instant::now() >= deadline) =>
        {
            LinkError::OpeningTimedOut {
                after: opening,
                side,
            }
        }
        _ => e,
    }
}

/// A socket whose opening must end by `deadline`, watched by `keepalive`.
fn prepare(
    tcp: TcpStream,
    deadline: Instant,
    keepalive: Option<Keepalive>,
) -> Result<Sock, LinkError> {
    tcp.set_nodelay(true)?;
    watch(&tcp, keepalive)?;
    Ok(Sock { tcp, deadline })
}

/// The serial of the certificate the other end presented, once the handshake has checked it.
fn serial(stream: &Stream) -> Result<Vec<u8>, LinkError> {
    let cert = stream
        .peer_certificate()
        .ok_or(LinkError::Refused(Refused::NoCertificate))?;
    let ee = webpki::EndEntityCert::try_from(cert)
        .map_err(|e| LinkError::Refused(Refused::NotThePool(e.to_string())))?;
    Ok(ee.serial().to_vec())
}

/// The one member of `expected` the certificate names.
fn member_named(cert: &CertificateDer<'_>, expected: &[&str]) -> Result<String, Refused> {
    let ee =
        webpki::EndEntityCert::try_from(cert).map_err(|e| Refused::NotThePool(e.to_string()))?;
    let named: Vec<&str> = expected
        .iter()
        .copied()
        .filter(|m| {
            ServerName::try_from(*m).is_ok_and(|n| ee.verify_is_valid_for_subject_name(&n).is_ok())
        })
        .collect();
    match named.as_slice() {
        [one] => Ok((*one).into()),
        _ => Err(Refused::WrongMember {
            expected: expected.iter().map(|m| (*m).into()).collect(),
            named: ee.valid_dns_names().map(String::from).collect(),
        }),
    }
}

/// A member's end of its links over TCP with TLS 1.3: it dials other members, and if it was
/// bound to an address, it answers them.
///
/// An end that answers opens each connection it takes on a thread of its own, from the first
/// `accept` on, so that a peer that stalls its opening holds no other's: `accept` returns the
/// next channel whose opening has finished, whichever connection came first. It holds at most
/// [`TlsLink::with_answering`] connections at once while they open, and closes any past that at
/// once, so that a flood of connections cannot take every thread.
pub struct TlsLink {
    credentials: Credentials,
    client: Arc<ClientConfig>,
    server: Arc<ServerConfig>,
    /// The thread that takes connections, once the first `accept` has started it. Dropped
    /// before the listener, it ends the thread, which holds a listener of its own.
    answerer: Mutex<Option<Answerer>>,
    listener: Option<TcpListener>,
    opening: Duration,
    keepalive: Option<Keepalive>,
    silence: Option<Duration>,
    answering: usize,
    rates: Arc<Mutex<BTreeMap<String, Rate>>>,
}

impl TlsLink {
    /// An end that only dials: a member behind address translation, which no other can reach.
    pub fn new(credentials: Credentials) -> Result<TlsLink, LinkError> {
        Ok(TlsLink {
            client: Arc::new(client_config(&credentials)?),
            server: Arc::new(server_config(&credentials)?),
            credentials,
            answerer: Mutex::new(None),
            listener: None,
            opening: OPENING,
            keepalive: Some(KEEPALIVE),
            silence: None,
            answering: ANSWERING,
            rates: Arc::default(),
        })
    }

    /// The same end, giving each channel `opening` to open, before it is given up
    /// ([`LinkError::OpeningTimedOut`]): dialling, the handshake and the opening words together.
    /// [`OPENING`] unless set.
    pub fn with_opening(mut self, opening: Duration) -> TlsLink {
        self.opening = opening;
        self.restart_answering();
        self
    }

    /// The same end, watching each channel's socket with `keepalive`, or with none.
    /// [`KEEPALIVE`] unless set.
    pub fn with_keepalive(mut self, keepalive: Option<Keepalive>) -> TlsLink {
        self.keepalive = keepalive;
        self.restart_answering();
        self
    }

    /// The same end, giving up a channel on which nothing has come for `silence` while this end
    /// waits to receive: [`LinkError::Silent`], and the channel ended both ways, so that a send
    /// waiting on the same peer ends too. None unless set, and a limit of zero is none.
    ///
    /// None is the default because the link cannot tell a quiet channel from a stopped peer: a
    /// job may wait, or take long over a step, while its channels carry nothing. A member that
    /// knows how often a channel's words come sets a limit above that.
    pub fn with_silence(mut self, silence: Option<Duration>) -> TlsLink {
        self.silence = silence.filter(|s| !s.is_zero());
        self
    }

    /// The same end, holding at most `limit` connections at once while they open, before a
    /// caller of `accept` takes them, at least one. A connection past it is closed at once, so
    /// that a flood of connections, each held until its opening's deadline, cannot take every
    /// thread; a member closed so dials again. [`ANSWERING`] unless set.
    pub fn with_answering(mut self, limit: usize) -> TlsLink {
        self.answering = limit.max(1);
        self.restart_answering();
        self
    }

    /// Ends the thread that takes connections, if one runs, so that the next `accept` starts one
    /// with the end's settings as they are now. The connections it held are closed, those still
    /// opening once their openings end.
    fn restart_answering(&mut self) {
        let answerer = self.answerer.get_mut().unwrap_or_else(|p| p.into_inner());
        drop(answerer.take());
    }

    /// An end that dials and answers at `address`, such as `0.0.0.0:7340`.
    pub fn bind(
        credentials: Credentials,
        address: impl ToSocketAddrs,
    ) -> Result<TlsLink, LinkError> {
        let mut link = TlsLink::new(credentials)?;
        link.listener = Some(TcpListener::bind(address)?);
        Ok(link)
    }

    /// The address it answers at, if it was bound to one.
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.listener.as_ref().and_then(|l| l.local_addr().ok())
    }

    /// This member's credentials.
    pub fn credentials(&self) -> &Credentials {
        &self.credentials
    }

    /// The channel is open: the opening's deadline is dropped, the silence allowed is set, and
    /// the stream is divided into the channel's two halves.
    fn opened(&self, stream: Stream, peer: Identity, job: String) -> Result<TlsChannel, LinkError> {
        let (tls, tcp) = stream.into_parts();
        tcp.set_read_timeout(self.silence)?;
        tcp.set_write_timeout(None)?;
        let receiving = tcp.try_clone()?;
        let shared = Arc::new(Shared {
            tls: Mutex::new(tls),
            silence: self.silence,
            silent: AtomicBool::new(false),
        });
        let end = End {
            peer,
            job,
            rates: self.rates.clone(),
        };
        Ok(TlsChannel {
            receiver: TlsReceiver {
                inbound: Inbound {
                    shared: shared.clone(),
                    tcp: receiving,
                    buf: vec![0; READ].into_boxed_slice(),
                    start: 0,
                    end: 0,
                },
                end: end.clone(),
                left: 0,
                received: Measure::default(),
            },
            sender: TlsSender {
                outbound: Outbound {
                    shared,
                    tcp,
                    records: Vec::new(),
                    closed: false,
                },
                end,
                sent: Measure::default(),
            },
        })
    }
}

impl Link for TlsLink {
    type Channel = TlsChannel;

    fn open(&self, member: &str, address: &str, job: &str) -> Result<TlsChannel, LinkError> {
        check_name(member)?;
        let name =
            ServerName::try_from(member.to_string()).map_err(|_| LinkError::Name(member.into()))?;
        let deadline = Instant::now() + self.opening;
        let opening = || {
            let sock = prepare(dial(address, deadline)?, deadline, self.keepalive)?;
            let conn = ClientConnection::new(self.client.clone(), name).map_err(LinkError::Tls)?;
            let mut stream = Stream::Dialled(StreamOwned::new(conn, sock));
            // The other end's certificate is checked here, its name with it. This end's is
            // checked by the other end after this end's part of the handshake ends, so its answer
            // to the opening is what says it was taken.
            stream.handshake()?;
            let peer = Identity {
                member: member.into(),
                serial: serial(&stream)?,
            };
            frame::open_as_dialler(&mut stream, job, mag_by::WIRE_VERSION)?;
            Ok((stream, peer))
        };
        let (stream, peer) =
            opening().map_err(|e| ran_out(e, self.opening, deadline, Side::Dialling))?;
        self.opened(stream, peer, job.into())
    }

    fn accept(&self, expected: &[&str]) -> Result<TlsChannel, LinkError> {
        for m in expected {
            check_name(m)?;
        }
        self.answer(|cert| member_named(cert, expected))
    }

    fn accept_any(&self) -> Result<TlsChannel, LinkError> {
        self.answer(member_any)
    }

    fn rate(&self, member: &str) -> Option<Rate> {
        self.rates.lock().ok()?.get(member).copied()
    }
}

impl TlsLink {
    /// Takes the next connection whose opening has finished as far as it goes on its own thread,
    /// and opens it as a channel from the member `named` reads from the certificate it presents,
    /// which the handshake has checked is the pool authority's.
    fn answer(
        &self,
        named: impl FnOnce(&CertificateDer<'_>) -> Result<String, Refused>,
    ) -> Result<TlsChannel, LinkError> {
        let Answered {
            mut stream,
            open,
            deadline,
        } = self.answering()?.next()?;
        let (peer, job) = taken(&mut stream, open, named)
            .map_err(|e| ran_out(e, self.opening, deadline, Side::Answering))?;
        self.opened(stream, peer, job)
    }

    /// The connections this end holds while they open, the thread that takes them started if
    /// this is the first `accept`.
    fn answering(&self) -> Result<Arc<Answering>, LinkError> {
        let listener = self.listener.as_ref().ok_or_else(|| {
            LinkError::Io(io::Error::new(
                io::ErrorKind::NotConnected,
                "this end was bound to no address, and only dials",
            ))
        })?;
        let mut answerer = self.answerer.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(answerer) = answerer.as_ref() {
            return Ok(answerer.answering.clone());
        }
        let wake = waking(listener.local_addr()?);
        let taking = listener.try_clone()?;
        let answering = Arc::new(Answering {
            held: Mutex::new(Held {
                count: 0,
                done: VecDeque::new(),
            }),
            finished: Condvar::new(),
            limit: self.answering,
            stop: AtomicBool::new(false),
        });
        let thread = {
            let answering = answering.clone();
            let server = self.server.clone();
            let (opening, keepalive) = (self.opening, self.keepalive);
            thread::Builder::new()
                .name("mag-mar answers".into())
                .spawn(move || take_connections(taking, &answering, server, opening, keepalive))?
        };
        *answerer = Some(Answerer {
            answering: answering.clone(),
            thread: Some(thread),
            wake,
        });
        Ok(answering)
    }
}

/// A connection an end that answers has taken, opened as far as it goes before a caller of
/// `accept` checks the member it is from: the handshake done, which checked that its
/// certificate is the pool authority's, and the opening the other end sent read, or why it was
/// not.
struct Answered {
    stream: Stream,
    open: Result<mag_by::Open, LinkError>,
    deadline: Instant,
}

/// What an end that answers holds of the connections it has taken and not yet handed to a
/// caller.
struct Held {
    /// The connections held: opening on their threads, or in `done`.
    count: usize,
    /// The openings finished as far as they go, or failed, in the order they finished.
    done: VecDeque<Result<Answered, LinkError>>,
}

/// The connections an end that answers holds while they open, shared by the thread that takes
/// them, the thread each opens on, and the callers of `accept`.
struct Answering {
    held: Mutex<Held>,
    /// Told each time an opening finishes, or fails.
    finished: Condvar,
    /// The most connections held at once.
    limit: usize,
    /// Set when the end is dropped: the thread that takes connections ends.
    stop: AtomicBool,
}

impl Answering {
    fn held(&self) -> MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Holds one more connection, unless `limit` are held already.
    fn hold(&self) -> bool {
        let mut held = self.held();
        if held.count >= self.limit {
            return false;
        }
        held.count += 1;
        true
    }

    /// Hands on a connection held, opened as far as it goes, or why it failed.
    fn finish(&self, answered: Result<Answered, LinkError>) {
        self.held().done.push_back(answered);
        self.finished.notify_one();
    }

    /// The next opening finished, or failed, waiting for one; it is no longer held.
    fn next(&self) -> Result<Answered, LinkError> {
        let mut held = self.held();
        loop {
            if let Some(answered) = held.done.pop_front() {
                held.count -= 1;
                return answered;
            }
            held = self.finished.wait(held).unwrap_or_else(|p| p.into_inner());
        }
    }
}

/// The thread that takes an answering end's connections, from its first `accept` until it is
/// dropped.
struct Answerer {
    answering: Arc<Answering>,
    thread: Option<thread::JoinHandle<()>>,
    /// An address at which a connection reaches the listener, to wake the thread.
    wake: SocketAddr,
}

impl Drop for Answerer {
    /// Ends the thread: a connection of its own wakes it from waiting for the next, and it ends
    /// on seeing the stop, closing its listener, so that the end's address is free once the end
    /// is dropped. Openings still running end by their deadline, and are dropped.
    fn drop(&mut self) {
        self.answering.stop.store(true, Ordering::SeqCst);
        if TcpStream::connect_timeout(&self.wake, Duration::from_secs(1)).is_ok()
            && let Some(thread) = self.thread.take()
        {
            let _ = thread.join();
        }
    }
}

/// An address at which a connection reaches a listener bound at `bound`: the same, or for a
/// listener bound to every address, loopback.
fn waking(bound: SocketAddr) -> SocketAddr {
    let ip = match bound.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    SocketAddr::new(ip, bound.port())
}

/// Takes connections on `listener` until `answering` is stopped, and opens each on a thread of
/// its own, so that one whose peer stalls holds no other: given `opening` from when it is taken,
/// and watched by `keepalive`. One past `answering`'s limit is closed at once. A connection the
/// system fails to take is handed on as the failure, and the next is taken a little later, so
/// that a failure that lasts, such as running out of file descriptors, does not spin.
fn take_connections(
    listener: TcpListener,
    answering: &Arc<Answering>,
    server: Arc<ServerConfig>,
    opening: Duration,
    keepalive: Option<Keepalive>,
) {
    loop {
        let taken = listener.accept();
        if answering.stop.load(Ordering::SeqCst) {
            return;
        }
        let tcp = match taken {
            Ok((tcp, _)) => tcp,
            Err(e) => {
                if answering.hold() {
                    answering.finish(Err(LinkError::Io(e)));
                }
                thread::sleep(Duration::from_millis(100));
                continue;
            }
        };
        let deadline = Instant::now() + opening;
        if !answering.hold() {
            drop(tcp);
            continue;
        }
        let on_its_own = {
            let answering = answering.clone();
            let server = server.clone();
            move || {
                let answered = opened_so_far(tcp, server, keepalive, deadline)
                    .map_err(|e| ran_out(e, opening, deadline, Side::Answering));
                answering.finish(answered);
            }
        };
        if let Err(e) = thread::Builder::new()
            .name("mag-mar opens".into())
            .spawn(on_its_own)
        {
            answering.finish(Err(LinkError::Io(e)));
        }
    }
}

/// Opens a connection an end that answers has taken as far as it goes before a caller checks the
/// member it is from: the handshake, which checks that the other end's certificate is the pool
/// authority's, and the opening it sends, read.
fn opened_so_far(
    tcp: TcpStream,
    server: Arc<ServerConfig>,
    keepalive: Option<Keepalive>,
    deadline: Instant,
) -> Result<Answered, LinkError> {
    let sock = prepare(tcp, deadline, keepalive)?;
    let conn = ServerConnection::new(server).map_err(LinkError::Tls)?;
    let mut stream = Stream::Answered(StreamOwned::new(conn, sock));
    stream.handshake()?;
    let open = frame::read_open(&mut stream);
    Ok(Answered {
        stream,
        open,
        deadline,
    })
}

/// Takes a connection opened so far as a channel from the member `named` reads from its
/// certificate, and answers its opening: the member, and the job. A member `named` refuses is
/// told the channel is closed, and the refusal returned before any failure of the opening.
fn taken(
    stream: &mut Stream,
    open: Result<mag_by::Open, LinkError>,
    named: impl FnOnce(&CertificateDer<'_>) -> Result<String, Refused>,
) -> Result<(Identity, String), LinkError> {
    let cert = stream
        .peer_certificate()
        .ok_or(LinkError::Refused(Refused::NoCertificate))?;
    let member = match named(cert) {
        Ok(member) => member,
        Err(refused) => {
            let _ = stream.close();
            return Err(LinkError::Refused(refused));
        }
    };
    let peer = Identity {
        member,
        serial: serial(stream)?,
    };
    let job = frame::answer_open(stream, mag_by::WIRE_VERSION, open?)?;
    Ok((peer, job))
}

/// What the two halves of an open channel share: its TLS session, locked only while records
/// are made or read, and never while either half waits on the socket. So the sending half never
/// waits for its own end to receive, nor the receiving half for it to send.
///
/// Only the sending half writes to the socket. What the session must say on the receiving
/// half's account (TLS 1.3's answer to a key update) waits in it for the next send, as TLS 1.3
/// allows.
struct Shared {
    tls: Mutex<rustls::Connection>,
    silence: Option<Duration>,
    /// Set when the receiving half found the other end silent, and ended the channel.
    silent: AtomicBool,
}

impl Shared {
    fn tls(&self) -> MutexGuard<'_, rustls::Connection> {
        self.tls.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// A failure on the open channel, as either half reports it.
    fn failed(&self, e: LinkError) -> LinkError {
        if let Some(after) = self.silence
            && self.silent.load(Ordering::SeqCst)
        {
            return LinkError::Silent { after };
        }
        match e {
            LinkError::Io(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::ConnectionReset
                        | io::ErrorKind::ConnectionAborted
                        | io::ErrorKind::BrokenPipe
                        | io::ErrorKind::TimedOut
                ) =>
            {
                LinkError::Lost(e)
            }
            e => e,
        }
    }
}

/// Plaintext made into records at a time: rustls's own limit on what it holds to send.
const RECORDS: usize = 64 << 10;

/// Bytes read from the socket at a time, before the session takes them.
const READ: usize = 64 << 10;

/// The one member the certificate names, whoever it is: its one DNS name, which must be a member
/// name. A certificate that names more than one could stand for any of them, and is refused.
fn member_any(cert: &CertificateDer<'_>) -> Result<String, Refused> {
    let ee =
        webpki::EndEntityCert::try_from(cert).map_err(|e| Refused::NotThePool(e.to_string()))?;
    let named: Vec<String> = ee.valid_dns_names().map(String::from).collect();
    match named.as_slice() {
        [one] if mag_by::is_member_name(one) => Ok(one.clone()),
        _ => Err(Refused::NotOneMember { named }),
    }
}

/// The channel's way out: the socket written by the sending half alone.
struct Outbound {
    shared: Arc<Shared>,
    tcp: TcpStream,
    /// Records made and not yet written; reused.
    records: Vec<u8>,
    closed: bool,
}

impl Outbound {
    /// Takes every record the session holds to send.
    fn take_records(&mut self, tls: &mut rustls::Connection) -> io::Result<()> {
        self.records.clear();
        while tls.wants_write() {
            tls.write_tls(&mut self.records)?;
        }
        Ok(())
    }

    fn close(&mut self) -> io::Result<()> {
        self.closed = true;
        {
            let shared = self.shared.clone();
            let mut tls = shared.tls();
            tls.send_close_notify();
            self.take_records(&mut tls)?;
        }
        self.tcp.write_all(&self.records)?;
        self.tcp.shutdown(Shutdown::Write)
    }
}

impl Write for Outbound {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let taken = {
            let shared = self.shared.clone();
            let mut tls = shared.tls();
            let taken = tls.writer().write(&buf[..buf.len().min(RECORDS)])?;
            self.take_records(&mut tls)?;
            taken
        };
        // The socket may wait here for the other end to read, with the session unlocked: this
        // end's receiving half goes on reading meanwhile.
        self.tcp.write_all(&self.records)?;
        Ok(taken)
    }

    /// Each write has written its records already.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for Outbound {
    /// A sending half dropped unclosed ends its way of the channel without telling the other
    /// end, which finds it cut short, even while the receiving half lives on.
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.tcp.shutdown(Shutdown::Write);
        }
    }
}

/// The channel's way in: the socket read by the receiving half alone.
struct Inbound {
    shared: Arc<Shared>,
    tcp: TcpStream,
    /// Bytes read from the socket, `start..end` of them not yet given to the session.
    buf: Box<[u8]>,
    start: usize,
    end: usize,
}

impl Inbound {
    /// Whether a failed read is the silence allowed running out.
    fn silence(&self, e: &io::Error, waited: Instant) -> bool {
        let Some(limit) = self.shared.silence else {
            return false;
        };
        match e.kind() {
            io::ErrorKind::WouldBlock => true,
            // Some systems report the socket's timeout so; keepalive reports a peer gone so too.
            io::ErrorKind::TimedOut => waited.elapsed() >= limit,
            _ => false,
        }
    }
}

impl Read for Inbound {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        loop {
            {
                let mut tls = self.shared.tls();
                match tls.reader().read(out) {
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                    done => return done,
                }
                if self.start < self.end {
                    let taken = tls.read_tls(&mut &self.buf[self.start..self.end])?;
                    if taken == 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "the TLS session took no more bytes",
                        ));
                    }
                    self.start += taken;
                    tls.process_new_packets()
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    continue;
                }
            }
            // Nothing is left to decrypt: wait on the socket with the session unlocked, so that
            // this end's sending half goes on sending meanwhile.
            let waited = Instant::now();
            match self.tcp.read(&mut self.buf) {
                Ok(read) => {
                    (self.start, self.end) = (0, read);
                    if read == 0 {
                        // The socket ended. The session is told, and the reader then says
                        // whether the other end closed the channel first.
                        self.shared.tls().read_tls(&mut io::empty())?;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) if self.silence(&e, waited) => {
                    self.shared.silent.store(true, Ordering::SeqCst);
                    let _ = self.tcp.shutdown(Shutdown::Both);
                    return Err(e);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// Who a half's channel is with and what it is for, and the link's rates by member.
#[derive(Clone)]
struct End {
    peer: Identity,
    job: String,
    rates: Arc<Mutex<BTreeMap<String, Rate>>>,
}

impl End {
    fn count(&self, sent: bool, m: Measure) {
        let mut all = self.rates.lock().unwrap_or_else(|p| p.into_inner());
        let rate = all.entry(self.peer.member.clone()).or_default();
        let measure = if sent {
            &mut rate.sent
        } else {
            &mut rate.received
        };
        measure.add(m.bytes, m.seconds);
    }
}

/// One job's channel to another member, over its own TCP connection and TLS session: its
/// sending half and its receiving half, used together by one thread, or split for two.
///
/// Dropped without [`Sender::close`], it ends without telling the other end, which then finds
/// it [`LinkError::Truncated`].
pub struct TlsChannel {
    sender: TlsSender,
    receiver: TlsReceiver,
}

/// The sending half of a [`TlsChannel`].
pub struct TlsSender {
    outbound: Outbound,
    end: End,
    sent: Measure,
}

/// The receiving half of a [`TlsChannel`].
pub struct TlsReceiver {
    inbound: Inbound,
    end: End,
    /// Bytes of the last body received not yet read.
    left: u64,
    received: Measure,
}

impl ChannelEnd for TlsSender {
    fn peer(&self) -> &Identity {
        &self.end.peer
    }

    fn job(&self) -> &str {
        &self.end.job
    }

    fn rate(&self) -> Rate {
        Rate {
            sent: self.sent,
            received: Measure::default(),
        }
    }
}

impl Sender for TlsSender {
    fn send(&mut self, header: &[u8], body: &[u8]) -> Result<(), LinkError> {
        let m = frame::write(&mut self.outbound, header, body)
            .map_err(|e| self.outbound.shared.failed(e))?;
        self.sent.add(m.bytes, m.seconds);
        self.end.count(true, m);
        Ok(())
    }

    fn close(mut self) -> Result<(), LinkError> {
        self.outbound
            .close()
            .map_err(|e| self.outbound.shared.failed(e.into()))
    }
}

impl ChannelEnd for TlsReceiver {
    fn peer(&self) -> &Identity {
        &self.end.peer
    }

    fn job(&self) -> &str {
        &self.end.job
    }

    fn rate(&self) -> Rate {
        Rate {
            sent: Measure::default(),
            received: self.received,
        }
    }
}

impl Receiver for TlsReceiver {
    fn recv(&mut self) -> Result<Incoming, LinkError> {
        let incoming =
            frame::read(&mut self.inbound, self.left).map_err(|e| self.inbound.shared.failed(e))?;
        self.left = incoming.body;
        Ok(incoming)
    }

    fn read_body(&mut self, into: &mut [u8]) -> Result<(), LinkError> {
        let m = frame::read_body(&mut self.inbound, &mut self.left, into)
            .map_err(|e| self.inbound.shared.failed(e))?;
        self.received.add(m.bytes, m.seconds);
        self.end.count(false, m);
        Ok(())
    }
}

impl ChannelEnd for TlsChannel {
    fn peer(&self) -> &Identity {
        self.sender.peer()
    }

    fn job(&self) -> &str {
        self.sender.job()
    }

    fn rate(&self) -> Rate {
        Rate {
            sent: self.sender.sent,
            received: self.receiver.received,
        }
    }
}

impl Sender for TlsChannel {
    fn send(&mut self, header: &[u8], body: &[u8]) -> Result<(), LinkError> {
        self.sender.send(header, body)
    }

    fn close(self) -> Result<(), LinkError> {
        self.sender.close()
    }
}

impl Receiver for TlsChannel {
    fn recv(&mut self) -> Result<Incoming, LinkError> {
        self.receiver.recv()
    }

    fn read_body(&mut self, into: &mut [u8]) -> Result<(), LinkError> {
        self.receiver.read_body(into)
    }
}

impl Channel for TlsChannel {
    type SendHalf = TlsSender;
    type ReceiveHalf = TlsReceiver;

    fn split(self) -> (TlsSender, TlsReceiver) {
        (self.sender, self.receiver)
    }
}

#[cfg(all(test, feature = "issue"))]
mod tests {
    use std::net::TcpListener;
    use std::thread;

    use super::*;
    use crate::issue::Authority;

    fn credentials(pool: &Authority, member: &str) -> Credentials {
        let issued = pool.issue(member, 1).unwrap();
        Credentials::from_pem(member, pool.certificate(), &issued.certificate, &issued.key).unwrap()
    }

    /// A channel's two ends, the server's and the laptop's, each link given `setting`.
    fn ends(
        setting: impl Fn(TlsLink) -> TlsLink + Clone + Send + 'static,
    ) -> (TlsChannel, TlsChannel) {
        let pool = Authority::new(1).unwrap();
        let server = setting(TlsLink::bind(credentials(&pool, "server"), "127.0.0.1:0").unwrap());
        let laptop = setting(TlsLink::new(credentials(&pool, "laptop")).unwrap());
        let at = server.local_addr().unwrap().to_string();
        let dialling = thread::spawn(move || laptop.open("server", &at, "j").unwrap());
        let accepted = server.accept(&["laptop"]).unwrap();
        (accepted, dialling.join().unwrap())
    }

    /// The keepalive each of a channel's two sockets is watched by.
    fn watched(channel: &TlsChannel) -> [Option<Keepalive>; 2] {
        [&channel.sender.outbound.tcp, &channel.receiver.inbound.tcp].map(|tcp| {
            let sock = socket2::SockRef::from(tcp);
            sock.keepalive().unwrap().then(|| Keepalive {
                idle: sock.tcp_keepalive_time().unwrap(),
                interval: sock.tcp_keepalive_interval().unwrap(),
                probes: sock.tcp_keepalive_retries().unwrap(),
            })
        })
    }

    #[test]
    fn both_ends_of_a_channel_are_watched_by_keepalive_as_set() {
        let (answered, dialled) = ends(|link| link);
        assert_eq!(watched(&answered), [Some(KEEPALIVE); 2]);
        assert_eq!(watched(&dialled), [Some(KEEPALIVE); 2]);
        assert_eq!(KEEPALIVE.limit(), Duration::from_secs(25));

        let other = Keepalive {
            idle: Duration::from_secs(7),
            interval: Duration::from_secs(3),
            probes: 4,
        };
        let (answered, dialled) = ends(move |link| link.with_keepalive(Some(other)));
        assert_eq!(watched(&answered), [Some(other); 2]);
        assert_eq!(watched(&dialled), [Some(other); 2]);

        let (answered, dialled) = ends(|link| link.with_keepalive(None));
        assert_eq!(watched(&answered), [None; 2]);
        assert_eq!(watched(&dialled), [None; 2]);
    }

    #[test]
    fn dialling_tries_each_address_in_turn_while_time_is_left() {
        let listening = TcpListener::bind("127.0.0.1:0").unwrap();
        let at = listening.local_addr().unwrap();
        // A port nothing listens on: refused at once.
        let closed = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let soon = || Instant::now() + Duration::from_secs(5);
        let tcp = connect_any("both", [closed, at], soon()).unwrap();
        assert_eq!(tcp.peer_addr().unwrap(), at);
        // The last refusal is what is reported when none answers, and no address at all is
        // said so.
        let refused = connect_any("closed", [closed], soon()).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::ConnectionRefused);
        let none = connect_any("nowhere", [], soon()).unwrap_err();
        assert_eq!(none.kind(), io::ErrorKind::InvalidInput);
        // With no time left, nothing is tried.
        let late = connect_any("both", [at], Instant::now()).unwrap_err();
        assert_eq!(late.kind(), io::ErrorKind::TimedOut);
    }

    #[test]
    fn addresses_that_drop_packets_share_one_deadline_between_them() {
        // Two addresses of the documentation range, which most networks drop: the second is
        // given what the first left, not a deadline of its own.
        let dropped = ["192.0.2.1:7340", "192.0.2.2:7340"].map(|a| a.parse().unwrap());
        let started = Instant::now();
        let failed = connect_any("two", dropped, started + Duration::from_millis(300)).unwrap_err();
        let took = started.elapsed();
        assert!(took < Duration::from_millis(500), "{took:?}: {failed}");
    }
}
