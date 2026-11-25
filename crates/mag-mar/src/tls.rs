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

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

use crate::{Channel, Identity, Incoming, Link, LinkError, Rate, Refused, check_name, frame};

/// How long the handshake and the opening of a channel may take before the channel is given up.
const OPENING: Duration = Duration::from_secs(10);

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
    /// certificate and key files it names.
    pub fn from_start(start: &mag_by::Start) -> Result<Credentials, LinkError> {
        let read = |path: &str| {
            std::fs::read_to_string(path)
                .map_err(|e| LinkError::Credentials(format!("reading {path}: {e}")))
        };
        Credentials::from_pem(
            &start.member,
            &start.authority,
            &read(&start.certificate_file)?,
            &read(&start.key_file)?,
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

/// The stream under a channel, as the end that dialled or the end that answered holds it.
enum Stream {
    Dialled(StreamOwned<ClientConnection, TcpStream>),
    Answered(StreamOwned<ServerConnection, TcpStream>),
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

    fn sock(&self) -> &TcpStream {
        match self {
            Stream::Dialled(s) => &s.sock,
            Stream::Answered(s) => &s.sock,
        }
    }

    /// Tells the other end the channel is closed, and closes this end's writing half.
    fn close(&mut self) -> Result<(), LinkError> {
        match self {
            Stream::Dialled(s) => s.conn.send_close_notify(),
            Stream::Answered(s) => s.conn.send_close_notify(),
        }
        self.flush()?;
        self.sock().shutdown(Shutdown::Write)?;
        Ok(())
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

fn prepare(sock: &TcpStream) -> Result<(), LinkError> {
    sock.set_nodelay(true)?;
    sock.set_read_timeout(Some(OPENING))?;
    sock.set_write_timeout(Some(OPENING))?;
    Ok(())
}

fn opened(sock: &TcpStream) -> Result<(), LinkError> {
    sock.set_read_timeout(None)?;
    sock.set_write_timeout(None)?;
    Ok(())
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
pub struct TlsLink {
    credentials: Credentials,
    client: Arc<ClientConfig>,
    server: Arc<ServerConfig>,
    listener: Option<TcpListener>,
    rates: Arc<Mutex<BTreeMap<String, Rate>>>,
}

impl TlsLink {
    /// An end that only dials: a member behind address translation, which no other can reach.
    pub fn new(credentials: Credentials) -> Result<TlsLink, LinkError> {
        Ok(TlsLink {
            client: Arc::new(client_config(&credentials)?),
            server: Arc::new(server_config(&credentials)?),
            credentials,
            listener: None,
            rates: Arc::default(),
        })
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

    fn channel(&self, stream: Stream, peer: Identity, job: String) -> TlsChannel {
        TlsChannel {
            stream,
            peer,
            job,
            left: 0,
            rate: Rate::default(),
            rates: self.rates.clone(),
        }
    }
}

impl Link for TlsLink {
    type Channel = TlsChannel;

    fn open(&self, member: &str, address: &str, job: &str) -> Result<TlsChannel, LinkError> {
        check_name(member)?;
        let name =
            ServerName::try_from(member.to_string()).map_err(|_| LinkError::Name(member.into()))?;
        let sock = TcpStream::connect(address)?;
        prepare(&sock)?;
        let conn = ClientConnection::new(self.client.clone(), name).map_err(LinkError::Tls)?;
        let mut stream = Stream::Dialled(StreamOwned::new(conn, sock));
        // The other end's certificate is checked here, its name with it. This end's is checked
        // by the other end after this end's part of the handshake ends, so its answer to the
        // opening is what says it was taken.
        stream.handshake()?;
        let peer = Identity {
            member: member.into(),
            serial: serial(&stream)?,
        };
        frame::open_as_dialler(&mut stream, job, mag_by::WIRE_VERSION)?;
        opened(stream.sock())?;
        Ok(self.channel(stream, peer, job.into()))
    }

    fn accept(&self, expected: &[&str]) -> Result<TlsChannel, LinkError> {
        for m in expected {
            check_name(m)?;
        }
        let listener = self.listener.as_ref().ok_or_else(|| {
            LinkError::Io(io::Error::new(
                io::ErrorKind::NotConnected,
                "this end was bound to no address, and only dials",
            ))
        })?;
        let (sock, _) = listener.accept()?;
        prepare(&sock)?;
        let conn = ServerConnection::new(self.server.clone()).map_err(LinkError::Tls)?;
        let mut stream = Stream::Answered(StreamOwned::new(conn, sock));
        stream.handshake()?;
        let cert = stream
            .peer_certificate()
            .ok_or(LinkError::Refused(Refused::NoCertificate))?;
        let member = match member_named(cert, expected) {
            Ok(member) => member,
            Err(refused) => {
                let _ = stream.close();
                return Err(LinkError::Refused(refused));
            }
        };
        let peer = Identity {
            member,
            serial: serial(&stream)?,
        };
        let job = frame::open_as_acceptor(&mut stream, mag_by::WIRE_VERSION)?;
        opened(stream.sock())?;
        Ok(self.channel(stream, peer, job))
    }

    fn rate(&self, member: &str) -> Option<Rate> {
        self.rates.lock().ok()?.get(member).copied()
    }
}

/// One job's channel to another member, over its own TCP connection and TLS session.
///
/// Dropped without [`Channel::close`], it ends without telling the other end, which then
/// finds it [`LinkError::Truncated`].
pub struct TlsChannel {
    stream: Stream,
    peer: Identity,
    job: String,
    /// Bytes of the last body received not yet read.
    left: u64,
    rate: Rate,
    rates: Arc<Mutex<BTreeMap<String, Rate>>>,
}

impl TlsChannel {
    fn count(&mut self, sent: bool, m: crate::Measure) {
        let mut all = self.rates.lock().unwrap_or_else(|p| p.into_inner());
        let link = all.entry(self.peer.member.clone()).or_default();
        for rate in [&mut self.rate, link] {
            let measure = if sent {
                &mut rate.sent
            } else {
                &mut rate.received
            };
            measure.add(m.bytes, m.seconds);
        }
    }
}

impl Channel for TlsChannel {
    fn peer(&self) -> &Identity {
        &self.peer
    }

    fn job(&self) -> &str {
        &self.job
    }

    fn send(&mut self, header: &[u8], body: &[u8]) -> Result<(), LinkError> {
        let m = frame::write(&mut self.stream, header, body)?;
        self.count(true, m);
        Ok(())
    }

    fn recv(&mut self) -> Result<Incoming, LinkError> {
        let incoming = frame::read(&mut self.stream, self.left)?;
        self.left = incoming.body;
        Ok(incoming)
    }

    fn read_body(&mut self, into: &mut [u8]) -> Result<(), LinkError> {
        let m = frame::read_body(&mut self.stream, &mut self.left, into)?;
        self.count(false, m);
        Ok(())
    }

    fn rate(&self) -> Rate {
        self.rate
    }

    fn close(mut self) -> Result<(), LinkError> {
        self.stream.close()
    }
}
