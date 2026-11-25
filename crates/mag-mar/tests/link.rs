//! Two members over loopback: authenticated by the pool's authority, refused for each reason a
//! link refuses, one channel per job, and a large tensor byte for byte.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::thread;

use mag_by::{Dtype, Stamp, Start, TakeBack, Tensor};
use mag_mar::issue::Authority;
use mag_mar::{Channel, Credentials, Link, LinkError, Refused, TlsLink};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};

fn credentials(authority: &Authority, member: &str) -> Credentials {
    let issued = authority.issue(member, 1).unwrap();
    Credentials::from_pem(
        member,
        authority.certificate(),
        &issued.certificate,
        &issued.key,
    )
    .unwrap()
}

/// A member that answers on loopback.
fn bound(authority: &Authority, member: &str) -> TlsLink {
    TlsLink::bind(credentials(authority, member), "127.0.0.1:0").unwrap()
}

fn address(link: &TlsLink) -> String {
    link.local_addr().unwrap().to_string()
}

/// `server` accepts what `expected` allows while `dial` runs on another thread; both results.
fn meet<T: Send + 'static>(
    server: &TlsLink,
    expected: &[&str],
    dial: impl FnOnce() -> T + Send + 'static,
) -> (Result<mag_mar::TlsChannel, LinkError>, T) {
    let dialling = thread::spawn(dial);
    let accepted = server.accept(expected);
    (accepted, dialling.join().unwrap())
}

#[test]
fn two_members_authenticate_each_other_and_exchange_words() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    let laptop_serial = laptop.credentials().identity().serial;
    let at = address(&server);
    let (accepted, dialled) = meet(&server, &["workstation", "laptop"], move || {
        laptop.open("server", &at, "label").map(|c| (c, laptop))
    });
    let mut on_server = accepted.unwrap();
    let (mut on_laptop, _laptop) = dialled.unwrap();

    // Each end knows the other by the member its certificate names, and that certificate's
    // serial.
    assert_eq!(on_server.peer().member, "laptop");
    assert_eq!(on_server.peer().serial, laptop_serial);
    assert_eq!(on_laptop.peer().member, "server");
    assert_eq!(
        on_laptop.peer().serial,
        server.credentials().identity().serial
    );
    assert_eq!((on_server.job(), on_laptop.job()), ("label", "label"));

    let back = TakeBack::Member {
        member: "laptop".into(),
        deadline_seconds: 2.0,
    };
    on_server.send_word(&back).unwrap();
    assert_eq!(on_laptop.recv_word::<TakeBack>().unwrap(), back);
    let stamp = Stamp {
        by: Some("someone".into()),
        devices: None,
        priority_ceiling: Some(2),
    };
    on_laptop.send_word(&stamp).unwrap();
    assert_eq!(on_server.recv_word::<Stamp>().unwrap(), stamp);

    // A word of another kind than the one asked for is refused as that.
    on_laptop.send_word(&stamp).unwrap();
    assert!(matches!(
        on_server.recv_word::<TakeBack>(),
        Err(LinkError::Word(_))
    ));

    // A word that came with a body, and a tensor whose body is not its length, are refused;
    // the body is still there to be read.
    on_laptop
        .send(&serde_json::to_vec(&stamp).unwrap(), b"xyz")
        .unwrap();
    assert!(matches!(
        on_server.recv_word::<Stamp>(),
        Err(LinkError::Body {
            expected: 0,
            found: 3
        })
    ));
    on_server.read_body(&mut [0; 3]).unwrap();
    let tensor = Tensor {
        from: "b0".into(),
        to: None,
        step: 0,
        dtype: Dtype::U8,
        shape: vec![4],
        bytes: 4,
    };
    on_laptop
        .send(&serde_json::to_vec(&tensor).unwrap(), b"xyz")
        .unwrap();
    assert!(matches!(
        on_server.recv_tensor(),
        Err(LinkError::Body {
            expected: 4,
            found: 3
        })
    ));
    on_server.read_body(&mut [0; 3]).unwrap();

    on_laptop.close().unwrap();
    assert!(matches!(on_server.recv(), Err(LinkError::Closed)));
}

#[test]
fn either_member_may_dial() {
    let pool = Authority::new(1).unwrap();
    let workstation = bound(&pool, "workstation");
    let server = Arc::new(bound(&pool, "server"));
    let at = address(&server);
    let w = TlsLink::new(credentials(&pool, "workstation")).unwrap();
    let (accepted, dialled) = meet(&server, &["workstation"], move || {
        w.open("server", &at, "one")
    });
    assert_eq!(accepted.unwrap().peer().member, "workstation");
    assert_eq!(dialled.unwrap().peer().member, "server");

    let at = address(&workstation);
    let s = server.clone();
    let (accepted, dialled) = meet(&workstation, &["server"], move || {
        s.open("workstation", &at, "two")
    });
    assert_eq!(accepted.unwrap().peer().member, "server");
    assert_eq!(dialled.unwrap().job(), "two");
}

#[test]
fn a_member_of_another_pool_is_refused_at_either_end() {
    let pool = Authority::new(1).unwrap();
    let other = Authority::new(1).unwrap();

    // This pool's member dials one of another pool, and refuses the certificate it answers
    // with. (A member of another pool dialling this one's is refused the same way, by itself:
    // the end that dials checks the other's certificate first.)
    let stranger = bound(&other, "server");
    let at = address(&stranger);
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    let (accepted, dialled) = meet(&stranger, &["laptop"], move || {
        laptop.open("server", &at, "j")
    });
    assert!(
        matches!(dialled, Err(LinkError::Refused(Refused::NotThePool(_)))),
        "{:?}",
        dialled.err()
    );
    assert!(accepted.is_err());

    // One that trusts this pool's authority, but presents another's certificate, is refused by
    // the end that answers.
    let server = bound(&pool, "server");
    let laptop = other.issue("laptop", 1).unwrap();
    let config = raw_client(
        &pool,
        &rustls::version::TLS13,
        Some((
            vec![CertificateDer::from_pem_slice(laptop.certificate.as_bytes()).unwrap()],
            PrivateKeyDer::from_pem_slice(laptop.key.as_bytes()).unwrap(),
        )),
    );
    let at = address(&server);
    let (accepted, dialled) = meet(&server, &["laptop"], move || raw_dial(config, &at));
    assert!(
        matches!(accepted, Err(LinkError::Refused(Refused::NotThePool(_)))),
        "{:?}",
        accepted.err()
    );
    assert!(dialled.is_err());
}

#[test]
fn a_member_that_is_not_the_one_expected_is_refused_at_either_end() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");

    // The answering end expects the workstation; the laptop dials.
    let at = address(&server);
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    let (accepted, dialled) = meet(&server, &["workstation"], move || {
        laptop.open("server", &at, "j")
    });
    match accepted {
        Err(LinkError::Refused(Refused::WrongMember { expected, named })) => {
            assert_eq!(expected, ["workstation"]);
            assert_eq!(named, ["laptop"]);
        }
        other => panic!("{:?}", other.err()),
    }
    assert!(
        matches!(dialled, Err(LinkError::RefusedBy(_))),
        "{:?}",
        dialled.err()
    );

    // The dialling end means the workstation, but the server answers.
    let at = address(&server);
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    let (accepted, dialled) = meet(&server, &["laptop"], move || {
        laptop.open("workstation", &at, "j")
    });
    match dialled {
        Err(LinkError::Refused(Refused::WrongMember { expected, named })) => {
            assert_eq!(expected, ["workstation"]);
            assert_eq!(named, ["server"]);
        }
        other => panic!("{:?}", other.err()),
    }
    assert!(accepted.is_err());
}

/// Credentials for `member` from a certificate the pool's authority signs naming every one of
/// `names`.
fn credentials_naming(pool: &Authority, member: &str, names: &[&str]) -> Credentials {
    use rcgen::{CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair};
    let issuer =
        Issuer::from_ca_cert_pem(pool.certificate(), KeyPair::from_pem(&pool.key()).unwrap())
            .unwrap();
    let mut params =
        CertificateParams::new(names.iter().map(|n| n.to_string()).collect::<Vec<_>>()).unwrap();
    params.is_ca = IsCa::ExplicitNoCa;
    params.extended_key_usages = vec![
        ExtendedKeyUsagePurpose::ServerAuth,
        ExtendedKeyUsagePurpose::ClientAuth,
    ];
    params.use_authority_key_identifier_extension = true;
    let key = KeyPair::generate().unwrap();
    let certificate = params.signed_by(&key, &issuer).unwrap().pem();
    Credentials::from_pem(
        member,
        pool.certificate(),
        &certificate,
        &key.serialize_pem(),
    )
    .unwrap()
}

#[test]
fn a_certificate_naming_two_expected_members_is_refused_and_one_naming_one_is_taken_as_it() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");

    // Signed by the pool's authority, naming both the workstation and the laptop: to an end that
    // expects both, it could stand for either, and is refused.
    let at = address(&server);
    let both = TlsLink::new(credentials_naming(
        &pool,
        "laptop",
        &["workstation", "laptop"],
    ))
    .unwrap();
    let (accepted, dialled) = meet(&server, &["workstation", "laptop"], move || {
        both.open("server", &at, "j")
    });
    match accepted {
        Err(LinkError::Refused(Refused::WrongMember { expected, named })) => {
            assert_eq!(expected, ["workstation", "laptop"]);
            assert_eq!(named, ["workstation", "laptop"]);
        }
        other => panic!("{:?}", other.err()),
    }
    assert!(dialled.is_err());

    // To an end that expects only the laptop, it names exactly one member expected, and is
    // taken as the laptop.
    let at = address(&server);
    let both = TlsLink::new(credentials_naming(
        &pool,
        "laptop",
        &["workstation", "laptop"],
    ))
    .unwrap();
    let (accepted, dialled) = meet(&server, &["laptop"], move || {
        both.open("server", &at, "j").map(|c| (c, both))
    });
    assert_eq!(accepted.unwrap().peer().member, "laptop");
    assert!(dialled.is_ok());
}

#[test]
fn a_peer_that_stalls_cannot_hold_the_opening_past_its_deadline() {
    use std::time::{Duration, Instant};
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server").with_opening(Duration::from_millis(300));
    let at = address(&server);
    // Before any certificate: the header of a handshake record of 16 KiB, then its body a byte
    // every 100 ms, which keeps each read within any per-read timeout, for two seconds.
    let started = Instant::now();
    let (accepted, ()) = meet(&server, &["laptop"], move || {
        let mut sock = TcpStream::connect(&at).unwrap();
        sock.write_all(&[0x16, 0x03, 0x01, 0x40, 0x00]).unwrap();
        for _ in 0..20 {
            if sock.write_all(&[0]).is_err() {
                return;
            }
            thread::sleep(Duration::from_millis(100));
        }
    });
    let took = started.elapsed();
    match accepted {
        Err(LinkError::Io(e)) => assert!(
            matches!(
                e.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            ),
            "{e:?}"
        ),
        other => panic!("{:?}", other.err()),
    }
    // Given up at the deadline, not when the peer stopped: 0.3 s and a read's slack, not 2 s.
    assert!(took < Duration::from_millis(1200), "{took:?}");
}

#[test]
fn a_peer_that_says_nothing_is_given_up_at_the_deadline() {
    use std::time::{Duration, Instant};
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server").with_opening(Duration::from_millis(300));
    let at = address(&server);
    let silent = thread::spawn(move || {
        let _sock = TcpStream::connect(&at).unwrap();
        thread::sleep(Duration::from_secs(3));
    });
    let started = Instant::now();
    let accepted = server.accept(&["laptop"]);
    let took = started.elapsed();
    assert!(
        matches!(accepted, Err(LinkError::Io(_))),
        "{:?}",
        accepted.err()
    );
    // At the deadline, not when the peer leaves three seconds later, nor after the default.
    assert!(took < Duration::from_millis(1200), "{took:?}");
    silent.join().unwrap();
}

#[test]
fn an_open_channel_is_not_bounded_by_the_opening_s_deadline() {
    use std::time::Duration;
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server").with_opening(Duration::from_millis(200));
    let laptop = TlsLink::new(credentials(&pool, "laptop"))
        .unwrap()
        .with_opening(Duration::from_millis(200));
    let at = address(&server);
    let (accepted, dialled) = meet(&server, &["laptop"], move || {
        laptop.open("server", &at, "j").map(|c| (c, laptop))
    });
    let mut on_server = accepted.unwrap();
    let (mut on_laptop, _laptop) = dialled.unwrap();
    // Well past both deadlines, a word still goes each way.
    thread::sleep(Duration::from_millis(500));
    let back = TakeBack::Member {
        member: "laptop".into(),
        deadline_seconds: 2.0,
    };
    on_server.send_word(&back).unwrap();
    assert_eq!(on_laptop.recv_word::<TakeBack>().unwrap(), back);
    on_laptop.send_word(&back).unwrap();
    assert_eq!(on_server.recv_word::<TakeBack>().unwrap(), back);
}

#[test]
fn a_certificate_signed_by_a_member_is_refused() {
    // A member's certificate cannot sign another: the laptop's key signs one naming the
    // workstation, presented with the laptop's certificate as the step between.
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let laptop = pool.issue("laptop", 1).unwrap();
    let key = rcgen::KeyPair::from_pem(&laptop.key).unwrap();
    let as_issuer = rcgen::Issuer::from_ca_cert_pem(&laptop.certificate, key).unwrap();
    let mut params = rcgen::CertificateParams::new(vec!["workstation".to_string()]).unwrap();
    params.extended_key_usages = vec![
        rcgen::ExtendedKeyUsagePurpose::ServerAuth,
        rcgen::ExtendedKeyUsagePurpose::ClientAuth,
    ];
    let forged_key = rcgen::KeyPair::generate().unwrap();
    let forged = params.signed_by(&forged_key, &as_issuer).unwrap();

    // Not as credentials ...
    assert!(matches!(
        Credentials::from_pem(
            "workstation",
            pool.certificate(),
            &forged.pem(),
            &forged_key.serialize_pem()
        ),
        Err(LinkError::Credentials(_))
    ));
    // ... and not on a link, with the laptop's certificate presented beside it.
    let config = raw_client(
        &pool,
        &rustls::version::TLS13,
        Some((
            vec![
                forged.der().clone(),
                CertificateDer::from_pem_slice(laptop.certificate.as_bytes()).unwrap(),
            ],
            PrivateKeyDer::from_pem_slice(forged_key.serialize_pem().as_bytes()).unwrap(),
        )),
    );
    let at = address(&server);
    let (accepted, _) = meet(&server, &["workstation"], move || raw_dial(config, &at));
    assert!(
        matches!(accepted, Err(LinkError::Refused(Refused::NotThePool(_)))),
        "{:?}",
        accepted.err()
    );
}

/// A TLS client outside the link, to present what a member would not.
fn raw_client(
    pool: &Authority,
    version: &'static rustls::SupportedProtocolVersion,
    auth: Option<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)>,
) -> rustls::ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(CertificateDer::from_pem_slice(pool.certificate().as_bytes()).unwrap())
        .unwrap();
    let builder = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_protocol_versions(&[version])
    .unwrap()
    .with_root_certificates(roots);
    match auth {
        Some((chain, key)) => builder.with_client_auth_cert(chain, key).unwrap(),
        None => builder.with_no_client_auth(),
    }
}

/// Dials `at` as the server with `config`, says something, and waits for the answer.
fn raw_dial(config: rustls::ClientConfig, at: &str) -> std::io::Result<()> {
    let conn =
        rustls::ClientConnection::new(Arc::new(config), ServerName::try_from("server").unwrap())
            .unwrap();
    let mut stream = rustls::StreamOwned::new(conn, TcpStream::connect(at)?);
    stream.write_all(&[0; 12])?;
    let mut answer = [0; 1];
    stream.read_exact(&mut answer)
}

#[test]
fn a_dialler_with_no_certificate_is_refused() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let config = raw_client(&pool, &rustls::version::TLS13, None);
    let at = address(&server);
    let (accepted, dialled) = meet(&server, &["laptop"], move || raw_dial(config, &at));
    assert!(
        matches!(accepted, Err(LinkError::Refused(Refused::NoCertificate))),
        "{:?}",
        accepted.err()
    );
    assert!(dialled.is_err());
}

#[test]
fn a_dialler_below_tls_1_3_is_refused() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let laptop = pool.issue("laptop", 1).unwrap();
    let config = raw_client(
        &pool,
        &rustls::version::TLS12,
        Some((
            vec![CertificateDer::from_pem_slice(laptop.certificate.as_bytes()).unwrap()],
            PrivateKeyDer::from_pem_slice(laptop.key.as_bytes()).unwrap(),
        )),
    );
    let at = address(&server);
    let (accepted, dialled) = meet(&server, &["laptop"], move || raw_dial(config, &at));
    assert!(
        matches!(accepted, Err(LinkError::Refused(Refused::Protocol(_)))),
        "{:?}",
        accepted.err()
    );
    assert!(dialled.is_err());
}

#[test]
fn a_name_a_certificate_cannot_carry_exactly_is_refused_before_dialling() {
    let pool = Authority::new(1).unwrap();
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    for name in ["Server", "server_1", "192.0.2.1", ""] {
        assert!(
            matches!(laptop.open(name, "127.0.0.1:1", "j"), Err(LinkError::Name(n)) if n == name),
            "{name}"
        );
        assert!(
            matches!(pool.issue(name, 1), Err(LinkError::Name(_))),
            "{name}"
        );
    }
}

#[test]
fn each_job_has_its_own_channel_and_none_hears_another() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let laptop = Arc::new(TlsLink::new(credentials(&pool, "laptop")).unwrap());
    let at = address(&server);
    let l = laptop.clone();
    let dialling = thread::spawn(move || {
        let a = l.open("server", &at, "job-a").unwrap();
        let b = l.open("server", &at, "job-b").unwrap();
        (a, b)
    });
    let first = server.accept(&["laptop"]).unwrap();
    let second = server.accept(&["laptop"]).unwrap();
    let (mut la, mut lb) = dialling.join().unwrap();
    let (mut sa, mut sb) = if first.job() == "job-a" {
        (first, second)
    } else {
        (second, first)
    };
    assert_eq!((sa.job(), sb.job()), ("job-a", "job-b"));

    // Interleaved, each word arrives on its own job's channel only, in order.
    for step in 0..3u64 {
        let word = |job: &str| Tensor {
            from: job.into(),
            to: None,
            step,
            dtype: Dtype::U8,
            shape: vec![1],
            bytes: 1,
        };
        lb.send_tensor(&word("b"), b"b").unwrap();
        la.send_tensor(&word("a"), b"a").unwrap();
        for (ch, job) in [(&mut sa, "a"), (&mut sb, "b")] {
            let t = ch.recv_tensor().unwrap();
            assert_eq!((t.from.as_str(), t.step), (job, step));
            let mut byte = [0];
            ch.read_body(&mut byte).unwrap();
            assert_eq!(byte, [job.as_bytes()[0]]);
        }
    }

    // One job's channel closing leaves the other's open.
    la.close().unwrap();
    assert!(matches!(sa.recv(), Err(LinkError::Closed)));
    sb.send(b"{}", b"").unwrap();
    assert_eq!(lb.recv().unwrap().header, b"{}");
    // And one dropped without closing is found cut short, not closed.
    drop(lb);
    assert!(matches!(sb.recv(), Err(LinkError::Truncated)));
}

#[test]
fn a_large_tensor_arrives_byte_for_byte_and_its_rate_is_measured() {
    const BYTES: usize = 64 << 20;
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    let at = address(&server);
    let sent: Vec<u8> = (0..BYTES).map(|i| (i * 31 + i / 4093) as u8).collect();
    let tensor = Tensor {
        from: "b0".into(),
        to: Some("b1".into()),
        step: 0,
        dtype: Dtype::F32,
        shape: vec![(BYTES / 4) as u64],
        bytes: BYTES as u64,
    };
    let (word, data) = (tensor.clone(), sent.clone());
    let sending = thread::spawn(move || {
        let mut ch = laptop.open("server", &at, "label").unwrap();
        ch.send_tensor(&word, &data).unwrap();
        // A body that is not the tensor's length is not sent.
        assert!(matches!(
            ch.send_tensor(&word, &data[1..]),
            Err(LinkError::Body { expected, found }) if expected == BYTES as u64
                && found == BYTES as u64 - 1
        ));
        let rate = ch.rate();
        ch.close().unwrap();
        (rate, laptop.rate("server"))
    });
    let mut ch = server.accept(&["laptop"]).unwrap();
    assert_eq!(ch.recv_tensor().unwrap(), tensor);
    let mut got = vec![0; BYTES];
    ch.read_body(&mut got).unwrap();
    assert!(got == sent, "the tensor's bytes differ");
    assert!(matches!(ch.recv(), Err(LinkError::Closed)));

    let (sent_rate, link_rate) = sending.join().unwrap();
    assert_eq!(sent_rate.sent.bytes, BYTES as u64);
    assert_eq!(link_rate, Some(sent_rate));
    assert!(sent_rate.sent.bytes_per_second().is_some());
    let received = ch.rate().received;
    assert_eq!(received.bytes, BYTES as u64);
    assert!(received.bytes_per_second().unwrap() > 0.0);
    assert_eq!(server.rate("laptop"), Some(ch.rate()));
    assert_eq!(server.rate("workstation"), None);
}

#[test]
fn a_certificate_issued_again_has_a_new_serial_and_the_link_reports_it() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let mut serials = Vec::new();
    for _ in 0..2 {
        let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
        let issued = laptop.credentials().identity().serial;
        let at = address(&server);
        let (accepted, _) = meet(&server, &["laptop"], move || {
            laptop.open("server", &at, "j")
        });
        let seen = accepted.unwrap().peer().clone();
        assert_eq!(seen.serial, issued);
        assert_eq!(seen.serial_hex().len(), 32);
        serials.push(seen.serial);
    }
    assert_ne!(serials[0], serials[1]);
}

#[test]
fn credentials_come_from_a_start_file_and_must_be_the_authoritys_and_name_the_member() {
    let pool = Authority::new(1).unwrap();
    let issued = pool.issue("server", 1).unwrap();
    let dir = std::env::temp_dir().join(format!("mag-mar-start-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (cert, key) = (dir.join("server.pem"), dir.join("server.key"));
    std::fs::write(&cert, &issued.certificate).unwrap();
    std::fs::write(&key, &issued.key).unwrap();

    let mut start = Start::from_json(include_str!("../../mag-by/starts/example.json")).unwrap();
    // The example's authority is another pool's: this member's certificate is not its.
    start.certificate_file = cert.to_str().unwrap().into();
    start.key_file = key.to_str().unwrap().into();
    assert!(matches!(
        Credentials::from_start(&start),
        Err(LinkError::Credentials(_))
    ));
    start.authority = pool.certificate().into();
    let c = Credentials::from_start(&start).unwrap();
    assert_eq!(c.identity().member, "server");
    assert_eq!(c.identity().serial, issued.serial);

    // A certificate for another member.
    start.member = "laptop".into();
    assert!(matches!(
        Credentials::from_start(&start),
        Err(LinkError::Credentials(_))
    ));
    start.key_file = dir.join("missing.key").to_str().unwrap().into();
    assert!(matches!(
        Credentials::from_start(&start),
        Err(LinkError::Credentials(_))
    ));
    std::fs::remove_dir_all(&dir).unwrap();
}
