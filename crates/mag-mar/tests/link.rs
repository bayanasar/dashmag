//! Two members over loopback: authenticated by the pool's authority, refused for each reason a
//! link refuses, one channel per job, and a large tensor byte for byte.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use mag_by::{Dtype, Stamp, Start, TakeBack, Tensor};
use mag_mar::issue::Authority;
use mag_mar::{
    Channel, ChannelEnd, Credentials, Link, LinkError, Receiver, Refused, Sender, Side, TlsLink,
};
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

/// Runs `f` on a thread of its own, and fails if it has not finished within `limit`: a test of a
/// link that would wait for ever fails, instead of hanging.
fn within<T: Send + 'static>(limit: Duration, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (done, finished) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let _ = done.send(f());
    });
    match finished.recv_timeout(limit) {
        Ok(t) => t,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!("still waiting after {limit:?}"),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => panic!("the test's thread failed"),
    }
}

/// A channel between `server` and `laptop` for job `j`: the server's end and the laptop's.
fn channel(server: &TlsLink, laptop: TlsLink) -> (mag_mar::TlsChannel, mag_mar::TlsChannel) {
    let at = address(server);
    let (accepted, dialled) = meet(server, &["laptop"], move || laptop.open("server", &at, "j"));
    (accepted.unwrap(), dialled.unwrap())
}

/// `bytes` bytes, a pattern of its own for each `seed`.
fn pattern(bytes: usize, seed: usize) -> Vec<u8> {
    (0..bytes)
        .map(|i| (i * (31 + 2 * seed) + i / 4093 + seed) as u8)
        .collect()
}

/// Writes a key file readable by its owner alone, as `certs` does.
fn write_key(path: &std::path::Path, key: &str) {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options
        .open(path)
        .unwrap()
        .write_all(key.as_bytes())
        .unwrap();
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
fn accept_any_takes_any_member_of_the_pool_and_reports_it_but_not_one_naming_two() {
    let pool = Authority::new(1).unwrap();
    let coordinator = bound(&pool, "server");

    // A member nothing names in advance, with a certificate of the pool's: taken, and reported
    // as the member its certificate names, with that certificate's serial.
    let at = address(&coordinator);
    let rented = TlsLink::new(credentials(&pool, "rented")).unwrap();
    let serial = rented.credentials().identity().serial;
    let dialling = thread::spawn(move || {
        rented
            .open("server", &at, mag_by::CONTROL)
            .map(|c| (c, rented))
    });
    let accepted = coordinator.accept_any().unwrap();
    assert_eq!(accepted.peer().member, "rented");
    assert_eq!(accepted.peer().serial, serial);
    assert_eq!(accepted.job(), mag_by::CONTROL);
    assert!(dialling.join().unwrap().is_ok());

    // A certificate of the pool's naming two members could stand for either: refused.
    let at = address(&coordinator);
    let both = TlsLink::new(credentials_naming(&pool, "laptop", &["laptop", "rented"])).unwrap();
    let dialling = thread::spawn(move || both.open("server", &at, mag_by::CONTROL));
    match coordinator.accept_any() {
        Err(LinkError::Refused(Refused::NotOneMember { named })) => {
            assert_eq!(named, ["laptop", "rented"]);
        }
        other => panic!("{:?}", other.err()),
    }
    assert!(dialling.join().unwrap().is_err());

    // A member of another pool: refused as not the pool's.
    let other = Authority::new(1).unwrap();
    let laptop = other.issue("laptop", 1).unwrap();
    let config = raw_client(
        &pool,
        &rustls::version::TLS13,
        Some((
            vec![CertificateDer::from_pem_slice(laptop.certificate.as_bytes()).unwrap()],
            PrivateKeyDer::from_pem_slice(laptop.key.as_bytes()).unwrap(),
        )),
    );
    let at = address(&coordinator);
    let dialling = thread::spawn(move || raw_dial(config, &at));
    let accepted = coordinator.accept_any();
    assert!(
        matches!(accepted, Err(LinkError::Refused(Refused::NotThePool(_)))),
        "{:?}",
        accepted.err()
    );
    assert!(dialling.join().unwrap().is_err());
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
        Err(LinkError::OpeningTimedOut { after, side }) => {
            assert_eq!((after, side), (Duration::from_millis(300), Side::Answering));
        }
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
    let err = accepted.err().unwrap();
    assert!(
        matches!(
            err,
            LinkError::OpeningTimedOut {
                side: Side::Answering,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(
        err.to_string(),
        "the channel did not open within 0.3 s, while answering"
    );
    // At the deadline, not when the peer leaves three seconds later, nor after the default.
    assert!(took < Duration::from_millis(1200), "{took:?}");
    silent.join().unwrap();
}

#[test]
fn a_dialler_whose_peer_says_nothing_is_given_up_at_the_deadline() {
    use std::time::{Duration, Instant};
    let pool = Authority::new(1).unwrap();
    let laptop = TlsLink::new(credentials(&pool, "laptop"))
        .unwrap()
        .with_opening(Duration::from_millis(300));
    // A peer that takes the connection and says nothing: the handshake waits for its answer.
    let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let at = silent.local_addr().unwrap().to_string();
    let holding = thread::spawn(move || {
        let (_sock, _) = silent.accept().unwrap();
        thread::sleep(Duration::from_secs(3));
    });
    let started = Instant::now();
    let err = laptop.open("server", &at, "j").err().unwrap();
    let took = started.elapsed();
    match &err {
        LinkError::OpeningTimedOut { after, side } => {
            assert_eq!(
                (*after, *side),
                (Duration::from_millis(300), Side::Dialling)
            );
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        err.to_string(),
        "the channel did not open within 0.3 s, while dialling"
    );
    assert!(took >= Duration::from_millis(250), "{took:?}");
    assert!(took < Duration::from_millis(1200), "{took:?}");
    holding.join().unwrap();
}

#[test]
fn peers_that_connect_and_say_nothing_do_not_delay_a_member_s_opening() {
    let pool = Authority::new(1).unwrap();
    // The default deadline, 10 s: taken one at a time, five silent peers would hold the next
    // opening for 50 s.
    let server = Arc::new(bound(&pool, "server"));
    let at = address(&server);
    let _silent: Vec<TcpStream> = (0..5).map(|_| TcpStream::connect(&at).unwrap()).collect();

    // `accept_any`, as the coordinator takes its members, and `accept`, each behind the five.
    for any in [true, false] {
        let s = server.clone();
        let accepting = thread::spawn(move || {
            if any {
                s.accept_any()
            } else {
                s.accept(&["laptop"])
            }
        });
        let laptop = TlsLink::new(credentials(&pool, "laptop"))
            .unwrap()
            .with_opening(Duration::from_secs(2));
        let started = Instant::now();
        let dialled = laptop.open("server", &at, mag_by::CONTROL);
        let took = started.elapsed();
        assert_eq!(dialled.unwrap().peer().member, "server");
        assert!(took < Duration::from_secs(1), "{took:?}");
        assert_eq!(accepting.join().unwrap().unwrap().peer().member, "laptop");
    }
}

#[test]
fn past_its_limit_an_answering_end_closes_new_connections_at_once() {
    let pool = Authority::new(1).unwrap();
    let server = Arc::new(
        bound(&pool, "server")
            .with_opening(Duration::from_secs(30))
            .with_answering(2),
    );
    let at = address(&server);
    let (taken_tx, taken) = std::sync::mpsc::channel();
    let s = server.clone();
    thread::spawn(
        move || {
            while taken_tx.send(s.accept(&["laptop"])).is_ok() {}
        },
    );
    // Two peers that say nothing are held while they open, up to the limit.
    let silent: Vec<TcpStream> = (0..2).map(|_| TcpStream::connect(&at).unwrap()).collect();
    // A third is closed at once, not held until a deadline 30 s away.
    let mut third = TcpStream::connect(&at).unwrap();
    third
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let started = Instant::now();
    match third.read(&mut [0; 1]) {
        Ok(0) => {}
        Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => {}
        other => panic!("{other:?}"),
    }
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "{:?}",
        started.elapsed()
    );
    // The two are still held.
    for mut held in &silent {
        held.set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let e = held.read(&mut [0; 1]).unwrap_err();
        assert!(
            matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ),
            "{e:?}"
        );
    }
    // One goes: its failure is returned, and its place is free for a member.
    let mut silent = silent;
    drop(silent.remove(0));
    match taken.recv_timeout(Duration::from_secs(5)).unwrap() {
        Err(LinkError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof),
        other => panic!("{:?}", other.err()),
    }
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    assert_eq!(
        laptop.open("server", &at, "j").unwrap().peer().member,
        "server"
    );
    let accepted = taken.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(accepted.unwrap().peer().member, "laptop");
}

#[test]
fn an_opening_no_caller_takes_by_its_deadline_is_given_up_and_a_refusal_comes_first() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server").with_opening(Duration::from_millis(300));
    let at = address(&server);
    // The first accept starts the thread that takes connections.
    let a = at.clone();
    let (accepted, ()) = meet(&server, &["laptop"], move || {
        drop(TcpStream::connect(&a).unwrap());
    });
    assert!(accepted.is_err());

    // A member's opening goes as far as it can with no caller, then waits past its deadline.
    let laptop = TlsLink::new(credentials(&pool, "laptop"))
        .unwrap()
        .with_opening(Duration::from_secs(5));
    let a = at.clone();
    let dialling = thread::spawn(move || laptop.open("server", &a, "j"));
    thread::sleep(Duration::from_millis(600));
    match server.accept(&["laptop"]) {
        Err(LinkError::OpeningTimedOut { after, side }) => {
            assert_eq!((after, side), (Duration::from_millis(300), Side::Answering));
        }
        other => panic!("{:?}", other.err()),
    }
    assert!(dialling.join().unwrap().is_err());

    // A member not expected, whose opening is not one: refused as that member, as before.
    let laptop = pool.issue("laptop", 1).unwrap();
    let config = raw_client(
        &pool,
        &rustls::version::TLS13,
        Some((
            vec![CertificateDer::from_pem_slice(laptop.certificate.as_bytes()).unwrap()],
            PrivateKeyDer::from_pem_slice(laptop.key.as_bytes()).unwrap(),
        )),
    );
    let (accepted, dialled) = meet(&server, &["workstation"], move || raw_dial(config, &at));
    match accepted {
        Err(LinkError::Refused(Refused::WrongMember { expected, named })) => {
            assert_eq!(
                (expected, named),
                (vec!["workstation".into()], vec!["laptop".into()])
            );
        }
        other => panic!("{:?}", other.err()),
    }
    assert!(dialled.is_err());
}

#[test]
fn an_answering_end_dropped_frees_its_address_and_takes_new_settings_after_an_accept() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let at = address(&server);
    let (accepted, ()) = meet(&server, &["laptop"], move || {
        drop(TcpStream::connect(&at).unwrap());
    });
    assert!(accepted.is_err());

    // The opening's deadline set after an accept is the one the next connection is given.
    let server = server.with_opening(Duration::from_millis(300));
    let at = address(&server);
    let silent = TcpStream::connect(&at).unwrap();
    let started = Instant::now();
    match server.accept(&["laptop"]) {
        Err(LinkError::OpeningTimedOut { after, side }) => {
            assert_eq!((after, side), (Duration::from_millis(300), Side::Answering));
        }
        other => panic!("{:?}", other.err()),
    }
    assert!(
        started.elapsed() < Duration::from_millis(1200),
        "{:?}",
        started.elapsed()
    );
    drop(silent);

    // Dropped, the end holds its address no longer, though a thread of its took connections.
    let at = server.local_addr().unwrap();
    drop(server);
    std::net::TcpListener::bind(at).unwrap();
}

#[test]
fn an_opening_dropped_or_refused_is_not_said_to_have_timed_out() {
    use std::io::ErrorKind;
    use std::time::Duration;
    let pool = Authority::new(1).unwrap();
    let laptop = TlsLink::new(credentials(&pool, "laptop"))
        .unwrap()
        .with_opening(Duration::from_secs(5));
    // Nothing listens: the dial is refused at once.
    let closed = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .to_string();
    match laptop.open("server", &closed, "j") {
        Err(LinkError::Io(e)) => assert_eq!(e.kind(), ErrorKind::ConnectionRefused),
        other => panic!("{:?}", other.err()),
    }
    // The answering end takes the connection and drops it before the handshake: the stream
    // ends, or is reset when the dialler's first words came before the drop and were unread.
    let dropping = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let at = dropping.local_addr().unwrap().to_string();
    let dropped = thread::spawn(move || drop(dropping.accept().unwrap()));
    match laptop.open("server", &at, "j") {
        Err(LinkError::Io(e)) => assert!(
            matches!(
                e.kind(),
                ErrorKind::UnexpectedEof | ErrorKind::ConnectionReset
            ),
            "{e:?}"
        ),
        other => panic!("{:?}", other.err()),
    }
    dropped.join().unwrap();
    // The dialling end connects and drops the connection before the handshake.
    let server = bound(&pool, "server").with_opening(Duration::from_secs(5));
    let at = address(&server);
    let (accepted, ()) = meet(&server, &["laptop"], move || {
        drop(TcpStream::connect(&at).unwrap());
    });
    match accepted {
        Err(LinkError::Io(e)) => assert_eq!(e.kind(), ErrorKind::UnexpectedEof),
        other => panic!("{:?}", other.err()),
    }
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

/// Every name `mag-by`'s rule takes for a member, rustls reads as a DNS name, and a certificate
/// carries it and gives it back as itself; and the names a certificate would carry other than
/// as written, or as an address, it refuses. (rustls also reads `_` in a name; the rule, as DNS
/// host names are, is stricter.)
#[test]
fn the_member_name_rule_is_what_a_certificate_carries() {
    let pool = Authority::new(1).unwrap();
    let long = "a".repeat(63);
    let longer = "a".repeat(64);
    let taken = [
        "workstation",
        "gpu-1",
        "a",
        "7a",
        "x.7.y",
        "box.lab",
        "x9.y-8.z",
        &*long,
    ];
    for name in taken {
        assert!(mag_by::is_member_name(name), "{name:?}");
        assert!(
            matches!(ServerName::try_from(name), Ok(ServerName::DnsName(_))),
            "{name:?}: rustls does not read it as a DNS name"
        );
        assert_eq!(pool.issue(name, 1).unwrap().member, name);
    }
    let refused = [
        "",
        "Workstation",
        "-gpu",
        "gpu-",
        "box..lab",
        "box.",
        ".box",
        "7",
        "42",
        "box.42",
        "1.2.3",
        "1.2.3.4",
        "01.2.3.4",
        "::1",
        "café",
        "gpu_1",
        &*longer,
    ];
    for name in refused {
        assert!(!mag_by::is_member_name(name), "{name:?}");
        assert!(
            matches!(pool.issue(name, 1), Err(LinkError::Name(_))),
            "{name:?}"
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
    write_key(&key, &issued.key);

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

#[test]
fn both_ends_send_64_mib_at_once_on_one_channel_and_neither_waits_on_the_other() {
    const BYTES: usize = 64 << 20;
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    let (on_server, on_laptop) = channel(&server, laptop);
    let ok = within(Duration::from_secs(120), move || {
        // Each end splits its channel: the sending half on a thread of its own, the receiving
        // half here. Both start sending at the same moment, each a body far larger than the
        // sockets' buffers, before either reads.
        let start = Arc::new(Barrier::new(2));
        let ends = [(on_server, 0), (on_laptop, 1)].map(|(channel, seed)| {
            let start = start.clone();
            thread::spawn(move || {
                let (mut sending, mut receiving) = channel.split();
                let sender = thread::spawn(move || {
                    let body = pattern(BYTES, seed);
                    start.wait();
                    sending.send(b"{}", &body).unwrap();
                    sending
                });
                let incoming = receiving.recv().unwrap();
                assert_eq!(incoming.body, BYTES as u64);
                let mut got = vec![0; BYTES];
                receiving.read_body(&mut got).unwrap();
                let sending = sender.join().unwrap();
                (got, sending.rate(), receiving.rate())
            })
        });
        let [server_end, laptop_end] = ends.map(|end| end.join().unwrap());
        // Each received the other's body byte for byte, and each half measured its own way.
        assert!(
            server_end.0 == pattern(BYTES, 1),
            "the laptop's body differs"
        );
        assert!(
            laptop_end.0 == pattern(BYTES, 0),
            "the server's body differs"
        );
        for (_, sent, received) in [&server_end, &laptop_end] {
            assert_eq!((sent.sent.bytes, sent.received.bytes), (BYTES as u64, 0));
            assert_eq!(
                (received.sent.bytes, received.received.bytes),
                (0, BYTES as u64)
            );
        }
        true
    });
    assert!(ok);
    let both = server.rate("laptop").unwrap();
    assert_eq!(
        (both.sent.bytes, both.received.bytes),
        (BYTES as u64, BYTES as u64)
    );
}

#[test]
fn a_whole_channel_still_takes_turns_and_its_halves_close_as_it_did() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    let (mut on_server, on_laptop) = channel(&server, laptop);
    let (mut sending, mut receiving) = on_laptop.split();
    assert_eq!(
        (sending.peer().member.as_str(), receiving.job()),
        ("server", "j")
    );
    sending.send(b"[1]", b"").unwrap();
    assert_eq!(on_server.recv().unwrap().header, b"[1]");
    on_server.send(b"[2]", b"ab").unwrap();
    let incoming = receiving.recv().unwrap();
    let mut body = [0; 2];
    receiving.read_body(&mut body).unwrap();
    assert_eq!((incoming.header.as_slice(), &body), (&b"[2]"[..], b"ab"));

    // Closing the sending half closes the laptop's way; the server's way stays open until the
    // server closes it.
    sending.close().unwrap();
    assert!(matches!(on_server.recv(), Err(LinkError::Closed)));
    on_server.send(b"[3]", b"").unwrap();
    assert_eq!(receiving.recv().unwrap().header, b"[3]");
    on_server.close().unwrap();
    assert!(matches!(receiving.recv(), Err(LinkError::Closed)));
}

#[test]
fn a_sending_half_dropped_unclosed_is_found_cut_short_while_its_receiving_half_lives() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    let (mut on_server, on_laptop) = channel(&server, laptop);
    let (sending, mut receiving) = on_laptop.split();
    drop(sending);
    let found = within(Duration::from_secs(10), move || {
        let found = on_server.recv();
        on_server.send(b"[1]", b"").map(|()| found)
    });
    assert!(matches!(found, Ok(Err(LinkError::Truncated))), "{found:?}");
    assert_eq!(receiving.recv().unwrap().header, b"[1]");
}

#[test]
fn a_peer_that_goes_silent_is_given_up_and_a_send_waiting_on_it_ends_too() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server").with_silence(Some(Duration::from_millis(300)));
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    // The laptop holds its end open and does nothing with it: neither reads nor sends.
    let (on_server, _stopped) = channel(&server, laptop);
    let (mut sending, mut receiving) = on_server.split();
    let (received, sent, took) = within(Duration::from_secs(20), move || {
        // A body far larger than the sockets' buffers, which the laptop never reads, so the send
        // waits; and a receive, which waits for a word the laptop never sends.
        let sender = thread::spawn(move || sending.send(b"{}", &vec![7; 64 << 20]));
        let started = Instant::now();
        let received = receiving.recv();
        let took = started.elapsed();
        (received, sender.join().unwrap(), took)
    });
    assert!(
        matches!(received, Err(LinkError::Silent { after }) if after == Duration::from_millis(300)),
        "{received:?}"
    );
    assert!(
        took >= Duration::from_millis(250) && took < Duration::from_millis(1500),
        "{took:?}"
    );
    // The channel was ended both ways, so the send that waited on the same peer ended with it.
    assert!(matches!(sent, Err(LinkError::Silent { .. })), "{sent:?}");
}

#[test]
fn silence_is_counted_from_the_last_bytes_received_not_from_the_receive() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server").with_silence(Some(Duration::from_millis(400)));
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    let (on_server, mut on_laptop) = channel(&server, laptop);
    let (_sending, mut receiving) = on_server.split();
    // A message every 100 ms for 1.2 s, three times the silence allowed in all, each part of the
    // last one's body arriving within it.
    let speaking = thread::spawn(move || {
        for i in 0..12u8 {
            thread::sleep(Duration::from_millis(100));
            on_laptop.send(b"{}", &[i]).unwrap();
        }
        on_laptop
    });
    for i in 0..12u8 {
        let incoming = receiving.recv().unwrap();
        let mut body = [0; 1];
        assert_eq!(incoming.body, 1);
        receiving.read_body(&mut body).unwrap();
        assert_eq!(body, [i]);
    }
    let _on_laptop = speaking.join().unwrap();
    // Then nothing, and the silence allowed runs out.
    assert!(matches!(receiving.recv(), Err(LinkError::Silent { .. })));
}

#[test]
fn a_peer_that_goes_away_without_closing_ends_a_receive_and_a_send_waiting_on_it() {
    let pool = Authority::new(1).unwrap();
    let server = bound(&pool, "server");
    let laptop = TlsLink::new(credentials(&pool, "laptop")).unwrap();
    let (on_server, on_laptop) = channel(&server, laptop);
    let (mut sending, mut receiving) = on_server.split();
    let (received, sent) = within(Duration::from_secs(20), move || {
        let sender = thread::spawn(move || sending.send(b"{}", &vec![7; 64 << 20]));
        let receiver = thread::spawn(move || receiving.recv());
        // Once the server is waiting both ways, the laptop's end goes, with bytes unread: its
        // socket is reset, as when a program is killed, and no close is said.
        thread::sleep(Duration::from_millis(300));
        drop(on_laptop);
        (receiver.join().unwrap(), sender.join().unwrap())
    });
    assert!(
        matches!(received, Err(LinkError::Truncated | LinkError::Lost(_))),
        "{received:?}"
    );
    assert!(matches!(sent, Err(LinkError::Lost(_))), "{sent:?}");
}

#[test]
fn a_live_peer_that_reads_nothing_for_longer_than_keepalive_s_limit_keeps_its_channel() {
    const BYTES: usize = 64 << 20;
    let pool = Authority::new(1).unwrap();
    let quick = mag_mar::Keepalive {
        idle: Duration::from_secs(1),
        interval: Duration::from_secs(1),
        probes: 1,
    };
    let server = bound(&pool, "server").with_keepalive(Some(quick));
    let laptop = TlsLink::new(credentials(&pool, "laptop"))
        .unwrap()
        .with_keepalive(Some(quick));
    let (mut on_server, mut on_laptop) = channel(&server, laptop);
    let got = within(Duration::from_secs(60), move || {
        let sender = thread::spawn(move || on_server.send(b"{}", &pattern(BYTES, 3)));
        // The laptop is busy for twice keepalive's limit before it reads: the server's send
        // waits on a full window all that time, and the channel holds.
        thread::sleep(quick.limit() * 2);
        let incoming = on_laptop.recv().unwrap();
        let mut got = vec![0; incoming.body as usize];
        on_laptop.read_body(&mut got).unwrap();
        sender.join().unwrap().map(|()| got)
    });
    assert!(got.unwrap() == pattern(BYTES, 3), "the body differs");
}

#[test]
fn dialling_an_address_that_drops_packets_gives_up_at_the_opening_s_deadline() {
    let pool = Authority::new(1).unwrap();
    let laptop = TlsLink::new(credentials(&pool, "laptop"))
        .unwrap()
        .with_opening(Duration::from_millis(400));
    // An address of the documentation range, which no one answers and most networks drop.
    let (dialled, took) = within(Duration::from_secs(10), move || {
        let started = Instant::now();
        let dialled = laptop.open("server", "192.0.2.1:7340", "j");
        (dialled.map(|_| ()), started.elapsed())
    });
    match dialled {
        // Dropped: given up at the deadline, not at the system's own, minutes later.
        Err(LinkError::OpeningTimedOut { after, side }) => {
            assert_eq!((after, side), (Duration::from_millis(400), Side::Dialling));
            assert!(took >= Duration::from_millis(350), "{took:?}");
        }
        // A machine with no route there is told so at once.
        Err(LinkError::Io(_)) => {}
        other => panic!("{other:?}"),
    }
    assert!(took < Duration::from_millis(1500), "{took:?}");
}

#[cfg(unix)]
#[test]
fn a_key_file_its_group_or_others_can_reach_is_refused() {
    use std::os::unix::fs::PermissionsExt;
    let pool = Authority::new(1).unwrap();
    let issued = pool.issue("server", 1).unwrap();
    let dir = std::env::temp_dir().join(format!("mag-mar-key-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (cert, key) = (dir.join("server.pem"), dir.join("server.key"));
    std::fs::write(&cert, &issued.certificate).unwrap();
    write_key(&key, &issued.key);
    let mut start = Start::from_json(include_str!("../../mag-by/starts/example.json")).unwrap();
    start.authority = pool.certificate().into();
    start.certificate_file = cert.to_str().unwrap().into();
    start.key_file = key.to_str().unwrap().into();

    // Its owner's alone, as `certs` writes it: taken.
    for mode in [0o600, 0o400, 0o700] {
        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(mode)).unwrap();
        assert!(Credentials::from_start(&start).is_ok(), "{mode:o}");
    }
    // Readable, writable or runnable by its group or by others: refused, with its mode.
    for mode in [0o640, 0o604, 0o644, 0o620, 0o602, 0o610, 0o601, 0o660] {
        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(mode)).unwrap();
        match Credentials::from_start(&start) {
            Err(LinkError::KeyExposed { path, mode: found }) => {
                assert_eq!((path.as_str(), found), (start.key_file.as_str(), mode));
            }
            other => panic!("{mode:o}: {other:?}"),
        }
    }
    // The certificate is public: its mode is not checked.
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::set_permissions(&cert, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(Credentials::from_start(&start).is_ok());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A certificate the pool's authority signs for `member`, valid until `until` seconds after
/// the Unix epoch, and its key, in PEM.
fn issued_until(pool: &Authority, member: &str, until: u64) -> (String, String) {
    use rcgen::{CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair};
    let issuer =
        Issuer::from_ca_cert_pem(pool.certificate(), KeyPair::from_pem(&pool.key()).unwrap())
            .unwrap();
    let mut params = CertificateParams::new(vec![member.to_string()]).unwrap();
    params.is_ca = IsCa::ExplicitNoCa;
    params.extended_key_usages = vec![
        ExtendedKeyUsagePurpose::ServerAuth,
        ExtendedKeyUsagePurpose::ClientAuth,
    ];
    params.use_authority_key_identifier_extension = true;
    params.not_before = time::OffsetDateTime::from_unix_timestamp(until as i64 - 3600).unwrap();
    params.not_after = time::OffsetDateTime::from_unix_timestamp(until as i64).unwrap();
    let key = KeyPair::generate().unwrap();
    (
        params.signed_by(&key, &issuer).unwrap().pem(),
        key.serialize_pem(),
    )
}

#[test]
fn a_certificate_past_its_end_is_refused_and_a_channel_open_when_it_ends_carries_on() {
    let pool = Authority::new(1).unwrap();
    let now = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    };

    // A certificate that has ended is not taken as credentials: the member cannot start on it.
    let (ended, key) = issued_until(&pool, "laptop", now() - 60);
    assert!(matches!(
        Credentials::from_pem("laptop", pool.certificate(), &ended, &key),
        Err(LinkError::Credentials(why)) if why.contains("Expired")
    ));

    // One that ends in two seconds opens a channel now.
    let server = bound(&pool, "server");
    let until = now() + 2;
    let (certificate, key) = issued_until(&pool, "laptop", until);
    let credentials = Credentials::from_pem("laptop", pool.certificate(), &certificate, &key);
    let laptop = Arc::new(TlsLink::new(credentials.unwrap()).unwrap());
    let l = laptop.clone();
    let at = address(&server);
    let (accepted, dialled) = meet(&server, &["laptop"], move || l.open("server", &at, "j"));
    let (mut on_server, mut on_laptop) = (accepted.unwrap(), dialled.unwrap());

    // Once it has ended, the open channel carries on: a certificate is checked when a channel
    // opens.
    while now() <= until + 1 {
        thread::sleep(Duration::from_millis(100));
    }
    on_laptop.send(b"[1]", b"").unwrap();
    assert_eq!(on_server.recv().unwrap().header, b"[1]");
    on_server.send(b"[2]", b"").unwrap();
    assert_eq!(on_laptop.recv().unwrap().header, b"[2]");

    // A new channel on it is refused, as not the pool's now.
    let at = address(&server);
    let (accepted, dialled) = meet(&server, &["laptop"], move || {
        laptop.open("server", &at, "k")
    });
    assert!(
        matches!(&accepted, Err(LinkError::Refused(Refused::NotThePool(why))) if why.contains("Expired")),
        "{:?}",
        accepted.err()
    );
    assert!(dialled.is_err());
}
