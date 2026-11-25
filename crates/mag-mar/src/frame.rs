//! Messages on a byte stream, and the opening of a channel.
//!
//! A message is a prefix of 12 bytes, the header's length (4 bytes) and the body's length (8
//! bytes), each little-endian; then the header, one of `mag-by`'s words in JSON, of at most
//! [`MAX_HEADER`] bytes; then the body, of any length. A body is written from the caller's
//! buffer and read into the caller's buffer, a piece at a time as the stream takes it, so a
//! tensor of several gigabytes is never held twice.
//!
//! A channel opens with one message each way, a `mag_by::Open` with no body: the end that dials
//! names the job and its wire version, and the other answers with the same job and its own.

use std::io::{self, Read, Write};
use std::time::Instant;

use mag_by::Open;

use crate::{Incoming, LinkError, Measure};

/// The longest header taken, in bytes: 1 MiB. A word is small; what is large goes in a body.
pub const MAX_HEADER: u32 = 1 << 20;

const PREFIX: usize = 12;

/// Writes one message, and returns the body's bytes and the time they took.
pub(crate) fn write<W: Write>(w: &mut W, header: &[u8], body: &[u8]) -> Result<Measure, LinkError> {
    if header.len() > MAX_HEADER as usize {
        return Err(LinkError::HeaderTooLong {
            bytes: header.len() as u64,
        });
    }
    let mut prefix = [0; PREFIX];
    prefix[..4].copy_from_slice(&(header.len() as u32).to_le_bytes());
    prefix[4..].copy_from_slice(&(body.len() as u64).to_le_bytes());
    w.write_all(&prefix)?;
    w.write_all(header)?;
    let start = Instant::now();
    w.write_all(body)?;
    w.flush()?;
    let mut m = Measure::default();
    if !body.is_empty() {
        m.add(body.len() as u64, start.elapsed().as_secs_f64());
    }
    Ok(m)
}

/// Fills `buf`. The stream ending before the first byte is the other end closing the channel
/// if `between` (the reader is between two messages); anywhere else it is a message cut short.
fn fill<R: Read>(r: &mut R, buf: &mut [u8], between: bool) -> Result<(), LinkError> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) if between && got == 0 => return Err(LinkError::Closed),
            Ok(0) => return Err(LinkError::Truncated),
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                return Err(LinkError::Truncated);
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// Reads the next message's header, and the length of its body. `left` is what is unread of
/// the last body.
pub(crate) fn read<R: Read>(r: &mut R, left: u64) -> Result<Incoming, LinkError> {
    if left != 0 {
        return Err(LinkError::BodyUnread { left });
    }
    let mut prefix = [0; PREFIX];
    fill(r, &mut prefix, true)?;
    let header_len = u32::from_le_bytes(prefix[..4].try_into().expect("4 bytes"));
    let body = u64::from_le_bytes(prefix[4..].try_into().expect("8 bytes"));
    if header_len > MAX_HEADER {
        return Err(LinkError::HeaderTooLong {
            bytes: header_len.into(),
        });
    }
    let mut header = vec![0; header_len as usize];
    fill(r, &mut header, false)?;
    Ok(Incoming { header, body })
}

/// Reads the body of the message last read, all `left` bytes of it, into `into`, and returns
/// its bytes and the time they took.
pub(crate) fn read_body<R: Read>(
    r: &mut R,
    left: &mut u64,
    into: &mut [u8],
) -> Result<Measure, LinkError> {
    if into.len() as u64 != *left {
        return Err(LinkError::Body {
            expected: *left,
            found: into.len() as u64,
        });
    }
    let start = Instant::now();
    fill(r, into, false)?;
    *left = 0;
    let mut m = Measure::default();
    if !into.is_empty() {
        m.add(into.len() as u64, start.elapsed().as_secs_f64());
    }
    Ok(m)
}

fn send_open<W: Write>(w: &mut W, wire: u64, job: &str) -> Result<(), LinkError> {
    let open = Open {
        wire,
        job: job.into(),
    };
    let header = serde_json::to_vec(&open).map_err(LinkError::Word)?;
    write(w, &header, &[])?;
    Ok(())
}

/// Reads the opening the end that dialled sends: the first half of taking a channel.
pub(crate) fn read_open<R: Read>(r: &mut R) -> Result<Open, LinkError> {
    let incoming = read(r, 0)?;
    if incoming.body != 0 {
        return Err(LinkError::Opening(format!(
            "the opening came with a body of {} bytes",
            incoming.body
        )));
    }
    serde_json::from_slice(&incoming.header)
        .map_err(|e| LinkError::Opening(format!("the first word was not an opening: {e}")))
}

/// Opens a channel for `job` from the end that dialled, speaking `wire`.
pub(crate) fn open_as_dialler<S: Read + Write>(
    s: &mut S,
    job: &str,
    wire: u64,
) -> Result<(), LinkError> {
    // An end that will not take the channel closes it, and a closed connection shows here as
    // the stream ending or being reset, whichever this end meets first.
    let refused = |e| match e {
        LinkError::Closed | LinkError::Truncated => {
            LinkError::RefusedBy("it closed the channel before it was open".into())
        }
        LinkError::Io(e)
            if matches!(
                e.kind(),
                io::ErrorKind::ConnectionReset
                    | io::ErrorKind::ConnectionAborted
                    | io::ErrorKind::BrokenPipe
            ) =>
        {
            LinkError::RefusedBy(format!("it closed the channel before it was open ({e})"))
        }
        e => e,
    };
    send_open(s, wire, job).map_err(refused)?;
    let answer = read_open(s).map_err(refused)?;
    if answer.wire != wire {
        return Err(LinkError::Wire {
            ours: wire,
            theirs: answer.wire,
        });
    }
    if answer.job != job {
        return Err(LinkError::Opening(format!(
            "opened for `{job}`, answered for `{}`",
            answer.job
        )));
    }
    Ok(())
}

/// Takes a channel at the end that answered, speaking `wire`, once its opening `open` is read
/// ([`read_open`]), and returns its job. It answers with its own version even when the two
/// differ, so that the end that dialled sees why.
pub(crate) fn answer_open<W: Write>(w: &mut W, wire: u64, open: Open) -> Result<String, LinkError> {
    send_open(w, wire, &open.job)?;
    if open.wire != wire {
        return Err(LinkError::Wire {
            ours: wire,
            theirs: open.wire,
        });
    }
    Ok(open.job)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    use super::*;

    /// Takes a channel at the end that answered: its opening read, then answered.
    fn open_as_acceptor<S: Read + Write>(s: &mut S, wire: u64) -> Result<String, LinkError> {
        let open = read_open(s)?;
        answer_open(s, wire, open)
    }

    fn written(messages: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (header, body) in messages {
            write(&mut out, header, body).unwrap();
        }
        out
    }

    #[test]
    fn a_message_is_its_prefix_its_header_and_its_body() {
        let out = written(&[(b"{}", b"abc")]);
        assert_eq!(&out[..4], &2u32.to_le_bytes());
        assert_eq!(&out[4..12], &3u64.to_le_bytes());
        assert_eq!(&out[12..], b"{}abc");
    }

    #[test]
    fn messages_read_back_in_order_each_body_whole() {
        let out = written(&[(b"one", b"12345"), (b"two", b""), (b"", b"xy")]);
        let mut r = Cursor::new(out);
        let mut left = 0;
        let first = read(&mut r, left).unwrap();
        assert_eq!(
            first,
            Incoming {
                header: b"one".to_vec(),
                body: 5
            }
        );
        left = first.body;
        let mut body = [0; 5];
        let m = read_body(&mut r, &mut left, &mut body).unwrap();
        assert_eq!((&body, left, m.bytes), (b"12345", 0, 5));
        let second = read(&mut r, left).unwrap();
        assert_eq!(
            second,
            Incoming {
                header: b"two".to_vec(),
                body: 0
            }
        );
        // An empty body reads as nothing, and counts nothing.
        assert_eq!(
            read_body(&mut r, &mut left, &mut []).unwrap(),
            Measure::default()
        );
        let third = read(&mut r, 0).unwrap();
        assert_eq!(
            third,
            Incoming {
                header: Vec::new(),
                body: 2
            }
        );
        left = 2;
        let mut body = [0; 2];
        read_body(&mut r, &mut left, &mut body).unwrap();
        assert_eq!(&body, b"xy");
        assert!(matches!(read(&mut r, 0), Err(LinkError::Closed)));
    }

    #[test]
    fn a_header_of_the_longest_length_is_taken_and_one_byte_more_refused() {
        let longest = vec![b'x'; MAX_HEADER as usize];
        let out = written(&[(&longest, b"")]);
        assert_eq!(
            read(&mut Cursor::new(out), 0).unwrap().header.len(),
            longest.len()
        );
        let longer = vec![b'x'; MAX_HEADER as usize + 1];
        let mut sink = Vec::new();
        assert!(matches!(
            write(&mut sink, &longer, b""),
            Err(LinkError::HeaderTooLong { bytes }) if bytes == u64::from(MAX_HEADER) + 1
        ));
        assert!(sink.is_empty(), "nothing is written of a message refused");
        // A prefix that states a longer header is refused before any of it is read.
        let mut prefix = (MAX_HEADER + 1).to_le_bytes().to_vec();
        prefix.extend_from_slice(&0u64.to_le_bytes());
        assert!(matches!(
            read(&mut Cursor::new(prefix), 0),
            Err(LinkError::HeaderTooLong { bytes }) if bytes == u64::from(MAX_HEADER) + 1
        ));
    }

    #[test]
    fn a_stream_that_ends_between_messages_is_closed_and_inside_one_cut_short() {
        let out = written(&[(b"head", b"body")]);
        // Ends before any byte: closed. Inside the prefix or the header: cut short.
        assert!(matches!(
            read(&mut Cursor::new(&out[..0]), 0),
            Err(LinkError::Closed)
        ));
        for end in [1, 11, 12, 15] {
            assert!(
                matches!(
                    read(&mut Cursor::new(&out[..end]), 0),
                    Err(LinkError::Truncated)
                ),
                "ending at {end}"
            );
        }
        // Inside the body.
        let mut r = Cursor::new(&out[..19]);
        let mut left = read(&mut r, 0).unwrap().body;
        let mut body = [0; 4];
        assert!(matches!(
            read_body(&mut r, &mut left, &mut body),
            Err(LinkError::Truncated)
        ));
    }

    #[test]
    fn a_body_is_read_whole_before_the_next_message_into_a_buffer_of_its_length() {
        let out = written(&[(b"a", b"1234"), (b"b", b"")]);
        let mut r = Cursor::new(out);
        let mut left = read(&mut r, 0).unwrap().body;
        assert!(matches!(
            read(&mut r, left),
            Err(LinkError::BodyUnread { left: 4 })
        ));
        for len in [3, 5] {
            let mut wrong = vec![0; len];
            assert!(matches!(
                read_body(&mut r, &mut left, &mut wrong),
                Err(LinkError::Body { expected: 4, found }) if found == len as u64
            ));
        }
        let mut body = [0; 4];
        read_body(&mut r, &mut left, &mut body).unwrap();
        assert_eq!(read(&mut r, left).unwrap().header, b"b");
    }

    /// The two ends of a connection on loopback.
    fn pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let dialled = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (answered, _) = listener.accept().unwrap();
        (dialled, answered)
    }

    #[test]
    fn a_channel_opens_for_the_job_named_by_the_end_that_dials() {
        let (mut d, mut a) = pair();
        let answering = thread::spawn(move || open_as_acceptor(&mut a, 1));
        open_as_dialler(&mut d, "job-a", 1).unwrap();
        assert_eq!(answering.join().unwrap().unwrap(), "job-a");
    }

    #[test]
    fn ends_of_two_wire_versions_each_refuse_the_channel_and_say_both() {
        let (mut d, mut a) = pair();
        let answering = thread::spawn(move || open_as_acceptor(&mut a, 1));
        assert!(matches!(
            open_as_dialler(&mut d, "job", 2),
            Err(LinkError::Wire { ours: 2, theirs: 1 })
        ));
        assert!(matches!(
            answering.join().unwrap(),
            Err(LinkError::Wire { ours: 1, theirs: 2 })
        ));
    }

    #[test]
    fn an_answer_for_another_job_or_no_opening_at_all_is_refused() {
        let (mut d, mut a) = pair();
        let answering = thread::spawn(move || {
            read_open(&mut a).unwrap();
            send_open(&mut a, 1, "job-b").unwrap();
        });
        assert!(matches!(
            open_as_dialler(&mut d, "job-a", 1),
            Err(LinkError::Opening(why)) if why == "opened for `job-a`, answered for `job-b`"
        ));
        answering.join().unwrap();

        let (mut d, mut a) = pair();
        write(&mut d, br#"{"step": 3}"#, b"").unwrap();
        assert!(matches!(
            open_as_acceptor(&mut a, 1),
            Err(LinkError::Opening(_))
        ));

        let (mut d, mut a) = pair();
        let open = serde_json::to_vec(&Open {
            wire: 1,
            job: "j".into(),
        })
        .unwrap();
        write(&mut d, &open, b"x").unwrap();
        assert!(matches!(
            open_as_acceptor(&mut a, 1),
            Err(LinkError::Opening(_))
        ));

        // An end that closes instead of answering refuses the one that dialled.
        let (mut d, a) = pair();
        drop(a);
        assert!(matches!(
            open_as_dialler(&mut d, "j", 1),
            Err(LinkError::RefusedBy(_))
        ));
    }
}
