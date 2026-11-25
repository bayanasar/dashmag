//! `loopback`: one tensor sent between two members of a new pool over loopback, as a measure of
//! the link.
//!
//!     loopback [GIB]   a tensor of GIB gibibytes (4 by default)
//!
//! It makes a pool's authority and two members, the sender's tensor and the receiver's buffer
//! for it, then sends the tensor on one channel and checks it arrived byte for byte. It prints
//! the size, the time and the rate as each end measured it, and on Linux the most memory the
//! process held above what it held before sending: what the link itself costs to carry the
//! tensor, beyond the two buffers. Run it with `--release`.

use std::thread;
use std::time::Instant;

use mag_by::{Dtype, Tensor};
use mag_mar::issue::Authority;
use mag_mar::{Channel, Credentials, Link, TlsLink};

/// A value of `/proc/self/status`, in kibibytes.
fn status_kib(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with(field))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn credentials(authority: &Authority, member: &str) -> Credentials {
    let issued = authority.issue(member, 1).expect("issued");
    Credentials::from_pem(
        member,
        authority.certificate(),
        &issued.certificate,
        &issued.key,
    )
    .expect("credentials")
}

fn main() {
    let gib: f64 = std::env::args()
        .nth(1)
        .map(|a| a.parse().expect("a size in GiB"))
        .unwrap_or(4.0);
    let bytes = (gib * (1u64 << 30) as f64) as usize;

    let pool = Authority::new(1).expect("an authority");
    let server = TlsLink::bind(credentials(&pool, "server"), "127.0.0.1:0").expect("bound");
    let laptop = TlsLink::new(credentials(&pool, "laptop")).expect("a link");
    let at = server.local_addr().expect("an address").to_string();

    // Both buffers written to, so that every page is held before the measure begins.
    let mut sent = vec![0u8; bytes];
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for chunk in sent.chunks_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        chunk.copy_from_slice(&x.to_le_bytes()[..chunk.len()]);
    }
    let mut got = vec![1u8; bytes];
    let tensor = Tensor {
        from: "b0".into(),
        to: Some("b1".into()),
        step: 0,
        dtype: Dtype::U8,
        shape: vec![bytes as u64],
        bytes: bytes as u64,
    };

    let dialling = thread::spawn(move || laptop.open("server", &at, "measure").expect("open"));
    let mut receiving = server.accept(&["laptop"]).expect("accepted");
    let mut sending = dialling.join().expect("the dialling thread");

    // Linux: 5 in clear_refs sets the high-water mark back to what is held now.
    let reset = std::fs::write("/proc/self/clear_refs", "5").is_ok();
    let before = status_kib("VmRSS:");
    let start = Instant::now();
    let rate = thread::scope(|s| {
        let sender = s.spawn(|| {
            sending.send_tensor(&tensor, &sent).expect("sent");
            sending.rate()
        });
        let word = receiving.recv_tensor().expect("the tensor's word");
        assert_eq!(word, tensor);
        receiving.read_body(&mut got).expect("the tensor's bytes");
        (sender.join().expect("the sending thread"), receiving.rate())
    });
    let wall = start.elapsed().as_secs_f64();
    let peak = status_kib("VmHWM:");
    assert!(got == sent, "the bytes differ");

    let (sent_rate, received_rate) = rate;
    let gb = |b: f64| b / 1e9;
    println!("tensor            {bytes} bytes ({gib} GiB), arrived byte for byte");
    println!(
        "wall time         {wall:.3} s, {:.3} GB/s",
        gb(bytes as f64 / wall)
    );
    println!(
        "sender measured   {:.3} GB/s",
        gb(sent_rate.sent.bytes_per_second().unwrap_or(0.0))
    );
    println!(
        "receiver measured {:.3} GB/s",
        gb(received_rate.received.bytes_per_second().unwrap_or(0.0))
    );
    match (reset, before, peak) {
        (true, Some(before), Some(peak)) => println!(
            "peak extra memory {} KiB above the {} KiB held before sending",
            peak.saturating_sub(before),
            before
        ),
        _ => println!("peak extra memory not measured here (Linux /proc only)"),
    }
    sending.close().expect("closed");
}
