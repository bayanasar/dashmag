# mag-mar

The program every member of a mag pool is to run. So far it is the link between members, and
the program's skeleton: it reads `MAG_START`, joins the pool at the coordinator, and gives its
devices back on SIGTERM within its time. Running the parts placed on its member, and serving the
pool on the member that coordinates, are not built yet.

## The member program

The host starts `mag-mar` on each member, with the environment variable `MAG_START` naming its
start file (`mag-by`'s `Start`):

    MAG_START=/etc/mag/server.json cargo run --release -p mag-mar

- **It loads its credentials** from the certificate and key files the start file names, and
  refuses a key file its group or others can reach.
- **It binds its `listen` address**, where the other members are to dial it. A start file with
  no `listen` gives a member that only dials, as one behind address translation does.
- **It joins.** Unless it is the coordinator, it dials the coordinator at the address its peer
  entry gives, and opens a control channel: a channel for no job, `mag_by::CONTROL`, the empty
  name, which no job may have. On it, it says `mag_by::Control::Join`, with itself as its start
  file describes it (its trust and its `give_back_seconds`, as the host wrote them) and what its
  devices offer. The offers are those the start file's pool gives for its devices, until the
  member measures its own. While the coordinator cannot be reached, it dials again every second.
- **On SIGTERM it gives back.** It says `mag_by::Control::Leave`, closes its way of the control
  channel, and waits for the coordinator to close the other way, the sign that the leaving is
  recorded. Then it exits 0. With no control channel open, before it joined or after the channel
  ended, it exits 0 at once.
- **Its `give_back_seconds` hold whatever any peer does.** The signal sets a deadline on a thread
  of its own, a tenth of the time before its end and never more than 1 s before it, and at that
  deadline the process exits 0, wherever the program is waiting: on a coordinator that stopped
  answering, or on a dial that has not opened. A link's keepalive cannot promise this, since TCP
  does not probe while data is unacknowledged.
- **What it does goes to standard error**, a line at a time. It exits 2 when it cannot start:
  `MAG_START` unset, a start file that cannot be read or is refused, credentials refused, or an
  address it cannot bind.

The library entry is `mag_mar::member::run`; the binary takes SIGTERM as its notice
(`member::Notice::sigterm`, on Unix), reads the start file and calls it.

The coordinator's own program is not built: a member that coordinates binds its address and waits
for its notice, and a member that dials it waits in its opening until the opening gives up, and
dials again.

## The link

- **One small interface**, `Link` and `Channel`, so that another transport can stand in for the
  first. A member opens a channel to another for one job, or accepts one the other opens. On it,
  each end sends and receives messages, each a header and a body. A channel reports the rate
  measured on it, and a link reports the rate to each member, for the placement loop's cost
  model. Then the channel is closed. Each job between two members has its own channel, so one
  job's words never reach another's.
- **Both ends send at once.** A channel splits into a sending half and a receiving half, each for
  a thread of its own. The two share the TLS session, locked only while records are made or
  read, never while either waits on the socket. So a send waits only for the other end to
  read, never for its own end, and two ends that each send a large body at the same moment both
  finish. A channel kept whole, on one thread, takes turns.
- **The first transport is TCP with TLS 1.3**, on rustls with ring, and both ends are
  authenticated. Each pool has its own certificate authority, which the host holds. The host
  issues every member a certificate signed by it, naming the member. A channel is taken only when
  the other end's certificate is signed by the pool's authority and names the member expected.
  Each refusal is typed: no certificate, another authority's certificate (or one signed by a
  member), a certificate of another member, a version of TLS below 1.3. Either end may dial, so a
  member behind address translation dials out.
- **The messages are `mag-by`'s words**, in JSON in the header. A channel opens with `Open`
  each way, naming the job and the wire version. A tensor is a `Tensor` word, with its bytes as
  the body. The body is written from the caller's buffer and read into the caller's buffer as the
  stream takes it, so a tensor of several gigabytes is never held twice.
- **Who the other end is**: the member its certificate names, and that certificate's serial.
  Every certificate has a new serial, so a certificate issued after a lease ended can be told
  from the one it ended on. Refusing the old one is the coordinator's, and not built yet.
- **Nothing waits for ever.** Dialling, the handshake and the opening words share one deadline,
  10 s unless set, and each address a name resolves to is tried in turn while time is left. On
  an open channel, TCP keepalive (probes after 10 s of quiet, every 5 s, three of them, unless
  set) finds a peer whose machine went away while the channel was quiet, and a receive waiting
  on it fails. While this end has sent bytes the peer has not acknowledged, TCP does not probe,
  and the system's limit on retransmitting decides, on Linux about 15 minutes. A peer whose
  machine is up but whose program has stopped still answers keepalive. So a link may also be
  given a limit on silence: nothing received for that long ends the channel both ways, a send
  waiting on the same peer with it. It is off unless set, since a job may wait, or take long over
  a step, while its channels are quiet. No bound is set on data left unacknowledged: on Linux
  that bound also cut a channel whose peer was alive but not reading.
- **A member's key is its own.** Read from a start file, a key file that its group or others may
  read, write or run is refused, on Unix.

A member's name must be a DNS name in lower case (`workstation`, `gpu-1`), since it is carried
in the certificate as one.

## Making the certificates

The host runs `certs` for its list of machines. It writes the authority's certificate and key,
and each member's certificate and key:

    cargo run -p mag-mar --features issue --example certs -- \
        new pool-certs workstation server laptop

`authority.pem` goes into every member's start file, as `authority`. `authority.key` stays
with the host. Each `MEMBER.pem` and `MEMBER.key` go to that machine, named by its start file's
`certificate_file` and `key_file`. Keys are written readable by their owner alone, and must stay
so. For a member that joins later, or comes back after its lease ended, the host issues from the
same authority:

    cargo run -p mag-mar --features issue --example certs -- add pool-certs rented

A member's certificate is valid for 365 days unless `--days` says otherwise, and the authority
for ten years. Once a member's certificate has ended, the member cannot start on it, and the
other members refuse any new channel to or from it; channels already open carry on, since a
certificate is checked when a channel opens. Before then, the host issues it a new one with
`add`, the same command, and the member starts again with the new files. No other member needs
anything new. `certs help` says the same.

## Measured

`cargo test -p mag-mar` runs two members over loopback, each refusal, two jobs' channels, a
tensor of 64 MiB byte for byte, and both ends of one channel sending 64 MiB at the same moment.
It runs the member program as a process of its own against a stand-in coordinator: the member
joins with its offers and leaves on SIGTERM; it exits 0 within its give-back time when the
coordinator stops answering, and when it is still opening its channel to one that never answers;
a start file without `listen` dials only. `loopback` sends one tensor of any size and reports the
rate and the memory the link took beyond the two buffers:

    cargo run --release -p mag-mar --features issue --example loopback -- 4

A tensor of 4 GiB arrived byte for byte at 1.25 to 1.42 GB/s, on one channel, on an x86-64
desktop CPU under WSL2 that was not idle. The process held at most 104 KiB more than before
sending (2026-11-25, three runs).

## Not built yet

- The coordinator and the door; joining again after the coordinator comes back.
- Running the parts placed on a member, channels to other members for their jobs, and giving
  back with the parts' state at a step.
- Refusing a member whose lease ended until it has a new certificate.
- A second transport over RDMA, which is to wait until a run under Slurm shows the link limiting.
- Which member may open which job's channel: a member takes a channel for any job from any member
  it expects, until the member program has a placement to check it against.
