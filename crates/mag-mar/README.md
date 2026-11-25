# mag-mar

The program every member of a mag pool is to run. So far it is the link between members, and
the program's skeleton: it reads `MAG_START`, joins the pool at the coordinator, and gives its
devices back on SIGTERM within its time. On the member that coordinates, it takes the members'
joins and leaves into the pool and writes the pool's ledger. Running the parts placed on a
member, and the coordinator's door, are not built yet.

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
  `MAG_START` unset, a start file that cannot be read or is refused, credentials refused, an
  address it cannot bind, or, as coordinator, a ledger it cannot begin. It exits 3 when, as
  coordinator, it could not write a line of its ledger.

The library entry is `mag_mar::member::run`; the binary takes SIGTERM as its notice
(`member::Notice::sigterm`, on Unix), reads the start file and calls it.

## The coordinator

The member a start file names as `coordinator` dials no one. It runs the coordinator, in the same
program (`mag_mar::coordinator`):

- **It begins a new ledger** at the start file's `ledger`. The pool (`mag_mortar::Pool`) begins
  with the coordinator's own member and devices, at the time it starts, in seconds since 1970.
  The ledger's header is written and synced before any member is taken. A file already at that
  place is refused, and the program exits 2: taking up the ledger of a coordinator that stopped
  is not built yet.
- **It takes a control channel from any member of the pool.** `Link::accept_any` takes whoever
  holds a certificate the pool's authority signed naming exactly one member, and reports that
  member. A member that joins later than the coordinator began is named by nothing the
  coordinator was told, only by its certificate. A channel for a job is refused, since no job is
  placed yet.
- **It checks a join.** The member a `Join` names, and the member of every device it offers, must
  be the member the certificate names. That member may hold no other live control channel, and
  may not be the coordinator itself. A join that fails is refused, and the channel ended without
  a close, so the member never reads it as an acknowledgement.
- **It records a join.** One `Change::Join` per offer is applied to the pool, and the new lines are
  written to the ledger and synced. The member is recorded as the coordinator's start file lists
  it, at the lower of the two trusts and the shorter of the two times to give a device back. A
  member the coordinator's start file does not list is recorded at `outside`, whatever its join
  says (below).
- **A member with no offers is recorded nowhere.** The pool is told of devices, not of members,
  and `Change::Join` carries one offer. Such a member holds its control channel, and its leave is
  answered at once.
- **It records a leave, then acknowledges it.** On `Leave`, one `Change::Leave` per device the
  member joined with is applied, the lines are written and synced, and only then does the
  coordinator close its way of the channel. That close is what the member waits for, within its
  `give_back_seconds`.
- **A member gone without leaving is recorded as left.** Its control channel ended without
  `Leave`: the member died, or its machine went away, and the coordinator can place nothing there.
- **On its own notice**, SIGTERM, it records its own devices' leave the same way and exits 0.
- **A ledger line that cannot be written stops it**, with exit status 3. Nothing after the last
  line written was acknowledged, and the members find their channels cut short.

**Not settled: the trust of a member that joins later.** A member the coordinator's start file
does not list is known only by its own join, and a machine holding a certificate the host meant
for `outside` could say `owner`. Either the host writes the trust into the certificate it signs,
and the coordinator believes nothing the member says, or such a member is trusted at `outside`.
Until that is settled, the coordinator takes the second, which cannot place data where it should
not go. `coordinator::recorded_member` is the one place either answer changes.

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
- **Any member of the pool**: `accept` takes only the members a caller expects; `accept_any`
  takes any member the pool's authority signed a certificate for, and refuses a certificate that
  names more than one member, which could stand for either.
- **Nothing waits for ever.** Dialling, the handshake and the opening words share one deadline,
  10 s unless set, and each address a name resolves to is tried in turn while time is left. An
  opening that runs out of it says so, with the time it had and whether this end was dialling or
  answering. On an open channel, TCP keepalive (probes after 10 s of quiet, every 5 s, three of
  them, unless set) finds a peer whose machine went away while the channel was quiet, and a
  receive waiting on it fails. While this end has sent bytes the peer has not acknowledged, TCP
  does not probe, and the system's limit on retransmitting decides, on Linux about 15 minutes. A
  peer whose machine is up but whose program has stopped still answers keepalive. So a link may
  also be given a limit on silence: nothing received for that long ends the channel both ways, a
  send waiting on the same peer with it. It is off unless set, since a job may wait, or take long
  over a step, while its channels are quiet. No bound is set on data left unacknowledged: on
  Linux that bound also cut a channel whose peer was alive but not reading.
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
a start file without `listen` dials only. And it runs the program as coordinator, with members
as processes of their own: two members' joins in its ledger, one entry per device; a leave on
SIGTERM in the ledger as the member exits, within its time, and the ledger replayed by
`mag_mortar::Pool::replay` to the same lines; joins refused for another member's name or device,
on a job's channel, as the coordinator, and for a member already joined; an unlisted member at
`outside`; a member with no offers recorded nowhere; a member killed recorded as left; a ledger
already there left as it is; and, on Linux, a leave whose ledger line cannot be written never
acknowledged. `loopback` sends one tensor of any size and reports the
rate and the memory the link took beyond the two buffers:

    cargo run --release -p mag-mar --features issue --example loopback -- 4

A tensor of 4 GiB arrived byte for byte at 1.25 to 1.42 GB/s, on one channel, on an x86-64
desktop CPU under WSL2 that was not idle. The process held at most 104 KiB more than before
sending (2026-11-25, three runs).

## Not built yet

- The door: submitting a job, ending one, taking back by need, and the pool's status.
- The coordinator's words to members: announcing a member to its peers, and sending placements.
- Taking up the ledger of a coordinator that stopped, and members joining again when one starts.
- Running the parts placed on a member, channels to other members for their jobs, and giving
  back with the parts' state at a step.
- Refusing a member whose lease ended until it has a new certificate.
- A second transport over RDMA, which is to wait until a run under Slurm shows the link limiting.
- Which member may open which job's channel: a member takes a channel for any job from any member
  it expects, until the member program has a placement to check it against.
