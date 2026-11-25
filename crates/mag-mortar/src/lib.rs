//! The placement library of a mag pool: the loop that decides where each part of each job runs.
//!
//! A host gives the pool its machines and takes them back, and never reads a graph. One of the
//! pool's machines, which the host names, coordinates: it runs a [`Pool`], which places every job
//! submitted to the pool against the others. mag keeps no server of its own, and this crate does
//! no I/O.
//!
//! On every event (a job arrives or ends, a device joins or leaves) [`Pool`] runs the same steps
//! over the jobs the event touches, which here are the jobs waiting:
//!
//! 1. **Generate** candidate arrangements: each touched job either waits or is placed. A chain's
//!    parts go together on one device, and each other atomic part on one device. A replicable
//!    part then gets a copy on every device that may take it and has room for it beside what
//!    is placed there (running jobs' parts and the candidate's own); a device left out is
//!    recorded with why. When the candidates number at most [`EXHAUSTIVE`] every one is listed;
//!    past that, counted without being built, the jobs are placed one at a time, highest
//!    priority first, each group of a job on the device that adds least to the busiest device's
//!    time, and the record says so.
//! 2. **Remove** a candidate that breaks a constraint, with a typed reason: first any part's
//!    eligibility over every part (no such device, the guard, the job's stamp, a pin, its
//!    tolerance class, its edges' dtypes), then the memory each device offers, then a job's
//!    need, a rate it must sustain, for every placed job. The guard compares levels: a part kept
//!    at a level goes only on a device of a member trusted at least as far, and a member the
//!    pool does not know is trusted at `outside`. The stamp keeps a job to the devices it names.
//! 3. **Predict** each remaining candidate's score: the sum over jobs of their priority times
//!    their rate, each rate divided by the job's best rate alone on the pool as it now is. A
//!    job's priority is its graph's, held to its stamp's ceiling. A waiting job's priority is
//!    that times `1 + waited / AGING_SECONDS`: 2 at 60 s, 3 at 120 s, so that no job starves,
//!    whatever its ceiling. Once placed, its waiting time starts again from 0.
//! 4. **Commit** the best candidate if it scores above leaving things as they are, and **record**
//!    the run.
//!
//! The rate model is this step's, and simple. A job's rate is the inverse of its busiest
//! device's time per step; a device's time is the sum of what every part on it, of any job,
//! takes per step, and a replicable part's copies share its items in proportion to their speed.
//! Edges are not timed, so a part others wait on is placed for its own time, not for theirs.
//!
//! An author asks the pool two things before submitting: [`Pool::describe`], the pool as one
//! virtual device, and [`Pool::plan`], the record a graph's arrival would make, with the pool
//! left as it is.
//!
//! The pool asks its host two things, [`Pool::asks`], which the host polls; the pool never calls
//! it. Both are pure functions of the pool's state and the time of asking, and leave it as it is:
//! - [`Pool::wants`], the kind of device it most wants while jobs wait: of the kinds that would
//!   take a waiting job's parts (their memory, a pinned backend, the trust the guard asks for,
//!   their tolerance classes and dtypes), the one whose device, joining, the loop would place
//!   parts on for the greatest gain in score, and of equal gains the smallest;
//! - [`Pool::gives_back`], the members that have held no part for the host's idle time, never
//!   the coordinator's own. When each member last held a part is kept from the changes the pool
//!   was told, so a replay keeps it, and the ledger is unchanged.
//!
//! When the host takes a device back by need, a kind of device within a deadline, the pool
//! chooses which device to give up, [`Pool::give_up`]: of the devices of that kind whose members
//! give a device back within the deadline, the one whose leave, the loop placing its jobs again
//! or leaving them waiting, costs the pool's score least, the coordinator's own devices last. It
//! too is a pure function of the pool's state and the time of asking; the leave itself is a
//! change the pool is then told.
//!
//! The pool does no I/O. Every change reaches it through [`Pool::apply`], which adds the change
//! and the record of the loop's run to its [`Ledger`]. The pool is deterministic, so the ledger
//! is all its host need keep: [`Pool::replay`] applies the changes again to a pool begun the same
//! way, and compares each record with the one logged. With the `files` feature the ledger is
//! written and read as lines (`mag_by::Ledger::line`, `mag_by::Ledger::from_lines`).
//!
//! What this step leaves out, and where it comes:
//! - probing candidates whose bands overlap, which needs runs (with `mag-mar`);
//! - the time of edges, within a device and across a link;
//! - dividing a device's time between the jobs on it by priority, as the design has it, rather
//!   than summing it; and a replicable job's copies taking more items where others are busy,
//!   rather than one busy copy setting its rate (M-6);
//! - dividing a chain across devices, choosing to share a device with a running job, and
//!   squeezing a running job (M-6). Running jobs are not moved here, so moving costs nothing;
//! - memory over time: a device's parts are summed as if all resident at once;
//! - a deadline: a job states no length yet;
//! - beginning a pool from the arrangement its members hold, after its coordinator stops. A
//!   coordinator of the same version replays the ledger; one of another version begins a pool
//!   that follows it ([`Pool::follow`]), from the members, devices and jobs it was told of, and
//!   places the jobs afresh, since no part runs on a member yet for it to ask where it is.
//!
//! The same members, offers and events give the same records: devices are taken in the order of
//! their [`DeviceId`], and the first of equal scores wins.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use mag_by::{
    Arrangement, Asks, Backend, Candidate, Change, Cost, Description, DeviceId, DeviceKind, Dtype,
    Entry, Event, Format, GRAPH_FORMAT, GRAPH_VERSION, Graph, Held, Keeps, LEDGER_FORMAT,
    LEDGER_VERSION, Ledger, Mark, Member, Move, Offer, POOL_FORMAT, POOL_VERSION, Pin, Place,
    Record, Refusal, Removal, Stamp, Trust, Unit,
};

/// This crate's version, which a ledger it begins records as `written_by`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The most candidates listed one by one before generation turns greedy (proposed).
pub const EXHAUSTIVE: usize = 500;

/// A waiting job's priority grows by its own value every this many seconds (proposed).
pub const AGING_SECONDS: f64 = 60.0;

/// The band given to an estimated cost, relative: wide, so that it is probed and not trusted.
pub const ESTIMATE_BAND: f64 = 1.0;

/// The member of the device a trial of a wanted kind adds: no member's name, since a member is
/// named in lower case, and after every one of them in order, so that of equal scores a device
/// the pool has wins.
const WANTED: &str = "~";

/// Why the pool refuses a change. A refused change leaves the pool as it was, and is not in its
/// ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChangeError {
    /// The graph fails [`Graph::check`].
    Graph(Refusal),
    /// The stamp's priority ceiling is 0, which no job's priority may be.
    ZeroCeiling,
    /// The stamp names no device, so the job could never run.
    NoDevices,
    /// The pool already holds a job of this name.
    Duplicate { job: String },
    /// The time is not finite.
    Time,
}

impl fmt::Display for ChangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChangeError::Graph(r) => write!(f, "the graph is refused: {r}"),
            ChangeError::ZeroCeiling => {
                write!(
                    f,
                    "the stamp's priority ceiling is 0; it must be at least 1"
                )
            }
            ChangeError::NoDevices => write!(f, "the stamp names no device the job may use"),
            ChangeError::Duplicate { job } => {
                write!(f, "the pool already holds a job named `{job}`")
            }
            ChangeError::Time => write!(f, "the time is not finite"),
        }
    }
}

impl std::error::Error for ChangeError {}

/// The device a pool gives up for a need, and every device it weighed: see [`Pool::give_up`].
#[derive(Clone, Debug, PartialEq)]
pub struct GiveUp {
    /// The device given up: the first of `ranked`.
    pub device: DeviceId,
    /// The score the pool loses by it, as the loop scores: 0 for a device that holds no part,
    /// and the more the more its jobs lose, by priority, after the loop places them again.
    pub loss: f64,
    /// Every device that meets the need, with its loss, in the order they were ranked: the
    /// coordinator's own last, then the least loss, the least memory and the device's id.
    pub ranked: Vec<(DeviceId, f64)>,
}

/// Why a pool gives up no device for a need: nothing it holds meets it. See [`Pool::give_up`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unmet {
    /// The devices of the kind asked for, each of a member that takes longer than the deadline
    /// to give a device back, or that the pool was never told of; empty when no device in the
    /// pool is of the kind.
    pub slow: Vec<DeviceId>,
}

impl fmt::Display for Unmet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.slow.is_empty() {
            return write!(f, "no device in the pool is of the kind asked for");
        }
        let slow: Vec<String> = self.slow.iter().map(ToString::to_string).collect();
        write!(
            f,
            "the devices of the kind asked for ({}) are each of a member that takes longer than \
             the deadline to give a device back",
            slow.join(", ")
        )
    }
}

impl std::error::Error for Unmet {}

/// Why a ledger was not replayed. Entries are counted from 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    /// The pool refuses this entry's change.
    Refused { entry: usize, error: ChangeError },
    /// This entry's change made a record other than the one logged: the ledger is not this
    /// pool's, or not this version's.
    Differs { entry: usize },
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplayError::Refused { entry, error } => {
                write!(f, "entry {entry} is refused on replay: {error}")
            }
            ReplayError::Differs { entry } => write!(
                f,
                "entry {entry} made another record on replay than the one logged"
            ),
        }
    }
}

impl std::error::Error for ReplayError {}

/// Why a pool is not begun from a description: it holds jobs, and a description has their places
/// but not their graphs. Such a pool comes back from its ledger instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HoldsJobs {
    pub jobs: Vec<String>,
}

impl fmt::Display for HoldsJobs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the pool holds jobs ({}), whose graphs a description does not carry; replay its \
             ledger instead",
            self.jobs.join(", ")
        )
    }
}

impl std::error::Error for HoldsJobs {}

/// A job the pool holds: its graph and stamp, the level each part is kept at, where its parts
/// are, and how long it has waited.
#[derive(Clone, Debug)]
struct Live {
    graph: Graph,
    stamp: Stamp,
    keeps: Keeps,
    arrangement: Arrangement,
    waited: f64,
    best_alone: f64,
}

impl Live {
    /// Its priority: its graph's, held to its stamp's ceiling.
    fn priority(&self) -> u32 {
        match self.stamp.priority_ceiling {
            Some(ceiling) => self.graph.job.priority.min(ceiling),
            None => self.graph.job.priority,
        }
    }

    fn waiting(&self) -> bool {
        self.arrangement
            .places
            .iter()
            .any(|p| matches!(p, Place::Waiting))
    }

    /// The groups placed as one: each chain's parts, and each atomic part outside a chain.
    fn groups(&self) -> Vec<Vec<usize>> {
        let g = &self.graph;
        let mut groups: Vec<Vec<usize>> = g
            .chains
            .iter()
            .map(|c| {
                c.parts
                    .iter()
                    .filter_map(|n| g.parts.iter().position(|p| &p.name == n))
                    .collect()
            })
            .collect();
        for (i, p) in g.parts.iter().enumerate() {
            if p.mark == Mark::Atomic && !groups.iter().any(|gr| gr.contains(&i)) {
                groups.push(vec![i]);
            }
        }
        groups
    }
}

/// A pool: its members, their offers, its jobs, and its ledger: what it began with and every
/// change it was told, with the record of the loop's run on each.
#[derive(Clone, Debug)]
pub struct Pool {
    members: Vec<Member>,
    offers: Vec<Offer>,
    jobs: Vec<Live>,
    ledger: Ledger,
    now: f64,
    /// For each member with a device in the pool, the pool's time when it last held a part, or,
    /// if it has held none since its devices joined, when they did. Kept from the changes alone,
    /// so a replay keeps it the same.
    held: BTreeMap<String, f64>,
    /// In a trial of a wanted kind that names no backend, the device added for it: no pin to a
    /// backend is met on it, since the kind does not say which it has.
    unnamed_backend: Option<DeviceId>,
}

/// One candidate under construction: the jobs as they would be, and what was left out.
struct Built {
    jobs: Vec<Live>,
    moves: Vec<Move>,
    left_out: Vec<(String, Removal)>,
    no_copy: Option<Removal>,
}

impl Pool {
    /// A pool of these members and their devices' offers, holding no job, at time 0. Of two
    /// reports of one member the lower trust stands, and of two offers for one device the later,
    /// as when they join. Its ledger is a first one, written by this crate's [`VERSION`].
    pub fn new(members: Vec<Member>, offers: Vec<Offer>) -> Self {
        Pool::begin(members, offers, 0.0)
    }

    /// A pool begun from a description, as a pool file holds one: its members, offers and time.
    /// A description of a pool that holds jobs is refused.
    pub fn from_description(d: &Description) -> Result<Pool, HoldsJobs> {
        if !d.jobs.is_empty() {
            return Err(HoldsJobs {
                jobs: d.jobs.iter().map(|h| h.arrangement.job.clone()).collect(),
            });
        }
        Ok(Pool::begin(d.members.clone(), d.offers.clone(), d.at))
    }

    /// Of two offers for one device the later stands, as a join replaces the earlier offer.
    fn begin(members: Vec<Member>, given: Vec<Offer>, at: f64) -> Self {
        let mut offers: Vec<Offer> = Vec::new();
        for o in given {
            offers.retain(|x| x.device != o.device);
            offers.push(o);
        }
        offers.sort_by(|a, b| a.device.cmp(&b.device));
        let mut pool = Pool {
            members: Vec::new(),
            offers: offers.clone(),
            jobs: Vec::new(),
            ledger: Ledger {
                at,
                written_by: Some(VERSION.into()),
                follows: None,
                members: members.clone(),
                offers,
                entries: Vec::new(),
            },
            now: at,
            held: BTreeMap::new(),
            unnamed_backend: None,
        };
        for m in members {
            pool.add_member(m);
        }
        for m in pool.present() {
            pool.held.insert(m, at);
        }
        pool
    }

    /// The pool a ledger describes: begun with its time, members and offers, then told each
    /// change at its time, each record compared with the one logged. Its own ledger is then the
    /// same, its header too: which version wrote it and which ledger it follows are as logged.
    pub fn replay(ledger: &Ledger) -> Result<Pool, ReplayError> {
        let mut pool = Pool::begin(ledger.members.clone(), ledger.offers.clone(), ledger.at);
        pool.ledger.written_by.clone_from(&ledger.written_by);
        pool.ledger.follows.clone_from(&ledger.follows);
        for (entry, e) in ledger.entries.iter().enumerate() {
            match pool.apply(e.change.clone(), e.at) {
                Ok(record) if record == &e.record => {}
                Ok(_) => return Err(ReplayError::Differs { entry }),
                Err(error) => return Err(ReplayError::Refused { entry, error }),
            }
        }
        Ok(pool)
    }

    /// A pool that follows `old`, a ledger another version of this crate wrote, begun at `at` from
    /// its facts alone; its ledger's header names the ledger it follows, `follows`.
    ///
    /// A ledger replays only on the version that wrote it, since another version's loop may
    /// decide otherwise (`ReplayError::Differs`). So the old ledger's decisions are not taken:
    /// only what it was told. The new pool begins with the members as the old one knew them, each
    /// at the lowest trust and the shortest time to give a device back it was reported with, and
    /// the devices it held when the old ledger ends, every device that joined and has not left.
    /// Then each job it held, submitted and not ended, is submitted again at `at`, with its graph
    /// and its stamp, in the order it arrived, and the loop places it as on any arrival. Each
    /// submission is an entry of the new ledger, so that the new ledger replays by itself.
    ///
    /// A job submitted again starts waiting from 0, and is placed afresh: the ledger's words
    /// carry no job held where it was, nor its time waited, so a pool beginning from them could
    /// not be replayed from its own ledger. A job this version refuses, as a graph its check now
    /// refuses, is left out and returned, with why.
    pub fn follow(old: &Ledger, follows: &str, at: f64) -> (Pool, Vec<(String, ChangeError)>) {
        let mut members: Vec<Member> = Vec::new();
        let mut report = |m: &Member| match members.iter_mut().find(|x| x.name == m.name) {
            Some(known) => {
                known.trust = known.trust.min(m.trust);
                known.give_back_seconds = known.give_back_seconds.min(m.give_back_seconds);
            }
            None => members.push(m.clone()),
        };
        old.members.iter().for_each(&mut report);
        let mut offers: BTreeMap<DeviceId, Offer> = old
            .offers
            .iter()
            .map(|o| (o.device.clone(), o.clone()))
            .collect();
        let mut jobs: Vec<(Graph, Stamp)> = Vec::new();
        for e in &old.entries {
            match &e.change {
                Change::Join { member, offer } => {
                    if let Some(m) = member {
                        report(m);
                    }
                    offers.insert(offer.device.clone(), offer.clone());
                }
                Change::Leave { device } => {
                    offers.remove(device);
                }
                Change::Submit { graph, stamp } => jobs.push((graph.clone(), stamp.clone())),
                Change::End { job } => jobs.retain(|(g, _)| &g.job.name != job),
            }
        }
        let mut pool = Pool::begin(members, offers.into_values().collect(), at);
        pool.ledger.follows = Some(follows.into());
        let mut refused = Vec::new();
        for (graph, stamp) in jobs {
            let job = graph.job.name.clone();
            if let Err(e) = pool.apply(Change::Submit { graph, stamp }, at) {
                refused.push((job, e));
            }
        }
        (pool, refused)
    }

    /// What the pool began with, then every change it took, with the record of each. Its host
    /// writes each entry when the change is taken, before acknowledging it, and restores the pool
    /// with [`Pool::replay`]; the pool itself writes nothing.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Every run of the loop so far, in order.
    pub fn records(&self) -> impl ExactSizeIterator<Item = &Record> {
        self.ledger.entries.iter().map(|e| &e.record)
    }

    /// Where a job's parts are, if the pool holds it.
    pub fn arrangement(&self, job: &str) -> Option<&Arrangement> {
        self.jobs
            .iter()
            .find(|j| j.graph.job.name == job)
            .map(|j| &j.arrangement)
    }

    /// The pool as an author sees it: its members and offers, where each job is and how long each
    /// waiting one has waited, at the pool's time, and the file formats it reads.
    pub fn describe(&self) -> Description {
        Description {
            at: self.now,
            members: self.members.clone(),
            offers: self.offers.clone(),
            jobs: self
                .jobs
                .iter()
                .map(|j| Held {
                    arrangement: j.arrangement.clone(),
                    priority: j.priority(),
                    waited_seconds: j.waited,
                })
                .collect(),
            formats: [
                (GRAPH_FORMAT, GRAPH_VERSION),
                (LEDGER_FORMAT, LEDGER_VERSION),
                (POOL_FORMAT, POOL_VERSION),
            ]
            .into_iter()
            .flat_map(|(name, newest)| {
                (1..=newest).map(move |version| Format {
                    name: name.into(),
                    version,
                })
            })
            .collect(),
        }
    }

    /// The record [`Pool::submit`] would make of this graph and stamp at `at` seconds, or its
    /// refusal. The pool is left as it is: it holds no new job, keeps no record and does not move
    /// its time.
    pub fn plan(&self, graph: &Graph, stamp: &Stamp, at: f64) -> Result<Record, ChangeError> {
        let mut trial = Pool {
            members: self.members.clone(),
            offers: self.offers.clone(),
            jobs: self.jobs.clone(),
            ledger: Ledger::default(),
            now: self.now,
            held: self.held.clone(),
            unnamed_backend: None,
        };
        trial.submit(graph.clone(), stamp.clone(), at).cloned()
    }

    /// What the pool asks of its host at `at` seconds: the kind of device it most wants
    /// ([`Pool::wants`]), and the members it would give back after `idle_seconds`
    /// ([`Pool::gives_back`]), never `coordinator`. The host polls it and decides; the pool is
    /// left as it is.
    pub fn asks(&self, coordinator: &str, idle_seconds: f64, at: f64) -> Asks {
        Asks {
            wants: self.wants(at),
            gives_back: self.gives_back(coordinator, idle_seconds, at),
        }
    }

    /// The kind of device the pool most wants at `at` seconds, while jobs wait: of the kinds that
    /// would take a waiting job's parts, the one whose device, joining now, the loop would place
    /// parts on for the greatest gain in score. `None` when no job waits, or when no such device
    /// would take any. The pool is left as it is.
    ///
    /// **The kinds tried.** For each waiting job, one kind for each of its groups (a chain's
    /// parts, or an atomic part outside a chain), one for each replicable part, as one copy, and
    /// one for the whole job on one device. A kind for some parts asks what the loop's
    /// constraints ask of a device for them all: their memory together; the backend a part is
    /// pinned to, if one is; the highest level any of them is kept at, as its member's trust,
    /// left out when it is `outside`; their tolerance classes; and the dtypes of the edges into
    /// and out of them, opaque bytes aside. A pin to a member, a stamp's devices and a rate the
    /// job needs are not a kind's to name: no new device meets the first two, and the third is
    /// for the loop's estimate of its time to settle.
    ///
    /// **The gain.** Each kind is tried as one device of exactly that kind joining the pool at
    /// `at`: its member trusted at the kind's trust (else `outside`), its memory, tolerance
    /// classes and dtypes the kind's and no more, no pin to a backend met on it unless the kind
    /// names that backend, and every cost on it estimated. The loop then runs on the waiting
    /// jobs as on a join, their priorities grown by their wait until `at`. The gain is the
    /// chosen candidate's score less the score of leaving things as they are; a kind whose
    /// device the chosen candidate puts nothing on, or for which nothing is chosen, gains
    /// nothing.
    ///
    /// **The order, and ties.** The greatest gain wins. Of equal gains, the smaller kind: less
    /// memory, then a lower trust, then no backend before one named, then the backends, the
    /// tolerance classes and the dtypes in the order `mag-by` declares them. A host that adds
    /// the device asked for, and polls again, is asked for the next.
    pub fn wants(&self, at: f64) -> Option<DeviceKind> {
        let mut kinds: Vec<DeviceKind> = Vec::new();
        for j in self.jobs.iter().filter(|j| j.waiting()) {
            let mut sets = j.groups();
            sets.extend(
                (0..j.graph.parts.len())
                    .filter(|&p| j.graph.parts[p].mark == Mark::Replicable)
                    .map(|p| vec![p]),
            );
            sets.push((0..j.graph.parts.len()).collect());
            for set in sets {
                let kind = kind_for(j, &set);
                if !kinds.contains(&kind) {
                    kinds.push(kind);
                }
            }
        }
        kinds.sort_by_key(kind_order);
        let mut best: Option<(DeviceKind, f64)> = None;
        for kind in kinds {
            if let Some(gain) = self.gain(&kind, at)
                && best.as_ref().is_none_or(|(_, top)| gain > *top)
            {
                best = Some((kind, gain));
            }
        }
        best.map(|(kind, _)| kind)
    }

    /// Which device the pool gives up when its host takes one back by need: of the devices of
    /// `kind`, the one whose loss costs the pool least, at `at` seconds. The host wants it within
    /// `deadline_seconds`. The pool is left as it is: the leave is the host's to tell it, as
    /// [`Pool::leave`], once the device's member has been told.
    ///
    /// **The devices weighed.** Each device `kind` admits ([`DeviceKind::admits`]), its member
    /// trusted as the guard takes it (`outside` for a member the pool was never told of), whose
    /// member gives a device back within the deadline: its `give_back_seconds` at most
    /// `deadline_seconds`. A device of a member the pool was never told of has no time to give
    /// back the pool knows, and meets no deadline. When none is left, nothing the pool holds
    /// meets the need, and [`Unmet`] lists the devices of the kind whose members are too slow.
    ///
    /// **The loss.** Each device is tried as leaving at `at`: the jobs with a part on it wait,
    /// and the loop runs as on a leave, placing them again where they fit or leaving them
    /// waiting. The loss is the score of the jobs as they are less the score of the jobs as the
    /// loop leaves them, both scored as the loop scores, with every job's priority grown by its
    /// wait until `at`, and every job's rate over its best rate alone on the pool as it is, with
    /// the device: one scale for every device weighed, so that losing the fastest device is not
    /// made cheap by the slower best rate the pool would have without it.
    ///
    /// **The order, and ties.** The coordinator's own devices come last: of the devices that meet
    /// the need, one of `coordinator`'s is given up only when no other member's is left. The host
    /// takes the coordinator's machine back last, and a device of it given up costs the pool
    /// its placements as any other does, while a need refused would leave the host only SIGTERM
    /// on the coordinator, which stops the door and every placement until a coordinator starts
    /// again. Then the least loss; of equal losses the device that offers the least memory, so
    /// that the pool keeps its larger devices for the jobs to come; then the devices in the order
    /// of their ids.
    pub fn give_up(
        &self,
        coordinator: &str,
        kind: &DeviceKind,
        deadline_seconds: f64,
        at: f64,
    ) -> Result<GiveUp, Unmet> {
        let in_time = |o: &&Offer| {
            self.member(&o.device)
                .is_some_and(|m| m.give_back_seconds <= deadline_seconds)
        };
        let (meet, slow): (Vec<&Offer>, Vec<&Offer>) = self
            .offers
            .iter()
            .filter(|o| kind.admits(o, self.trust(&o.device)))
            .partition(in_time);
        if meet.is_empty() {
            return Err(Unmet {
                slow: slow.into_iter().map(|o| o.device.clone()).collect(),
            });
        }
        let mut now = self.without_ledger();
        now.advance(at);
        let (before, _) = now.score(&now.jobs);
        let mut ranked: Vec<(&Offer, f64)> = meet
            .into_iter()
            .map(|o| {
                let mut after = now.without_ledger();
                after.remove_device(&o.device, at);
                let mut jobs = now.jobs.clone();
                for (j, moved) in jobs.iter_mut().zip(after.jobs) {
                    j.arrangement = moved.arrangement;
                }
                (o, before - now.score(&jobs).0)
            })
            .collect();
        let last = |o: &Offer| o.device.member == coordinator;
        ranked.sort_by(|(a, la), (b, lb)| {
            last(a)
                .cmp(&last(b))
                .then(la.total_cmp(lb))
                .then(a.memory_bytes.cmp(&b.memory_bytes))
                .then(a.device.cmp(&b.device))
        });
        let ranked: Vec<(DeviceId, f64)> = ranked
            .into_iter()
            .map(|(o, loss)| (o.device.clone(), loss))
            .collect();
        Ok(GiveUp {
            device: ranked[0].0.clone(),
            loss: ranked[0].1,
            ranked,
        })
    }

    /// The members the pool would give back at `at` seconds: each member with a device in the
    /// pool, other than `coordinator`, that holds no part now and has held none for at least
    /// `idle_seconds`, counted from when it last held one, or from when its devices joined if it
    /// has held none since. Longest idle first, then by name. The time is the pool's if `at` is
    /// earlier. The pool is left as it is.
    pub fn gives_back(&self, coordinator: &str, idle_seconds: f64, at: f64) -> Vec<String> {
        let now = self.now.max(at);
        let holding = self.holders();
        let mut idle: Vec<(&String, f64)> = self
            .held
            .iter()
            .filter(|(m, _)| m.as_str() != coordinator && !holding.contains(*m))
            .filter(|&(_, &since)| now - since >= idle_seconds)
            .map(|(m, &since)| (m, since))
            .collect();
        idle.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(b.0)));
        idle.into_iter().map(|(m, _)| m.clone()).collect()
    }

    /// The one way into the pool: it takes a change at `at` seconds, runs the loop, and adds the
    /// change with the loop's record to its ledger. A change it refuses leaves it as it was and is
    /// not logged.
    ///
    /// A time earlier than the pool's is taken as it is told, and logged as given: the pool's time
    /// does not go back, and no job's wait grows or shrinks for it. Hosts deliver changes in the
    /// order they happened, and a change reported a little late is still a change; refusing it
    /// would lose a job's end or a device's leaving. A replay takes the same path.
    pub fn apply(&mut self, change: Change, at: f64) -> Result<&Record, ChangeError> {
        if !at.is_finite() {
            return Err(ChangeError::Time);
        }
        let (holding, present) = (self.holders(), self.present());
        let record = match &change {
            Change::Submit { graph, stamp } => self.arrive(graph.clone(), stamp.clone(), at)?,
            Change::End { job } => self.finish(job, at),
            Change::Join { member, offer } => self.add_device(member.clone(), offer.clone(), at),
            Change::Leave { device } => self.remove_device(device, at),
        };
        // A member that held a part held it until now: one that holds a part after this change is
        // among those holding before the next, and is not idle meanwhile. A member whose first
        // device joins starts its idle time now, and one with no device left is no longer kept.
        let now_present = self.present();
        for m in holding {
            self.held.insert(m, self.now);
        }
        for m in now_present.difference(&present) {
            self.held.insert(m.clone(), self.now);
        }
        self.held.retain(|m, _| now_present.contains(m));
        self.ledger.entries.push(Entry { at, change, record });
        Ok(&self.ledger.entries.last().expect("just pushed").record)
    }

    /// A job arrives at `at` seconds, with the host's stamp: it is placed only on the devices the
    /// stamp names, if it names any, and its priority is held to the stamp's ceiling, if it has
    /// one. A graph that fails [`Graph::check`], a stamp with a ceiling of 0 or an empty list of
    /// devices, or a second job of a name the pool holds, is refused and the pool is unchanged.
    pub fn submit(&mut self, graph: Graph, stamp: Stamp, at: f64) -> Result<&Record, ChangeError> {
        self.apply(Change::Submit { graph, stamp }, at)
    }

    /// A job ends at `at` seconds; the jobs waiting are placed again where they now fit.
    pub fn end(&mut self, job: &str, at: f64) -> Result<&Record, ChangeError> {
        self.apply(Change::End { job: job.into() }, at)
    }

    /// A device joins at `at` seconds, with its member's report. Of two reports for one member
    /// the stricter stands: the lower trust, the shorter time to give a device back.
    pub fn join(
        &mut self,
        member: Option<Member>,
        offer: Offer,
        at: f64,
    ) -> Result<&Record, ChangeError> {
        self.apply(Change::Join { member, offer }, at)
    }

    /// A device leaves at `at` seconds: every job with a part on it waits, and is placed again
    /// where it fits.
    pub fn leave(&mut self, device: &DeviceId, at: f64) -> Result<&Record, ChangeError> {
        self.apply(
            Change::Leave {
                device: device.clone(),
            },
            at,
        )
    }

    fn arrive(&mut self, graph: Graph, stamp: Stamp, at: f64) -> Result<Record, ChangeError> {
        let keeps = graph.keeps().map_err(ChangeError::Graph)?;
        if stamp.priority_ceiling == Some(0) {
            return Err(ChangeError::ZeroCeiling);
        }
        if stamp.devices.as_ref().is_some_and(Vec::is_empty) {
            return Err(ChangeError::NoDevices);
        }
        let name = graph.job.name.clone();
        if self.jobs.iter().any(|j| j.graph.job.name == name) {
            return Err(ChangeError::Duplicate { job: name });
        }
        let parts = graph.parts.len();
        let mut live = Live {
            graph,
            stamp,
            keeps,
            arrangement: Arrangement {
                job: name.clone(),
                places: vec![Place::Waiting; parts],
            },
            waited: 0.0,
            best_alone: 0.0,
        };
        live.best_alone = self.best_alone(&live);
        self.advance(at);
        self.jobs.push(live);
        Ok(self.run(Event::Arrival { job: name }, Vec::new()))
    }

    fn finish(&mut self, job: &str, at: f64) -> Record {
        self.advance(at);
        self.jobs.retain(|j| j.graph.job.name != job);
        self.run(Event::End { job: job.into() }, Vec::new())
    }

    fn add_device(&mut self, member: Option<Member>, offer: Offer, at: f64) -> Record {
        self.advance(at);
        let lowered = member.map(|m| self.add_member(m)).unwrap_or_default();
        let device = offer.device.clone();
        self.offers.retain(|o| o.device != device);
        self.offers.push(offer);
        self.offers.sort_by(|a, b| a.device.cmp(&b.device));
        self.rebase();
        self.run(Event::Join { device }, lowered)
    }

    fn remove_device(&mut self, device: &DeviceId, at: f64) -> Record {
        self.advance(at);
        self.offers.retain(|o| &o.device != device);
        for j in &mut self.jobs {
            let on_it = j.arrangement.places.iter().any(|p| match p {
                Place::On(d) => d == device,
                Place::Copies(ds) => ds.contains(device),
                Place::Waiting => false,
            });
            if on_it {
                j.arrangement
                    .places
                    .iter_mut()
                    .for_each(|p| *p = Place::Waiting);
            }
        }
        self.rebase();
        self.run(
            Event::Leave {
                device: device.clone(),
            },
            Vec::new(),
        )
    }

    /// Records a member's report; returns its name if it lowers the trust the member was known
    /// at.
    fn add_member(&mut self, m: Member) -> Vec<String> {
        match self.members.iter_mut().find(|x| x.name == m.name) {
            Some(known) => {
                let lowered = m.trust < known.trust;
                known.trust = known.trust.min(m.trust);
                known.give_back_seconds = known.give_back_seconds.min(m.give_back_seconds);
                if lowered { vec![m.name] } else { Vec::new() }
            }
            None => {
                self.members.push(m);
                Vec::new()
            }
        }
    }

    /// The members with a device in the pool.
    fn present(&self) -> BTreeSet<String> {
        self.offers
            .iter()
            .map(|o| o.device.member.clone())
            .collect()
    }

    /// The members a part of a job is placed on now.
    fn holders(&self) -> BTreeSet<String> {
        self.jobs
            .iter()
            .flat_map(|j| j.arrangement.places.iter().flat_map(devices_of))
            .map(|d| d.member.clone())
            .collect()
    }

    /// The gain in score of a device of `kind` joining at `at`, if the loop would place a part
    /// on it: see [`Pool::wants`].
    fn gain(&self, kind: &DeviceKind, at: f64) -> Option<f64> {
        let device = DeviceId {
            member: WANTED.into(),
            index: 0,
        };
        let mut trial = Pool {
            members: self.members.clone(),
            offers: self.offers.clone(),
            jobs: self.jobs.clone(),
            ledger: Ledger::default(),
            now: self.now,
            held: BTreeMap::new(),
            unnamed_backend: kind.backend.is_none().then(|| device.clone()),
        };
        trial.advance(at);
        trial.members.push(Member {
            name: WANTED.into(),
            trust: kind.trust.unwrap_or(Trust::Outside),
            give_back_seconds: 0.0,
        });
        trial.offers.push(Offer {
            device: device.clone(),
            backend: kind.backend.unwrap_or(Backend::Flex),
            memory_bytes: kind.memory_bytes.unwrap_or(0),
            tolerances: kind.tolerances.clone(),
            dtypes: kind.dtypes.clone(),
            costs: Vec::new(),
        });
        trial.rebase();
        let (current, _) = trial.score(&trial.jobs);
        let record = trial.run(Event::Join { device }, Vec::new());
        let chosen = &record.candidates[record.chosen?];
        let on_it = chosen
            .arrangements
            .iter()
            .flat_map(|a| a.places.iter().flat_map(devices_of))
            .any(|d| d.member == WANTED);
        on_it.then(|| chosen.score.expect("a chosen candidate is scored") - current)
    }

    /// Each job's best rate alone, again, after the pool's devices change: a job that no device
    /// could take when it arrived has none until one joins.
    fn rebase(&mut self) {
        let best: Vec<f64> = self.jobs.iter().map(|j| self.best_alone(j)).collect();
        for (j, b) in self.jobs.iter_mut().zip(best) {
            j.best_alone = b;
        }
    }

    fn advance(&mut self, at: f64) {
        let dt = (at - self.now).max(0.0);
        for j in &mut self.jobs {
            if j.waiting() {
                j.waited += dt;
            }
        }
        self.now = self.now.max(at);
    }

    /// One run of the loop over the waiting jobs, which every event here touches.
    fn run(&mut self, event: Event, lowered: Vec<String>) -> Record {
        let touched: Vec<usize> = (0..self.jobs.len())
            .filter(|&i| self.jobs[i].waiting())
            .collect();
        let devices = self.offers.len();
        // Counted, not built: a job of g groups has devices^g placements, and each job may wait.
        let count = touched
            .iter()
            .map(|&i| (devices as f64).powi(self.jobs[i].groups().len() as i32) + 1.0)
            .product::<f64>();
        let greedy = count > EXHAUSTIVE as f64;
        let built: Vec<Built> = if greedy {
            vec![self.greedy(&touched)]
        } else {
            self.every_candidate(&touched)
        };
        let mut candidates = Vec::new();
        let mut best: Option<(usize, f64)> = None;
        for b in built {
            let arrangements = touched
                .iter()
                .map(|&i| b.jobs[i].arrangement.clone())
                .collect();
            let removed = b.no_copy.clone().or_else(|| self.removal(&b.jobs));
            let (score, band) = match removed {
                Some(_) => (None, None),
                None => {
                    let (s, band) = self.score(&b.jobs);
                    (Some(s), Some(band))
                }
            };
            if let Some(s) = score
                && best.is_none_or(|(_, top)| s > top)
            {
                best = Some((candidates.len(), s));
            }
            candidates.push(Candidate {
                arrangements,
                moves: b.moves,
                removed,
                left_out: b.left_out,
                score,
                band,
            });
        }
        // Leaving things as they are scores what the waiting jobs already have: nothing.
        let (current, _) = self.score(&self.jobs);
        let chosen = best.filter(|&(_, s)| s > current).map(|(i, _)| i);
        if let Some(c) = chosen {
            for (k, &j) in touched.iter().enumerate() {
                let placed = candidates[c].arrangements[k].clone();
                if !placed.places.iter().any(|p| matches!(p, Place::Waiting)) {
                    self.jobs[j].waited = 0.0;
                }
                self.jobs[j].arrangement = placed;
            }
        }
        Record {
            event,
            candidates,
            chosen,
            greedy,
            lowered,
        }
    }

    /// Every combination of the touched jobs' choices, waiting first, devices in order.
    fn every_candidate(&self, touched: &[usize]) -> Vec<Built> {
        // Each job's choices: None to wait, or a device per group.
        let mut combos: Vec<Vec<Option<Vec<usize>>>> = vec![Vec::new()];
        for &i in touched {
            let groups = self.jobs[i].groups().len();
            let mut next = Vec::new();
            for prefix in &combos {
                let mut wait = prefix.clone();
                wait.push(None);
                next.push(wait);
                for devices in device_tuples(self.offers.len(), groups) {
                    let mut p = prefix.clone();
                    p.push(Some(devices));
                    next.push(p);
                }
            }
            combos = next;
        }
        combos
            .into_iter()
            .map(|combo| {
                let mut built = Built {
                    jobs: self.jobs.clone(),
                    moves: Vec::new(),
                    left_out: Vec::new(),
                    no_copy: None,
                };
                for (k, &i) in touched.iter().enumerate() {
                    match &combo[k] {
                        None => built.moves.push(Move::Wait),
                        Some(devices) => {
                            built.moves.push(Move::Fit);
                            self.place(&mut built, i, devices);
                        }
                    }
                }
                built
            })
            .collect()
    }

    /// Places job `i` in `built`: group `g` on device `devices[g]`, then each replicable part's
    /// copies where they may go and fit.
    fn place(&self, built: &mut Built, i: usize, devices: &[usize]) {
        let groups = built.jobs[i].groups();
        for (g, group) in groups.iter().enumerate() {
            let d = &self.offers[devices[g]].device;
            for &p in group {
                built.jobs[i].arrangement.places[p] = Place::On(d.clone());
            }
        }
        let parts: Vec<usize> = (0..built.jobs[i].graph.parts.len())
            .filter(|&p| built.jobs[i].graph.parts[p].mark == Mark::Replicable)
            .collect();
        for p in parts {
            let mut copies = Vec::new();
            for o in &self.offers {
                let part = &built.jobs[i].graph.parts[p];
                let name = part.name.clone();
                if let Some(r) = self.eligibility(&built.jobs[i], p, &o.device) {
                    built.left_out.push((name, r));
                    continue;
                }
                let used = memory_used(&built.jobs, &o.device);
                if used + part.memory_bytes > o.memory_bytes {
                    built.left_out.push((
                        name,
                        Removal::Memory {
                            device: o.device.clone(),
                        },
                    ));
                    continue;
                }
                copies.push(o.device.clone());
                built.jobs[i].arrangement.places[p] = Place::Copies(copies.clone());
            }
            if copies.is_empty() {
                built.jobs[i].arrangement.places[p] = Place::Waiting;
                if built.no_copy.is_none() {
                    built.no_copy = built
                        .left_out
                        .iter()
                        .rev()
                        .find(|(n, _)| n == &built.jobs[i].graph.parts[p].name)
                        .map(|(_, r)| r.clone())
                        .or(Some(Removal::NoDevice {
                            device: DeviceId {
                                member: String::new(),
                                index: 0,
                            },
                        }));
                }
            }
        }
    }

    /// Past the bound: each touched job in turn, highest priority first, each group on the
    /// eligible device with room that adds least to the busiest device's time; a job with a
    /// group or a replicable part that fits nowhere waits.
    fn greedy(&self, touched: &[usize]) -> Built {
        let mut order: Vec<usize> = touched.to_vec();
        order.sort_by(|&a, &b| {
            let (pa, pb) = (self.priority(&self.jobs[a]), self.priority(&self.jobs[b]));
            pb.total_cmp(&pa).then(a.cmp(&b))
        });
        let mut built = Built {
            jobs: self.jobs.clone(),
            moves: vec![Move::Wait; touched.len()],
            left_out: Vec::new(),
            no_copy: None,
        };
        for i in order {
            let k = touched.iter().position(|&t| t == i).expect("touched");
            let mut trial = Built {
                jobs: built.jobs.clone(),
                moves: Vec::new(),
                left_out: Vec::new(),
                no_copy: None,
            };
            let mut devices = Vec::new();
            for group in trial.jobs[i].groups() {
                let mut best: Option<(usize, f64)> = None;
                for (d, o) in self.offers.iter().enumerate() {
                    let fits = group
                        .iter()
                        .all(|&p| self.eligibility(&trial.jobs[i], p, &o.device).is_none())
                        && memory_used(&trial.jobs, &o.device)
                            + group
                                .iter()
                                .map(|&p| trial.jobs[i].graph.parts[p].memory_bytes)
                                .sum::<u64>()
                            <= o.memory_bytes;
                    if !fits {
                        continue;
                    }
                    let mut with = trial.jobs.clone();
                    for &p in &group {
                        with[i].arrangement.places[p] = Place::On(o.device.clone());
                    }
                    let busiest = self.busiest(&with);
                    if best.is_none_or(|(_, b)| busiest < b) {
                        best = Some((d, busiest));
                    }
                }
                let Some((d, _)) = best else {
                    devices.clear();
                    break;
                };
                for &p in &group {
                    trial.jobs[i].arrangement.places[p] = Place::On(self.offers[d].device.clone());
                }
                devices.push(d);
            }
            if devices.len() != trial.jobs[i].groups().len() {
                continue;
            }
            self.place(&mut trial, i, &devices);
            if trial.no_copy.is_none() && self.removal(&trial.jobs).is_none() {
                built.jobs = trial.jobs;
                built.left_out.extend(trial.left_out);
                built.moves[k] = Move::Fit;
            }
        }
        built
    }

    /// The busiest device's time per step over every placed part.
    fn busiest(&self, jobs: &[Live]) -> f64 {
        self.loads(jobs).values().copied().fold(0.0, f64::max)
    }

    fn offer(&self, d: &DeviceId) -> Option<&Offer> {
        self.offers.iter().find(|o| &o.device == d)
    }

    fn member(&self, d: &DeviceId) -> Option<&Member> {
        self.members.iter().find(|m| m.name == d.member)
    }

    /// The trust of a device's member, as the guard takes it: `outside` for a member the pool
    /// does not know.
    fn trust(&self, d: &DeviceId) -> Trust {
        self.member(d).map_or(Trust::Outside, |m| m.trust)
    }

    /// The pool as it is, with an empty ledger: for a trial that is not kept.
    fn without_ledger(&self) -> Pool {
        Pool {
            members: self.members.clone(),
            offers: self.offers.clone(),
            jobs: self.jobs.clone(),
            ledger: Ledger::default(),
            now: self.now,
            held: self.held.clone(),
            unnamed_backend: self.unnamed_backend.clone(),
        }
    }

    /// A part kind's cost on a device: measured, or else estimated as the slowest any other
    /// device measured, or else one second, marked as an estimate either way.
    fn cost(&self, offer: &Offer, unit: &Unit) -> Cost {
        if let Some(c) = offer.costs.iter().find(|c| &c.unit == unit) {
            return c.clone();
        }
        let elsewhere = self
            .offers
            .iter()
            .flat_map(|o| o.costs.iter())
            .filter(|c| &c.unit == unit && !c.estimate)
            .map(|c| c.seconds_per_step)
            .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.max(t))));
        Cost {
            unit: unit.clone(),
            seconds_per_step: elsewhere.unwrap_or(1.0),
            band: ESTIMATE_BAND,
            estimate: true,
        }
    }

    /// Why a part of a job may not go on a device, if it may not: no such device, the guard, the
    /// job's stamp, a pin, its tolerance class, or a dtype its edges carry.
    fn eligibility(&self, j: &Live, i: usize, d: &DeviceId) -> Option<Removal> {
        let g = &j.graph;
        let part = &g.parts[i];
        let (name, device) = (part.name.clone(), d.clone());
        let Some(offer) = self.offer(d) else {
            return Some(Removal::NoDevice { device });
        };
        // A device whose member the pool does not know is trusted at the lowest level.
        let trust = self.trust(d);
        let keep = j.keeps.parts[i];
        if keep > trust {
            return Some(Removal::Guard {
                part: name,
                device,
                keep,
                trust,
            });
        }
        if let Some(allowed) = &j.stamp.devices
            && !allowed.contains(d)
        {
            return Some(Removal::Stamp {
                job: g.job.name.clone(),
                device,
            });
        }
        let pinned_elsewhere = match &part.pin {
            Some(Pin::Member(m)) => &d.member != m,
            Some(Pin::Backend(b)) => {
                &offer.backend != b || self.unnamed_backend.as_ref() == Some(d)
            }
            None => false,
        };
        if pinned_elsewhere {
            return Some(Removal::Pin { part: name, device });
        }
        if !offer.tolerances.contains(&part.tolerance) {
            return Some(Removal::Tolerance { part: name, device });
        }
        let dtypes_met = g
            .edges
            .iter()
            .filter(|e| e.from == part.name || e.to.as_deref() == Some(&part.name))
            .all(|e| e.dtype == Dtype::Bytes || offer.dtypes.contains(&e.dtype));
        if !dtypes_met {
            return Some(Removal::Dtype { part: name, device });
        }
        None
    }

    /// The first constraint the jobs' places break, if any: every part's eligibility first,
    /// then each device's memory, then each placed job's need.
    fn removal(&self, jobs: &[Live]) -> Option<Removal> {
        for j in jobs {
            for (i, place) in j.arrangement.places.iter().enumerate() {
                for d in devices_of(place) {
                    if let Some(r) = self.eligibility(j, i, d) {
                        return Some(r);
                    }
                }
            }
        }
        for o in &self.offers {
            if memory_used(jobs, &o.device) > o.memory_bytes {
                return Some(Removal::Memory {
                    device: o.device.clone(),
                });
            }
        }
        let rates = self.rates(jobs);
        for j in jobs {
            if let Some(need) = j.graph.job.need.rate_per_second
                && !j.waiting()
                && rates[j.graph.job.name.as_str()].0 < need
            {
                return Some(Removal::Need {
                    job: j.graph.job.name.clone(),
                });
            }
        }
        None
    }

    /// Each device's time per step, over every placed part of every job on it, and each job's
    /// widest band among the costs it used.
    fn loads(&self, jobs: &[Live]) -> BTreeMap<DeviceId, f64> {
        self.loads_and_bands(jobs).0
    }

    fn loads_and_bands<'a>(
        &self,
        jobs: &'a [Live],
    ) -> (BTreeMap<DeviceId, f64>, BTreeMap<&'a str, f64>) {
        let mut load: BTreeMap<DeviceId, f64> = BTreeMap::new();
        let mut band: BTreeMap<&str, f64> = BTreeMap::new();
        for j in jobs {
            let name = j.graph.job.name.as_str();
            band.insert(name, 0.0);
            for (i, place) in j.arrangement.places.iter().enumerate() {
                let unit = &j.graph.parts[i].unit;
                let costs: Vec<(&DeviceId, Cost)> = devices_of(place)
                    .into_iter()
                    .filter_map(|d| self.offer(d).map(|o| (d, self.cost(o, unit))))
                    .collect();
                // Copies share the items in proportion to their speed.
                let speed: f64 = costs.iter().map(|(_, c)| 1.0 / c.seconds_per_step).sum();
                for (d, c) in &costs {
                    let share = (1.0 / c.seconds_per_step) / speed;
                    *load.entry((*d).clone()).or_insert(0.0) += share * c.seconds_per_step;
                    let b = band.get_mut(name).expect("inserted above");
                    *b = b.max(c.band);
                }
            }
        }
        (load, band)
    }

    /// Each placed job's rate in steps per second, with its band; a waiting job's is 0.
    fn rates<'a>(&self, jobs: &'a [Live]) -> BTreeMap<&'a str, (f64, f64)> {
        let (load, band) = self.loads_and_bands(jobs);
        jobs.iter()
            .map(|j| {
                let name = j.graph.job.name.as_str();
                if j.waiting() {
                    return (name, (0.0, 0.0));
                }
                let busiest = j
                    .arrangement
                    .places
                    .iter()
                    .flat_map(devices_of)
                    .map(|d| load.get(d).copied().unwrap_or(0.0))
                    .fold(0.0, f64::max);
                let rate = if busiest > 0.0 { 1.0 / busiest } else { 0.0 };
                (name, (rate, band[name]))
            })
            .collect()
    }

    fn priority(&self, j: &Live) -> f64 {
        j.priority() as f64 * (1.0 + j.waited / AGING_SECONDS)
    }

    /// The candidate's score and its band: the sum over jobs of priority times rate, each rate
    /// over the job's best alone.
    fn score(&self, jobs: &[Live]) -> (f64, f64) {
        let rates = self.rates(jobs);
        let mut score = 0.0;
        let mut band: f64 = 0.0;
        for j in jobs {
            let (rate, b) = rates[j.graph.job.name.as_str()];
            if j.best_alone > 0.0 {
                score += self.priority(j) * rate / j.best_alone;
            }
            band = band.max(b);
        }
        (score, band)
    }

    /// The job's best rate alone on the pool as it now is, among the arrangements this step
    /// lists for it (greedily past the bound).
    fn best_alone(&self, job: &Live) -> f64 {
        let mut alone = Pool {
            members: self.members.clone(),
            offers: self.offers.clone(),
            jobs: vec![job.clone()],
            ledger: Ledger::default(),
            now: 0.0,
            held: BTreeMap::new(),
            unnamed_backend: None,
        };
        alone.jobs[0]
            .arrangement
            .places
            .iter_mut()
            .for_each(|p| *p = Place::Waiting);
        let groups = alone.jobs[0].groups().len();
        let built = if (alone.offers.len() as f64).powi(groups as i32) > EXHAUSTIVE as f64 {
            vec![alone.greedy(&[0])]
        } else {
            alone.every_candidate(&[0])
        };
        let name = job.graph.job.name.as_str();
        built
            .into_iter()
            .filter(|b| !b.jobs[0].waiting() && b.no_copy.is_none())
            .filter(|b| matches!(alone.removal(&b.jobs), None | Some(Removal::Need { .. })))
            .map(|b| alone.rates(&b.jobs)[name].0)
            .fold(0.0, f64::max)
    }
}

/// The kind of device that takes the parts `set` of job `j` together: see [`Pool::wants`].
fn kind_for(j: &Live, set: &[usize]) -> DeviceKind {
    let parts = &j.graph.parts;
    let backend = set.iter().find_map(|&p| match &parts[p].pin {
        Some(Pin::Backend(b)) => Some(*b),
        _ => None,
    });
    let trust = set
        .iter()
        .map(|&p| j.keeps.parts[p])
        .max()
        .filter(|&t| t > Trust::Outside);
    let mut tolerances: Vec<_> = set.iter().map(|&p| parts[p].tolerance).collect();
    tolerances.sort_by_key(|&t| t as u8);
    tolerances.dedup();
    let names: Vec<&str> = set.iter().map(|&p| parts[p].name.as_str()).collect();
    let mut dtypes: Vec<Dtype> = j
        .graph
        .edges
        .iter()
        .filter(|e| {
            names.contains(&e.from.as_str()) || e.to.as_deref().is_some_and(|t| names.contains(&t))
        })
        .map(|e| e.dtype)
        .filter(|&d| d != Dtype::Bytes)
        .collect();
    dtypes.sort_by_key(|&d| d as u8);
    dtypes.dedup();
    DeviceKind {
        backend,
        memory_bytes: Some(set.iter().map(|&p| parts[p].memory_bytes).sum()),
        trust,
        tolerances,
        dtypes,
    }
}

/// A kind's place in the order: its memory, trust, backend, tolerance classes and dtypes.
type KindKey = (Option<u64>, Option<Trust>, Option<u8>, Vec<u8>, Vec<u8>);

/// The order kinds are tried in, so that of equal gains the smaller wins: see [`Pool::wants`].
fn kind_order(k: &DeviceKind) -> KindKey {
    (
        k.memory_bytes,
        k.trust,
        k.backend.map(|b| b as u8),
        k.tolerances.iter().map(|&t| t as u8).collect(),
        k.dtypes.iter().map(|&d| d as u8).collect(),
    )
}

/// The devices a place uses.
fn devices_of(place: &Place) -> Vec<&DeviceId> {
    match place {
        Place::Waiting => Vec::new(),
        Place::On(d) => vec![d],
        Place::Copies(ds) => ds.iter().collect(),
    }
}

/// The memory the placed parts of `jobs` use on `device`.
fn memory_used(jobs: &[Live], device: &DeviceId) -> u64 {
    jobs.iter()
        .flat_map(|j| {
            j.arrangement
                .places
                .iter()
                .enumerate()
                .filter(|(_, p)| devices_of(p).contains(&device))
                .map(|(i, _)| j.graph.parts[i].memory_bytes)
        })
        .sum()
}

/// Every way to give each of `groups` groups one of `devices` devices, in order.
fn device_tuples(devices: usize, groups: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for _ in 0..groups {
        let mut next = Vec::new();
        for prefix in &out {
            for d in 0..devices {
                let mut p: Vec<usize> = prefix.clone();
                p.push(d);
                next.push(p);
            }
        }
        out = next;
    }
    if devices == 0 && groups > 0 {
        out.clear();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mag_by::Tolerance;

    #[test]
    fn kinds_are_tried_smallest_first_memory_then_trust_then_backend_then_classes() {
        let k = |memory: u64, trust: Option<Trust>, backend: Option<Backend>| DeviceKind {
            memory_bytes: Some(memory),
            trust,
            backend,
            ..DeviceKind::default()
        };
        let mut kinds = vec![
            k(2, None, None),
            k(1, Some(Trust::Owner), None),
            k(1, Some(Trust::Site), None),
            k(1, None, Some(Backend::Flex)),
            k(1, None, Some(Backend::Cuda)),
            k(1, None, None),
            DeviceKind {
                tolerances: vec![Tolerance::Bf16],
                ..k(1, None, None)
            },
            DeviceKind {
                tolerances: vec![Tolerance::Exact],
                ..k(1, None, None)
            },
            DeviceKind {
                dtypes: vec![Dtype::F64],
                ..k(1, None, None)
            },
        ];
        kinds.sort_by_key(kind_order);
        assert_eq!(
            kinds,
            [
                k(1, None, None),
                DeviceKind {
                    dtypes: vec![Dtype::F64],
                    ..k(1, None, None)
                },
                DeviceKind {
                    tolerances: vec![Tolerance::Exact],
                    ..k(1, None, None)
                },
                DeviceKind {
                    tolerances: vec![Tolerance::Bf16],
                    ..k(1, None, None)
                },
                k(1, None, Some(Backend::Cuda)),
                k(1, None, Some(Backend::Flex)),
                k(1, Some(Trust::Site), None),
                k(1, Some(Trust::Owner), None),
                k(2, None, None),
            ]
        );
    }

    #[test]
    fn a_pool_begins_with_each_member_with_a_device_idle_from_its_time() {
        let member = |name: &str| Member {
            name: name.into(),
            trust: Trust::Owner,
            give_back_seconds: 1.0,
        };
        let offer = Offer {
            device: DeviceId {
                member: "a".into(),
                index: 0,
            },
            backend: Backend::Flex,
            memory_bytes: 1,
            tolerances: vec![],
            dtypes: vec![],
            costs: vec![],
        };
        let pool = Pool::from_description(&Description {
            at: 7.0,
            members: vec![member("a"), member("none")],
            offers: vec![offer],
            jobs: vec![],
            formats: vec![],
        })
        .unwrap();
        assert_eq!(pool.held, BTreeMap::from([("a".to_string(), 7.0)]));
        assert!(pool.holders().is_empty());
    }
}
