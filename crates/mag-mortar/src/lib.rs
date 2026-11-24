//! The controller of a mag pool: the placement loop.
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
//!    eligibility over every part (no such device, the guard, a pin, its tolerance class, its
//!    edges' dtypes), then the memory each device offers, then a job's need, a rate it must
//!    sustain, for every placed job. The guard keeps a part carrying local-only data off a
//!    device of a member marked remote, and off a device of a member the pool does not know.
//! 3. **Predict** each remaining candidate's score: the sum over jobs of their priority times
//!    their rate, each rate divided by the job's best rate alone on the pool as it now is. A
//!    waiting job's priority is its own times `1 + waited / AGING_SECONDS`: 2 at 60 s, 3 at
//!    120 s. Once placed, its waiting time starts again from 0.
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
//! - a deadline: a job states no length yet.
//!
//! The same members, offers and events give the same records: devices are taken in the order of
//! their [`DeviceId`], and the first of equal scores wins.

use std::collections::BTreeMap;
use std::fmt;

use mag_by::{
    Arrangement, Candidate, Change, Cost, Description, DeviceId, Dtype, Entry, Event, Format,
    GRAPH_FORMAT, GRAPH_VERSION, Graph, Held, LEDGER_FORMAT, LEDGER_VERSION, Ledger, LocalOnly,
    Mark, Member, Move, Offer, POOL_FORMAT, POOL_VERSION, Pin, Place, Record, Refusal, Removal,
    Unit,
};

/// The most candidates listed one by one before generation turns greedy (proposed).
pub const EXHAUSTIVE: usize = 500;

/// A waiting job's priority grows by its own value every this many seconds (proposed).
pub const AGING_SECONDS: f64 = 60.0;

/// The band given to an estimated cost, relative: wide, so that it is probed and not trusted.
pub const ESTIMATE_BAND: f64 = 1.0;

/// Why the pool refuses a change. A refused change leaves the pool as it was, and is not in its
/// ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChangeError {
    /// The graph fails [`Graph::check`].
    Graph(Refusal),
    /// The pool already holds a job of this name.
    Duplicate { job: String },
    /// The time is not finite.
    Time,
}

impl fmt::Display for ChangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChangeError::Graph(r) => write!(f, "the graph is refused: {r}"),
            ChangeError::Duplicate { job } => {
                write!(f, "the pool already holds a job named `{job}`")
            }
            ChangeError::Time => write!(f, "the time is not finite"),
        }
    }
}

impl std::error::Error for ChangeError {}

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

/// A job the pool holds: its graph, its marks, where its parts are, and how long it has waited.
#[derive(Clone, Debug)]
struct Live {
    graph: Graph,
    marks: LocalOnly,
    arrangement: Arrangement,
    waited: f64,
    best_alone: f64,
}

impl Live {
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

/// A pool's controller: its members, their offers, its jobs, and its ledger: what it began with
/// and every change it was told, with the record of the loop's run on each.
#[derive(Clone, Debug)]
pub struct Pool {
    members: Vec<Member>,
    offers: Vec<Offer>,
    jobs: Vec<Live>,
    ledger: Ledger,
    now: f64,
}

/// One candidate under construction: the jobs as they would be, and what was left out.
struct Built {
    jobs: Vec<Live>,
    moves: Vec<Move>,
    left_out: Vec<(String, Removal)>,
    no_copy: Option<Removal>,
}

impl Pool {
    /// A pool of these members and their devices' offers, holding no job, at time 0.
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

    fn begin(members: Vec<Member>, mut offers: Vec<Offer>, at: f64) -> Self {
        offers.sort_by(|a, b| a.device.cmp(&b.device));
        let mut pool = Pool {
            members: Vec::new(),
            offers: offers.clone(),
            jobs: Vec::new(),
            ledger: Ledger {
                at,
                members: members.clone(),
                offers,
                entries: Vec::new(),
            },
            now: at,
        };
        for m in members {
            pool.add_member(m);
        }
        pool
    }

    /// The pool a ledger describes: begun with its time, members and offers, then told each
    /// change at its time, each record compared with the one logged. Its own ledger is then the
    /// same.
    pub fn replay(ledger: &Ledger) -> Result<Pool, ReplayError> {
        let mut pool = Pool::begin(ledger.members.clone(), ledger.offers.clone(), ledger.at);
        for (entry, e) in ledger.entries.iter().enumerate() {
            match pool.apply(e.change.clone(), e.at) {
                Ok(record) if record == &e.record => {}
                Ok(_) => return Err(ReplayError::Differs { entry }),
                Err(error) => return Err(ReplayError::Refused { entry, error }),
            }
        }
        Ok(pool)
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
                    priority: j.graph.job.priority,
                    waited_seconds: j.waited,
                })
                .collect(),
            formats: vec![
                Format {
                    name: GRAPH_FORMAT.into(),
                    version: GRAPH_VERSION,
                },
                Format {
                    name: LEDGER_FORMAT.into(),
                    version: LEDGER_VERSION,
                },
                Format {
                    name: POOL_FORMAT.into(),
                    version: POOL_VERSION,
                },
            ],
        }
    }

    /// The record [`Pool::submit`] would make of this graph at `at` seconds, or its refusal. The
    /// pool is left as it is: it holds no new job, keeps no record and does not move its time.
    pub fn plan(&self, graph: &Graph, at: f64) -> Result<Record, ChangeError> {
        let mut trial = Pool {
            members: self.members.clone(),
            offers: self.offers.clone(),
            jobs: self.jobs.clone(),
            ledger: Ledger::default(),
            now: self.now,
        };
        trial.submit(graph.clone(), at).cloned()
    }

    /// The one way into the pool: it takes a change at `at` seconds, runs the loop, and adds the
    /// change with the loop's record to its ledger. A change it refuses leaves it as it was and is
    /// not logged.
    pub fn apply(&mut self, change: Change, at: f64) -> Result<&Record, ChangeError> {
        if !at.is_finite() {
            return Err(ChangeError::Time);
        }
        let record = match &change {
            Change::Submit { graph } => self.arrive(graph.clone(), at)?,
            Change::End { job } => self.finish(job, at),
            Change::Join { member, offer } => self.add_device(member.clone(), offer.clone(), at),
            Change::Leave { device } => self.remove_device(device, at),
        };
        self.ledger.entries.push(Entry { at, change, record });
        Ok(&self.ledger.entries.last().expect("just pushed").record)
    }

    /// A job arrives at `at` seconds. A graph that fails [`Graph::check`], or a second job of a
    /// name the pool holds, is refused and the pool is unchanged.
    pub fn submit(&mut self, graph: Graph, at: f64) -> Result<&Record, ChangeError> {
        self.apply(Change::Submit { graph }, at)
    }

    /// A job ends at `at` seconds; the jobs waiting are placed again where they now fit.
    pub fn end(&mut self, job: &str, at: f64) -> Result<&Record, ChangeError> {
        self.apply(Change::End { job: job.into() }, at)
    }

    /// A device joins at `at` seconds, with its member's report. Of two reports for one member
    /// the stricter mark stands: remote over local, the shorter time to give a device back.
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

    fn arrive(&mut self, graph: Graph, at: f64) -> Result<Record, ChangeError> {
        let marks = graph.local_only().map_err(ChangeError::Graph)?;
        let name = graph.job.name.clone();
        if self.jobs.iter().any(|j| j.graph.job.name == name) {
            return Err(ChangeError::Duplicate { job: name });
        }
        let parts = graph.parts.len();
        let mut live = Live {
            graph,
            marks,
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
        let marked_remote = member.map(|m| self.add_member(m)).unwrap_or_default();
        let device = offer.device.clone();
        self.offers.retain(|o| o.device != device);
        self.offers.push(offer);
        self.offers.sort_by(|a, b| a.device.cmp(&b.device));
        self.rebase();
        self.run(Event::Join { device }, marked_remote)
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

    /// Records a member's report; returns its name if it marks remote a member known as local.
    fn add_member(&mut self, m: Member) -> Vec<String> {
        match self.members.iter_mut().find(|x| x.name == m.name) {
            Some(known) => {
                let newly_remote = m.remote && !known.remote;
                known.remote |= m.remote;
                known.give_back_seconds = known.give_back_seconds.min(m.give_back_seconds);
                if newly_remote {
                    vec![m.name]
                } else {
                    Vec::new()
                }
            }
            None => {
                self.members.push(m);
                Vec::new()
            }
        }
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
    fn run(&mut self, event: Event, marked_remote: Vec<String>) -> Record {
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
            marked_remote,
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

    /// Why a part of a job may not go on a device, if it may not: no such device, the guard, a
    /// pin, its tolerance class, or a dtype its edges carry.
    fn eligibility(&self, j: &Live, i: usize, d: &DeviceId) -> Option<Removal> {
        let g = &j.graph;
        let part = &g.parts[i];
        let (name, device) = (part.name.clone(), d.clone());
        let Some(offer) = self.offer(d) else {
            return Some(Removal::NoDevice { device });
        };
        // A device whose member the pool does not know is taken as remote.
        if j.marks.parts[i] && self.member(d).is_none_or(|m| m.remote) {
            return Some(Removal::Guard { part: name, device });
        }
        let pinned_elsewhere = match &part.pin {
            Some(Pin::Member(m)) => &d.member != m,
            Some(Pin::Backend(b)) => &offer.backend != b,
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
        j.graph.job.priority as f64 * (1.0 + j.waited / AGING_SECONDS)
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
