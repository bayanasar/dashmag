//! The controller of a mag pool: the placement loop.
//!
//! On every event (a job arrives or ends, a device joins or leaves) [`Pool`] runs the same steps
//! over the jobs the event touches:
//!
//! 1. **Generate** candidate arrangements: each touched job either waits or is placed. An atomic
//!    part goes on one device; a chain's parts go together on one device; a replicable part runs
//!    a copy on every device that offers its part kind. With few enough candidates (at most
//!    [`EXHAUSTIVE`]) every combination is listed; past that, jobs are placed greedily, highest
//!    priority first, and the record says so.
//! 2. **Remove** a candidate that breaks a constraint, recording the first reason found: the
//!    memory each device offers; a part's tolerance class and its edges' dtypes against the
//!    offer; a pin; the guard (a part carrying local-only data never on a device of a member
//!    marked remote); and the job's need, a rate it must sustain.
//! 3. **Predict** each remaining candidate's score: the sum over jobs of their priority times
//!    their rate, each job's rate divided by its best rate alone on the pool. A job's rate is the
//!    inverse of its busiest device's time per step: the device's time is the sum of what every
//!    part on it, of any job, takes per step, and a replicable part's copies share its items in
//!    proportion to their speed. A waiting job's priority grows with the time it has waited,
//!    doubling every [`AGING_SECONDS`].
//! 4. **Commit** the best candidate if it scores above leaving things as they are, and **record**
//!    the run.
//!
//! What this step leaves out, and where it comes: probing candidates whose bands overlap, which
//! needs runs (M-1.3, with `mag-mar`); the time of edges crossing a link; dividing a chain
//! across devices, sharing a device's memory with a running job's parts by choice, and
//! squeezing a running job (M-6). Running jobs are not moved here, so moving costs nothing.
//!
//! The same members, offers and events give the same records: candidates are generated in a
//! fixed order (devices by [`DeviceId`]) and the first of equal scores wins.

use std::collections::BTreeMap;

use mag_by::{
    Arrangement, Candidate, Cost, DeviceId, Dtype, Event, Graph, LocalOnly, Mark, Member, Move,
    Offer, Pin, Place, Record, Refusal, Removal, Unit,
};

/// The most candidates listed one by one before generation turns greedy (proposed).
pub const EXHAUSTIVE: usize = 500;

/// A waiting job's priority doubles for every this many seconds it has waited (proposed).
pub const AGING_SECONDS: f64 = 60.0;

/// The band given to an estimated cost, relative: wide, so that it is probed and not trusted.
pub const ESTIMATE_BAND: f64 = 1.0;

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
}

/// A pool's controller: its members, their offers, its jobs, and the record of every run of the
/// loop.
#[derive(Clone, Debug)]
pub struct Pool {
    members: Vec<Member>,
    offers: Vec<Offer>,
    jobs: Vec<Live>,
    records: Vec<Record>,
    now: f64,
}

/// One way to place one job: where each of its parts goes.
type Placement = Vec<Place>;

impl Pool {
    /// A pool of these members and their devices' offers, holding no job.
    pub fn new(members: Vec<Member>, mut offers: Vec<Offer>) -> Self {
        offers.sort_by(|a, b| a.device.cmp(&b.device));
        Pool {
            members,
            offers,
            jobs: Vec::new(),
            records: Vec::new(),
            now: 0.0,
        }
    }

    /// Every run of the loop so far, in order.
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// Where a job's parts are, if the pool holds it.
    pub fn arrangement(&self, job: &str) -> Option<&Arrangement> {
        self.jobs
            .iter()
            .find(|j| j.graph.job.name == job)
            .map(|j| &j.arrangement)
    }

    /// A job arrives at `at` seconds. A graph that fails [`Graph::check`] is refused and the
    /// pool is unchanged.
    pub fn submit(&mut self, graph: Graph, at: f64) -> Result<&Record, Refusal> {
        let marks = graph.local_only()?;
        let name = graph.job.name.clone();
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
        Ok(self.run(Event::Arrival { job: name }))
    }

    /// A job ends at `at` seconds; the jobs waiting are placed again where they now fit.
    pub fn end(&mut self, job: &str, at: f64) -> &Record {
        self.advance(at);
        self.jobs.retain(|j| j.graph.job.name != job);
        self.run(Event::End { job: job.into() })
    }

    /// A device joins at `at` seconds, with its member if the pool does not know it yet.
    pub fn join(&mut self, member: Option<Member>, offer: Offer, at: f64) -> &Record {
        self.advance(at);
        if let Some(m) = member
            && !self.members.iter().any(|x| x.name == m.name)
        {
            self.members.push(m);
        }
        let device = offer.device.clone();
        self.offers.push(offer);
        self.offers.sort_by(|a, b| a.device.cmp(&b.device));
        self.rebase();
        self.run(Event::Join { device })
    }

    /// A device leaves at `at` seconds: the parts on it wait, and are placed again where they
    /// fit.
    pub fn leave(&mut self, device: &DeviceId, at: f64) -> &Record {
        self.advance(at);
        self.offers.retain(|o| &o.device != device);
        self.rebase();
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
        self.run(Event::Leave {
            device: device.clone(),
        })
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
    fn run(&mut self, event: Event) -> &Record {
        let touched: Vec<usize> = (0..self.jobs.len())
            .filter(|&i| self.jobs[i].waiting())
            .collect();
        let options: Vec<Vec<Placement>> = touched
            .iter()
            .map(|&i| self.placements(&self.jobs[i]))
            .collect();
        let combinations = options
            .iter()
            .try_fold(1usize, |n, o| n.checked_mul(o.len() + 1))
            .unwrap_or(usize::MAX);
        let greedy = combinations > EXHAUSTIVE;
        let choices: Vec<Vec<Option<usize>>> = if greedy {
            vec![self.greedy(&touched, &options)]
        } else {
            all_choices(&options)
        };
        let mut candidates = Vec::new();
        let mut best: Option<(usize, f64)> = None;
        for choice in choices {
            let mut jobs = self.jobs.clone();
            let mut moves = Vec::new();
            for (k, &i) in touched.iter().enumerate() {
                match choice[k] {
                    Some(c) => {
                        jobs[i].arrangement.places = options[k][c].clone();
                        moves.push(Move::Fit);
                    }
                    None => moves.push(Move::Wait),
                }
            }
            let arrangements = touched
                .iter()
                .map(|&i| jobs[i].arrangement.clone())
                .collect();
            let removed = self.removal(&jobs);
            let (score, band) = match removed {
                Some(_) => (None, None),
                None => {
                    let (s, b) = self.score(&jobs);
                    (Some(s), Some(b))
                }
            };
            if let Some(s) = score
                && best.is_none_or(|(_, b)| s > b)
            {
                best = Some((candidates.len(), s));
            }
            candidates.push(Candidate {
                arrangements,
                moves,
                removed,
                score,
                band,
            });
        }
        // Leaving things as they are scores what the waiting jobs already have: nothing.
        let (current, _) = self.score(&self.jobs);
        let chosen = best.filter(|&(_, s)| s > current).map(|(i, _)| i);
        if let Some(i) = chosen {
            for (k, &j) in touched.iter().enumerate() {
                self.jobs[j].arrangement = candidates[i].arrangements[k].clone();
            }
        }
        self.records.push(Record {
            event,
            candidates,
            chosen,
            greedy,
        });
        self.records.last().expect("just pushed")
    }

    /// Every way to place a job: each chain's parts together and each other atomic part on one
    /// device, a replicable part's copies on every device that offers its part kind.
    fn placements(&self, job: &Live) -> Vec<Placement> {
        let g = &job.graph;
        let devices: Vec<&DeviceId> = self.offers.iter().map(|o| &o.device).collect();
        // The groups placed as one: each chain, and each atomic part outside a chain.
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
        let mut base: Placement = vec![Place::Waiting; g.parts.len()];
        for (i, p) in g.parts.iter().enumerate() {
            if p.mark == Mark::Replicable {
                // A copy on every device that may take it and has the memory for one.
                let copies: Vec<DeviceId> = self
                    .offers
                    .iter()
                    .filter(|o| {
                        self.eligibility(job, i, &o.device).is_none()
                            && o.memory_bytes >= p.memory_bytes
                    })
                    .map(|o| o.device.clone())
                    .collect();
                if copies.is_empty() {
                    return Vec::new();
                }
                base[i] = Place::Copies(copies);
            }
        }
        let mut out = vec![base];
        for group in groups {
            let mut next = Vec::new();
            for placement in &out {
                for &d in &devices {
                    let mut p = placement.clone();
                    for &i in &group {
                        p[i] = Place::On(d.clone());
                    }
                    next.push(p);
                }
            }
            out = next;
        }
        if devices.is_empty() {
            out.clear();
        }
        out
    }

    /// Greedy: each job in turn, highest priority first, takes its best placement beside those
    /// already taken, or waits.
    fn greedy(&self, touched: &[usize], options: &[Vec<Placement>]) -> Vec<Option<usize>> {
        let mut order: Vec<usize> = (0..touched.len()).collect();
        order.sort_by(|&a, &b| {
            let (pa, pb) = (
                self.priority(&self.jobs[touched[a]]),
                self.priority(&self.jobs[touched[b]]),
            );
            pb.total_cmp(&pa).then(a.cmp(&b))
        });
        let mut jobs = self.jobs.clone();
        let mut choice = vec![None; touched.len()];
        for k in order {
            let mut best: Option<(usize, f64)> = None;
            for (c, placement) in options[k].iter().enumerate() {
                let mut trial = jobs.clone();
                trial[touched[k]].arrangement.places = placement.clone();
                if self.removal(&trial).is_none() {
                    let (s, _) = self.score(&trial);
                    if best.is_none_or(|(_, b)| s > b) {
                        best = Some((c, s));
                    }
                }
            }
            if let Some((c, _)) = best {
                jobs[touched[k]].arrangement.places = options[k][c].clone();
                choice[k] = Some(c);
            }
        }
        choice
    }

    fn offer(&self, d: &DeviceId) -> Option<&Offer> {
        self.offers.iter().find(|o| &o.device == d)
    }

    fn member(&self, d: &DeviceId) -> Option<&Member> {
        self.members.iter().find(|m| m.name == d.member)
    }

    /// A part kind's cost on a device: measured, or else estimated from what other devices
    /// measured (the slowest of them), or else one second, marked as an estimate either way.
    fn cost(&self, offer: &Offer, unit: &Unit) -> Option<Cost> {
        if let Some(c) = offer.costs.iter().find(|c| &c.unit == unit) {
            return Some(c.clone());
        }
        let elsewhere = self
            .offers
            .iter()
            .flat_map(|o| o.costs.iter())
            .filter(|c| &c.unit == unit && !c.estimate)
            .map(|c| c.seconds_per_step)
            .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.max(t))));
        Some(Cost {
            unit: unit.clone(),
            seconds_per_step: elsewhere.unwrap_or(1.0),
            band: ESTIMATE_BAND,
            estimate: true,
        })
    }

    /// Why a part of a job may not go on a device, if it may not: no offer, the guard, a pin,
    /// its tolerance class, or a dtype its edges carry.
    fn eligibility(&self, j: &Live, i: usize, d: &DeviceId) -> Option<Removal> {
        let g = &j.graph;
        let part = &g.parts[i];
        let (name, device) = (part.name.clone(), d.clone());
        let Some(offer) = self.offer(d) else {
            return Some(Removal::Memory { device });
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

    /// The first constraint the jobs' places break, if any: a part's eligibility, then each
    /// device's memory, then each placed job's need.
    fn removal(&self, jobs: &[Live]) -> Option<Removal> {
        let mut memory: BTreeMap<&DeviceId, u64> = BTreeMap::new();
        for j in jobs {
            for (i, place) in j.arrangement.places.iter().enumerate() {
                let devices: Vec<&DeviceId> = match place {
                    Place::Waiting => continue,
                    Place::On(d) => vec![d],
                    Place::Copies(ds) => ds.iter().collect(),
                };
                for d in devices {
                    if let Some(r) = self.eligibility(j, i, d) {
                        return Some(r);
                    }
                    let used = memory.entry(d).or_insert(0);
                    *used += j.graph.parts[i].memory_bytes;
                    if *used > self.offer(d).map_or(0, |o| o.memory_bytes) {
                        return Some(Removal::Memory { device: d.clone() });
                    }
                }
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

    /// Each placed job's rate in steps per second, with the widest band among the costs it
    /// used; a waiting job's is 0.
    fn rates<'a>(&self, jobs: &'a [Live]) -> BTreeMap<&'a str, (f64, f64)> {
        // Each device's time per step, over every part of every job on it.
        let mut load: BTreeMap<&DeviceId, f64> = BTreeMap::new();
        let mut band: BTreeMap<&str, f64> = BTreeMap::new();
        for j in jobs {
            let name = j.graph.job.name.as_str();
            band.insert(name, 0.0);
            for (i, place) in j.arrangement.places.iter().enumerate() {
                let unit = &j.graph.parts[i].unit;
                let costs: Vec<(&DeviceId, Cost)> = match place {
                    Place::Waiting => continue,
                    Place::On(d) => vec![(d, self.offer_cost(d, unit))],
                    Place::Copies(ds) => ds.iter().map(|d| (d, self.offer_cost(d, unit))).collect(),
                };
                // Copies share the items in proportion to their speed.
                let speed: f64 = costs.iter().map(|(_, c)| 1.0 / c.seconds_per_step).sum();
                for (d, c) in &costs {
                    let share = (1.0 / c.seconds_per_step) / speed;
                    *load.entry(d).or_insert(0.0) += share * c.seconds_per_step;
                    let b = band.get_mut(name).expect("inserted above");
                    *b = b.max(c.band);
                }
            }
        }
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
                    .flat_map(|p| match p {
                        Place::Waiting => Vec::new(),
                        Place::On(d) => vec![d],
                        Place::Copies(ds) => ds.iter().collect(),
                    })
                    .map(|d| load.get(d).copied().unwrap_or(0.0))
                    .fold(0.0, f64::max);
                let rate = if busiest > 0.0 { 1.0 / busiest } else { 0.0 };
                (name, (rate, band[name]))
            })
            .collect()
    }

    fn offer_cost(&self, d: &DeviceId, unit: &Unit) -> Cost {
        self.offer(d)
            .and_then(|o| self.cost(o, unit))
            .expect("a placed device has an offer")
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

    /// The job's best rate alone on the pool as it is when it arrives.
    fn best_alone(&self, job: &Live) -> f64 {
        let alone = Pool {
            members: self.members.clone(),
            offers: self.offers.clone(),
            jobs: Vec::new(),
            records: Vec::new(),
            now: 0.0,
        };
        alone
            .placements(job)
            .into_iter()
            .filter_map(|places| {
                let mut j = job.clone();
                j.arrangement.places = places;
                let jobs = [j];
                alone.removal_without_need(&jobs)?;
                Some(alone.rates(&jobs)[job.graph.job.name.as_str()].0)
            })
            .fold(0.0, f64::max)
    }

    /// `Some(())` when nothing but the need would remove the candidate.
    fn removal_without_need(&self, jobs: &[Live]) -> Option<()> {
        match self.removal(jobs) {
            None | Some(Removal::Need { .. }) => Some(()),
            Some(_) => None,
        }
    }
}

/// Every combination of one choice per job: a placement, or waiting (`None`), waiting first.
fn all_choices(options: &[Vec<Placement>]) -> Vec<Vec<Option<usize>>> {
    let mut out = vec![Vec::new()];
    for o in options {
        let mut next = Vec::new();
        for prefix in &out {
            for c in std::iter::once(None).chain((0..o.len()).map(Some)) {
                let mut p: Vec<Option<usize>> = prefix.clone();
                p.push(c);
                next.push(p);
            }
        }
        out = next;
    }
    out
}
