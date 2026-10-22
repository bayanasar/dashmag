//! PyTorch's `OneCycleLR` with its defaults — two phases, cosine annealing, momentum
//! cycled — step by step.
//!
//! The rate climbs from `max_lr / 25` to `max_lr` over the first `pct_start` of the steps
//! and falls to `max_lr / (25 * 1e4)` by the last; beta1 moves the other way, 0.95 down to
//! 0.85 and back. Step `k` here is the `k`-th optimiser step, counted from 0: the rate and
//! beta1 PyTorch has in effect when it takes it.

use burn::optim::LearningRate;
use burn::optim::lr_scheduler::{LrScheduler, LrSchedulerRecord};

#[derive(Clone, Debug)]
pub struct OneCycle {
    pub max_lr: f64,
    pub total_steps: usize,
    pub pct_start: f64,
    pub div_factor: f64,
    pub final_div_factor: f64,
    pub base_momentum: f64,
    pub max_momentum: f64,
}

impl OneCycle {
    /// PyTorch's defaults for everything but the peak, the length and the warm-up share.
    pub fn new(max_lr: f64, total_steps: usize, pct_start: f64) -> Self {
        OneCycle {
            max_lr,
            total_steps,
            pct_start,
            div_factor: 25.0,
            final_div_factor: 1e4,
            base_momentum: 0.85,
            max_momentum: 0.95,
        }
    }

    fn anneal(start: f64, end: f64, pct: f64) -> f64 {
        end + (start - end) / 2.0 * ((std::f64::consts::PI * pct).cos() + 1.0)
    }

    /// The learning rate and beta1 in effect at optimiser step `step`.
    pub fn at(&self, step: usize) -> (f64, f64) {
        let initial = self.max_lr / self.div_factor;
        let min = initial / self.final_div_factor;
        // (end step, start lr, end lr, start momentum, end momentum), as PyTorch lays the
        // phases out; the first ends at a fractional step.
        let phases = [
            (
                self.pct_start * self.total_steps as f64 - 1.0,
                initial,
                self.max_lr,
                self.max_momentum,
                self.base_momentum,
            ),
            (
                (self.total_steps - 1) as f64,
                self.max_lr,
                min,
                self.base_momentum,
                self.max_momentum,
            ),
        ];
        let s = step as f64;
        let mut start = 0.0;
        for (i, &(end, lr0, lr1, m0, m1)) in phases.iter().enumerate() {
            if s <= end || i == phases.len() - 1 {
                let pct = (s - start) / (end - start);
                return (Self::anneal(lr0, lr1, pct), Self::anneal(m0, m1, pct));
            }
            start = end;
        }
        unreachable!("the last phase always answers")
    }
}

/// [`OneCycle`] as a Burn scheduler: each `step` returns the rate for the next optimiser
/// step. Not resumable: a run that stops starts over.
#[derive(Clone)]
pub struct OneCycleLr {
    pub schedule: OneCycle,
    pub next: usize,
}

impl OneCycleLr {
    pub fn new(schedule: OneCycle) -> Self {
        OneCycleLr { schedule, next: 0 }
    }
}

impl LrScheduler for OneCycleLr {
    fn step(&mut self) -> LearningRate {
        let (lr, _) = self.schedule.at(self.next);
        self.next += 1;
        lr
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::new()
    }

    fn load_record(&mut self, _record: LrSchedulerRecord) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    /// `reference/onecycle_torch.json`, recorded from torch 2.13: `OneCycleLR(max_lr=[3e-4,
    /// 1e-3], total_steps=1875, pct_start=0.1)` over AdamW, the rate of each of two parameter
    /// groups and beta1, at every step. Kept in the crate: no generator rebuilds it.
    #[test]
    fn every_step_is_pytorchs() {
        let j: serde_json::Value =
            serde_json::from_str(include_str!("../reference/onecycle_torch.json")).unwrap();
        let steps = j["steps"].as_u64().unwrap() as usize;
        let groups = [
            OneCycle::new(3e-4, steps, 0.1),
            OneCycle::new(1e-3, steps, 0.1),
        ];
        let mut worst: f64 = 0.0;
        for k in 0..steps {
            for (g, sched) in groups.iter().enumerate() {
                let (lr, beta1) = sched.at(k);
                let want_lr = j["lr"][k][g].as_f64().unwrap();
                let want_beta1 = j["beta1"][k][g].as_f64().unwrap();
                worst = worst
                    .max((lr - want_lr).abs() / want_lr)
                    .max((beta1 - want_beta1).abs() / want_beta1);
            }
        }
        assert!(worst < 1e-12, "worst relative difference {worst:e}");
    }

    #[test]
    fn the_scheduler_hands_out_steps_in_order() {
        let mut s = OneCycleLr::new(OneCycle::new(1e-3, 100, 0.1));
        let first = s.step();
        assert!((first - 1e-3 / 25.0).abs() < 1e-18, "{first}");
        let second = s.step();
        assert_eq!(second, s.schedule.at(1).0);
    }
}
