//! `dash_mag::schedule::OneCycle` as a Burn scheduler, which the Burn optimiser in [`crate::optim`]
//! steps with until the loop moves onto `dash-mag`. The schedule itself is `dash-mag`'s, held there
//! to PyTorch's at every step of a recorded run.

use burn::optim::LearningRate;
use burn::optim::lr_scheduler::{LrScheduler, LrSchedulerRecord};
use dash_mag::schedule::OneCycle;

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

    #[test]
    fn the_scheduler_hands_out_steps_in_order() {
        let mut s = OneCycleLr::new(OneCycle::new(1e-3, 100, 0.1));
        let first = s.step();
        assert!((first - 1e-3 / 25.0).abs() < 1e-18, "{first}");
        let second = s.step();
        assert_eq!(second, s.schedule.at(1).0);
    }
}
