//! PyTorch's `OneCycleLR` with its defaults — two phases, cosine annealing, momentum cycled — step
//! by step, in f64 on the host: no tensor, the same on every side.
//!
//! The rate climbs from `max_lr / 25` to `max_lr` over the first `pct_start` of the steps and falls
//! to `max_lr / (25 * 1e4)` by the last; beta1 moves the other way, 0.95 down to 0.85 and back.
//! Step `k` here is the `k`-th optimiser step, counted from 0: the rate and beta1 PyTorch has in
//! effect when it takes it. With [`crate::optim::AdamW`], beta1 goes into `betas.0` before the step
//! and the rate into each `update`; a parameter group with its own peak is a second `OneCycle`,
//! whose beta1 is the same, since the momentum does not depend on the peak.

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
