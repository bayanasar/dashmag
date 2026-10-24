//! AdamW as `torch.optim.AdamW` computes it, with a beta1 that the schedule sets per step.
//!
//! Burn's own AdamW has the same update but fixes beta1 when it is built, and
//! `OneCycleLR` moves beta1 every step — and PyTorch corrects the first moment's bias
//! with the beta1 in effect at the step, raised to the step count. Defaults are
//! PyTorch's: beta2 0.999, eps 1e-8 (Burn's is 1e-5).
//!
//! [`OneCycleAdamW`] is the optimiser as a run steps it, its schedules attached; it is
//! checked against PyTorch over a whole schedule, not only the update over a few steps.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use burn::module::{Module, ParamGroup};
use burn::optim::lr_scheduler::module_lr_scheduler::ModuleLrScheduler;
use burn::optim::{
    AdaptiveMomentumState, GradientsParams, LearningRate, ModuleOptimizer, Optimizer,
};
use burn::tensor::{Device, Tensor};

use crate::schedule::{OneCycle, OneCycleLr};

/// beta1, shared by the optimiser and whatever sets it each step.
#[derive(Clone, Debug)]
pub struct Beta1(Arc<AtomicU64>);

impl Beta1 {
    pub fn new(value: f64) -> Self {
        Beta1(Arc::new(AtomicU64::new(value.to_bits())))
    }

    pub fn set(&self, value: f64) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }

    pub fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
}

#[derive(Clone, Debug)]
pub struct AdamW {
    pub beta1: Beta1,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl AdamW {
    pub fn new(weight_decay: f64, beta1: Beta1) -> Self {
        AdamW {
            beta1,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        }
    }
}

impl Optimizer for AdamW {
    type State<const D: usize> = AdaptiveMomentumState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let (beta1, beta2) = (self.beta1.get(), self.beta2);
        // param.mul_(1 - lr * weight_decay), before the update reads it.
        let tensor = tensor.mul_scalar(1.0 - lr * self.weight_decay);
        let (time, m, v) = match state {
            Some(s) => (
                s.time + 1,
                s.moment_1
                    .mul_scalar(beta1)
                    .add(grad.clone().mul_scalar(1.0 - beta1)),
                s.moment_2
                    .mul_scalar(beta2)
                    .add(grad.square().mul_scalar(1.0 - beta2)),
            ),
            None => (
                1,
                grad.clone().mul_scalar(1.0 - beta1),
                grad.square().mul_scalar(1.0 - beta2),
            ),
        };
        let correction1 = 1.0 - beta1.powi(time as i32);
        let correction2 = 1.0 - beta2.powi(time as i32);
        // denom = sqrt(v) / sqrt(correction2) + eps; param -= lr / correction1 * m / denom
        let denom = v
            .clone()
            .sqrt()
            .div_scalar(correction2.sqrt())
            .add_scalar(self.eps);
        let tensor = tensor.sub(m.clone().div(denom).mul_scalar(lr / correction1));
        (tensor, Some(AdaptiveMomentumState::new(time, m, v)))
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.to_device(device)
    }
}

/// [`AdamW`] under `OneCycleLR`, as PyTorch steps a model with two parameter groups: one
/// schedule for every parameter, a second for `group` (the backbone, at a lower peak), and
/// beta1 set before each step from the first schedule's momentum — which does not depend
/// on the peak rate, so it is every group's.
pub struct OneCycleAdamW {
    optim: ModuleOptimizer,
    rates: ModuleLrScheduler,
    momentum: OneCycle,
    beta1: Beta1,
    step: usize,
}

impl OneCycleAdamW {
    pub fn new(
        schedule: OneCycle,
        group: ParamGroup,
        group_schedule: OneCycle,
        weight_decay: f64,
    ) -> Self {
        let beta1 = Beta1::new(schedule.max_momentum);
        OneCycleAdamW {
            optim: ModuleOptimizer::from(AdamW::new(weight_decay, beta1.clone())),
            rates: ModuleLrScheduler::new(OneCycleLr::new(schedule.clone()))
                .with_group(group, OneCycleLr::new(group_schedule)),
            momentum: schedule,
            beta1,
            step: 0,
        }
    }

    /// One optimiser step: this step's rate for each group and its beta1, then AdamW.
    pub fn step<M: Module>(&mut self, module: M, grads: GradientsParams) -> M {
        self.beta1.set(self.momentum.at(self.step).1);
        self.step += 1;
        self.optim.step(self.rates.step(), module, grads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn::tensor::TensorData;

    /// Recorded from torch 2.13 (`AdamW([p], weight_decay=0.05, foreach=False)`, lr and
    /// betas set on the group before each step):
    ///
    /// ```python
    /// p = torch.nn.Parameter(torch.tensor([0.5, -1.0, 2.0, 0.0]))
    /// for lr, b1, g in zip([1e-2, 2e-2, 5e-3], [0.95, 0.9, 0.85], grads):
    ///     opt.param_groups[0]["lr"] = lr; opt.param_groups[0]["betas"] = (b1, 0.999)
    ///     p.grad = torch.tensor(g); opt.step()
    /// ```
    #[test]
    fn three_steps_with_beta1_moving_are_pytorchs() {
        let device = Device::default();
        let beta1 = Beta1::new(0.0);
        let optim = AdamW::new(0.05, beta1.clone());
        let grads = [
            [0.1, -0.2, 0.3, 0.0],
            [-0.4, 0.1, 0.2, 0.05],
            [0.3, 0.3, -0.1, -0.2],
        ];
        let want = [
            [
                0.4897499978542328,
                -0.9894999861717224,
                1.9889999628067017,
                0.0,
            ],
            [
                0.5020747184753418,
                -0.989176332950592,
                1.9731782674789429,
                -0.01488273311406374,
            ],
            [
                0.5012967586517334,
                -0.9916787147521973,
                1.9718765020370483,
                -0.012077050283551216,
            ],
        ];
        let mut p = Tensor::<1>::from_data(TensorData::from([0.5f32, -1.0, 2.0, 0.0]), &device);
        let mut state = None;
        for (((lr, b1), g), want) in [1e-2, 2e-2, 5e-3]
            .into_iter()
            .zip([0.95, 0.9, 0.85])
            .zip(grads)
            .zip(want)
        {
            beta1.set(b1);
            let g = Tensor::<1>::from_data(TensorData::from(g.map(|x: f64| x as f32)), &device);
            (p, state) = optim.step(lr, p, g, state);
            let got = p.to_data().try_to_vec::<f32>().unwrap();
            for (a, b) in got.iter().zip(want) {
                assert!((f64::from(*a) - b).abs() < 1e-6, "{got:?} vs {want:?}");
            }
        }
    }

    #[derive(Module, Debug)]
    struct Groups {
        backbone: Param<Tensor<1>>,
        heads: Param<Tensor<1>>,
    }

    fn splitmix64(z: u64) -> u64 {
        let z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// `reference/adamw_torch.json`, recorded from torch 2.13 on the CPU in float32: two
    /// groups as a run builds them, `AdamW([{backbone, lr 3e-4}, {heads, lr 1e-3}],
    /// weight_decay=0.05, foreach=False)` under `OneCycleLR(max_lr=[3e-4, 1e-3],
    /// total_steps=1875, pct_start=0.1)`, each step `opt.step()`, `sched.step()`,
    /// `opt.zero_grad()`; the parameters after each step in `checkpoints`.
    ///
    /// Element `i` of 12 starts at `init[i]`; its gradient at step `k` in group `g` is
    /// `2^-exp[i % 6] * (bias + u - 0.5)`, `bias` 0.25 for `i < 6` and 0 after, `u` the top
    /// 24 bits of `splitmix64((k << 16) | (g << 8) | i)` over 2^24. Every value is exact in
    /// f32, so both sides are fed the same bits. The scales fall from 1 to 2^-30, past eps:
    /// late in the schedule, with the rate small and the second moments built up, the
    /// placement of eps and the order of decay and update are what the result turns on.
    ///
    /// The worst difference measured is 2.4e-7, one f32 step at these magnitudes. Each of
    /// these, tried once, moves it past the bound of 1e-6: decay after the update (2.3e-5),
    /// beta1 a step late (1.4e-5), beta1 held at 0.95 (4.6e-3), eps added before the bias
    /// correction (0.064) or under the root (0.56), the two groups' rates swapped (0.38).
    #[test]
    fn the_whole_schedule_is_pytorchs() {
        let j: serde_json::Value =
            serde_json::from_str(include_str!("../reference/adamw_torch.json")).unwrap();
        let ints = |key: &str| -> Vec<usize> {
            let v = j[key].as_array().unwrap();
            v.iter().map(|x| x.as_u64().unwrap() as usize).collect()
        };
        let (steps, exp, checkpoints) = (
            j["steps"].as_u64().unwrap() as usize,
            ints("exp"),
            ints("checkpoints"),
        );
        let init: Vec<f32> = j["init"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let grad = |k: usize, g: usize| -> Vec<f32> {
            (0..init.len())
                .map(|i| {
                    let bits = splitmix64(((k as u64) << 16) | ((g as u64) << 8) | i as u64) >> 40;
                    let u = bits as f64 / (1u64 << 24) as f64;
                    let bias = if i < 6 { 0.25 } else { 0.0 };
                    (2f64.powi(-(exp[i % 6] as i32)) * (bias + u - 0.5)) as f32
                })
                .collect()
        };

        let device = Device::default().autodiff();
        let tensor = |v: &[f32]| Tensor::<1>::from_data(TensorData::from(v), &device);
        let mut model = Groups {
            backbone: Param::from_tensor(tensor(&init)),
            heads: Param::from_tensor(tensor(&init)),
        };
        let mut optim = OneCycleAdamW::new(
            OneCycle::new(1e-3, steps, 0.1),
            ParamGroup::from_ids(vec![model.backbone.id]),
            OneCycle::new(3e-4, steps, 0.1),
            0.05,
        );
        let (mut worst, mut seen) = (0f64, 0);
        for k in 0..steps {
            // d/dp sum(p * g) is g exactly, and it reaches the optimiser as a run's does.
            let loss = model.backbone.val().mul(tensor(&grad(k, 0))).sum()
                + model.heads.val().mul(tensor(&grad(k, 1))).sum();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(model, grads);
            if checkpoints.contains(&(k + 1)) {
                for (g, p) in [&model.backbone, &model.heads].into_iter().enumerate() {
                    let got = p.val().into_data().try_to_vec::<f32>().unwrap();
                    let want = j["groups"][g]["at"][seen].as_array().unwrap();
                    for (a, b) in got.iter().zip(want) {
                        worst = worst.max((f64::from(*a) - b.as_f64().unwrap()).abs());
                    }
                }
                seen += 1;
            }
        }
        assert_eq!(seen, checkpoints.len());
        assert!(worst < 1e-6, "worst difference {worst:e}");
    }
}
