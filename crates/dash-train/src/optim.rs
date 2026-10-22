//! AdamW as `torch.optim.AdamW` computes it, with a beta1 that the schedule sets per step.
//!
//! Burn's own AdamW has the same update but fixes beta1 when it is built, and
//! `OneCycleLR` moves beta1 every step — and PyTorch corrects the first moment's bias
//! with the beta1 in effect at the step, raised to the step count. Defaults are
//! PyTorch's: beta2 0.999, eps 1e-8 (Burn's is 1e-5).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use burn::optim::{AdaptiveMomentumState, LearningRate, Optimizer};
use burn::tensor::{Device, Tensor};

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

#[cfg(test)]
mod tests {
    use super::*;
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
}
