//! Gradient clipping and AdamW, written once over [`Tensor`], as a training step takes them beside
//! the operations. The arithmetic is torch's, so the two sides differ only in their roundings.

use std::collections::HashMap;

pub use dash_contract::clip::{ClipGradNormOptions, NonFinite, NormType};

use crate::tensor::Tensor;
use crate::{DType, Result};

/// Added to the total before the coefficient is taken from it, as torch adds it.
const EPS: f64 = 1e-6;

/// Clips `grads` by their total norm, in place, and returns that norm as it was before clipping,
/// as [`dash_contract::clip`] defines it: the total in f32 over the gradients that have elements,
/// the coefficient as torch evaluates it, and each gradient multiplied in its own dtype.
pub fn clip_grad_norm(grads: &mut [Tensor], options: &ClipGradNormOptions) -> Result<f32> {
    let mut partials = Vec::with_capacity(grads.len());
    for grad in grads.iter().filter(|g| g.size() > 0) {
        partials.push(match options.norm_type {
            NormType::L2 => grad.as_dtype(DType::F32)?.square()?.sum()?,
            NormType::Inf => grad.abs()?.max()?.as_dtype(DType::F32)?,
        });
    }
    if partials.is_empty() {
        return Ok(0.0);
    }
    let refs: Vec<Tensor> = partials
        .iter()
        .map(|p| p.expand_dims(0))
        .collect::<Result<_>>()?;
    let refs: Vec<&Tensor> = refs.iter().collect();
    let partials = Tensor::concat(&refs, 0)?;
    let total = match options.norm_type {
        NormType::L2 => partials.sum()?.sqrt()?,
        NormType::Inf => partials.max()?,
    };
    // torch evaluates `max_norm / (total + 1e-6)` as `(total + 1e-6).reciprocal() * max_norm`.
    let one = Tensor::scalar(1.0, DType::F32)?;
    let coef = one
        .div(&total.add_scalar(EPS)?)?
        .mul_scalar(options.max_norm)?;
    // At most 1, with NaN kept NaN as `torch.clamp` keeps it: a comparison with NaN is false.
    let coef = coef.gt(&one)?.select(&one, &coef)?;
    let finite = total
        .abs()?
        .lt(&Tensor::scalar(f64::INFINITY, DType::F32)?)?;
    let coef = match options.nonfinite {
        NonFinite::Scale => coef,
        NonFinite::Zero => finite.select(&coef, &Tensor::scalar(0.0, DType::F32)?)?,
    };
    for grad in grads.iter_mut().filter(|g| g.size() > 0) {
        let scaled = grad.mul(&coef.as_dtype(grad.dtype())?)?;
        *grad = match options.nonfinite {
            NonFinite::Scale => scaled,
            NonFinite::Zero => {
                let zero = scaled.zeros_like()?;
                scaled.eq(&scaled)?.select(&scaled, &zero)?
            }
        };
    }
    total.item()
}

/// AdamW as `torch.optim.AdamW` applies it, its state per parameter name.
#[derive(Clone, Debug)]
pub struct AdamW {
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    /// The steps taken.
    pub step: u64,
    state: HashMap<String, (Tensor, Tensor)>,
}

impl AdamW {
    pub fn new(betas: (f64, f64), eps: f64, weight_decay: f64) -> Self {
        Self {
            betas,
            eps,
            weight_decay,
            step: 0,
            state: HashMap::new(),
        }
    }

    /// Counts a step: called once per step, before [`Self::update`] for each parameter.
    pub fn begin_step(&mut self) {
        self.step += 1;
    }

    /// The parameter `name` after this step, from its value and its gradient, at learning rate `lr`.
    pub fn update(&mut self, name: &str, p: &Tensor, g: &Tensor, lr: f64) -> Result<Tensor> {
        let (b1, b2) = self.betas;
        let t = self.step.max(1) as i32;
        let (m, v) = match self.state.remove(name) {
            Some(state) => state,
            None => (p.zeros_like()?, p.zeros_like()?),
        };
        let g = g.as_dtype(p.dtype())?;
        // In torch's order and form: the decay first, then the two moments.
        let p = if self.weight_decay != 0.0 {
            p.mul_scalar(1.0 - lr * self.weight_decay)?
        } else {
            p.clone()
        };
        let m = m.add(&g.sub(&m)?.mul_scalar(1.0 - b1)?)?;
        let v = v.mul_scalar(b2)?.add(&g.square()?.mul_scalar(1.0 - b2)?)?;
        let denominator = v
            .sqrt()?
            .mul_scalar(1.0 / (1.0 - b2.powi(t)).sqrt())?
            .add_scalar(self.eps)?;
        let step = lr / (1.0 - b1.powi(t));
        let p = p.sub(&m.div(&denominator)?.mul_scalar(step)?)?;
        self.state.insert(name.to_string(), (m, v));
        Ok(p)
    }

    /// Every array this holds, to hand to [`crate::eval`] after a step.
    pub fn state(&self) -> impl Iterator<Item = &Tensor> {
        self.state.values().flat_map(|(m, v)| [m, v])
    }
}
