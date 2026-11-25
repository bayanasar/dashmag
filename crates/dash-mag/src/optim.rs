//! Gradient clipping and AdamW, written once over [`Tensor`], as a training step takes them beside
//! the operations. The arithmetic is torch's on f32 parameters, LoRA's factors, so the two sides
//! differ only in their roundings. A scalar here is rounded to the tensor's dtype, so on bf16
//! parameters, where torch keeps its scalars in f32, it is not torch's to the bit.

use std::collections::HashMap;

pub use dash_contract::clip::{ClipGradNormOptions, NonFinite, NormType};

use crate::tensor::Tensor;
use crate::{DType, Result, refuse};

/// Added to the total before the coefficient is taken from it, as torch adds it.
const EPS: f64 = 1e-6;

/// Clips `grads` by their total norm, in place, and returns that norm as it was before clipping,
/// as [`dash_contract::clip`] defines it: the total in f32 over the gradients that have elements,
/// the coefficient as torch evaluates it, and each gradient multiplied in its own dtype. The total
/// is read back to the host on every call, where torch returns a tensor.
///
/// The total is over the tensors in `grads`, each once per place in the slice. A tied parameter,
/// one tensor a model uses in two places, has one gradient, the sum of both uses, and the caller
/// lists it once: listed twice, it counts twice in the total. A parameter without a gradient,
/// frozen or outside the loss, is left out by the caller.
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
///
/// Each parameter counts its own steps, the updates it has had, and its bias correction raises the
/// betas to that count, as torch keeps `step` per parameter. A parameter that joins later, such as
/// a backbone unfrozen mid-run, starts from 1, and one the caller leaves out of a step keeps its
/// moments and its count, as torch skips a parameter without a gradient. The caller names the
/// parameters each step: nothing here walks a model.
///
/// One rate per [`Self::update`] call and one `betas` for every parameter, set between steps, so
/// a schedule and parameter groups with their own rates need no more. One weight decay per
/// optimiser: groups that decay differently, such as none on norms and biases, are two instances,
/// each over its own parameters' names, since the state is keyed by name.
#[derive(Clone, Debug)]
pub struct AdamW {
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    /// The steps the optimiser has taken, counted by [`Self::begin_step`]: what a checkpoint keeps,
    /// and the count of a parameter whose moments were put back without their own.
    pub step: u64,
    state: HashMap<String, Moments>,
}

/// One parameter's state.
#[derive(Clone, Debug)]
struct Moments {
    m: Tensor,
    v: Tensor,
    /// The updates the parameter has had, or `None` for moments put back by [`AdamW::set_state`]
    /// alone: then it has had every step [`AdamW::step`] counts.
    steps: Option<u64>,
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

    /// The updates the parameter `name` has had, which its next one adds 1 to, or `None` if it
    /// holds no state. Between steps, for a checkpoint beside [`Self::named_state`].
    pub fn steps(&self, name: &str) -> Option<u64> {
        self.state
            .get(name)
            .map(|held| held.steps.unwrap_or(self.step))
    }

    /// Puts back the count [`Self::steps`] gave, for moments [`Self::set_state`] has put back.
    pub fn set_steps(&mut self, name: &str, steps: u64) -> Result<()> {
        match self.state.get_mut(name) {
            Some(held) => {
                held.steps = Some(steps);
                Ok(())
            }
            None => refuse(format!(
                "AdamW: a count of steps for {name}, which has no state"
            )),
        }
    }

    /// The parameter `name` after this step, from its value and its gradient, at learning rate `lr`.
    pub fn update(&mut self, name: &str, p: &Tensor, g: &Tensor, lr: f64) -> Result<Tensor> {
        if let Some(Moments { m, .. }) = self.state.get(name)
            && (m.shape() != p.shape() || m.dtype() != p.dtype())
        {
            return refuse(format!(
                "AdamW: moments of {:?} {:?} for {name}, of {:?} {:?}",
                m.dtype(),
                m.shape(),
                p.dtype(),
                p.shape()
            ));
        }
        let (b1, b2) = self.betas;
        let (m, v, steps) = match self.state.remove(name) {
            Some(Moments { m, v, steps }) => (m, v, steps.unwrap_or(self.step.max(1) - 1)),
            None => (p.zeros_like()?, p.zeros_like()?, 0),
        };
        let steps = steps + 1;
        let t = steps as i32;
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
            .div(&Tensor::scalar((1.0 - b2.powi(t)).sqrt(), p.dtype())?)?
            .add_scalar(self.eps)?;
        let step = lr / (1.0 - b1.powi(t));
        let p = p.sub(&m.div(&denominator)?.mul_scalar(step)?)?;
        self.state.insert(
            name.to_string(),
            Moments {
                m,
                v,
                steps: Some(steps),
            },
        );
        Ok(p)
    }

    /// Every array this holds, to hand to [`crate::eval`] after a step.
    pub fn state(&self) -> impl Iterator<Item = &Tensor> {
        self.state.values().flat_map(|held| [&held.m, &held.v])
    }

    /// Each parameter's name and its two moments, in order of name: what a checkpoint keeps, beside
    /// [`Self::step`] and each parameter's [`Self::steps`].
    pub fn named_state(&self) -> impl Iterator<Item = (&str, &Tensor, &Tensor)> {
        let mut held: Vec<_> = self.state.iter().collect();
        held.sort_by_key(|(name, _)| name.as_str());
        held.into_iter()
            .map(|(name, held)| (name.as_str(), &held.m, &held.v))
    }

    /// Puts a parameter's moments back, as [`Self::named_state`] gave them, so that a resumed run
    /// continues the one it left; [`Self::step`] goes back beside them. They are of one floating dtype and
    /// shape, and [`Self::update`] holds them to the parameter's.
    ///
    /// Until [`Self::set_steps`] gives it its own count, the parameter is taken to have had every
    /// step [`Self::step`] counts.
    pub fn set_state(&mut self, name: &str, m: Tensor, v: Tensor) -> Result<()> {
        if !m.dtype().is_floating() || m.dtype() != v.dtype() || m.shape() != v.shape() {
            return refuse(format!(
                "AdamW: moments of {:?} {:?} and {:?} {:?} for {name}",
                m.dtype(),
                m.shape(),
                v.dtype(),
                v.shape()
            ));
        }
        self.state
            .insert(name.to_string(), Moments { m, v, steps: None });
        Ok(())
    }
}
