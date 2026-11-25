//! Batch normalisation, torch's `F.batch_norm`, written once over [`Tensor`] from its reductions
//! and arithmetic, for any family's model with channels on axis 1 (`[N, C]`, `[N, C, L]`,
//! `[N, C, H, W]` and on to rank 6): the same source on every side.
//!
//! Per channel `c`, over the `m` elements of the other axes:
//!
//! ```text
//! out = (x - mean) / sqrt(var + eps) * weight + bias
//! ```
//!
//! In training, `mean` and `var` are the batch's own, `var` the biased variance (over `m`), and
//! the running statistics move towards them: `running_mean = (1 - momentum) running_mean + momentum
//! mean`, and `running_var` likewise towards the batch's unbiased variance, `var m / (m - 1)`. In
//! eval mode `mean` and `var` are the running statistics, which stay as they are. Training is an
//! argument, not a device or a mode a module keeps: [`batch_norm`]'s `training`. Without a weight
//! or a bias, the multiplication or the addition is left out (torch's `affine=False`).
//!
//! # As torch keeps the running variance
//!
//! torch normalises by the biased variance and keeps the unbiased one in `running_var`, and so does
//! this module, checked against torch 2.13's `F.batch_norm` (`tests/batch_norm.rs`, from
//! `tools/record_conv2d_batch_norm.py`, which asserts the rule on torch's own values). Burn's
//! `BatchNorm` module keeps the biased one, smaller by a factor `(m - 1) / m`: 1.6e-4 relative at
//! the 6272 values per channel of MobileNetV3-Small's last stage at a batch of 128 of 224 x 224
//! images, 1/128 at a batch of 128 without spatial axes. That changes no training step's loss or
//! gradients, which take the batch's own statistics, but it changes eval after training, so a
//! comparison with a Burn module's eval says which variance it uses.
//!
//! # Gradients
//!
//! Through [`crate::vjp`], as the composition of the operations it is written in: each side
//! differentiates the reductions and the arithmetic, and there is no backward pass written out for
//! it. A closed form (Burn's `batch_norm_train_backward`) rounds differently, and on Burn it would
//! have taken a contract operation of its own, run by Burn on one side and composed on the other,
//! for gradients that differ only in their roundings: the composed ones are held to torch's within
//! a count of roundings in `tests/batch_norm.rs`, on every side. Inside [`crate::vjp`] the new
//! running statistics are outputs without a cotangent, constants to any gradient.
//!
//! The forward pass in training is written in the order of Burn's `batch_norm_train`, and in eval
//! in the order of Burn's `batch_norm`: the mean over `x` with its channels moved to the front, the
//! variance as the mean of the centred values' squares, then `(x - mean) / sqrt(var + eps)`, times
//! the weight, plus the bias. So on Burn its output can be the bits of Burn's module (held so on
//! Flex in `tests/batch_norm.rs`); its gradients are not, and are held to Burn's within roundings.
//!
//! # Dtypes
//!
//! `x`, the weight and the bias may be f32, bf16 or f16; the arithmetic is in f32 and the output is
//! rounded once to `x`'s dtype. The running statistics stay in their own dtype, the new ones
//! rounded once to it.

use crate::tensor::Tensor;
use crate::{DType, Result, refuse};

/// The arguments of [`batch_norm`] that have a usual default: torch's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchNormOptions {
    /// Added to the variance before its square root. Default `1e-5`.
    pub eps: f64,
    /// How far, in training, the running statistics move towards the batch's, from 0 (not at
    /// all) to 1 (all the way). Default `0.1`. torch's cumulative average (`momentum=None`) is not
    /// taken: a caller that wants it passes `1 / (batches so far + 1)`.
    pub momentum: f64,
}

impl Default for BatchNormOptions {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            momentum: 0.1,
        }
    }
}

impl BatchNormOptions {
    /// The options with this `eps`.
    pub fn with_eps(self, eps: f64) -> Self {
        Self { eps, ..self }
    }

    /// The options with this momentum.
    pub fn with_momentum(self, momentum: f64) -> Self {
        Self { momentum, ..self }
    }
}

/// A channel's running statistics, `[C]` each, floating: what eval mode normalises by.
#[derive(Clone, Debug)]
pub struct Running {
    /// The running mean.
    pub mean: Tensor,
    /// The running variance, unbiased as torch keeps it.
    pub var: Tensor,
}

impl Running {
    /// torch's initial statistics: means of 0 and variances of 1, f32.
    pub fn new(channels: usize) -> Result<Self> {
        Ok(Self {
            mean: Tensor::zeros(&[channels], DType::F32)?,
            var: Tensor::ones(&[channels], DType::F32)?,
        })
    }
}

/// What [`batch_norm`] returns.
#[derive(Clone, Debug)]
pub struct BatchNorm {
    /// The output, `x`'s shape and dtype.
    pub out: Tensor,
    /// The running statistics after the call: updated in training, as they were in eval mode, none
    /// where none were given. Constants to any gradient.
    pub running: Option<Running>,
}

fn channel_vector(what: &str, t: &Tensor, channels: usize) -> Result<()> {
    if t.shape() != [channels] || !t.dtype().is_floating() {
        return refuse(format!(
            "batch_norm: {what} is [{channels}] floating, got {:?} {:?}",
            t.shape(),
            t.dtype()
        ));
    }
    Ok(())
}

/// Batch normalisation of `x` `[N, C, ...]` (rank 2 to 6) over every axis but the channels', with
/// the batch's statistics in training and the running ones in eval mode, scaled by `weight` and
/// shifted by `bias`, `[C]` each, where given (the module's description says how).
///
/// Refuses `x` of rank under 2 or not floating; a weight, bias or running statistic other than
/// `[C]` and floating; in training, fewer than 2 elements per channel (no unbiased variance), as
/// torch refuses them; eval mode without running statistics; and `eps` below 0 or a momentum
/// outside `0..=1`, or either not finite.
pub fn batch_norm(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running: Option<&Running>,
    training: bool,
    options: BatchNormOptions,
) -> Result<BatchNorm> {
    let shape = x.shape();
    if shape.len() < 2 || !x.dtype().is_floating() {
        return refuse(format!(
            "batch_norm: x is [N, C, ...] floating, got {shape:?} {:?}",
            x.dtype()
        ));
    }
    let BatchNormOptions { eps, momentum } = options;
    if !(eps.is_finite() && eps >= 0.0 && (0.0..=1.0).contains(&momentum)) {
        return refuse(format!(
            "batch_norm: eps {eps} is at least 0 and the momentum {momentum} from 0 to 1"
        ));
    }
    let channels = shape[1];
    for (what, t) in [("the weight", weight), ("the bias", bias)] {
        if let Some(t) = t {
            channel_vector(what, t, channels)?;
        }
    }
    if let Some(running) = running {
        channel_vector("the running mean", &running.mean, channels)?;
        channel_vector("the running variance", &running.var, channels)?;
    }
    let m = x.size() / channels.max(1);
    if training && m < 2 {
        return refuse(format!(
            "batch_norm: training wants more than one value per channel, got x {shape:?}"
        ));
    }
    // `[1, C, 1, ...]`, for what is per channel to broadcast against `x`.
    let mut per_channel = vec![1isize; shape.len()];
    per_channel[1] = channels as isize;
    let by_channel = |t: &Tensor| t.as_dtype(DType::F32)?.reshape(&per_channel);
    // `x` as `[C, m]`: the channels to the front, then the rest flattened.
    let flat = |t: &Tensor| t.swap_axes(0, 1)?.reshape(&[channels as isize, -1]);
    let xf = x.as_dtype(DType::F32)?;
    let (mean, var) = if training {
        let mean = flat(&xf)?.mean_axis(1, false)?;
        let centred = xf.sub(&mean.reshape(&per_channel)?)?;
        let var = flat(&centred.mul(&centred)?)?.mean_axis(1, false)?;
        (mean, var)
    } else {
        let Some(running) = running else {
            return refuse("batch_norm: eval mode normalises by running statistics: none given");
        };
        (
            running.mean.as_dtype(DType::F32)?,
            running.var.as_dtype(DType::F32)?,
        )
    };
    let std = var.add_scalar(eps)?.sqrt()?.reshape(&per_channel)?;
    let mut out = xf.sub(&mean.reshape(&per_channel)?)?.div(&std)?;
    if let Some(weight) = weight {
        out = out.mul(&by_channel(weight)?)?;
    }
    if let Some(bias) = bias {
        out = out.add(&by_channel(bias)?)?;
    }
    let out = out.as_dtype(x.dtype())?;
    let running = match running {
        None => None,
        Some(running) if !training => Some(running.clone()),
        Some(running) => {
            // torch's order: `momentum * batch + (1 - momentum) * running`.
            let toward = |old: &Tensor, batch: &Tensor| -> Result<Tensor> {
                let new = batch
                    .stop_gradient()?
                    .mul_scalar(momentum)?
                    .add(&old.as_dtype(DType::F32)?.mul_scalar(1.0 - momentum)?)?;
                new.as_dtype(old.dtype())?.stop_gradient()
            };
            let unbiased = var.mul_scalar(m as f64 / (m - 1) as f64)?;
            Some(Running {
                mean: toward(&running.mean, &mean)?,
                var: toward(&running.var, &unbiased)?,
            })
        }
    };
    Ok(BatchNorm { out, running })
}
