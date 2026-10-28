//! Gradients clipped by their global norm, as `torch.nn.utils.clip_grad_norm_` clips them:
//! `clip_grad_norm` measures the norm of all the gradients a model's parameters have, as one
//! vector, and multiplies each by `max_norm / (total + 1e-6)` when that is below 1, before the
//! optimiser steps. PPO and GRPO clip them so.
//!
//! As torch computes it:
//! - The total is over the parameters that have a gradient. One without (frozen, as LoRA's base
//!   weights are, or outside the loss) is skipped: it neither counts nor gets a gradient. A
//!   parameter the model holds twice counts and is scaled once, as `Module.parameters()` lists it
//!   once.
//! - The coefficient is `max_norm / (total + 1e-6)`, at most 1, evaluated as torch evaluates it,
//!   so that the same total gives torch's coefficient bit for bit. A coefficient of 1 leaves
//!   every gradient as it was, bit for bit.
//! - The total is returned, as it was before clipping, `[1]` f32. Without any gradient it is 0
//!   and nothing is scaled.
//!
//! The norm is accumulated in f32 whatever the gradients' dtype: a bf16 or f16 gradient is cast
//! to f32 for its sum of squares, and the total and the coefficient are f32. Each gradient is then
//! multiplied in its own dtype, by the coefficient rounded to it. torch rounds a bf16 gradient's
//! norm, the total and each step of the coefficient to bf16 instead (its totals in the recorded
//! cases are up to 2.6e-3 off); here only the coefficient is rounded, for the multiplication. An
//! f16 coefficient below 2^-14 is subnormal and keeps fewer bits; f64 gradients are summed in f32
//! too.
//!
//! A total that is not finite (a square overflows f32, or an element is infinite or NaN) is
//! handled as [`ClipGradNormOptions::nonfinite`] says: the gradients scaled all the same, torch's
//! default, or zeroed. torch can instead raise (`error_if_nonfinite=True`), which reads the total
//! on the host; here that read is the caller's, at a point where it reads values anyway, and it
//! can skip the optimiser step there. Under [`NormType::Inf`] a NaN element makes the total NaN,
//! as torch's `max` does.

/// The norm `clip_grad_norm` measures the gradients by: two of torch's `norm_type`s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NormType {
    /// `norm_type=2.0`, torch's default: the square root of the sum of every element's square.
    #[default]
    L2,
    /// `norm_type=inf`: the largest absolute value of any element.
    Inf,
}

/// What `clip_grad_norm` does to the gradients when their total norm is infinite or NaN.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NonFinite {
    /// Scale them by the coefficient all the same, as torch does by default
    /// (`error_if_nonfinite=False`). The coefficient of an infinite total is 0, so a finite
    /// element becomes 0 and an infinite one NaN; that of a NaN total is NaN, and so is every
    /// element.
    #[default]
    Scale,
    /// Set every element of every gradient to 0, so that no infinity or NaN reaches the
    /// optimiser. The optimiser still steps: AdamW decays the weights and moves them by the
    /// moments it holds.
    Zero,
}

/// The options of `clip_grad_norm`: torch's arguments as plain data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipGradNormOptions {
    /// The norm the gradients are scaled down to when theirs is larger: torch's `max_norm`,
    /// which has no default. 1.0 by default.
    pub max_norm: f64,
    /// [`NormType::L2`] by default, as torch.
    pub norm_type: NormType,
    /// [`NonFinite::Scale`] by default, as torch.
    pub nonfinite: NonFinite,
}

impl Default for ClipGradNormOptions {
    fn default() -> Self {
        ClipGradNormOptions {
            max_norm: 1.0,
            norm_type: NormType::L2,
            nonfinite: NonFinite::Scale,
        }
    }
}
