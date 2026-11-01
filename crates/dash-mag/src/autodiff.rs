//! The one way to a gradient.

use crate::Result;
use crate::arm::{Arm, Selected};
use crate::tensor::{A, Tensor};

/// Runs `f` on `primals` and gives each primal the gradient of the sum of `out * cotangent` over
/// the outputs that have a cotangent; an output without one takes no part. Returns the outputs,
/// constants to any further gradient, and the gradients, each in its primal's dtype, zeros where
/// a primal does not reach the outputs. An output narrower than f32 takes its cotangent rounded to
/// its own dtype.
///
/// A loss is one output with a cotangent of 1. A stack of layers run layer by layer, each from its
/// saved input, is one call per layer, each taking as its cotangent the gradient the layer above
/// gave its input; what a layer carries from an earlier run of tokens is a primal like its input.
/// A tensor `f` captures is a constant. `f` runs once here, and calling `vjp` inside `f` is not
/// supported.
pub fn vjp<F>(
    mut f: F,
    primals: &[Tensor],
    cotangents: &[Option<Tensor>],
) -> Result<(Vec<Tensor>, Vec<Tensor>)>
where
    F: FnMut(&[Tensor]) -> Result<Vec<Tensor>>,
{
    let mut run = |arrays: &[A]| -> Result<Vec<A>> {
        let tensors: Vec<Tensor> = arrays.iter().cloned().map(Tensor).collect();
        Ok(f(&tensors)?.into_iter().map(|t| t.0).collect())
    };
    let primals: Vec<A> = primals.iter().map(|t| t.0.clone()).collect();
    let cotangents: Vec<Option<A>> = cotangents
        .iter()
        .map(|c| c.as_ref().map(|c| c.0.clone()))
        .collect();
    let (outs, grads) = Selected::vjp(&mut run, &primals, &cotangents)?;
    Ok((
        outs.into_iter().map(Tensor).collect(),
        grads.into_iter().map(Tensor).collect(),
    ))
}
