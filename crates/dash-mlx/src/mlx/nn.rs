//! The activations this crate's operations use, and gradients over a layer's parameters.

use super::array::Array;
use super::error::Result;
use super::module::{FlattenedModuleParam, ModuleParameters};
use super::ops::{logaddexp, sigmoid};
use super::transforms::{compile_shapeless_once, value_and_grad_all};
use std::rc::Rc;

/// `x - logsumexp(x)` along `axis`, the last by default.
pub fn log_softmax(x: impl AsRef<Array>, axis: impl Into<Option<i32>>) -> Result<Array> {
    let x = x.as_ref();
    x.subtract(x.logsumexp_axis(axis.into().unwrap_or(-1), true)?)
}

/// `ln(1 + exp(x))`.
pub fn softplus(x: impl AsRef<Array>) -> Result<Array> {
    logaddexp(x.as_ref(), Array::from_int(0))
}

/// `x * sigmoid(x)`, compiled, as MLX's own `nn.silu` is, so that the two run as one kernel.
pub fn silu(x: impl AsRef<Array>) -> Result<Array> {
    let f = |args: &[Array]| Ok(vec![args[0].multiply(sigmoid(&args[0])?)?]);
    let out = compile_shapeless_once(f, std::slice::from_ref(x.as_ref()))?;
    Ok(out.into_iter().next().expect("silu's output"))
}

/// `f(layer, args)` and its gradient to `layer`'s trainable parameters, by their paths. `f`'s value
/// is a scalar. The layer holds the transform's arrays afterwards, with the same values.
pub fn value_and_grad<'a, M, A, F>(
    mut f: F,
) -> impl FnMut(&mut M, A) -> Result<(Array, FlattenedModuleParam)> + 'a
where
    M: ModuleParameters + 'a,
    A: Clone + 'a,
    F: FnMut(&mut M, A) -> Result<Array> + 'a,
{
    move |layer, args| {
        let (keys, primals): (Vec<Rc<str>>, Vec<Array>) = layer
            .trainable_parameters()
            .into_iter()
            .map(|(k, v)| (k, v.clone()))
            .unzip();
        let (values, grads) = value_and_grad_all(
            |p: &[Array]| {
                layer.update_flattened(keys.iter().cloned().zip(p.iter().cloned()).collect());
                Ok(vec![f(&mut *layer, args.clone())?])
            },
            &primals,
        )?;
        let value = values.into_iter().next().expect("the value");
        Ok((value, keys.iter().cloned().zip(grads).collect()))
    }
}
