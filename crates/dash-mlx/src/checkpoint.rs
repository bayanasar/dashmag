//! Recompute for a stack of layers (activation checkpointing): a stack trained with each layer's
//! forward pass run again during the backward pass, so that between the two passes only the
//! layers' inputs stay alive. This is dash-ops' `checkpoint` (its module documentation gives the
//! scheme and its uses), on MLX.
//!
//! [`forward`] runs the stack one layer at a time, evaluating each layer's output before the next,
//! and keeps each layer's input; it returns the stack's output, evaluated, with nothing behind it:
//! what follows the stack (a head, the loss) is differentiated as usual, with respect to that
//! output among the rest. [`backward`] takes the gradient of the output and walks the layers from
//! the last to the first: it runs a layer again from its saved input under a VJP with respect to
//! that input and the layer's trainable parameters, evaluates the result, and hands the gradient of
//! the input to the layer below. At most one layer's activations are alive at a time. It returns
//! the gradient of the stack's input, for what computed that input (an embedding) to carry on with,
//! and each layer's parameter gradients by their paths in the layer.
//!
//! The evaluation between layers is what bounds the memory on MLX: MLX's own `checkpoint`, a
//! function's intermediates computed again in its backward pass, kept a stack of eight layers'
//! activations alive when the whole step was one evaluation (measured, 1140 MiB against plain
//! autodiff's 1133, on eight shared-weight layers).
//!
//! **The layer runs twice** on the same input and must give the same result both times: no
//! dropout in training mode, or a seeded one. A parameter shared by several layers (a layer
//! repeated at several depths) gets one gradient per layer; the caller adds them, as plain autodiff
//! would.

use mlx_rs::Array;
use mlx_rs::error::{Exception, Result};
use mlx_rs::module::{FlattenedModuleParam, ModuleParameters};
use mlx_rs::transforms::{eval, fallible_vjp};
use std::rc::Rc;

/// The layers' inputs [`forward`] kept, one per layer, for [`backward`].
#[derive(Debug, Clone)]
pub struct Saved {
    inputs: Vec<Array>,
}

impl Saved {
    /// How many layers' inputs are kept.
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Whether none are.
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

/// Runs `layers` in turn on `x` with `f`, evaluating each layer's output before the next, and
/// keeps each layer's input: the stack's output, evaluated, and the inputs for [`backward`].
pub fn forward<L, F>(layers: &mut [L], x: &Array, mut f: F) -> Result<(Array, Saved)>
where
    F: FnMut(&mut L, &Array) -> Result<Array>,
{
    let mut inputs = Vec::with_capacity(layers.len());
    let mut h = mlx_rs::stop_gradient(x)?;
    for layer in layers.iter_mut() {
        eval([&h])?;
        inputs.push(h.clone());
        h = mlx_rs::stop_gradient(f(layer, &h)?)?;
    }
    eval([&h])?;
    Ok((h, Saved { inputs }))
}

/// The gradient of the stack's input and each layer's parameter gradients (keyed by their paths
/// in the layer, its trainable parameters alone), from `grad`, the gradient of the stack's output:
/// each layer run again from its saved input under a VJP, from the last layer to the first, each
/// evaluated before the next. `f` and `layers` must be those [`forward`] ran.
pub fn backward<L, F>(
    layers: &mut [L],
    saved: Saved,
    grad: &Array,
    mut f: F,
) -> Result<(Array, Vec<FlattenedModuleParam>)>
where
    L: ModuleParameters,
    F: FnMut(&mut L, &Array) -> Result<Array>,
{
    if saved.inputs.len() != layers.len() {
        return Err(Exception::custom(format!(
            "checkpoint::backward: {} layers, {} saved inputs",
            layers.len(),
            saved.inputs.len()
        )));
    }
    let mut grads = vec![FlattenedModuleParam::new(); layers.len()];
    let mut g = grad.clone();
    for (i, layer) in layers.iter_mut().enumerate().rev() {
        let params: Vec<(Rc<str>, Array)> = layer
            .trainable_parameters()
            .flatten()
            .into_iter()
            .map(|(k, v)| (k, v.clone()))
            .collect();
        let keys: Vec<Rc<str>> = params.iter().map(|(k, _)| k.clone()).collect();
        let mut primals = vec![saved.inputs[i].clone()];
        primals.extend(params.iter().map(|(_, v)| v.clone()));
        let (_, vjps) = fallible_vjp(
            |p: &[Array]| {
                layer.update_flattened(keys.iter().cloned().zip(p[1..].iter().cloned()).collect());
                Ok(vec![f(layer, &p[0])?])
            },
            &primals,
            std::slice::from_ref(&g),
        )?;
        // The layer holds its own arrays again, not the transform's.
        layer.update_flattened(params.into_iter().collect());
        eval(vjps.iter())?;
        let mut vjps = vjps.into_iter();
        g = vjps.next().expect("the input's gradient");
        grads[i] = keys.into_iter().zip(vjps).collect();
    }
    Ok((g, grads))
}
