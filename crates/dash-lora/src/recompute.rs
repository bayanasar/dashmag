//! Training through a stack of layers with each layer's forward pass run again during the
//! backward pass, so that between the two passes only the layers' inputs stay alive.
//!
//! [`forward`] runs the stack on Burn's inner backend, outside autodiff, keeping each layer's
//! input. [`backward`] takes the gradient of the loss with respect to the stack's output and
//! walks the layers from the last to the first: it runs a layer's forward pass again from its
//! saved input, recorded this time, backpropagates the incoming gradient through it at once,
//! adds the gradients of the layer's parameters to a [`GradientsParams`], and hands the
//! gradient of the layer's input to the layer below. At most one layer's activations are alive
//! at any time. This is what `torch.utils.checkpoint` around each decoder layer does.
//!
//! Burn 0.22.1 has no such scope built in. Its gradient checkpointing,
//! `Device::gradient_checkpointing()` (`BalancedCheckpointing` in burn-autodiff), works per
//! operation, not per layer: an operation marked memory-bound (the elementwise ones) is
//! recomputed in the backward pass from its parents instead of keeping its output, and so that
//! it can be, each of its compute-bound parents (matrix products, casts) is backed up from the
//! forward pass until the backward pass starts. The strategy is carried by each tensor
//! (`Tensor::with_gradient_checkpointing_strategy`) and an operation mixing two strategies
//! panics, so it can be switched on for one layer's operations; what it keeps inside the layer
//! is still decided operation by operation. On a LoRA'd stack it keeps more, not less: the f32
//! base product and low-rank term feeding each sum are backed up, where plain autodiff frees
//! them. Over 6144 tokens through 24 LoRA'd MLP blocks at Qwen3.5-0.8B's shapes on an RTX 5070
//! Ti (`examples/recompute_memory.rs`), the peak above the weights is 7.2 GiB with plain
//! autodiff, 14.3 GiB with balanced checkpointing, and 0.72 GiB with this module (PyTorch with
//! `torch.utils.checkpoint` around each block: 0.73 GiB).
//!
//! Hence this module, built on the public tensor API alone: `Module::valid` for a forward pass
//! that records nothing, a fresh leaf per layer input for the recorded one, and
//! `(y · g).sum().backward()` to backpropagate a given gradient `g` through `y` (the gradient
//! of that sum with respect to `y` is `g`, exactly).
//!
//! The layer function runs twice on the same input and must give the same result both times:
//! no dropout, or anything else random, inside it.

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::tensor::{GradientCheckpointingStrategy, Gradients, Tensor};

/// The inputs of a stack's layers, kept by [`forward`] for [`backward`].
#[derive(Debug)]
pub struct SavedInputs<const D: usize> {
    inputs: Vec<Tensor<D>>,
    strategy: Option<GradientCheckpointingStrategy>,
}

impl<const D: usize> SavedInputs<D> {
    /// How many layers' inputs are kept.
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Whether the stack had no layers.
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

/// Runs `layers` in order on `x`, `layer(l, x)` being one layer's forward pass, without
/// recording anything for autodiff. Returns the stack's output as a new leaf that requires a
/// gradient, with autodiff on, for the rest of the model and the loss; and each layer's input,
/// for [`backward`].
pub fn forward<L: Module, const D: usize>(
    layers: &[L],
    x: Tensor<D>,
    layer: impl Fn(&L, Tensor<D>) -> Tensor<D>,
) -> (Tensor<D>, SavedInputs<D>) {
    let strategy = layers_strategy(layers).or_else(|| x.gradient_checkpointing_strategy());
    let mut x = x.inner();
    let mut inputs = Vec::with_capacity(layers.len());
    for l in layers {
        inputs.push(x.clone());
        x = layer(&l.valid(), x);
    }
    (leaf(x, strategy), SavedInputs { inputs, strategy })
}

/// Backpropagates `grad`, the gradient of the loss with respect to the output of [`forward`],
/// through `layers` from the last to the first, running each layer's forward pass again from
/// its saved input. Adds the gradient of every parameter of `layers` that requires one to
/// `grads` (summing with what is already there), and returns the gradient with respect to the
/// stack's input, without autodiff.
pub fn backward<L: Module, const D: usize>(
    layers: &[L],
    saved: SavedInputs<D>,
    grad: Tensor<D>,
    layer: impl Fn(&L, Tensor<D>) -> Tensor<D>,
    grads: &mut GradientsParams,
) -> Tensor<D> {
    assert_eq!(
        layers.len(),
        saved.inputs.len(),
        "backward needs the layers forward ran"
    );
    let mut grad = grad.inner();
    for (l, input) in layers.iter().zip(saved.inputs).rev() {
        let x = leaf(input, saved.strategy);
        let y = layer(l, x.clone());
        let mut layer_grads = y.clone().mul(grad.cast(y.dtype())).sum().backward();
        l.visit(&mut Collect {
            grads: &mut layer_grads,
            out: grads,
        });
        grad = x
            .grad(&layer_grads)
            .unwrap_or_else(|| x.clone().inner().zeros_like());
    }
    grad
}

/// A new autodiff leaf requiring a gradient, under the strategy the stack's parameters use.
fn leaf<const D: usize>(
    x: Tensor<D>,
    strategy: Option<GradientCheckpointingStrategy>,
) -> Tensor<D> {
    let x = x.autodiff();
    let x = match strategy {
        Some(strategy) => x.with_gradient_checkpointing_strategy(strategy),
        None => x,
    };
    x.require_grad()
}

/// The checkpointing strategy of the first parameter that requires a gradient, if any.
fn layers_strategy<L: Module>(layers: &[L]) -> Option<GradientCheckpointingStrategy> {
    struct First(Option<GradientCheckpointingStrategy>);
    impl ModuleVisitor for First {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if self.0.is_none() && param.is_require_grad() {
                self.0 = param.val().gradient_checkpointing_strategy();
            }
        }
    }
    let mut first = First(None);
    for l in layers {
        l.visit(&mut first);
    }
    first.0
}

/// Moves each trainable parameter's gradient out of one layer's backward pass into the
/// optimizer's gradients, adding to any already there (a parameter shared between layers).
struct Collect<'a> {
    grads: &'a mut Gradients,
    out: &'a mut GradientsParams,
}

impl ModuleVisitor for Collect<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if !param.is_require_grad() {
            return;
        }
        let Some(grad) = param.val().grad_remove(self.grads) else {
            return;
        };
        let grad = match self.out.remove::<D>(param.id) {
            Some(previous) => previous.add(grad),
            None => grad,
        };
        self.out.register(param.id, grad);
    }
}
