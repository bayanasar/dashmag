//! Activation checkpointing: a stack of layers trained with each layer's forward pass run again
//! during the backward pass, so that between the two passes only the layers' inputs stay alive.
//! This is what `torch.utils.checkpoint` around each layer does.
//!
//! [`forward`] runs the stack on Burn's inner backend, outside autodiff, keeping each layer's
//! input, and returns the stack's output as a new autodiff leaf: what follows the stack (a head,
//! the loss) is recorded as usual, and the loss's backward pass stops at that leaf. [`backward`]
//! takes the loss's gradient with respect to the leaf and walks the layers from the last to the
//! first: it runs a layer's forward pass again from its saved input, recorded this time,
//! backpropagates the incoming gradient through it at once, adds the gradients of the layer's
//! parameters to the loss's [`Gradients`], and hands the gradient of the layer's input to the
//! layer below. At most one layer's activations are alive at any time. It returns the gradient
//! of the stack's input, which [`backward_through`] carries on through what computed that input
//! (an embedding), into the same [`Gradients`]. One training step:
//!
//! ```ignore
//! let h = model.embedding.forward(tokens); // recorded
//! let (out, saved) = checkpoint::forward(&model.layers, h.clone(), |l, x| l.forward(x));
//! let loss = model.loss(out.clone(), targets); // recorded
//! let mut grads = loss.backward();
//! let grad = out.grad_remove(&mut grads).unwrap();
//! let grad = checkpoint::backward(&model.layers, saved, grad, |l, x| l.forward(x), &mut grads);
//! checkpoint::backward_through(h, grad, &model.embedding, &mut grads);
//! let grads = GradientsParams::from_grads(grads, &model); // every parameter's, for the optimizer
//! ```
//!
//! A parameter used in more than one place (an embedding tied to the output projection, a layer
//! repeated at several depths) gets the sum of its gradients, as plain autodiff gives it.
//!
//! **Layers.** `layers` is a slice of any [`Module`], one per checkpointed unit: a layer; an
//! enum of layer types, for a stack of mixed layers; a tuple or `Vec` of modules, for a unit of
//! several layers checkpointed together (fewer inputs kept, more activations alive at once) or
//! for a layer whose parts the model holds apart (cloning a module shares its parameters). The
//! layer function may use tensors other than its input and the layer's parameters as constants
//! (masks, position tables); made without autodiff (`Tensor::inner`), they cost the first pass
//! nothing, while on autodiff they make the operations they enter go through it unrecorded. A
//! tensor it uses that requires a gradient and is not a parameter of the layer would have that
//! gradient lost, so [`forward`] panics on one.
//!
//! **The layer runs twice** on the same input, once outside autodiff and once recorded, and must
//! compute the same function both times: nothing random, and nothing that acts only in
//! training. Burn's `Dropout`, `BatchNorm` (batch statistics), `Noise` and `RRelu` act as in
//! training only on autodiff tensors, so a stack holding active ones would run them as in
//! inference in the first pass and as in training in the second, and its gradients would not
//! be those of the loss computed.
//!
//! **Burn's own checkpointing.** Burn 0.22.1 has no such scope built in. Its gradient
//! checkpointing, `Device::gradient_checkpointing()` (`BalancedCheckpointing` in burn-autodiff),
//! works per operation, not per layer: an operation marked memory-bound (the elementwise ones)
//! is recomputed in the backward pass from its parents instead of keeping its output, and so
//! that it can be, each of its compute-bound parents (matrix products, casts) is backed up from
//! the forward pass until the backward pass starts. The strategy is carried by each tensor
//! (`Tensor::with_gradient_checkpointing_strategy`) and an operation mixing two strategies
//! panics, so it can be switched on for one layer's operations; what it keeps inside the layer
//! is still decided operation by operation. On a stack of MLP blocks whose projections carry
//! LoRA adapters it keeps more, not less: the f32 base product and low-rank term feeding each
//! sum are backed up, where plain autodiff frees them. Over 6144 tokens through 24 such blocks
//! at Qwen3.5-0.8B's shapes on an RTX 5070 Ti (dash-lora's `examples/recompute_memory.rs`), the
//! peak above the weights is 7.2 GiB with plain autodiff, 14.2 GiB with balanced checkpointing,
//! and 0.69 GiB with this module (PyTorch with `torch.utils.checkpoint` around each block: 0.73
//! GiB). Each layer's forward pass runs twice: the step is 11 to 42% longer than plain
//! autodiff's, depending on the build (PyTorch's, 45%).
//! This module runs on a device with balanced checkpointing too: the leaves it makes carry the
//! strategy of the stack's parameters, so no operation mixes two.
//!
//! **How.** The public tensor API alone: `Module::valid` for a forward pass that records
//! nothing, a fresh leaf per layer input for the recorded one, and `(y · g).sum().backward()` to
//! backpropagate a given gradient `g` through `y` (the gradient of that sum with respect to `y`
//! is `g`, exactly).
//!
//! **Accuracy** (`tests/checkpoint.rs`: stacks of MLP and causal self-attention blocks built from
//! Burn's tensor operations, every parameter trained, alone and inside a model whose embedding is
//! tied to its output projection; and, where its kernels run, two causal blocks on
//! [`crate::attention`] at 1000 tokens, where 32 key blocks add to the last queries' gradients).
//! The loss and every gradient are plain autodiff's bit for bit, in f32 and in bf16, on Flex and
//! on CUDA with Burn's default kernels or with `autotune`, the attention kernels' blocks with
//! their default deterministic backward. With that backward's query gradient summed with atomics
//! (`AttentionOptions::deterministic` off) they are not: its last bits move from one backward
//! pass to the next, and the gradients below it with them (up to 3.0e-5 of a gradient's largest
//! value in bf16). What can move them by roundings, none of it costing accuracy:
//!
//! - An operation a backend computes one way outside autodiff and another way when recorded.
//!   Flex's layer norm is one: a fused kernel of Flex's own in the forward pass here, Burn's
//!   decomposition when recorded. It moves an f32 stack's gradients by up to 2.8e-7 of a
//!   tensor's largest value.
//! - Fusion. With `fusion`, the passes through one layer fuse its operations otherwise than the
//!   pass through the whole stack: f32 gradients up to 2.5e-7 apart, bf16 ones bit-identical on
//!   these stacks.
//! - Autotune with several threads on the device: a thread that asks for a shape another is
//!   tuning runs an untuned candidate meanwhile, so the two passes need not run the same
//!   kernels (with tests in parallel: up to 3.6e-7 apart in f32).
//! - The order of a sum. Burn's backward pass runs the recorded steps by depth (the longest
//!   path from a leaf) and adds a tensor's gradient contributions as they arrive. A layer whose
//!   input is a leaf can order its steps otherwise than the same layer deep in a stack: near the
//!   leaves, operations on its parameters (a transposed weight) set some steps' depths, deep in
//!   the stack its input does. A sum of three or more contributions then rounds differently.
//!   dash-lora's LoRA'd blocks show it in bf16 (`tests/recompute.rs` there): their gradients are
//!   as far from the same stack computed in f32 as plain autodiff's.

use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{GradientCheckpointingStrategy, Gradients, Tensor};

/// The inputs of a stack's layers, kept by [`forward`] for [`backward`], outside autodiff.
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
///
/// Panics if a layer's output is recorded all the same: the layer function used a tensor that
/// requires a gradient and is not a parameter of the layer.
pub fn forward<L: Module, const D: usize>(
    layers: &[L],
    x: Tensor<D>,
    layer: impl Fn(&L, Tensor<D>) -> Tensor<D>,
) -> (Tensor<D>, SavedInputs<D>) {
    let strategy = layers_strategy(layers).or_else(|| x.gradient_checkpointing_strategy());
    let mut x = x.inner();
    let mut inputs = Vec::with_capacity(layers.len());
    for (i, l) in layers.iter().enumerate() {
        inputs.push(x.clone());
        let y = layer(&l.valid(), x);
        assert!(
            !y.is_tracked(),
            "checkpoint::forward: layer {i} uses a tensor that requires a gradient and is not \
             one of its parameters; its gradient would be lost"
        );
        // Constants the layer function holds may carry autodiff: the next layer runs without.
        x = y.inner();
    }
    (leaf(x, strategy), SavedInputs { inputs, strategy })
}

/// Backpropagates `grad`, the gradient of the loss with respect to the output of [`forward`],
/// through `layers` from the last to the first, running each layer's forward pass again from
/// its saved input with the same `layer` function. Adds the gradient of every parameter of
/// `layers` that requires one to `grads` (summing with what is already there; the loss's own
/// backward pass gives a `Gradients` to start from), and returns the gradient with respect to
/// the stack's input, without autodiff.
pub fn backward<L: Module, const D: usize>(
    layers: &[L],
    saved: SavedInputs<D>,
    grad: Tensor<D>,
    layer: impl Fn(&L, Tensor<D>) -> Tensor<D>,
    grads: &mut Gradients,
) -> Tensor<D> {
    assert_eq!(
        layers.len(),
        saved.inputs.len(),
        "checkpoint::backward needs the layers forward ran"
    );
    let mut grad = grad.inner();
    for (l, input) in layers.iter().zip(saved.inputs).rev() {
        let x = leaf(input, saved.strategy);
        let y = layer(l, x.clone());
        let rest = backward_through(y, grad, l, grads);
        grad = x
            .grad(&rest)
            .unwrap_or_else(|| x.clone().inner().zeros_like());
    }
    grad
}

/// Backpropagates `grad`, the gradient of the loss with respect to `y`, through the computation
/// recorded for `y`, and adds the gradient this gives every parameter of `module` that requires
/// one to `grads`, summing with what is already there. Returns the rest of that backward pass:
/// the gradients of the other leaves `y` was computed from.
///
/// This carries the gradient [`backward`] returns on into the part of the model below the
/// stack. `y` must be recorded (computed from a tensor that requires a gradient).
pub fn backward_through<M: Module, const D: usize>(
    y: Tensor<D>,
    grad: Tensor<D>,
    module: &M,
    grads: &mut Gradients,
) -> Gradients {
    let grad = grad.inner().cast(y.dtype());
    let mut pass = y.mul(grad).sum().backward();
    module.visit(&mut Accumulate {
        from: &mut pass,
        into: grads,
    });
    pass
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

/// Moves each trainable parameter's gradient out of one backward pass into the gradients
/// collected so far, adding to any already there (a parameter used in more than one place).
struct Accumulate<'a> {
    from: &'a mut Gradients,
    into: &'a mut Gradients,
}

impl ModuleVisitor for Accumulate<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if !param.is_require_grad() {
            return;
        }
        let tensor = param.val();
        let Some(grad) = tensor.grad_remove(self.from) else {
            return;
        };
        let grad = match tensor.grad_remove(self.into) {
            Some(previous) => previous.add(grad),
            None => grad,
        };
        tensor.grad_replace(self.into, grad);
    }
}
