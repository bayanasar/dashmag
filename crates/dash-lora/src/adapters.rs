//! Several adapters over one base weight, and the choice of the one each forward pass uses.
//!
//! A [`LoraLinear`](crate::LoraLinear) holds any number of adapters by name over its one frozen
//! base weight, and its forward pass adds the low-rank term of one of them, or of none. Which one
//! is a setting of the layer: [`DEFAULT_ADAPTER`] unless chosen otherwise, and chosen for every
//! layer of a module tree at once by [`AdapterChoice::set_adapter`], on any `#[derive(Module)]`
//! model, whatever holds its layers (fields, `Vec`s, `Option`s, arrays, tuples, enums). A layer
//! that does not hold the chosen adapter runs its base alone, as PEFT's layers do; `None` runs
//! every layer's base alone, as PEFT's `disable_adapter()` does. However many adapters there are,
//! the base weights are one copy, and a forward pass costs what it would with the chosen adapter
//! alone: it runs the same operations in the same order, so its results are bit for bit those of
//! a layer holding that adapter alone (`tests/adapters.rs`).
//!
//! ```ignore
//! use dash_lora::{AdapterChoice, ApplyOptions, PeftAdapter};
//! // Trained: the adapter training starts from. Frozen: the same files, as the reference.
//! let start = PeftAdapter::read("sft")?;
//! start.apply(model.lora_layers_mut(), &device)?; // "default", trainable
//! let frozen = ApplyOptions { name: "reference".into(), trainable: false };
//! start.apply_with(model.lora_layers_mut(), &device, &frozen)?;
//! // Each step: the reference's outputs from a clone choosing it (the clone shares every
//! // tensor; nothing the reference runs requires a gradient, so nothing is recorded), then the
//! // policy's.
//! let reference = model.clone().set_adapter(Some("reference")).forward(x.clone()).detach();
//! let policy = model.forward(x); // "default", recorded
//! let loss = objective(policy, reference); // a KL penalty against the reference, say
//! let grads = GradientsParams::from_grads(loss.backward(), &model); // "default"'s alone
//! model = optim.step(lr, model, grads);
//! ```
//!
//! **Why a setting of the layers, chosen across the tree.** The model is the caller's code, and
//! its forward pass calls each layer's as it would without adapters; nothing is passed through
//! it. The choice is part of the module value, so it goes wherever the model goes and nowhere
//! else:
//!
//! - It survives what Burn does to a module: clones, `fork` and `to_device`, the optimizer's step
//!   (which maps the module and returns it), `valid` and `train`. A choice held as a Burn
//!   [`Flag`](burn::module::Flag) would not: `valid` and `freeze` set every flag, since a flag
//!   means training behaviour.
//! - [`dash_ops::checkpoint`] runs each layer twice, the second time in the backward pass, on the
//!   same module value: both runs use the same adapter. A choice held in a scope around the
//!   forward pass (a thread-local, as PEFT's context managers set an attribute for their block)
//!   ends before the backward pass begins, and the recomputed layers would run another adapter
//!   than the forward pass they stand for.
//! - Nothing is shared between models or threads. A policy and a reward model on one thread, both
//!   LoRA'd, do not see each other's choice, as they would a thread-local's; clones of one model
//!   trained on several devices from several threads do not, as they would a handle the layers
//!   shared.
//! - A scoped choice is a clone: `model.clone().set_adapter(Some("reference"))` shares every
//!   tensor of `model` (a clone of a module copies handles, not data) and leaves `model`'s choice
//!   as it was, so it cannot leak past its use.
//!
//! With one adapter there is nothing to choose: [`LoraLinear::with_lora`](crate::LoraLinear::with_lora)
//! adds it as [`DEFAULT_ADAPTER`], which every layer uses until told otherwise.
//!
//! **Training.** Whether an adapter trains is its own: its factors require a gradient or not
//! ([`Lora::new`] makes them trainable, Burn's `Module::no_grad` freezes them,
//! [`ApplyOptions::trainable`](crate::ApplyOptions) says which when reading PEFT's files), and
//! choosing never changes it. A frozen adapter, chosen or not, records nothing for autodiff,
//! gets no gradient, and so gets no optimizer state: Burn's optimizers keep state only for the
//! parameters a step hands them a gradient for. An adapter that is not chosen takes no part in
//! the forward pass at all. PEFT's `set_adapter` couples the two (it makes the adapter it
//! activates trainable and freezes the others, unless `inference_mode`); here, the reference
//! adapter of a KL penalty stays frozen while chosen, and the trained adapter stays trainable
//! while another is chosen.
//!
//! **How the choice reaches the layers.** Burn's `Module::map` is the one traversal every module
//! tree has, derived or not. [`AdapterChoice::set_adapter`] maps the tree with a mapper that
//! changes no parameter, while each layer's adapters take the choice as the traversal passes
//! them (the choice waits in a thread-local for the length of the traversal, a plain recursive
//! call on the calling thread); any other walk of the tree (an optimizer's step, `no_grad`,
//! loading a record) leaves the choice as it is. No tensor is touched or copied: cloning a layer
//! holding two adapters and choosing for it takes 0.2 to 0.3 µs on the host
//! (`examples/lora_layer.rs`; PEFT's `set_adapter` on the same layer, 18 to 21 µs).
//!
//! **Paths.** Each adapter's factors sit under its name in the module's parameter paths,
//! `<layer>.adapters.<name>.a` and `.b`, so that Burn's own records and parameter groups name
//! them the same whatever other adapters a layer holds. A name is therefore not empty and has no
//! `.`, as PyTorch requires of the keys of the `ModuleDict`s PEFT holds its adapters in.

use std::cell::RefCell;
use std::collections::BTreeSet;

use burn::module::{
    Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper, ModuleVisitor,
};
use burn::tensor::Device;

use crate::linear::Lora;

/// PEFT's default adapter name: the adapter `get_peft_model` and `PeftModel.from_pretrained`
/// create unless told otherwise, and the one `save_pretrained` writes at the top of its directory
/// (every other adapter in a subdirectory named after it). Here, the adapter
/// [`LoraLinear::with_lora`](crate::LoraLinear::with_lora) adds and a layer uses until
/// [`AdapterChoice::set_adapter`] chooses another.
pub const DEFAULT_ADAPTER: &str = "default";

/// The container type the adapters report to Burn's module traversals, after the convention of
/// derived modules (`Struct:<type>` for the container whose entries are being entered).
const CONTAINER: &str = "Struct:Adapters";

/// The adapters of one layer, by name in the order they were added, and the name of the one its
/// forward pass uses.
#[derive(Clone, Debug)]
pub(crate) struct Adapters {
    named: Vec<(String, Lora)>,
    active: Option<String>,
}

impl Adapters {
    pub(crate) fn new() -> Self {
        Self {
            named: Vec::new(),
            active: Some(DEFAULT_ADAPTER.to_string()),
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<&Lora> {
        self.named.iter().find(|(n, _)| n == name).map(|(_, l)| l)
    }

    /// Adds `lora` under `name`, or replaces the adapter of that name in its place; returns the
    /// one it replaces.
    pub(crate) fn insert(&mut self, name: &str, lora: Lora) -> Option<Lora> {
        assert!(
            !name.is_empty() && !name.contains('.'),
            "an adapter's name is not empty and has no '.' (it is a component of the parameters' \
             paths, as in PEFT's ModuleDicts): {name:?}"
        );
        match self.named.iter_mut().find(|(n, _)| n == name) {
            Some((_, old)) => Some(std::mem::replace(old, lora)),
            None => {
                self.named.push((name.to_string(), lora));
                None
            }
        }
    }

    pub(crate) fn remove(&mut self, name: &str) -> Option<Lora> {
        let i = self.named.iter().position(|(n, _)| n == name)?;
        Some(self.named.remove(i).1)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&str, &Lora)> {
        self.named.iter().map(|(n, l)| (n.as_str(), l))
    }

    pub(crate) fn active_name(&self) -> Option<&str> {
        self.active.as_deref()
    }

    /// The adapter the forward pass uses: the chosen one, if this layer holds it.
    pub(crate) fn active(&self) -> Option<&Lora> {
        self.active.as_deref().and_then(|name| self.get(name))
    }

    fn each(self, mut f: impl FnMut(Lora) -> Lora) -> Self {
        Self {
            named: self.named.into_iter().map(|(n, l)| (n, f(l))).collect(),
            active: self.active,
        }
    }
}

/// What a walk of a module tree asks of each layer's adapters as it passes them.
enum Walk {
    /// [`AdapterChoice::set_adapter`]: the choice, for every layer.
    Choose(Option<String>),
    /// [`AdapterChoice::adapter_names`]: every name a layer holds, gathered.
    Names(BTreeSet<String>),
}

thread_local! {
    /// The walk in progress on this thread, for as long as it runs; none otherwise. A module
    /// traversal is a plain recursive call on the calling thread, so what is set here before it
    /// starts is what every layer it reaches sees, and nothing outside it does.
    static WALK: RefCell<Option<Walk>> = const { RefCell::new(None) };
}

/// Runs `f`, a traversal of a module tree, with `walk` in progress, and returns what the walk
/// gathered. The walk ends with `f`, by panic too.
fn walking<R>(walk: Walk, f: impl FnOnce() -> R) -> (R, Walk) {
    struct End;
    impl Drop for End {
        fn drop(&mut self) {
            WALK.with_borrow_mut(|w| *w = None);
        }
    }
    WALK.with_borrow_mut(|w| {
        assert!(
            w.is_none(),
            "an adapter walk is already in progress on this thread"
        );
        *w = Some(walk);
    });
    let end = End;
    let out = f();
    let walk = WALK
        .with_borrow_mut(Option::take)
        .expect("the walk lasts until its traversal returns");
    drop(end);
    (out, walk)
}

struct Unchanged;
impl ModuleMapper for Unchanged {}

struct Nothing;
impl ModuleVisitor for Nothing {}

impl Module for Adapters {
    fn collect_devices(&self, devices: Devices) -> Devices {
        self.named
            .iter()
            .fold(devices, |devices, (_, lora)| lora.collect_devices(devices))
    }

    fn fork(self, device: &Device) -> Self {
        self.each(|lora| lora.fork(device))
    }

    fn to_device(self, device: &Device) -> Self {
        self.each(|lora| lora.to_device(device))
    }

    fn train(self) -> Self {
        self.each(Module::train)
    }

    fn valid(&self) -> Self {
        Self {
            named: self
                .named
                .iter()
                .map(|(n, l)| (n.clone(), l.valid()))
                .collect(),
            active: self.active.clone(),
        }
    }

    fn materialize(self) -> Self {
        self.each(Module::materialize)
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        WALK.with_borrow_mut(|walk| {
            if let Some(Walk::Names(names)) = walk {
                names.extend(self.named.iter().map(|(n, _)| n.clone()));
            }
        });
        for (name, lora) in &self.named {
            visitor.enter_module(name, CONTAINER);
            lora.visit(visitor);
            visitor.exit_module(name, CONTAINER);
        }
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        let Self { named, mut active } = self;
        WALK.with_borrow(|walk| {
            if let Some(Walk::Choose(choice)) = walk {
                active = choice.clone();
            }
        });
        let named = named
            .into_iter()
            .map(|(name, lora)| {
                mapper.enter_module(&name, CONTAINER);
                let lora = lora.map(mapper);
                mapper.exit_module(&name, CONTAINER);
                (name, lora)
            })
            .collect();
        Self { named, active }
    }
}

impl ModuleDisplayDefault for Adapters {
    fn content(&self, content: Content) -> Option<Content> {
        self.named
            .iter()
            .fold(content, |content, (name, lora)| content.add(name, lora))
            .add_debug_attribute("active", &self.active)
            .set_top_level_type("Adapters")
            .optional()
    }

    fn num_params(&self) -> usize {
        Module::num_params(self)
    }
}

impl ModuleDisplay for Adapters {}

/// Choosing the adapter of every [`LoraLinear`](crate::LoraLinear) in a module tree at once.
/// Implemented for every Burn module: a model, a block, a `Vec` of layers, one layer.
pub trait AdapterChoice: Module + Sized {
    /// The module with every [`LoraLinear`](crate::LoraLinear) in it using the adapter named
    /// `adapter` in its forward pass from now on, PEFT's `set_adapter`; a layer that does not
    /// hold one of that name runs its base alone. `None` runs every layer's base alone, PEFT's
    /// `disable_adapter()`. Which adapters train is left as it is.
    ///
    /// The module is consumed and returned, as Burn's `valid` and `no_grad` return theirs; to
    /// run another adapter once and keep the module's choice, choose on a clone, which shares
    /// every tensor: `model.clone().set_adapter(Some("reference")).forward(x)`.
    ///
    /// # Panics
    ///
    /// If no layer of the module holds an adapter named `adapter`: a misspelt name would
    /// otherwise run the base alone, silently (PEFT raises too).
    #[must_use = "the choice is made on the module returned; the one it was called on is consumed"]
    fn set_adapter(self, adapter: Option<&str>) -> Self {
        if let Some(name) = adapter {
            let names = self.adapter_names();
            assert!(
                names.iter().any(|n| n == name),
                "set_adapter({name:?}): no layer of this module holds an adapter of that name \
                 (its layers hold {names:?})"
            );
        }
        let (module, _) = walking(Walk::Choose(adapter.map(str::to_string)), || {
            self.map(&mut Unchanged)
        });
        module
    }

    /// The names of the adapters the layers of the module hold, each once, in sorted order.
    fn adapter_names(&self) -> Vec<String> {
        match walking(Walk::Names(BTreeSet::new()), || self.visit(&mut Nothing)) {
            ((), Walk::Names(names)) => names.into_iter().collect(),
            _ => unreachable!("the walk started gathering names"),
        }
    }
}

impl<M: Module> AdapterChoice for M {}
