//! LoRA on MLX: a linear layer whose base weight is frozen at its own precision (bf16) while the
//! low-rank adapters `A` and `B` are f32 and alone receive gradients, never folded into the base;
//! several adapters over one base by name, and the choice of the one a forward pass uses. This is
//! dash-lora's `LoraLinear`, `Lora` and adapter choice (their module documentation gives the
//! arithmetic and why the adapters are never merged).
//!
//! `y = x Wᵀ + scale · (x Aᵀ) Bᵀ`, `W` `[out, in]` frozen, `A` `[r, in]` and `B` `[out, r]` f32,
//! `scale = lora_alpha / r` (`/ √r` with rsLoRA): PEFT's layout, names and arithmetic.
//!
//! - The base product `x Wᵀ` at the weight's dtype (`x` cast to it if it differs): MLX's matmul,
//!   which accumulates bf16 in f32 and rounds each output once, as cuBLAS does under PyTorch.
//! - The low-rank term in f32 throughout, in PEFT's order, and its sum with the base in f32,
//!   rounded once to `x`'s dtype, PEFT's `(result + lora).to(result.dtype)`:
//!   [`crate::lowrank::low_rank`] with the base, whose kernel never holds the f32 term;
//!   with no adapter chosen, the base product alone at `x`'s dtype, as PEFT's layer returns it
//!   with its adapters disabled.
//!
//! The base weight is a frozen parameter: MLX's `value_and_grad` over the trainable parameters
//! never differentiates it. An adapter's factors train unless frozen (`freeze_parameters`).
//!
//! **The choice of adapter.** dash-lora sets it across a whole module tree through Burn's module
//! mapper; mlx-rs has no mapper over a module's own fields, so here it is a setting of each layer,
//! [`LoraLinear::set_adapter`], and [`set_adapter`] sets it on every layer a model hands over (its
//! LoRA'd layers, by whatever path holds them). A layer that does not hold the chosen adapter runs
//! its base alone; `None` runs every layer's base alone, as PEFT's `disable_adapter()` does. The
//! choice is read when the forward pass builds its graph, so a reference pass and a policy pass
//! may run one after the other on one model, each under its own choice.

use crate::lowrank::{LowRankOptions, low_rank};
use mlx_rs::error::{Exception, Result};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, Param, Parameter};
use mlx_rs::{Array, Dtype};

/// The name of the adapter PEFT's files and [`LoraLinear::with_lora`] use when none is given.
pub use dash_contract::lora::DEFAULT_ADAPTER;

fn contract(message: String) -> Exception {
    Exception::custom(format!("lora: {message}"))
}

/// A low-rank adapter of a [`LoraLinear`], in PEFT's layout.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Lora {
    /// `lora_A`: `[r, in]`, f32; trainable unless frozen.
    #[param]
    pub a: Param<Array>,
    /// `lora_B`: `[out, r]`, f32; trainable unless frozen.
    #[param]
    pub b: Param<Array>,
    /// The factor on the low-rank product: `lora_alpha / r`, or `lora_alpha / √r` with rsLoRA.
    pub scale: f64,
    /// PEFT's `lora_dropout`, applied to the adapter's input only, and only in training mode.
    pub dropout: f32,
    training: bool,
}

impl Lora {
    /// An adapter from its factors, `a` `[r, in]` and `b` `[out, r]`, kept as f32 whatever their
    /// dtype, and the factor on their product.
    pub fn new(a: &Array, b: &Array, scale: f64) -> Result<Self> {
        let (&[r, _], &[_, rb]) = (a.shape(), b.shape()) else {
            return Err(contract(format!(
                "lora_A is [r, in] and lora_B [out, r], got {:?} and {:?}",
                a.shape(),
                b.shape()
            )));
        };
        if r != rb {
            return Err(contract(format!(
                "lora_A's rank {r} and lora_B's {rb} differ"
            )));
        }
        Ok(Self {
            a: Param::new(a.as_dtype(Dtype::Float32)?),
            b: Param::new(b.as_dtype(Dtype::Float32)?),
            scale,
            dropout: 0.0,
            training: false,
        })
    }

    /// The same adapter with PEFT's `lora_dropout` on its input in training mode.
    pub fn with_dropout(mut self, prob: f32) -> Self {
        self.dropout = prob;
        self
    }

    /// The rank `r`.
    pub fn rank(&self) -> i32 {
        self.a.as_ref().shape()[0]
    }

    /// The low-rank term `scale · (x Aᵀ) Bᵀ` in f32, `x` `[.., in]` after the dropout.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.add_to(x, None)
    }

    /// `base + scale · (x Aᵀ) Bᵀ`, the sum in f32 rounded once to the base's dtype (the f32 term
    /// alone without a base), `x` after the dropout.
    fn add_to(&self, x: &Array, base: Option<&Array>) -> Result<Array> {
        let x = match (self.training, self.dropout > 0.0) {
            (true, true) => {
                let keep = 1.0 - self.dropout;
                let mask = mlx_rs::random::bernoulli(&Array::from_f32(keep), x.shape(), None)?;
                mlx_rs::ops::select(
                    &mask,
                    x.divide(Array::from_f32(keep).as_dtype(x.dtype())?)?,
                    Array::from_f32(0.0).as_dtype(x.dtype())?,
                )?
            }
            _ => x.clone(),
        };
        let options = LowRankOptions::default().with_scale(self.scale);
        low_rank(
            &x,
            &self.a.as_ref().t(),
            &self.b.as_ref().t(),
            base,
            options,
        )
    }
}

/// A linear layer with a frozen base weight and any number of named low-rank adapters, one of
/// which (or none) each forward pass adds ([module](self)).
#[derive(Debug, Clone, ModuleParameters)]
pub struct LoraLinear {
    /// `W` `[out, in]`: frozen, at its own precision.
    #[param]
    pub weight: Param<Array>,
    #[param]
    adapters: Vec<Lora>,
    names: Vec<String>,
    active: Option<String>,
}

impl LoraLinear {
    /// A layer over `weight` `[out, in]`, frozen, with no adapter yet; the chosen adapter is
    /// [`DEFAULT_ADAPTER`].
    pub fn frozen(weight: &Array) -> Result<Self> {
        if weight.ndim() != 2 || !weight.dtype().is_float() {
            return Err(contract(format!(
                "the weight must be [out, in] floating point, got {:?} {:?}",
                weight.shape(),
                weight.dtype()
            )));
        }
        let mut weight = Param::new(weight.clone());
        weight.freeze(true);
        Ok(Self {
            weight,
            adapters: Vec::new(),
            names: Vec::new(),
            active: Some(DEFAULT_ADAPTER.into()),
        })
    }

    /// The layer with `lora` as its [`DEFAULT_ADAPTER`].
    pub fn with_lora(self, lora: Lora) -> Result<Self> {
        self.with_adapter(DEFAULT_ADAPTER, lora)
    }

    /// The layer with `lora` under `name`.
    pub fn with_adapter(mut self, name: &str, lora: Lora) -> Result<Self> {
        self.insert_adapter(name, lora)?;
        Ok(self)
    }

    /// Adds or replaces the adapter `name`, returning the one it replaces. Its factors must fit
    /// the weight: `A` `[r, in]`, `B` `[out, r]`.
    pub fn insert_adapter(&mut self, name: &str, lora: Lora) -> Result<Option<Lora>> {
        let (out, inp) = (self.out_features(), self.in_features());
        let (a, b) = (lora.a.as_ref().shape(), lora.b.as_ref().shape());
        if a[1] != inp || b[0] != out {
            return Err(contract(format!(
                "adapter {name:?}: A {a:?} and B {b:?} do not fit a [{out}, {inp}] weight"
            )));
        }
        Ok(match self.names.iter().position(|n| n == name) {
            Some(i) => Some(std::mem::replace(&mut self.adapters[i], lora)),
            None => {
                self.names.push(name.into());
                self.adapters.push(lora);
                None
            }
        })
    }

    /// Removes the adapter `name`.
    pub fn remove_adapter(&mut self, name: &str) -> Option<Lora> {
        let i = self.names.iter().position(|n| n == name)?;
        self.names.remove(i);
        Some(self.adapters.remove(i))
    }

    /// The adapter `name`.
    pub fn adapter(&self, name: &str) -> Option<&Lora> {
        let i = self.names.iter().position(|n| n == name)?;
        Some(&self.adapters[i])
    }

    /// The adapter `name`, to freeze or update.
    pub fn adapter_mut(&mut self, name: &str) -> Option<&mut Lora> {
        let i = self.names.iter().position(|n| n == name)?;
        Some(&mut self.adapters[i])
    }

    /// The adapters by name, in the order they were added.
    pub fn adapters(&self) -> impl Iterator<Item = (&str, &Lora)> {
        self.names.iter().map(String::as_str).zip(&self.adapters)
    }

    /// The chosen adapter's name, or `None` for the base alone.
    pub fn active_adapter(&self) -> Option<&str> {
        self.active.as_deref()
    }

    /// Chooses the adapter the forward pass adds, or `None` for the base alone.
    pub fn set_adapter(&mut self, name: Option<&str>) {
        self.active = name.map(Into::into);
    }

    /// The chosen adapter, if this layer holds it.
    pub fn lora(&self) -> Option<&Lora> {
        self.adapter(self.active.as_deref()?)
    }

    /// `in`.
    pub fn in_features(&self) -> i32 {
        self.weight.as_ref().shape()[1]
    }

    /// `out`.
    pub fn out_features(&self) -> i32 {
        self.weight.as_ref().shape()[0]
    }
}

impl Module<&Array> for LoraLinear {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let weight = self.weight.as_ref();
        if x.shape().last() != Some(&self.in_features()) {
            return Err(contract(format!(
                "x must be [.., {}], got {:?}",
                self.in_features(),
                x.shape()
            )));
        }
        let base = x.as_dtype(weight.dtype())?.matmul(weight.t())?;
        match self.lora() {
            None => base.as_dtype(x.dtype()),
            // The sum is rounded once to the base's dtype; where that is not x's, the base goes in
            // as f32 (exactly) and the f32 sum is rounded once to x's.
            Some(lora) => {
                let base = match base.dtype() == x.dtype() {
                    true => base,
                    false => base.as_dtype(Dtype::Float32)?,
                };
                lora.add_to(x, Some(&base))?.as_dtype(x.dtype())
            }
        }
    }

    fn training_mode(&mut self, mode: bool) {
        for lora in &mut self.adapters {
            lora.training = mode;
        }
    }
}

/// Chooses `name` (or `None`, the bases alone) on every layer `layers` yields: a model's LoRA'd
/// layers, wherever it holds them.
pub fn set_adapter<'a>(layers: impl IntoIterator<Item = &'a mut LoraLinear>, name: Option<&str>) {
    for layer in layers {
        layer.set_adapter(name);
    }
}
