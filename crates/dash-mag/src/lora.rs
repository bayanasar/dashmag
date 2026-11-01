//! LoRA's layer, written once over [`crate::ops::low_rank`]: a frozen base at its own precision,
//! f32 adapters never folded into it, several by name over one base, the choice of one, and PEFT's
//! files through [`dash_peft`].
//!
//! `y = x Wᵀ + scale · (x Aᵀ) Bᵀ`, with PEFT's layout and arithmetic, as dash-lora and dash-mlx
//! compute it: the base product at the weight's dtype, the low-rank product in f32, their sum in
//! f32 and rounded once to `x`'s dtype. A layer holds plain tensors, so a training step passes an
//! adapter's factors to [`crate::vjp`] as primals and puts them back with [`LoraLinear::set_factors`];
//! the base, which `f` captures, stays a constant.
//!
//! PEFT's `lora_dropout` is not here: an adapter that asks for it is refused when read.

use std::collections::BTreeMap;

pub use dash_contract::lora::DEFAULT_ADAPTER;
pub use dash_peft::{
    AdapterLayer, ApplyOptions, FactorDtype, HostTensor, LoraWeights, PeftAdapter, PeftConfig,
    PeftError,
};

use crate::arm::{Arm, Selected};
use crate::ops::{LowRankOptions, low_rank};
use crate::tensor::Tensor;
use crate::{DType, Result, refuse};

/// One adapter: `a` `[r, in]` and `b` `[out, r]` in f32, and the factor on their product.
#[derive(Clone, Debug)]
pub struct Lora {
    pub a: Tensor,
    pub b: Tensor,
    pub scale: f64,
}

impl Lora {
    /// An adapter from its factors, cast to f32.
    pub fn new(a: &Tensor, b: &Tensor, scale: f64) -> Result<Self> {
        let (sa, sb) = (a.shape(), b.shape());
        if sa.len() != 2 || sb.len() != 2 || sa[0] != sb[1] {
            return refuse(format!(
                "Lora: lora_A is [r, in] and lora_B is [out, r], got {sa:?} and {sb:?}"
            ));
        }
        Ok(Self {
            a: a.as_dtype(DType::F32)?,
            b: b.as_dtype(DType::F32)?,
            scale,
        })
    }

    pub fn rank(&self) -> usize {
        self.a.shape()[0]
    }
}

/// A linear layer: its frozen weight `[out, in]`, and its adapters by name.
#[derive(Clone, Debug)]
pub struct LoraLinear {
    weight: Tensor,
    adapters: BTreeMap<String, Lora>,
    active: Option<String>,
}

impl LoraLinear {
    /// A layer with no adapter, running [`DEFAULT_ADAPTER`] once it has one.
    pub fn frozen(weight: &Tensor) -> Result<Self> {
        if weight.rank() != 2 {
            return refuse(format!("LoraLinear: a weight of {:?}", weight.shape()));
        }
        Ok(Self {
            weight: weight.stop_gradient()?,
            adapters: BTreeMap::new(),
            active: Some(DEFAULT_ADAPTER.to_string()),
        })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn in_features(&self) -> usize {
        self.weight.shape()[1]
    }

    pub fn out_features(&self) -> usize {
        self.weight.shape()[0]
    }

    /// Adds `lora` as the adapter `name`, in the place of one of that name, and returns that one.
    pub fn insert_adapter(&mut self, name: &str, lora: Lora) -> Result<Option<Lora>> {
        let (a, b) = (lora.a.shape(), lora.b.shape());
        if a[1] != self.in_features() || b[0] != self.out_features() {
            return refuse(format!(
                "LoraLinear: factors {a:?} and {b:?} over a weight of {:?}",
                self.weight.shape()
            ));
        }
        Ok(self.adapters.insert(name.to_string(), lora))
    }

    pub fn remove_adapter(&mut self, name: &str) -> Option<Lora> {
        self.adapters.remove(name)
    }

    pub fn adapter(&self, name: &str) -> Option<&Lora> {
        self.adapters.get(name)
    }

    /// Runs the adapter `name` from here on, or the base alone with `None`, or if it holds none of
    /// that name.
    pub fn set_adapter(&mut self, name: Option<&str>) {
        self.active = name.map(str::to_string);
    }

    /// The adapter a forward pass runs, if any.
    pub fn lora(&self) -> Option<&Lora> {
        self.adapters.get(self.active.as_deref()?)
    }

    /// Replaces the factors of the adapter a forward pass runs, as `vjp`'s primals or an
    /// optimiser's step give them.
    pub fn set_factors(&mut self, a: Tensor, b: Tensor) -> Result<()> {
        let Some(active) = self.active.as_deref() else {
            return refuse("LoraLinear: no adapter is chosen");
        };
        let Some(lora) = self.adapters.get_mut(active) else {
            return refuse(format!("LoraLinear: no adapter {active}"));
        };
        if lora.a.shape() != a.shape() || lora.b.shape() != b.shape() {
            return refuse(format!(
                "LoraLinear: factors {:?} and {:?} for {:?} and {:?}",
                a.shape(),
                b.shape(),
                lora.a.shape(),
                lora.b.shape()
            ));
        }
        lora.a = a;
        lora.b = b;
        Ok(())
    }

    /// Replaces the frozen base, as a generator with an adapter folded into its weights holds it: of
    /// the shape the layer has.
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.shape() != self.weight.shape() || !weight.dtype().is_floating() {
            return refuse(format!(
                "set_weight: {:?} {:?} for {:?}",
                weight.shape(),
                weight.dtype(),
                self.weight.shape()
            ));
        }
        self.weight = weight;
        Ok(())
    }

    /// `x` `[.., in]` to `[.., out]`, at `x`'s dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let base = |x: &Tensor| x.as_dtype(self.weight.dtype())?.matmul(&self.weight.t()?);
        let Some(lora) = self.lora() else {
            return base(x)?.as_dtype(dtype);
        };
        let shape = x.shape();
        if shape.len() != 2 && Selected::LORA_AS_ROWS {
            // As rows, so that the base product and the low-rank one read the same tensor.
            let rows = x.reshape(&[-1, self.in_features() as isize])?;
            let y = self.forward(&rows)?;
            let mut out: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
            *out.last_mut().expect("rank 1 or more") = self.out_features() as isize;
            return y.reshape(&out);
        }
        // The sum is rounded once, to x's dtype: a base in another dtype is added in f32.
        let product = base(x)?;
        let product = if product.dtype() == dtype {
            product
        } else {
            product.as_dtype(DType::F32)?
        };
        let options = LowRankOptions::default().with_scale(lora.scale);
        low_rank(x, &lora.a.t()?, &lora.b.t()?, Some(&product), options)?.as_dtype(dtype)
    }
}

fn peft(e: crate::Error) -> PeftError {
    PeftError::Invalid(e.0)
}

/// A factor as an f32 tensor: bf16 and f16 widened exactly, f64 rounded to f32.
fn tensor(host: &HostTensor) -> Result<Tensor> {
    let dtype = match host.dtype() {
        FactorDtype::F32 => DType::F32,
        FactorDtype::Bf16 => DType::BF16,
        FactorDtype::F16 => DType::F16,
        FactorDtype::F64 => DType::F64,
    };
    Tensor::from_bytes(host.bytes(), host.shape(), dtype)?.as_dtype(DType::F32)
}

fn host(t: &Tensor) -> HostTensor {
    let values = t.to_f32().expect("an f32 factor");
    HostTensor::from_f32(t.shape(), &values).expect("a tensor holds its shape's elements")
}

impl AdapterLayer for LoraLinear {
    type Lora = Lora;
    type Device = ();

    fn in_features(&self) -> usize {
        LoraLinear::in_features(self)
    }

    fn out_features(&self) -> usize {
        LoraLinear::out_features(self)
    }

    /// Whether an adapter trains is the caller's here: what it passes to [`crate::vjp`].
    fn make_lora(
        weights: &LoraWeights,
        scale: f64,
        dropout: f64,
        _trainable: bool,
        _: &(),
    ) -> std::result::Result<Lora, PeftError> {
        if dropout != 0.0 {
            return Err(PeftError::Invalid(format!(
                "lora_dropout {dropout}: dash-mag's LoRA has no dropout"
            )));
        }
        let (a, b) = (
            tensor(&weights.a).map_err(peft)?,
            tensor(&weights.b).map_err(peft)?,
        );
        Lora::new(&a, &b, scale).map_err(peft)
    }

    fn put_lora(&mut self, name: &str, lora: Lora) -> std::result::Result<(), PeftError> {
        self.insert_adapter(name, lora)
            .map(|_| ())
            .map_err(|e| PeftError::Mismatch(e.0))
    }

    fn drop_lora(&mut self, name: &str) {
        self.remove_adapter(name);
    }

    fn lora_weights(&self, name: &str) -> Option<(LoraWeights, f64)> {
        let lora = self.adapter(name)?;
        let weights = LoraWeights {
            a: host(&lora.a),
            b: host(&lora.b),
        };
        Some((weights, lora.scale))
    }
}
