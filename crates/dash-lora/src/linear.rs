//! A linear layer whose frozen base weight keeps its own precision and whose LoRA adapters are
//! applied beside it, never folded into it.
//!
//! `y = x Wᵀ + scale · (x Aᵀ) Bᵀ`, with `W` `[out, in]` frozen (bf16, or f32), `A` `[r, in]` and
//! `B` `[out, r]` in f32, and `scale = lora_alpha / r` (`lora_alpha / √r` with rsLoRA): PEFT's
//! layout, names and arithmetic. A layer holds any number of adapters by name over its one `W`,
//! and each forward pass adds the term of one of them, or of none ([`crate::adapters`]). The
//! precision of each part:
//!
//! - **Base product** `x Wᵀ`, at the weight's dtype: `x` is cast to it if it differs (it does
//!   not when `x` and `W` are both bf16, PEFT's case). For bf16, Flex converts both operands to
//!   f32, multiplies in f32 and rounds each output once to bf16; CUDA (cubek) accumulates in f32
//!   and writes bf16. cuBLAS, under PyTorch, does the same.
//! - **Low-rank product**, f32 throughout ([`dash_ops::lowrank`]): `x` read exactly into f32,
//!   `x Aᵀ` `[.., r]` and then `· Bᵀ` `[.., out]` as f32 products, then `· scale` in f32, in that
//!   order, as PEFT computes `lora_B(lora_A(x)) * scaling`. On a CubeCL device these are kernels
//!   of their own, which read `x` in its own dtype and sum each element in order (Burn's default
//!   matrix-product kernels run these skinny shapes several times slower than cuBLAS); elsewhere,
//!   Burn's matrix products. On CUDA, f32 products are IEEE f32 only with TF32 off
//!   (`CUBECL_CUDA_TF32=0` in this workspace); Burn 0.22 otherwise runs its own in TF32 on compute
//!   capability 8.0 and later, while PyTorch's default is IEEE f32.
//! - **Sum**, in f32: the base product upcast (exact) plus the low-rank term, one f32 addition,
//!   then rounded once to `x`'s dtype. With bf16 `x` this is PEFT's
//!   `(result + lora).to(torch_result_dtype)`. On a CubeCL device the kernel computing `· Bᵀ`
//!   adds the base and rounds, so the f32 `[.., out]` term is never held. With no adapter chosen,
//!   the base product alone, at `x`'s dtype, as PEFT's layer returns it with its adapters
//!   disabled.
//!
//! Against PEFT 0.20's `lora.Linear` on PyTorch 2.13 (`tests/linear.rs`), the output and the
//! gradients of `x`, `A` and `B` are bit-identical, for a bf16 and for an f32 base, on Flex and
//! on an RTX 5070 Ti with Burn's default kernels or with `burn/fusion`. With `burn/autotune`,
//! CUDA may choose a base matrix-product kernel that sums in another order, and then f32 results
//! move by a few roundings and a bf16 output element by one unit.
//!
//! The adapter is never merged into `W`, in training or in inference. A merged bf16 weight
//! `bf16(W + scale · B A)` keeps of the update only what survives rounding to `W`'s grid, whose
//! spacing near a typical weight is larger than most of the update's elements: most of the
//! update rounds away (`tests/linear.rs`: for an update with a thousandth of `W`'s spread, 83%
//! of its elements round to zero and what is left has a cosine of 0.575 with it). Kept apart,
//! the update reaches the output at f32 precision.
//!
//! Gradients reach the factors of the chosen adapter, if they require one, and nothing else.
//! `W` is a [`Param`] that does not require a gradient, so Burn's autodiff never computes one
//! for it: its matrix product records `W` as a constant, keeps only what the input's gradient
//! needs (`W` itself, not `x`), and its backward computes `∂L/∂x = ∂L/∂y · W` alone. A frozen
//! adapter's factors are constants the same way.
//!
//! Burn 0.22.1 has a LoRA of its own, `burn::module::LoraAdapter`, a reparameterization of a
//! `Param` that materializes `W + scale · (a b)` on every read. It does not fit here: it is the
//! merged form, computed at the factors' precision, so each forward makes an f32 copy of every
//! adapted weight and multiplies by it instead of by the bf16 weight; its backward computes the
//! gradient of that whole effective weight, `xᵀ ∂L/∂y`, `[in, out]` and as costly as the frozen
//! weight's own gradient, before projecting it onto the factors; its factors are laid out
//! `[in, r]` and `[r, out]`, the transposes of PEFT's; and a parameter carries one at most.

use burn::module::{Module, Param, ParamId};
use burn::nn::{Dropout, DropoutConfig};
use burn::tensor::{DType, Device, Tensor, TensorData};
use dash_ops::lowrank::{LowRankOptions, low_rank};

use crate::adapters::{Adapters, DEFAULT_ADAPTER};

/// A low-rank adapter of a [`LoraLinear`], in PEFT's layout.
#[derive(Module, Debug)]
pub struct Lora {
    /// `lora_A`: `[r, in]`, f32; trainable unless frozen (Burn's `Module::no_grad`).
    pub a: Param<Tensor<2>>,
    /// `lora_B`: `[out, r]`, f32; trainable unless frozen.
    pub b: Param<Tensor<2>>,
    /// The factor on the low-rank product: `lora_alpha / r`, or `lora_alpha / √r` with rsLoRA.
    pub scale: f64,
    /// PEFT's `lora_dropout`, applied to the adapter's input only, and only while training
    /// (on an autodiff device, outside `valid()`). A layer with dropout is random: it cannot be
    /// run again identically by [`dash_ops::checkpoint`].
    pub dropout: Dropout,
}

impl Lora {
    /// An adapter from its factors, `a` `[r, in]` and `b` `[out, r]`, stored as f32 trainable
    /// parameters whatever their dtype, and the factor on their product. `.no_grad()` (Burn's
    /// `Module::no_grad`) freezes it: its factors then record nothing and get no gradient.
    pub fn new(a: Tensor<2>, b: Tensor<2>, scale: f64) -> Self {
        let [r, _] = a.dims();
        let [_, rb] = b.dims();
        assert_eq!(
            r, rb,
            "lora_A is [r, in] and lora_B is [out, r]: their ranks differ"
        );
        Self {
            a: Param::from_tensor(a.cast(DType::F32).detach()),
            b: Param::from_tensor(b.cast(DType::F32).detach()),
            scale,
            dropout: DropoutConfig::new(0.0).init(),
        }
    }

    /// The same adapter, with PEFT's `lora_dropout` on its input during training.
    pub fn with_dropout(mut self, prob: f64) -> Self {
        self.dropout = DropoutConfig::new(prob).init();
        self
    }

    /// An adapter from host data, as read from a PEFT file: `a` `[r, in]`, `b` `[out, r]`.
    pub fn from_data(a: TensorData, b: TensorData, scale: f64, device: &Device) -> Self {
        Self::new(
            Tensor::from_data(a, (device, DType::F32)),
            Tensor::from_data(b, (device, DType::F32)),
            scale,
        )
    }

    /// The rank `r`.
    pub fn rank(&self) -> usize {
        self.a.val().dims()[0]
    }

    /// `scale · (x Aᵀ) Bᵀ`, in f32, for `x` `[.., in]` of any float dtype.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.low_rank(x, None)
    }

    /// `base + scale · (x Aᵀ) Bᵀ` ([`dash_ops::lowrank::low_rank`]): the sum in f32, rounded once
    /// to the base's dtype; without a base, the low-rank term in f32.
    fn low_rank<const D: usize>(&self, x: Tensor<D>, base: Option<Tensor<D>>) -> Tensor<D> {
        // The products read `x` in its own dtype, exactly; dropout, where there is one, acts on
        // the f32 input PEFT's acts on.
        let x = match self.dropout.prob {
            0.0 => x,
            _ => self.dropout.forward(x.cast(DType::F32)),
        };
        // The factors as matrix products take them, `x Aᵀ` then `· Bᵀ`: views, as Burn's own
        // `x.matmul(a.transpose())` takes them, and so recorded where its product would be.
        let u = self.a.val().transpose().unsqueeze();
        let v = self.b.val().transpose().unsqueeze();
        let options = LowRankOptions::default().with_scale(self.scale);
        low_rank(x, u, v, base, options)
    }
}

/// A linear layer with a frozen base weight and any number of LoRA adapters by name, of which
/// each forward pass uses the chosen one, or none. No bias: the projections LoRA targets in
/// today's language models have none.
///
/// The layer uses [`DEFAULT_ADAPTER`] until [`AdapterChoice::set_adapter`] chooses another for
/// it, or for the whole module tree it is part of; a layer that does not hold the chosen
/// adapter runs its base alone.
///
/// [`AdapterChoice::set_adapter`]: crate::AdapterChoice::set_adapter
#[derive(Module, Debug)]
pub struct LoraLinear {
    /// The base weight, `[out, in]` (PyTorch's layout), kept at the dtype it was given in
    /// (bf16 or f32). It never requires a gradient.
    pub weight: Param<Tensor<2>>,
    adapters: Adapters,
}

impl LoraLinear {
    /// A frozen layer from its base weight `[out, in]`, at the weight's own dtype, without an
    /// adapter. The weight is detached from any graph and never requires a gradient.
    pub fn frozen(weight: Tensor<2>) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), weight.detach()),
            adapters: Adapters::new(),
        }
    }

    /// A frozen layer from host data `[out, in]`, kept at the data's dtype.
    pub fn frozen_from_data(weight: TensorData, device: &Device) -> Self {
        let dtype = weight.dtype();
        Self::frozen(Tensor::from_data(weight, (device, dtype)))
    }

    /// The same layer with `lora` as its adapter [`DEFAULT_ADAPTER`], the one it uses unless
    /// another is chosen: a layer's single adapter. Panics if the shapes do not fit the weight.
    pub fn with_lora(self, lora: Lora) -> Self {
        self.with_adapter(DEFAULT_ADAPTER, lora)
    }

    /// Makes `lora` the layer's adapter [`DEFAULT_ADAPTER`], in the place of the one it held:
    /// [`with_lora`](Self::with_lora) in place.
    pub fn set_lora(&mut self, lora: Lora) {
        self.insert_adapter(DEFAULT_ADAPTER, lora);
    }

    /// The same layer with `lora` as its adapter named `name`, added beside the others or in the
    /// place of the one of that name. Panics if the shapes do not fit the weight, or if `name` is
    /// empty or has a `.`.
    pub fn with_adapter(mut self, name: &str, lora: Lora) -> Self {
        self.insert_adapter(name, lora);
        self
    }

    /// Adds `lora` as the adapter named `name`, or puts it in the place of the one of that name,
    /// which it returns. Panics as [`with_adapter`](Self::with_adapter) does.
    pub fn insert_adapter(&mut self, name: &str, lora: Lora) -> Option<Lora> {
        let [out, inp] = self.weight.val().dims();
        let [_, a_in] = lora.a.val().dims();
        let [b_out, _] = lora.b.val().dims();
        assert_eq!(
            (b_out, a_in),
            (out, inp),
            "the adapter is [out, r] x [r, in] = [{b_out}, {a_in}], the weight [{out}, {inp}]"
        );
        self.adapters.insert(name, lora)
    }

    /// Removes the adapter named `name` and returns it. If it was the chosen one, the layer runs
    /// its base alone until it holds one of that name again or another is chosen.
    pub fn remove_adapter(&mut self, name: &str) -> Option<Lora> {
        self.adapters.remove(name)
    }

    /// The adapter named `name`, if the layer holds one.
    pub fn adapter(&self, name: &str) -> Option<&Lora> {
        self.adapters.get(name)
    }

    /// The layer's adapters by name, in the order they were added.
    pub fn adapters(&self) -> impl Iterator<Item = (&str, &Lora)> {
        self.adapters.iter()
    }

    /// The name of the adapter the layer's forward pass uses where it holds it:
    /// [`DEFAULT_ADAPTER`] unless another was chosen; `None` if the layer runs its base alone.
    pub fn active_adapter(&self) -> Option<&str> {
        self.adapters.active_name()
    }

    /// The adapter the forward pass uses: the chosen one, if the layer holds it.
    pub fn lora(&self) -> Option<&Lora> {
        self.adapters.active()
    }

    /// Input features.
    pub fn in_features(&self) -> usize {
        self.weight.val().dims()[1]
    }

    /// Output features.
    pub fn out_features(&self) -> usize {
        self.weight.val().dims()[0]
    }

    /// `x Wᵀ + scale · (x Aᵀ) Bᵀ` with the chosen adapter's factors, or `x Wᵀ` if the layer
    /// holds none of that name, for `x` `[.., in]`, returned at `x`'s dtype; the precision of
    /// each part is in the [module documentation](self).
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dtype = x.dtype();
        let weight = self.weight.val();
        let base = |x: Tensor<D>| {
            x.cast(weight.dtype())
                .matmul(weight.clone().transpose().unsqueeze())
        };
        let Some(lora) = self.lora() else {
            return base(x).cast(dtype);
        };
        if D != 2 {
            // As rows, so that the base product and the low-rank one read the same tensor.
            let dims = x.dims();
            let rows = dims[..D - 1].iter().product();
            let y: Tensor<2> = self.forward(x.reshape([rows, dims[D - 1]]));
            let mut out = dims;
            out[D - 1] = self.out_features();
            return y.reshape(out);
        }
        // The sum is rounded once, to x's dtype: a base in another dtype is added in f32.
        let base = base(x.clone());
        let base = match base.dtype() == dtype {
            true => base,
            false => base.cast(DType::F32),
        };
        lora.low_rank(x, Some(base)).cast(dtype)
    }
}
