//! A linear layer whose frozen base weight keeps its own precision and whose LoRA adapter is
//! applied beside it, never folded into it.
//!
//! `y = x Wᵀ + scale · (x Aᵀ) Bᵀ`, with `W` `[out, in]` frozen (bf16, or f32), `A` `[r, in]` and
//! `B` `[out, r]` in f32, and `scale = lora_alpha / r` (`lora_alpha / √r` with rsLoRA): PEFT's
//! layout, names and arithmetic. The precision of each part:
//!
//! - **Base product** `x Wᵀ`, at the weight's dtype: `x` is cast to it if it differs (it does
//!   not when `x` and `W` are both bf16, PEFT's case). For bf16, Flex converts both operands to
//!   f32, multiplies in f32 and rounds each output once to bf16; CUDA (cubek) accumulates in f32
//!   and writes bf16. cuBLAS, under PyTorch, does the same.
//! - **Low-rank product**, f32 throughout: `x` upcast to f32 (exact from bf16), `x Aᵀ` `[.., r]`
//!   and then `· Bᵀ` `[.., out]` as f32 matrix products, then `· scale` in f32, in that order, as
//!   PEFT computes `lora_B(lora_A(x)) * scaling`. On CUDA, f32 products are IEEE f32 only with
//!   TF32 off (`CUBECL_CUDA_TF32=0` in this workspace); Burn 0.22 otherwise runs them in TF32 on
//!   compute capability 8.0 and later, while PyTorch's default is IEEE f32.
//! - **Sum**, in f32: the base product upcast (exact) plus the low-rank term, one f32 addition,
//!   then rounded once to `x`'s dtype. With bf16 `x` this is PEFT's
//!   `(result + lora).to(torch_result_dtype)`.
//!
//! Against PEFT 0.20's `lora.Linear` on PyTorch 2.13 (`tests/linear.rs`), the output and the
//! gradients of `x`, `A` and `B` are bit-identical, for a bf16 and for an f32 base, on Flex and
//! on an RTX 5070 Ti with Burn's default kernels or with `burn/fusion`. With `burn/autotune`,
//! CUDA may choose a matrix-product kernel that sums in another order, and then f32 results
//! move by a few roundings and a bf16 output element by one unit.
//!
//! The adapter is never merged into `W`, in training or in inference. A merged bf16 weight
//! `bf16(W + scale · B A)` keeps of the update only what survives rounding to `W`'s grid, whose
//! spacing near a typical weight is larger than most of the update's elements: most of the
//! update rounds away (`tests/linear.rs`: for an update with a thousandth of `W`'s spread, 83%
//! of its elements round to zero and what is left has a cosine of 0.575 with it). Kept apart,
//! the update reaches the output at f32 precision.
//!
//! Gradients reach `A` and `B` only. `W` is a [`Param`] that does not require a gradient, so
//! Burn's autodiff never computes one for it: its matrix product records `W` as a constant,
//! keeps only what the input's gradient needs (`W` itself, not `x`), and its backward computes
//! `∂L/∂x = ∂L/∂y · W` alone.
//!
//! Burn 0.22.1 has a LoRA of its own, `burn::module::LoraAdapter`, a reparameterization of a
//! `Param` that materializes `W + scale · (a b)` on every read. It does not fit here: it is the
//! merged form, computed at the factors' precision, so each forward makes an f32 copy of every
//! adapted weight and multiplies by it instead of by the bf16 weight; its backward computes the
//! gradient of that whole effective weight, `xᵀ ∂L/∂y`, `[in, out]` and as costly as the frozen
//! weight's own gradient, before projecting it onto the factors; and its factors are laid out
//! `[in, r]` and `[r, out]`, the transposes of PEFT's.

use burn::module::{Module, Param, ParamId};
use burn::nn::{Dropout, DropoutConfig};
use burn::tensor::{DType, Device, Tensor, TensorData};

/// The low-rank adapter of a [`LoraLinear`], in PEFT's layout.
#[derive(Module, Debug)]
pub struct Lora {
    /// `lora_A`: `[r, in]`, f32, trainable.
    pub a: Param<Tensor<2>>,
    /// `lora_B`: `[out, r]`, f32, trainable.
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
    /// parameters whatever their dtype, and the factor on their product.
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
        let x = self.dropout.forward(x.cast(DType::F32));
        let h = x.matmul(self.a.val().transpose().unsqueeze());
        h.matmul(self.b.val().transpose().unsqueeze())
            .mul_scalar(self.scale)
    }
}

/// A linear layer with a frozen base weight and, optionally, a LoRA adapter. No bias: the
/// projections LoRA targets in today's language models have none.
#[derive(Module, Debug)]
pub struct LoraLinear {
    /// The base weight, `[out, in]` (PyTorch's layout), kept at the dtype it was given in
    /// (bf16 or f32). It never requires a gradient.
    pub weight: Param<Tensor<2>>,
    /// The adapter, if this layer has one.
    pub lora: Option<Lora>,
}

impl LoraLinear {
    /// A frozen layer from its base weight `[out, in]`, at the weight's own dtype, without an
    /// adapter. The weight is detached from any graph and never requires a gradient.
    pub fn frozen(weight: Tensor<2>) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), weight.detach()),
            lora: None,
        }
    }

    /// A frozen layer from host data `[out, in]`, kept at the data's dtype.
    pub fn frozen_from_data(weight: TensorData, device: &Device) -> Self {
        let dtype = weight.dtype();
        Self::frozen(Tensor::from_data(weight, (device, dtype)))
    }

    /// The same layer with `lora` as its adapter. Panics if the shapes do not fit the weight.
    pub fn with_lora(mut self, lora: Lora) -> Self {
        let [out, inp] = self.weight.val().dims();
        let [_, a_in] = lora.a.val().dims();
        let [b_out, _] = lora.b.val().dims();
        assert_eq!(
            (b_out, a_in),
            (out, inp),
            "the adapter is [out, r] x [r, in] = [{b_out}, {a_in}], the weight [{out}, {inp}]"
        );
        self.lora = Some(lora);
        self
    }

    /// Input features.
    pub fn in_features(&self) -> usize {
        self.weight.val().dims()[1]
    }

    /// Output features.
    pub fn out_features(&self) -> usize {
        self.weight.val().dims()[0]
    }

    /// `x Wᵀ + scale · (x Aᵀ) Bᵀ` for `x` `[.., in]`, returned at `x`'s dtype; the precision of
    /// each part is in the [module documentation](self).
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dtype = x.dtype();
        let weight = self.weight.val();
        let base = |x: Tensor<D>| {
            x.cast(weight.dtype())
                .matmul(weight.clone().transpose().unsqueeze())
        };
        match &self.lora {
            None => base(x).cast(dtype),
            Some(lora) => base(x.clone())
                .cast(DType::F32)
                .add(lora.forward(x))
                .cast(dtype),
        }
    }
}
