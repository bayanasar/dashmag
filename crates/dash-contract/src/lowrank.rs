//! Low-rank products: `y = base + scale · (x U) V` for factors `U` `[in, r]` and `V` `[r, out]` of
//! small rank `r`, with their gradients. LoRA's term is this (`U` and `V` PEFT's `lora_A` and
//! `lora_B` weights transposed, `scaling = lora_alpha / r`, or `/ √r` with rsLoRA), as are the
//! adapters built on it (DoRA's low-rank part, LoRA-FA, PiSSA's residual, QLoRA's over a
//! dequantized base) and a linear layer factored into two (`base` absent, `scale` 1).
//!
//! ```text
//! h[n, j] = sum_k x[n, k] U[k, j]                      [N, r], f32
//! y[n, m] = base[n, m] + scale · sum_j h[n, j] V[j, m]  [N, out]
//! ```
//!
//! # Shapes and dtypes
//!
//! - `x`: `[.., in]`, f32, bf16 or f16, its leading dimensions any flattening of `N` rows.
//! - `U` `[in, r]` and `V` `[r, out]`: f32, as an adapter's factors are kept whatever the model's
//!   dtype (PEFT's default for a bf16 model), with any strides: a transposed view of PEFT's
//!   `[r, in]` and `[out, r]` is read in place.
//! - `base`: `[.., out]` with the leading dimensions of `x`, in any float dtype, or none.
//! - The output: `[.., out]`, the sum computed in f32 and rounded once to the base's dtype, as
//!   PEFT's `(result + lora).to(result.dtype)` does; without a base, `scale · (x U) V` in f32.
//!
//! Every product is f32, in IEEE f32: `x` is read exactly into f32, and the products and their
//! sums are f32 as PEFT's on an f32 adapter. The gradients: `dx` in `x`'s dtype, rounded once
//! from f32; `dU` and `dV` f32; the base's, the output's gradient itself. `dz = dy · scale` is
//! rounded to f32 before it enters the products, as autodiff of `(h V) · scale` computes it.

/// The arguments of `low_rank` that have a usual default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LowRankOptions {
    /// The factor on `(x U) V`, applied in f32 before the base is added: LoRA's
    /// `lora_alpha / r` (`lora_alpha / √r` with rsLoRA). Default: 1.
    pub scale: f64,
}

impl Default for LowRankOptions {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

impl LowRankOptions {
    /// The options with `scale`.
    pub fn with_scale(self, scale: f64) -> Self {
        Self { scale }
    }
}
