//! A linear layer over weights quantized to 4 bits by group, as GPTQ stores them: `y = x W (+
//! bias)` with `W` dequantized from its packed form, for inference. The weights stay packed in
//! memory; only the activations are a float dtype.
//!
//! ```text
//! g(i)    = i / group                                   the group of input row i
//! q[i, o] = (qweight[i / 8, o] >> 4 (i % 8)) & 15       nibble i % 8 of a word, low bits first
//! W[i, o] = (q[i, o] - zero) · scales[g(i), o]          exact in f32
//! y[n, o] = sum_i x[n, i] W[i, o] (+ bias[o])           [N, out]
//! ```
//!
//! # Layout, as the checkpoint stores it
//!
//! For a projection of `in` inputs to `out` outputs (`in` a multiple of `group`, `out` of 8):
//!
//! - `qweight`: `[in / 8, out]`, `i32`, eight 4-bit values per word packed along `in`, value
//!   `i % 8` in bits `4 (i % 8)` to `4 (i % 8) + 3` of word `i / 8`.
//! - `scales`: `[in / group, out]`, f16, one per group of `group` input rows and output column.
//! - `qzeros`: `[in / group, out / 8]`, `i32`, the groups' zero points packed along `out` the
//!   same way.
//! - `g_idx`: `[in]`, `i32`, each input row's group. Only `g_idx[i] = i / group` is taken (GPTQ's
//!   `desc_act: false`); a reordered one is refused.
//!
//! # Zero points
//!
//! Symmetric quantization (`sym: true`) fixes the zero at `2^(bits - 1)`, 8 for 4 bits, and
//! [`QuantFormat::Symmetric`] takes it so. `qzeros` is then checked rather than read: every
//! stored zero must be 8, which is what a published symmetric checkpoint holds (every word
//! `0x88888888`, not GPTQ v1's `zero - 1`, which would be `0x77777777`); one that is not is
//! refused, so that a checkpoint in another convention cannot be read one scale step off.
//! Asymmetric zeros are not taken yet.
//!
//! # Dtypes and rounding
//!
//! - `x`: `[.., in]`, bf16, f16 or f32, its leading dimensions any flattening of `N` rows.
//! - `bias`: `[out]`, any float dtype, or none.
//! - The output: `[.., out]` in `x`'s dtype.
//!
//! The definition is the product in f32: `q - zero` is a small integer and `scales` an f16 value,
//! so each `W[i, o]` is exact in f32, and `y` is that product with `x` read exactly into f32,
//! accumulated in f32, the bias added in f32 and rounded once to `x`'s dtype. An implementation
//! may round `W` to `x`'s dtype before its products (as the tensor cores need), and the product to
//! `x`'s dtype before the bias is added, and is held to the definition within the bounds its
//! tests record.
//!
//! There is no backward pass yet. A gradient with respect to `x` alone (the weights frozen) is
//! what a quantized model needs as a teacher; it is not asked for.

/// How the packed weights encode their values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantFormat {
    /// GPTQ's symmetric 4-bit format: groups of [`QuantOptions::group`] input rows, the zero fixed
    /// at 8 and checked against `qzeros`.
    #[default]
    Symmetric,
}

/// The arguments of `quantized_linear` that have a usual default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantOptions {
    /// Input rows per group of scales (GPTQ's `group_size`). Default: 128.
    pub group: usize,
    /// The format. Default: [`QuantFormat::Symmetric`].
    pub format: QuantFormat,
}

impl Default for QuantOptions {
    fn default() -> Self {
        Self {
            group: 128,
            format: QuantFormat::Symmetric,
        }
    }
}

impl QuantOptions {
    /// The options with groups of `group` input rows.
    pub fn with_group(self, group: usize) -> Self {
        Self { group, ..self }
    }

    /// The zero point of [`QuantFormat::Symmetric`]: `2^(bits - 1)` for 4 bits.
    pub const SYMMETRIC_ZERO: u32 = 8;
}
