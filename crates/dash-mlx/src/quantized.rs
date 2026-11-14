//! Weight-only int4 weights as GPTQ stores them, in the layout MLX's quantized matmul takes.
//!
//! GPTQ's symmetric int4 (`bits 4`, `group_size 128`, `sym: true`, `desc_act: false`) stores a
//! projection of `in → out` as `qweight`, I32 `[in/8, out]`, and `scales`, F16 `[in/128, out]`.
//! Each word of `qweight` holds eight 4-bit values `q` along `in`, low bits first. The weight is
//! `w[o, i] = (q[i, o] - 8) · scales[i / 128, o]`. `qzeros` and `g_idx` are not read: with
//! `sym: true` the zero is `2^(bits - 1) = 8`, and with `desc_act: false` row `i`'s group is
//! `i / 128`. Whoever loads a checkpoint checks them.
//!
//! MLX's affine quantization holds `w = scales · q + biases` for each group. It packs the same
//! eight values of `in`, in the same order, along the last axis of a U32 `[out, in/8]`, with
//! `scales` and `biases` `[out, in/128]`. So [`gptq_sym4_to_mlx`] transposes `qweight`'s words
//! without changing them, transposes the scales, and sets `biases = -8 · scales`, which is exact
//! wherever `8 · scales` is finite.
//!
//! MLX's quantized matmul takes `x`, the scales and the biases in the dtype they promote to, and
//! gives its output in it, so the scales and biases take `x`'s dtype: bf16 `x` with f16 scales
//! would run in f32. Its sums are f32 either way (MLX 0.32.4's Metal kernels): below a row count
//! that depends on the GPU and the shape (6 to 33 rows), it reads `x` into f32 and never forms
//! `W`; from there on, it dequantizes `W`'s tiles into that dtype and multiplies them with f32
//! accumulation. What is in `x`'s dtype is the scales, the biases, the output and, past that row
//! count, `W`. On an Apple M6,
//! bf16 `x` `[2048, 5120]` against a `5120 → 17408` projection took 85.7 ms in f32 (f32 scales),
//! against 20.0 ms with bf16 scales: 4.3 times. Over 512 and 2048 rows of that projection and of
//! `17408 → 5120`, f32 scales took 2.9 to 4.3 times as long. (Medians, measured 2026-11-14 with
//! MLX 0.32.4. The M6 was not idle: only terminals were open, at a load average of 1.07–1.37.)
//!
//! MLX's dequantize of the result gives `(q - 8) · scale` bit for bit with f32 scales, and that
//! value rounded once with f16 scales. bf16 scales are the f16 ones rounded once, so with them
//! each weight is within one bf16 unit of `(q - 8) · scale` rounded once (`tests/quantized.rs`).

use crate::mlx::error::{Exception, Result};
use crate::mlx::ops::quantized_matmul;
use crate::mlx::{Array, Dtype};

/// GPTQ's group size: the rows of `in` that share a scale.
pub const GROUP_SIZE: i32 = 128;

/// GPTQ's bits per weight.
pub const BITS: i32 = 4;

fn contract(message: String) -> Exception {
    Exception::custom(format!("gptq_sym4_to_mlx: {message}"))
}

/// GPTQ's symmetric int4 `qweight` (I32 `[in/8, out]`) and `scales` (`[in/128, out]`, F16 as GPTQ
/// stores them) as MLX's affine `w` (U32 `[out, in/8]`), `scales` and `biases` (`[out, in/128]` in
/// `dtype`), for `quantized_matmul(x, w, scales, biases, transpose = true,`
/// [`GROUP_SIZE`]`,` [`BITS`]`, "affine")` ([module](self)). `dtype` should be `x`'s.
///
/// Errors if `qweight` is not I32 with two axes and a multiple of 16 rows, if `scales` is not
/// f32, f16 or bf16 of shape `[in/128, out]` for that `qweight`, or if `dtype` is not one of those.
pub fn gptq_sym4_to_mlx(
    qweight: &Array,
    scales: &Array,
    dtype: Dtype,
) -> Result<(Array, Array, Array)> {
    let &[words, out] = qweight.shape() else {
        return Err(contract(format!(
            "qweight must be [in/8, out], got {:?}",
            qweight.shape()
        )));
    };
    if qweight.dtype() != Dtype::Int32 {
        return Err(contract(format!(
            "qweight must be I32, got {:?}",
            qweight.dtype()
        )));
    }
    let per_group = GROUP_SIZE * BITS / 32;
    if words == 0 || words % per_group != 0 {
        return Err(contract(format!(
            "in must be a positive multiple of {GROUP_SIZE}, got qweight's {words} rows of eight"
        )));
    }
    if scales.shape() != [words / per_group, out] {
        return Err(contract(format!(
            "scales must be [in/{GROUP_SIZE}, out] = [{}, {out}], got {:?}",
            words / per_group,
            scales.shape()
        )));
    }
    let float = |d: Dtype| matches!(d, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16);
    if !float(scales.dtype()) || !float(dtype) {
        return Err(contract(format!(
            "the scales and dtype must be f32, f16 or bf16, got {:?} and {dtype:?}",
            scales.dtype()
        )));
    }
    let w = qweight.view(Dtype::Uint32)?.t().contiguous()?;
    let scales = scales.as_dtype(dtype)?.t().contiguous()?;
    let biases = scales.multiply(Array::from_f32(-8.0).as_dtype(dtype)?)?;
    Ok((w, scales, biases))
}

/// `y = x W (+ bias)` over a weight in MLX's layout: `x` `[.., in]` in f32, bf16 or f16, `w` U32
/// `[out, in/8]` and `scales` `[out, in/128]` as [`gptq_sym4_to_mlx`] gives them, the scales in any
/// of those dtypes, and `bias` `[out]` in f32, bf16 or f16, or none. The output is `[.., out]` in
/// `x`'s dtype.
///
/// The scales are cast to `x`'s dtype and the biases made from them, `-8 · scales`, on each call,
/// so that MLX's quantized matmul takes everything in `x`'s dtype and gives the product in it,
/// summed in f32 ([module](self)). A bias is added to that product in f32 and the sum rounded to
/// `x`'s dtype. `dash_contract::quant` allows each rounding: the scales' to `x`'s dtype, `W`'s
/// where MLX forms it, and the product's before the bias.
///
/// Made on each call, they cost time rather than memory. On an Apple M6, against `5120 → 17408`
/// and `17408 → 5120` projections at one row of `x`, a call took 0.028–0.030 ms longer than with
/// scales and biases made once in bf16 (0.345–0.348 ms against 0.315–0.319), 0.010–0.015 ms
/// longer in f16; at 512 rows, 0.5% longer or less for `5120 → 17408`, within the runs' spread for
/// `17408 → 5120` (MLX 0.32.4, 2026-11-14, two runs, load averages 1.55–1.60). Kept, bf16 or f16
/// scales and biases would add 6.25% of the packed weight's bytes for each dtype.
///
/// Errors if `x` or `bias` is not f32, bf16 or f16, or if the shapes do not fit.
pub fn quantized_linear(
    x: &Array,
    w: &Array,
    scales: &Array,
    bias: Option<&Array>,
) -> Result<Array> {
    let dtype = x.dtype();
    let float = |d: Dtype| matches!(d, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16);
    if !float(dtype) || bias.is_some_and(|b| !float(b.dtype())) {
        return Err(Exception::custom(format!(
            "quantized_linear: x and the bias must be f32, bf16 or f16, got {dtype:?} and {:?}",
            bias.map(Array::dtype)
        )));
    }
    let scales = scales.as_dtype(dtype)?;
    let biases = scales.multiply(Array::from_f32(-8.0).as_dtype(dtype)?)?;
    let y = quantized_matmul(x, w, &scales, &biases, true, GROUP_SIZE, BITS)?;
    let Some(bias) = bias else {
        return Ok(y);
    };
    y.as_dtype(Dtype::Float32)?
        .add(bias.as_dtype(Dtype::Float32)?)?
        .as_dtype(dtype)
}
