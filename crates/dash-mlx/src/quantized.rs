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
//! MLX's quantized matmul computes in the dtype that `x`'s and the scales' promote to, so the
//! scales and biases take `x`'s dtype. bf16 `x` with f16 scales would run in f32. On an Apple M6,
//! bf16 `x` `[2048, 5120]` against a `5120 → 17408` projection took 85.7 ms in f32 (f32 scales),
//! against 20.0 ms with bf16 scales: 4.3 times. Over 512 and 2048 rows of that projection and of
//! `17408 → 5120`, f32 scales took 2.9 to 4.3 times as long. (Medians, measured 2026-11-14 with
//! MLX 0.32.4. The M6 was not idle: only terminals were open, at a load average of 1.07–1.37.)
//!
//! MLX's dequantize of the result gives `(q - 8) · scale` bit for bit with f32 scales, and that
//! value rounded once with f16 scales. bf16 scales are the f16 ones rounded once, so with them
//! each weight is within one bf16 unit of `(q - 8) · scale` rounded once (`tests/quantized.rs`).

use crate::mlx::error::{Exception, Result};
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
