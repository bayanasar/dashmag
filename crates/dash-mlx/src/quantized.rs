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
//!
//! # Backward
//!
//! Under MLX's autodiff the weight is frozen and `x` takes `dy Wᵀ`. MLX's own backward is the
//! quantized matmul again, untransposed, with `dy` and the scales and biases in `x`'s dtype. From
//! 4 rows of a matrix on, it takes its matrix path: like the forward's, it dequantizes `W`'s tiles
//! into that dtype and sums in f32. Below 4 rows it takes its vector path, which forms `W` in f32
//! but, from 1024 outputs on, splits the sum over them into 8 parts (32 past 8192), rounds each
//! part to `x`'s dtype and sums the parts in that dtype (MLX 0.32.4). `dash_contract::quant`
//! allows the matrix path's roundings, as it does the forward's, and not the vector path's: a
//! partial sum rounded to `x`'s dtype. So in bf16 and f16 below 4 rows, [`quantized_linear_in_vjp`]
//! gives `x` its gradient through a backward of its own: the same product with `dy` and the scales
//! read into f32, `W` exact, summed in f32 and rounded once to `x`'s dtype. In f32, MLX's own
//! backward already sums in f32 throughout. dash-mag's `tests/quant.rs` holds both paths to the
//! definition.

use crate::mlx::error::{Exception, Result};
use crate::mlx::ops::quantized_matmul;
use crate::mlx::transforms::custom_vjp;
use crate::mlx::{Array, Dtype};

/// GPTQ's group size: the rows of `in` that share a scale.
pub const GROUP_SIZE: i32 = 128;

/// GPTQ's bits per weight.
pub const BITS: i32 = 4;

/// The rows of a matrix from which MLX's quantized matmul, untransposed, takes its matrix path
/// (MLX 0.32.4's `vector_limit` for it, the same on every GPU).
const MATRIX_ROWS: i32 = 4;

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
/// Under MLX's autodiff the weight is frozen, and the bias takes the sum of `dy` over the rows in
/// f32, rounded once to its dtype. `x` takes MLX's own backward, which below 4 rows in bf16 and f16
/// rounds what `dash_contract::quant` does not allow ([module](self#backward)): where `x`'s
/// gradient may be asked, call [`quantized_linear_in_vjp`].
///
/// Errors if `x` or `bias` is not f32, bf16 or f16, or if the shapes do not fit.
pub fn quantized_linear(
    x: &Array,
    w: &Array,
    scales: &Array,
    bias: Option<&Array>,
) -> Result<Array> {
    linear(x, w, scales, bias, false)
}

/// [`quantized_linear`] for a call inside MLX's `vjp`: the same bits, and `x`'s gradient held to
/// the definition, below 4 rows in bf16 and f16 by a backward of its own in f32, otherwise by
/// MLX's ([module](self#backward)).
///
/// Below 4 rows in bf16 and f16 the product goes through `custom_vjp`, which costs time on every
/// call, so [`quantized_linear`] does without it. On an Apple M6, at one row of `x` in bf16, a call
/// took 0.0014–0.0035 ms longer against `5120 → 17408` and `17408 → 5120` projections (0.3–0.8%),
/// and in a graph of 64 calls against `5120 → 5120`, 0.0014–0.0034 ms longer each (1.7–4.4%)
/// (medians, MLX 0.32.4, 2026-11-20 UTC, two runs, load averages 1.22–2.33).
///
/// Errors as [`quantized_linear`] does.
pub fn quantized_linear_in_vjp(
    x: &Array,
    w: &Array,
    scales: &Array,
    bias: Option<&Array>,
) -> Result<Array> {
    linear(x, w, scales, bias, true)
}

/// [`quantized_linear`], and with `in_vjp` [`quantized_linear_in_vjp`].
fn linear(
    x: &Array,
    w: &Array,
    scales: &Array,
    bias: Option<&Array>,
    in_vjp: bool,
) -> Result<Array> {
    let dtype = x.dtype();
    let float = |d: Dtype| matches!(d, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16);
    if !float(dtype) || bias.is_some_and(|b| !float(b.dtype())) {
        return Err(Exception::custom(format!(
            "quantized_linear: x and the bias must be f32, bf16 or f16, got {dtype:?} and {:?}",
            bias.map(Array::dtype)
        )));
    }
    let cast = scales.as_dtype(dtype)?;
    let biases = cast.multiply(Array::from_f32(-8.0).as_dtype(dtype)?)?;
    let rows = match x.ndim() {
        0 | 1 => 1,
        n => x.shape()[n - 2],
    };
    let y = if in_vjp && dtype != Dtype::Float32 && rows < MATRIX_ROWS {
        with_f32_backward(x, w, scales, &cast, &biases)?
    } else {
        quantized_matmul(x, w, &cast, &biases, true, GROUP_SIZE, BITS)?
    };
    let Some(bias) = bias else {
        return Ok(y);
    };
    y.as_dtype(Dtype::Float32)?
        .add(bias.as_dtype(Dtype::Float32)?)?
        .as_dtype(dtype)
}

/// [`quantized_linear`]'s product, the same bits, with a backward of its own for `x`: below
/// [`MATRIX_ROWS`] rows in all, `dy` and `scales` (as given) read into f32, so that `W` is exact,
/// and the sum rounded once to `x`'s dtype; from there on, MLX's own, on all the rows as one matrix
/// ([module](self#backward)).
fn with_f32_backward(
    x: &Array,
    w: &Array,
    scales: &Array,
    cast: &Array,
    biases: &Array,
) -> Result<Array> {
    let (fw, fcast, fbiases) = (w.clone(), cast.clone(), biases.clone());
    let (w, scales, cast, biases) = (w.clone(), scales.clone(), cast.clone(), biases.clone());
    let product = custom_vjp(
        move |p: &[Array]| {
            let y = quantized_matmul(&p[0], &fw, &fcast, &fbiases, true, GROUP_SIZE, BITS)?;
            Ok(vec![y])
        },
        move |p: &[Array], cotangents: &[Array], _: &[Array]| {
            let (x, dy) = (&p[0], &cotangents[0]);
            let dy = dy.reshape(&[-1, dy.shape()[dy.ndim() - 1]])?;
            let dx = if dy.shape()[0] < MATRIX_ROWS {
                let scales = scales.as_dtype(Dtype::Float32)?;
                let biases = scales.multiply(Array::from_f32(-8.0))?;
                let dy = dy.as_dtype(Dtype::Float32)?;
                quantized_matmul(&dy, &w, &scales, &biases, false, GROUP_SIZE, BITS)?
            } else {
                quantized_matmul(&dy, &w, &cast, &biases, false, GROUP_SIZE, BITS)?
            };
            Ok(vec![dx.reshape(x.shape())?.as_dtype(x.dtype())?])
        },
    )?;
    Ok(product.apply(std::slice::from_ref(x))?.remove(0))
}
