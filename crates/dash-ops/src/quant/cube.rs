//! The quantized linear on the CubeCL backend: for a few rows (decoding, up to [`MAX_ROWS`]),
//! two kernels that read the packed weight once and dequantize it in registers; for more rows
//! (prefill), the weight dequantized to `x`'s dtype by a kernel and multiplied by Burn's matmul;
//! on a device without what they ask, the reference.
//!
//! - `gemv`: each unit owns one output column and a run of whole groups of input rows. Per group
//!   it reads the column's scale once, then per packed word the eight values, `(q - 8) · scale`
//!   in f32, multiplied by each row's input and summed in f32 in input order. A run of the inputs
//!   per cube along y spreads a narrow layer over the device; the runs' sums go to an f32
//!   `[runs, N, out]` buffer.
//! - `finish`: each unit adds one output's runs in order, then the bias, and rounds once to `x`'s
//!   dtype.
//!
//! So each output is the same bits whatever the launch, for given runs.

use super::{QuantBackend, probe, reference};
use crate::cube::Requirements;
use crate::lowrank::MaybeTensor;
use burn::backend::DType;
use burn::backend::Shape;
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::client::Client;
use cubecl::prelude::*;
use std::sync::atomic::Ordering;

/// The most rows the decoding kernels take; more take the prefill path. Their products are f32
/// multiply-adds, so past a few rows they are bound by those rather than by reading the weight:
/// at 32 rows of [5120 x 17408] they take 0.34 ms on an RTX 5070 Ti, where the prefill path
/// would take about 0.55, and 1.89 ms on a Radeon AI PRO R9700, where it would take about 0.8.
/// 16 rows is where both cards are close (2026-11-13).
pub(crate) const MAX_ROWS: usize = 16;
/// Units per cube.
const CUBE: usize = 256;
/// Units the kernels aim to spread a product over, beyond which the inputs are not split further.
const TARGET_UNITS: usize = 1 << 16;

#[cube(launch_unchecked, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
fn gemv_kernel<E: Float, S: Float>(
    x: &Tensor<E>,
    qweight: &Tensor<i32>,
    scales: &Tensor<S>,
    partial: &mut Tensor<f32>,
    run: usize,
    #[comptime] rows: usize,
    #[comptime] group: usize,
    #[define(E)] _dtype: ElemType,
    #[define(S)] _scales_dtype: ElemType,
) {
    let out = qweight.shape(1);
    let n_rows = x.shape(0);
    let x_stride = x.stride(0);
    let groups = scales.shape(0);
    let o = ABSOLUTE_POS_X as usize;
    // Every unit stages its share of the inputs, past the last column too: no unit leaves
    // before the cube's barriers.
    let active = o < out;
    let s = CUBE_POS_Y as usize;
    let g0 = s * run;
    let g1 = clamp_max(g0 + run, groups);
    let words = comptime!(group / 8);
    // The group's inputs for every row, `[rows, group]` f32, rows past `n_rows` zero.
    let mut xs = Shared::<[f32]>::new_slice(comptime!(rows * group));
    let mut acc = Array::<f32>::new(rows);
    #[unroll]
    for n in 0..rows {
        acc[n] = 0.0f32;
    }
    let unit = UNIT_POS as usize;
    let units = CUBE_DIM as usize;
    for g in g0..g1 {
        let mut idx = unit;
        while idx < comptime!(rows * group) {
            let n = idx / group;
            let k = idx % group;
            let mut v = 0.0f32;
            if n < n_rows {
                v = f32::cast_from(x[n * x_stride + g * group + k]);
            }
            xs[idx] = v;
            idx += units;
        }
        sync_cube();
        if active {
            let scale = f32::cast_from(scales[g * out + o]);
            for pp in 0..words {
                let word = u32::cast_from(qweight[(g * words + pp) * out + o]);
                #[unroll]
                for j in 0..8u32 {
                    let q = (word >> (4u32 * j)) & 15u32;
                    let w = (f32::cast_from(q) - 8.0f32) * scale;
                    let k = pp * 8 + j as usize;
                    #[unroll]
                    for n in 0..rows {
                        acc[n] += xs[n * group + k] * w;
                    }
                }
            }
        }
        sync_cube();
    }
    if active {
        #[unroll]
        for n in 0..rows {
            if n < n_rows {
                partial[(s * n_rows + n) * out + o] = acc[n];
            }
        }
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn finish_kernel<E: Float>(
    partial: &Tensor<f32>,
    bias: &Tensor<f32>,
    y: &mut Tensor<E>,
    #[comptime] has_bias: bool,
    #[define(E)] _dtype: ElemType,
) {
    let runs = partial.shape(0);
    let len = partial.shape(1) * partial.shape(2);
    let out = partial.shape(2);
    let i = ABSOLUTE_POS;
    if i >= len {
        terminate!();
    }
    let mut sum = partial[i];
    for s in 1..runs {
        sum += partial[s * len + i];
    }
    if comptime!(has_bias) {
        sum += bias[i % out];
    }
    y[i] = E::cast_from(sum);
}

/// `w[8 p + j, o] = (q - 8) · scale` in `w`'s dtype, one unit per packed word `(p, o)`.
#[cube(launch_unchecked, address_type = "dynamic")]
fn dequant_kernel<E: Float, S: Float>(
    qweight: &Tensor<i32>,
    scales: &Tensor<S>,
    w: &mut Tensor<E>,
    #[comptime] group: usize,
    #[define(E)] _dtype: ElemType,
    #[define(S)] _scales_dtype: ElemType,
) {
    let out = qweight.shape(1);
    let pos = ABSOLUTE_POS;
    if pos >= qweight.shape(0) * out {
        terminate!();
    }
    let p = pos / out;
    let o = pos % out;
    let word = u32::cast_from(qweight[pos]);
    let scale = f32::cast_from(scales[(p * 8 / group) * out + o]);
    #[unroll]
    for j in 0..8u32 {
        let q = (word >> (4u32 * j)) & 15u32;
        w[(p * 8 + j as usize) * out + o] = E::cast_from((f32::cast_from(q) - 8.0f32) * scale);
    }
}

/// The rows' comptime count: `n` rounded up to a power of two, so that few variants compile.
fn rows_variant(n: usize) -> usize {
    n.next_power_of_two()
}

/// `(runs, run)`: the input groups split into `runs` runs of `run` groups each.
fn split(groups: usize, out: usize) -> (usize, usize) {
    let wanted = TARGET_UNITS.div_ceil(out).clamp(1, groups);
    let run = groups.div_ceil(wanted);
    (groups.div_ceil(run), run)
}

fn requirements(rows: usize, groups: usize, out: usize, group: usize) -> Requirements {
    let (runs, _) = split(groups, out);
    let gemv = Requirements {
        shared_bytes: rows_variant(rows) * group * 4,
        ..Requirements::new((CUBE as u32, 1), (out.div_ceil(CUBE), runs), 4)
    };
    let finish = Requirements::new((CUBE as u32, 1), ((rows * out).div_ceil(CUBE), 1), 3);
    Requirements {
        address: AddressType::from_len((runs * rows * out).max(groups * 16 * out)),
        ..gemv.max(finish)
    }
}

/// Whether the kernels take `x` `[rows, inp]` in `dtype` over a weight of `out` outputs in groups
/// of `group` inputs on the device `client` serves.
pub(super) fn kernels_take(
    client: &Client,
    dtype: DType,
    [rows, inp, out]: [usize; 3],
    group: usize,
) -> bool {
    matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
        && (1..=MAX_ROWS).contains(&rows)
        && group > 0
        && group.is_multiple_of(8)
        && inp.is_multiple_of(group)
        && out > 0
        && requirements(rows, inp / group, out, group)
            .met_on(client, &[dtype, DType::F32, DType::I32, DType::F16])
}

/// Whether the prefill path takes `x` `[rows, inp]` in `dtype`: more rows than the decoding
/// kernels take, the weight dequantized to `dtype` by a kernel and multiplied by Burn's matmul.
pub(super) fn prefill_takes(
    client: &Client,
    dtype: DType,
    [rows, inp, out]: [usize; 3],
    group: usize,
) -> bool {
    matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
        && rows > MAX_ROWS
        && group > 0
        && group.is_multiple_of(8)
        && inp.is_multiple_of(group)
        && out > 0
        && Requirements {
            address: AddressType::from_len(inp * out),
            ..Requirements::new((CUBE as u32, 1), ((inp / 8 * out).div_ceil(CUBE), 1), 3)
        }
        .met_on(client, &[dtype, DType::I32, DType::F16])
}

impl QuantBackend for CubeBackend {
    fn quantized_linear(
        x: FloatTensor<Self>,
        qweight: IntTensor<Self>,
        scales: FloatTensor<Self>,
        bias: MaybeTensor<Self>,
        group: usize,
    ) -> FloatTensor<Self> {
        let [rows, inp] = x.meta.shape().dims::<2>();
        let out = qweight.meta.shape()[1];
        if qweight.dtype != DType::I32 || scales.dtype != DType::F16 {
            return reference::forward::<Self>(x, qweight, scales, bias, group);
        }
        if prefill_takes(&x.client, x.dtype, [rows, inp, out], group) {
            return prefill(x, qweight, scales, bias, group);
        }
        if !kernels_take(&x.client, x.dtype, [rows, inp, out], group) {
            return reference::forward::<Self>(x, qweight, scales, bias, group);
        }
        let client = x.client.clone();
        let x = into_contiguous(x);
        let qweight = into_contiguous(qweight);
        let scales = into_contiguous(scales);
        let bias = match bias {
            MaybeTensor::None => None,
            MaybeTensor::Some(bias) => Some(into_contiguous(match bias.dtype {
                DType::F32 => bias,
                _ => <Self as burn::backend::ops::FloatTensorOps<Self>>::float_cast(
                    bias,
                    burn::tensor::FloatDType::F32,
                ),
            })),
        };
        let groups = inp / group;
        let (runs, run) = split(groups, out);
        let partial = empty(&x, Shape::new([runs, rows, out]), DType::F32);
        let y = empty(&x, Shape::new([rows, out]), x.dtype);
        let address = AddressType::from_len((runs * rows * out).max(groups * 16 * out));
        let dtype = dtype_to_storage_type(x.dtype);
        probe::GEMV_LAUNCHES.fetch_add(1, Ordering::SeqCst);
        // SAFETY: unit (o, s) reads column o < out of qweight's rows and scales' rows in its run,
        // which lie within them; x's rows below `rows`, inputs below `inp`; and writes
        // `partial[s, n, o]` within `[runs, rows, out]`. `finish` reads `partial` and the bias
        // within their shapes and writes `y` within `[rows, out]`.
        unsafe {
            gemv_kernel::launch_unchecked(
                &client,
                CubeCount::Static(out.div_ceil(CUBE) as u32, runs as u32, 1),
                CubeDim::new_1d(CUBE as u32),
                address,
                x.clone().into_tensor_arg(),
                qweight.into_tensor_arg(),
                scales.into_tensor_arg(),
                partial.clone().into_tensor_arg(),
                run,
                rows_variant(rows),
                group,
                dtype,
                dtype_to_storage_type(DType::F16),
            );
            finish_kernel::launch_unchecked(
                &client,
                CubeCount::Static((rows * out).div_ceil(CUBE) as u32, 1, 1),
                CubeDim::new_1d(CUBE as u32),
                address,
                partial.clone().into_tensor_arg(),
                bias.clone()
                    .unwrap_or_else(|| partial.clone())
                    .into_tensor_arg(),
                y.clone().into_tensor_arg(),
                bias.is_some(),
                dtype,
            );
        }
        y
    }
}

fn empty(like: &CubeTensor, shape: Shape, dtype: DType) -> CubeTensor {
    empty_device_contiguous_dtype(like.client.clone(), like.device.clone(), shape, dtype)
}

/// `y = x W (+ bias)` with `W` dequantized to `x`'s dtype by [`dequant_kernel`] and multiplied by
/// Burn's matmul; the bias added in f32 and the sum rounded once to `x`'s dtype.
fn prefill(
    x: CubeTensor,
    qweight: CubeTensor,
    scales: CubeTensor,
    bias: MaybeTensor<CubeBackend>,
    group: usize,
) -> CubeTensor {
    use burn::backend::ops::FloatTensorOps;
    use burn::tensor::FloatDType;
    type B = CubeBackend;
    let [packed, out] = qweight.meta.shape().dims::<2>();
    let inp = packed * 8;
    let dtype = x.dtype;
    let qweight = into_contiguous(qweight);
    let scales = into_contiguous(scales);
    let w = empty(&x, Shape::new([inp, out]), dtype);
    probe::DEQUANT_LAUNCHES.fetch_add(1, Ordering::SeqCst);
    // SAFETY: unit `pos` reads word `pos < packed · out` of qweight and its group's scale within
    // `[inp / group, out]`, and writes rows `8 p` to `8 p + 7` of column `o` of `w` `[inp, out]`.
    unsafe {
        dequant_kernel::launch_unchecked(
            &x.client,
            CubeCount::Static((packed * out).div_ceil(CUBE) as u32, 1, 1),
            CubeDim::new_1d(CUBE as u32),
            AddressType::from_len(inp * out),
            qweight.into_tensor_arg(),
            scales.into_tensor_arg(),
            w.clone().into_tensor_arg(),
            group,
            dtype_to_storage_type(dtype),
            dtype_to_storage_type(DType::F16),
        );
    }
    let y = B::float_matmul(x, w);
    match bias {
        MaybeTensor::None => y,
        MaybeTensor::Some(bias) => {
            let shape = y.meta.shape().clone();
            let y32 = match dtype {
                DType::F32 => y,
                _ => B::float_cast(y, FloatDType::F32),
            };
            let bias = B::float_reshape(B::float_cast(bias, FloatDType::F32), Shape::new([1, out]));
            let sum = B::float_add(y32, B::float_expand(bias, shape));
            match dtype {
                DType::F32 => sum,
                _ => B::float_cast(sum, dtype.into()),
            }
        }
    }
}
