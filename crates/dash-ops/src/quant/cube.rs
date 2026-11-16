//! The quantized linear on the CubeCL backend: for a few rows (decoding, up to [`MAX_ROWS`]),
//! two kernels that read the packed weight once and dequantize it in registers; on a device with
//! RDNA4's 16 x 16 x 16 matrix products, past [`MATRIX_ROWS_FROM`] rows, one kernel on them that
//! reads the packed words straight into its operands (`wmma.rs`); elsewhere for more rows
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

use super::wmma::{BLOCK_K, BLOCK_N, UNITS, block_m, matrix_kernel};
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
/// The most rows the decoding kernels take where the kernel on RDNA4's matrix products takes the
/// call: past 8, the decoding kernels' cost is that of 16 (their rows' variants are powers of
/// two). On a Radeon AI PRO R9700 (2026-11-15 UTC), at 10 to 16 rows the decoding kernels took
/// 441-463 us at [5120 x 17408] and 436-463 at [17408 x 5120], the matrix kernel 254-263 and
/// 353-375; at 8 rows the decoding kernels 252-274 and 253-260, the matrix kernel 251-256 and
/// 353-367.
pub(crate) const MATRIX_ROWS_FROM: usize = 8;
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
    !matrix_takes(client, dtype, [rows, inp, out], group)
        && matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
        && (1..=MAX_ROWS).contains(&rows)
        && group > 0
        && group.is_multiple_of(8)
        && inp.is_multiple_of(group)
        && out > 0
        && requirements(rows, inp / group, out, group)
            .met_on(client, &[dtype, DType::F32, DType::I32, DType::F16])
}

/// What [`matrix_kernel`] asks: 128 units per cube (four planes of 32, the layout's), a cube per 64
/// or 128 rows ([`matrix_tiles`]) and 128 outputs (and run of the inputs), 4 or 8 KiB of shared
/// memory, six tensors, 16-byte vectors; beside `finish_kernel`'s, which adds the runs' sums where
/// the inputs are split.
fn matrix_requirements([rows, inp, out]: [usize; 3]) -> Requirements {
    let block = block_m(matrix_tiles(rows));
    let finish = Requirements::new((CUBE as u32, 1), ((rows * out).div_ceil(CUBE), 1), 3);
    let product = Requirements {
        shared_bytes: block * BLOCK_K * 2,
        vector_size: 8,
        plane_width: Some(32),
        // The most runs the inputs split into: one per step of 32.
        address: matrix_address([rows, inp, out], inp / BLOCK_K),
        ..Requirements::new((UNITS, 1), (rows.div_ceil(block), out.div_ceil(BLOCK_N)), 6)
    };
    product.max(finish)
}

/// Whether the kernel on RDNA4's matrix products takes `x` `[rows, inp]` in `dtype`: more than
/// [`MATRIX_ROWS_FROM`] rows (where the decoding kernels would cost as for 16), bf16 or f16, inputs a
/// multiple of 32 (a step's) and groups a multiple of 8 (a packed word's), on a device with RDNA4's
/// 16 x 16 x 16 products.
pub(super) fn matrix_takes(
    client: &Client,
    dtype: DType,
    [rows, inp, out]: [usize; 3],
    group: usize,
) -> bool {
    matches!(dtype, DType::BF16 | DType::F16)
        && rows > MATRIX_ROWS_FROM
        && group > 0
        && group.is_multiple_of(8)
        && inp.is_multiple_of(group)
        && inp.is_multiple_of(BLOCK_K)
        && out > 0
        && matrix_requirements([rows, inp, out])
            .met_on(client, &[dtype, DType::F32, DType::I32, DType::F16])
        && crate::wmma::rdna4_products(client, dtype)
}

/// The runs [`matrix_split`] gives where [`matrix_takes`] holds, 0 where it does not.
pub(super) fn matrix_runs(
    client: &Client,
    dtype: DType,
    [rows, inp, out]: [usize; 3],
    group: usize,
) -> usize {
    match matrix_takes(client, dtype, [rows, inp, out], group) {
        true => matrix_split(client, rows, inp, out).0,
        false => 0,
    }
}

/// Whether the prefill path takes `x` `[rows, inp]` in `dtype`: more rows than the decoding
/// kernels take, the weight dequantized to `dtype` by a kernel and multiplied by Burn's matmul,
/// where [`matrix_takes`] does not.
pub(super) fn prefill_takes(
    client: &Client,
    dtype: DType,
    [rows, inp, out]: [usize; 3],
    group: usize,
) -> bool {
    !matrix_takes(client, dtype, [rows, inp, out], group)
        && matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
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
        if matrix_takes(&x.client, x.dtype, [rows, inp, out], group) {
            return matrix(x, qweight, scales, bias, group);
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

/// `y = x W (+ bias)` by [`matrix_kernel`]: `x` read in vectors of eight from a contiguous copy
/// whose first element is 16-byte aligned (a fresh one where the input's is not), the bias in f32.
fn matrix(
    x: CubeTensor,
    qweight: CubeTensor,
    scales: CubeTensor,
    bias: MaybeTensor<CubeBackend>,
    group: usize,
) -> CubeTensor {
    let [rows, inp] = x.meta.shape().dims::<2>();
    let out = qweight.meta.shape()[1];
    let mut x = into_contiguous(x);
    if !x.handle.offset_start.unwrap_or(0).is_multiple_of(16) {
        let (client, device, dtype) = (x.client.clone(), x.device.clone(), x.dtype);
        let copy = burn_cubecl::cubecl::std::tensor::into_contiguous(
            &client,
            x.binding(),
            dtype_to_storage_type(dtype),
        );
        x = CubeTensor::new(client, copy.handle, *copy.metadata, device, dtype);
    }
    let qweight = into_contiguous(qweight);
    let scales = into_contiguous(scales);
    let bias = match bias {
        MaybeTensor::None => None,
        MaybeTensor::Some(bias) => Some(into_contiguous(match bias.dtype {
            DType::F32 => bias,
            _ => <CubeBackend as burn::backend::ops::FloatTensorOps<CubeBackend>>::float_cast(
                bias,
                burn::tensor::FloatDType::F32,
            ),
        })),
    };
    let y = empty(&x, Shape::new([rows, out]), x.dtype);
    let has_bias = bias.is_some();
    // Bound where the kernel reads no bias.
    let bias = bias.unwrap_or_else(|| empty(&x, Shape::new([1]), DType::F32));
    let client = x.client.clone();
    let dtype = dtype_to_storage_type(x.dtype);
    let mts = matrix_tiles(rows);
    let (runs, run) = matrix_split(&client, rows, inp, out);
    let split = runs > 1;
    // f32 sums of each run where the inputs are split, else bound and unread.
    let part = empty(
        &x,
        Shape::new(if split { [runs, rows, out] } else { [1, 1, 1] }),
        DType::F32,
    );
    probe::MATRIX_LAUNCHES.fetch_add(1, Ordering::SeqCst);
    // SAFETY: cube (i, j) reads rows below `rows` of x and words and scales of columns below `out`
    // within qweight `[inp / 8, out]` and scales `[inp / group, out]`, inputs below `inp` (a
    // multiple of 32); it writes y within `[rows, out]`; the bias is read below `out` when given.
    unsafe {
        matrix_kernel::launch_unchecked(
            &client,
            CubeCount::Static(
                rows.div_ceil(block_m(mts)) as u32,
                out.div_ceil(BLOCK_N) as u32,
                runs as u32,
            ),
            CubeDim::new_1d(UNITS),
            matrix_address([rows, inp, out], runs),
            x.into_tensor_arg(),
            qweight.into_tensor_arg(),
            scales.into_tensor_arg(),
            bias.clone().into_tensor_arg(),
            y.clone().into_tensor_arg(),
            part.clone().into_tensor_arg(),
            run,
            group,
            has_bias && !split,
            split,
            mts,
            dtype,
            dtype_to_storage_type(DType::F16),
        );
        if split {
            finish_kernel::launch_unchecked(
                &client,
                CubeCount::Static((rows * out).div_ceil(CUBE) as u32, 1, 1),
                CubeDim::new_1d(CUBE as u32),
                matrix_address([rows, inp, out], runs),
                part.into_tensor_arg(),
                bias.into_tensor_arg(),
                y.clone().into_tensor_arg(),
                has_bias,
                dtype,
            );
        }
    }
    y
}

/// `(runs, run)`: the steps of 32 inputs split into `runs` runs of `run` steps, so that a product
/// with fewer cubes of 64 or 128 rows and 128 outputs than twice the device's compute units fills
/// it; one run otherwise.
fn matrix_split(client: &Client, rows: usize, inp: usize, out: usize) -> (usize, usize) {
    let units = client
        .properties()
        .hardware
        .num_streaming_multiprocessors
        .unwrap_or(64) as usize;
    let cubes = rows.div_ceil(block_m(matrix_tiles(rows))) * out.div_ceil(BLOCK_N);
    let steps = inp / BLOCK_K;
    let wanted = (2 * units).div_ceil(cubes).clamp(1, steps);
    let run = steps.div_ceil(wanted);
    (steps.div_ceil(run), run)
}

/// The 16-row tiles each wave of [`matrix_kernel`] takes for `rows` rows of `x`: eight past 64
/// rows, so that each dequantized word serves eight products; four up to 64, where a cube of 128
/// rows would be at least half empty. On a Radeon AI PRO R9700 at `[5120 x 17408]` and
/// `[17408 x 5120]`, eight took 0.64 to 0.88 times four's time at 96, 128, 256, 512 and 2048 rows
/// (0.88 at 256 rows of the second shape), and 1.15 to 1.31 times at 32, 48 and 64 rows.
fn matrix_tiles(rows: usize) -> usize {
    if rows > 64 { 8 } else { 4 }
}

/// The index type [`matrix_kernel`] and `finish_kernel` need: every tensor they bind indexed.
fn matrix_address([rows, inp, out]: [usize; 3], runs: usize) -> AddressType {
    AddressType::from_len((rows * inp).max(runs * rows * out).max(inp / 8 * out))
}

/// `y = x W (+ bias)` with `W` dequantized to `x`'s dtype by [`dequant_kernel`] and multiplied by
/// Burn's matmul, whose product comes out in `x`'s dtype; with a bias, that product taken to f32,
/// the bias added and the sum rounded to `x`'s dtype again. In f32 nothing is rounded but the sums.
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
