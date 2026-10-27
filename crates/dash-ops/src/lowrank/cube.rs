//! The CubeCL kernels, for ranks 1 to [`MAX_RANK`], on devices with what [`requirements`] lists.
//!
//! Each product has one wide operand (`x`, `dy`, or the output it writes, `[N, in]` or
//! `[N, out]`) and a narrow side (`h` or `dh`, `[N, r]`, and a factor, `[r, in]` or `[out, r]`)
//! small enough to stay in caches, shared memory and registers. So each kernel streams its wide
//! operand once, in vectors along its rows, and every output element is one sum in order, with
//! fused multiply-adds, over its terms (or over a run of them, the runs added in order by
//! [`sum_kernel`]):
//!
//! - [`down_kernel`] (`[N, K] x [K, r]`): a cube owns a block of rows and a run of `K`; it stages
//!   the factor's `r x tile` slice of each tile in shared memory, and each unit sums one of the
//!   `r` outputs (or a few, above 16) for two rows. The 16 units of a row read the same vector of
//!   it, which the device broadcasts.
//! - [`up_kernel`] (`[N, r] x [r, M]`): a unit owns a vector of consecutive columns of a block of
//!   rows, with their factor rows in registers; per row it reads `h`'s `r` values (the same for
//!   every unit of the block) and the base's vector, and writes the rounded sum.
//! - [`gather_kernel`] (`[r, N] x [N, C]`): a unit owns a vector of columns and some of the `r`
//!   outputs, and walks a run of rows reading one vector per row; `h`'s value per row is the same
//!   for the 32 units of a plane.
//!
//! Every tensor is indexed through its strides: CubeCL may pad rows, and a factor may come
//! transposed (`B` read as `[r, out]` for `dh`, `A` as `[in, r]` for `dx`).

use super::{LowRankBackend, MAX_RANK, MaybeTensor, Wanted, probe, reference};
use crate::cube::Requirements;
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::FloatTensor;
use burn::backend::{DType, Shape, TensorMetadata};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::{into_contiguous, untile};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::tensor_vector_size_parallel;
use std::sync::atomic::Ordering;

/// Units per cube, in every kernel.
const CUBE: usize = 256;
/// Units per row in [`down_kernel`], one per output up to 16; a rank above that gives each unit
/// several outputs.
const DOWN_LANES: usize = 16;
/// Rows each unit of [`down_kernel`] sums, so that each value it reads from shared memory serves
/// two multiply-adds.
const DOWN_ROWS: usize = 2;
/// The factor's slice [`down_kernel`] stages at a time, `r x tile` f32: 16 KiB.
const DOWN_SHARED: usize = 4096;
/// The shortest run of `K` a `down` cube sums before the runs are added: sums of up to this many
/// terms are one run, in order.
const DOWN_MIN_RUN: usize = 256;
/// Vectors of each row [`down_kernel`] loads before it adds them up, rows [`up_kernel`] and
/// [`gather_kernel`] take at a time: loads in flight at once, which the products need to keep the
/// memory busy.
const DOWN_UNROLL: usize = 4;
const UP_UNROLL: usize = 4;
const GATHER_UNROLL: usize = 4;
/// Rank lanes per cube in [`gather_kernel`], beside 32 column lanes.
const GATHER_LANES: usize = 8;
/// The shortest run of rows a `gather` unit sums: sums over up to this many rows are one run.
const GATHER_MIN_RUN: usize = 64;
/// Units the kernels aim to spread a product over, beyond which runs are not split further (and
/// `up` gives its units more rows).
const TARGET_UNITS: usize = 1 << 17;
/// Widest vector a unit of `up` or `gather` carries: each lane is `r` registers in `up`.
const MAX_VECTOR: usize = 4;

/// The rows of [`down_kernel`]'s slice for rank `rank`: the rank, rounded up to whole outputs per
/// unit.
fn down_slice_rows(rank: usize) -> usize {
    let lanes = rank.min(DOWN_LANES);
    rank.div_ceil(lanes) * lanes
}

/// The `tile` of [`down_kernel`] for rank `rank`: the most columns of `K` whose slice fits
/// [`DOWN_SHARED`], a multiple of 8 (the widest vector of `x`).
fn down_tile(rank: usize) -> usize {
    (DOWN_SHARED / down_slice_rows(rank) / 8 * 8).max(8)
}

/// `out[s, n, j] = sum_{k in run s} (u[n, k] · scale) v[j, k]` (without `scaled`, `u[n, k]`), in
/// order over `k`, for the rows `n` of `u` `[rows, K]` (a unit stride along `K`) and the `rank`
/// rows `j` of `v`, read through `v_strides` (along `j`, along `k`); run `s` is
/// `[s · run, (s + 1) · run)`. `out` is `[S, rows, rank]`, contiguous. With `k_major`, `v`'s
/// unit stride is along `k`, and the slices are staged in that order.
#[cube(launch_unchecked, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
fn down_kernel<E: Float, N: Size>(
    u: &Tensor<Vector<E, N>>,
    v: &Tensor<f32>,
    out: &mut Tensor<f32>,
    v_rank_stride: usize,
    v_k_stride: usize,
    run: usize,
    scale: f32,
    #[comptime] rank: usize,
    #[comptime] lanes: usize,
    #[comptime] tile: usize,
    #[comptime] scaled: bool,
    #[comptime] k_major: bool,
    #[define(E)] _dtype: ElemType,
) {
    let n = u.vector_size();
    let rows = u.shape(0);
    let k_total = u.shape(1);
    let per = comptime!(rank.div_ceil(lanes));
    let stride = comptime!(tile + 1);
    let lane = UNIT_POS_X as usize;
    let block = CUBE_DIM_Y as usize;
    let row0 = CUBE_POS_X as usize * block * comptime!(DOWN_ROWS) + UNIT_POS_Y as usize;
    let s = CUBE_POS_Y as usize;
    let k0 = s * run;
    let k1 = clamp_max(k0 + run, k_total);
    let units = CUBE_DIM as usize;
    let unit = UNIT_POS as usize;
    let u_step = u.stride(0) / n;

    // The slice has `per · lanes` rows, so that every unit reads one; those past `rank` are
    // never written, and the sums they feed never stored.
    let mut slice = Shared::<[f32]>::new_slice(comptime!(per * lanes * (tile + 1)));
    let mut acc = Array::<f32>::new(comptime!(per * DOWN_ROWS));
    #[unroll]
    for i in 0..comptime!(per * DOWN_ROWS) {
        acc[i] = 0.0f32;
    }
    let (row1, ok0) = (row0 + block, row0 < rows);
    let ok1 = row1 < rows;

    let mut t0 = k0;
    while t0 < k1 {
        let len = clamp_max(tile, k1 - t0);
        // The previous tile's reads are done before its slice is overwritten.
        sync_cube();
        let mut i = unit;
        while i < comptime!(rank * tile) {
            let mut j = i / tile;
            let mut kk = i % tile;
            if !k_major {
                j = i % rank;
                kk = i / rank;
            }
            let mut value = 0.0f32;
            if kk < len {
                value = v[j * v_rank_stride + (t0 + kk) * v_k_stride];
            }
            slice[j * stride + kk] = value;
            i += units;
        }
        sync_cube();
        let vectors = len / n;
        let (at0, at1) = (row0 * u_step + t0 / n, row1 * u_step + t0 / n);
        // Four vectors of each row in flight at once; each value read from the slice serves both
        // rows' sums.
        let mut c = 0usize;
        while c + DOWN_UNROLL <= vectors {
            let mut xs = Array::<Vector<f32, N>>::new(comptime!(2 * DOWN_UNROLL));
            #[unroll]
            for q in 0..DOWN_UNROLL {
                xs[q] = load_row::<E, N>(u, at0 + c + q, ok0, scale, false);
                xs[DOWN_UNROLL + q] = load_row::<E, N>(u, at1 + c + q, ok1, scale, false);
            }
            #[unroll]
            for q in 0..DOWN_UNROLL {
                let k = (c + q) * n;
                let (x0, x1) = (xs[q], xs[DOWN_UNROLL + q]);
                accumulate::<N>(
                    &slice, x0, x1, k, lane, &mut acc, scale, per, lanes, stride, scaled,
                );
            }
            c += DOWN_UNROLL;
        }
        while c < vectors {
            let x0 = load_row::<E, N>(u, at0 + c, ok0, scale, false);
            let x1 = load_row::<E, N>(u, at1 + c, ok1, scale, false);
            accumulate::<N>(
                &slice,
                x0,
                x1,
                c * n,
                lane,
                &mut acc,
                scale,
                per,
                lanes,
                stride,
                scaled,
            );
            c += 1;
        }
        t0 += tile;
    }

    #[unroll]
    for p in 0..per {
        let j = lane + p * lanes;
        if j < rank {
            if ok0 {
                out[(s * rows + row0) * rank + j] = acc[p];
            }
            if ok1 {
                out[(s * rows + row1) * rank + j] = acc[per + p];
            }
        }
    }
}

/// Row `at`'s vector of `u` in f32, times `scale` when `scaled` (rounded before it enters a
/// product), or zeros where the row is past the last (`ok` false).
#[cube]
fn load_row<E: Float, N: Size>(
    u: &Tensor<Vector<E, N>>,
    at: usize,
    ok: bool,
    scale: f32,
    #[comptime] scaled: bool,
) -> Vector<f32, N> {
    let mut x = Vector::<f32, N>::new(0.0f32);
    if ok {
        x = Vector::cast_from(u[at]);
        if scaled {
            x *= Vector::new(scale);
        }
    }
    x
}

/// Adds the terms of columns `k0..k0 + n` to the sums of two rows, `x0` and `x1` their values
/// (times `scale` when `scaled`, rounded before they enter a product): `acc[p]` (row 0) and
/// `acc[per + p]` (row 1) for output `lane + p · lanes`, in order over `k`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn accumulate<N: Size>(
    slice: &Shared<[f32]>,
    x0: Vector<f32, N>,
    x1: Vector<f32, N>,
    k0: usize,
    lane: usize,
    acc: &mut Array<f32>,
    scale: f32,
    #[comptime] per: usize,
    #[comptime] lanes: usize,
    #[comptime] stride: usize,
    #[comptime] scaled: bool,
) {
    let n = x0.vector_size();
    #[unroll]
    for e in 0..n {
        let (mut a, mut b) = (x0.extract(e), x1.extract(e));
        if scaled {
            a *= scale;
            b *= scale;
        }
        #[unroll]
        for p in 0..per {
            let w = slice[(lane + p * lanes) * stride + k0 + e];
            acc[p] = fma(a, w, acc[p]);
            acc[per + p] = fma(b, w, acc[per + p]);
        }
    }
}

/// `out[n, m] = base[n, m] + scale · sum_j h[n, j] w[m, j]` (without `scaled`, the sum alone),
/// summed in order over `j` and in f32, then rounded once to `out`'s dtype; without a base,
/// the scaled sum. `h` is `[rows, rank]`, contiguous; the factor `w` is laid out `[rank, M]`, and
/// it, `base` and `out` (`[rows, M]`) have a unit stride along `M`, so that the units of a plane
/// read consecutive vectors of each. Each unit owns one vector of columns of `rows_per_unit`
/// consecutive rows.
#[cube(launch_unchecked, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
fn up_kernel<E: Float, O: Float, N: Size>(
    h: &Tensor<f32>,
    w: &Tensor<Vector<f32, N>>,
    base: ComptimeOption<&Tensor<Vector<E, N>>>,
    out: &mut Tensor<Vector<O, N>>,
    rows_per_unit: usize,
    scale: f32,
    #[comptime] rank: usize,
    #[comptime] scaled: bool,
    #[define(E, O)] _dtypes: [ElemType; 2],
) {
    let n = out.vector_size();
    let rows = out.shape(0);
    let cols = out.shape(1) / n;
    let cv = ABSOLUTE_POS % cols;
    let r0 = (ABSOLUTE_POS / cols) * rows_per_unit;
    if r0 >= rows {
        terminate!();
    }
    let r1 = clamp_max(r0 + rows_per_unit, rows);

    // The factor's values for the unit's columns: lane `e` of `factor[j]` is `w[j, n cv + e]`.
    let w_step = w.stride(0) / n;
    let mut factor = Array::<Vector<f32, N>>::new(rank);
    #[unroll]
    for j in 0..rank {
        factor[j] = w[j * w_step + cv];
    }

    let out_step = out.stride(0) / n;
    // Four rows at a time, so that four of the base's vectors are in flight at once.
    let mut r = r0;
    while r + UP_UNROLL <= r1 {
        let mut values = Array::<Vector<f32, N>>::new(UP_UNROLL);
        #[comptime]
        match base {
            ComptimeOption::Some(base) => {
                let step = base.stride(0) / n;
                #[unroll]
                for q in 0..UP_UNROLL {
                    values[q] = Vector::<f32, N>::cast_from(base[(r + q) * step + cv]);
                }
            }
            ComptimeOption::None => {}
        }
        #[unroll]
        for q in 0..UP_UNROLL {
            let sum = row_sum::<N>(h, &factor, r + q, scale, rank, scaled);
            #[comptime]
            let value = match base {
                ComptimeOption::Some(_) => values[q] + sum,
                ComptimeOption::None => sum,
            };
            out[(r + q) * out_step + cv] = Vector::cast_from(value);
        }
        r += UP_UNROLL;
    }
    while r < r1 {
        let sum = row_sum::<N>(h, &factor, r, scale, rank, scaled);
        #[comptime]
        let value = match base {
            ComptimeOption::Some(base) => {
                Vector::<f32, N>::cast_from(base[r * (base.stride(0) / n) + cv]) + sum
            }
            ComptimeOption::None => sum,
        };
        out[r * out_step + cv] = Vector::cast_from(value);
        r += 1;
    }
}

/// `scale · sum_j h[r, j] factor[j]` (without `scaled`, the sum alone), summed in order over `j`.
#[cube]
fn row_sum<N: Size>(
    h: &Tensor<f32>,
    factor: &Array<Vector<f32, N>>,
    r: usize,
    scale: f32,
    #[comptime] rank: usize,
    #[comptime] scaled: bool,
) -> Vector<f32, N> {
    let mut sum = Vector::<f32, N>::new(0.0f32);
    #[unroll]
    for j in 0..rank {
        sum = fma(Vector::new(h[r * rank + j]), factor[j], sum);
    }
    if scaled {
        sum *= Vector::new(scale);
    }
    sum
}

/// `out[s, c, j] = sum_{n in run s} (u[n, c] · scale) g[n, j]` (without `scaled`, `u[n, c]`), in
/// order over `n`, for the columns `c` of `u` `[rows, C]` (a unit stride along `C`) and the
/// `rank` columns `j` of `g` `[rows, rank]` (contiguous); run `s` is rows `[s · run, (s + 1) ·
/// run)`. `out` is written through `out_strides` (along `s`, along `c`, along `j`): the result
/// itself when there is one run, `[S, C, rank]` partial sums otherwise.
#[cube(launch_unchecked, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
fn gather_kernel<E: Float, N: Size>(
    u: &Tensor<Vector<E, N>>,
    g: &Tensor<f32>,
    out: &mut Tensor<f32>,
    out_run_stride: usize,
    out_col_stride: usize,
    out_rank_stride: usize,
    run: usize,
    scale: f32,
    #[comptime] rank: usize,
    #[comptime] lanes: usize,
    #[comptime] scaled: bool,
    #[define(E)] _dtype: ElemType,
) {
    let n = u.vector_size();
    let rows = u.shape(0);
    let cols = u.shape(1) / n;
    let cv = CUBE_POS_X as usize * CUBE_DIM_X as usize + UNIT_POS_X as usize;
    if cv >= cols {
        terminate!();
    }
    let lane = UNIT_POS_Y as usize;
    let s = CUBE_POS_Y as usize;
    let n0 = s * run;
    let n1 = clamp_max(n0 + run, rows);
    let per = comptime!(rank.div_ceil(lanes));
    let u_step = u.stride(0) / n;

    let mut acc = Array::<Vector<f32, N>>::new(per);
    #[unroll]
    for p in 0..per {
        acc[p] = Vector::new(0.0f32);
    }
    // Four rows at a time, so that four of `u`'s vectors are in flight at once.
    let mut r = n0;
    while r + GATHER_UNROLL <= n1 {
        let mut xs = Array::<Vector<f32, N>>::new(GATHER_UNROLL);
        #[unroll]
        for q in 0..GATHER_UNROLL {
            xs[q] = load_row::<E, N>(u, (r + q) * u_step + cv, true, scale, scaled);
        }
        #[unroll]
        for q in 0..GATHER_UNROLL {
            #[unroll]
            for p in 0..per {
                let j = lane + p * lanes;
                if j < rank {
                    acc[p] = fma(xs[q], Vector::new(g[(r + q) * rank + j]), acc[p]);
                }
            }
        }
        r += GATHER_UNROLL;
    }
    while r < n1 {
        let x = load_row::<E, N>(u, r * u_step + cv, true, scale, scaled);
        #[unroll]
        for p in 0..per {
            let j = lane + p * lanes;
            if j < rank {
                acc[p] = fma(x, Vector::new(g[r * rank + j]), acc[p]);
            }
        }
        r += 1;
    }
    #[unroll]
    for p in 0..per {
        let j = lane + p * lanes;
        if j < rank {
            #[unroll]
            for e in 0..n {
                let c = cv * n + e;
                out[s * out_run_stride + c * out_col_stride + j * out_rank_stride] =
                    acc[p].extract(e);
            }
        }
    }
}

/// `out[c, j] = sum_s partial[s, c, j]`, in order over `s`, for `partial` `[S, C, rank]`
/// (contiguous), `out` written through `out_strides` (along `c`, along `j`).
#[cube(launch_unchecked, address_type = "dynamic")]
fn sum_kernel(
    partial: &Tensor<f32>,
    out: &mut Tensor<f32>,
    out_col_stride: usize,
    out_rank_stride: usize,
    #[comptime] rank: usize,
) {
    let runs = partial.shape(0);
    let len = partial.shape(1) * rank;
    let i = ABSOLUTE_POS;
    if i >= len {
        terminate!();
    }
    let mut sum = partial[i];
    for s in 1..runs {
        sum += partial[s * len + i];
    }
    out[(i / rank) * out_col_stride + (i % rank) * out_rank_stride] = sum;
}

/// A tensor the kernels read in vectors along its rows: untiled, with a unit stride along its
/// last axis. One that already is is used as it is, padded rows and all.
fn rows_last(tensor: CubeTensor) -> CubeTensor {
    let tensor = untile(tensor);
    match tensor.meta.strides()[1] {
        1 => tensor,
        _ => into_contiguous(tensor),
    }
}

/// The widest vector, at most `cap` (rounded down to a power of two), that every one of `tensors`
/// can be read in along its rows.
fn vector_size(tensors: &[&CubeTensor], cap: usize) -> usize {
    let cap = 1 << cap.max(1).ilog2();
    tensors
        .iter()
        .map(|tensor| {
            tensor_vector_size_parallel(
                tensor.client.io_optimized_vector_sizes(tensor.dtype.size()),
                tensor.meta.shape(),
                tensor.meta.strides(),
                1,
            )
        })
        .fold(cap, usize::min)
        .max(1)
}

fn address_type(tensors: &[&CubeTensor]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}

fn empty(like: &CubeTensor, shape: Shape, dtype: DType) -> CubeTensor {
    empty_device_contiguous_dtype(like.client.clone(), like.device.clone(), shape, dtype)
}

/// The multiprocessors of the device, or a guess where it does not say.
fn processors(client: &Client) -> usize {
    client
        .properties()
        .hardware
        .num_streaming_multiprocessors
        .map_or(32, |count| count as usize)
}

/// How [`down_kernel`] splits `[rows, k] x [k, rank]`: `(cube_dim, cubes along x, runs, run)`.
fn down_launch(
    client: &Client,
    rows: usize,
    k: usize,
    rank: usize,
) -> ((u32, u32), usize, usize, usize) {
    let lanes = rank.min(DOWN_LANES);
    let block = CUBE / lanes;
    let blocks = rows.div_ceil(block * DOWN_ROWS);
    let tile = down_tile(rank);
    // Enough cubes for eight per multiprocessor, in runs of whole tiles no shorter than the least.
    let wanted = (8 * processors(client)).div_ceil(blocks);
    let most = k.div_ceil(DOWN_MIN_RUN).max(1);
    let runs = wanted.clamp(1, most);
    let run = k.div_ceil(runs).div_ceil(tile) * tile;
    let runs = k.div_ceil(run).max(1);
    ((lanes as u32, block as u32), blocks, runs, run)
}

/// How [`gather_kernel`] splits `[rank, rows] x [rows, cols]` for `cols / n` vectors of columns:
/// `(cube_dim, cubes along x, runs, run)`.
fn gather_launch(rows: usize, vectors: usize, rank: usize) -> ((u32, u32), usize, usize, usize) {
    let lanes = rank.min(GATHER_LANES);
    let width = CUBE / lanes;
    let blocks = vectors.div_ceil(width);
    let wanted = TARGET_UNITS.div_ceil(blocks * CUBE);
    let most = rows.div_ceil(GATHER_MIN_RUN).max(1);
    let runs = wanted.clamp(1, most);
    let run = rows.div_ceil(runs);
    let runs = rows.div_ceil(run).max(1);
    ((width as u32, lanes as u32), blocks, runs, run)
}

/// Rows per unit of [`up_kernel`] over `rows` rows of `vectors` vectors: as few as keep the units
/// near [`TARGET_UNITS`], from 4 (a unit reads `r` factor vectors for its rows) to 64.
fn up_rows(rows: usize, vectors: usize) -> usize {
    (rows * vectors).div_ceil(TARGET_UNITS).clamp(4, 64)
}

/// What one launch asks of the device: a cube of `cube_dim` units (at most [`CUBE`]), `cubes`
/// of them, `down`'s shared slice for rank `rank`, `tensors` bound at once, indexed with
/// `address`.
fn requirements(
    (cube_dim, cubes, tensors): ((u32, u32), (usize, usize), u32),
    rank: usize,
    address: AddressType,
) -> Requirements {
    Requirements {
        shared_bytes: down_slice_rows(rank) * (down_tile(rank) + 1) * 4,
        address,
        ..Requirements::new(cube_dim, cubes, tensors)
    }
}

/// Whether the device takes the forward and backward kernels for `x` `[rows, inp]` in `dtype`,
/// rank `rank` and `out` outputs.
pub(super) fn kernels_take(
    client: &Client,
    dtype: DType,
    rank: usize,
    [rows, inp, out]: [usize; 3],
) -> bool {
    (1..=MAX_RANK).contains(&rank)
        && matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
        && rows * inp * out > 0
        && fits(
            client,
            rank,
            [rows, inp, out],
            AddressType::from_len(rows * inp.max(out)),
            dtype,
        )
}

/// Whether the launches of a forward and a backward pass at these sizes fit the device, with
/// tensors of `dtype` and f32.
fn fits(
    client: &Client,
    rank: usize,
    [rows, inp, out]: [usize; 3],
    address: AddressType,
    dtype: DType,
) -> bool {
    let (down_dim, down_blocks, down_runs, _) = down_launch(client, rows, inp.max(out), rank);
    let vectors = inp.max(out);
    let (gather_dim, gather_blocks, gather_runs, _) = gather_launch(rows, vectors, rank);
    let up_units = rows.div_ceil(up_rows(rows, vectors)) * vectors;
    let launches = [
        (down_dim, (down_blocks, down_runs), 3),
        ((CUBE as u32, 1), (up_units.div_ceil(CUBE), 1), 4),
        (gather_dim, (gather_blocks, gather_runs), 3),
        (
            (CUBE as u32, 1),
            ((rows.max(inp).max(out) * rank).div_ceil(CUBE), 1),
            2,
        ),
    ];
    // Each launch on its own: their cubes differ in shape.
    launches
        .into_iter()
        .all(|launch| requirements(launch, rank, address).met_on(client, &[dtype, DType::F32]))
}

/// Whether the kernels take these tensors: `x` `[rows, inp]` in f32, bf16 or f16, the factors
/// `u` `[inp, r]` and `v` `[r, out]` f32 of a rank from 1 to [`MAX_RANK`], a base in a float dtype,
/// nothing empty, on a device that fits the launches.
fn applies(x: &CubeTensor, u: &CubeTensor, v: &CubeTensor, base: Option<&CubeTensor>) -> bool {
    let [rows, inp] = x.meta.shape().dims::<2>();
    let [_, rank] = u.meta.shape().dims::<2>();
    let [_, out] = v.meta.shape().dims::<2>();
    let float = |dtype| matches!(dtype, DType::F32 | DType::BF16 | DType::F16);
    let mut tensors = vec![x, u, v];
    tensors.extend(base);
    let address = address_type(&tensors).max(AddressType::from_len(rows * inp.max(out)));
    (1..=MAX_RANK).contains(&rank)
        && float(x.dtype)
        && u.dtype == DType::F32
        && v.dtype == DType::F32
        && base.is_none_or(|base| float(base.dtype))
        && rows * inp * out > 0
        && fits(&x.client, rank, [rows, inp, out], address, x.dtype)
}

/// `[rows, k] x [k, rank]`: `h[n, j] = sum_k (u[n, k] · scale) v[j, k]`, `v` `[rank, k]` read
/// through `v_strides` (along `j`, along `k`), into a new `[rows, rank]` f32 tensor.
fn down(
    u: &CubeTensor,
    v: &CubeTensor,
    v_strides: (usize, usize),
    rank: usize,
    scale: Option<f32>,
) -> CubeTensor {
    let [rows, k] = u.meta.shape().dims::<2>();
    let client = u.client.clone();
    let (cube_dim, blocks, runs, run) = down_launch(&client, rows, k, rank);
    let lanes = cube_dim.0 as usize;
    let h = empty(u, Shape::new([rows, rank]), DType::F32);
    let partial = (runs > 1).then(|| empty(u, Shape::new([runs, rows, rank]), DType::F32));
    let target = partial.as_ref().unwrap_or(&h);
    let cap = if u.dtype.size() == 2 { 8 } else { 4 };
    let n = vector_size(&[u], cap);
    let address = address_type(&[u, v, target]);
    let k_major = v_strides.1 == 1;
    probe::DOWN_LAUNCHES.fetch_add(1, Ordering::SeqCst);
    // SAFETY: every unit reads rows below `rows` and columns below `k` of `u` (a unit stride along
    // `k`, `k` a multiple of `n`), `v` within `[rank, k]` through its strides, and writes its
    // rows' outputs of run `s` within `[runs, rows, rank]`.
    unsafe {
        down_kernel::launch_unchecked(
            &client,
            CubeCount::Static(blocks as u32, runs as u32, 1),
            CubeDim::new_2d(cube_dim.0, cube_dim.1),
            address,
            n,
            u.clone().into_tensor_arg(),
            v.clone().into_tensor_arg(),
            target.clone().into_tensor_arg(),
            v_strides.0,
            v_strides.1,
            run,
            scale.unwrap_or(1.0),
            rank,
            lanes,
            down_tile(rank),
            scale.is_some(),
            k_major,
            dtype_to_storage_type(u.dtype),
        );
    }
    if let Some(partial) = partial {
        sum(&partial, &h, (rank, 1), rank);
    }
    h
}

/// `out = sum_s partial[s]` in order, for `partial` `[S, C, rank]`, written to `out` through
/// `out_strides` (along `c`, along `j`).
fn sum(partial: &CubeTensor, out: &CubeTensor, out_strides: (usize, usize), rank: usize) {
    let [_, cols, _] = partial.meta.shape().dims::<3>();
    let address = address_type(&[partial, out]);
    probe::SUM_LAUNCHES.fetch_add(1, Ordering::SeqCst);
    // SAFETY: unit `i` reads `partial[s, i]` for `i < C · rank` and writes element `(c, j)` of
    // `out`, within it by its strides.
    unsafe {
        sum_kernel::launch_unchecked(
            &partial.client,
            CubeCount::Static((cols * rank).div_ceil(CUBE) as u32, 1, 1),
            CubeDim::new_1d(CUBE as u32),
            address,
            partial.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            out_strides.0,
            out_strides.1,
            rank,
        );
    }
}

/// `[rows, rank] x [rank, M]`: `out = base + scale · h w` in `dtype` (the base's, when there is
/// one), the factor `w` `[rank, M]` taken with a unit stride along `M` (copied to one if it has
/// another).
fn up(
    h: &CubeTensor,
    w: CubeTensor,
    base: Option<&CubeTensor>,
    scale: Option<f32>,
    dtype: DType,
) -> CubeTensor {
    let [rows, rank] = h.meta.shape().dims::<2>();
    let w = rows_last(w);
    let cols = w.meta.shape()[1];
    let client = h.client.clone();
    let out = empty(h, Shape::new([rows, cols]), dtype);
    let mut vectors = vec![&out, &w];
    vectors.extend(base);
    let n = vector_size(&vectors, MAX_VECTOR.min((64 / rank).max(1)));
    let per_row = cols / n;
    let rows_per_unit = up_rows(rows, per_row);
    let units = rows.div_ceil(rows_per_unit) * per_row;
    let mut tensors = vec![h, &w, &out];
    tensors.extend(base);
    let address = address_type(&tensors);
    let base_dtype = base.map_or(dtype, |base| base.dtype);
    probe::UP_LAUNCHES.fetch_add(1, Ordering::SeqCst);
    // SAFETY: unit `i` owns columns `[n (i % per_row), n (i % per_row + 1))` of rows below `rows`,
    // within `out` and the base, and reads those columns of `w`'s `rank` rows (each a unit stride
    // along its rows, `cols` a multiple of `n`), and `h` within `[rows, rank]`.
    unsafe {
        up_kernel::launch_unchecked(
            &client,
            CubeCount::Static(units.div_ceil(CUBE) as u32, 1, 1),
            CubeDim::new_1d(CUBE as u32),
            address,
            n,
            h.clone().into_tensor_arg(),
            w.clone().into_tensor_arg(),
            base.map(|base| base.clone().into_tensor_arg()).into(),
            out.clone().into_tensor_arg(),
            rows_per_unit,
            scale.unwrap_or(1.0),
            rank,
            scale.is_some(),
            [
                dtype_to_storage_type(base_dtype),
                dtype_to_storage_type(dtype),
            ],
        );
    }
    out
}

/// `[rank, rows] x [rows, C]`: `out[c, j] = sum_n (u[n, c] · scale) g[n, j]`, into a new f32
/// tensor of `out_shape` written through `out_strides` (along `c`, along `j`).
fn gather(
    u: &CubeTensor,
    g: &CubeTensor,
    out_shape: Shape,
    out_strides: (usize, usize),
    scale: Option<f32>,
) -> CubeTensor {
    let [rows, cols] = u.meta.shape().dims::<2>();
    let [_, rank] = g.meta.shape().dims::<2>();
    let client = u.client.clone();
    let out = empty(u, out_shape, DType::F32);
    let n = vector_size(&[u], MAX_VECTOR);
    let (cube_dim, blocks, runs, run) = gather_launch(rows, cols / n, rank);
    let lanes = cube_dim.1 as usize;
    let partial = (runs > 1).then(|| empty(u, Shape::new([runs, cols, rank]), DType::F32));
    let (target, strides) = match &partial {
        Some(partial) => (partial, (cols * rank, rank, 1)),
        None => (&out, (0, out_strides.0, out_strides.1)),
    };
    let address = address_type(&[u, g, target]);
    probe::GATHER_LAUNCHES.fetch_add(1, Ordering::SeqCst);
    // SAFETY: each unit reads one vector of columns of the rows of its run, within `u` (a unit
    // stride along its rows, `cols` a multiple of `n`), `g` within `[rows, rank]`, and writes
    // its outputs within `target` through the strides given for it.
    unsafe {
        gather_kernel::launch_unchecked(
            &client,
            CubeCount::Static(blocks as u32, runs as u32, 1),
            CubeDim::new_2d(cube_dim.0, cube_dim.1),
            address,
            n,
            u.clone().into_tensor_arg(),
            g.clone().into_tensor_arg(),
            target.clone().into_tensor_arg(),
            strides.0,
            strides.1,
            strides.2,
            run,
            scale.unwrap_or(1.0),
            rank,
            lanes,
            scale.is_some(),
            dtype_to_storage_type(u.dtype),
        );
    }
    if let Some(partial) = partial {
        sum(&partial, &out, out_strides, rank);
    }
    out
}

/// A factor's strides `(along its rows, along its columns)`.
fn strides(tensor: &CubeTensor) -> (usize, usize) {
    let strides = tensor.meta.strides();
    (strides[0], strides[1])
}

impl LowRankBackend for CubeBackend {
    fn low_rank(
        x: FloatTensor<Self>,
        u: FloatTensor<Self>,
        v: FloatTensor<Self>,
        base: MaybeTensor<Self>,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let base = base.into_option();
        if !applies(&x, &u, &v, base.as_ref()) {
            return reference::forward::<Self>(x, u, v, base.into(), scale);
        }
        let [_, rank] = u.shape().dims::<2>();
        let x = rows_last(x);
        let (u, v) = (untile(u), untile(v));
        let base = base.map(rows_last);
        // `h = x U`: `down` reads `U` as `[r, in]`.
        let (u_k, u_j) = strides(&u);
        let h = down(&x, &u, (u_j, u_k), rank, None);
        // `y = base + scale · h V`.
        let dtype = base.as_ref().map_or(DType::F32, |base| base.dtype);
        let y = up(&h, v, base.as_ref(), Some(scale), dtype);
        (y, h)
    }

    fn low_rank_backward(
        x: FloatTensor<Self>,
        u: FloatTensor<Self>,
        v: FloatTensor<Self>,
        h: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        scale: f32,
        wanted: Wanted,
    ) -> (MaybeTensor<Self>, MaybeTensor<Self>, MaybeTensor<Self>) {
        if !applies(&x, &u, &v, Some(&grad)) || h.dtype != DType::F32 {
            return reference::backward::<Self>(x, u, v, h, grad, scale, wanted);
        }
        let [inp, rank] = u.shape().dims::<2>();
        let [_, out] = v.shape().dims::<2>();
        let x_dtype = x.dtype;
        let grad = rows_last(grad);
        let (u, v) = (untile(u), untile(v));
        let h = into_contiguous(untile(h));
        // `dV = hᵀ dz`, `[r, out]`, written as its transpose `[out, r]` is read.
        let dv = wanted
            .v
            .then(|| gather(&grad, &h, Shape::new([rank, out]), (1, out), Some(scale)));
        // `dh = dz Vᵀ`, `V` read as `[r, out]`, with a unit stride along `out` (copied to one if
        // it has another: staged along it, the slices take half the time).
        let dh = (wanted.x || wanted.u).then(|| {
            let v = rows_last(v.clone());
            let (v_j, v_m) = strides(&v);
            down(&grad, &v, (v_j, v_m), rank, Some(scale))
        });
        // `dU = xᵀ dh`, `[in, r]`.
        let du = match (&dh, wanted.u) {
            (Some(dh), true) => {
                let x = rows_last(x);
                Some(gather(&x, dh, Shape::new([inp, rank]), (rank, 1), None))
            }
            _ => None,
        };
        // `dx = dh Uᵀ`.
        let dx = match (&dh, wanted.x) {
            (Some(dh), true) => {
                let u_t = <CubeBackend as FloatTensorOps<CubeBackend>>::float_transpose(u);
                Some(up(dh, u_t, None, None, x_dtype))
            }
            _ => None,
        };
        (dx.into(), du.into(), dv.into())
    }
}
