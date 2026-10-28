//! The CubeCL path, and the two passes over the vocabulary it shares with [`crate::loss`], on
//! devices with what [`pass_requirements`] and [`gradient_requirements`] list.
//!
//! [`pass`], per block of rows and chunk of the vocabulary:
//!
//! 1. `z = hidden @ weight_c^T`, `[rows, C]`: a tensor-core matmul from bf16 operands straight
//!    to f32 (Burn's own matmul would round its output back to bf16);
//! 2. [`chunk_kernel`], one cube per row: the masked online log-sum-exp update of the running
//!    state, and, when `o` is accumulated, the softmax weights `exp(z - m)` with the target's
//!    and the masked columns zeroed, split into a bf16 part and a bf16 remainder;
//! 3. when `o` is accumulated, both halves times `weight_c` (one matmul over the two stacked,
//!    or two), and [`accumulate_kernel`] folds them into `o` after rescaling it by
//!    `exp(m_old - m_new)`.
//!
//! For the chosen-token log-probability, [`finish_rows_kernel`] and [`finish_grad_kernel`] then
//! turn the state into the outputs.
//!
//! [`gradient_pass`], per chunk of the vocabulary and block of rows: the logits again, then
//! [`gradient_kernel`] writes the gradient of the logits `g` (in the operands' dtype: bf16
//! rounds it for the tensor cores, as PyTorch's bf16 linear layer does), `g weight_c` is added to
//! the gradient of `hidden`, and once every block has its rows of `g`, `g^T hidden` is the
//! chunk's rows of the gradient of `weight`, written in place.

use super::reference::Rows;
use super::{ChosenLogprobBackend, LogprobOptions, RowBlocks, VocabChunk, coefficient, probe};
use crate::cube::Requirements;
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::ops::{FloatTensorOps, IntTensorOps};
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{DType, FloatDType, Shape, Slice, TensorData, TensorMetadata};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::kernel::matmul::{MatmulStrategy, matmul};
use burn_cubecl::ops::numeric::{empty_device_contiguous_dtype, zeros_client};
use burn_cubecl::tensor::CubeTensor;
use cubecl::client::Client;
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Columns of the per-row state: the running maximum, `s`, `r`, `a`, the target's logit, whether
/// it was seen, the last chunk's rescaling factor `exp(m_old - m_new)`, and the sum of the
/// unmasked logits.
const STATE: usize = 8;
const M: usize = 0;
const S: usize = 1;
const R: usize = 2;
const A: usize = 3;
const ZT: usize = 4;
const FOUND: usize = 5;
const ALPHA: usize = 6;
const ZSUM: usize = 7;

/// Threads per cube of [`chunk_kernel`] and [`gradient_kernel`], one cube per row; at most 32
/// planes.
const CHUNK_THREADS: u32 = 256;

/// The `f32` slots of [`chunk_kernel`]'s shared memory: six sums of up to 32 planes' partials.
const SCRATCH: usize = 192;

/// The most rows given to one gradient matmul `[rows, C] x [C, D]`. Burn's default matmul kernel
/// (no autotune) slows nine-fold from 4096 rows on, bf16 to f32 on an RTX 5070 Ti (7 TFLOPS at
/// 4096 rows, 64 at 3072); below that, one matmul over twice the rows beats two.
const STACK_ROWS: usize = 3584;

/// One chunk of the vocabulary for every row of a block: the cube `CUBE_POS_X` owns row
/// `CUBE_POS_X` of `z` `[rows, w]` (the logits of tokens `c0..c0 + w`), which is row
/// `r0 + CUBE_POS_X` of `targets` and `state`. Every 2-D tensor is indexed through its strides:
/// Burn allocates them with padded rows.
#[cube(launch_unchecked)]
fn chunk_kernel<P: Float, I: Numeric>(
    z: &Tensor<f32>,
    targets: &Tensor<I>,
    masked: &Tensor<u32>,
    state: &mut Tensor<f32>,
    weights: &mut Tensor<P>,
    weights_lo: &mut Tensor<P>,
    r0: u32,
    c0: u32,
    w: u32,
    masked_start: u32,
    masked_end: u32,
    #[comptime] write_weights: bool,
    #[comptime] split: bool,
    #[comptime] stacked: bool,
    #[comptime] logit_sum: bool,
    #[define(P, I)] _dtypes: [ElemType; 2],
) {
    let row = CUBE_POS_X as usize;
    let at = r0 as usize + row;
    let threads = CUBE_DIM_X as usize;
    let width = w as usize;
    let z_row = row * z.stride(0);
    let z_col = z.stride(1);
    let (w_row, w_col) = (weights.stride(0), weights.stride(1));
    let (l_row, l_col) = (weights_lo.stride(0), weights_lo.stride(1));
    let mut scratch = Shared::<[f32]>::new_slice(SCRATCH);

    // The target's column in this chunk, or `width` if it is elsewhere (or out of range).
    let target = i64::cast_from(targets[at * targets.stride(0)]) - i64::cast_from(c0);
    let mut target_col = width;
    if target >= 0 && target < i64::cast_from(w) {
        target_col = usize::cast_from(target);
    }

    // Pass 1: the chunk's maximum over the unmasked tokens.
    let mut local_max = f32::new(-3.402_823_5e38_f32);
    let mut j = UNIT_POS_X as usize;
    while j < width {
        if !is_masked(masked, masked_start, masked_end, c0 + j as u32) {
            local_max = max(local_max, z[z_row + j * z_col]);
        }
        j += threads;
    }
    let chunk_max = cube_max(local_max, &mut scratch);

    let base = at * state.stride(0);
    let col = state.stride(1);
    let m_old = state[base + M * col];
    let m_new = max(m_old, chunk_max);

    // Pass 2: exp(z - m_new) and the sums; the gradient's weights.
    let mut s = 0.0f32;
    let mut r = 0.0f32;
    let mut a = 0.0f32;
    let mut zt = 0.0f32;
    let mut found = 0.0f32;
    let mut zs = 0.0f32;
    let mut j = UNIT_POS_X as usize;
    while j < width {
        let zj = z[z_row + j * z_col];
        let skip = is_masked(masked, masked_start, masked_end, c0 + j as u32);
        let mut e = 0.0f32;
        if !skip {
            let d = zj - m_new;
            e = d.exp();
            s += e;
            a += e * d;
            if logit_sum {
                zs += zj;
            }
            if j == target_col {
                zt += zj;
                found += 1.0;
            } else {
                r += e;
            }
        }
        if write_weights {
            let mut kept = e;
            if j == target_col {
                kept = 0.0;
            }
            let hi = P::cast_from(kept);
            weights[row * w_row + j * w_col] = hi;
            if split {
                let lo = P::cast_from(kept - f32::cast_from(hi));
                if stacked {
                    weights[(CUBE_COUNT_X as usize + row) * w_row + j * w_col] = lo;
                } else {
                    weights_lo[row * l_row + j * l_col] = lo;
                }
            }
        }
        j += threads;
    }
    let s = cube_sum(s, &mut scratch, 0);
    let r = cube_sum(r, &mut scratch, 32);
    let a = cube_sum(a, &mut scratch, 64);
    let zt = cube_sum(zt, &mut scratch, 96);
    let found = cube_sum(found, &mut scratch, 128);
    let mut zs_total = 0.0f32;
    if logit_sum {
        zs_total = cube_sum(zs, &mut scratch, 160);
    }

    if UNIT_POS_X == 0 {
        let shift = m_old - m_new;
        let alpha = shift.exp();
        let s_old = state[base + S * col];
        state[base + M * col] = m_new;
        state[base + A * col] = alpha * (state[base + A * col] + s_old * shift) + a;
        state[base + S * col] = s_old * alpha + s;
        state[base + R * col] = state[base + R * col] * alpha + r;
        state[base + ZT * col] += zt;
        state[base + FOUND * col] += found;
        state[base + ALPHA * col] = alpha;
        if logit_sum {
            state[base + ZSUM * col] += zs_total;
        }
    }
}

/// Whether token `id` is among `masked[start..end]`, the sorted masked ids that fall in this
/// chunk: a binary search.
#[cube]
fn is_masked(masked: &Tensor<u32>, start: u32, end: u32, id: u32) -> bool {
    let mut lo = start;
    let mut hi = end;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if masked[mid as usize] < id {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // `masked` ends with a sentinel, so `masked[end]` is always in bounds.
    lo < end && masked[lo as usize] == id
}

/// The maximum over the cube, in every unit.
#[cube]
fn cube_max(value: f32, scratch: &mut Shared<[f32]>) -> f32 {
    let plane_value = plane_max(value);
    let plane = (UNIT_POS_X / PLANE_DIM) as usize;
    if UNIT_POS_PLANE == 0 {
        scratch[plane] = plane_value;
    }
    sync_cube();
    let planes = (CUBE_DIM_X / PLANE_DIM) as usize;
    let mut out = scratch[0];
    for k in 1..planes {
        out = max(out, scratch[k]);
    }
    sync_cube();
    out
}

/// The sum over the cube, in every unit; `slot` picks 32 entries of `scratch`.
#[cube]
fn cube_sum(value: f32, scratch: &mut Shared<[f32]>, slot: usize) -> f32 {
    let plane_value = plane_sum(value);
    let plane = (UNIT_POS_X / PLANE_DIM) as usize;
    if UNIT_POS_PLANE == 0 {
        scratch[slot + plane] = plane_value;
    }
    sync_cube();
    let planes = (CUBE_DIM_X / PLANE_DIM) as usize;
    let mut out = 0.0f32;
    for k in 0..planes {
        out += scratch[slot + k];
    }
    out
}

/// `o[r0 + row] = o[r0 + row] * alpha + m[row] (+ m_lo[row] when split)` over a block's
/// `[rows, d]`; when stacked, `m` is `[2 rows, d]` and its second half stands for `m_lo`.
#[cube(launch_unchecked)]
fn accumulate_kernel(
    o: &mut Tensor<f32>,
    m: &Tensor<f32>,
    m_lo: &Tensor<f32>,
    state: &Tensor<f32>,
    rows: u32,
    d: u32,
    r0: u32,
    #[comptime] split: bool,
    #[comptime] stacked: bool,
) {
    let i = ABSOLUTE_POS;
    let (rows, d) = (rows as usize, d as usize);
    if i >= rows * d {
        terminate!();
    }
    let row = i / d;
    let col = i % d;
    let at = r0 as usize + row;
    let o_at = at * o.stride(0) + col * o.stride(1);
    let alpha = state[at * state.stride(0) + ALPHA * state.stride(1)];
    let mut acc = o[o_at] * alpha + m[row * m.stride(0) + col * m.stride(1)];
    if split {
        if stacked {
            acc += m[(rows + row) * m.stride(0) + col * m.stride(1)];
        } else {
            acc += m_lo[row * m_lo.stride(0) + col * m_lo.stride(1)];
        }
    }
    o[o_at] = acc;
}

/// Per row: `logprob = zt - (m + ln s)` where the target was seen (else `neg_inf`, `-inf`, an
/// argument because WGSL spells no infinity) and `entropy = ln s - a / s`.
#[cube(launch_unchecked)]
fn finish_rows_kernel(
    state: &Tensor<f32>,
    logprob: &mut Tensor<f32>,
    entropy: &mut Tensor<f32>,
    rows: u32,
    neg_inf: f32,
) {
    let row = ABSOLUTE_POS;
    if row >= rows as usize {
        terminate!();
    }
    let base = row * state.stride(0);
    let col = state.stride(1);
    let s = state[base + S * col];
    let ln_s = s.ln();
    let lp_at = row * logprob.stride(0);
    let ent_at = row * entropy.stride(0);
    if state[base + FOUND * col] > 0.5 {
        logprob[lp_at] = state[base + ZT * col] - (state[base + M * col] + ln_s);
    } else {
        logprob[lp_at] = neg_inf;
    }
    entropy[ent_at] = ln_s - state[base + A * col] / s;
}

/// In place over `o` `[rows, d]`: `o = (r weight[t] - o) / s`, the gradient, with the `weight[t]`
/// term only where the target was seen; one unit per element.
#[cube(launch_unchecked)]
fn finish_grad_kernel<W: Float, I: Numeric>(
    state: &Tensor<f32>,
    o: &mut Tensor<f32>,
    weight: &Tensor<W>,
    targets: &Tensor<I>,
    rows: u32,
    d: u32,
    v: u32,
    #[define(W, I)] _dtypes: [ElemType; 2],
) {
    let i = ABSOLUTE_POS;
    let (rows, d) = (rows as usize, d as usize);
    if i >= rows * d {
        terminate!();
    }
    let row = i / d;
    let col = i % d;
    let base = row * state.stride(0);
    let scol = state.stride(1);
    let at = row * o.stride(0) + col * o.stride(1);
    let mut acc = -o[at];
    if state[base + FOUND * scol] > 0.5 {
        let t = usize::cast_from(i64::cast_from(targets[row * targets.stride(0)]));
        if t < v as usize {
            let wt = weight[t * weight.stride(0) + col * weight.stride(1)];
            acc += state[base + R * scol] * f32::cast_from(wt);
        }
    }
    o[at] = acc / state[base + S * scol];
}

/// The gradient of one chunk of logits for a block of rows: the cube `CUBE_POS_X` owns row
/// `CUBE_POS_X` of `z` `[rows, w]`, which is row `r0 + CUBE_POS_X` of `coef` (`[COUNT, n]`)
/// and `targets`, and writes row `g_r0 + CUBE_POS_X` of `g` (see [`gradient_pass`]).
#[cube(launch_unchecked)]
fn gradient_kernel<P: Float, I: Numeric>(
    z: &Tensor<f32>,
    coef: &Tensor<f32>,
    targets: &Tensor<I>,
    masked: &Tensor<u32>,
    g: &mut Tensor<P>,
    r0: u32,
    g_r0: u32,
    c0: u32,
    w: u32,
    masked_start: u32,
    masked_end: u32,
    #[comptime] entropy: bool,
    #[define(P, I)] _dtypes: [ElemType; 2],
) {
    let row = CUBE_POS_X as usize;
    let at = r0 as usize + row;
    let threads = CUBE_DIM_X as usize;
    let width = w as usize;
    let (k_row, k_col) = (coef.stride(0), coef.stride(1));
    let lse = coef[coefficient::LSE * k_row + at * k_col];
    let softmax = coef[coefficient::SOFTMAX * k_row + at * k_col];
    let uniform = coef[coefficient::UNIFORM * k_row + at * k_col];
    let target_value = coef[coefficient::TARGET * k_row + at * k_col] + uniform;
    let mut ent = 0.0f32;
    if entropy {
        ent = coef[coefficient::ENTROPY * k_row + at * k_col];
    }

    let target = i64::cast_from(targets[at * targets.stride(0)]) - i64::cast_from(c0);
    let mut target_col = width;
    if target >= 0 && target < i64::cast_from(w) {
        target_col = usize::cast_from(target);
    }
    let z_row = row * z.stride(0);
    let z_col = z.stride(1);
    let g_row = (g_r0 as usize + row) * g.stride(0);
    let g_col = g.stride(1);
    let mut j = UNIT_POS_X as usize;
    while j < width {
        let mut value = 0.0f32;
        if !is_masked(masked, masked_start, masked_end, c0 + j as u32) {
            if j == target_col {
                value = target_value;
            } else {
                let dz = z[z_row + j * z_col] - lse;
                let p = dz.exp();
                if entropy {
                    value = p * (softmax - ent * dz) + uniform;
                } else {
                    value = p * softmax + uniform;
                }
            }
        }
        g[g_row + j * g_col] = P::cast_from(value);
        j += threads;
    }
}

/// `acc[r0 + row] += m[row]` over a block's `[rows, d]`.
#[cube(launch_unchecked)]
fn add_rows_kernel(acc: &mut Tensor<f32>, m: &Tensor<f32>, rows: u32, d: u32, r0: u32) {
    let i = ABSOLUTE_POS;
    let (rows, d) = (rows as usize, d as usize);
    if i >= rows * d {
        terminate!();
    }
    let row = i / d;
    let col = i % d;
    let at = (r0 as usize + row) * acc.stride(0) + col * acc.stride(1);
    acc[at] += m[row * m.stride(0) + col * m.stride(1)];
}

/// `out[r0 + row] = m[row]` over `[rows, d]`, rounded to `out`'s dtype.
#[cube(launch_unchecked)]
fn write_rows_kernel<W: Float>(
    out: &mut Tensor<W>,
    m: &Tensor<f32>,
    rows: u32,
    d: u32,
    r0: u32,
    #[define(W)] _dtype: ElemType,
) {
    let i = ABSOLUTE_POS;
    let (rows, d) = (rows as usize, d as usize);
    if i >= rows * d {
        terminate!();
    }
    let row = i / d;
    let col = i % d;
    let at = (r0 as usize + row) * out.stride(0) + col * out.stride(1);
    out[at] = W::cast_from(m[row * m.stride(0) + col * m.stride(1)]);
}

/// What a [`pass`] over the vocabulary asks of the device for blocks of at most `rows` rows:
/// [`chunk_kernel`]'s cubes of 256 units, one per row of a block, which sum over planes of any
/// width at least 8 (one partial sum per plane in 32 slots of shared memory, [`SCRATCH`] f32 in
/// all), binding six tensors, and 64-bit integers for the targets' columns (a target may be
/// negative, an ignored row's). The other kernels are elementwise, sized by the device.
pub(crate) fn pass_requirements(rows: usize) -> Requirements {
    Requirements {
        shared_bytes: 4 * SCRATCH,
        min_plane_width: Some(CHUNK_THREADS / 32),
        int64: true,
        ..Requirements::new((CHUNK_THREADS, 1), (rows, 1), 6)
    }
}

/// What a [`gradient_pass`] asks for blocks of at most `rows` rows: [`gradient_kernel`]'s cubes
/// of 256 units, one per row of a block, binding five tensors, and 64-bit integers as the pass's
/// kernel; no shared memory and no plane operations.
pub(crate) fn gradient_requirements(rows: usize) -> Requirements {
    Requirements {
        int64: true,
        ..Requirements::new((CHUNK_THREADS, 1), (rows, 1), 5)
    }
}

/// Whether the kernels take a pass over the vocabulary (or, with `gradient`, the gradient pass)
/// for `n` rows of `hidden`, `weight` and `targets` in `dtypes`, chunked by `options`, on the
/// device `client` serves: its requirements met for the largest block of rows, in the dtypes
/// the kernels read (the operands', the targets', the weight's and the masked ids' `u32`).
pub(crate) fn kernels_take(
    client: &Client,
    [hidden, weight, targets]: [DType; 3],
    n: usize,
    options: LogprobOptions,
    gradient: bool,
) -> bool {
    let rows = options.row_blocks(n).next().map_or(0, |block| block.len());
    let requirements = match gradient {
        true => gradient_requirements(rows),
        false => pass_requirements(rows),
    };
    let operand = operand_dtype(hidden, weight);
    requirements.met_on(client, &[operand, weight, targets, DType::U32])
}

/// [`kernels_take`] for these tensors.
pub(crate) fn kernels_take_these(
    hidden: &CubeTensor,
    weight: &CubeTensor,
    targets: &CubeTensor,
    options: LogprobOptions,
    gradient: bool,
) -> bool {
    let dtypes = [hidden.dtype, weight.dtype, targets.dtype];
    kernels_take(&hidden.client, dtypes, hidden.shape()[0], options, gradient)
}

impl ChosenLogprobBackend for CubeBackend {
    fn chosen_logprob(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        masked: Vec<u32>,
        options: LogprobOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        if !kernels_take_these(&hidden, &weight, &targets, options, false) {
            let out =
                super::reference::forward::<Self>(hidden, weight, targets, &masked, options, false);
            return (out.logprob, out.entropy);
        }
        let (logprob, entropy, _) = forward(hidden, weight, targets, &masked, options, false);
        (logprob, entropy)
    }

    fn chosen_logprob_with_grad(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        masked: Vec<u32>,
        options: LogprobOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (logprob, entropy, grad) =
            match kernels_take_these(&hidden, &weight, &targets, options, false) {
                true => forward(hidden, weight, targets, &masked, options, true),
                false => {
                    let out = super::reference::forward::<Self>(
                        hidden, weight, targets, &masked, options, true,
                    );
                    (out.logprob, out.entropy, out.grad)
                }
            };
        (logprob, entropy, grad.expect("asked for the gradient"))
    }
}

type B = CubeBackend;

/// The chosen-token log-probability: one [`pass`], then the finishing kernels.
fn forward(
    hidden: CubeTensor,
    weight: CubeTensor,
    targets: CubeTensor,
    masked: &[u32],
    options: LogprobOptions,
    with_grad: bool,
) -> (CubeTensor, CubeTensor, Option<CubeTensor>) {
    let [n, d] = hidden.shape().dims::<2>();
    let [v, _] = weight.shape().dims::<2>();
    let device = hidden.device.clone();
    if n == 0 {
        let empty = |shape: Shape| B::float_zeros(shape, &device, FloatDType::F32);
        return (
            empty(Shape::new([0])),
            empty(Shape::new([0])),
            with_grad.then(|| empty(Shape::new([0, d]))),
        );
    }
    let client = hidden.client.clone();
    let targets = into_contiguous(targets);
    let (state, o) = pass(
        hidden,
        weight.clone(),
        targets.clone(),
        &super::reference::sorted(masked),
        options,
        with_grad,
        false,
        &probe::KERNEL_LAUNCHES,
    );

    let output = || {
        empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new([n]), DType::F32)
    };
    let (logprob, entropy) = (output(), output());
    let cube_dim = CubeDim::new(&client, n);
    let cube_count = calculate_cube_count_elemwise(&client, n, cube_dim);
    unsafe {
        finish_rows_kernel::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            state.clone().into_tensor_arg(),
            logprob.clone().into_tensor_arg(),
            entropy.clone().into_tensor_arg(),
            n as u32,
            f32::NEG_INFINITY,
        );
    }
    // The gradient, in place over `o`.
    if let Some(o) = &o {
        let len = n * d;
        let cube_dim = CubeDim::new(&client, len);
        let cube_count = calculate_cube_count_elemwise(&client, len, cube_dim);
        let weight = into_contiguous(weight);
        unsafe {
            finish_grad_kernel::launch_unchecked(
                &client,
                cube_count,
                cube_dim,
                state.into_tensor_arg(),
                o.clone().into_tensor_arg(),
                weight.clone().into_tensor_arg(),
                targets.clone().into_tensor_arg(),
                n as u32,
                d as u32,
                v as u32,
                [
                    dtype_to_storage_type(weight.dtype),
                    dtype_to_storage_type(targets.dtype),
                ],
            );
        }
    }
    (logprob, entropy, o)
}

/// The matmuls take both operands in one dtype: bf16 when both are, else f32. The bf16 path
/// splits `o`'s softmax weights in two bf16 parts; f16 parts would lose the probabilities below
/// 6e-8, which can add up over a large vocabulary, so f16 goes through f32 too.
fn operand_dtype(hidden: DType, weight: DType) -> DType {
    match (hidden, weight) {
        (DType::BF16, DType::BF16) => DType::BF16,
        _ => DType::F32,
    }
}

/// `hidden` in the operands' dtype ([`operand_dtype`]), contiguous, and that dtype; the weight
/// is cast chunk by chunk ([`weight_rows`]).
fn operands(hidden: CubeTensor, weight: &CubeTensor) -> (CubeTensor, DType) {
    let operand = operand_dtype(hidden.dtype, weight.dtype);
    let hidden = match hidden.dtype == operand {
        true => hidden,
        false => B::float_cast(hidden, FloatDType::F32),
    };
    (into_contiguous(hidden), operand)
}

/// Rows `range` of the (contiguous) weight in the operands' dtype.
fn weight_rows(weight: &CubeTensor, range: Range<usize>, operand: DType) -> CubeTensor {
    let rows = B::float_slice(weight.clone(), &[Slice::from(range), Slice::full()]);
    match rows.dtype == operand {
        true => rows,
        false => B::float_cast(rows, operand.into()),
    }
}

/// The sorted masked ids followed by a sentinel, on the device.
fn masked_ids(masked: &[u32], device: &burn_cubecl::CubeDevice) -> CubeTensor {
    let mut ids = masked.to_vec();
    ids.push(u32::MAX);
    B::int_from_data(TensorData::new(ids.clone(), [ids.len()]), device)
}

/// The masked ids' positions in `masked` that fall in the chunk `c0..c1`.
fn masked_range(masked: &[u32], c0: usize, c1: usize) -> (u32, u32) {
    (
        masked.partition_point(|&id| (id as usize) < c0) as u32,
        masked.partition_point(|&id| (id as usize) < c1) as u32,
    )
}

/// One pass over the vocabulary for all `n > 0` rows: the per-row state `[n, STATE]` f32 and,
/// when `accumulate`, `o` `[n, D]` f32; the sum of the logits too when `logit_sum`. `masked` is
/// sorted and free of repeats; each launch of [`chunk_kernel`] is counted in `launches`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pass(
    hidden: CubeTensor,
    weight: CubeTensor,
    targets: CubeTensor,
    masked: &[u32],
    options: LogprobOptions,
    accumulate: bool,
    logit_sum: bool,
    launches: &AtomicUsize,
) -> (CubeTensor, Option<CubeTensor>) {
    let [n, d] = hidden.shape().dims::<2>();
    let [v, dw] = weight.shape().dims::<2>();
    assert_eq!(d, dw, "hidden [n, D] and weight [V, D] must share D");
    assert_eq!(targets.shape().dims::<1>(), [n], "targets must be [n]");
    assert!(n > 0, "a pass needs at least one row");
    assert!(
        n <= u32::MAX as usize && v <= i32::MAX as usize && d <= u32::MAX as usize,
        "shapes beyond 32-bit indexing"
    );
    let (hidden, operand) = operands(hidden, &weight);
    let split = operand == DType::BF16;
    let weight = into_contiguous(weight);
    let targets = into_contiguous(targets);
    let client = hidden.client.clone();
    let device = hidden.device.clone();
    let masked_ids = masked_ids(masked, &device);

    let mut init = vec![0.0f32; n * STATE];
    for row in 0..n {
        init[row * STATE + M] = f32::MIN;
    }
    let state = B::float_from_data(TensorData::new(init, [n, STATE]), &device);
    let o = accumulate.then(|| {
        zeros_client(
            client.clone(),
            device.clone(),
            Shape::new([n, d]),
            DType::F32,
        )
    });
    let empty = |shape: [usize; 2], dtype: DType| {
        empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new(shape), dtype)
    };

    for block in options.row_blocks(n) {
        let (r0, r1) = (block.start, block.end);
        let rows = r1 - r0;
        let h = B::float_slice(hidden.clone(), &[Slice::from(r0..r1), Slice::full()]);
        // Both halves of the split weights go through one matmul of 2 x rows rows when that
        // stays under STACK_ROWS, two matmuls of `rows` rows otherwise.
        let stacked = accumulate && split && 2 * rows <= STACK_ROWS;

        let chunk = options.vocab_chunk_for(rows);
        for c0 in (0..v).step_by(chunk) {
            let c1 = (c0 + chunk).min(v);
            let w = c1 - c0;
            let weight_c = weight_rows(&weight, c0..c1, operand);
            let z = matmul(
                h.clone(),
                B::float_transpose(weight_c.clone()),
                None,
                MatmulStrategy::default(),
                DType::F32,
            )
            .expect("the logits matmul launches");
            let (masked_start, masked_end) = masked_range(masked, c0, c1);

            // The gradient's softmax weights, and their bf16 remainder (stacked under them, or
            // apart); a one-element placeholder stands for a buffer the kernel does not write.
            let weights_rows = if stacked { 2 * rows } else { rows };
            let weights = empty(
                if accumulate {
                    [weights_rows, w]
                } else {
                    [1, 1]
                },
                operand,
            );
            let weights_lo = empty(
                if accumulate && split && !stacked {
                    [rows, w]
                } else {
                    [1, 1]
                },
                operand,
            );
            launches.fetch_add(1, Ordering::SeqCst);
            unsafe {
                chunk_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(rows as u32, 1, 1),
                    CubeDim::new_1d(CHUNK_THREADS),
                    z.into_tensor_arg(),
                    targets.clone().into_tensor_arg(),
                    masked_ids.clone().into_tensor_arg(),
                    state.clone().into_tensor_arg(),
                    weights.clone().into_tensor_arg(),
                    weights_lo.clone().into_tensor_arg(),
                    r0 as u32,
                    c0 as u32,
                    w as u32,
                    masked_start,
                    masked_end,
                    accumulate,
                    split,
                    stacked,
                    logit_sum,
                    [
                        dtype_to_storage_type(operand),
                        dtype_to_storage_type(targets.dtype),
                    ],
                );
            }

            if let Some(o) = &o {
                let gemm = |lhs: CubeTensor| {
                    matmul(
                        lhs,
                        weight_c.clone(),
                        None,
                        MatmulStrategy::default(),
                        DType::F32,
                    )
                    .expect("the gradient matmul launches")
                };
                let m = gemm(weights);
                let m_lo = if split && !stacked {
                    gemm(weights_lo)
                } else {
                    m.clone()
                };
                let len = rows * d;
                let cube_dim = CubeDim::new(&client, len);
                let cube_count = calculate_cube_count_elemwise(&client, len, cube_dim);
                unsafe {
                    accumulate_kernel::launch_unchecked(
                        &client,
                        cube_count,
                        cube_dim,
                        o.clone().into_tensor_arg(),
                        m.into_tensor_arg(),
                        m_lo.into_tensor_arg(),
                        state.clone().into_tensor_arg(),
                        rows as u32,
                        d as u32,
                        r0 as u32,
                        split,
                        stacked,
                    );
                }
            }
        }
    }
    (state, o)
}

/// The columns of a pass's state as [`Rows`], each `[n]` f32.
pub(crate) fn rows(state: CubeTensor, logit_sum: bool) -> Rows<CubeBackend> {
    let n = state.shape()[0];
    let column = |k: usize| {
        let x = B::float_slice(state.clone(), &[Slice::full(), Slice::from(k..k + 1)]);
        B::float_reshape(x, Shape::new([n]))
    };
    Rows {
        m: column(M),
        s: column(S),
        r: column(R),
        a: column(A),
        zt: column(ZT),
        found: column(FOUND),
        zsum: logit_sum.then(|| column(ZSUM)),
    }
}

/// The width of the column slices of `g` that go through one weight-gradient matmul
/// `[w, n] x [n, D]`: twice `D`, at most the chunk. Burn's default matmul kernel (no autotune)
/// picks a slow instruction shape for a product four times taller than wide or wider than tall
/// (on an RTX 5070 Ti, `[4096, n] x [n, 1024]` runs at 8 TFLOPS, `[2048, n] x [n, 1024]` at 57 to
/// 68), and `[w, n]` is `g`'s transpose, read in place.
fn weight_slice(chunk: usize, d: usize) -> usize {
    (2 * d).clamp(1, chunk)
}

/// The gradient pass (see [`super::reference::gradient_pass`] for what it computes): returns
/// the gradients in `hidden`'s and `weight`'s dtypes, each `[0, D]` when not wanted. `masked` is
/// sorted and free of repeats; each launch of [`gradient_kernel`] is counted in `launches`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_pass(
    hidden: CubeTensor,
    weight: CubeTensor,
    targets: CubeTensor,
    masked: &[u32],
    options: LogprobOptions,
    coef: CubeTensor,
    entropy: bool,
    want_hidden: bool,
    want_weight: bool,
    launches: &AtomicUsize,
) -> (CubeTensor, CubeTensor) {
    let [n, d] = hidden.shape().dims::<2>();
    let [v, dw] = weight.shape().dims::<2>();
    assert_eq!(d, dw, "hidden [n, D] and weight [V, D] must share D");
    assert!(
        n <= u32::MAX as usize && v <= i32::MAX as usize && d <= u32::MAX as usize,
        "shapes beyond 32-bit indexing"
    );
    let (hidden_dtype, weight_dtype) = (hidden.dtype, weight.dtype);
    let client = hidden.client.clone();
    let device = hidden.device.clone();
    let zeros = |shape: [usize; 2], dtype: DType| {
        zeros_client(client.clone(), device.clone(), Shape::new(shape), dtype)
    };
    let empty = |shape: [usize; 2], dtype: DType| {
        empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new(shape), dtype)
    };
    if n == 0 {
        return (
            zeros([0, d], hidden_dtype),
            zeros([if want_weight { v } else { 0 }, d], weight_dtype),
        );
    }
    let (hidden, operand) = operands(hidden, &weight);
    let weight = into_contiguous(weight);
    let targets = into_contiguous(targets);
    let coef = into_contiguous(coef);
    let masked_ids = masked_ids(masked, &device);
    let acc = want_hidden.then(|| zeros([n, d], DType::F32));
    let grad_weight = want_weight.then(|| empty([v, d], weight_dtype));
    let blocks: Vec<_> = options.row_blocks(n).collect();
    let chunk = options.vocab_chunk_for(blocks[0].len());
    let slice = weight_slice(chunk, d);

    for c0 in (0..v).step_by(chunk) {
        let c1 = (c0 + chunk).min(v);
        let w = c1 - c0;
        let weight_c = weight_rows(&weight, c0..c1, operand);
        // The chunk's `g` for every row when the weight's gradient is wanted, else one block's.
        let g_all = want_weight.then(|| empty([n, w], operand));
        let (masked_start, masked_end) = masked_range(masked, c0, c1);
        for block in &blocks {
            let (r0, r1) = (block.start, block.end);
            let rows = r1 - r0;
            let h = B::float_slice(hidden.clone(), &[Slice::from(r0..r1), Slice::full()]);
            let z = matmul(
                h,
                B::float_transpose(weight_c.clone()),
                None,
                MatmulStrategy::default(),
                DType::F32,
            )
            .expect("the logits matmul launches");
            let (g, g_r0) = match &g_all {
                Some(g) => (g.clone(), r0),
                None => (empty([rows, w], operand), 0),
            };
            launches.fetch_add(1, Ordering::SeqCst);
            unsafe {
                gradient_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(rows as u32, 1, 1),
                    CubeDim::new_1d(CHUNK_THREADS),
                    z.into_tensor_arg(),
                    coef.clone().into_tensor_arg(),
                    targets.clone().into_tensor_arg(),
                    masked_ids.clone().into_tensor_arg(),
                    g.clone().into_tensor_arg(),
                    r0 as u32,
                    g_r0 as u32,
                    c0 as u32,
                    w as u32,
                    masked_start,
                    masked_end,
                    entropy,
                    [
                        dtype_to_storage_type(operand),
                        dtype_to_storage_type(targets.dtype),
                    ],
                );
            }
            if let Some(acc) = &acc {
                let g = match g_all.is_some() {
                    true => B::float_slice(g, &[Slice::from(r0..r1), Slice::full()]),
                    false => g,
                };
                let m = matmul(
                    g,
                    weight_c.clone(),
                    None,
                    MatmulStrategy::default(),
                    DType::F32,
                )
                .expect("the hidden-gradient matmul launches");
                let len = rows * d;
                let cube_dim = CubeDim::new(&client, len);
                let cube_count = calculate_cube_count_elemwise(&client, len, cube_dim);
                unsafe {
                    add_rows_kernel::launch_unchecked(
                        &client,
                        cube_count,
                        cube_dim,
                        acc.clone().into_tensor_arg(),
                        m.into_tensor_arg(),
                        rows as u32,
                        d as u32,
                        r0 as u32,
                    );
                }
            }
        }
        if let (Some(g_all), Some(grad_weight)) = (g_all, &grad_weight) {
            for s0 in (0..w).step_by(slice) {
                let s1 = (s0 + slice).min(w);
                let lhs = B::float_transpose(B::float_slice(
                    g_all.clone(),
                    &[Slice::full(), Slice::from(s0..s1)],
                ));
                write_weight_rows(lhs, &hidden, grad_weight, c0 + s0);
            }
        }
    }
    let grad_hidden = match acc {
        Some(acc) if hidden_dtype == DType::F32 => acc,
        Some(acc) => B::float_cast(acc, hidden_dtype.into()),
        None => zeros([0, d], hidden_dtype),
    };
    let grad_weight = grad_weight.unwrap_or_else(|| zeros([0, d], weight_dtype));
    (grad_hidden, grad_weight)
}

/// Rows `start..start + w` of `grad_weight` `[V, D]` set to `lhs` `[w, n]` times `hidden`
/// `[n, D]`: written by the matmul in place when the rows are a view the matmul can write
/// (the dtype is the operands' and the offsets are aligned), else through an f32 product and
/// [`write_rows_kernel`].
fn write_weight_rows(lhs: CubeTensor, hidden: &CubeTensor, grad_weight: &CubeTensor, start: usize) {
    let w = lhs.shape()[0];
    let [v, d] = grad_weight.shape().dims::<2>();
    let client = grad_weight.client.clone();
    let bytes = grad_weight.dtype.size() * d;
    let alignment = client.properties().memory.alignment as usize;
    let in_place = grad_weight.dtype == hidden.dtype
        && grad_weight.is_contiguous()
        && (start * bytes).is_multiple_of(alignment)
        && ((v - start - w) * bytes).is_multiple_of(alignment);
    if in_place {
        // Aligned offsets make the slice a view of `grad_weight`, not a copy.
        let out = B::float_slice(
            grad_weight.clone(),
            &[Slice::from(start..start + w), Slice::full()],
        );
        matmul(
            lhs,
            hidden.clone(),
            Some(out),
            MatmulStrategy::default(),
            grad_weight.dtype,
        )
        .expect("the weight-gradient matmul launches");
        return;
    }
    let m = matmul(
        lhs,
        hidden.clone(),
        None,
        MatmulStrategy::default(),
        DType::F32,
    )
    .expect("the weight-gradient matmul launches");
    let len = w * d;
    let cube_dim = CubeDim::new(&client, len);
    let cube_count = calculate_cube_count_elemwise(&client, len, cube_dim);
    unsafe {
        write_rows_kernel::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            grad_weight.clone().into_tensor_arg(),
            m.into_tensor_arg(),
            w as u32,
            d as u32,
            start as u32,
            dtype_to_storage_type(grad_weight.dtype),
        );
    }
}
