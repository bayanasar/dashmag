//! The CUDA path. Per block of rows and chunk of the vocabulary:
//!
//! 1. `z = hidden @ weight_c^T`, `[rows, C]`: a tensor-core matmul from bf16 operands straight
//!    to f32 (Burn's own matmul would round its output back to bf16);
//! 2. [`chunk_kernel`], one cube per row: the masked online log-sum-exp update of the running
//!    state, and, when the gradient is wanted, the softmax weights `exp(z - m)` with the target's
//!    and the masked columns zeroed, split into a bf16 part and a bf16 remainder;
//! 3. when the gradient is wanted, both halves times `weight_c` (one matmul over the two stacked,
//!    or two), and [`accumulate_kernel`] folds them into `o` after rescaling it by
//!    `exp(m_old - m_new)`.
//!
//! Then [`finish_rows_kernel`] and [`finish_grad_kernel`] turn the state into the outputs.

use super::{ChosenLogprobBackend, LogprobOptions, probe};
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::ops::{FloatTensorOps, IntTensorOps};
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{DType, FloatDType, Shape, Slice, TensorData, TensorMetadata};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::kernel::matmul::{MatmulStrategy, matmul};
use burn_cubecl::ops::numeric::{empty_device_contiguous_dtype, zeros_client};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use std::sync::atomic::Ordering;

/// Columns of the per-row state: the running maximum, `s`, `r`, `a`, the target's logit, whether
/// it was seen, and the last chunk's rescaling factor `exp(m_old - m_new)`.
const STATE: usize = 8;
const M: usize = 0;
const S: usize = 1;
const R: usize = 2;
const A: usize = 3;
const ZT: usize = 4;
const FOUND: usize = 5;
const ALPHA: usize = 6;

/// Threads per cube of [`chunk_kernel`], one cube per row; at most 32 planes.
const CHUNK_THREADS: u32 = 256;

/// The most rows given to one gradient matmul `[rows, C] x [C, D]`. Burn's default matmul kernel
/// (no autotune) slows nine-fold from 4096 rows on, bf16 to f32 on an RTX 5070 Ti (7 TFLOPS at
/// 4096 rows, 64 at 3072); below that, one matmul over twice the rows beats two.
const STACK_ROWS: usize = 3584;

/// One chunk of the vocabulary for every row of a block: the cube `CUBE_POS_X` owns row
/// `CUBE_POS_X` of `z` `[rows, w]` (the logits of tokens `c0..c0 + w`). Every 2-D tensor is
/// indexed through its strides: Burn allocates them with padded rows.
#[cube(launch_unchecked)]
fn chunk_kernel<P: Float, I: Numeric>(
    z: &Tensor<f32>,
    targets: &Tensor<I>,
    masked: &Tensor<u32>,
    state: &mut Tensor<f32>,
    weights: &mut Tensor<P>,
    weights_lo: &mut Tensor<P>,
    c0: u32,
    w: u32,
    masked_start: u32,
    masked_end: u32,
    #[comptime] write_weights: bool,
    #[comptime] split: bool,
    #[comptime] stacked: bool,
    #[define(P, I)] _dtypes: [ElemType; 2],
) {
    let row = CUBE_POS_X as usize;
    let threads = CUBE_DIM_X as usize;
    let width = w as usize;
    let z_row = row * z.stride(0);
    let z_col = z.stride(1);
    let (w_row, w_col) = (weights.stride(0), weights.stride(1));
    let (l_row, l_col) = (weights_lo.stride(0), weights_lo.stride(1));
    let mut scratch = Shared::<[f32]>::new_slice(160usize);

    // The target's column in this chunk, or `width` if it is elsewhere (or out of range).
    let target = i64::cast_from(targets[row * targets.stride(0)]) - i64::cast_from(c0);
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

    let base = row * state.stride(0);
    let col = state.stride(1);
    let m_old = state[base + M * col];
    let m_new = max(m_old, chunk_max);

    // Pass 2: exp(z - m_new) and the sums; the gradient's weights.
    let mut s = 0.0f32;
    let mut r = 0.0f32;
    let mut a = 0.0f32;
    let mut zt = 0.0f32;
    let mut found = 0.0f32;
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
    }
}

/// Whether token `id` is among `masked[start..end]`, the masked ids that fall in this chunk.
#[cube]
fn is_masked(masked: &Tensor<u32>, start: u32, end: u32, id: u32) -> bool {
    let mut hit = false;
    for k in start..end {
        if masked[k as usize] == id {
            hit = true;
        }
    }
    hit
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

/// `o = o * alpha + m (+ m_lo when split)` over `o` `[rows, d]`; when stacked, `m` is
/// `[2 rows, d]` and its second half stands for `m_lo`.
#[cube(launch_unchecked)]
fn accumulate_kernel(
    o: &mut Tensor<f32>,
    m: &Tensor<f32>,
    m_lo: &Tensor<f32>,
    state: &Tensor<f32>,
    rows: u32,
    d: u32,
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
    let (o_row, o_col) = (o.stride(0), o.stride(1));
    let at = row * o_row + col * o_col;
    let alpha = state[row * state.stride(0) + ALPHA * state.stride(1)];
    let mut acc = o[at] * alpha + m[row * m.stride(0) + col * m.stride(1)];
    if split {
        if stacked {
            acc += m[(rows + row) * m.stride(0) + col * m.stride(1)];
        } else {
            acc += m_lo[row * m_lo.stride(0) + col * m_lo.stride(1)];
        }
    }
    o[at] = acc;
}

/// Per row: `logprob = zt - (m + ln s)` where the target was seen (else `-inf`) and
/// `entropy = ln s - a / s`, written at row `r0 + row` of the outputs.
#[cube(launch_unchecked)]
fn finish_rows_kernel(
    state: &Tensor<f32>,
    logprob: &mut Tensor<f32>,
    entropy: &mut Tensor<f32>,
    rows: u32,
    r0: u32,
) {
    let row = ABSOLUTE_POS;
    if row >= rows as usize {
        terminate!();
    }
    let base = row * state.stride(0);
    let col = state.stride(1);
    let s = state[base + S * col];
    let ln_s = s.ln();
    let lp_at = (r0 as usize + row) * logprob.stride(0);
    let ent_at = (r0 as usize + row) * entropy.stride(0);
    if state[base + FOUND * col] > 0.5 {
        logprob[lp_at] = state[base + ZT * col] - (state[base + M * col] + ln_s);
    } else {
        logprob[lp_at] = f32::new(f32::NEG_INFINITY);
    }
    entropy[ent_at] = ln_s - state[base + A * col] / s;
}

/// `grad[r0 + row] = (r weight[t] - o) / s`, with the `weight[t]` term only where the target was
/// seen; one unit per element of the block's `[rows, d]`.
#[cube(launch_unchecked)]
fn finish_grad_kernel<W: Float, I: Numeric>(
    state: &Tensor<f32>,
    o: &Tensor<f32>,
    weight: &Tensor<W>,
    targets: &Tensor<I>,
    grad: &mut Tensor<f32>,
    rows: u32,
    d: u32,
    v: u32,
    r0: u32,
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
    let mut acc = -o[row * o.stride(0) + col * o.stride(1)];
    if state[base + FOUND * scol] > 0.5 {
        let t = usize::cast_from(i64::cast_from(targets[row * targets.stride(0)]));
        if t < v as usize {
            let wt = weight[t * weight.stride(0) + col * weight.stride(1)];
            acc += state[base + R * scol] * f32::cast_from(wt);
        }
    }
    let at = (r0 as usize + row) * grad.stride(0) + col * grad.stride(1);
    grad[at] = acc / state[base + S * scol];
}

impl ChosenLogprobBackend for CubeBackend {
    fn chosen_logprob(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        masked: Vec<u32>,
        options: LogprobOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
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
        let (logprob, entropy, grad) = forward(hidden, weight, targets, &masked, options, true);
        (logprob, entropy, grad.expect("asked for the gradient"))
    }
}

type B = CubeBackend;

fn forward(
    hidden: CubeTensor,
    weight: CubeTensor,
    targets: CubeTensor,
    masked: &[u32],
    options: LogprobOptions,
    with_grad: bool,
) -> (CubeTensor, CubeTensor, Option<CubeTensor>) {
    let [n, d] = hidden.shape().dims::<2>();
    let [v, dw] = weight.shape().dims::<2>();
    assert_eq!(d, dw, "hidden [n, D] and weight [V, D] must share D");
    assert_eq!(targets.shape().dims::<1>(), [n], "targets must be [n]");
    assert!(
        n <= u32::MAX as usize && v <= i32::MAX as usize && d <= u32::MAX as usize,
        "shapes beyond 32-bit indexing"
    );
    if n == 0 {
        let device = hidden.device.clone();
        let empty = |shape: Shape| B::float_zeros(shape, &device, FloatDType::F32);
        return (
            empty(Shape::new([0])),
            empty(Shape::new([0])),
            with_grad.then(|| empty(Shape::new([0, d]))),
        );
    }

    // The matmuls take both operands in one dtype: bf16 when both are, else f32. The bf16 path
    // splits the gradient's softmax weights in two bf16 parts; f16 parts would lose the
    // probabilities below 6e-8, which can add up over a large vocabulary, so f16 goes through
    // f32 too.
    let (hidden, weight) = if hidden.dtype == DType::BF16 && weight.dtype == DType::BF16 {
        (hidden, weight)
    } else {
        (to_f32(hidden), to_f32(weight))
    };
    let operand = hidden.dtype;
    let split = operand == DType::BF16;
    let mut masked = masked.to_vec();
    masked.sort_unstable();
    masked.dedup();
    let hidden = into_contiguous(hidden);
    let weight = into_contiguous(weight);
    let targets = into_contiguous(targets);

    let client = hidden.client.clone();
    let device = hidden.device.clone();
    // The sorted masked ids, once; each chunk scans only those in its range.
    let mut masked_ids = masked.to_vec();
    masked_ids.push(u32::MAX);
    let masked_ids = B::int_from_data(
        TensorData::new(masked_ids.clone(), [masked_ids.len()]),
        &device,
    );

    let logprob =
        empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new([n]), DType::F32);
    let entropy =
        empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new([n]), DType::F32);
    let grad = with_grad.then(|| {
        empty_device_contiguous_dtype(
            client.clone(),
            device.clone(),
            Shape::new([n, d]),
            DType::F32,
        )
    });

    for block in options.row_blocks(n) {
        let (r0, r1) = (block.start, block.end);
        let rows = r1 - r0;
        let h = B::float_slice(hidden.clone(), &[Slice::from(r0..r1), Slice::full()]);
        let t = into_contiguous(B::int_slice(targets.clone(), &[Slice::from(r0..r1)]));

        let mut init = vec![0.0f32; rows * STATE];
        for row in 0..rows {
            init[row * STATE + M] = f32::MIN;
        }
        let state = B::float_from_data(TensorData::new(init, [rows, STATE]), &device);
        let o = with_grad.then(|| {
            zeros_client(
                client.clone(),
                device.clone(),
                Shape::new([rows, d]),
                DType::F32,
            )
        });
        // Both halves of the split weights go through one matmul of 2 x rows rows when that
        // stays under STACK_ROWS, two matmuls of `rows` rows otherwise.
        let stacked = with_grad && split && 2 * rows <= STACK_ROWS;
        let empty = |shape: [usize; 2], dtype: DType| {
            empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new(shape), dtype)
        };

        let chunk = options.vocab_chunk_for(rows);
        for c0 in (0..v).step_by(chunk) {
            let c1 = (c0 + chunk).min(v);
            let w = c1 - c0;
            let weight_c = B::float_slice(weight.clone(), &[Slice::from(c0..c1), Slice::full()]);
            let z = matmul(
                h.clone(),
                B::float_transpose(weight_c.clone()),
                None,
                MatmulStrategy::default(),
                DType::F32,
            )
            .expect("the logits matmul launches");

            let masked_start = masked.partition_point(|&id| (id as usize) < c0) as u32;
            let masked_end = masked.partition_point(|&id| (id as usize) < c1) as u32;

            // The gradient's softmax weights, and their bf16 remainder (stacked under them, or
            // apart); a one-element placeholder stands for a buffer the kernel does not write.
            let weights_rows = if stacked { 2 * rows } else { rows };
            let weights = empty(if with_grad { [weights_rows, w] } else { [1, 1] }, operand);
            let weights_lo = empty(
                if with_grad && split && !stacked {
                    [rows, w]
                } else {
                    [1, 1]
                },
                operand,
            );
            probe::KERNEL_LAUNCHES.fetch_add(1, Ordering::SeqCst);
            unsafe {
                chunk_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(rows as u32, 1, 1),
                    CubeDim::new_1d(CHUNK_THREADS),
                    z.into_tensor_arg(),
                    t.clone().into_tensor_arg(),
                    masked_ids.clone().into_tensor_arg(),
                    state.clone().into_tensor_arg(),
                    weights.clone().into_tensor_arg(),
                    weights_lo.clone().into_tensor_arg(),
                    c0 as u32,
                    w as u32,
                    masked_start,
                    masked_end,
                    with_grad,
                    split,
                    stacked,
                    [
                        dtype_to_storage_type(operand),
                        dtype_to_storage_type(t.dtype),
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
                        split,
                        stacked,
                    );
                }
            }
        }

        let cube_dim = CubeDim::new(&client, rows);
        let cube_count = calculate_cube_count_elemwise(&client, rows, cube_dim);
        unsafe {
            finish_rows_kernel::launch_unchecked(
                &client,
                cube_count,
                cube_dim,
                state.clone().into_tensor_arg(),
                logprob.clone().into_tensor_arg(),
                entropy.clone().into_tensor_arg(),
                rows as u32,
                r0 as u32,
            );
        }
        if let (Some(o), Some(grad)) = (o, &grad) {
            let len = rows * d;
            let cube_dim = CubeDim::new(&client, len);
            let cube_count = calculate_cube_count_elemwise(&client, len, cube_dim);
            let weight_dtype = weight.dtype;
            unsafe {
                finish_grad_kernel::launch_unchecked(
                    &client,
                    cube_count,
                    cube_dim,
                    state.into_tensor_arg(),
                    o.into_tensor_arg(),
                    weight.clone().into_tensor_arg(),
                    t.clone().into_tensor_arg(),
                    grad.clone().into_tensor_arg(),
                    rows as u32,
                    d as u32,
                    v as u32,
                    r0 as u32,
                    [
                        dtype_to_storage_type(weight_dtype),
                        dtype_to_storage_type(t.dtype),
                    ],
                );
            }
        }
    }
    (logprob, entropy, grad)
}

fn to_f32(x: CubeTensor) -> CubeTensor {
    match x.dtype {
        DType::F32 => x,
        _ => B::float_cast(x, FloatDType::F32),
    }
}
