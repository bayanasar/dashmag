//! The passing of the state from chunk to chunk in the chunked gated delta rule, forward and
//! backward: the one part of the rule that is a loop.
//!
//! Forward, for chunks `i = 0 .. N` from the initial state `S_0`:
//!
//! ```text
//! U_i     = u0_i - w_i S_i
//! S_{i+1} = d_i S_i + ks_i^T U_i
//! ```
//!
//! keeping every entering state `S_i` `[B, H, N, Dk, Dv]` and every `U_i` `[B, H, N, C, Dv]` for
//! the output and the backward pass, and the final state `S_N`. Backward, given the cotangents of
//! the entering states, of `U` and of the final state, from `dS_N` back:
//!
//! ```text
//! dU_i = dU'_i + ks_i dS_{i+1}
//! dS_i = d_i dS_{i+1} + dS'_i - w_i^T dU_i
//! ```
//!
//! after which the gradients that need every chunk are formed batched: `du0 = dU`,
//! `dw = -dU S^T`, `dks = U dS_{+1}^T`, `dd = sum(dS_{+1} * S)`.
//!
//! Each step is two small matmuls that wait on the step before, so as MLX operations the loop is
//! a few hundred tiny dispatches, bound by their latency. The Metal kernels run the whole loop in
//! one dispatch each way. A state's columns evolve independently, so each threadgroup carries a
//! slice of `TV` of them, `Dk x TV` in threadgroup memory with a `C x TV` slice of `U` beside it
//! (24 KB at `Dk = 128`, `TV = 32`, within the 32 KB a threadgroup has), through every chunk; the
//! matmuls run on 8 x 8 simdgroup matrices, in f32. Heads whose widths are not multiples of 8, or
//! too wide for a slice to fit, run the operations.

use crate::kernel::{Launch, launch};
use mlx_rs::error::Result;
use mlx_rs::fast::TemplateArg;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{addmm, stack, swap_axes, zeros_like};
use mlx_rs::transforms::custom_vjp;
use mlx_rs::{Array, Dtype};

use super::chunk::{CHUNK_SIZE, Path, counters};

/// Simdgroups per threadgroup of the scan kernels.
const SIMDGROUPS: i32 = 4;

/// Threadgroup memory of one threadgroup, in bytes.
const THREADGROUP_BYTES: i32 = 32 * 1024;

/// The state's columns one threadgroup carries, for heads `dk` by `dv`, if the kernels take them:
/// the widest of 32, 24, 16 and 8 that divides `dv` and whose slices fit.
pub(super) fn columns(dk: i32, dv: i32) -> Option<i32> {
    if dk % 8 != 0 || dv % 8 != 0 {
        return None;
    }
    [32, 24, 16, 8]
        .into_iter()
        .find(|tv| dv % tv == 0 && (dk + CHUNK_SIZE) * tv * 4 <= THREADGROUP_BYTES)
}

/// The scan with its backward: every entering state, every `U`, and the final state.
pub(super) fn scan(
    u0: &Array,
    w: &Array,
    ks: &Array,
    decay: &Array,
    state: &Array,
    path: Path,
) -> Result<(Array, Array, Array)> {
    if path == Path::Autograd {
        let mut outputs = forward_ops(u0, w, ks, decay, state)?;
        let final_state = outputs.pop().expect("three outputs");
        let u = outputs.pop().expect("three outputs");
        return Ok((outputs.pop().expect("three outputs"), u, final_state));
    }
    let [dk, dv] = [state.shape()[2], state.shape()[3]];
    let tv = columns(dk, dv).filter(|_| matches!(path, Path::Kernels | Path::ScanKernels));
    let scan = custom_vjp(
        move |p: &[Array]| match tv {
            Some(tv) => forward_kernel(&p[0], &p[1], &p[2], &p[3], &p[4], tv),
            None => forward_ops(&p[0], &p[1], &p[2], &p[3], &p[4]),
        },
        move |p: &[Array], cotangents: &[Array], outputs: &[Array]| {
            let cotangent = |i: usize| -> Result<Array> {
                match cotangents.get(i) {
                    Some(c) => Ok(c.clone()),
                    None => zeros_like(&outputs[i]),
                }
            };
            let (w, ks, decay) = (&p[1], &p[2], &p[3]);
            let d = [cotangent(0)?, cotangent(1)?, cotangent(2)?];
            let (du, ds_next, ds0) = match tv {
                Some(tv) => backward_kernel(w, ks, decay, &d, tv)?,
                None => backward_ops(w, ks, decay, &d)?,
            };
            let (states, u) = (&outputs[0], &outputs[1]);
            let d_w = du.matmul(swap_axes(states, -1, -2)?)?.negative()?;
            let d_ks = u.matmul(swap_axes(&ds_next, -1, -2)?)?;
            let d_decay = ds_next.multiply(states)?.sum_axes(&[-2, -1], false)?;
            Ok(vec![du, d_w, d_ks, d_decay, ds0])
        },
    )?;
    let mut outputs = scan.apply(&[
        u0.clone(),
        w.clone(),
        ks.clone(),
        decay.clone(),
        state.clone(),
    ])?;
    let final_state = outputs.pop().expect("three outputs");
    let u = outputs.pop().expect("three outputs");
    Ok((outputs.pop().expect("three outputs"), u, final_state))
}

/// Chunk `i` of a `[B, H, N, ...]` array.
fn at(x: &Array, i: i32) -> Result<Array> {
    x.try_index((.., .., i))
}

/// `d_i` as `[B, H, 1, 1]`, to scale a state.
fn decay_at(decay: &Array, i: i32) -> Result<Array> {
    decay.try_index((.., .., i..i + 1))?.expand_dims(-1)
}

fn forward_ops(
    u0: &Array,
    w: &Array,
    ks: &Array,
    decay: &Array,
    state: &Array,
) -> Result<Vec<Array>> {
    let n = u0.shape()[2];
    let mut s = state.clone();
    let (mut states, mut us) = (
        Vec::with_capacity(n as usize),
        Vec::with_capacity(n as usize),
    );
    for i in 0..n {
        states.push(s.clone());
        let u = addmm(at(u0, i)?, at(w, i)?, &s, -1.0, 1.0)?;
        s = addmm(
            s.multiply(decay_at(decay, i)?)?,
            swap_axes(at(ks, i)?, -1, -2)?,
            &u,
            1.0,
            1.0,
        )?;
        us.push(u);
    }
    Ok(vec![stack(&states, 2)?, stack(&us, 2)?, s])
}

/// The backward loop: every chunk's `dU` and `dS_{i+1}`, and the initial state's gradient.
fn backward_ops(
    w: &Array,
    ks: &Array,
    decay: &Array,
    [d_states, d_u, d_final]: &[Array; 3],
) -> Result<(Array, Array, Array)> {
    let n = w.shape()[2];
    let mut ds = d_final.clone();
    let (mut dus, mut ds_next) = (
        Vec::with_capacity(n as usize),
        Vec::with_capacity(n as usize),
    );
    for i in (0..n).rev() {
        ds_next.push(ds.clone());
        let du = addmm(at(d_u, i)?, at(ks, i)?, &ds, 1.0, 1.0)?;
        ds = addmm(
            ds.multiply(decay_at(decay, i)?)?.add(at(d_states, i)?)?,
            swap_axes(at(w, i)?, -1, -2)?,
            &du,
            -1.0,
            1.0,
        )?;
        dus.push(du);
    }
    dus.reverse();
    ds_next.reverse();
    Ok((stack(&dus, 2)?, stack(&ds_next, 2)?, ds))
}

const HEADER: &str = "#include <metal_simdgroup_matrix>\n";

/// One threadgroup per batch row, head and slice of `TV` state columns, `SG` simdgroups each.
/// `S` holds the slice of the state, `Ub` the chunk's slice of `U`; offsets into the
/// `[B, H, N, rows, cols]` inputs and outputs are `((bh N + i) rows + r) cols + col`.
const FORWARD: &str = "
    const int N = u0_shape[2];
    const size_t bh = threadgroup_position_in_grid.y;
    const int col0 = threadgroup_position_in_grid.x * TV;
    const int tid = thread_index_in_threadgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int NT = SG * 32;
    threadgroup float S[DK * TV];
    threadgroup float Ub[C * TV];
    for (int e = tid; e < DK * TV; e += NT) {
        S[e] = s0[(bh * DK + e / TV) * DV + col0 + e % TV];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N; ++i) {
        const size_t rows_c = (bh * N + i) * C;
        const size_t rows_k = (bh * N + i) * DK;
        for (int e = tid; e < DK * TV; e += NT) {
            states[(rows_k + e / TV) * DV + col0 + e % TV] = S[e];
        }
        // w_i S, into Ub.
        for (int t = sg; t < (C / 8) * (TV / 8); t += SG) {
            const int tr = (t / (TV / 8)) * 8;
            const int tc = (t % (TV / 8)) * 8;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kk = 0; kk < DK; kk += 8) {
                simdgroup_float8x8 a;
                simdgroup_float8x8 b;
                simdgroup_load(a, w + (rows_c + tr) * DK + kk, DK);
                simdgroup_load(b, S + kk * TV + tc, TV);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, Ub + tr * TV + tc, TV);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float d = decay[bh * N + i];
        for (int e = tid; e < C * TV; e += NT) {
            const size_t at = (rows_c + e / TV) * DV + col0 + e % TV;
            const float value = u0[at] - Ub[e];
            Ub[e] = value;
            u[at] = value;
        }
        for (int e = tid; e < DK * TV; e += NT) {
            S[e] *= d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = d S + ks_i^T U, each simdgroup on its own tiles of S.
        for (int t = sg; t < (DK / 8) * (TV / 8); t += SG) {
            const int tr = (t / (TV / 8)) * 8;
            const int tc = (t % (TV / 8)) * 8;
            simdgroup_float8x8 acc;
            simdgroup_load(acc, S + tr * TV + tc, TV);
            for (int kk = 0; kk < C; kk += 8) {
                simdgroup_float8x8 a;
                simdgroup_float8x8 b;
                simdgroup_load(a, ks + (rows_c + kk) * DK + tr, DK, ulong2(0, 0), true);
                simdgroup_load(b, Ub + kk * TV + tc, TV);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, S + tr * TV + tc, TV);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int e = tid; e < DK * TV; e += NT) {
        s_final[(bh * DK + e / TV) * DV + col0 + e % TV] = S[e];
    }
";

/// The backward loop, laid out as the forward: `S` holds the slice of `dS`, `Ub` the slice of
/// `-dU` that the next step's product takes.
const BACKWARD: &str = "
    const int N = w_shape[2];
    const size_t bh = threadgroup_position_in_grid.y;
    const int col0 = threadgroup_position_in_grid.x * TV;
    const int tid = thread_index_in_threadgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int NT = SG * 32;
    threadgroup float S[DK * TV];
    threadgroup float Ub[C * TV];
    for (int e = tid; e < DK * TV; e += NT) {
        S[e] = d_final[(bh * DK + e / TV) * DV + col0 + e % TV];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = N - 1; i >= 0; --i) {
        const size_t rows_c = (bh * N + i) * C;
        const size_t rows_k = (bh * N + i) * DK;
        for (int e = tid; e < DK * TV; e += NT) {
            ds_next[(rows_k + e / TV) * DV + col0 + e % TV] = S[e];
        }
        // ks_i dS, into Ub.
        for (int t = sg; t < (C / 8) * (TV / 8); t += SG) {
            const int tr = (t / (TV / 8)) * 8;
            const int tc = (t % (TV / 8)) * 8;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kk = 0; kk < DK; kk += 8) {
                simdgroup_float8x8 a;
                simdgroup_float8x8 b;
                simdgroup_load(a, ks + (rows_c + tr) * DK + kk, DK);
                simdgroup_load(b, S + kk * TV + tc, TV);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, Ub + tr * TV + tc, TV);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float d = decay[bh * N + i];
        for (int e = tid; e < C * TV; e += NT) {
            const size_t at = (rows_c + e / TV) * DV + col0 + e % TV;
            const float value = d_u[at] + Ub[e];
            du[at] = value;
            Ub[e] = -value;
        }
        for (int e = tid; e < DK * TV; e += NT) {
            S[e] = d * S[e] + d_states[(rows_k + e / TV) * DV + col0 + e % TV];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dS += w_i^T (-dU), each simdgroup on its own tiles of dS.
        for (int t = sg; t < (DK / 8) * (TV / 8); t += SG) {
            const int tr = (t / (TV / 8)) * 8;
            const int tc = (t % (TV / 8)) * 8;
            simdgroup_float8x8 acc;
            simdgroup_load(acc, S + tr * TV + tc, TV);
            for (int kk = 0; kk < C; kk += 8) {
                simdgroup_float8x8 a;
                simdgroup_float8x8 b;
                simdgroup_load(a, w + (rows_c + kk) * DK + tr, DK, ulong2(0, 0), true);
                simdgroup_load(b, Ub + kk * TV + tc, TV);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, S + tr * TV + tc, TV);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int e = tid; e < DK * TV; e += NT) {
        ds0[(bh * DK + e / TV) * DV + col0 + e % TV] = S[e];
    }
";

fn config(b: i32, h: i32, dk: i32, dv: i32, tv: i32, outputs: Vec<Vec<i32>>) -> Launch {
    let threads = SIMDGROUPS * 32;
    Launch {
        grid: ((dv / tv) * threads, b * h, 1),
        thread_group: (threads, 1, 1),
        outputs: outputs
            .into_iter()
            .map(|shape| (shape, Dtype::Float32))
            .collect(),
        template: vec![
            ("DK".into(), TemplateArg::Int(dk)),
            ("DV".into(), TemplateArg::Int(dv)),
            ("TV".into(), TemplateArg::Int(tv)),
            ("C".into(), TemplateArg::Int(CHUNK_SIZE)),
            ("SG".into(), TemplateArg::Int(SIMDGROUPS)),
        ],
        init_value: None,
    }
}

fn forward_kernel(
    u0: &Array,
    w: &Array,
    ks: &Array,
    decay: &Array,
    state: &Array,
    tv: i32,
) -> Result<Vec<Array>> {
    let [b, h, n, c, dv] = [
        u0.shape()[0],
        u0.shape()[1],
        u0.shape()[2],
        u0.shape()[3],
        u0.shape()[4],
    ];
    let dk = w.shape()[4];
    counters::SCAN_FORWARD_KERNEL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let kernel = crate::kernel::kernel(
        "dash_mlx_delta_scan_forward",
        &["u0", "w", "ks", "decay", "s0"],
        &["states", "u", "s_final"],
        FORWARD,
        HEADER,
    )?;
    let outputs = vec![
        vec![b, h, n, dk, dv],
        vec![b, h, n, c, dv],
        vec![b, h, dk, dv],
    ];
    let inputs = [
        u0.clone(),
        w.clone(),
        ks.clone(),
        decay.clone(),
        state.clone(),
    ];
    // SAFETY: `tv` divides Dv and Dk is a multiple of 8 (`columns`); threadgroup (x, bh) takes
    // columns x TV .. (x + 1) TV of batch row and head bh, reads only those columns and rows
    // within each input's shape, and writes every state, `U` and final-state element of them,
    // which no other threadgroup touches.
    unsafe { launch(&kernel, &inputs, &config(b, h, dk, dv, tv, outputs)) }
}

fn backward_kernel(
    w: &Array,
    ks: &Array,
    decay: &Array,
    [d_states, d_u, d_final]: &[Array; 3],
    tv: i32,
) -> Result<(Array, Array, Array)> {
    let [b, h, n, c, dk] = [
        w.shape()[0],
        w.shape()[1],
        w.shape()[2],
        w.shape()[3],
        w.shape()[4],
    ];
    let dv = d_u.shape()[4];
    counters::SCAN_BACKWARD_KERNEL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let kernel = crate::kernel::kernel(
        "dash_mlx_delta_scan_backward",
        &["w", "ks", "decay", "d_states", "d_u", "d_final"],
        &["du", "ds_next", "ds0"],
        BACKWARD,
        HEADER,
    )?;
    let f32 = |x: &Array| x.as_dtype(Dtype::Float32);
    let outputs = vec![
        vec![b, h, n, c, dv],
        vec![b, h, n, dk, dv],
        vec![b, h, dk, dv],
    ];
    let inputs = [
        w.clone(),
        ks.clone(),
        decay.clone(),
        f32(d_states)?,
        f32(d_u)?,
        f32(d_final)?,
    ];
    // SAFETY: as the forward kernel's: threadgroup (x, bh) reads and writes only its own slice of
    // TV columns of batch row and head bh, every element of it in each of the three outputs.
    let mut grads = unsafe { launch(&kernel, &inputs, &config(b, h, dk, dv, tv, outputs))? };
    let ds0 = grads.pop().expect("three outputs");
    let ds_next = grads.pop().expect("three outputs");
    Ok((grads.pop().expect("three outputs"), ds_next, ds0))
}
