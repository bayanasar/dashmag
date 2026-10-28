//! The recurrent gated delta rule, a token at a time, for decoding: dash-ops'
//! `gated_delta::recurrent` on MLX.
//!
//! The rule, its arguments and its results are dash-ops' (its `gated_delta` module and that
//! module's `recurrent` document them): each sequence's `[Dk, Dv]` state decayed by `exp(g)`,
//! corrected towards `v` along `k` by `beta`, and read along the scaled `q`, queries and keys
//! first normalised to unit length when asked.
//!
//! Shapes as in dash-ops: `q`, `k` `[B, H, Dk]`, `v` `[B, H, Dv]`, `g` and `beta` `[B, H]`, the
//! state `[B, H, Dk, Dv]` in f32, row `b` one token of sequence `b`. With `cu_seqlens` `[S + 1]`
//! the rows are instead `T` packed tokens of `S` sequences, each sequence's in turn (possibly
//! none), and the state `[S, H, Dk, Dv]`: the step advances every sequence by its tokens.
//! Queries, keys and values share one float dtype, which the output takes; `g` and `beta` may be
//! in that dtype or f32. Everything is computed in f32.
//!
//! One Metal kernel runs a call. A threadgroup of 4 simdgroups takes 32 of a sequence's head's
//! state columns, a lane each, and simdgroup `w` rows `w Dk / 4 ..` of its lane's column in
//! registers; it reads the state once, carries it through the sequence's tokens, and writes it
//! once, the reads and writes running along the state's rows. Each token's query and key go
//! through threadgroup memory once; each simdgroup takes their norms by `simd_sum`, and the four
//! simdgroups' shares of `S^T k` and `S^T q` meet in threadgroup memory. Indices are 32-bit where
//! every index fits, 64-bit otherwise, as dash-ops chooses its address type; `g` and `beta` are
//! read in their own dtype. Chosen by measurement over dash-ops' arrangement (32 by 8 threads, the
//! norms and sums through threadgroup memory), which takes 1% less time for one token a sequence
//! and 16 to 21% more for four, and over arrangements without threadgroup memory between tokens,
//! all slower on this `[Dk, Dv]` state (`examples/gated_delta_recurrent_bench.rs`). mlx-lm's
//! decoding kernel, which keeps its state as `[Dv, Dk]`, takes 1 to 2% less time for one token and
//! 19 to 21% less for four: in that layout a lane's rows lie together and its sums over them are `simd_sum`s with no
//! barrier, where here the sums over a column meet in threadgroup memory, four barriers a token.
//! A gradient asked of the step is the reference's (MLX's operations, [`reference`](fn@reference)),
//! and so is a step with heads wider than 256.

use super::chunk::boundaries;
use crate::kernel::{Launch, launch};
use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::TemplateArg;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{arange, concatenate};
use mlx_rs::transforms::{custom_vjp, fallible_vjp};
use mlx_rs::{Array, Dtype};

/// The epsilon of transformers' (and fla's) l2 normalisation of `q` and `k`.
const L2NORM_EPS: f32 = 1e-6;

/// Simdgroups per threadgroup; the rows of the state are split among them.
const SIMDGROUPS: i32 = 4;
/// The widest key head the kernel takes: each thread keeps `ceil(Dk / 4)` state values in
/// registers. Wider heads run the reference.
const MAX_DK: i32 = 256;

pub use dash_contract::gated_delta::recurrent::GatedDeltaStepOptions;

/// Whether the step's Metal kernel, rather than the reference, takes keys `dk` wide. For tests, as
/// the [`counters`] are.
#[doc(hidden)]
pub fn kernel_takes(dk: i32) -> bool {
    dk <= MAX_DK
}

/// How many times the step's kernel was built into a graph.
#[doc(hidden)]
pub mod counters {
    use std::sync::atomic::AtomicUsize;

    /// The step's kernel.
    pub static STEP_KERNEL: AtomicUsize = AtomicUsize::new(0);
}

fn contract(message: String) -> Exception {
    Exception::custom(format!("gated_delta_step: {message}"))
}

/// A step of the recurrent gated delta rule, with dash-ops' arguments and results ([module](self)):
/// the output `[B, H, Dv]` in `q`'s dtype and the new states, shaped as `state`, in f32.
/// Differentiable, through the reference.
#[allow(
    clippy::too_many_arguments,
    reason = "the five inputs, the state, the boundaries that may be absent, the options"
)]
pub fn gated_delta_step(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    cu_seqlens: Option<&Array>,
    options: GatedDeltaStepOptions,
) -> Result<(Array, Array)> {
    let offsets = check(q, k, v, g, beta, state, cu_seqlens)?;
    if q.shape()[2] > MAX_DK {
        return reference_steps(q, k, v, g, beta, state, &offsets, options);
    }
    let scale = scale(options, q.shape()[2]);
    let qk_l2norm = options.qk_l2norm;
    let cu = match &offsets {
        Some(offsets) => Array::from_slice(offsets, &[offsets.len() as i32]),
        None => arange::<_, i32>(0, q.shape()[0] + 1, None)?,
    };
    let step = custom_vjp(
        move |p: &[Array]| {
            let scale = Array::from_slice(&[scale], &[1]);
            step_kernel(
                &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &cu, &scale, qk_l2norm,
            )
        },
        move |p: &[Array], cotangents: &[Array], _: &[Array]| {
            let f = |a: &[Array]| -> Result<Vec<Array>> {
                let (out, next) =
                    reference_steps(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &offsets, options)?;
                Ok(vec![out, next])
            };
            Ok(fallible_vjp(f, p, cotangents)?.1)
        },
    )?;
    let mut outputs = step.apply(&[
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        state.clone(),
    ])?;
    let next = outputs.pop().expect("two outputs");
    Ok((outputs.pop().expect("two outputs"), next))
}

fn scale(options: GatedDeltaStepOptions, dk: i32) -> f32 {
    options
        .scale
        .unwrap_or_else(|| (1.0 / (dk as f64).sqrt()) as f32)
}

/// Checks the inputs against each other, and returns the packed sequences' boundaries if any.
fn check(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    cu_seqlens: Option<&Array>,
) -> Result<Option<Vec<i32>>> {
    let [b, h, dk] = match q.shape() {
        [b, h, dk] => [*b, *h, *dk],
        shape => return Err(contract(format!("q must be [B, H, Dk], got {shape:?}"))),
    };
    if k.shape() != q.shape() {
        return Err(contract(format!(
            "k must be q's shape, got {:?}",
            k.shape()
        )));
    }
    let dv = match v.shape() {
        [vb, vh, dv] if [*vb, *vh] == [b, h] => *dv,
        shape => return Err(contract(format!("v must be [B, H, Dv], got {shape:?}"))),
    };
    for (name, x) in [("g", g), ("beta", beta)] {
        if x.shape() != [b, h] {
            return Err(contract(format!(
                "{name} must be [B, H], got {:?}",
                x.shape()
            )));
        }
    }
    let offsets = cu_seqlens
        .map(|cu| boundaries(cu, b, "gated_delta_step"))
        .transpose()?;
    let sequences = offsets.as_ref().map_or(b, |o| o.len() as i32 - 1);
    if state.shape() != [sequences, h, dk, dv] {
        return Err(contract(format!(
            "the state must be [S, H, Dk, Dv] = {:?}, one per sequence, got {:?}",
            [sequences, h, dk, dv],
            state.shape()
        )));
    }
    if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
        return Err(contract("q, k and v must share one dtype".into()));
    }
    if state.dtype() != Dtype::Float32 {
        return Err(contract(format!(
            "the state must be f32, got {:?}",
            state.dtype()
        )));
    }
    Ok(offsets)
}

/// The step from MLX's operations, in f32: what the tests hold the kernel to, and what a gradient
/// of the step is taken through. Packed sequences go token by token.
#[allow(clippy::too_many_arguments)]
pub fn reference(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    cu_seqlens: Option<&Array>,
    options: GatedDeltaStepOptions,
) -> Result<(Array, Array)> {
    let offsets = check(q, k, v, g, beta, state, cu_seqlens)?;
    reference_steps(q, k, v, g, beta, state, &offsets, options)
}

#[allow(clippy::too_many_arguments)]
fn reference_steps(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    offsets: &Option<Vec<i32>>,
    options: GatedDeltaStepOptions,
) -> Result<(Array, Array)> {
    let Some(offsets) = offsets else {
        return one_token(q, k, v, g, beta, state, options);
    };
    let row = |x: &Array, t: i32| x.try_index(t..t + 1);
    let (mut outs, mut states) = (Vec::new(), Vec::new());
    for s in 0..offsets.len() - 1 {
        let mut st = state.try_index(s as i32..s as i32 + 1)?;
        for t in offsets[s]..offsets[s + 1] {
            let (out, next) = one_token(
                &row(q, t)?,
                &row(k, t)?,
                &row(v, t)?,
                &row(g, t)?,
                &row(beta, t)?,
                &st,
                options,
            )?;
            outs.push(out);
            st = next;
        }
        states.push(st);
    }
    let out = match outs.is_empty() {
        true => q.try_index((0..0, .., ..))?.as_dtype(q.dtype())?,
        false => concatenate(&outs, 0)?,
    };
    Ok((out, concatenate(&states, 0)?))
}

/// One token of each of `B` sequences, from MLX's operations in f32.
fn one_token(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    options: GatedDeltaStepOptions,
) -> Result<(Array, Array)> {
    let (dtype, f32) = (q.dtype(), Dtype::Float32);
    let dk = q.shape()[2];
    let (mut q, mut k) = (q.as_dtype(f32)?, k.as_dtype(f32)?);
    if options.qk_l2norm {
        let norm = |x: &Array| -> Result<Array> {
            x.multiply(
                x.square()?
                    .sum_axis(-1, true)?
                    .add(Array::from_f32(L2NORM_EPS))?
                    .rsqrt()?,
            )
        };
        q = norm(&q)?;
        k = norm(&k)?;
    }
    let q = q.multiply(Array::from_f32(scale(options, dk)))?;
    let decay = g.as_dtype(f32)?.exp()?.expand_dims(-1)?.expand_dims(-1)?;
    let s = state.multiply(&decay)?;
    let k_col = k.expand_dims(-1)?;
    let kv = s.multiply(&k_col)?.sum_axis(-2, false)?;
    let delta = v
        .as_dtype(f32)?
        .subtract(&kv)?
        .multiply(beta.as_dtype(f32)?.expand_dims(-1)?)?;
    let s = s.add(k_col.multiply(delta.expand_dims(-2)?)?)?;
    let out = s.multiply(q.expand_dims(-1)?)?.sum_axis(-2, false)?;
    Ok((out.as_dtype(dtype)?, s))
}

/// transformers' epsilon, as `L2NORM_EPS`.
const HEADER: &str = "constant float DASH_DELTA_L2NORM_EPS = 1e-6f;\n";

/// Grid: x the state's columns in blocks of 32, a lane each; y the sequences and heads.
/// Simdgroup `w` holds rows `w R .. (w + 1) R` of its lane's column, `R = ceil(DK / SG)`, through
/// the sequence's tokens `cu[s] .. cu[s + 1]` (rows of the packed inputs).
const STEP: &str = "
    const int H = q_shape[1];
    constexpr int R = (DK + SG - 1) / SG;
    // Every row in range when the simdgroups split Dk evenly: the checks then compile away.
    constexpr bool EVEN = DK % SG == 0;
    const int lane = thread_index_in_simdgroup;
    const int w = simdgroup_index_in_threadgroup;
    const int tid = thread_index_in_threadgroup;
    const int col = threadgroup_position_in_grid.x * 32 + lane;
    const IDX sh = threadgroup_position_in_grid.y;
    const IDX b = sh / H;
    const int h = sh % H;
    const int row0 = w * R;
    const bool col_in = col < DV;
    const IDX state_base = sh * DK * DV;
    threadgroup float q_shared[DK];
    threadgroup float k_shared[DK];
    threadgroup float parts[SG][32];
    float s[R];
    for (int i = 0; i < R; ++i) {
        const int r = row0 + i;
        s[i] = (col_in && (EVEN || r < DK)) ? state[state_base + IDX(r) * DV + col] : 0.0f;
    }
    for (int t = cu[b]; t < cu[b + 1]; ++t) {
        const IDX th = IDX(t) * H + h;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < DK; i += SG * 32) {
            q_shared[i] = float(q[th * DK + i]);
            k_shared[i] = float(k[th * DK + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_scale = scale[0];
        float k_scale = 1.0f;
        if (L2NORM) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (int i = lane; i < DK; i += 32) {
                q_sum += q_shared[i] * q_shared[i];
                k_sum += k_shared[i] * k_shared[i];
            }
            q_scale *= metal::precise::rsqrt(simd_sum(q_sum) + DASH_DELTA_L2NORM_EPS);
            k_scale = metal::precise::rsqrt(simd_sum(k_sum) + DASH_DELTA_L2NORM_EPS);
        }
        const float decay = metal::precise::exp(float(g[th]));
        float kv = 0.0f;
        for (int i = 0; i < R; ++i) {
            const int r = row0 + i;
            s[i] *= decay;
            kv += (EVEN || r < DK) ? s[i] * k_shared[r] : 0.0f;
        }
        parts[w][lane] = kv * k_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kv_mem = 0.0f;
        for (int j = 0; j < SG; ++j) {
            kv_mem += parts[j][lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float value = col_in ? float(v[th * DV + col]) : 0.0f;
        const float kd = k_scale * float(beta[th]) * (value - kv_mem);
        float o = 0.0f;
        for (int i = 0; i < R; ++i) {
            const int r = row0 + i;
            if (EVEN || r < DK) {
                s[i] += k_shared[r] * kd;
                o += s[i] * q_shared[r];
            }
        }
        parts[w][lane] = o * q_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (w == 0 && col_in) {
            float total = 0.0f;
            for (int j = 0; j < SG; ++j) {
                total += parts[j][lane];
            }
            out[th * DV + col] = static_cast<DT>(total);
        }
    }
    for (int i = 0; i < R; ++i) {
        const int r = row0 + i;
        if (col_in && (EVEN || r < DK)) {
            new_state[state_base + IDX(r) * DV + col] = s[i];
        }
    }
";

#[allow(clippy::too_many_arguments)]
fn step_kernel(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    cu: &Array,
    scale: &Array,
    qk_l2norm: bool,
) -> Result<Vec<Array>> {
    let [t, h, dk] = [q.shape()[0], q.shape()[1], q.shape()[2]];
    let dv = v.shape()[2];
    let sequences = state.shape()[0];
    counters::STEP_KERNEL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let kernel = crate::kernel::kernel(
        "dash_mlx_delta_step",
        &["q", "k", "v", "g", "beta", "state", "cu", "scale"],
        &["out", "new_state"],
        STEP,
        HEADER,
    )?;
    // dash-ops' choice of address type: 32-bit indices where every index fits, 64-bit otherwise.
    let largest = [
        sequences as u64 * h as u64 * dk as u64 * dv as u64,
        t as u64 * h as u64 * dk.max(dv) as u64,
    ];
    let index_type = match largest.iter().all(|&n| n <= u32::MAX as u64) {
        true => Dtype::Uint32,
        false => Dtype::Uint64,
    };
    let config = Launch {
        grid: (((dv + 31) / 32) * SIMDGROUPS * 32, sequences * h, 1),
        thread_group: (SIMDGROUPS * 32, 1, 1),
        outputs: vec![
            (vec![t, h, dv], q.dtype()),
            (vec![sequences, h, dk, dv], Dtype::Float32),
        ],
        template: vec![
            ("DT".into(), TemplateArg::Dtype(q.dtype())),
            ("DK".into(), TemplateArg::Int(dk)),
            ("DV".into(), TemplateArg::Int(dv)),
            ("SG".into(), TemplateArg::Int(SIMDGROUPS)),
            ("L2NORM".into(), TemplateArg::Bool(qk_l2norm)),
            ("IDX".into(), TemplateArg::Dtype(index_type)),
        ],
        init_value: None,
    };
    let inputs = [
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        state.clone(),
        cu.as_dtype(Dtype::Int32)?,
        scale.clone(),
    ];
    // SAFETY: Dk is at most 256 (checked above), so the threadgroup's query, key and partial sums
    // fit its memory. Threadgroup (x, sh) takes columns 32 x .. of sequence and head sh, reading
    // and writing only columns below Dv and rows below Dk; its tokens are rows cu[s] .. cu[s + 1]
    // of the inputs, and the boundaries run from 0 to T without decreasing (checked). Lane 0's
    // simdgroup writes the output of each of those tokens at its column, and every simdgroup its
    // rows of the new state: each output element once.
    unsafe { launch(&kernel, &inputs, &config) }
}
