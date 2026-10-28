//! The chunked gated delta rule, forward and backward, for training: dash-ops'
//! `gated_delta::chunk` on MLX. The rule and the chunking are dash-ops' (its `gated_delta` module
//! and that module's `chunk` document them); this module computes them on MLX.
//!
//! In chunks of 64 tokens, with dash-ops' contract ([`super`] gives the layouts): queries and
//! keys `[B, T, H, Dk]`, values `[B, T, H, Dv]`, one float dtype for the three; `g`, the log of
//! the decay, and `beta` `[B, T, H]`, each in the queries' dtype or f32; initial and final states
//! `[B, H, Dk, Dv]` in f32, zeros where none is given; the output `[B, T, H, Dv]` in the queries'
//! dtype. Packed variable-length sequences lie along `T` of a batch of one, their boundaries
//! `cu_seqlens` `[S + 1]`, each with its own initial and final state, `[S, H, Dk, Dv]`; they run
//! one at a time, their boundaries read to the host, each exactly as a call on it alone. Options
//! ([`ChunkGatedDeltaOptions`]) normalise queries and keys to unit length first (transformers'
//! epsilon, `1e-6` under the square root), set the query scale (`1 / sqrt(Dk)` by default) and
//! ask for the final states. Everything inside is computed in f32. Within a sequence the padding
//! to a whole chunk writes nothing and decays nothing, so the final state is the one after the
//! sequence's last token.
//!
//! Per chunk, with `G` the cumulative sum of `g` inside the chunk (dash-ops' notation):
//!
//! - `A[i, j] = beta_i (k_i . k_j) exp(G_i - G_j)` for `i > j`, and `M = (I + A)^-1` (dash-ops'
//!   `T`);
//! - `U0 = M (beta V)` and `W = M (beta exp(G) K)`, the chunk's values and keys made independent;
//! - with `S` the state entering the chunk, `U = U0 - W S`, and the state leaving it
//!   `exp(G_last) S + (exp(G_last - G) K)^T U`;
//! - `O = (exp(G) Q) S + P U`, `P[i, j] = (q_i . k_j) exp(G_i - G_j)` for `i >= j`.
//!
//! Everything but the passing of the state from chunk to chunk is batched over all chunks and
//! differentiated by MLX. The inverse has its backward in closed form,
//! `dA = tril(-M^T dM M^T, -1)`, two matmuls instead of differentiating the doubling that forms
//! it. The scan over the chunks is a custom VJP: forward, each chunk's entering state and `U`
//! are kept; backward, the state's gradient is passed back from chunk to chunk, and the
//! gradients that need every chunk's states are formed batched afterwards.

use crate::kernel::{Launch, launch};
use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::TemplateArg;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{
    concatenate, cumsum, move_axis, ones, pad, select, swap_axes, tril, triu, zeros, zeros_like,
};
use mlx_rs::transforms::custom_vjp;
use mlx_rs::{Array, Dtype};

/// Tokens per chunk, as in transformers' `torch_chunk_gated_delta_rule`, fla and dash-ops.
pub const CHUNK_SIZE: i32 = 64;

/// How a call runs the inverses and the scan from chunk to chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Path {
    /// Metal kernels, with the backward passes written out.
    Kernels,
    /// The inverses' kernel; the scan as MLX's operations, its backward pass written out.
    InverseKernel,
    /// The scan's kernels; the inverses as MLX's operations, their backward pass written out.
    ScanKernels,
    /// MLX's operations, with the same backward passes written out: the reference.
    Operations,
    /// MLX's operations, differentiated by MLX throughout: a baseline.
    Autograd,
}

pub use dash_contract::gated_delta::chunk::ChunkGatedDeltaOptions;

/// Whether the scan's Metal kernels, rather than MLX's operations, take heads `dk` by `dv` (the
/// inverses' kernel takes every call). For tests, as the [`counters`] are.
#[doc(hidden)]
pub fn scan_kernels_take(dk: i32, dv: i32) -> bool {
    super::scan::columns(dk, dv).is_some()
}

/// How many times each kernel was built into a graph, so that tests can tell the kernels from
/// the operations, which compute the same values.
#[doc(hidden)]
pub mod counters {
    use std::sync::atomic::AtomicUsize;

    /// The forward scan.
    pub static SCAN_FORWARD_KERNEL: AtomicUsize = AtomicUsize::new(0);
    /// The backward scan.
    pub static SCAN_BACKWARD_KERNEL: AtomicUsize = AtomicUsize::new(0);
    /// The inverse by forward substitution.
    pub static INVERSE_KERNEL: AtomicUsize = AtomicUsize::new(0);
}

/// The epsilon of transformers' (and fla's) `l2norm`: `x / sqrt(sum(x^2) + eps)`.
const L2NORM_EPS: f32 = 1e-6;

fn contract(message: String) -> Exception {
    Exception::custom(format!("chunk_gated_delta_rule: {message}"))
}

/// The chunked gated delta rule over `B` sequences, or over `S` packed ones, with dash-ops'
/// arguments and results ([module](self)): the output `[B, T, H, Dv]` in the queries' dtype and,
/// if asked, the final states, shaped as the initial states, in f32. Differentiable with respect
/// to the queries, keys, values, `g`, `beta` and the initial states, through the output and the
/// final states.
#[allow(
    clippy::too_many_arguments,
    reason = "the five inputs, the two that may be absent, the options: dash-ops' arguments"
)]
pub fn chunk_gated_delta_rule(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    initial_state: Option<&Array>,
    cu_seqlens: Option<&Array>,
    options: ChunkGatedDeltaOptions,
) -> Result<(Array, Option<Array>)> {
    checked(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        options,
        Path::Kernels,
    )
}

/// [`chunk_gated_delta_rule`] with the scan from chunk to chunk and the inverses run as MLX
/// operations, without the kernels: the reference the tests hold the kernels to.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule_reference(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    initial_state: Option<&Array>,
    cu_seqlens: Option<&Array>,
    options: ChunkGatedDeltaOptions,
) -> Result<(Array, Option<Array>)> {
    checked(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        options,
        Path::Operations,
    )
}

/// [`chunk_gated_delta_rule`] as MLX's operations differentiated by MLX throughout, the inverses
/// and the scan included, without the written-out backward passes: a baseline for benchmarks.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule_autograd(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    initial_state: Option<&Array>,
    cu_seqlens: Option<&Array>,
    options: ChunkGatedDeltaOptions,
) -> Result<(Array, Option<Array>)> {
    checked(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        options,
        Path::Autograd,
    )
}

/// [`chunk_gated_delta_rule`] with each of its two Metal kernels on or off, the part without its
/// kernel run as MLX's operations with the same written-out backward pass: for benchmarks that
/// time what each kernel is worth.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule_parts(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    initial_state: Option<&Array>,
    cu_seqlens: Option<&Array>,
    options: ChunkGatedDeltaOptions,
    [inverse_kernel, scan_kernels]: [bool; 2],
) -> Result<(Array, Option<Array>)> {
    let path = match (inverse_kernel, scan_kernels) {
        (true, true) => Path::Kernels,
        (true, false) => Path::InverseKernel,
        (false, true) => Path::ScanKernels,
        (false, false) => Path::Operations,
    };
    checked(q, k, v, g, beta, initial_state, cu_seqlens, options, path)
}

/// The boundaries of packed sequences, read to the host and checked against `T`.
pub(crate) fn boundaries(cu_seqlens: &Array, t: i32, op: &str) -> Result<Vec<i32>> {
    let offsets = cu_seqlens.as_dtype(Dtype::Int64)?.contiguous()?;
    let offsets: Vec<i32> = offsets
        .as_slice::<i64>()
        .iter()
        .map(|&o| o as i32)
        .collect();
    let ordered = offsets.len() >= 2
        && offsets[0] == 0
        && offsets[offsets.len() - 1] == t
        && offsets.windows(2).all(|w| w[0] <= w[1]);
    if !ordered {
        return Err(Exception::custom(format!(
            "{op}: cu_seqlens must go from 0 to T = {t} without decreasing, got {offsets:?}"
        )));
    }
    Ok(offsets)
}

#[allow(clippy::too_many_arguments)]
fn checked(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    initial_state: Option<&Array>,
    cu_seqlens: Option<&Array>,
    options: ChunkGatedDeltaOptions,
    path: Path,
) -> Result<(Array, Option<Array>)> {
    let [b, t, h, dk] = match q.shape() {
        [b, t, h, dk] => [*b, *t, *h, *dk],
        shape => return Err(contract(format!("q must be [B, T, H, Dk], got {shape:?}"))),
    };
    let dv = match v.shape() {
        [vb, vt, vh, dv] if [*vb, *vt, *vh] == [b, t, h] => *dv,
        shape => {
            return Err(contract(format!(
                "values must be [B, T, H, Dv] with the queries' B, T, H, got {shape:?}"
            )));
        }
    };
    if k.shape() != q.shape() {
        return Err(contract(format!(
            "keys must have the queries' shape {:?}, got {:?}",
            q.shape(),
            k.shape()
        )));
    }
    for (name, x) in [("g", g), ("beta", beta)] {
        if x.shape() != [b, t, h] {
            return Err(contract(format!(
                "{name} must be [B, T, H], got {:?}",
                x.shape()
            )));
        }
    }
    if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
        return Err(contract(format!(
            "queries, keys and values must share one dtype, got {:?}, {:?}, {:?}",
            q.dtype(),
            k.dtype(),
            v.dtype()
        )));
    }
    if t == 0 {
        return Err(contract("the sequence must not be empty".into()));
    }
    let offsets = match cu_seqlens {
        None => None,
        Some(_) if b != 1 => {
            return Err(contract(format!(
                "packed sequences lie along T of a batch of one, got B = {b}"
            )));
        }
        Some(cu_seqlens) => Some(boundaries(cu_seqlens, t, "chunk_gated_delta_rule")?),
    };
    let sequences = offsets.as_ref().map_or(b, |o| o.len() as i32 - 1);
    let states = match initial_state {
        Some(state) if state.shape() == [sequences, h, dk, dv] => state.as_dtype(Dtype::Float32)?,
        Some(state) => {
            return Err(contract(format!(
                "the initial state must be [S, H, Dk, Dv] = {:?}, one per sequence, got {:?}",
                [sequences, h, dk, dv],
                state.shape()
            )));
        }
        None => zeros::<f32>(&[sequences, h, dk, dv])?,
    };
    let scale = options
        .scale
        .unwrap_or_else(|| (1.0 / (dk as f64).sqrt()) as f32);
    let qk_l2norm = options.qk_l2norm;
    let Some(offsets) = offsets else {
        let (out, last) = rule(q, k, v, g, beta, &states, qk_l2norm, scale, path)?;
        return Ok((out, options.output_final_state.then_some(last)));
    };
    // One sequence at a time, each exactly as a call on it alone.
    let (mut outs, mut finals) = (Vec::new(), Vec::new());
    for s in 0..sequences as usize {
        let (from, to) = (offsets[s], offsets[s + 1]);
        let state = states.try_index(s as i32..s as i32 + 1)?;
        if from == to {
            finals.push(state);
            continue;
        }
        let piece = |x: &Array| x.try_index((.., from..to));
        let (out, last) = rule(
            &piece(q)?,
            &piece(k)?,
            &piece(v)?,
            &piece(g)?,
            &piece(beta)?,
            &state,
            qk_l2norm,
            scale,
            path,
        )?;
        outs.push(out);
        finals.push(last);
    }
    let out = concatenate(&outs, 1)?;
    let finals = options
        .output_final_state
        .then(|| concatenate(&finals, 0))
        .transpose()?;
    Ok((out, finals))
}

/// transformers' `l2norm` over the last axis, in f32, as MLX's fused `rms_norm`, which has its own
/// backward: `x / sqrt(sum(x^2) + eps)` is `rms_norm(x)` with epsilon `eps / D`, over `sqrt(D)`.
fn l2norm(x: &Array) -> Result<Array> {
    let d = *x.shape().last().expect("a last axis") as f32;
    mlx_rs::fast::rms_norm(x, None, L2NORM_EPS / d)?.multiply(Array::from_f32(1.0 / d.sqrt()))
}

/// The rule from f32 parts. Returns the output in the queries' dtype and the final state.
#[allow(clippy::too_many_arguments)]
fn rule(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    use_qk_l2norm: bool,
    scale: f32,
    path: Path,
) -> Result<(Array, Array)> {
    let dtype = q.dtype();
    let [b, t, h, dk] = [q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]];
    let dv = v.shape()[3];
    let c = CHUNK_SIZE;
    let n = (t + c - 1) / c;
    let (mut q, mut k) = (q.as_dtype(Dtype::Float32)?, k.as_dtype(Dtype::Float32)?);
    if use_qk_l2norm {
        q = l2norm(&q)?;
        k = l2norm(&k)?;
    }
    let q = q.multiply(Array::from_f32(scale))?;

    // [B, T, H, ...] to [B, H, N, C, ...], zero-padded along T: the padding has beta 0 (writes
    // nothing) and g 0 (decays nothing).
    let pad_t = n * c - t;
    let chunked = |x: &Array, depth: Option<i32>| -> Result<Array> {
        let x = x.as_dtype(Dtype::Float32)?;
        let mut widths = vec![(0, 0); x.ndim()];
        widths[1] = (0, pad_t);
        let x = if pad_t > 0 {
            pad(&x, &widths[..], None, None)?
        } else {
            x
        };
        let x = swap_axes(&x, 1, 2)?;
        match depth {
            Some(depth) => x.reshape(&[b, h, n, c, depth]),
            None => x.reshape(&[b, h, n, c]),
        }
    };
    let (q, k, v) = (
        chunked(&q, Some(dk))?,
        chunked(&k, Some(dk))?,
        chunked(v, Some(dv))?,
    );
    let (g, beta) = (chunked(g, None)?, chunked(beta, None)?);

    let cum = cumsum(&g, -1, false, true)?;
    let last = cum.try_index((.., .., .., (c - 1)..))?;
    // exp(G_i - G_j) where i >= j, 0 above the diagonal.
    let upper = triu(ones::<bool>(&[c, c])?, 1)?;
    let diff = cum.expand_dims(-1)?.subtract(cum.expand_dims(-2)?)?;
    let pairwise = select(&upper, Array::from_f32(f32::NEG_INFINITY), &diff)?.exp()?;
    let kt = swap_axes(&k, -1, -2)?;
    let beta_col = beta.expand_dims(-1)?;
    let gamma = cum.exp()?.expand_dims(-1)?;
    let k_beta = k.multiply(&beta_col)?;
    let m = inverse_unit_lower(&k_beta.matmul(&kt)?.multiply(&pairwise)?, path)?;
    let u0 = m.matmul(v.multiply(&beta_col)?)?;
    let w = m.matmul(k_beta.multiply(&gamma)?)?;
    let qs = q.multiply(&gamma)?;
    let p = q.matmul(&kt)?.multiply(&pairwise)?;
    let ks = k.multiply(last.subtract(&cum)?.exp()?.expand_dims(-1)?)?;
    let decay = last.exp()?.squeeze_axes(&[-1])?;

    let (states, u, final_state) = super::scan::scan(&u0, &w, &ks, &decay, state, path)?;
    let o = qs.matmul(&states)?.add(p.matmul(&u)?)?;
    let o = o
        .reshape(&[b, h, n * c, dv])?
        .try_index((.., .., ..t, ..))?;
    Ok((swap_axes(&o, 1, 2)?.as_dtype(dtype)?, final_state))
}

/// `(I + N)^-1`, `N` the strictly lower triangle of `a` `[..., C, C]`, by recursive doubling:
/// the inverse of each pair of diagonal blocks from the inverses of the blocks, `log2 C` batched
/// steps. Its backward is the closed form `dA = tril(-M^T dM M^T, -1)`.
fn inverse_unit_lower(a: &Array, path: Path) -> Result<Array> {
    if path == Path::Autograd {
        return doubling(a);
    }
    let inverse = custom_vjp(
        move |p: &[Array]| match path {
            Path::Kernels | Path::InverseKernel => Ok(vec![substitution_kernel(&p[0])?]),
            _ => Ok(vec![doubling(&p[0])?]),
        },
        |_: &[Array], cotangents: &[Array], outputs: &[Array]| {
            let mt = swap_axes(&outputs[0], -1, -2)?;
            Ok(vec![tril(
                mt.matmul(&cotangents[0])?.matmul(&mt)?.negative()?,
                -1,
            )?])
        },
    )?;
    Ok(inverse.apply(std::slice::from_ref(a))?.remove(0))
}

/// One threadgroup of `C` threads per matrix. The strictly lower triangle `N` goes to threadgroup
/// memory once; thread `j` then forms column `j` of `(I + N)^-1` by forward substitution, in its
/// own registers, `M[i, j] = [i == j] - sum_{j <= k < i} N[i, k] M[k, j]`: no thread waits on
/// another after the load.
const SUBSTITUTION: &str = "
    const size_t m = threadgroup_position_in_grid.x;
    const int j = thread_position_in_threadgroup.x;
    threadgroup float lower[C * C];
    for (int r = 0; r < C; ++r) {
        lower[r * C + j] = a[(m * C + r) * C + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float column[C];
    for (int i = 0; i < C; ++i) {
        float value = i == j ? 1.0f : 0.0f;
        for (int k = j; k < i; ++k) {
            value -= lower[i * C + k] * column[k];
        }
        column[i] = i < j ? 0.0f : value;
        out[(m * C + i) * C + j] = column[i];
    }
";

/// The forward of [`inverse_unit_lower`] as a Metal kernel, for `[..., C, C]` f32.
fn substitution_kernel(a: &Array) -> Result<Array> {
    let shape = a.shape().to_vec();
    let c = shape[shape.len() - 1];
    let matrices: i32 = shape[..shape.len() - 2].iter().product();
    counters::INVERSE_KERNEL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let kernel = crate::kernel::kernel(
        "dash_mlx_delta_inverse_unit_lower",
        &["a"],
        &["out"],
        SUBSTITUTION,
        "",
    )?;
    let config = Launch {
        grid: (matrices * c, 1, 1),
        thread_group: (c, 1, 1),
        outputs: vec![(shape, Dtype::Float32)],
        template: vec![("C".into(), TemplateArg::Int(c))],
        init_value: None,
    };
    let a = a.as_dtype(Dtype::Float32)?;
    // SAFETY: threadgroup m < `matrices` of `C` threads takes matrix m; thread j reads column j of
    // it and writes column j of the result, every row of it. Every output element is written once.
    Ok(unsafe { launch(&kernel, &[a], &config)? }.remove(0))
}

/// The forward of [`inverse_unit_lower`] as MLX operations, the reference. The blocks are taken
/// by reshapes, which keep every
/// array's layout simple: `[L, m, 2s, m, 2s]` holds `m` pairs of blocks of size `s`, and the
/// diagonal pairs' lower-left blocks are what the doubling reads.
fn doubling(a: &Array) -> Result<Array> {
    let shape = a.shape().to_vec();
    let total = shape[shape.len() - 1];
    let lead: i32 = shape[..shape.len() - 2].iter().product();
    let lower = tril(a, -1)?.reshape(&[lead, total, total])?;
    let mut inv = ones::<f32>(&[lead, total, 1, 1])?;
    let mut size = 1;
    while size < total {
        let m = total / (2 * size);
        let blocks = lower.reshape(&[lead, m, 2 * size, m, 2 * size])?;
        let pairs = move_axis(blocks.diagonal(0, 1, 3)?, -1, 1)?;
        let below = pairs.try_index((.., .., size.., ..size))?;
        let halves = inv.reshape(&[lead, m, 2, size, size])?;
        let top = halves.try_index((.., .., 0, .., ..))?;
        let bottom = halves.try_index((.., .., 1, .., ..))?;
        let corner = bottom.matmul(&below)?.matmul(&top)?.negative()?;
        inv = concatenate(
            &[
                concatenate(&[top.clone(), zeros_like(&top)?], -1)?,
                concatenate(&[corner, bottom], -1)?,
            ],
            -2,
        )?;
        size *= 2;
    }
    inv.reshape(&shape)
}
