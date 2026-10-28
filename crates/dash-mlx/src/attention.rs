//! Scaled dot-product attention with grouped key-value heads and the masks models train with,
//! whose backward pass keeps no T x T scores, and decoding against a key-value cache: dash-ops'
//! `attention` on MLX.
//!
//! The operation, its arguments, its results and which keys each query sees are dash-ops' (its
//! `attention` module documents them): queries `[B, Tq, Hq, D]`, keys and values
//! `[B, Tk, Hkv, D]` with `Hq` a multiple of `Hkv`, the output `[B, Tq, Hq, D]` and the
//! log-sum-exp `[B, Tq, Hq]` in f32; positions aligned at the bottom right; causality and a
//! sliding window from [`AttentionOptions`]; key padding or packed sequences from
//! [`AttentionMask`]; a query that sees no key returns zeros and a log-sum-exp of `-inf` and takes
//! no part in any gradient.
//!
//! On MLX:
//!
//! - Both passes are blocked, 512 queries by 512 keys a tile, and take only the tiles some query
//!   sees, so tiles wholly outside the causal triangle or the window cost nothing. Each tile's
//!   products are MLX's matmuls in f32, the grouped query heads of a key-value head together so
//!   that nothing is expanded (from bf16 inputs the products are exact, and with MLX's TF32 they run
//!   on the neural accelerators of GPUs that have them); what lies between the products is one
//!   Metal kernel a tile, the masks applied per element from the geometry, never materialised.
//! - The forward pass runs each block of queries over the key blocks it sees with an online
//!   softmax, the output accumulated in f32 and rounded once, and gives the log-sum-exp with it
//!   (MLX's fused attention returns none, and in bf16 rounds the probabilities before the value
//!   product: 4.78e-3 off the records where dash-ops' bound is 3.4e-3, measured).
//! - The backward pass is FlashAttention-2's, `delta = rowsum(dO * O)`, then per tile `S = scale Q K^T`,
//!   `P = exp(S - m) / s` from the forward pass's maximum and sum (from `lse` for a caller of
//!   [`attention_backward`]), `dV += P^T dO`, `dP = dO V^T`, `dS = P (dP - delta)`, `dK += scale dS^T Q`
//!   and `dQ += scale dS K`, the products MLX's matmuls in f32, the grouped query heads of a
//!   key-value head together in each product so that nothing is expanded. No more than one tile's
//!   scores are formed at a time; each block of queries' gradient is summed over the key blocks in
//!   order and each block of keys' over the query blocks in order, so a run repeats bit for bit;
//!   [`AttentionOptions::deterministic`] off runs the same backward pass, as dash-ops' reference
//!   does.
//! - MLX keeps each queued operation's temporaries until the GPU has run it, and queues as far as
//!   its command buffers allow, so a pass holds many tiles' arrays at once: the smaller each
//!   operation's arrays, the less (the backward pass in tiles rather than a key block against all
//!   its queries: +2112 MiB rather than +2672 at 6144 tokens, measured). Evaluating inside the
//!   operation makes it worse under MLX's autodiff, which is where training runs it: one
//!   evaluation in the backward pass took that peak to +7461 MiB (measured; the likely cause, not
//!   verified, is that MLX keeps arrays evaluated during a trace attached to their inputs).
//!   Smaller command buffers help (+1598 MiB with `MLX_MAX_OPS_PER_BUFFER=2`, and faster), but MLX
//!   reads that setting once per process.
//! - When the caller needs neither the log-sum-exp nor dash-ops' bf16 bound, and the mask is none,
//!   causal or key padding, MLX's fused attention (`mlx_rs::fast::scaled_dot_product_attention`)
//!   is the better call: its forward pass takes a quarter of the time and holds no tiles, its
//!   backward pass about as long with a little less memory (measured, README). A window or packed
//!   sequences it takes only as a materialised `Tq x Tk` mask.
//! - A gradient flowing into the log-sum-exp is taken too: `dS = P (dP - delta + dlse)`.
//! - Packed sequences run one at a time, their offsets read to the host, each as a call on it
//!   alone; queries and keys past the last offset take no part.
//! - Decoding is one kernel that reads the cache in dash-ops' layout, `[B, S, Hkv, D]`, in place
//!   (flash-decoding): a threadgroup takes a sequence, a key-value head and a split of its cache,
//!   reads each key and value row once for the group's query heads and keeps the softmax online in
//!   f32; the splits are merged by their maxima and sums. Per-sequence lengths and the window bound
//!   each sequence's keys. Its gradient, to the query and to the cached keys and values, is MLX's,
//!   through its fused attention over a transposed view of the cache, which also takes the shapes
//!   the kernel does not (more than 8 query heads a key-value head, a head dimension not a multiple
//!   of 32 or above 512).

use crate::kernel::{Launch, launch};
use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::{ScaledDotProductAttentionMask, TemplateArg, scaled_dot_product_attention};
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{
    arange, broadcast_to, concatenate, full, maximum, minimum, pad, select, sum_axis, zeros,
};
use mlx_rs::transforms::{custom_vjp, fallible_vjp};
use mlx_rs::{Array, Dtype};
use std::sync::atomic::Ordering;

/// Keys a block of the blocked passes takes.
const KEY_BLOCK: i32 = 512;

pub use dash_contract::attention::{AttentionOptions, Window};

/// Which keys each query may see, beside the causality and the window: dash-contract's, with
/// MLX's arrays. `KeyPadding` is `[B, Tk]` bool, `true` where a key is left out; `Packed` holds
/// the queries' and keys' cumulative offsets, `[S + 1]` each (any integer dtype), `B = 1`.
pub type AttentionMask = dash_contract::attention::AttentionMask<Array, Array>;

/// How far a query sees before and after its position, `None` for no bound: the window, and
/// causality's bound after it. `usize::MAX` leaves a side of the window unbounded.
fn reach(options: &AttentionOptions) -> (Option<i64>, Option<i64>) {
    let bound = |x: usize| (x != usize::MAX).then_some(x.min(i64::MAX as usize) as i64);
    let left = options.window.and_then(|w| bound(w.left));
    let right = match options.causal {
        true => Some(0),
        false => options.window.and_then(|w| bound(w.right)),
    };
    (left, right)
}

/// Counters tests read to tell which code ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);
    pub(super) static TILES: AtomicUsize = AtomicUsize::new(0);

    /// Runs of the blocked backward pass.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }

    /// Tiles the blocked passes have formed scores for.
    pub fn tiles() -> usize {
        TILES.load(Ordering::SeqCst)
    }
}

fn contract(op: &str, message: String) -> Exception {
    Exception::custom(format!("{op}: {message}"))
}

/// The dimensions of a training call, checked: `[B, Tq, Tk, Hq, Hkv, D]`.
fn check(op: &str, q: &Array, k: &Array, v: &Array) -> Result<[i32; 6]> {
    let (&[b, tq, hq, d], &[bk, tk, hkv, dk]) = (q.shape(), k.shape()) else {
        return Err(contract(
            op,
            format!(
                "queries [B, Tq, Hq, D] and keys [B, Tk, Hkv, D], got {:?} and {:?}",
                q.shape(),
                k.shape()
            ),
        ));
    };
    if bk != b || dk != d || v.shape() != k.shape() {
        return Err(contract(
            op,
            format!(
                "keys and values must be [B, Tk, Hkv, D] with B = {b}, D = {d}: got {:?} and {:?}",
                k.shape(),
                v.shape()
            ),
        ));
    }
    if hkv == 0 || hq % hkv != 0 {
        return Err(contract(
            op,
            format!("query heads ({hq}) must be a multiple of key-value heads ({hkv})"),
        ));
    }
    if !q.dtype().is_float() || k.dtype() != q.dtype() || v.dtype() != q.dtype() {
        return Err(contract(
            op,
            format!(
                "queries, keys and values must share one float dtype: {:?}, {:?}, {:?}",
                q.dtype(),
                k.dtype(),
                v.dtype()
            ),
        ));
    }
    Ok([b, tq, tk, hq, hkv, d])
}

/// Which keys the queries of one call (one sequence, or a batch) see.
#[derive(Clone)]
struct Geometry {
    tq: i32,
    tk: i32,
    left: Option<i64>,
    right: Option<i64>,
    /// `[B, Tk]` bool, `true` where a key is kept.
    keep: Option<Array>,
    scale: f32,
    /// The tile kernels, or MLX's operations (the reference).
    kernels: bool,
}

impl Geometry {
    /// Query `i` sits at key position `i + offset`.
    fn offset(&self) -> i64 {
        self.tk as i64 - self.tq as i64
    }

    /// The queries that see some key of `k0..k1`, `q0..q1` (possibly empty).
    fn queries_for(&self, k0: i32, k1: i32) -> (i32, i32) {
        let off = self.offset();
        let lo = self.right.map_or(0, |r| k0 as i64 - off - r);
        let hi = self.left.map_or(self.tq as i64, |l| k1 as i64 - off + l);
        let clamp = |x: i64| x.clamp(0, self.tq as i64) as i32;
        (clamp(lo), clamp(hi))
    }

    /// The keys some query of `q0..q1` sees by the band, `k0..k1` (possibly empty).
    fn keys_for(&self, q0: i32, q1: i32) -> (i32, i32) {
        let off = self.offset();
        let lo = self.left.map_or(0, |l| q0 as i64 + off - l);
        let hi = self
            .right
            .map_or(self.tk as i64, |r| q1 as i64 - 1 + off + r + 1);
        let clamp = |x: i64| x.clamp(0, self.tk as i64) as i32;
        (clamp(lo), clamp(hi))
    }

    /// The tile kernels' geometry: `[q0, k0, qn, offset, left, right, rows a batch element
    /// holds, Tk]`, and their template flags.
    fn kernel_geometry(
        &self,
        (q0, q1): (i32, i32),
        k0: i32,
        rows_per_batch: i32,
    ) -> (Array, Vec<(String, TemplateArg)>) {
        let reach = |x: Option<i64>| x.unwrap_or(0).min(1 << 30) as i32;
        let geo = [
            q0,
            k0,
            q1 - q0,
            self.offset() as i32,
            reach(self.left),
            reach(self.right),
            rows_per_batch,
            self.tk,
        ];
        let template = vec![
            ("HAS_LEFT".into(), TemplateArg::Bool(self.left.is_some())),
            ("HAS_RIGHT".into(), TemplateArg::Bool(self.right.is_some())),
            ("PADDED".into(), TemplateArg::Bool(self.keep.is_some())),
        ];
        (Array::from_slice(&geo, &[8]), template)
    }

    /// The padding the tile kernels read, or a placeholder.
    fn kernel_keep(&self) -> Array {
        self.keep
            .clone()
            .unwrap_or_else(|| Array::from_slice(&[true], &[1]))
    }

    /// Whether the band (causality and window) leaves any key out anywhere.
    fn banded(&self) -> bool {
        self.left.is_some() || self.right.is_some()
    }

    /// `[q1 - q0, k1 - k0]` bool: which keys of `k0..k1` the queries `q0..q1` see by the band.
    fn band(&self, (q0, q1): (i32, i32), (k0, k1): (i32, i32)) -> Result<Array> {
        let off = self.offset() as i32;
        let pos = arange::<_, i32>(q0 + off, q1 + off, None)?.expand_dims(-1)?;
        let keys = arange::<_, i32>(k0, k1, None)?.expand_dims(0)?;
        let mut keep = broadcast_to(Array::from_bool(true), &[q1 - q0, k1 - k0])?;
        if let Some(left) = self.left {
            let floor = pos.subtract(Array::from_int(left.min(i32::MAX as i64) as i32))?;
            keep = keep.logical_and(keys.ge(&floor)?)?;
        }
        if let Some(right) = self.right {
            let ceiling = pos.add(Array::from_int(right.min(i32::MAX as i64) as i32))?;
            keep = keep.logical_and(keys.le(&ceiling)?)?;
        }
        Ok(keep)
    }

    /// The tile's mask, `[B or 1, 1, 1, q1 - q0, k1 - k0]` bool, or `None` if it keeps everything:
    /// the band and the padding.
    fn tile(&self, queries: (i32, i32), keys: (i32, i32)) -> Result<Option<Array>> {
        let band = match self.banded() {
            true => Some(self.band(queries, keys)?.reshape(&[
                1,
                1,
                1,
                queries.1 - queries.0,
                keys.1 - keys.0,
            ])?),
            false => None,
        };
        let padding = match &self.keep {
            Some(keep) => {
                let b = keep.shape()[0];
                Some(keep.try_index((.., keys.0..keys.1))?.reshape(&[
                    b,
                    1,
                    1,
                    1,
                    keys.1 - keys.0,
                ])?)
            }
            None => None,
        };
        Ok(match (band, padding) {
            (Some(a), Some(b)) => Some(a.logical_and(&b)?),
            (a, b) => a.or(b),
        })
    }
}

/// `x` `[B, T, Hq, D]` as `[B, Hkv, G, T, D]` f32, the query heads grouped by key-value head.
fn grouped(x: &Array, hkv: i32) -> Result<Array> {
    let &[b, t, hq, d] = x.shape() else {
        unreachable!("checked")
    };
    x.as_dtype(Dtype::Float32)?
        .reshape(&[b, t, hkv, hq / hkv, d])?
        .transpose_axes(&[0, 2, 3, 1, 4])
}

/// `x` `[B, Hkv, G, T, ...]` back as `[B, T, Hq, ...]`.
fn ungrouped(x: &Array) -> Result<Array> {
    let shape = x.shape().to_vec();
    let (b, hkv, g, t) = (shape[0], shape[1], shape[2], shape[3]);
    match shape.len() {
        5 => x
            .transpose_axes(&[0, 3, 1, 2, 4])?
            .reshape(&[b, t, hkv * g, shape[4]]),
        _ => x.transpose_axes(&[0, 3, 1, 2])?.reshape(&[b, t, hkv * g]),
    }
}

/// The key blocks of `0..tk`.
fn key_blocks(tk: i32) -> impl Iterator<Item = (i32, i32)> {
    (0..tk)
        .step_by(KEY_BLOCK as usize)
        .map(move |k0| (k0, (k0 + KEY_BLOCK).min(tk)))
}

/// Queries a block of the forward pass takes.
const QUERY_BLOCK: i32 = 512;

const HEADER: &str = "
constant float DASH_LOWEST = -3.4028234663852886e38f;

// Whether row `r` (query `q0 + r % qn` of its batch element) of a tile sees column `c` (key
// `k0 + c`): the window and causality around the query's position, and the padding. Generic over
// the pointers' address spaces: MLX passes small inputs as `constant`.
template <typename G, typename K>
inline bool dash_visible(uint r, uint c, G geo, K keep, bool has_left, bool has_right, bool padded) {
    const int i = geo[0] + int(r % uint(geo[2]));
    const int j = geo[1] + int(c);
    const int pos = i + geo[3];
    if (has_left && j < pos - geo[4]) {
        return false;
    }
    if (has_right && j > pos + geo[5]) {
        return false;
    }
    if (padded && !keep[size_t(r / uint(geo[6])) * size_t(geo[7]) + size_t(j)]) {
        return false;
    }
    return true;
}
";

/// The forward pass's online softmax over one tile of scores, a simdgroup a row: the row's new
/// maximum, `P = exp(S - m)` where it sees the key (0 elsewhere), the new sum, and the factor
/// `exp(m_old - m)` the output so far is rescaled by.
const SOFTMAX: &str = "
    const uint row = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint R = m_in_shape[0];
    if (row >= R) {
        return;
    }
    const uint C = scores_shape[1];
    const size_t base = size_t(row) * C;
    float mx = DASH_LOWEST;
    for (uint c = lane; c < C; c += 32) {
        if (dash_visible(row, c, geo, keep, HAS_LEFT, HAS_RIGHT, PADDED)) {
            mx = metal::max(mx, scores[base + c]);
        }
    }
    mx = simd_max(mx);
    const float m_old = m_in[row];
    const float m_new = metal::max(m_old, mx);
    float sum = 0.0f;
    for (uint c = lane; c < C; c += 32) {
        float p = 0.0f;
        if (dash_visible(row, c, geo, keep, HAS_LEFT, HAS_RIGHT, PADDED)) {
            p = metal::precise::exp(scores[base + c] - m_new);
        }
        probs[base + c] = p;
        sum += p;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        const float alpha = metal::precise::exp(m_old - m_new);
        m_out[row] = m_new;
        s_out[row] = s_in[row] * alpha + sum;
        alpha_out[row] = alpha;
    }
";

/// The backward pass's probabilities and score gradients over one tile, a thread an element:
/// `P = exp(S - m) / s` where the query sees the key, 0 elsewhere, and `dS = P (dP - delta)`.
const GRADS: &str = "
    const uint c = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const uint C = scores_shape[1];
    if (c >= C) {
        return;
    }
    const size_t at = size_t(row) * C + c;
    float p = 0.0f;
    if (dash_visible(row, c, geo, keep, HAS_LEFT, HAS_RIGHT, PADDED)) {
        p = metal::precise::exp(scores[at] - m[row]) / s[row];
    }
    probs[at] = p;
    dscores[at] = p * (dprobs[at] - delta[row]);
";

/// Rows of a tile in the kernels' order: `[B, Hkv, G, qn]` flattened.
fn tile_rows(x: &Array) -> i32 {
    x.shape()[..4].iter().product()
}

/// The forward pass's online softmax over a tile of `scores` `[B, Hkv, G, qn, kn]` for the queries
/// `queries` and keys from `k0`: `[P, m, s, alpha]`, `P` shaped as the scores, the rest
/// `[B, Hkv, G, qn]`.
fn tile_softmax(
    scores: &Array,
    [m, s]: [&Array; 2],
    geom: &Geometry,
    queries: (i32, i32),
    keys: (i32, i32),
) -> Result<[Array; 4]> {
    if !geom.kernels {
        let tile = geom.tile(queries, keys)?;
        let masked = match &tile {
            Some(keep) => select(keep, scores, Array::from_f32(f32::MIN))?,
            None => scores.clone(),
        };
        let new_m = maximum(m, masked.max_axis(-1, None)?)?;
        let p = scores.subtract(new_m.expand_dims(-1)?)?.exp()?;
        let p = match &tile {
            Some(keep) => select(keep, &p, Array::from_f32(0.0))?,
            None => p,
        };
        let alpha = m.subtract(&new_m)?.exp()?;
        let new_s = s.multiply(&alpha)?.add(sum_axis(&p, -1, None)?)?;
        return Ok([p, new_m, new_s, alpha]);
    }
    let shape = scores.shape().to_vec();
    let rows = tile_rows(scores);
    let per_batch = rows / shape[0];
    let (geo, mut template) = geom.kernel_geometry(queries, keys.0, per_batch);
    let kernel = crate::kernel::kernel(
        "dash_mlx_attention_softmax",
        &["scores", "m_in", "s_in", "geo", "keep"],
        &["probs", "m_out", "s_out", "alpha_out"],
        SOFTMAX,
        HEADER,
    )?;
    template.shrink_to_fit();
    let config = Launch {
        grid: (((rows + 7) / 8) * 256, 1, 1),
        thread_group: (256, 1, 1),
        outputs: vec![
            (vec![rows, shape[4]], Dtype::Float32),
            (vec![rows], Dtype::Float32),
            (vec![rows], Dtype::Float32),
            (vec![rows], Dtype::Float32),
        ],
        template,
        init_value: None,
    };
    let inputs = [
        scores.reshape(&[rows, shape[4]])?,
        m.reshape(&[rows])?,
        s.reshape(&[rows])?,
        geo,
        geom.kernel_keep(),
    ];
    // SAFETY: one simdgroup per row of the [rows, C] row-contiguous scores, rows past the last
    // returning at once: each lane reads and writes columns `lane, lane + 32, ...` below C of its
    // row in `scores` and `probs` (shaped alike), its row of m_in and s_in, `geo[0..8]`, and
    // `keep[b * Tk + j]` only when PADDED, with `j = k0 + c < Tk` and `b` below the batch;
    // lane 0 writes the row's m_out, s_out and alpha_out. Every output element is written once;
    // no input is written.
    let out = unsafe { launch(&kernel, &inputs, &config)? };
    let row_shape = &shape[..4];
    Ok([
        out[0].reshape(&shape)?,
        out[1].reshape(row_shape)?,
        out[2].reshape(row_shape)?,
        out[3].reshape(row_shape)?,
    ])
}

/// The backward pass's `[P, dS]` over a tile of `scores` and `dprobs` `[B, Hkv, G, qn, kn]`, with
/// the rows' shift `m`, sum `s` and `delta`, `[B, Hkv, G, qn, 1]` each.
fn tile_grads(
    scores: &Array,
    dprobs: &Array,
    [m, s, delta]: [&Array; 3],
    geom: &Geometry,
    queries: (i32, i32),
    keys: (i32, i32),
) -> Result<[Array; 2]> {
    if !geom.kernels {
        let p = scores.subtract(m)?.exp()?.divide(s)?;
        let p = match geom.tile(queries, keys)? {
            Some(keep) => select(&keep, &p, Array::from_f32(0.0))?,
            None => p,
        };
        let ds = p.multiply(dprobs.subtract(delta)?)?;
        return Ok([p, ds]);
    }
    let shape = scores.shape().to_vec();
    let rows = tile_rows(scores);
    let cols = shape[4];
    let (geo, template) = geom.kernel_geometry(queries, keys.0, rows / shape[0]);
    let kernel = crate::kernel::kernel(
        "dash_mlx_attention_grads",
        &["scores", "dprobs", "m", "s", "delta", "geo", "keep"],
        &["probs", "dscores"],
        GRADS,
        HEADER,
    )?;
    let config = Launch {
        grid: (((cols + 31) / 32) * 32, rows, 1),
        thread_group: (32, 8, 1),
        outputs: vec![(vec![rows, cols], Dtype::Float32); 2],
        template,
        init_value: None,
    };
    let flat_rows = |x: &Array| x.reshape(&[rows]);
    let inputs = [
        scores.reshape(&[rows, cols])?,
        dprobs.reshape(&[rows, cols])?,
        flat_rows(&broadcast_to(
            m,
            &[shape[0], shape[1], shape[2], shape[3], 1],
        )?)?,
        flat_rows(&broadcast_to(
            s,
            &[shape[0], shape[1], shape[2], shape[3], 1],
        )?)?,
        flat_rows(delta)?,
        geo,
        geom.kernel_keep(),
    ];
    // SAFETY: a thread per element of the [rows, C] row-contiguous tiles, those at `c >= C`
    // returning at once: each reads its element of `scores` and `dprobs`, its row of m, s and
    // delta, `geo[0..8]`, and `keep[b * Tk + j]` only when PADDED (`j = k0 + c < Tk`), and writes
    // its element of `probs` and `dscores` once. No input is written.
    let out = unsafe { launch(&kernel, &inputs, &config)? };
    Ok([out[0].reshape(&shape)?, out[1].reshape(&shape)?])
}

/// The forward pass, blocked: per block of queries, the key blocks it sees, each tile's scores
/// `scale Q K^T` from MLX's matmul and the online softmax, the output accumulated in f32. Returns
/// the output `[B, Hkv, G, Tq, D]` f32 (zeros where a query sees no key), and the maximum and sum
/// `[B, Hkv, G, Tq]` (f32's lowest and 0 there).
fn blocked_forward(qg: &Array, kg: &Array, vg: &Array, geom: &Geometry) -> Result<[Array; 3]> {
    let shape = qg.shape().to_vec();
    let (b, hkv, g, tq, d) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let qg = qg.multiply(Array::from_f32(geom.scale))?;
    let (mut outs, mut ms, mut ss) = (Vec::new(), Vec::new(), Vec::new());
    for q0 in (0..tq).step_by(QUERY_BLOCK as usize) {
        let q1 = (q0 + QUERY_BLOCK).min(tq);
        let qn = q1 - q0;
        let qs = qg.try_index((.., .., .., q0..q1, ..))?;
        let mut m = full::<f32>(&[b, hkv, g, qn], Array::from_f32(f32::MIN))?;
        let mut s = zeros::<f32>(&[b, hkv, g, qn])?;
        let mut o = zeros::<f32>(&[b, hkv, g, qn, d])?;
        let (lo, hi) = geom.keys_for(q0, q1);
        let first = (lo / KEY_BLOCK) * KEY_BLOCK;
        for k0 in (first..hi).step_by(KEY_BLOCK as usize) {
            let k1 = (k0 + KEY_BLOCK).min(geom.tk);
            probe::TILES.fetch_add(1, Ordering::SeqCst);
            let kb = kg.try_index((.., .., k0..k1, ..))?.expand_dims(2)?;
            let vb = vg.try_index((.., .., k0..k1, ..))?.expand_dims(2)?;
            let scores = qs.matmul(kb.swap_axes(-1, -2)?)?;
            let [p, new_m, new_s, alpha] =
                tile_softmax(&scores, [&m, &s], geom, (q0, q1), (k0, k1))?;
            o = o.multiply(alpha.expand_dims(-1)?)?.add(p.matmul(&vb)?)?;
            (m, s) = (new_m, new_s);
        }
        let seen = s.gt(Array::from_f32(0.0))?.expand_dims(-1)?;
        let denominator = select(&seen, s.expand_dims(-1)?, Array::from_f32(1.0))?;
        outs.push(o.divide(&denominator)?);
        ms.push(m);
        ss.push(s);
    }
    Ok([
        concatenate(&outs, 3)?,
        concatenate(&ms, 3)?,
        concatenate(&ss, 3)?,
    ])
}

/// How the backward pass normalises the scores: by the log-sum-exp `[B, Tq, Hq]`, as a caller of
/// [`attention_backward`] gives it, or by the forward pass's maximum and sum `[B, Hkv, G, Tq]`,
/// `P = exp(S - m) / s`, which keeps the rounding of `m + ln s` out of every probability.
enum Normaliser<'a> {
    Lse(&'a Array),
    MaxSum(&'a Array, &'a Array),
}

/// The gradients of one call, `[dq, dk, dv]` in the inputs' dtype: FlashAttention-2's backward
/// pass, blocked over the keys ([module](self)).
#[allow(clippy::too_many_arguments)]
fn blocked_backward(
    [q, k, v]: [&Array; 3],
    out: &Array,
    normaliser: Normaliser,
    d_out: &Array,
    d_lse: Option<&Array>,
    geom: &Geometry,
) -> Result<[Array; 3]> {
    probe::BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
    let &[b, tq, hq, d] = q.shape() else {
        unreachable!("checked")
    };
    let hkv = k.shape()[2];
    let g = hq / hkv;
    let scale = Array::from_f32(geom.scale);
    let qg = grouped(q, hkv)?;
    let og = grouped(out, hkv)?;
    let dog = grouped(d_out, hkv)?;
    let (shift, sum) = match normaliser {
        Normaliser::Lse(lse) => (
            grouped(&lse.expand_dims(-1)?, hkv)?.squeeze_axes(&[-1])?,
            None,
        ),
        Normaliser::MaxSum(m, s) => (m.clone(), Some(s.clone())),
    };
    // delta = rowsum(dO * O), less the log-sum-exp's own gradient.
    let mut delta = sum_axis(&dog.multiply(&og)?, -1, None)?;
    if let Some(d_lse) = d_lse {
        delta = delta.subtract(grouped(&d_lse.expand_dims(-1)?, hkv)?.squeeze_axes(&[-1])?)?;
    }
    let kg = k.as_dtype(Dtype::Float32)?.swap_axes(1, 2)?;
    let vg = v.as_dtype(Dtype::Float32)?.swap_axes(1, 2)?;
    // The query gradient by block of queries, each summed over the key blocks in order.
    let query_blocks: Vec<(i32, i32)> = (0..tq)
        .step_by(QUERY_BLOCK as usize)
        .map(|q0| (q0, (q0 + QUERY_BLOCK).min(tq)))
        .collect();
    let mut dqs: Vec<Option<Array>> = vec![None; query_blocks.len()];
    let mut dks = Vec::new();
    let mut dvs = Vec::new();
    let ones = Array::from_f32(1.0);
    for (k0, k1) in key_blocks(geom.tk) {
        let kn = k1 - k0;
        let (lo, hi) = geom.queries_for(k0, k1);
        let kb = kg.try_index((.., .., k0..k1, ..))?.expand_dims(2)?;
        let vb = vg.try_index((.., .., k0..k1, ..))?.expand_dims(2)?;
        let (mut dk, mut dv) = (None, None);
        // A tile at a time, 512 queries by 512 keys: what MLX holds while the GPU catches up
        // with the queued tiles scales with a tile's arrays.
        for (block, &(b0, b1)) in query_blocks.iter().enumerate() {
            let (q0, q1) = (b0.max(lo), b1.min(hi));
            if q0 >= q1 {
                continue;
            }
            probe::TILES.fetch_add(1, Ordering::SeqCst);
            let rows = |x: &Array| x.try_index((.., .., .., q0..q1));
            let qs = rows(&qg)?;
            let dos = rows(&dog)?;
            let ls = rows(&shift)?.expand_dims(-1)?;
            let ds = rows(&delta)?.expand_dims(-1)?;
            let scores = qs.multiply(&scale)?.matmul(kb.swap_axes(-1, -2)?)?;
            let dp = dos.matmul(vb.swap_axes(-1, -2)?)?;
            let ss = match &sum {
                Some(sum) => rows(sum)?.expand_dims(-1)?,
                None => broadcast_to(&ones, ls.shape())?,
            };
            let [p, dscores] = tile_grads(&scores, &dp, [&ls, &ss, &ds], geom, (q0, q1), (k0, k1))?;
            // dV and dK per query head, then summed over the group, as torch sums them.
            let group_sum = |x: Array| sum_axis(&x, 2, None);
            let add = |sum: Option<Array>, x: Array| match sum {
                Some(sum) => sum.add(x),
                None => Ok(x),
            };
            dv = Some(add(dv, group_sum(p.swap_axes(-1, -2)?.matmul(&dos)?)?)?);
            dk = Some(add(
                dk,
                group_sum(dscores.swap_axes(-1, -2)?.matmul(&qs)?)?,
            )?);
            let mut dq_tile = dscores.matmul(&kb)?.multiply(&scale)?;
            if (q0, q1) != (b0, b1) {
                let widths = [(0, 0), (0, 0), (0, 0), (q0 - b0, b1 - q1), (0, 0)];
                dq_tile = pad(&dq_tile, &widths[..], None, None)?;
            }
            dqs[block] = Some(add(dqs[block].take(), dq_tile)?);
        }
        let none = || zeros::<f32>(&[b, hkv, kn, d]);
        dks.push(match dk {
            Some(dk) => dk.multiply(&scale)?,
            None => none()?,
        });
        dvs.push(dv.map_or_else(none, Ok)?);
    }
    let dq = dqs
        .into_iter()
        .zip(&query_blocks)
        .map(|(dq, &(b0, b1))| dq.map_or_else(|| zeros::<f32>(&[b, hkv, g, b1 - b0, d]), Ok))
        .collect::<Result<Vec<_>>>()?;
    let dq = concatenate(&dq, 3)?;
    let dk = concatenate(&dks, 2)?.swap_axes(1, 2)?.as_dtype(k.dtype())?;
    let dv = concatenate(&dvs, 2)?.swap_axes(1, 2)?.as_dtype(v.dtype())?;
    Ok([ungrouped(&dq)?.as_dtype(q.dtype())?, dk, dv])
}

/// One unpacked call: the output and the log-sum-exp, differentiable through a custom VJP whose
/// backward pass is [`blocked_backward`].
fn unpacked(q: &Array, k: &Array, v: &Array, geom: Geometry) -> Result<(Array, Array)> {
    let hkv = k.shape()[2];
    let forward_geom = geom.clone();
    let op = custom_vjp(
        move |p: &[Array]| {
            let geom = &forward_geom;
            let (q, k, v) = (&p[0], &p[1], &p[2]);
            let qg = grouped(q, hkv)?;
            let kg = k.as_dtype(Dtype::Float32)?.swap_axes(1, 2)?;
            let vg = v.as_dtype(Dtype::Float32)?.swap_axes(1, 2)?;
            let [out, m, s] = blocked_forward(&qg, &kg, &vg, geom)?;
            let lse = ungrouped(&m.add(s.log()?)?)?;
            let out = ungrouped(&out)?.as_dtype(q.dtype())?;
            Ok(vec![out, lse, m, s])
        },
        move |p: &[Array], cotangents: &[Array], outputs: &[Array]| {
            let d_lse = cotangents.get(1);
            let grads = blocked_backward(
                [&p[0], &p[1], &p[2]],
                &outputs[0],
                Normaliser::MaxSum(&outputs[2], &outputs[3]),
                &cotangents[0],
                d_lse,
                &geom,
            )?;
            Ok(grads.to_vec())
        },
    )?;
    let mut outputs = op.apply(&[q.clone(), k.clone(), v.clone()])?;
    outputs.truncate(2);
    let lse = outputs.pop().expect("two outputs");
    Ok((outputs.pop().expect("two outputs"), lse))
}

/// Offsets read to the host and checked: from 0, not decreasing, the last at most `t`.
fn offsets(op: &str, cu: &Array, t: i32, what: &str) -> Result<Vec<i32>> {
    let values: Vec<i64> = cu
        .as_dtype(Dtype::Int64)?
        .contiguous()?
        .as_slice::<i64>()
        .to_vec();
    let ok = values.first() == Some(&0)
        && values.windows(2).all(|w| w[0] <= w[1])
        && values.last().is_some_and(|&last| last <= t as i64);
    match ok {
        true => Ok(values.into_iter().map(|x| x as i32).collect()),
        false => Err(contract(
            op,
            format!("{what} must go from 0 up to at most {t} without decreasing, got {values:?}"),
        )),
    }
}

/// Every call's work: the checks, the mask, packed sequences one at a time.
fn dispatch(
    op: &str,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: &AttentionMask,
    options: AttentionOptions,
    kernels: bool,
) -> Result<(Array, Array)> {
    let [b, tq, tk, hq, _hkv, d] = check(op, q, k, v)?;
    let (left, right) = reach(&options);
    let geom = |tq: i32, tk: i32, keep: Option<Array>| Geometry {
        tq,
        tk,
        left,
        right,
        keep,
        scale: options.scale_for(d as usize),
        kernels,
    };
    match mask {
        AttentionMask::None => unpacked(q, k, v, geom(tq, tk, None)),
        AttentionMask::KeyPadding(padding) => {
            if padding.shape() != [b, tk] || padding.dtype() != Dtype::Bool {
                return Err(contract(
                    op,
                    format!(
                        "a key padding mask must be [B, Tk] = [{b}, {tk}] bool, got {:?} {:?}",
                        padding.shape(),
                        padding.dtype()
                    ),
                ));
            }
            unpacked(q, k, v, geom(tq, tk, Some(padding.logical_not()?)))
        }
        AttentionMask::Packed {
            cu_seqlens_q,
            cu_seqlens_k,
        } => {
            if b != 1 {
                return Err(contract(
                    op,
                    format!("packed sequences take B = 1, got {b}"),
                ));
            }
            let cq = offsets(op, cu_seqlens_q, tq, "cu_seqlens_q")?;
            let ck = offsets(op, cu_seqlens_k, tk, "cu_seqlens_k")?;
            if cq.len() != ck.len() {
                return Err(contract(
                    op,
                    format!(
                        "cu_seqlens_q and cu_seqlens_k must hold as many sequences: {} and {}",
                        cq.len() - 1,
                        ck.len() - 1
                    ),
                ));
            }
            let mut outs = Vec::new();
            let mut lses = Vec::new();
            for s in 0..cq.len() - 1 {
                let (q0, q1, k0, k1) = (cq[s], cq[s + 1], ck[s], ck[s + 1]);
                if q1 == q0 {
                    continue;
                }
                if k1 == k0 {
                    outs.push(zeros::<f32>(&[1, q1 - q0, hq, d])?.as_dtype(q.dtype())?);
                    lses.push(full::<f32>(
                        &[1, q1 - q0, hq],
                        Array::from_f32(f32::NEG_INFINITY),
                    )?);
                    continue;
                }
                let part = |x: &Array, a: i32, z: i32| x.try_index((.., a..z));
                let (out, lse) = unpacked(
                    &part(q, q0, q1)?,
                    &part(k, k0, k1)?,
                    &part(v, k0, k1)?,
                    geom(q1 - q0, k1 - k0, None),
                )?;
                outs.push(out);
                lses.push(lse);
            }
            // Queries past the last offset belong to no sequence.
            let last = *cq.last().expect("one offset at least");
            if last < tq {
                outs.push(zeros::<f32>(&[1, tq - last, hq, d])?.as_dtype(q.dtype())?);
                lses.push(full::<f32>(
                    &[1, tq - last, hq],
                    Array::from_f32(f32::NEG_INFINITY),
                )?);
            }
            Ok((concatenate(&outs, 1)?, concatenate(&lses, 1)?))
        }
    }
}

/// Attention for training, with dash-ops' arguments and results ([module](self)): the output
/// `[B, Tq, Hq, D]` in the queries' dtype. Differentiable: the backward pass keeps no T x T scores.
pub fn attention(
    query: &Array,
    key: &Array,
    value: &Array,
    mask: &AttentionMask,
    options: AttentionOptions,
) -> Result<Array> {
    Ok(dispatch("attention", query, key, value, mask, options, true)?.0)
}

/// [`attention`] with the natural log-sum-exp of each query's scaled scores over the keys it
/// sees, `[B, Tq, Hq]` in f32, `-inf` for a query that sees no key.
pub fn attention_with_lse(
    query: &Array,
    key: &Array,
    value: &Array,
    mask: &AttentionMask,
    options: AttentionOptions,
) -> Result<(Array, Array)> {
    dispatch("attention_with_lse", query, key, value, mask, options, true)
}

/// [`attention_with_lse`] with each tile's softmax and gradients as MLX's operations rather than
/// the tile kernels: the reference the tests hold the kernels to.
#[doc(hidden)]
pub fn attention_with_lse_reference(
    query: &Array,
    key: &Array,
    value: &Array,
    mask: &AttentionMask,
    options: AttentionOptions,
) -> Result<(Array, Array)> {
    dispatch(
        "attention_with_lse",
        query,
        key,
        value,
        mask,
        options,
        false,
    )
}

/// The backward pass of [`attention_with_lse`] for callers that run their own: the gradients of
/// the query, the key and the value from the output's gradient, given the forward pass's output
/// and log-sum-exp.
#[allow(clippy::too_many_arguments)]
pub fn attention_backward(
    query: &Array,
    key: &Array,
    value: &Array,
    output: &Array,
    lse: &Array,
    grad_output: &Array,
    mask: &AttentionMask,
    options: AttentionOptions,
) -> Result<(Array, Array, Array)> {
    let op = "attention_backward";
    let [b, tq, tk, hq, _hkv, d] = check(op, query, key, value)?;
    if output.shape() != query.shape()
        || grad_output.shape() != query.shape()
        || lse.shape() != [b, tq, hq]
    {
        return Err(contract(
            op,
            format!(
                "output and grad_output must be shaped as the queries {:?} and lse [B, Tq, Hq], got {:?}, {:?}, {:?}",
                query.shape(),
                output.shape(),
                grad_output.shape(),
                lse.shape()
            ),
        ));
    }
    let kernels = true;
    let (left, right) = reach(&options);
    let geom = |tq: i32, tk: i32, keep: Option<Array>| Geometry {
        tq,
        tk,
        left,
        right,
        keep,
        scale: options.scale_for(d as usize),
        kernels,
    };
    let run = |q: &Array, k: &Array, v: &Array, o: &Array, l: &Array, g: &Array, geom: Geometry| {
        blocked_backward([q, k, v], o, Normaliser::Lse(l), g, None, &geom)
    };
    let [dq, dk, dv] = match mask {
        AttentionMask::None => run(
            query,
            key,
            value,
            output,
            lse,
            grad_output,
            geom(tq, tk, None),
        )?,
        AttentionMask::KeyPadding(padding) => run(
            query,
            key,
            value,
            output,
            lse,
            grad_output,
            geom(tq, tk, Some(padding.logical_not()?)),
        )?,
        AttentionMask::Packed {
            cu_seqlens_q,
            cu_seqlens_k,
        } => {
            let cq = offsets(op, cu_seqlens_q, tq, "cu_seqlens_q")?;
            let ck = offsets(op, cu_seqlens_k, tk, "cu_seqlens_k")?;
            let (mut dqs, mut dks, mut dvs) = (Vec::new(), Vec::new(), Vec::new());
            let zero = |t: i32, h: i32, dtype: Dtype| zeros::<f32>(&[1, t, h, d])?.as_dtype(dtype);
            let hkv = key.shape()[2];
            let (mut q_end, mut k_end) = (0, 0);
            for s in 0..cq.len().min(ck.len()) - 1 {
                let (q0, q1, k0, k1) = (cq[s], cq[s + 1], ck[s], ck[s + 1]);
                let part = |x: &Array, a: i32, z: i32| x.try_index((.., a..z));
                let [dq, dk, dv] = if q1 > q0 && k1 > k0 {
                    run(
                        &part(query, q0, q1)?,
                        &part(key, k0, k1)?,
                        &part(value, k0, k1)?,
                        &part(output, q0, q1)?,
                        &part(lse, q0, q1)?,
                        &part(grad_output, q0, q1)?,
                        geom(q1 - q0, k1 - k0, None),
                    )?
                } else {
                    [
                        zero(q1 - q0, hq, query.dtype())?,
                        zero(k1 - k0, hkv, key.dtype())?,
                        zero(k1 - k0, hkv, value.dtype())?,
                    ]
                };
                dqs.push(dq);
                dks.push(dk);
                dvs.push(dv);
                (q_end, k_end) = (q1, k1);
            }
            if q_end < tq {
                dqs.push(zero(tq - q_end, hq, query.dtype())?);
            }
            if k_end < tk {
                dks.push(zero(tk - k_end, hkv, key.dtype())?);
                dvs.push(zero(tk - k_end, hkv, value.dtype())?);
            }
            [
                concatenate(&dqs, 1)?,
                concatenate(&dks, 1)?,
                concatenate(&dvs, 1)?,
            ]
        }
    };
    Ok((dq, dk, dv))
}

/// Decoding, with dash-ops' arguments and results: `query` `[B, Hq, D]`, one token per sequence,
/// against `key` and `value` `[B, S, Hkv, D]`, the cache up to and including that token.
/// `cache_lengths`, `[B]` (any integer dtype), gives how many keys of the cache each sequence
/// holds, `S` for all when `None`; each sequence's query sits after its last key. Of the options
/// the scale and the window's `left` apply. Returns `[B, Hq, D]` in the query's dtype; a sequence
/// holding no key gets zeros. Differentiable, to the query and the cache, through MLX.
pub fn attention_decode(
    query: &Array,
    key: &Array,
    value: &Array,
    cache_lengths: Option<&Array>,
    options: AttentionOptions,
) -> Result<Array> {
    // Checked by the MLX path, which also serves gradients and the shapes the kernel does not take.
    let fallback = || decode_mlx(query, key, value, cache_lengths, options);
    let (&[b, hq, d], &[_, s, hkv, _]) = (query.shape(), key.shape()) else {
        return fallback();
    };
    if key.shape().first() != Some(&b) || value.shape() != key.shape() || hkv == 0 || hq % hkv != 0
    {
        return fallback();
    }
    let g = hq / hkv;
    let floats = |simdgroups: i32| simdgroups * g * d;
    let simdgroups = [4, 2, 1].into_iter().find(|&n| floats(n) <= 7168);
    let (Some(simdgroups), true, true) = (simdgroups, d % 32 == 0 && d <= 512, g <= 8 && s > 0)
    else {
        return fallback();
    };
    // Each sequence's keys, lo..hi: its length, and the window's left side before its last key.
    let hi = match cache_lengths {
        Some(lengths) if lengths.shape() == [b] => maximum(
            minimum(lengths.as_dtype(Dtype::Int32)?, Array::from_int(s))?,
            Array::from_int(0),
        )?,
        Some(_) => return fallback(),
        None => full::<i32>(&[b], Array::from_int(s))?,
    };
    let lo = match reach(&options).0 {
        Some(left) => maximum(
            hi.subtract(Array::from_int(1 + left.min(i32::MAX as i64 - 1) as i32))?,
            Array::from_int(0),
        )?,
        None => zeros::<i32>(&[b])?,
    };
    let range = mlx_rs::ops::stack(&[lo, hi], 1)?.as_dtype(Dtype::Int32)?;
    let splits = ((s + 63) / 64).min((1024 + b * hkv - 1) / (b * hkv)).max(1);
    let scale = options.scale_for(d as usize);
    let lengths = cache_lengths.cloned();
    let op = custom_vjp(
        move |p: &[Array]| {
            let [out, m, sum] =
                decode_kernel(&p[0], &p[1], &p[2], &range, scale, [g, splits, simdgroups])?;
            // The splits merged by their maxima and sums; a sequence holding no key gets zeros.
            let top = m.max_axis(2, true)?;
            let weight = m.subtract(&top)?.exp()?;
            let total = sum_axis(&weight.multiply(&sum)?, 2, None)?;
            let merged = sum_axis(&weight.expand_dims(-1)?.multiply(&out)?, 2, None)?;
            let seen = total.gt(Array::from_f32(0.0))?;
            let denominator = select(&seen, &total, Array::from_f32(1.0))?.expand_dims(-1)?;
            let out = merged.divide(&denominator)?.reshape(&[b, hq, d])?;
            Ok(vec![out.as_dtype(p[0].dtype())?])
        },
        move |p: &[Array], cotangents: &[Array], _: &[Array]| {
            let f = |a: &[Array]| -> Result<Vec<Array>> {
                Ok(vec![decode_mlx(
                    &a[0],
                    &a[1],
                    &a[2],
                    lengths.as_ref(),
                    options,
                )?])
            };
            Ok(fallible_vjp(f, p, cotangents)?.1)
        },
    )?;
    Ok(op
        .apply(&[query.clone(), key.clone(), value.clone()])?
        .remove(0))
}

/// One threadgroup a sequence, key-value head and split of the cache: its `SG` simdgroups take
/// every `SG`-th key of the split, each lane `D / 32` of the dimensions of the key, the value and
/// each query head's running output; the scores are `simd_sum`s, the softmax online per query
/// head in f32, and the simdgroups meet once at the end in threadgroup memory. The cache is read
/// in dash-ops' layout, `[B, S, Hkv, D]`, each key and value row once for the group's query heads.
const DECODE: &str = "
    const uint split = threadgroup_position_in_grid.x;
    const uint hk = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const int S = k_shape[1];
    const int HKV = k_shape[2];
    const int HQ = q_shape[1];
    constexpr int PER = D / 32;
    const int lo = range[b * 2];
    const int hi = range[b * 2 + 1];
    const int chunk = (hi - lo + SPLITS - 1) / SPLITS;
    const int j0 = lo + int(split) * chunk;
    const int j1 = metal::min(hi, j0 + chunk);
    const float sc = scale[0];
    float qv[G][PER];
    float m[G];
    float s[G];
    float o[G][PER];
    for (int g = 0; g < G; ++g) {
        const size_t qbase = (size_t(b) * HQ + hk * G + g) * D + lane * PER;
        for (int i = 0; i < PER; ++i) {
            qv[g][i] = float(q[qbase + i]) * sc;
            o[g][i] = 0.0f;
        }
        m[g] = DASH_LOWEST;
        s[g] = 0.0f;
    }
    for (int j = j0 + int(sg); j < j1; j += SG) {
        const size_t base = ((size_t(b) * S + j) * HKV + hk) * D + lane * PER;
        float kv[PER];
        float vv[PER];
        for (int i = 0; i < PER; ++i) {
            kv[i] = float(k[base + i]);
            vv[i] = float(v[base + i]);
        }
        for (int g = 0; g < G; ++g) {
            float dot = 0.0f;
            for (int i = 0; i < PER; ++i) {
                dot += qv[g][i] * kv[i];
            }
            dot = simd_sum(dot);
            const float nm = metal::max(m[g], dot);
            const float alpha = metal::precise::exp(m[g] - nm);
            const float p = metal::precise::exp(dot - nm);
            s[g] = s[g] * alpha + p;
            for (int i = 0; i < PER; ++i) {
                o[g][i] = o[g][i] * alpha + p * vv[i];
            }
            m[g] = nm;
        }
    }
    threadgroup float tm[SG * G];
    threadgroup float ts[SG * G];
    threadgroup float to[SG * G * D];
    for (int g = 0; g < G; ++g) {
        if (lane == 0) {
            tm[sg * G + g] = m[g];
            ts[sg * G + g] = s[g];
        }
        for (int i = 0; i < PER; ++i) {
            to[(sg * G + g) * D + lane * PER + i] = o[g][i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const size_t row = ((size_t(b) * HKV + hk) * SPLITS + split) * G;
    for (uint e = tid; e < uint(G * D); e += SG * 32) {
        const uint g = e / D;
        const uint dd = e % D;
        float mm = DASH_LOWEST;
        for (int w = 0; w < SG; ++w) {
            mm = metal::max(mm, tm[w * G + g]);
        }
        float acc = 0.0f;
        float total = 0.0f;
        for (int w = 0; w < SG; ++w) {
            const float f = metal::precise::exp(tm[w * G + g] - mm);
            acc += to[(w * G + g) * D + dd] * f;
            total += ts[w * G + g] * f;
        }
        out_part[(row + g) * D + dd] = acc;
        if (dd == 0) {
            m_part[row + g] = mm;
            s_part[row + g] = total;
        }
    }
";

/// The decoding kernel's partial results per split: the output `[B, Hkv, P, G, D]` (not yet
/// divided by the sum), and the maximum and sum `[B, Hkv, P, G]`, all f32.
fn decode_kernel(
    q: &Array,
    k: &Array,
    v: &Array,
    range: &Array,
    scale: f32,
    [g, splits, simdgroups]: [i32; 3],
) -> Result<[Array; 3]> {
    let (b, hkv, d) = (q.shape()[0], k.shape()[2], q.shape()[2]);
    let kernel = crate::kernel::kernel(
        "dash_mlx_attention_decode",
        &["q", "k", "v", "range", "scale"],
        &["out_part", "m_part", "s_part"],
        DECODE,
        HEADER,
    )?;
    let config = Launch {
        grid: (splits * simdgroups * 32, hkv, b),
        thread_group: (simdgroups * 32, 1, 1),
        outputs: vec![
            (vec![b, hkv, splits, g, d], Dtype::Float32),
            (vec![b, hkv, splits, g], Dtype::Float32),
            (vec![b, hkv, splits, g], Dtype::Float32),
        ],
        template: vec![
            ("G".into(), TemplateArg::Int(g)),
            ("D".into(), TemplateArg::Int(d)),
            ("SG".into(), TemplateArg::Int(simdgroups)),
            ("SPLITS".into(), TemplateArg::Int(splits)),
        ],
        init_value: None,
    };
    let inputs = [
        q.clone(),
        k.clone(),
        v.clone(),
        range.clone(),
        Array::from_slice(&[scale], &[1]),
    ];
    // SAFETY: threadgroup (split, hk, b) reads q's rows `b, hk * G .. hk * G + G` (inside
    // [B, Hq, D] since Hq = Hkv G), the cache rows `b, j, hk` for `lo <= j < hi <= S` (inside
    // [B, S, Hkv, D]; `range` holds `0 <= lo <= hi <= S`), `range[2b..2b + 2]` and `scale[0]`, and
    // writes the G rows of its split in each output, every element once (the D columns of each
    // row by the threads in turn, the maximum and sum by the thread at column 0). The threadgroup
    // memory, SG G (D + 2) floats, stays within 28 KiB by the choice of SG. No input is written.
    let out = unsafe { launch(&kernel, &inputs, &config)? };
    Ok([out[0].clone(), out[1].clone(), out[2].clone()])
}

/// Decoding through MLX's fused attention, the cache seen through a transposed view: the gradient
/// of the kernel path, and its path for the shapes the kernel does not take.
fn decode_mlx(
    query: &Array,
    key: &Array,
    value: &Array,
    cache_lengths: Option<&Array>,
    options: AttentionOptions,
) -> Result<Array> {
    let op = "attention_decode";
    let (&[b, hq, d], &[bk, s, hkv, dk]) = (query.shape(), key.shape()) else {
        return Err(contract(
            op,
            format!(
                "query [B, Hq, D] and cache [B, S, Hkv, D], got {:?} and {:?}",
                query.shape(),
                key.shape()
            ),
        ));
    };
    if bk != b || dk != d || value.shape() != key.shape() || hkv == 0 || hq % hkv != 0 {
        return Err(contract(
            op,
            format!(
                "cache [B, S, Hkv, D] with B = {b}, D = {d} and Hq a multiple of Hkv, got {:?} and {:?}",
                key.shape(),
                value.shape()
            ),
        ));
    }
    let left = reach(&options).0;
    let keys = arange::<_, i32>(0, s, None)?.reshape(&[1, 1, 1, s])?;
    let lengths = match cache_lengths {
        Some(lengths) => {
            if lengths.shape() != [b] {
                return Err(contract(
                    op,
                    format!(
                        "cache_lengths must be [B] = [{b}], got {:?}",
                        lengths.shape()
                    ),
                ));
            }
            Some(lengths.as_dtype(Dtype::Int32)?.reshape(&[b, 1, 1, 1])?)
        }
        None => None,
    };
    let mut keep: Option<Array> = None;
    if let Some(lengths) = &lengths {
        keep = Some(keys.lt(lengths)?);
    }
    if let Some(left) = left {
        let last = match &lengths {
            Some(lengths) => lengths.subtract(Array::from_int(1))?,
            None => Array::from_int(s - 1),
        };
        let floor = keys.ge(last.subtract(Array::from_int(left.min(i32::MAX as i64) as i32))?)?;
        keep = Some(match keep {
            Some(k) => k.logical_and(&floor)?,
            None => floor,
        });
    }
    let q = query.expand_dims(2)?;
    let (k, v) = (key.swap_axes(1, 2)?, value.swap_axes(1, 2)?);
    let scale = options.scale_for(d as usize);
    let out = match &keep {
        Some(keep) => {
            let out = scaled_dot_product_attention(
                &q,
                &k,
                &v,
                scale,
                ScaledDotProductAttentionMask::Array(keep),
                None,
            )?;
            // A sequence holding no key: zeros.
            let any = keep.any_axis(-1, None)?.reshape(&[b, 1, 1, 1])?;
            select(&any, &out, Array::from_f32(0.0))?.as_dtype(query.dtype())?
        }
        None => scaled_dot_product_attention(&q, &k, &v, scale, None, None)?,
    };
    out.squeeze_axes(&[2])
}

/// Causal attention for training with the scale given: [`attention`] with no mask and
/// [`AttentionOptions::causal`].
pub fn causal_attention(query: &Array, key: &Array, value: &Array, scale: f32) -> Result<Array> {
    attention(
        query,
        key,
        value,
        &AttentionMask::None,
        AttentionOptions::causal().with_scale(scale),
    )
}

/// [`causal_attention`] with each query's log-sum-exp: [`attention_with_lse`].
pub fn causal_attention_with_lse(
    query: &Array,
    key: &Array,
    value: &Array,
    scale: f32,
) -> Result<(Array, Array)> {
    attention_with_lse(
        query,
        key,
        value,
        &AttentionMask::None,
        AttentionOptions::causal().with_scale(scale),
    )
}

/// The backward pass of [`causal_attention`]: [`attention_backward`].
#[allow(clippy::too_many_arguments)]
pub fn causal_attention_backward(
    query: &Array,
    key: &Array,
    value: &Array,
    output: &Array,
    lse: &Array,
    grad_output: &Array,
    scale: f32,
) -> Result<(Array, Array, Array)> {
    attention_backward(
        query,
        key,
        value,
        output,
        lse,
        grad_output,
        &AttentionMask::None,
        AttentionOptions::causal().with_scale(scale),
    )
}
