//! The log-probability of chosen tokens under a large vocabulary, with the entropy beside it,
//! without holding the n x V logits: dash-ops' `logprob` on MLX.
//!
//! The operation, its arguments, its results and the online log-sum-exp it walks the vocabulary
//! with are dash-ops' (its `logprob` module documents them). For rows of `hidden` `[n, D]`, an
//! output weight `[V, D]` (frozen) and target ids `[n]`: the log-probability of each row's target
//! under `softmax(hidden weight^T)` and the entropy of that distribution, both f32 `[n]`. The
//! masked token ids take no part in either, and a target that is masked or outside `0..V` has
//! the log-probability `-inf`. Only `hidden` gets a gradient.
//!
//! On MLX:
//!
//! - The logits are formed a chunk of the vocabulary at a time, `[rows, C]` in f32, from `hidden`
//!   and the weight's chunk cast to f32. bf16 values are exact in f32 and in TF32, so from bf16
//!   inputs the logits are exact products summed in f32 whether MLX runs f32 products in TF32
//!   (on the neural accelerators of GPUs that have them) or not; from f32 inputs the bounds
//!   assume `MLX_ENABLE_TF32=0`.
//! - The forward pass keeps per row the running maximum `m`, `s = sum exp(z - m)`, the entropy's
//!   `a = sum exp(z - m) (z - m)` and the target's logit, rescaled as the maximum grows: one
//!   matmul a chunk. Then `logprob = z_t - (m + ln s)` and `entropy = ln s - a / s`. The same
//!   pass keeps what [`crate::loss`] also reads: `s` without the target's term and the sum of the
//!   logits.
//! - The backward pass is a custom VJP. MLX does not tell a function whether its gradient will
//!   be asked for, so rather than accumulate the gradient's sums in every forward pass, as dash-ops
//!   does when its output is tracked, the backward pass walks the vocabulary again from the saved
//!   maximum `m` and sum `s`: `e = exp(z - m)`, `r = sum_{j != t} e_j` and
//!   `o = sum_{j != t} e_j weight[j]`, one more matmul a chunk, then
//!   `d hidden[i] = g[i] (r[i] weight[t[i]] - o[i]) / s[i]`, as dash-ops forms it, without the
//!   first term where the target is masked or out of range (the gradient of `-lse`). Dividing by
//!   `s` once, rather than exponentiating against `m + ln s`, keeps the rounding of the
//!   log-sum-exp out of every weight. The target's
//!   own term is left out of both sums, as dash-ops leaves it out, so that a target near certainty
//!   does not cancel. A pass without a gradient takes one matmul a chunk, one with it three, where
//!   dash-ops takes two.
//! - `weight` enters as a constant: no gradient reaches it through this operation, as dash-ops'
//!   contract has it frozen. The entropy carries no gradient.
//! - Masked tokens' logits are set to f32's lowest finite value, not `-inf`, so that their terms
//!   vanish without `0 * inf`.
//!
//! # The vocabulary chunk
//!
//! The chunk sets what is held at once and the order of the sums, not what is computed, and the
//! fastest chunk differs between devices. For `vocab_chunk` 0 dash-mlx walks chunks of 3072
//! tokens at any number of rows ([`vocab_chunk`]), the fastest and leanest of 2048 to 32768 on an
//! Apple M6; dash-ops takes 4096 tokens below 2304 rows and 8192 from there, its choice on CUDA.
//! At 3072 rows of 1024 against 248,320 tokens in bf16, with MLX's default TF32, over two runs
//! each (`examples/logprob_bench.rs` with `CHUNKS=3072,4096,8192`):
//!
//! | vocabulary chunk | forward | forward + backward |
//! |---|---|---|
//! | 3072, dash-mlx's | 127.6-131.0 ms, +1118 MiB | 434.9-439.1 ms, +1193 MiB |
//! | 4096 | 127.0-130.6 ms, +1165 MiB | 444.5-452.7 ms, +1304 MiB |
//! | 8192 | 127.9-130.1 ms, +1549 MiB | 585.2-620.1 ms, +1672 MiB |

use crate::kernel::{Launch, launch};
use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::TemplateArg;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{arange, concatenate, full, maximum, select, zeros};
use mlx_rs::transforms::custom_vjp;
use mlx_rs::{Array, Dtype};
use std::sync::atomic::Ordering;

/// Vocabulary tokens per chunk when [`LogprobOptions::vocab_chunk`] is 0: the fastest and leanest
/// of 2048 to 32768 at 3072 rows, D 1024 and 248,320 tokens on an Apple M6 (measured,
/// `examples/logprob_bench.rs` with `CHUNKS`).
const DEFAULT_VOCAB_CHUNK: usize = 3072;

pub use dash_contract::logprob::LogprobOptions;

/// The vocabulary chunk dash-mlx walks: `options.vocab_chunk`, or for 0 its own choice, 3072
/// tokens at any number of rows ([module documentation](self#the-vocabulary-chunk)).
pub fn vocab_chunk(options: LogprobOptions) -> usize {
    match options.vocab_chunk {
        0 => DEFAULT_VOCAB_CHUNK,
        chunk => chunk,
    }
}

/// What [`chosen_logprob`] returns, both f32 `[n]`.
#[derive(Debug, Clone)]
pub struct ChosenLogprob {
    /// `log p(target)` per row; `-inf` where the target is masked or out of range. Carries the
    /// gradient for `hidden`.
    pub logprob: Array,
    /// The entropy of each row's distribution over the unmasked vocabulary. No gradient.
    pub entropy: Array,
}

/// Counters tests read to tell which passes ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static FORWARD_PASSES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_PASSES: AtomicUsize = AtomicUsize::new(0);

    /// Passes over the vocabulary for the log-sum-exp, one per block of rows.
    pub fn forward_passes() -> usize {
        FORWARD_PASSES.load(Ordering::SeqCst)
    }

    /// Passes over the vocabulary for the gradient, one per block of rows.
    pub fn backward_passes() -> usize {
        BACKWARD_PASSES.load(Ordering::SeqCst)
    }
}

/// Threads per row in the chunk kernels.
const THREADS: i32 = 256;

const HEADER: &str = "
constant float DASH_LOWEST = -3.4028234663852886e38f;

// Two rows' running (maximum, sum exp(z - m), sum exp(z - m) (z - m), the sum without the
// target's term), the first updated to their union's.
inline void dash_combine(
    thread float& m1, thread float& s1, thread float& a1, thread float& r1,
    float m2, float s2, float a2, float r2
) {
    const float m = metal::max(m1, m2);
    const float f1 = metal::precise::exp(m1 - m);
    const float f2 = metal::precise::exp(m2 - m);
    a1 = f1 * (a1 + s1 * (m1 - m)) + f2 * (a2 + s2 * (m2 - m));
    s1 = s1 * f1 + s2 * f2;
    r1 = r1 * f1 + r2 * f2;
    m1 = m;
}
";

/// One row of a chunk's logits a threadgroup: each thread runs the online log-sum-exp, the
/// entropy's sum, the sum without the target's term and the sum of the logits over its columns,
/// the threads' parts are combined in a fixed order (lanes by shuffles, then simdgroups), and the
/// row's running state is updated with the result. The target's logit is added where the target
/// lies in the chunk (`tl >= 0`).
const STATE: &str = "
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const int C = z_shape[1];
    const int t = tl[row];
    const device float* zr = z + size_t(row) * C;
    float tm = DASH_LOWEST;
    float ts = 0.0f;
    float ta = 0.0f;
    float tr = 0.0f;
    float tz = 0.0f;
    for (int j = tid; j < C; j += TG) {
        if (MASKED && flags[j]) {
            continue;
        }
        const float v = zr[j];
        if (v > tm) {
            const float sc = metal::precise::exp(tm - v);
            ta = sc * (ta + ts * (tm - v));
            ts *= sc;
            tr *= sc;
            tm = v;
        }
        const float d = v - tm;
        const float e = metal::precise::exp(d);
        ts += e;
        ta += e * d;
        if (j != t) {
            tr += e;
        }
        tz += v;
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        const float om = simd_shuffle_down(tm, off);
        const float os = simd_shuffle_down(ts, off);
        const float oa = simd_shuffle_down(ta, off);
        const float o_r = simd_shuffle_down(tr, off);
        tz += simd_shuffle_down(tz, off);
        dash_combine(tm, ts, ta, tr, om, os, oa, o_r);
    }
    threadgroup float gm[TG / 32];
    threadgroup float gs[TG / 32];
    threadgroup float ga[TG / 32];
    threadgroup float gr[TG / 32];
    threadgroup float gz[TG / 32];
    if (thread_index_in_simdgroup == 0) {
        gm[simdgroup_index_in_threadgroup] = tm;
        gs[simdgroup_index_in_threadgroup] = ts;
        ga[simdgroup_index_in_threadgroup] = ta;
        gr[simdgroup_index_in_threadgroup] = tr;
        gz[simdgroup_index_in_threadgroup] = tz;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float rm = m[row];
        float rs = s[row];
        float ra = a[row];
        float rr = r[row];
        float cz = 0.0f;
        for (int g = 0; g < TG / 32; ++g) {
            dash_combine(rm, rs, ra, rr, gm[g], gs[g], ga[g], gr[g]);
            cz += gz[g];
        }
        m2[row] = rm;
        s2[row] = rs;
        a2[row] = ra;
        r2[row] = rr;
        zs2[row] = zs[row] + cz;
        zt2[row] = zt[row] + (t >= 0 ? zr[t] : 0.0f);
    }
";

/// One row of a chunk's logits a threadgroup: the backward pass's weights `exp(z - m)`, zero for
/// masked tokens and the target, and their sum over the row, in a fixed order.
const WEIGHTS: &str = "
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const int C = z_shape[1];
    const size_t base = size_t(row) * C;
    const float mr = m[row];
    const int t = tl[row];
    float sum = 0.0f;
    for (int j = tid; j < C; j += TG) {
        float e = metal::precise::exp(z[base + j] - mr);
        if ((MASKED && flags[j]) || j == t) {
            e = 0.0f;
        }
        e_out[base + j] = e;
        sum += e;
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        sum += simd_shuffle_down(sum, off);
    }
    threadgroup float gsum[TG / 32];
    if (thread_index_in_simdgroup == 0) {
        gsum[simdgroup_index_in_threadgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (int g = 0; g < TG / 32; ++g) {
            total += gsum[g];
        }
        r_part[row] = total;
    }
";

/// What a pass over the vocabulary leaves per row, each `[rows]` f32: dash-ops' rows' state.
pub(crate) struct State {
    /// The largest unmasked logit (f32's lowest before any).
    pub(crate) m: Array,
    /// `sum_j exp(z - m)` over the unmasked tokens.
    pub(crate) s: Array,
    /// The same sum without the target's term: `s` where the target is masked or out of range.
    pub(crate) r: Array,
    /// `sum_j exp(z - m) (z - m)`, for the entropy.
    pub(crate) a: Array,
    /// The target's logit, 0 if it is masked or out of range.
    pub(crate) zt: Array,
    /// `sum_j z` over the unmasked tokens.
    pub(crate) zsum: Array,
}

/// What the passes over the vocabulary share, this module's and [`crate::loss`]'s: the weight,
/// the targets, the masked tokens and the chunking.
#[derive(Clone)]
pub(crate) struct Vocabulary {
    /// `[V, D]`, as given.
    pub(crate) weight: Array,
    /// `[n]` int32, clamped into `0..V`: where a valid target sits.
    pub(crate) targets: Array,
    /// `[n]` bool: the target is in `0..V` and not masked.
    pub(crate) valid: Array,
    /// The masked ids, sorted, and `[V]` bool flags for them, if any.
    pub(crate) masked: Vec<u32>,
    flags: Option<Array>,
    pub(crate) vocab: usize,
    chunk: usize,
    /// Each block of rows, `start..end`.
    pub(crate) blocks: Vec<(usize, usize)>,
    /// The chunk kernels, or MLX's operations (the reference).
    pub(crate) kernels: bool,
}

impl Vocabulary {
    /// The vocabulary of `weight` for `hidden`'s rows and `targets`, after the checks both
    /// operations make, their errors prefixed with `op`: shapes, dtypes, the masked ids (sorted
    /// here), a zero `row_chunk`.
    pub(crate) fn new(
        op: &str,
        hidden: &Array,
        weight: &Array,
        targets: &Array,
        masked: &[u32],
        options: LogprobOptions,
        kernels: bool,
    ) -> Result<Self> {
        let contract = |message: String| Exception::custom(format!("{op}: {message}"));
        let (n, d) = match hidden.shape() {
            [n, d] => (*n as usize, *d),
            shape => return Err(contract(format!("hidden must be [n, D], got {shape:?}"))),
        };
        let vocab = match weight.shape() {
            [v, dw] if *dw == d => *v as usize,
            shape => {
                return Err(contract(format!(
                    "weight must be [V, D] with D = {d}, got {shape:?}"
                )));
            }
        };
        if targets.shape() != [n as i32] {
            return Err(contract(format!(
                "targets must be [n] with n = {n}, got {:?}",
                targets.shape()
            )));
        }
        let integer = [
            Dtype::Int8,
            Dtype::Int16,
            Dtype::Int32,
            Dtype::Int64,
            Dtype::Uint8,
            Dtype::Uint16,
            Dtype::Uint32,
            Dtype::Uint64,
        ];
        if !integer.contains(&targets.dtype()) {
            return Err(contract(format!(
                "targets must be integers, got {:?}",
                targets.dtype()
            )));
        }
        for (name, a) in [("hidden", hidden), ("weight", weight)] {
            if !a.dtype().is_float() {
                return Err(contract(format!(
                    "{name} must be floating point, got {:?}",
                    a.dtype()
                )));
            }
        }
        if options.row_chunk == 0 {
            return Err(contract(format!(
                "row_chunk must be at least 1: {options:?}"
            )));
        }
        let mut masked = masked.to_vec();
        masked.sort_unstable();
        masked.dedup();
        if let Some(&last) = masked.last() {
            if last as usize >= vocab {
                return Err(contract(format!(
                    "masked token id {last} is outside the vocabulary of {vocab}"
                )));
            }
            if masked.len() >= vocab {
                return Err(contract("every token of the vocabulary is masked".into()));
            }
        }

        // The targets in range, and those masked.
        let t = targets.as_dtype(Dtype::Int64)?;
        let mut valid = t
            .ge(Array::from_int(0))?
            .logical_and(t.lt(Array::from_slice(&[vocab as i64], &[]))?)?;
        let t = select(&valid, &t, Array::from_int(0))?.as_dtype(Dtype::Int32)?;
        let flags = match masked.is_empty() {
            true => None,
            false => {
                let mut flags = vec![false; vocab];
                for &id in &masked {
                    flags[id as usize] = true;
                }
                let flags = Array::from_slice(&flags, &[vocab as i32]);
                valid = valid.logical_and(flags.take_axis(&t, 0)?.logical_not()?)?;
                Some(flags)
            }
        };
        let block = n.div_ceil(n.div_ceil(options.row_chunk).max(1)).max(1);
        Ok(Self {
            weight: weight.clone(),
            targets: t,
            valid,
            masked,
            flags,
            vocab,
            chunk: vocab_chunk(options),
            blocks: (0..n)
                .step_by(block)
                .map(|r0| (r0, (r0 + block).min(n)))
                .collect(),
            kernels,
        })
    }

    /// The same vocabulary over `weight`, an array of the same shape: the one a transform hands
    /// over in its place.
    pub(crate) fn with_weight(&self, weight: &Array) -> Self {
        Self {
            weight: weight.clone(),
            ..self.clone()
        }
    }

    /// The chunks of the vocabulary, `start..end`.
    pub(crate) fn chunks(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        (0..self.vocab)
            .step_by(self.chunk)
            .map(|c0| (c0 as i32, (c0 + self.chunk).min(self.vocab) as i32))
    }

    /// The masked tokens' flags in the chunk `c0..c1`, if it holds any.
    pub(crate) fn chunk_flags(&self, (c0, c1): (i32, i32)) -> Result<Option<Array>> {
        let hit = self.masked.partition_point(|&id| (id as i32) < c0)
            < self.masked.partition_point(|&id| (id as i32) < c1);
        match (&self.flags, hit) {
            (Some(flags), true) => Ok(Some(flags.try_index(c0..c1)?)),
            _ => Ok(None),
        }
    }

    /// The logits of rows `h` `[R, D]` f32 against the chunk `c0..c1` (its weight `w`, f32),
    /// f32. On the reference path masked tokens are set to f32's lowest value; the kernels skip
    /// them themselves.
    pub(crate) fn logits(&self, h: &Array, w: &Array, chunk: (i32, i32)) -> Result<Array> {
        let z = h.matmul(w.t())?;
        match (self.kernels, self.chunk_flags(chunk)?) {
            (false, Some(flags)) => select(&flags.expand_dims(0)?, Array::from_f32(f32::MIN), &z),
            _ => Ok(z),
        }
    }

    /// Where each row's valid target lies in the chunk `c0..c1`, or -1: `[R]` int32.
    pub(crate) fn local_targets(t: &Array, valid: &Array, (c0, c1): (i32, i32)) -> Result<Array> {
        let inside = t
            .ge(Array::from_int(c0))?
            .logical_and(t.lt(Array::from_int(c1))?)?
            .logical_and(valid)?;
        select(
            &inside,
            t.subtract(Array::from_int(c0))?,
            Array::from_int(-1),
        )
    }

    /// The flags a chunk kernel reads, and whether to read them.
    pub(crate) fn kernel_flags(&self, chunk: (i32, i32)) -> Result<(Array, bool)> {
        Ok(match self.chunk_flags(chunk)? {
            Some(flags) => (flags, true),
            None => (Array::from_slice(&[false], &[1]), false),
        })
    }

    /// The chunk kernel's update of the rows' running state from the chunk's logits `z`.
    fn state_kernel(&self, z: &Array, st: State, tl: &Array, chunk: (i32, i32)) -> Result<State> {
        let rows = z.shape()[0];
        let (flags, masked) = self.kernel_flags(chunk)?;
        let kernel = crate::kernel::kernel(
            "dash_mlx_logprob_state",
            &["z", "m", "s", "a", "zt", "r", "zs", "tl", "flags"],
            &["m2", "s2", "a2", "zt2", "r2", "zs2"],
            STATE,
            HEADER,
        )?;
        let config = Launch {
            grid: (rows * THREADS, 1, 1),
            thread_group: (THREADS, 1, 1),
            outputs: vec![(vec![rows], Dtype::Float32); 6],
            template: vec![
                ("TG".into(), TemplateArg::Int(THREADS)),
                ("MASKED".into(), TemplateArg::Bool(masked)),
            ],
            init_value: None,
        };
        let inputs = [
            z.clone(),
            st.m,
            st.s,
            st.a,
            st.zt,
            st.r,
            st.zsum,
            tl.clone(),
            flags,
        ];
        // SAFETY: one threadgroup per row of `z` ([rows, C], row-contiguous), each reading only
        // that row's C logits, `flags[0..C]` when MASKED (the chunk's flags, C long), its row of
        // m, s, a, zt, r, zs and tl, and `z[row, tl]` only where `0 <= tl < C`; thread 0 writes
        // the row's six outputs, each written once. No input is written.
        let out = unsafe { launch(&kernel, &inputs, &config)? };
        let [m, s, a, zt, r, zsum]: [Array; 6] = out.try_into().expect("six outputs");
        Ok(State {
            m,
            s,
            r,
            a,
            zt,
            zsum,
        })
    }

    /// The chunk kernel's backward weights `exp(z - m)` (zero for masked tokens and the target)
    /// and their sums over each row.
    fn weights_kernel(
        &self,
        z: &Array,
        m: &Array,
        tl: &Array,
        chunk: (i32, i32),
    ) -> Result<[Array; 2]> {
        let rows = z.shape()[0];
        let (flags, masked) = self.kernel_flags(chunk)?;
        let kernel = crate::kernel::kernel(
            "dash_mlx_logprob_weights",
            &["z", "m", "tl", "flags"],
            &["e_out", "r_part"],
            WEIGHTS,
            HEADER,
        )?;
        let config = Launch {
            grid: (rows * THREADS, 1, 1),
            thread_group: (THREADS, 1, 1),
            outputs: vec![
                (z.shape().to_vec(), Dtype::Float32),
                (vec![rows], Dtype::Float32),
            ],
            template: vec![
                ("TG".into(), TemplateArg::Int(THREADS)),
                ("MASKED".into(), TemplateArg::Bool(masked)),
            ],
            init_value: None,
        };
        let inputs = [z.clone(), m.clone(), tl.clone(), flags];
        // SAFETY: one threadgroup per row of `z` ([rows, C], row-contiguous): each thread reads
        // and writes columns `tid, tid + TG, ...` below C of that row in `z` and `e_out` (shaped as
        // `z`), `flags[0..C]` when MASKED, and its row of m and tl; thread 0 writes `r_part[row]`.
        // Every element of both outputs is written once; no input is written.
        let mut out = unsafe { launch(&kernel, &inputs, &config)? };
        let r = out.pop().expect("two outputs");
        let e = out.pop().expect("two outputs");
        Ok([e, r])
    }

    /// The pass over the vocabulary for rows `r0..r1` of `h` (f32): each row's state, as
    /// dash-ops' passes leave it.
    pub(crate) fn pass(&self, h: &Array, (r0, r1): (usize, usize)) -> Result<State> {
        probe::FORWARD_PASSES.fetch_add(1, Ordering::SeqCst);
        let rows = (r1 - r0) as i32;
        let h = h.try_index(r0 as i32..r1 as i32)?;
        let t = self.targets.try_index(r0 as i32..r1 as i32)?;
        let valid = self.valid.try_index(r0 as i32..r1 as i32)?;
        let mut st = State {
            m: full::<f32>(&[rows], Array::from_f32(f32::MIN))?,
            s: zeros::<f32>(&[rows])?,
            r: zeros::<f32>(&[rows])?,
            a: zeros::<f32>(&[rows])?,
            zt: zeros::<f32>(&[rows])?,
            zsum: zeros::<f32>(&[rows])?,
        };
        for (c0, c1) in self.chunks() {
            let w = self.weight.try_index(c0..c1)?.as_dtype(Dtype::Float32)?;
            let z = self.logits(&h, &w, (c0, c1))?;
            let tl = Self::local_targets(&t, &valid, (c0, c1))?;
            if self.kernels {
                st = self.state_kernel(&z, st, &tl, (c0, c1))?;
                continue;
            }
            let State {
                m,
                s,
                r,
                a,
                zt,
                zsum,
            } = st;
            // The target's logit, from the chunk that holds it.
            let picked = z
                .take_along_axis(maximum(&tl, Array::from_int(0))?.expand_dims(-1)?, -1)?
                .squeeze_axes(&[-1])?;
            let zt = zt.add(select(
                tl.ge(Array::from_int(0))?,
                &picked,
                Array::from_f32(0.0),
            )?)?;
            let unmasked = match self.chunk_flags((c0, c1))? {
                Some(flags) => select(&flags.expand_dims(0)?, Array::from_f32(0.0), &z)?,
                None => z.clone(),
            };
            let zsum = zsum.add(unmasked.sum_axis(-1, None)?)?;
            // The online log-sum-exp, the entropy's sum and the sum without the target's term,
            // rescaled to the new maximum.
            let new_m = maximum(&m, z.max_axis(-1, None)?)?;
            let scale = m.subtract(&new_m)?.exp()?;
            let d = z.subtract(new_m.expand_dims(-1)?)?;
            let e = d.exp()?;
            let columns = arange::<_, i32>(0, c1 - c0, None)?.expand_dims(0)?;
            let others = select(columns.eq(tl.expand_dims(-1)?)?, Array::from_f32(0.0), &e)?;
            st = State {
                a: scale
                    .multiply(a.add(s.multiply(m.subtract(&new_m)?)?)?)?
                    .add(e.multiply(&d)?.sum_axis(-1, None)?)?,
                s: s.multiply(&scale)?.add(e.sum_axis(-1, None)?)?,
                r: r.multiply(&scale)?.add(others.sum_axis(-1, None)?)?,
                m: new_m,
                zt,
                zsum,
            };
        }
        Ok(st)
    }

    /// The forward pass over rows `r0..r1` of `h` (f32): the log-probability, the entropy, and the
    /// final maximum and sum the backward pass starts from.
    fn forward(&self, h: &Array, (r0, r1): (usize, usize)) -> Result<[Array; 4]> {
        let State { m, s, a, zt, .. } = self.pass(h, (r0, r1))?;
        let valid = self.valid.try_index(r0 as i32..r1 as i32)?;
        let ln_s = s.log()?;
        let lse = m.add(&ln_s)?;
        let logprob = select(
            &valid,
            zt.subtract(&lse)?,
            Array::from_f32(f32::NEG_INFINITY),
        )?;
        let entropy = ln_s.subtract(a.divide(&s)?)?;
        Ok([logprob, entropy, m, s])
    }

    /// The backward pass over rows `r0..r1`: `d logprob[i] / d hidden[i]` in f32. Where the
    /// target is masked or out of range, as in dash-ops, only `-o`, the gradient of `-lse`.
    fn gradient(&self, h: &Array, [m, s]: [&Array; 2], (r0, r1): (usize, usize)) -> Result<Array> {
        probe::BACKWARD_PASSES.fetch_add(1, Ordering::SeqCst);
        let rows = (r1 - r0) as i32;
        let d = h.shape()[1];
        let h = h.try_index(r0 as i32..r1 as i32)?;
        let t = self.targets.try_index(r0 as i32..r1 as i32)?;
        let valid = self.valid.try_index(r0 as i32..r1 as i32)?;
        let m = m.try_index(r0 as i32..r1 as i32)?.expand_dims(-1)?;
        let s = s.try_index(r0 as i32..r1 as i32)?.expand_dims(-1)?;
        // Only a valid target is left out of the sums; -1 matches no token.
        let target_col = select(&valid, &t, Array::from_int(-1))?.expand_dims(-1)?;
        let mut r = zeros::<f32>(&[rows])?;
        let mut o = zeros::<f32>(&[rows, d])?;
        let m_row = m.squeeze_axes(&[-1])?;
        for (c0, c1) in self.chunks() {
            let w = self.weight.try_index(c0..c1)?.as_dtype(Dtype::Float32)?;
            let z = self.logits(&h, &w, (c0, c1))?;
            if self.kernels {
                let tl = Self::local_targets(&t, &valid, (c0, c1))?;
                let [e, r_part] = self.weights_kernel(&z, &m_row, &tl, (c0, c1))?;
                r = r.add(r_part)?;
                o = o.add(e.matmul(&w)?)?;
                continue;
            }
            let p = z.subtract(&m)?.exp()?;
            // The target's own term left out.
            let ids = arange::<_, i32>(c0, c1, None)?.expand_dims(0)?;
            let p = select(ids.eq(&target_col)?, Array::from_f32(0.0), &p)?;
            r = r.add(p.sum_axis(-1, None)?)?;
            o = o.add(p.matmul(&w)?)?;
        }
        let wt = self.weight.take_axis(&t, 0)?.as_dtype(Dtype::Float32)?;
        let r = select(&valid, &r, Array::from_f32(0.0))?;
        r.expand_dims(-1)?.multiply(&wt)?.subtract(&o)?.divide(&s)
    }
}

/// The log-probability of `targets` under `softmax(hidden @ weight^T)` with the `masked` token
/// ids removed, and the entropy of that distribution, without holding the n x V logits; dash-ops'
/// arguments and results ([module](self)).
///
/// - `hidden` `[n, D]`, f32 or bf16 (or f16): gets the gradient, in its dtype.
/// - `weight` `[V, D]`, frozen: no gradient reaches it through this operation.
/// - `targets` `[n]`, any integer dtype.
/// - `masked`: token ids excluded from the softmax, in any order; each must be `< V`, and at
///   least one token must stay unmasked.
///
/// Errors on mismatched shapes, a masked id `>= V`, every token masked, or a zero `row_chunk`.
pub fn chosen_logprob(
    hidden: &Array,
    weight: &Array,
    targets: &Array,
    masked: &[u32],
    options: LogprobOptions,
) -> Result<ChosenLogprob> {
    checked(hidden, weight, targets, masked, options, true)
}

/// [`chosen_logprob`] with each chunk's update as MLX's operations rather than the chunk
/// kernels: the reference the tests hold the kernels to.
#[doc(hidden)]
pub fn chosen_logprob_reference(
    hidden: &Array,
    weight: &Array,
    targets: &Array,
    masked: &[u32],
    options: LogprobOptions,
) -> Result<ChosenLogprob> {
    checked(hidden, weight, targets, masked, options, false)
}

fn checked(
    hidden: &Array,
    weight: &Array,
    targets: &Array,
    masked: &[u32],
    options: LogprobOptions,
    kernels: bool,
) -> Result<ChosenLogprob> {
    let vocabulary = Vocabulary::new(
        "chosen_logprob",
        hidden,
        weight,
        targets,
        masked,
        options,
        kernels,
    )?;
    if hidden.shape()[0] == 0 {
        return Ok(ChosenLogprob {
            logprob: zeros::<f32>(&[0])?,
            entropy: zeros::<f32>(&[0])?,
        });
    }

    let forward = vocabulary.clone();
    let backward = vocabulary;
    let op = custom_vjp(
        move |p: &[Array]| {
            let h = p[0].as_dtype(Dtype::Float32)?;
            let mut parts: [Vec<Array>; 4] = Default::default();
            for &rows in &forward.blocks {
                for (part, x) in parts.iter_mut().zip(forward.forward(&h, rows)?) {
                    part.push(x);
                }
            }
            parts.iter().map(|xs| concatenate(xs, 0)).collect()
        },
        move |p: &[Array], cotangents: &[Array], outputs: &[Array]| {
            let h = p[0].as_dtype(Dtype::Float32)?;
            let grads = backward
                .blocks
                .iter()
                .map(|&rows| backward.gradient(&h, [&outputs[2], &outputs[3]], rows))
                .collect::<Result<Vec<_>>>()?;
            let grad = concatenate(&grads, 0)?;
            let g = cotangents[0].as_dtype(Dtype::Float32)?.expand_dims(-1)?;
            Ok(vec![grad.multiply(&g)?.as_dtype(p[0].dtype())?])
        },
    )?;
    let mut outputs = op.apply(std::slice::from_ref(hidden))?;
    outputs.truncate(2);
    let entropy = outputs.pop().expect("two outputs");
    let logprob = outputs.pop().expect("two outputs");
    Ok(ChosenLogprob { logprob, entropy })
}
