//! Sampling the next token on the device, for a decoder's step: dash-ops' `sample` on MLX. From
//! the logits `[B, V]` of one position per sequence, one token id per row, with its
//! log-probability under the model's distribution, under the tempered distribution, and under the
//! distribution it was drawn from; temperature, top-k, top-p and masked ids as the options say;
//! greedy decoding at temperature 0.
//!
//! The definition (the cuts and their ties, the Gumbel-max draw, the log-probabilities, the rows
//! with no finite logit) and the random numbers (Philox4x32-10 keyed by the seed and counted by
//! the token, the row and the offset, and the Gumbel variate formed from each word without
//! `log1p`) are dash-ops' (its `sample` module documents both), so that the same seed, offset and
//! logits draw the same tokens on MLX as on dash-ops' devices. Beside dash-ops' `logprob` and
//! `sampling_logprob` the sampler returns `tempered_logprob`, the log-softmax of the tempered
//! logits over the unmasked ids at the token, without top-k or top-p: the value TRL's GRPO trainer
//! and verl take for the importance ratio. At temperature 1 with nothing cut the three are equal,
//! to the bit.
//!
//! On MLX:
//!
//! - One Metal kernel draws the Gumbel variates `[B, V]`: a thread per four tokens computes
//!   Philox's ten rounds for its counter and the four variates from its words, in the module
//!   documentation's f32 operations (the logarithms `metal::precise::log`).
//! - The rest is MLX's operations, in f32: the row's maximum and the sums of `exp(s)`; the cuts
//!   from the row sorted in descending order, as dash-ops' reference takes them (the `k`-th
//!   largest logit, and top-p's logit where the cumulative tempered mass first reaches `p` of the
//!   total); the draw as the first largest of `s + g` over the kept tokens (MLX's argmax takes the
//!   lowest id of a tie).
//! - Top-p's cumulative tempered mass is a balanced tree, Hillis and Steele's scan, as dash-ops'
//!   reference forms it. The kept set is the whole tie group at the logit where that mass first
//!   reaches `p` of the total, so where the crossing lies within the sums' rounding of a group's
//!   end, a cut in f32 may keep the next group or stop short of it. Measured against the cut in
//!   f64 on 163,840 cuts (`examples/sample_cuts.rs`: 40,960 rows of 248,320 bf16 logits, flat and
//!   peaked, at `p` 0.9 and 0.95), the tree's cut differs on 28 (1.71e-4 of them), a tree formed in
//!   pairs (Blelloch's) on 49 (3.00e-4) and MLX's scan on 75 (4.58e-4), every one where the f64
//!   crossing lies within 2e-6 of the row's mass of a group's end (1,946 of the cuts, 1.19%), none
//!   elsewhere. At a rollout's temperature, 0.7, which sharpens the weights, more crossings lie
//!   near a group's end (2,888 of the 163,840 cuts, 1.77%), and the tree's cut differs on 39
//!   (2.39e-4), the pairs' on 81 (4.95e-4) and the scan's on 150 (9.16e-4), again none elsewhere.
//!   The tree costs top-p 18 passes over the row: 35.8 to 35.9 ms at the bench's shape,
//!   against 22.6 for the pairs and 17.1 for the scan (measured). A comparison of tokens across
//!   devices marks the rows near a group's end and holds every other row to the same token.
//! - Greedy decoding is MLX's argmax of the row, the masked logits at `-inf`, and the token's
//!   log-softmax through MLX's fused log-sum-exp.
//!
//! The tokens are `[B]` int32. Nothing is differentiable.

use crate::kernel::{Launch, launch};
use mlx_rs::error::{Exception, Result};
use mlx_rs::ops::indexing::{TryIndexOp, argmax_axis};
use mlx_rs::ops::{concatenate, cumsum, full, select, sort_axis, sum_axis, zeros};
use mlx_rs::{Array, Dtype};

pub use dash_contract::sample::SampleOptions;

/// What [`sample`] returns, each `[B]`.
#[derive(Debug, Clone)]
pub struct Sampled {
    /// The token drawn for each row, int32.
    pub tokens: Array,
    /// Its log-probability under the model's distribution: the log-softmax of the raw logits over
    /// the unmasked vocabulary, at temperature 1. f32.
    pub logprob: Array,
    /// Its log-probability under the tempered distribution: the log-softmax of the logits divided
    /// by the temperature over the unmasked vocabulary, nothing cut; 0 when greedy. f32.
    pub tempered_logprob: Array,
    /// Its log-probability under the distribution it was drawn from: the softmax of the tempered
    /// logits over the tokens top-k and top-p keep; 0 when greedy. f32.
    pub sampling_logprob: Array,
}

const HEADER: &str = "
// Philox4x32-10 (Salmon et al. 2011): ten rounds, the key bumped by the Weyl constants before each
// but the first.
inline uint4 dash_philox(uint4 c, uint2 k) {
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k += uint2(0x9E3779B9u, 0xBB67AE85u);
        }
        const uint hi0 = metal::mulhi(0xD2511F53u, c.x);
        const uint lo0 = 0xD2511F53u * c.x;
        const uint hi1 = metal::mulhi(0xCD9E8D57u, c.z);
        const uint lo1 = 0xCD9E8D57u * c.z;
        c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    }
    return c;
}

// The Gumbel variate of the word x, in dash-ops' operations: a = (2y + 1) / 2^33 rounded to f32,
// y the word or its complement; E = -ln a below 2^31, else -ln_1p(-a) in the Pade form up to
// 2^-6 and Goldberg's above; g = -ln E.
inline float dash_gumbel(uint x) {
    const bool upper = (x >> 31) == 1u;
    const uint y = upper ? ~x : x;
    const float a = float((y << 1) | 1u) * 0x1p-33f;
    float e;
    if (!upper) {
        e = -metal::precise::log(a);
    } else if (a <= 0x1p-6f) {
        e = a * ((-a + 6.0f) / (a * -4.0f + 6.0f));
    } else {
        const float w = -a + 1.0f;
        e = -metal::precise::log(w) * (a / (-w + 1.0f));
    }
    return -metal::precise::log(e);
}
";

/// The Gumbel variates of a `[B, V]` sample: a thread per four tokens of a row, Philox's words for
/// the counter `(i / 4, row, offset)` under the key `seed`, as `params` (`[seed lo, seed hi,
/// offset lo, offset hi, V]`) give them.
const NOISE: &str = "
    const uint group = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const uint V = params[4];
    if (size_t(group) * 4 >= V) {
        return;
    }
    const uint4 words =
        dash_philox(uint4(group, row, params[2], params[3]), uint2(params[0], params[1]));
    const size_t base = size_t(row) * V + size_t(group) * 4;
    for (uint j = 0; j < 4 && group * 4 + j < V; ++j) {
        g[base + j] = WORDS ? as_type<float>(words[j]) : dash_gumbel(words[j]);
    }
";

/// The Gumbel variate of each of `words`.
const VARIATES: &str = "
    const uint i = thread_position_in_grid.x;
    if (i < words_shape[0]) {
        g[i] = dash_gumbel(words[i]);
    }
";

fn contract(message: String) -> Exception {
    Exception::custom(format!("sample: {message}"))
}

/// The noise kernel over `[b, v]`: the Gumbel variates, or with `words` Philox's words themselves
/// (their bits as f32).
fn noise(seed: u64, offset: u64, [b, v]: [i32; 2], words: bool) -> Result<Array> {
    let kernel =
        crate::kernel::kernel("dash_mlx_sample_noise", &["params"], &["g"], NOISE, HEADER)?;
    let groups = (v as usize).div_ceil(4) as i32;
    let config = Launch {
        grid: ((groups + 255) / 256 * 256, b, 1),
        thread_group: (256, 1, 1),
        outputs: vec![(vec![b, v], Dtype::Float32)],
        template: vec![("WORDS".into(), mlx_rs::fast::TemplateArg::Bool(words))],
        init_value: None,
    };
    let params = Array::from_slice(
        &[
            seed as u32,
            (seed >> 32) as u32,
            offset as u32,
            (offset >> 32) as u32,
            v as u32,
        ],
        &[5],
    );
    // SAFETY: thread (group, row) of a grid covering every row of [b, v] and every group of four
    // tokens (those at `4 group >= V` returning at once) reads `params[0..5]` and writes the
    // elements `row * V + 4 group + j`, `j < 4` and below V, of `g` [b, V] once each; every
    // element of `g` is written by one thread. No input is written.
    let out = unsafe { launch(&kernel, &[params], &config)? };
    Ok(out[0].clone())
}

/// The Gumbel variates the noise kernel forms from `words`, f32: its own function, run alone. For
/// tests.
#[doc(hidden)]
pub fn gumbel_variates(words: &[u32]) -> Result<Vec<f32>> {
    let kernel = crate::kernel::kernel(
        "dash_mlx_sample_variates",
        &["words"],
        &["g"],
        VARIATES,
        HEADER,
    )?;
    let n = words.len() as i32;
    let config = Launch {
        grid: ((n + 255) / 256 * 256, 1, 1),
        thread_group: (256, 1, 1),
        outputs: vec![(vec![n], Dtype::Float32)],
        template: vec![],
        init_value: None,
    };
    let words = Array::from_slice(words, &[n]);
    // SAFETY: thread i of a grid covering every word (those at `i >= n` returning at once) reads
    // `words[i]` and writes `g[i]` once. No input is written.
    let out = unsafe { launch(&kernel, &[words], &config)? };
    Ok(out[0].as_slice::<f32>().to_vec())
}

/// Philox's words for a `[b, v]` sample on the device, token by token: what the noise kernel
/// draws its variates from. For tests.
#[doc(hidden)]
pub fn philox_words(seed: u64, offset: u64, dims: [i32; 2]) -> Result<Vec<u32>> {
    let words = noise(seed, offset, dims, true)?;
    Ok(words
        .as_slice::<f32>()
        .iter()
        .map(|w| w.to_bits())
        .collect())
}

/// How top-p's cumulative tempered mass is summed.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutSum {
    /// MLX's scan.
    Scan,
    /// A balanced tree, Hillis and Steele's scan, as dash-ops' reference forms it.
    Tree,
    /// A balanced tree formed in pairs, Blelloch's up-sweep and down-sweep.
    Pairs,
}

/// The inclusive cumulative sums of each row of `x` `[B, V]` from a balanced tree of pair sums:
/// the row padded with zeros to a power of two, summed in pairs level by level, and each prefix
/// the sum of its ancestors' left siblings plus its own entry, so that each is a sum of at most
/// `log2 V` of the tree's partial sums, over arrays that halve at each level.
fn pairs_cumsum(x: &Array) -> Result<Array> {
    let (b, v) = (x.shape()[0], x.shape()[1]);
    let n = (v as u32).next_power_of_two() as i32;
    let padded = match n > v {
        true => concatenate(&[x.clone(), zeros::<f32>(&[b, n - v])?], 1)?,
        false => x.clone(),
    };
    let mut levels = vec![padded];
    while levels.last().expect("a level").shape()[1] > 1 {
        let top = levels.last().expect("a level");
        let len = top.shape()[1];
        levels.push(sum_axis(&top.reshape(&[b, len / 2, 2])?, 2, None)?);
    }
    // The exclusive prefix of each level from its parent's: a left child takes its parent's, a
    // right child its parent's plus its left sibling.
    let mut prefix = zeros::<f32>(&[b, 1])?;
    for level in levels[..levels.len() - 1].iter().rev() {
        let len = level.shape()[1];
        let left = level.reshape(&[b, len / 2, 2])?.try_index((.., .., 0))?;
        let right = prefix.add(&left)?;
        prefix = mlx_rs::ops::stack(&[prefix, right], 2)?.reshape(&[b, len])?;
    }
    prefix.add(&levels[0])?.try_index((.., 0..v))
}

/// The sum the sampler's top-p cut takes.
const CUT_SUM: CutSum = CutSum::Tree;

/// The inclusive cumulative sums of each row of `x` `[B, V]`: `ceil(log2 V)` steps, each adding
/// to every entry the one `2^k` before it, so that each sum is a balanced tree of that depth.
fn tree_cumsum(x: &Array) -> Result<Array> {
    let (b, v) = (x.shape()[0], x.shape()[1]);
    let mut x = x.clone();
    let mut shift = 1;
    while shift < v {
        let earlier = x.try_index((.., 0..v - shift))?;
        x = x.add(concatenate(&[zeros::<f32>(&[b, shift])?, earlier], 1)?)?;
        shift *= 2;
    }
    Ok(x)
}

/// The cut: the smallest logit kept, `[B, 1]`, or `None` when neither top-k nor top-p cuts. From
/// the row `zm` (the masked logits at -inf) sorted in descending order, as dash-ops' reference
/// takes it: the `k`-th largest logit, then top-p's logit where the cumulative tempered mass of
/// what top-k keeps first reaches `p` of its total. `m` is the row's maximum `[B, 1]`.
fn cut(
    zm: &Array,
    m: &Array,
    tau: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    sum: CutSum,
) -> Result<Option<Array>> {
    if top_k.is_none() && top_p.is_none() {
        return Ok(None);
    }
    let v = zm.shape()[1];
    let sorted = sort_axis(zm.negative()?, 1)?.negative()?;
    let mut threshold = top_k
        .map(|k| sorted.try_index((.., k as i32 - 1..k as i32)))
        .transpose()?;
    if let Some(p) = top_p {
        let mut weights = sorted.subtract(m)?.multiply(Array::from_f32(tau))?.exp()?;
        if let Some(kth) = &threshold {
            weights = select(sorted.lt(kth)?, Array::from_f32(0.0), weights)?;
        }
        let cumulative = match sum {
            CutSum::Scan => cumsum(&weights, 1, None, None)?,
            CutSum::Tree => tree_cumsum(&weights)?,
            CutSum::Pairs => pairs_cumsum(&weights)?,
        };
        let total = cumulative.try_index((.., v - 1..v))?;
        let target = total.multiply(Array::from_f32(p))?;
        // The first entry that reaches the target (the last one does).
        let reached = cumulative.ge(&target)?.as_dtype(Dtype::Float32)?;
        let first = argmax_axis(&reached, 1, true)?;
        threshold = Some(sorted.take_along_axis(&first, 1)?);
    }
    Ok(threshold)
}

/// The cut [`sample`] keeps `z >= cut` at, `[B]` f32 (`-inf` where nothing is cut), with top-p's
/// mass summed as `sum` says. No masked ids. For tests and measurements.
#[doc(hidden)]
pub fn top_p_cut(logits: &Array, options: &SampleOptions, sum: CutSum) -> Result<Array> {
    let b = logits.shape()[0];
    let z = logits.as_dtype(Dtype::Float32)?;
    let m = z.max_axis(1, true)?;
    let tau = (1.0 / options.temperature).min(f32::MAX);
    let top_k =
        (options.top_k > 0 && options.top_k < z.shape()[1] as usize).then_some(options.top_k);
    let top_p = (options.top_p < 1.0).then_some(options.top_p);
    match cut(&z, &m, tau, top_k, top_p, sum)? {
        Some(t) => t.reshape(&[b]),
        None => full::<f32>(&[b], Array::from_f32(f32::NEG_INFINITY)),
    }
}

/// Draws one token per row of `logits` `[B, V]` (f32, bf16 or f16), as `options` say; dash-ops'
/// definition and random numbers ([module](self)). No gradient flows through it.
///
/// Errors if `logits` is not 2-D or `V` is 0, if the temperature is negative or not finite, if
/// `top_p` is not within `0..=1`, if a masked id is `>= V`, or if every token is masked: where
/// dash-ops panics.
pub fn sample(logits: &Array, options: &SampleOptions) -> Result<Sampled> {
    let (b, v) = match logits.shape() {
        [b, v] if *v > 0 => (*b, *v),
        shape => {
            return Err(contract(format!(
                "logits must be [B, V] with V > 0, got {shape:?}"
            )));
        }
    };
    if !logits.dtype().is_float() {
        return Err(contract(format!(
            "logits must be floating point, got {:?}",
            logits.dtype()
        )));
    }
    let temperature = options.temperature;
    if !(temperature.is_finite() && temperature >= 0.0) {
        return Err(contract(format!(
            "the temperature must be finite and >= 0, got {temperature}"
        )));
    }
    if !(0.0..=1.0).contains(&options.top_p) {
        return Err(contract(format!(
            "top_p must be within 0..=1, got {}",
            options.top_p
        )));
    }
    let mut masked = options.masked.clone();
    masked.sort_unstable();
    masked.dedup();
    if let Some(&last) = masked.last() {
        if last as i64 >= v as i64 {
            return Err(contract(format!(
                "masked token id {last} is outside the vocabulary of {v}"
            )));
        }
        if masked.len() >= v as usize {
            return Err(contract("every token of the vocabulary is masked".into()));
        }
    }
    let empty = || zeros::<f32>(&[0]);
    if b == 0 {
        return Ok(Sampled {
            tokens: zeros::<i32>(&[0])?,
            logprob: empty()?,
            tempered_logprob: empty()?,
            sampling_logprob: empty()?,
        });
    }

    let inf = || Array::from_f32(f32::NEG_INFINITY);
    let z = logits.as_dtype(Dtype::Float32)?;
    // `[1, V]` true at the masked ids, and the row with their logits at -inf.
    let unmasked_flags = match masked.is_empty() {
        true => None,
        false => {
            let mut keep = vec![true; v as usize];
            for &id in &masked {
                keep[id as usize] = false;
            }
            Some(Array::from_slice(&keep, &[1, v]))
        }
    };
    let zm = match &unmasked_flags {
        Some(keep) => select(keep, &z, inf())?,
        None => z.clone(),
    };
    // A row with no finite unmasked logit takes its lowest unmasked id.
    let first_unmasked = (0..v as u32)
        .find(|i| masked.binary_search(i).is_err())
        .expect("an unmasked id") as i32;
    let fix = |degenerate: &Array, tokens: Array| -> Result<Array> {
        select(
            degenerate,
            Array::from_int(first_unmasked),
            tokens.as_dtype(Dtype::Int32)?,
        )
    };
    let nan = || Array::from_f32(f32::NAN);
    let row = |x: &Array| x.reshape(&[b]);

    if options.is_greedy() {
        // The argmax, and its log-softmax through MLX's fused log-sum-exp.
        let lse = zm.logsumexp_axis(1, true)?;
        let degenerate = lse.eq(inf())?;
        let tokens = fix(&degenerate, argmax_axis(&zm, 1, true)?)?;
        let logprob = z.take_along_axis(&tokens, 1)?.subtract(&lse)?;
        let zero = select(&degenerate, nan(), zeros::<f32>(&[b, 1])?)?;
        return Ok(Sampled {
            tokens: row(&tokens)?,
            logprob: row(&logprob)?,
            tempered_logprob: row(&zero)?,
            sampling_logprob: row(&zero)?,
        });
    }
    let m = zm.max_axis(1, true)?;
    let degenerate = m.eq(inf())?;
    // Each log-probability is `s_t - ln sum exp(s)` over its set, `s = (z - m) tau` (tau 1 for
    // `logprob`), so that at temperature 1 with nothing cut the three are the same operations on
    // the same values, and equal to the bit.
    let d = zm.subtract(&m)?;
    let s1 = sum_axis(&d.exp()?, 1, true)?;
    let (tokens, tempered, sampling) = {
        let tau = (1.0 / temperature).min(f32::MAX);
        let s = d.multiply(Array::from_f32(tau))?;
        let unmasked = v as usize - masked.len();
        let top_k = (options.top_k > 0 && options.top_k < unmasked).then_some(options.top_k);
        let top_p = (options.top_p < 1.0).then_some(options.top_p);
        let threshold = cut(&zm, &m, tau, top_k, top_p, CUT_SUM)?;
        let kept = match (&threshold, &unmasked_flags) {
            (Some(t), Some(keep)) => Some(zm.ge(t)?.logical_and(keep)?),
            (Some(t), None) => Some(zm.ge(t)?),
            (None, Some(keep)) => Some(keep.clone()),
            (None, None) => None,
        };
        let noise = noise(options.seed, options.offset, [b, v], false)?;
        let score = s.add(&noise)?;
        let score = match &kept {
            Some(kept) => select(kept, &score, inf())?,
            None => score,
        };
        let tokens = argmax_axis(&score, 1, true)?;
        // Masked tokens' weights are exp(-inf) = 0: the sum over every token is the sum over U,
        // and with nothing cut the kept tokens' sum is that one.
        let w = s.exp()?;
        let total = sum_axis(&w, 1, true)?;
        let total_kept = match &threshold {
            Some(_) => {
                let kept = kept.as_ref().expect("a cut keeps a set");
                sum_axis(&select(kept, &w, Array::from_f32(0.0))?, 1, true)?
            }
            None => total.clone(),
        };
        let s_t = s.take_along_axis(&tokens, 1)?;
        (
            tokens,
            s_t.subtract(total.log()?)?,
            s_t.subtract(total_kept.log()?)?,
        )
    };
    let tokens = fix(&degenerate, tokens)?;
    let logprob = d.take_along_axis(&tokens, 1)?.subtract(s1.log()?)?;
    let tempered = select(&degenerate, nan(), &tempered)?;
    let sampling = select(&degenerate, nan(), &sampling)?;
    Ok(Sampled {
        tokens: row(&tokens)?,
        logprob: row(&logprob)?,
        tempered_logprob: row(&tempered)?,
        sampling_logprob: row(&sampling)?,
    })
}
