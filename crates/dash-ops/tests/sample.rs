//! Sampling the next token, against its definition and between the kernel and the reference.
//!
//! The definition (the module documentation) is computed here once more in f64, and every draw is
//! held to it: the token must be one of the kept set, `logprob` the log-softmax of the raw logits
//! over the unmasked ids, `tempered_logprob` that of the tempered logits, `sampling_logprob` the
//! log-probability under the kept set's tempered softmax; and the outputs that are one number by
//! the definition (`tempered_logprob` and `logprob` at temperature 1, `sampling_logprob` and
//! `tempered_logprob` with nothing cut) to the same bits. The distribution is held to the kept
//! set's probabilities by chi-square tests over many draws: each call draws as many rows of one row
//! of logits (the rows draw independent numbers), each call with its own offset. Small rows test
//! the cuts' ties and limits (top-k with ties at the `k`-th logit, `k = 1`, `k >= V`; top-p with
//! ties at the cut, `p = 1`, `p = 1e-6`, `p = 0`; both; masked ids); rows of tied groups larger
//! than the kernel's shared memory, and rows built for each of the routes the kernel's top-p takes,
//! test its slower passes. Greedy decoding is held to Burn's argmax in every dtype, the draws to
//! the same bits from run to run. The Gumbel variates are held word by word to their values in f64,
//! at the top of the range and where their computation changes, and seeds whose words lie at the
//! very top draw the definition's tokens.
//!
//! The reference runs on Flex; on the CubeCL device `DASH_OPS_DEVICE` names (`tests/common`) the
//! kernel where the device has what it asks, held also to the reference on that device token for
//! token, at the decoding shape `V = 248320` with batches of 8 and 32 among others (there its
//! `tempered_logprob` to f64 too).
//!
//! Errors are stated as `max |got - want|`; each bound sits just above the largest measured over
//! every build the tests run with.
//!
//!   cargo test -p dash-ops --test sample
//!   CUBECL_CUDA_TF32=0 cargo test -p dash-ops --features cuda --test sample

mod common;

use burn::backend::DType;
use burn::tensor::{Device, Tensor, TensorData};
use dash_ops::sample::{
    SampleOptions, Sampled, gumbel_variates, kernel_takes, probe, sample, sample_reference,
};
use std::sync::Mutex;

/// The probe counter is global; tests that read it take turns.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `sampling_logprob`, `tempered_logprob` and `logprob` against their f64 values on rows of a few
/// tokens: f32 roundings of the logits' differences, the sums and the logarithms (measured: 4.70e-7;
/// `tempered_logprob` 4.697e-7 on lavapipe, 4.105e-7 elsewhere).
const LOGPROB_EXACT: f64 = 1e-6;

/// The same at the larger rows, `V` from 1000 to 248320 (measured: 1.006e-6 on lavapipe, 9.01e-7
/// elsewhere, both at 75729 kept tokens of 100000; `tempered_logprob` 1.346e-6 at 248320 on CUDA,
/// 6.563e-7 at 100000).
const LOGPROB_LARGE: f64 = 2e-6;

/// The kernel's log-probabilities against the reference's on one device: their sums are formed in
/// different orders, and the two land an ulp apart at values from 16 to 32 (measured: 1.91e-6,
/// 2^-19, `tempered_logprob` too: 1.907e-6); twice that.
#[cfg(feature = "cubecl")]
const KERNEL_REFERENCE: f64 = 4e-6;

/// The Gumbel variates against their f64 values, at the words [`gumbel_words`] lists (measured:
/// 1.01e-6 on Flex, CUDA and the CPU runtime, about half the f32 spacing of `g` at the top of the
/// range; 5.05e-6 on lavapipe and 4.16e-6 on OpenGL, whose `ln` errs by up to about 1e-7 near 1,
/// in Goldberg's form just above `a = 2^-6`).
const GUMBEL: f64 = 6e-6;

/// One call's outputs.
#[derive(Debug, Clone, PartialEq)]
struct Out {
    tokens: Vec<i64>,
    logprob: Vec<f32>,
    tempered: Vec<f32>,
    sampling: Vec<f32>,
}

impl Out {
    fn of(sampled: Sampled) -> Self {
        Out {
            tokens: sampled
                .tokens
                .to_data()
                .convert::<i64>()
                .try_to_vec::<i64>()
                .unwrap(),
            logprob: values(&sampled.logprob),
            tempered: values(&sampled.tempered_logprob),
            sampling: values(&sampled.sampling_logprob),
        }
    }

    /// The same bits: tokens, and the three log-probabilities bit for bit (NaN as NaN).
    fn same_bits(&self, other: &Out) -> bool {
        self.tokens == other.tokens
            && bits(&self.logprob) == bits(&other.logprob)
            && bits(&self.tempered) == bits(&other.tempered)
            && bits(&self.sampling) == bits(&other.sampling)
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn values(x: &Tensor<1>) -> Vec<f32> {
    x.to_data().convert::<f32>().try_to_vec::<f32>().unwrap()
}

/// `values` `[b, v]` as logits in `dtype` on `device`.
fn logits(values: &[f32], [b, v]: [usize; 2], dtype: DType, device: &Device) -> Tensor<2> {
    Tensor::<2>::from_data(TensorData::new(values.to_vec(), [b, v]), device).cast(dtype)
}

/// The operation on `device` (the kernel where it takes the call), asserting that it ran exactly
/// where `kernel_takes` says, and always under CUDA. Callers hold [`serial`].
fn run(
    device: &Device,
    values: &[f32],
    dims: [usize; 2],
    dtype: DType,
    options: &SampleOptions,
) -> Out {
    let launches = probe::kernel_launches();
    let out = Out::of(sample(logits(values, dims, dtype, device), options));
    let takes = kernel_takes(device, dtype, dims);
    if common::on_cuda(device) {
        assert!(takes, "CUDA turned the kernel away for {dims:?} {dtype:?}");
    }
    let want = usize::from(takes && dims[0] > 0);
    // Read after the outputs: under fusion the kernel runs when they are read.
    let ran = probe::kernel_launches() - launches;
    assert_eq!(ran, want, "the kernel ran {ran} times, expected {want}");
    out
}

/// The reference on `device`.
fn run_reference(
    device: &Device,
    values: &[f32],
    dims: [usize; 2],
    dtype: DType,
    options: &SampleOptions,
) -> Out {
    Out::of(sample_reference(
        logits(values, dims, dtype, device),
        options,
    ))
}

/// A seeded generator: xorshift64* and Box-Muller.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

/// `b` rows of `v` logits `scale * N(0, 1)`, a few of each row's tokens raised by up to `lift`
/// (a peaked distribution, as a model's), rounded to bf16 so that every dtype holds them exactly
/// but f16's range.
fn random_logits(seed: u64, [b, v]: [usize; 2], scale: f64, lift: f64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(b * v);
    for _ in 0..b {
        let mut row: Vec<f64> = (0..v).map(|_| scale * rng.normal()).collect();
        if lift > 0.0 {
            for k in 0..8.min(v) {
                let at = ((rng.uniform() * v as f64) as usize).min(v - 1);
                row[at] += lift * (1.0 - k as f64 / 8.0);
            }
        }
        out.extend(row.into_iter().map(|x| bf16(x as f32)));
    }
    out
}

fn bf16(x: f32) -> f32 {
    burn::backend::bf16::from_f32(x).to_f32()
}

fn f16(x: f32) -> f32 {
    burn::backend::f16::from_f32(x).to_f32()
}

/// The definition in f64 over one row.
struct Exact {
    /// The kept ids in increasing order, and their probabilities.
    kept: Vec<usize>,
    prob: Vec<f64>,
    /// The log-softmax of the raw logits over the unmasked ids, per token (`-inf` when masked).
    log_softmax: Vec<f64>,
    /// The log-softmax of the tempered logits over the unmasked ids, per token (`-inf` when
    /// masked); greedy, the point mass at the token: 0 there, `-inf` elsewhere.
    tempered: Vec<f64>,
}

impl Exact {
    fn of(z: &[f32], options: &SampleOptions) -> Self {
        let v = z.len();
        let masked = |i: usize| options.masked.contains(&(i as u32));
        let unmasked: Vec<usize> = (0..v).filter(|&i| !masked(i)).collect();
        let m = unmasked
            .iter()
            .map(|&i| z[i] as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let lse = m + unmasked
            .iter()
            .map(|&i| (z[i] as f64 - m).exp())
            .sum::<f64>()
            .ln();
        let log_softmax = (0..v)
            .map(|i| {
                if masked(i) {
                    f64::NEG_INFINITY
                } else {
                    z[i] as f64 - lse
                }
            })
            .collect();
        if options.temperature == 0.0 {
            let top = unmasked
                .iter()
                .copied()
                .find(|&i| z[i] as f64 == m)
                .unwrap();
            return Exact {
                kept: vec![top],
                prob: vec![1.0],
                log_softmax,
                tempered: tempered_log_softmax(z, options),
            };
        }
        let tau = (1.0f32 / options.temperature).min(f32::MAX) as f64;
        let w = |i: usize| ((z[i] as f64 - m) * tau).exp();
        let mut kept = unmasked.clone();
        if options.top_k > 0 && options.top_k < unmasked.len() {
            let mut sorted: Vec<f32> = unmasked.iter().map(|&i| z[i]).collect();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let kth = sorted[options.top_k - 1];
            kept.retain(|&i| z[i] >= kth);
        }
        if options.top_p < 1.0 {
            let mut by_logit = kept.clone();
            by_logit.sort_by(|&a, &b| z[b].partial_cmp(&z[a]).unwrap());
            let total: f64 = kept.iter().map(|&i| w(i)).sum();
            let mut cumulative = 0.0;
            let mut cut = z[by_logit[by_logit.len() - 1]];
            for &i in &by_logit {
                cumulative += w(i);
                if cumulative >= options.top_p as f64 * total {
                    cut = z[i];
                    break;
                }
            }
            kept.retain(|&i| z[i] >= cut);
        }
        let total: f64 = kept.iter().map(|&i| w(i)).sum();
        let prob = kept.iter().map(|&i| w(i) / total).collect();
        Exact {
            kept,
            prob,
            log_softmax,
            tempered: tempered_log_softmax(z, options),
        }
    }

    fn prob_of(&self, token: usize) -> Option<f64> {
        self.kept
            .iter()
            .position(|&i| i == token)
            .map(|at| self.prob[at])
    }
}

/// The log-softmax of the tempered logits `(z - m) tau` over the unmasked ids of the row `z` in
/// f64, per token (`-inf` when masked); greedy, the point mass at the lowest id holding the
/// largest logit (0 there, `-inf` elsewhere).
fn tempered_log_softmax(z: &[f32], options: &SampleOptions) -> Vec<f64> {
    let masked = |i: usize| options.masked.contains(&(i as u32));
    let unmasked: Vec<usize> = (0..z.len()).filter(|&i| !masked(i)).collect();
    let m = unmasked
        .iter()
        .map(|&i| z[i] as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    if options.temperature == 0.0 {
        let top = unmasked.iter().copied().find(|&i| z[i] as f64 == m);
        return (0..z.len())
            .map(|i| match Some(i) == top {
                true => 0.0,
                false => f64::NEG_INFINITY,
            })
            .collect();
    }
    let tau = (1.0f32 / options.temperature).min(f32::MAX) as f64;
    let s = |i: usize| (z[i] as f64 - m) * tau;
    let lse = unmasked.iter().map(|&i| s(i).exp()).sum::<f64>().ln();
    (0..z.len())
        .map(|i| match masked(i) {
            true => f64::NEG_INFINITY,
            false => s(i) - lse,
        })
        .collect()
}

/// The chi-square statistic of `counts` against `n` draws of `probs`, cells of fewer than five
/// expected draws pooled into one, and its degrees of freedom.
fn chi_square(counts: &[u64], probs: &[f64], n: u64) -> (f64, usize) {
    let mut cells = Vec::new();
    let (mut pooled_count, mut pooled_expected) = (0.0, 0.0);
    for (&c, &p) in counts.iter().zip(probs) {
        let expected = p * n as f64;
        if expected < 5.0 {
            pooled_count += c as f64;
            pooled_expected += expected;
        } else {
            cells.push((c as f64, expected));
        }
    }
    if pooled_expected > 0.0 {
        cells.push((pooled_count, pooled_expected));
    }
    let statistic = cells.iter().map(|(c, e)| (c - e) * (c - e) / e).sum();
    (statistic, cells.len().saturating_sub(1))
}

/// The significance level of every chi-square test here: a correct sampler fails one with
/// probability 1e-4 (the tests are seeded, so each passes or fails for good).
const ALPHA: f64 = 1e-4;

/// The chi-square quantile at `1 - ALPHA` for `df` degrees of freedom, by Wilson and Hilferty's
/// cube-root approximation (within a few parts in a thousand of the exact quantile from 3 degrees
/// of freedom on; 1 and 2 are exact).
fn chi_square_bound(df: usize) -> f64 {
    match df {
        0 => 0.0,
        1 => 15.137,
        2 => -2.0 * ALPHA.ln(),
        _ => {
            let z = 3.719_016; // the standard normal's quantile at 1 - 1e-4
            let k = df as f64;
            let c = 2.0 / (9.0 * k);
            k * (1.0 - c + z * c.sqrt()).powi(3)
        }
    }
}

/// `calls` calls of `rows` copies of the row `z`, offsets `0..calls`: every draw checked against
/// the definition, then the counts against its probabilities. Returns the chi-square statistic,
/// its degrees of freedom and bound.
fn check_distribution(
    label: &str,
    device: &Device,
    z: &[f32],
    options: &SampleOptions,
    rows: usize,
    calls: usize,
) -> (f64, usize, f64) {
    let v = z.len();
    let exact = Exact::of(z, options);
    let mut counts = vec![0u64; exact.kept.len()];
    let mut worst = (0.0f64, 0.0f64, 0.0f64);
    let values: Vec<f32> = (0..rows).flat_map(|_| z.iter().copied()).collect();
    for call in 0..calls {
        let options = SampleOptions {
            offset: call as u64,
            ..options.clone()
        };
        let out = run(device, &values, [rows, v], DType::F32, &options);
        for r in 0..rows {
            let t = out.tokens[r] as usize;
            let p = exact.prob_of(t).unwrap_or_else(|| {
                panic!(
                    "{label}: drew token {t}, outside the kept set {:?}",
                    exact.kept
                )
            });
            counts[exact.kept.iter().position(|&i| i == t).unwrap()] += 1;
            let sampling = out.sampling[r] as f64;
            let logprob = out.logprob[r] as f64;
            worst.0 = worst.0.max((sampling - p.ln()).abs());
            worst.1 = worst.1.max((logprob - exact.log_softmax[t]).abs());
            worst.2 = worst
                .2
                .max((out.tempered[r] as f64 - exact.tempered[t]).abs());
        }
    }
    let n = (rows * calls) as u64;
    let (statistic, df) = chi_square(&counts, &exact.prob, n);
    let bound = chi_square_bound(df);
    println!(
        "{label:34} {n} draws, kept {:?}{}: chi-square {statistic:.2} on {df} df (bound {bound:.2}); \
         sampling_logprob max abs {:.3e}, logprob max abs {:.3e}, tempered_logprob max abs {:.3e}",
        &exact.kept[..exact.kept.len().min(8)],
        if exact.kept.len() > 8 { "..." } else { "" },
        worst.0,
        worst.1,
        worst.2,
    );
    assert!(
        worst.0 <= LOGPROB_EXACT && worst.1 <= LOGPROB_EXACT && worst.2 <= LOGPROB_EXACT,
        "{label}: log-probabilities off by {worst:?}"
    );
    // One cell (a single kept token) leaves nothing to test.
    assert!(
        df == 0 || statistic <= bound,
        "{label}: chi-square {statistic} above {bound} ({df} df)"
    );
    // Every kept token expected 20 times or more turns up.
    for (at, &c) in counts.iter().enumerate() {
        if exact.prob[at] * n as f64 >= 20.0 {
            assert!(c > 0, "{label}: kept token {} never drawn", exact.kept[at]);
        }
    }
    (statistic, df, bound)
}

/// `ln p` for probabilities `p` (their logits at temperature 1).
fn logits_of(p: &[f64]) -> Vec<f32> {
    p.iter().map(|&p| p.ln() as f32).collect()
}

/// The distribution cases: each a row, options and the kept set the definition must give.
fn distribution_cases() -> Vec<(&'static str, Vec<f32>, SampleOptions, Vec<usize>)> {
    let options = SampleOptions::default();
    let twelve = vec![
        2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0, 1.5, 0.25, -0.25, 3.0,
    ];
    let ties = vec![5.0, 3.0, 3.0, 3.0, 1.0, 0.0, -1.0, 4.0, 2.0, 2.0, 0.5, -2.0];
    let two_tops = vec![1.0, 4.0, 2.0, 4.0, 0.0, -1.0];
    let staircase = logits_of(&[0.4, 0.2, 0.15, 0.1, 0.08, 0.04, 0.02, 0.01]);
    let flat_tail = logits_of(&[0.4, 0.2, 0.2, 0.2]);
    let both = logits_of(&[0.4, 0.25, 0.15, 0.1, 0.05, 0.05]);
    vec![
        (
            "temperature 0.7",
            twelve.clone(),
            SampleOptions {
                temperature: 0.7,
                seed: 1,
                ..options.clone()
            },
            (0..12).collect(),
        ),
        (
            "temperature 1.5",
            twelve.clone(),
            SampleOptions {
                temperature: 1.5,
                seed: 2,
                ..options.clone()
            },
            (0..12).collect(),
        ),
        (
            "top-k 3, ties kept",
            ties.clone(),
            SampleOptions {
                top_k: 3,
                seed: 3,
                ..options.clone()
            },
            vec![0, 1, 2, 3, 7],
        ),
        (
            "top-k 1, two maxima",
            two_tops.clone(),
            SampleOptions {
                top_k: 1,
                seed: 4,
                ..options.clone()
            },
            vec![1, 3],
        ),
        (
            "top-k V",
            twelve.clone(),
            SampleOptions {
                top_k: 12,
                seed: 5,
                ..options.clone()
            },
            (0..12).collect(),
        ),
        (
            "top-k above V",
            twelve.clone(),
            SampleOptions {
                top_k: 1000,
                temperature: 0.8,
                seed: 6,
                ..options.clone()
            },
            (0..12).collect(),
        ),
        (
            "top-p 0.7",
            staircase.clone(),
            SampleOptions {
                top_p: 0.7,
                seed: 7,
                ..options.clone()
            },
            vec![0, 1, 2],
        ),
        (
            "top-p 0.7, temperature 0.5",
            staircase.clone(),
            SampleOptions {
                top_p: 0.7,
                temperature: 0.5,
                seed: 8,
                ..options.clone()
            },
            vec![0, 1],
        ),
        (
            "top-p 0.5, ties at the cut",
            flat_tail.clone(),
            SampleOptions {
                top_p: 0.5,
                seed: 9,
                ..options.clone()
            },
            vec![0, 1, 2, 3],
        ),
        (
            "top-p 1e-6",
            twelve.clone(),
            SampleOptions {
                top_p: 1e-6,
                seed: 10,
                ..options.clone()
            },
            vec![11],
        ),
        (
            "top-p 0, two maxima",
            two_tops.clone(),
            SampleOptions {
                top_p: 0.0,
                seed: 11,
                ..options.clone()
            },
            vec![1, 3],
        ),
        (
            "top-p 1",
            staircase.clone(),
            SampleOptions {
                top_p: 1.0,
                seed: 12,
                ..options.clone()
            },
            (0..8).collect(),
        ),
        (
            "top-k 4 then top-p 0.7",
            both.clone(),
            SampleOptions {
                top_k: 4,
                top_p: 0.7,
                seed: 13,
                ..options.clone()
            },
            vec![0, 1],
        ),
        (
            "masked argmax, top-k 2",
            twelve.clone(),
            SampleOptions {
                top_k: 2,
                masked: vec![11, 7],
                seed: 14,
                ..options.clone()
            },
            vec![0, 8],
        ),
        (
            "masked, top-p 0.8",
            staircase.clone(),
            SampleOptions {
                top_p: 0.8,
                masked: vec![1, 5],
                temperature: 1.2,
                seed: 15,
                ..options.clone()
            },
            vec![0, 2, 3],
        ),
        (
            "masked, top-k 3, temperature 0.8",
            twelve.clone(),
            SampleOptions {
                top_k: 3,
                masked: vec![11, 0],
                temperature: 0.8,
                seed: 16,
                ..options.clone()
            },
            vec![1, 2, 8],
        ),
    ]
}

/// The definition's own kept sets for the cases above, as the cases state them: a check of the
/// f64 model the draws are held to.
#[test]
fn the_definition_keeps_what_the_cases_say() {
    for (label, z, options, kept) in distribution_cases() {
        assert_eq!(Exact::of(&z, &options).kept, kept, "{label}");
    }
}

/// Draws per distribution case: 2048 rows a call, 48 calls.
const ROWS: usize = 2048;
const CALLS: usize = 48;

fn distributions(device: &Device) {
    let _serial = serial();
    println!("{}", common::describe(device));
    for (label, z, options, _) in distribution_cases() {
        check_distribution(label, device, &z, &options, ROWS, CALLS);
    }
}

#[cfg(feature = "flex")]
#[test]
fn draws_follow_the_kept_sets_on_flex() {
    distributions(&Device::flex());
}

/// Greedy decoding is Burn's argmax of the row with the masked logits at `-inf`, bit for bit, in
/// every dtype, with ties (bf16 has many) to the lowest id; `logprob` is the log-softmax at the
/// token, `tempered_logprob` and `sampling_logprob` 0.
fn greedy_is_argmax(device: &Device, reference: bool) {
    let _serial = serial();
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        if !common::has(device, dtype) {
            continue;
        }
        for (dims, masked) in [
            ([5, 1000], vec![]),
            ([3, 1031], vec![0, 17, 1030]),
            ([1, 3], vec![]),
            ([4, 40_000], vec![5, 39_999]),
        ] {
            let [b, v] = dims;
            // Coarse logits: many ties at the top.
            let mut rng = Rng::new((b * v) as u64);
            let values: Vec<f32> = (0..b * v)
                .map(|_| f16(bf16((rng.normal() * 2.0).round() as f32 / 2.0)))
                .collect();
            let options = SampleOptions {
                masked: masked.clone(),
                ..SampleOptions::greedy()
            };
            let out = match reference {
                true => run_reference(device, &values, dims, dtype, &options),
                false => run(device, &values, dims, dtype, &options),
            };
            let mut with_mask = values.clone();
            for row in 0..b {
                for &id in &masked {
                    with_mask[row * v + id as usize] = f32::NEG_INFINITY;
                }
            }
            let argmax: Vec<i64> = logits(&with_mask, dims, dtype, device)
                .argmax(1)
                .to_data()
                .convert::<i64>()
                .try_to_vec::<i64>()
                .unwrap();
            assert_eq!(out.tokens, argmax, "{dtype:?} {dims:?}");
            let mut worst = 0.0f64;
            for row in 0..b {
                let z = &values[row * v..(row + 1) * v];
                let exact = Exact::of(z, &options);
                let t = out.tokens[row] as usize;
                assert!(exact.kept.contains(&t), "{dtype:?} {dims:?} row {row}");
                worst = worst.max((out.logprob[row] as f64 - exact.log_softmax[t]).abs());
                assert_eq!(out.sampling[row], 0.0, "{dtype:?} {dims:?} row {row}");
                assert_eq!(out.tempered[row], 0.0, "{dtype:?} {dims:?} row {row}");
            }
            println!(
                "greedy {dtype:?} {dims:?}: tokens equal Burn's argmax; logprob max abs {worst:.3e}"
            );
            assert!(
                worst <= LOGPROB_LARGE,
                "{dtype:?} {dims:?}: logprob off by {worst:e}"
            );
        }
    }
}

#[cfg(feature = "flex")]
#[test]
fn greedy_is_burns_argmax_on_flex() {
    greedy_is_argmax(&Device::flex(), false);
}

/// The same seed and offset draw the same bits; another offset or seed, other tokens.
fn reproducible(device: &Device) {
    let _serial = serial();
    let dims = [64, 5000];
    let values = random_logits(21, dims, 1.0, 0.0);
    let options = SampleOptions {
        temperature: 0.9,
        top_p: 0.9,
        seed: 1234,
        offset: 77,
        ..SampleOptions::default()
    };
    let first = run(device, &values, dims, DType::F32, &options);
    let again = run(device, &values, dims, DType::F32, &options);
    assert!(first.same_bits(&again), "two runs differ");
    let differ = |other: &Out| {
        first
            .tokens
            .iter()
            .zip(&other.tokens)
            .filter(|(a, b)| a != b)
            .count()
    };
    let next = run(
        device,
        &values,
        dims,
        DType::F32,
        &SampleOptions {
            offset: 78,
            ..options.clone()
        },
    );
    let reseeded = run(
        device,
        &values,
        dims,
        DType::F32,
        &SampleOptions {
            seed: 4321,
            ..options.clone()
        },
    );
    println!(
        "reproducible: same bits twice; offset + 1 changes {} of 64 tokens, another seed {}",
        differ(&next),
        differ(&reseeded)
    );
    // Thousands of tokens are kept per row: a repeat is rare.
    assert!(differ(&next) >= 56 && differ(&reseeded) >= 56);
    // The rows of one call draw apart: the same row repeated gives different tokens.
    let row: Vec<f32> = values[..5000].to_vec();
    let repeated: Vec<f32> = (0..64).flat_map(|_| row.iter().copied()).collect();
    let out = run(device, &repeated, dims, DType::F32, &options);
    let mut distinct = out.tokens.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() >= 56,
        "64 copies of a row drew only {} tokens",
        distinct.len()
    );
}

#[cfg(feature = "flex")]
#[test]
fn draws_are_reproducible_on_flex() {
    reproducible(&Device::flex());
}

/// Edges: one row; a vocabulary of one; every token but one masked; huge and tiny logits; `-inf`
/// logits; a temperature near 0 against 0; a row with no finite logit.
fn edges(device: &Device, reference: bool) {
    let _serial = serial();
    let go = |values: &[f32], dims: [usize; 2], options: &SampleOptions| match reference {
        true => run_reference(device, values, dims, DType::F32, options),
        false => run(device, values, dims, DType::F32, options),
    };
    let modes = [
        SampleOptions::greedy(),
        SampleOptions::default(),
        SampleOptions {
            temperature: 0.7,
            top_p: 0.9,
            ..SampleOptions::default()
        },
        SampleOptions {
            top_k: 3,
            ..SampleOptions::default()
        },
        SampleOptions {
            top_k: 3,
            top_p: 0.5,
            temperature: 2.0,
            ..SampleOptions::default()
        },
    ];
    for options in &modes {
        // A vocabulary of one.
        let out = go(&[0.25], [1, 1], options);
        assert_eq!(
            (
                out.tokens[0],
                out.logprob[0],
                out.tempered[0],
                out.sampling[0]
            ),
            (0, 0.0, 0.0, 0.0),
            "{options:?}"
        );
        // All but one masked.
        let v = 50;
        let values: Vec<f32> = (0..v).map(|i| i as f32 * 0.1).collect();
        let masked: Vec<u32> = (0..v as u32).filter(|&i| i != 17).collect();
        let out = go(
            &values,
            [1, v],
            &SampleOptions {
                masked,
                ..options.clone()
            },
        );
        assert_eq!(
            (
                out.tokens[0],
                out.logprob[0],
                out.tempered[0],
                out.sampling[0]
            ),
            (17, 0.0, 0.0, 0.0),
            "{options:?}"
        );
        // Huge and tiny logits, and -inf: finite log-probabilities, never a -inf token.
        let values = vec![
            1e30,
            -1e30,
            3e38,
            -3e38,
            f32::NEG_INFINITY,
            2.9e38,
            0.0,
            -0.0,
            1e-30,
            f32::NEG_INFINITY,
        ];
        let out = go(&values, [1, 10], options);
        let t = out.tokens[0] as usize;
        assert!(values[t] > 1e38, "{options:?}: drew {t}");
        assert!(
            out.logprob[0].is_finite()
                && out.tempered[0].is_finite()
                && out.sampling[0].is_finite(),
            "{options:?}: {out:?}"
        );
        let values = vec![-3e38, -3.1e38, f32::NEG_INFINITY, -3.2e38];
        let out = go(&values, [1, 4], options);
        assert!(
            out.tokens[0] != 2
                && out.logprob[0].is_finite()
                && out.tempered[0].is_finite()
                && out.sampling[0].is_finite(),
            "{options:?}: {out:?}"
        );
        // A row whose unmasked logits are all -inf: its lowest unmasked id, NaN log-probabilities.
        let values = vec![1.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 2.0];
        let out = go(
            &values,
            [1, 4],
            &SampleOptions {
                masked: vec![0, 3],
                ..options.clone()
            },
        );
        assert_eq!(out.tokens[0], 1, "{options:?}");
        assert!(
            out.logprob[0].is_nan() && out.tempered[0].is_nan() && out.sampling[0].is_nan(),
            "{options:?}: {out:?}"
        );
    }
    // A temperature near 0 draws the argmax wherever it leads by a margin, as greedy does.
    let dims = [16, 777];
    let values = random_logits(5, dims, 1.0, 3.0);
    let greedy = go(&values, dims, &SampleOptions::greedy());
    let cold = go(
        &values,
        dims,
        &SampleOptions {
            temperature: 1e-6,
            seed: 9,
            ..SampleOptions::default()
        },
    );
    for row in 0..dims[0] {
        let z = &values[row * dims[1]..(row + 1) * dims[1]];
        let mut sorted = z.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if sorted[0] - sorted[1] > 1e-3 {
            assert_eq!(cold.tokens[row], greedy.tokens[row], "row {row}");
            assert!(
                cold.sampling[row].abs() < 1e-6,
                "row {row}: {}",
                cold.sampling[row]
            );
        }
    }
}

#[cfg(feature = "flex")]
#[test]
fn edges_on_flex() {
    edges(&Device::flex(), false);
}

/// The outputs that are one number by the definition, to the same bits, through the operation (the
/// kernel where it runs) and through the reference: at temperature 1 `tempered_logprob` is
/// `logprob`, cut or not; with nothing cut (top-k 0 or at least the unmasked vocabulary, top-p 1)
/// `sampling_logprob` is `tempered_logprob`, at any temperature; so at temperature 1 with nothing
/// cut the three are one. In every dtype the device has, with masked ids and without.
fn identities(device: &Device) {
    let _serial = serial();
    let dtypes: Vec<DType> = [DType::F32, DType::BF16, DType::F16]
        .into_iter()
        .filter(|&dtype| common::has(device, dtype))
        .collect();
    let d = SampleOptions::default();
    // Each case: whether `tempered_logprob` is `logprob`, and whether `sampling_logprob` is
    // `tempered_logprob`.
    let cases = [
        ("temperature 1", 51, d.clone(), true, true),
        (
            "temperature 1, top-k above V, masked",
            52,
            SampleOptions {
                top_k: 1 << 20,
                masked: vec![0, 7, 999],
                ..d.clone()
            },
            true,
            true,
        ),
        (
            "temperature 0.7",
            53,
            SampleOptions {
                temperature: 0.7,
                ..d.clone()
            },
            false,
            true,
        ),
        (
            "temperature 1.3, top-p 1, masked",
            54,
            SampleOptions {
                temperature: 1.3,
                top_p: 1.0,
                masked: vec![1, 500],
                ..d.clone()
            },
            false,
            true,
        ),
        (
            "temperature 1, top-p 0.9",
            55,
            SampleOptions {
                top_p: 0.9,
                ..d.clone()
            },
            true,
            false,
        ),
        (
            "temperature 1, top-k 50, masked",
            56,
            SampleOptions {
                top_k: 50,
                masked: vec![2, 3],
                ..d.clone()
            },
            true,
            false,
        ),
    ];
    for (label, seed, options, tempered_is_logprob, sampling_is_tempered) in cases {
        let options = SampleOptions { seed, ..options };
        for (dims, scale, lift) in [
            ([5, 1000], 2.0, 6.0),
            ([3, 1031], 1.0, 0.0),
            ([4, 20_000], 3.0, 10.0),
        ] {
            let values = random_logits(seed, dims, scale, lift);
            for &dtype in &dtypes {
                for (who, out) in [
                    ("operation", run(device, &values, dims, dtype, &options)),
                    (
                        "reference",
                        run_reference(device, &values, dims, dtype, &options),
                    ),
                ] {
                    let at = format!("{label} {dims:?} {dtype:?}, the {who}'s");
                    if tempered_is_logprob {
                        assert_eq!(
                            bits(&out.tempered),
                            bits(&out.logprob),
                            "{at} tempered_logprob is not logprob"
                        );
                    }
                    if sampling_is_tempered {
                        assert_eq!(
                            bits(&out.sampling),
                            bits(&out.tempered),
                            "{at} sampling_logprob is not tempered_logprob"
                        );
                    }
                }
            }
        }
        let held = match (tempered_is_logprob, sampling_is_tempered) {
            (true, true) => "the three the same bits",
            (true, false) => "tempered_logprob logprob's bits",
            _ => "sampling_logprob tempered_logprob's bits",
        };
        println!("identities, {label}: {held}");
    }
}

#[cfg(feature = "flex")]
#[test]
fn identities_hold_bit_for_bit_on_flex() {
    identities(&Device::flex());
}

/// Groups of tokens tied at one logit, larger than the kernel's candidate capacity: the cuts fall
/// inside a tie group (top-k with `k` above the cube's units, top-p), so the kernel refines past
/// its fast path; the groups' draws against their probabilities, each draw against the definition.
fn tied_groups(device: &Device, calls: usize) {
    let _serial = serial();
    let group = 5000;
    let levels = [1.0f32, 0.5, 0.0, -0.5];
    let z: Vec<f32> = levels
        .iter()
        .flat_map(|&l| std::iter::repeat_n(l, group))
        .collect();
    let cases = [
        (
            "top-p 0.6 in tied groups",
            SampleOptions {
                top_p: 0.6,
                seed: 31,
                ..SampleOptions::default()
            },
        ),
        (
            "top-k 6000 in tied groups",
            SampleOptions {
                top_k: 6000,
                seed: 32,
                ..SampleOptions::default()
            },
        ),
        (
            "top-k 6000, top-p 0.5 in tied groups",
            SampleOptions {
                top_k: 6000,
                top_p: 0.5,
                temperature: 0.8,
                seed: 33,
                ..SampleOptions::default()
            },
        ),
        (
            "temperature 1.3 over tied groups",
            SampleOptions {
                temperature: 1.3,
                seed: 34,
                ..SampleOptions::default()
            },
        ),
    ];
    let rows = 256;
    for (label, options) in cases {
        let exact = Exact::of(&z, &options);
        let mut groups = vec![0u64; levels.len()];
        let mut group_prob = vec![0.0f64; levels.len()];
        for (at, &i) in exact.kept.iter().enumerate() {
            group_prob[i / group] += exact.prob[at];
        }
        let values: Vec<f32> = (0..rows).flat_map(|_| z.iter().copied()).collect();
        let (mut worst, mut tempered) = (0.0f64, 0.0f64);
        for call in 0..calls {
            let options = SampleOptions {
                offset: call as u64,
                ..options.clone()
            };
            let out = run(device, &values, [rows, z.len()], DType::F32, &options);
            for r in 0..rows {
                let t = out.tokens[r] as usize;
                let p = exact
                    .prob_of(t)
                    .unwrap_or_else(|| panic!("{label}: token {t} outside the kept set"));
                worst = worst.max((out.sampling[r] as f64 - p.ln()).abs());
                tempered = tempered.max((out.tempered[r] as f64 - exact.tempered[t]).abs());
                groups[t / group] += 1;
            }
        }
        let n = (rows * calls) as u64;
        let (statistic, df) = chi_square(&groups, &group_prob, n);
        let bound = chi_square_bound(df);
        println!(
            "{label:38} {n} draws, groups {groups:?} of {group_prob:.3?}: chi-square {statistic:.2} on \
             {df} df (bound {bound:.2}); sampling_logprob max abs {worst:.3e}, tempered_logprob max \
             abs {tempered:.3e}"
        );
        assert!(
            df == 0 || statistic <= bound,
            "{label}: chi-square {statistic} above {bound}"
        );
        assert!(
            worst <= LOGPROB_LARGE,
            "{label}: sampling_logprob off by {worst:e}"
        );
        assert!(
            tempered <= LOGPROB_LARGE,
            "{label}: tempered_logprob off by {tempered:e}"
        );
    }
}

#[cfg(feature = "flex")]
#[test]
fn tied_groups_on_flex() {
    tied_groups(&Device::flex(), 4);
}

/// The route the kernel's top-p alone takes for row `z` at temperature 1 (the kernel module's
/// steps 2 and 3, replayed on the host in f64, the bands in f32 as the kernel forms them) with
/// `capacity` candidate slots: 0 when the candidates fit at once, 1 when the bands down to the cut
/// do, 2 when the cut band's sub-bands do, 3 for the radix passes.
fn top_p_route(z: &[f32], p: f64, capacity: usize) -> u32 {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: Vec<f32> = z.iter().map(|&x| x - m).collect();
    let w_total: f64 = s.iter().map(|&x| (x as f64).exp()).sum();
    let floor = ((1.0 - p) * w_total / z.len() as f64).ln() - 0.01;
    let band = |x: f32| (((1.0f32 - x).to_bits() >> 22) as i64 - 254).min(7) as usize;
    let candidates: Vec<f32> = s.iter().copied().filter(|&x| x as f64 >= floor).collect();
    if candidates.len() <= capacity {
        return 0;
    }
    let (mut counts, mut weights) = ([0usize; 8], [0f64; 8]);
    for &x in &candidates {
        counts[band(x)] += 1;
        weights[band(x)] += (x as f64).exp();
    }
    let (mut cut, mut kept, mut above, mut weight) = (7, 0, 0.0, 0.0);
    for j in 0..8 {
        above = weight;
        kept += counts[j];
        weight += weights[j];
        if weight >= p * w_total {
            cut = j;
            break;
        }
    }
    if kept <= capacity {
        return 1;
    }
    if kept > 8 * capacity {
        return 3;
    }
    let x_lo = f32::from_bits((254 + cut as u32) << 22);
    let x_hi = match cut {
        7 => 1.0 - floor as f32,
        _ => f32::from_bits((255 + cut as u32) << 22),
    };
    let scale = 8.0 / (x_hi - x_lo);
    let (mut sub_n, mut sub_w) = ([0usize; 8], [0f64; 8]);
    for &x in candidates.iter().filter(|&&x| band(x) == cut) {
        let sub = (((1.0 - x - x_lo) * scale).max(0.0) as usize).min(7);
        sub_n[sub] += 1;
        sub_w[sub] += (x as f64).exp();
    }
    let (mut kept, mut weight) = (kept - counts[cut], above);
    for j in 0..8 {
        kept += sub_n[j];
        weight += sub_w[j];
        if weight >= p * w_total {
            break;
        }
    }
    if kept <= capacity { 2 } else { 3 }
}

/// A group of tokens of a constructed row: `(count, lo, hi)`.
type Group = (usize, f32, f32);

/// A row of `v` logits: the top one at 0, and groups of `count` tokens spread evenly over
/// `[lo, hi]`, interleaved through the row, the rest at -40.
fn constructed_row(v: usize, groups: &[Group]) -> Vec<f32> {
    let mut row = vec![-40.0f32; v];
    row[v / 2] = 0.0;
    let mut at = 0;
    for &(count, lo, hi) in groups {
        for k in 0..count {
            // A stride coprime to the row spreads each group over every unit.
            let i = (at * 7919) % v;
            at += 1;
            if i == v / 2 {
                continue;
            }
            row[i] = lo + (hi - lo) * k as f32 / count.max(2).saturating_sub(1) as f32;
        }
    }
    row
}

/// Rows whose top-p cut takes each of the kernel's routes with the device's capacity (0 where the
/// kernel does not run: the same rows, held to the definition only): every draw to the definition,
/// and the kernel's tokens to the reference's.
fn top_p_routes(device: &Device, capacity: usize) {
    let _serial = serial();
    let v = 100_000;
    // Each row: its route, its top-p, and its groups of tokens (count, lowest, highest logit).
    let rows: [(&str, u32, f64, &[Group]); 4] = [
        ("candidates fit", 0, 0.9, &[(3000, -9.0, -7.0)]),
        (
            "the bands down to the cut fit",
            1,
            0.9,
            &[(4000, -8.0, -7.0), (30_000, -12.0, -11.2)],
        ),
        (
            "the cut band's sub-bands fit",
            2,
            0.5,
            &[(30_000, -11.0, -7.0)],
        ),
        ("radix passes", 3, 0.9, &[(90_000, -2.0, -1.0)]),
    ];
    for (label, route, p, groups) in rows {
        let z = constructed_row(v, groups);
        if capacity > 0 {
            assert_eq!(
                top_p_route(&z, p, capacity),
                route,
                "{label}: the row takes another route"
            );
        }
        let options = SampleOptions {
            top_p: p as f32,
            seed: 41 + route as u64,
            ..SampleOptions::default()
        };
        let exact = Exact::of(&z, &options);
        // Fewer draws where only the reference runs (its sort reads the rows back to the host).
        let (rows, calls) = match capacity > 0 {
            true => (16, 2),
            false => (4, 1),
        };
        let values: Vec<f32> = (0..rows).flat_map(|_| z.iter().copied()).collect();
        let (mut worst, mut tempered) = (0.0f64, 0.0f64);
        for call in 0..calls {
            let options = SampleOptions {
                offset: call,
                ..options.clone()
            };
            let out = run(device, &values, [rows, v], DType::F32, &options);
            let want = run_reference(device, &values, [rows, v], DType::F32, &options);
            assert_eq!(
                out.tokens, want.tokens,
                "{label}: the kernel's tokens are not the reference's"
            );
            for r in 0..rows {
                let t = out.tokens[r] as usize;
                let prob = exact
                    .prob_of(t)
                    .unwrap_or_else(|| panic!("{label}: token {t} outside the kept set"));
                worst = worst.max((out.sampling[r] as f64 - prob.ln()).abs());
                tempered = tempered.max((out.tempered[r] as f64 - exact.tempered[t]).abs());
            }
        }
        println!(
            "top-p route {route} ({label}): {} kept of {v}; the reference's tokens; sampling_logprob max abs {worst:.3e}, tempered_logprob max abs {tempered:.3e}",
            exact.kept.len()
        );
        assert!(
            worst <= LOGPROB_LARGE,
            "{label}: sampling_logprob off by {worst:e}"
        );
        assert!(
            tempered <= LOGPROB_LARGE,
            "{label}: tempered_logprob off by {tempered:e}"
        );
    }
}

#[cfg(feature = "flex")]
#[test]
fn top_p_routes_on_flex() {
    top_p_routes(&Device::flex(), 0);
}

/// The definition's `g` in f64 for the word `x`: `a` rounded to f32 as the definition rounds it,
/// then `E = -ln a` or `-ln_1p(-a)` and `g = -ln E`.
fn gumbel_f64(x: u32) -> f64 {
    let upper = x >> 31 == 1;
    let y = if upper { !x } else { x };
    let a = f64::from(((y << 1) | 1) as f32 * 2f32.powi(-33));
    let e = if upper { -(-a).ln_1p() } else { -a.ln() };
    -e.ln()
}

/// The words the variates are checked at: the very top of the range (`!x < 512`, `a < 2^-23`,
/// where `1 - a` in f32 is 1 or nearly), both sides of `a = 2^-12` and of `a = 2^-6`, both sides
/// of the branch at `2^31`, the bottom, and 4096 spread over the whole range.
fn gumbel_words() -> Vec<(&'static str, Vec<u32>)> {
    let around = |y: u32| (y - 256..y + 256).map(|y| !y).collect::<Vec<_>>();
    vec![
        ("the top", (0..512).map(|y: u32| !y).collect()),
        ("a near 2^-12", around(1 << 20)),
        ("a near 2^-6", around(1 << 26)),
        (
            "the branch",
            vec![(1 << 31) - 2, (1 << 31) - 1, 1 << 31, (1 << 31) + 1],
        ),
        ("the bottom", (0..256).collect()),
        (
            "the whole range",
            (0..4096u32).map(|i| i.wrapping_mul(0x0010_0001)).collect(),
        ),
    ]
}

/// The Gumbel variates of the words [`gumbel_words`] lists against [`gumbel_f64`], on `device`:
/// the reference's and, on a CubeCL device, the kernel's own function (run alone, so on every
/// CubeCL device), which must give the reference's bits where the sampling kernel runs. A change
/// in how a compiler lowers the functions they use shows here first.
fn gumbel_variates_match_f64(device: &Device) {
    let stretches = gumbel_words();
    let words: Vec<u32> = stretches.iter().flat_map(|(_, w)| w.clone()).collect();
    let (reference, kernel) = gumbel_variates(device, &words);
    let mut failures = Vec::new();
    for (who, got) in [("reference", Some(&reference)), ("kernel", kernel.as_ref())] {
        let Some(got) = got else { continue };
        let (mut at, mut worst, mut parts) = (0, (0f64, 0u32), Vec::new());
        for (name, stretch) in &stretches {
            let mut here = 0f64;
            for (&x, &g) in stretch.iter().zip(&got[at..at + stretch.len()]) {
                let error = match g.is_finite() {
                    true => (g as f64 - gumbel_f64(x)).abs(),
                    false => f64::INFINITY,
                };
                here = here.max(error);
                if error > worst.0 {
                    worst = (error, x);
                }
            }
            at += stretch.len();
            parts.push(format!("{name} {here:.3e}"));
        }
        println!(
            "gumbel variates, the {who}'s: max abs {:.3e} at {:#010x} ({})",
            worst.0,
            worst.1,
            parts.join(", ")
        );
        if worst.0 > GUMBEL {
            failures.push(format!(
                "the {who}'s g off by {:e} at {:#010x}",
                worst.0, worst.1
            ));
        }
    }
    if let Some(kernel) = &kernel {
        let differ = (kernel.iter().zip(&reference))
            .filter(|(k, r)| k.to_bits() != r.to_bits())
            .count();
        // Where the sampling kernel runs, its tokens are held to the reference's.
        let runs = kernel_takes(device, DType::F32, [1, 1]);
        println!(
            "gumbel variates: the kernel's and the reference's differ at {differ} words (the \
             sampling kernel runs here: {runs})"
        );
        if differ > 0 && runs {
            failures.push(format!(
                "the kernel's variates differ from the reference's at {differ} words"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

#[cfg(feature = "flex")]
#[test]
fn gumbel_variates_match_f64_on_flex() {
    gumbel_variates_match_f64(&Device::flex());
}

/// Seeds whose first call puts a token's word in the top 128 of the range (`!x < 128`, so
/// `a < 2^-25`), found by a search on the host: `(seed, the token, the token the definition
/// draws)` for logits 0 but -30 at that token, at temperature 1. The definition's `g` there,
/// 17.6 to 17.9, leaves the token 12 or more below the others' scores; through CubeCL's `log1p`
/// (`ln(1 + x)`, so `ln 1 = 0` and `g = +inf`) it won.
const TOP_WORDS: [(u64, usize, i64); 2] = [(2_871_038, 2, 3), (16_390_493, 0, 1)];

/// The seeds of [`TOP_WORDS`] draw the definition's tokens, with their log-probabilities.
fn top_words_draw_as_defined(device: &Device, reference: bool) {
    let _serial = serial();
    for (seed, at, want) in TOP_WORDS {
        let mut values = vec![0.0f32; 4];
        values[at] = -30.0;
        let options = SampleOptions {
            seed,
            ..SampleOptions::default()
        };
        let out = match reference {
            true => run_reference(device, &values, [1, 4], DType::F32, &options),
            false => run(device, &values, [1, 4], DType::F32, &options),
        };
        assert_eq!(
            out.tokens[0], want,
            "seed {seed}: drew token {} (the one at -30 is {at})",
            out.tokens[0]
        );
        // At temperature 1 with nothing cut, all three are 0 - ln(3 + e^-30).
        let exact = -(3.0f64 + (-30.0f64).exp()).ln();
        for got in [out.logprob[0], out.tempered[0], out.sampling[0]] {
            assert!(
                (got as f64 - exact).abs() <= LOGPROB_EXACT,
                "seed {seed}: {got} against {exact}"
            );
        }
    }
    println!("the top words' seeds draw the definition's tokens");
}

#[cfg(feature = "flex")]
#[test]
fn top_words_draw_as_defined_on_flex() {
    top_words_draw_as_defined(&Device::flex(), false);
}

/// The kernel on the CubeCL device `DASH_OPS_DEVICE` names, whichever runtime; where the device
/// lacks what it asks, the reference on that device, held to the same checks.
#[cfg(feature = "cubecl")]
mod cube {
    use super::*;
    use dash_ops::sample::kernel_capacity;

    fn device() -> Device {
        common::cube_device()
    }

    #[test]
    fn draws_follow_the_kept_sets_on_cubecl() {
        distributions(&device());
    }

    #[test]
    fn greedy_is_burns_argmax_on_cubecl() {
        greedy_is_argmax(&device(), false);
    }

    #[test]
    fn draws_are_reproducible_on_cubecl() {
        reproducible(&device());
    }

    #[test]
    fn edges_on_cubecl() {
        edges(&device(), false);
    }

    #[test]
    fn identities_hold_bit_for_bit_on_cubecl() {
        identities(&device());
    }

    #[test]
    fn top_p_routes_on_cubecl() {
        top_p_routes(&device(), kernel_capacity(&device()));
    }

    #[test]
    fn gumbel_variates_match_f64_on_cubecl() {
        gumbel_variates_match_f64(&device());
    }

    #[test]
    fn top_words_draw_as_defined_on_cubecl() {
        top_words_draw_as_defined(&device(), false);
        top_words_draw_as_defined(&device(), true);
    }

    /// Fewer draws where the reference runs: its sort reads the rows back to the host.
    #[test]
    fn tied_groups_on_cubecl() {
        let device = device();
        let calls = match kernel_takes(&device, DType::F32, [256, 20_000]) {
            true => 16,
            false => 4,
        };
        tied_groups(&device, calls);
    }

    /// Whether the kernel takes `[b, v]` f32 logits on the device; where it does not, the
    /// comparisons below would hold the reference to itself, and are skipped, saying so.
    fn kernel_runs(dims: [usize; 2]) -> bool {
        let takes = kernel_takes(&device(), DType::F32, dims);
        if !takes {
            println!(
                "skipped: the kernel does not run on {}",
                common::describe(&device())
            );
        }
        takes
    }

    /// The modes the comparisons below run.
    fn modes() -> Vec<(&'static str, SampleOptions)> {
        let d = SampleOptions::default();
        vec![
            ("greedy", SampleOptions::greedy()),
            (
                "temperature 1",
                SampleOptions {
                    seed: 1,
                    ..d.clone()
                },
            ),
            (
                "temperature 0.7, top-p 0.95",
                SampleOptions {
                    temperature: 0.7,
                    top_p: 0.95,
                    seed: 2,
                    ..d.clone()
                },
            ),
            (
                "top-k 50",
                SampleOptions {
                    top_k: 50,
                    seed: 3,
                    ..d.clone()
                },
            ),
            (
                "top-k 50, top-p 0.9, T 0.8",
                SampleOptions {
                    top_k: 50,
                    top_p: 0.9,
                    temperature: 0.8,
                    seed: 4,
                    ..d.clone()
                },
            ),
            (
                "top-k 2000",
                SampleOptions {
                    top_k: 2000,
                    temperature: 1.2,
                    seed: 5,
                    ..d.clone()
                },
            ),
            (
                "top-p 0.5, masked",
                SampleOptions {
                    top_p: 0.5,
                    masked: vec![0, 3, 999, 40_000],
                    seed: 6,
                    ..d.clone()
                },
            ),
        ]
    }

    /// The kernel against the reference on the same device, token for token: the same tokens, the
    /// three log-probabilities within the stated bounds. Returns the largest differences, and the
    /// kernel's outputs.
    fn against_reference(
        label: &str,
        values: &[f32],
        dims: [usize; 2],
        dtype: DType,
        options: &SampleOptions,
    ) -> ([f64; 3], Out) {
        let device = device();
        let options = SampleOptions {
            masked: options
                .masked
                .iter()
                .copied()
                .filter(|&id| (id as usize) < dims[1])
                .collect(),
            ..options.clone()
        };
        let got = run(&device, values, dims, dtype, &options);
        let want = run_reference(&device, values, dims, dtype, &options);
        assert_eq!(
            got.tokens, want.tokens,
            "{label} {dims:?} {dtype:?}: tokens differ"
        );
        let diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| (x as f64 - y as f64).abs())
                .fold(0.0, f64::max)
        };
        let d = [
            diff(&got.logprob, &want.logprob),
            diff(&got.sampling, &want.sampling),
            diff(&got.tempered, &want.tempered),
        ];
        println!(
            "{label:40} {dims:?} {dtype:?}: the same tokens; logprob max abs {:.3e}, sampling_logprob max abs {:.3e}, tempered_logprob max abs {:.3e}",
            d[0], d[1], d[2]
        );
        (d, got)
    }

    /// The largest of each of three differences.
    fn widen(worst: &mut [f64; 3], d: [f64; 3]) {
        for (w, d) in worst.iter_mut().zip(d) {
            *w = w.max(d);
        }
    }

    /// Kernel and reference on rows of many shapes (vocabularies that no vector width or cube
    /// size divides, rows that fit the candidates and rows that do not), in f32, bf16 and f16.
    #[test]
    fn the_kernel_draws_the_references_tokens() {
        let _serial = serial();
        if !kernel_runs([8, 1031]) {
            return;
        }
        let device = device();
        println!(
            "{}  capacity {}",
            common::describe(&device),
            kernel_capacity(&device)
        );
        let mut worst = [0.0f64; 3];
        for (seed, dims, scale, lift) in [
            (1, [1, 7], 1.0, 0.0),
            (2, [3, 1000], 2.0, 6.0),
            (3, [8, 1031], 1.0, 0.0),
            (4, [5, 5000], 0.5, 0.0),
            (5, [4, 20_000], 3.0, 10.0),
            (6, [2, 60_001], 1.0, 0.0),
        ] {
            let values = random_logits(seed, dims, scale, lift);
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                if !common::has(&device, dtype) {
                    continue;
                }
                for (label, options) in modes() {
                    let (d, _) = against_reference(label, &values, dims, dtype, &options);
                    widen(&mut worst, d);
                }
            }
        }
        println!(
            "kernel vs reference: the same tokens; logprob max abs {:.3e}, sampling_logprob max abs {:.3e}, tempered_logprob max abs {:.3e}",
            worst[0], worst[1], worst[2]
        );
        assert!(worst.iter().all(|&w| w <= KERNEL_REFERENCE), "{worst:?}");
    }

    /// The decoding shape: `V = 248320`, batches of 8 and 32, bf16 and f32 logits, peaked (a
    /// model's) and flat (the refinement passes under top-p); the kernel's `tempered_logprob`
    /// also against its f64 value.
    #[test]
    fn the_kernel_draws_the_references_tokens_at_the_decoding_shape() {
        let _serial = serial();
        if !kernel_runs([32, 248_320]) {
            return;
        }
        let v = 248_320;
        let (mut worst, mut tempered) = ([0.0f64; 3], 0.0f64);
        for b in [8, 32] {
            let dims = [b, v];
            for (kind, scale, lift) in [("peaked", 3.0, 12.0), ("flat", 1.0, 0.0)] {
                let values = random_logits(b as u64, dims, scale, lift);
                for dtype in [DType::BF16, DType::F32] {
                    for (label, options) in modes() {
                        let label = format!("{kind} {label}");
                        let (d, got) = against_reference(&label, &values, dims, dtype, &options);
                        widen(&mut worst, d);
                        // Every mode's masked ids lie below V.
                        for r in 0..b {
                            let row = &values[r * v..(r + 1) * v];
                            let exact = tempered_log_softmax(row, &options)[got.tokens[r] as usize];
                            tempered = tempered.max((got.tempered[r] as f64 - exact).abs());
                        }
                    }
                }
            }
        }
        println!(
            "V 248320: kernel vs reference, the same tokens; logprob max abs {:.3e}, sampling_logprob max abs {:.3e}, tempered_logprob max abs {:.3e}",
            worst[0], worst[1], worst[2]
        );
        println!("V 248320: the kernel's tempered_logprob against f64, max abs {tempered:.3e}");
        assert!(worst.iter().all(|&w| w <= KERNEL_REFERENCE), "{worst:?}");
        assert!(
            tempered <= LOGPROB_LARGE,
            "tempered_logprob off by {tempered:e}"
        );
    }
}
