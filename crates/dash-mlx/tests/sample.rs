//! Sampling the next token, against dash-ops' definition and its random numbers, with dash-ops'
//! bounds.
//!
//! As dash-ops' tests: the definition computed once more in f64, every draw held to it (the token
//! one of the kept set, `logprob` the log-softmax of the raw logits over the unmasked ids,
//! `sampling_logprob` the log-probability under the kept set's tempered softmax, and here
//! `tempered_logprob` the tempered log-softmax over the unmasked ids); the distribution held to
//! the kept set's probabilities by chi-square tests over many draws (each call draws as many rows
//! of one row of logits, each call with its own offset); the cuts' ties and limits; greedy
//! decoding against MLX's argmax in every dtype; the same bits from run to run; the Gumbel
//! variates against their values in f64; the seeds whose words lie at the very top.
//!
//! Then what makes the draws dash-ops' own: Philox's words on the device against the generator on
//! the host (checked against Random123's known answers) word by word, and at the decoding shape
//! the token drawn against the definition's, `argmax (s + g)` over the kept set from the host's
//! words in f64, wherever the two best scores are not within rounding of each other.
//!
//! Errors are `max |got - want|`; each bound is dash-ops'.
//!
//!   cargo test -p dash-mlx --test sample -- --nocapture
#![cfg(target_os = "macos")]

use dash_mlx::sample::{SampleOptions, Sampled, gumbel_variates, philox_words, sample};
use mlx_rs::{Array, Dtype};
use std::sync::{Mutex, MutexGuard};

/// One test at a time: they share the GPU's default stream.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// dash-ops' bound on the log-probabilities against f64 on rows of a few tokens.
const LOGPROB_EXACT: f64 = 1e-6;
/// dash-ops' bound on larger rows, `V` from 1000 to 100000.
const LOGPROB_LARGE: f64 = 2e-6;
/// dash-ops' bound on the Gumbel variates against f64.
const GUMBEL: f64 = 6e-6;

/// Philox4x32-10 on the host: dash-ops' generator, the oracle for the device's words.
fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (mut c, mut k) = (counter, key);
    for round in 0..10 {
        if round > 0 {
            k = [
                k[0].wrapping_add(0x9E37_79B9),
                k[1].wrapping_add(0xBB67_AE85),
            ];
        }
        let p0 = u64::from(0xD251_1F53u32) * u64::from(c[0]);
        let p1 = u64::from(0xCD9E_8D57u32) * u64::from(c[2]);
        c = [
            (p1 >> 32) as u32 ^ c[1] ^ k[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ c[3] ^ k[1],
            p0 as u32,
        ];
    }
    c
}

/// The word of token `i` of row `row`: counter `(i / 4, row, offset)`, key `seed`, word `i % 4`.
fn word(seed: u64, offset: u64, row: u32, i: usize) -> u32 {
    let key = [seed as u32, (seed >> 32) as u32];
    philox4x32_10(
        [(i / 4) as u32, row, offset as u32, (offset >> 32) as u32],
        key,
    )[i % 4]
}

/// The definition's `g` in f64 for the word `x`, `a` rounded to f32 as the definition rounds it.
fn gumbel_f64(x: u32) -> f64 {
    let upper = x >> 31 == 1;
    let y = if upper { !x } else { x };
    let a = f64::from(((y << 1) | 1) as f32 * 2f32.powi(-33));
    let e = if upper { -(-a).ln_1p() } else { -a.ln() };
    -e.ln()
}

/// One call's outputs.
#[derive(Debug, Clone)]
struct Out {
    tokens: Vec<i32>,
    logprob: Vec<f32>,
    tempered: Vec<f32>,
    sampling: Vec<f32>,
}

impl Out {
    fn of(s: Sampled) -> Self {
        let f = |a: &Array| a.as_slice::<f32>().to_vec();
        Out {
            tokens: s.tokens.as_slice::<i32>().to_vec(),
            logprob: f(&s.logprob),
            tempered: f(&s.tempered_logprob),
            sampling: f(&s.sampling_logprob),
        }
    }

    fn bits(&self) -> (Vec<i32>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let b = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect();
        (
            self.tokens.clone(),
            b(&self.logprob),
            b(&self.tempered),
            b(&self.sampling),
        )
    }
}

fn run(values: &[f32], [b, v]: [usize; 2], dtype: Dtype, options: &SampleOptions) -> Out {
    let logits = Array::from_slice(values, &[b as i32, v as i32])
        .as_dtype(dtype)
        .unwrap();
    let out = sample(&logits, options).unwrap();
    assert_eq!(out.tokens.dtype(), Dtype::Int32);
    for x in [&out.logprob, &out.tempered_logprob, &out.sampling_logprob] {
        assert_eq!(x.dtype(), Dtype::Float32);
    }
    Out::of(out)
}

/// A seeded generator: xorshift64* and Box-Muller, as dash-ops' tests have it.
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

fn bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

fn f16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// `b` rows of `v` logits `scale * N(0, 1)`, a few of each row's tokens raised by up to `lift`,
/// rounded to bf16, as dash-ops' tests make them.
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

/// The definition in f64 over one row.
struct Exact {
    /// The kept ids in increasing order, and their probabilities.
    kept: Vec<usize>,
    prob: Vec<f64>,
    /// The log-softmax of the raw logits over the unmasked ids, per token (`-inf` when masked).
    log_softmax: Vec<f64>,
    /// The same of the tempered logits (0 at the argmax when greedy, as `sampling_logprob`).
    tempered: Vec<f64>,
    tau: f64,
    m: f64,
    /// Whether top-p's crossing lies within [`NEAR`] of the row's kept mass of either end of the
    /// tie group it falls in, where the f32 sums may round it into the next group.
    near_boundary: bool,
}

/// The margin, as a share of the kept mass, within which top-p's crossing is near a tie group's
/// end: about three times the f32 sums' error measured at 248,320 tokens.
const NEAR: f64 = 2e-6;

impl Exact {
    fn of(z: &[f32], options: &SampleOptions) -> Self {
        let v = z.len();
        let masked = |i: usize| options.masked.contains(&(i as u32));
        let unmasked: Vec<usize> = (0..v).filter(|&i| !masked(i)).collect();
        let m = unmasked
            .iter()
            .map(|&i| z[i] as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let soft = |tau: f64| -> Vec<f64> {
            let lse = unmasked
                .iter()
                .map(|&i| ((z[i] as f64 - m) * tau).exp())
                .sum::<f64>()
                .ln();
            (0..v)
                .map(|i| match masked(i) {
                    true => f64::NEG_INFINITY,
                    false => (z[i] as f64 - m) * tau - lse,
                })
                .collect()
        };
        let log_softmax = soft(1.0);
        if options.temperature == 0.0 {
            let top = unmasked
                .iter()
                .copied()
                .find(|&i| z[i] as f64 == m)
                .unwrap();
            let mut tempered = vec![f64::NEG_INFINITY; v];
            tempered[top] = 0.0;
            return Exact {
                kept: vec![top],
                prob: vec![1.0],
                log_softmax,
                tempered,
                tau: f64::INFINITY,
                m,
                near_boundary: false,
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
        let mut near_boundary = false;
        if options.top_p < 1.0 {
            let mut by_logit = kept.clone();
            by_logit.sort_by(|&a, &b| z[b].partial_cmp(&z[a]).unwrap());
            let total: f64 = kept.iter().map(|&i| w(i)).sum();
            let target = options.top_p as f64 * total;
            let mut cumulative = Vec::with_capacity(by_logit.len());
            let mut sum = 0.0;
            for &i in &by_logit {
                sum += w(i);
                cumulative.push(sum);
            }
            let at = cumulative
                .iter()
                .position(|&c| c >= target)
                .unwrap_or(by_logit.len() - 1);
            let cut = z[by_logit[at]];
            // The tie group the crossing falls in, `start..end` of the sorted kept tokens.
            let start = by_logit.iter().position(|&i| z[i] == cut).unwrap();
            let end = start
                + by_logit[start..]
                    .iter()
                    .take_while(|&&i| z[i] == cut)
                    .count();
            let before = if start == 0 {
                0.0
            } else {
                cumulative[start - 1]
            };
            near_boundary =
                target - before <= NEAR * total || cumulative[end - 1] - target <= NEAR * total;
            kept.retain(|&i| z[i] >= cut);
        }
        let total: f64 = kept.iter().map(|&i| w(i)).sum();
        let prob = kept.iter().map(|&i| w(i) / total).collect();
        Exact {
            kept,
            prob,
            log_softmax,
            tempered: soft(tau),
            tau,
            m,
            near_boundary,
        }
    }

    fn prob_of(&self, token: usize) -> Option<f64> {
        self.kept
            .iter()
            .position(|&i| i == token)
            .map(|at| self.prob[at])
    }

    /// The three log-probabilities' errors at `row` of `out`, its token `t`.
    fn errors(&self, out: &Out, row: usize) -> [f64; 3] {
        let t = out.tokens[row] as usize;
        let p = self.prob_of(t).expect("a kept token");
        [
            (out.logprob[row] as f64 - self.log_softmax[t]).abs(),
            (out.tempered[row] as f64 - self.tempered[t]).abs(),
            (out.sampling[row] as f64 - p.ln()).abs(),
        ]
    }
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

/// dash-ops' significance level: a correct sampler fails one test with probability 1e-4.
const ALPHA: f64 = 1e-4;

/// The chi-square quantile at `1 - ALPHA` (Wilson and Hilferty from 3 degrees of freedom).
fn chi_square_bound(df: usize) -> f64 {
    match df {
        0 => 0.0,
        1 => 15.137,
        2 => -2.0 * ALPHA.ln(),
        _ => {
            let z = 3.719_016;
            let k = df as f64;
            let c = 2.0 / (9.0 * k);
            k * (1.0 - c + z * c.sqrt()).powi(3)
        }
    }
}

/// `ln p` for probabilities `p`.
fn logits_of(p: &[f64]) -> Vec<f32> {
    p.iter().map(|&p| p.ln() as f32).collect()
}

/// dash-ops' distribution cases, and a sixteenth on the route its coming one takes: each a row,
/// options and the kept set the definition must give.
fn distribution_cases() -> Vec<(&'static str, Vec<f32>, SampleOptions, Vec<usize>)> {
    let o = SampleOptions::default();
    let twelve = vec![
        2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0, 1.5, 0.25, -0.25, 3.0,
    ];
    let ties = vec![5.0, 3.0, 3.0, 3.0, 1.0, 0.0, -1.0, 4.0, 2.0, 2.0, 0.5, -2.0];
    let two_tops = vec![1.0, 4.0, 2.0, 4.0, 0.0, -1.0];
    let staircase = logits_of(&[0.4, 0.2, 0.15, 0.1, 0.08, 0.04, 0.02, 0.01]);
    let flat_tail = logits_of(&[0.4, 0.2, 0.2, 0.2]);
    let both = logits_of(&[0.4, 0.25, 0.15, 0.1, 0.05, 0.05]);
    let with = |f: &dyn Fn(&mut SampleOptions)| {
        let mut o = o.clone();
        f(&mut o);
        o
    };
    vec![
        (
            "temperature 0.7",
            twelve.clone(),
            with(&|o| {
                o.temperature = 0.7;
                o.seed = 1
            }),
            (0..12).collect(),
        ),
        (
            "temperature 1.5",
            twelve.clone(),
            with(&|o| {
                o.temperature = 1.5;
                o.seed = 2
            }),
            (0..12).collect(),
        ),
        (
            "top-k 3, ties kept",
            ties,
            with(&|o| {
                o.top_k = 3;
                o.seed = 3
            }),
            vec![0, 1, 2, 3, 7],
        ),
        (
            "top-k 1, two maxima",
            two_tops.clone(),
            with(&|o| {
                o.top_k = 1;
                o.seed = 4
            }),
            vec![1, 3],
        ),
        (
            "top-k V",
            twelve.clone(),
            with(&|o| {
                o.top_k = 12;
                o.seed = 5
            }),
            (0..12).collect(),
        ),
        (
            "top-k above V",
            twelve.clone(),
            with(&|o| {
                o.top_k = 1000;
                o.temperature = 0.8;
                o.seed = 6
            }),
            (0..12).collect(),
        ),
        (
            "top-p 0.7",
            staircase.clone(),
            with(&|o| {
                o.top_p = 0.7;
                o.seed = 7
            }),
            vec![0, 1, 2],
        ),
        (
            "top-p 0.7, temperature 0.5",
            staircase.clone(),
            with(&|o| {
                o.top_p = 0.7;
                o.temperature = 0.5;
                o.seed = 8
            }),
            vec![0, 1],
        ),
        (
            "top-p 0.5, ties at the cut",
            flat_tail,
            with(&|o| {
                o.top_p = 0.5;
                o.seed = 9
            }),
            vec![0, 1, 2, 3],
        ),
        (
            "top-p 1e-6",
            twelve.clone(),
            with(&|o| {
                o.top_p = 1e-6;
                o.seed = 10
            }),
            vec![11],
        ),
        (
            "top-p 0, two maxima",
            two_tops,
            with(&|o| {
                o.top_p = 0.0;
                o.seed = 11
            }),
            vec![1, 3],
        ),
        (
            "top-p 1",
            staircase.clone(),
            with(&|o| {
                o.top_p = 1.0;
                o.seed = 12
            }),
            (0..8).collect(),
        ),
        (
            "top-k 4 then top-p 0.7",
            both,
            with(&|o| {
                o.top_k = 4;
                o.top_p = 0.7;
                o.seed = 13
            }),
            vec![0, 1],
        ),
        (
            "masked argmax, top-k 2",
            twelve.clone(),
            with(&|o| {
                o.top_k = 2;
                o.masked = vec![11, 7];
                o.seed = 14
            }),
            vec![0, 8],
        ),
        (
            "masked, top-p 0.8",
            staircase,
            with(&|o| {
                o.top_p = 0.8;
                o.masked = vec![1, 5];
                o.temperature = 1.2;
                o.seed = 15
            }),
            vec![0, 2, 3],
        ),
        // The route where the tempered sum over the unmasked ids feeds `tempered_logprob` alone:
        // masked ids under top-k at a temperature.
        (
            "masked, top-k 3, temperature 0.8",
            twelve,
            with(&|o| {
                o.top_k = 3;
                o.masked = vec![11, 0];
                o.temperature = 0.8;
                o.seed = 16
            }),
            vec![1, 2, 8],
        ),
    ]
}

#[test]
fn the_definition_keeps_what_the_cases_say() {
    for (label, z, options, kept) in distribution_cases() {
        assert_eq!(Exact::of(&z, &options).kept, kept, "{label}");
    }
}

/// dash-ops' draws per case: 2048 rows a call, 48 calls, each with its own offset.
const ROWS: usize = 2048;
const CALLS: usize = 48;

#[test]
fn draws_follow_the_kept_sets() {
    let _serial = serial();
    for (label, z, options, _) in distribution_cases() {
        let v = z.len();
        let exact = Exact::of(&z, &options);
        let mut counts = vec![0u64; exact.kept.len()];
        let mut worst = [0.0f64; 3];
        let values: Vec<f32> = (0..ROWS).flat_map(|_| z.iter().copied()).collect();
        for call in 0..CALLS {
            let options = SampleOptions {
                offset: call as u64,
                ..options.clone()
            };
            let out = run(&values, [ROWS, v], Dtype::Float32, &options);
            for r in 0..ROWS {
                let t = out.tokens[r] as usize;
                let at = exact
                    .kept
                    .iter()
                    .position(|&i| i == t)
                    .unwrap_or_else(|| panic!("{label}: drew {t}, outside {:?}", exact.kept));
                counts[at] += 1;
                for (w, e) in worst.iter_mut().zip(exact.errors(&out, r)) {
                    *w = w.max(e);
                }
            }
        }
        let n = (ROWS * CALLS) as u64;
        let (statistic, df) = chi_square(&counts, &exact.prob, n);
        let bound = chi_square_bound(df);
        println!(
            "{label:30} {n} draws: chi-square {statistic:.2} on {df} df (bound {bound:.2}); \
             logprob {:.3e}, tempered {:.3e}, sampling {:.3e}",
            worst[0], worst[1], worst[2]
        );
        assert!(
            worst.iter().all(|&w| w <= LOGPROB_EXACT),
            "{label}: {worst:?}"
        );
        assert!(
            df == 0 || statistic <= bound,
            "{label}: chi-square {statistic} above {bound}"
        );
        for (at, &c) in counts.iter().enumerate() {
            if exact.prob[at] * n as f64 >= 20.0 {
                assert!(c > 0, "{label}: kept token {} never drawn", exact.kept[at]);
            }
        }
    }
}

/// Greedy decoding is MLX's argmax of the row with the masked logits at `-inf`, in every dtype,
/// ties (bf16 has many) to the lowest id; `logprob` the log-softmax, the other two 0.
#[test]
fn greedy_is_the_argmax() {
    let _serial = serial();
    for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
        for (dims, masked) in [
            ([5, 1000], vec![]),
            ([3, 1031], vec![0, 17, 1030]),
            ([1, 3], vec![]),
            ([4, 40_000], vec![5, 39_999]),
        ] {
            let [b, v] = dims;
            let mut rng = Rng::new((b * v) as u64);
            let values: Vec<f32> = (0..b * v)
                .map(|_| f16(bf16((rng.normal() * 2.0).round() as f32 / 2.0)))
                .collect();
            let options = SampleOptions {
                masked: masked.clone(),
                ..SampleOptions::greedy()
            };
            let out = run(&values, dims, dtype, &options);
            let mut worst = 0.0f64;
            for row in 0..b {
                let mut z = values[row * v..(row + 1) * v].to_vec();
                for &id in &masked {
                    z[id as usize] = f32::NEG_INFINITY;
                }
                let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let argmax = z.iter().position(|&x| x == m).unwrap() as i32;
                assert_eq!(out.tokens[row], argmax, "{dtype:?} {dims:?} row {row}");
                let exact = Exact::of(&values[row * v..(row + 1) * v], &options);
                worst = worst.max(exact.errors(&out, row)[0]);
                assert_eq!((out.tempered[row], out.sampling[row]), (0.0, 0.0));
            }
            println!("greedy {dtype:?} {dims:?}: the argmax; logprob max abs {worst:.3e}");
            assert!(worst <= LOGPROB_LARGE, "{dtype:?} {dims:?}: {worst:e}");
        }
    }
}

/// At temperature 1 with nothing cut the three log-probabilities are one value, to the bit, with
/// and without masked ids.
#[test]
fn the_three_agree_at_temperature_one() {
    let _serial = serial();
    let dims = [8, 50_000];
    let values = random_logits(3, dims, 1.0, 4.0);
    for masked in [vec![], vec![0, 7, 49_999]] {
        let options = SampleOptions {
            seed: 5,
            masked,
            ..SampleOptions::default()
        };
        let out = run(&values, dims, Dtype::Float32, &options);
        let (_, logprob, tempered, sampling) = out.bits();
        assert_eq!(logprob, tempered);
        assert_eq!(logprob, sampling);
    }
}

/// The same seed and offset draw the same bits; another offset or seed, other tokens; the rows of
/// one call draw apart.
#[test]
fn draws_are_reproducible() {
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
    let first = run(&values, dims, Dtype::Float32, &options);
    assert!(first.bits() == run(&values, dims, Dtype::Float32, &options).bits());
    let differ = |other: &Out| {
        first
            .tokens
            .iter()
            .zip(&other.tokens)
            .filter(|(a, b)| a != b)
            .count()
    };
    let next = run(
        &values,
        dims,
        Dtype::Float32,
        &SampleOptions {
            offset: 78,
            ..options.clone()
        },
    );
    let reseeded = run(
        &values,
        dims,
        Dtype::Float32,
        &SampleOptions {
            seed: 4321,
            ..options.clone()
        },
    );
    println!(
        "reproducible: offset + 1 changes {} of 64 tokens, another seed {}",
        differ(&next),
        differ(&reseeded)
    );
    assert!(differ(&next) >= 56 && differ(&reseeded) >= 56);
    let repeated: Vec<f32> = (0..64)
        .flat_map(|_| values[..5000].iter().copied())
        .collect();
    let out = run(&repeated, dims, Dtype::Float32, &options);
    let mut distinct = out.tokens.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() >= 56,
        "64 copies drew {} tokens",
        distinct.len()
    );
}

/// dash-ops' edges: a vocabulary of one; every token but one masked; huge, tiny and `-inf`
/// logits; a temperature near 0 against 0; a row with no finite logit.
#[test]
fn edges() {
    let _serial = serial();
    let go = |values: &[f32], dims: [usize; 2], options: &SampleOptions| {
        run(values, dims, Dtype::Float32, options)
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
            out.tokens[0] != 2 && out.logprob[0].is_finite() && out.sampling[0].is_finite(),
            "{options:?}: {out:?}"
        );
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
        let mut sorted = values[row * dims[1]..(row + 1) * dims[1]].to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if sorted[0] - sorted[1] > 1e-3 {
            assert_eq!(cold.tokens[row], greedy.tokens[row], "row {row}");
            assert!(cold.sampling[row].abs() < 1e-6, "row {row}");
        }
    }
}

/// dash-ops' tied groups: four groups of 5000 tokens at one logit each, the cuts inside a group;
/// the groups' draws against their probabilities.
#[test]
fn tied_groups() {
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
        let mut worst = [0.0f64; 3];
        for call in 0..4 {
            let options = SampleOptions {
                offset: call as u64,
                ..options.clone()
            };
            let out = run(&values, [rows, z.len()], Dtype::Float32, &options);
            for r in 0..rows {
                let t = out.tokens[r] as usize;
                assert!(
                    exact.prob_of(t).is_some(),
                    "{label}: token {t} outside the kept set"
                );
                for (w, e) in worst.iter_mut().zip(exact.errors(&out, r)) {
                    *w = w.max(e);
                }
                groups[t / group] += 1;
            }
        }
        let n = (rows * 4) as u64;
        let (statistic, df) = chi_square(&groups, &group_prob, n);
        let bound = chi_square_bound(df);
        println!(
            "{label:38} groups {groups:?} of {group_prob:.3?}: chi-square {statistic:.2} on {df} df \
             (bound {bound:.2}); logprob {:.3e}, tempered {:.3e}, sampling {:.3e}",
            worst[0], worst[1], worst[2]
        );
        assert!(
            df == 0 || statistic <= bound,
            "{label}: {statistic} above {bound}"
        );
        assert!(
            worst.iter().all(|&w| w <= LOGPROB_LARGE),
            "{label}: {worst:?}"
        );
    }
}

/// Random123's known answers for the host generator, then the device's words against it, token
/// by token, with seeds and offsets past 2^32, rows past the first and a partial last group.
#[test]
fn the_devices_words_are_philoxs() {
    let _serial = serial();
    let cases: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
        (
            [0; 4],
            [0; 2],
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
        ),
        (
            [u32::MAX; 4],
            [u32::MAX; 2],
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
        ),
        (
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0],
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
        ),
    ];
    for (counter, key, want) in cases {
        assert_eq!(philox4x32_10(counter, key), want);
    }
    for (seed, offset) in [
        (0, 0),
        (1234, 77),
        ((7 << 32) | 5, (3 << 32) | 9),
        (u64::MAX, u64::MAX),
    ] {
        let (b, v) = (3usize, 1003usize);
        let got = philox_words(seed, offset, [b as i32, v as i32]).unwrap();
        for row in 0..b {
            for i in 0..v {
                assert_eq!(
                    got[row * v + i],
                    word(seed, offset, row as u32, i),
                    "seed {seed:#x} offset {offset:#x} row {row} token {i}"
                );
            }
        }
    }
    println!("the device's words are Philox4x32-10's, token for token");
}

/// dash-ops' words for the variates: the very top of the range, both sides of 2^-12 and 2^-6 and
/// of the branch at 2^31, the bottom, and 4096 over the whole range.
#[test]
fn gumbel_variates_match_f64() {
    let _serial = serial();
    let around = |y: u32| (y - 256..y + 256).map(|y| !y).collect::<Vec<_>>();
    let stretches: Vec<(&str, Vec<u32>)> = vec![
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
    ];
    let words: Vec<u32> = stretches.iter().flat_map(|(_, w)| w.clone()).collect();
    let got = gumbel_variates(&words).unwrap();
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
        "gumbel variates: max abs {:.3e} at {:#010x} ({})",
        worst.0,
        worst.1,
        parts.join(", ")
    );
    assert!(
        worst.0 <= GUMBEL,
        "g off by {:e} at {:#010x}",
        worst.0,
        worst.1
    );
}

/// dash-ops' seeds whose first call puts a token's word in the top 128 of the range: the
/// definition's tokens, with their log-probabilities.
#[test]
fn top_words_draw_as_defined() {
    let _serial = serial();
    for (seed, at, want) in [(2_871_038u64, 2usize, 3i32), (16_390_493, 0, 1)] {
        let mut values = vec![0.0f32; 4];
        values[at] = -30.0;
        let options = SampleOptions {
            seed,
            ..SampleOptions::default()
        };
        let out = run(&values, [1, 4], Dtype::Float32, &options);
        assert_eq!(out.tokens[0], want, "seed {seed}");
        let exact = -(3.0f64 + (-30.0f64).exp()).ln();
        for got in [out.logprob[0], out.tempered[0], out.sampling[0]] {
            assert!(
                (got as f64 - exact).abs() <= LOGPROB_EXACT,
                "seed {seed}: {got}"
            );
        }
    }
}

/// At the decoding shape and below, each draw against the definition from the host's words: the
/// token `argmax (s + g)` over the kept set, `s` and `g` in f64, exactly on every row but those
/// marked, which are reported: a top-p crossing near a tie group's end (the kept set may differ)
/// or two best scores within 1e-4 (the f32 roundings may order them either way, as between
/// dash-ops' kernel and reference); the log-probabilities within dash-ops' bound for large rows.
#[test]
fn draws_are_the_definitions_at_scale() {
    let _serial = serial();
    let cases = [
        (
            [8usize, 248_320usize],
            SampleOptions {
                temperature: 0.8,
                top_p: 0.9,
                seed: 3,
                offset: 11,
                ..SampleOptions::default()
            },
        ),
        (
            [4, 248_320],
            SampleOptions {
                top_k: 50,
                seed: 4,
                ..SampleOptions::default()
            },
        ),
        (
            [8, 100_000],
            SampleOptions {
                temperature: 1.2,
                masked: vec![0, 1, 99_999],
                seed: 5,
                ..SampleOptions::default()
            },
        ),
        (
            [16, 1000],
            SampleOptions {
                top_k: 40,
                top_p: 0.95,
                temperature: 0.7,
                seed: 6,
                offset: 1 << 40,
                ..SampleOptions::default()
            },
        ),
    ];
    for (dims, options) in cases {
        let [b, v] = dims;
        let values = random_logits(dims[1] as u64, dims, 2.0, 6.0);
        let out = run(&values, dims, Dtype::Bfloat16, &options);
        let (mut checked, mut near_cut, mut near_score, mut worst) = (0, 0, 0, [0.0f64; 3]);
        for row in 0..b {
            let z = &values[row * v..(row + 1) * v];
            let exact = Exact::of(z, &options);
            if exact.near_boundary {
                near_cut += 1;
                continue;
            }
            for (w, e) in worst.iter_mut().zip(exact.errors(&out, row)) {
                *w = w.max(e);
            }
            let mut scores: Vec<(f64, usize)> = exact
                .kept
                .iter()
                .map(|&i| {
                    let s = (z[i] as f64 - exact.m) * exact.tau;
                    (
                        s + gumbel_f64(word(options.seed, options.offset, row as u32, i)),
                        i,
                    )
                })
                .collect();
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            match scores.len() == 1 || scores[0].0 - scores[1].0 > 1e-4 {
                true => {
                    checked += 1;
                    assert_eq!(out.tokens[row] as usize, scores[0].1, "{dims:?} row {row}");
                }
                false => near_score += 1,
            }
        }
        println!(
            "{dims:?} {options:?}: {checked} of {b} rows the definition's token, marked {near_cut} \
             near a cut's group end and {near_score} near a tie of scores; logprob {:.3e}, \
             tempered {:.3e}, sampling {:.3e}",
            worst[0], worst[1], worst[2]
        );
        assert!(
            checked * 4 >= b * 3,
            "{dims:?}: only {checked} rows clear of ties"
        );
        assert!(
            worst.iter().all(|&w| w <= LOGPROB_LARGE),
            "{dims:?}: {worst:?}"
        );
    }
}

/// The contract's errors: dash-ops panics on each.
#[test]
fn refuses_what_the_contract_rules_out() {
    let _serial = serial();
    let logits = Array::from_slice(&[0.0f32; 6], &[2, 3]);
    let bad = [
        SampleOptions {
            temperature: -1.0,
            ..SampleOptions::default()
        },
        SampleOptions {
            temperature: f32::INFINITY,
            ..SampleOptions::default()
        },
        SampleOptions {
            top_p: 1.5,
            ..SampleOptions::default()
        },
        SampleOptions {
            masked: vec![3],
            ..SampleOptions::default()
        },
        SampleOptions {
            masked: vec![0, 1, 2],
            ..SampleOptions::default()
        },
    ];
    for options in &bad {
        assert!(sample(&logits, options).is_err(), "{options:?}");
    }
    let flat = Array::from_slice(&[0.0f32; 3], &[3]);
    assert!(sample(&flat, &SampleOptions::default()).is_err());
    let empty = mlx_rs::ops::zeros::<f32>(&[0, 5]).unwrap();
    let out = sample(&empty, &SampleOptions::default()).unwrap();
    assert_eq!(out.tokens.shape(), [0]);
}

/// Top-p's cut at the decoding width against the cut in f64, on 128 flat and 128 peaked rows of
/// 248,320 bf16 logits at `p` 0.9 and 0.95: equal on every row whose crossing is not near a tie
/// group's end ([`NEAR`]); those rows are counted and reported, with how many differ.
/// `examples/sample_cuts.rs` measures the rates over 10,240 rows of each.
#[test]
fn top_p_cuts_are_the_definitions_off_the_boundaries() {
    use dash_mlx::sample::{CutSum, top_p_cut};
    let _serial = serial();
    let (rows, v) = (128usize, 248_320usize);
    for (kind, scale, lift) in [("flat", 1.0, 0.0), ("peaked", 2.0, 6.0)] {
        let values = random_logits(41, [rows, v], scale, lift);
        let sorted: Vec<Vec<f32>> = std::thread::scope(|s| {
            let handles: Vec<_> = values
                .chunks(v)
                .map(|row| {
                    s.spawn(move || {
                        let mut row = row.to_vec();
                        row.sort_unstable_by(|a, b| b.total_cmp(a));
                        row
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let logits = Array::from_slice(&values, &[rows as i32, v as i32])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        for p in [0.9f32, 0.95] {
            let options = SampleOptions {
                top_p: p,
                ..SampleOptions::default()
            };
            let got = top_p_cut(&logits, &options, CutSum::Tree).unwrap();
            let got = got.as_slice::<f32>();
            let (mut near, mut near_differ) = (0, 0);
            for (r, row) in sorted.iter().enumerate() {
                let m = row[0] as f64;
                let mut cumulative = Vec::with_capacity(v);
                let mut sum = 0f64;
                for &x in row {
                    sum += (x as f64 - m).exp();
                    cumulative.push(sum);
                }
                let target = p as f64 * sum;
                let at = cumulative.iter().position(|&c| c >= target).unwrap();
                let cut = row[at];
                let start = row.iter().position(|&x| x == cut).unwrap();
                let end = start + row[start..].iter().take_while(|&&x| x == cut).count();
                let before = if start == 0 {
                    0.0
                } else {
                    cumulative[start - 1]
                };
                let marked =
                    target - before <= NEAR * sum || cumulative[end - 1] - target <= NEAR * sum;
                match marked {
                    true => {
                        near += 1;
                        near_differ += usize::from(got[r] != cut);
                    }
                    false => assert_eq!(got[r], cut, "{kind} p {p} row {r}"),
                }
            }
            println!(
                "{kind} p {p}: {} rows the definition's cut, {near} marked near a group's end \
                 ({near_differ} of them differ)",
                rows - near
            );
        }
    }
}
