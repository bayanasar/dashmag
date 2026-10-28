//! Sampling the next token on the device, for a decoder's step: from the logits `[B, V]` of one
//! position per sequence, one token id per row, with its log-probability under three
//! distributions: the model's, the model's at the temperature, and the one it was drawn from.
//! Temperature, top-k, top-p and masked ids as the options say; greedy decoding at temperature 0.
//!
//! # Definition
//!
//! For each row with logits `z` (bf16, f16 or f32, widened to f32 exactly) and `U` the vocabulary
//! less the masked ids, which as in [`crate::logprob`] take no part in anything, as if their
//! logits were `-inf`:
//!
//! ```text
//! m        = max_{i in U} z_i
//! lse      = m + ln sum_{i in U} exp(z_i - m)
//! logprob  = z_t - lse                                     for the token t drawn
//! ```
//!
//! `logprob` is the token's log-probability under the model's own distribution, the log-softmax of
//! the raw logits at temperature 1 over `U`: the number `chosen_logprob` ([`crate::logprob`])
//! gives for that token.
//!
//! **Greedy** ([`SampleOptions::temperature`] 0): `t` is the lowest id `i` in `U` with `z_i = m`,
//! which is Burn's `argmax` over the row with the masked logits at `-inf`; top-k and top-p change
//! nothing. The distribution drawn from is the point mass at `t`, so `sampling_logprob = 0`; the
//! model's distribution at temperature 0 is taken as the same point mass (the limit of the softmax
//! of `z / T` as `T` falls to 0 where the row's largest logit is not tied), so
//! `tempered_logprob = 0` too.
//!
//! **Sampling** (temperature `T > 0`), with `tau = 1 / T` rounded to f32 (at most `f32::MAX`), the
//! options' `k` ([`SampleOptions::top_k`]) and `p` ([`SampleOptions::top_p`]):
//!
//! ```text
//! s_i  = (z_i - m) * tau,      w_i = exp(s_i)        tempered logits (s_i <= 0) and weights
//! K    = U                                           if k = 0 or k >= |U|
//!      = { i in U : z_i >= z_(k) }                   else; z_(k) the k-th largest logit in U
//! P    = K                                           if p = 1
//!      = { i in K : z_i >= v }                       else; v the largest logit in K with
//!                                                      sum_{j in K, z_j >= v} w_j >= p W,
//!                                                      W = sum_{j in K} w_j
//! t    = argmax_{i in P} (s_i + g_i)                 ties to the lowest id
//! tempered_logprob = s_t - ln sum_{i in U} w_i
//! sampling_logprob = s_t - ln sum_{i in P} w_i
//! ```
//!
//! `g_i` is a standard Gumbel variate (below), so `t` is drawn from the softmax of the tempered
//! logits over `P` (the Gumbel-max trick: the largest of `s_i + g_i` falls on `i` with probability
//! `w_i / sum_P w`), and `sampling_logprob` is the token's log-probability under that
//! distribution; `tempered_logprob` is its log-probability under the softmax of the tempered
//! logits over all of `U`, nothing cut: the log-softmax of `z / T`, to the rounding of `tau`.
//! Top-k keeps every token tied with the `k`-th largest logit, so `|K| >= k`; top-p keeps the top
//! tokens of `K` until their tempered mass first reaches `p` of `K`'s, with every token tied with
//! the last one: top-k first, then top-p over what it keeps, both after the temperature, the order
//! of Hugging Face's samplers and vLLM's. `p = 0` keeps the largest logit and its ties; `p = 1`
//! keeps `K` whole. Both cuts compare logits, so the temperature does not move them (`s` is a
//! non-decreasing function of `z`); the masses that place `v` are the tempered ones, `w`, summed
//! in f32.
//!
//! The three side by side, at the token `t`, with what each is for:
//!
//! ```text
//! logprob           log softmax(z)      over U   the model's log-probability of the token
//! tempered_logprob  log softmax(z / T)  over U   RL's importance ratio at the rollout temperature
//! sampling_logprob  log softmax(z / T)  over P   the distribution the token was drawn from
//! ```
//!
//! RL trainers such as TRL's GRPO trainer and verl's actor divide the logits by the rollout
//! temperature before their log-softmax and cut nothing, so the ratio of their log-probability to
//! the rollout's wants `tempered_logprob`. The kernel and the reference return the same bits where
//! two of them are the same number, the one taken from the other rather than computed again:
//! `tempered_logprob` is `logprob` at temperature 1 (`tau = 1`), and `sampling_logprob` is
//! `tempered_logprob` where nothing is cut (`k = 0` or `k >= |U|`, and `p = 1`); so at temperature
//! 1 with nothing cut the three are one number.
//!
//! A row whose logits in `U` are all `-inf` has no distribution: its token is its lowest id in
//! `U`, and the three log-probabilities are NaN. Logits are otherwise finite, or `-inf` for a token
//! the model rules out (which is never drawn, and keeps its place in the cuts' counts); NaN and
//! `+inf` give unspecified results.
//!
//! # Random numbers
//!
//! `g_i` comes from the 32-bit word `x` that Philox4x32-10 (Salmon et al., "Parallel random
//! numbers: as easy as 1, 2, 3", 2011; cuRAND's and PyTorch's generator on the GPU) gives for the
//! key `(seed mod 2^32, seed / 2^32)` and the counter `(i / 4, row, offset mod 2^32, offset /
//! 2^32)`, word `i % 4`. `x` stands for the uniform `u = (x + 1/2) / 2^32` in `(0, 1)`, and
//! `g = -ln E` with `E = -ln u`, computed from `a = (2y + 1) / 2^33` (rounded to f32) as
//!
//! ```text
//! E = -ln a              y = x,       when x < 2^31  (u < 1/2, a = u)
//! E = -ln_1p(-a)         y = !x,      otherwise      (u > 1/2, a = 1 - u)
//! ```
//!
//! so that `E` keeps its precision where `u` nears 1. `g` lies within `[-3.13, 22.9]`: a token
//! whose tempered logit is more than 26.0 below the row's largest (a probability below `5.1e-12`
//! of the top token's) is never drawn.
//!
//! The same seed, offset and logits give the same tokens and the same log-probabilities, bit for
//! bit, on every run on one device and build. A decoding loop keeps one seed and advances the
//! offset every step; the rows of one call draw independent numbers (the row is in the counter),
//! and so do calls with different offsets.
//!
//! The sample is not differentiable.

/// How `sample` draws. The defaults draw from the model's distribution as it is: temperature
/// 1, no top-k, no top-p, no masked id, seed and offset 0.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleOptions {
    /// `T >= 0`, finite: the logits are divided by it before the softmax drawn from. 0 is greedy
    /// decoding (the argmax).
    pub temperature: f32,
    /// Keep the `top_k` largest logits (and every token tied with the last); 0 keeps all.
    pub top_k: usize,
    /// In `0..=1`: keep the top tokens until their tempered probability reaches `top_p` (and
    /// every token tied with the last); 1 keeps all.
    pub top_p: f32,
    /// The generator's key.
    pub seed: u64,
    /// The generator's counter: advance it every step, keeping the seed.
    pub offset: u64,
    /// Token ids taken out of the distribution, in any order, each `< V`: never drawn, and in no
    /// sum or count (as [`crate::logprob`]'s).
    pub masked: Vec<u32>,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            offset: 0,
            masked: Vec::new(),
        }
    }
}

impl SampleOptions {
    /// Greedy decoding: the argmax of each row.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Self::default()
        }
    }

    /// Whether these options decode greedily.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}
