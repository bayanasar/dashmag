//! Sampling the next token on the device, for a decoder's step: from the logits `[B, V]` of one
//! position per sequence, one token id per row, with its log-probability under the model's
//! distribution and under the distribution it was drawn from. Temperature, top-k, top-p and
//! masked ids as the options say; greedy decoding at temperature 0.
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
//! the raw logits at temperature 1 over `U`: the number [`crate::logprob::chosen_logprob`] gives
//! for that token.
//!
//! **Greedy** ([`SampleOptions::temperature`] 0): `t` is the lowest id `i` in `U` with `z_i = m`,
//! which is Burn's `argmax` over the row with the masked logits at `-inf`; top-k and top-p change
//! nothing. The distribution drawn from is the point mass at `t`, so `sampling_logprob = 0`.
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
//! sampling_logprob = s_t - ln sum_{i in P} w_i
//! ```
//!
//! `g_i` is a standard Gumbel variate (below), so `t` is drawn from the softmax of the tempered
//! logits over `P` (the Gumbel-max trick: the largest of `s_i + g_i` falls on `i` with probability
//! `w_i / sum_P w`), and `sampling_logprob` is the token's log-probability under that
//! distribution. Top-k keeps every token tied with the `k`-th largest logit, so `|K| >= k`; top-p
//! keeps the top tokens of `K` until their tempered mass first reaches `p` of `K`'s, with every
//! token tied with the last one: top-k first, then top-p over what it keeps, both after the
//! temperature, the order of Hugging Face's samplers and vLLM's. `p = 0` keeps the largest logit
//! and its ties; `p = 1` keeps `K` whole. Both cuts compare logits, so the temperature does not
//! move them (`s` is a non-decreasing function of `z`); the masses that place `v` are the tempered
//! ones, `w`, summed in f32.
//!
//! A row whose logits in `U` are all `-inf` has no distribution: its token is its lowest id in
//! `U`, and both log-probabilities are NaN. Logits are otherwise finite, or `-inf` for a token the
//! model rules out (which is never drawn, and keeps its place in the cuts' counts); NaN and `+inf`
//! give unspecified results.
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
//! That `ln_1p` is not CubeCL's `log1p`, which every compiler of CubeCL 0.11 turns into
//! `ln(1 + x)`: for `a < 2^-25` that rounds `1 - a` to 1, `E` to 0 and `g` to `+inf`, which wins
//! whatever its token's logit. At `V = 248320` with the temperature alone every token's word has
//! that chance, `2^-25`, so about `V 2^-25 = 0.74%` of the draws would return a token chosen almost
//! uniformly from the vocabulary. The kernel and the reference compute it in f32 as
//!
//! ```text
//! E = a (6 - a) / (6 - 4a)               a <= 2^-6   (a Padé form: relative error a^3 / 36)
//! E = -ln(w) a / (1 - w),  w = 1 - a     otherwise   (Goldberg's: 1 - w is exact)
//! ```
//!
//! where each product that meets a sum is exact, so that a compiler fusing them into multiply-adds
//! changes no bit. Against `g` in f64 from the same `a`, the variates are within 1.02e-6 on Flex,
//! CUDA and CubeCL's CPU runtime (about half the f32 spacing of `g` at the top of the range), and
//! within 5.1e-6 on lavapipe and OpenGL, whose `ln` is less exact (to about 1e-7 near 1).
//!
//! The same seed, offset and logits give the same tokens and the same log-probabilities, bit for
//! bit, on every run on one device and build. A decoding loop keeps one seed and advances the
//! offset every step; the rows of one call draw independent numbers (the row is in the counter),
//! and so do calls with different offsets.
//!
//! # How it is computed
//!
//! The trait's default body ([`SampleBackend`]) is the reference: the definition above in Burn's
//! tensor operations, the cuts from the row sorted in descending order (Burn's sort, which reads
//! the row back to the host on a CubeCL device) and a cumulative sum of the sorted weights, every
//! sum over a row a balanced tree of pairs, the Gumbel words drawn on the host. On a CubeCL device
//! with what the kernel asks (`kernel_takes`), one kernel launch per call: a cube per row, which
//! reads its row once for greedy decoding, twice for the temperature alone or top-k, two to four
//! times for top-p while the tokens down to the cut fit in shared memory (8192 of them on CUDA),
//! and more when they do not (a flat distribution under top-p, a `k` above 1024); it finds the cuts
//! with no sort of the row and no host round trip. Its module (`sample/cube.rs`) describes the
//! passes.
//!
//! The kernel and the reference draw the same words for the same tokens and compute `s` and `g`
//! alike:
//!
//! - Greedy: the same tokens, bit for bit (no arithmetic decides them).
//! - Top-k, or the temperature alone: the same kept set (the cut compares logits), and the same
//!   token unless two scores `s_i + g_i` of the kept set lie within their rounding of each other:
//!   the two evaluate `exp` and `ln` with the device's functions, which may differ in their last
//!   bit from one device to another.
//! - Top-p: the masses are f32 sums formed in different orders, so where `p W` lies within their
//!   rounding (a few parts in `10^7` of `W`) of the mass above a logit, the two may differ on
//!   whether that logit's tokens are kept, and the token drawn differs if it is one of them.
//!
//! bf16 and f16 logits are widened to f32 exactly, so what holds for f32 logits holds for them.
//!
//! The sample is not differentiable: on an autodiff device the outputs are constants, whatever the
//! logits require.

#[cfg(feature = "cubecl")]
pub(crate) mod cube;
pub(crate) mod philox;
pub(crate) mod reference;

use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, DType, Dispatch, IntDType, backend_extension,
    get_or_init_device_settings,
};
use burn::tensor::{Device, Int, Tensor, TensorData};
use std::sync::atomic::{AtomicUsize, Ordering};

/// How [`sample`] draws. The defaults draw from the model's distribution as it is: temperature
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

    /// `1 / T` in f32, at most `f32::MAX`: what the tempered logits are multiplied by.
    pub(crate) fn inverse_temperature(&self) -> f32 {
        (1.0 / self.temperature).min(f32::MAX)
    }

    /// The `k` of top-k that cuts anything for a vocabulary of `unmasked` tokens, if any.
    pub(crate) fn top_k_cut(&self, unmasked: usize) -> Option<usize> {
        (self.top_k > 0 && self.top_k < unmasked).then_some(self.top_k)
    }

    /// The `p` of top-p, if it cuts anything.
    pub(crate) fn top_p_cut(&self) -> Option<f32> {
        (self.top_p < 1.0).then_some(self.top_p)
    }

    /// The options with the masked ids sorted and free of repeats, as the backends take them.
    fn normalised(&self) -> Self {
        let mut masked = self.masked.clone();
        masked.sort_unstable();
        masked.dedup();
        Self {
            masked,
            ..self.clone()
        }
    }
}

/// Sampling as a backend extension.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(feature = "cubecl"),
    Fusion: cfg(feature = "fusion"),
    Autodiff
)]
pub trait SampleBackend: Backend {
    /// `(tokens, logprob, sampling_logprob)` for `logits` `[B, V]` (f32, bf16 or f16): the token
    /// ids `[B]` in `out_dtype` and two `[B]` f32 log-probabilities, as the
    /// [module documentation](self) defines them. `options` come with their masked ids sorted and
    /// free of repeats, each `< V`, fewer than `V`. The default body is the reference.
    #[fusion(meta = fusion_meta::sample)]
    fn sample(
        logits: FloatTensor<Self>,
        options: SampleOptions,
        out_dtype: IntDType,
    ) -> (IntTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        reference::sample::<Self>(logits, &options, out_dtype)
    }
}

#[cfg(feature = "flex")]
impl SampleBackend for burn::backend::Flex {}

/// Output metadata for Burn's lazy fusion: `[B]` tokens in the asked int dtype, two `[B]` f32.
#[cfg(feature = "fusion")]
mod fusion_meta {
    use super::SampleOptions;
    use burn::backend::fusion::custom::TensorSpec;
    use burn::backend::{DType, IntDType, Shape};

    pub(super) fn sample(
        logits: &TensorSpec,
        _options: &SampleOptions,
        out_dtype: &IntDType,
    ) -> (TensorSpec, TensorSpec, TensorSpec) {
        let rows = Shape::new([logits.shape[0]]);
        (
            TensorSpec::new(rows.clone(), DType::from(*out_dtype)),
            TensorSpec::new(rows.clone(), DType::F32),
            TensorSpec::new(rows, DType::F32),
        )
    }
}

/// Counters a test reads to tell which code ran. Not part of the API. The sample has no backward
/// pass to count.
#[doc(hidden)]
pub mod probe {
    use super::{AtomicUsize, Ordering};

    pub(super) static KERNEL_LAUNCHES: AtomicUsize = AtomicUsize::new(0);

    /// Launches of the sampling kernel: one per call it takes, none where the reference runs.
    pub fn kernel_launches() -> usize {
        KERNEL_LAUNCHES.load(Ordering::SeqCst)
    }
}

/// Whether the kernel, rather than the reference, takes logits `[B, V]` (`dims`) in `dtype` on
/// `device`: a CubeCL device with what it asks (the kernel module's documentation in the source).
/// `false` on any device not CubeCL's. For tests and benchmarks.
#[doc(hidden)]
pub fn kernel_takes(device: &Device, dtype: DType, dims: [usize; 2]) -> bool {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        let int = get_or_init_device_settings::<Dispatch>(device.as_dispatch()).int_dtype;
        return cube::kernel_takes(&client, dtype, int.into(), dims);
    }
    let _ = (device, dtype, dims);
    false
}

/// The shared-memory capacity of the kernel's fast path on `device`, in candidate tokens: rows
/// whose cuts keep more fall to its slower passes (the kernel module's documentation in the
/// source). 0 where the kernel does not run. For tests and benchmarks.
#[doc(hidden)]
pub fn kernel_capacity(device: &Device) -> usize {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        let int = get_or_init_device_settings::<Dispatch>(device.as_dispatch()).int_dtype;
        return match cube::kernel_takes(&client, DType::F32, int.into(), [1, 1]) {
            true => cube::capacity(&client),
            false => 0,
        };
    }
    let _ = device;
    0
}

/// The sample is forward only: an autodiff device runs the inner backend's (the kernel on a CubeCL
/// device) on the logits' values, and returns untracked outputs.
impl<B: Backend + SampleBackend, C: CheckpointStrategy> SampleBackend for Autodiff<B, C> {
    fn sample(
        logits: FloatTensor<Self>,
        options: SampleOptions,
        out_dtype: IntDType,
    ) -> (IntTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (tokens, logprob, sampling) =
            B::sample(<Self as AutodiffBackend>::inner(logits), options, out_dtype);
        (
            tokens,
            <Self as AutodiffBackend>::from_inner(logprob),
            <Self as AutodiffBackend>::from_inner(sampling),
        )
    }
}

/// What [`sample`] returns, each `[B]`.
#[derive(Debug, Clone)]
pub struct Sampled {
    /// The token drawn for each row, in the device's int dtype.
    pub tokens: Tensor<1, Int>,
    /// Its log-probability under the model's distribution: the log-softmax of the raw logits over
    /// the unmasked vocabulary, at temperature 1. f32.
    pub logprob: Tensor<1>,
    /// Its log-probability under the distribution it was drawn from: the softmax of the tempered
    /// logits over the tokens top-k and top-p keep; 0 when greedy. f32.
    pub sampling_logprob: Tensor<1>,
}

/// Draws one token per row of `logits` `[B, V]` (f32, bf16 or f16), as `options` say; see the
/// [module documentation](self) for the definition. No gradient flows through it.
///
/// # Panics
///
/// If `logits` is not 2-D or `V` is 0, if the temperature is negative or not finite, if `top_p` is
/// not within `0..=1`, if a masked id is `>= V`, or if every token is masked.
pub fn sample(logits: Tensor<2>, options: &SampleOptions) -> Sampled {
    let [_, v] = logits.dims();
    check(v, options);
    let device = logits.device();
    let out_dtype = get_or_init_device_settings::<Dispatch>(device.as_dispatch()).int_dtype;
    let (tokens, logprob, sampling) = <Dispatch as SampleBackend>::sample(
        logits.into_dispatch(),
        options.normalised(),
        out_dtype,
    );
    Sampled {
        tokens: Tensor::from_dispatch(tokens),
        logprob: Tensor::from_dispatch(logprob),
        sampling_logprob: Tensor::from_dispatch(sampling),
    }
}

/// [`sample`] through the reference alone (Burn's tensor operations) on any device: what Flex runs
/// and the kernel is held to. For tests and benchmarks.
#[doc(hidden)]
pub fn sample_reference(logits: Tensor<2>, options: &SampleOptions) -> Sampled {
    let [_, v] = logits.dims();
    check(v, options);
    let device = logits.device();
    let out_dtype = get_or_init_device_settings::<Dispatch>(device.as_dispatch()).int_dtype;
    let (tokens, logprob, sampling) =
        reference::sample::<Dispatch>(logits.into_dispatch(), &options.normalised(), out_dtype);
    Sampled {
        tokens: Tensor::from_dispatch(tokens),
        logprob: Tensor::from_dispatch(logprob),
        sampling_logprob: Tensor::from_dispatch(sampling),
    }
}

/// The Gumbel variates `g` (f32) that the generator's `words` give on `device`, as the
/// [module documentation](self) computes them: the reference's, from Burn's operations on the
/// device, and on a CubeCL device the kernel's own function, run alone (a unit per word, which
/// every CubeCL device runs, whether or not it takes the sampling kernel). For tests.
#[doc(hidden)]
pub fn gumbel_variates(device: &Device, words: &[u32]) -> (Vec<f32>, Option<Vec<f32>>) {
    let inputs: Vec<f32> = words.iter().map(|&x| philox::gumbel_input(x)).collect();
    let inputs = Tensor::<1>::from_data(TensorData::new(inputs, [words.len()]), device);
    let bool_dtype = get_or_init_device_settings::<Dispatch>(device.as_dispatch()).bool_dtype;
    let g = reference::gumbel_of_inputs::<Dispatch>(inputs.into_dispatch(), bool_dtype);
    let reference = Tensor::<1>::from_dispatch(g)
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32 variates");
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return (reference, Some(cube::gumbel_variates(&client, words)));
    }
    (reference, None)
}

/// The options' checks against a vocabulary of `v`.
fn check(v: usize, options: &SampleOptions) {
    assert!(v > 0, "sample: the vocabulary is empty");
    assert!(
        options.temperature.is_finite() && options.temperature >= 0.0,
        "sample: the temperature must be finite and >= 0, got {}",
        options.temperature
    );
    assert!(
        (0.0..=1.0).contains(&options.top_p),
        "sample: top_p must be within 0..=1, got {}",
        options.top_p
    );
    let mut masked = options.masked.clone();
    masked.sort_unstable();
    masked.dedup();
    if let Some(&last) = masked.last() {
        assert!(
            (last as usize) < v,
            "sample: masked token id {last} is outside the vocabulary of {v}"
        );
        assert!(
            masked.len() < v,
            "sample: every token of the vocabulary is masked"
        );
    }
}
