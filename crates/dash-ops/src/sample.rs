//! Sampling the next token on the device, for a decoder's step: from the logits `[B, V]` of one
//! position per sequence, one token id per row, with its log-probability under three
//! distributions: the model's, the model's at the temperature, and the one it was drawn from.
//! Temperature, top-k, top-p and masked ids as the options say; greedy decoding at temperature 0.
//!
//! [`dash_contract::sample`] defines the operation, which every implementation of it shares:
//! greedy decoding and the tempered draw, top-k, top-p and masked ids, the three log-probabilities,
//! the generator's stream and the Gumbel variates it gives, and the options ([`SampleOptions`]).
//! This module implements it on Burn.
//!
//! # Random numbers
//!
//! The `ln_1p` of the variates' definition is not CubeCL's `log1p`, which every compiler of
//! CubeCL 0.11 turns into `ln(1 + x)`: for `a < 2^-25` that rounds `1 - a` to 1, `E` to 0 and `g`
//! to `+inf`, which wins whatever its token's logit. At `V = 248320` with the temperature alone
//! every token's word has that chance, `2^-25`, so about `V 2^-25 = 0.74%` of the draws would
//! return a token chosen almost uniformly from the vocabulary. The kernel and the reference compute
//! it in f32 as
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
//! # How it is computed
//!
//! The trait's default body ([`SampleBackend`]) is the reference: the definition in Burn's tensor
//! operations, the cuts from the row sorted in descending order (Burn's sort, which reads the
//! row back to the host on a CubeCL device) and a cumulative sum of the sorted weights, every
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

pub use dash_contract::sample::SampleOptions;

/// What the references and the kernel read off [`SampleOptions`].
pub(crate) trait Cuts {
    /// `1 / T` in f32, at most `f32::MAX`: what the tempered logits are multiplied by.
    fn inverse_temperature(&self) -> f32;

    /// The `k` of top-k that cuts anything for a vocabulary of `unmasked` tokens, if any.
    fn top_k_cut(&self, unmasked: usize) -> Option<usize>;

    /// The `p` of top-p, if it cuts anything.
    fn top_p_cut(&self) -> Option<f32>;
}

impl Cuts for SampleOptions {
    fn inverse_temperature(&self) -> f32 {
        (1.0 / self.temperature).min(f32::MAX)
    }

    fn top_k_cut(&self, unmasked: usize) -> Option<usize> {
        (self.top_k > 0 && self.top_k < unmasked).then_some(self.top_k)
    }

    fn top_p_cut(&self) -> Option<f32> {
        (self.top_p < 1.0).then_some(self.top_p)
    }
}

/// `options` with the masked ids sorted and free of repeats, as the backends take them.
fn normalised(options: &SampleOptions) -> SampleOptions {
    let mut masked = options.masked.clone();
    masked.sort_unstable();
    masked.dedup();
    SampleOptions {
        masked,
        ..options.clone()
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
    /// `(tokens, logprob, tempered_logprob, sampling_logprob)` for `logits` `[B, V]` (f32, bf16 or
    /// f16): the token ids `[B]` in `out_dtype` and three `[B]` f32 log-probabilities, as
    /// [`dash_contract::sample`] defines them. `options` come with their masked ids sorted and
    /// free of repeats, each `< V`, fewer than `V`. The default body is the reference.
    #[fusion(meta = fusion_meta::sample)]
    fn sample(
        logits: FloatTensor<Self>,
        options: SampleOptions,
        out_dtype: IntDType,
    ) -> (
        IntTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        reference::sample::<Self>(logits, &options, out_dtype)
    }
}

#[cfg(feature = "flex")]
impl SampleBackend for burn::backend::Flex {}

/// Output metadata for Burn's lazy fusion: `[B]` tokens in the asked int dtype, three `[B]` f32.
#[cfg(feature = "fusion")]
mod fusion_meta {
    use super::SampleOptions;
    use burn::backend::fusion::custom::TensorSpec;
    use burn::backend::{DType, IntDType, Shape};

    pub(super) fn sample(
        logits: &TensorSpec,
        _options: &SampleOptions,
        out_dtype: &IntDType,
    ) -> (TensorSpec, TensorSpec, TensorSpec, TensorSpec) {
        let rows = Shape::new([logits.shape[0]]);
        (
            TensorSpec::new(rows.clone(), DType::from(*out_dtype)),
            TensorSpec::new(rows.clone(), DType::F32),
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
    ) -> (
        IntTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let (tokens, logprob, tempered, sampling) =
            B::sample(<Self as AutodiffBackend>::inner(logits), options, out_dtype);
        (
            tokens,
            <Self as AutodiffBackend>::from_inner(logprob),
            <Self as AutodiffBackend>::from_inner(tempered),
            <Self as AutodiffBackend>::from_inner(sampling),
        )
    }
}

/// What [`sample`] returns, each `[B]`: the token and its log-probability under three
/// distributions, side by side (the [module documentation](self) defines them).
#[derive(Debug, Clone)]
pub struct Sampled {
    /// The token drawn for each row, in the device's int dtype.
    pub tokens: Tensor<1, Int>,
    /// Its log-probability under the model's distribution: the log-softmax of the raw logits over
    /// the unmasked vocabulary, at temperature 1. The model's log-probability of the token. f32.
    pub logprob: Tensor<1>,
    /// Its log-probability under the model's distribution at the temperature: the log-softmax of
    /// the logits divided by `T` over the unmasked vocabulary, nothing cut; 0 when greedy. What an
    /// RL trainer's importance ratio takes at the rollout temperature (TRL's GRPO trainer and
    /// verl's actor divide the logits by it before their log-softmax). `logprob`'s bits at
    /// temperature 1. f32.
    pub tempered_logprob: Tensor<1>,
    /// Its log-probability under the distribution it was drawn from: the softmax of the tempered
    /// logits over the tokens top-k and top-p keep; 0 when greedy. `tempered_logprob`'s bits when
    /// they cut nothing. f32.
    pub sampling_logprob: Tensor<1>,
}

/// Draws one token per row of `logits` `[B, V]` (f32, bf16 or f16), as `options` say; see
/// [`dash_contract::sample`] for the definition. No gradient flows through it.
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
    let (tokens, logprob, tempered, sampling) = <Dispatch as SampleBackend>::sample(
        logits.into_dispatch(),
        normalised(options),
        out_dtype,
    );
    Sampled {
        tokens: Tensor::from_dispatch(tokens),
        logprob: Tensor::from_dispatch(logprob),
        tempered_logprob: Tensor::from_dispatch(tempered),
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
    let (tokens, logprob, tempered, sampling) =
        reference::sample::<Dispatch>(logits.into_dispatch(), &normalised(options), out_dtype);
    Sampled {
        tokens: Tensor::from_dispatch(tokens),
        logprob: Tensor::from_dispatch(logprob),
        tempered_logprob: Tensor::from_dispatch(tempered),
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
