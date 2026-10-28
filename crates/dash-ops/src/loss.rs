//! Cross-entropy over a large vocabulary straight from the hidden states and the output weight,
//! without the n x V logits: [`linear_cross_entropy`] is torch's
//! `F.cross_entropy(F.linear(hidden, weight), targets)` with its reductions, ignore index and
//! label smoothing, plus a z-loss and masked vocabulary ids, returning beside the loss each row's
//! log-probability of its target and, if asked, its entropy. Gradients flow to `hidden` and,
//! when it requires one, to `weight`: an output head trains through it.
//!
//! [`dash_contract::loss`] defines the loss, which every implementation of it shares: the
//! definition, the label smoothing, z-loss, masked ids and ignored rows, the [`Reduction`]s, the
//! shapes and dtypes, and the options ([`CrossEntropyOptions`]). This module implements it on
//! Burn, in the terms used there.
//!
//! # The gradient
//!
//! With `gl[i]`, `gp[i]`, `ge[i]` the upstream gradients of row `i`'s loss, log-probability and
//! entropy (0 for an ignored row), the gradient of the logits is, for an unmasked token `j`,
//!
//! ```text
//! dz[i, j] = p (A - ge (z - lse)) + C + [j = t] (gp - (1 - eps) gl)
//! A        = gl (1 + 2 lambda lse) - gp - ge entropy,     C = -gl eps / K
//! ```
//!
//! and 0 for a masked one; the gradient of `hidden` is `dz weight`, that of `weight` is
//! `dz^T hidden`. At the target, where `p` nears 1 the bracket cancels against `p A`, so the
//! target's value is formed from `1 - p(target)` computed as the sum over the other tokens, as
//! [`crate::logprob`] does.
//!
//! # How it is computed
//!
//! The forward pass is [`crate::logprob`]'s: the vocabulary in chunks of
//! [`LogprobOptions::vocab_chunk`] tokens, rows in blocks of [`LogprobOptions::row_chunk`], one
//! `[rows, C]` chunk of logits at a time, an online log-sum-exp per row, plus the sum of the
//! logits when there is label smoothing. Then one of two backward passes:
//!
//! - With a `weight` that requires no gradient and no entropy asked for, the forward pass also
//!   accumulates `o = sum_{j != t} exp(z - m) weight[j]` as `chosen_logprob` does, and since `dz`
//!   is `p A` but for the target's and the uniform terms, the backward pass is
//!   `(A / s) o + (target term) weight[t] + C sum_{j in U} weight[j]` per row: no second pass
//!   over the vocabulary.
//! - When `weight` requires a gradient (or the entropy is asked for, whose gradient `o` does not
//!   give), the backward pass walks the vocabulary again: each chunk's logits recomputed, `dz`
//!   formed from them and per-row coefficients, `dz weight_c` added to the gradient of `hidden`,
//!   and `dz^T hidden`, the chunk's rows of the gradient of `weight`, written in place. The
//!   weight's gradient needs every row's final log-sum-exp before its first chunk, which is why
//!   it takes the second pass; it holds one chunk of `dz` for all rows, `n x C`.
//!
//! On a CubeCL device (any runtime) the passes are `chosen_logprob`'s kernels: the matmuls take
//! bf16 operands when both inputs are bf16, with f32 results. In the second pass `dz` is rounded
//! to bf16 for the tensor cores, as torch's bf16 linear layer rounds the gradient of its output;
//! the forward pass's `o` takes its softmax weights in two bf16 parts, as `chosen_logprob` does.
//! Every other input pair is computed in f32 (f16 too: its parts would lose the smallest
//! probabilities). Everything accumulates in f32; the gradients come back in the dtypes of
//! `hidden` and `weight`, rounded once.
//!
//! Each pass runs its kernels where the device has what they ask, its reference otherwise: the
//! forward pass's kernel sums over planes of any width at least 8, which CUDA has and wgpu on
//! lavapipe or OpenGL does not; the second pass's kernel asks for cubes of 256 units and 64-bit
//! integers, which lavapipe has, OpenGL lacks the integers and CubeCL's CPU runtime the units
//! (it has as many units per cube as cores).

use crate::logprob::{LogprobOptions, VocabChunk};
use crate::logprob::coefficient;
use crate::logprob::reference::{self, Pass, Rows, cast, dims2, full, sorted, to_f32};
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, DType, Dispatch, FloatDType, Scalar, Shape, Slice,
    TensorData, TensorMetadata, backend_extension, get_or_init_device_settings,
};
use burn::tensor::{Device, Int, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};

pub use dash_contract::loss::{CrossEntropyOptions, Reduction};

/// What [`linear_cross_entropy`] returns, all f32.
#[derive(Debug, Clone)]
pub struct CrossEntropy {
    /// The loss: `[1]` for [`Reduction::Mean`] and [`Reduction::Sum`], `[n]` for
    /// [`Reduction::None`].
    pub loss: Tensor<1>,
    /// `log p(target)` per row, `[n]`: `-inf` where the target is masked or out of range, 0 where
    /// the row is ignored.
    pub logprob: Tensor<1>,
    /// The entropy of each row's distribution over the unmasked vocabulary, `[n]` (0 where the
    /// row is ignored), when [`CrossEntropyOptions::entropy`] asks for it.
    pub entropy: Option<Tensor<1>>,
}

/// The rows of the `[COUNT, n]` f32 tensor the backend's forward pass returns: per input row,
/// the three outputs and what the backward pass reads.
pub mod row {
    /// The row's loss (0 if ignored).
    pub const LOSS: usize = 0;
    /// `log p(target)` (0 if ignored, `-inf` if the target is masked or out of range).
    pub const LOGPROB: usize = 1;
    /// The entropy (0 if ignored).
    pub const ENTROPY: usize = 2;
    /// The log-sum-exp over the unmasked tokens.
    pub const LSE: usize = 3;
    /// `1 - p(target)`, summed over the other tokens (1 where the target is masked).
    pub const Q: usize = 4;
    /// `sum_j exp(z - m)`, `m` the row's largest unmasked logit.
    pub const S: usize = 5;
    /// 1 unless the row is ignored.
    pub const COUNTED: usize = 6;
    /// 1 where the target is an unmasked id in `0..V`.
    pub const FOUND: usize = 7;
    /// The number of rows.
    pub const COUNT: usize = 8;
}

/// The fused linear cross-entropy as a backend extension. Inputs, for both operations:
///
/// - `hidden` `[n, D]` and `weight` `[V, D]`, f32, bf16 or f16, one dtype (the kernels take bf16
///   to the tensor cores; any other pair is computed in f32);
/// - `targets` `[n]`, any int dtype;
/// - `options`: as [`linear_cross_entropy`] takes them.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(feature = "cubecl"),
    Fusion: cfg(feature = "fusion"),
    Autodiff
)]
pub trait LinearCrossEntropyBackend: Backend {
    /// The forward pass: `(rows, o)`, both f32. `rows` `[row::COUNT, n]` holds per input row its
    /// loss, log-probability and entropy and what the backward pass reads (see [`row`]); `o`
    /// `[n, D]` is accumulated for the gradient of `hidden` when `accumulate`, else `[0, D]`. On
    /// `Autodiff`, `rows` carries the gradient for `hidden` and `weight`, and `accumulate` is
    /// chosen by what requires a gradient.
    #[fusion(meta = fusion_meta::forward)]
    fn linear_cross_entropy(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        options: CrossEntropyOptions,
        accumulate: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        forward::<Self>(
            hidden,
            weight,
            targets,
            &options,
            accumulate,
            reference::pass,
        )
    }

    /// The backward pass: `(grad_hidden, grad_weight)` in `hidden`'s and `weight`'s dtypes, each
    /// `[0, D]` unless asked for. `rows` and `o` are the forward pass's, `upstream`
    /// `[row::COUNT, n]` f32 the gradient of `rows` (its first three rows are read). Not tracked
    /// by autodiff itself.
    #[fusion(meta = fusion_meta::backward)]
    #[allow(clippy::too_many_arguments)]
    fn linear_cross_entropy_backward(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        rows: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        o: FloatTensor<Self>,
        options: CrossEntropyOptions,
        grad_hidden: bool,
        grad_weight: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let masked = sorted(&options.masked);
        backward::<Self>(
            Saved {
                hidden,
                weight,
                targets,
                rows,
                o,
            },
            upstream,
            &options,
            [grad_hidden, grad_weight],
            |hidden, weight, targets, coef, want| {
                reference::gradient_pass::<Self>(
                    hidden,
                    weight,
                    targets,
                    &masked,
                    options.chunks,
                    coef,
                    options.entropy,
                    want[0],
                    want[1],
                )
            },
        )
    }
}

#[cfg(feature = "flex")]
impl LinearCrossEntropyBackend for burn::backend::Flex {}

/// The passes over the vocabulary in `chosen_logprob`'s kernels, each where the device has what
/// they ask, its reference otherwise.
#[cfg(feature = "cubecl")]
mod cube {
    use super::{CrossEntropyOptions, LinearCrossEntropyBackend, Saved, backward, forward, probe};
    use crate::logprob::cube::{self, kernels_take_these};
    use crate::logprob::reference::{self, Pass, sorted};
    use burn::backend::tensor::{FloatTensor, IntTensor};
    use burn_cubecl::CubeBackend;

    impl LinearCrossEntropyBackend for CubeBackend {
        fn linear_cross_entropy(
            hidden: FloatTensor<Self>,
            weight: FloatTensor<Self>,
            targets: IntTensor<Self>,
            options: CrossEntropyOptions,
            accumulate: bool,
        ) -> (FloatTensor<Self>, FloatTensor<Self>) {
            let kernels = kernels_take_these(&hidden, &weight, &targets, options.chunks, false);
            forward::<Self>(
                hidden,
                weight,
                targets,
                &options,
                accumulate,
                |hidden, weight, targets, masked, chunks, accumulate, logit_sum| {
                    if !kernels {
                        return reference::pass::<Self>(
                            hidden, weight, targets, masked, chunks, accumulate, logit_sum,
                        );
                    }
                    let (state, o) = cube::pass(
                        hidden,
                        weight,
                        targets,
                        masked,
                        chunks,
                        accumulate,
                        logit_sum,
                        &probe::PASS_LAUNCHES,
                    );
                    Pass {
                        rows: cube::rows(state, logit_sum),
                        o,
                    }
                },
            )
        }

        fn linear_cross_entropy_backward(
            hidden: FloatTensor<Self>,
            weight: FloatTensor<Self>,
            targets: IntTensor<Self>,
            rows: FloatTensor<Self>,
            upstream: FloatTensor<Self>,
            o: FloatTensor<Self>,
            options: CrossEntropyOptions,
            grad_hidden: bool,
            grad_weight: bool,
        ) -> (FloatTensor<Self>, FloatTensor<Self>) {
            let masked = sorted(&options.masked);
            let kernels = kernels_take_these(&hidden, &weight, &targets, options.chunks, true);
            backward::<Self>(
                Saved {
                    hidden,
                    weight,
                    targets,
                    rows,
                    o,
                },
                upstream,
                &options,
                [grad_hidden, grad_weight],
                |hidden, weight, targets, coef, want| match kernels {
                    true => cube::gradient_pass(
                        hidden,
                        weight,
                        targets,
                        &masked,
                        options.chunks,
                        coef,
                        options.entropy,
                        want[0],
                        want[1],
                        &probe::GRADIENT_LAUNCHES,
                    ),
                    false => reference::gradient_pass::<Self>(
                        hidden,
                        weight,
                        targets,
                        &masked,
                        options.chunks,
                        coef,
                        options.entropy,
                        want[0],
                        want[1],
                    ),
                },
            )
        }
    }
}

/// Output metadata for Burn's lazy fusion.
#[cfg(feature = "fusion")]
mod fusion_meta {
    use super::{CrossEntropyOptions, row};
    use burn::backend::fusion::custom::TensorSpec;
    use burn::backend::{DType, Shape};

    pub(super) fn forward(
        hidden: &TensorSpec,
        _weight: &TensorSpec,
        _targets: &TensorSpec,
        _options: &CrossEntropyOptions,
        accumulate: &bool,
    ) -> (TensorSpec, TensorSpec) {
        let (n, d) = (hidden.shape[0], hidden.shape[1]);
        (
            TensorSpec::new(Shape::new([row::COUNT, n]), DType::F32),
            TensorSpec::new(Shape::new([if *accumulate { n } else { 0 }, d]), DType::F32),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn backward(
        hidden: &TensorSpec,
        weight: &TensorSpec,
        _targets: &TensorSpec,
        _rows: &TensorSpec,
        _upstream: &TensorSpec,
        _o: &TensorSpec,
        _options: &CrossEntropyOptions,
        grad_hidden: &bool,
        grad_weight: &bool,
    ) -> (TensorSpec, TensorSpec) {
        let (n, d, v) = (hidden.shape[0], hidden.shape[1], weight.shape[0]);
        (
            TensorSpec::new(
                Shape::new([if *grad_hidden { n } else { 0 }, d]),
                hidden.dtype,
            ),
            TensorSpec::new(
                Shape::new([if *grad_weight { v } else { 0 }, d]),
                weight.dtype,
            ),
        )
    }
}

/// Counters a test reads to tell which code ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use super::{AtomicUsize, Ordering};

    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);
    pub(super) static PASS_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static GRADIENT_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static GRADIENT_PASSES: AtomicUsize = AtomicUsize::new(0);

    /// Runs of the written-out backward pass.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }

    /// Launches of the loss's CubeCL kernels, both passes' (none where the references run).
    pub fn kernel_launches() -> usize {
        pass_launches() + gradient_launches()
    }

    /// Launches of the forward pass's chunk kernel: one per chunk of the vocabulary and block of
    /// rows.
    pub fn pass_launches() -> usize {
        PASS_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of the second pass's gradient kernel: one per chunk and block.
    pub fn gradient_launches() -> usize {
        GRADIENT_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Second passes over the vocabulary made by the backward pass (on any backend); none when
    /// the forward pass accumulated the gradient of `hidden`.
    pub fn gradient_passes() -> usize {
        GRADIENT_PASSES.load(Ordering::SeqCst)
    }
}

/// Whether the CubeCL kernels, rather than the reference, take the forward pass and the second
/// pass of [`linear_cross_entropy`] on `device`, for `n` rows, `hidden`, `weight` and `targets`
/// in `dtypes` (in that order) and `chunks`: a device with what each pass's kernel asks (module
/// documentation). `[false, false]` on any device not CubeCL's. For tests, as the [`probe`]
/// counters are.
#[doc(hidden)]
pub fn kernels_take(
    device: &Device,
    dtypes: [DType; 3],
    n: usize,
    chunks: LogprobOptions,
) -> [bool; 2] {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return [false, true].map(|gradient| {
            crate::logprob::cube::kernels_take(&client, dtypes, n, chunks, gradient)
        });
    }
    let _ = (device, dtypes, n, chunks);
    [false, false]
}

/// The forward pass on any backend, given its pass over the vocabulary: `(rows, o)`.
fn forward<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    targets: IntTensor<B>,
    options: &CrossEntropyOptions,
    accumulate: bool,
    pass: impl FnOnce(
        FloatTensor<B>,
        FloatTensor<B>,
        IntTensor<B>,
        &[u32],
        LogprobOptions,
        bool,
        bool,
    ) -> Pass<B>,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let [n, d] = dims2(&hidden.shape(), "hidden");
    let [v, dw] = dims2(&weight.shape(), "weight");
    assert_eq!(d, dw, "hidden [n, D] and weight [V, D] must share D");
    assert_eq!(targets.shape().dims::<1>(), [n], "targets must be [n]");
    let device = hidden.device();
    if n == 0 {
        return (
            full::<B>([row::COUNT, 0], 0.0, &device),
            full::<B>([0, d], 0.0, &device),
        );
    }
    let masked = sorted(&options.masked);
    let classes = v - masked.len();
    let logit_sum = options.label_smoothing != 0.0;
    let Pass { rows, o } = pass(
        hidden,
        weight,
        targets.clone(),
        &masked,
        options.chunks,
        accumulate,
        logit_sum,
    );
    let rows = finish::<B>(rows, targets, options, classes);
    let o = o.unwrap_or_else(|| full::<B>([0, d], 0.0, &device));
    (rows, o)
}

/// The rows' outputs from a pass's state: `[row::COUNT, n]` f32.
fn finish<B: Backend>(
    st: Rows<B>,
    targets: IntTensor<B>,
    options: &CrossEntropyOptions,
    classes: usize,
) -> FloatTensor<B> {
    let n = st.m.shape()[0];
    let device = st.m.device();
    let bool_dtype = get_or_init_device_settings::<B>(&device).bool_dtype;
    let (eps, lambda) = (options.label_smoothing, options.z_loss);
    let scale = |x: FloatTensor<B>, k: f32| B::float_mul_scalar(x, Scalar::from(k));

    let ln_s = B::float_log(st.s.clone());
    let lse = B::float_add(st.m, ln_s.clone());
    let found = B::float_greater_elem(st.found.clone(), Scalar::from(0.5), bool_dtype);
    let counted = B::int_not_equal_elem(targets, Scalar::from(options.ignore_index), bool_dtype);
    let logprob = B::float_mask_where(
        full::<B>([n], f32::NEG_INFINITY, &device),
        found.clone(),
        B::float_sub(st.zt.clone(), lse.clone()),
    );
    // (1 - eps) (lse - zt), +inf where the target is not seen; no term at all for eps = 1.
    let mut loss = match eps < 1.0 {
        true => scale(
            B::float_mask_where(
                full::<B>([n], f32::INFINITY, &device),
                found.clone(),
                B::float_sub(lse.clone(), st.zt),
            ),
            1.0 - eps,
        ),
        false => full::<B>([n], 0.0, &device),
    };
    if let Some(zsum) = st.zsum {
        // eps (lse - mean_j z)
        let smooth = B::float_sub(lse.clone(), scale(zsum, 1.0 / classes as f32));
        loss = B::float_add(loss, scale(smooth, eps));
    }
    if lambda != 0.0 {
        loss = B::float_add(loss, scale(B::float_mul(lse.clone(), lse.clone()), lambda));
    }
    let entropy = B::float_sub(ln_s, B::float_div(st.a, st.s.clone()));
    let q = B::float_div(st.r, st.s.clone());
    let zero_ignored =
        |x: FloatTensor<B>| B::float_mask_where(full::<B>([n], 0.0, &device), counted.clone(), x);
    let as_float = |mask| B::bool_into_float(mask, FloatDType::F32);
    let mut out = vec![None; row::COUNT];
    out[row::LOSS] = Some(zero_ignored(loss));
    out[row::LOGPROB] = Some(zero_ignored(logprob));
    out[row::ENTROPY] = Some(zero_ignored(entropy));
    out[row::LSE] = Some(lse);
    out[row::Q] = Some(q);
    out[row::S] = Some(st.s);
    out[row::COUNTED] = Some(as_float(counted.clone()));
    out[row::FOUND] = Some(as_float(found));
    stack::<B>(out.into_iter().map(|x| x.expect("every row")).collect(), n)
}

/// What the backward pass keeps from the forward pass.
#[derive(Debug, Clone)]
struct Saved<B: Backend> {
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    targets: IntTensor<B>,
    /// `[row::COUNT, n]` f32.
    rows: FloatTensor<B>,
    /// `[n, D]` f32 if the forward pass accumulated it, else `[0, D]`.
    o: FloatTensor<B>,
}

/// The backward pass on any backend, given its gradient pass: `(grad_hidden, grad_weight)`.
fn backward<B: Backend>(
    saved: Saved<B>,
    upstream: FloatTensor<B>,
    options: &CrossEntropyOptions,
    want: [bool; 2],
    gradient_pass: impl FnOnce(
        FloatTensor<B>,
        FloatTensor<B>,
        IntTensor<B>,
        FloatTensor<B>,
        [bool; 2],
    ) -> (FloatTensor<B>, FloatTensor<B>),
) -> (FloatTensor<B>, FloatTensor<B>) {
    let Saved {
        hidden,
        weight,
        targets,
        rows,
        o,
    } = saved;
    let [n, d] = dims2(&hidden.shape(), "hidden");
    let [v, _] = dims2(&weight.shape(), "weight");
    let (hidden_dtype, weight_dtype) = (hidden.dtype(), weight.dtype());
    let device = hidden.device();
    let none = |dtype: DType| cast::<B>(full::<B>([0, d], 0.0, &device), dtype);
    if n == 0 {
        let grad_weight = match want[1] {
            true => cast::<B>(full::<B>([v, d], 0.0, &device), weight_dtype),
            false => none(weight_dtype),
        };
        return (none(hidden_dtype), grad_weight);
    }
    let masked = sorted(&options.masked);
    let classes = v - masked.len();
    let coef = coefficients::<B>(rows.clone(), to_f32::<B>(upstream), options, classes);

    let accumulated = o.shape()[0] == n && !want[1] && !options.entropy;
    if !accumulated {
        probe::GRADIENT_PASSES.fetch_add(1, Ordering::SeqCst);
        return gradient_pass(hidden, weight, targets, coef, want);
    }
    // grad_hidden = (A / s) o + T weight[t] + C sum_{j in U} weight[j], per row.
    let column = |x: FloatTensor<B>, k: usize| {
        let x = B::float_slice(x, &[Slice::from(k..k + 1), Slice::full()]);
        B::float_reshape(x, Shape::new([n, 1]))
    };
    let softmax = B::float_div(
        column(coef.clone(), coefficient::SOFTMAX),
        column(rows, row::S),
    );
    let mut grad = B::float_mul(o, softmax);
    let weight_t = reference::target_rows::<B>(weight.clone(), targets, v);
    grad = B::float_add(
        grad,
        B::float_mul(weight_t, column(coef.clone(), coefficient::TARGET)),
    );
    if options.label_smoothing != 0.0 {
        let total = vocab_sum::<B>(weight, &masked, options.chunks.vocab_chunk_for(n));
        grad = B::float_add(
            grad,
            B::float_mul(total, column(coef, coefficient::UNIFORM)),
        );
    }
    let grad_hidden = match want[0] {
        true => cast::<B>(grad, hidden_dtype),
        false => none(hidden_dtype),
    };
    (grad_hidden, none(weight_dtype))
}

/// The per-row coefficients of the gradient pass, `[coefficient::COUNT, n]` f32, from the
/// forward's rows and their upstream gradient (see the module docs).
fn coefficients<B: Backend>(
    rows: FloatTensor<B>,
    upstream: FloatTensor<B>,
    options: &CrossEntropyOptions,
    classes: usize,
) -> FloatTensor<B> {
    let n = rows.shape()[1];
    let device = rows.device();
    let bool_dtype = get_or_init_device_settings::<B>(&device).bool_dtype;
    let (eps, lambda) = (options.label_smoothing, options.z_loss);
    let at = |x: &FloatTensor<B>, k: usize| {
        let x = B::float_slice(x.clone(), &[Slice::from(k..k + 1), Slice::full()]);
        B::float_reshape(x, Shape::new([n]))
    };
    let flag = |k: usize| B::float_greater_elem(at(&rows, k), Scalar::from(0.5), bool_dtype);
    let zeros = || full::<B>([n], 0.0, &device);
    let scale = |x: FloatTensor<B>, k: f32| B::float_mul_scalar(x, Scalar::from(k));
    // The upstream gradients, 0 for an ignored row whatever they hold.
    let counted = flag(row::COUNTED);
    let up = |k: usize| B::float_mask_where(zeros(), counted.clone(), at(&upstream, k));
    let (gl, gp) = (up(row::LOSS), up(row::LOGPROB));
    let ge = options.entropy.then(|| up(row::ENTROPY));
    let lse = at(&rows, row::LSE);
    let q = at(&rows, row::Q);

    // A = gl (1 + 2 lambda lse) - gp - ge entropy
    let mut softmax = B::float_sub(gl.clone(), gp);
    if lambda != 0.0 {
        softmax = B::float_add(
            softmax,
            B::float_mul(gl.clone(), scale(lse.clone(), 2.0 * lambda)),
        );
    }
    // T = gl (eps + 2 lambda lse) - ge entropy - q A - ge (1 - q) logprob, where the target is
    // seen; the gradient of the target's logit is T + C.
    let mut target = B::float_neg(B::float_mul(q.clone(), softmax.clone()));
    let mut gl_target = scale(gl.clone(), eps);
    if lambda != 0.0 {
        gl_target = B::float_add(
            gl_target,
            B::float_mul(gl.clone(), scale(lse.clone(), 2.0 * lambda)),
        );
    }
    target = B::float_add(target, gl_target);
    if let Some(ge) = &ge {
        let entropy = B::float_mul(ge.clone(), at(&rows, row::ENTROPY));
        softmax = B::float_sub(softmax, entropy.clone());
        // A takes -ge entropy, so -q A takes +q ge entropy; with T's own -ge entropy that
        // leaves -(1 - q) ge entropy, beside -ge (1 - q) logprob.
        let p_target = B::float_neg(B::float_sub_scalar(q.clone(), Scalar::from(1.0)));
        target = B::float_sub(
            target,
            B::float_add(
                B::float_mul(p_target.clone(), entropy),
                B::float_mul(B::float_mul(ge.clone(), p_target), at(&rows, row::LOGPROB)),
            ),
        );
    }
    let target = B::float_mask_where(zeros(), flag(row::FOUND), target);
    let uniform = scale(gl, -eps / classes as f32);

    let mut out = vec![None; coefficient::COUNT];
    out[coefficient::LSE] = Some(lse);
    out[coefficient::SOFTMAX] = Some(softmax);
    out[coefficient::ENTROPY] = Some(ge.unwrap_or_else(zeros));
    out[coefficient::TARGET] = Some(target);
    out[coefficient::UNIFORM] = Some(uniform);
    stack::<B>(out.into_iter().map(|x| x.expect("every row")).collect(), n)
}

/// `sum_{j in U} weight[j]`, `[1, D]` f32, chunk by chunk of `chunk` rows.
fn vocab_sum<B: Backend>(weight: FloatTensor<B>, masked: &[u32], chunk: usize) -> FloatTensor<B> {
    let [v, d] = dims2(&weight.shape(), "weight");
    let device = weight.device();
    let mut total = full::<B>([1, d], 0.0, &device);
    for c0 in (0..v).step_by(chunk) {
        let rows = B::float_slice(
            weight.clone(),
            &[Slice::from(c0..(c0 + chunk).min(v)), Slice::full()],
        );
        total = B::float_add(total, B::float_sum_dim(to_f32::<B>(rows), 0));
    }
    if !masked.is_empty() {
        let int_dtype = get_or_init_device_settings::<B>(&device).int_dtype;
        let ids = masked.iter().map(|&id| id as i64).collect::<Vec<_>>();
        // In the device's int dtype from the start: not every device computes in i64.
        let ids = TensorData::new(ids, [masked.len()]).convert_dtype(int_dtype.into());
        let ids = B::int_from_data(ids, &device);
        let rows = to_f32::<B>(B::float_select(weight, 0, ids));
        total = B::float_sub(total, B::float_sum_dim(rows, 0));
    }
    total
}

/// `[k, n]` from `k` tensors `[n]`.
fn stack<B: Backend>(rows: Vec<FloatTensor<B>>, n: usize) -> FloatTensor<B> {
    let rows = rows
        .into_iter()
        .map(|x| B::float_reshape(x, Shape::new([1, n])))
        .collect();
    B::float_cat(rows, 0)
}

impl<B: Backend + LinearCrossEntropyBackend, C: CheckpointStrategy> LinearCrossEntropyBackend
    for Autodiff<B, C>
{
    fn linear_cross_entropy(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        options: CrossEntropyOptions,
        _accumulate: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        /// The backward pass: the inner backend's, for whichever inputs require a gradient.
        #[derive(Debug)]
        struct LinearCrossEntropyNode;

        impl<B: Backend + LinearCrossEntropyBackend> Backward<B, 2> for LinearCrossEntropyNode {
            type State = (Saved<B>, CrossEntropyOptions);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                probe::BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
                let [hidden_node, weight_node] = ops.parents;
                let upstream = grads.consume::<B>(&ops.node);
                let (saved, options) = ops.state;
                let (grad_hidden, grad_weight) = B::linear_cross_entropy_backward(
                    saved.hidden,
                    saved.weight,
                    saved.targets,
                    saved.rows,
                    upstream,
                    saved.o,
                    options,
                    hidden_node.is_some(),
                    weight_node.is_some(),
                );
                if let Some(node) = hidden_node {
                    grads.register::<B>(node.id, grad_hidden);
                }
                if let Some(node) = weight_node {
                    grads.register::<B>(node.id, grad_weight);
                }
            }
        }

        // `o` serves the gradient of `hidden` when that is the only one: a weight that requires
        // a gradient, or the entropy's, takes the second pass anyway.
        let accumulate = hidden.is_tracked() && !weight.is_tracked() && !options.entropy;
        let (hidden, hidden_node) = hidden.into_parts();
        let (weight, weight_node) = weight.into_parts();
        match LinearCrossEntropyNode
            .prepare::<C>([hidden_node, weight_node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (rows, o) = B::linear_cross_entropy(
                    hidden.clone(),
                    weight.clone(),
                    targets.clone(),
                    options.clone(),
                    accumulate,
                );
                let saved = Saved {
                    hidden,
                    weight,
                    targets,
                    rows: rows.clone(),
                    o: o.clone(),
                };
                (
                    prep.finish((saved, options), rows),
                    <Self as AutodiffBackend>::from_inner(o),
                )
            }
            OpsKind::UnTracked(prep) => {
                let (rows, o) = B::linear_cross_entropy(hidden, weight, targets, options, false);
                (prep.finish(rows), <Self as AutodiffBackend>::from_inner(o))
            }
        }
    }

    fn linear_cross_entropy_backward(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        rows: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        o: FloatTensor<Self>,
        options: CrossEntropyOptions,
        grad_hidden: bool,
        grad_weight: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (grad_hidden, grad_weight) = B::linear_cross_entropy_backward(
            hidden.into_primitive(),
            weight.into_primitive(),
            targets,
            rows.into_primitive(),
            upstream.into_primitive(),
            o.into_primitive(),
            options,
            grad_hidden,
            grad_weight,
        );
        (
            <Self as AutodiffBackend>::from_inner(grad_hidden),
            <Self as AutodiffBackend>::from_inner(grad_weight),
        )
    }
}

/// Cross-entropy of `softmax(hidden @ weight^T)` (the `masked` ids removed) against `targets`,
/// with each row's log-probability of its target and, if asked, its entropy, without holding
/// the n x V logits. See [`dash_contract::loss`] for the exact definition.
///
/// - `hidden` `[n, D]`: bf16 for the kernels' fast path, or f32 (f16 is computed in f32); on an
///   autodiff device it may require a gradient, and gets one.
/// - `weight` `[V, D]`, the same dtype: it may require a gradient too, and gets one.
/// - `targets` `[n]` token ids; [`CrossEntropyOptions::ignore_index`] marks rows to ignore.
///
/// # Panics
///
/// On mismatched shapes, a masked id `>= V` or every id masked, a zero `row_chunk`, a label
/// smoothing outside `0..=1`, or a negative or non-finite z-loss coefficient.
pub fn linear_cross_entropy(
    hidden: Tensor<2>,
    weight: Tensor<2>,
    targets: Tensor<1, Int>,
    options: &CrossEntropyOptions,
) -> CrossEntropy {
    let [n, d] = hidden.dims();
    let [v, dw] = weight.dims();
    let [nt] = targets.dims();
    assert_eq!(
        d, dw,
        "hidden [n, D] and weight [V, D] must share D: got {d} and {dw}"
    );
    assert_eq!(nt, n, "targets must have one id per row of hidden");
    assert!(v > 0, "the vocabulary is empty");
    assert!(
        options.chunks.row_chunk > 0,
        "row_chunk must be at least 1: {:?}",
        options.chunks
    );
    assert!(
        (0.0..=1.0).contains(&options.label_smoothing),
        "label_smoothing must be in 0..=1, got {}",
        options.label_smoothing
    );
    assert!(
        options.z_loss.is_finite() && options.z_loss >= 0.0,
        "z_loss must be finite and at least 0, got {}",
        options.z_loss
    );
    let masked = sorted(&options.masked);
    if let Some(&last) = masked.last() {
        assert!(
            (last as usize) < v,
            "masked token id {last} is outside the vocabulary of {v}"
        );
        assert!(masked.len() < v, "every token of the vocabulary is masked");
    }
    let options = CrossEntropyOptions {
        masked,
        ..options.clone()
    };
    let (rows, _) = <Dispatch as LinearCrossEntropyBackend>::linear_cross_entropy(
        hidden.into_dispatch(),
        weight.into_dispatch(),
        targets.into_dispatch(),
        options.clone(),
        false,
    );
    let rows = Tensor::<2>::from_dispatch(rows);
    // Burn's slicing returns an untracked tensor for an empty result, so an empty batch takes
    // all of `rows` (no element either way) and stays on the graph.
    let at = |k: usize| match n {
        0 => rows.clone().flatten::<1>(0, 1),
        _ => rows.clone().narrow(0, k, 1).squeeze_dim::<1>(0),
    };
    let losses = at(row::LOSS);
    let loss = match options.reduction {
        Reduction::None => losses,
        Reduction::Sum => losses.sum(),
        Reduction::Mean => losses.sum() / at(row::COUNTED).detach().sum(),
    };
    CrossEntropy {
        loss,
        logprob: at(row::LOGPROB),
        entropy: options.entropy.then(|| at(row::ENTROPY)),
    }
}
