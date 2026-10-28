//! The reference: `dash_contract::sample`'s definition in Burn's tensor operations, in f32. Every
//! backend runs it; the tests hold the kernel to it. The cuts come from the row sorted in
//! descending order: the `k`-th largest logit is the sorted row's entry `k - 1`, and top-p's logit
//! the sorted entry at which the cumulative sum of the sorted weights first reaches `p` of their
//! total. The Gumbel words are drawn on the host ([`super::philox`]) and the variates computed
//! from them here, as the kernel computes them.
//!
//! Every sum over a row is a balanced tree of pairs ([`tree_sum`], [`tree_cumsum`]), so that its
//! rounding grows with `log V` whatever order the backend's own reductions take. Burn's sum on
//! CUDA and a running f32 cumulative sum both lose the small weights at a vocabulary of 248320
//! (a relative error near `1e-4` in the log-sum-exp; the top-p cut moved by a whole tie group of
//! bf16 logits), where the trees stay within a few `1e-7` of the sums in f64.

use super::{Cuts, SampleOptions, philox};
use burn::backend::tensor::{BoolTensor, FloatTensor, IntTensor};
use burn::backend::{
    Backend, DType, FloatDType, IntDType, Scalar, Shape, Slice, TensorData, TensorMetadata,
    get_or_init_device_settings,
};

/// `(tokens, logprob, tempered_logprob, sampling_logprob)`: the tokens `[B]` in `out_dtype`, the
/// three log-probabilities `[B]` f32. `options.masked` is sorted, free of repeats, each id `< V`
/// and fewer than `V`.
pub(crate) fn sample<B: Backend>(
    logits: FloatTensor<B>,
    options: &SampleOptions,
    out_dtype: IntDType,
) -> (IntTensor<B>, FloatTensor<B>, FloatTensor<B>, FloatTensor<B>) {
    let shape = logits.shape();
    assert_eq!(
        shape.num_dims(),
        2,
        "sample: logits must be [B, V], got {shape:?}"
    );
    let [b, v] = shape.dims::<2>();
    let device = logits.device();
    if b == 0 {
        return (
            B::int_zeros(Shape::new([0]), &device, out_dtype),
            full::<B>([0], 0.0, &device),
            full::<B>([0], 0.0, &device),
            full::<B>([0], 0.0, &device),
        );
    }
    let bool_dtype = get_or_init_device_settings::<B>(&device).bool_dtype;
    let column = |x: FloatTensor<B>| B::float_reshape(x, Shape::new([b]));

    let z = to_f32::<B>(logits);
    let mask = masked_ids::<B>(&options.masked, [b, v], &device);
    let zm = match &mask {
        Some(mask) => B::float_mask_fill(z.clone(), mask.clone(), Scalar::from(f32::NEG_INFINITY)),
        None => z.clone(),
    };
    // m and the log-sum-exp over the unmasked tokens, `[B, 1]`.
    let m = B::float_max_dim(zm.clone(), 1);
    let s1 = tree_sum::<B>(B::float_exp(B::float_sub(zm.clone(), m.clone())));
    let lse = B::float_add(m.clone(), B::float_log(s1));
    // A row with no finite unmasked logit takes its lowest unmasked id.
    let degenerate = B::float_equal_elem(m.clone(), Scalar::from(f32::NEG_INFINITY), bool_dtype);
    let first_unmasked = match &mask {
        Some(mask) => {
            let unmasked = B::float_mask_fill(
                full::<B>([b, v], 1.0, &device),
                mask.clone(),
                Scalar::from(0.0),
            );
            B::float_argmax(unmasked, 1, out_dtype)
        }
        None => B::int_zeros(Shape::new([b, 1]), &device, out_dtype),
    };

    // The tokens, and the tempered and the sampling log-probabilities `[B, 1]` where they are not
    // another output's: `None` takes `logprob` for the tempered one at tau = 1, and the tempered
    // one for the sampling one where nothing is cut.
    let (tokens, tempered, sampling) = if options.is_greedy() {
        let tokens = B::float_argmax(zm.clone(), 1, out_dtype);
        // The point mass at the token, for both.
        let zero = full::<B>([b, 1], 0.0, &device);
        (tokens, Some(zero.clone()), Some(zero))
    } else {
        let tau = options.inverse_temperature();
        let unmasked = v - options.masked.len();
        let s = B::float_mul_scalar(B::float_sub(zm.clone(), m.clone()), Scalar::from(tau));
        let kept = kept::<B>(&zm, &m, mask.as_ref(), options, unmasked, tau, out_dtype);
        let cut = kept.is_some();
        let noise = gumbel::<B>(options, [b, v], &device, bool_dtype);
        let leave_out = match (kept, &mask) {
            (Some(kept), _) => Some(B::bool_not(kept)),
            (None, Some(mask)) => Some(mask.clone()),
            (None, None) => None,
        };
        let drop = |x: FloatTensor<B>, value: f32| match &leave_out {
            Some(leave_out) => B::float_mask_fill(x, leave_out.clone(), Scalar::from(value)),
            None => x,
        };
        let score = drop(B::float_add(s.clone(), noise), f32::NEG_INFINITY);
        let tokens = B::float_argmax(score, 1, out_dtype);
        let weights = B::float_exp(s.clone());
        let s_t = B::float_gather(1, s, tokens.clone());
        // Over U: the masked tokens' weights are exp(-inf) = 0.
        let tempered = (tau != 1.0).then(|| {
            let total = tree_sum::<B>(weights.clone());
            B::float_sub(s_t.clone(), B::float_log(total))
        });
        // Over P, where the cuts leave it short of U.
        let sampling = cut.then(|| {
            let total = tree_sum::<B>(drop(weights, 0.0));
            B::float_sub(s_t, B::float_log(total))
        });
        (tokens, tempered, sampling)
    };
    let tokens = B::int_mask_where(tokens, degenerate.clone(), first_unmasked);
    let z_t = B::float_gather(1, z, tokens.clone());
    let logprob = B::float_sub(z_t, lse);
    let tempered = tempered.unwrap_or_else(|| logprob.clone());
    let sampling = sampling.unwrap_or_else(|| tempered.clone());
    let nan = |x: FloatTensor<B>| {
        B::float_mask_where(x, degenerate.clone(), full::<B>([b, 1], f32::NAN, &device))
    };
    (
        B::int_reshape(tokens, Shape::new([b])),
        column(logprob),
        column(nan(tempered)),
        column(nan(sampling)),
    )
}

/// The tokens top-k and top-p keep, `[B, V]` (`P` in the definition), or `None` when
/// neither cuts: every unmasked token. `zm` is the row with the masked logits at `-inf`, `m` its
/// maximum `[B, 1]`.
#[allow(clippy::too_many_arguments)]
fn kept<B: Backend>(
    zm: &FloatTensor<B>,
    m: &FloatTensor<B>,
    mask: Option<&BoolTensor<B>>,
    options: &SampleOptions,
    unmasked: usize,
    tau: f32,
    index_dtype: IntDType,
) -> Option<BoolTensor<B>> {
    let top_k = options.top_k_cut(unmasked);
    let top_p = options.top_p_cut();
    if top_k.is_none() && top_p.is_none() {
        return None;
    }
    let [b, v] = zm.shape().dims::<2>();
    let device = zm.device();
    let bool_dtype = get_or_init_device_settings::<B>(&device).bool_dtype;
    // Descending, the masked tokens last (their logits are -inf).
    let sorted = B::float_sort(zm.clone(), 1, true);
    let entry = |k: usize| B::float_slice(sorted.clone(), &[Slice::full(), Slice::from(k..k + 1)]);
    // The k-th largest logit of the unmasked tokens: those at -inf sort after every other, so the
    // entry is theirs only when fewer than k are finite, and is then -inf.
    let mut threshold = top_k.map(|k| entry(k - 1));
    if let Some(p) = top_p {
        let mut weights = B::float_exp(B::float_mul_scalar(
            B::float_sub(sorted.clone(), m.clone()),
            Scalar::from(tau),
        ));
        if let Some(top_k) = &threshold {
            let outside = B::float_lower(sorted.clone(), top_k.clone(), bool_dtype);
            weights = B::float_mask_fill(weights, outside, Scalar::from(0.0));
        }
        let cumulative = tree_cumsum::<B>(weights);
        let total = B::float_slice(cumulative.clone(), &[Slice::full(), Slice::from(v - 1..v)]);
        let target = B::float_mul_scalar(total, Scalar::from(p));
        // The first entry that reaches the target (the last one does: the target is at most the
        // total), the first maximum of the indicator as argmax takes it.
        let reached = B::float_greater_equal(cumulative, target, bool_dtype);
        let first = B::float_argmax(B::bool_into_float(reached, FloatDType::F32), 1, index_dtype);
        threshold = Some(B::float_gather(1, sorted, first));
    }
    let threshold = threshold.expect("a cut");
    let z_kept = B::float_greater_equal(
        zm.clone(),
        B::float_expand(threshold, Shape::new([b, v])),
        bool_dtype,
    );
    Some(match mask {
        Some(mask) => B::bool_and(z_kept, B::bool_not(mask.clone())),
        None => z_kept,
    })
}

/// The sum of each row of `x` `[B, V]`, `[B, 1]`: the entries summed in pairs, the pairs' sums in
/// pairs, and so on (a row of odd length takes a zero), `ceil(log2 V)` halvings.
fn tree_sum<B: Backend>(x: FloatTensor<B>) -> FloatTensor<B> {
    let [b, mut v] = x.shape().dims::<2>();
    let device = x.device();
    let mut x = x;
    while v > 1 {
        if v % 2 == 1 {
            x = B::float_cat(vec![x, full::<B>([b, 1], 0.0, &device)], 1);
            v += 1;
        }
        let pairs = B::float_sum_dim(B::float_reshape(x, Shape::new([b, v / 2, 2])), 2);
        v /= 2;
        x = B::float_reshape(pairs, Shape::new([b, v]));
    }
    x
}

/// The inclusive cumulative sums of each row of `x` `[B, V]`: `ceil(log2 V)` steps, each adding to
/// every entry the one `2^k` before it (Hillis and Steele's scan), so that each sum is a tree of
/// that depth.
fn tree_cumsum<B: Backend>(x: FloatTensor<B>) -> FloatTensor<B> {
    let [b, v] = x.shape().dims::<2>();
    let device = x.device();
    let mut x = x;
    let mut shift = 1;
    while shift < v {
        let earlier = B::float_slice(x.clone(), &[Slice::full(), Slice::from(0..v - shift)]);
        let shifted = B::float_cat(vec![full::<B>([b, shift], 0.0, &device), earlier], 1);
        x = B::float_add(x, shifted);
        shift *= 2;
    }
    x
}

/// The Gumbel variates `g` of every token of a `[B, V]` sample, f32: the words drawn on the host,
/// the variates computed here from the signed inputs [`philox::gumbel_input`] gives.
fn gumbel<B: Backend>(
    options: &SampleOptions,
    [b, v]: [usize; 2],
    device: &B::Device,
    bool_dtype: burn::backend::BoolDType,
) -> FloatTensor<B> {
    let mut inputs = Vec::with_capacity(b * v);
    for row in 0..b {
        inputs.extend(philox::row_inputs(
            options.seed,
            options.offset,
            row as u32,
            v,
        ));
    }
    let c = B::float_from_data(TensorData::new(inputs, [b, v]), device);
    gumbel_of_inputs::<B>(c, bool_dtype)
}

/// The Gumbel variates of the signed inputs [`philox::gumbel_input`] gives, any shape, f32, in the
/// kernel's operations and order.
pub(crate) fn gumbel_of_inputs<B: Backend>(
    c: FloatTensor<B>,
    bool_dtype: burn::backend::BoolDType,
) -> FloatTensor<B> {
    // E = -ln a for a positive input; for a negative one (whose value is -a) -ln_1p(-a), in the
    // module documentation's two forms, not Burn's log1p (CubeCL's `ln(1 + x)`).
    let upper = B::float_lower_elem(c.clone(), Scalar::from(0.0f32), bool_dtype);
    let a = B::float_abs(c);
    let e_lower = B::float_neg(B::float_log(a.clone()));
    // a (6 - a) / (6 - 4a)
    let numerator = B::float_add_scalar(B::float_neg(a.clone()), Scalar::from(6.0f32));
    let denominator = B::float_add_scalar(
        B::float_mul_scalar(a.clone(), Scalar::from(-4.0f32)),
        Scalar::from(6.0f32),
    );
    let e_small = B::float_mul(a.clone(), B::float_div(numerator, denominator));
    // -ln(w) a / (1 - w), w = 1 - a
    let w = B::float_add_scalar(B::float_neg(a.clone()), Scalar::from(1.0f32));
    let one_less_w = B::float_add_scalar(B::float_neg(w.clone()), Scalar::from(1.0f32));
    let e_large = B::float_mul(
        B::float_neg(B::float_log(w)),
        B::float_div(a.clone(), one_less_w),
    );
    let small = B::float_lower_equal_elem(a, Scalar::from(2f32.powi(-6)), bool_dtype);
    let e_upper = B::float_mask_where(e_large, small, e_small);
    let e = B::float_mask_where(e_lower, upper, e_upper);
    B::float_neg(B::float_log(e))
}

/// `true` at the masked ids, `[B, V]` in the device's bool dtype; `None` without any.
fn masked_ids<B: Backend>(
    masked: &[u32],
    [b, v]: [usize; 2],
    device: &B::Device,
) -> Option<BoolTensor<B>> {
    if masked.is_empty() {
        return None;
    }
    let settings = get_or_init_device_settings::<B>(device);
    let mut row = vec![false; v];
    for &id in masked {
        row[id as usize] = true;
    }
    // In the device's bool dtype: CubeCL devices store bools as integers.
    let row = TensorData::new(row, [1, v]).convert_dtype(settings.bool_dtype.into());
    Some(B::bool_expand(
        B::bool_from_data(row, device),
        Shape::new([b, v]),
    ))
}

fn to_f32<B: Backend>(x: FloatTensor<B>) -> FloatTensor<B> {
    match x.dtype() {
        DType::F32 => x,
        _ => B::float_cast(x, FloatDType::F32),
    }
}

fn full<B: Backend>(shape: impl Into<Shape>, value: f32, device: &B::Device) -> FloatTensor<B> {
    B::float_full(shape.into(), Scalar::from(value), device, FloatDType::F32)
}
