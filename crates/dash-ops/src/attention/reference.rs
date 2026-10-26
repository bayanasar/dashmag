//! The reference: attention as the textbook writes it, built from Burn's tensor operations, in
//! f32 whatever the input dtype. It works on `[B, H, T, D]` views of the inputs (one batched
//! matmul per head), expands the key-value heads and materialises the T x T scores; it is what
//! Flex runs and what the kernels are held to, not what training on a GPU should run.

use burn::backend::tensor::{BoolTensor, FloatTensor};
use burn::backend::{Backend, FloatDType, Shape, TensorMetadata, get_or_init_device_settings};

fn dims4<B: Backend>(tensor: &FloatTensor<B>) -> [usize; 4] {
    tensor.shape().dims::<4>()
}

/// `[B, T, H, ...]` as `[B, H, T, ...]` and back: a view, no copy.
fn swap_time_and_heads<B: Backend>(tensor: FloatTensor<B>) -> FloatTensor<B> {
    B::float_swap_dims(tensor, 1, 2)
}

/// `[B, Hkv, T, D]` to `[B, Hkv * group, T, D]`, each key-value head repeated for the query heads
/// of its group (query head `h` reads key-value head `h / group`).
fn expand_heads<B: Backend>(tensor: FloatTensor<B>, group: usize) -> FloatTensor<B> {
    if group == 1 {
        return tensor;
    }
    let [b, hkv, t, d] = dims4::<B>(&tensor);
    let tensor = B::float_reshape(tensor, Shape::new([b, hkv, 1, t, d]));
    let tensor = B::float_expand(tensor, Shape::new([b, hkv, group, t, d]));
    B::float_reshape(tensor, Shape::new([b, hkv * group, t, d]))
}

/// The inverse of [`expand_heads`] for gradients: the sum over the query heads of each group.
fn sum_groups<B: Backend>(tensor: FloatTensor<B>, group: usize) -> FloatTensor<B> {
    if group == 1 {
        return tensor;
    }
    let [b, hq, t, d] = dims4::<B>(&tensor);
    let tensor = B::float_reshape(tensor, Shape::new([b, hq / group, group, t, d]));
    let tensor = B::float_sum_dim(tensor, 2);
    B::float_reshape(tensor, Shape::new([b, hq / group, t, d]))
}

/// True where key `j` is hidden from query `i`: `j > i + (tk - tq)`, broadcast to the scores'
/// `[B, H, Tq, Tk]`.
fn causal_mask<B: Backend>(scores: &FloatTensor<B>) -> BoolTensor<B> {
    let device = scores.device();
    let [b, h, tq, tk] = dims4::<B>(scores);
    let settings = get_or_init_device_settings::<B>(&device);
    let rows = B::int_reshape(
        B::int_arange(0..tq as i64, &device, settings.int_dtype),
        Shape::new([tq, 1]),
    );
    let cols = B::int_reshape(
        B::int_arange(0..tk as i64, &device, settings.int_dtype),
        Shape::new([1, tk]),
    );
    let rows = B::int_add_scalar(rows, ((tk - tq) as i64).into());
    let mask = B::int_lower(rows, cols, settings.bool_dtype);
    let mask = B::bool_reshape(mask, Shape::new([1, 1, tq, tk]));
    B::bool_expand(mask, Shape::new([b, h, tq, tk]))
}

/// `scale * q k^T` with the causal mask applied as `-inf`, in f32, `[B, Hq, Tq, Tk]`, and the
/// expanded f32 keys, from `[B, H, T, D]` views.
fn masked_scores<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    scale: f32,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let group = dims4::<B>(&query)[1] / dims4::<B>(&key)[1];
    let key = expand_heads::<B>(B::float_cast(key, FloatDType::F32), group);
    let scores = B::float_matmul(query, B::float_transpose(key.clone()));
    let scores = B::float_mul_scalar(scores, scale.into());
    let mask = causal_mask::<B>(&scores);
    (
        B::float_mask_fill(scores, mask, f32::NEG_INFINITY.into()),
        key,
    )
}

/// The output `[B, Tq, Hq, D]`, in the dtype of `query`, and the natural log-sum-exp of each
/// query's scaled, masked scores, `[B, Tq, Hq]` in f32.
pub(crate) fn forward<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    scale: f32,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let dtype = query.dtype();
    let [b, tq, hq, _] = dims4::<B>(&query);
    let group = hq / dims4::<B>(&key)[2];
    let query = swap_time_and_heads::<B>(B::float_cast(query, FloatDType::F32));
    let key = swap_time_and_heads::<B>(key);
    let value = swap_time_and_heads::<B>(B::float_cast(value, FloatDType::F32));
    let value = expand_heads::<B>(value, group);
    let (scores, _) = masked_scores::<B>(query, key, scale);
    // Every query sees at least key 0 (Tk >= Tq), so each row's maximum is finite.
    let max = B::float_max_dim(scores.clone(), 3);
    let exp = B::float_exp(B::float_sub(scores, max.clone()));
    let sum = B::float_sum_dim(exp.clone(), 3);
    let output = B::float_matmul(B::float_div(exp, sum.clone()), value);
    let lse = B::float_add(max, B::float_log(sum));
    let lse = B::float_reshape(lse, Shape::new([b, hq, tq]));
    (
        swap_time_and_heads::<B>(B::float_cast(output, FloatDType::from(dtype))),
        swap_time_and_heads::<B>(lse),
    )
}

/// The gradients of `query`, `key` and `value`, each in its input's layout and dtype: with
/// `P = exp(S - lse)`, `dV = P^T dO`, `dS = P * (dO V^T - rowsum(dO * O))`, `dQ = scale dS K`,
/// `dK = scale dS^T Q`, each key-value gradient summed over the query heads of its group.
pub(crate) fn backward<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    output: FloatTensor<B>,
    lse: FloatTensor<B>,
    grad_output: FloatTensor<B>,
    scale: f32,
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>) {
    let (query_dtype, key_dtype, value_dtype) = (query.dtype(), key.dtype(), value.dtype());
    let [b, tq, hq, _] = dims4::<B>(&query);
    let group = hq / dims4::<B>(&key)[2];
    let f32 = |tensor| B::float_cast(tensor, FloatDType::F32);
    let query = swap_time_and_heads::<B>(f32(query));
    let key = swap_time_and_heads::<B>(key);
    let value = expand_heads::<B>(swap_time_and_heads::<B>(f32(value)), group);
    let output = swap_time_and_heads::<B>(f32(output));
    let grad_output = swap_time_and_heads::<B>(f32(grad_output));
    let lse = B::float_reshape(
        B::float_swap_dims(f32(lse), 1, 2),
        Shape::new([b, hq, tq, 1]),
    );

    let (scores, key) = masked_scores::<B>(query.clone(), key, scale);
    let probs = B::float_exp(B::float_sub(scores, lse));
    let grad_value = B::float_matmul(B::float_transpose(probs.clone()), grad_output.clone());
    let grad_probs = B::float_matmul(grad_output.clone(), B::float_transpose(value));
    let delta = B::float_sum_dim(B::float_mul(grad_output, output), 3);
    let grad_scores = B::float_mul(probs, B::float_sub(grad_probs, delta));
    let grad_query = B::float_mul_scalar(B::float_matmul(grad_scores.clone(), key), scale.into());
    let grad_key = B::float_mul_scalar(
        B::float_matmul(B::float_transpose(grad_scores), query),
        scale.into(),
    );
    let back =
        |tensor, dtype| swap_time_and_heads::<B>(B::float_cast(tensor, FloatDType::from(dtype)));
    (
        back(grad_query, query_dtype),
        back(sum_groups::<B>(grad_key, group), key_dtype),
        back(sum_groups::<B>(grad_value, group), value_dtype),
    )
}

/// Decoding: `query` `[B, Hq, D]` against `key` and `value` `[B, S, Hkv, D]`, the forward with
/// one query; returns `[B, Hq, D]` in the dtype of `query`.
pub(crate) fn decode<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    scale: f32,
) -> FloatTensor<B> {
    let [b, hq, d] = query.shape().dims::<3>();
    let query = B::float_reshape(query, Shape::new([b, 1, hq, d]));
    let output = forward::<B>(query, key, value, scale).0;
    B::float_reshape(output, Shape::new([b, hq, d]))
}
