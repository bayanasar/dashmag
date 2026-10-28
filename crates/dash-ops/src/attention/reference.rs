//! The reference: attention as the textbook writes it, built from Burn's tensor operations, in
//! f32 whatever the input dtype. It works on `[B, H, T, D]` views of the inputs (one batched
//! matmul per head), expands the key-value heads, and materialises the T x T scores and, from
//! the mask and the options, which keys each query sees; it is what Flex runs and what the
//! kernels are held to, not what training on a GPU should run.

use super::{AttentionMaskPrimitive, AttentionOptions, CacheLengths, Reach};
use crate::reference::matmul;
use burn::backend::tensor::{BoolTensor, FloatTensor, IntTensor};
use burn::backend::{
    Backend, BoolDType, FloatDType, IntDType, Shape, Slice, TensorMetadata,
    get_or_init_device_settings,
};

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

/// The integer and boolean dtypes of `device`.
struct Dtypes {
    int: IntDType,
    boolean: BoolDType,
}

impl Dtypes {
    fn of<B: Backend>(device: &B::Device) -> Self {
        let settings = get_or_init_device_settings::<B>(device);
        Self {
            int: settings.int_dtype,
            boolean: settings.bool_dtype,
        }
    }
}

/// `0..n` as an Int tensor of shape `shape` (`[n, 1]` or `[1, n]`), in `int`.
fn positions<B: Backend>(
    n: usize,
    shape: [usize; 2],
    device: &B::Device,
    int: IntDType,
) -> IntTensor<B> {
    B::int_reshape(B::int_arange(0..n as i64, device, int), Shape::new(shape))
}

/// A bound of the window as an offset from the query's position: `None` when it does not bind
/// within `span` positions (keeps the integers in range).
fn bound(reach: usize, span: usize) -> Option<i64> {
    (reach < span).then_some(reach as i64)
}

/// Both operands broadcast to `[n, m]`.
fn grid<B: Backend>(column: IntTensor<B>, row: IntTensor<B>) -> (IntTensor<B>, IntTensor<B>) {
    let n = column.shape()[0];
    let m = row.shape()[1];
    (
        B::int_expand(column, Shape::new([n, m])),
        B::int_expand(row, Shape::new([n, m])),
    )
}

fn or<B: Backend>(a: Option<BoolTensor<B>>, b: BoolTensor<B>) -> BoolTensor<B> {
    match a {
        Some(a) => B::bool_or(a, b),
        None => b,
    }
}

/// True where key `j` (`[1, Tk]`, its position `rel` relative to its sequence's first key) lies
/// outside the window of query `i` (`[Tq, 1]`, at `pos`, the key position it is aligned with):
/// `rel > pos + right` or `rel < pos - left`. `[Tq, Tk]`, or `None` if neither side binds.
fn outside_window<B: Backend>(
    pos: IntTensor<B>,
    rel: IntTensor<B>,
    (left, right): (usize, usize),
    span: usize,
    dtypes: &Dtypes,
) -> Option<BoolTensor<B>> {
    let mut hidden = None;
    if let Some(right) = bound(right, span) {
        let (pos, rel) = grid::<B>(B::int_add_scalar(pos.clone(), right.into()), rel.clone());
        hidden = Some(B::int_greater(rel, pos, dtypes.boolean));
    }
    if let Some(left) = bound(left, span) {
        let (pos, rel) = grid::<B>(B::int_add_scalar(pos, (-left).into()), rel);
        hidden = Some(or::<B>(hidden, B::int_lower(rel, pos, dtypes.boolean)));
    }
    hidden
}

/// For each of `n` positions, which of the `S` sequences cut by the offsets `cu` (`[S + 1]`)
/// holds it, `[n, 1]` (clamped into `0..S`), and whether one does at all, `[n, 1]`.
fn sequences<B: Backend>(
    cu: &IntTensor<B>,
    n: usize,
    device: &B::Device,
    dtypes: &Dtypes,
) -> (IntTensor<B>, BoolTensor<B>) {
    let s = cu.shape()[0] - 1;
    let cu = B::int_cast(cu.clone(), dtypes.int);
    let at = positions::<B>(n, [n, 1], device, dtypes.int);
    let ends = B::int_reshape(
        B::int_slice(cu.clone(), &[Slice::from(1..s + 1)]),
        Shape::new([1, s]),
    );
    let (at_grid, ends) = grid::<B>(at.clone(), ends);
    // The sequence is the number of ends at or before the position.
    let passed = B::bool_into_int(
        B::int_greater_equal(at_grid, ends, dtypes.boolean),
        dtypes.int,
    );
    let index = B::int_sum_dim(passed, 1);
    let first = B::int_expand(
        B::int_reshape(
            B::int_slice(cu.clone(), &[Slice::from(0..1)]),
            Shape::new([1, 1]),
        ),
        Shape::new([n, 1]),
    );
    let last = B::int_expand(
        B::int_reshape(
            B::int_slice(cu, &[Slice::from(s..s + 1)]),
            Shape::new([1, 1]),
        ),
        Shape::new([n, 1]),
    );
    let inside = B::bool_and(
        B::int_greater_equal(at.clone(), first, dtypes.boolean),
        B::int_lower(at, last, dtypes.boolean),
    );
    (
        B::int_clamp(index, 0.into(), ((s - 1) as i64).into()),
        inside,
    )
}

/// `values[index]` for an `[n, 1]` index into the 1-D `values`, `[n, 1]`.
fn gather<B: Backend>(values: &IntTensor<B>, index: &IntTensor<B>) -> IntTensor<B> {
    let n = index.shape()[0];
    let picked = B::int_select(
        values.clone(),
        0,
        B::int_reshape(index.clone(), Shape::new([n])),
    );
    B::int_reshape(picked, Shape::new([n, 1]))
}

/// Which keys each query of packed sequences does not see, `[Tq, Tk]`: those of another sequence
/// or of none, and those outside the window within its own.
fn packed_hidden<B: Backend>(
    cu_q: IntTensor<B>,
    cu_k: IntTensor<B>,
    [tq, tk]: [usize; 2],
    reach: (usize, usize),
    device: &B::Device,
    dtypes: &Dtypes,
) -> BoolTensor<B> {
    let s = cu_q.shape()[0] - 1;
    if s == 0 || tq == 0 || tk == 0 {
        return B::bool_ones(Shape::new([tq, tk]), device, dtypes.boolean);
    }
    let cu_q = B::int_cast(cu_q, dtypes.int);
    let cu_k = B::int_cast(cu_k, dtypes.int);
    let starts = |cu: &IntTensor<B>| B::int_slice(cu.clone(), &[Slice::from(0..s)]);
    let ends = |cu: &IntTensor<B>| B::int_slice(cu.clone(), &[Slice::from(1..s + 1)]);

    let (seq_q, inside_q) = sequences::<B>(&cu_q, tq, device, dtypes);
    let (seq_k, inside_k) = sequences::<B>(&cu_k, tk, device, dtypes);
    // The query's place in its sequence, and that sequence's keys.
    let q_start = gather::<B>(&starts(&cu_q), &seq_q);
    let q_len = B::int_sub(gather::<B>(&ends(&cu_q), &seq_q), q_start.clone());
    let k_start = gather::<B>(&starts(&cu_k), &seq_q);
    let k_len = B::int_sub(gather::<B>(&ends(&cu_k), &seq_q), k_start);
    let local_q = B::int_sub(positions::<B>(tq, [tq, 1], device, dtypes.int), q_start);
    // The key position the query is aligned with, within its sequence: i + Lk - Lq.
    let pos = B::int_add(local_q, B::int_sub(k_len, q_len));
    // Each key's place in its own sequence.
    let key_start = gather::<B>(&starts(&cu_k), &seq_k);
    let rel = B::int_sub(positions::<B>(tk, [tk, 1], device, dtypes.int), key_start);
    let rel = B::int_reshape(rel, Shape::new([1, tk]));

    let (sq, sk) = grid::<B>(seq_q, B::int_reshape(seq_k, Shape::new([1, tk])));
    let other = B::int_not_equal(sq, sk, dtypes.boolean);
    let outside = B::bool_not(B::bool_and(
        B::bool_expand(inside_q, Shape::new([tq, tk])),
        B::bool_expand(
            B::bool_reshape(inside_k, Shape::new([1, tk])),
            Shape::new([tq, tk]),
        ),
    ));
    let hidden = B::bool_or(other, outside);
    match outside_window::<B>(pos, rel, reach, tq + tk + 1, dtypes) {
        Some(window) => B::bool_or(hidden, window),
        None => hidden,
    }
}

/// True where key `j` is hidden from query `i`, `[B', 1, Tq, Tk]` with `B'` 1 or `B` (it
/// broadcasts over the heads, and over the batch when `B'` is 1); `None` when every query sees
/// every key.
fn hidden<B: Backend>(
    mask: AttentionMaskPrimitive<B>,
    options: &AttentionOptions,
    [b, tq, tk]: [usize; 3],
    device: &B::Device,
) -> Option<BoolTensor<B>> {
    let dtypes = Dtypes::of::<B>(device);
    let reach = options.reach();
    // Not packed: query i sits at key position i + Tk - Tq.
    let aligned = || {
        let offset = tk as i64 - tq as i64;
        let pos = B::int_add_scalar(
            positions::<B>(tq, [tq, 1], device, dtypes.int),
            offset.into(),
        );
        let rel = positions::<B>(tk, [1, tk], device, dtypes.int);
        outside_window::<B>(pos, rel, reach, tq + tk + 1, &dtypes)
    };
    let as_4d = |mask: BoolTensor<B>| B::bool_reshape(mask, Shape::new([1, 1, tq, tk]));
    match mask {
        AttentionMaskPrimitive::None => aligned().map(as_4d),
        AttentionMaskPrimitive::KeyPadding(padding) => {
            let padding = B::bool_reshape(padding, Shape::new([b, 1, 1, tk]));
            let padding = B::bool_expand(padding, Shape::new([b, 1, tq, tk]));
            Some(match aligned() {
                Some(window) => B::bool_or(
                    padding,
                    B::bool_expand(as_4d(window), Shape::new([b, 1, tq, tk])),
                ),
                None => padding,
            })
        }
        AttentionMaskPrimitive::Packed {
            cu_seqlens_q,
            cu_seqlens_k,
        } => Some(as_4d(packed_hidden::<B>(
            cu_seqlens_q,
            cu_seqlens_k,
            [tq, tk],
            reach,
            device,
            &dtypes,
        ))),
    }
}

/// `scale * q k^T` in f32, `[B, Hq, Tq, Tk]`, `-inf` where `hidden` (broadcast to it), and the
/// expanded f32 keys, from `[B, H, T, D]` views.
fn masked_scores<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    hidden: Option<BoolTensor<B>>,
    scale: f32,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let group = dims4::<B>(&query)[1] / dims4::<B>(&key)[1];
    let key = expand_heads::<B>(B::float_cast(key, FloatDType::F32), group);
    let scores = matmul::<B>(query, B::float_transpose(key.clone()));
    let scores = B::float_mul_scalar(scores, scale.into());
    let scores = match hidden {
        Some(hidden) => {
            let hidden = B::bool_expand(hidden, scores.shape());
            B::float_mask_fill(scores, hidden, f32::NEG_INFINITY.into())
        }
        None => scores,
    };
    (scores, key)
}

/// `tensor` with `value` where it equals `target`.
fn replace<B: Backend>(tensor: FloatTensor<B>, target: f32, value: f32) -> FloatTensor<B> {
    let dtype = get_or_init_device_settings::<B>(&tensor.device()).bool_dtype;
    let at = B::float_equal_elem(tensor.clone(), target.into(), dtype);
    B::float_mask_fill(tensor, at, value.into())
}

/// The forward from a materialised mask: the output `[B, Tq, Hq, D]` in the dtype of `query`
/// and the log-sum-exp `[B, Tq, Hq]` in f32.
fn forward_hidden<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    hidden: Option<BoolTensor<B>>,
    scale: f32,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let dtype = query.dtype();
    let device = query.device();
    let [b, tq, hq, d] = dims4::<B>(&query);
    let [_, tk, hkv, _] = dims4::<B>(&key);
    if b * tq * hq * d == 0 || tk == 0 {
        // No key at all: zeros, and a log-sum-exp of -inf.
        return (
            B::float_zeros(Shape::new([b, tq, hq, d]), &device, FloatDType::from(dtype)),
            B::float_full(
                Shape::new([b, tq, hq]),
                f32::NEG_INFINITY.into(),
                &device,
                FloatDType::F32,
            ),
        );
    }
    let group = hq / hkv;
    let query = swap_time_and_heads::<B>(B::float_cast(query, FloatDType::F32));
    let key = swap_time_and_heads::<B>(key);
    let value = swap_time_and_heads::<B>(B::float_cast(value, FloatDType::F32));
    let value = expand_heads::<B>(value, group);
    let (scores, _) = masked_scores::<B>(query, key, hidden, scale);
    // A query that sees no key has a maximum of -inf: exponentiate against 0, so that its
    // probabilities are 0 rather than NaN, and divide by 1, so that its output is 0.
    let max = replace::<B>(B::float_max_dim(scores.clone(), 3), f32::NEG_INFINITY, 0.0);
    let exp = B::float_exp(B::float_sub(scores, max.clone()));
    let sum = B::float_sum_dim(exp.clone(), 3);
    let output = matmul::<B>(
        B::float_div(exp, replace::<B>(sum.clone(), 0.0, 1.0)),
        value,
    );
    // log(0) = -inf: the log-sum-exp of a query that sees no key.
    let lse = B::float_add(max, B::float_log(sum));
    let lse = B::float_reshape(lse, Shape::new([b, hq, tq]));
    (
        swap_time_and_heads::<B>(B::float_cast(output, FloatDType::from(dtype))),
        swap_time_and_heads::<B>(lse),
    )
}

/// The output `[B, Tq, Hq, D]`, in the dtype of `query`, and the natural log-sum-exp of each
/// query's scaled scores over the keys it sees, `[B, Tq, Hq]` in f32.
pub(crate) fn forward<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    mask: AttentionMaskPrimitive<B>,
    options: AttentionOptions,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let [b, tq, _, d] = dims4::<B>(&query);
    let tk = dims4::<B>(&key)[1];
    let hidden = hidden::<B>(mask, &options, [b, tq, tk], &query.device());
    forward_hidden::<B>(query, key, value, hidden, options.scale_for(d))
}

/// The gradients of `query`, `key` and `value`, each in its input's layout and dtype: with
/// `P = exp(S - lse)` (0 where a key is hidden, and for a query that sees none), `dV = P^T dO`,
/// `dS = P * (dO V^T - rowsum(dO * O))`, `dQ = scale dS K`, `dK = scale dS^T Q`, each key-value
/// gradient summed over the query heads of its group.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backward<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    output: FloatTensor<B>,
    lse: FloatTensor<B>,
    grad_output: FloatTensor<B>,
    mask: AttentionMaskPrimitive<B>,
    options: AttentionOptions,
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>) {
    let (query_dtype, key_dtype, value_dtype) = (query.dtype(), key.dtype(), value.dtype());
    let device = query.device();
    let [b, tq, hq, d] = dims4::<B>(&query);
    let [_, tk, hkv, _] = dims4::<B>(&key);
    if b * tq * hq * d == 0 || tk == 0 {
        let zeros = |shape: [usize; 4], dtype| {
            B::float_zeros(Shape::new(shape), &device, FloatDType::from(dtype))
        };
        return (
            zeros([b, tq, hq, d], query_dtype),
            zeros([b, tk, hkv, d], key_dtype),
            zeros([b, tk, hkv, d], value_dtype),
        );
    }
    let group = hq / hkv;
    let scale = options.scale_for(d);
    let hidden = hidden::<B>(mask, &options, [b, tq, tk], &device);
    let f32 = |tensor| B::float_cast(tensor, FloatDType::F32);
    let query = swap_time_and_heads::<B>(f32(query));
    let key = swap_time_and_heads::<B>(key);
    let value = expand_heads::<B>(swap_time_and_heads::<B>(f32(value)), group);
    let output = swap_time_and_heads::<B>(f32(output));
    let grad_output = swap_time_and_heads::<B>(f32(grad_output));
    // A query that sees no key has a log-sum-exp of -inf: +inf makes its probabilities 0.
    let lse = replace::<B>(f32(lse), f32::NEG_INFINITY, f32::INFINITY);
    let lse = B::float_reshape(B::float_swap_dims(lse, 1, 2), Shape::new([b, hq, tq, 1]));

    let (scores, key) = masked_scores::<B>(query.clone(), key, hidden, scale);
    let probs = B::float_exp(B::float_sub(scores, lse));
    let grad_value = matmul::<B>(B::float_transpose(probs.clone()), grad_output.clone());
    let grad_probs = matmul::<B>(grad_output.clone(), B::float_transpose(value));
    let delta = B::float_sum_dim(B::float_mul(grad_output, output), 3);
    let grad_scores = B::float_mul(probs, B::float_sub(grad_probs, delta));
    let grad_query = B::float_mul_scalar(matmul::<B>(grad_scores.clone(), key), scale.into());
    let grad_key = B::float_mul_scalar(
        matmul::<B>(B::float_transpose(grad_scores), query),
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

/// Decoding: `query` `[B, Hq, D]` against `key` and `value` `[B, S, Hkv, D]`, sequence `b`
/// holding the first `lengths[b]` keys (all `S` for [`CacheLengths::Full`]) with its query after
/// the last of them, and seeing `left` keys before that within the window; returns `[B, Hq, D]`
/// in the dtype of `query`.
pub(crate) fn decode<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    lengths: CacheLengths<B>,
    options: AttentionOptions,
) -> FloatTensor<B> {
    let device = query.device();
    let dtypes = Dtypes::of::<B>(&device);
    let [b, hq, d] = query.shape().dims::<3>();
    let s = dims4::<B>(&key)[1];
    let query = B::float_reshape(query, Shape::new([b, 1, hq, d]));
    let left = bound(options.reach().0, s + 1);
    let cols = positions::<B>(s, [1, s], &device, dtypes.int);
    // Hidden: keys at or past the sequence's length, and before its window.
    let hidden = match lengths {
        CacheLengths::Full => left.map(|left| {
            let first = s as i64 - 1 - left;
            let hidden = B::int_lower_elem(cols, first.into(), dtypes.boolean);
            B::bool_reshape(hidden, Shape::new([1, 1, 1, s]))
        }),
        CacheLengths::Lengths(lengths) => {
            let lengths = B::int_reshape(B::int_cast(lengths, dtypes.int), Shape::new([b, 1]));
            let (lengths, cols) = grid::<B>(lengths, cols);
            let mut hidden = B::int_greater_equal(cols.clone(), lengths.clone(), dtypes.boolean);
            if let Some(left) = left {
                let first = B::int_add_scalar(lengths, (-1 - left).into());
                hidden = B::bool_or(hidden, B::int_lower(cols, first, dtypes.boolean));
            }
            Some(B::bool_reshape(hidden, Shape::new([b, 1, 1, s])))
        }
    };
    let output = forward_hidden::<B>(query, key, value, hidden, options.scale_for(d)).0;
    B::float_reshape(output, Shape::new([b, hq, d]))
}
