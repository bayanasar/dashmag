//! The reference: the convolution, its gradients and the decoding step from Burn's tensor
//! operations, for any kernel width, in f32 whatever the inputs' dtype (as the kernels compute),
//! rounded to that dtype at the end.

use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, DType, FloatDType, Shape, Slice, TensorMetadata};

fn to_f32<B: Backend>(tensor: FloatTensor<B>) -> FloatTensor<B> {
    match tensor.dtype() {
        DType::F32 => tensor,
        _ => B::float_cast(tensor, FloatDType::F32),
    }
}

fn to_dtype<B: Backend>(tensor: FloatTensor<B>, dtype: DType) -> FloatTensor<B> {
    match dtype {
        DType::F32 => tensor,
        _ => B::float_cast(tensor, dtype.into()),
    }
}

fn slice3<B: Backend>(
    tensor: FloatTensor<B>,
    a: core::ops::Range<usize>,
    b: core::ops::Range<usize>,
    c: core::ops::Range<usize>,
) -> FloatTensor<B> {
    B::float_slice(tensor, &[Slice::from(a), Slice::from(b), Slice::from(c)])
}

/// `x` `[B, T, C]` with `k - 1` zero steps in front (`at_end`: behind): `[B, T + k - 1, C]`.
fn pad_steps<B: Backend>(x: FloatTensor<B>, k: usize, at_end: bool) -> FloatTensor<B> {
    if k == 1 {
        return x;
    }
    let [b, _, c] = x.shape().dims();
    let zeros = B::float_zeros(Shape::new([b, k - 1, c]), &x.device(), FloatDType::F32);
    match at_end {
        false => B::float_cat(vec![zeros, x], 1),
        true => B::float_cat(vec![x, zeros], 1),
    }
}

/// Tap `j` of `weight` `[C, K]` as `[1, 1, C]`.
fn tap<B: Backend>(weight: &FloatTensor<B>, j: usize) -> FloatTensor<B> {
    let [c, _] = weight.shape().dims();
    let column = B::float_slice(weight.clone(), &[Slice::from(0..c), Slice::from(j..j + 1)]);
    B::float_reshape(column, Shape::new([1, 1, c]))
}

/// The pre-activation from the left-padded input `[B, T + K - 1, C]`, all f32.
fn convolve<B: Backend>(
    padded: &FloatTensor<B>,
    weight: &FloatTensor<B>,
    t: usize,
) -> FloatTensor<B> {
    let [b, _, c] = padded.shape().dims();
    let [_, k] = weight.shape().dims();
    (0..k)
        .map(|j| {
            B::float_mul(
                slice3::<B>(padded.clone(), 0..b, j..j + t, 0..c),
                tap::<B>(weight, j),
            )
        })
        .reduce(B::float_add)
        .expect("a kernel width of at least 1")
}

fn silu<B: Backend>(y: FloatTensor<B>) -> FloatTensor<B> {
    B::float_mul(y.clone(), B::sigmoid(y))
}

pub(super) fn forward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    silu_after: bool,
) -> FloatTensor<B> {
    let dtype = x.dtype();
    let [_, t, _] = x.shape().dims();
    let weight = to_f32::<B>(weight);
    let k = weight.shape().dims::<2>()[1];
    let y = convolve::<B>(&pad_steps::<B>(to_f32::<B>(x), k, false), &weight, t);
    to_dtype::<B>(if silu_after { silu::<B>(y) } else { y }, dtype)
}

pub(super) fn backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    grad: FloatTensor<B>,
    silu_after: bool,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let (x_dtype, weight_dtype) = (x.dtype(), weight.dtype());
    let [b, t, c] = x.shape().dims();
    let weight = to_f32::<B>(weight);
    let k = weight.shape().dims::<2>()[1];
    let padded = pad_steps::<B>(to_f32::<B>(x), k, false);
    let grad = to_f32::<B>(grad);

    // The gradient of the pre-activation: SiLU'(y) = s (1 + y (1 - s)), s = sigmoid(y).
    let grad_y = match silu_after {
        false => grad,
        true => {
            let y = convolve::<B>(&padded, &weight, t);
            let s = B::sigmoid(y.clone());
            let one_minus_s = B::float_add_scalar(B::float_neg(s.clone()), 1.0f32.into());
            let slope = B::float_add_scalar(B::float_mul(y, one_minus_s), 1.0f32.into());
            B::float_mul(grad, B::float_mul(s, slope))
        }
    };

    // Input step `t` feeds output steps `t .. t + K - 1` through taps `K - 1 .. 0`:
    // grad_x[t] = sum_j weight[j] * grad_y[t + K - 1 - j].
    let grad_y_padded = pad_steps::<B>(grad_y.clone(), k, true);
    let grad_x = (0..k)
        .map(|j| {
            let shift = k - 1 - j;
            B::float_mul(
                slice3::<B>(grad_y_padded.clone(), 0..b, shift..shift + t, 0..c),
                tap::<B>(&weight, j),
            )
        })
        .reduce(B::float_add)
        .expect("a kernel width of at least 1");

    // grad_weight[c, j] = sum over b and t of grad_y[b, t, c] * x[b, t - (K - 1) + j, c].
    let columns = (0..k)
        .map(|j| {
            let product = B::float_mul(
                grad_y.clone(),
                slice3::<B>(padded.clone(), 0..b, j..j + t, 0..c),
            );
            B::float_reshape(B::float_sum_dims(product, &[0, 1]), Shape::new([c, 1]))
        })
        .collect();
    let grad_weight = B::float_cat(columns, 1);

    (
        to_dtype::<B>(grad_x, x_dtype),
        to_dtype::<B>(grad_weight, weight_dtype),
    )
}

pub(super) fn update<B: Backend>(
    x: FloatTensor<B>,
    state: FloatTensor<B>,
    weight: FloatTensor<B>,
    silu_after: bool,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let dtype = x.dtype();
    let [b, c] = x.shape().dims();
    let s = state.shape().dims::<3>()[2];
    let k = weight.shape().dims::<2>()[1];
    let window = B::float_cat(vec![state, B::float_reshape(x, Shape::new([b, c, 1]))], 2);
    let next = slice3::<B>(window.clone(), 0..b, 0..c, 1..s + 1);
    let last = to_f32::<B>(slice3::<B>(window, 0..b, 0..c, s + 1 - k..s + 1));
    let weight = B::float_reshape(to_f32::<B>(weight), Shape::new([1, c, k]));
    let y = B::float_reshape(
        B::float_sum_dim(B::float_mul(last, weight), 2),
        Shape::new([b, c]),
    );
    let out = if silu_after { silu::<B>(y) } else { y };
    (to_dtype::<B>(out, dtype), next)
}
