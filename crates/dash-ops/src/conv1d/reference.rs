//! The reference: the convolution, its gradients and the decoding step from Burn's tensor
//! operations, for any kernel width, in f32 whatever the inputs' dtype (as the kernels compute),
//! rounded to that dtype at the end.

use super::{Activation, Bias, CausalConv1dOptions};
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

/// `y + bias`, the bias `[C]` (f32) broadcast over the leading dimensions of `y` (`[..., C]`).
fn add_bias<B: Backend>(y: FloatTensor<B>, bias: Option<&FloatTensor<B>>) -> FloatTensor<B> {
    match bias {
        None => y,
        Some(bias) => {
            let rank = y.shape().num_dims();
            let mut dims = vec![1; rank];
            dims[rank - 1] = bias.shape().dims::<1>()[0];
            B::float_add(y, B::float_reshape(bias.clone(), Shape::from(dims)))
        }
    }
}

/// The pre-activation from the left-padded input `[B, T + K - 1, C]`, all f32.
fn convolve<B: Backend>(
    padded: &FloatTensor<B>,
    weight: &FloatTensor<B>,
    bias: Option<&FloatTensor<B>>,
    t: usize,
) -> FloatTensor<B> {
    let [b, _, c] = padded.shape().dims();
    let [_, k] = weight.shape().dims();
    let y = (0..k)
        .map(|j| {
            B::float_mul(
                slice3::<B>(padded.clone(), 0..b, j..j + t, 0..c),
                tap::<B>(weight, j),
            )
        })
        .reduce(B::float_add)
        .expect("a kernel width of at least 1");
    add_bias::<B>(y, bias)
}

fn activate<B: Backend>(y: FloatTensor<B>, activation: Activation) -> FloatTensor<B> {
    match activation {
        Activation::None => y,
        Activation::Silu => B::float_mul(y.clone(), B::sigmoid(y)),
    }
}

/// `grad * act'(y)`.
fn activation_grad<B: Backend>(
    y: FloatTensor<B>,
    grad: FloatTensor<B>,
    activation: Activation,
) -> FloatTensor<B> {
    match activation {
        Activation::None => grad,
        Activation::Silu => {
            // SiLU'(y) = s (1 + y (1 - s)), s = sigmoid(y).
            let s = B::sigmoid(y.clone());
            let one_minus_s = B::float_add_scalar(B::float_neg(s.clone()), 1.0f32.into());
            let slope = B::float_add_scalar(B::float_mul(y, one_minus_s), 1.0f32.into());
            B::float_mul(grad, B::float_mul(s, slope))
        }
    }
}

pub(super) fn forward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Bias<B>,
    options: CausalConv1dOptions,
) -> FloatTensor<B> {
    let dtype = x.dtype();
    let [_, t, _] = x.shape().dims();
    let weight = to_f32::<B>(weight);
    let k = weight.shape().dims::<2>()[1];
    let bias = bias.into_option().map(to_f32::<B>);
    let padded = pad_steps::<B>(to_f32::<B>(x), k, false);
    let y = convolve::<B>(&padded, &weight, bias.as_ref(), t);
    to_dtype::<B>(activate::<B>(y, options.activation), dtype)
}

pub(super) fn backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Bias<B>,
    grad: FloatTensor<B>,
    options: CausalConv1dOptions,
) -> (FloatTensor<B>, FloatTensor<B>, Bias<B>) {
    let (x_dtype, weight_dtype) = (x.dtype(), weight.dtype());
    let [b, t, c] = x.shape().dims();
    let weight = to_f32::<B>(weight);
    let k = weight.shape().dims::<2>()[1];
    let bias = bias
        .into_option()
        .map(|bias| (bias.dtype(), to_f32::<B>(bias)));
    let padded = pad_steps::<B>(to_f32::<B>(x), k, false);
    let grad = to_f32::<B>(grad);

    // The gradient of the pre-activation, which is recomputed for it.
    let grad_y = match options.activation {
        Activation::None => grad,
        activation => {
            let y = convolve::<B>(&padded, &weight, bias.as_ref().map(|(_, b)| b), t);
            activation_grad::<B>(y, grad, activation)
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

    // grad_bias[c] = sum over b and t of grad_y[b, t, c].
    let grad_bias = bias.map(|(dtype, _)| {
        let sum = B::float_reshape(B::float_sum_dims(grad_y, &[0, 1]), Shape::new([c]));
        to_dtype::<B>(sum, dtype)
    });

    (
        to_dtype::<B>(grad_x, x_dtype),
        to_dtype::<B>(grad_weight, weight_dtype),
        Bias::from(grad_bias),
    )
}

pub(super) fn update<B: Backend>(
    x: FloatTensor<B>,
    state: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Bias<B>,
    options: CausalConv1dOptions,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let dtype = x.dtype();
    let [b, c] = x.shape().dims();
    let s = state.shape().dims::<3>()[1];
    let k = weight.shape().dims::<2>()[1];
    let token = B::float_reshape(to_f32::<B>(x), Shape::new([b, 1, c]));
    let state = to_f32::<B>(state);
    // The state's last `K - 1` steps and the token, `[B, K, C]`.
    let window = match k {
        1 => token.clone(),
        _ => B::float_cat(
            vec![
                slice3::<B>(state.clone(), 0..b, s + 1 - k..s, 0..c),
                token.clone(),
            ],
            1,
        ),
    };
    // The next state, a new tensor rather than a view of a longer one.
    let next = match s {
        0 => state,
        1 => token,
        _ => B::float_cat(vec![slice3::<B>(state, 0..b, 1..s, 0..c), token], 1),
    };
    let taps = B::float_swap_dims(to_f32::<B>(weight), 0, 1);
    let taps = B::float_reshape(taps, Shape::new([1, k, c]));
    let y = B::float_reshape(
        B::float_sum_dim(B::float_mul(window, taps), 1),
        Shape::new([b, c]),
    );
    let bias = bias.into_option().map(to_f32::<B>);
    let y = add_bias::<B>(y, bias.as_ref());
    (
        to_dtype::<B>(activate::<B>(y, options.activation), dtype),
        next,
    )
}
