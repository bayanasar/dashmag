//! The reference: the convolution, its gradients and the decoding step from Burn's tensor
//! operations, for any kernel width, in f32 whatever the inputs' dtype (as the kernels compute),
//! rounded to that dtype at the end. A sequence continues from its initial state by prepending
//! it; packed sequences run one at a time, their boundaries read to the host.

use super::{Activation, Bias, Sequences, State, probe};
use crate::sequences::read_cu_seqlens;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, DType, FloatDType, Shape, Slice, TensorMetadata};
use std::sync::atomic::Ordering;

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

/// Steps `range` of `x` `[B, T, C]`.
fn steps<B: Backend>(x: &FloatTensor<B>, range: core::ops::Range<usize>) -> FloatTensor<B> {
    let [b, _, c] = x.shape().dims();
    slice3::<B>(x.clone(), 0..b, range, 0..c)
}

/// Rows `range` of states `[S, W, C]`, or of any `[N, ..., ...]` tensor.
fn rows<B: Backend>(x: &FloatTensor<B>, range: core::ops::Range<usize>) -> FloatTensor<B> {
    let [_, w, c] = x.shape().dims();
    slice3::<B>(x.clone(), range, 0..w, 0..c)
}

fn zeros<B: Backend>(like: &FloatTensor<B>, [b, t, c]: [usize; 3]) -> FloatTensor<B> {
    B::float_zeros(Shape::new([b, t, c]), &like.device(), FloatDType::F32)
}

/// `pieces` concatenated along `dim`, the empty ones left out (`[B, 0, C]` when all are).
fn cat<B: Backend>(pieces: Vec<FloatTensor<B>>, dim: usize, empty: [usize; 3]) -> FloatTensor<B> {
    let first = pieces[0].clone();
    let mut pieces: Vec<_> = pieces
        .into_iter()
        .filter(|piece| piece.shape()[dim] > 0)
        .collect();
    match pieces.len() {
        0 => zeros::<B>(&first, empty),
        1 => pieces.pop().unwrap(),
        _ => B::float_cat(pieces, dim),
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

/// The pre-activation of `t` steps from their left-padded inputs `[B, t + K - 1, C]`, all f32.
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

/// The parameters as f32: the weight `[C, K]` and the bias `[C]`, if any.
struct Parameters<B: Backend> {
    weight: FloatTensor<B>,
    bias: Option<FloatTensor<B>>,
    k: usize,
}

impl<B: Backend> Parameters<B> {
    fn new(weight: FloatTensor<B>, bias: Option<FloatTensor<B>>) -> Self {
        let weight = to_f32::<B>(weight);
        let k = weight.shape().dims::<2>()[1];
        Self {
            weight,
            bias: bias.map(to_f32::<B>),
            k,
        }
    }
}

/// The `K - 1` inputs before the first step of each sequence of `x` `[B, L, C]` (f32): the last
/// `K - 1` steps of its initial state, or zeros; then its inputs: `[B, K - 1 + L, C]`.
fn padded<B: Backend>(
    x: &FloatTensor<B>,
    initial: Option<&FloatTensor<B>>,
    k: usize,
) -> FloatTensor<B> {
    let [b, l, c] = x.shape().dims();
    if k == 1 {
        return x.clone();
    }
    let left = match initial {
        Some(state) => {
            let w = state.shape()[1];
            steps::<B>(state, w + 1 - k..w)
        }
        None => zeros::<B>(x, [b, k - 1, c]),
    };
    cat::<B>(vec![left, x.clone()], 1, [b, k - 1 + l, c])
}

/// The history the final states take their steps from: the initial states (if any), then the
/// inputs, `[B, W + L, C]` (f32).
fn history<B: Backend>(x: &FloatTensor<B>, initial: Option<&FloatTensor<B>>) -> FloatTensor<B> {
    let [b, l, c] = x.shape().dims();
    match initial {
        None => x.clone(),
        Some(state) => {
            let w = state.shape()[1];
            cat::<B>(vec![state.clone(), x.clone()], 1, [b, w + l, c])
        }
    }
}

/// The last `len` steps of `history` `[B, P, C]`, zeros in front where it is shorter.
fn last_steps<B: Backend>(history: &FloatTensor<B>, len: usize) -> FloatTensor<B> {
    let [b, p, c] = history.shape().dims();
    if p >= len {
        return steps::<B>(history, p - len..p);
    }
    cat::<B>(
        vec![zeros::<B>(history, [b, len - p, c]), history.clone()],
        1,
        [b, len, c],
    )
}

/// One batch of sequences `x` `[B, L, C]` (f32), each from its initial state (or zeros): the
/// output, f32, and the final states of `final_state_len` steps.
fn forward_batch<B: Backend>(
    x: &FloatTensor<B>,
    parameters: &Parameters<B>,
    initial: Option<&FloatTensor<B>>,
    activation: Activation,
    final_state_len: Option<usize>,
) -> (FloatTensor<B>, Option<FloatTensor<B>>) {
    let l = x.shape()[1];
    let out = match l {
        0 => x.clone(),
        _ => {
            let padded = padded::<B>(x, initial, parameters.k);
            let y = convolve::<B>(&padded, &parameters.weight, parameters.bias.as_ref(), l);
            activate::<B>(y, activation)
        }
    };
    let final_state = final_state_len.map(|len| last_steps::<B>(&history::<B>(x, initial), len));
    (out, final_state)
}

/// The spans of the packed sequences, their boundaries read to the host.
fn spans<B: Backend>(
    cu_seqlens: IntTensor<B>,
    sequences: usize,
    tokens: usize,
) -> Vec<core::ops::Range<usize>> {
    probe::HOST_READS.fetch_add(1, Ordering::SeqCst);
    let cu = read_cu_seqlens::<B>(cu_seqlens, sequences, tokens);
    cu.windows(2).map(|w| w[0]..w[1]).collect()
}

/// The number of sequences of a call over `x` `[B, T, C]`.
fn sequence_count<B: Backend>(x: &FloatTensor<B>, sequences: &Sequences<B>) -> usize {
    match sequences {
        Sequences::Batch => x.shape()[0],
        Sequences::Packed(cu_seqlens) => cu_seqlens.shape()[0] - 1,
    }
}

pub(super) fn forward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Bias<B>,
    initial_state: State<B>,
    sequences: Sequences<B>,
    activation: Activation,
    final_state_len: Option<usize>,
) -> (FloatTensor<B>, State<B>) {
    let dtype = x.dtype();
    let [_, t, c] = x.shape().dims();
    let count = sequence_count::<B>(&x, &sequences);
    let parameters = Parameters::new(weight, bias.into_option());
    let initial = initial_state.into_option();
    let x = to_f32::<B>(x);
    let (out, final_state) = match sequences {
        Sequences::Batch => forward_batch::<B>(
            &x,
            &parameters,
            initial.as_ref(),
            activation,
            final_state_len,
        ),
        Sequences::Packed(cu_seqlens) => {
            let mut outputs = Vec::new();
            let mut finals = Vec::new();
            for (s, span) in spans::<B>(cu_seqlens, count, t).into_iter().enumerate() {
                let initial = initial.as_ref().map(|state| rows::<B>(state, s..s + 1));
                let (out, final_state) = forward_batch::<B>(
                    &steps::<B>(&x, span),
                    &parameters,
                    initial.as_ref(),
                    activation,
                    final_state_len,
                );
                outputs.push(out);
                finals.extend(final_state);
            }
            let out = cat::<B>(outputs, 1, [1, 0, c]);
            let len = final_state_len.unwrap_or(0);
            let final_state = final_state_len.map(|_| cat::<B>(finals, 0, [0, len, c]));
            (out, final_state)
        }
    };
    (to_dtype::<B>(out, dtype), State::from(final_state))
}

/// The gradients of one batch of sequences `x` `[B, L, C]` (f32), each from its initial state
/// (or zeros), given those of the output `grad` (f32) and of the final states: the input's and
/// the initial states' (f32), and the weight's and bias's sums over the batch (f32).
fn backward_batch<B: Backend>(
    x: &FloatTensor<B>,
    parameters: &Parameters<B>,
    initial: Option<&FloatTensor<B>>,
    grad: &FloatTensor<B>,
    grad_final: Option<&FloatTensor<B>>,
    activation: Activation,
) -> [Option<FloatTensor<B>>; 4] {
    let [b, l, c] = x.shape().dims();
    let k = parameters.k;
    let w = initial.map_or(0, |state| state.shape()[1]);
    // The gradient of the history `[initial states, x]`, `[B, W + L, C]`, by parts.
    let mut grad_x = None;
    let mut grad_weight = None;
    let mut grad_bias = None;
    let mut grad_history = None;
    if l > 0 {
        let padded = padded::<B>(x, initial, k);
        // The gradient of the pre-activation, which is recomputed for it.
        let grad_y = match activation {
            Activation::None => grad.clone(),
            activation => {
                let y = convolve::<B>(&padded, &parameters.weight, parameters.bias.as_ref(), l);
                activation_grad::<B>(y, grad.clone(), activation)
            }
        };
        // Padded step `p` feeds output steps `p - (K - 1) .. p` through taps `K - 1 .. 0`:
        // grad_padded[p] = sum_j weight[j] * grad_y[p - j], zero outside `0..L`.
        let grad_padded = match k {
            1 => B::float_mul(grad_y.clone(), tap::<B>(&parameters.weight, 0)),
            _ => {
                let gap = zeros::<B>(x, [b, k - 1, c]);
                let wide = B::float_cat(vec![gap.clone(), grad_y.clone(), gap], 1);
                (0..k)
                    .map(|j| {
                        let shift = k - 1 - j;
                        B::float_mul(
                            steps::<B>(&wide, shift..shift + l + k - 1),
                            tap::<B>(&parameters.weight, j),
                        )
                    })
                    .reduce(B::float_add)
                    .expect("a kernel width of at least 1")
            }
        };
        grad_x = Some(steps::<B>(&grad_padded, k - 1..k - 1 + l));
        // The steps before the first: the initial states' last K - 1.
        if initial.is_some() && k > 1 {
            let left = steps::<B>(&grad_padded, 0..k - 1);
            grad_history = Some(match w + 1 - k {
                0 => left,
                front => B::float_cat(vec![zeros::<B>(x, [b, front, c]), left], 1),
            });
        }
        // grad_weight[c, j] = sum over b and t of grad_y[b, t, c] * padded[b, t + j, c].
        let columns = (0..k)
            .map(|j| {
                let product = B::float_mul(grad_y.clone(), steps::<B>(&padded, j..j + l));
                B::float_reshape(B::float_sum_dims(product, &[0, 1]), Shape::new([c, 1]))
            })
            .collect();
        grad_weight = Some(B::float_cat(columns, 1));
        // grad_bias[c] = sum over b and t of grad_y[b, t, c].
        grad_bias = parameters
            .bias
            .as_ref()
            .map(|_| B::float_reshape(B::float_sum_dims(grad_y, &[0, 1]), Shape::new([c])));
    }
    // The final states are the last W' steps of the history, zeros in front.
    if let Some(grad_final) = grad_final {
        let len = grad_final.shape()[1];
        let p = w + l;
        let onto_history = match (p >= len, p.min(len)) {
            (_, 0) => None,
            (true, _) => Some(match p - len {
                0 => grad_final.clone(),
                front => B::float_cat(vec![zeros::<B>(x, [b, front, c]), grad_final.clone()], 1),
            }),
            (false, _) => Some(steps::<B>(grad_final, len - p..len)),
        };
        if let Some(onto) = onto_history {
            if l > 0 {
                let part = steps::<B>(&onto, w..w + l);
                grad_x = Some(match grad_x {
                    None => part,
                    Some(grad_x) => B::float_add(grad_x, part),
                });
            }
            if w > 0 {
                let part = steps::<B>(&onto, 0..w);
                grad_history = Some(match grad_history {
                    None => part,
                    Some(grad) => B::float_add(grad, part),
                });
            }
        }
    }
    let grad_initial = initial.map(|_| grad_history.unwrap_or_else(|| zeros::<B>(x, [b, w, c])));
    [grad_x, grad_weight, grad_bias, grad_initial]
}

#[allow(clippy::too_many_arguments)]
pub(super) fn backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Bias<B>,
    initial_state: State<B>,
    sequences: Sequences<B>,
    grad: FloatTensor<B>,
    grad_final_state: State<B>,
    activation: Activation,
) -> (FloatTensor<B>, FloatTensor<B>, Bias<B>, State<B>) {
    let (x_dtype, weight_dtype) = (x.dtype(), weight.dtype());
    let [_, t, c] = x.shape().dims();
    let count = sequence_count::<B>(&x, &sequences);
    let bias = bias.into_option();
    let bias_dtype = bias.as_ref().map(|bias| bias.dtype());
    let parameters = Parameters::new(weight, bias);
    let k = parameters.k;
    let initial = initial_state.into_option();
    let grad_final = grad_final_state.into_option();
    let x = to_f32::<B>(x);
    let grad = to_f32::<B>(grad);
    let [grad_x, grad_weight, grad_bias, grad_initial] = match sequences {
        Sequences::Batch => backward_batch::<B>(
            &x,
            &parameters,
            initial.as_ref(),
            &grad,
            grad_final.as_ref(),
            activation,
        ),
        Sequences::Packed(cu_seqlens) => {
            let mut grads_x = Vec::new();
            let mut grads_initial = Vec::new();
            let mut grad_weight = None;
            let mut grad_bias = None;
            let sum =
                |total: Option<FloatTensor<B>>, part: Option<FloatTensor<B>>| match (total, part) {
                    (Some(total), Some(part)) => Some(B::float_add(total, part)),
                    (total, part) => total.or(part),
                };
            for (s, span) in spans::<B>(cu_seqlens, count, t).into_iter().enumerate() {
                let one = |states: &Option<FloatTensor<B>>| {
                    states.as_ref().map(|states| rows::<B>(states, s..s + 1))
                };
                let (initial, grad_final) = (one(&initial), one(&grad_final));
                let len = span.len();
                let [gx, gw, gb, gi] = backward_batch::<B>(
                    &steps::<B>(&x, span.clone()),
                    &parameters,
                    initial.as_ref(),
                    &steps::<B>(&grad, span),
                    grad_final.as_ref(),
                    activation,
                );
                grads_x.push(gx.unwrap_or_else(|| zeros::<B>(&x, [1, len, c])));
                grads_initial.extend(gi);
                grad_weight = sum(grad_weight, gw);
                grad_bias = sum(grad_bias, gb);
            }
            let w = initial.as_ref().map_or(0, |state| state.shape()[1]);
            let grad_initial = initial
                .as_ref()
                .map(|_| cat::<B>(grads_initial, 0, [0, w, c]));
            [
                Some(cat::<B>(grads_x, 1, [1, 0, c])),
                grad_weight,
                grad_bias,
                grad_initial,
            ]
        }
    };
    // Where no step contributed (no steps at all), the gradients are zero.
    let device = x.device();
    let grad_x = grad_x.unwrap_or_else(|| x.clone());
    let grad_weight =
        grad_weight.unwrap_or_else(|| B::float_zeros(Shape::new([c, k]), &device, FloatDType::F32));
    let grad_bias = bias_dtype.map(|dtype| {
        let grad =
            grad_bias.unwrap_or_else(|| B::float_zeros(Shape::new([c]), &device, FloatDType::F32));
        to_dtype::<B>(grad, dtype)
    });
    (
        to_dtype::<B>(grad_x, x_dtype),
        to_dtype::<B>(grad_weight, weight_dtype),
        Bias::from(grad_bias),
        State::from(grad_initial),
    )
}

pub(super) fn update<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Bias<B>,
    state: FloatTensor<B>,
    activation: Activation,
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
    (to_dtype::<B>(activate::<B>(y, activation), dtype), next)
}
