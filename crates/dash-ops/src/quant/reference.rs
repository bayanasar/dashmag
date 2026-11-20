//! The reference: `W` unpacked and dequantized to f32 in Burn's integer and float operations,
//! then `y = x W (+ bias)` in f32, rounded once to `x`'s dtype.

use crate::lowrank::MaybeTensor;
use crate::reference::matmul;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, DType, FloatDType, Shape, TensorMetadata};
use dash_contract::quant::QuantOptions;

/// `W` `[in, out]` in f32 from `qweight` `[in / 8, out]` and `scales` `[in / group, out]`.
pub(crate) fn dequantize<B: Backend>(
    qweight: IntTensor<B>,
    scales: FloatTensor<B>,
    group: usize,
) -> FloatTensor<B> {
    let shape = qweight.shape();
    let (packed, out) = (shape[0], shape[1]);
    let inp = packed * 8;
    // Nibble j of word w is input row 8 w + j: the eight laid side by side along a new axis
    // after the words', and the reshape puts the rows in order.
    let nibbles: Vec<IntTensor<B>> = (0..8)
        .map(|j| {
            let shifted = B::bitwise_right_shift_scalar(qweight.clone(), (4 * j).into());
            let nibble = B::bitwise_and_scalar(shifted, 15.into());
            B::int_reshape(nibble, Shape::new([packed, 1, out]))
        })
        .collect();
    let q = B::int_reshape(
        B::int_cat(nibbles, 1),
        Shape::new([inp / group, group, out]),
    );
    let centred = B::float_sub_scalar(
        B::int_into_float(q, FloatDType::F32),
        (QuantOptions::SYMMETRIC_ZERO as f32).into(),
    );
    let scales = B::float_reshape(
        B::float_cast(scales, FloatDType::F32),
        Shape::new([inp / group, 1, out]),
    );
    let scales = B::float_expand(scales, Shape::new([inp / group, group, out]));
    B::float_reshape(B::float_mul(centred, scales), Shape::new([inp, out]))
}

/// `dx = dy Wᵀ` for `dy` `[N, out]`: in f32, rounded once to `dy`'s dtype.
pub(super) fn backward<B: Backend>(
    dy: FloatTensor<B>,
    qweight: IntTensor<B>,
    scales: FloatTensor<B>,
    group: usize,
) -> FloatTensor<B> {
    let dtype = dy.dtype();
    let dy = match dtype {
        DType::F32 => dy,
        _ => B::float_cast(dy, FloatDType::F32),
    };
    let w = dequantize::<B>(qweight, scales, group);
    let dx = matmul::<B>(dy, B::float_transpose(w));
    match dtype {
        DType::F32 => dx,
        _ => B::float_cast(dx, dtype.into()),
    }
}

/// `y = x W (+ bias)` for `x` `[N, in]`: in f32, rounded once to `x`'s dtype.
pub(super) fn forward<B: Backend>(
    x: FloatTensor<B>,
    qweight: IntTensor<B>,
    scales: FloatTensor<B>,
    bias: MaybeTensor<B>,
    group: usize,
) -> FloatTensor<B> {
    let dtype = x.dtype();
    let x = match dtype {
        DType::F32 => x,
        _ => B::float_cast(x, FloatDType::F32),
    };
    let mut y = matmul::<B>(x, dequantize::<B>(qweight, scales, group));
    if let MaybeTensor::Some(bias) = bias {
        let shape = y.shape();
        let bias = B::float_reshape(
            B::float_cast(bias, FloatDType::F32),
            Shape::new([1, shape[1]]),
        );
        y = B::float_add(y, B::float_expand(bias, shape));
    }
    match dtype {
        DType::F32 => y,
        _ => B::float_cast(y, dtype.into()),
    }
}
