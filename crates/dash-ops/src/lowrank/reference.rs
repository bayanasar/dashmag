//! The reference: the low-rank product and its gradients from Burn's tensor operations, in f32,
//! in the order PEFT's `lora_B(lora_A(x.float())) * scaling` computes them and Burn's autodiff of
//! those operations differentiates them: `h = x U`, then `(h V) · scale`, then the f32 sum with
//! the base rounded once; backward, `dz = dy · scale`, `dV = hᵀ dz`, `dh = dz Vᵀ`, `dU = xᵀ dh`,
//! `dx = dh Uᵀ`.

use super::{MaybeTensor, Wanted};
use crate::reference::matmul;
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, DType, FloatDType, TensorMetadata};

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

/// `(y, h)`: `h = x U` `[N, r]` and `y = base + scale · h V` `[N, out]`, summed in f32 and
/// rounded once to the base's dtype; without a base, `scale · h V` in f32.
pub(super) fn forward<B: Backend>(
    x: FloatTensor<B>,
    u: FloatTensor<B>,
    v: FloatTensor<B>,
    base: MaybeTensor<B>,
    scale: f32,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let h = matmul::<B>(to_f32::<B>(x), u);
    let z = B::float_mul_scalar(matmul::<B>(h.clone(), v), scale.into());
    let y = match base {
        MaybeTensor::None => z,
        MaybeTensor::Some(base) => {
            let dtype = base.dtype();
            to_dtype::<B>(B::float_add(to_f32::<B>(base), z), dtype)
        }
    };
    (y, h)
}

/// The gradients `wanted` of `x`, `U` and `V` from the output's, `grad`, and `h = x U`: `dx` in
/// `x`'s dtype, `dU` and `dV` in f32. The base's gradient is `grad` itself.
pub(super) fn backward<B: Backend>(
    x: FloatTensor<B>,
    u: FloatTensor<B>,
    v: FloatTensor<B>,
    h: FloatTensor<B>,
    grad: FloatTensor<B>,
    scale: f32,
    wanted: Wanted,
) -> (MaybeTensor<B>, MaybeTensor<B>, MaybeTensor<B>) {
    let x_dtype = x.dtype();
    let dz = B::float_mul_scalar(to_f32::<B>(grad), scale.into());
    let dv = wanted
        .v
        .then(|| matmul::<B>(B::float_transpose(h), dz.clone()));
    let dh = (wanted.x || wanted.u).then(|| matmul::<B>(dz, B::float_transpose(v)));
    let du = match (&dh, wanted.u) {
        (Some(dh), true) => Some(matmul::<B>(B::float_transpose(to_f32::<B>(x)), dh.clone())),
        _ => None,
    };
    let dx = match (dh, wanted.x) {
        (Some(dh), true) => Some(to_dtype::<B>(
            matmul::<B>(dh, B::float_transpose(u)),
            x_dtype,
        )),
        _ => None,
    };
    (dx.into(), du.into(), dv.into())
}
