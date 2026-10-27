//! What the operations' references share.

use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, Shape, TensorMetadata};

/// `a b` over the last two dimensions, for `a` `[..., n, k]` and `b` `[..., k, m]`, batched over
/// the leading ones. With `k = 1` the product is an outer product, formed as a broadcast multiply,
/// which gives the same values: Burn 0.22 launches a matmul there that asks CubeCL's CPU runtime
/// for more shared memory than it has.
pub(crate) fn matmul<B: Backend>(a: FloatTensor<B>, b: FloatTensor<B>) -> FloatTensor<B> {
    let (a_shape, b_shape) = (a.shape(), b.shape());
    let rank = a_shape.num_dims();
    if a_shape[rank - 1] != 1 {
        return B::float_matmul(a, b);
    }
    let mut dims: Vec<usize> = (0..rank).map(|i| a_shape[i].max(b_shape[i])).collect();
    dims[rank - 2] = a_shape[rank - 2];
    dims[rank - 1] = b_shape[rank - 1];
    let shape = Shape::from(dims);
    B::float_mul(B::float_expand(a, shape.clone()), B::float_expand(b, shape))
}
