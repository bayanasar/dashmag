//! MLX's operations this crate calls, one call of mlx-c each.

use super::array::{Array, ArrayElement, Dtype, op};
use super::error::Result;
use super::sys;
use super::transforms::VectorArray;

pub mod indexing;

/// An optional flag, or its default.
fn flag(value: impl Into<Option<bool>>, default: bool) -> bool {
    value.into().unwrap_or(default)
}

macro_rules! unary {
    ($($(#[$doc:meta])* $name:ident = $c:ident;)*) => {
        impl Array {$(
            $(#[$doc])*
            pub fn $name(&self) -> Result<Array> {
                // SAFETY: valid handles, for the length of the call.
                op(|res, s| unsafe { sys::$c(res, self.as_ptr(), s) })
            }
        )*}
    };
}

macro_rules! binary {
    ($($(#[$doc:meta])* $name:ident = $c:ident;)*) => {
        impl Array {$(
            $(#[$doc])*
            pub fn $name(&self, other: impl AsRef<Array>) -> Result<Array> {
                // SAFETY: valid handles, for the length of the call.
                op(|res, s| unsafe { sys::$c(res, self.as_ptr(), other.as_ref().as_ptr(), s) })
            }
        )*}
    };
}

macro_rules! reduce {
    ($($(#[$doc:meta])* $name:ident = $c:ident;)*) => {
        impl Array {$(
            $(#[$doc])*
            pub fn $name(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
                let keep_dims = flag(keep_dims, false);
                // SAFETY: valid handles, for the length of the call.
                op(|res, s| unsafe { sys::$c(res, self.as_ptr(), keep_dims, s) })
            }
        )*}
    };
}

macro_rules! reduce_axis {
    ($($(#[$doc:meta])* $name:ident = $c:ident;)*) => {
        impl Array {$(
            $(#[$doc])*
            pub fn $name(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
                let keep_dims = flag(keep_dims, false);
                // SAFETY: valid handles, for the length of the call.
                op(|res, s| unsafe { sys::$c(res, self.as_ptr(), axis, keep_dims, s) })
            }
        )*}
    };
}

unary! {
    /// `|x|`.
    abs = mlx_abs;
    /// `cos(x)`.
    cos = mlx_cos;
    /// `exp(x)`.
    exp = mlx_exp;
    /// The largest integer no greater than `x`.
    floor = mlx_floor;
    /// `ln(x)`.
    log = mlx_log;
    /// `!x`.
    logical_not = mlx_logical_not;
    /// `-x`.
    negative = mlx_negative;
    /// `1 / sqrt(x)`.
    rsqrt = mlx_rsqrt;
    /// `sin(x)`.
    sin = mlx_sin;
    /// `sqrt(x)`.
    sqrt = mlx_sqrt;
    /// `x * x`.
    square = mlx_square;
    /// Every axis of size 1 removed.
    squeeze = mlx_squeeze;
    /// The axes reversed.
    transpose = mlx_transpose;
}

binary! {
    /// `x + y`.
    add = mlx_add;
    /// `x / y`.
    divide = mlx_divide;
    /// `x == y`.
    eq = mlx_equal;
    /// `x >= y`.
    ge = mlx_greater_equal;
    /// `x > y`.
    gt = mlx_greater;
    /// `x <= y`.
    le = mlx_less_equal;
    /// `x && y`.
    logical_and = mlx_logical_and;
    /// `x || y`.
    logical_or = mlx_logical_or;
    /// `x < y`.
    lt = mlx_less;
    /// The matrix product.
    matmul = mlx_matmul;
    /// `x * y`.
    multiply = mlx_multiply;
    /// `x != y`.
    ne = mlx_not_equal;
    /// `x - y`.
    subtract = mlx_subtract;
}

reduce! {
    /// The largest element.
    max = mlx_max;
    /// The mean of the elements.
    mean = mlx_mean;
    /// The smallest element.
    min = mlx_min;
    /// The sum of the elements.
    sum = mlx_sum;
}

reduce_axis! {
    /// Whether any element along an axis is true.
    any_axis = mlx_any_axis;
    /// `ln(sum(exp(x)))` along an axis.
    logsumexp_axis = mlx_logsumexp_axis;
    /// The largest element along an axis.
    max_axis = mlx_max_axis;
    /// The mean along an axis.
    mean_axis = mlx_mean_axis;
    /// The smallest element along an axis.
    min_axis = mlx_min_axis;
    /// The sum along an axis.
    sum_axis = mlx_sum_axis;
}

impl Array {
    /// The same values as `dtype`.
    pub fn as_dtype(&self, dtype: Dtype) -> Result<Array> {
        // SAFETY: valid handles, for the length of the call.
        op(|res, s| unsafe { sys::mlx_astype(res, self.as_ptr(), dtype as sys::mlx_dtype, s) })
    }

    /// A row-contiguous array of the same values, after evaluation.
    pub fn contiguous(&self) -> Result<Array> {
        // SAFETY: valid handles, for the length of the call.
        op(|res, s| unsafe { sys::mlx_contiguous(res, self.as_ptr(), false, s) })
    }

    /// The `offset`th diagonal of the matrices in `axis1` and `axis2`, as a new last axis.
    pub fn diagonal(&self, offset: i32, axis1: i32, axis2: i32) -> Result<Array> {
        // SAFETY: valid handles, for the length of the call.
        op(|res, s| unsafe { sys::mlx_diagonal(res, self.as_ptr(), offset, axis1, axis2, s) })
    }

    /// A new axis of size 1 at `axis`.
    pub fn expand_dims(&self, axis: i32) -> Result<Array> {
        expand_dims(self, axis)
    }

    /// The same values in this shape.
    pub fn reshape(&self, shape: &[i32]) -> Result<Array> {
        // SAFETY: valid handles and `shape`'s elements, for the length of the call.
        op(
            |res, s| unsafe {
                sys::mlx_reshape(res, self.as_ptr(), shape.as_ptr(), shape.len(), s)
            },
        )
    }

    /// These axes of size 1 removed.
    pub fn squeeze_axes(&self, axes: &[i32]) -> Result<Array> {
        // SAFETY: valid handles and `axes`' elements, for the length of the call.
        op(|res, s| unsafe {
            sys::mlx_squeeze_axes(res, self.as_ptr(), axes.as_ptr(), axes.len(), s)
        })
    }

    /// The sum over these axes.
    pub fn sum_axes(&self, axes: &[i32], keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let keep_dims = flag(keep_dims, false);
        // SAFETY: valid handles and `axes`' elements, for the length of the call.
        op(|res, s| unsafe {
            sys::mlx_sum_axes(res, self.as_ptr(), axes.as_ptr(), axes.len(), keep_dims, s)
        })
    }

    /// Two axes exchanged.
    pub fn swap_axes(&self, axis1: i32, axis2: i32) -> Result<Array> {
        swap_axes(self, axis1, axis2)
    }

    /// The axes reversed. Panics if MLX fails, which it does not on a valid array.
    pub fn t(&self) -> Array {
        self.transpose().unwrap()
    }

    /// The elements at `indices` along `axis`.
    pub fn take_axis(&self, indices: impl AsRef<Array>, axis: i32) -> Result<Array> {
        // SAFETY: valid handles, for the length of the call.
        op(|res, s| unsafe {
            sys::mlx_take_axis(res, self.as_ptr(), indices.as_ref().as_ptr(), axis, s)
        })
    }

    /// The elements at `indices`, which match this array's shape but along `axis`.
    pub fn take_along_axis(&self, indices: impl AsRef<Array>, axis: i32) -> Result<Array> {
        // SAFETY: valid handles, for the length of the call.
        op(|res, s| unsafe {
            sys::mlx_take_along_axis(res, self.as_ptr(), indices.as_ref().as_ptr(), axis, s)
        })
    }

    /// This array with `values` written at `indices` along `axis`.
    pub fn put_along_axis(
        &self,
        indices: impl AsRef<Array>,
        values: impl AsRef<Array>,
        axis: i32,
    ) -> Result<Array> {
        // SAFETY: valid handles, for the length of the call.
        op(|res, s| unsafe {
            sys::mlx_put_along_axis(
                res,
                self.as_ptr(),
                indices.as_ref().as_ptr(),
                values.as_ref().as_ptr(),
                axis,
                s,
            )
        })
    }

    /// The axes in this order.
    pub fn transpose_axes(&self, axes: &[i32]) -> Result<Array> {
        // SAFETY: valid handles and `axes`' elements, for the length of the call.
        op(|res, s| unsafe {
            sys::mlx_transpose_axes(res, self.as_ptr(), axes.as_ptr(), axes.len(), s)
        })
    }

    /// The same bytes read as `dtype`. Where `dtype`'s size differs from this array's, the last
    /// axis takes the same bytes in elements of the new size.
    pub fn view(&self, dtype: Dtype) -> Result<Array> {
        // SAFETY: valid handles, for the length of the call.
        op(|res, s| unsafe { sys::mlx_view(res, self.as_ptr(), dtype as sys::mlx_dtype, s) })
    }
}

/// `x` with gradients stopped: the same values, a constant to differentiation.
pub fn stop_gradient(x: impl AsRef<Array>) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_stop_gradient(res, x.as_ref().as_ptr(), s) })
}

/// `c * beta + a @ b * alpha`.
pub fn addmm(
    c: impl AsRef<Array>,
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    alpha: f32,
    beta: f32,
) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_addmm(
            res,
            c.as_ref().as_ptr(),
            a.as_ref().as_ptr(),
            b.as_ref().as_ptr(),
            alpha,
            beta,
            s,
        )
    })
}

/// `start, start + step, ..` below `stop`, as `T`: from 0 and by 1 where those are `None`.
pub fn arange<U: Into<f64>, T: ArrayElement>(
    start: impl Into<Option<U>>,
    stop: U,
    step: impl Into<Option<U>>,
) -> Result<Array> {
    let start = start.into().map_or(0.0, Into::into);
    let stop = stop.into();
    let step = step.into().map_or(1.0, Into::into);
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_arange(res, start, stop, step, T::DTYPE as sys::mlx_dtype, s) })
}

/// The sort's indices along an axis.
pub fn argsort_axis(a: impl AsRef<Array>, axis: i32) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_argsort_axis(res, a.as_ref().as_ptr(), axis, s) })
}

/// The indices of a partition about the `kth` smallest along an axis.
pub fn argpartition_axis(a: impl AsRef<Array>, kth: i32, axis: i32) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_argpartition_axis(res, a.as_ref().as_ptr(), kth, axis, s) })
}

/// `a` broadcast to `shape`.
pub fn broadcast_to(a: impl AsRef<Array>, shape: &[i32]) -> Result<Array> {
    // SAFETY: valid handles and `shape`'s elements, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_broadcast_to(res, a.as_ref().as_ptr(), shape.as_ptr(), shape.len(), s)
    })
}

/// The arrays joined along an axis.
pub fn concatenate(arrays: &[impl AsRef<Array>], axis: i32) -> Result<Array> {
    let arrays = VectorArray::from_arrays(arrays.iter())?;
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_concatenate_axis(res, arrays.as_ptr(), axis, s) })
}

/// A 1D convolution of `a` `[.., length, channels]` with `weight` `[width, channels / groups, out]`.
pub fn conv1d(
    a: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_conv1d(
            res,
            a.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride,
            padding,
            dilation,
            groups,
            s,
        )
    })
}

/// The cumulative sum along an axis, inclusive and forward by default.
pub fn cumsum(
    a: impl AsRef<Array>,
    axis: i32,
    reverse: impl Into<Option<bool>>,
    inclusive: impl Into<Option<bool>>,
) -> Result<Array> {
    let (reverse, inclusive) = (flag(reverse, false), flag(inclusive, true));
    let no_dtype = sys::mlx_optional_dtype {
        value: sys::mlx_dtype__MLX_FLOAT32,
        has_value: false,
    };
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_cumsum_axis(
            res,
            a.as_ref().as_ptr(),
            axis,
            reverse,
            inclusive,
            no_dtype,
            s,
        )
    })
}

/// The same values under a new handle and a node of the graph of its own, sharing the data.
pub fn copy(a: impl AsRef<Array>) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_copy(res, a.as_ref().as_ptr(), s) })
}

/// A new axis of size 1 at `axis`.
pub fn expand_dims(a: impl AsRef<Array>, axis: i32) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_expand_dims(res, a.as_ref().as_ptr(), axis, s) })
}

/// `values` broadcast to `shape`, as `T`.
pub fn full<T: ArrayElement>(shape: &[i32], values: impl AsRef<Array>) -> Result<Array> {
    // SAFETY: valid handles and `shape`'s elements, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_full(
            res,
            shape.as_ptr(),
            shape.len(),
            values.as_ref().as_ptr(),
            T::DTYPE as sys::mlx_dtype,
            s,
        )
    })
}

/// `ln(exp(a) + exp(b))`.
pub fn logaddexp(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_logaddexp(res, a.as_ref().as_ptr(), b.as_ref().as_ptr(), s) })
}

/// The larger of `a` and `b`, element by element.
pub fn maximum(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_maximum(res, a.as_ref().as_ptr(), b.as_ref().as_ptr(), s) })
}

/// The smaller of `a` and `b`, element by element.
pub fn minimum(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_minimum(res, a.as_ref().as_ptr(), b.as_ref().as_ptr(), s) })
}

/// Axis `src` moved to `dst`.
pub fn move_axis(a: impl AsRef<Array>, src: i32, dst: i32) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_moveaxis(res, a.as_ref().as_ptr(), src, dst, s) })
}

/// An array of ones, as `T`.
pub fn ones<T: ArrayElement>(shape: &[i32]) -> Result<Array> {
    // SAFETY: valid handles and `shape`'s elements, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_ones(
            res,
            shape.as_ptr(),
            shape.len(),
            T::DTYPE as sys::mlx_dtype,
            s,
        )
    })
}

/// `a` padded with zeros of its dtype, by `(low, high)` on each of its axes.
pub fn pad(a: impl AsRef<Array>, widths: &[(i32, i32)]) -> Result<Array> {
    let a = a.as_ref();
    let axes: Vec<i32> = (0..a.ndim() as i32).collect();
    let low: Vec<i32> = widths.iter().map(|w| w.0).collect();
    let high: Vec<i32> = widths.iter().map(|w| w.1).collect();
    let value = Array::from_int(0).as_dtype(a.dtype())?;
    // SAFETY: valid handles and the slices' elements, for the length of the call; a C string.
    op(|res, s| unsafe {
        sys::mlx_pad(
            res,
            a.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            low.as_ptr(),
            low.len(),
            high.as_ptr(),
            high.len(),
            value.as_ptr(),
            c"constant".as_ptr(),
            s,
        )
    })
}

/// `x` times `w` in MLX's affine quantization (`scales · q + biases` for each group of
/// `group_size` values along `w`'s last axis, `bits` bits each), `w` transposed where `transpose`.
/// It computes in the dtype that `x`'s and the scales' promote to.
pub fn quantized_matmul(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl AsRef<Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
) -> Result<Array> {
    let int = |value| sys::mlx_optional_int {
        value,
        has_value: true,
    };
    // SAFETY: valid handles, for the length of the call; a C string.
    op(|res, s| unsafe {
        sys::mlx_quantized_matmul(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases.as_ref().as_ptr(),
            transpose,
            int(group_size),
            int(bits),
            c"affine".as_ptr(),
            s,
        )
    })
}

/// `x` where `condition` holds, `y` elsewhere.
pub fn select(
    condition: impl AsRef<Array>,
    x: impl AsRef<Array>,
    y: impl AsRef<Array>,
) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_where(
            res,
            condition.as_ref().as_ptr(),
            x.as_ref().as_ptr(),
            y.as_ref().as_ptr(),
            s,
        )
    })
}

/// `1 / (1 + exp(-x))`.
pub fn sigmoid(x: impl AsRef<Array>) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_sigmoid(res, x.as_ref().as_ptr(), s) })
}

/// The softmax along an axis.
pub fn softmax_axis(
    a: impl AsRef<Array>,
    axis: i32,
    precise: impl Into<Option<bool>>,
) -> Result<Array> {
    let precise = flag(precise, false);
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_softmax_axis(res, a.as_ref().as_ptr(), axis, precise, s) })
}

/// Sorted along an axis.
pub fn sort_axis(a: impl AsRef<Array>, axis: i32) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_sort_axis(res, a.as_ref().as_ptr(), axis, s) })
}

/// The arrays stacked along a new axis.
pub fn stack(arrays: &[impl AsRef<Array>], axis: i32) -> Result<Array> {
    let arrays = VectorArray::from_arrays(arrays.iter())?;
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_stack_axis(res, arrays.as_ptr(), axis, s) })
}

/// The sum along an axis.
pub fn sum_axis(
    a: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    a.as_ref().sum_axis(axis, keep_dims)
}

/// Two axes exchanged.
pub fn swap_axes(a: impl AsRef<Array>, axis1: i32, axis2: i32) -> Result<Array> {
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_swapaxes(res, a.as_ref().as_ptr(), axis1, axis2, s) })
}

/// The lower triangle, from the `k`th diagonal (the main one by default).
pub fn tril(a: impl AsRef<Array>, k: impl Into<Option<i32>>) -> Result<Array> {
    let k = k.into().unwrap_or(0);
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_tril(res, a.as_ref().as_ptr(), k, s) })
}

/// The upper triangle, from the `k`th diagonal (the main one by default).
pub fn triu(a: impl AsRef<Array>, k: impl Into<Option<i32>>) -> Result<Array> {
    let k = k.into().unwrap_or(0);
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_triu(res, a.as_ref().as_ptr(), k, s) })
}

/// An array of zeros, as `T`.
pub fn zeros<T: ArrayElement>(shape: &[i32]) -> Result<Array> {
    zeros_dtype(shape, T::DTYPE)
}

/// An array of zeros, as `dtype`.
pub fn zeros_dtype(shape: &[i32], dtype: Dtype) -> Result<Array> {
    // SAFETY: valid handles and `shape`'s elements, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_zeros(res, shape.as_ptr(), shape.len(), dtype as sys::mlx_dtype, s)
    })
}

/// An array of zeros with `a`'s shape and dtype.
pub fn zeros_like(a: impl AsRef<Array>) -> Result<Array> {
    let a = a.as_ref();
    zeros_dtype(a.shape(), a.dtype())
}

#[cfg(test)]
mod tests {
    use crate::mlx::{Array, Dtype};

    #[test]
    fn a_view_keeps_the_bytes() {
        let words = [0, -1, i32::MIN, 0x0123_4567, -0x7654_3211, 0x3f80_0000];
        let a = Array::from_slice(&words, &[2, 3]);

        let u = a.view(Dtype::Uint32).unwrap();
        assert_eq!((u.shape(), u.dtype()), (&[2, 3][..], Dtype::Uint32));
        assert_eq!(u.as_slice::<u32>(), words.map(|w| w as u32));

        let f = a.view(Dtype::Float32).unwrap();
        let bits: Vec<u32> = f.as_slice::<f32>().iter().map(|x| x.to_bits()).collect();
        assert_eq!(bits, words.map(|w| w as u32));

        let b = a.view(Dtype::Uint8).unwrap();
        assert_eq!(b.shape(), &[2, 12]);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(b.as_slice::<u8>(), bytes);

        let back = b.view(Dtype::Int32).unwrap();
        assert_eq!(back.shape(), &[2, 3]);
        assert_eq!(back.as_slice::<i32>(), words);
    }
}
