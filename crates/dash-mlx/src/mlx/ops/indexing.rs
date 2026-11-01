//! Indexing an array as `a.try_index(1..3)` or `a.try_index((.., 0, 2..))`: a range keeps its axis,
//! an integer removes it, an array takes along the first axis. A tuple's trailing axes stay whole.

use crate::mlx::array::{Array, op};
use crate::mlx::error::{Exception, Result};
use crate::mlx::sys;
use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

/// What one index does to its axis.
#[derive(Debug, Clone, Copy)]
pub enum AxisIndex {
    /// Keep the one element at this index, negative from the end, and remove the axis.
    Take(i32),
    /// Keep this range, each end negative from the end.
    Slice(Bound<i32>, Bound<i32>),
}

/// A range or an integer used as an index.
pub trait ArrayIndex {
    /// What it does to its axis.
    fn axis_index(self) -> AxisIndex;
}

impl ArrayIndex for i32 {
    fn axis_index(self) -> AxisIndex {
        AxisIndex::Take(self)
    }
}

macro_rules! range_index {
    ($($t:ty),*) => {$(
        impl ArrayIndex for $t {
            fn axis_index(self) -> AxisIndex {
                AxisIndex::Slice(self.start_bound().cloned(), self.end_bound().cloned())
            }
        }
    )*};
}

range_index!(
    Range<i32>,
    RangeFrom<i32>,
    RangeInclusive<i32>,
    RangeTo<i32>,
    RangeToInclusive<i32>,
    RangeFull
);

fn start(bound: Bound<i32>) -> i32 {
    match bound {
        Bound::Included(start) => start,
        Bound::Excluded(start) => start + 1,
        Bound::Unbounded => 0,
    }
}

fn stop(bound: Bound<i32>, size: i32) -> i32 {
    match bound {
        Bound::Included(stop) => stop + 1,
        Bound::Excluded(stop) => stop,
        Bound::Unbounded => size,
    }
}

fn resolve(index: i32, size: i32) -> i32 {
    if index < 0 { size + index } else { index }
}

fn slice(a: &Array, starts: &[i32], stops: &[i32], strides: &[i32]) -> Result<Array> {
    // SAFETY: valid handles and the slices' elements, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_slice(
            res,
            a.as_ptr(),
            starts.as_ptr(),
            starts.len(),
            stops.as_ptr(),
            stops.len(),
            strides.as_ptr(),
            strides.len(),
            s,
        )
    })
}

fn too_many(indices: usize, ndim: usize) -> Exception {
    Exception::custom(format!("{indices} indices into an array of {ndim} axes"))
}

/// One index for each of the leading axes: the slice, then the integers' axes reshaped away.
fn get_item_nd(a: &Array, indices: &[AxisIndex]) -> Result<Array> {
    if indices.len() > a.ndim() {
        return Err(too_many(indices.len(), a.ndim()));
    }
    let mut starts = vec![0; a.ndim()];
    let mut stops = a.shape().to_vec();
    let strides = vec![1; a.ndim()];
    let mut squeeze = false;
    for (axis, index) in indices.iter().enumerate() {
        let size = a.dim(axis as i32);
        match *index {
            AxisIndex::Take(index) => {
                let index = resolve(index, size);
                (starts[axis], stops[axis]) = (index, index + 1);
                squeeze = true;
            }
            AxisIndex::Slice(from, to) => {
                (starts[axis], stops[axis]) = (start(from), stop(to, size));
            }
        }
    }
    let out = slice(a, &starts, &stops, &strides)?;
    if !squeeze {
        return Ok(out);
    }
    let kept = indices
        .iter()
        .enumerate()
        .filter(|(_, index)| matches!(index, AxisIndex::Slice(..)))
        .map(|(axis, _)| out.dim(axis as i32));
    let shape: Vec<i32> = kept
        .chain(out.shape()[indices.len()..].iter().copied())
        .collect();
    out.reshape(&shape)
}

/// Indexing an array.
pub trait TryIndexOp<Index> {
    /// The elements at `index`.
    fn try_index(&self, index: Index) -> Result<Array>;
}

impl<T: ArrayIndex> TryIndexOp<T> for Array {
    fn try_index(&self, index: T) -> Result<Array> {
        if self.ndim() == 0 {
            return Err(too_many(1, 0));
        }
        match index.axis_index() {
            AxisIndex::Take(index) => {
                let index = resolve(index, self.dim(0));
                self.take_axis(Array::from_int(index), 0)
            }
            AxisIndex::Slice(from, to) => {
                let mut starts = vec![0; self.ndim()];
                let mut stops = self.shape().to_vec();
                let strides = vec![1; self.ndim()];
                (starts[0], stops[0]) = (start(from), stop(to, stops[0]));
                slice(self, &starts, &stops, &strides)
            }
        }
    }
}

impl TryIndexOp<&Array> for Array {
    fn try_index(&self, indices: &Array) -> Result<Array> {
        self.take_axis(indices, 0)
    }
}

impl TryIndexOp<Array> for Array {
    fn try_index(&self, indices: Array) -> Result<Array> {
        self.take_axis(&indices, 0)
    }
}

macro_rules! tuple_index {
    ($($t:ident $n:tt),*) => {
        impl<$($t: ArrayIndex),*> TryIndexOp<($($t,)*)> for Array {
            fn try_index(&self, index: ($($t,)*)) -> Result<Array> {
                get_item_nd(self, &[$(index.$n.axis_index()),*])
            }
        }
    };
}

tuple_index!(A 0);
tuple_index!(A 0, B 1);
tuple_index!(A 0, B 1, C 2);
tuple_index!(A 0, B 1, C 2, D 3);
tuple_index!(A 0, B 1, C 2, D 3, E 4);

/// The index of the largest element along an axis.
pub fn argmax_axis(
    a: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    let keep_dims = keep_dims.into().unwrap_or(false);
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe { sys::mlx_argmax_axis(res, a.as_ref().as_ptr(), axis, keep_dims, s) })
}

#[cfg(test)]
mod tests {
    use super::TryIndexOp;
    use crate::mlx::Array;

    fn numbered() -> Array {
        Array::from_slice(&(0..24).collect::<Vec<i32>>(), &[2, 3, 4])
    }

    fn values(a: Array) -> Vec<i32> {
        a.contiguous().unwrap().as_slice::<i32>().to_vec()
    }

    #[test]
    fn a_range_keeps_its_axis_and_an_integer_removes_it() {
        let a = numbered();
        let s = a.try_index((.., 1, 1..3)).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(values(s), [5, 6, 17, 18]);
        let s = a.try_index(-1).unwrap();
        assert_eq!(s.shape(), &[3, 4]);
        assert_eq!(values(s)[0], 12);
        let s = a.try_index((.., -1)).unwrap();
        assert_eq!(s.shape(), &[2, 4]);
        assert_eq!(values(s)[0], 8);
        assert_eq!(a.try_index(1..2).unwrap().shape(), &[1, 3, 4]);
    }

    #[test]
    fn more_indices_than_axes_are_an_error() {
        let scalar = Array::from_int(1);
        assert!(scalar.try_index(0).is_err());
        assert!(scalar.try_index(0..1).is_err());
        assert!(numbered().try_index((0, 0, 0, 0)).is_err());
    }
}
