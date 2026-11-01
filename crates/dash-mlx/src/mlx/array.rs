//! MLX's array, its dtypes and its host data, and the two things every operation needs: the stream
//! it runs on and the array it returns.

use super::device::Stream;
use super::error::{self, Exception, Result, check};
use super::sys;
use half::{bf16, f16};
use std::ffi::{CStr, c_int, c_void};

/// An MLX array: a counted handle, cheap to clone, lazily evaluated. Not `Send`: its graph may hold
/// Rust closures, a custom VJP's, which need not be.
#[repr(transparent)]
pub struct Array(sys::mlx_array);

impl Drop for Array {
    fn drop(&mut self) {
        // SAFETY: this handle owns its reference, and is freed once.
        unsafe { sys::mlx_array_free(self.0) };
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        new_array(|res| unsafe { sys::mlx_array_set(res, self.0) }).expect("a clone of an array")
    }
}

impl AsRef<Array> for Array {
    fn as_ref(&self) -> &Array {
        self
    }
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // SAFETY: the string is new, filled here and freed once.
        unsafe {
            let mut text = sys::mlx_string_new();
            let status = sys::mlx_array_tostring(&mut text, self.0);
            let result = match status {
                0 => write!(
                    f,
                    "{}",
                    CStr::from_ptr(sys::mlx_string_data(text)).to_string_lossy()
                ),
                _ => Err(std::fmt::Error),
            };
            sys::mlx_string_free(text);
            result
        }
    }
}

/// An array filled by a call of mlx-c; `f` is handed its result to fill.
#[track_caller]
#[inline]
pub(crate) fn new_array(f: impl FnOnce(*mut sys::mlx_array) -> c_int) -> Result<Array> {
    error::init();
    // SAFETY: a new empty handle, which the result owns (and frees) whatever the status.
    let mut res = Array(unsafe { sys::mlx_array_new() });
    check(f(&mut res.0))?;
    Ok(res)
}

/// An operation's result, on the default device's stream; `f` is handed its result and the stream.
#[track_caller]
#[inline]
pub(crate) fn op(f: impl FnOnce(*mut sys::mlx_array, sys::mlx_stream) -> c_int) -> Result<Array> {
    let stream = Stream::new()?;
    new_array(|res| f(res, stream.as_ptr()))
}

mod sealed {
    pub trait Sealed {}
}

/// An element type of an [`Array`]'s host data. Sealed, since [`Array::try_as_slice`] trusts `DTYPE`
/// and `data` for the size of what it reads.
pub trait ArrayElement: Copy + sealed::Sealed {
    /// Its dtype.
    const DTYPE: Dtype;

    /// The data of an evaluated array of this dtype.
    #[doc(hidden)]
    fn data(array: &Array) -> *const Self;
}

macro_rules! array_elements {
    ($($t:ty = $dtype:ident, $data:ident;)*) => {$(
        impl sealed::Sealed for $t {}

        impl ArrayElement for $t {
            const DTYPE: Dtype = Dtype::$dtype;

            fn data(array: &Array) -> *const Self {
                // SAFETY: the caller's array, whose dtype it has checked.
                unsafe { sys::$data(array.0) as *const Self }
            }
        }
    )*};
}

array_elements! {
    bool = Bool, mlx_array_data_bool;
    u8 = Uint8, mlx_array_data_uint8;
    u16 = Uint16, mlx_array_data_uint16;
    u32 = Uint32, mlx_array_data_uint32;
    u64 = Uint64, mlx_array_data_uint64;
    i8 = Int8, mlx_array_data_int8;
    i16 = Int16, mlx_array_data_int16;
    i32 = Int32, mlx_array_data_int32;
    i64 = Int64, mlx_array_data_int64;
    f16 = Float16, mlx_array_data_float16;
    f32 = Float32, mlx_array_data_float32;
    f64 = Float64, mlx_array_data_float64;
    bf16 = Bfloat16, mlx_array_data_bfloat16;
}

/// An element type [`Array::from_slice`] takes: all but f64, which MLX computes on the CPU alone and
/// which a float literal without a type would otherwise pick. [`Array::from_slice_f64`] names it.
pub trait FromSliceElement: ArrayElement {}

macro_rules! from_slice_elements {
    ($($t:ty),*) => {$(impl FromSliceElement for $t {})*};
}

from_slice_elements!(bool, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, bf16);

impl Array {
    /// The array behind a handle, whose one reference it takes over.
    ///
    /// # Safety
    ///
    /// `handle` is a valid array handle nothing else frees.
    pub unsafe fn from_ptr(handle: sys::mlx_array) -> Self {
        Self(handle)
    }

    /// The handle, still owned by this array.
    pub fn as_ptr(&self) -> sys::mlx_array {
        self.0
    }

    /// A boolean scalar.
    pub fn from_bool(value: bool) -> Self {
        error::init();
        // SAFETY: makes a new handle from a value.
        Self(unsafe { sys::mlx_array_new_bool(value) })
    }

    /// An `i32` scalar.
    pub fn from_int(value: i32) -> Self {
        error::init();
        // SAFETY: makes a new handle from a value.
        Self(unsafe { sys::mlx_array_new_int(value) })
    }

    /// An `f32` scalar.
    pub fn from_f32(value: f32) -> Self {
        error::init();
        // SAFETY: makes a new handle from a value.
        Self(unsafe { sys::mlx_array_new_float32(value) })
    }

    /// An array with a copy of `data`, row-major, of this shape. Panics unless the two agree.
    pub fn from_slice<T: FromSliceElement>(data: &[T], shape: &[i32]) -> Self {
        Self::from_data(data, shape)
    }

    /// [`Array::from_slice`] for f64, which only the CPU computes ([`with_cpu`](super::with_cpu)).
    pub fn from_slice_f64(data: &[f64], shape: &[i32]) -> Self {
        Self::from_data(data, shape)
    }

    fn from_data<T: ArrayElement>(data: &[T], shape: &[i32]) -> Self {
        let size = shape.iter().try_fold(1usize, |size, &d| {
            size.checked_mul(usize::try_from(d).ok()?)
        });
        assert_eq!(
            Some(data.len()),
            size,
            "{} elements in shape {shape:?}",
            data.len()
        );
        error::init();
        // SAFETY: mlx-c copies `data`, whose length the assertion holds to the shape.
        Self(unsafe {
            sys::mlx_array_new_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len() as c_int,
                T::DTYPE as sys::mlx_dtype,
            )
        })
    }

    /// The number of elements.
    pub fn size(&self) -> usize {
        // SAFETY: a valid handle.
        unsafe { sys::mlx_array_size(self.0) }
    }

    /// The number of axes.
    pub fn ndim(&self) -> usize {
        // SAFETY: a valid handle.
        unsafe { sys::mlx_array_ndim(self.0) }
    }

    /// The shape.
    pub fn shape(&self) -> &[i32] {
        let ndim = self.ndim();
        if ndim == 0 {
            // The pointer may be null with no axes.
            return &[];
        }
        // SAFETY: `ndim` integers, for as long as this handle's array, which never changes shape.
        unsafe { std::slice::from_raw_parts(sys::mlx_array_shape(self.0), ndim) }
    }

    /// The size of an axis, negative from the last.
    pub fn dim(&self, axis: i32) -> i32 {
        let axis = if axis < 0 {
            self.ndim() as i32 + axis
        } else {
            axis
        };
        self.shape()[axis as usize]
    }

    /// The dtype.
    pub fn dtype(&self) -> Dtype {
        // SAFETY: a valid handle.
        Dtype::from_raw(unsafe { sys::mlx_array_dtype(self.0) })
    }

    /// Evaluates the array.
    pub fn eval(&self) -> Result<()> {
        error::init();
        // SAFETY: a valid handle.
        check(unsafe { sys::mlx_array_eval(self.0) })
    }

    /// The elements, evaluating the array, or an error if the dtype is not `T`'s, the data is not
    /// row-contiguous ([`Array::contiguous`]) or the array is empty, which holds no data.
    pub fn try_as_slice<T: ArrayElement>(&self) -> Result<&[T]> {
        if self.dtype() != T::DTYPE {
            return Err(Exception::custom(format!(
                "dtype mismatch: expected {:?}, found {:?}",
                T::DTYPE,
                self.dtype()
            )));
        }
        self.eval()?;
        let mut contiguous = false;
        // SAFETY: a valid handle.
        check(unsafe { sys::_mlx_array_is_row_contiguous(&mut contiguous, self.0) })?;
        if !contiguous {
            return Err(Exception::custom(
                "the array's data is not row-contiguous; call `contiguous` first",
            ));
        }
        let data = T::data(self);
        if data.is_null() || self.size() == 0 {
            return Err(Exception::custom("the array holds no data"));
        }
        // SAFETY: `size` elements of `T`, evaluated and row-contiguous, for as long as this array.
        Ok(unsafe { std::slice::from_raw_parts(data, self.size()) })
    }

    /// The elements as [`Array::try_as_slice`] gives them; panics where that is an error.
    pub fn as_slice<T: ArrayElement>(&self) -> &[T] {
        self.try_as_slice().unwrap()
    }
}

/// An array's element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(missing_docs)]
pub enum Dtype {
    Bool = sys::mlx_dtype__MLX_BOOL,
    Uint8 = sys::mlx_dtype__MLX_UINT8,
    Uint16 = sys::mlx_dtype__MLX_UINT16,
    Uint32 = sys::mlx_dtype__MLX_UINT32,
    Uint64 = sys::mlx_dtype__MLX_UINT64,
    Int8 = sys::mlx_dtype__MLX_INT8,
    Int16 = sys::mlx_dtype__MLX_INT16,
    Int32 = sys::mlx_dtype__MLX_INT32,
    Int64 = sys::mlx_dtype__MLX_INT64,
    Float16 = sys::mlx_dtype__MLX_FLOAT16,
    Float32 = sys::mlx_dtype__MLX_FLOAT32,
    Float64 = sys::mlx_dtype__MLX_FLOAT64,
    Bfloat16 = sys::mlx_dtype__MLX_BFLOAT16,
    Complex64 = sys::mlx_dtype__MLX_COMPLEX64,
}

impl Dtype {
    /// Whether it is a real floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16
        )
    }

    fn from_raw(raw: sys::mlx_dtype) -> Self {
        const ALL: [Dtype; 14] = [
            Dtype::Bool,
            Dtype::Uint8,
            Dtype::Uint16,
            Dtype::Uint32,
            Dtype::Uint64,
            Dtype::Int8,
            Dtype::Int16,
            Dtype::Int32,
            Dtype::Int64,
            Dtype::Float16,
            Dtype::Float32,
            Dtype::Float64,
            Dtype::Bfloat16,
            Dtype::Complex64,
        ];
        *ALL.iter()
            .find(|dtype| **dtype as sys::mlx_dtype == raw)
            .expect("a dtype of mlx-c's")
    }
}

#[cfg(test)]
mod tests {
    use super::Array;

    #[test]
    #[should_panic(expected = "0 elements in shape [65536, 65536]")]
    fn a_shape_whose_size_overflows_i32_is_refused() {
        Array::from_slice::<f32>(&[], &[65536, 65536]);
    }
}
