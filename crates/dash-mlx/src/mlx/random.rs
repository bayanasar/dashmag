//! MLX's random numbers. Each function takes a key; without one it uses the next of this thread's
//! own sequence, seeded from the clock.

use super::array::{Array, ArrayElement, new_array, op};
use super::error::Result;
use super::ops::indexing::TryIndexOp;
use super::sys;
use std::borrow::Cow;
use std::cell::RefCell;
use std::time::{SystemTime, UNIX_EPOCH};

/// A key from a seed.
pub fn key(seed: u64) -> Result<Array> {
    // SAFETY: fills the new handle.
    new_array(|res| unsafe { sys::mlx_random_key(res, seed) })
}

thread_local! {
    static STATE: RefCell<Option<Array>> = const { RefCell::new(None) };
}

fn next_key() -> Result<Array> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let current = match state.take() {
            Some(current) => current,
            None => {
                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
                key(now.as_nanos() as u64)?
            }
        };
        // SAFETY: valid handles, for the length of the call.
        let keys = op(|res, s| unsafe { sys::mlx_random_split_num(res, current.as_ptr(), 2, s) })?;
        *state = Some(keys.try_index(0)?);
        keys.try_index(1)
    })
}

fn resolve<'a>(key: impl Into<Option<&'a Array>>) -> Result<Cow<'a, Array>> {
    match key.into() {
        Some(key) => Ok(Cow::Borrowed(key)),
        None => Ok(Cow::Owned(next_key()?)),
    }
}

/// Normal variates as `T`, of mean `loc` (0 if `None`) and scale `scale` (1 if `None`).
pub fn normal<'a, T: ArrayElement>(
    shape: &[i32],
    loc: impl Into<Option<f32>>,
    scale: impl Into<Option<f32>>,
    key: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let (loc, scale) = (loc.into().unwrap_or(0.0), scale.into().unwrap_or(1.0));
    let key = resolve(key)?;
    // SAFETY: valid handles and `shape`'s elements, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_random_normal(
            res,
            shape.as_ptr(),
            shape.len(),
            T::DTYPE as sys::mlx_dtype,
            loc,
            scale,
            key.as_ptr(),
            s,
        )
    })
}

/// Booleans of this shape, each true with probability `p`.
pub fn bernoulli<'a>(p: &Array, shape: &[i32], key: impl Into<Option<&'a Array>>) -> Result<Array> {
    let key = resolve(key)?;
    // SAFETY: valid handles and `shape`'s elements, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_random_bernoulli(
            res,
            p.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            key.as_ptr(),
            s,
        )
    })
}

/// Samples of the categorical distribution these logits give along `axis`.
pub fn categorical<'a>(
    logits: impl AsRef<Array>,
    axis: i32,
    key: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let key = resolve(key)?;
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_random_categorical(res, logits.as_ref().as_ptr(), axis, key.as_ptr(), s)
    })
}
