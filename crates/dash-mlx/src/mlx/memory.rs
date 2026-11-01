//! MLX's allocator, for the benchmarks' peaks. All of it is one state per process.

use super::error::{self, Result, check};
use super::sys;

/// Frees what MLX keeps cached for reuse.
pub fn clear_cache() -> Result<()> {
    error::init();
    // SAFETY: no arguments.
    check(unsafe { sys::mlx_clear_cache() })
}

/// The bytes in live arrays.
pub fn active_memory() -> Result<usize> {
    error::init();
    let mut res = 0;
    // SAFETY: writes the one result.
    check(unsafe { sys::mlx_get_active_memory(&mut res) })?;
    Ok(res)
}

/// The most bytes in live arrays since the last [`reset_peak_memory`].
pub fn peak_memory() -> Result<usize> {
    error::init();
    let mut res = 0;
    // SAFETY: writes the one result.
    check(unsafe { sys::mlx_get_peak_memory(&mut res) })?;
    Ok(res)
}

/// Starts [`peak_memory`] again from now.
pub fn reset_peak_memory() -> Result<()> {
    error::init();
    // SAFETY: no arguments.
    check(unsafe { sys::mlx_reset_peak_memory() })
}
