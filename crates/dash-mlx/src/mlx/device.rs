//! MLX's devices and streams.

use super::error::{self, Result, check};
use super::sys;
use std::cell::Cell;
use std::ffi::CStr;

/// A device arrays are computed on.
pub struct Device(sys::mlx_device);

impl Device {
    /// The CPU.
    pub fn cpu() -> Self {
        error::init();
        // SAFETY: makes a new handle from values.
        Self(unsafe { sys::mlx_device_new_type(sys::mlx_device_type__MLX_CPU, 0) })
    }

    /// The GPU.
    pub fn gpu() -> Self {
        error::init();
        // SAFETY: makes a new handle from values.
        Self(unsafe { sys::mlx_device_new_type(sys::mlx_device_type__MLX_GPU, 0) })
    }

    /// Makes this the device operations run on from now on.
    pub fn set_default(device: &Device) {
        error::init();
        // SAFETY: a valid handle.
        unsafe { sys::mlx_set_default_device(device.0) };
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: this handle owns its reference, and is freed once.
        unsafe { sys::mlx_device_free(self.0) };
    }
}

thread_local! {
    static ON_CPU: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` with this thread's operations on the CPU, as f64 needs: MLX's GPU refuses it. MLX's default
/// device is one per process, so this is a setting here, of this thread alone. A Metal kernel stays
/// on the GPU.
pub fn with_cpu<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ON_CPU.set(self.0);
        }
    }
    let _restore = Restore(ON_CPU.replace(true));
    f()
}

/// A queue of work on a device.
pub struct Stream(sys::mlx_stream);

impl Stream {
    /// The default stream of the default device, or the CPU's inside [`with_cpu`]. An error if the
    /// device cannot run, as where MLX finds no Metal library.
    #[inline]
    pub fn new() -> Result<Self> {
        error::init();
        if ON_CPU.get() {
            // SAFETY: makes a new handle.
            return Ok(Self(unsafe { sys::mlx_default_cpu_stream_new() }));
        }
        // SAFETY: both handles are new, filled here, and each freed once by its owner.
        unsafe {
            let mut device = Device(sys::mlx_device_new());
            check(sys::mlx_get_default_device(&mut device.0))?;
            let mut stream = Self(sys::mlx_stream_new());
            check(sys::mlx_get_default_stream(&mut stream.0, device.0))?;
            Ok(stream)
        }
    }

    /// The GPU's default stream.
    pub fn gpu() -> Self {
        error::init();
        // SAFETY: makes a new handle.
        Self(unsafe { sys::mlx_default_gpu_stream_new() })
    }

    /// The handle, still owned by this stream.
    pub fn as_ptr(&self) -> sys::mlx_stream {
        self.0
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: this handle owns its reference, and is freed once.
        unsafe { sys::mlx_stream_free(self.0) };
    }
}

/// Waits for everything queued on the default stream to finish.
pub fn synchronize() -> Result<()> {
    let stream = Stream::new()?;
    // SAFETY: a valid handle, for the length of the call.
    check(unsafe { sys::mlx_synchronize(stream.as_ptr()) })
}

/// MLX's version.
pub fn version() -> Result<String> {
    error::init();
    // SAFETY: the string is new, filled here and freed once.
    unsafe {
        let mut text = sys::mlx_string_new();
        let status = sys::mlx_version(&mut text);
        let version = CStr::from_ptr(sys::mlx_string_data(text))
            .to_string_lossy()
            .into_owned();
        sys::mlx_string_free(text);
        check(status)?;
        Ok(version)
    }
}

#[cfg(test)]
mod tests {
    use super::with_cpu;
    use crate::mlx::{Array, Dtype};

    #[test]
    fn f64_runs_on_the_cpu_and_the_setting_ends_with_the_call() {
        let x = Array::from_slice_f64(&[1.5, 2.5], &[2]);
        let sum = with_cpu(|| x.add(&x).unwrap().sum(None).unwrap());
        assert_eq!(sum.dtype(), Dtype::Float64);
        assert_eq!(sum.as_slice::<f64>(), [8.0]);
        let after = std::panic::catch_unwind(|| with_cpu(|| panic!("in the scope")));
        assert!(after.is_err());
        // On the GPU again, where MLX refuses f64.
        assert!(x.add(&x).is_err());
    }
}
