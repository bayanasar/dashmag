//! MLX's devices and streams.

use super::error::{self, Result, check};
use super::sys;

/// A device arrays are computed on.
pub struct Device(sys::mlx_device);

impl Device {
    /// The CPU.
    pub fn cpu() -> Self {
        // SAFETY: makes a new handle from values.
        Self(unsafe { sys::mlx_device_new_type(sys::mlx_device_type__MLX_CPU, 0) })
    }

    /// The GPU.
    pub fn gpu() -> Self {
        // SAFETY: makes a new handle from values.
        Self(unsafe { sys::mlx_device_new_type(sys::mlx_device_type__MLX_GPU, 0) })
    }

    /// Makes this the device operations run on from now on.
    pub fn set_default(device: &Device) {
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

/// A queue of work on a device.
pub struct Stream(sys::mlx_stream);

impl Stream {
    /// The default stream of the default device. An error if the device cannot run, as where MLX
    /// finds no Metal library.
    #[inline]
    pub fn new() -> Result<Self> {
        error::init();
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
