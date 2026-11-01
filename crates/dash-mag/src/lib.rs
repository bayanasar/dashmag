//! The one crate an application names: the gateway to dash, which runs on this machine, and to
//! the `mag-` crates, planned and not started, which will pool devices across machines. A model and
//! an algorithm are written once against it and run on every platform.
//!
//! ```text
//! algorithm → dash-mag ─┬ dash-ops, dash-lora's arithmetic → Burn → CubeCL → CUDA (or Flex)
//!                       └ dash-mlx → MLX → Metal (macOS)
//! ```
//!
//! An application names only this crate: a [`Tensor`], its array operations, [`vjp`], the
//! operations in [`ops`], [`lora`], [`rope`] and [`optim`]. Which side runs is chosen at compile
//! time by the target, with no dispatch at run time, and a macOS build does not link Burn. Both
//! sides implement one internal trait, so that every method here exists on both or does not
//! compile; the gateway itself, its checks, its broadcasting and its dtype rules, is written once.
//!
//! This crate holds no model, generation loop or training algorithm: those are the application's.
//!
//! # Conventions
//!
//! A [`Tensor`] has a dynamic rank, from 0 to 6, and a [`DType`]. Axes may count from the end
//! (`-1` is the last), and shapes broadcast as NumPy's do. Two floating dtypes in one operation
//! promote to the wider, with f32 for bf16 beside f16. Operations return a [`Result`]: what the
//! gateway can check, it refuses before either side runs. The options of [`ops`] are
//! dash-contract's, the same on both sides, and so are their layouts and their dtypes.
//!
//! A [`Tensor`] stays on the thread that made it, on every side: an unevaluated MLX array can hold
//! closures that are not `Send`. Hand values between threads, not tensors:
//!
//! ```compile_fail
//! fn send<T: Send>(_: T) {}
//! send(dash_mag::Tensor::zeros(&[1], dash_mag::DType::F32).unwrap());
//! ```
//!
//! # Gradients
//!
//! [`vjp`] is the only way to a gradient: `vjp(f, primals, cotangents)` runs `f` on `primals` and
//! gives each primal the gradient of `sum(out * cotangent)`. On MLX it is MLX's own; on Burn it
//! records `f` and runs the backward pass, with dash-ops' written-out backward passes where the
//! operations have them. A tensor `f` captures rather than takes as a primal is a constant, as a
//! frozen weight is.
//!
//! # Numbers
//!
//! Different devices round differently, so the two sides agree within the bounds dash-ops'
//! operations are held to, not bit for bit. dash-ops and dash-mlx each hold their operations to
//! the same recorded references (dash-conformance); this crate's tests hold its entry points to
//! direct calls of them. f32 results assume IEEE f32 products: `CUBECL_CUDA_TF32=0` on CUDA
//! (dash-ops' patched cubecl-cuda, a `[patch]` a dependent workspace has to repeat) and
//! `MLX_ENABLE_TF32=0` on MLX.

// The macOS side is not written yet: until it is, the crate is empty there, as dash-mlx is off
// macOS, and the workspace builds.
#![cfg(not(target_os = "macos"))]

mod arm;
mod autodiff;
pub mod lora;
pub mod ops;
pub mod optim;
pub mod rope;
mod tensor;

pub use autodiff::vjp;
pub use tensor::{DType, Tensor, eval};

/// What every fallible operation here returns.
pub type Result<T> = std::result::Result<T, Error>;

/// Why an operation refused or failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub(crate) fn refuse<T>(message: impl Into<String>) -> Result<T> {
    Err(Error(message.into()))
}

/// What the side this build runs on is, for a log line: its device or framework.
pub fn describe() -> String {
    <arm::Selected as arm::Arm>::describe()
}

/// Returns what the device holds in caches to it, where the side has such a cache.
pub fn clear_cache() {
    <arm::Selected as arm::Arm>::clear_cache();
}
