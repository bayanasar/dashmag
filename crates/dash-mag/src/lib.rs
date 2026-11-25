//! The one crate an application names: the gateway to dash, which runs on this machine, and to
//! the `mag-` crates, planned and not started, which will pool devices across machines. A model and
//! an algorithm are written once against it and run on every platform.
//!
//! ```text
//! algorithm → dash-mag ─┬ dash-ops → Burn → CubeCL → CUDA, ROCm (or Flex)
//!                       └ dash-mlx → MLX → Metal (macOS)
//! ```
//!
//! An application names only this crate: a [`Tensor`], its array operations, [`vjp`], the
//! operations in [`ops`], [`lora`], [`rope`] and [`optim`]. Which side runs is chosen at compile
//! time by the target, with no dispatch at run time, and a macOS build does not link Burn. Both
//! sides implement one internal trait, so that every method here exists on both or does not
//! compile; the gateway itself, its checks, its broadcasting and its dtype rules, is written once.
//!
//! Beside them, the modules training uses, each written once: [`loss`], a multi-head classifier's
//! masked, class-weighted cross-entropy over [`Tensor`]; [`norm`], batch normalisation in training
//! and in eval mode, from the tensor's own reductions; [`schedule`], `OneCycleLR`; and on the host,
//! [`metrics`] and [`calibrate`].
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
//! (dash-ops' patched cubecl-cuda, a `[patch]` a dependent workspace has to repeat; without them
//! a device that would run f32 matmuls in TF32 refuses the first tensor) and `MLX_ENABLE_TF32=0`
//! on MLX.

mod arm;
mod autodiff;
pub mod calibrate;
pub mod lora;
pub mod loss;
pub mod metrics;
pub mod norm;
pub mod ops;
pub mod optim;
pub mod peft;
pub mod rope;
pub mod schedule;
mod tensor;

pub use autodiff::vjp;
pub use tensor::{
    DType, Memory, Tensor, eval, eval_async, memory, reset_peak_memory, set_cache_limit,
};

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

/// Chooses the device this process runs on, by its index, before the first [`Tensor`]: on CUDA and
/// ROCm, among the cards the process sees (`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`); Flex and
/// MLX have one device, 0. Without a call it is device 0. Choosing again the device in use is
/// nothing; choosing another once a tensor is made is refused, as is an index the side has not
/// (on CUDA and ROCm the first tensor fails instead, as Burn fails it).
///
/// One process runs on one device: a machine's other cards are other processes, each choosing its
/// own.
pub fn use_device(index: usize) -> Result<()> {
    <arm::Selected as arm::Arm>::use_device(index)
}

/// Returns what the device holds in caches to it, where the side has such a cache.
pub fn clear_cache() {
    <arm::Selected as arm::Arm>::clear_cache();
}
