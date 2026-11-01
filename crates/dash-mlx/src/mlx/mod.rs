//! A thin layer over Apple's mlx-c, holding just what this crate's operations call: [`Array`],
//! MLX's errors as [`Exception`](error::Exception)s, the operations, Rust closures for
//! [`transforms::fallible_vjp`], [`transforms::custom_vjp`] and [`transforms::checkpoint`],
//! [`fast::MetalKernel`], and parameters by their paths in place of a derive
//! ([`module_parameters!`](crate::module_parameters)). [`sys`] is the bindings `build.rs` generates
//! from the headers in `mlx-c/`.

pub mod array;
pub mod device;
pub mod error;
pub mod fast;
pub mod memory;
pub mod module;
pub mod nn;
pub mod ops;
pub mod random;
pub mod sys;
pub mod transforms;

pub use array::{Array, ArrayElement, Dtype, FromSliceElement};
pub use device::{Device, Stream, synchronize, version, with_cpu};
pub use ops::stop_gradient;
