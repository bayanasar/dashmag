//! 2-D convolution of images, torch's `F.conv2d`: any kernel size, stride, symmetric zero padding
//! and dilation, channels in groups, and an optional bias; with the gradients of the input, the
//! weight and the bias: dash-ops' `conv2d` on MLX.
//!
//! [`dash_contract::conv2d`] defines the operation: its NCHW and OIHW layouts, its dtypes and
//! options, and the bound in roundings each element of the output and of each gradient is held
//! to. This module implements it with MLX's own `conv2d`, which MLX differentiates by its own
//! backward pass. MLX convolves channels-last, `x` `[N, H, W, C]` and the weight `[O, KH, KW,
//! C / G]`: the call transposes `x` and the weight to that layout, convolves, adds the bias over
//! the last axis, and transposes the output back to `[N, O, HO, WO]`; MLX's backward pass goes
//! through the same transposes. What the transposes cost per call is not measured. A model that
//! convolves many times could hold its activations channels-last on this side instead; that is
//! not built.
//!
//! f32 only, as on Burn: bf16 and f16 are refused until they are held to the contract. f32
//! products are IEEE f32 with `MLX_ENABLE_TF32=0`.

use crate::mlx::error::{Exception, Result};
use crate::mlx::ops::conv2d as mlx_conv2d;
use crate::mlx::{Array, Dtype};

pub use dash_contract::conv2d::Conv2dOptions;

fn contract(message: String) -> Exception {
    Exception::custom(format!("conv2d: {message}"))
}

fn dims(shape: &[i32]) -> Vec<usize> {
    shape.iter().map(|&d| d as usize).collect()
}

fn int(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| contract(format!("{what} {value} is past MLX's i32")))
}

fn pair(value: [usize; 2], what: &str) -> Result<[i32; 2]> {
    Ok([int(value[0], what)?, int(value[1], what)?])
}

/// The convolution of `x` `[N, C, H, W]` with `weight` `[O, C / G, KH, KW]`, plus `bias` `[O]` if
/// given: `[N, O, HO, WO]` in the inputs' dtype ([`dash_contract::conv2d`]). Differentiable with
/// respect to `x`, the weight and the bias, by MLX's own backward pass.
///
/// Refuses, with the contract's message (`conv2d: ...`), shapes that do not fit together
/// ([`Conv2dOptions::output_shape`]), a stride, dilation or group count of 0, inputs of more than
/// one dtype, and bf16 or f16, which this implementation is not held to yet.
pub fn conv2d(
    x: &Array,
    weight: &Array,
    bias: Option<&Array>,
    options: Conv2dOptions,
) -> Result<Array> {
    let bias_len = match bias.map(Array::shape) {
        None => None,
        Some(&[o]) => Some(o as usize),
        Some(shape) => return Err(contract(format!("the bias is [O], got {shape:?}"))),
    };
    let out = options
        .output_shape(&dims(x.shape()), &dims(weight.shape()), bias_len)
        .map_err(Exception::custom)?;
    let dtype = x.dtype();
    if weight.dtype() != dtype || bias.is_some_and(|b| b.dtype() != dtype) {
        return Err(contract(format!(
            "x, the weight and the bias share one dtype, got {dtype:?}, {:?} and {:?}",
            weight.dtype(),
            bias.map(Array::dtype)
        )));
    }
    if dtype != Dtype::Float32 {
        return Err(contract(format!(
            "{dtype:?} is not implemented: only f32 is held to the contract yet"
        )));
    }
    let channels_last = x.transpose_axes(&[0, 2, 3, 1])?;
    let kernel = weight.transpose_axes(&[0, 2, 3, 1])?;
    let mut y = mlx_conv2d(
        &channels_last,
        &kernel,
        pair(options.stride, "a stride of")?,
        pair(options.padding, "a padding of")?,
        pair(options.dilation, "a dilation of")?,
        int(options.groups, "a group count of")?,
    )?;
    if let Some(bias) = bias {
        y = y.add(bias)?;
    }
    let y = y.transpose_axes(&[0, 3, 1, 2])?;
    let want: Vec<i32> = out.iter().map(|&d| d as i32).collect();
    if y.shape() != want.as_slice() {
        return Err(contract(format!(
            "MLX gave {:?} where the contract's shape is {want:?}",
            y.shape()
        )));
    }
    Ok(y)
}
