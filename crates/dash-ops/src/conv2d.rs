//! 2-D convolution of images, torch's `F.conv2d`: any kernel size, stride, symmetric zero padding
//! and dilation, channels in groups, and an optional bias; with the gradients of the input, the
//! weight and the bias.
//!
//! [`dash_contract::conv2d`] defines the operation, which every implementation of it shares: what
//! [`conv2d`] computes, its NCHW and OIHW layouts, its dtypes and options, and the bound in
//! roundings each element is held to. This module implements it on Burn.
//!
//! # Burn's own
//!
//! Burn has the convolution: [`conv2d`] checks the call against the contract and runs Burn's
//! `conv2d` on the tensors' device, and under `Autodiff` Burn records it as one operation with its
//! own backward pass (the transposed convolution for the input, the correlation for the weight).
//! So it is not a backend extension, as this crate's other operations are, and has no kernels or
//! reference of its own: Flex runs its tiled im2col and matrix product (with paths of its own for
//! 1 x 1 and depthwise kernels), the CubeCL runtimes Burn's convolution kernels. f32 on CUDA wants
//! `CUBECL_CUDA_TF32=0`, as every f32 product here does.

use burn::tensor::module::conv2d as burn_conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{DType, Tensor};

pub use dash_contract::conv2d::Conv2dOptions;

/// The convolution of `x` `[N, C, H, W]` with `weight` `[O, C / G, KH, KW]`, plus `bias` `[O]` if
/// given: `[N, O, HO, WO]` in the inputs' dtype ([`dash_contract::conv2d`]). Differentiable with
/// respect to `x`, the weight and the bias, by Burn's own backward pass.
///
/// # Panics
///
/// Outside the contract, with the contract's message (`conv2d: ...`): shapes that do not fit
/// together ([`Conv2dOptions::output_shape`]), a stride, dilation or group count of 0; inputs of
/// more than one dtype; and bf16 or f16, which this implementation is not held to yet.
pub fn conv2d(
    x: Tensor<4>,
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    options: Conv2dOptions,
) -> Tensor<4> {
    let shape = options
        .output_shape(
            &x.dims(),
            &weight.dims(),
            bias.as_ref().map(|b| b.dims()[0]),
        )
        .unwrap_or_else(|why| panic!("{why}"));
    let dtype = x.dtype();
    let dtypes_agree = weight.dtype() == dtype && bias.as_ref().is_none_or(|b| b.dtype() == dtype);
    assert!(
        dtypes_agree,
        "conv2d: x, the weight and the bias share one dtype, got {dtype:?}, {:?} and {:?}",
        weight.dtype(),
        bias.as_ref().map(Tensor::dtype)
    );
    assert!(
        dtype == DType::F32,
        "conv2d: {dtype:?} is not implemented: only f32 is held to the contract yet"
    );
    let y = burn_conv2d(
        x,
        weight,
        bias,
        ConvOptions::new(
            options.stride,
            options.padding,
            options.dilation,
            options.groups,
        ),
    );
    debug_assert_eq!(
        y.dims(),
        shape,
        "Burn's output shape against the contract's"
    );
    y
}
