//! 2-D convolution against torch: `dash_conformance`'s `conv2d` body through dash-mlx's adapter
//! (`tests/common`), on the GPU: strides, paddings, dilations, groups, with and without a bias; the
//! output and the gradients of the input, the weight and the bias, each element within its count of
//! roundings (`dash_contract::conv2d`, Bounds); and what dash-mlx refuses.
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test conv2d -- --nocapture
#![cfg(target_os = "macos")]

mod common;

use common::{Mlx, require_ieee_f32_matmul};
use dash_conformance::conv2d;
use dash_mlx::conv2d::{Conv2dOptions, conv2d};
use dash_mlx::mlx::ops::zeros_dtype;
use dash_mlx::mlx::{Array, Dtype};

#[test]
fn matches_torch() {
    require_ieee_f32_matmul();
    let worst = conv2d::against_torch(&Mlx, "mlx").all_ran();
    for part in ["out", "grad_x", "grad_weight", "grad_bias"] {
        assert!(worst.of(part) > 0.0, "{part} compared nothing");
    }
}

fn zeros(shape: &[i32], dtype: Dtype) -> Array {
    zeros_dtype(shape, dtype).unwrap()
}

/// The refusal's message, which names the operation.
fn refusal(x: &Array, weight: &Array, bias: Option<&Array>, options: Conv2dOptions) -> String {
    conv2d(x, weight, bias, options)
        .expect_err("refused")
        .what()
        .to_string()
}

#[test]
fn what_is_outside_the_contract_is_refused() {
    let o = Conv2dOptions::default();
    let x = zeros(&[1, 2, 4, 4], Dtype::Float32);
    let w = zeros(&[2, 2, 3, 3], Dtype::Float32);
    let why = refusal(
        &zeros(&[1, 2, 4, 4], Dtype::Bfloat16),
        &zeros(&[2, 2, 3, 3], Dtype::Bfloat16),
        None,
        o,
    );
    assert!(
        why.starts_with("conv2d: Bfloat16 is not implemented"),
        "{why}"
    );
    let why = refusal(&x, &zeros(&[2, 2, 3, 3], Dtype::Float16), None, o);
    assert!(
        why.starts_with("conv2d: x, the weight and the bias share one dtype"),
        "{why}"
    );
    let why = refusal(
        &zeros(&[1, 3, 4, 4], Dtype::Float32),
        &zeros(&[2, 1, 3, 3], Dtype::Float32),
        None,
        o.with_groups(2),
    );
    assert!(why.starts_with("conv2d: 2 groups"), "{why}");
    let why = refusal(&x, &w, Some(&zeros(&[3], Dtype::Float32)), o);
    assert!(
        why.starts_with("conv2d: a bias of 3 for 2 output channels"),
        "{why}"
    );
    let why = refusal(&x, &w, Some(&zeros(&[2, 1], Dtype::Float32)), o);
    assert!(why.starts_with("conv2d: the bias is [O]"), "{why}");
}
