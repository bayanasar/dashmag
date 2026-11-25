//! conv2d's entry point refuses, on every side and before either runs, what is outside the
//! contract or what neither side takes yet, with the contract's words (`conv2d: ...`). Its values
//! and gradients are each side's own, held to torch by dash-conformance and to the gateway's bits
//! by `burn_direct.rs` and `mlx_direct.rs`.

use dash_mag::ops::{Conv2dOptions, conv2d};
use dash_mag::{DType, Tensor};

fn zeros(shape: &[usize], dtype: DType) -> Tensor {
    Tensor::zeros(shape, dtype).unwrap()
}

/// The refusal's message.
fn refusal(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, options: Conv2dOptions) -> String {
    conv2d(x, weight, bias, options).expect_err("refused").0
}

#[test]
fn what_neither_side_takes_is_refused_alike() {
    let o = Conv2dOptions::default();
    let x = zeros(&[1, 2, 4, 4], DType::F32);
    let w = zeros(&[2, 2, 3, 3], DType::F32);
    let cases = [
        (
            refusal(
                &zeros(&[1, 2, 4, 4], DType::BF16),
                &zeros(&[2, 2, 3, 3], DType::BF16),
                None,
                o,
            ),
            "conv2d: BF16 is not implemented",
        ),
        (
            refusal(&x, &zeros(&[2, 2, 3, 3], DType::F16), None, o),
            "conv2d: x, the weight and the bias share one dtype",
        ),
        (
            refusal(
                &zeros(&[1, 3, 4, 4], DType::F32),
                &zeros(&[2, 1, 3, 3], DType::F32),
                None,
                o.with_groups(2),
            ),
            "conv2d: 2 groups",
        ),
        (
            refusal(&x, &w, Some(&zeros(&[3], DType::F32)), o),
            "conv2d: a bias of 3 for 2 output channels",
        ),
        (
            refusal(&x, &w, None, o.with_stride([0, 1])),
            "conv2d: stride [0, 1]",
        ),
        (
            refusal(&zeros(&[1, 2, 2, 4], DType::F32), &w, None, o),
            "conv2d: the kernel 3 x 3",
        ),
        (
            refusal(&zeros(&[2, 4, 4], DType::F32), &w, None, o),
            "conv2d: x has shape [2, 4, 4], of rank 4 expected",
        ),
        (
            refusal(&x, &w, Some(&zeros(&[2, 1], DType::F32)), o),
            "conv2d: bias has shape [2, 1], of rank 1 expected",
        ),
        (
            refusal(
                &zeros(&[1, 2, 4, 4], DType::F64),
                &zeros(&[2, 2, 3, 3], DType::F64),
                None,
                o,
            ),
            "conv2d: x of F64",
        ),
    ];
    for (why, want) in cases {
        assert!(
            why.starts_with(want),
            "{why:?} does not start with {want:?}"
        );
    }
}

#[test]
fn an_output_has_the_contracts_shape() {
    let options = Conv2dOptions::default()
        .with_stride([2, 2])
        .with_padding([1, 1])
        .with_groups(4);
    let x = zeros(&[2, 4, 9, 9], DType::F32);
    let w = zeros(&[8, 1, 3, 3], DType::F32);
    let b = zeros(&[8], DType::F32);
    let y = conv2d(&x, &w, Some(&b), options).unwrap();
    assert_eq!(y.shape(), [2, 8, 5, 5]);
    assert_eq!(y.dtype(), DType::F32);
}
