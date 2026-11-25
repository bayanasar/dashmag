//! 2-D convolution against torch: `dash_conformance::conv2d`'s body (strides, paddings, dilations,
//! groups, with and without a bias; the output and the gradients of the input, the weight and the
//! bias, each element within its count of roundings) through the Burn adapter, on Flex and on a
//! CubeCL device, any runtime (`DASH_OPS_DEVICE`); and what dash-ops refuses.
//!
//!   cargo test --release -p dash-ops --test conv2d -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test conv2d
//!   cargo test --release -p dash-ops --features rocm --test conv2d

mod common;

use burn::tensor::{DType, Device, Tensor, TensorData};
use common::adapter::Burn;
use dash_conformance::conv2d;
use dash_ops::conv2d::{Conv2dOptions, conv2d};

/// Every case on `device`, alone and with gradients, each element within its roundings.
fn against_torch(device: Device, label: &str) {
    let worst = conv2d::against_torch(&Burn::new(device), label).all_ran();
    for part in ["out", "grad_x", "grad_weight", "grad_bias"] {
        assert!(worst.of(part) > 0.0, "{label}: {part} compared nothing");
    }
}

#[test]
fn matches_torch_on_flex() {
    against_torch(Device::flex(), "flex");
}

#[cfg(feature = "cubecl")]
#[test]
fn matches_torch_on_the_cubecl_device() {
    let device = common::cube_device();
    let label = common::runtime_of(&device).unwrap_or("cubecl");
    against_torch(device, label);
}

fn zeros(shape: [usize; 4], dtype: DType) -> Tensor<4> {
    Tensor::<4>::zeros(shape, &Device::flex()).cast(dtype)
}

#[test]
#[should_panic(expected = "conv2d: BF16 is not implemented")]
fn bf16_is_refused_until_it_is_held() {
    conv2d(
        zeros([1, 2, 4, 4], DType::BF16),
        zeros([2, 2, 3, 3], DType::BF16),
        None,
        Conv2dOptions::default(),
    );
}

#[test]
#[should_panic(expected = "conv2d: x, the weight and the bias share one dtype")]
fn mixed_dtypes_are_refused() {
    conv2d(
        zeros([1, 2, 4, 4], DType::F32),
        zeros([2, 2, 3, 3], DType::F16),
        None,
        Conv2dOptions::default(),
    );
}

#[test]
#[should_panic(expected = "conv2d: 2 groups")]
fn channels_that_do_not_split_into_the_groups_are_refused() {
    conv2d(
        zeros([1, 3, 4, 4], DType::F32),
        zeros([2, 1, 3, 3], DType::F32),
        None,
        Conv2dOptions::default().with_groups(2),
    );
}

#[test]
#[should_panic(expected = "conv2d: a bias of 3 for 2 output channels")]
fn a_bias_of_the_wrong_length_is_refused() {
    let bias = Tensor::<1>::from_data(TensorData::new(vec![0f32; 3], [3]), &Device::flex());
    conv2d(
        zeros([1, 2, 4, 4], DType::F32),
        zeros([2, 2, 3, 3], DType::F32),
        Some(bias),
        Conv2dOptions::default(),
    );
}
