//! On CUDA, a device that would run f32 matmuls in TF32 refuses the first tensor, unless
//! `CUBECL_CUDA_TF32=1` says that is wanted. The variable is read once per process, so each case
//! runs in a process of its own: this test again, told which by `DASH_MAG_TF32_CASE`.

#![cfg(feature = "cuda")]

use std::process::Command;

use dash_mag::{DType, Tensor};

const TEST: &str = "a_device_in_tf32_refuses_the_first_tensor_unless_asked_for";

#[test]
fn a_device_in_tf32_refuses_the_first_tensor_unless_asked_for() {
    if let Ok(case) = std::env::var("DASH_MAG_TF32_CASE") {
        let first = Tensor::zeros(&[2], DType::F32);
        match case.as_str() {
            "refused" => {
                // A card before compute capability 8.0 has no TF32 to run in.
                if dash_ops::f32_products_in_tf32(&burn::tensor::Device::cuda(0)) {
                    let error = first.expect_err("refused");
                    assert!(error.0.contains("CUBECL_CUDA_TF32"), "{error}");
                    assert!(Tensor::from_i32(&[1], &[1]).is_err(), "from_i32");
                    assert!(Tensor::from_bool(&[true], &[1]).is_err(), "from_bool");
                    assert!(Tensor::from_f32(&[1.0], &[1]).is_err(), "from_bytes");
                }
            }
            _ => assert_eq!(first.unwrap().shape(), [2]),
        }
        return;
    }
    for (case, tf32) in [
        ("refused", None),
        ("accepted", Some("0")),
        ("accepted", Some("1")),
    ] {
        let mut child = Command::new(std::env::current_exe().unwrap());
        child
            .args(["--exact", TEST])
            .env("DASH_MAG_TF32_CASE", case);
        match tf32 {
            Some(value) => child.env("CUBECL_CUDA_TF32", value),
            None => child.env_remove("CUBECL_CUDA_TF32"),
        };
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "{case} with {tf32:?}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}
