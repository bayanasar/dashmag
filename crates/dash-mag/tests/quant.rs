//! The quantized linear through the gateway: on the Burn side the same bits as dash-ops' own, its
//! weight refused where dash-ops' is, and refused inside `vjp`.

#![cfg(not(target_os = "macos"))]

use burn::tensor::{DType as BurnDType, Device, Int, Tensor as Burn, TensorData};
use dash_mag::ops::{QuantOptions, QuantizedWeight, quantized_linear};
use dash_mag::{DType, Tensor, vjp};

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda(0)
    }
    #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
    {
        Device::rocm(0)
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        Device::flex()
    }
}

/// `in` 256, `out` 16, groups of 128: packed words, zeros all 8, f16 scales as f32 values.
fn parts() -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let words = (0..32 * 16)
        .map(|i: u32| i.wrapping_mul(2_654_435_761) as i32)
        .collect();
    let zeros = vec![0x8888_8888u32 as i32; 2 * 2];
    let scales = (0..2 * 16).map(|i| 0.001 + 0.0005 * i as f32).collect();
    (words, zeros, scales)
}

fn weight(zeros: &[i32]) -> dash_mag::Result<QuantizedWeight> {
    let (words, _, scales) = parts();
    QuantizedWeight::from_gptq(
        &Tensor::from_i32(&words, &[32, 16])?,
        &Tensor::from_i32(zeros, &[2, 2])?,
        &Tensor::from_f32(&scales, &[2, 16])?.as_dtype(DType::F16)?,
        None,
        QuantOptions::default(),
    )
}

#[test]
fn the_same_bits_as_dash_ops() {
    let (words, zeros, scales) = parts();
    let w = weight(&zeros).unwrap();
    let device = device();
    let direct = dash_ops::quant::QuantizedWeight::from_gptq(
        Burn::<2, Int>::from_data(TensorData::new(words, [32, 16]), (&device, BurnDType::I32)),
        Burn::<2, Int>::from_data(TensorData::new(zeros, [2, 2]), (&device, BurnDType::I32)),
        Burn::<2>::from_data(TensorData::new(scales, [2, 16]), (&device, BurnDType::F16)),
        None,
        QuantOptions::default(),
    )
    .unwrap();
    for rows in [1, 40] {
        let values: Vec<f32> = (0..rows * 256)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect();
        let bias: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        for dtype in [DType::F32, DType::BF16] {
            let x = Tensor::from_f32(&values, &[1, rows, 256])
                .unwrap()
                .as_dtype(dtype)
                .unwrap();
            let b = Tensor::from_f32(&bias, &[16]).unwrap();
            let got = quantized_linear(&x, &w, Some(&b)).unwrap();
            assert_eq!(got.shape(), [1, rows, 16]);
            assert_eq!(got.dtype(), dtype);
            let burn_dtype = match dtype {
                DType::F32 => BurnDType::F32,
                _ => BurnDType::BF16,
            };
            let bx = Burn::<3>::from_data(TensorData::new(values.clone(), [1, rows, 256]), &device)
                .cast(burn::tensor::FloatDType::from(burn_dtype));
            let bb = Burn::<1>::from_data(TensorData::new(bias.clone(), [16]), &device);
            let want: Vec<f32> = dash_ops::quant::quantized_linear(bx, &direct, Some(bb))
                .cast(burn::tensor::FloatDType::F32)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let got: Vec<f32> = got.as_dtype(DType::F32).unwrap().to_f32().unwrap();
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{rows} rows {dtype:?}"
            );
        }
    }
}

#[test]
fn refusals() {
    let err = weight(&[0x7777_7777u32 as i32; 4]).err().unwrap();
    assert!(format!("{err}").contains("0x77777777"), "{err}");
    let w = weight(&parts().1).unwrap();
    let x = Tensor::from_f32(&vec![0.5; 2 * 128], &[2, 128]).unwrap();
    assert!(quantized_linear(&x, &w, None).is_err(), "in disagrees");
    let x = Tensor::from_f32(&vec![0.5; 2 * 256], &[2, 256]).unwrap();
    assert!(quantized_linear(&x, &w, None).is_ok());
    let inside = vjp(
        |p: &[Tensor]| Ok(vec![quantized_linear(&p[0], &w, None)?]),
        &[x],
        &[None],
    );
    assert!(inside.is_err(), "refused inside vjp");
}
