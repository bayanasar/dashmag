//! The quantized linear through the gateway. On the Burn side, the same bits as dash-ops' own. On
//! MLX, against its definition computed on the host in f64, held as dash-ops' test holds the Burn
//! side: the largest error over the output's largest value. On both, the weight refused where
//! dash-ops' is and a call refused inside `vjp`; on MLX, groups other than 128 refused.

#[cfg(not(target_os = "macos"))]
use burn::tensor::{DType as BurnDType, Device, Int, Tensor as Burn, TensorData};
use dash_mag::ops::{QuantOptions, QuantizedWeight, quantized_linear};
use dash_mag::{DType, Tensor, vjp};

#[cfg(not(target_os = "macos"))]
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

#[cfg(not(target_os = "macos"))]
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
        std::slice::from_ref(&x),
        &[None],
    );
    assert!(inside.is_err(), "refused inside vjp");
    let constant = vjp(
        |_: &[Tensor]| Ok(vec![quantized_linear(&x, &w, None)?]),
        std::slice::from_ref(&x),
        &[None],
    );
    assert!(
        constant.is_err(),
        "refused inside vjp on a captured constant too"
    );
    assert!(quantized_linear(&x, &w, None).is_ok(), "and taken after it");
}

/// `W` `[in, out]` in f64 by the definition, `(q - 8) · scale`, the scales as f16 holds them.
#[cfg(target_os = "macos")]
fn definition() -> Vec<f64> {
    let (words, _, scales) = parts();
    let scales = Tensor::from_f32(&scales, &[2, 16])
        .and_then(|s| s.as_dtype(DType::F16)?.as_dtype(DType::F32)?.to_f32())
        .unwrap();
    let mut w = vec![0f64; 256 * 16];
    for i in 0..256 {
        for o in 0..16 {
            let q = (words[i / 8 * 16 + o] as u32 >> (4 * (i % 8))) & 15;
            w[i * 16 + o] = (q as f64 - 8.0) * scales[i / 128 * 16 + o] as f64;
        }
    }
    w
}

/// `n` multiples of 2^-23 in `[-1, 1)`, from a hash of `seed`.
#[cfg(target_os = "macos")]
fn spread(n: usize, seed: u32) -> Vec<f32> {
    (0..n as u32)
        .map(|i| {
            let h = (i ^ seed.wrapping_mul(0x9e37_79b9)).wrapping_mul(1_103_515_245);
            (h >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

#[cfg(target_os = "macos")]
#[test]
fn the_definition_on_mlx() {
    let w = weight(&parts().1).unwrap();
    let exact_w = definition();
    let bias = spread(16, 1);
    let b = Tensor::from_f32(&bias, &[16]).unwrap();
    for rows in [1, 40] {
        let values = spread(rows * 256, 2);
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            for with_bias in [true, false] {
                let x = Tensor::from_f32(&values, &[1, rows, 256])
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                // The definition reads x as given, in its dtype.
                let xs = x.as_dtype(DType::F32).unwrap().to_f32().unwrap();
                let y = quantized_linear(&x, &w, with_bias.then_some(&b)).unwrap();
                assert_eq!(y.shape(), [1, rows, 16]);
                assert_eq!(y.dtype(), dtype);
                let got = y.as_dtype(DType::F32).unwrap().to_f32().unwrap();
                let (mut worst, mut scale) = (0f64, 0f64);
                for n in 0..rows {
                    for o in 0..16 {
                        let exact = (0..256)
                            .map(|i| xs[n * 256 + i] as f64 * exact_w[i * 16 + o])
                            .sum::<f64>()
                            + if with_bias { bias[o] as f64 } else { 0.0 };
                        worst = worst.max((got[n * 16 + o] as f64 - exact).abs());
                        scale = scale.max(exact.abs());
                    }
                }
                println!(
                    "MLX {dtype:?} {rows} rows, bias {with_bias}: max abs {worst:.3e}, relative {:.3e}",
                    worst / scale
                );
                // Relative to the output's largest value, just above the largest measured on an
                // Apple M6 with MLX 0.32.4 (2026-11-14), each at 40 rows: the product's rounding
                // to `x`'s dtype and the sum's with a bias; past MLX's vector path (at most 33
                // rows here), `W`'s rounding to `x`'s dtype; from bf16 the scales' rounding too.
                let bound = match dtype {
                    DType::F32 => 2e-7,    // 1.92e-7
                    DType::BF16 => 3.5e-3, // 3.44e-3
                    _ => 4.4e-4,           // 4.36e-4
                };
                assert!(
                    worst / scale <= bound,
                    "{dtype:?} {rows} rows, bias {with_bias}: {:.3e}",
                    worst / scale
                );
            }
        }
    }
    // An f64 bias is added in f32, as the definition adds it: the same bits as the bias in f32.
    let x = Tensor::from_f32(&spread(256, 3), &[1, 256]).unwrap();
    let y = quantized_linear(&x, &w, Some(&b))
        .unwrap()
        .to_f32()
        .unwrap();
    let wide = b.as_dtype(DType::F64).unwrap();
    let y64 = quantized_linear(&x, &w, Some(&wide))
        .unwrap()
        .to_f32()
        .unwrap();
    assert!(y.iter().zip(&y64).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[cfg(target_os = "macos")]
#[test]
fn groups_other_than_128_are_refused_on_mlx() {
    let qweight = Tensor::from_i32(&parts().0, &[32, 16]).unwrap();
    for group in [64, 256] {
        let groups = 256 / group;
        let zeros = Tensor::from_i32(&vec![0x8888_8888u32 as i32; groups * 2], &[groups, 2]);
        let scales = Tensor::from_f32(&vec![0.01; groups * 16], &[groups, 16])
            .and_then(|s| s.as_dtype(DType::F16));
        let err = QuantizedWeight::from_gptq(
            &qweight,
            &zeros.unwrap(),
            &scales.unwrap(),
            None,
            QuantOptions::default().with_group(group),
        )
        .err()
        .unwrap();
        assert!(
            format!("{err}").contains(&format!("groups of 128, not {group}")),
            "{err}"
        );
    }
}
