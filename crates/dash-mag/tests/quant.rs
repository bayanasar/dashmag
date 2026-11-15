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

/// `W` `[in, out]` in f64 by the definition, `(q - 8) · scale`, the scales as f16 holds them; and
/// beside it `(q + 8) · scale`, which [`the_definition_on_mlx`]'s bound sums over.
#[cfg(target_os = "macos")]
fn definition() -> (Vec<f64>, Vec<f64>) {
    let (words, _, scales) = parts();
    let scales = Tensor::from_f32(&scales, &[2, 16])
        .and_then(|s| s.as_dtype(DType::F16)?.as_dtype(DType::F32)?.to_f32())
        .unwrap();
    let mut w = vec![0f64; 256 * 16];
    let mut span = vec![0f64; 256 * 16];
    for i in 0..256 {
        for o in 0..16 {
            let q = (words[i / 8 * 16 + o] as u32 >> (4 * (i % 8))) & 15;
            let scale = scales[i / 128 * 16 + o] as f64;
            w[i * 16 + o] = (q as f64 - 8.0) * scale;
            span[i * 16 + o] = (q as f64 + 8.0) * scale;
        }
    }
    (w, span)
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
    let (exact_w, span) = definition();
    let bias = spread(16, 1);
    let b = Tensor::from_f32(&bias, &[16]).unwrap();
    // The bound on each output's error, from what MLX's path may round (each rounding one that
    // `dash_contract::quant` allows), with `p = sum_i x_i W_i` and `y = p (+ bias)` exact, `u` f32's
    // unit roundoff (2^-24) and `ux` that of `x`'s dtype (half a unit: 2^-24 in f32, 2^-8 in bf16,
    // 2^-11 in f16):
    //
    // - The scales rounded to `x`'s dtype: exact from f16 to f16 or f32; to bf16 within 2^-8 of
    //   each. One scale's rounding is one factor on its group's part of `p`, so the term is
    //   `us sum_g |sum_{i in g} x_i W_i|`, `us` 2^-8 from bf16 and 0 otherwise, and each
    //   `W'_i = (q_i - 8) s'` is within `us |W_i|`.
    // - On MLX's matrix path only, `W'` rounded to `x`'s dtype: MLX forms `s' q + b'` in f32,
    //   where it is exact, and rounds it once, within `uw |W'_i| <= uw (1 + us) |W_i|` (no weight
    //   here is subnormal), `uw = ux` in bf16 and f16 and 0 in f32. Which path runs depends on
    //   the row count, the shape and the GPU, so the term is kept at every row count.
    // - The sums in f32 over `in`, in any order: each term goes through at most `in + 2`
    //   roundings (its product, `in - 1` additions and, on the vector path, which sums `x q` and
    //   `x` apart, `scale · sum` and `bias · sum` and their sum), so they are within
    //   `γ(in + 2) sum_i |x_i| (q_i + 8) |s'|`, `γ(n) = n u / (1 - n u)`, `|s'| <= (1 + us) scale`.
    //   `(q + 8) |s'|` bounds both `|W''|` on the matrix path and what the vector path's two sums
    //   take from each `x_i`.
    //
    // `e`, the sum of those, bounds the product in f32 against `p`. Rounded to `x`'s dtype, it is
    // within `d = e + ux (|p| + e) + η`, `η` half the dtype's subnormal spacing; with a bias,
    // added in f32 and rounded again, within `d + (u + ux (1 + u)) (|y| + d) + η`. The test holds
    // the largest error to the largest of those bounds, both over the output's largest value.
    let u = 2f64.powi(-24);
    let gamma = 258.0 * u / (1.0 - 258.0 * u);
    // 1, 8 and 9 rows on the vector path, 32 its last and 33 the matrix path's first on an Apple
    // M6 with MLX 0.32.4 for this shape (its `get_qmv_batch_limit`), 40 on the matrix path.
    for rows in [1, 8, 9, 32, 33, 40] {
        let values = spread(rows * 256, 2);
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let (ux, us, uw, eta) = match dtype {
                DType::F32 => (u, 0.0, 0.0, 2f64.powi(-150)),
                DType::BF16 => (2f64.powi(-8), 2f64.powi(-8), 2f64.powi(-8), 2f64.powi(-134)),
                _ => (2f64.powi(-11), 0.0, 2f64.powi(-11), 2f64.powi(-25)),
            };
            let x = Tensor::from_f32(&values, &[1, rows, 256])
                .unwrap()
                .as_dtype(dtype)
                .unwrap();
            // The definition reads x as given, in its dtype.
            let xs = x.as_dtype(DType::F32).unwrap().to_f32().unwrap();
            let plain = quantized_linear(&x, &w, None)
                .and_then(|y| y.as_dtype(DType::F32)?.to_f32())
                .unwrap();
            for with_bias in [true, false] {
                let y = quantized_linear(&x, &w, with_bias.then_some(&b)).unwrap();
                assert_eq!(y.shape(), [1, rows, 16]);
                assert_eq!(y.dtype(), dtype);
                let got = y.as_dtype(DType::F32).unwrap().to_f32().unwrap();
                if with_bias {
                    // The bias's part, which the bound has room to hide, held exactly: added in
                    // f32 to the product rounded to `x`'s dtype, and the sum rounded once.
                    let sums: Vec<f32> = (0..rows * 16).map(|k| plain[k] + bias[k % 16]).collect();
                    let want = Tensor::from_f32(&sums, &[rows * 16])
                        .and_then(|s| s.as_dtype(dtype)?.as_dtype(DType::F32)?.to_f32())
                        .unwrap();
                    assert!(
                        got.iter()
                            .zip(&want)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{dtype:?} {rows} rows: the bias not added in f32 to the rounded product"
                    );
                }
                let (mut worst, mut scale, mut allowed) = (0f64, 0f64, 0f64);
                for n in 0..rows {
                    for o in 0..16 {
                        let (mut groups, mut xw, mut xspan) = ([0f64; 2], 0f64, 0f64);
                        for i in 0..256 {
                            let x = xs[n * 256 + i] as f64;
                            groups[i / 128] += x * exact_w[i * 16 + o];
                            xw += (x * exact_w[i * 16 + o]).abs();
                            xspan += x.abs() * span[i * 16 + o];
                        }
                        let p = groups[0] + groups[1];
                        let exact = p + if with_bias { bias[o] as f64 } else { 0.0 };
                        let e = us * (groups[0].abs() + groups[1].abs())
                            + uw * (1.0 + us) * xw
                            + gamma * (1.0 + us) * xspan;
                        let d = e + ux * (p.abs() + e) + eta;
                        let bound = if with_bias {
                            d + (u + ux * (1.0 + u)) * (exact.abs() + d) + eta
                        } else {
                            d
                        };
                        worst = worst.max((got[n * 16 + o] as f64 - exact).abs());
                        scale = scale.max(exact.abs());
                        allowed = allowed.max(bound);
                    }
                }
                println!(
                    "MLX {dtype:?} {rows} rows, bias {with_bias}: max abs {worst:.3e}, relative {:.3e}, bound {:.3e}",
                    worst / scale,
                    allowed / scale
                );
                // The largest measured on an Apple M6 with MLX 0.32.4 (2026-11-14), each at 40
                // rows, were 1.92e-7 in f32, 3.44e-3 from bf16 and 4.36e-4 from f16; over these
                // row counts they are the same (2026-11-15 UTC), f32's at 33 and 40 rows, bf16's
                // at 32, 33 and 40, f16's at 40. The bounds are larger: the f32 sums' term, for
                // any order, is most of f32's, and `W`'s rounding most of bf16's and f16's.
                assert!(
                    worst <= allowed,
                    "{dtype:?} {rows} rows, bias {with_bias}: {:.3e} over the bound {:.3e}",
                    worst / scale,
                    allowed / scale
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
