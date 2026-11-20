//! The quantized linear through the gateway. On the Burn side, the same bits as dash-ops' own,
//! forward and, inside `vjp`, backward. On MLX, against its definition computed on the host in
//! f64, forward and, inside `vjp`, backward, each element held, as dash-ops' tests hold the Burn
//! side's, to what its path's roundings allow. On both, the weight refused where dash-ops' is; on
//! MLX, groups other than 128.

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

/// `in` 256, `out` outputs, groups of 128: packed words, zeros all 8, f16 scales as f32 values.
fn parts_of(out: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let words = (0..32 * out as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) as i32)
        .collect();
    let zeros = vec![0x8888_8888u32 as i32; 2 * out / 8];
    let scales = (0..2 * out).map(|i| 0.001 + 0.0005 * i as f32).collect();
    (words, zeros, scales)
}

/// [`parts_of`] 16 outputs.
fn parts() -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    parts_of(16)
}

fn weight(zeros: &[i32]) -> dash_mag::Result<QuantizedWeight> {
    weight_of(16, zeros)
}

fn weight_of(out: usize, zeros: &[i32]) -> dash_mag::Result<QuantizedWeight> {
    let (words, _, scales) = parts_of(out);
    QuantizedWeight::from_gptq(
        &Tensor::from_i32(&words, &[32, out])?,
        &Tensor::from_i32(zeros, &[2, out / 8])?,
        &Tensor::from_f32(&scales, &[2, out])?.as_dtype(DType::F16)?,
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
}

/// Inside `vjp` on the Burn side: `x`'s and the bias's gradients the same bits as dash-ops' own
/// backward under Burn's autodiff, and a captured constant `x`, as a frozen teacher's, the
/// forward's bits.
#[cfg(not(target_os = "macos"))]
#[test]
fn inside_vjp_the_same_bits_as_dash_ops() {
    let (words, zeros, scales) = parts();
    let w = weight(&zeros).unwrap();
    let device = device().autodiff();
    let direct = dash_ops::quant::QuantizedWeight::from_gptq(
        Burn::<2, Int>::from_data(TensorData::new(words, [32, 16]), (&device, BurnDType::I32)),
        Burn::<2, Int>::from_data(TensorData::new(zeros, [2, 2]), (&device, BurnDType::I32)),
        Burn::<2>::from_data(TensorData::new(scales, [2, 16]), (&device, BurnDType::F16)),
        None,
        QuantOptions::default(),
    )
    .unwrap();
    let bits = |t: Burn<3>| -> Vec<u32> {
        t.cast(burn::tensor::FloatDType::F32)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let gateway_bits = |t: &Tensor| -> Vec<u32> {
        t.as_dtype(DType::F32)
            .unwrap()
            .to_f32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    for rows in [1, 40] {
        let values: Vec<f32> = (0..rows * 256)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect();
        let cotangent: Vec<f32> = (0..rows * 16)
            .map(|i| ((i % 13) as f32 - 6.0) / 4.0)
            .collect();
        let bias: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        for (dtype, burn_dtype) in [(DType::F32, BurnDType::F32), (DType::BF16, BurnDType::BF16)] {
            let as_dtype = |t: Tensor| t.as_dtype(dtype).unwrap();
            let x = as_dtype(Tensor::from_f32(&values, &[1, rows, 256]).unwrap());
            let dy = as_dtype(Tensor::from_f32(&cotangent, &[1, rows, 16]).unwrap());
            let b = Tensor::from_f32(&bias, &[16]).unwrap();
            let (outs, grads) = vjp(
                |p: &[Tensor]| Ok(vec![quantized_linear(&p[0], &w, Some(&p[1]))?]),
                &[x.clone(), b.clone()],
                &[Some(dy)],
            )
            .unwrap();
            assert_eq!((grads[0].dtype(), grads[1].dtype()), (dtype, DType::F32));

            let cast = burn::tensor::FloatDType::from(burn_dtype);
            let bx = Burn::<3>::from_data(TensorData::new(values.clone(), [1, rows, 256]), &device)
                .cast(cast)
                .require_grad();
            let bb =
                Burn::<1>::from_data(TensorData::new(bias.clone(), [16]), &device).require_grad();
            let bdy =
                Burn::<3>::from_data(TensorData::new(cotangent.clone(), [1, rows, 16]), &device)
                    .cast(cast);
            let y = dash_ops::quant::quantized_linear(bx.clone(), &direct, Some(bb.clone()));
            let g = (y.clone().cast(burn::tensor::FloatDType::F32)
                * bdy.cast(burn::tensor::FloatDType::F32))
            .sum()
            .backward();
            assert_eq!(gateway_bits(&outs[0]), bits(y), "{rows} rows {dtype:?} y");
            assert_eq!(
                gateway_bits(&grads[0]),
                bits(bx.grad(&g).unwrap()),
                "{rows} rows {dtype:?} dx"
            );
            assert_eq!(
                gateway_bits(&grads[1]),
                bits(bb.grad(&g).unwrap().unsqueeze::<3>()),
                "{rows} rows {dtype:?} dbias"
            );

            // A captured constant: the forward's bits.
            let (constant, _) = vjp(
                |p: &[Tensor]| {
                    Ok(vec![
                        quantized_linear(&x, &w, None)?.sum()?.add(&p[0].sum()?)?,
                    ])
                },
                std::slice::from_ref(&b),
                &[None],
            )
            .unwrap();
            let plain = quantized_linear(&x, &w, None).unwrap().sum().unwrap();
            assert_eq!(
                gateway_bits(&constant[0]),
                gateway_bits(&plain.add(&b.sum().unwrap()).unwrap())
            );
        }
    }
}

/// `W` `[in, out]` in f64 by the definition, `(q - 8) · scale`, the scales as f16 holds them; and
/// beside it `(q + 8) · scale`, which [`the_definition_on_mlx`]'s bound sums over.
#[cfg(target_os = "macos")]
fn definition() -> (Vec<f64>, Vec<f64>) {
    definition_of(16)
}

/// [`definition`] for [`parts_of`] `out` outputs.
#[cfg(target_os = "macos")]
fn definition_of(out: usize) -> (Vec<f64>, Vec<f64>) {
    let (words, _, scales) = parts_of(out);
    let scales = Tensor::from_f32(&scales, &[2, out])
        .and_then(|s| s.as_dtype(DType::F16)?.as_dtype(DType::F32)?.to_f32())
        .unwrap();
    let mut w = vec![0f64; 256 * out];
    let mut span = vec![0f64; 256 * out];
    for i in 0..256 {
        for o in 0..out {
            let q = (words[i / 8 * out + o] as u32 >> (4 * (i % 8))) & 15;
            let scale = scales[i / 128 * out + o] as f64;
            w[i * out + o] = (q as f64 - 8.0) * scale;
            span[i * out + o] = (q as f64 + 8.0) * scale;
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
    // each output to its own bound, from its own terms.
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
                let (mut worst, mut scale, mut share) = (0f64, 0f64, 0f64);
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
                        let err = (got[n * 16 + o] as f64 - exact).abs();
                        assert!(
                            err <= bound,
                            "{dtype:?} {rows} rows, bias {with_bias}, output ({n}, {o}): off by \
                             {err:.3e}, allowed {bound:.3e}"
                        );
                        worst = worst.max(err);
                        scale = scale.max(exact.abs());
                        share = share.max(err / bound);
                    }
                }
                // The largest measured on an Apple M6 with MLX 0.32.4 (2026-11-14), each at 40
                // rows, were 1.92e-7 in f32, 3.44e-3 from bf16 and 4.36e-4 from f16; over these
                // row counts they are the same (2026-11-15 UTC), f32's at 33 and 40 rows, bf16's
                // at 32, 33 and 40, f16's at 40. The bounds are larger: the f32 sums' term, for
                // any order, is most of f32's, and `W`'s rounding most of bf16's and f16's. No
                // output used more of its own bound than 1.52e-3 in f32 (at 40 rows), 0.272 from
                // bf16 (at 33 and 40) and 0.225 from f16 (at 40), each with the bias (2026-11-15
                // UTC).
                println!(
                    "MLX {dtype:?} {rows} rows, bias {with_bias}: max abs {worst:.3e}, relative \
                     {:.3e}, at most {share:.3e} of what its roundings allow",
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

/// Inside `vjp` on MLX: `x`'s and the bias's gradients against the definition's backward
/// (`dash_contract::quant`), `dy Wᵀ` and the sum of `dy` over the rows, computed on the host in
/// f64, each element held to what its path's roundings allow; the bias in f32 beside `x` in another
/// dtype at 9 rows, as dash-ops' test has it; beside a frozen bias, `x`'s gradient the same bits;
/// rows on a batch axis, the same bits as one matrix of them; the forward inside `vjp`, and on a
/// captured constant `x`, the forward's bits.
#[cfg(target_os = "macos")]
#[test]
fn the_backward_on_mlx() {
    // The bound on each element's error, from what its path may round (each rounding one that
    // `dash_contract::quant` allows), with `p` its value by the definition, `S` the sum of its
    // terms' magnitudes (`|dy_o W_io|` over the outputs for `dx`, `|dy_n|` over the rows for the
    // bias), `u` f32's unit roundoff (2^-24) and `ux` that of the gradient's dtype:
    //
    // - `x`'s gradient in f32 at any row count, and below 4 rows in bf16 and f16, where dash-mlx
    //   takes it in f32 rather than through MLX's vector path: `dy` read exactly into f32 and `W`
    //   exact in it (the scales as f16 holds them), the products and their sum in f32 in any order.
    //   Each term goes through at most `out` roundings (its product and `out - 1` additions), so
    //   the sum is within `γ(out) S`, `γ(n) = n u / (1 - n u)`.
    // - From 4 rows on in bf16 and f16, MLX's own backward, its matrix path: the scales rounded to
    //   `x`'s dtype (exact from f16 to f16; to bf16 within 2^-8 of each, and each term has a scale
    //   of its own, so `us S`, `us` 2^-8 from bf16 and 0 from f16), and `W'` rounded to that dtype,
    //   as the forward's matrix path rounds it (`uw = ux`, within `uw (1 + us) |W|`). The products
    //   of two values in that dtype are exact in f32 and summed in it, `γ(out) (1 + us) (1 + uw) S`.
    // - The bias's gradient: `dy` summed over the rows in f32, `γ(rows) S`.
    //
    // `e`, the sum of a path's terms, bounds the value in f32 against `p`; rounded to the
    // gradient's dtype, it is within `e + ux (|p| + e) + η`, `η` half the dtype's subnormal
    // spacing. The test holds each element to its own bound, from its own terms.
    let u = 2f64.powi(-24);
    let gamma = |n: usize| n as f64 * u / (1.0 - n as f64 * u);
    let units = |dtype: DType| match dtype {
        DType::F32 => (u, 2f64.powi(-150)),
        DType::BF16 => (2f64.powi(-8), 2f64.powi(-134)),
        _ => (2f64.powi(-11), 2f64.powi(-25)),
    };
    let bits = |t: &Tensor| -> Vec<u32> {
        t.as_dtype(DType::F32)
            .and_then(|t| t.to_f32())
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let values = |t: &Tensor| t.as_dtype(DType::F32).and_then(|t| t.to_f32()).unwrap();
    // From 1024 outputs MLX's vector path splits its sum (below 4 rows), and with both sides
    // multiples of 64 its matrix path runs on the neural accelerators of GPUs that have them (from
    // 4 rows, in bf16 and f16).
    for out in [16, 1024] {
        let w = weight_of(out, &parts_of(out).1).unwrap();
        let (exact_w, _) = definition_of(out);
        for rows in [1, 3, 4, 9, 100] {
            let x = Tensor::from_f32(&spread(rows * 256, 4), &[1, rows, 256]).unwrap();
            let dy = Tensor::from_f32(&spread(rows * out, 5), &[1, rows, out]).unwrap();
            let b = Tensor::from_f32(&spread(out, 6), &[out]).unwrap();
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                let bias_dtype = if rows == 9 { DType::F32 } else { dtype };
                let x = x.as_dtype(dtype).unwrap();
                let dy = dy.as_dtype(dtype).unwrap();
                let b = b.as_dtype(bias_dtype).unwrap();
                let at = format!("{out} outputs, {dtype:?} {rows} rows");
                let (outs, grads) = vjp(
                    |p: &[Tensor]| Ok(vec![quantized_linear(&p[0], &w, Some(&p[1]))?]),
                    &[x.clone(), b.clone()],
                    &[Some(dy.clone())],
                )
                .unwrap();
                let plain = quantized_linear(&x, &w, Some(&b)).unwrap();
                assert!(
                    bits(&outs[0]) == bits(&plain),
                    "{at}: the forward inside vjp"
                );
                assert_eq!((grads[0].dtype(), grads[1].dtype()), (dtype, bias_dtype));
                assert_eq!((grads[0].shape(), grads[1].shape()), (x.shape(), b.shape()));
                // A frozen bias beside a trained `x`, LoRA's usual case: `x`'s gradient alone, the
                // same bits.
                let (_, alone) = vjp(
                    |p: &[Tensor]| Ok(vec![quantized_linear(&p[0], &w, Some(&b))?]),
                    std::slice::from_ref(&x),
                    &[Some(dy.clone())],
                )
                .unwrap();
                assert_eq!(alone.len(), 1);
                assert!(
                    bits(&alone[0]) == bits(&grads[0]),
                    "{at}: x's gradient beside a frozen bias"
                );

                // The definition reads `dy` as given, in its dtype.
                let (dys, dx, db) = (values(&dy), values(&grads[0]), values(&grads[1]));
                let (ux, eta) = units(dtype);
                let (us, uw) = match dtype {
                    DType::BF16 if rows >= 4 => (2f64.powi(-8), ux),
                    DType::F16 if rows >= 4 => (0.0, ux),
                    _ => (0.0, 0.0),
                };
                let (mut worst_x, mut scale_x, mut share_x) = (0f64, 0f64, 0f64);
                for n in 0..rows {
                    for i in 0..256 {
                        let (mut p, mut s) = (0f64, 0f64);
                        for o in 0..out {
                            let term = dys[n * out + o] as f64 * exact_w[i * out + o];
                            p += term;
                            s += term.abs();
                        }
                        let e =
                            us * s + uw * (1.0 + us) * s + gamma(out) * (1.0 + us) * (1.0 + uw) * s;
                        let bound = e + ux * (p.abs() + e) + eta;
                        let err = (dx[n * 256 + i] as f64 - p).abs();
                        assert!(
                            err <= bound,
                            "{at}, dx ({n}, {i}): off by {err:.3e}, allowed {bound:.3e}"
                        );
                        worst_x = worst_x.max(err);
                        scale_x = scale_x.max(p.abs());
                        share_x = share_x.max(err / bound);
                    }
                }
                let (ub, eta_b) = units(bias_dtype);
                let (mut worst_b, mut share_b) = (0f64, 0f64);
                for o in 0..out {
                    let terms = (0..rows).map(|n| dys[n * out + o] as f64);
                    let p = terms.clone().sum::<f64>();
                    let s = terms.map(f64::abs).sum::<f64>();
                    let e = gamma(rows) * s;
                    let bound = e + ub * (p.abs() + e) + eta_b;
                    let err = (db[o] as f64 - p).abs();
                    assert!(
                        err <= bound,
                        "{at}, dbias {o}: off by {err:.3e}, allowed {bound:.3e}"
                    );
                    worst_b = worst_b.max(err);
                    share_b = share_b.max(err / bound);
                }
                // On an Apple M6 with MLX 0.32.4 (2026-11-20 UTC, three runs, the same lines), no
                // element of `x`'s gradient used more of its own bound than 0.135 in f32 (16
                // outputs, 100 rows); below 4 rows 0.952 from bf16 and 0.928 from f16 (16 outputs,
                // 3 rows), and on MLX's matrix path 0.380 from bf16 and 0.573 from f16 (16
                // outputs, 100 rows). The bias's gradient, a sum in f32 rounded once, up to 0.996
                // (bf16, 1024 outputs, 3 rows).
                println!(
                    "MLX {at}, backward: dx max abs {worst_x:.3e}, relative {:.3e}, at most \
                     {share_x:.3e} of what its roundings allow; dbias ({bias_dtype:?}) \
                     {worst_b:.3e}, {share_b:.3e}",
                    worst_x / scale_x
                );
            }
        }
    }
    // Rows on a batch axis, fewer than 4 to a matrix but 5 in all: `x`'s gradient by MLX's matrix
    // path over all of them, the same bits as from one matrix of them.
    let w = weight_of(1024, &parts_of(1024).1).unwrap();
    let x = Tensor::from_f32(&spread(5 * 256, 9), &[5, 256]).unwrap();
    let dy = Tensor::from_f32(&spread(5 * 1024, 10), &[5, 1024]).unwrap();
    for dtype in [DType::BF16, DType::F16] {
        let grad = |rows: [isize; 2]| {
            let x = x.as_dtype(dtype)?.reshape(&[rows[0], rows[1], 256])?;
            let dy = dy.as_dtype(dtype)?.reshape(&[rows[0], rows[1], 1024])?;
            let (_, grads) = vjp(
                |p: &[Tensor]| Ok(vec![quantized_linear(&p[0], &w, None)?]),
                &[x],
                &[Some(dy)],
            )?;
            grads[0].reshape(&[5, 256])
        };
        assert!(
            bits(&grad([5, 1]).unwrap()) == bits(&grad([1, 5]).unwrap()),
            "{dtype:?}: five rows on a batch axis"
        );
    }
    // A captured constant `x`, as a frozen teacher's, below 4 rows and from them: the forward's
    // bits.
    let b = Tensor::from_f32(&spread(1024, 8), &[1024]).unwrap();
    for rows in [1, 9] {
        let x = Tensor::from_f32(&spread(rows * 256, 7), &[1, rows, 256])
            .and_then(|x| x.as_dtype(DType::BF16))
            .unwrap();
        let (constant, _) = vjp(
            |p: &[Tensor]| Ok(vec![quantized_linear(&x, &w, Some(&p[0]))?]),
            std::slice::from_ref(&b),
            &[None],
        )
        .unwrap();
        let plain = quantized_linear(&x, &w, Some(&b)).unwrap();
        assert!(bits(&constant[0]) == bits(&plain), "{rows} rows");
    }
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
