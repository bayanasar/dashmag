//! The quantized linear against its definition computed on the host in f64, and the refusals of
//! `QuantizedWeight::from_gptq`.

mod common;

use burn::tensor::{DType, Device, FloatDType, Int, Tensor, TensorData, f16};
use dash_ops::quant::{
    QuantOptions, QuantizedWeight, kernels_take, matrix_runs, matrix_takes, prefill_takes, probe,
    quantized_linear,
};

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform on `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    }
}

/// A checkpoint's tensors on the host: values `q` `[in, out]` in 0..16, scales `[in / group,
/// out]` (as f16 values), packed as GPTQ packs them, with every zero 8.
struct Packed {
    inp: usize,
    out: usize,
    group: usize,
    q: Vec<u8>,
    scales: Vec<f16>,
    qweight: Vec<i32>,
    qzeros: Vec<i32>,
}

fn packed(seed: u64, inp: usize, out: usize, group: usize) -> Packed {
    let mut rng = Rng(seed);
    let q: Vec<u8> = (0..inp * out).map(|_| (rng.next() % 16) as u8).collect();
    let scales: Vec<f16> = (0..inp / group * out)
        .map(|_| f16::from_f32(0.002 + 0.01 * rng.unit().abs()))
        .collect();
    let mut qweight = vec![0i32; inp / 8 * out];
    for i in 0..inp {
        for o in 0..out {
            qweight[i / 8 * out + o] |= (q[i * out + o] as i32) << (4 * (i % 8));
        }
    }
    let qzeros = vec![0x8888_8888u32 as i32; inp / group * out / 8];
    Packed {
        inp,
        out,
        group,
        q,
        scales,
        qweight,
        qzeros,
    }
}

fn ints<const D: usize>(values: &[i32], shape: [usize; D], device: &Device) -> Tensor<D, Int> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), shape),
        (device, DType::I32),
    )
}

fn weight(p: &Packed, device: &Device, g_idx: bool) -> Result<QuantizedWeight, String> {
    let scales = Tensor::<2>::from_data(
        TensorData::new(p.scales.clone(), [p.inp / p.group, p.out]),
        (device, DType::F16),
    );
    let g = g_idx.then(|| {
        let g: Vec<i32> = (0..p.inp).map(|i| (i / p.group) as i32).collect();
        ints(&g, [p.inp], device)
    });
    QuantizedWeight::from_gptq(
        ints(&p.qweight, [p.inp / 8, p.out], device),
        ints(&p.qzeros, [p.inp / p.group, p.out / 8], device),
        scales,
        g,
        QuantOptions::default().with_group(p.group),
    )
}

/// `W[i, o] = (q - 8) · scale` in f64.
fn w(p: &Packed, i: usize, o: usize) -> f64 {
    (p.q[i * p.out + o] as f64 - 8.0) * p.scales[i / p.group * p.out + o].to_f64()
}

#[allow(unused_mut, clippy::vec_init_then_push)]
fn devices() -> Vec<(&'static str, Device)> {
    let mut devices = Vec::new();
    #[cfg(feature = "flex")]
    devices.push(("flex", Device::flex()));
    #[cfg(feature = "cubecl")]
    devices.push(("cubecl", common::cube_device()));
    devices
}

#[test]
fn dequantize_is_exact() {
    for (name, device) in devices() {
        let p = packed(1, 256, 48, 128);
        let got: Vec<f32> = weight(&p, &device, true)
            .unwrap()
            .dequantize()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for i in 0..p.inp {
            for o in 0..p.out {
                assert_eq!(got[i * p.out + o] as f64, w(&p, i, o), "{name} W[{i}, {o}]");
            }
        }
    }
}

/// How a path may round, beyond computing the definition in f32: whether `W` is rounded to `x`'s
/// dtype before its products, and whether the product is rounded to it before the bias is added.
#[derive(Clone, Copy)]
struct Roundings {
    w: bool,
    product: bool,
}

/// What one output may be off by, from the roundings its path is allowed (`dash_contract::quant`),
/// not from any input set: `y` the output by the definition, `s` the sum of `|x_i w_io|` and `|b|`
/// over the output's terms, `n` the inputs, `unit` the unit roundoff of `x`'s dtype (2^-8 in bf16,
/// 2^-11 in f16, 2^-24 in f32). Before the output's rounding, the value computed is off by at most
/// `e`: `W`'s rounding, `unit · s`; the products and their sum in f32 in any order, `gamma · s` with
/// `gamma = (n + 2) 2^-24 / (1 - (n + 2) 2^-24)` (over the products of `W` rounded, a little more);
/// and the product's rounding before the bias, of a value within `s (1 + w)(1 + gamma)`. The
/// output's rounding adds `unit · (|y| + e)`. Every f32 operation is taken to round to nearest; an
/// accumulator that truncates could be off by twice the `gamma` term.
fn allowed(y: f64, s: f64, n: usize, unit: f64, r: Roundings) -> f64 {
    let f32_unit = 2f64.powi(-24);
    let gamma = (n + 2) as f64 * f32_unit / (1.0 - (n + 2) as f64 * f32_unit);
    let w = if r.w { unit } else { 0.0 };
    let product = if r.product { unit } else { 0.0 };
    let e = w * s + gamma * s * (1.0 + w) + product * s * (1.0 + w) * (1.0 + gamma);
    e + unit * (y.abs() + e)
}

/// The unit roundoff of `dtype`: half a unit in the last place, relative.
fn unit_roundoff(dtype: DType) -> f64 {
    match dtype {
        DType::BF16 => 2f64.powi(-8),
        DType::F16 => 2f64.powi(-11),
        _ => 2f64.powi(-24),
    }
}

/// The tests that read the [`probe`] counters, one at a time: the counters are the process's.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn matches_the_definition() {
    let _serial = serial();
    for (name, device) in devices() {
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            if !common::has(&device, dtype) {
                continue;
            }
            // 8 and 9 rows step over the line past which RDNA4's matrix products take the call.
            for rows in [1, 5, 8, 9, 16, 17, 100] {
                let p = packed(2, 384, 40, 128);
                let takes = kernels_take(&device, dtype, [rows, p.inp, p.out], p.group);
                let pre = prefill_takes(&device, dtype, [rows, p.inp, p.out], p.group);
                let mat = matrix_takes(&device, dtype, [rows, p.inp, p.out], p.group);
                let before = (
                    probe::kernel_launches(),
                    probe::dequant_launches(),
                    probe::matrix_launches(),
                );
                let mut rng = Rng(3);
                let x: Vec<f32> = (0..rows * p.inp).map(|_| rng.unit()).collect();
                let bias: Vec<f32> = (0..p.out).map(|_| rng.unit()).collect();
                let xt = Tensor::<3>::from_data(TensorData::new(x, [1, rows, p.inp]), &device)
                    .cast(FloatDType::from(dtype));
                // The definition reads x as given: in its dtype, rounded.
                let xs: Vec<f32> = xt
                    .clone()
                    .cast(FloatDType::F32)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap();
                let weight = weight(&p, &device, false).unwrap();
                let bt = Tensor::<1>::from_data(TensorData::new(bias.clone(), [p.out]), &device);
                let y = quantized_linear(xt, &weight, Some(bt));
                assert_eq!(y.dims(), [1, rows, p.out]);
                assert_eq!(y.dtype(), dtype);
                let got: Vec<f32> = y
                    .cast(FloatDType::F32)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap();
                // The prefill path rounds `W` and the product before the bias; the kernel on matrix
                // products rounds `W`; the reference and the decoding kernels only the output. In f32
                // nothing is rounded but the sums.
                let wide = dtype == DType::F32;
                let roundings = Roundings {
                    w: (pre || mat) && !wide,
                    product: pre && !wide,
                };
                let unit = unit_roundoff(dtype);
                let (mut worst, mut scale, mut share) = (0f64, 0f64, 0f64);
                for n in 0..rows {
                    for o in 0..p.out {
                        let terms = (0..p.inp).map(|i| xs[n * p.inp + i] as f64 * w(&p, i, o));
                        let exact = terms.clone().sum::<f64>() + bias[o] as f64;
                        let s = terms.map(f64::abs).sum::<f64>() + (bias[o] as f64).abs();
                        let err = (got[n * p.out + o] as f64 - exact).abs();
                        let bound = allowed(exact, s, p.inp, unit, roundings);
                        assert!(
                            err <= bound,
                            "{name} {dtype:?} {rows} rows, output ({n}, {o}): off by {err:.3e}, \
                             allowed {bound:.3e}"
                        );
                        worst = worst.max(err);
                        scale = scale.max(exact.abs());
                        share = share.max(err / bound);
                    }
                }
                println!(
                    "{name} {dtype:?} {rows} rows ({}): max abs {worst:.3e}, relative {:.3e}, \
                     at most {:.2} of what its roundings allow",
                    if takes {
                        "kernels"
                    } else if mat {
                        "matrix products"
                    } else if pre {
                        "prefill"
                    } else {
                        "reference"
                    },
                    worst / scale,
                    share
                );
                assert_eq!(
                    probe::kernel_launches() > before.0,
                    takes,
                    "{name} {dtype:?} {rows}"
                );
                assert_eq!(
                    probe::dequant_launches() > before.1,
                    pre,
                    "{name} {dtype:?} {rows}"
                );
                assert_eq!(
                    probe::matrix_launches() > before.2,
                    mat,
                    "{name} {dtype:?} {rows}"
                );
                if name == "cubecl" && common::on_cuda(&device) {
                    assert_eq!(takes, rows <= 16, "CUDA takes up to 16 rows");
                    assert_eq!(pre, rows > 16, "CUDA takes the prefill path past 16 rows");
                }
            }
        }
    }
}

/// The kernel on RDNA4's matrix products where it does not split the inputs between cubes, as
/// where it does (`matches_the_definition`'s 17 and 100 rows of 40 outputs), each output held to the
/// definition within what rounding `W` and the output allows ([`allowed`]): 1024 rows of 1024
/// outputs over `matches_the_definition`'s 384
/// inputs make 128 cubes of 64 rows and 128 outputs, enough to fill a device of 64 compute units.
/// Only on a device the kernel takes, and that does not split this shape ([`matrix_runs`]), saying
/// so where it does.
#[test]
fn matrix_products_whole_inputs() {
    let _serial = serial();
    for (name, device) in devices() {
        for dtype in [DType::BF16, DType::F16] {
            let p = packed(5, 384, 1024, 128);
            let rows = 1024;
            if !matrix_takes(&device, dtype, [rows, p.inp, p.out], p.group) {
                continue;
            }
            let runs = matrix_runs(&device, dtype, [rows, p.inp, p.out], p.group);
            if runs != 1 {
                println!("skipped {dtype:?}: this device splits 1024 x 1024 into {runs} runs");
                continue;
            }
            let before = probe::matrix_launches();
            let mut rng = Rng(7);
            let x: Vec<f32> = (0..rows * p.inp).map(|_| rng.unit()).collect();
            let xt = Tensor::<2>::from_data(TensorData::new(x, [rows, p.inp]), &device)
                .cast(FloatDType::from(dtype));
            let xs: Vec<f32> = xt
                .clone()
                .cast(FloatDType::F32)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let weight = weight(&p, &device, false).unwrap();
            let got: Vec<f32> = quantized_linear(xt, &weight, None)
                .cast(FloatDType::F32)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let roundings = Roundings {
                w: true,
                product: false,
            };
            let unit = unit_roundoff(dtype);
            let (mut worst, mut scale, mut share) = (0f64, 0f64, 0f64);
            for n in 0..rows {
                for o in 0..p.out {
                    let terms = (0..p.inp).map(|i| xs[n * p.inp + i] as f64 * w(&p, i, o));
                    let exact = terms.clone().sum::<f64>();
                    let s = terms.map(f64::abs).sum::<f64>();
                    let err = (got[n * p.out + o] as f64 - exact).abs();
                    let bound = allowed(exact, s, p.inp, unit, roundings);
                    assert!(
                        err <= bound,
                        "{dtype:?} output ({n}, {o}): off by {err:.3e}, allowed {bound:.3e}"
                    );
                    worst = worst.max(err);
                    scale = scale.max(exact.abs());
                    share = share.max(err / bound);
                }
            }
            println!(
                "{name} {dtype:?} {rows} x {} (matrix products, whole inputs): relative {:.3e}, \
                 at most {:.2} of what its roundings allow",
                p.out,
                worst / scale,
                share
            );
            assert!(
                probe::matrix_launches() > before,
                "{dtype:?}: the kernel ran"
            );
        }
    }
}

#[test]
fn refuses_what_it_cannot_read() {
    let device = devices().remove(0).1;
    let p = packed(4, 256, 16, 128);
    assert!(weight(&p, &device, true).is_ok());
    // GPTQ v1's convention: zero - 1 stored.
    let mut v1 = packed(4, 256, 16, 128);
    v1.qzeros[3] = 0x7777_7777;
    let err = weight(&v1, &device, false).unwrap_err();
    assert!(err.contains("0x77777777"), "{err}");
    // Reordered groups.
    let g: Vec<i32> = (0..256usize)
        .map(|i| ((i + 1) % 256 / 128) as i32)
        .collect();
    let scales = Tensor::<2>::from_data(
        TensorData::new(p.scales.clone(), [2, 16]),
        (&device, DType::F16),
    );
    let err = QuantizedWeight::from_gptq(
        ints(&p.qweight, [32, 16], &device),
        ints(&p.qzeros, [2, 2], &device),
        scales.clone(),
        Some(ints(&g, [256], &device)),
        QuantOptions::default(),
    )
    .unwrap_err();
    assert!(err.contains("g_idx"), "{err}");
    // Shapes that disagree, and scales not f16.
    let err = QuantizedWeight::from_gptq(
        ints(&p.qweight, [32, 16], &device),
        ints(&p.qzeros, [2, 2], &device),
        scales.clone(),
        None,
        QuantOptions::default().with_group(64),
    )
    .unwrap_err();
    assert!(err.contains("wants scales"), "{err}");
    let err = QuantizedWeight::from_gptq(
        ints(&p.qweight, [32, 16], &device),
        ints(&p.qzeros, [2, 2], &device),
        scales.cast(FloatDType::F32),
        None,
        QuantOptions::default(),
    )
    .unwrap_err();
    assert!(err.contains("f16"), "{err}");
}

/// The backward pass on `Autodiff`: `x`'s gradient against `dy Wᵀ` and the bias's against the sum
/// of `dy` over the rows, both computed on the host in f64, each element within what summing in
/// f32 and the one rounding to its dtype allow ([`allowed`], neither `W` nor the product rounded);
/// the bias in f32 beside `x` in another dtype at 9 rows; beside a frozen bias, `x`'s gradient the
/// same and none for the bias; and with neither asking for a gradient, nothing recorded.
#[test]
fn backward_gives_x_and_the_bias_their_gradients() {
    let _serial = serial();
    for (name, plain) in devices() {
        let device = plain.autodiff();
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            if !common::has(&device, dtype) {
                continue;
            }
            for rows in [1, 9, 100] {
                let p = packed(4, 384, 40, 128);
                let weight = weight(&p, &device, false).unwrap();
                let bias_dtype = if rows == 9 { DType::F32 } else { dtype };
                let mut rng = Rng(5);
                let mut host = |n: usize| (0..n).map(|_| rng.unit()).collect::<Vec<f32>>();
                let (x, dy, b) = (host(rows * p.inp), host(rows * p.out), host(p.out));
                let x = Tensor::<3>::from_data(TensorData::new(x, [1, rows, p.inp]), &device)
                    .cast(FloatDType::from(dtype))
                    .require_grad();
                let dy = Tensor::<3>::from_data(TensorData::new(dy, [1, rows, p.out]), &device)
                    .cast(FloatDType::from(dtype));
                let bias = Tensor::<1>::from_data(TensorData::new(b, [p.out]), &device)
                    .cast(FloatDType::from(bias_dtype))
                    .require_grad();
                let runs = probe::backward_runs();
                let y = quantized_linear(x.clone(), &weight, Some(bias.clone()));
                // The gradient of `sum(y · dy)` with respect to `y` is `dy`, exactly.
                let grads = (y.cast(FloatDType::F32) * dy.clone().cast(FloatDType::F32))
                    .sum()
                    .backward();
                assert_eq!(probe::backward_runs(), runs + 1, "{name} {dtype:?} {rows}");
                let dx = x.grad(&grads).expect("x's gradient");
                let db = bias.grad(&grads).expect("the bias's gradient");
                assert_eq!(
                    (dx.dtype(), db.dtype()),
                    (dtype, bias_dtype),
                    "{name} {rows}"
                );
                let read = |data: TensorData| data.convert::<f32>().try_to_vec::<f32>().unwrap();
                let dys = read(dy.clone().cast(FloatDType::F32).into_data());
                let dxs = read(dx.cast(FloatDType::F32).into_data());
                let dbs = read(db.cast(FloatDType::F32).into_data());
                // A frozen bias beside a trained `x`, LoRA's usual case: `x`'s gradient alone, the
                // same bits, and none for the bias. (Burn's `detach` keeps a tensor's requirement.)
                let frozen = bias.clone().detach().set_require_grad(false);
                let y = quantized_linear(x.clone(), &weight, Some(frozen.clone()));
                let grads = (y.cast(FloatDType::F32) * dy.clone().cast(FloatDType::F32))
                    .sum()
                    .backward();
                assert!(frozen.grad(&grads).is_none(), "{name} {dtype:?} {rows}");
                let alone = x.grad(&grads).expect("x's gradient beside a frozen bias");
                assert_eq!(
                    read(alone.cast(FloatDType::F32).into_data()),
                    dxs,
                    "{name} {dtype:?} {rows}: x's gradient beside a frozen bias"
                );
                let none = Roundings {
                    w: false,
                    product: false,
                };
                let (mut worst_x, mut share_x) = (0f64, 0f64);
                for n in 0..rows {
                    for i in 0..p.inp {
                        let terms = (0..p.out).map(|o| dys[n * p.out + o] as f64 * w(&p, i, o));
                        let exact = terms.clone().sum::<f64>();
                        let s = terms.map(f64::abs).sum::<f64>();
                        let err = (dxs[n * p.inp + i] as f64 - exact).abs();
                        let bound = allowed(exact, s, p.out, unit_roundoff(dtype), none);
                        assert!(
                            err <= bound,
                            "{name} {dtype:?} {rows} rows, dx ({n}, {i}): off by {err:.3e}, \
                             allowed {bound:.3e}"
                        );
                        worst_x = worst_x.max(err);
                        share_x = share_x.max(err / bound);
                    }
                }
                let (mut worst_b, mut share_b) = (0f64, 0f64);
                for o in 0..p.out {
                    let terms = (0..rows).map(|n| dys[n * p.out + o] as f64);
                    let exact = terms.clone().sum::<f64>();
                    let s = terms.map(f64::abs).sum::<f64>();
                    let err = (dbs[o] as f64 - exact).abs();
                    let bound = allowed(exact, s, rows, unit_roundoff(bias_dtype), none);
                    assert!(
                        err <= bound,
                        "{name} {dtype:?} {rows} rows, dbias {o}: off by {err:.3e}, allowed \
                         {bound:.3e}"
                    );
                    worst_b = worst_b.max(err);
                    share_b = share_b.max(err / bound);
                }
                println!(
                    "{name} {dtype:?} {rows} rows, backward: dx max abs {worst_x:.3e}, at most \
                     {share_x:.2} of what its roundings allow; dbias ({bias_dtype:?}) {worst_b:.3e}, \
                     {share_b:.2}"
                );
            }
        }
        // Neither `x` nor the bias asks for a gradient: the output records nothing.
        let p = packed(4, 384, 40, 128);
        let weight = weight(&p, &device, false).unwrap();
        let x = Tensor::<2>::zeros([3, p.inp], &device);
        let bias = Tensor::<1>::zeros([p.out], &device);
        assert!(!quantized_linear(x.clone(), &weight, Some(bias)).is_require_grad());
        assert!(!quantized_linear(x, &weight, None).is_require_grad());
    }
}
