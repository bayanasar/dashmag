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
            for rows in [1, 5, 16, 17, 100] {
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
                let (mut worst, mut scale) = (0f64, 0f64);
                for n in 0..rows {
                    for o in 0..p.out {
                        let exact: f64 = (0..p.inp)
                            .map(|i| xs[n * p.inp + i] as f64 * w(&p, i, o))
                            .sum::<f64>()
                            + bias[o] as f64;
                        worst = worst.max((got[n * p.out + o] as f64 - exact).abs());
                        scale = scale.max(exact.abs());
                    }
                }
                println!(
                    "{name} {dtype:?} {rows} rows ({}): max abs {worst:.3e}, relative {:.3e}",
                    if takes {
                        "kernels"
                    } else if mat {
                        "matrix products"
                    } else if pre {
                        "prefill"
                    } else {
                        "reference"
                    },
                    worst / scale
                );
                // Relative to the output's largest value, just above the largest measured on Flex,
                // an RTX 5070 Ti (with and without fusion and autotune) and a Radeon AI PRO R9700
                // (2026-11-13), the same on each to the bit: the output's rounding to its dtype, and
                // on the prefill path `W`'s too. The kernel on RDNA4's matrix products rounds `W` as
                // the prefill path does, and the output once.
                let bound = match (dtype, pre || mat) {
                    (DType::F32, _) => 5e-7,        // 4.64e-7
                    (DType::BF16, false) => 2.2e-3, // 2.13e-3
                    (DType::BF16, true) => 3.9e-3,  // 3.78e-3
                    (DType::F16, false) => 2.7e-4,  // 2.64e-4
                    (_, true) => 4.4e-4,            // 4.28e-4
                    _ => unreachable!(),
                };
                assert!(
                    worst / scale <= bound,
                    "{name} {dtype:?} {rows}: {:.3e}",
                    worst / scale
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
/// where it does (`matches_the_definition`'s 17 and 100 rows of 40 outputs), held to the definition
/// within the prefill path's bounds: 1024 rows of 1024 outputs over `matches_the_definition`'s 384
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
            let (mut worst, mut scale) = (0f64, 0f64);
            for n in 0..rows {
                for o in 0..p.out {
                    let exact: f64 = (0..p.inp)
                        .map(|i| xs[n * p.inp + i] as f64 * w(&p, i, o))
                        .sum();
                    worst = worst.max((got[n * p.out + o] as f64 - exact).abs());
                    scale = scale.max(exact.abs());
                }
            }
            println!(
                "{name} {dtype:?} {rows} x {} (matrix products, whole inputs): relative {:.3e}",
                p.out,
                worst / scale
            );
            let bound = if dtype == DType::BF16 { 3.9e-3 } else { 4.4e-4 };
            assert!(worst / scale <= bound, "{dtype:?}: {:.3e}", worst / scale);
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
