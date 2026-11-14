//! The quantized linear's decoding kernels against Burn's bf16 matmul over the same weight
//! unpacked: wall clock per call, best of three passes, and the bytes of weight read per second.
//! Each pass cycles through four copies of the weight, so that it is read from memory rather than
//! from a cache that holds one copy.
//!
//!   cargo run --release -p dash-ops --features cuda --example quant_bench -- [IN x OUT ...]
//!
//! Shapes `INxOUT` (default `5120x17408 17408x5120`), each at 1 to 2048 rows of bf16 input,
//! weights in groups of 128.
#[cfg(any(feature = "cuda", feature = "rocm"))]
use burn::tensor::{DType, Device, Distribution, FloatDType, Int, Tensor, TensorData};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use dash_ops::quant::{
    QuantOptions, QuantizedWeight, kernels_take, prefill_takes, quantized_linear,
};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use std::time::Instant;

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn time(device: &Device, mut f: impl FnMut() -> Tensor<2>) -> f64 {
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < 0.5 {
        drop(f());
        device.sync().unwrap();
    }
    let reps = 50;
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let start = Instant::now();
        let mut keep = Vec::new();
        for _ in 0..reps {
            keep.push(f());
            if keep.len() > 4 {
                keep.remove(0);
            }
        }
        device.sync().unwrap();
        best = best.min(start.elapsed().as_secs_f64() / reps as f64);
    }
    best
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn main() {
    #[cfg(feature = "cuda")]
    let (device, runtime) = (Device::cuda(0), "CUDA");
    #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
    let (device, runtime) = (Device::rocm(0), "ROCm");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let shapes: Vec<(usize, usize)> = match args.is_empty() {
        true => vec![(5120, 17408), (17408, 5120)],
        false => args
            .iter()
            .map(|a| {
                let (i, o) = a.split_once('x').expect("INxOUT");
                (i.parse().unwrap(), o.parse().unwrap())
            })
            .collect(),
    };
    let group = 128;
    println!("int4 weights in groups of {group}, bf16 inputs, on {runtime}");
    for (inp, out) in shapes {
        let make = |seed: u32| {
            let words: Vec<i32> = (0..inp / 8 * out)
                .map(|i| (i as u32 ^ seed).wrapping_mul(2_654_435_761) as i32)
                .collect();
            let qweight = Tensor::<2, Int>::from_data(
                TensorData::new(words, [inp / 8, out]),
                (&device, DType::I32),
            );
            let qzeros = Tensor::<2, Int>::from_data(
                TensorData::new(
                    vec![0x8888_8888u32 as i32; inp / group * out / 8],
                    [inp / group, out / 8],
                ),
                (&device, DType::I32),
            );
            let scales = Tensor::<2>::random(
                [inp / group, out],
                Distribution::Uniform(0.001, 0.01),
                (&device, DType::F16),
            );
            QuantizedWeight::from_gptq(
                qweight,
                qzeros,
                scales,
                None,
                QuantOptions::default().with_group(group),
            )
            .unwrap()
        };
        // Four copies, each its own buffers, cycled through: a pass reads them from memory.
        let weights: Vec<QuantizedWeight> = (0..4).map(make).collect();
        let dense: Vec<Tensor<2>> = weights
            .iter()
            .map(|w| w.dequantize().cast(FloatDType::BF16))
            .collect();
        let bytes = (inp * out / 2 + inp / group * out * 2) as f64;
        for rows in [1, 2, 4, 8, 16, 32, 128, 512, 2048] {
            let x = Tensor::<2>::random(
                [rows, inp],
                Distribution::Normal(0.0, 1.0),
                (&device, DType::BF16),
            );
            let takes = kernels_take(&device, DType::BF16, [rows, inp, out], group);
            let pre = prefill_takes(&device, DType::BF16, [rows, inp, out], group);
            let mut i = 0;
            let q = time(&device, || {
                i += 1;
                quantized_linear(x.clone(), &weights[i % 4], None)
            });
            let d = time(&device, || {
                i += 1;
                x.clone().matmul(dense[i % 4].clone())
            });
            println!(
                "[{rows}x{inp}] x [{inp}x{out}]: int4 {:.1} us ({:.0} GB/s of weight, {}), bf16 matmul {:.1} us ({:.0} GB/s)",
                q * 1e6,
                bytes / q / 1e9,
                if takes {
                    "kernels"
                } else if pre {
                    "prefill"
                } else {
                    "reference"
                },
                d * 1e6,
                (inp * out * 2) as f64 / d / 1e9
            );
        }
    }
}

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
fn main() {
    eprintln!("quant_bench measures a GPU backend: run it with --features cuda or rocm");
}
