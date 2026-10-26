//! bf16 matmul throughput on CUDA at Qwen3.5-0.8B's training shapes, each product checked on
//! the host.
//!
//!   cargo run --release -p dash-ops --features cuda,fusion --example matmul_bench [shape...]
//!
//! This crate's `cuda` feature leaves Burn's autotune off: add `burn/autotune` to the features
//! for the kernels Burn picks when it tunes, and `cubecl/cuda-cpp` to compile them with CubeCL's
//! CUDA C++ backend (NVRTC) rather than its default LLVM one. TFLOPS on an RTX 5070 Ti (sm_120,
//! 70 SMs) under WSL2, driver 617.14, best of three passes; cuBLAS through torch 2.13, same card,
//! same timing:
//!
//! | `[m x k] x [k x n]`         | `cuda,fusion` | `+burn/autotune` | `+cubecl/cuda-cpp` | cuBLAS |
//! |-----------------------------|---------------|------------------|--------------------|--------|
//! | [6144x1024]x[1024x3584]     | 71            | 81.5             | 84-95              | 96.8   |
//! | [6144x3584]x[3584x1024]     | 6.3           | 74.5             | fails, see below   | 98.0   |
//! | [1200x1024]x[1024x248320]   | 25.5          | 62               | 76.6               | 91.9   |
//! | [8x1024]x[1024x3584]        | 2.0           | 2.8              | 2.5                | 5.4    |
//!
//! The last row is bound by the cost of a dispatch (`dispatch_bench`), not by the kernel.
//! Why the others lag, from nsys launch shapes, CubeCL's autotune log and cubek-matmul 0.3.1:
//!
//! - Without autotune, every matmul runs the single-stage plane kernel with an instruction shape
//!   picked from the aspect ratio: [6144x3584]x[3584x1024] gets 32x8x16 and 512x128 tiles, 96
//!   CTAs of 16 warps on 70 SMs.
//! - The LLVM backend advertises no async copy, TMA or mbarrier on any card, so every candidate
//!   loads global memory synchronously. The winners are 128x128 tiles of 4 warps at 208-252
//!   registers a thread, two CTAs an SM: ordered double buffering for [6144x3584]x[3584x1024]
//!   (384 CTAs, 2.7 waves), simple multi-row for [1200x1024]x[1024x248320] (m = 1200 in ten tiles
//!   of 128). Nothing better is in the set: `CUBECL_AUTOTUNE_LEVEL=full`, which measures every
//!   candidate without stopping at 80% of the measured 101 TFLOPS peak, lands within 1%.
//! - The C++ backend enables the TMA candidates; warp-specialised TMA kernels win, at 128x64 tiles
//!   for [1200x1024]x[1024x248320], which runs as fast at m = 1280 (79.8 TFLOPS). For
//!   [6144x3584]x[3584x1024] a TMA candidate asks for a box dimension over 256, its launch fails
//!   inside autotune's profiling, tuning stops and the product is never written; Burn reports
//!   that only when the output is read, which is why this benchmark reads it. Computed as its
//!   transpose, [1024x3584]x[3584x6144], it runs at 87.6 TFLOPS. The samples are noisy (in one
//!   run, medians of 3.3-4.4 ms for minima of 0.46-0.95 ms, with about a second of NVRTC
//!   compilation per candidate), so the pick for [6144x1024]x[1024x3584] varies between runs.
#[cfg(feature = "cuda")]
use burn::tensor::{DType, Device, Distribution, FloatDType, Tensor};
#[cfg(feature = "cuda")]
use std::time::Instant;

/// (name, m, k, n, timed repetitions).
#[cfg(feature = "cuda")]
const SHAPES: [(&str, usize, usize, usize, usize); 4] = [
    ("mlp_up", 6144, 1024, 3584, 50),
    ("mlp_down", 6144, 3584, 1024, 50),
    ("lm_head", 1200, 1024, 248320, 10),
    ("decode_up", 8, 1024, 3584, 500),
];

/// Standard deviation of every input element.
#[cfg(feature = "cuda")]
const STD: f64 = 0.02;

#[cfg(feature = "cuda")]
fn random(device: &Device, shape: [usize; 2]) -> Tensor<2> {
    Tensor::<2>::random(shape, Distribution::Normal(0.0, STD), (device, DType::BF16))
}

/// Seconds per `a x b`: products for half a second first (the first one tunes, and the clocks
/// come up), then the best of three passes of `reps` back to back with at most four results
/// alive, each timed from the first launch to the device going idle. Also returns the last
/// product.
#[cfg(feature = "cuda")]
fn time_matmul(device: &Device, a: &Tensor<2>, b: &Tensor<2>, reps: usize) -> (f64, Tensor<2>) {
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < 0.5 {
        let _ = a.clone().matmul(b.clone());
        device.sync().unwrap();
    }
    let mut best = f64::INFINITY;
    let mut outs = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..reps {
            outs.push(a.clone().matmul(b.clone()));
            if outs.len() > 4 {
                outs.remove(0);
            }
        }
        device.sync().unwrap();
        best = best.min(start.elapsed().as_secs_f64() / reps as f64);
    }
    (best, outs.pop().unwrap())
}

#[cfg(feature = "cuda")]
fn host(t: Tensor<2>) -> Vec<f32> {
    t.cast(FloatDType::F32)
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

/// Largest error of nine entries of `c = a x b` against the products recomputed on the host in
/// f64, relative to the size of an entry, `sqrt(k) * STD^2`. A product rounded to bf16 is off by
/// at most 2^-9 of itself, which is about 0.01 at four standard deviations; reading `c` also
/// surfaces a launch that failed, which Burn otherwise reports only when the output is read.
#[cfg(feature = "cuda")]
fn check(a: &Tensor<2>, b: &Tensor<2>, c: &Tensor<2>) -> f64 {
    let [m, k] = a.dims();
    let [_, n] = b.dims();
    let mut worst = 0f64;
    for i in [0, m / 2, m - 1] {
        let row = host(a.clone().slice([i..i + 1, 0..k]));
        let out = host(c.clone().slice([i..i + 1, 0..n]));
        for j in [0, n / 3, n - 1] {
            let col = host(b.clone().slice([0..k, j..j + 1]));
            let exact: f64 = row
                .iter()
                .zip(&col)
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum();
            worst = worst.max((out[j] as f64 - exact).abs());
        }
    }
    worst / ((k as f64).sqrt() * STD * STD)
}

#[cfg(feature = "cuda")]
fn main() {
    let only: Vec<String> = std::env::args().skip(1).collect();
    let device = Device::cuda(0);
    println!(
        "bf16 matmul on CUDA, fusion {}",
        if cfg!(feature = "fusion") {
            "on"
        } else {
            "off"
        }
    );
    for (name, m, k, n, reps) in SHAPES {
        if !only.is_empty() && !only.iter().any(|s| s == name) {
            continue;
        }
        let a = random(&device, [m, k]);
        let b = random(&device, [k, n]);
        let (s, c) = time_matmul(&device, &a, &b, reps);
        let err = check(&a, &b, &c);
        let tflops = 2.0 * (m * k * n) as f64 / s / 1e12;
        println!(
            "{name:9} [{m}x{k}]x[{k}x{n}] {:8.3} ms {tflops:6.1} TFLOPS, error {err:.4}",
            s * 1e3
        );
        assert!(err < 0.02, "{name}: the product is wrong");
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("matmul_bench measures the CUDA backend: run it with --features cuda,fusion");
}
