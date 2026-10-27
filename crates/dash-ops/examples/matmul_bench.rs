//! Matmul throughput on CUDA at Qwen3.5-0.8B's training and decoding shapes, bf16 or f32, each
//! product checked on the host.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example matmul_bench \
//!       -- [--f32] [shape...]
//!
//! Shapes (all by default): `mlp_up`, `mlp_down`, `lm_head`, `decode_up` (one decoding step),
//! `lora_a` and `lora_b` (a rank-16 adapter's two products), or any `[M x K] x [K x N]` as
//! `MxKxN`. Add `autotune` to the features for the kernels Burn picks by timing them, and
//! `cubecl/cuda-cpp` to compile kernels with CubeCL's CUDA C++ backend (NVRTC) rather than its
//! default LLVM one. It prints each product's wall-clock time, which a busy machine inflates; in
//! a device trace, the kernels between the two pauses around the timed passes are the timed
//! products' and nothing else's.
//!
//! Device time per product in microseconds, from nsys (each kernel's median duration, which
//! time-slicing on a shared card does not inflate), on an RTX 5070 Ti (sm_120, 70 SMs) under
//! WSL2, driver 617.14; cuBLAS through torch 2.13 on the same card with the same inputs and
//! timing, reduced-precision reductions off. With autotune, which picks by timing, the range of
//! three processes for bf16 (one for f32). The C++ columns were measured with a fix to
//! cubek-matmul applied locally, described below; without it the autotune column fails at
//! [6144x3584]x[3584x1024].
//!
//! | bf16                        | LLVM  | LLVM, autotune | C++   | C++, autotune | cuBLAS |
//! |-----------------------------|-------|----------------|-------|---------------|--------|
//! | [6144x1024]x[1024x3584]     | 615   | 535-542        | 776   | 463-483       | 459    |
//! | [6144x3584]x[3584x1024]     | 7171  | 593-596        | 1553  | 453-494       | 457    |
//! | [1200x1024]x[1024x248320]   | 24183 | 9769-9803      | 23990 | 8022-8037     | 6528   |
//! | [8x1024]x[1024x3584]        | 25.2  | 13.1-13.9      | 35.3  | 11.7-17.0     | 8.2    |
//! | [2048x1024]x[1024x16]       | 21.6  | 12.3-12.9      | 25.0  | 9.2-10.1      | 5.9    |
//! | [2048x16]x[16x3584]         | 15.0  | 16.7-17.1      | 16.3  | 15.3-19.1     | 18.1   |
//!
//! | f32, IEEE                   | LLVM  | LLVM, autotune | C++   | C++, autotune | cuBLAS |
//! |-----------------------------|-------|----------------|-------|---------------|--------|
//! | [6144x1024]x[1024x3584]     | 2272  | 2278           | 1955  | 1958          | 1372   |
//! | [6144x3584]x[3584x1024]     | 2454  | 2460           | 2133  | 2132          | 1484   |
//! | [1200x1024]x[1024x248320]   | 32323 | 32550          | 29015 | 29229         | 21476  |
//! | [8x1024]x[1024x3584]        | 85.9  | 90.2           | 96.0  | 89.6          | 15.2   |
//! | [2048x1024]x[1024x16]       | 206   | 25.4           | 175   | 25.1          | 12.9   |
//! | [2048x16]x[16x3584]         | 46.0  | 51.8           | 45.8  | 38.9          | 35.8   |
//!
//! Where the time goes, from the launch shapes in the traces, CubeCL's autotune log and
//! cubek-matmul 0.3.1:
//!
//! - Without autotune, Burn runs cubek's `Strategy::Auto` for every product: the single-stage
//!   cmma kernel, or a unit kernel where no tensor-core instruction takes the dtype (IEEE f32),
//!   its stage picked from the aspect ratio. [6144x3584]x[3584x1024] gets 512x128 tiles, 96 CTAs
//!   of 16 warps on 70 SMs, a kernel LLVM compiles to 64 registers a thread and NVRTC to 128,
//!   running 4.6 times faster. Burn 0.22 offers no other choice that does not time: its matmul
//!   strategy is autotune or `Auto`.
//! - The LLVM backend advertises no asynchronous copy, TMA or mbarrier, so every candidate loads
//!   synchronously. The C++ backend adds the TMA candidates, and the warp-specialised one is the
//!   fastest measured at the three large shapes (`CUBECL_AUTOTUNE_LEVEL=full`, which times every
//!   candidate: 465, 456 and 7714 us). At the default level autotune stops once a candidate
//!   reaches 80% of the peak it estimates, which can be a slower one: one of three processes ran
//!   [6144x3584]x[3584x1024] on the simple TMA kernel at 494 us, the other two on the
//!   warp-specialised one at 453.
//! - [1200x1024]x[1024x248320] stays 23% behind. Its 1200 rows run as ten tiles of 128, 6.7% of
//!   the work padding, and ten is not a multiple of the traversal's swizzle width, four, so the
//!   traversal falls back to row by row, each tile row streaming the whole 508 MB right-hand
//!   side: the simple TMA kernel takes 723 us per tile row at 1024 rows (8 tiles, swizzled) and
//!   797 at 1200. At 1024 and 1536 rows (8 and 12 tiles) autotune ends 6% and 5% behind cuBLAS,
//!   at 1408 (11) 16%.
//! - The skinny products are bound by memory, not by the tensor cores. [2048x1024]x[1024x16]
//!   reads its 4.3 MB through 32 CTAs, so at most 32 of the 70 SMs pull them: 470 GB/s, where
//!   cuBLAS launches 64 CTAs and reaches 730. [8x1024]x[1024x3584] streams the 7.3 MB right-hand
//!   side at 630 GB/s at best (double buffering, 112 CTAs), cuBLAS with as many CTAs at 890.
//! - IEEE f32 has no tensor-core path: the unit kernels win, at 255 registers a thread at the
//!   large shapes, 35-44% behind cuBLAS's SGEMM there on the C++ backend and 51-66% on LLVM.
//!
//! The C++ backend is not this workspace's default. With autotune, as published, it fails at
//! problems at least four times taller than wide, such as [6144x3584]x[3584x1024]: two TMA
//! candidates stage 512 rows of the left-hand side, past the 256 elements a tensor map's box may
//! span, and their launch fails. The failure is also claimed on the inputs those kernels only
//! load through their tensor maps, so every later candidate is skipped and the product is never
//! written; `Device::sync` returns `Ok` and the error surfaces when the output is read, which is
//! why this benchmark reads it. Both are upstream's: cubek-matmul accepts the stage, and
//! CubeCL counts a tensor map as written even when the kernel only loads from it. The C++
//! columns above were measured with cubek-matmul refusing the stage, a fix applied locally.
#[cfg(feature = "cuda")]
use burn::tensor::{DType, Device, Distribution, Tensor};
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};

/// (name, m, k, n, timed repetitions).
#[cfg(feature = "cuda")]
const SHAPES: [(&str, usize, usize, usize, usize); 6] = [
    ("mlp_up", 6144, 1024, 3584, 50),
    ("mlp_down", 6144, 3584, 1024, 50),
    ("lm_head", 1200, 1024, 248320, 10),
    ("decode_up", 8, 1024, 3584, 500),
    ("lora_a", 2048, 1024, 16, 500),
    ("lora_b", 2048, 16, 3584, 500),
];

/// Standard deviation of every input element.
#[cfg(feature = "cuda")]
const STD: f64 = 0.02;

/// The pause on each side of the timed passes: in a device trace (nsys), the kernels between the
/// two are the timed products' and nothing else's.
#[cfg(feature = "cuda")]
const GAP: Duration = Duration::from_millis(100);

#[cfg(feature = "cuda")]
fn random(device: &Device, shape: [usize; 2], dtype: DType) -> Tensor<2> {
    Tensor::<2>::random(shape, Distribution::Normal(0.0, STD), (device, dtype))
}

/// Seconds per `a x b`: products for half a second first (the first one tunes, and the clocks
/// come up), then, between two pauses of [`GAP`], the best of three passes of `reps` back to
/// back with at most four results alive, each timed from the first launch to the device going
/// idle. Also returns the last product.
#[cfg(feature = "cuda")]
fn time_matmul(
    device: &Device,
    a: &Tensor<2>,
    b: &Tensor<2>,
    reps: usize,
) -> Result<(f64, Tensor<2>), String> {
    let sync = || device.sync().map_err(|err| format!("{err:?}"));
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < 0.5 {
        let _ = a.clone().matmul(b.clone());
        sync()?;
    }
    std::thread::sleep(GAP);
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
        sync()?;
        best = best.min(start.elapsed().as_secs_f64() / reps as f64);
    }
    std::thread::sleep(GAP);
    Ok((best, outs.pop().unwrap()))
}

/// The elements of `t` on the host, or why they cannot be read: Burn reports a launch that
/// failed only when its output is read, so this is also where a failure surfaces.
#[cfg(feature = "cuda")]
fn host(t: Tensor<2>) -> Result<Vec<f32>, String> {
    t.try_into_data_as::<f32>()
        .map_err(|err| format!("{err:?}"))?
        .try_into_vec::<f32>()
        .map_err(|err| format!("{err:?}"))
}

/// Largest error of nine entries of `c = a x b` against the products recomputed on the host in
/// f64, relative to the size of an entry, `sqrt(k) * STD^2`. A product rounded to bf16 is off by
/// at most 2^-9 of itself, about 0.01 at four standard deviations; one with inputs rounded to
/// TF32 by about 1e-3; one computed in f32 by about 1e-6.
#[cfg(feature = "cuda")]
fn check(a: &Tensor<2>, b: &Tensor<2>, c: &Tensor<2>) -> Result<f64, String> {
    let [m, k] = a.dims();
    let [_, n] = b.dims();
    let mut worst = 0f64;
    for i in [0, m / 2, m - 1] {
        let row = host(a.clone().slice([i..i + 1, 0..k]))?;
        let out = host(c.clone().slice([i..i + 1, 0..n]))?;
        for j in [0, n / 3, n - 1] {
            let col = host(b.clone().slice([0..k, j..j + 1]))?;
            let exact: f64 = row
                .iter()
                .zip(&col)
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum();
            worst = worst.max((out[j] as f64 - exact).abs());
        }
    }
    Ok(worst / ((k as f64).sqrt() * STD * STD))
}

/// `MxKxN` as `(m, k, n)`, for `[m x k] x [k x n]`.
#[cfg(feature = "cuda")]
fn parse_shape(arg: &str) -> Option<(usize, usize, usize)> {
    let dims: Vec<usize> = arg
        .split('x')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    match dims[..] {
        [m, k, n] if m > 0 && k > 0 && n > 0 => Some((m, k, n)),
        _ => None,
    }
}

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let f32 = args.iter().any(|arg| arg == "--f32");
    let mut shapes = Vec::new();
    for arg in args.iter().filter(|arg| !arg.starts_with("--")) {
        if let Some(&(name, m, k, n, reps)) = SHAPES.iter().find(|shape| shape.0 == arg) {
            shapes.push((name.to_string(), m, k, n, reps));
        } else if let Some((m, k, n)) = parse_shape(arg) {
            // Passes of about a teraflop, between the named shapes' 10 and 500 products.
            let reps = (1e12 / (2 * m * k * n) as f64).clamp(10.0, 500.0) as usize;
            shapes.push((arg.clone(), m, k, n, reps));
        } else {
            eprintln!("{arg}: neither a shape's name nor MxKxN");
            std::process::exit(2);
        }
    }
    if shapes.is_empty() {
        let all = SHAPES
            .iter()
            .map(|&(name, m, k, n, reps)| (name.to_string(), m, k, n, reps));
        shapes.extend(all);
    }
    let (dtype, name) = match f32 {
        true => (DType::F32, "f32"),
        false => (DType::BF16, "bf16"),
    };
    // bf16 outputs are rounded to bf16; f32 ones are IEEE f32 with the workspace's
    // `CUBECL_CUDA_TF32=0`, TF32 otherwise.
    let ieee = std::env::var("CUBECL_CUDA_TF32").as_deref() == Ok("0");
    let tolerance = match (f32, ieee) {
        (true, true) => 1e-4,
        _ => 0.02,
    };
    let device = Device::cuda(0);
    println!(
        "{name} matmul on CUDA, autotune {}, fusion {}",
        if cfg!(feature = "autotune") {
            "on"
        } else {
            "off"
        },
        if cfg!(feature = "fusion") {
            "on"
        } else {
            "off"
        }
    );
    let mut failed = Vec::new();
    for (shape, m, k, n, reps) in shapes {
        let a = random(&device, [m, k], dtype);
        let b = random(&device, [k, n], dtype);
        let measured = time_matmul(&device, &a, &b, reps)
            .and_then(|(s, c)| check(&a, &b, &c).map(|err| (s, err)));
        match measured {
            Ok((s, err)) => {
                let tflops = 2.0 * (m * k * n) as f64 / s / 1e12;
                println!(
                    "{shape:9} [{m}x{k}]x[{k}x{n}] {:9.4} ms {tflops:6.2} TFLOPS, error {err:.2e}, \
                     {} products timed",
                    s * 1e3,
                    3 * reps
                );
                if err.is_nan() || err >= tolerance {
                    println!(
                        "{shape:9} the product is wrong: error {err:.2e} over {tolerance:.0e}"
                    );
                    failed.push(shape);
                }
            }
            Err(err) => {
                println!("{shape:9} [{m}x{k}]x[{k}x{n}] failed: {err}");
                failed.push(shape);
            }
        }
    }
    if !failed.is_empty() {
        eprintln!("failed: {}", failed.join(", "));
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("matmul_bench measures the CUDA backend: run it with --features cuda");
}
