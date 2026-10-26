//! What one dispatch costs on CUDA: a decode-shaped chain of bf16 [8x1024]x[1024x1024] matmuls
//! and element-wise ops.
//!
//!   cargo run --release -p dash-ops --features cuda,fusion --example dispatch_bench [phase...]
//!
//! Phases (all by default): `chain`, `readback`, `independent`, `dependent`, `graph`. Add
//! `burn/autotune` to the features for the kernels Burn tunes (see `matmul_bench`). Milliseconds
//! per step, each step reading one element on the host, on an RTX 5070 Ti under WSL2 (driver
//! 617.14) with autotune, ranges over three runs on a shared machine; torch 2.13 on the same card:
//!
//! | step                             | Burn, fusion | Burn, graph | torch eager | torch graph |
//! |----------------------------------|--------------|-------------|-------------|-------------|
//! | 24 x (matmul + 10 x (mul, tanh)) | 1.24-1.26    | 0.79-0.85   | 3.1-4.4     | 0.53-0.56   |
//! | 24 x matmul                      | 0.96-1.03    | 0.35-0.54   | 0.25-0.31   | 0.18-0.19   |
//!
//! The loop this benchmark comes from read with `into_data()`, which is lazy in Burn: untouched,
//! it copies and waits for nothing, so that loop timed the issuing of a step (0.49-0.72 ms for 24
//! matmuls), not the step. Where a matmul's 40 us go, from nsys in steady state:
//!
//! - The kernel takes 11.7-12.8 us on 32 CTAs; cuBLAS's takes 5.6 us.
//! - Burn's fusion runs on its own thread and CubeCL's server on another. Each hand-off is a
//!   queue of 32 tasks, run when full or flushed; an idle thread spins, yields, then sleeps in
//!   150 us steps. In a step, the first launch comes about 0.5 ms after the previous read; the
//!   GPU is busy 0.28 ms of 1.17 (nsys attached).
//! - CubeCL's server is the narrowest stage: back to back, it launches every 29 us (median 12)
//!   while each kernel takes 12.8; cuLaunchKernel alone has a median of 5.6 us and a mean of 14.5
//!   on WSL2. torch launches every 9.6 us, cuLaunchKernel median 4.6 us.
//! - A read crosses both threads, then a device-to-host copy and an event wait of about 70 us on
//!   WSL2; on an idle pipeline it takes 120-370 us, most of it threads waking from their sleeps.
//!   Without fusion, one thread fewer: 24 matmuls 0.71 ms.
//!
//! CUDA graphs remove all of it but the kernels and the read. CubeCL 0.11.1 has them on its client
//! (`graph_prepare`, `start_capture`, `stop_capture`, `Graph::replay`); Burn's tensor API, and
//! fusion, do not reach them. The `graph` phase records one step on Burn's CubeCL backend and
//! replays it, checked bit for bit against eager steps (eager without fusion: 4.5-8.3 and
//! 0.73-1.55 ms).
#[cfg(feature = "cuda")]
use burn::tensor::{DType, Device, Distribution, Tensor};
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
fn random(device: &Device, shape: [usize; 2]) -> Tensor<2> {
    Tensor::<2>::random(
        shape,
        Distribution::Normal(0.0, 0.02),
        (device, DType::BF16),
    )
}

/// Milliseconds per step of `layers` x (matmul [8x1024]x[1024x1024], then `ops` x mul + tanh),
/// each step ending by reading one element of its output on the host. With `lazy`, the read is
/// `into_data()` with its bytes never touched, as in the loop this benchmark comes from: Burn
/// reads lazily, so that copies nothing and waits for nothing, and the step is timed as issued.
#[cfg(feature = "cuda")]
fn chain(device: &Device, layers: usize, ops: usize, steps: usize, lazy: bool) -> f64 {
    let w = random(device, [1024, 1024]);
    let mut x = random(device, [8, 1024]);
    let step = |x: Tensor<2>| {
        let mut x = x;
        for _ in 0..layers {
            let mut h = x.matmul(w.clone());
            for _ in 0..ops {
                h = (h * 0.999).tanh();
            }
            x = h;
        }
        x
    };
    let read = |x: &Tensor<2>| {
        let first = x.clone().slice([0..1, 0..1]);
        match lazy {
            true => drop(first.into_data()),
            false => assert!(first.into_scalar::<f32>().is_finite()),
        }
    };
    for _ in 0..3 {
        x = step(x);
        read(&x);
    }
    let start = Instant::now();
    for _ in 0..steps {
        x = step(x);
        read(&x);
    }
    start.elapsed().as_secs_f64() / steps as f64 * 1e3
}

/// Microseconds per one-element read of a tensor already computed.
#[cfg(feature = "cuda")]
fn readback(device: &Device, reps: usize) -> f64 {
    let x = random(device, [8, 1024]);
    let read = || x.clone().slice([0..1, 0..1]).into_scalar::<f32>();
    for _ in 0..10 {
        read();
    }
    let start = Instant::now();
    for _ in 0..reps {
        read();
    }
    start.elapsed().as_secs_f64() / reps as f64 * 1e6
}

/// Microseconds per matmul for `n` of them issued back to back and one sync at the end, and
/// microseconds per matmul until the loop issuing them returned. `dependent` feeds each
/// product into the next; otherwise all `n` read the same input.
#[cfg(feature = "cuda")]
fn back_to_back(device: &Device, n: usize, dependent: bool) -> (f64, f64) {
    let w = random(device, [1024, 1024]);
    let x = random(device, [8, 1024]);
    let run = |n: usize| {
        let start = Instant::now();
        let mut h = x.clone();
        for _ in 0..n {
            if dependent {
                h = h.matmul(w.clone());
            } else {
                let _ = x.clone().matmul(w.clone());
            }
        }
        let issued = start.elapsed().as_secs_f64();
        device.sync().unwrap();
        (start.elapsed().as_secs_f64(), issued)
    };
    run(100);
    let (total, issued) = run(n);
    (total / n as f64 * 1e6, issued / n as f64 * 1e6)
}

/// The chain run straight on Burn's CubeCL backend, without fusion: eagerly, and recorded once
/// as a CUDA graph through CubeCL's capture API (`Client::graph_prepare`, `start_capture`,
/// `stop_capture`, `Graph::replay`), which Burn's tensor API does not reach.
#[cfg(feature = "cuda")]
mod graph {
    use burn::backend::ops::FloatTensorOps;
    use burn::backend::{Bytes, Distribution, FloatDType, Shape, Slice};
    use burn_cubecl::{CubeBackend as B, CubeDevice, tensor::CubeTensor};
    use cubecl::cuda::CudaDevice;
    use std::time::Instant;

    fn step(x: CubeTensor, w: &CubeTensor, layers: usize, ops: usize) -> CubeTensor {
        let mut x = x;
        for _ in 0..layers {
            let mut h = B::float_matmul(x, w.clone());
            for _ in 0..ops {
                h = B::float_tanh(B::float_mul_scalar(h, 0.999.into()));
            }
            x = h;
        }
        x
    }

    /// The bytes of `x`, read to the host.
    fn read(x: &CubeTensor) -> Vec<u8> {
        let data = cubecl::future::block_on(B::float_into_data(x.clone())).unwrap();
        data.as_bytes().to_vec()
    }

    /// The bytes of the first element of `x`, read to the host.
    fn read_first(x: &CubeTensor) -> Vec<u8> {
        let first = Slice::new(0, Some(1), 1);
        read(&B::float_slice(x.clone(), &[first, first]))
    }

    /// Milliseconds per call of `next`, over `steps` calls after three untimed ones.
    fn time_steps<T>(steps: usize, mut next: impl FnMut() -> T) -> f64 {
        for _ in 0..3 {
            next();
        }
        let start = Instant::now();
        for _ in 0..steps {
            next();
        }
        start.elapsed().as_secs_f64() / steps as f64 * 1e3
    }

    /// Milliseconds per step, each step reading one element of its output on the host: run
    /// eagerly, and replayed from a CUDA graph recorded from one step (or why the graph could
    /// not be recorded). Before it is timed, the graph is run three steps with each output
    /// written back into the recorded input, and must give the bits three eager steps give.
    pub fn chain(layers: usize, ops: usize, steps: usize) -> (f64, Result<f64, String>) {
        let device = CubeDevice::from(CudaDevice::new(0));
        let random = |shape: [usize; 2]| {
            let normal = Distribution::Normal(0.0, 0.02);
            B::float_random(Shape::new(shape), normal, &device, FloatDType::BF16)
        };
        let w = random([1024, 1024]);
        let x = random([8, 1024]);
        let mut h = x.clone();
        let eager = time_steps(steps, || {
            h = step(h.clone(), &w, layers, ops);
            read_first(&h)
        });
        let expected = read(&(0..3).fold(x.clone(), |h, _| step(h, &w, layers, ops)));

        // The warm-up step leaves in the pools what the recorded one allocates. The recording
        // reads `x` and writes `y`; it runs nothing, so `x` still holds the first input.
        let client = device.client();
        let recorded = client
            .graph_prepare()
            .and_then(|()| {
                drop(step(x.clone(), &w, layers, ops));
                client.start_capture()
            })
            .and_then(|()| {
                let y = step(x.clone(), &w, layers, ops);
                client.stop_capture().map(|graph| (graph, y))
            });
        let (graph, y) = match recorded {
            Ok(recorded) => recorded,
            Err(err) => return (eager, Err(format!("{err}"))),
        };
        // SAFETY: `x`, `w`, `y` and the buffers the recording wrote stay alive while the graph
        // does, and are written only by it and below, on the stream it replays on.
        let replay = || unsafe { graph.replay() }.unwrap();
        for _ in 0..3 {
            replay();
            client.write(&x.handle, Bytes::from_bytes_vec(read(&y)));
        }
        assert!(
            read(&y) == expected,
            "the replayed chain differs from the eager one"
        );
        let replayed = time_steps(steps, || {
            replay();
            read_first(&y)
        });
        (eager, Ok(replayed))
    }
}

#[cfg(feature = "cuda")]
fn main() {
    let only: Vec<String> = std::env::args().skip(1).collect();
    let wanted = |phase: &str| only.is_empty() || only.iter().any(|s| s == phase);
    let device = Device::cuda(0);
    println!(
        "bf16, [8x1024]x[1024x1024] matmuls, on CUDA, fusion {}",
        if cfg!(feature = "fusion") {
            "on"
        } else {
            "off"
        }
    );
    if wanted("chain") {
        for (ops, label) in [(10, "24 x (matmul + 10 x (mul, tanh))"), (0, "24 x matmul")] {
            let read = chain(&device, 24, ops, 50, false);
            let lazy = chain(&device, 24, ops, 50, true);
            println!(
                "{label}: {read:.3} ms/step reading one element, {lazy:.3} ms/step issued \
                 ({:.1} and {:.1} us per matmul)",
                read * 1e3 / 24.0,
                lazy * 1e3 / 24.0
            );
        }
    }
    if wanted("readback") {
        println!("one-element read alone: {:.1} us", readback(&device, 200));
    }
    for (phase, dependent) in [("independent", false), ("dependent", true)] {
        if wanted(phase) {
            let (total, issued) = back_to_back(&device, 2000, dependent);
            println!(
                "2000 {phase} matmuls, one sync: {total:.1} us each, issued in {issued:.1} us each"
            );
        }
    }
    if wanted("graph") {
        for (ops, label) in [(10, "24 x (matmul + 10 x (mul, tanh))"), (0, "24 x matmul")] {
            let (eager, replayed) = graph::chain(24, ops, 50);
            match replayed {
                Ok(ms) => println!(
                    "{label}, no fusion, reading one element: {eager:.3} ms/step eager, \
                     {ms:.3} ms/step replayed from a CUDA graph"
                ),
                Err(err) => println!(
                    "{label}, no fusion, reading one element: {eager:.3} ms/step eager; \
                     no CUDA graph: {err}"
                ),
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("dispatch_bench measures the CUDA backend: run it with --features cuda,fusion");
}
