//! Peak device memory and time of one training step's forward and backward passes over a long
//! sequence through a stack of LoRA'd MLP blocks at Qwen3.5-0.8B's shapes (hidden 1024, MLP
//! 3584, 24 layers; r 16, alpha 32; bf16 base, f32 adapters), in three modes:
//!
//! - `none`: plain autodiff, one graph through every block;
//! - `balanced`: Burn's own gradient checkpointing (`Device::gradient_checkpointing()`), which
//!   recomputes memory-bound operations from backed-up compute-bound parents;
//! - `layer`: each block recomputed in the backward pass ([`dash_lora::recompute`]).
//!
//! One mode per process, so that memory pools do not carry over. Peak memory is the largest
//! `bytes_in_use` of the device's pools (live allocations, not padding or cached pages), sampled
//! by a thread every 100 µs and at the points the step passes through; `reserved` is the largest
//! the pools held. Both are above what the weights and the input take before the step starts,
//! also printed. The gradient checksum (sum of every adapter gradient's squares) lets the modes
//! be compared.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-lora --features cuda --example recompute_memory -- layer
//!   ... -- <none|balanced|layer> [layers 24] [tokens 6144] [timed repetitions 3]
//!
//! On an RTX 5070 Ti, 24 layers, 6144 tokens: the peak in use above the weights and input
//! (536 MiB), the largest over repeated runs (a sampled peak varies by some 60 MiB from run to
//! run), and the median step; PyTorch 2.13 with PEFT 0.20 on the same blocks and card beside it
//! (`torch.utils.checkpoint` around each block for `layer`; its peak is the allocator's exact
//! `max_memory_allocated`):
//!
//! | mode       | peak above | step, this workspace's build | with Burn's `autotune` | `autotune` + `fusion` | PyTorch          |
//! |------------|-----------:|-----------------------------:|-----------------------:|----------------------:|-----------------:|
//! | `none`     |   7347 MiB |                       981 ms |                 403 ms |                269 ms | 6415 MiB, 202 ms |
//! | `balanced` |  14595 MiB |     11.0 s (pools near full) |                        |                       |                  |
//! | `layer`    |    733 MiB |                      1340 ms |                 563 ms |                334 ms |  743 MiB, 292 ms |
//!
//! Without `autotune` two of the base products, M 6144, K 3584, N 1024, run a default kernel
//! at 7.2 ms each (6 TFLOP/s) where the others take 0.6 ms: half the step. The rest of the gap
//! to PyTorch is unfused casts and elementwise operations and the rank-16 f32 products.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use burn::module::Module;
use burn::optim::GradientsParams;
use burn::tensor::{DType, Device, Tensor, TensorData, activation};
use dash_lora::{Lora, LoraLinear, recompute};

const HIDDEN: usize = 1024;
const MLP: usize = 3584;
const RANK: usize = 16;
const ALPHA: f64 = 32.0;

/// `x + down(silu(gate(x)) * up(x))`, every projection LoRA'd.
#[derive(Module, Debug)]
struct Block {
    gate: LoraLinear,
    up: LoraLinear,
    down: LoraLinear,
}

impl Block {
    fn new(device: &Device, rng: &mut Rng) -> Self {
        let mut linear = |out: usize, inp: usize| {
            let si = (inp as f64).sqrt();
            let w = rng.tensor([out, inp], DType::BF16, device, |g| {
                g.uniform(-1.0, 1.0) / si
            });
            let a = rng.tensor([RANK, inp], DType::F32, device, |g| {
                g.uniform(-1.0, 1.0) / si
            });
            let b = rng.tensor([out, RANK], DType::F32, device, |g| 1e-3 * g.normal());
            LoraLinear::frozen(w).with_lora(Lora::new(a, b, ALPHA / RANK as f64))
        };
        Self {
            gate: linear(MLP, HIDDEN),
            up: linear(MLP, HIDDEN),
            down: linear(HIDDEN, MLP),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let h = activation::silu(self.gate.forward(x.clone())).mul(self.up.forward(x.clone()));
        x.add(self.down.forward(h))
    }
}

fn loss(out: Tensor<2>) -> Tensor<1> {
    out.cast(DType::F32).powi_scalar(2).mean()
}

/// A seeded generator on the host (SplitMix64), so that the stack's values depend on the seed
/// alone, not on Burn's process-wide generator.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn normal(&mut self) -> f64 {
        let u = 1.0 - self.unit();
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    fn tensor(
        &mut self,
        shape: [usize; 2],
        dtype: DType,
        device: &Device,
        mut draw: impl FnMut(&mut Self) -> f64,
    ) -> Tensor<2> {
        let values: Vec<f32> = (0..shape[0] * shape[1])
            .map(|_| draw(self) as f32)
            .collect();
        Tensor::from_data(
            TensorData::new(values, shape).convert_dtype(dtype),
            (device, dtype),
        )
    }
}

/// Samples the device's pool usage from another thread and keeps the largest values.
struct Peak {
    in_use: Arc<AtomicU64>,
    reserved: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    device: Device,
}

impl Peak {
    fn start(device: &Device) -> Self {
        let (in_use, reserved, stop) = (
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicBool::new(false)),
        );
        let thread = {
            let (in_use, reserved, stop, device) = (
                in_use.clone(),
                reserved.clone(),
                stop.clone(),
                device.clone(),
            );
            std::thread::spawn(move || {
                // DASH_LORA_SPIN=1 samples without pausing, for peaks shorter than 100 µs; it
                // slows the step down, so time is read from a run without it.
                let spin = std::env::var("DASH_LORA_SPIN").is_ok_and(|v| v == "1");
                while !stop.load(Ordering::Relaxed) {
                    if let Some(u) = device.memory_pool_usage() {
                        in_use.fetch_max(u.bytes_in_use, Ordering::Relaxed);
                        reserved.fetch_max(u.bytes_reserved, Ordering::Relaxed);
                    }
                    if !spin {
                        std::thread::sleep(Duration::from_micros(100));
                    }
                }
            })
        };
        Self {
            in_use,
            reserved,
            stop,
            thread: Some(thread),
            device: device.clone(),
        }
    }

    /// A sample from the main thread, at a point the step passes through.
    fn mark(&self) {
        if let Some(u) = self.device.memory_pool_usage() {
            self.in_use.fetch_max(u.bytes_in_use, Ordering::Relaxed);
            self.reserved.fetch_max(u.bytes_reserved, Ordering::Relaxed);
        }
    }

    fn reset(&self) {
        self.in_use.store(0, Ordering::Relaxed);
        self.reserved.store(0, Ordering::Relaxed);
    }

    fn stop(mut self) -> (u64, u64) {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap();
        (
            self.in_use.load(Ordering::Relaxed),
            self.reserved.load(Ordering::Relaxed),
        )
    }
}

/// One step's forward and backward passes; returns the adapters' gradients.
fn step(mode: &str, blocks: &[Block], x: &Tensor<2>, peak: &Peak) -> (f64, GradientsParams) {
    match mode {
        "none" | "balanced" => {
            let out = blocks.iter().fold(x.clone(), |h, b| b.forward(h));
            let loss = loss(out);
            peak.mark();
            let mut grads = loss.backward();
            peak.mark();
            let params = GradientsParams::from_module(&mut grads, &blocks.to_vec());
            (loss.into_scalar::<f32>() as f64, params)
        }
        "layer" => {
            let (out, saved) = recompute::forward(blocks, x.clone(), |b, h| b.forward(h));
            let loss = loss(out.clone());
            peak.mark();
            let head = loss.backward();
            let mut params = GradientsParams::new();
            recompute::backward(
                blocks,
                saved,
                out.grad(&head).unwrap(),
                |b, h| {
                    let y = b.forward(h);
                    peak.mark();
                    y
                },
                &mut params,
            );
            peak.mark();
            (loss.into_scalar::<f32>() as f64, params)
        }
        other => panic!("mode {other}: none, balanced or layer"),
    }
}

fn checksum(blocks: &[Block], params: &GradientsParams) -> f64 {
    let mut sum = 0.0;
    for b in blocks {
        for layer in [&b.gate, &b.up, &b.down] {
            let lora = layer.lora.as_ref().unwrap();
            for id in [lora.a.id, lora.b.id] {
                let g = params.get::<2>(id).expect("every adapter has a gradient");
                sum += g.powi_scalar(2).sum().into_scalar::<f32>() as f64;
            }
        }
    }
    sum
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("layer");
    let layers: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(24);
    let tokens: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(6144);
    let reps: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(3);

    #[cfg(feature = "cuda")]
    let base = Device::cuda(0);
    #[cfg(not(feature = "cuda"))]
    let base = Device::flex();
    let device = match mode {
        "balanced" => base.clone().autodiff().gradient_checkpointing(),
        _ => base.clone().autodiff(),
    };
    let mut rng = Rng(0);
    let blocks: Vec<Block> = (0..layers).map(|_| Block::new(&device, &mut rng)).collect();
    let x = rng.tensor([tokens, HIDDEN], DType::BF16, &device, Rng::normal);
    device.sync().unwrap();
    let before = device
        .memory_pool_usage()
        .map(|u| u.bytes_in_use)
        .unwrap_or(0);

    let peak = Peak::start(&device);
    // Warm-up: kernels compiled and tuned, pools grown.
    let (loss, params) = step(mode, &blocks, &x, &peak);
    let sum = checksum(&blocks, &params);
    drop(params);
    device.sync().unwrap();

    let mut times = Vec::with_capacity(reps);
    peak.reset();
    for _ in 0..reps {
        let start = Instant::now();
        let (_, params) = step(mode, &blocks, &x, &peak);
        // Every gradient computed and on the device before the clock stops.
        let _ = params.len();
        drop(params);
        device.sync().unwrap();
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let (in_use, reserved) = peak.stop();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    println!(
        "mode {mode:<8} layers {layers} tokens {tokens}: weights+input {:.0} MiB; peak in use {:.0} MiB \
         ({:.0} above), reserved {:.0} MiB; step {:.1} ms (median of {reps}, min {:.1}); loss {loss:.6}, \
         grad checksum {sum:.6e}",
        mib(before),
        mib(in_use),
        mib(in_use.saturating_sub(before)),
        mib(reserved),
        times[times.len() / 2],
        times[0],
    );
}
