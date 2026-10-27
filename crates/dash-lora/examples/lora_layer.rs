//! Device time and memory of one LoRA'd linear layer, at the projection shapes of a language
//! model's blocks: `in -> out` of 1024 -> 3584 and 3584 -> 1024 (an MLP's up and down
//! projections at Qwen3.5-0.8B's widths), 1024 -> 2048 and 1024 -> 256 (attention projections),
//! r 16, alpha 32, a bf16 base and f32 adapters, over 2048 and 6144 tokens of bf16 input. For each
//! shape and length, these passes:
//!
//! - `base fwd`: the base product alone (the layer with no adapter chosen), outside autodiff;
//! - `low-rank fwd`: the adapter's term alone, `scale · (x Aᵀ) Bᵀ` in f32 (`Lora::forward`);
//! - `layer fwd`: the layer, base and adapter, outside autodiff, as in inference;
//! - `low-rank fwd+bwd`, `layer fwd+bwd`: the same with the adapter trained and `x` requiring a
//!   gradient, as a layer inside a model, and the backward pass from a gradient of ones;
//! - `two-adapter step`: two adapters over the one base, the frozen reference's output (`x` not
//!   tracked, as under `no_grad`), then the trained adapter's forward and backward passes from
//!   the gradient of `y - reference`.
//!
//! Every pass is first run at least three times and for 200 ms (kernels compiled, autotune's
//! candidates timed, pools grown, the device's clocks raised); then each is run [`REPS`] times,
//! each run followed by a device sync, and the time printed is the median of those, wall clock.
//! Device time is measured under Nsight Systems: the timed runs of each pass follow the warm-ups
//! and one another with the device idle for 100 ms between passes, so the kernels of a pass's
//! timed runs are those between two such gaps, one run's sequence repeated [`REPS`] times, and a
//! run's device time is the sum of its kernels' durations (the `PASS` lines say in which order the
//! passes run).
//!
//! Memory: `weights` is the device's `bytes_in_use` once one layer holding both adapters is built
//! (the base once, two adapters), beside two layers of one adapter each, as when the model is held
//! twice; `peak above` is the largest `bytes_in_use` during a pass's timed runs above the weights
//! and the input, sampled by a thread every 100 µs and at the points a pass goes through.
//! `choose` is the time of `clone().set_adapter(..)` on the layer, on the host.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-lora --features cuda --example lora_layer
//!   ... -- [tokens, e.g. 2048 6144]
//!
//! On an otherwise idle RTX 5070 Ti, device time in ms of each pass in the warm-up, which runs the
//! passes without gaps: the timed passes are [`GAP`] of idle device apart, too little work for the
//! card, which lowers its clocks within two seconds of them (memory 13.8 to 7.0 GHz), PyTorch's
//! passes losing more than these. This example built with `cuda` alone (Burn's default kernels) and
//! with `dash-ops/fusion` and `autotune`, beside PyTorch 2.13 with PEFT 0.20 on the same layer and
//! card (`lora_B(lora_A(x.float())) * scaling` on cuBLAS), each run in turn over three rounds, which
//! agree to 1% (2026-10-27):
//!
//! | `in -> out` | tokens | pass | PEFT | default | `fusion` and `autotune` |
//! |---|---|---|---:|---:|---:|
//! | 1024 -> 3584 | 2048 | layer fwd+bwd | 0.734 | 0.626 | 0.570 |
//! | 3584 -> 1024 | 2048 | layer fwd+bwd | 0.585 | 0.640 | 0.566 |
//! | 1024 -> 2048 | 2048 | layer fwd+bwd | 0.396 | 0.396 | 0.364 |
//! | 1024 -> 256 | 2048 | layer fwd+bwd | 0.105 | 0.730 | 0.118 |
//! | 1024 -> 3584 | 6144 | layer fwd+bwd | 2.73 | 8.41 | 1.73 |
//! | 3584 -> 1024 | 6144 | layer fwd+bwd | 2.02 | 8.50 | 1.57 |
//! | 1024 -> 2048 | 6144 | layer fwd+bwd | 1.53 | 4.85 | 1.01 |
//! | 1024 -> 256 | 6144 | layer fwd+bwd | 0.283 | 1.37 | 0.267 |
//! | 1024 -> 3584 | 2048 | two-adapter step | 1.10 | 0.907 | 0.863 |
//! | 3584 -> 1024 | 2048 | two-adapter step | 0.890 | 0.921 | 0.838 |
//! | 1024 -> 2048 | 2048 | two-adapter step | 0.596 | 0.561 | 0.518 |
//! | 1024 -> 256 | 2048 | two-adapter step | 0.153 | 1.39 | 0.163 |
//! | 1024 -> 3584 | 6144 | two-adapter step | 4.23 | 9.36 | 3.17 |
//! | 3584 -> 1024 | 6144 | two-adapter step | 2.94 | 16.0 | 2.44 |
//! | 1024 -> 2048 | 6144 | two-adapter step | 2.33 | 5.34 | 1.46 |
//! | 1024 -> 256 | 6144 | two-adapter step | 0.425 | 2.07 | 0.365 |
//! | 1024 -> 3584 | 2048 | low-rank fwd+bwd | 0.270 | 0.189 | 0.189 |
//! | 3584 -> 1024 | 2048 | low-rank fwd+bwd | 0.238 | 0.166 | 0.166 |
//! | 1024 -> 2048 | 2048 | low-rank fwd+bwd | 0.146 | 0.138 | 0.138 |
//! | 1024 -> 256 | 2048 | low-rank fwd+bwd | 0.0640 | 0.0700 | 0.0700 |
//! | 1024 -> 3584 | 6144 | low-rank fwd+bwd | 1.05 | 0.794 | 0.800 |
//! | 3584 -> 1024 | 6144 | low-rank fwd+bwd | 0.869 | 0.523 | 0.539 |
//! | 1024 -> 2048 | 6144 | low-rank fwd+bwd | 0.642 | 0.380 | 0.411 |
//! | 1024 -> 256 | 6144 | low-rank fwd+bwd | 0.193 | 0.141 | 0.141 |
//!
//! The low-rank products are `dash_ops::lowrank`'s kernels in every build, within 1.1x of PEFT's
//! and mostly ahead of them. Where the default build's layer is far behind, the base product is:
//! Burn's default bf16 matrix product, which on a card shared with other work took 1.0 ms for
//! `[2048, 1024] x [1024, 256]` (cuBLAS 0.02) and 8.8 ms for `[6144, 3584] x [3584, 1024]` (cuBLAS
//! 0.73), forward or for the input's gradient, where autotune found kernels for them that took 0.04
//! and 0.84 ms; on an idle card the second takes 7.2 ms by default and 0.59 with autotune (cuBLAS
//! 0.46, `dash-ops`' `matmul_bench`). Before the low-rank kernels,
//! with Burn's matrix products for them too, the default build's layer took 2.4 / 2.3 / 1.6 / 1.5
//! ms forward and backward at 2048 tokens (`in -> out` as above, an idle card; PEFT 0.73 / 0.58 /
//! 0.39 / 0.11 ms). Memory above the weights and inputs, at the peak of a pass: for 1024 -> 3584
//! over 2048 tokens 36 MiB forward and backward where PEFT's peak is 170 MiB (its f32 copy of `x`
//! and f32 `[N, out]` terms, and cuBLAS's workspace), over 6144 tokens 108 against 382 MiB.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use burn::module::Module;
use burn::tensor::{DType, Device, Tensor, TensorData};
use dash_lora::{AdapterChoice, Lora, LoraLinear};

/// `(in, out)` of the projections measured.
const SHAPES: [(usize, usize); 4] = [(1024, 3584), (3584, 1024), (1024, 2048), (1024, 256)];
const TOKENS: [usize; 2] = [2048, 6144];
const RANK: usize = 16;
const ALPHA: f64 = 32.0;
const REFERENCE: &str = "reference";
const WARMUP: usize = 3;
/// Timed runs of each pass.
const REPS: usize = 7;
/// The idle time after each pass's timed runs, which sets them apart in a trace. On an idle card
/// the timed passes this many apart lower the device's clocks within two seconds (module
/// documentation): their times run high, PyTorch's more than these.
const GAP: Duration = Duration::from_millis(100);
/// The least time each pass's warm-up runs for.
const WARM: Duration = Duration::from_millis(200);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    Base,
    LowRank,
    Forward,
    LowRankTrain,
    Train,
    TwoAdapters,
}

impl Pass {
    const ALL: [Pass; 6] = [
        Pass::Base,
        Pass::LowRank,
        Pass::Forward,
        Pass::LowRankTrain,
        Pass::Train,
        Pass::TwoAdapters,
    ];

    fn name(self) -> &'static str {
        match self {
            Pass::Base => "base fwd",
            Pass::LowRank => "low-rank fwd",
            Pass::Forward => "layer fwd",
            Pass::LowRankTrain => "low-rank fwd+bwd",
            Pass::Train => "layer fwd+bwd",
            Pass::TwoAdapters => "two-adapter step",
        }
    }
}

/// One layer and its input, on the autodiff device and, for the passes outside autodiff, on the
/// plain device (the same tensors).
struct Setup {
    /// "default" (trainable, chosen) and "reference" (frozen).
    layer: LoraLinear,
    /// The same layer choosing "reference".
    reference: LoraLinear,
    /// `layer.valid()`, and that with no adapter chosen.
    valid: LoraLinear,
    valid_base: LoraLinear,
    x: Tensor<2>,
    x_valid: Tensor<2>,
}

/// What a pass leaves to keep alive until the device has run it: outputs and gradients.
type Outputs = Vec<Tensor<2>>;

fn run(pass: Pass, s: &Setup) -> Outputs {
    match pass {
        Pass::Base => vec![s.valid_base.forward(s.x_valid.clone())],
        Pass::LowRank => vec![s.valid.lora().unwrap().forward(s.x_valid.clone())],
        Pass::Forward => vec![s.valid.forward(s.x_valid.clone())],
        Pass::LowRankTrain => {
            let lora = s.layer.lora().unwrap();
            let x = s.x.clone().require_grad();
            let grads = lora.forward(x.clone()).backward();
            gradients(&grads, &x, lora)
        }
        Pass::Train => {
            let x = s.x.clone().require_grad();
            let grads = s.layer.forward(x.clone()).backward();
            gradients(&grads, &x, s.layer.lora().unwrap())
        }
        Pass::TwoAdapters => {
            let reference = s.reference.forward(s.x.clone()).detach();
            let x = s.x.clone().require_grad();
            let grads = s.layer.forward(x.clone()).sub(reference).backward();
            gradients(&grads, &x, s.layer.lora().unwrap())
        }
    }
}

fn gradients(grads: &burn::tensor::Gradients, x: &Tensor<2>, lora: &Lora) -> Outputs {
    vec![
        x.grad(grads).expect("x requires a gradient"),
        lora.a.val().grad(grads).expect("A trains"),
        lora.b.val().grad(grads).expect("B trains"),
    ]
}

/// A layer `[out, in]` (bf16) with the trained adapter "default" and, `reference`, the frozen
/// "reference" beside it.
fn layer(inp: usize, out: usize, reference: bool, device: &Device, rng: &mut Rng) -> LoraLinear {
    let si = (inp as f64).sqrt();
    let weight = rng.tensor([out, inp], DType::BF16, device, |g| {
        g.uniform(-1.0, 1.0) / si
    });
    let mut layer = LoraLinear::frozen(weight).with_lora(adapter(inp, out, device, rng));
    if reference {
        layer.insert_adapter(REFERENCE, adapter(inp, out, device, rng).no_grad());
    }
    layer
}

/// An adapter for a `[out, in]` layer: A as PEFT's kaiming-uniform init, B small and non-zero.
fn adapter(inp: usize, out: usize, device: &Device, rng: &mut Rng) -> Lora {
    let si = (inp as f64).sqrt();
    let a = rng.tensor([RANK, inp], DType::F32, device, |g| {
        g.uniform(-1.0, 1.0) / si
    });
    let b = rng.tensor([out, RANK], DType::F32, device, |g| 1e-3 * g.normal());
    Lora::new(a, b, ALPHA / RANK as f64)
}

/// A seeded generator on the host (SplitMix64), so that the layer's values depend on the seed
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

/// Samples the device's pool usage from another thread and keeps the largest value in use.
struct Peak {
    in_use: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    device: Device,
}

impl Peak {
    fn start(device: &Device) -> Self {
        let (in_use, stop) = (
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicBool::new(false)),
        );
        let thread = {
            let (in_use, stop, device) = (in_use.clone(), stop.clone(), device.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if let Some(u) = device.memory_pool_usage() {
                        in_use.fetch_max(u.bytes_in_use, Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_micros(100));
                }
            })
        };
        Self {
            in_use,
            stop,
            thread: Some(thread),
            device: device.clone(),
        }
    }

    /// A sample from the main thread, at a point a pass goes through.
    fn mark(&self) {
        if let Some(u) = self.device.memory_pool_usage() {
            self.in_use.fetch_max(u.bytes_in_use, Ordering::Relaxed);
        }
    }

    fn stop(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap();
        self.in_use.load(Ordering::Relaxed)
    }
}

fn in_use(device: &Device) -> u64 {
    device.sync().unwrap();
    device
        .memory_pool_usage()
        .map(|u| u.bytes_in_use)
        .unwrap_or(0)
}

const MIB: f64 = 1024.0 * 1024.0;

/// Builds the layer at one shape and length, and prints what it holds against two layers.
fn build(inp: usize, out: usize, tokens: usize, device: &Device) -> (String, Setup) {
    let shape = format!("in {inp:>4} out {out:>4} tokens {tokens:>4}");
    let mut rng = Rng(0);

    // Two layers of one adapter each, as when a model is held twice, then dropped.
    let empty = in_use(device);
    let trees = {
        let one = layer(inp, out, false, device, &mut rng);
        let two = layer(inp, out, false, device, &mut rng);
        let bytes = in_use(device).saturating_sub(empty);
        drop((one, two));
        bytes
    };
    let empty = in_use(device);
    let layer = layer(inp, out, true, device, &mut rng);
    let weights = in_use(device).saturating_sub(empty);
    let base_bytes = (out * inp * 2) as f64;
    let adapter_bytes = (RANK * (inp + out) * 4) as f64;

    let n = 1000;
    let start = Instant::now();
    for i in 0..n {
        let choice = if i % 2 == 0 { Some(REFERENCE) } else { None };
        std::hint::black_box(layer.clone().set_adapter(choice));
    }
    let choose_us = start.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!(
        "{shape} | weights: one layer, two adapters {:.2} MiB in use (base {:.2} + 2 x {:.3}); \
         two layers {:.2} MiB | choose {choose_us:.2} us",
        weights as f64 / MIB,
        base_bytes / MIB,
        adapter_bytes / MIB,
        trees as f64 / MIB,
    );

    let x = Rng(1).tensor([tokens, inp], DType::BF16, device, Rng::normal);
    let setup = Setup {
        reference: layer.clone().set_adapter(Some(REFERENCE)),
        valid: layer.valid(),
        valid_base: layer.valid().set_adapter(None),
        x_valid: x.clone().inner(),
        x,
        layer,
    };
    (shape, setup)
}

/// Runs `pass` at least [`WARMUP`] times and for [`WARM`]: kernels compiled, autotune's
/// candidates timed, pools grown.
fn warm(pass: Pass, setup: &Setup, device: &Device) {
    let start = Instant::now();
    let mut runs = 0;
    while runs < WARMUP || start.elapsed() < WARM {
        let outputs = run(pass, setup);
        device.sync().unwrap();
        drop(outputs);
        runs += 1;
    }
}

/// Times [`REPS`] runs of `pass`, each followed by a sync, with the device idle for [`GAP`]
/// after them; prints the median and the peak memory above `held`.
fn time(shape: &str, pass: Pass, setup: &Setup, device: &Device, held: u64) {
    println!("PASS {shape} | {}", pass.name());
    let peak = Peak::start(device);
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        let outputs = run(pass, setup);
        peak.mark();
        device.sync().unwrap();
        times.push(start.elapsed().as_secs_f64() * 1e3);
        peak.mark();
        drop(outputs);
    }
    let peak = peak.stop();
    std::thread::sleep(GAP);
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{shape} | {:<16} | {:7.3} ms (median of {REPS}, min {:.3}) | peak {:7.1} MiB above \
         weights and inputs",
        pass.name(),
        times[REPS / 2],
        times[0],
        peak.saturating_sub(held) as f64 / MIB,
    );
}

fn main() {
    let tokens: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("token counts"))
        .collect();
    let tokens = if tokens.is_empty() {
        TOKENS.to_vec()
    } else {
        tokens
    };

    #[cfg(feature = "cuda")]
    let device = Device::cuda(0).autodiff();
    #[cfg(not(feature = "cuda"))]
    let device = Device::flex().autodiff();

    let setups: Vec<(String, Setup)> = tokens
        .iter()
        .flat_map(|&t| SHAPES.map(|(inp, out)| (inp, out, t)))
        .map(|(inp, out, t)| build(inp, out, t, &device))
        .collect();
    let held = in_use(&device);
    // Every pass warmed up first, so that nothing compiles or tunes among the timed runs, which
    // follow one another with the device idle only between passes.
    for (_, setup) in &setups {
        for pass in Pass::ALL {
            warm(pass, setup, &device);
        }
    }
    device.sync().unwrap();
    std::thread::sleep(GAP);
    for (shape, setup) in &setups {
        for pass in Pass::ALL {
            time(shape, pass, setup, &device, held);
        }
    }
}
