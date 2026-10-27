//! Device memory and time of two adapters over one base, against a module tree per adapter: a
//! stack of LoRA'd MLP blocks at Qwen3.5-0.8B's shapes (hidden 1024, MLP 3584, 24 layers; r 16,
//! alpha 32; bf16 base, f32 adapters), trained against a frozen reference adapter. One training
//! step: the reference's output over a sequence, recorded by nothing, then the trained adapter's
//! forward and backward passes, the loss the mean squared difference of the two outputs. Modes:
//!
//! - `one`: one tree, one adapter (the reference step runs the base, `set_adapter(None)`);
//! - `shared`: one tree, the trained adapter and the frozen reference over one base, the
//!   reference's output from a clone choosing it (`set_adapter(Some("reference"))`);
//! - `trees`: two trees, each with a base of its own, the reference's tree run for its output,
//!   as when the model is built or read twice;
//! - `trees-shared`: two trees sharing the base tensors (one a clone of the other with its
//!   adapter replaced): what holding two adapters took before, with the caller choosing the tree.
//!
//! One mode per process, so that memory pools do not carry over. `weights+input` is the device's
//! `bytes_in_use` once the model and the input are built (live allocations), printed beside what
//! the tensors take; `peak above` is the largest `bytes_in_use` during the timed steps above that,
//! sampled by a thread every 100 µs and at the points the step passes through; the step time is
//! the median over the repetitions. `choose` is the time of `clone().set_adapter(...)` on the
//! whole stack, on the host.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-lora --features cuda --example adapters_memory -- shared
//!   ... -- <one|shared|trees|trees-shared> [layers 24] [tokens 2048] [timed repetitions 5]
//!
//! On an RTX 5070 Ti, 24 layers, 2048 tokens, this workspace's build (Burn's default kernels);
//! PyTorch 2.13 with PEFT 0.20 on the same blocks and card beside it (one PEFT model holding both
//! adapters, `set_adapter` around a `no_grad` reference pass as TRL's trainers switch adapters;
//! two PEFT models for `trees`; its memory the allocator's `memory_allocated` and
//! `max_memory_allocated`). Device time is the sum of the step's kernels under nsys; the step, the
//! median wall-clock time, taken while other work shared the card and the host.
//!
//! | mode           | weights+input | peak above | device time | step   | PyTorch: weights+input | peak above | device time | step  |
//! |----------------|--------------:|-----------:|------------:|-------:|-----------------------:|-----------:|------------:|------:|
//! | `one`          |     528.2 MiB |   2449 MiB |      205 ms | 212 ms |              528.2 MiB |   2185 MiB |       65 ms | 74 ms |
//! | `shared`       |     548.5 MiB |   2441 MiB |      249 ms | 258 ms |              548.5 MiB |   2185 MiB |       77 ms | 86 ms |
//! | `trees`        |    1052.5 MiB |   2441 MiB |      249 ms | 257 ms |             1052.5 MiB |   2185 MiB |       78 ms | 85 ms |
//! | `trees-shared` |     548.5 MiB |   2433 MiB |             | 257 ms |                        |            |             |       |
//!
//! The weights in use are the tensors' bytes exactly, as in PyTorch: the bf16 base once (504 MiB)
//! however many adapters it carries, 20.25 MiB per f32 adapter; two trees hold the base twice. The
//! second adapter costs no device time either: `shared` and `trees` run the same 2858 kernels a
//! step, 249.1 ms of device time each, and `trees-shared` takes their wall-clock time; `one` runs
//! its reference pass on the base alone, without the low-rank products. Choosing for the whole
//! stack takes 22 to 29 µs on the host; PEFT's `set_adapter` on the same stack, 1.3 to 1.5 ms.
//!
//! The gap to PyTorch is the layer's, with one adapter as with two: of the 249 ms, the 574 rank-16
//! f32 products take 158 ms (0.28 ms each, Burn's default matrix-product kernel on these skinny
//! shapes; cuBLAS takes 15 ms for them), the 214 bf16 base products 46 ms (cuBLAS, 35), the
//! elementwise rest 45 ms (PyTorch, 27). With Burn's `autotune` and `fusion` the step takes
//! 105 ms for `one` and 117 to 121 ms for `shared`, `trees` and `trees-shared` alike in most
//! runs (ten timed steps each), where PyTorch takes 74 and 85; on the loaded host some runs of
//! every mode took two to three times as long.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use burn::module::Module;
use burn::optim::GradientsParams;
use burn::tensor::{DType, Device, Tensor, TensorData, activation};
use dash_lora::{AdapterChoice, Lora, LoraLinear};

const HIDDEN: usize = 1024;
const MLP: usize = 3584;
const RANK: usize = 16;
const ALPHA: f64 = 32.0;
const REFERENCE: &str = "reference";

/// `x + down(silu(gate(x)) * up(x))`, every projection LoRA'd.
#[derive(Module, Debug)]
struct Block {
    gate: LoraLinear,
    up: LoraLinear,
    down: LoraLinear,
}

impl Block {
    fn frozen(device: &Device, rng: &mut Rng) -> Self {
        let mut linear = |out: usize, inp: usize| {
            let si = (inp as f64).sqrt();
            LoraLinear::frozen(rng.tensor([out, inp], DType::BF16, device, |g| {
                g.uniform(-1.0, 1.0) / si
            }))
        };
        Self {
            gate: linear(MLP, HIDDEN),
            up: linear(MLP, HIDDEN),
            down: linear(HIDDEN, MLP),
        }
    }

    fn layers_mut(&mut self) -> [&mut LoraLinear; 3] {
        [&mut self.gate, &mut self.up, &mut self.down]
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let h = activation::silu(self.gate.forward(x.clone())).mul(self.up.forward(x.clone()));
        x.add(self.down.forward(h))
    }
}

fn forward(blocks: &[Block], x: &Tensor<2>) -> Tensor<2> {
    blocks.iter().fold(x.clone(), |h, b| b.forward(h))
}

/// An adapter for a `[out, in]` layer: A as PEFT's kaiming-uniform init, B small and non-zero.
fn adapter(layer: &LoraLinear, device: &Device, rng: &mut Rng) -> Lora {
    let (out, inp) = (layer.out_features(), layer.in_features());
    let si = (inp as f64).sqrt();
    let a = rng.tensor([RANK, inp], DType::F32, device, |g| {
        g.uniform(-1.0, 1.0) / si
    });
    let b = rng.tensor([out, RANK], DType::F32, device, |g| 1e-3 * g.normal());
    Lora::new(a, b, ALPHA / RANK as f64)
}

/// Gives every layer of `blocks` an adapter named `name`, trainable or frozen.
fn add_adapters(blocks: &mut [Block], name: &str, trainable: bool, device: &Device, rng: &mut Rng) {
    for block in blocks {
        for layer in block.layers_mut() {
            let lora = adapter(layer, device, rng);
            let lora = if trainable { lora } else { lora.no_grad() };
            layer.insert_adapter(name, lora);
        }
    }
}

/// The trees a mode trains: the trained one, and the reference's if it has one of its own.
struct Model {
    policy: Vec<Block>,
    reference: Option<Vec<Block>>,
}

fn build(mode: &str, layers: usize, device: &Device) -> Model {
    let mut rng = Rng(0);
    let mut policy: Vec<Block> = (0..layers)
        .map(|_| Block::frozen(device, &mut rng))
        .collect();
    add_adapters(
        &mut policy,
        dash_lora::DEFAULT_ADAPTER,
        true,
        device,
        &mut rng,
    );
    match mode {
        "one" => Model {
            policy,
            reference: None,
        },
        "shared" => {
            add_adapters(&mut policy, REFERENCE, false, device, &mut rng);
            Model {
                policy,
                reference: None,
            }
        }
        "trees" => {
            // A second base, equal in value, and the reference as its single adapter.
            let mut rng_base = Rng(0);
            let mut reference: Vec<Block> = (0..layers)
                .map(|_| Block::frozen(device, &mut rng_base))
                .collect();
            for block in &mut reference {
                for layer in block.layers_mut() {
                    let lora = adapter(layer, device, &mut rng).no_grad();
                    layer.set_lora(lora);
                }
            }
            Model {
                policy,
                reference: Some(reference),
            }
        }
        "trees-shared" => {
            // The same base tensors, the reference as the clone's single adapter.
            let mut reference = policy.clone();
            for block in &mut reference {
                for layer in block.layers_mut() {
                    let lora = adapter(layer, device, &mut rng).no_grad();
                    layer.set_lora(lora);
                }
            }
            Model {
                policy,
                reference: Some(reference),
            }
        }
        other => panic!("mode {other}: one, shared, trees or trees-shared"),
    }
}

/// One step's forward and backward passes; returns the loss and the trained adapter's gradients.
fn step(mode: &str, model: &Model, x: &Tensor<2>, peak: &Peak) -> (f64, GradientsParams) {
    let reference = match (&model.reference, mode) {
        (Some(tree), _) => forward(tree, x),
        (None, "shared") => forward(&model.policy.clone().set_adapter(Some(REFERENCE)), x),
        (None, _) => forward(&model.policy.clone().set_adapter(None), x),
    }
    .detach();
    peak.mark();
    let out = forward(&model.policy, x);
    let loss = out
        .cast(DType::F32)
        .sub(reference.cast(DType::F32))
        .powi_scalar(2)
        .mean();
    peak.mark();
    let grads = loss.backward();
    peak.mark();
    let params = GradientsParams::from_grads(grads, &model.policy);
    (loss.into_scalar::<f32>() as f64, params)
}

/// What the model's tensors take: the bf16 bases (one per tree that has its own) and the f32
/// adapters, in bytes.
fn tensor_bytes(mode: &str, layers: usize) -> (u64, u64) {
    // Per block, three [out, in] weights of HIDDEN x MLP; each adapter's A [r, in] and B [out, r].
    let base = (layers * 3 * HIDDEN * MLP * 2) as u64;
    let adapter = (layers * 3 * RANK * (HIDDEN + MLP) * 4) as u64;
    let bases = if mode == "trees" { 2 } else { 1 };
    let adapters = if mode == "one" { 1 } else { 2 };
    (bases * base, adapters * adapter)
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

    /// A sample from the main thread, at a point the step passes through.
    fn mark(&self) {
        if let Some(u) = self.device.memory_pool_usage() {
            self.in_use.fetch_max(u.bytes_in_use, Ordering::Relaxed);
        }
    }

    fn reset(&self) {
        self.in_use.store(0, Ordering::Relaxed);
    }

    fn stop(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap();
        self.in_use.load(Ordering::Relaxed)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("shared")
        .to_string();
    let layers: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(24);
    let tokens: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(2048);
    let reps: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(5);

    #[cfg(feature = "cuda")]
    let device = Device::cuda(0).autodiff();
    #[cfg(not(feature = "cuda"))]
    let device = Device::flex().autodiff();

    let model = build(&mode, layers, &device);
    let x = Rng(1).tensor([tokens, HIDDEN], DType::BF16, &device, Rng::normal);
    device.sync().unwrap();
    let weights = device
        .memory_pool_usage()
        .map(|u| u.bytes_in_use)
        .unwrap_or(0);

    // Choosing an adapter for the whole stack, on the host.
    let n = 1000;
    let start = Instant::now();
    for i in 0..n {
        let name = if i % 2 == 0 { Some(REFERENCE) } else { None };
        let chosen = model.policy.clone().set_adapter(match mode.as_str() {
            "shared" => name,
            _ => None,
        });
        std::hint::black_box(&chosen);
    }
    let choose_us = start.elapsed().as_secs_f64() * 1e6 / n as f64;

    let peak = Peak::start(&device);
    // Warm-up: kernels compiled, pools grown.
    let (loss, params) = step(&mode, &model, &x, &peak);
    let trained = params.len();
    drop(params);
    device.sync().unwrap();
    peak.reset();
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let (_, params) = step(&mode, &model, &x, &peak);
        drop(params);
        device.sync().unwrap();
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let in_use = peak.stop();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    let (base, adapters) = tensor_bytes(&mode, layers);
    let input = (tokens * HIDDEN * 2) as u64;
    println!(
        "mode {mode:<12} layers {layers} tokens {tokens}: weights+input {:.1} MiB in use (tensors: \
         bases {:.1} + adapters {:.1} + input {:.1} = {:.1} MiB); peak {:.0} MiB above; step {:.1} ms \
         (median of {reps}, min {:.1}); choose {choose_us:.1} us; {trained} trained factors; loss {loss:.6e}",
        mib(weights),
        mib(base),
        mib(adapters),
        mib(input),
        mib(base + adapters + input),
        mib(in_use.saturating_sub(weights)),
        times[times.len() / 2],
        times[0],
    );
}
