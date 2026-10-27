//! Time and memory of one decoding step of the recurrent gated delta rule at batch 8, 16 heads,
//! Dk = Dv = 128 (one of Qwen3.5-0.8B's Gated DeltaNet layers), with the l2 norm of `q` and `k`.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example gated_delta_recurrent_bench
//!
//! For the step (one CubeCL kernel on CUDA) and for the reference built from Burn's tensor
//! operations, with f32 inputs and with bf16 `q`, `k`, `v`, `beta`:
//!
//! - one step: a step then a device sync, median and minimum over the repetitions; the latency
//!   of one token through one layer, launch and synchronisation included;
//! - back to back: steps threading the state, one sync at the end, per step; the launch cost is
//!   hidden behind the device's work only as far as the host keeps ahead of it. One layer's
//!   8 MiB state stays in the card's L2 cache from one step to the next;
//! - 18 layers in turn: back to back over 18 states (Qwen3.5-0.8B's Gated DeltaNet layers, 144
//!   MiB), so that each step finds its state in device memory, not in L2, as in decoding;
//! - memory (the kernel only): device bytes in use while a step's output and new state are
//!   alive, over the bytes its inputs hold, with the state handed over (written in place) and
//!   with the caller keeping the old state. The kernel allocates nothing else, so this is its
//!   peak; the reference frees its intermediates before it could be read, so it is not measured.
//!
//! `--only one-step|back-to-back|layers|layers-sync` runs one of those loops alone, with the
//! kernel and f32 inputs, for a profiler (`nsys profile ...`) to time the kernel in it;
//! `layers-sync` syncs after every step, so the device idles between steps as it does under a
//! host slower than the kernel.
//!
//! Without the `cuda` feature it runs on Flex (CPU), as a smoke test.

use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use dash_ops::gated_delta::recurrent::{
    GatedDeltaStepOptions, gated_delta_step, gated_delta_step_reference, probe,
};
use std::time::{Duration, Instant};

const B: usize = 8;
const H: usize = 16;
const DK: usize = 128;
const DV: usize = 128;

type Step = fn(
    Tensor<3>,
    Tensor<3>,
    Tensor<3>,
    Tensor<2>,
    Tensor<2>,
    Tensor<4>,
    Option<Tensor<1, Int>>,
    GatedDeltaStepOptions,
) -> (Tensor<3>, Tensor<4>);

/// Deterministic uniform values in [-1, 1).
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn tensor<const D: usize>(
    values: Vec<f32>,
    dims: [usize; D],
    dtype: DType,
    device: &Device,
) -> Tensor<D> {
    let tensor: Tensor<D> = Tensor::from_data(TensorData::new(values, dims), (device, DType::F32));
    tensor.cast(dtype)
}

struct Inputs {
    q: Tensor<3>,
    k: Tensor<3>,
    v: Tensor<3>,
    g: Tensor<2>,
    beta: Tensor<2>,
}

impl Inputs {
    fn new(dtype: DType, device: &Device) -> Self {
        let g = values(B * H, 4)
            .into_iter()
            .map(|x| -0.1 - x.abs())
            .collect();
        let beta = values(B * H, 5)
            .into_iter()
            .map(|x| 0.5 + 0.4 * x)
            .collect();
        Self {
            q: tensor(values(B * H * DK, 1), [B, H, DK], dtype, device),
            k: tensor(values(B * H * DK, 2), [B, H, DK], dtype, device),
            v: tensor(values(B * H * DV, 3), [B, H, DV], dtype, device),
            g: tensor(g, [B, H], DType::F32, device),
            beta: tensor(beta, [B, H], dtype, device),
        }
    }

    fn step(&self, step: Step, state: Tensor<4>) -> (Tensor<3>, Tensor<4>) {
        step(
            self.q.clone(),
            self.k.clone(),
            self.v.clone(),
            self.g.clone(),
            self.beta.clone(),
            state,
            None,
            GatedDeltaStepOptions::default().with_qk_l2norm(true),
        )
    }
}

fn state(device: &Device) -> Tensor<4> {
    tensor(
        values(B * H * DK * DV, 6),
        [B, H, DK, DV],
        DType::F32,
        device,
    )
    .mul_scalar(0.1)
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// One step then a sync, `reps` times: (median, minimum) in microseconds.
fn one_step(inputs: &Inputs, step: Step, device: &Device, reps: usize) -> (f64, f64) {
    let mut state = state(device);
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let (out, next) = inputs.step(step, state);
        device.sync().unwrap();
        times.push(micros(start.elapsed()));
        drop(out);
        state = next;
    }
    times.sort_by(f64::total_cmp);
    (times[reps / 2], times[0])
}

/// `steps` steps threading the state, one sync at the end: microseconds per step, best of
/// `rounds`.
fn back_to_back(inputs: &Inputs, step: Step, device: &Device, steps: usize, rounds: usize) -> f64 {
    let mut best = f64::INFINITY;
    let mut state = state(device);
    for _ in 0..rounds {
        device.sync().unwrap();
        let start = Instant::now();
        for _ in 0..steps {
            let (out, next) = inputs.step(step, state);
            drop(out);
            state = next;
        }
        device.sync().unwrap();
        best = best.min(micros(start.elapsed()) / steps as f64);
    }
    best
}

/// Steps 18 states in turn (the Gated DeltaNet layers of Qwen3.5-0.8B), `rounds` times, one
/// sync at the end (or after every step): microseconds per step, best of `repeats`.
fn layers_in_turn(
    inputs: &Inputs,
    step: Step,
    device: &Device,
    rounds: usize,
    repeats: usize,
    sync_each: bool,
) -> f64 {
    const LAYERS: usize = 18;
    let mut states: Vec<Option<Tensor<4>>> = (0..LAYERS).map(|_| Some(state(device))).collect();
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        device.sync().unwrap();
        let start = Instant::now();
        for _ in 0..rounds {
            for slot in states.iter_mut() {
                let (out, next) = inputs.step(step, slot.take().unwrap());
                drop(out);
                *slot = Some(next);
                if sync_each {
                    device.sync().unwrap();
                }
            }
        }
        device.sync().unwrap();
        best = best.min(micros(start.elapsed()) / (rounds * LAYERS) as f64);
    }
    best
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Device bytes in use with a step's results alive, over those its inputs hold.
fn memory(inputs: &Inputs, step: Step, device: &Device, keep_old_state: bool) -> Option<f64> {
    let state = state(device);
    device.sync().unwrap();
    let before = device.memory_pool_usage()?.bytes_in_use;
    let kept = keep_old_state.then(|| state.clone());
    let (out, next) = inputs.step(step, state);
    device.sync().unwrap();
    let during = device.memory_pool_usage()?.bytes_in_use;
    drop((out, next, kept));
    Some(mib(during.saturating_sub(before)))
}

fn main() {
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(not(feature = "cuda"))]
    let device = Device::flex();
    let (reps, steps, rounds, layer_rounds) = if cfg!(feature = "cuda") {
        (500, 2000, 5, 100)
    } else {
        (5, 5, 1, 1)
    };

    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [flag, only] = args.as_slice()
        && flag == "--only"
    {
        let inputs = Inputs::new(DType::F32, &device);
        let step = gated_delta_step as Step;
        let _ = back_to_back(&inputs, step, &device, 50, 1);
        let result = match only.as_str() {
            "one-step" => one_step(&inputs, step, &device, 300).0,
            "back-to-back" => back_to_back(&inputs, step, &device, 300, 1),
            "layers" => layers_in_turn(&inputs, step, &device, 20, 1, false),
            "layers-sync" => layers_in_turn(&inputs, step, &device, 20, 1, true),
            other => {
                panic!("--only takes one-step, back-to-back, layers or layers-sync, not {other}")
            }
        };
        println!("{only}: {result:.2} us per step");
        return;
    }

    println!(
        "gated delta rule, one step: B {B}, H {H}, Dk {DK}, Dv {DV}, l2norm; state {:.1} MiB f32",
        mib((B * H * DK * DV * 4) as u64)
    );
    for dtype in [DType::F32, DType::BF16] {
        let inputs = Inputs::new(dtype, &device);
        for (name, step) in [
            ("kernel", gated_delta_step as Step),
            ("reference", gated_delta_step_reference as Step),
        ] {
            // Warm-up: compile the kernels, fill the memory pools.
            let launches = probe::kernel_launches();
            let _ = back_to_back(&inputs, step, &device, 50, 1);
            let ran_kernel = probe::kernel_launches() > launches;
            let (median, min) = one_step(&inputs, step, &device, reps);
            let per_step = back_to_back(&inputs, step, &device, steps, rounds);
            let per_layer = layers_in_turn(&inputs, step, &device, layer_rounds, rounds, false);
            let in_place = memory(&inputs, step, &device, false);
            let kept = memory(&inputs, step, &device, true);
            let mem = match (name, in_place, kept) {
                ("reference", ..) => String::new(),
                (_, Some(a), Some(b)) => {
                    format!("; +{a:.2} MiB (state handed over), +{b:.2} MiB (old state kept)")
                }
                _ => "; memory not reported by this device".into(),
            };
            println!(
                "{dtype:?} {name:9} (kernel ran: {ran_kernel}): one step {median:7.1} us median, \
                 {min:7.1} us min; back to back {per_step:7.2} us/step; 18 layers in turn \
                 {per_layer:7.2} us/step{mem}"
            );
        }
    }
}
