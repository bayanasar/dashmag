//! Times the low-rank product `scale · (x U) V` on CUDA, alone (f32 out) and with a base added
//! (bf16 out), forward and forward + backward: dash-ops' kernels against the same product built
//! from Burn's own operations (`x.cast(f32).matmul(u).matmul(v).mul_scalar(scale)`, then the f32
//! sum with the base and the cast back), which is what runs without them. The shapes are a
//! language model's projections: `in -> out` of 1024 -> 3584, 3584 -> 1024, 1024 -> 2048 and
//! 1024 -> 256, rank 16, over 2048 and 6144 tokens of bf16 input, the factors f32.
//!
//!   CUBECL_CUDA_TF32=0 cargo run --release -p dash-ops --features cuda --example lowrank_bench
//!
//! Each time is the median over 9 runs, each followed by a device sync, after 3 warm-up runs (which
//! also let Burn's autotuner settle where it is on); wall clock, on a shared machine. "fwd+bwd"
//! runs the backward pass from a gradient of ones, `x` and both factors requiring gradients.
//! `dash-lora`'s `examples/lora_layer.rs` measures the whole LoRA'd layer against PEFT by device
//! time.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("lowrank_bench needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    bench::run();
}

#[cfg(feature = "cuda")]
mod bench {
    use burn::tensor::{DType, Device, Tensor, TensorData};
    use dash_ops::lowrank::{LowRankOptions, low_rank};
    use std::time::Instant;

    const SHAPES: [(usize, usize); 4] = [(1024, 3584), (3584, 1024), (1024, 2048), (1024, 256)];
    const TOKENS: [usize; 2] = [2048, 6144];
    const RANK: usize = 16;
    const SCALE: f64 = 2.0;
    const WARMUP: usize = 3;
    const REPS: usize = 9;

    fn random(seed: u64, shape: [usize; 2], dtype: DType, device: &Device) -> Tensor<2> {
        let mut state = seed;
        let values: Vec<f32> = (0..shape[0] * shape[1])
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let unit = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64;
                (2.0 * unit - 1.0) as f32
            })
            .collect();
        let data = TensorData::new(values, shape).convert_dtype(dtype);
        Tensor::from_data(data, (device, dtype))
    }

    /// The product from Burn's operations, as `dash_lora` computed it before.
    fn composed(x: Tensor<2>, u: Tensor<2>, v: Tensor<2>, base: Option<Tensor<2>>) -> Tensor<2> {
        let z = x.cast(DType::F32).matmul(u).matmul(v).mul_scalar(SCALE);
        match base {
            None => z,
            Some(base) => {
                let dtype = base.dtype();
                base.cast(DType::F32).add(z).cast(dtype)
            }
        }
    }

    fn median(mut f: impl FnMut() -> Vec<Tensor<2>>, device: &Device) -> f64 {
        for _ in 0..WARMUP {
            let outputs = f();
            device.sync().unwrap();
            drop(outputs);
        }
        let mut times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let start = Instant::now();
            let outputs = f();
            device.sync().unwrap();
            times.push(start.elapsed().as_secs_f64() * 1e3);
            drop(outputs);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[REPS / 2]
    }

    pub fn run() {
        let device = Device::cuda(0).autodiff();
        println!(
            "{:<28} {:>11} {:>11} {:>11} {:>11}",
            "ms (median)", "fwd ops", "fwd Burn", "train ops", "train Burn"
        );
        for tokens in TOKENS {
            for (inp, out) in SHAPES {
                let x = random(1, [tokens, inp], DType::BF16, &device);
                let a = random(2, [RANK, inp], DType::F32, &device).mul_scalar(0.05);
                let b = random(3, [out, RANK], DType::F32, &device).mul_scalar(0.01);
                let base = random(4, [tokens, out], DType::BF16, &device);
                for with_base in [false, true] {
                    let options = LowRankOptions::default().with_scale(SCALE);
                    let base = with_base.then(|| base.clone());
                    let (xi, ui, vi) = (x.clone().inner(), a.clone().inner(), b.clone().inner());
                    let bi = base.clone().map(Tensor::inner);
                    let (ut, vt) = (ui.clone().transpose(), vi.clone().transpose());
                    let fwd_ops = median(
                        || {
                            vec![low_rank(
                                xi.clone(),
                                ut.clone(),
                                vt.clone(),
                                bi.clone(),
                                options,
                            )]
                        },
                        &device,
                    );
                    let fwd_burn = median(
                        || vec![composed(xi.clone(), ut.clone(), vt.clone(), bi.clone())],
                        &device,
                    );
                    let train = |ours: bool| {
                        let (x, a, b) = (
                            x.clone().require_grad(),
                            a.clone().require_grad(),
                            b.clone().require_grad(),
                        );
                        let (u, v) = (a.clone().transpose(), b.clone().transpose());
                        let y = match ours {
                            true => low_rank(x.clone(), u, v, base.clone(), options),
                            false => composed(x.clone(), u, v, base.clone()),
                        };
                        let grads = y.backward();
                        vec![
                            x.grad(&grads).unwrap(),
                            a.grad(&grads).unwrap(),
                            b.grad(&grads).unwrap(),
                        ]
                    };
                    let train_ops = median(|| train(true), &device);
                    let train_burn = median(|| train(false), &device);
                    let what = format!(
                        "{tokens} x {inp} -> {out}{}",
                        if with_base { " + base" } else { "" }
                    );
                    println!(
                        "{what:<28} {fwd_ops:>11.3} {fwd_burn:>11.3} {train_ops:>11.3} {train_burn:>11.3}"
                    );
                }
            }
        }
    }
}
