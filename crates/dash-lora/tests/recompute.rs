//! A stack of LoRA'd blocks trained with each block recomputed in the backward pass gives the
//! gradients plain autodiff gives: every adapter's, and the stack input's.
//!
//! In f32 they are bit-identical, on Flex and on CUDA, and under Burn's own checkpointing too
//! (on CUDA with Burn's autotune, within a few f32 roundings: see `same_computation`).
//! In bf16 they are not, and need not be: Burn adds up a tensor's gradient contributions as
//! they arrive, a block's input has five consumers, and the per-block graphs reach them in a
//! different order than the whole-stack graph, so their bf16 sums round differently; the
//! difference then travels down the stack (the top block is bit-identical). Both are as far
//! from the same stack computed in f32, which is what is asserted, over several random stacks.
//! The stacks are drawn on the host from a seeded generator, not from Burn's process-wide one,
//! so they do not depend on which other tests run at the same time.
//!
//!   cargo test -p dash-lora --test recompute -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test -p dash-lora --features cuda --test recompute -- --nocapture

mod common;

use burn::module::Module;
use burn::optim::GradientsParams;
use burn::tensor::{DType, Device, Tensor, TensorData, activation};
use common::*;
use dash_lora::{Lora, LoraLinear, recompute};

/// `x + down(silu(gate(x)) * up(x))`, every projection LoRA'd: a language model's MLP block.
#[derive(Module, Debug)]
struct Block {
    gate: LoraLinear,
    up: LoraLinear,
    down: LoraLinear,
}

impl Block {
    fn new(d: usize, inner: usize, r: usize, dtype: DType, device: &Device, rng: &mut Rng) -> Self {
        let mut linear = |out: usize, inp: usize| {
            let (si, sr) = ((inp as f64).sqrt(), (r as f64).sqrt());
            let w = rng.tensor([out, inp], dtype, device, |g| g.uniform(-1.0, 1.0) / si);
            let a = rng.tensor([r, inp], DType::F32, device, |g| g.uniform(-1.0, 1.0) / si);
            let b = rng.tensor([out, r], DType::F32, device, |g| g.uniform(-1.0, 1.0) / sr);
            LoraLinear::frozen(w).with_lora(Lora::new(a, b, 2.0))
        };
        Self {
            gate: linear(inner, d),
            up: linear(inner, d),
            down: linear(d, inner),
        }
    }

    /// The same block with its base weights upcast to f32 (exactly) and the same adapters.
    fn to_f32(&self) -> Self {
        let f32 = |l: &LoraLinear| {
            let lora = l.lora.as_ref().unwrap();
            LoraLinear::frozen(l.weight.val().cast(DType::F32)).with_lora(Lora::new(
                lora.a.val(),
                lora.b.val(),
                lora.scale,
            ))
        };
        Self {
            gate: f32(&self.gate),
            up: f32(&self.up),
            down: f32(&self.down),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let g = self.gate.forward(x.clone());
        let dtype = g.dtype();
        let h = activation::silu(g.cast(DType::F32))
            .cast(dtype)
            .mul(self.up.forward(x.clone()));
        x.add(self.down.forward(h))
    }

    fn loras(&self) -> [&Lora; 3] {
        [&self.gate, &self.up, &self.down].map(|l| l.lora.as_ref().unwrap())
    }
}

fn loss(out: Tensor<2>) -> Tensor<1> {
    out.cast(DType::F32).powi_scalar(2).mean()
}

/// Loss, input gradient, and every adapter gradient in block order (gate, up, down; A, B).
struct Result {
    loss: TensorData,
    dx: TensorData,
    grads: Vec<TensorData>,
}

fn in_order(blocks: &[Block], params: &GradientsParams) -> Vec<TensorData> {
    let mut out = Vec::new();
    for b in blocks {
        for lora in b.loras() {
            for id in [lora.a.id, lora.b.id] {
                out.push(
                    params
                        .get::<2>(id)
                        .expect("every adapter has a gradient")
                        .into_data(),
                );
            }
        }
    }
    out
}

/// How many distinct adapter parameters the stack has (a block may appear twice).
fn distinct(blocks: &[Block]) -> usize {
    let mut ids: Vec<_> = blocks
        .iter()
        .flat_map(|b| b.loras().into_iter().flat_map(|l| [l.a.id, l.b.id]))
        .collect();
    ids.sort();
    ids.dedup();
    ids.len()
}

fn plain(blocks: &[Block], x: &Tensor<2>) -> Result {
    let x = x.clone().detach().require_grad();
    let loss = loss(blocks.iter().fold(x.clone(), |h, b| b.forward(h)));
    let mut grads = loss.backward();
    let dx = x.grad(&grads).unwrap().into_data();
    let params = GradientsParams::from_module(&mut grads, &blocks.to_vec());
    assert_eq!(
        params.len(),
        distinct(blocks),
        "only the adapters have gradients"
    );
    Result {
        loss: loss.into_data(),
        dx,
        grads: in_order(blocks, &params),
    }
}

fn recomputed(blocks: &[Block], x: &Tensor<2>) -> Result {
    let (out, saved) = recompute::forward(blocks, x.clone(), |b, h| b.forward(h));
    assert_eq!(saved.len(), blocks.len());
    let loss = loss(out.clone());
    let head = loss.backward();
    let mut params = GradientsParams::new();
    let dx = recompute::backward(
        blocks,
        saved,
        out.grad(&head).unwrap(),
        |b, h| b.forward(h),
        &mut params,
    );
    assert_eq!(params.len(), distinct(blocks));
    Result {
        loss: loss.into_data(),
        dx: dx.into_data(),
        grads: in_order(blocks, &params),
    }
}

fn label(i: usize) -> String {
    let (block, rest) = (i / 6, i % 6);
    let proj = ["gate", "up", "down"][rest / 2];
    format!("block {block} {proj} {}", ["A", "B"][rest % 2])
}

/// Four blocks and an input, drawn from `seed` on the host.
fn stack(device: &Device, dtype: DType, seed: u64) -> (Vec<Block>, Tensor<2>) {
    let (d, inner, r, t, n) = (32, 96, 4, 24, 4);
    let mut rng = Rng::new(seed);
    let blocks: Vec<Block> = (0..n)
        .map(|_| Block::new(d, inner, r, dtype, device, &mut rng))
        .collect();
    let x = rng.tensor([t, d], dtype, device, Rng::normal);
    (blocks, x)
}

/// Root-mean-square error of `actual` against `expected`, over `expected`'s root mean square.
fn rms_rel(actual: &TensorData, expected: &TensorData) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (a, e) in actual.iter::<f64>().zip(expected.iter::<f64>()) {
        num += (a - e) * (a - e);
        den += e * e;
    }
    (num / den).sqrt()
}

/// How much farther from the same stack computed in f32 the recomputed bf16 gradients are than
/// plain autodiff's, for the stack drawn from `seed`: the largest ratio over the input gradient
/// and the 24 adapter gradients of their largest-element errors (each over its tensor's largest
/// value), and the ratio of their root-mean-square errors summed over those 25 tensors.
fn bf16_ratios(name: &str, device: &Device, seed: u64, verbose: bool) -> (f64, f64) {
    let (blocks, x) = stack(device, DType::BF16, seed);
    let (p, r) = (plain(&blocks, &x), recomputed(&blocks, &x));
    let exact = plain(
        &blocks.iter().map(Block::to_f32).collect::<Vec<_>>(),
        &x.clone().cast(DType::F32),
    );
    let (abs, rel) = same_computation(name);
    check(
        &format!("{name} bf16 seed {seed} loss"),
        &r.loss,
        &p.loss,
        abs,
        rel,
    );
    let (mut worst, mut rms_r, mut rms_p) = (0f64, 0f64, 0f64);
    let pairs = std::iter::once(("input gradient".to_string(), &r.dx, &p.dx, &exact.dx)).chain(
        r.grads
            .iter()
            .zip(&p.grads)
            .zip(&exact.grads)
            .enumerate()
            .map(|(i, ((r, p), e))| (label(i), r, p, e)),
    );
    for (what, r, p, e) in pairs {
        let (er, ep) = (err(r, e).rel, err(p, e).rel);
        if verbose {
            println!("  bf16 {what:<16} vs f32: recomputed {er:.2e}, plain {ep:.2e}");
        }
        worst = worst.max(er / ep);
        rms_r += rms_rel(r, e);
        rms_p += rms_rel(p, e);
    }
    (worst, rms_r / rms_p)
}

/// How far the recomputed stack may be from plain autodiff when both compute the same thing,
/// relative to the largest value: nothing on Flex, a few f32 roundings on CUDA. On CUDA they
/// agree bit for bit with Burn's default kernels and with `burn/fusion`, but with
/// `burn/autotune` the two passes need not run the same kernels: autotune may settle on a
/// different fused kernel for the whole-stack backward pass than for the per-block ones (their
/// operation streams differ), and while one thread tunes a matrix product of a given shape,
/// another asking for the same shape runs an untuned candidate. Measured over sixteen runs
/// each of autotune with and without fusion, half with the tests in parallel: adapter
/// gradients up to 7.7e-7, the input gradient 3.8e-7, the bf16 stack's loss 1.2e-7 (one ulp);
/// the f32 stack's loss never differed.
fn same_computation(device_name: &str) -> (f64, f64) {
    match device_name {
        "flex" => (0.0, 0.0),
        _ => (f64::INFINITY, 1e-6),
    }
}

#[test]
fn recomputed_f32_stack_gives_plain_autodiff_s_gradients() {
    for (name, device) in devices() {
        let (blocks, x) = stack(&device, DType::F32, 7);
        let (p, r) = (plain(&blocks, &x), recomputed(&blocks, &x));
        let (abs, rel) = same_computation(name);
        check(&format!("{name} f32 loss"), &r.loss, &p.loss, abs, rel);
        check(
            &format!("{name} f32 input gradient"),
            &r.dx,
            &p.dx,
            abs,
            rel,
        );
        for (i, (r, p)) in r.grads.iter().zip(&p.grads).enumerate() {
            check(&format!("{name} f32 {}", label(i)), r, p, abs, rel);
        }
    }
}

/// Over 64 stacks (`bf16_ratios_over_many_seeds`; Flex and CUDA give the same minimum, median
/// and maximum) the summed root-mean-square ratio has a median of 1.00 and ranges from 0.83 to
/// 1.29: recomputing costs no accuracy, it only rounds differently. The largest element-error
/// ratio of a single tensor is noisier, median 1.41, up to 2.46. The bounds sit just above
/// those maxima; the 8 stacks checked here include the worst of the 64 (seed 1).
#[test]
fn recomputed_bf16_stack_is_as_close_to_f32_as_plain_autodiff() {
    for (name, device) in devices() {
        for seed in 0..8 {
            let (worst, overall) = bf16_ratios(name, &device, seed, seed == 0);
            println!(
                "{name} seed {seed}: recomputed vs plain distance from f32, largest ratio {worst:.2}, \
                 overall {overall:.3}"
            );
            bound(worst <= 2.6 && overall <= 1.35, || {
                format!(
                    "{name} seed {seed}: recomputed bf16 gradients {worst:.2} ({overall:.3} \
                     overall) times plain's distance from f32"
                )
            });
        }
    }
}

/// The distribution the bounds above come from:
///
///   cargo test -p dash-lora --test recompute -- --ignored --nocapture many_seeds
#[test]
#[ignore]
fn bf16_ratios_over_many_seeds() {
    for (name, device) in devices() {
        let (mut worst, mut overall) = (Vec::new(), Vec::new());
        for seed in 0..64 {
            let (w, o) = bf16_ratios(name, &device, seed, false);
            worst.push(w);
            overall.push(o);
        }
        let stats = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[0], v[v.len() / 2], v[v.len() - 1])
        };
        let ((w0, w1, w2), (o0, o1, o2)) = (stats(&mut worst), stats(&mut overall));
        println!(
            "{name}: largest element-error ratio min {w0:.2} median {w1:.2} max {w2:.2}; \
             overall rms ratio min {o0:.3} median {o1:.3} max {o2:.3}"
        );
    }
}

/// On a device with Burn's own (per-operation) checkpointing the leaves the backward pass
/// makes carry the parameters' strategy, so no operation mixes two.
#[test]
fn recompute_runs_under_balanced_checkpointing() {
    let device = Device::flex().autodiff().gradient_checkpointing();
    let (blocks, x) = stack(&device, DType::F32, 7);
    let (p, r) = (plain(&blocks, &x), recomputed(&blocks, &x));
    check("flex balanced f32 input gradient", &r.dx, &p.dx, 0.0, 0.0);
    for (i, (r, p)) in r.grads.iter().zip(&p.grads).enumerate() {
        check(&format!("flex balanced f32 {}", label(i)), r, p, 0.0, 0.0);
    }
}

/// A block used at two depths of the stack shares its parameters; their gradient is the sum
/// of both uses', as plain autodiff gives.
#[test]
fn shared_parameters_accumulate_across_layers() {
    let device = Device::flex().autodiff();
    let (mut blocks, x) = stack(&device, DType::F32, 7);
    blocks[2] = blocks[0].clone();
    let (p, r) = (plain(&blocks, &x), recomputed(&blocks, &x));
    for (i, (r, p)) in r.grads.iter().zip(&p.grads).enumerate() {
        check(&format!("flex shared f32 {}", label(i)), r, p, 1e-7, 1e-6);
    }
}
