//! A stack of LoRA'd blocks trained with each block recomputed in the backward pass
//! (`dash_ops::checkpoint`) gives the gradients plain autodiff gives: every adapter's, and the
//! stack input's. Each projection is LoRA's layer as dash-mag and PEFT compute it, its low-rank
//! term [`dash_ops::lowrank`]'s kernels where they run, with a frozen base: the parameters a
//! stack trains are some of those its layers hold.
//!
//! In f32 they are bit-identical, on Flex and on CUDA, and under Burn's own checkpointing too
//! (on CUDA with Burn's autotune and fusion together, within a few f32 roundings: see
//! `same_computation`).
//! In bf16 they need not be: Burn adds up a tensor's gradient contributions as they arrive, a
//! block's input has five consumers, and the per-block graphs can reach them in a different order
//! than the whole-stack graph, so that their bf16 sums round differently and the difference
//! travels down the stack. On 2026-10-26 they did; on Flex and on CUDA on 2026-11-20 they do not
//! (below).
//! What is asserted is that both are as far from the same stack computed in f32, over several
//! random stacks.
//! The stacks are drawn on the host from a seeded generator, not from Burn's process-wide one,
//! so they do not depend on which other tests run at the same time.
//!
//!   cargo test -p dash-ops --test checkpoint_lowrank -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test -p dash-ops --features cuda --test checkpoint_lowrank -- --nocapture

mod common;

use std::sync::{Mutex, MutexGuard};

use burn::module::{Module, Param, ParamId};
use burn::optim::GradientsParams;
use burn::tensor::{DType, Device, Tensor, TensorData, activation};
use dash_ops::checkpoint;
use dash_ops::lowrank::{LowRankOptions, low_rank};

/// The tests take turns. With Burn's `autotune`, a thread that asks for a shape another thread
/// is tuning runs an untuned candidate meanwhile, and the passes compared would then need not
/// run the same kernels.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A LoRA'd projection as LoRA's layer computes one (dash-mag's, and PEFT's `lora.Linear`):
/// `x Wᵀ + scale · (x Aᵀ) Bᵀ`, the base product at the weight's dtype, the low-rank term by
/// [`dash_ops::lowrank`] in f32, their sum in f32 rounded once to `x`'s dtype. `W` is frozen.
#[derive(Module, Debug)]
struct LoraProj {
    weight: Param<Tensor<2>>,
    a: Param<Tensor<2>>,
    b: Param<Tensor<2>>,
    scale: f64,
}

impl LoraProj {
    fn new(weight: Tensor<2>, a: Tensor<2>, b: Tensor<2>, scale: f64) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), weight.detach()),
            a: Param::from_tensor(a.cast(DType::F32).detach()),
            b: Param::from_tensor(b.cast(DType::F32).detach()),
            scale,
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let dtype = x.dtype();
        let weight = self.weight.val();
        let base = x.clone().cast(weight.dtype()).matmul(weight.transpose());
        let base = if base.dtype() == dtype {
            base
        } else {
            base.cast(DType::F32)
        };
        let (u, v) = (self.a.val().transpose(), self.b.val().transpose());
        let options = LowRankOptions::default().with_scale(self.scale);
        low_rank(x, u, v, Some(base), options).cast(dtype)
    }
}

/// `x + down(silu(gate(x)) * up(x))`, every projection LoRA'd: a language model's MLP block.
#[derive(Module, Debug)]
struct Block {
    gate: LoraProj,
    up: LoraProj,
    down: LoraProj,
}

impl Block {
    fn new(d: usize, inner: usize, r: usize, dtype: DType, device: &Device, rng: &mut Rng) -> Self {
        let mut linear = |out: usize, inp: usize| {
            let (si, sr) = ((inp as f64).sqrt(), (r as f64).sqrt());
            let w = rng.tensor([out, inp], dtype, device, |g| g.uniform(-1.0, 1.0) / si);
            let a = rng.tensor([r, inp], DType::F32, device, |g| g.uniform(-1.0, 1.0) / si);
            let b = rng.tensor([out, r], DType::F32, device, |g| g.uniform(-1.0, 1.0) / sr);
            LoraProj::new(w, a, b, 2.0)
        };
        Self {
            gate: linear(inner, d),
            up: linear(inner, d),
            down: linear(d, inner),
        }
    }

    /// The same block with its base weights upcast to f32 (exactly) and the same adapters.
    fn to_f32(&self) -> Self {
        let f32 = |l: &LoraProj| {
            LoraProj::new(
                l.weight.val().cast(DType::F32),
                l.a.val(),
                l.b.val(),
                l.scale,
            )
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

    fn loras(&self) -> [&LoraProj; 3] {
        [&self.gate, &self.up, &self.down]
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
    let block = |b: &Block, h: Tensor<2>| b.forward(h);
    let (out, saved) = checkpoint::forward(blocks, x.clone(), block);
    assert_eq!(saved.len(), blocks.len());
    let loss = loss(out.clone());
    let mut grads = loss.backward();
    let grad = out.grad_remove(&mut grads).unwrap();
    let dx = checkpoint::backward(blocks, saved, grad, block, &mut grads);
    let params = GradientsParams::from_grads(grads, &blocks.to_vec());
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
/// relative to the largest value. Nothing on Flex, nor on CUDA with Burn's default kernels,
/// with `burn/fusion` or with `autotune`: they agree bit for bit. With `autotune` and
/// `burn/fusion` together the two passes need not run the same kernels: autotune may settle on
/// a different fused kernel for the whole-stack backward pass than for the per-block ones
/// (their operation streams differ); over three runs, adapter gradients up to 5.4e-7. The
/// tests take turns: while one thread tunes a matrix product of a given shape, another asking
/// for the same shape runs an untuned candidate, and over fourteen runs of `autotune` builds
/// with the tests in parallel the f32 gradients moved up to 1.2e-6 apart, the f32 loss twice by
/// one rounding.
fn same_computation(device_name: &str) -> (f64, f64) {
    match device_name {
        "flex" => (0.0, 0.0),
        _ if cfg!(feature = "autotune") => (f64::INFINITY, 1e-6),
        _ => (0.0, 0.0),
    }
}

#[test]
fn recomputed_f32_stack_gives_plain_autodiff_s_gradients() {
    let _serial = serial();
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

/// Over 64 stacks (`bf16_ratios_over_many_seeds`) on 2026-10-26, Flex and CUDA giving the same
/// minimum, median and maximum, the summed root-mean-square ratio had a median of 1.00 and ranged
/// from 0.83 to 1.29: recomputing cost no accuracy, it only rounded differently. The largest
/// element-error ratio of a single tensor was noisier, median 1.41, up to 2.46. The bounds sit just
/// above those maxima; the 8 stacks checked here include the worst of the 64 then (seed 1). On
/// Flex and on CUDA on 2026-11-20 every ratio of the 64 is 1.00, on Flex with dash-lora's layer as
/// with this one.
#[test]
fn recomputed_bf16_stack_is_as_close_to_f32_as_plain_autodiff() {
    let _serial = serial();
    for (name, device) in devices_with(DType::BF16) {
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
///   cargo test -p dash-ops --test checkpoint_lowrank -- --ignored --nocapture many_seeds
#[test]
#[ignore]
fn bf16_ratios_over_many_seeds() {
    let _serial = serial();
    for (name, device) in devices_with(DType::BF16) {
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
    let _serial = serial();
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
    let _serial = serial();
    let device = Device::flex().autodiff();
    let (mut blocks, x) = stack(&device, DType::F32, 7);
    blocks[2] = blocks[0].clone();
    let (p, r) = (plain(&blocks, &x), recomputed(&blocks, &x));
    for (i, (r, p)) in r.grads.iter().zip(&p.grads).enumerate() {
        check(&format!("flex shared f32 {}", label(i)), r, p, 1e-7, 1e-6);
    }
}

// What the tests above share: a seeded generator, the devices, and error measurement.

/// A seeded generator on the host (SplitMix64) for test data. Burn's own random tensors draw from
/// one generator per process, which tests running in parallel threads share and interleave, so
/// their values would depend on what else runs; these depend on the seed alone.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform on [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f64 {
        let u = 1.0 - self.unit();
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// A tensor of `shape` filled by `draw`, stored at `dtype` (rounded on the host).
    fn tensor<const D: usize>(
        &mut self,
        shape: [usize; D],
        dtype: DType,
        device: &Device,
        mut draw: impl FnMut(&mut Self) -> f64,
    ) -> Tensor<D> {
        let n = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|_| draw(self) as f32).collect();
        let data = TensorData::new(values, shape).convert_dtype(dtype);
        Tensor::from_data(data, (device, dtype))
    }
}

/// The largest absolute difference of `actual` from `expected`, both read as f64, and that
/// difference over the largest absolute expected value.
struct Err {
    abs: f64,
    rel: f64,
}

fn err(actual: &TensorData, expected: &TensorData) -> Err {
    assert_eq!(actual.shape(), expected.shape());
    let (mut abs, mut scale) = (0f64, 0f64);
    for (x, y) in actual.iter::<f64>().zip(expected.iter::<f64>()) {
        assert!(x.is_finite(), "non-finite value {x}");
        abs = abs.max((x - y).abs());
        scale = scale.max(y.abs());
    }
    Err {
        abs,
        rel: if scale > 0.0 { abs / scale } else { abs },
    }
}

/// Asserts an error bound. With `DASH_OPS_MEASURE=1` an exceeded bound is printed instead
/// (`BOUND EXCEEDED: ...`), so that one run measures every error of a new configuration.
fn bound(ok: bool, message: impl FnOnce() -> String) {
    if ok {
        return;
    }
    let message = message();
    if std::env::var("DASH_OPS_MEASURE").is_ok_and(|v| v == "1") {
        println!("BOUND EXCEEDED: {message}");
    } else {
        panic!("{message}");
    }
}

/// Measures, prints and bounds the error of `actual` against `expected`.
fn check(what: &str, actual: &TensorData, expected: &TensorData, max_abs: f64, max_rel: f64) {
    let e = err(actual, expected);
    println!("{what:<44} max abs {:.3e}  rel {:.3e}", e.abs, e.rel);
    bound(e.abs <= max_abs && e.rel <= max_rel, || {
        format!(
            "{what}: max abs {:.3e} (bound {max_abs:.1e}), rel {:.3e} (bound {max_rel:.1e})",
            e.abs, e.rel
        )
    });
}

/// The devices the tests run on: Flex, and the CubeCL device `DASH_OPS_DEVICE` names with a
/// CubeCL runtime's feature.
// Each device is pushed under its own feature.
#[allow(unused_mut, clippy::vec_init_then_push)]
fn devices() -> Vec<(&'static str, Device)> {
    let mut devices = Vec::new();
    #[cfg(feature = "flex")]
    devices.push(("flex", Device::flex().autodiff()));
    #[cfg(feature = "cubecl")]
    devices.push(("cubecl", common::cube_device().autodiff()));
    devices
}

/// The [`devices`] that have `dtype`; a test in that dtype skips the others, saying so.
fn devices_with(dtype: DType) -> Vec<(&'static str, Device)> {
    devices()
        .into_iter()
        .filter(|(name, device)| {
            let has = device.supports_dtype(dtype);
            if !has {
                println!("skipped: {name} ({device:?}) has no {dtype:?}");
            }
            has
        })
        .collect()
}
