//! Activation checkpointing (`dash_ops::checkpoint`) against plain autodiff through the same
//! stack. The stacks are transformer-style blocks built from Burn's own modules (`Linear`,
//! `LayerNorm`, `RmsNorm`) and tensor operations, every parameter trained: MLP blocks and causal
//! self-attention blocks, mixed in one stack through an enum, over `[B, T, C]` sequences, with
//! the causal mask a constant the layer function holds; checkpointed one layer or two at a time,
//! and inside a model whose embedding, below the stack, is tied to its output projection. The
//! loss, the stack input's gradient and every parameter's gradient are compared, in f32 and in
//! bf16, on Flex and on a CubeCL device (`DASH_OPS_DEVICE`; the bf16 cases where it has bf16):
//! bit for bit, but for what `same_computation` and the layer norm test state. The layer function
//! counts its calls: each unit runs once in each pass. On a device where `dash_ops::attention`'s
//! kernels run, a stack of attention blocks built on them is held to the same at a length where
//! many key blocks add to each query's gradient.
//!
//! The stacks are drawn on the host from a seeded generator, not from Burn's process-wide one,
//! so they do not depend on which other tests run at the same time.
//!
//!   cargo test --release -p dash-ops --test checkpoint -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test checkpoint -- --nocapture
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test checkpoint
//!
//! and with `cuda,fusion`, `cuda,autotune` and `cuda,fusion,autotune`. `DASH_OPS_MEASURE=1`
//! prints a bound a run exceeds instead of failing (`BOUND EXCEEDED: ...`), so that one run
//! measures every error of a new build, from which the bounds are set.

mod common;

use std::cell::Cell;
use std::sync::{Mutex, MutexGuard};

use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, RmsNorm};
use burn::tensor::{Bool, DType, Device, Gradients, Int, Tensor, TensorData, activation};
use dash_ops::attention::{AttentionMask, AttentionOptions, attention};
use dash_ops::checkpoint;

/// The tests take turns. With Burn's `autotune`, a thread that asks for a shape another thread
/// is tuning runs an untuned candidate meanwhile, and the passes compared would then need not
/// run the same kernels.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const HEADS: usize = 2;

/// `x + fc2(gelu(fc1(norm(x))))`: a transformer's MLP block.
#[derive(Module, Debug)]
struct Mlp {
    norm: Norm,
    fc1: Linear,
    fc2: Linear,
}

/// The MLP blocks' normalisation: RMS norm, or layer norm (which Flex computes with a fused
/// kernel outside autodiff and through Burn's decomposition when recorded).
#[derive(Module, Debug)]
enum Norm {
    Rms(RmsNorm),
    Layer(LayerNorm),
}

impl Norm {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        match self {
            Norm::Rms(n) => n.forward(x),
            Norm::Layer(n) => n.forward(x),
        }
    }
}

/// `x + out(attention(qkv(norm(x))))`: causal self-attention over `T`, `HEADS` heads, from
/// Burn's tensor operations, the softmax in f32.
#[derive(Module, Debug)]
struct Attention {
    norm: RmsNorm,
    qkv: Linear,
    out: Linear,
}

/// `x + out(attention(qkv(norm(x))))` through `dash_ops::attention`: causal self-attention over
/// `T`, `C / 128` heads of 128, its backward pass the module's written-out one.
#[derive(Module, Debug)]
struct KernelAttention {
    norm: RmsNorm,
    qkv: Linear,
    out: Linear,
}

/// One layer of a stack of mixed layers.
#[derive(Module, Debug)]
enum Layer {
    Mlp(Mlp),
    Attention(Attention),
    /// Attention through `dash_ops::attention`, its backward pass deterministic (the default).
    Kernel(KernelAttention),
    /// The same with the backward's query gradient summed with atomics.
    KernelAtomic(KernelAttention),
}

impl Mlp {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let h = self.fc1.forward(self.norm.forward(x.clone()));
        let dtype = h.dtype();
        let h = activation::gelu(h.cast(DType::F32)).cast(dtype);
        x.add(self.fc2.forward(h))
    }
}

impl Attention {
    /// `mask` `[T, T]` is `true` where a key is left out (above the diagonal).
    fn forward(&self, x: Tensor<3>, mask: &Tensor<2, Bool>) -> Tensor<3> {
        let [b, t, c] = x.dims();
        let d = c / HEADS;
        let qkv = self.qkv.forward(self.norm.forward(x.clone()));
        let dtype = qkv.dtype();
        let heads = |i: usize| {
            qkv.clone()
                .narrow(2, i * c, c)
                .reshape([b, t, HEADS, d])
                .swap_dims(1, 2)
        };
        let (q, k, v) = (heads(0), heads(1), heads(2));
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .cast(DType::F32)
            .mul_scalar(1.0 / (d as f64).sqrt())
            .mask_fill(
                mask.clone().unsqueeze::<4>().expand([b, HEADS, t, t]),
                f32::NEG_INFINITY,
            );
        let p = activation::softmax(scores, 3).cast(dtype);
        let o = p.matmul(v).swap_dims(1, 2).reshape([b, t, c]);
        x.add(self.out.forward(o))
    }
}

impl KernelAttention {
    fn forward(&self, x: Tensor<3>, deterministic: bool) -> Tensor<3> {
        let [b, t, c] = x.dims();
        let qkv = self.qkv.forward(self.norm.forward(x.clone()));
        let heads = |i: usize| {
            qkv.clone()
                .narrow(2, i * c, c)
                .reshape([b, t, c / 128, 128])
        };
        let options = AttentionOptions::causal().with_deterministic(deterministic);
        let o = attention(heads(0), heads(1), heads(2), AttentionMask::None, options);
        x.add(self.out.forward(o.reshape([b, t, c])))
    }
}

impl Layer {
    /// `mask` is the causal mask of [`Attention`]; the kernels' blocks take causality as an
    /// option.
    fn forward(&self, x: Tensor<3>, mask: &Tensor<2, Bool>) -> Tensor<3> {
        match self {
            Layer::Mlp(m) => m.forward(x),
            Layer::Attention(a) => a.forward(x, mask),
            Layer::Kernel(a) => a.forward(x, true),
            Layer::KernelAtomic(a) => a.forward(x, false),
        }
    }
}

/// A seeded generator on the host (SplitMix64).
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

    /// A tensor of `shape` filled by `draw`, its values rounded to `precision.values` on the
    /// host and stored at `precision.storage` (exactly: f32 holds every bf16 value).
    fn tensor<const D: usize>(
        &mut self,
        shape: [usize; D],
        precision: Precision,
        device: &Device,
        mut draw: impl FnMut(&mut Self) -> f64,
    ) -> Tensor<D> {
        let n = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|_| draw(self) as f32).collect();
        let data = TensorData::new(values, shape)
            .convert_dtype(precision.values)
            .convert_dtype(precision.storage);
        Tensor::from_data(data, (device, precision.storage))
    }

    fn param<const D: usize>(
        &mut self,
        shape: [usize; D],
        precision: Precision,
        device: &Device,
        draw: impl FnMut(&mut Self) -> f64,
    ) -> Param<Tensor<D>> {
        Param::from_tensor(self.tensor(shape, precision, device, draw))
    }
}

/// The values a stack is drawn at, and the dtype it computes in.
#[derive(Clone, Copy, Debug)]
struct Precision {
    values: DType,
    storage: DType,
}

const F32: Precision = Precision {
    values: DType::F32,
    storage: DType::F32,
};
const BF16: Precision = Precision {
    values: DType::BF16,
    storage: DType::BF16,
};
/// bf16 values computed in f32: the reference for a bf16 stack.
const BF16_IN_F32: Precision = Precision {
    values: DType::BF16,
    storage: DType::F32,
};

fn linear(rng: &mut Rng, d_in: usize, d_out: usize, p: Precision, device: &Device) -> Linear {
    let s = 1.0 / (d_in as f64).sqrt();
    Linear {
        weight: rng.param([d_in, d_out], p, device, |g| g.uniform(-s, s)),
        bias: Some(rng.param([d_out], p, device, |g| g.uniform(-s, s))),
    }
}

fn layer_norm(rng: &mut Rng, d: usize, p: Precision, device: &Device) -> LayerNorm {
    let mut norm = LayerNormConfig::new(d).init(device);
    norm.gamma = rng.param([d], p, device, |g| g.uniform(0.5, 1.5));
    norm.beta = Some(rng.param([d], p, device, |g| g.uniform(-0.5, 0.5)));
    norm
}

fn rms_norm(rng: &mut Rng, d: usize, p: Precision, device: &Device) -> RmsNorm {
    RmsNorm {
        gamma: rng.param([d], p, device, |g| g.uniform(0.5, 1.5)),
        epsilon: 1e-5,
    }
}

const B: usize = 2;
const T: usize = 12;
const C: usize = 32;
const INNER: usize = 64;

/// Which normalisation the MLP blocks of a stack use.
#[derive(Clone, Copy, Debug)]
enum NormKind {
    Rms,
    Layer,
}

/// `n` layers drawn from `seed`, MLP and attention blocks in turn, and an input `[B, T, C]`.
fn stack(
    n: usize,
    norm: NormKind,
    p: Precision,
    device: &Device,
    seed: u64,
) -> (Vec<Layer>, Tensor<3>) {
    let mut rng = Rng(seed);
    let layers = (0..n)
        .map(|i| match i % 2 {
            0 => Layer::Mlp(Mlp {
                norm: match norm {
                    NormKind::Rms => Norm::Rms(rms_norm(&mut rng, C, p, device)),
                    NormKind::Layer => Norm::Layer(layer_norm(&mut rng, C, p, device)),
                },
                fc1: linear(&mut rng, C, INNER, p, device),
                fc2: linear(&mut rng, INNER, C, p, device),
            }),
            _ => Layer::Attention(Attention {
                norm: rms_norm(&mut rng, C, p, device),
                qkv: linear(&mut rng, C, 3 * C, p, device),
                out: linear(&mut rng, C, C, p, device),
            }),
        })
        .collect();
    let x = rng.tensor([B, T, C], p, device, Rng::normal);
    (layers, x)
}

/// `true` above the diagonal: key `j` is left out of query `i`'s attention when `j > i`.
fn causal_mask(device: &Device) -> Tensor<2, Bool> {
    let mask: Vec<bool> = (0..T * T).map(|k| k % T > k / T).collect();
    Tensor::from_data(TensorData::new(mask, [T, T]), device)
}

fn loss(out: Tensor<3>) -> Tensor<1> {
    out.cast(DType::F32).powi_scalar(2).mean()
}

/// The gradient of every trainable parameter of `module`, labelled by its path, in order.
fn param_grads<M: Module>(module: &M, grads: &Gradients) -> Vec<(String, TensorData)> {
    struct Collect<'a> {
        grads: &'a Gradients,
        path: Vec<String>,
        out: Vec<(String, TensorData)>,
    }
    impl ModuleVisitor for Collect<'_> {
        fn enter_module(&mut self, name: &str, _: &str) {
            self.path.push(name.to_string());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if !param.is_require_grad() {
                return;
            }
            let label = format!("{} #{}", self.path.join("."), self.out.len());
            let grad = param
                .val()
                .grad(self.grads)
                .unwrap_or_else(|| panic!("{label} has no gradient"));
            self.out.push((label, grad.into_data()));
        }
    }
    let mut collect = Collect {
        grads,
        path: Vec::new(),
        out: Vec::new(),
    };
    module.visit(&mut collect);
    collect.out
}

/// The loss, the stack input's gradient, and every parameter's.
struct Run {
    loss: TensorData,
    dx: TensorData,
    params: Vec<(String, TensorData)>,
}

fn plain(layers: &[Layer], x: &Tensor<3>, mask: &Tensor<2, Bool>) -> Run {
    let x = x.clone().detach().require_grad();
    let loss = loss(layers.iter().fold(x.clone(), |h, l| l.forward(h, mask)));
    let grads = loss.backward();
    Run {
        loss: loss.into_data(),
        dx: x.grad(&grads).expect("input gradient").into_data(),
        params: param_grads(&layers.to_vec(), &grads),
    }
}

/// The stack checkpointed in units of `per_unit` layers; the layer function counts its calls.
fn checkpointed(layers: &[Layer], x: &Tensor<3>, mask: &Tensor<2, Bool>, per_unit: usize) -> Run {
    let units: Vec<Vec<Layer>> = layers.chunks(per_unit).map(<[Layer]>::to_vec).collect();
    let calls = Cell::new(0);
    let unit = |u: &Vec<Layer>, h: Tensor<3>| {
        calls.set(calls.get() + 1);
        u.iter().fold(h, |h, l| l.forward(h, mask))
    };
    let (out, saved) = checkpoint::forward(&units, x.clone(), unit);
    assert_eq!((saved.len(), calls.get()), (units.len(), units.len()));
    assert!(out.is_require_grad(), "the output is a leaf for the loss");
    let loss = loss(out.clone());
    let mut grads = loss.backward();
    let grad = out.grad_remove(&mut grads).expect("output gradient");
    let dx = checkpoint::backward(&units, saved, grad, unit, &mut grads);
    assert_eq!(calls.get(), 2 * units.len(), "each unit runs twice");
    Run {
        loss: loss.into_data(),
        dx: dx.into_data(),
        params: param_grads(&layers.to_vec(), &grads),
    }
}

/// Errors of `actual` against `expected`, both read as f64: the largest absolute difference,
/// and that difference over the largest absolute expected value.
fn err(actual: &TensorData, expected: &TensorData) -> (f64, f64) {
    assert_eq!(actual.shape(), expected.shape(), "shapes differ");
    let (mut abs, mut scale) = (0f64, 0f64);
    for (a, e) in actual.iter::<f64>().zip(expected.iter::<f64>()) {
        assert!(a.is_finite(), "non-finite value {a}");
        abs = abs.max((a - e).abs());
        scale = scale.max(e.abs());
    }
    (abs, if scale > 0.0 { abs / scale } else { abs })
}

/// Asserts a bound; with `DASH_OPS_MEASURE=1`, prints an exceeded one instead.
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

/// Measures, prints and bounds the error of `actual` against `expected`, relative to the
/// largest expected value; returns it.
fn check(what: &str, actual: &TensorData, expected: &TensorData, max_rel: f64) -> f64 {
    let (abs, rel) = err(actual, expected);
    println!("{what:<48} max abs {abs:.3e}  rel {rel:.3e}");
    bound(rel <= max_rel, || {
        format!("{what}: max abs {abs:.3e}, rel {rel:.3e} (bound {max_rel:.1e})")
    });
    rel
}

/// Every output of `r` against `p`.
fn check_run(name: &str, r: &Run, p: &Run, max_rel: f64) {
    check(&format!("{name} loss"), &r.loss, &p.loss, max_rel);
    check(&format!("{name} input gradient"), &r.dx, &p.dx, max_rel);
    assert_eq!(r.params.len(), p.params.len());
    for ((label, r), (_, p)) in r.params.iter().zip(&p.params) {
        check(&format!("{name} {label}"), r, p, max_rel);
    }
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

/// Bounds on the checkpointed stack's distance from plain autodiff through the same stack, each
/// relative to the tensor's largest value: `(f32, bf16)`, set (`DASH_OPS_MEASURE=1`) just above
/// the largest measured on each build. On Flex, and on CUDA with Burn's default kernels or with
/// `autotune`, the two compute the same thing: nothing. With `fusion` (with or without
/// `autotune`), the passes through one layer fuse its operations otherwise than the pass through
/// the whole stack: f32 gradients up to 2.5e-7 apart over five runs, bf16 ones bit-identical on
/// these stacks. With `fusion` and `autotune` both, each pass's fused kernels are tuned apart
/// too, differently from run to run: f32 gradients up to 4.12e-7 apart over 37 runs, bf16 ones
/// still bit-identical; the f32 bound is 1e-6, a few roundings, for the run that goes further. The
/// tests take turns: with `autotune` and the tests in parallel, a thread that asks for a shape
/// another is tuning runs an untuned candidate meanwhile, and over eight runs the passes moved up
/// to 3.6e-7 apart in f32 and 1.5e-4 in bf16.
fn same_computation(device: &str) -> (f64, f64) {
    match device {
        "flex" => (0.0, 0.0),
        _ if cfg!(feature = "fusion") && cfg!(feature = "autotune") => (1e-6, 0.0),
        _ if cfg!(feature = "fusion") => (2.6e-7, 0.0),
        _ => (0.0, 0.0),
    }
}

#[test]
fn checkpointed_stack_gives_plain_autodiff_s_gradients() {
    let _serial = serial();
    for (name, device) in devices() {
        let mask = causal_mask(&device);
        for (precision, label, max_rel) in [
            (F32, "f32", same_computation(name).0),
            (BF16, "bf16", same_computation(name).1),
        ] {
            if !common::has(&device, precision.values) {
                continue;
            }
            let (layers, x) = stack(4, NormKind::Rms, precision, &device, 7);
            let p = plain(&layers, &x, &mask);
            for per_unit in [1, 2] {
                let r = checkpointed(&layers, &x, &mask, per_unit);
                check_run(
                    &format!("{name} {label}, {per_unit} per unit,"),
                    &r,
                    &p,
                    max_rel,
                );
            }
        }
    }
}

/// Two attention blocks on `dash_ops::attention`'s kernels, heads of 128 in bf16 over a sequence
/// of 1000, where 32 key blocks add to the last queries' gradients: with the default
/// deterministic backward, checkpointed one block at a time, the loss and every gradient are
/// plain autodiff's bit for bit, as `same_computation` allows for bf16. With the query gradient
/// summed with atomics they are not: its last bits move from one backward pass to the next, and
/// the gradients below it with them (printed; within a bf16 unit of the largest value). Where the
/// kernels do not run, the reference runs, which is deterministic either way.
#[cfg(feature = "cubecl")]
#[test]
fn a_stack_on_the_attention_kernels_gives_plain_autodiff_s_gradients() {
    let _serial = serial();
    let device = common::cube_device().autodiff();
    let (b, t, c) = (1, 1000, 256);
    if !common::has(&device, DType::BF16)
        || !dash_ops::attention::kernels_take(&device, DType::BF16, [b, t, t, 2, 2, 128])[1]
    {
        println!("skipped: the attention kernels do not run here");
        return;
    }
    let mask = causal_mask(&device);
    let max_rel = same_computation("cubecl").1;
    for atomic in [false, true] {
        let mut rng = Rng(11);
        let layers: Vec<Layer> = (0..2)
            .map(|_| {
                let block = KernelAttention {
                    norm: rms_norm(&mut rng, c, BF16, &device),
                    qkv: linear(&mut rng, c, 3 * c, BF16, &device),
                    out: linear(&mut rng, c, c, BF16, &device),
                };
                if atomic {
                    Layer::KernelAtomic(block)
                } else {
                    Layer::Kernel(block)
                }
            })
            .collect();
        let x = rng.tensor([b, t, c], BF16, &device, Rng::normal);
        let p = plain(&layers, &x, &mask);
        let r = checkpointed(&layers, &x, &mask, 1);
        if atomic {
            let mut worst = err(&r.dx, &p.dx).1;
            for ((_, r), (_, p)) in r.params.iter().zip(&p.params) {
                worst = worst.max(err(r, p).1);
            }
            println!(
                "attention kernels with atomics: checkpointed against plain, largest {worst:.3e}"
            );
            assert!(
                worst <= 1.0 / 128.0,
                "{worst:.3e}: more than a bf16 unit of the largest value"
            );
        } else {
            check_run("attention kernels, deterministic,", &r, &p, max_rel);
        }
    }
}

/// How much farther from the same stack computed in f32 the checkpointed bf16 gradients are
/// than plain autodiff's, for the stack drawn from `seed`: the largest ratio over the input
/// gradient and every parameter gradient of their errors (each over its tensor's largest
/// value), and the ratio of their root-mean-square errors summed over those tensors.
fn bf16_ratios(name: &str, norm: NormKind, device: &Device, seed: u64) -> (f64, f64) {
    let mask = causal_mask(device);
    let (layers, x) = stack(4, norm, BF16, device, seed);
    let (p, r) = (
        plain(&layers, &x, &mask),
        checkpointed(&layers, &x, &mask, 1),
    );
    let (layers32, x32) = stack(4, norm, BF16_IN_F32, device, seed);
    let exact = plain(&layers32, &x32, &mask);
    let (mut worst, mut rms_r, mut rms_p) = (0f64, 0f64, 0f64);
    let pairs = std::iter::once(("input gradient", &r.dx, &p.dx, &exact.dx)).chain(
        r.params
            .iter()
            .zip(&p.params)
            .zip(&exact.params)
            .map(|(((label, r), (_, p)), (_, e))| (label.as_str(), r, p, e)),
    );
    for (what, r, p, e) in pairs {
        let (er, ep) = (err(r, e).1, err(p, e).1);
        if seed == 0 {
            println!("  {name} bf16 {what:<28} vs f32: checkpointed {er:.2e}, plain {ep:.2e}");
        }
        worst = worst.max(er / ep);
        rms_r += rms_rel(r, e);
        rms_p += rms_rel(p, e);
    }
    (worst, rms_r / rms_p)
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

/// Layer norm is the exception on Flex: Flex computes it with a fused kernel of its own outside
/// autodiff, and Burn's decomposition (means, a variance, a division) runs when it is recorded,
/// so the forward pass and the recorded pass round it differently. Plain autodiff's gradients
/// are the decomposition's throughout; the checkpointed ones are the decomposition's evaluated
/// at the fused kernel's values. In f32 they differ by a few roundings, up to 2.8e-7 of a
/// tensor's largest value; in bf16 such a rounding now and then moves a bf16 value by a unit,
/// and the checkpointed gradients are as far from the same stack computed in f32 as plain
/// autodiff's: over these 8 stacks, at most 2.24 times as far on a single tensor, and over all
/// of a stack's tensors between 0.955 and 1.115 times (root-mean-square errors). On CUDA layer
/// norm is the decomposition in both passes, and the gradients agree as for the other stacks.
#[test]
fn a_layer_norm_stack_differs_on_flex_by_roundings() {
    let _serial = serial();
    for (name, device) in devices() {
        let mask = causal_mask(&device);
        let (layers, x) = stack(4, NormKind::Layer, F32, &device, 7);
        let p = plain(&layers, &x, &mask);
        let r = checkpointed(&layers, &x, &mask, 1);
        let max_rel = match name {
            "flex" => 3e-7,
            _ => same_computation(name).0,
        };
        check_run(&format!("{name} layer norm f32"), &r, &p, max_rel);
        if !common::has(&device, DType::BF16) {
            continue;
        }
        for seed in 0..8 {
            let (worst, overall) = bf16_ratios(name, NormKind::Layer, &device, seed);
            println!(
                "{name} layer norm seed {seed}: checkpointed vs plain distance from f32, \
                 largest ratio {worst:.2}, overall {overall:.3}"
            );
            bound(worst <= 2.3 && overall <= 1.12, || {
                format!(
                    "{name} seed {seed}: checkpointed bf16 gradients {worst:.2} ({overall:.3} \
                     overall) times plain's distance from f32"
                )
            });
        }
    }
}

/// A model around the stack: an embedding below it, tied to the output projection above it,
/// whose gradient is the sum of both uses'. The stack input's gradient reaches the embedding
/// through `checkpoint::backward_through`.
#[derive(Module, Debug)]
struct Model {
    embedding: Param<Tensor<2>>,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

const VOCAB: usize = 48;

impl Model {
    fn new(device: &Device, seed: u64) -> (Self, Tensor<2, Int>, Tensor<2, Int>) {
        let (layers, _) = stack(4, NormKind::Rms, F32, device, seed);
        let mut rng = Rng(seed ^ 0x5eed);
        let embedding = rng.param([VOCAB, C], F32, device, Rng::normal);
        let norm = rms_norm(&mut rng, C, F32, device);
        // Distinct token ids: no row of the embedding's gradient sums several tokens'
        // contributions, so it does not depend on the order a kernel adds them in.
        let mut ids: Vec<i64> = (0..VOCAB as i64).collect();
        for i in (1..ids.len()).rev() {
            ids.swap(i, (rng.unit() * (i + 1) as f64) as usize);
        }
        let tokens = Tensor::from_data(TensorData::new(ids[..B * T].to_vec(), [B, T]), device);
        let targets = (0..B * T).map(|_| (rng.unit() * VOCAB as f64) as i64);
        let targets = Tensor::from_data(TensorData::new(targets.collect(), [B, T]), device);
        (
            Self {
                embedding,
                layers,
                norm,
            },
            tokens,
            targets,
        )
    }

    fn embed(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        self.embedding
            .val()
            .select(0, tokens.reshape([B * T]))
            .reshape([B, T, C])
    }

    /// Cross-entropy of the next tokens under logits from the tied output projection.
    fn loss(&self, h: Tensor<3>, targets: Tensor<2, Int>) -> Tensor<1> {
        let logits = self
            .norm
            .forward(h)
            .matmul(self.embedding.val().transpose().unsqueeze());
        let logp = activation::log_softmax(logits, 2);
        logp.gather(2, targets.unsqueeze_dim(2)).mean().neg()
    }
}

#[test]
fn a_model_around_the_stack_gives_plain_autodiff_s_gradients() {
    let _serial = serial();
    for (name, device) in devices() {
        let (model, tokens, targets) = Model::new(&device, 11);
        let mask = causal_mask(&device);
        let f = |l: &Layer, h: Tensor<3>| l.forward(h, &mask);

        let h = model.embed(tokens.clone());
        let loss = model.loss(model.layers.iter().fold(h, |h, l| f(l, h)), targets.clone());
        let plain = param_grads(&model, &loss.backward());

        let h = model.embed(tokens);
        let (out, saved) = checkpoint::forward(&model.layers, h.clone(), f);
        let loss = model.loss(out.clone(), targets);
        let mut grads = loss.backward();
        let grad = out.grad_remove(&mut grads).unwrap();
        let grad = checkpoint::backward(&model.layers, saved, grad, f, &mut grads);
        checkpoint::backward_through(h, grad, &model.embedding, &mut grads);
        let checkpointed = param_grads(&model, &grads);

        assert_eq!(checkpointed.len(), plain.len());
        for ((label, r), (_, p)) in checkpointed.iter().zip(&plain) {
            check(
                &format!("{name} model {label}"),
                r,
                p,
                same_computation(name).0,
            );
        }
    }
}

/// On a device with Burn's own (per-operation) checkpointing the leaves the backward pass makes
/// carry the parameters' strategy, so no operation mixes two; when every parameter is frozen and
/// only the input's gradient is asked for, the input's strategy.
#[cfg(feature = "flex")]
#[test]
fn checkpointing_runs_under_balanced_checkpointing() {
    let _serial = serial();
    let device = Device::flex().autodiff().gradient_checkpointing();
    let mask = causal_mask(&device);
    let (layers, x) = stack(4, NormKind::Rms, F32, &device, 7);
    let (p, r) = (
        plain(&layers, &x, &mask),
        checkpointed(&layers, &x, &mask, 1),
    );
    check_run("flex balanced f32", &r, &p, 0.0);

    let frozen: Vec<Layer> = layers.into_iter().map(Module::no_grad).collect();
    let (p, r) = (
        plain(&frozen, &x, &mask),
        checkpointed(&frozen, &x, &mask, 1),
    );
    assert!(r.params.is_empty());
    check_run("flex balanced frozen f32", &r, &p, 0.0);
}

/// A layer used at two depths of the stack shares its parameters; their gradient is the sum of
/// both uses', as plain autodiff gives.
#[cfg(feature = "flex")]
#[test]
fn shared_parameters_accumulate_across_layers() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    let mask = causal_mask(&device);
    let (mut layers, x) = stack(4, NormKind::Rms, F32, &device, 7);
    layers[2] = layers[0].clone();
    let (p, r) = (
        plain(&layers, &x, &mask),
        checkpointed(&layers, &x, &mask, 1),
    );
    assert_eq!(r.params.len(), p.params.len());
    check_run("flex shared f32", &r, &p, 1e-6);
}

/// A tensor that requires a gradient, used by the layer function without being a parameter of
/// the layer, would have its gradient lost: refused.
#[cfg(feature = "flex")]
#[test]
#[should_panic(expected = "requires a gradient and is not one of its parameters")]
fn a_trainable_tensor_outside_the_layers_is_refused() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    let (layers, x) = stack(2, NormKind::Rms, F32, &device, 7);
    let mask = causal_mask(&device);
    let scale = Tensor::<1>::ones([C], &device).require_grad();
    checkpoint::forward(&layers, x, |l, h| {
        l.forward(h, &mask).mul(scale.clone().unsqueeze())
    });
}
