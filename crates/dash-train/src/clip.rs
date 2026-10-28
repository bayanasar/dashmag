//! Gradients clipped by their global norm, as `torch.nn.utils.clip_grad_norm_` clips them.
//!
//! [`dash_contract::clip`] defines the clipping, which every implementation of it shares: the
//! total over the gradients there are, the coefficient as torch evaluates it, the dtypes, a total
//! that is not finite, and the options ([`ClipGradNormOptions`]). This module implements it on
//! Burn.
//!
//! Burn clips each gradient by its own norm. PPO and GRPO clip them by one norm, that of every
//! gradient together: [`clip_grad_norm`] measures the norm of all the gradients a module's
//! parameters have, as one vector, and multiplies each by `max_norm / (total + 1e-6)` when that
//! is below 1. It works on the [`GradientsParams`] Burn's optimisers step with, before they do:
//!
//! ```ignore
//! let mut grads = GradientsParams::from_grads(loss.backward(), &model);
//! let total = clip_grad_norm(&model, &mut grads, &ClipGradNormOptions::default());
//! model = optim.step(lr, model, grads);
//! ```
//!
//! On any Burn device:
//! - The total is over the parameters that have a gradient in `grads`; a parameter the module
//!   holds twice (one [`ParamId`]) counts and is scaled once.
//! - Nothing is read back to the host: the total is returned as a tensor, `[1]` f32 on the first
//!   gradient's device, for the caller to log where it reads values anyway. The read torch's
//!   `error_if_nonfinite=True` makes is the caller's: `total.into_scalar::<f32>().is_finite()`.
//! - Under [`NormType::Inf`] a NaN element makes the total NaN: Burn's largest absolute value
//!   keeps NaN, on Flex and CubeCL alike.

use std::collections::HashSet;

use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::optim::GradientsParams;
use burn::tensor::{DType, Device, Tensor};

pub use dash_contract::clip::{ClipGradNormOptions, NonFinite, NormType};

/// Added to the total before the coefficient is taken from it, as torch adds it.
const EPS: f64 = 1e-6;

/// Clips the gradients `grads` holds for `module`'s parameters by their total norm, in place,
/// and returns that norm, as it was before clipping: `[1]` f32, on the device of the first
/// gradient the module's parameters visit. See [`dash_contract::clip`] and the
/// [module docs](self).
pub fn clip_grad_norm<M: Module>(
    module: &M,
    grads: &mut GradientsParams,
    options: &ClipGradNormOptions,
) -> Tensor<1> {
    let mut norms = Norms {
        grads,
        norm_type: options.norm_type,
        seen: HashSet::new(),
        device: None,
        partials: Vec::new(),
    };
    module.visit(&mut norms);
    let Some(total) = norms.total() else {
        let device = module.devices().into_iter().next().unwrap_or_default();
        return Tensor::zeros([1], (&device.without_autodiff(), DType::F32));
    };
    // torch evaluates `max_norm / (total + 1e-6)` as `(total + 1e-6).reciprocal() * max_norm`
    // (`Tensor.__rtruediv__`); the same two roundings give its coefficient bit for bit.
    let coef = total
        .clone()
        .add_scalar(EPS)
        .recip()
        .mul_scalar(options.max_norm);
    // At most 1, with NaN kept NaN as `torch.clamp` keeps it: a comparison with NaN is false.
    let coef = coef.clone().mask_fill(coef.greater_elem(1.0), 1.0);
    let coef = match options.nonfinite {
        NonFinite::Scale => coef,
        NonFinite::Zero => coef.mask_fill(total.clone().is_finite().bool_not(), 0.0),
    };
    module.visit(&mut Scale {
        grads,
        coef,
        cast: Vec::new(),
        zero_nan: options.nonfinite == NonFinite::Zero,
        seen: HashSet::new(),
    });
    total
}

/// `tensor` on `device`, moved only if it is elsewhere.
fn on(tensor: Tensor<1>, device: &Device) -> Tensor<1> {
    if tensor.device() == *device {
        tensor
    } else {
        tensor.to_device(device)
    }
}

/// The first pass: for each gradient, in f32 on the first gradient's device, its sum of squares
/// or its largest absolute value.
struct Norms<'a> {
    grads: &'a GradientsParams,
    norm_type: NormType,
    seen: HashSet<ParamId>,
    device: Option<Device>,
    partials: Vec<Tensor<1>>,
}

impl Norms<'_> {
    /// The total norm, if any parameter has a gradient.
    fn total(self) -> Option<Tensor<1>> {
        if self.partials.is_empty() {
            return None;
        }
        let partials = Tensor::cat(self.partials, 0);
        Some(match self.norm_type {
            NormType::L2 => partials.sum().sqrt(),
            NormType::Inf => partials.max(),
        })
    }
}

impl ModuleVisitor for Norms<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if !self.seen.insert(param.id) {
            return;
        }
        let Some(grad) = self.grads.get::<D>(param.id) else {
            return;
        };
        // Nothing to add, and nothing torch could take the largest of either.
        if grad.shape().num_elements() == 0 {
            return;
        }
        let device = self.device.get_or_insert_with(|| grad.device()).clone();
        let partial = match self.norm_type {
            NormType::L2 => grad.cast(DType::F32).square().sum(),
            // The largest absolute value is exact, and so is its cast from bf16 or f16.
            NormType::Inf => grad.max_abs().cast(DType::F32),
        };
        self.partials.push(on(partial, &device));
    }
}

/// The second pass: each gradient multiplied, in its own dtype, by the coefficient.
struct Scale<'a> {
    grads: &'a mut GradientsParams,
    coef: Tensor<1>,
    /// The coefficient on each device and in each dtype a gradient has asked for, cast once.
    cast: Vec<(Device, DType, Tensor<1>)>,
    /// Whether NaN, which a coefficient of 0 leaves where an element was infinite or NaN, is
    /// made 0.
    zero_nan: bool,
    seen: HashSet<ParamId>,
}

impl Scale<'_> {
    fn coef(&mut self, device: Device, dtype: DType) -> Tensor<1> {
        if let Some((_, _, coef)) = self
            .cast
            .iter()
            .find(|(d, t, _)| *d == device && *t == dtype)
        {
            return coef.clone();
        }
        let coef = on(self.coef.clone(), &device).cast(dtype);
        self.cast.push((device, dtype, coef.clone()));
        coef
    }
}

impl ModuleVisitor for Scale<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if !self.seen.insert(param.id) {
            return;
        }
        let Some(grad) = self.grads.remove::<D>(param.id) else {
            return;
        };
        let grad = if grad.shape().num_elements() == 0 {
            grad
        } else {
            let coef = self.coef(grad.device(), grad.dtype());
            let grad = grad.mul(coef.unsqueeze::<D>());
            if self.zero_nan {
                grad.clone().mask_fill(grad.is_nan(), 0.0)
            } else {
                grad
            }
        };
        self.grads.register(param.id, grad);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{TensorData, bf16};

    fn splitmix64(z: u64) -> u64 {
        let z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Parameters of every rank the cases use, in one module.
    #[derive(Module, Debug)]
    struct Net {
        r1: Vec<Param<Tensor<1>>>,
        r2: Vec<Param<Tensor<2>>>,
        r3: Vec<Param<Tensor<3>>>,
        r4: Vec<Param<Tensor<4>>>,
    }

    impl Net {
        fn new() -> Self {
            Net {
                r1: Vec::new(),
                r2: Vec::new(),
                r3: Vec::new(),
                r4: Vec::new(),
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Grad {
        /// Trained: in the loss, with a gradient.
        Yes,
        /// Frozen, as a LoRA base weight: in the loss, without a gradient.
        Frozen,
        /// Trainable, but outside the loss: without a gradient either.
        Unused,
    }

    struct Spec {
        shape: Vec<usize>,
        seed: u64,
        exp: i32,
        dtype: DType,
        grad: Grad,
        digest: Option<u64>,
    }

    struct Case {
        name: String,
        norm_type: NormType,
        max_norm: f64,
        specs: Vec<Spec>,
        /// torch's total and coefficient, and the exact total (float64).
        total: f64,
        coef: f64,
        exact: f64,
    }

    impl Case {
        /// The dtype of its gradients: every trained parameter of a case has the same.
        fn dtype(&self) -> DType {
            self.specs
                .iter()
                .find(|s| s.grad == Grad::Yes)
                .map(|s| s.dtype)
                .unwrap()
        }

        fn label(&self) -> String {
            let dtype = if self.dtype() == DType::BF16 {
                "bf16"
            } else {
                "f32"
            };
            format!("{} {dtype}", self.name)
        }
    }

    fn dtype(name: &str) -> DType {
        match name {
            "f32" => DType::F32,
            "bf16" => DType::BF16,
            _ => panic!("{name}: no such dtype in the fixture"),
        }
    }

    fn cases() -> Vec<Case> {
        let j: serde_json::Value =
            serde_json::from_str(include_str!("../reference/clip_grad_norm_torch.json")).unwrap();
        let f = |v: &serde_json::Value| v.as_f64().unwrap();
        j["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| Case {
                name: c["name"].as_str().unwrap().to_string(),
                norm_type: match c["norm_type"].as_str().unwrap() {
                    "2" => NormType::L2,
                    "inf" => NormType::Inf,
                    other => panic!("norm_type {other}"),
                },
                max_norm: f(&c["max_norm"]),
                specs: c["params"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(c["digests"].as_array().unwrap())
                    .map(|(p, d)| Spec {
                        shape: p["shape"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|n| n.as_u64().unwrap() as usize)
                            .collect(),
                        seed: p["seed"].as_u64().unwrap(),
                        exp: p["exp"].as_i64().unwrap() as i32,
                        dtype: dtype(p["dtype"].as_str().unwrap()),
                        grad: match p["grad"].as_str().unwrap() {
                            "yes" => Grad::Yes,
                            "frozen" => Grad::Frozen,
                            "unused" => Grad::Unused,
                            other => panic!("grad {other}"),
                        },
                        digest: d.as_str().map(|h| u64::from_str_radix(h, 16).unwrap()),
                    })
                    .collect(),
                total: f(&c["total"]),
                coef: f(&c["coef"]),
                exact: f(&c["exact"]),
            })
            .collect()
    }

    /// The elements of `spec`'s gradient (a frozen parameter's value), each exact in its dtype.
    fn values(spec: &Spec) -> Vec<f32> {
        let bits = if spec.dtype == DType::BF16 { 8 } else { 24 };
        let n: usize = spec.shape.iter().product();
        let scale = 2f64.powi(-(bits - 1) - spec.exp);
        (0..n as u64)
            .map(|i| {
                let k = (splitmix64((spec.seed << 32) | i) >> (64 - bits)) as i64;
                ((k - (1 << (bits - 1))) as f64 * scale) as f32
            })
            .collect()
    }

    /// The fixture's digest of a gradient's elements, from their bits in order.
    fn digest(bits: impl Iterator<Item = u64>) -> u64 {
        bits.enumerate().fold(0, |h, (i, b)| {
            h.wrapping_add(splitmix64(((i as u64) << 32) | b))
        })
    }

    /// The bits of `x` in `dtype` (f32's, or bf16's, which holds it exactly).
    fn bits_of(x: f32, dtype: DType) -> u64 {
        match dtype {
            DType::BF16 => u64::from(bf16::from_f32(x).to_bits()),
            _ => u64::from(x.to_bits()),
        }
    }

    /// `x * coef` rounded once to `dtype`, as torch's clipped element: for bf16 both factors are
    /// bf16 values, whose product f32 holds exactly, rounded to nearest, ties to even.
    fn scaled(x: f32, coef: f32, dtype: DType) -> f32 {
        match dtype {
            DType::BF16 => bf16::from_f32(x * coef).to_f32(),
            _ => x * coef,
        }
    }

    /// `param`'s gradient, if it has one: its dtype, and its elements as f32 (which holds bf16's).
    fn read_as<const D: usize>(
        grads: &GradientsParams,
        param: &Param<Tensor<D>>,
    ) -> Option<(DType, Vec<f32>)> {
        let data = grads.get::<D>(param.id)?.into_data();
        let values = match data.dtype() {
            DType::BF16 => data
                .try_to_vec::<bf16>()
                .unwrap()
                .into_iter()
                .map(bf16::to_f32)
                .collect(),
            DType::F32 => data.try_to_vec::<f32>().unwrap(),
            other => panic!("a gradient in {other:?}"),
        };
        Some((data.dtype(), values))
    }

    fn read<const D: usize>(grads: &GradientsParams, param: &Param<Tensor<D>>) -> Option<Vec<f32>> {
        read_as(grads, param).map(|(_, values)| values)
    }

    /// One parameter of `spec` on `device`, and its term of the loss.
    fn param<const D: usize>(
        spec: &Spec,
        device: &Device,
    ) -> (Param<Tensor<D>>, Option<Tensor<1>>) {
        let shape: [usize; D] = spec.shape.clone().try_into().unwrap();
        let g = Tensor::<D>::from_data(TensorData::new(values(spec), shape), (device, spec.dtype));
        let zeros = || Tensor::<D>::zeros(shape, (device, spec.dtype));
        match spec.grad {
            // d/dp sum(p * g) is g exactly, and it reaches the function as a run's gradient does.
            Grad::Yes => {
                let p = Param::from_tensor(zeros());
                let term = p.val().mul(g).sum().cast(DType::F32);
                (p, Some(term))
            }
            Grad::Frozen => {
                let p = Param::initialized(ParamId::new(), g.clone());
                let term = p.val().mul(g).sum().cast(DType::F32);
                (p, Some(term))
            }
            Grad::Unused => (Param::from_tensor(zeros()), None),
        }
    }

    /// What one case measures, each a relative error: our total against torch's and against the
    /// exact one, and the largest of every clipped element against torch's.
    struct Measured {
        total: f64,
        vs_torch: f64,
        vs_exact: f64,
        clipped: f64,
    }

    /// The case's module on `device`, its gradients from a backward pass, clipped; every clipped
    /// gradient held to torch's, rebuilt from torch's coefficient and checked against torch's
    /// digest, and every parameter without a gradient asserted to have none still.
    fn run(case: &Case, device: &Device) -> Measured {
        let mut net = Net::new();
        let mut at = Vec::new();
        let mut loss: Option<Tensor<1>> = None;
        for spec in &case.specs {
            let term = match spec.shape.len() {
                1 => {
                    let (p, term) = param::<1>(spec, device);
                    at.push((1, net.r1.len()));
                    net.r1.push(p);
                    term
                }
                2 => {
                    let (p, term) = param::<2>(spec, device);
                    at.push((2, net.r2.len()));
                    net.r2.push(p);
                    term
                }
                3 => {
                    let (p, term) = param::<3>(spec, device);
                    at.push((3, net.r3.len()));
                    net.r3.push(p);
                    term
                }
                4 => {
                    let (p, term) = param::<4>(spec, device);
                    at.push((4, net.r4.len()));
                    net.r4.push(p);
                    term
                }
                rank => panic!("rank {rank}"),
            };
            loss = match (loss, term) {
                (Some(l), Some(t)) => Some(l + t),
                (l, t) => l.or(t),
            };
        }
        let mut grads = GradientsParams::from_grads(loss.unwrap().backward(), &net);
        let held = grads.len();
        let options = ClipGradNormOptions {
            max_norm: case.max_norm,
            norm_type: case.norm_type,
            ..Default::default()
        };
        let total = f64::from(clip_grad_norm(&net, &mut grads, &options).into_scalar::<f32>());
        let label = case.label();
        assert_eq!(grads.len(), held, "{label}: gradients added or lost");

        let coef = case.coef as f32;
        let mut clipped = 0f64;
        for (spec, &(rank, i)) in case.specs.iter().zip(&at) {
            let got = match rank {
                1 => read_as(&grads, &net.r1[i]),
                2 => read_as(&grads, &net.r2[i]),
                3 => read_as(&grads, &net.r3[i]),
                _ => read_as(&grads, &net.r4[i]),
            };
            let Some((dtype, got)) = got else {
                assert_ne!(spec.grad, Grad::Yes, "{label}: a gradient went missing");
                continue;
            };
            assert_eq!(
                spec.grad,
                Grad::Yes,
                "{label}: a parameter without one got a gradient"
            );
            assert_eq!(dtype, spec.dtype, "{label}: a gradient changed dtype");
            let g = values(spec);
            let want: Vec<f32> = g.iter().map(|&x| scaled(x, coef, spec.dtype)).collect();
            assert_eq!(
                digest(want.iter().map(|&x| bits_of(x, spec.dtype))),
                spec.digest.unwrap(),
                "{label}: the rebuilt reference is not torch's output"
            );
            if coef == 1.0 {
                // Nothing to clip: the gradients come back as they went in.
                for (a, b) in got.iter().zip(&g) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{label}: a gradient changed");
                }
            }
            for (a, b) in got.iter().zip(&want) {
                let err = if *b == 0.0 {
                    if *a == 0.0 { 0.0 } else { f64::INFINITY }
                } else {
                    (f64::from(*a) - f64::from(*b)).abs() / f64::from(*b).abs()
                };
                clipped = clipped.max(err);
            }
        }
        Measured {
            total,
            vs_torch: (total - case.total).abs() / case.total,
            vs_exact: (total - case.exact).abs() / case.exact,
            clipped,
        }
    }

    /// f32's unit roundoff: one rounding to nearest moves a value by at most this, relatively.
    const U: f64 = f32::EPSILON as f64 / 2.0;

    /// Totals against torch's, the 2-norm of f32 gradients. torch's CPU kernel sums a tensor's
    /// squares in 8 f32 lanes, each lane in sequence and each square rounded before it is added
    /// (an emulation reproduces its norms bit for bit): chains of up to 131,072 here, which land
    /// 2.6e-6 (eight gradients, 2.19M elements) and 8.0e-6 (one, 1M) from the exact totals, where
    /// torch on CUDA lands 6.3e-8 off at most. The bound is torch's own error, rounded up; ours is
    /// far smaller (below).
    const TORCH_F32: f64 = 1e-5;
    /// The clipped f32 elements against torch's: the coefficient is inversely proportional to
    /// the total, so it carries the total's relative error, and each side rounds it three times
    /// (the sum, the reciprocal, the product) and the element once more.
    const TORCH_F32_CLIPPED: f64 = TORCH_F32 + 8.0 * U;
    /// Totals against torch's, the 2-norm of bf16 gradients: torch rounds each gradient's norm
    /// to bf16 and the total again, 2^-8 each at most, after summing the squares in f32 (its f32
    /// kernel lands 5.4e-5 off at most on these values).
    const TORCH_BF16: f64 = 1.0 / 128.0 + 1e-4;
    /// Totals against the exact ones, f32 gradients. Flex sums a tensor's squares in 8
    /// accumulators of SIMD lanes, each lane in sequence: 128 lanes with AVX-512 (chains of up to
    /// 8,192 here), 64 with AVX2, 32 with 4-lane SIMD. CubeCL sums in a tree. Measured: 1.6e-7 on
    /// Flex with AVX-512, 6.3e-8 on CUDA. An emulation of Flex's sum, which reproduces Flex's
    /// errors, gives 4.8e-7 with 8 lanes and 6.9e-7 with 4.
    const EXACT_F32: f64 = 1e-6;
    /// Totals against the exact ones, bf16 gradients. A bf16 square has at most 16 significant
    /// bits, so once a chain's sum is large the bits it drops are alike from one square to the
    /// next, a drift that does not average out. Measured: 2.8e-5 on Flex with AVX-512, 1.9e-7 on
    /// CUDA; emulated, 4.3e-5 with 8 lanes and 3.3e-5 with 4. Squared and summed in bf16 instead,
    /// the single gradient's total lands 4.7e-4 off.
    const EXACT_BF16: f64 = 5e-5;

    /// Every recorded case, on the default device (CUDA in a build with `cuda`), against torch's
    /// `clip_grad_norm_`: `reference/clip_grad_norm_torch.json`, recorded from torch 2.13 on the
    /// CPU, one thread. Each line prints the measured errors with their bounds (`--nocapture`):
    ///
    ///   cargo test --release -p dash-train --lib clip -- --nocapture
    ///   CUBECL_CUDA_TF32=0 cargo test --release -p dash-train --features cuda --lib clip
    ///   CUBECL_CUDA_TF32=0 cargo test --release -p dash-train --features cuda,fusion,autotune --lib clip
    ///
    /// Element `i` of a parameter of seed `s` and exponent `e` is `2^-e * (k - 2^(b-1)) /
    /// 2^(b-1)`, `k` the top `b` bits of `splitmix64((s << 32) | i)`: `b` 24 for f32, 8 for
    /// bf16, every value exact in its dtype, so both sides are fed the same bits. The fixture
    /// keeps torch's total, its coefficient, the exact total (float64), and a digest of each
    /// clipped gradient: the sum, wrapping, of `splitmix64((i << 32) | bits_i)` over its elements.
    /// The test rebuilds torch's output as `g * coef` rounded once to the gradient's dtype (the
    /// recording asserted that this is torch's output bit for bit), checks the digest, and holds
    /// every element to it.
    ///
    /// The cases: `clips`, eight gradients of four ranks, 2.19M elements, whose norms are of the
    /// same order (max_norm 1, total 1194); `under`, the same with max_norm 1500, which must
    /// leave every gradient as it was; `inf`, the same under the inf-norm (max_norm 0.3, the
    /// largest element in the fifth gradient); `frozen`, f32 adapters with frozen bf16 weights of
    /// up to 2^12 and a trainable parameter outside the loss, none of which may count or get a
    /// gradient (max_norm 0.7); `single`, one gradient of 1,000,003 elements (max_norm 5). Each
    /// in f32 and bf16, but for `frozen`.
    ///
    /// Bounds: see the constants above. Under the inf-norm the total is the largest absolute
    /// element on every side, exact, and so is the coefficient, both bit for bit: bound 0. With
    /// bf16 gradients torch's coefficient, from its bf16 total, is in each case the bf16 value the
    /// exact coefficient rounds to (the recording checked it), and so is ours: the exact
    /// coefficient sits at least 2.6e-4 of itself from a bf16 rounding boundary, more than our
    /// total may be off. Every clipped element then matches torch's bit for bit: bound 0.
    #[test]
    fn every_case_is_torchs() {
        let device = Device::default().autodiff();
        println!("on {device:?}");
        let mut failed = Vec::new();
        for case in cases() {
            let m = run(&case, &device);
            let bf16 = case.dtype() == DType::BF16;
            let (torch, exact, clipped) = match (case.norm_type, bf16) {
                (NormType::Inf, _) => (0.0, 0.0, 0.0),
                (NormType::L2, false) => (TORCH_F32, EXACT_F32, TORCH_F32_CLIPPED),
                (NormType::L2, true) => (TORCH_BF16, EXACT_BF16, 0.0),
            };
            let label = case.label();
            println!(
                "{label:12} total {:<18} vs torch {:9.3e} (bound {torch:.2e})  vs exact {:9.3e} \
                 (bound {exact:.2e})  clipped vs torch {:9.3e} (bound {clipped:.2e})",
                m.total, m.vs_torch, m.vs_exact, m.clipped
            );
            for (what, err, bound) in [
                ("total vs torch", m.vs_torch, torch),
                ("total vs exact", m.vs_exact, exact),
                ("clipped vs torch", m.clipped, clipped),
            ] {
                if err.is_nan() || err > bound {
                    failed.push(format!("{label}: {what} {err:e} above {bound:e}"));
                }
            }
        }
        assert!(failed.is_empty(), "{failed:#?}");
    }

    /// Two gradients, `a` and `b`, registered on the default device and clipped with `options`:
    /// the total and both gradients, read back.
    fn clip_two(a: &[f32], b: &[f32], options: ClipGradNormOptions) -> (f32, Vec<f32>, Vec<f32>) {
        let device = Device::default();
        let tensor = |v: &[f32]| Tensor::<1>::from_data(TensorData::from(v), &device);
        let mut net = Net::new();
        net.r1.push(Param::from_tensor(tensor(a)));
        net.r1.push(Param::from_tensor(tensor(b)));
        let mut grads = GradientsParams::new();
        grads.register(net.r1[0].id, tensor(a));
        grads.register(net.r1[1].id, tensor(b));
        let total = clip_grad_norm(&net, &mut grads, &options).into_scalar::<f32>();
        let a = read(&grads, &net.r1[0]).unwrap();
        let b = read(&grads, &net.r1[1]).unwrap();
        (total, a, b)
    }

    /// torch's own outputs, `clip_grad_norm_` with `error_if_nonfinite=False` (torch 2.13):
    ///
    /// ```text
    /// [1, inf, -2], [3, -4]  2-norm    total inf  [0, nan, -0], [0, -0]
    /// [1, nan, -2], [3, -4]  2-norm    total nan  all nan
    /// [1e20, -2], [3, -4]    2-norm    total inf  [0, -0], [0, -0]
    /// [1, nan, -2], [3, -4]  inf-norm  total nan  all nan
    /// [1, -inf, -2], [3, -4] inf-norm  total inf  [0, nan, -0], [0, -0]
    /// ```
    ///
    /// The NaN is also put among 100,003 elements, which the reductions take through their
    /// vector paths.
    #[test]
    fn a_total_not_finite_scales_as_torchs_by_default() {
        let (inf, nan) = (f32::INFINITY, f32::NAN);
        let l2 = ClipGradNormOptions::default();
        let max = ClipGradNormOptions {
            norm_type: NormType::Inf,
            ..l2
        };
        let zero_or_nan = |v: &[f32], nan_at: Option<usize>| {
            v.iter().enumerate().all(|(i, x)| {
                if Some(i) == nan_at {
                    x.is_nan()
                } else {
                    *x == 0.0
                }
            })
        };
        for (a, options, nan_at) in [
            ([1.0, inf, -2.0], l2, Some(1)),
            ([1e20, -2.0, 0.0], l2, None),
            ([1.0, -inf, -2.0], max, Some(1)),
        ] {
            let (total, a, b) = clip_two(&a, &[3.0, -4.0], options);
            assert_eq!(total, inf, "{options:?}");
            assert!(
                zero_or_nan(&a, nan_at) && zero_or_nan(&b, None),
                "{a:?} {b:?}"
            );
        }
        let mut long = vec![0.5; 100_003];
        long[54_321] = nan;
        for options in [l2, max] {
            for a in [&[1.0, nan, -2.0][..], &long[..]] {
                let (total, a, b) = clip_two(a, &[3.0, -4.0], options);
                assert!(total.is_nan(), "{options:?}: {total}");
                assert!(a.iter().chain(&b).all(|x| x.is_nan()), "{options:?}");
            }
        }
    }

    #[test]
    fn a_total_not_finite_zeroes_every_gradient_when_asked() {
        let (inf, nan) = (f32::INFINITY, f32::NAN);
        for norm_type in [NormType::L2, NormType::Inf] {
            let options = ClipGradNormOptions {
                norm_type,
                nonfinite: NonFinite::Zero,
                ..Default::default()
            };
            for a in [[1.0, inf, -2.0], [1.0, nan, -2.0], [1e20, -2.0, 0.0]] {
                if norm_type == NormType::Inf && a[0] == 1e20 {
                    continue; // finite under the inf-norm
                }
                let (total, a, b) = clip_two(&a, &[3.0, -4.0], options);
                assert!(!total.is_finite(), "{options:?}: {total}");
                assert!(a.iter().chain(&b).all(|x| *x == 0.0), "{a:?} {b:?}");
            }
            // A finite total clips as under `Scale`.
            let (total, a, b) = clip_two(&[3.0], &[-4.0], options);
            assert_eq!(
                (total, a, b),
                clip_two(
                    &[3.0],
                    &[-4.0],
                    ClipGradNormOptions {
                        norm_type,
                        ..Default::default()
                    }
                )
            );
        }
    }

    /// No gradient at all: a total of 0, as torch's, and nothing registered.
    #[test]
    fn without_a_gradient_the_total_is_zero() {
        let device = Device::default();
        let mut net = Net::new();
        net.r1.push(Param::from_tensor(Tensor::ones([3], &device)));
        let mut grads = GradientsParams::new();
        let total = clip_grad_norm(&net, &mut grads, &ClipGradNormOptions::default());
        assert_eq!(total.into_scalar::<f32>(), 0.0);
        assert!(grads.is_empty());
    }

    /// One parameter the module holds twice counts once and is scaled once.
    #[test]
    fn a_parameter_held_twice_counts_once() {
        let device = Device::default();
        let p = Param::from_tensor(Tensor::<1>::zeros([2], &device));
        let mut net = Net::new();
        net.r1.push(p.clone());
        net.r1.push(p.clone());
        let mut grads = GradientsParams::new();
        grads.register(p.id, Tensor::<1>::from_floats([3.0, -4.0], &device));
        let total = clip_grad_norm(&net, &mut grads, &ClipGradNormOptions::default());
        assert_eq!(total.into_scalar::<f32>(), 5.0);
        let coef = 1.0 / (5.0 + EPS as f32);
        assert_eq!(read(&grads, &p).unwrap(), [3.0 * coef, -4.0 * coef]);
    }

    /// Gradients of two dtypes in one module: one f32 total, each gradient multiplied in its own
    /// dtype by the coefficient rounded to it.
    #[test]
    fn each_gradient_is_scaled_in_its_own_dtype() {
        let device = Device::default();
        let (a, b, c) = ([0.5f32, -1.25, 3.0], [2.0f32, -0.75], [-1.5f32, 0.375, 4.0]);
        let tensor =
            |v: &[f32], dtype| Tensor::<1>::from_data(TensorData::from(v), (&device, dtype));
        let mut net = Net::new();
        let mut grads = GradientsParams::new();
        for (v, dtype) in [(&a[..], DType::F32), (&b, DType::BF16), (&c, DType::F32)] {
            let p = Param::from_tensor(tensor(v, dtype));
            grads.register(p.id, tensor(v, dtype));
            net.r1.push(p);
        }
        let options = ClipGradNormOptions {
            max_norm: 0.7,
            ..Default::default()
        };
        let total = clip_grad_norm(&net, &mut grads, &options).into_scalar::<f32>();
        let exact: f32 = a
            .iter()
            .chain(&b)
            .chain(&c)
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((total - exact).abs() <= exact * 1e-6, "{total} vs {exact}");
        let coef = (total + EPS as f32).recip() * 0.7;
        let coef16 = bf16::from_f32(coef).to_f32();
        assert_eq!(read(&grads, &net.r1[0]).unwrap(), a.map(|x| x * coef));
        assert_eq!(
            read(&grads, &net.r1[1]).unwrap(),
            b.map(|x| bf16::from_f32(x * coef16).to_f32())
        );
        assert_eq!(read(&grads, &net.r1[2]).unwrap(), c.map(|x| x * coef));
    }
}
