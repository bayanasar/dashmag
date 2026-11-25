//! AdamW under `OneCycleLR` and clipping by the global norm, against values recorded from torch
//! 2.13 (`tests/fixtures/adamw_torch.json`, `clip_grad_norm_torch.json`). Each test prints its
//! measured errors with their bounds (`--nocapture`):
//!
//!   cargo test --release -p dash-mag --test optim_torch -- --nocapture

use dash_mag::optim::{AdamW, ClipGradNormOptions, NonFinite, NormType, clip_grad_norm};
use dash_mag::schedule::OneCycle;
use dash_mag::{DType, Tensor};
use half::bf16;

fn splitmix64(z: u64) -> u64 {
    let z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn fixture(name: &str) -> serde_json::Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn t(values: &[f32]) -> Tensor {
    Tensor::from_f32(values, &[values.len()]).unwrap()
}

/// `tests/fixtures/adamw_torch.json`, recorded from torch 2.13 on the CPU in float32: two groups as
/// a run builds them, `AdamW([{backbone, lr 3e-4}, {heads, lr 1e-3}], weight_decay=0.05,
/// foreach=False)` under `OneCycleLR(max_lr=[3e-4, 1e-3], total_steps=1875, pct_start=0.1)`, each
/// step `opt.step()`, `sched.step()`, `opt.zero_grad()`; the parameters after each step in
/// `checkpoints`. Here one optimiser, beta1 set before each step from the schedule, each group's
/// parameter updated at its own schedule's rate.
///
/// Element `i` of 12 starts at `init[i]`; its gradient at step `k` in group `g` is `2^-exp[i % 6] *
/// (bias + u - 0.5)`, `bias` 0.25 for `i < 6` and 0 after, `u` the top 24 bits of
/// `splitmix64((k << 16) | (g << 8) | i)` over 2^24. Every value is exact in f32, so both sides
/// are fed the same bits. The scales fall from 1 to 2^-30, past eps: late in the schedule, with the
/// rate small and the second moments built up, the placement of eps and the order of decay and
/// update are what the result turns on.
///
/// Bound 1e-6, absolute, as for the Burn optimiser this recording held first, which ended 2.4e-7
/// off, one f32 step at these magnitudes; decay after the update moved that to 2.3e-5, beta1 a step
/// late to 1.4e-5.
#[test]
fn adamw_over_the_whole_schedule_is_torchs() {
    let j = fixture("adamw_torch.json");
    let ints = |key: &str| -> Vec<usize> {
        let v = j[key].as_array().unwrap();
        v.iter().map(|x| x.as_u64().unwrap() as usize).collect()
    };
    let (steps, exp, checkpoints) = (
        j["steps"].as_u64().unwrap() as usize,
        ints("exp"),
        ints("checkpoints"),
    );
    let init: Vec<f32> = j["init"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let grad = |k: usize, g: usize| -> Vec<f32> {
        (0..init.len())
            .map(|i| {
                let bits = splitmix64(((k as u64) << 16) | ((g as u64) << 8) | i as u64) >> 40;
                let u = bits as f64 / (1u64 << 24) as f64;
                let bias = if i < 6 { 0.25 } else { 0.0 };
                (2f64.powi(-(exp[i % 6] as i32)) * (bias + u - 0.5)) as f32
            })
            .collect()
    };
    let weight_decay = j["weight_decay"].as_f64().unwrap();
    let pct_start = j["pct_start"].as_f64().unwrap();
    // The fixture's groups, in its order: backbone, heads.
    let schedules: Vec<OneCycle> = j["max_lr"]
        .as_array()
        .unwrap()
        .iter()
        .map(|lr| OneCycle::new(lr.as_f64().unwrap(), steps, pct_start))
        .collect();
    let names: Vec<&str> = j["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["name"].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 2);

    let mut optimiser = AdamW::new((0.9, 0.999), 1e-8, weight_decay);
    let mut params = [t(&init), t(&init)];
    let (mut worst, mut seen) = (0f64, 0);
    for k in 0..steps {
        optimiser.betas.0 = schedules[0].at(k).1;
        optimiser.begin_step();
        for (g, p) in params.iter_mut().enumerate() {
            let lr = schedules[g].at(k).0;
            *p = optimiser.update(names[g], p, &t(&grad(k, g)), lr).unwrap();
        }
        if checkpoints.contains(&(k + 1)) {
            for (g, p) in params.iter().enumerate() {
                let got = p.to_f32().unwrap();
                let want = j["groups"][g]["at"][seen].as_array().unwrap();
                assert_eq!(got.len(), want.len());
                for (a, b) in got.iter().zip(want) {
                    worst = worst.max((f64::from(*a) - b.as_f64().unwrap()).abs());
                }
            }
            seen += 1;
        }
    }
    assert_eq!(seen, checkpoints.len());
    assert_eq!(optimiser.steps(names[0]), Some(steps as u64));
    println!("AdamW under OneCycle, {steps} steps: worst difference {worst:e} (bound 1e-6)");
    assert!(worst < 1e-6, "worst difference {worst:e}");
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

fn cases() -> Vec<Case> {
    let j = fixture("clip_grad_norm_torch.json");
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
                    dtype: match p["dtype"].as_str().unwrap() {
                        "f32" => DType::F32,
                        "bf16" => DType::BF16,
                        other => panic!("{other}: no such dtype in the fixture"),
                    },
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

/// `x * coef` rounded once to `dtype`, as torch's clipped element: for bf16 both factors are bf16
/// values, whose product f32 holds exactly, rounded to nearest, ties to even.
fn scaled(x: f32, coef: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => bf16::from_f32(x * coef).to_f32(),
        _ => x * coef,
    }
}

/// What one case measures, each a relative error: our total against torch's and against the exact
/// one, and the largest of every clipped element against torch's.
struct Measured {
    total: f64,
    vs_torch: f64,
    vs_exact: f64,
    clipped: f64,
}

/// The case's gradients, those of its trained parameters, clipped; every clipped gradient held to
/// torch's, rebuilt from torch's coefficient and checked against torch's digest. A frozen
/// parameter and one outside the loss have no gradient, and the caller lists none for them.
fn run(case: &Case) -> Measured {
    let trained: Vec<&Spec> = case.specs.iter().filter(|s| s.grad == Grad::Yes).collect();
    let mut grads: Vec<Tensor> = trained
        .iter()
        .map(|spec| {
            Tensor::from_f32(&values(spec), &spec.shape)
                .unwrap()
                .as_dtype(spec.dtype)
                .unwrap()
        })
        .collect();
    let options = ClipGradNormOptions {
        max_norm: case.max_norm,
        norm_type: case.norm_type,
        ..Default::default()
    };
    let total = f64::from(clip_grad_norm(&mut grads, &options).unwrap());
    let label = case.label();
    assert_eq!(
        grads.len(),
        trained.len(),
        "{label}: gradients added or lost"
    );

    let coef = case.coef as f32;
    let mut clipped = 0f64;
    for (spec, got) in trained.iter().zip(&grads) {
        assert_eq!(got.dtype(), spec.dtype, "{label}: a gradient changed dtype");
        assert_eq!(got.shape(), spec.shape, "{label}: a gradient changed shape");
        let got = got.to_f32().unwrap();
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

/// Totals against torch's, the 2-norm of f32 gradients. torch's CPU kernel sums a tensor's squares
/// in 8 f32 lanes, each lane in sequence and each square rounded before it is added (an emulation
/// reproduces its norms bit for bit): chains of up to 131,072 here, which land 2.6e-6 (eight
/// gradients, 2.19M elements) and 8.0e-6 (one, 1M) from the exact totals, where torch on CUDA lands
/// 6.3e-8 off at most. The bound is torch's own error, rounded up; ours is far smaller (below).
const TORCH_F32: f64 = 1e-5;
/// The clipped f32 elements against torch's: the coefficient is inversely proportional to the
/// total, so it carries the total's relative error, and each side rounds it three times (the sum,
/// the reciprocal, the product) and the element once more.
const TORCH_F32_CLIPPED: f64 = TORCH_F32 + 8.0 * U;
/// Totals against torch's, the 2-norm of bf16 gradients: torch rounds each gradient's norm to bf16
/// and the total again, 2^-8 each at most, after summing the squares in f32 (its f32 kernel lands
/// 5.4e-5 off at most on these values).
const TORCH_BF16: f64 = 1.0 / 128.0 + 1e-4;
/// Totals against the exact ones, f32 gradients. Flex sums a tensor's squares in 8 accumulators of
/// SIMD lanes, each lane in sequence: 128 lanes with AVX-512 (chains of up to 8,192 here), 64 with
/// AVX2, 32 with 4-lane SIMD. CubeCL sums in a tree. Measured for the Burn clipping that first held
/// these bounds: 1.6e-7 on Flex with AVX-512, 6.3e-8 on CUDA; an emulation of Flex's sum, which
/// reproduces Flex's errors, gives 4.8e-7 with 8 lanes and 6.9e-7 with 4.
const EXACT_F32: f64 = 1e-6;
/// Totals against the exact ones, bf16 gradients. A bf16 square has at most 16 significant bits, so
/// once a chain's sum is large the bits it drops are alike from one square to the next, a drift
/// that does not average out. Measured for the Burn clipping: 2.8e-5 on Flex with AVX-512, 1.9e-7
/// on CUDA; emulated, 4.3e-5 with 8 lanes and 3.3e-5 with 4. Squared and summed in bf16 instead,
/// the single gradient's total lands 4.7e-4 off.
const EXACT_BF16: f64 = 5e-5;

/// Every recorded case against torch's `clip_grad_norm_`:
/// `tests/fixtures/clip_grad_norm_torch.json`, recorded from torch 2.13 on the CPU, one thread.
///
/// Element `i` of a parameter of seed `s` and exponent `e` is `2^-e * (k - 2^(b-1)) / 2^(b-1)`, `k`
/// the top `b` bits of `splitmix64((s << 32) | i)`: `b` 24 for f32, 8 for bf16, every value exact
/// in its dtype, so both sides are fed the same bits. The fixture keeps torch's total, its
/// coefficient, the exact total (float64), and a digest of each clipped gradient: the sum,
/// wrapping, of `splitmix64((i << 32) | bits_i)` over its elements. The test rebuilds torch's
/// output as `g * coef` rounded once to the gradient's dtype (the recording asserted that this is
/// torch's output bit for bit), checks the digest, and holds every element to it.
///
/// The cases: `clips`, eight gradients of four ranks, 2.19M elements, whose norms are of the same
/// order (max_norm 1, total 1194); `under`, the same with max_norm 1500, which must leave every
/// gradient as it was; `inf`, the same under the inf-norm (max_norm 0.3, the largest element in the
/// fifth gradient); `frozen`, f32 adapters beside frozen bf16 weights of up to 2^12 and a trainable
/// parameter outside the loss, of which only the adapters have gradients to list (max_norm 0.7);
/// `single`, one gradient of 1,000,003 elements (max_norm 5). Each in f32 and bf16, but for
/// `frozen`.
///
/// Bounds: see the constants above. Under the inf-norm the total is the largest absolute element on
/// every side, exact, and so is the coefficient, both bit for bit: bound 0. With bf16 gradients
/// torch's coefficient, from its bf16 total, is in each case the bf16 value the exact coefficient
/// rounds to (the recording checked it), and so is ours: the exact coefficient sits at least 2.6e-4
/// of itself from a bf16 rounding boundary, more than our total may be off. Every clipped element
/// then matches torch's bit for bit: bound 0.
#[test]
fn clipping_every_case_is_torchs() {
    println!("on {}", dash_mag::describe());
    let mut failed = Vec::new();
    for case in cases() {
        let m = run(&case);
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

/// Two gradients, `a` and `b`, clipped with `options`: the total and both gradients, read back.
fn clip_two(a: &[f32], b: &[f32], options: ClipGradNormOptions) -> (f32, Vec<f32>, Vec<f32>) {
    let mut grads = [t(a), t(b)];
    let total = clip_grad_norm(&mut grads, &options).unwrap();
    (
        total,
        grads[0].to_f32().unwrap(),
        grads[1].to_f32().unwrap(),
    )
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
/// The NaN is also put among 100,003 elements, which the reductions take through their vector
/// paths.
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

/// Gradients of two dtypes: one f32 total, each gradient multiplied in its own dtype by the
/// coefficient rounded to it.
#[test]
fn each_gradient_is_scaled_in_its_own_dtype_by_the_coefficient_rounded_to_it() {
    let (a, b, c) = ([0.5f32, -1.25, 3.0], [2.0f32, -0.75], [-1.5f32, 0.375, 4.0]);
    let mut grads = [t(&a), t(&b).as_dtype(DType::BF16).unwrap(), t(&c)];
    let options = ClipGradNormOptions {
        max_norm: 0.7,
        ..Default::default()
    };
    let total = clip_grad_norm(&mut grads, &options).unwrap();
    let exact: f32 = a
        .iter()
        .chain(&b)
        .chain(&c)
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    assert!((total - exact).abs() <= exact * 1e-6, "{total} vs {exact}");
    let coef = (total + 1e-6f32).recip() * 0.7;
    let coef16 = bf16::from_f32(coef).to_f32();
    assert_eq!(grads[0].to_f32().unwrap(), a.map(|x| x * coef));
    assert_eq!(grads[1].dtype(), DType::BF16);
    assert_eq!(
        grads[1].to_f32().unwrap(),
        b.map(|x| bf16::from_f32(x * coef16).to_f32())
    );
    assert_eq!(grads[2].to_f32().unwrap(), c.map(|x| x * coef));
}
