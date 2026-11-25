//! Batch normalisation against torch 2.13's `F.batch_norm`, on every side: the output, the new
//! running statistics and the gradients of `x`, the weight and the bias through [`vjp`], in
//! training and in eval mode, from `tests/fixtures/batch_norm_torch.json`
//! (`tools/record_conv2d_batch_norm.py`, torch's values evaluated in float64 from float32
//! inputs). On Burn, also against Burn's own `batch_norm_train` and `batch_norm`: the output the
//! same bits, the gradients within their roundings of the closed form's.
//!
//! Each element is held to a count of roundings: its error over `2^-24` times its magnitude, which
//! the recording keeps beside each value. The magnitude bounds what rounding its operations can
//! cost: the output's `|w| (|x - mean| + mean |x|) / sigma + |b|`, `x`'s gradient's
//! `|w| / sigma (|dy| + mean |dy| + n mean(|dy| n))` with `n = (|x - mean| + mean |x|) / sigma`,
//! the weight's `sum |dy| n`, the bias's `sum |dy|`, the running mean's `(1 - momentum) |rm| +
//! momentum mean |x|` and the running variance's `(1 - momentum) |rv| + momentum m / (m - 1)
//! mean((|x - mean| + mean |x|)^2)`; in eval mode the output's `|w| (|x| + |rm|) / sigma + |b|`,
//! and the gradients' likewise with the running statistics. The `mean |x|` terms carry the error
//! of the mean, which a sum of `m` elements makes, into what depends on it. An element whose value
//! depends on sums of `m` terms is allowed `m + 8` roundings, `m` for the sums in any order and 8
//! for the arithmetic around them; in eval mode, the output and `x`'s gradient, which sum nothing,
//! 8; the running statistics in eval mode are those given, bit for bit.
//!
//!   cargo test --release -p dash-mag --test batch_norm -- --nocapture

use dash_mag::norm::{BatchNorm, BatchNormOptions, Running, batch_norm};
use dash_mag::{DType, Tensor, vjp};
use serde_json::Value;
use std::collections::BTreeMap;

/// f32's unit roundoff.
const UNIT: f64 = 1.0 / (1u64 << 24) as f64;

/// Roundings beyond the sums', for the arithmetic around them.
const ARITHMETIC: f64 = 8.0;

fn fixture() -> Value {
    let path = format!(
        "{}/tests/fixtures/batch_norm_torch.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// A recorded input: f32 values, kept exactly in the recording's f64.
fn input(case: &Value, key: &str, shape: &[usize]) -> Tensor {
    let values: Vec<f32> = floats(&case[key]).into_iter().map(|v| v as f32).collect();
    Tensor::from_f32(&values, shape).unwrap()
}

/// One recorded case, its inputs as tensors.
struct Case {
    name: String,
    value: Value,
    shape: Vec<usize>,
    training: bool,
    affine: bool,
    options: BatchNormOptions,
    /// Elements per channel.
    m: usize,
}

impl Case {
    fn all() -> Vec<Case> {
        let record = fixture();
        let cases: BTreeMap<String, Value> =
            serde_json::from_value(record["cases"].clone()).unwrap();
        cases
            .into_iter()
            .map(|(name, value)| {
                let shape: Vec<usize> = value["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect();
                let m = shape.iter().product::<usize>() / shape[1];
                Case {
                    name,
                    training: value["training"].as_bool().unwrap(),
                    affine: value["affine"].as_bool().unwrap(),
                    options: BatchNormOptions::default()
                        .with_eps(value["eps"].as_f64().unwrap())
                        .with_momentum(value["momentum"].as_f64().unwrap()),
                    m,
                    shape,
                    value,
                }
            })
            .collect()
    }

    fn channels(&self) -> usize {
        self.shape[1]
    }

    fn x(&self) -> Tensor {
        input(&self.value, "x", &self.shape)
    }

    fn d_out(&self) -> Tensor {
        input(&self.value, "d_out", &self.shape)
    }

    fn vector(&self, key: &str) -> Tensor {
        input(&self.value, key, &[self.channels()])
    }

    fn running(&self) -> Running {
        Running {
            mean: self.vector("running_mean"),
            var: self.vector("running_var"),
        }
    }

    /// The roundings an element of `part` is allowed.
    fn allowed(&self, part: &str) -> f64 {
        let sums = match (self.training, part) {
            (false, "out" | "grad_x") => 0.0,
            _ => self.m as f64,
        };
        sums + ARITHMETIC
    }

    /// `got`'s largest error against the recorded `part` in roundings of each element's
    /// magnitude, asserted under what the part is allowed. `key` names the recorded value where it
    /// is not `part` (the new running statistics).
    fn count(&self, part: &str, key: &str, got: &Tensor) -> f64 {
        let want = floats(&self.value[key]);
        let magnitude = floats(&self.value["magnitude"][part]);
        let got = got.to_f32().unwrap();
        assert_eq!(got.len(), want.len(), "{} {part}", self.name);
        let allowed = self.allowed(part);
        let gamma = allowed * UNIT / (1.0 - allowed * UNIT);
        let mut worst = 0f64;
        for (i, ((&g, &w), &s)) in got.iter().zip(&want).zip(&magnitude).enumerate() {
            let err = (g as f64 - w).abs();
            assert!(
                err <= gamma * s,
                "{} {part}[{i}]: {g} against {w}, {:.2} roundings of {s:.3e}, {allowed} allowed",
                self.name,
                err / (UNIT * s)
            );
            worst = worst.max(err / (UNIT * s));
        }
        println!(
            "{} {part}: at most {worst:.2} roundings ({allowed} allowed)",
            self.name
        );
        worst
    }

    /// The call through [`vjp`]: the output and the running statistics, and the gradients of `x`
    /// and, where the case has them, the weight and the bias, the statistics carrying no
    /// cotangent.
    fn run(&self) -> (BatchNorm, Vec<Tensor>) {
        let running = self.running();
        let mut primals = vec![self.x()];
        if self.affine {
            primals.push(self.vector("weight"));
            primals.push(self.vector("bias"));
        }
        let (outs, grads) = vjp(
            |p| {
                let out = batch_norm(
                    &p[0],
                    p.get(1),
                    p.get(2),
                    Some(&running),
                    self.training,
                    self.options,
                )?;
                let running = out.running.expect("the running statistics");
                Ok(vec![out.out, running.mean, running.var])
            },
            &primals,
            &[Some(self.d_out()), None, None],
        )
        .unwrap();
        let [out, mean, var] = <[Tensor; 3]>::try_from(outs).unwrap();
        let out = BatchNorm {
            out,
            running: Some(Running { mean, var }),
        };
        (out, grads)
    }
}

#[test]
fn matches_torch_in_training_and_in_eval_mode() {
    let mut worst: BTreeMap<String, f64> = BTreeMap::new();
    let cases = Case::all();
    assert!(cases.iter().any(|c| c.training) && cases.iter().any(|c| !c.training));
    for case in &cases {
        let (out, grads) = case.run();
        let mut note = |part: &str, roundings: f64| {
            let key = format!("{} {part}", if case.training { "training" } else { "eval" });
            let w = worst.entry(key).or_default();
            *w = w.max(roundings);
        };
        assert_eq!(out.out.shape(), case.shape);
        assert_eq!(out.out.dtype(), DType::F32);
        note("out", case.count("out", "out", &out.out));
        let running = out.running.unwrap();
        if case.training {
            note(
                "running_mean",
                case.count("running_mean", "new_running_mean", &running.mean),
            );
            note(
                "running_var",
                case.count("running_var", "new_running_var", &running.var),
            );
        } else {
            let given = case.running();
            for (what, got, want) in [
                ("mean", &running.mean, &given.mean),
                ("variance", &running.var, &given.var),
            ] {
                let bits = |t: &Tensor| -> Vec<u32> {
                    t.to_f32().unwrap().iter().map(|v| v.to_bits()).collect()
                };
                assert_eq!(
                    bits(got),
                    bits(want),
                    "{}: eval moved the {what}",
                    case.name
                );
            }
        }
        note("grad_x", case.count("grad_x", "grad_x", &grads[0]));
        if case.affine {
            note(
                "grad_weight",
                case.count("grad_weight", "grad_weight", &grads[1]),
            );
            note("grad_bias", case.count("grad_bias", "grad_bias", &grads[2]));
        }
    }
    for (part, roundings) in &worst {
        println!("worst {part}: {roundings:.2} roundings");
    }
}

/// The running variance is torch's, the unbiased batch variance, and the test tells it from
/// Burn's, the biased one: at every recorded training case the two differ by more than the
/// roundings the running variance is allowed.
#[test]
fn the_running_variance_is_unbiased_as_torchs() {
    for case in Case::all().iter().filter(|c| c.training) {
        let momentum = case.options.momentum;
        let m = case.m as f64;
        let before = floats(&case.value["running_var"]);
        let after = floats(&case.value["new_running_var"]);
        let magnitude = floats(&case.value["magnitude"]["running_var"]);
        let allowed = case.allowed("running_var");
        for c in 0..case.channels() {
            let unbiased = (after[c] - (1.0 - momentum) * before[c]) / momentum;
            let biased_gap = momentum * unbiased / m;
            assert!(
                biased_gap > allowed * UNIT * magnitude[c],
                "{} channel {c}: the biased variance is within the bound ({biased_gap:e})",
                case.name
            );
        }
    }
}

#[test]
fn training_and_eval_without_affine_or_running_statistics() {
    let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 7.0], &[3, 2]).unwrap();
    let options = BatchNormOptions::default();
    // Training without running statistics normalises and keeps none.
    let out = batch_norm(&x, None, None, None, true, options).unwrap();
    assert!(out.running.is_none());
    let y = out.out.to_f32().unwrap();
    // Each channel's values normalised to a mean of 0.
    assert!((y[0] + y[2] + y[4]).abs() < 1e-6 && (y[1] + y[3] + y[5]).abs() < 1e-6);
    // Eval mode without them is refused, as torch refuses it.
    assert!(batch_norm(&x, None, None, None, false, options).is_err());
    // torch's initial statistics leave eval as `x / sqrt(1 + eps)`.
    let running = Running::new(2).unwrap();
    let out = batch_norm(&x, None, None, Some(&running), false, options).unwrap();
    let y = out.out.to_f32().unwrap();
    let scale = (1.0f32 + 1e-5).sqrt();
    for (y, x) in y.iter().zip([1.0f32, 2.0, 3.0, 4.0, 5.0, 7.0]) {
        assert!((y - x / scale).abs() <= f32::EPSILON * x, "{y} against {x}");
    }
}

#[test]
fn bf16_comes_back_in_bf16_with_f32_statistics() {
    let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 7.0, -1.0, 0.5], &[2, 2, 2])
        .unwrap()
        .as_dtype(DType::BF16)
        .unwrap();
    let running = Running::new(2).unwrap();
    let out = batch_norm(
        &x,
        None,
        None,
        Some(&running),
        true,
        BatchNormOptions::default(),
    )
    .unwrap();
    assert_eq!(out.out.dtype(), DType::BF16);
    let running = out.running.unwrap();
    assert_eq!(running.mean.dtype(), DType::F32);
    assert_eq!(running.var.dtype(), DType::F32);
}

#[test]
fn what_torch_refuses_is_refused() {
    let options = BatchNormOptions::default();
    let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let one = Tensor::from_f32(&[1.0, 2.0], &[1, 2]).unwrap();
    let three = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
    // One value per channel has no unbiased variance.
    assert!(batch_norm(&one, None, None, None, true, options).is_err());
    assert!(
        batch_norm(
            &one,
            None,
            None,
            Some(&Running::new(2).unwrap()),
            false,
            options
        )
        .is_ok()
    );
    // A rank under 2, a weight, bias or statistic of the wrong length.
    let flat = Tensor::from_f32(&[1.0, 2.0], &[2]).unwrap();
    assert!(batch_norm(&flat, None, None, None, true, options).is_err());
    assert!(batch_norm(&x, Some(&three), None, None, true, options).is_err());
    assert!(batch_norm(&x, None, Some(&three), None, true, options).is_err());
    let wrong = Running {
        mean: three.clone(),
        var: three,
    };
    assert!(batch_norm(&x, None, None, Some(&wrong), true, options).is_err());
    // Options out of range.
    assert!(batch_norm(&x, None, None, None, true, options.with_momentum(1.5)).is_err());
    assert!(batch_norm(&x, None, None, None, true, options.with_eps(-1.0)).is_err());
    assert!(batch_norm(&x, None, None, None, true, options.with_eps(f64::NAN)).is_err());
}

/// On Burn, the output is the bits of Burn's `batch_norm_train` in training and of `batch_norm` in
/// eval mode, which the gateway's is written in the order of; the gradients, composed here and
/// Burn's closed form there, differ by their roundings, which this measures.
#[cfg(not(target_os = "macos"))]
#[test]
fn against_burns_own_batch_norm() {
    use burn::tensor::module::{batch_norm as burn_eval, batch_norm_train};
    use burn::tensor::{Device, Tensor as Burn, TensorData};

    fn device() -> Device {
        #[cfg(feature = "cuda")]
        {
            Device::cuda(0)
        }
        #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
        {
            Device::rocm(0)
        }
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        {
            Device::flex()
        }
    }

    fn burn<const D: usize>(case: &Case, key: &str, shape: [usize; D]) -> Burn<D> {
        let values: Vec<f32> = floats(&case.value[key])
            .into_iter()
            .map(|v| v as f32)
            .collect();
        Burn::<D>::from_data(TensorData::new(values, shape), &device())
    }

    fn bits<const D: usize>(t: Burn<D>) -> Vec<u32> {
        let values = t.inner().into_data().try_to_vec::<f32>().unwrap();
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn mine(t: &Tensor) -> Vec<u32> {
        t.to_f32().unwrap().iter().map(|v| v.to_bits()).collect()
    }

    let mut compared = 0;
    for case in Case::all() {
        let (Ok(shape), true) = (<[usize; 4]>::try_from(case.shape.clone()), case.affine) else {
            continue;
        };
        let c = case.channels();
        let (out, grads) = case.run();
        let x = burn(&case, "x", shape).autodiff().require_grad();
        let w = burn(&case, "weight", [c]).autodiff().require_grad();
        let b = burn(&case, "bias", [c]).autodiff().require_grad();
        let eps = case.options.eps;
        let y = if case.training {
            batch_norm_train(x.clone(), w.clone(), b.clone(), eps).output
        } else {
            let (rm, rv) = (
                burn(&case, "running_mean", [c]),
                burn(&case, "running_var", [c]),
            );
            burn_eval(
                x.clone(),
                w.clone(),
                b.clone(),
                rm.autodiff(),
                rv.autodiff(),
                eps,
            )
        };
        assert_eq!(
            mine(&out.out),
            bits(y.clone()),
            "{}: Burn's output",
            case.name
        );
        let g = (y * burn(&case, "d_out", shape).autodiff())
            .sum()
            .backward();
        for (part, i, theirs) in [
            ("grad_x", 0, bits(x.grad(&g).unwrap())),
            ("grad_weight", 1, bits(w.grad(&g).unwrap())),
            ("grad_bias", 2, bits(b.grad(&g).unwrap())),
        ] {
            let ours = grads[i].to_f32().unwrap();
            let magnitude = floats(&case.value["magnitude"][part]);
            let mut apart = 0f64;
            for ((o, t), s) in ours.iter().zip(&theirs).zip(&magnitude) {
                let gap = (*o as f64 - f32::from_bits(*t) as f64).abs();
                apart = apart.max(gap / (UNIT * s));
            }
            let allowed = 2.0 * case.allowed(part);
            println!(
                "{} {part}: Burn's closed form {apart:.2} roundings apart ({allowed} allowed)",
                case.name
            );
            assert!(apart <= allowed, "{} {part}", case.name);
        }
        compared += 1;
    }
    assert!(compared >= 2, "a training and an eval case");
}
