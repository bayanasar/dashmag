//! LoRA's layer and AdamW against values recorded from PEFT 0.20 on torch 2.13 (CPU), in
//! `tests/fixtures/lora`:
//!
//! - `linear.safetensors`: one layer, `x` `[48, 96]`, `W` `[80, 96]`, r 16, alpha 32, with a bf16
//!   base (x and W bf16, A and B f32: PEFT's default for a bf16 model) and with an f32 base: the
//!   output and the gradients of `x`, A and B for an upstream gradient.
//! - `tiny.safetensors` with `peft_tiny`: two blocks of `x + o_proj(silu(q_proj(x)) * v_proj(x))`,
//!   then `head` and cross-entropy over 32 tokens, hidden 64, inner 128, 64 classes; q_proj and
//!   v_proj carry the adapter (r 8, alpha 20), o_proj and head are frozen. Three steps of
//!   [`AdamW`] against `torch.optim.AdamW`, lr 2e-3, betas (0.9, 0.999), eps 1e-8, weight decay
//!   0.01: the logits, loss and gradients, the factors after the first and the third step, every
//!   step's gradients from torch's own factors, and the optimiser alone on torch's gradients. The
//!   bf16 model reads its adapter from the files PEFT's `save_pretrained` wrote.
//! - `multi.safetensors` with `peft_multi`: the same model with a trainable "default" adapter (r 8,
//!   on q_proj and v_proj) and a frozen "reference" one (r 4, rsLoRA, on q_proj and o_proj) over one
//!   base: the logits under each choice and with none, against PEFT's `set_adapter` and
//!   `disable_adapter()`; then, under "default", `CE + 0.5 KL(default || reference)`, its
//!   gradients, and one AdamW step, against PEFT's, and bit for bit against the model holding
//!   "default" alone.
//!
//! Each comparison prints the largest absolute difference and that difference relative to the
//! reference's largest value, to three figures rounded up, and is held to a `(max abs, relative)`
//! pair; a loss to an absolute bound alone. The pairs are dash-lora's on the same recordings: its
//! Flex arm on Flex, which rounds as PyTorch's CPU kernels do at these sizes, and its CUDA arm on
//! every other device (CUDA, ROCm, MLX), whose products sum in other orders. Two exceptions:
//!
//! - Flex's bf16 factors after three free-running steps take the CUDA arm's `(1e-3, 6e-3)` in place
//!   of `(5e-4, 3e-3)`: from torch's own factors after step 1 a bf16 activation rounds to its
//!   neighbour here and not in torch (the loss 2.8e-5 apart, the gradients 5.2e-4), the steps after
//!   compound it, and with this AdamW the factors end 3.7e-3 apart where Burn's ended 1.9e-3.
//! - On MLX the two-adapter model's bf16 logits take dash-mlx's bound on these recordings, one bf16
//!   unit relative with no absolute half: MLX's products round a larger logit to its neighbour.
//!
//! A bf16 gradient is also held to its distance from the same model computed in f64, no more than
//! 1.1 times PEFT's own.
//!
//!   cargo test --release -p dash-mag --test lora_peft -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-mag --features cuda --test lora_peft -- --nocapture
//!   MLX_ENABLE_TF32=0 cargo test --release -p dash-mag --test lora_peft -- --nocapture

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use dash_mag::lora::{ApplyOptions, DEFAULT_ADAPTER, Lora, LoraLinear, PeftAdapter};
use dash_mag::optim::AdamW;
use dash_mag::{DType, Result, Tensor, vjp};
use safetensors::{Dtype as StDtype, SafeTensors};

/// Flex, the CPU: no device feature, and not macOS.
const ON_CPU: bool = cfg!(all(
    not(target_os = "macos"),
    not(feature = "cuda"),
    not(feature = "rocm")
));

const REFERENCE: &str = "reference";
const LAYERS: usize = 2;
const SCALE: f64 = 20.0 / 8.0;
const LR: f64 = 2e-3;
const STEPS: usize = 3;
const BETA: f64 = 0.5;
const ONE_BF16_UNIT: f64 = 1.0 / 128.0;

/// The tests take turns: with Burn's autotune, a thread asking for a shape another thread is
/// tuning runs an untuned candidate meanwhile, and two passes compared bit for bit would then need
/// not run the same kernels.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/lora")
        .join(name)
}

/// One recorded tensor as stored.
struct Recorded {
    dtype: StDtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// Every tensor of a safetensors file.
struct Fixture(HashMap<String, Recorded>);

impl Fixture {
    fn load(name: &str) -> Self {
        let bytes = std::fs::read(fixture(name)).unwrap();
        let file = SafeTensors::deserialize(&bytes).unwrap();
        Fixture(
            file.tensors()
                .into_iter()
                .map(|(key, view)| {
                    let recorded = Recorded {
                        dtype: view.dtype(),
                        shape: view.shape().to_vec(),
                        bytes: view.data().to_vec(),
                    };
                    (key, recorded)
                })
                .collect(),
        )
    }

    fn get(&self, key: &str) -> &Recorded {
        self.0
            .get(key)
            .unwrap_or_else(|| panic!("the fixture has no {key}"))
    }

    /// Every value, in f64.
    fn values(&self, key: &str) -> Vec<f64> {
        let r = self.get(key);
        match r.dtype {
            StDtype::F64 => r
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect(),
            StDtype::F32 => r
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b) as f64)
                .collect(),
            StDtype::BF16 => r
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16) as f64)
                .collect(),
            StDtype::I64 => r
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as f64)
                .collect(),
            other => panic!("{key}: {other:?}"),
        }
    }

    /// A tensor in the stored dtype, f32 or bf16.
    fn tensor(&self, key: &str) -> Tensor {
        let r = self.get(key);
        let dtype = match r.dtype {
            StDtype::F32 => DType::F32,
            StDtype::BF16 => DType::BF16,
            other => panic!("{key}: {other:?}"),
        };
        Tensor::from_bytes(&r.bytes, &r.shape, dtype).unwrap()
    }

    /// The targets as a one-hot `[tokens, classes]` in f32, which picks a log-probability out
    /// exactly: every other term is a zero.
    fn one_hot(&self, classes: usize) -> Tensor {
        let targets = self.values("targets");
        let mut hot = vec![0f32; targets.len() * classes];
        for (row, &t) in targets.iter().enumerate() {
            hot[row * classes + t as usize] = 1.0;
        }
        Tensor::from_f32(&hot, &[targets.len(), classes]).unwrap()
    }
}

fn values(t: &Tensor) -> Vec<f64> {
    t.to_f32().unwrap().into_iter().map(f64::from).collect()
}

/// `x` to three significant figures, rounded up: the form the README and PROGRESS quote.
fn up3(x: f64) -> String {
    if x <= 0.0 {
        return "0".into();
    }
    let mut e = x.log10().floor() as i32;
    let mut m = (x / 10f64.powi(e - 2) * (1.0 - 1e-12)).ceil();
    if m >= 1000.0 {
        m /= 10.0;
        e += 1;
    }
    format!("{:.2}e{e}", m / 100.0)
}

/// `(max abs, relative)`: both halves hold.
type Bound = (f64, f64);

/// Relative alone, as dash-lora held one layer.
fn relative(bound: f64) -> Bound {
    (f64::INFINITY, bound)
}

/// The largest difference and that over the largest reference value, printed, and held to `bound`;
/// returns the relative one.
fn check(label: &str, got: &[f64], want: &[f64], bound: Bound) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}: lengths");
    let (mut worst, mut scale) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        assert!(g.is_finite(), "{label}: {g}");
        worst = worst.max((g - w).abs());
        scale = scale.max(w.abs());
    }
    let relative = if scale > 0.0 { worst / scale } else { worst };
    println!("{label:52} max abs {worst:.3e}  relative {}", up3(relative));
    assert!(
        worst <= bound.0 && relative <= bound.1,
        "{label}: {worst:.3e} and {relative:.3e} against {bound:?}"
    );
    relative
}

/// A loss against PEFT's, held to an absolute bound: the loss is near 4, so a relative one says
/// little.
fn check_loss(label: &str, got: f64, want: f64, bound: f64) {
    let difference = (got - want).abs();
    println!("{label:52} {got:.8} (PEFT {want:.8}) difference {difference:.3e}");
    assert!(
        difference <= bound,
        "{label}: {difference:.3e} above {bound:.1e}"
    );
}

/// A bf16 gradient against the same model's gradient computed exactly (f64): no further from it
/// than 1.1 times PEFT's own; the distance from PEFT's printed.
fn check_against_exact(label: &str, got: &[f64], peft: &[f64], exact: &[f64]) {
    check(
        &format!("{label} vs PEFT"),
        got,
        peft,
        relative(f64::INFINITY),
    );
    let theirs = check(
        "  PEFT's own vs exact",
        peft,
        exact,
        relative(f64::INFINITY),
    );
    let ours = check("  dash-mag vs exact", got, exact, relative(1.1 * theirs));
    println!("  dash-mag {:.2} times PEFT's distance", ours / theirs);
}

/// After AdamW's first step from the same start, factors against torch's, in two parts. The first
/// step moves each element by `lr · g / (|g| + eps)`: where `|g|` is far above eps that is `±lr`
/// whatever the gradient's last bits, but where `|g|` is within a few eps of zero the step follows
/// the gradient's last bits, which differ between two correct computations. Elements with a torch
/// gradient below 100 eps are held only to the bound every AdamW step obeys, at most `2 lr` apart.
fn check_first_step(label: &str, got: &[f64], want: &[f64], grad: &[f64], bound: Bound) {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
    let (mut well, mut ill, mut n_ill) = (0f64, 0f64, 0usize);
    for ((a, e), g) in got.iter().zip(want).zip(grad) {
        if g.abs() >= 100.0 * 1e-8 {
            well = well.max((a - e).abs());
        } else {
            ill = ill.max((a - e).abs());
            n_ill += 1;
        }
    }
    println!(
        "{label:52} max abs {well:.3e}  relative {}; {n_ill} with |g| < 100 eps: {ill:.3e}",
        up3(well / scale)
    );
    assert!(
        well <= bound.0 && well / scale <= bound.1,
        "{label}: {well:.3e} and {:.3e} against {bound:?}",
        well / scale
    );
    assert!(ill <= 2.0 * LR, "{label}: an element moved {ill:.3e}");
}

fn adamw() -> AdamW {
    AdamW::new((0.9, 0.999), 1e-8, 0.01)
}

// One layer.

#[test]
fn one_layer_matches_peft() {
    let _turn = serial();
    let f = Fixture::load("linear.safetensors");
    for tag in ["f32", "bf16"] {
        let key = |k: &str| format!("{tag}.{k}");
        let scale = f.values(&key("scale"))[0];
        let mut layer = LoraLinear::frozen(&f.tensor(&key("w"))).unwrap();
        let lora = Lora::new(&f.tensor(&key("a")), &f.tensor(&key("b")), scale).unwrap();
        layer.insert_adapter(DEFAULT_ADAPTER, lora.clone()).unwrap();
        let (x, gy) = (f.tensor(&key("x")), f.tensor(&key("gy")));
        let mut held = layer.clone();
        let (out, grads) = vjp(
            |p| {
                held.set_factors(p[1].clone(), p[2].clone())?;
                Ok(vec![held.forward(&p[0])?])
            },
            &[x.clone(), lora.a, lora.b],
            &[Some(gy)],
        )
        .unwrap();
        assert_eq!(
            out[0].dtype(),
            x.dtype(),
            "the output has the input's dtype"
        );
        assert_eq!(grads[0].dtype(), x.dtype());
        assert_eq!(
            (grads[1].dtype(), grads[2].dtype()),
            (DType::F32, DType::F32)
        );
        // In x's dtype: a few f32 roundings on Flex, one bf16 unit off the CPU.
        let in_x = if ON_CPU || tag == "f32" {
            1e-6
        } else {
            ONE_BF16_UNIT
        };
        for (what, got, bound) in [
            ("y", &out[0], relative(in_x)),
            ("dx", &grads[0], relative(in_x)),
            ("da", &grads[1], relative(1e-6)),
            ("db", &grads[2], relative(1e-6)),
        ] {
            check(
                &format!("{tag} {what}"),
                &values(got),
                &f.values(&key(what)),
                bound,
            );
        }
    }
}

// The model of two blocks.

#[derive(Clone)]
struct Block {
    q_proj: LoraLinear,
    v_proj: LoraLinear,
    o_proj: LoraLinear,
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        // torch's silu on bf16 computes in f32 and rounds once; so does this.
        let s = q.as_dtype(DType::F32)?.silu()?.as_dtype(q.dtype())?;
        let h = s.mul(&self.v_proj.forward(x)?)?;
        x.add(&self.o_proj.forward(&h)?)
    }
}

/// The model's code knows nothing of adapters: each layer's forward pass, as without them.
#[derive(Clone)]
struct Tiny {
    layers: Vec<Block>,
    head: LoraLinear,
}

impl Tiny {
    /// The frozen model from a fixture's base weights.
    fn frozen(f: &Fixture, prec: &str) -> Self {
        let w = |k: &str| LoraLinear::frozen(&f.tensor(&format!("{prec}.{k}.weight"))).unwrap();
        Tiny {
            layers: (0..LAYERS)
                .map(|i| Block {
                    q_proj: w(&format!("layers.{i}.q_proj")),
                    v_proj: w(&format!("layers.{i}.v_proj")),
                    o_proj: w(&format!("layers.{i}.o_proj")),
                })
                .collect(),
            head: w("head"),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        self.head.forward(&x)?.as_dtype(DType::F32)
    }

    fn named_mut(&mut self) -> Vec<(String, &mut LoraLinear)> {
        let mut out = Vec::new();
        for (i, b) in self.layers.iter_mut().enumerate() {
            out.push((format!("layers.{i}.q_proj"), &mut b.q_proj));
            out.push((format!("layers.{i}.v_proj"), &mut b.v_proj));
            out.push((format!("layers.{i}.o_proj"), &mut b.o_proj));
        }
        out.push(("head".into(), &mut self.head));
        out
    }

    fn named(&self) -> Vec<(String, &LoraLinear)> {
        let mut out = Vec::new();
        for (i, b) in self.layers.iter().enumerate() {
            out.push((format!("layers.{i}.q_proj"), &b.q_proj));
            out.push((format!("layers.{i}.v_proj"), &b.v_proj));
            out.push((format!("layers.{i}.o_proj"), &b.o_proj));
        }
        out.push(("head".into(), &self.head));
        out
    }

    /// The choice for every layer at once, as PEFT's `set_adapter` makes it.
    fn set_adapter(&mut self, choice: Option<&str>) {
        for (_, layer) in self.named_mut() {
            layer.set_adapter(choice);
        }
    }

    /// The layers whose chosen adapter trains, by path, in order.
    fn trained(&self) -> Vec<(String, &LoraLinear)> {
        self.named()
            .into_iter()
            .filter(|(_, l)| l.lora().is_some())
            .collect()
    }

    /// The chosen adapters' factors, A then B, layer by layer: `vjp`'s primals.
    fn factors(&self) -> Vec<Tensor> {
        self.trained()
            .iter()
            .flat_map(|(_, l)| {
                let lora = l.lora().unwrap();
                [lora.a.clone(), lora.b.clone()]
            })
            .collect()
    }

    /// The factors put back into the chosen adapters, in the order of [`Self::factors`].
    fn set_factors(&mut self, factors: &[Tensor]) -> Result<()> {
        let mut factors = factors.chunks(2);
        for (_, layer) in self.named_mut() {
            if layer.lora().is_some() {
                let pair = factors.next().expect("a pair per trained layer");
                layer.set_factors(pair[0].clone(), pair[1].clone())?;
            }
        }
        Ok(())
    }

    /// The chosen adapters' factors under their PEFT names, `<path>.lora_A` and `<path>.lora_B`.
    fn factor_names(&self) -> Vec<String> {
        self.trained()
            .iter()
            .flat_map(|(path, _)| [format!("{path}.lora_A"), format!("{path}.lora_B")])
            .collect()
    }
}

fn log_softmax(logits: &Tensor) -> Result<Tensor> {
    logits.sub(&logits.logsumexp_axis(1, true)?)
}

fn cross_entropy(logits: &Tensor, one_hot: &Tensor) -> Result<Tensor> {
    log_softmax(logits)?
        .mul(one_hot)?
        .sum_axis(1, false)?
        .mean()?
        .neg()
}

/// The loss and its gradients for the chosen adapters' factors.
fn loss_and_grads(
    model: &Tiny,
    x: &Tensor,
    loss: impl Fn(&Tensor) -> Result<Tensor>,
) -> (f64, Vec<Tensor>) {
    let mut held = model.clone();
    let (out, grads) = vjp(
        |p| {
            held.set_factors(p)?;
            Ok(vec![loss(&held.forward(x)?)?])
        },
        &model.factors(),
        &[Some(Tensor::scalar(1.0, DType::F32).unwrap())],
    )
    .unwrap();
    (out[0].item().unwrap() as f64, grads)
}

/// One AdamW step of the chosen adapters' factors.
fn step(model: &mut Tiny, optim: &mut AdamW, grads: &[Tensor]) {
    optim.begin_step();
    let stepped: Vec<Tensor> = model
        .factor_names()
        .iter()
        .zip(model.factors().iter().zip(grads))
        .map(|(name, (p, g))| optim.update(name, p, g, LR).unwrap())
        .collect();
    model.set_factors(&stepped).unwrap();
}

/// The tiny model with its one adapter: as PEFT saved it for bf16, from the recorded start for f32.
fn tiny(f: &Fixture, prec: &str) -> Tiny {
    let mut model = Tiny::frozen(f, prec);
    match prec {
        "bf16" => {
            // Every layer named: those the adapter does not cover are left without it.
            let adapter = PeftAdapter::read(fixture("peft_tiny")).unwrap();
            adapter.apply(model.named_mut(), &()).unwrap();
        }
        _ => {
            for (path, layer) in model.named_mut() {
                if path.ends_with("q_proj") || path.ends_with("v_proj") {
                    let lora = Lora::new(
                        &f.tensor(&format!("{prec}.init.{path}.lora_A")),
                        &f.tensor(&format!("{prec}.init.{path}.lora_B")),
                        SCALE,
                    )
                    .unwrap();
                    layer.insert_adapter(DEFAULT_ADAPTER, lora).unwrap();
                }
            }
        }
    }
    assert_eq!(model.trained().len(), 2 * LAYERS);
    for (_, layer) in model.trained() {
        assert_eq!(layer.lora().unwrap().scale, SCALE);
    }
    model
}

/// dash-lora's bounds for the model, per quantity: `(max abs, relative)` pairs, the losses'
/// absolute. The measured maxima behind them were in dash-lora's `tests/tiny.rs`, removed with it.
struct Bounds {
    logits: Bound,
    loss: f64,
    grad: Bound,
    step1: Bound,
    step3: Bound,
    /// Every step's gradients and loss from torch's factors at that step (teacher-forced).
    forced_grad: Bound,
    forced_loss: f64,
}

fn tiny_bounds(prec: &str) -> Bounds {
    match (ON_CPU, prec) {
        (true, "f32") => Bounds {
            logits: (1e-6, 5e-7),
            loss: 1e-6,
            grad: (2e-8, 1e-6),
            step1: (3e-8, 2e-7),
            step3: (4e-7, 2e-6),
            forced_grad: (2e-8, 1e-6),
            forced_loss: 2e-6,
        },
        (false, "f32") => Bounds {
            logits: (1e-6, 5e-7),
            loss: 1e-6,
            grad: (2e-8, 1e-6),
            step1: (3e-8, 2e-7),
            step3: (4e-7, 3e-6),
            forced_grad: (2e-8, 1e-6),
            forced_loss: 2e-6,
        },
        // Step 3 the CUDA arm's: the module documentation says why.
        (true, _) => Bounds {
            logits: (1e-6, 1e-6),
            loss: 2e-6,
            grad: (1e-7, 5e-6),
            step1: (3e-8, 2e-7),
            step3: (1e-3, 6e-3),
            forced_grad: (2e-5, 1e-3),
            forced_loss: 5e-5,
        },
        (false, _) => Bounds {
            logits: (1.5e-3, 7e-4),
            loss: 2e-6,
            grad: (5e-5, 3e-3),
            step1: (2e-5, 1e-4),
            step3: (1e-3, 6e-3),
            forced_grad: (5e-5, 3e-3),
            forced_loss: 5e-5,
        },
    }
}

fn trains_as_peft_does(prec: &str) {
    let _turn = serial();
    let b = tiny_bounds(prec);
    let f = Fixture::load("tiny.safetensors");
    let mut model = tiny(&f, prec);
    let x = f.tensor(&format!("{prec}.x"));
    let one_hot = f.one_hot(64);
    let losses = f.values(&format!("{prec}.losses"));
    assert_eq!(losses.len(), STEPS);
    let mut optim = adamw();
    for (s, torch_loss) in losses.iter().copied().enumerate() {
        let (loss, grads) = loss_and_grads(&model, &x, |z| cross_entropy(z, &one_hot));
        println!("{prec} step {s}: loss {loss:.7} (torch {torch_loss:.7})");
        if s == 0 {
            check(
                &format!("{prec} logits"),
                &values(&model.forward(&x).unwrap()),
                &f.values(&format!("{prec}.logits")),
                b.logits,
            );
            check_loss(
                &format!("{prec} loss"),
                loss,
                f.values(&format!("{prec}.loss"))[0],
                b.loss,
            );
            for (name, grad) in model.factor_names().iter().zip(&grads) {
                let label = format!("{prec} grad {name}");
                let peft = f.values(&format!("{prec}.grad0.{name}"));
                check(&label, &values(grad), &peft, b.grad);
                if prec == "bf16" {
                    let exact = f.values(&format!("bf16ref.grad0.{name}"));
                    check_against_exact(&label, &values(grad), &peft, &exact);
                }
            }
        }
        step(&mut model, &mut optim, &grads);
        let done = s + 1;
        for (name, factor) in model.factor_names().iter().zip(model.factors()) {
            let label = format!("{prec} after step {done} {name}");
            let want = f.values(&format!("{prec}.step{done}.{name}"));
            if done == 1 {
                let grad = f.values(&format!("{prec}.grad0.{name}"));
                check_first_step(&label, &values(&factor), &want, &grad, b.step1);
            } else if done == STEPS {
                check(&label, &values(&factor), &want, b.step3);
            }
        }
    }
}

#[test]
fn f32_model_trains_as_peft_does() {
    trains_as_peft_does("f32");
}

#[test]
fn bf16_model_trains_as_peft_does() {
    trains_as_peft_does("bf16");
}

/// Every step's gradients from torch's own factors at that step: the forward and backward passes
/// alone, step after step, with no drift carried over.
#[test]
fn every_step_s_gradients_match_from_torch_s_factors() {
    let _turn = serial();
    let f = Fixture::load("tiny.safetensors");
    for prec in ["f32", "bf16"] {
        let b = tiny_bounds(prec);
        let mut model = tiny(&f, prec);
        let x = f.tensor(&format!("{prec}.x"));
        let one_hot = f.one_hot(64);
        let losses = f.values(&format!("{prec}.losses"));
        for (s, torch_loss) in losses.iter().copied().enumerate() {
            if s > 0 {
                let factors: Vec<Tensor> = model
                    .factor_names()
                    .iter()
                    .map(|name| f.tensor(&format!("{prec}.step{s}.{name}")))
                    .collect();
                model.set_factors(&factors).unwrap();
            }
            let (loss, grads) = loss_and_grads(&model, &x, |z| cross_entropy(z, &one_hot));
            check_loss(
                &format!("{prec} from torch's step {s}: loss"),
                loss,
                torch_loss,
                b.forced_loss,
            );
            for (name, grad) in model.factor_names().iter().zip(&grads) {
                check(
                    &format!("{prec} from torch's step {s}: grad {name}"),
                    &values(grad),
                    &f.values(&format!("{prec}.grad{s}.{name}")),
                    b.forced_grad,
                );
            }
        }
    }
}

/// AdamW fed torch's own gradients, step by step: the optimiser alone, against
/// `torch.optim.AdamW`, after the first and the third step.
#[test]
fn adamw_matches_torch_on_torch_s_gradients() {
    let _turn = serial();
    let f = Fixture::load("tiny.safetensors");
    // dash-lora's, from Burn's AdamW on these gradients: 2.3e-7 on Flex, 3.4e-7 on CUDA with fusion.
    let bound = if ON_CPU { (5e-8, 4e-7) } else { (7e-8, 4.5e-7) };
    for prec in ["f32", "bf16"] {
        let mut model = tiny(&f, prec);
        let mut optim = adamw();
        for s in 0..STEPS {
            let grads: Vec<Tensor> = model
                .factor_names()
                .iter()
                .map(|name| f.tensor(&format!("{prec}.grad{s}.{name}")))
                .collect();
            step(&mut model, &mut optim, &grads);
            let done = s + 1;
            if done == 1 || done == STEPS {
                for (name, factor) in model.factor_names().iter().zip(model.factors()) {
                    check(
                        &format!("{prec} AdamW alone, step {done} {name}"),
                        &values(&factor),
                        &f.values(&format!("{prec}.step{done}.{name}")),
                        bound,
                    );
                }
            }
        }
    }
}

// Two adapters over one base.

/// The model holding "default" from `peft_multi`, trainable, and "reference" from its
/// subdirectory, frozen; or one of them alone.
fn multi(f: &Fixture, prec: &str, held: &[&str]) -> Tiny {
    let mut model = Tiny::frozen(f, prec);
    for &name in held {
        let dir = match name {
            DEFAULT_ADAPTER => fixture("peft_multi"),
            other => fixture("peft_multi").join(other),
        };
        let adapter = PeftAdapter::read(dir).unwrap();
        let options = ApplyOptions {
            name: name.into(),
            trainable: name == DEFAULT_ADAPTER,
        };
        adapter
            .apply_with(model.named_mut(), &(), &options)
            .unwrap();
    }
    model
}

fn kl_loss(logits: &Tensor, reference: &Tensor, one_hot: &Tensor) -> Result<Tensor> {
    let logp = log_softmax(logits)?;
    let ce = logp.mul(one_hot)?.sum_axis(1, false)?.mean()?.neg()?;
    let refp = log_softmax(reference)?;
    let kl = logp
        .exp()?
        .mul(&logp.sub(&refp)?)?
        .sum_axis(1, false)?
        .mean()?;
    ce.add(&kl.mul_scalar(BETA)?)
}

/// dash-lora's bounds for two adapters over one base, as [`Bounds`] are for the model.
struct MultiBounds {
    logits: Bound,
    loss: f64,
    grad: Bound,
    step1: Bound,
}

fn multi_bounds(prec: &str) -> MultiBounds {
    match (ON_CPU, prec) {
        (_, "f32") => MultiBounds {
            logits: (1e-6, 5e-7),
            loss: 1e-6,
            grad: (2e-8, 1e-6),
            step1: (3e-8, 2e-7),
        },
        (true, _) => MultiBounds {
            logits: (0.0, 0.0),
            loss: 1e-6,
            grad: (7e-8, 3.5e-6),
            step1: (3e-8, 2e-7),
        },
        // On MLX the logits dash-mlx's: the module documentation says why.
        (false, _) => MultiBounds {
            logits: if cfg!(target_os = "macos") {
                relative(ONE_BF16_UNIT)
            } else {
                (1.5e-3, 7e-4)
            },
            loss: 1e-6,
            grad: (3e-5, 1.5e-3),
            step1: (3e-7, 2.5e-6),
        },
    }
}

#[test]
fn every_layer_switches_and_trains_against_the_reference_as_peft_does() {
    let _turn = serial();
    let f = Fixture::load("multi.safetensors");
    let one_hot = f.one_hot(64);
    for prec in ["f32", "bf16"] {
        let b = multi_bounds(prec);
        let x = f.tensor(&format!("{prec}.x"));
        let mut model = multi(&f, prec, &[DEFAULT_ADAPTER, REFERENCE]);
        // Kernels compiled and tuned before the passes compared.
        {
            let warm = multi(&f, prec, &[DEFAULT_ADAPTER]);
            let logits = warm.forward(&x).unwrap();
            let reference = logits.stop_gradient().unwrap();
            loss_and_grads(&warm, &x, |z| kl_loss(z, &reference, &one_hot));
        }
        for choice in [Some(DEFAULT_ADAPTER), Some(REFERENCE), None] {
            model.set_adapter(choice);
            let name = choice.unwrap_or("base");
            check(
                &format!("{prec} logits, {name}"),
                &values(&model.forward(&x).unwrap()),
                &f.values(&format!("{prec}.logits.{name}")),
                b.logits,
            );
            // The same bits as a model holding that adapter alone.
            let mut alone = multi(&f, prec, choice.as_slice());
            alone.set_adapter(choice);
            assert!(
                model.forward(&x).unwrap().to_bytes().unwrap()
                    == alone.forward(&x).unwrap().to_bytes().unwrap(),
                "{prec} {name}: not what it gives alone"
            );
        }
        let reference_before: Vec<Vec<u8>> = reference_factors(&model);

        // The reference's logits, then the policy's loss, gradients and one step.
        model.set_adapter(Some(REFERENCE));
        let reference = model.forward(&x).unwrap().stop_gradient().unwrap();
        model.set_adapter(Some(DEFAULT_ADAPTER));
        let (loss, grads) = loss_and_grads(&model, &x, |z| kl_loss(z, &reference, &one_hot));
        check_loss(
            &format!("{prec} loss"),
            loss,
            f.values(&format!("{prec}.loss"))[0],
            b.loss,
        );
        // "default" holds q_proj and v_proj of each block: its 8 factors alone are the primals.
        assert_eq!(model.factor_names().len(), 4 * LAYERS);
        for (name, grad) in model.factor_names().iter().zip(&grads) {
            let label = format!("{prec} grad {name}");
            let peft = f.values(&format!("{prec}.grad.{name}"));
            check(&label, &values(grad), &peft, b.grad);
            if prec == "bf16" {
                let exact = f.values(&format!("bf16ref.grad.{name}"));
                check_against_exact(&label, &values(grad), &peft, &exact);
            }
        }
        let mut optim = adamw();
        step(&mut model, &mut optim, &grads);
        for (name, factor) in model.factor_names().iter().zip(model.factors()) {
            check_first_step(
                &format!("{prec} after the step {name}"),
                &values(&factor),
                &f.values(&format!("{prec}.step1.{name}")),
                &f.values(&format!("{prec}.grad.{name}")),
                b.step1,
            );
        }
        // AdamW holds moments for "default"'s factors alone; the reference has not moved.
        assert_eq!(optim.named_state().count(), 4 * LAYERS);
        assert!(reference_factors(&model) == reference_before, "{prec}");

        // The same step for the model holding "default" alone: the same bits.
        let mut single = multi(&f, prec, &[DEFAULT_ADAPTER]);
        let (single_loss, single_grads) =
            loss_and_grads(&single, &x, |z| kl_loss(z, &reference, &one_hot));
        assert_eq!(single_loss.to_bits(), loss.to_bits(), "{prec} loss alone");
        for ((name, g), s) in model.factor_names().iter().zip(&grads).zip(&single_grads) {
            assert!(
                g.to_bytes().unwrap() == s.to_bytes().unwrap(),
                "{prec} grad {name} alone"
            );
        }
        let mut single_optim = adamw();
        step(&mut single, &mut single_optim, &single_grads);
        for ((name, a), s) in model
            .factor_names()
            .iter()
            .zip(model.factors())
            .zip(single.factors())
        {
            assert!(
                a.to_bytes().unwrap() == s.to_bytes().unwrap(),
                "{prec} {name} after the step alone"
            );
        }
    }
}

fn reference_factors(model: &Tiny) -> Vec<Vec<u8>> {
    model
        .named()
        .iter()
        .filter_map(|(_, l)| l.adapter(REFERENCE))
        .flat_map(|lora| [lora.a.to_bytes().unwrap(), lora.b.to_bytes().unwrap()])
        .collect()
}
