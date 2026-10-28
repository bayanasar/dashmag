//! LoRA against dash-lora's fixtures, recorded from PEFT 0.20 on torch 2.13, with dash-lora's
//! bounds for builds whose products sum in another order than the CPU's:
//!
//! - `linear.safetensors`: one LoRA'd layer, `x` `[48, 96]`, `W` `[80, 96]`, r 16, alpha 32, with a
//!   bf16 base (x and W bf16, A and B f32) and an f32 base: the output and the gradients of `x`, A
//!   and B for an upstream gradient; only A and B train.
//! - `multi.safetensors` with `peft_multi`: a two-block model whose `q_proj` and `v_proj` hold a
//!   trainable "default" adapter (r 8) and whose `q_proj` and `o_proj` hold a frozen "reference"
//!   one (r 4) over one base: the logits under each choice and with none, and, under the default,
//!   the loss `CE + 0.5 KL(default || reference)` and its gradients, which reach the default
//!   adapter alone. An adapter over a shared base gives what a layer holding it alone gives.
//!
//! Errors are relative to each reference's largest value, printed to three figures and rounded
//! up. f32 needs `MLX_ENABLE_TF32=0`:
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test lora -- --nocapture
#![cfg(target_os = "macos")]

use dash_mlx::lora::{DEFAULT_ADAPTER, Lora, LoraLinear, set_adapter};
use mlx_rs::error::{Exception, Result};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::transforms::fallible_vjp;
use mlx_rs::{Array, Dtype};
use safetensors::{Dtype as StDtype, SafeTensors};
use std::collections::HashMap;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../dash-lora/tests/fixtures");
const REFERENCE: &str = "reference";
const BETA: f32 = 0.5;

fn require_ieee_f32_matmul() {
    assert_eq!(
        std::env::var("MLX_ENABLE_TF32").as_deref(),
        Ok("0"),
        "run with MLX_ENABLE_TF32=0: MLX's f32 matmuls run in TF32 otherwise, about 1e-3 off"
    );
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

/// Every tensor of a safetensors file as an array in its own dtype (f64 read as f32).
fn load(path: &str) -> HashMap<String, Array> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let file = SafeTensors::deserialize(&bytes).expect("safetensors");
    file.tensors()
        .into_iter()
        .map(|(name, view)| {
            let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
            let data = view.data();
            let array = match view.dtype() {
                StDtype::F32 => Array::from_slice(
                    &data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect::<Vec<_>>(),
                    &shape,
                ),
                StDtype::F64 => Array::from_slice(
                    &data
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|b| f64::from_le_bytes(*b) as f32)
                        .collect::<Vec<_>>(),
                    &shape,
                ),
                StDtype::BF16 => Array::from_slice(
                    &data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
                        .collect::<Vec<_>>(),
                    &shape,
                )
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
                StDtype::I64 => Array::from_slice(
                    &data
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|b| i64::from_le_bytes(*b))
                        .collect::<Vec<_>>(),
                    &shape,
                ),
                other => panic!("{name}: dtype {other:?}"),
            };
            (name, array)
        })
        .collect()
}

fn values(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
    a.as_slice::<f32>().to_vec()
}

fn check(label: &str, got: &Array, want: &Array, bound: f64) -> f64 {
    let (g, w) = (values(got), values(want));
    assert_eq!(g.len(), w.len(), "{label}: lengths");
    let (mut abs, mut scale) = (0f64, 0f64);
    for (g, w) in g.iter().zip(&w) {
        assert!(g.is_finite(), "{label}: non-finite {g}");
        abs = abs.max((*g as f64 - *w as f64).abs());
        scale = scale.max((*w as f64).abs());
    }
    let rel = if scale > 0.0 { abs / scale } else { abs };
    println!("{label:44} max abs {abs:.3e}  relative {}", up3(rel));
    assert!(rel <= bound, "{label}: {rel:.3e} above {bound:.1e}");
    rel
}

#[test]
fn one_layer_matches_peft() -> Result<()> {
    require_ieee_f32_matmul();
    let f = load(&format!("{FIXTURES}/linear.safetensors"));
    // dash-lora's bounds off the CPU: f32 a few roundings; bf16 one unit, 2^-7 of the largest.
    for (tag, bound) in [("f32", 1e-6), ("bf16", 2f64.powi(-7))] {
        let key = |k: &str| f[&format!("{tag}.{k}")].clone();
        let scale = values(&key("scale"))[0] as f64;
        let mut layer =
            LoraLinear::frozen(&key("w"))?.with_lora(Lora::new(&key("a"), &key("b"), scale)?)?;
        let (x, gy) = (key("x"), key("gy"));
        let y = layer.forward(&x)?;
        assert_eq!(y.dtype(), x.dtype(), "the output has the input's dtype");
        check(&format!("{tag} y"), &y, &key("y"), bound);
        // The input's gradient, the layer fixed.
        let dx = fallible_vjp(
            |a: &[Array]| Ok(vec![layer.clone().forward(&a[0])?]),
            std::slice::from_ref(&x),
            std::slice::from_ref(&gy),
        )?
        .1
        .remove(0);
        assert_eq!(dx.dtype(), x.dtype());
        check(&format!("{tag} dx"), &dx, &key("dx"), bound);
        // The parameters' gradients: A and B alone, the base weight frozen.
        let loss = |layer: &mut LoraLinear, (x, gy): (&Array, &Array)| -> Result<Array> {
            layer
                .forward(x)?
                .as_dtype(Dtype::Float32)?
                .multiply(gy.as_dtype(Dtype::Float32)?)?
                .sum(None)
        };
        let (_, grads) = mlx_rs::nn::value_and_grad(loss)(&mut layer, (&x, &gy))?;
        let mut names: Vec<_> = grads.keys().map(|k| k.to_string()).collect();
        names.sort();
        assert_eq!(
            names,
            ["adapters.0.a", "adapters.0.b"],
            "only A and B train"
        );
        for (name, want) in [("adapters.0.a", "da"), ("adapters.0.b", "db")] {
            assert_eq!(grads[name].dtype(), Dtype::Float32);
            check(&format!("{tag} {want}"), &grads[name], &key(want), 1e-6);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, ModuleParameters)]
struct Block {
    #[param]
    q_proj: LoraLinear,
    #[param]
    v_proj: LoraLinear,
    #[param]
    o_proj: LoraLinear,
}

impl Block {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        let q = self.q_proj.forward(x)?;
        // torch's silu on bf16 computes in f32 and rounds once; so does this.
        let s = mlx_rs::nn::silu(q.as_dtype(Dtype::Float32)?)?.as_dtype(q.dtype())?;
        let h = s.multiply(self.v_proj.forward(x)?)?;
        x.add(self.o_proj.forward(&h)?)
    }
}

/// The model's code knows nothing of adapters: each layer's forward pass, as without them.
#[derive(Debug, Clone, ModuleParameters)]
struct Tiny {
    #[param]
    layers: Vec<Block>,
    #[param]
    head: LoraLinear,
}

impl Tiny {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }
        self.head.forward(&x)?.as_dtype(Dtype::Float32)
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

    fn lora_layers(&mut self) -> impl Iterator<Item = &mut LoraLinear> {
        self.named_mut().into_iter().map(|(_, l)| l)
    }
}

/// The frozen model from the fixture's base weights.
fn frozen(f: &HashMap<String, Array>, prec: &str) -> Result<Tiny> {
    let w = |k: &str| LoraLinear::frozen(&f[&format!("{prec}.{k}.weight")]);
    Ok(Tiny {
        layers: (0..2)
            .map(|i| {
                Ok(Block {
                    q_proj: w(&format!("layers.{i}.q_proj"))?,
                    v_proj: w(&format!("layers.{i}.v_proj"))?,
                    o_proj: w(&format!("layers.{i}.o_proj"))?,
                })
            })
            .collect::<Result<_>>()?,
        head: w("head")?,
    })
}

/// A PEFT adapter directory's factors and scale, applied to the model's layers under `name`.
fn apply(model: &mut Tiny, dir: &str, name: &str, trainable: bool) -> Result<()> {
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/adapter_config.json")).unwrap(),
    )
    .unwrap();
    // PEFT's scale: lora_alpha / r, or / sqrt(r) with rsLoRA.
    let (alpha, r) = (
        config["lora_alpha"].as_f64().unwrap(),
        config["r"].as_f64().unwrap(),
    );
    let scale = match config["use_rslora"].as_bool() == Some(true) {
        true => alpha / r.sqrt(),
        false => alpha / r,
    };
    let factors = load(&format!("{dir}/adapter_model.safetensors"));
    for (path, layer) in model.named_mut() {
        let key = |f: &str| format!("base_model.model.{path}.lora_{f}.weight");
        if let (Some(a), Some(b)) = (factors.get(&key("A")), factors.get(&key("B"))) {
            let mut lora = Lora::new(a, b, scale)?;
            if !trainable {
                lora.freeze_parameters(true);
            }
            layer.insert_adapter(name, lora)?;
        }
    }
    Ok(())
}

fn both(f: &HashMap<String, Array>, prec: &str) -> Result<Tiny> {
    let mut model = frozen(f, prec)?;
    apply(
        &mut model,
        &format!("{FIXTURES}/peft_multi"),
        DEFAULT_ADAPTER,
        true,
    )?;
    apply(
        &mut model,
        &format!("{FIXTURES}/peft_multi/{REFERENCE}"),
        REFERENCE,
        false,
    )?;
    Ok(model)
}

fn kl_loss(logits: &Array, reference: &Array, targets: &Array) -> Result<Array> {
    let logp = mlx_rs::nn::log_softmax(logits, 1)?;
    let ce = logp
        .take_along_axis(targets.expand_dims(1)?, 1)?
        .mean(None)?
        .negative()?;
    let refp = mlx_rs::nn::log_softmax(reference, 1)?;
    let kl = logp
        .exp()?
        .multiply(logp.subtract(&refp)?)?
        .sum_axis(1, None)?
        .mean(None)?;
    ce.add(kl.multiply(Array::from_f32(BETA))?)
}

#[test]
fn every_layer_switches_and_trains_against_the_reference_as_peft_does() -> Result<()> {
    require_ieee_f32_matmul();
    let f = load(&format!("{FIXTURES}/multi.safetensors"));
    let targets = f["targets"].as_dtype(Dtype::Int32)?;
    // dash-lora's bounds off the CPU: (logits, loss, gradients). Its bf16 logits' 7e-4 is what CUDA
    // measured where two correct bf16 products round an element to neighbouring values and the
    // difference reaches a logit; here MLX's products do the same to a larger logit, so the bound
    // is the one dash-lora gives a single layer off the CPU, one bf16 unit (2^-7 of the largest).
    for (prec, [logits_bound, loss_bound, grad_bound]) in [
        ("f32", [5e-7, 1e-6, 1e-6]),
        ("bf16", [2f64.powi(-7), 1e-6, 1.5e-3]),
    ] {
        let mut model = both(&f, prec)?;
        let x = f[&format!("{prec}.x")].clone();
        for choice in [Some(DEFAULT_ADAPTER), Some(REFERENCE), None] {
            set_adapter(model.lora_layers(), choice);
            let logits = model.forward(&x)?;
            let key = format!("{prec}.logits.{}", choice.unwrap_or("base"));
            check(
                &format!("{prec} logits, {}", choice.unwrap_or("no adapter")),
                &logits,
                &f[&key],
                logits_bound,
            );
        }
        // The reference's logits, then the policy's loss and its gradients.
        set_adapter(model.lora_layers(), Some(REFERENCE));
        let reference = mlx_rs::stop_gradient(&model.forward(&x)?)?;
        set_adapter(model.lora_layers(), Some(DEFAULT_ADAPTER));
        let loss = |model: &mut Tiny,
                    (x, reference, targets): (&Array, &Array, &Array)|
         -> Result<Array> { kl_loss(&model.forward(x)?, reference, targets) };
        let (value, grads) =
            mlx_rs::nn::value_and_grad(loss)(&mut model, (&x, &reference, &targets))?;
        check(
            &format!("{prec} loss"),
            &value,
            &f[&format!("{prec}.loss")],
            loss_bound,
        );
        // The default adapter's factors alone get gradients: its 8 tensors, nothing frozen.
        let mut names: Vec<_> = grads.keys().map(|k| k.to_string()).collect();
        names.sort();
        assert_eq!(names.len(), 8, "{prec}: {names:?}");
        for name in names {
            assert!(
                name.contains("adapters.0."),
                "{prec}: {name} got a gradient"
            );
            let path = name.split(".adapters.").next().unwrap();
            let factor = if name.ends_with(".a") {
                "lora_A"
            } else {
                "lora_B"
            };
            let got = &grads[name.as_str()];
            let peft = &f[&format!("{prec}.grad.{path}.{factor}")];
            if prec == "f32" {
                check(
                    &format!("{prec} grad {path} {factor}"),
                    got,
                    peft,
                    grad_bound,
                );
                continue;
            }
            // bf16: dash-lora's measure, the distance to the same model's gradients computed
            // exactly (f64) no more than 1.1 times PEFT's own; the distance to PEFT printed.
            let exact = &f[&format!("bf16ref.grad.{path}.{factor}")];
            check(
                &format!("bf16 grad {path} {factor} vs PEFT"),
                got,
                peft,
                f64::INFINITY,
            );
            let theirs = check("  PEFT's own vs exact", peft, exact, f64::INFINITY);
            let mine = check("  dash-mlx vs exact", got, exact, 1.1 * theirs);
            println!(
                "  dash-mlx {:.2} times PEFT's distance to the exact gradient",
                mine / theirs
            );
        }
    }
    Ok(())
}

/// A layer choosing one of several adapters over its base gives, bit for bit, what a layer
/// holding that adapter alone gives.
#[test]
fn an_adapter_over_a_shared_base_is_that_adapter_alone() -> Result<()> {
    let f = load(&format!("{FIXTURES}/multi.safetensors"));
    for prec in ["f32", "bf16"] {
        let x = f[&format!("{prec}.x")].clone();
        for choice in [DEFAULT_ADAPTER, REFERENCE] {
            let mut shared = both(&f, prec)?;
            set_adapter(shared.lora_layers(), Some(choice));
            let mut alone = frozen(&f, prec)?;
            let dir = match choice {
                DEFAULT_ADAPTER => format!("{FIXTURES}/peft_multi"),
                other => format!("{FIXTURES}/peft_multi/{other}"),
            };
            apply(&mut alone, &dir, choice, true)?;
            set_adapter(alone.lora_layers(), Some(choice));
            let bits = |a: &Array| values(a).iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert!(
                bits(&shared.forward(&x)?) == bits(&alone.forward(&x)?),
                "{prec} {choice}"
            );
        }
    }
    Ok(())
}

/// The contract's errors: factors that do not fit the weight.
#[test]
fn refuses_factors_that_do_not_fit() -> Result<()> {
    let w = Array::from_slice(&[0.0f32; 12], &[3, 4]);
    let layer = LoraLinear::frozen(&w)?;
    let a = Array::from_slice(&[0.0f32; 10], &[2, 5]);
    let b = Array::from_slice(&[0.0f32; 6], &[3, 2]);
    let r: std::result::Result<_, Exception> = layer.with_lora(Lora::new(&a, &b, 1.0)?);
    assert!(r.is_err());
    Ok(())
}
