//! A tiny model with LoRA on its q_proj/v_proj-style linears, against the same model wrapped by
//! `peft.get_peft_model` (PEFT 0.20, torch 2.13, CPU, recorded in
//! `tests/fixtures/tiny.safetensors`): same base weights, adapter and input, then three steps of
//! Burn's AdamW against `torch.optim.AdamW` with the same hyperparameters.
//!
//! Two blocks of `x + o_proj(silu(q_proj(x)) * v_proj(x))`, then `head` and cross-entropy over
//! 32 tokens: hidden 64, inner 128, 64 classes, r 8, alpha 20 (scale 2.5); q_proj and v_proj
//! carry the adapter, o_proj and head are frozen. AdamW: lr 2e-3, betas (0.9, 0.999), eps 1e-8,
//! weight decay 0.01. The bf16 model (bf16 base and activations, f32 adapters: PEFT's default
//! for a bf16 model) reads its adapter from the files PEFT's `save_pretrained` wrote
//! (`tests/fixtures/peft_tiny`); the f32 model is f32 throughout.
//!
//!   cargo test -p dash-lora --test tiny -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test -p dash-lora --features cuda --test tiny -- --nocapture

mod common;

use std::collections::HashMap;

use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{DType, Device, Int, Tensor, TensorData, activation};
use common::*;
use dash_lora::{Lora, LoraLinear, PeftAdapter};

const LAYERS: usize = 2;
const SCALE: f64 = 20.0 / 8.0;
const LR: f64 = 2e-3;
const STEPS: usize = 3;

#[derive(Module, Debug)]
struct Block {
    q_proj: LoraLinear,
    v_proj: LoraLinear,
    o_proj: LoraLinear,
}

impl Block {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let q = self.q_proj.forward(x.clone());
        let dtype = q.dtype();
        // torch's silu on bf16 computes in f32 and rounds once; so does this.
        let s = activation::silu(q.cast(DType::F32)).cast(dtype);
        let h = s.mul(self.v_proj.forward(x.clone()));
        x.add(self.o_proj.forward(h))
    }
}

#[derive(Module, Debug)]
struct Tiny {
    layers: Vec<Block>,
    head: LoraLinear,
}

impl Tiny {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let x = self.layers.iter().fold(x, |x, layer| layer.forward(x));
        self.head.forward(x).cast(DType::F32)
    }

    /// The adapted layers under their PEFT module paths.
    fn adapted_mut(&mut self) -> Vec<(String, &mut LoraLinear)> {
        self.layers
            .iter_mut()
            .enumerate()
            .flat_map(|(i, b)| {
                [
                    (format!("layers.{i}.q_proj"), &mut b.q_proj),
                    (format!("layers.{i}.v_proj"), &mut b.v_proj),
                ]
            })
            .collect()
    }

    fn adapted(&self) -> Vec<(String, &LoraLinear)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, b)| {
                [
                    (format!("layers.{i}.q_proj"), &b.q_proj),
                    (format!("layers.{i}.v_proj"), &b.v_proj),
                ]
            })
            .collect()
    }

    fn lora(&self, path: &str) -> &Lora {
        let i: usize = path.split('.').nth(1).unwrap().parse().unwrap();
        let layer = match path.rsplit('.').next().unwrap() {
            "q_proj" => &self.layers[i].q_proj,
            "v_proj" => &self.layers[i].v_proj,
            other => panic!("{other}"),
        };
        layer.lora().unwrap()
    }
}

fn paths() -> Vec<String> {
    (0..LAYERS)
        .flat_map(|i| [format!("layers.{i}.q_proj"), format!("layers.{i}.v_proj")])
        .collect()
}

fn cross_entropy(logits: Tensor<2>, targets: Tensor<1, Int>) -> Tensor<1> {
    activation::log_softmax(logits, 1)
        .gather(1, targets.unsqueeze_dim(1))
        .mean()
        .neg()
}

/// The frozen model from the fixture's base weights, without adapters.
fn frozen(f: &HashMap<String, TensorData>, prec: &str, device: &Device) -> Tiny {
    let w = |k: &str| LoraLinear::frozen_from_data(get(f, &format!("{prec}.{k}.weight")), device);
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

fn model(f: &HashMap<String, TensorData>, prec: &str, device: &Device) -> Tiny {
    let mut model = frozen(f, prec, device);
    match prec {
        // The adapter as PEFT saved it, placed by module path.
        "bf16" => {
            let adapter = PeftAdapter::read(fixture("peft_tiny")).unwrap();
            adapter.apply(model.adapted_mut(), device).unwrap();
        }
        _ => {
            for (path, layer) in model.adapted_mut() {
                layer.set_lora(Lora::from_data(
                    get(f, &format!("{prec}.init.{path}.lora_A")),
                    get(f, &format!("{prec}.init.{path}.lora_B")),
                    SCALE,
                    device,
                ));
            }
        }
    }
    for path in paths() {
        assert_eq!(model.lora(&path).scale, SCALE);
    }
    model
}

fn adamw() -> burn::optim::ModuleOptimizer {
    AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(0.01)
        .init()
}

/// Error bounds, (max abs, rel), per quantity, just above what was measured on Flex and on an
/// RTX 5070 Ti; the comments give the measured maxima, abs / rel. "CUDA" covers every
/// configuration this crate builds in: Burn's default kernels, `burn/fusion`, `burn/autotune`
/// and both, with autotune sixteen runs each, half of them with the tests in parallel. Autotune
/// chooses kernels per process, and while one thread tunes a matrix product of a given shape,
/// another asking for the same shape runs an untuned candidate, so its results vary from run
/// to run. The f32 model agrees to f32 rounding everywhere. The bf16
/// model's activations are bf16: where two correct computations round an activation to
/// neighbouring bf16 values (a sum near a rounding tie, computed in a different order, or an f32
/// silu one ulp apart), the difference propagates. Flex rounds like PyTorch's CPU kernels
/// almost everywhere; CUDA's default kernels round more of these ties the other way (autotune's
/// choices happened to round fewer). Either way Burn's bf16 gradients are as far from the same
/// model computed in f64 as PyTorch's (`compare_with_exact`). Three free-running steps compound
/// those differences through AdamW, hence the looser bound after step 3.
struct Bounds {
    logits: (f64, f64),
    loss: f64,
    grad: (f64, f64),
    step1: (f64, f64),
    step3: (f64, f64),
    /// Gradients and loss at every step from torch's parameters (teacher-forced).
    forced_grad: (f64, f64),
    forced_loss: f64,
}

fn bounds(device: &str, prec: &str) -> Bounds {
    match (device, prec) {
        ("flex", "f32") => Bounds {
            logits: (1e-6, 5e-7),      // 4.8e-7 / 2.1e-7
            loss: 1e-6,                // 4.8e-7
            grad: (2e-8, 1e-6),        // 7.9e-9 / 4.3e-7
            step1: (3e-8, 2e-7),       // 1.5e-8 / 8.4e-8
            step3: (4e-7, 2e-6),       // 2.1e-7 / 1.2e-6
            forced_grad: (2e-8, 1e-6), // 1.0e-8 / 5.6e-7
            forced_loss: 2e-6,         // 9.5e-7
        },
        // CUDA in every configuration.
        (_, "f32") => Bounds {
            logits: (1e-6, 5e-7),      // 6.0e-7 / 2.6e-7 (autotune)
            loss: 1e-6,                // 9.5e-7 (2 ulps, autotune)
            grad: (2e-8, 1e-6),        // 1.1e-8 / 6.5e-7 (fusion, autotune)
            step1: (3e-8, 2e-7),       // 1.5e-8 / 1.2e-7 (fusion)
            step3: (4e-7, 3e-6),       // 3.1e-7 / 2.4e-6 (autotune)
            forced_grad: (2e-8, 1e-6), // 1.5e-8 / 8.2e-7 (autotune)
            forced_loss: 2e-6,         // 9.5e-7
        },
        ("flex", "bf16") => Bounds {
            logits: (1e-6, 1e-6),      // 0 / 0: bit-identical
            loss: 2e-6,                // 9.5e-7
            grad: (1e-7, 5e-6),        // 6.1e-8 / 3.0e-6
            step1: (3e-8, 2e-7),       // 1.5e-8 / 8.4e-8
            step3: (5e-4, 3e-3),       // 3.4e-4 / 1.9e-3
            forced_grad: (2e-5, 1e-3), // 9.8e-6 / 5.2e-4, at step 1
            forced_loss: 5e-5,         // 2.8e-5, at step 1
        },
        // CUDA in every configuration.
        (_, "bf16") => Bounds {
            logits: (1.5e-3, 7e-4),    // 9.8e-4 / 4.5e-4 (0 with autotune)
            loss: 2e-6,                // 9.5e-7
            grad: (5e-5, 3e-3),        // 3.0e-5 / 1.9e-3 (3.6e-7 relative with autotune)
            step1: (2e-5, 1e-4),       // 9.4e-6 / 5.3e-5
            step3: (1e-3, 6e-3),       // 5.9e-4 / 4.2e-3
            forced_grad: (5e-5, 3e-3), // 3.5e-5 / 2.2e-3 (autotune, at step 0)
            forced_loss: 5e-5,         // 2.8e-5, at step 1
        },
        other => panic!("no bounds for {other:?}"),
    }
}

fn train(device_name: &str, device: &Device, prec: &str) {
    let b = &bounds(device_name, prec);
    let f = load("tiny.safetensors");
    let mut model = model(&f, prec, device);
    let x: Tensor<2> = tensor(get(&f, &format!("{prec}.x")), device);
    let targets = int_tensor(get(&f, "targets"), device);
    let losses: Vec<f64> = get(&f, &format!("{prec}.losses")).iter::<f64>().collect();
    assert_eq!(losses.len(), STEPS);
    let mut optim = adamw();
    let p = format!("{device_name} {prec}");

    for (step, torch_loss) in losses.iter().copied().enumerate() {
        let logits = model.forward(x.clone());
        let loss = cross_entropy(logits.clone(), targets.clone());
        let grads = loss.backward();
        let loss_value = loss.into_scalar::<f32>() as f64;
        println!(
            "{p} step {step}: loss {loss_value:.7} (torch {:.7}, diff {:.2e})",
            torch_loss,
            loss_value - torch_loss
        );
        if step == 0 {
            check(
                &format!("{p} logits"),
                &logits.into_data(),
                &get(&f, &format!("{prec}.logits")),
                b.logits.0,
                b.logits.1,
            );
            check(
                &format!("{p} loss"),
                &TensorData::new(vec![loss_value as f32], [1]),
                &get(&f, &format!("{prec}.loss")),
                b.loss,
                1.0,
            );
            for path in paths() {
                let lora = model.lora(&path);
                for (factor, param) in [("lora_A", &lora.a), ("lora_B", &lora.b)] {
                    check(
                        &format!("{p} grad {path}.{factor}"),
                        &param.val().grad(&grads).unwrap().into_data(),
                        &get(&f, &format!("{prec}.grad0.{path}.{factor}")),
                        b.grad.0,
                        b.grad.1,
                    );
                }
            }
            if prec == "bf16" {
                compare_with_exact(&f, &model, &grads);
            }
        }
        let params = GradientsParams::from_grads(grads, &model);
        assert_eq!(
            params.len(),
            2 * paths().len(),
            "only the adapters have gradients"
        );
        model = optim.step(LR, model, params);
        let done = step + 1;
        for path in paths() {
            let lora = model.lora(&path);
            for (factor, param) in [("lora_A", &lora.a), ("lora_B", &lora.b)] {
                let what = format!("{p} after step {done} {path}.{factor}");
                let actual = param.val().into_data();
                let expected = get(&f, &format!("{prec}.step{done}.{path}.{factor}"));
                if done == 1 {
                    let grad = get(&f, &format!("{prec}.grad0.{path}.{factor}"));
                    check_first_step(&what, &actual, &expected, &grad, b.step1);
                } else if done == STEPS {
                    check(&what, &actual, &expected, b.step3.0, b.step3.1);
                }
            }
        }
    }
    // For a check outside this repository that PEFT loads what this crate writes:
    // DASH_LORA_EXPORT=<dir> writes the trained adapter to <dir>/<device>_<prec>.
    if let Ok(dir) = std::env::var("DASH_LORA_EXPORT") {
        let config = match prec {
            "bf16" => PeftAdapter::read(fixture("peft_tiny")).unwrap().config,
            _ => dash_lora::PeftConfig::new(8, 20.0, &["q_proj", "v_proj"]),
        };
        let adapter = PeftAdapter::from_layers(config, model.adapted()).unwrap();
        adapter
            .write(std::path::Path::new(&dir).join(format!("{device_name}_{prec}")))
            .unwrap();
    }
}

/// The bf16 model's step-0 gradients against the same bf16 values computed in f64 without
/// intermediate rounding: Burn's bf16 path is no further from exact arithmetic than PEFT's.
fn compare_with_exact(
    f: &HashMap<String, TensorData>,
    model: &Tiny,
    grads: &burn::tensor::Gradients,
) {
    for path in paths() {
        let lora = model.lora(&path);
        for (factor, param) in [("lora_A", &lora.a), ("lora_B", &lora.b)] {
            let exact = get(f, &format!("bf16ref.grad0.{path}.{factor}"));
            let torch = err(&get(f, &format!("bf16.grad0.{path}.{factor}")), &exact);
            let burn = err(&param.val().grad(grads).unwrap().into_data(), &exact);
            println!(
                "  vs f64 {path}.{factor}: burn rel {:.2e}, torch rel {:.2e}",
                burn.rel, torch.rel
            );
            // Measured: at most 1.04 times torch's distance (CUDA), equal on Flex.
            bound(burn.rel <= 1.1 * torch.rel, || {
                format!(
                    "{path}.{factor}: burn {:.2e} from exact, torch {:.2e}",
                    burn.rel, torch.rel
                )
            });
        }
    }
}

/// Every step's gradients from torch's own parameters at that step (torch's adapter before
/// the step): the forward and backward passes alone, step after step, with no drift carried
/// over from earlier steps.
fn teacher_forced(device_name: &str, device: &Device, prec: &str) {
    let b = bounds(device_name, prec);
    let f = load("tiny.safetensors");
    let mut model = model(&f, prec, device);
    let x: Tensor<2> = tensor(get(&f, &format!("{prec}.x")), device);
    let targets = int_tensor(get(&f, "targets"), device);
    let losses: Vec<f64> = get(&f, &format!("{prec}.losses")).iter::<f64>().collect();
    assert_eq!(losses.len(), STEPS);
    for (step, torch_loss) in losses.iter().copied().enumerate() {
        if step > 0 {
            for (path, layer) in model.adapted_mut() {
                layer.set_lora(Lora::from_data(
                    get(&f, &format!("{prec}.step{step}.{path}.lora_A")),
                    get(&f, &format!("{prec}.step{step}.{path}.lora_B")),
                    SCALE,
                    device,
                ));
            }
        }
        let loss = cross_entropy(model.forward(x.clone()), targets.clone());
        let grads = loss.backward();
        let loss_value = loss.into_scalar::<f32>() as f64;
        check(
            &format!("{device_name} {prec} from torch's step {step}: loss"),
            &TensorData::new(vec![loss_value], [1]),
            &TensorData::new(vec![torch_loss], [1]),
            b.forced_loss,
            1.0,
        );
        for path in paths() {
            let lora = model.lora(&path);
            for (factor, param) in [("lora_A", &lora.a), ("lora_B", &lora.b)] {
                check(
                    &format!("{device_name} {prec} from torch's step {step}: grad {path}.{factor}"),
                    &param.val().grad(&grads).unwrap().into_data(),
                    &get(&f, &format!("{prec}.grad{step}.{path}.{factor}")),
                    b.forced_grad.0,
                    b.forced_grad.1,
                );
            }
        }
    }
}

/// After AdamW's first step from the same start, parameters against torch's, in two parts.
/// The first step moves each element by `lr · g / (|g| + eps)`: where `|g|` is far above eps
/// that is `±lr` whatever the gradient's last bits, but where `|g|` is within a few eps of
/// zero the step follows the gradient's last bits, which differ between two correct
/// computations (a gradient of 2e-8 is a sum that cancels to 1e-6 of its terms). Elements with
/// a torch gradient below 100 eps are held only to the bound every AdamW step obeys, a
/// difference of at most `2 lr`.
fn check_first_step(
    what: &str,
    actual: &TensorData,
    expected: &TensorData,
    grad: &TensorData,
    limit: (f64, f64),
) {
    let a: Vec<f64> = actual.iter::<f64>().collect();
    let e: Vec<f64> = expected.iter::<f64>().collect();
    let g: Vec<f64> = grad.iter::<f64>().collect();
    let scale = e.iter().fold(0f64, |m, v| m.max(v.abs()));
    let (mut well, mut ill, mut n_ill) = (0f64, 0f64, 0usize);
    for ((a, e), g) in a.iter().zip(&e).zip(&g) {
        if g.abs() >= 100.0 * 1e-8 {
            well = well.max((a - e).abs());
        } else {
            ill = ill.max((a - e).abs());
            n_ill += 1;
        }
    }
    println!(
        "{what:<44} max abs {well:.3e}  rel {:.3e}; {n_ill} element(s) with |g| < 100 eps: {ill:.3e}",
        well / scale
    );
    bound(well <= limit.0 && well / scale <= limit.1, || {
        format!("{what}: max abs {well:.3e} (bound {:.1e})", limit.0)
    });
    bound(ill <= 2.0 * LR, || {
        format!("{what}: an element moved {ill:.3e}, more than 2 lr")
    });
}

/// Burn's AdamW fed torch's own gradients, step by step: the optimizer alone.
fn optimizer_alone(device_name: &str, device: &Device, prec: &str, bound: (f64, f64)) {
    let f = load("tiny.safetensors");
    let mut model = model(&f, prec, device);
    let mut optim = adamw();
    for step in 0..STEPS {
        let mut params = GradientsParams::new();
        for path in paths() {
            let lora = model.lora(&path);
            for (factor, param) in [("lora_A", &lora.a), ("lora_B", &lora.b)] {
                let g: Tensor<2> = tensor(
                    get(&f, &format!("{prec}.grad{step}.{path}.{factor}")),
                    device,
                );
                params.register(param.id, g);
            }
        }
        model = optim.step(LR, model, params);
        let done = step + 1;
        if done == 1 || done == STEPS {
            for path in paths() {
                let lora = model.lora(&path);
                for (factor, param) in [("lora_A", &lora.a), ("lora_B", &lora.b)] {
                    check(
                        &format!("{device_name} {prec} AdamW alone, step {done} {path}.{factor}"),
                        &param.val().into_data(),
                        &get(&f, &format!("{prec}.step{done}.{path}.{factor}")),
                        bound.0,
                        bound.1,
                    );
                }
            }
        }
    }
}

#[test]
fn f32_model_trains_as_peft_does() {
    for (name, device) in devices() {
        train(name, &device, "f32");
    }
}

#[test]
fn bf16_model_trains_as_peft_does() {
    for (name, device) in devices_with(DType::BF16) {
        train(name, &device, "bf16");
    }
}

/// Bounds just above the largest of three steps. Flex: 3.0e-8 / 2.3e-7. CUDA, in every
/// configuration: 5.96e-8 / 3.4e-7, with `burn/fusion`, which runs AdamW's elementwise update
/// as one fused kernel, in another order and with fused multiply-adds: 4 f32 ulps of the
/// parameters at step 3 (3.0e-8 / 2.3e-7 without fusion, with or without autotune).
#[test]
fn adamw_matches_torch_on_torch_gradients() {
    for (name, device) in devices() {
        let bound = match name {
            "flex" => (5e-8, 4e-7),
            _ => (7e-8, 4.5e-7),
        };
        optimizer_alone(name, &device, "f32", bound);
        if device.supports_dtype(DType::BF16) {
            optimizer_alone(name, &device, "bf16", bound);
        }
    }
}

#[test]
fn every_step_s_gradients_match_from_torch_s_parameters() {
    for (name, device) in devices() {
        teacher_forced(name, &device, "f32");
        if device.supports_dtype(DType::BF16) {
            teacher_forced(name, &device, "bf16");
        }
    }
}
