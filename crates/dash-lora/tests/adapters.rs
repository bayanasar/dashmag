//! Several adapters over one base weight, one chosen per forward pass.
//!
//! - One layer: the layer holding several adapters, with one of them chosen or none, gives what
//!   a layer holding that adapter alone (or none) gives, output and gradients, bit for bit; and
//!   gradients reach the chosen adapter only if it trains.
//! - The tiny model of `tests/tiny.rs` (two blocks of `x + o_proj(silu(q_proj(x)) * v_proj(x))`,
//!   then `head`) against PEFT 0.20 on torch 2.13 (CPU, `tests/fixtures/multi.safetensors`): two
//!   adapters read from PEFT's directories (`tests/fixtures/peft_multi`, "default" at the top,
//!   "reference" in its subdirectory) into one model over one base, "default" (r 8, alpha 20,
//!   on q_proj and v_proj) trainable and "reference" (r 4, alpha 6, rsLoRA, on q_proj and o_proj)
//!   frozen. The logits with each chosen and with none, against PEFT's `set_adapter` and
//!   `disable_adapter()`, every layer switched by one call; one training step of cross-entropy
//!   plus a KL penalty against the reference, the reference's logits from the same model: the
//!   loss and "default"'s gradients against PEFT's, none for "reference", Burn's AdamW keeping
//!   state for "default"'s factors alone, which then match torch's AdamW. For an f32 and a bf16
//!   base (bf16 activations, f32 adapters, PEFT's default for a bf16 model).
//!
//!   cargo test -p dash-lora --test adapters -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test -p dash-lora --features cuda --test adapters -- --nocapture

mod common;

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{DType, Device, Gradients, Int, Tensor, TensorData, activation};
use common::*;
use dash_lora::{AdapterChoice, ApplyOptions, DEFAULT_ADAPTER, Lora, LoraLinear, PeftAdapter};
use dash_ops::checkpoint;

/// The tests take turns: with Burn's `autotune`, a thread asking for a shape another thread is
/// tuning runs an untuned candidate meanwhile, and two passes compared bit for bit would then
/// need not run the same kernels.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The output and the gradients of the input and of the chosen adapter's factors, for the
/// upstream gradient `gy`.
struct Pass {
    y: TensorData,
    dx: TensorData,
    factors: Option<(TensorData, TensorData)>,
}

fn pass(layer: &LoraLinear, x: &Tensor<2>, gy: &Tensor<2>) -> Pass {
    let x = x.clone().detach().require_grad();
    let y = layer.forward(x.clone());
    let grads = y.clone().mul(gy.clone()).sum().backward();
    let factors = layer.lora().and_then(|lora| {
        Some((
            lora.a.val().grad(&grads)?.into_data(),
            lora.b.val().grad(&grads)?.into_data(),
        ))
    });
    Pass {
        y: y.into_data(),
        dx: x.grad(&grads).unwrap().into_data(),
        factors,
    }
}

/// A [r, in] and B [out, r] drawn on the host.
fn factors(
    rng: &mut Rng,
    out: usize,
    inp: usize,
    r: usize,
    device: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let a = rng.tensor([r, inp], DType::F32, device, |g| {
        g.uniform(-1.0, 1.0) / (inp as f64).sqrt()
    });
    let b = rng.tensor([out, r], DType::F32, device, |g| {
        g.uniform(-1.0, 1.0) / (r as f64).sqrt()
    });
    (a, b)
}

/// One layer holding two adapters, each chosen in turn and then none, against a layer holding
/// that adapter alone, or none: the same operations in the same order, so the same bits.
fn one_layer_matches_each_adapter_alone(name: &str, device: &Device, dtype: DType) {
    let (out, inp, t) = (80, 96, 48);
    let mut rng = Rng::new(21);
    let w = rng.tensor([out, inp], dtype, device, |g| {
        g.uniform(-1.0, 1.0) / (inp as f64).sqrt()
    });
    let (a1, b1) = factors(&mut rng, out, inp, 16, device);
    let (a2, b2) = factors(&mut rng, out, inp, 4, device);
    let x = rng.tensor([t, inp], dtype, device, Rng::normal);
    let gy = rng.tensor([t, out], dtype, device, Rng::normal);
    let adapters = [("policy", a1, b1, 2.0), ("reference", a2, b2, 3.0)];
    let mut shared = LoraLinear::frozen(w.clone());
    for (n, a, b, scale) in &adapters {
        shared.insert_adapter(n, Lora::new(a.clone(), b.clone(), *scale));
    }
    for choice in [Some("policy"), Some("reference"), None] {
        // A layer of its own over the same base weight, with factors of its own, equal in value.
        let alone = LoraLinear::frozen(w.clone());
        let alone = match adapters.iter().find(|(n, ..)| Some(*n) == choice) {
            Some((_, a, b, scale)) => alone.with_lora(Lora::new(a.clone(), b.clone(), *scale)),
            None => alone,
        };
        let chosen = shared.clone().set_adapter(choice);
        assert_eq!(chosen.active_adapter(), choice);
        assert_eq!(chosen.lora().is_some(), choice.is_some());
        // Kernels compiled and tuned before the passes compared.
        pass(&alone, &x, &gy);
        let (want, got) = (pass(&alone, &x, &gy), pass(&chosen, &x, &gy));
        let what = format!("{name} {dtype:?} {}", choice.unwrap_or("no adapter"));
        check(&format!("{what} y"), &got.y, &want.y, 0.0, 0.0);
        check(&format!("{what} dx"), &got.dx, &want.dx, 0.0, 0.0);
        match (got.factors, want.factors) {
            (Some((ga, gb)), Some((wa, wb))) => {
                check(&format!("{what} dA"), &ga, &wa, 0.0, 0.0);
                check(&format!("{what} dB"), &gb, &wb, 0.0, 0.0);
            }
            (None, None) => assert!(choice.is_none()),
            _ => panic!("{what}: one layer's adapter got gradients and the other's not"),
        }
    }
}

#[test]
fn an_adapter_over_a_shared_base_is_that_adapter_alone() {
    let _serial = serial();
    for (name, device) in devices() {
        one_layer_matches_each_adapter_alone(name, &device, DType::F32);
        if device.supports_dtype(DType::BF16) {
            one_layer_matches_each_adapter_alone(name, &device, DType::BF16);
        }
    }
}

/// Gradients reach the chosen adapter if it trains, and nothing else: not the base, not the
/// adapters not chosen, not a frozen one chosen.
#[test]
fn gradients_reach_only_the_chosen_trainable_adapter() {
    let _serial = serial();
    for (name, device) in devices() {
        let (out, inp) = (24, 40);
        let mut rng = Rng::new(5);
        let w = rng.tensor([out, inp], DType::F32, &device, Rng::normal);
        let (a1, b1) = factors(&mut rng, out, inp, 8, &device);
        let (a2, b2) = factors(&mut rng, out, inp, 4, &device);
        let (a3, b3) = factors(&mut rng, out, inp, 2, &device);
        let layer = LoraLinear::frozen(w)
            .with_lora(Lora::new(a1, b1, 2.0))
            .with_adapter("frozen", Lora::new(a2, b2, 1.0).no_grad())
            .with_adapter("other", Lora::new(a3, b3, 0.5));
        let x = rng.tensor([16, inp], DType::F32, &device, Rng::normal);
        for (choice, trains) in [
            (Some(DEFAULT_ADAPTER), true),
            (Some("other"), true),
            (Some("frozen"), false),
            (None, false),
        ] {
            let layer = layer.clone().set_adapter(choice);
            let x = x.clone().require_grad();
            let grads = layer.forward(x.clone()).powi_scalar(2).sum().backward();
            assert!(x.grad(&grads).is_some(), "the input's gradient flows on");
            assert!(layer.weight.val().grad(&grads).is_none());
            for (n, lora) in layer.adapters() {
                let has = lora.a.val().grad(&grads).is_some();
                assert_eq!(has, lora.b.val().grad(&grads).is_some());
                assert_eq!(
                    has,
                    trains && Some(n) == choice,
                    "{name}: adapter {n} with {choice:?} chosen"
                );
            }
            let params = GradientsParams::from_grads(grads, &layer);
            assert_eq!(
                params.len(),
                if trains { 2 } else { 0 },
                "{name} {choice:?}"
            );
        }
    }
}

// --- the tiny model, against PEFT ---------------------------------------------------------

const LAYERS: usize = 2;
const BETA: f64 = 0.5;
const LR: f64 = 2e-3;
const REFERENCE: &str = "reference";

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

/// The model's code knows nothing of adapters: each layer's forward pass, as without them.
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

    /// Every linear layer under its PEFT module path.
    fn named(&self) -> Vec<(String, &LoraLinear)> {
        let mut out = Vec::new();
        for (i, b) in self.layers.iter().enumerate() {
            out.push((format!("layers.{i}.q_proj"), &b.q_proj));
            out.push((format!("layers.{i}.v_proj"), &b.v_proj));
            out.push((format!("layers.{i}.o_proj"), &b.o_proj));
        }
        out.push(("head".to_string(), &self.head));
        out
    }

    fn named_mut(&mut self) -> Vec<(String, &mut LoraLinear)> {
        let mut out = Vec::new();
        for (i, b) in self.layers.iter_mut().enumerate() {
            out.push((format!("layers.{i}.q_proj"), &mut b.q_proj));
            out.push((format!("layers.{i}.v_proj"), &mut b.v_proj));
            out.push((format!("layers.{i}.o_proj"), &mut b.o_proj));
        }
        out.push(("head".to_string(), &mut self.head));
        out
    }
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

fn adapter(name: &str) -> PeftAdapter {
    match name {
        DEFAULT_ADAPTER => PeftAdapter::read(fixture("peft_multi")).unwrap(),
        other => PeftAdapter::read(fixture("peft_multi").join(other)).unwrap(),
    }
}

/// The model with both adapters read from PEFT's directories, as PEFT reads them:
/// `from_pretrained(..., is_trainable=True)` and `load_adapter(..., "reference")`.
fn both(f: &HashMap<String, TensorData>, prec: &str, device: &Device) -> Tiny {
    let mut model = frozen(f, prec, device);
    adapter(DEFAULT_ADAPTER)
        .apply(model.named_mut(), device)
        .unwrap();
    let frozen = ApplyOptions {
        name: REFERENCE.into(),
        trainable: false,
    };
    adapter(REFERENCE)
        .apply_with(model.named_mut(), device, &frozen)
        .unwrap();
    model
}

/// The model holding one adapter alone, as its single adapter; or none.
fn alone(f: &HashMap<String, TensorData>, prec: &str, device: &Device, name: Option<&str>) -> Tiny {
    let mut model = frozen(f, prec, device);
    if let Some(name) = name {
        adapter(name).apply(model.named_mut(), device).unwrap();
    }
    model
}

fn kl_loss(logits: Tensor<2>, reference: Tensor<2>, targets: Tensor<1, Int>) -> Tensor<1> {
    let logp = activation::log_softmax(logits, 1);
    let ce = logp
        .clone()
        .gather(1, targets.unsqueeze_dim(1))
        .mean()
        .neg();
    let refp = activation::log_softmax(reference, 1);
    let kl = logp.clone().exp().mul(logp.sub(refp)).sum_dim(1).mean();
    ce.add(kl.mul_scalar(BETA))
}

fn adamw() -> burn::optim::ModuleOptimizer {
    AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(0.01)
        .init()
}

fn choice_key(choice: Option<&str>) -> &str {
    choice.unwrap_or("base")
}

/// Error bounds, (max abs, rel), per quantity, just above the largest measured on Flex and on
/// the device of each CubeCL build the tests run in (an RTX 5070 Ti through CUDA with Burn's
/// default kernels, `burn/fusion`, `autotune`, and both, autotune twice each, half of the runs
/// with the tests in parallel; the same card through wgpu on Vulkan, which has no bf16; CubeCL's
/// CPU runtime, which gives Flex's bf16 values); the comments give those maxima, abs / rel. They are those of the same model with one adapter (`tests/tiny.rs`):
/// the f32 model agrees to f32 rounding; the bf16 model's activations are bf16, and where two
/// correct computations round one to neighbouring bf16 values the difference propagates (on CUDA,
/// one logit by one bf16 unit). The bf16 gradients are no further from the same model computed in
/// f64 than PEFT's (`compare_with_exact`: at most 1.04 times).
struct Bounds {
    logits: (f64, f64),
    loss: f64,
    grad: (f64, f64),
    step1: (f64, f64),
}

fn bounds(device: &str, prec: &str) -> Bounds {
    match (device, prec) {
        // Flex and every CubeCL build.
        (_, "f32") => Bounds {
            logits: (1e-6, 5e-7), // 7.2e-7 / 3.5e-7 (CUDA with autotune, the logits with no adapter)
            loss: 1e-6,           // 4.8e-7, one f32 unit
            grad: (2e-8, 1e-6),   // 1.2e-8 / 6.1e-7 (CUDA with autotune)
            step1: (3e-8, 2e-7),  // 1.5e-8 / 1.2e-7
        },
        ("flex", "bf16") => Bounds {
            logits: (0.0, 0.0),   // bit-identical
            loss: 1e-6,           // 9.5e-7, two f32 units
            grad: (7e-8, 3.5e-6), // 5.9e-8 / 2.9e-6
            step1: (3e-8, 2e-7),  // 1.5e-8 / 1.2e-7
        },
        // CUDA in every configuration (CubeCL's CPU runtime measures as Flex).
        (_, "bf16") => Bounds {
            logits: (1.5e-3, 7e-4), // 9.8e-4 / 4.5e-4: one logit one bf16 unit apart
            loss: 1e-6,             // 9.5e-7
            grad: (3e-5, 1.5e-3),   // 2.3e-5 / 1.1e-3
            step1: (3e-7, 2.5e-6),  // 2.6e-7 / 2.1e-6
        },
        other => panic!("no bounds for {other:?}"),
    }
}

/// Every layer of the model reports `choice`.
fn assert_chosen(model: &Tiny, choice: Option<&str>) {
    for (path, layer) in model.named() {
        assert_eq!(layer.active_adapter(), choice, "{path}");
    }
}

fn switches(device_name: &str, device: &Device, prec: &str) {
    let b = bounds(device_name, prec);
    let f = load("multi.safetensors");
    let x: Tensor<2> = tensor(get(&f, &format!("{prec}.x")), device);
    let model = both(&f, prec, device);
    assert_eq!(model.adapter_names(), [DEFAULT_ADAPTER, REFERENCE]);
    // Who holds what: q_proj both, v_proj "default", o_proj "reference", the head neither.
    for (path, layer) in model.named() {
        let names: Vec<&str> = layer.adapters().map(|(n, _)| n).collect();
        let expect: &[&str] = match path.rsplit('.').next().unwrap() {
            "q_proj" => &[DEFAULT_ADAPTER, REFERENCE],
            "v_proj" => &[DEFAULT_ADAPTER],
            "o_proj" => &[REFERENCE],
            _ => &[],
        };
        assert_eq!(names, expect, "{path}");
    }
    // Read as the files say: the scale of each, the trainability asked for.
    let q = &model.layers[0].q_proj;
    assert_eq!(q.adapter(DEFAULT_ADAPTER).unwrap().scale, 2.5);
    assert_eq!(q.adapter(REFERENCE).unwrap().scale, 3.0);
    assert!(q.adapter(DEFAULT_ADAPTER).unwrap().a.is_require_grad());
    assert!(!q.adapter(REFERENCE).unwrap().a.is_require_grad());
    // "default" until told otherwise, as PEFT's first adapter.
    assert_chosen(&model, Some(DEFAULT_ADAPTER));

    let p = format!("{device_name} {prec}");
    for choice in [Some(DEFAULT_ADAPTER), Some(REFERENCE), None] {
        let chosen = model.clone().set_adapter(choice);
        assert_chosen(&chosen, choice);
        let single = alone(&f, prec, device, choice);
        // Kernels compiled and tuned before the passes compared, and run: with Burn's fusion, a
        // pass left queued would fuse with the next one's operations.
        single.forward(x.clone()).into_data();
        let logits = chosen.forward(x.clone());
        check(
            &format!("{p} logits with {}", choice_key(choice)),
            &logits.clone().into_data(),
            &get(&f, &format!("{prec}.logits.{}", choice_key(choice))),
            b.logits.0,
            b.logits.1,
        );
        // Bit for bit what the model holding that adapter alone gives.
        let single = single.forward(x.clone());
        check(
            &format!("{p} logits with {} vs it alone", choice_key(choice)),
            &logits.into_data(),
            &single.into_data(),
            0.0,
            0.0,
        );
    }
    // The model it was chosen on keeps its own choice.
    assert_chosen(&model, Some(DEFAULT_ADAPTER));
}

#[test]
fn every_layer_switches_at_once_as_peft_s_do_f32() {
    let _serial = serial();
    for (name, device) in devices() {
        switches(name, &device, "f32");
    }
}

#[test]
fn every_layer_switches_at_once_as_peft_s_do_bf16() {
    let _serial = serial();
    for (name, device) in devices_with(DType::BF16) {
        switches(name, &device, "bf16");
    }
}

/// The parameter ids of every adapter named `name` in the model.
fn ids(model: &Tiny, name: &str) -> Vec<burn::module::ParamId> {
    model
        .named()
        .into_iter()
        .filter_map(|(_, l)| l.adapter(name))
        .flat_map(|lora| [lora.a.id, lora.b.id])
        .collect()
}

fn trains(device_name: &str, device: &Device, prec: &str) {
    let b = bounds(device_name, prec);
    let f = load("multi.safetensors");
    let x: Tensor<2> = tensor(get(&f, &format!("{prec}.x")), device);
    let targets = int_tensor(get(&f, "targets"), device);
    let p = format!("{device_name} {prec}");
    let mut model = both(&f, prec, device);
    // Kernels compiled and tuned before the passes compared, and run.
    {
        let warm = alone(&f, prec, device, Some(DEFAULT_ADAPTER));
        let logits = warm.forward(x.clone());
        let grads = kl_loss(logits.clone(), logits.detach(), targets.clone()).backward();
        let lora = warm.layers[0].q_proj.lora().unwrap();
        lora.a.val().grad(&grads).unwrap().into_data();
        device.sync().unwrap();
    }
    let reference_before: Vec<TensorData> = model
        .named()
        .into_iter()
        .filter_map(|(_, l)| l.adapter(REFERENCE))
        .flat_map(|lora| [lora.a.val().into_data(), lora.b.val().into_data()])
        .collect();

    // One step: the reference from a clone choosing it, then the policy, which is "default".
    let reference = model
        .clone()
        .set_adapter(Some(REFERENCE))
        .forward(x.clone());
    // Nothing the reference runs requires a gradient: nothing of it is recorded.
    assert!(!reference.is_tracked(), "the frozen reference was recorded");
    // Run before the policy's pass, which the model holding "default" alone repeats below.
    let reference: Tensor<2> = tensor(reference.into_data(), device);
    let logits = model.forward(x.clone());
    let loss = kl_loss(logits, reference.clone(), targets.clone());
    let grads = loss.backward();
    let loss_value = loss.into_scalar::<f32>() as f64;
    check(
        &format!("{p} loss"),
        &TensorData::new(vec![loss_value], [1]),
        &get(&f, &format!("{prec}.loss")),
        b.loss,
        1.0,
    );
    for (path, layer) in model.named() {
        assert!(layer.weight.val().grad(&grads).is_none(), "{path}");
        if let Some(lora) = layer.adapter(REFERENCE) {
            assert!(lora.a.val().grad(&grads).is_none(), "{path}");
            assert!(lora.b.val().grad(&grads).is_none(), "{path}");
        }
    }
    let trained_grads = default_grads(&model, &grads);
    for (what, grad) in &trained_grads {
        check(
            &format!("{p} grad {what}"),
            grad,
            &get(&f, &format!("{prec}.grad.{what}")),
            b.grad.0,
            b.grad.1,
        );
        if prec == "bf16" {
            compare_with_exact(&f, what, grad);
        }
    }
    let params = GradientsParams::from_grads(grads, &model);
    let trained = ids(&model, DEFAULT_ADAPTER);
    assert_eq!(
        params.len(),
        trained.len(),
        "only \"default\" has gradients"
    );
    assert!(trained.iter().all(|id| params.get::<2>(*id).is_some()));
    assert!(
        ids(&model, REFERENCE)
            .iter()
            .all(|id| params.get::<2>(*id).is_none())
    );
    let mut optim = adamw();
    model = optim.step(LR, model, params);
    // Run before the next pass is queued: with Burn's fusion, queued work fuses with what follows.
    let stepped = default_factors(&model);

    // The same step for a model holding "default" alone, the reference's logits given: the same
    // gradients and the same factors after the step, bit for bit.
    let mut single = alone(&f, prec, device, Some(DEFAULT_ADAPTER));
    let single_grads = kl_loss(single.forward(x.clone()), reference, targets).backward();
    for ((what, grad), (_, alone)) in trained_grads
        .iter()
        .zip(default_grads(&single, &single_grads))
    {
        check(
            &format!("{p} grad {what} vs it alone"),
            grad,
            &alone,
            0.0,
            0.0,
        );
    }
    let single_params = GradientsParams::from_grads(single_grads, &single);
    let mut single_optim = adamw();
    single = single_optim.step(LR, single, single_params);
    let single_stepped = default_factors(&single);

    // AdamW keeps two moments per parameter it stepped: "default"'s factors, as for the model
    // that holds nothing else; none for the frozen reference.
    let states = optim.to_record().len();
    println!(
        "{p}: AdamW holds {states} state tensors for {} trained factors",
        trained.len()
    );
    assert_eq!(states, 2 * trained.len());
    assert_eq!(states, single_optim.to_record().len());

    let reference_after: Vec<TensorData> = model
        .named()
        .into_iter()
        .filter_map(|(_, l)| l.adapter(REFERENCE))
        .flat_map(|lora| [lora.a.val().into_data(), lora.b.val().into_data()])
        .collect();
    for (before, after) in reference_before.iter().zip(&reference_after) {
        assert_eq!(before.as_bytes(), after.as_bytes(), "the reference moved");
    }
    for ((what, actual), (_, alone)) in stepped.iter().zip(&single_stepped) {
        check(
            &format!("{p} after the step {what} vs it alone"),
            actual,
            alone,
            0.0,
            0.0,
        );
        check_first_step(
            &format!("{p} after the step {what}"),
            actual,
            &get(&f, &format!("{prec}.step1.{what}")),
            &get(&f, &format!("{prec}.grad.{what}")),
            b.step1,
        );
    }
}

/// "default"'s factors in the model's order, under `<path>.lora_A` and `<path>.lora_B`.
fn default_factors(model: &Tiny) -> Vec<(String, TensorData)> {
    model
        .named()
        .into_iter()
        .filter_map(|(path, l)| Some((path, l.adapter(DEFAULT_ADAPTER)?)))
        .flat_map(|(path, lora)| {
            [
                (format!("{path}.lora_A"), lora.a.val().into_data()),
                (format!("{path}.lora_B"), lora.b.val().into_data()),
            ]
        })
        .collect()
}

/// The gradients of "default"'s factors, as [`default_factors`] names them.
fn default_grads(model: &Tiny, grads: &Gradients) -> Vec<(String, TensorData)> {
    model
        .named()
        .into_iter()
        .filter_map(|(path, l)| Some((path, l.adapter(DEFAULT_ADAPTER)?)))
        .flat_map(|(path, lora)| {
            let grad = |p: &Param<Tensor<2>>| p.val().grad(grads).unwrap().into_data();
            [
                (format!("{path}.lora_A"), grad(&lora.a)),
                (format!("{path}.lora_B"), grad(&lora.b)),
            ]
        })
        .collect()
}

/// The bf16 model's gradients against the same bf16 values computed in f64 without
/// intermediate rounding: Burn's bf16 path no further from exact arithmetic than PEFT's.
fn compare_with_exact(f: &HashMap<String, TensorData>, key: &str, burn: &TensorData) {
    let exact = get(f, &format!("bf16ref.grad.{key}"));
    let torch = err(&get(f, &format!("bf16.grad.{key}")), &exact);
    let burn = err(burn, &exact);
    println!(
        "  vs f64 {key}: burn rel {:.2e}, torch rel {:.2e}",
        burn.rel, torch.rel
    );
    // Measured: at most 1.04 times torch's distance (CUDA), equal on Flex.
    bound(burn.rel <= 1.1 * torch.rel, || {
        format!(
            "{key}: burn {:.2e} from exact, torch {:.2e}",
            burn.rel, torch.rel
        )
    });
}

/// After AdamW's first step from the same start, parameters against torch's, in two parts (as
/// in `tests/tiny.rs`): where `|g|` is far above eps the step is `±lr` whatever the gradient's
/// last bits; elements with a torch gradient below 100 eps follow those last bits, and are held
/// only to the bound every AdamW step obeys, a difference of at most `2 lr`.
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

#[test]
fn trains_against_a_frozen_reference_as_peft_does_f32() {
    let _serial = serial();
    for (name, device) in devices() {
        trains(name, &device, "f32");
    }
}

#[test]
fn trains_against_a_frozen_reference_as_peft_does_bf16() {
    let _serial = serial();
    for (name, device) in devices_with(DType::BF16) {
        trains(name, &device, "bf16");
    }
}

// --- the choice as part of the module ------------------------------------------------------

/// The parameter paths of a module, as Burn's records and parameter groups see them.
fn paths<M: Module>(module: &M) -> Vec<String> {
    #[derive(Default)]
    struct Paths {
        stack: Vec<String>,
        out: Vec<String>,
    }
    impl ModuleVisitor for Paths {
        fn enter_module(&mut self, name: &str, _: &str) {
            self.stack.push(name.to_string());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.stack.pop();
        }
        fn visit_float<const D: usize>(&mut self, _: &Param<Tensor<D>>) {
            self.out.push(self.stack.join("."));
        }
    }
    let mut paths = Paths::default();
    module.visit(&mut paths);
    paths.out
}

/// The choice survives what Burn does to a module, and nothing else changes it; the factors sit
/// under their adapter's name in the parameter paths.
#[test]
fn the_choice_goes_where_the_module_goes() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    let f = load("multi.safetensors");
    let model = both(&f, "f32", &device).set_adapter(Some(REFERENCE));
    assert_chosen(&model, Some(REFERENCE));
    assert_chosen(&model.valid(), Some(REFERENCE));
    assert_chosen(&model.clone().train(), Some(REFERENCE));
    assert_chosen(&model.clone().fork(&device), Some(REFERENCE));
    assert_chosen(&model.clone().to_device(&Device::flex()), Some(REFERENCE));
    assert_chosen(&model.clone().no_grad(), Some(REFERENCE));
    // An optimizer's step maps the module: the choice stays.
    let x: Tensor<2> = tensor(get(&f, "f32.x"), &device);
    let model = model.set_adapter(Some(DEFAULT_ADAPTER));
    let grads = GradientsParams::from_grads(model.forward(x).sum().backward(), &model);
    let model = adamw().step(LR, model, grads);
    assert_chosen(&model, Some(DEFAULT_ADAPTER));
    // A clone chooses for itself.
    let base = model.clone().set_adapter(None);
    assert_chosen(&base, None);
    assert_chosen(&model, Some(DEFAULT_ADAPTER));

    let paths = paths(&model);
    for path in [
        "layers.0.q_proj.weight",
        "layers.0.q_proj.adapters.default.a",
        "layers.0.q_proj.adapters.reference.b",
        "layers.1.o_proj.adapters.reference.a",
        "layers.1.v_proj.adapters.default.b",
    ] {
        assert!(paths.iter().any(|p| p == path), "{path} not in {paths:?}");
    }
    // A weight per linear layer, two factors per adapter: four of each adapter.
    assert_eq!(paths.len(), LAYERS * 3 + 1 + 2 * (4 + 4));
    // The display names them too.
    let shown = format!("{}", model.layers[0].q_proj);
    assert!(
        shown.contains("default") && shown.contains("reference"),
        "{shown}"
    );
}

/// Each adapter's factors are recorded under its name, whatever else a layer holds: a record of
/// the model holding "default" alone loads into the model holding both, and leaves "reference" as
/// it was.
#[test]
fn records_name_each_adapter_s_factors() {
    let device = Device::flex();
    let f = load("multi.safetensors");
    let mut single = alone(&f, "f32", &device, Some(DEFAULT_ADAPTER));
    for (_, layer) in single.named_mut() {
        let doubled = layer.lora().map(|lora| {
            let twice = |p: &Param<Tensor<2>>| p.val().mul_scalar(2.0);
            Lora::new(twice(&lora.a), twice(&lora.b), lora.scale)
        });
        if let Some(doubled) = doubled {
            layer.set_lora(doubled);
        }
    }
    let model = both(&f, "f32", &device);
    let record = single.clone().into_record().allow_partial(true);
    let loaded = model.clone().load_record(record);
    let bytes = |p: &Param<Tensor<2>>| p.val().into_data().as_bytes().to_vec();
    let mut n = 0;
    for (((path, got), (_, was)), (_, doubled)) in loaded
        .named()
        .into_iter()
        .zip(model.named())
        .zip(single.named())
    {
        if let Some(lora) = got.adapter(DEFAULT_ADAPTER) {
            let want = doubled.adapter(DEFAULT_ADAPTER).unwrap();
            assert!(
                bytes(&lora.a) == bytes(&want.a) && bytes(&lora.b) == bytes(&want.b),
                "{path}"
            );
            n += 1;
        }
        if let Some(lora) = got.adapter(REFERENCE) {
            let want = was.adapter(REFERENCE).unwrap();
            assert!(
                bytes(&lora.a) == bytes(&want.a) && bytes(&lora.b) == bytes(&want.b),
                "{path}"
            );
            assert!(
                !lora.a.is_require_grad(),
                "{path}: the reference stays frozen"
            );
        }
        assert_eq!(got.active_adapter(), Some(DEFAULT_ADAPTER));
    }
    assert_eq!(n, 4);
}

/// A misspelt adapter would run the base alone, silently: refused.
#[test]
#[should_panic(expected = "no layer of this module holds an adapter of that name")]
fn choosing_an_adapter_no_layer_holds_panics() {
    let device = Device::flex();
    let f = load("multi.safetensors");
    let _ = both(&f, "f32", &device).set_adapter(Some("refrence"));
}

/// `dash_ops::checkpoint` runs each layer twice, the second time in the backward pass on the
/// module itself (the first on `valid()`): both runs use the adapter chosen, and the gradients
/// are plain autodiff's.
#[test]
fn a_recomputed_stack_runs_the_adapter_chosen() {
    let _serial = serial();
    for (name, device) in devices() {
        let f = load("multi.safetensors");
        let x: Tensor<2> = tensor(get(&f, "f32.x"), &device);
        // Both adapters trainable, so that a wrong choice would show in the gradients.
        let mut model = frozen(&f, "f32", &device);
        adapter(DEFAULT_ADAPTER)
            .apply(model.named_mut(), &device)
            .unwrap();
        let other = ApplyOptions {
            name: REFERENCE.into(),
            trainable: true,
        };
        adapter(REFERENCE)
            .apply_with(model.named_mut(), &device, &other)
            .unwrap();
        let blocks = model.layers.set_adapter(Some(REFERENCE));
        let loss = |out: Tensor<2>| out.powi_scalar(2).mean();

        let x0 = x.clone().require_grad();
        let mut grads = loss(blocks.iter().fold(x0.clone(), |h, b| b.forward(h))).backward();
        let plain_dx = x0.grad(&grads).unwrap().into_data();
        let plain = GradientsParams::from_module(&mut grads, &blocks);

        let block = |b: &Block, h: Tensor<2>| b.forward(h);
        let (out, saved) = checkpoint::forward(&blocks, x.clone(), block);
        let mut grads = loss(out.clone()).backward();
        let grad = out.grad_remove(&mut grads).unwrap();
        let dx = checkpoint::backward(&blocks, saved, grad, block, &mut grads);
        let recomputed = GradientsParams::from_grads(grads, &blocks);

        // Bit-identical, but on CUDA with autotune and fusion together, where the per-block
        // passes may settle on other fused kernels than the whole stack's (`tests/recompute.rs`):
        // up to 1.7e-7 of the largest value over four runs.
        let (abs, rel) = match name {
            "flex" => (0.0, 0.0),
            _ if cfg!(feature = "autotune") => (f64::INFINITY, 3e-7),
            _ => (0.0, 0.0),
        };
        check(
            &format!("{name} recomputed dx"),
            &dx.into_data(),
            &plain_dx,
            abs,
            rel,
        );
        let mut n = 0;
        for b in &blocks {
            for layer in [&b.q_proj, &b.v_proj, &b.o_proj] {
                for (adapter, lora) in layer.adapters() {
                    for param in [&lora.a, &lora.b] {
                        let (r, p) = (recomputed.get::<2>(param.id), plain.get::<2>(param.id));
                        assert_eq!(r.is_some(), adapter == REFERENCE, "{name} {adapter}");
                        assert_eq!(p.is_some(), adapter == REFERENCE, "{name} {adapter}");
                        if let (Some(r), Some(p)) = (r, p) {
                            check(
                                &format!("{name} recomputed {adapter} grad"),
                                &r.into_data(),
                                &p.into_data(),
                                abs,
                                rel,
                            );
                            n += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(n, 2 * 2 * LAYERS, "q_proj's and o_proj's reference factors");
    }
}
