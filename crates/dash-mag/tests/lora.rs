//! LoRA's layer: the base product plus the low-rank term at `x`'s dtype, an adapter chosen or none,
//! factors refused that do not fit; and on every side PEFT's files through the layers and back
//! byte for byte. Against values recorded from PEFT itself: `tests/lora_peft.rs`.

mod common;

use common::{bits, close, tensor};
use dash_mag::lora::{
    AdapterLayer, ApplyOptions, DEFAULT_ADAPTER, HostTensor, Lora, LoraLinear, LoraWeights,
    PeftAdapter, PeftConfig, set_adapter,
};
use dash_mag::{DType, Tensor};
use std::path::PathBuf;

fn fixture(dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/lora")
        .join(dir)
}

fn layer(dtype: DType) -> LoraLinear {
    let mut layer = LoraLinear::frozen(&tensor(1, &[8, 16], dtype)).unwrap();
    let lora = Lora::new(
        &tensor(2, &[4, 16], DType::F32),
        &tensor(3, &[8, 4], DType::F32),
        2.0,
    )
    .unwrap();
    layer.insert_adapter(DEFAULT_ADAPTER, lora).unwrap();
    layer
}

#[test]
fn with_no_adapter_it_is_the_base_alone() {
    let mut layer = layer(DType::BF16);
    let x = tensor(4, &[3, 16], DType::BF16);
    let base = x.matmul(&layer.weight().t().unwrap()).unwrap();
    let with = layer.forward(&x).unwrap();
    assert_ne!(bits(&with), bits(&base));
    layer.set_adapter(None);
    assert_eq!(bits(&layer.forward(&x).unwrap()), bits(&base));
    layer.set_adapter(Some("another"));
    assert_eq!(bits(&layer.forward(&x).unwrap()), bits(&base));
    layer.set_adapter(Some(DEFAULT_ADAPTER));
    assert_eq!(bits(&layer.forward(&x).unwrap()), bits(&with));
}

#[test]
fn it_is_the_base_plus_the_low_rank_term_at_x_s_dtype() {
    let layer = layer(DType::F32);
    let x = tensor(4, &[3, 16], DType::F32);
    let lora = layer.lora().unwrap();
    let base = x.matmul(&layer.weight().t().unwrap()).unwrap();
    let term = x
        .matmul(&lora.a.t().unwrap())
        .unwrap()
        .matmul(&lora.b.t().unwrap())
        .unwrap()
        .mul_scalar(2.0)
        .unwrap();
    let want = base.add(&term).unwrap();
    close(&layer.forward(&x).unwrap(), &want.to_f32().unwrap(), 1e-6);
    // Inputs of any rank, as rows.
    let cube = tensor(5, &[2, 3, 16], DType::F32);
    let out = layer.forward(&cube).unwrap();
    assert_eq!(out.shape(), [2, 3, 8]);
    let rows = layer.forward(&cube.reshape(&[6, 16]).unwrap()).unwrap();
    assert_eq!(bits(&out), bits(&rows));
}

#[test]
fn factors_are_refused_when_they_do_not_fit() {
    let mut layer = layer(DType::F32);
    let a = tensor(6, &[4, 15], DType::F32);
    let b = tensor(7, &[8, 4], DType::F32);
    assert!(
        layer
            .insert_adapter("x", Lora::new(&a, &b, 1.0).unwrap())
            .is_err()
    );
    assert!(Lora::new(&a, &tensor(8, &[8, 3], DType::F32), 1.0).is_err());
    assert!(layer.set_factors(a, b).is_err());
}

#[test]
fn peft_s_files_through_the_layers_and_back_are_byte_identical() {
    for dir in [
        "peft_tiny",
        "peft_patterns",
        "peft_multi",
        "peft_multi/reference",
    ] {
        let config = std::fs::read_to_string(fixture(dir).join("adapter_config.json")).unwrap();
        let weights = std::fs::read(fixture(dir).join("adapter_model.safetensors")).unwrap();
        let adapter = PeftAdapter::read(fixture(dir)).unwrap();
        let mut named = layers(&adapter);
        adapter
            .apply(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
            .unwrap();
        for (path, layer) in &named {
            let lora = layer.adapter(DEFAULT_ADAPTER).unwrap();
            assert_eq!(lora.scale, adapter.scale(path).unwrap(), "{dir} {path}");
            assert_eq!(lora.a.dtype(), DType::F32);
        }
        let back =
            PeftAdapter::from_layers(adapter.config.clone(), named.iter().map(|(p, l)| (p, l)))
                .unwrap();
        let (config2, weights2) = back.to_bytes().unwrap();
        assert!(weights2 == weights, "{dir}: the factors changed on the way");
        assert_eq!(config2, config, "{dir}");
    }
}

fn layers(adapter: &PeftAdapter) -> Vec<(String, LoraLinear)> {
    adapter
        .modules()
        .map(|(path, w)| {
            let weight = Tensor::zeros(&[w.b.shape()[0], w.a.shape()[1]], DType::BF16).unwrap();
            (path.to_string(), LoraLinear::frozen(&weight).unwrap())
        })
        .collect()
}

#[test]
fn a_frozen_adapter_beside_the_default_each_keeps_its_file_s_factors() {
    let adapter = PeftAdapter::read(fixture("peft_multi")).unwrap();
    let reference = PeftAdapter::read(fixture("peft_multi/reference")).unwrap();
    let mut named = layers(&adapter);
    for (path, layer) in layers(&reference) {
        if !named.iter().any(|(p, _)| *p == path) {
            named.push((path, layer));
        }
    }
    adapter
        .apply(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
        .unwrap();
    let frozen = ApplyOptions {
        name: "reference".into(),
        trainable: false,
    };
    reference
        .apply_with(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &(), &frozen)
        .unwrap();
    for (path, layer) in &named {
        for (name, file) in [(DEFAULT_ADAPTER, &adapter), ("reference", &reference)] {
            let held = AdapterLayer::lora_weights(layer, name).map(|(w, _)| w);
            assert_eq!(held.as_ref(), file.module(path), "{path} {name}");
        }
    }
}

/// Silently leaving dropout out would train something else, so it is refused.
#[test]
fn dropout_is_refused_when_read() {
    let weights = LoraWeights {
        a: HostTensor::from_f32(vec![1, 4], &[1.0; 4]).unwrap(),
        b: HostTensor::from_f32(vec![3, 1], &[1.0; 3]).unwrap(),
    };
    assert!(<LoraLinear as AdapterLayer>::make_lora(&weights, 1.0, 0.1, true, &()).is_err());
    assert!(<LoraLinear as AdapterLayer>::make_lora(&weights, 1.0, 0.0, true, &()).is_ok());
    let _ = PeftConfig::new(1, 1.0, &["m"]);
}

/// A chosen base and rows of any rank.
#[test]
fn set_weight_replaces_the_base_and_refuses_another_shape() {
    let mut layer = layer(DType::F32);
    layer.set_adapter(None);
    let x = tensor(4, &[3, 16], DType::F32);
    let weight = tensor(6, &[8, 16], DType::F32);
    layer.set_weight(weight.clone()).unwrap();
    assert_eq!(
        bits(&layer.forward(&x).unwrap()),
        bits(&x.matmul(&weight.t().unwrap()).unwrap())
    );
    assert!(layer.set_weight(tensor(6, &[8, 15], DType::F32)).is_err());
    let integers = Tensor::from_i32(&[0; 128], &[8, 16]).unwrap();
    assert!(layer.set_weight(integers).is_err());
}

/// An adapter of rank 0, which PEFT's format does not refuse, goes into the layer and comes back out,
/// on every side.
#[test]
fn an_empty_adapter_goes_through_and_back() {
    let mut adapter = PeftAdapter::new(PeftConfig::new(0, 1.0, &["m"]));
    adapter.insert(
        "m",
        LoraWeights {
            a: HostTensor::from_f32(vec![0, 4], &[]).unwrap(),
            b: HostTensor::from_f32(vec![3, 0], &[]).unwrap(),
        },
    );
    let mut named = layers(&adapter);
    adapter
        .apply(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
        .unwrap();
    let back = PeftAdapter::from_layers(adapter.config.clone(), named.iter().map(|(p, l)| (p, l)))
        .unwrap();
    assert_eq!(back.module("m"), adapter.module("m"));
}

/// Choosing an adapter across layers, as PEFT does across a model: every layer takes it, a layer
/// without it runs its base alone, and a name no layer holds is refused with no layer changed.
#[test]
fn a_name_no_layer_holds_is_refused() {
    let mut held = layer(DType::F32);
    let mut bare = LoraLinear::frozen(&tensor(9, &[8, 16], DType::F32)).unwrap();
    let x = tensor(4, &[3, 16], DType::F32);
    let with = bits(&held.forward(&x).unwrap());
    assert!(set_adapter([&mut held, &mut bare], Some("defualt")).is_err());
    assert_eq!(
        bits(&held.forward(&x).unwrap()),
        with,
        "unchanged after the refusal"
    );
    set_adapter([&mut held, &mut bare], None).unwrap();
    assert!(held.lora().is_none() && bare.lora().is_none());
    set_adapter([&mut held, &mut bare], Some(DEFAULT_ADAPTER)).unwrap();
    assert_eq!(bits(&held.forward(&x).unwrap()), with);
    assert!(
        bare.lora().is_none(),
        "a layer without it runs its base alone"
    );
}

/// Each sequence of a bf16 batch gives, bit for bit, what it gives alone: on the Burn side, where
/// a batch goes through as its rows, out of `vjp` and inside it.
#[cfg(not(target_os = "macos"))]
#[test]
fn each_sequence_of_a_batch_is_what_it_gives_alone() {
    let mut layer = LoraLinear::frozen(&tensor(11, &[80, 96], DType::BF16)).unwrap();
    let lora = Lora::new(
        &tensor(12, &[16, 96], DType::F32),
        &tensor(13, &[80, 16], DType::F32),
        2.0,
    )
    .unwrap();
    layer.insert_adapter(DEFAULT_ADAPTER, lora).unwrap();
    let batch = tensor(14, &[2, 12, 96], DType::BF16);
    let y = layer.forward(&batch).unwrap();
    assert_eq!(y.shape(), [2, 12, 80]);
    let held = layer.clone();
    let (inside, _) = dash_mag::vjp(
        |p| Ok(vec![held.forward(&p[0])?]),
        std::slice::from_ref(&batch),
        &[None],
    )
    .unwrap();
    for i in 0..2 {
        let one = batch.slice(0, i..i + 1).unwrap().squeeze(0).unwrap();
        let alone = bits(&layer.forward(&one).unwrap());
        for (what, out) in [("out of vjp", &y), ("inside vjp", &inside[0])] {
            let got = out.slice(0, i..i + 1).unwrap().squeeze(0).unwrap();
            assert_eq!(bits(&got), alone, "sequence {i}, {what}");
        }
    }
}

/// Seeded host values (SplitMix64): uniform on `[-1, 1)`, or standard normal (Box-Muller).
struct Draws(u64);

impl Draws {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }

    fn normal(&mut self) -> f64 {
        let u = 1.0 - self.unit();
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    fn tensor(&mut self, shape: &[usize], mut draw: impl FnMut(&mut Self) -> f64) -> Tensor {
        let n = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|_| draw(self) as f32).collect();
        Tensor::from_f32(&values, shape).unwrap()
    }
}

fn host(t: &Tensor) -> Vec<f64> {
    t.as_dtype(DType::F32)
        .unwrap()
        .to_f32()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

/// The adapter reaches the output at f32 precision; folded into a bf16 weight, most of it rounds
/// away. `W` ~ U(-1/√in, 1/√in), and an update with a thousandth of `W`'s spread, as a fine-tuning
/// update is: the folded weight keeps few of the update's elements and points elsewhere, while the
/// layer's output is the base's plus the whole low-rank term, within one f32 rounding of the
/// largest output.
#[test]
fn the_update_is_kept_apart_not_rounded_into_the_weight() {
    let mut draws = Draws(3);
    let (out, inp, r) = (256, 256, 16);
    let w = draws.tensor(&[out, inp], |d| d.uniform() / (inp as f64).sqrt());
    let a = draws.tensor(&[r, inp], Draws::normal);
    let b = draws.tensor(&[out, r], Draws::normal);
    // W's elements spread 1/√(3 in); B A's spread √r; scale · B A spreads a thousandth of W.
    let scale = 1e-3 / ((3.0 * inp as f64).sqrt() * (r as f64).sqrt());
    let w_bf16 = w.as_dtype(DType::BF16).unwrap();
    let update = b.matmul(&a).unwrap().mul_scalar(scale).unwrap();
    let base = w_bf16.as_dtype(DType::F32).unwrap();

    // Folded: what survives in bf16(W + update) - W.
    let folded = base
        .add(&update)
        .unwrap()
        .as_dtype(DType::BF16)
        .unwrap()
        .as_dtype(DType::F32)
        .unwrap()
        .sub(&base)
        .unwrap();
    let (u, m) = (host(&update), host(&folded));
    let dot = u.iter().zip(&m).map(|(u, m)| u * m).sum::<f64>();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cosine = dot / (norm(&u) * norm(&m));
    let zeros = m.iter().filter(|v| **v == 0.0).count() as f64 / m.len() as f64;
    println!(
        "folded into bf16: cosine with the update {cosine:.3}, {:.0}% of it rounded to 0",
        zeros * 100.0
    );
    assert!(zeros > 0.5 && cosine < 0.8);

    // Kept apart: with f32 input the layer adds the whole low-rank term to the base product, in
    // f32, the sum rounded once: within one f32 unit of the largest output.
    let mut layer = LoraLinear::frozen(&w_bf16).unwrap();
    layer
        .insert_adapter(DEFAULT_ADAPTER, Lora::new(&a, &b, scale).unwrap())
        .unwrap();
    let x = draws.tensor(&[8, inp], Draws::normal);
    let y = host(&layer.forward(&x).unwrap());
    layer.set_adapter(None);
    let alone = host(&layer.forward(&x).unwrap());
    let term = host(&x.matmul(&update.t().unwrap()).unwrap());
    let largest = y.iter().fold(0f64, |m, v| m.max(v.abs()));
    let worst = y
        .iter()
        .zip(&alone)
        .zip(&term)
        .fold(0f64, |m, ((y, b), t)| m.max((y - b - t).abs()));
    println!(
        "kept apart: the low-rank term reaches the output within {worst:.2e}, {:.2} f32 units of \
         the largest output",
        worst / (largest * f32::EPSILON as f64)
    );
    assert!(worst <= largest * f32::EPSILON as f64);
}
