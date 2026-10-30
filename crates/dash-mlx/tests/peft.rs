//! PEFT's files into LoRA's MLX layers and out again, on dash-lora's `peft_*` fixtures (written by
//! PEFT 0.20), as dash-lora's tests hold its Burn layers to them: each adapter applied to layers of
//! its modules' shapes with PEFT's scale and dropout, read back out byte for byte; a frozen
//! adapter beside the trained default, MLX's trainable parameters holding the default's factors
//! alone; an adapter of rank 0 through and back.
//!
//!   cargo test -p dash-mlx --test peft
#![cfg(target_os = "macos")]

use dash_mlx::lora::{DEFAULT_ADAPTER, LoraLinear};
use dash_mlx::peft::{
    AdapterLayer, ApplyOptions, CONFIG_FILE, HostTensor, LoraWeights, PeftAdapter, PeftConfig,
    WEIGHTS_FILE,
};
use mlx_rs::Dtype;
use mlx_rs::module::ModuleParameters;
use std::path::PathBuf;

fn fixture(dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../dash-lora/tests/fixtures")
        .join(dir)
}

const ADAPTERS: [&str; 4] = [
    "peft_tiny",
    "peft_patterns",
    "peft_multi",
    "peft_multi/reference",
];

/// A frozen bf16 layer for each module the adapter holds, of its shape.
fn layers(adapter: &PeftAdapter) -> Vec<(String, LoraLinear)> {
    adapter
        .modules()
        .map(|(path, w)| {
            let (out, inp) = (w.b.shape()[0] as i32, w.a.shape()[1] as i32);
            let weight = mlx_rs::ops::zeros::<f32>(&[out, inp])
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            (path.to_string(), LoraLinear::frozen(&weight).unwrap())
        })
        .collect()
}

#[test]
fn through_the_layers_and_back_is_byte_identical() {
    for dir in ADAPTERS {
        let config = std::fs::read_to_string(fixture(dir).join(CONFIG_FILE)).unwrap();
        let weights = std::fs::read(fixture(dir).join(WEIGHTS_FILE)).unwrap();
        let adapter = PeftAdapter::read(fixture(dir)).unwrap();
        let mut named = layers(&adapter);
        adapter
            .apply(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
            .unwrap();
        for (path, layer) in &named {
            let lora = layer.adapter(DEFAULT_ADAPTER).unwrap();
            assert_eq!(lora.scale, adapter.scale(path).unwrap(), "{dir} {path}");
            assert_eq!(lora.dropout as f64, adapter.config.lora_dropout());
            assert_eq!(lora.a.as_ref().dtype(), Dtype::Float32);
        }
        let back =
            PeftAdapter::from_layers(adapter.config.clone(), named.iter().map(|(p, l)| (p, l)))
                .unwrap();
        let (config2, weights2) = back.to_bytes().unwrap();
        assert!(weights2 == weights, "{dir}: the factors changed on the way");
        assert_eq!(config2, config, "{dir}");
        println!(
            "{dir}: through {} MLX layers and back, byte for byte",
            named.len()
        );
    }
}

/// A trained default and, beside it, a frozen adapter read under another name (a KL penalty's
/// reference), over layers for every module either adapts: MLX's trainable parameters hold the
/// default's factors alone, and each adapter's factors are its file's.
#[test]
fn a_frozen_adapter_beside_the_trained_default() {
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
        let trainable = layer.trainable_parameters().flatten();
        let mut keys: Vec<&str> = trainable.keys().map(|k| &**k).collect();
        keys.sort();
        let expected: &[&str] = match adapter.module(path) {
            // The default is added first, so it is adapter 0.
            Some(_) => &["adapters.0.a", "adapters.0.b"],
            None => &[],
        };
        assert_eq!(keys, expected, "{path}");
        for (name, file) in [(DEFAULT_ADAPTER, &adapter), ("reference", &reference)] {
            let held = AdapterLayer::lora_weights(layer, name).map(|(w, _)| w);
            assert_eq!(held.as_ref(), file.module(path), "{path} {name}");
        }
    }
}

/// An adapter of rank 0, which the format does not refuse, goes in and comes back out. MLX holds
/// no data for an empty array, and `as_slice` panics on one.
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
