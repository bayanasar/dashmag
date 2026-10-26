//! PEFT's adapter format: files written by PEFT 0.20's `save_pretrained` read and written back
//! byte for byte, through the caller's layers too; per-module scales as PEFT computes them;
//! a new configuration as PEFT writes one; variants this crate does not implement refused.
//!
//!   cargo test -p dash-lora --test peft_format -- --nocapture

mod common;

use burn::tensor::{DType, Device, Tensor};
use common::*;
use dash_lora::peft::{CONFIG_FILE, WEIGHTS_FILE};
use dash_lora::{LoraLinear, PeftAdapter, PeftConfig, PeftError};
use serde_json::Value;

fn files(dir: &str) -> (String, Vec<u8>) {
    let dir = fixture(dir);
    (
        std::fs::read_to_string(dir.join(CONFIG_FILE)).unwrap(),
        std::fs::read(dir.join(WEIGHTS_FILE)).unwrap(),
    )
}

/// Frozen layers shaped like the fixture's adapted modules, under their PEFT paths.
fn layers(adapter: &PeftAdapter, device: &Device) -> Vec<(String, LoraLinear)> {
    adapter
        .modules()
        .map(|(path, w)| {
            let (out, inp) = (w.b.shape().as_slice()[0], w.a.shape().as_slice()[1]);
            let weight = Tensor::<2>::zeros([out, inp], (device, DType::BF16));
            (path.to_string(), LoraLinear::frozen(weight))
        })
        .collect()
}

#[test]
fn read_then_written_back_is_byte_identical() {
    for dir in ["peft_tiny", "peft_patterns"] {
        let (config, weights) = files(dir);
        let adapter = PeftAdapter::read(fixture(dir)).unwrap();
        assert_eq!(adapter.modules().count(), 4);
        let (config2, weights2) = adapter.to_bytes().unwrap();
        assert!(
            weights2 == weights,
            "{dir}: {WEIGHTS_FILE} differs from PEFT's"
        );
        assert_eq!(config2, config, "{dir}: {CONFIG_FILE} differs from PEFT's");

        // Through the file system as well.
        let out =
            std::env::temp_dir().join(format!("dash-lora-round-trip-{dir}-{}", std::process::id()));
        adapter.write(&out).unwrap();
        assert!(std::fs::read(out.join(WEIGHTS_FILE)).unwrap() == weights);
        assert_eq!(
            std::fs::read_to_string(out.join(CONFIG_FILE)).unwrap(),
            config
        );
        std::fs::remove_dir_all(&out).unwrap();
        println!(
            "{dir}: {} + {} bytes written back identical",
            weights.len(),
            config.len()
        );
    }
}

/// Into the caller's layers as trainable f32 parameters, and out again: still PEFT's bytes.
#[test]
fn through_the_layers_and_back_is_byte_identical() {
    let device = Device::flex().autodiff();
    let (config, weights) = files("peft_tiny");
    let adapter = PeftAdapter::read(fixture("peft_tiny")).unwrap();
    let mut layers = layers(&adapter, &device);
    adapter
        .apply(layers.iter_mut().map(|(p, l)| (p.as_str(), l)), &device)
        .unwrap();
    for (path, layer) in &layers {
        let lora = layer.lora.as_ref().unwrap();
        assert_eq!(lora.a.val().dtype(), DType::F32);
        assert!(lora.a.is_require_grad() && lora.b.is_require_grad());
        assert!(!layer.weight.is_require_grad());
        assert_eq!(lora.scale, adapter.scale(path).unwrap());
    }
    let back = PeftAdapter::from_layers(
        adapter.config.clone(),
        layers.iter().map(|(p, l)| (p.as_str(), l)),
    )
    .unwrap();
    let (config2, weights2) = back.to_bytes().unwrap();
    assert!(
        weights2 == weights,
        "{WEIGHTS_FILE} differs after the layers"
    );
    assert_eq!(config2, config);
}

#[test]
fn scales_and_ranks_are_peft_s() {
    let adapter = PeftAdapter::read(fixture("peft_patterns")).unwrap();
    let expected: serde_json::Map<String, Value> = serde_json::from_str(
        &std::fs::read_to_string(fixture("peft_patterns_scaling.json")).unwrap(),
    )
    .unwrap();
    assert!(adapter.config.use_rslora());
    for (path, scale) in &expected {
        let scale = scale.as_f64().unwrap();
        let ours = adapter.scale(path).unwrap();
        println!(
            "{path}: rank {}, scale {ours} (PEFT {scale})",
            adapter.module(path).unwrap().rank()
        );
        assert_eq!(ours.to_bits(), scale.to_bits(), "{path}");
    }
    // rank_pattern {"v_proj": 4}, alpha_pattern {"layers.1.q_proj": 12}, r 8, alpha 16.
    assert_eq!(adapter.module("layers.0.v_proj").unwrap().rank(), 4);
    assert_eq!(adapter.module("layers.1.q_proj").unwrap().rank(), 8);
    assert_eq!(adapter.config.alpha_for("layers.1.q_proj").unwrap(), 12.0);
    assert_eq!(adapter.config.alpha_for("layers.0.q_proj").unwrap(), 16.0);
}

#[test]
fn a_new_configuration_is_the_one_peft_writes() {
    let peft: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture("peft_default_config.json")).unwrap(),
    )
    .unwrap();
    let ours: Value =
        serde_json::from_str(&PeftConfig::new(16, 32.0, &["q_proj", "v_proj"]).to_json()).unwrap();
    // target_modules is a set in PEFT, written in whatever order the set iterates.
    let normalize = |mut v: Value| {
        let modules = v["target_modules"].as_array_mut().unwrap();
        modules.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
        v
    };
    assert_eq!(normalize(ours), normalize(peft));
}

#[test]
fn variants_and_inconsistent_files_are_refused() {
    let (config, weights) = files("peft_tiny");
    let with = |key: &str, value: Value| {
        let mut c: serde_json::Map<String, Value> = serde_json::from_str(&config).unwrap();
        c.insert(key.to_string(), value);
        serde_json::to_string(&c).unwrap()
    };
    for (key, value) in [
        ("use_dora", Value::Bool(true)),
        ("lora_bias", Value::Bool(true)),
        ("peft_type", Value::from("LOHA")),
        ("bias", Value::from("all")),
        ("alora_invocation_tokens", serde_json::json!([1, 2])),
        ("init_lora_weights", Value::from("mica")),
    ] {
        let r = PeftAdapter::from_bytes(&with(key, value.clone()), &weights);
        assert!(
            matches!(r, Err(PeftError::Unsupported(_))),
            "{key} = {value}: {r:?}"
        );
    }
    // A regular expression in alpha_pattern is refused rather than guessed at.
    let r = PeftAdapter::from_bytes(
        &with(
            "alpha_pattern",
            serde_json::json!({"layers\\.[01]\\.q_proj": 4}),
        ),
        &weights,
    )
    .unwrap()
    .scale("layers.0.q_proj");
    assert!(matches!(r, Err(PeftError::Unsupported(_))), "{r:?}");
    // A rank the configuration does not give.
    let r = PeftAdapter::from_bytes(&with("r", Value::from(4)), &weights);
    assert!(matches!(r, Err(PeftError::Invalid(_))), "{r:?}");
}

#[test]
fn apply_refuses_to_drop_weights() {
    let device = Device::flex().autodiff();
    let adapter = PeftAdapter::read(fixture("peft_tiny")).unwrap();
    let mut layers = layers(&adapter, &device);
    // One adapted module left unnamed.
    let r = adapter.apply(
        layers.iter_mut().skip(1).map(|(p, l)| (p.as_str(), l)),
        &device,
    );
    assert!(matches!(r, Err(PeftError::Mismatch(_))), "{r:?}");
    // A layer of the wrong shape.
    let mut wrong = LoraLinear::frozen(Tensor::<2>::zeros([3, 5], &device));
    let mut named: Vec<(&str, &mut LoraLinear)> = layers
        .iter_mut()
        .skip(1)
        .map(|(p, l)| (p.as_str(), l))
        .collect();
    named.push(("layers.0.q_proj", &mut wrong));
    let r = adapter.apply(named, &device);
    assert!(matches!(r, Err(PeftError::Mismatch(_))), "{r:?}");
    // A named layer the adapter does not cover is left without an adapter.
    let mut extra = LoraLinear::frozen(Tensor::<2>::zeros([3, 5], &device));
    let mut named: Vec<(&str, &mut LoraLinear)> =
        layers.iter_mut().map(|(p, l)| (p.as_str(), l)).collect();
    named.push(("layers.0.o_proj", &mut extra));
    adapter.apply(named, &device).unwrap();
    assert!(extra.lora.is_none());
}

/// The adapter a layer gets carries PEFT's scale and `lora_dropout`, which acts only while
/// training: an inference module's output does not depend on it.
#[test]
fn lora_takes_scale_and_dropout_from_the_configuration() {
    let device = Device::flex().autodiff();
    let mut adapter = PeftAdapter::read(fixture("peft_tiny")).unwrap();
    adapter.config.set("lora_dropout", 0.25);
    let lora = adapter.lora("layers.0.q_proj", &device).unwrap().unwrap();
    assert_eq!(lora.dropout.prob, 0.25);
    assert_eq!(lora.scale, 2.5);
    assert!(adapter.lora("layers.0.o_proj", &device).unwrap().is_none());
}

/// What `json.dumps(config, indent=2, sort_keys=True)` prints: a field set later still lands
/// in sorted order, and text outside ASCII is escaped (`"modèle"` prints as `"mod\u00e8le"`).
#[test]
fn configuration_text_is_python_s() {
    let mut config = PeftConfig::new(16, 32.0, &["q_proj"]);
    config.set("base_model_name_or_path", "modèle");
    config.set(
        "aaa_added_later",
        serde_json::json!({"z": 1, "a": [2.5, "é"]}),
    );
    let text = config.to_json();
    assert!(text.starts_with("{\n  \"aaa_added_later\": {\n    \"a\": [\n      2.5,\n      \"\\u00e9\"\n    ],\n    \"z\": 1\n  },\n"), "{text}");
    assert!(
        text.contains("\"base_model_name_or_path\": \"mod\\u00e8le\","),
        "{text}"
    );
    assert!(text.is_ascii());
    let back = PeftConfig::from_json(&text).unwrap();
    assert_eq!(back.get("base_model_name_or_path").unwrap(), "modèle");
}

/// Pattern keys match a whole trailing run of path components, as PEFT's
/// `re.match(rf"(.*\.)?({key})$", path)` does (values checked against `get_pattern_key`).
#[test]
fn pattern_keys_match_whole_path_components() {
    let mut config = PeftConfig::new(8, 16.0, &["q_proj", "v_proj"]);
    config.set(
        "alpha_pattern",
        serde_json::json!({"layers.0.v_proj": 6, "proj": 4}),
    );
    for (path, alpha) in [
        ("layers.0.q_proj", 16.0),
        ("layers.0.v_proj", 6.0),
        ("model.layers.0.v_proj", 6.0),
        ("xlayers.0.v_proj", 16.0),
        ("proj", 4.0),
        ("a.proj", 4.0),
    ] {
        assert_eq!(config.alpha_for(path).unwrap(), alpha, "{path}");
    }
}
