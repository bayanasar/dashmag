//! PEFT's files with no tensor library: the `peft_*` fixtures in dash-mag's `tests/fixtures/lora`
//! (written by PEFT 0.20's `save_pretrained`) read and written back byte for byte, and an adapter
//! applied to layers and read out of them again through [`AdapterLayer`], here a layer that keeps
//! the host data it is given; per-module scales as PEFT computes them; a new configuration as PEFT
//! writes one; variants this crate does not implement refused.
//!
//!   cargo test -p dash-peft

use dash_peft::{
    AdapterLayer, ApplyOptions, CONFIG_FILE, DEFAULT_ADAPTER, LoraWeights, PeftAdapter, PeftConfig,
    PeftError, WEIGHTS_FILE,
};
use serde_json::Value;
use std::path::PathBuf;

fn fixture(dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../dash-mag/tests/fixtures/lora")
        .join(dir)
}

fn files(dir: &str) -> (String, Vec<u8>) {
    let dir = fixture(dir);
    (
        std::fs::read_to_string(dir.join(CONFIG_FILE)).unwrap(),
        std::fs::read(dir.join(WEIGHTS_FILE)).unwrap(),
    )
}

const ADAPTERS: [&str; 4] = [
    "peft_tiny",
    "peft_patterns",
    "peft_multi",
    "peft_multi/reference",
];

#[test]
fn read_then_written_back_is_byte_identical() {
    for dir in ADAPTERS {
        let (config, weights) = files(dir);
        let adapter = PeftAdapter::from_bytes(&config, &weights).unwrap();
        let (config2, weights2) = adapter.to_bytes().unwrap();
        assert!(
            weights2 == weights,
            "{dir}: {WEIGHTS_FILE} differs from PEFT's"
        );
        assert_eq!(config2, config, "{dir}: {CONFIG_FILE} differs from PEFT's");
    }
}

/// A layer that keeps what it is given: each adapter's factors as read, its scale and whether
/// it trains.
struct HostLayer {
    out: usize,
    inp: usize,
    adapters: Vec<(String, HostLora)>,
}

#[derive(Clone, Debug, PartialEq)]
struct HostLora {
    weights: LoraWeights,
    scale: f64,
    dropout: f64,
    trainable: bool,
}

impl HostLayer {
    fn new(out: usize, inp: usize) -> Self {
        Self {
            out,
            inp,
            adapters: Vec::new(),
        }
    }

    fn get(&self, name: &str) -> Option<&HostLora> {
        self.adapters
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, l)| l)
    }
}

impl AdapterLayer for HostLayer {
    type Lora = HostLora;
    type Device = ();

    fn in_features(&self) -> usize {
        self.inp
    }

    fn out_features(&self) -> usize {
        self.out
    }

    fn make_lora(
        weights: &LoraWeights,
        scale: f64,
        dropout: f64,
        trainable: bool,
        _: &(),
    ) -> Result<HostLora, PeftError> {
        Ok(HostLora {
            weights: weights.clone(),
            scale,
            dropout,
            trainable,
        })
    }

    fn put_lora(&mut self, name: &str, lora: HostLora) -> Result<(), PeftError> {
        self.drop_lora(name);
        self.adapters.push((name.to_string(), lora));
        Ok(())
    }

    fn drop_lora(&mut self, name: &str) {
        self.adapters.retain(|(n, _)| n != name);
    }

    fn lora_weights(&self, name: &str) -> Option<(LoraWeights, f64)> {
        self.get(name).map(|l| (l.weights.clone(), l.scale))
    }
}

/// A layer of the right shape for each module the adapter holds.
fn layers(adapter: &PeftAdapter) -> Vec<(String, HostLayer)> {
    adapter
        .modules()
        .map(|(path, w)| {
            (
                path.to_string(),
                HostLayer::new(w.b.shape()[0], w.a.shape()[1]),
            )
        })
        .collect()
}

#[test]
fn through_layers_and_back_is_byte_identical() {
    for dir in ADAPTERS {
        let (config, weights) = files(dir);
        let adapter = PeftAdapter::from_bytes(&config, &weights).unwrap();
        let mut layers = layers(&adapter);
        adapter
            .apply(layers.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
            .unwrap();
        for (path, layer) in &layers {
            let lora = layer.get(DEFAULT_ADAPTER).unwrap();
            assert_eq!(lora.scale, adapter.scale(path).unwrap(), "{dir} {path}");
            assert_eq!(lora.dropout, adapter.config.lora_dropout());
            assert!(lora.trainable);
        }
        let back =
            PeftAdapter::from_layers(adapter.config.clone(), layers.iter().map(|(p, l)| (p, l)))
                .unwrap();
        let (config2, weights2) = back.to_bytes().unwrap();
        assert!(weights2 == weights, "{dir}: the factors changed on the way");
        assert_eq!(config2, config);
    }
}

#[test]
fn a_named_adapter_frozen_beside_the_default() {
    let adapter = PeftAdapter::read(fixture("peft_tiny")).unwrap();
    let mut named = layers(&adapter);
    adapter
        .apply(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
        .unwrap();
    let frozen = ApplyOptions {
        name: "reference".into(),
        trainable: false,
    };
    adapter
        .apply_with(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &(), &frozen)
        .unwrap();
    for (_, layer) in &named {
        assert!(layer.get(DEFAULT_ADAPTER).unwrap().trainable);
        assert!(!layer.get("reference").unwrap().trainable);
    }
}

#[test]
fn layers_that_do_not_fit_are_refused_before_any_changes() {
    let adapter = PeftAdapter::read(fixture("peft_tiny")).unwrap();
    // One layer the wrong shape: nothing applied anywhere.
    let mut named = layers(&adapter);
    named[1].1.out += 1;
    let err = adapter
        .apply(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
        .unwrap_err();
    assert!(matches!(err, PeftError::Mismatch(_)), "{err}");
    assert!(named.iter().all(|(_, l)| l.adapters.is_empty()));
    // One module no layer takes.
    let mut named = layers(&adapter);
    named.pop();
    let err = adapter
        .apply(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
        .unwrap_err();
    assert!(err.to_string().contains("no layer takes"), "{err}");
    assert!(named.iter().all(|(_, l)| l.adapters.is_empty()));
}

#[test]
fn a_layer_the_adapter_does_not_cover_loses_that_name() {
    let adapter = PeftAdapter::read(fixture("peft_tiny")).unwrap();
    let mut layers = layers(&adapter);
    let (_, first) = adapter.modules().next().unwrap();
    // A layer no module of the adapter names, holding an adapter of the default name: applying
    // the adapter as that name removes it there.
    let mut extra = HostLayer::new(first.b.shape()[0], first.a.shape()[1]);
    let lora = HostLora {
        weights: first.clone(),
        scale: 1.0,
        dropout: 0.0,
        trainable: true,
    };
    extra.put_lora(DEFAULT_ADAPTER, lora).unwrap();
    adapter
        .apply(
            layers
                .iter_mut()
                .map(|(p, l)| (p.as_str(), l))
                .chain([("not.adapted", &mut extra)]),
            &(),
        )
        .unwrap();
    assert!(extra.get(DEFAULT_ADAPTER).is_none());
}

#[test]
fn a_layer_whose_scale_the_config_does_not_give_is_refused() {
    let adapter = PeftAdapter::read(fixture("peft_tiny")).unwrap();
    let mut layers = layers(&adapter);
    adapter
        .apply(layers.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
        .unwrap();
    layers[0].1.adapters[0].1.scale *= 2.0;
    let err = PeftAdapter::from_layers(adapter.config.clone(), layers.iter().map(|(p, l)| (p, l)))
        .unwrap_err();
    assert!(matches!(err, PeftError::Mismatch(_)), "{err}");
}

#[test]
fn a_host_tensor_holds_exactly_its_shape() {
    use dash_peft::{FactorDtype, HostTensor};
    assert!(HostTensor::new(FactorDtype::Bf16, vec![2, 3], vec![0; 12]).is_ok());
    for (shape, bytes) in [(vec![2, 3], 11), (vec![2, 3], 13), (vec![usize::MAX, 2], 0)] {
        let err = HostTensor::new(FactorDtype::Bf16, shape.clone(), vec![0; bytes]).unwrap_err();
        assert!(matches!(err, PeftError::Invalid(_)), "{shape:?}: {err}");
    }
    assert!(HostTensor::from_f32(vec![2, 2], &[1.0, 2.0, 3.0]).is_err());
    let t = HostTensor::from_f32(vec![1, 2], &[1.0, -2.0]).unwrap();
    assert_eq!(
        (t.dtype(), t.shape(), t.bytes().len()),
        (FactorDtype::F32, &[1, 2][..], 8)
    );
}

#[test]
fn written_through_the_file_system_is_byte_identical() {
    for dir in ADAPTERS {
        let (config, weights) = files(dir);
        let adapter = PeftAdapter::read(fixture(dir)).unwrap();
        let out = std::env::temp_dir().join(format!(
            "dash-peft-round-trip-{}-{}",
            dir.replace('/', "-"),
            std::process::id()
        ));
        adapter.write(&out).unwrap();
        assert!(
            std::fs::read(out.join(WEIGHTS_FILE)).unwrap() == weights,
            "{dir}"
        );
        assert_eq!(
            std::fs::read_to_string(out.join(CONFIG_FILE)).unwrap(),
            config,
            "{dir}"
        );
        std::fs::remove_dir_all(&out).unwrap();
    }
}

/// Two adapters PEFT saved from one model ("default" at the top of the directory, "reference" in
/// its subdirectory), read into one set of layers under their names, one trainable and one
/// frozen, and written back the way PEFT lays them out: each adapter's two files as PEFT wrote
/// them, byte for byte.
#[test]
fn two_adapters_through_one_set_of_layers_are_written_back_as_peft_wrote_them() {
    let dirs = [
        (DEFAULT_ADAPTER, "peft_multi"),
        ("reference", "peft_multi/reference"),
    ];
    let adapters: Vec<PeftAdapter> = dirs
        .iter()
        .map(|(_, dir)| PeftAdapter::read(fixture(dir)).unwrap())
        .collect();
    // A layer per module either adapter adapts.
    let mut shapes = std::collections::BTreeMap::new();
    for adapter in &adapters {
        for (path, w) in adapter.modules() {
            let shape = (w.b.shape()[0], w.a.shape()[1]);
            assert_eq!(*shapes.entry(path.to_string()).or_insert(shape), shape);
        }
    }
    let mut layers: Vec<(String, HostLayer)> = shapes
        .into_iter()
        .map(|(path, (out, inp))| (path, HostLayer::new(out, inp)))
        .collect();
    adapters[0]
        .apply(layers.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
        .unwrap();
    let frozen = ApplyOptions {
        name: "reference".into(),
        trainable: false,
    };
    adapters[1]
        .apply_with(
            layers.iter_mut().map(|(p, l)| (p.as_str(), l)),
            &(),
            &frozen,
        )
        .unwrap();
    // One base per layer; q_proj holds both adapters, v_proj and o_proj one each.
    assert_eq!(layers.len(), 6);
    for (path, layer) in &layers {
        let names: Vec<&str> = layer.adapters.iter().map(|(n, _)| n.as_str()).collect();
        let expect: &[&str] = match path.rsplit('.').next().unwrap() {
            "q_proj" => &[DEFAULT_ADAPTER, "reference"],
            "v_proj" => &[DEFAULT_ADAPTER],
            _ => &["reference"],
        };
        assert_eq!(names, expect, "{path}");
    }

    let out = std::env::temp_dir().join(format!("dash-peft-two-adapters-{}", std::process::id()));
    for ((name, dir), read) in dirs.iter().zip(&adapters) {
        let named = layers.iter().map(|(p, l)| (p.as_str(), l));
        let back = PeftAdapter::from_layers_named(read.config.clone(), name, named).unwrap();
        // PEFT's layout: the default adapter at the top, every other in a subdirectory.
        let sub = match *name {
            DEFAULT_ADAPTER => out.clone(),
            other => out.join(other),
        };
        back.write(&sub).unwrap();
        let (config, weights) = files(dir);
        assert!(
            std::fs::read(sub.join(WEIGHTS_FILE)).unwrap() == weights,
            "{name}: {WEIGHTS_FILE} differs from PEFT's"
        );
        assert_eq!(
            std::fs::read_to_string(sub.join(CONFIG_FILE)).unwrap(),
            config,
            "{name}: {CONFIG_FILE} differs from PEFT's"
        );
    }
    std::fs::remove_dir_all(&out).unwrap();

    // Applying an adapter again under a name replaces it whole: a layer it does not cover loses
    // its adapter of that name.
    let trainable = ApplyOptions {
        name: "reference".into(),
        trainable: true,
    };
    adapters[0]
        .apply_with(
            layers.iter_mut().map(|(p, l)| (p.as_str(), l)),
            &(),
            &trainable,
        )
        .unwrap();
    for (path, layer) in &layers {
        assert_eq!(
            layer.get("reference").is_some(),
            !path.ends_with("o_proj"),
            "{path}"
        );
    }
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
        let ours = adapter.scale(path).unwrap();
        assert_eq!(ours.to_bits(), scale.as_f64().unwrap().to_bits(), "{path}");
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

/// The adapter a layer gets carries PEFT's scale and `lora_dropout`.
#[test]
fn a_layer_s_adapter_takes_scale_and_dropout_from_the_configuration() {
    let mut adapter = PeftAdapter::read(fixture("peft_tiny")).unwrap();
    adapter.config.set("lora_dropout", 0.25);
    let lora = adapter
        .lora::<HostLayer>("layers.0.q_proj", &())
        .unwrap()
        .unwrap();
    assert_eq!((lora.scale, lora.dropout), (2.5, 0.25));
    assert!(
        adapter
            .lora::<HostLayer>("layers.0.o_proj", &())
            .unwrap()
            .is_none()
    );
}

/// What `json.dumps(config, indent=2, sort_keys=True)` prints: a field set later still lands in
/// sorted order, and text outside ASCII is escaped (`"modèle"` prints as `"mod\u00e8le"`).
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
