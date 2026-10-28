//! PEFT's files with no tensor library: dash-lora's `peft_*` fixtures (written by PEFT 0.20) read
//! and written back byte for byte, and an adapter applied to layers and read out of them again
//! through [`AdapterLayer`], here a layer that keeps the host data it is given.
//!
//!   cargo test -p dash-peft

use dash_peft::{
    AdapterLayer, ApplyOptions, CONFIG_FILE, DEFAULT_ADAPTER, LoraWeights, PeftAdapter, PeftError,
    WEIGHTS_FILE,
};
use std::path::PathBuf;

fn fixture(dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../dash-lora/tests/fixtures")
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
