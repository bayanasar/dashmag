//! Every epoch's weights are kept, and the ones at the top of the run are the best epoch's.
//!
//! Two runs compared only at the epochs each selected mix the selection into the result; a
//! comparison at a fixed epoch needs every epoch's weights. This trains two epochs on a
//! small synthetic cache and reads each one back.

use std::fs;
use std::path::{Path, PathBuf};

use burn::tensor::Device;
use dash_cli::model_json;
use dash_train::data::npy_header_bytes;
use dash_train::rng::Rng;
use dash_train::train::{self, Init, Recipe, Run, epoch_dir};
use serde_json::{Value, json};

/// `targets.len()` frames of `size x size` noise, and their targets.
fn split(cache: &Path, name: &str, size: usize, targets: &[[i64; 2]], rng: &mut Rng) {
    let mut npy = npy_header_bytes([targets.len(), size, size, 3]);
    npy.extend((0..targets.len() * size * size * 3).map(|_| rng.next_u64() as u8));
    fs::write(cache.join(format!("{name}.npy")), npy).unwrap();
    let rows: String = targets
        .iter()
        .map(|t| json!({ "targets": t }).to_string() + "\n")
        .collect();
    fs::write(cache.join(format!("{name}.jsonl")), rows).unwrap();
}

#[test]
fn every_epoch_is_kept_and_the_best_is_at_the_top() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("every-epoch");
    let _ = fs::remove_dir_all(&dir);
    let cache = dir.join("cache");
    fs::create_dir_all(&cache).unwrap();
    let mut rng = Rng::new(7);
    split(
        &cache,
        "train",
        32,
        &[[0, 0], [1, 1], [2, 0], [0, 1]],
        &mut rng,
    );
    split(&cache, "val", 32, &[[1, 0], [2, 1]], &mut rng);

    let description = json!({
        "arch": "mobilenet_v3_small",
        "heads": ["a", "b"],
        "classes": {"a": ["x", "y", "z"], "b": ["p", "q"]},
        "input": [1, 3, 32, 32],
        "preprocessing": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
    });
    let d = model_json::parse(&description.to_string()).unwrap();
    let run = Run {
        recipe: Recipe {
            epochs: 2,
            batch: 2,
            ..Recipe::default()
        },
        spec: d.spec,
        mean: d.mean.unwrap(),
        std: d.std.unwrap(),
        cache,
        init: Init::Scratch,
        out: dir.join("run"),
        description,
        device: Device::flex().autodiff(),
    };
    let outcome = train::train(&run).unwrap();

    let metrics: Vec<Value> = fs::read_to_string(run.out.join("metrics.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(metrics.len(), 2);
    let mut weights = Vec::new();
    for (epoch, report) in metrics.iter().enumerate() {
        let kept = epoch_dir(&run.out, epoch);
        let t = train::load(&kept, &Device::flex()).unwrap();
        assert_eq!(t.description["trained"]["epoch"], json!(epoch));
        assert_eq!(&t.description["val"], report, "epoch {epoch}'s own report");
        weights.push(fs::read(kept.join("model.safetensors")).unwrap());
    }
    assert_ne!(weights[0], weights[1], "a step moves the weights");

    let best = epoch_dir(&run.out, outcome.best_epoch.unwrap());
    for file in ["model.safetensors", "model.json"] {
        assert!(
            fs::read(run.out.join(file)).unwrap() == fs::read(best.join(file)).unwrap(),
            "{file} at the top is the best epoch's"
        );
    }
    fs::remove_dir_all(&dir).unwrap();
}
