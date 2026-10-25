//! Every epoch's weights are kept, and the ones at the top of the run are the best epoch's;
//! and a run's weights are fixed by its seed, whatever the number of threads Flex runs on.
//!
//! Two runs compared only at the epochs each selected mix the selection into the result; a
//! comparison at a fixed epoch needs every epoch's weights, and two runs that differ only
//! in their thread count must not differ at all. Both tests train two epochs on a small
//! synthetic cache and read each epoch back.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

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

/// Flex draws initial weights and dropout masks from one generator per process, which
/// `train` seeds, and building a module to load weights into draws from it too; a test
/// running beside a training would shift its draws, so each test holds this for its whole
/// body.
static TRAINING: Mutex<()> = Mutex::new(());

/// A two-epoch run from scratch in batches of `batch`, over a synthetic cache of 32 x 32
/// frames with `targets` for training and two for validation, both in `dir`.
fn run_in(dir: &Path, targets: &[[i64; 2]], batch: usize) -> Run {
    let _ = fs::remove_dir_all(dir);
    let cache = dir.join("cache");
    fs::create_dir_all(&cache).unwrap();
    let mut rng = Rng::new(7);
    split(&cache, "train", 32, targets, &mut rng);
    split(&cache, "val", 32, &[[1, 0], [2, 1]], &mut rng);

    let description = json!({
        "arch": "mobilenet_v3_small",
        "heads": ["a", "b"],
        "classes": {"a": ["x", "y", "z"], "b": ["p", "q"]},
        "input": [1, 3, 32, 32],
        "preprocessing": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
    });
    let d = model_json::parse(&description.to_string()).unwrap();
    Run {
        recipe: Recipe {
            epochs: 2,
            batch,
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
    }
}

#[test]
fn every_epoch_is_kept_and_the_best_is_at_the_top() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("every-epoch");
    let _turn = TRAINING.lock().unwrap_or_else(PoisonError::into_inner);
    let run = run_in(&dir, &[[0, 0], [1, 1], [2, 0], [0, 1]], 2);
    let outcome = train::train(&run).unwrap();
    // The last epoch is not the best, so a run that wrote the top level after every epoch
    // would leave the last one there and fail below.
    assert_eq!(outcome.best_epoch, Some(0));

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

/// Every tensor of a safetensors file — name, dtype, shape and bytes — without the
/// metadata, which Burn writes in hash order.
fn tensors(path: &Path) -> Vec<(String, String, Vec<usize>, Vec<u8>)> {
    let bytes = fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut out: Vec<_> = st
        .tensors()
        .into_iter()
        .map(|(name, t)| {
            let dtype = format!("{:?}", t.dtype());
            (name, dtype, t.shape().to_vec(), t.data().to_vec())
        })
        .collect();
    out.sort();
    out
}

/// Flex splits work across rayon's pool by the part of the output each task writes, so
/// the size of the pool must not change a bit of what a seed trains. Its convolutions split
/// at any size; batches of 16 also make the neck's matrix product (16 x 576 x 1024) large
/// enough to split.
#[test]
fn the_seed_fixes_the_weights_whatever_the_thread_count() {
    let _turn = TRAINING.lock().unwrap_or_else(PoisonError::into_inner);
    let targets: Vec<[i64; 2]> = (0..32).map(|i| [i % 3, i % 2]).collect();
    let mut runs = Vec::new();
    for threads in [1, 4] {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("threads-{threads}"));
        let run = run_in(&dir, &targets, 16);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| train::train(&run)).unwrap();
        runs.push(
            (0..2)
                .map(|epoch| tensors(&epoch_dir(&run.out, epoch).join("model.safetensors")))
                .collect::<Vec<_>>(),
        );
        fs::remove_dir_all(&dir).unwrap();
    }
    assert!(!runs[0][0].is_empty());
    for (epoch, (one, four)) in runs[0].iter().zip(&runs[1]).enumerate() {
        for (a, b) in one.iter().zip(four) {
            assert!(a == b, "epoch {epoch}: {} differs at 1 and 4 threads", a.0);
        }
        assert_eq!(one.len(), four.len());
    }
}
