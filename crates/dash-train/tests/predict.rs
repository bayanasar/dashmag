//! A model's predictions written down: each frame of a split, in the cache's order, with
//! each head's top-1 class, the heads in the model's order and not sorted.

use std::fs;
use std::path::{Path, PathBuf};

use burn::tensor::Device;
use dash_models::{MobileNetV3Small, weights};
use dash_train::data::{Split, npy_header_bytes};
use dash_train::rng::Rng;
use dash_train::train;
use serde_json::{Value, json};

/// A model with untrained weights, written as `train` leaves one: two heads, the second
/// named first in the alphabet.
fn model_in(dir: &Path, device: &Device) {
    fs::create_dir_all(dir).unwrap();
    let description = json!({
        "arch": "mobilenet_v3_small",
        "heads": ["zone", "angle"],
        "classes": {"zone": ["x", "y", "z"], "angle": ["p", "q"]},
        "input": [1, 3, 32, 32],
        "preprocessing": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
    });
    fs::write(dir.join("model.json"), description.to_string()).unwrap();
    let model = MobileNetV3Small::new(&[3, 2], 0.0, device);
    fs::write(
        dir.join("model.safetensors"),
        weights::to_safetensors(&model).unwrap(),
    )
    .unwrap();
}

/// A split of `rows.len()` 32 x 32 noise frames, its rows as given.
fn split(cache: &Path, name: &str, rows: &[Value]) {
    fs::create_dir_all(cache).unwrap();
    let mut rng = Rng::new(5);
    let mut npy = npy_header_bytes([rows.len(), 32, 32, 3]);
    npy.extend((0..rows.len() * 32 * 32 * 3).map(|_| rng.next_u64() as u8));
    fs::write(cache.join(format!("{name}.npy")), npy).unwrap();
    let text: String = rows.iter().map(|r| r.to_string() + "\n\n").collect();
    fs::write(cache.join(format!("{name}.jsonl")), text).unwrap();
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    dir
}

#[test]
fn each_frame_gets_each_heads_top_class() {
    let (dir, device) = (scratch("predict"), Device::flex());
    model_in(&dir.join("run"), &device);
    let masked = [-100, -100];
    let rows: Vec<Value> = ["a/1.jpg", "a/2.jpg", "b/3.jpg", "c/4.jpg", "c/5.jpg"]
        .iter()
        .map(|image| json!({"image": image, "targets": masked, "other": 1}))
        .collect();
    split(&dir.join("cache"), "frames", &rows);

    let t = train::load(&dir.join("run"), &device).unwrap();
    let text = train::predictions(&t, &dir.join("cache"), "frames", 2, &device).unwrap();
    let s = Split::open(&dir.join("cache"), "frames", 2).unwrap();
    let pred = train::predict(&t.model, &s, t.mean, t.std, 5, &device).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 5);
    for (i, line) in lines.iter().enumerate() {
        let image = rows[i]["image"].as_str().unwrap();
        let (zone, angle) = (["x", "y", "z"][pred[0][i]], ["p", "q"][pred[1][i]]);
        let want = format!(
            r#"{{"image": "{image}", "predicted": {{"zone": "{zone}", "angle": "{angle}"}}}}"#
        );
        assert_eq!(*line, want);
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_row_without_an_image_cannot_be_written_down() {
    let (dir, device) = (scratch("predict-no-image"), Device::flex());
    model_in(&dir.join("run"), &device);
    let rows = [
        json!({"image": "a.jpg", "targets": [0, 1]}),
        json!({"targets": [1, 0]}),
    ];
    split(&dir.join("cache"), "val", &rows);
    let t = train::load(&dir.join("run"), &device).unwrap();
    let err = train::predictions(&t, &dir.join("cache"), "val", 8, &device).unwrap_err();
    let path = dir.join("cache/val.jsonl");
    assert_eq!(err, format!("{} row 2: no image", path.display()));
    fs::remove_dir_all(&dir).unwrap();
}
