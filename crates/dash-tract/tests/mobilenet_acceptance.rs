//! Phase-0 known-answer acceptance: MobileNetV3-Small on the fixed Samoyed image
//! must return ImageNet class 258 through the `dash-core` trait.
//!
//! Ignored by default because it needs large model fixtures. Generate them with
//! `tools/export_mobilenet.py` (writes to `fixtures/`), then:
//!   cargo test -p dash-tract -- --ignored
//! Fixture location overridable via the `DASH_FIXTURES` env var.

use dash_core::{Artifact, ArtifactMeta, BackendKind, HostTensor, Runtime, Tensor};
use dash_tract::TractRuntime;
use std::{env, fs, path::PathBuf};

fn fixtures_dir() -> PathBuf {
    env::var("DASH_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"))
}

#[test]
#[ignore = "needs MobileNetV3-Small fixtures: run tools/export_mobilenet.py, then `cargo test -- --ignored`"]
fn mobilenet_v3_small_top1_is_samoyed() {
    let dir = fixtures_dir();
    let model = fs::read(dir.join("mnv3_legacy.onnx"))
        .expect("onnx fixture missing — run tools/export_mobilenet.py");
    let raw =
        fs::read(dir.join("input_1x3x224x224.f32")).expect("input fixture missing");
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let art = Artifact {
        backend: BackendKind::Tract,
        bytes: model,
        meta: ArtifactMeta::default(),
    };
    let session = TractRuntime::new().load(&art).unwrap();
    let out = session
        .run(&[Tensor::Host(HostTensor::from_f32(vec![1, 3, 224, 224], &floats))])
        .unwrap();
    let logits = out[0].to_host().unwrap().as_f32().unwrap();
    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(idx, 258, "MobileNetV3-Small top-1 must be class 258 (Samoyed)");
}
