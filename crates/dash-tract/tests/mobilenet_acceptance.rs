//! Phase-0 known-answer acceptance: MobileNetV3-Small on the fixed Samoyed image
//! must return ImageNet class 258 through the `dash-core` trait — with the right
//! logit and the right shape, not merely the right argmax.
//!
//! Ignored by default because it needs large model fixtures. Generate them with
//! `tools/export_mobilenet.py` (writes to `fixtures/`), then:
//!   cargo test -p dash-tract -- --ignored
//! Fixture location overridable via the `DASH_FIXTURES` env var.

use dash_core::{
    argmax, Artifact, ArtifactMeta, BackendKind, DType, HostTensor, Runtime, TensorSpec, Tensor,
};
use dash_tract::TractRuntime;
use std::{env, fs, path::PathBuf};

/// PyTorch reference for the fixed input: class 258 "Samoyed".
const REF_CLASS: usize = 258;
const REF_LOGIT: f32 = 11.7283;

fn fixtures_dir() -> PathBuf {
    env::var("DASH_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"))
}

fn input_floats(dir: &PathBuf) -> Vec<f32> {
    let raw = fs::read(dir.join("input_1x3x224x224.f32")).expect("input fixture missing");
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "needs MobileNetV3-Small fixtures: run tools/export_mobilenet.py, then `cargo test -- --ignored`"]
fn mobilenet_v3_small_matches_torch_reference() {
    let dir = fixtures_dir();
    let model = fs::read(dir.join("mnv3_legacy.onnx"))
        .expect("onnx fixture missing — run tools/export_mobilenet.py");
    let floats = input_floats(&dir);

    let art = Artifact {
        backend: BackendKind::Tract,
        bytes: model,
        meta: ArtifactMeta {
            inputs: vec![TensorSpec::new(DType::F32, &[1, 3, 224, 224])],
            outputs: vec![TensorSpec::new(DType::F32, &[1, 1000])],
            ..Default::default()
        },
    };
    let session = TractRuntime::new().load(&art).unwrap();
    let out = session
        .run(&[Tensor::Host(HostTensor::from_f32(
            vec![1, 3, 224, 224],
            &floats,
        ))])
        .unwrap();

    assert_eq!(out.len(), 1, "classifier has exactly one output");
    // Shape must be the model's real shape, not a flattened buffer.
    assert_eq!(
        out[0].shape(),
        &[1, 1000],
        "output shape must match the model, uniformly across backends"
    );

    let logits = out[0].to_host().unwrap().as_f32().unwrap();
    let (idx, val) = argmax(&logits).unwrap();
    assert_eq!(idx, REF_CLASS, "top-1 must be class 258 (Samoyed)");
    // "bit-exact vs torch" is a claim; test it rather than asserting it in prose.
    assert!(
        (val - REF_LOGIT).abs() < 1e-3,
        "logit {val} drifted from torch reference {REF_LOGIT}"
    );
}
