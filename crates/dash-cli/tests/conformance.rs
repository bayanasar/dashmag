//! Cross-backend conformance: the architecture's central claim is *"one codebase,
//! correct everywhere"* — identical outputs from structurally different runtimes.
//!
//! Asserting each backend against a hardcoded constant separately does not test
//! that claim; only comparing the backends **against each other** does. This is
//! the test that fails if the abstraction starts lying.
//!
//! Needs both fixtures and `iree-run-module`:
//!   DASH_IREE_RUN_MODULE=<path> cargo test -p dash-cli --features iree -- --ignored

#![cfg(all(feature = "tract", feature = "iree"))]

use dash_core::{
    argmax, AccelClass, Artifact, ArtifactMeta, BackendKind, DType, HostTensor, Runtime,
    TensorSpec, Tensor,
};
use std::{env, fs, path::PathBuf};

fn fixtures_dir() -> PathBuf {
    env::var("DASH_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"))
}

fn meta() -> ArtifactMeta {
    ArtifactMeta {
        inputs: vec![TensorSpec::new(DType::F32, &[1, 3, 224, 224])],
        outputs: vec![TensorSpec::new(DType::F32, &[1, 1000])],
        ..Default::default()
    }
}

fn input(dir: &PathBuf) -> Tensor {
    let raw = fs::read(dir.join("input_1x3x224x224.f32")).expect("input fixture missing");
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Tensor::Host(HostTensor::from_f32(vec![1, 3, 224, 224], &floats))
}

#[test]
#[ignore = "needs fixtures + iree-run-module; run with --ignored"]
fn tract_and_iree_agree_on_the_same_model() {
    let dir = fixtures_dir();

    let tract_out = dash_tract::TractRuntime::new()
        .load(&Artifact {
            backend: BackendKind::Tract,
            bytes: fs::read(dir.join("mnv3_legacy.onnx")).expect("onnx fixture missing"),
            meta: meta(),
        })
        .unwrap()
        .run(&[input(&dir)])
        .unwrap();

    let iree_out = dash_iree::IreeRuntime::new("local-task", AccelClass::Cpu)
        .load(&Artifact {
            backend: BackendKind::Iree,
            bytes: fs::read(dir.join("mnv3_cpu.vmfb")).expect("vmfb fixture missing"),
            meta: meta(),
        })
        .unwrap()
        .run(&[input(&dir)])
        .unwrap();

    assert_eq!(
        tract_out.len(),
        iree_out.len(),
        "backends must agree on output count"
    );
    // The shape agreement is the part that silently regressed before ArtifactMeta
    // became load-bearing: tract kept [1,1000] while the raw-buffer backends
    // returned a flat [1000].
    assert_eq!(
        tract_out[0].shape(),
        iree_out[0].shape(),
        "backends must agree on output shape"
    );

    let a = tract_out[0].to_host().unwrap().as_f32().unwrap();
    let b = iree_out[0].to_host().unwrap().as_f32().unwrap();
    assert_eq!(a.len(), b.len());

    let (ia, va) = argmax(&a).unwrap();
    let (ib, vb) = argmax(&b).unwrap();
    assert_eq!(ia, ib, "backends must agree on the predicted class");
    assert!(
        (va - vb).abs() < 1e-3,
        "top logits diverged: tract {va} vs iree {vb}"
    );

    // Full-vector agreement, not just the argmax: a backend can get the winner
    // right while being wrong everywhere else.
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-3,
        "max per-logit divergence {worst} exceeds tolerance"
    );
}
