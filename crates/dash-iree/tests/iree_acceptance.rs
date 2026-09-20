//! Phase-0 known-answer acceptance for the IREE backend (CPU `.vmfb`).
//!
//! Ignored by default: needs `iree-run-module` (on PATH or `DASH_IREE_RUN_MODULE`)
//! and a compiled llvm-cpu module (see `tools/compile_iree.sh`). Run:
//!   DASH_IREE_RUN_MODULE=<path> cargo test -p dash-iree -- --ignored
//!
//! A `.vmfb` is per-target machine code, so the module is named by the
//! environment rather than hardcoded — the test runs where the artifact was
//! deployed, not where it was built:
//!
//!   DASH_FIXTURES        fixture directory
//!   DASH_IREE_CPU_VMFB   llvm-cpu module (default: the `mnv3_<arch>_cpu.vmfb`
//!                        that `tools/compile_iree.sh` writes on this machine)
//!
//! A relative value resolves against the fixture directory; an absolute one is
//! used as given.

use dash_core::{
    argmax, AccelClass, Artifact, ArtifactMeta, BackendKind, DType, HostTensor, Runtime, Tensor,
    TensorSpec,
};
use dash_iree::IreeRuntime;
use std::{env, fs, path::PathBuf};

const REF_CLASS: usize = 258;
const REF_LOGIT: f32 = 11.7283;

fn fixtures_dir() -> PathBuf {
    env::var("DASH_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"))
}

#[test]
#[ignore = "needs iree-run-module + a cpu .vmfb (tools/compile_iree.sh); run with --ignored"]
fn iree_cpu_matches_torch_reference() {
    let dir = fixtures_dir();
    let path = dir.join(
        env::var("DASH_IREE_CPU_VMFB")
            .unwrap_or_else(|_| format!("mnv3_{}_cpu.vmfb", env::consts::ARCH)),
    );
    let vmfb = fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{} unreadable: {e} — run tools/compile_iree.sh or set DASH_IREE_CPU_VMFB",
            path.display()
        )
    });
    let raw = fs::read(dir.join("input_1x3x224x224.f32")).expect("input fixture missing");
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let art = Artifact {
        backend: BackendKind::Iree,
        bytes: vmfb,
        meta: ArtifactMeta {
            inputs: vec![TensorSpec::new(DType::F32, &[1, 3, 224, 224])],
            outputs: vec![TensorSpec::new(DType::F32, &[1, 1000])],
            ..Default::default()
        },
    };
    let session = IreeRuntime::new("local-task", AccelClass::Cpu)
        .load(&art)
        .unwrap();
    let out = session
        .run(&[Tensor::Host(HostTensor::from_f32(
            vec![1, 3, 224, 224],
            &floats,
        ))])
        .unwrap();

    assert_eq!(out.len(), 1);
    // The raw buffer from iree-run-module carries no shape; ArtifactMeta must
    // restore it so IREE agrees with tract instead of returning a flat [1000].
    assert_eq!(
        out[0].shape(),
        &[1, 1000],
        "declared output shape must be restored from ArtifactMeta"
    );

    let logits = out[0].to_host().unwrap().as_f32().unwrap();
    let (idx, val) = argmax(&logits).unwrap();
    assert_eq!(idx, REF_CLASS, "IREE CPU top-1 must be class 258 (Samoyed)");
    assert!(
        (val - REF_LOGIT).abs() < 1e-3,
        "logit {val} drifted from torch reference {REF_LOGIT}"
    );
}
