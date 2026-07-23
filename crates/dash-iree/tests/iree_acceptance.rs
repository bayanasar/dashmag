//! Phase-0 known-answer acceptance for the IREE backend (CPU `.vmfb`).
//!
//! Ignored by default: needs `iree-run-module` (on PATH or `DASH_IREE_RUN_MODULE`)
//! and a compiled `fixtures/mnv3_cpu.vmfb` (see `tools/compile_iree.sh`). Run:
//!   DASH_IREE_RUN_MODULE=<path> cargo test -p dash-iree -- --ignored

use dash_core::{AccelClass, Artifact, ArtifactMeta, BackendKind, HostTensor, Runtime, Tensor};
use dash_iree::IreeRuntime;
use std::{env, fs, path::PathBuf};

fn fixtures_dir() -> PathBuf {
    env::var("DASH_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"))
}

#[test]
#[ignore = "needs iree-run-module + fixtures/mnv3_cpu.vmfb (tools/compile_iree.sh); run with --ignored"]
fn iree_cpu_top1_is_samoyed() {
    let dir = fixtures_dir();
    let vmfb = fs::read(dir.join("mnv3_cpu.vmfb"))
        .expect("mnv3_cpu.vmfb missing — run tools/compile_iree.sh");
    let raw = fs::read(dir.join("input_1x3x224x224.f32")).expect("input fixture missing");
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let art = Artifact {
        backend: BackendKind::Iree,
        bytes: vmfb,
        meta: ArtifactMeta::default(),
    };
    let session = IreeRuntime::new("local-task", AccelClass::Cpu)
        .load(&art)
        .unwrap();
    let out = session
        .run(&[Tensor::Host(HostTensor::from_f32(vec![1, 3, 224, 224], &floats))])
        .unwrap();
    let logits = out[0].to_host().unwrap().as_f32().unwrap();
    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(idx, 258, "IREE CPU top-1 must be class 258 (Samoyed)");
}
