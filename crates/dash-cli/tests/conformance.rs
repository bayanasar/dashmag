//! Cross-backend conformance: the architecture's central claim is *"one codebase,
//! correct everywhere"* — identical outputs from structurally different runtimes.
//!
//! Asserting each backend against a hardcoded constant separately does not test
//! that claim; only comparing the backends **against each other** does. This is
//! the test that fails if the abstraction starts lying.
//!
//! Needs both fixtures and `iree-run-module`:
//!   DASH_IREE_RUN_MODULE=<path> cargo test -p dash-cli --features iree -- --ignored --nocapture
//!
//! `--nocapture` is not decoration: the run prints which IREE targets it actually
//! compared, and a passing test says nothing without it.
//!
//! A `.vmfb` is per-target machine code, so the artifact cannot be hardcoded: the
//! test runs where the artifact was *deployed*, which is not where it was built.
//! The environment names it, and nothing is inferred:
//!
//!   DASH_FIXTURES        fixture directory
//!   DASH_IREE_CPU_VMFB   llvm-cpu module (default: the `mnv3_<arch>_cpu.vmfb`
//!                        that `tools/compile_iree.sh` writes on this machine)
//!   DASH_IREE_CUDA_VMFB  cuda module — when set, the CUDA path is held to the
//!                        same bar as the CPU one
//!   DASH_IREE_MH_CPU_VMFB / DASH_IREE_MH_CUDA_VMFB
//!                        the same, for the five-output fixture built by
//!                        `tools/export_multihead.py`
//!   DASH_IREE_REQUIRE_CUDA=1
//!                        fail instead of skipping when no CUDA module is named.
//!                        A board run sets it, so a claim about that board's GPU
//!                        cannot come from a run that never touched it.
//!
//! Relative values resolve against the fixture directory; absolute ones are used
//! as given.

#![cfg(all(feature = "tract", feature = "iree"))]

use dash_core::{
    argmax, AccelClass, Artifact, ArtifactMeta, BackendKind, DType, HostTensor, Runtime, Tensor,
    TensorSpec,
};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn fixtures_dir() -> PathBuf {
    env::var("DASH_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"))
}

/// The name `tools/compile_iree.sh` gives the llvm-cpu module built *here*.
/// This is a default, not a detection: an artifact compiled elsewhere and
/// deployed to this machine is named by `DASH_IREE_CPU_VMFB` instead.
fn default_cpu_vmfb() -> String {
    format!("mnv3_{}_cpu.vmfb", env::consts::ARCH)
}

/// One IREE configuration to hold against tract: the device the runtime
/// dispatches on, the class the registry ranks by, and the module built for it.
struct IreeTarget {
    device: &'static str,
    accel: AccelClass,
    vmfb: PathBuf,
}

/// The CPU module is always checked, so the test never passes without comparing
/// something. The CUDA module is checked when the environment names one: a GPU
/// artifact is built for one compute capability and cannot be defaulted, but
/// where it exists it answers to exactly the same bar.
///
/// Opt-in is right; silence about it is not. A run that skipped the GPU leg is
/// indistinguishable in its result from one that checked it, and the sentence
/// written down afterwards — "conformance passes on the Orin" — is read as
/// covering the GPU either way. So the caller reports the targets it compared,
/// and `DASH_IREE_REQUIRE_CUDA=1` turns the skip into a failure for runs that
/// are being made to stand behind a board's GPU.
fn iree_targets(dir: &Path) -> Vec<IreeTarget> {
    iree_targets_named(
        dir,
        "DASH_IREE_CPU_VMFB",
        "DASH_IREE_CUDA_VMFB",
        default_cpu_vmfb(),
    )
}

/// The same rule for any fixture family: the CPU module has a default name, the
/// CUDA module is named by the environment or not checked at all.
fn iree_targets_named(
    dir: &Path,
    cpu_env: &str,
    cuda_env: &str,
    cpu_default: String,
) -> Vec<IreeTarget> {
    let mut targets = vec![IreeTarget {
        device: "local-task",
        accel: AccelClass::Cpu,
        vmfb: dir.join(env::var(cpu_env).unwrap_or(cpu_default)),
    }];
    match env::var(cuda_env) {
        Ok(path) => targets.push(IreeTarget {
            device: "cuda",
            accel: AccelClass::Cuda,
            vmfb: dir.join(path),
        }),
        Err(_) if env::var("DASH_IREE_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
            panic!(
                "DASH_IREE_REQUIRE_CUDA=1 but no {cuda_env}: the CUDA leg would \
                    have been skipped, and this run is not allowed to pass without it"
            )
        }
        Err(_) => {}
    }
    targets
}

/// Every output, element by element, not just the argmax and not just output 0.
/// A backend can get the winner right on the first head and be wrong on the rest;
/// with five heads that is no longer a hypothetical.
fn assert_agrees(tract_out: &[Tensor], iree_out: &[Tensor], dev: &str) {
    assert_eq!(
        tract_out.len(),
        iree_out.len(),
        "backends must agree on output count (iree {dev})"
    );
    for (i, (t, r)) in tract_out.iter().zip(iree_out).enumerate() {
        // The shape agreement is the part that silently regressed before ArtifactMeta
        // became load-bearing: tract kept [1,1000] while the raw-buffer backends
        // returned a flat [1000].
        assert_eq!(
            t.shape(),
            r.shape(),
            "backends must agree on the shape of output {i} (iree {dev})"
        );
        let a = t.to_host().unwrap().as_f32().unwrap();
        let b = r.to_host().unwrap().as_f32().unwrap();
        assert_eq!(a.len(), b.len(), "output {i} length (iree {dev})");

        let (ia, va) = argmax(&a).unwrap();
        let (ib, vb) = argmax(&b).unwrap();
        assert_eq!(
            ia, ib,
            "backends must agree on the predicted class of output {i} (iree {dev})"
        );
        assert!(
            (va - vb).abs() < 1e-3,
            "output {i} top logits diverged: tract {va} vs iree {dev} {vb}"
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
            "output {i}: max per-logit divergence {worst} exceeds tolerance (iree {dev})"
        );
    }
}

/// Load and run one IREE target. `function` names the entry point: artifacts that
/// came through `iree-import-onnx` call it `main_graph` (the runtime's default),
/// iree-turbine calls it `main`.
fn run_iree(
    dir: &Path,
    target: &IreeTarget,
    meta: &ArtifactMeta,
    input: Tensor,
    function: Option<&str>,
) -> Vec<Tensor> {
    let dev = target.device;
    let path = if target.vmfb.is_absolute() {
        target.vmfb.clone()
    } else {
        dir.join(&target.vmfb)
    };
    let bytes =
        fs::read(&path).unwrap_or_else(|e| panic!("{dev} vmfb {} unreadable: {e}", path.display()));
    let mut rt = dash_iree::IreeRuntime::new(dev, target.accel);
    if let Some(f) = function {
        rt = rt.with_function(f);
    }
    rt.load(&Artifact {
        backend: BackendKind::Iree,
        bytes,
        meta: meta.clone(),
    })
    .unwrap()
    .run(&[input])
    .unwrap()
}

/// The run states its own coverage, so a passing result cannot be read as covering
/// a target it never touched.
fn report(what: &str, targets: &[IreeTarget]) {
    let compared: Vec<String> = targets
        .iter()
        .map(|t| format!("{} <- {}", t.device, t.vmfb.display()))
        .collect();
    eprintln!(
        "conformance ({what}): tract vs {} IREE target(s): {}",
        targets.len(),
        compared.join(", ")
    );
}

fn meta() -> ArtifactMeta {
    ArtifactMeta {
        inputs: vec![TensorSpec::new(DType::F32, &[1, 3, 224, 224])],
        outputs: vec![TensorSpec::new(DType::F32, &[1, 1000])],
        ..Default::default()
    }
}

fn input(dir: &Path) -> Tensor {
    let raw = fs::read(dir.join("input_1x3x224x224.f32")).expect("input fixture missing");
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Tensor::Host(HostTensor::from_f32(vec![1, 3, 224, 224], &floats))
}

/// Five heads, the widths Phase 1 pinned: time_of_day 4, weather 6, road_type 6,
/// surface 4, traffic 4.
fn mh5_meta() -> ArtifactMeta {
    ArtifactMeta {
        inputs: vec![TensorSpec::new(DType::F32, &[1, 3, 64, 64])],
        outputs: [4, 6, 6, 4, 4]
            .iter()
            .map(|n| TensorSpec::new(DType::F32, &[1, *n]))
            .collect(),
        ..Default::default()
    }
}

fn mh5_input(dir: &Path) -> Tensor {
    let raw = fs::read(dir.join("mh5_input_1x3x64x64.f32")).expect("mh5 input fixture missing");
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Tensor::Host(HostTensor::from_f32(vec![1, 3, 64, 64], &floats))
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

    let targets = iree_targets(&dir);
    report("mobilenetv3, 1 output", &targets);

    for target in &targets {
        let iree_out = run_iree(&dir, target, &meta(), input(&dir), None);
        assert_agrees(&tract_out, &iree_out, target.device);
    }
}

/// Phase 1's model is one backbone with five attribute heads, and until this test
/// existed nothing in the repo had ever declared more than one output — the plumbing
/// was written for `Vec<TensorSpec>` and never exercised. `tools/export_multihead.py`
/// builds the fixture; the head widths are the real ones (4, 6, 6, 4, 4).
#[test]
#[ignore = "needs the mh5 fixtures + iree-run-module; run with --ignored"]
fn tract_and_iree_agree_on_every_output_of_a_five_head_model() {
    let dir = fixtures_dir();
    let meta = mh5_meta();

    let tract_out = dash_tract::TractRuntime::new()
        .load(&Artifact {
            backend: BackendKind::Tract,
            bytes: fs::read(dir.join("mh5.onnx")).expect("mh5.onnx fixture missing"),
            meta: meta.clone(),
        })
        .unwrap()
        .run(&[mh5_input(&dir)])
        .unwrap();
    assert_eq!(tract_out.len(), 5, "the fixture must have five outputs");

    let targets = iree_targets_named(
        &dir,
        "DASH_IREE_MH_CPU_VMFB",
        "DASH_IREE_MH_CUDA_VMFB",
        format!("mh5_{}_cpu.vmfb", env::consts::ARCH),
    );
    report("five heads", &targets);

    for target in &targets {
        // Turbine names the entry point `main`; the ONNX importer names it `main_graph`.
        let iree_out = run_iree(&dir, target, &meta, mh5_input(&dir), Some("main"));
        assert_agrees(&tract_out, &iree_out, target.device);
    }
}
