//! The module is torchvision's MobileNetV3-Small: torchvision's ImageNet checkpoint, read
//! by burn-store's PyTorch reader, gives torch's answer on Flex, and the same logits as
//! tract running torch's own export of that network.
//!
//! Ignored by default: it needs torchvision's published checkpoint, which no Python is
//! needed to fetch —
//!   curl -LO https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth
//! — at `DASH_TORCHVISION_MNV3` (default: torch hub's cache,
//! `~/.cache/torch/hub/checkpoints/`), and the fixtures from `tools/export_mobilenet.py`
//! (overridable via `DASH_FIXTURES`). Then:
//!   cargo test -p dash-models --features pytorch -- --ignored
//! For an architecture this machine is not, build the test for it and run it under qemu's
//! user mode, which can pin the CPU model — a Cortex-A53 here (needs a musl cross linker):
//!   cargo test -p dash-models --features pytorch --release \
//!     --target aarch64-unknown-linux-musl --no-run
//!   qemu-aarch64 -cpu cortex-a53 target/aarch64-unknown-linux-musl/release/deps/torchvision-* \
//!     --ignored

use burn::tensor::{Device, Tensor, TensorData};
use dash_core::{
    Artifact, ArtifactMeta, BackendKind, DType, HostTensor, Runtime, Tensor as DashTensor,
    TensorSpec, argmax,
};
use dash_models::{MobileNetV3Small, weights};
use dash_tract::TractRuntime;
use std::{env, fs, path::PathBuf};

/// torch on its own preprocessing of the public test image: class 258 "Samoyed".
const REF_CLASS: usize = 258;
const REF_LOGIT: f32 = 11.7283;
/// Per logit, against tract on the same input. tract runs torch's ONNX export, in which
/// batch norm is folded into the convolutions; Burn runs them unfolded. Measured on x86-64
/// (2026-10-20), over all 1000 logits: 2.6e-5 on the recorded input, 2.2e-5 through
/// dash-image; on aarch64 under qemu's Cortex-A53, 2.3e-5 and 2.0e-5.
const TOLERANCE: f32 = 1e-4;

fn fixtures_dir() -> PathBuf {
    env::var("DASH_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"))
}

fn checkpoint() -> PathBuf {
    env::var("DASH_TORCHVISION_MNV3")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = env::var("HOME").expect("HOME unset; set DASH_TORCHVISION_MNV3");
            PathBuf::from(home).join(".cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth")
        })
}

fn imagenet_model(device: &Device) -> MobileNetV3Small {
    let mut model = MobileNetV3Small::new(&[1000], 0.2, device);
    // torchvision's classifier is `Sequential(Linear, Hardswish, Dropout, Linear)`: its
    // linear layers are indices 0 and 3, the neck and the one head here.
    weights::load_pytorch(
        &mut model,
        checkpoint(),
        None,
        &[
            (r"^classifier\.0\.", "neck.0."),
            (r"^classifier\.3\.", "heads.0."),
        ],
    )
    .expect("checkpoint did not load whole");
    model
}

fn burn_logits(model: &MobileNetV3Small, input: &[f32], device: &Device) -> Vec<f32> {
    let x = Tensor::<4>::from_data(TensorData::new(input.to_vec(), [1, 3, 224, 224]), device);
    let out = model.forward(x);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].dims(), [1, 1000]);
    out[0].to_data().try_to_vec::<f32>().unwrap()
}

fn tract_logits(input: &[f32]) -> Vec<f32> {
    let art = Artifact {
        backend: BackendKind::Tract,
        bytes: fs::read(fixtures_dir().join("mnv3_legacy.onnx")).expect("onnx fixture missing"),
        meta: ArtifactMeta {
            inputs: vec![TensorSpec::new(DType::F32, &[1, 3, 224, 224])],
            outputs: vec![TensorSpec::new(DType::F32, &[1, 1000])],
            ..Default::default()
        },
    };
    let session = TractRuntime::new().load(&art).unwrap();
    let out = session
        .run(&[DashTensor::Host(HostTensor::from_f32(
            vec![1, 3, 224, 224],
            input,
        ))])
        .unwrap();
    out[0].to_host().unwrap().as_f32().unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "needs torchvision's checkpoint and the MobileNetV3-Small fixtures; run with --ignored"]
fn imagenet_weights_give_torchs_answer_on_flex() {
    let device = Device::flex();
    let model = imagenet_model(&device);
    let raw =
        fs::read(fixtures_dir().join("input_1x3x224x224.f32")).expect("input fixture missing");
    let input: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let got = burn_logits(&model, &input, &device);
    let (class, logit) = argmax(&got).unwrap();
    assert_eq!(class, REF_CLASS, "top-1 must be class 258 (Samoyed)");
    assert!(
        (logit - REF_LOGIT).abs() < TOLERANCE,
        "logit {logit} is not torch's {REF_LOGIT}"
    );
    let d = max_diff(&got, &tract_logits(&input));
    eprintln!("recorded input: 258 @ {logit:.6}; max |Burn - tract| over 1000 logits {d:.2e}");
    assert!(
        d < TOLERANCE,
        "Burn and tract differ by {d:.2e} on some logit"
    );
}

#[test]
#[ignore = "needs torchvision's checkpoint and the MobileNetV3-Small fixtures; run with --ignored"]
fn image_through_dash_image_gives_tracts_answer_on_flex() {
    let device = Device::flex();
    let model = imagenet_model(&device);
    let jpeg = fs::read(fixtures_dir().join("dog.jpg")).expect("dog.jpg missing");
    let input = dash_image::image_input(
        &jpeg,
        224,
        224,
        dash_image::IMAGENET_MEAN,
        dash_image::IMAGENET_STD,
    )
    .unwrap();

    let got = burn_logits(&model, &input, &device);
    let want = tract_logits(&input);
    let (class, logit) = argmax(&got).unwrap();
    assert_eq!(class, REF_CLASS, "top-1 must be class 258 (Samoyed)");
    let d = max_diff(&got, &want);
    eprintln!("dash-image input: 258 @ {logit:.6}; max |Burn - tract| over 1000 logits {d:.2e}");
    assert!(
        d < TOLERANCE,
        "Burn and tract differ by {d:.2e} on some logit"
    );
}
