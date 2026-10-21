//! torchvision's MobileNetV3-Small through the whole Burn path — checkpoint read in Rust,
//! weights written to safetensors, an artifact chosen by the registry, the module built
//! and run on Flex — against tract running torch's ONNX export, both behind the same
//! trait.
//!
//! Ignored by default: it needs torchvision's checkpoint (see `dash-models`'
//! `tests/torchvision.rs` for where and how to fetch it) and the MobileNetV3-Small
//! fixtures. Then:
//!   cargo test -p dash-burn -- --ignored

use burn::tensor::Device;
use dash_burn::BurnRuntime;
use dash_core::{
    AccelClass, Artifact, ArtifactMeta, BackendKind, DType, HostTensor, ModelSpec, OutputSpec,
    Registry, TargetProfile, Tensor, TensorSpec, argmax,
};
use dash_models::{MobileNetV3Small, weights};
use dash_tract::TractRuntime;
use std::{env, fs, path::PathBuf};

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

fn meta(model: Option<ModelSpec>) -> ArtifactMeta {
    ArtifactMeta {
        inputs: vec![TensorSpec::new(DType::F32, &[1, 3, 224, 224])],
        outputs: vec![TensorSpec::new(DType::F32, &[1, 1000])],
        model,
        ..Default::default()
    }
}

#[test]
#[ignore = "needs torchvision's checkpoint and the MobileNetV3-Small fixtures; run with --ignored"]
fn burn_artifact_answers_as_tract_does_through_the_registry() {
    let dir = fixtures_dir();
    let mut module = MobileNetV3Small::new(&[1000], 0.2, &Device::flex());
    weights::load_pytorch(
        &mut module,
        checkpoint(),
        None,
        &[
            (r"^classifier\.0\.", "neck.0."),
            (r"^classifier\.3\.", "heads.0."),
        ],
    )
    .unwrap();
    let classes = fs::read_to_string(dir.join("imagenet_classes.txt")).unwrap();
    let spec = ModelSpec {
        arch: "mobilenet_v3_small".to_string(),
        outputs: vec![OutputSpec {
            name: "imagenet".to_string(),
            classes: classes.lines().map(str::to_string).collect(),
        }],
        undefined_when: Vec::new(),
    };
    let burn = Artifact {
        backend: BackendKind::Burn,
        bytes: weights::to_safetensors(&module).unwrap(),
        meta: meta(Some(spec)),
    };
    let tract = Artifact {
        backend: BackendKind::Tract,
        bytes: fs::read(dir.join("mnv3_legacy.onnx")).unwrap(),
        meta: meta(None),
    };

    let mut reg = Registry::new();
    reg.register(Box::new(TractRuntime::new()));
    reg.register(Box::new(BurnRuntime::flex()));
    let cpu = TargetProfile {
        accel: AccelClass::Cpu,
        no_std: false,
    };
    let raw = fs::read(dir.join("input_1x3x224x224.f32")).unwrap();
    let x: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let run = |art: &Artifact| {
        let rt = reg
            .select(art, &cpu)
            .expect("no backend admitted the artifact");
        assert_eq!(rt.kind(), art.backend, "dispatch picked the wrong backend");
        let out = rt
            .load(art)
            .unwrap()
            .run(&[Tensor::Host(HostTensor::from_f32(vec![1, 3, 224, 224], &x))])
            .unwrap();
        assert_eq!(out[0].shape(), &[1, 1000]);
        out[0].to_host().unwrap().as_f32().unwrap()
    };
    let (got, want) = (run(&burn), run(&tract));

    let (class, logit) = argmax(&got).unwrap();
    assert_eq!(class, 258, "top-1 must be class 258 (Samoyed)");
    assert!(
        (logit - 11.7283).abs() < 1e-4,
        "logit {logit} is not torch's 11.7283"
    );
    let d = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!("Burn artifact: 258 @ {logit:.6}; max |Burn - tract| over 1000 logits {d:.2e}");
    assert!(d < 1e-4, "Burn and tract differ by {d:.2e} on some logit");
}
