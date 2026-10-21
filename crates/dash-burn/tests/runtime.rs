//! The Burn runtime through the `dash-core` trait, with no fixtures: a module with random
//! weights, written to safetensors and handed over as an artifact, must answer exactly as
//! the module itself does — and an artifact naming a model this binary does not have must
//! be refused with a reason.

use burn::tensor::{Device, Distribution, Tensor};
use dash_burn::BurnRuntime;
use dash_core::{
    AccelClass, Artifact, ArtifactMeta, BackendKind, DType, HostTensor, ModelSpec, OutputSpec,
    Registry, Runtime, TargetProfile, Tensor as DashTensor, TensorSpec,
};
use dash_models::Model;

fn spec(arch: &str) -> ModelSpec {
    let out = |name: &str, n: usize| OutputSpec {
        name: name.to_string(),
        classes: (0..n).map(|i| format!("{name}{i}")).collect(),
    };
    ModelSpec {
        arch: arch.to_string(),
        outputs: vec![out("a", 3), out("b", 2)],
        undefined_when: Vec::new(),
    }
}

fn artifact(bytes: Vec<u8>, model: Option<ModelSpec>) -> Artifact {
    Artifact {
        backend: BackendKind::Burn,
        bytes,
        meta: ArtifactMeta {
            inputs: vec![TensorSpec::new(DType::F32, &[2, 3, 64, 64])],
            outputs: vec![
                TensorSpec::new(DType::F32, &[2, 3]),
                TensorSpec::new(DType::F32, &[2, 2]),
            ],
            model,
            ..Default::default()
        },
    }
}

#[test]
fn weights_through_an_artifact_answer_as_the_module_does() {
    let device = Device::flex();
    let model = Model::build(&spec("mobilenet_v3_small"), &device).unwrap();
    let bytes = model.to_safetensors().unwrap();

    let x = Tensor::<4>::random([2, 3, 64, 64], Distribution::Normal(0.0, 1.0), &device);
    let input: Vec<f32> = x.to_data().try_to_vec().unwrap();
    let want: Vec<Vec<f32>> = model
        .forward(x)
        .into_iter()
        .map(|y| y.into_data().try_to_vec().unwrap())
        .collect();

    let session = BurnRuntime::flex()
        .load(&artifact(bytes, Some(spec("mobilenet_v3_small"))))
        .unwrap();
    let out = session
        .run(&[DashTensor::Host(HostTensor::from_f32(
            vec![2, 3, 64, 64],
            &input,
        ))])
        .unwrap();

    assert_eq!(out.len(), 2, "one output per head");
    assert_eq!(out[0].shape(), &[2, 3]);
    assert_eq!(out[1].shape(), &[2, 2]);
    for (got, want) in out.iter().zip(&want) {
        // Same module, same weights, same device: the same bits, not merely close ones.
        assert_eq!(&got.to_host().unwrap().as_f32().unwrap(), want);
    }
}

#[test]
fn a_model_this_binary_does_not_have_is_refused_with_a_reason() {
    let mut reg = Registry::new();
    reg.register(Box::new(BurnRuntime::flex()));
    let cpu = TargetProfile {
        accel: AccelClass::Cpu,
        no_std: false,
    };

    let unknown = artifact(Vec::new(), Some(spec("resnet18")));
    assert!(reg.select(&unknown, &cpu).is_none());
    let why = &reg.explain(&unknown, &cpu)[0].1;
    assert!(why.contains("\"resnet18\" is not compiled"), "{why}");

    let undescribed = artifact(Vec::new(), None);
    assert!(reg.select(&undescribed, &cpu).is_none());
    let why = &reg.explain(&undescribed, &cpu)[0].1;
    assert!(why.contains("needs meta.model"), "{why}");
}

#[test]
fn weights_for_a_different_model_are_refused_not_half_loaded() {
    let device = Device::flex();
    // Three heads written, two expected: a tensor with nowhere to go.
    let mut three = spec("mobilenet_v3_small");
    three.outputs.push(OutputSpec {
        name: "c".to_string(),
        classes: vec!["c0".to_string(), "c1".to_string()],
    });
    let bytes = Model::build(&three, &device)
        .unwrap()
        .to_safetensors()
        .unwrap();
    let err = BurnRuntime::flex()
        .load(&artifact(bytes, Some(spec("mobilenet_v3_small"))))
        .err()
        .expect("a mismatched file must not load");
    assert!(err.to_string().contains("heads.2"), "{err}");
}
