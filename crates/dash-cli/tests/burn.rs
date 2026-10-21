//! `dash --backend burn`: weights and a description in, the module's own answer out.
//!
//! A two-head MobileNetV3-Small with random weights is written as `model.safetensors`
//! and `model.json`, as a trainer would leave them; `dash` must build the model the
//! description names, load the weights, and return the module's logits bit for bit.

use burn::tensor::{Device, Distribution, Tensor};
use dash_core::{ModelSpec, OutputSpec};
use dash_models::Model;
use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("burn-{}-{name}", std::process::id()))
}

#[test]
fn dash_runs_a_burn_model_from_weights_and_its_description() {
    let device = Device::flex();
    let spec = ModelSpec {
        arch: "mobilenet_v3_small".to_string(),
        outputs: vec![
            OutputSpec {
                name: "a".to_string(),
                classes: vec!["a0".into(), "a1".into(), "a2".into()],
            },
            OutputSpec {
                name: "b".to_string(),
                classes: vec!["b0".into(), "b1".into()],
            },
        ],
        undefined_when: Vec::new(),
    };
    let model = Model::build(&spec, &device).unwrap();
    let x = Tensor::<4>::random([1, 3, 64, 64], Distribution::Normal(0.0, 1.0), &device);
    let input: Vec<f32> = x.to_data().try_to_vec().unwrap();
    let want: Vec<u8> = model
        .forward(x)
        .into_iter()
        .flat_map(|y| y.into_data().try_to_vec::<f32>().unwrap())
        .flat_map(f32::to_le_bytes)
        .collect();

    let (weights, meta, x_path, logits) =
        (tmp("model.safetensors"), tmp("model.json"), tmp("x.f32"), tmp("logits.f32"));
    fs::write(&weights, model.to_safetensors().unwrap()).unwrap();
    fs::write(
        &meta,
        r#"{"arch": "mobilenet_v3_small", "heads": ["a", "b"],
            "classes": {"a": ["a0", "a1", "a2"], "b": ["b0", "b1"]},
            "input": [1, 3, 64, 64]}"#,
    )
    .unwrap();
    fs::write(&x_path, input.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_dash"))
        .args(["--backend", "burn", "--device", "cpu", "--model"])
        .arg(&weights)
        .arg("--meta")
        .arg(&meta)
        .arg("--input")
        .arg(&x_path)
        .arg("--logits")
        .arg(&logits)
        .output()
        .unwrap();
    let got = fs::read(&logits);
    for p in [&weights, &meta, &x_path, &logits] {
        fs::remove_file(p).ok();
    }
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("dispatch -> Burn"));
    assert_eq!(got.unwrap(), want, "dash must return the module's logits bit for bit");
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.starts_with("a: a"), "{text}");
    assert!(text.lines().nth(1).unwrap().starts_with("b: b"), "{text}");
}
