//! `dash --meta`: a model's description names its outputs and says when one means
//! nothing, and changes no logit.
//!
//! Runs the `dash` binary on the five-head fixture (`fixtures/mh5.onnx`, random weights,
//! the attribute model's head widths 3, 3, 3, 2, 2) through tract. Its top-1 indices on
//! the fixture input are 0, 2, 2, 1, 1 (`mh5_reference.json`).

use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

fn fixtures_dir() -> PathBuf {
    env::var("DASH_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"))
}

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("meta-{}-{name}", std::process::id()))
}

/// The attribute model's vocabulary, with one undefined condition.
fn description(when_class: &str) -> String {
    format!(
        r#"{{
  "heads": ["time_of_day", "weather", "road_type", "surface", "traffic"],
  "classes": {{"time_of_day": ["daytime", "night", "dawn_dusk"],
              "weather": ["clear", "overcast", "rainy"],
              "road_type": ["highway", "city_street", "parking_lot"],
              "surface": ["dry", "wet"],
              "traffic": ["flowing", "congested"]}},
  "undefined_when": {{"weather": {{"time_of_day": "{when_class}"}}}},
  "input": [1, 3, 64, 64],
  "calibration_applied": {{"head": "surface", "class": "wet", "shift": 0.5,
                          "folded_into": "heads.3.bias"}}
}}"#
    )
}

/// Run `dash` on the fixture; returns stdout and the logits it wrote.
fn dash(extra: &[&str], tag: &str) -> (String, Vec<u8>) {
    let dir = fixtures_dir();
    let logits = tmp(&format!("{tag}.f32"));
    let out = Command::new(env!("CARGO_BIN_EXE_dash"))
        .args(["--backend", "tract", "--model"])
        .arg(dir.join("mh5.onnx"))
        .arg("--input")
        .arg(dir.join("mh5_input_1x3x64x64.f32"))
        .arg("--logits")
        .arg(&logits)
        .args(extra)
        .output()
        .expect("dash did not start");
    assert!(
        out.status.success(),
        "dash failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = fs::read(&logits).unwrap();
    fs::remove_file(&logits).ok();
    (String::from_utf8(out.stdout).unwrap(), bytes)
}

fn with_meta(when_class: &str, tag: &str) -> (String, Vec<u8>) {
    let meta = tmp(&format!("{tag}.json"));
    fs::write(&meta, description(when_class)).unwrap();
    let r = dash(&["--meta", meta.to_str().unwrap()], tag);
    fs::remove_file(&meta).ok();
    r
}

#[test]
fn outputs_are_named_and_the_undefined_one_is_not_answered() {
    let (_, plain) = dash(
        &[
            "--shape", "1,3,64,64", "--out-shape", "1,3", "--out-shape", "1,3",
            "--out-shape", "1,3", "--out-shape", "1,2", "--out-shape", "1,2",
        ],
        "plain",
    );
    assert_eq!(plain.len(), (3 + 3 + 3 + 2 + 2) * 4, "every output's logits, in order");

    // time_of_day's top-1 is class 0, `daytime`: the condition holds.
    let (text, named) = with_meta("daytime", "fires");
    assert_eq!(named, plain, "the description must not change a single logit bit");
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines[0].starts_with("time_of_day: daytime  (class 0"), "{text}");
    assert!(
        lines[1].starts_with("weather: undefined (time_of_day is daytime)"),
        "{text}"
    );
    assert!(!lines[1].contains("rainy"), "an undefined output must not print an answer");
    assert!(lines[2].starts_with("road_type: parking_lot  (class 2"), "{text}");
    assert!(lines[3].starts_with("surface: wet  (class 1"), "{text}");
    assert!(lines[4].starts_with("traffic: congested  (class 1"), "{text}");
    // Printed as a record; nothing in dash adds it to a logit.
    assert!(lines[5].starts_with("calibration applied in training, already in the weights:"));
    assert!(lines[5].contains(r#""shift":0.5"#), "{text}");

    // Under a condition that does not hold, weather answers like any other output.
    let (text, named) = with_meta("night", "holds-not");
    assert_eq!(named, plain);
    assert!(text.lines().nth(1).unwrap().starts_with("weather: rainy  (class 2"), "{text}");
}

#[test]
fn a_flag_that_contradicts_the_recorded_preprocessing_is_warned_about() {
    let meta = tmp("pre.json");
    let text = description("night").replace(
        r#""input": [1, 3, 64, 64],"#,
        r#""input": [1, 3, 64, 64],
  "preprocessing": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},"#,
    );
    fs::write(&meta, text).unwrap();
    let stderr = |extra: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_dash"))
            .args(["--backend", "tract", "--model"])
            .arg(fixtures_dir().join("mh5.onnx"))
            .arg("--image")
            .arg(fixtures_dir().join("dog.jpg"))
            .arg("--meta")
            .arg(&meta)
            .args(extra)
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stderr).unwrap()
    };
    let wrong = stderr(&["--mean", "0.5,0.5,0.5"]);
    let same = stderr(&["--mean", "0.485,0.456,0.406"]);
    let none = stderr(&[]);
    fs::remove_file(&meta).ok();
    assert!(wrong.contains("warning: --mean [0.5, 0.5, 0.5] overrides"), "{wrong}");
    assert!(!wrong.contains("--std"), "{wrong}");
    assert!(!same.contains("warning"), "{same}");
    assert!(!none.contains("warning"), "{none}");
}

#[test]
fn a_description_that_does_not_fit_the_model_is_refused() {
    let meta = tmp("bad.json");
    fs::write(&meta, description("midnight")).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_dash"))
        .args(["--backend", "tract", "--model"])
        .arg(fixtures_dir().join("mh5.onnx"))
        .arg("--input")
        .arg(fixtures_dir().join("mh5_input_1x3x64x64.f32"))
        .arg("--meta")
        .arg(&meta)
        .output()
        .unwrap();
    fs::remove_file(&meta).ok();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("midnight"));
}
