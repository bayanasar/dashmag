//! `model.json`: what a model's outputs mean, written by whoever trained it.
//!
//! ```json
//! {
//!   "arch": "mobilenet_v3_small",
//!   "heads": ["time_of_day", "weather"],
//!   "classes": {"time_of_day": ["daytime", "night", "dawn_dusk"],
//!               "weather": ["clear", "overcast", "rainy"]},
//!   "undefined_when": {"weather": {"time_of_day": "night"}},
//!   "input": [1, 3, 224, 224],
//!   "preprocessing": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
//!   "calibration_applied": {"head": "weather", "class": "rainy", "shift": 0.5,
//!                           "folded_into": "heads.1.bias"}
//! }
//! ```
//!
//! `heads` gives the output order and `classes` each output's classes in logit order.
//! `undefined_when` maps an output to the conditions under which it means nothing: any one
//! of them holding, the output is undefined. `arch` is needed only by a backend that
//! builds the model from code. `input` and `preprocessing` are defaults for `--shape`,
//! `--mean` and `--std`. `calibration_applied` is a record of a shift already folded into
//! the weights: it is printed, never applied. Other keys are ignored.

use dash_core::{ModelSpec, OutputSpec, UndefinedWhen};
use serde_json::Value;

#[derive(Debug)]
pub struct Description {
    pub spec: ModelSpec,
    pub input: Option<Vec<usize>>,
    pub mean: Option<[f32; 3]>,
    pub std: Option<[f32; 3]>,
    /// As written, compact.
    pub calibration_applied: Option<String>,
}

fn strings(v: &Value, what: &str) -> Result<Vec<String>, String> {
    v.as_array()
        .ok_or(format!("{what} is not a list"))?
        .iter()
        .map(|s| s.as_str().map(str::to_string).ok_or(format!("{what} holds a non-string")))
        .collect()
}

fn rgb(v: &Value, what: &str) -> Result<[f32; 3], String> {
    let vals: Vec<f32> = v
        .as_array()
        .ok_or(format!("{what} is not a list"))?
        .iter()
        .map(|x| x.as_f64().map(|f| f as f32).ok_or(format!("{what} holds a non-number")))
        .collect::<Result<_, _>>()?;
    vals.try_into()
        .map_err(|v: Vec<f32>| format!("{what} needs three values, has {}", v.len()))
}

pub fn parse(text: &str) -> Result<Description, String> {
    let j: Value = serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))?;
    let heads = strings(j.get("heads").ok_or("no \"heads\"")?, "heads")?;
    let classes = j.get("classes").ok_or("no \"classes\"")?;
    let outputs = heads
        .iter()
        .map(|h| {
            let c = classes.get(h).ok_or(format!("no classes for head {h:?}"))?;
            Ok(OutputSpec {
                name: h.clone(),
                classes: strings(c, &format!("classes of {h:?}"))?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut undefined_when = Vec::new();
    if let Some(u) = j.get("undefined_when").filter(|u| !u.is_null()) {
        for (output, conds) in u.as_object().ok_or("undefined_when is not an object")? {
            for (when, is) in conds
                .as_object()
                .ok_or(format!("undefined_when of {output:?} is not an object"))?
            {
                let is = is
                    .as_str()
                    .ok_or(format!("undefined_when of {output:?}: {when:?} is not a class name"))?;
                undefined_when.push(UndefinedWhen {
                    output: output.clone(),
                    when: when.clone(),
                    is: is.to_string(),
                });
            }
        }
    }
    let spec = ModelSpec {
        arch: j.get("arch").and_then(Value::as_str).unwrap_or_default().to_string(),
        outputs,
        undefined_when,
    };
    // A condition naming an output or a class the model lacks would never fire; find it
    // now, not on the frame where it should have.
    spec.undefined(&vec![0; spec.outputs.len()])
        .map_err(|e| e.to_string())?;

    let input = match j.get("input").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(
            v.as_array()
                .ok_or("input is not a list")?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize).ok_or("input holds a non-integer"))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };
    let pre = j.get("preprocessing");
    let field = |k: &str| pre.and_then(|p| p.get(k)).filter(|v| !v.is_null());
    Ok(Description {
        spec,
        input,
        mean: field("mean").map(|v| rgb(v, "preprocessing.mean")).transpose()?,
        std: field("std").map(|v| rgb(v, "preprocessing.std")).transpose()?,
        calibration_applied: j
            .get("calibration_applied")
            .filter(|v| !v.is_null())
            .map(Value::to_string),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const V1_SHAPE: &str = r#"{
        "heads": ["time_of_day", "weather"],
        "classes": {"time_of_day": ["daytime", "night", "dawn_dusk"],
                    "weather": ["clear", "overcast", "rainy"]},
        "undefined_when": {"weather": {"time_of_day": "night"}},
        "input": [1, 3, 224, 224],
        "entry_point": "ignored",
        "preprocessing": {"resize": "described, not read", "mean": [0.485, 0.456, 0.406],
                          "std": [0.229, 0.224, 0.225]},
        "calibration_applied": null
    }"#;

    #[test]
    fn reads_the_trainers_description() {
        let d = parse(V1_SHAPE).unwrap();
        assert_eq!(d.spec.arch, "");
        assert_eq!(d.spec.widths(), [3, 3]);
        assert_eq!(d.spec.undefined(&[1, 0]).unwrap(), [false, true]);
        assert_eq!(d.input.as_deref(), Some(&[1, 3, 224, 224][..]));
        assert_eq!(d.std, Some([0.229, 0.224, 0.225]));
        assert!(d.calibration_applied.is_none());
    }

    #[test]
    fn a_condition_that_could_never_fire_is_an_error() {
        let bad = V1_SHAPE.replace(r#""time_of_day": "night""#, r#""time_of_day": "midnight""#);
        assert!(parse(&bad).unwrap_err().contains("midnight"));
        let bad = V1_SHAPE.replace(r#""weather": ["clear""#, r#""sky": ["clear""#);
        assert!(parse(&bad).unwrap_err().contains("weather"));
    }
}
