//! Move one head's decision boundary for one class, chosen on validation, folded into the
//! head's bias.
//!
//! A head trained on a rare class can rank it well and still seldom pick it: the boundary
//! sits where the training prior put it. A constant added to the class's logit moves the
//! boundary without changing the ranking. The constant is the point of a grid with the
//! best macro-F1 for that head on the validation split — the first such point, on a tie —
//! and it is added to the head's bias, so the model carries it with no change of shape and
//! nothing for a runtime to apply. The description records it as `calibration_applied`.
//!
//! The grid and the choice are `dash_mag::calibrate`'s; reading the run, scoring the cache and
//! folding the shift into this model's bias are the loop's, and stay with it.

use std::fs;
use std::path::Path;

use burn::tensor::{Device, Tensor, TensorData};
use dash_mag::calibrate::{Point, choose};
use dash_mag::loss::IGNORE;
use dash_models::weights;
use serde_json::json;

use crate::data::{Split, batch};
use crate::train::{Trained, load};

/// Calibrate the model in `model_dir` on `cache`'s validation split and write the result to
/// `out` (weights and description). Refuses a model already calibrated: shifts add.
pub fn calibrate(
    model_dir: &Path,
    cache: &Path,
    head: &str,
    class: &str,
    shifts: &[f64],
    out: &Path,
    device: &Device,
) -> Result<Point, String> {
    let Trained {
        mut model,
        desc,
        mut description,
        mean,
        std,
    } = load(model_dir, device)?;
    if !description["calibration_applied"].is_null() {
        return Err(format!("{} is calibrated already", model_dir.display()));
    }
    let k = desc
        .spec
        .outputs
        .iter()
        .position(|o| o.name == head)
        .ok_or(format!("no head {head:?}"))?;
    let c = desc.spec.outputs[k]
        .classes
        .iter()
        .position(|x| x == class)
        .ok_or(format!("head {head:?} has no class {class:?}"))?;

    let val = Split::open(cache, "val", desc.spec.outputs.len())?;
    let keep: Vec<usize> = (0..val.len)
        .filter(|&i| val.targets[i][k] != IGNORE)
        .collect();
    let mut logits = Vec::with_capacity(keep.len());
    for chunk in keep.chunks(256) {
        let b = batch(&val, chunk, None, mean, std)?;
        let x = Tensor::<4>::from_data(TensorData::new(b.input, b.shape), device);
        let out = model.forward(x).swap_remove(k);
        let [n, w] = out.dims();
        let v = out
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| format!("{e:?}"))?;
        logits.extend((0..n).map(|i| v[i * w..(i + 1) * w].to_vec()));
    }
    let truth: Vec<usize> = keep.iter().map(|&i| val.targets[i][k] as usize).collect();
    let (best, points) = choose(&logits, &truth, c, shifts);
    for p in &points {
        println!(
            "shift {:5.2}  {head} macro-F1 {:.4}  {class} recall {:.4}",
            p.shift, p.macro_f1, p.recall
        );
    }
    println!(
        "chosen on validation: shift {} (macro-F1 {} -> {})",
        best.shift, points[0].macro_f1, best.macro_f1
    );

    let width = desc.spec.outputs[k].classes.len();
    let mut delta = vec![0f32; width];
    delta[c] = best.shift as f32;
    let delta = Tensor::<1>::from_data(TensorData::new(delta, [width]), device);
    let bias = model.heads[k].bias.take().ok_or("the head has no bias")?;
    model.heads[k].bias = Some(bias.map(|b| b + delta));

    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    fs::write(
        out.join("model.safetensors"),
        weights::to_safetensors(&model)?,
    )
    .map_err(|e| e.to_string())?;
    description["calibration_applied"] = json!({
        "head": head,
        "class": class,
        "shift": best.shift,
        "folded_into": format!("heads.{k}.bias"),
        "chosen_on": "validation",
        "val_macro_f1": [points[0].macro_f1, best.macro_f1],
        "grid": points.iter().map(|p| [p.shift, p.macro_f1, p.recall]).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&description).map_err(|e| e.to_string())?;
    fs::write(out.join("model.json"), text + "\n").map_err(|e| e.to_string())?;
    Ok(best)
}
