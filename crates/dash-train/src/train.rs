//! The training run: epochs over the training split, validation after each, every epoch's
//! weights kept and the best epoch's beside them.
//!
//! Burn switches batch norm and dropout by device, not by a call: a module whose tensors
//! are on an autodiff device trains (batch statistics, dropout on), and
//! [`Module::valid`] gives the same module on the plain device, which infers. Validation
//! here runs on `valid()`, so what selects the best epoch is the inference the saved model
//! will give.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

use burn::module::{Module, ParamGroup};
use burn::optim::GradientsParams;
use burn::tensor::{Device, Tensor, TensorData};
use dash_cli::model_json;
use dash_core::ModelSpec;
use dash_models::{MobileNetV3Small, weights};
use serde_json::{Value, json};

use crate::data::{Augment, Batch, Split, batch};
use crate::loss::{self, ClassWeight, IGNORE};
use crate::metrics::{argmax, macro_f1, round4};
use crate::optim::OneCycleAdamW;
use crate::rng::Rng;
use crate::schedule::OneCycle;

/// The recipe. Defaults are the PyTorch recipe this crate reproduces.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub epochs: usize,
    pub batch: usize,
    /// The heads' (and neck's) peak learning rate.
    pub lr: f64,
    /// The backbone's peak learning rate, as a fraction of `lr`.
    pub backbone_lr_scale: f64,
    pub weight_decay: f64,
    pub class_weight: ClassWeight,
    pub pct_start: f64,
    pub dropout: f64,
    pub augment: Augment,
    pub seed: u64,
    /// Stop after this many optimiser steps, keeping the full run's schedule: a timing run.
    pub max_steps: Option<usize>,
}

impl Default for Recipe {
    fn default() -> Self {
        Recipe {
            epochs: 15,
            batch: 128,
            lr: 1e-3,
            backbone_lr_scale: 0.3,
            weight_decay: 0.05,
            class_weight: ClassWeight::Sqrt,
            pct_start: 0.1,
            dropout: 0.2,
            augment: Augment::default(),
            seed: 0,
            max_steps: None,
        }
    }
}

/// Where the weights start.
#[derive(Clone, Debug)]
pub enum Init {
    /// torchvision's ImageNet checkpoint (`.pth`): the backbone and the neck. The heads are
    /// the initialiser's — PyTorch's `nn.Linear` default, which is Burn's.
    Torchvision(PathBuf),
    /// Nothing loaded.
    Scratch,
}

pub struct Run {
    pub recipe: Recipe,
    pub spec: ModelSpec,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub cache: PathBuf,
    pub init: Init,
    pub out: PathBuf,
    /// The model's description as given; written back beside the weights with what the
    /// run adds.
    pub description: Value,
    /// A device with autodiff.
    pub device: Device,
}

/// What a finished run reports.
pub struct Outcome {
    pub best_epoch: Option<usize>,
    pub best_mean_macro_f1: f64,
    pub steps: usize,
}

fn build(run: &Run) -> Result<MobileNetV3Small, String> {
    if run.spec.arch != "mobilenet_v3_small" {
        return Err(format!(
            "dash-train trains mobilenet_v3_small, not {:?}",
            run.spec.arch
        ));
    }
    let mut model = MobileNetV3Small::new(&run.spec.widths(), run.recipe.dropout, &run.device);
    if let Init::Torchvision(path) = &run.init {
        let mut imagenet = MobileNetV3Small::new(&[1000], run.recipe.dropout, &run.device);
        // torchvision's classifier is `Sequential(Linear, Hardswish, Dropout, Linear)`.
        weights::load_pytorch(
            &mut imagenet,
            path,
            None,
            &[
                (r"^classifier\.0\.", "neck.0."),
                (r"^classifier\.3\.", "heads.0."),
            ],
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
        model.features = imagenet.features;
        model.neck = imagenet.neck;
    }
    Ok(model)
}

/// A model as `train` or `calibrate` wrote it: weights, description, preprocessing.
pub struct Trained {
    pub model: MobileNetV3Small,
    pub desc: model_json::Description,
    /// The description as written, every key kept.
    pub description: Value,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// Read `model.json` and `model.safetensors` from `dir`, the model built on `device`.
pub fn load(dir: &Path, device: &Device) -> Result<Trained, String> {
    let path = dir.join("model.json");
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let description: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let desc = model_json::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if desc.spec.arch != "mobilenet_v3_small" {
        return Err(format!(
            "{}: arch {:?} is not mobilenet_v3_small",
            path.display(),
            desc.spec.arch
        ));
    }
    let (mean, std) = (
        desc.mean
            .ok_or(format!("{}: no preprocessing mean", path.display()))?,
        desc.std
            .ok_or(format!("{}: no preprocessing std", path.display()))?,
    );
    let mut model = MobileNetV3Small::new(&desc.spec.widths(), 0.0, device);
    let bytes = fs::read(dir.join("model.safetensors")).map_err(|e| e.to_string())?;
    weights::load_safetensors(&mut model, bytes)?;
    Ok(Trained {
        model,
        desc,
        description,
        mean,
        std,
    })
}

/// Each head's top-1 class on every frame of `split`, `[head][frame]`, in inference mode.
pub fn predict(
    model: &MobileNetV3Small,
    split: &Split,
    mean: [f32; 3],
    std: [f32; 3],
    batch_size: usize,
    device: &Device,
) -> Result<Vec<Vec<usize>>, String> {
    let heads = model.heads.len();
    let mut pred = vec![Vec::with_capacity(split.len); heads];
    let all: Vec<usize> = (0..split.len).collect();
    for chunk in all.chunks(batch_size) {
        let b = batch(split, chunk, None, mean, std)?;
        let x = Tensor::<4>::from_data(TensorData::new(b.input, b.shape), device);
        for (h, logits) in model.forward(x).into_iter().enumerate() {
            let [n, c] = logits.dims();
            let v = logits
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| format!("{e:?}"))?;
            pred[h].extend((0..n).map(|i| argmax(&v[i * c..(i + 1) * c])));
        }
    }
    Ok(pred)
}

/// Each frame of the cache's `split` with each head's top-1 class, a JSON line a frame in
/// the cache's order: `{"image": <the row's image>, "predicted": {<head>: <class>, ...}}`,
/// the heads in the model's order — a model's answers, written down to be scored
/// elsewhere.
pub fn predictions(
    t: &Trained,
    cache: &Path,
    split: &str,
    batch_size: usize,
    device: &Device,
) -> Result<String, String> {
    let s = Split::open(cache, split, t.desc.spec.outputs.len())?;
    let pred = predict(&t.model, &s, t.mean, t.std, batch_size, device)?;
    let path = cache.join(format!("{split}.jsonl"));
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    // The rows `Split::open` read, blank lines skipped as it skips them.
    let images = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            let row: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
            row["image"].as_str().map(String::from).ok_or(format!(
                "{} row {}: no image",
                path.display(),
                i + 1
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(written(&t.desc.spec, &images, &pred))
}

/// What [`predictions`] writes for frames `images`, given `pred[head][frame]`.
fn written(spec: &ModelSpec, images: &[String], pred: &[Vec<usize>]) -> String {
    let quoted = |s: &str| serde_json::to_string(s).expect("a string");
    let mut out = String::new();
    for (i, image) in images.iter().enumerate() {
        let classes: Vec<String> = spec
            .outputs
            .iter()
            .zip(pred)
            .map(|(o, p)| format!("{}: {}", quoted(&o.name), quoted(&o.classes[p[i]])))
            .collect();
        out += &format!(
            "{{\"image\": {}, \"predicted\": {{{}}}}}\n",
            quoted(image),
            classes.join(", ")
        );
    }
    out
}

/// Macro-F1 per head on `split`, rounded as the recipe rounds it, and their mean.
pub fn score(spec: &ModelSpec, split: &Split, pred: &[Vec<usize>]) -> (Value, f64) {
    let mut report = serde_json::Map::new();
    let mut sum = 0.0;
    for (h, out) in spec.outputs.iter().enumerate() {
        let (p, t): (Vec<usize>, Vec<usize>) = split
            .targets
            .iter()
            .zip(&pred[h])
            .filter(|(t, _)| t[h] != IGNORE)
            .map(|(t, &p)| (p, t[h] as usize))
            .unzip();
        let (f1, recall) = macro_f1(&p, &t, out.classes.len());
        let f1 = round4(f1);
        sum += f1;
        let recall: serde_json::Map<String, Value> = out
            .classes
            .iter()
            .zip(recall)
            .map(|(c, r)| (c.clone(), r.map(round4).map_or(Value::Null, Value::from)))
            .collect();
        report.insert(
            out.name.clone(),
            json!({"macro_f1": f1, "n": t.len(), "recall": recall}),
        );
    }
    let mean = round4(sum / spec.outputs.len() as f64);
    report.insert("mean_macro_f1".into(), json!(mean));
    (Value::Object(report), mean)
}

/// `model` after `epoch` as `model.safetensors` and `model.json` hold it: the weights, and
/// the description with what the run adds.
fn snapshot(
    run: &Run,
    model: &MobileNetV3Small,
    epoch: usize,
    report: &Value,
    steps: usize,
) -> Result<(Vec<u8>, String), String> {
    let bytes = weights::to_safetensors(&model.valid())?;
    let mut d = run.description.clone();
    let r = &run.recipe;
    d["arch"] = json!(run.spec.arch);
    d["calibration_applied"] = Value::Null;
    d["trained"] = json!({
        "by": "dash-train",
        "epoch": epoch,
        "steps": steps,
        "seed": r.seed,
        "recipe": {
            "epochs": r.epochs, "batch": r.batch, "lr": r.lr,
            "backbone_lr_scale": r.backbone_lr_scale, "weight_decay": r.weight_decay,
            "class_weight": format!("{:?}", r.class_weight).to_lowercase(),
            "pct_start": r.pct_start, "dropout": r.dropout,
            "crop_scale": [r.augment.scale.0, r.augment.scale.1],
            "crop_ratio": [r.augment.ratio.0, r.augment.ratio.1],
            "flip": r.augment.flip,
            "init": match &run.init {
                Init::Torchvision(p) => p.file_name().map_or(Value::Null, |n| json!(n.to_string_lossy())),
                Init::Scratch => Value::Null,
            },
        },
    });
    d["val_mean_macro_f1"] = report["mean_macro_f1"].clone();
    d["val"] = report.clone();
    let text = serde_json::to_string_pretty(&d).map_err(|e| e.to_string())?;
    Ok((bytes, text + "\n"))
}

fn save(dir: &Path, (bytes, text): &(Vec<u8>, String)) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    fs::write(dir.join("model.safetensors"), bytes).map_err(|e| e.to_string())?;
    fs::write(dir.join("model.json"), text).map_err(|e| e.to_string())
}

/// Where epoch `epoch` (counted from 0, as `metrics.jsonl` counts it) is kept.
pub fn epoch_dir(out: &Path, epoch: usize) -> PathBuf {
    out.join("epochs").join(format!("{epoch:02}"))
}

/// Train, validating after each epoch. Each epoch's weights and description are written
/// to [`epoch_dir`] as `model.safetensors` and `model.json`, so that runs can be compared
/// at a fixed epoch as well as at the one each selected; the best epoch's are written to
/// `run.out` as well, and every epoch's report is appended to `metrics.jsonl`.
///
/// The seed fixes the weights, whatever the number of threads. On Flex the initial weights
/// and the dropout masks come from one generator per process, which this seeds, so two runs
/// at once in one process interleave their draws and neither is reproducible.
pub fn train(run: &Run) -> Result<Outcome, String> {
    let r = &run.recipe;
    let heads = run.spec.outputs.len();
    fs::create_dir_all(&run.out).map_err(|e| e.to_string())?;
    let train_split = Split::open(&run.cache, "train", heads)?;
    let val_split = Split::open(&run.cache, "val", heads)?;
    let weights: Vec<Vec<f32>> = run
        .spec
        .outputs
        .iter()
        .enumerate()
        .map(|(h, o)| loss::class_weights(&train_split.column(h), o.classes.len(), r.class_weight))
        .collect();

    run.device.seed(r.seed);
    let mut model = build(run)?;
    let per_epoch = train_split.len / r.batch;
    if per_epoch == 0 {
        return Err(format!(
            "{} training frames make no batch of {}",
            train_split.len, r.batch
        ));
    }
    let total = r.epochs * per_epoch;
    // The heads and the neck on the recipe's peak; the backbone on its own, lower one.
    let mut optim = OneCycleAdamW::new(
        OneCycle::new(r.lr, total, r.pct_start),
        ParamGroup::ids_from_module(model.features.clone()),
        OneCycle::new(r.lr * r.backbone_lr_scale, total, r.pct_start),
        r.weight_decay,
    );
    let infer = run.device.clone().without_autodiff();

    println!(
        "train {}  val {}  steps {total} ({per_epoch} an epoch)  class weights {}",
        train_split.len,
        val_split.len,
        run.spec
            .outputs
            .iter()
            .zip(&weights)
            .map(|(o, w)| format!(
                "{}={:?}",
                o.name,
                w.iter().map(|x| round4(f64::from(*x))).collect::<Vec<_>>()
            ))
            .collect::<Vec<_>>()
            .join("  ")
    );

    let (tx, rx) = mpsc::sync_channel::<Result<Batch, String>>(2);
    let stop_at = r.max_steps.unwrap_or(total).min(total);
    std::thread::scope(|s| {
        let train_split = &train_split;
        // The producer draws every shuffle and crop from one generator, so a seed fixes
        // them whatever the consumer's pace.
        s.spawn(move || {
            let mut rng = Rng::new(r.seed);
            let mut order: Vec<usize> = (0..train_split.len).collect();
            for _ in 0..r.epochs {
                rng.shuffle(&mut order);
                for chunk in order.chunks_exact(r.batch) {
                    let b = batch(
                        train_split,
                        chunk,
                        Some((&r.augment, &mut rng)),
                        run.mean,
                        run.std,
                    );
                    if tx.send(b).is_err() {
                        return;
                    }
                }
            }
        });

        let rx = rx;
        let mut step = 0;
        let mut best = f64::NEG_INFINITY;
        let mut best_epoch = None;
        let mut metrics = OpenOptions::new()
            .create(true)
            .append(true)
            .open(run.out.join("metrics.jsonl"))
            .map_err(|e| e.to_string())?;
        for epoch in 0..r.epochs {
            if step >= stop_at {
                break;
            }
            let t0 = Instant::now();
            let (mut total_loss, mut count, mut waited) = (0.0f64, 0usize, 0.0f64);
            for _ in 0..per_epoch {
                if step >= stop_at {
                    break;
                }
                let tw = Instant::now();
                let b = rx
                    .recv()
                    .map_err(|_| "the batch producer stopped".to_string())??;
                waited += tw.elapsed().as_secs_f64();
                let ts = Instant::now();
                let x = Tensor::<4>::from_data(TensorData::new(b.input, b.shape), &run.device);
                let loss = loss::loss(model.forward(x), &b.targets, &weights);
                let value = loss
                    .clone()
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| format!("{e:?}"))?[0];
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optim.step(model, grads);
                total_loss += f64::from(value);
                count += 1;
                step += 1;
                if step <= 5 {
                    println!(
                        "step {step}: loss {value:.4}  {:.2} s",
                        ts.elapsed().as_secs_f64()
                    );
                }
            }
            let train_seconds = t0.elapsed().as_secs_f64();
            let valid = model.valid();
            let pred = predict(&valid, &val_split, run.mean, run.std, r.batch, &infer)?;
            let (mut report, mean) = score(&run.spec, &val_split, &pred);
            report["epoch"] = json!(epoch);
            report["train_loss"] = json!(round4(total_loss / count.max(1) as f64));
            report["seconds"] = json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
            report["seconds_per_step"] = json!(round4(train_seconds / count.max(1) as f64));
            report["data_wait_per_step"] = json!(round4(waited / count.max(1) as f64));
            writeln!(metrics, "{report}").map_err(|e| e.to_string())?;
            println!(
                "epoch {epoch}  loss {}  mean macro-F1 {mean}  {}  ({} s, {} s a step)",
                report["train_loss"],
                run.spec
                    .outputs
                    .iter()
                    .map(|o| format!("{} {}", o.name, report[&o.name]["macro_f1"]))
                    .collect::<Vec<_>>()
                    .join("  "),
                report["seconds"],
                report["seconds_per_step"],
            );
            let snap = snapshot(run, &model, epoch, &report, step)?;
            save(&epoch_dir(&run.out, epoch), &snap)?;
            if mean > best {
                best = mean;
                best_epoch = Some(epoch);
                save(&run.out, &snap)?;
            }
        }
        drop(rx);
        Ok(Outcome {
            best_epoch,
            best_mean_macro_f1: best,
            steps: step,
        })
    })
}

#[cfg(test)]
mod tests {
    use dash_core::OutputSpec;

    use super::*;

    #[test]
    fn each_frame_is_written_with_its_own_classes_in_the_models_head_order() {
        let head = |name: &str, classes: &[&str]| OutputSpec {
            name: name.into(),
            classes: classes.iter().map(|c| c.to_string()).collect(),
        };
        let spec = ModelSpec {
            arch: "mobilenet_v3_small".into(),
            outputs: vec![head("zone", &["x", "y", "z"]), head("angle", &["p", "q"])],
            undefined_when: Vec::new(),
        };
        let images = [
            "a.jpg".to_string(),
            "b \"1\".jpg".to_string(),
            "c.jpg".to_string(),
        ];
        let text = written(&spec, &images, &[vec![2, 0, 1], vec![0, 1, 1]]);
        assert_eq!(
            text,
            "{\"image\": \"a.jpg\", \"predicted\": {\"zone\": \"z\", \"angle\": \"p\"}}\n\
             {\"image\": \"b \\\"1\\\".jpg\", \"predicted\": {\"zone\": \"x\", \"angle\": \"q\"}}\n\
             {\"image\": \"c.jpg\", \"predicted\": {\"zone\": \"y\", \"angle\": \"q\"}}\n"
        );
    }
}
