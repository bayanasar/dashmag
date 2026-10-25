//! `dash-train` — decode a training set, train a multi-head classifier, calibrate it, and
//! write down what it predicts.
//!
//! ```text
//! dash-train cache     --trainset rows.jsonl --root DIR --meta model.json --out CACHE
//! dash-train train     --cache CACHE --meta model.json --init imagenet.pth --out RUN [--seed N]
//! dash-train calibrate --model RUN --cache CACHE --head H --class C --out RUN/calibrated
//! dash-train evaluate  --model RUN --cache CACHE [--split val]
//! dash-train predict   --model RUN --cache CACHE --split NAME [--split ...] --out FILE
//! ```
//!
//! `--meta` is the model's description (`dash --meta` reads the same file): its heads and
//! classes, `arch`, and the preprocessing mean and std. A trained model's `model.json` is
//! that description with what the run adds.

use std::path::PathBuf;
use std::process::ExitCode;
use std::{env, fs};

use burn::tensor::Device;
use dash_cli::model_json;
use dash_train::calibrate;
use dash_train::data::Split;
use dash_train::train::{self, Init, Recipe, Run};

fn usage() -> ! {
    eprintln!(
        "usage:
  dash-train cache --trainset FILE --root DIR --meta model.json --out DIR
                   [--size H,W] [--limit N] [--threads N]
  dash-train train --cache DIR --meta model.json --out DIR [--init torchvision.pth]
                   [--seed N] [--epochs N] [--batch N] [--lr X] [--backbone-lr-scale X]
                   [--weight-decay X] [--class-weight none|sqrt|inv] [--pct-start X]
                   [--max-steps N] [--device flex|cuda:N|rocm:N|metal]
  dash-train calibrate --model DIR --cache DIR --head NAME --class NAME --out DIR
                   [--grid LO:HI:STEP] [--device ...]
  dash-train evaluate --model DIR --cache DIR [--split val] [--device ...]
  dash-train predict --model DIR --cache DIR --split NAME [--split ...] --out FILE
                   [--batch N] [--device ...]"
    );
    std::process::exit(2)
}

fn device(name: &str) -> Result<Device, String> {
    let (kind, index) = match name.split_once(':') {
        Some((k, i)) => (
            k,
            i.parse::<usize>()
                .map_err(|_| format!("device index in {name:?}"))?,
        ),
        None => (name, 0),
    };
    let _ = index;
    match kind {
        #[cfg(feature = "flex")]
        "flex" => Ok(Device::flex()),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::cuda(index)),
        #[cfg(feature = "rocm")]
        "rocm" => Ok(Device::rocm(index)),
        #[cfg(feature = "metal")]
        "metal" => Ok(Device::metal(burn::tensor::DeviceKind::DefaultDevice)),
        _ => Err(format!("device {name:?} is not built into this binary")),
    }
}

struct Args(std::iter::Peekable<std::vec::IntoIter<String>>);

impl Args {
    fn value(&mut self, flag: &str) -> Result<String, String> {
        self.0.next().ok_or(format!("{flag} needs a value"))
    }
    fn parse<T: std::str::FromStr>(&mut self, flag: &str) -> Result<T, String> {
        let v = self.value(flag)?;
        v.parse().map_err(|_| format!("{flag}: cannot read {v:?}"))
    }
}

fn required<T>(v: Option<T>, flag: &str) -> Result<T, String> {
    v.ok_or(format!("{flag} is required"))
}

fn description(path: &str) -> Result<(String, model_json::Description), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let d = model_json::parse(&text).map_err(|e| format!("{path}: {e}"))?;
    Ok((text, d))
}

fn cache(mut a: Args) -> Result<(), String> {
    let (mut trainset, mut root, mut meta, mut out) = (None, None, None, None);
    let (mut size, mut limit, mut threads) = (None, None, None);
    while let Some(f) = a.0.next() {
        match f.as_str() {
            "--trainset" => trainset = Some(a.value(&f)?),
            "--root" => root = Some(a.value(&f)?),
            "--meta" => meta = Some(a.value(&f)?),
            "--out" => out = Some(a.value(&f)?),
            "--size" => {
                let v = a.value(&f)?;
                let (h, w) = v.split_once(',').ok_or("--size is H,W")?;
                size = Some((
                    h.parse().map_err(|_| "--size H")?,
                    w.parse().map_err(|_| "--size W")?,
                ));
            }
            "--limit" => limit = Some(a.parse(&f)?),
            "--threads" => threads = Some(a.parse(&f)?),
            _ => usage(),
        }
    }
    let (_, d) = description(&required(meta, "--meta")?)?;
    let size = match (size, d.input.as_deref()) {
        (Some(s), _) => s,
        (None, Some(&[_, _, h, w])) => (h, w),
        _ => return Err("no --size, and the description has no NCHW input".into()),
    };
    let threads =
        threads.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    dash_train::cache::build(
        &PathBuf::from(required(trainset, "--trainset")?),
        &PathBuf::from(root.unwrap_or_else(|| ".".into())),
        &d.spec,
        &PathBuf::from(required(out, "--out")?),
        size,
        limit,
        threads,
    )?;
    Ok(())
}

fn train_cmd(mut a: Args) -> Result<(), String> {
    let mut recipe = Recipe::default();
    let (mut cache, mut meta, mut out, mut init) = (None, None, None, None);
    let mut dev = "flex".to_string();
    while let Some(f) = a.0.next() {
        match f.as_str() {
            "--cache" => cache = Some(a.value(&f)?),
            "--meta" => meta = Some(a.value(&f)?),
            "--out" => out = Some(a.value(&f)?),
            "--init" => init = Some(a.value(&f)?),
            "--device" => dev = a.value(&f)?,
            "--seed" => recipe.seed = a.parse(&f)?,
            "--epochs" => recipe.epochs = a.parse(&f)?,
            "--batch" => recipe.batch = a.parse(&f)?,
            "--lr" => recipe.lr = a.parse(&f)?,
            "--backbone-lr-scale" => recipe.backbone_lr_scale = a.parse(&f)?,
            "--weight-decay" => recipe.weight_decay = a.parse(&f)?,
            "--class-weight" => recipe.class_weight = a.value(&f)?.parse()?,
            "--pct-start" => recipe.pct_start = a.parse(&f)?,
            "--max-steps" => recipe.max_steps = Some(a.parse(&f)?),
            _ => usage(),
        }
    }
    let meta = required(meta, "--meta")?;
    let (text, d) = description(&meta)?;
    let run = Run {
        mean: d.mean.ok_or(format!("{meta}: no preprocessing mean"))?,
        std: d.std.ok_or(format!("{meta}: no preprocessing std"))?,
        spec: d.spec,
        recipe,
        cache: PathBuf::from(required(cache, "--cache")?),
        init: init.map_or(Init::Scratch, |p| Init::Torchvision(PathBuf::from(p))),
        out: PathBuf::from(required(out, "--out")?),
        description: serde_json::from_str(&text).map_err(|e| e.to_string())?,
        device: device(&dev)?.autodiff(),
    };
    let o = train::train(&run)?;
    match o.best_epoch {
        Some(e) => println!(
            "best epoch {e}: mean macro-F1 {} ({} steps) -> {}",
            o.best_mean_macro_f1,
            o.steps,
            run.out.display()
        ),
        None => println!("no epoch finished ({} steps)", o.steps),
    }
    Ok(())
}

fn calibrate_cmd(mut a: Args) -> Result<(), String> {
    let (mut model, mut cache, mut head, mut class, mut out) = (None, None, None, None, None);
    let mut grid = "0:4:0.25".to_string();
    let mut dev = "flex".to_string();
    while let Some(f) = a.0.next() {
        match f.as_str() {
            "--model" => model = Some(a.value(&f)?),
            "--cache" => cache = Some(a.value(&f)?),
            "--head" => head = Some(a.value(&f)?),
            "--class" => class = Some(a.value(&f)?),
            "--out" => out = Some(a.value(&f)?),
            "--grid" => grid = a.value(&f)?,
            "--device" => dev = a.value(&f)?,
            _ => usage(),
        }
    }
    let g: Vec<f64> = grid
        .split(':')
        .map(|v| {
            v.parse()
                .map_err(|_| format!("--grid {grid:?} is LO:HI:STEP"))
        })
        .collect::<Result<_, _>>()?;
    let [lo, hi, step] = g[..] else {
        return Err(format!("--grid {grid:?} is LO:HI:STEP"));
    };
    calibrate::calibrate(
        &PathBuf::from(required(model, "--model")?),
        &PathBuf::from(required(cache, "--cache")?),
        &required(head, "--head")?,
        &required(class, "--class")?,
        &calibrate::grid(lo, hi, step),
        &PathBuf::from(required(out, "--out")?),
        &device(&dev)?,
    )?;
    Ok(())
}

fn evaluate_cmd(mut a: Args) -> Result<(), String> {
    let (mut model, mut cache) = (None, None);
    let mut split = "val".to_string();
    let mut dev = "flex".to_string();
    while let Some(f) = a.0.next() {
        match f.as_str() {
            "--model" => model = Some(a.value(&f)?),
            "--cache" => cache = Some(a.value(&f)?),
            "--split" => split = a.value(&f)?,
            "--device" => dev = a.value(&f)?,
            _ => usage(),
        }
    }
    let device = device(&dev)?;
    let t = train::load(&PathBuf::from(required(model, "--model")?), &device)?;
    let s = Split::open(
        &PathBuf::from(required(cache, "--cache")?),
        &split,
        t.desc.spec.outputs.len(),
    )?;
    let pred = train::predict(&t.model, &s, t.mean, t.std, 128, &device)?;
    let (report, _) = train::score(&t.desc.spec, &s, &pred);
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn predict_cmd(mut a: Args) -> Result<(), String> {
    let (mut model, mut cache, mut out) = (None, None, None);
    let mut splits = Vec::new();
    let mut batch = 128;
    let mut dev = "flex".to_string();
    while let Some(f) = a.0.next() {
        match f.as_str() {
            "--model" => model = Some(a.value(&f)?),
            "--cache" => cache = Some(a.value(&f)?),
            "--split" => splits.push(a.value(&f)?),
            "--out" => out = Some(a.value(&f)?),
            "--batch" => batch = a.parse(&f)?,
            "--device" => dev = a.value(&f)?,
            _ => usage(),
        }
    }
    if splits.is_empty() {
        return Err("--split is required".into());
    }
    let out = required(out, "--out")?;
    let device = device(&dev)?;
    let t = train::load(&PathBuf::from(required(model, "--model")?), &device)?;
    let cache = PathBuf::from(required(cache, "--cache")?);
    let mut text = String::new();
    for split in &splits {
        text += &train::predictions(&t, &cache, split, batch, &device)?;
    }
    fs::write(&out, text).map_err(|e| format!("{out}: {e}"))
}

fn main() -> ExitCode {
    let mut args = env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .into_iter()
        .peekable();
    let Some(cmd) = args.next() else { usage() };
    let a = Args(args);
    let result = match cmd.as_str() {
        "cache" => cache(a),
        "train" => train_cmd(a),
        "calibrate" => calibrate_cmd(a),
        "evaluate" => evaluate_cmd(a),
        "predict" => predict_cmd(a),
        _ => usage(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dash-train: {e}");
            ExitCode::FAILURE
        }
    }
}
