//! `dash-import` — a PyTorch checkpoint in, `model.safetensors` out, for a model that
//! `dash-models` builds.
//!
//!   dash-import --pth <checkpoint> --meta <model.json> --out <model.safetensors>
//!               [--key <entry>] [--remap <regex>=<replacement>]...
//!
//! The architecture and its outputs come from `model.json` (its `arch`, `heads` and
//! `classes`). The checkpoint is read in Rust; `--key` picks the entry holding the weights
//! when the file is a training checkpoint (`state_dict`), `--remap` renames tensors whose
//! names differ from the module's. Loading is strict — every parameter filled, every
//! tensor placed — so a checkpoint for a different model is refused, not half-imported.
//! The output is safetensors in PyTorch's convention: `dash --backend burn` runs it, and
//! PyTorch can still read it.

use burn::tensor::Device;
use dash_cli::model_json;
use dash_models::Model;
use std::fs;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage: dash-import --pth <checkpoint> --meta <model.json> --out <model.safetensors> \
         [--key <entry>] [--remap <regex>=<replacement>]..."
    );
    std::process::exit(2);
}

fn run() -> Result<(), String> {
    let (mut pth, mut meta, mut out, mut key) = (None, None, None, None);
    let mut remap: Vec<(String, String)> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pth" => pth = args.next(),
            "--meta" => meta = args.next(),
            "--out" => out = args.next(),
            "--key" => key = args.next(),
            "--remap" => {
                let r = args.next().ok_or("--remap needs <regex>=<replacement>")?;
                let (from, to) = r
                    .split_once('=')
                    .ok_or(format!("--remap {r:?} is not <regex>=<replacement>"))?;
                remap.push((from.to_string(), to.to_string()));
            }
            "-h" | "--help" => usage(),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let (Some(pth), Some(meta), Some(out)) = (pth, meta, out) else {
        usage()
    };
    let text = fs::read_to_string(&meta).map_err(|e| format!("read meta {meta}: {e}"))?;
    let desc = model_json::parse(&text).map_err(|e| format!("meta {meta}: {e}"))?;
    if desc.spec.arch.is_empty() {
        return Err(format!(
            "meta {meta}: no \"arch\"; dash-import needs to know what to build"
        ));
    }
    let mut model = Model::build(&desc.spec, &Device::flex())?;
    let pairs: Vec<(&str, &str)> = remap
        .iter()
        .map(|(f, t)| (f.as_str(), t.as_str()))
        .collect();
    model.load_pytorch(&pth, key.as_deref(), &pairs)?;
    let bytes = model.to_safetensors()?;
    fs::write(&out, &bytes).map_err(|e| format!("write {out}: {e}"))?;
    let outputs: Vec<String> = desc
        .spec
        .outputs
        .iter()
        .map(|o| format!("{} ({})", o.name, o.classes.len()))
        .collect();
    eprintln!(
        "{pth} -> {out}: {} with outputs {}, {} bytes",
        desc.spec.arch,
        outputs.join(", "),
        bytes.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
