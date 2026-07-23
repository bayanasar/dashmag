//! `dash` — backend-agnostic Dashmag inference runner.
//!
//! Builds a [`Registry`] of whatever backends are compiled in, tags the input
//! artifact with its `BackendKind`, and lets the registry *dispatch* to an
//! admissible backend. Same path the acceptance tests drive.

use dash_core::{
    AccelClass, Artifact, ArtifactMeta, BackendKind, HostTensor, Registry, TargetProfile, Tensor,
};
use std::fs;
use std::process::ExitCode;

/// Map an IREE-style device string to the accelerator class the registry ranks by.
fn accel_for_device(device: &str) -> AccelClass {
    match device {
        "cuda" => AccelClass::Cuda,
        "vulkan" => AccelClass::Vulkan,
        "metal" => AccelClass::Metal,
        _ => AccelClass::Cpu, // local-task / local-sync / anything else
    }
}

#[allow(unused_variables)]
fn build_registry(device: &str, accel: AccelClass) -> Registry {
    #[allow(unused_mut)]
    let mut reg = Registry::new();
    #[cfg(feature = "tract")]
    reg.register(Box::new(dash_tract::TractRuntime::new()));
    #[cfg(feature = "iree")]
    reg.register(Box::new(dash_iree::IreeRuntime::new(device.to_string(), accel)));
    #[cfg(feature = "tensorrt")]
    reg.register(Box::new(dash_tensorrt::TensorRtRuntime::new()));
    #[cfg(feature = "edgetpu")]
    reg.register(Box::new(dash_edgetpu::EdgeTpuRuntime::new()));
    reg
}

fn parse_backend(s: &str) -> Option<BackendKind> {
    match s {
        "tract" => Some(BackendKind::Tract),
        "iree" => Some(BackendKind::Iree),
        "edgetpu" => Some(BackendKind::EdgeTpu),
        "tensorrt" => Some(BackendKind::TensorRt),
        _ => None,
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: dash --backend <tract|iree|edgetpu> --model <artifact> --input <f32blob> \
         [--labels <txt>] [--shape N,C,H,W] [--device <local-task|cuda|vulkan>]"
    );
    std::process::exit(2);
}

fn main() -> ExitCode {
    let (mut backend, mut model, mut input, mut labels) = (None, None, None, None);
    let mut shape = vec![1usize, 3, 224, 224];
    let mut device = String::from("local-task");

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--backend" => backend = args.next(),
            "--model" => model = args.next(),
            "--input" => input = args.next(),
            "--labels" => labels = args.next(),
            "--device" => device = args.next().unwrap_or(device),
            "--shape" => {
                shape = args
                    .next()
                    .unwrap_or_default()
                    .split(',')
                    .filter_map(|x| x.parse().ok())
                    .collect()
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown arg: {other}");
                usage();
            }
        }
    }

    let (backend, model, input) = match (backend, model, input) {
        (Some(b), Some(m), Some(i)) => (b, m, i),
        _ => usage(),
    };
    let kind = parse_backend(&backend).unwrap_or_else(|| {
        eprintln!("unknown backend: {backend}");
        usage()
    });

    let bytes = match fs::read(&model) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read model {model}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let art = Artifact {
        backend: kind,
        bytes,
        meta: ArtifactMeta::default(),
    };

    let accel = accel_for_device(&device);
    let reg = build_registry(&device, accel);
    let target = TargetProfile {
        accel,
        no_std: false,
    };
    let rt = match reg.select(&art, &target) {
        Some(rt) => rt,
        None => {
            eprintln!(
                "no compiled-in backend admits a {kind:?} artifact on {accel:?} ({} registered)",
                reg.len()
            );
            return ExitCode::FAILURE;
        }
    };
    eprintln!("dispatch -> {:?} on {device}  {:?}", rt.kind(), rt.capability());

    let session = match rt.load(&art) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::FAILURE;
        }
    };

    let raw = match fs::read(&input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read input {input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let out = match session.run(&[Tensor::Host(HostTensor::from_f32(shape, &floats))]) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("run: {e}");
            return ExitCode::FAILURE;
        }
    };
    let logits = out[0].to_host().and_then(|h| h.as_f32()).unwrap();
    let (idx, val) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let label = labels
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| s.lines().nth(idx).map(str::to_string))
        .unwrap_or_else(|| "<no labels>".into());
    println!("top-1: class {idx}  logit {val:.4}  =>  {label}");
    ExitCode::SUCCESS
}
