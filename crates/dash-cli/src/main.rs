//! `dash` — backend-agnostic Dashmag inference runner.
//!
//! Builds a [`Registry`] of whatever backends are compiled in, tags the input
//! artifact with its `BackendKind` and declared IO, and lets the registry
//! *dispatch*. Same path the acceptance tests drive.

use dash_core::{
    argmax, AccelClass, Artifact, ArtifactMeta, BackendKind, DType, HostTensor, Registry,
    TensorSpec, TargetProfile, Tensor,
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
    reg.register(Box::new(dash_iree::IreeRuntime::new(
        device.to_string(),
        accel,
    )));
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

/// Strict shape parsing: a malformed dimension is an error, never a silently
/// different tensor.
fn parse_shape(s: &str) -> Result<Vec<usize>, String> {
    if s.trim().is_empty() {
        return Err("shape is empty".to_string());
    }
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid dimension {:?} in shape {:?}", p.trim(), s))
        })
        .collect()
}

fn usage() -> ! {
    eprintln!(
        "usage: dash --backend <tract|iree|tensorrt|edgetpu> --model <artifact> \
         --input <blob> [--labels <txt>] [--shape N,C,H,W] [--out-shape N,C] \
         [--device <local-task|cuda|vulkan>]\n\
         \n\
         --out-shape declares the model's output shape so every backend returns\n\
         the same shape (runners that emit raw buffers otherwise return flat)."
    );
    std::process::exit(2);
}

fn run() -> Result<(), String> {
    let (mut backend, mut model, mut input, mut labels) = (None, None, None, None);
    let mut shape = vec![1usize, 3, 224, 224];
    let mut out_shape: Option<Vec<usize>> = None;
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
                shape = parse_shape(&args.next().ok_or("--shape needs a value")?)?;
            }
            "--out-shape" => {
                out_shape = Some(parse_shape(&args.next().ok_or("--out-shape needs a value")?)?);
            }
            "-h" | "--help" => usage(),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    let (backend, model, input) = match (backend, model, input) {
        (Some(b), Some(m), Some(i)) => (b, m, i),
        _ => usage(),
    };
    let kind = parse_backend(&backend).ok_or_else(|| format!("unknown backend {backend:?}"))?;

    let bytes = fs::read(&model).map_err(|e| format!("read model {model}: {e}"))?;

    // Declare IO so shape survives backends whose runner returns a raw buffer.
    let meta = ArtifactMeta {
        inputs: vec![TensorSpec::new(DType::F32, &shape)],
        outputs: out_shape
            .as_ref()
            .map(|s| vec![TensorSpec::new(DType::F32, s)])
            .unwrap_or_default(),
        quantized: kind == BackendKind::EdgeTpu,
        dynamic_shapes: false,
    };
    let art = Artifact {
        backend: kind,
        bytes,
        meta,
    };

    let accel = accel_for_device(&device);
    let reg = build_registry(&device, accel);
    let target = TargetProfile {
        accel,
        no_std: false,
    };
    let rt = reg.select(&art, &target).ok_or_else(|| {
        // Dispatch that cannot explain itself is indistinguishable from broken.
        let why = reg
            .explain(&art, &target)
            .into_iter()
            .map(|(k, r)| format!("\n  {k:?}: {r}"))
            .collect::<String>();
        format!(
            "no compiled-in backend admits a {kind:?} artifact on {accel:?} \
             ({} registered){why}",
            reg.len()
        )
    })?;
    eprintln!(
        "dispatch -> {:?} on {device}  {:?}",
        rt.kind(),
        rt.capability()
    );

    let session = rt.load(&art).map_err(|e| e.to_string())?;

    let raw = fs::read(&input).map_err(|e| format!("read input {input}: {e}"))?;
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expected: usize = shape.iter().product();
    if floats.len() != expected {
        return Err(format!(
            "input has {} f32 values, shape {:?} needs {}",
            floats.len(),
            shape,
            expected
        ));
    }

    let out = session
        .run(&[Tensor::Host(HostTensor::from_f32(shape, &floats))])
        .map_err(|e| e.to_string())?;
    let first = out.first().ok_or("backend returned no outputs")?;
    let host = first.host().map_err(|e| e.to_string())?;
    let logits = host.as_f32().map_err(|e| e.to_string())?;
    let (idx, val) = argmax(&logits).map_err(|e| e.to_string())?;

    let label = labels
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| s.lines().nth(idx).map(str::to_string))
        .unwrap_or_else(|| "<no labels>".into());
    println!(
        "top-1: class {idx}  logit {val:.4}  =>  {label}   [shape {:?}]",
        first.shape()
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
