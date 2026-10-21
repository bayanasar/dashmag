//! `dash` — backend-agnostic Dashmag inference runner.
//!
//! Builds a [`Registry`] of whatever backends are compiled in, tags the input
//! artifact with its `BackendKind` and declared IO, and lets the registry
//! *dispatch*. Same path the acceptance tests drive.

use dash_core::{
    argmax, AccelClass, Artifact, ArtifactMeta, BackendKind, DType, HostTensor, Registry,
    TensorSpec, TargetProfile, Tensor,
};
use dash_cli::model_json;
use std::fs;
use std::process::ExitCode;

/// Map an IREE-style device string to the accelerator class the registry ranks by.
fn accel_for_device(device: &str) -> AccelClass {
    match device {
        "cuda" => AccelClass::Cuda,
        "rocm" => AccelClass::Rocm,
        "vulkan" => AccelClass::Vulkan,
        "metal" => AccelClass::Metal,
        _ => AccelClass::Cpu, // local-task / local-sync / anything else
    }
}

#[allow(unused_variables)]
fn build_registry(device: &str, accel: AccelClass, function: Option<&str>) -> Registry {
    #[allow(unused_mut)]
    let mut reg = Registry::new();
    #[cfg(feature = "tract")]
    reg.register(Box::new(dash_tract::TractRuntime::new()));
    #[cfg(feature = "iree")]
    {
        let mut rt = dash_iree::IreeRuntime::new(device.to_string(), accel);
        if let Some(f) = function {
            rt = rt.with_function(f);
        }
        reg.register(Box::new(rt));
    }
    #[cfg(feature = "tensorrt")]
    reg.register(Box::new(dash_tensorrt::TensorRtRuntime::new()));
    #[cfg(feature = "edgetpu")]
    reg.register(Box::new(dash_edgetpu::EdgeTpuRuntime::new()));
    // One Burn runtime per compiled-in device; --device picks among them by accelerator.
    #[cfg(feature = "burn")]
    reg.register(Box::new(dash_burn::BurnRuntime::flex()));
    #[cfg(feature = "burn-cuda")]
    reg.register(Box::new(dash_burn::BurnRuntime::cuda(0)));
    #[cfg(feature = "burn-rocm")]
    reg.register(Box::new(dash_burn::BurnRuntime::rocm(0)));
    #[cfg(feature = "burn-metal")]
    reg.register(Box::new(dash_burn::BurnRuntime::metal()));
    reg
}

fn parse_backend(s: &str) -> Option<BackendKind> {
    match s {
        "tract" => Some(BackendKind::Tract),
        "iree" => Some(BackendKind::Iree),
        "edgetpu" => Some(BackendKind::EdgeTpu),
        "tensorrt" => Some(BackendKind::TensorRt),
        "burn" => Some(BackendKind::Burn),
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

fn parse_rgb(s: &str) -> Result<[f32; 3], String> {
    let v: Vec<f32> = s
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f32>()
                .map_err(|e| format!("bad value {t:?}: {e}"))
        })
        .collect::<Result<_, _>>()?;
    v.try_into()
        .map_err(|v: Vec<f32>| format!("expected three values R,G,B, got {}", v.len()))
}

#[cfg(feature = "image")]
fn image_tensor(
    path: &str,
    shape: &[usize],
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>,
) -> Result<Vec<f32>, String> {
    let [1, 3, h, w] = shape[..] else {
        return Err(format!("--image needs --shape 1,3,H,W, got {shape:?}"));
    };
    let bytes = fs::read(path).map_err(|e| format!("read image {path}: {e}"))?;
    dash_image::image_input(
        &bytes,
        h,
        w,
        mean.unwrap_or(dash_image::IMAGENET_MEAN),
        std.unwrap_or(dash_image::IMAGENET_STD),
    )
}

#[cfg(not(feature = "image"))]
fn image_tensor(
    _: &str,
    _: &[usize],
    _: Option<[f32; 3]>,
    _: Option<[f32; 3]>,
) -> Result<Vec<f32>, String> {
    Err("--image needs dash built with the `image` feature".into())
}

fn usage() -> ! {
    eprintln!(
        "usage: dash --backend <tract|iree|tensorrt|edgetpu|burn> --model <artifact> \
         (--input <blob> | --image <jpeg>) [--meta <model.json>] [--labels <txt>] \
         [--shape N,C,H,W] [--out-shape N,C] [--device <cpu|local-task|cuda|rocm|vulkan|metal>] \
         [--function <entry>] [--mean R,G,B] [--std R,G,B] [--logits <file>]\n\
         \n\
         --input takes a prepared little-endian f32 tensor. --image takes a JPEG and\n\
         prepares it here: the whole frame resized to --shape's H x W exactly as\n\
         Pillow's bilinear resize does, scaled to [0, 1], normalised by --mean and\n\
         --std (the model's, from --meta, else ImageNet's), NCHW.\n\
         \n\
         --meta reads the model's description: each output's name and classes, the\n\
         outputs that mean nothing under a condition (printed as undefined, not as an\n\
         answer), and defaults for --shape, --out-shape, --mean and --std. A burn\n\
         artifact is weights only and needs it, to know which model to build.\n\
         \n\
         --out-shape declares the model's output shape so every backend returns\n\
         the same shape (runners that emit raw buffers otherwise return flat).\n\
         Repeat it once per output, in declaration order, for a model with more\n\
         than one head: --out-shape 1,3 --out-shape 1,5 --out-shape 1,4\n\
         \n\
         --logits writes every output's logits, in order, as little-endian f32."
    );
    std::process::exit(2);
}

fn run() -> Result<(), String> {
    let (mut backend, mut model, mut input, mut labels) = (None, None, None, None);
    let mut shape: Option<Vec<usize>> = None;
    // One entry per output, in declaration order. A five-head attribute model needs
    // five, and a single --out-shape cannot say that.
    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut device = String::from("local-task");
    let mut function: Option<String> = None;
    let mut image: Option<String> = None;
    let mut mean: Option<[f32; 3]> = None;
    let mut std: Option<[f32; 3]> = None;
    let mut meta_path: Option<String> = None;
    let mut logits_path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--backend" => backend = args.next(),
            "--model" => model = args.next(),
            "--input" => input = args.next(),
            "--labels" => labels = args.next(),
            "--meta" => meta_path = args.next(),
            "--logits" => logits_path = args.next(),
            "--device" => device = args.next().unwrap_or(device),
            "--function" => function = args.next(),
            "--image" => image = args.next(),
            "--mean" => mean = Some(parse_rgb(&args.next().ok_or("--mean needs a value")?)?),
            "--std" => std = Some(parse_rgb(&args.next().ok_or("--std needs a value")?)?),
            "--shape" => {
                shape = Some(parse_shape(&args.next().ok_or("--shape needs a value")?)?);
            }
            "--out-shape" => {
                out_shapes.push(parse_shape(&args.next().ok_or("--out-shape needs a value")?)?);
            }
            "-h" | "--help" => usage(),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    let (backend, model) = match (backend, model) {
        (Some(b), Some(m)) if input.is_some() != image.is_some() => (b, m),
        _ => usage(),
    };
    let kind = parse_backend(&backend).ok_or_else(|| format!("unknown backend {backend:?}"))?;

    let desc = match &meta_path {
        Some(p) => {
            let text = fs::read_to_string(p).map_err(|e| format!("read meta {p}: {e}"))?;
            Some(model_json::parse(&text).map_err(|e| format!("meta {p}: {e}"))?)
        }
        None => None,
    };
    // Explicit flags win; the model's own description comes next; then the old defaults.
    let shape = shape
        .or_else(|| desc.as_ref().and_then(|d| d.input.clone()))
        .unwrap_or_else(|| vec![1, 3, 224, 224]);
    let mean = mean.or_else(|| desc.as_ref().and_then(|d| d.mean));
    let std = std.or_else(|| desc.as_ref().and_then(|d| d.std));
    if let (true, Some(d)) = (out_shapes.is_empty(), &desc) {
        out_shapes = d.spec.widths().iter().map(|&w| vec![shape[0], w]).collect();
    }

    let bytes = fs::read(&model).map_err(|e| format!("read model {model}: {e}"))?;

    // Declare IO so shape survives backends whose runner returns a raw buffer.
    let meta = ArtifactMeta {
        inputs: vec![TensorSpec::new(DType::F32, &shape)],
        outputs: out_shapes
            .iter()
            .map(|s| TensorSpec::new(DType::F32, s))
            .collect(),
        quantized: kind == BackendKind::EdgeTpu,
        dynamic_shapes: false,
        model: desc.as_ref().map(|d| d.spec.clone()),
    };
    let art = Artifact {
        backend: kind,
        bytes,
        meta,
    };

    let accel = accel_for_device(&device);
    let reg = build_registry(&device, accel, function.as_deref());
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

    let floats = match (input, image) {
        (Some(input), _) => {
            let raw = fs::read(&input).map_err(|e| format!("read input {input}: {e}"))?;
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        (None, Some(image)) => image_tensor(&image, &shape, mean, std)?,
        (None, None) => unreachable!("checked above"),
    };
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
    if out.is_empty() {
        return Err("backend returned no outputs".into());
    }
    // A declared output count that the backend does not meet is the failure this
    // CLI exists to surface, not something to discover one head later.
    if !out_shapes.is_empty() && out.len() != out_shapes.len() {
        return Err(format!(
            "declared {} output(s), backend returned {}",
            out_shapes.len(),
            out.len()
        ));
    }

    let mut all = Vec::with_capacity(out.len());
    let mut top1 = Vec::with_capacity(out.len());
    for tensor in &out {
        let host = tensor.host().map_err(|e| e.to_string())?;
        let logits = host.as_f32().map_err(|e| e.to_string())?;
        top1.push(argmax(&logits).map_err(|e| e.to_string())?);
        all.push(logits);
    }
    if let Some(p) = &logits_path {
        let raw: Vec<u8> = all.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(p, raw).map_err(|e| format!("write logits {p}: {e}"))?;
    }

    if let Some(d) = &desc {
        // Names come from the model's description; the logits are the same either way.
        let idx: Vec<usize> = top1.iter().map(|&(i, _)| i).collect();
        let undefined = d.spec.undefined(&idx).map_err(|e| e.to_string())?;
        for (i, (tensor, &(class, val))) in out.iter().zip(&top1).enumerate() {
            let o = &d.spec.outputs[i];
            if undefined[i] {
                let why: Vec<String> = d
                    .spec
                    .undefined_when
                    .iter()
                    .filter(|c| c.output == o.name)
                    .map(|c| format!("{} is {}", c.when, c.is))
                    .collect();
                println!(
                    "{}: undefined ({})   [shape {:?}]",
                    o.name,
                    why.join(", "),
                    tensor.shape()
                );
            } else {
                let name = o.classes.get(class).map_or("<no such class>", String::as_str);
                println!(
                    "{}: {name}  (class {class}, logit {val:.4})   [shape {:?}]",
                    o.name,
                    tensor.shape()
                );
            }
        }
        if let Some(c) = &d.calibration_applied {
            println!("calibration applied in training, already in the weights: {c}");
        }
        return Ok(());
    }

    // Labels name classes of one head, so they apply to the first output only;
    // a multi-head model gets one top-1 line per head, in declaration order.
    let names = labels.and_then(|p| fs::read_to_string(p).ok());
    for (i, (tensor, &(idx, val))) in out.iter().zip(&top1).enumerate() {
        let label = if i == 0 {
            names
                .as_deref()
                .and_then(|s| s.lines().nth(idx))
                .unwrap_or("<no labels>")
        } else {
            "<no labels>"
        };
        let head = if out.len() > 1 {
            format!("output {i} ")
        } else {
            String::new()
        };
        println!(
            "{head}top-1: class {idx}  logit {val:.4}  =>  {label}   [shape {:?}]",
            tensor.shape()
        );
    }
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
