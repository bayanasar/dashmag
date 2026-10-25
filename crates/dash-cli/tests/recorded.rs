//! A Burn model against logits another implementation recorded on the same inputs.
//!
//! The usual way to show a model moved between frameworks intact: the old framework ran
//! it on a set of frames and wrote down the decoded pixels and its logits; the new one
//! is fed the same pixels and must give the same logits, then fed the original images
//! through `dash-image` and must give the same top-1 on every output of every frame.
//!
//! Ignored by default; it reads a directory named by `DASH_RECORDED`:
//!
//!   model.safetensors, model.json     the model, as `dash --backend burn` takes it
//!   frames.json                       frame names, in row order
//!   input_u8_<N>x<H>x<W>x3.bin        the decoded frames, row-major NHWC uint8
//!   logits.json                       {frame: {output: [logits]}}, as recorded
//!
//! and, for the image check, the JPEGs in `DASH_RECORDED_IMAGES`. Then:
//!   cargo test -p dash-cli --features burn --test recorded -- --ignored --nocapture
//!
//! The model runs on Flex; with `DASH_BURN_DEVICE=cuda`, on the first CUDA device, in a
//! build with `burn-cuda` and with TF32 off (`dash-burn`'s documentation says why):
//!   CUBECL_CUDA_TF32=0 DASH_BURN_DEVICE=cuda cargo test -p dash-cli --features burn-cuda \
//!     --test recorded -- --ignored --nocapture

use dash_burn::BurnRuntime;
use dash_cli::model_json;
use dash_core::{
    Artifact, ArtifactMeta, BackendKind, HostTensor, Runtime, Session, Tensor, argmax,
};
use std::path::{Path, PathBuf};
use std::{env, fs};

/// Per logit, on identical input pixels.
const TOLERANCE: f32 = 1e-4;
const BATCH: usize = 32;

/// The Burn device under test: Flex, or the first CUDA device with `DASH_BURN_DEVICE=cuda`.
fn burn_runtime() -> BurnRuntime {
    match env::var("DASH_BURN_DEVICE").as_deref() {
        Err(_) | Ok("flex") => BurnRuntime::flex(),
        #[cfg(feature = "burn-cuda")]
        Ok("cuda") => {
            assert_eq!(
                env::var("CUBECL_CUDA_TF32").as_deref(),
                Ok("0"),
                "set CUBECL_CUDA_TF32=0: TF32 would round every matmul input to 10 mantissa bits"
            );
            BurnRuntime::cuda(0)
        }
        Ok(other) => panic!("DASH_BURN_DEVICE={other} is not a device this build has"),
    }
}

fn session(dir: &Path) -> (Box<dyn Session>, model_json::Description) {
    let desc = model_json::parse(&fs::read_to_string(dir.join("model.json")).unwrap()).unwrap();
    let art = Artifact {
        backend: BackendKind::Burn,
        bytes: fs::read(dir.join("model.safetensors")).unwrap(),
        meta: ArtifactMeta {
            model: Some(desc.spec.clone()),
            ..Default::default()
        },
    };
    (burn_runtime().load(&art).unwrap(), desc)
}

/// Logits per frame per output, for a batch of NCHW f32 frames.
fn run(s: &dyn Session, x: &[f32], n: usize, h: usize, w: usize) -> Vec<Vec<Vec<f32>>> {
    let out = s
        .run(&[Tensor::Host(HostTensor::from_f32(vec![n, 3, h, w], x))])
        .unwrap();
    let per_output: Vec<Vec<f32>> = out
        .iter()
        .map(|t| t.to_host().unwrap().as_f32().unwrap())
        .collect();
    (0..n)
        .map(|i| {
            per_output
                .iter()
                .zip(&out)
                .map(|(v, t)| {
                    let width = t.shape()[1];
                    v[i * width..(i + 1) * width].to_vec()
                })
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "needs recorded inputs and logits in DASH_RECORDED; run with --ignored"]
fn burn_reproduces_recorded_logits() {
    let dir = PathBuf::from(env::var("DASH_RECORDED").expect("set DASH_RECORDED"));
    let (s, desc) = session(&dir);
    let names: Vec<String> =
        serde_json::from_str(&fs::read_to_string(dir.join("frames.json")).unwrap()).unwrap();
    let recorded: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("logits.json")).unwrap()).unwrap();
    let want = |name: &str, o: usize| -> Vec<f32> {
        recorded[name][&desc.spec.outputs[o].name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    let bin = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| {
            p.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("input_u8_")
        })
        .expect("no input_u8_*.bin");
    let dims: Vec<usize> = bin
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .trim_start_matches("input_u8_")
        .split('x')
        .map(|d| d.parse().unwrap())
        .collect();
    let [n, h, w, 3] = dims[..] else {
        panic!("input file {bin:?} is not NxHxWx3")
    };
    assert_eq!(n, names.len(), "one decoded frame per name");
    let pixels = fs::read(&bin).unwrap();
    let (mean, std) = (desc.mean.unwrap(), desc.std.unwrap());
    let outputs = desc.spec.outputs.len();

    // Same pixels: the same logits.
    let (mut worst, mut top1_same) = (0f32, 0usize);
    let mut top1_recorded = Vec::with_capacity(n);
    for start in (0..n).step_by(BATCH) {
        let m = BATCH.min(n - start);
        let mut x = Vec::with_capacity(m * 3 * h * w);
        for i in start..start + m {
            let img = dash_image::Rgb8 {
                width: w,
                height: h,
                data: pixels[i * h * w * 3..(i + 1) * h * w * 3].to_vec(),
            };
            x.extend(dash_image::to_nchw(&img, mean, std));
        }
        for (k, got) in run(s.as_ref(), &x, m, h, w).iter().enumerate() {
            let name = &names[start + k];
            let mut tops = Vec::with_capacity(outputs);
            for (o, g) in got.iter().enumerate() {
                let r = want(name, o);
                worst = g
                    .iter()
                    .zip(&r)
                    .map(|(a, b)| (a - b).abs())
                    .fold(worst, f32::max);
                let t = argmax(&r).unwrap().0;
                top1_same += usize::from(argmax(g).unwrap().0 == t);
                tops.push(t);
            }
            top1_recorded.push(tops);
        }
    }
    eprintln!(
        "recorded pixels: {n} frames, max |new - recorded| {worst:.2e}; \
         top-1 agrees on {top1_same} of {} frame-outputs",
        n * outputs
    );
    assert!(worst < TOLERANCE, "a logit differs by {worst:.2e}");
    assert_eq!(top1_same, n * outputs);

    // From the original images, through this repository's own decode and resize.
    let Ok(images) = env::var("DASH_RECORDED_IMAGES") else {
        eprintln!("DASH_RECORDED_IMAGES unset: image check skipped");
        return;
    };
    let mut agree = 0;
    for (i, name) in names.iter().enumerate() {
        let jpeg = fs::read(Path::new(&images).join(name)).unwrap();
        let x = dash_image::image_input(&jpeg, h, w, mean, std).unwrap();
        let got = &run(s.as_ref(), &x, 1, h, w)[0];
        for (o, g) in got.iter().enumerate() {
            agree += usize::from(argmax(g).unwrap().0 == top1_recorded[i][o]);
        }
    }
    eprintln!(
        "original images through dash-image: top-1 agrees on {agree} of {} frame-outputs",
        n * outputs
    );
    assert_eq!(agree, n * outputs);
}
