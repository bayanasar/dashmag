//! `dash-edgetpu` — Coral Edge TPU backend: a vendor-delegate leaf for the
//! closed Edge TPU appliance (`/dev/apex_0` via `libedgetpu`).
//!
//! ## STATUS: Phase-0.5 spike — NOT YET PROVEN ON HARDWARE
//!
//! The crate compiles and implements the `dash-core` trait, but it has **not**
//! been executed on a real Edge TPU. Two things gate that proof:
//!   1. An **Edge-TPU artifact**: an int8-quantized `.tflite` compiled by
//!      `edgetpu_compiler`. MobileNetV3-Small's SE / hard-swish blocks are known
//!      to map poorly to the Edge TPU (partial CPU fallback) — the risk the
//!      execution plan flagged. Producing this artifact is a separate spike.
//!   2. A reachable Coral running the runner (`tflite_runtime` + `libedgetpu`).
//!
//! Design (matching `dash-iree`/`dash-tensorrt`): drive a subprocess runner that
//! executes the edgetpu `.tflite` on the TPU, exchanging tensors through a
//! private 0700 temporary directory. The runner is located via
//! `DASH_EDGETPU_RUN` (see `tools/edgetpu_run.py`). The production shape is an
//! FFI shim over `libedgetpu` + the tflite C API.
//!
//! Edge TPU is int8-only at the input; the runner dequantizes outputs to f32 so
//! the rest of the pipeline stays uniform. `Capability` reflects both.

use dash_core::{
    AccelClass, Artifact, ArtifactMeta, BackendKind, Capability, DType, DTypeSet, Error,
    HostTensor, Result as DashResult, Runtime, Session, Tensor,
};
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn runner_script() -> DashResult<String> {
    std::env::var("DASH_EDGETPU_RUN").map_err(|_| {
        Error::Load("set DASH_EDGETPU_RUN to the edgetpu runner (see tools/edgetpu_run.py)".into())
    })
}

/// The Coral Edge TPU backend (closed appliance behind `libedgetpu`).
pub struct EdgeTpuRuntime {
    cap: Capability,
}

impl EdgeTpuRuntime {
    pub fn new() -> Self {
        EdgeTpuRuntime {
            cap: Capability {
                accel: AccelClass::VendorNpu,
                // int8/uint8 in; the runner dequantizes outputs to f32.
                dtypes: DTypeSet::of(&[DType::U8, DType::I8, DType::F32]),
                dynamic_shapes: false,
                quantized: true,
                no_std: false,
            },
        }
    }
}

impl Default for EdgeTpuRuntime {
    fn default() -> Self {
        Self::new()
    }
}

pub struct EdgeTpuSession {
    dir: TempDir,
    model_path: PathBuf,
    runner: String,
    meta: ArtifactMeta,
}

impl Runtime for EdgeTpuRuntime {
    fn kind(&self) -> BackendKind {
        BackendKind::EdgeTpu
    }
    fn capability(&self) -> &Capability {
        &self.cap
    }
    fn load(&self, art: &Artifact) -> DashResult<Box<dyn Session>> {
        if art.backend != BackendKind::EdgeTpu {
            return Err(Error::Unsupported(format!(
                "dash-edgetpu cannot run a {:?} artifact",
                art.backend
            )));
        }
        if let Err(why) = self.cap.admits(&art.meta) {
            return Err(Error::Unsupported(format!("dash-edgetpu: {why}")));
        }
        let dir = tempfile::Builder::new()
            .prefix("dash-edgetpu-")
            .tempdir()
            .map_err(|e| Error::Load(format!("private temp dir: {e}")))?;
        let model_path = dir.path().join("model_edgetpu.tflite");
        std::fs::write(&model_path, &art.bytes).map_err(|e| Error::Load(e.to_string()))?;
        Ok(Box::new(EdgeTpuSession {
            dir,
            model_path,
            runner: runner_script()?,
            meta: art.meta.clone(),
        }))
    }
}

impl Session for EdgeTpuSession {
    fn run(&self, inputs: &[Tensor]) -> DashResult<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(Error::Unsupported(
                "dash-edgetpu spike supports a single input".into(),
            ));
        }
        let h = inputs[0].host()?;
        // The Edge TPU is a quantized appliance: silently handing it f32 bytes
        // would reshape wrong inside the runner instead of failing here.
        if !matches!(h.dtype, DType::U8 | DType::I8) {
            return Err(Error::Shape(format!(
                "dash-edgetpu expects quantized u8/i8 input, got {:?}",
                h.dtype
            )));
        }
        let in_path = self.dir.path().join("in.bin");
        let out_prefix = self.dir.path().join("out");
        std::fs::write(&in_path, &h.data).map_err(|e| Error::Run(e.to_string()))?;

        // Runner contract: `<runner> <model.tflite> <input.bin> <out_prefix>`,
        // writing `<out_prefix>.<i>.bin` as dequantized little-endian f32.
        let status = Command::new("python3")
            .arg(&self.runner)
            .arg(&self.model_path)
            .arg(&in_path)
            .arg(&out_prefix)
            .output()
            .map_err(|e| Error::Run(format!("spawn edgetpu runner: {e}")))?;
        if !status.status.success() {
            return Err(Error::Run(format!(
                "edgetpu runner failed: {}",
                String::from_utf8_lossy(&status.stderr).trim()
            )));
        }

        let mut outs = Vec::new();
        for i in 0.. {
            let p = PathBuf::from(format!("{}.{i}.bin", out_prefix.display()));
            if !p.exists() {
                break;
            }
            let raw = std::fs::read(&p).map_err(|e| Error::Run(e.to_string()))?;
            if raw.len() % 4 != 0 {
                return Err(Error::Run(format!(
                    "output {i} byte length {} is not a multiple of 4",
                    raw.len()
                )));
            }
            let data: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            outs.push(Tensor::Host(HostTensor::f32_shaped(
                &data,
                self.meta.output_shape(i),
            )));
        }
        if outs.is_empty() {
            return Err(Error::Run(
                "edgetpu runner produced no output files".into(),
            ));
        }
        Ok(outs)
    }
}
