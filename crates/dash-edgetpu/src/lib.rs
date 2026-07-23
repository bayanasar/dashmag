//! `dash-edgetpu` — Coral Edge TPU backend: a vendor-delegate leaf for the
//! closed Edge TPU appliance (`/dev/apex_0` via `libedgetpu`).
//!
//! ## STATUS: Phase-0.5 spike — NOT YET PROVEN ON HARDWARE
//!
//! The crate compiles and implements the `dash-core` trait, but it has **not**
//! been executed on a real Edge TPU. Two things gate that proof and neither is
//! done here:
//!   1. An **Edge-TPU artifact**: an int8-quantized `.tflite` compiled by
//!      `edgetpu_compiler`. MobileNetV3-Small's SE / hard-swish blocks are known
//!      to map poorly to the Edge TPU (partial CPU fallback) — the risk the
//!      execution plan flagged. Producing this artifact is a separate spike.
//!   2. A reachable Coral running the runner (`tflite_runtime` + `libedgetpu`).
//!
//! Design (matches `dash-iree`/`dash-tensorrt`): drive a subprocess runner that
//! executes the edgetpu `.tflite` on the TPU and round-trips tensors through
//! host files. The runner is located via `DASH_EDGETPU_RUN` (a python script
//! using `tflite_runtime` + the libedgetpu delegate; see `tools/edgetpu_run.py`).
//! The production shape is an FFI shim over `libedgetpu` + the tflite C API.
//!
//! Edge TPU is int8-only; `Capability` reflects that (`quantized: true`).

use dash_core::{
    AccelClass, Artifact, BackendKind, Capability, DType, DTypeSet, Error, HostTensor,
    Result as DashResult, Runtime, Session, Tensor,
};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn runner_bin() -> DashResult<String> {
    std::env::var("DASH_EDGETPU_RUN").map_err(|_| {
        Error::Load("set DASH_EDGETPU_RUN to the edgetpu runner (see tools/edgetpu_run.py)".into())
    })
}

fn tmp_path(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("dash-edgetpu-{}-{n}-{tag}", std::process::id()))
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
                dtypes: DTypeSet::of(&[DType::U8, DType::I8]),
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
    model_path: PathBuf,
    runner: String,
}

impl Runtime for EdgeTpuRuntime {
    fn kind(&self) -> BackendKind {
        BackendKind::EdgeTpu
    }
    fn capability(&self) -> &Capability {
        &self.cap
    }
    fn can_run(&self, art: &Artifact) -> bool {
        art.backend == BackendKind::EdgeTpu
    }
    fn load(&self, art: &Artifact) -> DashResult<Box<dyn Session>> {
        if art.backend != BackendKind::EdgeTpu {
            return Err(Error::Unsupported(format!(
                "dash-edgetpu cannot run a {:?} artifact",
                art.backend
            )));
        }
        let model_path = tmp_path("model_edgetpu.tflite");
        std::fs::write(&model_path, &art.bytes).map_err(|e| Error::Load(e.to_string()))?;
        Ok(Box::new(EdgeTpuSession {
            model_path,
            runner: runner_bin()?,
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
        let h = inputs[0].to_host()?;
        let in_path = tmp_path("in.bin");
        let out_path = tmp_path("out.f32");
        std::fs::write(&in_path, &h.data).map_err(|e| Error::Run(e.to_string()))?;

        // Runner contract: `<runner> <model.tflite> <input.bin> <output.f32>`
        // (dequantizes to f32 on the device so the rest of the pipeline is uniform).
        let status = Command::new("python3")
            .arg(&self.runner)
            .arg(&self.model_path)
            .arg(&in_path)
            .arg(&out_path)
            .output()
            .map_err(|e| Error::Run(format!("spawn edgetpu runner: {e}")))?;
        let _ = std::fs::remove_file(&in_path);
        if !status.status.success() {
            let _ = std::fs::remove_file(&out_path);
            return Err(Error::Run(format!(
                "edgetpu runner failed: {}",
                String::from_utf8_lossy(&status.stderr).trim()
            )));
        }
        let raw = std::fs::read(&out_path).map_err(|e| Error::Run(e.to_string()))?;
        let _ = std::fs::remove_file(&out_path);
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n = data.len();
        Ok(vec![Tensor::Host(HostTensor::from_f32(vec![n], &data))])
    }
}

impl Drop for EdgeTpuSession {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.model_path);
    }
}
