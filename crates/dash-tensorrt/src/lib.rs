//! `dash-tensorrt` — NVIDIA TensorRT backend: a vendor-delegate leaf for Jetson
//! GPUs (Nano, TX2, Orin). One SDK, so this single crate covers every TensorRT
//! board — the correct axis of sharing, versus fusing unrelated vendor SDKs.
//!
//! ## Phase-0.5 spike — read this
//!
//! Drives the **`trtexec`** tool as a subprocess and round-trips tensors through
//! host files. It proves the closed-SDK path spans the `dash-core` trait and
//! returns the correct class on a real Jetson GPU. It is **not** the production
//! shape: that is a thin FFI shim over `libnvinfer` (`IExecutionContext`,
//! `enqueueV3`) with device-resident tensors. This stub pays a process launch +
//! host round-trip per call and parses `trtexec`'s JSON output.
//!
//! Numerics note: TensorRT uses its own (often fp16) kernels, so results match
//! the reference *class*, not a bit-exact logit — that is expected for a vendor
//! delegate, and why `Capability` here is not a bit-exactness claim.
//!
//! The `.engine` artifact is GPU- and TensorRT-version-specific and must be built
//! on the target (`trtexec --onnx=… --saveEngine=… --fp16`). `trtexec` is located
//! via `DASH_TRTEXEC` or `PATH`.

use dash_core::{
    AccelClass, Artifact, BackendKind, Capability, DType, DTypeSet, Error, HostTensor,
    Result as DashResult, Runtime, Session, Tensor,
};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn trtexec_bin() -> String {
    std::env::var("DASH_TRTEXEC").unwrap_or_else(|_| "trtexec".to_string())
}

fn tmp_path(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("dash-trt-{}-{n}-{tag}", std::process::id()))
}

/// Scrape `Created (input|output) binding for NAME` out of trtexec's log.
fn scrape_binding(log: &str, kind: &str) -> Option<String> {
    let needle = format!("Created {kind} binding for ");
    log.lines().find_map(|l| {
        l.find(&needle)
            .map(|i| l[i + needle.len()..].split_whitespace().next().unwrap_or("").to_string())
    })
}

/// TensorRT backend bound to the GPU on the current (Jetson) host.
pub struct TensorRtRuntime {
    cap: Capability,
}

impl TensorRtRuntime {
    pub fn new() -> Self {
        TensorRtRuntime {
            cap: Capability {
                accel: AccelClass::Cuda,
                dtypes: DTypeSet::of(&[DType::F32, DType::F16]),
                dynamic_shapes: false,
                quantized: false,
                no_std: false,
            },
        }
    }
}

impl Default for TensorRtRuntime {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TensorRtSession {
    engine_path: PathBuf,
    input_binding: String,
    output_binding: String,
}

impl Runtime for TensorRtRuntime {
    fn kind(&self) -> BackendKind {
        BackendKind::TensorRt
    }
    fn capability(&self) -> &Capability {
        &self.cap
    }
    fn can_run(&self, art: &Artifact) -> bool {
        art.backend == BackendKind::TensorRt
    }
    fn load(&self, art: &Artifact) -> DashResult<Box<dyn Session>> {
        if art.backend != BackendKind::TensorRt {
            return Err(Error::Unsupported(format!(
                "dash-tensorrt cannot run a {:?} artifact",
                art.backend
            )));
        }
        let engine_path = tmp_path("model.engine");
        std::fs::write(&engine_path, &art.bytes).map_err(|e| Error::Load(e.to_string()))?;

        // Auto-detect binding names by loading the engine once (random input).
        let probe = Command::new(trtexec_bin())
            .arg(format!("--loadEngine={}", engine_path.display()))
            .output()
            .map_err(|e| Error::Load(format!("spawn {}: {e}", trtexec_bin())))?;
        let log = format!(
            "{}{}",
            String::from_utf8_lossy(&probe.stdout),
            String::from_utf8_lossy(&probe.stderr)
        );
        let input_binding = scrape_binding(&log, "input")
            .ok_or_else(|| Error::Load("could not find input binding in trtexec log".into()))?;
        let output_binding = scrape_binding(&log, "output")
            .ok_or_else(|| Error::Load("could not find output binding in trtexec log".into()))?;

        Ok(Box::new(TensorRtSession {
            engine_path,
            input_binding,
            output_binding,
        }))
    }
}

impl Session for TensorRtSession {
    fn run(&self, inputs: &[Tensor]) -> DashResult<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(Error::Unsupported(
                "dash-tensorrt spike supports a single input binding".into(),
            ));
        }
        let h = inputs[0].to_host()?;
        if h.dtype != DType::F32 {
            return Err(Error::Shape(format!(
                "dash-tensorrt spike expects f32 input, got {:?}",
                h.dtype
            )));
        }
        let in_path = tmp_path("in.bin");
        std::fs::write(&in_path, &h.data).map_err(|e| Error::Run(e.to_string()))?;
        let out_json = tmp_path("out.json");

        let out = Command::new(trtexec_bin())
            .arg(format!("--loadEngine={}", self.engine_path.display()))
            .arg(format!("--loadInputs={}:{}", self.input_binding, in_path.display()))
            .arg(format!("--exportOutput={}", out_json.display()))
            .output()
            .map_err(|e| Error::Run(format!("spawn {}: {e}", trtexec_bin())))?;
        let _ = std::fs::remove_file(&in_path);
        if !out.status.success() {
            let _ = std::fs::remove_file(&out_json);
            return Err(Error::Run(format!(
                "trtexec run failed: {}",
                String::from_utf8_lossy(&out.stderr)
                    .lines()
                    .filter(|l| l.contains("[E]"))
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }

        let json = std::fs::read_to_string(&out_json).map_err(|e| Error::Run(e.to_string()))?;
        let _ = std::fs::remove_file(&out_json);
        let parsed: serde_json::Value =
            serde_json::from_str(&json).map_err(|e| Error::Run(format!("parse trtexec json: {e}")))?;
        let arr = parsed
            .as_array()
            .ok_or_else(|| Error::Run("trtexec output json is not an array".into()))?;
        let obj = arr
            .iter()
            .find(|o| o.get("name").and_then(|n| n.as_str()) == Some(&self.output_binding))
            .or_else(|| arr.first())
            .ok_or_else(|| Error::Run("no output tensor in trtexec json".into()))?;
        let values = obj
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Run("no 'values' array in trtexec output".into()))?;
        let data: Vec<f32> = values.iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect();
        let n = data.len();
        Ok(vec![Tensor::Host(HostTensor::from_f32(vec![n], &data))])
    }
}

impl Drop for TensorRtSession {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.engine_path);
    }
}
