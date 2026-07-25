//! `dash-tensorrt` — NVIDIA TensorRT backend: a vendor-delegate leaf for Jetson
//! GPUs (Nano, TX2, Orin). One SDK, so this single crate covers every TensorRT
//! board — the correct axis of sharing, versus fusing unrelated vendor SDKs.
//!
//! ## Phase-0.5 spike — read this
//!
//! Drives the **`trtexec`** tool as a subprocess and round-trips tensors through
//! a private temporary directory. It proves the closed-SDK path spans the
//! `dash-core` trait and returns the correct class on a real Jetson GPU. It is
//! **not** the production shape: that is a thin FFI shim over `libnvinfer`
//! (`IExecutionContext`, `enqueueV3`) with device-resident tensors.
//!
//! Numerics note: TensorRT uses its own (often fp16) kernels, so results match
//! the reference *class*, not a bit-exact logit — expected for a vendor
//! delegate, and why `Capability` here advertises f16 alongside f32.
//!
//! Output shape comes from [`dash_core::ArtifactMeta`] when declared, and
//! otherwise from the binding dimensions `trtexec` prints, so results agree in
//! shape with `dash-tract` rather than arriving flat.
//!
//! The `.engine` artifact is GPU- and TensorRT-version-specific and must be built
//! on the target (`trtexec --onnx=… --saveEngine=… --fp16`). `trtexec` is located
//! via `DASH_TRTEXEC` or `PATH`.

use dash_core::{
    AccelClass, Artifact, ArtifactMeta, BackendKind, Capability, DType, DTypeSet, Error,
    HostTensor, Result as DashResult, Runtime, Session, Tensor,
};
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn trtexec_bin() -> String {
    std::env::var("DASH_TRTEXEC").unwrap_or_else(|_| "trtexec".to_string())
}

/// One binding as reported by trtexec.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Binding {
    name: String,
    dims: Option<Vec<usize>>,
}

/// Scrape every `Created <kind> binding for NAME with dimensions AxBxC` line.
///
/// All bindings, in order — taking only the first would silently drop outputs
/// on any multi-output model (every detector).
fn scrape_bindings(log: &str, kind: &str) -> Vec<Binding> {
    let needle = format!("Created {kind} binding for ");
    log.lines()
        .filter_map(|l| {
            let i = l.find(&needle)?;
            let rest = &l[i + needle.len()..];
            let mut it = rest.split_whitespace();
            let name = it.next()?.to_string();
            if name.is_empty() {
                return None;
            }
            // ... "with dimensions 1x3x224x224"
            let dims = rest.split("with dimensions ").nth(1).and_then(|d| {
                d.split_whitespace()
                    .next()?
                    .split('x')
                    .map(|p| p.parse::<usize>().ok())
                    .collect::<Option<Vec<_>>>()
            });
            Some(Binding { name, dims })
        })
        .collect()
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
                // TensorRT engines are commonly built fp16/int8.
                quantized: true,
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
    dir: TempDir,
    engine_path: PathBuf,
    inputs: Vec<Binding>,
    outputs: Vec<Binding>,
    meta: ArtifactMeta,
}

impl Runtime for TensorRtRuntime {
    fn kind(&self) -> BackendKind {
        BackendKind::TensorRt
    }
    fn capability(&self) -> &Capability {
        &self.cap
    }
    fn load(&self, art: &Artifact) -> DashResult<Box<dyn Session>> {
        if art.backend != BackendKind::TensorRt {
            return Err(Error::Unsupported(format!(
                "dash-tensorrt cannot run a {:?} artifact",
                art.backend
            )));
        }
        if let Err(why) = self.cap.admits(&art.meta) {
            return Err(Error::Unsupported(format!("dash-tensorrt: {why}")));
        }
        let dir = tempfile::Builder::new()
            .prefix("dash-trt-")
            .tempdir()
            .map_err(|e| Error::Load(format!("private temp dir: {e}")))?;
        let engine_path = dir.path().join("model.engine");
        std::fs::write(&engine_path, &art.bytes).map_err(|e| Error::Load(e.to_string()))?;

        // Load the engine once to discover binding names and dimensions.
        let probe = Command::new(trtexec_bin())
            .arg(format!("--loadEngine={}", engine_path.display()))
            .output()
            .map_err(|e| Error::Load(format!("spawn {}: {e}", trtexec_bin())))?;
        let log = format!(
            "{}{}",
            String::from_utf8_lossy(&probe.stdout),
            String::from_utf8_lossy(&probe.stderr)
        );
        let inputs = scrape_bindings(&log, "input");
        let outputs = scrape_bindings(&log, "output");
        if inputs.is_empty() || outputs.is_empty() {
            return Err(Error::Load(
                "could not discover trtexec input/output bindings from the engine".into(),
            ));
        }
        Ok(Box::new(TensorRtSession {
            dir,
            engine_path,
            inputs,
            outputs,
            meta: art.meta.clone(),
        }))
    }
}

impl Session for TensorRtSession {
    fn run(&self, inputs: &[Tensor]) -> DashResult<Vec<Tensor>> {
        if inputs.len() != self.inputs.len() {
            return Err(Error::Shape(format!(
                "engine expects {} input(s), got {}",
                self.inputs.len(),
                inputs.len()
            )));
        }
        let mut cmd = Command::new(trtexec_bin());
        cmd.arg(format!("--loadEngine={}", self.engine_path.display()));

        for (i, (t, binding)) in inputs.iter().zip(&self.inputs).enumerate() {
            let h = t.host()?;
            if h.dtype != DType::F32 {
                return Err(Error::Shape(format!(
                    "dash-tensorrt spike expects f32 input, got {:?}",
                    h.dtype
                )));
            }
            let p = self.dir.path().join(format!("in{i}.bin"));
            std::fs::write(&p, &h.data).map_err(|e| Error::Run(e.to_string()))?;
            cmd.arg(format!("--loadInputs={}:{}", binding.name, p.display()));
        }

        let out_json = self.dir.path().join("out.json");
        cmd.arg(format!("--exportOutput={}", out_json.display()));

        let out = cmd
            .output()
            .map_err(|e| Error::Run(format!("spawn {}: {e}", trtexec_bin())))?;
        if !out.status.success() {
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
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| Error::Run(format!("parse trtexec json: {e}")))?;
        let arr = parsed
            .as_array()
            .ok_or_else(|| Error::Run("trtexec output json is not an array".into()))?;

        // Every declared output binding, in order — not just the first.
        let mut outs = Vec::with_capacity(self.outputs.len());
        for (i, binding) in self.outputs.iter().enumerate() {
            let obj = arr
                .iter()
                .find(|o| o.get("name").and_then(|n| n.as_str()) == Some(&binding.name))
                .or_else(|| arr.get(i))
                .ok_or_else(|| {
                    Error::Run(format!("output '{}' missing from trtexec json", binding.name))
                })?;
            let values = obj
                .get("values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Run(format!("no 'values' for output '{}'", binding.name)))?;
            let data: Vec<f32> = values
                .iter()
                .filter_map(|v| v.as_f64().map(|x| x as f32))
                .collect();
            // Declared meta wins; otherwise use the dims trtexec reported.
            let declared = self
                .meta
                .output_shape(i)
                .or(binding.dims.as_deref());
            outs.push(Tensor::Host(HostTensor::f32_shaped(&data, declared)));
        }
        Ok(outs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = "\
[I] Created input binding for input.1 with dimensions 1x3x224x224
[I] Created output binding for 419 with dimensions 1x1000
[I] Created output binding for boxes with dimensions 1x100x4";

    #[test]
    fn scrapes_all_bindings_with_dims() {
        let ins = scrape_bindings(LOG, "input");
        assert_eq!(ins.len(), 1);
        assert_eq!(ins[0].name, "input.1");
        assert_eq!(ins[0].dims.as_deref(), Some(&[1usize, 3, 224, 224][..]));

        // A multi-output engine must yield every output, not just the first.
        let outs = scrape_bindings(LOG, "output");
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].name, "419");
        assert_eq!(outs[1].name, "boxes");
        assert_eq!(outs[1].dims.as_deref(), Some(&[1usize, 100, 4][..]));
    }

    #[test]
    fn no_bindings_in_empty_log() {
        assert!(scrape_bindings("", "input").is_empty());
    }
}
