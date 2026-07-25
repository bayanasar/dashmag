//! `dash-iree` — the IREE backend: the "run-a-compiled-artifact" extreme,
//! paired against `dash-tract`'s "interpret-a-graph-file" extreme so the
//! `dash-core` trait is forced honest across two maximally-different backends.
//!
//! ## Phase-0 spike — read this
//!
//! This implementation drives the `iree-run-module` **command-line tool as a
//! subprocess** and round-trips tensors through files in a private temporary
//! directory. It exists to prove: (a) the compiled-artifact lowering runs
//! correctly, and (b) the `dash-core` trait spans a compiled-artifact backend
//! on both CPU and CUDA.
//!
//! It is **not** the production shape. The design calls for a thin FFI shim over
//! IREE's C runtime with device-resident, zero-copy tensors (Phase 2). This stub
//! pays a process launch and a host round-trip per call.
//!
//! Output shape is recovered from [`dash_core::ArtifactMeta`]: `iree-run-module`
//! writes a raw, shape-erased buffer, so without declared output specs the
//! result would be flat and would disagree with `dash-tract` on the same model.
//! Declare `meta.outputs` and every backend returns the same shape.
//!
//! `iree-run-module` must be resolvable — on `PATH`, or via the
//! `DASH_IREE_RUN_MODULE` environment variable.

use dash_core::{
    AccelClass, Artifact, ArtifactMeta, BackendKind, Capability, DType, DTypeSet, Error,
    HostTensor, Result as DashResult, Runtime, Session, Tensor,
};
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn iree_run_module_bin() -> String {
    std::env::var("DASH_IREE_RUN_MODULE").unwrap_or_else(|_| "iree-run-module".to_string())
}

/// An IREE backend bound to one device (`local-task`/`local-sync`/`cuda`/...).
pub struct IreeRuntime {
    device: String,
    function: String,
    cap: Capability,
}

impl IreeRuntime {
    /// `device` is an IREE device string; `accel` is how the registry should
    /// classify it for dispatch.
    pub fn new(device: impl Into<String>, accel: AccelClass) -> Self {
        IreeRuntime {
            device: device.into(),
            function: "main_graph".to_string(),
            cap: Capability {
                accel,
                dtypes: DTypeSet::of(&[DType::F32]),
                dynamic_shapes: false,
                quantized: false,
                no_std: false,
            },
        }
    }

    /// Override the entry-point name (ONNX import produces `main_graph`).
    pub fn with_function(mut self, f: impl Into<String>) -> Self {
        self.function = f.into();
        self
    }
}

pub struct IreeSession {
    /// Owns a 0700 private directory; dropped (and recursively removed) with the
    /// session. Model bytes never touch a world-readable predictable path.
    dir: TempDir,
    vmfb_path: PathBuf,
    device: String,
    function: String,
    meta: ArtifactMeta,
}

impl Runtime for IreeRuntime {
    fn kind(&self) -> BackendKind {
        BackendKind::Iree
    }
    fn capability(&self) -> &Capability {
        &self.cap
    }
    fn load(&self, art: &Artifact) -> DashResult<Box<dyn Session>> {
        if art.backend != BackendKind::Iree {
            return Err(Error::Unsupported(format!(
                "dash-iree cannot run a {:?} artifact",
                art.backend
            )));
        }
        if let Err(why) = self.cap.admits(&art.meta) {
            return Err(Error::Unsupported(format!("dash-iree: {why}")));
        }
        let dir = tempfile::Builder::new()
            .prefix("dash-iree-")
            .tempdir()
            .map_err(|e| Error::Load(format!("private temp dir: {e}")))?;
        let vmfb_path = dir.path().join("module.vmfb");
        std::fs::write(&vmfb_path, &art.bytes).map_err(|e| Error::Load(e.to_string()))?;
        Ok(Box::new(IreeSession {
            dir,
            vmfb_path,
            device: self.device.clone(),
            function: self.function.clone(),
            meta: art.meta.clone(),
        }))
    }
}

impl Session for IreeSession {
    fn run(&self, inputs: &[Tensor]) -> DashResult<Vec<Tensor>> {
        let mut cmd = Command::new(iree_run_module_bin());
        cmd.arg(format!("--module={}", self.vmfb_path.display()));
        cmd.arg(format!("--device={}", self.device));
        cmd.arg(format!("--function={}", self.function));

        for (i, t) in inputs.iter().enumerate() {
            let h = t.host()?;
            if h.dtype != DType::F32 {
                return Err(Error::Shape(format!(
                    "dash-iree spike expects f32 inputs, got {:?}",
                    h.dtype
                )));
            }
            let p = self.dir.path().join(format!("in{i}.bin"));
            std::fs::write(&p, &h.data).map_err(|e| Error::Run(e.to_string()))?;
            let shape = h
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            cmd.arg(format!("--input={shape}xf32=@{}", p.display()));
        }

        // One --output flag per declared output. With no declaration we can only
        // ask for one and say so, rather than silently dropping the rest.
        let n_out = self.meta.outputs.len().max(1);
        let out_paths: Vec<PathBuf> = (0..n_out)
            .map(|i| self.dir.path().join(format!("out{i}.bin")))
            .collect();
        for p in &out_paths {
            cmd.arg(format!("--output=@{}", p.display()));
        }

        let output = cmd
            .output()
            .map_err(|e| Error::Run(format!("spawn {}: {e}", iree_run_module_bin())))?;
        if !output.status.success() {
            return Err(Error::Run(format!(
                "iree-run-module failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        let mut outs = Vec::with_capacity(out_paths.len());
        for (i, p) in out_paths.iter().enumerate() {
            let raw = std::fs::read(p).map_err(|e| {
                Error::Run(format!("reading output {i}: {e} (declared {n_out} outputs)"))
            })?;
            if let Some(dt) = self.meta.output_dtype(i) {
                if dt != DType::F32 {
                    return Err(Error::Shape(format!(
                        "dash-iree spike decodes f32 outputs only, output {i} is declared {dt:?}"
                    )));
                }
            }
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
        Ok(outs)
    }
}
