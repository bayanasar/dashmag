//! `dash-iree` — the IREE backend: the "run-a-compiled-artifact" extreme,
//! paired against `dash-tract`'s "interpret-a-graph-file" extreme so the
//! `dash-core` trait is forced honest across two maximally-different backends.
//!
//! ## Phase-0 spike — read this
//!
//! This implementation drives the `iree-run-module` **command-line tool as a
//! subprocess** and round-trips tensors through host files. It exists to prove:
//! (a) `torch.export` → IREE `.vmfb` lowering runs correctly, and (b) the
//! `dash-core` trait spans a compiled-artifact backend on both CPU and CUDA.
//!
//! It is **not** the production shape. The design calls for a thin FFI shim over
//! IREE's C runtime with device-resident, zero-copy tensors (Phase 2). This stub
//! pays a host round-trip and a process launch per call and returns a flat,
//! shape-erased output. Do not benchmark it; do not ship it.
//!
//! `iree-run-module` must be resolvable — on `PATH`, or via the
//! `DASH_IREE_RUN_MODULE` environment variable.

use dash_core::{
    AccelClass, Artifact, BackendKind, Capability, DType, DTypeSet, Error, HostTensor,
    Result as DashResult, Runtime, Session, Tensor,
};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn iree_run_module_bin() -> String {
    std::env::var("DASH_IREE_RUN_MODULE").unwrap_or_else(|_| "iree-run-module".to_string())
}

fn tmp_path(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("dash-iree-{pid}-{n}-{tag}"))
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
    vmfb_path: PathBuf,
    device: String,
    function: String,
}

impl Runtime for IreeRuntime {
    fn kind(&self) -> BackendKind {
        BackendKind::Iree
    }
    fn capability(&self) -> &Capability {
        &self.cap
    }
    fn can_run(&self, art: &Artifact) -> bool {
        art.backend == BackendKind::Iree
    }
    fn load(&self, art: &Artifact) -> DashResult<Box<dyn Session>> {
        if art.backend != BackendKind::Iree {
            return Err(Error::Unsupported(format!(
                "dash-iree cannot run a {:?} artifact",
                art.backend
            )));
        }
        let path = tmp_path("module.vmfb");
        std::fs::write(&path, &art.bytes).map_err(|e| Error::Load(e.to_string()))?;
        Ok(Box::new(IreeSession {
            vmfb_path: path,
            device: self.device.clone(),
            function: self.function.clone(),
        }))
    }
}

impl Session for IreeSession {
    fn run(&self, inputs: &[Tensor]) -> DashResult<Vec<Tensor>> {
        let mut cmd = Command::new(iree_run_module_bin());
        cmd.arg(format!("--module={}", self.vmfb_path.display()));
        cmd.arg(format!("--device={}", self.device));
        cmd.arg(format!("--function={}", self.function));

        let mut scratch = Vec::new();
        for t in inputs {
            let h = t.to_host()?;
            if h.dtype != DType::F32 {
                return Err(Error::Shape(format!(
                    "dash-iree stub expects f32 inputs, got {:?}",
                    h.dtype
                )));
            }
            let p = tmp_path("in.bin");
            std::fs::write(&p, &h.data).map_err(|e| Error::Run(e.to_string()))?;
            let shape = h
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            cmd.arg(format!("--input={shape}xf32=@{}", p.display()));
            scratch.push(p);
        }

        let out_path = tmp_path("out.bin");
        cmd.arg(format!("--output=@{}", out_path.display()));

        let output = cmd
            .output()
            .map_err(|e| Error::Run(format!("spawn {}: {e}", iree_run_module_bin())))?;
        for p in &scratch {
            let _ = std::fs::remove_file(p);
        }
        if !output.status.success() {
            let _ = std::fs::remove_file(&out_path);
            return Err(Error::Run(format!(
                "iree-run-module failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        let raw = std::fs::read(&out_path).map_err(|e| Error::Run(e.to_string()))?;
        let _ = std::fs::remove_file(&out_path);
        if raw.len() % 4 != 0 {
            return Err(Error::Run("iree output byte length not a multiple of 4".into()));
        }
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n = data.len();
        // Phase-0 stub: raw output carries no shape, so return a flat [n] tensor.
        // The real backend recovers output shape from the module's reflection.
        Ok(vec![Tensor::Host(HostTensor::from_f32(vec![n], &data))])
    }
}

impl Drop for IreeSession {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.vmfb_path);
    }
}
