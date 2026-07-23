//! `dash-tract` — pure-Rust CPU inference backend for Dashmag.
//!
//! Wraps [`tract`](https://github.com/sonos/tract) behind the `dash-core`
//! [`Runtime`] trait. This is the *portable floor*: no C dependencies, the only
//! realistic runtime for seL4 / Fuchsia / RISC-V-bare / MCU-class targets, and a
//! zero-C-dependency fallback everywhere else.
//!
//! It is deliberately the "interpret-a-graph-file" extreme of the backend
//! design — paired against IREE's "run-a-compiled-artifact" extreme — so the
//! `dash-core` trait is forced honest before a third backend arrives.

use dash_core::{
    AccelClass, Artifact, BackendKind, Capability, DType, DTypeSet, Error, HostTensor,
    Result as DashResult, Runtime, Session, Tensor as DashTensor,
};
use tract_onnx::prelude::*;

/// A fully lowered, runnable tract plan.
type Plan = TypedRunnableModel<TypedModel>;

/// The pure-Rust CPU backend.
pub struct TractRuntime {
    cap: Capability,
}

impl TractRuntime {
    pub fn new() -> Self {
        TractRuntime {
            cap: Capability {
                accel: AccelClass::Cpu,
                dtypes: DTypeSet::of(&[DType::F32]),
                // Phase 0 pins static shapes; dynamic-shape support is a later step.
                dynamic_shapes: false,
                quantized: false,
                // This host build links std. A genuine no_std embedded build of
                // tract is a separate cargo feature, not this default.
                no_std: false,
            },
        }
    }
}

impl Default for TractRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// A loaded model, ready to run repeatedly.
pub struct TractSession {
    plan: Plan,
}

impl Runtime for TractRuntime {
    fn kind(&self) -> BackendKind {
        BackendKind::Tract
    }

    fn capability(&self) -> &Capability {
        &self.cap
    }

    fn can_run(&self, art: &Artifact) -> bool {
        art.backend == BackendKind::Tract
    }

    fn load(&self, art: &Artifact) -> DashResult<Box<dyn Session>> {
        if art.backend != BackendKind::Tract {
            return Err(Error::Unsupported(format!(
                "dash-tract cannot run a {:?} artifact",
                art.backend
            )));
        }
        let mut reader = std::io::Cursor::new(&art.bytes);
        let plan = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|e| Error::Load(e.to_string()))?
            .into_optimized()
            .map_err(|e| Error::Load(e.to_string()))?
            .into_runnable()
            .map_err(|e| Error::Load(e.to_string()))?;
        Ok(Box::new(TractSession { plan }))
    }
}

impl Session for TractSession {
    fn run(&self, inputs: &[DashTensor]) -> DashResult<Vec<DashTensor>> {
        let mut tvals: TVec<TValue> = tvec!();
        for t in inputs {
            let h = t.to_host()?;
            if h.dtype != DType::F32 {
                return Err(Error::Shape(format!(
                    "dash-tract expects f32 inputs, got {:?}",
                    h.dtype
                )));
            }
            let data = h.as_f32()?;
            let arr = tract_ndarray::ArrayD::from_shape_vec(tract_ndarray::IxDyn(&h.shape), data)
                .map_err(|e| Error::Shape(e.to_string()))?;
            let tt: Tensor = arr.into();
            tvals.push(tt.into());
        }

        let result = self.plan.run(tvals).map_err(|e| Error::Run(e.to_string()))?;

        let mut outs = Vec::with_capacity(result.len());
        for r in result {
            let view = r
                .to_array_view::<f32>()
                .map_err(|e| Error::Run(e.to_string()))?;
            let shape = view.shape().to_vec();
            let data: Vec<f32> = view.iter().copied().collect();
            outs.push(DashTensor::Host(HostTensor::from_f32(shape, &data)));
        }
        Ok(outs)
    }
}
