//! `dash-burn` — the [`Runtime`] for models that are code.
//!
//! Every other backend is handed a graph (an ONNX file, a compiled module, a vendor
//! engine) and runs it. This one is handed weights: the artifact's [`ArtifactMeta::model`]
//! names an architecture compiled into this binary (`dash-models`), the runtime builds it
//! on its device, loads the safetensors into it, and runs the module itself — the same
//! module a trainer updates. A model it does not have is refused through
//! [`Capability::admits`], so [`dash_core::Registry::explain`] can say why.
//!
//! One runtime per Burn device, each advertising its own accelerator class, so the
//! registry chooses among devices exactly as it chooses among backends. Which devices
//! exist is a build decision (one Cargo feature each): Flex is pure Rust and the only one
//! a static musl binary can carry; CUDA, ROCm and Metal load their vendor's libraries at
//! run time and need a dynamically linked build.
//!
//! On CUDA devices of compute capability 8.0 and later, Burn 0.22 runs f32 matmuls and
//! convolutions in TF32, and offers no switch to turn that off. Results there are not
//! fp32 results and are compared as such.
//!
//! Outputs come back as host tensors for now. Burn's tensors live on their device, which
//! makes this the first backend that could hand back `Tensor::Device`; that is a later step.

use burn::tensor::{Device, Tensor as BurnTensor, TensorData};
use dash_core::{
    AccelClass, Artifact, BackendKind, Capability, DType, DTypeSet, Error, HostTensor,
    Result as DashResult, Runtime, Session, Tensor as DashTensor,
};
use dash_models::Model;

/// A Burn device behind the `dash-core` trait.
pub struct BurnRuntime {
    device: Device,
    cap: Capability,
}

impl BurnRuntime {
    fn on(device: Device, accel: AccelClass) -> Self {
        BurnRuntime {
            device,
            cap: Capability {
                accel,
                dtypes: DTypeSet::of(&[DType::F32]),
                // The module is built for no particular input size; any batch and any
                // spatial size the architecture accepts will run.
                dynamic_shapes: true,
                quantized: false,
                no_std: false,
                architectures: dash_models::ARCHITECTURES,
            },
        }
    }

    /// Flex: Burn's pure-Rust CPU backend.
    #[cfg(feature = "flex")]
    pub fn flex() -> Self {
        Self::on(Device::flex(), AccelClass::Cpu)
    }

    /// The `index`-th CUDA device.
    #[cfg(feature = "cuda")]
    pub fn cuda(index: usize) -> Self {
        Self::on(Device::cuda(index), AccelClass::Cuda)
    }

    /// The `index`-th ROCm device.
    #[cfg(feature = "rocm")]
    pub fn rocm(index: usize) -> Self {
        Self::on(Device::rocm(index), AccelClass::Rocm)
    }

    /// A Metal device.
    #[cfg(feature = "metal")]
    pub fn metal(kind: burn::tensor::DeviceKind) -> Self {
        Self::on(Device::metal(kind), AccelClass::Metal)
    }
}

impl Runtime for BurnRuntime {
    fn kind(&self) -> BackendKind {
        BackendKind::Burn
    }

    fn capability(&self) -> &Capability {
        &self.cap
    }

    fn load(&self, art: &Artifact) -> DashResult<Box<dyn Session>> {
        if art.backend != BackendKind::Burn {
            return Err(Error::Unsupported(format!(
                "dash-burn cannot run a {:?} artifact",
                art.backend
            )));
        }
        if let Err(why) = self.cap.admits(&art.meta) {
            return Err(Error::Unsupported(format!("dash-burn: {why}")));
        }
        // `admits` has already refused an artifact without a model description.
        let spec = art
            .meta
            .model
            .as_ref()
            .expect("admitted without meta.model");
        let mut model = Model::build(spec, &self.device).map_err(Error::Load)?;
        model
            .load_safetensors(art.bytes.clone())
            .map_err(Error::Load)?;
        Ok(Box::new(BurnSession {
            model,
            device: self.device.clone(),
        }))
    }
}

struct BurnSession {
    model: Model,
    device: Device,
}

impl Session for BurnSession {
    fn run(&self, inputs: &[DashTensor]) -> DashResult<Vec<DashTensor>> {
        let [input] = inputs else {
            return Err(Error::Shape(format!(
                "dash-burn models take one input, got {}",
                inputs.len()
            )));
        };
        let h = input.host()?;
        let [n, c, height, width] = h.shape[..] else {
            return Err(Error::Shape(format!(
                "dash-burn expects an NCHW input, got shape {:?}",
                h.shape
            )));
        };
        let data = h.as_f32()?;
        let x =
            BurnTensor::<4>::from_data(TensorData::new(data, [n, c, height, width]), &self.device);
        self.model
            .forward(x)
            .into_iter()
            .map(|y| {
                let shape = y.dims().to_vec();
                let v = y
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| Error::Run(format!("{e:?}")))?;
                Ok(DashTensor::Host(HostTensor::from_f32(shape, &v)))
            })
            .collect()
    }
}
