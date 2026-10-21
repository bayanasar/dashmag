//! Model architectures as Burn modules.
//!
//! A model is a Rust type here, not a file: the graph is the code, and only weights move
//! between machines (as safetensors, or a published PyTorch checkpoint read directly).
//! The same module is trained and run, so there is no exported format to prove equal to
//! the model it came from.
//!
//! Parameter paths follow the published PyTorch definitions — `features.3.block.1.0.weight`
//! is the same tensor in both — so published weights load through burn-store's PyTorch
//! reader without a conversion table. Getting the names right is not enough to get the
//! numbers right; each architecture's module docs list the constants that differ from
//! Burn's defaults and fail silently when missed.

pub mod mobilenet_v3;
pub mod weights;

use burn::tensor::{Device, Tensor};
use dash_core::ModelSpec;

pub use mobilenet_v3::MobileNetV3Small;

/// The architectures this crate builds, by the name a [`ModelSpec`] gives them.
pub const ARCHITECTURES: &[&str] = &["mobilenet_v3_small"];

/// Any architecture this crate builds, chosen at run time from a [`ModelSpec`].
#[derive(Debug)]
pub enum Model {
    MobileNetV3Small(MobileNetV3Small),
}

impl Model {
    /// The architecture `spec` names, with one output per entry of `spec.outputs`, as wide
    /// as its class list. Weights are the initialiser's until loaded.
    pub fn build(spec: &ModelSpec, device: &Device) -> Result<Self, String> {
        match spec.arch.as_str() {
            "mobilenet_v3_small" => Ok(Model::MobileNetV3Small(MobileNetV3Small::new(
                &spec.widths(),
                0.2,
                device,
            ))),
            other => Err(format!(
                "architecture {other:?} is not built by this crate (it builds {ARCHITECTURES:?})"
            )),
        }
    }

    pub fn load_safetensors(&mut self, bytes: Vec<u8>) -> Result<(), String> {
        match self {
            Model::MobileNetV3Small(m) => weights::load_safetensors(m, bytes),
        }
    }

    pub fn to_safetensors(&self) -> Result<Vec<u8>, String> {
        match self {
            Model::MobileNetV3Small(m) => weights::to_safetensors(m),
        }
    }

    /// One `[N, width]` tensor of logits per output, in output order.
    pub fn forward(&self, x: Tensor<4>) -> Vec<Tensor<2>> {
        match self {
            Model::MobileNetV3Small(m) => m.forward(x),
        }
    }
}
