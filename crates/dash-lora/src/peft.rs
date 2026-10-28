//! LoRA adapters in PEFT's format, and the Burn side of them. The format (PEFT 0.20's
//! `adapter_model.safetensors` and `adapter_config.json`, read and written back byte for byte,
//! the configuration, the module paths, applying an adapter to the caller's layers and writing
//! theirs out) is [`dash_peft`]'s, re-exported here; this module makes [`LoraLinear`] one of its
//! [`AdapterLayer`]s, the factors read onto a Burn device as f32 parameters and written out as f32.

use burn::module::Module;
use burn::tensor::{DType, Device, Tensor, TensorData};
pub use dash_peft::{
    AdapterLayer, ApplyOptions, CONFIG_FILE, FactorDtype, HostTensor, LoraWeights,
    PEFT_FORMAT_VERSION, PeftAdapter, PeftConfig, PeftError, WEIGHTS_FILE,
};

use crate::linear::{Lora, LoraLinear};

impl AdapterLayer for LoraLinear {
    type Lora = Lora;
    type Device = Device;

    fn in_features(&self) -> usize {
        LoraLinear::in_features(self)
    }

    fn out_features(&self) -> usize {
        LoraLinear::out_features(self)
    }

    /// The factors as f32 parameters on `device`; a frozen adapter (Burn's `Module::no_grad`)
    /// records nothing for autodiff.
    fn make_lora(
        weights: &LoraWeights,
        scale: f64,
        dropout: f64,
        trainable: bool,
        device: &Device,
    ) -> Result<Lora, PeftError> {
        let lora = Lora::from_data(data(&weights.a), data(&weights.b), scale, device)
            .with_dropout(dropout);
        Ok(if trainable { lora } else { lora.no_grad() })
    }

    fn put_lora(&mut self, name: &str, lora: Lora) -> Result<(), PeftError> {
        self.insert_adapter(name, lora);
        Ok(())
    }

    fn drop_lora(&mut self, name: &str) {
        self.remove_adapter(name);
    }

    fn lora_weights(&self, name: &str) -> Option<(LoraWeights, f64)> {
        let lora = self.adapter(name)?;
        let weights = LoraWeights {
            a: host(lora.a.val()),
            b: host(lora.b.val()),
        };
        Some((weights, lora.scale))
    }
}

/// A factor as Burn's host data, in its stored dtype.
fn data(tensor: &HostTensor) -> TensorData {
    let dtype = match tensor.dtype() {
        FactorDtype::F32 => DType::F32,
        FactorDtype::Bf16 => DType::BF16,
        FactorDtype::F16 => DType::F16,
        FactorDtype::F64 => DType::F64,
    };
    TensorData::from_bytes_vec(tensor.bytes().to_vec(), tensor.shape().to_vec(), dtype)
}

/// A factor as f32 host data.
fn host(tensor: Tensor<2>) -> HostTensor {
    let data = tensor.into_data().convert_dtype(DType::F32);
    let shape = data.shape().as_slice().to_vec();
    HostTensor::new(FactorDtype::F32, shape, data.as_bytes().to_vec())
        .expect("Burn's data holds its shape's elements")
}
