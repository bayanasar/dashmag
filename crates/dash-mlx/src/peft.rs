//! LoRA adapters in PEFT's format, and the MLX side of them. The format (PEFT 0.20's
//! `adapter_model.safetensors` and `adapter_config.json`, read and written back byte for byte,
//! the configuration, the module paths, applying an adapter to the caller's layers and writing
//! theirs out) is [`dash_peft`]'s, re-exported here, as dash-lora re-exports it for Burn; this
//! module makes [`LoraLinear`] one of its [`AdapterLayer`]s, the factors read as f32 arrays and
//! written out as f32. MLX makes arrays on no device, so the adapter's device is `()`:
//! `adapter.apply(layers, &())`.

pub use dash_peft::{
    AdapterLayer, ApplyOptions, CONFIG_FILE, FactorDtype, HostTensor, LoraWeights,
    PEFT_FORMAT_VERSION, PeftAdapter, PeftConfig, PeftError, WEIGHTS_FILE,
};
use mlx_rs::module::ModuleParameters;
use mlx_rs::{Array, Dtype};

use crate::lora::{Lora, LoraLinear};

fn mlx(e: mlx_rs::error::Exception) -> PeftError {
    PeftError::Invalid(e.what().to_string())
}

impl AdapterLayer for LoraLinear {
    type Lora = Lora;
    type Device = ();

    fn in_features(&self) -> usize {
        LoraLinear::in_features(self) as usize
    }

    fn out_features(&self) -> usize {
        LoraLinear::out_features(self) as usize
    }

    /// The factors as f32 arrays; a frozen adapter's parameters are frozen, so that MLX's
    /// `value_and_grad` over the trainable parameters leaves them out.
    fn make_lora(
        weights: &LoraWeights,
        scale: f64,
        dropout: f64,
        trainable: bool,
        _: &(),
    ) -> Result<Lora, PeftError> {
        let mut lora = Lora::new(&array(&weights.a)?, &array(&weights.b)?, scale)
            .map_err(mlx)?
            .with_dropout(dropout as f32);
        if !trainable {
            lora.freeze_parameters(true);
        }
        Ok(lora)
    }

    fn put_lora(&mut self, name: &str, lora: Lora) -> Result<(), PeftError> {
        self.insert_adapter(name, lora)
            .map(|_| ())
            .map_err(|e| PeftError::Mismatch(e.what().to_string()))
    }

    fn drop_lora(&mut self, name: &str) {
        self.remove_adapter(name);
    }

    fn lora_weights(&self, name: &str) -> Option<(LoraWeights, f64)> {
        let lora = self.adapter(name)?;
        let weights = LoraWeights {
            a: host(lora.a.as_ref()),
            b: host(lora.b.as_ref()),
        };
        Some((weights, lora.scale))
    }
}

/// A factor as an f32 array: bf16 and f16 widened exactly, f64 rounded to f32, as dash-lora reads
/// them.
fn array(tensor: &HostTensor) -> Result<Array, PeftError> {
    let bytes = tensor.bytes();
    let values: Vec<f32> = match tensor.dtype() {
        FactorDtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        FactorDtype::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
            .collect(),
        FactorDtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_le_bytes(*b).to_f32())
            .collect(),
        FactorDtype::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b) as f32)
            .collect(),
    };
    // A HostTensor holds exactly its shape's elements.
    let shape: Vec<i32> = tensor.shape().iter().map(|&d| d as i32).collect();
    Ok(Array::from_slice(&values, &shape))
}

/// A factor as f32 host data.
fn host(a: &Array) -> HostTensor {
    let a = a
        .as_dtype(Dtype::Float32)
        .and_then(|a| a.contiguous())
        .expect("an f32 factor");
    // `as_slice` fails on an empty array, for which MLX holds no data.
    let values: &[f32] = if a.size() == 0 { &[] } else { a.as_slice() };
    HostTensor::from_f32(a.shape().iter().map(|&d| d as usize).collect(), values)
        .expect("an array holds its shape's elements")
}
