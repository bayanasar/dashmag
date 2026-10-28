//! What LoRA adapters in PEFT's format share whatever reads them: the name of the adapter PEFT
//! makes and saves by default, and how an adapter read from PEFT's files is given to a model's
//! layers, under a name and trainable or frozen (PEFT's `load_adapter(path, adapter_name,
//! is_trainable)`).
//!
//! The files themselves (`adapter_config.json`, `adapter_model.safetensors`) are read and written
//! by `dash-lora`, whose configuration keeps the JSON as it was read.

/// PEFT's default adapter name: the adapter `get_peft_model` and `PeftModel.from_pretrained`
/// create unless told otherwise, and the one `save_pretrained` writes at the top of its directory
/// (every other adapter in a subdirectory named after it).
pub const DEFAULT_ADAPTER: &str = "default";

/// How an adapter read in PEFT's format is given to a model's layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplyOptions {
    /// The name the adapter takes in each layer, PEFT's `adapter_name`: [`DEFAULT_ADAPTER`] by
    /// default.
    pub name: String,
    /// Whether its factors train (require a gradient), PEFT's `is_trainable`: true by default.
    /// A frozen adapter records nothing for autodiff, gets no gradient and so no optimizer state;
    /// its dropout, if it has any, still acts in training, as PEFT's does.
    pub trainable: bool,
}

impl Default for ApplyOptions {
    fn default() -> Self {
        Self {
            name: DEFAULT_ADAPTER.to_string(),
            trainable: true,
        }
    }
}
