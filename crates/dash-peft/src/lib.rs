//! LoRA adapters in PEFT's format: a directory with `adapter_model.safetensors` and
//! `adapter_config.json`, as PEFT 0.20's `save_pretrained` writes them, read and written with no
//! tensor library. dash-lora (Burn) and dash-mlx (MLX) give their layers adapters from it and
//! write theirs out through it, each implementing [`AdapterLayer`] for its layer.
//!
//! In the weights file every adapted module has two tensors,
//! `base_model.model.<module path>.lora_A.weight` `[r, in]` and `….lora_B.weight` `[out, r]`;
//! the file's metadata is `{"format": "pt"}`. The configuration gives `r`, `lora_alpha`,
//! `use_rslora`, `rank_pattern`, `alpha_pattern`, `lora_dropout`, `target_modules` and much else.
//! The scale of a module is `alpha / r` (`alpha / √r` with rsLoRA), with its `alpha` taken from
//! `alpha_pattern` when a key matches its path and its `r` read from the shape of its `lora_A`.
//!
//! Module paths are PEFT's (`model.layers.3.self_attn.q_proj`); which of the caller's layers a
//! path names is the caller's to say, since the model lives outside this crate:
//! [`PeftAdapter::apply`] and [`PeftAdapter::from_layers`] take `(path, layer)` pairs.
//!
//! One directory holds one adapter. A model holding several reads each from its own directory
//! under its own name ([`PeftAdapter::apply_with`], PEFT's `load_adapter(path, adapter_name)`)
//! and writes each back to its own ([`PeftAdapter::from_layers_named`]); PEFT's `save_pretrained`
//! puts [`DEFAULT_ADAPTER`] at the top of its directory and every other adapter in a
//! subdirectory named after it.
//!
//! Reading and writing back changes nothing: tensors keep their dtype and bytes, tensors that
//! are not LoRA factors (`modules_to_save`, trainable tokens) are carried through untouched,
//! configuration fields this crate does not interpret are kept, and both files are written as
//! PEFT writes them (the same `safetensors` serializer, the configuration as `json.dumps` with
//! `indent=2, sort_keys=True` prints it). Adapters whose forward pass is not plain LoRA (DoRA,
//! `lora_bias`, aLoRA and the other variants) are refused when read.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::Path;

use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde_json::{Map, Value};

/// The name PEFT gives an adapter when none is given, and the one each layer runs until told
/// otherwise; and how [`PeftAdapter::apply_with`] gives the caller's layers an adapter: both
/// dash-contract's, shared by every implementation.
pub use dash_contract::lora::{ApplyOptions, DEFAULT_ADAPTER};

/// The weights file in an adapter directory.
pub const WEIGHTS_FILE: &str = "adapter_model.safetensors";
/// The configuration file in an adapter directory.
pub const CONFIG_FILE: &str = "adapter_config.json";

/// What a `PeftModel` prefixes to its base model's module paths in the state dict it saves.
const PREFIX: &str = "base_model.model.";
const A_SUFFIX: &str = ".lora_A.weight";
const B_SUFFIX: &str = ".lora_B.weight";

/// The PEFT version whose format this crate writes, recorded as `peft_version` in a new
/// configuration.
pub const PEFT_FORMAT_VERSION: &str = "0.20.0";

/// Why an adapter could not be read, written or applied.
#[derive(Debug)]
pub enum PeftError {
    /// Reading or writing a file failed.
    Io(std::io::Error),
    /// `adapter_config.json` is not JSON, or not an object.
    Json(serde_json::Error),
    /// `adapter_model.safetensors` is not a safetensors file.
    SafeTensors(safetensors::SafeTensorError),
    /// The adapter uses a PEFT feature this crate does not implement.
    Unsupported(String),
    /// The files are not a consistent LoRA adapter.
    Invalid(String),
    /// The caller's layers and the adapter do not fit each other.
    Mismatch(String),
}

impl fmt::Display for PeftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PeftError::Io(e) => write!(f, "adapter file: {e}"),
            PeftError::Json(e) => write!(f, "{CONFIG_FILE}: {e}"),
            PeftError::SafeTensors(e) => write!(f, "{WEIGHTS_FILE}: {e:?}"),
            PeftError::Unsupported(what) => write!(f, "unsupported PEFT adapter: {what}"),
            PeftError::Invalid(what) => write!(f, "invalid PEFT adapter: {what}"),
            PeftError::Mismatch(what) => write!(f, "adapter and layers disagree: {what}"),
        }
    }
}

impl std::error::Error for PeftError {}

impl From<std::io::Error> for PeftError {
    fn from(e: std::io::Error) -> Self {
        PeftError::Io(e)
    }
}

impl From<serde_json::Error> for PeftError {
    fn from(e: serde_json::Error) -> Self {
        PeftError::Json(e)
    }
}

impl From<safetensors::SafeTensorError> for PeftError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        PeftError::SafeTensors(e)
    }
}

/// `adapter_config.json`, kept whole: the fields this crate reads have accessors, and every
/// field is written back as it was read.
#[derive(Clone, Debug, PartialEq)]
pub struct PeftConfig {
    fields: Map<String, Value>,
}

impl PeftConfig {
    /// A LoRA configuration with PEFT 0.20's default for every field but these three, as
    /// `LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_modules)` saved by a
    /// `PeftModel` writes it (`inference_mode` true, as saving sets it).
    pub fn new(r: usize, lora_alpha: f64, target_modules: &[&str]) -> Self {
        let alpha = if lora_alpha.fract() == 0.0 && lora_alpha.abs() < 2f64.powi(53) {
            Value::from(lora_alpha as i64)
        } else {
            Value::from(lora_alpha)
        };
        let fields = serde_json::json!({
            "alora_invocation_tokens": null,
            "alpha_pattern": {},
            "arrow_config": null,
            "auto_mapping": null,
            "base_model_name_or_path": null,
            "bias": "none",
            "corda_config": null,
            "ensure_weight_tying": false,
            "eva_config": null,
            "exclude_modules": null,
            "fan_in_fan_out": false,
            "inference_mode": true,
            "init_lora_weights": true,
            "layer_replication": null,
            "layers_pattern": null,
            "layers_to_transform": null,
            "loftq_config": {},
            "lora_alpha": alpha,
            "lora_bias": false,
            "lora_dropout": 0.0,
            "lora_ga_config": null,
            "megatron_config": null,
            "megatron_core": "megatron.core",
            "modules_to_save": null,
            "monteclora_config": null,
            "peft_type": "LORA",
            "peft_version": PEFT_FORMAT_VERSION,
            "qalora_group_size": 16,
            "r": r,
            "rank_pattern": {},
            "revision": null,
            "target_modules": target_modules,
            "target_parameters": null,
            "task_type": null,
            "trainable_token_indices": null,
            "use_bdlora": null,
            "use_dora": false,
            "use_qalora": false,
            "use_rslora": false,
            "velora_config": null,
        });
        let Value::Object(fields) = fields else {
            unreachable!("a JSON object literal")
        };
        Self { fields }
    }

    /// Parses `adapter_config.json`.
    pub fn from_json(text: &str) -> Result<Self, PeftError> {
        match serde_json::from_str(text)? {
            Value::Object(fields) => Ok(Self { fields }),
            _ => Err(PeftError::Invalid(format!(
                "{CONFIG_FILE} is not a JSON object"
            ))),
        }
    }

    /// The configuration as PEFT writes it, `json.dumps(config, indent=2, sort_keys=True)`:
    /// keys sorted at every level, two-space indentation, characters outside ASCII escaped, no
    /// final newline. Numbers keep their value; a float Python would print in exponent form
    /// (`1e-05`) is printed as Rust prints it (`1e-5`).
    pub fn to_json(&self) -> String {
        let sorted = sorted(&Value::Object(self.fields.clone()));
        let text = serde_json::to_string_pretty(&sorted).expect("a JSON value always serializes");
        ascii_escaped(&text)
    }

    /// A field, as JSON.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.fields.get(key)
    }

    /// Sets a field (`task_type`, `base_model_name_or_path`, `lora_dropout`, ...).
    pub fn set(&mut self, key: &str, value: impl Into<Value>) {
        self.fields.insert(key.to_string(), value.into());
    }

    /// `r`, the rank of every module `rank_pattern` does not name.
    pub fn r(&self) -> Result<usize, PeftError> {
        self.fields
            .get("r")
            .and_then(Value::as_u64)
            .map(|r| r as usize)
            .ok_or_else(|| PeftError::Invalid("`r` is missing or not a positive integer".into()))
    }

    /// `lora_alpha`, the alpha of every module `alpha_pattern` does not name.
    pub fn lora_alpha(&self) -> Result<f64, PeftError> {
        self.fields
            .get("lora_alpha")
            .and_then(Value::as_f64)
            .ok_or_else(|| PeftError::Invalid("`lora_alpha` is missing or not a number".into()))
    }

    /// Whether the scale is `alpha / √r` (rsLoRA) rather than `alpha / r`.
    pub fn use_rslora(&self) -> bool {
        self.fields.get("use_rslora").and_then(Value::as_bool) == Some(true)
    }

    /// `lora_dropout`, 0 when absent.
    pub fn lora_dropout(&self) -> f64 {
        self.fields
            .get("lora_dropout")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
    }

    /// The rank PEFT gives the module at `path`: `rank_pattern`'s value for the first key that
    /// matches, else `r`.
    pub fn rank_for(&self, path: &str) -> Result<usize, PeftError> {
        match pattern_value(&self.fields, "rank_pattern", path)? {
            Some(v) => v.as_u64().map(|r| r as usize).ok_or_else(|| {
                PeftError::Invalid(format!("rank_pattern value for {path} is not an integer"))
            }),
            None => self.r(),
        }
    }

    /// The alpha PEFT gives the module at `path`: `alpha_pattern`'s value for the first key
    /// that matches, else `lora_alpha`.
    pub fn alpha_for(&self, path: &str) -> Result<f64, PeftError> {
        match pattern_value(&self.fields, "alpha_pattern", path)? {
            Some(v) => v.as_f64().ok_or_else(|| {
                PeftError::Invalid(format!("alpha_pattern value for {path} is not a number"))
            }),
            None => self.lora_alpha(),
        }
    }

    /// The scale PEFT applies to the module at `path` whose adapter has rank `r`:
    /// `alpha / r`, or `alpha / √r` with rsLoRA, computed in f64 as PEFT does in Python.
    pub fn scale(&self, path: &str, r: usize) -> Result<f64, PeftError> {
        let alpha = self.alpha_for(path)?;
        Ok(if self.use_rslora() {
            alpha / (r as f64).sqrt()
        } else {
            alpha / r as f64
        })
    }

    /// Refuses adapters whose forward pass is not `x Wᵀ + scale · (x Aᵀ) Bᵀ`.
    pub fn check_supported(&self) -> Result<(), PeftError> {
        match self.fields.get("peft_type").and_then(Value::as_str) {
            Some("LORA") => {}
            other => {
                return Err(PeftError::Unsupported(format!(
                    "peft_type {other:?}; only LORA adapters are read"
                )));
            }
        }
        // Each of these, when set, changes the layer's forward pass: DoRA's magnitude vector,
        // a bias on lora_B, quantization-aware groups, block-diagonal factors, activated LoRA
        // (adapter on only after invocation tokens), Arrow routing, MonteCLoRA, VeLoRA, LoRA
        // on raw parameters rather than modules.
        for key in [
            "use_dora",
            "lora_bias",
            "use_qalora",
            "use_bdlora",
            "alora_invocation_tokens",
            "arrow_config",
            "monteclora_config",
            "velora_config",
            "target_parameters",
        ] {
            match self.fields.get(key) {
                None | Some(Value::Null) | Some(Value::Bool(false)) => {}
                Some(v) => {
                    return Err(PeftError::Unsupported(format!("{key} = {v}")));
                }
            }
        }
        if self.fields.get("init_lora_weights").and_then(Value::as_str) == Some("mica") {
            return Err(PeftError::Unsupported(
                "init_lora_weights = \"mica\" (a LoRA variant)".into(),
            ));
        }
        match self.fields.get("bias").and_then(Value::as_str) {
            None | Some("none") => {}
            Some(bias) => {
                return Err(PeftError::Unsupported(format!(
                    "bias = {bias:?}; the base layers' biases are not adapted here"
                )));
            }
        }
        Ok(())
    }
}

/// `value` with every object's keys in sorted order, whatever order the map keeps.
fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|k| (k.clone(), sorted(&map[k])))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

/// JSON text with every character outside ASCII written as `\uXXXX` (UTF-16, lower-case hex),
/// as Python's `json.dumps` writes it by default. Only string contents hold such characters.
fn ascii_escaped(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for unit in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

/// PEFT's `get_pattern_key`: the value of the first key, in the file's (sorted) order, for
/// which `re.match(rf"(.*\.)?({key})$", path)` holds. Keys are taken literally, a dot matching
/// a dot, which is what module paths mean by them; a key with any other regular-expression
/// syntax is refused rather than guessed at.
fn pattern_value<'a>(
    fields: &'a Map<String, Value>,
    pattern: &str,
    path: &str,
) -> Result<Option<&'a Value>, PeftError> {
    let Some(map) = fields.get(pattern) else {
        return Ok(None);
    };
    let map = match map {
        Value::Null => return Ok(None),
        Value::Object(map) => map,
        _ => return Err(PeftError::Invalid(format!("{pattern} is not an object"))),
    };
    for (key, value) in map {
        if key.contains([
            '^', '$', '*', '+', '?', '{', '}', '[', ']', '\\', '|', '(', ')',
        ]) {
            return Err(PeftError::Unsupported(format!(
                "{pattern} key {key:?} is a regular expression; only module paths are matched"
            )));
        }
        if path == key || path.ends_with(&format!(".{key}")) {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// The dtypes a LoRA factor is read and written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorDtype {
    /// IEEE f32.
    F32,
    /// bfloat16.
    Bf16,
    /// IEEE f16.
    F16,
    /// IEEE f64.
    F64,
}

impl FactorDtype {
    /// Bytes an element takes.
    pub fn size(self) -> usize {
        match self {
            FactorDtype::F32 => 4,
            FactorDtype::Bf16 | FactorDtype::F16 => 2,
            FactorDtype::F64 => 8,
        }
    }
}

/// A tensor as host data: its dtype, its shape and its elements' bytes, row-major and
/// little-endian, as the weights file holds them. The bytes always hold exactly the shape's
/// elements in the dtype: [`new`](Self::new) checks it, and the fields stay private.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    dtype: FactorDtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl HostTensor {
    /// A tensor from its elements' bytes. Errs unless `bytes` holds the product of `shape`
    /// elements of `dtype.size()` bytes each.
    pub fn new(dtype: FactorDtype, shape: Vec<usize>, bytes: Vec<u8>) -> Result<Self, PeftError> {
        let expected = shape
            .iter()
            .try_fold(dtype.size(), |n, &d| n.checked_mul(d));
        if expected != Some(bytes.len()) {
            return Err(PeftError::Invalid(format!(
                "{} bytes for a {dtype:?} tensor of shape {shape:?}",
                bytes.len()
            )));
        }
        Ok(Self {
            dtype,
            shape,
            bytes,
        })
    }

    /// An f32 tensor of `shape` from its values, row-major. Errs as [`new`](Self::new) does.
    pub fn from_f32(shape: Vec<usize>, values: &[f32]) -> Result<Self, PeftError> {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::new(FactorDtype::F32, shape, bytes)
    }

    /// The elements' dtype.
    pub fn dtype(&self) -> FactorDtype {
        self.dtype
    }

    /// The shape: `[r, in]` or `[out, r]` for a factor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The elements, `dtype().size()` bytes each.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// The two factors of one module's adapter, as stored in the file.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraWeights {
    /// `lora_A`, `[r, in]`.
    pub a: HostTensor,
    /// `lora_B`, `[out, r]`.
    pub b: HostTensor,
}

impl LoraWeights {
    /// The rank `r`.
    pub fn rank(&self) -> usize {
        self.a.shape[0]
    }
}

/// A tensor of the weights file that is not a LoRA factor, carried through byte for byte.
#[derive(Clone, Debug)]
struct RawTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// A backend's LoRA layer as [`PeftAdapter`] reads adapters into it and writes them out: its
/// weight's shape, how the backend makes an adapter from the file's factors, and the layer's
/// adapters by name. dash-lora implements it for its Burn layer, dash-mlx for its MLX one.
pub trait AdapterLayer {
    /// The backend's adapter: its two factors, its scale and its dropout.
    type Lora;
    /// What the backend makes an adapter on (a Burn device; `()` for a backend with none).
    type Device: ?Sized;

    /// `in`, the weight's columns.
    fn in_features(&self) -> usize;

    /// `out`, the weight's rows.
    fn out_features(&self) -> usize;

    /// An adapter from the file's factors, `a` `[r, in]` and `b` `[out, r]` in their stored
    /// dtype, with PEFT's scale and `lora_dropout`, its factors f32 and trainable or frozen.
    fn make_lora(
        weights: &LoraWeights,
        scale: f64,
        dropout: f64,
        trainable: bool,
        device: &Self::Device,
    ) -> Result<Self::Lora, PeftError>;

    /// Adds `lora` as the adapter named `name`, or puts it in the place of the one of that name.
    /// [`PeftAdapter::apply_with`] makes every adapter before it puts any, so that a refusal
    /// leaves the layers unchanged; a `put_lora` that errs partway leaves the layers before it
    /// changed. An implementation should not err here once `make_lora` has succeeded
    /// (dash-lora's never does).
    fn put_lora(&mut self, name: &str, lora: Self::Lora) -> Result<(), PeftError>;

    /// Removes the adapter named `name`, if the layer holds one.
    fn drop_lora(&mut self, name: &str);

    /// The adapter named `name`, if the layer holds one: its factors as f32 host data,
    /// `[r, in]` and `[out, r]`, and its scale. f32 whatever dtype the factors were read in, since
    /// a layer's factors are f32: an adapter read from a bf16 file and written back from the
    /// layers is an f32 file, with the same values.
    fn lora_weights(&self, name: &str) -> Option<(LoraWeights, f64)>;
}

/// A LoRA adapter in PEFT's format.
#[derive(Clone, Debug)]
pub struct PeftAdapter {
    /// `adapter_config.json`.
    pub config: PeftConfig,
    modules: BTreeMap<String, LoraWeights>,
    other: BTreeMap<String, RawTensor>,
    metadata: Option<HashMap<String, String>>,
    prefixed: bool,
}

impl PeftAdapter {
    /// An adapter with no modules yet, for [`insert`](Self::insert).
    pub fn new(config: PeftConfig) -> Self {
        Self {
            config,
            modules: BTreeMap::new(),
            other: BTreeMap::new(),
            metadata: Some(HashMap::from([("format".to_string(), "pt".to_string())])),
            prefixed: true,
        }
    }

    /// Reads an adapter directory written by PEFT (or by [`write`](Self::write)).
    pub fn read(dir: impl AsRef<Path>) -> Result<Self, PeftError> {
        let dir = dir.as_ref();
        let config = std::fs::read_to_string(dir.join(CONFIG_FILE))?;
        let weights = std::fs::read(dir.join(WEIGHTS_FILE))?;
        Self::from_bytes(&config, &weights)
    }

    /// Reads an adapter from the contents of its two files.
    pub fn from_bytes(config_json: &str, weights: &[u8]) -> Result<Self, PeftError> {
        let config = PeftConfig::from_json(config_json)?;
        config.check_supported()?;
        let (_, header) = SafeTensors::read_metadata(weights)?;
        let metadata = header.metadata().clone();
        let tensors = SafeTensors::deserialize(weights)?;

        let mut a = BTreeMap::new();
        let mut b = BTreeMap::new();
        let mut other = BTreeMap::new();
        let mut prefixed = BTreeSet::new();
        for (name, view) in tensors.tensors() {
            let (factor, path) = if let Some(path) = name.strip_suffix(A_SUFFIX) {
                (&mut a, path)
            } else if let Some(path) = name.strip_suffix(B_SUFFIX) {
                (&mut b, path)
            } else {
                other.insert(
                    name.clone(),
                    RawTensor {
                        dtype: view.dtype(),
                        shape: view.shape().to_vec(),
                        data: view.data().to_vec(),
                    },
                );
                continue;
            };
            let path = match path.strip_prefix(PREFIX) {
                Some(path) => {
                    prefixed.insert(true);
                    path
                }
                None => {
                    prefixed.insert(false);
                    path
                }
            };
            factor.insert(path.to_string(), to_host(&name, &view)?);
        }
        if prefixed.len() > 1 {
            return Err(PeftError::Invalid(format!(
                "some LoRA keys start with {PREFIX:?} and some do not"
            )));
        }

        let mut modules = BTreeMap::new();
        for (path, a) in a {
            let b = b
                .remove(&path)
                .ok_or_else(|| PeftError::Invalid(format!("{path} has lora_A but no lora_B")))?;
            let (sa, sb) = (a.shape.as_slice(), b.shape.as_slice());
            if sa.len() != 2 || sb.len() != 2 || sa[0] != sb[1] {
                return Err(PeftError::Invalid(format!(
                    "{path}: lora_A {sa:?} and lora_B {sb:?} are not [r, in] and [out, r]"
                )));
            }
            // The rank is the shape's; the configuration is checked against it where its
            // patterns can be read (a regular-expression key cannot, and is left to the shape).
            match config.rank_for(&path) {
                Ok(expected) if sa[0] != expected => {
                    return Err(PeftError::Invalid(format!(
                        "{path} has rank {} where the configuration gives {expected}",
                        sa[0]
                    )));
                }
                Ok(_) | Err(PeftError::Unsupported(_)) => {}
                Err(e) => return Err(e),
            }
            modules.insert(path, LoraWeights { a, b });
        }
        if let Some(path) = b.keys().next() {
            return Err(PeftError::Invalid(format!(
                "{path} has lora_B but no lora_A"
            )));
        }

        Ok(Self {
            config,
            modules,
            other,
            metadata,
            prefixed: prefixed.into_iter().next().unwrap_or(true),
        })
    }

    /// Writes `adapter_model.safetensors` and `adapter_config.json` into `dir`, creating it.
    pub fn write(&self, dir: impl AsRef<Path>) -> Result<(), PeftError> {
        let dir = dir.as_ref();
        let (config, weights) = self.to_bytes()?;
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join(WEIGHTS_FILE), weights)?;
        std::fs::write(dir.join(CONFIG_FILE), config)?;
        Ok(())
    }

    /// The contents of `adapter_config.json` and of `adapter_model.safetensors`.
    pub fn to_bytes(&self) -> Result<(String, Vec<u8>), PeftError> {
        let prefix = if self.prefixed { PREFIX } else { "" };
        let mut views = Vec::with_capacity(2 * self.modules.len() + self.other.len());
        for (path, w) in &self.modules {
            views.push((format!("{prefix}{path}{A_SUFFIX}"), view(&w.a)?));
            views.push((format!("{prefix}{path}{B_SUFFIX}"), view(&w.b)?));
        }
        for (name, raw) in &self.other {
            let view = TensorView::new(raw.dtype, raw.shape.clone(), &raw.data)?;
            views.push((name.clone(), view));
        }
        let weights = safetensors::serialize(views, self.metadata.clone())?;
        Ok((self.config.to_json(), weights))
    }

    /// The adapted modules' paths and weights, in path order.
    pub fn modules(&self) -> impl Iterator<Item = (&str, &LoraWeights)> {
        self.modules.iter().map(|(path, w)| (path.as_str(), w))
    }

    /// The weights of the module at `path`, if it is adapted.
    pub fn module(&self, path: &str) -> Option<&LoraWeights> {
        self.modules.get(path)
    }

    /// Adds or replaces the weights of the module at `path`.
    pub fn insert(&mut self, path: impl Into<String>, weights: LoraWeights) {
        self.modules.insert(path.into(), weights);
    }

    /// Names of the tensors in the weights file that are not LoRA factors, such as the
    /// weights of `modules_to_save`. They are written back as read, and never applied.
    pub fn other_tensors(&self) -> impl Iterator<Item = &str> {
        self.other.keys().map(String::as_str)
    }

    /// The scale of the module at `path`, from the configuration and its rank.
    pub fn scale(&self, path: &str) -> Result<f64, PeftError> {
        let w = self
            .modules
            .get(path)
            .ok_or_else(|| PeftError::Mismatch(format!("the adapter has no module {path}")))?;
        self.config.scale(path, w.rank())
    }

    /// The adapter of the module at `path` as a trainable adapter of the backend `L` on
    /// `device` (f32 factors, PEFT's scale and `lora_dropout`), or `None` if the module is not
    /// adapted.
    pub fn lora<L: AdapterLayer>(
        &self,
        path: &str,
        device: &L::Device,
    ) -> Result<Option<L::Lora>, PeftError> {
        self.make_lora::<L>(path, true, device)
    }

    fn make_lora<L: AdapterLayer>(
        &self,
        path: &str,
        trainable: bool,
        device: &L::Device,
    ) -> Result<Option<L::Lora>, PeftError> {
        let Some(w) = self.modules.get(path) else {
            return Ok(None);
        };
        let scale = self.config.scale(path, w.rank())?;
        let dropout = self.config.lora_dropout();
        L::make_lora(w, scale, dropout, trainable, device).map(Some)
    }

    /// Gives each of the caller's layers its adapter, as [`DEFAULT_ADAPTER`], trainable:
    /// [`apply_with`](Self::apply_with) with the default [`ApplyOptions`].
    pub fn apply<'a, L: AdapterLayer + 'a, S: AsRef<str>>(
        &self,
        layers: impl IntoIterator<Item = (S, &'a mut L)>,
        device: &L::Device,
    ) -> Result<(), PeftError> {
        self.apply_with(layers, device, &ApplyOptions::default())
    }

    /// Gives each of the caller's layers its adapter, under the name and with the trainability
    /// `options` give, beside the adapters the layer holds by other names: `layers` pairs PEFT
    /// module paths with the layers they name. A named layer the adapter does not cover is left
    /// without an adapter of that name (one it held is removed), so that the adapter of that name
    /// is this one in every layer. The layers' choice of adapter is left as it is: as with
    /// PEFT's `load_adapter`, the new adapter runs once chosen. Errs, before changing any layer,
    /// if a module's shapes do not fit its layer, or if the adapter holds weights no named layer
    /// takes: a LoRA module nobody named, or tensors such as `modules_to_save` (place those with
    /// [`lora`](Self::lora) and your own code).
    pub fn apply_with<'a, L: AdapterLayer + 'a, S: AsRef<str>>(
        &self,
        layers: impl IntoIterator<Item = (S, &'a mut L)>,
        device: &L::Device,
        options: &ApplyOptions,
    ) -> Result<(), PeftError> {
        let layers: Vec<(S, &'a mut L)> = layers.into_iter().collect();
        let mut unclaimed: BTreeSet<&str> = self.modules.keys().map(String::as_str).collect();
        for (path, layer) in &layers {
            let path = path.as_ref();
            if let Some(w) = self.modules.get(path) {
                let (sa, sb) = (w.a.shape.as_slice(), w.b.shape.as_slice());
                let (out, inp) = (layer.out_features(), layer.in_features());
                if sa[1] != inp || sb[0] != out {
                    return Err(PeftError::Mismatch(format!(
                        "{path}: lora_A {sa:?} and lora_B {sb:?} do not fit a [{out}, {inp}] weight"
                    )));
                }
                unclaimed.remove(path);
            }
        }
        if !unclaimed.is_empty() || !self.other.is_empty() {
            let mut names: Vec<&str> = unclaimed.into_iter().collect();
            names.extend(self.other.keys().map(String::as_str));
            return Err(PeftError::Mismatch(format!(
                "no layer takes {}",
                names.join(", ")
            )));
        }
        let mut loras = Vec::with_capacity(layers.len());
        for (path, _) in &layers {
            loras.push(self.make_lora::<L>(path.as_ref(), options.trainable, device)?);
        }
        for ((_, layer), lora) in layers.into_iter().zip(loras) {
            match lora {
                Some(lora) => layer.put_lora(&options.name, lora)?,
                None => layer.drop_lora(&options.name),
            }
        }
        Ok(())
    }

    /// An adapter holding the caller's layers' [`DEFAULT_ADAPTER`]:
    /// [`from_layers_named`](Self::from_layers_named) with that name.
    pub fn from_layers<'a, L: AdapterLayer + 'a, S: AsRef<str>>(
        config: PeftConfig,
        layers: impl IntoIterator<Item = (S, &'a L)>,
    ) -> Result<Self, PeftError> {
        Self::from_layers_named(config, DEFAULT_ADAPTER, layers)
    }

    /// An adapter holding the caller's layers' adapter named `name` under the PEFT module paths
    /// given, as f32 (an adapter that was read in bf16 comes back f32), with `config`. Layers
    /// without an adapter of that name are skipped. Errs if a layer's scale is not the one
    /// `config` gives its path and rank, since PEFT would load it with that one.
    pub fn from_layers_named<'a, L: AdapterLayer + 'a, S: AsRef<str>>(
        config: PeftConfig,
        name: &str,
        layers: impl IntoIterator<Item = (S, &'a L)>,
    ) -> Result<Self, PeftError> {
        config.check_supported()?;
        let mut adapter = Self::new(config);
        for (path, layer) in layers {
            let path = path.as_ref();
            let Some((weights, scale)) = layer.lora_weights(name) else {
                continue;
            };
            let r = weights.rank();
            let expected = adapter.config.scale(path, r)?;
            if (expected - scale).abs() > 1e-12 * expected.abs() {
                return Err(PeftError::Mismatch(format!(
                    "{path}: the layer's scale is {scale} where the configuration gives {expected}"
                )));
            }
            match adapter.config.rank_for(path) {
                Ok(expected) if r != expected => {
                    return Err(PeftError::Mismatch(format!(
                        "{path}: rank {r} where the configuration gives {expected}"
                    )));
                }
                Ok(_) | Err(PeftError::Unsupported(_)) => {}
                Err(e) => return Err(e),
            }
            adapter.insert(path, weights);
        }
        Ok(adapter)
    }
}

fn to_host(name: &str, view: &TensorView<'_>) -> Result<HostTensor, PeftError> {
    let dtype = match view.dtype() {
        Dtype::F32 => FactorDtype::F32,
        Dtype::BF16 => FactorDtype::Bf16,
        Dtype::F16 => FactorDtype::F16,
        Dtype::F64 => FactorDtype::F64,
        other => {
            return Err(PeftError::Unsupported(format!(
                "{name} is {other:?}; LoRA factors are read as f32, bf16, f16 or f64"
            )));
        }
    };
    HostTensor::new(dtype, view.shape().to_vec(), view.data().to_vec()).map_err(|e| match e {
        PeftError::Invalid(what) => PeftError::Invalid(format!("{name}: {what}")),
        e => e,
    })
}

fn view(tensor: &HostTensor) -> Result<TensorView<'_>, PeftError> {
    let dtype = match tensor.dtype {
        FactorDtype::F32 => Dtype::F32,
        FactorDtype::Bf16 => Dtype::BF16,
        FactorDtype::F16 => Dtype::F16,
        FactorDtype::F64 => Dtype::F64,
    };
    Ok(TensorView::new(dtype, tensor.shape.clone(), &tensor.bytes)?)
}
