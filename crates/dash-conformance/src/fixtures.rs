//! The recorded fixtures, read where `dash-ops` keeps them: `crates/dash-ops/tests/fixtures`,
//! found from this crate's manifest.

use crate::{DType, HostTensor};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::PathBuf;

/// The path of a fixture file, `relative` to the fixtures' directory.
pub(crate) fn path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../dash-ops/tests/fixtures")
        .join(relative)
}

/// One recorded tensor: its shape and values.
#[derive(Clone, Debug)]
enum Values {
    /// f32 or bf16, as f32 values.
    Float(DType, Vec<f32>),
    /// Integers.
    Int(Vec<i64>),
    /// Booleans, stored as u8.
    Bool(Vec<bool>),
}

/// A safetensors file of recordings, read whole: its tensors (f32, bf16, i64 and booleans as u8)
/// and its metadata.
#[derive(Clone, Debug)]
pub(crate) struct File {
    name: String,
    tensors: HashMap<String, (Vec<usize>, Values)>,
    metadata: HashMap<String, String>,
}

impl File {
    /// Reads the fixture file at `relative`.
    ///
    /// # Panics
    ///
    /// If it cannot be read, or holds a tensor in a dtype other than f32, bf16, i64 and u8.
    pub(crate) fn read(relative: &str) -> Self {
        let path = path(relative);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let (_, header) = SafeTensors::read_metadata(&bytes).expect("a safetensors file");
        let metadata = header.metadata().clone().unwrap_or_default();
        let file = SafeTensors::deserialize(&bytes).expect("a safetensors file");
        let tensors = file
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                let data = view.data();
                let values = match view.dtype() {
                    Dtype::F32 => Values::Float(
                        DType::F32,
                        data.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| f32::from_le_bytes(*b))
                            .collect(),
                    ),
                    // A bf16 value is the top half of the f32 of the same value.
                    Dtype::BF16 => Values::Float(
                        DType::BF16,
                        data.as_chunks::<2>()
                            .0
                            .iter()
                            .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
                            .collect(),
                    ),
                    Dtype::I64 => Values::Int(
                        data.as_chunks::<8>()
                            .0
                            .iter()
                            .map(|b| i64::from_le_bytes(*b))
                            .collect(),
                    ),
                    Dtype::U8 => Values::Bool(data.iter().map(|&b| b != 0).collect()),
                    other => panic!("{relative}: {name} is {other:?}"),
                };
                (name, (view.shape().to_vec(), values))
            })
            .collect();
        Self {
            name: relative.to_string(),
            tensors,
            metadata,
        }
    }

    /// The file's metadata.
    pub(crate) fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Whether the file has a tensor `name`.
    pub(crate) fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The names of the file's tensors.
    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    fn get(&self, name: &str) -> &(Vec<usize>, Values) {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("{}: no fixture {name}", self.name))
    }

    /// A recorded float tensor, in its recorded dtype.
    ///
    /// # Panics
    ///
    /// If there is none of that name, or it does not hold floats.
    pub(crate) fn tensor(&self, name: &str) -> HostTensor {
        match self.get(name) {
            (shape, Values::Float(dtype, values)) => {
                HostTensor::new(*dtype, shape.clone(), values.clone())
            }
            (_, Values::Int(_) | Values::Bool(_)) => {
                panic!("{}: {name} holds no floats", self.name)
            }
        }
    }

    /// A recorded integer tensor's values.
    ///
    /// # Panics
    ///
    /// If there is none of that name, or it does not hold integers.
    pub(crate) fn ints(&self, name: &str) -> Vec<i64> {
        match self.get(name) {
            (_, Values::Int(values)) => values.clone(),
            (_, Values::Float(..) | Values::Bool(_)) => {
                panic!("{}: {name} holds no integers", self.name)
            }
        }
    }

    /// A recorded boolean tensor: its shape and values.
    ///
    /// # Panics
    ///
    /// If there is none of that name, or it does not hold booleans.
    pub(crate) fn bools(&self, name: &str) -> (Vec<usize>, Vec<bool>) {
        match self.get(name) {
            (shape, Values::Bool(values)) => (shape.clone(), values.clone()),
            _ => panic!("{}: {name} holds no booleans", self.name),
        }
    }
}
