//! The graph file: a [`Graph`] in JSON, as an agent writes, reads back and edits it.
//!
//! The file is one object. `format` is `"mag-graph"` and `version` is `1`; then the graph's
//! fields: `job`, `inputs`, `parts`, `chains`, `edges` and `note`. The file's version is its own,
//! apart from the wire form's and from the crate's. Enumerations are written in
//! snake case (`"replicable"`, `"optimiser_step"`, `"bf16"`), and a pin as `{"member": name}` or
//! `{"backend": "rocm"}`. A job states no objective: its need and its priority say what it
//! wants.
//!
//! Reading checks `format` and `version` first, so that a file of a newer version is refused as
//! that rather than for fields this reader does not know. It refuses a key repeated in one object,
//! at any depth: JSON allows one and keeps the last, so a file saying `"local_only": true` and then
//! `false` would otherwise be read by its last word, and a schema cannot see it. It refuses
//! unknown fields, so that a misspelt field is an error and not a silent default, and then runs
//! [`Graph::check`]. Fields
//! with a default may be left out: `inputs`, `chains` and `edges` (empty), a job's `need`
//! (none), a part's `mark` (atomic), `weights` (none) and `pin` (none), an edge's `to` (a result
//! of the job), `release` (false) and every `note` (none). An input's `local_only` and an edge's
//! `dtype` have no default. An option is present or left out; `null` is refused.
//!
//! [`graph_schema`] is the file's JSON Schema, generated from the types it is read into, so that
//! an agent in any language can have its output constrained to a valid file before it submits.
//! It is kept in the repository at `schemas/mag-graph.schema.json`, and a test fails when the two
//! differ.
//!
//! Writing gives `format` and `version` first and the fields in a fixed order, every field but
//! an absent option, two-space indented with a final newline, so that reading what was written
//! and writing it again gives the same text.
//!
//! The pool file is read and written the same way: a [`Description`] in JSON, `format`
//! `"mag-pool"`, `version` 1, as `describe` gives it, so that a pool can be written to a file and
//! planned against elsewhere. [`pool_schema`] is its schema, kept at
//! `schemas/mag-pool.schema.json`.

use core::fmt;
use std::string::{String, ToString};

use serde_json::Value;

use crate::{
    Description, GRAPH_FORMAT as FORMAT, GRAPH_VERSION as VERSION, Graph, POOL_FORMAT,
    POOL_VERSION, Refusal,
};

/// Why a graph file was not read.
#[derive(Debug)]
pub enum ReadError {
    /// Not JSON, or not a graph: a field missing, unknown or of the wrong type.
    Json(serde_json::Error),
    /// `format` is not `"mag-graph"`, or the file is not an object. The JSON found, if any.
    Format { found: Option<String> },
    /// `version` is not 1. The JSON found, if any.
    Version { found: Option<String> },
    /// A key appears twice in one object. Where, as a path such as `inputs[0].local_only`.
    RepeatedKey { at: String },
    /// The graph was read but does not pass [`Graph::check`].
    Refused(Refusal),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Json(e) => write!(f, "not a graph file: {e}"),
            ReadError::Format { found: Some(found) } => {
                write!(f, "the format is {found}, not \"{FORMAT}\"")
            }
            ReadError::Format { found: None } => {
                write!(f, "not a graph file: no \"format\": \"{FORMAT}\"")
            }
            ReadError::Version { found: Some(found) } => {
                write!(
                    f,
                    "the version is {found}; this reader reads version {VERSION}"
                )
            }
            ReadError::Version { found: None } => write!(f, "the graph file has no version"),
            ReadError::RepeatedKey { at } => {
                write!(
                    f,
                    "`{at}` is given twice; a key may appear once in an object"
                )
            }
            ReadError::Refused(r) => write!(f, "the graph is refused: {r}"),
        }
    }
}

impl core::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            ReadError::Json(e) => Some(e),
            ReadError::Refused(r) => Some(r),
            _ => None,
        }
    }
}

/// Why the pool file was not read.
#[derive(Debug)]
pub enum PoolFileError {
    /// Not JSON, or not a pool's description: a field missing, unknown, `null` or of the wrong
    /// type.
    Json(serde_json::Error),
    /// `format` is not `"mag-pool"`, or the file is not an object. The JSON found, if any.
    Format { found: Option<String> },
    /// `version` is not 1. The JSON found, if any.
    Version { found: Option<String> },
    /// A key appears twice in one object. Where, as a path such as `offers[0].memory_bytes`.
    RepeatedKey { at: String },
}

impl fmt::Display for PoolFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolFileError::Json(e) => write!(f, "not a pool file: {e}"),
            PoolFileError::Format { found: Some(found) } => {
                write!(f, "the format is {found}, not \"{POOL_FORMAT}\"")
            }
            PoolFileError::Format { found: None } => {
                write!(f, "not a pool file: no \"format\": \"{POOL_FORMAT}\"")
            }
            PoolFileError::Version { found: Some(found) } => write!(
                f,
                "the version is {found}; this reader reads version {POOL_VERSION}"
            ),
            PoolFileError::Version { found: None } => write!(f, "the pool file has no version"),
            PoolFileError::RepeatedKey { at } => {
                write!(
                    f,
                    "`{at}` is given twice; a key may appear once in an object"
                )
            }
        }
    }
}

impl core::error::Error for PoolFileError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            PoolFileError::Json(e) => Some(e),
            _ => None,
        }
    }
}

/// Why a file was not read, before it is read as the type it holds.
enum Bad {
    Json(serde_json::Error),
    Format(Option<String>),
    Version(Option<String>),
    RepeatedKey(String),
}

/// A file's fields after its `format` and `version`, those checked first so that a file of a
/// newer version is refused as that rather than for fields this reader does not know; then any
/// repeated key.
fn read(text: &str, format: &str, version: u64) -> Result<Value, Bad> {
    let mut value: Value = serde_json::from_str(text).map_err(Bad::Json)?;
    let Value::Object(fields) = &mut value else {
        return Err(Bad::Format(None));
    };
    match fields.remove("format") {
        Some(Value::String(f)) if f == format => {}
        other => return Err(Bad::Format(other.map(|v| v.to_string()))),
    }
    match fields.remove("version") {
        Some(Value::Number(v)) if v.as_u64() == Some(version) => {}
        other => return Err(Bad::Version(other.map(|v| v.to_string()))),
    }
    let repeats: NoRepeats = serde_json::from_str(text).map_err(Bad::Json)?;
    if let Some(at) = repeats.0 {
        return Err(Bad::RepeatedKey(at));
    }
    Ok(value)
}

#[derive(serde::Serialize)]
struct File<'a, T> {
    format: &'static str,
    version: u64,
    #[serde(flatten)]
    body: &'a T,
}

/// A file: `format` and `version` first, then the fields in a fixed order, two-space indented
/// with a final newline. A float that is not finite is written as null, which reading refuses.
fn write<T: serde::Serialize>(format: &'static str, version: u64, body: &T) -> String {
    let file = File {
        format,
        version,
        body,
    };
    let mut text = serde_json::to_string_pretty(&file).expect("a file always serialises to JSON");
    text.push('\n');
    text
}

impl Graph {
    /// Reads a graph file and checks the graph.
    pub fn from_json(text: &str) -> Result<Graph, ReadError> {
        let value = read(text, FORMAT, VERSION).map_err(|bad| match bad {
            Bad::Json(e) => ReadError::Json(e),
            Bad::Format(found) => ReadError::Format { found },
            Bad::Version(found) => ReadError::Version { found },
            Bad::RepeatedKey(at) => ReadError::RepeatedKey { at },
        })?;
        let graph: Graph = serde_json::from_value(value).map_err(ReadError::Json)?;
        graph.check().map_err(ReadError::Refused)?;
        Ok(graph)
    }

    /// Writes the graph as a graph file. The graph is written as it is, checked or not.
    pub fn to_json(&self) -> String {
        write(FORMAT, VERSION, self)
    }
}

impl Description {
    /// Reads a pool file: a pool's description, as `describe` gives it.
    pub fn from_json(text: &str) -> Result<Description, PoolFileError> {
        let value = read(text, POOL_FORMAT, POOL_VERSION).map_err(|bad| match bad {
            Bad::Json(e) => PoolFileError::Json(e),
            Bad::Format(found) => PoolFileError::Format { found },
            Bad::Version(found) => PoolFileError::Version { found },
            Bad::RepeatedKey(at) => PoolFileError::RepeatedKey { at },
        })?;
        serde_json::from_value(value).map_err(PoolFileError::Json)
    }

    /// Writes the description as a pool file, as the graph file is written.
    pub fn to_json(&self) -> String {
        write(POOL_FORMAT, POOL_VERSION, self)
    }
}

/// A file's JSON Schema (draft 2020-12), as text with a final newline: generated from the type a
/// file is read into, with `format` and `version` as constants, unknown fields refused, and no
/// `null` anywhere, since the reader refuses one wherever it stands.
fn file_schema<T: schemars::JsonSchema>(format: &str, version: u64) -> String {
    let generator = schemars::generate::SchemaSettings::draft2020_12().into_generator();
    let mut schema = generator.into_root_schema_for::<T>().to_value();
    strip_null(&mut schema);
    let root = schema
        .as_object_mut()
        .expect("a schema for a struct is an object");
    root.insert(
        "title".into(),
        format!("{format}, version {version}").into(),
    );
    let properties = root
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .expect("a struct's schema has properties");
    properties.insert("format".into(), serde_json::json!({ "const": format }));
    properties.insert("version".into(), serde_json::json!({ "const": version }));
    let required = root
        .get_mut("required")
        .and_then(Value::as_array_mut)
        .expect("a file has required fields");
    required.insert(0, "version".into());
    required.insert(0, "format".into());
    let mut text = serde_json::to_string_pretty(&schema).expect("a schema serialises");
    text.push('\n');
    text
}

/// The graph file's JSON Schema, kept at `schemas/mag-graph.schema.json`.
pub fn graph_schema() -> String {
    file_schema::<Graph>(FORMAT, VERSION)
}

/// The pool file's JSON Schema, kept at `schemas/mag-pool.schema.json`.
pub fn pool_schema() -> String {
    file_schema::<Description>(POOL_FORMAT, POOL_VERSION)
}

/// Takes `null` out of every type a schema allows, as `Option<T>` puts it there.
pub(crate) fn strip_null(v: &mut Value) {
    match v {
        Value::Object(map) => {
            if let Some(Value::Array(types)) = map.get_mut("type") {
                types.retain(|t| t != "null");
                if types.len() == 1 {
                    let only = types.remove(0);
                    map.insert("type".into(), only);
                }
            }
            for key in ["anyOf", "oneOf"] {
                if let Some(Value::Array(alternatives)) = map.get_mut(key) {
                    alternatives.retain(|a| a.get("type").and_then(Value::as_str) != Some("null"));
                    if alternatives.len() == 1 {
                        let only = alternatives.remove(0);
                        map.remove(key);
                        if let Value::Object(inner) = only {
                            map.extend(inner);
                        }
                    }
                }
            }
            map.values_mut().for_each(strip_null);
        }
        Value::Array(items) => items.iter_mut().for_each(strip_null),
        _ => {}
    }
}

/// Any JSON, read for one thing: the first key repeated in an object, at any depth, with its path.
/// Parsing into a [`Value`] keeps the last of a repeated key and says nothing, so this reads the
/// text itself.
pub(crate) struct NoRepeats(pub(crate) Option<String>);

impl<'de> serde::Deserialize<'de> for NoRepeats {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(NoRepeatsVisitor)
    }
}

struct NoRepeatsVisitor;

impl<'de> serde::de::Visitor<'de> for NoRepeatsVisitor {
    type Value = NoRepeats;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<NoRepeats, A::Error> {
        let mut seen = std::collections::BTreeSet::new();
        let mut found = None;
        while let Some(key) = map.next_key::<String>()? {
            let inner: NoRepeats = map.next_value()?;
            if found.is_some() {
                continue;
            }
            if !seen.insert(key.clone()) {
                found = Some(key);
            } else if let Some(at) = inner.0 {
                let sep = if at.starts_with('[') { "" } else { "." };
                found = Some(std::format!("{key}{sep}{at}"));
            }
        }
        Ok(NoRepeats(found))
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<NoRepeats, A::Error> {
        let mut found = None;
        let mut i = 0usize;
        while let Some(inner) = seq.next_element::<NoRepeats>()? {
            if found.is_none()
                && let Some(at) = inner.0
            {
                let sep = if at.starts_with('[') { "" } else { "." };
                found = Some(std::format!("[{i}]{sep}{at}"));
            }
            i += 1;
        }
        Ok(NoRepeats(found))
    }

    fn visit_bool<E>(self, _: bool) -> Result<NoRepeats, E> {
        Ok(NoRepeats(None))
    }
    fn visit_i64<E>(self, _: i64) -> Result<NoRepeats, E> {
        Ok(NoRepeats(None))
    }
    fn visit_u64<E>(self, _: u64) -> Result<NoRepeats, E> {
        Ok(NoRepeats(None))
    }
    fn visit_f64<E>(self, _: f64) -> Result<NoRepeats, E> {
        Ok(NoRepeats(None))
    }
    fn visit_str<E>(self, _: &str) -> Result<NoRepeats, E> {
        Ok(NoRepeats(None))
    }
    fn visit_unit<E>(self) -> Result<NoRepeats, E> {
        Ok(NoRepeats(None))
    }
}
