//! The graph file: a [`Graph`] in JSON, as an agent writes, reads back and edits it.
//!
//! The file is one object. `format` is `"mag-graph"` and `version` is `1`; then the graph's
//! fields: `job`, `inputs`, `parts`, `chains` and `edges`. Enumerations are written in
//! snake case (`"replicable"`, `"optimiser_step"`, `"bf16"`), and a pin as `{"member": name}` or
//! `{"backend": "rocm"}`. A job states no objective: its need and its priority say what it
//! wants.
//!
//! Reading checks `format` and `version` first, so that a file of a newer version is refused as
//! that rather than for fields this reader does not know. It refuses unknown fields, so that a
//! misspelt field is an error and not a silent default, and then runs [`Graph::check`]. Fields
//! with a default may be left out: `inputs`, `chains` and `edges` (empty), a job's `need`
//! (none), a part's `mark` (atomic), `weights` (none) and `pin` (none), an edge's `to` (a result
//! of the job) and `release` (false). An input's `local_only` has no default. An option is
//! present or left out; `null` is refused.
//!
//! Writing gives `format` and `version` first and the fields in a fixed order, every field but
//! an absent option, two-space indented with a final newline, so that reading what was written
//! and writing it again gives the same text.

use alloc::string::{String, ToString};
use core::fmt;

use serde_json::Value;

use crate::{Graph, Refusal};

/// The graph file's `format`.
pub const FORMAT: &str = "mag-graph";

/// The graph file's `version` this crate reads and writes.
pub const VERSION: u64 = 1;

/// Why a graph file was not read.
#[derive(Debug)]
pub enum ReadError {
    /// Not JSON, or not a graph: a field missing, unknown or of the wrong type.
    Json(serde_json::Error),
    /// `format` is not `"mag-graph"`, or the file is not an object. The JSON found, if any.
    Format { found: Option<String> },
    /// `version` is not 1. The JSON found, if any.
    Version { found: Option<String> },
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

#[derive(serde::Serialize)]
struct File<'a> {
    format: &'static str,
    version: u64,
    #[serde(flatten)]
    graph: &'a Graph,
}

impl Graph {
    /// Reads a graph file and checks the graph.
    pub fn from_json(text: &str) -> Result<Graph, ReadError> {
        let mut value: Value = serde_json::from_str(text).map_err(ReadError::Json)?;
        let Value::Object(fields) = &mut value else {
            return Err(ReadError::Format { found: None });
        };
        match fields.remove("format") {
            Some(Value::String(f)) if f == FORMAT => {}
            other => {
                return Err(ReadError::Format {
                    found: other.map(|v| v.to_string()),
                });
            }
        }
        match fields.remove("version") {
            Some(Value::Number(v)) if v.as_u64() == Some(VERSION) => {}
            other => {
                return Err(ReadError::Version {
                    found: other.map(|v| v.to_string()),
                });
            }
        }
        let graph: Graph = serde_json::from_value(value).map_err(ReadError::Json)?;
        graph.check().map_err(ReadError::Refused)?;
        Ok(graph)
    }

    /// Writes the graph as a graph file. The graph is written as it is, checked or not.
    pub fn to_json(&self) -> String {
        let file = File {
            format: FORMAT,
            version: VERSION,
            graph: self,
        };
        // A graph holds only strings, integers, floats and booleans. A float that is not finite
        // (a need `check` refuses) is written as null, which reading refuses.
        let mut text =
            serde_json::to_string_pretty(&file).expect("a graph always serialises to JSON");
        text.push('\n');
        text
    }
}
