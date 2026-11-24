//! The ledger's lines: a [`Ledger`] in JSON Lines, as the host of a pool appends it and reads it
//! back.
//!
//! The first line is the header: `format` `"mag-ledger"`, `version` 1, and the members and offers
//! the pool began with. Each line after it is one [`Entry`], written when the pool takes its
//! change. A line is one JSON object and ends with its newline; JSON escapes a newline inside a
//! string, so none can break a line. The words are written as the graph file's are.
//!
//! This makes lines and reads them. Opening, appending and syncing a file, or keeping the lines
//! wherever the host keeps its state, is the host's. A change is acknowledged only once its line
//! is written, newline and all, so a last line without its newline was cut short and never
//! acknowledged: the reader drops it and says so. A line cut short anywhere else is refused, and
//! so is what the graph file's reader refuses: another format, a newer version, an unknown field,
//! a key repeated in one object, a `null`.
//!
//! Every `f64` is written in the shortest form that reads back to the same bits, and is read back
//! to them, so that a replayed record can be compared with the one logged.
//!
//! [`ledger_schema`] is the JSON Schema of one line, header or entry, generated from the types,
//! kept in the repository at `schemas/mag-ledger.schema.json`, and checked by a test.

use core::fmt;
use std::string::{String, ToString};
use std::vec::Vec;

use serde_json::Value;

use crate::json::{NoRepeats, strip_null};
use crate::{Entry, LEDGER_FORMAT, LEDGER_VERSION, Ledger, Member, Offer};

/// Why a ledger's lines were not read. Lines are counted from 1, the header's.
#[derive(Debug)]
pub enum LedgerError {
    /// The first line's `format` is not `"mag-ledger"`, it is not an object, or there is no first
    /// line. The JSON found, if any.
    Format { found: Option<String> },
    /// The header's `version` is not 1. The JSON found, if any.
    Version { found: Option<String> },
    /// A line is not JSON, or not a header or an entry: a field missing, unknown, `null` or of
    /// the wrong type.
    Json {
        line: usize,
        error: serde_json::Error,
    },
    /// A key appears twice in one object of a line. Where, as a path such as
    /// `record.candidates[0].score`.
    RepeatedKey { line: usize, at: String },
    /// A line other than the last was cut short: its JSON ends early.
    Torn { line: usize },
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LedgerError::Format { found: Some(found) } => {
                write!(f, "the format is {found}, not \"{LEDGER_FORMAT}\"")
            }
            LedgerError::Format { found: None } => {
                write!(f, "not a ledger: no \"format\": \"{LEDGER_FORMAT}\"")
            }
            LedgerError::Version { found: Some(found) } => write!(
                f,
                "the version is {found}; this reader reads version {LEDGER_VERSION}"
            ),
            LedgerError::Version { found: None } => write!(f, "the ledger has no version"),
            LedgerError::Json { line, error } => write!(f, "line {line}: {error}"),
            LedgerError::RepeatedKey { line, at } => write!(
                f,
                "line {line}: `{at}` is given twice; a key may appear once in an object"
            ),
            LedgerError::Torn { line } => write!(
                f,
                "line {line} was cut short, and is not the last: the ledger is damaged"
            ),
        }
    }
}

impl core::error::Error for LedgerError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            LedgerError::Json { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// A ledger read from its lines.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadLedger {
    pub ledger: Ledger,
    /// The number of a last line that was cut short and dropped, if there was one.
    pub dropped: Option<usize>,
}

#[derive(serde::Serialize)]
struct HeaderOut<'a> {
    format: &'static str,
    version: u64,
    members: &'a [Member],
    offers: &'a [Offer],
}

/// The header's fields after `format` and `version`.
#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct Header {
    /// The members the pool began with, as they were given to it.
    members: Vec<Member>,
    /// The offers the pool began with, in the order of the devices' ids.
    offers: Vec<Offer>,
}

/// One line of a ledger, for its schema.
#[derive(schemars::JsonSchema)]
#[serde(untagged)]
#[allow(dead_code, clippy::large_enum_variant)]
enum Line {
    Header(Header),
    Entry(Entry),
}

impl Ledger {
    /// The first line: the format and version, and the members and offers the pool began with,
    /// with its newline.
    pub fn header_line(&self) -> String {
        let header = HeaderOut {
            format: LEDGER_FORMAT,
            version: LEDGER_VERSION,
            members: &self.members,
            offers: &self.offers,
        };
        let mut text = serde_json::to_string(&header).expect("a header always serialises");
        text.push('\n');
        text
    }

    /// One entry as a line, with its newline.
    pub fn line(entry: &Entry) -> String {
        let mut text = serde_json::to_string(entry).expect("an entry always serialises");
        text.push('\n');
        text
    }

    /// The whole ledger as lines: the header, then each entry.
    pub fn to_lines(&self) -> String {
        let mut text = self.header_line();
        for e in &self.entries {
            text.push_str(&Ledger::line(e));
        }
        text
    }

    /// Reads a ledger from its lines, dropping a last line cut short.
    pub fn from_lines(text: &str) -> Result<ReadLedger, LedgerError> {
        let mut lines: Vec<&str> = text.split('\n').collect();
        // What follows the last newline: nothing, when the last line is whole.
        let tail = lines.pop().unwrap_or_default();
        let dropped = (!tail.is_empty()).then_some(lines.len() + 1);
        let Some((head, entries)) = lines.split_first() else {
            return Err(match dropped {
                Some(line) => LedgerError::Torn { line },
                None => LedgerError::Format { found: None },
            });
        };
        let (members, offers) = header(head)?;
        let entries = entries
            .iter()
            .enumerate()
            .map(|(i, l)| entry(l, i + 2))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ReadLedger {
            ledger: Ledger {
                members,
                offers,
                entries,
            },
            dropped,
        })
    }
}

/// A line's JSON; a line that ends early was cut short.
fn parse(text: &str, line: usize) -> Result<Value, LedgerError> {
    serde_json::from_str(text).map_err(|error| {
        if error.is_eof() {
            LedgerError::Torn { line }
        } else {
            LedgerError::Json { line, error }
        }
    })
}

fn no_repeats(text: &str, line: usize) -> Result<(), LedgerError> {
    let repeats: NoRepeats =
        serde_json::from_str(text).map_err(|error| LedgerError::Json { line, error })?;
    match repeats.0 {
        Some(at) => Err(LedgerError::RepeatedKey { line, at }),
        None => Ok(()),
    }
}

/// The header's members and offers, its format and version checked first.
fn header(text: &str) -> Result<(Vec<Member>, Vec<Offer>), LedgerError> {
    let mut value = parse(text, 1)?;
    let Value::Object(fields) = &mut value else {
        return Err(LedgerError::Format { found: None });
    };
    match fields.remove("format") {
        Some(Value::String(f)) if f == LEDGER_FORMAT => {}
        other => {
            return Err(LedgerError::Format {
                found: other.map(|v| v.to_string()),
            });
        }
    }
    match fields.remove("version") {
        Some(Value::Number(v)) if v.as_u64() == Some(LEDGER_VERSION) => {}
        other => {
            return Err(LedgerError::Version {
                found: other.map(|v| v.to_string()),
            });
        }
    }
    no_repeats(text, 1)?;
    let h: Header =
        serde_json::from_value(value).map_err(|error| LedgerError::Json { line: 1, error })?;
    Ok((h.members, h.offers))
}

fn entry(text: &str, line: usize) -> Result<Entry, LedgerError> {
    let value = parse(text, line)?;
    no_repeats(text, line)?;
    serde_json::from_value(value).map_err(|error| LedgerError::Json { line, error })
}

/// The JSON Schema (draft 2020-12) of one line of a ledger, header or entry, as text with a
/// final newline: generated from the types a line is read into, with the header's `format` and
/// `version` as constants, unknown fields refused, and no `null` anywhere.
pub fn ledger_schema() -> String {
    let generator = schemars::generate::SchemaSettings::draft2020_12().into_generator();
    let mut schema = generator.into_root_schema_for::<Line>().to_value();
    strip_null(&mut schema);
    let root = schema
        .as_object_mut()
        .expect("a schema for an enum is an object");
    root.insert(
        "title".into(),
        format!("{LEDGER_FORMAT}, version {LEDGER_VERSION}: one line").into(),
    );
    let header = root
        .get_mut("$defs")
        .and_then(|d| d.get_mut("Header"))
        .and_then(Value::as_object_mut)
        .expect("the header has a definition");
    let properties = header
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .expect("a struct's schema has properties");
    properties.insert(
        "format".into(),
        serde_json::json!({ "const": LEDGER_FORMAT }),
    );
    properties.insert(
        "version".into(),
        serde_json::json!({ "const": LEDGER_VERSION }),
    );
    let required = header
        .get_mut("required")
        .and_then(Value::as_array_mut)
        .expect("a header has required fields");
    required.insert(0, "version".into());
    required.insert(0, "format".into());
    let mut text = serde_json::to_string_pretty(&schema).expect("a schema serialises");
    text.push('\n');
    text
}
