//! Files of version 1, read to the same meaning in version 2's words.
//!
//! Version 1 marked data and members with a bit each; version 2 states levels of trust. A
//! version 1 file's JSON is rewritten into version 2's words, field by field, and then read as
//! version 2, with every check version 2 makes:
//! - an input's and a blob's `local_only`: `true` is `keep` `owner`, `false` is `outside`;
//! - an edge's `release`: `true` is `release` `outside`, `false` or left out is no release;
//! - a member's `remote`: `true` is `trust` `outside`, `false` is `owner`;
//! - in a ledger, a submission carries the empty stamp, which bounds nothing; a guard's removal
//!   is of data kept at `owner` from a member trusted at `outside`, the only guard version 1
//!   had; and a record's `marked_remote` is its `lowered`.
//!
//! A version 2 word in a version 1 file is refused, as version 1 refused an unknown field.

use std::format;
use std::string::String;

use serde_json::{Map, Value};

type Error = serde_json::Error;

fn error(message: String) -> Error {
    <Error as serde::de::Error>::custom(message)
}

fn level(on: bool, if_on: &'static str, if_off: &'static str) -> Option<&'static str> {
    Some(if on { if_on } else { if_off })
}

/// Replaces a field of version 1's, a bool, by version 2's, a level (none: the field left out).
fn rewrite(
    object: &mut Map<String, Value>,
    old: &str,
    new: &str,
    required: bool,
    to: impl Fn(bool) -> Option<&'static str>,
) -> Result<(), Error> {
    let found = object.remove(old);
    if object.contains_key(new) {
        return Err(error(format!(
            "unknown field `{new}` in a file of version 1"
        )));
    }
    match found {
        Some(Value::Bool(on)) => {
            if let Some(level) = to(on) {
                object.insert(new.into(), level.into());
            }
            Ok(())
        }
        Some(other) => Err(error(format!(
            "invalid type: {other}, expected a boolean for `{old}`"
        ))),
        None if required => Err(error(format!("missing field `{old}`"))),
        None => Ok(()),
    }
}

/// Refuses fields that only version 2 has.
fn refuse(object: &Map<String, Value>, fields: &[&str]) -> Result<(), Error> {
    match fields.iter().find(|f| object.contains_key(**f)) {
        Some(f) => Err(error(format!("unknown field `{f}` in a file of version 1"))),
        None => Ok(()),
    }
}

/// The objects of an array field, if it is one: what is not is left for reading to refuse.
fn each<'a>(v: &'a mut Value, field: &str) -> impl Iterator<Item = &'a mut Map<String, Value>> {
    v.get_mut(field)
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object_mut)
}

/// A graph's fields, as the graph file and a ledger's submission hold them.
pub(crate) fn graph(v: &mut Value) -> Result<(), Error> {
    let keep = |on| level(on, "owner", "outside");
    for input in each(v, "inputs") {
        rewrite(input, "local_only", "keep", true, keep)?;
    }
    let parts = v.get_mut("parts").and_then(Value::as_array_mut);
    for part in parts.into_iter().flatten() {
        for blob in each(part, "weights") {
            rewrite(blob, "local_only", "keep", true, keep)?;
        }
    }
    for edge in each(v, "edges") {
        rewrite(edge, "release", "release", false, |on| {
            on.then_some("outside")
        })?;
    }
    Ok(())
}

fn member(m: &mut Map<String, Value>) -> Result<(), Error> {
    rewrite(m, "remote", "trust", true, |on| {
        level(on, "outside", "owner")
    })
}

/// A pool file's fields.
pub(crate) fn description(v: &mut Value) -> Result<(), Error> {
    each(v, "members").try_for_each(member)
}

/// A ledger's header's fields, after its format and version.
pub(crate) fn header(v: &mut Value) -> Result<(), Error> {
    if let Some(header) = v.as_object() {
        refuse(header, &["written_by", "follows"])?;
    }
    each(v, "members").try_for_each(member)
}

/// A ledger's entry.
pub(crate) fn entry(v: &mut Value) -> Result<(), Error> {
    if let Some(change) = v.get_mut("change").and_then(Value::as_object_mut) {
        if let Some(submit) = change.get_mut("submit").and_then(Value::as_object_mut) {
            refuse(submit, &["stamp"])?;
            if let Some(g) = submit.get_mut("graph") {
                graph(g)?;
            }
            submit.insert("stamp".into(), Value::Object(Map::new()));
        }
        let joined = change
            .get_mut("join")
            .and_then(|j| j.get_mut("member"))
            .and_then(Value::as_object_mut);
        if let Some(m) = joined {
            member(m)?;
        }
    }
    if let Some(record) = v.get_mut("record").and_then(Value::as_object_mut) {
        let found = record.remove("marked_remote");
        refuse(record, &["lowered"])?;
        match found {
            Some(names) => record.insert("lowered".into(), names),
            None => return Err(error("missing field `marked_remote`".into())),
        };
        let candidates = record.get_mut("candidates").and_then(Value::as_array_mut);
        for c in candidates.into_iter().flatten() {
            if let Some(r) = c.get_mut("removed") {
                removal(r)?;
            }
            let left_out = c.get_mut("left_out").and_then(Value::as_array_mut);
            for pair in left_out.into_iter().flatten() {
                if let Some(r) = pair.get_mut(1) {
                    removal(r)?;
                }
            }
        }
    }
    Ok(())
}

fn removal(v: &mut Value) -> Result<(), Error> {
    let Some(r) = v.as_object_mut() else {
        return Ok(());
    };
    refuse(r, &["stamp"])?;
    if let Some(guard) = r.get_mut("guard").and_then(Value::as_object_mut) {
        refuse(guard, &["keep", "trust"])?;
        guard.insert("keep".into(), "owner".into());
        guard.insert("trust".into(), "outside".into());
    }
    Ok(())
}
