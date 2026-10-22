//! Decode a training set once, into the arrays [`crate::data`] reads.
//!
//! Decoding full-size JPEGs inside the training loop would pay for the same work every
//! epoch. Each frame is decoded and resized whole to the model's input size — squashed,
//! not cropped — through `dash-image`, the preprocessing inference uses, so a model never
//! sees a different transform in training than in deployment.
//!
//! The training set is JSON lines, one frame a row:
//! `{"image": "<path relative to root>", "split": "train", "labels": {"<head>": "<class>" | null}}`.
//! A null label masks the head on that frame. Other keys are carried through.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use dash_core::ModelSpec;
use dash_image::{decode_jpeg, resize_bilinear};
use serde_json::{Value, json};

use crate::data::npy_header_bytes;
use crate::loss::IGNORE;

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(windows)]
fn write_at(file: &File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_write(buf, offset)?;
        buf = &buf[n..];
        offset += n as u64;
    }
    Ok(())
}

/// One row's targets in output order, or why the row does not fit the model.
pub fn targets(spec: &ModelSpec, labels: &Value) -> Result<Vec<i64>, String> {
    spec.outputs
        .iter()
        .map(|o| match labels.get(&o.name) {
            None => Err(format!("no label for head {:?}", o.name)),
            Some(Value::Null) => Ok(IGNORE),
            Some(Value::String(c)) => o
                .classes
                .iter()
                .position(|x| x == c)
                .map(|i| i as i64)
                .ok_or(format!("head {:?} has no class {c:?}", o.name)),
            Some(other) => Err(format!(
                "label of {:?} is {other}, not a class name",
                o.name
            )),
        })
        .collect()
}

/// Write `<out>/<split>.npy` and `<out>/<split>.jsonl` for every split in `trainset`, in
/// the order the splits first appear; `limit` keeps the first frames of each. Returns each
/// split's name and frame count.
pub fn build(
    trainset: &Path,
    root: &Path,
    spec: &ModelSpec,
    out: &Path,
    size: (usize, usize),
    limit: Option<usize>,
    threads: usize,
) -> Result<Vec<(String, usize)>, String> {
    let (height, width) = size;
    let reader =
        BufReader::new(File::open(trainset).map_err(|e| format!("{}: {e}", trainset.display()))?);
    // Each split's rows in file order, with their targets.
    type Rows = Vec<(Value, Vec<i64>)>;
    let mut splits: Vec<(String, Rows)> = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let at = |e: String| format!("{} line {}: {e}", trainset.display(), i + 1);
        let row: Value = serde_json::from_str(&line).map_err(|e| at(e.to_string()))?;
        let split = row["split"]
            .as_str()
            .ok_or_else(|| at("no split".into()))?
            .to_string();
        row["image"].as_str().ok_or_else(|| at("no image".into()))?;
        let t = targets(spec, &row["labels"]).map_err(at)?;
        match splits.iter_mut().find(|(s, _)| *s == split) {
            Some((_, rows)) => rows.push((row, t)),
            None => splits.push((split, vec![(row, t)])),
        }
    }
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut written = Vec::new();
    for (split, mut rows) in splits {
        if let Some(n) = limit {
            rows.truncate(n);
        }
        let npy = out.join(format!("{split}.npy"));
        let header = npy_header_bytes([rows.len(), height, width, 3]);
        let frame = height * width * 3;
        {
            let mut f = File::create(&npy).map_err(|e| format!("{}: {e}", npy.display()))?;
            f.write_all(&header).map_err(|e| e.to_string())?;
            f.set_len((header.len() + rows.len() * frame) as u64)
                .map_err(|e| e.to_string())?;
        }
        let file = OpenOptions::new()
            .write(true)
            .open(&npy)
            .map_err(|e| e.to_string())?;
        let next = AtomicUsize::new(0);
        let failures: Vec<String> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..threads.max(1))
                .map(|_| {
                    s.spawn(|| -> Result<(), String> {
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some((row, _)) = rows.get(i) else {
                                return Ok(());
                            };
                            let path = root.join(row["image"].as_str().unwrap());
                            let bytes =
                                fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                            let img = decode_jpeg(&bytes)
                                .map_err(|e| format!("{}: {e}", path.display()))?;
                            let img = resize_bilinear(&img, width, height);
                            write_at(&file, &img.data, (header.len() + i * frame) as u64)
                                .map_err(|e| e.to_string())?;
                        }
                    })
                })
                .collect();
            workers
                .into_iter()
                .filter_map(|w| w.join().unwrap().err())
                .collect()
        });
        if let Some(e) = failures.into_iter().next() {
            return Err(e);
        }
        let jsonl = out.join(format!("{split}.jsonl"));
        let mut f = io::BufWriter::new(File::create(&jsonl).map_err(|e| e.to_string())?);
        for (row, t) in &rows {
            let mut row = row.clone();
            row["targets"] = json!(t);
            writeln!(f, "{row}").map_err(|e| e.to_string())?;
        }
        f.flush().map_err(|e| e.to_string())?;
        println!("{split}: {} frames -> {}", rows.len(), npy.display());
        written.push((split, rows.len()));
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dash_core::OutputSpec;

    #[test]
    fn labels_become_targets_in_output_order() {
        let spec = ModelSpec {
            arch: String::new(),
            outputs: vec![
                OutputSpec {
                    name: "a".into(),
                    classes: vec!["x".into(), "y".into()],
                },
                OutputSpec {
                    name: "b".into(),
                    classes: vec!["p".into(), "q".into(), "r".into()],
                },
            ],
            undefined_when: vec![],
        };
        assert_eq!(
            targets(&spec, &json!({"b": "r", "a": null})).unwrap(),
            [IGNORE, 2]
        );
        assert!(
            targets(&spec, &json!({"a": "z", "b": "p"}))
                .unwrap_err()
                .contains("\"z\"")
        );
        assert!(
            targets(&spec, &json!({"a": "x"}))
                .unwrap_err()
                .contains("\"b\"")
        );
    }
}
