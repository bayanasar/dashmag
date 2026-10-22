//! The decoded frames a run trains on, and the augmentation applied to them.
//!
//! A split is two files in a cache directory, as [`crate::cache`] writes them:
//! `<split>.npy`, an `N x H x W x 3` array of `uint8` RGB (NumPy's format, so either side
//! can read the other's), and `<split>.jsonl`, one row per frame in the same order with
//! `targets`: one class index per head, [`IGNORE`](crate::loss::IGNORE) where the frame has
//! no label for it. Frames are read from the file as they are needed, so several runs on
//! one machine share the page cache rather than each holding a copy.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

use dash_image::{Rgb8, resize_bilinear, to_nchw};

use crate::rng::Rng;

/// Where an `N x H x W x 3` `uint8` array starts in a `.npy` file, and its shape.
pub fn npy_header(file: &mut File) -> Result<(u64, [usize; 4]), String> {
    let mut head = [0u8; 10];
    file.read_exact(&mut head).map_err(|e| e.to_string())?;
    if &head[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let (len, start) = match head[6] {
        1 => (u16::from_le_bytes([head[8], head[9]]) as usize, 10),
        2 | 3 => {
            let mut more = [0u8; 2];
            file.read_exact(&mut more).map_err(|e| e.to_string())?;
            (
                u32::from_le_bytes([head[8], head[9], more[0], more[1]]) as usize,
                12,
            )
        }
        v => return Err(format!(".npy version {v} is not read here")),
    };
    let mut dict = vec![0u8; len];
    file.read_exact(&mut dict).map_err(|e| e.to_string())?;
    let dict = String::from_utf8_lossy(&dict);
    if !dict.contains("'descr': '|u1'") || !dict.contains("'fortran_order': False") {
        return Err(format!("not a C-ordered uint8 array: {dict}"));
    }
    let shape = dict
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or(format!("no shape in {dict}"))?;
    let dims: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|_| format!("shape {shape:?}")))
        .collect::<Result<_, _>>()?;
    match dims[..] {
        [n, h, w, 3] => Ok(((start + len) as u64, [n, h, w, 3])),
        _ => Err(format!("expected N x H x W x 3, got {dims:?}")),
    }
}

/// The header NumPy writes for an `N x H x W x 3` `uint8` array, padded so the data starts
/// on a 64-byte boundary.
pub fn npy_header_bytes(shape: [usize; 4]) -> Vec<u8> {
    let [n, h, w, c] = shape;
    let dict =
        format!("{{'descr': '|u1', 'fortran_order': False, 'shape': ({n}, {h}, {w}, {c}), }}");
    // magic and version (8), length (2), the dict, spaces, a newline
    let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&((dict.len() + pad + 1) as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat_n(b' ', pad));
    out.push(b'\n');
    out
}

#[cfg(unix)]
pub(crate) fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
pub(crate) fn read_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset)? {
            0 => return Err(io::ErrorKind::UnexpectedEof.into()),
            n => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
        }
    }
    Ok(())
}

/// One split of a cache.
pub struct Split {
    file: File,
    offset: u64,
    pub len: usize,
    pub height: usize,
    pub width: usize,
    /// `targets[i][h]`: frame `i`'s class index on head `h`.
    pub targets: Vec<Vec<i64>>,
}

impl Split {
    pub fn open(cache: &Path, split: &str, heads: usize) -> Result<Self, String> {
        let npy = cache.join(format!("{split}.npy"));
        let mut file = File::open(&npy).map_err(|e| format!("{}: {e}", npy.display()))?;
        let (offset, [len, height, width, _]) =
            npy_header(&mut file).map_err(|e| format!("{}: {e}", npy.display()))?;
        let rows = cache.join(format!("{split}.jsonl"));
        let reader =
            BufReader::new(File::open(&rows).map_err(|e| format!("{}: {e}", rows.display()))?);
        let mut targets = Vec::with_capacity(len);
        for (i, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| e.to_string())?;
            if line.trim().is_empty() {
                continue;
            }
            let row: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| format!("{} line {}: {e}", rows.display(), i + 1))?;
            let t: Vec<i64> = row["targets"]
                .as_array()
                .ok_or(format!("{} line {}: no targets", rows.display(), i + 1))?
                .iter()
                .map(|v| v.as_i64().ok_or("a target is not an integer"))
                .collect::<Result<_, _>>()?;
            if t.len() != heads {
                return Err(format!(
                    "{} line {}: {} targets for {heads} heads",
                    rows.display(),
                    i + 1,
                    t.len()
                ));
            }
            targets.push(t);
        }
        if targets.len() != len {
            return Err(format!(
                "{} has {len} frames, {} has {} rows",
                npy.display(),
                rows.display(),
                targets.len()
            ));
        }
        Ok(Split {
            file,
            offset,
            len,
            height,
            width,
            targets,
        })
    }

    pub fn image(&self, i: usize) -> Result<Rgb8, String> {
        let size = self.height * self.width * 3;
        let mut data = vec![0u8; size];
        read_at(&self.file, &mut data, self.offset + (i * size) as u64)
            .map_err(|e| e.to_string())?;
        Ok(Rgb8 {
            width: self.width,
            height: self.height,
            data,
        })
    }

    /// Head `h`'s targets over the whole split.
    pub fn column(&self, h: usize) -> Vec<i64> {
        self.targets.iter().map(|t| t[h]).collect()
    }
}

/// torchvision's `RandomResizedCrop(scale, ratio)` and `RandomHorizontalFlip(p)`.
#[derive(Clone, Copy, Debug)]
pub struct Augment {
    pub scale: (f64, f64),
    pub ratio: (f64, f64),
    pub flip: f64,
}

impl Default for Augment {
    /// The recipe's: a crop of 80-100% of the area at an aspect of 0.9-1.1, and a flip half
    /// the time. Geometric only — brightness and colour carry labels.
    fn default() -> Self {
        Augment {
            scale: (0.8, 1.0),
            ratio: (0.9, 1.1),
            flip: 0.5,
        }
    }
}

/// `RandomResizedCrop.get_params`: (top, left, height, width) of the crop. Ten tries at a
/// random area and aspect; failing those, the centre crop at the nearest allowed aspect.
pub fn crop_box(
    height: usize,
    width: usize,
    scale: (f64, f64),
    ratio: (f64, f64),
    rng: &mut Rng,
) -> (usize, usize, usize, usize) {
    let area = (height * width) as f64;
    let log_ratio = (ratio.0.ln(), ratio.1.ln());
    for _ in 0..10 {
        let target = area * rng.range(scale.0, scale.1);
        let aspect = rng.range(log_ratio.0, log_ratio.1).exp();
        // Python's round(): halves to even.
        let w = (target * aspect).sqrt().round_ties_even() as usize;
        let h = (target / aspect).sqrt().round_ties_even() as usize;
        if 0 < w && w <= width && 0 < h && h <= height {
            let top = rng.below(height - h + 1);
            let left = rng.below(width - w + 1);
            return (top, left, h, w);
        }
    }
    let in_ratio = width as f64 / height as f64;
    let (h, w) = if in_ratio < ratio.0 {
        ((width as f64 / ratio.0).round_ties_even() as usize, width)
    } else if in_ratio > ratio.1 {
        (height, (height as f64 * ratio.1).round_ties_even() as usize)
    } else {
        (height, width)
    };
    ((height - h) / 2, (width - w) / 2, h, w)
}

fn crop(img: &Rgb8, top: usize, left: usize, h: usize, w: usize) -> Rgb8 {
    let mut data = Vec::with_capacity(h * w * 3);
    for y in top..top + h {
        let row = (y * img.width + left) * 3;
        data.extend_from_slice(&img.data[row..row + w * 3]);
    }
    Rgb8 {
        width: w,
        height: h,
        data,
    }
}

fn flip(img: &mut Rgb8) {
    let w = img.width;
    for row in img.data.chunks_exact_mut(w * 3) {
        for x in 0..w / 2 {
            for c in 0..3 {
                row.swap(x * 3 + c, (w - 1 - x) * 3 + c);
            }
        }
    }
}

/// A random crop, resized back to the frame's size, and maybe flipped.
pub fn augment(img: &Rgb8, aug: &Augment, rng: &mut Rng) -> Rgb8 {
    let (top, left, h, w) = crop_box(img.height, img.width, aug.scale, aug.ratio, rng);
    let mut out = resize_bilinear(&crop(img, top, left, h, w), img.width, img.height);
    if rng.uniform() < aug.flip {
        flip(&mut out);
    }
    out
}

/// A batch: the network's input, `N x 3 x H x W` normalised, and each head's targets.
pub struct Batch {
    pub input: Vec<f32>,
    pub shape: [usize; 4],
    /// `targets[h][i]`: head `h`'s target for the batch's frame `i`.
    pub targets: Vec<Vec<i64>>,
}

/// Frames `indices` of `split`, augmented when `aug` is given, as one batch.
pub fn batch(
    split: &Split,
    indices: &[usize],
    aug: Option<(&Augment, &mut Rng)>,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Batch, String> {
    let heads = split.targets.first().map_or(0, Vec::len);
    let mut input = Vec::with_capacity(indices.len() * 3 * split.height * split.width);
    let mut targets = vec![Vec::with_capacity(indices.len()); heads];
    let mut aug = aug;
    for &i in indices {
        let img = split.image(i)?;
        let img = match aug.as_mut() {
            Some((a, rng)) => augment(&img, a, rng),
            None => img,
        };
        input.extend(to_nchw(&img, mean, std));
        for (h, col) in targets.iter_mut().enumerate() {
            col.push(split.targets[i][h]);
        }
    }
    Ok(Batch {
        input,
        shape: [indices.len(), 3, split.height, split.width],
        targets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_round_trips_and_is_aligned() {
        let bytes = npy_header_bytes([16105, 224, 224, 3]);
        assert_eq!(bytes.len() % 64, 0);
        let path = std::env::temp_dir().join(format!("dash-train-npy-{}", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let got = npy_header(&mut File::open(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(got, (bytes.len() as u64, [16105, 224, 224, 3]));
    }

    #[test]
    fn crops_stay_inside_and_within_the_ranges() {
        let mut rng = Rng::new(3);
        for _ in 0..2000 {
            let (top, left, h, w) = crop_box(224, 224, (0.8, 1.0), (0.9, 1.1), &mut rng);
            assert!(top + h <= 224 && left + w <= 224);
            let area = (h * w) as f64 / (224.0 * 224.0);
            assert!((0.78..=1.0).contains(&area), "area {area}");
        }
    }

    #[test]
    fn a_flip_mirrors_each_row() {
        let mut img = Rgb8 {
            width: 3,
            height: 1,
            data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        };
        flip(&mut img);
        assert_eq!(img.data, [7, 8, 9, 4, 5, 6, 1, 2, 3]);
    }
}
