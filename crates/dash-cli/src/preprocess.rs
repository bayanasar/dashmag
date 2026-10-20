//! From an image file to a model input: decode, resize, normalise, NCHW.
//!
//! A model sees whatever its training preprocessing produced, so the transform here is
//! not a convenience — it is half of the model's contract. It reproduces Pillow's
//! `Image.resize(size, Image.BILINEAR)` exactly: the same coefficients, the same 22-bit
//! fixed point, horizontal pass first, the intermediate rounded to 8 bits. Given the
//! same RGB pixels, the output bytes are Pillow's bytes.
//!
//! Decoding is the one step that is not exact. JPEG decoders are allowed to differ by
//! a level or two per pixel, and Pillow uses libjpeg-turbo while this uses zune-jpeg
//! (pure Rust, so a static cross-build is unaffected). zune-jpeg is built without its
//! SIMD paths, so the host and every board decode through the same scalar code and
//! hand the model identical pixels.

use std::io::Cursor;

use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// Pillow's `PRECISION_BITS` for 8-bit images.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// ImageNet's per-channel mean and standard deviation, RGB order.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// An interleaved RGB image, 3 bytes per pixel, row-major.
pub struct Rgb8 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

pub fn decode_jpeg(bytes: &[u8]) -> Result<Rgb8, String> {
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(Cursor::new(bytes), options);
    let data = decoder
        .decode()
        .map_err(|e| format!("jpeg decode: {e:?}"))?;
    let info = decoder.info().ok_or("jpeg decode: no image info")?;
    let (width, height) = (info.width as usize, info.height as usize);
    if data.len() != width * height * 3 {
        return Err(format!(
            "jpeg decode: {} bytes for {width}x{height} RGB",
            data.len()
        ));
    }
    Ok(Rgb8 {
        width,
        height,
        data,
    })
}

/// One output position's window into the input: where it starts, how many taps,
/// and the fixed-point weight of each.
struct Coeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    weights: Vec<i32>,
}

fn bilinear(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

/// Pillow's `precompute_coeffs` followed by `normalize_coeffs_8bpc`, for the whole
/// input range. The arithmetic order is Pillow's, because the rounding is.
fn coeffs(in_size: usize, out_size: usize) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filterscale; // the bilinear filter's support is 1.0
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = vec![0i32; out_size * ksize];
    let mut taps = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C's (int) cast truncates toward zero, as `as` does.
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64);
        let n = (xmax - xmin) as usize;
        let mut ww = 0.0;
        for (x, tap) in taps.iter_mut().enumerate().take(n) {
            *tap = bilinear(((x as i64 + xmin) as f64 - center + 0.5) * ss);
            ww += *tap;
        }
        for x in 0..n {
            let w = if ww != 0.0 { taps[x] / ww } else { taps[x] };
            let fixed = w * f64::from(1u32 << PRECISION_BITS);
            weights[xx * ksize + x] = if w < 0.0 {
                (-0.5 + fixed) as i32
            } else {
                (0.5 + fixed) as i32
            };
        }
        bounds.push((xmin as usize, n));
    }
    Coeffs {
        ksize,
        bounds,
        weights,
    }
}

fn clip8(v: i32) -> u8 {
    if v >= 1 << PRECISION_BITS << 8 {
        255
    } else if v <= 0 {
        0
    } else {
        (v >> PRECISION_BITS) as u8
    }
}

/// `Image.resize((width, height), Image.BILINEAR)` on an RGB image, byte for byte.
pub fn resize_bilinear(src: &Rgb8, width: usize, height: usize) -> Rgb8 {
    let round = 1i32 << (PRECISION_BITS - 1);

    // Horizontal pass, skipped (as Pillow skips it) when the width does not change.
    let horiz = if width != src.width {
        let c = coeffs(src.width, width);
        let mut out = vec![0u8; width * src.height * 3];
        for y in 0..src.height {
            let row = &src.data[y * src.width * 3..(y + 1) * src.width * 3];
            for (xx, &(xmin, n)) in c.bounds.iter().enumerate() {
                let k = &c.weights[xx * c.ksize..xx * c.ksize + n];
                for ch in 0..3 {
                    let mut ss = round;
                    for (x, &w) in k.iter().enumerate() {
                        ss += i32::from(row[(xmin + x) * 3 + ch]) * w;
                    }
                    out[(y * width + xx) * 3 + ch] = clip8(ss);
                }
            }
        }
        Rgb8 {
            width,
            height: src.height,
            data: out,
        }
    } else {
        Rgb8 {
            width: src.width,
            height: src.height,
            data: src.data.clone(),
        }
    };

    if height == horiz.height {
        return horiz;
    }
    let c = coeffs(horiz.height, height);
    let mut out = vec![0u8; width * height * 3];
    for (yy, &(ymin, n)) in c.bounds.iter().enumerate() {
        let k = &c.weights[yy * c.ksize..yy * c.ksize + n];
        for xx in 0..width {
            for ch in 0..3 {
                let mut ss = round;
                for (y, &w) in k.iter().enumerate() {
                    ss += i32::from(horiz.data[((ymin + y) * width + xx) * 3 + ch]) * w;
                }
                out[(yy * width + xx) * 3 + ch] = clip8(ss);
            }
        }
    }
    Rgb8 {
        width,
        height,
        data: out,
    }
}

/// Scale to [0, 1], normalise per channel, and lay out as NCHW with N = 1.
pub fn to_nchw(img: &Rgb8, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let plane = img.width * img.height;
    let mut out = vec![0f32; 3 * plane];
    for (i, px) in img.data.chunks_exact(3).enumerate() {
        for ch in 0..3 {
            out[ch * plane + i] = (f32::from(px[ch]) / 255.0 - mean[ch]) / std[ch];
        }
    }
    out
}

/// The whole transform: decode a JPEG, resize the whole frame to `height` x `width`,
/// normalise. This is the attribute model's training preprocessing.
pub fn image_input(
    bytes: &[u8],
    height: usize,
    width: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Vec<f32>, String> {
    // A zero output size divides by zero in the coefficients and surfaces as an
    // allocation panic far from the cause; refuse it here, where the cause is known.
    if height == 0 || width == 0 {
        return Err(format!(
            "cannot resize to {height}x{width}: both sides must be non-zero"
        ));
    }
    let img = decode_jpeg(bytes)?;
    Ok(to_nchw(&resize_bilinear(&img, width, height), mean, std))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three 8-bit levels in the channel with the smallest std (0.224), the tightest
    /// place a level difference can show. Measured: on the public test image the
    /// tensor differs from Pillow's by at most one level; over 30 real-world frames at
    /// 1280 x 854, at most two (zune-jpeg and libjpeg-turbo disagree on 2-3% of
    /// decoded bytes, by up to 4 levels, which the resize averages down). The
    /// attribute model's five top-1 answers were identical on all 30 frames.
    const DECODE_TOLERANCE: f32 = 3.0 / 255.0 / 0.224;

    /// The pattern the expected bytes below were generated from, with Pillow 12.3:
    /// `Image.frombytes("RGB", (w, h), src).resize((dw, dh), Image.BILINEAR)`.
    fn pattern(w: usize, h: usize) -> Rgb8 {
        let data = (0..h)
            .flat_map(|y| (0..w).flat_map(move |x| (0..3).map(move |c| (x, y, c))))
            .map(|(x, y, c)| ((x * 37 + y * 101 + c * 59 + (x * y) % 7 * 23) % 256) as u8)
            .collect();
        Rgb8 {
            width: w,
            height: h,
            data,
        }
    }

    #[test]
    fn downscale_matches_pillow_byte_for_byte() {
        let expected: [u8; 60] = [
            100, 118, 130, 120, 151, 130, 142, 101, 96, 111, 116, 139, 131, 156, 115, 108, 120,
            105, 135, 110, 139, 132, 114, 124, 117, 107, 116, 114, 90, 143, 123, 129, 120, 86, 122,
            178, 102, 117, 126, 116, 130, 111, 78, 115, 167, 115, 123, 127, 86, 110, 152, 119, 143,
            126, 108, 100, 128, 86, 123, 176,
        ];
        assert_eq!(resize_bilinear(&pattern(13, 9), 5, 4).data, expected);
    }

    #[test]
    fn upscale_matches_pillow_byte_for_byte() {
        let expected: [u8; 189] = [
            0, 59, 118, 19, 78, 137, 43, 102, 161, 68, 127, 186, 93, 152, 211, 117, 176, 192, 142,
            201, 47, 167, 226, 29, 185, 244, 47, 36, 95, 154, 59, 118, 131, 89, 132, 115, 119, 102,
            146, 104, 117, 176, 88, 147, 178, 118, 177, 99, 148, 207, 56, 171, 230, 33, 94, 153,
            212, 123, 182, 122, 162, 181, 42, 201, 62, 81, 121, 61, 120, 41, 100, 156, 80, 139,
            183, 119, 178, 100, 148, 207, 10, 152, 83, 142, 124, 119, 114, 107, 145, 97, 155, 107,
            145, 139, 134, 129, 109, 168, 120, 103, 162, 199, 137, 132, 127, 173, 104, 35, 191, 12,
            71, 115, 55, 105, 49, 108, 152, 93, 152, 208, 150, 200, 140, 182, 226, 87, 126, 182,
            201, 159, 90, 149, 204, 7, 66, 102, 70, 129, 106, 119, 96, 98, 157, 79, 53, 112, 143,
            122, 99, 112, 180, 102, 85, 97, 129, 172, 156, 87, 146, 235, 38, 97, 47, 106, 165, 100,
            159, 90, 128, 187, 33, 28, 87, 103, 104, 35, 94, 179, 25, 84, 79, 95, 154, 155, 86,
            145, 255, 58, 117,
        ];
        assert_eq!(resize_bilinear(&pattern(6, 4), 9, 7).data, expected);
    }

    #[test]
    fn same_size_is_identity() {
        let p = pattern(5, 4);
        assert_eq!(resize_bilinear(&p, 5, 4).data, p.data);
    }

    use std::{fs, path::PathBuf};

    /// A JPEG plus Pillow's side of the comparison, from `tools/preprocess_reference.py`.
    /// `DASH_IMAGE_FIXTURE` names the stem (default `dog`, the public test image).
    struct Reference {
        jpeg: Vec<u8>,
        rgb: Rgb8,
        out: (usize, usize),
        resized: Vec<u8>,
        tensor: Vec<f32>,
    }

    fn reference() -> Reference {
        let dir = std::env::var("DASH_FIXTURES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"));
        let stem = std::env::var("DASH_IMAGE_FIXTURE").unwrap_or_else(|_| "dog".into());
        let read = |name: String| {
            fs::read(dir.join(&name))
                .unwrap_or_else(|e| panic!("{name}: {e}; run tools/preprocess_reference.py"))
        };
        let meta: serde_json::Value =
            serde_json::from_slice(&read(format!("{stem}.preprocess.json"))).unwrap();
        let n = |v: &serde_json::Value| v.as_u64().unwrap() as usize;
        let (h, w) = (n(&meta["out"][0]), n(&meta["out"][1]));
        Reference {
            jpeg: read(format!("{stem}.jpg")),
            rgb: Rgb8 {
                width: n(&meta["width"]),
                height: n(&meta["height"]),
                data: read(format!("{stem}_rgb.u8")),
            },
            out: (h, w),
            resized: read(format!("{stem}_{h}x{w}.u8")),
            tensor: read(format!("{stem}_{h}x{w}.f32"))
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }

    #[test]
    #[ignore = "needs tools/preprocess_reference.py's output; run with --ignored"]
    fn resize_of_pillow_pixels_is_pillow_byte_for_byte() {
        let r = reference();
        let got = resize_bilinear(&r.rgb, r.out.1, r.out.0);
        let differing = got
            .data
            .iter()
            .zip(&r.resized)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(got.data.len(), r.resized.len());
        assert_eq!(
            differing,
            0,
            "{differing} of {} bytes differ from Pillow",
            r.resized.len()
        );
    }

    #[test]
    #[ignore = "needs tools/preprocess_reference.py's output; run with --ignored"]
    fn jpeg_to_tensor_is_pillow_within_decoder_tolerance() {
        let r = reference();
        let decoded = decode_jpeg(&r.jpeg).unwrap();
        assert_eq!((decoded.width, decoded.height), (r.rgb.width, r.rgb.height));
        let worst_px = decoded
            .data
            .iter()
            .zip(&r.rgb.data)
            .map(|(a, b)| a.abs_diff(*b))
            .max();
        let off_px = decoded
            .data
            .iter()
            .zip(&r.rgb.data)
            .filter(|(a, b)| a != b)
            .count();
        let t = image_input(&r.jpeg, r.out.0, r.out.1, IMAGENET_MEAN, IMAGENET_STD).unwrap();
        let worst = t
            .iter()
            .zip(&r.tensor)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "decode: {off_px} of {} bytes differ from libjpeg-turbo, worst {worst_px:?} levels; \
             tensor: worst {worst:.4}",
            r.rgb.data.len()
        );
        assert!(
            worst <= DECODE_TOLERANCE,
            "tensor diverges by {worst} from Pillow's"
        );
    }

    #[test]
    fn zero_output_size_is_an_error_not_a_panic() {
        for (h, w) in [(0, 224), (224, 0)] {
            let err = image_input(&[], h, w, IMAGENET_MEAN, IMAGENET_STD).unwrap_err();
            assert!(err.contains("non-zero"), "{err}");
        }
    }

    #[test]
    fn nchw_layout_and_normalisation() {
        let img = Rgb8 {
            width: 2,
            height: 1,
            data: vec![255, 0, 51, 0, 255, 102],
        };
        let t = to_nchw(&img, [0.0; 3], [1.0; 3]);
        assert_eq!(t, vec![1.0, 0.0, 0.0, 1.0, 0.2, 0.4]);
    }
}
