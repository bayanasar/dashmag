//! Rotary embeddings against transformers' form, worked out in f64, and against transformers 5.15's
//! own rotations on torch 2.13 (CPU), recorded in `tests/fixtures/rope/rope.safetensors` from
//! seeded `[1, 12, 2, 64]` inputs: `LlamaRotaryEmbedding` with `apply_rotary_pos_emb` for every
//! `rope_type` dash-mag takes (default, linear, dynamic, YaRN with and without `mscale`, Llama 3),
//! at positions up to 16392, in f32 and for YaRN in bf16; and GPT-J's `create_sinusoidal_positions`
//! with its `apply_rotary_pos_emb` for the features paired as neighbours.

mod common;

use common::{close, relative, tensor};
use dash_mag::rope::{Rope, RopeLayout, RopeOptions, RopeScaling, frequencies};
use dash_mag::{DType, Tensor};
use safetensors::{Dtype as StDtype, SafeTensors};
use std::collections::HashMap;

#[test]
fn it_is_transformers_rotation() {
    let (dim, base) = (4, 10000f32);
    let rope = Rope::new(dim, base, 8, DType::F32).unwrap();
    let x = tensor(1, &[1, 2, 1, 6], DType::F32);
    let y = rope.apply_range(&x, 3..5).unwrap();
    let values = x.to_f32().unwrap();
    let frequency = |j: usize| 1.0 / base.powf((2 * (j % 2)) as f32 / dim as f32);
    let mut want = Vec::new();
    for (t, position) in (3..5).enumerate() {
        let row = &values[t * 6..t * 6 + 6];
        // x cos + rotate_half(x) sin, rotate_half(x) = (-x3, -x4, x1, x2).
        let rotated = [-row[2], -row[3], row[0], row[1]];
        for j in 0..4 {
            let angle = (position as f32 * frequency(j)) as f64;
            want.push(row[j] * angle.cos() as f32 + rotated[j] * angle.sin() as f32);
        }
        want.extend_from_slice(&row[4..]);
    }
    close(&y, &want, 1e-6);
}

#[test]
fn a_position_at_a_time_is_the_range_s_row() {
    let rope = Rope::new(8, 10000.0, 16, DType::BF16).unwrap();
    let x = tensor(2, &[1, 4, 3, 16], DType::BF16);
    let whole = rope.apply_range(&x, 5..9).unwrap().to_f32().unwrap();
    for t in 0..4 {
        let token = x.slice(1, t..t + 1).unwrap().squeeze(1).unwrap();
        let at = Tensor::from_i32(&[5 + t as i32], &[1]).unwrap();
        let one = rope.apply_at(&token, &at).unwrap().to_f32().unwrap();
        assert_eq!(relative(&one, &whole[t * 48..(t + 1) * 48]), 0.0);
    }
}

#[test]
fn it_refuses_what_it_does_not_hold() {
    let rope = Rope::new(8, 10000.0, 4, DType::F32).unwrap();
    assert!(Rope::new(7, 10000.0, 4, DType::F32).is_err());
    let x = tensor(3, &[1, 4, 1, 16], DType::F32);
    assert!(rope.apply_range(&x, 1..5).is_err(), "past the table");
    assert!(rope.apply_range(&x, 0..3).is_err(), "not T positions");
    assert!(
        rope.apply_range(&tensor(4, &[1, 4, 1, 6], DType::F32), 0..4)
            .is_err(),
        "too narrow"
    );
}

/// `Rope` keeps f64 on every side. It rotates at f32's accuracy where it holds tables and better where
/// it does not, so it is held to f32's.
#[test]
fn it_rotates_f64_as_it_does_f32() {
    let rope = Rope::new(4, 10000.0, 8, DType::F64).unwrap();
    let narrow = Rope::new(4, 10000.0, 8, DType::F32).unwrap();
    let x = tensor(5, &[1, 2, 1, 6], DType::F64);
    let y = rope.apply_range(&x, 3..5).unwrap();
    assert_eq!(y.dtype(), DType::F64);
    let want = narrow
        .apply_range(&x.as_dtype(DType::F32).unwrap(), 3..5)
        .unwrap();
    close(&y, &want.to_f32().unwrap(), 1e-6);

    let token = x.slice(1, 0..1).unwrap().squeeze(1).unwrap();
    let at = Tensor::from_i32(&[3], &[1]).unwrap();
    let one = rope.apply_at(&token, &at).unwrap();
    assert_eq!(one.dtype(), DType::F64);
    let row = want.slice(1, 0..1).unwrap().squeeze(1).unwrap();
    close(&one, &row.to_f32().unwrap(), 1e-6);
}

/// A rotation is of one dtype, on the side with tables and on the side without.
#[test]
fn it_refuses_a_dtype_other_than_its_own() {
    let rope = Rope::new(4, 10000.0, 8, DType::F32).unwrap();
    let x = tensor(6, &[1, 2, 1, 6], DType::BF16);
    assert!(rope.apply_range(&x, 3..5).is_err());
    let token = x.slice(1, 0..1).unwrap().squeeze(1).unwrap();
    let at = Tensor::from_i32(&[3], &[1]).unwrap();
    assert!(rope.apply_at(&token, &at).is_err());
    assert!(Rope::new(4, 10000.0, 8, DType::I32).is_err());
}

/// Positions are integers on every side, on the side with tables and on the side without.
#[test]
fn it_refuses_positions_that_are_not_integers() {
    let rope = Rope::new(4, 10000.0, 8, DType::F32).unwrap();
    let token = tensor(7, &[2, 1, 6], DType::F32);
    let floats = Tensor::from_f32(&[1.0, 3.0], &[2]).unwrap();
    assert!(rope.apply_at(&token, &floats).is_err());
    let booleans = Tensor::from_bool(&[true, false], &[2]).unwrap();
    assert!(rope.apply_at(&token, &booleans).is_err());

    let narrow = Tensor::from_i32(&[1, 3], &[2]).unwrap();
    let wide = narrow.as_dtype(DType::I64).unwrap();
    assert_eq!(
        rope.apply_at(&token, &wide).unwrap().to_f32().unwrap(),
        rope.apply_at(&token, &narrow).unwrap().to_f32().unwrap()
    );
}

/// One recorded tensor's values in f64, and its shape and dtype.
struct Recorded {
    values: Vec<f64>,
    shape: Vec<usize>,
    dtype: StDtype,
}

fn recordings() -> HashMap<String, Recorded> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/rope/rope.safetensors"
    );
    let bytes = std::fs::read(path).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    file.tensors()
        .into_iter()
        .map(|(name, view)| {
            let data = view.data();
            let values: Vec<f64> = match view.dtype() {
                StDtype::F32 => data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b) as f64)
                    .collect(),
                StDtype::F64 => data
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| f64::from_le_bytes(*b))
                    .collect(),
                StDtype::BF16 => data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16) as f64)
                    .collect(),
                StDtype::I64 => data
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_le_bytes(*b) as f64)
                    .collect(),
                other => panic!("{name}: {other:?}"),
            };
            let recorded = Recorded {
                values,
                shape: view.shape().to_vec(),
                dtype: view.dtype(),
            };
            (name, recorded)
        })
        .collect()
}

/// The largest difference relative to the largest reference value.
fn worst(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0f64, |m, (g, w)| m.max((*g as f64 - w).abs()))
        / scale
}

/// Each case: its name, the base, the options, and the length its rotation holds (its last
/// position plus one, as transformers' dynamic scaling sees the sequence).
fn cases() -> Vec<(&'static str, f32, RopeOptions)> {
    let halves = |scaling| RopeOptions {
        layout: RopeLayout::Halves,
        scaling,
    };
    let pairs = RopeOptions {
        layout: RopeLayout::Pairs,
        scaling: RopeScaling::None,
    };
    vec![
        ("default", 10000.0, halves(RopeScaling::None)),
        ("default_far", 10000.0, halves(RopeScaling::None)),
        (
            "linear",
            10000.0,
            halves(RopeScaling::Linear { factor: 4.0 }),
        ),
        (
            "dynamic",
            10000.0,
            halves(RopeScaling::Dynamic {
                factor: 2.0,
                max_positions: 64,
            }),
        ),
        ("yarn", 10000.0, halves(RopeScaling::yarn(4.0, 2048))),
        ("yarn_bf16", 10000.0, halves(RopeScaling::yarn(4.0, 2048))),
        (
            "yarn_mscale",
            10000.0,
            halves(RopeScaling::Yarn {
                factor: 40.0,
                original_max_positions: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: None,
                mscale: Some(1.0),
                mscale_all_dim: Some(0.707),
                truncate: true,
            }),
        ),
        (
            "llama3",
            500000.0,
            halves(RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_positions: 8192,
            }),
        ),
        ("pairs", 10000.0, pairs),
        ("pairs_far", 10000.0, pairs),
    ]
}

/// Every case against transformers' own: the frequencies and the attention factor where it records
/// them, and the rotation of a recorded input, in f32 within a few f32 roundings of the largest
/// output and in bf16 within one bf16 unit; and one position at a time the range's bits. On MLX the
/// default rotation is MLX's own `fast` one, which keeps the numbers an application had there and
/// is not held to transformers' at these positions: its cases are left to the tests above.
#[test]
fn every_rope_type_is_transformers_own() {
    let recorded = recordings();
    for (name, base, options) in cases() {
        if cfg!(target_os = "macos") && options == RopeOptions::default() {
            continue;
        }
        let get = |k: &str| &recorded[&format!("{name}.{k}")];
        let positions = &get("positions").values;
        let (start, end) = (positions[0] as usize, positions[1] as usize);
        let dim = *get("x").shape.last().unwrap();
        if let Some(inv) = recorded.get(&format!("{name}.inv_freq")) {
            let (ours, attention) = frequencies(dim, base, options.scaling, end);
            let off = worst(&ours, &inv.values);
            let theirs = recorded[&format!("{name}.attention")].values[0];
            println!(
                "{name}: frequencies {off:.2e} apart, attention {attention} (transformers {theirs})"
            );
            assert!(
                off <= 2.0 * f32::EPSILON as f64,
                "{name}: frequencies {off:e}"
            );
            assert_eq!(attention, theirs as f32, "{name}: attention factor");
        }
        let x = get("x");
        let dtype = match x.dtype {
            StDtype::BF16 => DType::BF16,
            _ => DType::F32,
        };
        let input: Vec<f32> = x.values.iter().map(|v| *v as f32).collect();
        let x_t = Tensor::from_f32(&input, &x.shape)
            .unwrap()
            .as_dtype(dtype)
            .unwrap();
        let rope = Rope::with_options(dim, base, end, dtype, options).unwrap();
        let y = rope.apply_range(&x_t, start..end).unwrap();
        let got = y.as_dtype(DType::F32).unwrap().to_f32().unwrap();
        let off = worst(&got, &get("y").values);
        let bound = match dtype {
            DType::BF16 => 1.0 / 128.0,
            _ => 4.0 * f32::EPSILON as f64,
        };
        println!("{name}: the rotation {off:.2e} apart (bound {bound:.1e})");
        assert!(off <= bound, "{name}: {off:e} above {bound:e}");
        // One position at a time, as decoding asks.
        let shape = &x.shape;
        let (heads, d) = (shape[2], shape[3]);
        for t in [0, shape[1] - 1] {
            let token = x_t.slice(1, t..t + 1).unwrap().squeeze(1).unwrap();
            let at = Tensor::from_i32(&[(start + t) as i32], &[1]).unwrap();
            let one = rope.apply_at(&token, &at).unwrap();
            let row = y.slice(1, t..t + 1).unwrap().squeeze(1).unwrap();
            assert_eq!(one.shape(), [1, heads, d]);
            assert_eq!(
                relative(
                    &one.as_dtype(DType::F32).unwrap().to_f32().unwrap(),
                    &row.as_dtype(DType::F32).unwrap().to_f32().unwrap()
                ),
                0.0,
                "{name} position {}",
                start + t
            );
        }
    }
}
