//! What the tests share: values that are the same on every side, and comparisons.

#![allow(dead_code)]

use dash_mag::{DType, Tensor};

fn splitmix64(z: u64) -> u64 {
    let z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// `n` values in `[-1, 1)`, from `seed`, with few significant bits so that f32, bf16 and f16 hold
/// them exactly and every side starts from the same numbers.
pub fn values(seed: u64, n: usize) -> Vec<f32> {
    (0..n as u64)
        .map(|i| {
            let k = (splitmix64((seed << 32) | i) >> 56) as i32 - 128;
            k as f32 / 128.0
        })
        .collect()
}

pub fn tensor(seed: u64, shape: &[usize], dtype: DType) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(&values(seed, n), shape)
        .unwrap()
        .as_dtype(dtype)
        .unwrap()
}

pub fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// The largest absolute difference, over the largest absolute value of `want`.
pub fn relative(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs() as f64));
    let worst = got
        .iter()
        .zip(want)
        .fold(0f64, |m, (a, b)| m.max((*a as f64 - *b as f64).abs()));
    if scale == 0.0 { worst } else { worst / scale }
}

pub fn close(t: &Tensor, want: &[f32], bound: f64) {
    let got = t.to_f32().unwrap();
    let error = relative(&got, want);
    assert!(
        error <= bound,
        "{got:?} against {want:?}: {error:e} above {bound:e}"
    );
}
