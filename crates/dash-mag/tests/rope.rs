//! Rotary embeddings against transformers' form, worked out in f64.

mod common;

use common::{close, relative, tensor};
use dash_mag::rope::Rope;
use dash_mag::{DType, Tensor};

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
