//! The array operations: what each computes, its axes, broadcasting and dtypes, and what it
//! refuses. The expected values are worked out by hand, so the same file holds every side.

#![cfg(not(target_os = "macos"))]

mod common;

use common::{bits, close, tensor};
use dash_mag::{DType, Tensor};

fn t(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(values, shape).unwrap()
}

#[test]
fn bytes_round_trip_in_every_floating_dtype() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let x = tensor(1, &[2, 3], dtype);
        assert_eq!(x.dtype(), dtype);
        assert_eq!(x.shape(), [2, 3]);
        let bytes = x.to_bytes().unwrap();
        let back = Tensor::from_bytes(&bytes, &[2, 3], dtype).unwrap();
        assert_eq!(back.to_bytes().unwrap(), bytes, "{dtype:?}");
    }
}

#[test]
fn rank_zero_is_a_shape_of_nothing() {
    let x = Tensor::scalar(2.5, DType::F32).unwrap();
    assert_eq!(x.shape(), Vec::<usize>::new());
    assert_eq!(x.item().unwrap(), 2.5);
    let s = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).sum().unwrap();
    assert_eq!(s.shape(), Vec::<usize>::new());
    assert_eq!(s.item().unwrap(), 10.0);
    // It broadcasts, and it reshapes to and from rank 1.
    close(&x.add(&t(&[1.0, 2.0], &[2])).unwrap(), &[3.5, 4.5], 0.0);
    assert_eq!(x.expand_dims(0).unwrap().shape(), [1]);
    assert_eq!(
        x.expand_dims(0).unwrap().squeeze(0).unwrap().shape(),
        Vec::<usize>::new()
    );
}

#[test]
fn arithmetic_broadcasts_as_numpy_does() {
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(&[10.0, 20.0, 30.0], &[3]);
    close(
        &a.add(&b).unwrap(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        0.0,
    );
    let c = t(&[2.0, 3.0], &[2, 1]);
    close(&a.mul(&c).unwrap(), &[2.0, 4.0, 6.0, 12.0, 15.0, 18.0], 0.0);
    close(&a.sub(&c).unwrap(), &[-1.0, 0.0, 1.0, 1.0, 2.0, 3.0], 0.0);
    close(
        &a.div(&c).unwrap(),
        &[0.5, 1.0, 1.5, 4.0 / 3.0, 5.0 / 3.0, 2.0],
        1e-7,
    );
    close(
        &a.maximum(&c).unwrap(),
        &[2.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        0.0,
    );
    close(
        &a.minimum(&c).unwrap(),
        &[1.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        0.0,
    );
    assert!(a.add(&t(&[1.0, 2.0], &[2])).is_err());
}

#[test]
fn dtypes_meet_in_the_wider() {
    let bf = tensor(1, &[4], DType::BF16);
    let hf = tensor(2, &[4], DType::F16);
    let f = tensor(3, &[4], DType::F32);
    assert_eq!(bf.add(&bf).unwrap().dtype(), DType::BF16);
    assert_eq!(bf.add(&hf).unwrap().dtype(), DType::F32);
    assert_eq!(bf.add(&f).unwrap().dtype(), DType::F32);
    assert_eq!(hf.mul(&f).unwrap().dtype(), DType::F32);
    // Exactly what a cast and an f32 sum give.
    let want = bf.as_dtype(DType::F32).unwrap().add(&f).unwrap();
    assert_eq!(bits(&bf.add(&f).unwrap()), bits(&want));
    let m = t(&[1.0, 2.0], &[1, 2]).as_dtype(DType::BF16).unwrap();
    let n = t(&[3.0, 4.0], &[2, 1]);
    assert_eq!(m.matmul(&n).unwrap().dtype(), DType::F32);
}

#[test]
fn elementwise_functions() {
    let x = t(&[-1.0, 0.0, 2.0], &[3]);
    close(&x.neg().unwrap(), &[1.0, -0.0, -2.0], 0.0);
    close(&x.abs().unwrap(), &[1.0, 0.0, 2.0], 0.0);
    close(&x.square().unwrap(), &[1.0, 0.0, 4.0], 0.0);
    let (e, s) = (1f32.exp(), |v: f32| 1.0 / (1.0 + (-v).exp()));
    close(&x.exp().unwrap(), &[1.0 / e, 1.0, e * e], 1e-6);
    close(&t(&[1.0, 4.0], &[2]).sqrt().unwrap(), &[1.0, 2.0], 0.0);
    close(&t(&[1.0, e], &[2]).log().unwrap(), &[0.0, 1.0], 1e-6);
    close(&x.sigmoid().unwrap(), &[s(-1.0), 0.5, s(2.0)], 1e-6);
    close(&x.silu().unwrap(), &[-s(-1.0), 0.0, 2.0 * s(2.0)], 1e-6);
    let p = |v: f32| (1.0 + v.exp()).ln();
    close(&x.softplus().unwrap(), &[p(-1.0), p(0.0), p(2.0)], 1e-6);
    close(&x.clip(-0.5, 1.0).unwrap(), &[-0.5, 0.0, 1.0], 0.0);
}

#[test]
fn matmul_over_the_last_two_axes() {
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
    close(&a.matmul(&b).unwrap(), &[4.0, 5.0, 10.0, 11.0], 0.0);
    // A batch broadcast, and a row and a column.
    let batch = t(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);
    let m = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    close(
        &batch.matmul(&m).unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
        0.0,
    );
    let v = t(&[1.0, 2.0], &[2]);
    let row = v.matmul(&m).unwrap();
    assert_eq!(row.shape(), [2]);
    close(&row, &[7.0, 10.0], 0.0);
    let column = m.matmul(&v).unwrap();
    assert_eq!(column.shape(), [2]);
    close(&column, &[5.0, 11.0], 0.0);
    assert_eq!(v.matmul(&v).unwrap().item().unwrap(), 5.0);
    assert!(a.matmul(&a).is_err());
}

#[test]
fn shapes() {
    let x = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    assert_eq!(x.reshape(&[3, -1]).unwrap().shape(), [3, 2]);
    assert!(x.reshape(&[4, -1]).is_err());
    assert!(x.reshape(&[-1, -1]).is_err());
    assert_eq!(x.expand_dims(-1).unwrap().shape(), [2, 3, 1]);
    assert_eq!(
        x.expand_dims(1).unwrap().squeeze(1).unwrap().shape(),
        [2, 3]
    );
    assert!(x.squeeze(0).is_err());
    close(&x.t().unwrap(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0], 0.0);
    close(
        &x.swap_axes(0, -1).unwrap(),
        &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
        0.0,
    );
    assert_eq!(x.flatten(0, -1).unwrap().shape(), [6]);
    assert!(x.transpose(&[0, 0]).is_err());
    let cube = tensor(1, &[2, 3, 4], DType::F32);
    assert_eq!(cube.transpose(&[2, 0, 1]).unwrap().shape(), [4, 2, 3]);
    assert_eq!(cube.flatten(1, 2).unwrap().shape(), [2, 12]);
    assert_eq!(
        t(&[1.0, 2.0], &[2, 1])
            .broadcast_to(&[3, 2, 4])
            .unwrap()
            .shape(),
        [3, 2, 4]
    );
    assert!(t(&[1.0, 2.0], &[2]).broadcast_to(&[3]).is_err());
    assert!(
        t(&[1.0], &[1, 1, 1, 1, 1, 1]).expand_dims(0).is_err(),
        "rank 7"
    );
}

#[test]
fn joining_cutting_and_picking() {
    let x = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    close(
        &Tensor::concat(&[&x, &x], 0)
            .unwrap()
            .slice(0, 2..4)
            .unwrap(),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        0.0,
    );
    assert_eq!(Tensor::concat(&[&x, &x], -1).unwrap().shape(), [2, 6]);
    assert!(Tensor::concat(&[&x, &x.t().unwrap()], 0).is_err());
    assert_eq!(Tensor::stack(&[&x, &x, &x], 1).unwrap().shape(), [2, 3, 3]);
    let parts = x.split_at(&[1, 1, 2], -1).unwrap();
    assert_eq!(
        parts.iter().map(|p| p.shape()[1]).collect::<Vec<_>>(),
        [1, 0, 1, 1]
    );
    close(&parts[3], &[2.0, 5.0], 0.0);
    close(&x.slice(1, 1..3).unwrap(), &[1.0, 2.0, 4.0, 5.0], 0.0);
    assert!(x.slice(1, 1..4).is_err());
    let at = Tensor::from_i32(&[2, 0], &[2]).unwrap();
    close(&x.take(&at, 1).unwrap(), &[2.0, 0.0, 5.0, 3.0], 0.0);
    assert!(x.take(&x, 1).is_err());
    let padded = t(&[1.0, 2.0], &[2]).pad(0, 1, 2, -1.0).unwrap();
    close(&padded, &[-1.0, 1.0, 2.0, -1.0, -1.0], 0.0);
}

#[test]
fn reductions() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 9.0], &[2, 3]);
    assert_eq!(x.sum().unwrap().item().unwrap(), 24.0);
    assert_eq!(x.mean().unwrap().item().unwrap(), 4.0);
    assert_eq!(x.max().unwrap().item().unwrap(), 9.0);
    assert_eq!(x.min().unwrap().item().unwrap(), 1.0);
    let kept = x.sum_axis(1, true).unwrap();
    assert_eq!(kept.shape(), [2, 1]);
    close(&kept, &[6.0, 18.0], 0.0);
    let dropped = x.mean_axis(-1, false).unwrap();
    assert_eq!(dropped.shape(), [2]);
    close(&dropped, &[2.0, 6.0], 0.0);
    close(&x.max_axis(0, false).unwrap(), &[4.0, 5.0, 9.0], 0.0);
    close(&x.min_axis(0, false).unwrap(), &[1.0, 2.0, 3.0], 0.0);
    let v = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
    close(&v.var_axis(0, 0, false).unwrap(), &[1.25], 1e-6);
    close(&v.var_axis(0, 1, false).unwrap(), &[5.0 / 3.0], 1e-6);
    assert!(t(&[1.0], &[1]).var_axis(0, 1, false).is_err());
    assert_eq!(x.argmax(1, false).unwrap().to_i32().unwrap(), [2, 2]);
    assert_eq!(x.argmax(0, true).unwrap().shape(), [1, 3]);
    close(
        &t(&[3.0, 1.0, 2.0], &[3]).sort(0).unwrap(),
        &[1.0, 2.0, 3.0],
        0.0,
    );
}

#[test]
fn comparisons_and_selection() {
    let (a, b) = (t(&[1.0, 2.0, 3.0], &[3]), t(&[2.0, 2.0, 2.0], &[3]));
    assert_eq!(a.gt(&b).unwrap().to_bool().unwrap(), [false, false, true]);
    assert_eq!(a.ge(&b).unwrap().to_bool().unwrap(), [false, true, true]);
    assert_eq!(a.lt(&b).unwrap().to_bool().unwrap(), [true, false, false]);
    assert_eq!(a.le(&b).unwrap().to_bool().unwrap(), [true, true, false]);
    assert_eq!(a.eq(&b).unwrap().to_bool().unwrap(), [false, true, false]);
    let c = a.gt(&b).unwrap();
    let d = a.lt(&b).unwrap();
    assert_eq!(
        c.logical_or(&d).unwrap().to_bool().unwrap(),
        [true, false, true]
    );
    assert_eq!(
        c.logical_and(&d).unwrap().to_bool().unwrap(),
        [false, false, false]
    );
    assert_eq!(
        c.logical_not().unwrap().to_bool().unwrap(),
        [true, true, false]
    );
    close(&c.select(&a, &b).unwrap(), &[2.0, 2.0, 3.0], 0.0);
    assert!(a.select(&a, &b).is_err());
    assert!(a.logical_and(&b).is_err());
}

#[test]
fn casts_between_kinds() {
    let m = Tensor::from_bool(&[true, false], &[2]).unwrap();
    close(&m.as_dtype(DType::F32).unwrap(), &[1.0, 0.0], 0.0);
    let i = Tensor::from_i32(&[3, -2], &[2]).unwrap();
    close(&i.as_dtype(DType::F32).unwrap(), &[3.0, -2.0], 0.0);
    assert_eq!(
        t(&[3.5, -2.0], &[2])
            .as_dtype(DType::I32)
            .unwrap()
            .to_i32()
            .unwrap(),
        [3, -2]
    );
    assert_eq!(
        i.as_dtype(DType::Bool).unwrap().to_bool().unwrap(),
        [true, true]
    );
    assert_eq!(
        tensor(9, &[2, 2], DType::F32)
            .reshape(&[-1])
            .unwrap()
            .shape(),
        [4]
    );
    assert_eq!(i.reshape(&[1, 2]).unwrap().to_i32().unwrap(), [3, -2]);
    assert_eq!(
        m.reshape(&[2, 1]).unwrap().to_bool().unwrap(),
        [true, false]
    );
}

#[test]
fn rms_norm_is_computed_in_f32_and_rounded_once() {
    let x = t(&[3.0, 4.0], &[1, 2]);
    let w = t(&[2.0, 1.0], &[2]);
    let denominator = 12.5f32.sqrt();
    close(
        &x.rms_norm(None, 0.0).unwrap(),
        &[3.0 / denominator, 4.0 / denominator],
        1e-6,
    );
    close(
        &x.rms_norm(Some(&w), 0.0).unwrap(),
        &[6.0 / denominator, 4.0 / denominator],
        1e-6,
    );
    let bf = x
        .as_dtype(DType::BF16)
        .unwrap()
        .rms_norm(None, 0.0)
        .unwrap();
    assert_eq!(bf.dtype(), DType::BF16);
    let want = x
        .rms_norm(None, 0.0)
        .unwrap()
        .as_dtype(DType::BF16)
        .unwrap();
    assert_eq!(bits(&bf), bits(&want));
}
