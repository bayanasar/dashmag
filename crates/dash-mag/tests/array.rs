//! The array operations: what each computes, its axes, broadcasting and dtypes, and what it
//! refuses. The expected values are worked out by hand, so the same file holds every side.

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

/// The same code runs on every side: `where(done, pad, token)` over token ids, as a generation loop
/// writes it, and over booleans.
#[test]
fn selection_takes_every_kind() {
    let done = Tensor::from_bool(&[true, false, true], &[3]).unwrap();
    let pad = Tensor::from_i32(&[7, 7, 7], &[3]).unwrap();
    let tokens = Tensor::from_i32(&[1, 2, 3], &[3]).unwrap();
    let picked = done.select(&pad, &tokens).unwrap();
    assert_eq!(picked.dtype(), DType::I32);
    assert_eq!(picked.to_i32().unwrap(), [7, 2, 7]);
    // A scalar's broadcast too.
    let one = Tensor::from_i32(&[9], &[1]).unwrap();
    assert_eq!(
        done.select(&one, &tokens).unwrap().to_i32().unwrap(),
        [9, 2, 9]
    );
    let yes = Tensor::from_bool(&[true, true, true], &[3]).unwrap();
    let no = Tensor::from_bool(&[false, false, false], &[3]).unwrap();
    let flags = done.select(&no, &yes).unwrap();
    assert_eq!(flags.dtype(), DType::Bool);
    assert_eq!(flags.to_bool().unwrap(), [false, true, false]);
    // Kinds meet as dtypes do.
    let values = t(&[0.5, 0.5, 0.5], &[3]);
    let mixed = done.select(&values, &tokens).unwrap();
    assert_eq!(mixed.dtype(), DType::F32);
    close(&mixed, &[0.5, 2.0, 0.5], 0.0);
}

#[test]
fn integers_compare_and_booleans_are_equal_or_not() {
    let tokens = Tensor::from_i32(&[1, 2, 3], &[3]).unwrap();
    let eos = Tensor::from_i32(&[2], &[1]).unwrap();
    assert_eq!(
        tokens.eq(&eos).unwrap().to_bool().unwrap(),
        [false, true, false]
    );
    assert_eq!(
        tokens.gt(&eos).unwrap().to_bool().unwrap(),
        [false, false, true]
    );
    assert_eq!(
        tokens.le(&eos).unwrap().to_bool().unwrap(),
        [true, true, false]
    );
    let a = Tensor::from_bool(&[true, false], &[2]).unwrap();
    let b = Tensor::from_bool(&[true, true], &[2]).unwrap();
    assert_eq!(a.eq(&b).unwrap().to_bool().unwrap(), [true, false]);
    assert!(a.gt(&b).is_err(), "booleans have no order");
}

#[test]
fn integers_have_one_width_until_asked_for_another() {
    let i = Tensor::from_i32(&[3, -2], &[2]).unwrap();
    assert_eq!(i.dtype(), DType::I32);
    let wide = i.as_dtype(DType::I64).unwrap();
    assert_eq!(wide.dtype(), DType::I64);
    assert_eq!(wide.to_i32().unwrap(), [3, -2]);
    let narrow = wide.as_dtype(DType::I32).unwrap();
    assert_eq!(narrow.dtype(), DType::I32);
    assert_eq!(narrow.to_i32().unwrap(), [3, -2]);
    let from_float = t(&[3.5], &[1]);
    assert_eq!(from_float.as_dtype(DType::I64).unwrap().dtype(), DType::I64);
    assert_eq!(from_float.as_dtype(DType::I32).unwrap().dtype(), DType::I32);
    let from_bool = Tensor::from_bool(&[true], &[1]).unwrap();
    assert_eq!(from_bool.as_dtype(DType::I64).unwrap().dtype(), DType::I64);
    assert_eq!(from_bool.as_dtype(DType::I32).unwrap().dtype(), DType::I32);
    assert_eq!(
        t(&[1.0, 2.0], &[1, 2]).argmax(1, false).unwrap().dtype(),
        DType::I32
    );
}

#[test]
fn matmul_of_integers_is_refused_here() {
    let i = Tensor::from_i32(&[1, 2, 3, 4], &[2, 2]).unwrap();
    assert!(i.matmul(&i).is_err());
    assert!(
        Tensor::from_bool(&[true], &[1, 1])
            .unwrap()
            .matmul(&t(&[1.0], &[1, 1]))
            .is_ok(),
        "a boolean beside a float meets it as a float"
    );
}

#[test]
fn put_along_axis_writes_at_the_indices_along_the_axis() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // One place a row.
    let at = Tensor::from_i32(&[2, 0], &[2, 1]).unwrap();
    let v = t(&[10.0, 20.0], &[2, 1]);
    close(
        &x.put_along_axis(&at, &v, 1).unwrap(),
        &[1.0, 2.0, 10.0, 20.0, 5.0, 6.0],
        0.0,
    );
    // The other axis, counted from the end.
    let at = Tensor::from_i32(&[1, 0, 1], &[1, 3]).unwrap();
    let v = t(&[7.0, 8.0, 9.0], &[1, 3]);
    close(
        &x.put_along_axis(&at, &v, -2).unwrap(),
        &[1.0, 8.0, 3.0, 7.0, 5.0, 9.0],
        0.0,
    );
    // Several places a row, and a third axis across which each row keeps its own.
    let at = Tensor::from_i32(&[0, 2, 1, 2], &[2, 2]).unwrap();
    let v = t(&[7.0, 8.0, 9.0, 10.0], &[2, 2]);
    close(
        &x.put_along_axis(&at, &v, 1).unwrap(),
        &[7.0, 2.0, 8.0, 4.0, 9.0, 10.0],
        0.0,
    );
    let cube = t(&[0.0; 12], &[2, 3, 2]);
    let at = Tensor::from_i32(&[2, 2, 0, 0], &[2, 1, 2]).unwrap();
    let v = t(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    close(
        &cube.put_along_axis(&at, &v, 1).unwrap(),
        &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        0.0,
    );
    // The values take the tensor's dtype.
    let y = x.as_dtype(DType::BF16).unwrap().put_along_axis(
        &Tensor::from_i32(&[0], &[1, 1]).unwrap(),
        &t(&[1.5], &[1, 1]),
        1,
    );
    assert!(y.is_err(), "a row short");
    let at = Tensor::from_i32(&[0, 1], &[2, 1]).unwrap();
    let y = x
        .as_dtype(DType::BF16)
        .unwrap()
        .put_along_axis(&at, &t(&[1.5, 2.5], &[2, 1]), 1)
        .unwrap();
    assert_eq!(y.dtype(), DType::BF16);
    close(&y, &[1.5, 2.0, 3.0, 4.0, 2.5, 6.0], 0.0);
}

#[test]
fn put_along_axis_refuses_what_does_not_fit() {
    let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let at = Tensor::from_i32(&[0, 1], &[2, 1]).unwrap();
    let v = t(&[1.0, 2.0], &[2, 1]);
    assert!(x.put_along_axis(&at, &v, 2).is_err(), "an axis it lacks");
    assert!(x.put_along_axis(&v, &v, 1).is_err(), "floating indices");
    assert!(at.put_along_axis(&at, &at, 1).is_err(), "an integer tensor");
    let short = Tensor::from_i32(&[0], &[1, 1]).unwrap();
    assert!(
        x.put_along_axis(&short, &t(&[1.0], &[1, 1]), 1).is_err(),
        "not this tensor's shape across the axis"
    );
    assert!(
        x.put_along_axis(&at, &t(&[1.0, 2.0], &[1, 2]), 1).is_err(),
        "values not the indices' shape"
    );
}

#[test]
fn logsumexp_axis_is_its_definition_and_does_not_overflow() {
    let x = t(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
    let want = [1.313_261_7, 3.313_261_7];
    close(&x.logsumexp_axis(1, false).unwrap(), &want, 1e-6);
    close(&x.logsumexp_axis(-1, false).unwrap(), &want, 1e-6);
    assert_eq!(x.logsumexp_axis(1, false).unwrap().shape(), [2]);
    assert_eq!(x.logsumexp_axis(1, true).unwrap().shape(), [2, 1]);
    // exp(1000) overflows f32; 1000 + ln 2 does not.
    let big = t(&[1000.0, 1000.0], &[2]);
    close(&big.logsumexp_axis(0, false).unwrap(), &[1000.6931], 1e-6);
    let i = Tensor::from_i32(&[1, 2], &[2]).unwrap();
    assert!(i.logsumexp_axis(0, false).is_err());
    assert!(x.logsumexp_axis(2, false).is_err());
}

#[test]
fn eval_async_gives_what_eval_gives_and_memory_calls_ask_nothing_of_a_side() {
    let x = t(&[1.0, 2.0], &[2]).square().unwrap();
    dash_mag::eval_async(&[&x]).unwrap();
    close(&x, &[1.0, 4.0], 0.0);
    dash_mag::set_cache_limit(1 << 30);
    dash_mag::reset_peak_memory();
    // Each side says what it knows: MLX its allocator's figures, Burn none.
    assert_eq!(dash_mag::memory().is_some(), cfg!(target_os = "macos"));
}

/// A read gives row order whatever the data's layout: a transposed tensor's and a slice's strides
/// are the side's to undo.
#[test]
fn reads_are_in_row_order() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let turned = x.t().unwrap();
    assert_eq!(turned.to_f32().unwrap(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let column = x.slice(1, 1..2).unwrap();
    assert_eq!(column.to_f32().unwrap(), [2.0, 5.0]);
    // Row-contiguous at an offset, which is read in place.
    let row = x.slice(0, 1..2).unwrap();
    assert_eq!(row.to_f32().unwrap(), [4.0, 5.0, 6.0]);
    assert_eq!(row.to_bytes().unwrap().len(), 12);
    let i = Tensor::from_i32(&[1, 2, 3, 4], &[2, 2]).unwrap();
    assert_eq!(i.t().unwrap().to_i32().unwrap(), [1, 3, 2, 4]);
    let b = Tensor::from_bool(&[true, false, false, false], &[2, 2]).unwrap();
    assert_eq!(
        b.t().unwrap().to_bool().unwrap(),
        [true, false, false, false]
    );
    let b = Tensor::from_bool(&[false, true, false, false], &[2, 2]).unwrap();
    assert_eq!(
        b.t().unwrap().to_bool().unwrap(),
        [false, false, true, false]
    );
}

/// Where a row's largest element is infinite, it is the answer: an entirely masked row is `-inf`.
#[test]
fn logsumexp_axis_takes_the_largest_where_it_is_infinite() {
    let inf = f32::INFINITY;
    let x = t(&[-inf, -inf, inf, 1.0, -inf, 0.0], &[3, 2]);
    assert_eq!(
        x.logsumexp_axis(1, false).unwrap().to_f32().unwrap(),
        [-inf, inf, 0.0]
    );
    // Long enough for a side's SIMD path, too.
    let x = t(&[-inf; 8], &[8]);
    assert_eq!(x.logsumexp_axis(0, false).unwrap().item().unwrap(), -inf);
    let mut row = [1.0; 8];
    row[5] = inf;
    assert_eq!(
        t(&row, &[8])
            .logsumexp_axis(0, false)
            .unwrap()
            .item()
            .unwrap(),
        inf
    );
}

/// f64 against the definition, at rows short and long: MLX's own is exact below four elements only.
#[test]
fn logsumexp_axis_is_exact_in_f64() {
    let row = [0.1, 0.2, 0.47, -0.3, 0.84, 1.21, -0.08, 0.66];
    for n in [2, 3, 4, 8] {
        let x = Tensor::from_f64(&row[..n], &[n]).unwrap();
        let y = x.logsumexp_axis(0, false).unwrap();
        assert_eq!(y.dtype(), DType::F64);
        let got = f64::from_le_bytes(y.to_bytes().unwrap().try_into().unwrap());
        let want = row[..n].iter().map(|v| v.exp()).sum::<f64>().ln();
        assert!(
            (got - want).abs() <= 1e-15 * want.abs().max(1.0),
            "{n}: {got} against {want}"
        );
    }
}

/// An empty tensor is refused, whichever axis is empty.
#[test]
fn logsumexp_axis_refuses_an_empty_tensor() {
    for (shape, axis) in [(&[0, 2], 1), (&[0, 2], 0), (&[2, 0], 0), (&[2, 0], 1)] {
        let empty = Tensor::zeros(shape, DType::F32).unwrap();
        assert!(
            empty.logsumexp_axis(axis, false).is_err(),
            "{shape:?} along {axis}"
        );
    }
}

/// Refused before either side runs: one would refuse, the other would run them.
#[test]
fn rms_norm_refuses_what_is_not_floating_and_a_weight_not_of_its_features() {
    let i = Tensor::from_i32(&[1, 2], &[2]).unwrap();
    assert!(i.rms_norm(None, 1e-6).is_err(), "an integer x");
    let x = t(&[3.0, 4.0], &[1, 2]);
    assert!(x.rms_norm(Some(&i), 1e-6).is_err(), "an integer weight");
    for shape in [&[1, 2][..], &[1][..], &[][..], &[3][..]] {
        let n = shape.iter().product();
        let w = Tensor::ones(&[n], DType::F32).unwrap();
        let w = w
            .reshape(&shape.iter().map(|&d| d as isize).collect::<Vec<_>>())
            .unwrap();
        assert!(x.rms_norm(Some(&w), 1e-6).is_err(), "a weight of {shape:?}");
    }
    assert!(
        Tensor::scalar(2.0, DType::F32)
            .unwrap()
            .rms_norm(None, 1e-6)
            .is_err(),
        "no features"
    );
    assert!(x.rms_norm(Some(&t(&[2.0, 1.0], &[2])), 1e-6).is_ok());
}

/// A dimension past `i32::MAX` is refused on every side: MLX's shapes would wrap it. Each shape here
/// is empty, so that nothing is allocated if it gets through.
#[test]
fn a_dimension_past_i32_is_refused() {
    let past = i32::MAX as usize + 1;
    let empty = Tensor::zeros(&[0, 2], DType::F32).unwrap();
    assert!(Tensor::zeros(&[past, 0], DType::F32).is_err(), "full");
    assert!(
        Tensor::from_bytes(&[], &[past, 0], DType::F32).is_err(),
        "from_bytes"
    );
    assert!(Tensor::from_i32(&[], &[past, 0]).is_err(), "from_i32");
    assert!(Tensor::from_bool(&[], &[past, 0]).is_err(), "from_bool");
    assert!(empty.reshape(&[past as isize, 0]).is_err(), "reshape");
    assert!(empty.broadcast_to(&[past, 0, 2]).is_err(), "broadcast_to");
}

/// What comes out is what `x` and the weight promote to, each pair on every side: a bf16 `x` with an
/// f32 weight gives f32, unrounded, the same as casting `x` first.
#[test]
fn rms_norm_comes_out_in_what_x_and_the_weight_promote_to() {
    let x = |dtype| tensor(81, &[3, 8], dtype);
    let w = |dtype| tensor(82, &[8], dtype);
    for (a, b, out) in [
        (DType::BF16, DType::BF16, DType::BF16),
        (DType::F32, DType::F32, DType::F32),
        (DType::BF16, DType::F32, DType::F32),
        (DType::F32, DType::BF16, DType::F32),
        (DType::F16, DType::BF16, DType::F32),
        (DType::F32, DType::F64, DType::F64),
    ] {
        let y = x(a).rms_norm(Some(&w(b)), 1e-6).unwrap();
        assert_eq!(y.dtype(), out, "{a:?} with {b:?}");
    }
    assert_eq!(
        x(DType::BF16).rms_norm(None, 1e-6).unwrap().dtype(),
        DType::BF16
    );
    assert_eq!(
        x(DType::F32).rms_norm(None, 1e-6).unwrap().dtype(),
        DType::F32
    );

    let weight = w(DType::F32);
    let mixed = x(DType::BF16).rms_norm(Some(&weight), 1e-6).unwrap();
    let cast = x(DType::BF16).as_dtype(DType::F32).unwrap();
    assert_eq!(
        bits(&mixed),
        bits(&cast.rms_norm(Some(&weight), 1e-6).unwrap())
    );
}
