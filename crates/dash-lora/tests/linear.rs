//! One LoRA'd linear layer against PEFT's `lora.Linear` (PEFT 0.20, torch 2.13, CPU): the
//! output and the gradients of the input, of A and of B for a given upstream gradient, with a
//! bf16 base (x and W bf16, A and B f32: PEFT's default for a bf16 model) and with an f32 base.
//! `tests/fixtures/linear.safetensors`: x [48, 96], W [80, 96], r 16, alpha 32.
//!
//!   cargo test -p dash-lora --test linear -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test -p dash-lora --features cuda --test linear -- --nocapture

mod common;

use burn::optim::GradientsParams;
use burn::tensor::{DType, Device, Tensor};
use common::*;
use dash_lora::{Lora, LoraLinear};

/// Error bounds by device and by the dtype of the tensor compared, relative to the tensor's
/// largest value (the gradients of A reach 94, so an absolute bound would say nothing).
///
/// Measured on Flex, and on an RTX 5070 Ti with Burn's default kernels and with `burn/fusion`:
/// bit-identical to PEFT, output and all three gradients, at both precisions (at these sizes
/// every matrix product, PyTorch's CPU kernels included, accumulates each output element over
/// K in order with fused multiply-adds). With `burn/autotune` (with or without fusion; sixteen
/// runs each, half with the tests in parallel), CUDA may run a kernel that sums in another
/// order: f32 results then move by up to 3.6e-7, and one element of a bf16 output (y, or dx)
/// rounds to its neighbouring bf16 value (1.3e-3). So f32 tensors are held to a few f32
/// roundings on both devices, bf16 tensors to the same on Flex and to one bf16 unit, 2^-7 of the
/// largest value, on CUDA.
fn bound_for(device_name: &str, dtype: DType) -> (f64, f64) {
    match (device_name, dtype) {
        ("flex", _) | (_, DType::F32) => (f64::INFINITY, 1e-6),
        _ => (f64::INFINITY, 2f64.powi(-7)),
    }
}

fn run(device_name: &str, device: &Device, tag: &str) {
    let f = load("linear.safetensors");
    let key = |k: &str| get(&f, &format!("{tag}.{k}"));
    let layer = LoraLinear::frozen_from_data(key("w"), device).with_lora(Lora::from_data(
        key("a"),
        key("b"),
        scalar(key("scale")),
        device,
    ));
    let x: Tensor<2> = tensor::<2>(key("x"), device).require_grad();
    let y = layer.forward(x.clone());
    assert_eq!(y.dtype(), x.dtype(), "the output has the input's dtype");

    let gy: Tensor<2> = tensor(key("gy"), device);
    let grads = y.clone().mul(gy).sum().backward();

    let lora = layer.lora.as_ref().unwrap();
    // Gradients reach A and B only: the base weight requires none and gets none.
    assert!(!layer.weight.is_require_grad());
    assert!(layer.weight.val().grad(&grads).is_none());
    let da = lora.a.val().grad(&grads).unwrap();
    let db = lora.b.val().grad(&grads).unwrap();
    let dx = x.grad(&grads).unwrap();
    assert_eq!((da.dtype(), db.dtype()), (DType::F32, DType::F32));
    assert_eq!(dx.dtype(), x.dtype());

    let p = format!("{device_name} {tag}");
    assert!(y.device() == *device, "the layer ran on {device_name}");
    for (what, actual) in [("y", y), ("dx", dx), ("dA", da), ("dB", db)] {
        let (abs, rel) = bound_for(device_name, actual.dtype());
        let expected = key(&what.to_lowercase());
        check(
            &format!("{p} {what}"),
            &actual.into_data(),
            &expected,
            abs,
            rel,
        );
    }

    // What the optimizer is handed: the two factors, nothing else.
    let params = GradientsParams::from_grads(grads, &layer);
    assert_eq!(params.len(), 2);
    assert!(params.get::<2>(lora.a.id).is_some() && params.get::<2>(lora.b.id).is_some());
    assert!(params.get::<2>(layer.weight.id).is_none());
}

#[test]
fn bf16_base_matches_peft() {
    for (name, device) in devices_with(DType::BF16) {
        run(name, &device, "bf16");
    }
}

#[test]
fn f32_base_matches_peft() {
    for (name, device) in devices() {
        run(name, &device, "f32");
    }
}

/// The adapter reaches the output at f32 precision; merged into a bf16 weight, most of it
/// rounds away. W ~ U(-1/√in, 1/√in), and an update with a thousandth of W's spread, a
/// fraction of bf16's spacing at W's typical magnitude, as a fine-tuning update is: the merged
/// weight keeps few of the update's elements and points elsewhere, while the unmerged layer's
/// output is the base's plus the whole low-rank term.
#[test]
fn the_update_is_kept_apart_not_rounded_into_the_weight() {
    let device = Device::flex();
    let mut rng = Rng::new(3);
    let (out, inp, r) = (256, 256, 16);
    let w = rng.tensor([out, inp], DType::F32, &device, |g| {
        g.uniform(-1.0, 1.0) / (inp as f64).sqrt()
    });
    let a = rng.tensor([r, inp], DType::F32, &device, Rng::normal);
    let b = rng.tensor([out, r], DType::F32, &device, Rng::normal);
    // W's elements spread 1/√(3 in); B A's spread √r; scale · B A spreads a thousandth of W.
    let scale = 1e-3 / ((3.0 * inp as f64).sqrt() * (r as f64).sqrt());
    let w_bf16 = w.cast(DType::BF16);
    let update = b.clone().matmul(a.clone()).mul_scalar(scale);

    // Merged: what survives in bf16(W + update) - W.
    let merged = w_bf16
        .clone()
        .cast(DType::F32)
        .add(update.clone())
        .cast(DType::BF16)
        .cast(DType::F32)
        .sub(w_bf16.clone().cast(DType::F32));
    let kept = update
        .clone()
        .mul(merged.clone())
        .sum()
        .into_scalar::<f32>() as f64;
    let cosine = kept
        / (update.clone().powi_scalar(2).sum().into_scalar::<f32>() as f64).sqrt()
        / (merged.clone().powi_scalar(2).sum().into_scalar::<f32>() as f64).sqrt();
    let zeros = merged
        .clone()
        .equal_elem(0.0)
        .int()
        .sum()
        .into_scalar::<i64>() as f64
        / (out * inp) as f64;
    println!(
        "merged into bf16: cosine with the update {cosine:.3}, {:.0}% of it rounded to 0",
        zeros * 100.0
    );
    assert!(zeros > 0.5 && cosine < 0.8);

    // Unmerged: the layer adds the whole low-rank term to the base product, in f32. With f32
    // input the output is f32 and the base product is computed as the frozen layer computes
    // it, so the difference from the frozen layer is the low-rank term up to the rounding of
    // one f32 addition at the output's magnitude: half a unit in the last place, at most
    // 2^-24 of the largest output, a whole one allowed here.
    let frozen = LoraLinear::frozen(w_bf16.clone());
    let layer = LoraLinear::frozen(w_bf16).with_lora(Lora::new(a, b, scale));
    let x = rng.tensor([8, inp], DType::F32, &device, Rng::normal);
    let y = layer.forward(x.clone());
    let largest = y.clone().abs().max().into_scalar::<f32>() as f64;
    let added = y.sub(frozen.forward(x.clone()));
    let e = err(
        &added.into_data(),
        &x.matmul(update.transpose()).into_data(),
    );
    println!(
        "unmerged: the low-rank term reaches the output within {:.2e}, {:.2} ulp of the largest output",
        e.abs,
        e.abs / (largest * f32::EPSILON as f64)
    );
    assert!(
        e.abs <= largest * f32::EPSILON as f64,
        "unmerged output off by {e:?}"
    );
}

/// A batch `[b, t, in]` gives, sequence by sequence, what `[t, in]` gives, on a plain
/// (inference) device too: the layer runs unmerged without autodiff.
#[test]
fn batched_input_is_each_sequence_s_output() {
    for device in [Device::flex(), Device::flex().autodiff()] {
        let f = load("linear.safetensors");
        let layer = LoraLinear::frozen_from_data(get(&f, "bf16.w"), &device).with_lora(
            Lora::from_data(get(&f, "bf16.a"), get(&f, "bf16.b"), 2.0, &device),
        );
        let x: Tensor<3> = Rng::new(5).tensor([2, 12, 96], DType::BF16, &device, Rng::normal);
        let y = layer.forward(x.clone());
        assert_eq!(y.dims(), [2, 12, 80]);
        for i in 0..2 {
            let one: Tensor<2> = x.clone().slice_dim(0, i..i + 1).squeeze_dim(0);
            let expected = layer.forward(one).into_data();
            let got: Tensor<2> = y.clone().slice_dim(0, i..i + 1).squeeze_dim(0);
            check(
                &format!("batch item {i}"),
                &got.into_data(),
                &expected,
                0.0,
                0.0,
            );
        }
    }
}
