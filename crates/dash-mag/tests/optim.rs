//! Clipping and AdamW against their definitions, evaluated in f64.

#![cfg(not(target_os = "macos"))]

mod common;

use common::{close, tensor};
use dash_mag::optim::{AdamW, ClipGradNormOptions, NonFinite, NormType, clip_grad_norm};
use dash_mag::{DType, Tensor};

fn t(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(values, shape).unwrap()
}

#[test]
fn clipping_scales_every_gradient_by_one_norm() {
    let mut grads = [t(&[3.0], &[1]), t(&[-4.0], &[1])];
    let options = ClipGradNormOptions {
        max_norm: 1.0,
        ..Default::default()
    };
    assert_eq!(clip_grad_norm(&mut grads, &options).unwrap(), 5.0);
    let coef = 1.0 / (5.0 + 1e-6f32);
    close(&grads[0], &[3.0 * coef], 1e-7);
    close(&grads[1], &[-4.0 * coef], 1e-7);
}

#[test]
fn below_the_limit_nothing_moves() {
    let mut grads = [t(&[3.0, 4.0], &[2])];
    let options = ClipGradNormOptions {
        max_norm: 10.0,
        ..Default::default()
    };
    assert_eq!(clip_grad_norm(&mut grads, &options).unwrap(), 5.0);
    close(&grads[0], &[3.0, 4.0], 0.0);
}

#[test]
fn each_gradient_is_scaled_in_its_own_dtype() {
    let mut grads = [tensor(1, &[8], DType::BF16), tensor(2, &[8], DType::F32)];
    let options = ClipGradNormOptions {
        max_norm: 0.01,
        ..Default::default()
    };
    clip_grad_norm(&mut grads, &options).unwrap();
    assert_eq!(grads[0].dtype(), DType::BF16);
    assert_eq!(grads[1].dtype(), DType::F32);
}

#[test]
fn the_inf_norm_and_a_total_not_finite() {
    let options = ClipGradNormOptions {
        max_norm: 1.0,
        norm_type: NormType::Inf,
        ..Default::default()
    };
    let mut grads = [t(&[3.0, -7.0], &[2])];
    assert_eq!(clip_grad_norm(&mut grads, &options).unwrap(), 7.0);

    let scale = ClipGradNormOptions::default();
    let mut grads = [t(&[1.0, f32::NAN], &[2])];
    assert!(clip_grad_norm(&mut grads, &scale).unwrap().is_nan());
    assert!(grads[0].to_f32().unwrap().iter().all(|v| v.is_nan()));

    let zero = ClipGradNormOptions {
        nonfinite: NonFinite::Zero,
        ..Default::default()
    };
    let mut grads = [t(&[1.0, f32::INFINITY], &[2]), t(&[2.0], &[1])];
    assert!(clip_grad_norm(&mut grads, &zero).unwrap().is_infinite());
    close(&grads[0], &[0.0, 0.0], 0.0);
    close(&grads[1], &[0.0], 0.0);
}

#[test]
fn no_gradients_a_total_of_zero() {
    assert_eq!(
        clip_grad_norm(&mut [], &ClipGradNormOptions::default()).unwrap(),
        0.0
    );
}

/// Three steps of AdamW against torch's formulas, in f64.
#[test]
fn adamw_follows_torch_s_steps() {
    let (b1, b2, eps, wd, lr) = (0.9f64, 0.999f64, 1e-5f64, 0.01f64, 0.5f64);
    let mut opt = AdamW::new((b1, b2), eps, wd);
    let mut p = t(&[1.0, -2.0], &[2]);
    let mut exact = [1.0f64, -2.0];
    let (mut m, mut v) = ([0f64; 2], [0f64; 2]);
    for (step, g) in [[0.5f32, 0.25], [-1.0, 0.75], [2.0, -0.125]]
        .iter()
        .enumerate()
    {
        opt.begin_step();
        p = opt.update("p", &p, &t(g, &[2]), lr).unwrap();
        let k = step as i32 + 1;
        for i in 0..2 {
            let g = g[i] as f64;
            exact[i] *= 1.0 - lr * wd;
            m[i] += (g - m[i]) * (1.0 - b1);
            v[i] = v[i] * b2 + g * g * (1.0 - b2);
            let denominator = v[i].sqrt() / (1.0 - b2.powi(k)).sqrt() + eps;
            exact[i] -= lr / (1.0 - b1.powi(k)) * (m[i] / denominator);
        }
        close(&p, &[exact[0] as f32, exact[1] as f32], 3e-7);
    }
    assert_eq!(opt.step, 3);
    assert_eq!(opt.state().count(), 2);
}
