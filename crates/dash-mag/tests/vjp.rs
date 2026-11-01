//! The vector-Jacobian product: against gradients worked out by hand and against central
//! differences, what a cotangent means, what a constant is, and that a stack run layer by layer
//! gives what it gives whole.

#![cfg(not(target_os = "macos"))]

mod common;

use common::{bits, close, relative, tensor, values};
use dash_mag::{DType, Tensor, vjp};

fn t(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(values, shape).unwrap()
}

#[test]
fn a_loss_gives_its_gradient() {
    // d/dx sum(x^2 * c) = 2 x c.
    let x = t(&[1.0, -2.0, 3.0], &[3]);
    let c = t(&[1.0, 0.5, 2.0], &[3]);
    let one = Tensor::scalar(1.0, DType::F32).unwrap();
    let (outs, grads) = vjp(
        |p| Ok(vec![p[0].square()?.mul(&c)?.sum()?]),
        std::slice::from_ref(&x),
        &[Some(one)],
    )
    .unwrap();
    assert_eq!(outs[0].item().unwrap(), 1.0 + 2.0 + 18.0);
    close(&grads[0], &[2.0, -2.0, 12.0], 0.0);
}

#[test]
fn a_cotangent_weighs_its_output() {
    // y = 3 x: the gradient of sum(y * w) is 3 w.
    let x = t(&[1.0, 2.0], &[2]);
    let w = t(&[5.0, -1.0], &[2]);
    let (outs, grads) = vjp(|p| Ok(vec![p[0].mul_scalar(3.0)?]), &[x], &[Some(w)]).unwrap();
    close(&outs[0], &[3.0, 6.0], 0.0);
    close(&grads[0], &[15.0, -3.0], 0.0);
}

#[test]
fn outputs_sum_and_one_without_a_cotangent_takes_no_part() {
    let x = t(&[2.0], &[1]);
    let one = t(&[1.0], &[1]);
    let (outs, grads) = vjp(
        |p| Ok(vec![p[0].square()?, p[0].mul_scalar(10.0)?, p[0].exp()?]),
        &[x],
        &[Some(one.clone()), Some(one), None],
    )
    .unwrap();
    assert_eq!(outs.len(), 3);
    close(&grads[0], &[4.0 + 10.0], 0.0);
    close(&outs[2], &[2f32.exp()], 1e-6);
}

#[test]
fn a_primal_the_outputs_do_not_reach_gets_zeros_in_its_dtype() {
    let x = t(&[2.0], &[1]);
    let unused = tensor(3, &[2, 2], DType::BF16);
    let one = t(&[1.0], &[1]);
    let (_, grads) = vjp(|p| Ok(vec![p[0].square()?]), &[x, unused], &[Some(one)]).unwrap();
    close(&grads[0], &[4.0], 0.0);
    assert_eq!(grads[1].dtype(), DType::BF16);
    assert_eq!(grads[1].shape(), [2, 2]);
    close(&grads[1], &[0.0; 4], 0.0);
}

#[test]
fn outputs_that_reach_no_primal_give_zeros() {
    let x = t(&[2.0], &[1]);
    let constant = t(&[7.0], &[1]);
    let (outs, grads) = vjp(
        |_| Ok(vec![constant.clone()]),
        &[x],
        &[Some(t(&[1.0], &[1]))],
    )
    .unwrap();
    close(&outs[0], &[7.0], 0.0);
    close(&grads[0], &[0.0], 0.0);
}

#[test]
fn a_captured_tensor_is_a_constant_and_stop_gradient_cuts() {
    let x = t(&[2.0, 3.0], &[2]);
    let c = t(&[4.0, 5.0], &[2]);
    let one = Tensor::scalar(1.0, DType::F32).unwrap();
    let (_, grads) = vjp(
        |p| Ok(vec![p[0].mul(&c)?.sum()?]),
        std::slice::from_ref(&x),
        &[Some(one.clone())],
    )
    .unwrap();
    close(&grads[0], &[4.0, 5.0], 0.0);
    let (_, grads) = vjp(
        |p| Ok(vec![p[0].mul(&p[0].stop_gradient()?)?.sum()?]),
        &[x],
        &[Some(one)],
    )
    .unwrap();
    close(&grads[0], &[2.0, 3.0], 0.0);
}

#[test]
fn it_refuses_what_it_cannot_do() {
    let x = t(&[2.0, 3.0], &[2]);
    let one = t(&[1.0], &[1]);
    assert!(
        vjp(
            |p| Ok(vec![p[0].clone()]),
            std::slice::from_ref(&x),
            &[Some(one.clone())]
        )
        .is_err()
    );
    assert!(vjp(|p| Ok(vec![p[0].clone()]), std::slice::from_ref(&x), &[]).is_err());
    let i = Tensor::from_i32(&[1], &[1]).unwrap();
    assert!(vjp(|p| Ok(vec![p[0].clone()]), &[i], &[Some(one)]).is_err());
}

#[test]
fn outputs_come_back_constant_and_primals_stay_as_they_were() {
    let x = t(&[2.0, 3.0], &[2]);
    let before = bits(&x);
    let one = t(&[1.0, 1.0], &[2]);
    let (outs, _) = vjp(
        |p| Ok(vec![p[0].square()?]),
        std::slice::from_ref(&x),
        &[Some(one.clone())],
    )
    .unwrap();
    // The output feeds a second product as a constant: only `y` is differentiated here.
    let (_, grads) = vjp(
        |p| Ok(vec![p[0].mul(&outs[0])?]),
        std::slice::from_ref(&x),
        &[Some(one)],
    )
    .unwrap();
    close(&grads[0], &[4.0, 9.0], 0.0);
    assert_eq!(bits(&x), before);
}

/// A small stack in f32, every operation of the array layer on the path.
fn layer(x: &Tensor, w: &Tensor) -> dash_mag::Result<Tensor> {
    let y = x.matmul(w)?;
    let y = y.silu()?.add(&y.mul_scalar(0.5)?)?;
    let parts = y.split_at(&[2], -1)?;
    Tensor::concat(&[&parts[1], &parts[0]], -1)
}

fn loss(x: &Tensor, w: &[Tensor]) -> dash_mag::Result<Tensor> {
    let mut y = x.clone();
    for w in w {
        y = layer(&y, w)?;
    }
    y.square()?.mean()
}

#[test]
fn it_agrees_with_central_differences() {
    let x = tensor(1, &[3, 4], DType::F64);
    let w: Vec<Tensor> = (0..3)
        .map(|i| tensor(10 + i, &[4, 4], DType::F64))
        .collect();
    let one = Tensor::scalar(1.0, DType::F64).unwrap();
    let primals: Vec<Tensor> = w.iter().cloned().chain([x.clone()]).collect();
    let (_, grads) = vjp(|p| Ok(vec![loss(&p[3], &p[..3])?]), &primals, &[Some(one)]).unwrap();
    // Along a random direction of every primal at once.
    let directions: Vec<Tensor> = primals
        .iter()
        .enumerate()
        .map(|(i, p)| tensor(50 + i as u64, &p.shape(), DType::F64))
        .collect();
    let eps = 1e-4;
    let moved = |sign: f64| -> Vec<Tensor> {
        primals
            .iter()
            .zip(&directions)
            .map(|(p, d)| p.add(&d.mul_scalar(sign * eps).unwrap()).unwrap())
            .collect()
    };
    let value = |p: &[Tensor]| loss(&p[3], &p[..3]).unwrap().as_dtype(DType::F64).unwrap();
    let f64s = |t: &Tensor| -> Vec<f64> {
        t.to_bytes()
            .unwrap()
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let numeric = (f64s(&value(&moved(1.0)))[0] - f64s(&value(&moved(-1.0)))[0]) / (2.0 * eps);
    let analytic: f64 = grads
        .iter()
        .zip(&directions)
        .map(|(g, d)| f64s(g).iter().zip(f64s(d)).map(|(a, b)| a * b).sum::<f64>())
        .sum();
    let error = (numeric - analytic).abs() / numeric.abs();
    println!("analytic {analytic} numeric {numeric} relative {error:.3e}");
    assert!(error < 1e-5, "{analytic} against {numeric}: {error:e}");
}

/// A stack run layer by layer, each layer's product taking the gradient the one above gave its
/// input: dash-ops' and dash-mlx's recompute, as `vjp` expresses it.
#[test]
fn a_stack_layer_by_layer_gives_what_it_gives_whole() {
    let x = tensor(1, &[3, 4], DType::F32);
    let w: Vec<Tensor> = (0..4)
        .map(|i| tensor(10 + i, &[4, 4], DType::F32))
        .collect();
    let one = Tensor::scalar(1.0, DType::F32).unwrap();
    let (_, whole) = vjp(|p| Ok(vec![loss(&x, p)?]), &w, &[Some(one.clone())]).unwrap();

    let mut inputs = vec![x.clone()];
    for w in &w {
        inputs.push(layer(inputs.last().unwrap(), w).unwrap());
    }
    let top = inputs.pop().unwrap();
    let (_, top_grad) = vjp(
        |p| Ok(vec![p[0].square()?.mean()?]),
        std::slice::from_ref(&top),
        &[Some(one)],
    )
    .unwrap();
    let mut cotangent = top_grad[0].clone();
    let mut by_layer = vec![None; w.len()];
    for l in (0..w.len()).rev() {
        let (_, grads) = vjp(
            |p| Ok(vec![layer(&p[0], &p[1])?]),
            &[inputs[l].clone(), w[l].clone()],
            &[Some(cotangent.clone())],
        )
        .unwrap();
        cotangent = grads[0].clone();
        by_layer[l] = Some(grads[1].clone());
    }
    for (whole, by_layer) in whole.iter().zip(&by_layer) {
        let got = by_layer.as_ref().unwrap().to_f32().unwrap();
        let error = relative(&got, &whole.to_f32().unwrap());
        println!("layer by layer against whole: {error:.3e}");
        assert!(error < 2e-5, "{error:e}");
    }
}

/// RMSNorm reads and rounds through f32, so its differences are taken in f32, with a wide step.
#[test]
fn rms_norm_agrees_with_central_differences() {
    let x = tensor(3, &[2, 8], DType::F32).add_scalar(0.25).unwrap();
    let w = tensor(4, &[8], DType::F32);
    let d = tensor(5, &[2, 8], DType::F32);
    let f = |x: &Tensor| x.rms_norm(Some(&w), 1e-6)?.square()?.mul(&w)?.sum();
    let one = Tensor::scalar(1.0, DType::F32).unwrap();
    let (_, grads) = vjp(
        |p| Ok(vec![f(&p[0])?]),
        std::slice::from_ref(&x),
        &[Some(one)],
    )
    .unwrap();
    let eps = 1e-2;
    let up = f(&x.add(&d.mul_scalar(eps).unwrap()).unwrap())
        .unwrap()
        .item()
        .unwrap() as f64;
    let down = f(&x.sub(&d.mul_scalar(eps).unwrap()).unwrap())
        .unwrap()
        .item()
        .unwrap() as f64;
    let numeric = (up - down) / (2.0 * eps);
    let analytic = grads[0].mul(&d).unwrap().sum().unwrap().item().unwrap() as f64;
    let error = (numeric - analytic).abs() / numeric.abs();
    println!("rms_norm: analytic {analytic} numeric {numeric} relative {error:.3e}");
    assert!(error < 1e-2, "{analytic} against {numeric}: {error:e}");
}

#[test]
fn values_are_exact_in_bf16() {
    // What the shared values promise, on which every comparison across sides rests.
    let v = values(7, 64);
    let back = tensor(7, &[64], DType::BF16).to_f32().unwrap();
    assert_eq!(v, back);
}
