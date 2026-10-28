//! The chunked gated delta rule: `dash_conformance::gated_delta_chunk`'s tests through dash-mlx's
//! adapter (`tests/common`), on the GPU: transformers' `torch_chunk_gated_delta_rule` on seeded
//! inputs, at f32 and with its f32 casts turned into f64, and the bf16 case against f64 from the
//! same inputs, with dash-ops' bounds. The adapter asserts which kernels ran.
//!
//! What is dash-mlx's own stays here: the scan's kernels against the scan as MLX's operations at
//! the workload's shapes, and each kernel run twice to the same bits.
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test gated_delta_chunk -- --nocapture
#![cfg(target_os = "macos")]

mod common;

use common::{Mlx, require_ieee_f32_matmul};
use dash_conformance::gated_delta_chunk::{self, BF16_TOL};
use dash_mlx::gated_delta::chunk::{ChunkGatedDeltaOptions, chunk_gated_delta_rule};
use mlx_rs::error::Result;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::transforms::fallible_vjp;
use mlx_rs::{Array, Dtype};
use std::sync::{Mutex, MutexGuard};

/// One test at a time: they read the kernel counters, and share the GPU's default stream.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn values(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
    a.as_slice::<f32>().to_vec()
}

/// `x` to three significant figures, rounded up: the form the README and PROGRESS quote.
fn up3(x: f64) -> String {
    if x <= 0.0 {
        return "0".into();
    }
    let mut e = x.log10().floor() as i32;
    let mut m = (x / 10f64.powi(e - 2) * (1.0 - 1e-12)).ceil();
    if m >= 1000.0 {
        m /= 10.0;
        e += 1;
    }
    format!("{:.2}e{e}", m / 100.0)
}

#[test]
fn fp32_matches_torch() {
    let _serial = serial();
    require_ieee_f32_matmul();
    gated_delta_chunk::check_f32_cases(&Mlx, "mlx").all_ran();
}

#[test]
fn bf16_matches_torch_at_f64() {
    let _serial = serial();
    gated_delta_chunk::check_bf16_case(&Mlx, "mlx-bf16", BF16_TOL).all_ran();
}

/// Seeded inputs at the workload's shapes, the model's distributions: `g = -softplus(x) 0.4`,
/// `beta = sigmoid(x)`.
fn workload(t: i32, h: i32, d: i32, dtype: Dtype) -> Vec<Array> {
    let normal = |shape: &[i32], scale: f32, seed: u64| {
        let key = mlx_rs::random::key(seed).unwrap();
        mlx_rs::random::normal::<f32>(shape, None, None, &key)
            .unwrap()
            .multiply(Array::from_f32(scale))
            .unwrap()
    };
    let g = mlx_rs::nn::softplus(normal(&[1, t, h], 1.0, 3))
        .unwrap()
        .multiply(Array::from_f32(-0.4))
        .unwrap();
    let beta = mlx_rs::ops::sigmoid(normal(&[1, t, h], 1.0, 4)).unwrap();
    vec![
        normal(&[1, t, h, d], 1.0, 0).as_dtype(dtype).unwrap(),
        normal(&[1, t, h, d], 1.0, 1).as_dtype(dtype).unwrap(),
        normal(&[1, t, h, d], 1.0, 2).as_dtype(dtype).unwrap(),
        g.as_dtype(dtype).unwrap(),
        beta.as_dtype(dtype).unwrap(),
        normal(&[1, h, d, d], 0.1, 5),
        normal(&[1, t, h, d], 1.0, 6).as_dtype(dtype).unwrap(),
        normal(&[1, h, d, d], 1.0, 7),
    ]
}

/// The scan kernels against the scan as MLX operations, at the workload's shapes (6144 tokens,
/// 16 heads of 128) and at a length that leaves the last chunk ragged: the output, the final
/// state and every gradient, in f32.
#[test]
fn kernels_match_the_reference_at_the_workload_shapes() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    use dash_mlx::gated_delta::chunk::{chunk_gated_delta_rule_reference, counters};
    use std::sync::atomic::Ordering;
    for t in [6144, 1000] {
        let inputs = workload(t, 16, 128, Dtype::Float32);
        let (primals, cotangents) = (&inputs[..6], &inputs[6..]);
        let run = |kernels: bool| -> Result<Vec<Vec<f32>>> {
            let rule = |a: &[Array]| -> Result<Vec<Array>> {
                let f = if kernels {
                    chunk_gated_delta_rule
                } else {
                    chunk_gated_delta_rule_reference
                };
                let options = ChunkGatedDeltaOptions::default()
                    .with_qk_l2norm(true)
                    .with_final_state(true);
                let (out, state) = f(
                    &a[0],
                    &a[1],
                    &a[2],
                    &a[3],
                    &a[4],
                    Some(&a[5]),
                    None,
                    options,
                )?;
                Ok(vec![out, state.expect("asked for")])
            };
            let (outputs, grads) = fallible_vjp(rule, primals, cotangents)?;
            Ok(outputs.iter().chain(&grads).map(values).collect())
        };
        let before = (
            counters::SCAN_FORWARD_KERNEL.load(Ordering::SeqCst),
            counters::SCAN_BACKWARD_KERNEL.load(Ordering::SeqCst),
        );
        let kernels = run(true)?;
        assert!(counters::SCAN_FORWARD_KERNEL.load(Ordering::SeqCst) > before.0);
        assert!(counters::SCAN_BACKWARD_KERNEL.load(Ordering::SeqCst) > before.1);
        let reference = run(false)?;
        let names = ["out", "state", "dq", "dk", "dv", "dg", "dbeta", "dh0"];
        let mut failed = Vec::new();
        for ((name, a), e) in names.iter().zip(&kernels).zip(&reference) {
            let (abs, rel) = dash_conformance::errors_f32(a, e);
            println!(
                "T {t} kernels against the operations {name:6} max abs {abs:.3e} rel {}",
                up3(rel as f64)
            );
            if rel > 2e-6 {
                failed.push(format!("{name} {rel:.3e}"));
            }
        }
        assert!(failed.is_empty(), "T {t}: {failed:?}");
    }
    Ok(())
}

/// By default a run repeats bit for bit: the chunked rule on its kernels (the inverses, the scan
/// forward and backward) and the step's kernel, each run twice on the same inputs, give the same
/// bits, in f32 and bf16, over one sequence and over packed sequences, every output and gradient.
#[test]
fn kernels_repeat_bit_for_bit() -> Result<()> {
    let _serial = serial();
    use dash_mlx::gated_delta::chunk::counters;
    use dash_mlx::gated_delta::recurrent::{GatedDeltaStepOptions, gated_delta_step};
    use mlx_rs::ops::broadcast_to;
    use std::sync::atomic::Ordering;
    let bits = |arrays: &[Array]| -> Vec<Vec<u32>> {
        arrays
            .iter()
            .map(|a| values(a).iter().map(|v| v.to_bits()).collect())
            .collect()
    };
    let [t, h, d] = [1000, 16, 128];
    let boundaries = Array::from_slice(&[0i32, 300, 301, 301, t], &[5]);
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let inputs = workload(t, h, d, dtype);
        for cu_seqlens in [None, Some(&boundaries)] {
            let label = format!("{dtype:?}, packed {}", cu_seqlens.is_some());
            let (mut primals, mut cotangents) = (inputs[..6].to_vec(), inputs[6..].to_vec());
            if cu_seqlens.is_some() {
                // One initial state, and one final state's gradient, per packed sequence.
                primals[5] = broadcast_to(&primals[5], &[4, h, d, d])?.contiguous()?;
                cotangents[1] = broadcast_to(&cotangents[1], &[4, h, d, d])?.contiguous()?;
            }
            let run = || {
                let rule = |a: &[Array]| -> Result<Vec<Array>> {
                    let options = ChunkGatedDeltaOptions::default()
                        .with_qk_l2norm(true)
                        .with_final_state(true);
                    let (out, state) = chunk_gated_delta_rule(
                        &a[0],
                        &a[1],
                        &a[2],
                        &a[3],
                        &a[4],
                        Some(&a[5]),
                        cu_seqlens,
                        options,
                    )?;
                    Ok(vec![out, state.expect("asked for")])
                };
                let (outputs, grads) = fallible_vjp(rule, &primals, &cotangents)?;
                Ok::<_, mlx_rs::error::Exception>(bits(&[outputs, grads].concat()))
            };
            let before = [
                &counters::INVERSE_KERNEL,
                &counters::SCAN_FORWARD_KERNEL,
                &counters::SCAN_BACKWARD_KERNEL,
            ]
            .map(|c| c.load(Ordering::SeqCst));
            let first = run()?;
            let after = [
                &counters::INVERSE_KERNEL,
                &counters::SCAN_FORWARD_KERNEL,
                &counters::SCAN_BACKWARD_KERNEL,
            ]
            .map(|c| c.load(Ordering::SeqCst));
            assert!(
                before.iter().zip(&after).all(|(b, a)| a > b),
                "{label}: not every kernel ran"
            );
            assert!(
                first == run()?,
                "{label}: two runs of the chunked rule differ"
            );
        }
        // The step: one token of each of 32 sequences, and the 1000 tokens packed as 4
        // sequences, each from its own state.
        let rows = |x: &Array| -> Result<Array> {
            let mut shape = x.shape()[1..].to_vec();
            shape[0] = t;
            x.reshape(&shape)
        };
        let [q, k, v, g, beta] = [0, 1, 2, 3, 4].map(|i| rows(&inputs[i]).unwrap());
        let options = GatedDeltaStepOptions::default().with_qk_l2norm(true);
        for (label, rows, cu_seqlens, states) in
            [("batch", 32, None, 32), ("packed", t, Some(&boundaries), 4)]
        {
            let take = |x: &Array| -> Result<Array> { x.try_index(..rows) };
            let state = broadcast_to(&inputs[5], &[states, h, d, d])?.contiguous()?;
            let (q, k, v, g, beta) = (take(&q)?, take(&k)?, take(&v)?, take(&g)?, take(&beta)?);
            let step = || {
                let (out, next) =
                    gated_delta_step(&q, &k, &v, &g, &beta, &state, cu_seqlens, options)?;
                Ok::<_, mlx_rs::error::Exception>(bits(&[out, next]))
            };
            assert!(
                step()? == step()?,
                "{dtype:?} {label} step: two runs differ"
            );
        }
    }
    Ok(())
}
