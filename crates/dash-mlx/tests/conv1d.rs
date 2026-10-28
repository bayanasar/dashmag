//! The causal depthwise conv1d: `dash_conformance`'s `conv1d` and `conv1d_state` tests through
//! dash-mlx's adapter (`tests/common`), on the GPU: every kernel width with and without a bias
//! and SiLU, decoding steps from a recorded state, a prefill followed by single steps; batches and
//! packed sequences from given states with final states, a sequence split in two, packed
//! sequences against one call each, empty sequences, either output alone in the loss. The adapter
//! asserts that the kernels ran exactly where `kernels_take` says they take a call.
//!
//! The bounds are dash-ops' for the CubeCL runtimes other than CUDA (`other`): the f32 output and
//! step and the bias gradient err a little more there, the exponential and the order of the sums
//! being their own, as on Metal.
//!
//! What is dash-mlx's own stays here: the kernels against the reference at a size the fixtures do
//! not reach, and each kernel run twice to the same bits.
//!
//!   cargo test -p dash-mlx --test conv1d -- --nocapture
#![cfg(target_os = "macos")]

mod common;

use common::Mlx;
use dash_conformance::conv1d_state::{assert_packed, assert_splits};
use dash_conformance::{conv1d, conv1d_state, errors};
use dash_mlx::conv1d::{
    Activation, CausalConv1dOptions, CausalConv1dStepOptions, causal_conv1d, causal_conv1d_update,
    kernels_take, probe, reference,
};
use mlx_rs::error::Result;
use mlx_rs::transforms::fallible_vjp;
use mlx_rs::{Array, Dtype};
use std::sync::{Mutex, MutexGuard};

/// The bounds of the runtimes with an exponential and orders of summation of their own.
const OTHER: bool = true;

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
fn sequences_match_torch() {
    let _serial = serial();
    conv1d::sequences(&Mlx, "mlx", OTHER).all_ran();
}

#[test]
fn steps_match_torch() {
    let _serial = serial();
    conv1d::steps(&[&Mlx], "mlx", OTHER).all_ran();
}

#[test]
fn prefill_then_steps_equals_the_sequence() {
    let _serial = serial();
    let against_sequence = conv1d::prefill_then_steps(&Mlx, "mlx", OTHER).all_ran();
    println!(
        "mlx steps against the sequence: max abs {}",
        up3(against_sequence)
    );
}

#[test]
fn states_and_packed_sequences_match_torch() {
    let _serial = serial();
    for (part, rel) in conv1d_state::fixtures(&Mlx, "mlx").all_ran() {
        println!("mlx states {part}: largest {}", up3(rel));
    }
}

#[test]
fn a_split_sequence_equals_the_whole() {
    let _serial = serial();
    assert_splits(&conv1d_state::splits(&Mlx, "mlx").all_ran(), OTHER);
}

#[test]
fn packed_sequences_equal_one_call_each() {
    let _serial = serial();
    assert_packed(&conv1d_state::packed_against_calls(&Mlx, "mlx").all_ran());
}

#[test]
fn empty_sequences_pass_their_states_through() {
    let _serial = serial();
    conv1d_state::empty_sequences(&Mlx).all_ran();
}

#[test]
fn either_output_alone_has_its_own_gradients() {
    let _serial = serial();
    conv1d_state::outputs_alone(&Mlx).all_ran();
}

/// `causal_conv1d`'s signature, for the kernels and the reference alike.
type Conv = fn(
    &Array,
    &Array,
    Option<&Array>,
    Option<&Array>,
    Option<&Array>,
    CausalConv1dOptions,
) -> Result<(Array, Option<Array>)>;

/// A deterministic uniform sample in `[-scale, scale)`.
fn uniform(n: usize, scale: f32, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// The kernels against the reference at a size the fixtures do not reach: many runs per
/// sequence, a ragged last run, hundreds of threadgroups, two batches, a state and a bias, every
/// width the kernels take.
#[test]
fn kernels_match_the_reference_at_scale() -> Result<()> {
    let _serial = serial();
    let [b, t, c, w] = [2, 1000, 1544, 8];
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let x = Array::from_slice(&uniform((b * t * c) as usize, 3.0, &mut seed), &[b, t, c]);
    let state = Array::from_slice(&uniform((b * w * c) as usize, 3.0, &mut seed), &[b, w, c]);
    let bias = Array::from_slice(&uniform(c as usize, 0.5, &mut seed), &[c]);
    let grad = Array::from_slice(&uniform((b * t * c) as usize, 1.0, &mut seed), &[b, t, c]);
    let grad_final = Array::from_slice(&uniform((b * w * c) as usize, 1.0, &mut seed), &[b, w, c]);
    let names = [
        "out",
        "final_state",
        "grad_x",
        "grad_weight",
        "grad_bias",
        "grad_state",
    ];
    let bounds = [3e-7, 0.0, 3e-7, 2e-6, 2e-6, 3e-7];
    let mut largest = [0.0f64; 6];
    for k in 2..=8 {
        let weight = Array::from_slice(&uniform((c * k) as usize, 0.5, &mut seed), &[c, k]);
        let options = CausalConv1dOptions::default()
            .with_activation(Activation::Silu)
            .with_final_state(true);
        let primals = [x.clone(), weight, bias.clone(), state.clone()];
        let run = |f: Conv| {
            let conv = |a: &[Array]| -> Result<Vec<Array>> {
                let (out, last) = f(&a[0], &a[1], Some(&a[2]), Some(&a[3]), None, options)?;
                Ok(vec![out, last.expect("asked for")])
            };
            let (outputs, grads) =
                fallible_vjp(conv, &primals, &[grad.clone(), grad_final.clone()])?;
            Ok::<_, mlx_rs::error::Exception>(
                outputs.iter().chain(&grads).map(values).collect::<Vec<_>>(),
            )
        };
        let kernels = run(causal_conv1d)?;
        let reference = run(reference::causal_conv1d)?;
        for (i, ((name, bound), (a, e))) in names
            .iter()
            .zip(bounds)
            .zip(kernels.iter().zip(&reference))
            .enumerate()
        {
            let (abs, rel) = errors(a, e);
            assert!(
                rel <= bound,
                "K {k} {name}: {rel:.3e} (max abs {abs:.3e}) exceeds {bound:.1e}"
            );
            largest[i] = largest[i].max(rel);
        }
    }
    for (name, rel) in names.iter().zip(largest) {
        println!(
            "kernels against the reference, widths 2 to 8, {name}: largest {}",
            up3(rel)
        );
    }
    Ok(())
}

/// The bits of each array, read as f32 (exact for f32, bf16 and f16).
fn bits(arrays: &[Array]) -> Vec<Vec<u32>> {
    arrays
        .iter()
        .map(|a| values(a).iter().map(|v| v.to_bits()).collect())
        .collect()
}

/// By default a run repeats bit for bit: each kernel run twice on the same inputs gives the same
/// bits, in f32 and bf16: the forward and backward passes from initial states to final states,
/// over a batch and over packed sequences (the weight's and bias's gradients summed across the
/// sequences), and the decoding step.
#[test]
fn kernels_repeat_bit_for_bit() -> Result<()> {
    let _serial = serial();
    let [b, t, c, k] = [2, 1000, 1544, 4];
    let w = k - 1;
    assert!(kernels_take(k));
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let mut sample = |shape: &[i32], scale: f32, dtype: Dtype| {
            let n = shape.iter().product::<i32>() as usize;
            Array::from_slice(&uniform(n, scale, &mut seed), shape).as_dtype(dtype)
        };
        let weight = sample(&[c, k], 0.5, dtype)?;
        let bias = sample(&[c], 0.5, dtype)?;
        let options = CausalConv1dOptions::default()
            .with_activation(Activation::Silu)
            .with_final_state(true);
        let packed = Array::from_slice(&[0i32, 300, 301, 301, t], &[5]);
        for (label, x, state, cu_seqlens) in [
            (
                "batch",
                sample(&[b, t, c], 3.0, dtype)?,
                sample(&[b, w, c], 3.0, Dtype::Float32)?,
                None,
            ),
            (
                "packed",
                sample(&[1, t, c], 3.0, dtype)?,
                sample(&[4, w, c], 3.0, Dtype::Float32)?,
                Some(&packed),
            ),
        ] {
            let grad = sample(x.shape(), 1.0, Dtype::Float32)?;
            let grad_final = sample(state.shape(), 1.0, Dtype::Float32)?;
            let primals = [x, weight.clone(), bias.clone(), state];
            let run = || {
                let conv = |a: &[Array]| -> Result<Vec<Array>> {
                    let (out, last) =
                        causal_conv1d(&a[0], &a[1], Some(&a[2]), Some(&a[3]), cu_seqlens, options)?;
                    Ok(vec![out, last.expect("asked for")])
                };
                let (outputs, grads) =
                    fallible_vjp(conv, &primals, &[grad.clone(), grad_final.clone()])?;
                Ok::<_, mlx_rs::error::Exception>(bits(&[outputs, grads].concat()))
            };
            let before = probe::forward_launches();
            let first = run()?;
            assert!(
                probe::forward_launches() > before,
                "{label}: the kernels did not run"
            );
            assert!(first == run()?, "{dtype:?} {label}: two runs differ");
        }
        let x = sample(&[32, c], 3.0, dtype)?;
        let state = sample(&[32, w, c], 3.0, Dtype::Float32)?;
        let step_options = CausalConv1dStepOptions::default().with_activation(Activation::Silu);
        let step = || {
            let (out, next) = causal_conv1d_update(&x, &weight, Some(&bias), &state, step_options)?;
            Ok::<_, mlx_rs::error::Exception>(bits(&[out, next]))
        };
        assert!(step()? == step()?, "{dtype:?} step: two runs differ");
    }
    Ok(())
}
