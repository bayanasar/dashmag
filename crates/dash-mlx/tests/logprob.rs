//! The chosen-token log-probability and entropy: `dash_conformance`'s `logprob` tests through
//! dash-mlx's adapter (`tests/common`), on the GPU, through its Metal chunk kernels and through the
//! reference built from MLX's operations: every case with and without its masked ids against
//! PyTorch in float64, in f32 and bf16, in chunks small and large and the implementation's own,
//! with and without the gradient, and an empty batch. The adapter asserts that each call makes one
//! pass over the vocabulary per block of rows, and one back where the gradient is asked for.
//!
//! What is dash-mlx's own stays here: the contract's refusals, two runs to the same bits, and the
//! kernels against the reference at a size the fixtures do not reach.
//!
//! f32 needs `MLX_ENABLE_TF32=0`:
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test logprob -- --nocapture
#![cfg(target_os = "macos")]

mod common;

use common::{Mlx, Reference, require_ieee_f32_matmul};
use dash_conformance::logprob::{self as conformance, Case, SMALL_CHUNKS};
use dash_conformance::{Adapter, DType};
use dash_mlx::logprob::{ChosenLogprob, LogprobOptions, chosen_logprob, chosen_logprob_reference};
use mlx_rs::error::Result;
use mlx_rs::transforms::fallible_vjp;
use mlx_rs::{Array, Dtype};
use std::sync::{Mutex, MutexGuard};

/// One test at a time: they read the probe counters, and share the GPU's default stream.
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
fn matches_pytorch() {
    let _serial = serial();
    require_ieee_f32_matmul();
    conformance::matches_pytorch(&Mlx, "kernels").all_ran();
    conformance::matches_pytorch(&Reference, "reference").all_ran();
}

#[test]
fn masked_tokens_leave_the_softmax() {
    let _serial = serial();
    require_ieee_f32_matmul();
    conformance::masked_tokens_leave_the_softmax(&Mlx, "kernels").all_ran();
    conformance::masked_tokens_leave_the_softmax(&Reference, "reference").all_ran();
}

#[test]
fn chunking_does_not_change_the_answer() {
    let _serial = serial();
    require_ieee_f32_matmul();
    conformance::chunking_does_not_change_the_answer(&Mlx, "kernels").all_ran();
    conformance::chunking_does_not_change_the_answer(&Reference, "reference").all_ran();
}

#[test]
fn without_gradients() {
    let _serial = serial();
    require_ieee_f32_matmul();
    conformance::without_gradients(&Mlx, "kernels").all_ran();
    conformance::without_gradients(&Reference, "reference").all_ran();
}

#[test]
fn empty_batch() {
    let _serial = serial();
    conformance::empty_batch(&Mlx).all_ran();
    conformance::empty_batch(&Reference).all_ran();
}

/// The contract's errors: shapes, masked ids, every token masked, a zero row chunk.
#[test]
fn refuses_what_the_contract_rules_out() {
    let _serial = serial();
    let hidden = Array::from_slice(&[0.0f32; 8], &[2, 4]);
    let weight = Array::from_slice(&[0.0f32; 12], &[3, 4]);
    let targets = Array::from_slice(&[0i64, 1], &[2]);
    let options = LogprobOptions::default();
    let narrow = Array::from_slice(&[0.0f32; 6], &[3, 2]);
    assert!(chosen_logprob(&hidden, &narrow, &targets, &[], options).is_err());
    assert!(chosen_logprob(&hidden, &weight, &targets, &[3], options).is_err());
    assert!(chosen_logprob(&hidden, &weight, &targets, &[0, 1, 2], options).is_err());
    let zero = LogprobOptions {
        row_chunk: 0,
        ..options
    };
    assert!(chosen_logprob(&hidden, &weight, &targets, &[], zero).is_err());
    let floats = Array::from_slice(&[0.0f32, 1.0], &[2]);
    assert!(chosen_logprob(&hidden, &weight, &floats, &[], options).is_err());
}

/// By default a run repeats bit for bit: the forward and backward passes, run twice, give the
/// same bits, in f32 and bf16, with masked ids and several chunks and row blocks.
#[test]
fn repeats_bit_for_bit() {
    let _serial = serial();
    let case = Case::load("peaked");
    for dtype in [DType::F32, DType::BF16] {
        let bits = || {
            let out = Mlx
                .chosen_logprob(&case.call(dtype, &case.masked, SMALL_CHUNKS))
                .unwrap();
            [out.logprob, out.entropy, out.grad_hidden.unwrap()]
                .map(|x| x.values.iter().map(|f| f.to_bits()).collect::<Vec<_>>())
        };
        assert!(bits() == bits(), "{dtype:?}: two runs differ");
    }
}

type Op = fn(&Array, &Array, &Array, &[u32], LogprobOptions) -> Result<ChosenLogprob>;

/// The kernels against the reference at a size the fixtures do not reach: D 1024, 20,000 tokens in
/// chunks of 3072 (the last partial), two blocks of rows, masked ids including a row's target and
/// a run of tokens across a chunk's edge, targets out of range; f32.
#[test]
fn kernels_match_the_reference_at_scale() -> Result<()> {
    let _serial = serial();
    require_ieee_f32_matmul();
    let (n, d, v) = (700, 1024, 20_000);
    let normal = |shape: &[i32], seed: u64| -> Result<Array> {
        mlx_rs::random::normal::<f32>(shape, None, None, &mlx_rs::random::key(seed)?)
    };
    let hidden = normal(&[n, d], 7)?;
    let weight = normal(&[v, d], 8)?.multiply(Array::from_f32(0.05))?;
    let upstream = normal(&[n], 9)?;
    let mut ids: Vec<i64> = (0..n as i64).map(|i| (i * 7919) % v as i64).collect();
    ids[3] = -1;
    ids[4] = v as i64;
    let masked = vec![ids[5] as u32, 0, 3070, 3071, 3072, 3073, (v - 1) as u32];
    let targets = Array::from_slice(&ids, &[n]);
    let options = LogprobOptions {
        vocab_chunk: 0,
        row_chunk: 400,
    };
    let run = |op: Op| -> Result<[Vec<f32>; 3]> {
        let mut entropy = None;
        let f = |a: &[Array]| -> Result<Vec<Array>> {
            let out = op(&a[0], &weight, &targets, &masked, options)?;
            entropy = Some(out.entropy);
            Ok(vec![out.logprob])
        };
        let (outputs, grads) = fallible_vjp(
            f,
            std::slice::from_ref(&hidden),
            std::slice::from_ref(&upstream),
        )?;
        Ok([
            values(&outputs[0]),
            values(&entropy.expect("ran")),
            values(&grads[0]),
        ])
    };
    let kernels = run(chosen_logprob)?;
    let reference = run(chosen_logprob_reference)?;
    for (i, what) in ["logprob", "entropy", "grad_hidden"].iter().enumerate() {
        let (abs, rel) = conformance::errors(what, &kernels[i], &reference[i]);
        println!(
            "kernels against the reference {what:12} max abs {abs:9.3e}  relative {}",
            up3(rel)
        );
        assert!(rel <= 1e-6, "{what}: {rel:e}");
    }
    for row in [3, 4, 5] {
        assert_eq!(kernels[0][row], f32::NEG_INFINITY, "row {row}");
    }
    Ok(())
}
