//! The chosen-token log-probability against PyTorch: `dash_conformance::logprob`'s tests (every
//! recorded case with and without its masked ids, in chunks of every kind, without a gradient and
//! for no rows; the crate says what the recording holds), run through `common::adapter`, and
//! what is about dash-ops alone.
//!
//! On Flex the reference path (Burn's tensor operations) is held to them, in f32 and bf16; on a
//! CubeCL device the kernel path where the device has what the chunk kernel asks, in bf16 (the
//! tensor-core path) and f32, and on a device without it the reference path on that device. Each
//! call also proves which code ran: the chunk kernel launches exactly as often as `kernels_take`
//! says (always under CUDA), and the written-out backward runs where a gradient is asked for.
//! Here besides: `chosen_logprob_with_grad`'s f32 gradient from bf16 inputs against the same
//! fixtures, the kernel path against the reference path at the real hidden width (1024) and with
//! row blocks large enough for its two-matmul branch, and a weight that requires a gradient.
//!
//! Errors are stated as `max |got - want|` and that over `max |want|` ("relative"), per output;
//! each tolerance sits just above the largest error measured.
//!
//!   cargo test --release -p dash-ops --test logprob
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test logprob
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda,fusion --test logprob
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test logprob

mod common;

use burn::backend::{DType, Dispatch};
use burn::tensor::{Device, Int, Tensor, TensorData};
use common::adapter::{Burn, assert_logprob_launches};
use dash_conformance::logprob::{self, CASES, Case, SMALL_CHUNKS, check};
use dash_ops::logprob::{ChosenLogprobBackend, LogprobOptions, chosen_logprob};
use std::sync::Mutex;

/// The probe counters are global; tests that read them take turns.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What one run returns, all as f32: the log-probabilities, the entropies and the gradient.
type Run = [Vec<f32>; 3];

/// The operation through `chosen_logprob_with_grad` on `device` (without autodiff), with inputs
/// in `dtype`: the gradient is the f32 one the backward pass would multiply by `upstream`, before
/// any rounding to `dtype`.
fn run_with_grad(
    device: &Device,
    case: &Case,
    dtype: DType,
    masked: &[u32],
    options: LogprobOptions,
) -> Run {
    let (n, v, d) = (case.n, case.v, case.d);
    let hidden =
        Tensor::<2>::from_data(TensorData::new(case.hidden.clone(), [n, d]), device).cast(dtype);
    let weight =
        Tensor::<2>::from_data(TensorData::new(case.weight.clone(), [v, d]), device).cast(dtype);
    let targets = Tensor::<1, Int>::from_data(TensorData::new(case.targets.clone(), [n]), device);
    let dtypes = [dtype, dtype, targets.dtype()];
    let launches = dash_ops::logprob::probe::kernel_launches();
    let (logprob, entropy, grad) = <Dispatch as ChosenLogprobBackend>::chosen_logprob_with_grad(
        hidden.into_dispatch(),
        weight.into_dispatch(),
        targets.into_dispatch(),
        masked.to_vec(),
        options,
    );
    let grad = Tensor::<2>::from_dispatch(grad);
    assert_eq!(grad.dtype(), DType::F32);
    let grad = values(&grad)
        .chunks_exact(d)
        .zip(&case.upstream)
        .flat_map(|(row, g)| row.iter().map(move |x| x * g))
        .collect();
    let run = [
        values(&Tensor::<1>::from_dispatch(logprob)),
        values(&Tensor::<1>::from_dispatch(entropy)),
        grad,
    ];
    assert_logprob_launches(device, dtypes, [n, v], options, launches);
    run
}

fn values<const D: usize>(x: &Tensor<D>) -> Vec<f32> {
    x.to_data().convert::<f32>().try_to_vec::<f32>().unwrap()
}

fn parts(run: &Run) -> [&[f32]; 3] {
    [&run[0], &run[1], &run[2]]
}

/// The f32 gradient from bf16 inputs, before its rounding to bf16, against PyTorch on Flex.
const FLEX_BF16_F32_GRAD: [f64; 3] = [2e-7, 2e-6, 1e-6];

#[cfg(feature = "flex")]
#[test]
fn matches_pytorch_on_flex() {
    let _serial = serial();
    logprob::matches_pytorch(&Burn::new(Device::flex()), "flex").all_ran();
}

#[cfg(feature = "flex")]
#[test]
fn bf16_inputs_with_an_f32_gradient_on_flex() {
    let _serial = serial();
    for name in CASES {
        let case = Case::load(name);
        let got = run_with_grad(
            &Device::flex(),
            &case,
            DType::BF16,
            &case.masked,
            SMALL_CHUNKS,
        );
        check(
            &format!("flex bf16 f32-grad {name}"),
            parts(&got),
            parts(&case.masked_reference),
            FLEX_BF16_F32_GRAD,
        );
    }
}

/// Masking a token must change every row's answer (it leaves the log-sum-exp), and without the
/// mask the operation must match PyTorch without it.
#[cfg(feature = "flex")]
#[test]
fn masked_tokens_leave_the_softmax_on_flex() {
    let _serial = serial();
    logprob::masked_tokens_leave_the_softmax(&Burn::new(Device::flex()), "flex").all_ran();
}

/// The chunking is an implementation detail: one chunk, tiny chunks and uneven row blocks agree.
#[cfg(feature = "flex")]
#[test]
fn chunking_does_not_change_the_answer_on_flex() {
    let _serial = serial();
    logprob::chunking_does_not_change_the_answer(&Burn::new(Device::flex()), "flex").all_ran();
}

/// Without autodiff the same values come out, and nothing is recorded.
#[cfg(feature = "flex")]
#[test]
fn untracked_on_flex() {
    let _serial = serial();
    logprob::without_gradients(&Burn::new(Device::flex()), "flex").all_ran();
}

#[cfg(feature = "flex")]
#[test]
fn empty_batch_on_flex() {
    let _serial = serial();
    logprob::empty_batch(&Burn::new(Device::flex())).all_ran();
}

#[cfg(feature = "flex")]
#[test]
#[should_panic(expected = "frozen")]
fn a_weight_that_requires_a_gradient_is_refused() {
    let device = Device::flex().autodiff();
    let hidden = Tensor::<2>::zeros([2, 3], &device).require_grad();
    let weight = Tensor::<2>::zeros([5, 3], &device).require_grad();
    let targets = Tensor::<1, Int>::from_ints([0, 1], &device);
    let _ = chosen_logprob(hidden, weight, targets, &[], LogprobOptions::default());
}

/// The same checks on a CubeCL device, any runtime: the one `DASH_OPS_DEVICE` names. Each call
/// asserts the chunk kernel's launches against `kernels_take`, so where the device lacks what the
/// kernel asks these hold the reference path on that device to the same fixtures.
#[cfg(feature = "cubecl")]
mod cube {
    use super::*;

    /// bf16 inputs, with the f32 gradient before its rounding to bf16
    /// (`chosen_logprob_with_grad`). Measured: 4.5e-7, 5.8e-7, 9.0e-7 (the largest at D = 1024,
    /// against Flex).
    const CUBE_BF16_F32_GRAD: [f64; 3] = [1e-6, 1e-6, 2e-6];

    fn device() -> Device {
        common::cube_device()
    }

    fn cube() -> Burn {
        Burn::new(device())
    }

    #[test]
    fn matches_pytorch_on_cubecl() {
        let _serial = serial();
        println!("{}", common::describe(&device()));
        let cube = cube();
        logprob::matches_pytorch(&cube, "cubecl").all_ran_without(&cube.lacking());
    }

    #[test]
    fn bf16_inputs_with_an_f32_gradient_on_cubecl() {
        let _serial = serial();
        if !common::has(&device(), DType::BF16) {
            return;
        }
        for name in CASES {
            let case = Case::load(name);
            let got = run_with_grad(&device(), &case, DType::BF16, &case.masked, SMALL_CHUNKS);
            check(
                &format!("cubecl bf16 f32-grad {name}"),
                parts(&got),
                parts(&case.masked_reference),
                CUBE_BF16_F32_GRAD,
            );
        }
    }

    #[test]
    fn masked_tokens_leave_the_softmax_on_cubecl() {
        let _serial = serial();
        let cube = cube();
        logprob::masked_tokens_leave_the_softmax(&cube, "cubecl").all_ran_without(&cube.lacking());
    }

    #[test]
    fn chunking_does_not_change_the_answer_on_cubecl() {
        let _serial = serial();
        let cube = cube();
        logprob::chunking_does_not_change_the_answer(&cube, "cubecl")
            .all_ran_without(&cube.lacking());
    }

    #[test]
    fn untracked_on_cubecl() {
        let _serial = serial();
        let cube = cube();
        logprob::without_gradients(&cube, "cubecl").all_ran_without(&cube.lacking());
    }

    #[test]
    fn empty_batch_on_cubecl() {
        let _serial = serial();
        logprob::empty_batch(&cube()).all_ran();
    }

    /// Random inputs, rounded to bf16 so that both devices see one value.
    #[cfg(feature = "flex")]
    fn random_case(n: usize, v: usize, d: usize, masked: &[u32]) -> Case {
        let mut rng = 0x2545_f491_4f6c_dd1du64 ^ (n * v * d) as u64;
        let mut normal = |scale: f32| {
            // Box-Muller on a xorshift.
            let mut uniform = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (uniform(), uniform());
            let x = (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos();
            burn::backend::bf16::from_f32(x as f32 * scale).to_f32()
        };
        Case {
            name: format!("random {n} x {v} x {d}"),
            n,
            v,
            d,
            hidden: (0..n * d).map(|_| normal(1.0)).collect(),
            weight: (0..v * d)
                .map(|_| normal(1.6 / (d as f32).sqrt()))
                .collect(),
            targets: (0..n as i64).map(|i| (i * 7919 + 13) % v as i64).collect(),
            upstream: (0..n).map(|_| normal(1.0)).collect(),
            masked: masked.to_vec(),
            masked_reference: Default::default(),
            unmasked_reference: Default::default(),
        }
    }

    /// The CubeCL path with bf16 operands (f32 on a device without bf16) against the reference
    /// path on Flex in f32, on the same values.
    #[cfg(feature = "flex")]
    fn agrees_with_flex(label: &str, case: &Case, options: LogprobOptions) {
        let dtype = match common::has(&device(), DType::BF16) {
            true => DType::BF16,
            false => DType::F32,
        };
        let want = run_with_grad(&Device::flex(), case, DType::F32, &case.masked, options);
        let got = run_with_grad(&device(), case, dtype, &case.masked, options);
        check(
            &format!("{label} {dtype:?}"),
            parts(&got),
            parts(&want),
            CUBE_BF16_F32_GRAD,
        );
    }

    /// At the real hidden width (1024): several row blocks, chunks and a partial last chunk;
    /// then the defaults (one chunk of the vocabulary, one block of rows).
    #[cfg(feature = "flex")]
    #[test]
    fn agrees_with_flex_at_full_width() {
        let _serial = serial();
        let case = random_case(100, 5000, 1024, &[17, 4096, 4999]);
        let small = LogprobOptions {
            vocab_chunk: 1024,
            row_chunk: 40,
        };
        agrees_with_flex("cubecl vs flex, D 1024", &case, small);
        agrees_with_flex("cubecl defaults vs flex", &case, LogprobOptions::default());
    }

    /// A block of 1800 rows takes the two halves of the split weights through two matmuls; 900
    /// rows, through one stacked matmul.
    #[cfg(feature = "flex")]
    #[test]
    fn agrees_with_flex_with_large_row_blocks() {
        let _serial = serial();
        let case = random_case(1800, 700, 64, &[0, 350]);
        let options = |row_chunk| LogprobOptions {
            vocab_chunk: 256,
            row_chunk,
        };
        agrees_with_flex("cubecl vs flex, 1800 rows", &case, options(1800));
        agrees_with_flex("cubecl vs flex, 2 x 900 rows", &case, options(1000));
    }
}
