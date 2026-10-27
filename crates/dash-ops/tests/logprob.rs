//! The chosen-token log-probability against PyTorch.
//!
//! `fixtures/logprob/cases.safetensors` holds three cases (inputs rounded to bf16, so that a
//! bf16 test casts them without loss) and their reference, computed by PyTorch in float64 and
//! stored as f32: the log-probabilities, the entropies and the gradient of `sum_i g[i] logprob[i]`
//! for `hidden`, with and without the masked token ids. `basic` has two masked ids at the two ends
//! of the vocabulary; `peaked` has logits up to 49, rows whose target has p = 0.99 and a masked
//! token that is the argmax of the most rows; `wide` has D = 256. In each, the last row's target
//! is a masked token, so its log-probability is `-inf`.
//!
//! On Flex the reference path (Burn's tensor operations) is held to them, in f32 and bf16; on a
//! CubeCL device the kernel path where the device has what the chunk kernel asks, in bf16 (the
//! tensor-core path) and f32, and also to the reference path at the real hidden width (1024) and
//! with row blocks large enough for its two-matmul branch; on a device without it, the reference
//! path on that device. Each run also proves which code ran: the chunk kernel launches exactly as
//! often as `kernels_take` says (always under CUDA), and the written-out backward's runs are
//! counted.
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
use dash_ops::logprob::{
    ChosenLogprobBackend, LogprobOptions, chosen_logprob, kernels_take, probe,
};
use safetensors::SafeTensors;
use std::sync::Mutex;

/// The probe counters are global; tests that read them take turns.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/logprob");
const CASES: [&str; 3] = ["basic", "peaked", "wide"];

/// Chunks small enough that every case crosses several, the last one partial, and row blocks
/// that split the rows unevenly.
const SMALL_CHUNKS: LogprobOptions = LogprobOptions {
    vocab_chunk: 128,
    row_chunk: 10,
};

/// The chunk kernel's launches in one pass over a vocabulary of `v` tokens for `n` rows: one per
/// chunk of the vocabulary in each block of rows, the blocks of equal size but for the last.
fn pass_launches(n: usize, v: usize, options: LogprobOptions) -> usize {
    let block = n.div_ceil(n.div_ceil(options.row_chunk).max(1)).max(1);
    (0..n)
        .step_by(block)
        .map(|r0| v.div_ceil(options.vocab_chunk_for(block.min(n - r0))))
        .sum()
}

/// Asserts that since `before` the chunk kernel ran one pass's launches for `n` rows over `v`
/// tokens in `dtypes` if `kernels_take` says the kernels take them on `device`, and never
/// otherwise; under CUDA they must take them.
fn assert_launches(
    device: &Device,
    dtypes: [DType; 3],
    [n, v]: [usize; 2],
    options: LogprobOptions,
    before: usize,
) {
    let takes = kernels_take(device, dtypes, n, options);
    if common::on_cuda(device) {
        assert!(takes, "CUDA turned the chunk kernel away");
    }
    let want = match takes && n > 0 {
        true => pass_launches(n, v, options),
        false => 0,
    };
    let ran = probe::kernel_launches() - before;
    assert_eq!(
        ran, want,
        "the chunk kernel ran {ran} times, expected {want}"
    );
}

struct Case {
    n: usize,
    v: usize,
    d: usize,
    hidden: Vec<f32>,
    weight: Vec<f32>,
    targets: Vec<i64>,
    upstream: Vec<f32>,
    masked: Vec<u32>,
    logprob: Vec<f32>,
    entropy: Vec<f32>,
    grad_hidden: Vec<f32>,
    logprob_unmasked: Vec<f32>,
    entropy_unmasked: Vec<f32>,
    grad_hidden_unmasked: Vec<f32>,
}

fn load(name: &str) -> Case {
    let bytes = std::fs::read(format!("{FIXTURES}/cases.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let f32s = |key: &str| -> Vec<f32> {
        let view = tensors.tensor(&format!("{name}.{key}")).unwrap();
        assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name}.{key}");
        view.data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let targets = tensors.tensor(&format!("{name}.targets")).unwrap();
    assert_eq!(targets.dtype(), safetensors::Dtype::I64);
    let targets = targets
        .data()
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FIXTURES}/meta.json")).unwrap())
            .unwrap();
    let meta = &meta[name];
    let dim = |key: &str| meta[key].as_u64().unwrap() as usize;
    Case {
        n: dim("n"),
        v: dim("v"),
        d: dim("d"),
        hidden: f32s("hidden"),
        weight: f32s("weight"),
        targets,
        upstream: f32s("upstream"),
        masked: meta["masked"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect(),
        logprob: f32s("logprob"),
        entropy: f32s("entropy"),
        grad_hidden: f32s("grad_hidden"),
        logprob_unmasked: f32s("logprob_unmasked"),
        entropy_unmasked: f32s("entropy_unmasked"),
        grad_hidden_unmasked: f32s("grad_hidden_unmasked"),
    }
}

/// What one run returns, all as f32.
struct Run {
    logprob: Vec<f32>,
    entropy: Vec<f32>,
    grad: Vec<f32>,
}

/// The operation on `device` (an autodiff device) with inputs in `dtype`, and the gradient of
/// `sum_i upstream[i] logprob[i]` for `hidden`.
fn run(device: &Device, case: &Case, dtype: DType, masked: &[u32], options: LogprobOptions) -> Run {
    let (n, v, d) = (case.n, case.v, case.d);
    let hidden = Tensor::<2>::from_data(TensorData::new(case.hidden.clone(), [n, d]), device)
        .cast(dtype)
        .require_grad();
    let weight =
        Tensor::<2>::from_data(TensorData::new(case.weight.clone(), [v, d]), device).cast(dtype);
    let targets = Tensor::<1, Int>::from_data(TensorData::new(case.targets.clone(), [n]), device);
    let upstream = Tensor::<1>::from_data(TensorData::new(case.upstream.clone(), [n]), device);
    let dtypes = [dtype, dtype, targets.dtype()];
    let launches = probe::kernel_launches();
    let out = chosen_logprob(hidden.clone(), weight, targets, masked, options);
    assert_eq!(out.logprob.dtype(), DType::F32);
    assert_eq!(out.entropy.dtype(), DType::F32);
    let logprob = values(&out.logprob);
    let entropy = values(&out.entropy);
    let grads = (out.logprob * upstream).sum().backward();
    let grad = hidden.grad(&grads).expect("hidden has a gradient");
    assert_eq!(
        grad.dtype(),
        dtype,
        "the gradient comes back in hidden's dtype"
    );
    // Read after the values: under fusion the kernels run when their results are read.
    let grad = values(&grad);
    assert_launches(device, dtypes, [n, v], options, launches);
    Run {
        logprob,
        entropy,
        grad,
    }
}

/// The same through `chosen_logprob_with_grad`, on a device without autodiff: the gradient is
/// the f32 one the backward pass would multiply by `upstream`, before any rounding to `dtype`.
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
    let launches = probe::kernel_launches();
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
    let run = Run {
        logprob: values(&Tensor::<1>::from_dispatch(logprob)),
        entropy: values(&Tensor::<1>::from_dispatch(entropy)),
        grad,
    };
    assert_launches(device, dtypes, [n, v], options, launches);
    run
}

fn values<const D: usize>(x: &Tensor<D>) -> Vec<f32> {
    x.to_data().convert::<f32>().try_to_vec::<f32>().unwrap()
}

/// `(max |got - want|, that / max |want|)` over the finite entries; `-inf` must match `-inf`.
fn error(what: &str, got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut max_abs = 0.0f64;
    let mut scale = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.is_infinite() {
            assert_eq!(g, w, "{what}[{i}]: want {w}, got {g}");
            continue;
        }
        assert!(g.is_finite(), "{what}[{i}]: want {w}, got {g}");
        max_abs = max_abs.max((g as f64 - w as f64).abs());
        scale = scale.max((w as f64).abs());
    }
    (max_abs, max_abs / scale)
}

/// Asserts each output's relative error is within its tolerance, and prints the measured ones.
fn check(label: &str, run: &Run, want: [&[f32]; 3], tolerance: [f64; 3]) {
    for ((what, got), (want, tolerance)) in [
        ("logprob", &run.logprob),
        ("entropy", &run.entropy),
        ("grad_hidden", &run.grad),
    ]
    .into_iter()
    .zip(want.into_iter().zip(tolerance))
    {
        let (abs, rel) = error(what, got, want);
        println!(
            "{label:28} {what:12} max abs {abs:9.3e}  relative {rel:9.3e}  (tolerance {tolerance:.0e})"
        );
        assert!(
            rel <= tolerance,
            "{label} {what}: relative error {rel:e} above {tolerance:e} (max abs {abs:e})"
        );
    }
}

fn masked_reference(case: &Case) -> [&[f32]; 3] {
    [&case.logprob, &case.entropy, &case.grad_hidden]
}

fn unmasked_reference(case: &Case) -> [&[f32]; 3] {
    [
        &case.logprob_unmasked,
        &case.entropy_unmasked,
        &case.grad_hidden_unmasked,
    ]
}

/// The reference path (Burn's tensor operations in f32) against float64 PyTorch. Measured: 8.8e-8,
/// 1.1e-6, 7.0e-7. The entropy of `peaked` (logits up to 49) is at f32's noise floor: PyTorch's
/// own f32 lands 9e-7 from float64 there.
const FLEX_F32: [f64; 3] = [2e-7, 2e-6, 1e-6];
/// bf16 inputs on Flex are widened to f32 inside; the gradient is rounded to bf16 on the way out
/// (half an ulp is 2^-9 of the value, at most 2^-8 relative to a power of two).
const FLEX_BF16: [f64; 3] = [2e-7, 2e-6, 4e-3];

#[cfg(feature = "flex")]
#[test]
fn matches_pytorch_on_flex() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    for name in CASES {
        let case = load(name);
        let runs = probe::backward_runs();
        let got = run(&device, &case, DType::F32, &case.masked, SMALL_CHUNKS);
        assert!(
            probe::backward_runs() > runs,
            "the written-out backward did not run"
        );
        check(
            &format!("flex f32 {name}"),
            &got,
            masked_reference(&case),
            FLEX_F32,
        );
    }
}

#[cfg(feature = "flex")]
#[test]
fn bf16_inputs_on_flex() {
    let _serial = serial();
    for name in CASES {
        let case = load(name);
        let got = run(
            &Device::flex().autodiff(),
            &case,
            DType::BF16,
            &case.masked,
            SMALL_CHUNKS,
        );
        check(
            &format!("flex bf16 {name}"),
            &got,
            masked_reference(&case),
            FLEX_BF16,
        );
        let got = run_with_grad(
            &Device::flex(),
            &case,
            DType::BF16,
            &case.masked,
            SMALL_CHUNKS,
        );
        check(
            &format!("flex bf16 f32-grad {name}"),
            &got,
            masked_reference(&case),
            FLEX_F32,
        );
    }
}

/// Masking a token must change every row's answer (it leaves the log-sum-exp), and without the
/// mask the operation must match PyTorch without it.
#[cfg(feature = "flex")]
#[test]
fn masked_tokens_leave_the_softmax_on_flex() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    for name in CASES {
        let case = load(name);
        let masked = run(&device, &case, DType::F32, &case.masked, SMALL_CHUNKS);
        let unmasked = run(&device, &case, DType::F32, &[], SMALL_CHUNKS);
        check(
            &format!("flex unmasked {name}"),
            &unmasked,
            unmasked_reference(&case),
            FLEX_F32,
        );
        assert_masking_matters(name, &case, &masked, &unmasked);
    }
}

/// The mask moves each row's log-probability by what it moves PyTorch's, some by far more than
/// any tolerance here (in `peaked` the masked token is most rows' argmax; elsewhere its
/// probability is too small to show); the last row's target is masked: `-inf` with the mask only.
fn assert_masking_matters(name: &str, case: &Case, masked: &Run, unmasked: &Run) {
    let n = case.n;
    let mut largest = 0.0f32;
    for i in 0..n - 1 {
        let want = case.logprob[i] - case.logprob_unmasked[i];
        let got = masked.logprob[i] - unmasked.logprob[i];
        assert!(
            (got - want).abs() <= 2e-5 + 1e-3 * want.abs(),
            "{name} row {i}: the mask moved logprob by {got:e}, PyTorch's by {want:e}"
        );
        largest = largest.max(got.abs());
    }
    assert!(
        largest > 5e-3,
        "{name}: the mask moved no logprob by more than {largest:e}"
    );
    assert_eq!(
        masked.logprob[n - 1],
        f32::NEG_INFINITY,
        "{name}: a masked target"
    );
    assert!(
        unmasked.logprob[n - 1].is_finite(),
        "{name}: the same target unmasked"
    );
    let (_, rel) = error("grad", &masked.grad, &unmasked.grad);
    assert!(
        rel > 1e-3,
        "{name}: the mask moved the gradient by only {rel:e}"
    );
}

/// The chunking is an implementation detail: one chunk, tiny chunks and uneven row blocks agree.
#[cfg(feature = "flex")]
#[test]
fn chunking_does_not_change_the_answer_on_flex() {
    let _serial = serial();
    let device = Device::flex().autodiff();
    let case = load("peaked");
    for options in [
        LogprobOptions {
            vocab_chunk: case.v,
            row_chunk: case.n,
        },
        LogprobOptions {
            vocab_chunk: 7,
            row_chunk: 5,
        },
        LogprobOptions {
            vocab_chunk: 1000,
            row_chunk: 1,
        },
    ] {
        let got = run(&device, &case, DType::F32, &case.masked, options);
        check(
            &format!("flex {options:?}"),
            &got,
            masked_reference(&case),
            FLEX_F32,
        );
    }
}

/// Without autodiff the same values come out, and nothing is recorded.
#[cfg(feature = "flex")]
#[test]
fn untracked_on_flex() {
    let _serial = serial();
    let device = Device::flex();
    let case = load("basic");
    let (n, v, d) = (case.n, case.v, case.d);
    let runs = probe::backward_runs();
    let out = chosen_logprob(
        Tensor::<2>::from_data(TensorData::new(case.hidden.clone(), [n, d]), &device),
        Tensor::<2>::from_data(TensorData::new(case.weight.clone(), [v, d]), &device),
        Tensor::<1, Int>::from_data(TensorData::new(case.targets.clone(), [n]), &device),
        &case.masked,
        SMALL_CHUNKS,
    );
    let (_, rel) = error("logprob", &values(&out.logprob), &case.logprob);
    assert!(rel <= FLEX_F32[0]);
    let (_, rel) = error("entropy", &values(&out.entropy), &case.entropy);
    assert!(rel <= FLEX_F32[1]);
    assert_eq!(probe::backward_runs(), runs);
}

/// An empty batch (no rows at all) gives empty outputs and an empty gradient.
fn empty_batch(device: &Device) {
    let hidden = Tensor::<2>::zeros([0, 3], device).require_grad();
    let weight = Tensor::<2>::ones([5, 3], device);
    let targets = Tensor::<1, Int>::zeros([0], device);
    let out = chosen_logprob(
        hidden.clone(),
        weight,
        targets,
        &[1],
        LogprobOptions::default(),
    );
    assert_eq!(out.logprob.dims(), [0]);
    assert_eq!(out.entropy.dims(), [0]);
    let grads = out.logprob.sum().backward();
    assert_eq!(hidden.grad(&grads).map(|g| g.dims()), Some([0, 3]));
}

#[cfg(feature = "flex")]
#[test]
fn empty_batch_on_flex() {
    empty_batch(&Device::flex().autodiff());
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

/// The same checks on a CubeCL device, any runtime: the one `DASH_OPS_DEVICE` names. Each run
/// asserts the chunk kernel's launches against `kernels_take`, so where the device lacks what the
/// kernel asks these hold the reference path on that device to the same fixtures.
#[cfg(feature = "cubecl")]
mod cube {
    use super::*;

    /// bf16 inputs: the tensor-core path, with the gradient's weights split in two bf16 parts.
    /// The gradient is rounded to bf16 on the way out. Measured: 1.8e-7, 6.0e-7, 3.3e-3.
    const CUBE_BF16: [f64; 3] = [2e-7, 2e-6, 4e-3];
    /// The same, with the f32 gradient before that rounding (`chosen_logprob_with_grad`).
    /// Measured: 4.5e-7, 5.8e-7, 9.0e-7 (the largest at D = 1024, against Flex).
    const CUBE_BF16_F32_GRAD: [f64; 3] = [1e-6, 1e-6, 2e-6];
    /// f32 inputs: f32 matmuls, nothing split. Measured: 8.8e-8, 9.1e-7, 3.1e-7.
    const CUBE_F32: [f64; 3] = [2e-7, 2e-6, 1e-6];

    fn device() -> Device {
        common::cube_device()
    }

    /// The dtypes `device` has of bf16 and f32, with their tolerances against PyTorch.
    fn dtypes(device: &Device) -> Vec<(DType, [f64; 3])> {
        [(DType::BF16, CUBE_BF16), (DType::F32, CUBE_F32)]
            .into_iter()
            .filter(|&(dtype, _)| common::has(device, dtype))
            .collect()
    }

    #[test]
    fn matches_pytorch_on_cubecl_bf16() {
        let _serial = serial();
        println!("{}", common::describe(&device()));
        if !common::has(&device(), DType::BF16) {
            return;
        }
        for name in CASES {
            let case = load(name);
            let runs = probe::backward_runs();
            let got = run(
                &device().autodiff(),
                &case,
                DType::BF16,
                &case.masked,
                SMALL_CHUNKS,
            );
            assert!(
                probe::backward_runs() > runs,
                "the written-out backward did not run"
            );
            check(
                &format!("cubecl bf16 {name}"),
                &got,
                masked_reference(&case),
                CUBE_BF16,
            );
            let got = run_with_grad(&device(), &case, DType::BF16, &case.masked, SMALL_CHUNKS);
            check(
                &format!("cubecl bf16 f32-grad {name}"),
                &got,
                masked_reference(&case),
                CUBE_BF16_F32_GRAD,
            );
        }
    }

    #[test]
    fn matches_pytorch_on_cubecl_f32() {
        let _serial = serial();
        for name in CASES {
            let case = load(name);
            let runs = probe::backward_runs();
            let got = run(
                &device().autodiff(),
                &case,
                DType::F32,
                &case.masked,
                SMALL_CHUNKS,
            );
            assert!(
                probe::backward_runs() > runs,
                "the written-out backward did not run"
            );
            check(
                &format!("cubecl f32 {name}"),
                &got,
                masked_reference(&case),
                CUBE_F32,
            );
        }
    }

    #[test]
    fn empty_batch_on_cubecl() {
        empty_batch(&device().autodiff());
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
            logprob: vec![],
            entropy: vec![],
            grad_hidden: vec![],
            logprob_unmasked: vec![],
            entropy_unmasked: vec![],
            grad_hidden_unmasked: vec![],
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
            &got,
            [&want.logprob, &want.entropy, &want.grad],
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

    #[test]
    fn masked_tokens_leave_the_softmax_on_cubecl() {
        let _serial = serial();
        let device = device().autodiff();
        for (dtype, tolerance) in dtypes(&device) {
            for name in CASES {
                let case = load(name);
                let masked = run(&device, &case, dtype, &case.masked, SMALL_CHUNKS);
                let unmasked = run(&device, &case, dtype, &[], SMALL_CHUNKS);
                check(
                    &format!("cubecl {dtype:?} unmasked {name}"),
                    &unmasked,
                    unmasked_reference(&case),
                    tolerance,
                );
                assert_masking_matters(name, &case, &masked, &unmasked);
            }
        }
    }
}
