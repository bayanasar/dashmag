//! The chunked gated delta rule against transformers' `torch_chunk_gated_delta_rule`.
//!
//! The fixtures in `tests/fixtures/gated_delta_chunk/` hold inputs, the gradients fed back
//! (`d_out`, `d_state`), and what torch computed from them: the output, the final state and the
//! gradient of every input, at f32 (`out`, `dq`, ...), and with the function's f32 casts turned into
//! f64 (`out_f64`, ...; stored rounded to f32). The bf16 case holds bf16 inputs (as f32 values),
//! torch's own bf16 results (`out_torch_bf16`, ...) and the f64 results from the same inputs.
//!
//! With a CubeCL runtime's feature the same checks run on that runtime's device
//! (`tests/common`), the kernels where the device has what they ask and the reference elsewhere.
//!
//!   cargo test --release -p dash-ops --test gated_delta_chunk -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test gated_delta_chunk
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test gated_delta_chunk

mod common;

use burn::tensor::{DType, Device, Tensor, TensorData};
use dash_ops::gated_delta::chunk::{ChunkGatedDeltaOptions, chunk_gated_delta_rule, probe};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

/// One test at a time, so that the counters each test reads (written-out backward runs, kernel
/// launches) count only that test's work.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

struct Case {
    name: &'static str,
    meta: HashMap<String, serde_json::Value>,
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Case {
    fn load(name: &'static str) -> Self {
        let path = format!(
            "{}/tests/fixtures/gated_delta_chunk/{name}.safetensors",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let (_, metadata) = SafeTensors::read_metadata(&bytes).unwrap();
        let meta = metadata
            .metadata()
            .as_ref()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::from_str(v).unwrap()))
            .collect();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let tensors = st
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                assert_eq!(view.dtype(), Dtype::F32, "{name}");
                let data = view
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                (name, (view.shape().to_vec(), data))
            })
            .collect();
        Self {
            name,
            meta,
            tensors,
        }
    }

    fn flag(&self, key: &str) -> bool {
        self.meta[key].as_bool().unwrap()
    }

    fn get(&self, key: &str) -> &[f32] {
        &self.tensors[key].1
    }

    fn tensor<const D: usize>(&self, key: &str, device: &Device) -> Tensor<D> {
        let (shape, data) = &self.tensors[key];
        Tensor::from_data(TensorData::new(data.clone(), shape.clone()), device)
    }
}

/// What one run of the operation gave: output, final state, and gradients, as f32 vectors.
type Results = HashMap<&'static str, Vec<f32>>;

fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.cast(DType::F32)
        .into_data()
        .try_into_vec::<f32>()
        .unwrap()
}

/// Runs the case's forward and backward on `device` with inputs in `dtype`.
fn run(case: &Case, device: &Device, dtype: DType) -> Results {
    let leaf3 = |key: &str| case.tensor::<3>(key, device).cast(dtype).require_grad();
    let leaf4 = |key: &str| case.tensor::<4>(key, device).cast(dtype).require_grad();
    let (q, k, v) = (leaf4("q"), leaf4("k"), leaf4("v"));
    let (g, beta) = (leaf3("g"), leaf3("beta"));
    let h0 = case
        .flag("initial_state")
        .then(|| case.tensor::<4>("h0", device).require_grad());
    let state_grad = case.flag("final_state_grad");
    let runs = probe::backward_runs();
    let options = ChunkGatedDeltaOptions {
        qk_l2norm: case.flag("l2norm"),
        output_final_state: state_grad,
        ..Default::default()
    };
    let (out, state) = chunk_gated_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        h0.clone(),
        None,
        options,
    );
    let mut loss = (out.clone().cast(DType::F32) * case.tensor::<4>("d_out", device)).sum();
    if state_grad {
        let state = state.clone().unwrap();
        assert_eq!(state.dtype(), DType::F32);
        loss = loss + (state * case.tensor::<4>("d_state", device)).sum();
    }
    let grads = loss.backward();
    assert!(
        probe::backward_runs() > runs,
        "the written-out backward did not run"
    );
    assert_eq!(out.dtype(), dtype);
    let mut res = Results::new();
    res.insert("out", to_vec(out));
    if let Some(state) = state {
        res.insert("state", to_vec(state));
    }
    res.insert("dq", to_vec(q.grad(&grads).unwrap()));
    res.insert("dk", to_vec(k.grad(&grads).unwrap()));
    res.insert("dv", to_vec(v.grad(&grads).unwrap()));
    res.insert("dg", to_vec(g.grad(&grads).unwrap()));
    res.insert("dbeta", to_vec(beta.grad(&grads).unwrap()));
    if let Some(h0) = h0 {
        res.insert("dh0", to_vec(h0.grad(&grads).unwrap()));
    }
    res
}

/// Max abs error, and max abs error relative to the reference's max abs value.
fn errors(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_err = 0f32;
    let mut scale = 0f32;
    for (a, e) in actual.iter().zip(expected) {
        assert!(a.is_finite(), "non-finite value {a}");
        max_err = max_err.max((a - e).abs());
        scale = scale.max(e.abs());
    }
    (max_err, max_err / scale)
}

/// Compares every result with the reference `<name><suffix>`, printing the errors, and asserts
/// the relative error is at most `tol`.
fn check(case: &Case, label: &str, res: &Results, suffix: &str, tol: f32) {
    let mut keys: Vec<_> = res.keys().copied().collect();
    keys.sort();
    let mut failed = Vec::new();
    for key in keys {
        let (abs, rel) = errors(&res[key], case.get(&format!("{key}{suffix}")));
        println!(
            "{:20} {label:12} {key:6} vs {:13} max abs {abs:.3e} rel {rel:.3e}",
            case.name,
            format!("torch{suffix}")
        );
        if rel > tol {
            failed.push(format!("{key} {rel:.3e}"));
        }
    }
    assert!(
        failed.is_empty(),
        "{} {label}: relative errors above {tol:.1e}: {failed:?}",
        case.name
    );
}

const F32_CASES: [&str; 3] = ["f32_l2norm_state", "f32_plain", "f32_short"];

/// fp32 against torch at fp32 (and at f64): the requirement is 1e-5 relative; the measured errors
/// are at most 1.2e-6, on Flex, on CUDA, and through the reference on lavapipe, on OpenGL over
/// D3D12 and on CubeCL's CPU runtime (printed with --nocapture).
const F32_TOL: f32 = 2e-6;

/// bf16 inputs, against the f64 reference from the same inputs, where the reference or the f32
/// kernels run: they compute in f32 and round once, to the outputs' dtype, so the error is about
/// one bf16 rounding of the largest values (2^-9 relative): measured at most 2.5e-3. torch's own
/// bf16 path, which normalises queries and keys in bf16, is further off (up to 8.1e-3, printed
/// beside it).
const BF16_TOL: f32 = 3e-3;

/// The same where the kernels on tensor cores run, whose products take bf16 operands (the state,
/// `W`, `U`, `T`, `V'` and the backward's operands rounded to bf16, as fla's are): measured at
/// most 6.12e-3 (the output; the queries' gradient 5.34e-3), on CUDA in every build.
#[cfg(feature = "cubecl")]
const BF16_TC_TOL: f32 = 6.5e-3;

/// f16 inputs: the kernels against the f32 reference from the same inputs. f16 rounds to within
/// 2^-11 relative: measured at most 4.2e-4 through the reference on lavapipe and on CubeCL's CPU
/// runtime; on tensor cores, with f16 operands, at most 7.5e-4, on CUDA in every build.
#[cfg(feature = "cubecl")]
const F16_VS_REFERENCE: f32 = 5e-4;
#[cfg(feature = "cubecl")]
const F16_TC_VS_REFERENCE: f32 = 8e-4;

#[cfg(feature = "flex")]
fn check_f32_cases(device: &Device, label: &str) {
    for name in F32_CASES {
        let case = Case::load(name);
        let res = run(&case, device, DType::F32);
        check(&case, label, &res, "", F32_TOL);
        check(&case, label, &res, "_f64", F32_TOL);
    }
}

fn check_bf16_case(device: &Device, label: &str, tol: f32) {
    let case = Case::load("bf16_l2norm_state");
    let res = run(&case, device, DType::BF16);
    check(&case, label, &res, "_f64", tol);
    // For the record: torch's own bf16 path against the same f64 reference.
    let mut keys: Vec<_> = res.keys().copied().collect();
    keys.sort();
    for key in keys {
        let (abs, rel) = errors(
            case.get(&format!("{key}_torch_bf16")),
            case.get(&format!("{key}_f64")),
        );
        println!(
            "{:20} {:12} {key:6} vs torch_f64     max abs {abs:.3e} rel {rel:.3e}",
            case.name, "torch-bf16"
        );
    }
}

#[cfg(feature = "flex")]
#[test]
fn fp32_on_flex_matches_torch() {
    let _serial = serial();
    check_f32_cases(&Device::flex().autodiff(), "flex");
}

#[cfg(feature = "flex")]
#[test]
fn bf16_on_flex_matches_torch() {
    let _serial = serial();
    check_bf16_case(&Device::flex().autodiff(), "flex-bf16", BF16_TOL);
}

/// Without autodiff, and through Autodiff without a gradient, the output and the final state are
/// those of the gradient run.
#[cfg(feature = "flex")]
#[test]
fn forward_without_gradients_on_flex() {
    let _serial = serial();
    let case = Case::load("f32_l2norm_state");
    for device in [Device::flex(), Device::flex().autodiff()] {
        let t4 = |key: &str| case.tensor::<4>(key, &device);
        let t3 = |key: &str| case.tensor::<3>(key, &device);
        let options = ChunkGatedDeltaOptions::default()
            .with_qk_l2norm(true)
            .with_final_state(true);
        let (out, state) = chunk_gated_delta_rule(
            t4("q"),
            t4("k"),
            t4("v"),
            t3("g"),
            t3("beta"),
            Some(t4("h0")),
            None,
            options,
        );
        let (_, rel) = errors(&to_vec(out), case.get("out"));
        assert!(rel <= F32_TOL);
        let (_, rel) = errors(&to_vec(state.unwrap()), case.get("state"));
        assert!(rel <= F32_TOL);
    }
}

/// Seeded inputs for shapes too large to record: `(q, k, v, g, beta, h0, d_out, d_state)` as f32
/// vectors, the model's distributions (`g = -softplus(x) 0.4`, `beta = sigmoid(x)`); keys and
/// queries normalised when the l2norm is not done in the kernel.
#[cfg(feature = "cubecl")]
fn seeded(
    seed: u64,
    [b, t, h, dk, dv]: [usize; 5],
    normalise: bool,
) -> [(Vec<usize>, Vec<f32>); 8] {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut normal = move |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                let (u1, u2) = (uniform(), uniform());
                ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
            })
            .collect()
    };
    let rows = |x: Vec<f32>, d: usize| -> Vec<f32> {
        x.chunks(d)
            .flat_map(|r| {
                let n = r.iter().map(|v| v * v).sum::<f32>().sqrt();
                r.iter().map(move |v| v / n).collect::<Vec<_>>()
            })
            .collect()
    };
    let (mut q, mut k) = (normal(b * t * h * dk), normal(b * t * h * dk));
    if normalise {
        q = rows(q, dk);
        k = rows(k, dk);
    }
    let v = normal(b * t * h * dv);
    let g = normal(b * t * h)
        .into_iter()
        .map(|x| -(1.0 + (x - 0.5).exp()).ln() * 0.4)
        .collect();
    let beta = normal(b * t * h)
        .into_iter()
        .map(|x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    let h0 = normal(b * h * dk * dv)
        .into_iter()
        .map(|x| x * 0.5)
        .collect();
    [
        (vec![b, t, h, dk], q),
        (vec![b, t, h, dk], k),
        (vec![b, t, h, dv], v),
        (vec![b, t, h], g),
        (vec![b, t, h], beta),
        (vec![b, h, dk, dv], h0),
        (vec![b, t, h, dv], normal(b * t * h * dv)),
        (vec![b, h, dk, dv], normal(b * h * dk * dv)),
    ]
}

/// The same checks on a CubeCL device, any runtime: the one `DASH_OPS_DEVICE` names.
#[cfg(feature = "cubecl")]
mod cube {
    use super::*;
    use dash_ops::gated_delta::chunk::{
        chunk_gated_delta_rule_reference, chunk_gated_delta_rule_reference_backward, kernels_take,
        tensor_cores_take,
    };

    /// How many kernel launches a forward and a backward pass over `[B, T, H, Dk, Dv]` in `dtype`
    /// make on `device`: three for a forward pass and, for a backward pass, six on tensor cores
    /// and nine on the f32 kernels; none for a pass the reference takes. Under CUDA the kernels
    /// take both: cards of compute capability 8.0 and later have all they ask.
    fn expected_launches(device: &Device, dtype: DType, dims: [usize; 5]) -> usize {
        let [forward, backward] = kernels_take(device, dtype, dims, None);
        if common::runtime_of(device) == Some("cuda") {
            assert!(
                forward && backward,
                "CUDA turned the kernels away for {dims:?}"
            );
        }
        let [_, tc_backward] = tensor_cores_take(device, dtype, dims, None);
        let backward_launches = if tc_backward { 6 } else { 9 };
        3 * usize::from(forward) + backward_launches * usize::from(backward)
    }

    /// How many of those launches run on tensor cores: all of them where the kernels on tensor
    /// cores take the pass, which under CUDA they do for bf16 and f16 inputs with heads of 32, 64
    /// or 128.
    fn expected_tc_launches(device: &Device, dtype: DType, dims: [usize; 5]) -> usize {
        let [forward, backward] = tensor_cores_take(device, dtype, dims, None);
        let narrow = |d: usize| matches!(d, 32 | 64 | 128);
        if common::runtime_of(device) == Some("cuda")
            && matches!(dtype, DType::BF16 | DType::F16)
            && narrow(dims[3])
            && narrow(dims[4])
        {
            assert!(
                forward && backward,
                "CUDA turned the tensor cores away for {dims:?}"
            );
        }
        3 * usize::from(forward) + 6 * usize::from(backward)
    }

    /// Whether `device` stores `dtype`; the tests in a dtype it does not skip, saying so.
    fn has(device: &Device, dtype: DType) -> bool {
        let has = device.supports_dtype(dtype);
        if !has {
            println!("skipped: {} has no {dtype:?}", common::describe(device));
        }
        has
    }

    /// Asserts that `f` launched exactly `expected` kernels, `tc` of them on tensor cores: under
    /// fusion the kernels run when their results are read, so `f` reads them.
    fn launches_kernels<R>(expected: usize, tc: usize, f: impl FnOnce() -> R) -> R {
        let before = probe::kernel_launches();
        let before_tc = probe::tensor_core_launches();
        let result = f();
        let launched = probe::kernel_launches() - before;
        let launched_tc = probe::tensor_core_launches() - before_tc;
        assert_eq!(
            (launched, launched_tc),
            (expected, tc),
            "the kernels made {launched} launches ({launched_tc} on tensor cores), expected \
             {expected} ({tc})"
        );
        result
    }

    /// The recorded cases in `dtype` on `device`, each held to torch, and to its kernels'
    /// expected launches.
    fn check_cases_on(device: &Device, label: &str, dtype: DType) {
        let names: &[&'static str] = if dtype == DType::F32 {
            &F32_CASES
        } else {
            &["bf16_l2norm_state"]
        };
        for &name in names {
            let case = Case::load(name);
            let (q, v) = (&case.tensors["q"].0, &case.tensors["v"].0);
            let dims = [q[0], q[1], q[2], q[3], v[3]];
            let expected = expected_launches(device, dtype, dims);
            let tc = expected_tc_launches(device, dtype, dims);
            println!(
                "{name}: {dims:?} {dtype:?}, {expected} kernel launches, {tc} on tensor cores"
            );
            let ad = device.clone().autodiff();
            if dtype == DType::F32 {
                let res = launches_kernels(expected, tc, || run(&case, &ad, dtype));
                check(&case, label, &res, "", F32_TOL);
                check(&case, label, &res, "_f64", F32_TOL);
            } else {
                let tol = if tc > 0 { BF16_TC_TOL } else { BF16_TOL };
                launches_kernels(expected, tc, || check_bf16_case(&ad, label, tol));
            }
        }
    }

    fn t4(x: &(Vec<usize>, Vec<f32>), device: &Device) -> Tensor<4> {
        Tensor::from_data(TensorData::new(x.1.clone(), x.0.clone()), device)
    }

    fn t3(x: &(Vec<usize>, Vec<f32>), device: &Device) -> Tensor<3> {
        Tensor::from_data(TensorData::new(x.1.clone(), x.0.clone()), device)
    }

    /// The kernels (forward, and backward through Autodiff) against the reference (forward and
    /// written-out backward, from Burn's tensor operations) on the same device, from the same
    /// inputs rounded to `dtype`. Returns the largest relative error over the outputs and
    /// gradients, printing each.
    fn kernels_against_reference(label: &str, dims: [usize; 5], l2norm: bool, dtype: DType) -> f32 {
        kernels_against_reference_on(label, dims, l2norm, dtype).0
    }

    /// [`kernels_against_reference`], and whether the kernels on tensor cores ran.
    fn kernels_against_reference_on(
        label: &str,
        dims: [usize; 5],
        l2norm: bool,
        dtype: DType,
    ) -> (f32, bool) {
        let device = common::cube_device();
        let ad = device.clone().autodiff();
        // Where no kernel takes the shape the reference runs, as it does on lavapipe, OpenGL and
        // CubeCL's CPU runtime: there two chunks, the second partial, cover what any length
        // does, at a fraction of the reference's time.
        let dims = if kernels_take(&device, dtype, dims, None) == [false, false] {
            [dims[0], dims[1].min(80), dims[2], dims[3], dims[4]]
        } else {
            dims
        };
        let expected = expected_launches(&device, dtype, dims);
        let tc = expected_tc_launches(&device, dtype, dims);
        let launches_tc = probe::tensor_core_launches();
        let x = seeded(7, dims, !l2norm);
        let leaf4 = |i: usize| t4(&x[i], &ad).cast(dtype).require_grad();
        let leaf3 = |i: usize| t3(&x[i], &ad).cast(dtype).require_grad();
        let (q, k, v, g, beta) = (leaf4(0), leaf4(1), leaf4(2), leaf3(3), leaf3(4));
        let h0 = t4(&x[5], &ad).require_grad();
        let launches = probe::kernel_launches();
        let options = ChunkGatedDeltaOptions::default()
            .with_qk_l2norm(l2norm)
            .with_final_state(true);
        let (out, state) = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            Some(h0.clone()),
            None,
            options,
        );
        let state = state.unwrap();
        let loss = (out.clone().cast(DType::F32) * t4(&x[6], &ad)).sum()
            + (state.clone() * t4(&x[7], &ad)).sum();
        let grads = loss.backward();
        let ours = [
            to_vec(out),
            to_vec(state),
            to_vec(q.grad(&grads).unwrap()),
            to_vec(k.grad(&grads).unwrap()),
            to_vec(v.grad(&grads).unwrap()),
            to_vec(g.grad(&grads).unwrap()),
            to_vec(beta.grad(&grads).unwrap()),
            to_vec(h0.grad(&grads).unwrap()),
        ];
        // After the reads: under Burn's fusion the kernels run when their results are needed.
        assert_eq!(
            probe::kernel_launches(),
            launches + expected,
            "the kernels made other launches"
        );
        assert_eq!(
            probe::tensor_core_launches(),
            launches_tc + tc,
            "the kernels made other launches on tensor cores"
        );
        // The reference in f32 from the inputs as the kernels saw them.
        let r4 = |i: usize| t4(&x[i], &device).cast(dtype).cast(DType::F32);
        let r3 = |i: usize| t3(&x[i], &device).cast(dtype).cast(DType::F32);
        let (out, state) = chunk_gated_delta_rule_reference(
            r4(0),
            r4(1),
            r4(2),
            r3(3),
            r3(4),
            Some(t4(&x[5], &device)),
            None,
            options,
        );
        let (dq, dk, dv, dg, dbeta, dh0) = chunk_gated_delta_rule_reference_backward(
            r4(0),
            r4(1),
            r4(2),
            r3(3),
            r3(4),
            Some(t4(&x[5], &device)),
            None,
            t4(&x[6], &device),
            t4(&x[7], &device),
            options,
        );
        let theirs = [
            to_vec(out),
            to_vec(state),
            to_vec(dq),
            to_vec(dk),
            to_vec(dv),
            to_vec(dg),
            to_vec(dbeta),
            to_vec(dh0),
        ];
        let names = ["out", "state", "dq", "dk", "dv", "dg", "dbeta", "dh0"];
        let mut worst = 0f32;
        for ((name, a), e) in names.iter().zip(&ours).zip(&theirs) {
            let (abs, rel) = errors(a, e);
            println!("{label:28} {name:6} vs reference  max abs {abs:.3e} rel {rel:.3e}");
            worst = worst.max(rel);
        }
        (worst, tc > 0)
    }

    #[test]
    fn fp32_on_cubecl_matches_torch() {
        let _serial = serial();
        let device = common::cube_device();
        println!("{}", common::describe(&device));
        check_cases_on(&device, "cubecl", DType::F32);
    }

    #[test]
    fn bf16_on_cubecl_matches_torch() {
        let _serial = serial();
        let device = common::cube_device();
        if has(&device, DType::BF16) {
            check_cases_on(&device, "cubecl-bf16", DType::BF16);
        }
    }

    /// Qwen3.5-0.8B's head shapes (Dk = Dv = 128), a sequence ending mid-chunk, two batch
    /// elements: the kernels within 1e-6 of the reference in fp32 (measured at most 5.6e-7).
    #[test]
    fn kernels_match_the_reference_at_model_shapes() {
        let _serial = serial();
        let dims = [2, 300, 3, 128, 128];
        let worst = kernels_against_reference("f32 l2norm D=128", dims, true, DType::F32);
        assert!(worst <= 1e-6, "relative error {worst:.3e}");
        let worst = kernels_against_reference("f32 plain D=128", dims, false, DType::F32);
        assert!(worst <= 1e-6, "relative error {worst:.3e}");
    }

    /// The same in bf16, Qwen3.5-0.8B's head shapes and other widths the kernels on tensor cores
    /// take (`Dk` and `Dv` of 32, 64 and 128, apart and together). The f32 kernels compute in f32
    /// from the bf16 inputs and round once, to the outputs' dtype (2^-9 relative), so they stay
    /// within a bf16 rounding of the f32 reference (measured at most 3.2e-3). On tensor cores the
    /// products' operands are rounded to bf16 as well [measured, the largest of each case: the
    /// final state 7.01e-3; the queries' gradient 4.40e-3; the gate's 1.124e-2; the keys'
    /// 5.54e-3]. The gate's gradient is a difference of sums within each chunk (and takes
    /// `<S, dS>` over the chunk's bf16 states), so its error relative to its largest value is the
    /// largest, the furthest without the l2 normalisation, as fla's is.
    #[test]
    fn kernels_match_the_reference_in_bf16() {
        let _serial = serial();
        if !has(&common::cube_device(), DType::BF16) {
            return;
        }
        for (dims, l2norm, tc_tol) in [
            ([1, 200, 2, 128, 128], true, 7.5e-3),
            ([2, 150, 2, 64, 64], true, 5e-3),
            ([1, 130, 3, 64, 128], false, 1.2e-2),
            ([1, 100, 2, 128, 32], true, 6e-3),
        ] {
            let label = format!("bf16 {dims:?} l2norm={l2norm}");
            let (worst, tc) = kernels_against_reference_on(&label, dims, l2norm, DType::BF16);
            let tol = if tc { tc_tol } else { 4e-3 };
            assert!(worst <= tol, "{label}: relative error {worst:.3e}");
        }
    }

    /// And in f16 (on tensor cores, with f16 operands).
    #[test]
    fn kernels_match_the_reference_in_f16() {
        let _serial = serial();
        if !has(&common::cube_device(), DType::F16) {
            return;
        }
        let dims = [1, 200, 2, 128, 128];
        let (worst, tc) = kernels_against_reference_on("f16 l2norm D=128", dims, true, DType::F16);
        let tol = if tc {
            F16_TC_VS_REFERENCE
        } else {
            F16_VS_REFERENCE
        };
        assert!(worst <= tol, "relative error {worst:.3e}");
    }

    /// Heads 256 wide and between, on the kernels for wide heads (queries and keys taken a half
    /// at a time): within 1.5e-6 of the reference in fp32, sums twice as long as the narrow
    /// heads' [measured: 1.19e-6], within a bf16 rounding in bf16 [measured: 3.31e-3].
    #[test]
    fn kernels_match_the_reference_for_wide_heads() {
        let _serial = serial();
        for (dims, l2norm) in [
            ([2, 150, 2, 256, 256], true),
            ([1, 100, 2, 192, 80], true),
            ([1, 130, 2, 256, 64], false),
            ([1, 90, 3, 64, 256], true),
        ] {
            let label = format!("f32 {dims:?} l2norm={l2norm}");
            let worst = kernels_against_reference(&label, dims, l2norm, DType::F32);
            assert!(worst <= 1.5e-6, "{label}: relative error {worst:.3e}");
        }
        if has(&common::cube_device(), DType::BF16) {
            let dims = [1, 200, 2, 256, 256];
            let worst =
                kernels_against_reference("bf16 [1, 200, 2, 256, 256]", dims, true, DType::BF16);
            assert!(worst <= 3.5e-3, "bf16: relative error {worst:.3e}");
        }
    }

    /// Heads wider than the kernels hold run the reference, and agree with Flex.
    #[cfg(feature = "flex")]
    #[test]
    fn large_heads_fall_back_to_the_reference() {
        let _serial = serial();
        let dims = [1, 70, 2, 272, 48];
        let x = seeded(3, dims, false);
        let run = |device: &Device| {
            let (out, state) = chunk_gated_delta_rule(
                t4(&x[0], device),
                t4(&x[1], device),
                t4(&x[2], device),
                t3(&x[3], device),
                t3(&x[4], device),
                Some(t4(&x[5], device)),
                None,
                ChunkGatedDeltaOptions::default()
                    .with_qk_l2norm(true)
                    .with_final_state(true),
            );
            (to_vec(out), to_vec(state.unwrap()))
        };
        let launches = probe::kernel_launches();
        let (out, state) = run(&common::cube_device());
        assert_eq!(
            probe::kernel_launches(),
            launches,
            "large heads must not reach the kernels"
        );
        let (out_flex, state_flex) = run(&Device::flex());
        let (_, rel_out) = errors(&out, &out_flex);
        let (_, rel_state) = errors(&state, &state_flex);
        println!("large heads: out rel {rel_out:.3e}, state rel {rel_state:.3e} (vs flex)");
        assert!(rel_out <= 1e-6 && rel_state <= 1e-6);
    }
}
