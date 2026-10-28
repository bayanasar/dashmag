//! The gated delta rule over packed variable-length sequences, with a query scale of its own, and
//! with heads 256 wide: `dash_conformance::gated_delta_packed`'s tests (the chunked rule and the
//! step against fla 0.5.2 and transformers in float64, packed against per-sequence calls, the
//! step's gradients, the default scale) through the Burn adapter, on Flex and on a CubeCL device,
//! any runtime; on the CubeCL device each case held to the kernels' expected launches.
//!
//! What is dash-ops' own stays here: packed against per-sequence calls and the packed step
//! against token-by-token steps with each sequence's or token's inputs narrowed out of the packed
//! tensors on the device; the kernels against the reference on the same device, packed and in
//! bf16, with heads 256 wide, and the step's kernel at decoding sizes.
//!
//!   cargo test --release -p dash-ops --test gated_delta_packed -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test gated_delta_packed
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test gated_delta_packed

mod common;

use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use common::adapter::Burn;
use dash_conformance::errors_f32 as errors;
use dash_conformance::gated_delta_packed::{
    self as packed, BF16_VS_F64, Fixture, PACKED_VS_SEQUENCES, Results, STEP_VS_REF, STEPS, check,
    check_chunk_bf16, check_chunk_f32, check_step_gradients, check_steps, packed_against_sequences,
};
use dash_ops::gated_delta::chunk::{self, ChunkGatedDeltaOptions, chunk_gated_delta_rule};
use dash_ops::gated_delta::recurrent::{GatedDeltaStepOptions, gated_delta_step};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

/// One test at a time, so that the counters each test reads count only that test's work.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The fixture's tensor `key` on `device`, f32.
fn tensor<const D: usize>(case: &Fixture, key: &str, device: &Device) -> Tensor<D> {
    let host = case.tensor(key);
    Tensor::from_data(TensorData::new(host.values, host.shape), device)
}

fn cu_seqlens(offsets: &[i64], device: &Device) -> Tensor<1, Int> {
    Tensor::from_data(TensorData::new(offsets.to_vec(), [offsets.len()]), device)
}

fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.cast(DType::F32)
        .into_data()
        .try_into_vec::<f32>()
        .unwrap()
}

/// The sequences' token spans.
fn spans(case: &Fixture) -> Vec<std::ops::Range<usize>> {
    case.cu_seqlens
        .windows(2)
        .map(|w| w[0] as usize..w[1] as usize)
        .collect()
}

/// The options a chunked fixture was recorded with, the final states returned.
fn chunk_options(case: &Fixture) -> ChunkGatedDeltaOptions {
    ChunkGatedDeltaOptions {
        qk_l2norm: case.flag("l2norm"),
        scale: case.scale(),
        output_final_state: true,
    }
}

/// The chunked rule over the fixture's packed sequences on `device` (an autodiff device), f32:
/// forward and backward.
fn packed_chunk(case: &Fixture, device: &Device) -> Results {
    let leaf4 = |key: &str| tensor::<4>(case, key, device).require_grad();
    let (q, k, v) = (leaf4("q"), leaf4("k"), leaf4("v"));
    let g = tensor::<3>(case, "g", device).require_grad();
    let beta = tensor::<3>(case, "beta", device).require_grad();
    let h0 = case
        .flag("initial_state")
        .then(|| tensor::<4>(case, "h0", device).require_grad());
    let runs = chunk::probe::backward_runs();
    let (out, state) = chunk_gated_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        h0.clone(),
        Some(cu_seqlens(&case.cu_seqlens, device)),
        chunk_options(case),
    );
    let state = state.expect("the final states, asked for");
    let mut loss = (out.clone() * tensor::<4>(case, "d_out", device)).sum();
    if case.flag("final_state_grad") {
        loss = loss + (state.clone() * tensor::<4>(case, "d_state", device)).sum();
    }
    let grads = loss.backward();
    assert!(
        chunk::probe::backward_runs() > runs,
        "the written-out backward did not run"
    );
    let mut res = Results::new();
    res.insert("out", to_vec(out));
    res.insert("state", to_vec(state));
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

/// The same inputs, each sequence a call of its own (a batch of one, unpacked) on its inputs
/// narrowed out of the packed tensors, views at an offset into them, the results laid out as the
/// packed call's.
fn chunk_per_sequence(case: &Fixture, device: &Device) -> Results {
    let options = chunk_options(case);
    let state_len = ["heads", "dk", "dv"]
        .map(|key| case.number(key))
        .iter()
        .product();
    let mut parts: HashMap<&'static str, Vec<Vec<f32>>> = HashMap::new();
    let mut push = |key: &'static str, x: Vec<f32>| parts.entry(key).or_default().push(x);
    for (s, span) in spans(case).into_iter().enumerate() {
        let slice4 = |key: &str| tensor::<4>(case, key, device).narrow(1, span.start, span.len());
        let slice3 = |key: &str| tensor::<3>(case, key, device).narrow(1, span.start, span.len());
        let state_of = |key: &str| tensor::<4>(case, key, device).narrow(0, s, 1);
        let h0 = case
            .flag("initial_state")
            .then(|| state_of("h0").require_grad());
        if span.is_empty() {
            // Nothing to run: the state passes through, and so does its gradient.
            let zeros = || vec![0.0; state_len];
            push("state", h0.map_or_else(zeros, to_vec));
            if case.flag("initial_state") {
                let d = if case.flag("final_state_grad") {
                    to_vec(state_of("d_state"))
                } else {
                    zeros()
                };
                push("dh0", d);
            }
            continue;
        }
        let (q, k, v) = (
            slice4("q").require_grad(),
            slice4("k").require_grad(),
            slice4("v").require_grad(),
        );
        let (g, beta) = (slice3("g").require_grad(), slice3("beta").require_grad());
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
        let state = state.unwrap();
        let mut loss = (out.clone() * slice4("d_out")).sum();
        if case.flag("final_state_grad") {
            loss = loss + (state.clone() * state_of("d_state")).sum();
        }
        let grads = loss.backward();
        push("out", to_vec(out));
        push("state", to_vec(state));
        push("dq", to_vec(q.grad(&grads).unwrap()));
        push("dk", to_vec(k.grad(&grads).unwrap()));
        push("dv", to_vec(v.grad(&grads).unwrap()));
        push("dg", to_vec(g.grad(&grads).unwrap()));
        push("dbeta", to_vec(beta.grad(&grads).unwrap()));
        if let Some(h0) = h0 {
            push("dh0", to_vec(h0.grad(&grads).unwrap()));
        }
    }
    // Per sequence, token-major tensors are [1, len, H, D]: concatenated in order they are the
    // packed tensors; states [1, H, Dk, Dv] likewise.
    parts
        .into_iter()
        .map(|(key, xs)| (key, xs.concat()))
        .collect()
}

/// Packed against per-sequence calls on one device, each sequence's inputs views of the packed
/// tensors: the largest relative difference.
fn packed_against_sequences_from_views(case: &Fixture, device: &Device, label: &str) -> f32 {
    let packed = packed_chunk(case, device);
    let apart = chunk_per_sequence(case, device);
    let mut keys: Vec<_> = packed.keys().copied().collect();
    keys.sort();
    let mut worst = 0f32;
    for key in keys {
        let (abs, rel) = errors(&packed[key], &apart[key]);
        println!(
            "{:16} {label:12} {key:6} packed vs per sequence max abs {abs:.3e} rel {rel:.3e}",
            case.name
        );
        worst = worst.max(rel);
    }
    worst
}

/// The options a step fixture was recorded with.
fn step_options(case: &Fixture) -> GatedDeltaStepOptions {
    GatedDeltaStepOptions {
        qk_l2norm: case.flag("l2norm"),
        scale: case.scale(),
    }
}

/// The step over the fixture's packed tokens on `device`, without gradients: output and states.
fn packed_step(case: &Fixture, device: &Device) -> Results {
    let (out, state) = gated_delta_step(
        tensor(case, "q", device),
        tensor(case, "k", device),
        tensor(case, "v", device),
        tensor(case, "g", device),
        tensor(case, "beta", device),
        tensor(case, "h0", device),
        Some(cu_seqlens(&case.cu_seqlens, device)),
        step_options(case),
    );
    assert_eq!(state.dtype(), DType::F32);
    Results::from([("out", to_vec(out)), ("state", to_vec(state))])
}

/// The step token by token, each sequence apart, each token's inputs and each sequence's state
/// narrowed out of the packed tensors: what the packed step must give.
fn step_per_token(case: &Fixture, device: &Device) -> Results {
    let options = step_options(case);
    let row3 = |key: &str, t: usize| tensor::<3>(case, key, device).narrow(0, t, 1);
    let row2 = |key: &str, t: usize| tensor::<2>(case, key, device).narrow(0, t, 1);
    let (mut outs, mut states) = (Vec::new(), Vec::new());
    for (s, span) in spans(case).into_iter().enumerate() {
        let mut state = tensor::<4>(case, "h0", device).narrow(0, s, 1);
        for t in span {
            let (out, next) = gated_delta_step(
                row3("q", t),
                row3("k", t),
                row3("v", t),
                row2("g", t),
                row2("beta", t),
                state,
                None,
                options,
            );
            outs.push(to_vec(out));
            state = next;
        }
        states.push(to_vec(state));
    }
    Results::from([("out", outs.concat()), ("state", states.concat())])
}

/// The packed step against transformers in float64 and fla, and against itself token by token
/// from views of the packed tensors (to the bit).
fn check_steps_from_views(device: &Device, label: &str) {
    for name in STEPS {
        let case = Fixture::load(name);
        let res = packed_step(&case, device);
        check(&case, label, &res, "_f64", STEP_VS_REF);
        check(&case, label, &res, "_fla", STEP_VS_REF);
        let apart = step_per_token(&case, device);
        for key in ["out", "state"] {
            let (abs, rel) = errors(&res[key], &apart[key]);
            println!(
                "{name:16} {label:12} {key:6} packed vs token by token max abs {abs:.3e} rel {rel:.3e}"
            );
            assert!(rel <= PACKED_VS_SEQUENCES, "{name} {key}: {rel:.3e}");
        }
    }
}

#[cfg(feature = "flex")]
mod flex {
    use super::*;

    #[test]
    fn packed_chunks_on_flex_match_fla_and_transformers() {
        let _serial = serial();
        check_chunk_f32(&Burn::new(Device::flex().autodiff()), "flex").all_ran();
    }

    #[test]
    fn packed_bf16_chunks_on_flex() {
        let _serial = serial();
        let flex = Burn::new(Device::flex().autodiff());
        check_chunk_bf16(&flex, "flex-bf16", BF16_VS_F64).all_ran();
    }

    #[test]
    fn packed_chunks_on_flex_are_per_sequence_calls() {
        let _serial = serial();
        let flex = Burn::new(Device::flex().autodiff());
        for name in ["chunk_f32", "chunk_f32_scale"] {
            let worst = packed_against_sequences(&flex, &Fixture::load(name), "flex").all_ran();
            assert!(worst <= PACKED_VS_SEQUENCES, "{name}: {worst:.3e}");
        }
    }

    #[test]
    fn packed_chunks_on_flex_from_views_are_per_sequence_calls() {
        let _serial = serial();
        let device = Device::flex().autodiff();
        for name in ["chunk_f32", "chunk_f32_scale"] {
            let case = Fixture::load(name);
            let worst = packed_against_sequences_from_views(&case, &device, "flex");
            assert!(worst <= PACKED_VS_SEQUENCES, "{name}: {worst:.3e}");
        }
    }

    #[test]
    fn packed_steps_on_flex_match_fla_and_transformers() {
        let _serial = serial();
        check_steps(&Burn::new(Device::flex()), "flex").all_ran();
    }

    #[test]
    fn packed_steps_on_flex_from_views_match_fla_and_transformers() {
        let _serial = serial();
        check_steps_from_views(&Device::flex(), "flex");
    }

    #[test]
    fn tracked_packed_steps_on_flex_take_the_reference_gradients() {
        let _serial = serial();
        check_step_gradients(&Burn::new(Device::flex()), "flex").all_ran();
    }

    #[test]
    fn the_default_scale_is_one_over_the_root_of_dk() {
        let _serial = serial();
        packed::the_default_scale_is_one_over_the_root_of_dk(&Burn::new(Device::flex())).all_ran();
    }
}

/// Seeded normal values (xorshift and Box-Muller), the same on every device.
#[cfg(feature = "cubecl")]
fn normal(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..len)
        .map(|_| {
            let (u1, u2) = (uniform(), uniform());
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

/// The same checks on a CubeCL device, any runtime: the one `DASH_OPS_DEVICE` names.
#[cfg(feature = "cubecl")]
mod cube {
    use super::*;
    use dash_conformance::gated_delta_packed::{
        BF16_TC_VS_F64, CHUNK_F32, F32_VS_F64, F32_VS_FLA, run_chunk, run_step,
    };
    use dash_ops::gated_delta::chunk::{
        chunk_gated_delta_rule_reference, chunk_gated_delta_rule_reference_backward, kernels_take,
        tensor_cores_take,
    };
    use dash_ops::gated_delta::recurrent::{self, gated_delta_step_reference, kernel_takes};

    /// The kernel launches a packed forward and backward pass over `[1, T, H, Dk, Dv]` in
    /// `dtype` make on `device`: four forward and, backward, seven on tensor cores or ten on the
    /// f32 kernels where the kernels take them (the chunks' layout first in each), none where the
    /// reference does. Under CUDA the kernels take heads up to their width.
    fn expected_launches(device: &Device, dtype: DType, dims: [usize; 5], seqs: usize) -> usize {
        let int = cu_seqlens(&[0, 1], device).dtype();
        let [forward, backward] = kernels_take(device, dtype, dims, Some((seqs, int)));
        if common::runtime_of(device) == Some("cuda")
            && dims[3].max(dims[4]) <= chunk::MAX_KERNEL_HEAD_DIM
        {
            assert!(
                forward && backward,
                "CUDA turned the kernels away for {dims:?}"
            );
        }
        let [_, tc_backward] = tensor_cores_take(device, dtype, dims, Some((seqs, int)));
        let backward_launches = if tc_backward { 7 } else { 10 };
        4 * usize::from(forward) + backward_launches * usize::from(backward)
    }

    /// How many of those run on tensor cores: three forward and six backward (all but the
    /// chunks' layout) where the kernels on tensor cores take them, which under CUDA they do for
    /// bf16 inputs with heads of 32, 64 or 128.
    fn expected_tc_launches(device: &Device, dtype: DType, dims: [usize; 5], seqs: usize) -> usize {
        let int = cu_seqlens(&[0, 1], device).dtype();
        let [forward, backward] = tensor_cores_take(device, dtype, dims, Some((seqs, int)));
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

    fn has(device: &Device, dtype: DType) -> bool {
        let has = device.supports_dtype(dtype);
        if !has {
            println!("skipped: {} has no {dtype:?}", common::describe(device));
        }
        has
    }

    /// Asserts that `f` launched exactly `expected` kernels, `tc` of them on tensor cores.
    fn launches<R>(expected: usize, tc: usize, f: impl FnOnce() -> R) -> R {
        let before = chunk::probe::kernel_launches();
        let before_tc = chunk::probe::tensor_core_launches();
        let result = f();
        let launched = chunk::probe::kernel_launches() - before;
        let launched_tc = chunk::probe::tensor_core_launches() - before_tc;
        assert_eq!(
            (launched, launched_tc),
            (expected, tc),
            "the kernels made {launched} launches ({launched_tc} on tensor cores), expected \
             {expected} ({tc})"
        );
        result
    }

    fn dims_of(case: &Fixture) -> ([usize; 5], usize) {
        let q = case.tensor("q").shape;
        let v = case.tensor("v").shape;
        ([q[0], q[1], q[2], q[3], v[3]], case.cu_seqlens.len() - 1)
    }

    #[test]
    fn packed_chunks_on_cubecl_match_fla_and_transformers() {
        let _serial = serial();
        let device = common::cube_device();
        println!("{}", common::describe(&device));
        for name in CHUNK_F32 {
            let case = Fixture::load(name);
            let (dims, seqs) = dims_of(&case);
            let expected = expected_launches(&device, DType::F32, dims, seqs);
            println!("{name}: {dims:?}, {seqs} sequences, {expected} kernel launches");
            let res = launches(expected, 0, || {
                let cube = Burn::new(device.clone().autodiff());
                run_chunk(&cube, &case, dash_conformance::DType::F32).expect("a run")
            });
            check(&case, "cubecl", &res, "_f64", F32_VS_F64);
            if case.flag("fla") {
                check(&case, "cubecl", &res, "_fla", F32_VS_FLA);
            }
        }
    }

    #[test]
    fn packed_bf16_chunks_on_cubecl() {
        let _serial = serial();
        let device = common::cube_device();
        if !has(&device, DType::BF16) {
            return;
        }
        let case = Fixture::load("chunk_bf16");
        let (dims, seqs) = dims_of(&case);
        let expected = expected_launches(&device, DType::BF16, dims, seqs);
        let tc = expected_tc_launches(&device, DType::BF16, dims, seqs);
        let tol = if tc > 0 { BF16_TC_VS_F64 } else { BF16_VS_F64 };
        launches(expected, tc, || {
            check_chunk_bf16(&Burn::new(device.clone().autodiff()), "cubecl-bf16", tol).all_ran()
        });
    }

    #[test]
    fn packed_chunks_on_cubecl_are_per_sequence_calls() {
        let _serial = serial();
        let cube = Burn::new(common::cube_device().autodiff());
        for name in ["chunk_f32", "chunk_f32_scale", "chunk_wide_k"] {
            let worst = packed_against_sequences(&cube, &Fixture::load(name), "cubecl").all_ran();
            assert!(worst <= PACKED_VS_SEQUENCES, "{name}: {worst:.3e}");
        }
    }

    #[test]
    fn packed_chunks_on_cubecl_from_views_are_per_sequence_calls() {
        let _serial = serial();
        let device = common::cube_device().autodiff();
        for name in ["chunk_f32", "chunk_f32_scale", "chunk_wide_k"] {
            let case = Fixture::load(name);
            let worst = packed_against_sequences_from_views(&case, &device, "cubecl");
            assert!(worst <= PACKED_VS_SEQUENCES, "{name}: {worst:.3e}");
        }
    }

    fn t4(x: &[f32], dims: [usize; 4], device: &Device) -> Tensor<4> {
        Tensor::from_data(TensorData::new(x.to_vec(), dims), device)
    }

    fn t3(x: &[f32], dims: [usize; 3], device: &Device) -> Tensor<3> {
        Tensor::from_data(TensorData::new(x.to_vec(), dims), device)
    }

    /// Packed sequences of `lens` tokens, `h` heads of `d`: the kernels (forward, and backward
    /// through Autodiff) against the reference (packed, its sequences one at a time) on the same
    /// device, from the same inputs rounded to `dtype`. Returns the largest relative error over the
    /// outputs and gradients, printing each.
    fn kernels_against_reference(lens: &[usize], h: usize, d: usize, dtype: DType) -> (f32, bool) {
        let device = common::cube_device();
        let ad = device.clone().autodiff();
        // Where no kernel takes the shape the reference runs (lavapipe, OpenGL, CubeCL's CPU
        // runtime): sequences of at most 48 tokens, boundaries still inside chunks, do there.
        let full: usize = lens.iter().sum();
        let int = cu_seqlens(&[0, 1], &device).dtype();
        let shorter: Vec<usize>;
        let lens = if kernels_take(&device, dtype, [1, full, h, d, d], Some((lens.len(), int)))
            == [false, false]
        {
            shorter = lens.iter().map(|&n| n.min(48)).collect();
            &shorter[..]
        } else {
            lens
        };
        let t: usize = lens.iter().sum();
        let seqs = lens.len();
        let mut offsets = vec![0i64];
        for &n in lens {
            offsets.push(offsets.last().unwrap() + n as i64);
        }
        let expected = expected_launches(&device, dtype, [1, t, h, d, d], seqs);
        let tc = expected_tc_launches(&device, dtype, [1, t, h, d, d], seqs);
        let x = |seed: u64, len: usize| normal(seed, len);
        let unit = |v: Vec<f32>| -> Vec<f32> {
            v.chunks(d)
                .flat_map(|r| {
                    let n = r.iter().map(|x| x * x).sum::<f32>().sqrt();
                    r.iter().map(move |x| x / n).collect::<Vec<_>>()
                })
                .collect()
        };
        let q = unit(x(1, t * h * d));
        let k = unit(x(2, t * h * d));
        let v = x(3, t * h * d);
        let g: Vec<f32> = x(4, t * h)
            .into_iter()
            .map(|y| -(1.0 + (y - 0.5).exp()).ln() * 0.4)
            .collect();
        let beta: Vec<f32> = x(5, t * h)
            .into_iter()
            .map(|y| 1.0 / (1.0 + (-y).exp()))
            .collect();
        let h0: Vec<f32> = x(6, seqs * h * d * d)
            .into_iter()
            .map(|y| y * 0.5)
            .collect();
        let d_out = x(7, t * h * d);
        let d_state = x(8, seqs * h * d * d);
        let tok4 = [1, t, h, d];
        let st4 = [seqs, h, d, d];
        let options = ChunkGatedDeltaOptions::default().with_final_state(true);

        let leaf4 = |v: &[f32]| t4(v, tok4, &ad).cast(dtype).require_grad();
        let leaf3 = |v: &[f32]| t3(v, [1, t, h], &ad).cast(dtype).require_grad();
        let (lq, lk, lv) = (leaf4(&q), leaf4(&k), leaf4(&v));
        let (lg, lb) = (leaf3(&g), leaf3(&beta));
        let lh0 = t4(&h0, st4, &ad).require_grad();
        let ours = launches(expected, tc, || {
            let (out, state) = chunk_gated_delta_rule(
                lq.clone(),
                lk.clone(),
                lv.clone(),
                lg.clone(),
                lb.clone(),
                Some(lh0.clone()),
                Some(cu_seqlens(&offsets, &ad)),
                options,
            );
            let state = state.unwrap();
            let loss = (out.clone().cast(DType::F32) * t4(&d_out, tok4, &ad)).sum()
                + (state.clone() * t4(&d_state, st4, &ad)).sum();
            let grads = loss.backward();
            [
                to_vec(out),
                to_vec(state),
                to_vec(lq.grad(&grads).unwrap()),
                to_vec(lk.grad(&grads).unwrap()),
                to_vec(lv.grad(&grads).unwrap()),
                to_vec(lg.grad(&grads).unwrap()),
                to_vec(lb.grad(&grads).unwrap()),
                to_vec(lh0.grad(&grads).unwrap()),
            ]
        });
        let r4 = |v: &[f32]| t4(v, tok4, &device).cast(dtype).cast(DType::F32);
        let r3 = |v: &[f32]| t3(v, [1, t, h], &device).cast(dtype).cast(DType::F32);
        let (out, state) = chunk_gated_delta_rule_reference(
            r4(&q),
            r4(&k),
            r4(&v),
            r3(&g),
            r3(&beta),
            Some(t4(&h0, st4, &device)),
            Some(cu_seqlens(&offsets, &device)),
            options,
        );
        let (dq, dk, dv, dg, dbeta, dh0) = chunk_gated_delta_rule_reference_backward(
            r4(&q),
            r4(&k),
            r4(&v),
            r3(&g),
            r3(&beta),
            Some(t4(&h0, st4, &device)),
            Some(cu_seqlens(&offsets, &device)),
            t4(&d_out, tok4, &device),
            t4(&d_state, st4, &device),
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
        let label = format!("{lens:?} H{h} D{d} {dtype:?}");
        let mut worst = 0f32;
        for ((name, a), e) in names.iter().zip(&ours).zip(&theirs) {
            let (abs, rel) = errors(a, e);
            println!("{label:32} {name:6} vs reference  max abs {abs:.3e} rel {rel:.3e}");
            worst = worst.max(rel);
        }
        (worst, tc > 0)
    }

    /// Qwen3.5-0.8B's head width, sequences ending inside chunks and one shorter than a chunk:
    /// the kernels within 8e-7 of the reference in f32 [measured: 6.3e-7].
    #[test]
    fn packed_kernels_match_the_reference() {
        let _serial = serial();
        let (worst, _) = kernels_against_reference(&[100, 37, 163], 3, 128, DType::F32);
        assert!(worst <= 8e-7, "relative error {worst:.3e}");
    }

    /// The same in bf16: the f32 kernels within a bf16 rounding of the f32 reference [measured:
    /// 3.07e-3], those on tensor cores, with bf16 operands, within 5e-3 [measured: 4.70e-3, the
    /// final states].
    #[test]
    fn packed_kernels_match_the_reference_in_bf16() {
        let _serial = serial();
        if !has(&common::cube_device(), DType::BF16) {
            return;
        }
        let (worst, tc) = kernels_against_reference(&[90, 110], 2, 128, DType::BF16);
        let tol = if tc { 5e-3 } else { 3.5e-3 };
        assert!(worst <= tol, "relative error {worst:.3e}");
    }

    /// Heads 256 wide, packed: the kernels where they take them, else the reference, against the
    /// reference [measured: 6.8e-7].
    #[test]
    fn wide_heads_match_the_reference() {
        let _serial = serial();
        let (worst, _) = kernels_against_reference(&[70, 30, 64], 2, 256, DType::F32);
        assert!(worst <= 8e-7, "relative error {worst:.3e}");
    }

    /// Whether the step's kernel takes the fixture's packed tokens on `device`. Under CUDA it
    /// takes every head it holds.
    fn step_takes(device: &Device, case: &Fixture) -> bool {
        let q = case.tensor("q").shape;
        let dv = case.tensor("v").shape[2];
        let int = cu_seqlens(&[0, 1], device).dtype();
        let seqs = case.cu_seqlens.len() - 1;
        let takes = kernel_takes(
            device,
            [DType::F32; 3],
            [q[0], q[1], q[2], dv],
            Some((seqs, int)),
        );
        if common::runtime_of(device) == Some("cuda") {
            assert!(takes, "CUDA turned the step's kernel away");
        }
        takes
    }

    #[test]
    fn packed_steps_on_cubecl_match_fla_and_transformers() {
        let _serial = serial();
        let device = common::cube_device();
        for name in STEPS {
            let case = Fixture::load(name);
            let kernel = step_takes(&device, &case);
            println!("{name}: the step's kernel takes it: {kernel}");
            let before = recurrent::probe::kernel_launches();
            let res = run_step(&Burn::new(device.clone()), &case).expect("a run");
            let launched = recurrent::probe::kernel_launches() - before;
            assert_eq!(launched, usize::from(kernel), "{name}: {launched} launches");
            check(&case, "cubecl", &res, "_f64", STEP_VS_REF);
            check(&case, "cubecl", &res, "_fla", STEP_VS_REF);
        }
        check_steps(&Burn::new(device), "cubecl").all_ran();
    }

    #[test]
    fn packed_steps_on_cubecl_from_views_match_fla_and_transformers() {
        let _serial = serial();
        check_steps_from_views(&common::cube_device(), "cubecl");
    }

    #[test]
    fn tracked_packed_steps_on_cubecl_take_the_reference_gradients() {
        let _serial = serial();
        let before = recurrent::probe::kernel_launches();
        check_step_gradients(&Burn::new(common::cube_device()), "cubecl").all_ran();
        assert_eq!(
            recurrent::probe::kernel_launches(),
            before,
            "a tracked step ran the kernel"
        );
    }

    /// Decoding sizes, sequences bringing 0 to 4 tokens: the step's kernel against the
    /// reference, f32 and bf16, and the scale option [measured: output 4.7e-7 in f32, 1.2e-3 in
    /// bf16 (each rounds nearly equal f32 values to bf16 on its own); states 1.9e-7].
    #[test]
    fn packed_step_kernel_matches_the_reference() {
        let _serial = serial();
        let device = common::cube_device();
        let counts = [1usize, 2, 4, 1, 0, 3, 1, 2];
        let (seqs, h, d) = (counts.len(), 16, 128);
        let t: usize = counts.iter().sum();
        let mut offsets = vec![0i64];
        for &n in &counts {
            offsets.push(offsets.last().unwrap() + n as i64);
        }
        for (dtype, options, tol) in [
            (
                DType::F32,
                GatedDeltaStepOptions::default().with_qk_l2norm(true),
                [6e-7, 3e-7],
            ),
            (
                DType::F32,
                GatedDeltaStepOptions::default().with_scale(0.5),
                [6e-7, 3e-7],
            ),
            (
                DType::BF16,
                GatedDeltaStepOptions::default().with_qk_l2norm(true),
                [1.5e-3, 3e-7],
            ),
        ] {
            if !has(&device, dtype) {
                continue;
            }
            let tok = |x: Vec<f32>, w: usize| t3(&x, [t, h, w], &device).cast(dtype);
            // Without the normalisation in the step, the caller normalises, as models do.
            let unit = |x: Vec<f32>| -> Vec<f32> {
                if options.qk_l2norm {
                    return x;
                }
                x.chunks(d)
                    .flat_map(|r| {
                        let n = r.iter().map(|y| y * y).sum::<f32>().sqrt();
                        r.iter().map(move |y| y / n).collect::<Vec<_>>()
                    })
                    .collect()
            };
            let q = tok(unit(normal(11, t * h * d)), d);
            let k = tok(unit(normal(12, t * h * d)), d);
            let v = tok(normal(13, t * h * d), d);
            let g: Vec<f32> = normal(14, t * h)
                .into_iter()
                .map(|y| -(1.0 + (1.5 * y).exp()).ln())
                .collect();
            let g = Tensor::<2>::from_data(TensorData::new(g, [t, h]), &device);
            let beta: Vec<f32> = normal(15, t * h)
                .into_iter()
                .map(|y| 1.0 / (1.0 + (-y).exp()))
                .collect();
            let beta = Tensor::<2>::from_data(TensorData::new(beta, [t, h]), &device).cast(dtype);
            let state: Vec<f32> = normal(16, seqs * h * d * d)
                .into_iter()
                .map(|y| 0.1 * y)
                .collect();
            let state = t4(&state, [seqs, h, d, d], &device);
            let kernel = kernel_takes(
                &device,
                [dtype, DType::F32, dtype],
                [t, h, d, d],
                Some((seqs, cu_seqlens(&offsets, &device).dtype())),
            );
            if common::runtime_of(&device) == Some("cuda") {
                assert!(kernel, "CUDA turned the step's kernel away");
            }
            println!("packed step {dtype:?}: the step's kernel takes it: {kernel}");
            let (ref_out, ref_state) = gated_delta_step_reference(
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                beta.clone(),
                state.clone(),
                Some(cu_seqlens(&offsets, &device)),
                options,
            );
            let before = recurrent::probe::kernel_launches();
            let (out, new_state) = gated_delta_step(
                q,
                k,
                v,
                g,
                beta,
                state,
                Some(cu_seqlens(&offsets, &device)),
                options,
            );
            let (out, new_state) = (to_vec(out), to_vec(new_state));
            assert_eq!(
                recurrent::probe::kernel_launches() - before,
                usize::from(kernel)
            );
            for (what, a, e, tol) in [
                ("out", &out, to_vec(ref_out), tol[0]),
                ("state", &new_state, to_vec(ref_state), tol[1]),
            ] {
                let (abs, rel) = errors(a, &e);
                println!(
                    "packed step {dtype:?} {options:?} {what}: max abs {abs:.3e} rel {rel:.3e}"
                );
                assert!(
                    rel <= tol,
                    "{what}: relative error {rel:.3e} above {tol:.0e}"
                );
            }
        }
    }
}
