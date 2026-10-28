//! The recurrent gated delta rule's step, held to transformers 5.15's
//! `torch_recurrent_gated_delta_rule`: `dash_conformance::gated_delta_recurrent`'s tests (single
//! steps in f32 and from bf16 inputs, decoding from the chunked rule's state after a prompt, the
//! gradients of a tracked step) through the Burn adapter, on Flex and on a CubeCL device, any
//! runtime (`tests/common`), the kernel where the device has what it asks and the reference
//! elsewhere; on the CubeCL device each held to the kernel's launches.
//!
//! What is dash-ops' own stays here: decoding from the chunked rule's state with each token's
//! inputs sliced out of the recorded sequences on the device, an untracked step on an autodiff
//! device, and the kernel against the reference at decoding sizes, with ragged and wider heads,
//! strided inputs, a state inside a larger buffer and a state the caller still holds.
//!
//!   cargo test --release -p dash-ops --test gated_delta_recurrent
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test gated_delta_recurrent
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda,fusion --test gated_delta_recurrent
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test gated_delta_recurrent

mod common;

use burn::tensor::{Device, Tensor};
use common::adapter::{self, Burn};
use dash_conformance::gated_delta_recurrent::{
    F32_OUT, F32_STATE, Fixture, assert_close, check_bf16_step, check_f32_steps, check_gradients,
    check_prefill_then_decode,
};
use dash_ops::gated_delta::recurrent::{GatedDeltaStepOptions, gated_delta_step};

/// Continues token by token from the state torch's chunked rule left after the prompt, slicing
/// each token's inputs out of the recorded `[B, T, H, D]` tensors on the device (views of a batch
/// of two whose rows are a whole sequence apart), and threading the state through by value.
fn check_prefill_then_decode_from_views(device: &Device) {
    let fixture = Fixture::open("prefill_decode.safetensors");
    let name = |tensor: &str| format!("prefill_decode.{tensor}");
    let load = |n: &str| fixture.data(&name(n));
    let [q, k, v]: [Tensor<4>; 3] = ["q", "k", "v"].map(|n| adapter::tensor(&load(n), device));
    let [g, beta]: [Tensor<3>; 2] = ["g", "beta"].map(|n| adapter::tensor(&load(n), device));
    let [b, tokens, h, dk] = q.dims();
    let dv = v.dims()[3];
    let mut state: Tensor<4> = adapter::tensor(&load("prefill_state"), device);
    let mut outs = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let token = |x: &Tensor<4>, d: usize| x.clone().narrow(1, t, 1).reshape([b, h, d]);
        let (out, next) = gated_delta_step(
            token(&q, dk),
            token(&k, dk),
            token(&v, dv),
            g.clone().narrow(1, t, 1).reshape([b, h]),
            beta.clone().narrow(1, t, 1).reshape([b, h]),
            state,
            None,
            GatedDeltaStepOptions::default().with_qk_l2norm(true),
        );
        outs.push(out);
        state = next;
    }
    assert_close(
        &name("out"),
        &adapter::host(Tensor::stack::<4>(outs, 1)),
        &load("out"),
        F32_OUT,
    );
    assert_close(
        &name("final_state"),
        &adapter::host(state),
        &load("final_state"),
        F32_STATE,
    );
}

/// On an autodiff device a step whose inputs are not tracked gives untracked outputs equal to
/// the plain device's.
fn check_untracked_autodiff(device: &Device) {
    let fixture = Fixture::open("steps_f32.safetensors");
    let run = |device: &Device| {
        let name = |tensor: &str| format!("step_small_l2.{tensor}");
        let load3 = |tensor: &str| adapter::tensor::<3>(&fixture.data(&name(tensor)), device);
        let load2 = |tensor: &str| adapter::tensor::<2>(&fixture.data(&name(tensor)), device);
        gated_delta_step(
            load3("q"),
            load3("k"),
            load3("v"),
            load2("g"),
            load2("beta"),
            adapter::tensor::<4>(&fixture.data(&name("state")), device),
            None,
            GatedDeltaStepOptions::default().with_qk_l2norm(true),
        )
    };
    let (out, state) = run(&device.clone().autodiff());
    assert!(
        !out.is_tracked() && !state.is_tracked(),
        "an untracked step recorded a graph"
    );
    let (plain_out, plain_state) = run(device);
    assert_eq!(out.into_data(), plain_out.into_data());
    assert_eq!(state.into_data(), plain_state.into_data());
}

#[cfg(feature = "flex")]
#[test]
fn step_matches_torch_on_flex() {
    check_f32_steps(&Burn::new(Device::flex())).all_ran();
}

#[cfg(feature = "flex")]
#[test]
fn prefill_with_the_chunked_rule_then_decode_on_flex() {
    check_prefill_then_decode(&Burn::new(Device::flex())).all_ran();
}

#[cfg(feature = "flex")]
#[test]
fn prefill_with_the_chunked_rule_then_decode_from_views_on_flex() {
    check_prefill_then_decode_from_views(&Device::flex());
}

#[cfg(feature = "flex")]
#[test]
fn bf16_inputs_on_flex() {
    check_bf16_step(&Burn::new(Device::flex())).all_ran();
}

/// With tracked inputs the step runs the reference on the autodiff backend: its gradients are
/// torch's.
#[cfg(feature = "flex")]
#[test]
fn tracked_inputs_take_the_reference_and_its_gradients_on_flex() {
    check_gradients(&Burn::new(Device::flex().autodiff())).all_ran();
}

#[cfg(feature = "flex")]
#[test]
fn untracked_inputs_on_an_autodiff_device_on_flex() {
    check_untracked_autodiff(&Device::flex());
}

/// The same checks on a CubeCL device, any runtime: the one `DASH_OPS_DEVICE` names.
#[cfg(feature = "cubecl")]
mod cube {
    use super::*;
    use burn::tensor::{DType, TensorData};
    use dash_ops::gated_delta::recurrent::{gated_delta_step_reference, kernel_takes, probe};
    use std::sync::{Mutex, MutexGuard};

    /// The largest absolute difference, and that over the largest absolute expected value.
    fn errors(actual: &TensorData, expected: &TensorData) -> (f32, f32) {
        assert_eq!(actual.shape(), expected.shape(), "shapes differ");
        let (mut max_abs, mut max_ref) = (0f32, 0f32);
        for (a, e) in actual.iter::<f32>().zip(expected.iter::<f32>()) {
            assert!(a.is_finite(), "non-finite output {a}");
            max_abs = max_abs.max((a - e).abs());
            max_ref = max_ref.max(e.abs());
        }
        (max_abs, max_abs / max_ref.max(f32::MIN_POSITIVE))
    }

    /// bf16 output of the kernel against the reference's: both round nearly equal f32 values to
    /// bf16 on their own, so they may differ by one bf16 ulp, 2^-7 [9.3e-4].
    const BF16_KERNEL_VS_REFERENCE: f32 = 8e-3;
    /// The same in f16: one f16 ulp, 2^-10 [4.7e-4].
    const F16_KERNEL_VS_REFERENCE: f32 = 1e-3;

    fn device() -> Device {
        common::cube_device()
    }

    /// The kernel counter is shared by the whole process: the tests take turns, so that one
    /// test's launches are not counted, or found, in another's.
    fn serial() -> MutexGuard<'static, ()> {
        static TURN: Mutex<()> = Mutex::new(());
        TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Whether the kernel takes steps of shape `[B, H, Dk, Dv]` with `q`, `k`, `v` and `beta`
    /// in `dtype` and `g` in f32 on `device`, rather than the reference. Under CUDA it takes
    /// every head it holds: every card has all it asks.
    fn takes(device: &Device, dtype: DType, dims: [usize; 4]) -> bool {
        let takes = kernel_takes(device, [dtype, DType::F32, dtype], dims, None);
        if common::runtime_of(device) == Some("cuda") && dims[2] <= 256 {
            assert!(takes, "CUDA turned the kernel away for {dims:?}");
        }
        takes
    }

    /// Whether `device` stores `dtype`; the tests in a dtype it does not skip, saying so.
    fn has(device: &Device, dtype: DType) -> bool {
        let has = device.supports_dtype(dtype);
        if !has {
            println!("skipped: {} has no {dtype:?}", common::describe(device));
        }
        has
    }

    /// Asserts that `f` launched the step's kernel at least `n` times when `kernel` (the kernel
    /// takes the steps on this device), and never otherwise. Under fusion the kernel runs when
    /// its results are read, so `f` reads them.
    fn launches_kernel<R>(kernel: bool, n: usize, f: impl FnOnce() -> R) -> R {
        let before = probe::kernel_launches();
        let result = f();
        let launched = probe::kernel_launches() - before;
        if kernel {
            assert!(
                launched >= n,
                "the kernel ran {launched} times, expected {n}"
            );
        } else {
            assert_eq!(
                launched, 0,
                "the kernel ran where the reference should have"
            );
        }
        result
    }

    #[test]
    fn step_matches_torch_on_cubecl() {
        let _turn = serial();
        let device = device();
        println!("{}", common::describe(&device));
        // Dk 16 / Dv 24 twice, and Dk = Dv = 128.
        let kernel = [[1, 2, 16, 24], [1, 2, 128, 128]]
            .iter()
            .all(|&dims| takes(&device, DType::F32, dims));
        launches_kernel(kernel, 3, || {
            check_f32_steps(&Burn::new(device.clone())).all_ran()
        });
    }

    #[test]
    fn prefill_with_the_chunked_rule_then_decode_on_cubecl() {
        let _turn = serial();
        let device = device();
        let kernel = takes(&device, DType::F32, [1, 2, 128, 128]);
        launches_kernel(kernel, 16, || {
            check_prefill_then_decode(&Burn::new(device.clone())).all_ran()
        });
    }

    #[test]
    fn prefill_with_the_chunked_rule_then_decode_from_views_on_cubecl() {
        let _turn = serial();
        let device = device();
        let kernel = takes(&device, DType::F32, [1, 2, 128, 128]);
        launches_kernel(kernel, 16, || check_prefill_then_decode_from_views(&device));
    }

    #[test]
    fn bf16_inputs_on_cubecl() {
        let _turn = serial();
        let device = device();
        if has(&device, DType::BF16) {
            let kernel = takes(&device, DType::BF16, [1, 2, 128, 128]);
            launches_kernel(kernel, 1, || {
                check_bf16_step(&Burn::new(device.clone())).all_ran()
            });
        }
    }

    #[test]
    fn tracked_inputs_take_the_reference_and_its_gradients_on_cubecl() {
        let _turn = serial();
        let before = probe::kernel_launches();
        check_gradients(&Burn::new(device().autodiff())).all_ran();
        assert_eq!(
            probe::kernel_launches(),
            before,
            "a tracked step ran the kernel"
        );
    }

    #[test]
    fn untracked_inputs_on_an_autodiff_device_run_the_kernel() {
        let _turn = serial();
        let device = device();
        let kernel = takes(&device, DType::F32, [1, 2, 16, 24]);
        launches_kernel(kernel, 2, || check_untracked_autodiff(&device));
    }

    /// Deterministic standard normal values (xorshift and Box-Muller), the same on every device.
    fn randn(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut uniform = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        };
        (0..n)
            .map(|_| {
                let (u, v) = (uniform(), uniform());
                scale * (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
            })
            .collect()
    }

    struct Inputs {
        q: Tensor<3>,
        k: Tensor<3>,
        v: Tensor<3>,
        g: Tensor<2>,
        beta: Tensor<2>,
        state: Tensor<4>,
    }

    fn tensor<const D: usize>(
        values: Vec<f32>,
        dims: [usize; D],
        dtype: DType,
        device: &Device,
    ) -> Tensor<D> {
        let tensor: Tensor<D> =
            Tensor::from_data(TensorData::new(values, dims), (device, DType::F32));
        tensor.cast(dtype)
    }

    /// Random inputs of the given shape: `q`, `k`, `v` and `beta` in `dtype`, `g` and the state
    /// in f32.
    fn inputs(shape: [usize; 4], dtype: DType, seed: u64, device: &Device) -> Inputs {
        let [b, h, dk, dv] = shape;
        let g: Vec<f32> = randn(b * h, seed + 3, 1.5)
            .into_iter()
            .map(|x| -(1.0 + x.exp()).ln())
            .collect();
        let beta: Vec<f32> = randn(b * h, seed + 4, 1.0)
            .into_iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Inputs {
            q: tensor(randn(b * h * dk, seed, 1.0), [b, h, dk], dtype, device),
            k: tensor(randn(b * h * dk, seed + 1, 1.0), [b, h, dk], dtype, device),
            v: tensor(randn(b * h * dv, seed + 2, 1.0), [b, h, dv], dtype, device),
            g: tensor(g, [b, h], DType::F32, device),
            beta: tensor(beta, [b, h], dtype, device),
            state: tensor(
                randn(b * h * dk * dv, seed + 5, 0.1),
                [b, h, dk, dv],
                DType::F32,
                device,
            ),
        }
    }

    fn kernel_against_reference(inputs: Inputs, use_qk_l2norm: bool, what: &str, tol: [f32; 2]) {
        let Inputs {
            q,
            k,
            v,
            g,
            beta,
            state,
        } = inputs;
        let [b, h, dk] = q.dims();
        let kernel = takes(&q.device(), q.dtype(), [b, h, dk, v.dims()[2]]);
        let (ref_out, ref_state) = gated_delta_step_reference(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            state.clone(),
            None,
            GatedDeltaStepOptions::default().with_qk_l2norm(use_qk_l2norm),
        );
        let (out, new_state) = launches_kernel(kernel, 1, || {
            let (out, new_state) = gated_delta_step(
                q,
                k,
                v,
                g,
                beta,
                state,
                None,
                GatedDeltaStepOptions::default().with_qk_l2norm(use_qk_l2norm),
            );
            (out.into_data(), new_state.into_data())
        });
        let (abs, rel) = errors(&out, &ref_out.into_data());
        println!(
            "{what} out: max abs {abs:.3e}, relative {rel:.3e} (tolerance {:.0e})",
            tol[0]
        );
        assert!(rel <= tol[0], "{what} out: relative error {rel:.3e}");
        let (abs, rel) = errors(&new_state, &ref_state.into_data());
        println!(
            "{what} state: max abs {abs:.3e}, relative {rel:.3e} (tolerance {:.0e})",
            tol[1]
        );
        assert!(rel <= tol[1], "{what} state: relative error {rel:.3e}");
    }

    /// The kernel against the reference on the same device at decoding sizes: batch 8, 16
    /// heads, Dk = Dv = 128.
    #[test]
    fn kernel_matches_the_reference_at_decoding_sizes() {
        let _turn = serial();
        let device = device();
        for (dtype, use_qk_l2norm, tol) in [
            (DType::F32, true, [F32_OUT, F32_STATE]),
            (DType::F32, false, [F32_OUT, F32_STATE]),
            (DType::BF16, true, [BF16_KERNEL_VS_REFERENCE, F32_STATE]),
            (DType::F16, true, [F16_KERNEL_VS_REFERENCE, F32_STATE]),
        ] {
            if !has(&device, dtype) {
                continue;
            }
            let what = format!("B8 H16 D128 {dtype:?} l2norm={use_qk_l2norm}");
            kernel_against_reference(
                inputs([8, 16, 128, 128], dtype, 7, &device),
                use_qk_l2norm,
                &what,
                tol,
            );
        }
    }

    /// Dk not a multiple of the row groups and Dv not a multiple of a cube's columns.
    #[test]
    fn kernel_handles_ragged_head_sizes() {
        let _turn = serial();
        let device = device();
        for shape in [[2, 3, 20, 40], [1, 2, 7, 33], [3, 1, 256, 8]] {
            let what = format!("{shape:?}");
            kernel_against_reference(
                inputs(shape, DType::F32, 11, &device),
                true,
                &what,
                [F32_OUT, F32_STATE],
            );
        }
    }

    /// Heads above the kernel's limit run the reference.
    #[test]
    fn large_heads_fall_back_to_the_reference() {
        let _turn = serial();
        let device = device();
        let Inputs {
            q,
            k,
            v,
            g,
            beta,
            state,
        } = inputs([1, 2, 264, 16], DType::F32, 13, &device);
        let (ref_out, ref_state) = gated_delta_step_reference(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            state.clone(),
            None,
            GatedDeltaStepOptions::default().with_qk_l2norm(true),
        );
        let before = probe::kernel_launches();
        let (out, new_state) = gated_delta_step(
            q,
            k,
            v,
            g,
            beta,
            state,
            None,
            GatedDeltaStepOptions::default().with_qk_l2norm(true),
        );
        // Under fusion the reference above runs fused and the fallback inside the step does not.
        let (_, rel) = errors(&out.into_data(), &ref_out.into_data());
        assert!(rel <= F32_OUT, "fallback out: relative error {rel:.3e}");
        let (_, rel) = errors(&new_state.into_data(), &ref_state.into_data());
        assert!(rel <= F32_STATE, "fallback state: relative error {rel:.3e}");
        assert_eq!(
            probe::kernel_launches(),
            before,
            "the kernel ran for Dk = 264"
        );
    }

    /// `q`, `k`, `v` sliced out of one projection and `g`, `beta` transposed views: the kernel
    /// reads them through their strides.
    #[test]
    fn kernel_reads_strided_inputs() {
        let _turn = serial();
        let device = device();
        let (b, h, dk, dv) = (2, 3, 32, 48);
        let Inputs {
            q,
            k,
            v,
            g,
            beta,
            state,
        } = inputs([b, h, dk, dv], DType::F32, 17, &device);
        let (ref_out, ref_state) = gated_delta_step_reference(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            state.clone(),
            None,
            GatedDeltaStepOptions::default().with_qk_l2norm(true),
        );
        let qkv = Tensor::cat(vec![q, k, v], 2);
        let g_t = g.swap_dims(0, 1).into_data();
        let beta_t = beta.swap_dims(0, 1).into_data();
        let g = Tensor::<2>::from_data(g_t, &device).swap_dims(0, 1);
        let beta = Tensor::<2>::from_data(beta_t, &device).swap_dims(0, 1);
        let kernel = takes(&device, DType::F32, [b, h, dk, dv]);
        let (out, new_state) = launches_kernel(kernel, 1, || {
            let (out, new_state) = gated_delta_step(
                qkv.clone().narrow(2, 0, dk),
                qkv.clone().narrow(2, dk, dk),
                qkv.narrow(2, 2 * dk, dv),
                g,
                beta,
                state,
                None,
                GatedDeltaStepOptions::default().with_qk_l2norm(true),
            );
            (out.into_data(), new_state.into_data())
        });
        let (_, rel) = errors(&out, &ref_out.into_data());
        assert!(rel <= F32_OUT, "strided out: relative error {rel:.3e}");
        let (_, rel) = errors(&new_state, &ref_state.into_data());
        assert!(rel <= F32_STATE, "strided state: relative error {rel:.3e}");
    }

    /// The kernel reads nothing past the state it is given: the state is the first heads of a
    /// buffer whose next head is NaN, and Dk = 20 leaves rows past Dk in the last row group.
    #[test]
    fn kernel_reads_nothing_past_the_state() {
        let _turn = serial();
        let device = device();
        // 20 * 64 * 4 bytes per head is a multiple of the memory alignment, so the narrowed
        // state is a view of the padded buffer rather than a copy.
        let (h, dk, dv) = (3, 20, 64);
        let Inputs {
            q,
            k,
            v,
            g,
            beta,
            state,
        } = inputs([1, h, dk, dv], DType::F32, 23, &device);
        let (ref_out, ref_state) = gated_delta_step_reference(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            state.clone(),
            None,
            GatedDeltaStepOptions::default().with_qk_l2norm(true),
        );
        let poison = Tensor::<4>::full([1, 1, dk, dv], f32::NAN, &device);
        let state = Tensor::cat(vec![state, poison], 1).narrow(1, 0, h);
        let kernel = takes(&device, DType::F32, [1, h, dk, dv]);
        let (out, new_state) = launches_kernel(kernel, 1, || {
            let (out, new_state) = gated_delta_step(
                q,
                k,
                v,
                g,
                beta,
                state,
                None,
                GatedDeltaStepOptions::default().with_qk_l2norm(true),
            );
            (out.into_data(), new_state.into_data())
        });
        let (_, rel) = errors(&out, &ref_out.into_data());
        assert!(rel <= F32_OUT, "out: relative error {rel:.3e}");
        let (_, rel) = errors(&new_state, &ref_state.into_data());
        assert!(rel <= F32_STATE, "state: relative error {rel:.3e}");
    }

    /// A state the caller still holds is left as it was: the kernel writes over the state only
    /// when it was handed the only reference.
    #[test]
    fn a_shared_state_is_not_written_over() {
        let _turn = serial();
        let device = device();
        let Inputs {
            q,
            k,
            v,
            g,
            beta,
            state,
        } = inputs([2, 2, 64, 64], DType::F32, 19, &device);
        let before = state.to_data();
        let kernel = takes(&device, DType::F32, [2, 2, 64, 64]);
        let new_state = launches_kernel(kernel, 1, || {
            gated_delta_step(
                q,
                k,
                v,
                g,
                beta,
                state.clone(),
                None,
                GatedDeltaStepOptions::default().with_qk_l2norm(true),
            )
            .1
            .into_data()
        });
        assert_ne!(new_state, before);
        assert_eq!(
            state.into_data(),
            before,
            "the caller's state was written over"
        );
    }
}
