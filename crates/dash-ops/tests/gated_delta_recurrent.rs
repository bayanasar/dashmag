//! The recurrent gated delta rule's step, held to transformers 5.15's
//! `torch_recurrent_gated_delta_rule`.
//!
//! The fixtures in `tests/fixtures/gated_delta_recurrent/` were recorded on the CPU with
//! transformers' torch code, in fp32 but where bf16 is named (with fla installed, the
//! module-level `torch_chunk_gated_delta_rule` routes to fla's kernel; what is recorded is the
//! torch code behind it):
//!
//! - `steps_f32`: one step from a random state, Dk 16 / Dv 24 with and without the l2 norm of
//!   `q` and `k`, and Qwen3.5-0.8B's head, Dk = Dv = 128, with it;
//! - `step_bf16`: `q`, `k`, `v` and `beta` in bf16 and `g` in f32, as Qwen3.5 computes them,
//!   with torch's own result on them and its fp32 result on the same values upcast;
//! - `step_grad`: torch autograd gradients through one step;
//! - `prefill_decode`: `torch_chunk_gated_delta_rule` over a 100-token prompt with
//!   `output_final_state=True`, then `torch_recurrent_gated_delta_rule` token by token over 16
//!   more from that state.
//!
//! Errors are reported as the largest absolute difference and as that over the largest absolute
//! reference value ("relative"), and the relative one is asserted just above the largest measured
//! on Flex and on CUDA, with and without fusion (in brackets below).
//!
//!   cargo test --release -p dash-ops --test gated_delta_recurrent
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test gated_delta_recurrent
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda,fusion --test gated_delta_recurrent

use burn::tensor::{DType, Device, Tensor, TensorData, bf16};
use dash_ops::gated_delta::recurrent::gated_delta_step;
use safetensors::{Dtype, SafeTensors};
use std::path::Path;

/// fp32 output, against torch or the reference [4.6e-7].
const F32_OUT: f32 = 8e-7;
/// fp32 new state, against torch or the reference [1.9e-7].
const F32_STATE: f32 = 4e-7;
/// bf16 inputs against torch's own bf16 path, which l2-normalises `q` and `k` in bf16 where this
/// computes in f32: output [6.9e-3], state [3.4e-3].
const BF16_VS_TORCH_BF16: [f32; 2] = [8e-3, 4e-3];
/// bf16 inputs against torch in fp32 on the same values upcast: the output differs by its
/// rounding to bf16, at most half a bf16 ulp, 2^-8 [2.9e-3]; the state [1.1e-7].
const BF16_VS_UPCAST: [f32; 2] = [4e-3, 4e-7];
/// Gradients through the reference against torch autograd [1.3e-6].
const GRAD: f32 = 2e-6;

/// One recorded safetensors file.
struct Fixture(Vec<u8>);

impl Fixture {
    fn open(file: &str) -> Self {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gated_delta_recurrent")
            .join(file);
        Self(std::fs::read(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display())))
    }

    /// A recorded tensor in its recorded dtype (f32 or bf16).
    fn data(&self, name: &str) -> TensorData {
        let file = SafeTensors::deserialize(&self.0).expect("a safetensors file");
        let view = file
            .tensor(name)
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        let shape = view.shape().to_vec();
        match view.dtype() {
            Dtype::F32 => TensorData::new(
                view.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>(),
                shape,
            ),
            Dtype::BF16 => TensorData::new(
                view.data()
                    .chunks_exact(2)
                    .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
                    .collect::<Vec<_>>(),
                shape,
            ),
            other => panic!("{name}: unexpected dtype {other:?}"),
        }
    }

    /// A recorded tensor on `device`, in its recorded dtype.
    fn tensor<const D: usize>(&self, name: &str, device: &Device) -> Tensor<D> {
        let data = self.data(name);
        let dtype = data.dtype();
        Tensor::from_data(data, (device, dtype))
    }
}

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

/// Prints the errors of `actual` against `expected` and asserts the relative one.
fn assert_close<const D: usize>(
    what: &str,
    actual: Tensor<D>,
    expected: &TensorData,
    rel_tol: f32,
) {
    let actual = actual.into_data();
    let (abs, rel) = errors(&actual, expected);
    println!("{what}: max abs {abs:.3e}, relative {rel:.3e} (tolerance {rel_tol:.0e})");
    assert!(
        rel <= rel_tol,
        "{what}: relative error {rel:.3e} above {rel_tol:.0e}"
    );
}

/// Runs one recorded fp32 step on `device` and holds output and state to torch.
fn check_f32_step(fixture: &Fixture, case: &str, use_qk_l2norm: bool, device: &Device) {
    let name = |tensor: &str| format!("{case}.{tensor}");
    let (out, state) = gated_delta_step(
        fixture.tensor(&name("q"), device),
        fixture.tensor(&name("k"), device),
        fixture.tensor(&name("v"), device),
        fixture.tensor(&name("g"), device),
        fixture.tensor(&name("beta"), device),
        fixture.tensor(&name("state"), device),
        use_qk_l2norm,
    );
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(state.dtype(), DType::F32);
    assert_close(&name("out"), out, &fixture.data(&name("out")), F32_OUT);
    assert_close(
        &name("new_state"),
        state,
        &fixture.data(&name("new_state")),
        F32_STATE,
    );
}

fn check_f32_steps(device: &Device) {
    let fixture = Fixture::open("steps_f32.safetensors");
    check_f32_step(&fixture, "step_small", false, device);
    check_f32_step(&fixture, "step_small_l2", true, device);
    check_f32_step(&fixture, "step_head128", true, device);
}

/// Continues token by token from the state torch's chunked rule left after the prompt, slicing
/// each token's inputs out of the recorded `[B, T, H, D]` tensors on the device, and threading
/// the state through by value.
fn check_prefill_then_decode(device: &Device) {
    let fixture = Fixture::open("prefill_decode.safetensors");
    let name = |tensor: &str| format!("prefill_decode.{tensor}");
    let [q, k, v]: [Tensor<4>; 3] = ["q", "k", "v"].map(|n| fixture.tensor(&name(n), device));
    let [g, beta]: [Tensor<3>; 2] = ["g", "beta"].map(|n| fixture.tensor(&name(n), device));
    let [b, tokens, h, dk] = q.dims();
    let dv = v.dims()[3];
    let mut state: Tensor<4> = fixture.tensor(&name("prefill_state"), device);
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
            true,
        );
        outs.push(out);
        state = next;
    }
    assert_close(
        &name("out"),
        Tensor::stack::<4>(outs, 1),
        &fixture.data(&name("out")),
        F32_OUT,
    );
    assert_close(
        &name("final_state"),
        state,
        &fixture.data(&name("final_state")),
        F32_STATE,
    );
}

/// bf16 `q`, `k`, `v`, `beta` and f32 `g`: the output comes back in bf16. Held to torch's own
/// bf16 path (which l2-normalises in bf16, where this computes in f32) and to torch in fp32 on
/// the same values upcast (the arithmetic this step does, up to the output's rounding to bf16).
fn check_bf16_step(device: &Device) {
    let fixture = Fixture::open("step_bf16.safetensors");
    let name = |tensor: &str| format!("step_bf16.{tensor}");
    let q: Tensor<3> = fixture.tensor(&name("q"), device);
    assert_eq!(q.dtype(), DType::BF16);
    let (out, state) = gated_delta_step(
        q,
        fixture.tensor(&name("k"), device),
        fixture.tensor(&name("v"), device),
        fixture.tensor(&name("g"), device),
        fixture.tensor(&name("beta"), device),
        fixture.tensor(&name("state"), device),
        true,
    );
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(state.dtype(), DType::F32);
    let out = out.into_data();
    let state = state.into_data();
    for (what, data, tol) in [
        ("out_torch_bf16", &out, BF16_VS_TORCH_BF16[0]),
        ("new_state_torch_bf16", &state, BF16_VS_TORCH_BF16[1]),
        ("out_f32_upcast", &out, BF16_VS_UPCAST[0]),
        ("new_state_f32_upcast", &state, BF16_VS_UPCAST[1]),
    ] {
        let (abs, rel) = errors(data, &fixture.data(&name(what)));
        println!(
            "step_bf16 against {what}: max abs {abs:.3e}, relative {rel:.3e} (tolerance {tol:.0e})"
        );
        assert!(
            rel <= tol,
            "step_bf16 against {what}: relative error {rel:.3e} above {tol:.0e}"
        );
    }
}

/// With tracked inputs the step runs the reference on the autodiff backend: its gradients are
/// torch's.
fn check_gradients(device: &Device) {
    let fixture = Fixture::open("step_grad.safetensors");
    let name = |tensor: &str| format!("step_grad.{tensor}");
    let [q, k, v]: [Tensor<3>; 3] =
        ["q", "k", "v"].map(|n| fixture.tensor(&name(n), device).require_grad());
    let [g, beta]: [Tensor<2>; 2] =
        ["g", "beta"].map(|n| fixture.tensor(&name(n), device).require_grad());
    let state: Tensor<4> = fixture.tensor(&name("state"), device).require_grad();
    let (out, new_state) = gated_delta_step(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        state.clone(),
        true,
    );
    let w_out: Tensor<3> = fixture.tensor(&name("w_out"), device);
    let w_state: Tensor<4> = fixture.tensor(&name("w_state"), device);
    let loss = (out.clone() * w_out).sum() + (new_state.clone() * w_state).sum();
    let grads = loss.backward();
    assert_close(
        "step_grad out",
        out.detach(),
        &fixture.data(&name("out")),
        F32_OUT,
    );
    assert_close(
        "step_grad new_state",
        new_state.detach(),
        &fixture.data(&name("new_state")),
        F32_STATE,
    );
    fn check<const D: usize>(fixture: &Fixture, input: &str, grad: Tensor<D>) {
        let expected = fixture.data(&format!("step_grad.grad_{input}"));
        assert_close(&format!("step_grad grad_{input}"), grad, &expected, GRAD);
    }
    for (input, x) in [("q", q), ("k", k), ("v", v)] {
        check(&fixture, input, x.grad(&grads).unwrap());
    }
    for (input, x) in [("g", g), ("beta", beta)] {
        check(&fixture, input, x.grad(&grads).unwrap());
    }
    check(&fixture, "state", state.grad(&grads).unwrap());
}

/// On an autodiff device a step whose inputs are not tracked gives untracked outputs equal to
/// the plain device's.
fn check_untracked_autodiff(device: &Device) {
    let fixture = Fixture::open("steps_f32.safetensors");
    let run = |device: &Device| {
        let name = |tensor: &str| format!("step_small_l2.{tensor}");
        gated_delta_step(
            fixture.tensor(&name("q"), device),
            fixture.tensor(&name("k"), device),
            fixture.tensor(&name("v"), device),
            fixture.tensor(&name("g"), device),
            fixture.tensor(&name("beta"), device),
            fixture.tensor(&name("state"), device),
            true,
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

#[test]
fn step_matches_torch_on_flex() {
    check_f32_steps(&Device::flex());
}

#[test]
fn prefill_with_the_chunked_rule_then_decode_on_flex() {
    check_prefill_then_decode(&Device::flex());
}

#[test]
fn bf16_inputs_on_flex() {
    check_bf16_step(&Device::flex());
}

#[test]
fn tracked_inputs_take_the_reference_and_its_gradients_on_flex() {
    check_gradients(&Device::flex().autodiff());
}

#[test]
fn untracked_inputs_on_an_autodiff_device_on_flex() {
    check_untracked_autodiff(&Device::flex());
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use dash_ops::gated_delta::recurrent::{gated_delta_step_reference, probe};
    use std::sync::{Mutex, MutexGuard};

    /// bf16 output of the kernel against the reference's: both round nearly equal f32 values to
    /// bf16 on their own, so they may differ by one bf16 ulp, 2^-7 [9.3e-4].
    const BF16_KERNEL_VS_REFERENCE: f32 = 8e-3;

    fn device() -> Device {
        Device::cuda(0)
    }

    /// The kernel counter is shared by the whole process: the CUDA tests take turns, so that
    /// one test's launches are not counted, or found, in another's.
    fn serial() -> MutexGuard<'static, ()> {
        static TURN: Mutex<()> = Mutex::new(());
        TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Asserts that `f` launched the step's kernel at least `n` times. Under fusion the kernel
    /// runs when its results are read, so `f` reads them.
    fn launches_kernel<R>(n: usize, f: impl FnOnce() -> R) -> R {
        let before = probe::kernel_launches();
        let result = f();
        let launched = probe::kernel_launches() - before;
        assert!(
            launched >= n,
            "the CUDA kernel ran {launched} times, expected {n}"
        );
        result
    }

    #[test]
    fn step_matches_torch_on_cuda() {
        let _turn = serial();
        launches_kernel(3, || check_f32_steps(&device()));
    }

    #[test]
    fn prefill_with_the_chunked_rule_then_decode_on_cuda() {
        let _turn = serial();
        launches_kernel(16, || check_prefill_then_decode(&device()));
    }

    #[test]
    fn bf16_inputs_on_cuda() {
        let _turn = serial();
        launches_kernel(1, || check_bf16_step(&device()));
    }

    #[test]
    fn tracked_inputs_take_the_reference_and_its_gradients_on_cuda() {
        let _turn = serial();
        let before = probe::kernel_launches();
        check_gradients(&device().autodiff());
        assert_eq!(
            probe::kernel_launches(),
            before,
            "a tracked step ran the kernel"
        );
    }

    #[test]
    fn untracked_inputs_on_an_autodiff_device_run_the_kernel() {
        let _turn = serial();
        launches_kernel(2, || check_untracked_autodiff(&device()));
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
        let (ref_out, ref_state) = gated_delta_step_reference(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            state.clone(),
            use_qk_l2norm,
        );
        let (out, new_state) = launches_kernel(1, || {
            let (out, new_state) = gated_delta_step(q, k, v, g, beta, state, use_qk_l2norm);
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
        ] {
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
            true,
        );
        let before = probe::kernel_launches();
        let (out, new_state) = gated_delta_step(q, k, v, g, beta, state, true);
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
            true,
        );
        let qkv = Tensor::cat(vec![q, k, v], 2);
        let g_t = g.swap_dims(0, 1).into_data();
        let beta_t = beta.swap_dims(0, 1).into_data();
        let g = Tensor::<2>::from_data(g_t, &device).swap_dims(0, 1);
        let beta = Tensor::<2>::from_data(beta_t, &device).swap_dims(0, 1);
        let (out, new_state) = launches_kernel(1, || {
            let (out, new_state) = gated_delta_step(
                qkv.clone().narrow(2, 0, dk),
                qkv.clone().narrow(2, dk, dk),
                qkv.narrow(2, 2 * dk, dv),
                g,
                beta,
                state,
                true,
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
            true,
        );
        let poison = Tensor::<4>::full([1, 1, dk, dv], f32::NAN, &device);
        let state = Tensor::cat(vec![state, poison], 1).narrow(1, 0, h);
        let (out, new_state) = launches_kernel(1, || {
            let (out, new_state) = gated_delta_step(q, k, v, g, beta, state, true);
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
        let new_state = launches_kernel(1, || {
            gated_delta_step(q, k, v, g, beta, state.clone(), true)
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
