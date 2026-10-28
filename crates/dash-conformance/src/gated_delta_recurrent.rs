//! The recurrent gated delta rule's step, held to transformers 5.15's
//! `torch_recurrent_gated_delta_rule`.
//!
//! The fixtures in `gated_delta_recurrent/` were recorded on the CPU with transformers' torch
//! code, in fp32 but where bf16 is named (with fla installed, the module-level
//! `torch_chunk_gated_delta_rule` routes to fla's kernel; what is recorded is the torch code
//! behind it):
//!
//! - `steps_f32`: one step from a random state, Dk 16 / Dv 24 with and without the l2 norm of
//!   `q` and `k`, and a head of Dk = Dv = 128 with it;
//! - `step_bf16`: `q`, `k`, `v` and `beta` in bf16 and `g` in f32, as models compute them, with
//!   torch's own result on them and its fp32 result on the same values upcast;
//! - `step_grad`: torch autograd gradients through one step;
//! - `prefill_decode`: `torch_chunk_gated_delta_rule` over a 100-token prompt with
//!   `output_final_state=True`, then `torch_recurrent_gated_delta_rule` token by token over 16
//!   more from that state.
//!
//! Errors are reported as the largest absolute difference and as that over the largest absolute
//! reference value ("relative", [`crate::errors_f32`]), and the relative one is asserted just
//! above the largest measured (in brackets below) on Flex; on CUDA with and without fusion and
//! autotune; under wgpu on lavapipe and on OpenGL over D3D12; and on CubeCL's CPU runtime.
//!
//! - [`check_f32_steps`], [`check_bf16_step`]: single steps against torch;
//! - [`check_prefill_then_decode`]: decoding from the chunked rule's state after a prompt;
//! - [`check_gradients`]: the step's gradients, where the implementation gives them.

use crate::fixtures::File;
use crate::{
    Adapter, DType, GatedDeltaStep, GatedDeltaStepCotangents, HostTensor, Outcome, Tally,
    errors_f32, ran,
};
use dash_contract::gated_delta::recurrent::GatedDeltaStepOptions;

/// fp32 output, against torch or the reference [4.7e-7].
pub const F32_OUT: f32 = 8e-7;
/// fp32 new state, against torch or the reference [2.6e-7, on lavapipe; 1.9e-7 on CUDA].
pub const F32_STATE: f32 = 4e-7;
/// bf16 inputs against torch's own bf16 path, which l2-normalises `q` and `k` in bf16 where this
/// computes in f32: output [6.9e-3], state [3.4e-3].
pub const BF16_VS_TORCH_BF16: [f32; 2] = [8e-3, 4e-3];
/// bf16 inputs against torch in fp32 on the same values upcast: the output differs by its
/// rounding to bf16, at most half a bf16 ulp, 2^-8 [2.9e-3]; the state [1.1e-7].
pub const BF16_VS_UPCAST: [f32; 2] = [4e-3, 4e-7];
/// Gradients against torch autograd [1.3e-6].
pub const GRAD: f32 = 2e-6;

/// One recorded safetensors file.
pub struct Fixture(File);

impl Fixture {
    /// Reads `file` of the step's fixtures.
    pub fn open(file: &str) -> Self {
        Self(File::read(&format!("gated_delta_recurrent/{file}")))
    }

    /// A recorded tensor in its recorded dtype (f32 or bf16).
    pub fn data(&self, name: &str) -> HostTensor {
        self.0.tensor(name)
    }
}

/// The largest absolute difference, and that over the largest absolute expected value.
fn errors(actual: &HostTensor, expected: &HostTensor) -> (f32, f32) {
    assert_eq!(actual.shape, expected.shape, "shapes differ");
    errors_f32(&actual.values, &expected.values)
}

/// Prints the errors of `actual` against `expected` and asserts the relative one, for the bodies
/// here and for tests that run the cases themselves.
pub fn assert_close(what: &str, actual: &HostTensor, expected: &HostTensor, rel_tol: f32) {
    let (abs, rel) = errors(actual, expected);
    println!("{what}: max abs {abs:.3e}, relative {rel:.3e} (tolerance {rel_tol:.0e})");
    assert!(
        rel <= rel_tol,
        "{what}: relative error {rel:.3e} above {rel_tol:.0e}"
    );
}

/// A step of the recorded case `case`'s tensors `q`, `k`, `v`, `g`, `beta` and `state`, each in
/// its recorded dtype.
fn step(fixture: &Fixture, case: &str, use_qk_l2norm: bool) -> GatedDeltaStep {
    let name = |tensor: &str| format!("{case}.{tensor}");
    GatedDeltaStep {
        q: fixture.data(&name("q")),
        k: fixture.data(&name("k")),
        v: fixture.data(&name("v")),
        g: fixture.data(&name("g")),
        beta: fixture.data(&name("beta")),
        state: fixture.data(&name("state")),
        cu_seqlens: None,
        options: GatedDeltaStepOptions::default().with_qk_l2norm(use_qk_l2norm),
        grad: None,
    }
}

/// Runs one recorded fp32 step and holds output and state to torch.
fn check_f32_step(adapter: &dyn Adapter, fixture: &Fixture, case: &str, use_qk_l2norm: bool) {
    let name = |tensor: &str| format!("{case}.{tensor}");
    let call = step(fixture, case, use_qk_l2norm);
    let Some(result) = ran(adapter, case, adapter.gated_delta_step(&call)) else {
        return;
    };
    assert_eq!(result.out.dtype, DType::F32);
    assert_eq!(result.state.dtype, DType::F32);
    assert_close(
        &name("out"),
        &result.out,
        &fixture.data(&name("out")),
        F32_OUT,
    );
    assert_close(
        &name("new_state"),
        &result.state,
        &fixture.data(&name("new_state")),
        F32_STATE,
    );
}

/// The three recorded fp32 steps.
pub fn check_f32_steps(adapter: &dyn Adapter) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let fixture = Fixture::open("steps_f32.safetensors");
    check_f32_step(adapter, &fixture, "step_small", false);
    check_f32_step(adapter, &fixture, "step_small_l2", true);
    check_f32_step(adapter, &fixture, "step_head128", true);
    tally.outcome(())
}

/// Token `t` of a `[B, T, H, ...]` tensor, as `[B, H, ...]`.
fn token(x: &HostTensor, t: usize) -> HostTensor {
    let (b, tokens) = (x.shape[0], x.shape[1]);
    let row: usize = x.shape[2..].iter().product();
    let mut values = Vec::with_capacity(b * row);
    for bi in 0..b {
        let at = (bi * tokens + t) * row;
        values.extend_from_slice(&x.values[at..at + row]);
    }
    let mut shape = vec![b];
    shape.extend_from_slice(&x.shape[2..]);
    HostTensor::new(x.dtype, shape, values)
}

/// Continues token by token from the state torch's chunked rule left after the prompt, the
/// state threaded from step to step.
pub fn check_prefill_then_decode(adapter: &dyn Adapter) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let fixture = Fixture::open("prefill_decode.safetensors");
    let name = |tensor: &str| format!("prefill_decode.{tensor}");
    let [q, k, v, g, beta] = ["q", "k", "v", "g", "beta"].map(|n| fixture.data(&name(n)));
    let [b, tokens, h, _] = q.shape[..] else {
        panic!("q is [B, T, H, Dk]")
    };
    let dv = v.shape[3];
    let mut state = fixture.data(&name("prefill_state"));
    let mut outs = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let call = GatedDeltaStep {
            q: token(&q, t),
            k: token(&k, t),
            v: token(&v, t),
            g: token(&g, t),
            beta: token(&beta, t),
            state,
            cu_seqlens: None,
            options: GatedDeltaStepOptions::default().with_qk_l2norm(true),
            grad: None,
        };
        let Some(result) = ran(adapter, "prefill_decode", adapter.gated_delta_step(&call)) else {
            return tally.outcome(());
        };
        outs.push(result.out);
        state = result.state;
    }
    // The outputs stacked along the time, `[B, T, H, Dv]`.
    let row = h * dv;
    let mut values = Vec::with_capacity(b * tokens * row);
    for bi in 0..b {
        for out in &outs {
            values.extend_from_slice(&out.values[bi * row..(bi + 1) * row]);
        }
    }
    let out = HostTensor::new(outs[0].dtype, vec![b, tokens, h, dv], values);
    assert_close(&name("out"), &out, &fixture.data(&name("out")), F32_OUT);
    assert_close(
        &name("final_state"),
        &state,
        &fixture.data(&name("final_state")),
        F32_STATE,
    );
    tally.outcome(())
}

/// bf16 `q`, `k`, `v`, `beta` and f32 `g`: the output comes back in bf16. Held to torch's own
/// bf16 path (which l2-normalises in bf16, where this computes in f32) and to torch in fp32 on
/// the same values upcast (the arithmetic this step does, up to the output's rounding to bf16).
pub fn check_bf16_step(adapter: &dyn Adapter) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let fixture = Fixture::open("step_bf16.safetensors");
    let name = |tensor: &str| format!("step_bf16.{tensor}");
    let call = step(&fixture, "step_bf16", true);
    assert_eq!(call.q.dtype, DType::BF16);
    let Some(result) = ran(adapter, "step_bf16", adapter.gated_delta_step(&call)) else {
        return tally.outcome(());
    };
    assert_eq!(result.out.dtype, DType::BF16);
    assert_eq!(result.state.dtype, DType::F32);
    for (what, data, tol) in [
        ("out_torch_bf16", &result.out, BF16_VS_TORCH_BF16[0]),
        ("new_state_torch_bf16", &result.state, BF16_VS_TORCH_BF16[1]),
        ("out_f32_upcast", &result.out, BF16_VS_UPCAST[0]),
        ("new_state_f32_upcast", &result.state, BF16_VS_UPCAST[1]),
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
    tally.outcome(())
}

/// The step's gradients, where the implementation gives them, are torch's.
pub fn check_gradients(adapter: &dyn Adapter) -> Outcome {
    let tally = Tally::default();
    let adapter = &tally.count(adapter);
    let fixture = Fixture::open("step_grad.safetensors");
    let name = |tensor: &str| format!("step_grad.{tensor}");
    let call = GatedDeltaStep {
        grad: Some(GatedDeltaStepCotangents {
            out: fixture.data(&name("w_out")),
            state: fixture.data(&name("w_state")),
        }),
        ..step(&fixture, "step_grad", true)
    };
    let Some(result) = ran(adapter, "step_grad", adapter.gated_delta_step(&call)) else {
        return tally.outcome(());
    };
    assert_close(
        "step_grad out",
        &result.out,
        &fixture.data(&name("out")),
        F32_OUT,
    );
    assert_close(
        "step_grad new_state",
        &result.state,
        &fixture.data(&name("new_state")),
        F32_STATE,
    );
    let grads = result.grads.expect("the gradients asked for");
    for (input, grad) in [
        ("q", grads.q),
        ("k", grads.k),
        ("v", grads.v),
        ("g", grads.g),
        ("beta", grads.beta),
        ("state", grads.state),
    ] {
        let expected = fixture.data(&format!("step_grad.grad_{input}"));
        assert_close(&format!("step_grad grad_{input}"), &grad, &expected, GRAD);
    }
    tally.outcome(())
}
