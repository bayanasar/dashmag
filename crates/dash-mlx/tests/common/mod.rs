//! The conformance tests' adapter for dash-mlx: `dash_conformance`'s test bodies run their calls
//! here, on MLX's GPU. Inputs are made from the host's f32 values in the dtype each is to be given
//! in, gradients come from MLX's autodiff (a VJP whose cotangents are the call's, each rounded to
//! its output's dtype, a term left out of the loss taking none), and outputs are read back as f32.
//!
//! What the bodies cannot ask of every implementation, which of dash-mlx's kernels a call ran, is
//! checked here around the calls: conv1d's forward and decoding kernels, the chunked gated delta
//! rule's inverses' and scan's kernels and the step's kernel ran exactly where they take the call;
//! so did conv1d's backward kernels over plain sequences, and the scan's backward kernel wherever
//! gradients were asked for.
#![allow(dead_code)]

use dash_conformance::{
    Adapter, Attention, AttentionDecode, AttentionOut, ChosenLogprob, ChosenLogprobOut,
    ChunkGatedDelta, ChunkGatedDeltaGrads, ChunkGatedDeltaOut, Conv1d, Conv1dGrads, Conv1dOut,
    Conv1dStep, Conv1dStepOut, DType, GatedDeltaStep, GatedDeltaStepGrads, GatedDeltaStepOut,
    HostTensor, Unsupported,
};
use dash_mlx::conv1d::{self, causal_conv1d, causal_conv1d_update};
use dash_mlx::gated_delta::chunk::{self, chunk_gated_delta_rule};
use dash_mlx::gated_delta::recurrent::{self, gated_delta_step};
use mlx_rs::error::Exception;
use mlx_rs::ops::zeros_dtype;
use mlx_rs::transforms::fallible_vjp;
use mlx_rs::{Array, Dtype};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// MLX's GPU, as the conformance bodies drive it.
pub struct Mlx;

/// MLX's dtype for the contract's.
pub fn mlx_dtype(dtype: DType) -> Dtype {
    match dtype {
        DType::F32 => Dtype::Float32,
        DType::BF16 => Dtype::Bfloat16,
        DType::F16 => Dtype::Float16,
    }
}

/// The contract's dtype for MLX's.
pub fn contract_dtype(dtype: Dtype) -> DType {
    match dtype {
        Dtype::Float32 => DType::F32,
        Dtype::Bfloat16 => DType::BF16,
        Dtype::Float16 => DType::F16,
        other => panic!("an output in {other:?}"),
    }
}

/// `host` as an array in its dtype: its f32 values, converted on the device (exactly: they are
/// the dtype's own).
pub fn array(host: &HostTensor) -> Array {
    let shape: Vec<i32> = host.shape.iter().map(|&d| d as i32).collect();
    Array::from_slice(&host.values, &shape)
        .as_dtype(mlx_dtype(host.dtype))
        .expect("a cast")
}

/// `a` read to the host as f32, with its dtype. An empty array has no data to read (MLX gives it
/// no buffer).
pub fn host(a: &Array) -> HostTensor {
    let dtype = contract_dtype(a.dtype());
    let shape = a.shape().iter().map(|&d| d as usize).collect();
    let values = match a.size() {
        0 => Vec::new(),
        _ => a
            .as_dtype(Dtype::Float32)
            .and_then(|a| a.contiguous())
            .expect("a read")
            .as_slice::<f32>()
            .to_vec(),
    };
    HostTensor::new(dtype, shape, values)
}

/// `offsets` as `cu_seqlens`, int32.
pub fn offsets(offsets: &[i64]) -> Array {
    let values: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
    Array::from_slice(&values, &[values.len() as i32])
}

/// The f32 bounds assume IEEE f32 matmuls; MLX runs them in TF32 on GPUs that have it unless
/// `MLX_ENABLE_TF32=0`, which puts errors near 5e-4 (measured on an Apple M6). Checked with one
/// 64 x 64 product against the same product summed in f64.
pub fn require_ieee_f32_matmul() {
    let n = 64;
    let a: Vec<f32> = (0..n * n)
        .map(|i| ((i * 7919 % 1013) as f32 / 1013.0) - 0.5)
        .collect();
    let b: Vec<f32> = (0..n * n)
        .map(|i| ((i * 104729 % 997) as f32 / 997.0) - 0.5)
        .collect();
    let product = Array::from_slice(&a, &[n as i32, n as i32])
        .matmul(Array::from_slice(&b, &[n as i32, n as i32]))
        .unwrap();
    let product = host(&product).values;
    let (mut err, mut scale) = (0f64, 0f64);
    for i in 0..n {
        for j in 0..n {
            let exact: f64 = (0..n)
                .map(|k| a[i * n + k] as f64 * b[k * n + j] as f64)
                .sum();
            err = err.max((product[i * n + j] as f64 - exact).abs());
            scale = scale.max(exact.abs());
        }
    }
    assert!(
        err / scale < 1e-5,
        "f32 matmuls run at {:.1e} relative error, not IEEE f32: run with MLX_ENABLE_TF32=0",
        err / scale
    );
}

/// The operation's refusal, an error that names it (`<op>: ...`), as the adapter reports it: the
/// body skips the call, and a test that asserts every call ran fails with the reason. Any other
/// failure is the test's, and panics.
fn refused(op: &'static str) -> impl Fn(Exception) -> Unsupported {
    move |e| {
        let what = e.what();
        match what
            .strip_prefix(op)
            .and_then(|rest| rest.strip_prefix(": "))
        {
            Some(_) => Unsupported(what.to_string()),
            None => panic!("{op} failed: {what}"),
        }
    }
}

/// What a call gave: its output, its states where it gives them, and the gradients of its
/// primals where it was asked for them.
type Ran = (Array, Option<Array>, Option<Vec<Array>>);

/// Runs `op`, as `f`, at `primals`: alone where `cotangents` is `None`, and otherwise through
/// MLX's VJP of the loss `sum(out * d_out) + sum(states * d_states)`, a term left out where its
/// cotangent is absent, each cotangent rounded to its output's dtype: `out`'s, of the shape and
/// dtype given (the contract's: its inputs'), or the states', f32.
fn run(
    op: &'static str,
    f: impl Fn(&[Array]) -> mlx_rs::error::Result<(Array, Option<Array>)>,
    primals: &[Array],
    out: (&[i32], Dtype),
    cotangents: Option<(Option<&HostTensor>, Option<&HostTensor>)>,
) -> Result<Ran, Unsupported> {
    let Some((d_out, d_states)) = cotangents else {
        let (out, states) = f(primals).map_err(refused(op))?;
        return Ok((out, states, None));
    };
    // The outputs, kept beside the VJP, and the loss's terms.
    let outputs = RefCell::new(None);
    let terms = |p: &[Array]| -> mlx_rs::error::Result<Vec<Array>> {
        let (out, states) = f(p)?;
        *outputs.borrow_mut() = Some((out.clone(), states.clone()));
        let mut terms = vec![out];
        if d_states.is_some() {
            terms.push(states.expect("the states asked for"));
        }
        Ok(terms)
    };
    let rounded = |d: &HostTensor, dtype| array(d).as_dtype(dtype).expect("a cast");
    let (shape, dtype) = out;
    let mut cotangents = vec![match d_out {
        Some(d_out) => rounded(d_out, dtype),
        None => zeros_dtype(shape, dtype).expect("zeros"),
    }];
    cotangents.extend(d_states.map(|d| rounded(d, Dtype::Float32)));
    let (_, grads) = fallible_vjp(terms, primals, &cotangents).map_err(refused(op))?;
    let (out, states) = outputs.into_inner().expect("the forward pass ran");
    Ok((out, states, Some(grads)))
}

/// Asserts that `count` grew past `before` where `kernels`, and not otherwise.
fn launched(kernels: bool, count: impl Fn() -> usize, before: usize, what: &str) {
    assert_eq!(
        count() > before,
        kernels,
        "{what}: the kernel ran {} times, expected {}",
        count() - before,
        if kernels { "some" } else { "none" }
    );
}

fn load(counter: &'static AtomicUsize) -> impl Fn() -> usize {
    || counter.load(Ordering::SeqCst)
}

impl Adapter for Mlx {
    fn describe(&self) -> String {
        "MLX on the Apple GPU".into()
    }

    fn has(&self, _: DType) -> bool {
        true
    }

    fn causal_conv1d(&self, call: &Conv1d) -> Result<Conv1dOut, Unsupported> {
        let x = array(&call.x);
        let weight = array(&call.weight);
        let bias = call.bias.as_ref().map(array);
        let initial = call.initial_state.as_ref().map(array);
        let cu = call.cu_seqlens.as_deref().map(offsets);
        let k = weight.shape()[1];
        let what = format!("causal_conv1d {:?} K = {k}", x.shape());
        let kernels = conv1d::kernels_take(k) && x.size() > 0;
        let before = [
            conv1d::probe::forward_launches(),
            conv1d::probe::backward_launches(),
        ];
        let f = |p: &[Array]| {
            let mut p = p.iter();
            let x = p.next().expect("x");
            let weight = p.next().expect("the weight");
            let bias = bias.as_ref().map(|_| p.next().expect("the bias"));
            let initial = initial
                .as_ref()
                .map(|_| p.next().expect("the initial states"));
            causal_conv1d(x, weight, bias, initial, cu.as_ref(), call.options)
        };
        let mut primals = vec![x.clone(), weight.clone()];
        primals.extend(bias.iter().cloned());
        primals.extend(initial.iter().cloned());
        let cotangents = call
            .grad
            .as_ref()
            .map(|d| (d.out.as_ref(), d.final_state.as_ref()));
        let out = (x.shape(), x.dtype());
        let (out, final_state, grads) = run("causal_conv1d", f, &primals, out, cotangents)?;
        let grads = grads.map(|grads| {
            let mut grads = grads.iter().map(host);
            let mut next = || grads.next().expect("a gradient");
            Conv1dGrads {
                x: next(),
                weight: next(),
                bias: bias.as_ref().map(|_| next()),
                initial_state: initial.as_ref().map(|_| next()),
            }
        });
        launched(kernels, conv1d::probe::forward_launches, before[0], &what);
        let plain = initial.is_none() && cu.is_none() && !call.options.output_final_state;
        if grads.is_some() && plain {
            launched(kernels, conv1d::probe::backward_launches, before[1], &what);
        }
        Ok(Conv1dOut {
            out: host(&out),
            final_state: final_state.as_ref().map(host),
            grads,
        })
    }

    fn causal_conv1d_update(&self, call: &Conv1dStep) -> Result<Conv1dStepOut, Unsupported> {
        let weight = array(&call.weight);
        let k = weight.shape()[1];
        let before = conv1d::probe::update_launches();
        let (out, state) = causal_conv1d_update(
            &array(&call.x),
            &weight,
            call.bias.as_ref().map(array).as_ref(),
            &array(&call.state),
            call.options,
        )
        .map_err(refused("causal_conv1d_update"))?;
        let (out, state) = (host(&out), host(&state));
        launched(
            conv1d::kernels_take(k),
            conv1d::probe::update_launches,
            before,
            "causal_conv1d_update",
        );
        Ok(Conv1dStepOut { out, state })
    }

    fn chunk_gated_delta_rule(
        &self,
        call: &ChunkGatedDelta,
    ) -> Result<ChunkGatedDeltaOut, Unsupported> {
        let [q, k, v, g, beta] = [&call.q, &call.k, &call.v, &call.g, &call.beta].map(array);
        let initial = call.initial_state.as_ref().map(array);
        let cu = call.cu_seqlens.as_deref().map(offsets);
        let [b, t, h, dk] = q.shape()[..] else {
            panic!("q is [B, T, H, Dk]")
        };
        let dv = v.shape()[3];
        let what = format!("chunk_gated_delta_rule {:?} Dv = {dv}", q.shape());
        let counters = [
            &chunk::counters::INVERSE_KERNEL,
            &chunk::counters::SCAN_FORWARD_KERNEL,
            &chunk::counters::SCAN_BACKWARD_KERNEL,
        ];
        let before = counters.map(|c| c.load(Ordering::SeqCst));
        let has_initial = initial.is_some();
        let f = |p: &[Array]| {
            let initial = has_initial.then(|| &p[5]);
            chunk_gated_delta_rule(
                &p[0],
                &p[1],
                &p[2],
                &p[3],
                &p[4],
                initial,
                cu.as_ref(),
                call.options,
            )
        };
        let mut primals = vec![q.clone(), k, v, g, beta];
        primals.extend(initial);
        let cotangents = call
            .grad
            .as_ref()
            .map(|d| (Some(&d.out), d.final_state.as_ref()));
        let out = (&[b, t, h, dv][..], q.dtype());
        let (out, final_state, grads) =
            run("chunk_gated_delta_rule", f, &primals, out, cotangents)?;
        let grads = grads.map(|grads| {
            let mut grads = grads.iter().map(host);
            let mut next = || grads.next().expect("a gradient");
            ChunkGatedDeltaGrads {
                q: next(),
                k: next(),
                v: next(),
                g: next(),
                beta: next(),
                initial_state: has_initial.then(&mut next),
            }
        });
        let scan = chunk::scan_kernels_take(dk, dv);
        let [inverse, forward, backward] = counters.map(load);
        launched(true, inverse, before[0], &what);
        launched(scan, forward, before[1], &what);
        launched(scan && grads.is_some(), backward, before[2], &what);
        Ok(ChunkGatedDeltaOut {
            out: host(&out),
            final_state: final_state.as_ref().map(host),
            grads,
        })
    }

    fn gated_delta_step(&self, call: &GatedDeltaStep) -> Result<GatedDeltaStepOut, Unsupported> {
        let primals = [&call.q, &call.k, &call.v, &call.g, &call.beta, &call.state].map(array);
        let cu = call.cu_seqlens.as_deref().map(offsets);
        let [n, h, dk] = primals[0].shape()[..] else {
            panic!("q is [B, H, Dk]")
        };
        let dv = primals[2].shape()[2];
        let before = recurrent::counters::STEP_KERNEL.load(Ordering::SeqCst);
        let f = |p: &[Array]| {
            let (out, state) = gated_delta_step(
                &p[0],
                &p[1],
                &p[2],
                &p[3],
                &p[4],
                &p[5],
                cu.as_ref(),
                call.options,
            )?;
            Ok((out, Some(state)))
        };
        let cotangents = call.grad.as_ref().map(|d| (Some(&d.out), Some(&d.state)));
        let out = (&[n, h, dv][..], primals[0].dtype());
        let (out, state, grads) = run("gated_delta_step", f, &primals, out, cotangents)?;
        let grads = grads.map(|grads| {
            let [q, k, v, g, beta, state] = <[Array; 6]>::try_from(grads)
                .expect("six gradients")
                .map(|grad| host(&grad));
            GatedDeltaStepGrads {
                q,
                k,
                v,
                g,
                beta,
                state,
            }
        });
        launched(
            recurrent::kernel_takes(dk),
            load(&recurrent::counters::STEP_KERNEL),
            before,
            &format!("gated_delta_step {:?}", primals[0].shape()),
        );
        Ok(GatedDeltaStepOut {
            out: host(&out),
            state: host(&state.expect("the new states")),
            grads,
        })
    }

    fn chosen_logprob(&self, _: &ChosenLogprob) -> Result<ChosenLogprobOut, Unsupported> {
        Err(Unsupported(
            "chosen_logprob: not through this adapter yet".into(),
        ))
    }

    fn attention(&self, _: &Attention) -> Result<AttentionOut, Unsupported> {
        Err(Unsupported(
            "attention: not through this adapter yet".into(),
        ))
    }

    fn attention_decode(&self, _: &AttentionDecode) -> Result<HostTensor, Unsupported> {
        Err(Unsupported(
            "attention_decode: not through this adapter yet".into(),
        ))
    }
}
