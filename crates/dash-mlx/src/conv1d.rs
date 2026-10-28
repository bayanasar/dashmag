//! Causal depthwise 1-D convolution over sequences, of any small kernel width, with an optional
//! bias and activation; from a given state or from zeros, over a batch or packed variable-length
//! sequences; with its gradients, and the step a decoder takes: dash-ops' `conv1d` on MLX.
//!
//! Per channel `c` of a sequence of `L` steps, with `K` the kernel width:
//!
//! ```text
//! y[t, c]   = bias[c] + sum_{j < K} weight[c, j] * h[t - (K - 1) + j, c]
//! out[t, c] = act(y[t, c])
//! ```
//!
//! `h` is the sequence's history: its inputs `x[0..L]`, preceded by its initial state, the `W`
//! inputs before its first step (oldest first), and by zeros before those; with no initial state,
//! by zeros alone. The sequence's final state is the last `W'` steps of its history, zeros where
//! the history is shorter: what the next call continues from. A sequence run in pieces, each from
//! the final state the previous one left, has the whole sequence's outputs and gradients.
//!
//! Layouts and dtypes as in dash-ops: `x` and the output `[B, T, C]`; packed sequences `B = 1`
//! with `cu_seqlens` `[N + 1]` (any int dtype) from 0 to `T` without decreasing, a sequence
//! possibly empty; `weight` `[C, K]`, `bias` `[C]`, in the dtype of `x`; states, initial and
//! final, `[S, W, C]` in f32, `S = B` or `N`; an initial state holds `W >= K - 1` steps, a final
//! state `W'` steps, those of the initial state by default, `K - 1` without one. One decoding
//! step's input and output `[B, C]`, its state `[B, W, C]`. Arithmetic is in f32, and the output
//! and each gradient are rounded once to the dtype of the input they belong to; states and their
//! gradients stay f32.
//!
//! # Parts
//!
//! [`reference`](mod@reference) builds everything from MLX's operations, for any width, and MLX
//! differentiates it. For widths 2 to [`MAX_KERNEL_WIDTH`], three Metal kernels run instead: the
//! forward pass, each thread walking a run of 64 steps of one channel with the last `K - 1`
//! inputs in registers; one backward pass that recomputes the pre-activation from `x` and forms
//! the input and initial states' gradients and partial sums of the weight and bias gradients over
//! the runs, which MLX adds up; and the decoding step. Training keeps the inputs and the output
//! for this operation. The final states are the last steps of the history, taken by MLX's
//! operations, so their gradient reaches the inputs and the initial states through MLX. Packed
//! sequences run one at a time, their boundaries read to the host, each through the kernels; the
//! weight and bias go to them in f32, so that each sequence's share of their gradients is added
//! in f32 and rounded once. The kernels index in 64 bits, so no size makes them read or write out
//! of bounds; each is built once per thread and kept.

use crate::kernel::{Launch, launch};
use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::TemplateArg;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{concatenate, sum_axis, zeros_dtype};
use mlx_rs::transforms::{custom_vjp, fallible_vjp};
use mlx_rs::{Array, Dtype};
use std::sync::atomic::Ordering;

/// The widest kernel the Metal kernels take; wider ones run the reference.
pub const MAX_KERNEL_WIDTH: i32 = 8;

/// Steps each thread of the sequence kernels walks through, and over which the backward pass sums
/// partial weight and bias gradients.
const RUN: i32 = 64;

/// Threads per threadgroup along the channels.
const GROUP: i32 = 256;

pub use dash_contract::conv1d::{Activation, CausalConv1dOptions, CausalConv1dStepOptions};

/// The activations this implementation computes: every one the contract has now; one it adds later
/// is refused until it is.
fn supported(op: &str, activation: Activation) -> Result<()> {
    match activation {
        Activation::None | Activation::Silu => Ok(()),
        other => Err(contract(
            op,
            format!("activation {other:?} is not supported"),
        )),
    }
}

/// Counters a test reads to tell which code ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static FORWARD_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static UPDATE_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);
    pub(super) static HOST_READS: AtomicUsize = AtomicUsize::new(0);

    /// Launches of this module's Metal kernels, all of them (none where the reference runs).
    pub fn kernel_launches() -> usize {
        forward_launches() + backward_launches() + update_launches()
    }

    /// Runs of the backward pass written out for the kernels.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }

    /// Launches of the forward kernel.
    pub fn forward_launches() -> usize {
        FORWARD_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of the backward kernel.
    pub fn backward_launches() -> usize {
        BACKWARD_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of the decoding step's kernel.
    pub fn update_launches() -> usize {
        UPDATE_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Reads of `cu_seqlens` to the host.
    pub fn host_reads() -> usize {
        HOST_READS.load(Ordering::SeqCst)
    }
}

/// Whether the Metal kernels, rather than the reference, take a call with a kernel of width
/// `width`. For tests, as the [`probe`] counters are.
#[doc(hidden)]
pub fn kernels_take(width: i32) -> bool {
    (2..=MAX_KERNEL_WIDTH).contains(&width)
}

fn contract(op: &str, message: String) -> Exception {
    Exception::custom(format!("{op}: {message}"))
}

/// Checks `weight` `[C, K]` and `bias` `[C]` against `C` channels and the dtype of `x`, and
/// returns `K`.
fn check_parameters(
    op: &str,
    c: i32,
    dtype: Dtype,
    weight: &Array,
    bias: Option<&Array>,
) -> Result<i32> {
    let k = match weight.shape() {
        [wc, k] if *wc == c && *k >= 1 => *k,
        shape => {
            return Err(contract(
                op,
                format!("weight must be [C, K] with C = {c} and K >= 1, got {shape:?}"),
            ));
        }
    };
    if weight.dtype() != dtype {
        return Err(contract(op, "x and weight must share one dtype".into()));
    }
    if let Some(bias) = bias {
        if bias.shape() != [c] {
            return Err(contract(
                op,
                format!("bias must be [C] = [{c}], got {:?}", bias.shape()),
            ));
        }
        if bias.dtype() != dtype {
            return Err(contract(op, "x and bias must share one dtype".into()));
        }
    }
    Ok(k)
}

/// Checks a state `[S, W, C]` against `S` sequences, `C` channels and a kernel of width `k`.
fn check_state(op: &str, what: &str, state: &Array, s: i32, c: i32, k: i32) -> Result<()> {
    match state.shape() {
        [ss, w, sc] if *ss == s && *sc == c && *w + 1 >= k => {}
        shape => {
            return Err(contract(
                op,
                format!(
                    "the {what} {shape:?} must be [S, W, C] with S = {s}, C = {c} and W >= {}",
                    k - 1
                ),
            ));
        }
    }
    if state.dtype() != Dtype::Float32 {
        return Err(contract(op, format!("the {what} is f32")));
    }
    Ok(())
}

/// The causal depthwise convolution of `x` `[B, T, C]` with `weight` `[C, K]`, plus `bias` `[C]`
/// if given, followed by `options.activation`, each sequence continuing from its initial state
/// `[S, W, C]` (f32) if given; packed sequences with `cu_seqlens` `[N + 1]`, `B = 1`. Returns the
/// output `[B, T, C]` in the dtype of `x` and, if asked, the final states `[S, W', C]`, f32.
/// Differentiable with respect to `x`, `weight`, `bias` and `initial_state`, through the output
/// and the final states.
pub fn causal_conv1d(
    x: &Array,
    weight: &Array,
    bias: Option<&Array>,
    initial_state: Option<&Array>,
    cu_seqlens: Option<&Array>,
    options: CausalConv1dOptions,
) -> Result<(Array, Option<Array>)> {
    convolve(x, weight, bias, initial_state, cu_seqlens, options, true)
}

/// One decoding step: the new token `x` `[B, C]`, `weight` `[C, K]`, `bias` `[C]` if given and
/// the f32 `state` `[B, W, C]` (the last `W >= K - 1` inputs, oldest first), to the token's
/// output `[B, C]` in the dtype of `x` and the next state, f32 `[B, W, C]`. A gradient asked of
/// it is the reference's.
pub fn causal_conv1d_update(
    x: &Array,
    weight: &Array,
    bias: Option<&Array>,
    state: &Array,
    options: CausalConv1dStepOptions,
) -> Result<(Array, Array)> {
    let op = "causal_conv1d_update";
    supported(op, options.activation)?;
    let (b, c) = match x.shape() {
        [b, c] => (*b, *c),
        shape => return Err(contract(op, format!("x must be [B, C], got {shape:?}"))),
    };
    let k = check_parameters(op, c, x.dtype(), weight, bias)?;
    check_state(op, "state", state, b, c, k)?;
    if !kernels_take(k) {
        return reference::step(x, weight, bias, state, options.activation);
    }
    let activation = options.activation;
    let has_bias = bias.is_some();
    let step = custom_vjp(
        move |p: &[Array]| update_kernel(&p[0], &p[1], &p[2], &p[3], activation),
        move |p: &[Array], cotangents: &[Array], _: &[Array]| {
            let f = |a: &[Array]| -> Result<Vec<Array>> {
                let bias = has_bias.then_some(&a[2]);
                let (out, next) = reference::step(&a[0], &a[1], bias, &a[3], activation)?;
                Ok(vec![out, next])
            };
            Ok(fallible_vjp(f, p, cotangents)?.1)
        },
    )?;
    let bias = bias_or_zeros(bias, c, x.dtype())?;
    let mut outputs = step.apply(&[x.clone(), weight.clone(), bias, state.clone()])?;
    let next = outputs.pop().expect("two outputs");
    Ok((outputs.pop().expect("two outputs"), next))
}

fn bias_or_zeros(bias: Option<&Array>, c: i32, dtype: Dtype) -> Result<Array> {
    match bias {
        Some(bias) => Ok(bias.clone()),
        None => zeros_dtype(&[c], dtype),
    }
}

/// [`causal_conv1d`], through the kernels where they take the call when `kernels`.
fn convolve(
    x: &Array,
    weight: &Array,
    bias: Option<&Array>,
    initial_state: Option<&Array>,
    cu_seqlens: Option<&Array>,
    options: CausalConv1dOptions,
    kernels: bool,
) -> Result<(Array, Option<Array>)> {
    let op = "causal_conv1d";
    supported(op, options.activation)?;
    let [b, t, c] = match x.shape() {
        [b, t, c] => [*b, *t, *c],
        shape => return Err(contract(op, format!("x must be [B, T, C], got {shape:?}"))),
    };
    let k = check_parameters(op, c, x.dtype(), weight, bias)?;
    let bounds = match cu_seqlens {
        None => None,
        Some(cu_seqlens) => {
            if b != 1 {
                return Err(contract(
                    op,
                    format!("packed sequences lie along T of a batch of one, got B = {b}"),
                ));
            }
            probe::HOST_READS.fetch_add(1, Ordering::SeqCst);
            let cu = cu_seqlens.as_dtype(Dtype::Int64)?;
            let cu = cu
                .as_slice::<i64>()
                .iter()
                .map(|&v| v as i32)
                .collect::<Vec<_>>();
            let ordered = cu.len() >= 2
                && cu[0] == 0
                && cu[cu.len() - 1] == t
                && cu.windows(2).all(|w| w[0] <= w[1]);
            if !ordered {
                return Err(contract(
                    op,
                    format!("cu_seqlens must go from 0 to T = {t} without decreasing, got {cu:?}"),
                ));
            }
            Some(cu)
        }
    };
    let sequences = bounds.as_ref().map_or(b, |cu| cu.len() as i32 - 1);
    if let Some(state) = initial_state {
        check_state(op, "initial state", state, sequences, c, k)?;
    }
    let final_len = options.output_final_state.then(|| {
        options
            .final_state_len
            .map(|len| len.min(i32::MAX as usize) as i32)
            .or(initial_state.map(|state| state.shape()[1]))
            .unwrap_or(k - 1)
    });
    let activation = options.activation;
    let Some(cu) = bounds else {
        return batch(
            x,
            weight,
            bias,
            initial_state,
            activation,
            final_len,
            kernels,
        );
    };
    // One sequence at a time, each a batch of one. The weight and bias go in as f32, so that
    // their gradients from every sequence are added in f32 and rounded once, by the conversion's
    // backward, rather than once per sequence.
    let weight = &weight.as_dtype(Dtype::Float32)?;
    let bias = bias.map(|b| b.as_dtype(Dtype::Float32)).transpose()?;
    let bias = bias.as_ref();
    let (mut outs, mut finals) = (Vec::new(), Vec::new());
    for s in 0..sequences as usize {
        let piece = x.try_index((.., cu[s]..cu[s + 1], ..))?;
        let state = initial_state
            .map(|state| state.try_index(s as i32..s as i32 + 1))
            .transpose()?;
        let (out, last) = batch(
            &piece,
            weight,
            bias,
            state.as_ref(),
            activation,
            final_len,
            kernels,
        )?;
        if cu[s + 1] > cu[s] {
            outs.push(out);
        }
        finals.extend(last);
    }
    let out = match outs.len() {
        0 => zeros_dtype(&[1, 0, c], x.dtype())?,
        1 => outs.pop().expect("one"),
        _ => concatenate(&outs, 1)?,
    };
    let finals = final_len.map(|_| concatenate(&finals, 0)).transpose()?;
    Ok((out, finals))
}

/// Every batch element one sequence, each from its initial state or zeros: the output, and the
/// final states of `final_len` steps if asked.
fn batch(
    x: &Array,
    weight: &Array,
    bias: Option<&Array>,
    initial_state: Option<&Array>,
    activation: Activation,
    final_len: Option<i32>,
    kernels: bool,
) -> Result<(Array, Option<Array>)> {
    let [b, t, c] = [x.shape()[0], x.shape()[1], x.shape()[2]];
    let k = weight.shape()[1];
    let out = if t == 0 {
        zeros_dtype(&[b, 0, c], x.dtype())?
    } else if kernels && kernels_take(k) {
        let state = match initial_state {
            Some(state) => state.clone(),
            None => zeros_dtype(&[b, k - 1, c], Dtype::Float32)?,
        };
        sequence_kernels(x, &state, weight, bias, activation)?
    } else {
        reference::sequence(x, weight, bias, initial_state, activation)?
    };
    let finals = final_len
        .map(|len| reference::last_steps(&reference::history(x, initial_state)?, len))
        .transpose()?;
    Ok((out, finals))
}

/// The sequence through the kernels, with the backward pass written out.
fn sequence_kernels(
    x: &Array,
    state: &Array,
    weight: &Array,
    bias: Option<&Array>,
    activation: Activation,
) -> Result<Array> {
    let c = x.shape()[2];
    let conv = custom_vjp(
        move |p: &[Array]| {
            Ok(vec![forward_kernel(
                &p[0], &p[1], &p[2], &p[3], activation,
            )?])
        },
        move |p: &[Array], cotangents: &[Array], _: &[Array]| {
            probe::BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
            backward_kernels(&p[0], &p[1], &p[2], &p[3], &cotangents[0], activation)
        },
    )?;
    let bias = bias_or_zeros(bias, c, x.dtype())?;
    Ok(conv
        .apply(&[x.clone(), state.clone(), weight.clone(), bias])?
        .remove(0))
}

/// The template arguments every kernel here takes.
fn template(dtype: Dtype, k: i32, activation: Activation) -> Vec<(String, TemplateArg)> {
    vec![
        ("DT".into(), TemplateArg::Dtype(dtype)),
        ("K".into(), TemplateArg::Int(k)),
        ("RUN".into(), TemplateArg::Int(RUN)),
        (
            "SILU".into(),
            TemplateArg::Bool(activation == Activation::Silu),
        ),
    ]
}

/// The activation, in f32: SiLU or the identity.
const HEADER: &str = "
inline float dash_conv_act(float y, bool silu) {
    return silu ? y / (1.0f + metal::precise::exp(-y)) : y;
}
";

/// One thread per channel and run of `RUN` steps; the last `K` inputs ride in registers. Before
/// the first step the inputs come from the state `[B, W, C]`.
const FORWARD: &str = "
    const int n_t = x_shape[1];
    const int C = x_shape[2];
    const int W = state_shape[1];
    const int c = thread_position_in_grid.x;
    const int runs = (n_t + RUN - 1) / RUN;
    const int b = thread_position_in_grid.y / runs;
    const int t0 = (thread_position_in_grid.y % runs) * RUN;
    if (c >= C) {
        return;
    }
    float w[K];
    for (int j = 0; j < K; ++j) {
        w[j] = float(weight[c * K + j]);
    }
    const float bc = float(bias[c]);
    float win[K];
    for (int j = 0; j < K - 1; ++j) {
        const int t = t0 - (K - 1) + j;
        win[j] = t >= 0 ? float(x[(size_t(b) * n_t + t) * C + c]) : state[(size_t(b) * W + W + t) * C + c];
    }
    const int t1 = min(t0 + RUN, n_t);
    for (int t = t0; t < t1; ++t) {
        win[K - 1] = float(x[(size_t(b) * n_t + t) * C + c]);
        float y = 0.0f;
        for (int j = 0; j < K; ++j) {
            y += w[j] * win[j];
        }
        out[(size_t(b) * n_t + t) * C + c] = static_cast<DT>(dash_conv_act(y + bc, SILU));
        for (int j = 0; j < K - 1; ++j) {
            win[j] = win[j + 1];
        }
    }
";

/// The backward pass, laid out as the forward. Each step's gradient before the activation, `gp`,
/// comes from the recomputed pre-activation; the input gradient at `t` takes `gp` at
/// `t .. t + K - 1`, so a run computes `gp` `K - 1` steps past its end. The first run of each
/// sequence also keeps `gp` at the first `K - 1` steps, from which it forms the initial state's
/// gradient, and zeros for the state's older steps.
const BACKWARD: &str = "
    const int n_t = x_shape[1];
    const int C = x_shape[2];
    const int W = state_shape[1];
    const int c = thread_position_in_grid.x;
    const int row = thread_position_in_grid.y;
    const int runs = (n_t + RUN - 1) / RUN;
    const int b = row / runs;
    const int t0 = (row % runs) * RUN;
    if (c >= C) {
        return;
    }
    const int t1 = min(t0 + RUN, n_t);
    float w[K];
    for (int j = 0; j < K; ++j) {
        w[j] = float(weight[c * K + j]);
    }
    const float bc = float(bias[c]);
    float win[K];
    for (int j = 0; j < K - 1; ++j) {
        const int t = t0 - (K - 1) + j;
        win[j] = t >= 0 ? float(x[(size_t(b) * n_t + t) * C + c]) : state[(size_t(b) * W + W + t) * C + c];
    }
    float gp[K];
    float dw[K];
    for (int j = 0; j < K; ++j) {
        gp[j] = 0.0f;
        dw[j] = 0.0f;
    }
    float db = 0.0f;
    float first[K - 1];
    for (int i = 0; i < K - 1; ++i) {
        first[i] = 0.0f;
    }
    for (int t = t0; t < t1 + K - 1; ++t) {
        float g_pre = 0.0f;
        if (t < n_t) {
            win[K - 1] = float(x[(size_t(b) * n_t + t) * C + c]);
            float y = 0.0f;
            for (int j = 0; j < K; ++j) {
                y += w[j] * win[j];
            }
            y += bc;
            const float g = float(grad[(size_t(b) * n_t + t) * C + c]);
            if (SILU) {
                const float s = 1.0f / (1.0f + metal::precise::exp(-y));
                g_pre = g * s * (1.0f + y * (1.0f - s));
            } else {
                g_pre = g;
            }
            if (t < t1) {
                for (int j = 0; j < K; ++j) {
                    dw[j] += g_pre * win[j];
                }
                db += g_pre;
            }
            for (int j = 0; j < K - 1; ++j) {
                win[j] = win[j + 1];
            }
        }
        for (int j = 0; j < K - 1; ++j) {
            gp[j] = gp[j + 1];
        }
        gp[K - 1] = g_pre;
        const int tau = t - (K - 1);
        if (tau >= t0) {
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) {
                acc += w[j] * gp[K - 1 - j];
            }
            dx[(size_t(b) * n_t + tau) * C + c] = static_cast<DT>(acc);
        }
        if (t0 == 0 && t < K - 1) {
            first[t] = g_pre;
        }
    }
    for (int j = 0; j < K; ++j) {
        dw_runs[(size_t(row) * C + c) * K + j] = dw[j];
    }
    db_runs[size_t(row) * C + c] = db;
    if (t0 == 0) {
        for (int s = 0; s < W - (K - 1); ++s) {
            dstate[(size_t(b) * W + s) * C + c] = 0.0f;
        }
        // State step i is history step i - (K - 1): its gradient takes gp at steps i - j, summed
        // over the taps in the order the input gradient sums them.
        for (int i = 0; i < K - 1; ++i) {
            float acc = 0.0f;
            for (int j = 0; j <= i; ++j) {
                acc += w[j] * first[i - j];
            }
            dstate[(size_t(b) * W + W - (K - 1) + i) * C + c] = acc;
        }
    }
";

/// One decoding step, one thread per channel and batch row, the channels adjacent.
const UPDATE: &str = "
    const int C = x_shape[1];
    const int W = state_shape[1];
    const int c = thread_position_in_grid.x;
    const int b = thread_position_in_grid.y;
    if (c >= C) {
        return;
    }
    float y = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        y += float(weight[c * K + j]) * state[(size_t(b) * W + W - (K - 1) + j) * C + c];
    }
    const float xv = float(x[size_t(b) * C + c]);
    y += float(weight[c * K + K - 1]) * xv;
    out[size_t(b) * C + c] = static_cast<DT>(dash_conv_act(y + float(bias[c]), SILU));
    for (int s = 0; s < W - 1; ++s) {
        next_state[(size_t(b) * W + s) * C + c] = state[(size_t(b) * W + s + 1) * C + c];
    }
    next_state[(size_t(b) * W + W - 1) * C + c] = xv;
";

fn config(
    grid: (i32, i32),
    outputs: Vec<(Vec<i32>, Dtype)>,
    dtype: Dtype,
    k: i32,
    activation: Activation,
) -> Launch {
    Launch {
        grid: (grid.0, grid.1, 1),
        thread_group: (grid.0.min(GROUP), 1, 1),
        outputs,
        template: template(dtype, k, activation),
        init_value: None,
    }
}

fn forward_kernel(
    x: &Array,
    state: &Array,
    weight: &Array,
    bias: &Array,
    activation: Activation,
) -> Result<Array> {
    let [b, t, c] = [x.shape()[0], x.shape()[1], x.shape()[2]];
    let k = weight.shape()[1];
    let kernel = crate::kernel::kernel(
        "dash_mlx_conv1d_forward",
        &["x", "state", "weight", "bias"],
        &["out"],
        FORWARD,
        HEADER,
    )?;
    probe::FORWARD_LAUNCHES.fetch_add(1, Ordering::SeqCst);
    let runs = (t + RUN - 1) / RUN;
    let outputs = vec![(vec![b, t, c], x.dtype())];
    let inputs = [x.clone(), state.clone(), weight.clone(), bias.clone()];
    let config = config((c, b * runs), outputs, x.dtype(), k, activation);
    // SAFETY: thread (c, row) returns unless c < C, and walks steps t0 .. min(t0 + RUN, T) of
    // sequence row / runs < B; it reads x, the weight and the bias within their shapes, and the
    // state's last K - 1 steps, which it holds (W >= K - 1, checked), and writes only its own
    // steps' outputs. The runs cover every output element once.
    Ok(unsafe { launch(&kernel, &inputs, &config)? }.remove(0))
}

/// The gradients of `x`, the state, the weight and the bias, given the output's.
fn backward_kernels(
    x: &Array,
    state: &Array,
    weight: &Array,
    bias: &Array,
    grad: &Array,
    activation: Activation,
) -> Result<Vec<Array>> {
    let [b, t, c] = [x.shape()[0], x.shape()[1], x.shape()[2]];
    let [w, k] = [state.shape()[1], weight.shape()[1]];
    let kernel = crate::kernel::kernel(
        "dash_mlx_conv1d_backward",
        &["x", "state", "weight", "bias", "grad"],
        &["dx", "dstate", "dw_runs", "db_runs"],
        BACKWARD,
        HEADER,
    )?;
    probe::BACKWARD_LAUNCHES.fetch_add(1, Ordering::SeqCst);
    let runs = (t + RUN - 1) / RUN;
    let outputs = vec![
        (vec![b, t, c], x.dtype()),
        (vec![b, w, c], Dtype::Float32),
        (vec![b * runs, c, k], Dtype::Float32),
        (vec![b * runs, c], Dtype::Float32),
    ];
    let inputs = [
        x.clone(),
        state.clone(),
        weight.clone(),
        bias.clone(),
        grad.as_dtype(x.dtype())?,
    ];
    let config = config((c, b * runs), outputs, x.dtype(), k, activation);
    // SAFETY: as the forward pass's, reading `grad` as it reads x; thread (c, row) writes dx at its
    // own steps, dw_runs and db_runs at its own (row, c), and, in the first run, every step of
    // dstate at (b, c). Every output element is written once.
    let mut grads = unsafe { launch(&kernel, &inputs, &config)? };
    let db = sum_axis(grads.pop().expect("four outputs"), 0, false)?.as_dtype(bias.dtype())?;
    let dw = sum_axis(grads.pop().expect("four outputs"), 0, false)?.as_dtype(weight.dtype())?;
    grads.extend([dw, db]);
    Ok(grads)
}

fn update_kernel(
    x: &Array,
    weight: &Array,
    bias: &Array,
    state: &Array,
    activation: Activation,
) -> Result<Vec<Array>> {
    let [b, w, c] = [state.shape()[0], state.shape()[1], state.shape()[2]];
    let k = weight.shape()[1];
    let kernel = crate::kernel::kernel(
        "dash_mlx_conv1d_update",
        &["x", "weight", "bias", "state"],
        &["out", "next_state"],
        UPDATE,
        HEADER,
    )?;
    probe::UPDATE_LAUNCHES.fetch_add(1, Ordering::SeqCst);
    let outputs = vec![(vec![b, c], x.dtype()), (vec![b, w, c], Dtype::Float32)];
    let inputs = [x.clone(), weight.clone(), bias.clone(), state.clone()];
    let config = config((c, b), outputs, x.dtype(), k, activation);
    // SAFETY: thread (c, b) returns unless c < C, reads row b of x and the state and channel c of
    // the weight and bias within their shapes, and writes out at (b, c) and every step of the next
    // state at (b, c), the only thread to.
    unsafe { launch(&kernel, &inputs, &config) }
}

/// The reference, built from MLX's operations for any kernel width and differentiated by MLX, in
/// f32 whatever the inputs' dtype, rounded to that dtype at the end: what other widths run, and
/// what the tests hold the kernels to.
pub mod reference {
    use super::*;

    fn activate(y: Array, activation: Activation) -> Result<Array> {
        match activation {
            Activation::None => Ok(y),
            Activation::Silu => y.multiply(mlx_rs::ops::sigmoid(&y)?),
            other => Err(contract(
                "conv1d",
                format!("activation {other:?} is not supported"),
            )),
        }
    }

    /// The history the final states take their steps from: the initial states (if any), then the
    /// inputs, `[B, W + L, C]` (f32).
    pub fn history(x: &Array, initial: Option<&Array>) -> Result<Array> {
        let x = x.as_dtype(Dtype::Float32)?;
        match initial {
            None => Ok(x),
            Some(state) => concatenate(&[state.clone(), x], 1),
        }
    }

    /// The last `len` steps of `history` `[B, P, C]`, zeros in front where it is shorter.
    pub fn last_steps(history: &Array, len: i32) -> Result<Array> {
        let [b, p, c] = [history.shape()[0], history.shape()[1], history.shape()[2]];
        if p >= len {
            return history.try_index((.., (p - len)..p, ..));
        }
        let front = zeros_dtype(&[b, len - p, c], Dtype::Float32)?;
        if p == 0 {
            return Ok(front);
        }
        concatenate(&[front, history.clone()], 1)
    }

    /// The output of one batch of sequences, each from its initial state or zeros.
    pub fn sequence(
        x: &Array,
        weight: &Array,
        bias: Option<&Array>,
        initial: Option<&Array>,
        activation: Activation,
    ) -> Result<Array> {
        let [b, t, c] = [x.shape()[0], x.shape()[1], x.shape()[2]];
        let k = weight.shape()[1];
        let xf = x.as_dtype(Dtype::Float32)?;
        let padded = if k == 1 {
            xf
        } else {
            let left = match initial {
                Some(state) => {
                    let w = state.shape()[1];
                    state.try_index((.., (w + 1 - k)..w, ..))?
                }
                None => zeros_dtype(&[b, k - 1, c], Dtype::Float32)?,
            };
            concatenate(&[left, xf], 1)?
        };
        let weight = weight.as_dtype(Dtype::Float32)?;
        let mut y: Option<Array> = None;
        for j in 0..k {
            let tap = weight.try_index((.., j))?;
            let term = padded.try_index((.., j..j + t, ..))?.multiply(&tap)?;
            y = Some(match y {
                None => term,
                Some(sum) => sum.add(term)?,
            });
        }
        let mut y = y.expect("a kernel width of at least 1");
        if let Some(bias) = bias {
            y = y.add(bias.as_dtype(Dtype::Float32)?)?;
        }
        activate(y, activation)?.as_dtype(x.dtype())
    }

    /// [`causal_conv1d`] from MLX's operations alone.
    pub fn causal_conv1d(
        x: &Array,
        weight: &Array,
        bias: Option<&Array>,
        initial_state: Option<&Array>,
        cu_seqlens: Option<&Array>,
        options: CausalConv1dOptions,
    ) -> Result<(Array, Option<Array>)> {
        convolve(x, weight, bias, initial_state, cu_seqlens, options, false)
    }

    /// One decoding step from MLX's operations, as
    /// [`causal_conv1d_update`] takes it.
    pub fn step(
        x: &Array,
        weight: &Array,
        bias: Option<&Array>,
        state: &Array,
        activation: Activation,
    ) -> Result<(Array, Array)> {
        let w = state.shape()[1];
        let token = x.as_dtype(Dtype::Float32)?.expand_dims(1)?;
        let history = concatenate(&[state.clone(), token], 1)?;
        let out = sequence(&x.expand_dims(1)?, weight, bias, Some(state), activation)?;
        Ok((out.squeeze_axes(&[1])?, last_steps(&history, w)?))
    }
}
