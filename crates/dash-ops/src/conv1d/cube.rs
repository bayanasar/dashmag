//! The CubeCL kernels, for kernel widths 1 to [`MAX_KERNEL_WIDTH`], on devices with what
//! [`requirements`] lists.
//!
//! The operation is memory-bound: `K` multiply-adds per element against a read and a write.
//! So each thread owns a run of [`TILE`] consecutive steps of one or two neighbouring channels
//! (one vector load), walks it in order, and keeps the window of the last `K - 1` inputs (and,
//! in the backward pass, of the last `K - 1` gradients) in registers: every element is read
//! once, plus a `K - 1`-step halo per run, and neighbouring threads read neighbouring channels,
//! which coalesces in the `[B, T, C]` layout. The kernel width, the bias's presence and the
//! activation are compile-time parameters, so each combination compiles to its own kernel with
//! its windows unrolled into registers and no branch on them.
//!
//! Sequences that continue from initial states, or that are packed, run kernels of their own
//! (`*_sequences_kernel`), so that the others are as they were. Runs stay 64 steps of the packed
//! `T`, wherever the sequences' boundaries fall: a thread finds the sequence its run starts in by
//! a binary search of `cu_seqlens` (read on the device, clamped to `[0, T]`), fills its window
//! from the history before the run (the inputs, the sequence's initial state, zeros), and where a
//! sequence starts inside its run, refills the window from that sequence's initial state (or
//! zeros). In the backward pass each slot of the gradients' window carries the sequence it
//! belongs to: a step's input gradient takes only its own sequence's slots, and where a sequence
//! starts within the window, the steps before its start are the last `K - 1` of its initial
//! state, whose gradient the same window gives. The other rows of the initial state's gradient
//! only the final state reaches, which a small kernel writes. A run computes its own steps and
//! those of the states before sequences starting within `K - 1` steps after them (run 0 also
//! those before step 0), so that each is computed once, by one thread.
//!
//! The weight and bias gradients are reductions over `B x T` per weight. The backward kernel
//! leaves one partial sum per run (`[B * runs, K + 1, C]` with a bias, `[B * runs, K, C]`
//! without, f32), and a second kernel adds them up per weight, so the long reduction is spread
//! over every thread of the grid rather than walked down by one thread per weight.
//!
//! Every tensor is indexed through its own strides: CubeCL may pad rows (a pitched layout), so a
//! tensor's strides are not implied by its shape. The kernels read vectors of up to [`LANES`]
//! channels, which needs a unit channel stride and outer strides that are multiples of the
//! vector width; the launchers copy an input whose channels are not adjacent, and pick a width
//! that holds for every tensor involved.

use super::{
    Activation, Bias, CausalConv1dBackend, MAX_KERNEL_WIDTH, Sequences, State, probe, reference,
};
use crate::cube::Requirements;
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::tensor::FloatTensor;
use burn::backend::{DType, Shape, TensorMetadata};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::{into_contiguous, untile};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::tensor_vector_size_parallel;
use std::sync::atomic::Ordering;

/// Steps one thread walks in the sequence kernels: the halo it reads twice is `K - 1` steps in
/// 64. A run longer than the window also guarantees that a run which does not start the sequence
/// has a whole window of steps before it.
const TILE: usize = 64;
/// Channels one thread carries, at most. Each lane holds the taps and windows in f32 registers,
/// so wider vectors cost threads, and they measured slower: at width 4 on one sequence of 6144
/// steps and 6144 channels (Qwen3.5-0.8B's), bf16, on an RTX 5070 Ti, the forward kernel takes
/// 203 us with 2 lanes, 220 with 4 and 234 with 8, and the backward kernel 335, 372 and 342
/// (with 2, 8 and 4); in f32 the width makes no difference.
const LANES: usize = 2;
/// Threads per cube.
const CUBE: u32 = 256;

/// Tap `j` of `weight` `[C, K]` for the vector of channels starting at `c0`.
#[cube]
fn tap<E: Float, N: Size>(weight: &Tensor<E>, c0: usize, j: usize) -> Vector<f32, N> {
    let mut v = Vector::<f32, N>::empty();
    let n = v.vector_size();
    let (stride_c, stride_j) = (weight.stride(0), weight.stride(1));
    #[unroll]
    for i in 0..n {
        v.insert(
            i,
            f32::cast_from(weight[(c0 + i) * stride_c + j * stride_j]),
        );
    }
    v
}

/// The `width` taps of `weight` `[C, K]` for the vector of channels starting at `c0`.
#[cube]
fn taps<E: Float, N: Size>(
    weight: &Tensor<E>,
    c0: usize,
    #[comptime] width: usize,
) -> Array<Vector<f32, N>> {
    let mut taps = Array::<Vector<f32, N>>::new(width);
    #[unroll]
    for j in 0..width {
        taps[j] = tap::<E, N>(weight, c0, j);
    }
    taps
}

/// `bias` `[C]` for the vector of channels starting at `c0`.
#[cube]
fn channels<E: Float, N: Size>(bias: &Tensor<E>, c0: usize) -> Vector<f32, N> {
    let mut v = Vector::<f32, N>::empty();
    let n = v.vector_size();
    let stride = bias.stride(0);
    #[unroll]
    for i in 0..n {
        v.insert(i, f32::cast_from(bias[(c0 + i) * stride]));
    }
    v
}

/// The pre-activation `shift + sum_j taps[j] * window[j]`, summed in that order. The forward
/// pass and the decoding step share it, so a prefill followed by single steps gives the
/// sequence's outputs to the bit.
#[cube]
fn convolve<N: Size>(
    taps: &Array<Vector<f32, N>>,
    window: &Array<Vector<f32, N>>,
    shift: Vector<f32, N>,
    #[comptime] width: usize,
) -> Vector<f32, N> {
    let mut y = shift;
    #[unroll]
    for j in 0..width {
        y += taps[j] * window[j];
    }
    y
}

/// The activation of the pre-activation `y`; SiLU as `y / (1 + exp(-y))`, which gives 0 rather
/// than NaN where `exp(-y)` overflows.
#[cube]
fn activate<N: Size>(y: Vector<f32, N>, #[comptime] activation: Activation) -> Vector<f32, N> {
    match activation {
        Activation::None => y,
        Activation::Silu => y / (Vector::<f32, N>::new(1.0f32) + (-y).exp()),
    }
}

/// The gradient of the pre-activation from the output's, `grad`: `grad * act'(y)`, the
/// pre-activation `y` recomputed from the window when the activation needs it.
#[cube]
fn pre_activation_grad<N: Size>(
    taps: &Array<Vector<f32, N>>,
    window: &Array<Vector<f32, N>>,
    shift: Vector<f32, N>,
    grad: Vector<f32, N>,
    #[comptime] width: usize,
    #[comptime] activation: Activation,
) -> Vector<f32, N> {
    match activation {
        Activation::None => grad,
        Activation::Silu => {
            // SiLU'(y) = sigma (1 + y (1 - sigma)).
            let y = convolve::<N>(taps, window, shift, width);
            let one = Vector::<f32, N>::new(1.0f32);
            let sigma = one / (one + (-y).exp());
            grad * sigma * (one + y * (one - sigma))
        }
    }
}

/// Moves a window one step on: `window[j] = window[j + 1]`, the last slot left for the next
/// step.
#[cube]
fn slide<T: CubePrimitive>(window: &mut Array<T>, #[comptime] width: usize) {
    #[unroll]
    for j in 0..comptime![width - 1] {
        window[j] = window[j + 1];
    }
}

/// A window of `width` vectors for a run starting at step `t0`: the `width - 1` inputs before
/// it, zero before the sequence, and a last slot for the current step. Runs start at multiples
/// of the tile, which is longer than the window, so a run that does not start the sequence has
/// the whole window before it.
#[cube]
fn history<E: Float, N: Size>(
    x: &Tensor<Vector<E, N>>,
    x_at: usize,
    x_step: usize,
    t0: usize,
    #[comptime] width: usize,
) -> Array<Vector<f32, N>> {
    let mut window = Array::<Vector<f32, N>>::new(width);
    #[unroll]
    for j in 0..width {
        window[j] = Vector::<f32, N>::zeroed();
    }
    if t0 > 0 {
        #[unroll]
        for j in 0..comptime![width - 1] {
            window[j] = Vector::cast_from(x[x_at + (t0 + j + 1 - width) * x_step]);
        }
    }
    window
}

/// The thread's run: `(batch, run, channel vector)` from its position, the runs of each batch
/// laid out after one another and the channel vectors innermost.
#[cube]
fn locate(pos: usize, runs: usize, row: usize) -> (usize, usize, usize) {
    let cv = pos % row;
    let rest = pos / row;
    (rest / runs, rest % runs, cv)
}

/// Where step 0 of batch `b`, channel vector `cv` of a `[B, T, C]` tensor is, and how far one
/// step moves, both in vectors of `n`.
#[cube]
fn origin<T: CubePrimitive>(tensor: &Tensor<T>, b: usize, cv: usize, n: usize) -> (usize, usize) {
    (b * (tensor.stride(0) / n) + cv, tensor.stride(1) / n)
}

/// [`origin`] of a tensor the kernel writes.
#[cube]
fn origin_mut<T: CubePrimitive>(
    tensor: &mut Tensor<T>,
    b: usize,
    cv: usize,
    n: usize,
) -> (usize, usize) {
    (b * (tensor.stride(0) / n) + cv, tensor.stride(1) / n)
}

/// The boundary `cu_seqlens[s]` (contiguous), clamped to `[0, steps]` (a negative one wraps past
/// `steps`), so that malformed boundaries cannot send a thread out of its tensors.
#[cube]
fn offset<I: Numeric>(cu_seqlens: &Tensor<I>, s: usize, steps: usize) -> usize {
    clamp_max(usize::cast_from(cu_seqlens[s]), steps)
}

/// The sequence that holds step `t`: the last `s` of the `S` with `cu_seqlens[s] <= t`, so that
/// empty sequences before it are passed over.
#[cube]
fn sequence_of<I: Numeric>(cu_seqlens: &Tensor<I>, t: usize, steps: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = cu_seqlens.shape(0) - 2;
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if offset::<I>(cu_seqlens, mid, steps) <= t {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// The sequence after `seq` that holds step `t`, which starts there: empty ones are passed over.
#[cube]
fn next_sequence<I: Numeric>(cu_seqlens: &Tensor<I>, seq: usize, t: usize, steps: usize) -> usize {
    let last = cu_seqlens.shape(0) - 2;
    let mut s = clamp_max(seq + 1, last);
    while s < last && offset::<I>(cu_seqlens, s + 1, steps) <= t {
        s += 1;
    }
    s
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn forward_kernel<E: Float, N: Size>(
    x: &Tensor<Vector<E, N>>,
    weight: &Tensor<E>,
    bias: ComptimeOption<&Tensor<E>>,
    out: &mut Tensor<Vector<E, N>>,
    #[comptime] width: usize,
    #[comptime] tile: usize,
    #[comptime] activation: Activation,
    #[define(E)] _dtype: ElemType,
) {
    let n = x.vector_size();
    let steps = x.shape(1);
    let row = x.shape(2) / n;
    let runs = steps.div_ceil(tile);
    let (b, run, cv) = locate(ABSOLUTE_POS, runs, row);
    if b >= x.shape(0) {
        terminate!();
    }
    let t0 = run * tile;
    let c0 = cv * n;
    let taps = taps::<E, N>(weight, c0, width);
    #[comptime]
    let shift = match bias {
        ComptimeOption::Some(bias) => channels::<E, N>(bias, c0),
        ComptimeOption::None => Vector::<f32, N>::zeroed(),
    };
    let (x_at, x_step) = origin(x, b, cv, n);
    let out_step = out.stride(1) / n;
    let out_at = b * (out.stride(0) / n) + cv;

    // window[j] holds x[t - (K - 1) + j]; the current step goes last.
    let mut window = history::<E, N>(x, x_at, x_step, t0, width);
    let end = clamp_max(t0 + tile, steps);
    for t in t0..end {
        window[comptime![width - 1]] = Vector::cast_from(x[x_at + t * x_step]);
        let y = convolve::<N>(&taps, &window, shift, width);
        out[out_at + t * out_step] = Vector::cast_from(activate::<N>(y, activation));
        slide::<Vector<f32, N>>(&mut window, width);
    }
}

/// The forward kernel over sequences that continue from initial states `[S, W, C]` (f32,
/// `W >= K - 1`), or that are packed (`cu_seqlens` `[S + 1]`, `B = 1`), or both: the history
/// before each sequence is its initial state, or zeros.
// The options' `if let`s are decided when the kernel compiles, the `if`s inside them when it
// runs: they do not collapse into one.
#[allow(clippy::collapsible_if)]
#[cube(launch_unchecked, address_type = "dynamic")]
fn forward_sequences_kernel<E: Float, I: Numeric, N: Size>(
    x: &Tensor<Vector<E, N>>,
    weight: &Tensor<E>,
    bias: ComptimeOption<&Tensor<E>>,
    initial: ComptimeOption<&Tensor<Vector<f32, N>>>,
    cu_seqlens: ComptimeOption<&Tensor<I>>,
    out: &mut Tensor<Vector<E, N>>,
    #[comptime] width: usize,
    #[comptime] tile: usize,
    #[comptime] activation: Activation,
    #[define(E, I)] _dtypes: [ElemType; 2],
) {
    let n = x.vector_size();
    let steps = x.shape(1);
    let row = x.shape(2) / n;
    let runs = steps.div_ceil(tile);
    let (b, run, cv) = locate(ABSOLUTE_POS, runs, row);
    if b >= x.shape(0) {
        terminate!();
    }
    let t0 = run * tile;
    let c0 = cv * n;
    let taps = taps::<E, N>(weight, c0, width);
    #[comptime]
    let shift = match bias {
        ComptimeOption::Some(bias) => channels::<E, N>(bias, c0),
        ComptimeOption::None => Vector::<f32, N>::zeroed(),
    };
    let (x_at, x_step) = origin(x, b, cv, n);
    let out_step = out.stride(1) / n;
    let out_at = b * (out.stride(0) / n) + cv;

    // The sequence the run starts in (batch element `b` unpacked), where it starts and ends.
    let mut seq = b;
    let mut start = 0usize;
    let mut end = steps;
    #[comptime]
    if let ComptimeOption::Some(cu) = cu_seqlens {
        seq = sequence_of::<I>(cu, t0, steps);
        start = clamp_max(offset::<I>(cu, seq, steps), t0);
        end = offset::<I>(cu, seq + 1, steps);
    }

    // window[j] holds the history at t - (K - 1) + j: the inputs from the sequence's start, its
    // initial state before that (or zeros). The current step goes last.
    let mut window = Array::<Vector<f32, N>>::new(width);
    #[unroll]
    for j in 0..width {
        window[j] = Vector::<f32, N>::zeroed();
    }
    #[unroll]
    for j in 0..comptime![width - 1] {
        // The step t0 + j + 1 - K, as `q - K` (it may lie before step 0).
        let q = t0 + j + 1;
        if q >= start + width {
            window[j] = Vector::cast_from(x[x_at + (q - width) * x_step]);
        } else {
            #[comptime]
            if let ComptimeOption::Some(initial) = initial {
                // Row W - (start - (q - K)) of the state, at least 0 as W >= K - 1.
                let (h_at, h_step) = origin(initial, seq, cv, n);
                window[j] = initial[h_at + (initial.shape(1) + q - start - width) * h_step];
            }
        }
    }
    let run_end = clamp_max(t0 + tile, steps);
    for t in t0..run_end {
        #[comptime]
        if let ComptimeOption::Some(cu) = cu_seqlens {
            if t == end {
                // A sequence starts here: its history is its initial state, or zeros.
                seq = next_sequence::<I>(cu, seq, t, steps);
                end = offset::<I>(cu, seq + 1, steps);
                #[unroll]
                for j in 0..comptime![width - 1] {
                    window[j] = Vector::<f32, N>::zeroed();
                }
                #[comptime]
                if let ComptimeOption::Some(initial) = initial {
                    let (h_at, h_step) = origin(initial, seq, cv, n);
                    let first = initial.shape(1) + 1 - width;
                    #[unroll]
                    for j in 0..comptime![width - 1] {
                        window[j] = initial[h_at + (first + j) * h_step];
                    }
                }
            }
        }
        window[comptime![width - 1]] = Vector::cast_from(x[x_at + t * x_step]);
        let y = convolve::<N>(&taps, &window, shift, width);
        out[out_at + t * out_step] = Vector::cast_from(activate::<N>(y, activation));
        slide::<Vector<f32, N>>(&mut window, width);
    }
}

/// The final states `[S, W', C]` (f32): each sequence's last `W'` steps of history, its inputs,
/// then its initial state's last steps (if any), then zeros. One thread per sequence and
/// channel vector.
#[cube(launch_unchecked, address_type = "dynamic")]
fn final_state_kernel<E: Float, I: Numeric, N: Size>(
    x: &Tensor<Vector<E, N>>,
    initial: ComptimeOption<&Tensor<Vector<f32, N>>>,
    cu_seqlens: ComptimeOption<&Tensor<I>>,
    state: &mut Tensor<Vector<f32, N>>,
    #[define(E, I)] _dtypes: [ElemType; 2],
) {
    let n = x.vector_size();
    let steps = x.shape(1);
    let row = x.shape(2) / n;
    let lane = ABSOLUTE_POS;
    if lane >= state.shape(0) * row {
        terminate!();
    }
    let seq = lane / row;
    let cv = lane % row;
    // The sequence's batch row, first step and length.
    let mut b = seq;
    let mut start = 0usize;
    let mut len = steps;
    #[comptime]
    if let ComptimeOption::Some(cu) = cu_seqlens {
        b = 0;
        start = offset::<I>(cu, seq, steps);
        let stop = offset::<I>(cu, seq + 1, steps);
        len = 0;
        if stop > start {
            len = stop - start;
        }
    }
    let (x_at, x_step) = origin(x, b, cv, n);
    let (s_at, s_step) = origin_mut(state, seq, cv, n);
    let w_out = state.shape(1);
    for i in 0..w_out {
        // History step len - W' + i of the sequence.
        let mut v = Vector::<f32, N>::zeroed();
        if i + len >= w_out {
            v = Vector::cast_from(x[x_at + (start + len + i - w_out) * x_step]);
        } else {
            #[comptime]
            if let ComptimeOption::Some(initial) = initial {
                let w_in = initial.shape(1);
                if w_in + len + i >= w_out {
                    let (h_at, h_step) = origin(initial, seq, cv, n);
                    v = initial[h_at + (w_in + len + i - w_out) * h_step];
                }
            }
        }
        state[s_at + i * s_step] = v;
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn backward_kernel<E: Float, N: Size>(
    x: &Tensor<Vector<E, N>>,
    weight: &Tensor<E>,
    bias: ComptimeOption<&Tensor<E>>,
    grad: &Tensor<Vector<E, N>>,
    grad_x: &mut Tensor<Vector<E, N>>,
    partial: &mut Tensor<Vector<f32, N>>,
    #[comptime] width: usize,
    #[comptime] tile: usize,
    #[comptime] activation: Activation,
    #[define(E)] _dtype: ElemType,
) {
    let n = x.vector_size();
    let steps = x.shape(1);
    let row = x.shape(2) / n;
    let runs = steps.div_ceil(tile);
    let (b, run, cv) = locate(ABSOLUTE_POS, runs, row);
    if b >= x.shape(0) {
        terminate!();
    }
    let t0 = run * tile;
    let c0 = cv * n;
    let taps = taps::<E, N>(weight, c0, width);
    let has_bias = bias.is_some();
    #[comptime]
    let shift = match bias {
        ComptimeOption::Some(bias) => channels::<E, N>(bias, c0),
        ComptimeOption::None => Vector::<f32, N>::zeroed(),
    };
    let (x_at, x_step) = origin(x, b, cv, n);
    let (grad_at, grad_step) = origin(grad, b, cv, n);
    let gx_step = grad_x.stride(1) / n;
    let gx_at = b * (grad_x.stride(0) / n) + cv;

    // window[j] holds x[s - (K - 1) + j], grads[j] the pre-activation's gradient there; the
    // current step goes last in both.
    let mut window = history::<E, N>(x, x_at, x_step, t0, width);
    let mut grads = Array::<Vector<f32, N>>::new(width);
    // This run's share of the weight gradient, tap by tap, and of the bias gradient.
    let mut sums = Array::<Vector<f32, N>>::new(width);
    #[unroll]
    for j in 0..width {
        grads[j] = Vector::<f32, N>::zeroed();
        sums[j] = Vector::<f32, N>::zeroed();
    }
    let mut sum_bias = Vector::<f32, N>::zeroed();

    // Input step t feeds output steps t .. t + K - 1, so the run's input gradients need the
    // output gradients K - 1 steps past its end; past the sequence they are zero.
    for s in t0..t0 + tile + comptime![width - 1] {
        let mut xs = Vector::<f32, N>::zeroed();
        let mut gs = Vector::<f32, N>::zeroed();
        if s < steps {
            xs = Vector::cast_from(x[x_at + s * x_step]);
            gs = Vector::cast_from(grad[grad_at + s * grad_step]);
        }
        window[comptime![width - 1]] = xs;
        if s < steps {
            gs = pre_activation_grad::<N>(&taps, &window, shift, gs, width, activation);
        }
        grads[comptime![width - 1]] = gs;
        if s < t0 + tile {
            #[unroll]
            for j in 0..width {
                sums[j] += gs * window[j];
            }
            if has_bias {
                sum_bias += gs;
            }
        }
        // grad_x[t] = sum_j weight[j] * g[t + K - 1 - j], with t = s - (K - 1): g[s - j] is
        // grads[K - 1 - j].
        if s + 1 >= t0 + width {
            let t = s + 1 - width;
            if t < steps {
                let mut gx = Vector::<f32, N>::zeroed();
                #[unroll]
                for j in 0..width {
                    gx += taps[j] * grads[comptime![width - 1] - j];
                }
                grad_x[gx_at + t * gx_step] = Vector::cast_from(gx);
            }
        }
        slide::<Vector<f32, N>>(&mut window, width);
        slide::<Vector<f32, N>>(&mut grads, width);
    }

    // partial[b * runs + run, j, cv], contiguous, in vectors: the taps' sums, then the bias's.
    let rows = partial.shape(1);
    let at = (b * runs + run) * rows * row + cv;
    #[unroll]
    for j in 0..width {
        partial[at + j * row] = sums[j];
    }
    if has_bias {
        partial[at + width * row] = sum_bias;
    }
}

/// The weighted sum of a window's gradients that slot `own`'s sequence takes:
/// `sum_j taps[j] * grads[K - 1 - j]` over the slots of that sequence (`tags`), the others taken
/// as zero, summed in the order the plain kernel sums.
#[cube]
fn own_sum<N: Size>(
    taps: &Array<Vector<f32, N>>,
    grads: &Array<Vector<f32, N>>,
    tags: &Array<usize>,
    own: usize,
    #[comptime] width: usize,
) -> Vector<f32, N> {
    let mut sum = Vector::<f32, N>::zeroed();
    #[unroll]
    for j in 0..width {
        let m = comptime![width - 1 - j];
        let mut g = Vector::<f32, N>::zeroed();
        if tags[m] == own {
            g = grads[m];
        }
        sum += taps[j] * g;
    }
    sum
}

/// The backward kernel over sequences from initial states, or packed ones, or with the final
/// states' gradient `grad_final` `[S, W', C]`: the input gradient, the initial states' gradient
/// for their last `K - 1` rows (the backward kernel of states writes the others), and the weight
/// and bias gradients' partial sums (`grad_initial` is a placeholder without initial states).
/// The window's slots carry their sequences (`tags`: 0 before step 0, `S + 1` past the last step,
/// `s + 1` in sequence `s`) and those sequences' ends.
// As in the forward kernel, compile-time `if let`s around run-time `if`s.
#[allow(clippy::collapsible_if)]
#[cube(launch_unchecked, address_type = "dynamic")]
fn backward_sequences_kernel<E: Float, I: Numeric, N: Size>(
    x: &Tensor<Vector<E, N>>,
    weight: &Tensor<E>,
    bias: ComptimeOption<&Tensor<E>>,
    initial: ComptimeOption<&Tensor<Vector<f32, N>>>,
    cu_seqlens: ComptimeOption<&Tensor<I>>,
    grad: &Tensor<Vector<E, N>>,
    grad_final: ComptimeOption<&Tensor<Vector<f32, N>>>,
    grad_x: &mut Tensor<Vector<E, N>>,
    grad_initial: &mut Tensor<Vector<f32, N>>,
    partial: &mut Tensor<Vector<f32, N>>,
    #[comptime] width: usize,
    #[comptime] tile: usize,
    #[comptime] activation: Activation,
    #[define(E, I)] _dtypes: [ElemType; 2],
) {
    let n = x.vector_size();
    let steps = x.shape(1);
    let row = x.shape(2) / n;
    let runs = steps.div_ceil(tile);
    let (b, run, cv) = locate(ABSOLUTE_POS, runs, row);
    if b >= x.shape(0) {
        terminate!();
    }
    let t0 = run * tile;
    let c0 = cv * n;
    let taps = taps::<E, N>(weight, c0, width);
    let has_bias = bias.is_some();
    #[comptime]
    let shift = match bias {
        ComptimeOption::Some(bias) => channels::<E, N>(bias, c0),
        ComptimeOption::None => Vector::<f32, N>::zeroed(),
    };
    let (x_at, x_step) = origin(x, b, cv, n);
    let (grad_at, grad_step) = origin(grad, b, cv, n);
    let gx_step = grad_x.stride(1) / n;
    let gx_at = b * (grad_x.stride(0) / n) + cv;
    let has_initial = initial.is_some();

    // The sequence the run starts in (batch element `b` unpacked), where it starts and ends, and
    // the tag of the steps past the last.
    let mut seq = b;
    let mut start = 0usize;
    let mut end = steps;
    let mut past = x.shape(0) + 1;
    #[comptime]
    if let ComptimeOption::Some(cu) = cu_seqlens {
        past = cu.shape(0);
        seq = sequence_of::<I>(cu, t0, steps);
        start = clamp_max(offset::<I>(cu, seq, steps), t0);
        end = offset::<I>(cu, seq + 1, steps);
    }

    // window[j] holds the history at s - (K - 1) + j, as in the forward kernel; grads[j] the
    // pre-activation's gradient there, tags[j] and ends[j] its sequence and that sequence's end.
    let mut window = Array::<Vector<f32, N>>::new(width);
    let mut grads = Array::<Vector<f32, N>>::new(width);
    let mut tags = Array::<usize>::new(width);
    let mut ends = Array::<usize>::new(width);
    let mut sums = Array::<Vector<f32, N>>::new(width);
    #[unroll]
    for j in 0..width {
        window[j] = Vector::<f32, N>::zeroed();
        grads[j] = Vector::<f32, N>::zeroed();
        tags[j] = 0usize;
        ends[j] = 0usize;
        sums[j] = Vector::<f32, N>::zeroed();
    }
    let mut sum_bias = Vector::<f32, N>::zeroed();
    #[unroll]
    for j in 0..comptime![width - 1] {
        let q = t0 + j + 1;
        if q >= start + width {
            window[j] = Vector::cast_from(x[x_at + (q - width) * x_step]);
        } else {
            #[comptime]
            if let ComptimeOption::Some(initial) = initial {
                let (h_at, h_step) = origin(initial, seq, cv, n);
                window[j] = initial[h_at + (initial.shape(1) + q - start - width) * h_step];
            }
        }
    }

    for s in t0..t0 + tile + comptime![width - 1] {
        let mut xs = Vector::<f32, N>::zeroed();
        let mut gs = Vector::<f32, N>::zeroed();
        let mut tag = past;
        if s < steps {
            #[comptime]
            if let ComptimeOption::Some(cu) = cu_seqlens {
                if s == end {
                    seq = next_sequence::<I>(cu, seq, s, steps);
                    end = offset::<I>(cu, seq + 1, steps);
                    #[unroll]
                    for j in 0..comptime![width - 1] {
                        window[j] = Vector::<f32, N>::zeroed();
                    }
                    #[comptime]
                    if let ComptimeOption::Some(initial) = initial {
                        let (h_at, h_step) = origin(initial, seq, cv, n);
                        let first = initial.shape(1) + 1 - width;
                        #[unroll]
                        for j in 0..comptime![width - 1] {
                            window[j] = initial[h_at + (first + j) * h_step];
                        }
                    }
                }
            }
            xs = Vector::cast_from(x[x_at + s * x_step]);
            gs = Vector::cast_from(grad[grad_at + s * grad_step]);
            tag = seq + 1;
        }
        window[comptime![width - 1]] = xs;
        if s < steps {
            gs = pre_activation_grad::<N>(&taps, &window, shift, gs, width, activation);
        }
        grads[comptime![width - 1]] = gs;
        tags[comptime![width - 1]] = tag;
        ends[comptime![width - 1]] = end;
        if s < t0 + tile {
            #[unroll]
            for j in 0..width {
                sums[j] += gs * window[j];
            }
            if has_bias {
                sum_bias += gs;
            }
        }
        // Step v = s - (K - 1) has all its gradients now. The run computes it if it is one of
        // its own steps, or (run 0) lies before step 0.
        if s + 1 >= t0 + width || t0 == 0 {
            let own = tags[0];
            if own != 0 && own != past {
                // A step of sequence `own - 1`: its input gradient, from its own sequence's
                // outputs, and from its final state if within its last W' steps.
                let mut gx = own_sum::<N>(&taps, &grads, &tags, own, width);
                #[comptime]
                if let ComptimeOption::Some(grad_final) = grad_final {
                    let len = grad_final.shape(1);
                    if s + 1 + len >= width + ends[0] {
                        let (f_at, f_step) = origin(grad_final, own - 1, cv, n);
                        gx += grad_final[f_at + (s + 1 + len - width - ends[0]) * f_step];
                    }
                }
                grad_x[gx_at + (s + 1 - width) * gx_step] = Vector::cast_from(gx);
            }
            if has_initial {
                // Each sequence starting in slot m (1 <= m < K) has v as its initial state's row
                // W - m.
                if own != tags[comptime![width - 1]] {
                    #[unroll]
                    for m in 1..width {
                        let q = tags[m];
                        if q != tags[m - 1] && q != past {
                            let mut gh = own_sum::<N>(&taps, &grads, &tags, q, width);
                            #[comptime]
                            if let ComptimeOption::Some(grad_final) = grad_final {
                                let len = grad_final.shape(1);
                                if s + 1 + len >= width + ends[m] {
                                    let (f_at, f_step) = origin(grad_final, q - 1, cv, n);
                                    gh +=
                                        grad_final[f_at + (s + 1 + len - width - ends[m]) * f_step];
                                }
                            }
                            let (h_at, h_step) = origin_mut(grad_initial, q - 1, cv, n);
                            let w_in = grad_initial.shape(1);
                            grad_initial[h_at + (w_in - m) * h_step] = gh;
                        }
                    }
                }
            }
        }
        slide::<Vector<f32, N>>(&mut window, width);
        slide::<Vector<f32, N>>(&mut grads, width);
        slide::<usize>(&mut tags, width);
        slide::<usize>(&mut ends, width);
    }

    // partial[b * runs + run, j, cv], contiguous, in vectors: the taps' sums, then the bias's.
    let rows = partial.shape(1);
    let at = (b * runs + run) * rows * row + cv;
    #[unroll]
    for j in 0..width {
        partial[at + j * row] = sums[j];
    }
    if has_bias {
        partial[at + width * row] = sum_bias;
    }
}

/// The rows of the initial states' gradient `[S, W, C]` the backward kernel does not write: of a
/// sequence with steps, those before its last `K - 1`, which only its final state reaches; of an
/// empty one, all. The final states' gradient where it reaches them (a sequence shorter than
/// its final state), zeros elsewhere. One thread per sequence and channel vector.
#[cube(launch_unchecked, address_type = "dynamic")]
fn initial_state_grad_kernel<I: Numeric, N: Size>(
    cu_seqlens: ComptimeOption<&Tensor<I>>,
    grad_final: ComptimeOption<&Tensor<Vector<f32, N>>>,
    grad_initial: &mut Tensor<Vector<f32, N>>,
    steps: u32,
    #[comptime] width: usize,
    #[define(I)] _int: ElemType,
) {
    let n = grad_initial.vector_size();
    let row = grad_initial.shape(2) / n;
    let lane = ABSOLUTE_POS;
    if lane >= grad_initial.shape(0) * row {
        terminate!();
    }
    let seq = lane / row;
    let cv = lane % row;
    let steps = steps as usize;
    let mut len = steps;
    #[comptime]
    if let ComptimeOption::Some(cu) = cu_seqlens {
        let start = offset::<I>(cu, seq, steps);
        let stop = offset::<I>(cu, seq + 1, steps);
        len = 0;
        if stop > start {
            len = stop - start;
        }
    }
    let w_in = grad_initial.shape(1);
    let mut rows = w_in;
    if len > 0 {
        rows = w_in + 1 - width;
    }
    let (h_at, h_step) = origin_mut(grad_initial, seq, cv, n);
    for i in 0..rows {
        let mut g = Vector::<f32, N>::zeroed();
        #[comptime]
        if let ComptimeOption::Some(grad_final) = grad_final {
            // Row i is history step i - W of the sequence; its final state covers steps
            // len - W' .. len.
            let w_out = grad_final.shape(1);
            if i + w_out >= w_in + len {
                let (f_at, f_step) = origin(grad_final, seq, cv, n);
                g = grad_final[f_at + (i + w_out - w_in - len) * f_step];
            }
        }
        grad_initial[h_at + i * h_step] = g;
    }
}

/// `grad_weight[c, j]` `[C, K]` and `grad_bias[c]` `[C]` from the contiguous partial sums
/// `[R, K (+ 1), C]`: one thread per weight, adding up its `R` partial sums.
// The bias's `if let` is decided when the kernel compiles, the `if` inside it when it runs: they
// do not collapse into one.
#[allow(clippy::collapsible_if)]
#[cube(launch_unchecked, address_type = "dynamic")]
fn reduce_kernel<E: Float>(
    partial: &Tensor<f32>,
    grad_weight: &mut Tensor<E>,
    grad_bias: ComptimeOption<&mut Tensor<E>>,
    #[define(E)] _dtype: ElemType,
) {
    let runs = partial.shape(0);
    let channels = partial.shape(2);
    let lanes = partial.shape(1) * channels;
    // Threads run along the channels of one tap, so the reads of a row coalesce.
    let lane = ABSOLUTE_POS;
    if lane >= lanes {
        terminate!();
    }
    let mut sum = 0.0f32;
    for r in 0..runs {
        sum += partial[r * lanes + lane];
    }
    let j = lane / channels;
    let c = lane % channels;
    let width = grad_weight.shape(1);
    if j < width {
        let at = c * grad_weight.stride(0) + j * grad_weight.stride(1);
        grad_weight[at] = E::cast_from(sum);
    }
    #[comptime]
    if let ComptimeOption::Some(grad_bias) = grad_bias {
        if j == width {
            let at = c * grad_bias.stride(0);
            grad_bias[at] = E::cast_from(sum);
        }
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn update_kernel<E: Float, N: Size>(
    x: &Tensor<Vector<E, N>>,
    state: &Tensor<Vector<f32, N>>,
    weight: &Tensor<E>,
    bias: ComptimeOption<&Tensor<E>>,
    out: &mut Tensor<Vector<E, N>>,
    next: &mut Tensor<Vector<f32, N>>,
    #[comptime] width: usize,
    #[comptime] activation: Activation,
    #[define(E)] _dtype: ElemType,
) {
    let n = x.vector_size();
    let row = x.shape(1) / n;
    let lane = ABSOLUTE_POS;
    if lane >= x.shape(0) * row {
        terminate!();
    }
    let b = lane / row;
    let cv = lane % row;
    let c0 = cv * n;
    let len = state.shape(1);
    let (s_at, s_step) = origin(state, b, cv, n);
    let n_at = b * (next.stride(0) / n) + cv;
    let n_step = next.stride(1) / n;
    let taps = taps::<E, N>(weight, c0, width);
    #[comptime]
    let shift = match bias {
        ComptimeOption::Some(bias) => channels::<E, N>(bias, c0),
        ComptimeOption::None => Vector::<f32, N>::zeroed(),
    };
    let token = Vector::<f32, N>::cast_from(x[b * (x.stride(0) / n) + cv]);

    // The state's last K - 1 steps (len >= K - 1), then the token.
    let mut window = Array::<Vector<f32, N>>::new(width);
    #[unroll]
    for j in 0..comptime![width - 1] {
        window[j] = state[s_at + (len + 1 + j - width) * s_step];
    }
    window[comptime![width - 1]] = token;
    let y = convolve::<N>(&taps, &window, shift, width);
    let out_at = b * (out.stride(0) / n) + cv;
    out[out_at] = Vector::cast_from(activate::<N>(y, activation));

    // The next state: the state without its oldest step, then the token.
    if len > 0 {
        for i in 1..len {
            next[n_at + (i - 1) * n_step] = state[s_at + i * s_step];
        }
        next[n_at + (len - 1) * n_step] = token;
    }
}

/// The widest vector all `tensors` can be read in along their last axis, the channels.
fn vector_size(tensors: &[&CubeTensor]) -> usize {
    tensors
        .iter()
        .map(|tensor| {
            tensor_vector_size_parallel(
                tensor.client.io_optimized_vector_sizes(tensor.dtype.size()),
                tensor.meta.shape(),
                tensor.meta.strides(),
                tensor.meta.shape().num_dims() - 1,
            )
        })
        .min()
        .unwrap_or(1)
        .min(LANES)
}

fn address_type(tensors: &[&CubeTensor]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}

fn cubes(units: usize) -> CubeCount {
    CubeCount::Static(units.div_ceil(CUBE as usize) as u32, 1, 1)
}

/// A tensor the kernels can read: laid out in rows, with a unit stride along its last axis, the
/// channels. A tensor that already is one is used as it is, padded rows and all.
fn channels_last(tensor: CubeTensor) -> CubeTensor {
    let tensor = untile(tensor);
    let last = tensor.meta.shape().num_dims() - 1;
    match tensor.meta.strides()[last] {
        1 => tensor,
        _ => into_contiguous(tensor),
    }
}

/// What the launches of a pass ask of the device, each of `units` threads binding `tensors`
/// tensors, indexed with `address`: cubes of 256 units, as many as the threads need along x; no
/// shared memory and no plane operations; with 64-bit `cu_seqlens`, 64-bit integers to convert
/// them. Vectors of up to [`LANES`] channels, of a width the device offers
/// (`io_optimized_vector_sizes`), so none it lacks. A launch's threads are counted with one
/// channel each, the most it could need.
fn requirements(launches: &[(usize, usize)], address: AddressType, int64: bool) -> Requirements {
    launches
        .iter()
        .map(|&(units, tensors)| {
            let cubes = units.div_ceil(CUBE as usize);
            Requirements {
                address,
                int64,
                ..Requirements::new((CUBE, 1), (cubes, 1), tensors as u32)
            }
        })
        .reduce(Requirements::max)
        .expect("a launch")
}

/// Whether `dtype`, of `cu_seqlens`, is 64 bits wide.
fn wide(dtype: DType) -> bool {
    matches!(dtype, DType::I64 | DType::U64)
}

/// Whether the device of `inputs[0]` takes `launches` (threads and tensors bound, each), indexed
/// as `inputs` need, `inputs[0]`'s dtype, f32 and the `cu_seqlens` dtype `int` read and written.
fn fits(launches: &[(usize, usize)], inputs: &[&CubeTensor], int: Option<DType>) -> bool {
    let mut dtypes = vec![inputs[0].dtype, DType::F32];
    dtypes.extend(int);
    requirements(launches, address_type(inputs), int.is_some_and(wide))
        .met_on(&inputs[0].client, &dtypes)
}

/// Whether the kernels take the forward pass, the backward pass and a decoding step on the
/// device `client` serves for `x` `[B, T, C]`, a kernel of `width` taps and inputs in `dtype`,
/// with every state and `packed` sequences (their count and `cu_seqlens` dtype), counting the
/// most threads and tensors of the kernels (the backward kernel's over sequences: ten).
pub(super) fn kernels_take(
    client: &Client,
    dtype: DType,
    width: usize,
    shape: [usize; 3],
    packed: Option<(usize, DType)>,
) -> bool {
    let [batch, steps, channels] = shape;
    let sequences = packed.map_or(batch, |(sequences, _)| sequences);
    let units = batch * steps.div_ceil(TILE) * channels;
    let address = AddressType::from_len(batch * steps * channels);
    let int = packed.map(|(_, int)| int);
    let mut dtypes = vec![dtype, DType::F32];
    dtypes.extend(int);
    (1..=MAX_KERNEL_WIDTH).contains(&width)
        && matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
        && requirements(
            &[(units, 10), (sequences * channels, 4)],
            address,
            int.is_some_and(wide),
        )
        .met_on(client, &dtypes)
}

/// Whether the kernels take these inputs: a kernel width from 1 to [`MAX_KERNEL_WIDTH`], one
/// float dtype of f32, bf16 and f16 for the weight, the bias and the `others`, and f32 `states`
/// with at least one step.
fn kernels_apply(
    weight: &CubeTensor,
    bias: &Bias<CubeBackend>,
    others: &[&CubeTensor],
    states: &[Option<&CubeTensor>],
) -> bool {
    let width = weight.meta.shape()[1];
    let dtype = weight.dtype;
    (1..=MAX_KERNEL_WIDTH).contains(&width)
        && matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
        && others.iter().all(|tensor| tensor.dtype == dtype)
        && match bias {
            Bias::None => true,
            Bias::Some(bias) => bias.dtype == dtype,
        }
        && states
            .iter()
            .flatten()
            .all(|state| state.dtype == DType::F32 && state.meta.shape()[1] > 0)
}

/// Whether a call over `x` `[B, T, C]` has the layout the kernels index by: packed, `B = 1` and
/// at least one sequence; every state `[S, W, C]` with `S` its sequences. The public functions
/// check this, and the reference panics on what they reject; the kernels, unchecked, would read
/// out of their tensors.
fn layout_agrees(
    x: &CubeTensor,
    cu_seqlens: Option<&CubeTensor>,
    states: &[Option<&CubeTensor>],
) -> bool {
    let [batch, _, channels] = x.meta.shape().dims::<3>();
    let sequences = match cu_seqlens {
        None => batch,
        Some(cu) if batch == 1 && cu.meta.shape().num_dims() == 1 && cu.meta.shape()[0] >= 2 => {
            cu.meta.shape()[0] - 1
        }
        Some(_) => return false,
    };
    states.iter().flatten().all(|state| {
        let shape = state.meta.shape();
        shape.num_dims() == 3 && shape[0] == sequences && shape[2] == channels
    })
}

/// The element type the kernels read `cu_seqlens` as (any, when there are none).
fn int_type(cu_seqlens: Option<&CubeTensor>) -> ElemType {
    dtype_to_storage_type(cu_seqlens.map_or(DType::U32, |cu| cu.dtype))
}

fn empty(reference: &CubeTensor, shape: Shape, dtype: DType) -> CubeTensor {
    empty_device_contiguous_dtype(
        reference.client.clone(),
        reference.device.clone(),
        shape,
        dtype,
    )
}

impl CausalConv1dBackend for CubeBackend {
    fn causal_conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Bias<Self>,
        initial_state: State<Self>,
        sequences: Sequences<Self>,
        activation: Activation,
        final_state_len: Option<usize>,
    ) -> (FloatTensor<Self>, State<Self>) {
        let [batch, steps, channels] = x.shape().dims();
        let initial = initial_state.into_option();
        let cu = sequences.into_option();
        let count = cu.as_ref().map_or(batch, |cu| cu.shape()[0] - 1);
        let units = batch * steps.div_ceil(TILE) * channels;
        let tensors = 3 + usize::from(matches!(bias, Bias::Some(_)));
        let tensors = tensors + usize::from(initial.is_some()) + usize::from(cu.is_some());
        let mut launches = vec![(units, tensors)];
        launches.extend(final_state_len.map(|_| (count * channels, 4)));
        let mut inputs = vec![&x, &weight];
        inputs.extend(initial.as_ref());
        if let Bias::Some(bias) = &bias {
            inputs.push(bias);
        }
        if !kernels_apply(&weight, &bias, &[&x], &[initial.as_ref()])
            || !layout_agrees(&x, cu.as_ref(), &[initial.as_ref()])
            || batch * steps * channels == 0
            || final_state_len == Some(0)
            || !fits(&launches, &inputs, cu.as_ref().map(|cu| cu.dtype))
        {
            return reference::forward::<Self>(
                x,
                weight,
                bias,
                State::from(initial),
                Sequences::from(cu),
                activation,
                final_state_len,
            );
        }
        let x = channels_last(x);
        let weight = untile(weight);
        let bias = bias.into_option().map(untile);
        let initial = initial.map(channels_last);
        let cu = cu.map(into_contiguous);
        let width = weight.shape().dims::<2>()[1];
        let out = empty(&x, x.shape(), x.dtype);
        let final_state =
            final_state_len.map(|len| empty(&x, Shape::new([count, len, channels]), DType::F32));
        let mut vectors = vec![&x, &out];
        vectors.extend(initial.as_ref());
        vectors.extend(final_state.as_ref());
        let n = vector_size(&vectors);
        let units = batch * steps.div_ceil(TILE) * (channels / n);
        let mut tensors = vec![&x, &weight, &out];
        tensors.extend(bias.as_ref());
        tensors.extend(initial.as_ref());
        tensors.extend(final_state.as_ref());
        let address = address_type(&tensors);
        let client = x.client.clone();
        let dtype = dtype_to_storage_type(x.dtype);
        let int = int_type(cu.as_ref());
        probe::FORWARD_LAUNCHES.fetch_add(1, Ordering::SeqCst);
        if initial.is_none() && cu.is_none() {
            unsafe {
                forward_kernel::launch_unchecked(
                    &client,
                    cubes(units),
                    CubeDim::new_1d(CUBE),
                    address,
                    n,
                    x.clone().into_tensor_arg(),
                    weight.into_tensor_arg(),
                    bias.map(|bias| bias.into_tensor_arg()).into(),
                    out.clone().into_tensor_arg(),
                    width,
                    TILE,
                    activation,
                    dtype,
                );
            }
        } else {
            probe::SEQUENCE_LAUNCHES.fetch_add(1, Ordering::SeqCst);
            unsafe {
                forward_sequences_kernel::launch_unchecked(
                    &client,
                    cubes(units),
                    CubeDim::new_1d(CUBE),
                    address,
                    n,
                    x.clone().into_tensor_arg(),
                    weight.into_tensor_arg(),
                    bias.map(|bias| bias.into_tensor_arg()).into(),
                    initial.clone().map(|state| state.into_tensor_arg()).into(),
                    cu.clone().map(|cu| cu.into_tensor_arg()).into(),
                    out.clone().into_tensor_arg(),
                    width,
                    TILE,
                    activation,
                    [dtype, int],
                );
            }
        }
        if let Some(final_state) = &final_state {
            probe::STATE_LAUNCHES.fetch_add(1, Ordering::SeqCst);
            unsafe {
                final_state_kernel::launch_unchecked(
                    &client,
                    cubes(count * (channels / n)),
                    CubeDim::new_1d(CUBE),
                    address,
                    n,
                    x.into_tensor_arg(),
                    initial.map(|state| state.into_tensor_arg()).into(),
                    cu.map(|cu| cu.into_tensor_arg()).into(),
                    final_state.clone().into_tensor_arg(),
                    [dtype, int],
                );
            }
        }
        (out, State::from(final_state))
    }

    fn causal_conv1d_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Bias<Self>,
        initial_state: State<Self>,
        sequences: Sequences<Self>,
        grad: FloatTensor<Self>,
        grad_final_state: State<Self>,
        activation: Activation,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        Bias<Self>,
        State<Self>,
    ) {
        let [batch, steps, channels] = x.shape().dims();
        let initial = initial_state.into_option();
        let grad_final = grad_final_state.into_option();
        let cu = sequences.into_option();
        let count = cu.as_ref().map_or(batch, |cu| cu.shape()[0] - 1);
        // The backward kernel's threads and tensors, more than the reduction's.
        let units = batch * steps.div_ceil(TILE) * channels;
        let tensors = 5 + usize::from(matches!(bias, Bias::Some(_)));
        let tensors = tensors
            + 2 * usize::from(initial.is_some())
            + usize::from(cu.is_some())
            + usize::from(grad_final.is_some());
        let mut launches = vec![(units, tensors)];
        launches.extend(initial.as_ref().map(|_| (count * channels, 3)));
        let mut inputs = vec![&x, &weight, &grad];
        inputs.extend(initial.as_ref());
        inputs.extend(grad_final.as_ref());
        if let Bias::Some(bias) = &bias {
            inputs.push(bias);
        }
        if !kernels_apply(
            &weight,
            &bias,
            &[&x, &grad],
            &[initial.as_ref(), grad_final.as_ref()],
        ) || !layout_agrees(&x, cu.as_ref(), &[initial.as_ref(), grad_final.as_ref()])
            || batch * steps * channels == 0
            || !fits(&launches, &inputs, cu.as_ref().map(|cu| cu.dtype))
        {
            return reference::backward::<Self>(
                x,
                weight,
                bias,
                State::from(initial),
                Sequences::from(cu),
                grad,
                State::from(grad_final),
                activation,
            );
        }
        let x = channels_last(x);
        let grad = channels_last(grad);
        let weight = untile(weight);
        let bias = bias.into_option().map(untile);
        let initial = initial.map(channels_last);
        let grad_final = grad_final.map(channels_last);
        let cu = cu.map(into_contiguous);
        let width = weight.shape().dims::<2>()[1];
        let client = x.client.clone();
        let runs = steps.div_ceil(TILE);
        let rows = width + usize::from(bias.is_some());
        let grad_x = empty(&x, x.shape(), x.dtype);
        let grad_weight = empty(&x, Shape::new([channels, width]), weight.dtype);
        let grad_bias = bias
            .as_ref()
            .map(|bias| empty(&x, Shape::new([channels]), bias.dtype));
        let partial = empty(&x, Shape::new([batch * runs, rows, channels]), DType::F32);
        let grad_initial = initial
            .as_ref()
            .map(|state| empty(&x, state.shape(), DType::F32));
        let mut vectors = vec![&x, &grad, &grad_x, &partial];
        vectors.extend(initial.as_ref());
        vectors.extend(grad_final.as_ref());
        vectors.extend(grad_initial.as_ref());
        let n = vector_size(&vectors);
        let units = batch * runs * (channels / n);
        let mut tensors = vec![&x, &weight, &grad, &grad_x, &partial];
        tensors.extend(bias.as_ref());
        tensors.extend(initial.as_ref());
        tensors.extend(grad_final.as_ref());
        tensors.extend(grad_initial.as_ref());
        let address = address_type(&tensors);
        let mut outputs = vec![&partial, &grad_weight];
        outputs.extend(grad_bias.as_ref());
        let reduce_address = address_type(&outputs);
        let dtype = dtype_to_storage_type(x.dtype);
        let int = int_type(cu.as_ref());
        let plain = initial.is_none() && cu.is_none() && grad_final.is_none();
        probe::BACKWARD_LAUNCHES.fetch_add(2, Ordering::SeqCst);
        if plain {
            unsafe {
                backward_kernel::launch_unchecked(
                    &client,
                    cubes(units),
                    CubeDim::new_1d(CUBE),
                    address,
                    n,
                    x.into_tensor_arg(),
                    weight.into_tensor_arg(),
                    bias.map(|bias| bias.into_tensor_arg()).into(),
                    grad.into_tensor_arg(),
                    grad_x.clone().into_tensor_arg(),
                    partial.clone().into_tensor_arg(),
                    width,
                    TILE,
                    activation,
                    dtype,
                );
            }
        } else {
            probe::SEQUENCE_LAUNCHES.fetch_add(1, Ordering::SeqCst);
            unsafe {
                backward_sequences_kernel::launch_unchecked(
                    &client,
                    cubes(units),
                    CubeDim::new_1d(CUBE),
                    address,
                    n,
                    x.into_tensor_arg(),
                    weight.into_tensor_arg(),
                    bias.map(|bias| bias.into_tensor_arg()).into(),
                    initial.map(|state| state.into_tensor_arg()).into(),
                    cu.clone().map(|cu| cu.into_tensor_arg()).into(),
                    grad.into_tensor_arg(),
                    grad_final.clone().map(|grad| grad.into_tensor_arg()).into(),
                    grad_x.clone().into_tensor_arg(),
                    grad_initial
                        .clone()
                        .unwrap_or_else(|| empty(&partial, Shape::new([1, 1, n]), DType::F32))
                        .into_tensor_arg(),
                    partial.clone().into_tensor_arg(),
                    width,
                    TILE,
                    activation,
                    [dtype, int],
                );
            }
        }
        unsafe {
            reduce_kernel::launch_unchecked(
                &client,
                cubes(rows * channels),
                CubeDim::new_1d(CUBE),
                reduce_address,
                partial.into_tensor_arg(),
                grad_weight.clone().into_tensor_arg(),
                grad_bias
                    .as_ref()
                    .map(|grad_bias| grad_bias.clone().into_tensor_arg())
                    .into(),
                dtype_to_storage_type(grad_weight.dtype),
            );
        }
        if let Some(grad_initial) = &grad_initial {
            probe::STATE_LAUNCHES.fetch_add(1, Ordering::SeqCst);
            unsafe {
                initial_state_grad_kernel::launch_unchecked(
                    &client,
                    cubes(count * (channels / n)),
                    CubeDim::new_1d(CUBE),
                    address,
                    n,
                    cu.map(|cu| cu.into_tensor_arg()).into(),
                    grad_final.map(|grad| grad.into_tensor_arg()).into(),
                    grad_initial.clone().into_tensor_arg(),
                    steps as u32,
                    width,
                    int,
                );
            }
        }
        (
            grad_x,
            grad_weight,
            Bias::from(grad_bias),
            State::from(grad_initial),
        )
    }

    fn causal_conv1d_update(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Bias<Self>,
        state: FloatTensor<Self>,
        activation: Activation,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let [batch, channels] = x.shape().dims();
        let mut inputs = vec![&x, &state, &weight];
        if let Bias::Some(bias) = &bias {
            inputs.push(bias);
        }
        let tensors = 5 + usize::from(matches!(bias, Bias::Some(_)));
        if !kernels_apply(&weight, &bias, &[&x], &[])
            || state.dtype != DType::F32
            || batch * channels == 0
            || !fits(&[(batch * channels, tensors)], &inputs, None)
        {
            return reference::update::<Self>(x, weight, bias, state, activation);
        }
        let (x, state) = (channels_last(x), channels_last(state));
        let weight = untile(weight);
        let bias = bias.into_option().map(untile);
        let width = weight.shape().dims::<2>()[1];
        let client = x.client.clone();
        let out = empty(&x, x.shape(), x.dtype);
        let next = empty(&x, state.shape(), DType::F32);
        let n = vector_size(&[&x, &state, &out, &next]);
        let mut tensors = vec![&x, &state, &weight, &out, &next];
        tensors.extend(bias.as_ref());
        let address = address_type(&tensors);
        let dtype = dtype_to_storage_type(x.dtype);
        probe::UPDATE_LAUNCHES.fetch_add(1, Ordering::SeqCst);
        unsafe {
            update_kernel::launch_unchecked(
                &client,
                cubes(batch * (channels / n)),
                CubeDim::new_1d(CUBE),
                address,
                n,
                x.into_tensor_arg(),
                state.into_tensor_arg(),
                weight.into_tensor_arg(),
                bias.map(|bias| bias.into_tensor_arg()).into(),
                out.clone().into_tensor_arg(),
                next.clone().into_tensor_arg(),
                width,
                activation,
                dtype,
            );
        }
        (out, next)
    }
}
