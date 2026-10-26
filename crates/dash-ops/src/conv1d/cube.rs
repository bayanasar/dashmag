//! The CUDA kernels, written in CubeCL for kernel widths 1 to [`MAX_KERNEL_WIDTH`].
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
    Activation, Bias, CausalConv1dBackend, CausalConv1dOptions, MAX_KERNEL_WIDTH, probe, reference,
};
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::tensor::FloatTensor;
use burn::backend::{DType, Shape, TensorMetadata};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::{into_contiguous, untile};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
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
fn slide<N: Size>(window: &mut Array<Vector<f32, N>>, #[comptime] width: usize) {
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
        slide::<N>(&mut window, width);
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
        slide::<N>(&mut window, width);
        slide::<N>(&mut grads, width);
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

/// Whether the kernels take these inputs: a kernel width from 1 to [`MAX_KERNEL_WIDTH`], and one
/// float dtype of f32, bf16 and f16 for the weight, the bias and the `others`.
fn kernels_apply(weight: &CubeTensor, bias: &Bias<CubeBackend>, others: &[&CubeTensor]) -> bool {
    let width = weight.meta.shape()[1];
    let dtype = weight.dtype;
    (1..=MAX_KERNEL_WIDTH).contains(&width)
        && matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
        && others.iter().all(|tensor| tensor.dtype == dtype)
        && match bias {
            Bias::None => true,
            Bias::Some(bias) => bias.dtype == dtype,
        }
}

impl CausalConv1dBackend for CubeBackend {
    fn causal_conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Bias<Self>,
        options: CausalConv1dOptions,
    ) -> FloatTensor<Self> {
        if !kernels_apply(&weight, &bias, &[&x]) {
            return reference::forward::<Self>(x, weight, bias, options);
        }
        let x = channels_last(x);
        let weight = untile(weight);
        let bias = bias.into_option().map(untile);
        let [batch, steps, channels] = x.shape().dims();
        let width = weight.shape().dims::<2>()[1];
        let out =
            empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), x.shape(), x.dtype);
        if batch * steps * channels == 0 {
            return out;
        }
        let n = vector_size(&[&x, &out]);
        let units = batch * steps.div_ceil(TILE) * (channels / n);
        let mut tensors = vec![&x, &weight, &out];
        tensors.extend(bias.as_ref());
        let address = address_type(&tensors);
        let (client, dtype) = (x.client.clone(), dtype_to_storage_type(x.dtype));
        probe::FORWARD_LAUNCHES.fetch_add(1, Ordering::SeqCst);
        unsafe {
            forward_kernel::launch_unchecked(
                &client,
                cubes(units),
                CubeDim::new_1d(CUBE),
                address,
                n,
                x.into_tensor_arg(),
                weight.into_tensor_arg(),
                bias.map(|bias| bias.into_tensor_arg()).into(),
                out.clone().into_tensor_arg(),
                width,
                TILE,
                options.activation,
                dtype,
            );
        }
        out
    }

    fn causal_conv1d_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Bias<Self>,
        grad: FloatTensor<Self>,
        options: CausalConv1dOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, Bias<Self>) {
        let [batch, steps, channels] = x.shape().dims();
        if !kernels_apply(&weight, &bias, &[&x, &grad]) || batch * steps * channels == 0 {
            return reference::backward::<Self>(x, weight, bias, grad, options);
        }
        let x = channels_last(x);
        let grad = channels_last(grad);
        let weight = untile(weight);
        let bias = bias.into_option().map(untile);
        let width = weight.shape().dims::<2>()[1];
        let (client, device) = (x.client.clone(), x.device.clone());
        let runs = steps.div_ceil(TILE);
        let rows = width + usize::from(bias.is_some());
        let grad_x =
            empty_device_contiguous_dtype(client.clone(), device.clone(), x.shape(), x.dtype);
        let grad_weight = empty_device_contiguous_dtype(
            client.clone(),
            device.clone(),
            Shape::new([channels, width]),
            weight.dtype,
        );
        let grad_bias = bias.as_ref().map(|bias| {
            empty_device_contiguous_dtype(
                client.clone(),
                device.clone(),
                Shape::new([channels]),
                bias.dtype,
            )
        });
        let partial = empty_device_contiguous_dtype(
            client.clone(),
            device,
            Shape::new([batch * runs, rows, channels]),
            DType::F32,
        );
        let n = vector_size(&[&x, &grad, &grad_x, &partial]);
        let units = batch * runs * (channels / n);
        let mut tensors = vec![&x, &weight, &grad, &grad_x, &partial];
        tensors.extend(bias.as_ref());
        let address = address_type(&tensors);
        let mut outputs = vec![&partial, &grad_weight];
        outputs.extend(grad_bias.as_ref());
        let reduce_address = address_type(&outputs);
        let dtype = dtype_to_storage_type(x.dtype);
        probe::BACKWARD_LAUNCHES.fetch_add(2, Ordering::SeqCst);
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
                options.activation,
                dtype,
            );
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
        (grad_x, grad_weight, Bias::from(grad_bias))
    }

    fn causal_conv1d_update(
        x: FloatTensor<Self>,
        state: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Bias<Self>,
        options: CausalConv1dOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let [batch, channels] = x.shape().dims();
        if !kernels_apply(&weight, &bias, &[&x])
            || state.dtype != DType::F32
            || batch * channels == 0
        {
            return reference::update::<Self>(x, state, weight, bias, options);
        }
        let (x, state) = (channels_last(x), channels_last(state));
        let weight = untile(weight);
        let bias = bias.into_option().map(untile);
        let width = weight.shape().dims::<2>()[1];
        let (client, device) = (x.client.clone(), x.device.clone());
        let out = empty_device_contiguous_dtype(client.clone(), device.clone(), x.shape(), x.dtype);
        let next = empty_device_contiguous_dtype(client.clone(), device, state.shape(), DType::F32);
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
                options.activation,
                dtype,
            );
        }
        (out, next)
    }
}
