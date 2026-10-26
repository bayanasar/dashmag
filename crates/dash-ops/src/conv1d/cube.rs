//! The CUDA kernels, written in CubeCL for kernel width 4.
//!
//! The operation is memory-bound: four multiply-adds per element against a read and a write.
//! So each thread owns a run of [`TILE`] consecutive steps of one or two neighbouring channels
//! (one vector load), walks it in order, and keeps the window of the last three inputs (and, in
//! the backward pass, of the last three gradients) in registers: every element is read once,
//! plus a three-step halo per run, and neighbouring threads read neighbouring channels, which
//! coalesces in the `[B, T, C]` layout.
//!
//! The weight gradient is a reduction over `B x T` per weight. The backward kernel leaves one
//! partial sum per run (`[B * runs, 4, C]`, f32), and a second kernel adds them up per weight,
//! so the long reduction is spread over every thread of the grid rather than walked down by one
//! thread per weight.
//!
//! Every tensor is indexed through its own strides: CubeCL may pad rows (a pitched layout), so a
//! tensor's strides are not implied by its shape. The sequence kernels read vectors of up to
//! [`LANES`] channels, which needs a unit channel stride and outer strides that are multiples of
//! the vector width; the launchers pick a width that holds for every tensor involved.

use super::{CausalConv1dBackend, KERNEL_WIDTH, counters, reference};
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::tensor::FloatTensor;
use burn::backend::{DType, Shape, TensorMetadata};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::{into_contiguous, untile};
use burn_cubecl::ops::numeric::{empty_device_contiguous_dtype, empty_device_dtype};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use cubecl::tensor_vector_size_parallel;
use std::sync::atomic::Ordering;

/// Steps one thread walks in the sequence kernels: the halo it reads twice is 3 steps in 64.
const TILE: usize = 64;
/// Channels one thread carries, at most. Each lane holds four taps and a three-step window (six
/// more values in the backward pass) in f32 registers, so wider vectors cost threads, and they
/// measured slower: at Qwen3.5-0.8B's width (one sequence of 6144 steps, 6144 channels, bf16) on
/// an RTX 5070 Ti, the forward kernel takes 203 us with 2 lanes, 220 with 4 and 234 with 8, and
/// the backward kernel 335, 372 and 342 (with 2, 8 and 4); in f32 the width makes no difference.
const LANES: usize = 2;
/// Threads per cube.
const CUBE: u32 = 256;

/// Tap `j` of `weight` `[C, 4]` for the vector of channels starting at `c0`.
#[cube]
fn tap<E: Float, N: Size>(weight: &Tensor<E>, c0: usize, #[comptime] j: usize) -> Vector<f32, N> {
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

/// SiLU, `y / (1 + exp(-y))`, which gives 0 rather than NaN where `exp(-y)` overflows.
#[cube]
fn activate<N: Size>(y: Vector<f32, N>, #[comptime] silu: bool) -> Vector<f32, N> {
    let mut out = y;
    if silu {
        out = y / (Vector::<f32, N>::new(1.0f32) + (-y).exp());
    }
    out
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
    out: &mut Tensor<Vector<E, N>>,
    #[comptime] tile: usize,
    #[comptime] silu: bool,
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
    let w0 = tap::<E, N>(weight, c0, 0usize);
    let w1 = tap::<E, N>(weight, c0, 1usize);
    let w2 = tap::<E, N>(weight, c0, 2usize);
    let w3 = tap::<E, N>(weight, c0, 3usize);
    let (x_at, x_step) = origin(x, b, cv, n);
    let out_step = out.stride(1) / n;
    let out_at = b * (out.stride(0) / n) + cv;

    let mut xm3 = Vector::<f32, N>::zeroed();
    let mut xm2 = Vector::<f32, N>::zeroed();
    let mut xm1 = Vector::<f32, N>::zeroed();
    // Runs start at multiples of the tile, so a run that does not start the sequence has three
    // steps before it.
    if t0 > 0 {
        xm3 = Vector::cast_from(x[x_at + (t0 - 3) * x_step]);
        xm2 = Vector::cast_from(x[x_at + (t0 - 2) * x_step]);
        xm1 = Vector::cast_from(x[x_at + (t0 - 1) * x_step]);
    }

    let end = clamp_max(t0 + tile, steps);
    for t in t0..end {
        let x0 = Vector::<f32, N>::cast_from(x[x_at + t * x_step]);
        let y = w0 * xm3 + w1 * xm2 + w2 * xm1 + w3 * x0;
        out[out_at + t * out_step] = Vector::cast_from(activate::<N>(y, silu));
        xm3 = xm2;
        xm2 = xm1;
        xm1 = x0;
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn backward_kernel<E: Float, N: Size>(
    x: &Tensor<Vector<E, N>>,
    weight: &Tensor<E>,
    grad: &Tensor<Vector<E, N>>,
    grad_x: &mut Tensor<Vector<E, N>>,
    partial: &mut Tensor<Vector<f32, N>>,
    #[comptime] tile: usize,
    #[comptime] silu: bool,
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
    let w0 = tap::<E, N>(weight, c0, 0usize);
    let w1 = tap::<E, N>(weight, c0, 1usize);
    let w2 = tap::<E, N>(weight, c0, 2usize);
    let w3 = tap::<E, N>(weight, c0, 3usize);
    let one = Vector::<f32, N>::new(1.0f32);
    let (x_at, x_step) = origin(x, b, cv, n);
    let (grad_at, grad_step) = origin(grad, b, cv, n);
    let gx_step = grad_x.stride(1) / n;
    let gx_at = b * (grad_x.stride(0) / n) + cv;

    let mut xm3 = Vector::<f32, N>::zeroed();
    let mut xm2 = Vector::<f32, N>::zeroed();
    let mut xm1 = Vector::<f32, N>::zeroed();
    if t0 > 0 {
        xm3 = Vector::cast_from(x[x_at + (t0 - 3) * x_step]);
        xm2 = Vector::cast_from(x[x_at + (t0 - 2) * x_step]);
        xm1 = Vector::cast_from(x[x_at + (t0 - 1) * x_step]);
    }
    // The gradients of the pre-activation at s - 1, s - 2, s - 3.
    let mut gm1 = Vector::<f32, N>::zeroed();
    let mut gm2 = Vector::<f32, N>::zeroed();
    let mut gm3 = Vector::<f32, N>::zeroed();
    // This run's share of the weight gradient, tap by tap.
    let mut a0 = Vector::<f32, N>::zeroed();
    let mut a1 = Vector::<f32, N>::zeroed();
    let mut a2 = Vector::<f32, N>::zeroed();
    let mut a3 = Vector::<f32, N>::zeroed();

    // Input step t feeds output steps t .. t + 3, so the run's input gradients need the output
    // gradients three steps past its end; past the sequence they are zero.
    for s in t0..t0 + tile + 3 {
        let mut xs = Vector::<f32, N>::zeroed();
        let mut gs = Vector::<f32, N>::zeroed();
        if s < steps {
            xs = Vector::cast_from(x[x_at + s * x_step]);
            gs = Vector::cast_from(grad[grad_at + s * grad_step]);
            if silu {
                // SiLU'(y) = sigma (1 + y (1 - sigma)), the pre-activation recomputed from x.
                let y = w0 * xm3 + w1 * xm2 + w2 * xm1 + w3 * xs;
                let sigma = one / (one + (-y).exp());
                gs = gs * sigma * (one + y * (one - sigma));
            }
        }
        if s < t0 + tile {
            a0 += gs * xm3;
            a1 += gs * xm2;
            a2 += gs * xm1;
            a3 += gs * xs;
        }
        if s >= t0 + 3 {
            let t = s - 3;
            if t < steps {
                grad_x[gx_at + t * gx_step] =
                    Vector::cast_from(w0 * gs + w1 * gm1 + w2 * gm2 + w3 * gm3);
            }
        }
        xm3 = xm2;
        xm2 = xm1;
        xm1 = xs;
        gm3 = gm2;
        gm2 = gm1;
        gm1 = gs;
    }

    // partial[b * runs + run, j, cv], contiguous, in vectors.
    let at = (b * runs + run) * 4 * row + cv;
    partial[at] = a0;
    partial[at + row] = a1;
    partial[at + 2 * row] = a2;
    partial[at + 3 * row] = a3;
}

/// `grad_weight[c, j]` `[C, 4]` from the contiguous partial sums `[R, 4, C]`: one thread per
/// weight.
#[cube(launch_unchecked, address_type = "dynamic")]
fn weight_grad_kernel<E: Float>(
    partial: &Tensor<f32>,
    grad_weight: &mut Tensor<E>,
    #[define(E)] _dtype: ElemType,
) {
    let rows = partial.shape(0);
    let channels = partial.shape(2);
    let lanes = 4 * channels;
    // Threads run along the channels of one tap, so the reads of a row coalesce.
    let lane = ABSOLUTE_POS;
    if lane >= lanes {
        terminate!();
    }
    let mut sum = 0.0f32;
    for r in 0..rows {
        sum += partial[r * lanes + lane];
    }
    let j = lane / channels;
    let c = lane % channels;
    let at = c * grad_weight.stride(0) + j * grad_weight.stride(1);
    grad_weight[at] = E::cast_from(sum);
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn update_kernel<E: Float>(
    x: &Tensor<E>,
    state: &Tensor<E>,
    weight: &Tensor<E>,
    out: &mut Tensor<E>,
    next: &mut Tensor<E>,
    #[comptime] silu: bool,
    #[define(E)] _dtype: ElemType,
) {
    let channels = x.shape(1);
    let lane = ABSOLUTE_POS;
    if lane >= x.shape(0) * channels {
        terminate!();
    }
    let b = lane / channels;
    let c = lane % channels;
    let len = state.shape(2);
    let s_at = b * state.stride(0) + c * state.stride(1);
    let s_step = state.stride(2);
    let n_at = b * next.stride(0) + c * next.stride(1);
    let n_step = next.stride(2);
    let w_at = c * weight.stride(0);
    let w_step = weight.stride(1);

    let new = x[b * x.stride(0) + c * x.stride(1)];
    let y = f32::cast_from(weight[w_at]) * f32::cast_from(state[s_at + (len - 3) * s_step])
        + f32::cast_from(weight[w_at + w_step]) * f32::cast_from(state[s_at + (len - 2) * s_step])
        + f32::cast_from(weight[w_at + 2 * w_step])
            * f32::cast_from(state[s_at + (len - 1) * s_step])
        + f32::cast_from(weight[w_at + 3 * w_step]) * f32::cast_from(new);
    let mut value = y;
    if silu {
        value = y / (1.0f32 + (-y).exp());
    }
    let out_at = b * out.stride(0) + c * out.stride(1);
    out[out_at] = E::cast_from(value);
    for j in 1..len {
        next[n_at + (j - 1) * n_step] = state[s_at + j * s_step];
    }
    next[n_at + (len - 1) * n_step] = new;
}

/// The widest vector all `tensors` can be read in along `axis`.
fn vector_size(tensors: &[&CubeTensor], axis: usize) -> usize {
    tensors
        .iter()
        .map(|tensor| {
            tensor_vector_size_parallel(
                tensor.client.io_optimized_vector_sizes(tensor.dtype.size()),
                tensor.meta.shape(),
                tensor.meta.strides(),
                axis,
            )
        })
        .min()
        .unwrap_or(1)
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

/// A `[B, T, C]` tensor the sequence kernels can read: laid out in rows, with a unit channel
/// stride. A tensor that already is one is used as it is, padded rows and all.
fn channels_last(tensor: CubeTensor) -> CubeTensor {
    let tensor = untile(tensor);
    match tensor.meta.strides()[2] {
        1 => tensor,
        _ => into_contiguous(tensor),
    }
}

/// Whether the kernels apply: kernel width 4 and one dtype for all inputs.
fn kernels_apply(weight: &CubeTensor, others: &[&CubeTensor]) -> bool {
    weight.shape().dims::<2>()[1] == KERNEL_WIDTH
        && others.iter().all(|tensor| tensor.dtype == weight.dtype)
}

impl CausalConv1dBackend for CubeBackend {
    fn causal_conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        silu: bool,
    ) -> FloatTensor<Self> {
        if !kernels_apply(&weight, &[&x]) {
            return reference::forward::<Self>(x, weight, silu);
        }
        let x = channels_last(x);
        let weight = untile(weight);
        let [batch, steps, channels] = x.shape().dims();
        let out = empty_device_dtype(x.client.clone(), x.device.clone(), x.shape(), x.dtype);
        if batch * steps * channels == 0 {
            return out;
        }
        let n = vector_size(&[&x, &out], 2).min(LANES);
        let units = batch * steps.div_ceil(TILE) * (channels / n);
        let address = address_type(&[&x, &weight, &out]);
        let (client, dtype) = (x.client.clone(), dtype_to_storage_type(x.dtype));
        counters::FORWARD_KERNEL.fetch_add(1, Ordering::SeqCst);
        unsafe {
            forward_kernel::launch_unchecked(
                &client,
                cubes(units),
                CubeDim::new_1d(CUBE),
                address,
                n,
                x.into_tensor_arg(),
                weight.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                TILE,
                silu,
                dtype,
            );
        }
        out
    }

    fn causal_conv1d_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        silu: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let [batch, steps, channels] = x.shape().dims();
        if !kernels_apply(&weight, &[&x, &grad]) || batch * steps * channels == 0 {
            return reference::backward::<Self>(x, weight, grad, silu);
        }
        let x = channels_last(x);
        let grad = channels_last(grad);
        let weight = untile(weight);
        let (client, device) = (x.client.clone(), x.device.clone());
        let runs = steps.div_ceil(TILE);
        let grad_x = empty_device_dtype(client.clone(), device.clone(), x.shape(), x.dtype);
        let grad_weight = empty_device_dtype(
            client.clone(),
            device.clone(),
            Shape::new([channels, KERNEL_WIDTH]),
            weight.dtype,
        );
        let partial = empty_device_contiguous_dtype(
            client.clone(),
            device,
            Shape::new([batch * runs, KERNEL_WIDTH, channels]),
            DType::F32,
        );
        let n = vector_size(&[&x, &grad, &grad_x, &partial], 2).min(LANES);
        let units = batch * runs * (channels / n);
        let address = address_type(&[&x, &weight, &grad, &grad_x, &partial]);
        let dtype = dtype_to_storage_type(x.dtype);
        counters::BACKWARD_KERNEL.fetch_add(1, Ordering::SeqCst);
        unsafe {
            backward_kernel::launch_unchecked(
                &client,
                cubes(units),
                CubeDim::new_1d(CUBE),
                address,
                n,
                x.into_tensor_arg(),
                weight.into_tensor_arg(),
                grad.into_tensor_arg(),
                grad_x.clone().into_tensor_arg(),
                partial.clone().into_tensor_arg(),
                TILE,
                silu,
                dtype,
            );
            let address = address_type(&[&partial, &grad_weight]);
            weight_grad_kernel::launch_unchecked(
                &client,
                cubes(KERNEL_WIDTH * channels),
                CubeDim::new_1d(CUBE),
                address,
                partial.into_tensor_arg(),
                grad_weight.clone().into_tensor_arg(),
                dtype_to_storage_type(grad_weight.dtype),
            );
        }
        (grad_x, grad_weight)
    }

    fn causal_conv1d_update(
        x: FloatTensor<Self>,
        state: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        silu: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let [batch, channels] = x.shape().dims();
        if !kernels_apply(&weight, &[&x, &state]) || batch * channels == 0 {
            return reference::update::<Self>(x, state, weight, silu);
        }
        let (x, state, weight) = (untile(x), untile(state), untile(weight));
        let (client, device) = (x.client.clone(), x.device.clone());
        let out = empty_device_dtype(client.clone(), device.clone(), x.shape(), x.dtype);
        let next = empty_device_dtype(client.clone(), device, state.shape(), state.dtype);
        let address = address_type(&[&x, &state, &weight, &out, &next]);
        let dtype = dtype_to_storage_type(x.dtype);
        counters::UPDATE_KERNEL.fetch_add(1, Ordering::SeqCst);
        unsafe {
            update_kernel::launch_unchecked(
                &client,
                cubes(batch * channels),
                CubeDim::new_1d(CUBE),
                address,
                x.into_tensor_arg(),
                state.into_tensor_arg(),
                weight.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                next.clone().into_tensor_arg(),
                silu,
                dtype,
            );
        }
        (out, next)
    }
}
