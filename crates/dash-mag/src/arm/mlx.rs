//! The side on MLX: dash-mlx's operations and MLX's arrays, on Metal.
//!
//! An array is MLX's own, at any rank. MLX's GPU has no f64, so an operation that reads or makes f64
//! runs on the CPU ([`on`]); the operations with Metal kernels refuse it ([`no_f64`]).

use dash_mlx::mlx::error::Exception;
use dash_mlx::mlx::ops::indexing::{argmax_axis, slice};
use dash_mlx::mlx::ops::{
    broadcast_to, concatenate, copy, full, maximum, minimum, select, sigmoid, sort_axis,
    stop_gradient,
};
use dash_mlx::mlx::transforms::fallible_vjp;
use dash_mlx::mlx::{Array, Dtype, fast, memory, nn, transforms, version, with_cpu};
use half::{bf16, f16};

use super::{
    Arm, Binary, Compare, Drawn, Logical, Mask, Memory, Reduce, RopeAt, Unary, VjpFn, VjpOut,
};
use crate::ops::{
    AttentionOptions, CausalConv1dOptions, CausalConv1dStepOptions, ChunkGatedDeltaOptions,
    GatedDeltaStepOptions, LogprobOptions, LowRankOptions, SampleOptions,
};
use crate::{DType, Error, Result, refuse};

impl From<Exception> for Error {
    fn from(e: Exception) -> Self {
        Error(e.what().to_owned())
    }
}

fn mlx_dtype(dtype: DType) -> Dtype {
    match dtype {
        DType::Bool => Dtype::Bool,
        DType::I32 => Dtype::Int32,
        DType::I64 => Dtype::Int64,
        DType::F16 => Dtype::Float16,
        DType::BF16 => Dtype::Bfloat16,
        DType::F32 => Dtype::Float32,
        DType::F64 => Dtype::Float64,
    }
}

fn dims(shape: &[usize]) -> Vec<i32> {
    shape.iter().map(|&d| d as i32).collect()
}

/// `f`, on the CPU if any of `arrays` is f64.
fn on<T>(arrays: &[&Array], f: impl FnOnce() -> T) -> T {
    if arrays.iter().any(|a| a.dtype() == Dtype::Float64) {
        with_cpu(f)
    } else {
        f()
    }
}

/// `a` to read on the host: itself when its data is row-contiguous, so that the read waits for `a`
/// alone. A `contiguous` node would be scheduled behind all that is queued after `a`, which is what a
/// pipelined decoder has just queued.
fn readable(a: &Array) -> std::result::Result<Array, Exception> {
    if a.is_row_contiguous()? {
        return Ok(a.clone());
    }
    on(&[a], || a.contiguous())
}

/// Refuses f64 where the operation has Metal kernels, which stay on the GPU.
fn no_f64<'a>(op: &str, arrays: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    if arrays.into_iter().any(|a| a.dtype() == Dtype::Float64) {
        return refuse(format!("{op}: f64 on MLX, whose GPU has none"));
    }
    Ok(())
}

/// The output a cotangent is taken at, in the cotangent's dtype, so that MLX's cast of it on the way
/// back rounds the cotangent to the output's own. A node of its own even in that dtype: MLX skips an
/// output that is a `stop_gradient` without taking a cotangent for it, which would hand the next
/// output this one's, and it would keep one cotangent for an output returned twice. An output that
/// is not floating takes no part: a constant in its place uses up its cotangent.
fn seeded(out: &Array, cotangent: &Array) -> std::result::Result<Array, Exception> {
    if !out.dtype().is_float() {
        return on(&[cotangent], || copy(cotangent));
    }
    on(&[out, cotangent], || {
        if out.dtype() == cotangent.dtype() {
            copy(out)
        } else {
            out.as_dtype(cotangent.dtype())
        }
    })
}

/// `exp(x)` of f64, on the host. MLX's `simd::exp` works in `Simd<float, N>`, so its f64 comes back with
/// f32's error; its log and sqrt do not. The value is the host's `c`, and `c (x - stop_gradient(x) + 1)`
/// gives it its gradient, `c`, with no custom gradient, whose `stop_gradient` MLX runs on the default
/// device. An infinite `x` comes back NaN, from `inf - inf`.
fn exact_exp(x: &Array) -> std::result::Result<Array, Exception> {
    if x.size() == 0 {
        return Ok(x.clone());
    }
    let held = x.contiguous()?;
    let c: Vec<f64> = held
        .try_as_slice::<f64>()?
        .iter()
        .map(|v| v.exp())
        .collect();
    let one = Array::from_slice_f64(&[1.0], &[]);
    Array::from_slice_f64(&c, x.shape()).multiply(x.subtract(stop_gradient(x)?)?.add(one)?)
}

/// The definition in f64, on [`exact_exp`].
fn exact_sigmoid(x: &Array) -> std::result::Result<Array, Exception> {
    let one = Array::from_slice_f64(&[1.0], &[]);
    one.divide(exact_exp(&x.negative()?)?.add(&one)?)
}

pub(crate) struct Mlx;

impl Arm for Mlx {
    type Array = Array;

    const HAS_ROPE: bool = true;
    const LORA_AS_ROWS: bool = false;

    fn describe() -> String {
        format!("MLX {}, Metal", version().unwrap_or_default())
    }

    fn clear_cache() {
        let _ = memory::clear_cache();
    }

    fn from_bytes(bytes: &[u8], shape: &[usize], dtype: DType) -> Result<Array> {
        let shape = dims(shape);
        Ok(match dtype {
            DType::F16 => {
                let values: Vec<f16> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16::from_le_bytes(*b))
                    .collect();
                Array::from_slice(&values, &shape)
            }
            DType::BF16 => {
                let values: Vec<bf16> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16::from_le_bytes(*b))
                    .collect();
                Array::from_slice(&values, &shape)
            }
            DType::F32 => {
                let values: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                Array::from_slice(&values, &shape)
            }
            DType::F64 => {
                let values: Vec<f64> = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| f64::from_le_bytes(*b))
                    .collect();
                Array::from_slice_f64(&values, &shape)
            }
            other => return refuse(format!("{other:?} is not a floating dtype")),
        })
    }

    fn from_i32(values: &[i32], shape: &[usize]) -> Result<Array> {
        Ok(Array::from_slice(values, &dims(shape)))
    }

    fn from_bool(values: &[bool], shape: &[usize]) -> Result<Array> {
        Ok(Array::from_slice(values, &dims(shape)))
    }

    /// The value rounded once to `dtype`, on the host. A scalar stays there: MLX's `full` is a copy on
    /// the device, and `mul_scalar` and the like make scalars all the time.
    fn full(shape: &[usize], value: f64, dtype: DType) -> Result<Array> {
        if shape.is_empty() {
            return match dtype {
                DType::F16 => Ok(Array::from_slice(&[f16::from_f64(value)], &[])),
                DType::BF16 => Ok(Array::from_slice(&[bf16::from_f64(value)], &[])),
                DType::F32 => Ok(Array::from_f32(value as f32)),
                DType::F64 => Ok(Array::from_slice_f64(&[value], &[])),
                other => refuse(format!("{other:?} is not a floating dtype")),
            };
        }
        let shape = dims(shape);
        Ok(match dtype {
            DType::F16 => full::<f16>(&shape, Array::from_slice(&[f16::from_f64(value)], &[]))?,
            DType::BF16 => full::<bf16>(&shape, Array::from_slice(&[bf16::from_f64(value)], &[]))?,
            DType::F32 => full::<f32>(&shape, Array::from_f32(value as f32))?,
            DType::F64 => {
                let value = Array::from_slice_f64(&[value], &[]);
                with_cpu(|| full::<f64>(&shape, value))?
            }
            other => return refuse(format!("{other:?} is not a floating dtype")),
        })
    }

    fn shape(a: &Array) -> Vec<usize> {
        a.shape().iter().map(|&d| d as usize).collect()
    }

    fn dtype(a: &Array) -> DType {
        match a.dtype() {
            Dtype::Bool => DType::Bool,
            Dtype::Int32 => DType::I32,
            Dtype::Int64 => DType::I64,
            Dtype::Float16 => DType::F16,
            Dtype::Bfloat16 => DType::BF16,
            Dtype::Float32 => DType::F32,
            Dtype::Float64 => DType::F64,
            other => panic!("dash-mag: an array of {other:?}, which nothing here makes"),
        }
    }

    fn to_bytes(a: &Array) -> Result<Vec<u8>> {
        if a.size() == 0 {
            return Ok(Vec::new());
        }
        let a = readable(a)?;
        Ok(match a.dtype() {
            Dtype::Float16 => a
                .try_as_slice::<f16>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            Dtype::Bfloat16 => a
                .try_as_slice::<bf16>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            Dtype::Float32 => a
                .try_as_slice::<f32>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            Dtype::Float64 => a
                .try_as_slice::<f64>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            other => return refuse(format!("to_bytes: {other:?} is not floating")),
        })
    }

    fn to_i32(a: &Array) -> Result<Vec<i32>> {
        if a.size() == 0 {
            return Ok(Vec::new());
        }
        let a = readable(&a.as_dtype(Dtype::Int32)?)?;
        Ok(a.try_as_slice::<i32>()?.to_vec())
    }

    fn to_bool(a: &Array) -> Result<Vec<bool>> {
        if a.size() == 0 {
            return Ok(Vec::new());
        }
        let a = readable(a)?;
        Ok(a.try_as_slice::<bool>()?.to_vec())
    }

    fn eval(arrays: &[&Array]) -> Result<()> {
        Ok(transforms::eval(arrays.iter().copied())?)
    }

    fn eval_async(arrays: &[&Array]) -> Result<()> {
        Ok(transforms::async_eval(arrays.iter().copied())?)
    }

    fn memory() -> Option<Memory> {
        Some(Memory {
            active: memory::active_memory().ok()?,
            peak: memory::peak_memory().ok()?,
            cache: memory::cache_memory().ok()?,
        })
    }

    fn reset_peak_memory() {
        let _ = memory::reset_peak_memory();
    }

    fn set_cache_limit(bytes: usize) {
        let _ = memory::set_cache_limit(bytes);
    }

    fn cast(a: &Array, dtype: DType) -> Result<Array> {
        let dtype = mlx_dtype(dtype);
        let cast = || a.as_dtype(dtype);
        Ok(if dtype == Dtype::Float64 {
            with_cpu(cast)?
        } else {
            on(&[a], cast)?
        })
    }

    fn stop_gradient(a: &Array) -> Result<Array> {
        Ok(on(&[a], || stop_gradient(a))?)
    }

    fn binary(op: Binary, a: &Array, b: &Array) -> Result<Array> {
        Ok(on(&[a, b], || match op {
            Binary::Add => a.add(b),
            Binary::Sub => a.subtract(b),
            Binary::Mul => a.multiply(b),
            Binary::Div => a.divide(b),
            Binary::Maximum => maximum(a, b),
            Binary::Minimum => minimum(a, b),
        })?)
    }

    fn compare(op: Compare, a: &Array, b: &Array) -> Result<Array> {
        Ok(on(&[a, b], || match op {
            Compare::Gt => a.gt(b),
            Compare::Ge => a.ge(b),
            Compare::Lt => a.lt(b),
            Compare::Le => a.le(b),
            Compare::Eq => a.eq(b),
        })?)
    }

    fn logical(op: Logical, a: &Array, b: &Array) -> Result<Array> {
        Ok(match op {
            Logical::And => a.logical_and(b),
            Logical::Or => a.logical_or(b),
        }?)
    }

    fn logical_not(a: &Array) -> Result<Array> {
        Ok(a.logical_not()?)
    }

    fn unary(op: Unary, a: &Array) -> Result<Array> {
        let wide = a.dtype() == Dtype::Float64;
        Ok(on(&[a], || match op {
            Unary::Neg => a.negative(),
            Unary::Abs => a.abs(),
            Unary::Exp if wide => exact_exp(a),
            Unary::Exp => a.exp(),
            Unary::Log => a.log(),
            Unary::Sqrt => a.sqrt(),
            Unary::Square => a.square(),
            Unary::Sigmoid if wide => exact_sigmoid(a),
            Unary::Sigmoid => sigmoid(a),
            Unary::Silu if wide => a.multiply(exact_sigmoid(a)?),
            Unary::Silu => nn::silu(a),
            Unary::Softplus if wide => exact_exp(a)?.add(Array::from_slice_f64(&[1.0], &[]))?.log(),
            Unary::Softplus => nn::softplus(a),
        })?)
    }

    fn matmul(a: &Array, b: &Array) -> Result<Array> {
        Ok(on(&[a, b], || a.matmul(b))?)
    }

    fn select(condition: &Array, a: &Array, b: &Array) -> Result<Array> {
        Ok(on(&[a, b], || select(condition, a, b))?)
    }

    fn reshape(a: &Array, shape: &[usize]) -> Result<Array> {
        Ok(on(&[a], || a.reshape(&dims(shape)))?)
    }

    fn permute(a: &Array, axes: &[usize]) -> Result<Array> {
        Ok(on(&[a], || a.transpose_axes(&dims(axes)))?)
    }

    fn broadcast_to(a: &Array, shape: &[usize]) -> Result<Array> {
        Ok(on(&[a], || broadcast_to(a, &dims(shape)))?)
    }

    fn concat(arrays: &[&Array], axis: usize) -> Result<Array> {
        Ok(on(arrays, || concatenate(arrays, axis as i32))?)
    }

    fn take(a: &Array, indices: &Array, axis: usize) -> Result<Array> {
        Ok(on(&[a], || a.take_axis(indices, axis as i32))?)
    }

    fn slice(a: &Array, axis: usize, start: usize, end: usize) -> Result<Array> {
        let mut starts = vec![0; a.ndim()];
        let mut stops = a.shape().to_vec();
        let strides = vec![1; a.ndim()];
        (starts[axis], stops[axis]) = (start as i32, end as i32);
        Ok(on(&[a], || slice(a, &starts, &stops, &strides))?)
    }

    fn put_along_axis(a: &Array, indices: &Array, values: &Array, axis: usize) -> Result<Array> {
        Ok(on(&[a, values], || {
            a.put_along_axis(indices, values, axis as i32)
        })?)
    }

    fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Option<Result<Array>> {
        if x.dtype() == Dtype::Float64 || weight.is_some_and(|w| w.dtype() == Dtype::Float64) {
            return None;
        }
        Some(fast::rms_norm(x, weight, eps).map_err(Error::from))
    }

    fn logsumexp(a: &Array, axis: usize) -> Option<Result<Array>> {
        Some(on(&[a], || a.logsumexp_axis(axis as i32, true)).map_err(Error::from))
    }

    fn rope(x: &Array, dim: usize, base: f32, at: RopeAt<&Array>) -> Result<Array> {
        let dim = dim as i32;
        Ok(match at {
            RopeAt::Start(start) => fast::rope(x, dim, false, base, 1.0, start as i32),
            RopeAt::Rows(positions) => fast::rope_dynamic(x, dim, false, base, 1.0, positions),
        }?)
    }

    fn reduce(op: Reduce, a: &Array, axis: Option<usize>) -> Result<Array> {
        Ok(on(&[a], || match (op, axis) {
            (Reduce::Sum, None) => a.sum(None),
            (Reduce::Mean, None) => a.mean(None),
            (Reduce::Max, None) => a.max(None),
            (Reduce::Min, None) => a.min(None),
            (Reduce::Sum, Some(axis)) => a.sum_axis(axis as i32, true),
            (Reduce::Mean, Some(axis)) => a.mean_axis(axis as i32, true),
            (Reduce::Max, Some(axis)) => a.max_axis(axis as i32, true),
            (Reduce::Min, Some(axis)) => a.min_axis(axis as i32, true),
        })?)
    }

    fn argmax(a: &Array, axis: usize) -> Result<Array> {
        Ok(on(&[a], || {
            argmax_axis(a, axis as i32, true)?.as_dtype(Dtype::Int32)
        })?)
    }

    fn sort(a: &Array, axis: usize) -> Result<Array> {
        Ok(on(&[a], || sort_axis(a, axis as i32))?)
    }

    fn vjp(
        f: VjpFn<'_, Array>,
        primals: &[Array],
        cotangents: &[Option<Array>],
    ) -> Result<VjpOut<Array>> {
        if primals.iter().any(|p| !p.dtype().is_float()) {
            return refuse("vjp: every primal is floating");
        }
        // MLX makes its tracer of a primal on the primal's stream, or the default device's if it has
        // none, as an array from the host: on the GPU, which f64 cannot be on.
        let primals = primals
            .iter()
            .map(|p| match p.dtype() {
                Dtype::Float64 => with_cpu(|| copy(p)),
                _ => Ok(p.clone()),
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let taken: Vec<Array> = cotangents.iter().flatten().cloned().collect();
        let mut outs = Vec::new();
        let (_, grads) = fallible_vjp(
            |p: &[Array]| {
                let out = f(p).map_err(|e| Exception::custom(e.0))?;
                if out.len() != cotangents.len() {
                    return Err(Exception::custom(format!(
                        "vjp: {} outputs and {} cotangents",
                        out.len(),
                        cotangents.len()
                    )));
                }
                let mut seeds = Vec::with_capacity(taken.len());
                for (o, cotangent) in out.iter().zip(cotangents) {
                    let Some(cotangent) = cotangent else { continue };
                    if o.shape() != cotangent.shape() {
                        return Err(Exception::custom(format!(
                            "vjp: an output {:?} and its cotangent {:?}",
                            o.shape(),
                            cotangent.shape()
                        )));
                    }
                    seeds.push(seeded(o, cotangent)?);
                }
                outs = out;
                Ok(seeds)
            },
            &primals,
            &taken,
        )?;
        let outs = outs
            .iter()
            .map(|o| on(&[o], || stop_gradient(o)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok((outs, grads))
    }

    fn attention(
        query: &Array,
        key: &Array,
        value: &Array,
        mask: Mask<&Array>,
        options: AttentionOptions,
    ) -> Result<Array> {
        no_f64("attention", [query, key, value])?;
        let mask = match mask {
            Mask::None => dash_mlx::attention::AttentionMask::None,
            Mask::KeyPadding(m) => dash_mlx::attention::AttentionMask::KeyPadding(m.clone()),
            Mask::Packed {
                cu_seqlens_q,
                cu_seqlens_k,
            } => dash_mlx::attention::AttentionMask::Packed {
                cu_seqlens_q: cu_seqlens_q.clone(),
                cu_seqlens_k: cu_seqlens_k.clone(),
            },
        };
        Ok(dash_mlx::attention::attention(
            query, key, value, &mask, options,
        )?)
    }

    fn attention_decode(
        query: &Array,
        key: &Array,
        value: &Array,
        cache_lengths: Option<&Array>,
        options: AttentionOptions,
    ) -> Result<Array> {
        no_f64("attention_decode", [query, key, value])?;
        Ok(dash_mlx::attention::attention_decode(
            query,
            key,
            value,
            cache_lengths,
            options,
        )?)
    }

    fn causal_conv1d(
        x: &Array,
        weight: &Array,
        bias: Option<&Array>,
        initial_state: Option<&Array>,
        cu_seqlens: Option<&Array>,
        options: CausalConv1dOptions,
    ) -> Result<(Array, Option<Array>)> {
        no_f64(
            "causal_conv1d",
            [x, weight].into_iter().chain(bias).chain(initial_state),
        )?;
        Ok(dash_mlx::conv1d::causal_conv1d(
            x,
            weight,
            bias,
            initial_state,
            cu_seqlens,
            options,
        )?)
    }

    fn causal_conv1d_update(
        x: &Array,
        weight: &Array,
        bias: Option<&Array>,
        state: &Array,
        options: CausalConv1dStepOptions,
    ) -> Result<(Array, Array)> {
        no_f64(
            "causal_conv1d_update",
            [x, weight, state].into_iter().chain(bias),
        )?;
        Ok(dash_mlx::conv1d::causal_conv1d_update(
            x, weight, bias, state, options,
        )?)
    }

    fn chunk_gated_delta_rule(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        initial_state: Option<&Array>,
        cu_seqlens: Option<&Array>,
        options: ChunkGatedDeltaOptions,
    ) -> Result<(Array, Option<Array>)> {
        no_f64(
            "chunk_gated_delta_rule",
            [q, k, v, g, beta].into_iter().chain(initial_state),
        )?;
        Ok(dash_mlx::gated_delta::chunk::chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            options,
        )?)
    }

    fn gated_delta_step(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state: &Array,
        cu_seqlens: Option<&Array>,
        options: GatedDeltaStepOptions,
    ) -> Result<(Array, Array)> {
        no_f64("gated_delta_step", [q, k, v, g, beta, state])?;
        Ok(dash_mlx::gated_delta::recurrent::gated_delta_step(
            q, k, v, g, beta, state, cu_seqlens, options,
        )?)
    }

    fn chosen_logprob(
        hidden: &Array,
        weight: &Array,
        targets: &Array,
        masked: &[u32],
        options: LogprobOptions,
    ) -> Result<(Array, Array)> {
        no_f64("chosen_logprob", [hidden, weight])?;
        let out = dash_mlx::logprob::chosen_logprob(hidden, weight, targets, masked, options)?;
        Ok((out.logprob, out.entropy))
    }

    fn sample(logits: &Array, options: &SampleOptions) -> Result<Drawn<Array>> {
        no_f64("sample", [logits])?;
        let out = dash_mlx::sample::sample(logits, options)?;
        Ok(Drawn {
            tokens: out.tokens,
            logprob: out.logprob,
            tempered_logprob: out.tempered_logprob,
            sampling_logprob: out.sampling_logprob,
        })
    }

    fn low_rank(
        x: &Array,
        u: &Array,
        v: &Array,
        base: Option<&Array>,
        options: LowRankOptions,
    ) -> Result<Array> {
        no_f64("low_rank", [x, u, v].into_iter().chain(base))?;
        Ok(dash_mlx::lowrank::low_rank(x, u, v, base, options)?)
    }
}
