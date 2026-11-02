//! The side on Burn: dash-ops' operations and Burn's tensors, on CUDA in a build with `cuda` and
//! on Flex otherwise.
//!
//! An array holds Burn's dispatch tensor, which has no rank in its type, and takes up the rank it
//! has for the length of one call: each operation is a function generic over the rank, and
//! [`ranked!`] picks the instance. Rank 0 has no Burn tensor of its own: it is held as `[1]`.

use std::marker::PhantomData;
use std::sync::OnceLock;

use burn::backend::Dispatch;
use burn::backend::tensor::FloatTensor;
use burn::tensor::activation::{sigmoid, silu, softplus};
use burn::tensor::{
    Bool, DType as BurnDType, Device, IndexingUpdateOp, Int, IntDType, Tensor, TensorData,
};
use dash_ops::attention::{AttentionMask, attention, attention_decode};
use dash_ops::conv1d::{causal_conv1d, causal_conv1d_update};
use dash_ops::gated_delta::chunk::chunk_gated_delta_rule;
use dash_ops::gated_delta::recurrent::gated_delta_step;
use dash_ops::logprob::chosen_logprob;
use dash_ops::lowrank::low_rank;
use dash_ops::sample::sample;

use super::{Arm, Binary, Compare, Drawn, Logical, Mask, Reduce, Unary, VjpFn, VjpOut};
use crate::ops::{
    AttentionOptions, CausalConv1dOptions, CausalConv1dStepOptions, ChunkGatedDeltaOptions,
    GatedDeltaStepOptions, LogprobOptions, LowRankOptions, SampleOptions,
};
use crate::{DType, Result, refuse};

/// Burn's tensor with no rank or kind in its type.
type Raw = FloatTensor<Dispatch>;

/// What a tensor's kind has to be for the functions below, whatever it is.
trait Kindly: burn::tensor::kind::Basic + burn::tensor::kind::Autodiff {}
impl<K: burn::tensor::kind::Basic + burn::tensor::kind::Autodiff> Kindly for K {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Float,
    Int,
    Bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Array {
    raw: Raw,
    kind: Kind,
    /// Rank 0, held as `[1]`.
    scalar: bool,
    /// Not `Send`, as MLX's arrays are not: code that moves a tensor between threads would compile
    /// here and not on macOS.
    one_thread: PhantomData<*const ()>,
}

/// `$f::<R>($args)` with `R` this array's Burn rank.
macro_rules! ranked {
    ($rank:expr, $f:ident $(, $g:ty)? ; $($arg:expr),* $(,)?) => {
        match $rank {
            1 => $f::<1 $(, $g)?>($($arg),*),
            2 => $f::<2 $(, $g)?>($($arg),*),
            3 => $f::<3 $(, $g)?>($($arg),*),
            4 => $f::<4 $(, $g)?>($($arg),*),
            5 => $f::<5 $(, $g)?>($($arg),*),
            6 => $f::<6 $(, $g)?>($($arg),*),
            rank => panic!("dash-mag: rank {rank}; ranks go to 6"),
        }
    };
}

/// [`ranked!`] with the tensor kind `Float`, `Int` or `Bool` this array has too.
macro_rules! kinded {
    ($kind:expr, $rank:expr, $f:ident ; $($arg:expr),* $(,)?) => {
        match $kind {
            Kind::Float => ranked!($rank, $f, burn::tensor::Float; $($arg),*),
            Kind::Int => ranked!($rank, $f, Int; $($arg),*),
            Kind::Bool => ranked!($rank, $f, Bool; $($arg),*),
        }
    };
}

/// Refuses every tensor on a device that would run f32 matmuls in TF32, far past the bounds this
/// crate's results are held to, unless `CUBECL_CUDA_TF32=1` says that is wanted. All tensors start
/// at the four constructors that ask this. An application's build has the published cubecl-cuda
/// unless it repeats the workspace's patch, and nothing else would say so.
#[cfg(feature = "cuda")]
fn check_device() -> Result<()> {
    static REFUSED: OnceLock<bool> = OnceLock::new();
    let refused = *REFUSED.get_or_init(|| {
        std::env::var("CUBECL_CUDA_TF32").as_deref() != Ok("1")
            && dash_ops::f32_products_in_tf32(device())
    });
    if refused {
        return refuse(
            "this device would run f32 matmuls in TF32: set CUBECL_CUDA_TF32=0 with the patch of \
             cubecl-cuda in your manifest (dash-mag's README, \"On CUDA\"), or CUBECL_CUDA_TF32=1 \
             to accept TF32",
        );
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn check_device() -> Result<()> {
    Ok(())
}

fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(|| {
        #[cfg(feature = "cuda")]
        {
            Device::cuda(0)
        }
        #[cfg(all(not(feature = "cuda"), feature = "flex"))]
        {
            Device::flex()
        }
        #[cfg(not(any(feature = "cuda", feature = "flex")))]
        {
            Device::default()
        }
    })
}

fn burn_dtype(dtype: DType) -> Result<BurnDType> {
    match dtype {
        DType::F16 => Ok(BurnDType::F16),
        DType::BF16 => Ok(BurnDType::BF16),
        DType::F32 => Ok(BurnDType::F32),
        DType::F64 => Ok(BurnDType::F64),
        other => refuse(format!("{other:?} is not a floating dtype")),
    }
}

fn dims<const D: usize>(shape: &[usize]) -> [usize; D] {
    shape.try_into().expect("the rank this was picked for")
}

fn rank_of(shape: &[usize]) -> usize {
    shape.len().max(1)
}

/// The shape Burn holds: rank 0 as `[1]`.
fn held(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1]
    } else {
        shape.to_vec()
    }
}

impl Array {
    fn new(raw: Raw, kind: Kind, scalar: bool) -> Self {
        Self {
            raw,
            kind,
            scalar,
            one_thread: PhantomData,
        }
    }

    fn float(raw: Raw) -> Self {
        Self::new(raw, Kind::Float, false)
    }

    fn rank(&self) -> usize {
        burn::backend::TensorMetadata::shape(&self.raw).len()
    }

    fn t<const D: usize>(&self) -> Tensor<D> {
        Tensor::from_dispatch(self.raw.clone())
    }

    fn i<const D: usize>(&self) -> Tensor<D, Int> {
        Tensor::from_dispatch(self.raw.clone())
    }

    fn b<const D: usize>(&self) -> Tensor<D, Bool> {
        Tensor::from_dispatch(self.raw.clone())
    }

    fn is_autodiff(&self) -> bool {
        match self.kind {
            Kind::Float => self.t::<1>().is_autodiff(),
            Kind::Int => self.i::<1>().is_autodiff(),
            Kind::Bool => self.b::<1>().is_autodiff(),
        }
    }
}

fn wrap<const D: usize>(t: Tensor<D>) -> Array {
    Array::float(t.into_dispatch())
}

/// `t` in autodiff when the call's other inputs are: dash-ops' operations take all of a call's
/// inputs on one side of it, and a constant then takes no gradient.
fn ad<const D: usize, K: Kindly>(t: Tensor<D, K>, on: bool) -> Tensor<D, K> {
    if on { t.autodiff() } else { t }
}

fn from_data_float<const D: usize>(data: TensorData, dtype: BurnDType) -> Raw {
    Tensor::<D>::from_data(data, (device(), dtype)).into_dispatch()
}

fn from_data_int<const D: usize>(data: TensorData) -> Raw {
    Tensor::<D, Int>::from_data(data, device())
        .cast(IntDType::I32)
        .into_dispatch()
}

fn from_data_bool<const D: usize>(data: TensorData) -> Raw {
    Tensor::<D, Bool>::from_data(data, device()).into_dispatch()
}

fn full_float<const D: usize>(shape: &[usize], value: f64, dtype: BurnDType) -> Raw {
    Tensor::<D>::full(dims::<D>(shape), value, (device(), dtype)).into_dispatch()
}

fn read<const D: usize, K: Kindly>(raw: Raw) -> TensorData {
    Tensor::<D, K>::from_dispatch(raw).inner().into_data()
}

fn read_bool<const D: usize>(a: &Array) -> TensorData {
    a.b::<D>().inner().int().into_data()
}

fn cast_float<const D: usize>(a: &Array, dtype: BurnDType) -> Raw {
    a.t::<D>().cast(dtype).into_dispatch()
}

fn int_to_float<const D: usize>(a: &Array, dtype: BurnDType) -> Raw {
    a.i::<D>().float().cast(dtype).into_dispatch()
}

fn bool_to_float<const D: usize>(a: &Array, dtype: BurnDType) -> Raw {
    a.b::<D>().float().cast(dtype).into_dispatch()
}

fn int_dtype(dtype: DType) -> IntDType {
    if dtype == DType::I64 {
        IntDType::I64
    } else {
        IntDType::I32
    }
}

fn int_to_int<const D: usize>(a: &Array, dtype: DType) -> Raw {
    a.i::<D>().cast(int_dtype(dtype)).into_dispatch()
}

fn float_to_int<const D: usize>(a: &Array, dtype: DType) -> Raw {
    a.t::<D>().int().cast(int_dtype(dtype)).into_dispatch()
}

fn bool_to_int<const D: usize>(a: &Array, dtype: DType) -> Raw {
    a.b::<D>().int().cast(int_dtype(dtype)).into_dispatch()
}

fn int_to_bool<const D: usize>(a: &Array) -> Raw {
    a.i::<D>().bool().into_dispatch()
}

fn float_to_bool<const D: usize>(a: &Array) -> Raw {
    a.t::<D>().bool().into_dispatch()
}

fn detach<const D: usize>(a: &Array) -> Raw {
    a.t::<D>().detach().into_dispatch()
}

fn binary<const D: usize>(op: Binary, a: &Array, b: &Array) -> Raw {
    let (x, y) = (a.t::<D>(), b.t::<D>());
    match op {
        Binary::Add => x + y,
        Binary::Sub => x - y,
        Binary::Mul => x * y,
        Binary::Div => x / y,
        Binary::Maximum => x.max_pair(y),
        Binary::Minimum => x.min_pair(y),
    }
    .into_dispatch()
}

fn compare<const D: usize, K: Kindly + burn::tensor::kind::Ordered>(
    op: Compare,
    a: &Array,
    b: &Array,
) -> Raw {
    let x = Tensor::<D, K>::from_dispatch(a.raw.clone());
    let y = Tensor::<D, K>::from_dispatch(b.raw.clone());
    match op {
        Compare::Gt => x.greater(y),
        Compare::Ge => x.greater_equal(y),
        Compare::Lt => x.lower(y),
        Compare::Le => x.lower_equal(y),
        Compare::Eq => x.equal(y),
    }
    .into_dispatch()
}

fn logical<const D: usize>(op: Logical, a: &Array, b: &Array) -> Raw {
    let (x, y) = (a.b::<D>(), b.b::<D>());
    match op {
        Logical::And => x.bool_and(y),
        Logical::Or => x.bool_or(y),
    }
    .into_dispatch()
}

fn logical_not<const D: usize>(a: &Array) -> Raw {
    a.b::<D>().bool_not().into_dispatch()
}

/// Burn's sigmoid runs through f32 on CubeCL (f64 comes back with f32's error), so f64 takes the
/// definition.
fn exact_sigmoid<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    x.neg().exp().add_scalar(1.0).recip()
}

fn unary<const D: usize>(op: Unary, a: &Array) -> Raw {
    let x = a.t::<D>();
    match op {
        Unary::Neg => x.neg(),
        Unary::Abs => x.abs(),
        Unary::Exp => x.exp(),
        Unary::Log => x.log(),
        Unary::Sqrt => x.sqrt(),
        Unary::Square => x.square(),
        Unary::Sigmoid if x.dtype() == BurnDType::F64 => exact_sigmoid(x),
        Unary::Silu if x.dtype() == BurnDType::F64 => x.clone() * exact_sigmoid(x),
        Unary::Sigmoid => sigmoid(x),
        Unary::Silu => silu(x),
        Unary::Softplus => softplus(x, 1.0),
    }
    .into_dispatch()
}

fn matmul<const D: usize>(a: &Array, b: &Array) -> Raw {
    a.t::<D>().matmul(b.t::<D>()).into_dispatch()
}

fn select<const D: usize, K: Kindly>(condition: &Array, a: &Array, b: &Array) -> Raw {
    Tensor::<D, K>::from_dispatch(b.raw.clone())
        .mask_where(
            condition.b::<D>(),
            Tensor::<D, K>::from_dispatch(a.raw.clone()),
        )
        .into_dispatch()
}

fn reshape_to<const D: usize, K: Kindly>(raw: Raw, shape: &[usize]) -> Raw {
    let t = Tensor::<D, K>::from_dispatch(raw);
    match shape.len() {
        1 => t.reshape(dims::<1>(shape)).into_dispatch(),
        2 => t.reshape(dims::<2>(shape)).into_dispatch(),
        3 => t.reshape(dims::<3>(shape)).into_dispatch(),
        4 => t.reshape(dims::<4>(shape)).into_dispatch(),
        5 => t.reshape(dims::<5>(shape)).into_dispatch(),
        6 => t.reshape(dims::<6>(shape)).into_dispatch(),
        rank => panic!("dash-mag: rank {rank}; ranks go to 6"),
    }
}

fn permute_to<const D: usize, K: Kindly>(raw: Raw, axes: &[usize]) -> Raw {
    Tensor::<D, K>::from_dispatch(raw)
        .permute(dims::<D>(axes))
        .into_dispatch()
}

fn expand_to<const D: usize, K: Kindly>(raw: Raw, shape: &[usize]) -> Raw {
    Tensor::<D, K>::from_dispatch(raw)
        .expand(dims::<D>(shape))
        .into_dispatch()
}

fn cat<const D: usize, K: Kindly>(arrays: &[&Array], axis: usize) -> Raw {
    let tensors = arrays
        .iter()
        .map(|a| Tensor::<D, K>::from_dispatch(a.raw.clone()))
        .collect();
    Tensor::cat(tensors, axis).into_dispatch()
}

fn take_axis<const D: usize, K: Kindly>(raw: Raw, indices: &Array, axis: usize) -> Raw {
    Tensor::<D, K>::from_dispatch(raw)
        .select(axis, indices.i::<1>())
        .into_dispatch()
}

fn put_axis<const D: usize>(a: &Array, indices: &Array, values: &Array, axis: usize) -> Raw {
    put_axis_owned::<D>(a.clone(), indices, values, axis)
}

/// Burn's scatter writes in place when nothing else holds the buffer, and copies it otherwise.
fn put_axis_owned<const D: usize>(a: Array, indices: &Array, values: &Array, axis: usize) -> Raw {
    let on = a.is_autodiff() || values.is_autodiff();
    ad(Tensor::<D>::from_dispatch(a.raw), on)
        .scatter(
            axis,
            ad(indices.i::<D>(), on),
            ad(values.t::<D>(), on),
            IndexingUpdateOp::Assign,
        )
        .into_dispatch()
}

fn narrow<const D: usize, K: Kindly>(raw: Raw, axis: usize, start: usize, end: usize) -> Raw {
    let t = Tensor::<D, K>::from_dispatch(raw);
    if start == end {
        // Burn's narrow takes no empty range.
        let mut shape = t.dims();
        shape[axis] = 0;
        let dtype = t.dtype();
        return Tensor::<D, K>::empty(shape, (device(), dtype)).into_dispatch();
    }
    t.narrow(axis, start, end - start).into_dispatch()
}

fn reduce_all<const D: usize>(op: Reduce, a: &Array) -> Raw {
    let x = a.t::<D>();
    match op {
        Reduce::Sum => x.sum(),
        Reduce::Mean => x.mean(),
        Reduce::Max => x.max(),
        Reduce::Min => x.min(),
    }
    .into_dispatch()
}

fn reduce_axis<const D: usize>(op: Reduce, a: &Array, axis: usize) -> Raw {
    let x = a.t::<D>();
    match op {
        Reduce::Sum => x.sum_dim(axis),
        Reduce::Mean => x.mean_dim(axis),
        Reduce::Max => x.max_dim(axis),
        Reduce::Min => x.min_dim(axis),
    }
    .into_dispatch()
}

fn argmax_axis<const D: usize>(a: &Array, axis: usize) -> Raw {
    a.t::<D>().argmax(axis).cast(IntDType::I32).into_dispatch()
}

fn sort_axis<const D: usize>(a: &Array, axis: usize) -> Raw {
    a.t::<D>().sort(axis).into_dispatch()
}

/// A primal as a leaf of the graph `vjp` records: whatever graph it came from is left behind.
fn leaf<const D: usize>(a: &Array) -> Raw {
    a.t::<D>().inner().autodiff().require_grad().into_dispatch()
}

/// `sum(out * cotangent)`, taken in the output's dtype with the cotangent rounded to it, and cast
/// to f32 so that outputs of different dtypes add up.
fn inner_product<const D: usize>(out: &Array, cotangent: &Array) -> Tensor<1> {
    let o = out.t::<D>();
    let c = cotangent.t::<D>().inner().cast(o.dtype());
    (o * c).sum().cast(BurnDType::F32)
}

fn is_tracked<const D: usize>(a: &Array) -> bool {
    a.t::<D>().is_tracked()
}

fn grad_of<const D: usize>(leaf: &Array, grads: Option<&burn::tensor::Gradients>) -> Raw {
    let t = leaf.t::<D>();
    match grads.and_then(|grads| t.grad(grads)) {
        Some(grad) => grad.into_dispatch(),
        None => t.inner().zeros_like().into_dispatch(),
    }
}

fn without_autodiff<const D: usize>(a: &Array) -> Raw {
    a.t::<D>().inner().into_dispatch()
}

fn low_rank_ranked<const D: usize>(
    x: &Array,
    u: &Array,
    v: &Array,
    base: Option<&Array>,
    options: LowRankOptions,
    on: bool,
) -> Raw {
    low_rank::<D>(
        ad(x.t::<D>(), on),
        ad(u.t::<2>(), on),
        ad(v.t::<2>(), on),
        base.map(|b| ad(b.t::<D>(), on)),
        options,
    )
    .into_dispatch()
}

pub(crate) struct Burn;

impl Arm for Burn {
    type Array = Array;

    fn describe() -> String {
        format!("Burn, {:?}", device())
    }

    fn clear_cache() {
        device().memory_cleanup();
    }

    fn from_bytes(bytes: &[u8], shape: &[usize], dtype: DType) -> Result<Array> {
        check_device()?;
        let dtype = burn_dtype(dtype)?;
        let data = TensorData::from_bytes_vec(bytes.to_vec(), held(shape), dtype);
        let raw = ranked!(rank_of(shape), from_data_float; data, dtype);
        Ok(Array::new(raw, Kind::Float, shape.is_empty()))
    }

    fn from_i32(values: &[i32], shape: &[usize]) -> Result<Array> {
        check_device()?;
        let data = TensorData::new(values.to_vec(), held(shape));
        let raw = ranked!(rank_of(shape), from_data_int; data);
        Ok(Array::new(raw, Kind::Int, shape.is_empty()))
    }

    fn from_bool(values: &[bool], shape: &[usize]) -> Result<Array> {
        check_device()?;
        let data = TensorData::new(values.to_vec(), held(shape));
        let raw = ranked!(rank_of(shape), from_data_bool; data);
        Ok(Array::new(raw, Kind::Bool, shape.is_empty()))
    }

    fn full(shape: &[usize], value: f64, dtype: DType) -> Result<Array> {
        check_device()?;
        let dtype = burn_dtype(dtype)?;
        let raw = ranked!(rank_of(shape), full_float; &held(shape), value, dtype);
        Ok(Array::new(raw, Kind::Float, shape.is_empty()))
    }

    fn shape(a: &Array) -> Vec<usize> {
        if a.scalar {
            Vec::new()
        } else {
            burn::backend::TensorMetadata::shape(&a.raw).to_vec()
        }
    }

    fn dtype(a: &Array) -> DType {
        match a.kind {
            Kind::Bool => DType::Bool,
            Kind::Int => match burn::backend::TensorMetadata::dtype(&a.raw) {
                BurnDType::I32 => DType::I32,
                _ => DType::I64,
            },
            Kind::Float => match burn::backend::TensorMetadata::dtype(&a.raw) {
                BurnDType::F16 => DType::F16,
                BurnDType::BF16 => DType::BF16,
                BurnDType::F64 => DType::F64,
                _ => DType::F32,
            },
        }
    }

    fn to_bytes(a: &Array) -> Result<Vec<u8>> {
        let data = ranked!(a.rank(), read, burn::tensor::Float; a.raw.clone());
        Ok(data.as_bytes().to_vec())
    }

    fn to_i32(a: &Array) -> Result<Vec<i32>> {
        let data = ranked!(a.rank(), read, Int; a.raw.clone());
        data.convert::<i32>()
            .try_to_vec::<i32>()
            .map_err(|e| crate::Error(format!("{e:?}")))
    }

    fn to_bool(a: &Array) -> Result<Vec<bool>> {
        // Read as integers: how a device stores booleans is its own.
        let data = ranked!(a.rank(), read_bool; a);
        let values = data
            .convert::<i32>()
            .try_to_vec::<i32>()
            .map_err(|e| crate::Error(format!("{e:?}")))?;
        Ok(values.into_iter().map(|v| v != 0).collect())
    }

    /// Burn queues work and runs it when it must; nothing here holds a graph of its own to bound.
    fn eval(_: &[&Array]) -> Result<()> {
        Ok(())
    }

    fn cast(a: &Array, dtype: DType) -> Result<Array> {
        let rank = a.rank();
        let (raw, kind) = match (a.kind, dtype) {
            (Kind::Float, DType::Bool) => (ranked!(rank, float_to_bool; a), Kind::Bool),
            (Kind::Int, DType::Bool) => (ranked!(rank, int_to_bool; a), Kind::Bool),
            (Kind::Bool, DType::Bool) => (a.raw.clone(), Kind::Bool),
            (Kind::Float, d @ (DType::I32 | DType::I64)) => {
                (ranked!(rank, float_to_int; a, d), Kind::Int)
            }
            (Kind::Bool, d @ (DType::I32 | DType::I64)) => {
                (ranked!(rank, bool_to_int; a, d), Kind::Int)
            }
            (Kind::Int, d @ (DType::I32 | DType::I64)) => {
                (ranked!(rank, int_to_int; a, d), Kind::Int)
            }
            (Kind::Float, d) => (ranked!(rank, cast_float; a, burn_dtype(d)?), Kind::Float),
            (Kind::Int, d) => (ranked!(rank, int_to_float; a, burn_dtype(d)?), Kind::Float),
            (Kind::Bool, d) => (ranked!(rank, bool_to_float; a, burn_dtype(d)?), Kind::Float),
        };
        Ok(Array::new(raw, kind, a.scalar))
    }

    fn stop_gradient(a: &Array) -> Result<Array> {
        Ok(match a.kind {
            Kind::Float => Array::new(ranked!(a.rank(), detach; a), Kind::Float, a.scalar),
            _ => a.clone(),
        })
    }

    fn binary(op: Binary, a: &Array, b: &Array) -> Result<Array> {
        let raw = ranked!(a.rank(), binary; op, a, b);
        Ok(Array::new(raw, Kind::Float, a.scalar))
    }

    fn compare(op: Compare, a: &Array, b: &Array) -> Result<Array> {
        let raw = match a.kind {
            Kind::Float => ranked!(a.rank(), compare, burn::tensor::Float; op, a, b),
            Kind::Int => ranked!(a.rank(), compare, Int; op, a, b),
            Kind::Bool => return refuse("compare: booleans reach here as integers"),
        };
        Ok(Array::new(raw, Kind::Bool, a.scalar))
    }

    fn logical(op: Logical, a: &Array, b: &Array) -> Result<Array> {
        let raw = ranked!(a.rank(), logical; op, a, b);
        Ok(Array::new(raw, Kind::Bool, a.scalar))
    }

    fn logical_not(a: &Array) -> Result<Array> {
        let raw = ranked!(a.rank(), logical_not; a);
        Ok(Array::new(raw, Kind::Bool, a.scalar))
    }

    fn unary(op: Unary, a: &Array) -> Result<Array> {
        let raw = ranked!(a.rank(), unary; op, a);
        Ok(Array::new(raw, Kind::Float, a.scalar))
    }

    fn matmul(a: &Array, b: &Array) -> Result<Array> {
        Ok(Array::float(ranked!(a.rank(), matmul; a, b)))
    }

    fn select(condition: &Array, a: &Array, b: &Array) -> Result<Array> {
        let raw = kinded!(a.kind, a.rank(), select; condition, a, b);
        Ok(Array::new(raw, a.kind, a.scalar))
    }

    fn reshape(a: &Array, shape: &[usize]) -> Result<Array> {
        let raw = kinded!(a.kind, a.rank(), reshape_to; a.raw.clone(), &held(shape));
        Ok(Array::new(raw, a.kind, shape.is_empty()))
    }

    fn permute(a: &Array, axes: &[usize]) -> Result<Array> {
        if a.scalar {
            return Ok(a.clone());
        }
        let raw = kinded!(a.kind, a.rank(), permute_to; a.raw.clone(), axes);
        Ok(Array::new(raw, a.kind, false))
    }

    fn broadcast_to(a: &Array, shape: &[usize]) -> Result<Array> {
        // The array's axes aligned at the end, then each of length 1 expanded.
        let from = Self::shape(a);
        let mut aligned = vec![1; shape.len() - from.len()];
        aligned.extend_from_slice(&from);
        let raw = if aligned == held(&from) {
            a.raw.clone()
        } else {
            kinded!(a.kind, a.rank(), reshape_to; a.raw.clone(), &held(&aligned))
        };
        let raw = kinded!(a.kind, rank_of(shape), expand_to; raw, &held(shape));
        Ok(Array::new(raw, a.kind, shape.is_empty()))
    }

    fn concat(arrays: &[&Array], axis: usize) -> Result<Array> {
        let first = arrays[0];
        let raw = kinded!(first.kind, first.rank(), cat; arrays, axis);
        Ok(Array::new(raw, first.kind, false))
    }

    fn take(a: &Array, indices: &Array, axis: usize) -> Result<Array> {
        let raw = kinded!(a.kind, a.rank(), take_axis; a.raw.clone(), indices, axis);
        Ok(Array::new(raw, a.kind, false))
    }

    fn slice(a: &Array, axis: usize, start: usize, end: usize) -> Result<Array> {
        let raw = kinded!(a.kind, a.rank(), narrow; a.raw.clone(), axis, start, end);
        Ok(Array::new(raw, a.kind, false))
    }

    fn put_along_axis(a: &Array, indices: &Array, values: &Array, axis: usize) -> Result<Array> {
        Ok(Array::float(
            ranked!(a.rank(), put_axis; a, indices, values, axis),
        ))
    }

    fn put_along_axis_owned(
        a: Array,
        indices: &Array,
        values: &Array,
        axis: usize,
    ) -> Result<Array> {
        Ok(Array::float(
            ranked!(a.rank(), put_axis_owned; a, indices, values, axis),
        ))
    }

    fn reduce(op: Reduce, a: &Array, axis: Option<usize>) -> Result<Array> {
        Ok(match axis {
            None => Array::new(ranked!(a.rank(), reduce_all; op, a), Kind::Float, true),
            Some(axis) => Array::float(ranked!(a.rank(), reduce_axis; op, a, axis)),
        })
    }

    fn argmax(a: &Array, axis: usize) -> Result<Array> {
        Ok(Array::new(
            ranked!(a.rank(), argmax_axis; a, axis),
            Kind::Int,
            false,
        ))
    }

    fn sort(a: &Array, axis: usize) -> Result<Array> {
        Ok(Array::float(ranked!(a.rank(), sort_axis; a, axis)))
    }

    fn vjp(
        f: VjpFn<'_, Array>,
        primals: &[Array],
        cotangents: &[Option<Array>],
    ) -> Result<VjpOut<Array>> {
        let mut leaves = Vec::with_capacity(primals.len());
        for p in primals {
            if p.kind != Kind::Float {
                return refuse("vjp: every primal is floating");
            }
            leaves.push(Array::new(
                ranked!(p.rank(), leaf; p),
                Kind::Float,
                p.scalar,
            ));
        }
        let outs = f(&leaves)?;
        if outs.len() != cotangents.len() {
            return refuse(format!(
                "vjp: {} outputs and {} cotangents",
                outs.len(),
                cotangents.len()
            ));
        }
        let mut loss: Option<Tensor<1>> = None;
        for (out, cotangent) in outs.iter().zip(cotangents) {
            let Some(cotangent) = cotangent else { continue };
            if Self::shape(out) != Self::shape(cotangent) {
                return refuse(format!(
                    "vjp: an output {:?} and its cotangent {:?}",
                    Self::shape(out),
                    Self::shape(cotangent)
                ));
            }
            if out.kind != Kind::Float || !ranked!(out.rank(), is_tracked; out) {
                continue;
            }
            let term = ranked!(out.rank(), inner_product; out, cotangent);
            loss = Some(match loss {
                Some(loss) => loss + term,
                None => term,
            });
        }
        let grads = loss.map(|loss| loss.backward());
        let gradients = leaves
            .iter()
            .map(|l| {
                Array::new(
                    ranked!(l.rank(), grad_of; l, grads.as_ref()),
                    Kind::Float,
                    l.scalar,
                )
            })
            .collect();
        let outs = outs
            .iter()
            .map(|o| match o.kind {
                Kind::Float => Array::new(
                    ranked!(o.rank(), without_autodiff; o),
                    Kind::Float,
                    o.scalar,
                ),
                _ => o.clone(),
            })
            .collect();
        Ok((outs, gradients))
    }

    fn attention(
        query: &Array,
        key: &Array,
        value: &Array,
        mask: Mask<&Array>,
        options: AttentionOptions,
    ) -> Result<Array> {
        let on = query.is_autodiff() || key.is_autodiff() || value.is_autodiff();
        let mask = match mask {
            Mask::None => AttentionMask::None,
            Mask::KeyPadding(m) => AttentionMask::KeyPadding(ad(m.b::<2>(), on)),
            Mask::Packed {
                cu_seqlens_q,
                cu_seqlens_k,
            } => AttentionMask::Packed {
                cu_seqlens_q: ad(cu_seqlens_q.i::<1>(), on),
                cu_seqlens_k: ad(cu_seqlens_k.i::<1>(), on),
            },
        };
        Ok(wrap(attention(
            ad(query.t::<4>(), on),
            ad(key.t::<4>(), on),
            ad(value.t::<4>(), on),
            mask,
            options,
        )))
    }

    fn attention_decode(
        query: &Array,
        key: &Array,
        value: &Array,
        cache_lengths: Option<&Array>,
        options: AttentionOptions,
    ) -> Result<Array> {
        let on = query.is_autodiff() || key.is_autodiff() || value.is_autodiff();
        Ok(wrap(attention_decode(
            ad(query.t::<3>(), on),
            ad(key.t::<4>(), on),
            ad(value.t::<4>(), on),
            cache_lengths.map(|c| ad(c.i::<1>(), on)),
            options,
        )))
    }

    fn causal_conv1d(
        x: &Array,
        weight: &Array,
        bias: Option<&Array>,
        initial_state: Option<&Array>,
        cu_seqlens: Option<&Array>,
        options: CausalConv1dOptions,
    ) -> Result<(Array, Option<Array>)> {
        let on = x.is_autodiff()
            || weight.is_autodiff()
            || bias.is_some_and(Array::is_autodiff)
            || initial_state.is_some_and(Array::is_autodiff);
        let (out, state) = causal_conv1d(
            ad(x.t::<3>(), on),
            ad(weight.t::<2>(), on),
            bias.map(|b| ad(b.t::<1>(), on)),
            initial_state.map(|s| ad(s.t::<3>(), on)),
            cu_seqlens.map(|c| ad(c.i::<1>(), on)),
            options,
        );
        Ok((wrap(out), state.map(wrap)))
    }

    fn causal_conv1d_update(
        x: &Array,
        weight: &Array,
        bias: Option<&Array>,
        state: &Array,
        options: CausalConv1dStepOptions,
    ) -> Result<(Array, Array)> {
        let on = x.is_autodiff()
            || weight.is_autodiff()
            || bias.is_some_and(Array::is_autodiff)
            || state.is_autodiff();
        let (out, state) = causal_conv1d_update(
            ad(x.t::<2>(), on),
            ad(weight.t::<2>(), on),
            bias.map(|b| ad(b.t::<1>(), on)),
            ad(state.t::<3>(), on),
            options,
        );
        Ok((wrap(out), wrap(state)))
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
        let on = q.is_autodiff()
            || k.is_autodiff()
            || v.is_autodiff()
            || g.is_autodiff()
            || beta.is_autodiff()
            || initial_state.is_some_and(Array::is_autodiff);
        let (out, state) = chunk_gated_delta_rule(
            ad(q.t::<4>(), on),
            ad(k.t::<4>(), on),
            ad(v.t::<4>(), on),
            ad(g.t::<3>(), on),
            ad(beta.t::<3>(), on),
            initial_state.map(|s| ad(s.t::<4>(), on)),
            cu_seqlens.map(|c| ad(c.i::<1>(), on)),
            options,
        );
        Ok((wrap(out), state.map(wrap)))
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
        let on = q.is_autodiff()
            || k.is_autodiff()
            || v.is_autodiff()
            || g.is_autodiff()
            || beta.is_autodiff()
            || state.is_autodiff();
        let (out, state) = gated_delta_step(
            ad(q.t::<3>(), on),
            ad(k.t::<3>(), on),
            ad(v.t::<3>(), on),
            ad(g.t::<2>(), on),
            ad(beta.t::<2>(), on),
            ad(state.t::<4>(), on),
            cu_seqlens.map(|c| ad(c.i::<1>(), on)),
            options,
        );
        Ok((wrap(out), wrap(state)))
    }

    fn chosen_logprob(
        hidden: &Array,
        weight: &Array,
        targets: &Array,
        masked: &[u32],
        options: LogprobOptions,
    ) -> Result<(Array, Array)> {
        let on = hidden.is_autodiff();
        let out = chosen_logprob(
            ad(hidden.t::<2>(), on),
            ad(weight.t::<2>().inner(), on),
            ad(targets.i::<1>(), on),
            masked,
            options,
        );
        Ok((wrap(out.logprob), wrap(out.entropy)))
    }

    fn sample(logits: &Array, options: &SampleOptions) -> Result<Drawn<Array>> {
        let out = sample(logits.t::<2>(), options);
        Ok(Drawn {
            tokens: Array::new(
                out.tokens.cast(IntDType::I32).into_dispatch(),
                Kind::Int,
                false,
            ),
            logprob: wrap(out.logprob),
            tempered_logprob: wrap(out.tempered_logprob),
            sampling_logprob: wrap(out.sampling_logprob),
        })
    }

    fn low_rank(
        x: &Array,
        u: &Array,
        v: &Array,
        base: Option<&Array>,
        options: LowRankOptions,
    ) -> Result<Array> {
        let on = x.is_autodiff()
            || u.is_autodiff()
            || v.is_autodiff()
            || base.is_some_and(Array::is_autodiff);
        Ok(Array::float(
            ranked!(x.rank(), low_rank_ranked; x, u, v, base, options, on),
        ))
    }
}
