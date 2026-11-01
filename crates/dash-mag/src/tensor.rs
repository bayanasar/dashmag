//! The tensor an application holds, and its array operations: written once, over whichever side
//! the build has. Axes are normalised, shapes broadcast and dtypes promoted here, so that both
//! sides refuse and promote alike.

use std::ops::Range;

pub use crate::arm::Memory;
use crate::arm::{Arm, Binary, Compare, Logical, Reduce, Selected, Unary};
use crate::{Result, refuse};

pub(crate) type A = <Selected as Arm>::Array;

/// The element types a [`Tensor`] may have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    pub fn is_floating(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    /// Bytes per element of a floating dtype.
    fn size(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F64 | Self::I64 => 8,
            Self::Bool => 1,
            Self::F32 | Self::I32 => 4,
        }
    }

    /// The dtype two operands meet in: the wider float, f32 for bf16 beside f16; a float beside
    /// an integer or a boolean stays that float.
    fn promote(a: Self, b: Self) -> Self {
        use DType::*;
        match (a, b) {
            (a, b) if a == b => a,
            (F64, _) | (_, F64) => F64,
            (F32, _) | (_, F32) => F32,
            (F16, BF16) | (BF16, F16) => F32,
            (F16, _) | (_, F16) => F16,
            (BF16, _) | (_, BF16) => BF16,
            (I64, _) | (_, I64) => I64,
            _ => I32,
        }
    }
}

/// A tensor, of rank 0 to 6.
#[derive(Clone)]
pub struct Tensor(pub(crate) A);

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?}, {:?})", self.shape(), self.dtype())
    }
}

/// Computes these tensors now, where the side computes lazily: a training or generation loop
/// bounds what it holds with this, as it would on MLX.
pub fn eval(tensors: &[&Tensor]) -> Result<()> {
    let arrays: Vec<&A> = tensors.iter().map(|t| &t.0).collect();
    Selected::eval(&arrays)
}

/// Starts computing these tensors and returns at once, where the side can: reading one waits for it.
/// A decoding loop queues each step before it reads the last, and a side that queues anyway needs
/// nothing here.
pub fn eval_async(tensors: &[&Tensor]) -> Result<()> {
    let arrays: Vec<&A> = tensors.iter().map(|t| &t.0).collect();
    Selected::eval_async(&arrays)
}

/// What the device holds, where the side can say.
pub fn memory() -> Option<Memory> {
    Selected::memory()
}

/// Starts [`Memory::peak`] again from now, where the side keeps one.
pub fn reset_peak_memory() {
    Selected::reset_peak_memory();
}

/// How many bytes the side may keep cached for reuse, where it has such a limit.
pub fn set_cache_limit(bytes: usize) {
    Selected::set_cache_limit(bytes);
}

fn elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn check_rank(op: &str, rank: usize) -> Result<()> {
    if rank > 6 {
        return refuse(format!("{op}: rank {rank}; ranks go to 6"));
    }
    Ok(())
}

/// `axis` counted from the front, for a rank.
fn normal_axis(op: &str, axis: isize, rank: usize) -> Result<usize> {
    let r = rank as isize;
    if axis < -r || axis >= r {
        return refuse(format!("{op}: axis {axis} of a tensor of rank {rank}"));
    }
    Ok(if axis < 0 { axis + r } else { axis } as usize)
}

fn broadcast_shapes(op: &str, a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let at = |s: &[usize], i: usize| {
        let pad = rank - s.len();
        if i < pad { 1 } else { s[i - pad] }
    };
    (0..rank)
        .map(|i| match (at(a, i), at(b, i)) {
            (x, y) if x == y => Ok(x),
            (1, y) => Ok(y),
            (x, 1) => Ok(x),
            _ => refuse(format!("{op}: shapes {a:?} and {b:?} do not broadcast")),
        })
        .collect()
}

impl Tensor {
    pub(crate) fn wrap(a: Result<A>) -> Result<Self> {
        a.map(Self)
    }

    /// A tensor from little-endian bytes of a floating `dtype`, as a checkpoint holds them.
    pub fn from_bytes(bytes: &[u8], shape: &[usize], dtype: DType) -> Result<Self> {
        check_rank("from_bytes", shape.len())?;
        if !dtype.is_floating() {
            return refuse(format!("from_bytes: {dtype:?} is not floating"));
        }
        if bytes.len() != elements(shape) * dtype.size() {
            return refuse(format!(
                "from_bytes: {} bytes for {shape:?} of {dtype:?}",
                bytes.len()
            ));
        }
        Self::wrap(Selected::from_bytes(bytes, shape, dtype))
    }

    pub fn from_f32(values: &[f32], shape: &[usize]) -> Result<Self> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::from_bytes(&bytes, shape, DType::F32)
    }

    pub fn from_f64(values: &[f64], shape: &[usize]) -> Result<Self> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::from_bytes(&bytes, shape, DType::F64)
    }

    /// An `i32` tensor, on every side.
    pub fn from_i32(values: &[i32], shape: &[usize]) -> Result<Self> {
        check_rank("from_i32", shape.len())?;
        if values.len() != elements(shape) {
            return refuse(format!("from_i32: {} values for {shape:?}", values.len()));
        }
        Self::wrap(Selected::from_i32(values, shape))
    }

    pub fn from_bool(values: &[bool], shape: &[usize]) -> Result<Self> {
        check_rank("from_bool", shape.len())?;
        if values.len() != elements(shape) {
            return refuse(format!("from_bool: {} values for {shape:?}", values.len()));
        }
        Self::wrap(Selected::from_bool(values, shape))
    }

    pub fn full(shape: &[usize], value: f64, dtype: DType) -> Result<Self> {
        check_rank("full", shape.len())?;
        if !dtype.is_floating() {
            return refuse(format!("full: {dtype:?} is not floating"));
        }
        Self::wrap(Selected::full(shape, value, dtype))
    }

    pub fn zeros(shape: &[usize], dtype: DType) -> Result<Self> {
        Self::full(shape, 0.0, dtype)
    }

    pub fn ones(shape: &[usize], dtype: DType) -> Result<Self> {
        Self::full(shape, 1.0, dtype)
    }

    /// A value of rank 0, rounded to `dtype`.
    pub fn scalar(value: f64, dtype: DType) -> Result<Self> {
        Self::full(&[], value, dtype)
    }

    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(&self.shape(), self.dtype())
    }

    pub fn shape(&self) -> Vec<usize> {
        Selected::shape(&self.0)
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// The number of elements.
    pub fn size(&self) -> usize {
        elements(&self.shape())
    }

    pub fn dtype(&self) -> DType {
        Selected::dtype(&self.0)
    }

    /// The little-endian bytes of a floating tensor, in its dtype, as a checkpoint holds them.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        if !self.dtype().is_floating() {
            return refuse(format!("to_bytes: {:?} is not floating", self.dtype()));
        }
        Selected::to_bytes(&self.0)
    }

    /// The elements as f32: f16 and bf16 widened exactly, f64 rounded.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let bytes = self.to_bytes()?;
        Ok(match self.dtype() {
            DType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            DType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::bf16::from_le_bytes(*b).to_f32())
                .collect(),
            DType::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_le_bytes(*b).to_f32())
                .collect(),
            DType::F64 => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b) as f32)
                .collect(),
            _ => unreachable!("to_bytes refused it"),
        })
    }

    /// The elements as `i32`.
    pub fn to_i32(&self) -> Result<Vec<i32>> {
        if !matches!(self.dtype(), DType::I32 | DType::I64) {
            return refuse(format!("to_i32: {:?} is not an integer", self.dtype()));
        }
        Selected::to_i32(&self.0)
    }

    pub fn to_bool(&self) -> Result<Vec<bool>> {
        if self.dtype() != DType::Bool {
            return refuse(format!("to_bool: {:?} is not boolean", self.dtype()));
        }
        Selected::to_bool(&self.0)
    }

    /// The one element of a tensor that holds one, as f32.
    pub fn item(&self) -> Result<f32> {
        let values = self.to_f32()?;
        match values.as_slice() {
            [v] => Ok(*v),
            _ => refuse(format!("item: a tensor of {} elements", values.len())),
        }
    }

    pub fn as_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        Self::wrap(Selected::cast(&self.0, dtype))
    }

    /// The same values, a constant to gradients.
    pub fn stop_gradient(&self) -> Result<Self> {
        Self::wrap(Selected::stop_gradient(&self.0))
    }

    /// Both operands in the dtype they meet in, of one shape.
    fn aligned(&self, op: &str, other: &Tensor) -> Result<(A, A)> {
        let dtype = DType::promote(self.dtype(), other.dtype());
        let shape = broadcast_shapes(op, &self.shape(), &other.shape())?;
        let a = self.as_dtype(dtype)?.broadcast_to(&shape)?;
        let b = other.as_dtype(dtype)?.broadcast_to(&shape)?;
        Ok((a.0, b.0))
    }

    fn binary(&self, op: Binary, other: &Tensor) -> Result<Self> {
        let (a, b) = self.aligned(&format!("{op:?}"), other)?;
        if !Selected::dtype(&a).is_floating() {
            return refuse(format!("{op:?}: arithmetic here is on floating tensors"));
        }
        Self::wrap(Selected::binary(op, &a, &b))
    }

    fn compare(&self, op: Compare, other: &Tensor) -> Result<Self> {
        let (a, b) = self.aligned(&format!("{op:?}"), other)?;
        if Selected::dtype(&a) != DType::Bool {
            return Self::wrap(Selected::compare(op, &a, &b));
        }
        if op != Compare::Eq {
            return refuse(format!("{op:?}: booleans have no order"));
        }
        // Equality of booleans, as integers: each side stores them its own way.
        let a = Selected::cast(&a, DType::I32)?;
        let b = Selected::cast(&b, DType::I32)?;
        Self::wrap(Selected::compare(op, &a, &b))
    }

    fn logical(&self, op: Logical, other: &Tensor) -> Result<Self> {
        if self.dtype() != DType::Bool || other.dtype() != DType::Bool {
            return refuse(format!("{op:?}: both operands are boolean"));
        }
        let (a, b) = self.aligned(&format!("{op:?}"), other)?;
        Self::wrap(Selected::logical(op, &a, &b))
    }

    fn unary(&self, op: Unary) -> Result<Self> {
        if !self.dtype().is_floating() {
            return refuse(format!("{op:?}: {:?} is not floating", self.dtype()));
        }
        Self::wrap(Selected::unary(op, &self.0))
    }

    pub fn add(&self, other: &Tensor) -> Result<Self> {
        self.binary(Binary::Add, other)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Self> {
        self.binary(Binary::Sub, other)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Self> {
        self.binary(Binary::Mul, other)
    }

    pub fn div(&self, other: &Tensor) -> Result<Self> {
        self.binary(Binary::Div, other)
    }

    pub fn maximum(&self, other: &Tensor) -> Result<Self> {
        self.binary(Binary::Maximum, other)
    }

    pub fn minimum(&self, other: &Tensor) -> Result<Self> {
        self.binary(Binary::Minimum, other)
    }

    /// `self + value`, the value rounded to this tensor's dtype.
    pub fn add_scalar(&self, value: f64) -> Result<Self> {
        self.add(&Self::scalar(value, self.dtype())?)
    }

    /// `self * value`, the value rounded to this tensor's dtype.
    pub fn mul_scalar(&self, value: f64) -> Result<Self> {
        self.mul(&Self::scalar(value, self.dtype())?)
    }

    pub fn gt(&self, other: &Tensor) -> Result<Self> {
        self.compare(Compare::Gt, other)
    }

    pub fn ge(&self, other: &Tensor) -> Result<Self> {
        self.compare(Compare::Ge, other)
    }

    pub fn lt(&self, other: &Tensor) -> Result<Self> {
        self.compare(Compare::Lt, other)
    }

    pub fn le(&self, other: &Tensor) -> Result<Self> {
        self.compare(Compare::Le, other)
    }

    pub fn eq(&self, other: &Tensor) -> Result<Self> {
        self.compare(Compare::Eq, other)
    }

    pub fn logical_and(&self, other: &Tensor) -> Result<Self> {
        self.logical(Logical::And, other)
    }

    pub fn logical_or(&self, other: &Tensor) -> Result<Self> {
        self.logical(Logical::Or, other)
    }

    pub fn logical_not(&self) -> Result<Self> {
        if self.dtype() != DType::Bool {
            return refuse(format!("logical_not: {:?} is not boolean", self.dtype()));
        }
        Self::wrap(Selected::logical_not(&self.0))
    }

    pub fn neg(&self) -> Result<Self> {
        self.unary(Unary::Neg)
    }

    pub fn abs(&self) -> Result<Self> {
        self.unary(Unary::Abs)
    }

    pub fn exp(&self) -> Result<Self> {
        self.unary(Unary::Exp)
    }

    pub fn log(&self) -> Result<Self> {
        self.unary(Unary::Log)
    }

    pub fn sqrt(&self) -> Result<Self> {
        self.unary(Unary::Sqrt)
    }

    pub fn square(&self) -> Result<Self> {
        self.unary(Unary::Square)
    }

    pub fn sigmoid(&self) -> Result<Self> {
        self.unary(Unary::Sigmoid)
    }

    pub fn silu(&self) -> Result<Self> {
        self.unary(Unary::Silu)
    }

    pub fn softplus(&self) -> Result<Self> {
        self.unary(Unary::Softplus)
    }

    /// The matrix product over the last two axes, the axes before them broadcast. A left operand
    /// of rank 1 is a row and a right one a column, each dropped from the result.
    pub fn matmul(&self, other: &Tensor) -> Result<Self> {
        let (a, b) = (self.shape(), other.shape());
        if a.is_empty() || b.is_empty() {
            return refuse(format!("matmul: shapes {a:?} and {b:?}"));
        }
        let lhs = if a.len() == 1 {
            self.expand_dims(0)?
        } else {
            self.clone()
        };
        let rhs = if b.len() == 1 {
            other.expand_dims(-1)?
        } else {
            other.clone()
        };
        let (sa, sb) = (lhs.shape(), rhs.shape());
        if sa[sa.len() - 1] != sb[sb.len() - 2] {
            return refuse(format!("matmul: shapes {a:?} and {b:?}"));
        }
        let batch = broadcast_shapes("matmul", &sa[..sa.len() - 2], &sb[..sb.len() - 2])?;
        let with = |s: &[usize]| {
            let mut full = batch.clone();
            full.extend_from_slice(&s[s.len() - 2..]);
            full
        };
        let dtype = DType::promote(lhs.dtype(), rhs.dtype());
        if !dtype.is_floating() {
            return refuse(format!("matmul: {dtype:?} is not floating"));
        }
        let lhs = lhs.as_dtype(dtype)?.broadcast_to(&with(&sa))?;
        let rhs = rhs.as_dtype(dtype)?.broadcast_to(&with(&sb))?;
        let mut out = Self::wrap(Selected::matmul(&lhs.0, &rhs.0))?;
        if b.len() == 1 {
            out = out.squeeze(-1)?;
        }
        if a.len() == 1 {
            out = out.squeeze(if b.len() == 1 { -1 } else { -2 })?;
        }
        Ok(out)
    }

    /// `a` where this boolean tensor is true, else `b`, of any kind, in the dtype they meet in.
    pub fn select(&self, a: &Tensor, b: &Tensor) -> Result<Self> {
        if self.dtype() != DType::Bool {
            return refuse(format!("select: a condition of {:?}", self.dtype()));
        }
        let dtype = DType::promote(a.dtype(), b.dtype());
        let shape = broadcast_shapes("select", &a.shape(), &b.shape())?;
        let shape = broadcast_shapes("select", &self.shape(), &shape)?;
        let condition = self.broadcast_to(&shape)?;
        let a = a.as_dtype(dtype)?.broadcast_to(&shape)?;
        let b = b.as_dtype(dtype)?.broadcast_to(&shape)?;
        Self::wrap(Selected::select(&condition.0, &a.0, &b.0))
    }

    /// `self` held between `low` and `high`, each rounded to this tensor's dtype.
    pub fn clip(&self, low: f64, high: f64) -> Result<Self> {
        let dtype = self.dtype();
        self.maximum(&Self::scalar(low, dtype)?)?
            .minimum(&Self::scalar(high, dtype)?)
    }

    /// A new shape of the same number of elements; one axis of -1 takes what is left.
    pub fn reshape(&self, shape: &[isize]) -> Result<Self> {
        check_rank("reshape", shape.len())?;
        let total = self.size();
        let known: usize = shape
            .iter()
            .filter(|&&d| d >= 0)
            .map(|&d| d as usize)
            .product();
        let free = shape.iter().filter(|&&d| d < 0).count();
        if free > 1 || shape.iter().any(|&d| d < -1) {
            return refuse(format!("reshape: {shape:?}"));
        }
        let resolved: Vec<usize> = shape
            .iter()
            .map(|&d| {
                if d < 0 {
                    total.checked_div(known).unwrap_or(0)
                } else {
                    d as usize
                }
            })
            .collect();
        if elements(&resolved) != total {
            return refuse(format!(
                "reshape: {total} elements as {shape:?}, from {:?}",
                self.shape()
            ));
        }
        Self::wrap(Selected::reshape(&self.0, &resolved))
    }

    /// Reorders the axes: `axes` is a permutation of them.
    pub fn transpose(&self, axes: &[usize]) -> Result<Self> {
        let rank = self.rank();
        let mut seen = vec![false; rank];
        let valid = axes.len() == rank
            && axes.iter().all(|&a| {
                let fresh = a < rank && !seen[a];
                if fresh {
                    seen[a] = true;
                }
                fresh
            });
        if !valid {
            return refuse(format!("transpose: {axes:?} of a tensor of rank {rank}"));
        }
        Self::wrap(Selected::permute(&self.0, axes))
    }

    pub fn swap_axes(&self, a: isize, b: isize) -> Result<Self> {
        let rank = self.rank();
        let (a, b) = (
            normal_axis("swap_axes", a, rank)?,
            normal_axis("swap_axes", b, rank)?,
        );
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(a, b);
        Self::wrap(Selected::permute(&self.0, &axes))
    }

    /// The axes reversed.
    pub fn t(&self) -> Result<Self> {
        let axes: Vec<usize> = (0..self.rank()).rev().collect();
        Self::wrap(Selected::permute(&self.0, &axes))
    }

    /// A new axis of length 1 at `axis`, which counts in the result's axes.
    pub fn expand_dims(&self, axis: isize) -> Result<Self> {
        let mut shape = self.shape();
        check_rank("expand_dims", shape.len() + 1)?;
        let axis = normal_axis("expand_dims", axis, shape.len() + 1)?;
        shape.insert(axis, 1);
        Self::wrap(Selected::reshape(&self.0, &shape))
    }

    /// Removes `axis`, of length 1.
    pub fn squeeze(&self, axis: isize) -> Result<Self> {
        let mut shape = self.shape();
        let axis = normal_axis("squeeze", axis, shape.len())?;
        if shape[axis] != 1 {
            return refuse(format!("squeeze: axis {axis} of {shape:?}"));
        }
        shape.remove(axis);
        Self::wrap(Selected::reshape(&self.0, &shape))
    }

    /// Axes `start..=end` as one.
    pub fn flatten(&self, start: isize, end: isize) -> Result<Self> {
        let shape = self.shape();
        let start = normal_axis("flatten", start, shape.len())?;
        let end = normal_axis("flatten", end, shape.len())?;
        if start > end {
            return refuse(format!("flatten: {start} after {end}"));
        }
        let mut out = shape[..start].to_vec();
        out.push(elements(&shape[start..=end]));
        out.extend_from_slice(&shape[end + 1..]);
        Self::wrap(Selected::reshape(&self.0, &out))
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        check_rank("broadcast_to", shape.len())?;
        let own = self.shape();
        if own == shape {
            return Ok(self.clone());
        }
        if broadcast_shapes("broadcast_to", &own, shape)? != shape {
            return refuse(format!("broadcast_to: {own:?} to {shape:?}"));
        }
        Self::wrap(Selected::broadcast_to(&self.0, shape))
    }

    /// Joins tensors of one shape but for `axis`, in the dtype they meet in.
    pub fn concat(tensors: &[&Tensor], axis: isize) -> Result<Self> {
        let Some(first) = tensors.first() else {
            return refuse("concat: no tensors");
        };
        let rank = first.rank();
        let axis = normal_axis("concat", axis, rank)?;
        let dtype = tensors
            .iter()
            .fold(first.dtype(), |d, t| DType::promote(d, t.dtype()));
        let cast: Vec<Tensor> = tensors
            .iter()
            .map(|t| t.as_dtype(dtype))
            .collect::<Result<_>>()?;
        for t in &cast {
            let (a, b) = (first.shape(), t.shape());
            let same = a.len() == b.len()
                && a.iter()
                    .zip(&b)
                    .enumerate()
                    .all(|(i, (x, y))| i == axis || x == y);
            if !same {
                return refuse(format!("concat: {a:?} and {b:?} along axis {axis}"));
            }
        }
        let arrays: Vec<&A> = cast.iter().map(|t| &t.0).collect();
        Self::wrap(Selected::concat(&arrays, axis))
    }

    /// Joins tensors of one shape along a new axis.
    pub fn stack(tensors: &[&Tensor], axis: isize) -> Result<Self> {
        let expanded: Vec<Tensor> = tensors
            .iter()
            .map(|t| t.expand_dims(axis))
            .collect::<Result<_>>()?;
        let refs: Vec<&Tensor> = expanded.iter().collect();
        Self::concat(&refs, axis)
    }

    /// The tensor cut along `axis` before each of the positions `at`, which do not decrease.
    pub fn split_at(&self, at: &[usize], axis: isize) -> Result<Vec<Self>> {
        let shape = self.shape();
        let axis = normal_axis("split_at", axis, shape.len())?;
        let mut parts = Vec::with_capacity(at.len() + 1);
        let mut start = 0;
        for &end in at.iter().chain(std::iter::once(&shape[axis])) {
            parts.push(self.slice(axis as isize, start..end)?);
            start = end;
        }
        Ok(parts)
    }

    /// The elements at `indices`, integer and of rank 1, along `axis`.
    pub fn take(&self, indices: &Tensor, axis: isize) -> Result<Self> {
        let axis = normal_axis("take", axis, self.rank())?;
        if !matches!(indices.dtype(), DType::I32 | DType::I64) || indices.rank() != 1 {
            return refuse(format!(
                "take: indices of {:?}, {:?}",
                indices.dtype(),
                indices.shape()
            ));
        }
        Self::wrap(Selected::take(&self.0, &indices.0, axis))
    }

    /// `range` of `axis`.
    pub fn slice(&self, axis: isize, range: Range<usize>) -> Result<Self> {
        let shape = self.shape();
        let axis = normal_axis("slice", axis, shape.len())?;
        if range.start > range.end || range.end > shape[axis] {
            return refuse(format!("slice: {range:?} of axis {axis} of {shape:?}"));
        }
        Self::wrap(Selected::slice(&self.0, axis, range.start, range.end))
    }

    /// This tensor with `values` written at `indices` along `axis`: `indices` integer and `values` in
    /// this tensor's dtype after rounding, those two of one shape, which is this tensor's but along
    /// `axis`. Each place is written once: repeated indices leave which value lands to the side, and
    /// an index out of range is the caller's to avoid, as not every side checks it.
    pub fn put_along_axis(&self, indices: &Tensor, values: &Tensor, axis: isize) -> Result<Self> {
        if !self.dtype().is_floating() {
            return refuse(format!(
                "put_along_axis: {:?} is not floating",
                self.dtype()
            ));
        }
        if !matches!(indices.dtype(), DType::I32 | DType::I64) {
            return refuse(format!("put_along_axis: indices of {:?}", indices.dtype()));
        }
        let at = normal_axis("put_along_axis", axis, self.rank())?;
        let (own, shape) = (self.shape(), indices.shape());
        let fits = shape.len() == own.len()
            && shape
                .iter()
                .zip(&own)
                .enumerate()
                .all(|(i, (a, b))| i == at || a == b);
        if !fits || values.shape() != shape {
            return refuse(format!(
                "put_along_axis: {own:?} along {at}, indices {shape:?}, values {:?}",
                values.shape()
            ));
        }
        let values = values.as_dtype(self.dtype())?;
        Self::wrap(Selected::put_along_axis(&self.0, &indices.0, &values.0, at))
    }

    /// `before` and `after` more elements of `value` along `axis`.
    pub fn pad(&self, axis: isize, before: usize, after: usize, value: f64) -> Result<Self> {
        let shape = self.shape();
        let at = normal_axis("pad", axis, shape.len())?;
        let piece = |n: usize| {
            let mut s = shape.clone();
            s[at] = n;
            Self::full(&s, value, self.dtype())
        };
        let mut parts = Vec::new();
        if before > 0 {
            parts.push(piece(before)?);
        }
        parts.push(self.clone());
        if after > 0 {
            parts.push(piece(after)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Self::concat(&refs, axis)
    }

    fn reduce(&self, op: Reduce) -> Result<Self> {
        if !self.dtype().is_floating() {
            return refuse(format!("{op:?}: {:?} is not floating", self.dtype()));
        }
        Self::wrap(Selected::reduce(op, &self.0, None))
    }

    fn reduce_axis(&self, op: Reduce, axis: isize, keep: bool) -> Result<Self> {
        if !self.dtype().is_floating() {
            return refuse(format!("{op:?}: {:?} is not floating", self.dtype()));
        }
        let axis = normal_axis(&format!("{op:?}"), axis, self.rank())?;
        let kept = Self::wrap(Selected::reduce(op, &self.0, Some(axis)))?;
        if keep {
            Ok(kept)
        } else {
            kept.squeeze(axis as isize)
        }
    }

    pub fn sum(&self) -> Result<Self> {
        self.reduce(Reduce::Sum)
    }

    pub fn mean(&self) -> Result<Self> {
        self.reduce(Reduce::Mean)
    }

    pub fn max(&self) -> Result<Self> {
        self.reduce(Reduce::Max)
    }

    pub fn min(&self) -> Result<Self> {
        self.reduce(Reduce::Min)
    }

    pub fn sum_axis(&self, axis: isize, keep: bool) -> Result<Self> {
        self.reduce_axis(Reduce::Sum, axis, keep)
    }

    pub fn mean_axis(&self, axis: isize, keep: bool) -> Result<Self> {
        self.reduce_axis(Reduce::Mean, axis, keep)
    }

    pub fn max_axis(&self, axis: isize, keep: bool) -> Result<Self> {
        self.reduce_axis(Reduce::Max, axis, keep)
    }

    pub fn min_axis(&self, axis: isize, keep: bool) -> Result<Self> {
        self.reduce_axis(Reduce::Min, axis, keep)
    }

    /// `ln(sum(exp(self)))` over `axis`, without overflowing where the sum would. Where the largest element
    /// is infinite that is the answer, and the gradient along that row is NaN on every side. An empty
    /// tensor is refused.
    pub fn logsumexp_axis(&self, axis: isize, keep: bool) -> Result<Self> {
        if !self.dtype().is_floating() {
            return refuse(format!(
                "logsumexp_axis: {:?} is not floating",
                self.dtype()
            ));
        }
        if self.size() == 0 {
            return refuse(format!("logsumexp_axis: an empty {:?}", self.shape()));
        }
        let at = normal_axis("logsumexp_axis", axis, self.rank())?;
        let kept = match Selected::logsumexp(&self.0, at) {
            Some(kept) => Self::wrap(kept)?,
            None => {
                let top = self.max_axis(axis, true)?.stop_gradient()?;
                let sum = top.add(&self.sub(&top)?.exp()?.sum_axis(axis, true)?.log()?)?;
                // `inf - inf` would make it NaN.
                let infinity = Self::scalar(f64::INFINITY, self.dtype())?;
                top.abs()?.eq(&infinity)?.select(&top, &sum)?
            }
        };
        if keep { Ok(kept) } else { kept.squeeze(axis) }
    }

    /// The variance along `axis`, `ddof` the correction of the denominator, `n - ddof`.
    pub fn var_axis(&self, axis: isize, ddof: usize, keep: bool) -> Result<Self> {
        let at = normal_axis("var_axis", axis, self.rank())?;
        let n = self.shape()[at];
        if n <= ddof {
            return refuse(format!(
                "var_axis: {n} elements with a correction of {ddof}"
            ));
        }
        let centred = self.sub(&self.mean_axis(axis, true)?)?;
        centred
            .square()?
            .sum_axis(axis, keep)?
            .mul_scalar(1.0 / (n - ddof) as f64)
    }

    /// The position of the largest element along `axis`, `i32`.
    pub fn argmax(&self, axis: isize, keep: bool) -> Result<Self> {
        if !self.dtype().is_floating() {
            return refuse(format!("argmax: {:?} is not floating", self.dtype()));
        }
        let at = normal_axis("argmax", axis, self.rank())?;
        let kept = Self::wrap(Selected::argmax(&self.0, at))?;
        if keep {
            Ok(kept)
        } else {
            kept.squeeze(at as isize)
        }
    }

    /// Ascending along `axis`.
    pub fn sort(&self, axis: isize) -> Result<Self> {
        if !self.dtype().is_floating() {
            return refuse(format!("sort: {:?} is not floating", self.dtype()));
        }
        let at = normal_axis("sort", axis, self.rank())?;
        Self::wrap(Selected::sort(&self.0, at))
    }

    /// RMSNorm over the last axis: `x / sqrt(mean(x^2) + eps) * weight`, the weight given as its factor.
    /// It comes out in the dtype `x` and the weight promote to, rounded once: a bf16 `x` with an f32 weight
    /// gives f32, as MLX's does. In f32, or the side's own where it has one: MLX's `fast` function, as
    /// mlx-lm's stack runs it, which differs from the f32 form in low bits.
    pub fn rms_norm(&self, weight: Option<&Tensor>, eps: f64) -> Result<Self> {
        let dtype = self.dtype();
        let shape = self.shape();
        let Some(features) = shape.last().filter(|_| dtype.is_floating()) else {
            return refuse(format!("rms_norm: {dtype:?} of {shape:?}"));
        };
        if let Some(w) = weight
            && (!w.dtype().is_floating() || w.shape() != [*features])
        {
            return refuse(format!(
                "rms_norm: a weight of {:?} {:?} for {features} features",
                w.dtype(),
                w.shape()
            ));
        }
        let out = weight.map_or(dtype, |w| DType::promote(dtype, w.dtype()));
        if let Some(own) = Selected::rms_norm(&self.0, weight.map(|w| &w.0), eps as f32) {
            return Self::wrap(own)?.as_dtype(out);
        }
        let x = self.as_dtype(DType::F32)?;
        let scale = x.square()?.mean_axis(-1, true)?.add_scalar(eps)?.sqrt()?;
        let mut y = x.div(&scale)?;
        if let Some(weight) = weight {
            y = y.mul(&weight.as_dtype(DType::F32)?)?;
        }
        y.as_dtype(out)
    }
}
