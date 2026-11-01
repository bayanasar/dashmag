//! Rotary position embeddings, computed on the host once and applied on either side alike.

use std::ops::Range;

use crate::tensor::Tensor;
use crate::{DType, Result, refuse};

/// The rotation of positions `0..len` over the first `dim` features of a head: transformers' form,
/// `x cos + rotate_half(x) sin`, the frequencies `1 / base^(2i / dim)` in f32, each angle's sine and
/// cosine rounded once to f32 and once to `dtype`. The features after `dim` pass unchanged.
#[derive(Clone, Debug)]
pub struct Rope {
    cos: Tensor,
    sin: Tensor,
    dim: usize,
}

impl Rope {
    pub fn new(dim: usize, base: f32, len: usize, dtype: DType) -> Result<Self> {
        if dim == 0 || !dim.is_multiple_of(2) {
            return refuse(format!("Rope: {dim} features; an even number is rotated"));
        }
        let half = dim / 2;
        let frequencies: Vec<f32> = (0..half)
            .map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(len * dim);
        let mut sin = Vec::with_capacity(len * dim);
        for position in 0..len {
            for j in 0..dim {
                let angle = position as f32 * frequencies[j % half];
                cos.push((angle as f64).cos() as f32);
                sin.push((angle as f64).sin() as f32);
            }
        }
        Ok(Self {
            cos: Tensor::from_f32(&cos, &[len, dim])?.as_dtype(dtype)?,
            sin: Tensor::from_f32(&sin, &[len, dim])?.as_dtype(dtype)?,
            dim,
        })
    }

    /// The last position this holds, plus one.
    pub fn len(&self) -> usize {
        self.cos.shape()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let d = *x.shape().last().unwrap_or(&0);
        if d < self.dim {
            return refuse(format!(
                "Rope: {} features to rotate, of {:?}",
                self.dim,
                x.shape()
            ));
        }
        let half = self.dim / 2;
        let turned = x.slice(-1, 0..self.dim)?;
        let first = turned.slice(-1, 0..half)?;
        let second = turned.slice(-1, half..self.dim)?;
        let rotated = Tensor::concat(&[&second.neg()?, &first], -1)?;
        let out = turned.mul(cos)?.add(&rotated.mul(sin)?)?;
        if d == self.dim {
            return Ok(out);
        }
        Tensor::concat(&[&out, &x.slice(-1, self.dim..d)?], -1)
    }

    /// `x` `[B, T, H, D]`, at the positions `positions` of every sequence, `T` of them.
    pub fn apply_range(&self, x: &Tensor, positions: Range<usize>) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 4 || shape[1] != positions.len() || positions.end > self.len() {
            return refuse(format!(
                "Rope: positions {positions:?} of {} for {shape:?}",
                self.len()
            ));
        }
        let table = |t: &Tensor| {
            t.slice(0, positions.clone())?.reshape(&[
                1,
                positions.len() as isize,
                1,
                self.dim as isize,
            ])
        };
        self.rotate(x, &table(&self.cos)?, &table(&self.sin)?)
    }

    /// `x` `[B, H, D]`, one token per sequence, at `positions` `[B]`.
    pub fn apply_at(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 3 || positions.shape() != [shape[0]] {
            return refuse(format!(
                "Rope: positions {:?} for {shape:?}",
                positions.shape()
            ));
        }
        let table = |t: &Tensor| {
            t.take(positions, 0)?
                .reshape(&[shape[0] as isize, 1, self.dim as isize])
        };
        self.rotate(x, &table(&self.cos)?, &table(&self.sin)?)
    }
}
