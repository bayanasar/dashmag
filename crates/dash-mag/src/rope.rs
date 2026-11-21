//! Rotary position embeddings, as transformers computes them, on every side: the features paired
//! as halves (transformers' `rotate_half`) or as neighbours (GPT-J's `rotate_every_two`, MLX's
//! `traditional`), and the frequencies as transformers' `rope_type`s scale them for a context
//! longer than training's ([`RopeScaling`]). A side with its own rotation (MLX's `fast` functions)
//! rotates with it for the halves at unscaled frequencies; otherwise the rotation is computed on
//! the host once and applied from tables, written once here.

use std::ops::Range;

use crate::arm::{Arm, RopeAt, Selected};
use crate::tensor::Tensor;
use crate::{DType, Result, refuse};

/// How a head's rotated features pair up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RopeLayout {
    /// Feature `i` with feature `i + dim / 2`: transformers' `x cos + rotate_half(x) sin`, as
    /// Llama, Qwen and most of transformers' models rotate.
    #[default]
    Halves,
    /// Feature `2i` with feature `2i + 1`: GPT-J's `rotate_every_two`, the original LLaMA's, and
    /// MLX's `traditional`.
    Pairs,
}

/// How the frequencies `1 / base^(2i / dim)` are scaled for a context longer than training's, as
/// transformers' `rope_type` (`modeling_rope_utils`, 5.15) computes them, in f32 as it does.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum RopeScaling {
    /// transformers' `default`.
    #[default]
    None,
    /// `linear`: every frequency divided by `factor`.
    Linear { factor: f32 },
    /// `dynamic` (NTK-aware): the base raised for the length the rotation holds, when that passes
    /// `max_positions`. transformers raises it for the longest position a forward pass has seen,
    /// and lowers it again for a short one: a [`Rope`] holding positions up to the sequence's
    /// longest rotates as it does there.
    Dynamic { factor: f32, max_positions: usize },
    /// `yarn`: each frequency between its interpolated and extrapolated value, by how many turns it
    /// makes over `original_max_positions` (between `beta_slow` and `beta_fast`), and the sines and
    /// cosines scaled by the attention factor: `attention_factor`, or from `mscale` and
    /// `mscale_all_dim` as transformers derives it. `factor` is the one transformers uses: 5.15
    /// takes `max_position_embeddings / original_max_position_embeddings` only where the
    /// configuration gives no `factor` (earlier versions took that ratio whenever it gave
    /// `original_max_position_embeddings`, as DeepSeek-V3's do); an application reading such a
    /// configuration passes the value its transformers would use.
    Yarn {
        factor: f32,
        original_max_positions: usize,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: Option<f32>,
        mscale: Option<f32>,
        mscale_all_dim: Option<f32>,
        /// Whether the correction range is rounded out to whole dimensions (transformers' default).
        truncate: bool,
    },
    /// `llama3`: low frequencies divided by `factor`, high ones kept, those between smoothed.
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_positions: usize,
    },
}

impl RopeScaling {
    /// YaRN as transformers defaults it: `beta_fast` 32, `beta_slow` 1, the attention factor from
    /// `factor`, the correction range truncated.
    pub fn yarn(factor: f32, original_max_positions: usize) -> Self {
        Self::Yarn {
            factor,
            original_max_positions,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
            mscale: None,
            mscale_all_dim: None,
            truncate: true,
        }
    }
}

/// `base^exponent` taken in f64 and rounded once to f32: the correctly rounded power for the bases
/// and widths checked, and the bits torch gave on the recordings. `f32::powf` is the platform's
/// `powf`, which is not correctly rounded everywhere: macOS's libm gives 0x405b01d9 for
/// 500000^(3/32) against 0x405b01d8, and glibc 2.39 and 2.41 give 0x424f2dd5 for 1e6 raised to
/// `32.0 / 112.0` in f32 against 0x424f2dd4. The exponent is always the f32 quotient, as torch's
/// `arange / dim` gives it.
fn power(base: f32, exponent: f32) -> f32 {
    (base as f64).powf(exponent as f64) as f32
}

/// The frequencies of a rotation of `dim` features at `base`, scaled by `scaling` for a rotation
/// holding `len` positions, and the factor on its sines and cosines: transformers' `inv_freq` and
/// `attention_scaling`, in f32 as it computes them.
pub fn frequencies(dim: usize, base: f32, scaling: RopeScaling, len: usize) -> (Vec<f32>, f32) {
    let half = dim / 2;
    let at = |base: f32| -> Vec<f32> {
        (0..half)
            .map(|i| 1.0 / power(base, (2 * i) as f32 / dim as f32))
            .collect()
    };
    match scaling {
        RopeScaling::None => (at(base), 1.0),
        RopeScaling::Linear { factor } => (at(base).into_iter().map(|f| f / factor).collect(), 1.0),
        RopeScaling::Dynamic {
            factor,
            max_positions,
        } => {
            if len <= max_positions {
                return (at(base), 1.0);
            }
            // As the forward pass computes it, from the length as a tensor: in f32, but for the
            // power, which torch takes in f64 and rounds to f32.
            let grown = factor * len as f32 / max_positions as f32 - (factor as f64 - 1.0) as f32;
            let exponent = dim as f64 / (dim as f64 - 2.0);
            let raised = (grown as f64).powf(exponent) as f32;
            (at(base * raised), 1.0)
        }
        RopeScaling::Yarn {
            factor,
            original_max_positions,
            beta_fast,
            beta_slow,
            attention_factor,
            mscale,
            mscale_all_dim,
            truncate,
        } => {
            let get_mscale = |scale: f64, m: f64| {
                if scale <= 1.0 {
                    1.0
                } else {
                    0.1 * m * scale.ln() + 1.0
                }
            };
            let factor64 = factor as f64;
            let attention = match (attention_factor, mscale, mscale_all_dim) {
                (Some(a), _, _) => a as f64,
                (None, Some(m), Some(all)) if m != 0.0 && all != 0.0 => {
                    get_mscale(factor64, m as f64) / get_mscale(factor64, all as f64)
                }
                _ => get_mscale(factor64, 1.0),
            };
            // The correction range, in f64 as transformers' Python computes it.
            let (dim64, base64) = (dim as f64, base as f64);
            let correction = |turns: f64| {
                dim64 * (original_max_positions as f64 / (turns * 2.0 * std::f64::consts::PI)).ln()
                    / (2.0 * base64.ln())
            };
            let (mut low, mut high) = (correction(beta_fast as f64), correction(beta_slow as f64));
            if truncate {
                low = low.floor();
                high = high.ceil();
            }
            let (low, mut high) = (low.max(0.0), high.min(dim64 - 1.0));
            if low == high {
                high += 0.001;
            }
            let inverse: Vec<f32> = (0..half)
                .map(|i| power(base, (2 * i) as f32 / dim as f32))
                .collect();
            let frequencies = inverse
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let ramp = ((i as f32 - low as f32) / (high - low) as f32).clamp(0.0, 1.0);
                    let extrapolated = 1.0 - ramp;
                    let (interpolation, extrapolation) = (1.0 / (factor * p), 1.0 / p);
                    interpolation * (1.0 - extrapolated) + extrapolation * extrapolated
                })
                .collect();
            (frequencies, attention as f32)
        }
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_positions,
        } => {
            let old = original_max_positions as f64;
            let low_wavelength = (old / low_freq_factor as f64) as f32;
            let high_wavelength = (old / high_freq_factor as f64) as f32;
            let span = (high_freq_factor as f64 - low_freq_factor as f64) as f32;
            let two_pi = (2.0 * std::f64::consts::PI) as f32;
            let frequencies = at(base)
                .into_iter()
                .map(|f| {
                    let wavelength = two_pi / f;
                    let scaled = if wavelength > low_wavelength {
                        f / factor
                    } else {
                        f
                    };
                    // transformers' `~(wavelen < high) * ~(wavelen > low)`: the wavelengths are
                    // finite, so the negated comparisons are these.
                    let medium = wavelength >= high_wavelength && wavelength <= low_wavelength;
                    if !medium {
                        return scaled;
                    }
                    let smooth = (old as f32 / wavelength - low_freq_factor) / span;
                    (1.0 - smooth) * scaled / factor + smooth * scaled
                })
                .collect();
            (frequencies, 1.0)
        }
    }
}

/// How a [`Rope`] pairs its features and scales its frequencies; the defaults are transformers'.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RopeOptions {
    pub layout: RopeLayout,
    pub scaling: RopeScaling,
}

/// The rotation of positions `0..len` over the first `dim` features of a head, as transformers
/// computes it: by default `x cos + rotate_half(x) sin` at the frequencies `1 / base^(2i / dim)` in
/// f32, or as [`RopeOptions`] pair and scale them ([`frequencies`]). From tables, each angle's sine
/// and cosine are rounded once to f32, scaled by the attention factor in f32 where YaRN gives one,
/// and rounded once to `dtype`. The features after `dim` pass unchanged.
#[derive(Clone, Debug)]
pub struct Rope {
    /// The cosines and sines `[len, dim]`, on a side without its own rotation.
    tables: Option<(Tensor, Tensor)>,
    dim: usize,
    base: f32,
    len: usize,
    dtype: DType,
    layout: RopeLayout,
}

impl Rope {
    /// transformers' default rotation: the halves, at unscaled frequencies.
    pub fn new(dim: usize, base: f32, len: usize, dtype: DType) -> Result<Self> {
        Self::with_options(dim, base, len, dtype, RopeOptions::default())
    }

    pub fn with_options(
        dim: usize,
        base: f32,
        len: usize,
        dtype: DType,
        options: RopeOptions,
    ) -> Result<Self> {
        if dim == 0 || !dim.is_multiple_of(2) {
            return refuse(format!("Rope: {dim} features; an even number is rotated"));
        }
        if !dtype.is_floating() {
            return refuse(format!("Rope: {dtype:?} is not floating"));
        }
        let layout = options.layout;
        if Selected::HAS_ROPE && options == RopeOptions::default() {
            return Ok(Self {
                tables: None,
                dim,
                base,
                len,
                dtype,
                layout,
            });
        }
        let (frequencies, attention) = frequencies(dim, base, options.scaling, len);
        let half = dim / 2;
        let mut cos = Vec::with_capacity(len * dim);
        let mut sin = Vec::with_capacity(len * dim);
        for position in 0..len {
            for j in 0..dim {
                let i = match layout {
                    RopeLayout::Halves => j % half,
                    RopeLayout::Pairs => j / 2,
                };
                let angle = position as f32 * frequencies[i];
                cos.push((angle as f64).cos() as f32 * attention);
                sin.push((angle as f64).sin() as f32 * attention);
            }
        }
        Ok(Self {
            tables: Some((
                Tensor::from_f32(&cos, &[len, dim])?.as_dtype(dtype)?,
                Tensor::from_f32(&sin, &[len, dim])?.as_dtype(dtype)?,
            )),
            dim,
            base,
            len,
            dtype,
            layout,
        })
    }

    /// The last position this holds, plus one.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn check_features(&self, x: &Tensor) -> Result<usize> {
        if x.dtype() != self.dtype {
            return refuse(format!(
                "Rope: {:?} under a rotation of {:?}",
                x.dtype(),
                self.dtype
            ));
        }
        let d = *x.shape().last().unwrap_or(&0);
        if d < self.dim {
            return refuse(format!(
                "Rope: {} features to rotate, of {:?}",
                self.dim,
                x.shape()
            ));
        }
        Ok(d)
    }

    fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let d = self.check_features(x)?;
        let half = self.dim / 2;
        let turned = x.slice(-1, 0..self.dim)?;
        let rotated = match self.layout {
            RopeLayout::Halves => {
                let first = turned.slice(-1, 0..half)?;
                let second = turned.slice(-1, half..self.dim)?;
                Tensor::concat(&[&second.neg()?, &first], -1)?
            }
            RopeLayout::Pairs => {
                // (x0, x1, x2, x3, ..) to (-x1, x0, -x3, x2, ..).
                let mut pairs: Vec<isize> = turned.shape().iter().map(|&n| n as isize).collect();
                let last = pairs.len() - 1;
                pairs[last] = half as isize;
                pairs.push(2);
                let paired = turned.reshape(&pairs)?;
                let even = paired.slice(-1, 0..1)?;
                let odd = paired.slice(-1, 1..2)?;
                let shape: Vec<isize> = turned.shape().iter().map(|&n| n as isize).collect();
                Tensor::concat(&[&odd.neg()?, &even], -1)?.reshape(&shape)?
            }
        };
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
        let Some((cos, sin)) = &self.tables else {
            // The side's own takes `[B, H, T, D]`.
            self.check_features(x)?;
            let turned = x.swap_axes(1, 2)?;
            let at = RopeAt::Start(positions.start);
            let out = Tensor::wrap(Selected::rope(&turned.0, self.dim, self.base, at))?;
            return out.swap_axes(1, 2);
        };
        let table = |t: &Tensor| {
            t.slice(0, positions.clone())?.reshape(&[
                1,
                positions.len() as isize,
                1,
                self.dim as isize,
            ])
        };
        self.rotate(x, &table(cos)?, &table(sin)?)
    }

    /// `x` `[B, H, D]`, one token per sequence, at `positions` `[B]`, integer. A position past [`Rope::len`] is the
    /// caller's to avoid: a check would cost a read of `positions`, which a pipelined decoder cannot wait
    /// for, and not every side stops there. Flex panics, CUDA can return garbage, and MLX, with no table,
    /// rotates as asked.
    pub fn apply_at(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        let integer = matches!(positions.dtype(), DType::I32 | DType::I64);
        if !integer || shape.len() != 3 || positions.shape() != [shape[0]] {
            return refuse(format!(
                "Rope: positions of {:?} {:?} for {shape:?}",
                positions.dtype(),
                positions.shape()
            ));
        }
        let Some((cos, sin)) = &self.tables else {
            // The side's own takes `[B, H, 1, D]`.
            self.check_features(x)?;
            let (b, h, d) = (shape[0] as isize, shape[1] as isize, shape[2] as isize);
            let turned = x.reshape(&[b, h, 1, d])?;
            let at = RopeAt::Rows(&positions.0);
            let out = Tensor::wrap(Selected::rope(&turned.0, self.dim, self.base, at))?;
            return out.reshape(&[b, h, d]);
        };
        let table = |t: &Tensor| {
            t.take(positions, 0)?
                .reshape(&[shape[0] as isize, 1, self.dim as isize])
        };
        self.rotate(x, &table(cos)?, &table(sin)?)
    }
}
