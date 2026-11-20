//! The operations' entry points. What each computes, its layouts, dtypes, masks and options are
//! [`dash_contract`]'s, the same on both sides; this module checks the ranks and passes the call
//! on, so that a wrong rank or dtype is a refusal here on every side rather than a panic on one.

pub use dash_contract::attention::{AttentionOptions, Window};
pub use dash_contract::conv1d::{Activation, CausalConv1dOptions, CausalConv1dStepOptions};
pub use dash_contract::gated_delta::chunk::ChunkGatedDeltaOptions;
pub use dash_contract::gated_delta::recurrent::GatedDeltaStepOptions;
pub use dash_contract::logprob::LogprobOptions;
pub use dash_contract::lowrank::LowRankOptions;
pub use dash_contract::quant::{QuantFormat, QuantOptions};
pub use dash_contract::sample::SampleOptions;

use crate::arm::{Arm, Mask, Selected};
use crate::tensor::Tensor;
use crate::{DType, Result, refuse};

/// Attention's mask: true where a key is left out, or packed sequences' offsets.
pub type AttentionMask = dash_contract::attention::AttentionMask<Tensor, Tensor>;

fn rank(op: &str, name: &str, t: &Tensor, want: usize) -> Result<()> {
    if t.rank() != want {
        return refuse(format!(
            "{op}: {name} has shape {:?}, of rank {want} expected",
            t.shape()
        ));
    }
    Ok(())
}

fn integer(op: &str, name: &str, t: &Tensor) -> Result<()> {
    rank(op, name, t, 1)?;
    if !matches!(t.dtype(), DType::I32 | DType::I64) {
        return refuse(format!("{op}: {name} of {:?}", t.dtype()));
    }
    Ok(())
}

/// The operations' inputs are dash-contract's dtypes, f32, bf16 and f16, on every side: the kernels
/// have no f64, MLX's GPU has none, and an integer has no place here.
fn contract_dtypes<'a>(
    op: &str,
    tensors: impl IntoIterator<Item = (&'a str, &'a Tensor)>,
) -> Result<()> {
    for (name, t) in tensors {
        if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return refuse(format!("{op}: {name} of {:?}", t.dtype()));
        }
    }
    Ok(())
}

fn option(t: Option<&Tensor>) -> Option<&crate::tensor::A> {
    t.map(|t| &t.0)
}

/// Attention: `query` `[B, Tq, Hq, D]`, `key` and `value` `[B, Tk, Hkv, D]`.
pub fn attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: &AttentionMask,
    options: AttentionOptions,
) -> Result<Tensor> {
    let op = "attention";
    rank(op, "query", query, 4)?;
    rank(op, "key", key, 4)?;
    rank(op, "value", value, 4)?;
    contract_dtypes(op, [("query", query), ("key", key), ("value", value)])?;
    let mask = match mask {
        AttentionMask::None => Mask::None,
        AttentionMask::KeyPadding(m) => {
            rank(op, "the key padding", m, 2)?;
            if m.dtype() != DType::Bool {
                return refuse(format!("{op}: a key padding of {:?}", m.dtype()));
            }
            Mask::KeyPadding(&m.0)
        }
        AttentionMask::Packed {
            cu_seqlens_q,
            cu_seqlens_k,
        } => {
            integer(op, "cu_seqlens_q", cu_seqlens_q)?;
            integer(op, "cu_seqlens_k", cu_seqlens_k)?;
            Mask::Packed {
                cu_seqlens_q: &cu_seqlens_q.0,
                cu_seqlens_k: &cu_seqlens_k.0,
            }
        }
    };
    Tensor::wrap(Selected::attention(
        &query.0, &key.0, &value.0, mask, options,
    ))
}

/// A query per sequence, `[B, Hq, D]`, against a cache of keys and values `[B, S, Hkv, D]`.
pub fn attention_decode(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    cache_lengths: Option<&Tensor>,
    options: AttentionOptions,
) -> Result<Tensor> {
    let op = "attention_decode";
    rank(op, "query", query, 3)?;
    rank(op, "key", key, 4)?;
    rank(op, "value", value, 4)?;
    contract_dtypes(op, [("query", query), ("key", key), ("value", value)])?;
    if let Some(lengths) = cache_lengths {
        integer(op, "cache_lengths", lengths)?;
    }
    Tensor::wrap(Selected::attention_decode(
        &query.0,
        &key.0,
        &value.0,
        option(cache_lengths),
        options,
    ))
}

/// Causal depthwise conv1d: `x` `[B, T, C]`, `weight` `[C, K]`; returns the output and, when the
/// options ask, the final state.
pub fn causal_conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    initial_state: Option<&Tensor>,
    cu_seqlens: Option<&Tensor>,
    options: CausalConv1dOptions,
) -> Result<(Tensor, Option<Tensor>)> {
    let op = "causal_conv1d";
    rank(op, "x", x, 3)?;
    rank(op, "weight", weight, 2)?;
    if let Some(bias) = bias {
        rank(op, "bias", bias, 1)?;
    }
    if let Some(state) = initial_state {
        rank(op, "the initial state", state, 3)?;
    }
    if let Some(offsets) = cu_seqlens {
        integer(op, "cu_seqlens", offsets)?;
    }
    contract_dtypes(
        op,
        [("x", x), ("weight", weight)]
            .into_iter()
            .chain(bias.map(|t| ("bias", t)))
            .chain(initial_state.map(|t| ("the initial state", t))),
    )?;
    let (out, state) = Selected::causal_conv1d(
        &x.0,
        &weight.0,
        option(bias),
        option(initial_state),
        option(cu_seqlens),
        options,
    )?;
    Ok((Tensor(out), state.map(Tensor)))
}

/// One decoding step: `x` `[B, C]`, `state` `[B, W, C]`; returns the output and the next state.
pub fn causal_conv1d_update(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    state: &Tensor,
    options: CausalConv1dStepOptions,
) -> Result<(Tensor, Tensor)> {
    let op = "causal_conv1d_update";
    rank(op, "x", x, 2)?;
    rank(op, "weight", weight, 2)?;
    if let Some(bias) = bias {
        rank(op, "bias", bias, 1)?;
    }
    rank(op, "state", state, 3)?;
    contract_dtypes(
        op,
        [("x", x), ("weight", weight), ("state", state)]
            .into_iter()
            .chain(bias.map(|t| ("bias", t))),
    )?;
    let (out, state) =
        Selected::causal_conv1d_update(&x.0, &weight.0, option(bias), &state.0, options)?;
    Ok((Tensor(out), Tensor(state)))
}

/// The chunked gated delta rule: `q`, `k`, `v` `[B, T, H, D]`, `g`, `beta` `[B, T, H]`; returns the
/// output and, when the options ask, the final state.
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: Option<&Tensor>,
    cu_seqlens: Option<&Tensor>,
    options: ChunkGatedDeltaOptions,
) -> Result<(Tensor, Option<Tensor>)> {
    let op = "chunk_gated_delta_rule";
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        rank(op, name, t, 4)?;
    }
    rank(op, "g", g, 3)?;
    rank(op, "beta", beta, 3)?;
    if let Some(state) = initial_state {
        rank(op, "the initial state", state, 4)?;
    }
    if let Some(offsets) = cu_seqlens {
        integer(op, "cu_seqlens", offsets)?;
    }
    contract_dtypes(
        op,
        [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]
            .into_iter()
            .chain(initial_state.map(|t| ("the initial state", t))),
    )?;
    let (out, state) = Selected::chunk_gated_delta_rule(
        &q.0,
        &k.0,
        &v.0,
        &g.0,
        &beta.0,
        option(initial_state),
        option(cu_seqlens),
        options,
    )?;
    Ok((Tensor(out), state.map(Tensor)))
}

/// The gated delta rule a token at a time: `q`, `k`, `v` `[B, H, D]`, `g`, `beta` `[B, H]`, `state`
/// `[B, H, Dk, Dv]`; returns the output and the next state.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_step(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    cu_seqlens: Option<&Tensor>,
    options: GatedDeltaStepOptions,
) -> Result<(Tensor, Tensor)> {
    let op = "gated_delta_step";
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        rank(op, name, t, 3)?;
    }
    rank(op, "g", g, 2)?;
    rank(op, "beta", beta, 2)?;
    rank(op, "state", state, 4)?;
    if let Some(offsets) = cu_seqlens {
        integer(op, "cu_seqlens", offsets)?;
    }
    contract_dtypes(
        op,
        [
            ("q", q),
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("state", state),
        ],
    )?;
    let (out, state) = Selected::gated_delta_step(
        &q.0,
        &k.0,
        &v.0,
        &g.0,
        &beta.0,
        &state.0,
        option(cu_seqlens),
        options,
    )?;
    Ok((Tensor(out), Tensor(state)))
}

/// The log-probabilities and entropies [`chosen_logprob`] returns, `[n]` f32 each.
#[derive(Clone, Debug)]
pub struct ChosenLogprob {
    pub logprob: Tensor,
    pub entropy: Tensor,
}

/// The chosen tokens' log-probabilities over a large vocabulary with a frozen output weight:
/// `hidden` `[n, D]`, `weight` `[V, D]`, `targets` `[n]`, `masked` the sorted ids left out.
pub fn chosen_logprob(
    hidden: &Tensor,
    weight: &Tensor,
    targets: &Tensor,
    masked: &[u32],
    options: LogprobOptions,
) -> Result<ChosenLogprob> {
    let op = "chosen_logprob";
    rank(op, "hidden", hidden, 2)?;
    rank(op, "weight", weight, 2)?;
    integer(op, "targets", targets)?;
    contract_dtypes(op, [("hidden", hidden), ("weight", weight)])?;
    let (logprob, entropy) =
        Selected::chosen_logprob(&hidden.0, &weight.0, &targets.0, masked, options)?;
    Ok(ChosenLogprob {
        logprob: Tensor(logprob),
        entropy: Tensor(entropy),
    })
}

/// What [`sample`] returns, `[B]` each.
#[derive(Clone, Debug)]
pub struct Sampled {
    /// The token drawn for each row, `i32`.
    pub tokens: Tensor,
    /// Its log-probability under the model, at temperature 1. f32.
    pub logprob: Tensor,
    /// Its log-probability at the temperature, nothing cut; 0 when greedy. f32.
    pub tempered_logprob: Tensor,
    /// Its log-probability under the distribution it was drawn from; 0 when greedy. f32.
    pub sampling_logprob: Tensor,
}

/// The next token of each row, drawn on the device: `logits` `[B, V]`.
pub fn sample(logits: &Tensor, options: &SampleOptions) -> Result<Sampled> {
    rank("sample", "logits", logits, 2)?;
    contract_dtypes("sample", [("logits", logits)])?;
    let drawn = Selected::sample(&logits.0, options)?;
    Ok(Sampled {
        tokens: Tensor(drawn.tokens),
        logprob: Tensor(drawn.logprob),
        tempered_logprob: Tensor(drawn.tempered_logprob),
        sampling_logprob: Tensor(drawn.sampling_logprob),
    })
}

/// `base + scale · (x U) V`, the sum rounded once to the base's dtype; without a base, the term in
/// f32: `x` `[.., in]`, `u` `[in, r]` and `v` `[r, out]` f32.
pub fn low_rank(
    x: &Tensor,
    u: &Tensor,
    v: &Tensor,
    base: Option<&Tensor>,
    options: LowRankOptions,
) -> Result<Tensor> {
    let op = "low_rank";
    if x.rank() == 0 {
        return refuse(format!("{op}: x has shape {:?}", x.shape()));
    }
    rank(op, "u", u, 2)?;
    rank(op, "v", v, 2)?;
    if let Some(base) = base {
        rank(op, "base", base, x.rank())?;
    }
    contract_dtypes(
        op,
        [("x", x), ("u", u), ("v", v)]
            .into_iter()
            .chain(base.map(|t| ("base", t))),
    )?;
    Tensor::wrap(Selected::low_rank(&x.0, &u.0, &v.0, option(base), options))
}

/// A weight `[in, out]` quantized to 4 bits by group, as GPTQ stores it, checked once and kept as
/// the side runs it (on Burn the packed `qweight` and the f16 `scales` as stored, on MLX the words
/// in MLX's layout and the scales transposed), for [`quantized_linear`]. `dash_contract::quant`
/// gives the layout and the definition.
#[derive(Clone)]
pub struct QuantizedWeight {
    qweight: Tensor,
    scales: Tensor,
    inp: usize,
    out: usize,
    options: QuantOptions,
}

impl QuantizedWeight {
    /// The weight from a checkpoint's tensors as stored: `qweight` `[in / 8, out]` and `qzeros`
    /// `[in / group, out / 8]` `i32`, `scales` `[in / group, out]` f16, `g_idx` `[in]` `i32` or
    /// none. Refuses shapes that disagree, `in` not a multiple of the group or `out` of 8, scales
    /// not f16, a stored zero other than 8 (GPTQ v1's `zero - 1` among them), or a `g_idx` other
    /// than `i / group`, and on MLX a group other than 128. Reads `qzeros` and `g_idx` to the host
    /// once.
    pub fn from_gptq(
        qweight: &Tensor,
        qzeros: &Tensor,
        scales: &Tensor,
        g_idx: Option<&Tensor>,
        options: QuantOptions,
    ) -> Result<Self> {
        let op = "QuantizedWeight::from_gptq";
        let group = options.group;
        rank(op, "qweight", qweight, 2)?;
        let (packed, out) = (qweight.shape()[0], qweight.shape()[1]);
        let inp = packed * 8;
        if group == 0 || !inp.is_multiple_of(group) || !out.is_multiple_of(8) {
            return refuse(format!(
                "{op}: in = {inp} must be a multiple of the group ({group}), and out = {out} of 8"
            ));
        }
        let groups = inp / group;
        if scales.shape() != [groups, out] || qzeros.shape() != [groups, out / 8] {
            return refuse(format!(
                "{op}: qweight [{packed}, {out}] wants scales [{groups}, {out}] and qzeros \
                 [{groups}, {}], got {:?} and {:?}",
                out / 8,
                scales.shape(),
                qzeros.shape()
            ));
        }
        if qweight.dtype() != DType::I32 || qzeros.dtype() != DType::I32 {
            return refuse(format!("{op}: qweight and qzeros are i32"));
        }
        if scales.dtype() != DType::F16 {
            return refuse(format!("{op}: scales are f16, got {:?}", scales.dtype()));
        }
        let QuantFormat::Symmetric = options.format;
        let zero = QuantOptions::SYMMETRIC_ZERO as i32;
        let words = qzeros.to_i32()?;
        if let Some(i) = words
            .iter()
            .position(|&w| (0..8).any(|j| (w >> (4 * j)) & 15 != zero))
        {
            return refuse(format!(
                "{op}: a symmetric weight's stored zeros are all {zero}, but word {i} of qzeros \
                 is {:#010x} (GPTQ v1 stores zero - 1, 0x77777777)",
                words[i] as u32
            ));
        }
        if let Some(g_idx) = g_idx {
            if g_idx.shape() != [inp] {
                return refuse(format!("{op}: g_idx is [{inp}], got {:?}", g_idx.shape()));
            }
            let g = g_idx.to_i32()?;
            if let Some(i) = (0..inp).find(|&i| g[i] as usize != i / group) {
                return refuse(format!(
                    "{op}: g_idx[{i}] is {}, where groups in order give {} (reordered groups, \
                     GPTQ's desc_act, are not taken)",
                    g[i],
                    i / group
                ));
            }
        }
        let (qweight, scales) = Selected::prepare_quantized(&qweight.0, &scales.0, group)?;
        Ok(Self {
            qweight: Tensor::wrap(Ok(qweight))?,
            scales: Tensor::wrap(Ok(scales))?,
            inp,
            out,
            options,
        })
    }

    /// `in`, the inputs.
    pub fn in_features(&self) -> usize {
        self.inp
    }

    /// `out`, the outputs.
    pub fn out_features(&self) -> usize {
        self.out
    }

    /// The options it was built with.
    pub fn options(&self) -> QuantOptions {
        self.options
    }
}

/// `y = x W (+ bias)` over `weight`, for `x` `[.., in]` in f32, bf16 or f16 and `bias` `[out]`,
/// the output in `x`'s dtype. The definition is the product in f32, the bias added and the sum
/// rounded once (`dash_contract::quant`); a side may round `W`, and the product before the bias,
/// to `x`'s dtype, and is held to the definition within the bounds its tests record. Inside
/// [`crate::vjp`], `x` and the bias take their gradients by the definition's backward
/// (`dash_contract::quant`), the weight frozen. On MLX the scales are cast to `x`'s dtype on each
/// call; MLX sums in f32 and gives the product in `x`'s dtype, and past a few rows it forms `W` in
/// that dtype. So does its backward for `x` from 4 rows on; below them, in bf16 and f16, `x`'s
/// gradient is taken in f32 with `W` exact (`dash_mlx::quantized`).
pub fn quantized_linear(
    x: &Tensor,
    weight: &QuantizedWeight,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let op = "quantized_linear";
    if crate::autodiff::in_vjp() && !Selected::QUANTIZED_BACKWARD {
        return refuse(format!(
            "{op}: its backward pass is not on this side yet: not inside vjp"
        ));
    }
    if x.rank() < 2 || x.shape()[x.rank() - 1] != weight.inp {
        return refuse(format!(
            "{op}: x [.., {}] of rank 2 or more, got {:?}",
            weight.inp,
            x.shape()
        ));
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return refuse(format!("{op}: x is f32, bf16 or f16, got {:?}", x.dtype()));
    }
    if let Some(bias) = bias
        && (bias.shape() != [weight.out] || !bias.dtype().is_floating())
    {
        return refuse(format!(
            "{op}: bias is [{}] in a float dtype, got {:?} {:?}",
            weight.out,
            bias.shape(),
            bias.dtype()
        ));
    }
    Tensor::wrap(Selected::quantized_linear(
        &x.0,
        &weight.qweight.0,
        &weight.scales.0,
        option(bias),
        weight.options.group,
    ))
}
