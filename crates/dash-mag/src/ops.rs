//! The operations' entry points. What each computes, its layouts, dtypes, masks and options are
//! [`dash_contract`]'s, the same on both sides; this module checks the ranks and passes the call
//! on, so that a wrong rank or dtype is a refusal here on every side rather than a panic on one.

pub use dash_contract::attention::{AttentionOptions, Window};
pub use dash_contract::conv1d::{Activation, CausalConv1dOptions, CausalConv1dStepOptions};
pub use dash_contract::gated_delta::chunk::ChunkGatedDeltaOptions;
pub use dash_contract::gated_delta::recurrent::GatedDeltaStepOptions;
pub use dash_contract::logprob::LogprobOptions;
pub use dash_contract::lowrank::LowRankOptions;
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
