//! Scaled dot-product attention with grouped key-value heads and the masks models train with:
//! none, causality, key padding, packed variable-length sequences and a sliding window; its
//! backward pass; and decoding against a key-value cache.
//!
//! # The operation
//!
//! For each batch element, query head `h` and query `i`, over the keys `j` the query sees:
//!
//! ```text
//! out[i] = sum_j p[i, j] v[j],   p[i, j] = exp(s[i, j] - lse[i]),   s[i, j] = scale q[i] . k[j]
//! lse[i] = log sum_j exp(s[i, j])
//! ```
//!
//! with the keys and values of key-value head `h / (Hq / Hkv)`: `Hq` is a multiple of `Hkv`, as
//! with PyTorch's `enable_gqa=True`. `scale` defaults to `1 / sqrt(D)`.
//!
//! For training, `attention` returns the output and `attention_with_lse` beside it the natural
//! log-sum-exp of each query's scaled scores over the keys it sees; gradients flow to the queries,
//! keys and values, and `attention_backward` gives them from the forward's output and log-sum-exp
//! and the output's gradient, for callers running their own backward pass.
//!
//! # Shapes and dtypes
//!
//! The crate's layout: queries `[B, Tq, Hq, D]`, keys and values `[B, Tk, Hkv, D]`, the output
//! `[B, Tq, Hq, D]` and the log-sum-exp `[B, Tq, Hq]`, the order a projection's `[B, T, H * D]`
//! output reshapes to without a copy (PyTorch's `scaled_dot_product_attention` takes the heads
//! before the time, `[B, H, T, D]`; flash-attn takes this order). `Tq` and `Tk` are independent.
//! A decoding step drops the time: one query per sequence, `[B, Hq, D]`, against the cache
//! `[B, S, Hkv, D]`, returns `[B, Hq, D]`. Queries, keys and values share one float dtype (f32,
//! bf16 or f16), which the output takes; the log-sum-exp is f32; everything is accumulated in
//! f32, and each gradient is rounded once, to its input's dtype.
//!
//! # Which keys a query sees
//!
//! Positions are aligned at the bottom right: query `i` sits at key position `i + Tk - Tq`, so
//! that with `Tq == Tk` query `i` is key `i`, a chunk of queries continuing a cached prefix
//! comes after the whole prefix, and a single query comes after every key. Around that position
//! [`AttentionOptions`] says how far a query sees, as flash-attn's `causal` and `window_size`:
//!
//! - `causal`: no key after the query's position;
//! - `window`, [`Window`] `{ left, right }`: no key more than `left` positions before it or
//!   `right` after it (with `causal`, none after it whatever `right` says).
//!
//! Beside that, the [`AttentionMask`] passed as an argument, since it is data:
//!
//! - [`AttentionMask::None`]: every key, for encoders and cross-attention;
//! - [`AttentionMask::KeyPadding`]: a boolean mask `[B, Tk]`, `true` where a key is left out
//!   (padding), as Burn's masks are;
//! - [`AttentionMask::Packed`]: variable-length sequences packed end to end along the time,
//!   `B = 1`, their boundaries the cumulative offsets `cu_seqlens_q` and `cu_seqlens_k`
//!   (`[S + 1]`, Int) of flash-attn's variable-length API. A query sees only the keys of its own
//!   sequence, and causality and the window apply within each sequence, aligned at its bottom
//!   right. Offsets start at 0 and do not decrease; positions past the last offset belong to no
//!   sequence: those queries see no key and those keys are seen by no query.
//!
//! A query that sees no key at all (every key padded, a causal query before the first key when
//! `Tk < Tq`, an empty window) returns zeros and a log-sum-exp of `-inf`, and takes no part in
//! any gradient: its own query gradient is zero. flash-attn does the same, and so does PyTorch
//! 2.13's `scaled_dot_product_attention` on the CPU (its math backend and its kernel alike).
//!
//! # Decoding
//!
//! `attention_decode` takes one query per sequence, `[B, Hq, D]`, against the cache `[B, S, Hkv,
//! D]` up to and including that token's own key and value (it may be a slice of a longer cache).
//! Per-sequence cache lengths, `[B]` (Int), say how many keys of the cache each sequence holds,
//! the first `lengths[b]` of it, the last of them the query's own; without them every sequence
//! holds all `S`. Each sequence's query sits after its last key; of the options, the scale and
//! the window's `left` apply (no key comes after the query). It returns `[B, Hq, D]` in the
//! queries' dtype, zeros for a sequence holding no key. Decoding takes no gradient.

/// A sliding window around each query's position (after the bottom-right alignment): the query
/// sees no key more than `left` positions before it or `right` after it, as flash-attn's
/// `window_size = (left, right)`. `usize::MAX` leaves that side unbounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Window {
    /// How many keys before the query's position it sees, besides the one at it.
    pub left: usize,
    /// How many keys after the query's position it sees; with `causal`, none whatever this says.
    pub right: usize,
}

/// The arguments of attention that have a usual default.
///
/// The default is attention over every key, with no causality and no window, scaled by
/// `1 / sqrt(D)`: what PyTorch's `scaled_dot_product_attention` and flash-attn do by default,
/// and what encoders and cross-attention use; and a backward pass that gives the same bits on
/// every run. [`AttentionOptions::causal`] gives the decoder's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionOptions {
    /// Whether a query sees no key after its own position. Default: `false`.
    pub causal: bool,
    /// The factor on the scores; `None` (the default) for `1 / sqrt(D)`.
    pub scale: Option<f32>,
    /// A sliding window, or `None` (the default) for none.
    pub window: Option<Window>,
    /// Whether the backward pass gives the same bits on every run. Default: `true`: the query
    /// gradient is summed over the key blocks in a fixed order. `false` lets an implementation
    /// add it with f32 atomics in whatever order its parallel work runs, as flash-attn does
    /// unless asked for `deterministic`, for less backward time, and the query gradient's last
    /// bits vary from run to run. The output and the key and value gradients are the same either
    /// way, and a path with one way of summing it (a reference) has that way whatever this says.
    pub deterministic: bool,
}

impl Default for AttentionOptions {
    fn default() -> Self {
        Self {
            causal: false,
            scale: None,
            window: None,
            deterministic: true,
        }
    }
}

impl AttentionOptions {
    /// Causal attention, otherwise the defaults.
    pub fn causal() -> Self {
        Self {
            causal: true,
            ..Self::default()
        }
    }

    /// The options with causality on or off.
    pub fn with_causal(self, causal: bool) -> Self {
        Self { causal, ..self }
    }

    /// The options with the factor on the scores.
    pub fn with_scale(self, scale: f32) -> Self {
        Self {
            scale: Some(scale),
            ..self
        }
    }

    /// The options with a sliding window of `left` keys before each query's position and
    /// `right` after it.
    pub fn with_window(self, left: usize, right: usize) -> Self {
        Self {
            window: Some(Window { left, right }),
            ..self
        }
    }

    /// The options with the backward pass deterministic or not
    /// ([`AttentionOptions::deterministic`]).
    pub fn with_deterministic(self, deterministic: bool) -> Self {
        Self {
            deterministic,
            ..self
        }
    }

    /// The factor on the scores for a head dimension of `d`.
    pub fn scale_for(&self, d: usize) -> f32 {
        self.scale.unwrap_or((1.0 / (d as f64).sqrt()) as f32)
    }
}

/// Which keys each query may see, beside the causality and the window of [`AttentionOptions`];
/// the [module documentation](self) gives the definitions.
///
/// The tensors are an implementation's own: `Mask` its boolean tensor `[B, Tk]`, `Offsets` its
/// int tensor `[S + 1]`. dash-ops names the mask for Burn's.
#[derive(Clone, Debug, Default)]
pub enum AttentionMask<Mask, Offsets> {
    /// Every key.
    #[default]
    None,
    /// `[B, Tk]`, `true` where a key is left out.
    KeyPadding(Mask),
    /// Packed variable-length sequences, `B = 1`: the cumulative offsets of the queries and of
    /// the keys, `[S + 1]` each, sequence `s` holding queries `cu_seqlens_q[s]..cu_seqlens_q[s + 1]`
    /// and keys `cu_seqlens_k[s]..cu_seqlens_k[s + 1]`.
    Packed {
        /// The queries' offsets, `[S + 1]`.
        cu_seqlens_q: Offsets,
        /// The keys' offsets, `[S + 1]`.
        cu_seqlens_k: Offsets,
    },
}
