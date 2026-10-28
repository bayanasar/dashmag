//! Causal depthwise 1-D convolution over sequences, of any small kernel width, with an optional
//! bias and activation; from a given state or from zeros, over a batch or packed variable-length
//! sequences; with its gradients, and the step a decoder takes: `causal_conv1d` over sequences,
//! `causal_conv1d_update` one decoding step.
//!
//! Per channel `c` of a sequence of `L` steps, with `K` the kernel width:
//!
//! ```text
//! y[t, c]   = bias[c] + sum_{j < K} weight[c, j] * h[t - (K - 1) + j, c]
//! out[t, c] = act(y[t, c])
//! ```
//!
//! `h` is the sequence's history: its inputs `x[0..L]`, preceded by its initial state, the `W`
//! inputs before its first step (oldest first), and by zeros before those; with no initial state,
//! by zeros alone. `weight[c, K - 1]` multiplies the current step, `weight[c, 0]` the step `K - 1`
//! before it. Without a bias, `bias[c]` is zero; `act` is the identity or SiLU ([`Activation`]).
//! The sequence's final state is the last `W'` steps of its history, `h[L - W'..L]`, zeros where
//! the history is shorter: what the next call continues from. A sequence run in pieces, each piece
//! from the final state the previous one left, has the outputs of the whole sequence, and through
//! autodiff the gradients of the whole, the final state's gradient passed back to the piece
//! before. Gradients flow to `x`, the weight, the bias and the initial states, from the output
//! and from the final states.
//!
//! This is torch's `F.conv1d(torch.cat([state, x]), weight[:, None, :], bias, groups=C)` on `x`
//! in `[C, L]` (`F.pad(x, (K - 1, 0))` without a state), its last `L` outputs, then the
//! activation: what the causal-conv1d package's `causal_conv1d_fn` computes with
//! `initial_states` and `return_final_states`, and over packed sequences with `seq_idx` (there
//! without states), and transformers' fallback for it.
//!
//! # Sequences, layouts and dtypes
//!
//! - `x`, the output and their gradients: `[B, T, C]`, every batch element one sequence. Inputs
//!   may have any strides; outputs are contiguous.
//! - Packed variable-length sequences: `B = 1` and `cu_seqlens` `[N + 1]` (Int), as the
//!   [gated delta rule](crate::gated_delta) takes them: sequence `s` is steps
//!   `cu_seqlens[s]..cu_seqlens[s + 1]`, the offsets going from 0 to `T` without decreasing. No
//!   step of one sequence reads another's, forward or backward: packed, each sequence has what a
//!   call on it alone gives. A sequence may be empty: it has no outputs, its final state is the
//!   last `W'` steps of its initial state (zeros in front where `W' > W`), and its initial
//!   state's gradient is what the final state's gradient gives it.
//! - `weight`: `[C, K]`, torch's `[C, 1, K]` without its middle axis. `bias`: `[C]`, optional.
//! - States, initial and final, and their gradients: `[S, W, C]`, f32, one per sequence (`S = B`,
//!   or `N` packed): the last inputs before a sequence's first step, oldest first. An initial
//!   state holds `W >= K - 1` steps; a final state `W'` steps, those of the initial state by
//!   default ([`CausalConv1dOptions::final_state_len`]), `K - 1` without one.
//! - One decoding step's input and output: `[B, C]`, its state `[B, W, C]`.
//!
//! `x`, `weight` and `bias` share one float dtype (f32, bf16 or f16). Arithmetic is in f32
//! whatever it is, and the output and each gradient are rounded once, to the dtype of the input
//! they belong to; the weight and bias gradients are reduced over every step of every sequence
//! in f32 too.
//!
//! States are f32, as every state an operation carries from call to call: they hold inputs, which
//! f32 holds exactly in any of the three dtypes, so that a sequence continued from a state reads
//! the values the whole sequence would, and models can keep every state of every layer in one
//! dtype. A state costs `W x C x 4` bytes per sequence (72 KiB at `W = 3`, `C = 6144`).
//! transformers' cache keeps it as `[B, C, W]` in the model's dtype: `state.transpose(1, 2).float()`
//! converts it.
//!
//! # Decoding
//!
//! A prompt's final state (`causal_conv1d` with [`CausalConv1dOptions::with_final_state`])
//! starts decoding; `causal_conv1d_update` takes one token per sequence and the state to the
//! token's output and the next state: the output is the last step of `causal_conv1d` over the
//! token from the state, the next state its final state, the state without its oldest step
//! followed by the token. A prompt's outputs, then each step's, are the outputs of
//! `causal_conv1d` over the whole sequence. Sequences that bring several tokens to one step (a
//! tool's reply appended to a prompt) take `causal_conv1d` from their states, packed if their
//! lengths differ.

/// The activation applied to the convolution's output (after the bias).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Activation {
    /// None: the convolution's output as it is.
    #[default]
    None,
    /// SiLU, also called swish: `y * sigmoid(y)`.
    Silu,
}

/// The arguments of `causal_conv1d` that have a usual default: no activation, no final state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CausalConv1dOptions {
    /// The activation after the convolution and the bias. Default: none.
    pub activation: Activation,
    /// Return each sequence's final state, the last `W'` steps of its history (causal-conv1d's
    /// `return_final_states`, fla's `output_final_state`). Default: no.
    pub output_final_state: bool,
    /// `W'`, the final states' length: `None` for that of the initial states, or `K - 1` without
    /// them, the least a kernel of width `K` continues from. Default: `None`.
    pub final_state_len: Option<usize>,
}

impl CausalConv1dOptions {
    /// The options with `activation`.
    pub fn with_activation(self, activation: Activation) -> Self {
        Self { activation, ..self }
    }

    /// The options returning the final states, or not.
    pub fn with_final_state(self, output_final_state: bool) -> Self {
        Self {
            output_final_state,
            ..self
        }
    }

    /// The options returning final states `len` steps long.
    pub fn with_final_state_len(self, len: usize) -> Self {
        Self {
            output_final_state: true,
            final_state_len: Some(len),
            ..self
        }
    }
}

/// The arguments of `causal_conv1d_update` that have a usual default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CausalConv1dStepOptions {
    /// The activation after the convolution and the bias. Default: none.
    pub activation: Activation,
}

impl CausalConv1dStepOptions {
    /// The options with `activation`.
    pub fn with_activation(self, activation: Activation) -> Self {
        Self { activation }
    }
}

impl From<CausalConv1dOptions> for CausalConv1dStepOptions {
    /// The step that continues a call with these options.
    fn from(options: CausalConv1dOptions) -> Self {
        Self {
            activation: options.activation,
        }
    }
}
