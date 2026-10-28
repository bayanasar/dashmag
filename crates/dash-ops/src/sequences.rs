//! Packed variable-length sequences, as the gated delta rule and causal conv1d take them
//! ([`Sequences`]): a batch of one whose tokens lie end to end along `T`, the boundaries the
//! cumulative offsets `cu_seqlens` `[S + 1]` (Int, any int dtype) of `S` sequences, from 0 to `T`
//! without decreasing (a sequence may be empty), as flash-attn and fla take them. Attention,
//! whose queries and keys have boundaries of their own and may end before `T`, takes them as its
//! mask.

use burn::backend::tensor::IntTensor;
use burn::backend::{Backend, ExtensionType};

/// The tokens of a call as sequences, as the backend extensions take them (an `Option` of a
/// tensor does not cross Burn's dispatch boundary; this enum does).
#[derive(ExtensionType, Clone, Debug)]
#[extension_type(fusion: cfg(feature = "fusion"))]
pub enum Sequences<B: Backend> {
    /// Every batch element is one sequence (a step: every row).
    Batch,
    /// Packed sequences: the boundaries `cu_seqlens` `[S + 1]`, any int dtype, over the tokens
    /// of a batch of one (a step: over its rows).
    Packed(IntTensor<B>),
}

impl<B: Backend> Sequences<B> {
    /// The boundaries, if packed.
    pub fn into_option(self) -> Option<IntTensor<B>> {
        match self {
            Sequences::Batch => None,
            Sequences::Packed(cu_seqlens) => Some(cu_seqlens),
        }
    }
}

impl<B: Backend> From<Option<IntTensor<B>>> for Sequences<B> {
    fn from(cu_seqlens: Option<IntTensor<B>>) -> Self {
        match cu_seqlens {
            None => Sequences::Batch,
            Some(cu_seqlens) => Sequences::Packed(cu_seqlens),
        }
    }
}

/// The boundaries of `sequences` packed sequences over `tokens` tokens, read to the host, as
/// the references take them.
///
/// # Panics
///
/// If there are not `sequences + 1` of them, or they do not go from 0 to `tokens` without
/// decreasing.
pub(crate) fn read_cu_seqlens<B: Backend>(
    cu_seqlens: IntTensor<B>,
    sequences: usize,
    tokens: usize,
) -> Vec<usize> {
    let data = burn::tensor::try_read_sync(B::int_into_data(cu_seqlens))
        .expect("cu_seqlens: a synchronous read")
        .expect("cu_seqlens: read to the host");
    let cu: Vec<i64> = data.iter::<i64>().collect();
    assert_eq!(
        cu.len(),
        sequences + 1,
        "cu_seqlens must hold S + 1 = {} offsets for the {sequences} states, got {}",
        sequences + 1,
        cu.len()
    );
    assert!(
        cu[0] == 0 && cu.windows(2).all(|w| w[0] <= w[1]) && cu[sequences] == tokens as i64,
        "cu_seqlens must go from 0 to T = {tokens} without decreasing, got {cu:?}"
    );
    cu.into_iter().map(|x| x as usize).collect()
}
