//! The recurrent gated delta rule: one token at a time, for decoding.
//!
//! [`gated_delta_step`] advances the recurrent state of a Gated DeltaNet layer's sequences by a
//! token each, as one iteration of transformers' `torch_recurrent_gated_delta_rule` (and fla's
//! `fused_recurrent_gated_delta_rule`) does; or, over packed sequences, by each sequence's tokens
//! in turn. For each sequence and head, with `S` the `[Dk, Dv]` state and a token's `q`, `k`, `v`,
//! `g`, `beta`:
//!
//! ```text
//! q, k  <- q / sqrt(sum q^2 + 1e-6), k / sqrt(sum k^2 + 1e-6)   (with the l2 normalisation only)
//! q     <- scale q                                               (scale 1 / sqrt(Dk) by default)
//! S     <- exp(g) S
//! delta <- beta (v - S^T k)
//! S     <- S + k delta^T
//! out   <- S^T q
//! ```
//!
//! Shapes follow [`crate::gated_delta`]: `q`, `k` `[B, H, Dk]`, `v` `[B, H, Dv]`, `g` and
//! `beta` `[B, H]`, the state `[B, H, Dk, Dv]` in f32: batch row `b` is one token of sequence
//! `b`. With `cu_seqlens` `[S + 1]` the rows are instead `T` packed tokens of `S` sequences
//! (sequence `s` the rows `cu_seqlens[s]..cu_seqlens[s + 1]`, in order, possibly none), the state
//! `[S, H, Dk, Dv]`: a decoder serving sequences that bring different numbers of tokens to one
//! step (speculative tokens to verify, a prompt's tail) advances them all in one call, and one
//! token per sequence (`cu_seqlens = [0, 1, ..., S]`) is the batched step. Queries, keys and
//! values share one float dtype, which the output takes, as transformers casts its output back to
//! the queries' dtype; `g` and `beta` may each be in that dtype or in f32 (models compute `g` in
//! f32 and `beta` in their own dtype), and are read as f32 whatever their dtype. Everything is
//! computed in f32, the l2 normalisation included: on bf16 inputs this is what fla's kernels do,
//! while transformers' torch path normalises in the input dtype, so the two differ there by the
//! rounding of the normalised `q` and `k` to bf16.
//!
//! The step is a backend extension. Flex runs the reference built from Burn's tensor operations
//! (packed sequences token by token); a CubeCL device (any runtime) runs one fused CubeCL kernel
//! per call, which reads each state once, carries it through the sequence's tokens in registers,
//! and writes the new state in place when the caller hands over the only reference to it, as a
//! decoding loop that threads the state through does. The kernel asks of the device 256 units per
//! cube and 6 KiB of shared memory, and takes heads up to 256 wide; elsewhere the reference runs
//! (`kernel_takes` tells which). It runs under CUDA, and under wgpu on OpenGL over D3D12 and on
//! lavapipe (Vulkan on the CPU); CubeCL's CPU runtime, with as many units per cube as cores, runs
//! the reference. The step is forward only: decoding takes no gradient. On an autodiff device a
//! step whose inputs are not tracked runs the inner backend's step and returns untracked tensors;
//! one with a tracked input runs the reference on the autodiff backend, so its gradients are those
//! of the reference, slow but right.

use super::{Sequences, query_scale, read_cu_seqlens};
#[cfg(feature = "flex")]
use burn::backend::Flex;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::tensor::FloatTensor;
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Dispatch, TensorMetadata, backend_extension,
};
use burn::tensor::{DType, Device, FloatDType, Int, Shape, Slice, Tensor};

#[cfg(feature = "cubecl")]
mod kernel;

/// The epsilon of transformers' (and fla's) l2 normalisation of `q` and `k`.
const L2NORM_EPS: f32 = 1e-6;

/// The options of [`gated_delta_step`] that have a usual default; the defaults are fla's and
/// transformers': no normalisation, the scale `1 / sqrt(Dk)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GatedDeltaStepOptions {
    /// Normalise queries and keys to unit length first, `x / sqrt(sum(x^2) + 1e-6)`
    /// (transformers' and fla's `use_qk_l2norm_in_kernel`).
    pub qk_l2norm: bool,
    /// The queries' scale, applied after the normalisation; `None` for `1 / sqrt(Dk)` (fla's
    /// `scale=None`).
    pub scale: Option<f32>,
}

impl GatedDeltaStepOptions {
    /// The options with the l2 normalisation of queries and keys on or off.
    pub fn with_qk_l2norm(self, qk_l2norm: bool) -> Self {
        Self { qk_l2norm, ..self }
    }

    /// The options with the query scale `scale`.
    pub fn with_scale(self, scale: f32) -> Self {
        Self {
            scale: Some(scale),
            ..self
        }
    }
}

impl From<super::chunk::ChunkGatedDeltaOptions> for GatedDeltaStepOptions {
    /// The step that continues a chunked pass with these options.
    fn from(options: super::chunk::ChunkGatedDeltaOptions) -> Self {
        Self {
            qk_l2norm: options.qk_l2norm,
            scale: options.scale,
        }
    }
}

/// Counters tests read to tell which path ran. Not part of the API. The step has no written-out
/// backward pass to count: a tracked step runs the reference on the autodiff backend.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static KERNEL_LAUNCHES: AtomicUsize = AtomicUsize::new(0);

    /// Launches of the step's kernel, one a call.
    pub fn kernel_launches() -> usize {
        KERNEL_LAUNCHES.load(Ordering::SeqCst)
    }
}

#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(feature = "cubecl"),
    Fusion: cfg(feature = "fusion"),
    Autodiff
)]
pub trait GatedDeltaStepBackend: Backend {
    /// A step of the recurrent gated delta rule; see the [module documentation](self), with the
    /// query scale resolved. Returns the output `[B, H, Dv]` (packed, `[T, H, Dv]`) in `q`'s
    /// dtype and the new states `[B, H, Dk, Dv]` (packed, `[S, H, Dk, Dv]`), f32. The default
    /// body is the reference, built from Burn's tensor operations.
    #[fusion(meta = |q, _k, v, _g, _beta, state, _sequences, _use_qk_l2norm, _scale| (
        burn::backend::fusion::custom::TensorSpec::new(v.shape.clone(), q.dtype),
        state.clone(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_step(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        state: FloatTensor<Self>,
        #[extension_type] sequences: Sequences<Self>,
        use_qk_l2norm: bool,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        reference::<Self>(q, k, v, g, beta, state, sequences, use_qk_l2norm, scale)
    }
}

#[cfg(feature = "flex")]
impl GatedDeltaStepBackend for Flex {}

/// Decoding takes no gradient, so an untracked step runs the inner backend's step (the kernel on
/// a CubeCL device) and records nothing. A step with a tracked input falls back to the reference
/// on the autodiff backend, which records Burn's own operations: no written-out backward pass.
impl<B: Backend + GatedDeltaStepBackend, C: CheckpointStrategy> GatedDeltaStepBackend
    for Autodiff<B, C>
{
    fn gated_delta_step(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        state: FloatTensor<Self>,
        sequences: Sequences<Self>,
        use_qk_l2norm: bool,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let tracked = [&q, &k, &v, &g, &beta, &state]
            .iter()
            .any(|tensor| tensor.is_tracked());
        if tracked {
            return reference::<Self>(q, k, v, g, beta, state, sequences, use_qk_l2norm, scale);
        }
        let inner = <Self as AutodiffBackend>::inner;
        let sequences = match sequences {
            Sequences::Batch => Sequences::Batch,
            Sequences::Packed(cu_seqlens) => Sequences::Packed(cu_seqlens),
        };
        let (out, state) = B::gated_delta_step(
            inner(q),
            inner(k),
            inner(v),
            inner(g),
            inner(beta),
            inner(state),
            sequences,
            use_qk_l2norm,
            scale,
        );
        (
            <Self as AutodiffBackend>::from_inner(out),
            <Self as AutodiffBackend>::from_inner(state),
        )
    }
}

/// A step of the recurrent gated delta rule.
///
/// - `q`, `k`: `[B, H, Dk]`; `v`: `[B, H, Dv]`, all three of one float dtype;
/// - `g`: `[B, H]`, the log of the decay (`<= 0`); `beta`: `[B, H]`; each in `q`'s dtype or f32
///   (read as f32 whatever their dtype);
/// - `state`: `[B, H, Dk, Dv]`, f32: zeros before the first token, or the final state of a
///   chunked pass over the prompt;
/// - `cu_seqlens`: `None` for one token per sequence; else `[S + 1]`, the boundaries of `S`
///   sequences whose tokens are the `B = T` rows in turn (from 0 to `T` without decreasing; on a
///   device running the kernel they are not read back to check them), `state` then
///   `[S, H, Dk, Dv]`;
/// - `options`: the l2 normalisation of `q` and `k`, the query scale (`1 / sqrt(Dk)` by default).
///
/// Returns the output `[B, H, Dv]` in `q`'s dtype and the new states, shaped as `state`, in f32.
/// Where the kernel runs, the new states are written over the old ones when `state` is the only
/// reference to them; pass it by value from one step to the next to get that.
///
/// # Panics
///
/// If the shapes do not fit together as above, if `q`, `k` and `v` differ in dtype, or if the
/// state is not f32; where the reference runs, if `cu_seqlens` does not go from 0 to `T` without
/// decreasing.
#[allow(
    clippy::too_many_arguments,
    reason = "the five inputs, the state, the boundaries that may be absent, the options"
)]
pub fn gated_delta_step(
    q: Tensor<3>,
    k: Tensor<3>,
    v: Tensor<3>,
    g: Tensor<2>,
    beta: Tensor<2>,
    state: Tensor<4>,
    cu_seqlens: Option<Tensor<1, Int>>,
    options: GatedDeltaStepOptions,
) -> (Tensor<3>, Tensor<4>) {
    let dims = check_inputs(
        [
            &q.shape(),
            &k.shape(),
            &v.shape(),
            &g.shape(),
            &beta.shape(),
            &state.shape(),
        ],
        [q.dtype(), k.dtype(), v.dtype(), state.dtype()],
        cu_seqlens.as_ref().map(|cu| cu.dims()[0]),
    );
    let (out, state) = <Dispatch as GatedDeltaStepBackend>::gated_delta_step(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        g.into_dispatch(),
        beta.into_dispatch(),
        state.into_dispatch(),
        Sequences::from(cu_seqlens.map(Tensor::into_dispatch)),
        options.qk_l2norm,
        query_scale(options.scale, dims[2]),
    );
    (Tensor::from_dispatch(out), Tensor::from_dispatch(state))
}

/// [`gated_delta_step`] through Burn's tensor operations alone, on any device: the reference
/// that Flex runs and the kernel is held to. For tests and benchmarks.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_step_reference(
    q: Tensor<3>,
    k: Tensor<3>,
    v: Tensor<3>,
    g: Tensor<2>,
    beta: Tensor<2>,
    state: Tensor<4>,
    cu_seqlens: Option<Tensor<1, Int>>,
    options: GatedDeltaStepOptions,
) -> (Tensor<3>, Tensor<4>) {
    let dims = check_inputs(
        [
            &q.shape(),
            &k.shape(),
            &v.shape(),
            &g.shape(),
            &beta.shape(),
            &state.shape(),
        ],
        [q.dtype(), k.dtype(), v.dtype(), state.dtype()],
        cu_seqlens.as_ref().map(|cu| cu.dims()[0]),
    );
    let (out, state) = reference::<Dispatch>(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        g.into_dispatch(),
        beta.into_dispatch(),
        state.into_dispatch(),
        Sequences::from(cu_seqlens.map(Tensor::into_dispatch)),
        options.qk_l2norm,
        query_scale(options.scale, dims[2]),
    );
    (Tensor::from_dispatch(out), Tensor::from_dispatch(state))
}

/// Whether the kernel, rather than the reference, takes a step over inputs of shape
/// `[B, H, Dk, Dv]` (packed: `B` the tokens) with `q`, `k` and `v` in `dtypes[0]`, `g` in
/// `dtypes[1]` and `beta` in `dtypes[2]` on `device`; `packed` gives the number of packed
/// sequences and the dtype of their boundaries. Heads up to 256 wide, on a device with what the
/// kernel asks (module documentation). `false` on any device not CubeCL's. For tests and
/// benchmarks.
#[doc(hidden)]
pub fn kernel_takes(
    device: &Device,
    dtypes: [DType; 3],
    dims: [usize; 4],
    packed: Option<(usize, DType)>,
) -> bool {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return kernel::kernel_takes(&client, dtypes, dims, packed);
    }
    let _ = (device, dtypes, dims, packed);
    false
}

/// Checks that the step's inputs fit together and returns `[B, H, Dk, Dv]` (`B` the rows).
/// `shapes` are those of `q, k, v, g, beta, state`; `dtypes` those of `q, k, v, state`;
/// `boundaries` the length of `cu_seqlens` when packed.
fn check_inputs(shapes: [&Shape; 6], dtypes: [DType; 4], boundaries: Option<usize>) -> [usize; 4] {
    let [q, k, v, g, beta, state] = shapes;
    assert!(
        q.num_dims() == 3 && k.num_dims() == 3 && v.num_dims() == 3,
        "gated_delta_step: q, k and v are [B, H, D], got {q:?}, {k:?}, {v:?}"
    );
    let (b, h, dk, dv) = (q[0], q[1], q[2], v[2]);
    assert_eq!(
        k.as_slice(),
        &[b, h, dk],
        "gated_delta_step: k must match q [B, H, Dk]"
    );
    assert_eq!(
        v.as_slice()[..2],
        [b, h],
        "gated_delta_step: v must be [B, H, Dv] with q's B, H"
    );
    assert_eq!(g.as_slice(), &[b, h], "gated_delta_step: g must be [B, H]");
    assert_eq!(
        beta.as_slice(),
        &[b, h],
        "gated_delta_step: beta must be [B, H]"
    );
    let sequences = match boundaries {
        None => b,
        Some(n) => {
            assert!(
                n >= 2,
                "gated_delta_step: cu_seqlens must hold S + 1 >= 2 offsets, got {n}"
            );
            n - 1
        }
    };
    assert_eq!(
        state.as_slice(),
        &[sequences, h, dk, dv],
        "gated_delta_step: state must be [S, H, Dk, Dv], one per sequence"
    );
    let [q, k, v, state] = dtypes;
    assert!(
        q.is_float() && k == q && v == q,
        "gated_delta_step: q, k and v must share one float dtype, got {q:?}, {k:?}, {v:?}"
    );
    assert_eq!(state, DType::F32, "gated_delta_step: the state is f32");
    [b, h, dk, dv]
}

/// The reference: the operations of `torch_recurrent_gated_delta_rule` for one token per
/// sequence, in f32, from Burn's tensor operations; packed sequences token by token. Every
/// backend can run it; it is what the kernel is held to.
#[allow(clippy::too_many_arguments)]
fn reference<B: Backend>(
    q: FloatTensor<B>,
    k: FloatTensor<B>,
    v: FloatTensor<B>,
    g: FloatTensor<B>,
    beta: FloatTensor<B>,
    state: FloatTensor<B>,
    sequences: Sequences<B>,
    use_qk_l2norm: bool,
    scale: f32,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let Sequences::Packed(cu_seqlens) = sequences else {
        return step::<B>(q, k, v, g, beta, state, use_qk_l2norm, scale);
    };
    let [tokens, h, _] = q.shape().dims();
    let dv = v.shape()[2];
    let cu = read_cu_seqlens::<B>(cu_seqlens, state.shape()[0], tokens);
    let row = |x: &FloatTensor<B>, t: usize| {
        let mut slices = vec![Slice::from(..); x.shape().num_dims()];
        slices[0] = Slice::from(t..t + 1);
        B::float_slice(x.clone(), &slices)
    };
    let mut outs = Vec::with_capacity(tokens);
    let mut states = Vec::with_capacity(cu.len() - 1);
    for (s, span) in cu.windows(2).enumerate() {
        let mut st = row(&state, s);
        for t in span[0]..span[1] {
            let (out, next) = step::<B>(
                row(&q, t),
                row(&k, t),
                row(&v, t),
                row(&g, t),
                row(&beta, t),
                st,
                use_qk_l2norm,
                scale,
            );
            outs.push(out);
            st = next;
        }
        states.push(st);
    }
    let out = if outs.is_empty() {
        B::float_zeros(Shape::new([0, h, dv]), &q.device(), q.dtype().into())
    } else {
        B::float_cat(outs, 0)
    };
    (out, B::float_cat(states, 0))
}

/// One token of each of `B` sequences: `torch_recurrent_gated_delta_rule`'s operations for one
/// iteration, in f32.
#[allow(clippy::too_many_arguments)]
fn step<B: Backend>(
    q: FloatTensor<B>,
    k: FloatTensor<B>,
    v: FloatTensor<B>,
    g: FloatTensor<B>,
    beta: FloatTensor<B>,
    state: FloatTensor<B>,
    use_qk_l2norm: bool,
    scale: f32,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let out_dtype = FloatDType::from(q.dtype());
    let [b, h, dk] = q.shape().dims();
    let dv = v.shape()[2];
    let f32 = |tensor| B::float_cast(tensor, FloatDType::F32);
    let (mut q, mut k) = (f32(q), f32(k));
    if use_qk_l2norm {
        q = l2norm::<B>(q);
        k = l2norm::<B>(k);
    }
    let q = B::float_mul_scalar(q, scale.into());

    let decay = B::float_reshape(B::float_exp(f32(g)), Shape::new([b, h, 1, 1]));
    let state = B::float_mul(f32(state), decay);
    let k = B::float_reshape(k, Shape::new([b, h, dk, 1]));
    // S^T k, as transformers writes it: (S * k[..., None]).sum(-2).
    let kv_mem = B::float_sum_dim(B::float_mul(state.clone(), k.clone()), 2);
    let v = B::float_reshape(f32(v), Shape::new([b, h, 1, dv]));
    let beta = B::float_reshape(f32(beta), Shape::new([b, h, 1, 1]));
    let delta = B::float_mul(B::float_sub(v, kv_mem), beta);
    let state = B::float_add(state, B::float_mul(k, delta));
    let q = B::float_reshape(q, Shape::new([b, h, dk, 1]));
    let out = B::float_sum_dim(B::float_mul(state.clone(), q), 2);
    let out = B::float_cast(B::float_reshape(out, Shape::new([b, h, dv])), out_dtype);
    (out, state)
}

/// `x * (1 / sqrt(sum(x^2) + eps))` over the last of three dimensions, in transformers' order of
/// operations (its `rsqrt` is `1 / sqrt` on the CPU).
fn l2norm<B: Backend>(x: FloatTensor<B>) -> FloatTensor<B> {
    let sum = B::float_sum_dim(B::float_mul(x.clone(), x.clone()), 2);
    let norm = B::float_sqrt(B::float_add_scalar(sum, L2NORM_EPS.into()));
    B::float_mul(x, B::float_recip(norm))
}
