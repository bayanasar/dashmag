//! The chunked gated delta rule, forward and backward, for training.
//!
//! [`chunk_gated_delta_rule`] computes what transformers' `torch_chunk_gated_delta_rule` does
//! (and fla's `chunk_gated_delta_rule`, which transformers calls when fla is installed): for each
//! batch element and head, the recurrence
//!
//! ```text
//! S_t = exp(g_t) S_{t-1} + beta_t k_t (v_t - exp(g_t) S_{t-1}^T k_t)^T
//! o_t = S_t^T q_t / sqrt(Dk)
//! ```
//!
//! computed in chunks of 64 tokens: within a chunk through the WY representation of the
//! products of Householder-like updates, and the state carried from chunk to chunk. The state
//! is `[Dk, Dv]` per head and always f32.
//!
//! - Inputs: queries and keys `[B, T, H, Dk]`, values `[B, T, H, Dv]` in one float dtype (f32,
//!   bf16 or f16); `g`, the log of the decay (`<= 0`), and `beta`, both `[B, T, H]`, each in that
//!   dtype or in f32 (models compute `g` in f32); both are read as f32, so another float dtype
//!   works too; an optional initial state `[B, H, Dk, Dv]`, f32.
//! - Outputs: `o` `[B, T, H, Dv]` in the queries' dtype, and when asked the final state
//!   `[B, H, Dk, Dv]` in f32, from which [`super::recurrent`] continues decoding.
//! - Options: `use_qk_l2norm_in_kernel` normalises queries and keys to unit length first (with
//!   transformers' epsilon, `1e-6` under the square root). The query scale is `1/sqrt(Dk)`; the
//!   chunk size is 64.
//! - Everything inside is computed in f32, as transformers does, whatever the inputs' dtype; the
//!   outputs and gradients are rounded once, to their dtype. (transformers normalises bf16 queries
//!   and keys in bf16 before casting; here they are normalised in f32.)
//! - Packed variable-length sequences (fla's `cu_seqlens`) are not supported: every batch element
//!   is one sequence of length T. Shorter sequences padded at the end are exact for the tokens
//!   before the padding; the final state is then not the state after the last real token.
//!
//! Gradients flow to the queries, keys, values, `g`, `beta` and the initial state, and from both
//! the output and the final state. The backward pass is written out: it keeps only the inputs
//! from the forward pass, recomputes each chunk's quantities, passes the state's gradient back
//! from chunk to chunk, and forms the input gradients in each chunk.
//!
//! Every backend runs the reference built from Burn's tensor operations (`chunk/reference.rs`).
//! On a CubeCL device (any runtime), heads up to 128 wide (`Dk, Dv <= 128`) run CubeCL kernels
//! instead (`chunk/cube.rs`), all in f32 without tensor cores: per chunk, one kernel forms
//! everything that does not depend on the state, one passes the state from chunk to chunk, one
//! forms the output; the backward pass runs the first two again, passes the state's gradient
//! back, and forms the gradients in five more per-chunk kernels. The kernels ask of the device
//! 256 units per cube, up to 90 KiB of shared memory forward and 98 KiB backward, and backward
//! planes of exactly 32 units: CUDA cards of compute capability 8.0 and later have all of it. On
//! a device without it, and for wider heads, the reference runs instead (`kernels_take` tells
//! which): under wgpu on OpenGL over D3D12 and on lavapipe (Vulkan on the CPU), both with 32 KiB
//! of shared memory and no plane operations, and on CubeCL's CPU runtime (as many units per cube
//! as cores), for instance.
//!
//! Measured against transformers' function at f32 (the tests' fixtures), the output, the final
//! state and every gradient agree to within 1.12e-6 relative to each tensor's largest value; from
//! bf16 inputs, to within 2.41e-3 of the exact result (torch's own bf16 path: 8.06e-3), and the
//! CUDA kernels to within 3.17e-3 of the f32 reference at larger shapes. On an
//! RTX 5070 Ti, one sequence of 6144 tokens with 16 heads of 128 and bf16 inputs takes 2.4 ms
//! forward and 9.3 ms forward and backward, and grows the memory pool by 687 MiB
//! (`examples/gated_delta_chunk_bench.rs`); fla 0.5.2, with bf16 tensor cores, takes 0.87 ms and
//! 3.6 ms and 422 MiB.

mod reference;

#[cfg(feature = "cubecl")]
mod cube;

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::FloatTensor;
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Dispatch, TensorMetadata, backend_extension,
};
use burn::tensor::{DType, Device, FloatDType, Shape, Slice, Tensor, TensorCreationOptions};
use std::sync::atomic::Ordering;

pub use reference::CHUNK as CHUNK_SIZE;

/// Counters tests read to tell which path ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static KERNEL_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);

    /// Launches of this module's CubeCL kernels: three for a forward pass, nine for a backward
    /// pass.
    pub fn kernel_launches() -> usize {
        KERNEL_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Runs of the written-out backward pass on `Autodiff`.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }
}

/// Whether the CubeCL kernels take a forward and a backward pass over inputs of `dtype` and
/// shape `[B, T, H, Dk, Dv]` on `device`, rather than the reference: heads up to 128 wide, on a
/// device with what the kernels ask (module documentation). `[false, false]` on any device not
/// CubeCL's. For tests, as the [`probe`] counters are.
#[doc(hidden)]
pub fn kernels_take(device: &Device, dtype: DType, dims: [usize; 5]) -> [bool; 2] {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return [false, true].map(|backward| cube::kernels_take(&client, dtype, dims, backward));
    }
    let _ = (device, dtype, dims);
    [false, false]
}

#[cfg(feature = "fusion")]
mod fusion_meta {
    use burn::backend::fusion::custom::TensorSpec;
    use burn::tensor::{DType, Shape};

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        q: &TensorSpec,
        _k: &TensorSpec,
        v: &TensorSpec,
        _g: &TensorSpec,
        _beta: &TensorSpec,
        initial_state: &TensorSpec,
        _use_qk_l2norm: &bool,
        _output_final_state: &bool,
    ) -> (TensorSpec, TensorSpec) {
        let [b, t, h, _] = q.shape.dims::<4>();
        let dv = v.shape.dims::<4>()[3];
        (
            TensorSpec::new(Shape::new([b, t, h, dv]), q.dtype),
            TensorSpec::new(initial_state.shape.clone(), DType::F32),
        )
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn backward(
        q: &TensorSpec,
        k: &TensorSpec,
        v: &TensorSpec,
        g: &TensorSpec,
        beta: &TensorSpec,
        initial_state: &TensorSpec,
        _d_out: &TensorSpec,
        _d_final_state: &TensorSpec,
        _use_qk_l2norm: &bool,
    ) -> (
        TensorSpec,
        TensorSpec,
        TensorSpec,
        TensorSpec,
        TensorSpec,
        TensorSpec,
    ) {
        (
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            TensorSpec::new(initial_state.shape.clone(), DType::F32),
        )
    }
}

/// The chunked gated delta rule as a backend extension. Call it through
/// [`chunk_gated_delta_rule`]; the methods take the layout described in the module docs, with
/// the initial state always present (zeros when there is none).
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(feature = "cubecl"),
    Fusion: cfg(feature = "fusion"),
    Autodiff
)]
pub trait ChunkGatedDeltaBackend: Backend {
    /// The forward pass: the output `[B, T, H, Dv]` in the queries' dtype and the final state
    /// `[B, H, Dk, Dv]` in f32. `output_final_state` only tells Autodiff whether the final state
    /// takes part in the graph; the final state is always computed.
    #[fusion(meta = fusion_meta::forward)]
    #[allow(clippy::too_many_arguments)]
    fn chunk_gated_delta_forward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        use_qk_l2norm: bool,
        output_final_state: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let _ = output_final_state;
        reference::forward::<Self>(q, k, v, g, beta, initial_state, use_qk_l2norm)
    }

    /// The backward pass: the gradients of the queries, keys, values, `g`, `beta` (each in its
    /// input's dtype) and the initial state (f32), given the gradients of the output and of the
    /// final state (zeros when the final state is not used).
    #[fusion(meta = fusion_meta::backward)]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn chunk_gated_delta_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        d_out: FloatTensor<Self>,
        d_final_state: FloatTensor<Self>,
        use_qk_l2norm: bool,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        reference::backward::<Self>(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            d_out,
            d_final_state,
            use_qk_l2norm,
        )
    }
}

#[cfg(feature = "flex")]
impl ChunkGatedDeltaBackend for burn::backend::Flex {}

/// What the backward pass keeps from the forward: the inputs, nothing computed from them.
#[derive(Clone, Debug)]
struct Saved<B: Backend> {
    q: FloatTensor<B>,
    k: FloatTensor<B>,
    v: FloatTensor<B>,
    g: FloatTensor<B>,
    beta: FloatTensor<B>,
    initial_state: FloatTensor<B>,
    use_qk_l2norm: bool,
    /// When the final state takes part in the graph, the node's output is the output (as f32)
    /// and the final state flattened into one tensor, since an autodiff node has one output; this
    /// is the output's shape, for splitting the gradient.
    packed: Option<Shape>,
}

#[derive(Debug)]
struct ChunkBackward;

impl<B: ChunkGatedDeltaBackend> Backward<B, 6> for ChunkBackward {
    type State = Saved<B>;

    fn backward(self, ops: Ops<Self::State, 6>, grads: &mut Gradients, _: &mut Checkpointer) {
        probe::BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
        let s = ops.state;
        let grad = grads.consume::<B>(&ops.node);
        let state_shape = s.initial_state.shape();
        let (d_out, d_final_state) = match s.packed {
            None => {
                let zeros = B::float_zeros(state_shape, &grad.device(), FloatDType::F32);
                (grad, zeros)
            }
            Some(out_shape) => {
                let n = out_shape.num_elements();
                let m = state_shape.num_elements();
                let d_out = B::float_reshape(
                    B::float_slice(grad.clone(), &[Slice::from(0..n)]),
                    out_shape,
                );
                let d_state =
                    B::float_reshape(B::float_slice(grad, &[Slice::from(n..n + m)]), state_shape);
                (d_out, d_state)
            }
        };
        let (dq, dk, dv, dg, d_beta, d_initial) = B::chunk_gated_delta_backward(
            s.q,
            s.k,
            s.v,
            s.g,
            s.beta,
            s.initial_state,
            d_out,
            d_final_state,
            s.use_qk_l2norm,
        );
        for (parent, grad) in ops
            .parents
            .into_iter()
            .zip([dq, dk, dv, dg, d_beta, d_initial])
        {
            if let Some(node) = parent {
                grads.register::<B>(node.id, grad);
            }
        }
    }
}

impl<B: ChunkGatedDeltaBackend, C: CheckpointStrategy> ChunkGatedDeltaBackend for Autodiff<B, C> {
    fn chunk_gated_delta_forward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        use_qk_l2norm: bool,
        output_final_state: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let nodes = [
            q.node(),
            k.node(),
            v.node(),
            g.node(),
            beta.node(),
            initial_state.node(),
        ];
        let (q, k, v) = (q.into_primitive(), k.into_primitive(), v.into_primitive());
        let (g, beta) = (g.into_primitive(), beta.into_primitive());
        let initial_state = initial_state.into_primitive();
        match ChunkBackward.prepare::<C>(nodes).compute_bound().stateful() {
            OpsKind::UnTracked(prep) => {
                let (out, state) = B::chunk_gated_delta_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state,
                    use_qk_l2norm,
                    output_final_state,
                );
                (prep.finish(out), Self::from_inner(state))
            }
            OpsKind::Tracked(prep) => {
                let mut saved = Saved {
                    q: q.clone(),
                    k: k.clone(),
                    v: v.clone(),
                    g: g.clone(),
                    beta: beta.clone(),
                    initial_state: initial_state.clone(),
                    use_qk_l2norm,
                    packed: None,
                };
                let (out, state) = B::chunk_gated_delta_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state,
                    use_qk_l2norm,
                    output_final_state,
                );
                if !output_final_state {
                    return (prep.finish(saved, out), Self::from_inner(state));
                }
                let (out_shape, out_dtype) = (out.shape(), out.dtype());
                let state_shape = state.shape();
                let (n, m) = (out_shape.num_elements(), state_shape.num_elements());
                saved.packed = Some(out_shape.clone());
                let packed = B::float_cat(
                    vec![
                        B::float_reshape(reference::to_f32::<B>(out), Shape::new([n])),
                        B::float_reshape(state, Shape::new([m])),
                    ],
                    0,
                );
                let packed = prep.finish(saved, packed);
                let out = Self::float_reshape(
                    Self::float_slice(packed.clone(), &[Slice::from(0..n)]),
                    out_shape,
                );
                let out = if out_dtype == DType::F32 {
                    out
                } else {
                    Self::float_cast(out, out_dtype.into())
                };
                let state = Self::float_reshape(
                    Self::float_slice(packed, &[Slice::from(n..n + m)]),
                    state_shape,
                );
                (out, state)
            }
        }
    }

    /// Not differentiable itself (no double backward): runs on the inner backend.
    fn chunk_gated_delta_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        d_out: FloatTensor<Self>,
        d_final_state: FloatTensor<Self>,
        use_qk_l2norm: bool,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let (dq, dk, dv, dg, d_beta, d_initial) = B::chunk_gated_delta_backward(
            Self::inner(q),
            Self::inner(k),
            Self::inner(v),
            Self::inner(g),
            Self::inner(beta),
            Self::inner(initial_state),
            Self::inner(d_out),
            Self::inner(d_final_state),
            use_qk_l2norm,
        );
        (
            Self::from_inner(dq),
            Self::from_inner(dk),
            Self::from_inner(dv),
            Self::from_inner(dg),
            Self::from_inner(d_beta),
            Self::from_inner(d_initial),
        )
    }
}

/// The chunked gated delta rule, transformers' `torch_chunk_gated_delta_rule` (chunk size 64,
/// query scale `1/sqrt(Dk)`).
///
/// - `q`, `k`: `[B, T, H, Dk]`; `v`: `[B, T, H, Dv]`, one float dtype for the three.
/// - `g`: `[B, T, H]`, the log of the decay, `<= 0`; `beta`: `[B, T, H]`; each in the queries'
///   dtype or f32 (read as f32 whatever their dtype).
/// - `initial_state`: `[B, H, Dk, Dv]`, f32 (other dtypes are cast); zeros when `None`.
/// - `output_final_state`: return the state after the last token, `[B, H, Dk, Dv]` in f32.
/// - `use_qk_l2norm_in_kernel`: normalise queries and keys to unit length first.
///
/// Returns the output `[B, T, H, Dv]` in the queries' dtype and, if asked, the final state.
/// Packed variable-length sequences (`cu_seqlens`) are not supported.
///
/// # Panics
///
/// If the shapes do not agree, or `q`, `k`, `v` differ in dtype.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors transformers' torch_chunk_gated_delta_rule, argument for argument"
)]
pub fn chunk_gated_delta_rule(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    initial_state: Option<Tensor<4>>,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
) -> (Tensor<4>, Option<Tensor<4>>) {
    let [b, t, h, dk] = q.dims();
    let dv = v.dims()[3];
    assert_eq!(
        k.dims(),
        [b, t, h, dk],
        "keys must have the queries' shape [B, T, H, Dk]"
    );
    assert_eq!(
        v.dims(),
        [b, t, h, dv],
        "values must be [B, T, H, Dv] with the queries' B, T, H"
    );
    assert_eq!(g.dims(), [b, t, h], "g must be [B, T, H]");
    assert_eq!(beta.dims(), [b, t, h], "beta must be [B, T, H]");
    assert!(
        q.dtype() == k.dtype() && q.dtype() == v.dtype(),
        "queries, keys and values must share one dtype, got {:?}, {:?}, {:?}",
        q.dtype(),
        k.dtype(),
        v.dtype()
    );
    assert!(t > 0, "the sequence must not be empty");
    let initial_state = match initial_state {
        Some(state) => {
            assert_eq!(
                state.dims(),
                [b, h, dk, dv],
                "the initial state must be [B, H, Dk, Dv]"
            );
            if state.dtype() == DType::F32 {
                state
            } else {
                state.cast(DType::F32)
            }
        }
        None => Tensor::zeros(
            [b, h, dk, dv],
            TensorCreationOptions::new(q.device()).with_dtype(DType::F32),
        ),
    };
    let (out, state) = <Dispatch as ChunkGatedDeltaBackend>::chunk_gated_delta_forward(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        g.into_dispatch(),
        beta.into_dispatch(),
        initial_state.into_dispatch(),
        use_qk_l2norm_in_kernel,
        output_final_state,
    );
    (
        Tensor::from_dispatch(out),
        output_final_state.then(|| Tensor::from_dispatch(state)),
    )
}

/// The reference built from Burn's tensor operations, on any device, without autodiff: the
/// output and the final state. For tests and benchmarks that hold the kernels to it.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule_reference(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    initial_state: Tensor<4>,
    use_qk_l2norm_in_kernel: bool,
) -> (Tensor<4>, Tensor<4>) {
    let (out, state) = reference::forward::<Dispatch>(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        g.into_dispatch(),
        beta.into_dispatch(),
        initial_state.into_dispatch(),
        use_qk_l2norm_in_kernel,
    );
    (Tensor::from_dispatch(out), Tensor::from_dispatch(state))
}

/// The reference's written-out backward pass, as [`chunk_gated_delta_rule_reference`]: the
/// gradients of the queries, keys, values, `g`, `beta` and the initial state.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule_reference_backward(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    initial_state: Tensor<4>,
    d_out: Tensor<4>,
    d_final_state: Tensor<4>,
    use_qk_l2norm_in_kernel: bool,
) -> (
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    Tensor<3>,
    Tensor<3>,
    Tensor<4>,
) {
    let (dq, dk, dv, dg, d_beta, d_initial) = reference::backward::<Dispatch>(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        g.into_dispatch(),
        beta.into_dispatch(),
        initial_state.into_dispatch(),
        d_out.into_dispatch(),
        d_final_state.into_dispatch(),
        use_qk_l2norm_in_kernel,
    );
    (
        Tensor::from_dispatch(dq),
        Tensor::from_dispatch(dk),
        Tensor::from_dispatch(dv),
        Tensor::from_dispatch(dg),
        Tensor::from_dispatch(d_beta),
        Tensor::from_dispatch(d_initial),
    )
}
