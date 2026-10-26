//! The recurrent gated delta rule: one token at a time, for decoding.
//!
//! [`gated_delta_step`] advances a Gated DeltaNet layer's recurrent state by one token, as one
//! iteration of transformers' `torch_recurrent_gated_delta_rule` does. For each batch row and
//! head, with `S` the `[Dk, Dv]` state:
//!
//! ```text
//! q, k  <- q / sqrt(sum q^2 + 1e-6), k / sqrt(sum k^2 + 1e-6)   (with use_qk_l2norm only)
//! q     <- q / sqrt(Dk)
//! S     <- exp(g) S
//! delta <- beta (v - S^T k)
//! S     <- S + k delta^T
//! out   <- S^T q
//! ```
//!
//! Shapes follow [`crate::gated_delta`]: `q`, `k` `[B, H, Dk]`, `v` `[B, H, Dv]`, `g` and
//! `beta` `[B, H]`, the state `[B, H, Dk, Dv]` in f32. Queries, keys and values share one float
//! dtype, which the output takes, as transformers casts its output back to the queries' dtype;
//! `g` and `beta` may each be in that dtype or in f32 (models compute `g` in f32 and `beta` in
//! their own dtype), and are read as f32 whatever their dtype. Everything is computed in f32, the
//! l2 normalisation included: on bf16 inputs this is what fla's kernels do, while transformers'
//! torch path normalises in the input dtype, so the two differ there by the rounding of the
//! normalised `q` and `k` to bf16.
//!
//! The step is a backend extension. Flex runs the reference built from Burn's tensor operations;
//! CUDA runs one fused CubeCL kernel per step, which reads the state once and writes the new
//! state in place when the caller hands over the only reference to it, as a decoding loop that
//! threads the state through does. It is forward only: decoding takes no gradient. On an
//! autodiff device a step whose inputs are not tracked runs the inner backend's step and returns
//! untracked tensors; one with a tracked input runs the reference on the autodiff backend, so its
//! gradients are those of the reference, slow but right.
//!
//! Several sequences packed into one row (`cu_seqlens`) are not supported: each batch row is one
//! sequence.

use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::tensor::FloatTensor;
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Dispatch, Flex, TensorMetadata, backend_extension,
};
use burn::tensor::{DType, FloatDType, Shape, Tensor};

#[cfg(feature = "cuda")]
mod kernel;

/// The epsilon of transformers' (and fla's) l2 normalisation of `q` and `k`.
const L2NORM_EPS: f32 = 1e-6;

/// Counters tests read to tell which path ran. Not part of the API. The step has no written-out
/// backward pass to count: a tracked step runs the reference on the autodiff backend.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static KERNEL_LAUNCHES: AtomicUsize = AtomicUsize::new(0);

    /// Launches of the step's kernel, one a step.
    pub fn kernel_launches() -> usize {
        KERNEL_LAUNCHES.load(Ordering::SeqCst)
    }
}

#[backend_extension(Flex, Cube: cfg(feature = "cuda"), Fusion: cfg(feature = "fusion"), Autodiff)]
pub trait GatedDeltaStepBackend: Backend {
    /// One token of the recurrent gated delta rule; see the [module documentation](self).
    /// Returns the output `[B, H, Dv]`, in `q`'s dtype, and the new state `[B, H, Dk, Dv]`, f32.
    /// The default body is the reference, built from Burn's tensor operations.
    #[fusion(meta = |q, _k, v, _g, _beta, state, _use_qk_l2norm| (
        burn::backend::fusion::custom::TensorSpec::new(v.shape.clone(), q.dtype),
        state.clone(),
    ))]
    fn gated_delta_step(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        state: FloatTensor<Self>,
        use_qk_l2norm: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        reference::<Self>(q, k, v, g, beta, state, use_qk_l2norm)
    }
}

impl GatedDeltaStepBackend for Flex {}

/// Decoding takes no gradient, so an untracked step runs the inner backend's step (the kernel on
/// CUDA) and records nothing. A step with a tracked input falls back to the reference on the
/// autodiff backend, which records Burn's own operations: no written-out backward pass.
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
        use_qk_l2norm: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let tracked = [&q, &k, &v, &g, &beta, &state]
            .iter()
            .any(|tensor| tensor.is_tracked());
        if tracked {
            return reference::<Self>(q, k, v, g, beta, state, use_qk_l2norm);
        }
        let inner = <Self as AutodiffBackend>::inner;
        let (out, state) = B::gated_delta_step(
            inner(q),
            inner(k),
            inner(v),
            inner(g),
            inner(beta),
            inner(state),
            use_qk_l2norm,
        );
        (
            <Self as AutodiffBackend>::from_inner(out),
            <Self as AutodiffBackend>::from_inner(state),
        )
    }
}

/// One token of the recurrent gated delta rule.
///
/// - `q`, `k`: `[B, H, Dk]`; `v`: `[B, H, Dv]`, all three of one float dtype;
/// - `g`: `[B, H]`, the log of the decay (`<= 0`); `beta`: `[B, H]`; each in `q`'s dtype or f32
///   (read as f32 whatever their dtype);
/// - `state`: `[B, H, Dk, Dv]`, f32: zeros before the first token, or the final state of a
///   chunked pass over the prompt;
/// - `use_qk_l2norm`: transformers' `use_qk_l2norm_in_kernel`, which Qwen3.5 sets.
///
/// Returns the output `[B, H, Dv]` in `q`'s dtype and the new state `[B, H, Dk, Dv]` in f32. The
/// queries are scaled by `1 / sqrt(Dk)`, as in transformers. On CUDA the new state is written
/// over the old one when `state` is the only reference to it; pass it by value from one step to
/// the next to get that.
///
/// # Panics
///
/// If the shapes do not fit together as above, if `q`, `k` and `v` differ in dtype, or if the
/// state is not f32.
pub fn gated_delta_step(
    q: Tensor<3>,
    k: Tensor<3>,
    v: Tensor<3>,
    g: Tensor<2>,
    beta: Tensor<2>,
    state: Tensor<4>,
    use_qk_l2norm: bool,
) -> (Tensor<3>, Tensor<4>) {
    check_inputs(
        [
            &q.shape(),
            &k.shape(),
            &v.shape(),
            &g.shape(),
            &beta.shape(),
            &state.shape(),
        ],
        [q.dtype(), k.dtype(), v.dtype(), state.dtype()],
    );
    let (out, state) = <Dispatch as GatedDeltaStepBackend>::gated_delta_step(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        g.into_dispatch(),
        beta.into_dispatch(),
        state.into_dispatch(),
        use_qk_l2norm,
    );
    (Tensor::from_dispatch(out), Tensor::from_dispatch(state))
}

/// [`gated_delta_step`] through Burn's tensor operations alone, on any device: the reference
/// that Flex runs and the CUDA kernel is held to. For tests and benchmarks.
#[doc(hidden)]
pub fn gated_delta_step_reference(
    q: Tensor<3>,
    k: Tensor<3>,
    v: Tensor<3>,
    g: Tensor<2>,
    beta: Tensor<2>,
    state: Tensor<4>,
    use_qk_l2norm: bool,
) -> (Tensor<3>, Tensor<4>) {
    check_inputs(
        [
            &q.shape(),
            &k.shape(),
            &v.shape(),
            &g.shape(),
            &beta.shape(),
            &state.shape(),
        ],
        [q.dtype(), k.dtype(), v.dtype(), state.dtype()],
    );
    let (out, state) = reference::<Dispatch>(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        g.into_dispatch(),
        beta.into_dispatch(),
        state.into_dispatch(),
        use_qk_l2norm,
    );
    (Tensor::from_dispatch(out), Tensor::from_dispatch(state))
}

/// Checks that the step's inputs fit together and returns `[B, H, Dk, Dv]`. `shapes` are those
/// of `q, k, v, g, beta, state`; `dtypes` those of `q, k, v, state`.
fn check_inputs(shapes: [&Shape; 6], dtypes: [DType; 4]) -> [usize; 4] {
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
    assert_eq!(
        state.as_slice(),
        &[b, h, dk, dv],
        "gated_delta_step: state must be [B, H, Dk, Dv]"
    );
    let [q, k, v, state] = dtypes;
    assert!(
        q.is_float() && k == q && v == q,
        "gated_delta_step: q, k and v must share one float dtype, got {q:?}, {k:?}, {v:?}"
    );
    assert_eq!(state, DType::F32, "gated_delta_step: the state is f32");
    [b, h, dk, dv]
}

/// The reference: the operations of `torch_recurrent_gated_delta_rule` for one token, in f32,
/// from Burn's tensor operations. Every backend can run it; it is what the kernel is held to.
fn reference<B: Backend>(
    q: FloatTensor<B>,
    k: FloatTensor<B>,
    v: FloatTensor<B>,
    g: FloatTensor<B>,
    beta: FloatTensor<B>,
    state: FloatTensor<B>,
    use_qk_l2norm: bool,
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
    let q = B::float_mul_scalar(q, (1.0 / (dk as f64).sqrt() as f32).into());

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
