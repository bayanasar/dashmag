//! Low-rank products: `y = base + scale · (x U) V` for factors `U` `[in, r]` and `V` `[r, out]` of
//! small rank `r`, with their gradients. LoRA's term is this (`U` and `V` PEFT's `lora_A` and
//! `lora_B` weights transposed, `scaling = lora_alpha / r`, or `/ √r` with rsLoRA), as are the
//! adapters built on it (DoRA's low-rank part, LoRA-FA, PiSSA's residual, QLoRA's over a
//! dequantized base) and a linear layer factored into two (`base` absent, `scale` 1).
//!
//! ```text
//! h[n, j] = sum_k x[n, k] U[k, j]                      [N, r], f32
//! y[n, m] = base[n, m] + scale · sum_j h[n, j] V[j, m]  [N, out]
//! ```
//!
//! # Shapes and dtypes
//!
//! - `x`: `[.., in]`, f32, bf16 or f16, its leading dimensions any flattening of `N` rows.
//! - `U` `[in, r]` and `V` `[r, out]`: f32, as an adapter's factors are kept whatever the model's
//!   dtype (PEFT's default for a bf16 model), with any strides: a transposed view of PEFT's
//!   `[r, in]` and `[out, r]` is read in place.
//! - `base`: `[.., out]` with the leading dimensions of `x`, in any float dtype, or none.
//! - The output: `[.., out]`, the sum computed in f32 and rounded once to the base's dtype, as
//!   PEFT's `(result + lora).to(result.dtype)` does; without a base, `scale · (x U) V` in f32.
//!
//! Every product is f32 (IEEE f32 on CUDA with `CUBECL_CUDA_TF32=0`): `x` is read exactly into
//! f32, and the products and their sums are f32 as PEFT's on an f32 adapter. The gradients: `dx`
//! in `x`'s dtype, rounded once from f32; `dU` and `dV` f32; the base's, the output's gradient
//! itself. `dz = dy · scale` is rounded to f32 before it enters the products, as autodiff of
//! `(h V) · scale` computes it.
//!
//! # Backends
//!
//! The trait's default bodies ([`LowRankBackend`]) are the reference: the definition above in
//! Burn's matrix products and elementwise operations, in PEFT's order. On a CubeCL device (any
//! runtime) ranks 1 to [`MAX_RANK`] run four CubeCL kernels instead, each streaming its wide
//! operand once while the narrow `[N, r]` side and the factors stay on chip:
//!
//! - `down`, `h = x U` and backward `dh = (dy · scale) Vᵀ`: a cube owns 32 rows (more for ranks
//!   under 16) and a run of `in`, stages the factor's slice for it in shared memory, and each
//!   unit sums one output of two rows over the run in order;
//! - `up`, `y = base + scale · h V` and backward `dx = dh Uᵀ`: each unit owns a few consecutive
//!   columns of a block of rows, holds their factor columns in registers, and writes the rounded
//!   sum, so the base is read and the output written once, and the f32 `[N, out]` term is never
//!   held;
//! - `gather`, `dV = hᵀ dz` and `dU = xᵀ dh`: each unit owns a few columns of the wide operand and
//!   some of the `r` outputs per column, and sums them over a run of rows in order;
//! - a last kernel adds the runs' partial sums in order, where `down` or `gather` split their sum
//!   into runs to spread it over the device (only for long sums: a sum of up to 512 terms, or of
//!   up to 64 rows, is one run).
//!
//! So every output element is a sum in order, with fused multiply-adds, over runs of its
//! terms, the runs then added in order: for short sums the order a plain matrix product takes,
//! and on any device the same bits whatever the launch. The kernels ask for cubes of up to 256
//! units, 17 KiB of shared memory and no plane operations, which CUDA, wgpu (lavapipe and OpenGL
//! too) and ROCm offer; CubeCL's CPU runtime, with as many units per cube as cores, runs the
//! reference.
//!
//! # Measured
//!
//! On an otherwise idle RTX 5070 Ti (2026-10-27), `scale · (x U) V` alone with r 16 and bf16 `x`,
//! by device time (the sum of the kernels' times under Nsight Systems), against PEFT's
//! `lora_B(lora_A(x.float())) * scaling` on PyTorch 2.13 (cuBLAS), for `in -> out` 1024 -> 3584,
//! 3584 -> 1024, 1024 -> 2048 and 1024 -> 256 (`examples/lowrank_bench.rs` and dash-lora's
//! `examples/lora_layer.rs`, the latter's passes as its documentation says; Burn's matrix
//! products over 6144 tokens were timed on a card shared with other work):
//!
//! | tokens | pass | dash-ops | PEFT | Burn's matrix products |
//! |---|---|---|---|---|
//! | 2048 | forward | 0.042 / 0.052 / 0.032 / 0.018 ms | 0.088 / 0.108 / 0.048 / 0.027 ms | 0.32 / 0.90 / 0.29 / 0.26 ms |
//! | 2048 | forward + backward | 0.19 / 0.17 / 0.14 / 0.069 ms | 0.27 / 0.24 / 0.15 / 0.063 ms | 1.6 / 1.8 / 1.2 / 0.84 ms |
//! | 6144 | forward + backward | 0.79 / 0.52 / 0.38 / 0.14 ms | 1.06 / 0.87 / 0.64 / 0.19 ms | 3.6 / 3.3 / 2.8 / 2.1 ms |
//!
//! The kernels read `x` in bf16, where PEFT copies it to f32 first, and with a base write the
//! rounded sum where PEFT writes, scales and adds the f32 term in three more passes over it. The
//! last column is what ran here before: Burn's default matrix-product kernels on these shapes, with
//! the elementwise operations between them. The kernels are not tuned, and `fusion` leaves their
//! times as they are; in builds with `autotune`, `down` comes out of the compiler differently (the
//! same kernel, another hash) and at 6144 rows of 3584 takes 1.5 times as long (0.35 ms against
//! 0.22), which is left to find. `down` is also the furthest from the memory's speed: at 6144 rows
//! it streams `x` at about 240 GB/s, where `up` moves its rows at about 600.
//!
//! On `Autodiff` the forward records one node on `x` and the factors, keeping them and `h`
//! (`[N, r]` f32), whose backward runs [`LowRankBackend::low_rank_backward`] for the gradients its
//! parents want; with a base, one more on the base and that node, which gives the base its
//! gradient (the output's) and hands it to the first, so that the gradients reaching `x` add up in
//! the order Burn's autodiff of the reference adds them.

use burn::backend::autodiff::NodeId;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::tensor::FloatTensor;
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Dispatch, ExtensionType, TensorMetadata, backend_extension,
};
use burn::tensor::{DType, Device, FloatDType, Shape, Tensor};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

#[cfg(feature = "flex")]
use burn::backend::Flex;

#[cfg(feature = "cubecl")]
mod cube;
mod reference;

/// The largest rank the CubeCL kernels take; larger ones run the reference. A unit of the `up`
/// kernel holds a factor column of `r` values per output column it owns in registers.
pub const MAX_RANK: usize = 64;

/// The arguments of [`low_rank`] that have a usual default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LowRankOptions {
    /// The factor on `(x U) V`, applied in f32 before the base is added: LoRA's
    /// `lora_alpha / r` (`lora_alpha / √r` with rsLoRA). Default: 1.
    pub scale: f64,
}

impl Default for LowRankOptions {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

impl LowRankOptions {
    /// The options with `scale`.
    pub fn with_scale(self, scale: f64) -> Self {
        Self { scale }
    }
}

/// A tensor or none, as the backend extension takes and returns them (an `Option` of a tensor
/// does not cross Burn's dispatch boundary; this enum does): the base, and the gradients a
/// backward pass is asked for.
#[derive(ExtensionType, Clone, Debug)]
#[extension_type(fusion: cfg(feature = "fusion"))]
pub enum MaybeTensor<B: Backend> {
    /// No tensor: no base, or a gradient not asked for.
    None,
    /// The tensor.
    Some(FloatTensor<B>),
}

impl<B: Backend> MaybeTensor<B> {
    /// The tensor as an `Option`.
    pub fn into_option(self) -> Option<FloatTensor<B>> {
        match self {
            MaybeTensor::None => None,
            MaybeTensor::Some(tensor) => Some(tensor),
        }
    }
}

impl<B: Backend> From<Option<FloatTensor<B>>> for MaybeTensor<B> {
    fn from(tensor: Option<FloatTensor<B>>) -> Self {
        match tensor {
            None => MaybeTensor::None,
            Some(tensor) => MaybeTensor::Some(tensor),
        }
    }
}

/// Which gradients a backward pass computes: of `x`, of `U`, of `V`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Wanted {
    /// The input's gradient.
    pub x: bool,
    /// `U`'s gradient.
    pub u: bool,
    /// `V`'s gradient.
    pub v: bool,
}

/// Counters a test reads to tell which code ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static DOWN_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static UP_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static GATHER_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static SUM_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);

    /// Launches of this module's CubeCL kernels, all of them (none where the reference runs).
    pub fn kernel_launches() -> usize {
        down_launches() + up_launches() + gather_launches() + sum_launches()
    }

    /// Runs of the backward pass written out for `Autodiff`.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }

    /// Launches of `down`: one per forward pass, one per backward pass that wants `dx` or `dU`.
    pub fn down_launches() -> usize {
        DOWN_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of `up`: one per forward pass, one per backward pass that wants `dx`.
    pub fn up_launches() -> usize {
        UP_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of `gather`: one per backward pass for `dV` and one for `dU`, if wanted.
    pub fn gather_launches() -> usize {
        GATHER_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of the kernel adding partial sums, after a `down` or `gather` split into runs.
    pub fn sum_launches() -> usize {
        SUM_LAUNCHES.load(Ordering::SeqCst)
    }
}

/// Whether the CubeCL kernels, rather than the reference, take [`low_rank`] and its backward pass
/// on `device` for `x` `[rows, inp]` in `dtype`, rank `rank` and `out` outputs: ranks 1 to
/// [`MAX_RANK`], `x` in f32, bf16 or f16, on a device with what the kernels ask (module
/// documentation). `false` on any device not CubeCL's. For tests, as the [`probe`] counters are.
#[doc(hidden)]
pub fn kernels_take(
    device: &Device,
    dtype: DType,
    rank: usize,
    [rows, inp, out]: [usize; 3],
) -> bool {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return cube::kernels_take(&client, dtype, rank, [rows, inp, out]);
    }
    let _ = (device, dtype, rank, rows, inp, out);
    false
}

#[cfg(feature = "fusion")]
mod fusion_meta {
    use super::{MaybeTensorMetadata, Wanted};
    use burn::backend::fusion::custom::TensorSpec;
    use burn::tensor::{DType, Shape};

    pub(super) fn forward(
        x: &TensorSpec,
        u: &TensorSpec,
        v: &TensorSpec,
        base: &MaybeTensorMetadata,
        _scale: &f32,
    ) -> (TensorSpec, TensorSpec) {
        let rows = x.shape[0];
        let y = match base {
            MaybeTensorMetadata::Some(base) => base.clone(),
            MaybeTensorMetadata::None => {
                TensorSpec::new(Shape::new([rows, v.shape[1]]), DType::F32)
            }
        };
        (
            y,
            TensorSpec::new(Shape::new([rows, u.shape[1]]), DType::F32),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn backward(
        x: &TensorSpec,
        u: &TensorSpec,
        v: &TensorSpec,
        _h: &TensorSpec,
        _grad: &TensorSpec,
        _scale: &f32,
        wanted: &Wanted,
    ) -> (
        MaybeTensorMetadata,
        MaybeTensorMetadata,
        MaybeTensorMetadata,
    ) {
        let pick = |want: bool, spec: &TensorSpec| match want {
            true => MaybeTensorMetadata::Some(spec.clone()),
            false => MaybeTensorMetadata::None,
        };
        (pick(wanted.x, x), pick(wanted.u, u), pick(wanted.v, v))
    }
}

/// The low-rank product as a backend extension, on rows: call it through [`low_rank`]. The
/// default bodies are the reference; the [module](self) gives the definition and dtypes.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(feature = "cubecl"),
    Fusion: cfg(feature = "fusion"),
    Autodiff
)]
pub trait LowRankBackend: Backend {
    /// `(y, h)` for `x` `[N, in]`, `u` `[in, r]`, `v` `[r, out]` and the base `[N, out]`: `h = x U`
    /// `[N, r]` f32, what the backward pass needs, and `y = base + scale · h V` `[N, out]` in the
    /// base's dtype, or in f32 without one. On `Autodiff`, `y` carries the gradients; `h` none.
    #[fusion(meta = fusion_meta::forward)]
    fn low_rank(
        x: FloatTensor<Self>,
        u: FloatTensor<Self>,
        v: FloatTensor<Self>,
        #[extension_type] base: MaybeTensor<Self>,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        reference::forward::<Self>(x, u, v, base, scale)
    }

    /// The gradients of `y` with respect to `x`, `u` and `v` that `wanted` names (the others
    /// `None`), given `h` from the forward pass and the gradient of `y`, `grad` `[N, out]` in
    /// `y`'s dtype: `dx` `[N, in]` in `x`'s dtype, `dU` `[in, r]` and `dV` `[r, out]` f32.
    #[fusion(meta = fusion_meta::backward)]
    #[allow(clippy::type_complexity)]
    fn low_rank_backward(
        x: FloatTensor<Self>,
        u: FloatTensor<Self>,
        v: FloatTensor<Self>,
        h: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        scale: f32,
        wanted: Wanted,
    ) -> (MaybeTensor<Self>, MaybeTensor<Self>, MaybeTensor<Self>) {
        reference::backward::<Self>(x, u, v, h, grad, scale, wanted)
    }
}

#[cfg(feature = "flex")]
impl LowRankBackend for Flex {}

/// The gradient the base's node hands to the low-rank node: an `Autodiff` node has one output,
/// so with a base the low-rank node outputs a token, and the node adding the base on it outputs
/// `y` and leaves `y`'s gradient here.
type Handover<B> = Arc<Mutex<Option<FloatTensor<B>>>>;

/// The node `Autodiff` records for the low-rank product: parents `x`, `U` and `V`.
#[derive(Debug)]
struct LowRankNode;

/// What the backward pass keeps: the inputs, through the checkpointer, `h` and the scale; with a
/// base, where the base's node leaves the output's gradient.
#[derive(Clone, Debug)]
struct NodeState<B: Backend> {
    x: NodeId,
    u: NodeId,
    v: NodeId,
    h: FloatTensor<B>,
    scale: f32,
    handover: Option<Handover<B>>,
}

impl<B: Backend + LowRankBackend> Backward<B, 3> for LowRankNode {
    type State = NodeState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        probe::BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
        let node_grad = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let grad = match state.handover {
            None => node_grad,
            Some(shared) => shared
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .expect("the base's node runs first and hands its gradient over"),
        };
        let [x_node, u_node, v_node] = ops.parents;
        let wanted = Wanted {
            x: x_node.is_some(),
            u: u_node.is_some(),
            v: v_node.is_some(),
        };
        let mut retrieve = |id| checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(id);
        let (x, u, v) = (retrieve(state.x), retrieve(state.u), retrieve(state.v));
        let (dx, du, dv) = B::low_rank_backward(x, u, v, state.h, grad, state.scale, wanted);
        for (node, grad) in [(x_node, dx), (u_node, du), (v_node, dv)] {
            if let (Some(node), MaybeTensor::Some(grad)) = (node, grad) {
                grads.register::<B>(node.id, grad);
            }
        }
    }
}

/// The node adding the base: parents the base and the low-rank node's token, output `y`. Its
/// backward gives the base `y`'s gradient as it is, and hands it to the low-rank node.
///
/// The two nodes stand where Burn's autodiff of `base.cast(f32) + scale · (x U) V` puts the sum
/// and the product's first node: the low-rank node one level above `x` and the factors, and the
/// sum above it and the base. Burn runs a backward pass level by level (a node's level one above
/// its highest input's), so for a base computed from `x` and a weight at the factors' level (as
/// views of the same kind), the gradients that reach `x` from the base and from the low-rank term
/// arrive in the order they do through Burn's own operations, and add up (in `x`'s dtype) to the
/// same values.
#[derive(Debug)]
struct BaseNode;

impl<B: Backend> Backward<B, 2> for BaseNode {
    type State = Handover<B>;

    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _: &mut Checkpointer) {
        let grad = grads.consume::<B>(&ops.node);
        let [base_node, token_node] = ops.parents;
        if let Some(node) = token_node {
            *ops.state.lock().unwrap_or_else(|e| e.into_inner()) = Some(grad.clone());
            // The token's gradient carries nothing; it only makes the low-rank node run.
            let token = B::float_zeros(Shape::new([1]), &grad.device(), FloatDType::F32);
            grads.register::<B>(node.id, token);
        }
        if let Some(node) = base_node {
            grads.register::<B>(node.id, grad);
        }
    }
}

impl<B: Backend + LowRankBackend, C: CheckpointStrategy> LowRankBackend for Autodiff<B, C> {
    fn low_rank(
        x: FloatTensor<Self>,
        u: FloatTensor<Self>,
        v: FloatTensor<Self>,
        base: MaybeTensor<Self>,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let base = base.into_option();
        let base_node = base.as_ref().map(|base| base.node());
        let nodes = [x.node(), u.node(), v.node()];
        let untracked = <Self as AutodiffBackend>::from_inner;
        let prepared = LowRankNode.prepare::<C>(nodes).compute_bound().stateful();
        let (ids, prep) = match prepared {
            OpsKind::Tracked(mut prep) => {
                let ids = [
                    prep.checkpoint(&x),
                    prep.checkpoint(&u),
                    prep.checkpoint(&v),
                ];
                (Some(ids), Ok(prep))
            }
            OpsKind::UnTracked(prep) => (None, Err(prep)),
        };
        let (y, h) = B::low_rank(
            x.into_primitive(),
            u.into_primitive(),
            v.into_primitive(),
            MaybeTensor::from(base.map(|base| base.into_primitive())),
            scale,
        );
        let Some(base_node) = base_node else {
            // No base: the low-rank node outputs `y`.
            let y = match (ids, prep) {
                (Some([x, u, v]), Ok(prep)) => {
                    let state = NodeState {
                        x,
                        u,
                        v,
                        h: h.clone(),
                        scale,
                        handover: None,
                    };
                    prep.finish(state, y)
                }
                (_, Err(prep)) => prep.finish(y),
                _ => unreachable!("tracked with its inputs kept"),
            };
            return (y, untracked(h));
        };
        let shared: Handover<B> = Arc::new(Mutex::new(None));
        let token = B::float_zeros(Shape::new([1]), &y.device(), FloatDType::F32);
        let token = match (ids, prep) {
            (Some([x, u, v]), Ok(prep)) => {
                let state = NodeState {
                    x,
                    u,
                    v,
                    h: h.clone(),
                    scale,
                    handover: Some(shared.clone()),
                };
                prep.finish(state, token)
            }
            (_, Err(prep)) => prep.finish(token),
            _ => unreachable!("tracked with its inputs kept"),
        };
        let y = match BaseNode
            .prepare::<C>([base_node, token.node()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(shared, y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        };
        (y, untracked(h))
    }

    // `low_rank_backward` keeps the default body: under `Autodiff` it is differentiable through
    // Burn's own operations, which is all a second derivative needs.
}

/// `base + scale · (x U) V` for `x` `[.., in]`, `u` `[in, r]` and `v` `[r, out]` (f32, any
/// strides), and `base` `[.., out]` with the leading dimensions of `x`: the sum in f32, rounded
/// once to the base's dtype; without a base, `scale · (x U) V` in f32. The [module](self) gives the
/// precision of each part. Differentiable with respect to `x`, `u`, `v` and `base`. With LoRA's
/// factors in PEFT's layout, `u` and `v` are `A` and `B` transposed (`a.transpose()`, a view).
///
/// A 2-D `x` is taken as it is; any other is flattened to rows and the output shaped back, which
/// records a reshape on each.
///
/// # Panics
///
/// If the shapes disagree, `u` or `v` is not f32, or `x` is not f32, bf16 or f16.
pub fn low_rank<const D: usize>(
    x: Tensor<D>,
    u: Tensor<2>,
    v: Tensor<2>,
    base: Option<Tensor<D>>,
    options: LowRankOptions,
) -> Tensor<D> {
    let op = "low_rank";
    let dims = x.dims();
    let inp = dims[D - 1];
    let [u_in, r] = u.dims();
    let [v_r, out] = v.dims();
    assert!(
        u_in == inp && v_r == r,
        "{op}: x [.., {inp}], U [in, r] = [{u_in}, {r}] and V [r, out] = [{v_r}, {out}] disagree"
    );
    assert!(
        u.dtype() == DType::F32 && v.dtype() == DType::F32,
        "{op}: U and V are f32, got {:?} and {:?}",
        u.dtype(),
        v.dtype()
    );
    assert!(
        matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16),
        "{op}: x is f32, bf16 or f16, got {:?}",
        x.dtype()
    );
    let mut out_dims = dims;
    out_dims[D - 1] = out;
    if let Some(base) = &base {
        assert_eq!(
            base.dims(),
            out_dims,
            "{op}: the base must be x's leading dimensions by out"
        );
    }
    let rows = dims[..D - 1].iter().product();
    // A matrix goes in as it is: a reshape, even to its own shape, is an operation autodiff
    // records, and would set the product a level above where Burn's own matmul on `x` sits.
    let rows_of = |t: Tensor<D>, width: usize| match D {
        2 => t.into_dispatch(),
        _ => t.reshape([rows, width]).into_dispatch(),
    };
    let (y, _) = <Dispatch as LowRankBackend>::low_rank(
        rows_of(x, inp),
        u.into_dispatch(),
        v.into_dispatch(),
        MaybeTensor::from(base.map(|base| rows_of(base, out))),
        options.scale as f32,
    );
    match D {
        2 => Tensor::from_dispatch(y),
        _ => Tensor::<2>::from_dispatch(y).reshape(out_dims),
    }
}
