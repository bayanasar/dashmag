//! The chunked gated delta rule, forward and backward, for training.
//!
//! [`chunk_gated_delta_rule`] computes the gated delta rule (the [parent module](super) states
//! it) over sequences of any length, as transformers' `torch_chunk_gated_delta_rule` and fla's
//! `chunk_gated_delta_rule` do: in chunks of 64 tokens, within a chunk through the WY
//! representation of the products of its Householder-like updates, and the state carried from
//! chunk to chunk.
//!
//! - Inputs: queries and keys `[B, T, H, Dk]`, values `[B, T, H, Dv]` in one float dtype (f32,
//!   bf16 or f16); `g`, the log of the decay (`<= 0`), and `beta`, both `[B, T, H]`, each in that
//!   dtype or in f32 (models compute `g` in f32), read as f32; an optional initial state per
//!   sequence, f32.
//! - Sequences: every batch element one sequence, or packed sequences, `B = 1` with
//!   `cu_seqlens` `[S + 1]` (the [parent module](super) gives the layout); the states are then
//!   `[S, H, Dk, Dv]`. Each packed sequence is chunked on its own, as fla does: its chunks start at
//!   its first token, the last one shorter, so no chunk holds two sequences and every chunk's
//!   arithmetic is what a call on that sequence alone does.
//! - Outputs: `o` `[B, T, H, Dv]` in the queries' dtype and, when asked, each sequence's final
//!   state `[B, H, Dk, Dv]` (packed, `[S, H, Dk, Dv]`) in f32, from which [`super::recurrent`]
//!   continues decoding.
//! - Options ([`ChunkGatedDeltaOptions`]): the l2 normalisation of queries and keys, the query
//!   scale (`1 / sqrt(Dk)` by default), and whether to return the final states.
//! - Arithmetic accumulates in f32 whatever the inputs' dtype; the outputs and gradients are
//!   rounded once, to their dtype. From f32 inputs everything is computed in IEEE f32, as
//!   transformers does. From bf16 and f16 inputs the kernels on tensor cores (below) take the
//!   products' operands in the input dtype, as fla does: the intermediates that only feed
//!   products are rounded to it. (transformers normalises bf16 queries and keys in bf16 before
//!   casting; here, as in fla, they are normalised in f32.)
//!
//! Gradients flow to the queries, keys, values, `g`, `beta` and the initial states, and from both
//! the output and the final states. The backward pass is written out: it keeps the inputs from
//! the forward pass and, where the kernels on tensor cores run, the per-chunk `T = (I + A)^-1` (in
//! the inputs' dtype, 8 KiB a chunk of each head, as fla keeps it); it recomputes each chunk's
//! other quantities, passes the state's gradient back from chunk to chunk, and forms the input
//! gradients in each chunk.
//!
//! Every backend runs the reference built from Burn's tensor operations (`chunk/reference.rs`),
//! packed sequences one at a time. On a CubeCL device (any runtime), heads up to
//! [`MAX_KERNEL_HEAD_DIM`] = 256 wide (fla's limit) run CubeCL kernels instead (`chunk/cube.rs`):
//! per chunk, one kernel forms everything that does not depend on the state, one passes the state
//! from chunk to chunk (each sequence's chunks in turn, the sequences side by side), one forms
//! the output; the backward pass forms the first kernel's quantities again, passes the state
//! forward again and its gradient back, and forms the gradients in per-chunk kernels. Packed, a
//! small kernel first lays each sequence's chunks out from `cu_seqlens` on the device, without
//! reading them to the host: the kernels launch for the most chunks `T` tokens in `S` sequences
//! can make, `ceil(T / 64) + S - 1`, and those past the last chunk return at once.
//!
//! Two sets of kernels take that shape. From bf16 and f16 inputs with `Dk` and `Dv` of 32, 64 or
//! 128, on a device with `mma.sync` m16n8k16 from the dtype into f32 and `ldmatrix` (CUDA cards of
//! compute capability 8.0 and later), every product runs on tensor cores (`chunk/cube/tc.rs`),
//! with f32 accumulation. Their intermediates that only feed products are kept in the input
//! dtype, as fla keeps them: `W`, `U`, the normalised keys, `T`, the state entering each chunk and
//! `V'`, and backward `gamma Q` (formed from the queries where the state's gradient is passed
//! back), the state's gradient leaving each chunk, `dV'`, `dV_b` and `dKbg`; the output's gradient
//! is read in that dtype by each kernel that uses it. The decays, `A` and its inversion, the state
//! carried from chunk to chunk and its gradient (in registers), the partial gradients summed
//! within a kernel, the keys' gradient passed between the last two kernels, and the final states
//! and the initial states' gradient stay f32. Backward they take six kernels: `W`, `U` and the
//! keys from the kept `T`, the state passed forward, `P^T dO` (which writes only it, and each
//! query row's normalisation factor), the state's gradient passed back, then one kernel for all
//! but the WY part (the queries' and values' gradients whole) and one for the WY part (the keys',
//! the gate's and `beta`'s). They ask of the device 256 units per cube (128 for the state
//! passings, so that each cube holds a block of 32 state columns in four warps, and for `P^T dO`,
//! so that three of its cubes fit on a multiprocessor), planes of exactly 32 units (the warps the
//! fragments are laid out over), 16-byte vectors, and shared memory up to 80.25 KiB (the output
//! kernel, at `Dk = Dv = 128`; backward, the first kernel's 73.3 KiB where it runs again, the
//! kernel for all but the WY part 68.5 KiB); the state passings hold 44 KiB forward and 48.5 KiB
//! backward at `Dk = 128`, so that two of them fit on a multiprocessor of 100 KiB. Otherwise (f32 inputs, other widths, or a device without the instructions) the f32
//! kernels run, in IEEE f32 without tensor cores, nine of them backward. Queries and keys wider
//! than 128 run f32 kernels of their own (`chunk/cube/wide.rs`) that hold them and the state's
//! rows a half (128 columns) at a time. The f32 kernels ask 256 units per cube; shared memory up
//! to 90 KiB forward and 98 KiB backward for heads up to 128 wide, and with wider heads up to 97.5
//! KiB (the first kernel's rows of the wider of `Dk` and `Dv`, at 256); and backward, planes of
//! exactly 32 units; their state passings fit one cube per multiprocessor. CUDA cards of compute
//! capability 8.0 and later have all of it. On a device without it, and for heads wider than 256,
//! the reference runs instead (`kernels_take` and `tensor_cores_take` tell which): under wgpu on
//! OpenGL over D3D12 and on lavapipe (Vulkan on the CPU), both with 32 KiB of shared memory and no
//! plane operations, and on CubeCL's CPU runtime (as many units per cube as cores), for instance.
//!
//! Measured against transformers' function at f32 (the tests' fixtures), the output, the final
//! state and every gradient agree to within 1.13e-6 relative to each tensor's largest value; from
//! bf16 inputs, through the reference and the f32 kernels, to within 2.41e-3 of the exact result
//! (torch's own bf16 path: 8.06e-3). Packed, against transformers in float64 called once per
//! sequence, to within 1.23e-6, and to fla's packed f32 path within 2.04e-6; packed and
//! per-sequence calls agree to the bit. The f32 kernels agree with the f32 reference to within
//! 1.20e-6 (heads up to 256 wide), and 3.31e-3 from bf16 inputs. On tensor cores, from bf16 inputs,
//! the errors grow to those of fla's own bf16 path: against float64 to within 6.12e-3 (packed,
//! 5.58e-3 where fla's is 6.36e-3, the output and final states agreeing with fla's errors to three
//! digits), and against the f32 reference to within 7.02e-3 (the final states), and 1.13e-2 for
//! the gate's gradient without the l2 normalisation, a difference of sums within each chunk; from
//! f16 inputs, to within 7.46e-4 of the f32 reference. On an otherwise idle RTX 5070 Ti, 6144
//! tokens with 16 heads of 128 and bf16 inputs take 2.57 ms of device time forward and backward in
//! this operation's kernels as one sequence, 2.44 ms as four packed ones (fla 0.5.2, run in turn
//! with them: 2.54 and 2.37 ms), and 0.81 and 0.79 ms forward (fla: 0.69 and 0.70 ms); the loss
//! around it in the benchmark adds 1.01 ms of Burn's kernels (PyTorch's: 0.57). With heads of 256,
//! on the f32 kernels, 19.7 and 5.4 ms (fla: 6.6 and 1.9 ms) (`examples/gated_delta_chunk_bench.rs`,
//! device times summed from `nsys`).

mod reference;

#[cfg(feature = "cubecl")]
mod cube;

use super::{Sequences, query_scale, read_cu_seqlens};
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::FloatTensor;
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Dispatch, TensorMetadata, backend_extension,
};
use burn::tensor::{DType, Device, FloatDType, Int, Shape, Slice, Tensor, TensorCreationOptions};
use std::sync::atomic::Ordering;

pub use reference::CHUNK as CHUNK_SIZE;

/// The widest heads (`Dk` and `Dv`) the CubeCL kernels take, as fla's; wider heads run the
/// reference.
pub const MAX_KERNEL_HEAD_DIM: usize = 256;

/// The options of [`chunk_gated_delta_rule`] that have a usual default; the defaults are fla's
/// and transformers': no normalisation, the scale `1 / sqrt(Dk)`, no final state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChunkGatedDeltaOptions {
    /// Normalise queries and keys to unit length first, `x / sqrt(sum(x^2) + 1e-6)`
    /// (transformers' and fla's `use_qk_l2norm_in_kernel`).
    pub qk_l2norm: bool,
    /// The queries' scale, applied after the normalisation; `None` for `1 / sqrt(Dk)` (fla's
    /// `scale=None`).
    pub scale: Option<f32>,
    /// Return each sequence's state after its last token (fla's `output_final_state`).
    pub output_final_state: bool,
}

impl ChunkGatedDeltaOptions {
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

    /// The options returning the final states, or not.
    pub fn with_final_state(self, output_final_state: bool) -> Self {
        Self {
            output_final_state,
            ..self
        }
    }
}

/// Counters tests read to tell which path ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static KERNEL_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static TENSOR_CORE_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);

    /// Launches of this module's CubeCL kernels: three for a forward pass; for a backward pass
    /// nine on the f32 kernels, six on tensor cores; one more each over packed sequences (the
    /// chunks' layout).
    pub fn kernel_launches() -> usize {
        KERNEL_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Those of [`kernel_launches`] that ran the kernels on tensor cores (all but the chunks'
    /// layout).
    pub fn tensor_core_launches() -> usize {
        TENSOR_CORE_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Runs of the written-out backward pass on `Autodiff`.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }
}

/// Whether the CubeCL kernels take a forward and a backward pass over inputs of `dtype` and
/// shape `[B, T, H, Dk, Dv]` on `device`, rather than the reference; `packed` gives the number of
/// packed sequences and the dtype of their `cu_seqlens`. Heads up to [`MAX_KERNEL_HEAD_DIM`] wide,
/// on a device with what the kernels ask (module documentation). `[false, false]` on any device
/// not CubeCL's. For tests, as the [`probe`] counters are.
#[doc(hidden)]
pub fn kernels_take(
    device: &Device,
    dtype: DType,
    dims: [usize; 5],
    packed: Option<(usize, DType)>,
) -> [bool; 2] {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return [false, true]
            .map(|backward| cube::kernels_take(&client, dtype, dims, packed, backward));
    }
    let _ = (device, dtype, dims, packed);
    [false, false]
}

/// Whether the kernels on tensor cores take a forward and a backward pass, as
/// [`kernels_take`]: bf16 or f16 inputs, `Dk` and `Dv` of 32, 64 or 128, on a device with
/// `mma.sync` m16n8k16 and `ldmatrix` for the dtype and what the kernels ask (module
/// documentation). Where they do not, the f32 kernels or the reference run. For tests.
#[doc(hidden)]
pub fn tensor_cores_take(
    device: &Device,
    dtype: DType,
    dims: [usize; 5],
    packed: Option<(usize, DType)>,
) -> [bool; 2] {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return [false, true]
            .map(|backward| cube::tensor_cores_take(&client, dtype, dims, packed, backward));
    }
    let _ = (device, dtype, dims, packed);
    [false, false]
}

/// The shape of what a forward pass over `[B, T, H, Dk, Dv]` in `dtype` keeps for its backward
/// pass when asked to (`save`), with `packed` the number of packed sequences: the per-chunk
/// `T = (I + A)^-1`, `[B, H, N, C, C]` in the inputs' dtype (8 KiB a chunk of each head), where
/// the kernels on tensor cores may take the call (bf16 or f16, `Dk` and `Dv` of 32, 64 or 128),
/// `N` the chunks of each head as they lay them out (packed, the most `S` sequences can need);
/// `None` otherwise, where a placeholder of one f32 stands in. A rule of shapes and dtypes alone,
/// as Burn's fusion asks of an output: on a device without tensor cores the forward keeps an
/// unwritten tensor of that shape, which the backward does not read.
#[cfg(any(feature = "cubecl", feature = "fusion"))]
pub(crate) fn saved_shape(
    dtype: DType,
    [b, t, h, dk, dv]: [usize; 5],
    packed: Option<usize>,
    save: bool,
) -> Option<Shape> {
    let narrow = |d: usize| matches!(d, 32 | 64 | 128);
    let n = t.div_ceil(CHUNK_SIZE) + packed.map_or(0, |seqs| seqs.saturating_sub(1));
    (save && matches!(dtype, DType::BF16 | DType::F16) && narrow(dk) && narrow(dv))
        .then(|| Shape::new([b, h, n, CHUNK_SIZE, CHUNK_SIZE]))
}

/// The one-element f32 tensor a forward pass returns where it keeps nothing for the backward.
pub(crate) fn nothing_saved<B: Backend>(device: &B::Device) -> FloatTensor<B> {
    B::float_zeros(Shape::new([1]), device, FloatDType::F32)
}

#[cfg(feature = "fusion")]
mod fusion_meta {
    use crate::sequences::SequencesMetadata;
    use burn::backend::fusion::custom::TensorSpec;
    use burn::tensor::{DType, Shape};

    /// [`super::saved_shape`] of a call's specs.
    fn saved(
        q: &TensorSpec,
        v: &TensorSpec,
        sequences: &SequencesMetadata,
        save: bool,
    ) -> TensorSpec {
        let [b, t, h, dk] = q.shape.dims::<4>();
        let dv = v.shape.dims::<4>()[3];
        let packed = match sequences {
            SequencesMetadata::Batch => None,
            SequencesMetadata::Packed(cu_seqlens) => Some(cu_seqlens.shape[0] - 1),
        };
        match super::saved_shape(q.dtype, [b, t, h, dk, dv], packed, save) {
            Some(shape) => TensorSpec::new(shape, q.dtype),
            None => TensorSpec::new(Shape::new([1]), DType::F32),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        q: &TensorSpec,
        _k: &TensorSpec,
        v: &TensorSpec,
        _g: &TensorSpec,
        _beta: &TensorSpec,
        initial_state: &TensorSpec,
        sequences: &SequencesMetadata,
        _use_qk_l2norm: &bool,
        _scale: &f32,
        _output_final_state: &bool,
        save_for_backward: &bool,
    ) -> (TensorSpec, TensorSpec, TensorSpec) {
        let [b, t, h, _] = q.shape.dims::<4>();
        let dv = v.shape.dims::<4>()[3];
        (
            TensorSpec::new(Shape::new([b, t, h, dv]), q.dtype),
            TensorSpec::new(initial_state.shape.clone(), DType::F32),
            saved(q, v, sequences, *save_for_backward),
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
        _sequences: &SequencesMetadata,
        _saved: &TensorSpec,
        _d_out: &TensorSpec,
        _d_final_state: &TensorSpec,
        _use_qk_l2norm: &bool,
        _scale: &f32,
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
/// an initial state for every sequence always present (zeros when there is none) and the query
/// scale resolved.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(feature = "cubecl"),
    Fusion: cfg(feature = "fusion"),
    Autodiff
)]
pub trait ChunkGatedDeltaBackend: Backend {
    /// The forward pass: the output `[B, T, H, Dv]` in the queries' dtype, the final states
    /// `[B, H, Dk, Dv]` (packed, `[S, H, Dk, Dv]`) in f32, and what it keeps for the backward
    /// pass when `save_for_backward` (shaped by `saved_shape`; the kernels on tensor cores keep
    /// the per-chunk `T`, the reference nothing). `output_final_state` only tells Autodiff whether
    /// the final states take part in the graph; they are always computed.
    #[fusion(meta = fusion_meta::forward)]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn chunk_gated_delta_forward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        #[extension_type] sequences: Sequences<Self>,
        use_qk_l2norm: bool,
        scale: f32,
        output_final_state: bool,
        save_for_backward: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let _ = (output_final_state, save_for_backward);
        let device = q.device();
        let (out, state) = reference_forward::<Self>(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            sequences,
            use_qk_l2norm,
            scale,
        );
        (out, state, nothing_saved::<Self>(&device))
    }

    /// The backward pass: the gradients of the queries, keys, values, `g`, `beta` (each in its
    /// input's dtype) and the initial states (f32), given what the forward pass kept (`saved`;
    /// what it does not find there it computes again) and the gradients of the output and of
    /// the final states (zeros when the final states are not used).
    #[fusion(meta = fusion_meta::backward)]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn chunk_gated_delta_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        #[extension_type] sequences: Sequences<Self>,
        saved: FloatTensor<Self>,
        d_out: FloatTensor<Self>,
        d_final_state: FloatTensor<Self>,
        use_qk_l2norm: bool,
        scale: f32,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let _ = saved;
        reference_backward::<Self>(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            sequences,
            d_out,
            d_final_state,
            use_qk_l2norm,
            scale,
        )
    }
}

#[cfg(feature = "flex")]
impl ChunkGatedDeltaBackend for burn::backend::Flex {}

/// The reference forward on any backend: batched, or packed sequences one at a time (their
/// boundaries read to the host).
#[allow(clippy::too_many_arguments)]
pub(crate) fn reference_forward<B: Backend>(
    q: FloatTensor<B>,
    k: FloatTensor<B>,
    v: FloatTensor<B>,
    g: FloatTensor<B>,
    beta: FloatTensor<B>,
    initial_state: FloatTensor<B>,
    sequences: Sequences<B>,
    use_qk_l2norm: bool,
    scale: f32,
) -> (FloatTensor<B>, FloatTensor<B>) {
    match sequences {
        Sequences::Batch => {
            reference::forward::<B>(q, k, v, g, beta, initial_state, use_qk_l2norm, scale)
        }
        Sequences::Packed(cu_seqlens) => {
            let cu = read_cu_seqlens::<B>(cu_seqlens, initial_state.shape()[0], q.shape()[1]);
            reference::forward_packed::<B>(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                &cu,
                use_qk_l2norm,
                scale,
            )
        }
    }
}

/// The reference's written-out backward on any backend, as [`reference_forward`].
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn reference_backward<B: Backend>(
    q: FloatTensor<B>,
    k: FloatTensor<B>,
    v: FloatTensor<B>,
    g: FloatTensor<B>,
    beta: FloatTensor<B>,
    initial_state: FloatTensor<B>,
    sequences: Sequences<B>,
    d_out: FloatTensor<B>,
    d_final_state: FloatTensor<B>,
    use_qk_l2norm: bool,
    scale: f32,
) -> (
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
) {
    match sequences {
        Sequences::Batch => reference::backward::<B>(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            d_out,
            d_final_state,
            use_qk_l2norm,
            scale,
        ),
        Sequences::Packed(cu_seqlens) => {
            let cu = read_cu_seqlens::<B>(cu_seqlens, initial_state.shape()[0], q.shape()[1]);
            reference::backward_packed::<B>(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                &cu,
                d_out,
                d_final_state,
                use_qk_l2norm,
                scale,
            )
        }
    }
}

/// What the backward pass keeps from the forward: the inputs, nothing computed from them.
#[derive(Clone, Debug)]
struct Saved<B: Backend> {
    q: FloatTensor<B>,
    k: FloatTensor<B>,
    v: FloatTensor<B>,
    g: FloatTensor<B>,
    beta: FloatTensor<B>,
    initial_state: FloatTensor<B>,
    sequences: Sequences<B>,
    /// What the forward pass kept for the backward (the per-chunk `T` on tensor cores).
    kept: FloatTensor<B>,
    use_qk_l2norm: bool,
    scale: f32,
    /// When the final states take part in the graph, the node's output is the output (as f32)
    /// and the final states flattened into one tensor, since an autodiff node has one output;
    /// this is the output's shape, for splitting the gradient.
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
            s.sequences,
            s.kept,
            d_out,
            d_final_state,
            s.use_qk_l2norm,
            s.scale,
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

/// The sequences of an `Autodiff` call on its inner backend: int tensors are not tracked.
fn inner_sequences<B: Backend, C: CheckpointStrategy>(
    sequences: Sequences<Autodiff<B, C>>,
) -> Sequences<B> {
    match sequences {
        Sequences::Batch => Sequences::Batch,
        Sequences::Packed(cu_seqlens) => Sequences::Packed(cu_seqlens),
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
        sequences: Sequences<Self>,
        use_qk_l2norm: bool,
        scale: f32,
        output_final_state: bool,
        save_for_backward: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        // Autodiff keeps what the backward pass wants exactly when it records the node.
        let _ = save_for_backward;
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
        let sequences = inner_sequences(sequences);
        match ChunkBackward.prepare::<C>(nodes).compute_bound().stateful() {
            OpsKind::UnTracked(prep) => {
                let (out, state, kept) = B::chunk_gated_delta_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state,
                    sequences,
                    use_qk_l2norm,
                    scale,
                    output_final_state,
                    false,
                );
                (
                    prep.finish(out),
                    Self::from_inner(state),
                    Self::from_inner(kept),
                )
            }
            OpsKind::Tracked(prep) => {
                let inputs = (
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    g.clone(),
                    beta.clone(),
                    initial_state.clone(),
                    sequences.clone(),
                );
                let (out, state, kept) = B::chunk_gated_delta_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state,
                    sequences,
                    use_qk_l2norm,
                    scale,
                    output_final_state,
                    true,
                );
                let nothing = nothing_saved::<Self>(&out.device());
                let mut saved = Saved {
                    q: inputs.0,
                    k: inputs.1,
                    v: inputs.2,
                    g: inputs.3,
                    beta: inputs.4,
                    initial_state: inputs.5,
                    sequences: inputs.6,
                    kept,
                    use_qk_l2norm,
                    scale,
                    packed: None,
                };
                if !output_final_state {
                    return (prep.finish(saved, out), Self::from_inner(state), nothing);
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
                (out, state, nothing)
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
        sequences: Sequences<Self>,
        saved: FloatTensor<Self>,
        d_out: FloatTensor<Self>,
        d_final_state: FloatTensor<Self>,
        use_qk_l2norm: bool,
        scale: f32,
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
            inner_sequences(sequences),
            Self::inner(saved),
            Self::inner(d_out),
            Self::inner(d_final_state),
            use_qk_l2norm,
            scale,
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

/// Checks the inputs of one call against each other and returns the initial states, f32 (zeros
/// where none are given), `[S, H, Dk, Dv]` with `S` the number of sequences.
fn check_inputs(
    [q, k, v]: [&Tensor<4>; 3],
    [g, beta]: [&Tensor<3>; 2],
    initial_state: Option<Tensor<4>>,
    cu_seqlens: Option<&Tensor<1, Int>>,
) -> Tensor<4> {
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
    let sequences = match cu_seqlens {
        None => b,
        Some(cu_seqlens) => {
            assert_eq!(
                b, 1,
                "packed sequences lie along T of a batch of one, got B = {b}"
            );
            let [n] = cu_seqlens.dims();
            assert!(n >= 2, "cu_seqlens must hold S + 1 >= 2 offsets, got {n}");
            n - 1
        }
    };
    match initial_state {
        Some(state) => {
            assert_eq!(
                state.dims(),
                [sequences, h, dk, dv],
                "the initial state must be [S, H, Dk, Dv], one per sequence"
            );
            if state.dtype() == DType::F32 {
                state
            } else {
                state.cast(DType::F32)
            }
        }
        None => Tensor::zeros(
            [sequences, h, dk, dv],
            TensorCreationOptions::new(q.device()).with_dtype(DType::F32),
        ),
    }
}

/// The chunked gated delta rule over `B` sequences, or over `S` packed ones (chunk size 64; the
/// [module](self) and its [parent](super) give the definition).
///
/// - `q`, `k`: `[B, T, H, Dk]`; `v`: `[B, T, H, Dv]`, one float dtype for the three.
/// - `g`: `[B, T, H]`, the log of the decay, `<= 0`; `beta`: `[B, T, H]`; each in the queries'
///   dtype or f32 (read as f32 whatever their dtype).
/// - `initial_state`: `[B, H, Dk, Dv]` (packed, `[S, H, Dk, Dv]`), f32 (other dtypes are cast);
///   zeros when `None`.
/// - `cu_seqlens`: `None` for one sequence per batch element; else `[S + 1]`, the boundaries of
///   `S` sequences packed along `T` of a batch of one (`B = 1`), from 0 to `T` without
///   decreasing. On a device running the kernels they are not read back to check them.
/// - `options`: the l2 normalisation of queries and keys, the query scale, the final states.
///
/// Returns the output `[B, T, H, Dv]` in the queries' dtype and, if asked, the final states,
/// shaped as the initial states, in f32.
///
/// # Panics
///
/// If the shapes do not agree, `q`, `k`, `v` differ in dtype, `T` is 0, or packed sequences
/// come with `B > 1`; where the reference runs, if `cu_seqlens` does not go from 0 to `T`
/// without decreasing.
#[allow(
    clippy::too_many_arguments,
    reason = "the five inputs, the two that may be absent, the options: fla's arguments"
)]
pub fn chunk_gated_delta_rule(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    initial_state: Option<Tensor<4>>,
    cu_seqlens: Option<Tensor<1, Int>>,
    options: ChunkGatedDeltaOptions,
) -> (Tensor<4>, Option<Tensor<4>>) {
    let initial_state = check_inputs(
        [&q, &k, &v],
        [&g, &beta],
        initial_state,
        cu_seqlens.as_ref(),
    );
    let scale = query_scale(options.scale, q.dims()[3]);
    let (out, state, _) = <Dispatch as ChunkGatedDeltaBackend>::chunk_gated_delta_forward(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        g.into_dispatch(),
        beta.into_dispatch(),
        initial_state.into_dispatch(),
        Sequences::from(cu_seqlens.map(Tensor::into_dispatch)),
        options.qk_l2norm,
        scale,
        options.output_final_state,
        false,
    );
    (
        Tensor::from_dispatch(out),
        options
            .output_final_state
            .then(|| Tensor::from_dispatch(state)),
    )
}

/// The reference built from Burn's tensor operations, on any device, without autodiff: the
/// output and the final states. For tests and benchmarks that hold the kernels to it.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule_reference(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    initial_state: Option<Tensor<4>>,
    cu_seqlens: Option<Tensor<1, Int>>,
    options: ChunkGatedDeltaOptions,
) -> (Tensor<4>, Tensor<4>) {
    let initial_state = check_inputs(
        [&q, &k, &v],
        [&g, &beta],
        initial_state,
        cu_seqlens.as_ref(),
    );
    let scale = query_scale(options.scale, q.dims()[3]);
    let (out, state) = reference_forward::<Dispatch>(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        g.into_dispatch(),
        beta.into_dispatch(),
        initial_state.into_dispatch(),
        Sequences::from(cu_seqlens.map(Tensor::into_dispatch)),
        options.qk_l2norm,
        scale,
    );
    (Tensor::from_dispatch(out), Tensor::from_dispatch(state))
}

/// The reference's written-out backward pass, as [`chunk_gated_delta_rule_reference`]: the
/// gradients of the queries, keys, values, `g`, `beta` and the initial states, given those of
/// the output and the final states.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule_reference_backward(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    initial_state: Option<Tensor<4>>,
    cu_seqlens: Option<Tensor<1, Int>>,
    d_out: Tensor<4>,
    d_final_state: Tensor<4>,
    options: ChunkGatedDeltaOptions,
) -> (
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    Tensor<3>,
    Tensor<3>,
    Tensor<4>,
) {
    let initial_state = check_inputs(
        [&q, &k, &v],
        [&g, &beta],
        initial_state,
        cu_seqlens.as_ref(),
    );
    let scale = query_scale(options.scale, q.dims()[3]);
    let (dq, dk, dv, dg, d_beta, d_initial) = reference_backward::<Dispatch>(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        g.into_dispatch(),
        beta.into_dispatch(),
        initial_state.into_dispatch(),
        Sequences::from(cu_seqlens.map(Tensor::into_dispatch)),
        d_out.into_dispatch(),
        d_final_state.into_dispatch(),
        options.qk_l2norm,
        scale,
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
