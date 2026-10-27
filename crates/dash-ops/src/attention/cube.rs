//! The CubeCL backend, for every CubeCL runtime: launches the kernels of `kernels.rs` for the
//! inputs they take (bf16 or f16, head dimension 64, 128 or 256) on a device with what they ask
//! (the [`Requirements`] of each pass, `mma.sync` m16n8k16 and `ldmatrix` for the dtype), and
//! runs the reference for the rest.
//!
//! Inputs are read through their strides, in place where the kernels can (see [`readable`]).
//! Every tensor the kernels write is allocated contiguous, in the layout the module
//! documentation gives: CubeCL's default allocation pads rows to the device's pitch, and the
//! kernels index their outputs and the per-row statistics densely.

use super::kernels::{
    BLOCK_KV, FORWARD_BLOCK_Q, backward_kernel, decode_merge_kernel, decode_split_kernel,
    delta_kernel, finish_kernel, forward_kernel,
};
use super::{
    AttentionBackend, AttentionMaskPrimitive, AttentionOptions, CacheLengths, MaskShape,
    check_decode_shapes, check_shapes, probe, reference,
};
use crate::cube::Requirements;
use burn::backend::cubecl::{dtype_to_elem_type, dtype_to_storage_type};
use burn::backend::ops::{BoolTensorOps, FloatTensorOps, IntTensorOps};
use burn::backend::tensor::FloatTensor;
use burn::backend::{DType, FloatDType, IntDType, Shape};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::{into_contiguous, untile};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::client::Client;
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::features::MmaConfig;
use cubecl::ir::{AddressType, DeviceProperties, ElemType, FloatKind, Type};
use cubecl::prelude::*;
use std::sync::atomic::Ordering;

/// Elements per 16-byte vector: the kernels' unit of every load, so strides count in it.
const VECTOR: usize = 8;

/// The plane width the kernels are written for: the tensor-core fragments give each of 32
/// lanes its rows and columns, and the reductions shuffle across those lanes.
const PLANE: u32 = 32;

/// The forward's key block for head dimension `d`: 32 keys at 256, so that 128 queries, the keys
/// and the values fit in 99 KB of shared memory; 64 below.
fn block_kv(d: usize) -> u32 {
    if d > 128 { 32 } else { 64 }
}

/// `[B, Tq, Tk, Hq, Hkv, D]` of queries `[B, Tq, Hq, D]` and keys `[B, Tk, Hkv, D]`.
fn dims(query: &CubeTensor, key: &CubeTensor) -> [usize; 6] {
    let [b, tq, hq, d] = query.meta.shape().dims::<4>();
    let [_, tk, hkv, _] = key.meta.shape().dims::<4>();
    [b, tq, tk, hq, hkv, d]
}

/// Bytes of shared memory a block of `rows` queries keeps its rows' key intervals and its warp
/// groups' spans in, beside the padding flags of a tile of `keys` keys.
fn bounds_bytes(rows: usize, groups: usize, keys: usize) -> usize {
    4 * (2 * rows + 4 * groups + keys)
}

/// What the forward kernel asks: 256 units per cube, a cube per 128 queries of each query head
/// (and batch element), the query, key and value tiles, the rows' key intervals, the warps'
/// spans and the padding flags in shared memory, eight tensors, 16-byte vectors and planes of
/// 32.
fn forward_requirements([_, tq, _, hq, _, d]: [usize; 6]) -> Requirements {
    let bkv = block_kv(d) as usize;
    let rows = FORWARD_BLOCK_Q as usize;
    let tiles = (rows + 2 * bkv) * d * 2;
    Requirements {
        shared_bytes: tiles + bounds_bytes(rows, rows / 16, bkv),
        vector_size: VECTOR,
        plane_width: Some(PLANE),
        ..Requirements::new(
            (2 * FORWARD_BLOCK_Q, 1),
            (tq.div_ceil(FORWARD_BLOCK_Q as usize), hq),
            8,
        )
    }
}

/// What the backward kernels ask, the most of each: 256 units per cube; `delta`'s cube per
/// `256 / (D / 8)` rows of the `dQ` accumulator (whole 32-query blocks) or per 2048 elements of
/// the key and value gradients' accumulators, split; the products' per 32 keys of each key-value
/// head (two, split); the conversion's per 32 queries of each query head, beside those per 2048
/// elements of the key and value gradients, split; the products' four 32-row tiles and their
/// 32 x 32 tiles of `P` and `dS` in shared memory; the products' fifteen tensors; f32 atomic
/// additions; 16-byte vectors and planes of 32.
fn backward_requirements(dims: [usize; 6], split: bool) -> Requirements {
    let [b, tq, tk, hq, hkv, d] = dims;
    let rows_per_cube = 256 / (d / VECTOR);
    let block = BLOCK_KV as usize;
    let q_blocks = tq.div_ceil(block);
    let kv_cubes = if split {
        (b * tk * hkv * d / VECTOR).div_ceil(256)
    } else {
        0
    };
    let rows = b * q_blocks * block * hq;
    let delta = Requirements::new(
        (256, 1),
        (rows.div_ceil(rows_per_cube).max(kv_cubes), 1),
        11,
    );
    let products = Requirements {
        shared_bytes: 4 * block * d * 2 + 2 * block * block * 2 + 4 * 4 * block,
        ..Requirements::new(
            (256, 1),
            (tk.div_ceil(block) * (1 + usize::from(split)), hkv),
            15,
        )
    };
    let finish = Requirements::new((256, 1), (b * q_blocks * hq + kv_cubes, 1), 6);
    Requirements {
        vector_size: VECTOR,
        plane_width: Some(PLANE),
        ..delta.max(products).max(finish)
    }
}

/// Whether the products split each key block's sweep between two cubes: when that fills the
/// device's multiprocessors more evenly, one cube each at a time. The products' cubes are
/// `ceil(Tk / 32) Hkv B`; `n` equal ones on `P` multiprocessors run in `ceil(n / P)` rounds, and
/// their halves in `ceil(2 n / P)` rounds of half the length. Split, the key and value gradients
/// go through f32 accumulators, which costs a pass over them, hence the margin.
fn splits(client: &Client, [b, _, tk, _, hkv, _]: [usize; 6]) -> bool {
    if let Some(split) = probe::forced_split() {
        return split;
    }
    let processors = client
        .properties()
        .hardware
        .num_streaming_multiprocessors
        .unwrap_or(64) as usize;
    let cubes = tk.div_ceil(BLOCK_KV as usize) * hkv * b;
    let rounds = cubes.div_ceil(processors);
    let halves = (2 * cubes).div_ceil(processors);
    20 * halves < 19 * 2 * rounds
}

/// Whether the device has `mma.sync` m16n8k16 from `dtype` into f32, and `ldmatrix` for it.
fn tensor_cores(properties: &DeviceProperties, dtype: DType) -> bool {
    let elem = dtype_to_elem_type(dtype);
    let mma = MmaConfig {
        a_type: elem,
        b_type: elem,
        cd_type: ElemType::Float(FloatKind::F32),
        m: 16,
        n: 8,
        k: 16,
    };
    properties.features.matmul.mma.contains(&mma)
        && properties.features.matmul.ldmatrix.contains(&elem)
}

/// Whether the device adds to f32 in global memory atomically, as the backward sums `dQ`.
fn float_atomics(properties: &DeviceProperties) -> bool {
    properties
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
}

/// Whether the kernels take a forward (or backward) pass over inputs of `dtype` and dimensions
/// `[B, Tq, Tk, Hq, Hkv, D]` on the device `client` serves: bf16 or f16, a head dimension of 64,
/// 128 or 256, something to compute, positions and elements that `i32` and `u32` index, and a
/// device with what the kernels ask. The reference runs otherwise.
pub(super) fn kernels_take(
    client: &Client,
    dtype: DType,
    dims: [usize; 6],
    backward: bool,
) -> bool {
    let [b, tq, tk, hq, hkv, d] = dims;
    let properties = client.properties();
    let requirements = if backward {
        backward_requirements(dims, splits(client, dims))
    } else {
        forward_requirements(dims)
    };
    (!backward || float_atomics(properties))
        && matches!(dtype, DType::BF16 | DType::F16)
        && matches!(d, 64 | 128 | 256)
        && b * tq * hq > 0
        && tk > 0
        && tq + tk < 1 << 29
        && (b * tq.next_multiple_of(BLOCK_KV as usize).max(tk) * hq.max(hkv) * d) < 1 << 32
        && b <= properties.hardware.max_cube_count.2 as usize
        && tensor_cores(properties, dtype)
        && requirements.met_on(client, &[dtype, DType::F32, DType::I32])
}

/// What the decoding kernels ask for `[B, S, Hq, Hkv, D]` in `splits` splits: 128 units per
/// cube for the splits (four planes of 32, sharing their partial results through shared memory)
/// and `D` for the merge; six tensors; 16-byte vectors.
fn decode_requirements([b, _, hq, hkv, d]: [usize; 5], splits: usize) -> Requirements {
    let group = hq / hkv;
    let chunks = d / VECTOR;
    let slots = 4 * (32 / chunks);
    let split = Requirements {
        shared_bytes: slots * group * (2 * 4 + chunks * 32),
        vector_size: VECTOR,
        plane_width: Some(PLANE),
        ..Requirements::new((128, 1), (splits, hkv), 6)
    };
    let merge = Requirements::new((d as u32, 1), (b * hq, 1), 3);
    split.max(merge)
}

/// How decoding splits the cache, for `[B, S, Hq, Hkv, D]` and a window reaching `left` keys
/// before the query: about eight cubes per multiprocessor, each split a whole number of the first
/// pass's key slots long, over the keys a query can see at most. Returns `(split_len, splits)`.
fn decode_splits(client: &Client, [b, s, _, hkv, d]: [usize; 5], left: usize) -> (usize, usize) {
    let slots = 4 * (32 / (d / VECTOR));
    let span = s.min(left.saturating_add(1)).max(1);
    let processors = client
        .properties()
        .hardware
        .num_streaming_multiprocessors
        .unwrap_or(64) as usize;
    let wanted = (8 * processors).div_ceil(b * hkv).max(1);
    let split_len = span.div_ceil(wanted).max(slots).next_multiple_of(slots);
    (split_len, span.div_ceil(split_len))
}

/// Whether the decoding kernels take a step over inputs of `dtype` and dimensions
/// `[B, S, Hq, Hkv, D]` on the device `client` serves: as [`kernels_take`], with at most 16
/// query heads per key-value head (the first pass keeps a group's query rows in registers and
/// their partial sums in shared memory).
pub(super) fn decode_kernels_take(client: &Client, dtype: DType, dims: [usize; 5]) -> bool {
    let [b, s, hq, hkv, d] = dims;
    let properties = client.properties();
    let (_, splits) = decode_splits(client, dims, usize::MAX);
    matches!(dtype, DType::BF16 | DType::F16)
        && matches!(d, 64 | 128 | 256)
        && b * hq > 0
        && s > 0
        && s < 1 << 29
        && hq / hkv <= 16
        && (b * s * hkv * d).max(b * hq * splits * d) < 1 << 32
        && b <= properties.hardware.max_cube_count.2 as usize
        && decode_requirements(dims, splits).met_on(client, &[dtype, DType::F32, DType::I32])
}

/// Whether the kernels index every one of `tensors` with `u32`.
fn indexable(tensors: &[&CubeTensor]) -> bool {
    tensors
        .iter()
        .all(|tensor| tensor.required_address_type() == AddressType::U32)
}

/// Whether the kernels read `tensor` as it is: rows (not a matmul's storage tiles), the last
/// dimension contiguous, every other stride a whole number of 16-byte vectors, the first element
/// 16-byte aligned.
fn in_place(tensor: &CubeTensor) -> bool {
    let strides = tensor.meta.strides();
    let rank = strides.len();
    !tensor.meta.is_tiled()
        && strides[rank - 1] == 1
        && strides[..rank - 1]
            .iter()
            .all(|stride| stride % VECTOR == 0)
        && tensor.handle.offset_start.unwrap_or(0).is_multiple_of(16)
}

/// The input as the kernels read it: itself when they can ([`in_place`]), a contiguous copy
/// otherwise, counted by `probe::input_copies`.
fn readable(tensor: CubeTensor) -> CubeTensor {
    if in_place(&tensor) {
        return tensor;
    }
    probe::INPUT_COPIES.fetch_add(1, Ordering::SeqCst);
    let tensor = untile(tensor);
    if in_place(&tensor) {
        return tensor;
    }
    // A fresh allocation, so aligned even where the input was contiguous but not aligned.
    let (client, device, dtype) = (tensor.client.clone(), tensor.device.clone(), tensor.dtype);
    let copy = burn_cubecl::cubecl::std::tensor::into_contiguous(
        &client,
        tensor.binding(),
        dtype_to_storage_type(dtype),
    );
    CubeTensor::new(client, copy.handle, *copy.metadata, device, dtype)
}

/// `strides[axes[i]]` in 16-byte vectors: `[0, 2, 1]` gives the batch, head and time strides of
/// a `[B, T, H, D]` input, `[0, 1]` the batch and head strides of a `[B, H, D]` one.
fn vector_strides<const N: usize>(tensor: &CubeTensor, axes: [usize; N]) -> [u32; N] {
    let strides = tensor.meta.strides();
    axes.map(|axis| u32::try_from(strides[axis] / VECTOR).expect("strides fit in u32"))
}

/// The batch, head and time strides of a `[B, T, H, D]` input, in 16-byte vectors.
fn sequence_strides(tensor: &CubeTensor) -> [u32; 3] {
    vector_strides(tensor, [0, 2, 1])
}

/// An Int tensor as the kernels read it: `i32`, untiled.
fn int32(tensor: CubeTensor) -> CubeTensor {
    let tensor = if tensor.dtype == DType::I32 {
        tensor
    } else {
        CubeBackend::int_cast(tensor, IntDType::I32)
    };
    untile(tensor)
}

/// A one-element `i32` tensor bound where a kernel's mask argument goes unread.
fn unread(like: &CubeTensor) -> CubeTensor {
    empty_device_contiguous_dtype(
        like.client.clone(),
        like.device.clone(),
        Shape::new([1]),
        DType::I32,
    )
}

/// The mask and the options as the training kernels take them.
struct Mask {
    /// `[B, Tk]` `i32`, nonzero where a key is left out, or unread.
    keys: CubeTensor,
    /// Its batch and key strides, in elements.
    key_strides: [u32; 2],
    /// The offsets of packed sequences, contiguous `i32` `[S + 1]`, or unread.
    cu_q: CubeTensor,
    cu_k: CubeTensor,
    /// `S`.
    seqs: u32,
    padded: bool,
    packed: bool,
    /// How far a query sees before and after its position, at most `Tq + Tk + 1` (unbounded).
    left: i32,
    right: i32,
}

impl Mask {
    fn new(
        mask: AttentionMaskPrimitive<CubeBackend>,
        options: &AttentionOptions,
        [_, tq, tk, ..]: [usize; 6],
        like: &CubeTensor,
    ) -> Self {
        let unbounded = tq + tk + 1;
        let (left, right) = options.reach();
        let mut out = Self {
            keys: unread(like),
            key_strides: [0, 0],
            cu_q: unread(like),
            cu_k: unread(like),
            seqs: 0,
            padded: false,
            packed: false,
            left: left.min(unbounded) as i32,
            right: right.min(unbounded) as i32,
        };
        match mask {
            AttentionMaskPrimitive::None => {}
            AttentionMaskPrimitive::KeyPadding(keys) => {
                let keys = untile(CubeBackend::bool_into_int(keys, IntDType::I32));
                let strides = keys.meta.strides();
                out.key_strides = [strides[0], strides[1]]
                    .map(|stride| u32::try_from(stride).expect("strides fit in u32"));
                out.keys = keys;
                out.padded = true;
            }
            AttentionMaskPrimitive::Packed {
                cu_seqlens_q,
                cu_seqlens_k,
            } => {
                out.seqs = (cu_seqlens_q.meta.shape()[0] - 1) as u32;
                out.cu_q = into_contiguous(int32(cu_seqlens_q));
                out.cu_k = into_contiguous(int32(cu_seqlens_k));
                out.packed = true;
            }
        }
        out
    }
}

fn launched() {
    probe::KERNEL_LAUNCHES.fetch_add(1, Ordering::SeqCst);
}

/// The forward: output `[B, Tq, Hq, D]` and log-sum-exp `[B, Tq, Hq]` f32, both contiguous.
fn forward(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    mask: &Mask,
    scale: f32,
) -> (CubeTensor, CubeTensor) {
    let (query, key, value) = (readable(query), readable(key), readable(value));
    let [b, tq, tk, hq, hkv, d] = dims(&query, &key);
    let client = query.client.clone();
    let device = query.device.clone();
    let output = empty_device_contiguous_dtype(
        client.clone(),
        device.clone(),
        Shape::new([b, tq, hq, d]),
        query.dtype,
    );
    let lse =
        empty_device_contiguous_dtype(client.clone(), device, Shape::new([b, tq, hq]), DType::F32);
    let [q_sb, q_sh, q_st] = sequence_strides(&query);
    let [k_sb, k_sh, k_st] = sequence_strides(&key);
    let [v_sb, v_sh, v_st] = sequence_strides(&value);
    let [m_sb, m_st] = mask.key_strides;
    let q_blocks = tq.div_ceil(FORWARD_BLOCK_Q as usize);
    let cube_count = CubeCount::Static(q_blocks as u32, hq as u32, b as u32);
    let dtype = dtype_to_storage_type(query.dtype);
    launched();
    // SAFETY: the shapes were checked against each other, and the kernel bounds every index it
    // forms by them: rows below Tq, keys below Tk (whatever the offsets say), heads and batch
    // elements from the cube count.
    unsafe {
        forward_kernel::launch_unchecked(
            &client,
            cube_count,
            CubeDim::new_1d(2 * FORWARD_BLOCK_Q),
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            lse.clone().into_tensor_arg(),
            mask.keys.clone().into_tensor_arg(),
            mask.cu_q.clone().into_tensor_arg(),
            mask.cu_k.clone().into_tensor_arg(),
            q_blocks as u32,
            tq as u32,
            tk as u32,
            (hq / hkv) as u32,
            q_sb,
            q_sh,
            q_st,
            k_sb,
            k_sh,
            k_st,
            v_sb,
            v_sh,
            v_st,
            m_sb,
            m_st,
            mask.seqs,
            mask.left,
            mask.right,
            scale,
            d as u32,
            FORWARD_BLOCK_Q,
            block_kv(d),
            mask.padded,
            mask.packed,
            dtype,
        );
    }
    (output, lse)
}

/// The log-sum-exp as the backward kernels read it: f32, contiguous `[B, Tq, Hq]`.
fn lse_readable(lse: CubeTensor) -> CubeTensor {
    if lse.dtype == DType::F32 && !lse.meta.is_tiled() && lse.is_contiguous() {
        return lse;
    }
    probe::INPUT_COPIES.fetch_add(1, Ordering::SeqCst);
    let lse = if lse.dtype == DType::F32 {
        lse
    } else {
        CubeBackend::float_cast(lse, FloatDType::F32)
    };
    into_contiguous(lse)
}

/// A one-element f32 tensor bound where a kernel's accumulator goes unused.
fn unused_f32(like: &CubeTensor) -> CubeTensor {
    empty_device_contiguous_dtype(
        like.client.clone(),
        like.device.clone(),
        Shape::new([1]),
        DType::F32,
    )
}

/// The backward, each gradient contiguous in its input's layout: `delta` (zeroing the f32
/// accumulators), the five products (`dQ` added to its accumulator with atomics, and, split, `dK`
/// and `dV` to theirs), and the gradients rounded from the accumulators.
#[allow(clippy::too_many_arguments)]
fn backward(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    output: CubeTensor,
    lse: CubeTensor,
    grad_output: CubeTensor,
    mask: &Mask,
    scale: f32,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    let (query, key, value) = (readable(query), readable(key), readable(value));
    let (output, grad_output) = (readable(output), readable(grad_output));
    let lse = lse_readable(lse);
    let [b, tq, tk, hq, hkv, d] = dims(&query, &key);
    let client = query.client.clone();
    let device = query.device.clone();
    let dtype = dtype_to_storage_type(query.dtype);
    let [q_sb, q_sh, q_st] = sequence_strides(&query);
    let [k_sb, k_sh, k_st] = sequence_strides(&key);
    let [v_sb, v_sh, v_st] = sequence_strides(&value);
    let [o_sb, o_sh, o_st] = sequence_strides(&output);
    let [do_sb, do_sh, do_st] = sequence_strides(&grad_output);
    let [m_sb, m_st] = mask.key_strides;
    let group = (hq / hkv) as u32;
    let contiguous = |shape: Vec<usize>, dtype: DType| {
        empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::from(shape), dtype)
    };

    let rows = b * tq * hq;
    let delta = contiguous(vec![b, tq, hq], DType::F32);
    // The log-sum-exp as the other kernels read it: `+inf` for a query that sees no key.
    let lse_bwd = contiguous(vec![b, tq, hq], DType::F32);
    // `dQ` in f32, summed over the key blocks by the products' atomics: a tile of 32 x D values per
    // 32-query block and query head, in the products' register order (`dq_tile()`).
    let split = splits(&client, [b, tq, tk, hq, hkv, d]);
    if split {
        probe::SPLIT_RUNS.fetch_add(1, Ordering::SeqCst);
    }
    let q_blocks = tq.div_ceil(BLOCK_KV as usize);
    let zero_rows = b * q_blocks * BLOCK_KV as usize * hq;
    let dq_acc = contiguous(vec![b, q_blocks, hq, BLOCK_KV as usize * d], DType::F32);
    // Split, `dK` and `dV` in f32, each the sum of the two halves of its key block's sweep.
    let kv_chunks = if split { b * tk * hkv * d / VECTOR } else { 0 };
    let (dk_acc, dv_acc) = if split {
        let shape = vec![b, tk, hkv, d];
        (
            contiguous(shape.clone(), DType::F32),
            contiguous(shape, DType::F32),
        )
    } else {
        (unused_f32(&query), unused_f32(&query))
    };
    // Packed, each query's interval of keys, for the products.
    let bounds = if mask.packed {
        contiguous(vec![tq, 2], DType::I32)
    } else {
        unread(&query)
    };
    let rows_per_cube = 256 / (d / VECTOR);
    launched();
    // SAFETY: as for the forward; `delta` reads the output and its gradient row by row, and
    // zeroes the accumulators, whose sizes bound the units that write them.
    unsafe {
        delta_kernel::launch_unchecked(
            &client,
            CubeCount::Static(
                zero_rows
                    .div_ceil(rows_per_cube)
                    .max(kv_chunks.div_ceil(256)) as u32,
                1,
                1,
            ),
            CubeDim::new_1d(256),
            output.into_tensor_arg(),
            grad_output.clone().into_tensor_arg(),
            delta.clone().into_tensor_arg(),
            lse.into_tensor_arg(),
            lse_bwd.clone().into_tensor_arg(),
            dq_acc.clone().into_tensor_arg(),
            dk_acc.clone().into_tensor_arg(),
            dv_acc.clone().into_tensor_arg(),
            bounds.clone().into_tensor_arg(),
            mask.cu_q.clone().into_tensor_arg(),
            mask.cu_k.clone().into_tensor_arg(),
            rows as u32,
            zero_rows as u32,
            kv_chunks as u32,
            hq as u32,
            tq as u32,
            tk as u32,
            o_sb,
            o_sh,
            o_st,
            do_sb,
            do_sh,
            do_st,
            mask.seqs,
            mask.left,
            mask.right,
            d as u32,
            mask.packed,
            dtype,
        );
    }

    let grad_query = contiguous(vec![b, tq, hq, d], query.dtype);
    let grad_key = contiguous(vec![b, tk, hkv, d], key.dtype);
    let grad_value = contiguous(vec![b, tk, hkv, d], value.dtype);
    launched();
    // SAFETY: as for the forward; the accumulator's rows are those of the queries, below Tq.
    unsafe {
        backward_kernel::launch_unchecked(
            &client,
            CubeCount::Static(
                (tk.div_ceil(BLOCK_KV as usize) * (1 + usize::from(split))) as u32,
                hkv as u32,
                b as u32,
            ),
            CubeDim::new_1d(256),
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            grad_output.into_tensor_arg(),
            lse_bwd.into_tensor_arg(),
            delta.into_tensor_arg(),
            dq_acc.clone().into_tensor_arg(),
            grad_key.clone().into_tensor_arg(),
            grad_value.clone().into_tensor_arg(),
            dk_acc.clone().into_tensor_arg(),
            dv_acc.clone().into_tensor_arg(),
            mask.keys.clone().into_tensor_arg(),
            bounds.into_tensor_arg(),
            mask.cu_q.clone().into_tensor_arg(),
            mask.cu_k.clone().into_tensor_arg(),
            tq as u32,
            tk as u32,
            group,
            q_sb,
            q_sh,
            q_st,
            k_sb,
            k_sh,
            k_st,
            v_sb,
            v_sh,
            v_st,
            do_sb,
            do_sh,
            do_st,
            m_sb,
            m_st,
            mask.seqs,
            mask.left,
            mask.right,
            scale,
            d as u32,
            mask.padded,
            mask.packed,
            split,
            dtype,
        );
    }

    let tiles = b * q_blocks * hq;
    launched();
    // SAFETY: one cube per tile of the `dQ` accumulator, rows past the queries not written; then,
    // split, one unit per eight elements of `dK` and `dV`.
    unsafe {
        finish_kernel::launch_unchecked(
            &client,
            CubeCount::Static((tiles + kv_chunks.div_ceil(256)) as u32, 1, 1),
            CubeDim::new_1d(256),
            dq_acc.into_tensor_arg(),
            grad_query.clone().into_tensor_arg(),
            dk_acc.into_tensor_arg(),
            dv_acc.into_tensor_arg(),
            grad_key.clone().into_tensor_arg(),
            grad_value.clone().into_tensor_arg(),
            q_blocks as u32,
            hq as u32,
            tiles as u32,
            tq as u32,
            kv_chunks as u32,
            scale,
            d as u32,
            split,
            dtype,
        );
    }
    (grad_query, grad_key, grad_value)
}

/// Decoding, `query` `[B, Hq, D]` against `key` and `value` `[B, S, Hkv, D]`, sequence `b`
/// holding `lengths[b]` keys (all `S` without lengths) and its query the last `left + 1` of
/// them: the cache split by [`decode_splits`], each split counted back from the sequence's last
/// key, then the splits merged.
fn decode(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    lengths: Option<CubeTensor>,
    left: usize,
    scale: f32,
) -> CubeTensor {
    let (query, key, value) = (readable(query), readable(key), readable(value));
    let [b, len, hkv, d] = key.meta.shape().dims::<4>();
    let hq = query.meta.shape()[1];
    let client = query.client.clone();
    let device = query.device.clone();
    let dtype = dtype_to_storage_type(query.dtype);
    let group = hq / hkv;
    let (split_len, splits) = decode_splits(&client, [b, len, hq, hkv, d], left);
    let has_lengths = lengths.is_some();
    let lengths = match lengths {
        Some(lengths) => int32(lengths),
        None => unread(&query),
    };
    let l_st = u32::try_from(lengths.meta.strides()[0]).expect("strides fit in u32");

    let part_o = empty_device_contiguous_dtype(
        client.clone(),
        device.clone(),
        Shape::new([b, hq, splits, d]),
        DType::F32,
    );
    let part_lse = empty_device_contiguous_dtype(
        client.clone(),
        device.clone(),
        Shape::new([b, hq, splits]),
        DType::F32,
    );
    let [q_sb, q_sh] = vector_strides(&query, [0, 1]);
    let [k_sb, k_sh, k_st] = sequence_strides(&key);
    let [v_sb, v_sh, v_st] = sequence_strides(&value);
    launched();
    // SAFETY: the shapes were checked against each other, and the kernel keeps every key it
    // reads below the cache's length, whatever the lengths say.
    unsafe {
        decode_split_kernel::launch_unchecked(
            &client,
            CubeCount::Static(splits as u32, hkv as u32, b as u32),
            CubeDim::new_1d(128),
            query.clone().into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            part_o.clone().into_tensor_arg(),
            part_lse.clone().into_tensor_arg(),
            lengths.into_tensor_arg(),
            len as u32,
            split_len as u32,
            q_sb,
            q_sh,
            k_sb,
            k_sh,
            k_st,
            v_sb,
            v_sh,
            v_st,
            l_st,
            left.min(len + 1) as i32,
            scale,
            d as u32,
            group as u32,
            has_lengths,
            dtype,
        );
    }

    let output =
        empty_device_contiguous_dtype(client.clone(), device, Shape::new([b, hq, d]), query.dtype);
    launched();
    // SAFETY: one cube per output row, one unit per element.
    unsafe {
        decode_merge_kernel::launch_unchecked(
            &client,
            CubeCount::Static((b * hq) as u32, 1, 1),
            CubeDim::new_1d(d as u32),
            part_o.into_tensor_arg(),
            part_lse.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            splits as u32,
            d as u32,
            dtype,
        );
    }
    output
}

/// Whether the kernels take this call's inputs: those of one dtype, on a device and at
/// dimensions they take ([`kernels_take`]), every tensor indexable with `u32`.
fn takes(inputs: &[&CubeTensor], dims: [usize; 6], backward: bool) -> bool {
    let dtype = inputs[0].dtype;
    inputs.iter().all(|input| input.dtype == dtype)
        && kernels_take(&inputs[0].client, dtype, dims, backward)
        && indexable(inputs)
}

/// The shape of a tensor as a slice, for the checks.
fn shape(tensor: &CubeTensor) -> Vec<usize> {
    tensor.meta.shape().as_slice().to_vec()
}

impl AttentionBackend for CubeBackend {
    fn attention_forward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: AttentionMaskPrimitive<Self>,
        options: AttentionOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // The kernels index without bounds checks: hold the shapes to each other first.
        let dims = check_shapes(
            &shape(&query),
            &shape(&key),
            &shape(&value),
            MaskShape::of(&mask),
        );
        if !takes(&[&query, &key, &value], dims, false) {
            return reference::forward::<Self>(query, key, value, mask, options);
        }
        let mask = Mask::new(mask, &options, dims, &query);
        forward(query, key, value, &mask, options.scale_for(dims[5]))
    }

    fn attention_backward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        output: FloatTensor<Self>,
        lse: FloatTensor<Self>,
        grad_output: FloatTensor<Self>,
        mask: AttentionMaskPrimitive<Self>,
        options: AttentionOptions,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let dims = check_shapes(
            &shape(&query),
            &shape(&key),
            &shape(&value),
            MaskShape::of(&mask),
        );
        let [b, tq, _, hq, _, _] = dims;
        assert_eq!(shape(&output), shape(&query), "attention_backward: output");
        assert_eq!(
            shape(&grad_output),
            shape(&query),
            "attention_backward: grad"
        );
        assert_eq!(shape(&lse), [b, tq, hq], "attention_backward: lse");
        let inputs = [&query, &key, &value, &output, &grad_output];
        if !takes(&inputs, dims, true) || !indexable(&[&lse]) {
            return reference::backward::<Self>(
                query,
                key,
                value,
                output,
                lse,
                grad_output,
                mask,
                options,
            );
        }
        let mask = Mask::new(mask, &options, dims, &query);
        backward(
            query,
            key,
            value,
            output,
            lse,
            grad_output,
            &mask,
            options.scale_for(dims[5]),
        )
    }

    fn attention_decode(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        lengths: CacheLengths<Self>,
        options: AttentionOptions,
    ) -> FloatTensor<Self> {
        let lengths_shape = match &lengths {
            CacheLengths::Full => None,
            CacheLengths::Lengths(lengths) => Some(shape(lengths)),
        };
        let dims = check_decode_shapes(
            &shape(&query),
            &shape(&key),
            &shape(&value),
            lengths_shape.as_deref(),
        );
        let dtype = query.dtype;
        let inputs = [&query, &key, &value];
        let takes = inputs.iter().all(|input| input.dtype == dtype)
            && decode_kernels_take(&query.client, dtype, dims)
            && indexable(&inputs);
        if !takes {
            return reference::decode::<Self>(query, key, value, lengths, options);
        }
        let lengths = match lengths {
            CacheLengths::Full => None,
            CacheLengths::Lengths(lengths) => Some(lengths),
        };
        let left = options.reach().0;
        decode(query, key, value, lengths, left, options.scale_for(dims[4]))
    }
}
