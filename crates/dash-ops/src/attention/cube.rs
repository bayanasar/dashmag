//! The CUDA backend: launches the kernels of `kernels.rs` for the inputs they take (bf16 or
//! f16, head dimension 64, 128 or 256, a device with `mma.sync` and `ldmatrix` for that dtype)
//! and runs the reference for the rest.
//!
//! Inputs are read through their strides. Every tensor the kernels write is allocated
//! contiguous: CubeCL's default allocation pads rows to the device's pitch, and the kernels
//! index their outputs and the per-row statistics densely.

use super::kernels::{
    BLOCK_KV, BLOCK_Q, FORWARD_BLOCK_Q, decode_merge_kernel, decode_split_kernel, delta_kernel,
    dkdv_kernel, dq_kernel, forward_kernel,
};
use super::{CausalAttentionBackend, KERNEL_LAUNCHES, reference};
use burn::backend::cubecl::{dtype_to_elem_type, dtype_to_storage_type};
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::FloatTensor;
use burn::backend::{DType, FloatDType, Shape};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::ir::features::MmaConfig;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use std::sync::atomic::Ordering;

/// Elements per 16-byte vector: the kernels' unit of every load, so strides count in it.
const VECTOR: usize = 8;

/// The key block of the forward and `dQ` kernels for head dimension `d`: 32 keys at 256, so that
/// queries (128 rows in the forward, 64 with the output gradients in `dQ`), keys and values fit
/// in 99 KB of shared memory; 64 below.
fn block_kv(d: usize) -> u32 {
    if d > 128 { 32 } else { 64 }
}

/// Bytes of shared memory the `dQ` kernel, the largest, takes at head dimension `d`.
fn shared_bytes(d: usize) -> usize {
    (2 * BLOCK_Q as usize + 2 * block_kv(d) as usize) * d * 2
}

/// Whether the kernels take these inputs.
fn supported(query: &CubeTensor, key: &CubeTensor, value: &CubeTensor) -> bool {
    let dtype = query.dtype;
    if !matches!(dtype, DType::BF16 | DType::F16) || key.dtype != dtype || value.dtype != dtype {
        return false;
    }
    let d = query.meta.shape()[3];
    if !matches!(d, 64 | 128 | 256) {
        return false;
    }
    // Empty inputs launch nothing: the reference returns their empty results.
    if query.meta.shape().num_elements() == 0 || key.meta.shape().num_elements() == 0 {
        return false;
    }
    let elem = dtype_to_elem_type(dtype);
    let properties = query.client.properties();
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
        && properties.hardware.max_shared_memory_size >= shared_bytes(d)
}

/// The input as the kernels read it: rows (not a matmul's storage tiles), last dimension
/// contiguous, every stride a whole number of 16-byte vectors, the first element 16-byte
/// aligned; copied to a contiguous tensor otherwise.
fn readable(tensor: CubeTensor) -> CubeTensor {
    let strides = tensor.meta.strides();
    let rank = strides.len();
    let aligned_offset = tensor.handle.offset_start.unwrap_or(0).is_multiple_of(16);
    let ok = !tensor.meta.is_tiled()
        && strides[rank - 1] == 1
        && strides[..rank - 1]
            .iter()
            .all(|stride| stride % VECTOR == 0)
        && aligned_offset;
    if ok { tensor } else { into_contiguous(tensor) }
}

/// The batch, head and row strides of a `[B, H, T, D]` input, in 16-byte vectors.
fn vector_strides(tensor: &CubeTensor) -> [u32; 3] {
    let strides = tensor.meta.strides();
    [0, 1, 2].map(|i| u32::try_from(strides[i] / VECTOR).expect("strides fit in u32"))
}

fn dims(tensor: &CubeTensor) -> [usize; 4] {
    let shape = tensor.meta.shape();
    [shape[0], shape[1], shape[2], shape[3]]
}

fn launched() {
    KERNEL_LAUNCHES.fetch_add(1, Ordering::SeqCst);
}

/// The forward: output `[B, Hq, Tq, D]` and log-sum-exp `[B, Hq, Tq]` f32, both contiguous.
fn forward(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    scale: f32,
) -> (CubeTensor, CubeTensor) {
    let (query, key, value) = (readable(query), readable(key), readable(value));
    let [b, hq, tq, d] = dims(&query);
    let [_, hkv, tk, _] = dims(&key);
    let client = query.client.clone();
    let device = query.device.clone();
    let output = empty_device_contiguous_dtype(
        client.clone(),
        device.clone(),
        Shape::new([b, hq, tq, d]),
        query.dtype,
    );
    let lse =
        empty_device_contiguous_dtype(client.clone(), device, Shape::new([b, hq, tq]), DType::F32);
    let [q_sb, q_sh, q_st] = vector_strides(&query);
    let [k_sb, k_sh, k_st] = vector_strides(&key);
    let [v_sb, v_sh, v_st] = vector_strides(&value);
    let q_blocks = tq.div_ceil(FORWARD_BLOCK_Q as usize);
    let cube_count = CubeCount::Static(q_blocks as u32, hq as u32, b as u32);
    let dtype = dtype_to_storage_type(query.dtype);
    launched();
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
            scale,
            d as u32,
            FORWARD_BLOCK_Q,
            block_kv(d),
            dtype,
        );
    }
    (output, lse)
}

/// The backward: `delta`, then `dQ`, then `dK` and `dV`.
#[allow(clippy::too_many_arguments)]
fn backward(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    output: CubeTensor,
    lse: CubeTensor,
    grad_output: CubeTensor,
    scale: f32,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    let (query, key, value) = (readable(query), readable(key), readable(value));
    let (output, grad_output) = (readable(output), readable(grad_output));
    let lse = into_contiguous(if lse.dtype == DType::F32 {
        lse
    } else {
        CubeBackend::float_cast(lse, FloatDType::F32)
    });
    let [b, hq, tq, d] = dims(&query);
    let [_, hkv, tk, _] = dims(&key);
    let client = query.client.clone();
    let device = query.device.clone();
    let dtype = dtype_to_storage_type(query.dtype);
    let [q_sb, q_sh, q_st] = vector_strides(&query);
    let [k_sb, k_sh, k_st] = vector_strides(&key);
    let [v_sb, v_sh, v_st] = vector_strides(&value);
    let [o_sb, o_sh, o_st] = vector_strides(&output);
    let [do_sb, do_sh, do_st] = vector_strides(&grad_output);
    let group = (hq / hkv) as u32;

    let rows = b * hq * tq;
    let delta = empty_device_contiguous_dtype(
        client.clone(),
        device.clone(),
        Shape::new([b, hq, tq]),
        DType::F32,
    );
    let rows_per_cube = 256 / (d / VECTOR);
    launched();
    unsafe {
        delta_kernel::launch_unchecked(
            &client,
            CubeCount::Static(rows.div_ceil(rows_per_cube) as u32, 1, 1),
            CubeDim::new_1d(256),
            output.into_tensor_arg(),
            grad_output.clone().into_tensor_arg(),
            delta.clone().into_tensor_arg(),
            rows as u32,
            hq as u32,
            tq as u32,
            o_sb,
            o_sh,
            o_st,
            do_sb,
            do_sh,
            do_st,
            d as u32,
            dtype,
        );
    }

    let grad_query = empty_device_contiguous_dtype(
        client.clone(),
        device.clone(),
        Shape::new([b, hq, tq, d]),
        query.dtype,
    );
    launched();
    unsafe {
        dq_kernel::launch_unchecked(
            &client,
            CubeCount::Static(tq.div_ceil(BLOCK_Q as usize) as u32, hq as u32, b as u32),
            CubeDim::new_1d(256),
            query.clone().into_tensor_arg(),
            key.clone().into_tensor_arg(),
            value.clone().into_tensor_arg(),
            grad_output.clone().into_tensor_arg(),
            lse.clone().into_tensor_arg(),
            delta.clone().into_tensor_arg(),
            grad_query.clone().into_tensor_arg(),
            tq.div_ceil(BLOCK_Q as usize) as u32,
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
            scale,
            d as u32,
            block_kv(d),
            dtype,
        );
    }

    let grad_key = empty_device_contiguous_dtype(
        client.clone(),
        device.clone(),
        Shape::new([b, hkv, tk, d]),
        key.dtype,
    );
    let grad_value = empty_device_contiguous_dtype(
        client.clone(),
        device,
        Shape::new([b, hkv, tk, d]),
        value.dtype,
    );
    launched();
    unsafe {
        dkdv_kernel::launch_unchecked(
            &client,
            CubeCount::Static(tk.div_ceil(BLOCK_KV as usize) as u32, hkv as u32, b as u32),
            CubeDim::new_1d(256),
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            grad_output.into_tensor_arg(),
            lse.into_tensor_arg(),
            delta.into_tensor_arg(),
            grad_key.clone().into_tensor_arg(),
            grad_value.clone().into_tensor_arg(),
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
            scale,
            d as u32,
            dtype,
        );
    }
    (grad_query, grad_key, grad_value)
}

/// Decoding: the cache split so that there are about eight cubes per multiprocessor, each split
/// a whole number of the first pass's key slots long.
fn decode(query: CubeTensor, key: CubeTensor, value: CubeTensor, scale: f32) -> CubeTensor {
    let (query, key, value) = (readable(query), readable(key), readable(value));
    let [b, hq, _, d] = dims(&query);
    let [_, hkv, len, _] = dims(&key);
    let client = query.client.clone();
    let device = query.device.clone();
    let dtype = dtype_to_storage_type(query.dtype);
    let group = hq / hkv;

    let slots = 4 * (32 / (d / VECTOR));
    let processors = client
        .properties()
        .hardware
        .num_streaming_multiprocessors
        .unwrap_or(64) as usize;
    let wanted = (8 * processors).div_ceil(b * hkv).max(1);
    let split_len = len.div_ceil(wanted).max(slots).next_multiple_of(slots);
    let splits = len.div_ceil(split_len);

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
    let [q_sb, q_sh, _] = vector_strides(&query);
    let [k_sb, k_sh, k_st] = vector_strides(&key);
    let [v_sb, v_sh, v_st] = vector_strides(&value);
    launched();
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
            scale,
            d as u32,
            group as u32,
            dtype,
        );
    }

    let output = empty_device_contiguous_dtype(
        client.clone(),
        device,
        Shape::new([b, hq, 1, d]),
        query.dtype,
    );
    launched();
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

impl CausalAttentionBackend for CubeBackend {
    fn causal_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        scale: f32,
    ) -> FloatTensor<Self> {
        Self::causal_attention_with_lse(query, key, value, scale).0
    }

    fn causal_attention_with_lse(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        if supported(&query, &key, &value) {
            forward(query, key, value, scale)
        } else {
            reference::forward::<Self>(query, key, value, scale)
        }
    }

    fn causal_attention_backward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        output: FloatTensor<Self>,
        lse: FloatTensor<Self>,
        grad_output: FloatTensor<Self>,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let same_dtype = output.dtype == query.dtype && grad_output.dtype == query.dtype;
        if same_dtype && supported(&query, &key, &value) {
            backward(query, key, value, output, lse, grad_output, scale)
        } else {
            reference::backward::<Self>(query, key, value, output, lse, grad_output, scale)
        }
    }

    fn attention_decode(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        scale: f32,
    ) -> FloatTensor<Self> {
        let group = query.meta.shape()[1] / key.meta.shape()[1];
        // The first pass keeps `group` query rows of eight f32 per lane in registers and their
        // partial sums in shared memory; past 16 heads per group the reference is the safer path.
        if supported(&query, &key, &value) && group <= 16 {
            decode(query, key, value, scale)
        } else {
            reference::forward::<Self>(query, key, value, scale).0
        }
    }
}
