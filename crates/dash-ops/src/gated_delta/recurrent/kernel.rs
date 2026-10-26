//! The step as one CubeCL kernel.
//!
//! A step is memory-bound: it reads the `[Dk, Dv]` state of every batch row and head once and
//! writes it once, about 4 FLOPs per element, and at decoding sizes (batch 8, 16 heads,
//! Dk = Dv = 128) the state is 8 MiB. One cube covers one batch row, one head and 32 columns of
//! the state: unit `(x, y)` holds column `x` of rows `y * R .. (y + 1) * R` in registers, with
//! `R = ceil(Dk / 8)`, so the state is read with one coalesced row segment per plane and every
//! unit reads and writes only its own elements, which lets the new state overwrite the old.
//! The two sums over Dk (`S^T k`, `S^T q`) and the l2 norms of `q` and `k` go through shared
//! memory, not plane operations, so the kernel does not depend on the plane width.

use super::{GatedDeltaStepBackend, check_inputs, probe, reference};
use burn::backend::TensorMetadata;
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::tensor::FloatTensor;
use burn::tensor::Shape;
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::{into_contiguous, untile};
use burn_cubecl::ops::numeric::empty_device_dtype;
use cubecl::prelude::*;
use std::sync::atomic::Ordering;

/// Columns of the state a cube covers, one per unit along x.
const COLS: usize = 32;
/// Groups of rows a cube splits Dk into, one per unit along y.
const ROW_GROUPS: usize = 8;
/// The largest Dk the kernel takes: each unit keeps `3 * ceil(Dk / 8)` values in registers.
/// Larger heads run the reference.
const MAX_DK: usize = 256;

#[cube(launch_unchecked, address_type = "dynamic")]
fn gated_delta_step_kernel<Q: Float, G: Float, Bt: Float>(
    q: &Tensor<Q>,
    k: &Tensor<Q>,
    v: &Tensor<Q>,
    g: &Tensor<G>,
    beta: &Tensor<Bt>,
    state: &Tensor<f32>,
    out: &mut Tensor<Q>,
    new_state: &mut Tensor<f32>,
    scale: f32,
    #[comptime] dk: usize,
    #[comptime] cols: usize,
    #[comptime] row_groups: usize,
    #[comptime] l2norm: bool,
    #[define(Q, G, Bt)] _dtypes: [ElemType; 3],
) {
    let units = comptime!(cols * row_groups);
    let rows = comptime!(dk.div_ceil(row_groups));

    let x = UNIT_POS_X as usize;
    let y = UNIT_POS_Y as usize;
    let unit = y * cols + x;
    let heads = state.shape(1);
    let dv = state.shape(3);
    // The column tiles of one head are neighbours in launch order, so that the cubes reading
    // one head's rows run together.
    let tiles = dv.div_ceil(cols);
    let cube = CUBE_POS_X as usize;
    let bh = cube / tiles;
    let b = bh / heads;
    let h = bh % heads;
    let col = (cube % tiles) * cols + x;
    let col_in = col < dv;
    let row0 = y * rows;

    // This unit's column of the state, rows row0..row0 + rows. The state is contiguous.
    let state_base = bh * dk * dv;
    let mut s = Array::<f32>::new(rows);
    #[unroll]
    for i in 0..rows {
        let r = row0 + i;
        let mut value = 0.0f32;
        if col_in && r < dk {
            value = state[state_base + r * dv + col];
        }
        s[i] = value;
    }

    // q and k of this row and head, in f32, into shared memory.
    let mut q_shared = Shared::<[f32]>::new_slice(dk);
    let mut k_shared = Shared::<[f32]>::new_slice(dk);
    let q_base = b * q.stride(0) + h * q.stride(1);
    let k_base = b * k.stride(0) + h * k.stride(1);
    let mut i = unit;
    while i < dk {
        q_shared[i] = f32::cast_from(q[q_base + i * q.stride(2)]);
        k_shared[i] = f32::cast_from(k[k_base + i * k.stride(2)]);
        i += units;
    }

    let mut q_inv_norm = 1.0f32;
    let mut k_inv_norm = 1.0f32;
    if l2norm {
        // Sum of squares: each unit over its own elements, then each lane over the row groups,
        // then every unit over the lanes, in one order, so that all units agree.
        let mut q_squares = Shared::<[f32]>::new_slice(units);
        let mut k_squares = Shared::<[f32]>::new_slice(units);
        let mut q_part = 0.0f32;
        let mut k_part = 0.0f32;
        let mut i = unit;
        while i < dk {
            let qi = q_shared[i];
            let ki = k_shared[i];
            q_part += qi * qi;
            k_part += ki * ki;
            i += units;
        }
        q_squares[unit] = q_part;
        k_squares[unit] = k_part;
        sync_cube();
        if y == 0 {
            let mut q_lane = 0.0f32;
            let mut k_lane = 0.0f32;
            #[unroll]
            for j in 0..row_groups {
                q_lane += q_squares[j * cols + x];
                k_lane += k_squares[j * cols + x];
            }
            q_squares[x] = q_lane;
            k_squares[x] = k_lane;
        }
        sync_cube();
        let mut q_sum = 0.0f32;
        let mut k_sum = 0.0f32;
        #[unroll]
        for j in 0..cols {
            q_sum += q_squares[j];
            k_sum += k_squares[j];
        }
        // 1e-6: L2NORM_EPS, the epsilon of transformers' and fla's l2norm.
        q_inv_norm = 1.0f32 / (q_sum + 1e-6f32).sqrt();
        k_inv_norm = 1.0f32 / (k_sum + 1e-6f32).sqrt();
    } else {
        sync_cube();
    }

    // This unit's rows of k and of the scaled q.
    let mut k_rows = Array::<f32>::new(rows);
    let mut q_rows = Array::<f32>::new(rows);
    #[unroll]
    for i in 0..rows {
        let r = row0 + i;
        let mut kr = 0.0f32;
        let mut qr = 0.0f32;
        if r < dk {
            kr = k_shared[r] * k_inv_norm;
            qr = q_shared[r] * q_inv_norm * scale;
        }
        k_rows[i] = kr;
        q_rows[i] = qr;
    }

    let decay = f32::cast_from(g[b * g.stride(0) + h * g.stride(1)]).exp();
    let beta_bh = f32::cast_from(beta[b * beta.stride(0) + h * beta.stride(1)]);
    let mut value = 0.0f32;
    if col_in {
        value = f32::cast_from(v[b * v.stride(0) + h * v.stride(1) + col * v.stride(2)]);
    }

    // Decay the state, and (S^T k)[col] over the row groups.
    let mut kv_parts = Shared::<[f32]>::new_slice(units);
    let mut kv_part = 0.0f32;
    #[unroll]
    for i in 0..rows {
        s[i] *= decay;
        kv_part += s[i] * k_rows[i];
    }
    kv_parts[unit] = kv_part;
    sync_cube();
    let mut kv_mem = 0.0f32;
    #[unroll]
    for j in 0..row_groups {
        kv_mem += kv_parts[j * cols + x];
    }
    let delta = (value - kv_mem) * beta_bh;

    // S += k delta^T, written back; (S^T q)[col] over the row groups.
    let mut out_parts = Shared::<[f32]>::new_slice(units);
    let mut out_part = 0.0f32;
    #[unroll]
    for i in 0..rows {
        let r = row0 + i;
        s[i] += k_rows[i] * delta;
        if col_in && r < dk {
            new_state[state_base + r * dv + col] = s[i];
        }
        out_part += s[i] * q_rows[i];
    }
    out_parts[unit] = out_part;
    sync_cube();
    if y == 0 && col_in {
        let mut o = 0.0f32;
        #[unroll]
        for j in 0..row_groups {
            o += out_parts[j * cols + x];
        }
        let index = b * out.stride(0) + h * out.stride(1) + col * out.stride(2);
        out[index] = Q::cast_from(o);
    }
}

impl GatedDeltaStepBackend for CubeBackend {
    fn gated_delta_step(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        state: FloatTensor<Self>,
        use_qk_l2norm: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // The kernel indexes without bounds checks: hold the shapes to each other first.
        let [b, h, dk, dv] = check_inputs(
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
        if b * h * dv == 0 || dk == 0 || dk > MAX_DK {
            return reference::<Self>(q, k, v, g, beta, state, use_qk_l2norm);
        }

        // Inputs are read through their strides; the state is indexed as contiguous.
        let (q, k, v, g, beta) = (untile(q), untile(k), untile(v), untile(g), untile(beta));
        let state = into_contiguous(state);
        let (client, device) = (q.client.clone(), q.device.clone());
        let out = empty_device_dtype(
            client.clone(),
            device.clone(),
            Shape::new([b, h, dv]),
            q.dtype,
        );
        let in_place = state.can_mut() && state.is_nonoverlapping();
        let new_state = if in_place {
            state.clone()
        } else {
            empty_device_dtype(client.clone(), device, state.shape(), state.dtype)
        };

        let address = [&q, &k, &v, &g, &beta, &state, &out, &new_state]
            .iter()
            .map(|tensor| tensor.required_address_type())
            .max()
            .unwrap_or_default();
        let dtypes = [q.dtype, g.dtype, beta.dtype].map(dtype_to_storage_type);
        let cube_count = CubeCount::Static((b * h * dv.div_ceil(COLS)) as u32, 1, 1);
        let cube_dim = CubeDim::new_2d(COLS as u32, ROW_GROUPS as u32);
        let scale = (1.0 / (dk as f64).sqrt()) as f32;
        // The state is the kernel's sixth buffer (q, k, v, g, beta, state): an in-place new
        // state aliases it.
        let new_state_arg = if in_place {
            state.as_tensor_alias(5)
        } else {
            new_state.clone().into_tensor_arg()
        };
        probe::KERNEL_LAUNCHES.fetch_add(1, Ordering::SeqCst);
        // SAFETY: the shapes were checked against each other above, so every index the kernel
        // forms is in bounds: rows below Dk, columns below Dv, batch rows and heads from the
        // cube count.
        unsafe {
            gated_delta_step_kernel::launch_unchecked(
                &client,
                cube_count,
                cube_dim,
                address,
                q.into_tensor_arg(),
                k.into_tensor_arg(),
                v.into_tensor_arg(),
                g.into_tensor_arg(),
                beta.into_tensor_arg(),
                state.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                new_state_arg,
                scale,
                dk,
                COLS,
                ROW_GROUPS,
                use_qk_l2norm,
                dtypes,
            );
        }
        (out, new_state)
    }
}
