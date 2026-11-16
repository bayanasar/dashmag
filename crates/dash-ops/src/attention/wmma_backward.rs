//! The backward on a GPU whose manual MMA is 16 x 16 x 16 from bf16 or f16 into f32 in RDNA4's
//! register layout (`crate::wmma`), where `kernels.rs`' backward (`mma.sync` m16n8k16 and
//! `ldmatrix`) does not run: the query gradient in a fixed order, and the key and value gradients,
//! after `kernels.rs`' `delta_kernel()` (`delta = rowsum(dO O)`, and the log-sum-exp as these read
//! it). The same sums as `kernels.rs`' fixed-order backward, unsplit and without atomics.
//!
//! As in `wmma.rs`, every product runs transposed, so that a lane holds one row of it: the
//! accumulator of `Y^T` holds, in lane `l`, row `l % 16` of `Y` and eight consecutive columns, and
//! is the B operand of a next product over those columns as it stands.
//!
//! - [`dq_wmma_kernel`], a lane per query, as the forward: each lane holds its query's row of
//!   `Q` and of `dO` in registers; per 32 keys, `S^T = K Q^T` and `dP^T = V dO^T` from the key and
//!   value tiles by rows, `P = exp(S scale - lse)` and `dS = P (dP - delta)` with the lane's own
//!   `lse` and `delta`, and `dQ^T += K^T dS^T` from the key tile transposed.
//! - [`dkdv_wmma_kernel`], a lane per key: each lane holds its key's row of `K` and of `V` in
//!   registers; per step of the queries that see the block, `P^T` and `dS^T` the same way from the
//!   query and output-gradient tiles by rows (with each query's `lse` and `delta` from shared
//!   memory), and `dV^T += dO^T P^T`, `dK^T += Q^T dS^T` from those tiles transposed.
//!
//! A wave's accumulator over the whole head dimension would not fit beside the rows it holds at
//! `D` 256 (and for `dK` and `dV` together at 128), so `ds` waves share each 16 rows, each
//! accumulating `D / ds` of the columns and each forming the scores and their gradient for the
//! rows itself.

// See `kernels.rs`: CubeCL's frontend takes the plain operators.
#![allow(
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::assign_op_pattern,
    clippy::collapsible_if
)]

use super::kernels::{V8, fetch_padding, load_rows, query_range, row_bounds, swizzle};
use super::wmma::{load_vt, vt_index};
use cubecl::cmma::MmaDefinition;
use cubecl::prelude::*;

/// One f32 accumulator element per vector, as the layout gives them.
type R1 = Const<1>;

/// Keys per tile of the query gradient's sweep.
pub(crate) const DQ_BLOCK_KV: u32 = 32;

/// The query gradient's shape for a head dimension `d`: 16-query groups per cube and waves per
/// group.
pub(crate) fn dq_shape(d: u32) -> (u32, u32) {
    if d > 128 { (2, 4) } else { (4, 1) }
}

/// The key and value gradients' shape for a head dimension `d`: 16-key groups per cube, waves
/// per group, and queries per step.
pub(crate) fn dkdv_shape(d: u32) -> (u32, u32, u32) {
    match d {
        0..=64 => (4, 1, 32),
        65..=128 => (4, 2, 32),
        _ => (2, 4, 16),
    }
}

/// `c += a b`, eight accumulator elements from `c[8 tile..]`.
#[cube]
fn mma<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    a: Vector<E, V8>,
    b: Vector<E, V8>,
    acc: &mut Array<Vector<f32, R1>>,
    tile: u32,
) {
    let mut av = Array::<Vector<E, V8>>::new(1usize);
    av[0] = a;
    let mut bv = Array::<Vector<E, V8>>::new(1usize);
    bv[0] = b;
    let mut c = Array::<Vector<f32, R1>>::new(8usize);
    #[unroll]
    for i in 0..8u32 {
        c[i as usize] = acc[(8 * tile + i) as usize];
    }
    let out = def.execute(&av, &bv, &c);
    #[unroll]
    for i in 0..8u32 {
        acc[(8 * tile + i) as usize] = out[i as usize];
    }
}

#[cube]
fn zeros(#[comptime] len: u32) -> Array<Vector<f32, R1>> {
    let mut acc = Array::<Vector<f32, R1>>::new(len as usize);
    #[unroll]
    for i in 0..len {
        acc[i as usize] = Vector::new(0.0f32);
    }
    acc
}

/// Accumulator tile `tile`, eight f32 of one row, in the input dtype, times `factor`.
#[cube]
fn tile_as<E: Float>(acc: &Array<Vector<f32, R1>>, tile: u32, factor: f32) -> Vector<E, V8> {
    let mut x = Vector::<E, V8>::new(E::from_int(0));
    #[unroll]
    for i in 0..8u32 {
        x.insert(
            i as usize,
            E::cast_from(acc[(8 * tile + i) as usize].extract(0usize) * factor),
        );
    }
    x
}

/// The lane's row `row` (below `limit`, else zeros) of one (batch, head) slice of `src`, `D / 16`
/// vectors of eight: the elements `16 kk + 8 (lane / 16)..+8` at `kk`, a B operand per step of 16
/// along `D`.
#[cube]
fn lane_row<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    base: usize,
    stride: u32,
    row: u32,
    limit: u32,
    #[comptime] d: u32,
) -> Array<Vector<E, V8>> {
    let half = UNIT_POS_PLANE / 16;
    let mut x = Array::<Vector<E, V8>>::new(comptime!((d / 16) as usize));
    #[unroll]
    for kk in 0..comptime!(d / 16) {
        let mut value = Vector::<E, V8>::new(E::from_int(0));
        if row < limit {
            value = src[base + row as usize * stride as usize + (kk * 2 + half) as usize];
        }
        x[kk as usize] = value;
    }
    x
}

/// The query gradient in a fixed order, the backward's default here: one cube per (`16 groups`
/// queries, query head, batch), query blocks taken last first; `ds` waves per 16 queries, a lane
/// per query, wave `w` accumulating columns `(w / groups) D / ds..` of `dQ`. Sweeps the key tiles
/// its queries see in order (the forward's span), so that the same inputs give the same bits on
/// every run. `lse` is the log-sum-exp as `delta_kernel()` leaves it (`+inf` for a query that
/// sees no key). Writes `dQ` contiguous `[B, Tq, Hq, D]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dq_wmma_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    d_o: &Tensor<Vector<E, V8>>,
    lse: &Tensor<f32>,
    delta: &Tensor<f32>,
    dq: &mut Tensor<Vector<E, V8>>,
    key_mask: &Tensor<i32>,
    cu_q: &Tensor<i32>,
    cu_k: &Tensor<i32>,
    q_blocks: u32,
    tq: u32,
    tk: u32,
    group: u32,
    q_sb: u32,
    q_sh: u32,
    q_st: u32,
    k_sb: u32,
    k_sh: u32,
    k_st: u32,
    v_sb: u32,
    v_sh: u32,
    v_st: u32,
    do_sb: u32,
    do_sh: u32,
    do_st: u32,
    m_sb: u32,
    m_st: u32,
    seqs: u32,
    left: i32,
    right: i32,
    scale: f32,
    #[comptime] d: u32,
    #[comptime] groups: u32,
    #[comptime] ds: u32,
    #[comptime] padded: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let block_q = comptime!(16 * groups);
    let block_kv = comptime!(DQ_BLOCK_KV);
    let threads = comptime!(32 * groups * ds);
    let chunks = comptime!(d / 8);
    let kgroups = comptime!(DQ_BLOCK_KV / 8);
    let cols = comptime!(d / 16 / ds);
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let wave = UNIT_POS / 32;
    let qg = wave % groups;
    let part = wave / groups;
    let lane = UNIT_POS_PLANE;
    let col = lane % 16;
    let half = lane / 16;

    let h = CUBE_POS_Y;
    let b = CUBE_POS_Z;
    let hq = CUBE_COUNT_Y;
    let hk = h / group;
    let q0 = (q_blocks - 1 - CUBE_POS_X) * block_q;

    let q_base = b as usize * q_sb as usize + h as usize * q_sh as usize;
    let do_base = b as usize * do_sb as usize + h as usize * do_sh as usize;
    let k_base = b as usize * k_sb as usize + hk as usize * k_sh as usize;
    let v_base = b as usize * v_sb as usize + hk as usize * v_sh as usize;
    let m_base = b as usize * m_sb as usize;

    // This lane's query and the keys it sees; the wave's and the cube's spans, as the forward's.
    let row = q0 + qg * 16 + col;
    let (lo, hi) = row_bounds(row, tq, tk, left, right, cu_q, cu_k, seqs, packed);
    let all_lo = plane_max(lo);
    let all_hi = plane_min(hi);
    let mut s_span = Shared::<[i32]>::new_slice(comptime!((2 * groups) as usize));
    let span_lo = plane_min(lo);
    let span_hi = plane_max(hi);
    if lane == 0 && part == 0 {
        s_span[(2 * qg) as usize] = span_lo;
        s_span[(2 * qg + 1) as usize] = span_hi;
    }
    sync_cube();
    let mut block_lo = tk as i32;
    let mut block_hi = 0i32;
    #[unroll]
    for w in 0..groups {
        block_lo = min(block_lo, s_span[(2 * w) as usize]);
        block_hi = max(block_hi, s_span[(2 * w + 1) as usize]);
    }
    let size = block_kv as i32;
    let first_tile = block_lo / size;
    let n_blocks = select(
        block_hi > block_lo,
        (block_hi + size - 1) / size - first_tile,
        0i32,
    ) as u32;
    let key0 = first_tile as u32 * block_kv;

    let q_frag = lane_row::<E>(q, q_base, q_st, row, tq, d);
    let do_frag = lane_row::<E>(d_o, do_base, do_st, row, tq, d);
    let mut l_row = f32::new(f32::INFINITY);
    let mut delta_row = 0.0f32;
    if row < tq {
        let at = (b as usize * tq as usize + row as usize) * hq as usize + h as usize;
        l_row = lse[at];
        delta_row = delta[at];
    }

    let tile = comptime!((DQ_BLOCK_KV * d / 8) as usize);
    let mut s_k = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let mut s_v = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let mut s_kt = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let mut s_pad =
        Shared::<[i32]>::new_slice(comptime!(if padded { DQ_BLOCK_KV } else { 1u32 }) as usize);

    let mut acc_dq = zeros(comptime!(8 * cols));

    for j in 0..n_blocks {
        let k0 = key0 + j * block_kv;
        sync_cube();
        load_rows::<E>(
            k,
            s_k.as_mut_slice(),
            k_base,
            k_st as usize,
            k0,
            tk,
            block_kv,
            chunks,
            threads,
        );
        load_rows::<E>(
            v,
            s_v.as_mut_slice(),
            v_base,
            v_st as usize,
            k0,
            tk,
            block_kv,
            chunks,
            threads,
        );
        load_vt::<E>(
            k,
            s_kt.as_mut_slice(),
            k_base,
            k_st as usize,
            k0,
            tk,
            block_kv,
            chunks,
            threads,
        );
        if comptime!(padded) {
            let flag = fetch_padding(key_mask, m_base, m_st, k0, tk, block_kv);
            if UNIT_POS < block_kv {
                s_pad[UNIT_POS as usize] = flag;
            }
        }
        sync_cube();

        let start = k0 as i32;
        let masked = padded || !(all_lo <= start && start + size <= all_hi);

        // S^T and dP^T, one 16 x 16 tile per 16 keys: eight keys of the lane's query per tile.
        let mut acc_s = zeros(comptime!(block_kv / 2));
        let mut acc_dp = zeros(comptime!(block_kv / 2));
        #[unroll]
        for mt in 0..comptime!(block_kv / 16) {
            #[unroll]
            for kk in 0..comptime!(d / 16) {
                let at = swizzle(mt * 16 + col, kk * 2 + half, chunks) as usize;
                mma::<E>(&def, s_k[at], q_frag[kk as usize], &mut acc_s, mt);
                mma::<E>(&def, s_v[at], do_frag[kk as usize], &mut acc_dp, mt);
            }
        }

        // P = exp(S scale - lse) and dS = P (dP - delta), the lane's query's.
        #[unroll]
        for i in 0..comptime!(block_kv / 2) {
            let mt = i / 8;
            let e = i % 8;
            let mut s = acc_s[i as usize].extract(0usize) * scale;
            if masked {
                let key = start + (mt * 16 + half * 8 + e) as i32;
                let mut visible = key >= lo && key < hi;
                if comptime!(padded) {
                    visible = visible && s_pad[(key - start) as usize] == 0i32;
                }
                s = select(visible, s, f32::NEG_INFINITY);
            }
            let p = (s - l_row).exp();
            let dsv = p * (acc_dp[i as usize].extract(0usize) - delta_row);
            acc_s[i as usize] = Vector::new(dsv);
        }

        // dQ^T += K^T dS^T, 16 keys at a time: dS^T's tile is B as the accumulator holds it.
        #[unroll]
        for kt in 0..comptime!(block_kv / 16) {
            let bds = tile_as::<E>(&acc_s, kt, 1.0f32);
            #[unroll]
            for c in 0..cols {
                let dt = part * cols + c;
                let a = s_kt[vt_index(dt * 16 + col, kt * 2 + half, kgroups) as usize];
                mma::<E>(&def, a, bds, &mut acc_dq, c);
            }
        }
    }

    if row < tq {
        let at = (b as usize * tq as usize + row as usize) * hq as usize + h as usize;
        #[unroll]
        for c in 0..cols {
            let dt = part * cols + c;
            dq[at * comptime!((d / 8) as usize) + (dt * 2 + half) as usize] =
                tile_as::<E>(&acc_dq, c, scale);
        }
    }
}

/// The key and value gradients: one cube per (`16 groups` keys, key-value head, batch); `ds`
/// waves per 16 keys, a lane per key, wave `w` accumulating columns `(w / groups) D / ds..` of `dK`
/// and `dV`. Sweeps the query heads of the group and, for each, the `sq`-query steps of the
/// queries that see the block ([`query_range()`]), in order. Writes `dK` (times `scale`) and `dV`
/// contiguous `[B, Tk, Hkv, D]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dkdv_wmma_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    d_o: &Tensor<Vector<E, V8>>,
    lse: &Tensor<f32>,
    delta: &Tensor<f32>,
    dk: &mut Tensor<Vector<E, V8>>,
    dv: &mut Tensor<Vector<E, V8>>,
    key_mask: &Tensor<i32>,
    cu_q: &Tensor<i32>,
    cu_k: &Tensor<i32>,
    tq: u32,
    tk: u32,
    group: u32,
    q_sb: u32,
    q_sh: u32,
    q_st: u32,
    k_sb: u32,
    k_sh: u32,
    k_st: u32,
    v_sb: u32,
    v_sh: u32,
    v_st: u32,
    do_sb: u32,
    do_sh: u32,
    do_st: u32,
    m_sb: u32,
    m_st: u32,
    seqs: u32,
    left: i32,
    right: i32,
    scale: f32,
    #[comptime] d: u32,
    #[comptime] groups: u32,
    #[comptime] ds: u32,
    #[comptime] sq: u32,
    #[comptime] padded: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let block_k = comptime!(16 * groups);
    let threads = comptime!(32 * groups * ds);
    let chunks = comptime!(d / 8);
    let qgroups = comptime!(sq / 8);
    let cols = comptime!(d / 16 / ds);
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let wave = UNIT_POS / 32;
    let kg = wave % groups;
    let part = wave / groups;
    let lane = UNIT_POS_PLANE;
    let col = lane % 16;
    let half = lane / 16;

    let hk = CUBE_POS_Y;
    let b = CUBE_POS_Z;
    let hkv = CUBE_COUNT_Y;
    let hq = hkv * group;
    let k0 = CUBE_POS_X * block_k;
    let k_base = b as usize * k_sb as usize + hk as usize * k_sh as usize;
    let v_base = b as usize * v_sb as usize + hk as usize * v_sh as usize;

    // This lane's key, its rows of K and V, and whether it is padding.
    let key = k0 + kg * 16 + col;
    let k_frag = lane_row::<E>(k, k_base, k_st, key, tk, d);
    let v_frag = lane_row::<E>(v, v_base, v_st, key, tk, d);
    let mut pad = false;
    if comptime!(padded) {
        if key < tk {
            pad = key_mask[b as usize * m_sb as usize + key as usize * m_st as usize] != 0i32;
        }
    }
    // The wave's keys, for whether a step needs masking.
    let wave_lo = (k0 + kg * 16) as i32;
    let wave_hi = wave_lo + 16;

    // The queries that see some key of the block, in `sq`-query steps.
    let k1 = min(k0 + block_k, tk);
    let (first_query, end_query) = query_range(
        k0 as i32, k1 as i32, tq, tk, left, right, cu_q, cu_k, seqs, packed,
    );
    let first_step = first_query as u32 / sq;
    let end_step = (end_query as u32 + sq - 1) / sq;
    let per_head = select(end_query > first_query, end_step - first_step, 0u32);
    let total = group * per_head;
    let stats = b as usize * tq as usize * hq as usize;

    let tile = comptime!((sq * d / 8) as usize);
    let mut s_q = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let mut s_do = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let mut s_qt = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let mut s_dot = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let mut s_lse = Shared::<[f32]>::new_slice(sq as usize);
    let mut s_delta = Shared::<[f32]>::new_slice(sq as usize);
    let mut s_lo = Shared::<[i32]>::new_slice(sq as usize);
    let mut s_hi = Shared::<[i32]>::new_slice(sq as usize);

    let mut acc_dk = zeros(comptime!(8 * cols));
    let mut acc_dv = zeros(comptime!(8 * cols));

    for it in 0..total {
        let h = hk * group + it / per_head;
        let q0 = (first_step + it % per_head) * sq;
        let q_base = b as usize * q_sb as usize + h as usize * q_sh as usize;
        let do_base = b as usize * do_sb as usize + h as usize * do_sh as usize;
        sync_cube();
        load_rows::<E>(
            q,
            s_q.as_mut_slice(),
            q_base,
            q_st as usize,
            q0,
            tq,
            sq,
            chunks,
            threads,
        );
        load_rows::<E>(
            d_o,
            s_do.as_mut_slice(),
            do_base,
            do_st as usize,
            q0,
            tq,
            sq,
            chunks,
            threads,
        );
        load_vt::<E>(
            q,
            s_qt.as_mut_slice(),
            q_base,
            q_st as usize,
            q0,
            tq,
            sq,
            chunks,
            threads,
        );
        load_vt::<E>(
            d_o,
            s_dot.as_mut_slice(),
            do_base,
            do_st as usize,
            q0,
            tq,
            sq,
            chunks,
            threads,
        );
        if UNIT_POS < sq {
            let row = q0 + UNIT_POS;
            let mut l = f32::new(f32::INFINITY);
            let mut dl = 0.0f32;
            if row < tq {
                let at = stats + row as usize * hq as usize + h as usize;
                l = lse[at];
                dl = delta[at];
            }
            let (lo, hi) = row_bounds(row, tq, tk, left, right, cu_q, cu_k, seqs, packed);
            s_lse[UNIT_POS as usize] = l;
            s_delta[UNIT_POS as usize] = dl;
            s_lo[UNIT_POS as usize] = lo;
            s_hi[UNIT_POS as usize] = hi;
        }
        sync_cube();

        // Unless every query of the step sees each of the wave's keys (the intervals never move
        // back), and none of them is padding.
        let masked = padded || !(s_lo[(sq - 1) as usize] <= wave_lo && wave_hi <= s_hi[0usize]);

        // S^T = K Q^T and dP^T = V dO^T: the lane's key against eight queries per tile.
        let mut acc_s = zeros(comptime!(sq / 2));
        let mut acc_dp = zeros(comptime!(sq / 2));
        #[unroll]
        for qt in 0..comptime!(sq / 16) {
            #[unroll]
            for kk in 0..comptime!(d / 16) {
                let at = swizzle(qt * 16 + col, kk * 2 + half, chunks) as usize;
                mma::<E>(&def, s_q[at], k_frag[kk as usize], &mut acc_s, qt);
                mma::<E>(&def, s_do[at], v_frag[kk as usize], &mut acc_dp, qt);
            }
        }

        // P^T = exp(S^T scale - lse) and dS^T = P^T (dP^T - delta), each query's own.
        let kv = key as i32;
        #[unroll]
        for i in 0..comptime!(sq / 2) {
            let qt = i / 8;
            let e = i % 8;
            let r = (qt * 16 + half * 8 + e) as usize;
            let mut s = acc_s[i as usize].extract(0usize) * scale;
            if masked {
                let visible = kv >= s_lo[r] && kv < s_hi[r] && !pad;
                s = select(visible, s, f32::NEG_INFINITY);
            }
            let p = (s - s_lse[r]).exp();
            acc_s[i as usize] = Vector::new(p);
            acc_dp[i as usize] = Vector::new(p * (acc_dp[i as usize].extract(0usize) - s_delta[r]));
        }

        // dV^T += dO^T P^T and dK^T += Q^T dS^T, 16 queries at a time.
        #[unroll]
        for qt in 0..comptime!(sq / 16) {
            let bp = tile_as::<E>(&acc_s, qt, 1.0f32);
            let bds = tile_as::<E>(&acc_dp, qt, 1.0f32);
            #[unroll]
            for c in 0..cols {
                let dt = part * cols + c;
                let at = vt_index(dt * 16 + col, qt * 2 + half, qgroups) as usize;
                mma::<E>(&def, s_dot[at], bp, &mut acc_dv, c);
                mma::<E>(&def, s_qt[at], bds, &mut acc_dk, c);
            }
        }
    }

    if key < tk {
        let at = (b as usize * tk as usize + key as usize) * hkv as usize + hk as usize;
        #[unroll]
        for c in 0..cols {
            let dt = part * cols + c;
            let slot = at * comptime!((d / 8) as usize) + (dt * 2 + half) as usize;
            dk[slot] = tile_as::<E>(&acc_dk, c, scale);
            dv[slot] = tile_as::<E>(&acc_dv, c, 1.0f32);
        }
    }
}
