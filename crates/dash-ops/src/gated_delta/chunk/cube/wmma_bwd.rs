//! The backward pass for bf16 and f16 inputs on AMD RDNA4's 16 x 16 x 16 matrix products, where
//! the kernels of `tc.rs` do not run, for `Dk` and `Dv` of 32, 64 or 128: what `tc.rs`' backward
//! computes, through the same per-chunk quantities, with the same intermediates in the input
//! dtype (`T`, `W`, `U`, the keys, the states and their gradients, `V'`, `dV'`, `dv_local`, `dV_b`,
//! `dKbg`, `qg`) and the f32 sums `tc.rs` keeps. The forward's prep (`wmma.rs`, its backward
//! form, which also writes the keys and `U` by rows, `T^T` and `W^T`) and state passing run again
//! first.
//!
//! Every product runs as in `wmma.rs`: `Y = X Z` as `Y^T = Z^T X^T`, both operands staged along
//! the inner dimension, the accumulator holding in lane `l` row `l % 16` of `Y` and eight
//! consecutive columns. Where a product's operand is the transpose of what an earlier one leaves,
//! the kernel stages it transposed (eight rows of eight read, swapped in registers, written), or
//! the earlier kernel writes both forms.
//!
//! Shared memory, at `Dk = Dv = 128`: `dv_local` 32.5 KiB, the state passing 48.75 KiB, the
//! products with the states 59.0 KiB (four regions, each holding several operands in turn), the
//! WY kernel 50.5 KiB, within the 64 KiB a workgroup has on RDNA4.

// Kernel code is expanded by CubeCL's frontend, which takes the plain operators.
#![allow(
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::too_many_arguments
)]

use super::tc::{V8, swizzle};
use super::wmma::{eight, frag, mma, one, tile_as, tile_f32, transpose8, zeros};
use super::{chunk_tokens, state_chunks};
use cubecl::cmma::MmaDefinition;
use cubecl::prelude::*;

/// Units of [`wmma_dvlocal_kernel`]: four waves.
pub(super) const FOUR_WAVES: u32 = 128;
/// Units of the other per-chunk kernels: eight waves.
pub(super) const EIGHT_WAVES: u32 = 256;

/// The factor a chunk's query (or key) row is scaled by, as `tc.rs`' `stage_tokens` forms it:
/// `scale / sqrt(sum x^2 + 1e-6)` with the l2 normalisation (`ss` the sum of the squares of the
/// 16-byte chunk `x`, summed here over the `chunks` consecutive lanes holding the row), `scale`
/// without.
#[cube]
fn row_factor<E: Float>(
    x: Vector<E, V8>,
    scale: f32,
    #[comptime] chunks: u32,
    #[comptime] l2norm: bool,
) -> f32 {
    let mut factor = scale;
    if comptime!(l2norm) {
        let f = Vector::<f32, V8>::cast_from(x);
        let sq = f * f;
        let mut ss = 0.0f32;
        #[unroll]
        for e in 0..8usize {
            ss += sq.extract(e);
        }
        #[unroll]
        for s in 0..comptime!(chunks.trailing_zeros()) {
            ss += plane_shuffle_xor(ss, 1u32 << s);
        }
        factor = scale / (ss + 1e-6f32).sqrt();
    }
    factor
}

/// A chunk's rows of head `h` of a `[B T, H, D]` tensor (zero past `count`) into the swizzled
/// tile `dst` (`D / 8` chunks to a row), each row times its [`row_factor`], which goes to
/// `factors[r]`; `units` units take part.
#[cube]
fn stage_scaled<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    factors: &mut [f32],
    first: u32,
    count: u32,
    h: u32,
    heads: u32,
    scale: f32,
    #[comptime] d: u32,
    #[comptime] l2norm: bool,
    #[comptime] units: u32,
) {
    let chunks = comptime!(d / 8);
    #[unroll]
    for i in 0..comptime!(64 * d / 8 / units) {
        let idx = i * units + UNIT_POS;
        let r = idx / chunks;
        let c = idx % chunks;
        let mut x = Vector::<E, V8>::new(E::from_int(0));
        if r < count {
            x = src[(((first + r) * heads + h) * chunks + c) as usize];
        }
        let factor = row_factor::<E>(x, scale, chunks, l2norm);
        dst[swizzle(r, c, chunks) as usize] =
            Vector::cast_from(Vector::<f32, V8>::cast_from(x) * Vector::new(factor));
        if c == 0 {
            factors[r as usize] = factor;
        }
    }
}

/// `count` 16-byte vectors from `src[base..]` into the swizzled tile `dst` of `chunks` chunks to
/// a row, `units` units taking part.
#[cube]
fn stage_copy<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    base: u32,
    #[comptime] total: u32,
    #[comptime] chunks: u32,
    #[comptime] units: u32,
) {
    #[unroll]
    for i in 0..comptime!((total + units - 1) / units) {
        let idx = i * units + UNIT_POS;
        if idx < total {
            dst[swizzle(idx / chunks, idx % chunks, chunks) as usize] = src[(base + idx) as usize];
        }
    }
}

/// Chunks `[c0, c0 + chunks)` of a chunk's rows of head `h` of a `[B T, H, D]` tensor (zero past
/// `count`), transposed into `dst`: row `8 c + e` holds column `8 (c0 + c) + e` of the 64 tokens,
/// eight chunks to a row. With `factors` 1 or 2, each element times `f1` of its token, rounded to
/// the input dtype; with 2, then times `f2`, rounded again; `units` units take part.
#[cube]
fn stage_tokens_t<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    f1: &[f32],
    f2: &[f32],
    first: u32,
    count: u32,
    h: u32,
    heads: u32,
    c0: u32,
    #[comptime] d: u32,
    #[comptime] chunks: u32,
    #[comptime] factors: u32,
    #[comptime] units: u32,
) {
    let row_chunks = comptime!(d / 8);
    let blocks = comptime!(8 * chunks);
    #[unroll]
    for i in 0..comptime!((blocks + units - 1) / units) {
        let idx = i * units + UNIT_POS;
        if idx < blocks {
            let c = idx % chunks;
            let grp = idx / chunks;
            let mut rows = Array::<Vector<E, V8>>::new(8usize);
            #[unroll]
            for j in 0..8u32 {
                let r = grp * 8 + j;
                let mut x = Vector::<E, V8>::new(E::from_int(0));
                if r < count {
                    x = src[(((first + r) * heads + h) * row_chunks + c0 + c) as usize];
                }
                rows[j as usize] = x;
            }
            let cols = transpose8::<E>(&rows, 0u32);
            #[unroll]
            for e in 0..8u32 {
                let mut col = cols[e as usize];
                if comptime!(factors >= 1) {
                    col = Vector::cast_from(Vector::<f32, V8>::cast_from(col) * eight(f1, grp * 8));
                }
                if comptime!(factors >= 2) {
                    col = Vector::cast_from(Vector::<f32, V8>::cast_from(col) * eight(f2, grp * 8));
                }
                dst[swizzle(c * 8 + e, grp, 8u32) as usize] = col;
            }
        }
    }
}

/// Backward, per chunk: `dv_local = P^T dO` with `P^T = (K Q^T) exp(G_i - G_j)` (`i >= j`),
/// written transposed (`[B, H, N, Dv, C]`, for the state passing's `dV'^T`), and each query row's
/// factor (with the l2 norm). Four waves: wave `w` forms `P^T`'s rows `16 w` (its columns from
/// `16 w` on), whose accumulators are the left operand of `P^T dO` as they stand, and
/// `dv_local^T`'s columns `16 w`. The queries' tile holds `dO^T` once `P^T` is formed.
#[cube(launch_unchecked)]
pub(super) fn wmma_dvlocal_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    kn: &Tensor<Vector<E, V8>>,
    gcum: &Tensor<f32>,
    d_out: &Tensor<Vector<E, V8>>,
    dvl_t: &mut Tensor<Vector<E, V8>>,
    q_factor: &mut Tensor<f32>,
    plan: &Tensor<u32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    scale: f32,
    #[comptime] dk: u32,
    #[comptime] dv: u32,
    #[comptime] l2norm: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let ck = comptime!(dk / 8);
    let cmax = comptime!(if dk > dv { dk / 8 } else { dv / 8 });
    let cube = CUBE_POS_X;
    let c = cube % n_chunks;
    let bh = cube / n_chunks;
    let b = bh / heads;
    let h = bh % heads;
    let (first, count) = chunk_tokens(plan, b as usize, c as usize, seq_len as usize, packed);
    if packed && count == 0usize {
        terminate!();
    }
    let first = first as u32;
    let count = count as u32;
    let tid = UNIT_POS as usize;
    let jt = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let row = lane % 16;
    let half = lane / 16;

    // The queries (normalised, scaled), then dO^T; the keys (normalised).
    let mut s_q =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * cmax) as usize), 16usize);
    let mut s_k =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * ck) as usize), 16usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut fs = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        gs[tid] = gcum[cube as usize * 64usize + tid];
    }
    stage_scaled::<E>(
        q,
        s_q.as_mut_slice(),
        fs.as_mut_slice(),
        first,
        count,
        h,
        heads,
        scale,
        dk,
        l2norm,
        FOUR_WAVES,
    );
    stage_copy::<E>(
        kn,
        s_k.as_mut_slice(),
        cube * 64 * ck,
        comptime!(64 * ck),
        ck,
        FOUR_WAVES,
    );
    sync_cube();
    if comptime!(l2norm) {
        if tid < 64usize {
            q_factor[cube as usize * 64usize + tid] = fs[tid];
        }
    }

    // P^T = (K Q^T) exp(G_i - G_j), i >= j: the wave's 16 rows j, the columns i from 16 w on.
    let mut acc_p = zeros(4u32);
    #[unroll]
    for kk in 0..comptime!(dk / 16) {
        let x = frag::<E>(s_k.as_slice(), 16 * jt, kk, ck);
        #[unroll]
        for it in 0..4u32 {
            if it >= jt {
                let a = frag::<E>(s_q.as_slice(), 16 * it, kk, ck);
                mma::<E>(&def, &a, &x, &mut acc_p, it);
            }
        }
    }
    let j = 16 * jt + row;
    let gj = gs[j as usize];
    #[unroll]
    for it in 0..4u32 {
        #[unroll]
        for e in 0..8u32 {
            let i = 16 * it + 8 * half + e;
            let at = (8 * it + e) as usize;
            let p = acc_p[at].extract(0usize);
            acc_p[at] = Vector::new(select(i >= j, p * (gs[i as usize] - gj).exp(), 0.0f32));
        }
    }
    // dO^T where the queries were.
    sync_cube();
    stage_tokens_t::<E>(
        d_out,
        s_q.as_mut_slice(),
        gs.as_slice(),
        gs.as_slice(),
        first,
        count,
        h,
        heads,
        0u32,
        dv,
        comptime!(dv / 8),
        0u32,
        FOUR_WAVES,
    );
    sync_cube();

    // dv_local^T = dO^T P: P^T's tiles as the left operands (their k the queries); the steps
    // left of column 16 w are zero.
    #[unroll]
    for vt in 0..comptime!(dv / 16) {
        let mut acc = zeros(1u32);
        #[unroll]
        for it in 0..4u32 {
            if it >= jt {
                let a = one::<E>(tile_as::<E>(&acc_p, it));
                let x = frag::<E>(s_q.as_slice(), 16 * vt, it, 8u32);
                mma::<E>(&def, &a, &x, &mut acc, 0u32);
            }
        }
        dvl_t[((cube * dv + 16 * vt + row) * 8 + 2 * jt + half) as usize] =
            tile_as::<E>(&acc, 0u32);
    }
}

/// Chunk `c` of row `r` of a chunk's tokens from `first` (zero from `count` on) of head `h` of a
/// `[B T, H, D]` tensor of `chunks` chunks to a row, read whether or not `r` is in range (from
/// token 0 if not), so that a unit's reads are not each behind a branch: on RDNA4 a read behind
/// one is waited for before the next is issued.
#[cube]
fn token_chunk<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    first: u32,
    r: u32,
    count: u32,
    h: u32,
    heads: u32,
    c: u32,
    #[comptime] chunks: u32,
) -> Vector<E, V8> {
    let ok = r < count;
    let t = select(ok, first + r, 0u32);
    let x = src[((t * heads + h) * chunks + c) as usize];
    select(ok, x, Vector::new(E::from_int(0)))
}

/// Eight rows of one 16-byte chunk of a chunk's tokens ([`token_chunk`]): block `idx` of a
/// transposed tile of `chunks` chunks (its rows `8 (idx / chunks)` on, chunk `c0 + idx % chunks`).
#[cube]
fn token_block<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    rows: &mut Array<Vector<E, V8>>,
    first: u32,
    count: u32,
    h: u32,
    heads: u32,
    c0: u32,
    idx: u32,
    #[comptime] d: u32,
    #[comptime] chunks: u32,
) {
    let c = idx % chunks;
    let grp = idx / chunks;
    #[unroll]
    for j in 0..8u32 {
        rows[j as usize] = token_chunk::<E>(
            src,
            first,
            grp * 8 + j,
            count,
            h,
            heads,
            c0 + c,
            comptime!(d / 8),
        );
    }
}

/// Writes a [`token_block`] transposed into `dst`, as [`stage_tokens_t`] does.
#[cube]
fn token_block_t<E: Float>(
    rows: &Array<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    f1: &[f32],
    f2: &[f32],
    idx: u32,
    #[comptime] chunks: u32,
    #[comptime] factors: u32,
) {
    let c = idx % chunks;
    let grp = idx / chunks;
    let cols = transpose8::<E>(rows, 0u32);
    #[unroll]
    for e in 0..8u32 {
        let mut col = cols[e as usize];
        if comptime!(factors >= 1) {
            col = Vector::cast_from(Vector::<f32, V8>::cast_from(col) * eight(f1, grp * 8));
        }
        if comptime!(factors >= 2) {
            col = Vector::cast_from(Vector::<f32, V8>::cast_from(col) * eight(f2, grp * 8));
        }
        dst[swizzle(c * 8 + e, grp, 8u32) as usize] = col;
    }
}

/// Blocks of `dO`'s 32 state columns a cube of [`wmma_state_backward_kernel`] stages transposed,
/// one to each of its last 32 units.
const SB_O_BLOCKS: u32 = 32;

/// Units of [`wmma_state_backward_kernel`]: eight waves, four where `Dk` is 32 (the state's
/// gradient a cube carries then has four tiles).
pub(super) fn state_backward_units(dk: u32) -> u32 {
    if dk >= 64 { EIGHT_WAVES } else { FOUR_WAVES }
}

/// What [`wmma_state_backward_kernel`] reads of chunk `chunk` but the state, into registers: the
/// keys by rows (`pk`) and `W^T` (`pw`), `8 Dk / units` vectors a unit; eight rows of one
/// 16-byte chunk of the queries (the first `Dk` units) or of `dO`'s 32 state columns (the last
/// 32) into `pt`, as [`token_block`] reads them; the lane's vector of `dv_local^T` for each of
/// its wave's `ptiles` tiles of `dV'^T` (`pl`); and the returned `G` (the first 64 units) or
/// query row factor (the next 64).
#[cube]
fn sb_read<E: Float>(
    kn: &Tensor<Vector<E, V8>>,
    w_t: &Tensor<Vector<E, V8>>,
    q: &Tensor<Vector<E, V8>>,
    q_factor: &Tensor<f32>,
    d_out: &Tensor<Vector<E, V8>>,
    dvl_t: &Tensor<Vector<E, V8>>,
    gcum: &Tensor<f32>,
    plan: &Tensor<u32>,
    pk: &mut Array<Vector<E, V8>>,
    pw: &mut Array<Vector<E, V8>>,
    pt: &mut Array<Vector<E, V8>>,
    pl: &mut Array<Vector<E, V8>>,
    chunk: u32,
    b: u32,
    h: u32,
    heads: u32,
    n_chunks: u32,
    seq_len: u32,
    scale: f32,
    v0: u32,
    lrow: u32,
    lcol: u32,
    #[comptime] dk: u32,
    #[comptime] dv: u32,
    #[comptime] l2norm: bool,
    #[comptime] packed: bool,
    #[comptime] units: u32,
    #[comptime] ptiles: u32,
) -> f32 {
    let ck = comptime!(dk / 8);
    let (f0, cnt) = chunk_tokens(
        plan,
        b as usize,
        (chunk % n_chunks) as usize,
        seq_len as usize,
        packed,
    );
    let first = f0 as u32;
    let count = cnt as u32;
    #[unroll]
    for i in 0..comptime!(64 * ck / units) {
        let at = chunk * 64 * ck + i * units + UNIT_POS;
        pk[i as usize] = kn[at as usize];
        pw[i as usize] = w_t[at as usize];
    }
    let o0 = comptime!(units - SB_O_BLOCKS);
    if UNIT_POS < dk {
        token_block::<E>(q, pt, first, count, h, heads, 0u32, UNIT_POS, dk, ck);
    } else if UNIT_POS >= o0 {
        token_block::<E>(
            d_out,
            pt,
            first,
            count,
            h,
            heads,
            v0 / 8,
            UNIT_POS - o0,
            dv,
            4u32,
        );
    }
    #[unroll]
    for s in 0..ptiles {
        pl[s as usize] = dvl_t[((chunk * dv + lrow) * 8 + lcol + 2 * s) as usize];
    }
    let mut g = scale;
    if UNIT_POS < 64 {
        g = gcum[(chunk * 64 + UNIT_POS) as usize];
    } else if UNIT_POS < 128 {
        if comptime!(l2norm) {
            g = q_factor[(chunk * 64 + UNIT_POS - 64) as usize];
        }
    }
    g
}

/// Writes the keys by rows and `W^T` [`sb_read`] read into their tiles, and `G`, `exp(G)` and the
/// query rows' factors.
#[cube]
fn sb_stage<E: Float>(
    pk: &Array<Vector<E, V8>>,
    pw: &Array<Vector<E, V8>>,
    g: f32,
    s_k: &mut [Vector<E, V8>],
    s_wt: &mut [Vector<E, V8>],
    s_g: &mut [f32],
    s_gr: &mut [f32],
    s_f: &mut [f32],
    #[comptime] dk: u32,
    #[comptime] units: u32,
) {
    let ck = comptime!(dk / 8);
    #[unroll]
    for i in 0..comptime!(64 * ck / units) {
        let idx = i * units + UNIT_POS;
        s_k[swizzle(idx / ck, idx % ck, ck) as usize] = pk[i as usize];
        s_wt[swizzle(idx / 8, idx % 8, 8u32) as usize] = pw[i as usize];
    }
    if UNIT_POS < 64 {
        s_gr[UNIT_POS as usize] = g;
        s_g[UNIT_POS as usize] = g.exp();
    } else if UNIT_POS < 128 {
        s_f[(UNIT_POS - 64) as usize] = g;
    }
}

/// The state passing, backward, as `tc.rs`' `tc_state_backward_kernel`: for each of the
/// sequence's chunks from the last, store `dS_{c+1}` (transposed, in the input dtype), form
/// `dV'^T = lam (dS^T K^T) + dv_local^T`, then `dS^T = exp(G_last) dS^T + dO^T qg + (-dV'^T) W`,
/// the state's gradient f32 in registers, laid out as the forward's state (`wmma.rs`): one cube
/// per (sequence and head, block of 32 state columns) of [`state_backward_units`], wave `w`
/// holding `st` consecutive tiles of `dS^T` from tile `st w` (row tiles of `Dk / 16`) and
/// forming `ptiles` of the eight tiles of `dV'^T` from `ptiles w` (row tiles of four). One tile
/// holds the keys by rows for `dV'`, then `qg^T` (`gamma` times each query row times its factor,
/// each rounded to the input dtype, as `tc.rs` forms it). As the forward does, it reads the next
/// chunk's inputs (all but the state) into registers while forming this chunk's `dS^T`; the keys,
/// `W^T` and decays go to shared memory after it, the query and `dO` rows once `dV'` is formed and
/// the keys' tile is free.
#[cube(launch_unchecked)]
pub(super) fn wmma_state_backward_kernel<E: Float>(
    kn: &Tensor<Vector<E, V8>>,
    w_t: &Tensor<Vector<E, V8>>,
    q: &Tensor<Vector<E, V8>>,
    q_factor: &Tensor<f32>,
    d_out: &Tensor<Vector<E, V8>>,
    dvl_t: &Tensor<Vector<E, V8>>,
    gcum: &Tensor<f32>,
    d_last: &Tensor<f32>,
    offsets: &Tensor<u32>,
    plan: &Tensor<u32>,
    dsn_t: &mut Tensor<Vector<E, V8>>,
    dvn_t: &mut Tensor<Vector<E, V8>>,
    d_first: &mut Tensor<f32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    scale: f32,
    #[comptime] dk: u32,
    #[comptime] dv: u32,
    #[comptime] l2norm: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let ck = comptime!(dk / 8);
    let y = CUBE_POS_Y;
    let v0 = CUBE_POS_X * 32;
    let wave = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let row = lane % 16;
    let half = lane / 16;
    let (base, n) = state_chunks(
        offsets,
        y as usize,
        heads as usize,
        n_chunks as usize,
        packed,
    );
    let base = base as u32;
    let n = n as u32;
    let b = y / heads;
    let h = y % heads;

    let units = comptime!(state_backward_units(dk));
    let st = comptime!(dk / 8 / (units / 32));
    let ptiles = comptime!(8 / (units / 32));
    let s_first = wave * st;
    let sv = s_first / comptime!(dk / 16);
    let sd0 = s_first % comptime!(dk / 16);
    let vr = 16 * sv + row;
    let p_first = wave * ptiles;
    let pv = p_first / 4;
    let pt0 = p_first % 4;
    let pr = 16 * pv + row;

    // The keys by rows, then qg^T; W^T; dS^T (32 rows of Dk); -dV'^T and dO^T (32 rows of the
    // chunk's 64 tokens each).
    let mut s_k =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * ck) as usize), 16usize);
    let mut s_wt =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * ck) as usize), 16usize);
    let mut s_s =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((32 * ck) as usize), 16usize);
    let mut s_v = Shared::<[Vector<E, V8>]>::new_aligned_slice(256usize, 16usize);
    let mut s_o = Shared::<[Vector<E, V8>]>::new_aligned_slice(256usize, 16usize);
    let mut s_g = Shared::<[f32]>::new_slice(64usize);
    let mut s_gr = Shared::<[f32]>::new_slice(64usize);
    let mut s_f = Shared::<[f32]>::new_slice(64usize);

    let mut sreg = zeros(st);
    #[unroll]
    for s in 0..st {
        #[unroll]
        for e in 0..8u32 {
            let d = 16 * (sd0 + s) + 8 * half + e;
            sreg[(8 * s + e) as usize] =
                Vector::new(d_last[((y * dk + d) * dv + v0 + vr) as usize]);
        }
    }

    // The first chunk's inputs, read; its keys, W^T and decays staged.
    let per = comptime!(64 * ck / units);
    let mut pk = Array::<Vector<E, V8>>::new(per as usize);
    let mut pw = Array::<Vector<E, V8>>::new(per as usize);
    let mut pt = Array::<Vector<E, V8>>::new(8usize);
    let mut pl = Array::<Vector<E, V8>>::new(ptiles as usize);
    let lrow = v0 + pr;
    let lcol = 2 * pt0 + half;
    let o0 = comptime!(units - SB_O_BLOCKS);
    if n > 0 {
        let g = sb_read::<E>(
            kn,
            w_t,
            q,
            q_factor,
            d_out,
            dvl_t,
            gcum,
            plan,
            &mut pk,
            &mut pw,
            &mut pt,
            &mut pl,
            base + n - 1,
            b,
            h,
            heads,
            n_chunks,
            seq_len,
            scale,
            v0,
            lrow,
            lcol,
            dk,
            dv,
            l2norm,
            packed,
            units,
            ptiles,
        );
        sb_stage::<E>(
            &pk,
            &pw,
            g,
            s_k.as_mut_slice(),
            s_wt.as_mut_slice(),
            s_g.as_mut_slice(),
            s_gr.as_mut_slice(),
            s_f.as_mut_slice(),
            dk,
            units,
        );
    }

    for cr in 0..n {
        let chunk = base + n - 1 - cr;
        // dS_{c+1}, stored and staged.
        #[unroll]
        for s in 0..st {
            let x = tile_as::<E>(&sreg, s);
            let cc = 2 * (sd0 + s) + half;
            dsn_t[((chunk * dv + v0 + vr) * ck + cc) as usize] = x;
            s_s[swizzle(vr, cc, ck) as usize] = x;
        }
        let g_last = gcum[(chunk * 64 + 63) as usize];
        sync_cube();

        // dV'^T = lam (dS^T K^T) + dv_local^T.
        let mut acc_v = zeros(ptiles);
        #[unroll]
        for kk in 0..comptime!(dk / 16) {
            let x = frag::<E>(s_s.as_slice(), 16 * pv, kk, ck);
            #[unroll]
            for s in 0..ptiles {
                let a = frag::<E>(s_k.as_slice(), 16 * (pt0 + s), kk, ck);
                mma::<E>(&def, &a, &x, &mut acc_v, s);
            }
        }
        #[unroll]
        for s in 0..ptiles {
            let t0 = 16 * (pt0 + s) + 8 * half;
            let mut lam = Vector::<f32, V8>::new(0.0f32);
            #[unroll]
            for e in 0..8u32 {
                lam.insert(e as usize, (g_last - s_gr[(t0 + e) as usize]).exp());
            }
            let x = tile_f32(&acc_v, s) * lam + Vector::<f32, V8>::cast_from(pl[s as usize]);
            let cc = 2 * (pt0 + s) + half;
            dvn_t[((chunk * dv + v0 + pr) * 8 + cc) as usize] = Vector::cast_from(x);
            s_v[swizzle(pr, cc, 8u32) as usize] = Vector::cast_from(x * Vector::new(-1.0f32));
        }
        sync_cube();
        // qg^T where the keys were; dO^T.
        if UNIT_POS < dk {
            token_block_t::<E>(
                &pt,
                s_k.as_mut_slice(),
                s_f.as_slice(),
                s_g.as_slice(),
                UNIT_POS,
                ck,
                2u32,
            );
        } else if UNIT_POS >= o0 {
            token_block_t::<E>(
                &pt,
                s_o.as_mut_slice(),
                s_g.as_slice(),
                s_g.as_slice(),
                UNIT_POS - o0,
                4u32,
                0u32,
            );
        }
        sync_cube();

        // The next chunk's inputs into registers, while
        // dS^T = exp(G_last) dS^T + dO^T qg + (-dV'^T) W.
        let more = cr + 1 < n;
        let mut g = 0.0f32;
        if more {
            g = sb_read::<E>(
                kn,
                w_t,
                q,
                q_factor,
                d_out,
                dvl_t,
                gcum,
                plan,
                &mut pk,
                &mut pw,
                &mut pt,
                &mut pl,
                chunk - 1,
                b,
                h,
                heads,
                n_chunks,
                seq_len,
                scale,
                v0,
                lrow,
                lcol,
                dk,
                dv,
                l2norm,
                packed,
                units,
                ptiles,
            );
        }
        let decay = g_last.exp();
        #[unroll]
        for x in 0..comptime!(8 * st) {
            sreg[x as usize] = sreg[x as usize] * Vector::new(decay);
        }
        #[unroll]
        for kt in 0..4u32 {
            let x = frag::<E>(s_o.as_slice(), 16 * sv, kt, 8u32);
            #[unroll]
            for s in 0..st {
                let a = frag::<E>(s_k.as_slice(), 16 * (sd0 + s), kt, 8u32);
                mma::<E>(&def, &a, &x, &mut sreg, s);
            }
        }
        #[unroll]
        for kt in 0..4u32 {
            let x = frag::<E>(s_v.as_slice(), 16 * sv, kt, 8u32);
            #[unroll]
            for s in 0..st {
                let a = frag::<E>(s_wt.as_slice(), 16 * (sd0 + s), kt, 8u32);
                mma::<E>(&def, &a, &x, &mut sreg, s);
            }
        }
        sync_cube();
        if more {
            sb_stage::<E>(
                &pk,
                &pw,
                g,
                s_k.as_mut_slice(),
                s_wt.as_mut_slice(),
                s_g.as_mut_slice(),
                s_gr.as_mut_slice(),
                s_f.as_mut_slice(),
                dk,
                units,
            );
        }
    }

    #[unroll]
    for s in 0..st {
        #[unroll]
        for e in 0..8u32 {
            let d = 16 * (sd0 + s) + 8 * half + e;
            d_first[((y * dk + d) * dv + v0 + vr) as usize] =
                sreg[(8 * s + e) as usize].extract(0usize);
        }
    }
}

/// The index of 16-byte chunk `c` of row `r` in a swizzled tile of `chunks` chunks to a row:
/// `tc.rs`' `swizzle` for 4 and for 8 or more, and for rows of 2 chunks the chunk XOR-ed with bit
/// 2 of the row.
#[cube]
fn swz(r: u32, c: u32, #[comptime] chunks: u32) -> u32 {
    if comptime!(chunks >= 8) {
        r * chunks + (c ^ (r & 7))
    } else if comptime!(chunks == 4) {
        r * chunks + (c ^ ((r >> 1) & 3))
    } else {
        r * chunks + (c ^ ((r >> 2) & 1))
    }
}

/// [`frag`] from the tile that starts `base` vectors into `tile`, swizzled by [`swz`].
#[cube]
fn fragb<E: Float>(
    tile: &[Vector<E, V8>],
    base: u32,
    r0: u32,
    kk: u32,
    #[comptime] chunks: u32,
) -> Array<Vector<E, V8>> {
    let lane = UNIT_POS_PLANE;
    let mut x = Array::<Vector<E, V8>>::new(1usize);
    x[0] = tile[(base + swz(r0 + lane % 16, 2 * kk + lane / 16, chunks)) as usize];
    x
}

/// Sums each of a chunk's 64 rows over the lanes and waves holding it: `p` the lane's part of
/// row `16 it + l % 16` (both half-waves hold a part of it, and the waves of `hf` 0 and 1 each
/// half of the row tile's columns), `out[r]` the row's sum, added in that order. Two barriers.
#[cube]
fn row_total(p: f32, it: u32, hf: u32, red: &mut [f32], out: &mut [f32]) {
    let lane = UNIT_POS_PLANE;
    let t = p + plane_shuffle_xor(p, 16u32);
    if lane < 16 {
        red[(hf * 64 + 16 * it + lane) as usize] = t;
    }
    sync_cube();
    if UNIT_POS < 64 {
        out[UNIT_POS as usize] = red[UNIT_POS as usize] + red[(64 + UNIT_POS) as usize];
    }
    sync_cube();
}

/// The dot product of eight f32 and eight of the input dtype.
#[cube]
fn dot8<E: Float>(a: Vector<f32, V8>, b: Vector<E, V8>) -> f32 {
    let p = a * Vector::<f32, V8>::cast_from(b);
    let mut x = 0.0f32;
    #[unroll]
    for e in 0..8usize {
        x += p.extract(e);
    }
    x
}

/// Units of [`wmma_dqk_kernel`] that load the transposed operands of a block of its loop over
/// `Dv`, each source from its own range of units, so that no unit holds more than eight vectors:
/// `V'^T` from unit 128, `S^T` from 160 and `dS^T` from 192 (units 0 to 127 load `dO`).
const DQK_VNT_UNITS: u32 = 128;
const DQK_S_UNITS: u32 = 160;
const DQK_DS_UNITS: u32 = 192;

/// The global reads of block `vb` of [`wmma_dqk_kernel`]'s loop over `Dv` (16 columns): the
/// unit's vector of `dO` (units 0 to 127), or one 16-byte chunk of 8 consecutive rows of the
/// block's 16 rows of `V'^T`, `S^T` or `dS^T`. Returned to be written by [`dqk_block_store`]
/// after the products of the block before, so that the reads are in flight while those run.
#[cube]
fn dqk_block_load<E: Float>(
    d_out: &Tensor<Vector<E, V8>>,
    vnt: &Tensor<Vector<E, V8>>,
    states: &Tensor<Vector<E, V8>>,
    dsn_t: &Tensor<Vector<E, V8>>,
    cube: u32,
    first: u32,
    count: u32,
    h: u32,
    heads: u32,
    vb: u32,
    #[comptime] dk: u32,
    #[comptime] dv: u32,
) -> Array<Vector<E, V8>> {
    let ck = comptime!(dk / 8);
    let cv = comptime!(dv / 8);
    let u = UNIT_POS;
    let mut vals = Array::<Vector<E, V8>>::new(8usize);
    if u < DQK_VNT_UNITS {
        let r = u / 2;
        let mut x = Vector::<E, V8>::new(E::from_int(0));
        if r < count {
            x = d_out[(((first + r) * heads + h) * cv + 2 * vb + u % 2) as usize];
        }
        vals[0] = x;
    } else if u < DQK_VNT_UNITS + 16 {
        let idx = u - DQK_VNT_UNITS;
        let c = idx % 8;
        let grp = idx / 8;
        #[unroll]
        for j in 0..8u32 {
            vals[j as usize] = vnt[(cube * dv * 8 + (16 * vb + grp * 8 + j) * 8 + c) as usize];
        }
    } else if u >= DQK_S_UNITS && u < comptime!(DQK_S_UNITS + 2 * dk / 8) {
        let idx = u - DQK_S_UNITS;
        let c = idx % ck;
        let grp = idx / ck;
        #[unroll]
        for j in 0..8u32 {
            vals[j as usize] = states[(cube * dv * ck + (16 * vb + grp * 8 + j) * ck + c) as usize];
        }
    } else if u >= DQK_DS_UNITS && u < comptime!(DQK_DS_UNITS + 2 * dk / 8) {
        let idx = u - DQK_DS_UNITS;
        let c = idx % ck;
        let grp = idx / ck;
        #[unroll]
        for j in 0..8u32 {
            vals[j as usize] = dsn_t[(cube * dv * ck + (16 * vb + grp * 8 + j) * ck + c) as usize];
        }
    }
    vals
}

/// Writes what [`dqk_block_load`] read into the loop's region of shared memory: `dO` by rows at
/// `o_at` (rows of 2 chunks), and `V'`, `S` and `dS` by rows at `v_at`, `s_at` and `z_at`, each
/// unit's 8 rows of 8 transposed in registers so that row `8 c + e` holds element `e` of chunk
/// `c` of the 16 rows read, 2 chunks to a row ([`swz`]).
#[cube]
fn dqk_block_store<E: Float>(
    vals: &Array<Vector<E, V8>>,
    r4: &mut [Vector<E, V8>],
    o_at: u32,
    v_at: u32,
    s_at: u32,
    z_at: u32,
    #[comptime] dk: u32,
) {
    let ck = comptime!(dk / 8);
    let u = UNIT_POS;
    if u < DQK_VNT_UNITS {
        r4[(o_at + swz(u / 2, u % 2, 2u32)) as usize] = vals[0];
    } else if u < DQK_VNT_UNITS + 16 {
        let idx = u - DQK_VNT_UNITS;
        let cols = transpose8::<E>(vals, 0u32);
        #[unroll]
        for e in 0..8u32 {
            r4[(v_at + swz((idx % 8) * 8 + e, idx / 8, 2u32)) as usize] = cols[e as usize];
        }
    } else if u >= DQK_S_UNITS && u < comptime!(DQK_S_UNITS + 2 * dk / 8) {
        let idx = u - DQK_S_UNITS;
        let cols = transpose8::<E>(vals, 0u32);
        #[unroll]
        for e in 0..8u32 {
            r4[(s_at + swz((idx % ck) * 8 + e, idx / ck, 2u32)) as usize] = cols[e as usize];
        }
    } else if u >= DQK_DS_UNITS && u < comptime!(DQK_DS_UNITS + 2 * dk / 8) {
        let idx = u - DQK_DS_UNITS;
        let cols = transpose8::<E>(vals, 0u32);
        #[unroll]
        for e in 0..8u32 {
            r4[(z_at + swz((idx % ck) * 8 + e, idx / ck, 2u32)) as usize] = cols[e as usize];
        }
    }
}

/// Backward, per chunk, everything but the WY part, as `tc.rs`' `tc_dqk_kernel`: `dV_b = T^T dV'`,
/// the values' gradient `dV_b beta` and `sum_v dV_b V`; over `Dv`, 16 columns at a time,
/// `dQg = dO S^T`, `dkt = V' dS^T`, `dKbg = -dV_b S^T`, `<S, dS>`, `dP = dO V'^T` and `dP^T`; then
/// `P` and `P^T` again, `dQ = dPe K` and `dK = dPe^T Q`. It writes the queries' gradient (through
/// the l2norm when asked), `dV_b` and `dKbg` by rows in the input dtype (the WY kernel's operands),
/// the keys' gradient without the WY part, `lam dkt + beta gamma dKbg + dK_attn` (f32), per token
/// the gate's and `beta`'s parts, and per chunk the gate's tail. Eight waves: wave `w` takes row
/// tile `w % 4` and half (`w / 4`) of each product's columns. Four regions of shared memory, each
/// holding several operands in turn.
#[cube(launch_unchecked)]
pub(super) fn wmma_dqk_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    beta: &Tensor<f32>,
    gcum: &Tensor<f32>,
    ttr: &Tensor<Vector<E, V8>>,
    dvn_t: &Tensor<Vector<E, V8>>,
    d_out: &Tensor<Vector<E, V8>>,
    vnt: &Tensor<Vector<E, V8>>,
    states: &Tensor<Vector<E, V8>>,
    dsn_t: &Tensor<Vector<E, V8>>,
    kn: &Tensor<Vector<E, V8>>,
    knt: &Tensor<Vector<E, V8>>,
    dq: &mut Tensor<Vector<E, V8>>,
    dv_out: &mut Tensor<Vector<E, V8>>,
    dvb: &mut Tensor<Vector<E, V8>>,
    dkbg: &mut Tensor<Vector<E, V8>>,
    dk_part: &mut Tensor<f32>,
    rg_part: &mut Tensor<f32>,
    rb_part: &mut Tensor<f32>,
    tail: &mut Tensor<f32>,
    plan: &Tensor<u32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    scale: f32,
    #[comptime] dk: u32,
    #[comptime] dv: u32,
    #[comptime] l2norm: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let ck = comptime!(dk / 8);
    let cv = comptime!(dv / 8);
    let cmax = comptime!(if dk > dv { dk / 8 } else { dv / 8 });
    let cube = CUBE_POS_X;
    let c = cube % n_chunks;
    let bh = cube / n_chunks;
    let b = bh / heads;
    let h = bh % heads;
    let (first, count) = chunk_tokens(plan, b as usize, c as usize, seq_len as usize, packed);
    if packed && count == 0usize {
        terminate!();
    }
    let first = first as u32;
    let count = count as u32;
    let tid = UNIT_POS as usize;
    let wave = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let row = lane % 16;
    let half = lane / 16;
    let it = wave % 4;
    let hf = wave / 4;
    let i = 16 * it + row;

    // R1: dV'^T, then the queries by rows, then Q^T. R2: dV_b by rows, then K^T. R3: T^T, then
    // dPe by rows, then dPe^T by rows. R4: a block of dO, V', S and dS (16 columns of Dv each,
    // rows of 2 chunks), then the keys by rows.
    let mut r1 =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * cmax) as usize), 16usize);
    let mut r2 =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * cmax) as usize), 16usize);
    let mut r3 = Shared::<[Vector<E, V8>]>::new_aligned_slice(512usize, 16usize);
    let r4_len = comptime!(if 64 * ck > 256 + 4 * dk {
        64 * ck
    } else {
        256 + 4 * dk
    });
    let mut r4 = Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!(r4_len as usize), 16usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut bs = Shared::<[f32]>::new_slice(64usize);
    let mut inv_q = Shared::<[f32]>::new_slice(64usize);
    let mut red = Shared::<[f32]>::new_slice(128usize);
    let mut s_bv = Shared::<[f32]>::new_slice(64usize);
    let mut s_bg = Shared::<[f32]>::new_slice(64usize);
    let mut s_kt = Shared::<[f32]>::new_slice(64usize);
    let mut s_qg = Shared::<[f32]>::new_slice(64usize);
    let mut s_row = Shared::<[f32]>::new_slice(64usize);
    let mut s_col = Shared::<[f32]>::new_slice(64usize);
    let mut s_pq = Shared::<[f32]>::new_slice(64usize);
    let mut s_ds = Shared::<[f32]>::new_slice(8usize);

    if tid < 64usize {
        gs[tid] = gcum[cube as usize * 64usize + tid];
        let mut x = 0.0f32;
        if (tid as u32) < count {
            x = beta[(first as usize + tid) * heads as usize + h as usize];
        }
        bs[tid] = x;
    }
    let g_last = gcum[(cube * 64 + 63) as usize];
    stage_copy::<E>(
        ttr,
        r3.as_mut_slice(),
        cube * 512,
        512u32,
        8u32,
        EIGHT_WAVES,
    );
    stage_copy::<E>(
        dvn_t,
        r1.as_mut_slice(),
        cube * dv * 8,
        comptime!(dv * 8),
        8u32,
        EIGHT_WAVES,
    );
    sync_cube();

    // dV_b = T^T dV': row tile `it`, `Dv / 32` column tiles from `hf Dv / 32`; T^T's tiles left
    // of the diagonal are zero.
    let nvt = comptime!(dv / 32);
    let mut acc_b = zeros(nvt);
    #[unroll]
    for kt in 0..4u32 {
        if kt >= it {
            let x = frag::<E>(r3.as_slice(), 16 * it, kt, 8u32);
            #[unroll]
            for s in 0..nvt {
                let a = frag::<E>(r1.as_slice(), 16 * (hf * nvt + s), kt, 8u32);
                mma::<E>(&def, &a, &x, &mut acc_b, s);
            }
        }
    }
    let mut part_bv = 0.0f32;
    #[unroll]
    for s in 0..nvt {
        let cc = 2 * (hf * nvt + s) + half;
        let x = tile_as::<E>(&acc_b, s);
        dvb[((cube * 64 + i) * cv + cc) as usize] = x;
        r2[swz(i, cc, cv) as usize] = x;
        if i < count {
            let at = (((first + i) * heads + h) * cv + cc) as usize;
            let f = tile_f32(&acc_b, s);
            dv_out[at] = Vector::cast_from(f * Vector::new(bs[i as usize]));
            part_bv += dot8::<E>(f, v[at]);
        }
    }
    row_total(part_bv, it, hf, red.as_mut_slice(), s_bv.as_mut_slice());

    // Over Dv, 16 columns at a time: dQg = dO S^T, dkt = V' dS^T, dV_b S^T (row tile `it`,
    // `Dk / 32` column tiles from `hf Dk / 32`), dP = dO V'^T and dP^T = V' dO^T (row tile `it`,
    // column tiles `2 hf` and `2 hf + 1`), and <S, dS>.
    let ndt = comptime!(dk / 32);
    let mut acc1 = zeros(ndt);
    let mut acc2 = zeros(ndt);
    let mut acc3 = zeros(ndt);
    let mut acc_dp = zeros(2u32);
    let mut acc_dpt = zeros(2u32);
    let mut part_ds = 0.0f32;
    let o_at = 0u32;
    let v_at = 128u32;
    let s_at = 256u32;
    let z_at = comptime!(256 + 2 * dk);
    // dO by rows (64 rows of 2 chunks), V' by rows from V'^T, and S and dS by rows (Dk rows)
    // from S^T and dS^T: each block read into registers before the products of the block before
    // it, written to shared memory after them.
    let mut ahead = dqk_block_load::<E>(
        d_out, vnt, states, dsn_t, cube, first, count, h, heads, 0u32, dk, dv,
    );
    for vb in 0..comptime!(dv / 16) {
        sync_cube();
        dqk_block_store::<E>(&ahead, r4.as_mut_slice(), o_at, v_at, s_at, z_at, dk);
        sync_cube();
        if vb + 1 < comptime!(dv / 16) {
            ahead = dqk_block_load::<E>(
                d_out,
                vnt,
                states,
                dsn_t,
                cube,
                first,
                count,
                h,
                heads,
                vb + 1,
                dk,
                dv,
            );
        }
        // <S, dS> over the block, each unit its vectors.
        #[unroll]
        for m in 0..comptime!((2 * dk + EIGHT_WAVES - 1) / EIGHT_WAVES) {
            let idx = m * EIGHT_WAVES + UNIT_POS;
            if idx < comptime!(2 * dk) {
                let x = Vector::<f32, V8>::cast_from(r4[(s_at + idx) as usize]);
                part_ds += dot8::<E>(x, r4[(z_at + idx) as usize]);
            }
        }
        let xo = fragb::<E>(r4.as_slice(), o_at, 16 * it, 0u32, 2u32);
        let xv = fragb::<E>(r4.as_slice(), v_at, 16 * it, 0u32, 2u32);
        let xb = frag::<E>(r2.as_slice(), 16 * it, vb, cv);
        #[unroll]
        for s in 0..ndt {
            let dt = hf * ndt + s;
            let a_s = fragb::<E>(r4.as_slice(), s_at, 16 * dt, 0u32, 2u32);
            let a_z = fragb::<E>(r4.as_slice(), z_at, 16 * dt, 0u32, 2u32);
            mma::<E>(&def, &a_s, &xo, &mut acc1, s);
            mma::<E>(&def, &a_z, &xv, &mut acc2, s);
            mma::<E>(&def, &a_s, &xb, &mut acc3, s);
        }
        #[unroll]
        for s in 0..2u32 {
            let jt = 2 * hf + s;
            let a_v = fragb::<E>(r4.as_slice(), v_at, 16 * jt, 0u32, 2u32);
            mma::<E>(&def, &a_v, &xo, &mut acc_dp, s);
            let a_o = fragb::<E>(r4.as_slice(), o_at, 16 * jt, 0u32, 2u32);
            mma::<E>(&def, &a_o, &xv, &mut acc_dpt, s);
        }
    }
    sync_cube();

    // The queries (normalised, unscaled) and the keys (normalised) by rows.
    stage_scaled::<E>(
        q,
        r1.as_mut_slice(),
        inv_q.as_mut_slice(),
        first,
        count,
        h,
        heads,
        1.0f32,
        dk,
        l2norm,
        EIGHT_WAVES,
    );
    stage_copy::<E>(
        kn,
        r4.as_mut_slice(),
        cube * 64 * ck,
        comptime!(64 * ck),
        ck,
        EIGHT_WAVES,
    );
    sync_cube();

    // dKbg out; the sums of the gate's and beta's gradients; dq's part gamma dQg (times the
    // scale) and dk's lam dkt + beta gamma dKbg.
    let gi = gs[i as usize];
    let gamma = gi.exp();
    let lam = (g_last - gi).exp();
    let bi = bs[i as usize];
    let mut part_bg = 0.0f32;
    let mut part_kt = 0.0f32;
    let mut part_qg = 0.0f32;
    #[unroll]
    for s in 0..ndt {
        let cc = 2 * (hf * ndt + s) + half;
        let kb = tile_f32(&acc3, s) * Vector::new(-1.0f32);
        dkbg[((cube * 64 + i) * ck + cc) as usize] = Vector::cast_from(kb);
        let kk = r4[swizzle(i, cc, ck) as usize];
        let qq = r1[swizzle(i, cc, ck) as usize];
        let a1 = tile_f32(&acc1, s);
        let a2 = tile_f32(&acc2, s);
        part_bg += dot8::<E>(kb, kk);
        part_kt += dot8::<E>(a2, kk);
        part_qg += dot8::<E>(a1, qq);
        let n2 = a2 * Vector::new(lam) + kb * Vector::new(bi * gamma);
        let n1 = a1 * Vector::new(gamma * scale);
        #[unroll]
        for e in 0..8u32 {
            acc2[(8 * s + e) as usize] = Vector::new(n2.extract(e as usize));
            acc1[(8 * s + e) as usize] = Vector::new(n1.extract(e as usize));
        }
    }
    row_total(part_bg, it, hf, red.as_mut_slice(), s_bg.as_mut_slice());
    row_total(part_kt, it, hf, red.as_mut_slice(), s_kt.as_mut_slice());
    row_total(part_qg, it, hf, red.as_mut_slice(), s_qg.as_mut_slice());

    // P = Q K^T (row tile `it`, column tiles 2 hf and 2 hf + 1) and P^T = K Q^T (the same);
    // dPe = dP exp(G_i - G_j) scale (i >= j) and its transpose; the gate's part from P, its rows'
    // sums minus its columns' (the rows' of P^T).
    let mut acc_s = zeros(2u32);
    let mut acc_st = zeros(2u32);
    #[unroll]
    for kk in 0..comptime!(dk / 16) {
        let xq = frag::<E>(r1.as_slice(), 16 * it, kk, ck);
        let xk = frag::<E>(r4.as_slice(), 16 * it, kk, ck);
        #[unroll]
        for s in 0..2u32 {
            let jt = 2 * hf + s;
            let a_k = frag::<E>(r4.as_slice(), 16 * jt, kk, ck);
            mma::<E>(&def, &a_k, &xq, &mut acc_s, s);
            let a_q = frag::<E>(r1.as_slice(), 16 * jt, kk, ck);
            mma::<E>(&def, &a_q, &xk, &mut acc_st, s);
        }
    }
    let mut part_row = 0.0f32;
    let mut part_col = 0.0f32;
    #[unroll]
    for s in 0..2u32 {
        let jt = 2 * hf + s;
        #[unroll]
        for e in 0..8u32 {
            let j = 16 * jt + 8 * half + e;
            let at = (8 * s + e) as usize;
            let gj = gs[j as usize];
            // dPe[i, j] and dPe^T[i, j] = dPe[j, i].
            let d = select(
                i >= j,
                acc_dp[at].extract(0usize) * (gi - gj).exp() * scale,
                0.0f32,
            );
            let dt = select(
                j >= i,
                acc_dpt[at].extract(0usize) * (gj - gi).exp() * scale,
                0.0f32,
            );
            part_row += d * acc_s[at].extract(0usize);
            part_col += dt * acc_st[at].extract(0usize);
            acc_dp[at] = Vector::new(d);
            acc_dpt[at] = Vector::new(dt);
        }
        r3[swz(i, 2 * jt + half, 8u32) as usize] = tile_as::<E>(&acc_dp, s);
    }
    // K^T where dV_b was.
    stage_copy::<E>(
        knt,
        r2.as_mut_slice(),
        cube * dk * 8,
        comptime!(dk * 8),
        8u32,
        EIGHT_WAVES,
    );
    row_total(part_row, it, hf, red.as_mut_slice(), s_row.as_mut_slice());
    row_total(part_col, it, hf, red.as_mut_slice(), s_col.as_mut_slice());

    // dq's part += dPe K: dPe's tiles left of the diagonal.
    #[unroll]
    for kt in 0..4u32 {
        if kt <= it {
            let x = frag::<E>(r3.as_slice(), 16 * it, kt, 8u32);
            #[unroll]
            for s in 0..ndt {
                let a = frag::<E>(r2.as_slice(), 16 * (hf * ndt + s), kt, 8u32);
                mma::<E>(&def, &a, &x, &mut acc1, s);
            }
        }
    }
    sync_cube();
    // dPe^T by rows where dPe was; Q^T where the queries were.
    #[unroll]
    for s in 0..2u32 {
        let jt = 2 * hf + s;
        r3[swz(i, 2 * jt + half, 8u32) as usize] = tile_as::<E>(&acc_dpt, s);
    }
    stage_tokens_t::<E>(
        q,
        r1.as_mut_slice(),
        inv_q.as_slice(),
        inv_q.as_slice(),
        first,
        count,
        h,
        heads,
        0u32,
        dk,
        ck,
        1u32,
        EIGHT_WAVES,
    );
    sync_cube();
    // dk's part += dPe^T Q: dPe^T's tiles right of the diagonal.
    #[unroll]
    for kt in 0..4u32 {
        if kt >= it {
            let x = frag::<E>(r3.as_slice(), 16 * it, kt, 8u32);
            #[unroll]
            for s in 0..ndt {
                let a = frag::<E>(r1.as_slice(), 16 * (hf * ndt + s), kt, 8u32);
                mma::<E>(&def, &a, &x, &mut acc2, s);
            }
        }
    }

    // The queries' gradient, through the l2norm when asked: inv (dq - q (q . dq)), the
    // normalised queries read again as the tiles held them.
    let mut qrow = Array::<Vector<E, V8>>::new(comptime!(ndt as usize));
    #[unroll]
    for s in 0..ndt {
        let cc = 2 * (hf * ndt + s) + half;
        let mut x = Vector::<E, V8>::new(E::from_int(0));
        if i < count {
            x = q[(((first + i) * heads + h) * ck + cc) as usize];
        }
        qrow[s as usize] =
            Vector::cast_from(Vector::<f32, V8>::cast_from(x) * Vector::new(inv_q[i as usize]));
    }
    if comptime!(l2norm) {
        let mut part_pq = 0.0f32;
        #[unroll]
        for s in 0..ndt {
            part_pq += dot8::<E>(tile_f32(&acc1, s), qrow[s as usize]);
        }
        row_total(part_pq, it, hf, red.as_mut_slice(), s_pq.as_mut_slice());
    }
    #[unroll]
    for s in 0..ndt {
        let cc = 2 * (hf * ndt + s) + half;
        let mut x = tile_f32(&acc1, s);
        if comptime!(l2norm) {
            let qq = Vector::<f32, V8>::cast_from(qrow[s as usize]);
            x = (x - qq * Vector::new(s_pq[i as usize])) * Vector::new(inv_q[i as usize]);
        }
        if i < count {
            dq[(((first + i) * heads + h) * ck + cc) as usize] = Vector::cast_from(x);
        }
        let d0 = 16 * (hf * ndt + s) + 8 * half;
        #[unroll]
        for e in 0..8u32 {
            dk_part[((cube * 64 + i) * dk + d0 + e) as usize] =
                acc2[(8 * s + e) as usize].extract(0usize);
        }
    }

    // Per token, the gate's and beta's parts; per chunk, the gate's tail at the last row,
    // exp(G_last) <S, dS> + sum lam s_kt, summed over each wave and then the waves in order.
    let mut lam_kt = 0.0f32;
    if tid < 64usize {
        let g = gs[tid];
        let gm = g.exp();
        let lm = (g_last - g).exp();
        let bg = s_bg[tid];
        lam_kt = lm * s_kt[tid];
        let d_gamma = s_qg[tid] * scale + bs[tid] * bg;
        rg_part[cube as usize * 64usize + tid] = s_row[tid] - s_col[tid] + d_gamma * gm - lam_kt;
        rb_part[cube as usize * 64usize + tid] = s_bv[tid] + gm * bg;
    }
    let ds_wave = plane_sum(part_ds);
    let rl_wave = plane_sum(lam_kt);
    if lane == 0 {
        s_ds[wave as usize] = g_last.exp() * ds_wave + rl_wave;
    }
    sync_cube();
    if tid == 0usize {
        let mut x = 0.0f32;
        #[unroll]
        for y in 0..8usize {
            x += s_ds[y];
        }
        tail[cube as usize] = x;
    }
}

/// Backward, per chunk, the WY part and the sums, as `tc.rs`' `tc_wy_kernel`:
/// `dA = -(dV_b U^T + dKbg W^T)` and its transpose (`i > j`), `K K^T` again, the gate's part
/// (the row sums of `dAe (beta K K^T)` less its column sums, the rows' of the transpose) and
/// `beta`'s; the keys' gradient `dK + beta (dAe K) + dAe^T (beta K)` (through the l2norm when
/// asked); then the gate's gradient summed back from the chunk's last row, and `beta`'s. Eight
/// waves: wave `w` forms row tile `w % 4` and column tiles `2 (w / 4)` and `2 (w / 4) + 1` of
/// `dA`, of its transpose and of `K K^T`, and row tile `w % 4` and half (`w / 4`) of the keys'
/// gradient's columns.
#[cube(launch_unchecked)]
pub(super) fn wmma_wy_kernel<E: Float>(
    k: &Tensor<Vector<E, V8>>,
    knt: &Tensor<Vector<E, V8>>,
    gcum: &Tensor<f32>,
    beta: &Tensor<f32>,
    dvb: &Tensor<Vector<E, V8>>,
    u: &Tensor<Vector<E, V8>>,
    dkbg: &Tensor<Vector<E, V8>>,
    w: &Tensor<Vector<E, V8>>,
    dk_part: &Tensor<f32>,
    rg_part: &Tensor<f32>,
    rb_part: &Tensor<f32>,
    tail: &Tensor<f32>,
    dk_out: &mut Tensor<Vector<E, V8>>,
    dg: &mut Tensor<f32>,
    dbeta: &mut Tensor<f32>,
    plan: &Tensor<u32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    #[comptime] dk: u32,
    #[comptime] dv: u32,
    #[comptime] l2norm: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let ck = comptime!(dk / 8);
    let cv = comptime!(dv / 8);
    let cmax = comptime!(if dk > dv { dk / 8 } else { dv / 8 });
    let kreg = comptime!(if dk >= 128 { 64 * dk / 8 } else { 1024u32 });
    let cube = CUBE_POS_X;
    let c = cube % n_chunks;
    let bh = cube / n_chunks;
    let b = bh / heads;
    let h = bh % heads;
    let (first, count) = chunk_tokens(plan, b as usize, c as usize, seq_len as usize, packed);
    if packed && count == 0usize {
        terminate!();
    }
    let first = first as u32;
    let count = count as u32;
    let tid = UNIT_POS as usize;
    let wave = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let row = lane % 16;
    let half = lane / 16;
    let it = wave % 4;
    let hf = wave / 4;
    let i = 16 * it + row;

    // RA: dV_b, then dKbg, then K^T. RB: U, then W, then (beta K)^T. RK: the keys by rows, then
    // dAe and dAe^T by rows.
    let mut ra =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * cmax) as usize), 16usize);
    let mut rb =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * cmax) as usize), 16usize);
    let mut rk = Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!(kreg as usize), 16usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut bs = Shared::<[f32]>::new_slice(64usize);
    let mut inv_k = Shared::<[f32]>::new_slice(64usize);
    let mut red = Shared::<[f32]>::new_slice(128usize);
    let mut s_ba = Shared::<[f32]>::new_slice(64usize);
    let mut s_dr = Shared::<[f32]>::new_slice(64usize);
    let mut s_dc = Shared::<[f32]>::new_slice(64usize);
    let mut s_pk = Shared::<[f32]>::new_slice(64usize);
    let mut rg = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        gs[tid] = gcum[cube as usize * 64usize + tid];
        let mut x = 0.0f32;
        if (tid as u32) < count {
            x = beta[(first as usize + tid) * heads as usize + h as usize];
        }
        bs[tid] = x;
    }
    stage_copy::<E>(
        dvb,
        ra.as_mut_slice(),
        cube * 64 * cv,
        comptime!(64 * cv),
        cv,
        EIGHT_WAVES,
    );
    stage_copy::<E>(
        u,
        rb.as_mut_slice(),
        cube * 64 * cv,
        comptime!(64 * cv),
        cv,
        EIGHT_WAVES,
    );
    stage_scaled::<E>(
        k,
        rk.as_mut_slice(),
        inv_k.as_mut_slice(),
        first,
        count,
        h,
        heads,
        1.0f32,
        dk,
        l2norm,
        EIGHT_WAVES,
    );
    sync_cube();

    // dV_b U^T and its transpose; K K^T.
    let mut acc_a = zeros(2u32);
    let mut acc_at = zeros(2u32);
    let mut acc_kk = zeros(2u32);
    #[unroll]
    for kk in 0..comptime!(dv / 16) {
        let xb = frag::<E>(ra.as_slice(), 16 * it, kk, cv);
        let xu = frag::<E>(rb.as_slice(), 16 * it, kk, cv);
        #[unroll]
        for s in 0..2u32 {
            let jt = 2 * hf + s;
            let a_u = frag::<E>(rb.as_slice(), 16 * jt, kk, cv);
            mma::<E>(&def, &a_u, &xb, &mut acc_a, s);
            let a_b = frag::<E>(ra.as_slice(), 16 * jt, kk, cv);
            mma::<E>(&def, &a_b, &xu, &mut acc_at, s);
        }
    }
    #[unroll]
    for kk in 0..comptime!(dk / 16) {
        let x = frag::<E>(rk.as_slice(), 16 * it, kk, ck);
        #[unroll]
        for s in 0..2u32 {
            let a = frag::<E>(rk.as_slice(), 16 * (2 * hf + s), kk, ck);
            mma::<E>(&def, &a, &x, &mut acc_kk, s);
        }
    }
    sync_cube();
    // dKbg W^T and its transpose.
    stage_copy::<E>(
        dkbg,
        ra.as_mut_slice(),
        cube * 64 * ck,
        comptime!(64 * ck),
        ck,
        EIGHT_WAVES,
    );
    stage_copy::<E>(
        w,
        rb.as_mut_slice(),
        cube * 64 * ck,
        comptime!(64 * ck),
        ck,
        EIGHT_WAVES,
    );
    sync_cube();
    #[unroll]
    for kk in 0..comptime!(dk / 16) {
        let xb = frag::<E>(ra.as_slice(), 16 * it, kk, ck);
        let xw = frag::<E>(rb.as_slice(), 16 * it, kk, ck);
        #[unroll]
        for s in 0..2u32 {
            let jt = 2 * hf + s;
            let a_w = frag::<E>(rb.as_slice(), 16 * jt, kk, ck);
            mma::<E>(&def, &a_w, &xb, &mut acc_a, s);
            let a_b = frag::<E>(ra.as_slice(), 16 * jt, kk, ck);
            mma::<E>(&def, &a_b, &xw, &mut acc_at, s);
        }
    }
    sync_cube();

    // dAe = -dA exp(G_i - G_j) (i > j) and its transpose, by rows where the keys were; the
    // gate's and beta's parts. K^T where dKbg was, (beta K)^T where W was.
    let gi = gs[i as usize];
    let bi = bs[i as usize];
    let mut p_ba = 0.0f32;
    let mut p_dr = 0.0f32;
    let mut p_dc = 0.0f32;
    #[unroll]
    for s in 0..2u32 {
        let jt = 2 * hf + s;
        #[unroll]
        for e in 0..8u32 {
            let j = 16 * jt + 8 * half + e;
            let at = (8 * s + e) as usize;
            let gj = gs[j as usize];
            let kk = acc_kk[at].extract(0usize);
            let d = select(
                i > j,
                (0.0f32 - acc_a[at].extract(0usize)) * (gi - gj).exp(),
                0.0f32,
            );
            let dt = select(
                j > i,
                (0.0f32 - acc_at[at].extract(0usize)) * (gj - gi).exp(),
                0.0f32,
            );
            let dkk = d * kk;
            p_ba += dkk;
            p_dr += dkk * bi;
            p_dc += dt * kk * bs[j as usize];
            acc_a[at] = Vector::new(d);
            acc_at[at] = Vector::new(dt);
        }
        rk[swz(i, 2 * jt + half, 8u32) as usize] = tile_as::<E>(&acc_a, s);
        rk[(512 + swz(i, 2 * jt + half, 8u32)) as usize] = tile_as::<E>(&acc_at, s);
    }
    stage_copy::<E>(
        knt,
        ra.as_mut_slice(),
        cube * dk * 8,
        comptime!(dk * 8),
        8u32,
        EIGHT_WAVES,
    );
    #[unroll]
    for m in 0..comptime!((dk * 8 + EIGHT_WAVES - 1) / EIGHT_WAVES) {
        let idx = m * EIGHT_WAVES + UNIT_POS;
        if idx < comptime!(dk * 8) {
            let x = Vector::<f32, V8>::cast_from(knt[(cube * dk * 8 + idx) as usize]);
            rb[swizzle(idx / 8, idx % 8, 8u32) as usize] =
                Vector::cast_from(x * eight(bs.as_slice(), 8 * (idx % 8)));
        }
    }
    row_total(p_ba, it, hf, red.as_mut_slice(), s_ba.as_mut_slice());
    row_total(p_dr, it, hf, red.as_mut_slice(), s_dr.as_mut_slice());
    row_total(p_dc, it, hf, red.as_mut_slice(), s_dc.as_mut_slice());

    // dAe K (dAe's tiles left of the diagonal) and dAe^T (beta K) (its transpose's right).
    let ndt = comptime!(dk / 32);
    let mut acc1 = zeros(ndt);
    let mut acc2 = zeros(ndt);
    #[unroll]
    for kt in 0..4u32 {
        if kt <= it {
            let x = fragb::<E>(rk.as_slice(), 0u32, 16 * it, kt, 8u32);
            #[unroll]
            for s in 0..ndt {
                let a = frag::<E>(ra.as_slice(), 16 * (hf * ndt + s), kt, 8u32);
                mma::<E>(&def, &a, &x, &mut acc1, s);
            }
        }
        if kt >= it {
            let x = fragb::<E>(rk.as_slice(), 512u32, 16 * it, kt, 8u32);
            #[unroll]
            for s in 0..ndt {
                let a = frag::<E>(rb.as_slice(), 16 * (hf * ndt + s), kt, 8u32);
                mma::<E>(&def, &a, &x, &mut acc2, s);
            }
        }
    }

    // dk = dK + beta (dAe K) + dAe^T (beta K), through the l2norm: inv (dk - k (k . dk)), the
    // normalised keys read again as the tiles held them.
    let mut ys = Array::<Vector<f32, V8>>::new(comptime!(ndt as usize));
    let mut krow = Array::<Vector<E, V8>>::new(comptime!(ndt as usize));
    let mut p_k = 0.0f32;
    #[unroll]
    for s in 0..ndt {
        let cc = 2 * (hf * ndt + s) + half;
        let d0 = 8 * cc;
        let mut part = Vector::<f32, V8>::new(0.0f32);
        #[unroll]
        for e in 0..8u32 {
            part.insert(
                e as usize,
                dk_part[((cube * 64 + i) * dk + d0 + e) as usize],
            );
        }
        let y = part + tile_f32(&acc1, s) * Vector::new(bi) + tile_f32(&acc2, s);
        ys[s as usize] = y;
        if comptime!(l2norm) {
            let mut x = Vector::<E, V8>::new(E::from_int(0));
            if i < count {
                x = k[(((first + i) * heads + h) * ck + cc) as usize];
            }
            let kn = Vector::<E, V8>::cast_from(
                Vector::<f32, V8>::cast_from(x) * Vector::new(inv_k[i as usize]),
            );
            krow[s as usize] = kn;
            p_k += dot8::<E>(y, kn);
        }
    }
    if comptime!(l2norm) {
        row_total(p_k, it, hf, red.as_mut_slice(), s_pk.as_mut_slice());
    }
    #[unroll]
    for s in 0..ndt {
        let cc = 2 * (hf * ndt + s) + half;
        let mut x = ys[s as usize];
        if comptime!(l2norm) {
            let kk = Vector::<f32, V8>::cast_from(krow[s as usize]);
            x = (x - kk * Vector::new(s_pk[i as usize])) * Vector::new(inv_k[i as usize]);
        }
        if i < count {
            dk_out[(((first + i) * heads + h) * ck + cc) as usize] = Vector::cast_from(x);
        }
    }

    // The gate's gradient, the reverse cumulative sum within the chunk from its tail (unit `t`
    // scans row 63 - t, the first two waves each 32 rows, the second adding the first's total);
    // beta's.
    if tid < 64usize {
        let r = 63usize - tid;
        let mut x = rg_part[cube as usize * 64usize + r] + (s_dr[r] - s_dc[r]);
        if tid == 0usize {
            x += tail[cube as usize];
        }
        rg[tid] = plane_inclusive_sum(x);
        if (tid as u32) < count {
            dbeta[(first as usize + tid) * heads as usize + h as usize] =
                rb_part[cube as usize * 64usize + tid] + s_ba[tid];
        }
    }
    sync_cube();
    if tid < 64usize {
        let r = 63usize - tid;
        let mut x = rg[tid];
        if tid >= 32usize {
            x += rg[31usize];
        }
        if (r as u32) < count {
            dg[(first as usize + r) * heads as usize + h as usize] = x;
        }
    }
}
