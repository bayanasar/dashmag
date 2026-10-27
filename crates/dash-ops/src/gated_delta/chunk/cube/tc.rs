//! The kernels for bf16 and f16 inputs, whose matrix products run on tensor cores
//! (`mma.sync.m16n8k16` from the input dtype into f32, operands read from shared memory with
//! `ldmatrix`), for `Dk` and `Dv` of 32, 64 or 128. They compute what the f32 kernels of the
//! parent module compute, in the same chunked layouts and through the same per-chunk quantities
//! (notation of `reference.rs`), but keep the intermediates that only feed products in the input
//! dtype, as fla does: `W`, `U` and the normalised keys `K`, `T = (I + A)^-1`, the state entering
//! each chunk and `V'`; in the backward pass `qg = gamma Q`, the chunked output gradient, the
//! state's gradient leaving each chunk, `dV'`, `dV_b` and `dKbg`. Everything else (the decays,
//! `A` before its inversion and the inversion itself, the state carried from chunk to chunk and
//! its gradient, every sum the gradients are made of) stays f32. The state passings fold the
//! decay `lam = exp(G_last - G)` into `V'` (and `dS`'s product into `dV'`), as fla does, rather
//! than into the keys.
//!
//! Forward: [`tc_prep_kernel`] (keeping `T` for the backward pass when asked),
//! [`tc_state_forward_kernel`], [`tc_output_kernel`]. Backward: [`tc_wu_kernel`] (`W`, `U` and the
//! keys from the kept `T`; the prep kernel again where none was kept), the forward state passing
//! again, [`tc_dvlocal_kernel`], [`tc_state_backward_kernel`], then [`tc_dqk_kernel`], everything
//! but the WY part in one kernel (`dvb_kernel`, `dstate_kernel`, `dattn_kernel` and the queries'
//! side of `combine_kernel` of the f32 set), and [`tc_wy_kernel`], the WY part and the rest of the
//! sums (`dwy_kernel` and the keys', the gate's and `beta`'s side of `combine_kernel`): their
//! partial gradients stay in registers, and only the keys' gradient without the WY part, `dV_b`,
//! `dKbg` and per-token sums pass between the two. The kernels with several phases read the next
//! phase's tiles into registers while they compute on the current one.
//!
//! Tiles of 16-bit values are staged in shared memory as rows of 16-byte chunks (eight
//! elements), XOR-swizzled so that the eight rows an `ldmatrix` reads at one chunk fall in eight
//! different bank groups (`attention/kernels.rs` stages its tiles the same way). A tile is
//! declared either as 8-element vectors (staged from global memory 16 bytes at a time) or as
//! 2-element ones (written from the accumulators, whose lanes hold pairs of columns); the
//! loaders below take either, `per` being the vectors in a chunk.
//!
//! Accumulators are held in the register layout PTX gives `mma.m16n8k16`: lane `l` holds rows
//! `l / 4` and `l / 4 + 8` and columns `2 (l % 4)`, `2 (l % 4) + 1` of each 16 x 8 tile, one
//! 2-element vector per row; this is also the layout of an A operand, so that a tile of `P`
//! becomes the left operand of the next product without leaving registers.
//!
//! The per-chunk kernels run eight warps (256 units) per chunk of each head; warp `w` takes the
//! 16 rows `16 (w % 4)` of a chunk's 64-row products and half of their columns (`w / 4`), or all
//! 64 columns of the 64 x 64 ones. The state passings run four warps per (sequence and head,
//! block of 32 state columns); warp `w` takes 16 rows of `V'` and a quarter of the state's rows.

// Kernel code is expanded by CubeCL's frontend, which takes the plain operators.
#![allow(
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::collapsible_else_if
)]

use super::{chunk_tokens, state_chunks};
use cubecl::cmma::{MatrixIdent, MmaDefinition};
use cubecl::prelude::*;

/// Eight 16-bit elements: one 16-byte chunk, one row of an `ldmatrix` 8 x 8 matrix.
pub(super) type V8 = Const<8>;
/// Two elements: one 32-bit MMA register of bf16 or f16, or two f32 accumulator columns.
pub(super) type R2 = Const<2>;

/// Units of a per-chunk kernel: eight warps.
pub(super) const CHUNK_UNITS: u32 = 256;
/// Units of a state passing: four warps.
pub(super) const STATE_UNITS: u32 = 128;
/// State columns (of `Dv`) one state passing cube carries.
pub(super) const STATE_COLS: usize = 32;

/// The index of 16-byte chunk `c` of row `r` in a swizzled tile of `chunks` chunks per row.
/// Rows of 8 or more chunks XOR the chunk with the row's low three bits; rows of 4 chunks (32
/// columns) with bits 1 and 2, which spreads eight consecutive 64-byte rows over eight bank
/// groups as well.
#[cube]
fn swizzle(r: u32, c: u32, #[comptime] chunks: u32) -> u32 {
    if comptime!(chunks >= 8) {
        r * chunks + (c ^ (r & 7))
    } else {
        r * chunks + (c ^ ((r >> 1) & 3))
    }
}

/// The A operand (16 x 16) at rows `[m0, m0 + 16)` and columns `[16 kk, 16 kk + 16)` of a tile
/// whose rows are A's rows.
#[cube]
fn a_rows<E: Float, N: Size>(
    def: &MmaDefinition<E, E, f32>,
    tile: &[Vector<E, N>],
    m0: u32,
    kk: u32,
    #[comptime] chunks: u32,
    #[comptime] per: u32,
) -> Array<Vector<E, R2>> {
    let lane = UNIT_POS_PLANE;
    let idx = (swizzle(m0 + lane % 16, 2 * kk + lane / 16, chunks) * per) as usize;
    def.load_matrix::<Vector<E, N>, R2>(
        &tile[idx..idx + per as usize],
        MatrixIdent::A,
        4usize,
        false,
    )
}

/// The A operand (16 x 16) of `X^T`, X a tile whose rows are A's columns: A's rows are X's
/// columns `[m0, m0 + 16)`, A's columns X's rows `[k0, k0 + 16)`.
#[cube]
fn a_cols<E: Float, N: Size>(
    def: &MmaDefinition<E, E, f32>,
    tile: &[Vector<E, N>],
    m0: u32,
    k0: u32,
    #[comptime] chunks: u32,
    #[comptime] per: u32,
) -> Array<Vector<E, R2>> {
    let lane = UNIT_POS_PLANE;
    let matrix = lane / 8;
    let idx = (swizzle(
        k0 + lane % 8 + 8 * (matrix / 2),
        m0 / 8 + matrix % 2,
        chunks,
    ) * per) as usize;
    def.load_matrix::<Vector<E, N>, R2>(
        &tile[idx..idx + per as usize],
        MatrixIdent::A,
        4usize,
        true,
    )
}

/// B operands of `X^T` for two adjacent 8-column tiles, X a tile whose rows are B's columns:
/// X's rows `[n0, n0 + 16)`, columns `[16 kk, 16 kk + 16)`. Returns the two registers of the
/// first tile, then the two of the second.
#[cube]
fn b_rows<E: Float, N: Size>(
    def: &MmaDefinition<E, E, f32>,
    tile: &[Vector<E, N>],
    n0: u32,
    kk: u32,
    #[comptime] chunks: u32,
    #[comptime] per: u32,
) -> Array<Vector<E, R2>> {
    let lane = UNIT_POS_PLANE;
    let idx = (swizzle(
        n0 + lane % 8 + 8 * (lane / 16),
        2 * kk + (lane / 8) % 2,
        chunks,
    ) * per) as usize;
    def.load_matrix::<Vector<E, N>, R2>(
        &tile[idx..idx + per as usize],
        MatrixIdent::B,
        4usize,
        false,
    )
}

/// B operands of X for two adjacent 8-column tiles, X a tile whose rows are B's rows: X's rows
/// `[k0, k0 + 16)`, columns `[n0, n0 + 16)`.
#[cube]
fn b_cols<E: Float, N: Size>(
    def: &MmaDefinition<E, E, f32>,
    tile: &[Vector<E, N>],
    k0: u32,
    n0: u32,
    #[comptime] chunks: u32,
    #[comptime] per: u32,
) -> Array<Vector<E, R2>> {
    let lane = UNIT_POS_PLANE;
    let matrix = lane / 8;
    let idx = (swizzle(
        k0 + lane % 8 + 8 * (matrix % 2),
        n0 / 8 + matrix / 2,
        chunks,
    ) * per) as usize;
    def.load_matrix::<Vector<E, N>, R2>(
        &tile[idx..idx + per as usize],
        MatrixIdent::B,
        4usize,
        true,
    )
}

/// `acc[tile] += a b`, `b` given as its two registers, `acc` holding two vectors per tile.
#[cube]
fn mma<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    a: &Array<Vector<E, R2>>,
    b0: Vector<E, R2>,
    b1: Vector<E, R2>,
    acc: &mut Array<Vector<f32, R2>>,
    tile: u32,
) {
    let mut b = Array::<Vector<E, R2>>::new(2usize);
    b[0] = b0;
    b[1] = b1;
    let mut c = Array::<Vector<f32, R2>>::new(2usize);
    c[0] = acc[(2 * tile) as usize];
    c[1] = acc[(2 * tile + 1) as usize];
    let out = def.execute(a, &b, &c);
    acc[(2 * tile) as usize] = out[0];
    acc[(2 * tile + 1) as usize] = out[1];
}

/// `acc[tiles from tile0] += A B` over the `ksteps` 16-wide steps from `k0`, for `pairs` pairs
/// of 8-column tiles from column `n0`: A from a tile whose rows are A's rows (`a_t` false) or
/// its columns (`a_t` true), B from a tile whose rows are B's columns (`b_t` true) or its rows.
#[cube]
#[allow(clippy::too_many_arguments)]
fn gemm<E: Float, NA: Size, NB: Size>(
    def: &MmaDefinition<E, E, f32>,
    a: &[Vector<E, NA>],
    b: &[Vector<E, NB>],
    acc: &mut Array<Vector<f32, R2>>,
    m0: u32,
    n0: u32,
    k0: u32,
    tile0: u32,
    #[comptime] ksteps: u32,
    #[comptime] pairs: u32,
    #[comptime] a_t: bool,
    #[comptime] b_t: bool,
    #[comptime] a_chunks: u32,
    #[comptime] a_per: u32,
    #[comptime] b_chunks: u32,
    #[comptime] b_per: u32,
) {
    gemm_at::<E, NA, NB>(
        def, a, b, acc, m0, n0, k0, k0, tile0, ksteps, pairs, a_t, b_t, a_chunks, a_per, b_chunks,
        b_per,
    );
}

/// [`gemm`] with the inner dimension at `ka0` in A's tile and `kb0` in B's (one tile holding the
/// whole inner dimension, the other a block of it).
#[cube]
#[allow(clippy::too_many_arguments)]
fn gemm_at<E: Float, NA: Size, NB: Size>(
    def: &MmaDefinition<E, E, f32>,
    a: &[Vector<E, NA>],
    b: &[Vector<E, NB>],
    acc: &mut Array<Vector<f32, R2>>,
    m0: u32,
    n0: u32,
    ka0: u32,
    kb0: u32,
    tile0: u32,
    #[comptime] ksteps: u32,
    #[comptime] pairs: u32,
    #[comptime] a_t: bool,
    #[comptime] b_t: bool,
    #[comptime] a_chunks: u32,
    #[comptime] a_per: u32,
    #[comptime] b_chunks: u32,
    #[comptime] b_per: u32,
) {
    #[unroll]
    for kk in 0..ksteps {
        let ka = ka0 + 16 * kk;
        let kb = kb0 + 16 * kk;
        if comptime!(a_t) {
            let ra = a_cols::<E, NA>(def, a, m0, ka, a_chunks, a_per);
            gemm_step::<E, NB>(def, &ra, b, acc, n0, kb, tile0, pairs, b_t, b_chunks, b_per);
        } else {
            let ra = a_rows::<E, NA>(def, a, m0, ka / 16, a_chunks, a_per);
            gemm_step::<E, NB>(def, &ra, b, acc, n0, kb, tile0, pairs, b_t, b_chunks, b_per);
        }
    }
}

/// One 16-wide step of [`gemm`]: `acc += ra B[k..k + 16, ..]` for `pairs` pairs of tiles.
#[cube]
#[allow(clippy::too_many_arguments)]
fn gemm_step<E: Float, NB: Size>(
    def: &MmaDefinition<E, E, f32>,
    ra: &Array<Vector<E, R2>>,
    b: &[Vector<E, NB>],
    acc: &mut Array<Vector<f32, R2>>,
    n0: u32,
    k: u32,
    tile0: u32,
    #[comptime] pairs: u32,
    #[comptime] b_t: bool,
    #[comptime] b_chunks: u32,
    #[comptime] b_per: u32,
) {
    #[unroll]
    for np in 0..pairs {
        if comptime!(b_t) {
            let rb = b_rows::<E, NB>(def, b, n0 + 16 * np, k / 16, b_chunks, b_per);
            mma::<E>(def, ra, rb[0], rb[1], acc, tile0 + 2 * np);
            mma::<E>(def, ra, rb[2], rb[3], acc, tile0 + 2 * np + 1);
        } else {
            let rb = b_cols::<E, NB>(def, b, k, n0 + 16 * np, b_chunks, b_per);
            mma::<E>(def, ra, rb[0], rb[1], acc, tile0 + 2 * np);
            mma::<E>(def, ra, rb[2], rb[3], acc, tile0 + 2 * np + 1);
        }
    }
}

/// The A operand made of accumulator tiles `tile` and `tile + 1` (16 rows x 16 columns),
/// converted to the input dtype: the layouts agree, so no data moves between lanes.
#[cube]
fn accumulator_as_a<E: Float>(acc: &Array<Vector<f32, R2>>, tile: u32) -> Array<Vector<E, R2>> {
    let mut a = Array::<Vector<E, R2>>::new(4usize);
    #[unroll]
    for i in 0..4u32 {
        a[i as usize] = Vector::cast_from(acc[(2 * tile + i) as usize]);
    }
    a
}

#[cube]
fn zeros(#[comptime] len: u32) -> Array<Vector<f32, R2>> {
    let mut acc = Array::<Vector<f32, R2>>::new(len as usize);
    #[unroll]
    for i in 0..len {
        acc[i as usize] = Vector::new(0.0f32);
    }
    acc
}

/// Stages a chunk's rows of head `h` of a `[B T, H, D]` tensor (read 16 bytes at a time) into
/// the swizzled tile `dst` of 64 rows of `D / 8` chunks: its `count` tokens from token `first`,
/// zero past them; each row times `scale`, over its length when `l2norm` (transformers' l2norm,
/// `x / sqrt(sum x^2 + 1e-6)`, summed over the lanes holding the row), and times `rows[r]` when
/// `scaled`. The whole cube of 256 units takes part.
#[cube]
#[allow(clippy::too_many_arguments)]
fn stage_tokens<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    rows: &[f32],
    first: u32,
    count: u32,
    h: u32,
    heads: u32,
    scale: f32,
    #[comptime] d: u32,
    #[comptime] l2norm: bool,
    #[comptime] scaled: bool,
) {
    let chunks = comptime!(d / 8);
    #[unroll]
    for i in 0..comptime!(64 * d / 8 / 256) {
        let idx = i * 256 + UNIT_POS;
        let r = idx / chunks;
        let c = idx % chunks;
        let mut x = Vector::<E, V8>::new(E::from_int(0));
        if r < count {
            x = src[(((first + r) * heads + h) * chunks + c) as usize];
        }
        let f = Vector::<f32, V8>::cast_from(x);
        let mut factor = scale;
        if comptime!(l2norm) {
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
        if comptime!(scaled) {
            factor *= rows[r as usize];
        }
        dst[swizzle(r, c, chunks) as usize] = Vector::cast_from(f * Vector::new(factor));
    }
}

/// Stages rows of a chunked `[.., 64, D]` tensor of the input dtype (chunk `chunk`, read 16
/// bytes at a time) into the swizzled tile `dst`, `units` units taking part; with `rows`, rows
/// `r` of the tile are rows `r` of the chunk (`D / 8` chunks per row).
#[cube]
fn stage_chunk<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    chunk: u32,
    #[comptime] rows: u32,
    #[comptime] d: u32,
    #[comptime] units: u32,
) {
    let chunks = comptime!(d / 8);
    let total = comptime!(rows * d / 8);
    #[unroll]
    for i in 0..comptime!(total.div_ceil(units)) {
        let idx = i * units + UNIT_POS;
        if idx < total {
            let r = idx / chunks;
            let c = idx % chunks;
            dst[swizzle(r, c, chunks) as usize] = src[((chunk * rows + r) * chunks + c) as usize];
        }
    }
}

/// Per chunk, everything the state passing needs that does not depend on the state, as
/// `prep_kernel` forms it: `G` (f32), `T = (I + A)^-1` (in the input dtype, for the backward
/// pass, when `store_t`), `W = T (beta gamma K)`, `U = T (beta V)` and the keys `K` (normalised
/// when asked), these three in the input dtype. `A = beta (K K^T) exp(G_i - G_j)` (`i > j`) and
/// the three products run on tensor cores; the inversion is `prep_kernel`'s, in f32.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tc_prep_kernel<E: Float>(
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    g: &Tensor<f32>,
    beta: &Tensor<f32>,
    w: &mut Tensor<Vector<E, R2>>,
    u: &mut Tensor<Vector<E, R2>>,
    kn: &mut Tensor<Vector<E, V8>>,
    gcum: &mut Tensor<f32>,
    tmat: &mut Tensor<Vector<E, V8>>,
    plan: &Tensor<u32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    #[comptime] dk: u32,
    #[comptime] dv: u32,
    #[comptime] l2norm: bool,
    #[comptime] store_t: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
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
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;

    let mut s_k =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    // beta gamma K for W, then beta V for U.
    let mut s_x =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * cmax) as usize), 16usize);
    let mut s_t = Shared::<[Vector<E, V8>]>::new_aligned_slice(512usize, 16usize);
    let mut am = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut tm = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut bs = Shared::<[f32]>::new_slice(64usize);
    let mut fs = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        let mut gv = 0.0f32;
        let mut bv = 0.0f32;
        if (tid as u32) < count {
            gv = g[((first as usize + tid) * heads as usize) + h as usize];
            bv = beta[((first as usize + tid) * heads as usize) + h as usize];
        }
        gs[tid] = gv;
        bs[tid] = bv;
    }
    stage_tokens::<E>(
        k,
        s_k.as_mut_slice(),
        fs.as_slice(),
        first,
        count,
        h,
        heads,
        1.0f32,
        dk,
        l2norm,
        false,
    );
    // The values read into registers while A and its inverse are formed.
    let pre_v = fetch_tokens::<E>(v, first, count, h, heads, dv, 256u32);
    sync_cube();
    // G = cumsum(g): each of the first two warps scans its 32 tokens, the second adding the
    // first's total.
    if tid < 64usize {
        fs[tid] = plane_inclusive_sum(gs[tid]);
    }
    sync_cube();
    if tid < 64usize {
        let mut x = fs[tid];
        if tid >= 32usize {
            x += fs[31usize];
        }
        gs[tid] = x;
    }
    // The keys as the state passings read them.
    #[unroll]
    for i in 0..comptime!(64 * dk / 8 / 256) {
        let idx = i * 256 + UNIT_POS;
        let r = idx / ck;
        let cc = idx % ck;
        kn[((cube * 64 + r) * ck + cc) as usize] = s_k[swizzle(r, cc, ck) as usize];
    }
    sync_cube();
    if tid < 64usize {
        fs[tid] = bs[tid] * gs[tid].exp();
        gcum[cube as usize * 64usize + tid] = gs[tid];
    }

    // A[i, j] = beta_i (k_i . k_j) exp(G_i - G_j), i > j: warp w forms rows 16 (w % 4) and
    // columns 32 (w / 4) of K K^T.
    let mrow = (warp % 4) * 16;
    let ncol = (warp / 4) * 32;
    let mut acc = zeros(8u32);
    gemm::<E, V8, V8>(
        &def,
        s_k.as_slice(),
        s_k.as_slice(),
        &mut acc,
        mrow,
        ncol,
        0u32,
        0u32,
        comptime!(dk / 16),
        2u32,
        false,
        true,
        ck,
        1u32,
        ck,
        1u32,
    );
    #[unroll]
    for nt in 0..4u32 {
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            let val = acc[(2 * nt + half) as usize];
            #[unroll]
            for e in 0..2u32 {
                let j = ncol + nt * 8 + 2 * tq + e;
                let mut x = 0.0f32;
                if i > j {
                    x = bs[i as usize]
                        * val.extract(e as usize)
                        * (gs[i as usize] - gs[j as usize]).exp();
                }
                am[(i * 65 + j) as usize] = x;
            }
        }
    }
    #[unroll]
    for m_ in 0..comptime![(4160usize).div_ceil(256usize)] {
        let e = tid + 256usize * m_;
        if e < 4160usize {
            tm[e] = 0.0f32;
        }
    }
    sync_cube();

    // T = (I + A)^-1, as prep_kernel forms it: the four diagonal blocks of 16 by forward
    // substitution, one column per unit; then block row r: T[r, c] = -T[r, r] sum_{c<=m<r}
    // A[r, m] T[m, c].
    if tid < 64usize {
        let blk0 = (tid / 16usize) * 16usize;
        let j = tid;
        for il in 0..16usize {
            let i = blk0 + il;
            let mut x = 0.0f32;
            if i == j {
                x = 1.0f32;
            } else if i > j {
                x = 0.0f32 - am[i * 65usize + j];
                let mut l = j + 1usize;
                while l < i {
                    x -= am[i * 65usize + l] * tm[l * 65usize + j];
                    l += 1usize;
                }
            }
            tm[i * 65usize + j] = x;
        }
    }
    // beta gamma K for W.
    #[unroll]
    for i in 0..comptime!(64 * dk / 8 / 256) {
        let idx = i * 256 + UNIT_POS;
        let r = idx / ck;
        let at = swizzle(r, idx % ck, ck) as usize;
        let x = Vector::<f32, V8>::cast_from(s_k[at]);
        s_x[at] = Vector::cast_from(x * Vector::new(fs[r as usize]));
    }
    sync_cube();
    let il = tid / 16usize;
    let jl = tid % 16usize;
    for r in 1..4usize {
        // X[c] = sum_{c<=m<r} A[r, m] T[m, c], stashed in the strictly upper part of `am`.
        let i = r * 16usize + il;
        for cc in 0..r {
            let mut x = 0.0f32;
            let mut l = cc * 16usize;
            while l < r * 16usize {
                x += am[i * 65usize + l] * tm[l * 65usize + cc * 16usize + jl];
                l += 1usize;
            }
            am[il * 65usize + 16usize + cc * 16usize + jl] = x;
        }
        sync_cube();
        for cc in 0..r {
            let mut x = 0.0f32;
            for l in 0..16usize {
                x -= tm[i * 65usize + r * 16usize + l]
                    * am[l * 65usize + 16usize + cc * 16usize + jl];
            }
            tm[i * 65usize + cc * 16usize + jl] = x;
        }
        sync_cube();
    }

    // T in the input dtype: the tile, and for the backward pass the chunk's T.
    #[unroll]
    for i in 0..2u32 {
        let idx = i * 256 + UNIT_POS;
        let r = idx / 8;
        let cc = idx % 8;
        let mut x = Vector::<f32, V8>::new(0.0f32);
        #[unroll]
        for e in 0..8u32 {
            x.insert(e as usize, tm[(r * 65 + cc * 8 + e) as usize]);
        }
        let y = Vector::<E, V8>::cast_from(x);
        s_t[swizzle(r, cc, 8u32) as usize] = y;
        if comptime!(store_t) {
            tmat[((cube * 64 + r) * 8 + cc) as usize] = y;
        }
    }
    sync_cube();

    // W = T (beta gamma K): warp w forms rows 16 (w % 4) and columns Dk / 2 (w / 4).
    let wk = comptime!(dk / 2);
    let mut acc_w = zeros(comptime!(dk / 8));
    gemm::<E, V8, V8>(
        &def,
        s_t.as_slice(),
        s_x.as_slice(),
        &mut acc_w,
        mrow,
        (warp / 4) * wk,
        0u32,
        0u32,
        4u32,
        comptime!(dk / 32),
        false,
        false,
        8u32,
        1u32,
        ck,
        1u32,
    );
    #[unroll]
    for nt in 0..comptime!(dk / 16) {
        let col = (warp / 4) * wk + nt * 8 + 2 * tq;
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            w[(((cube * 64 + i) * dk + col) / 2) as usize] =
                Vector::cast_from(acc_w[(2 * nt + half) as usize]);
        }
    }
    sync_cube();

    // U = T (beta V).
    store_tokens_scaled::<E>(&pre_v, s_x.as_mut_slice(), bs.as_slice(), dv);
    sync_cube();
    let wv = comptime!(dv / 2);
    let mut acc_u = zeros(comptime!(dv / 8));
    gemm::<E, V8, V8>(
        &def,
        s_t.as_slice(),
        s_x.as_slice(),
        &mut acc_u,
        mrow,
        (warp / 4) * wv,
        0u32,
        0u32,
        4u32,
        comptime!(dv / 32),
        false,
        false,
        8u32,
        1u32,
        cv,
        1u32,
    );
    #[unroll]
    for nt in 0..comptime!(dv / 16) {
        let col = (warp / 4) * wv + nt * 8 + 2 * tq;
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            u[(((cube * 64 + i) * dv + col) / 2) as usize] =
                Vector::cast_from(acc_u[(2 * nt + half) as usize]);
        }
    }
}

/// The state passing, forward, as `state_forward_kernel` does it: for each of the sequence's
/// chunks in turn, store the entering state (in the input dtype), form `V' = U - W S`, then
/// `S = exp(G_last) S + K^T (lam V')`, on tensor cores, the state f32 in registers. One cube of
/// four warps per (sequence and head, block of 32 state columns); warp `w` forms rows `16 w` of
/// `V'` and a quarter of the state's rows (with `Dk = 32`, half of its rows and of its columns).
/// The next chunk's `W` and `K` are read into registers during the current one.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tc_state_forward_kernel<E: Float>(
    kn: &Tensor<Vector<E, V8>>,
    w: &Tensor<Vector<E, V8>>,
    u: &Tensor<Vector<E, R2>>,
    gcum: &Tensor<f32>,
    h0: &Tensor<Vector<f32, R2>>,
    offsets: &Tensor<u32>,
    states: &mut Tensor<Vector<E, R2>>,
    v_new: &mut Tensor<Vector<E, R2>>,
    last: &mut Tensor<Vector<f32, R2>>,
    n_chunks: u32,
    heads: u32,
    #[comptime] dk: u32,
    #[comptime] dv: u32,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
    let ck = comptime!(dk / 8);
    let y = CUBE_POS_Y;
    let v0 = CUBE_POS_X * 32;
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;
    let (base, n) = state_chunks(
        offsets,
        y as usize,
        heads as usize,
        n_chunks as usize,
        packed,
    );
    let base = base as u32;
    let n = n as u32;

    // The state's tiles this warp holds: `mts` row tiles from `mt0`, `nts` column tiles from
    // `nt0`.
    let mts = comptime!(if dk >= 64 { dk / 64 } else { 1u32 });
    let nts = comptime!(if dk >= 64 { 4u32 } else { 2u32 });
    let mut mt0 = warp % 2;
    let mut nt0 = 2 * (warp / 2);
    if comptime!(dk >= 64) {
        mt0 = warp * mts;
        nt0 = 0u32;
    }

    let mut s_w =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    let mut s_k =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    // The state in the input dtype, `Dk` rows of 32 columns, and `lam V'`, 64 rows of 32.
    let mut s_s =
        Shared::<[Vector<E, R2>]>::new_aligned_slice(comptime!((dk * 16) as usize), 16usize);
    let mut s_v = Shared::<[Vector<E, R2>]>::new_aligned_slice(1024usize, 16usize);

    let mut sreg = Array::<Vector<f32, R2>>::new(comptime!((mts * nts * 2) as usize));
    #[unroll]
    for mi in 0..mts {
        #[unroll]
        for ni in 0..nts {
            #[unroll]
            for half in 0..2u32 {
                let d = 16 * (mt0 + mi) + gq + 8 * half;
                let col = v0 + 8 * (nt0 + ni) + 2 * tq;
                sreg[((mi * nts + ni) * 2 + half) as usize] =
                    h0[(((y * dk + d) * dv + col) / 2) as usize];
            }
        }
    }

    let per = comptime!(64 * dk / 8 / 128);
    let mut pre_w = Array::<Vector<E, V8>>::new(per as usize);
    let mut pre_k = Array::<Vector<E, V8>>::new(per as usize);
    if n > 0 {
        stage_chunk::<E>(w, s_w.as_mut_slice(), base, 64u32, dk, 128u32);
        stage_chunk::<E>(kn, s_k.as_mut_slice(), base, 64u32, dk, 128u32);
    }

    for c in 0..n {
        let chunk = base + c;
        // The entering state, stored and staged.
        #[unroll]
        for mi in 0..mts {
            #[unroll]
            for ni in 0..nts {
                #[unroll]
                for half in 0..2u32 {
                    let d = 16 * (mt0 + mi) + gq + 8 * half;
                    let x = Vector::<E, R2>::cast_from(sreg[((mi * nts + ni) * 2 + half) as usize]);
                    states[(((chunk * dk + d) * dv + v0 + 8 * (nt0 + ni) + 2 * tq) / 2) as usize] =
                        x;
                    s_s[(swizzle(d, nt0 + ni, 4u32) * 4 + tq) as usize] = x;
                }
            }
        }
        // This chunk's U and decays, and the next chunk's W and K, into registers.
        let mut ureg = Array::<Vector<E, R2>>::new(8usize);
        #[unroll]
        for nt in 0..4u32 {
            #[unroll]
            for half in 0..2u32 {
                let i = 16 * warp + gq + 8 * half;
                ureg[(2 * nt + half) as usize] =
                    u[(((chunk * 64 + i) * dv + v0 + 8 * nt + 2 * tq) / 2) as usize];
            }
        }
        let g_last = gcum[(chunk * 64 + 63) as usize];
        let lam_a = (g_last - gcum[(chunk * 64 + 16 * warp + gq) as usize]).exp();
        let lam_b = (g_last - gcum[(chunk * 64 + 16 * warp + gq + 8) as usize]).exp();
        let more = c + 1 < n;
        if more {
            #[unroll]
            for i in 0..per {
                let idx = i * 128 + UNIT_POS;
                let at = ((chunk + 1) * 64 * ck + idx) as usize;
                pre_w[i as usize] = w[at];
                pre_k[i as usize] = kn[at];
            }
        }
        sync_cube();

        // V' = U - W S.
        let mut acc_v = zeros(8u32);
        gemm::<E, V8, R2>(
            &def,
            s_w.as_slice(),
            s_s.as_slice(),
            &mut acc_v,
            16 * warp,
            0u32,
            0u32,
            0u32,
            comptime!(dk / 16),
            2u32,
            false,
            false,
            ck,
            1u32,
            4u32,
            4u32,
        );
        #[unroll]
        for nt in 0..4u32 {
            #[unroll]
            for half in 0..2u32 {
                let i = 16 * warp + gq + 8 * half;
                let x = Vector::<f32, R2>::cast_from(ureg[(2 * nt + half) as usize])
                    - acc_v[(2 * nt + half) as usize];
                v_new[(((chunk * 64 + i) * dv + v0 + 8 * nt + 2 * tq) / 2) as usize] =
                    Vector::cast_from(x);
                let lam = select(half == 0, lam_a, lam_b);
                s_v[(swizzle(i, nt, 4u32) * 4 + tq) as usize] =
                    Vector::cast_from(x * Vector::new(lam));
            }
        }
        sync_cube();

        // S = exp(G_last) S + K^T (lam V').
        let decay = g_last.exp();
        #[unroll]
        for x in 0..comptime!(mts * nts * 2) {
            sreg[x as usize] = sreg[x as usize] * Vector::new(decay);
        }
        #[unroll]
        for mi in 0..mts {
            gemm::<E, V8, R2>(
                &def,
                s_k.as_slice(),
                s_v.as_slice(),
                &mut sreg,
                16 * (mt0 + mi),
                8 * nt0,
                0u32,
                mi * nts,
                4u32,
                comptime!(nts / 2),
                true,
                false,
                ck,
                1u32,
                4u32,
                4u32,
            );
        }
        sync_cube();
        if more {
            #[unroll]
            for i in 0..per {
                let idx = i * 128 + UNIT_POS;
                let r = idx / ck;
                let at = swizzle(r, idx % ck, ck) as usize;
                s_w[at] = pre_w[i as usize];
                s_k[at] = pre_k[i as usize];
            }
        }
    }

    #[unroll]
    for mi in 0..mts {
        #[unroll]
        for ni in 0..nts {
            #[unroll]
            for half in 0..2u32 {
                let d = 16 * (mt0 + mi) + gq + 8 * half;
                let col = v0 + 8 * (nt0 + ni) + 2 * tq;
                last[(((y * dk + d) * dv + col) / 2) as usize] =
                    sreg[((mi * nts + ni) * 2 + half) as usize];
            }
        }
    }
}

/// Per chunk, the output `O = gamma (Q S) + P V'` with `P = (Q K^T) exp(G_i - G_j)` (`i >= j`),
/// from the raw queries and keys (normalised and scaled here), the entering state and `V'` in
/// the input dtype; written in the queries' dtype in the `[B, T, H, Dv]` layout. Warp `w` forms
/// `P`'s rows `16 (w % 4)` whole, and `O`'s rows there and its columns `Dv / 2 (w / 4)`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tc_output_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    gcum: &Tensor<f32>,
    states: &Tensor<Vector<E, V8>>,
    v_new: &Tensor<Vector<E, V8>>,
    out: &mut Tensor<Vector<E, R2>>,
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
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
    let ck = comptime!(dk / 8);
    let cv = comptime!(dv / 8);
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
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;

    let mut s_q =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    let mut s_k =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    let mut s_s =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((dk * dv / 8) as usize), 16usize);
    let mut s_v =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dv / 8) as usize), 16usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        gs[tid] = gcum[cube as usize * 64usize + tid];
    }
    stage_tokens::<E>(
        q,
        s_q.as_mut_slice(),
        gs.as_slice(),
        first,
        count,
        h,
        heads,
        scale,
        dk,
        l2norm,
        false,
    );
    stage_tokens::<E>(
        k,
        s_k.as_mut_slice(),
        gs.as_slice(),
        first,
        count,
        h,
        heads,
        1.0f32,
        dk,
        l2norm,
        false,
    );
    // The state and V' read into registers while P is formed.
    let pre_s = fetch_block::<E>(states, cube, 0u32, dk, dv, dv);
    let pre_v = fetch_block::<E>(v_new, cube, 0u32, 64u32, dv, dv);
    sync_cube();

    let mrow = (warp % 4) * 16;
    // P = (Q K^T) exp(G_i - G_j), i >= j, the warp's 16 rows whole.
    let mut acc_p = zeros(16u32);
    gemm::<E, V8, V8>(
        &def,
        s_q.as_slice(),
        s_k.as_slice(),
        &mut acc_p,
        mrow,
        0u32,
        0u32,
        0u32,
        comptime!(dk / 16),
        4u32,
        false,
        true,
        ck,
        1u32,
        ck,
        1u32,
    );
    let ia = mrow + gq;
    let ib = ia + 8;
    let ga = gs[ia as usize];
    let gb = gs[ib as usize];
    #[unroll]
    for nt in 0..8u32 {
        let j = nt * 8 + 2 * tq;
        let mut pa = acc_p[(2 * nt) as usize];
        let mut pb = acc_p[(2 * nt + 1) as usize];
        #[unroll]
        for e in 0..2u32 {
            let je = j + e;
            let gj = gs[je as usize];
            pa.insert(
                e as usize,
                select(ia >= je, pa.extract(e as usize) * (ga - gj).exp(), 0.0f32),
            );
            pb.insert(
                e as usize,
                select(ib >= je, pb.extract(e as usize) * (gb - gj).exp(), 0.0f32),
            );
        }
        acc_p[(2 * nt) as usize] = pa;
        acc_p[(2 * nt + 1) as usize] = pb;
    }
    store_block::<E>(&pre_s, s_s.as_mut_slice(), dk, dv);
    store_block::<E>(&pre_v, s_v.as_mut_slice(), 64u32, dv);
    sync_cube();

    // O = gamma (Q S) + P V', the warp's rows and half of the columns.
    let n0 = (warp / 4) * comptime!(dv / 2);
    let mut acc_o = zeros(comptime!(dv / 8));
    gemm::<E, V8, V8>(
        &def,
        s_q.as_slice(),
        s_s.as_slice(),
        &mut acc_o,
        mrow,
        n0,
        0u32,
        0u32,
        comptime!(dk / 16),
        comptime!(dv / 32),
        false,
        false,
        ck,
        1u32,
        cv,
        1u32,
    );
    let gamma_a = ga.exp();
    let gamma_b = gb.exp();
    #[unroll]
    for nt in 0..comptime!(dv / 16) {
        acc_o[(2 * nt) as usize] = acc_o[(2 * nt) as usize] * Vector::new(gamma_a);
        acc_o[(2 * nt + 1) as usize] = acc_o[(2 * nt + 1) as usize] * Vector::new(gamma_b);
    }
    #[unroll]
    for kk in 0..4u32 {
        let a = accumulator_as_a::<E>(&acc_p, 2 * kk);
        #[unroll]
        for np in 0..comptime!(dv / 32) {
            let rb = b_cols::<E, V8>(&def, s_v.as_slice(), 16 * kk, n0 + 16 * np, cv, 1u32);
            mma::<E>(&def, &a, rb[0], rb[1], &mut acc_o, 2 * np);
            mma::<E>(&def, &a, rb[2], rb[3], &mut acc_o, 2 * np + 1);
        }
    }
    #[unroll]
    for nt in 0..comptime!(dv / 16) {
        let col = n0 + nt * 8 + 2 * tq;
        if ia < count {
            out[((((first + ia) * heads + h) * dv + col) / 2) as usize] =
                Vector::cast_from(acc_o[(2 * nt) as usize]);
        }
        if ib < count {
            out[((((first + ib) * heads + h) * dv + col) / 2) as usize] =
                Vector::cast_from(acc_o[(2 * nt + 1) as usize]);
        }
    }
}

/// Stages a block of `w` columns, from 16-byte chunk `c0`, of the `rows` rows of chunk `chunk` of
/// a chunked tensor of the input dtype whose rows are `d` long, into the swizzled tile `dst`
/// (`w / 8` chunks per row); `units` units take part.
#[cube]
#[allow(clippy::too_many_arguments)]
fn stage_block<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    chunk: u32,
    c0: u32,
    #[comptime] rows: u32,
    #[comptime] d: u32,
    #[comptime] w: u32,
    #[comptime] units: u32,
) {
    let chunks = comptime!(w / 8);
    let total = comptime!(rows * w / 8);
    #[unroll]
    for i in 0..comptime!(total.div_ceil(units)) {
        let idx = i * units + UNIT_POS;
        if idx < total {
            let r = idx / chunks;
            let c = idx % chunks;
            dst[swizzle(r, c, chunks) as usize] =
                src[((chunk * rows + r) * comptime!(d / 8) + c0 + c) as usize];
        }
    }
}

/// Writes accumulator pairs as an `R2` tile of the input dtype: the warp's `tiles` 8-column
/// tiles from column `n0` of rows `m0 + l / 4` (and `+ 8`), the tile `chunks` 16-byte chunks
/// wide.
#[cube]
fn store_tile<E: Float>(
    acc: &Array<Vector<f32, R2>>,
    dst: &mut [Vector<E, R2>],
    m0: u32,
    n0: u32,
    #[comptime] tiles: u32,
    #[comptime] chunks: u32,
) {
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;
    #[unroll]
    for nt in 0..tiles {
        #[unroll]
        for half in 0..2u32 {
            let i = m0 + gq + 8 * half;
            dst[(swizzle(i, n0 / 8 + nt, chunks) * 4 + tq) as usize] =
                Vector::cast_from(acc[(2 * nt + half) as usize]);
        }
    }
}

/// Writes accumulator pairs to a chunked tensor `[.., 64, d]` (of the input dtype, or f32):
/// chunk `chunk`, the warp's rows `m0 + l / 4` (and `+ 8`) and `tiles` 8-column tiles from
/// column `n0`, times `factor`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn store_chunked<F: Float>(
    acc: &Array<Vector<f32, R2>>,
    dst: &mut Tensor<Vector<F, R2>>,
    chunk: u32,
    m0: u32,
    n0: u32,
    factor: f32,
    #[comptime] tiles: u32,
    #[comptime] d: u32,
) {
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;
    #[unroll]
    for nt in 0..tiles {
        #[unroll]
        for half in 0..2u32 {
            let i = m0 + gq + 8 * half;
            let col = n0 + 8 * nt + 2 * tq;
            dst[(((chunk * 64 + i) * d + col) / 2) as usize] =
                Vector::cast_from(acc[(2 * nt + half) as usize] * Vector::new(factor));
        }
    }
}

/// Backward, per chunk, as `dvlocal_kernel`: `dv_local = P^T dO` (f32), `qg = gamma Q` and the
/// output's gradient in the chunked layout (both in the input dtype), from the raw queries and
/// keys and the output's gradient in the input dtype. Warp `w` forms rows `16 (w % 4)` and
/// columns `32 (w / 4)` of `P`, staged for the product with `dO`, then rows `16 (w % 4)` and
/// columns `Dv / 2 (w / 4)` of `P^T dO`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tc_dvlocal_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    gcum: &Tensor<f32>,
    d_out: &Tensor<Vector<E, V8>>,
    dv_local: &mut Tensor<Vector<E, R2>>,
    qg: &mut Tensor<Vector<E, V8>>,
    d_o: &mut Tensor<Vector<E, V8>>,
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
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
    let ck = comptime!(dk / 8);
    let cv = comptime!(dv / 8);
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
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;

    let mut s_q =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    let mut s_k =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    let mut s_o =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dv / 8) as usize), 16usize);
    let mut s_p = Shared::<[Vector<E, R2>]>::new_aligned_slice(2048usize, 16usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        gs[tid] = gcum[cube as usize * 64usize + tid];
    }
    stage_tokens::<E>(
        q,
        s_q.as_mut_slice(),
        gs.as_slice(),
        first,
        count,
        h,
        heads,
        scale,
        dk,
        l2norm,
        false,
    );
    stage_tokens::<E>(
        k,
        s_k.as_mut_slice(),
        gs.as_slice(),
        first,
        count,
        h,
        heads,
        1.0f32,
        dk,
        l2norm,
        false,
    );
    stage_tokens::<E>(
        d_out,
        s_o.as_mut_slice(),
        gs.as_slice(),
        first,
        count,
        h,
        heads,
        1.0f32,
        dv,
        false,
        false,
    );
    sync_cube();
    #[unroll]
    for i in 0..comptime!(64 * dk / 8 / 256) {
        let idx = i * 256 + UNIT_POS;
        let r = idx / ck;
        let cc = idx % ck;
        let x = Vector::<f32, V8>::cast_from(s_q[swizzle(r, cc, ck) as usize]);
        qg[((cube * 64 + r) * ck + cc) as usize] =
            Vector::cast_from(x * Vector::new(gs[r as usize].exp()));
    }
    #[unroll]
    for i in 0..comptime!(64 * dv / 8 / 256) {
        let idx = i * 256 + UNIT_POS;
        let r = idx / cv;
        let cc = idx % cv;
        d_o[((cube * 64 + r) * cv + cc) as usize] = s_o[swizzle(r, cc, cv) as usize];
    }

    // P = (Q K^T) exp(G_i - G_j), i >= j.
    let mrow = (warp % 4) * 16;
    let ncol = (warp / 4) * 32;
    let mut acc_p = zeros(8u32);
    gemm::<E, V8, V8>(
        &def,
        s_q.as_slice(),
        s_k.as_slice(),
        &mut acc_p,
        mrow,
        ncol,
        0u32,
        0u32,
        comptime!(dk / 16),
        2u32,
        false,
        true,
        ck,
        1u32,
        ck,
        1u32,
    );
    #[unroll]
    for nt in 0..4u32 {
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            let gi = gs[i as usize];
            let mut p = acc_p[(2 * nt + half) as usize];
            #[unroll]
            for e in 0..2u32 {
                let j = ncol + nt * 8 + 2 * tq + e;
                p.insert(
                    e as usize,
                    select(
                        i >= j,
                        p.extract(e as usize) * (gi - gs[j as usize]).exp(),
                        0.0f32,
                    ),
                );
            }
            acc_p[(2 * nt + half) as usize] = p;
        }
    }
    store_tile::<E>(&acc_p, s_p.as_mut_slice(), mrow, ncol, 4u32, 8u32);
    sync_cube();

    // dv_local = P^T dO.
    let n0 = (warp / 4) * comptime!(dv / 2);
    let mut acc = zeros(comptime!(dv / 8));
    gemm::<E, R2, V8>(
        &def,
        s_p.as_slice(),
        s_o.as_slice(),
        &mut acc,
        mrow,
        n0,
        0u32,
        0u32,
        4u32,
        comptime!(dv / 32),
        true,
        false,
        8u32,
        4u32,
        cv,
        1u32,
    );
    store_chunked::<E>(
        &acc,
        dv_local,
        cube,
        mrow,
        n0,
        1.0f32,
        comptime!(dv / 16),
        dv,
    );
}

/// The state passing, backward, as `state_backward_kernel`: for each of the sequence's chunks
/// from the last, store `dS_{c+1}` (in the input dtype), form `dV' = lam (K dS) + dv_local`, then
/// `dS = exp(G_last) dS + qg^T dO - W^T dV'`, on tensor cores, the state's gradient f32 in
/// registers. Laid out as [`tc_state_forward_kernel`]; one buffer holds `K` for the first
/// product and `qg` for the second (each read into registers during the previous chunk).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tc_state_backward_kernel<E: Float>(
    kn: &Tensor<Vector<E, V8>>,
    w: &Tensor<Vector<E, V8>>,
    qg: &Tensor<Vector<E, V8>>,
    d_o: &Tensor<Vector<E, R2>>,
    dv_local: &Tensor<Vector<E, R2>>,
    gcum: &Tensor<f32>,
    d_last: &Tensor<Vector<f32, R2>>,
    offsets: &Tensor<u32>,
    ds_next: &mut Tensor<Vector<E, R2>>,
    dvn: &mut Tensor<Vector<E, R2>>,
    d_first: &mut Tensor<Vector<f32, R2>>,
    n_chunks: u32,
    heads: u32,
    #[comptime] dk: u32,
    #[comptime] dv: u32,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
    let ck = comptime!(dk / 8);
    let y = CUBE_POS_Y;
    let v0 = CUBE_POS_X * 32;
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;
    let (base, n) = state_chunks(
        offsets,
        y as usize,
        heads as usize,
        n_chunks as usize,
        packed,
    );
    let base = base as u32;
    let n = n as u32;

    let mts = comptime!(if dk >= 64 { dk / 64 } else { 1u32 });
    let nts = comptime!(if dk >= 64 { 4u32 } else { 2u32 });
    let mut mt0 = warp % 2;
    let mut nt0 = 2 * (warp / 2);
    if comptime!(dk >= 64) {
        mt0 = warp * mts;
        nt0 = 0u32;
    }

    // K, then qg.
    let mut s_x =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    let mut s_w =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    let mut s_s =
        Shared::<[Vector<E, R2>]>::new_aligned_slice(comptime!((dk * 16) as usize), 16usize);
    // -dV' and dO, 64 rows of 32 columns each.
    let mut s_v = Shared::<[Vector<E, R2>]>::new_aligned_slice(1024usize, 16usize);
    let mut s_o = Shared::<[Vector<E, R2>]>::new_aligned_slice(1024usize, 16usize);

    let mut sreg = Array::<Vector<f32, R2>>::new(comptime!((mts * nts * 2) as usize));
    #[unroll]
    for mi in 0..mts {
        #[unroll]
        for ni in 0..nts {
            #[unroll]
            for half in 0..2u32 {
                let d = 16 * (mt0 + mi) + gq + 8 * half;
                let col = v0 + 8 * (nt0 + ni) + 2 * tq;
                sreg[((mi * nts + ni) * 2 + half) as usize] =
                    d_last[(((y * dk + d) * dv + col) / 2) as usize];
            }
        }
    }

    let per = comptime!(64 * dk / 8 / 128);
    let mut pre_k = Array::<Vector<E, V8>>::new(per as usize);
    let mut pre_w = Array::<Vector<E, V8>>::new(per as usize);
    let mut pre_q = Array::<Vector<E, V8>>::new(per as usize);
    if n > 0 {
        let last_chunk = base + n - 1;
        stage_chunk::<E>(kn, s_x.as_mut_slice(), last_chunk, 64u32, dk, 128u32);
        stage_chunk::<E>(w, s_w.as_mut_slice(), last_chunk, 64u32, dk, 128u32);
        #[unroll]
        for i in 0..per {
            let idx = i * 128 + UNIT_POS;
            pre_q[i as usize] = qg[(last_chunk * 64 * ck + idx) as usize];
        }
    }

    for cr in 0..n {
        let chunk = base + n - 1 - cr;
        // dS_{c+1}, stored and staged; this chunk's dO staged, its dv_local and decays read.
        #[unroll]
        for mi in 0..mts {
            #[unroll]
            for ni in 0..nts {
                #[unroll]
                for half in 0..2u32 {
                    let d = 16 * (mt0 + mi) + gq + 8 * half;
                    let x = Vector::<E, R2>::cast_from(sreg[((mi * nts + ni) * 2 + half) as usize]);
                    ds_next
                        [(((chunk * dk + d) * dv + v0 + 8 * (nt0 + ni) + 2 * tq) / 2) as usize] = x;
                    s_s[(swizzle(d, nt0 + ni, 4u32) * 4 + tq) as usize] = x;
                }
            }
        }
        let mut lreg = Array::<Vector<f32, R2>>::new(8usize);
        #[unroll]
        for nt in 0..4u32 {
            #[unroll]
            for half in 0..2u32 {
                let i = 16 * warp + gq + 8 * half;
                let at = (((chunk * 64 + i) * dv + v0 + 8 * nt + 2 * tq) / 2) as usize;
                lreg[(2 * nt + half) as usize] = Vector::cast_from(dv_local[at]);
                s_o[(swizzle(i, nt, 4u32) * 4 + tq) as usize] = d_o[at];
            }
        }
        let g_last = gcum[(chunk * 64 + 63) as usize];
        let lam_a = (g_last - gcum[(chunk * 64 + 16 * warp + gq) as usize]).exp();
        let lam_b = (g_last - gcum[(chunk * 64 + 16 * warp + gq + 8) as usize]).exp();
        sync_cube();

        // dV' = lam (K dS) + dv_local.
        let mut acc_v = zeros(8u32);
        gemm::<E, V8, R2>(
            &def,
            s_x.as_slice(),
            s_s.as_slice(),
            &mut acc_v,
            16 * warp,
            0u32,
            0u32,
            0u32,
            comptime!(dk / 16),
            2u32,
            false,
            false,
            ck,
            1u32,
            4u32,
            4u32,
        );
        #[unroll]
        for nt in 0..4u32 {
            #[unroll]
            for half in 0..2u32 {
                let i = 16 * warp + gq + 8 * half;
                let lam = select(half == 0, lam_a, lam_b);
                let x = acc_v[(2 * nt + half) as usize] * Vector::new(lam)
                    + lreg[(2 * nt + half) as usize];
                dvn[(((chunk * 64 + i) * dv + v0 + 8 * nt + 2 * tq) / 2) as usize] =
                    Vector::cast_from(x);
                s_v[(swizzle(i, nt, 4u32) * 4 + tq) as usize] =
                    Vector::cast_from(x * Vector::new(-1.0f32));
            }
        }
        sync_cube();
        // qg in place of K; the previous chunk's K, W and qg into registers.
        #[unroll]
        for i in 0..per {
            let idx = i * 128 + UNIT_POS;
            s_x[swizzle(idx / ck, idx % ck, ck) as usize] = pre_q[i as usize];
        }
        let more = cr + 1 < n;
        if more {
            #[unroll]
            for i in 0..per {
                let idx = i * 128 + UNIT_POS;
                let at = ((chunk - 1) * 64 * ck + idx) as usize;
                pre_k[i as usize] = kn[at];
                pre_w[i as usize] = w[at];
                pre_q[i as usize] = qg[at];
            }
        }
        sync_cube();

        // dS = exp(G_last) dS + qg^T dO - W^T dV'.
        let decay = g_last.exp();
        #[unroll]
        for x in 0..comptime!(mts * nts * 2) {
            sreg[x as usize] = sreg[x as usize] * Vector::new(decay);
        }
        #[unroll]
        for mi in 0..mts {
            gemm::<E, V8, R2>(
                &def,
                s_x.as_slice(),
                s_o.as_slice(),
                &mut sreg,
                16 * (mt0 + mi),
                8 * nt0,
                0u32,
                mi * nts,
                4u32,
                comptime!(nts / 2),
                true,
                false,
                ck,
                1u32,
                4u32,
                4u32,
            );
            gemm::<E, V8, R2>(
                &def,
                s_w.as_slice(),
                s_v.as_slice(),
                &mut sreg,
                16 * (mt0 + mi),
                8 * nt0,
                0u32,
                mi * nts,
                4u32,
                comptime!(nts / 2),
                true,
                false,
                ck,
                1u32,
                4u32,
                4u32,
            );
        }
        sync_cube();
        if more {
            #[unroll]
            for i in 0..per {
                let idx = i * 128 + UNIT_POS;
                let at = swizzle(idx / ck, idx % ck, ck) as usize;
                s_x[at] = pre_k[i as usize];
                s_w[at] = pre_w[i as usize];
            }
        }
    }

    #[unroll]
    for mi in 0..mts {
        #[unroll]
        for ni in 0..nts {
            #[unroll]
            for half in 0..2u32 {
                let d = 16 * (mt0 + mi) + gq + 8 * half;
                let col = v0 + 8 * (nt0 + ni) + 2 * tq;
                d_first[(((y * dk + d) * dv + col) / 2) as usize] =
                    sreg[((mi * nts + ni) * 2 + half) as usize];
            }
        }
    }
}

/// [`stage_block`] into a tile of pairs (`w / 8` chunks per row, four pairs each), so that the
/// lanes of an accumulator can read the tile's elements at their own rows and columns.
#[cube]
#[allow(clippy::too_many_arguments)]
fn stage_block_r2<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    dst: &mut [Vector<E, R2>],
    chunk: u32,
    c0: u32,
    #[comptime] rows: u32,
    #[comptime] d: u32,
    #[comptime] w: u32,
    #[comptime] units: u32,
) {
    let chunks = comptime!(w / 8);
    let total = comptime!(rows * w / 8);
    #[unroll]
    for i in 0..comptime!(total.div_ceil(units)) {
        let idx = i * units + UNIT_POS;
        if idx < total {
            let r = idx / chunks;
            let c = idx % chunks;
            let x = src[((chunk * rows + r) * comptime!(d / 8) + c0 + c) as usize];
            let at = (swizzle(r, c, chunks) * 4) as usize;
            #[unroll]
            for e in 0..4usize {
                let mut y = Vector::<E, R2>::new(E::from_int(0));
                y.insert(0usize, x.extract(2 * e));
                y.insert(1usize, x.extract(2 * e + 1));
                dst[at + e] = y;
            }
        }
    }
}

/// The sum over the four lanes of a quad (the lanes holding one accumulator row).
#[cube]
fn quad_sum(value: f32) -> f32 {
    let value = value + plane_shuffle_xor(value, 1u32);
    value + plane_shuffle_xor(value, 2u32)
}

/// [`row_totals`] of three sums at once (`red` holds 384), into `out_x`, `out_y`, `out_z`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn row_totals3(
    xa: f32,
    xb: f32,
    ya: f32,
    yb: f32,
    za: f32,
    zb: f32,
    red: &mut [f32],
    out_x: &mut [f32],
    out_y: &mut [f32],
    out_z: &mut [f32],
) {
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let xa = quad_sum(xa);
    let xb = quad_sum(xb);
    let ya = quad_sum(ya);
    let yb = quad_sum(yb);
    let za = quad_sum(za);
    let zb = quad_sum(zb);
    let row = (warp % 4) * 16 + lane / 4;
    if lane % 4 == 0 {
        let a = (row * 2 + warp / 4) as usize;
        let b = ((row + 8) * 2 + warp / 4) as usize;
        red[a] = xa;
        red[b] = xb;
        red[128 + a] = ya;
        red[128 + b] = yb;
        red[256 + a] = za;
        red[256 + b] = zb;
    }
    sync_cube();
    if UNIT_POS < 64 {
        let t = (UNIT_POS * 2) as usize;
        out_x[UNIT_POS as usize] = red[t] + red[t + 1];
        out_y[UNIT_POS as usize] = red[128 + t] + red[129 + t];
        out_z[UNIT_POS as usize] = red[256 + t] + red[257 + t];
    }
    sync_cube();
}

/// Each of the chunk's 64 rows' totals of per-lane partial sums, into `out[row]`, for the
/// per-chunk kernels whose warps `w` and `w + 4` share rows `16 (w % 4)`: `part_a` for the
/// lane's row `l / 4`, `part_b` for row `l / 4 + 8`. Every unit takes part; `red` holds 128.
#[cube]
fn row_totals(part_a: f32, part_b: f32, red: &mut [f32], out: &mut [f32]) {
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let a = quad_sum(part_a);
    let b = quad_sum(part_b);
    let row = (warp % 4) * 16 + lane / 4;
    if lane % 4 == 0 {
        red[(row * 2 + warp / 4) as usize] = a;
        red[((row + 8) * 2 + warp / 4) as usize] = b;
    }
    sync_cube();
    if UNIT_POS < 64 {
        out[UNIT_POS as usize] = red[(UNIT_POS * 2) as usize] + red[(UNIT_POS * 2 + 1) as usize];
    }
    sync_cube();
}

/// Row sums minus column sums of a 64 x 64 block held as accumulators, warp `w` its rows
/// `16 (w % 4)` and columns `32 (w / 4)` (`m`, four 8-column tiles), into `out[t]`: the gate's
/// part from `P` or from `A`. `red` holds 128 row partials, `red_c` 256 column partials.
#[cube]
fn row_minus_col(m: &Array<Vector<f32, R2>>, red: &mut [f32], red_c: &mut [f32], out: &mut [f32]) {
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;
    let mut ra = 0.0f32;
    let mut rb = 0.0f32;
    let mut cs = Array::<f32>::new(8usize);
    #[unroll]
    for nt in 0..4u32 {
        let x = m[(2 * nt) as usize];
        let y = m[(2 * nt + 1) as usize];
        #[unroll]
        for e in 0..2u32 {
            ra += x.extract(e as usize);
            rb += y.extract(e as usize);
            cs[(2 * nt + e) as usize] = x.extract(e as usize) + y.extract(e as usize);
        }
    }
    let ra = quad_sum(ra);
    let rb = quad_sum(rb);
    #[unroll]
    for x in 0..8usize {
        let mut v = cs[x];
        v += plane_shuffle_xor(v, 4u32);
        v += plane_shuffle_xor(v, 8u32);
        v += plane_shuffle_xor(v, 16u32);
        cs[x] = v;
    }
    let row = (warp % 4) * 16 + gq;
    if tq == 0 {
        red[(row * 2 + warp / 4) as usize] = ra;
        red[((row + 8) * 2 + warp / 4) as usize] = rb;
    }
    if gq == 0 {
        #[unroll]
        for nt in 0..4u32 {
            #[unroll]
            for e in 0..2u32 {
                let col = (warp / 4) * 32 + 8 * nt + 2 * tq + e;
                red_c[((warp % 4) * 64 + col) as usize] = cs[(2 * nt + e) as usize];
            }
        }
    }
    sync_cube();
    if UNIT_POS < 64 {
        let t = UNIT_POS as usize;
        out[t] = red[2 * t] + red[2 * t + 1]
            - red_c[t]
            - red_c[64 + t]
            - red_c[128 + t]
            - red_c[192 + t];
    }
    sync_cube();
}

/// Backward, per chunk, where the forward pass kept `T`: what [`tc_prep_kernel`] forms but `A`
/// and its inversion, from that `T`: `G`, `W = T (beta gamma K)`, `U = T (beta V)` and the keys.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tc_wu_kernel<E: Float>(
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    g: &Tensor<f32>,
    beta: &Tensor<f32>,
    tmat: &Tensor<Vector<E, V8>>,
    w: &mut Tensor<Vector<E, R2>>,
    u: &mut Tensor<Vector<E, R2>>,
    kn: &mut Tensor<Vector<E, V8>>,
    gcum: &mut Tensor<f32>,
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
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
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
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;

    let mut s_k =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    let mut s_x =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * cmax) as usize), 16usize);
    let mut s_t = Shared::<[Vector<E, V8>]>::new_aligned_slice(512usize, 16usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut bs = Shared::<[f32]>::new_slice(64usize);
    let mut fs = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        let mut gv = 0.0f32;
        let mut bv = 0.0f32;
        if (tid as u32) < count {
            gv = g[((first as usize + tid) * heads as usize) + h as usize];
            bv = beta[((first as usize + tid) * heads as usize) + h as usize];
        }
        gs[tid] = gv;
        bs[tid] = bv;
    }
    stage_tokens::<E>(
        k,
        s_k.as_mut_slice(),
        fs.as_slice(),
        first,
        count,
        h,
        heads,
        1.0f32,
        dk,
        l2norm,
        false,
    );
    stage_chunk::<E>(tmat, s_t.as_mut_slice(), cube, 64u32, 64u32, 256u32);
    // The values read into registers while W is formed.
    let pre_v = fetch_tokens::<E>(v, first, count, h, heads, dv, 256u32);
    sync_cube();
    if tid < 64usize {
        fs[tid] = plane_inclusive_sum(gs[tid]);
    }
    #[unroll]
    for i in 0..comptime!(64 * dk / 8 / 256) {
        let idx = i * 256 + UNIT_POS;
        let r = idx / ck;
        let cc = idx % ck;
        kn[((cube * 64 + r) * ck + cc) as usize] = s_k[swizzle(r, cc, ck) as usize];
    }
    sync_cube();
    if tid < 64usize {
        let mut x = fs[tid];
        if tid >= 32usize {
            x += fs[31usize];
        }
        gs[tid] = x;
    }
    sync_cube();
    if tid < 64usize {
        fs[tid] = bs[tid] * gs[tid].exp();
        gcum[cube as usize * 64usize + tid] = gs[tid];
    }
    sync_cube();
    // beta gamma K for W.
    #[unroll]
    for i in 0..comptime!(64 * dk / 8 / 256) {
        let idx = i * 256 + UNIT_POS;
        let r = idx / ck;
        let at = swizzle(r, idx % ck, ck) as usize;
        let x = Vector::<f32, V8>::cast_from(s_k[at]);
        s_x[at] = Vector::cast_from(x * Vector::new(fs[r as usize]));
    }
    sync_cube();

    let mrow = (warp % 4) * 16;
    let wk = comptime!(dk / 2);
    let mut acc_w = zeros(comptime!(dk / 8));
    gemm::<E, V8, V8>(
        &def,
        s_t.as_slice(),
        s_x.as_slice(),
        &mut acc_w,
        mrow,
        (warp / 4) * wk,
        0u32,
        0u32,
        4u32,
        comptime!(dk / 32),
        false,
        false,
        8u32,
        1u32,
        ck,
        1u32,
    );
    #[unroll]
    for nt in 0..comptime!(dk / 16) {
        let col = (warp / 4) * wk + nt * 8 + 2 * tq;
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            w[(((cube * 64 + i) * dk + col) / 2) as usize] =
                Vector::cast_from(acc_w[(2 * nt + half) as usize]);
        }
    }
    sync_cube();
    store_tokens_scaled::<E>(&pre_v, s_x.as_mut_slice(), bs.as_slice(), dv);
    sync_cube();
    let wv = comptime!(dv / 2);
    let mut acc_u = zeros(comptime!(dv / 8));
    gemm::<E, V8, V8>(
        &def,
        s_t.as_slice(),
        s_x.as_slice(),
        &mut acc_u,
        mrow,
        (warp / 4) * wv,
        0u32,
        0u32,
        4u32,
        comptime!(dv / 32),
        false,
        false,
        8u32,
        1u32,
        cv,
        1u32,
    );
    #[unroll]
    for nt in 0..comptime!(dv / 16) {
        let col = (warp / 4) * wv + nt * 8 + 2 * tq;
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            u[(((cube * 64 + i) * dv + col) / 2) as usize] =
                Vector::cast_from(acc_u[(2 * nt + half) as usize]);
        }
    }
}

/// This unit's share of [`stage_block`], read into registers only: issued while the cube
/// computes on other tiles, so that the reads are in flight meanwhile, and written with
/// [`store_block`] once the tile it goes to is free.
#[cube]
fn fetch_block<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    chunk: u32,
    c0: u32,
    #[comptime] rows: u32,
    #[comptime] d: u32,
    #[comptime] w: u32,
) -> Array<Vector<E, V8>> {
    let chunks = comptime!(w / 8);
    let total = comptime!(rows * w / 8);
    let n = comptime!(total.div_ceil(256u32));
    let mut regs = Array::<Vector<E, V8>>::new(n as usize);
    #[unroll]
    for i in 0..n {
        let idx = i * 256 + UNIT_POS;
        let mut x = Vector::<E, V8>::new(E::from_int(0));
        if idx < total {
            let r = idx / chunks;
            let c = idx % chunks;
            x = src[((chunk * rows + r) * comptime!(d / 8) + c0 + c) as usize];
        }
        regs[i as usize] = x;
    }
    regs
}

/// Writes what [`fetch_block`] read to the swizzled tile `dst` (`w / 8` chunks per row).
#[cube]
fn store_block<E: Float>(
    regs: &Array<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    #[comptime] rows: u32,
    #[comptime] w: u32,
) {
    let chunks = comptime!(w / 8);
    let total = comptime!(rows * w / 8);
    #[unroll]
    for i in 0..comptime!(total.div_ceil(256u32)) {
        let idx = i * 256 + UNIT_POS;
        if idx < total {
            dst[swizzle(idx / chunks, idx % chunks, chunks) as usize] = regs[i as usize];
        }
    }
}

/// This unit's share of a chunk's rows of head `h` of a `[B T, H, D]` tensor, as
/// [`stage_tokens`] reads them (zero past `count`), into registers.
#[cube]
fn fetch_tokens<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    first: u32,
    count: u32,
    h: u32,
    heads: u32,
    #[comptime] d: u32,
    #[comptime] units: u32,
) -> Array<Vector<E, V8>> {
    let chunks = comptime!(d / 8);
    let n = comptime!(64 * d / 8 / units);
    let mut regs = Array::<Vector<E, V8>>::new(n as usize);
    #[unroll]
    for i in 0..n {
        let idx = i * units + UNIT_POS;
        let r = idx / chunks;
        let mut x = Vector::<E, V8>::new(E::from_int(0));
        if r < count {
            x = src[(((first + r) * heads + h) * chunks + idx % chunks) as usize];
        }
        regs[i as usize] = x;
    }
    regs
}

/// Writes what [`fetch_tokens`] read to the swizzled tile `dst`, each row over its length when
/// `l2norm` (unscaled), the row's factor (`1 / sqrt(sum x^2 + 1e-6)`, else 1) to `inv[r]`.
#[cube]
fn store_tokens<E: Float>(
    regs: &Array<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    inv: &mut [f32],
    #[comptime] d: u32,
    #[comptime] l2norm: bool,
) {
    let chunks = comptime!(d / 8);
    #[unroll]
    for i in 0..comptime!(64 * d / 8 / 256) {
        let idx = i * 256 + UNIT_POS;
        let r = idx / chunks;
        let c = idx % chunks;
        let x = regs[i as usize];
        let mut factor = 1.0f32;
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
            factor = 1.0f32 / (ss + 1e-6f32).sqrt();
            dst[swizzle(r, c, chunks) as usize] = Vector::cast_from(f * Vector::new(factor));
        } else {
            dst[swizzle(r, c, chunks) as usize] = x;
        }
        if c == 0 {
            inv[r as usize] = factor;
        }
    }
}

/// Writes what [`fetch_tokens`] read to the swizzled tile `dst`, each row times `rows[r]`.
#[cube]
fn store_tokens_scaled<E: Float>(
    regs: &Array<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    rows: &[f32],
    #[comptime] d: u32,
) {
    let chunks = comptime!(d / 8);
    #[unroll]
    for i in 0..comptime!(64 * d / 8 / 256) {
        let idx = i * 256 + UNIT_POS;
        let r = idx / chunks;
        let f = Vector::<f32, V8>::cast_from(regs[i as usize]);
        dst[swizzle(r, idx % chunks, chunks) as usize] =
            Vector::cast_from(f * Vector::new(rows[r as usize]));
    }
}

/// The pair of a swizzled tile of 16-byte chunks at row `i` and columns `col`, `col + 1` (`col`
/// even), as f32: the chunk read whole (the four lanes of a quad read the same one), the pair
/// picked by the lane's place in it, which every lane keeps through a kernel.
#[cube]
fn pair_v8<E: Float>(
    tile: &[Vector<E, V8>],
    i: u32,
    col: u32,
    #[comptime] chunks: u32,
) -> Vector<f32, R2> {
    let x = Vector::<f32, V8>::cast_from(tile[swizzle(i, col / 8, chunks) as usize]);
    let k = (col % 8) / 2;
    let a = select(
        k == 0,
        x.extract(0usize),
        select(
            k == 1,
            x.extract(2usize),
            select(k == 2, x.extract(4usize), x.extract(6usize)),
        ),
    );
    let b = select(
        k == 0,
        x.extract(1usize),
        select(
            k == 1,
            x.extract(3usize),
            select(k == 2, x.extract(5usize), x.extract(7usize)),
        ),
    );
    let mut y = Vector::<f32, R2>::new(0.0f32);
    y.insert(0usize, a);
    y.insert(1usize, b);
    y
}

/// The dot product of two pairs.
#[cube]
fn dot2(a: Vector<f32, R2>, b: Vector<f32, R2>) -> f32 {
    a.extract(0usize) * b.extract(0usize) + a.extract(1usize) * b.extract(1usize)
}

/// Backward, per chunk, everything but the WY part (`dvb_kernel`, `dstate_kernel`,
/// `dattn_kernel` and the queries' side of `combine_kernel` in one): `dV_b = T^T dV'`, the
/// values' gradient `dV_b beta` and `sum_v dV_b V`; over `Dv` a block of 64 at a time,
/// `dQg = dO S^T`, `dkt = V' dS^T`, `dKbg = -dV_b S^T`, `<S, dS>` and `dP = dO V'^T`; then from
/// `P`, `dQ = dPe K` and `dK = dPe^T Q`. It writes the queries' gradient (through the l2norm when
/// asked), `dV_b` and `dKbg` (in the input dtype, the WY kernel's operands), the keys' gradient
/// without the WY part, `dK_attn + lam dkt + beta gamma dKbg` (f32), and per token the gate's and
/// `beta`'s gradients without it, and per chunk the gate's tail at the last row. Warp `w` takes
/// rows `16 (w % 4)` and half of the columns (`w / 4`) of each product. Each phase reads the next
/// one's tiles into registers while it computes (`dO`, `V'`, `S` and `dS` blocks, then the
/// queries and keys). Tiles: `T` then `dPe` (64 x 64) and `dV_b` as pairs, written from the
/// accumulators; `dV'`, a block of `S` then the queries (normalised, unscaled), a block of `dS`
/// then the keys, a block of `dO` and of `V'` as 16-byte chunks.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tc_dqk_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, R2>>,
    beta: &Tensor<f32>,
    gcum: &Tensor<f32>,
    tmat: &Tensor<Vector<E, V8>>,
    dvn: &Tensor<Vector<E, V8>>,
    d_o: &Tensor<Vector<E, V8>>,
    v_new: &Tensor<Vector<E, V8>>,
    states: &Tensor<Vector<E, V8>>,
    ds_next: &Tensor<Vector<E, V8>>,
    dq: &mut Tensor<Vector<E, R2>>,
    dv_out: &mut Tensor<Vector<E, R2>>,
    dvb: &mut Tensor<Vector<E, R2>>,
    dkbg: &mut Tensor<Vector<E, R2>>,
    dk_part: &mut Tensor<Vector<f32, R2>>,
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
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
    let ck = comptime!(dk / 8);
    let cv = comptime!(dv / 8);
    let dmax = comptime!(if dk > dv { dk } else { dv });
    let vbw = comptime!(if dv > 64 { 64u32 } else { dv });
    let cb = comptime!(vbw / 8);
    let nvb = comptime!(dv / vbw);
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
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;

    let big = comptime!((64 * dmax / 8) as usize);
    let blk = comptime!((64 * vbw / 8) as usize);
    let mut t1 = Shared::<[Vector<E, V8>]>::new_aligned_slice(big, 16usize);
    let mut t5 = Shared::<[Vector<E, V8>]>::new_aligned_slice(big, 16usize);
    let mut t3 = Shared::<[Vector<E, V8>]>::new_aligned_slice(blk, 16usize);
    let mut t4 = Shared::<[Vector<E, V8>]>::new_aligned_slice(blk, 16usize);
    let mut t2 =
        Shared::<[Vector<E, R2>]>::new_aligned_slice(comptime!((64 * dv / 2) as usize), 16usize);
    let mut tt = Shared::<[Vector<E, R2>]>::new_aligned_slice(2048usize, 16usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut bs = Shared::<[f32]>::new_slice(64usize);
    let mut inv_q = Shared::<[f32]>::new_slice(64usize);
    let mut inv_k = Shared::<[f32]>::new_slice(64usize);
    let mut s_bv = Shared::<[f32]>::new_slice(64usize);
    let mut s_bg = Shared::<[f32]>::new_slice(64usize);
    let mut s_kt = Shared::<[f32]>::new_slice(64usize);
    let mut s_qg = Shared::<[f32]>::new_slice(64usize);
    let mut s_dp = Shared::<[f32]>::new_slice(64usize);
    let mut s_pq = Shared::<[f32]>::new_slice(64usize);
    let mut red = Shared::<[f32]>::new_slice(128usize);
    let mut red_c = Shared::<[f32]>::new_slice(384usize);

    if tid < 64usize {
        gs[tid] = gcum[cube as usize * 64usize + tid];
        let mut x = 0.0f32;
        if (tid as u32) < count {
            x = beta[(first as usize + tid) * heads as usize + h as usize];
        }
        bs[tid] = x;
    }
    let g_last = gcum[(cube * 64 + 63) as usize];
    let mrow = (warp % 4) * 16;

    // T and dV' staged; the first block of dO, V', S and dS read meanwhile.
    stage_block_r2::<E>(
        tmat,
        tt.as_mut_slice(),
        cube,
        0u32,
        64u32,
        64u32,
        64u32,
        256u32,
    );
    stage_block::<E>(dvn, t1.as_mut_slice(), cube, 0u32, 64u32, dv, dv, 256u32);
    let mut pre_o = fetch_block::<E>(d_o, cube, 0u32, 64u32, dv, vbw);
    let mut pre_v = fetch_block::<E>(v_new, cube, 0u32, 64u32, dv, vbw);
    let mut pre_s = fetch_block::<E>(states, cube, 0u32, dk, dv, vbw);
    let mut pre_z = fetch_block::<E>(ds_next, cube, 0u32, dk, dv, vbw);
    sync_cube();

    // dV_b = T^T dV': rows 16 (w % 4), columns Dv / 2 (w / 4).
    let nv0 = (warp / 4) * comptime!(dv / 2);
    let mut acc_b = zeros(comptime!(dv / 8));
    gemm::<E, R2, V8>(
        &def,
        tt.as_slice(),
        t1.as_slice(),
        &mut acc_b,
        mrow,
        nv0,
        0u32,
        0u32,
        4u32,
        comptime!(dv / 32),
        true,
        false,
        8u32,
        4u32,
        cv,
        1u32,
    );
    store_tile::<E>(&acc_b, t2.as_mut_slice(), mrow, nv0, comptime!(dv / 16), cv);
    let mut part_a = 0.0f32;
    let mut part_b = 0.0f32;
    #[unroll]
    for nt in 0..comptime!(dv / 16) {
        let col = nv0 + 8 * nt + 2 * tq;
        #[unroll]
        for half in 0..2u32 {
            let j = mrow + gq + 8 * half;
            let x = acc_b[(2 * nt + half) as usize];
            dvb[(((cube * 64 + j) * dv + col) / 2) as usize] = Vector::cast_from(x);
            if j < count {
                let at = ((((first + j) * heads + h) * dv + col) / 2) as usize;
                dv_out[at] = Vector::cast_from(x * Vector::new(bs[j as usize]));
                let p = dot2(x, Vector::<f32, R2>::cast_from(v[at]));
                if half == 0 {
                    part_a += p;
                } else {
                    part_b += p;
                }
            }
        }
    }
    row_totals(part_a, part_b, red.as_mut_slice(), s_bv.as_mut_slice());

    // Over Dv a block at a time: dQg = dO S^T, dkt = V' dS^T, dV_b S^T (Dk columns: rows
    // 16 (w % 4), columns Dk / 2 (w / 4)), dP = dO V'^T (rows 16 (w % 4), columns 32 (w / 4)),
    // and <S, dS>; the next block, and after the last the queries and keys, read meanwhile.
    let nk0 = (warp / 4) * comptime!(dk / 2);
    let ncol = (warp / 4) * 32;
    let mut acc1 = zeros(comptime!(dk / 8));
    let mut acc2 = zeros(comptime!(dk / 8));
    let mut acc3 = zeros(comptime!(dk / 8));
    let mut acc_dp = zeros(8u32);
    let mut s_ds = 0.0f32;
    let mut pre_q = Array::<Vector<E, V8>>::new(comptime!((64 * dk / 8 / 256) as usize));
    let mut pre_k = Array::<Vector<E, V8>>::new(comptime!((64 * dk / 8 / 256) as usize));
    #[unroll]
    for vb in 0..nvb {
        store_block::<E>(&pre_o, t3.as_mut_slice(), 64u32, vbw);
        store_block::<E>(&pre_v, t4.as_mut_slice(), 64u32, vbw);
        store_block::<E>(&pre_s, t1.as_mut_slice(), dk, vbw);
        store_block::<E>(&pre_z, t5.as_mut_slice(), dk, vbw);
        if vb + 1 < nvb {
            let c0 = (vb + 1) * cb;
            pre_o = fetch_block::<E>(d_o, cube, c0, 64u32, dv, vbw);
            pre_v = fetch_block::<E>(v_new, cube, c0, 64u32, dv, vbw);
            pre_s = fetch_block::<E>(states, cube, c0, dk, dv, vbw);
            pre_z = fetch_block::<E>(ds_next, cube, c0, dk, dv, vbw);
        } else {
            pre_q = fetch_tokens::<E>(q, first, count, h, heads, dk, 256u32);
            pre_k = fetch_tokens::<E>(k, first, count, h, heads, dk, 256u32);
        }
        sync_cube();
        #[unroll]
        for i in 0..comptime!((dk * vbw / 8).div_ceil(256u32)) {
            let idx = i * 256 + UNIT_POS;
            if idx < comptime!(dk * vbw / 8) {
                let x = Vector::<f32, V8>::cast_from(t1[idx as usize]);
                let z = Vector::<f32, V8>::cast_from(t5[idx as usize]);
                let xz = x * z;
                #[unroll]
                for e in 0..8usize {
                    s_ds += xz.extract(e);
                }
            }
        }
        gemm::<E, V8, V8>(
            &def,
            t3.as_slice(),
            t1.as_slice(),
            &mut acc1,
            mrow,
            nk0,
            0u32,
            0u32,
            comptime!(vbw / 16),
            comptime!(dk / 32),
            false,
            true,
            cb,
            1u32,
            cb,
            1u32,
        );
        gemm::<E, V8, V8>(
            &def,
            t4.as_slice(),
            t5.as_slice(),
            &mut acc2,
            mrow,
            nk0,
            0u32,
            0u32,
            comptime!(vbw / 16),
            comptime!(dk / 32),
            false,
            true,
            cb,
            1u32,
            cb,
            1u32,
        );
        gemm_at::<E, R2, V8>(
            &def,
            t2.as_slice(),
            t1.as_slice(),
            &mut acc3,
            mrow,
            nk0,
            vb * vbw,
            0u32,
            0u32,
            comptime!(vbw / 16),
            comptime!(dk / 32),
            false,
            true,
            cv,
            4u32,
            cb,
            1u32,
        );
        gemm::<E, V8, V8>(
            &def,
            t3.as_slice(),
            t4.as_slice(),
            &mut acc_dp,
            mrow,
            ncol,
            0u32,
            0u32,
            comptime!(vbw / 16),
            2u32,
            false,
            true,
            cb,
            1u32,
            cb,
            1u32,
        );
        sync_cube();
    }

    // The queries and keys, normalised; dKbg out; the sums of the gate's and beta's gradients;
    // dq's part gamma dQg (times the scale) and dk's lam dkt + beta gamma dKbg.
    store_tokens::<E>(&pre_q, t1.as_mut_slice(), inv_q.as_mut_slice(), dk, l2norm);
    store_tokens::<E>(&pre_k, t5.as_mut_slice(), inv_k.as_mut_slice(), dk, l2norm);
    sync_cube();
    let mut bg_a = 0.0f32;
    let mut bg_b = 0.0f32;
    let mut kt_a = 0.0f32;
    let mut kt_b = 0.0f32;
    let mut qg_a = 0.0f32;
    let mut qg_b = 0.0f32;
    #[unroll]
    for nt in 0..comptime!(dk / 16) {
        let col = nk0 + 8 * nt + 2 * tq;
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            let x = 2 * nt + half;
            let kk = pair_v8::<E>(t5.as_slice(), i, col, ck);
            let qq = pair_v8::<E>(t1.as_slice(), i, col, ck);
            let kb = acc3[x as usize] * Vector::new(-1.0f32);
            dkbg[(((cube * 64 + i) * dk + col) / 2) as usize] = Vector::cast_from(kb);
            let a1 = acc1[x as usize];
            let a2 = acc2[x as usize];
            let gi = gs[i as usize];
            let gamma = gi.exp();
            let lam = (g_last - gi).exp();
            acc2[x as usize] = a2 * Vector::new(lam) + kb * Vector::new(bs[i as usize] * gamma);
            acc1[x as usize] = a1 * Vector::new(gamma * scale);
            if half == 0 {
                bg_a += dot2(kb, kk);
                kt_a += dot2(a2, kk);
                qg_a += dot2(a1, qq);
            } else {
                bg_b += dot2(kb, kk);
                kt_b += dot2(a2, kk);
                qg_b += dot2(a1, qq);
            }
        }
    }
    row_totals3(
        bg_a,
        bg_b,
        kt_a,
        kt_b,
        qg_a,
        qg_b,
        red_c.as_mut_slice(),
        s_bg.as_mut_slice(),
        s_kt.as_mut_slice(),
        s_qg.as_mut_slice(),
    );

    // P, dPe (times the scale: the queries' tile is unscaled) and the gate's part from P.
    let mut acc_s = zeros(8u32);
    gemm::<E, V8, V8>(
        &def,
        t1.as_slice(),
        t5.as_slice(),
        &mut acc_s,
        mrow,
        ncol,
        0u32,
        0u32,
        comptime!(dk / 16),
        2u32,
        false,
        true,
        ck,
        1u32,
        ck,
        1u32,
    );
    #[unroll]
    for nt in 0..4u32 {
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            let gi = gs[i as usize];
            let x = 2 * nt + half;
            let mut dpe = acc_dp[x as usize];
            let mut m = acc_s[x as usize];
            #[unroll]
            for e in 0..2u32 {
                let j = ncol + nt * 8 + 2 * tq + e;
                let mut d = 0.0f32;
                if i >= j {
                    d = dpe.extract(e as usize) * (gi - gs[j as usize]).exp() * scale;
                }
                m.insert(e as usize, d * m.extract(e as usize));
                dpe.insert(e as usize, d);
            }
            acc_dp[x as usize] = dpe;
            acc_s[x as usize] = m;
        }
    }
    store_tile::<E>(&acc_dp, tt.as_mut_slice(), mrow, ncol, 4u32, 8u32);
    row_minus_col(
        &acc_s,
        red.as_mut_slice(),
        red_c.as_mut_slice(),
        s_dp.as_mut_slice(),
    );

    // dq's part += dPe K, dk's part += dPe^T Q.
    gemm::<E, R2, V8>(
        &def,
        tt.as_slice(),
        t5.as_slice(),
        &mut acc1,
        mrow,
        nk0,
        0u32,
        0u32,
        4u32,
        comptime!(dk / 32),
        false,
        false,
        8u32,
        4u32,
        ck,
        1u32,
    );
    gemm::<E, R2, V8>(
        &def,
        tt.as_slice(),
        t1.as_slice(),
        &mut acc2,
        mrow,
        nk0,
        0u32,
        0u32,
        4u32,
        comptime!(dk / 32),
        true,
        false,
        8u32,
        4u32,
        ck,
        1u32,
    );

    // The queries' gradient, through the l2norm when asked: inv (dq - q (q . dq)).
    let mut pq_a = 0.0f32;
    let mut pq_b = 0.0f32;
    if comptime!(l2norm) {
        #[unroll]
        for nt in 0..comptime!(dk / 16) {
            let col = nk0 + 8 * nt + 2 * tq;
            #[unroll]
            for half in 0..2u32 {
                let i = mrow + gq + 8 * half;
                let p = dot2(
                    acc1[(2 * nt + half) as usize],
                    pair_v8::<E>(t1.as_slice(), i, col, ck),
                );
                if half == 0 {
                    pq_a += p;
                } else {
                    pq_b += p;
                }
            }
        }
        row_totals(pq_a, pq_b, red.as_mut_slice(), s_pq.as_mut_slice());
    }
    #[unroll]
    for nt in 0..comptime!(dk / 16) {
        let col = nk0 + 8 * nt + 2 * tq;
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            let mut x = acc1[(2 * nt + half) as usize];
            if comptime!(l2norm) {
                let qq = pair_v8::<E>(t1.as_slice(), i, col, ck);
                x = (x - qq * Vector::new(s_pq[i as usize])) * Vector::new(inv_q[i as usize]);
            }
            if i < count {
                dq[((((first + i) * heads + h) * dk + col) / 2) as usize] = Vector::cast_from(x);
            }
        }
    }
    store_chunked::<f32>(
        &acc2,
        dk_part,
        cube,
        mrow,
        nk0,
        1.0f32,
        comptime!(dk / 16),
        dk,
    );

    // Per token, the gate's and beta's parts; per chunk, the gate's tail at the last row,
    // exp(G_last) <S, dS> + sum lam s_kt, summed over each warp and then the warps.
    let mut lam_kt = 0.0f32;
    if tid < 64usize {
        let gi = gs[tid];
        let gamma = gi.exp();
        let lam = (g_last - gi).exp();
        let bg = s_bg[tid];
        lam_kt = lam * s_kt[tid];
        let d_gamma = s_qg[tid] * scale + bs[tid] * bg;
        rg_part[cube as usize * 64usize + tid] = s_dp[tid] + d_gamma * gamma - lam_kt;
        rb_part[cube as usize * 64usize + tid] = s_bv[tid] + gamma * bg;
    }
    let ds_warp = plane_sum(s_ds);
    let rl_warp = plane_sum(lam_kt);
    if lane == 0 {
        red[warp as usize] = g_last.exp() * ds_warp + rl_warp;
    }
    sync_cube();
    if tid == 0usize {
        let mut x = 0.0f32;
        #[unroll]
        for y in 0..8usize {
            x += red[y];
        }
        tail[cube as usize] = x;
    }
}

/// Backward, per chunk, the WY part and the sums (`dwy_kernel` and the rest of
/// `combine_kernel` in one): `dA = -(dV_b U^T + dKbg W^T)` (`i > j`), the gate's gradient from
/// `A = beta (K K^T) exp(G_i - G_j)` (formed again here) and `beta`'s, the keys' gradient
/// `dK + beta (dAe K) + dAe^T (beta K)` (through the l2norm when asked); then the gate's
/// gradient, summed back from the chunk's last row, and `beta`'s. The operands of `dA` are
/// staged 64 columns at a time, each block read into registers while the cube computes on the
/// one before (and the keys while it computes on the last); warp `w` forms rows `16 (w % 4)` and
/// columns `32 (w / 4)` of `dA` and `K K^T`, and rows `16 (w % 4)` and columns `Dk / 2 (w / 4)`
/// of the keys' gradient.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tc_wy_kernel<E: Float>(
    k: &Tensor<Vector<E, V8>>,
    gcum: &Tensor<f32>,
    beta: &Tensor<f32>,
    dvb: &Tensor<Vector<E, V8>>,
    u: &Tensor<Vector<E, V8>>,
    dkbg: &Tensor<Vector<E, V8>>,
    w: &Tensor<Vector<E, V8>>,
    dk_part: &Tensor<Vector<f32, R2>>,
    rg_part: &Tensor<f32>,
    rb_part: &Tensor<f32>,
    tail: &Tensor<f32>,
    dk_out: &mut Tensor<Vector<E, R2>>,
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
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
    let ck = comptime!(dk / 8);
    let vbw = comptime!(if dv > 64 { 64u32 } else { dv });
    let kbw = comptime!(if dk > 64 { 64u32 } else { dk });
    let bmax = comptime!(if vbw > kbw { vbw } else { kbw });
    let nvb = comptime!(dv / vbw);
    let nkb = comptime!(dk / kbw);
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
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let gq = lane / 4;
    let tq = lane % 4;

    let mut s_a =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * bmax / 8) as usize), 16usize);
    let mut s_b =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * bmax / 8) as usize), 16usize);
    let mut s_k =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    let mut s_kb =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * dk / 8) as usize), 16usize);
    let mut tt = Shared::<[Vector<E, R2>]>::new_aligned_slice(2048usize, 16usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut bs = Shared::<[f32]>::new_slice(64usize);
    let mut inv_k = Shared::<[f32]>::new_slice(64usize);
    let mut s_da = Shared::<[f32]>::new_slice(64usize);
    let mut s_ba = Shared::<[f32]>::new_slice(64usize);
    let mut s_pk = Shared::<[f32]>::new_slice(64usize);
    let mut rg = Shared::<[f32]>::new_slice(64usize);
    let mut red = Shared::<[f32]>::new_slice(128usize);
    let mut red_c = Shared::<[f32]>::new_slice(256usize);

    if tid < 64usize {
        gs[tid] = gcum[cube as usize * 64usize + tid];
        let mut x = 0.0f32;
        if (tid as u32) < count {
            x = beta[(first as usize + tid) * heads as usize + h as usize];
        }
        bs[tid] = x;
    }
    let mrow = (warp % 4) * 16;
    let ncol = (warp / 4) * 32;
    // dV_b U^T over Dv, then dKbg W^T over Dk, a block at a time, the next read meanwhile.
    let mut acc = zeros(8u32);
    let mut pre_a = fetch_block::<E>(dvb, cube, 0u32, 64u32, dv, vbw);
    let mut pre_b = fetch_block::<E>(u, cube, 0u32, 64u32, dv, vbw);
    #[unroll]
    for vb in 0..nvb {
        store_block::<E>(&pre_a, s_a.as_mut_slice(), 64u32, vbw);
        store_block::<E>(&pre_b, s_b.as_mut_slice(), 64u32, vbw);
        if vb + 1 < nvb {
            let c0 = (vb + 1) * comptime!(vbw / 8);
            pre_a = fetch_block::<E>(dvb, cube, c0, 64u32, dv, vbw);
            pre_b = fetch_block::<E>(u, cube, c0, 64u32, dv, vbw);
        }
        sync_cube();
        gemm::<E, V8, V8>(
            &def,
            s_a.as_slice(),
            s_b.as_slice(),
            &mut acc,
            mrow,
            ncol,
            0u32,
            0u32,
            comptime!(vbw / 16),
            2u32,
            false,
            true,
            comptime!(vbw / 8),
            1u32,
            comptime!(vbw / 8),
            1u32,
        );
        sync_cube();
    }
    let mut pre_c = fetch_block::<E>(dkbg, cube, 0u32, 64u32, dk, kbw);
    let mut pre_d = fetch_block::<E>(w, cube, 0u32, 64u32, dk, kbw);
    let mut pre_k = Array::<Vector<E, V8>>::new(comptime!((64 * dk / 8 / 256) as usize));
    #[unroll]
    for kb in 0..nkb {
        store_block::<E>(&pre_c, s_a.as_mut_slice(), 64u32, kbw);
        store_block::<E>(&pre_d, s_b.as_mut_slice(), 64u32, kbw);
        if kb + 1 < nkb {
            let c0 = (kb + 1) * comptime!(kbw / 8);
            pre_c = fetch_block::<E>(dkbg, cube, c0, 64u32, dk, kbw);
            pre_d = fetch_block::<E>(w, cube, c0, 64u32, dk, kbw);
        } else {
            pre_k = fetch_tokens::<E>(k, first, count, h, heads, dk, 256u32);
        }
        sync_cube();
        gemm::<E, V8, V8>(
            &def,
            s_a.as_slice(),
            s_b.as_slice(),
            &mut acc,
            mrow,
            ncol,
            0u32,
            0u32,
            comptime!(kbw / 16),
            2u32,
            false,
            true,
            comptime!(kbw / 8),
            1u32,
            comptime!(kbw / 8),
            1u32,
        );
        sync_cube();
    }
    store_tokens::<E>(&pre_k, s_k.as_mut_slice(), inv_k.as_mut_slice(), dk, l2norm);
    sync_cube();
    // beta K.
    #[unroll]
    for i in 0..comptime!(64 * dk / 8 / 256) {
        let idx = i * 256 + UNIT_POS;
        let r = idx / ck;
        let at = swizzle(r, idx % ck, ck) as usize;
        let x = Vector::<f32, V8>::cast_from(s_k[at]);
        s_kb[at] = Vector::cast_from(x * Vector::new(bs[r as usize]));
    }
    let mut acc_a = zeros(8u32);
    gemm::<E, V8, V8>(
        &def,
        s_k.as_slice(),
        s_k.as_slice(),
        &mut acc_a,
        mrow,
        ncol,
        0u32,
        0u32,
        comptime!(dk / 16),
        2u32,
        false,
        true,
        ck,
        1u32,
        ck,
        1u32,
    );
    // dAe = dA exp(G_i - G_j); the gate's part, sums of dAe A; beta's, sum_j dAe_ij (K K^T)_ij.
    let mut pa = 0.0f32;
    let mut pb = 0.0f32;
    #[unroll]
    for nt in 0..4u32 {
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            let gi = gs[i as usize];
            let bi = bs[i as usize];
            let x = 2 * nt + half;
            let mut dae = acc[x as usize];
            let mut kk = acc_a[x as usize];
            let mut p = 0.0f32;
            #[unroll]
            for e in 0..2u32 {
                let j = ncol + nt * 8 + 2 * tq + e;
                let mut d = 0.0f32;
                if i > j {
                    d = (0.0f32 - dae.extract(e as usize)) * (gi - gs[j as usize]).exp();
                }
                let dkk = d * kk.extract(e as usize);
                p += dkk;
                dae.insert(e as usize, d);
                kk.insert(e as usize, dkk * bi);
            }
            if half == 0 {
                pa += p;
            } else {
                pb += p;
            }
            acc[x as usize] = dae;
            acc_a[x as usize] = kk;
        }
    }
    store_tile::<E>(&acc, tt.as_mut_slice(), mrow, ncol, 4u32, 8u32);
    row_totals(pa, pb, red.as_mut_slice(), s_ba.as_mut_slice());
    row_minus_col(
        &acc_a,
        red.as_mut_slice(),
        red_c.as_mut_slice(),
        s_da.as_mut_slice(),
    );

    // dk = dK (the other kernel's part) + beta (dAe K) + dAe^T (beta K), through the l2norm.
    let n0 = (warp / 4) * comptime!(dk / 2);
    let mut acc1 = zeros(comptime!(dk / 8));
    gemm::<E, R2, V8>(
        &def,
        tt.as_slice(),
        s_k.as_slice(),
        &mut acc1,
        mrow,
        n0,
        0u32,
        0u32,
        4u32,
        comptime!(dk / 32),
        false,
        false,
        8u32,
        4u32,
        ck,
        1u32,
    );
    let mut acc2 = zeros(comptime!(dk / 8));
    gemm::<E, R2, V8>(
        &def,
        tt.as_slice(),
        s_kb.as_slice(),
        &mut acc2,
        mrow,
        n0,
        0u32,
        0u32,
        4u32,
        comptime!(dk / 32),
        true,
        false,
        8u32,
        4u32,
        ck,
        1u32,
    );
    let mut pk_a = 0.0f32;
    let mut pk_b = 0.0f32;
    #[unroll]
    for nt in 0..comptime!(dk / 16) {
        let col = n0 + 8 * nt + 2 * tq;
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            let x = 2 * nt + half;
            let y = dk_part[(((cube * 64 + i) * dk + col) / 2) as usize]
                + acc1[x as usize] * Vector::new(bs[i as usize])
                + acc2[x as usize];
            acc1[x as usize] = y;
            if comptime!(l2norm) {
                let p = dot2(y, pair_v8::<E>(s_k.as_slice(), i, col, ck));
                if half == 0 {
                    pk_a += p;
                } else {
                    pk_b += p;
                }
            }
        }
    }
    if comptime!(l2norm) {
        row_totals(pk_a, pk_b, red.as_mut_slice(), s_pk.as_mut_slice());
    }
    #[unroll]
    for nt in 0..comptime!(dk / 16) {
        let col = n0 + 8 * nt + 2 * tq;
        #[unroll]
        for half in 0..2u32 {
            let i = mrow + gq + 8 * half;
            let mut x = acc1[(2 * nt + half) as usize];
            if comptime!(l2norm) {
                let kk = pair_v8::<E>(s_k.as_slice(), i, col, ck);
                x = (x - kk * Vector::new(s_pk[i as usize])) * Vector::new(inv_k[i as usize]);
            }
            if i < count {
                dk_out[((((first + i) * heads + h) * dk + col) / 2) as usize] =
                    Vector::cast_from(x);
            }
        }
    }

    // The gate's gradient, the reverse cumulative sum within the chunk from its tail (unit `t`
    // scans row 63 - t, the first two warps each 32 rows, the second adding the first's total);
    // beta's.
    if tid < 64usize {
        let row = 63usize - tid;
        let mut x = rg_part[cube as usize * 64usize + row] + s_da[row];
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
        let row = 63usize - tid;
        let mut x = rg[tid];
        if tid >= 32usize {
            x += rg[31usize];
        }
        if (row as u32) < count {
            dg[(first as usize + row) * heads as usize + h as usize] = x;
        }
    }
}
