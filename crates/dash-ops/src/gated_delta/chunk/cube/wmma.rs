//! The forward pass for bf16 and f16 inputs on AMD RDNA4's 16 x 16 x 16 matrix products from the
//! input dtype into f32 (`v_wmma_*_16x16x16`, waves of 32), where the kernels of `tc.rs`
//! (`mma.sync` m16n8k16 and `ldmatrix`) do not run, for `Dk` and `Dv` of 32, 64 or 128. They
//! compute what `tc.rs`' forward computes, through the same per-chunk quantities, with the same
//! intermediates in the input dtype (`W`, `U`, the normalised keys, `T = (I + A)^-1`, the state
//! entering each chunk and `V'`) and the same roundings; `A`, its inversion (`tc.rs`' code), the
//! decays and the state carried from chunk to chunk stay f32.
//!
//! The layout, as `crate::wmma` reads it and checks before these kernels are taken: lane `l` holds,
//! of a 16 x 16 A (`m` x `k`), row `l % 16` and the eight columns `8 (l / 16)..`; of a B (`k` x
//! `n`), column `l % 16` and the eight rows `8 (l / 16)..`; of the f32 accumulator, column `l % 16`
//! and the eight rows `8 (l / 16)..`. Both operands are then a row of 16 bytes: eight consecutive
//! elements along the inner dimension. So every product `Y = X Z` here runs as `Y^T = Z^T X^T`,
//! with `X` staged by rows and `Z` staged transposed, both along the inner dimension, each operand
//! one 16-byte read ([`frag`]); and the accumulator holds, in lane `l`, row `l % 16` of `Y` and
//! eight consecutive columns, which is written as one 16-byte vector, and is the operand `X` of a
//! next product over `Y`'s columns as it stands (`P` in `P V'`, as `attention/wmma.rs` does).
//!
//! The products over a chunk's tokens (`W`, `U`, the state's update) want the keys, values and
//! `V'` transposed, tokens along the rows: the prep kernel writes the keys and `U` transposed
//! (`[.., Dk, 64]`, `[.., Dv, 64]`), and the state passing keeps the state transposed (`S^T`,
//! `[.., Dv, Dk]`) and writes `V'` transposed, which are the operands the next kernel reads.
//!
//! Tiles are rows of 16-byte chunks, XOR-swizzled as `tc.rs`' ([`swizzle`]). Shared memory, at
//! `Dk = Dv = 128`: the prep kernel 57.25 KiB, the state passing 44 KiB, the output kernel
//! 48.25 KiB, within the 64 KiB a workgroup has on RDNA4.

// Kernel code is expanded by CubeCL's frontend, which takes the plain operators.
#![allow(
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::assign_op_pattern,
    clippy::collapsible_if
)]

use super::tc::{V8, stage_tokens, swizzle};
use super::{chunk_tokens, state_chunks};
use cubecl::cmma::MmaDefinition;
use cubecl::prelude::*;

/// One f32 accumulator element per vector, as the layout gives them.
type R1 = Const<1>;

/// Units of a per-chunk kernel: eight waves.
pub(super) const CHUNK_UNITS: u32 = 256;
/// Units of the state passing: four waves.
pub(super) const STATE_UNITS: u32 = 128;
/// State columns (of `Dv`) one state passing cube carries.
pub(super) const STATE_COLS: usize = 32;

/// The operand at rows `[r0, r0 + 16)` and inner columns `[16 kk, 16 kk + 16)` of a swizzled
/// tile of `chunks` chunks per row: lane `l` reads row `r0 + l % 16`, chunk `2 kk + l / 16`.
#[cube]
fn frag<E: Float>(
    tile: &[Vector<E, V8>],
    r0: u32,
    kk: u32,
    #[comptime] chunks: u32,
) -> Array<Vector<E, V8>> {
    let lane = UNIT_POS_PLANE;
    let mut x = Array::<Vector<E, V8>>::new(1usize);
    x[0] = tile[swizzle(r0 + lane % 16, 2 * kk + lane / 16, chunks) as usize];
    x
}

/// One operand from a vector a lane holds.
#[cube]
fn one<E: Float>(x: Vector<E, V8>) -> Array<Vector<E, V8>> {
    let mut a = Array::<Vector<E, V8>>::new(1usize);
    a[0] = x;
    a
}

/// `acc[tile] += Y^T` of `zt` (`Z^T`'s rows) and `x` (`X`'s rows), as the module says.
#[cube]
fn mma<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    zt: &Array<Vector<E, V8>>,
    x: &Array<Vector<E, V8>>,
    acc: &mut Array<Vector<f32, R1>>,
    tile: u32,
) {
    let mut c = Array::<Vector<f32, R1>>::new(8usize);
    #[unroll]
    for i in 0..8u32 {
        c[i as usize] = acc[(8 * tile + i) as usize];
    }
    let out = def.execute(zt, x, &c);
    #[unroll]
    for i in 0..8u32 {
        acc[(8 * tile + i) as usize] = out[i as usize];
    }
}

#[cube]
fn zeros(#[comptime] tiles: u32) -> Array<Vector<f32, R1>> {
    let mut acc = Array::<Vector<f32, R1>>::new(comptime!((8 * tiles) as usize));
    #[unroll]
    for i in 0..comptime!(8 * tiles) {
        acc[i as usize] = Vector::new(0.0f32);
    }
    acc
}

/// Accumulator tile `tile`, eight f32 of one row, in the input dtype.
#[cube]
fn tile_as<E: Float>(acc: &Array<Vector<f32, R1>>, tile: u32) -> Vector<E, V8> {
    let mut x = Vector::<E, V8>::new(E::from_int(0));
    #[unroll]
    for i in 0..8u32 {
        x.insert(
            i as usize,
            E::cast_from(acc[(8 * tile + i) as usize].extract(0usize)),
        );
    }
    x
}

/// Accumulator tile `tile` as eight f32.
#[cube]
fn tile_f32(acc: &Array<Vector<f32, R1>>, tile: u32) -> Vector<f32, V8> {
    let mut x = Vector::<f32, V8>::new(0.0f32);
    #[unroll]
    for i in 0..8u32 {
        x.insert(i as usize, acc[(8 * tile + i) as usize].extract(0usize));
    }
    x
}

/// Eight f32 from `values[at..at + 8]`.
#[cube]
fn eight(values: &[f32], at: u32) -> Vector<f32, V8> {
    let mut x = Vector::<f32, V8>::new(0.0f32);
    #[unroll]
    for i in 0..8u32 {
        x.insert(i as usize, values[(at + i) as usize]);
    }
    x
}

/// The 8 x 8 block of rows `rows[base..base + 8]` transposed: vector `e` holds element `e` of
/// each row.
#[cube]
fn transpose8<E: Float>(rows: &Array<Vector<E, V8>>, base: u32) -> Array<Vector<E, V8>> {
    let mut cols = Array::<Vector<E, V8>>::new(8usize);
    #[unroll]
    for e in 0..8u32 {
        let mut x = Vector::<E, V8>::new(E::from_int(0));
        #[unroll]
        for j in 0..8u32 {
            x.insert(j as usize, rows[(base + j) as usize].extract(e as usize));
        }
        cols[e as usize] = x;
    }
    cols
}

/// Per chunk, everything the state passing needs that does not depend on the state, as
/// `tc.rs`' prep kernel forms it: `G` (f32), `W = T (beta gamma K)` by rows `[.., 64, Dk]`, and
/// transposed, `U^T = (beta V)^T T^T` `[.., Dv, 64]` and the keys `K^T` (normalised when asked)
/// `[.., Dk, 64]`, these three in the input dtype; `A = beta (K K^T) exp(G_i - G_j)` (`i > j`) and
/// the products on the matrix units, the inversion `tc.rs`' in f32. Eight waves; the shared tile
/// holds the keys by rows for `K K^T`, then `beta gamma K` transposed, then `beta V` transposed.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn wmma_prep_kernel<E: Float>(
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    g: &Tensor<f32>,
    beta: &Tensor<f32>,
    w: &mut Tensor<Vector<E, V8>>,
    ut: &mut Tensor<Vector<E, V8>>,
    knt: &mut Tensor<Vector<E, V8>>,
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
        s_x.as_mut_slice(),
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
    // The values, eight tokens of one 16-byte chunk per block, read into registers while `A`
    // and its inverse are formed.
    let v_blocks = comptime!(8 * cv);
    let v_per = comptime!((v_blocks + CHUNK_UNITS - 1) / CHUNK_UNITS);
    let mut pre_v = Array::<Vector<E, V8>>::new(comptime!((8 * v_per) as usize));
    #[unroll]
    for i in 0..v_per {
        let idx = i * CHUNK_UNITS + UNIT_POS;
        let cc = idx % cv;
        let grp = idx / cv;
        #[unroll]
        for j in 0..8u32 {
            let r = grp * 8 + j;
            let mut x = Vector::<E, V8>::new(E::from_int(0));
            if idx < v_blocks && r < count {
                x = v[(((first + r) * heads + h) * cv + cc) as usize];
            }
            pre_v[(i * 8 + j) as usize] = x;
        }
    }
    sync_cube();
    // G = cumsum(g): each of the first two waves scans its 32 tokens, the second adding the
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
    sync_cube();
    if tid < 64usize {
        fs[tid] = bs[tid] * gs[tid].exp();
        gcum[cube as usize * 64usize + tid] = gs[tid];
    }

    // A[i, j] = beta_i (k_i . k_j) exp(G_i - G_j), i > j: wave w forms row tile `w % 4` of
    // K K^T at column tiles `w / 4` and `w / 4 + 2`, those on or below the diagonal.
    let mt = wave % 4;
    let nt0 = wave / 4;
    let mut acc_a = zeros(2u32);
    #[unroll]
    for kk in 0..comptime!(dk / 16) {
        let x = frag::<E>(s_x.as_slice(), 16 * mt, kk, ck);
        #[unroll]
        for s in 0..2u32 {
            let nt = nt0 + 2 * s;
            if nt <= mt {
                let zt = frag::<E>(s_x.as_slice(), 16 * nt, kk, ck);
                mma::<E>(&def, &zt, &x, &mut acc_a, s);
            }
        }
    }
    let i = 16 * mt + row;
    #[unroll]
    for s in 0..2u32 {
        let nt = nt0 + 2 * s;
        if nt <= mt {
            #[unroll]
            for e in 0..8u32 {
                let j = 16 * nt + 8 * half + e;
                let mut x = 0.0f32;
                if i > j {
                    x = bs[i as usize]
                        * acc_a[(8 * s + e) as usize].extract(0usize)
                        * (gs[i as usize] - gs[j as usize]).exp();
                }
                am[(i * 65 + j) as usize] = x;
            }
        }
    }
    // The keys in blocks of eight tokens by one chunk, into registers, before the tile turns
    // into `beta gamma K` transposed.
    let k_blocks = comptime!(8 * ck);
    let k_per = comptime!((k_blocks + CHUNK_UNITS - 1) / CHUNK_UNITS);
    let mut kblk = Array::<Vector<E, V8>>::new(comptime!((8 * k_per) as usize));
    #[unroll]
    for i in 0..k_per {
        let idx = i * CHUNK_UNITS + UNIT_POS;
        let cc = idx % ck;
        let grp = idx / ck;
        #[unroll]
        for j in 0..8u32 {
            let mut x = Vector::<E, V8>::new(E::from_int(0));
            if idx < k_blocks {
                x = s_x[swizzle(grp * 8 + j, cc, ck) as usize];
            }
            kblk[(i * 8 + j) as usize] = x;
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
    // K^T for the state passing, and beta gamma K transposed for W: row `d` of either holds the
    // chunk's 64 tokens, eight chunks.
    #[unroll]
    for i in 0..k_per {
        let idx = i * CHUNK_UNITS + UNIT_POS;
        let cc = idx % ck;
        let grp = idx / ck;
        if idx < k_blocks {
            let f = eight(fs.as_slice(), grp * 8);
            let cols = transpose8::<E>(&kblk, i * 8);
            #[unroll]
            for e in 0..8u32 {
                let d = cc * 8 + e;
                let col = cols[e as usize];
                knt[((cube * dk + d) * 8 + grp) as usize] = col;
                s_x[swizzle(d, grp, 8u32) as usize] =
                    Vector::cast_from(Vector::<f32, V8>::cast_from(col) * f);
            }
        }
    }

    // T = (I + A)^-1, as tc.rs' prep kernel forms it: the four diagonal blocks of 16 by forward
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

    // T in the input dtype.
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
        s_t[swizzle(r, cc, 8u32) as usize] = Vector::<E, V8>::cast_from(x);
    }
    sync_cube();

    // W = T (beta gamma K): wave w forms row tile `rt` and `Dk / 32` column tiles from `dt0`;
    // T's tiles above the diagonal are zero.
    let wt = comptime!(dk / 32);
    let w_first = wave * wt;
    let rt = w_first / comptime!(dk / 16);
    let dt0 = w_first % comptime!(dk / 16);
    let mut acc_w = zeros(wt);
    #[unroll]
    for kt in 0..4u32 {
        if kt <= rt {
            let x = frag::<E>(s_t.as_slice(), 16 * rt, kt, 8u32);
            #[unroll]
            for s in 0..wt {
                let zt = frag::<E>(s_x.as_slice(), 16 * (dt0 + s), kt, 8u32);
                mma::<E>(&def, &zt, &x, &mut acc_w, s);
            }
        }
    }
    let i = 16 * rt + row;
    #[unroll]
    for s in 0..wt {
        w[((cube * 64 + i) * ck + 2 * (dt0 + s) + half) as usize] = tile_as::<E>(&acc_w, s);
    }
    sync_cube();

    // beta V transposed.
    #[unroll]
    for i in 0..v_per {
        let idx = i * CHUNK_UNITS + UNIT_POS;
        let cc = idx % cv;
        let grp = idx / cv;
        if idx < v_blocks {
            let f = eight(bs.as_slice(), grp * 8);
            let cols = transpose8::<E>(&pre_v, i * 8);
            #[unroll]
            for e in 0..8u32 {
                let col = cols[e as usize];
                s_x[swizzle(cc * 8 + e, grp, 8u32) as usize] =
                    Vector::cast_from(Vector::<f32, V8>::cast_from(col) * f);
            }
        }
    }
    sync_cube();

    // U^T = (beta V)^T T^T: wave w forms row tile `vt` (of Dv) and `Dv / 32` token tiles from
    // `tt0`; T^T's tiles below the diagonal are zero.
    let ut_tiles = comptime!(dv / 32);
    let u_first = wave * ut_tiles;
    let vt = u_first / 4;
    let tt0 = u_first % 4;
    let mut acc_u = zeros(ut_tiles);
    #[unroll]
    for kt in 0..4u32 {
        let x = frag::<E>(s_x.as_slice(), 16 * vt, kt, 8u32);
        #[unroll]
        for s in 0..ut_tiles {
            if kt <= tt0 + s {
                let zt = frag::<E>(s_t.as_slice(), 16 * (tt0 + s), kt, 8u32);
                mma::<E>(&def, &zt, &x, &mut acc_u, s);
            }
        }
    }
    let vr = 16 * vt + row;
    #[unroll]
    for s in 0..ut_tiles {
        ut[((cube * dv + vr) * 8 + 2 * (tt0 + s) + half) as usize] = tile_as::<E>(&acc_u, s);
    }
}

/// The state passing, forward: for each of the sequence's chunks in turn, store the entering
/// state (transposed, in the input dtype), form `V'^T = U^T - S^T W^T`, then
/// `S^T = exp(G_last) S^T + (lam V')^T K`, on the matrix units, the state f32 in registers. One
/// cube of four waves per (sequence and head, block of 32 state columns); wave `w` forms `V'^T`'s
/// row tile `w / 2` at token tiles `2 (w % 2)` and `2 (w % 2) + 1`, and a quarter of `S^T`'s
/// tiles. The next chunk's `W` and `K^T` are read into registers during the current one.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn wmma_state_forward_kernel<E: Float>(
    knt: &Tensor<Vector<E, V8>>,
    w: &Tensor<Vector<E, V8>>,
    ut: &Tensor<Vector<E, V8>>,
    gcum: &Tensor<f32>,
    h0: &Tensor<f32>,
    offsets: &Tensor<u32>,
    states: &mut Tensor<Vector<E, V8>>,
    vnt: &mut Tensor<Vector<E, V8>>,
    last: &mut Tensor<f32>,
    n_chunks: u32,
    heads: u32,
    #[comptime] dk: u32,
    #[comptime] dv: u32,
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

    // The state's tiles this wave holds: row tile `sv` of S^T (of the block's 32 columns of S),
    // `Dk / 32` column tiles from `sd0`.
    let st = comptime!(dk / 32);
    let s_first = wave * st;
    let sv = s_first / comptime!(dk / 16);
    let sd0 = s_first % comptime!(dk / 16);
    let vr = 16 * sv + row;
    // V'^T's tiles: row tile `pv`, token tiles `pt0`, `pt0 + 1`.
    let pv = wave / 2;
    let pt0 = 2 * (wave % 2);
    let pr = 16 * pv + row;

    let mut s_w =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * ck) as usize), 16usize);
    let mut s_kt =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((dk * 8) as usize), 16usize);
    // S^T in the input dtype, 32 rows of Dk, and (lam V')^T, 32 rows of 64 tokens.
    let mut s_s =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((32 * ck) as usize), 16usize);
    let mut s_v = Shared::<[Vector<E, V8>]>::new_aligned_slice(256usize, 16usize);

    let mut sreg = zeros(st);
    #[unroll]
    for s in 0..st {
        #[unroll]
        for e in 0..8u32 {
            let d = 16 * (sd0 + s) + 8 * half + e;
            sreg[(8 * s + e) as usize] = Vector::new(h0[((y * dk + d) * dv + v0 + vr) as usize]);
        }
    }

    let per = comptime!(64 * dk / 8 / STATE_UNITS);
    let mut pre_w = Array::<Vector<E, V8>>::new(per as usize);
    let mut pre_k = Array::<Vector<E, V8>>::new(per as usize);
    if n > 0 {
        #[unroll]
        for i in 0..per {
            let idx = i * STATE_UNITS + UNIT_POS;
            let at = (base * 64 * ck + idx) as usize;
            s_w[swizzle(idx / ck, idx % ck, ck) as usize] = w[at];
            s_kt[swizzle(idx / 8, idx % 8, 8u32) as usize] = knt[at];
        }
    }

    for c in 0..n {
        let chunk = base + c;
        // The entering state, stored and staged.
        #[unroll]
        for s in 0..st {
            let x = tile_as::<E>(&sreg, s);
            let cc = 2 * (sd0 + s) + half;
            states[((chunk * dv + v0 + vr) * ck + cc) as usize] = x;
            s_s[swizzle(vr, cc, ck) as usize] = x;
        }
        // This chunk's U^T, and the next chunk's W and K^T, into registers.
        let mut ureg = Array::<Vector<E, V8>>::new(2usize);
        #[unroll]
        for s in 0..2u32 {
            ureg[s as usize] = ut[((chunk * dv + v0 + pr) * 8 + 2 * (pt0 + s) + half) as usize];
        }
        let g_last = gcum[(chunk * 64 + 63) as usize];
        let more = c + 1 < n;
        if more {
            #[unroll]
            for i in 0..per {
                let at = ((chunk + 1) * 64 * ck + i * STATE_UNITS + UNIT_POS) as usize;
                pre_w[i as usize] = w[at];
                pre_k[i as usize] = knt[at];
            }
        }
        sync_cube();

        // V'^T = U^T - S^T W^T.
        let mut acc_v = zeros(2u32);
        #[unroll]
        for kk in 0..comptime!(dk / 16) {
            let x = frag::<E>(s_s.as_slice(), 16 * pv, kk, ck);
            #[unroll]
            for s in 0..2u32 {
                let zt = frag::<E>(s_w.as_slice(), 16 * (pt0 + s), kk, ck);
                mma::<E>(&def, &zt, &x, &mut acc_v, s);
            }
        }
        #[unroll]
        for s in 0..2u32 {
            let t0 = 16 * (pt0 + s) + 8 * half;
            let x = Vector::<f32, V8>::cast_from(ureg[s as usize]) - tile_f32(&acc_v, s);
            let cc = 2 * (pt0 + s) + half;
            vnt[((chunk * dv + v0 + pr) * 8 + cc) as usize] = Vector::cast_from(x);
            let mut lam = Vector::<f32, V8>::new(0.0f32);
            #[unroll]
            for e in 0..8u32 {
                lam.insert(
                    e as usize,
                    (g_last - gcum[(chunk * 64 + t0 + e) as usize]).exp(),
                );
            }
            s_v[swizzle(pr, cc, 8u32) as usize] = Vector::cast_from(x * lam);
        }
        sync_cube();

        // S^T = exp(G_last) S^T + (lam V')^T K.
        let decay = g_last.exp();
        #[unroll]
        for x in 0..comptime!(8 * st) {
            sreg[x as usize] = sreg[x as usize] * Vector::new(decay);
        }
        #[unroll]
        for kt in 0..4u32 {
            let x = frag::<E>(s_v.as_slice(), 16 * sv, kt, 8u32);
            #[unroll]
            for s in 0..st {
                let zt = frag::<E>(s_kt.as_slice(), 16 * (sd0 + s), kt, 8u32);
                mma::<E>(&def, &zt, &x, &mut sreg, s);
            }
        }
        sync_cube();
        if more {
            #[unroll]
            for i in 0..per {
                let idx = i * STATE_UNITS + UNIT_POS;
                s_w[swizzle(idx / ck, idx % ck, ck) as usize] = pre_w[i as usize];
                s_kt[swizzle(idx / 8, idx % 8, 8u32) as usize] = pre_k[i as usize];
            }
        }
    }

    #[unroll]
    for s in 0..st {
        #[unroll]
        for e in 0..8u32 {
            let d = 16 * (sd0 + s) + 8 * half + e;
            last[((y * dk + d) * dv + v0 + vr) as usize] =
                sreg[(8 * s + e) as usize].extract(0usize);
        }
    }
}

/// Per chunk, the output `O = gamma (Q S) + P V'` with `P = (Q K^T) exp(G_i - G_j)` (`i >= j`),
/// from the raw queries and keys (normalised and scaled here), the entering state `S^T` and
/// `V'^T` in the input dtype; written in the queries' dtype in the `[B, T, H, Dv]` layout. Wave
/// `w` holds the queries of row tile `w % 4` in registers, forms `P` there (the column tiles on
/// or below the diagonal) and `O`'s columns `Dv / 2 (w / 4)`. The shared tile holds the keys for
/// `P`, then `V'^T`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn wmma_output_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    gcum: &Tensor<f32>,
    states: &Tensor<Vector<E, V8>>,
    vnt: &Tensor<Vector<E, V8>>,
    out: &mut Tensor<Vector<E, V8>>,
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
    let rt = wave % 4;
    let i = 16 * rt + row;

    let mut s_kv =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((64 * cmax) as usize), 16usize);
    let mut s_s =
        Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!((dv * ck) as usize), 16usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        gs[tid] = gcum[cube as usize * 64usize + tid];
    }
    stage_tokens::<E>(
        k,
        s_kv.as_mut_slice(),
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
    // The lane's query, `Dk / 16` vectors of eight, scaled (and normalised) as tc.rs stages it.
    let mut q_frag = Array::<Vector<E, V8>>::new(comptime!((dk / 16) as usize));
    let mut ss = 0.0f32;
    #[unroll]
    for kk in 0..comptime!(dk / 16) {
        let mut x = Vector::<E, V8>::new(E::from_int(0));
        if i < count {
            x = q[(((first + i) * heads + h) * ck + 2 * kk + half) as usize];
        }
        q_frag[kk as usize] = x;
        if comptime!(l2norm) {
            let f = Vector::<f32, V8>::cast_from(x);
            let sq = f * f;
            #[unroll]
            for e in 0..8usize {
                ss += sq.extract(e);
            }
        }
    }
    let mut factor = scale;
    if comptime!(l2norm) {
        ss += plane_shuffle_xor(ss, 16u32);
        factor = scale / (ss + 1e-6f32).sqrt();
    }
    #[unroll]
    for kk in 0..comptime!(dk / 16) {
        let f = Vector::<f32, V8>::cast_from(q_frag[kk as usize]);
        q_frag[kk as usize] = Vector::cast_from(f * Vector::new(factor));
    }
    // The state and V'^T read into registers while P is formed.
    let s_total = comptime!(dv * ck);
    let s_per = comptime!((s_total + CHUNK_UNITS - 1) / CHUNK_UNITS);
    let mut pre_s = Array::<Vector<E, V8>>::new(s_per as usize);
    #[unroll]
    for x in 0..s_per {
        let idx = x * CHUNK_UNITS + UNIT_POS;
        if idx < s_total {
            pre_s[x as usize] = states[(cube * s_total + idx) as usize];
        }
    }
    let v_total = comptime!(dv * 8);
    let v_per = comptime!((v_total + CHUNK_UNITS - 1) / CHUNK_UNITS);
    let mut pre_v = Array::<Vector<E, V8>>::new(v_per as usize);
    #[unroll]
    for x in 0..v_per {
        let idx = x * CHUNK_UNITS + UNIT_POS;
        if idx < v_total {
            pre_v[x as usize] = vnt[(cube * v_total + idx) as usize];
        }
    }
    sync_cube();

    // P = (Q K^T) exp(G_i - G_j), i >= j: the wave's 16 rows, column tiles up to its own.
    let mut acc_p = zeros(4u32);
    #[unroll]
    for kk in 0..comptime!(dk / 16) {
        let x = one::<E>(q_frag[kk as usize]);
        #[unroll]
        for nt in 0..4u32 {
            if nt <= rt {
                let zt = frag::<E>(s_kv.as_slice(), 16 * nt, kk, ck);
                mma::<E>(&def, &zt, &x, &mut acc_p, nt);
            }
        }
    }
    let gi = gs[i as usize];
    #[unroll]
    for nt in 0..4u32 {
        #[unroll]
        for e in 0..8u32 {
            let j = 16 * nt + 8 * half + e;
            let at = (8 * nt + e) as usize;
            let p = acc_p[at].extract(0usize);
            acc_p[at] = Vector::new(select(i >= j, p * (gi - gs[j as usize]).exp(), 0.0f32));
        }
    }
    sync_cube();
    #[unroll]
    for x in 0..s_per {
        let idx = x * CHUNK_UNITS + UNIT_POS;
        if idx < s_total {
            s_s[swizzle(idx / ck, idx % ck, ck) as usize] = pre_s[x as usize];
        }
    }
    #[unroll]
    for x in 0..v_per {
        let idx = x * CHUNK_UNITS + UNIT_POS;
        if idx < v_total {
            s_kv[swizzle(idx / 8, idx % 8, 8u32) as usize] = pre_v[x as usize];
        }
    }
    sync_cube();

    // O = gamma (Q S) + P V', the wave's rows and `Dv / 32` column tiles from `ot0`.
    let ot = comptime!(dv / 32);
    let ot0 = (wave / 4) * ot;
    let mut acc_o = zeros(ot);
    #[unroll]
    for kk in 0..comptime!(dk / 16) {
        let x = one::<E>(q_frag[kk as usize]);
        #[unroll]
        for s in 0..ot {
            let zt = frag::<E>(s_s.as_slice(), 16 * (ot0 + s), kk, ck);
            mma::<E>(&def, &zt, &x, &mut acc_o, s);
        }
    }
    let gamma = gi.exp();
    #[unroll]
    for x in 0..comptime!(8 * ot) {
        acc_o[x as usize] = acc_o[x as usize] * Vector::new(gamma);
    }
    #[unroll]
    for kt in 0..4u32 {
        if kt <= rt {
            let x = one::<E>(tile_as::<E>(&acc_p, kt));
            #[unroll]
            for s in 0..ot {
                let zt = frag::<E>(s_kv.as_slice(), 16 * (ot0 + s), kt, 8u32);
                mma::<E>(&def, &zt, &x, &mut acc_o, s);
            }
        }
    }
    if i < count {
        #[unroll]
        for s in 0..ot {
            out[(((first + i) * heads + h) * cv + 2 * (ot0 + s) + half) as usize] =
                tile_as::<E>(&acc_o, s);
        }
    }
}
