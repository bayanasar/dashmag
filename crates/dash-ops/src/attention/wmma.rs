//! The forward on a GPU whose manual MMA is 16 x 16 x 16 from bf16 or f16 into f32, in the
//! register layout of AMD's RDNA4 (`v_wmma_*_16x16x16`, waves of 32), where the kernels of
//! `kernels.rs` (`mma.sync` m16n8k16 and `ldmatrix`) do not run.
//!
//! The layout, as `crate::wmma` reads it from CubeCL's `position_of_nth` and checks before this
//! kernel is taken: lane `l` holds, of a 16 x 16 A (`m` x
//! `k`), row `l % 16` and the eight columns `8 (l / 16)..8 (l / 16) + 8`; of a B (`k` x `n`),
//! column `l % 16` and the eight rows `8 (l / 16)..`; of the f32 accumulator (`m` x `n`), column
//! `l % 16` and the eight rows `8 (l / 16)..`. The accumulator's layout is B's.
//!
//! So the kernel computes the transposed products, each wave for 16 queries, one per column:
//!
//! - `S^T = K Q^T` (keys x queries): A is 16 keys' rows of the key tile, eight consecutive
//!   elements of a row from shared memory; B is the wave's queries, which a lane holds for its
//!   query in registers for the whole sweep (`D / 16` vectors of eight, or its wave's part of
//!   them, below).
//! - The softmax runs down each column: a lane's eight keys, then the other half of the wave
//!   (`lane ^ 16`) through one shuffle. Each lane's query has one maximum and one rescaling
//!   factor, so the output's rescaling is a multiplication by a lane's own scalar.
//! - `O^T += V^T P^T` (head dimension x queries): `P^T` is the probabilities' accumulator
//!   converted to the input dtype in place, the B operand as it stands. A is `V^T`: the value
//!   tile is stored transposed in shared memory, eight keys of one column of `V` per 16-byte
//!   vector, so that a lane reads its eight keys at once.
//!
//! The output's accumulator, `D / 16` tiles of eight f32 per lane (or its wave's part of them),
//! holds for each lane eight consecutive elements of its query's output row per tile, written as
//! one 16-byte vector.
//!
//! Masks are those of `kernels.rs`: each query sees one interval of keys ([`row_bounds()`]),
//! padding on top, key by key. A wave masks element by element only the key tiles that cut one
//! of its queries' intervals or carry padding; a cube sweeps the key tiles some query of it sees.
//!
//! Tiles ([`forward_shape()`]): eight groups of 16 queries per cube, each of `ds` waves. Up to `D`
//! 128 a wave holds its queries whole and 32 keys are taken per step. At `D` 256 a lane's query
//! (64 registers) and accumulator (128) leave the rest of its 256 registers too few for the
//! products (211 of them spilled), so two waves share each 16 queries: each holds half of the
//! query and of the output accumulator, contracts the scores over its half of `D`, and the two
//! halves are added through shared memory in the order of the waves ([`sum_scores()`]), so that
//! both waves hold the same scores and each product runs once; 16 keys are taken per step. The key
//! tile and the transposed value tile take `2 * keys * D * 2` bytes of shared memory, the partial
//! scores `ds * 8 * 32 * keys / 2 * 4` (32 KiB in all at `D` 256), within the 64 KiB a workgroup
//! has on RDNA4. The key tile is read with no branch before a read ([`load_rows_flat()`]). The
//! value tile is read from a copy of `V` transposed ahead of the kernel ([`blocked_kernel()`]),
//! a 16-byte read and write for each column of eight keys ([`load_vt_blocked()`]). On a Radeon AI
//! PRO R9700 (causal, Hq 24, Hkv 4, bf16, `D` 64 to 256, T 4096 and 8192, 2026-11-19) the forward
//! took 0.84 to 0.92 times as long so between the backward's launches and 0.88 to 0.95 alone as
//! transposing the tile in each step, eight 4-byte reads and two writes for two columns
//! ([`load_vt()`]); the copies took 0.015 to 0.081 ms between the backward's launches and 0.017
//! to 0.129 ahead of the forward alone.

// See `kernels.rs`: CubeCL's frontend takes the plain operators.
#![allow(
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::assign_op_pattern,
    clippy::collapsible_if
)]

use super::kernels::{V8, fetch_padding, row_bounds, swizzle};
use cubecl::cmma::MmaDefinition;
use cubecl::prelude::*;

/// One f32 accumulator element per vector, as the layout gives them.
type R1 = Const<1>;

/// Two elements a vector: the view [`load_vt()`] reads a tile through.
pub(super) type V2 = Const<2>;

/// The forward's shape for a head dimension `d`: 16-query groups per cube, waves per group (each
/// contracting `D / ds` of the scores and accumulating `D / ds` of the output's columns), and keys
/// per step. See the module's documentation. On a Radeon AI PRO R9700 (the forward alone, causal,
/// Hq 24, Hkv 4, bf16, T 4096 and 8192, 2026-11-17): eight groups took 0.92 to 0.99 times the time
/// of four at `D` 64, 128 and 256, and sixteen 1.18 to 1.26 times eight at `D` 128 (and T 2048);
/// with four groups and the tiles read behind branches, two waves a group took 1.45 to 1.48 times
/// one at `D` 128, and 32 keys a step 1.10 to 1.13 times 16 at `D` 256.
pub(crate) fn forward_shape(d: u32) -> (u32, u32, u32) {
    if d > 128 { (8, 2, 16) } else { (8, 1, 32) }
}

/// Bytes of shared memory the partial scores take: with `ds` waves per group, each wave's
/// `keys / 2` f32 per lane.
pub(crate) fn score_bytes(groups: u32, ds: u32, keys: u32) -> usize {
    if ds > 1 {
        (ds * groups * 32 * (keys / 2) * 4) as usize
    } else {
        0
    }
}

/// Sums the `ds` waves' partial scores `acc_s` (`len` f32 a lane) of wave group `g` through
/// `s_ps`, in the order of the waves: every wave of the group then holds the whole sums. A barrier
/// between the writes and the reads; the caller's barrier before its next writes.
#[cube]
fn sum_scores(
    acc_s: &mut Array<Vector<f32, R1>>,
    s_ps: &mut [f32],
    g: u32,
    part: u32,
    #[comptime] groups: u32,
    #[comptime] ds: u32,
    #[comptime] len: u32,
) {
    let lane = UNIT_POS_PLANE;
    #[unroll]
    for i in 0..len {
        s_ps[(((part * groups + g) * len + i) * 32 + lane) as usize] =
            acc_s[i as usize].extract(0usize);
    }
    sync_cube();
    #[unroll]
    for i in 0..len {
        let mut x = 0.0f32;
        #[unroll]
        for p in 0..ds {
            x += s_ps[(((p * groups + g) * len + i) * 32 + lane) as usize];
        }
        acc_s[i as usize] = Vector::new(x);
    }
}

/// The index of the vector holding keys `[8 g, 8 g + 8)` of column `col` of `V` in the transposed
/// value tile: rows of `groups` vectors, the group XOR-ed with bits 2 and 3 of the column (for
/// `groups` 4), so that the 16 columns a wave's half reads at one group fall in distinct banks.
#[cube]
pub(super) fn vt_index(col: u32, g: u32, #[comptime] groups: u32) -> u32 {
    col * groups + (g ^ ((col >> 2) & comptime!(groups - 1)))
}

/// `kernels.rs`' `load_rows()` with every row read: rows at or past `limit` are read from row 0
/// and replaced by zeros, so that no read sits behind a branch, which on RDNA4 is waited for
/// before the next read is issued. Every thread takes every step: `rows * chunks` is a
/// multiple of `threads`.
#[cube]
pub(super) fn load_rows_flat<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    base: usize,
    row_stride: usize,
    row0: u32,
    limit: u32,
    #[comptime] rows: u32,
    #[comptime] chunks: u32,
    #[comptime] threads: u32,
) {
    comptime!(assert!(
        rows * chunks % threads == 0,
        "load_rows_flat: every thread takes every step, so `rows * chunks` is a multiple of `threads`"
    ));
    let per_thread = comptime!(rows * chunks / threads);
    #[unroll]
    for i in 0..per_thread {
        let idx = i * threads + UNIT_POS;
        let r = idx / chunks;
        let c = idx % chunks;
        let row = row0 + r;
        let ok = row < limit;
        let value = src[base + select(ok, row, 0u32) as usize * row_stride + c as usize];
        dst[swizzle(r, c, chunks) as usize] = select(ok, value, Vector::new(E::from_int(0)));
    }
}

/// Copies keys `[k0, k0 + keys)` of one (batch, head) slice of `src` (rows `row_stride` vectors
/// apart from vector `base`; zeros at or past `limit`) into `dst` transposed: for each column
/// `c` of `V` and group `g` of eight keys, one vector at [`vt_index()`]. `src` is viewed two
/// elements a vector, and `base` and `row_stride` count those. An item is one pair of columns of
/// eight rows: eight 4-byte reads, in flight together, the rows past `limit` read as
/// [`load_rows_flat()`] reads them, and two transposed vectors written. The items are spread over
/// every thread; where there are fewer items than threads, whole waves take one and the rest none.
#[cube]
pub(super) fn load_vt<E: Float>(
    src: &Tensor<Vector<E, V2>>,
    dst: &mut [Vector<E, V8>],
    base: usize,
    row_stride: usize,
    k0: u32,
    limit: u32,
    #[comptime] keys: u32,
    #[comptime] chunks: u32,
    #[comptime] threads: u32,
) {
    let groups = comptime!(keys / 8);
    let pairs = comptime!(chunks * 4);
    let items = comptime!(groups * pairs);
    comptime!(assert!(
        items % threads == 0 || items < threads && items % 32 == 0,
        "load_vt: every thread takes every step, or whole waves take one and the rest none"
    ));
    let per_thread = comptime!(items.div_ceil(threads));
    #[unroll]
    for i in 0..per_thread {
        let idx = i * threads + UNIT_POS;
        if comptime!(items >= threads) || idx < items {
            load_vt_item::<E>(src, dst, base, row_stride, k0, limit, idx, groups, pairs);
        }
    }
}

/// One item of [`load_vt()`]: the pair of columns `idx % pairs` of key group `idx / pairs`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_vt_item<E: Float>(
    src: &Tensor<Vector<E, V2>>,
    dst: &mut [Vector<E, V8>],
    base: usize,
    row_stride: usize,
    k0: u32,
    limit: u32,
    idx: u32,
    #[comptime] groups: u32,
    #[comptime] pairs: u32,
) {
    let p = idx % pairs;
    let g = idx / pairs;
    let mut even = Vector::<E, V8>::new(E::from_int(0));
    let mut odd = Vector::<E, V8>::new(E::from_int(0));
    #[unroll]
    for j in 0..8u32 {
        let row = k0 + g * 8 + j;
        let ok = row < limit;
        let at = base + select(ok, row, 0u32) as usize * row_stride + p as usize;
        let x = select(ok, src[at], Vector::new(E::from_int(0)));
        even.insert(j as usize, x.extract(0usize));
        odd.insert(j as usize, x.extract(1usize));
    }
    dst[vt_index(2 * p, g, groups) as usize] = even;
    dst[vt_index(2 * p + 1, g, groups) as usize] = odd;
}

/// Keys of a block of [`blocked_kernel()`]'s output: `D` columns of 16 keys, each column two
/// 16-byte vectors of eight.
pub(crate) const KEY_BLOCK: usize = 16;

/// Blocks of [`KEY_BLOCK`] keys [`blocked_kernel()`] writes for `t` keys: the keys rounded up to
/// 32, the most a kernel's step reads, so that a step past the last key reads zeros.
pub(crate) fn key_blocks(t: usize) -> usize {
    t.div_ceil(32) * 2
}

/// Copies `src`, `[B, T, H, D]` (strides in 8-element vectors, viewed two elements a vector),
/// into `dst`, contiguous `[B, H, blocks, D, 16]` ([`key_blocks()`]): for each block of 16 keys
/// and each column of `V`, the column's 16 keys, zeros at or past `t`. A unit per pair of columns
/// and eight keys, as [`load_vt()`] takes them: eight 4-byte reads, two 16-byte writes. One cube
/// per 256 such pairs of a (head, batch) slice.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn blocked_kernel<E: Float>(
    src: &Tensor<Vector<E, V2>>,
    dst: &mut Tensor<Vector<E, V8>>,
    t: u32,
    blocks: u32,
    s_sb: u32,
    s_sh: u32,
    s_st: u32,
    #[comptime] d: u32,
    #[define(E)] _dtype: ElemType,
) {
    let pairs = comptime!(d / 2);
    let idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS;
    let g = idx / pairs;
    let p = idx % pairs;
    if g < blocks * 2 {
        let h = CUBE_POS_Y;
        let b = CUBE_POS_Z;
        let base = (b as usize * s_sb as usize + h as usize * s_sh as usize) * 4;
        let mut even = Vector::<E, V8>::new(E::from_int(0));
        let mut odd = Vector::<E, V8>::new(E::from_int(0));
        #[unroll]
        for j in 0..8u32 {
            let row = g * 8 + j;
            let ok = row < t;
            let at = base + select(ok, row, 0u32) as usize * s_st as usize * 4 + p as usize;
            let x = select(ok, src[at], Vector::new(E::from_int(0)));
            even.insert(j as usize, x.extract(0usize));
            odd.insert(j as usize, x.extract(1usize));
        }
        let block =
            ((b * CUBE_COUNT_Y + h) * blocks + g / 2) as usize * comptime!((2 * d) as usize);
        dst[block + (4 * p + g % 2) as usize] = even;
        dst[block + (4 * p + 2 + g % 2) as usize] = odd;
    }
}

/// Copies keys `[k0, k0 + keys)` of one (batch, head) slice of a [`blocked_kernel()`] output
/// (from vector `base`) into `dst` as [`load_vt()`] leaves it: one 16-byte read and write per
/// column and eight keys, a step's reads contiguous. `k0` is a multiple of 16, as every kernel's
/// key tiles start at a multiple of their size. Every thread takes every step.
#[cube]
pub(super) fn load_vt_blocked<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    base: usize,
    k0: u32,
    #[comptime] keys: u32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
) {
    let groups = comptime!(keys / 8);
    let items = comptime!(groups * d);
    comptime!(assert!(
        items % threads == 0,
        "load_vt_blocked: every thread takes every step, so `keys / 8 * D` is a multiple of `threads`"
    ));
    #[unroll]
    for i in 0..comptime!(items / threads) {
        let idx = i * threads + UNIT_POS;
        let g = idx % groups;
        let c = idx / groups;
        let at = base + ((k0 / 16 + g / 2) * comptime!(2 * d) + c * 2 + g % 2) as usize;
        dst[vt_index(c, g, groups) as usize] = src[at];
    }
}

/// The training forward on RDNA4's layout. One cube per (`16 groups` queries, query head,
/// batch), query blocks taken last first; `ds` waves per 16 queries, a lane per query and half of
/// the key or head dimension, wave `w` accumulating columns `(w / groups) D / ds..` of the output. Writes the output, contiguous `[B, Tq, Hq, D]` in 16-byte vectors, and
/// the natural log-sum-exp, contiguous `[B, Tq, Hq]`, as `forward_kernel()` does. Input strides
/// are in 8-element vectors. `vt` is `V` as [`blocked_kernel()`] leaves it, `kv_blocks` blocks
/// a (batch, head) slice.
#[cube(launch_unchecked)]
pub(crate) fn forward_wmma_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    vt: &Tensor<Vector<E, V8>>,
    o: &mut Tensor<Vector<E, V8>>,
    lse: &mut Tensor<f32>,
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
    kv_blocks: u32,
    m_sb: u32,
    m_st: u32,
    seqs: u32,
    left: i32,
    right: i32,
    scale: f32,
    #[comptime] d: u32,
    #[comptime] groups: u32,
    #[comptime] ds: u32,
    #[comptime] block_kv: u32,
    #[comptime] padded: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let block_q = comptime!(16 * groups);
    let threads = comptime!(32 * groups * ds);
    let chunks = comptime!(d / 8);
    let kgroups = comptime!(block_kv / 8);
    let cols = comptime!(d / 16 / ds);
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let wave = UNIT_POS / 32;
    let qg = wave % groups;
    // `% ds`: with one wave per group, 0 as the compiler sees it, so that the kernel is the one
    // without the split.
    let part = (wave / groups) % ds;
    let lane = UNIT_POS_PLANE;
    let col = lane % 16;
    let half = lane / 16;

    let h = CUBE_POS_Y;
    let b = CUBE_POS_Z;
    let hq = CUBE_COUNT_Y;
    let hk = h / group;
    let q0 = (q_blocks - 1 - CUBE_POS_X) * block_q;

    let q_base = b as usize * q_sb as usize + h as usize * q_sh as usize;
    let k_base = b as usize * k_sb as usize + hk as usize * k_sh as usize;
    let vt_base = ((b * (hq / group) + hk) * kv_blocks) as usize * comptime!((2 * d) as usize);
    let m_base = b as usize * m_sb as usize;

    // This lane's query and the keys it sees; the wave's span of keys and the keys all its
    // queries see; the cube's key tiles.
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

    // The lane's part of its query, `D / 16 / ds` vectors of eight from column `part D / ds`: B
    // of `S^T = K Q^T`.
    let kk0 = part * cols;
    let mut q_frag = Array::<Vector<E, V8>>::new(cols as usize);
    #[unroll]
    for kk in 0..cols {
        let mut value = Vector::<E, V8>::new(E::from_int(0));
        if row < tq {
            value = q[q_base + row as usize * q_st as usize + ((kk0 + kk) * 2 + half) as usize];
        }
        q_frag[kk as usize] = value;
    }

    let mut s_k = Shared::<[Vector<E, V8>]>::new_aligned_slice(
        comptime!((block_kv * d / 8) as usize),
        16usize,
    );
    let mut s_vt = Shared::<[Vector<E, V8>]>::new_aligned_slice(
        comptime!((block_kv * d / 8) as usize),
        16usize,
    );
    let mut s_pad =
        Shared::<[i32]>::new_slice(comptime!(if padded { block_kv } else { 1u32 }) as usize);
    let mut s_ps = Shared::<[f32]>::new_slice(comptime!(if ds > 1 {
        ds * groups * 32 * (block_kv / 2)
    } else {
        1u32
    }) as usize);

    let mut run_max = f32::new(f32::NEG_INFINITY);
    let mut run_sum = 0.0f32;
    let mut acc_o = Array::<Vector<f32, R1>>::new(comptime!((8 * cols) as usize));
    #[unroll]
    for i in 0..comptime!(8 * cols) {
        acc_o[i as usize] = Vector::new(0.0f32);
    }

    for j in 0..n_blocks {
        let k0 = key0 + j * block_kv;
        sync_cube();
        load_rows_flat::<E>(
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
        load_vt_blocked::<E>(vt, s_vt.as_mut_slice(), vt_base, k0, block_kv, d, threads);
        if comptime!(padded) {
            let flag = fetch_padding(key_mask, m_base, m_st, k0, tk, block_kv);
            if UNIT_POS < block_kv {
                s_pad[UNIT_POS as usize] = flag;
            }
        }
        sync_cube();

        let start = k0 as i32;
        // Mask unless every query of the wave sees the whole tile.
        let masked = padded || !(all_lo <= start && start + size <= all_hi);

        // S^T, one 16 x 16 tile per 16 keys: eight keys of the lane's query per tile, over the
        // wave's part of D, then summed over the group's waves.
        let mut acc_s = Array::<Vector<f32, R1>>::new(comptime!((block_kv / 2) as usize));
        #[unroll]
        for mt in 0..comptime!(block_kv / 16) {
            let mut c = Array::<Vector<f32, R1>>::new(8usize);
            #[unroll]
            for i in 0..8u32 {
                c[i as usize] = Vector::new(0.0f32);
            }
            #[unroll]
            for kk in 0..cols {
                let mut a = Array::<Vector<E, V8>>::new(1usize);
                a[0] = s_k[swizzle(mt * 16 + col, (kk0 + kk) * 2 + half, chunks) as usize];
                let mut bq = Array::<Vector<E, V8>>::new(1usize);
                bq[0] = q_frag[kk as usize];
                c = def.execute(&a, &bq, &c);
            }
            #[unroll]
            for i in 0..8u32 {
                acc_s[(mt * 8 + i) as usize] = c[i as usize];
            }
        }
        if comptime!(ds > 1) {
            sum_scores(
                &mut acc_s,
                s_ps.as_mut_slice(),
                qg,
                part,
                groups,
                ds,
                comptime!(block_kv / 2),
            );
        }

        // Scale, mask and the tile's maximum down the lane's column, then across the halves.
        let mut tile_max = run_max;
        #[unroll]
        for mt in 0..comptime!(block_kv / 16) {
            #[unroll]
            for i in 0..8u32 {
                let at = (mt * 8 + i) as usize;
                let mut s = acc_s[at].extract(0usize) * scale;
                if masked {
                    let key = start + (mt * 16 + half * 8 + i) as i32;
                    let mut visible = key >= lo && key < hi;
                    if comptime!(padded) {
                        visible = visible && s_pad[(key - start) as usize] == 0i32;
                    }
                    s = select(visible, s, f32::NEG_INFINITY);
                }
                tile_max = max(tile_max, s);
                acc_s[at] = Vector::new(s);
            }
        }
        tile_max = max(tile_max, plane_shuffle_xor(tile_max, 16u32));
        // A query with every key hidden so far keeps -inf: exponentiate against 0 then, so that
        // its probabilities and rescaling factor are 0 rather than NaN.
        let base = select(tile_max == f32::NEG_INFINITY, 0.0f32, tile_max);
        let alpha = (run_max - base).exp();
        run_max = tile_max;
        let mut tile_sum = 0.0f32;
        #[unroll]
        for i in 0..comptime!(block_kv / 2) {
            let p = (acc_s[i as usize].extract(0usize) - base).exp();
            tile_sum += p;
            acc_s[i as usize] = Vector::new(p);
        }
        // Each lane sums its own keys; the halves are added once, after the loop.
        run_sum = run_sum * alpha + tile_sum;
        #[unroll]
        for i in 0..comptime!(8 * cols) {
            acc_o[i as usize] = acc_o[i as usize] * Vector::new(alpha);
        }

        // O^T += V^T P^T, 16 keys at a time: P^T's tile is B as the accumulator holds it.
        #[unroll]
        for kt in 0..comptime!(block_kv / 16) {
            let mut p = Vector::<E, V8>::new(E::from_int(0));
            #[unroll]
            for i in 0..8u32 {
                p.insert(
                    i as usize,
                    E::cast_from(acc_s[(kt * 8 + i) as usize].extract(0usize)),
                );
            }
            let mut bp = Array::<Vector<E, V8>>::new(1usize);
            bp[0] = p;
            #[unroll]
            for t in 0..cols {
                let dt = part * cols + t;
                let mut a = Array::<Vector<E, V8>>::new(1usize);
                a[0] = s_vt[vt_index(dt * 16 + col, kt * 2 + half, kgroups) as usize];
                let mut c = Array::<Vector<f32, R1>>::new(8usize);
                #[unroll]
                for i in 0..8u32 {
                    c[i as usize] = acc_o[(t * 8 + i) as usize];
                }
                let out = def.execute(&a, &bp, &c);
                #[unroll]
                for i in 0..8u32 {
                    acc_o[(t * 8 + i) as usize] = out[i as usize];
                }
            }
        }
    }

    let total = run_sum + plane_shuffle_xor(run_sum, 16u32);
    let inv = select(total > 0.0f32, 1.0f32 / total, 0.0f32);
    // Row `row` of this head in `[B, Tq, Hq]` order.
    let at = (b as usize * tq as usize + row as usize) * hq as usize + h as usize;
    if row < tq {
        if half == 0 && part == 0 {
            lse[at] = run_max + total.ln();
        }
        #[unroll]
        for t in 0..cols {
            let dt = part * cols + t;
            let mut out = Vector::<E, V8>::new(E::from_int(0));
            #[unroll]
            for i in 0..8u32 {
                out.insert(
                    i as usize,
                    E::cast_from(acc_o[(t * 8 + i) as usize].extract(0usize) * inv),
                );
            }
            o[at * comptime!((d / 8) as usize) + (dt * 2 + half) as usize] = out;
        }
    }
}
