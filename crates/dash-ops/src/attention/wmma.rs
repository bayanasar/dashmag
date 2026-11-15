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
//!   query in registers for the whole sweep (`D / 16` vectors of eight).
//! - The softmax runs down each column: a lane's eight keys, then the other half of the wave
//!   (`lane ^ 16`) through one shuffle. Each lane's query has one maximum and one rescaling
//!   factor, so the output's rescaling is a multiplication by a lane's own scalar.
//! - `O^T += V^T P^T` (head dimension x queries): `P^T` is the probabilities' accumulator
//!   converted to the input dtype in place, the B operand as it stands. A is `V^T`: the value
//!   tile is stored transposed in shared memory, eight keys of one column of `V` per 16-byte
//!   vector, so that a lane reads its eight keys at once.
//!
//! The output's accumulator, `D / 16` tiles of eight f32 per lane (128 registers at `D` 256),
//! holds for each lane eight consecutive elements of its query's output row, written as one
//! 16-byte vector.
//!
//! Masks are those of `kernels.rs`: each query sees one interval of keys ([`row_bounds()`]),
//! padding on top, key by key. A wave masks element by element only the key tiles that cut one
//! of its queries' intervals or carry padding; a cube sweeps the key tiles some query of it sees.
//!
//! Tiles: four waves (64 queries) per cube and 32 keys per step; the key tile and the transposed
//! value tile take `2 * 32 * D * 2` bytes of shared memory (32 KiB at `D` 256), within the
//! 64 KiB a workgroup has on RDNA4.

// See `kernels.rs`: CubeCL's frontend takes the plain operators.
#![allow(
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::assign_op_pattern,
    clippy::collapsible_if
)]

use super::kernels::{V8, fetch_padding, load_rows, row_bounds, swizzle};
use cubecl::cmma::MmaDefinition;
use cubecl::prelude::*;

/// One f32 accumulator element per vector, as the layout gives them.
type R1 = Const<1>;

/// Queries per cube: four waves of 16.
pub(crate) const WMMA_BLOCK_Q: u32 = 64;
/// Keys per step.
pub(crate) const WMMA_BLOCK_KV: u32 = 32;

/// The index of the vector holding keys `[8 g, 8 g + 8)` of column `col` of `V` in the transposed
/// value tile: rows of `groups` vectors, the group XOR-ed with bits 2 and 3 of the column (for
/// `groups` 4), so that the 16 columns a wave's half reads at one group fall in distinct banks.
#[cube]
fn vt_index(col: u32, g: u32, #[comptime] groups: u32) -> u32 {
    col * groups + (g ^ ((col >> 2) & comptime!(groups - 1)))
}

/// Copies keys `[k0, k0 + keys)` of one (batch, head) slice of `src` (rows `row_stride` vectors
/// apart from vector `base`; zeros at or past `limit`) into `dst` transposed: for each column
/// `c` of `V` and group `g` of eight keys, one vector at [`vt_index()`]. Each thread reads eight
/// rows' vectors at one chunk and writes eight transposed ones.
#[cube]
fn load_vt<E: Float>(
    src: &Tensor<Vector<E, V8>>,
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
    let blocks = comptime!(groups * chunks);
    let per_thread = comptime!((blocks + threads - 1) / threads);
    #[unroll]
    for i in 0..per_thread {
        let idx = i * threads + UNIT_POS;
        if idx < blocks {
            let c = idx % chunks;
            let g = idx / chunks;
            let mut rows = Array::<Vector<E, V8>>::new(8usize);
            #[unroll]
            for j in 0..8u32 {
                let row = k0 + g * 8 + j;
                let mut value = Vector::<E, V8>::new(E::from_int(0));
                if row < limit {
                    value = src[base + row as usize * row_stride + c as usize];
                }
                rows[j as usize] = value;
            }
            #[unroll]
            for e in 0..8u32 {
                let mut column = Vector::<E, V8>::new(E::from_int(0));
                #[unroll]
                for j in 0..8u32 {
                    column.insert(j as usize, rows[j as usize].extract(e as usize));
                }
                dst[vt_index(c * 8 + e, g, groups) as usize] = column;
            }
        }
    }
}

/// The training forward on RDNA4's layout. One cube per (64-query block, query head, batch),
/// query blocks taken last first; a wave per 16 queries, a lane per query and half of the key
/// or head dimension. Writes the output, contiguous `[B, Tq, Hq, D]` in 16-byte vectors, and
/// the natural log-sum-exp, contiguous `[B, Tq, Hq]`, as `forward_kernel()` does. Input strides
/// are in 8-element vectors.
#[cube(launch_unchecked)]
pub(crate) fn forward_wmma_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
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
    v_sb: u32,
    v_sh: u32,
    v_st: u32,
    m_sb: u32,
    m_st: u32,
    seqs: u32,
    left: i32,
    right: i32,
    scale: f32,
    #[comptime] d: u32,
    #[comptime] padded: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let block_q = comptime!(WMMA_BLOCK_Q);
    let block_kv = comptime!(WMMA_BLOCK_KV);
    let waves = comptime!(block_q / 16);
    let threads = comptime!(block_q * 2);
    let chunks = comptime!(d / 8);
    let groups = comptime!(block_kv / 8);
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let wave = UNIT_POS / 32;
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
    let v_base = b as usize * v_sb as usize + hk as usize * v_sh as usize;
    let m_base = b as usize * m_sb as usize;

    // This lane's query and the keys it sees; the wave's span of keys and the keys all its
    // queries see; the cube's key tiles.
    let row = q0 + wave * 16 + col;
    let (lo, hi) = row_bounds(row, tq, tk, left, right, cu_q, cu_k, seqs, packed);
    let all_lo = plane_max(lo);
    let all_hi = plane_min(hi);
    let mut s_span = Shared::<[i32]>::new_slice(comptime!((2 * waves) as usize));
    let span_lo = plane_min(lo);
    let span_hi = plane_max(hi);
    if lane == 0 {
        s_span[(2 * wave) as usize] = span_lo;
        s_span[(2 * wave + 1) as usize] = span_hi;
    }
    sync_cube();
    let mut block_lo = tk as i32;
    let mut block_hi = 0i32;
    #[unroll]
    for w in 0..waves {
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

    // The lane's query, `D / 16` vectors of eight: B of `S^T = K Q^T`.
    let mut q_frag = Array::<Vector<E, V8>>::new(comptime!((d / 16) as usize));
    #[unroll]
    for kk in 0..comptime!(d / 16) {
        let mut value = Vector::<E, V8>::new(E::from_int(0));
        if row < tq {
            value = q[q_base + row as usize * q_st as usize + (kk * 2 + half) as usize];
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

    let mut run_max = f32::new(f32::NEG_INFINITY);
    let mut run_sum = 0.0f32;
    let mut acc_o = Array::<Vector<f32, R1>>::new(comptime!((d / 2) as usize));
    #[unroll]
    for i in 0..comptime!(d / 2) {
        acc_o[i as usize] = Vector::new(0.0f32);
    }

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
        load_vt::<E>(
            v,
            s_vt.as_mut_slice(),
            v_base,
            v_st as usize,
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
        // Mask unless every query of the wave sees the whole tile.
        let masked = padded || !(all_lo <= start && start + size <= all_hi);

        // S^T, one 16 x 16 tile per 16 keys: eight keys of the lane's query per tile.
        let mut acc_s = Array::<Vector<f32, R1>>::new(comptime!((block_kv / 2) as usize));
        #[unroll]
        for mt in 0..comptime!(block_kv / 16) {
            let mut c = Array::<Vector<f32, R1>>::new(8usize);
            #[unroll]
            for i in 0..8u32 {
                c[i as usize] = Vector::new(0.0f32);
            }
            #[unroll]
            for kk in 0..comptime!(d / 16) {
                let mut a = Array::<Vector<E, V8>>::new(1usize);
                a[0] = s_k[swizzle(mt * 16 + col, kk * 2 + half, chunks) as usize];
                let mut bq = Array::<Vector<E, V8>>::new(1usize);
                bq[0] = q_frag[kk as usize];
                c = def.execute(&a, &bq, &c);
            }
            #[unroll]
            for i in 0..8u32 {
                acc_s[(mt * 8 + i) as usize] = c[i as usize];
            }
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
        for i in 0..comptime!(d / 2) {
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
            for dt in 0..comptime!(d / 16) {
                let mut a = Array::<Vector<E, V8>>::new(1usize);
                a[0] = s_vt[vt_index(dt * 16 + col, kt * 2 + half, groups) as usize];
                let mut c = Array::<Vector<f32, R1>>::new(8usize);
                #[unroll]
                for i in 0..8u32 {
                    c[i as usize] = acc_o[(dt * 8 + i) as usize];
                }
                let out = def.execute(&a, &bp, &c);
                #[unroll]
                for i in 0..8u32 {
                    acc_o[(dt * 8 + i) as usize] = out[i as usize];
                }
            }
        }
    }

    let total = run_sum + plane_shuffle_xor(run_sum, 16u32);
    let inv = select(total > 0.0f32, 1.0f32 / total, 0.0f32);
    // Row `row` of this head in `[B, Tq, Hq]` order.
    let at = (b as usize * tq as usize + row as usize) * hq as usize + h as usize;
    if row < tq {
        if half == 0 {
            lse[at] = run_max + total.ln();
        }
        #[unroll]
        for dt in 0..comptime!(d / 16) {
            let mut out = Vector::<E, V8>::new(E::from_int(0));
            #[unroll]
            for i in 0..8u32 {
                out.insert(
                    i as usize,
                    E::cast_from(acc_o[(dt * 8 + i) as usize].extract(0usize) * inv),
                );
            }
            o[at * comptime!((d / 8) as usize) + (dt * 2 + half) as usize] = out;
        }
    }
}
