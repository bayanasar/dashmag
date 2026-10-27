//! The CubeCL kernels: the training forward, the three backward kernels, and decoding.
//!
//! Tiles of queries, keys, values and output gradients are staged in shared memory as rows of
//! 16-byte chunks (eight bf16 or f16), XOR-swizzled so that the eight rows an `ldmatrix` reads at
//! one chunk fall in eight different bank groups. Products run on `mma.sync.m16n8k16` with f32
//! accumulators held in registers, in the register layout PTX gives that instruction: lane `l`
//! holds rows `l / 4` and `l / 4 + 8` and columns `2 (l % 4)` and `2 (l % 4) + 1` of each 16 x 8
//! accumulator tile, which is also the layout of an A operand, so a tile of probabilities becomes
//! the left operand of the next product without leaving registers.
//!
//! Tile shapes, for a head dimension D of 64, 128 or 256 (sm_80 and later; consumer parts give a
//! block 99 KB of shared memory). The constraint at D = 256 is that a warp's f32 accumulator
//! for 16 rows of the output (or of `dQ`) is 128 registers per lane, so a warp owns 16 query
//! rows, and that a 16-row tile of queries, keys, values or output gradients is 8 KB:
//!
//! - forward: 128 queries per block, eight warps of 16 rows (two per scheduler, so that one
//!   warp's `ldmatrix` and softmax overlap the other's products), and key blocks of 32 at
//!   D = 256 (64 below): queries, keys and values take 64 + 16 + 16 KB.
//! - `dQ`: 64 queries per block (the output gradients take the other 32 KB), eight warps: two
//!   per 16 query rows, each taking 16 of every 32 keys, their partial `dQ` added through
//!   shared memory at the end.
//! - `dK`, `dV`: 32 keys per block and 32 queries per step, eight warps. The two D-long products
//!   (`S = Q K^T`, `dP = dO V^T`) are split 2 x 4 over the warps (16 queries x 8 keys each), the
//!   two 32-long ones (`dV += P^T dO`, `dK += dS^T Q`) 2 x 4 the other way (16 keys x D/4
//!   columns each), so each warp's `dK` and `dV` accumulators are 32 registers apiece at
//!   D = 256; `P` and `dS` cross between the two splits through 2 KB of shared memory each.
//!
//! Every loop reads its next tile from global memory into registers while it computes on the
//! current one in shared memory, and writes it there after a barrier.
//!
//! Masks. Whatever the call's mask and options, each query sees one interval of keys,
//! `[lo, hi)` ([`row_bounds()`]): its bottom-right aligned position widened by the window
//! (causality is a window reaching nothing after the position), within its own sequence when
//! packed; padding comes on top, key by key. The intervals of successive queries never move
//! back (`lo` and `hi` do not decrease from one query to the next), so a block of queries sees
//! one span of keys, from the first its first query sees to the last its last one sees, and a
//! block of keys is seen by one span of queries ([`query_range()`]). The forward and `dQ`
//! sweep only the key blocks of their span and `dK`, `dV` only the query blocks of theirs, and
//! each warp masks element by element only the tiles that cut one of its queries' intervals or
//! carry padding (the decision is the same for the warp's lanes: a branch only some lanes of a
//! warp take, or one around the products, costs these kernels a tenth of their time in how
//! `ptxas` schedules their loops). A query that sees no key gets the empty interval `(Tk, 0)`,
//! as do the rows past the queries.
//!
//! Inputs are read through their batch, head and time strides, counted in 16-byte vectors; a
//! row of `D` elements is contiguous. Outputs are written contiguous in the module's layout:
//! row `i` of head `h` of batch element `b` of a `[B, T, H, D]` tensor is row `(b T + i) H + h`
//! of it seen as `[B T H, D]`, and that is also the element of the `[B, T, H]` statistics (the
//! log-sum-exp, `delta`) that belongs to it. The padding mask (`[B, Tk]`, nonzero where a key is
//! left out) is read through its strides in elements, the offsets of packed sequences are
//! contiguous; both are `i32`.

// Kernel code is expanded by CubeCL's frontend, which takes the plain operators: `div_ceil`,
// `is_multiple_of` and the compound assignments on vector elements are written out, and a
// condition known when the kernel is compiled (`comptime!`) stays apart from a runtime one.
#![allow(
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::assign_op_pattern,
    clippy::collapsible_if
)]

use cubecl::cmma::{MatrixIdent, MmaDefinition};
use cubecl::prelude::*;

/// Eight elements of 16 bits: one 16-byte chunk, one row of an `ldmatrix` 8 x 8 matrix.
pub(crate) type V8 = Const<8>;
/// Two elements: one 32-bit MMA register of bf16 or f16, or two f32 accumulator columns.
pub(crate) type R2 = Const<2>;

/// Rows of queries per block in the `dQ` kernel (eight warps, two per 16 rows).
pub(crate) const BLOCK_Q: u32 = 64;
/// Rows of queries per block in the forward (eight warps of 16): with one block per
/// multiprocessor, two warps per scheduler hide each other's latency.
pub(crate) const FORWARD_BLOCK_Q: u32 = 128;
/// Keys per block and queries per step in the `dK`/`dV` kernel.
pub(crate) const BLOCK_KV: u32 = 32;

/// The index of 16-byte chunk `c` of row `r` in a swizzled tile of `chunks` chunks per row.
/// Rows of 8 or more chunks XOR the chunk with the row's low three bits; rows of 4 chunks (the
/// 32 x 32 tiles of `P` and `dS`) with bits 1 and 2, which spreads eight consecutive 64-byte
/// rows over eight bank groups as well.
#[cube]
fn swizzle(r: u32, c: u32, #[comptime] chunks: u32) -> u32 {
    if comptime!(chunks >= 8) {
        r * chunks + (c ^ (r & 7))
    } else {
        r * chunks + (c ^ ((r >> 1) & 3))
    }
}

/// Copies rows `[row0, row0 + rows)` of one (batch, head) slice of `src`, which starts at vector
/// `base` and has rows `row_stride` vectors apart, into the swizzled tile `dst`, all threads of
/// the cube taking part. Rows at or past `limit` are written as zeros.
#[cube]
fn load_rows<E: Float>(
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
    let per_thread = comptime!(rows * chunks / threads);
    #[unroll]
    for i in 0..per_thread {
        let idx = i * threads + UNIT_POS;
        let r = idx / chunks;
        let c = idx % chunks;
        let row = row0 + r;
        let mut value = Vector::<E, V8>::new(E::from_int(0));
        if row < limit {
            value = src[base + row as usize * row_stride + c as usize];
        }
        dst[swizzle(r, c, chunks) as usize] = value;
    }
}

/// This thread's share of [`load_rows()`], read into registers only: issued before the cube
/// computes on the tile in shared memory, so that the load is in flight meanwhile, and written
/// with [`store_rows()`] once every warp is done with that tile.
#[cube]
fn fetch_rows<E: Float>(
    src: &Tensor<Vector<E, V8>>,
    base: usize,
    row_stride: usize,
    row0: u32,
    limit: u32,
    #[comptime] rows: u32,
    #[comptime] chunks: u32,
    #[comptime] threads: u32,
) -> Array<Vector<E, V8>> {
    let per_thread = comptime!(rows * chunks / threads);
    let mut regs = Array::<Vector<E, V8>>::new(per_thread as usize);
    #[unroll]
    for i in 0..per_thread {
        let idx = i * threads + UNIT_POS;
        let row = row0 + idx / chunks;
        let mut value = Vector::<E, V8>::new(E::from_int(0));
        if row < limit {
            value = src[base + row as usize * row_stride + (idx % chunks) as usize];
        }
        regs[i as usize] = value;
    }
    regs
}

/// Writes what [`fetch_rows()`] read to the swizzled tile `dst`.
#[cube]
fn store_rows<E: Float>(
    regs: &Array<Vector<E, V8>>,
    dst: &mut [Vector<E, V8>],
    #[comptime] rows: u32,
    #[comptime] chunks: u32,
    #[comptime] threads: u32,
) {
    let per_thread = comptime!(rows * chunks / threads);
    #[unroll]
    for i in 0..per_thread {
        let idx = i * threads + UNIT_POS;
        dst[swizzle(idx / chunks, idx % chunks, chunks) as usize] = regs[i as usize];
    }
}

/// The A operand (16 x 16) at rows `[row0, row0 + 16)` and columns `[16 kk, 16 kk + 16)` of a
/// swizzled row-major tile.
#[cube]
fn load_a<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    tile: &[Vector<E, V8>],
    row0: u32,
    kk: u32,
    #[comptime] chunks: u32,
) -> Array<Vector<E, R2>> {
    let lane = UNIT_POS_PLANE;
    let idx = swizzle(row0 + lane % 16, 2 * kk + lane / 16, chunks) as usize;
    def.load_matrix::<Vector<E, V8>, R2>(&tile[idx..idx + 1], MatrixIdent::A, 4usize, false)
}

/// The A operand (16 x 16) of `X^T`, X a swizzled row-major tile whose rows are the k
/// dimension: A's rows are X's columns `[m0, m0 + 16)`, A's columns X's rows `[k0, k0 + 16)`.
#[cube]
fn load_a_transposed<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    tile: &[Vector<E, V8>],
    m0: u32,
    k0: u32,
    #[comptime] chunks: u32,
) -> Array<Vector<E, R2>> {
    let lane = UNIT_POS_PLANE;
    let matrix = lane / 8;
    let idx = swizzle(
        k0 + lane % 8 + 8 * (matrix / 2),
        m0 / 8 + matrix % 2,
        chunks,
    ) as usize;
    def.load_matrix::<Vector<E, V8>, R2>(&tile[idx..idx + 1], MatrixIdent::A, 4usize, true)
}

/// B operands of `X^T` for two adjacent 8-column tiles, X a swizzled row-major tile whose rows
/// are the n dimension: X's rows `[n0, n0 + 16)`, columns `[16 kk, 16 kk + 16)`. Returns the
/// two registers of the first tile, then the two of the second.
#[cube]
fn load_b_rows_pair<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    tile: &[Vector<E, V8>],
    n0: u32,
    kk: u32,
    #[comptime] chunks: u32,
) -> Array<Vector<E, R2>> {
    let lane = UNIT_POS_PLANE;
    let idx = swizzle(
        n0 + lane % 8 + 8 * (lane / 16),
        2 * kk + (lane / 8) % 2,
        chunks,
    ) as usize;
    def.load_matrix::<Vector<E, V8>, R2>(&tile[idx..idx + 1], MatrixIdent::B, 4usize, false)
}

/// B operands of `X^T` for one 8-column tile over two k-steps, X's rows `[n0, n0 + 8)` and
/// columns `[16 kk, 16 kk + 32)`. Returns the two registers of step `kk`, then of `kk + 1`.
#[cube]
fn load_b_rows_steps<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    tile: &[Vector<E, V8>],
    n0: u32,
    kk: u32,
    #[comptime] chunks: u32,
) -> Array<Vector<E, R2>> {
    let lane = UNIT_POS_PLANE;
    let idx = swizzle(n0 + lane % 8, 2 * kk + lane / 8, chunks) as usize;
    def.load_matrix::<Vector<E, V8>, R2>(&tile[idx..idx + 1], MatrixIdent::B, 4usize, false)
}

/// B operands of X for two adjacent 8-column tiles, X a swizzled row-major tile whose rows are
/// the k dimension: X's rows `[k0, k0 + 16)`, columns `[16 np, 16 np + 16)`.
#[cube]
fn load_b_cols_pair<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    tile: &[Vector<E, V8>],
    k0: u32,
    np: u32,
    #[comptime] chunks: u32,
) -> Array<Vector<E, R2>> {
    let lane = UNIT_POS_PLANE;
    let matrix = lane / 8;
    let idx = swizzle(
        k0 + lane % 8 + 8 * (matrix % 2),
        2 * np + matrix / 2,
        chunks,
    ) as usize;
    def.load_matrix::<Vector<E, V8>, R2>(&tile[idx..idx + 1], MatrixIdent::B, 4usize, true)
}

/// The A operand of `X^T` from a 32-column tile of two-element vectors (the `P` and `dS`
/// tiles): A's rows are X's columns `[m0, m0 + 16)`, A's columns X's rows `[k0, k0 + 16)`.
#[cube]
fn load_a_transposed_narrow<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    tile: &[Vector<E, R2>],
    m0: u32,
    k0: u32,
) -> Array<Vector<E, R2>> {
    let lane = UNIT_POS_PLANE;
    let matrix = lane / 8;
    let chunk = swizzle(k0 + lane % 8 + 8 * (matrix / 2), m0 / 8 + matrix % 2, 4u32);
    let idx = (chunk * 4) as usize;
    def.load_matrix::<Vector<E, R2>, R2>(&tile[idx..idx + 4], MatrixIdent::A, 4usize, true)
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

/// The maximum over the four lanes of a quad (the lanes holding one accumulator row).
#[cube]
fn quad_max(value: f32) -> f32 {
    let value = max(value, plane_shuffle_xor(value, 1u32));
    max(value, plane_shuffle_xor(value, 2u32))
}

/// The sum over the four lanes of a quad.
#[cube]
fn quad_sum(value: f32) -> f32 {
    let value = value + plane_shuffle_xor(value, 1u32);
    value + plane_shuffle_xor(value, 2u32)
}

/// `scale * s` for the two columns `col`, `col + 1` of an accumulator vector of row `row` of the
/// block, which sees the keys `[rows[2 row], rows[2 row + 1])`: `-inf` outside them and, with
/// `padded`, where the tile's padding flag (`pad[col - k0]`, the tile starting at key `k0`) is
/// set; only computed when `masked`, so that only the tiles that need the bounds read them (a
/// kernel at its register limit cannot keep them in registers across its loop).
#[cube]
fn scale_and_mask(
    s: Vector<f32, R2>,
    scale: f32,
    col: i32,
    rows: &[i32],
    row: u32,
    pad: &[i32],
    k0: i32,
    masked: bool,
    #[comptime] padded: bool,
) -> Vector<f32, R2> {
    let mut out = s * Vector::new(scale);
    if masked {
        let lo = rows[(2 * row) as usize];
        let hi = rows[(2 * row + 1) as usize];
        #[unroll]
        for e in 0..2u32 {
            let c = col + e as i32;
            let mut visible = c >= lo && c < hi;
            if comptime!(padded) {
                visible = visible && pad[(c - k0) as usize] == 0i32;
            }
            out.insert(
                e as usize,
                select(visible, out.extract(e as usize), f32::NEG_INFINITY),
            );
        }
    }
    out
}

/// The sequence holding position `pos` among the `seqs` sequences whose offsets are `cu`
/// (`[seqs + 1]`, not decreasing): the last `s` with `cu[s] <= pos`, so `-1` before the first
/// offset and `seqs` from the last one on, which belong to no sequence. A binary search.
#[cube]
fn sequence_of(cu: &Tensor<i32>, seqs: u32, pos: i32) -> i32 {
    let mut lo = 0u32;
    let mut hi = seqs + 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cu[mid as usize] <= pos {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo as i32 - 1
}

/// The keys query `row` of `tq` sees when the queries are not packed, `[lo, hi)` (its position
/// `row + tk - tq`, from `left` before it to `right` after it), or `(tk, 0)` when it sees none or
/// is past the queries: [`row_bounds()`] without the offsets, arithmetic alone.
#[cube]
fn aligned_bounds(row: u32, tq: u32, tk: u32, left: i32, right: i32) -> (i32, i32) {
    let keys = tk as i32;
    let pos = row as i32 + keys - tq as i32;
    let mut lo = max(0i32, pos - left);
    let mut hi = min(keys, pos + right + 1);
    if row >= tq || hi <= lo {
        lo = keys;
        hi = 0i32;
    }
    (lo, hi)
}

/// The keys query `row` sees, `[lo, hi)`, or `(tk, 0)` when it sees none or is past the `tq`
/// queries. Its position is `row + tk - tq`, or, packed, `i + Lk - Lq` within its sequence of
/// `Lq` queries and `Lk` keys (`i` its place among the queries); it sees the keys from `left`
/// before that position to `right` after it, within its sequence. `left` and `right` are at most
/// `tq + tk + 1`, which stands for no bound, so that nothing below leaves `i32`.
#[cube]
fn row_bounds(
    row: u32,
    tq: u32,
    tk: u32,
    left: i32,
    right: i32,
    cu_q: &Tensor<i32>,
    cu_k: &Tensor<i32>,
    seqs: u32,
    #[comptime] packed: bool,
) -> (i32, i32) {
    let keys = tk as i32;
    let mut lo = keys;
    let mut hi = 0i32;
    if comptime!(packed) {
        if row < tq {
            let s = sequence_of(cu_q, seqs, row as i32);
            if s >= 0 && s < seqs as i32 {
                let q_start = cu_q[s as usize];
                let q_len = cu_q[(s + 1) as usize] - q_start;
                let k_start = cu_k[s as usize];
                let k_len = cu_k[(s + 1) as usize] - k_start;
                let pos = row as i32 - q_start + k_len - q_len;
                lo = max(k_start + max(0i32, pos - left), 0i32);
                hi = min(k_start + min(k_len, pos + right + 1), keys);
            }
            if hi <= lo {
                lo = keys;
                hi = 0i32;
            }
        }
    } else {
        let (l, h) = aligned_bounds(row, tq, tk, left, right);
        lo = l;
        hi = h;
    }
    (lo, hi)
}

/// The queries that see some key of `[k0, k1)`: `[first, end)`, where `first` is the first query
/// whose interval ends after `k0` and `end` one past the last whose interval starts before
/// `k1`, from [`row_bounds()`]'s definitions solved for the query (the intervals never move
/// back, so the queries between see the block too, or nothing at all). Packed, `k0` and
/// `k1 - 1` fall in the sequences whose queries hold `first` and `end`.
#[cube]
fn query_range(
    k0: i32,
    k1: i32,
    tq: u32,
    tk: u32,
    left: i32,
    right: i32,
    cu_q: &Tensor<i32>,
    cu_k: &Tensor<i32>,
    seqs: u32,
    #[comptime] packed: bool,
) -> (i32, i32) {
    let queries = tq as i32;
    let mut first = 0i32;
    let mut end = 0i32;
    if comptime!(packed) {
        let last = seqs as i32;
        let sa = sequence_of(cu_k, seqs, k0);
        // From the last offset on, no query sees a key.
        if sa < last {
            if sa >= 0 {
                let k_start = cu_k[sa as usize];
                let k_len = cu_k[(sa + 1) as usize] - k_start;
                let q_start = cu_q[sa as usize];
                let q_len = cu_q[(sa + 1) as usize] - q_start;
                first = q_start + max(0i32, min(k0 - k_start - (k_len - q_len) - right, q_len));
            }
            let sb = sequence_of(cu_k, seqs, k1 - 1);
            if sb >= last {
                end = cu_q[seqs as usize];
            } else if sb >= 0 {
                let k_start = cu_k[sb as usize];
                let k_len = cu_k[(sb + 1) as usize] - k_start;
                let q_start = cu_q[sb as usize];
                let q_len = cu_q[(sb + 1) as usize] - q_start;
                end = q_start + max(0i32, min(k1 - k_start - (k_len - q_len) + left, q_len));
            }
        }
    } else {
        let offset = tk as i32 - queries;
        first = k0 - offset - right;
        end = k1 - offset + left;
    }
    (max(0i32, min(first, queries)), max(0i32, min(end, queries)))
}

/// `S = Q K^T` for one warp's 16 query rows `[row0, row0 + 16)` of `tile_q` against the `keys`
/// keys `[key0, key0 + keys)` of `tile_k`: `keys / 8` accumulator tiles.
#[cube]
fn scores<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    tile_q: &[Vector<E, V8>],
    tile_k: &[Vector<E, V8>],
    row0: u32,
    key0: u32,
    #[comptime] d: u32,
    #[comptime] keys: u32,
) -> Array<Vector<f32, R2>> {
    let chunks = comptime!(d / 8);
    let mut acc = zeros(comptime!(keys / 4));
    #[unroll]
    for kk in 0..comptime!(d / 16) {
        let a = load_a::<E>(def, tile_q, row0, kk, chunks);
        #[unroll]
        for np in 0..comptime!(keys / 16) {
            let b = load_b_rows_pair::<E>(def, tile_k, key0 + np * 16, kk, chunks);
            mma::<E>(def, &a, b[0], b[1], &mut acc, 2 * np);
            mma::<E>(def, &a, b[2], b[3], &mut acc, 2 * np + 1);
        }
    }
    acc
}

/// `acc += P V` (or `dS K`): the warp's `keys / 8` accumulator tiles of probabilities as the
/// left operand, against rows `[key0, key0 + keys)` of `tile_v`, into `d / 8` output tiles.
#[cube]
fn accumulate_pv<E: Float>(
    def: &MmaDefinition<E, E, f32>,
    probs: &Array<Vector<f32, R2>>,
    tile_v: &[Vector<E, V8>],
    acc: &mut Array<Vector<f32, R2>>,
    key0: u32,
    #[comptime] d: u32,
    #[comptime] keys: u32,
) {
    let chunks = comptime!(d / 8);
    #[unroll]
    for kk in 0..comptime!(keys / 16) {
        let a = accumulator_as_a::<E>(probs, 2 * kk);
        #[unroll]
        for np in 0..comptime!(d / 16) {
            let b = load_b_cols_pair::<E>(def, tile_v, key0 + kk * 16, np, chunks);
            mma::<E>(def, &a, b[0], b[1], acc, 2 * np);
            mma::<E>(def, &a, b[2], b[3], acc, 2 * np + 1);
        }
    }
}

/// Whether a warp's tile (keys `[start, start + keys)`, the warp's 16 rows from `row0`) needs
/// masking element by element: unless every row sees every key. Packed, from the warp's spans
/// (`spans[at + 2]`, the latest start of its rows' intervals, and `spans[at + 3]`, the earliest
/// end); otherwise in closed form, from the aligned positions of its first and last rows.
#[cube]
fn tile_masked(
    row0: u32,
    start: i32,
    keys: i32,
    tq: u32,
    tk: u32,
    left: i32,
    right: i32,
    spans: &[i32],
    at: usize,
    #[comptime] packed: bool,
) -> bool {
    if comptime!(packed) {
        !(spans[at + 2] <= start && start + keys <= spans[at + 3])
    } else {
        let pos = row0 as i32 + tk as i32 - tq as i32;
        !(pos + 15 - left <= start && start + keys - 1 <= pos + right && start + keys <= tk as i32)
    }
}

/// What a block of queries sees. Each lane's rows `row_a` and `row_b` (`local_a` and `local_b`
/// within the block) get their intervals ([`row_bounds()`]) written to `rows` (`2 r`, `2 r + 1`
/// for row `r`), and warp group `group` of `groups` writes to `spans[4 group..4 group + 4]` the
/// span of its rows' keys (the union's start and end) and the keys all of them see (the latest
/// start and the earliest end), when `writes` (the warps of a group compute the same). After a
/// barrier every thread reads the spans and returns the block's first key tile of `tile` keys
/// and how many follow.
#[cube]
fn block_bounds(
    rows: &mut [i32],
    spans: &mut [i32],
    row_a: u32,
    row_b: u32,
    local_a: u32,
    local_b: u32,
    group: u32,
    writes: bool,
    tq: u32,
    tk: u32,
    left: i32,
    right: i32,
    cu_q: &Tensor<i32>,
    cu_k: &Tensor<i32>,
    seqs: u32,
    #[comptime] packed: bool,
    #[comptime] groups: u32,
    #[comptime] tile: u32,
) -> (u32, u32) {
    let (lo_a, hi_a) = row_bounds(row_a, tq, tk, left, right, cu_q, cu_k, seqs, packed);
    let (lo_b, hi_b) = row_bounds(row_b, tq, tk, left, right, cu_q, cu_k, seqs, packed);
    let span_lo = plane_min(min(lo_a, lo_b));
    let span_hi = plane_max(max(hi_a, hi_b));
    let all_lo = plane_max(max(lo_a, lo_b));
    let all_hi = plane_min(min(hi_a, hi_b));
    if writes {
        if UNIT_POS_PLANE % 4 == 0 {
            rows[(2 * local_a) as usize] = lo_a;
            rows[(2 * local_a + 1) as usize] = hi_a;
            rows[(2 * local_b) as usize] = lo_b;
            rows[(2 * local_b + 1) as usize] = hi_b;
        }
        if UNIT_POS_PLANE == 0 {
            spans[(4 * group) as usize] = span_lo;
            spans[(4 * group + 1) as usize] = span_hi;
            spans[(4 * group + 2) as usize] = all_lo;
            spans[(4 * group + 3) as usize] = all_hi;
        }
    }
    sync_cube();
    let mut lo = tk as i32;
    let mut hi = 0i32;
    #[unroll]
    for w in 0..groups {
        lo = min(lo, spans[(4 * w) as usize]);
        hi = max(hi, spans[(4 * w + 1) as usize]);
    }
    let size = tile as i32;
    let first = lo / size;
    let count = select(hi > lo, (hi + size - 1) / size - first, 0i32);
    (first as u32, count as u32)
}

/// The padding flags of keys `[k0, k0 + keys)` of batch element `b`, as `fetch_rows()` reads a
/// tile: thread `UNIT_POS < keys` reads that key's flag (1 past `limit`).
#[cube]
fn fetch_padding(
    key_mask: &Tensor<i32>,
    base: usize,
    stride: u32,
    k0: u32,
    limit: u32,
    #[comptime] keys: u32,
) -> i32 {
    let mut flag = 1i32;
    let key = k0 + UNIT_POS;
    if UNIT_POS < keys && key < limit {
        flag = key_mask[base + key as usize * stride as usize];
    }
    flag
}

/// The training forward. One cube per (`block_q`-query block, query head, batch), a warp per 16
/// query rows; query blocks are taken last first, so that the longest causal rows start first.
/// The cube sweeps the key blocks its queries see. Writes the output, contiguous
/// `[B, Tq, Hq, D]`, and the natural log-sum-exp of
/// each query's scaled scores over the keys it sees, contiguous `[B, Tq, Hq]` (`-inf`, with a
/// zero output, for a query that sees none). Input strides are in 8-element vectors.
#[cube(launch_unchecked)]
pub(crate) fn forward_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    o: &mut Tensor<Vector<E, R2>>,
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
    #[comptime] block_q: u32,
    #[comptime] block_kv: u32,
    #[comptime] padded: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let threads = comptime!(2 * block_q);
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
    let chunks = comptime!(d / 8);
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let g = lane / 4;
    let t = lane % 4;

    let h = CUBE_POS_Y;
    let b = CUBE_POS_Z;
    let hq = CUBE_COUNT_Y;
    let hk = h / group;
    let q0 = (q_blocks - 1 - CUBE_POS_X) * block_q;

    let q_base = b as usize * q_sb as usize + h as usize * q_sh as usize;
    let k_base = b as usize * k_sb as usize + hk as usize * k_sh as usize;
    let v_base = b as usize * v_sb as usize + hk as usize * v_sh as usize;
    let m_base = b as usize * m_sb as usize;

    // The keys each row of the block sees, each warp's span of them, and the block's key
    // tiles.
    let local_a = warp * 16 + g;
    let local_b = local_a + 8;
    let row_a = q0 + local_a;
    let row_b = q0 + local_b;
    let mut s_rows = Shared::<[i32]>::new_slice(comptime!((2 * block_q) as usize));
    let mut s_span = Shared::<[i32]>::new_slice(comptime!((block_q / 4) as usize));
    let (first_tile, n_blocks) = block_bounds(
        s_rows.as_mut_slice(),
        s_span.as_mut_slice(),
        row_a,
        row_b,
        local_a,
        local_b,
        warp,
        true,
        tq,
        tk,
        left,
        right,
        cu_q,
        cu_k,
        seqs,
        packed,
        comptime!(block_q / 16),
        block_kv,
    );
    let key0 = first_tile * block_kv;
    let k_limit = select(n_blocks > 0, tk, 0u32);

    let mut s_q = Shared::<[Vector<E, V8>]>::new_aligned_slice(
        comptime!((block_q * d / 8) as usize),
        16usize,
    );
    let mut s_k = Shared::<[Vector<E, V8>]>::new_aligned_slice(
        comptime!((block_kv * d / 8) as usize),
        16usize,
    );
    let mut s_v = Shared::<[Vector<E, V8>]>::new_aligned_slice(
        comptime!((block_kv * d / 8) as usize),
        16usize,
    );
    let mut s_pad =
        Shared::<[i32]>::new_slice(comptime!(if padded { block_kv } else { 1u32 }) as usize);

    load_rows::<E>(
        q,
        s_q.as_mut_slice(),
        q_base,
        q_st as usize,
        q0,
        tq,
        block_q,
        chunks,
        threads,
    );
    load_rows::<E>(
        k,
        s_k.as_mut_slice(),
        k_base,
        k_st as usize,
        key0,
        k_limit,
        block_kv,
        chunks,
        threads,
    );
    load_rows::<E>(
        v,
        s_v.as_mut_slice(),
        v_base,
        v_st as usize,
        key0,
        k_limit,
        block_kv,
        chunks,
        threads,
    );
    if comptime!(padded) {
        let flag = fetch_padding(key_mask, m_base, m_st, key0, k_limit, block_kv);
        if UNIT_POS < block_kv {
            s_pad[UNIT_POS as usize] = flag;
        }
    }

    let mut max_a = f32::new(f32::NEG_INFINITY);
    let mut max_b = f32::new(f32::NEG_INFINITY);
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut acc_o = zeros(comptime!(d / 4));
    let tile = block_kv as i32;

    for j in 0..n_blocks {
        let k0 = key0 + j * block_kv;
        // K(j) and V(j) are in shared memory: read K(j + 1) and V(j + 1) meanwhile.
        sync_cube();
        let limit = select(j + 1 < n_blocks, tk, 0u32);
        let k_next = fetch_rows::<E>(
            k,
            k_base,
            k_st as usize,
            k0 + block_kv,
            limit,
            block_kv,
            chunks,
            threads,
        );
        let v_next = fetch_rows::<E>(
            v,
            v_base,
            v_st as usize,
            k0 + block_kv,
            limit,
            block_kv,
            chunks,
            threads,
        );
        let mut pad_next = 1i32;
        if comptime!(padded) {
            pad_next = fetch_padding(key_mask, m_base, m_st, k0 + block_kv, limit, block_kv);
        }

        let start = k0 as i32;
        let span = (4 * warp) as usize;
        {
            let mut acc_s = scores::<E>(
                &def,
                s_q.as_slice(),
                s_k.as_slice(),
                warp * 16,
                0u32,
                d,
                block_kv,
            );

            // Mask unless every row of the warp sees the whole tile (rows past the queries
            // aside: nothing of theirs is written).
            let masked = padded
                || tile_masked(
                    q0 + warp * 16,
                    start,
                    tile,
                    tq,
                    tk,
                    left,
                    right,
                    s_span.as_slice(),
                    span,
                    packed,
                );
            let mut tile_max_a = max_a;
            let mut tile_max_b = max_b;
            #[unroll]
            for nt in 0..comptime!(block_kv / 8) {
                let col = start + (nt * 8 + 2 * t) as i32;
                let sa = scale_and_mask(
                    acc_s[(2 * nt) as usize],
                    scale,
                    col,
                    s_rows.as_slice(),
                    local_a,
                    s_pad.as_slice(),
                    start,
                    masked,
                    padded,
                );
                let sb = scale_and_mask(
                    acc_s[(2 * nt + 1) as usize],
                    scale,
                    col,
                    s_rows.as_slice(),
                    local_b,
                    s_pad.as_slice(),
                    start,
                    masked,
                    padded,
                );
                tile_max_a = max(tile_max_a, max(sa.extract(0usize), sa.extract(1usize)));
                tile_max_b = max(tile_max_b, max(sb.extract(0usize), sb.extract(1usize)));
                acc_s[(2 * nt) as usize] = sa;
                acc_s[(2 * nt + 1) as usize] = sb;
            }
            tile_max_a = quad_max(tile_max_a);
            tile_max_b = quad_max(tile_max_b);
            // A row with every key hidden so far keeps -inf; exponentiate against 0 then, so
            // that its probabilities and rescaling factor are 0 rather than NaN.
            let base_a = select(tile_max_a == f32::NEG_INFINITY, 0.0f32, tile_max_a);
            let base_b = select(tile_max_b == f32::NEG_INFINITY, 0.0f32, tile_max_b);
            let alpha_a = (max_a - base_a).exp();
            let alpha_b = (max_b - base_b).exp();
            max_a = tile_max_a;
            max_b = tile_max_b;
            let mut tile_sum_a = 0.0f32;
            let mut tile_sum_b = 0.0f32;
            #[unroll]
            for nt in 0..comptime!(block_kv / 8) {
                let pa = (acc_s[(2 * nt) as usize] - Vector::new(base_a)).exp();
                let pb = (acc_s[(2 * nt + 1) as usize] - Vector::new(base_b)).exp();
                tile_sum_a += pa.extract(0usize) + pa.extract(1usize);
                tile_sum_b += pb.extract(0usize) + pb.extract(1usize);
                acc_s[(2 * nt) as usize] = pa;
                acc_s[(2 * nt + 1) as usize] = pb;
            }
            // Each lane sums its own columns; the quad's lanes are added once, after the loop.
            sum_a = sum_a * alpha_a + tile_sum_a;
            sum_b = sum_b * alpha_b + tile_sum_b;
            #[unroll]
            for nt in 0..comptime!(d / 8) {
                acc_o[(2 * nt) as usize] = acc_o[(2 * nt) as usize] * Vector::new(alpha_a);
                acc_o[(2 * nt + 1) as usize] = acc_o[(2 * nt + 1) as usize] * Vector::new(alpha_b);
            }
            accumulate_pv::<E>(&def, &acc_s, s_v.as_slice(), &mut acc_o, 0u32, d, block_kv);
        }
        sync_cube();
        store_rows::<E>(&k_next, s_k.as_mut_slice(), block_kv, chunks, threads);
        store_rows::<E>(&v_next, s_v.as_mut_slice(), block_kv, chunks, threads);
        if comptime!(padded) {
            if UNIT_POS < block_kv {
                s_pad[UNIT_POS as usize] = pad_next;
            }
        }
    }

    let total_a = quad_sum(sum_a);
    let total_b = quad_sum(sum_b);
    let inv_a = select(total_a > 0.0f32, 1.0f32 / total_a, 0.0f32);
    let inv_b = select(total_b > 0.0f32, 1.0f32 / total_b, 0.0f32);
    // Rows `row_a` and `row_b` of this head in `[B, Tq, Hq]` order.
    let first = b as usize * tq as usize * hq as usize + h as usize;
    let at_a = first + row_a as usize * hq as usize;
    let at_b = first + row_b as usize * hq as usize;
    if t == 0 {
        if row_a < tq {
            lse[at_a] = max_a + total_a.ln();
        }
        if row_b < tq {
            lse[at_b] = max_b + total_b.ln();
        }
    }
    let half_d = comptime!(d / 2) as usize;
    #[unroll]
    for nt in 0..comptime!(d / 8) {
        let col = (nt * 4 + t) as usize;
        if row_a < tq {
            o[at_a * half_d + col] =
                Vector::cast_from(acc_o[(2 * nt) as usize] * Vector::new(inv_a));
        }
        if row_b < tq {
            o[at_b * half_d + col] =
                Vector::cast_from(acc_o[(2 * nt + 1) as usize] * Vector::new(inv_b));
        }
    }
}

/// `delta = rowsum(dO * O)` in f32, contiguous `[B, Tq, Hq]` like the log-sum-exp: `d / 8` lanes
/// per row, eight elements each, 256 threads per cube, rows taken in that order. Beside it, the
/// log-sum-exp as `dQ` and `dK`/`dV` read it, `lse_bwd`: `+inf` where it is `-inf` (a query that
/// sees no key), which makes that query's probabilities `exp(s - lse)` zero rather than NaN.
/// (Guarding the log-sum-exp there instead costs those kernels a tenth of their time: two selects
/// change how `ptxas` schedules their loops.) Packed (`B = 1`),
/// it also writes each query's interval of keys ([`row_bounds()`]) to `bounds`, `[Tq, 2]`, once
/// (from the rows of head 0), so that `dK`/`dV`, which meets each query once per key block, reads
/// it rather than searching the offsets again.
#[cube(launch_unchecked)]
pub(crate) fn delta_kernel<E: Float>(
    o: &Tensor<Vector<E, V8>>,
    d_o: &Tensor<Vector<E, V8>>,
    delta: &mut Tensor<f32>,
    lse: &Tensor<f32>,
    lse_bwd: &mut Tensor<f32>,
    bounds: &mut Tensor<i32>,
    cu_q: &Tensor<i32>,
    cu_k: &Tensor<i32>,
    rows: u32,
    hq: u32,
    tq: u32,
    tk: u32,
    o_sb: u32,
    o_sh: u32,
    o_st: u32,
    do_sb: u32,
    do_sh: u32,
    do_st: u32,
    seqs: u32,
    left: i32,
    right: i32,
    #[comptime] d: u32,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let chunks = comptime!(d / 8);
    let row = CUBE_POS_X * comptime!(256 / chunks) + UNIT_POS / chunks;
    let c = UNIT_POS % chunks;
    let mut acc = 0.0f32;
    if row < rows {
        let h = row % hq;
        let i = (row / hq) % tq;
        let b = row / (hq * tq);
        let ov = o[b as usize * o_sb as usize
            + h as usize * o_sh as usize
            + i as usize * o_st as usize
            + c as usize];
        let gv = d_o[b as usize * do_sb as usize
            + h as usize * do_sh as usize
            + i as usize * do_st as usize
            + c as usize];
        let prod = Vector::<f32, V8>::cast_from(ov) * Vector::<f32, V8>::cast_from(gv);
        #[unroll]
        for e in 0..8u32 {
            acc += prod.extract(e as usize);
        }
    }
    #[unroll]
    for s in 0..comptime!(chunks.trailing_zeros()) {
        acc += plane_shuffle_xor(acc, 1u32 << s);
    }
    if c == 0 && row < rows {
        delta[row as usize] = acc;
        let l = lse[row as usize];
        lse_bwd[row as usize] = select(l == f32::NEG_INFINITY, f32::INFINITY, l);
        if comptime!(packed) {
            let i = row / hq;
            if row % hq == 0 {
                let (lo, hi) = row_bounds(i, tq, tk, left, right, cu_q, cu_k, seqs, true);
                bounds[(2 * i) as usize] = lo;
                bounds[(2 * i + 1) as usize] = hi;
            }
        }
    }
}

/// `dQ`. One cube per (64-query block, query head, batch), eight warps: two per 16 query rows,
/// each taking half of every key block, their partial `dQ` added at the end. Sweeps the key
/// blocks its queries see: `S` and `dP = dO V^T`
/// recomputed per block, `P` from the forward's log-sum-exp, `dS = P (dP - delta)`,
/// `dQ += dS K`. Writes `dQ` contiguous `[B, Tq, Hq, D]`.
#[cube(launch_unchecked)]
pub(crate) fn dq_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    d_o: &Tensor<Vector<E, V8>>,
    lse: &Tensor<f32>,
    delta: &Tensor<f32>,
    dq: &mut Tensor<Vector<E, R2>>,
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
    scale: f32,
    m_sb: u32,
    m_st: u32,
    seqs: u32,
    left: i32,
    right: i32,
    #[comptime] d: u32,
    #[comptime] block_kv: u32,
    #[comptime] padded: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
    let chunks = comptime!(d / 8);
    let half = comptime!(block_kv / 2);
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let g = lane / 4;
    let t = lane % 4;
    // Two warps per 16 query rows, each taking half of every key block.
    let rows = warp % 4;
    let key_half = warp / 4;
    let key0 = key_half * half;

    let h = CUBE_POS_Y;
    let b = CUBE_POS_Z;
    let hq = CUBE_COUNT_Y;
    let hk = h / group;
    let q0 = (q_blocks - 1 - CUBE_POS_X) * BLOCK_Q;

    let q_base = b as usize * q_sb as usize + h as usize * q_sh as usize;
    let do_base = b as usize * do_sb as usize + h as usize * do_sh as usize;
    let k_base = b as usize * k_sb as usize + hk as usize * k_sh as usize;
    let v_base = b as usize * v_sb as usize + hk as usize * v_sh as usize;
    let m_base = b as usize * m_sb as usize;

    // The keys each row of the block sees, each pair of warps' span of them (the two compute
    // the same; the first writes), and the block's key tiles.
    let local_a = rows * 16 + g;
    let local_b = local_a + 8;
    let row_a = q0 + local_a;
    let row_b = q0 + local_b;
    let mut s_rows = Shared::<[i32]>::new_slice(comptime!((2 * BLOCK_Q) as usize));
    let mut s_span = Shared::<[i32]>::new_slice(16usize);
    let (first_tile, n_blocks) = block_bounds(
        s_rows.as_mut_slice(),
        s_span.as_mut_slice(),
        row_a,
        row_b,
        local_a,
        local_b,
        rows,
        key_half == 0,
        tq,
        tk,
        left,
        right,
        cu_q,
        cu_k,
        seqs,
        packed,
        4u32,
        block_kv,
    );
    let first_key = first_tile * block_kv;
    let k_limit = select(n_blocks > 0, tk, 0u32);

    let mut s_q = Shared::<[Vector<E, V8>]>::new_aligned_slice(
        comptime!((BLOCK_Q * d / 8) as usize),
        16usize,
    );
    let mut s_do = Shared::<[Vector<E, V8>]>::new_aligned_slice(
        comptime!((BLOCK_Q * d / 8) as usize),
        16usize,
    );
    let mut s_k = Shared::<[Vector<E, V8>]>::new_aligned_slice(
        comptime!((block_kv * d / 8) as usize),
        16usize,
    );
    let mut s_v = Shared::<[Vector<E, V8>]>::new_aligned_slice(
        comptime!((block_kv * d / 8) as usize),
        16usize,
    );
    let mut s_pad =
        Shared::<[i32]>::new_slice(comptime!(if padded { block_kv } else { 1u32 }) as usize);

    load_rows::<E>(
        q,
        s_q.as_mut_slice(),
        q_base,
        q_st as usize,
        q0,
        tq,
        BLOCK_Q,
        chunks,
        256u32,
    );
    load_rows::<E>(
        d_o,
        s_do.as_mut_slice(),
        do_base,
        do_st as usize,
        q0,
        tq,
        BLOCK_Q,
        chunks,
        256u32,
    );
    load_rows::<E>(
        k,
        s_k.as_mut_slice(),
        k_base,
        k_st as usize,
        first_key,
        k_limit,
        block_kv,
        chunks,
        256u32,
    );
    load_rows::<E>(
        v,
        s_v.as_mut_slice(),
        v_base,
        v_st as usize,
        first_key,
        k_limit,
        block_kv,
        chunks,
        256u32,
    );
    if comptime!(padded) {
        let flag = fetch_padding(key_mask, m_base, m_st, first_key, k_limit, block_kv);
        if UNIT_POS < block_kv {
            s_pad[UNIT_POS as usize] = flag;
        }
    }

    // Rows `row_a` and `row_b` of this head in `[B, Tq, Hq]` order.
    let first = b as usize * tq as usize * hq as usize + h as usize;
    let at_a = first + row_a as usize * hq as usize;
    let at_b = first + row_b as usize * hq as usize;
    let mut lse_a = 0.0f32;
    let mut lse_b = 0.0f32;
    let mut delta_a = 0.0f32;
    let mut delta_b = 0.0f32;
    if row_a < tq {
        lse_a = lse[at_a];
        delta_a = delta[at_a];
    }
    if row_b < tq {
        lse_b = lse[at_b];
        delta_b = delta[at_b];
    }
    let mut acc_dq = zeros(comptime!(d / 4));
    let span = half as i32;

    for j in 0..n_blocks {
        let k0 = first_key + j * block_kv;
        // K(j) and V(j) are in shared memory: read K(j + 1) and V(j + 1) meanwhile.
        sync_cube();
        let limit = select(j + 1 < n_blocks, tk, 0u32);
        let k_next = fetch_rows::<E>(
            k,
            k_base,
            k_st as usize,
            k0 + block_kv,
            limit,
            block_kv,
            chunks,
            256u32,
        );
        let v_next = fetch_rows::<E>(
            v,
            v_base,
            v_st as usize,
            k0 + block_kv,
            limit,
            block_kv,
            chunks,
            256u32,
        );
        let mut pad_next = 1i32;
        if comptime!(padded) {
            pad_next = fetch_padding(key_mask, m_base, m_st, k0 + block_kv, limit, block_kv);
        }

        // This warp's half of the tile.
        let start = (k0 + key0) as i32;
        let at = (4 * rows) as usize;
        {
            let mut acc_s = scores::<E>(
                &def,
                s_q.as_slice(),
                s_k.as_slice(),
                rows * 16,
                key0,
                d,
                half,
            );
            let acc_dp = scores::<E>(
                &def,
                s_do.as_slice(),
                s_v.as_slice(),
                rows * 16,
                key0,
                d,
                half,
            );

            // Mask unless every row of the warp sees the whole half (rows past the queries
            // aside: nothing of theirs is written).
            let masked = padded
                || tile_masked(
                    q0 + rows * 16,
                    start,
                    span,
                    tq,
                    tk,
                    left,
                    right,
                    s_span.as_slice(),
                    at,
                    packed,
                );
            #[unroll]
            for nt in 0..comptime!(half / 8) {
                let col = start + (nt * 8 + 2 * t) as i32;
                let sa = scale_and_mask(
                    acc_s[(2 * nt) as usize],
                    scale,
                    col,
                    s_rows.as_slice(),
                    local_a,
                    s_pad.as_slice(),
                    k0 as i32,
                    masked,
                    padded,
                );
                let sb = scale_and_mask(
                    acc_s[(2 * nt + 1) as usize],
                    scale,
                    col,
                    s_rows.as_slice(),
                    local_b,
                    s_pad.as_slice(),
                    k0 as i32,
                    masked,
                    padded,
                );
                let pa = (sa - Vector::new(lse_a)).exp();
                let pb = (sb - Vector::new(lse_b)).exp();
                acc_s[(2 * nt) as usize] = pa * (acc_dp[(2 * nt) as usize] - Vector::new(delta_a));
                acc_s[(2 * nt + 1) as usize] =
                    pb * (acc_dp[(2 * nt + 1) as usize] - Vector::new(delta_b));
            }
            accumulate_pv::<E>(&def, &acc_s, s_k.as_slice(), &mut acc_dq, key0, d, half);
        }
        sync_cube();
        store_rows::<E>(&k_next, s_k.as_mut_slice(), block_kv, chunks, 256u32);
        store_rows::<E>(&v_next, s_v.as_mut_slice(), block_kv, chunks, 256u32);
        if comptime!(padded) {
            if UNIT_POS < block_kv {
                s_pad[UNIT_POS as usize] = pad_next;
            }
        }
    }

    // The second warp of each pair hands its partial dQ to the first through the query and
    // output-gradient tiles, now free: four f32 per 16-byte slot, lane-minor, in the same
    // register order on both sides.
    let slots = comptime!(d / 8);
    let region = (warp % 2) * comptime!(32 * d / 8);
    sync_cube();
    if key_half == 1 {
        #[unroll]
        for j in 0..slots {
            let mut packed_dq = Vector::<f32, Const<4>>::empty();
            packed_dq.insert(0usize, acc_dq[(2 * j) as usize].extract(0usize));
            packed_dq.insert(1usize, acc_dq[(2 * j) as usize].extract(1usize));
            packed_dq.insert(2usize, acc_dq[(2 * j + 1) as usize].extract(0usize));
            packed_dq.insert(3usize, acc_dq[(2 * j + 1) as usize].extract(1usize));
            let at = (region + j * 32 + lane) as usize;
            if rows < 2 {
                s_q[at] = Vector::<E, V8>::reinterpret(packed_dq);
            } else {
                s_do[at] = Vector::<E, V8>::reinterpret(packed_dq);
            }
        }
    }
    sync_cube();
    if key_half == 0 {
        let half_d = comptime!(d / 2) as usize;
        #[unroll]
        for j in 0..slots {
            let at = (region + j * 32 + lane) as usize;
            let mut slot = s_q[at];
            if rows >= 2 {
                slot = s_do[at];
            }
            let other = Vector::<f32, Const<4>>::reinterpret(slot);
            let mut lo = acc_dq[(2 * j) as usize];
            let mut hi = acc_dq[(2 * j + 1) as usize];
            lo.insert(0usize, lo.extract(0usize) + other.extract(0usize));
            lo.insert(1usize, lo.extract(1usize) + other.extract(1usize));
            hi.insert(0usize, hi.extract(0usize) + other.extract(2usize));
            hi.insert(1usize, hi.extract(1usize) + other.extract(3usize));
            // Accumulator vector 2j is row g, 2j + 1 row g + 8, both columns 8 j + 2 t.
            let col = (j * 4 + t) as usize;
            if row_a < tq {
                dq[at_a * half_d + col] = Vector::cast_from(lo * Vector::new(scale));
            }
            if row_b < tq {
                dq[at_b * half_d + col] = Vector::cast_from(hi * Vector::new(scale));
            }
        }
    }
}

/// Whether a step of `dK`/`dV` (queries `[q0, q0 + 32)`, keys `[k0, k0 + 32)`) needs masking
/// element by element: unless every query sees every key. Packed, from the queries' intervals in
/// `s_lo` and `s_hi`: the last query's starts at or before `k0` and the first one's ends at or
/// after the block's end (the intervals never move back). Otherwise in closed form from the
/// aligned positions, the queries all before `tq` and the keys all before `tk`.
#[cube]
fn step_masked(
    q0: u32,
    k0: u32,
    tq: u32,
    tk: u32,
    left: i32,
    right: i32,
    s_lo: &[i32],
    s_hi: &[i32],
    #[comptime] packed: bool,
) -> bool {
    let start = k0 as i32;
    let block = BLOCK_KV as i32;
    if comptime!(packed) {
        !(s_lo[(BLOCK_KV - 1) as usize] <= start && start + block <= s_hi[0usize])
    } else {
        let pos = q0 as i32 + tk as i32 - tq as i32;
        !(q0 + BLOCK_KV <= tq
            && k0 + BLOCK_KV <= tk
            && pos + block - 1 - left <= start
            && start + block - 1 <= pos + right)
    }
}

/// The keys query `row` of a `dK`/`dV` step starting at query `q0` sees: packed, staged in `s_lo`
/// and `s_hi`; otherwise [`aligned_bounds()`].
#[cube]
fn step_bounds(
    q0: u32,
    row: u32,
    tq: u32,
    tk: u32,
    left: i32,
    right: i32,
    s_lo: &[i32],
    s_hi: &[i32],
    #[comptime] packed: bool,
) -> (i32, i32) {
    if comptime!(packed) {
        (s_lo[row as usize], s_hi[row as usize])
    } else {
        aligned_bounds(q0 + row, tq, tk, left, right)
    }
}

/// `dK` and `dV`. One cube per (32-key block, key-value head, batch), eight warps, sweeping the
/// query heads of the group and, for each, the 32-query steps of the queries that see the block
/// ([`query_range()`]). `S` and `dP` are split 2 x 4 over the warps (16 queries x 8 keys each);
/// `P` and `dS` go through shared memory to the 2 x 4 split of `dV += P^T dO` and
/// `dK += dS^T Q` (16 keys x D/4 columns each). Writes `dK` and `dV` contiguous
/// `[B, Tk, Hkv, D]`.
#[cube(launch_unchecked)]
pub(crate) fn dkdv_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    d_o: &Tensor<Vector<E, V8>>,
    lse: &Tensor<f32>,
    delta: &Tensor<f32>,
    dk: &mut Tensor<Vector<E, R2>>,
    dv: &mut Tensor<Vector<E, R2>>,
    key_mask: &Tensor<i32>,
    bounds: &Tensor<i32>,
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
    #[comptime] padded: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 8usize, 16usize);
    let chunks = comptime!(d / 8);
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let g = lane / 4;
    let t = lane % 4;

    let hk = CUBE_POS_Y;
    let b = CUBE_POS_Z;
    let hkv = CUBE_COUNT_Y;
    let hq = hkv * group;
    let k0 = CUBE_POS_X * BLOCK_KV;

    let k_base = b as usize * k_sb as usize + hk as usize * k_sh as usize;
    let v_base = b as usize * v_sb as usize + hk as usize * v_sh as usize;

    let tile = comptime!((BLOCK_KV * d / 8) as usize);
    let mut s_k = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let mut s_v = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let mut s_q = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let mut s_do = Shared::<[Vector<E, V8>]>::new_aligned_slice(tile, 16usize);
    let narrow = comptime!((BLOCK_KV * BLOCK_KV / 2) as usize);
    let mut s_p = Shared::<[Vector<E, R2>]>::new_aligned_slice(narrow, 16usize);
    let mut s_ds = Shared::<[Vector<E, R2>]>::new_aligned_slice(narrow, 16usize);
    let mut s_lse = Shared::<[f32]>::new_slice(comptime!(BLOCK_KV as usize));
    let mut s_delta = Shared::<[f32]>::new_slice(comptime!(BLOCK_KV as usize));
    // Packed, the keys each query of the step sees, read from `bounds`; otherwise they are
    // computed where needed.
    let staged = comptime!(if packed { BLOCK_KV } else { 1u32 }) as usize;
    let mut s_lo = Shared::<[i32]>::new_slice(staged);
    let mut s_hi = Shared::<[i32]>::new_slice(staged);

    load_rows::<E>(
        k,
        s_k.as_mut_slice(),
        k_base,
        k_st as usize,
        k0,
        tk,
        BLOCK_KV,
        chunks,
        256u32,
    );
    load_rows::<E>(
        v,
        s_v.as_mut_slice(),
        v_base,
        v_st as usize,
        k0,
        tk,
        BLOCK_KV,
        chunks,
        256u32,
    );

    // S and dP: queries [16 rg, 16 rg + 16) of the step, keys [8 cg, 8 cg + 8) of the block.
    // dK and dV: keys [16 rg, 16 rg + 16), columns [cols cg, cols cg + cols).
    let rg = warp / 4;
    let cg = warp % 4;
    let cols = comptime!(d / 4);
    let mut acc_dk = zeros(comptime!(d / 16));
    let mut acc_dv = zeros(comptime!(d / 16));

    // This lane's two keys of the block, and whether they are padding.
    let key_col = 8 * cg + 2 * t;
    let col = (k0 + key_col) as i32;
    let mut pad_0 = false;
    let mut pad_1 = false;
    if comptime!(padded) {
        let m_base = b as usize * m_sb as usize;
        let c = k0 + key_col;
        if c < tk {
            pad_0 = key_mask[m_base + c as usize * m_st as usize] != 0i32;
        }
        if c + 1 < tk {
            pad_1 = key_mask[m_base + (c + 1) as usize * m_st as usize] != 0i32;
        }
    }

    // The queries that see some key of the block, in 32-query steps.
    let k1 = min(k0 + BLOCK_KV, tk);
    let (first_query, end_query) = query_range(
        k0 as i32, k1 as i32, tq, tk, left, right, cu_q, cu_k, seqs, packed,
    );
    let first_step = first_query as u32 / BLOCK_KV;
    let end_step = (end_query as u32 + BLOCK_KV - 1) / BLOCK_KV;
    let per_head = select(end_query > first_query, end_step - first_step, 0u32);
    let qr_a = 16 * rg + g;
    let qr_b = qr_a + 8;

    // The steps of all the group's query heads, one after the other: step `it` is query block
    // `first_step + it % per_head` of head `hk * group + it / per_head`. Each step reads the
    // next step's queries, output gradients and row statistics into registers while it computes.
    let total = group * per_head;
    // The statistics of query `i` of head `h` are element `stats + i Hq + h`, `[B, Tq, Hq]`.
    let stats = b as usize * tq as usize * hq as usize;
    if total > 0 {
        let h = hk * group;
        let q_base = b as usize * q_sb as usize + h as usize * q_sh as usize;
        let do_base = b as usize * do_sb as usize + h as usize * do_sh as usize;
        let q0 = first_step * BLOCK_KV;
        load_rows::<E>(
            q,
            s_q.as_mut_slice(),
            q_base,
            q_st as usize,
            q0,
            tq,
            BLOCK_KV,
            chunks,
            256u32,
        );
        load_rows::<E>(
            d_o,
            s_do.as_mut_slice(),
            do_base,
            do_st as usize,
            q0,
            tq,
            BLOCK_KV,
            chunks,
            256u32,
        );
        if UNIT_POS < BLOCK_KV {
            let row = q0 + UNIT_POS;
            let mut l = 0.0f32;
            let mut dl = 0.0f32;
            if row < tq {
                let at = stats + row as usize * hq as usize + h as usize;
                l = lse[at];
                dl = delta[at];
            }
            s_lse[UNIT_POS as usize] = l;
            s_delta[UNIT_POS as usize] = dl;
            if comptime!(packed) {
                let mut lo = tk as i32;
                let mut hi = 0i32;
                if row < tq {
                    lo = bounds[(2 * row) as usize];
                    hi = bounds[(2 * row + 1) as usize];
                }
                s_lo[UNIT_POS as usize] = lo;
                s_hi[UNIT_POS as usize] = hi;
            }
        }
    }

    for it in 0..total {
        let q0 = (first_step + it % per_head) * BLOCK_KV;
        // This step's queries, output gradients and statistics are in shared memory.
        sync_cube();
        let next = it + 1;
        let has_next = next < total;
        let next_h = hk * group + next / per_head;
        let next_q0 = (first_step + next % per_head) * BLOCK_KV;
        let next_limit = select(has_next, tq, 0u32);
        let next_q_base = b as usize * q_sb as usize + next_h as usize * q_sh as usize;
        let next_do_base = b as usize * do_sb as usize + next_h as usize * do_sh as usize;
        let q_next = fetch_rows::<E>(
            q,
            next_q_base,
            q_st as usize,
            next_q0,
            next_limit,
            BLOCK_KV,
            chunks,
            256u32,
        );
        let do_next = fetch_rows::<E>(
            d_o,
            next_do_base,
            do_st as usize,
            next_q0,
            next_limit,
            BLOCK_KV,
            chunks,
            256u32,
        );
        let mut lse_next = 0.0f32;
        let mut delta_next = 0.0f32;
        let mut lo_next = tk as i32;
        let mut hi_next = 0i32;
        if UNIT_POS < BLOCK_KV && next_q0 + UNIT_POS < next_limit {
            let row = next_q0 + UNIT_POS;
            let at = stats + row as usize * hq as usize + next_h as usize;
            lse_next = lse[at];
            delta_next = delta[at];
            if comptime!(packed) {
                lo_next = bounds[(2 * row) as usize];
                hi_next = bounds[(2 * row + 1) as usize];
            }
        }

        let mut acc_s = zeros(2u32);
        let mut acc_dp = zeros(2u32);
        #[unroll]
        for kp in 0..comptime!(d / 32) {
            let bk = load_b_rows_steps::<E>(&def, s_k.as_slice(), 8 * cg, 2 * kp, chunks);
            let a0 = load_a::<E>(&def, s_q.as_slice(), 16 * rg, 2 * kp, chunks);
            let a1 = load_a::<E>(&def, s_q.as_slice(), 16 * rg, 2 * kp + 1, chunks);
            mma::<E>(&def, &a0, bk[0], bk[1], &mut acc_s, 0u32);
            mma::<E>(&def, &a1, bk[2], bk[3], &mut acc_s, 0u32);
            let bv = load_b_rows_steps::<E>(&def, s_v.as_slice(), 8 * cg, 2 * kp, chunks);
            let e0 = load_a::<E>(&def, s_do.as_slice(), 16 * rg, 2 * kp, chunks);
            let e1 = load_a::<E>(&def, s_do.as_slice(), 16 * rg, 2 * kp + 1, chunks);
            mma::<E>(&def, &e0, bv[0], bv[1], &mut acc_dp, 0u32);
            mma::<E>(&def, &e1, bv[2], bv[3], &mut acc_dp, 0u32);
        }

        // Every element against its row's interval and its key's padding, unless every row
        // of the step sees the whole block: the first row's interval ends at or after the
        // block's end and the last row's starts at or before its start (the intervals never
        // move back), and nothing is padding.
        let masked = padded
            || step_masked(
                q0,
                k0,
                tq,
                tk,
                left,
                right,
                s_lo.as_slice(),
                s_hi.as_slice(),
                packed,
            );
        let mut sa = acc_s[0] * Vector::new(scale);
        let mut sb = acc_s[1] * Vector::new(scale);
        if masked {
            let (lo_a, hi_a) = step_bounds(
                q0,
                qr_a,
                tq,
                tk,
                left,
                right,
                s_lo.as_slice(),
                s_hi.as_slice(),
                packed,
            );
            let (lo_b, hi_b) = step_bounds(
                q0,
                qr_b,
                tq,
                tk,
                left,
                right,
                s_lo.as_slice(),
                s_hi.as_slice(),
                packed,
            );
            #[unroll]
            for e in 0..2u32 {
                let c = col + e as i32;
                let mut padding = pad_0;
                if e == 1 {
                    padding = pad_1;
                }
                let seen_a = c >= lo_a && c < hi_a && !padding;
                let seen_b = c >= lo_b && c < hi_b && !padding;
                sa.insert(
                    e as usize,
                    select(seen_a, sa.extract(e as usize), f32::NEG_INFINITY),
                );
                sb.insert(
                    e as usize,
                    select(seen_b, sb.extract(e as usize), f32::NEG_INFINITY),
                );
            }
        }
        let pa = (sa - Vector::new(s_lse[qr_a as usize])).exp();
        let pb = (sb - Vector::new(s_lse[qr_b as usize])).exp();
        let dsa = pa * (acc_dp[0] - Vector::new(s_delta[qr_a as usize]));
        let dsb = pb * (acc_dp[1] - Vector::new(s_delta[qr_b as usize]));
        s_p[(swizzle(qr_a, cg, 4u32) * 4 + t) as usize] = Vector::cast_from(pa);
        s_p[(swizzle(qr_b, cg, 4u32) * 4 + t) as usize] = Vector::cast_from(pb);
        s_ds[(swizzle(qr_a, cg, 4u32) * 4 + t) as usize] = Vector::cast_from(dsa);
        s_ds[(swizzle(qr_b, cg, 4u32) * 4 + t) as usize] = Vector::cast_from(dsb);
        sync_cube();

        #[unroll]
        for kk in 0..comptime!(BLOCK_KV / 16) {
            let ap = load_a_transposed_narrow::<E>(&def, s_p.as_slice(), 16 * rg, 16 * kk);
            let ads = load_a_transposed_narrow::<E>(&def, s_ds.as_slice(), 16 * rg, 16 * kk);
            #[unroll]
            for np in 0..comptime!(cols / 16) {
                let pair = cg * comptime!(cols / 16) + np;
                let bo = load_b_cols_pair::<E>(&def, s_do.as_slice(), 16 * kk, pair, chunks);
                mma::<E>(&def, &ap, bo[0], bo[1], &mut acc_dv, 2 * np);
                mma::<E>(&def, &ap, bo[2], bo[3], &mut acc_dv, 2 * np + 1);
                let bq = load_b_cols_pair::<E>(&def, s_q.as_slice(), 16 * kk, pair, chunks);
                mma::<E>(&def, &ads, bq[0], bq[1], &mut acc_dk, 2 * np);
                mma::<E>(&def, &ads, bq[2], bq[3], &mut acc_dk, 2 * np + 1);
            }
        }

        // Every warp is done with this step's tiles: the next step's go in.
        sync_cube();
        store_rows::<E>(&q_next, s_q.as_mut_slice(), BLOCK_KV, chunks, 256u32);
        store_rows::<E>(&do_next, s_do.as_mut_slice(), BLOCK_KV, chunks, 256u32);
        if UNIT_POS < BLOCK_KV {
            s_lse[UNIT_POS as usize] = lse_next;
            s_delta[UNIT_POS as usize] = delta_next;
            if comptime!(packed) {
                s_lo[UNIT_POS as usize] = lo_next;
                s_hi[UNIT_POS as usize] = hi_next;
            }
        }
    }

    let key_a = k0 + 16 * rg + g;
    let key_b = key_a + 8;
    // Keys `key_a` and `key_b` of this head as rows of `[B Tk Hkv, D]`.
    let first = b as usize * tk as usize * hkv as usize + hk as usize;
    let row_a = first + key_a as usize * hkv as usize;
    let row_b = first + key_b as usize * hkv as usize;
    let half_d = comptime!(d / 2) as usize;
    #[unroll]
    for nt in 0..comptime!(d / 32) {
        let col = ((cg * cols + nt * 8) / 2 + t) as usize;
        if key_a < tk {
            let at = row_a * half_d + col;
            dk[at] = Vector::cast_from(acc_dk[(2 * nt) as usize] * Vector::new(scale));
            dv[at] = Vector::cast_from(acc_dv[(2 * nt) as usize]);
        }
        if key_b < tk {
            let at = row_b * half_d + col;
            dk[at] = Vector::cast_from(acc_dk[(2 * nt + 1) as usize] * Vector::new(scale));
            dv[at] = Vector::cast_from(acc_dv[(2 * nt + 1) as usize]);
        }
    }
}

/// Decoding, first pass. One cube per (split of the cache, key-value head, batch), four warps;
/// each group of `d / 8` lanes takes one key at a time, each lane eight of its dimensions, for
/// all `group` query heads at once, so a key or value row is read once per group. Sequence `b`
/// holds `lengths[b]` keys (`len`, the cache's length, without `has_lengths`) and its query sees
/// the last `left + 1` of them; split `x` covers the `split_len` keys before the last
/// `x split_len`, within those, so that splits past a sequence's keys or before its window are
/// empty. Writes each split's output, normalised within the split, `[B, Hq, splits, D]`, and its
/// log-sum-exp, `[B, Hq, splits]` (`-inf` for an empty split), both f32 and contiguous.
#[cube(launch_unchecked)]
pub(crate) fn decode_split_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    part_o: &mut Tensor<f32>,
    part_lse: &mut Tensor<f32>,
    lengths: &Tensor<i32>,
    len: u32,
    split_len: u32,
    q_sb: u32,
    q_sh: u32,
    k_sb: u32,
    k_sh: u32,
    k_st: u32,
    v_sb: u32,
    v_sh: u32,
    v_st: u32,
    l_st: u32,
    left: i32,
    scale: f32,
    #[comptime] d: u32,
    #[comptime] group: u32,
    #[comptime] has_lengths: bool,
    #[define(E)] _dtype: ElemType,
) {
    let chunks = comptime!(d / 8);
    let keys_per_warp = comptime!(32 / chunks);
    let slots = comptime!(4 * keys_per_warp);
    let split = CUBE_POS_X;
    let hk = CUBE_POS_Y;
    let b = CUBE_POS_Z;
    let splits = CUBE_COUNT_X;
    let hkv = CUBE_COUNT_Y;
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS_PLANE;
    let c = lane % chunks;
    let slot = warp * keys_per_warp + lane / chunks;
    // The keys this sequence's query sees, `[first, last)`, and this split's share of them,
    // counted back from the last.
    let mut last = len as i32;
    if comptime!(has_lengths) {
        last = max(0i32, min(lengths[b as usize * l_st as usize], len as i32));
    }
    let first = max(0i32, last - 1 - left);
    let end = max(last - (split * split_len) as i32, 0i32);
    let start = max(first, end - split_len as i32);
    let start = start as u32;
    let end = max(end as u32, start);

    let k_base = b as usize * k_sb as usize + hk as usize * k_sh as usize;
    let v_base = b as usize * v_sb as usize + hk as usize * v_sh as usize;

    let mut query = Array::<Vector<f32, V8>>::new(group as usize);
    let mut run_max = Array::<f32>::new(group as usize);
    let mut run_sum = Array::<f32>::new(group as usize);
    let mut acc = Array::<Vector<f32, V8>>::new(group as usize);
    #[unroll]
    for hh in 0..group {
        let h = hk * group + hh;
        let qv = q[b as usize * q_sb as usize + h as usize * q_sh as usize + c as usize];
        query[hh as usize] = Vector::<f32, V8>::cast_from(qv) * Vector::new(scale);
        run_max[hh as usize] = f32::NEG_INFINITY;
        run_sum[hh as usize] = 0.0f32;
        acc[hh as usize] = Vector::new(0.0f32);
    }

    // Every lane runs the same number of steps, so the shuffles below see whole warps.
    let steps = (end - start + slots - 1) / slots;
    for step in 0..steps {
        let key = start + step * slots + slot;
        let valid = key < end;
        let mut kv = Vector::<f32, V8>::new(0.0f32);
        let mut vv = Vector::<f32, V8>::new(0.0f32);
        if valid {
            kv = Vector::cast_from(k[k_base + key as usize * k_st as usize + c as usize]);
            vv = Vector::cast_from(v[v_base + key as usize * v_st as usize + c as usize]);
        }
        #[unroll]
        for hh in 0..group {
            let prod = query[hh as usize] * kv;
            let mut s = 0.0f32;
            #[unroll]
            for e in 0..8u32 {
                s += prod.extract(e as usize);
            }
            #[unroll]
            for i in 0..comptime!(chunks.trailing_zeros()) {
                s += plane_shuffle_xor(s, 1u32 << i);
            }
            s = select(valid, s, f32::NEG_INFINITY);
            let new_max = max(run_max[hh as usize], s);
            let base = select(new_max == f32::NEG_INFINITY, 0.0f32, new_max);
            let alpha = (run_max[hh as usize] - base).exp();
            let p = (s - base).exp();
            run_sum[hh as usize] = run_sum[hh as usize] * alpha + p;
            acc[hh as usize] = acc[hh as usize] * Vector::new(alpha) + vv * Vector::new(p);
            run_max[hh as usize] = new_max;
        }
    }

    // Merge the slots of the cube through shared memory.
    let mut s_max = Shared::<[f32]>::new_slice(comptime!((slots * group) as usize));
    let mut s_sum = Shared::<[f32]>::new_slice(comptime!((slots * group) as usize));
    let mut s_acc =
        Shared::<[Vector<f32, V8>]>::new_slice(comptime!((slots * group * chunks) as usize));
    #[unroll]
    for hh in 0..group {
        let at = slot * group + hh;
        if c == 0 {
            s_max[at as usize] = run_max[hh as usize];
            s_sum[at as usize] = run_sum[hh as usize];
        }
        s_acc[(at * chunks + c) as usize] = acc[hh as usize];
    }
    sync_cube();

    let hq = hkv * group;
    // 128 threads over group x chunks outputs of eight elements.
    let outputs = comptime!(group * chunks);
    let mut index = UNIT_POS;
    while index < outputs {
        let hh = index / chunks;
        let cc = index % chunks;
        let mut m = f32::new(f32::NEG_INFINITY);
        #[unroll]
        for sl in 0..slots {
            m = max(m, s_max[(sl * group + hh) as usize]);
        }
        let base = select(m == f32::NEG_INFINITY, 0.0f32, m);
        let mut total = 0.0f32;
        let mut out = Vector::<f32, V8>::new(0.0f32);
        #[unroll]
        for sl in 0..slots {
            let w = (s_max[(sl * group + hh) as usize] - base).exp();
            total += s_sum[(sl * group + hh) as usize] * w;
            out += s_acc[((sl * group + hh) * chunks + cc) as usize] * Vector::new(w);
        }
        let inv = select(total > 0.0f32, 1.0f32 / total, 0.0f32);
        let h = hk * group + hh;
        let row = ((b * hq + h) * splits + split) as usize;
        let out = out * Vector::new(inv);
        #[unroll]
        for e in 0..8u32 {
            part_o[row * d as usize + (cc * 8 + e) as usize] = out.extract(e as usize);
        }
        if cc == 0 {
            part_lse[row] = base + total.ln();
        }
        index += 128u32;
    }
}
/// Decoding, second pass: one cube per (query head, batch), one thread per output element,
/// merging the splits by their log-sum-exp. Writes the output contiguous `[B, Hq, D]`.
#[cube(launch_unchecked)]
pub(crate) fn decode_merge_kernel<E: Float>(
    part_o: &Tensor<f32>,
    part_lse: &Tensor<f32>,
    out: &mut Tensor<E>,
    splits: u32,
    #[comptime] d: u32,
    #[define(E)] _dtype: ElemType,
) {
    let row = CUBE_POS_X as usize;
    let col = UNIT_POS as usize;
    let first = row * splits as usize;
    let mut m = f32::new(f32::NEG_INFINITY);
    for s in 0..splits {
        m = max(m, part_lse[first + s as usize]);
    }
    let base = select(m == f32::NEG_INFINITY, 0.0f32, m);
    let mut total = 0.0f32;
    let mut acc = 0.0f32;
    for s in 0..splits {
        let w = (part_lse[first + s as usize] - base).exp();
        total += w;
        acc += w * part_o[(first + s as usize) * d as usize + col];
    }
    out[row * d as usize + col] = E::cast_from(select(total > 0.0f32, acc / total, 0.0f32));
}
