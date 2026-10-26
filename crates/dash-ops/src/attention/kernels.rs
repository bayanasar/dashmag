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
//! Inputs are read through their batch, head and time strides, counted in 16-byte vectors; a
//! row of `D` elements is contiguous. Outputs are written contiguous in the module's layout:
//! row `i` of head `h` of batch element `b` of a `[B, T, H, D]` tensor is row `(b T + i) H + h`
//! of it seen as `[B T H, D]`, and that is also the element of the `[B, T, H]` statistics (the
//! log-sum-exp, `delta`) that belongs to it.

// Kernel code is expanded by CubeCL's frontend, which takes the plain operators: `div_ceil` and
// the compound assignments on vector elements are written out.
#![allow(clippy::manual_div_ceil, clippy::assign_op_pattern)]

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

/// `scale * s` for the two columns `col`, `col + 1` of an accumulator vector of row `row`,
/// `-inf` where the key is hidden from the query (`col > row + offset`), past the keys
/// (`col >= tk`) or the query past the queries (`row >= tq`); only computed when `masked`.
#[cube]
fn scale_and_mask(
    s: Vector<f32, R2>,
    scale: f32,
    row: u32,
    col: u32,
    offset: u32,
    tq: u32,
    tk: u32,
    masked: bool,
) -> Vector<f32, R2> {
    let mut out = s * Vector::new(scale);
    if masked {
        #[unroll]
        for e in 0..2u32 {
            let c = col + e;
            let visible = c <= row + offset && c < tk && row < tq;
            out.insert(
                e as usize,
                select(visible, out.extract(e as usize), f32::NEG_INFINITY),
            );
        }
    }
    out
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

/// The training forward. One cube per (`block_q`-query block, query head, batch), a warp per 16
/// query rows; query blocks are taken last first, so that the longest causal rows start first.
/// Writes the output, contiguous `[B, Tq, Hq, D]`, and the natural log-sum-exp of each query's
/// scaled scores, contiguous `[B, Tq, Hq]`. Input strides are in 8-element vectors.
#[cube(launch_unchecked)]
pub(crate) fn forward_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    o: &mut Tensor<Vector<E, R2>>,
    lse: &mut Tensor<f32>,
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
    scale: f32,
    #[comptime] d: u32,
    #[comptime] block_q: u32,
    #[comptime] block_kv: u32,
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
    let offset = tk - tq;

    let q_base = b as usize * q_sb as usize + h as usize * q_sh as usize;
    let k_base = b as usize * k_sb as usize + hk as usize * k_sh as usize;
    let v_base = b as usize * v_sb as usize + hk as usize * v_sh as usize;

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
        0u32,
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
        0u32,
        tk,
        block_kv,
        chunks,
        threads,
    );

    let row_a = q0 + warp * 16 + g;
    let row_b = row_a + 8;
    let mut max_a = f32::new(f32::NEG_INFINITY);
    let mut max_b = f32::new(f32::NEG_INFINITY);
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut acc_o = zeros(comptime!(d / 4));

    let kv_end = min(q0 + block_q + offset, tk);
    let n_blocks = (kv_end + block_kv - 1) / block_kv;
    for j in 0..n_blocks {
        let k0 = j * block_kv;
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

        let mut acc_s = scores::<E>(
            &def,
            s_q.as_slice(),
            s_k.as_slice(),
            warp * 16,
            0u32,
            d,
            block_kv,
        );

        // The block reaches past the first query's last key, or past the keys: mask it.
        let masked = k0 + block_kv > q0 + offset + 1 || k0 + block_kv > tk;
        let mut tile_max_a = max_a;
        let mut tile_max_b = max_b;
        #[unroll]
        for nt in 0..comptime!(block_kv / 8) {
            let col = k0 + nt * 8 + 2 * t;
            let sa = scale_and_mask(
                acc_s[(2 * nt) as usize],
                scale,
                row_a,
                col,
                offset,
                tq,
                tk,
                masked,
            );
            let sb = scale_and_mask(
                acc_s[(2 * nt + 1) as usize],
                scale,
                row_b,
                col,
                offset,
                tq,
                tk,
                masked,
            );
            tile_max_a = max(tile_max_a, max(sa.extract(0usize), sa.extract(1usize)));
            tile_max_b = max(tile_max_b, max(sb.extract(0usize), sb.extract(1usize)));
            acc_s[(2 * nt) as usize] = sa;
            acc_s[(2 * nt + 1) as usize] = sb;
        }
        tile_max_a = quad_max(tile_max_a);
        tile_max_b = quad_max(tile_max_b);
        // A row with every key hidden so far keeps -inf; exponentiate against 0 then, so that
        // its probabilities and rescaling factor are 0 rather than NaN.
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
        sync_cube();
        store_rows::<E>(&k_next, s_k.as_mut_slice(), block_kv, chunks, threads);
        store_rows::<E>(&v_next, s_v.as_mut_slice(), block_kv, chunks, threads);
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
/// per row, eight elements each, 256 threads per cube, rows taken in that order.
#[cube(launch_unchecked)]
pub(crate) fn delta_kernel<E: Float>(
    o: &Tensor<Vector<E, V8>>,
    d_o: &Tensor<Vector<E, V8>>,
    delta: &mut Tensor<f32>,
    rows: u32,
    hq: u32,
    tq: u32,
    o_sb: u32,
    o_sh: u32,
    o_st: u32,
    do_sb: u32,
    do_sh: u32,
    do_st: u32,
    #[comptime] d: u32,
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
    }
}

/// `dQ`. One cube per (64-query block, query head, batch), eight warps: two per 16 query rows,
/// each taking half of every key block, their partial `dQ` added at the end. Sweeps the key
/// blocks up to the diagonal: `S` and `dP = dO V^T` recomputed per block, `P` from the
/// forward's log-sum-exp, `dS = P (dP - delta)`, `dQ += dS K`. Writes `dQ` contiguous
/// `[B, Tq, Hq, D]`.
#[cube(launch_unchecked)]
pub(crate) fn dq_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    d_o: &Tensor<Vector<E, V8>>,
    lse: &Tensor<f32>,
    delta: &Tensor<f32>,
    dq: &mut Tensor<Vector<E, R2>>,
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
    #[comptime] d: u32,
    #[comptime] block_kv: u32,
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
    let offset = tk - tq;

    let q_base = b as usize * q_sb as usize + h as usize * q_sh as usize;
    let do_base = b as usize * do_sb as usize + h as usize * do_sh as usize;
    let k_base = b as usize * k_sb as usize + hk as usize * k_sh as usize;
    let v_base = b as usize * v_sb as usize + hk as usize * v_sh as usize;

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
        0u32,
        tk,
        block_kv,
        chunks,
        256u32,
    );
    load_rows::<E>(
        v,
        s_v.as_mut_slice(),
        v_base,
        v_st as usize,
        0u32,
        tk,
        block_kv,
        chunks,
        256u32,
    );

    let row_a = q0 + rows * 16 + g;
    let row_b = row_a + 8;
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

    let kv_end = min(q0 + BLOCK_Q + offset, tk);
    let n_blocks = (kv_end + block_kv - 1) / block_kv;
    for j in 0..n_blocks {
        let k0 = j * block_kv;
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

        let masked = k0 + block_kv > q0 + offset + 1 || k0 + block_kv > tk;
        #[unroll]
        for nt in 0..comptime!(half / 8) {
            let col = k0 + key0 + nt * 8 + 2 * t;
            let sa = scale_and_mask(
                acc_s[(2 * nt) as usize],
                scale,
                row_a,
                col,
                offset,
                tq,
                tk,
                masked,
            );
            let sb = scale_and_mask(
                acc_s[(2 * nt + 1) as usize],
                scale,
                row_b,
                col,
                offset,
                tq,
                tk,
                masked,
            );
            let pa = (sa - Vector::new(lse_a)).exp();
            let pb = (sb - Vector::new(lse_b)).exp();
            acc_s[(2 * nt) as usize] = pa * (acc_dp[(2 * nt) as usize] - Vector::new(delta_a));
            acc_s[(2 * nt + 1) as usize] =
                pb * (acc_dp[(2 * nt + 1) as usize] - Vector::new(delta_b));
        }
        accumulate_pv::<E>(&def, &acc_s, s_k.as_slice(), &mut acc_dq, key0, d, half);
        sync_cube();
        store_rows::<E>(&k_next, s_k.as_mut_slice(), block_kv, chunks, 256u32);
        store_rows::<E>(&v_next, s_v.as_mut_slice(), block_kv, chunks, 256u32);
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
            let mut packed = Vector::<f32, Const<4>>::empty();
            packed.insert(0usize, acc_dq[(2 * j) as usize].extract(0usize));
            packed.insert(1usize, acc_dq[(2 * j) as usize].extract(1usize));
            packed.insert(2usize, acc_dq[(2 * j + 1) as usize].extract(0usize));
            packed.insert(3usize, acc_dq[(2 * j + 1) as usize].extract(1usize));
            let at = (region + j * 32 + lane) as usize;
            if rows < 2 {
                s_q[at] = Vector::<E, V8>::reinterpret(packed);
            } else {
                s_do[at] = Vector::<E, V8>::reinterpret(packed);
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

/// `dK` and `dV`. One cube per (32-key block, key-value head, batch), eight warps, sweeping the
/// query heads of the group and, for each, the 32-query steps from the diagonal down. `S` and
/// `dP` are split 2 x 4 over the warps (16 queries x 8 keys each); `P` and `dS` go through
/// shared memory to the 2 x 4 split of `dV += P^T dO` and `dK += dS^T Q` (16 keys x D/4
/// columns each). Writes `dK` and `dV` contiguous `[B, Tk, Hkv, D]`.
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
    #[comptime] d: u32,
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
    let offset = tk - tq;

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

    // The first query that sees key k0 is k0 - offset.
    let first_query = select(k0 > offset, k0 - offset, 0u32);
    let first_step = first_query / BLOCK_KV;
    let n_steps = (tq + BLOCK_KV - 1) / BLOCK_KV;
    let qr_a = 16 * rg + g;
    let qr_b = qr_a + 8;
    let key_col = 8 * cg + 2 * t;

    // The steps of all the group's query heads, one after the other: step `it` is query block
    // `first_step + it % per_head` of head `hk * group + it / per_head`. Each step reads the
    // next step's queries, output gradients and row statistics into registers while it computes.
    let per_head = n_steps - first_step;
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
        if UNIT_POS < BLOCK_KV && next_q0 + UNIT_POS < next_limit {
            let at = stats + (next_q0 + UNIT_POS) as usize * hq as usize + next_h as usize;
            lse_next = lse[at];
            delta_next = delta[at];
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

        let masked = k0 + BLOCK_KV > q0 + offset + 1 || q0 + BLOCK_KV > tq || k0 + BLOCK_KV > tk;
        let sa = scale_and_mask(
            acc_s[0],
            scale,
            q0 + qr_a,
            k0 + key_col,
            offset,
            tq,
            tk,
            masked,
        );
        let sb = scale_and_mask(
            acc_s[1],
            scale,
            q0 + qr_b,
            k0 + key_col,
            offset,
            tq,
            tk,
            masked,
        );
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
/// all `group` query heads at once, so a key or value row is read once per group. Writes each
/// split's output, normalised within the split, `[B, Hq, splits, D]`, and its log-sum-exp,
/// `[B, Hq, splits]`, both f32 and contiguous.
#[cube(launch_unchecked)]
pub(crate) fn decode_split_kernel<E: Float>(
    q: &Tensor<Vector<E, V8>>,
    k: &Tensor<Vector<E, V8>>,
    v: &Tensor<Vector<E, V8>>,
    part_o: &mut Tensor<f32>,
    part_lse: &mut Tensor<f32>,
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
    scale: f32,
    #[comptime] d: u32,
    #[comptime] group: u32,
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
    let start = split * split_len;
    let end = min(start + split_len, len);

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
