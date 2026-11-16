//! The prefill path on RDNA4's 16 x 16 x 16 matrix products (`crate::wmma`): `y = x W (+ bias)`
//! for more rows than the decoding kernels take, from bf16 or f16 `x`, the weight read packed and
//! dequantized in registers, never written out.
//!
//! In RDNA4's layout, lane `l` holds of a B operand (`k` x `n`) column `l % 16` and the eight
//! rows `8 (l / 16)..+8`: eight consecutive inputs of one output, which is what one of GPTQ's
//! packed words holds (`qweight[k / 8, n]`, the eight values `k` low bits first). So a lane reads
//! one word per 16 x 16 tile of `W`, from global memory, and forms its B operand from it:
//! `(q - 8) · scale` in f32, rounded to `x`'s dtype, which the contract allows (`W` rounded to
//! `x`'s dtype). A is `x`, staged in shared memory a cube's rows by 32 inputs at a time.
//!
//! Each cube takes `16 mts` rows (64, or 128 past 64 rows) and 128 outputs, four waves of 32
//! outputs each: per 16 inputs a wave dequantizes two words per lane and multiplies each by `mts`
//! 16-row tiles of `x`, so that the dequantization is amortised over the rows. Each step's share of
//! `x`, words and scales is read into registers while the step before computes, and staged after
//! the barrier that ends it. The products accumulate in f32; the bias is added in f32 and the sum
//! rounded once to `x`'s dtype. Cubes are numbered rows first, so that those running at once read
//! the same columns of the weight. Where a product makes too few cubes to fill the device (a narrow
//! output at few rows), the inputs are split into runs along z: each cube writes its run's f32
//! sums, and the decoding path's `finish_kernel` adds the runs in order, adds the bias and rounds
//! once. How many runs depends on the device's count of compute units, so the output is the same
//! bits from run to run on one device, and may differ in its last bits between devices.

// See `attention/kernels.rs`: CubeCL's frontend takes the plain operators.
#![allow(clippy::manual_div_ceil, clippy::assign_op_pattern)]

use cubecl::cmma::MmaDefinition;
use cubecl::prelude::*;

/// Eight 16-bit elements: a lane's A or B operand, one 16-byte vector.
type V8 = Const<8>;
/// One f32 accumulator element per vector, as the layout gives them.
type R1 = Const<1>;

/// Outputs per cube: four waves of 32.
pub(crate) const BLOCK_N: usize = 128;
/// Inputs staged per step: two products of 16.
pub(crate) const BLOCK_K: usize = 32;
/// Units per cube.
pub(crate) const UNITS: u32 = 128;
/// 16-byte chunks of a staged row of `x`.
const CK: usize = BLOCK_K / 8;

/// Rows of `x` per cube for `mts` 16-row tiles per wave.
pub(crate) fn block_m(mts: usize) -> usize {
    16 * mts
}

/// The index of 16-byte chunk `c` (of four) of row `r` of the staged `x` tile, the chunk XOR-ed
/// with bits 2 and 3 of the row, so that the 16 rows a half-wave reads at one chunk fall in 16
/// different 16-byte slots of a 256-byte window.
#[cube]
fn x_slot(r: usize, c: usize) -> usize {
    r * 4 + (c ^ ((r >> 2) & 3))
}

/// `y = x W (+ bias)`: `x` `[rows, inp]` contiguous in vectors of eight, `qweight` `[inp / 8,
/// out]`, `scales` `[inp / group, out]`, `bias` `[out]` f32 (read when `has_bias`), `y` `[rows,
/// out]`. One cube per `16 mts` rows (x), 128 outputs (y) and run of `run` steps of 32 inputs (z).
/// With `partial`, each run's f32 sums go to `part` `[runs, rows, out]` instead, without the bias,
/// for `finish_kernel` to add in order, add the bias to and round.
#[cube(launch_unchecked, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn matrix_kernel<E: Float, S: Float>(
    x: &Tensor<Vector<E, V8>>,
    qweight: &Tensor<i32>,
    scales: &Tensor<S>,
    bias: &Tensor<f32>,
    y: &mut Tensor<E>,
    part: &mut Tensor<f32>,
    run: usize,
    #[comptime] group: usize,
    #[comptime] has_bias: bool,
    #[comptime] partial: bool,
    #[comptime] mts: usize,
    #[define(E)] _dtype: ElemType,
    #[define(S)] _scales_dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let block_m = comptime!(16 * mts);
    let kks = comptime!(CK / 2);
    let per = comptime!(16 * mts * CK / UNITS as usize);
    let rows = y.shape(0);
    let out = y.shape(1);
    let packed = qweight.shape(0);
    let row_vectors = packed;
    let wave = (UNIT_POS / 32) as usize;
    let lane = UNIT_POS_PLANE as usize;
    let col = lane % 16;
    let half = lane / 16;
    let m0 = CUBE_POS_X as usize * block_m;
    let n0 = CUBE_POS_Y as usize * comptime!(BLOCK_N) + wave * 32;

    let mut xs = Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!(16 * mts * CK), 16usize);
    // Tile (mt, nt): rows m0 + 16 mt + 8 half + i, output n0 + 16 nt + col.
    let mut acc = Array::<Vector<f32, R1>>::new(comptime!(16 * mts));
    #[unroll]
    for i in 0..comptime!(16 * mts) {
        acc[i] = Vector::new(0.0f32);
    }
    // The next step's share of x, and its words and scales, read while the current one computes.
    let mut xr = Array::<Vector<E, V8>>::new(per);
    let mut wr = Array::<u32>::new(comptime!(2 * kks));
    let mut sr = Array::<f32>::new(comptime!(2 * kks));

    let steps = packed / CK;
    let z = CUBE_POS_Z as usize;
    let first = z * run;
    let last = clamp_max(first + run, steps);
    let mut kb = first;
    if kb < last {
        fetch::<E, S>(
            x,
            qweight,
            scales,
            &mut xr,
            &mut wr,
            &mut sr,
            kb,
            m0,
            n0,
            rows,
            out,
            row_vectors,
            group,
            mts,
        );
    }
    while kb < last {
        sync_cube();
        #[unroll]
        for i in 0..per {
            let idx = UNIT_POS as usize + i * UNITS as usize;
            xs[x_slot(idx / CK, idx % CK)] = xr[i];
        }
        // This step's B operands: (q - 8) scale in f32, rounded to x's dtype.
        let mut bw = Array::<Vector<E, V8>>::new(comptime!(2 * kks));
        #[unroll]
        for i in 0..comptime!(2 * kks) {
            let word = wr[i];
            let scale = sr[i];
            let mut w = Vector::<E, V8>::new(E::from_int(0));
            #[unroll]
            for j in 0..8u32 {
                let q = (word >> (4u32 * j)) & 15u32;
                w.insert(
                    j as usize,
                    E::cast_from((f32::cast_from(q) - 8.0f32) * scale),
                );
            }
            bw[i] = w;
        }
        sync_cube();
        if kb + 1 < last {
            fetch::<E, S>(
                x,
                qweight,
                scales,
                &mut xr,
                &mut wr,
                &mut sr,
                kb + 1,
                m0,
                n0,
                rows,
                out,
                row_vectors,
                group,
                mts,
            );
        }
        #[unroll]
        for kk in 0..kks {
            #[unroll]
            for mt in 0..mts {
                let mut a = Array::<Vector<E, V8>>::new(1usize);
                a[0] = xs[x_slot(mt * 16 + col, kk * 2 + half)];
                #[unroll]
                for nt in 0..2usize {
                    let mut b = Array::<Vector<E, V8>>::new(1usize);
                    b[0] = bw[kk * 2 + nt];
                    let mut c = Array::<Vector<f32, R1>>::new(8usize);
                    let tile = (mt * 2 + nt) * 8;
                    #[unroll]
                    for i in 0..8usize {
                        c[i] = acc[tile + i];
                    }
                    let d = def.execute(&a, &b, &c);
                    #[unroll]
                    for i in 0..8usize {
                        acc[tile + i] = d[i];
                    }
                }
            }
        }
        kb += 1;
    }

    #[unroll]
    for nt in 0..2usize {
        let n = n0 + nt * 16 + col;
        if n < out {
            let mut add = 0.0f32;
            if comptime!(has_bias) {
                add = bias[n];
            }
            #[unroll]
            for mt in 0..mts {
                #[unroll]
                for i in 0..8usize {
                    let row = m0 + mt * 16 + half * 8 + i;
                    if row < rows {
                        let v = acc[(mt * 2 + nt) * 8 + i].extract(0usize);
                        if comptime!(partial) {
                            part[(z * rows + row) * out + n] = v;
                        } else {
                            y[row * out + n] = E::cast_from(v + add);
                        }
                    }
                }
            }
        }
    }
}

/// This unit's share of step `kb`: its vectors of the `16 mts` rows of x by 32 inputs, and
/// the lane's words and scales, two outputs for each 16 inputs (zero words past `out`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn fetch<E: Float, S: Float>(
    x: &Tensor<Vector<E, V8>>,
    qweight: &Tensor<i32>,
    scales: &Tensor<S>,
    xr: &mut Array<Vector<E, V8>>,
    wr: &mut Array<u32>,
    sr: &mut Array<f32>,
    kb: usize,
    m0: usize,
    n0: usize,
    rows: usize,
    out: usize,
    row_vectors: usize,
    #[comptime] group: usize,
    #[comptime] mts: usize,
) {
    let half = UNIT_POS_PLANE as usize / 16;
    let col = UNIT_POS_PLANE as usize % 16;
    #[unroll]
    for i in 0..comptime!(16 * mts * CK / UNITS as usize) {
        let idx = UNIT_POS as usize + i * UNITS as usize;
        let row = m0 + idx / CK;
        let mut value = Vector::<E, V8>::new(E::from_int(0));
        if row < rows {
            value = x[row * row_vectors + kb * CK + idx % CK];
        }
        xr[i] = value;
    }
    #[unroll]
    for kk in 0..comptime!(CK / 2) {
        // The word holding inputs 8 p..8 p + 8, all in group 8 p / group.
        let p = kb * CK + kk * 2 + half;
        let g = p * 8 / group;
        #[unroll]
        for nt in 0..2usize {
            let n = n0 + nt * 16 + col;
            let mut word = 0x8888_8888u32;
            let mut scale = 0.0f32;
            if n < out {
                word = u32::cast_from(qweight[p * out + n]);
                scale = f32::cast_from(scales[g * out + n]);
            }
            wr[kk * 2 + nt] = word;
            sr[kk * 2 + nt] = scale;
        }
    }
}
