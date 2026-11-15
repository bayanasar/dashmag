//! The prefill path on RDNA4's 16 x 16 x 16 matrix products (`crate::wmma`): `y = x W (+ bias)`
//! for more rows than the decoding kernels take, from bf16 or f16 `x`, the weight read packed and
//! dequantized in registers, never written out.
//!
//! In RDNA4's layout, lane `l` holds of a B operand (`k` x `n`) column `l % 16` and the eight
//! rows `8 (l / 16)..+8`: eight consecutive inputs of one output, which is what one of GPTQ's
//! packed words holds (`qweight[k / 8, n]`, the eight values `k` low bits first). So a lane reads
//! one word per 16 x 16 tile of `W`, from global memory, and forms its B operand from it:
//! `(q - 8) · scale` in f32, rounded to `x`'s dtype, which the contract allows (`W` rounded to
//! `x`'s dtype). A is `x`, staged in shared memory 64 rows by 32 inputs at a time.
//!
//! Each cube takes 64 rows and 128 outputs, four waves of 32 outputs each: per 16 inputs a wave
//! dequantizes two words per lane and multiplies each by four 16-row tiles of `x`, so that the
//! dequantization is amortised over the rows. The products accumulate in f32; the bias is added
//! in f32 and the sum rounded once to `x`'s dtype. Cubes are numbered rows first, so that those
//! running at once read the same columns of the weight.

// See `attention/kernels.rs`: CubeCL's frontend takes the plain operators.
#![allow(clippy::manual_div_ceil, clippy::assign_op_pattern)]

use cubecl::cmma::MmaDefinition;
use cubecl::prelude::*;

/// Eight 16-bit elements: a lane's A or B operand, one 16-byte vector.
type V8 = Const<8>;
/// One f32 accumulator element per vector, as the layout gives them.
type R1 = Const<1>;

/// Rows of `x` per cube.
pub(crate) const BLOCK_M: usize = 64;
/// Outputs per cube: four waves of 32.
pub(crate) const BLOCK_N: usize = 128;
/// Inputs staged per step: two products of 16.
pub(crate) const BLOCK_K: usize = 32;
/// Units per cube.
pub(crate) const UNITS: u32 = 128;

/// The index of 16-byte chunk `c` (of four) of row `r` of the staged `x` tile, the chunk XOR-ed
/// with bits 2 and 3 of the row, so that the 16 rows a half-wave reads at one chunk fall in 16
/// different 16-byte slots of a 256-byte window.
#[cube]
fn x_slot(r: usize, c: usize) -> usize {
    r * 4 + (c ^ ((r >> 2) & 3))
}

/// `y = x W (+ bias)`: `x` `[rows, inp]` contiguous in vectors of eight, `qweight`
/// `[inp / 8, out]`, `scales` `[inp / group, out]`, `bias` `[out]` f32 (read when `has_bias`), `y`
/// `[rows, out]`. One cube per 64 rows (x) and 128 outputs (y).
#[cube(launch_unchecked, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn matrix_kernel<E: Float, S: Float>(
    x: &Tensor<Vector<E, V8>>,
    qweight: &Tensor<i32>,
    scales: &Tensor<S>,
    bias: &Tensor<f32>,
    y: &mut Tensor<E>,
    #[comptime] group: usize,
    #[comptime] has_bias: bool,
    #[define(E)] _dtype: ElemType,
    #[define(S)] _scales_dtype: ElemType,
) {
    let def = MmaDefinition::<E, E, f32>::new(16usize, 16usize, 16usize);
    let rows = y.shape(0);
    let out = y.shape(1);
    let packed = qweight.shape(0);
    let row_vectors = packed;
    let wave = (UNIT_POS / 32) as usize;
    let lane = UNIT_POS_PLANE as usize;
    let col = lane % 16;
    let half = lane / 16;
    let m0 = CUBE_POS_X as usize * comptime!(BLOCK_M);
    let n0 = CUBE_POS_Y as usize * comptime!(BLOCK_N) + wave * 32;

    let mut xs = Shared::<[Vector<E, V8>]>::new_aligned_slice(comptime!(BLOCK_M * 4), 16usize);
    // Tile (mt, nt): rows m0 + 16 mt + 8 half + i, output n0 + 16 nt + col.
    let mut acc = Array::<Vector<f32, R1>>::new(64usize);
    #[unroll]
    for i in 0..64usize {
        acc[i] = Vector::new(0.0f32);
    }

    let steps = packed / 4;
    for kb in 0..steps {
        sync_cube();
        #[unroll]
        for i in 0..2usize {
            let idx = UNIT_POS as usize + i * comptime!(UNITS as usize);
            let r = idx / 4;
            let c = idx % 4;
            let row = m0 + r;
            let mut value = Vector::<E, V8>::new(E::from_int(0));
            if row < rows {
                value = x[row * row_vectors + kb * 4 + c];
            }
            xs[x_slot(r, c)] = value;
        }
        sync_cube();
        #[unroll]
        for kk in 0..2usize {
            // The word holding inputs 8 p..8 p + 8, all in group 8 p / group.
            let p = kb * 4 + kk * 2 + half;
            let g = p * 8 / comptime!(group);
            #[unroll]
            for nt in 0..2usize {
                let n = n0 + nt * 16 + col;
                let mut word = 0x8888_8888u32;
                let mut scale = 0.0f32;
                if n < out {
                    word = u32::cast_from(qweight[p * out + n]);
                    scale = f32::cast_from(scales[g * out + n]);
                }
                let mut w = Vector::<E, V8>::new(E::from_int(0));
                #[unroll]
                for j in 0..8u32 {
                    let q = (word >> (4u32 * j)) & 15u32;
                    w.insert(
                        j as usize,
                        E::cast_from((f32::cast_from(q) - 8.0f32) * scale),
                    );
                }
                let mut b = Array::<Vector<E, V8>>::new(1usize);
                b[0] = w;
                #[unroll]
                for mt in 0..4usize {
                    let mut a = Array::<Vector<E, V8>>::new(1usize);
                    a[0] = xs[x_slot(mt * 16 + col, kk * 2 + half)];
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
            for mt in 0..4usize {
                #[unroll]
                for i in 0..8usize {
                    let row = m0 + mt * 16 + half * 8 + i;
                    if row < rows {
                        let v = acc[(mt * 2 + nt) * 8 + i].extract(0usize) + add;
                        y[row * out + n] = E::cast_from(v);
                    }
                }
            }
        }
    }
}
