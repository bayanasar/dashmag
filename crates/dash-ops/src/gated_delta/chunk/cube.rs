//! The chunked gated delta rule on the CubeCL backend, in CubeCL kernels, all in f32 (no tensor
//! cores), for heads up to [`MAX_HEAD_DIM`] wide, on devices with what [`requirements`] lists.
//! Notation as in `reference.rs`.
//!
//! Forward: [`prep_kernel`] per chunk (`G`, `A`, `T = (I + A)^-1`, `W`, `U`, `kt` from the raw
//! inputs, normalising keys when asked); [`state_forward_kernel`] passes the state from chunk to
//! chunk; [`output_kernel`] per chunk. Backward: the first two again, then [`dvlocal_kernel`] per
//! chunk (`P^T dO`, `gamma Q`), [`state_backward_kernel`] (the state's gradient, chunk to chunk),
//! and per chunk [`dvb_kernel`] (`dV_b = T^T dV'` and the values' gradient), [`dstate_kernel`] (the
//! products with the state and its gradient), [`dattn_kernel`] (through `P`), [`dwy_kernel`]
//! (through `A`), [`combine_kernel`] (the sums, the gate's reverse cumulative sum within each
//! chunk, the l2norm's backward).
//!
//! Two identities keep the backward short: `T^T dW = -(T^T dV') S^T`, so `dKbg` comes from `dV_b`
//! and the state like the other products with it; and `T^T dT T^T = dV_b U^T + dKbg W^T`, so
//! `dA` needs no product with `T`.
//!
//! The per-chunk kernels run one cube of 256 threads per chunk of each batch element and head;
//! thread `(ty, tx)` forms rows `ty + 16 a` and columns `tx + 16 b` of each product from operands
//! staged in shared memory, rows padded against bank conflicts. The state passes run one cube per
//! batch element, head and block of 32 state columns.

use super::reference::{self, CHUNK};
use super::{ChunkGatedDeltaBackend, probe};
use crate::cube::Requirements;
use burn::backend::TensorMetadata;
use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::FloatTensor;
use burn::tensor::{DType, Shape, Slice};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::client::Client;
use cubecl::prelude::*;
use std::sync::atomic::Ordering;

/// Columns of the state (of `Dv`) one cube carries through the state passing.
const BV: usize = 32;
/// Threads per cube, in every kernel here.
const THREADS: u32 = 256;
/// The largest head dimensions the kernels hold in shared memory; larger heads run the
/// reference.
pub(super) const MAX_HEAD_DIM: usize = 128;

type V4 = Vector<f32, Const<4>>;
type V2 = Vector<f32, Const<2>>;

/// The state passing, forward. One cube per (batch and head, block of 32 state columns), 256
/// threads; thread `(ty, tx)` owns the block's columns `2 tx, 2 tx + 1`, rows `4 ty .. 4 ty + 4` of
/// `V'` and rows `8 ty .. 8 ty + 8` of the state (in registers), so that each step of the inner
/// products reads its operands with one or two vector loads from shared memory. For each chunk
/// in turn: store the entering state; stage `W` (transposed) and `kt`, loaded into registers
/// during the previous chunk; form `V' = U - W S`; then `S = exp(G_last) S + kt^T V'`.
/// `Dk <= 128`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn state_forward_kernel(
    kt: &Tensor<f32>,
    w: &Tensor<f32>,
    u: &Tensor<f32>,
    gl: &Tensor<f32>,
    h0: &Tensor<f32>,
    states: &mut Tensor<f32>,
    v_new: &mut Tensor<f32>,
    last: &mut Tensor<f32>,
    n_chunks: u32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
) {
    let bh = CUBE_POS_Y as usize;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let col0 = CUBE_POS_X as usize * 32usize + 2usize * tx;
    let n = n_chunks as usize;
    // W^T by vectors of 4 rows: wt[k * 17 + i / 4] (rows padded to 17 vectors).
    let mut wt: Shared<[V4]> = Shared::new_slice(128usize * 17usize);
    // kt by vectors of 4 columns: ks[i * 32 + k / 4].
    let mut ks: Shared<[V4]> = Shared::new_slice(64usize * 32usize);
    // The state block and V' by pairs of columns: ss[k * 16 + tx], vs[i * 16 + tx].
    let mut ss: Shared<[V2]> = Shared::new_slice(128usize * 16usize);
    let mut vs: Shared<[V2]> = Shared::new_slice(64usize * 16usize);
    let mut sreg = Array::<f32>::new(16usize);
    // Prefetched W (8 vectors of 4 rows) and kt (8 vectors of 4 columns) of the next chunk.
    let mut pre_w = Array::<f32>::new(32usize);
    let mut pre_k = Array::<f32>::new(32usize);
    let mut pre_u = Array::<f32>::new(8usize);

    #[unroll]
    for q in 0..8usize {
        let k = 8usize * ty + q;
        let mut pair = V2::empty();
        #[unroll]
        for cc in 0..2usize {
            let col = col0 + cc;
            let mut x = f32::new(0.0f32);
            if col < dv && k < dk {
                x = h0[(bh * dk + k) * dv + col];
            }
            sreg[q * 2usize + cc] = x;
            pair.insert(cc, x);
        }
        ss[k * 16usize + tx] = pair;
    }

    // The first chunk's W and kt into the prefetch registers: W element (4 i4 + r, k) and kt
    // element (i, 4 k4 + r), for this thread's 8 (k, i4) and (i, k4) pairs.
    let first = bh * n * 64usize * dk;
    #[unroll]
    for m in 0..8usize {
        let e = tid + 256usize * m;
        let k = e % 128usize;
        let i4 = e / 128usize;
        let i = e / 32usize;
        let k4 = e % 32usize;
        #[unroll]
        for r in 0..4usize {
            let mut wv = f32::new(0.0f32);
            let mut kv = f32::new(0.0f32);
            if k < dk {
                wv = w[first + (4usize * i4 + r) * dk + k];
            }
            if 4usize * k4 + r < dk {
                kv = kt[first + i * dk + 4usize * k4 + r];
            }
            pre_w[m * 4usize + r] = wv;
            pre_k[m * 4usize + r] = kv;
        }
    }
    #[unroll]
    for p in 0..4usize {
        #[unroll]
        for cc in 0..2usize {
            let col = col0 + cc;
            let mut x = f32::new(0.0f32);
            if col < dv {
                x = u[(bh * n * 64usize + 4usize * ty + p) * dv + col];
            }
            pre_u[p * 2usize + cc] = x;
        }
    }
    let mut pre_gl = gl[bh * n];

    for c in 0..n {
        let chunk = bh * n + c;
        #[unroll]
        for m in 0..8usize {
            let e = tid + 256usize * m;
            let mut wv = V4::empty();
            let mut kv = V4::empty();
            #[unroll]
            for r in 0..4usize {
                wv.insert(r, pre_w[m * 4usize + r]);
                kv.insert(r, pre_k[m * 4usize + r]);
            }
            wt[(e % 128usize) * 17usize + e / 128usize] = wv;
            ks[(e / 32usize) * 32usize + e % 32usize] = kv;
        }
        #[unroll]
        for q in 0..8usize {
            let k = 8usize * ty + q;
            #[unroll]
            for cc in 0..2usize {
                let col = col0 + cc;
                if col < dv && k < dk {
                    states[(chunk * dk + k) * dv + col] = sreg[q * 2usize + cc];
                }
            }
        }
        sync_cube();
        let mut acc = Array::<f32>::new(8usize);
        #[unroll]
        for x in 0..8usize {
            acc[x] = pre_u[x];
        }
        let decay = pre_gl;
        if c + 1usize < n {
            let next = (chunk + 1usize) * 64usize * dk;
            #[unroll]
            for m in 0..8usize {
                let e = tid + 256usize * m;
                let k = e % 128usize;
                let i4 = e / 128usize;
                let i = e / 32usize;
                let k4 = e % 32usize;
                #[unroll]
                for r in 0..4usize {
                    let mut wv = f32::new(0.0f32);
                    let mut kv = f32::new(0.0f32);
                    if k < dk {
                        wv = w[next + (4usize * i4 + r) * dk + k];
                    }
                    if 4usize * k4 + r < dk {
                        kv = kt[next + i * dk + 4usize * k4 + r];
                    }
                    pre_w[m * 4usize + r] = wv;
                    pre_k[m * 4usize + r] = kv;
                }
            }
            #[unroll]
            for p in 0..4usize {
                #[unroll]
                for cc in 0..2usize {
                    let col = col0 + cc;
                    let mut x = f32::new(0.0f32);
                    if col < dv {
                        x = u[((chunk + 1usize) * 64usize + 4usize * ty + p) * dv + col];
                    }
                    pre_u[p * 2usize + cc] = x;
                }
            }
            pre_gl = gl[chunk + 1usize];
        }

        // V' = U - W S.
        #[unroll]
        for k in 0..128usize {
            let wv = wt[k * 17usize + ty];
            let sv = ss[k * 16usize + tx];
            #[unroll]
            for p in 0..4usize {
                #[unroll]
                for cc in 0..2usize {
                    acc[p * 2usize + cc] -= wv.extract(p) * sv.extract(cc);
                }
            }
        }
        #[unroll]
        for p in 0..4usize {
            let i = 4usize * ty + p;
            let mut pair = V2::empty();
            #[unroll]
            for cc in 0..2usize {
                pair.insert(cc, acc[p * 2usize + cc]);
                let col = col0 + cc;
                if col < dv {
                    v_new[(chunk * 64usize + i) * dv + col] = acc[p * 2usize + cc];
                }
            }
            vs[i * 16usize + tx] = pair;
        }
        sync_cube();

        // S = exp(G_last) S + kt^T V'.
        let mut acc2 = Array::<f32>::new(16usize);
        #[unroll]
        for x in 0..16usize {
            acc2[x] = decay * sreg[x];
        }
        #[unroll]
        for i in 0..64usize {
            let vv = vs[i * 16usize + tx];
            let k0 = ks[i * 32usize + 2usize * ty];
            let k1 = ks[i * 32usize + 2usize * ty + 1usize];
            #[unroll]
            for r in 0..4usize {
                #[unroll]
                for cc in 0..2usize {
                    acc2[r * 2usize + cc] += k0.extract(r) * vv.extract(cc);
                    acc2[(4usize + r) * 2usize + cc] += k1.extract(r) * vv.extract(cc);
                }
            }
        }
        #[unroll]
        for q in 0..8usize {
            let mut pair = V2::empty();
            #[unroll]
            for cc in 0..2usize {
                sreg[q * 2usize + cc] = acc2[q * 2usize + cc];
                pair.insert(cc, acc2[q * 2usize + cc]);
            }
            ss[(8usize * ty + q) * 16usize + tx] = pair;
        }
        sync_cube();
    }

    #[unroll]
    for q in 0..8usize {
        let k = 8usize * ty + q;
        #[unroll]
        for cc in 0..2usize {
            let col = col0 + cc;
            if col < dv && k < dk {
                last[(bh * dk + k) * dv + col] = sreg[q * 2usize + cc];
            }
        }
    }
}

/// The state passing, backward, laid out as [`state_forward_kernel`]. For each chunk from the
/// last: store `dS_{c+1}`; form `dV' = kt dS + dv_local` (`kt` staged transposed); then
/// `dS = exp(G_last) dS + qg^T dO - W^T dV'` (`qg` staged where `kt` was). The next chunk's `kt`,
/// `W`, `qg`, `dv_local`, `dO` are loaded into registers during the current one. `Dk <= 128`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn state_backward_kernel(
    kt: &Tensor<f32>,
    w: &Tensor<f32>,
    qg: &Tensor<f32>,
    d_o: &Tensor<f32>,
    dv_local: &Tensor<f32>,
    gl: &Tensor<f32>,
    d_last: &Tensor<f32>,
    ds_next: &mut Tensor<f32>,
    dvn: &mut Tensor<f32>,
    d_first: &mut Tensor<f32>,
    n_chunks: u32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
) {
    let bh = CUBE_POS_Y as usize;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let col0 = CUBE_POS_X as usize * 32usize + 2usize * tx;
    let n = n_chunks as usize;
    // kt^T by vectors of 4 rows (xs[k * 17 + i / 4]) for dV', then qg by vectors of 4 columns
    // (xs[i * 32 + k / 4]) for dS.
    let mut xs: Shared<[V4]> = Shared::new_slice(128usize * 17usize);
    // W by vectors of 4 columns: ys[i * 32 + k / 4].
    let mut ys: Shared<[V4]> = Shared::new_slice(64usize * 32usize);
    let mut dss: Shared<[V2]> = Shared::new_slice(128usize * 16usize);
    let mut dvs: Shared<[V2]> = Shared::new_slice(64usize * 16usize);
    let mut dos: Shared<[V2]> = Shared::new_slice(64usize * 16usize);
    let mut dsreg = Array::<f32>::new(16usize);
    let mut pre_k = Array::<f32>::new(32usize);
    let mut pre_w = Array::<f32>::new(32usize);
    let mut pre_q = Array::<f32>::new(32usize);
    let mut pre_l = Array::<f32>::new(8usize);
    let mut pre_o = Array::<f32>::new(8usize);

    #[unroll]
    for q in 0..8usize {
        let k = 8usize * ty + q;
        let mut pair = V2::empty();
        #[unroll]
        for cc in 0..2usize {
            let col = col0 + cc;
            let mut x = f32::new(0.0f32);
            if col < dv && k < dk {
                x = d_last[(bh * dk + k) * dv + col];
            }
            dsreg[q * 2usize + cc] = x;
            pair.insert(cc, x);
        }
        dss[k * 16usize + tx] = pair;
    }

    let last_chunk = bh * n + n - 1usize;
    let first = last_chunk * 64usize * dk;
    #[unroll]
    for m in 0..8usize {
        let e = tid + 256usize * m;
        let k = e % 128usize;
        let i4 = e / 128usize;
        let i = e / 32usize;
        let k4 = e % 32usize;
        #[unroll]
        for r in 0..4usize {
            let mut kv = f32::new(0.0f32);
            let mut wv = f32::new(0.0f32);
            let mut qv = f32::new(0.0f32);
            if k < dk {
                kv = kt[first + (4usize * i4 + r) * dk + k];
            }
            if 4usize * k4 + r < dk {
                wv = w[first + i * dk + 4usize * k4 + r];
                qv = qg[first + i * dk + 4usize * k4 + r];
            }
            pre_k[m * 4usize + r] = kv;
            pre_w[m * 4usize + r] = wv;
            pre_q[m * 4usize + r] = qv;
        }
    }
    #[unroll]
    for p in 0..4usize {
        #[unroll]
        for cc in 0..2usize {
            let col = col0 + cc;
            let mut x = f32::new(0.0f32);
            let mut y = f32::new(0.0f32);
            if col < dv {
                let idx = (last_chunk * 64usize + 4usize * ty + p) * dv + col;
                x = dv_local[idx];
                y = d_o[idx];
            }
            pre_l[p * 2usize + cc] = x;
            pre_o[p * 2usize + cc] = y;
        }
    }
    let mut pre_gl = gl[last_chunk];

    for cr in 0..n {
        let chunk = bh * n + (n - 1usize - cr);
        #[unroll]
        for m in 0..8usize {
            let e = tid + 256usize * m;
            let mut kv = V4::empty();
            let mut wv = V4::empty();
            #[unroll]
            for r in 0..4usize {
                kv.insert(r, pre_k[m * 4usize + r]);
                wv.insert(r, pre_w[m * 4usize + r]);
            }
            xs[(e % 128usize) * 17usize + e / 128usize] = kv;
            ys[(e / 32usize) * 32usize + e % 32usize] = wv;
        }
        #[unroll]
        for q in 0..8usize {
            let k = 8usize * ty + q;
            #[unroll]
            for cc in 0..2usize {
                let col = col0 + cc;
                if col < dv && k < dk {
                    ds_next[(chunk * dk + k) * dv + col] = dsreg[q * 2usize + cc];
                }
            }
        }
        let mut acc = Array::<f32>::new(8usize);
        #[unroll]
        for p in 0..4usize {
            let i = 4usize * ty + p;
            let mut pair = V2::empty();
            #[unroll]
            for cc in 0..2usize {
                acc[p * 2usize + cc] = pre_l[p * 2usize + cc];
                pair.insert(cc, pre_o[p * 2usize + cc]);
            }
            dos[i * 16usize + tx] = pair;
        }
        let decay = pre_gl;
        sync_cube();

        // dV' = kt dS + dv_local.
        #[unroll]
        for k in 0..128usize {
            let kv = xs[k * 17usize + ty];
            let sv = dss[k * 16usize + tx];
            #[unroll]
            for p in 0..4usize {
                #[unroll]
                for cc in 0..2usize {
                    acc[p * 2usize + cc] += kv.extract(p) * sv.extract(cc);
                }
            }
        }
        #[unroll]
        for p in 0..4usize {
            let i = 4usize * ty + p;
            let mut pair = V2::empty();
            #[unroll]
            for cc in 0..2usize {
                pair.insert(cc, acc[p * 2usize + cc]);
                let col = col0 + cc;
                if col < dv {
                    dvn[(chunk * 64usize + i) * dv + col] = acc[p * 2usize + cc];
                }
            }
            dvs[i * 16usize + tx] = pair;
        }
        sync_cube();
        #[unroll]
        for m in 0..8usize {
            let e = tid + 256usize * m;
            let mut qv = V4::empty();
            #[unroll]
            for r in 0..4usize {
                qv.insert(r, pre_q[m * 4usize + r]);
            }
            xs[(e / 32usize) * 32usize + e % 32usize] = qv;
        }
        sync_cube();
        if cr + 1usize < n {
            let next = (chunk - 1usize) * 64usize * dk;
            #[unroll]
            for m in 0..8usize {
                let e = tid + 256usize * m;
                let k = e % 128usize;
                let i4 = e / 128usize;
                let i = e / 32usize;
                let k4 = e % 32usize;
                #[unroll]
                for r in 0..4usize {
                    let mut kv = f32::new(0.0f32);
                    let mut wv = f32::new(0.0f32);
                    let mut qv = f32::new(0.0f32);
                    if k < dk {
                        kv = kt[next + (4usize * i4 + r) * dk + k];
                    }
                    if 4usize * k4 + r < dk {
                        wv = w[next + i * dk + 4usize * k4 + r];
                        qv = qg[next + i * dk + 4usize * k4 + r];
                    }
                    pre_k[m * 4usize + r] = kv;
                    pre_w[m * 4usize + r] = wv;
                    pre_q[m * 4usize + r] = qv;
                }
            }
            #[unroll]
            for p in 0..4usize {
                #[unroll]
                for cc in 0..2usize {
                    let col = col0 + cc;
                    let mut x = f32::new(0.0f32);
                    let mut y = f32::new(0.0f32);
                    if col < dv {
                        let idx = ((chunk - 1usize) * 64usize + 4usize * ty + p) * dv + col;
                        x = dv_local[idx];
                        y = d_o[idx];
                    }
                    pre_l[p * 2usize + cc] = x;
                    pre_o[p * 2usize + cc] = y;
                }
            }
            pre_gl = gl[chunk - 1usize];
        }

        // dS = exp(G_last) dS + qg^T dO - W^T dV'.
        let mut acc2 = Array::<f32>::new(16usize);
        #[unroll]
        for x in 0..16usize {
            acc2[x] = decay * dsreg[x];
        }
        #[unroll]
        for i in 0..64usize {
            let ov = dos[i * 16usize + tx];
            let dvv = dvs[i * 16usize + tx];
            #[unroll]
            for half in 0..2usize {
                let qv = xs[i * 32usize + 2usize * ty + half];
                let wv = ys[i * 32usize + 2usize * ty + half];
                #[unroll]
                for r in 0..4usize {
                    #[unroll]
                    for cc in 0..2usize {
                        acc2[(half * 4usize + r) * 2usize + cc] +=
                            qv.extract(r) * ov.extract(cc) - wv.extract(r) * dvv.extract(cc);
                    }
                }
            }
        }
        #[unroll]
        for q in 0..8usize {
            let mut pair = V2::empty();
            #[unroll]
            for cc in 0..2usize {
                dsreg[q * 2usize + cc] = acc2[q * 2usize + cc];
                pair.insert(cc, acc2[q * 2usize + cc]);
            }
            dss[(8usize * ty + q) * 16usize + tx] = pair;
        }
        sync_cube();
    }

    #[unroll]
    for q in 0..8usize {
        let k = 8usize * ty + q;
        #[unroll]
        for cc in 0..2usize {
            let col = col0 + cc;
            if col < dv && k < dk {
                d_first[(bh * dk + k) * dv + col] = dsreg[q * 2usize + cc];
            }
        }
    }
}

/// Loads chunk `c` of a `[B, T, H, D]` tensor into shared memory as f32, rows padded to `D + 1`
/// (`dst[i * (D + 1) + d]`), zero past the sequence's end; optionally normalises each row
/// (transformers' l2norm) and scales it.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_rows<E: Float>(
    src: &Tensor<E>,
    dst: &mut Shared<[f32]>,
    norm: &mut Shared<[f32]>,
    b: usize,
    h: usize,
    c: usize,
    seq_len: usize,
    heads: usize,
    scale: f32,
    #[comptime] depth: usize,
    #[comptime] l2norm: bool,
) {
    let tid = UNIT_POS_X as usize;
    let stride = comptime![depth + 1usize];
    #[unroll]
    for m_ in 0..comptime![(64usize * depth).div_ceil(256usize)] {
        let e = tid + 256usize * m_;
        if e < 64usize * depth {
            let i = e / depth;
            let d = e % depth;
            let t = c * 64usize + i;
            let mut x = f32::new(0.0f32);
            if t < seq_len {
                x = f32::cast_from(src[((b * seq_len + t) * heads + h) * depth + d]);
            }
            dst[i * stride + d] = x;
        }
    }
    sync_cube();
    if l2norm {
        if tid < 64usize {
            let mut ss = f32::new(0.0f32);
            for d in 0..depth {
                let x = dst[tid * stride + d];
                ss += x * x;
            }
            norm[tid] = scale / (ss + f32::new(1e-6f32)).sqrt();
        }
    } else if tid < 64usize {
        norm[tid] = scale;
    }
    sync_cube();
    #[unroll]
    for m_ in 0..comptime![(64usize * depth).div_ceil(256usize)] {
        let e = tid + 256usize * m_;
        if e < 64usize * depth {
            let i = e / depth;
            let d = e % depth;
            dst[i * stride + d] *= norm[i];
        }
    }
    sync_cube();
}

/// Per chunk, everything the state passing needs that does not depend on the state: `G`, `A`,
/// `T = (I + A)^-1`, `W = T (beta gamma K)`, `U = T (beta V)`, `kt = lam K`, from the raw inputs
/// (keys normalised here when asked). One cube per chunk of each batch element and head, 16 x 16
/// threads.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn prep_kernel<E: Float>(
    k: &Tensor<E>,
    v: &Tensor<E>,
    g: &Tensor<f32>,
    beta: &Tensor<f32>,
    w: &mut Tensor<f32>,
    u: &mut Tensor<f32>,
    kt: &mut Tensor<f32>,
    gcum: &mut Tensor<f32>,
    tmat: &mut Tensor<f32>,
    amat: &mut Tensor<f32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] l2norm: bool,
    #[comptime] store_ta: bool,
    #[define(E)] _dtype: ElemType,
) {
    let dmax = comptime![if dk > dv { dk } else { dv }];
    let dks = comptime![dk + 1usize];
    let dvs = comptime![dv + 1usize];
    let nbk = comptime![dk.div_ceil(16usize)];
    let nbv = comptime![dv.div_ceil(16usize)];
    let cube = CUBE_POS_X as usize;
    let n = n_chunks as usize;
    let hh = heads as usize;
    let t_len = seq_len as usize;
    let c = cube % n;
    let bh = cube / n;
    let b = bh / hh;
    let h = bh % hh;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;

    let mut rows = Shared::<[f32]>::new_slice(64usize * (dmax + 1usize));
    let mut am = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut tm = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut bs = Shared::<[f32]>::new_slice(64usize);
    let mut norm = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        let t = c * 64usize + tid;
        let mut gv = f32::new(0.0f32);
        let mut bv = f32::new(0.0f32);
        if t < t_len {
            gv = g[(b * t_len + t) * hh + h];
            bv = beta[(b * t_len + t) * hh + h];
        }
        gs[tid] = gv;
        bs[tid] = bv;
    }
    load_rows::<E>(
        k,
        &mut rows,
        &mut norm,
        b,
        h,
        c,
        t_len,
        hh,
        f32::new(1.0f32),
        dk,
        l2norm,
    );
    if tid == 0usize {
        let mut acc = f32::new(0.0f32);
        for i in 0..64usize {
            acc += gs[i];
            gs[i] = acc;
        }
    }
    sync_cube();

    // A[i, j] = beta_i (k_i . k_j) exp(G_i - G_j), i > j.
    let mut acc = Array::<f32>::new(16usize);
    #[unroll]
    for x in 0..16usize {
        acc[x] = f32::new(0.0f32);
    }
    for d in 0..dk {
        let mut ai = Array::<f32>::new(4usize);
        let mut bj = Array::<f32>::new(4usize);
        #[unroll]
        for a in 0..4usize {
            ai[a] = rows[(ty + 16usize * a) * dks + d];
            bj[a] = rows[(tx + 16usize * a) * dks + d];
        }
        #[unroll]
        for a in 0..4usize {
            #[unroll]
            for bb in 0..4usize {
                acc[a * 4usize + bb] += ai[a] * bj[bb];
            }
        }
    }
    #[unroll]
    for a in 0..4usize {
        #[unroll]
        for bb in 0..4usize {
            let i = ty + 16usize * a;
            let j = tx + 16usize * bb;
            let mut x = f32::new(0.0f32);
            if i > j {
                x = bs[i] * acc[a * 4usize + bb] * (gs[i] - gs[j]).exp();
            }
            am[i * 65usize + j] = x;
        }
    }
    sync_cube();

    // T = (I + A)^-1 by blocks of 16: the four diagonal blocks by forward substitution, one
    // column per thread (T[i, j] = -A[i, j] - sum_{j<l<i} A[i, l] T[l, j]); then block row r:
    // T[r, c] = -T[r, r] sum_{c<=m<r} A[r, m] T[m, c].
    #[unroll]
    for m_ in 0..comptime![(4160usize).div_ceil(256usize)] {
        let e = tid + 256usize * m_;
        if e < 4160usize {
            tm[e] = f32::new(0.0f32);
        }
    }
    sync_cube();
    if tid < 64usize {
        let blk0 = (tid / 16usize) * 16usize;
        let j = tid;
        for il in 0..16usize {
            let i = blk0 + il;
            let mut x = f32::new(0.0f32);
            if i == j {
                x = f32::new(1.0f32);
            } else if i > j {
                x = f32::new(0.0f32) - am[i * 65usize + j];
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
        // X[c] = sum_{c<=m<r} A[r, m] T[m, c], stashed in the strictly upper part of `am`
        // (rows 0..16, columns 16..64), which A does not use and nothing reads in this phase.
        let i = r * 16usize + il;
        for cc in 0..r {
            let mut x = f32::new(0.0f32);
            let mut l = cc * 16usize;
            while l < r * 16usize {
                x += am[i * 65usize + l] * tm[l * 65usize + cc * 16usize + jl];
                l += 1usize;
            }
            am[il * 65usize + 16usize + cc * 16usize + jl] = x;
        }
        sync_cube();
        for cc in 0..r {
            let mut x = f32::new(0.0f32);
            for l in 0..16usize {
                x -= tm[i * 65usize + r * 16usize + l]
                    * am[l * 65usize + 16usize + cc * 16usize + jl];
            }
            tm[i * 65usize + cc * 16usize + jl] = x;
        }
        sync_cube();
    }

    if store_ta {
        #[unroll]
        for m_ in 0..comptime![(4096usize).div_ceil(256usize)] {
            let e = tid + 256usize * m_;
            if e < 4096usize {
                let x = (e / 64usize) * 65usize + e % 64usize;
                tmat[cube * 4096usize + e] = tm[x];
                amat[cube * 4096usize + e] = am[x];
            }
        }
    }
    if tid < 64usize {
        gcum[cube * 64usize + tid] = gs[tid];
    }

    // W = T (beta gamma K) and kt = lam K.
    let g_last = gs[63usize];
    let mut accw = Array::<f32>::new(4usize * nbk);
    #[unroll]
    for x in 0..4usize * nbk {
        accw[x] = f32::new(0.0f32);
    }
    for j in 0..64usize {
        let sj = bs[j] * gs[j].exp();
        let mut ti = Array::<f32>::new(4usize);
        #[unroll]
        for a in 0..4usize {
            ti[a] = tm[(ty + 16usize * a) * 65usize + j] * sj;
        }
        #[unroll]
        for bb in 0..nbk {
            let d = tx + 16usize * bb;
            let mut kv = f32::new(0.0f32);
            if d < dk {
                kv = rows[j * dks + d];
            }
            #[unroll]
            for a in 0..4usize {
                accw[a * nbk + bb] += ti[a] * kv;
            }
        }
    }
    #[unroll]
    for a in 0..4usize {
        let i = ty + 16usize * a;
        let lam = (g_last - gs[i]).exp();
        #[unroll]
        for bb in 0..nbk {
            let d = tx + 16usize * bb;
            if d < dk {
                w[(cube * 64usize + i) * dk + d] = accw[a * nbk + bb];
                kt[(cube * 64usize + i) * dk + d] = rows[i * dks + d] * lam;
            }
        }
    }
    sync_cube();

    // U = T (beta V).
    load_rows::<E>(
        v,
        &mut rows,
        &mut norm,
        b,
        h,
        c,
        t_len,
        hh,
        f32::new(1.0f32),
        dv,
        false,
    );
    let mut accu = Array::<f32>::new(4usize * nbv);
    #[unroll]
    for x in 0..4usize * nbv {
        accu[x] = f32::new(0.0f32);
    }
    for j in 0..64usize {
        let sj = bs[j];
        let mut ti = Array::<f32>::new(4usize);
        #[unroll]
        for a in 0..4usize {
            ti[a] = tm[(ty + 16usize * a) * 65usize + j] * sj;
        }
        #[unroll]
        for bb in 0..nbv {
            let d = tx + 16usize * bb;
            let mut vv = f32::new(0.0f32);
            if d < dv {
                vv = rows[j * dvs + d];
            }
            #[unroll]
            for a in 0..4usize {
                accu[a * nbv + bb] += ti[a] * vv;
            }
        }
    }
    #[unroll]
    for a in 0..4usize {
        let i = ty + 16usize * a;
        #[unroll]
        for bb in 0..nbv {
            let d = tx + 16usize * bb;
            if d < dv {
                u[(cube * 64usize + i) * dv + d] = accu[a * nbv + bb];
            }
        }
    }
}

/// Per chunk, the output `O = (gamma Q) S + P V'` with `P = (Q K^T) exp(G_i - G_j)` (`i >= j`),
/// from the raw queries and keys (normalised and scaled here), written in the queries' dtype in
/// the `[B, T, H, Dv]` layout. One cube per chunk of each batch element and head.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn output_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    gcum: &Tensor<f32>,
    states: &Tensor<f32>,
    v_new: &Tensor<f32>,
    out: &mut Tensor<E>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    scale: f32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] l2norm: bool,
    #[define(E)] _dtype: ElemType,
) {
    let dks = comptime![dk + 1usize];
    let nbv = comptime![dv.div_ceil(16usize)];
    let ndb = comptime![dk.div_ceil(16usize)];
    let cube = CUBE_POS_X as usize;
    let n = n_chunks as usize;
    let hh = heads as usize;
    let t_len = seq_len as usize;
    let c = cube % n;
    let bh = cube / n;
    let b = bh / hh;
    let h = bh % hh;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;

    // ks holds the keys, then 16 rows of the state or of V' at a time.
    let ksz = comptime![if 64usize * (dk + 1usize) > 16usize * dv {
        64usize * (dk + 1usize)
    } else {
        16usize * dv
    }];
    let mut qs = Shared::<[f32]>::new_slice(64usize * dks);
    let mut ks = Shared::<[f32]>::new_slice(ksz);
    let mut pm = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut norm = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        gs[tid] = gcum[cube * 64usize + tid];
    }
    load_rows::<E>(q, &mut qs, &mut norm, b, h, c, t_len, hh, scale, dk, l2norm);
    load_rows::<E>(
        k,
        &mut ks,
        &mut norm,
        b,
        h,
        c,
        t_len,
        hh,
        f32::new(1.0f32),
        dk,
        l2norm,
    );

    // P = (Q K^T) exp(G_i - G_j), i >= j.
    let mut acc = Array::<f32>::new(16usize);
    #[unroll]
    for x in 0..16usize {
        acc[x] = f32::new(0.0f32);
    }
    for d in 0..dk {
        let mut ai = Array::<f32>::new(4usize);
        let mut bj = Array::<f32>::new(4usize);
        #[unroll]
        for a in 0..4usize {
            ai[a] = qs[(ty + 16usize * a) * dks + d];
            bj[a] = ks[(tx + 16usize * a) * dks + d];
        }
        #[unroll]
        for a in 0..4usize {
            #[unroll]
            for bb in 0..4usize {
                acc[a * 4usize + bb] += ai[a] * bj[bb];
            }
        }
    }
    #[unroll]
    for a in 0..4usize {
        #[unroll]
        for bb in 0..4usize {
            let i = ty + 16usize * a;
            let j = tx + 16usize * bb;
            let mut x = f32::new(0.0f32);
            if i >= j {
                x = acc[a * 4usize + bb] * (gs[i] - gs[j]).exp();
            }
            pm[i * 65usize + j] = x;
        }
    }
    sync_cube();
    // gamma Q, in place.
    #[unroll]
    for m_ in 0..comptime![(64usize * dk).div_ceil(256usize)] {
        let e = tid + 256usize * m_;
        if e < 64usize * dk {
            let i = e / dk;
            let d = e % dk;
            qs[i * dks + d] *= gs[i].exp();
        }
    }
    sync_cube();

    // O = (gamma Q) S + P V', the thread's rows ty + 16 a and columns tx + 16 b, the state and
    // V' streamed through `ks` 16 rows at a time.
    let mut o = Array::<f32>::new(4usize * nbv);
    #[unroll]
    for x in 0..4usize * nbv {
        o[x] = f32::new(0.0f32);
    }
    for d0 in 0..ndb {
        #[unroll]
        for m_ in 0..comptime![(16usize * dv).div_ceil(256usize)] {
            let e = tid + 256usize * m_;
            if e < 16usize * dv {
                let d = d0 * 16usize + e / dv;
                let mut x = f32::new(0.0f32);
                if d < dk {
                    x = states[(cube * dk + d) * dv + e % dv];
                }
                ks[e] = x;
            }
        }
        sync_cube();
        for dd in 0..16usize {
            let d = d0 * 16usize + dd;
            let mut qv = Array::<f32>::new(4usize);
            #[unroll]
            for a in 0..4usize {
                let mut x = f32::new(0.0f32);
                if d < dk {
                    x = qs[(ty + 16usize * a) * dks + d];
                }
                qv[a] = x;
            }
            #[unroll]
            for bb in 0..nbv {
                let col = tx + 16usize * bb;
                let mut sv = f32::new(0.0f32);
                if col < dv {
                    sv = ks[dd * dv + col];
                }
                #[unroll]
                for a in 0..4usize {
                    o[a * nbv + bb] += qv[a] * sv;
                }
            }
        }
        sync_cube();
    }
    for j0 in 0..4usize {
        #[unroll]
        for m_ in 0..comptime![(16usize * dv).div_ceil(256usize)] {
            let e = tid + 256usize * m_;
            if e < 16usize * dv {
                ks[e] = v_new[(cube * 64usize + j0 * 16usize + e / dv) * dv + e % dv];
            }
        }
        sync_cube();
        for jj in 0..16usize {
            let j = j0 * 16usize + jj;
            let mut pv = Array::<f32>::new(4usize);
            #[unroll]
            for a in 0..4usize {
                pv[a] = pm[(ty + 16usize * a) * 65usize + j];
            }
            #[unroll]
            for bb in 0..nbv {
                let col = tx + 16usize * bb;
                let mut vv = f32::new(0.0f32);
                if col < dv {
                    vv = ks[jj * dv + col];
                }
                #[unroll]
                for a in 0..4usize {
                    o[a * nbv + bb] += pv[a] * vv;
                }
            }
        }
        sync_cube();
    }
    #[unroll]
    for a in 0..4usize {
        let t = c * 64usize + ty + 16usize * a;
        #[unroll]
        for bb in 0..nbv {
            let col = tx + 16usize * bb;
            if t < t_len && col < dv {
                out[((b * t_len + t) * hh + h) * dv + col] = E::cast_from(o[a * nbv + bb]);
            }
        }
    }
}

/// Backward, per chunk: `P^T dO` (`dv_local`), `qg = gamma Q`, and `dO` in the chunked f32
/// layout the later kernels read.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn dvlocal_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    gcum: &Tensor<f32>,
    d_out: &Tensor<f32>,
    dv_local: &mut Tensor<f32>,
    qg: &mut Tensor<f32>,
    d_o: &mut Tensor<f32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    scale: f32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] l2norm: bool,
    #[define(E)] _dtype: ElemType,
) {
    let dks = comptime![dk + 1usize];
    let nbv = comptime![dv.div_ceil(16usize)];
    let cube = CUBE_POS_X as usize;
    let n = n_chunks as usize;
    let hh = heads as usize;
    let t_len = seq_len as usize;
    let c = cube % n;
    let bh = cube / n;
    let b = bh / hh;
    let h = bh % hh;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    // ks holds the keys, then 16 rows of dO at a time.
    let ksz = comptime![if 64usize * (dk + 1usize) > 16usize * dv {
        64usize * (dk + 1usize)
    } else {
        16usize * dv
    }];
    let mut qs = Shared::<[f32]>::new_slice(64usize * dks);
    let mut ks = Shared::<[f32]>::new_slice(ksz);
    let mut pm = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut norm = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        gs[tid] = gcum[cube * 64usize + tid];
    }
    load_rows::<E>(q, &mut qs, &mut norm, b, h, c, t_len, hh, scale, dk, l2norm);
    load_rows::<E>(
        k,
        &mut ks,
        &mut norm,
        b,
        h,
        c,
        t_len,
        hh,
        f32::new(1.0f32),
        dk,
        l2norm,
    );
    let mut acc = Array::<f32>::new(16usize);
    #[unroll]
    for x in 0..16usize {
        acc[x] = f32::new(0.0f32);
    }
    for d in 0..dk {
        let mut ai = Array::<f32>::new(4usize);
        let mut bj = Array::<f32>::new(4usize);
        #[unroll]
        for a in 0..4usize {
            ai[a] = qs[(ty + 16usize * a) * dks + d];
            bj[a] = ks[(tx + 16usize * a) * dks + d];
        }
        #[unroll]
        for a in 0..4usize {
            #[unroll]
            for bb in 0..4usize {
                acc[a * 4usize + bb] += ai[a] * bj[bb];
            }
        }
    }
    #[unroll]
    for a in 0..4usize {
        #[unroll]
        for bb in 0..4usize {
            let i = ty + 16usize * a;
            let j = tx + 16usize * bb;
            let mut x = f32::new(0.0f32);
            if i >= j {
                x = acc[a * 4usize + bb] * (gs[i] - gs[j]).exp();
            }
            pm[i * 65usize + j] = x;
        }
    }
    #[unroll]
    for m_ in 0..comptime![(64usize * dk).div_ceil(256usize)] {
        let e = tid + 256usize * m_;
        if e < 64usize * dk {
            let i = e / dk;
            let d = e % dk;
            qg[(cube * 64usize + i) * dk + d] = qs[i * dks + d] * gs[i].exp();
        }
    }
    sync_cube();

    // dv_local = P^T dO, the thread's rows ty + 16 a and columns tx + 16 b, dO streamed through
    // `ks` 16 rows at a time (and written in the chunked layout on the way).
    let mut o = Array::<f32>::new(4usize * nbv);
    #[unroll]
    for x in 0..4usize * nbv {
        o[x] = f32::new(0.0f32);
    }
    for i0 in 0..4usize {
        #[unroll]
        for m_ in 0..comptime![(16usize * dv).div_ceil(256usize)] {
            let e = tid + 256usize * m_;
            if e < 16usize * dv {
                let i = i0 * 16usize + e / dv;
                let col = e % dv;
                let t = c * 64usize + i;
                let mut x = f32::new(0.0f32);
                if t < t_len {
                    x = d_out[((b * t_len + t) * hh + h) * dv + col];
                }
                ks[e] = x;
                d_o[(cube * 64usize + i) * dv + col] = x;
            }
        }
        sync_cube();
        for ii in 0..16usize {
            let i = i0 * 16usize + ii;
            let mut pv = Array::<f32>::new(4usize);
            #[unroll]
            for a in 0..4usize {
                pv[a] = pm[i * 65usize + ty + 16usize * a];
            }
            #[unroll]
            for bb in 0..nbv {
                let col = tx + 16usize * bb;
                let mut dd = f32::new(0.0f32);
                if col < dv {
                    dd = ks[ii * dv + col];
                }
                #[unroll]
                for a in 0..4usize {
                    o[a * nbv + bb] += pv[a] * dd;
                }
            }
        }
        sync_cube();
    }
    #[unroll]
    for a in 0..4usize {
        let j = ty + 16usize * a;
        #[unroll]
        for bb in 0..nbv {
            let col = tx + 16usize * bb;
            if col < dv {
                dv_local[(cube * 64usize + j) * dv + col] = o[a * nbv + bb];
            }
        }
    }
}

/// Backward, per chunk: `dV_b = T^T dV'`, the values' gradient `dV_b beta` (written in the
/// values' dtype and layout), and `sum_v dV_b V`, a part of `beta`'s gradient.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn dvb_kernel<E: Float>(
    tmat: &Tensor<f32>,
    dvn: &Tensor<f32>,
    v: &Tensor<E>,
    beta: &Tensor<f32>,
    dv_out: &mut Tensor<E>,
    dvb: &mut Tensor<f32>,
    dbeta_v: &mut Tensor<f32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    #[comptime] dv: usize,
    #[define(E)] _dtype: ElemType,
) {
    let dvs = comptime![dv + 1usize];
    let nbv = comptime![dv.div_ceil(16usize)];
    let cube = CUBE_POS_X as usize;
    let n = n_chunks as usize;
    let hh = heads as usize;
    let t_len = seq_len as usize;
    let c = cube % n;
    let bh = cube / n;
    let b = bh / hh;
    let h = bh % hh;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let mut tm = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut ds = Shared::<[f32]>::new_slice(64usize * dvs);
    let mut bs = Shared::<[f32]>::new_slice(64usize);
    let mut red = Shared::<[f32]>::new_slice(1024usize);

    #[unroll]
    for m_ in 0..comptime![(4096usize).div_ceil(256usize)] {
        let e = tid + 256usize * m_;
        if e < 4096usize {
            tm[(e / 64usize) * 65usize + e % 64usize] = tmat[cube * 4096usize + e];
        }
    }
    #[unroll]
    for m_ in 0..comptime![(64usize * dv).div_ceil(256usize)] {
        let e = tid + 256usize * m_;
        if e < 64usize * dv {
            ds[(e / dv) * dvs + e % dv] = dvn[cube * 64usize * dv + e];
        }
    }
    if tid < 64usize {
        let t = c * 64usize + tid;
        let mut x = f32::new(0.0f32);
        if t < t_len {
            x = beta[(b * t_len + t) * hh + h];
        }
        bs[tid] = x;
    }
    sync_cube();

    let mut acc = Array::<f32>::new(4usize * nbv);
    #[unroll]
    for x in 0..4usize * nbv {
        acc[x] = f32::new(0.0f32);
    }
    for i in 0..64usize {
        let mut ti = Array::<f32>::new(4usize);
        #[unroll]
        for a in 0..4usize {
            ti[a] = tm[i * 65usize + ty + 16usize * a];
        }
        #[unroll]
        for bb in 0..nbv {
            let d = tx + 16usize * bb;
            let mut x = f32::new(0.0f32);
            if d < dv {
                x = ds[i * dvs + d];
            }
            #[unroll]
            for a in 0..4usize {
                acc[a * nbv + bb] += ti[a] * x;
            }
        }
    }
    #[unroll]
    for a in 0..4usize {
        let j = ty + 16usize * a;
        let t = c * 64usize + j;
        let mut part = f32::new(0.0f32);
        #[unroll]
        for bb in 0..nbv {
            let d = tx + 16usize * bb;
            if d < dv {
                let x = acc[a * nbv + bb];
                dvb[(cube * 64usize + j) * dv + d] = x;
                if t < t_len {
                    let idx = ((b * t_len + t) * hh + h) * dv + d;
                    dv_out[idx] = E::cast_from(x * bs[j]);
                    part += x * f32::cast_from(v[idx]);
                }
            }
        }
        red[j * 16usize + tx] = part;
    }
    sync_cube();
    if tid < 64usize {
        let mut x = f32::new(0.0f32);
        for y in 0..16usize {
            x += red[tid * 16usize + y];
        }
        dbeta_v[cube * 64usize + tid] = x;
    }
}

/// Backward, per chunk, the products with the state and its gradient (inner dimension Dv):
/// `dQg = dO S^T`, `dkt = V' dS^T`, `dKbg = -dV_b S^T` (which is `T^T dW`), and `<S, dS>`. The
/// operands are staged 32 columns at a time as vectors of 4 (rows padded to 9 vectors), so that
/// each thread reads 4 steps of the inner products per load.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn dstate_kernel(
    d_o: &Tensor<f32>,
    v_new: &Tensor<f32>,
    dvb: &Tensor<f32>,
    states: &Tensor<f32>,
    ds_next: &Tensor<f32>,
    dqg: &mut Tensor<f32>,
    dkt: &mut Tensor<f32>,
    dkbg: &mut Tensor<f32>,
    s_ds: &mut Tensor<f32>,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
) {
    let nbk = comptime![dk.div_ceil(16usize)];
    let dkp = comptime![dk.div_ceil(16usize) * 16usize];
    let nvb = comptime![dv.div_ceil(32usize)];
    let cube = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let mut so: Shared<[V4]> = Shared::new_slice(64usize * 9usize);
    let mut sv: Shared<[V4]> = Shared::new_slice(64usize * 9usize);
    let mut sd: Shared<[V4]> = Shared::new_slice(64usize * 9usize);
    let mut ss: Shared<[V4]> = Shared::new_slice(dkp * 9usize);
    let mut sz: Shared<[V4]> = Shared::new_slice(dkp * 9usize);
    let mut red = Shared::<[f32]>::new_slice(256usize);
    let mut acc1 = Array::<f32>::new(4usize * nbk);
    let mut acc2 = Array::<f32>::new(4usize * nbk);
    let mut acc3 = Array::<f32>::new(4usize * nbk);
    #[unroll]
    for x in 0..4usize * nbk {
        acc1[x] = f32::new(0.0f32);
        acc2[x] = f32::new(0.0f32);
        acc3[x] = f32::new(0.0f32);
    }
    let mut part = f32::new(0.0f32);
    for vb in 0..nvb {
        // 64 rows x 8 vectors of dO, V', dV_b; dkp rows x 8 vectors of S, dS.
        #[unroll]
        for m in 0..2usize {
            let e = tid + 256usize * m;
            let i = e / 8usize;
            let v4 = e % 8usize;
            let mut x = V4::empty();
            let mut y = V4::empty();
            let mut z = V4::empty();
            #[unroll]
            for r in 0..4usize {
                let col = vb * 32usize + 4usize * v4 + r;
                let mut xv = f32::new(0.0f32);
                let mut yv = f32::new(0.0f32);
                let mut zv = f32::new(0.0f32);
                if col < dv {
                    let idx = (cube * 64usize + i) * dv + col;
                    xv = d_o[idx];
                    yv = v_new[idx];
                    zv = dvb[idx];
                }
                x.insert(r, xv);
                y.insert(r, yv);
                z.insert(r, zv);
            }
            so[i * 9usize + v4] = x;
            sv[i * 9usize + v4] = y;
            sd[i * 9usize + v4] = z;
        }
        #[unroll]
        for m in 0..comptime![(dkp * 8usize).div_ceil(256usize)] {
            let e = tid + 256usize * m;
            if e < dkp * 8usize {
                let kk = e / 8usize;
                let v4 = e % 8usize;
                let mut x = V4::empty();
                let mut y = V4::empty();
                #[unroll]
                for r in 0..4usize {
                    let col = vb * 32usize + 4usize * v4 + r;
                    let mut xv = f32::new(0.0f32);
                    let mut yv = f32::new(0.0f32);
                    if col < dv && kk < dk {
                        let idx = (cube * dk + kk) * dv + col;
                        xv = states[idx];
                        yv = ds_next[idx];
                    }
                    part += xv * yv;
                    x.insert(r, xv);
                    y.insert(r, yv);
                }
                ss[kk * 9usize + v4] = x;
                sz[kk * 9usize + v4] = y;
            }
        }
        sync_cube();
        for v4 in 0..8usize {
            let mut o = Array::<V4>::new(4usize);
            let mut x = Array::<V4>::new(4usize);
            let mut d = Array::<V4>::new(4usize);
            #[unroll]
            for a in 0..4usize {
                let row = (ty + 16usize * a) * 9usize + v4;
                o[a] = so[row];
                x[a] = sv[row];
                d[a] = sd[row];
            }
            #[unroll]
            for bb in 0..nbk {
                let row = (tx + 16usize * bb) * 9usize + v4;
                let s1 = ss[row];
                let s2 = sz[row];
                #[unroll]
                for r in 0..4usize {
                    #[unroll]
                    for a in 0..4usize {
                        acc1[a * nbk + bb] += o[a].extract(r) * s1.extract(r);
                        acc2[a * nbk + bb] += x[a].extract(r) * s2.extract(r);
                        acc3[a * nbk + bb] -= d[a].extract(r) * s1.extract(r);
                    }
                }
            }
        }
        sync_cube();
    }
    #[unroll]
    for a in 0..4usize {
        #[unroll]
        for bb in 0..nbk {
            let kk = tx + 16usize * bb;
            if kk < dk {
                let idx = (cube * 64usize + ty + 16usize * a) * dk + kk;
                dqg[idx] = acc1[a * nbk + bb];
                dkt[idx] = acc2[a * nbk + bb];
                dkbg[idx] = acc3[a * nbk + bb];
            }
        }
    }
    red[tid] = part;
    sync_cube();
    if tid == 0usize {
        let mut x = f32::new(0.0f32);
        for y in 0..256usize {
            x += red[y];
        }
        s_ds[cube] = x;
    }
}

/// Backward, per chunk, the attention part: `dP = dO V'^T` (`i >= j`), the gate's gradient
/// from `P` (row sums minus column sums of `dP P`), `dQ = dPe K` and `dK = dPe^T Q` with
/// `dPe = dP exp(G_i - G_j)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn dattn_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    gcum: &Tensor<f32>,
    d_o: &Tensor<f32>,
    v_new: &Tensor<f32>,
    dq_attn: &mut Tensor<f32>,
    dk_attn: &mut Tensor<f32>,
    dg_p: &mut Tensor<f32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    scale: f32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] l2norm: bool,
    #[define(E)] _dtype: ElemType,
) {
    let dks = comptime![dk + 1usize];
    let nbk = comptime![dk.div_ceil(16usize)];
    let nvb = comptime![dv.div_ceil(32usize)];
    let cube = CUBE_POS_X as usize;
    let n = n_chunks as usize;
    let hh = heads as usize;
    let t_len = seq_len as usize;
    let c = cube % n;
    let bh = cube / n;
    let b = bh / hh;
    let h = bh % hh;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let mut qs = Shared::<[f32]>::new_slice(64usize * dks);
    let mut ks = Shared::<[f32]>::new_slice(64usize * dks);
    let mut dpm = Shared::<[f32]>::new_slice(64usize * 65usize);
    // dO and V' blocks (64 x 33 each), then dP P.
    let mut r4 = Shared::<[f32]>::new_slice(4224usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut norm = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        gs[tid] = gcum[cube * 64usize + tid];
    }
    let mut acc = Array::<f32>::new(16usize);
    #[unroll]
    for x in 0..16usize {
        acc[x] = f32::new(0.0f32);
    }
    for vb in 0..nvb {
        #[unroll]
        for m_ in 0..comptime![(2048usize).div_ceil(256usize)] {
            let e = tid + 256usize * m_;
            if e < 2048usize {
                let i = e / 32usize;
                let vv = e % 32usize;
                let col = vb * 32usize + vv;
                let mut x = f32::new(0.0f32);
                let mut y = f32::new(0.0f32);
                if col < dv {
                    let idx = (cube * 64usize + i) * dv + col;
                    x = d_o[idx];
                    y = v_new[idx];
                }
                r4[i * 33usize + vv] = x;
                r4[2112usize + i * 33usize + vv] = y;
            }
        }
        sync_cube();
        for vv in 0..32usize {
            let mut o = Array::<f32>::new(4usize);
            let mut x = Array::<f32>::new(4usize);
            #[unroll]
            for a in 0..4usize {
                o[a] = r4[(ty + 16usize * a) * 33usize + vv];
                x[a] = r4[2112usize + (tx + 16usize * a) * 33usize + vv];
            }
            #[unroll]
            for a in 0..4usize {
                #[unroll]
                for bb in 0..4usize {
                    acc[a * 4usize + bb] += o[a] * x[bb];
                }
            }
        }
        sync_cube();
    }
    #[unroll]
    for a in 0..4usize {
        #[unroll]
        for bb in 0..4usize {
            let i = ty + 16usize * a;
            let j = tx + 16usize * bb;
            let mut x = f32::new(0.0f32);
            if i >= j {
                x = acc[a * 4usize + bb];
            }
            dpm[i * 65usize + j] = x;
        }
    }
    load_rows::<E>(q, &mut qs, &mut norm, b, h, c, t_len, hh, scale, dk, l2norm);
    load_rows::<E>(
        k,
        &mut ks,
        &mut norm,
        b,
        h,
        c,
        t_len,
        hh,
        f32::new(1.0f32),
        dk,
        l2norm,
    );
    #[unroll]
    for x in 0..16usize {
        acc[x] = f32::new(0.0f32);
    }
    for d in 0..dk {
        let mut ai = Array::<f32>::new(4usize);
        let mut bj = Array::<f32>::new(4usize);
        #[unroll]
        for a in 0..4usize {
            ai[a] = qs[(ty + 16usize * a) * dks + d];
            bj[a] = ks[(tx + 16usize * a) * dks + d];
        }
        #[unroll]
        for a in 0..4usize {
            #[unroll]
            for bb in 0..4usize {
                acc[a * 4usize + bb] += ai[a] * bj[bb];
            }
        }
    }
    #[unroll]
    for a in 0..4usize {
        #[unroll]
        for bb in 0..4usize {
            let i = ty + 16usize * a;
            let j = tx + 16usize * bb;
            let mut m = f32::new(0.0f32);
            if i >= j {
                let decay = (gs[i] - gs[j]).exp();
                let dp = dpm[i * 65usize + j];
                m = dp * acc[a * 4usize + bb] * decay;
                dpm[i * 65usize + j] = dp * decay;
            }
            r4[i * 65usize + j] = m;
        }
    }
    sync_cube();
    if tid < 64usize {
        let mut rs = f32::new(0.0f32);
        let mut cs = f32::new(0.0f32);
        for x in 0..64usize {
            rs += r4[tid * 65usize + x];
            cs += r4[x * 65usize + tid];
        }
        dg_p[cube * 64usize + tid] = rs - cs;
    }
    let mut accq = Array::<f32>::new(4usize * nbk);
    let mut acck = Array::<f32>::new(4usize * nbk);
    #[unroll]
    for x in 0..4usize * nbk {
        accq[x] = f32::new(0.0f32);
        acck[x] = f32::new(0.0f32);
    }
    for j in 0..64usize {
        let mut pe = Array::<f32>::new(4usize);
        let mut pt = Array::<f32>::new(4usize);
        #[unroll]
        for a in 0..4usize {
            pe[a] = dpm[(ty + 16usize * a) * 65usize + j];
            pt[a] = dpm[j * 65usize + ty + 16usize * a];
        }
        #[unroll]
        for bb in 0..nbk {
            let kk = tx + 16usize * bb;
            let mut kv = f32::new(0.0f32);
            let mut qv = f32::new(0.0f32);
            if kk < dk {
                kv = ks[j * dks + kk];
                qv = qs[j * dks + kk];
            }
            #[unroll]
            for a in 0..4usize {
                accq[a * nbk + bb] += pe[a] * kv;
                acck[a * nbk + bb] += pt[a] * qv;
            }
        }
    }
    #[unroll]
    for a in 0..4usize {
        #[unroll]
        for bb in 0..nbk {
            let kk = tx + 16usize * bb;
            if kk < dk {
                let idx = (cube * 64usize + ty + 16usize * a) * dk + kk;
                dq_attn[idx] = accq[a * nbk + bb];
                dk_attn[idx] = acck[a * nbk + bb];
            }
        }
    }
}

/// `acc += X[:, cols] Y[:, cols]^T` for one block of 32 columns of two chunked `[.., 64, depth]`
/// tensors, the thread's 4 x 4 tile (rows `ty + 16 a`, columns `tx + 16 b`), staged in `blk`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn rows_product(
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    blk: &mut Shared<[f32]>,
    acc: &mut Array<f32>,
    cube: usize,
    vb: usize,
    #[comptime] depth: usize,
) {
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    #[unroll]
    for m_ in 0..comptime![(2048usize).div_ceil(256usize)] {
        let e = tid + 256usize * m_;
        if e < 2048usize {
            let i = e / 32usize;
            let vv = e % 32usize;
            let col = vb * 32usize + vv;
            let mut xv = f32::new(0.0f32);
            let mut yv = f32::new(0.0f32);
            if col < depth {
                let idx = (cube * 64usize + i) * depth + col;
                xv = x[idx];
                yv = y[idx];
            }
            blk[i * 33usize + vv] = xv;
            blk[2112usize + i * 33usize + vv] = yv;
        }
    }
    sync_cube();
    for vv in 0..32usize {
        let mut o = Array::<f32>::new(4usize);
        let mut z = Array::<f32>::new(4usize);
        #[unroll]
        for a in 0..4usize {
            o[a] = blk[(ty + 16usize * a) * 33usize + vv];
            z[a] = blk[2112usize + (tx + 16usize * a) * 33usize + vv];
        }
        #[unroll]
        for a in 0..4usize {
            #[unroll]
            for bb in 0..4usize {
                acc[a * 4usize + bb] += o[a] * z[bb];
            }
        }
    }
    sync_cube();
}

/// Backward, per chunk, the WY part: `dA = -(dV_b U^T + dKbg W^T)` (`i > j`, which is
/// `-(T^T dT T^T)`), the gate's gradient from `A`, and from `A = beta (K K^T) decay` the keys'
/// gradient `dAe K beta + dAe^T (beta K)` and `beta`'s `sum_k (dAe K) K`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn dwy_kernel<E: Float>(
    k: &Tensor<E>,
    gcum: &Tensor<f32>,
    beta: &Tensor<f32>,
    amat: &Tensor<f32>,
    dvb: &Tensor<f32>,
    dkbg: &Tensor<f32>,
    u: &Tensor<f32>,
    w: &Tensor<f32>,
    dk_a: &mut Tensor<f32>,
    dbeta_a: &mut Tensor<f32>,
    dg_a: &mut Tensor<f32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] l2norm: bool,
    #[define(E)] _dtype: ElemType,
) {
    let dks = comptime![dk + 1usize];
    let nbk = comptime![dk.div_ceil(16usize)];
    let nvb = comptime![dv.div_ceil(32usize)];
    let nkb = comptime![dk.div_ceil(32usize)];
    let cube = CUBE_POS_X as usize;
    let n = n_chunks as usize;
    let hh = heads as usize;
    let t_len = seq_len as usize;
    let c = cube % n;
    let bh = cube / n;
    let b = bh / hh;
    let h = bh % hh;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let mut blk = Shared::<[f32]>::new_slice(4224usize);
    let mut dam = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut mam = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut ks = Shared::<[f32]>::new_slice(64usize * dks);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut bs = Shared::<[f32]>::new_slice(64usize);
    let mut norm = Shared::<[f32]>::new_slice(64usize);
    let mut red = Shared::<[f32]>::new_slice(1024usize);

    if tid < 64usize {
        gs[tid] = gcum[cube * 64usize + tid];
        let t = c * 64usize + tid;
        let mut x = f32::new(0.0f32);
        if t < t_len {
            x = beta[(b * t_len + t) * hh + h];
        }
        bs[tid] = x;
    }
    let mut acc = Array::<f32>::new(16usize);
    #[unroll]
    for x in 0..16usize {
        acc[x] = f32::new(0.0f32);
    }
    // dV_b U^T over Dv, then dKbg W^T over Dk.
    for vb in 0..nvb {
        rows_product(dvb, u, &mut blk, &mut acc, cube, vb, dv);
    }
    for kb in 0..nkb {
        rows_product(dkbg, w, &mut blk, &mut acc, cube, kb, dk);
    }
    #[unroll]
    for a in 0..4usize {
        #[unroll]
        for bb in 0..4usize {
            let i = ty + 16usize * a;
            let j = tx + 16usize * bb;
            let mut m = f32::new(0.0f32);
            let mut de = f32::new(0.0f32);
            if i > j {
                let da = f32::new(0.0f32) - acc[a * 4usize + bb];
                m = da * amat[cube * 4096usize + i * 64usize + j];
                de = da * (gs[i] - gs[j]).exp();
            }
            mam[i * 65usize + j] = m;
            dam[i * 65usize + j] = de;
        }
    }
    load_rows::<E>(
        k,
        &mut ks,
        &mut norm,
        b,
        h,
        c,
        t_len,
        hh,
        f32::new(1.0f32),
        dk,
        l2norm,
    );
    if tid < 64usize {
        let mut rs = f32::new(0.0f32);
        let mut cs = f32::new(0.0f32);
        for x in 0..64usize {
            rs += mam[tid * 65usize + x];
            cs += mam[x * 65usize + tid];
        }
        dg_a[cube * 64usize + tid] = rs - cs;
    }
    let mut acc1 = Array::<f32>::new(4usize * nbk);
    let mut acc2 = Array::<f32>::new(4usize * nbk);
    #[unroll]
    for x in 0..4usize * nbk {
        acc1[x] = f32::new(0.0f32);
        acc2[x] = f32::new(0.0f32);
    }
    for j in 0..64usize {
        let mut pe = Array::<f32>::new(4usize);
        let mut pt = Array::<f32>::new(4usize);
        let bj = bs[j];
        #[unroll]
        for a in 0..4usize {
            pe[a] = dam[(ty + 16usize * a) * 65usize + j];
            pt[a] = dam[j * 65usize + ty + 16usize * a] * bj;
        }
        #[unroll]
        for bb in 0..nbk {
            let kk = tx + 16usize * bb;
            let mut kv = f32::new(0.0f32);
            if kk < dk {
                kv = ks[j * dks + kk];
            }
            #[unroll]
            for a in 0..4usize {
                acc1[a * nbk + bb] += pe[a] * kv;
                acc2[a * nbk + bb] += pt[a] * kv;
            }
        }
    }
    #[unroll]
    for a in 0..4usize {
        let r = ty + 16usize * a;
        let mut part = f32::new(0.0f32);
        #[unroll]
        for bb in 0..nbk {
            let kk = tx + 16usize * bb;
            if kk < dk {
                dk_a[(cube * 64usize + r) * dk + kk] =
                    acc1[a * nbk + bb] * bs[r] + acc2[a * nbk + bb];
                part += acc1[a * nbk + bb] * ks[r * dks + kk];
            }
        }
        red[r * 16usize + tx] = part;
    }
    sync_cube();
    if tid < 64usize {
        let mut x = f32::new(0.0f32);
        for y in 0..16usize {
            x += red[tid * 16usize + y];
        }
        dbeta_a[cube * 64usize + tid] = x;
    }
}

/// Backward, per chunk, the sum of the parts: the queries' and keys' gradients (through the
/// l2norm when asked, in their dtype and layout), the gate's (the reverse cumulative sum of
/// `dG` within the chunk) and `beta`'s. One warp per token, eight tokens per warp: written for
/// planes of 32 units, which [`requirements`] asks of the device.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn combine_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    gcum: &Tensor<f32>,
    beta: &Tensor<f32>,
    dqg: &Tensor<f32>,
    dkt: &Tensor<f32>,
    dkbg: &Tensor<f32>,
    dq_attn: &Tensor<f32>,
    dk_attn: &Tensor<f32>,
    dk_a: &Tensor<f32>,
    s_ds: &Tensor<f32>,
    dbeta_v: &Tensor<f32>,
    dbeta_a: &Tensor<f32>,
    dg_p: &Tensor<f32>,
    dg_a: &Tensor<f32>,
    dq: &mut Tensor<E>,
    dk_out: &mut Tensor<E>,
    dg: &mut Tensor<f32>,
    dbeta: &mut Tensor<f32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    scale: f32,
    #[comptime] dk: usize,
    #[comptime] l2norm: bool,
    #[define(E)] _dtype: ElemType,
) {
    let nm = comptime![dk.div_ceil(32usize)];
    let cube = CUBE_POS_X as usize;
    let n = n_chunks as usize;
    let hh = heads as usize;
    let t_len = seq_len as usize;
    let c = cube % n;
    let bh = cube / n;
    let b = bh / hh;
    let h = bh % hh;
    let tid = UNIT_POS_X as usize;
    let lane = tid % 32usize;
    let warp = tid / 32usize;
    let mut rg = Shared::<[f32]>::new_slice(64usize);
    let mut rl = Shared::<[f32]>::new_slice(64usize);
    let mut rb = Shared::<[f32]>::new_slice(64usize);
    let g_last = gcum[cube * 64usize + 63usize];

    for j in 0..8usize {
        let i = warp + 8usize * j;
        let t = c * 64usize + i;
        let live = t < t_len;
        let raw = ((b * t_len + t) * hh + h) * dk;
        let chunked = (cube * 64usize + i) * dk;
        let mut qv = Array::<f32>::new(nm);
        let mut kv = Array::<f32>::new(nm);
        let mut sq = f32::new(0.0f32);
        let mut sk = f32::new(0.0f32);
        #[unroll]
        for m in 0..nm {
            let d = lane + 32usize * m;
            let mut x = f32::new(0.0f32);
            let mut y = f32::new(0.0f32);
            if live && d < dk {
                x = f32::cast_from(q[raw + d]);
                y = f32::cast_from(k[raw + d]);
            }
            qv[m] = x;
            kv[m] = y;
            sq += x * x;
            sk += y * y;
        }
        let sq = plane_sum(sq);
        let sk = plane_sum(sk);
        let mut invq = f32::new(1.0f32);
        let mut invk = f32::new(1.0f32);
        if l2norm {
            invq = f32::new(1.0f32) / (sq + f32::new(1e-6f32)).sqrt();
            invk = f32::new(1.0f32) / (sk + f32::new(1e-6f32)).sqrt();
        }
        let gi = gcum[cube * 64usize + i];
        let gamma = gi.exp();
        let lam = (g_last - gi).exp();
        let mut bi = f32::new(0.0f32);
        if live {
            bi = beta[(b * t_len + t) * hh + h];
        }
        let mut s_qg = f32::new(0.0f32);
        let mut s_bg = f32::new(0.0f32);
        let mut s_kt = f32::new(0.0f32);
        let mut dqh = Array::<f32>::new(nm);
        let mut dkk = Array::<f32>::new(nm);
        #[unroll]
        for m in 0..nm {
            let d = lane + 32usize * m;
            let mut a = f32::new(0.0f32);
            let mut bq = f32::new(0.0f32);
            if live && d < dk {
                let x = chunked + d;
                let qg_x = dqg[x];
                let kbg_x = dkbg[x];
                let kt_x = dkt[x];
                s_qg += qg_x * qv[m] * invq * scale;
                s_bg += kbg_x * kv[m] * invk;
                s_kt += kt_x * kv[m] * invk;
                a = (qg_x * gamma + dq_attn[x]) * scale;
                bq = dk_attn[x] + kt_x * lam + kbg_x * bi * gamma + dk_a[x];
            }
            dqh[m] = a;
            dkk[m] = bq;
        }
        let s_qg = plane_sum(s_qg);
        let s_bg = plane_sum(s_bg);
        let s_kt = plane_sum(s_kt);
        let lam_dlam = lam * s_kt;
        if lane == 0usize {
            let d_gamma = s_qg + bi * s_bg;
            rg[i] =
                dg_p[cube * 64usize + i] + dg_a[cube * 64usize + i] + d_gamma * gamma - lam_dlam;
            rl[i] = lam_dlam;
            rb[i] = dbeta_v[cube * 64usize + i] + gamma * s_bg + dbeta_a[cube * 64usize + i];
        }
        let mut pq = f32::new(0.0f32);
        let mut pk = f32::new(0.0f32);
        #[unroll]
        for m in 0..nm {
            pq += qv[m] * invq * dqh[m];
            pk += kv[m] * invk * dkk[m];
        }
        let pq = plane_sum(pq);
        let pk = plane_sum(pk);
        #[unroll]
        for m in 0..nm {
            let d = lane + 32usize * m;
            if live && d < dk {
                let mut x = dqh[m];
                let mut y = dkk[m];
                if l2norm {
                    x = invq * (x - qv[m] * invq * pq);
                    y = invk * (y - kv[m] * invk * pk);
                }
                dq[raw + d] = E::cast_from(x);
                dk_out[raw + d] = E::cast_from(y);
            }
        }
    }
    sync_cube();
    if tid == 0usize {
        let mut tail = g_last.exp() * s_ds[cube];
        for x in 0..64usize {
            tail += rl[x];
        }
        rg[63usize] += tail;
        let mut acc = f32::new(0.0f32);
        for x in 0..64usize {
            let y = 63usize - x;
            acc += rg[y];
            rg[y] = acc;
        }
    }
    sync_cube();
    if tid < 64usize {
        let t = c * 64usize + tid;
        if t < t_len {
            dg[(b * t_len + t) * hh + h] = rg[tid];
            dbeta[(b * t_len + t) * hh + h] = rb[tid];
        }
    }
}

/// Counts one kernel launch in `probe::kernel_launches`.
fn launched() {
    probe::KERNEL_LAUNCHES.fetch_add(1, Ordering::SeqCst);
}

fn empty_like(reference: &CubeTensor, shape: Shape) -> CubeTensor {
    empty_device_contiguous_dtype(
        reference.client.clone(),
        reference.device.clone(),
        shape,
        DType::F32,
    )
}

/// The state passing, forward: the state entering each chunk `[B, H, N, Dk, Dv]`, `V'`
/// `[B, H, N, C, Dv]`, and the final state, from `kt`, `W` `[B, H, N, C, Dk]`, `U`
/// `[B, H, N, C, Dv]`, `exp(G_last)` `[B, H, N]` and the initial state.
fn state_forward(
    kt: CubeTensor,
    w: CubeTensor,
    u: CubeTensor,
    gl: CubeTensor,
    initial_state: CubeTensor,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    let [b, h, n, _, dk] = w.shape().dims::<5>();
    let dv = u.shape()[4];
    let states = empty_like(&w, Shape::new([b, h, n, dk, dv]));
    let v_new = empty_like(&w, Shape::new([b, h, n, CHUNK, dv]));
    let last = empty_like(&w, Shape::new([b, h, dk, dv]));
    launched();
    state_forward_kernel::launch(
        &w.client.clone(),
        CubeCount::Static(dv.div_ceil(BV) as u32, (b * h) as u32, 1),
        CubeDim::new_1d(THREADS),
        kt.into_tensor_arg(),
        w.into_tensor_arg(),
        u.into_tensor_arg(),
        gl.into_tensor_arg(),
        initial_state.into_tensor_arg(),
        states.clone().into_tensor_arg(),
        v_new.clone().into_tensor_arg(),
        last.clone().into_tensor_arg(),
        n as u32,
        dk,
        dv,
    );
    (states, v_new, last)
}

/// The state passing, backward: the gradient of the state leaving each chunk
/// `[B, H, N, Dk, Dv]`, `dV'` `[B, H, N, C, Dv]`, and the initial state's gradient.
#[allow(clippy::too_many_arguments)]
fn state_backward(
    kt: CubeTensor,
    w: CubeTensor,
    qg: CubeTensor,
    d_o: CubeTensor,
    dv_local: CubeTensor,
    gl: CubeTensor,
    d_final_state: CubeTensor,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    let [b, h, n, _, dk] = w.shape().dims::<5>();
    let dv = d_o.shape()[4];
    let ds_next = empty_like(&w, Shape::new([b, h, n, dk, dv]));
    let dvn = empty_like(&w, Shape::new([b, h, n, CHUNK, dv]));
    let d_first = empty_like(&w, Shape::new([b, h, dk, dv]));
    launched();
    state_backward_kernel::launch(
        &w.client.clone(),
        CubeCount::Static(dv.div_ceil(BV) as u32, (b * h) as u32, 1),
        CubeDim::new_1d(THREADS),
        kt.into_tensor_arg(),
        w.into_tensor_arg(),
        qg.into_tensor_arg(),
        d_o.into_tensor_arg(),
        dv_local.into_tensor_arg(),
        gl.into_tensor_arg(),
        d_final_state.into_tensor_arg(),
        ds_next.clone().into_tensor_arg(),
        dvn.clone().into_tensor_arg(),
        d_first.clone().into_tensor_arg(),
        n as u32,
        dk,
        dv,
    );
    (ds_next, dvn, d_first)
}

/// `exp(G_last)` per chunk, `[B, H, N, 1]`, contiguous.
fn state_decay(gcum: &CubeTensor) -> CubeTensor {
    type B = CubeBackend;
    into_contiguous(<B as FloatTensorOps<B>>::float_exp(<B as FloatTensorOps<
        B,
    >>::float_slice(
        gcum.clone(),
        &[
            Slice::from(..),
            Slice::from(..),
            Slice::from(..),
            Slice::from(CHUNK - 1..CHUNK),
        ],
    )))
}

/// The forward pass: the prep kernel, the state passing, the output kernel.
#[allow(clippy::type_complexity)]
fn forward_fused(
    q: CubeTensor,
    k: CubeTensor,
    v: CubeTensor,
    g: CubeTensor,
    beta: CubeTensor,
    initial_state: CubeTensor,
    use_qk_l2norm: bool,
) -> (CubeTensor, CubeTensor) {
    let [b, t, h, dk] = q.shape().dims::<4>();
    let dv = v.shape()[3];
    let n = t.div_ceil(CHUNK);
    let (q, k, v) = (into_contiguous(q), into_contiguous(k), into_contiguous(v));
    let p = prep(&k, v, g, beta, use_qk_l2norm, false, [b, t, h, dk, dv]);
    let gl = state_decay(&p.gcum);
    let h0 = into_contiguous(reference::to_f32::<CubeBackend>(initial_state));
    let (states, v_new, last) = state_forward(p.kt, p.w, p.u, gl, h0);
    let out = empty_device_contiguous_dtype(
        q.client.clone(),
        q.device.clone(),
        Shape::new([b, t, h, dv]),
        q.dtype,
    );
    let dtype = dtype_to_storage_type(q.dtype);
    launched();
    output_kernel::launch(
        &q.client.clone(),
        CubeCount::Static((b * h * n) as u32, 1, 1),
        CubeDim::new_1d(THREADS),
        q.into_tensor_arg(),
        k.into_tensor_arg(),
        p.gcum.into_tensor_arg(),
        states.into_tensor_arg(),
        v_new.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        t as u32,
        n as u32,
        h as u32,
        reference::scale(dk),
        dk,
        dv,
        use_qk_l2norm,
        dtype,
    );
    (out, last)
}

/// What the prep kernel forms, per chunk, all f32: `W`, `U`, `kt` `[B, H, N, C, D]`, `G`
/// `[B, H, N, C]`, and for the backward pass `T` and `A` `[B, H, N, C, C]`.
struct Prep {
    w: CubeTensor,
    u: CubeTensor,
    kt: CubeTensor,
    gcum: CubeTensor,
    t: CubeTensor,
    a: CubeTensor,
}

fn prep(
    k: &CubeTensor,
    v: CubeTensor,
    g: CubeTensor,
    beta: CubeTensor,
    use_qk_l2norm: bool,
    store_ta: bool,
    [b, t, h, dk, dv]: [usize; 5],
) -> Prep {
    let n = t.div_ceil(CHUNK);
    let g = into_contiguous(reference::to_f32::<CubeBackend>(g));
    let beta = into_contiguous(reference::to_f32::<CubeBackend>(beta));
    let w = empty_like(k, Shape::new([b, h, n, CHUNK, dk]));
    let u = empty_like(k, Shape::new([b, h, n, CHUNK, dv]));
    let kt = empty_like(k, Shape::new([b, h, n, CHUNK, dk]));
    let gcum = empty_like(k, Shape::new([b, h, n, CHUNK]));
    let ta = if store_ta {
        Shape::new([b, h, n, CHUNK, CHUNK])
    } else {
        Shape::new([1])
    };
    let tm = empty_like(k, ta.clone());
    let am = empty_like(k, ta);
    launched();
    prep_kernel::launch(
        &k.client.clone(),
        CubeCount::Static((b * h * n) as u32, 1, 1),
        CubeDim::new_1d(THREADS),
        k.clone().into_tensor_arg(),
        v.into_tensor_arg(),
        g.into_tensor_arg(),
        beta.into_tensor_arg(),
        w.clone().into_tensor_arg(),
        u.clone().into_tensor_arg(),
        kt.clone().into_tensor_arg(),
        gcum.clone().into_tensor_arg(),
        tm.clone().into_tensor_arg(),
        am.clone().into_tensor_arg(),
        t as u32,
        n as u32,
        h as u32,
        dk,
        dv,
        use_qk_l2norm,
        store_ta,
        dtype_to_storage_type(k.dtype),
    );
    Prep {
        w,
        u,
        kt,
        gcum,
        t: tm,
        a: am,
    }
}

fn launch_dims(b: usize, h: usize, n: usize) -> (CubeCount, CubeDim) {
    (
        CubeCount::Static((b * h * n) as u32, 1, 1),
        CubeDim::new_1d(THREADS),
    )
}

/// The backward pass: the forward's prep and state passing again, then the kernels above.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn backward_fused(
    q: CubeTensor,
    k: CubeTensor,
    v: CubeTensor,
    g: CubeTensor,
    beta: CubeTensor,
    initial_state: CubeTensor,
    d_out: CubeTensor,
    d_final_state: CubeTensor,
    use_qk_l2norm: bool,
) -> (
    CubeTensor,
    CubeTensor,
    CubeTensor,
    CubeTensor,
    CubeTensor,
    CubeTensor,
) {
    type B = CubeBackend;
    let [b, t, h, dk] = q.shape().dims::<4>();
    let dv = v.shape()[3];
    let n = t.div_ceil(CHUNK);
    let (g_dtype, beta_dtype) = (g.dtype, beta.dtype);
    let (q, k, v) = (into_contiguous(q), into_contiguous(k), into_contiguous(v));
    let g = into_contiguous(reference::to_f32::<B>(g));
    let beta = into_contiguous(reference::to_f32::<B>(beta));
    let p = prep(
        &k,
        v.clone(),
        g,
        beta.clone(),
        use_qk_l2norm,
        true,
        [b, t, h, dk, dv],
    );
    let gl = state_decay(&p.gcum);
    let h0 = into_contiguous(reference::to_f32::<B>(initial_state));
    let (states, v_new, _) = state_forward(p.kt.clone(), p.w.clone(), p.u.clone(), gl.clone(), h0);

    let client = q.client.clone();
    let dtype = dtype_to_storage_type(q.dtype);
    let (count, dim) = launch_dims(b, h, n);
    let scale = reference::scale(dk);
    let chunked = |depth: usize| Shape::new([b, h, n, CHUNK, depth]);
    let per_token = Shape::new([b, h, n, CHUNK]);
    let d_out = into_contiguous(reference::to_f32::<B>(d_out));
    let dv_local = empty_like(&q, chunked(dv));
    let qg = empty_like(&q, chunked(dk));
    let d_o = empty_like(&q, chunked(dv));
    launched();
    dvlocal_kernel::launch(
        &client,
        count.clone(),
        dim,
        q.clone().into_tensor_arg(),
        k.clone().into_tensor_arg(),
        p.gcum.clone().into_tensor_arg(),
        d_out.into_tensor_arg(),
        dv_local.clone().into_tensor_arg(),
        qg.clone().into_tensor_arg(),
        d_o.clone().into_tensor_arg(),
        t as u32,
        n as u32,
        h as u32,
        scale,
        dk,
        dv,
        use_qk_l2norm,
        dtype,
    );
    let (ds_next, dvn, d_initial) = state_backward(
        p.kt,
        p.w.clone(),
        qg,
        d_o.clone(),
        dv_local,
        gl,
        into_contiguous(reference::to_f32::<B>(d_final_state)),
    );

    let dv_out = empty_device_contiguous_dtype(
        client.clone(),
        q.device.clone(),
        Shape::new([b, t, h, dv]),
        q.dtype,
    );
    let dvb = empty_like(&q, chunked(dv));
    let dbeta_v = empty_like(&q, per_token.clone());
    launched();
    dvb_kernel::launch(
        &client,
        count.clone(),
        dim,
        p.t.into_tensor_arg(),
        dvn.into_tensor_arg(),
        v.into_tensor_arg(),
        beta.clone().into_tensor_arg(),
        dv_out.clone().into_tensor_arg(),
        dvb.clone().into_tensor_arg(),
        dbeta_v.clone().into_tensor_arg(),
        t as u32,
        n as u32,
        h as u32,
        dv,
        dtype,
    );

    let dqg = empty_like(&q, chunked(dk));
    let dkt = empty_like(&q, chunked(dk));
    let dkbg = empty_like(&q, chunked(dk));
    let s_ds = empty_like(&q, Shape::new([b, h, n]));
    launched();
    dstate_kernel::launch(
        &client,
        count.clone(),
        dim,
        d_o.clone().into_tensor_arg(),
        v_new.clone().into_tensor_arg(),
        dvb.clone().into_tensor_arg(),
        states.into_tensor_arg(),
        ds_next.into_tensor_arg(),
        dqg.clone().into_tensor_arg(),
        dkt.clone().into_tensor_arg(),
        dkbg.clone().into_tensor_arg(),
        s_ds.clone().into_tensor_arg(),
        dk,
        dv,
    );

    let dq_attn = empty_like(&q, chunked(dk));
    let dk_attn = empty_like(&q, chunked(dk));
    let dg_p = empty_like(&q, per_token.clone());
    launched();
    dattn_kernel::launch(
        &client,
        count.clone(),
        dim,
        q.clone().into_tensor_arg(),
        k.clone().into_tensor_arg(),
        p.gcum.clone().into_tensor_arg(),
        d_o.into_tensor_arg(),
        v_new.into_tensor_arg(),
        dq_attn.clone().into_tensor_arg(),
        dk_attn.clone().into_tensor_arg(),
        dg_p.clone().into_tensor_arg(),
        t as u32,
        n as u32,
        h as u32,
        scale,
        dk,
        dv,
        use_qk_l2norm,
        dtype,
    );

    let dk_a = empty_like(&q, chunked(dk));
    let dbeta_a = empty_like(&q, per_token.clone());
    let dg_a = empty_like(&q, per_token);
    launched();
    dwy_kernel::launch(
        &client,
        count.clone(),
        dim,
        k.clone().into_tensor_arg(),
        p.gcum.clone().into_tensor_arg(),
        beta.clone().into_tensor_arg(),
        p.a.into_tensor_arg(),
        dvb.into_tensor_arg(),
        dkbg.clone().into_tensor_arg(),
        p.u.into_tensor_arg(),
        p.w.into_tensor_arg(),
        dk_a.clone().into_tensor_arg(),
        dbeta_a.clone().into_tensor_arg(),
        dg_a.clone().into_tensor_arg(),
        t as u32,
        n as u32,
        h as u32,
        dk,
        dv,
        use_qk_l2norm,
        dtype,
    );

    let dq = empty_device_contiguous_dtype(
        client.clone(),
        q.device.clone(),
        Shape::new([b, t, h, dk]),
        q.dtype,
    );
    let dk_out = empty_device_contiguous_dtype(
        client.clone(),
        q.device.clone(),
        Shape::new([b, t, h, dk]),
        q.dtype,
    );
    let dg = empty_like(&q, Shape::new([b, t, h]));
    let dbeta = empty_like(&q, Shape::new([b, t, h]));
    launched();
    combine_kernel::launch(
        &client,
        count,
        dim,
        q.into_tensor_arg(),
        k.into_tensor_arg(),
        p.gcum.into_tensor_arg(),
        beta.into_tensor_arg(),
        dqg.into_tensor_arg(),
        dkt.into_tensor_arg(),
        dkbg.into_tensor_arg(),
        dq_attn.into_tensor_arg(),
        dk_attn.into_tensor_arg(),
        dk_a.into_tensor_arg(),
        s_ds.into_tensor_arg(),
        dbeta_v.into_tensor_arg(),
        dbeta_a.into_tensor_arg(),
        dg_p.into_tensor_arg(),
        dg_a.into_tensor_arg(),
        dq.clone().into_tensor_arg(),
        dk_out.clone().into_tensor_arg(),
        dg.clone().into_tensor_arg(),
        dbeta.clone().into_tensor_arg(),
        t as u32,
        n as u32,
        h as u32,
        scale,
        dk,
        use_qk_l2norm,
        dtype,
    );
    (
        dq,
        dk_out,
        dv_out,
        reference::to_dtype::<B>(dg, g_dtype),
        reference::to_dtype::<B>(dbeta, beta_dtype),
        d_initial,
    )
}

/// Bytes of shared memory the kernels of a forward (or backward) pass hold at most, summed from
/// each kernel's `Shared` arrays: f32 4 bytes, `V2` 8, `V4` 16. The state passings hold the most,
/// whatever the heads' width: 90 KiB forward, 98 KiB backward.
fn shared_bytes(dk: usize, dv: usize, backward: bool) -> usize {
    // A chunk's 64 rows padded to `depth + 1` columns, and its 64 x 64 blocks padded likewise.
    let rows = |depth: usize| 64 * (depth + 1);
    let square = 64 * 65;
    let prep = 4 * (rows(dk.max(dv)) + 2 * square + 3 * 64);
    let state_forward = 16 * (128 * 17 + 64 * 32) + 8 * (128 * 16 + 64 * 16);
    let output = 4 * (rows(dk) + rows(dk).max(16 * dv) + square + 2 * 64);
    if !backward {
        return prep.max(state_forward).max(output);
    }
    let dvlocal = output;
    let state_backward = 16 * (128 * 17 + 64 * 32) + 8 * (128 * 16 + 2 * 64 * 16);
    let dvb = 4 * (square + rows(dv) + 64 + 1024);
    let dstate = 16 * (3 * 64 * 9 + 2 * dk.div_ceil(16) * 16 * 9) + 4 * 256;
    let dattn = 4 * (2 * rows(dk) + square + 4224 + 2 * 64);
    let dwy = 4 * (4224 + 2 * square + rows(dk) + 3 * 64 + 1024);
    let combine = 4 * 3 * 64;
    [
        prep,
        state_forward,
        dvlocal,
        state_backward,
        dvb,
        dstate,
        dattn,
        dwy,
        combine,
    ]
    .into_iter()
    .fold(0, usize::max)
}

/// What the kernels of a forward (or backward) pass over `[B, T, H, Dk, Dv]` ask of the device:
/// 256 units per cube; `B H N` cubes, one per chunk of each batch element and head, and the
/// state passings' `ceil(Dv / 32)` by `B H`; the shared memory of [`shared_bytes`]; the tensors
/// of the kernel binding the most (`prep_kernel`'s 10 forward, `combine_kernel`'s 19 backward);
/// vectors of four f32; and backward, planes of 32 units, which `combine_kernel` sums over with
/// `plane_sum`, one plane per token, as CUDA's warps.
pub(super) fn requirements([b, t, h, dk, dv]: [usize; 5], backward: bool) -> Requirements {
    let n = t.div_ceil(CHUNK);
    let per_chunk = Requirements::new((THREADS, 1), (b * h * n, 1), if backward { 19 } else { 10 });
    let state = Requirements::new((THREADS, 1), (dv.div_ceil(BV), b * h), 10);
    Requirements {
        shared_bytes: shared_bytes(dk, dv, backward),
        vector_size: 4,
        plane_width: backward.then_some(32),
        ..per_chunk.max(state)
    }
}

/// Whether the kernels take a forward (or backward) pass over inputs of `dtype` and shape
/// `[B, T, H, Dk, Dv]` on the device `client` serves; the reference runs otherwise.
pub(super) fn kernels_take(
    client: &Client,
    dtype: DType,
    dims: [usize; 5],
    backward: bool,
) -> bool {
    let [_, _, _, dk, dv] = dims;
    dk <= MAX_HEAD_DIM
        && dv <= MAX_HEAD_DIM
        && requirements(dims, backward).met_on(client, &[dtype])
}

/// `[B, T, H, Dk, Dv]` of the queries `[B, T, H, Dk]` and the values `[B, T, H, Dv]`.
fn dims(q: &CubeTensor, v: &CubeTensor) -> [usize; 5] {
    let [b, t, h, dk] = q.shape().dims::<4>();
    [b, t, h, dk, v.shape()[3]]
}

impl ChunkGatedDeltaBackend for CubeBackend {
    fn chunk_gated_delta_forward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        use_qk_l2norm: bool,
        _output_final_state: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        if !kernels_take(&q.client, q.dtype, dims(&q, &v), false) {
            return reference::forward::<Self>(q, k, v, g, beta, initial_state, use_qk_l2norm);
        }
        forward_fused(q, k, v, g, beta, initial_state, use_qk_l2norm)
    }

    fn chunk_gated_delta_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        g: FloatTensor<Self>,
        beta: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        d_out: FloatTensor<Self>,
        d_final_state: FloatTensor<Self>,
        use_qk_l2norm: bool,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        if !kernels_take(&q.client, q.dtype, dims(&q, &v), true) {
            return reference::backward::<Self>(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                d_out,
                d_final_state,
                use_qk_l2norm,
            );
        }
        backward_fused(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            d_out,
            d_final_state,
            use_qk_l2norm,
        )
    }
}
