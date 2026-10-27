//! The kernels for queries and keys wider than [`HALF`] (`128 < Dk <= 256`), where those of the
//! parent module would hold more than a device's shared memory: queries, keys and the state's
//! rows are taken a half of `Dk` (128 columns) at a time. They do what their namesakes in the
//! parent do, with the same layouts and thread tiles; `prep_kernel`, `dvb_kernel` and
//! `combine_kernel` take wide heads as they are. Each kernel here holds at most 86 KiB of shared
//! memory, and `prep_kernel` 97.5 KiB at `Dk = Dv = 256`.
//!
//! - The l2 normalisation needs whole rows: [`row_scales`] forms each row's scale from the row in
//!   global memory first, and [`load_cols`] stages half a row already scaled.
//! - The state passings keep two halves of 8 state rows per thread, rows `8 ty ..` and
//!   `128 + 8 ty ..`, and stage `W`, `kt`, `qg` one half at a time through one buffer, loading
//!   it straight from global memory (no prefetching into registers, as the narrow passes do).
//! - The per-chunk products over `Dk` add up over the halves; those that form `Dk` columns form
//!   one half of them at a time.

use super::{V2, V4, chunk_tokens, rows_product, state_chunks};
use cubecl::prelude::*;

/// Columns of `Dk` a kernel here holds at a time.
pub(super) const HALF: usize = 128;

/// The scale of each of a chunk's 64 rows of head `h` of a `[B T, H, D]` tensor: `scale`, over
/// the row's length when `l2norm` (`1 / sqrt(sum x^2 + 1e-6)`, transformers' l2norm), into
/// `norm`. Four units sum each row's squares from global memory; `scratch` holds their 256
/// partial sums.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn row_scales<E: Float>(
    src: &Tensor<E>,
    norm: &mut Shared<[f32]>,
    scratch: &mut Shared<[f32]>,
    first: usize,
    count: usize,
    h: usize,
    heads: usize,
    scale: f32,
    #[comptime] depth: usize,
    #[comptime] l2norm: bool,
) {
    let tid = UNIT_POS_X as usize;
    if l2norm {
        let i = tid / 4usize;
        let mut ss = f32::new(0.0f32);
        if i < count {
            let base = ((first + i) * heads + h) * depth;
            let mut d = tid % 4usize;
            while d < depth {
                let x = f32::cast_from(src[base + d]);
                ss += x * x;
                d += 4usize;
            }
        }
        scratch[tid] = ss;
        sync_cube();
        if tid < 64usize {
            let s = scratch[4usize * tid]
                + scratch[4usize * tid + 1usize]
                + scratch[4usize * tid + 2usize]
                + scratch[4usize * tid + 3usize];
            norm[tid] = scale / (s + f32::new(1e-6f32)).sqrt();
        }
    } else if tid < 64usize {
        norm[tid] = scale;
    }
    sync_cube();
}

/// Columns `c0 .. c0 + 128` of a chunk's rows of head `h` of a `[B T, H, D]` tensor, each row
/// times its `norm`, into `dst` as f32 rows padded to 129 (`dst[i * 129 + d]`): the chunk's
/// `count` tokens from token `first`, zero past them and past `D`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn load_cols<E: Float>(
    src: &Tensor<E>,
    dst: &mut Shared<[f32]>,
    norm: &Shared<[f32]>,
    first: usize,
    count: usize,
    h: usize,
    heads: usize,
    c0: usize,
    #[comptime] depth: usize,
) {
    let tid = UNIT_POS_X as usize;
    #[unroll]
    for m_ in 0..comptime![64usize * HALF / 256usize] {
        let e = tid + 256usize * m_;
        let i = e / HALF;
        let d = e % HALF;
        let col = c0 + d;
        let mut x = f32::new(0.0f32);
        if i < count && col < depth {
            x = f32::cast_from(src[((first + i) * heads + h) * depth + col]) * norm[i];
        }
        dst[i * (HALF + 1usize) + d] = x;
    }
    sync_cube();
}

/// The state passing, forward, for `128 < Dk <= 256`: [`super::state_forward_kernel`]'s tiles,
/// with two halves of the state's rows per thread. For each chunk: store the entering state;
/// `V' = U - W S`, staging `W^T` a half at a time; then `S = exp(G_last) S + kt^T V'`, staging
/// `kt` a half at a time, each half updating its own rows.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub(super) fn state_forward_wide_kernel(
    kt: &Tensor<f32>,
    w: &Tensor<f32>,
    u: &Tensor<f32>,
    gl: &Tensor<f32>,
    h0: &Tensor<f32>,
    offsets: &Tensor<u32>,
    states: &mut Tensor<f32>,
    v_new: &mut Tensor<f32>,
    last: &mut Tensor<f32>,
    n_chunks: u32,
    heads: u32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] packed: bool,
) {
    let bh = CUBE_POS_Y as usize;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let col0 = CUBE_POS_X as usize * 32usize + 2usize * tx;
    let (base, n) = state_chunks(offsets, bh, heads as usize, n_chunks as usize, packed);
    // One half of W^T by vectors of 4 rows (st[k * 17 + i / 4]), then of kt by vectors of 4
    // columns (st[i * 32 + k / 4]).
    let mut st: Shared<[V4]> = Shared::new_slice(128usize * 17usize);
    // The state block and V' by pairs of columns: ss[k * 16 + tx], vs[i * 16 + tx].
    let mut ss: Shared<[V2]> = Shared::new_slice(256usize * 16usize);
    let mut vs: Shared<[V2]> = Shared::new_slice(64usize * 16usize);
    // Rows hf 128 + 8 ty + q of the state block, columns col0, col0 + 1:
    // sreg[(hf 8 + q) 2 + cc].
    let mut sreg = Array::<f32>::new(32usize);

    #[unroll]
    for hf in 0..2usize {
        #[unroll]
        for q in 0..8usize {
            let k = hf * HALF + 8usize * ty + q;
            let mut pair = V2::empty();
            #[unroll]
            for cc in 0..2usize {
                let col = col0 + cc;
                let mut x = f32::new(0.0f32);
                if col < dv && k < dk {
                    x = h0[(bh * dk + k) * dv + col];
                }
                sreg[(hf * 8usize + q) * 2usize + cc] = x;
                pair.insert(cc, x);
            }
            ss[k * 16usize + tx] = pair;
        }
    }
    sync_cube();

    for c in 0..n {
        let chunk = base + c;
        let off = chunk * 64usize * dk;
        #[unroll]
        for hf in 0..2usize {
            #[unroll]
            for q in 0..8usize {
                let k = hf * HALF + 8usize * ty + q;
                #[unroll]
                for cc in 0..2usize {
                    let col = col0 + cc;
                    if col < dv && k < dk {
                        states[(chunk * dk + k) * dv + col] = sreg[(hf * 8usize + q) * 2usize + cc];
                    }
                }
            }
        }
        let mut acc = Array::<f32>::new(8usize);
        #[unroll]
        for p in 0..4usize {
            #[unroll]
            for cc in 0..2usize {
                let col = col0 + cc;
                let mut x = f32::new(0.0f32);
                if col < dv {
                    x = u[(chunk * 64usize + 4usize * ty + p) * dv + col];
                }
                acc[p * 2usize + cc] = x;
            }
        }
        let decay = gl[chunk];

        // V' = U - W S, a half of Dk at a time.
        #[unroll]
        for hf in 0..2usize {
            #[unroll]
            for m in 0..8usize {
                let e = tid + 256usize * m;
                let kl = e % 128usize;
                let i4 = e / 128usize;
                let k = hf * HALF + kl;
                let mut wv = V4::empty();
                #[unroll]
                for r in 0..4usize {
                    let mut x = f32::new(0.0f32);
                    if k < dk {
                        x = w[off + (4usize * i4 + r) * dk + k];
                    }
                    wv.insert(r, x);
                }
                st[kl * 17usize + i4] = wv;
            }
            sync_cube();
            for kl in 0..128usize {
                let wv = st[kl * 17usize + ty];
                let sv = ss[(hf * HALF + kl) * 16usize + tx];
                #[unroll]
                for p in 0..4usize {
                    #[unroll]
                    for cc in 0..2usize {
                        acc[p * 2usize + cc] -= wv.extract(p) * sv.extract(cc);
                    }
                }
            }
            sync_cube();
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

        // S = exp(G_last) S + kt^T V', each half of the rows from its half of kt.
        #[unroll]
        for hf in 0..2usize {
            #[unroll]
            for m in 0..8usize {
                let e = tid + 256usize * m;
                let i = e / 32usize;
                let k4 = e % 32usize;
                let mut kv = V4::empty();
                #[unroll]
                for r in 0..4usize {
                    let k = hf * HALF + 4usize * k4 + r;
                    let mut x = f32::new(0.0f32);
                    if k < dk {
                        x = kt[off + i * dk + k];
                    }
                    kv.insert(r, x);
                }
                st[i * 32usize + k4] = kv;
            }
            sync_cube();
            let mut acc2 = Array::<f32>::new(16usize);
            #[unroll]
            for x in 0..16usize {
                acc2[x] = decay * sreg[hf * 16usize + x];
            }
            for i in 0..64usize {
                let vv = vs[i * 16usize + tx];
                let k0 = st[i * 32usize + 2usize * ty];
                let k1 = st[i * 32usize + 2usize * ty + 1usize];
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
                    sreg[hf * 16usize + q * 2usize + cc] = acc2[q * 2usize + cc];
                    pair.insert(cc, acc2[q * 2usize + cc]);
                }
                ss[(hf * HALF + 8usize * ty + q) * 16usize + tx] = pair;
            }
            sync_cube();
        }
    }

    #[unroll]
    for hf in 0..2usize {
        #[unroll]
        for q in 0..8usize {
            let k = hf * HALF + 8usize * ty + q;
            #[unroll]
            for cc in 0..2usize {
                let col = col0 + cc;
                if col < dv && k < dk {
                    last[(bh * dk + k) * dv + col] = sreg[(hf * 8usize + q) * 2usize + cc];
                }
            }
        }
    }
}

/// The state passing, backward, for `128 < Dk <= 256`, laid out as
/// [`state_forward_wide_kernel`]. For each chunk from the last: store `dS_{c+1}`;
/// `dV' = kt dS + dv_local`, staging `kt^T` a half at a time; then each half of the rows of
/// `dS = exp(G_last) dS + qg^T dO - W^T dV'`, staging that half of `qg`, then of `W`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub(super) fn state_backward_wide_kernel(
    kt: &Tensor<f32>,
    w: &Tensor<f32>,
    qg: &Tensor<f32>,
    d_o: &Tensor<f32>,
    dv_local: &Tensor<f32>,
    gl: &Tensor<f32>,
    d_last: &Tensor<f32>,
    offsets: &Tensor<u32>,
    ds_next: &mut Tensor<f32>,
    dvn: &mut Tensor<f32>,
    d_first: &mut Tensor<f32>,
    n_chunks: u32,
    heads: u32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] packed: bool,
) {
    let bh = CUBE_POS_Y as usize;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let col0 = CUBE_POS_X as usize * 32usize + 2usize * tx;
    let (base, n) = state_chunks(offsets, bh, heads as usize, n_chunks as usize, packed);
    // One half of kt^T by vectors of 4 rows (st[k * 17 + i / 4]), or of qg or W by vectors of 4
    // columns (st[i * 32 + k / 4]).
    let mut st: Shared<[V4]> = Shared::new_slice(128usize * 17usize);
    let mut dss: Shared<[V2]> = Shared::new_slice(256usize * 16usize);
    let mut dvs: Shared<[V2]> = Shared::new_slice(64usize * 16usize);
    let mut dos: Shared<[V2]> = Shared::new_slice(64usize * 16usize);
    let mut dsreg = Array::<f32>::new(32usize);

    #[unroll]
    for hf in 0..2usize {
        #[unroll]
        for q in 0..8usize {
            let k = hf * HALF + 8usize * ty + q;
            let mut pair = V2::empty();
            #[unroll]
            for cc in 0..2usize {
                let col = col0 + cc;
                let mut x = f32::new(0.0f32);
                if col < dv && k < dk {
                    x = d_last[(bh * dk + k) * dv + col];
                }
                dsreg[(hf * 8usize + q) * 2usize + cc] = x;
                pair.insert(cc, x);
            }
            dss[k * 16usize + tx] = pair;
        }
    }
    sync_cube();

    for cr in 0..n {
        let chunk = base + (n - 1usize - cr);
        let off = chunk * 64usize * dk;
        #[unroll]
        for hf in 0..2usize {
            #[unroll]
            for q in 0..8usize {
                let k = hf * HALF + 8usize * ty + q;
                #[unroll]
                for cc in 0..2usize {
                    let col = col0 + cc;
                    if col < dv && k < dk {
                        ds_next[(chunk * dk + k) * dv + col] =
                            dsreg[(hf * 8usize + q) * 2usize + cc];
                    }
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
                let col = col0 + cc;
                let mut x = f32::new(0.0f32);
                let mut y = f32::new(0.0f32);
                if col < dv {
                    let idx = (chunk * 64usize + i) * dv + col;
                    x = dv_local[idx];
                    y = d_o[idx];
                }
                acc[p * 2usize + cc] = x;
                pair.insert(cc, y);
            }
            dos[i * 16usize + tx] = pair;
        }
        let decay = gl[chunk];

        // dV' = kt dS + dv_local, a half of Dk at a time.
        #[unroll]
        for hf in 0..2usize {
            #[unroll]
            for m in 0..8usize {
                let e = tid + 256usize * m;
                let kl = e % 128usize;
                let i4 = e / 128usize;
                let k = hf * HALF + kl;
                let mut kv = V4::empty();
                #[unroll]
                for r in 0..4usize {
                    let mut x = f32::new(0.0f32);
                    if k < dk {
                        x = kt[off + (4usize * i4 + r) * dk + k];
                    }
                    kv.insert(r, x);
                }
                st[kl * 17usize + i4] = kv;
            }
            sync_cube();
            for kl in 0..128usize {
                let kv = st[kl * 17usize + ty];
                let sv = dss[(hf * HALF + kl) * 16usize + tx];
                #[unroll]
                for p in 0..4usize {
                    #[unroll]
                    for cc in 0..2usize {
                        acc[p * 2usize + cc] += kv.extract(p) * sv.extract(cc);
                    }
                }
            }
            sync_cube();
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

        // dS = exp(G_last) dS + qg^T dO - W^T dV', a half of the rows at a time.
        #[unroll]
        for hf in 0..2usize {
            let mut acc2 = Array::<f32>::new(16usize);
            #[unroll]
            for x in 0..16usize {
                acc2[x] = decay * dsreg[hf * 16usize + x];
            }
            // + qg^T dO, this half of qg staged by vectors of 4 columns.
            stage_columns(qg, &mut st, off, hf, dk);
            for i in 0..64usize {
                let yv = dos[i * 16usize + tx];
                #[unroll]
                for half in 0..2usize {
                    let xv = st[i * 32usize + 2usize * ty + half];
                    #[unroll]
                    for r in 0..4usize {
                        #[unroll]
                        for cc in 0..2usize {
                            acc2[(half * 4usize + r) * 2usize + cc] +=
                                xv.extract(r) * yv.extract(cc);
                        }
                    }
                }
            }
            sync_cube();
            // - W^T dV', this half of W staged likewise.
            stage_columns(w, &mut st, off, hf, dk);
            for i in 0..64usize {
                let yv = dvs[i * 16usize + tx];
                #[unroll]
                for half in 0..2usize {
                    let xv = st[i * 32usize + 2usize * ty + half];
                    #[unroll]
                    for r in 0..4usize {
                        #[unroll]
                        for cc in 0..2usize {
                            acc2[(half * 4usize + r) * 2usize + cc] -=
                                xv.extract(r) * yv.extract(cc);
                        }
                    }
                }
            }
            sync_cube();
            #[unroll]
            for q in 0..8usize {
                let mut pair = V2::empty();
                #[unroll]
                for cc in 0..2usize {
                    dsreg[hf * 16usize + q * 2usize + cc] = acc2[q * 2usize + cc];
                    pair.insert(cc, acc2[q * 2usize + cc]);
                }
                dss[(hf * HALF + 8usize * ty + q) * 16usize + tx] = pair;
            }
        }
        sync_cube();
    }

    #[unroll]
    for hf in 0..2usize {
        #[unroll]
        for q in 0..8usize {
            let k = hf * HALF + 8usize * ty + q;
            #[unroll]
            for cc in 0..2usize {
                let col = col0 + cc;
                if col < dv && k < dk {
                    d_first[(bh * dk + k) * dv + col] = dsreg[(hf * 8usize + q) * 2usize + cc];
                }
            }
        }
    }
}

/// Half `hf` of a chunk's `[64, Dk]` rows of `x` (from element `off`) into `st` by vectors of 4
/// columns, `st[i * 32 + k / 4]`, zero past `Dk`; then a barrier.
#[cube]
fn stage_columns(
    x: &Tensor<f32>,
    st: &mut Shared<[V4]>,
    off: usize,
    hf: usize,
    #[comptime] dk: usize,
) {
    let tid = UNIT_POS_X as usize;
    #[unroll]
    for m in 0..8usize {
        let e = tid + 256usize * m;
        let i = e / 32usize;
        let k4 = e % 32usize;
        let mut v = V4::empty();
        #[unroll]
        for r in 0..4usize {
            let k = hf * HALF + 4usize * k4 + r;
            let mut y = f32::new(0.0f32);
            if k < dk {
                y = x[off + i * dk + k];
            }
            v.insert(r, y);
        }
        st[i * 32usize + k4] = v;
    }
    sync_cube();
}

/// `acc = Q K^T` over the halves of `Dk`, the thread's 4 x 4 tile (rows `ty + 16 a`, columns
/// `tx + 16 b`), queries and keys staged a half at a time in `qs` and `ks`, scaled by `qn` and
/// `kn`. With `qg_out`, also writes `gamma Q` (`gamma = exp(G)`, from `gs`) to `qg`, the chunked
/// `[.., 64, Dk]` tensor, at chunk `cube`; without, `qg` is not touched (any tensor will do).
#[cube]
#[allow(clippy::too_many_arguments)]
fn scores<E: Float, F: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    qs: &mut Shared<[f32]>,
    ks: &mut Shared<[f32]>,
    qn: &Shared<[f32]>,
    kn: &Shared<[f32]>,
    gs: &Shared<[f32]>,
    qg: &mut Tensor<F>,
    acc: &mut Array<f32>,
    first: usize,
    count: usize,
    h: usize,
    heads: usize,
    cube: usize,
    #[comptime] dk: usize,
    #[comptime] qg_out: bool,
) {
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    #[unroll]
    for x in 0..16usize {
        acc[x] = f32::new(0.0f32);
    }
    #[unroll]
    for hf in 0..comptime![dk.div_ceil(HALF)] {
        load_cols::<E>(q, qs, qn, first, count, h, heads, hf * HALF, dk);
        load_cols::<E>(k, ks, kn, first, count, h, heads, hf * HALF, dk);
        if qg_out {
            #[unroll]
            for m_ in 0..comptime![64usize * HALF / 256usize] {
                let e = tid + 256usize * m_;
                let i = e / HALF;
                let d = hf * HALF + e % HALF;
                if d < dk {
                    qg[(cube * 64usize + i) * dk + d] =
                        F::cast_from(qs[i * (HALF + 1usize) + e % HALF] * gs[i].exp());
                }
            }
        }
        for d in 0..HALF {
            let mut ai = Array::<f32>::new(4usize);
            let mut bj = Array::<f32>::new(4usize);
            #[unroll]
            for a in 0..4usize {
                ai[a] = qs[(ty + 16usize * a) * (HALF + 1usize) + d];
                bj[a] = ks[(tx + 16usize * a) * (HALF + 1usize) + d];
            }
            #[unroll]
            for a in 0..4usize {
                #[unroll]
                for bb in 0..4usize {
                    acc[a * 4usize + bb] += ai[a] * bj[bb];
                }
            }
        }
        sync_cube();
    }
}

/// [`super::output_kernel`] for `128 < Dk <= 256`: `P` over the halves of `Dk`, then
/// `(gamma Q) S` a half of `Q` (and of the state's rows) at a time, then `P V'`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub(super) fn output_wide_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    gcum: &Tensor<f32>,
    states: &Tensor<f32>,
    v_new: &Tensor<f32>,
    out: &mut Tensor<E>,
    plan: &Tensor<u32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    scale: f32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] l2norm: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let nbv = comptime![dv.div_ceil(16usize)];
    let cube = CUBE_POS_X as usize;
    let n = n_chunks as usize;
    let hh = heads as usize;
    let t_len = seq_len as usize;
    let c = cube % n;
    let bh = cube / n;
    let b = bh / hh;
    let h = bh % hh;
    let (first, count) = chunk_tokens(plan, b, c, t_len, packed);
    if packed && count == 0usize {
        terminate!();
    }
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;

    // ks holds a half of the keys, then 16 rows of the state or of V' at a time.
    let ksz = comptime![if 64usize * (HALF + 1usize) > 16usize * dv {
        64usize * (HALF + 1usize)
    } else {
        16usize * dv
    }];
    let mut qs = Shared::<[f32]>::new_slice(64usize * (HALF + 1usize));
    let mut ks = Shared::<[f32]>::new_slice(ksz);
    let mut pm = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut qn = Shared::<[f32]>::new_slice(64usize);
    let mut kn = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        gs[tid] = gcum[cube * 64usize + tid];
    }
    row_scales::<E>(q, &mut qn, &mut pm, first, count, h, hh, scale, dk, l2norm);
    row_scales::<E>(
        k,
        &mut kn,
        &mut pm,
        first,
        count,
        h,
        hh,
        f32::new(1.0f32),
        dk,
        l2norm,
    );
    let mut acc = Array::<f32>::new(16usize);
    // No `gamma Q` to write: `out` stands in for the unused tensor.
    scores::<E, E>(
        q, k, &mut qs, &mut ks, &qn, &kn, &gs, out, &mut acc, first, count, h, hh, cube, dk, false,
    );
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

    let mut o = Array::<f32>::new(4usize * nbv);
    #[unroll]
    for x in 0..4usize * nbv {
        o[x] = f32::new(0.0f32);
    }
    let mut gam = Array::<f32>::new(4usize);
    #[unroll]
    for a in 0..4usize {
        gam[a] = gs[ty + 16usize * a].exp();
    }
    // (gamma Q) S: each half of Q against the same half of the state's rows, streamed through
    // `ks` 16 rows at a time.
    #[unroll]
    for hf in 0..comptime![dk.div_ceil(HALF)] {
        load_cols::<E>(q, &mut qs, &qn, first, count, h, hh, hf * HALF, dk);
        for d0 in 0..comptime![HALF / 16usize] {
            #[unroll]
            for m_ in 0..comptime![(16usize * dv).div_ceil(256usize)] {
                let e = tid + 256usize * m_;
                if e < 16usize * dv {
                    let d = hf * HALF + d0 * 16usize + e / dv;
                    let mut x = f32::new(0.0f32);
                    if d < dk {
                        x = states[(cube * dk + d) * dv + e % dv];
                    }
                    ks[e] = x;
                }
            }
            sync_cube();
            for dd in 0..16usize {
                let dl = d0 * 16usize + dd;
                let mut qv = Array::<f32>::new(4usize);
                #[unroll]
                for a in 0..4usize {
                    qv[a] = qs[(ty + 16usize * a) * (HALF + 1usize) + dl] * gam[a];
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
    }
    // P V', V' streamed through `ks` 16 rows at a time.
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
        let i = ty + 16usize * a;
        #[unroll]
        for bb in 0..nbv {
            let col = tx + 16usize * bb;
            if i < count && col < dv {
                out[((first + i) * hh + h) * dv + col] = E::cast_from(o[a * nbv + bb]);
            }
        }
    }
}

/// [`super::dvlocal_kernel`] for `128 < Dk <= 256`: `P` over the halves of `Dk`, writing
/// `qg = gamma Q` a half at a time on the way, then `P^T dO`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub(super) fn dvlocal_wide_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    gcum: &Tensor<f32>,
    d_out: &Tensor<f32>,
    dv_local: &mut Tensor<f32>,
    qg: &mut Tensor<f32>,
    d_o: &mut Tensor<f32>,
    plan: &Tensor<u32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    scale: f32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] l2norm: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let nbv = comptime![dv.div_ceil(16usize)];
    let cube = CUBE_POS_X as usize;
    let n = n_chunks as usize;
    let hh = heads as usize;
    let t_len = seq_len as usize;
    let c = cube % n;
    let bh = cube / n;
    let b = bh / hh;
    let h = bh % hh;
    let (first, count) = chunk_tokens(plan, b, c, t_len, packed);
    if packed && count == 0usize {
        terminate!();
    }
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let ksz = comptime![if 64usize * (HALF + 1usize) > 16usize * dv {
        64usize * (HALF + 1usize)
    } else {
        16usize * dv
    }];
    let mut qs = Shared::<[f32]>::new_slice(64usize * (HALF + 1usize));
    let mut ks = Shared::<[f32]>::new_slice(ksz);
    let mut pm = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut qn = Shared::<[f32]>::new_slice(64usize);
    let mut kn = Shared::<[f32]>::new_slice(64usize);

    if tid < 64usize {
        gs[tid] = gcum[cube * 64usize + tid];
    }
    row_scales::<E>(q, &mut qn, &mut pm, first, count, h, hh, scale, dk, l2norm);
    row_scales::<E>(
        k,
        &mut kn,
        &mut pm,
        first,
        count,
        h,
        hh,
        f32::new(1.0f32),
        dk,
        l2norm,
    );
    let mut acc = Array::<f32>::new(16usize);
    scores::<E, f32>(
        q, k, &mut qs, &mut ks, &qn, &kn, &gs, qg, &mut acc, first, count, h, hh, cube, dk, true,
    );
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

    // dv_local = P^T dO, dO streamed through `ks` 16 rows at a time (and written in the chunked
    // layout on the way).
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
                let mut x = f32::new(0.0f32);
                if i < count {
                    x = d_out[((first + i) * hh + h) * dv + col];
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

/// [`super::dstate_kernel`] for `128 < Dk <= 256`: the same products over `Dv`, forming the
/// `Dk` columns of `dQg`, `dkt` and `dKbg` a half at a time.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub(super) fn dstate_wide_kernel(
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
    let nvb = comptime![dv.div_ceil(32usize)];
    let cube = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let mut so: Shared<[V4]> = Shared::new_slice(64usize * 9usize);
    let mut sv: Shared<[V4]> = Shared::new_slice(64usize * 9usize);
    let mut sd: Shared<[V4]> = Shared::new_slice(64usize * 9usize);
    let mut ss: Shared<[V4]> = Shared::new_slice(HALF * 9usize);
    let mut sz: Shared<[V4]> = Shared::new_slice(HALF * 9usize);
    let mut red = Shared::<[f32]>::new_slice(256usize);
    let mut part = f32::new(0.0f32);
    #[unroll]
    for hf in 0..comptime![dk.div_ceil(HALF)] {
        let mut acc1 = Array::<f32>::new(32usize);
        let mut acc2 = Array::<f32>::new(32usize);
        let mut acc3 = Array::<f32>::new(32usize);
        #[unroll]
        for x in 0..32usize {
            acc1[x] = f32::new(0.0f32);
            acc2[x] = f32::new(0.0f32);
            acc3[x] = f32::new(0.0f32);
        }
        for vb in 0..nvb {
            // 64 rows x 8 vectors of dO, V', dV_b; 128 rows x 8 vectors of S, dS.
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
            for m in 0..4usize {
                let e = tid + 256usize * m;
                let kl = e / 8usize;
                let v4 = e % 8usize;
                let kk = hf * HALF + kl;
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
                ss[kl * 9usize + v4] = x;
                sz[kl * 9usize + v4] = y;
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
                for bb in 0..8usize {
                    let row = (tx + 16usize * bb) * 9usize + v4;
                    let s1 = ss[row];
                    let s2 = sz[row];
                    #[unroll]
                    for r in 0..4usize {
                        #[unroll]
                        for a in 0..4usize {
                            acc1[a * 8usize + bb] += o[a].extract(r) * s1.extract(r);
                            acc2[a * 8usize + bb] += x[a].extract(r) * s2.extract(r);
                            acc3[a * 8usize + bb] -= d[a].extract(r) * s1.extract(r);
                        }
                    }
                }
            }
            sync_cube();
        }
        #[unroll]
        for a in 0..4usize {
            #[unroll]
            for bb in 0..8usize {
                let kk = hf * HALF + tx + 16usize * bb;
                if kk < dk {
                    let idx = (cube * 64usize + ty + 16usize * a) * dk + kk;
                    dqg[idx] = acc1[a * 8usize + bb];
                    dkt[idx] = acc2[a * 8usize + bb];
                    dkbg[idx] = acc3[a * 8usize + bb];
                }
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

/// [`super::dattn_kernel`] for `128 < Dk <= 256`: `dP = dO V'^T` as there, `P` over the halves
/// of `Dk`, then `dQ = dPe K` and `dK = dPe^T Q` a half of `Dk` at a time.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub(super) fn dattn_wide_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    gcum: &Tensor<f32>,
    d_o: &Tensor<f32>,
    v_new: &Tensor<f32>,
    dq_attn: &mut Tensor<f32>,
    dk_attn: &mut Tensor<f32>,
    dg_p: &mut Tensor<f32>,
    plan: &Tensor<u32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    scale: f32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] l2norm: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
    let nvb = comptime![dv.div_ceil(32usize)];
    let cube = CUBE_POS_X as usize;
    let n = n_chunks as usize;
    let hh = heads as usize;
    let t_len = seq_len as usize;
    let c = cube % n;
    let bh = cube / n;
    let b = bh / hh;
    let h = bh % hh;
    let (first, count) = chunk_tokens(plan, b, c, t_len, packed);
    if packed && count == 0usize {
        terminate!();
    }
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    // qs holds the dO and V' blocks (64 x 33 each), then a half of the queries, then dP P.
    let mut qs = Shared::<[f32]>::new_slice(64usize * (HALF + 1usize));
    let mut ks = Shared::<[f32]>::new_slice(64usize * (HALF + 1usize));
    let mut dpm = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut qn = Shared::<[f32]>::new_slice(64usize);
    let mut kn = Shared::<[f32]>::new_slice(64usize);

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
                qs[i * 33usize + vv] = x;
                qs[2112usize + i * 33usize + vv] = y;
            }
        }
        sync_cube();
        for vv in 0..32usize {
            let mut o = Array::<f32>::new(4usize);
            let mut x = Array::<f32>::new(4usize);
            #[unroll]
            for a in 0..4usize {
                o[a] = qs[(ty + 16usize * a) * 33usize + vv];
                x[a] = qs[2112usize + (tx + 16usize * a) * 33usize + vv];
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
    row_scales::<E>(q, &mut qn, &mut ks, first, count, h, hh, scale, dk, l2norm);
    row_scales::<E>(
        k,
        &mut kn,
        &mut ks,
        first,
        count,
        h,
        hh,
        f32::new(1.0f32),
        dk,
        l2norm,
    );
    // No `gamma Q` to write: `dq_attn` stands in for the unused tensor.
    scores::<E, f32>(
        q, k, &mut qs, &mut ks, &qn, &kn, &gs, dq_attn, &mut acc, first, count, h, hh, cube, dk,
        false,
    );
    // dP P exp(G_i - G_j) into `qs` for the gate's gradient; dPe = dP exp(G_i - G_j).
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
            qs[i * 65usize + j] = m;
        }
    }
    sync_cube();
    if tid < 64usize {
        let mut rs = f32::new(0.0f32);
        let mut cs = f32::new(0.0f32);
        for x in 0..64usize {
            rs += qs[tid * 65usize + x];
            cs += qs[x * 65usize + tid];
        }
        dg_p[cube * 64usize + tid] = rs - cs;
    }
    sync_cube();
    #[unroll]
    for hf in 0..comptime![dk.div_ceil(HALF)] {
        load_cols::<E>(q, &mut qs, &qn, first, count, h, hh, hf * HALF, dk);
        load_cols::<E>(k, &mut ks, &kn, first, count, h, hh, hf * HALF, dk);
        let mut accq = Array::<f32>::new(32usize);
        let mut acck = Array::<f32>::new(32usize);
        #[unroll]
        for x in 0..32usize {
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
            for bb in 0..8usize {
                let kl = tx + 16usize * bb;
                let kv = ks[j * (HALF + 1usize) + kl];
                let qv = qs[j * (HALF + 1usize) + kl];
                #[unroll]
                for a in 0..4usize {
                    accq[a * 8usize + bb] += pe[a] * kv;
                    acck[a * 8usize + bb] += pt[a] * qv;
                }
            }
        }
        #[unroll]
        for a in 0..4usize {
            #[unroll]
            for bb in 0..8usize {
                let kk = hf * HALF + tx + 16usize * bb;
                if kk < dk {
                    let idx = (cube * 64usize + ty + 16usize * a) * dk + kk;
                    dq_attn[idx] = accq[a * 8usize + bb];
                    dk_attn[idx] = acck[a * 8usize + bb];
                }
            }
        }
        sync_cube();
    }
}

/// [`super::dwy_kernel`] for `128 < Dk <= 256`: `dA` as there, then the keys' gradient through
/// `A` a half of `Dk` at a time, and `beta`'s summed over both halves.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub(super) fn dwy_wide_kernel<E: Float>(
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
    plan: &Tensor<u32>,
    seq_len: u32,
    n_chunks: u32,
    heads: u32,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] l2norm: bool,
    #[comptime] packed: bool,
    #[define(E)] _dtype: ElemType,
) {
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
    let (first, count) = chunk_tokens(plan, b, c, t_len, packed);
    if packed && count == 0usize {
        terminate!();
    }
    let tid = UNIT_POS_X as usize;
    let tx = tid % 16usize;
    let ty = tid / 16usize;
    let mut blk = Shared::<[f32]>::new_slice(4224usize);
    let mut dam = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut mam = Shared::<[f32]>::new_slice(64usize * 65usize);
    let mut ks = Shared::<[f32]>::new_slice(64usize * (HALF + 1usize));
    let mut gs = Shared::<[f32]>::new_slice(64usize);
    let mut bs = Shared::<[f32]>::new_slice(64usize);
    let mut kn = Shared::<[f32]>::new_slice(64usize);
    let mut red = Shared::<[f32]>::new_slice(1024usize);

    if tid < 64usize {
        gs[tid] = gcum[cube * 64usize + tid];
        let mut x = f32::new(0.0f32);
        if tid < count {
            x = beta[(first + tid) * hh + h];
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
    row_scales::<E>(
        k,
        &mut kn,
        &mut blk,
        first,
        count,
        h,
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
    let mut part = Array::<f32>::new(4usize);
    #[unroll]
    for a in 0..4usize {
        part[a] = f32::new(0.0f32);
    }
    #[unroll]
    for hf in 0..comptime![dk.div_ceil(HALF)] {
        load_cols::<E>(k, &mut ks, &kn, first, count, h, hh, hf * HALF, dk);
        let mut acc1 = Array::<f32>::new(32usize);
        let mut acc2 = Array::<f32>::new(32usize);
        #[unroll]
        for x in 0..32usize {
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
            for bb in 0..8usize {
                let kv = ks[j * (HALF + 1usize) + tx + 16usize * bb];
                #[unroll]
                for a in 0..4usize {
                    acc1[a * 8usize + bb] += pe[a] * kv;
                    acc2[a * 8usize + bb] += pt[a] * kv;
                }
            }
        }
        #[unroll]
        for a in 0..4usize {
            let r = ty + 16usize * a;
            #[unroll]
            for bb in 0..8usize {
                let kl = tx + 16usize * bb;
                let kk = hf * HALF + kl;
                if kk < dk {
                    dk_a[(cube * 64usize + r) * dk + kk] =
                        acc1[a * 8usize + bb] * bs[r] + acc2[a * 8usize + bb];
                    part[a] += acc1[a * 8usize + bb] * ks[r * (HALF + 1usize) + kl];
                }
            }
        }
        sync_cube();
    }
    #[unroll]
    for a in 0..4usize {
        red[(ty + 16usize * a) * 16usize + tx] = part[a];
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
