//! The chunked gated delta rule built from Burn's tensor operations: the forward pass, and the
//! backward pass written out from the forward's chunked form. Every backend can run it; the
//! tests hold the CUDA kernels to it.
//!
//! Work within a chunk is batched over all chunks of all heads (`[B, H, N, C, ...]` tensors, `N`
//! chunks of `C = 64` tokens); only the passing of the state from chunk to chunk, forward and
//! backward, is a loop. Everything is computed in f32, whatever the inputs' dtype.
//!
//! Notation, per chunk (`Q` already scaled by `1/sqrt(Dk)`):
//!
//! - `G = cumsum(g)` within the chunk, `gamma = exp(G)`, `lam = exp(G_last - G)`;
//! - `A[i, j] = beta_i (k_i . k_j) exp(G_i - G_j)` for `i > j`, else 0; `T = (I + A)^-1`;
//! - `W = T (beta gamma K)`, `U = T (beta V)` (row-scaled);
//! - with `S` the state entering the chunk: `V' = U - W S`;
//! - `O = (gamma Q) S + P V'` with `P[i, j] = (q_i . k_j) exp(G_i - G_j)` for `i >= j`;
//! - the state leaving the chunk: `exp(G_last) S + (lam K)^T V'`.

use burn::backend::Backend;
use burn::backend::TensorMetadata;
use burn::backend::tensor::FloatTensor;
use burn::tensor::{DType, FloatDType, Shape, Slice, TensorData};

/// Tokens per chunk, as in transformers' `torch_chunk_gated_delta_rule` and fla.
pub const CHUNK: usize = 64;

/// The epsilon of transformers' (and fla's) `l2norm`: `x / sqrt(sum(x^2) + eps)`.
const L2NORM_EPS: f32 = 1e-6;

type Ft<B> = FloatTensor<B>;

/// Sizes of one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dims {
    pub b: usize,
    pub t: usize,
    pub h: usize,
    pub dk: usize,
    pub dv: usize,
    /// Chunks per sequence, `ceil(t / CHUNK)`.
    pub n: usize,
}

impl Dims {
    fn of(q: &Shape, v: &Shape) -> Self {
        let [b, t, h, dk] = q.dims::<4>();
        let dv = v.dims::<4>()[3];
        Self {
            b,
            t,
            h,
            dk,
            dv,
            n: t.div_ceil(CHUNK),
        }
    }

    /// The padded sequence length.
    fn tp(&self) -> usize {
        self.n * CHUNK
    }

    fn chunked(&self, depth: usize) -> Shape {
        Shape::new([self.b, self.h, self.n, CHUNK, depth])
    }

    fn per_token(&self) -> Shape {
        Shape::new([self.b, self.h, self.n, CHUNK])
    }

    fn square(&self) -> Shape {
        Shape::new([self.b, self.h, self.n, CHUNK, CHUNK])
    }
}

pub(crate) fn to_f32<B: Backend>(x: Ft<B>) -> Ft<B> {
    if x.dtype() == DType::F32 {
        x
    } else {
        B::float_cast(x, FloatDType::F32)
    }
}

pub(crate) fn to_dtype<B: Backend>(x: Ft<B>, dtype: DType) -> Ft<B> {
    if x.dtype() == dtype {
        x
    } else {
        B::float_cast(x, dtype.into())
    }
}

fn full() -> Slice {
    Slice::from(..)
}

fn mul<B: Backend>(a: Ft<B>, b: Ft<B>) -> Ft<B> {
    B::float_mul(a, b)
}

fn add<B: Backend>(a: Ft<B>, b: Ft<B>) -> Ft<B> {
    B::float_add(a, b)
}

fn sub<B: Backend>(a: Ft<B>, b: Ft<B>) -> Ft<B> {
    B::float_sub(a, b)
}

fn mm<B: Backend>(a: Ft<B>, b: Ft<B>) -> Ft<B> {
    crate::reference::matmul::<B>(a, b)
}

fn tr<B: Backend>(a: Ft<B>) -> Ft<B> {
    B::float_transpose(a)
}

/// `[B, H, N, C] -> [B, H, N, C, depth]`, each token's value repeated along the last axis.
fn col<B: Backend>(x: Ft<B>, depth: usize) -> Ft<B> {
    let [b, h, n, c] = x.shape().dims::<4>();
    B::float_expand(
        B::float_reshape(x, Shape::new([b, h, n, c, 1])),
        Shape::new([b, h, n, c, depth]),
    )
}

/// `[B, H, N, C] -> [B, H, N, C, C]`, `out[.., i, j] = x[.., j]`.
fn row<B: Backend>(x: Ft<B>) -> Ft<B> {
    let [b, h, n, c] = x.shape().dims::<4>();
    B::float_expand(
        B::float_reshape(x, Shape::new([b, h, n, 1, c])),
        Shape::new([b, h, n, c, c]),
    )
}

/// Sum over the last axis: `[.., D] -> [..]`.
fn sum_last<B: Backend>(x: Ft<B>) -> Ft<B> {
    let mut shape = x.shape();
    let rank = shape.num_dims();
    let summed = B::float_sum_dim(x, rank - 1);
    shape.remove(rank - 1);
    B::float_reshape(summed, shape)
}

/// Sum over the second-to-last axis of a `[.., C, C]` tensor: `out[.., j] = sum_i x[.., i, j]`.
fn sum_rows<B: Backend>(x: Ft<B>) -> Ft<B> {
    let mut shape = x.shape();
    let rank = shape.num_dims();
    let summed = B::float_sum_dim(x, rank - 2);
    shape.remove(rank - 2);
    B::float_reshape(summed, shape)
}

/// A `[C, C]` 0/1 mask, `1` where `i > j` (strict) or `i >= j`, expanded to `shape`.
fn lower_mask<B: Backend>(strict: bool, shape: Shape, device: &B::Device) -> Ft<B> {
    let mut m = vec![0f32; CHUNK * CHUNK];
    for i in 0..CHUNK {
        for j in 0..CHUNK {
            if i > j || (!strict && i == j) {
                m[i * CHUNK + j] = 1.0;
            }
        }
    }
    let m = B::float_from_data(
        TensorData::new(m, Shape::new([1, 1, 1, CHUNK, CHUNK])),
        device,
    );
    B::float_expand(m, shape)
}

fn identity<B: Backend>(shape: Shape, device: &B::Device) -> Ft<B> {
    let mut m = vec![0f32; CHUNK * CHUNK];
    for i in 0..CHUNK {
        m[i * CHUNK + i] = 1.0;
    }
    let m = B::float_from_data(
        TensorData::new(m, Shape::new([1, 1, 1, CHUNK, CHUNK])),
        device,
    );
    B::float_expand(m, shape)
}

/// `[B, T, H, D] -> [B, H, N, C, D]`, zero-padded along T.
fn chunk4<B: Backend>(x: Ft<B>, d: &Dims, depth: usize) -> Ft<B> {
    let x = B::float_swap_dims(x, 1, 2);
    let x = if d.tp() == d.t {
        x
    } else {
        let z = B::float_zeros(
            Shape::new([d.b, d.h, d.tp(), depth]),
            &x.device(),
            FloatDType::F32,
        );
        B::float_slice_assign(z, &[full(), full(), Slice::from(0..d.t), full()], x)
    };
    B::float_reshape(x, d.chunked(depth))
}

/// `[B, T, H] -> [B, H, N, C]`, zero-padded along T.
fn chunk3<B: Backend>(x: Ft<B>, d: &Dims) -> Ft<B> {
    let x = B::float_swap_dims(x, 1, 2);
    let x = if d.tp() == d.t {
        x
    } else {
        let z = B::float_zeros(Shape::new([d.b, d.h, d.tp()]), &x.device(), FloatDType::F32);
        B::float_slice_assign(z, &[full(), full(), Slice::from(0..d.t)], x)
    };
    B::float_reshape(x, d.per_token())
}

/// `[B, H, N, C, D] -> [B, T, H, D]`, dropping the padding.
fn unchunk4<B: Backend>(x: Ft<B>, d: &Dims, depth: usize) -> Ft<B> {
    let x = B::float_reshape(x, Shape::new([d.b, d.h, d.tp(), depth]));
    let x = if d.tp() == d.t {
        x
    } else {
        B::float_slice(x, &[full(), full(), Slice::from(0..d.t), full()])
    };
    B::float_swap_dims(x, 1, 2)
}

/// `[B, H, N, C] -> [B, T, H]`, dropping the padding.
fn unchunk3<B: Backend>(x: Ft<B>, d: &Dims) -> Ft<B> {
    let x = B::float_reshape(x, Shape::new([d.b, d.h, d.tp()]));
    let x = if d.tp() == d.t {
        x
    } else {
        B::float_slice(x, &[full(), full(), Slice::from(0..d.t)])
    };
    B::float_swap_dims(x, 1, 2)
}

/// transformers' `l2norm` over the last axis of a `[B, T, H, D]` f32 tensor: the normalised
/// tensor and `1 / sqrt(sum(x^2) + eps)`, `[B, T, H, 1]`.
fn l2norm<B: Backend>(x: Ft<B>) -> (Ft<B>, Ft<B>) {
    let shape = x.shape();
    let sq = B::float_sum_dim(mul::<B>(x.clone(), x.clone()), 3);
    let inv = B::float_recip(B::float_sqrt(B::float_add_scalar(sq, L2NORM_EPS.into())));
    let xn = mul::<B>(x, B::float_expand(inv.clone(), shape));
    (xn, inv)
}

/// The backward of [`l2norm`]: `dx = inv (dxn - xn sum(xn dxn))`.
fn l2norm_backward<B: Backend>(xn: Ft<B>, inv: Ft<B>, dxn: Ft<B>) -> Ft<B> {
    let shape = xn.shape();
    let s = B::float_sum_dim(mul::<B>(xn.clone(), dxn.clone()), 3);
    let s = B::float_expand(s, shape.clone());
    mul::<B>(B::float_expand(inv, shape), sub::<B>(dxn, mul::<B>(xn, s)))
}

/// The inputs in chunked f32 form.
struct Prepared<B: Backend> {
    pub d: Dims,
    /// `[B, H, N, C, Dk]`, normalised if asked, scaled by `1/sqrt(Dk)`.
    pub q: Ft<B>,
    /// `[B, H, N, C, Dk]`, normalised if asked.
    pub k: Ft<B>,
    /// `[B, H, N, C, Dv]`.
    pub v: Ft<B>,
    /// `[B, H, N, C]`.
    pub g: Ft<B>,
    /// `[B, H, N, C]`.
    pub beta: Ft<B>,
    /// With the in-kernel l2norm, the normalised queries and keys `[B, T, H, Dk]` and their
    /// inverse norms `[B, T, H, 1]`, for the backward pass.
    pub norms: Option<[Ft<B>; 4]>,
}

pub(crate) fn scale(dk: usize) -> f32 {
    1.0 / (dk as f32).sqrt()
}

fn prepare<B: Backend>(
    q: Ft<B>,
    k: Ft<B>,
    v: Ft<B>,
    g: Ft<B>,
    beta: Ft<B>,
    use_qk_l2norm: bool,
) -> Prepared<B> {
    let d = Dims::of(&q.shape(), &v.shape());
    let (q, k) = (to_f32::<B>(q), to_f32::<B>(k));
    let (q, k, norms) = if use_qk_l2norm {
        let (qn, qi) = l2norm::<B>(q);
        let (kn, ki) = l2norm::<B>(k);
        (qn.clone(), kn.clone(), Some([qn, qi, kn, ki]))
    } else {
        (q, k, None)
    };
    let q = B::float_mul_scalar(q, scale(d.dk).into());
    Prepared {
        q: chunk4::<B>(q, &d, d.dk),
        k: chunk4::<B>(k, &d, d.dk),
        v: chunk4::<B>(to_f32::<B>(v), &d, d.dv),
        g: chunk3::<B>(to_f32::<B>(g), &d),
        beta: chunk3::<B>(to_f32::<B>(beta), &d),
        norms,
        d,
    }
}

/// Chunk `c` of a `[B, H, N, R, D]` tensor, as `[B, H, R, D]`.
fn at<B: Backend>(x: &Ft<B>, c: usize) -> Ft<B> {
    let [b, h, _, rows, depth] = x.shape().dims::<5>();
    B::float_reshape(
        B::float_slice(
            x.clone(),
            &[full(), full(), Slice::from(c..c + 1), full(), full()],
        ),
        Shape::new([b, h, rows, depth]),
    )
}

/// `exp(G_last)` of chunk `c`, `[B, H, N, 1]`, expanded to `shape`.
fn state_decay_at<B: Backend>(gl: &Ft<B>, c: usize, shape: Shape) -> Ft<B> {
    let [b, h, _, _] = shape.dims::<4>();
    let x = B::float_slice(gl.clone(), &[full(), full(), Slice::from(c..c + 1), full()]);
    B::float_expand(B::float_reshape(x, Shape::new([b, h, 1, 1])), shape)
}

fn stack<B: Backend>(parts: Vec<Ft<B>>) -> Ft<B> {
    let parts = parts
        .into_iter()
        .map(|x| {
            let [b, h, r, c] = x.shape().dims::<4>();
            B::float_reshape(x, Shape::new([b, h, 1, r, c]))
        })
        .collect();
    B::float_cat(parts, 2)
}

/// `(I + A)^-1` for strictly lower triangular `A` `[B, H, N, C, C]`, by forward substitution one
/// row at a time, each step batched over all chunks, as transformers does.
fn unit_lower_inverse<B: Backend>(a: Ft<B>) -> Ft<B> {
    let shape = a.shape();
    let device = a.device();
    let mut m = B::float_neg(a);
    for i in 1..CHUNK {
        let rows = Slice::from(i..i + 1);
        let cols = Slice::from(0..i);
        let row = B::float_slice(m.clone(), &[full(), full(), full(), rows, cols]);
        let sub = B::float_slice(m.clone(), &[full(), full(), full(), cols, cols]);
        let new = add::<B>(row.clone(), mm::<B>(row, sub));
        m = B::float_slice_assign(m, &[full(), full(), full(), rows, cols], new);
    }
    add::<B>(m, identity::<B>(shape, &device))
}

/// The state passing, forward, from `kt = lam K` `[B, H, N, C, Dk]`, `W`, `U`,
/// `gl = exp(G_last)` `[B, H, N, 1]` and the initial state `[B, H, Dk, Dv]`: the state entering
/// each chunk `[B, H, N, Dk, Dv]`, `V'` `[B, H, N, C, Dv]`, and the final state.
fn state_forward<B: Backend>(
    kt: Ft<B>,
    w: Ft<B>,
    u: Ft<B>,
    gl: Ft<B>,
    initial_state: Ft<B>,
) -> (Ft<B>, Ft<B>, Ft<B>) {
    let n = w.shape()[2];
    let mut s = initial_state;
    let mut states = Vec::with_capacity(n);
    let mut v_new = Vec::with_capacity(n);
    for c in 0..n {
        states.push(s.clone());
        let vn = sub::<B>(at::<B>(&u, c), mm::<B>(at::<B>(&w, c), s.clone()));
        let decay = state_decay_at::<B>(&gl, c, s.shape());
        s = add::<B>(
            mul::<B>(s, decay),
            mm::<B>(tr::<B>(at::<B>(&kt, c)), vn.clone()),
        );
        v_new.push(vn);
    }
    (stack::<B>(states), stack::<B>(v_new), s)
}

/// The state passing, backward, from `kt`, `W`, `qg = gamma Q` `[B, H, N, C, Dk]`, `dO`,
/// `dv_local = P^T dO` `[B, H, N, C, Dv]`, `gl` and the final state's gradient: the gradient of
/// the state leaving each chunk `[B, H, N, Dk, Dv]`, `dV'` `[B, H, N, C, Dv]`, and the initial
/// state's gradient:
///
/// ```text
/// dV'_c = kt_c dS_{c+1} + dv_local_c
/// dS_c  = exp(G_last) dS_{c+1} + qg_c^T dO_c - W_c^T dV'_c
/// ```
#[allow(clippy::too_many_arguments)]
fn state_backward<B: Backend>(
    kt: Ft<B>,
    w: Ft<B>,
    qg: Ft<B>,
    d_o: Ft<B>,
    dv_local: Ft<B>,
    gl: Ft<B>,
    d_final_state: Ft<B>,
) -> (Ft<B>, Ft<B>, Ft<B>) {
    let n = w.shape()[2];
    let qg_do = mm::<B>(tr::<B>(qg), d_o);
    let mut ds = d_final_state;
    let mut ds_next = Vec::with_capacity(n);
    let mut dvn = Vec::with_capacity(n);
    for c in (0..n).rev() {
        ds_next.push(ds.clone());
        let dvn_c = add::<B>(mm::<B>(at::<B>(&kt, c), ds.clone()), at::<B>(&dv_local, c));
        let decay = state_decay_at::<B>(&gl, c, ds.shape());
        ds = sub::<B>(
            add::<B>(mul::<B>(ds, decay), at::<B>(&qg_do, c)),
            mm::<B>(tr::<B>(at::<B>(&w, c)), dvn_c.clone()),
        );
        dvn.push(dvn_c);
    }
    ds_next.reverse();
    dvn.reverse();
    (stack::<B>(ds_next), stack::<B>(dvn), ds)
}

/// The work within each chunk that does not depend on the state.
struct Intra<B: Backend> {
    /// `G`, `[B, H, N, C]`.
    pub gcum: Ft<B>,
    /// `exp(G)`, `[B, H, N, C]`.
    pub gamma: Ft<B>,
    /// `exp(G_i - G_j)` for `i >= j`, else 0, `[B, H, N, C, C]`.
    pub decay: Ft<B>,
    /// `A`, `[B, H, N, C, C]`.
    pub a: Ft<B>,
    /// `T = (I + A)^-1`, `[B, H, N, C, C]`.
    pub t: Ft<B>,
    /// `W = T (beta gamma K)`, `[B, H, N, C, Dk]`.
    pub w: Ft<B>,
    /// `U = T (beta V)`, `[B, H, N, C, Dv]`.
    pub u: Ft<B>,
}

fn intra<B: Backend>(p: &Prepared<B>) -> Intra<B> {
    let d = &p.d;
    let device = p.q.device();
    let sq = d.square();
    let gcum = B::float_cumsum(p.g.clone(), 3);
    let gamma = B::float_exp(gcum.clone());
    let incl = lower_mask::<B>(false, sq.clone(), &device);
    let strict = lower_mask::<B>(true, sq.clone(), &device);
    // Masked before the exponential: above the diagonal G_i - G_j > 0 and could overflow.
    let gi = col::<B>(gcum.clone(), CHUNK);
    let diff = mul::<B>(sub::<B>(gi, row::<B>(gcum.clone())), incl.clone());
    let decay = mul::<B>(B::float_exp(diff), incl);
    let kbeta = mul::<B>(p.k.clone(), col::<B>(p.beta.clone(), d.dk));
    let a = mul::<B>(
        mm::<B>(kbeta.clone(), tr::<B>(p.k.clone())),
        mul::<B>(decay.clone(), strict),
    );
    let t = unit_lower_inverse::<B>(a.clone());
    let w = mm::<B>(t.clone(), mul::<B>(kbeta, col::<B>(gamma.clone(), d.dk)));
    let u = mm::<B>(
        t.clone(),
        mul::<B>(p.v.clone(), col::<B>(p.beta.clone(), d.dv)),
    );
    Intra {
        gcum,
        gamma,
        decay,
        a,
        t,
        w,
        u,
    }
}

/// `G_last` per chunk, `[B, H, N, 1]`.
fn g_last<B: Backend>(gcum: Ft<B>) -> Ft<B> {
    B::float_slice(
        gcum,
        &[full(), full(), full(), Slice::from(CHUNK - 1..CHUNK)],
    )
}

/// `lam = exp(G_last - G)`, `[B, H, N, C]`.
fn lam<B: Backend>(gcum: Ft<B>) -> Ft<B> {
    let shape = gcum.shape();
    let last = B::float_expand(g_last::<B>(gcum.clone()), shape);
    B::float_exp(sub::<B>(last, gcum))
}

/// The forward pass: the output `[B, T, H, Dv]` in the queries' dtype and the final state
/// `[B, H, Dk, Dv]` in f32.
pub(crate) fn forward<B: Backend>(
    q: Ft<B>,
    k: Ft<B>,
    v: Ft<B>,
    g: Ft<B>,
    beta: Ft<B>,
    initial_state: Ft<B>,
    use_qk_l2norm: bool,
) -> (Ft<B>, Ft<B>) {
    let dtype = q.dtype();
    let p = prepare::<B>(q, k, v, g, beta, use_qk_l2norm);
    let d = p.d;
    let Intra {
        gcum,
        gamma,
        decay,
        w,
        u,
        ..
    } = intra::<B>(&p);
    let kt = mul::<B>(p.k.clone(), col::<B>(lam::<B>(gcum.clone()), d.dk));
    let gl = B::float_exp(g_last::<B>(gcum));
    let (states, v_new, last) = state_forward::<B>(kt, w, u, gl, to_f32::<B>(initial_state));
    let inter = mm::<B>(mul::<B>(p.q.clone(), col::<B>(gamma, d.dk)), states);
    let attn = mul::<B>(mm::<B>(p.q, tr::<B>(p.k)), decay);
    let o = add::<B>(inter, mm::<B>(attn, v_new));
    (to_dtype::<B>(unchunk4::<B>(o, &d, d.dv), dtype), last)
}

/// The gradients of one call, each in its input's dtype (the initial state's in f32).
type Grads<B> = (Ft<B>, Ft<B>, Ft<B>, Ft<B>, Ft<B>, Ft<B>);

/// The backward pass, written out: recomputes the forward's chunked quantities, passes the
/// state's gradient back from chunk to chunk, then forms every input's gradient batched over all
/// chunks. `d_out` is `[B, T, H, Dv]`, `d_final_state` `[B, H, Dk, Dv]` (zeros when the final
/// state is not used).
#[allow(clippy::too_many_arguments)]
pub(crate) fn backward<B: Backend>(
    q: Ft<B>,
    k: Ft<B>,
    v: Ft<B>,
    g: Ft<B>,
    beta: Ft<B>,
    initial_state: Ft<B>,
    d_out: Ft<B>,
    d_final_state: Ft<B>,
    use_qk_l2norm: bool,
) -> Grads<B> {
    let dtypes = [q.dtype(), k.dtype(), v.dtype(), g.dtype(), beta.dtype()];
    let p = prepare::<B>(q, k, v, g, beta, use_qk_l2norm);
    let d = p.d;
    let device = p.q.device();
    let Intra {
        gcum,
        gamma,
        decay,
        a,
        t,
        w,
        u,
    } = intra::<B>(&p);

    // Recompute the state entering each chunk and V'.
    let lam = lam::<B>(gcum.clone());
    let kt = mul::<B>(p.k.clone(), col::<B>(lam.clone(), d.dk));
    let gl = B::float_exp(g_last::<B>(gcum));
    let (states, v_new, _) = state_forward::<B>(
        kt.clone(),
        w.clone(),
        u,
        gl.clone(),
        to_f32::<B>(initial_state),
    );

    // Pass the state's gradient back.
    let d_o = chunk4::<B>(to_f32::<B>(d_out), &d, d.dv);
    let qg = mul::<B>(p.q.clone(), col::<B>(gamma.clone(), d.dk));
    let attn = mul::<B>(mm::<B>(p.q.clone(), tr::<B>(p.k.clone())), decay.clone());
    let dv_local = mm::<B>(tr::<B>(attn.clone()), d_o.clone());
    let (ds_next, dvn, d_initial) = state_backward::<B>(
        kt,
        w,
        qg,
        d_o.clone(),
        dv_local,
        gl.clone(),
        to_f32::<B>(d_final_state),
    );

    let sq = d.square();
    let incl = lower_mask::<B>(false, sq.clone(), &device);
    let strict = lower_mask::<B>(true, sq, &device);

    // O = (gamma Q) S + P V'.
    let d_qg = mm::<B>(d_o.clone(), tr::<B>(states.clone()));
    let mut dq = mul::<B>(d_qg.clone(), col::<B>(gamma.clone(), d.dk));
    let mut d_gamma = sum_last::<B>(mul::<B>(d_qg, p.q.clone()));
    let d_attn = mul::<B>(mm::<B>(d_o, tr::<B>(v_new.clone())), incl);
    let d_attn_e = mul::<B>(d_attn.clone(), decay.clone());
    dq = add::<B>(dq, mm::<B>(d_attn_e.clone(), p.k.clone()));
    let mut dk = mm::<B>(tr::<B>(d_attn_e), p.q.clone());
    let m = mul::<B>(d_attn, attn);
    let mut d_g = sub::<B>(sum_last::<B>(m.clone()), sum_rows::<B>(m));

    // S_next = exp(G_last) S + (lam K)^T V'.
    let d_kt = mm::<B>(v_new, tr::<B>(ds_next.clone()));
    dk = add::<B>(dk, mul::<B>(d_kt.clone(), col::<B>(lam.clone(), d.dk)));
    let lam_dlam = mul::<B>(sum_last::<B>(mul::<B>(d_kt, p.k.clone())), lam);
    d_g = sub::<B>(d_g, lam_dlam.clone());
    // dG_last gathers sum_i lam_i dlam_i and exp(G_last) <S, dS_next>.
    let [b, h, n, kk, vv] = states.shape().dims::<5>();
    let s_ds = B::float_sum_dim(
        B::float_reshape(
            mul::<B>(states.clone(), ds_next),
            Shape::new([b, h, n, kk * vv]),
        ),
        3,
    );
    let d_last = add::<B>(B::float_sum_dim(lam_dlam, 3), mul::<B>(s_ds, gl));
    let last = Slice::from(CHUNK - 1..CHUNK);
    let g_tail = B::float_slice(d_g.clone(), &[full(), full(), full(), last]);
    d_g = B::float_slice_assign(
        d_g,
        &[full(), full(), full(), last],
        add::<B>(g_tail, d_last),
    );

    // V' = U - W S.
    let dw = B::float_neg(mm::<B>(dvn.clone(), tr::<B>(states)));
    let du = dvn;

    // W = T (beta gamma K), U = T (beta V).
    let beta_v = col::<B>(p.beta.clone(), d.dv);
    let beta_k = col::<B>(p.beta.clone(), d.dk);
    let gamma_k = col::<B>(gamma.clone(), d.dk);
    let vb = mul::<B>(p.v.clone(), beta_v.clone());
    let kbg = mul::<B>(mul::<B>(p.k.clone(), beta_k.clone()), gamma_k.clone());
    let d_t = add::<B>(
        mm::<B>(du.clone(), tr::<B>(vb)),
        mm::<B>(dw.clone(), tr::<B>(kbg)),
    );
    let d_vb = mm::<B>(tr::<B>(t.clone()), du);
    let d_kbg = mm::<B>(tr::<B>(t.clone()), dw);
    let dv = mul::<B>(d_vb.clone(), beta_v);
    let mut d_beta = add::<B>(
        sum_last::<B>(mul::<B>(d_vb, p.v.clone())),
        sum_last::<B>(mul::<B>(
            mul::<B>(d_kbg.clone(), p.k.clone()),
            gamma_k.clone(),
        )),
    );
    dk = add::<B>(
        dk,
        mul::<B>(mul::<B>(d_kbg.clone(), beta_k.clone()), gamma_k),
    );
    d_gamma = add::<B>(
        d_gamma,
        sum_last::<B>(mul::<B>(mul::<B>(d_kbg, p.k.clone()), beta_k.clone())),
    );
    d_g = add::<B>(d_g, mul::<B>(d_gamma, gamma));

    // T = (I + A)^-1: dA = -(T^T dT T^T), strictly lower.
    let t_tr = tr::<B>(t);
    let d_a = mul::<B>(
        B::float_neg(mm::<B>(mm::<B>(t_tr.clone(), d_t), t_tr)),
        strict.clone(),
    );
    // A = (beta K K^T) decay, strictly lower.
    let d_ae = mul::<B>(d_a.clone(), mul::<B>(decay, strict));
    let d_kbeta = mm::<B>(d_ae.clone(), p.k.clone());
    let kbeta = mul::<B>(p.k.clone(), beta_k.clone());
    dk = add::<B>(dk, mul::<B>(d_kbeta.clone(), beta_k));
    dk = add::<B>(dk, mm::<B>(tr::<B>(d_ae), kbeta));
    d_beta = add::<B>(d_beta, sum_last::<B>(mul::<B>(d_kbeta, p.k.clone())));
    let m = mul::<B>(d_a, a);
    d_g = add::<B>(d_g, sub::<B>(sum_last::<B>(m.clone()), sum_rows::<B>(m)));

    // G = cumsum(g) within each chunk: dg is the reverse cumulative sum of dG.
    let total = B::float_expand(B::float_sum_dim(d_g.clone(), 3), d_g.shape());
    let dg = add::<B>(sub::<B>(total, B::float_cumsum(d_g.clone(), 3)), d_g);

    let mut dq = unchunk4::<B>(B::float_mul_scalar(dq, scale(d.dk).into()), &d, d.dk);
    let mut dk = unchunk4::<B>(dk, &d, d.dk);
    if let Some([qn, qi, kn, ki]) = p.norms {
        dq = l2norm_backward::<B>(qn, qi, dq);
        dk = l2norm_backward::<B>(kn, ki, dk);
    }
    (
        to_dtype::<B>(dq, dtypes[0]),
        to_dtype::<B>(dk, dtypes[1]),
        to_dtype::<B>(unchunk4::<B>(dv, &d, d.dv), dtypes[2]),
        to_dtype::<B>(unchunk3::<B>(dg, &d), dtypes[3]),
        to_dtype::<B>(unchunk3::<B>(d_beta, &d), dtypes[4]),
        d_initial,
    )
}
