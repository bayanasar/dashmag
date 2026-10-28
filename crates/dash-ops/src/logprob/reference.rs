//! The reference: the chunked algorithm of the module docs in Burn's tensor operations, f32
//! throughout. Every backend can run it; the tests hold the CubeCL kernels to it, and it to
//! PyTorch. Its two passes over the vocabulary serve [`crate::loss`] too: [`pass`], the forward
//! pass, and [`gradient_pass`], the second pass a trained output weight needs.

use super::{LogprobOptions, RowBlocks, coefficient};
use crate::reference::matmul;
use burn::backend::tensor::{BoolTensor, FloatTensor, IntTensor};
use burn::backend::{
    Backend, DType, FloatDType, Scalar, Shape, Slice, TensorData, TensorMetadata,
    get_or_init_device_settings,
};

/// What one forward pass produces.
pub(crate) struct Forward<B: Backend> {
    /// `[n]` f32.
    pub logprob: FloatTensor<B>,
    /// `[n]` f32.
    pub entropy: FloatTensor<B>,
    /// `[n, D]` f32: the gradient of `logprob[i]` with respect to `hidden[i]`.
    pub grad: Option<FloatTensor<B>>,
}

/// What a pass over the vocabulary leaves per row (see the module docs), each `[n]` f32.
pub(crate) struct Rows<B: Backend> {
    /// The maximum of the unmasked logits (`f32::MIN` before any).
    pub m: FloatTensor<B>,
    /// `sum_j exp(z - m)` over the unmasked tokens.
    pub s: FloatTensor<B>,
    /// The same sum without the target's term.
    pub r: FloatTensor<B>,
    /// `sum_j exp(z - m) (z - m)`, for the entropy.
    pub a: FloatTensor<B>,
    /// The target's logit, 0 if it is masked or out of range.
    pub zt: FloatTensor<B>,
    /// 1 once the target has been seen, 0 if it is masked or out of range.
    pub found: FloatTensor<B>,
    /// `sum_j z` over the unmasked tokens, when asked for.
    pub zsum: Option<FloatTensor<B>>,
}

/// What [`pass`] returns: the rows' state and, when asked for, `o` `[n, D]` f32, the sum
/// `sum_{j != t} exp(z - m) weight[j]` over the unmasked tokens.
pub(crate) struct Pass<B: Backend> {
    pub rows: Rows<B>,
    pub o: Option<FloatTensor<B>>,
}

/// The running state of the online log-sum-exp for one block of rows, each `[rows, 1]` f32.
struct Running<B: Backend> {
    /// Starts at `f32::MIN`, not `-inf`, so that `m_old - m_new` is never `inf - inf`.
    m: FloatTensor<B>,
    s: FloatTensor<B>,
    r: FloatTensor<B>,
    a: FloatTensor<B>,
    zt: FloatTensor<B>,
    found: FloatTensor<B>,
    zsum: Option<FloatTensor<B>>,
    /// `[rows, D]`.
    o: Option<FloatTensor<B>>,
}

/// The chosen-token log-probability: one [`pass`], then the outputs.
pub(crate) fn forward<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    targets: IntTensor<B>,
    masked: &[u32],
    options: LogprobOptions,
    with_grad: bool,
) -> Forward<B> {
    let [n, d] = dims2(&hidden.shape(), "hidden");
    let [v, _] = dims2(&weight.shape(), "weight");
    let device = hidden.device();
    if n == 0 {
        return Forward {
            logprob: full::<B>([0], 0.0, &device),
            entropy: full::<B>([0], 0.0, &device),
            grad: with_grad.then(|| full::<B>([0, d], 0.0, &device)),
        };
    }
    let bool_dtype = get_or_init_device_settings::<B>(&device).bool_dtype;
    let Pass { rows, o } = pass::<B>(
        hidden,
        weight.clone(),
        targets.clone(),
        &sorted(masked),
        options,
        with_grad,
        false,
    );

    // logprob = zt - (m + ln s) where the target was seen, -inf elsewhere.
    let ln_s = B::float_log(rows.s.clone());
    let lse = B::float_add(rows.m, ln_s.clone());
    let seen = B::float_greater_elem(rows.found.clone(), Scalar::from(0.5), bool_dtype);
    let logprob = B::float_mask_where(
        full::<B>([n], f32::NEG_INFINITY, &device),
        seen,
        B::float_sub(rows.zt, lse),
    );
    let entropy = B::float_sub(ln_s, B::float_div(rows.a, rows.s.clone()));
    let grad = o.map(|o| {
        // (r weight[t] - o) / s, with the weight[t] term only where the target was seen.
        let column = |x: FloatTensor<B>| B::float_reshape(x, Shape::new([n, 1]));
        let weight_t = target_rows::<B>(weight, targets, v);
        let coef = column(B::float_mul(rows.r, rows.found));
        B::float_div(
            B::float_sub(B::float_mul(weight_t, coef), o),
            column(rows.s),
        )
    });
    Forward {
        logprob,
        entropy,
        grad,
    }
}

/// One pass over the vocabulary for all `n > 0` rows, block by block of rows: the online
/// log-sum-exp's state per row, `o` when `accumulate`, and the sum of the logits when
/// `logit_sum`. `masked` is sorted and free of repeats.
pub(crate) fn pass<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    targets: IntTensor<B>,
    masked: &[u32],
    options: LogprobOptions,
    accumulate: bool,
    logit_sum: bool,
) -> Pass<B> {
    let [n, d] = dims2(&hidden.shape(), "hidden");
    let [v, dw] = dims2(&weight.shape(), "weight");
    assert_eq!(d, dw, "hidden [n, D] and weight [V, D] must share D");
    assert_eq!(targets.shape().dims::<1>(), [n], "targets must be [n]");
    assert!(n > 0, "a pass needs at least one row");
    let device = hidden.device();
    let hidden = to_f32::<B>(hidden);

    let mut blocks = Vec::new();
    for block in options.row_blocks(n) {
        let h = B::float_slice(hidden.clone(), &[Slice::from(block.clone()), Slice::full()]);
        let t = B::int_slice(targets.clone(), &[Slice::from(block)]);
        blocks.push(block_pass::<B>(
            h,
            weight.clone(),
            t,
            masked,
            options,
            accumulate,
            logit_sum,
            v,
            &device,
        ));
    }
    let column = |pick: fn(&Running<B>) -> FloatTensor<B>| {
        let parts = blocks
            .iter()
            .map(|block| {
                let x = pick(block);
                let rows = x.shape()[0];
                B::float_reshape(x, Shape::new([rows]))
            })
            .collect();
        cat::<B>(parts)
    };
    let rows = Rows {
        m: column(|b| b.m.clone()),
        s: column(|b| b.s.clone()),
        r: column(|b| b.r.clone()),
        a: column(|b| b.a.clone()),
        zt: column(|b| b.zt.clone()),
        found: column(|b| b.found.clone()),
        zsum: logit_sum.then(|| column(|b| b.zsum.clone().expect("asked for"))),
    };
    let o = accumulate.then(|| {
        cat::<B>(
            blocks
                .iter()
                .map(|b| b.o.clone().expect("asked for"))
                .collect(),
        )
    });
    Pass { rows, o }
}

/// One block of rows: `hidden` `[rows, D]` f32, `targets` `[rows]`.
#[allow(clippy::too_many_arguments)]
fn block_pass<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    targets: IntTensor<B>,
    masked: &[u32],
    options: LogprobOptions,
    accumulate: bool,
    logit_sum: bool,
    v: usize,
    device: &B::Device,
) -> Running<B> {
    let bool_dtype = get_or_init_device_settings::<B>(device).bool_dtype;
    let [rows, d] = dims2(&hidden.shape(), "hidden");
    let column = |value: f32| full::<B>([rows, 1], value, device);
    let mut st = Running::<B> {
        m: column(f32::MIN),
        s: column(0.0),
        r: column(0.0),
        a: column(0.0),
        zt: column(0.0),
        found: column(0.0),
        zsum: logit_sum.then(|| column(0.0)),
        o: accumulate.then(|| full::<B>([rows, d], 0.0, device)),
    };
    let targets_col = B::int_reshape(targets, Shape::new([rows, 1]));

    let chunk = options.vocab_chunk_for(rows);
    for c0 in (0..v).step_by(chunk) {
        let c1 = (c0 + chunk).min(v);
        let weight_c = to_f32::<B>(B::float_slice(
            weight.clone(),
            &[Slice::from(c0..c1), Slice::full()],
        ));
        let z = matmul::<B>(hidden.clone(), B::float_transpose(weight_c.clone()));
        let chunk = Chunk::<B>::new(&targets_col, masked, c0, c1, device);
        let masked_fill = |x: FloatTensor<B>, value: f32| chunk.masked_fill(x, value);

        // Online update: m_new = max(m, max_j z), alpha = exp(m - m_new).
        let zm = masked_fill(z.clone(), f32::MIN);
        let chunk_max = B::float_max_dim(zm.clone(), 1);
        let grows = B::float_greater(chunk_max.clone(), st.m.clone(), bool_dtype);
        let m_new = B::float_mask_where(st.m.clone(), grows, chunk_max);
        let shift = B::float_sub(st.m.clone(), m_new.clone());
        let alpha = B::float_exp(shift.clone());
        // Masked entries: `dz = 0` keeps `e * dz` finite, then `e = 0` drops them.
        let dz = masked_fill(B::float_sub(zm, m_new.clone()), 0.0);
        let e = masked_fill(B::float_exp(dz.clone()), 0.0);

        // a = alpha (a + s (m - m_new)) + sum e dz, before s moves on.
        st.a = B::float_add(
            B::float_mul(
                alpha.clone(),
                B::float_add(st.a, B::float_mul(st.s.clone(), shift)),
            ),
            B::float_sum_dim(B::float_mul(e.clone(), dz), 1),
        );
        st.s = B::float_add(
            B::float_mul(st.s, alpha.clone()),
            B::float_sum_dim(e.clone(), 1),
        );
        let e_other = B::float_mask_fill(e, chunk.is_target.clone(), Scalar::from(0.0));
        st.r = B::float_add(
            B::float_mul(st.r, alpha.clone()),
            B::float_sum_dim(e_other.clone(), 1),
        );
        let is_target = B::bool_into_float(chunk.is_target.clone(), FloatDType::F32);
        st.zt = B::float_add(
            st.zt,
            B::float_sum_dim(B::float_mul(z.clone(), is_target.clone()), 1),
        );
        st.found = B::float_add(st.found, B::float_sum_dim(is_target, 1));
        st.zsum = st
            .zsum
            .map(|zsum| B::float_add(zsum, B::float_sum_dim(masked_fill(z, 0.0), 1)));
        st.o =
            st.o.map(|o| B::float_add(B::float_mul(o, alpha), matmul::<B>(e_other, weight_c)));
        st.m = m_new;
    }
    st
}

/// The masked columns of one chunk `c0..c1` and the target's column, `[rows, c1 - c0]`.
struct Chunk<B: Backend> {
    /// `true` at the masked columns, broadcast over the rows; `None` if the chunk has none.
    mask: Option<BoolTensor<B>>,
    /// `true` at each row's target, if it falls in this chunk and is not masked.
    is_target: BoolTensor<B>,
}

impl<B: Backend> Chunk<B> {
    fn new(
        targets_col: &IntTensor<B>,
        masked: &[u32],
        c0: usize,
        c1: usize,
        device: &B::Device,
    ) -> Self {
        let settings = get_or_init_device_settings::<B>(device);
        let rows = targets_col.shape()[0];
        let w = c1 - c0;
        let in_chunk: Vec<bool> = (c0..c1)
            .map(|j| masked.binary_search(&(j as u32)).is_ok())
            .collect();
        let mask: Option<BoolTensor<B>> = in_chunk.iter().any(|&m| m).then(|| {
            // In the device's bool dtype: CubeCL devices store bools as integers.
            let row = TensorData::new(in_chunk, [1, w]).convert_dtype(settings.bool_dtype.into());
            B::bool_expand(B::bool_from_data(row, device), Shape::new([rows, w]))
        });
        let columns = B::int_reshape(
            B::int_arange(c0 as i64..c1 as i64, device, settings.int_dtype),
            Shape::new([1, w]),
        );
        let mut is_target = B::int_equal(
            B::int_expand(targets_col.clone(), Shape::new([rows, w])),
            B::int_expand(columns, Shape::new([rows, w])),
            settings.bool_dtype,
        );
        if let Some(mask) = &mask {
            is_target = B::bool_and(is_target, B::bool_not(mask.clone()));
        }
        Self { mask, is_target }
    }

    fn masked_fill(&self, x: FloatTensor<B>, value: f32) -> FloatTensor<B> {
        match &self.mask {
            Some(mask) => B::float_mask_fill(x, mask.clone(), Scalar::from(value)),
            None => x,
        }
    }
}

/// The gradient pass: the vocabulary walked once more, the logits recomputed chunk by chunk,
/// and from them and the per-row coefficients `coef` `[coefficient::COUNT, n]` f32 the gradient
/// of the logits
///
/// ```text
/// g[i, j] = exp(z - lse) (A - E (z - lse)) + C    an unmasked token other than the target
/// g[i, t] = T + C                                 the target, when unmasked
/// g[i, j] = 0                                     a masked token
/// ```
///
/// (`lse`, `A`, `E`, `T`, `C` the rows of `coef`; `E` is taken as 0 unless `entropy`), then
/// `grad_hidden = g weight` and `grad_weight = g^T hidden`. Returns them in `hidden`'s and
/// `weight`'s dtypes, each `[0, D]` when not wanted. `masked` is sorted and free of repeats.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_pass<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    targets: IntTensor<B>,
    masked: &[u32],
    options: LogprobOptions,
    coef: FloatTensor<B>,
    entropy: bool,
    want_hidden: bool,
    want_weight: bool,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let [n, d] = dims2(&hidden.shape(), "hidden");
    let [v, _] = dims2(&weight.shape(), "weight");
    let (hidden_dtype, weight_dtype) = (hidden.dtype(), weight.dtype());
    let device = hidden.device();
    let none = || full::<B>([0, d], 0.0, &device);
    let hidden = to_f32::<B>(hidden);
    let blocks: Vec<_> = options.row_blocks(n).collect();
    let chunk = options.vocab_chunk_for(blocks.first().map_or(1, |b| b.len()));
    let chunks: Vec<_> = (0..v)
        .step_by(chunk)
        .map(|c0| c0..(c0 + chunk).min(v))
        .collect();

    let mut grad_hidden = Vec::new();
    let mut grad_weight: Vec<Option<FloatTensor<B>>> = vec![None; chunks.len()];
    for block in &blocks {
        let rows = block.len();
        let h = B::float_slice(hidden.clone(), &[Slice::from(block.clone()), Slice::full()]);
        let t = B::int_reshape(
            B::int_slice(targets.clone(), &[Slice::from(block.clone())]),
            Shape::new([rows, 1]),
        );
        let row = |k: usize| {
            let x = B::float_slice(
                coef.clone(),
                &[Slice::from(k..k + 1), Slice::from(block.clone())],
            );
            B::float_reshape(x, Shape::new([rows, 1]))
        };
        let (lse, softmax, ent, target, uniform) = (
            row(coefficient::LSE),
            row(coefficient::SOFTMAX),
            row(coefficient::ENTROPY),
            row(coefficient::TARGET),
            row(coefficient::UNIFORM),
        );
        let target_value = B::float_add(target, uniform.clone());
        let mut gh = want_hidden.then(|| full::<B>([rows, d], 0.0, &device));
        for (k, range) in chunks.iter().enumerate() {
            let w = range.len();
            let weight_c = to_f32::<B>(B::float_slice(
                weight.clone(),
                &[Slice::from(range.clone()), Slice::full()],
            ));
            let z = matmul::<B>(h.clone(), B::float_transpose(weight_c.clone()));
            let chunk = Chunk::<B>::new(&t, masked, range.start, range.end, &device);
            let dz = B::float_sub(z, lse.clone());
            let p = B::float_exp(dz.clone());
            let mut g = B::float_mul(p.clone(), softmax.clone());
            if entropy {
                g = B::float_sub(g, B::float_mul(B::float_mul(p, dz), ent.clone()));
            }
            g = B::float_add(g, uniform.clone());
            g = B::float_mask_where(
                g,
                chunk.is_target.clone(),
                B::float_expand(target_value.clone(), Shape::new([rows, w])),
            );
            g = chunk.masked_fill(g, 0.0);
            gh = gh.map(|gh| B::float_add(gh, matmul::<B>(g.clone(), weight_c)));
            if want_weight {
                let part = matmul::<B>(B::float_transpose(g), h.clone());
                grad_weight[k] = Some(match grad_weight[k].take() {
                    Some(sum) => B::float_add(sum, part),
                    None => part,
                });
            }
        }
        grad_hidden.extend(gh);
    }

    let grad_hidden = if !want_hidden {
        none()
    } else if n == 0 {
        cast::<B>(full::<B>([0, d], 0.0, &device), hidden_dtype)
    } else {
        cast::<B>(cat::<B>(grad_hidden), hidden_dtype)
    };
    let grad_weight = if !want_weight {
        none()
    } else if n == 0 {
        cast::<B>(full::<B>([v, d], 0.0, &device), weight_dtype)
    } else {
        let parts = grad_weight
            .into_iter()
            .map(|part| part.expect("every chunk meets at least one row"))
            .collect();
        cast::<B>(cat::<B>(parts), weight_dtype)
    };
    (grad_hidden, grad_weight)
}

/// `weight[t]` for each target, `[n, D]` f32; a target outside `0..V` reads a clamped row.
pub(crate) fn target_rows<B: Backend>(
    weight: FloatTensor<B>,
    targets: IntTensor<B>,
    v: usize,
) -> FloatTensor<B> {
    let t = B::int_clamp(targets, Scalar::from(0i64), Scalar::from(v as i64 - 1));
    to_f32::<B>(B::float_select(weight, 0, t))
}

/// The masked ids sorted and free of repeats.
pub(crate) fn sorted(masked: &[u32]) -> Vec<u32> {
    let mut masked = masked.to_vec();
    masked.sort_unstable();
    masked.dedup();
    masked
}

pub(crate) fn dims2(shape: &Shape, name: &str) -> [usize; 2] {
    assert_eq!(shape.num_dims(), 2, "{name} must be 2-D, got {shape:?}");
    shape.dims::<2>()
}

pub(crate) fn to_f32<B: Backend>(x: FloatTensor<B>) -> FloatTensor<B> {
    match x.dtype() {
        DType::F32 => x,
        _ => B::float_cast(x, FloatDType::F32),
    }
}

/// `x` (f32) in `dtype`.
pub(crate) fn cast<B: Backend>(x: FloatTensor<B>, dtype: DType) -> FloatTensor<B> {
    if x.dtype() == dtype {
        x
    } else {
        B::float_cast(x, dtype.into())
    }
}

pub(crate) fn full<B: Backend>(
    shape: impl Into<Shape>,
    value: f32,
    device: &B::Device,
) -> FloatTensor<B> {
    B::float_full(shape.into(), Scalar::from(value), device, FloatDType::F32)
}

fn cat<B: Backend>(mut parts: Vec<FloatTensor<B>>) -> FloatTensor<B> {
    if parts.len() == 1 {
        parts.pop().expect("one part")
    } else {
        B::float_cat(parts, 0)
    }
}
