//! The reference: the chunked algorithm of the module docs in Burn's tensor operations, f32
//! throughout. Every backend can run it; the tests hold the CUDA kernels to it, and it to
//! PyTorch.

use super::LogprobOptions;
use burn::backend::tensor::{BoolTensor, FloatTensor, IntTensor};
use burn::backend::{
    Backend, FloatDType, Scalar, Shape, Slice, TensorData, TensorMetadata,
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

/// The running state of the online log-sum-exp, each `[rows, 1]` f32 (see the module docs).
struct Running<B: Backend> {
    /// The running maximum of the unmasked logits; starts at `f32::MIN`, not `-inf`, so that
    /// `m_old - m_new` is never `inf - inf`.
    m: FloatTensor<B>,
    /// `sum_j exp(z - m)` over the unmasked tokens.
    s: FloatTensor<B>,
    /// The same sum without the target's term.
    r: FloatTensor<B>,
    /// `sum_j exp(z - m) (z - m)`, for the entropy.
    a: FloatTensor<B>,
    /// The target's logit, and 1 once it has been seen (0 if it is masked or out of range).
    zt: FloatTensor<B>,
    found: FloatTensor<B>,
    /// `sum_{j != t} exp(z - m) weight[j]`, `[rows, D]`, when the gradient is wanted.
    o: Option<FloatTensor<B>>,
}

pub(crate) fn forward<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    targets: IntTensor<B>,
    masked: &[u32],
    options: LogprobOptions,
    with_grad: bool,
) -> Forward<B> {
    let [n, d] = dims2(&hidden.shape(), "hidden");
    let [v, dw] = dims2(&weight.shape(), "weight");
    assert_eq!(d, dw, "hidden [n, D] and weight [V, D] must share D");
    assert_eq!(targets.shape().dims::<1>(), [n], "targets must be [n]");
    let device = hidden.device();
    let mut masked = masked.to_vec();
    masked.sort_unstable();
    masked.dedup();
    let masked = masked.as_slice();
    if n == 0 {
        return Forward {
            logprob: full::<B>([0], 0.0, &device),
            entropy: full::<B>([0], 0.0, &device),
            grad: with_grad.then(|| full::<B>([0, d], 0.0, &device)),
        };
    }
    let hidden = to_f32::<B>(hidden);

    let mut logprobs = Vec::new();
    let mut entropies = Vec::new();
    let mut grads = Vec::new();
    for block in options.row_blocks(n) {
        let h = B::float_slice(hidden.clone(), &[Slice::from(block.clone()), Slice::full()]);
        let t = B::int_slice(targets.clone(), &[Slice::from(block)]);
        let out = rows::<B>(h, weight.clone(), t, masked, options, with_grad, v, &device);
        logprobs.push(out.logprob);
        entropies.push(out.entropy);
        grads.extend(out.grad);
    }
    Forward {
        logprob: cat::<B>(logprobs),
        entropy: cat::<B>(entropies),
        grad: with_grad.then(|| cat::<B>(grads)),
    }
}

/// One block of rows: `hidden` `[rows, D]` f32, `targets` `[rows]`.
#[allow(clippy::too_many_arguments)]
fn rows<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    targets: IntTensor<B>,
    masked: &[u32],
    options: LogprobOptions,
    with_grad: bool,
    v: usize,
    device: &B::Device,
) -> Forward<B> {
    let settings = get_or_init_device_settings::<B>(device);
    let (bool_dtype, int_dtype) = (settings.bool_dtype, settings.int_dtype);
    let [rows, d] = dims2(&hidden.shape(), "hidden");
    let column = |value: f32| full::<B>([rows, 1], value, device);
    let mut st = Running::<B> {
        m: column(f32::MIN),
        s: column(0.0),
        r: column(0.0),
        a: column(0.0),
        zt: column(0.0),
        found: column(0.0),
        o: with_grad.then(|| full::<B>([rows, d], 0.0, device)),
    };
    let targets_col = B::int_reshape(targets.clone(), Shape::new([rows, 1]));

    let chunk = options.vocab_chunk_for(rows);
    for c0 in (0..v).step_by(chunk) {
        let c1 = (c0 + chunk).min(v);
        let w = c1 - c0;
        let weight_c = to_f32::<B>(B::float_slice(
            weight.clone(),
            &[Slice::from(c0..c1), Slice::full()],
        ));
        let z = B::float_matmul(hidden.clone(), B::float_transpose(weight_c.clone()));

        // The chunk's masked columns, broadcast over the rows.
        let in_chunk: Vec<bool> = (c0..c1)
            .map(|j| masked.binary_search(&(j as u32)).is_ok())
            .collect();
        let mask: Option<BoolTensor<B>> = in_chunk.iter().any(|&m| m).then(|| {
            let row = B::bool_from_data(TensorData::new(in_chunk, [1, w]), device);
            B::bool_expand(row, Shape::new([rows, w]))
        });
        let masked_fill = |x: FloatTensor<B>, value: f32| match &mask {
            Some(mask) => B::float_mask_fill(x, mask.clone(), Scalar::from(value)),
            None => x,
        };

        // The target's column, if it falls in this chunk and is not masked.
        let columns = B::int_reshape(
            B::int_arange(c0 as i64..c1 as i64, device, int_dtype),
            Shape::new([1, w]),
        );
        let mut is_target = B::int_equal(
            B::int_expand(targets_col.clone(), Shape::new([rows, w])),
            B::int_expand(columns, Shape::new([rows, w])),
            bool_dtype,
        );
        if let Some(mask) = &mask {
            is_target = B::bool_and(is_target, B::bool_not(mask.clone()));
        }

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
        let e_other = B::float_mask_fill(e, is_target.clone(), Scalar::from(0.0));
        st.r = B::float_add(
            B::float_mul(st.r, alpha.clone()),
            B::float_sum_dim(e_other.clone(), 1),
        );
        let is_target = B::bool_into_float(is_target, FloatDType::F32);
        st.zt = B::float_add(
            st.zt,
            B::float_sum_dim(B::float_mul(z, is_target.clone()), 1),
        );
        st.found = B::float_add(st.found, B::float_sum_dim(is_target, 1));
        st.o =
            st.o.map(|o| B::float_add(B::float_mul(o, alpha), B::float_matmul(e_other, weight_c)));
        st.m = m_new;
    }

    // logprob = zt - (m + ln s) where the target was seen, -inf elsewhere.
    let ln_s = B::float_log(st.s.clone());
    let lse = B::float_add(st.m, ln_s.clone());
    let seen = B::float_greater_elem(st.found.clone(), Scalar::from(0.5), bool_dtype);
    let logprob = B::float_mask_where(
        full::<B>([rows, 1], f32::NEG_INFINITY, device),
        seen,
        B::float_sub(st.zt, lse),
    );
    let entropy = B::float_sub(ln_s, B::float_div(st.a, st.s.clone()));
    let grad = st.o.map(|o| {
        // (r weight[t] - o) / s, with the weight[t] term only where the target was seen.
        let t = B::int_clamp(targets, Scalar::from(0i64), Scalar::from(v as i64 - 1));
        let weight_t = to_f32::<B>(B::float_select(weight, 0, t));
        let coef = B::float_mul(st.r, st.found);
        B::float_div(B::float_sub(B::float_mul(weight_t, coef), o), st.s)
    });
    Forward {
        logprob: B::float_reshape(logprob, Shape::new([rows])),
        entropy: B::float_reshape(entropy, Shape::new([rows])),
        grad,
    }
}

fn dims2(shape: &Shape, name: &str) -> [usize; 2] {
    assert_eq!(shape.num_dims(), 2, "{name} must be 2-D, got {shape:?}");
    shape.dims::<2>()
}

fn to_f32<B: Backend>(x: FloatTensor<B>) -> FloatTensor<B> {
    match x.dtype() {
        burn::backend::DType::F32 => x,
        _ => B::float_cast(x, FloatDType::F32),
    }
}

fn full<B: Backend>(shape: impl Into<Shape>, value: f32, device: &B::Device) -> FloatTensor<B> {
    B::float_full(shape.into(), Scalar::from(value), device, FloatDType::F32)
}

fn cat<B: Backend>(mut parts: Vec<FloatTensor<B>>) -> FloatTensor<B> {
    if parts.len() == 1 {
        parts.pop().expect("one part")
    } else {
        B::float_cat(parts, 0)
    }
}
