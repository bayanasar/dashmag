//! Low-rank products `y = base + scale · (x U) V`, with their gradients: dash-ops' `lowrank` on
//! MLX. The operation, its shapes, its dtypes and the precision of each part are dash-ops' (its
//! `lowrank` module documents them): `x` `[.., in]` in f32, bf16 or f16, read exactly into f32;
//! the factors `U` `[in, r]` and `V` `[r, out]` in f32, with any strides (PEFT's `lora_A` and
//! `lora_B` transposed: views); every product in f32, then `· scale` in f32, in PEFT's order
//! (`lora_B(lora_A(x.float())) * scaling`); with a base `[.., out]`, the sum in f32 rounded once
//! to the base's dtype, PEFT's `(result + lora).to(result.dtype)`; without one, the f32 term.
//!
//! On MLX, `h = x U` is MLX's matmul. With a base, dash-ops' `up` kernel forms
//! `y = base + scale · h V`: each thread one output element, the sum over the rank in order in
//! f32, the base read and the output written once in their dtype, so the f32 `[.., out]` term is
//! never held (MLX's operations would cast the base to f32, form the term, add and cast back: a
//! layer's forward pass took 13.2 to 14.2 ms that way at 6144 x 1024 x 6144, rank 16, measured).
//! Its backward pass is written out as dash-ops': `dz = dy · scale` rounded to f32, `dV = hᵀ dz`,
//! `dh = dz Vᵀ`, `dU = xᵀ dh`, `dx = dh Uᵀ` rounded once to `x`'s dtype, the base's gradient `dy`
//! itself; MLX's matmuls. Without a base, MLX's operations, differentiated by MLX. f32 products are
//! IEEE f32 with `MLX_ENABLE_TF32=0`; with MLX's TF32 they run on the neural accelerators of GPUs
//! that have them, `x` read exactly when it is bf16.

use crate::kernel::{Launch, launch};
use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::TemplateArg;
use mlx_rs::transforms::custom_vjp;
use mlx_rs::{Array, Dtype};

/// dash-ops' `up`: one thread an output element, `base + scale · sum_j h[n, j] V[j, m]` in f32,
/// the sum over the rank in order, rounded once to the base's dtype.
const UP: &str = "
    const uint m = thread_position_in_grid.x;
    const uint n = thread_position_in_grid.y;
    const uint OUT = v_shape[1];
    const uint R = v_shape[0];
    if (m >= OUT) {
        return;
    }
    float acc = 0.0f;
    for (uint j = 0; j < R; ++j) {
        acc += h[size_t(n) * R + j] * v[size_t(j) * OUT + m];
    }
    const size_t at = size_t(n) * OUT + m;
    y[at] = static_cast<DT>(float(base[at]) + scale[0] * acc);
";

/// `base + scale · h V` by the `up` kernel: `h` `[N, r]` f32, `v` `[r, out]` f32, `base`
/// `[N, out]`; the result in the base's dtype.
fn up(h: &Array, v: &Array, base: &Array, scale: f32) -> Result<Array> {
    let (n, out) = (h.shape()[0], v.shape()[1]);
    let kernel = crate::kernel::kernel(
        "dash_mlx_lowrank_up",
        &["h", "v", "base", "scale"],
        &["y"],
        UP,
        "",
    )?;
    let config = Launch {
        grid: (((out + 31) / 32) * 32, n, 1),
        thread_group: (32, 8, 1),
        outputs: vec![(vec![n, out], base.dtype())],
        template: vec![("DT".into(), TemplateArg::Dtype(base.dtype()))],
        init_value: None,
    };
    let inputs = [
        h.clone(),
        v.clone(),
        base.clone(),
        Array::from_slice(&[scale], &[1]),
    ];
    // SAFETY: thread (m, n), those at `m >= out` returning at once, reads row n of `h` [N, r], the
    // r elements of column m of `v` [r, out] (all row-contiguous), `base[n, m]` and `scale[0]`, and
    // writes `y[n, m]` once; the grid covers every (m, n) of [N, out]. No input is written.
    let out = unsafe { launch(&kernel, &inputs, &config)? };
    Ok(out[0].clone())
}

pub use dash_contract::lowrank::LowRankOptions;

fn contract(message: String) -> Exception {
    Exception::custom(format!("low_rank: {message}"))
}

/// `base + scale · (x U) V` for `x` `[.., in]`, `u` `[in, r]` and `v` `[r, out]` (f32, any
/// strides), and `base` `[.., out]` with the leading dimensions of `x`: the sum in f32, rounded
/// once to the base's dtype; without a base, `scale · (x U) V` in f32 ([module](self)).
/// Differentiable with respect to `x`, `u`, `v` and `base`.
///
/// Errors if the shapes disagree, `u` or `v` is not f32, or `x` is not floating point.
pub fn low_rank(
    x: &Array,
    u: &Array,
    v: &Array,
    base: Option<&Array>,
    options: LowRankOptions,
) -> Result<Array> {
    let inp = *x
        .shape()
        .last()
        .ok_or_else(|| contract("x must have a last axis".into()))?;
    let (&[u_in, r], &[v_r, out]) = (u.shape(), v.shape()) else {
        return Err(contract(format!(
            "u must be [in, r] and v [r, out], got {:?} and {:?}",
            u.shape(),
            v.shape()
        )));
    };
    if u_in != inp || v_r != r {
        return Err(contract(format!(
            "x [.., {inp}], u [{u_in}, {r}] and v [{v_r}, {out}] do not chain"
        )));
    }
    if u.dtype() != Dtype::Float32 || v.dtype() != Dtype::Float32 {
        return Err(contract(format!(
            "the factors must be f32, got {:?} and {:?}",
            u.dtype(),
            v.dtype()
        )));
    }
    if !x.dtype().is_float() {
        return Err(contract(format!(
            "x must be floating point, got {:?}",
            x.dtype()
        )));
    }
    let scale = options.scale as f32;
    let term = || {
        x.as_dtype(Dtype::Float32)?
            .matmul(u)?
            .matmul(v)?
            .multiply(Array::from_f32(scale))
    };
    match base {
        None => term(),
        Some(base) => {
            let leading = &x.shape()[..x.ndim() - 1];
            if base.shape().last() != Some(&out) || &base.shape()[..base.ndim() - 1] != leading {
                return Err(contract(format!(
                    "base must be [.., {out}] with x's leading dimensions {leading:?}, got {:?}",
                    base.shape()
                )));
            }
            // Rows flattened for the kernel, and the result shaped back.
            let rows = |a: &Array, width: i32| a.reshape(&[-1, width]);
            let shape = base.shape().to_vec();
            let op = custom_vjp(
                move |p: &[Array]| {
                    let h = rows(&p[0], inp)?.as_dtype(Dtype::Float32)?.matmul(&p[1])?;
                    Ok(vec![
                        up(&h, &p[2], &rows(&p[3], out)?, scale)?.reshape(&shape)?,
                    ])
                },
                move |p: &[Array], cotangents: &[Array], _: &[Array]| {
                    let (x, u, v, base) = (&p[0], &p[1], &p[2], &p[3]);
                    let x32 = rows(x, inp)?.as_dtype(Dtype::Float32)?;
                    let h = x32.matmul(u)?;
                    let dy = &cotangents[0];
                    let dz = rows(dy, out)?
                        .as_dtype(Dtype::Float32)?
                        .multiply(Array::from_f32(scale))?;
                    let dv = h.t().matmul(&dz)?;
                    let dh = dz.matmul(v.t())?;
                    let du = x32.t().matmul(&dh)?;
                    let dx = dh.matmul(u.t())?.reshape(x.shape())?.as_dtype(x.dtype())?;
                    Ok(vec![dx, du, dv, dy.as_dtype(base.dtype())?])
                },
            )?;
            Ok(op
                .apply(&[x.clone(), u.clone(), v.clone(), base.clone()])?
                .remove(0))
        }
    }
}
