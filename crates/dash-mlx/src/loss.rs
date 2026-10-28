//! Cross-entropy over a large vocabulary straight from the hidden states and the output weight,
//! without the n x V logits: dash-ops' `loss` on MLX.
//!
//! The operation, its options (reductions, ignore index, label smoothing, z-loss, masked ids, the
//! entropy), its results, their definition and their gradient are dash-ops' (its `loss` module
//! documents them). [`linear_cross_entropy`] returns the loss, each row's log-probability of its
//! target and, if asked, its entropy, all f32; gradients reach `hidden` and `weight`, an output
//! head training through it.
//!
//! On MLX:
//!
//! - The forward pass is [`crate::logprob`]'s pass over the vocabulary, chunk by chunk and block
//!   by block of rows, its chunk kernel keeping per row the maximum, the sum of `exp(z - m)` with
//!   and without the target's term, the entropy's sum, the target's logit and the sum of the
//!   logits; then each row's loss as dash-ops forms it.
//! - The backward pass is a custom VJP that walks the vocabulary again, as dash-ops does when the
//!   weight trains or the entropy is asked for: per row the coefficients of dash-ops' gradient
//!   pass from the upstream gradients and the forward pass's rows, then each chunk's logits
//!   recomputed, a kernel forming the chunk's gradient of the logits from them (`exp(z - m) / s`
//!   for the softmax, the target's term from `1 - p(target)` summed over the other tokens), and
//!   two matmuls: the chunk's part of the gradient of `hidden` and its rows of the gradient of
//!   `weight`. dash-ops' other backward pass, which accumulates the gradient of `hidden` in the
//!   forward pass when the weight is frozen, would save MLX no pass: the forward pass does not
//!   know whether a gradient will follow.
//! - MLX does not tell a VJP which inputs it differentiates. Where the weight's gradient is not
//!   asked for, its products are never evaluated, and the pass costs what the gradient of
//!   `hidden` alone costs. Where it is, the two products of each chunk must run before the next
//!   chunk's, or MLX, evaluating the gradient of `hidden` through every chunk first, holds every
//!   chunk's gradient of the logits until the weight's products run: `n x V` f32 (measured, +5205
//!   MiB against +2638 at the bench's shapes). A gate tells the two cases apart: an operation
//!   whose inputs are the weight and, through `stop_gradient`, one of the loss's outputs, and
//!   whose output, 0, is added to each output. MLX runs its VJP only when it differentiates the
//!   weight, and before the loss's, which then makes each chunk's two products wait for each
//!   other (MLX's `depends`). Neither changes a value, only the order MLX evaluates in.
//! - Every product is f32, the chunks of a bf16 or f16 weight cast to f32 (bf16 values are exact
//!   in f32 and TF32); the f32 bounds assume `MLX_ENABLE_TF32=0`, as [`crate::logprob`]'s. The
//!   gradients come back in the dtypes of `hidden` and `weight`, rounded once.

use crate::kernel::{Launch, launch};
use crate::logprob::{State, Vocabulary};
use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::TemplateArg;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{arange, broadcast_to, concatenate, select, zeros};
use mlx_rs::transforms::custom_vjp;
use mlx_rs::{Array, Dtype};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::Ordering;

pub use dash_contract::loss::{CrossEntropyOptions, Reduction};

/// What [`linear_cross_entropy`] returns, all f32.
#[derive(Debug, Clone)]
pub struct CrossEntropy {
    /// The loss: `[1]` for [`Reduction::Mean`] and [`Reduction::Sum`], `[n]` for
    /// [`Reduction::None`].
    pub loss: Array,
    /// `log p(target)` per row, `[n]`: `-inf` where the target is masked or out of range, 0 where
    /// the row is ignored.
    pub logprob: Array,
    /// The entropy of each row's distribution over the unmasked vocabulary, `[n]` (0 where the
    /// row is ignored), when [`CrossEntropyOptions::entropy`] asks for it.
    pub entropy: Option<Array>,
}

/// Counters tests read to tell which passes ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static FORWARD_PASSES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static GRADIENT_PASSES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static INTERLEAVED_PASSES: AtomicUsize = AtomicUsize::new(0);

    /// Passes over the vocabulary for the loss, one per block of rows.
    pub fn forward_passes() -> usize {
        FORWARD_PASSES.load(Ordering::SeqCst)
    }

    /// Passes over the vocabulary for the gradients, one per block of rows.
    pub fn gradient_passes() -> usize {
        GRADIENT_PASSES.load(Ordering::SeqCst)
    }

    /// Of those, the passes that knew the weight's gradient was asked for.
    pub fn interleaved_passes() -> usize {
        INTERLEAVED_PASSES.load(Ordering::SeqCst)
    }
}

/// dash-ops' gradient of the logits for one chunk, a thread an element: 0 for a masked token,
/// `T + C` for the row's target (`tl`, -1 when the chunk does not hold it), and for every other
/// token `p A - p (z - lse) E + C` with `p = exp(z - m) / s` and `z - lse = (z - m) - ln s` (the
/// `E` term only with `ENTROPY`).
const GRADIENT: &str = "
    const uint j = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const int C = z_shape[1];
    if (int(j) >= C) {
        return;
    }
    const size_t at = size_t(row) * C + j;
    float g;
    if (MASKED && flags[j]) {
        g = 0.0f;
    } else if (int(j) == tl[row]) {
        g = target[row];
    } else {
        const float d = z[at] - m[row];
        const float p = metal::precise::exp(d) / s[row];
        g = p * softmax[row];
        if (ENTROPY) {
            g -= p * (d - metal::precise::log(s[row])) * entropy[row];
        }
        g += uniform[row];
    }
    g_out[at] = g;
";

/// Threads per threadgroup along a chunk's row in the gradient kernel.
const THREADS: i32 = 256;

fn contract(message: String) -> Exception {
    Exception::custom(format!("linear_cross_entropy: {message}"))
}

/// The per-row coefficients of the gradient pass, each `[rows]` f32 (dash-ops' `coefficient`):
/// the maximum and sum the softmax is formed from, the softmax term's factor `A`, the entropy's
/// `E`, the target's value `T + C` and the term `C` every unmasked token takes.
struct Coefficients {
    m: Array,
    s: Array,
    softmax: Array,
    entropy: Array,
    target: Array,
    uniform: Array,
}

impl Coefficients {
    fn rows(&self, (r0, r1): (usize, usize)) -> Result<Self> {
        let at = |x: &Array| x.try_index(r0 as i32..r1 as i32);
        Ok(Self {
            m: at(&self.m)?,
            s: at(&self.s)?,
            softmax: at(&self.softmax)?,
            entropy: at(&self.entropy)?,
            target: at(&self.target)?,
            uniform: at(&self.uniform)?,
        })
    }
}

/// What both passes share beyond the vocabulary: the options and the rows counted and found.
#[derive(Clone)]
struct Loss {
    vocabulary: Vocabulary,
    /// `[n]` bool: the target is not the ignore index.
    counted: Array,
    eps: f32,
    lambda: f32,
    entropy: bool,
    /// The number of unmasked ids, `K`.
    classes: usize,
}

impl Loss {
    /// The forward pass: per row `[loss, logprob, entropy, m, s, q]`, each `[n]` f32, the loss,
    /// the log-probability and the entropy 0 where the row is ignored, `q = r / s`.
    fn forward(&self, hidden: &Array, weight: &Array) -> Result<Vec<Array>> {
        let vocabulary = self.vocabulary.with_weight(weight);
        let h = hidden.as_dtype(Dtype::Float32)?;
        let mut parts: [Vec<Array>; 6] = Default::default();
        for &rows in &vocabulary.blocks {
            probe::FORWARD_PASSES.fetch_add(1, Ordering::SeqCst);
            let State {
                m,
                s,
                r,
                a,
                zt,
                zsum,
            } = vocabulary.pass(&h, rows)?;
            for (part, x) in parts.iter_mut().zip([m, s, r, a, zt, zsum]) {
                part.push(x);
            }
        }
        let [m, s, r, a, zt, zsum] = parts.map(|xs| concatenate(&xs, 0));
        let (m, s, r, a, zt, zsum) = (m?, s?, r?, a?, zt?, zsum?);
        let found = &vocabulary.valid;
        let (eps, lambda) = (self.eps, self.lambda);
        let ln_s = s.log()?;
        let lse = m.add(&ln_s)?;
        let logprob = select(
            found,
            zt.subtract(&lse)?,
            Array::from_f32(f32::NEG_INFINITY),
        )?;
        // (1 - eps) (lse - zt), +inf where the target is not seen; no term at all for eps = 1.
        let mut loss = match eps < 1.0 {
            true => select(found, lse.subtract(&zt)?, Array::from_f32(f32::INFINITY))?
                .multiply(Array::from_f32(1.0 - eps))?,
            false => zeros::<f32>(m.shape())?,
        };
        if eps != 0.0 {
            // eps (lse - mean_j z)
            let mean = zsum.multiply(Array::from_f32(1.0 / self.classes as f32))?;
            loss = loss.add(lse.subtract(mean)?.multiply(Array::from_f32(eps))?)?;
        }
        if lambda != 0.0 {
            loss = loss.add(lse.multiply(&lse)?.multiply(Array::from_f32(lambda))?)?;
        }
        let entropy = ln_s.subtract(a.divide(&s)?)?;
        let q = r.divide(&s)?;
        let zero_ignored = |x: Array| select(&self.counted, x, Array::from_f32(0.0));
        Ok(vec![
            zero_ignored(loss)?,
            zero_ignored(logprob)?,
            zero_ignored(entropy)?,
            m,
            s,
            q,
        ])
    }

    /// dash-ops' coefficients from the forward pass's outputs and the upstream gradients of the
    /// loss, the log-probability and the entropy (0 for an ignored row, whatever they hold).
    fn coefficients(&self, outputs: &[Array], cotangents: &[Array]) -> Result<Coefficients> {
        let (eps, lambda) = (self.eps, self.lambda);
        let up = |k: usize| -> Result<Array> {
            select(
                &self.counted,
                cotangents[k].as_dtype(Dtype::Float32)?,
                Array::from_f32(0.0),
            )
        };
        let (gl, gp) = (up(0)?, up(1)?);
        let [m, s, q] = [&outputs[3], &outputs[4], &outputs[5]];
        let lse = m.add(s.log()?)?;
        let twice = Array::from_f32(2.0 * lambda);
        // A = gl (1 + 2 lambda lse) - gp - ge entropy
        let mut softmax = gl.subtract(&gp)?;
        if lambda != 0.0 {
            softmax = softmax.add(gl.multiply(lse.multiply(&twice)?)?)?;
        }
        // T = gl (eps + 2 lambda lse) - ge entropy - q A - ge (1 - q) logprob, where the target
        // is seen; the gradient of the target's logit is T + C.
        let mut target = q.multiply(&softmax)?.negative()?;
        let mut gl_target = gl.multiply(Array::from_f32(eps))?;
        if lambda != 0.0 {
            gl_target = gl_target.add(gl.multiply(lse.multiply(&twice)?)?)?;
        }
        target = target.add(gl_target)?;
        let ge = match self.entropy {
            true => {
                let ge = up(2)?;
                let entropy = ge.multiply(&outputs[2])?;
                softmax = softmax.subtract(&entropy)?;
                // A takes -ge entropy, so -q A takes +q ge entropy; with T's own -ge entropy
                // that leaves -(1 - q) ge entropy, beside -ge (1 - q) logprob.
                let p_target = q.subtract(Array::from_f32(1.0))?.negative()?;
                target = target.subtract(
                    p_target
                        .multiply(entropy)?
                        .add(ge.multiply(&p_target)?.multiply(&outputs[1])?)?,
                )?;
                ge
            }
            false => zeros::<f32>(gl.shape())?,
        };
        let target = select(&self.vocabulary.valid, target, Array::from_f32(0.0))?;
        let uniform = gl.multiply(Array::from_f32(-eps / self.classes as f32))?;
        Ok(Coefficients {
            m: m.clone(),
            s: s.clone(),
            softmax,
            entropy: ge,
            target: target.add(&uniform)?,
            uniform,
        })
    }

    /// The gradient kernel on a chunk's logits `z` `[rows, C]`.
    fn gradient_kernel(
        &self,
        z: &Array,
        coef: &Coefficients,
        tl: &Array,
        chunk: (i32, i32),
    ) -> Result<Array> {
        let (rows, c) = (z.shape()[0], z.shape()[1]);
        let (flags, masked) = self.vocabulary.kernel_flags(chunk)?;
        let kernel = crate::kernel::kernel(
            "dash_mlx_loss_gradient",
            &[
                "z", "m", "s", "softmax", "entropy", "target", "uniform", "tl", "flags",
            ],
            &["g_out"],
            GRADIENT,
            "",
        )?;
        let config = Launch {
            grid: ((c + THREADS - 1) / THREADS * THREADS, rows, 1),
            thread_group: (THREADS, 1, 1),
            outputs: vec![(vec![rows, c], Dtype::Float32)],
            template: vec![
                ("MASKED".into(), TemplateArg::Bool(masked)),
                ("ENTROPY".into(), TemplateArg::Bool(self.entropy)),
            ],
            init_value: None,
        };
        let inputs = [
            z.clone(),
            coef.m.clone(),
            coef.s.clone(),
            coef.softmax.clone(),
            coef.entropy.clone(),
            coef.target.clone(),
            coef.uniform.clone(),
            tl.clone(),
            flags,
        ];
        // SAFETY: thread (j, row), those at `j >= C` returning at once, reads `z[row, j]` of
        // `z` [rows, C] (row-contiguous), `flags[j]` when MASKED (the chunk's flags, C long) and
        // row `row` of the seven `[rows]` inputs, and writes `g_out[row, j]` once; the grid
        // covers every (j, row) of [rows, C]. No input is written.
        let out = unsafe { launch(&kernel, &inputs, &config)? };
        Ok(out[0].clone())
    }

    /// The same gradient as MLX's operations: the reference.
    fn gradient_ops(
        &self,
        z: &Array,
        coef: &Coefficients,
        tl: &Array,
        chunk: (i32, i32),
    ) -> Result<Array> {
        let column = |x: &Array| x.expand_dims(-1);
        let d = z.subtract(column(&coef.m)?)?;
        let p = d.exp()?.divide(column(&coef.s)?)?;
        let mut g = p.multiply(column(&coef.softmax)?)?;
        if self.entropy {
            let dz = d.subtract(column(&coef.s.log()?)?)?;
            g = g.subtract(p.multiply(dz)?.multiply(column(&coef.entropy)?)?)?;
        }
        g = g.add(column(&coef.uniform)?)?;
        let columns = arange::<_, i32>(0, chunk.1 - chunk.0, None)?.expand_dims(0)?;
        g = select(columns.eq(column(tl)?)?, column(&coef.target)?, g)?;
        match self.vocabulary.chunk_flags(chunk)? {
            Some(flags) => select(&flags.expand_dims(0)?, Array::from_f32(0.0), g),
            None => Ok(g),
        }
    }

    /// The gradient pass: `(grad_hidden, grad_weight)` in the dtypes of `hidden` and `weight`.
    /// With `weight_wanted`, each chunk's part of the gradient of `hidden` waits for its rows of
    /// the weight's, so that each chunk's gradient of the logits is freed before the next is made.
    fn backward(
        &self,
        hidden: &Array,
        weight: &Array,
        outputs: &[Array],
        cotangents: &[Array],
        weight_wanted: bool,
    ) -> Result<[Array; 2]> {
        let vocabulary = self.vocabulary.with_weight(weight);
        let coef = self.coefficients(outputs, cotangents)?;
        let h = hidden.as_dtype(Dtype::Float32)?;
        let chunks: Vec<(i32, i32)> = vocabulary.chunks().collect();
        let mut grad_hidden = Vec::with_capacity(vocabulary.blocks.len());
        let mut grad_weight: Vec<Option<Array>> = vec![None; chunks.len()];
        for &(r0, r1) in &vocabulary.blocks {
            probe::GRADIENT_PASSES.fetch_add(1, Ordering::SeqCst);
            if weight_wanted {
                probe::INTERLEAVED_PASSES.fetch_add(1, Ordering::SeqCst);
            }
            let rows = r0 as i32..r1 as i32;
            let h = h.try_index(rows.clone())?;
            let t = vocabulary.targets.try_index(rows.clone())?;
            let valid = vocabulary.valid.try_index(rows)?;
            let coef = coef.rows((r0, r1))?;
            let mut gh: Option<Array> = None;
            for (k, &chunk) in chunks.iter().enumerate() {
                let w = weight
                    .try_index(chunk.0..chunk.1)?
                    .as_dtype(Dtype::Float32)?;
                let z = vocabulary.logits(&h, &w, chunk)?;
                let tl = Vocabulary::local_targets(&t, &valid, chunk)?;
                let g = match vocabulary.kernels {
                    true => self.gradient_kernel(&z, &coef, &tl, chunk)?,
                    false => self.gradient_ops(&z, &coef, &tl, chunk)?,
                };
                let part = g.matmul(&w)?;
                let mut sum = match gh {
                    Some(sum) => sum.add(part)?,
                    None => part,
                };
                let part = g.t().matmul(&h)?;
                let rows = match grad_weight[k].take() {
                    Some(rows) => rows.add(part)?,
                    None => part,
                };
                // Each waits for the other: whichever MLX evaluates first, a chunk's two products run
                // before the next chunk's.
                let rows = match weight_wanted {
                    true => {
                        sum = depends(&sum, &rows)?;
                        depends(&rows, &sum)?
                    }
                    false => rows,
                };
                gh = Some(sum);
                grad_weight[k] = Some(rows);
            }
            grad_hidden.push(gh.expect("at least one chunk"));
        }
        // Each chunk's rows rounded to the weight's dtype before they are joined.
        let grad_weight = grad_weight
            .into_iter()
            .map(|rows| {
                rows.expect("every chunk meets every block")
                    .as_dtype(weight.dtype())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok([
            concatenate(&grad_hidden, 0)?.as_dtype(hidden.dtype())?,
            concatenate(&grad_weight, 0)?,
        ])
    }
}

/// `a`, evaluated only once `after` is: MLX's `depends`, which mlx-rs does not wrap. Evaluation
/// order alone; the values are `a`'s.
fn depends(a: &Array, after: &Array) -> Result<Array> {
    let (inputs, dependencies) = ([a.as_ptr()], [after.as_ptr()]);
    // SAFETY: both vectors are made from live arrays, whose handles mlx-c copies; `res` and `out`
    // are new handles mlx-c fills; each handle made here is freed once, `out` by the Array it
    // becomes.
    unsafe {
        let inputs = mlx_sys::mlx_vector_array_new_data(inputs.as_ptr(), 1);
        let dependencies = mlx_sys::mlx_vector_array_new_data(dependencies.as_ptr(), 1);
        let mut res = mlx_sys::mlx_vector_array_new();
        let mut status = mlx_sys::mlx_depends(&mut res, inputs, dependencies);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(dependencies);
        let mut out = mlx_sys::mlx_array_new();
        if status == 0 {
            status = mlx_sys::mlx_vector_array_get(&mut out, res, 0);
        }
        mlx_sys::mlx_vector_array_free(res);
        let out = Array::from_ptr(out);
        match status {
            0 => Ok(out),
            _ => Err(Exception::custom(
                "linear_cross_entropy: mlx_depends failed",
            )),
        }
    }
}

/// Cross-entropy of `softmax(hidden @ weight^T)` (the `masked` ids removed) against `targets`,
/// with each row's log-probability of its target and, if asked, its entropy, without holding the
/// n x V logits; dash-ops' arguments, options and results ([module](self)).
///
/// - `hidden` `[n, D]`, f32, bf16 or f16: gets a gradient, in its dtype.
/// - `weight` `[V, D]`, f32, bf16 or f16: gets a gradient too, in its dtype.
/// - `targets` `[n]`, any integer dtype; [`CrossEntropyOptions::ignore_index`] marks rows to
///   ignore.
///
/// Errors on mismatched shapes, an empty vocabulary, a masked id `>= V` or every id masked, a
/// zero `row_chunk`, a label smoothing outside `0..=1`, or a negative or non-finite z-loss
/// coefficient: where dash-ops panics.
pub fn linear_cross_entropy(
    hidden: &Array,
    weight: &Array,
    targets: &Array,
    options: &CrossEntropyOptions,
) -> Result<CrossEntropy> {
    checked(hidden, weight, targets, options, true)
}

/// [`linear_cross_entropy`] with each chunk's state update and gradient as MLX's operations
/// rather than the kernels: the reference the tests hold the kernels to.
#[doc(hidden)]
pub fn linear_cross_entropy_reference(
    hidden: &Array,
    weight: &Array,
    targets: &Array,
    options: &CrossEntropyOptions,
) -> Result<CrossEntropy> {
    checked(hidden, weight, targets, options, false)
}

fn checked(
    hidden: &Array,
    weight: &Array,
    targets: &Array,
    options: &CrossEntropyOptions,
    kernels: bool,
) -> Result<CrossEntropy> {
    let vocabulary = Vocabulary::new(
        "linear_cross_entropy",
        hidden,
        weight,
        targets,
        &options.masked,
        options.chunks,
        kernels,
    )?;
    if vocabulary.vocab == 0 {
        return Err(contract("the vocabulary is empty".into()));
    }
    let eps = options.label_smoothing;
    if !(0.0..=1.0).contains(&eps) {
        return Err(contract(format!(
            "label_smoothing must be in 0..=1, got {eps}"
        )));
    }
    let lambda = options.z_loss;
    if !(lambda.is_finite() && lambda >= 0.0) {
        return Err(contract(format!(
            "z_loss must be finite and at least 0, got {lambda}"
        )));
    }
    let n = hidden.shape()[0];
    let counted = targets
        .as_dtype(Dtype::Int64)?
        .ne(Array::from_slice(&[options.ignore_index], &[]))?;
    let (losses, logprob, entropy) = match n {
        // No rows: nothing to walk, and the gradients are zeros of the inputs' shapes.
        0 => (
            zeros::<f32>(&[0])?,
            zeros::<f32>(&[0])?,
            zeros::<f32>(&[0])?,
        ),
        _ => {
            let loss = Loss {
                classes: vocabulary.vocab - vocabulary.masked.len(),
                vocabulary,
                counted: counted.clone(),
                eps,
                lambda,
                entropy: options.entropy,
            };
            let backward = loss.clone();
            // Whether the weight's gradient is asked for, which MLX does not tell a VJP: a gate
            // adds 0 to each output, its inputs the weight and, through `stop_gradient`, one of the
            // operation's outputs. MLX runs its VJP only when it differentiates the weight (the
            // other input carries no gradient), and before the operation's: the gate comes after
            // the operation in the graph.
            let wanted = Rc::new(Cell::new(false));
            let seen = wanted.clone();
            let op = custom_vjp(
                move |p: &[Array]| loss.forward(&p[0], &p[1]),
                move |p: &[Array], cotangents: &[Array], outputs: &[Array]| {
                    Ok(backward
                        .backward(&p[0], &p[1], outputs, cotangents, wanted.take())?
                        .to_vec())
                },
            )?;
            let gate = custom_vjp(
                |_: &[Array]| Ok(vec![Array::from_f32(0.0)]),
                move |p: &[Array], _: &[Array], _: &[Array]| {
                    seen.set(true);
                    let zero = |x: &Array| {
                        broadcast_to(Array::from_f32(0.0).as_dtype(x.dtype())?, x.shape())
                    };
                    Ok(vec![zero(&p[0])?, zero(&p[1])?])
                },
            )?;
            let out = op.apply(&[hidden.clone(), weight.clone()])?;
            let after = mlx_rs::stop_gradient(&out[0])?;
            let zero = gate.apply(&[weight.clone(), after])?.remove(0);
            let gated = |x: &Array| x.add(&zero);
            (gated(&out[0])?, gated(&out[1])?, gated(&out[2])?)
        }
    };
    let loss = match options.reduction {
        Reduction::None => losses,
        Reduction::Sum => losses.sum(None)?.reshape(&[1])?,
        Reduction::Mean => losses
            .sum(None)?
            .divide(counted.as_dtype(Dtype::Float32)?.sum(None)?)?
            .reshape(&[1])?,
    };
    Ok(CrossEntropy {
        loss,
        logprob,
        entropy: options.entropy.then_some(entropy),
    })
}
