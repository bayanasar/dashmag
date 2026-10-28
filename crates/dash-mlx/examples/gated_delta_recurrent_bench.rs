//! A decoding step of the gated delta rule at the workload's shapes: batch 32, Qwen3.5-0.8B's 16
//! heads of 128, bf16 queries, keys and values, an f32 state; against mlx-lm's kernel
//! (`common/mlx_lm.rs`), the public path MLX applications decode with. Timed as a model's step
//! runs it, 18 layers each with its own state in one evaluation, reported per layer:
//!
//! - with the normalisation inside: dash-mlx's kernel, which normalises queries and keys itself,
//!   against mlx-lm's preparation (`rms_norm` on each, the decay's exponential) and its kernel;
//! - the two kernels alone, on queries and keys normalised beforehand;
//! - four tokens a sequence in one call (draft tokens to verify, a tool's reply): dash-mlx's
//!   packed step against mlx-lm's kernel over four tokens.
//!
//!   cargo run --release -p dash-mlx --example gated_delta_recurrent_bench

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gated_delta_recurrent_bench runs on Apple silicon only");
}

#[cfg(target_os = "macos")]
fn main() -> mlx_rs::error::Result<()> {
    bench::run()
}

#[cfg(target_os = "macos")]
#[path = "common/mod.rs"]
mod common;

#[cfg(target_os = "macos")]
#[path = "common/mlx_lm.rs"]
mod mlx_lm;

#[cfg(target_os = "macos")]
mod bench {
    use crate::common::{measure, normal, report};
    use crate::mlx_lm;
    use dash_mlx::gated_delta::recurrent::{GatedDeltaStepOptions, gated_delta_step};
    use mlx_rs::error::Result;
    use mlx_rs::ops::{arange, swap_axes};
    use mlx_rs::transforms::eval;
    use mlx_rs::{Array, Dtype};

    const B: i32 = 32;
    const H: i32 = 16;
    const D: i32 = 128;
    /// Qwen3.5-0.8B's gated delta layers.
    const LAYERS: usize = 18;

    /// The inputs of `tokens` tokens of each of `B` sequences: dash-mlx's packed rows
    /// `[B tokens, H, D]` and mlx-lm's `[B, tokens, H, D]` hold the same values.
    struct Inputs {
        q: Array,
        k: Array,
        v: Array,
        g: Array,
        beta: Array,
    }

    impl Inputs {
        fn new(tokens: i32) -> Result<Self> {
            let bf16 = Dtype::Bfloat16;
            let rows = B * tokens;
            Ok(Self {
                q: normal(&[rows, H, D], 1.0, bf16, 0)?,
                k: normal(&[rows, H, D], 1.0, bf16, 1)?,
                v: normal(&[rows, H, D], 1.0, bf16, 2)?,
                g: mlx_rs::nn::softplus(normal(&[rows, H], 1.0, Dtype::Float32, 3)?)?
                    .multiply(Array::from_f32(-0.4))?,
                beta: mlx_rs::ops::sigmoid(normal(&[rows, H], 1.0, Dtype::Float32, 4)?)?
                    .as_dtype(bf16)?,
            })
        }

        /// mlx-lm's layout: a batch axis in front of the tokens.
        fn mlx_lm(x: &Array, tokens: i32) -> Result<Array> {
            let mut shape = vec![B, tokens];
            shape.extend_from_slice(&x.shape()[1..]);
            x.reshape(&shape)
        }
    }

    pub fn run() -> Result<()> {
        let states = (0..LAYERS)
            .map(|i| normal(&[B, H, D, D], 0.1, Dtype::Float32, 10 + i as u64))
            .collect::<Result<Vec<_>>>()?;
        // mlx-lm keeps the state as [B, H, Dv, Dk]: the same values, transposed outside the
        // timing.
        let mlx_lm_states = states
            .iter()
            .map(|s| swap_axes(s, -1, -2)?.contiguous())
            .collect::<Result<Vec<_>>>()?;
        eval(states.iter().chain(&mlx_lm_states))?;

        println!("| operation | variant | shape | median | peak above the inputs |");
        println!("|---|---|---|---|---|");
        let per_layer = |(ms, mib): (f64, f64)| (ms / LAYERS as f64, mib);
        for tokens in [1, 4] {
            let x = Inputs::new(tokens)?;
            eval([&x.q, &x.k, &x.v, &x.g, &x.beta])?;
            let shape = format!(
                "B {B}, {tokens} token(s) each, H {H}, Dk = Dv = {D}, bf16, {LAYERS} layers in one evaluation"
            );
            let cu = (tokens > 1)
                .then(|| arange::<_, i32>(0, B * tokens + 1, tokens))
                .transpose()?;
            let dash = |l2norm: bool, q: &Array, k: &Array| {
                let options = GatedDeltaStepOptions::default().with_qk_l2norm(l2norm);
                let mut outs = Vec::new();
                for state in &states {
                    let (out, next) =
                        gated_delta_step(q, k, &x.v, &x.g, &x.beta, state, cu.as_ref(), options)?;
                    outs.extend([out, next]);
                }
                Ok(outs)
            };
            let lm = |prepare: bool, q: &Array, k: &Array, decay: &Array| {
                let mut outs = Vec::new();
                for state in &mlx_lm_states {
                    let [q, k, decay] = match prepare {
                        true => mlx_lm::prepare(q, k, decay)?,
                        false => [q.clone(), k.clone(), decay.clone()],
                    };
                    let lay = |a: &Array| Inputs::mlx_lm(a, tokens);
                    outs.extend(mlx_lm::gated_delta_kernel(
                        &lay(&q)?,
                        &lay(&k)?,
                        &lay(&x.v)?,
                        &lay(&decay)?,
                        &lay(&x.beta)?,
                        state,
                    )?);
                }
                Ok(outs)
            };
            let op = "gated delta rule, decoding step, per layer";
            let measured = per_layer(measure(|| dash(true, &x.q, &x.k), 5, 20)?);
            report(op, "dash-mlx kernel, normalising inside", &shape, measured);
            let measured = per_layer(measure(|| lm(true, &x.q, &x.k, &x.g), 5, 20)?);
            report(op, "mlx-lm: rms_norm, exp and its kernel", &shape, measured);
            // The kernels alone, on inputs prepared as each takes them, outside the timing.
            let [q, k, decay] = mlx_lm::prepare(&x.q, &x.k, &x.g)?;
            eval([&q, &k, &decay])?;
            let scale = Array::from_f32((D as f32).sqrt()).as_dtype(q.dtype())?;
            let q_unit = q.multiply(&scale)?;
            eval([&q_unit])?;
            let dash_alone = || {
                let options = GatedDeltaStepOptions::default();
                let mut outs = Vec::new();
                for state in &states {
                    let (out, next) = gated_delta_step(
                        &q_unit,
                        &k,
                        &x.v,
                        &x.g,
                        &x.beta,
                        state,
                        cu.as_ref(),
                        options,
                    )?;
                    outs.extend([out, next]);
                }
                Ok(outs)
            };
            let measured = per_layer(measure(dash_alone, 5, 20)?);
            report(op, "dash-mlx kernel alone", &shape, measured);
            let measured = per_layer(measure(|| lm(false, &q, &k, &decay), 5, 20)?);
            report(op, "mlx-lm kernel alone", &shape, measured);
        }
        Ok(())
    }
}
