//! The chunked gated delta rule at the workload's shapes: one sequence of 6144 tokens, 16 heads
//! of 128, bf16 queries, keys and values, the normalisation inside, an f32 initial
//! state; forward, and forward and backward with every input's gradient through the output and
//! the final state, as a training step needs them. Against public baselines, each timed here:
//!
//! - MLX's operations differentiated by MLX throughout, the scan from chunk to chunk and the
//!   chunks' inverses included;
//! - MLX's operations with the scan's and the inverses' backward passes written out (dash-mlx's
//!   reference, without its kernels);
//! - for the forward pass, mlx-lm's kernel over the whole sequence (`common/mlx_lm.rs`), as mlx-lm
//!   runs a prompt's prefill: one token at a time, after its own normalisation. It has no backward
//!   pass, and mlx-lm's operations for one (its token-by-token loop) keep every token's state.
//!
//! MLX runs f32 matmuls in TF32 on GPUs that have it unless `MLX_ENABLE_TF32=0`; the rule computes
//! in f32 inside, so the setting changes its error, and these timings are taken both ways:
//!
//!   cargo run --release -p dash-mlx --example gated_delta_chunk_bench
//!   MLX_ENABLE_TF32=0 cargo run --release -p dash-mlx --example gated_delta_chunk_bench

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gated_delta_chunk_bench runs on Apple silicon only");
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
    use dash_mlx::gated_delta::chunk::{
        ChunkGatedDeltaOptions, chunk_gated_delta_rule_autograd, chunk_gated_delta_rule_parts,
    };
    use dash_mlx::gated_delta::recurrent::{GatedDeltaStepOptions, gated_delta_step};
    use mlx_rs::error::Result;
    use mlx_rs::ops::swap_axes;
    use mlx_rs::transforms::{eval, fallible_vjp};
    use mlx_rs::{Array, Dtype};

    const T: i32 = 6144;
    const H: i32 = 16;
    const D: i32 = 128;

    type Rule = Box<
        dyn Fn(
            &Array,
            &Array,
            &Array,
            &Array,
            &Array,
            Option<&Array>,
            Option<&Array>,
            ChunkGatedDeltaOptions,
        ) -> Result<(Array, Option<Array>)>,
    >;

    /// The rule with the inverses' kernel and the scan's kernels each on or off; off, that part
    /// runs as MLX's operations with the same written-out backward pass.
    fn parts(kernels: [bool; 2]) -> Rule {
        Box::new(move |q, k, v, g, beta, h0, cu, options| {
            chunk_gated_delta_rule_parts(q, k, v, g, beta, h0, cu, options, kernels)
        })
    }

    pub fn run() -> Result<()> {
        let bf16 = Dtype::Bfloat16;
        let q = normal(&[1, T, H, D], 1.0, bf16, 0)?;
        let k = normal(&[1, T, H, D], 1.0, bf16, 1)?;
        let v = normal(&[1, T, H, D], 1.0, bf16, 2)?;
        // The model's distributions: g = -softplus(x) * 0.4 <= 0, beta = sigmoid(x).
        let g = mlx_rs::nn::softplus(normal(&[1, T, H], 1.0, Dtype::Float32, 3)?)?
            .multiply(Array::from_f32(-0.4))?;
        let beta =
            mlx_rs::ops::sigmoid(normal(&[1, T, H], 1.0, Dtype::Float32, 4)?)?.as_dtype(bf16)?;
        let h0 = normal(&[1, H, D, D], 0.1, Dtype::Float32, 5)?;
        let d_out = normal(&[1, T, H, D], 1.0, bf16, 6)?;
        let d_state = normal(&[1, H, D, D], 1.0, Dtype::Float32, 7)?;
        eval([&q, &k, &v, &g, &beta, &h0, &d_out, &d_state])?;
        let tf32 = std::env::var("MLX_ENABLE_TF32").unwrap_or_else(|_| "default".into());
        let shape = format!("B 1, T {T}, H {H}, Dk = Dv = {D}, bf16, MLX_ENABLE_TF32={tf32}");

        println!("| operation | variant | shape | median | peak above the inputs |");
        println!("|---|---|---|---|---|");
        let options = ChunkGatedDeltaOptions::default()
            .with_qk_l2norm(true)
            .with_final_state(true);
        let primals = [
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            h0.clone(),
        ];
        // Each kernel against the same part as MLX's operations, the other part held fixed.
        let rules: [(&str, Rule); 5] = [
            ("dash-mlx kernels", parts([true, true])),
            (
                "inverses' kernel; scan as MLX's operations",
                parts([true, false]),
            ),
            (
                "scan's kernels; inverses as MLX's operations",
                parts([false, true]),
            ),
            (
                "MLX's operations, backward passes written out",
                parts([false, false]),
            ),
            (
                "MLX's operations through autograd",
                Box::new(chunk_gated_delta_rule_autograd),
            ),
        ];
        for (name, rule) in rules {
            let call = |a: &[Array]| -> Result<Vec<Array>> {
                let (out, state) = rule(
                    &a[0],
                    &a[1],
                    &a[2],
                    &a[3],
                    &a[4],
                    Some(&a[5]),
                    None,
                    options,
                )?;
                Ok(vec![out, state.expect("asked for")])
            };
            let forward = || call(&primals);
            report(
                "gated delta rule, forward",
                name,
                &shape,
                measure(forward, 3, 10)?,
            );
            let backward = || {
                let (mut out, grads) =
                    fallible_vjp(call, &primals, &[d_out.clone(), d_state.clone()])?;
                out.extend(grads);
                Ok(out)
            };
            let measured = measure(backward, 2, 5)?;
            report(
                "gated delta rule, forward + backward",
                name,
                &shape,
                measured,
            );
        }

        // mlx-lm's prefill: its normalisation and decay, then its kernel over every token. Its
        // state is [B, H, Dv, Dk], transposed outside the timing.
        let lm_state = swap_axes(&h0, -1, -2)?.contiguous()?;
        eval([&lm_state])?;
        let prefill = || {
            let [q, k, decay] = mlx_lm::prepare(&q, &k, &g)?;
            mlx_lm::gated_delta_kernel(&q, &k, &v, &decay, &beta, &lm_state)
        };
        let name = "mlx-lm's kernel, one token at a time";
        report(
            "gated delta rule, forward",
            name,
            &shape,
            measure(prefill, 3, 10)?,
        );

        // dash-mlx's step over the sequence as one packed sequence of T tokens: the same
        // arithmetic, one token at a time, in dash-ops' layouts.
        let rows = |x: &Array| {
            let mut shape = x.shape()[1..].to_vec();
            shape[0] = T;
            x.reshape(&shape)
        };
        let (qs, ks, vs, gs, betas) = (rows(&q)?, rows(&k)?, rows(&v)?, rows(&g)?, rows(&beta)?);
        let cu = Array::from_slice(&[0, T], &[2]);
        eval([&qs, &ks, &vs, &gs, &betas, &cu])?;
        let steps = || {
            let options = GatedDeltaStepOptions::default().with_qk_l2norm(true);
            let (out, state) =
                gated_delta_step(&qs, &ks, &vs, &gs, &betas, &h0, Some(&cu), options)?;
            Ok(vec![out, state])
        };
        let name = "dash-mlx's step kernel, one token at a time";
        report(
            "gated delta rule, forward",
            name,
            &shape,
            measure(steps, 3, 10)?,
        );
        Ok(())
    }
}
