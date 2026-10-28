//! The causal depthwise conv1d at the workload's shapes: Qwen3.5-0.8B's 6144 channels of query,
//! key and value, kernel width 4, SiLU, no bias, bf16; a 6144-step sequence continued from a
//! state, its gradient taken with respect to the input and the state (the weight is frozen in
//! training); and one decoding step at batch 32. dash-mlx's kernels against the operations an
//! MLX application runs today, mlx-lm's: the state's rows prepended, MLX's `conv1d`, SiLU.
//!
//!   cargo run --release -p dash-mlx --example conv1d_bench

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("conv1d_bench runs on Apple silicon only");
}

#[cfg(target_os = "macos")]
fn main() -> mlx_rs::error::Result<()> {
    bench::run()
}

#[cfg(target_os = "macos")]
#[path = "common/mod.rs"]
mod common;

#[cfg(target_os = "macos")]
mod bench {
    use crate::common::{measure, normal, report};
    use dash_mlx::conv1d::{
        Activation, CausalConv1dOptions, CausalConv1dStepOptions, causal_conv1d,
        causal_conv1d_update,
    };
    use mlx_rs::error::Result;
    use mlx_rs::ops::indexing::TryIndexOp;
    use mlx_rs::ops::{concatenate, conv1d, expand_dims};
    use mlx_rs::transforms::{eval, fallible_vjp};
    use mlx_rs::{Array, Dtype};

    const T: i32 = 6144;
    const C: i32 = 6144;
    const K: i32 = 4;
    const BATCH: i32 = 32;
    /// Qwen3.5-0.8B's gated delta layers, each with a conv1d.
    const LAYERS: usize = 18;

    type Conv = fn(&Array, &Array, &Array) -> Result<Array>;
    type Step = fn(&Array, &Array, &Array) -> Result<[Array; 2]>;

    /// dash-mlx, from its f32 state `[B, K - 1, C]`.
    fn dash_mlx_conv(x: &Array, state: &Array, weight: &Array) -> Result<Array> {
        let options = CausalConv1dOptions::default().with_activation(Activation::Silu);
        Ok(causal_conv1d(x, weight, None, Some(state), None, options)?.0)
    }

    /// mlx-lm's GatedDeltaNet: its state `[B, K - 1, C]` in the model's dtype prepended, MLX's
    /// depthwise `conv1d` in that dtype, then SiLU.
    fn mlx_lm_conv(x: &Array, state: &Array, weight: &Array) -> Result<Array> {
        let input = concatenate(&[state.clone(), x.clone()], 1)?;
        mlx_rs::nn::silu(conv1d(&input, expand_dims(weight, 2)?, 1, 0, 1, C)?)
    }

    fn dash_mlx_step(token: &Array, state: &Array, weight: &Array) -> Result<[Array; 2]> {
        let options = CausalConv1dStepOptions::default().with_activation(Activation::Silu);
        let (out, next) = causal_conv1d_update(token, weight, None, state, options)?;
        Ok([out, next])
    }

    /// mlx-lm's decoding step: the same convolution over the state's rows and the token, and the
    /// next state cut from the same concatenation.
    fn mlx_lm_step(token: &Array, state: &Array, weight: &Array) -> Result<[Array; 2]> {
        let input = concatenate(&[state.clone(), expand_dims(token, 1)?], 1)?;
        let out = mlx_rs::nn::silu(conv1d(&input, expand_dims(weight, 2)?, 1, 0, 1, C)?)?;
        Ok([out, input.try_index((.., 1.., ..))?.contiguous()?])
    }

    pub fn run() -> Result<()> {
        let dtype = Dtype::Bfloat16;
        let x = normal(&[1, T, C], 1.0, dtype, 0)?;
        // dash-mlx's states are f32; mlx-lm keeps them in the model's dtype.
        let state = normal(&[1, K - 1, C], 1.0, dtype, 1)?.as_dtype(Dtype::Float32)?;
        let weight = normal(&[C, K], 0.5, dtype, 2)?;
        let grad = normal(&[1, T, C], 1.0, dtype, 3)?;
        let mlx_lm_state = state.as_dtype(dtype)?;
        eval([&x, &state, &weight, &grad, &mlx_lm_state])?;

        println!("| operation | variant | shape | median | peak above the inputs |");
        println!("|---|---|---|---|---|");
        let shape = format!("B 1, T {T}, C {C}, K {K}, bf16, SiLU");
        let paths: [(&str, Conv, &Array); 2] = [
            ("dash-mlx kernels", dash_mlx_conv, &state),
            ("mlx-lm's operations", mlx_lm_conv, &mlx_lm_state),
        ];
        for (name, conv, state) in paths {
            let forward = || Ok(vec![conv(&x, state, &weight)?]);
            report("conv1d forward", name, &shape, measure(forward, 3, 10)?);
            // The weight is captured: a constant to MLX, so only the input and the state get
            // gradients, as in training. The output is evaluated with them, as training needs it.
            let backward = || {
                let f = |a: &[Array]| Ok(vec![conv(&a[0], &a[1], &weight)?]);
                let (mut out, grads) =
                    fallible_vjp(f, &[x.clone(), state.clone()], std::slice::from_ref(&grad))?;
                out.extend(grads);
                Ok(out)
            };
            let measured = measure(backward, 3, 10)?;
            report("conv1d forward + backward", name, &shape, measured);
        }

        // One step, evaluated on its own: mostly the cost of evaluating and waiting. Then 18
        // steps, one per delta layer with its own state, in one evaluation as a model's decoding
        // step runs them, reported per layer.
        let token = normal(&[BATCH, C], 1.0, dtype, 4)?;
        let states = (0..LAYERS)
            .map(|i| normal(&[BATCH, K - 1, C], 1.0, dtype, 100 + i as u64))
            .collect::<Result<Vec<_>>>()?;
        let f32_states = states
            .iter()
            .map(|s| s.as_dtype(Dtype::Float32))
            .collect::<Result<Vec<_>>>()?;
        eval(states.iter().chain(&f32_states).chain([&token]))?;
        let shape = format!("B {BATCH}, C {C}, K {K}, bf16, SiLU");
        let layers_shape = format!("{shape}, {LAYERS} in one evaluation");
        let per_layer = |(ms, mib): (f64, f64)| (ms / LAYERS as f64, mib);
        let steps: [(&str, Step, &[Array]); 2] = [
            ("dash-mlx kernel", dash_mlx_step, &f32_states),
            ("mlx-lm's operations", mlx_lm_step, &states),
        ];
        for (name, step, states) in steps {
            let one = || Ok(step(&token, &states[0], &weight)?.to_vec());
            report("conv1d decoding step", name, &shape, measure(one, 10, 50)?);
            let all = || {
                let mut outs = Vec::new();
                for state in states {
                    outs.extend(step(&token, state, &weight)?);
                }
                Ok(outs)
            };
            let measured = per_layer(measure(all, 5, 20)?);
            report(
                "conv1d decoding step, per layer",
                name,
                &layers_shape,
                measured,
            );
        }
        Ok(())
    }
}
