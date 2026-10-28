//! The chosen tokens' log-probabilities and entropies at the workload's shapes: 3072 rows of 1024
//! against a frozen bf16 head of 248,320 tokens, forward, and forward and backward with the
//! gradient to the hidden states. Against MLX's operations as mlx-lm's trainer forms its loss,
//! the whole logits held, then log-softmax, timed in the same process. The f32 logits run in TF32
//! where MLX's default allows it (exact from bf16 inputs either way); timed both ways:
//!
//!   cargo run --release -p dash-mlx --example logprob_bench
//!   MLX_ENABLE_TF32=0 cargo run --release -p dash-mlx --example logprob_bench

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("logprob_bench runs on Apple silicon only");
}

#[cfg(target_os = "macos")]
fn main() -> mlx_rs::error::Result<()> {
    bench::run()
}

#[cfg(target_os = "macos")]
#[path = "common/mod.rs"]
mod common;

#[cfg(target_os = "macos")]
#[allow(non_snake_case)]
mod bench {
    use crate::common::{measure, normal, report};
    use dash_mlx::logprob::{LogprobOptions, chosen_logprob, vocab_chunk};
    use mlx_rs::error::Result;
    use mlx_rs::ops::{ones, softmax_axis, sum_axis, zeros};
    use mlx_rs::stop_gradient;
    use mlx_rs::transforms::{eval, fallible_vjp};
    use mlx_rs::{Array, Dtype};

    const D: i32 = 1024;

    /// The workload's sizes, or `N` and `V` from the environment.
    fn size(name: &str, default: i32) -> i32 {
        std::env::var(name).map_or(default, |v| v.parse().expect("an integer"))
    }

    pub fn run() -> Result<()> {
        let (n, v) = (size("N", 3072), size("V", 248_320));
        let (N, V) = (n, v);
        let bf16 = Dtype::Bfloat16;
        let weight = normal(&[V, D], 0.02, bf16, 1)?;
        let hidden = normal(&[N, D], 1.0, bf16, 2)?;
        let ids: Vec<i32> = (0..N).map(|i| (i * 7919) % V).collect();
        let targets = Array::from_slice(&ids, &[N]);
        eval([&weight, &hidden, &targets])?;
        let tf32 = std::env::var("MLX_ENABLE_TF32").unwrap_or_else(|_| "default".into());
        let shape = format!("n {N}, D {D}, V {V}, bf16, MLX_ENABLE_TF32={tf32}");
        let op = "log-prob and entropy";
        println!("| operation | variant | shape | median | peak above the inputs |");
        println!("|---|---|---|---|---|");

        let chunks: Vec<usize> = match std::env::var("CHUNKS") {
            Ok(list) => list.split(',').map(|c| c.parse().unwrap()).collect(),
            Err(_) => vec![0],
        };
        for chunk in chunks {
            let options = LogprobOptions {
                vocab_chunk: chunk,
                ..LogprobOptions::default()
            };
            let name = format!("dash-mlx, vocabulary chunks of {}", vocab_chunk(options));
            let forward = || {
                let out = chosen_logprob(&hidden, &weight, &targets, &[], options)?;
                Ok(vec![out.logprob, out.entropy])
            };
            report(
                &format!("{op}, forward"),
                &name,
                &shape,
                measure(forward, 2, 5)?,
            );
            let both = || {
                let f = |a: &[Array]| -> Result<Vec<Array>> {
                    let out = chosen_logprob(&a[0], &weight, &targets, &[], options)?;
                    Ok(vec![out.logprob, stop_gradient(&out.entropy)?])
                };
                let (mut out, grads) = fallible_vjp(
                    f,
                    std::slice::from_ref(&hidden),
                    &[ones::<f32>(&[N])?, zeros::<f32>(&[N])?],
                )?;
                out.extend(grads);
                Ok(out)
            };
            report(
                &format!("{op}, forward + backward"),
                &name,
                &shape,
                measure(both, 2, 5)?,
            );
        }

        // MLX's operations over the whole logits.
        let wt = weight.t();
        let columns = targets.expand_dims(-1)?;
        let whole = |h: &Array| -> Result<Vec<Array>> {
            let logits = h.matmul(&wt)?.as_dtype(Dtype::Float32)?;
            let lse = logits.logsumexp_axis(-1, true)?;
            let logprob = logits.take_along_axis(&columns, -1)?.subtract(&lse)?;
            let p = softmax_axis(&logits, -1, None)?;
            let entropy = lse.subtract(sum_axis(&p.multiply(&logits)?, -1, true)?)?;
            Ok(vec![logprob.squeeze()?, entropy.squeeze()?])
        };
        let name = "MLX's operations, the whole logits";
        report(
            &format!("{op}, forward"),
            name,
            &shape,
            measure(|| whole(&hidden), 2, 5)?,
        );
        let both = || {
            let f = |a: &[Array]| -> Result<Vec<Array>> {
                let out = whole(&a[0])?;
                Ok(vec![out[0].clone(), stop_gradient(&out[1])?])
            };
            let (mut out, grads) = fallible_vjp(
                f,
                std::slice::from_ref(&hidden),
                &[ones::<f32>(&[N])?, zeros::<f32>(&[N])?],
            )?;
            out.extend(grads);
            Ok(out)
        };
        report(
            &format!("{op}, forward + backward"),
            name,
            &shape,
            measure(both, 2, 5)?,
        );
        Ok(())
    }
}
