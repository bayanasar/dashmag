//! Sampling the next token at the decoding shape: 32 rows of bf16 logits over 248,320 tokens, a
//! step of a batch of 32 sequences. Greedy, the temperature alone, top-k 50, top-p 0.9, and both
//! at temperature 0.8; against mlx-lm's sampler as MLX's operations (`make_sampler`'s steps:
//! `logits - logsumexp`, top-k by `argpartition`, top-p by `argsort` and `cumsum` at temperature
//! 1, `random.categorical` of the logits over the temperature), the drawn token's log-probability
//! gathered as dash-mlx returns it, timed in the same process. Each call takes the next offset, as
//! a decoding loop does.
//!
//!   cargo run --release -p dash-mlx --example sample_bench

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("sample_bench runs on Apple silicon only");
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
    use dash_mlx::sample::{SampleOptions, sample};
    use mlx_rs::error::Result;
    use mlx_rs::ops::indexing::TryIndexOp;
    use mlx_rs::ops::{argpartition_axis, argsort_axis, cumsum, full, select, sort_axis};
    use mlx_rs::transforms::eval;
    use mlx_rs::{Array, Dtype};
    use std::cell::Cell;

    /// mlx-lm's sampler on `logits` `[B, V]`: the token and its log-probability under the model.
    fn mlx_lm(logits: &Array, options: &SampleOptions, step: u64) -> Result<Vec<Array>> {
        let logits = logits.as_dtype(Dtype::Float32)?;
        let logprobs = logits.subtract(logits.logsumexp_axis(-1, true)?)?;
        let inf = Array::from_f32(f32::NEG_INFINITY);
        if options.temperature == 0.0 {
            let tokens = mlx_rs::ops::indexing::argmax_axis(&logprobs, -1, true)?;
            return Ok(vec![tokens.clone(), logprobs.take_along_axis(&tokens, -1)?]);
        }
        let mut masked = logprobs.clone();
        if options.top_k > 0 {
            // apply_top_k: every token past the k largest to -inf.
            let k = options.top_k as i32;
            let v = masked.shape()[1];
            let order = argpartition_axis(masked.negative()?, k - 1, -1)?;
            let rest = order.try_index((.., k..v))?;
            masked = masked.put_along_axis(&rest, full::<f32>(rest.shape(), &inf)?, -1)?;
        }
        if options.top_p < 1.0 {
            // apply_top_p: the cumulative probability in ascending order, mapped back.
            let probs = masked.exp()?;
            let sorted = argsort_axis(&masked, -1)?;
            let sorted_probs = probs.take_along_axis(&sorted, -1)?;
            let cumulative = cumsum(&sorted_probs, -1, None, None)?;
            let back = argsort_axis(&sorted, -1)?;
            let cumulative = cumulative.take_along_axis(&back, -1)?;
            let keep = cumulative.gt(Array::from_f32(1.0 - options.top_p))?;
            masked = select(&keep, &masked, &inf)?;
        }
        let key = mlx_rs::random::key(step)?;
        let scaled = masked.multiply(Array::from_f32(1.0 / options.temperature))?;
        let tokens = mlx_rs::random::categorical(&scaled, -1, None, &key)?.expand_dims(-1)?;
        Ok(vec![tokens.clone(), logprobs.take_along_axis(&tokens, -1)?])
    }

    pub fn run() -> Result<()> {
        let (b, v) = (32, 248_320);
        let logits = normal(&[b, v], 2.0, Dtype::Bfloat16, 1)?;
        eval([&logits])?;
        let shape = format!("B {b}, V {v}, bf16");
        // Without a warm-up the first case's times varied twofold from run to run (measured).
        for offset in 0..200 {
            let warm = SampleOptions {
                offset,
                ..SampleOptions::default()
            };
            let out = sample(&logits, &warm)?;
            eval([&out.tokens])?;
        }
        println!("| operation | variant | shape | median | peak above the inputs |");
        println!("|---|---|---|---|---|");
        let cases = [
            ("greedy", SampleOptions::greedy()),
            ("temperature 1", SampleOptions::default()),
            (
                "top-k 50",
                SampleOptions {
                    top_k: 50,
                    ..SampleOptions::default()
                },
            ),
            (
                "top-p 0.9",
                SampleOptions {
                    top_p: 0.9,
                    ..SampleOptions::default()
                },
            ),
            (
                "top-k 50, top-p 0.9, temperature 0.8",
                SampleOptions {
                    top_k: 50,
                    top_p: 0.9,
                    temperature: 0.8,
                    ..SampleOptions::default()
                },
            ),
        ];
        // The sort the cuts take, alone: the row as f32, sorted in descending order.
        let z = logits.as_dtype(Dtype::Float32)?;
        eval([&z])?;
        let sort = || Ok(vec![sort_axis(z.negative()?, 1)?.negative()?]);
        report(
            "the row sorted in descending order, alone",
            "MLX's sort_axis on f32",
            &shape,
            measure(sort, 5, 20)?,
        );
        for (label, options) in cases {
            let step = Cell::new(0u64);
            let dash = || {
                step.set(step.get() + 1);
                let options = SampleOptions {
                    offset: step.get(),
                    ..options.clone()
                };
                let out = sample(&logits, &options)?;
                Ok(vec![
                    out.tokens,
                    out.logprob,
                    out.tempered_logprob,
                    out.sampling_logprob,
                ])
            };
            report(
                &format!("sample, {label}"),
                "dash-mlx",
                &shape,
                measure(dash, 5, 20)?,
            );
            let baseline = || {
                step.set(step.get() + 1);
                mlx_lm(&logits, &options, step.get())
            };
            report(
                &format!("sample, {label}"),
                "mlx-lm's sampler as MLX's operations",
                &shape,
                measure(baseline, 5, 20)?,
            );
        }
        Ok(())
    }
}
