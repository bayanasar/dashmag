//! Attention at the workload's shapes: 6144 tokens, 8 query and 2 key-value heads of 256, bf16,
//! causal; forward, and forward and backward to the queries, keys and values; and the decoding
//! step at batch 32 against 3072 cached keys, six layers in one evaluation. Against MLX's fused
//! `scaled_dot_product_attention` (its backward pass MLX's own), timed in the same process.
//!
//!   cargo run --release -p dash-mlx --example attention_bench

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("attention_bench runs on Apple silicon only");
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
    use dash_mlx::attention::{AttentionOptions, attention_decode, causal_attention};
    use mlx_rs::error::Result;
    use mlx_rs::fast::{ScaledDotProductAttentionMask, scaled_dot_product_attention};
    use mlx_rs::transforms::{eval, fallible_vjp};
    use mlx_rs::{Array, Dtype};

    pub fn run() -> Result<()> {
        let size = |name: &str, default: i32| {
            std::env::var(name).map_or(default, |v| v.parse().expect("an integer"))
        };
        let (t, hq, hkv, d) = (size("T", 6144), 8, 2, 256);
        let bf16 = Dtype::Bfloat16;
        let q = normal(&[1, t, hq, d], 1.0, bf16, 1)?;
        let k = normal(&[1, t, hkv, d], 1.0, bf16, 2)?;
        let v = normal(&[1, t, hkv, d], 1.0, bf16, 3)?;
        let d_out = normal(&[1, t, hq, d], 1.0, bf16, 4)?;
        eval([&q, &k, &v, &d_out])?;
        let scale = 1.0 / (d as f32).sqrt();
        let tf32 = std::env::var("MLX_ENABLE_TF32").unwrap_or_else(|_| "default".into());
        let shape =
            format!("T {t}, {hq} / {hkv} heads of {d}, bf16, causal, MLX_ENABLE_TF32={tf32}");
        println!("| operation | variant | shape | median | peak above the inputs |");
        println!("|---|---|---|---|---|");
        let primals = [q.clone(), k.clone(), v.clone()];
        type Attend = fn(&[Array], f32) -> Result<Vec<Array>>;
        let dash: Attend = |a, scale| Ok(vec![causal_attention(&a[0], &a[1], &a[2], scale)?]);
        // MLX's layout is [B, H, T, D]: the same arrays seen through swapped axes.
        let fused: Attend = |a, scale| {
            let swap = |x: &Array| x.swap_axes(1, 2);
            let out = scaled_dot_product_attention(
                swap(&a[0])?,
                swap(&a[1])?,
                swap(&a[2])?,
                scale,
                ScaledDotProductAttentionMask::Causal,
                None,
            )?;
            Ok(vec![out.swap_axes(1, 2)?])
        };
        for (name, attend) in [
            ("dash-mlx", dash),
            ("MLX's scaled_dot_product_attention", fused),
        ] {
            report(
                "attention, forward",
                name,
                &shape,
                measure(|| attend(&primals, scale), 3, 10)?,
            );
            let both = || {
                let (mut out, grads) = fallible_vjp(
                    |a: &[Array]| attend(a, scale),
                    &primals,
                    std::slice::from_ref(&d_out),
                )?;
                out.extend(grads);
                Ok(out)
            };
            report(
                "attention, forward + backward",
                name,
                &shape,
                measure(both, 2, 5)?,
            );
        }

        // Decoding: one query per sequence against its cache, six layers in one evaluation.
        let (b, s, layers) = (32, 3072, 6);
        let q = normal(&[b, hq, d], 1.0, bf16, 5)?;
        let caches = (0..layers)
            .map(|l| {
                Ok((
                    normal(&[b, s, hkv, d], 1.0, bf16, 10 + 2 * l)?,
                    normal(&[b, s, hkv, d], 1.0, bf16, 11 + 2 * l)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        eval(caches.iter().flat_map(|(k, v)| [k, v]).chain([&q]))?;
        let shape = format!(
            "B {b}, {hq} / {hkv} heads of {d}, {s} cached keys, bf16, {layers} layers in one evaluation"
        );
        let per_layer = |(ms, mib): (f64, f64)| (ms / layers as f64, mib);
        let step = || {
            caches
                .iter()
                .map(|(k, v)| attention_decode(&q, k, v, None, AttentionOptions::default()))
                .collect::<Result<Vec<_>>>()
        };
        report(
            "attention decoding step, per layer",
            "dash-mlx, cache [B, S, H, D] read in place",
            &shape,
            per_layer(measure(step, 5, 20)?),
        );
        let mlx_caches = caches
            .iter()
            .map(|(k, v)| {
                Ok((
                    k.swap_axes(1, 2)?.contiguous()?,
                    v.swap_axes(1, 2)?.contiguous()?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        eval(mlx_caches.iter().flat_map(|(k, v)| [k, v]))?;
        let q4 = q.expand_dims(2)?;
        let step = || {
            mlx_caches
                .iter()
                .map(|(k, v)| scaled_dot_product_attention(&q4, k, v, scale, None, None))
                .collect::<Result<Vec<_>>>()
        };
        report(
            "attention decoding step, per layer",
            "MLX's scaled_dot_product_attention, cache [B, H, S, D]",
            &shape,
            per_layer(measure(step, 5, 20)?),
        );
        Ok(())
    }
}
