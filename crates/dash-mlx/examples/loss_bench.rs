//! The fused linear cross-entropy at the workload's shapes: 3072 rows of 1024 against a bf16 head
//! of 248,320 tokens, the mean loss; forward, forward and backward with the gradient to the hidden
//! states alone (a frozen head), and with the head's gradient too. Against mlx-rs's
//! `losses::CrossEntropy` over the whole logits as mlx-lm's trainer forms its loss (the head's
//! bf16 product cast to f32), timed in the same process. The f32 products run in TF32 where MLX's
//! default allows it; timed both ways:
//!
//!   cargo run --release -p dash-mlx --example loss_bench
//!   MLX_ENABLE_TF32=0 cargo run --release -p dash-mlx --example loss_bench

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("loss_bench runs on Apple silicon only");
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
    use dash_mlx::logprob::vocab_chunk;
    use dash_mlx::loss::{CrossEntropyOptions, linear_cross_entropy};
    use mlx_rs::builder::Builder;
    use mlx_rs::error::Result;
    use mlx_rs::losses::{CrossEntropyBuilder, LossReduction};
    use mlx_rs::ops::ones;
    use mlx_rs::transforms::{eval, fallible_vjp};
    use mlx_rs::{Array, Dtype};

    const D: i32 = 1024;

    /// The workload's sizes, or `N` and `V` from the environment.
    fn size(name: &str, default: i32) -> i32 {
        std::env::var(name).map_or(default, |v| v.parse().expect("an integer"))
    }

    type Loss<'a> = Box<dyn Fn(&Array, &Array) -> Result<Array> + 'a>;

    /// The forward pass, and forward and backward for the hidden states alone and for both
    /// inputs, of `loss`.
    fn time(op: &str, name: &str, shape: &str, loss: Loss, h: &Array, w: &Array) -> Result<()> {
        let forward = || Ok(vec![loss(h, w)?]);
        report(
            &format!("{op}, forward"),
            name,
            shape,
            measure(forward, 2, 5)?,
        );
        let frozen = || {
            let f = |a: &[Array]| -> Result<Vec<Array>> { Ok(vec![loss(&a[0], w)?]) };
            let (mut out, grads) = fallible_vjp(f, std::slice::from_ref(h), &[ones::<f32>(&[1])?])?;
            out.extend(grads);
            Ok(out)
        };
        report(
            &format!("{op}, forward + backward, frozen head"),
            name,
            shape,
            measure(frozen, 2, 5)?,
        );
        let both = || {
            let f = |a: &[Array]| -> Result<Vec<Array>> { Ok(vec![loss(&a[0], &a[1])?]) };
            let (mut out, grads) = fallible_vjp(f, &[h.clone(), w.clone()], &[ones::<f32>(&[1])?])?;
            out.extend(grads);
            Ok(out)
        };
        report(
            &format!("{op}, forward + backward, head trained"),
            name,
            shape,
            measure(both, 2, 5)?,
        );
        Ok(())
    }

    pub fn run() -> Result<()> {
        let (N, V) = (size("N", 3072), size("V", 248_320));
        let bf16 = Dtype::Bfloat16;
        let weight = normal(&[V, D], 0.02, bf16, 1)?;
        let hidden = normal(&[N, D], 1.0, bf16, 2)?;
        let ids: Vec<i32> = (0..N).map(|i| (i * 7919) % V).collect();
        let targets = Array::from_slice(&ids, &[N]);
        eval([&weight, &hidden, &targets])?;
        let tf32 = std::env::var("MLX_ENABLE_TF32").unwrap_or_else(|_| "default".into());
        let shape = format!("n {N}, D {D}, V {V}, bf16, MLX_ENABLE_TF32={tf32}");
        let op = "cross-entropy, mean";
        println!("| operation | variant | shape | median | peak above the inputs |");
        println!("|---|---|---|---|---|");

        let options = CrossEntropyOptions::default();
        let name = format!(
            "dash-mlx, vocabulary chunks of {}",
            vocab_chunk(options.chunks)
        );
        let fused: Loss = Box::new(|h, w| Ok(linear_cross_entropy(h, w, &targets, &options)?.loss));
        time(op, &name, &shape, fused, &hidden, &weight)?;

        // mlx-rs's cross-entropy over the whole logits.
        let ce = CrossEntropyBuilder::new()
            .reduction(LossReduction::Mean)
            .build()
            .expect("the defaults");
        let whole: Loss = Box::new(|h, w| {
            let logits = h.matmul(w.t())?.as_dtype(Dtype::Float32)?;
            ce.apply(&logits, &targets)?.reshape(&[1])
        });
        time(
            op,
            "mlx-rs's CrossEntropy, the whole logits",
            &shape,
            whole,
            &hidden,
            &weight,
        )
    }
}
