//! A LoRA'd linear layer at the workload's shapes: 6144 tokens of 1024 in bf16 through a frozen
//! bf16 `[6144, 1024]` weight with a rank-16 f32 adapter; forward, and forward and backward to
//! the input and the factors. Against mlx-lm's `LoRALinear` (its forward pass written in mlx-rs:
//! `linear(x) + (scale * ((x @ lora_a) @ lora_b)).astype(x.dtype)`, the factors f32), timed in
//! the same process.
//!
//!   cargo run --release -p dash-mlx --example lora_bench

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("lora_bench runs on Apple silicon only");
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
    use dash_mlx::lora::{Lora, LoraLinear};
    use mlx_rs::error::Result;
    use mlx_rs::module::Module;
    use mlx_rs::transforms::{eval, fallible_vjp};
    use mlx_rs::{Array, Dtype};

    const N: i32 = 6144;
    const IN: i32 = 1024;
    const OUT: i32 = 6144;
    const R: i32 = 16;

    pub fn run() -> Result<()> {
        let bf16 = Dtype::Bfloat16;
        let x = normal(&[N, IN], 1.0, bf16, 1)?;
        let w = normal(&[OUT, IN], 0.02, bf16, 2)?;
        let a = normal(&[R, IN], 0.02, Dtype::Float32, 3)?;
        let b = normal(&[OUT, R], 0.02, Dtype::Float32, 4)?;
        let gy = normal(&[N, OUT], 1.0, bf16, 5)?;
        eval([&x, &w, &a, &b, &gy])?;
        let scale = 2.0;
        let shape = format!("x [{N}, {IN}] bf16, W [{OUT}, {IN}] bf16 frozen, r {R} f32");
        println!("| operation | variant | shape | median | peak above the inputs |");
        println!("|---|---|---|---|---|");
        type Layer = fn(&[Array], &Array, f64) -> Result<Array>;
        let dash: Layer = |p, w, scale| {
            let mut layer = LoraLinear::frozen(w)?.with_lora(Lora::new(&p[1], &p[2], scale)?)?;
            layer.forward(&p[0])
        };
        // mlx-lm's LoRALinear: lora_a [in, r] and lora_b [r, out], the term rounded to x's dtype
        // and added there.
        let mlx_lm: Layer = |p, w, scale| {
            let y = p[0].matmul(w.t())?;
            let z = p[0].matmul(p[1].t())?.matmul(p[2].t())?;
            y.add(
                z.multiply(Array::from_f32(scale as f32))?
                    .as_dtype(p[0].dtype())?,
            )
        };
        let primals = [x.clone(), a.clone(), b.clone()];
        for (name, layer) in [
            ("dash-mlx LoraLinear", dash),
            ("mlx-lm's LoRALinear", mlx_lm),
        ] {
            report(
                "LoRA linear, forward",
                name,
                &shape,
                measure(|| Ok(vec![layer(&primals, &w, scale)?]), 3, 10)?,
            );
            let both = || {
                let (mut out, grads) = fallible_vjp(
                    |p: &[Array]| Ok(vec![layer(p, &w, scale)?]),
                    &primals,
                    std::slice::from_ref(&gy),
                )?;
                out.extend(grads);
                Ok(out)
            };
            report(
                "LoRA linear, forward + backward (x, A, B)",
                name,
                &shape,
                measure(both, 3, 10)?,
            );
        }
        Ok(())
    }
}
