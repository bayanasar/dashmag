//! Recompute against plain autodiff on a stack of LoRA'd MLP blocks (silu-gated, residual; a
//! frozen bf16 base and a rank-16 f32 adapter in each linear): 8 blocks, 4096 tokens of 1024, a
//! hidden width of 3584, bf16; one forward and backward pass to every adapter and the input.
//! Plain: MLX's autodiff over the whole stack in one evaluation. Recompute: `checkpoint::forward`
//! and `checkpoint::backward`, one layer's activations alive at a time.
//!
//!   cargo run --release -p dash-mlx --example checkpoint_bench

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("checkpoint_bench runs on Apple silicon only");
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
    use dash_mlx::checkpoint;
    use dash_mlx::lora::{Lora, LoraLinear};
    use mlx_rs::error::Result;
    use mlx_rs::macros::ModuleParameters;
    use mlx_rs::module::Module;
    use mlx_rs::transforms::{eval, fallible_vjp};
    use mlx_rs::{Array, Dtype};

    const N: i32 = 4096;
    const D: i32 = 1024;
    const HIDDEN: i32 = 3584;
    const DEPTH: usize = 8;

    #[derive(Debug, Clone, ModuleParameters)]
    struct Block {
        #[param]
        up: LoraLinear,
        #[param]
        gate: LoraLinear,
        #[param]
        down: LoraLinear,
    }

    impl Block {
        fn new(seed: u64) -> Result<Self> {
            let layer = |out: i32, inp: i32, seed: u64| -> Result<LoraLinear> {
                LoraLinear::frozen(&normal(&[out, inp], 0.02, Dtype::Bfloat16, seed)?)?.with_lora(
                    Lora::new(
                        &normal(&[16, inp], 0.02, Dtype::Float32, seed + 1)?,
                        &normal(&[out, 16], 0.02, Dtype::Float32, seed + 2)?,
                        2.0,
                    )?,
                )
            };
            Ok(Self {
                up: layer(HIDDEN, D, seed)?,
                gate: layer(HIDDEN, D, seed + 10)?,
                down: layer(D, HIDDEN, seed + 20)?,
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let g = self.gate.forward(x)?;
            let s = mlx_rs::nn::silu(g.as_dtype(Dtype::Float32)?)?.as_dtype(g.dtype())?;
            x.add(self.down.forward(&s.multiply(self.up.forward(x)?)?)?)
        }

        /// The block's adapter factors, in a fixed order.
        fn factors(&self) -> Vec<Array> {
            [&self.up, &self.gate, &self.down]
                .iter()
                .flat_map(|l| {
                    let lora = l.lora().expect("an adapter");
                    [lora.a.as_ref().clone(), lora.b.as_ref().clone()]
                })
                .collect()
        }

        fn with_factors(&self, f: &[Array]) -> Result<Self> {
            let mut block = self.clone();
            for (i, layer) in [&mut block.up, &mut block.gate, &mut block.down]
                .into_iter()
                .enumerate()
            {
                let lora = layer.lora().expect("an adapter").clone();
                *layer = LoraLinear::frozen(layer.weight.as_ref())?.with_lora(Lora::new(
                    &f[2 * i],
                    &f[2 * i + 1],
                    lora.scale,
                )?)?;
            }
            Ok(block)
        }
    }

    pub fn run() -> Result<()> {
        let mut layers = (0..DEPTH)
            .map(|i| Block::new(100 * i as u64))
            .collect::<Result<Vec<_>>>()?;
        let x = normal(&[N, D], 1.0, Dtype::Bfloat16, 1)?;
        let gy = normal(&[N, D], 1.0, Dtype::Bfloat16, 2)?;
        eval(
            layers
                .iter()
                .flat_map(|l| l.factors())
                .chain([x.clone(), gy.clone()])
                .collect::<Vec<_>>()
                .iter(),
        )?;
        let shape = format!("{DEPTH} blocks, {N} tokens of {D}, hidden {HIDDEN}, bf16, rank 16");
        println!("| operation | variant | shape | median | peak above the inputs |");
        println!("|---|---|---|---|---|");

        // Plain autodiff: the input and every block's factors as primals, one evaluation.
        let mut primals = vec![x.clone()];
        primals.extend(layers.iter().flat_map(|l| l.factors()));
        let plain = || {
            let blocks = layers.clone();
            let f = |p: &[Array]| -> Result<Vec<Array>> {
                let mut h = p[0].clone();
                for (i, block) in blocks.iter().enumerate() {
                    h = block.with_factors(&p[1 + 6 * i..7 + 6 * i])?.forward(&h)?;
                }
                Ok(vec![h])
            };
            let (mut out, grads) = fallible_vjp(f, &primals, std::slice::from_ref(&gy))?;
            out.extend(grads);
            Ok(out)
        };
        report(
            "forward + backward of a stack",
            "plain autodiff",
            &shape,
            measure(plain, 1, 3)?,
        );

        let recompute = || {
            let (out, saved) = checkpoint::forward(&mut layers, &x, |l, h| l.forward(h))?;
            let (dx, grads) = checkpoint::backward(&mut layers, saved, &gy, |l, h| l.forward(h))?;
            let mut all = vec![out, dx];
            all.extend(grads.into_iter().flat_map(|g| g.into_values()));
            Ok(all)
        };
        report(
            "forward + backward of a stack",
            "dash-mlx checkpoint",
            &shape,
            measure(recompute, 1, 3)?,
        );
        Ok(())
    }
}
