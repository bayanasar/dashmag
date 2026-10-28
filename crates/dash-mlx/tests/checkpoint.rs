//! Recompute against plain autodiff: a stack of LoRA'd blocks (silu-gated, residual, a frozen
//! bf16 or f32 base with a trainable adapter each) run through `checkpoint::forward` and
//! `checkpoint::backward` gives the input's gradient and every adapter's, bit for bit, as MLX's
//! autodiff over the whole stack gives them, in f32 and bf16; the layers hold their own arrays
//! afterwards, and only trainable parameters get gradients.
//!
//!   cargo test -p dash-mlx --test checkpoint
#![cfg(target_os = "macos")]

use dash_mlx::checkpoint;
use dash_mlx::lora::{Lora, LoraLinear};
use mlx_rs::error::Result;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::transforms::fallible_vjp;
use mlx_rs::{Array, Dtype};

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
    fn new(d: i32, hidden: i32, dtype: Dtype, seed: u64) -> Result<Self> {
        let normal = |shape: &[i32], scale: f32, seed: u64| -> Result<Array> {
            mlx_rs::random::normal::<f32>(shape, None, None, &mlx_rs::random::key(seed)?)?
                .multiply(Array::from_f32(scale))
        };
        let layer = |out: i32, inp: i32, seed: u64| -> Result<LoraLinear> {
            LoraLinear::frozen(&normal(&[out, inp], 0.05, seed)?.as_dtype(dtype)?)?.with_lora(
                Lora::new(
                    &normal(&[8, inp], 0.05, seed + 1)?,
                    &normal(&[out, 8], 0.05, seed + 2)?,
                    2.0,
                )?,
            )
        };
        Ok(Self {
            up: layer(hidden, d, seed)?,
            gate: layer(hidden, d, seed + 10)?,
            down: layer(d, hidden, seed + 20)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let g = self.gate.forward(x)?;
        let s = mlx_rs::nn::silu(g.as_dtype(Dtype::Float32)?)?.as_dtype(g.dtype())?;
        x.add(self.down.forward(&s.multiply(self.up.forward(x)?)?)?)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct Stack {
    #[param]
    layers: Vec<Block>,
}

fn bits(a: &Array) -> Vec<u32> {
    let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
    a.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
}

/// Every parameter's bits by its path, frozen ones included.
fn by_key(stack: &Stack) -> std::collections::BTreeMap<String, Vec<u32>> {
    stack
        .parameters()
        .flatten()
        .into_iter()
        .map(|(k, v)| (k.to_string(), bits(v)))
        .collect()
}

#[test]
fn gives_plain_autodiff_s_gradients_bit_for_bit() -> Result<()> {
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let (n, d, hidden, depth) = (64, 128, 256, 4);
        let mut stack = Stack {
            layers: (0..depth)
                .map(|i| Block::new(d, hidden, dtype, 100 * i as u64))
                .collect::<Result<_>>()?,
        };
        let x = mlx_rs::random::normal::<f32>(&[n, d], None, None, &mlx_rs::random::key(7)?)?
            .as_dtype(dtype)?;
        let gy = mlx_rs::random::normal::<f32>(&[n, d], None, None, &mlx_rs::random::key(8)?)?
            .as_dtype(dtype)?;

        // Plain autodiff over the whole stack: the parameters' gradients, and the input's.
        let loss = |stack: &mut Stack, (x, gy): (&Array, &Array)| -> Result<Array> {
            let mut h = x.clone();
            for layer in &mut stack.layers {
                h = layer.forward(&h)?;
            }
            h.as_dtype(Dtype::Float32)?
                .multiply(gy.as_dtype(Dtype::Float32)?)?
                .sum(None)
        };
        let (_, plain) = mlx_rs::nn::value_and_grad(loss)(&mut stack, (&x, &gy))?;
        let dx_plain = fallible_vjp(
            |p: &[Array]| {
                let mut stack = stack.clone();
                let mut h = p[0].clone();
                for layer in &mut stack.layers {
                    h = layer.forward(&h)?;
                }
                Ok(vec![h])
            },
            std::slice::from_ref(&x),
            std::slice::from_ref(&gy),
        )?
        .1
        .remove(0);

        // Recomputed, one layer at a time.
        let before = by_key(&stack);
        let (out, saved) = checkpoint::forward(&mut stack.layers, &x, |l, h| l.forward(h))?;
        assert_eq!(saved.len(), depth);
        assert_eq!(out.dtype(), dtype);
        let (dx, grads) = checkpoint::backward(&mut stack.layers, saved, &gy, |l, h| l.forward(h))?;
        assert!(
            bits(&dx) == bits(&dx_plain),
            "{dtype:?}: the input's gradient differs"
        );
        let mut compared = 0;
        for (i, layer_grads) in grads.iter().enumerate() {
            for (key, g) in layer_grads {
                let plain_key = format!("layers.{i}.{key}");
                let want = plain
                    .get(plain_key.as_str())
                    .unwrap_or_else(|| panic!("{plain_key} missing from plain autodiff"));
                assert!(bits(g) == bits(want), "{dtype:?} {plain_key}: differs");
                compared += 1;
            }
        }
        assert_eq!(
            compared,
            plain.len(),
            "{dtype:?}: every trainable parameter's gradient, nothing else"
        );
        assert_eq!(
            compared,
            depth * 3 * 2,
            "{dtype:?}: the adapters' factors alone train"
        );
        // The layers hold their own arrays again.
        let after = by_key(&stack);
        assert!(before == after, "{dtype:?}: the parameters changed");
        println!("{dtype:?}: the input's gradient and {compared} parameter gradients bit for bit");
    }
    Ok(())
}
