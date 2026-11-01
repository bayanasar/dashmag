//! One training step of the gate's model (`tests/step.rs`), the whole sequence, with clipping and
//! AdamW: through dash-mag, and written directly on dash-mlx. Each process times one of the two, so
//! that they can run in turn; each prints the median over its steps after warm-up, split into building
//! the graph and the backward pass, evaluating them, and clipping, updating and evaluating that.
//!
//!   MLX_ENABLE_TF32=0 cargo run --release -p dash-mag --example step_bench -- gateway f32
//!   MLX_ENABLE_TF32=0 cargo run --release -p dash-mag --example step_bench -- direct bf16

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("step_bench compares against dash-mlx, which runs on macOS");
}

#[cfg(target_os = "macos")]
#[path = "../tests/common/mod.rs"]
mod common;

#[cfg(target_os = "macos")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (way, dtype) = (args[1].as_str(), args[2].as_str());
    let (warmup, steps) = (5, 50);
    let mut times = Vec::with_capacity(steps);
    let mut first = None;
    let mut run = |mut step: Box<dyn FnMut() -> ([f64; 3], f32, f32)>| {
        for i in 0..warmup + steps {
            let (t, loss, norm) = step();
            first.get_or_insert((loss, norm));
            if i >= warmup {
                times.push(t);
            }
        }
    };
    match way {
        "gateway" => run(gateway::stepper(dtype)),
        "direct" => run(direct::stepper(dtype)),
        other => panic!("{other}: gateway or direct"),
    }
    let median = |pick: fn(&[f64; 3]) -> f64| {
        let mut v: Vec<f64> = times.iter().map(pick).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (loss, norm) = first.unwrap();
    println!(
        "{way} {dtype}: {:.3} ms a step: build {:.3}, gradients {:.3}, update {:.3}; first step loss {loss} norm {norm}",
        median(|t| t.iter().sum()),
        median(|t| t[0]),
        median(|t| t[1]),
        median(|t| t[2]),
    );
}

#[cfg(target_os = "macos")]
const VOCAB: usize = 64;
#[cfg(target_os = "macos")]
const WIDTH: usize = 32;
#[cfg(target_os = "macos")]
const RANK: usize = 4;
#[cfg(target_os = "macos")]
const HEADS: usize = 2;
#[cfg(target_os = "macos")]
const DELTA: usize = 16;
#[cfg(target_os = "macos")]
const KERNEL: usize = 4;
#[cfg(target_os = "macos")]
const QUERY_HEADS: usize = 4;
#[cfg(target_os = "macos")]
const KEY_HEADS: usize = 2;
#[cfg(target_os = "macos")]
const HEAD: usize = 64;
#[cfg(target_os = "macos")]
const CHUNK: usize = 64;

/// The gate's sequence.
#[cfg(target_os = "macos")]
fn sequence() -> (Vec<i32>, Vec<i32>) {
    let ids: Vec<i32> = common::values(9, CHUNK + 16)
        .iter()
        .map(|v| ((v + 1.0) * 32.0) as i32 % VOCAB as i32)
        .collect();
    let mut targets = ids[1..].to_vec();
    targets.push(ids[0]);
    (ids, targets)
}

/// The model of `tests/step.rs`, its whole-sequence path, through dash-mag.
#[cfg(target_os = "macos")]
mod gateway {
    use super::*;
    use crate::common::tensor;
    use dash_mag::lora::{DEFAULT_ADAPTER, Lora, LoraLinear};
    use dash_mag::ops::{
        Activation, AttentionMask, AttentionOptions, CausalConv1dOptions, ChunkGatedDeltaOptions,
        LogprobOptions, attention, causal_conv1d, chosen_logprob, chunk_gated_delta_rule,
    };
    use dash_mag::optim::{AdamW, ClipGradNormOptions, clip_grad_norm};
    use dash_mag::rope::Rope;
    use dash_mag::{DType, Result, Tensor, eval, vjp};
    use std::time::Instant;

    fn projection(seed: u64, out: usize, inp: usize, dtype: DType) -> LoraLinear {
        let mut layer =
            LoraLinear::frozen(&tensor(seed, &[out, inp], dtype).mul_scalar(0.25).unwrap())
                .unwrap();
        let a = tensor(seed + 1, &[RANK, inp], DType::F32)
            .mul_scalar(0.125)
            .unwrap();
        let b = tensor(seed + 2, &[out, RANK], DType::F32)
            .mul_scalar(0.125)
            .unwrap();
        layer
            .insert_adapter(DEFAULT_ADAPTER, Lora::new(&a, &b, 2.0).unwrap())
            .unwrap();
        layer
    }

    #[derive(Clone)]
    enum Block {
        Delta {
            qkv: LoraLinear,
            decay: LoraLinear,
            gate: LoraLinear,
            out: LoraLinear,
            conv: Tensor,
        },
        Attention {
            q: LoraLinear,
            k: LoraLinear,
            v: LoraLinear,
            out: LoraLinear,
        },
    }

    impl Block {
        fn delta(seed: u64, dtype: DType) -> Self {
            let channels = 3 * HEADS * DELTA;
            Block::Delta {
                qkv: projection(seed, channels, WIDTH, dtype),
                decay: projection(seed + 10, HEADS, WIDTH, dtype),
                gate: projection(seed + 20, HEADS, WIDTH, dtype),
                out: projection(seed + 30, WIDTH, HEADS * DELTA, dtype),
                conv: tensor(seed + 40, &[channels, KERNEL], dtype)
                    .mul_scalar(0.5)
                    .unwrap(),
            }
        }

        fn attention(seed: u64, dtype: DType) -> Self {
            Block::Attention {
                q: projection(seed, QUERY_HEADS * HEAD, WIDTH, dtype),
                k: projection(seed + 10, KEY_HEADS * HEAD, WIDTH, dtype),
                v: projection(seed + 20, KEY_HEADS * HEAD, WIDTH, dtype),
                out: projection(seed + 30, WIDTH, QUERY_HEADS * HEAD, dtype),
            }
        }

        fn layers(&self) -> Vec<&LoraLinear> {
            match self {
                Block::Delta {
                    qkv,
                    decay,
                    gate,
                    out,
                    ..
                } => vec![qkv, decay, gate, out],
                Block::Attention { q, k, v, out } => vec![q, k, v, out],
            }
        }

        fn layers_mut(&mut self) -> Vec<&mut LoraLinear> {
            match self {
                Block::Delta {
                    qkv,
                    decay,
                    gate,
                    out,
                    ..
                } => vec![qkv, decay, gate, out],
                Block::Attention { q, k, v, out } => vec![q, k, v, out],
            }
        }

        fn forward(&self, x: &Tensor, rope: &Rope) -> Result<Tensor> {
            let shape = x.shape();
            let (b, t) = (shape[0] as isize, shape[1] as isize);
            let y = match self {
                Block::Delta {
                    qkv,
                    decay,
                    gate,
                    out,
                    conv,
                } => {
                    let options = CausalConv1dOptions::default().with_activation(Activation::Silu);
                    let (c, _) = causal_conv1d(&qkv.forward(x)?, conv, None, None, None, options)?;
                    let parts = c.split_at(&[HEADS * DELTA, 2 * HEADS * DELTA], -1)?;
                    let head = |p: &Tensor| p.reshape(&[b, t, HEADS as isize, DELTA as isize]);
                    let g = decay.forward(x)?.as_dtype(DType::F32)?.softplus()?.neg()?;
                    let beta = gate.forward(x)?.as_dtype(DType::F32)?.sigmoid()?;
                    let (h, _) = chunk_gated_delta_rule(
                        &head(&parts[0])?,
                        &head(&parts[1])?,
                        &head(&parts[2])?,
                        &g,
                        &beta,
                        None,
                        None,
                        ChunkGatedDeltaOptions::default(),
                    )?;
                    out.forward(&h.reshape(&[b, t, (HEADS * DELTA) as isize])?)?
                }
                Block::Attention { q, k, v, out } => {
                    let heads =
                        |p: &Tensor, h: usize| p.reshape(&[b, t, h as isize, HEAD as isize]);
                    let positions = 0..t as usize;
                    let query =
                        rope.apply_range(&heads(&q.forward(x)?, QUERY_HEADS)?, positions.clone())?;
                    let key = rope.apply_range(&heads(&k.forward(x)?, KEY_HEADS)?, positions)?;
                    let value = heads(&v.forward(x)?, KEY_HEADS)?;
                    let options = AttentionOptions::default()
                        .with_causal(true)
                        .with_scale(1.0 / (HEAD as f32).sqrt());
                    let h = attention(&query, &key, &value, &AttentionMask::None, options)?;
                    out.forward(&h.reshape(&[b, t, (QUERY_HEADS * HEAD) as isize])?)?
                }
            };
            x.add(&y)?.as_dtype(x.dtype())
        }
    }

    #[derive(Clone)]
    struct Model {
        embed: Tensor,
        blocks: Vec<Block>,
        rope: Rope,
    }

    impl Model {
        fn factors(&self) -> Vec<Tensor> {
            let mut factors = Vec::new();
            for block in &self.blocks {
                for layer in block.layers() {
                    let lora = layer.lora().unwrap();
                    factors.push(lora.a.clone());
                    factors.push(lora.b.clone());
                }
            }
            factors
        }

        fn set_factors(&mut self, factors: &[Tensor]) -> Result<()> {
            let mut at = factors.as_chunks::<2>().0.iter();
            for block in &mut self.blocks {
                for layer in block.layers_mut() {
                    let pair = at.next().expect("two factors per layer");
                    layer.set_factors(pair[0].clone(), pair[1].clone())?;
                }
            }
            Ok(())
        }

        fn loss(&self, ids: &[i32], targets: &[i32]) -> Result<Tensor> {
            let at = Tensor::from_i32(ids, &[ids.len()])?;
            let mut x = self.embed.take(&at, 0)?.expand_dims(0)?;
            for block in &self.blocks {
                x = block.forward(&x, &self.rope)?;
            }
            let h = x.squeeze(0)?.rms_norm(None, 1e-6)?;
            let targets = Tensor::from_i32(targets, &[targets.len()])?;
            chosen_logprob(&h, &self.embed, &targets, &[], LogprobOptions::default())?
                .logprob
                .mean()?
                .neg()
        }
    }

    pub fn stepper(dtype: &str) -> Box<dyn FnMut() -> ([f64; 3], f32, f32)> {
        let dtype = if dtype == "bf16" {
            DType::BF16
        } else {
            DType::F32
        };
        let mut model = Model {
            embed: tensor(1, &[VOCAB, WIDTH], dtype),
            blocks: vec![
                Block::delta(100, dtype),
                Block::attention(200, dtype),
                Block::delta(300, dtype),
            ],
            rope: Rope::new(HEAD / 2, 10000.0, 1024, dtype).unwrap(),
        };
        let mut optimiser = AdamW::new((0.9, 0.999), 1e-5, 0.0);
        let names: Vec<String> = (0..model.factors().len()).map(|i| i.to_string()).collect();
        let (ids, targets) = sequence();
        let options = ClipGradNormOptions {
            max_norm: 0.5,
            ..Default::default()
        };
        Box::new(move || {
            let start = Instant::now();
            let mut held = model.clone();
            let one = Tensor::scalar(1.0, DType::F32).unwrap();
            let (outs, mut grads) = vjp(
                |p| {
                    held.set_factors(p)?;
                    Ok(vec![held.loss(&ids, &targets)?])
                },
                &model.factors(),
                &[Some(one)],
            )
            .unwrap();
            let built = start.elapsed();
            let refs: Vec<&Tensor> = outs.iter().chain(&grads).collect();
            eval(&refs).unwrap();
            let evaluated = start.elapsed();
            let norm = clip_grad_norm(&mut grads, &options).unwrap();
            optimiser.begin_step();
            let stepped: Vec<Tensor> = model
                .factors()
                .iter()
                .zip(&grads)
                .zip(&names)
                .map(|((p, g), name)| optimiser.update(name, p, g, 1e-3).unwrap())
                .collect();
            model.set_factors(&stepped).unwrap();
            let refs: Vec<&Tensor> = stepped.iter().chain(optimiser.state()).collect();
            eval(&refs).unwrap();
            let done = start.elapsed();
            (
                [
                    built.as_secs_f64() * 1e3,
                    (evaluated - built).as_secs_f64() * 1e3,
                    (done - evaluated).as_secs_f64() * 1e3,
                ],
                outs[0].item().unwrap(),
                norm,
            )
        })
    }
}

/// The same model and step, written on dash-mlx and its arrays.
#[cfg(target_os = "macos")]
mod direct {
    use super::*;
    use dash_mlx::attention::{AttentionMask, AttentionOptions, attention};
    use dash_mlx::conv1d::{Activation, CausalConv1dOptions, causal_conv1d};
    use dash_mlx::gated_delta::chunk::{ChunkGatedDeltaOptions, chunk_gated_delta_rule};
    use dash_mlx::logprob::{LogprobOptions, chosen_logprob};
    use dash_mlx::lora::{DEFAULT_ADAPTER, Lora, LoraLinear};
    use dash_mlx::mlx::error::Result;
    use dash_mlx::mlx::module::{Module, Param};
    use dash_mlx::mlx::ops::indexing::TryIndexOp;
    use dash_mlx::mlx::ops::{concatenate, select, sigmoid, zeros_like};
    use dash_mlx::mlx::transforms::{eval, fallible_vjp};
    use dash_mlx::mlx::{Array, Dtype, fast, nn};
    use std::collections::HashMap;
    use std::time::Instant;

    fn tensor(seed: u64, shape: &[i32], dtype: Dtype) -> Array {
        let n = shape.iter().product::<i32>() as usize;
        Array::from_slice(&common::values(seed, n), shape)
            .as_dtype(dtype)
            .unwrap()
    }

    fn scalar(value: f64, dtype: Dtype) -> Array {
        Array::from_f32(value as f32).as_dtype(dtype).unwrap()
    }

    fn scaled(seed: u64, shape: &[i32], dtype: Dtype, by: f64) -> Array {
        tensor(seed, shape, dtype)
            .multiply(scalar(by, dtype))
            .unwrap()
    }

    fn projection(seed: u64, out: i32, inp: i32, dtype: Dtype) -> LoraLinear {
        let a = scaled(seed + 1, &[RANK as i32, inp], Dtype::Float32, 0.125);
        let b = scaled(seed + 2, &[out, RANK as i32], Dtype::Float32, 0.125);
        LoraLinear::frozen(&scaled(seed, &[out, inp], dtype, 0.25))
            .unwrap()
            .with_lora(Lora::new(&a, &b, 2.0).unwrap())
            .unwrap()
    }

    #[derive(Clone)]
    enum Block {
        Delta {
            qkv: LoraLinear,
            decay: LoraLinear,
            gate: LoraLinear,
            out: LoraLinear,
            conv: Array,
        },
        Attention {
            q: LoraLinear,
            k: LoraLinear,
            v: LoraLinear,
            out: LoraLinear,
        },
    }

    /// The rotation `tests/step.rs` takes from dash-mag's `Rope`: MLX's own over `[B, T, H, D]`.
    fn rope(x: &Array) -> Result<Array> {
        let turned = x.swap_axes(1, 2)?;
        fast::rope(&turned, (HEAD / 2) as i32, false, 10000.0, 1.0, 0)?.swap_axes(1, 2)
    }

    impl Block {
        fn delta(seed: u64, dtype: Dtype) -> Self {
            let (channels, w) = ((3 * HEADS * DELTA) as i32, WIDTH as i32);
            Block::Delta {
                qkv: projection(seed, channels, w, dtype),
                decay: projection(seed + 10, HEADS as i32, w, dtype),
                gate: projection(seed + 20, HEADS as i32, w, dtype),
                out: projection(seed + 30, w, (HEADS * DELTA) as i32, dtype),
                conv: scaled(seed + 40, &[channels, KERNEL as i32], dtype, 0.5),
            }
        }

        fn attention(seed: u64, dtype: Dtype) -> Self {
            let w = WIDTH as i32;
            Block::Attention {
                q: projection(seed, (QUERY_HEADS * HEAD) as i32, w, dtype),
                k: projection(seed + 10, (KEY_HEADS * HEAD) as i32, w, dtype),
                v: projection(seed + 20, (KEY_HEADS * HEAD) as i32, w, dtype),
                out: projection(seed + 30, w, (QUERY_HEADS * HEAD) as i32, dtype),
            }
        }

        fn layers(&self) -> Vec<&LoraLinear> {
            match self {
                Block::Delta {
                    qkv,
                    decay,
                    gate,
                    out,
                    ..
                } => vec![qkv, decay, gate, out],
                Block::Attention { q, k, v, out } => vec![q, k, v, out],
            }
        }

        fn layers_mut(&mut self) -> Vec<&mut LoraLinear> {
            match self {
                Block::Delta {
                    qkv,
                    decay,
                    gate,
                    out,
                    ..
                } => vec![qkv, decay, gate, out],
                Block::Attention { q, k, v, out } => vec![q, k, v, out],
            }
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let (b, t) = (x.dim(0), x.dim(1));
            let y = match self {
                Block::Delta {
                    qkv,
                    decay,
                    gate,
                    out,
                    conv,
                } => {
                    let options = CausalConv1dOptions::default().with_activation(Activation::Silu);
                    let (c, _) = causal_conv1d(&qkv.forward(x)?, conv, None, None, None, options)?;
                    let d = (HEADS * DELTA) as i32;
                    let head = |p: Array| p.reshape(&[b, t, HEADS as i32, DELTA as i32]);
                    let (q, k, v) = (
                        head(c.try_index((.., .., 0..d))?)?,
                        head(c.try_index((.., .., d..2 * d))?)?,
                        head(c.try_index((.., .., 2 * d..))?)?,
                    );
                    let g =
                        nn::softplus(decay.forward(x)?.as_dtype(Dtype::Float32)?)?.negative()?;
                    let beta = sigmoid(gate.forward(x)?.as_dtype(Dtype::Float32)?)?;
                    let (h, _) = chunk_gated_delta_rule(
                        &q,
                        &k,
                        &v,
                        &g,
                        &beta,
                        None,
                        None,
                        ChunkGatedDeltaOptions::default(),
                    )?;
                    out.forward(&h.reshape(&[b, t, d])?)?
                }
                Block::Attention { q, k, v, out } => {
                    let heads = |p: Array, h: usize| p.reshape(&[b, t, h as i32, HEAD as i32]);
                    let query = rope(&heads(q.forward(x)?, QUERY_HEADS)?)?;
                    let key = rope(&heads(k.forward(x)?, KEY_HEADS)?)?;
                    let value = heads(v.forward(x)?, KEY_HEADS)?;
                    let options = AttentionOptions::default()
                        .with_causal(true)
                        .with_scale(1.0 / (HEAD as f32).sqrt());
                    let h = attention(&query, &key, &value, &AttentionMask::None, options)?;
                    out.forward(&h.reshape(&[b, t, (QUERY_HEADS * HEAD) as i32])?)?
                }
            };
            x.add(&y)?.as_dtype(x.dtype())
        }
    }

    #[derive(Clone)]
    struct Model {
        embed: Array,
        blocks: Vec<Block>,
    }

    impl Model {
        fn factors(&self) -> Vec<Array> {
            let mut factors = Vec::new();
            for block in &self.blocks {
                for layer in block.layers() {
                    let lora = layer.adapter(DEFAULT_ADAPTER).unwrap();
                    factors.push(lora.a.value.clone());
                    factors.push(lora.b.value.clone());
                }
            }
            factors
        }

        fn set_factors(&mut self, factors: &[Array]) {
            let mut at = factors.as_chunks::<2>().0.iter();
            for block in &mut self.blocks {
                for layer in block.layers_mut() {
                    let pair = at.next().expect("two factors per layer");
                    let lora = layer.adapter_mut(DEFAULT_ADAPTER).unwrap();
                    lora.a = Param::new(pair[0].clone());
                    lora.b = Param::new(pair[1].clone());
                }
            }
        }

        fn loss(&mut self, ids: &[i32], targets: &[i32]) -> Result<Array> {
            let at = Array::from_slice(ids, &[ids.len() as i32]);
            let mut x = self.embed.take_axis(&at, 0)?.expand_dims(0)?;
            for block in &mut self.blocks {
                x = block.forward(&x)?;
            }
            let dtype = x.dtype();
            let h = fast::rms_norm(x.squeeze_axes(&[0])?, None, 1e-6)?.as_dtype(dtype)?;
            let targets = Array::from_slice(targets, &[targets.len() as i32]);
            chosen_logprob(&h, &self.embed, &targets, &[], LogprobOptions::default())?
                .logprob
                .mean(None)?
                .negative()
        }
    }

    /// `dash_mag::optim::clip_grad_norm` with its defaults: the total in f32, each gradient scaled.
    fn clip(grads: &mut [Array], max_norm: f64) -> Result<f32> {
        let partials = grads
            .iter()
            .map(|g| {
                g.as_dtype(Dtype::Float32)?
                    .square()?
                    .sum(None)?
                    .expand_dims(0)
            })
            .collect::<Result<Vec<_>>>()?;
        let total = concatenate(&partials, 0)?.sum(None)?.sqrt()?;
        let one = scalar(1.0, Dtype::Float32);
        let coef = one
            .divide(total.add(scalar(1e-6, Dtype::Float32))?)?
            .multiply(scalar(max_norm, Dtype::Float32))?;
        let coef = select(coef.gt(&one)?, &one, &coef)?;
        for g in grads.iter_mut() {
            *g = g.multiply(coef.as_dtype(g.dtype())?)?;
        }
        Ok(total.as_slice::<f32>()[0])
    }

    /// `dash_mag::optim::AdamW`'s arithmetic.
    struct AdamW {
        betas: (f64, f64),
        eps: f64,
        step: i32,
        state: HashMap<usize, (Array, Array)>,
    }

    impl AdamW {
        fn update(&mut self, name: usize, p: &Array, g: &Array, lr: f64) -> Result<Array> {
            let (b1, b2) = self.betas;
            let d = p.dtype();
            let t = self.step;
            let (m, v) = match self.state.remove(&name) {
                Some(state) => state,
                None => (zeros_like(p)?, zeros_like(p)?),
            };
            let m = m.add(g.subtract(&m)?.multiply(scalar(1.0 - b1, d))?)?;
            let v = v
                .multiply(scalar(b2, d))?
                .add(g.square()?.multiply(scalar(1.0 - b2, d))?)?;
            let denominator = v
                .sqrt()?
                .divide(scalar((1.0 - b2.powi(t)).sqrt(), d))?
                .add(scalar(self.eps, d))?;
            let step = lr / (1.0 - b1.powi(t));
            let p = p.subtract(m.divide(&denominator)?.multiply(scalar(step, d))?)?;
            self.state.insert(name, (m, v));
            Ok(p)
        }
    }

    pub fn stepper(dtype: &str) -> Box<dyn FnMut() -> ([f64; 3], f32, f32)> {
        let dtype = if dtype == "bf16" {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        };
        let mut model = Model {
            embed: tensor(1, &[VOCAB as i32, WIDTH as i32], dtype),
            blocks: vec![
                Block::delta(100, dtype),
                Block::attention(200, dtype),
                Block::delta(300, dtype),
            ],
        };
        let mut optimiser = AdamW {
            betas: (0.9, 0.999),
            eps: 1e-5,
            step: 0,
            state: HashMap::new(),
        };
        let (ids, targets) = sequence();
        Box::new(move || {
            let start = Instant::now();
            let mut held = model.clone();
            let one = scalar(1.0, Dtype::Float32);
            let (outs, mut grads) = fallible_vjp(
                |p: &[Array]| {
                    held.set_factors(p);
                    Ok(vec![held.loss(&ids, &targets)?])
                },
                &model.factors(),
                std::slice::from_ref(&one),
            )
            .unwrap();
            let built = start.elapsed();
            eval(outs.iter().chain(&grads)).unwrap();
            let evaluated = start.elapsed();
            let norm = clip(&mut grads, 0.5).unwrap();
            optimiser.step += 1;
            let stepped: Vec<Array> = model
                .factors()
                .iter()
                .zip(&grads)
                .enumerate()
                .map(|(i, (p, g))| optimiser.update(i, p, g, 1e-3).unwrap())
                .collect();
            model.set_factors(&stepped);
            eval(
                stepped
                    .iter()
                    .chain(optimiser.state.values().flat_map(|(m, v)| [m, v])),
            )
            .unwrap();
            let done = start.elapsed();
            (
                [
                    built.as_secs_f64() * 1e3,
                    (evaluated - built).as_secs_f64() * 1e3,
                    (done - evaluated).as_secs_f64() * 1e3,
                ],
                outs[0].as_slice::<f32>()[0],
                norm,
            )
        })
    }
}
