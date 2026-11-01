//! The gate: one training step of a small hybrid model with LoRA, from one source. The model below
//! names nothing but this crate, and the same file runs on every side.
//!
//! Each side is held to the recording in `tests/fixtures/step.json` (RTX 5070 Ti, IEEE f32), within
//! the bound a different device's rounding takes, and to itself: the gradient against central
//! differences, a step that lowers the loss, and a prefix run once with its states handed on, whose
//! gradient is that of the whole sequence.
//!
//!   cargo test --release -p dash-mag --test step -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-mag --features cuda --test step -- --nocapture
//!   MLX_ENABLE_TF32=0 cargo test --release -p dash-mag --test step -- --nocapture
//!
//! `DASH_MAG_RECORD=<path>` writes the recording instead of reading it.

#![cfg(not(target_os = "macos"))]

mod common;

use common::{relative, tensor};
use dash_mag::lora::{DEFAULT_ADAPTER, Lora, LoraLinear};
use dash_mag::ops::{
    Activation, AttentionMask, AttentionOptions, CausalConv1dOptions, ChunkGatedDeltaOptions,
    LogprobOptions, attention, causal_conv1d, chosen_logprob, chunk_gated_delta_rule,
};
use dash_mag::optim::{AdamW, ClipGradNormOptions, clip_grad_norm};
use dash_mag::rope::Rope;
use dash_mag::{DType, Result, Tensor, vjp};
use serde_json::{Value, json};

const VOCAB: usize = 64;
const WIDTH: usize = 32;
const RANK: usize = 4;
const HEADS: usize = 2;
const DELTA: usize = 16;
const KERNEL: usize = 4;
const QUERY_HEADS: usize = 4;
const KEY_HEADS: usize = 2;
const HEAD: usize = 64;
const CHUNK: usize = 64;

/// A projection with one adapter, its values drawn from `seed`.
fn projection(seed: u64, out: usize, inp: usize, dtype: DType) -> LoraLinear {
    let mut layer =
        LoraLinear::frozen(&tensor(seed, &[out, inp], dtype).mul_scalar(0.25).unwrap()).unwrap();
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

/// What a block carries on to the next run of tokens.
#[derive(Clone)]
enum State {
    Delta { conv: Tensor, recurrent: Tensor },
    Attention { key: Tensor, value: Tensor },
}

impl State {
    fn arrays(&self) -> [&Tensor; 2] {
        match self {
            State::Delta { conv, recurrent } => [conv, recurrent],
            State::Attention { key, value } => [key, value],
        }
    }

    fn with(&self, a: Tensor, b: Tensor) -> Self {
        match self {
            State::Delta { .. } => State::Delta {
                conv: a,
                recurrent: b,
            },
            State::Attention { .. } => State::Attention { key: a, value: b },
        }
    }
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

    /// `x` `[B, T, WIDTH]`, continuing `past`, from position `start`; the state after it if asked.
    fn forward(
        &self,
        x: &Tensor,
        past: Option<&State>,
        start: usize,
        rope: &Rope,
        keep: bool,
    ) -> Result<(Tensor, Option<State>)> {
        let shape = x.shape();
        let (b, t) = (shape[0] as isize, shape[1] as isize);
        let (y, state) = match self {
            Block::Delta {
                qkv,
                decay,
                gate,
                out,
                conv,
            } => {
                let (conv0, recurrent0) = match past {
                    Some(State::Delta { conv, recurrent }) => (Some(conv), Some(recurrent)),
                    _ => (None, None),
                };
                let options = CausalConv1dOptions::default()
                    .with_activation(Activation::Silu)
                    .with_final_state(keep);
                let (c, conv_state) =
                    causal_conv1d(&qkv.forward(x)?, conv, None, conv0, None, options)?;
                let parts = c.split_at(&[HEADS * DELTA, 2 * HEADS * DELTA], -1)?;
                let head = |p: &Tensor| p.reshape(&[b, t, HEADS as isize, DELTA as isize]);
                let g = decay.forward(x)?.as_dtype(DType::F32)?.softplus()?.neg()?;
                let beta = gate.forward(x)?.as_dtype(DType::F32)?.sigmoid()?;
                let options = ChunkGatedDeltaOptions::default().with_final_state(keep);
                let (h, recurrent) = chunk_gated_delta_rule(
                    &head(&parts[0])?,
                    &head(&parts[1])?,
                    &head(&parts[2])?,
                    &g,
                    &beta,
                    recurrent0,
                    None,
                    options,
                )?;
                let y = out.forward(&h.reshape(&[b, t, (HEADS * DELTA) as isize])?)?;
                let state = match (conv_state, recurrent) {
                    (Some(conv), Some(recurrent)) => Some(State::Delta { conv, recurrent }),
                    _ => None,
                };
                (y, state)
            }
            Block::Attention { q, k, v, out } => {
                let heads = |p: &Tensor, h: usize| p.reshape(&[b, t, h as isize, HEAD as isize]);
                let positions = start..start + t as usize;
                let query =
                    rope.apply_range(&heads(&q.forward(x)?, QUERY_HEADS)?, positions.clone())?;
                let key = rope.apply_range(&heads(&k.forward(x)?, KEY_HEADS)?, positions)?;
                let value = heads(&v.forward(x)?, KEY_HEADS)?;
                let (key, value) = match past {
                    Some(State::Attention { key: k0, value: v0 }) => (
                        Tensor::concat(&[k0, &key], 1)?,
                        Tensor::concat(&[v0, &value], 1)?,
                    ),
                    _ => (key, value),
                };
                let options = AttentionOptions::default()
                    .with_causal(true)
                    .with_scale(1.0 / (HEAD as f32).sqrt());
                let h = attention(&query, &key, &value, &AttentionMask::None, options)?;
                let y = out.forward(&h.reshape(&[b, t, (QUERY_HEADS * HEAD) as isize])?)?;
                (y, keep.then_some(State::Attention { key, value }))
            }
        };
        Ok((x.add(&y)?.as_dtype(x.dtype())?, state))
    }
}

#[derive(Clone)]
struct Model {
    embed: Tensor,
    blocks: Vec<Block>,
    rope: Rope,
}

impl Model {
    fn new(dtype: DType) -> Self {
        Model {
            embed: tensor(1, &[VOCAB, WIDTH], dtype),
            blocks: vec![
                Block::delta(100, dtype),
                Block::attention(200, dtype),
                Block::delta(300, dtype),
            ],
            rope: Rope::new(HEAD / 2, 10000.0, 1024, dtype).unwrap(),
        }
    }

    /// Every adapter's factors, in one order: what trains.
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
        let mut at = factors.chunks_exact(2);
        for block in &mut self.blocks {
            for layer in block.layers_mut() {
                let pair = at.next().expect("two factors per layer");
                layer.set_factors(pair[0].clone(), pair[1].clone())?;
            }
        }
        Ok(())
    }

    fn embed_ids(&self, ids: &[i32]) -> Result<Tensor> {
        let at = Tensor::from_i32(ids, &[ids.len()])?;
        self.embed.take(&at, 0)?.expand_dims(0)
    }

    /// The mean negative log-probability of the next token, from the top layer's output.
    fn loss(&self, top: &Tensor, targets: &[i32], skip: usize) -> Result<Tensor> {
        let h = top.squeeze(0)?.rms_norm(None, 1e-6)?;
        let n = h.shape()[0];
        let rows = h.slice(0, skip..n)?;
        let targets = Tensor::from_i32(&targets[skip..], &[n - skip])?;
        chosen_logprob(&rows, &self.embed, &targets, &[], LogprobOptions::default())?
            .logprob
            .mean()?
            .neg()
    }

    fn whole_loss(&self, ids: &[i32], targets: &[i32]) -> Result<Tensor> {
        let mut x = self.embed_ids(ids)?;
        for block in &self.blocks {
            x = block.forward(&x, None, 0, &self.rope, false)?.0;
        }
        self.loss(&x, targets, 0)
    }

    /// The gradients of the whole sequence's loss with respect to [`Self::factors`], and the loss.
    fn gradients(&self, ids: &[i32], targets: &[i32]) -> Result<(f32, Vec<Tensor>)> {
        let mut held = self.clone();
        let one = Tensor::scalar(1.0, DType::F32)?;
        let (outs, grads) = vjp(
            |p| {
                held.set_factors(p)?;
                Ok(vec![held.whole_loss(ids, targets)?])
            },
            &self.factors(),
            &[Some(one)],
        )?;
        Ok((outs[0].item()?, grads))
    }

    /// The same gradients with the first `prefix` tokens run once, block by block, each block's
    /// states handed to the rest of the sequence and their gradients carried back through the
    /// prefix: how a training step shares a prefix between its episodes.
    fn gradients_over_a_prefix(
        &self,
        ids: &[i32],
        targets: &[i32],
        prefix: usize,
    ) -> Result<(f32, Vec<Tensor>)> {
        let mut x = self.embed_ids(&ids[..prefix])?;
        let mut prefix_inputs = Vec::new();
        let mut states = Vec::new();
        for block in &self.blocks {
            prefix_inputs.push(x.clone());
            let (y, state) = block.forward(&x, None, 0, &self.rope, true)?;
            states.push(state.expect("asked for"));
            x = y;
        }
        let mut x = self.embed_ids(&ids[prefix..])?;
        let mut suffix_inputs = Vec::new();
        for (block, state) in self.blocks.iter().zip(&states) {
            suffix_inputs.push(x.clone());
            x = block.forward(&x, Some(state), prefix, &self.rope, false)?.0;
        }
        let one = Tensor::scalar(1.0, DType::F32)?;
        // Only the suffix's tokens are read here: the prefix's reads come through its own output.
        let n = ids.len();
        let (outs, top) = vjp(
            |p| Ok(vec![self.loss_of_suffix(&p[0], targets, prefix, n)?]),
            std::slice::from_ref(&x),
            &[Some(one)],
        )?;
        let mut dy = top[0].clone();
        let mut grads: Vec<Vec<Tensor>> = vec![Vec::new(); self.blocks.len()];
        let mut d_state: Vec<[Tensor; 2]> = Vec::new();
        for l in (0..self.blocks.len()).rev() {
            let (g, dx, ds) =
                self.block_vjp(l, &suffix_inputs[l], Some(&states[l]), prefix, &dy, None)?;
            grads[l] = g;
            d_state.push(ds.expect("a state's gradient"));
            dy = dx;
        }
        d_state.reverse();
        let mut dy: Option<Tensor> = None;
        for l in (0..self.blocks.len()).rev() {
            let out_like = match &dy {
                Some(dy) => dy.clone(),
                None => prefix_inputs[l].zeros_like()?,
            };
            let (g, dx, _) =
                self.block_vjp(l, &prefix_inputs[l], None, 0, &out_like, Some(&d_state[l]))?;
            for (total, more) in grads[l].iter_mut().zip(g) {
                *total = total.add(&more)?;
            }
            dy = Some(dx);
        }
        Ok((outs[0].item()?, grads.into_iter().flatten().collect()))
    }

    fn loss_of_suffix(
        &self,
        top: &Tensor,
        targets: &[i32],
        prefix: usize,
        n: usize,
    ) -> Result<Tensor> {
        let h = top.squeeze(0)?.rms_norm(None, 1e-6)?;
        let targets = Tensor::from_i32(&targets[prefix..], &[n - prefix])?;
        // Each token's term of the whole sequence's mean.
        chosen_logprob(&h, &self.embed, &targets, &[], LogprobOptions::default())?
            .logprob
            .sum()?
            .mul_scalar(-1.0 / n as f64)
    }

    /// One block's product: its adapters' gradients, its input's and, given `past`, the state's.
    /// `state_cotangent` asks instead for the gradient of the state it leaves.
    fn block_vjp(
        &self,
        l: usize,
        x: &Tensor,
        past: Option<&State>,
        start: usize,
        dy: &Tensor,
        state_cotangent: Option<&[Tensor; 2]>,
    ) -> Result<(Vec<Tensor>, Tensor, Option<[Tensor; 2]>)> {
        let mut block = self.blocks[l].clone();
        let factors: Vec<Tensor> = block
            .layers()
            .iter()
            .flat_map(|layer| {
                let lora = layer.lora().unwrap();
                [lora.a.clone(), lora.b.clone()]
            })
            .collect();
        let mut primals = factors.clone();
        primals.push(x.clone());
        if let Some(past) = past {
            let [a, b] = past.arrays();
            primals.push(a.clone());
            primals.push(b.clone());
        }
        let n = factors.len();
        let like = past.cloned();
        let keep = state_cotangent.is_some();
        let mut cotangents = vec![Some(dy.clone())];
        if let Some([a, b]) = state_cotangent {
            cotangents.push(Some(a.clone()));
            cotangents.push(Some(b.clone()));
        }
        let rope = self.rope.clone();
        let (_, mut grads) = vjp(
            |p| {
                let mut at = p[..n].chunks_exact(2);
                for layer in block.layers_mut() {
                    let pair = at.next().expect("two factors per layer");
                    layer.set_factors(pair[0].clone(), pair[1].clone())?;
                }
                let past = like
                    .as_ref()
                    .map(|s| s.with(p[n + 1].clone(), p[n + 2].clone()));
                let (y, state) = block.forward(&p[n], past.as_ref(), start, &rope, keep)?;
                let mut out = vec![y];
                if let Some(state) = state {
                    let [a, b] = state.arrays();
                    out.push(a.clone());
                    out.push(b.clone());
                }
                Ok(out)
            },
            &primals,
            &cotangents,
        )?;
        let d_state = past.is_some().then(|| {
            let b = grads.pop().expect("the state's second");
            let a = grads.pop().expect("the state's first");
            [a, b]
        });
        let dx = grads.pop().expect("the input's");
        Ok((grads, dx, d_state))
    }
}

/// The sequence every case reads, a chunk and a bit: a prefix cut at a whole chunk is shared.
fn sequence() -> (Vec<i32>, Vec<i32>) {
    let ids: Vec<i32> = common::values(9, CHUNK + 16)
        .iter()
        .map(|v| ((v + 1.0) * 32.0) as i32 % VOCAB as i32)
        .collect();
    let mut targets = ids[1..].to_vec();
    targets.push(ids[0]);
    (ids, targets)
}

fn lists(tensors: &[Tensor]) -> Value {
    Value::Array(tensors.iter().map(|t| json!(t.to_f32().unwrap())).collect())
}

fn case(model: &Model, ids: &[i32], targets: &[i32]) -> Value {
    let (loss, mut grads) = model.gradients(ids, targets).unwrap();
    let raw = lists(&grads);
    let options = ClipGradNormOptions {
        max_norm: 0.5,
        ..Default::default()
    };
    let norm = clip_grad_norm(&mut grads, &options).unwrap();
    let mut optimiser = AdamW::new((0.9, 0.999), 1e-5, 0.0);
    optimiser.begin_step();
    let names: Vec<String> = (0..grads.len()).map(|i| i.to_string()).collect();
    let stepped: Vec<Tensor> = model
        .factors()
        .iter()
        .zip(&grads)
        .zip(&names)
        .map(|((p, g), name)| optimiser.update(name, p, g, 1e-3).unwrap())
        .collect();
    let mut after = model.clone();
    after.set_factors(&stepped).unwrap();
    let loss_after = after.whole_loss(ids, targets).unwrap().item().unwrap();
    println!("loss {loss} after a step {loss_after}, gradient norm {norm}");
    assert!(
        loss_after < loss,
        "a step lowers the loss: {loss} to {loss_after}"
    );
    json!({
        "loss": loss,
        "norm": norm,
        "loss_after": loss_after,
        "gradients": raw,
        "stepped": lists(&stepped),
    })
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

/// The largest error over the case's arrays, each over its own largest value; the same for the
/// scalars.
fn against(got: &Value, want: &Value, key: &str) -> f64 {
    got[key]
        .as_array()
        .unwrap()
        .iter()
        .zip(want[key].as_array().unwrap())
        .map(|(a, b)| relative(&floats(a), &floats(b)))
        .fold(0.0, f64::max)
}

/// The bound for a different device's rounding, relative to each array's largest value, measured
/// between Flex and the RTX 5070 Ti, with room for autotune's other kernels.
const GRADIENT_BOUND: [(DType, f64); 2] = [(DType::F32, 1e-5), (DType::BF16, 2e-2)];

#[test]
fn one_step_from_one_source() {
    println!("on {}", dash_mag::describe());
    let (ids, targets) = sequence();
    let mut record = serde_json::Map::new();
    for (name, dtype) in [("f32", DType::F32), ("bf16", DType::BF16)] {
        println!("{name}:");
        record.insert(name.to_string(), case(&Model::new(dtype), &ids, &targets));
    }
    let record = Value::Object(record);
    if let Ok(path) = std::env::var("DASH_MAG_RECORD") {
        std::fs::write(&path, serde_json::to_string(&record).unwrap()).unwrap();
        println!("recorded to {path}");
        return;
    }
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/step.json");
    let want: Value = serde_json::from_str(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    for (name, (_, bound)) in ["f32", "bf16"].iter().zip(GRADIENT_BOUND) {
        let (got, want) = (&record[*name], &want[*name]);
        let gradients = against(got, want, "gradients");
        let stepped = against(got, want, "stepped");
        let scalar = |key: &str| {
            relative(
                &[got[key].as_f64().unwrap() as f32],
                &[want[key].as_f64().unwrap() as f32],
            )
        };
        println!(
            "{name} against the recording: gradients {gradients:.3e}, stepped factors {stepped:.3e}, \
             loss {:.3e}, norm {:.3e}, loss after {:.3e} (bound {bound:.1e})",
            scalar("loss"),
            scalar("norm"),
            scalar("loss_after"),
        );
        assert!(
            gradients <= bound,
            "{name} gradients {gradients:e} above {bound:e}"
        );
        assert!(
            scalar("loss") <= bound && scalar("norm") <= bound,
            "{name} scalars"
        );
    }
    #[cfg(feature = "cuda")]
    {
        assert!(
            dash_ops::conv1d::probe::kernel_launches() > 0,
            "the conv1d kernels ran"
        );
        assert!(
            dash_ops::attention::probe::kernel_launches() > 0,
            "the attention kernels ran"
        );
        assert!(
            dash_ops::logprob::probe::kernel_launches() > 0,
            "the logprob kernels ran"
        );
        assert!(
            dash_ops::gated_delta::chunk::probe::kernel_launches() > 0,
            "the delta rule ran"
        );
    }
}

#[test]
fn the_gradient_agrees_with_central_differences() {
    let (ids, targets) = sequence();
    let model = Model::new(DType::F32);
    let (_, grads) = model.gradients(&ids, &targets).unwrap();
    let factors = model.factors();
    let directions: Vec<Tensor> = factors
        .iter()
        .enumerate()
        .map(|(i, f)| tensor(1000 + i as u64, &f.shape(), DType::F32))
        .collect();
    let eps = 2e-2;
    let at = |sign: f64| -> f64 {
        let moved: Vec<Tensor> = factors
            .iter()
            .zip(&directions)
            .map(|(f, d)| f.add(&d.mul_scalar(sign * eps).unwrap()).unwrap())
            .collect();
        let mut held = model.clone();
        held.set_factors(&moved).unwrap();
        held.whole_loss(&ids, &targets).unwrap().item().unwrap() as f64
    };
    let numeric = (at(1.0) - at(-1.0)) / (2.0 * eps);
    let analytic: f64 = grads
        .iter()
        .zip(&directions)
        .map(|(g, d)| g.mul(d).unwrap().sum().unwrap().item().unwrap() as f64)
        .sum();
    let error = (numeric - analytic).abs() / numeric.abs();
    println!("analytic {analytic} numeric {numeric} relative {error:.3e}");
    assert!(error < 5e-3, "{analytic} against {numeric}: {error:e}");
}

/// The prefix's first read is its own: this holds a prefix the loss reads nothing of, the way a
/// system turn is shared, so its gradient reaches the prefix through the states alone.
#[test]
fn a_shared_prefix_gives_the_whole_sequence_s_gradient() {
    let (ids, targets) = sequence();
    let model = Model::new(DType::F32);
    let n = ids.len();
    // The whole sequence's loss over the suffix's tokens alone.
    let mut held = model.clone();
    let one = Tensor::scalar(1.0, DType::F32).unwrap();
    let (outs, whole) = vjp(
        |p| {
            held.set_factors(p)?;
            let mut x = held.embed_ids(&ids)?;
            for block in &held.blocks {
                x = block.forward(&x, None, 0, &held.rope, false)?.0;
            }
            let suffix = x.slice(1, CHUNK..n)?;
            Ok(vec![held.loss_of_suffix(&suffix, &targets, CHUNK, n)?])
        },
        &model.factors(),
        &[Some(one)],
    )
    .unwrap();
    let (loss, shared) = model
        .gradients_over_a_prefix(&ids, &targets, CHUNK)
        .unwrap();
    let loss_error = relative(&[loss], &[outs[0].item().unwrap()]);
    let error = whole
        .iter()
        .zip(&shared)
        .map(|(a, b)| relative(&b.to_f32().unwrap(), &a.to_f32().unwrap()))
        .fold(0.0, f64::max);
    println!("over a prefix against whole: loss {loss_error:.3e}, gradients {error:.3e}");
    assert!(loss_error < 1e-5, "{loss_error:e}");
    assert!(error < 2e-3, "{error:e}");
}
