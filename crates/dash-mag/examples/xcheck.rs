//! Runs the operations a hybrid model of gated delta and attention layers uses, at its shapes, on
//! synthetic inputs from a fixed seed, and writes each output, so that devices can be compared
//! operation by operation, against each other and against a reference.
//!
//! ```text
//! cargo run --release -p dash-mag --features rocm --example xcheck -- run DIR [TOKENS [CASES]]
//! cargo run --release -p dash-mag --example xcheck -- compare DIR_A DIR_B
//! cargo run --release -p dash-mag --example xcheck -- ulps DIR_BF16 DIR_REFERENCE
//! ```
//!
//! `CASES` is a comma-separated list of `head`, `lse`, `proj`, `mlp`, `elementwise`, `conv`,
//! `gated_delta`, `gda` and `attention`; all of them by default. The environment chooses the rest:
//! - `XCHECK_DTYPE`: `f32` (the default), `bf16`, or `f32r`, the inputs rounded to bf16 and then
//!   computed in f32, the reference a bf16 run is held to (`ulps`);
//! - `XCHECK_GDA_CORR=c`: for `gda`, keys that lean towards one direction per head, as a trained
//!   model's do, with `beta` near 1 and `g` near 0;
//! - `XCHECK_SAVE_INPUTS`: for `gda`, also write its inputs, for another library to run them.
//!
//! The inputs are made from integers and sums of uniform numbers only, so that every machine makes
//! the same bits; each run prints a hash of them to check. Shapes are a 0.8B model's: width 1024,
//! MLP 3584, vocabulary 248320, 8 query and 2 key-value heads of width 256, 16 gated delta heads
//! of width 128, a convolution of width 4 over 6144 channels. `compare` gives, per output, the
//! largest and mean difference and the share of equal elements; `ulps`, a bf16 run's distance
//! from its reference in units in the last place of bf16.

use std::{env, fs, path::Path, time::Instant};

use dash_mag::ops::{
    self, AttentionOptions, CausalConv1dOptions, ChunkGatedDeltaOptions, LogprobOptions,
};
use dash_mag::{DType, Tensor, ops::Activation};

const WIDTH: usize = 1024;
const MLP: usize = 3584;
const VOCAB: usize = 248_320;
const ROWS: usize = 256;
const KEPT_ROWS: usize = 16;

type R<T> = Result<T, Box<dyn std::error::Error>>;

/// One case: what it writes, at a number of tokens.
type Case = fn(&mut Out, usize) -> R<()>;

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1), exactly representable.
    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 23) as f32 - 1.0
    }
    /// Close to a standard normal: the sum of four uniforms, scaled to unit variance.
    fn normal(&mut self) -> f32 {
        let s = self.uniform() + self.uniform() + self.uniform() + self.uniform();
        s * 0.866_025_4
    }
    fn normals(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * scale).collect()
    }
}

fn fnv(values: &[f32], mut h: u64) -> u64 {
    for v in values {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

struct Out<'a> {
    dir: &'a Path,
    inputs: u64,
}

impl Out<'_> {
    fn input(&mut self, values: &[f32]) {
        self.inputs = fnv(values, self.inputs);
    }
    fn save(&self, name: &str, t: &Tensor) -> R<()> {
        let values = t.to_f32()?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(self.dir.join(format!("{name}.f32")), bytes)?;
        let shape: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
        fs::write(self.dir.join(format!("{name}.shape")), shape.join(" "))?;
        Ok(())
    }
}

/// How the inputs are given: `f32`, `bf16`, or `f32r` (rounded to bf16, then computed in f32: the
/// reference a bf16 run is held to). From `XCHECK_DTYPE`, `f32` by default.
fn mode() -> String {
    env::var("XCHECK_DTYPE").unwrap_or_else(|_| "f32".into())
}

fn tensor(out: &mut Out, values: Vec<f32>, shape: &[usize]) -> R<Tensor> {
    out.input(&values);
    let t = Tensor::from_f32(&values, shape)?;
    Ok(match mode().as_str() {
        "bf16" => t.as_dtype(DType::BF16)?,
        "f32r" => t.as_dtype(DType::BF16)?.as_dtype(DType::F32)?,
        _ => t,
    })
}

/// An f32 tensor whatever the mode, as LoRA's factors are.
fn tensor_f32(out: &mut Out, values: Vec<f32>, shape: &[usize]) -> R<Tensor> {
    out.input(&values);
    Ok(Tensor::from_f32(&values, shape)?)
}

/// A projection of the gated delta layer's input, `in_proj_qkv`, with a LoRA adapter of rank 16.
fn projection(out: &mut Out, t: usize) -> R<()> {
    const OUT: usize = 6144;
    let mut rng = Rng(8);
    let x = tensor(out, rng.normals(t * WIDTH, 1.0), &[t, WIDTH])?;
    let w = tensor(out, rng.normals(WIDTH * OUT, 0.03), &[WIDTH, OUT])?;
    let u = tensor_f32(out, rng.normals(WIDTH * 16, 0.03), &[WIDTH, 16])?;
    let v = tensor_f32(out, rng.normals(16 * OUT, 0.03), &[16, OUT])?;
    timed("projection with LoRA", || {
        let base = x.matmul(&w)?;
        out.save("proj_base", &base)?;
        let options = ops::LowRankOptions::default().with_scale(2.0);
        out.save(
            "proj_lora",
            &ops::low_rank(&x, &u, &v, Some(&base), options)?,
        )?;
        Ok(())
    })
}

fn timed<T>(name: &str, f: impl FnOnce() -> R<T>) -> R<T> {
    let start = Instant::now();
    let r = f()?;
    println!("{name}: {:.2} s", start.elapsed().as_secs_f64());
    Ok(r)
}

/// The output layer over the full vocabulary, with each row's target well ahead of the rest, as a
/// trained model's are on most tokens: the loss is small and is the difference of two large
/// numbers.
fn head(out: &mut Out) -> R<()> {
    let mut rng = Rng(1);
    let w = rng.normals(VOCAB * WIDTH, 0.02);
    let targets: Vec<i32> = (0..ROWS)
        .map(|_| (rng.next() % VOCAB as u64) as i32)
        .collect();
    let mut hidden = rng.normals(ROWS * WIDTH, 1.0);
    for (r, &t) in targets.iter().enumerate() {
        let row = &w[t as usize * WIDTH..(t as usize + 1) * WIDTH];
        let norm2: f32 = row.iter().map(|x| x * x).sum();
        // Margins of 14 to 20 above the noise: losses from about 1e-1 down to 1e-4.
        let margin = 14.0 + 6.0 * (r as f32 / ROWS as f32);
        for (h, x) in hidden[r * WIDTH..(r + 1) * WIDTH].iter_mut().zip(row) {
            *h += margin * x / norm2;
        }
    }
    let w = tensor(out, w, &[VOCAB, WIDTH])?;
    let hidden = tensor(out, hidden, &[ROWS, WIDTH])?;
    let flat: Vec<i32> = targets
        .iter()
        .enumerate()
        .map(|(r, &t)| (r * VOCAB) as i32 + t)
        .collect();
    let target_t = Tensor::from_i32(&targets, &[ROWS])?;
    let flat = Tensor::from_i32(&flat, &[ROWS])?;
    timed("head: logits, logsumexp, loss", || {
        let logits = hidden.matmul(&w.t()?)?;
        let lse = logits.logsumexp_axis(-1, false)?;
        let picked = logits.reshape(&[-1])?.take(&flat, 0)?;
        let loss = lse.sub(&picked)?;
        out.save("head_logits_rows", &logits.slice(0, 0..KEPT_ROWS)?)?;
        out.save("head_logsumexp", &lse)?;
        out.save("head_target_logit", &picked)?;
        out.save("head_loss", &loss)?;
        Ok(())
    })?;
    timed("chosen_logprob", || {
        let lp = ops::chosen_logprob(&hidden, &w, &target_t, &[], LogprobOptions::default())?;
        out.save("chosen_logprob", &lp.logprob)?;
        out.save("chosen_entropy", &lp.entropy)?;
        Ok(())
    })
}

/// logsumexp over the vocabulary when one entry dominates, as a confident model's target does:
/// each row is one entry of 0 at a position that moves from row to row, the rest near -20, so the
/// rest add up to about 5e-4 of the total while each is under half a unit in the last place of 1.
/// The same rows are taken in blocks of several sizes, to see whether the sum depends on how a
/// device splits the work.
fn lse(out: &mut Out, _t: usize) -> R<()> {
    let mut rng = Rng(7);
    const MAX_ROWS: usize = 1024;
    let mut x: Vec<f32> = (0..MAX_ROWS * VOCAB)
        .map(|_| -20.0 + 0.5 * rng.uniform())
        .collect();
    for r in 0..MAX_ROWS {
        x[r * VOCAB + (r * 7919 + r * r * 31) % VOCAB] = 0.0;
    }
    let all = tensor(out, x, &[MAX_ROWS, VOCAB])?;
    for rows in [1, 3, 64, 1024] {
        timed(&format!("logsumexp over {rows} rows"), || {
            let lse = all.slice(0, 0..rows)?.logsumexp_axis(-1, false)?;
            out.save(&format!("lse_{rows}"), &lse)?;
            Ok(())
        })?;
    }
    out.save("lse_input_rows", &all.slice(0, 0..64)?)?;
    Ok(())
}

fn mlp_and_norm(out: &mut Out, t: usize) -> R<()> {
    let mut rng = Rng(2);
    let x = tensor(out, rng.normals(t * WIDTH, 1.0), &[t, WIDTH])?;
    let g = tensor(
        out,
        rng.normals(WIDTH, 0.1).iter().map(|v| 1.0 + v).collect(),
        &[WIDTH],
    )?;
    let up = tensor(out, rng.normals(WIDTH * MLP, 0.03), &[WIDTH, MLP])?;
    timed("mlp matmul", || {
        out.save("mlp_up", &x.matmul(&up)?)?;
        Ok(())
    })?;
    timed("rms norm", || {
        // As the application writes it, from array operations.
        let ms = x.square()?.mean_axis(-1, true)?.add_scalar(1e-6)?;
        let by_ops = x.div(&ms.sqrt()?)?.mul(&g)?;
        out.save("rms_norm_ops", &by_ops)?;
        out.save("rms_norm", &x.rms_norm(Some(&g), 1e-6)?)?;
        Ok(())
    })
}

fn elementwise(out: &mut Out, t: usize) -> R<()> {
    let mut rng = Rng(3);
    let a = tensor(out, rng.normals(t * 16, 3.0), &[t, 16])?;
    timed("elementwise", || {
        out.save("softplus", &a.softplus()?)?;
        out.save("sigmoid", &a.sigmoid()?)?;
        out.save("silu", &a.silu()?)?;
        out.save("exp", &a.exp()?)?;
        Ok(())
    })
}

fn conv(out: &mut Out, t: usize) -> R<()> {
    const C: usize = 6144;
    let mut rng = Rng(4);
    let x = tensor(out, rng.normals(t * C, 1.0), &[1, t, C])?;
    let w = tensor(out, rng.normals(C * 4, 0.5), &[C, 4])?;
    timed("causal_conv1d", || {
        let options = CausalConv1dOptions::default().with_activation(Activation::Silu);
        let (y, _) = ops::causal_conv1d(&x, &w, None, None, None, options)?;
        out.save("conv1d", &y)?;
        Ok(())
    })
}

fn gated_delta(out: &mut Out, t: usize) -> R<()> {
    const H: usize = 16;
    const D: usize = 128;
    let mut rng = Rng(5);
    let q = tensor(out, rng.normals(t * H * D, 1.0), &[1, t, H, D])?;
    let k = tensor(out, rng.normals(t * H * D, 1.0), &[1, t, H, D])?;
    let v = tensor(out, rng.normals(t * H * D, 1.0), &[1, t, H, D])?;
    // Log decays in about [-0.5, 0): most heads keep most of their state from token to token.
    let g: Vec<f32> = (0..t * H)
        .map(|_| -0.25 * (rng.uniform() + 1.0) - 1e-3)
        .collect();
    let g = tensor(out, g, &[1, t, H])?;
    let beta: Vec<f32> = (0..t * H).map(|_| 0.5 + 0.49 * rng.uniform()).collect();
    let beta = tensor(out, beta, &[1, t, H])?;
    timed("chunk_gated_delta_rule", || {
        let options = ChunkGatedDeltaOptions::default()
            .with_qk_l2norm(true)
            .with_final_state(true);
        let (o, state) = ops::chunk_gated_delta_rule(&q, &k, &v, &g, &beta, None, None, options)?;
        out.save("gated_delta_out", &o)?;
        out.save("gated_delta_state", &state.ok_or("no final state")?)?;
        Ok(())
    })
}

/// The chunked gated delta rule as an application calls it: queries and keys normalised and scaled
/// by the caller (`qk_l2norm` off, scale 1), `g` in f32 beside the other inputs in the mode's dtype.
/// Run twice: as given, and with `g` one unit in its last place off in about a quarter of its
/// elements, as two devices' `exp` and `softplus` leave it, to see how much that alone moves.
fn gated_delta_as_called(out: &mut Out, t: usize) -> R<()> {
    const H: usize = 16;
    const D: usize = 128;
    let mut rng = Rng(9);
    // `XCHECK_GDA_CORR=c`: each head's queries and keys lean towards one direction of the head's
    // own (`c` times a unit direction, scaled to the noise's length, plus the noise), so that keys
    // are alike as a trained model's are, and beta and the decay sit near 1 and near 0. Unset,
    // they are independent.
    let corr: Option<f32> = env::var("XCHECK_GDA_CORR")
        .ok()
        .and_then(|c| c.parse().ok());
    let directions = rng.normals(H * D, 1.0);
    let unit = |rng: &mut Rng, scale: f32| -> Vec<f32> {
        let mut x = rng.normals(t * H * D, 1.0);
        if let Some(c) = corr {
            for (i, row) in x.chunks_mut(D).enumerate() {
                let dir = &directions[(i % H) * D..(i % H + 1) * D];
                let n = dir.iter().map(|v| v * v).sum::<f32>().sqrt() / (D as f32).sqrt();
                row.iter_mut().zip(dir).for_each(|(v, d)| *v += c * d / n);
            }
        }
        for row in x.chunks_mut(D) {
            let n = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            row.iter_mut().for_each(|v| *v *= scale / n);
        }
        x
    };
    let q = tensor(out, unit(&mut rng, 1.0 / (D as f32).sqrt()), &[1, t, H, D])?;
    let k = tensor(out, unit(&mut rng, 1.0), &[1, t, H, D])?;
    let v = tensor(out, rng.normals(t * H * D, 1.0), &[1, t, H, D])?;
    let beta: Vec<f32> = (0..t * H)
        .map(|_| match corr {
            Some(_) => 0.95 + 0.049 * rng.uniform(),
            None => 0.5 + 0.49 * rng.uniform(),
        })
        .collect();
    let beta = tensor(out, beta, &[1, t, H])?;
    let g: Vec<f32> = (0..t * H)
        .map(|_| match corr {
            Some(_) => -0.01 * (rng.uniform() + 1.0) - 1e-4,
            None => -0.25 * (rng.uniform() + 1.0) - 1e-3,
        })
        .collect();
    let moved: Vec<f32> = g
        .iter()
        .map(|&x| match rng.next() % 1000 {
            0..118 => f32::from_bits(x.to_bits() + 1),
            118..236 => f32::from_bits(x.to_bits() - 1),
            _ => x,
        })
        .collect();
    let options = ChunkGatedDeltaOptions::default()
        .with_qk_l2norm(false)
        .with_scale(1.0)
        .with_final_state(true);
    if env::var_os("XCHECK_SAVE_INPUTS").is_some() {
        for (name, x) in [("in_q", &q), ("in_k", &k), ("in_v", &v), ("in_beta", &beta)] {
            out.save(name, x)?;
        }
    }
    for (name, g) in [("gda", g), ("gda_moved", moved)] {
        let g = tensor_f32(out, g, &[1, t, H])?;
        if env::var_os("XCHECK_SAVE_INPUTS").is_some() {
            out.save(&format!("in_g_{name}"), &g)?;
        }
        timed(name, || {
            let (o, state) =
                ops::chunk_gated_delta_rule(&q, &k, &v, &g, &beta, None, None, options)?;
            out.save(&format!("{name}_out"), &o)?;
            out.save(&format!("{name}_state"), &state.ok_or("no final state")?)?;
            Ok(())
        })?;
    }
    Ok(())
}

fn attention(out: &mut Out, t: usize) -> R<()> {
    const D: usize = 256;
    let mut rng = Rng(6);
    let q = tensor(out, rng.normals(t * 8 * D, 1.0), &[1, t, 8, D])?;
    let k = tensor(out, rng.normals(t * 2 * D, 1.0), &[1, t, 2, D])?;
    let v = tensor(out, rng.normals(t * 2 * D, 1.0), &[1, t, 2, D])?;
    timed("attention", || {
        let o = ops::attention(
            &q,
            &k,
            &v,
            &ops::AttentionMask::None,
            AttentionOptions::causal(),
        )?;
        out.save("attention", &o)?;
        Ok(())
    })
}

fn run(dir: &Path, t: usize, only: Option<&str>) -> R<()> {
    fs::create_dir_all(dir)?;
    println!("on {}, {t} tokens, {}", dash_mag::describe(), mode());
    let mut out = Out {
        dir,
        inputs: 0xcbf2_9ce4_8422_2325,
    };
    let all: [(&str, Case); 9] = [
        ("head", |o, _| head(o)),
        ("proj", projection),
        ("lse", lse),
        ("mlp", mlp_and_norm),
        ("elementwise", elementwise),
        ("conv", conv),
        ("gated_delta", gated_delta),
        ("gda", gated_delta_as_called),
        ("attention", attention),
    ];
    for (name, case) in all {
        if only.is_none_or(|o| o.split(',').any(|x| x == name)) {
            case(&mut out, t)?;
        }
    }
    println!("inputs {:016x}", out.inputs);
    Ok(())
}

fn load(dir: &Path, name: &str) -> R<Vec<f32>> {
    let bytes = fs::read(dir.join(format!("{name}.f32")))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn compare(a: &Path, b: &Path) -> R<()> {
    let mut names: Vec<String> = fs::read_dir(a)?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".f32").map(String::from))
        .collect();
    names.sort();
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "output", "max |a-b|", "max |a|", "max rel", "mean |a-b|", "equal %"
    );
    for name in names {
        let (x, y) = (load(a, &name)?, load(b, &name)?);
        if x.len() != y.len() {
            println!("{name:<20} lengths differ: {} and {}", x.len(), y.len());
            continue;
        }
        let (mut max_d, mut max_a, mut max_rel, mut sum_d, mut equal) = (0f64, 0f64, 0f64, 0f64, 0);
        for (&p, &q) in x.iter().zip(&y) {
            let (p, q) = (p as f64, q as f64);
            let d = (p - q).abs();
            max_d = max_d.max(d);
            max_a = max_a.max(p.abs());
            if p != 0.0 {
                max_rel = max_rel.max(d / p.abs());
            }
            sum_d += d;
            equal += (p == q) as usize;
        }
        println!(
            "{name:<20} {max_d:>12.3e} {max_a:>12.3e} {max_rel:>12.3e} {:>12.3e} {:>8.2}",
            sum_d / x.len() as f64,
            100.0 * equal as f64 / x.len() as f64
        );
    }
    Ok(())
}

/// A bf16 run against its reference: each output against the reference rounded to bf16, in units
/// in the last place of bf16 there.
fn ulps(a: &Path, reference: &Path) -> R<()> {
    let mut names: Vec<String> = fs::read_dir(a)?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".f32").map(String::from))
        .collect();
    names.sort();
    println!(
        "{:<20} {:>9} {:>9} {:>9} {:>9} {:>12}",
        "output", "0 ulp %", "<=1 ulp %", "<=2 ulp %", "max ulp", "mean ulp"
    );
    for name in names {
        let (x, r) = (load(a, &name)?, load(reference, &name)?);
        let (mut e0, mut e1, mut e2, mut max, mut sum, mut n) = (0, 0, 0, 0f64, 0f64, 0);
        for (&y, &r) in x.iter().zip(&r) {
            let rb = half::bf16::from_f32(r);
            let ulp = {
                let next = half::bf16::from_bits(rb.to_bits() + 1).to_f32();
                (next - rb.to_f32()).abs() as f64
            };
            if !ulp.is_finite() || ulp == 0.0 {
                continue;
            }
            let d = ((y - rb.to_f32()).abs() as f64) / ulp;
            n += 1;
            e0 += (d == 0.0) as usize;
            e1 += (d <= 1.0) as usize;
            e2 += (d <= 2.0) as usize;
            max = max.max(d);
            sum += d;
        }
        let pct = |c: usize| 100.0 * c as f64 / n.max(1) as f64;
        println!(
            "{name:<20} {:>9.2} {:>9.2} {:>9.2} {max:>9.1} {:>12.3}",
            pct(e0),
            pct(e1),
            pct(e2),
            sum / n.max(1) as f64
        );
    }
    Ok(())
}

fn main() -> R<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => {
            let t = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4096);
            run(Path::new(&args[1]), t, args.get(3).map(String::as_str))
        }
        Some("compare") => compare(Path::new(&args[1]), Path::new(&args[2])),
        Some("ulps") => ulps(Path::new(&args[1]), Path::new(&args[2])),
        _ => Err("usage: xcheck run DIR [TOKENS [CASES]] | xcheck compare DIR_A DIR_B".into()),
    }
}
