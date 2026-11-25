//! Cross-entropy for several heads, each with its own class weights and masked targets: the loss of
//! a classifier with several heads, as the convolutional family trains one.
//!
//! Written out, because its normalisation is the point: PyTorch's
//! `CrossEntropyLoss(weight=w, ignore_index=-100)` divides the weighted sum by the sum of the
//! weights of the targets present, not by their count. A head whose targets are all masked in a
//! batch is left out, and the rest are averaged, `sum(parts) / len(parts)`. The targets and the
//! weights are host values, read once per batch; the logits take their gradient through [`vjp`].
//!
//! [`vjp`]: crate::vjp

use crate::tensor::Tensor;
use crate::{Result, refuse};

/// The target of a head with no label on a frame.
pub const IGNORE: i64 = -100;

/// How a head's classes are weighted against their frequency on the training split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassWeight {
    None,
    /// Inverse square root of the frequency.
    Sqrt,
    /// Inverse frequency.
    Inv,
}

impl std::str::FromStr for ClassWeight {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "none" => Ok(ClassWeight::None),
            "sqrt" => Ok(ClassWeight::Sqrt),
            "inv" => Ok(ClassWeight::Inv),
            other => Err(format!("class weighting {other:?} is none, sqrt or inv")),
        }
    }
}

/// Each class's weight, normalised to a mean of 1 over the head's classes: `(total / (n *
/// count)) ^ p`, a count of 0 taken as 1. `targets` is one head's column.
pub fn class_weights(targets: &[i64], classes: usize, mode: ClassWeight) -> Vec<f32> {
    let mut counts = vec![0f32; classes];
    for &t in targets.iter().filter(|&&t| t != IGNORE) {
        counts[t as usize] += 1.0;
    }
    let counts: Vec<f32> = counts.into_iter().map(|c| c.max(1.0)).collect();
    let total: f32 = counts.iter().sum();
    let w: Vec<f32> = counts
        .iter()
        .map(|&c| match mode {
            ClassWeight::None => 1.0,
            ClassWeight::Sqrt => (total / (classes as f32 * c)).powf(0.5),
            ClassWeight::Inv => total / (classes as f32 * c),
        })
        .collect();
    let mean = w.iter().sum::<f32>() / classes as f32;
    w.into_iter().map(|x| x / mean).collect()
}

/// One head's loss, a scalar in the logits' dtype, or `None` when every target in the batch is
/// masked. `logits` is `[n, classes]`, floating; `targets` holds `n` classes or [`IGNORE`], and
/// `weights` one weight per class. A target out of range is refused.
pub fn head_loss(logits: &Tensor, targets: &[i64], weights: &[f32]) -> Result<Option<Tensor>> {
    let shape = logits.shape();
    if shape.len() != 2 || !logits.dtype().is_floating() {
        return refuse(format!(
            "head_loss: logits of {:?} {shape:?}, not [n, classes] floating",
            logits.dtype()
        ));
    }
    let (n, classes) = (shape[0], shape[1]);
    if targets.len() != n || weights.len() != classes {
        return refuse(format!(
            "head_loss: {} targets and {} weights for logits {shape:?}",
            targets.len(),
            weights.len()
        ));
    }
    if n * classes > i32::MAX as usize {
        return refuse(format!("head_loss: logits {shape:?} past i32's indices"));
    }
    // Each present target's place in the flattened logits, and its weight.
    let mut index = Vec::with_capacity(n);
    let mut weight = Vec::with_capacity(n);
    let mut total = 0f32;
    for (i, &t) in targets.iter().enumerate() {
        if t == IGNORE {
            continue;
        }
        if t < 0 || t as usize >= classes {
            return refuse(format!(
                "head_loss: target {t} of frame {i}, with {classes} classes"
            ));
        }
        let w = weights[t as usize];
        index.push((i * classes + t as usize) as i32);
        weight.push(w);
        total += w;
    }
    if index.is_empty() {
        return Ok(None);
    }
    let dtype = logits.dtype();
    let log_probs = logits.sub(&logits.logsumexp_axis(-1, true)?)?;
    let picked = log_probs
        .reshape(&[-1])?
        .take(&Tensor::from_i32(&index, &[index.len()])?, 0)?;
    let weight = Tensor::from_f32(&weight, &[weight.len()])?.as_dtype(dtype)?;
    let sum = picked.mul(&weight)?.sum()?.neg()?;
    Ok(Some(sum.div(&Tensor::scalar(f64::from(total), dtype)?)?))
}

/// The heads' losses averaged over the heads with at least one target in the batch, a scalar.
/// `logits[h]`, `targets[h]` and `weights[h]` are head `h`'s logits, column and class weights. A
/// batch with every head masked on every frame is refused: it has no loss.
pub fn loss(logits: &[Tensor], targets: &[Vec<i64>], weights: &[Vec<f32>]) -> Result<Tensor> {
    if targets.len() != logits.len() || weights.len() != logits.len() {
        return refuse(format!(
            "loss: {} heads of logits, {} of targets, {} of weights",
            logits.len(),
            targets.len(),
            weights.len()
        ));
    }
    let mut parts = Vec::with_capacity(logits.len());
    for ((l, t), w) in logits.iter().zip(targets).zip(weights) {
        if let Some(part) = head_loss(l, t, w)? {
            parts.push(part);
        }
    }
    let Some(first) = parts.first() else {
        return refuse("loss: a batch with every head masked on every frame");
    };
    let mut sum = first.clone();
    for part in &parts[1..] {
        sum = sum.add(part)?;
    }
    sum.div(&Tensor::scalar(parts.len() as f64, sum.dtype())?)
}
