//! Cross-entropy for several heads, each with its own class weights and masked targets.
//!
//! Written out rather than taken from Burn, because its normalisation is the point:
//! PyTorch's `CrossEntropyLoss(weight=w, ignore_index=-100)` divides the weighted sum by the
//! sum of the weights of the targets present, not by their count. A head whose targets are
//! all masked in a batch is left out, and the rest are averaged — the recipe's
//! `sum(parts) / len(parts)`.

use burn::tensor::activation::log_softmax;
use burn::tensor::{Int, Tensor, TensorData};

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
    fn from_str(s: &str) -> Result<Self, String> {
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

/// One head's loss, or `None` when every target in the batch is masked.
pub fn head_loss(logits: Tensor<2>, targets: &[i64], weights: &[f32]) -> Option<Tensor<1>> {
    if targets.iter().all(|&t| t == IGNORE) {
        return None;
    }
    let device = logits.device();
    let [n, _] = logits.dims();
    let mut index = Vec::with_capacity(n);
    let mut weight = Vec::with_capacity(n);
    let mut total = 0f32;
    for &t in targets {
        if t == IGNORE {
            index.push(0i64);
            weight.push(0f32);
        } else {
            let w = weights[t as usize];
            index.push(t);
            weight.push(w);
            total += w;
        }
    }
    let index = Tensor::<2, Int>::from_data(TensorData::new(index, [n, 1]), &device);
    let weight = Tensor::<1>::from_data(TensorData::new(weight, [n]), &device);
    let picked = log_softmax(logits, 1).gather(1, index).reshape([n]);
    Some(picked.mul(weight).sum().neg().div_scalar(total))
}

/// The heads' losses averaged over the heads with at least one target in the batch.
/// `targets[h]` and `weights[h]` are head `h`'s column and class weights.
pub fn loss(logits: Vec<Tensor<2>>, targets: &[Vec<i64>], weights: &[Vec<f32>]) -> Tensor<1> {
    let parts: Vec<Tensor<1>> = logits
        .into_iter()
        .zip(targets.iter().zip(weights))
        .filter_map(|(l, (t, w))| head_loss(l, t, w))
        .collect();
    let n = parts.len();
    assert!(n > 0, "a batch with every head masked on every frame");
    parts
        .into_iter()
        .reduce(|a, b| a + b)
        .unwrap()
        .div_scalar(n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn t2(rows: &[&[f32]], device: &Device) -> Tensor<2> {
        let n = rows.len();
        let c = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::from_data(TensorData::new(flat, [n, c]), device)
    }

    /// Recorded from torch 2.13: `CrossEntropyLoss(weight=w, ignore_index=-100)` on each
    /// head, then the mean over the heads with a target (the third has none).
    #[test]
    fn the_reduction_is_pytorchs() {
        let device = Device::default();
        let l0 = t2(
            &[
                &[2.0, -1.0, 0.5],
                &[0.1, 0.2, 0.3],
                &[-1.5, 2.5, 0.0],
                &[1.0, 1.0, -2.0],
                &[0.0, -0.5, 3.0],
                &[0.7, 0.1, -0.3],
            ],
            &device,
        );
        let l1 = t2(
            &[
                &[1.2, -0.4],
                &[-0.3, 0.9],
                &[2.0, 0.0],
                &[0.5, 0.5],
                &[-1.0, 1.0],
                &[0.2, -0.2],
            ],
            &device,
        );
        let l2 = t2(&[&[9.0f32, -9.0][..]; 6], &device);
        let targets = vec![
            vec![0, 2, 1, IGNORE, 2, 1],
            vec![1, IGNORE, 0, 0, 1, IGNORE],
            vec![IGNORE; 6],
        ];
        let weights = vec![vec![0.6, 1.1, 1.3], vec![0.8, 1.2], vec![1.0, 1.0]];
        let scalar = |t: Tensor<1>| f64::from(t.into_data().try_to_vec::<f32>().unwrap()[0]);
        let h0 = scalar(head_loss(l0.clone(), &targets[0], &weights[0]).unwrap());
        let h1 = scalar(head_loss(l1.clone(), &targets[1], &weights[1]).unwrap());
        assert!((h0 - 0.5607861876487732).abs() < 1e-6, "{h0}");
        assert!((h1 - 0.7372636795043945).abs() < 1e-6, "{h1}");
        assert!(head_loss(l2.clone(), &targets[2], &weights[2]).is_none());
        let all = scalar(loss(vec![l0, l1, l2], &targets, &weights));
        assert!((all - 0.6490249633789062).abs() < 1e-6, "{all}");
    }

    #[test]
    fn weights_have_mean_one_and_favour_the_rare_class() {
        let t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, IGNORE];
        let w = class_weights(&t, 2, ClassWeight::Sqrt);
        assert!((w[0] + w[1] - 2.0).abs() < 1e-6);
        assert!((w[1] / w[0] - 3.0).abs() < 1e-5, "sqrt(9 / 1): {w:?}");
        assert_eq!(class_weights(&t, 2, ClassWeight::None), [1.0, 1.0]);
    }
}
