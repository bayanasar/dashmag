//! The multi-head classifier's loss: its reduction against torch's, its gradient against the one
//! worked out in f64, its class weights, and what it refuses.

use dash_mag::loss::{ClassWeight, IGNORE, class_weights, head_loss, loss};
use dash_mag::{DType, Tensor, vjp};

fn t2(rows: &[&[f32]]) -> Tensor {
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_f32(&flat, &[rows.len(), rows[0].len()]).unwrap()
}

/// Each head's logits, `[frame][class]`, its targets and its class weights.
type Batch = (Vec<Vec<Vec<f32>>>, Vec<Vec<i64>>, Vec<Vec<f32>>);

/// Three heads of six frames: two with masked frames, the third masked on every frame.
fn batch() -> Batch {
    let logits = vec![
        vec![
            vec![2.0, -1.0, 0.5],
            vec![0.1, 0.2, 0.3],
            vec![-1.5, 2.5, 0.0],
            vec![1.0, 1.0, -2.0],
            vec![0.0, -0.5, 3.0],
            vec![0.7, 0.1, -0.3],
        ],
        vec![
            vec![1.2, -0.4],
            vec![-0.3, 0.9],
            vec![2.0, 0.0],
            vec![0.5, 0.5],
            vec![-1.0, 1.0],
            vec![0.2, -0.2],
        ],
        vec![vec![9.0, -9.0]; 6],
    ];
    let targets = vec![
        vec![0, 2, 1, IGNORE, 2, 1],
        vec![1, IGNORE, 0, 0, 1, IGNORE],
        vec![IGNORE; 6],
    ];
    let weights = vec![vec![0.6, 1.1, 1.3], vec![0.8, 1.2], vec![1.0, 1.0]];
    (logits, targets, weights)
}

fn tensors(logits: &[Vec<Vec<f32>>]) -> Vec<Tensor> {
    logits
        .iter()
        .map(|h| t2(&h.iter().map(Vec::as_slice).collect::<Vec<_>>()))
        .collect()
}

/// Recorded from torch 2.13: `CrossEntropyLoss(weight=w, ignore_index=-100)` on each head, then
/// the mean over the heads with a target (the third has none).
#[test]
fn the_reduction_is_torchs() {
    let (logits, targets, weights) = batch();
    let l = tensors(&logits);
    let scalar = |t: Tensor| f64::from(t.item().unwrap());
    let h0 = scalar(head_loss(&l[0], &targets[0], &weights[0]).unwrap().unwrap());
    let h1 = scalar(head_loss(&l[1], &targets[1], &weights[1]).unwrap().unwrap());
    assert!((h0 - 0.5607861876487732).abs() < 1e-6, "{h0}");
    assert!((h1 - 0.7372636795043945).abs() < 1e-6, "{h1}");
    assert!(
        head_loss(&l[2], &targets[2], &weights[2])
            .unwrap()
            .is_none()
    );
    let all = loss(&l, &targets, &weights).unwrap();
    assert_eq!(all.rank(), 0);
    let all = scalar(all);
    assert!((all - 0.6490249633789062).abs() < 1e-6, "{all}");
}

/// The gradient through [`vjp`] against the one worked out in f64: frame `i` of a head with
/// targets, weight `w_i` of the head's total `W`, gets `w_i / W (softmax_i - onehot_i)`, over the
/// number of heads with targets; a masked frame and a head without targets get zeros.
#[test]
fn the_gradient_is_the_weighted_softmax_less_the_target() {
    let (logits, targets, weights) = batch();
    let l = tensors(&logits);
    let one = Tensor::scalar(1.0, DType::F32).unwrap();
    let (_, grads) = vjp(|p| Ok(vec![loss(p, &targets, &weights)?]), &l, &[Some(one)]).unwrap();
    let heads = 2.0;
    let mut worst = 0f64;
    for h in 0..3 {
        let total: f64 = targets[h]
            .iter()
            .filter(|&&t| t != IGNORE)
            .map(|&t| f64::from(weights[h][t as usize]))
            .sum();
        let got = grads[h].to_f32().unwrap();
        let classes = logits[h][0].len();
        for (i, row) in logits[h].iter().enumerate() {
            let max = row
                .iter()
                .fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
            let z: f64 = row.iter().map(|&x| (f64::from(x) - max).exp()).sum();
            for c in 0..classes {
                let want = match targets[h][i] {
                    IGNORE => 0.0,
                    t => {
                        let p = (f64::from(row[c]) - max).exp() / z;
                        let onehot = if c as i64 == t { 1.0 } else { 0.0 };
                        f64::from(weights[h][t as usize]) / total * (p - onehot) / heads
                    }
                };
                worst = worst.max((f64::from(got[i * classes + c]) - want).abs());
            }
        }
    }
    assert!(worst < 1e-7, "worst difference {worst:e}");
}

#[test]
fn weights_have_mean_one_and_favour_the_rare_class() {
    let t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, IGNORE];
    let w = class_weights(&t, 2, ClassWeight::Sqrt);
    assert!((w[0] + w[1] - 2.0).abs() < 1e-6);
    assert!((w[1] / w[0] - 3.0).abs() < 1e-5, "sqrt(9 / 1): {w:?}");
    let w = class_weights(&t, 2, ClassWeight::Inv);
    assert!((w[0] + w[1] - 2.0).abs() < 1e-6);
    assert!((w[1] / w[0] - 9.0).abs() < 1e-5, "9 / 1: {w:?}");
    assert_eq!(class_weights(&t, 2, ClassWeight::None), [1.0, 1.0]);
    // A class never seen counts as seen once.
    let w = class_weights(&[0, 0, 0, 0], 2, ClassWeight::Inv);
    assert!((w[1] / w[0] - 4.0).abs() < 1e-5, "{w:?}");
}

#[test]
fn a_weighting_is_named_as_the_command_line_names_it() {
    assert_eq!("none".parse(), Ok(ClassWeight::None));
    assert_eq!("sqrt".parse(), Ok(ClassWeight::Sqrt));
    assert_eq!("inv".parse(), Ok(ClassWeight::Inv));
    assert!("log".parse::<ClassWeight>().is_err());
}

/// Logits that are not `[n, classes]` floating, targets or weights that do not fit, a target out
/// of range, heads that do not line up, and a batch without a target are refused.
#[test]
fn what_does_not_fit_is_refused() {
    let l = t2(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let w = [1.0, 1.0];
    assert!(head_loss(&l, &[0, 1], &w).unwrap().is_some());
    let flat = Tensor::from_f32(&[1.0, 2.0], &[2]).unwrap();
    assert!(head_loss(&flat, &[0, 1], &w).is_err());
    let ints = Tensor::from_i32(&[1, 2, 3, 4], &[2, 2]).unwrap();
    assert!(head_loss(&ints, &[0, 1], &w).is_err());
    assert!(head_loss(&l, &[0], &w).is_err());
    assert!(head_loss(&l, &[0, 1], &[1.0]).is_err());
    assert!(head_loss(&l, &[0, 2], &w).is_err());
    assert!(head_loss(&l, &[-1, 0], &w).is_err());
    let heads = [l.clone(), l.clone()];
    assert!(loss(&heads, &[vec![0, 1]], &[w.to_vec(), w.to_vec()]).is_err());
    assert!(loss(&heads, &[vec![0, 1], vec![1, 0]], &[w.to_vec()]).is_err());
    let masked = vec![vec![IGNORE, IGNORE]; 2];
    assert!(loss(&heads, &masked, &[w.to_vec(), w.to_vec()]).is_err());
}
