//! Macro-F1 per head, as a classification recipe computes and rounds it, on the host.

/// The mean F1 over the classes present in `truth`, and each class's recall (`None` for a
/// class absent from `truth`). Masked frames are left out by the caller.
pub fn macro_f1(pred: &[usize], truth: &[usize], classes: usize) -> (f64, Vec<Option<f64>>) {
    let mut f1s = Vec::new();
    let mut recall = Vec::with_capacity(classes);
    for c in 0..classes {
        let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
        for (&p, &t) in pred.iter().zip(truth) {
            match (p == c, t == c) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
        if tp + fn_ == 0 {
            recall.push(None);
            continue;
        }
        f1s.push(2.0 * tp as f64 / (2 * tp + fp + fn_) as f64);
        recall.push(Some(tp as f64 / (tp + fn_) as f64));
    }
    let f1 = if f1s.is_empty() {
        f64::NAN
    } else {
        f1s.iter().sum::<f64>() / f1s.len() as f64
    };
    (f1, recall)
}

/// Python's `round(x, 4)`: to the nearest of the decimal values with four places, ties
/// to even on the exact binary value. A recipe that picks its best epoch on rounded numbers
/// sees two epochs that tie there as a tie here too.
pub fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap()
}

/// Index of the largest value, the first on a tie (`argmax`).
pub fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}
