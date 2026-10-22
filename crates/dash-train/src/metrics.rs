//! Macro-F1 per head, as the recipe computes and rounds it.

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
/// to even on the exact binary value. The recipe picks its best epoch on rounded numbers,
/// so two epochs that tie there are a tie here too.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_classes_do_not_count() {
        // Class 2 never occurs in truth: its F1 is left out, not counted as 0.
        let (f1, recall) = macro_f1(&[0, 0, 1, 2], &[0, 1, 1, 0], 3);
        // class 0: tp 1, fp 1, fn 1 -> 0.5; class 1: tp 1, fp 0, fn 1 -> 2/3
        assert!((f1 - (0.5 + 2.0 / 3.0) / 2.0).abs() < 1e-12);
        assert_eq!(recall, [Some(0.5), Some(0.5), None]);
    }

    #[test]
    fn rounding_is_pythons() {
        assert_eq!(
            round4(0.88845),
            0.8884,
            "0.88845 is 0.888449999... in binary"
        );
        assert_eq!(round4(0.12345678), 0.1235);
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
    }
}
