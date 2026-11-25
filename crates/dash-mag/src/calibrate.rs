//! One head's decision boundary for one class, moved by the shift best on validation, on the host.
//!
//! A head trained on a rare class can rank it well and still seldom pick it: the boundary sits
//! where the training prior put it. A constant added to the class's logit moves the boundary
//! without changing the ranking. The constant is the point of a grid with the best macro-F1 for
//! that head on the validation split, the first such point on a tie. Folding it into the head's
//! bias, so that the model carries it, is the application's, which holds the model.

use crate::metrics::{argmax, macro_f1, round4};

/// The values PyTorch's `np.arange(lo, hi + step / 2, step)` gives.
pub fn grid(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let stop = hi + step / 2.0;
    let n = ((stop - lo) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| lo + i as f64 * step).collect()
}

/// One grid point: the shift, the head's macro-F1 and the class's recall, rounded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub shift: f64,
    pub macro_f1: f64,
    pub recall: f64,
}

/// The grid scored on one head's validation logits (`[frame][class]`, masked frames
/// already left out), and the first point with the best macro-F1. `shifts` is not empty.
pub fn choose(
    logits: &[Vec<f32>],
    truth: &[usize],
    class: usize,
    shifts: &[f64],
) -> (Point, Vec<Point>) {
    let classes = logits.first().map_or(0, Vec::len);
    let points: Vec<Point> = shifts
        .iter()
        .map(|&b| {
            let pred: Vec<usize> = logits
                .iter()
                .map(|row| {
                    let mut row = row.clone();
                    row[class] += b as f32;
                    argmax(&row)
                })
                .collect();
            let (f1, recall) = macro_f1(&pred, truth, classes);
            Point {
                shift: round4(b),
                macro_f1: round4(f1),
                recall: recall[class].map_or(f64::NAN, round4),
            }
        })
        .collect();
    let mut best = points[0];
    for p in &points {
        if p.macro_f1 > best.macro_f1 {
            best = *p;
        }
    }
    (best, points)
}
