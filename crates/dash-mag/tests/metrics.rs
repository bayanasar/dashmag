//! Macro-F1, its rounding, and the choice of a calibration shift, on the host.

use dash_mag::calibrate::{choose, grid};
use dash_mag::metrics::{argmax, macro_f1, round4};

#[test]
fn absent_classes_do_not_count() {
    // Class 2 never occurs in truth: its F1 is left out, not counted as 0.
    let (f1, recall) = macro_f1(&[0, 0, 1, 2], &[0, 1, 1, 0], 3);
    // class 0: tp 1, fp 1, fn 1 -> 0.5; class 1: tp 1, fp 0, fn 1 -> 2/3
    assert!((f1 - (0.5 + 2.0 / 3.0) / 2.0).abs() < 1e-12);
    assert_eq!(recall, [Some(0.5), Some(0.5), None]);
    // Every answer right: 1, and every recall 1.
    let (f1, recall) = macro_f1(&[2, 0, 1], &[2, 0, 1], 3);
    assert_eq!(f1, 1.0);
    assert_eq!(recall, [Some(1.0); 3]);
    // Nothing to score.
    assert!(macro_f1(&[], &[], 2).0.is_nan());
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
    assert_eq!(argmax(&[-1.0, -3.0]), 0);
    assert_eq!(argmax(&[0.0, 0.5, 2.0]), 2);
}

#[test]
fn the_grid_is_numpys() {
    let g = grid(0.0, 4.0, 0.25);
    assert_eq!(g.len(), 17);
    assert_eq!(g[16], 4.0);
    assert_eq!(grid(0.0, 1.0, 0.5), [0.0, 0.5, 1.0]);
    assert_eq!(grid(-1.0, 0.0, 0.5), [-1.0, -0.5, 0.0]);
    assert!(grid(1.0, 0.0, 0.5).is_empty());
}

#[test]
fn the_first_best_shift_wins() {
    // Class 1 is under-called: logits favour class 0 by 1.0 on frames that are class 1.
    let logits = vec![
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![3.0, 0.0],
    ];
    let truth = [1, 1, 0, 0];
    let (best, points) = choose(&logits, &truth, 1, &grid(0.0, 3.0, 0.5));
    assert_eq!(points.len(), 7);
    assert_eq!(
        points[0].macro_f1,
        round4(2.0 / 3.0 / 2.0),
        "all called 0: F1 2/3 and 0"
    );
    assert_eq!(points[0].recall, 0.0);
    // From 1.5 up to 2.0 every frame is right; 1.5 is the first.
    assert_eq!(best.shift, 1.5);
    assert_eq!(best.macro_f1, 1.0);
    assert_eq!(best.recall, 1.0);
    // Past 2.0 class 1 takes a frame of class 0 too.
    assert!(points[5].macro_f1 < 1.0, "{:?}", points[5]);
}
