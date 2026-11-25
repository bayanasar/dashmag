//! `OneCycleLR` against torch's, step by step.

use dash_mag::schedule::OneCycle;

/// `tests/fixtures/onecycle_torch.json`, recorded from torch 2.13: `OneCycleLR(max_lr=[3e-4,
/// 1e-3], total_steps=1875, pct_start=0.1)` over AdamW, the rate of each of two parameter groups
/// and beta1, at every step. Kept in the repository: no generator rebuilds it.
#[test]
fn every_step_is_torchs() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/onecycle_torch.json"
    );
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let steps = j["steps"].as_u64().unwrap() as usize;
    let groups = [
        OneCycle::new(3e-4, steps, 0.1),
        OneCycle::new(1e-3, steps, 0.1),
    ];
    let mut worst: f64 = 0.0;
    for k in 0..steps {
        for (g, sched) in groups.iter().enumerate() {
            let (lr, beta1) = sched.at(k);
            let want_lr = j["lr"][k][g].as_f64().unwrap();
            let want_beta1 = j["beta1"][k][g].as_f64().unwrap();
            worst = worst
                .max((lr - want_lr).abs() / want_lr)
                .max((beta1 - want_beta1).abs() / want_beta1);
        }
    }
    println!("OneCycle, {steps} steps: worst relative difference {worst:e} (bound 1e-12)");
    assert!(worst < 1e-12, "worst relative difference {worst:e}");
}

/// The ends of the cycle as torch's defaults put them: `max_lr / 25` at the first step, the peak
/// where the warm-up ends, `max_lr / 25e4` at the last; beta1 0.95, 0.85 and 0.95.
#[test]
fn the_cycle_starts_peaks_and_ends_where_torch_puts_them() {
    let s = OneCycle::new(1e-3, 100, 0.1);
    let (lr, beta1) = s.at(0);
    assert!((lr - 1e-3 / 25.0).abs() < 1e-18, "{lr}");
    assert!((beta1 - 0.95).abs() < 1e-15, "{beta1}");
    let (lr, beta1) = s.at(9);
    assert!((lr - 1e-3).abs() < 1e-18, "{lr}");
    assert!((beta1 - 0.85).abs() < 1e-15, "{beta1}");
    let (lr, beta1) = s.at(99);
    assert!((lr - 1e-3 / 25e4).abs() < 1e-20, "{lr}");
    assert!((beta1 - 0.95).abs() < 1e-15, "{beta1}");
}
