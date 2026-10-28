//! How often top-p's cut differs from the definition's: on rows of 248,320 bf16 logits, flat
//! (`N(0, 1)`, whose crossing falls in tie groups of hundreds of tokens) and peaked (`2 N(0, 1)`
//! with a few tokens raised by up to 6, as a model's), at `p` 0.9 and 0.95 and temperatures 1 and
//! 0.7 (a rollout's, which sharpens the weights and moves the groups' masses at the cut), the cut
//! (the smallest logit kept) from MLX's scan, from the balanced tree dash-ops' reference forms
//! (Hillis and Steele's scan) and from a balanced tree formed in pairs (Blelloch's) against the cut
//! in f64.
//! A row is near a boundary when the f64 crossing lies within 2e-6 of the row's tempered mass of
//! either end of the tie group it falls in: there, the f32 sums may round it into the next group.
//!
//!   cargo run --release -p dash-mlx --example sample_cuts
//!   ROWS=1024 cargo run --release -p dash-mlx --example sample_cuts

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("sample_cuts runs on Apple silicon only");
}

#[cfg(target_os = "macos")]
fn main() -> mlx_rs::error::Result<()> {
    cuts::run()
}

#[cfg(target_os = "macos")]
mod cuts {
    use dash_mlx::sample::{CutSum, SampleOptions, top_p_cut};
    use mlx_rs::error::Result;
    use mlx_rs::{Array, Dtype};

    const V: usize = 248_320;
    const BATCH: usize = 64;
    /// The margin, as a share of the row's mass, within which a crossing is near a boundary.
    const NEAR: f64 = 2e-6;

    /// xorshift64* and Box-Muller.
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (a, b) = (self.uniform(), self.uniform());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        }
    }

    fn row(seed: u64, peaked: bool) -> Vec<f32> {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let scale = if peaked { 2.0 } else { 1.0 };
        let mut z: Vec<f64> = (0..V).map(|_| scale * rng.normal()).collect();
        if peaked {
            for k in 0..8 {
                let at = ((rng.uniform() * V as f64) as usize).min(V - 1);
                z[at] += 6.0 * (1.0 - k as f64 / 8.0);
            }
        }
        z.into_iter()
            .map(|x| half::bf16::from_f32(x as f32).to_f32())
            .collect()
    }

    /// The definition's cut in f64 for `z` at `p` (temperature 1), and whether its crossing is
    /// near a tie group's boundary.
    fn exact(sorted: &[f32], p: f64, tau: f64) -> (f32, bool) {
        let m = sorted[0] as f64;
        let mut cumulative = Vec::with_capacity(sorted.len());
        let mut sum = 0f64;
        for &x in sorted {
            sum += ((x as f64 - m) * tau).exp();
            cumulative.push(sum);
        }
        let target = p * sum;
        let i = cumulative.iter().position(|&c| c >= target).unwrap();
        let v = sorted[i];
        let start = sorted.iter().position(|&x| x == v).unwrap();
        let end = start + sorted[start..].iter().take_while(|&&x| x == v).count();
        let before = if start == 0 {
            0.0
        } else {
            cumulative[start - 1]
        };
        let near = target - before <= NEAR * sum || cumulative[end - 1] - target <= NEAR * sum;
        (v, near)
    }

    #[derive(Default)]
    struct Tally {
        rows: usize,
        near: usize,
        scan: [usize; 2],
        tree: [usize; 2],
        pairs: [usize; 2],
    }

    pub fn run() -> Result<()> {
        let rows: usize = std::env::var("ROWS").map_or(40_960, |r| r.parse().unwrap());
        // (p, temperature): the cut at temperature 1, and at a rollout's 0.7.
        let settings = [(0.9f32, 1.0f32), (0.95, 1.0), (0.9, 0.7), (0.95, 0.7)];
        println!(
            "| logits | p | T | rows | near a boundary | scan differs (near, elsewhere) | tree differs (near, elsewhere) | pairs differ (near, elsewhere) |"
        );
        println!("|---|---|---|---|---|---|---|---|");
        for peaked in [false, true] {
            let mut tallies: Vec<Tally> = settings.iter().map(|_| Tally::default()).collect();
            for batch in 0..rows.div_ceil(BATCH) {
                let seeds: Vec<u64> = (0..BATCH)
                    .map(|r| (batch * BATCH + r) as u64 + if peaked { 1 << 40 } else { 0 })
                    .collect();
                // The rows and their sorted copies, on the host's threads.
                let made: Vec<(Vec<f32>, Vec<f32>)> = std::thread::scope(|s| {
                    let handles: Vec<_> = seeds
                        .iter()
                        .map(|&seed| {
                            s.spawn(move || {
                                let z = row(seed, peaked);
                                let mut sorted = z.clone();
                                sorted.sort_unstable_by(|a, b| b.total_cmp(a));
                                (z, sorted)
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let flat: Vec<f32> = made.iter().flat_map(|(z, _)| z.iter().copied()).collect();
                let logits = Array::from_slice(&flat, &[BATCH as i32, V as i32])
                    .as_dtype(Dtype::Bfloat16)?;
                for (tally, &(p, t)) in tallies.iter_mut().zip(&settings) {
                    // The definition's tau: 1 / T rounded to f32.
                    let tau = (1.0 / t).min(f32::MAX) as f64;
                    let options = SampleOptions {
                        top_p: p,
                        temperature: t,
                        ..SampleOptions::default()
                    };
                    let scan = top_p_cut(&logits, &options, CutSum::Scan)?;
                    let tree = top_p_cut(&logits, &options, CutSum::Tree)?;
                    let pairs = top_p_cut(&logits, &options, CutSum::Pairs)?;
                    let (scan, tree, pairs) = (
                        scan.as_slice::<f32>(),
                        tree.as_slice::<f32>(),
                        pairs.as_slice::<f32>(),
                    );
                    let want: Vec<(f32, bool)> = std::thread::scope(|s| {
                        let handles: Vec<_> = made
                            .iter()
                            .map(|(_, sorted)| s.spawn(move || exact(sorted, p as f64, tau)))
                            .collect();
                        handles.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    for (r, &(v, near)) in want.iter().enumerate() {
                        tally.rows += 1;
                        tally.near += usize::from(near);
                        let at = usize::from(!near);
                        tally.scan[at] += usize::from(scan[r] != v);
                        tally.tree[at] += usize::from(tree[r] != v);
                        tally.pairs[at] += usize::from(pairs[r] != v);
                    }
                }
            }
            for (tally, (p, t)) in tallies.iter().zip(settings) {
                println!(
                    "| {} | {p} | {t} | {} | {} | {} ({}, {}) | {} ({}, {}) | {} ({}, {}) |",
                    if peaked { "peaked" } else { "flat" },
                    tally.rows,
                    tally.near,
                    tally.scan[0] + tally.scan[1],
                    tally.scan[0],
                    tally.scan[1],
                    tally.tree[0] + tally.tree[1],
                    tally.tree[0],
                    tally.tree[1],
                    tally.pairs[0] + tally.pairs[1],
                    tally.pairs[0],
                    tally.pairs[1],
                );
            }
        }
        Ok(())
    }
}
