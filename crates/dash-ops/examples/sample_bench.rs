//! Time of one decoding step's sampling at a 248320-token vocabulary on the first CUDA device:
//! greedy (`greedy`), temperature 0.7 with top-p 0.95 (`top-p`), top-k 50 (`top-k`), and beside
//! them temperature 0.7 alone (`temperature`), with top-k 50 and top-p 0.95 (`top-k-top-p`) and
//! with top-p 1e-6 (`top-p-tiny`); over logits peaked as a model's (`3 N(0, 1)`, eight tokens
//! raised by up to 12) and flat (`N(0, 1)`, where top-p keeps over half the vocabulary and the
//! kernel refines past its fast path). bf16 logits, f32 with `--f32`.
//!
//!   cargo run --release -p dash-ops --features cuda --example sample_bench -- [B ...] [--reps K] \
//!       [--mode MODE] [--kind peaked|flat] [--f32] [--explain]
//!
//! For each batch `B` (default 8 and 32), mode and kind of logits (all, or the one named): K timed
//! calls (default 200) after 20 warm-up ones, a device sync around each, and the median and minimum
//! of their wall time. Run it under `nsys profile -t cuda` for the device time: the kernel is
//! `sample_kernel`, one launch per call. `--explain` also prints, per row, the route the kernel's
//! top-p 0.95 takes (its module documentation, steps 2 and 3), replayed on the host.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("sample_bench measures the CUDA path: build it with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    bench::main();
}

#[cfg(feature = "cuda")]
mod bench {
    use burn::backend::DType;
    use burn::tensor::{Device, Tensor, TensorData};
    use dash_ops::sample::{SampleOptions, kernel_takes, sample};
    use std::time::Instant;

    const V: usize = 248_320;

    pub fn main() {
        let mut batches = Vec::new();
        let mut reps = 200;
        let (mut only_mode, mut only_kind) = (None, None);
        let explain = std::env::args().any(|a| a == "--explain");
        let dtype = match std::env::args().any(|a| a == "--f32") {
            true => DType::F32,
            false => DType::BF16,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--reps" => reps = args.next().expect("a count").parse().expect("a number"),
                "--mode" => only_mode = args.next(),
                "--kind" => only_kind = args.next(),
                "--explain" | "--f32" => {}
                b => batches.push(b.parse::<usize>().expect("a batch size")),
            }
        }
        if batches.is_empty() {
            batches = vec![8, 32];
        }
        let device = Device::cuda(0);
        let modes = [
            ("greedy", "greedy", SampleOptions::greedy()),
            (
                "top-p",
                "T 0.7, top-p 0.95",
                SampleOptions {
                    temperature: 0.7,
                    top_p: 0.95,
                    ..SampleOptions::default()
                },
            ),
            (
                "top-k",
                "top-k 50",
                SampleOptions {
                    top_k: 50,
                    ..SampleOptions::default()
                },
            ),
            (
                "temperature",
                "T 0.7",
                SampleOptions {
                    temperature: 0.7,
                    ..SampleOptions::default()
                },
            ),
            (
                "top-p-tiny",
                "T 0.7, top-p 1e-6",
                SampleOptions {
                    temperature: 0.7,
                    top_p: 1e-6,
                    ..SampleOptions::default()
                },
            ),
            (
                "top-k-top-p",
                "T 0.7, top-k 50, top-p 0.95",
                SampleOptions {
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.95,
                    ..SampleOptions::default()
                },
            ),
        ];
        println!("V {V}, {dtype:?} logits, {reps} timed calls after 20 warm-up ones");
        for &b in &batches {
            assert!(
                kernel_takes(&device, dtype, [b, V]),
                "the kernel takes [{b}, {V}]"
            );
            for (kind, scale, lift) in [("peaked", 3.0, 12.0), ("flat", 1.0, 0.0)] {
                if only_kind.as_deref().is_some_and(|only| only != kind) {
                    continue;
                }
                let values = logits(b as u64, [b, V], scale, lift);
                if explain {
                    explain_rows(kind, &values, [b, V], 1.0 / 0.7, 0.95);
                }
                let logits =
                    Tensor::<2>::from_data(TensorData::new(values, [b, V]), &device).cast(dtype);
                device.sync().unwrap();
                for (name, label, options) in &modes {
                    if only_mode.as_deref().is_some_and(|only| only != *name) {
                        continue;
                    }
                    let mut step = 0u64;
                    let mut call = || {
                        // A decoding loop's offset, one step a call.
                        let options = SampleOptions {
                            offset: step,
                            seed: 7,
                            ..options.clone()
                        };
                        step += 1;
                        let out = sample(logits.clone(), &options);
                        device.sync().unwrap();
                        out
                    };
                    for _ in 0..20 {
                        call();
                    }
                    let mut times: Vec<f64> = (0..reps)
                        .map(|_| {
                            let start = Instant::now();
                            call();
                            start.elapsed().as_secs_f64() * 1e6
                        })
                        .collect();
                    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    println!(
                        "B {b:2}  {kind:6}  {label:18}  wall median {:7.1} us  min {:7.1} us",
                        times[times.len() / 2],
                        times[0]
                    );
                }
            }
        }
    }

    /// What the kernel's top-p path does with each row (the kernel module's steps 2 and 3, in f64
    /// on the host): the candidates above the first bound, and those down to the band where the
    /// weight reaches `p W`; both against the 8192 slots of shared memory CUDA gives.
    fn explain_rows(kind: &str, values: &[f32], [b, v]: [usize; 2], tau: f64, p: f64) {
        for row in 0..b {
            let z: Vec<f64> = values[row * v..(row + 1) * v]
                .iter()
                .map(|&x| burn::backend::bf16::from_f32(x).to_f32() as f64)
                .collect();
            let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let s: Vec<f64> = z.iter().map(|&x| (x - m) * tau).collect();
            let w_total: f64 = s.iter().map(|x| x.exp()).sum();
            let floor = ((1.0 - p) * w_total / v as f64).ln() - 0.01;
            let band = |x: f64| ((((1.0 - x) as f32).to_bits() >> 22) as i64 - 254).min(7) as usize;
            let mut counts = [0usize; 8];
            let mut weights = [0f64; 8];
            for &x in s.iter().filter(|&&x| x >= floor) {
                counts[band(x)] += 1;
                weights[band(x)] += x.exp();
            }
            let candidates: usize = counts.iter().sum();
            let (mut cum, mut narrowed, mut cut) = (0.0, 0, 7);
            for j in 0..8 {
                cum += weights[j];
                narrowed += counts[j];
                if cum >= p * w_total {
                    cut = j;
                    break;
                }
            }
            let path = if candidates <= 8192 {
                "fast"
            } else if narrowed <= 8192 {
                "narrowed"
            } else {
                "radix"
            };
            println!(
                "{kind} row {row}: {candidates} candidates, cut band {cut}, {narrowed} narrowed: {path}"
            );
        }
    }

    /// `b` rows of `scale * N(0, 1)` logits, a few of each row's tokens raised by up to `lift`;
    /// xorshift64* and Box-Muller, seeded.
    fn logits(seed: u64, [b, v]: [usize; 2], scale: f64, lift: f64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut uniform = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let x = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(b * v);
        for _ in 0..b {
            let mut row: Vec<f64> = (0..v)
                .map(|_| {
                    let (x, y) = (uniform(), uniform());
                    scale * (-2.0 * x.ln()).sqrt() * (2.0 * std::f64::consts::PI * y).cos()
                })
                .collect();
            if lift > 0.0 {
                for k in 0..8 {
                    let at = ((uniform() * v as f64) as usize).min(v - 1);
                    row[at] += lift * (1.0 - k as f64 / 8.0);
                }
            }
            out.extend(row.into_iter().map(|x| x as f32));
        }
        out
    }
}
