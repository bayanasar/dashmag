//! A cache written one position per row a step, with `put_along_axis` and with
//! `put_along_axis_owned`: milliseconds a step (min / median / max), each step evaluated. On Burn
//! the owned form writes in place where nothing else holds the buffer, and the other copies the
//! whole tensor; on MLX both copy while the buffer's last GPU work still holds it, as right after
//! an `eval`.
//!
//!   cargo run --release -p dash-mag --example put_owned_bench -- [rows] [columns] [steps]
//!
//! By default `[32, 2^21]` f32 (256 MiB) and 20 steps after 2 of warm-up.

use dash_mag::{DType, Tensor};
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("rows, columns and steps are counts"))
        .collect();
    let rows = args.first().copied().unwrap_or(32);
    let cols = args.get(1).copied().unwrap_or(1 << 21);
    let steps = args.get(2).copied().unwrap_or(20);
    let at = Tensor::from_i32(&vec![7; rows], &[rows, 1]).unwrap();
    let v = Tensor::from_f32(&vec![1.0; rows], &[rows, 1]).unwrap();
    println!(
        "[{rows}, {cols}] f32 ({} MiB), {steps} steps",
        (rows * cols * 4) >> 20
    );
    for owned in [false, true] {
        let mut x = Tensor::zeros(&[rows, cols], DType::F32).unwrap();
        dash_mag::eval(&[&x]).unwrap();
        let mut times = Vec::with_capacity(steps);
        for i in 0..steps + 2 {
            let start = Instant::now();
            x = match owned {
                true => x.put_along_axis_owned(&at, &v, 1),
                false => x.put_along_axis(&at, &v, 1),
            }
            .unwrap();
            dash_mag::eval(&[&x]).unwrap();
            if i >= 2 {
                times.push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{:<22} {:.3} / {:.3} / {:.3} ms a step",
            if owned {
                "put_along_axis_owned"
            } else {
                "put_along_axis"
            },
            times[0],
            times[times.len() / 2],
            times[times.len() - 1]
        );
    }
}
