//! What dispatching one operation costs through this crate's layer over mlx-c: dependent adds on
//! 1024 f32s, the same scratch benchmark as the Burn comparison's (a thousand ops, median of five
//! after three warm-up runs), and each op evaluated alone.
//!
//!   cargo run --release -p dash-mlx --example dispatch_bench

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("dispatch_bench runs on macOS");
}

#[cfg(target_os = "macos")]
fn main() {
    use dash_mlx::mlx::Array;
    use dash_mlx::mlx::random::normal;
    use dash_mlx::mlx::transforms::eval;
    use std::time::Instant;

    fn median_ms(reps: usize, mut f: impl FnMut()) -> f64 {
        for _ in 0..3 {
            f();
        }
        let mut t: Vec<f64> = (0..reps)
            .map(|_| {
                let s = Instant::now();
                f();
                s.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        t.sort_by(f64::total_cmp);
        t[t.len() / 2]
    }

    let x = normal::<f32>(&[1024], None, None, None).unwrap();
    eval([&x]).unwrap();
    let one = Array::from_f32(1.0);
    let ops = 1000;
    let ms = median_ms(5, || {
        let mut y = x.clone();
        for _ in 0..ops {
            y = y.add(&one).unwrap();
        }
        eval([&y]).unwrap();
    });
    println!(
        "dispatch: {ops} dependent adds on [1024] f32, one evaluation: {ms:.3} ms, {:.2} us an op",
        ms * 1e3 / ops as f64
    );
    let ms = median_ms(5, || {
        let mut y = x.clone();
        for _ in 0..ops {
            y = y.add(&one).unwrap();
            eval([&y]).unwrap();
        }
    });
    println!(
        "dispatch: {ops} dependent adds, each evaluated: {ms:.3} ms, {:.2} us an op",
        ms * 1e3 / ops as f64
    );
}
