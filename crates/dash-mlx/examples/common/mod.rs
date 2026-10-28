//! What every benchmark here measures, the same way: the median wall time of several runs after
//! warm-up, each built, evaluated and synchronised; then, from an emptied cache, MLX's peak
//! allocation above what was live before one more run, its outputs included.

use mlx_rs::error::Result;
use mlx_rs::memory::{active_memory, clear_cache, peak_memory, reset_peak_memory};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Stream};
use std::time::Instant;

fn synchronize() {
    let status = unsafe { mlx_sys::mlx_synchronize(Stream::default().as_ptr()) };
    assert_eq!(status, 0, "mlx_synchronize");
}

/// Median milliseconds and peak MiB above the live arrays of `f`'s runs.
pub fn measure(
    mut f: impl FnMut() -> Result<Vec<Array>>,
    warmup: usize,
    runs: usize,
) -> Result<(f64, f64)> {
    for _ in 0..warmup {
        eval(f()?.iter())?;
        synchronize();
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let out = f()?;
        eval(out.iter())?;
        synchronize();
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    synchronize();
    clear_cache()?;
    let base = active_memory()?;
    reset_peak_memory()?;
    let out = f()?;
    eval(out.iter())?;
    synchronize();
    let peak = (peak_memory()? - base) as f64 / 1048576.0;
    drop(out);
    Ok((times[runs / 2], peak))
}

/// One line of a result table.
pub fn report(op: &str, variant: &str, shape: &str, (ms, mib): (f64, f64)) {
    println!("| {op} | {variant} | {shape} | {ms:.3} ms | +{mib:.0} MiB |");
}

/// Normal samples of `shape`, scaled, as `dtype`.
pub fn normal(shape: &[i32], scale: f32, dtype: mlx_rs::Dtype, seed: u64) -> Result<Array> {
    let key = mlx_rs::random::key(seed)?;
    let x = mlx_rs::random::normal::<f32>(shape, None, None, &key)?;
    x.multiply(Array::from_f32(scale))?.as_dtype(dtype)
}
