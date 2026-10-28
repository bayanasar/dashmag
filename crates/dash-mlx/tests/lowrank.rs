//! The low-rank product `base + scale · (x Aᵀ) Bᵀ` against dash-ops' recording
//! (`dash-ops/tests/fixtures/lowrank/lowrank.safetensors`) with dash-ops' bounds: ten cases of
//! seeded inputs, f32, bf16 and f16, ranks 1 to 64, with and without a base, rows as a matrix and
//! as a batch, long sums; the output and the gradients of `x`, `A`, `B` and the base for an
//! upstream gradient, computed in float64 from the stored inputs. PEFT's factors go in as
//! transposed views, as dash-ops takes them. Errors are the largest absolute difference and that
//! over the largest reference magnitude, printed to three figures and rounded up.
//!
//!   MLX_ENABLE_TF32=0 cargo test -p dash-mlx --test lowrank -- --nocapture
#![cfg(target_os = "macos")]

use dash_mlx::lowrank::{LowRankOptions, low_rank};
use mlx_rs::error::Result;
use mlx_rs::transforms::fallible_vjp;
use mlx_rs::{Array, Dtype};
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::BTreeMap;

const FIXTURES: &[u8] = include_bytes!("../../dash-ops/tests/fixtures/lowrank/lowrank.safetensors");

/// dash-ops' bounds by the result's dtype: f32 within a few f32 roundings; a result rounded to
/// bf16 or f16 within half a unit of that dtype, plus those roundings. The base's gradient is the
/// upstream gradient itself.
fn bound(dtype: Dtype) -> f64 {
    match dtype {
        Dtype::Float32 => 8e-7,
        Dtype::Bfloat16 => 3.5e-3,
        Dtype::Float16 => 3e-4,
        other => panic!("no bound for {other:?}"),
    }
}

fn require_ieee_f32_matmul() {
    assert_eq!(
        std::env::var("MLX_ENABLE_TF32").as_deref(),
        Ok("0"),
        "run with MLX_ENABLE_TF32=0: MLX's f32 matmuls run in TF32 otherwise, about 1e-3 off"
    );
}

/// `x` to three significant figures, rounded up: the form the README and PROGRESS quote.
fn up3(x: f64) -> String {
    if x <= 0.0 {
        return "0".into();
    }
    let mut e = x.log10().floor() as i32;
    let mut m = (x / 10f64.powi(e - 2) * (1.0 - 1e-12)).ceil();
    if m >= 1000.0 {
        m /= 10.0;
        e += 1;
    }
    format!("{:.2}e{e}", m / 100.0)
}

fn recorded(name: &str) -> Option<Vec<f32>> {
    let tensors = SafeTensors::deserialize(FIXTURES).expect("lowrank fixtures");
    let view = tensors.tensor(name).ok()?;
    Some(
        view.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    )
}

fn values(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
    a.as_slice::<f32>().to_vec()
}

fn err(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let (mut abs, mut scale) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite value {g}");
        abs = abs.max((*g as f64 - *w as f64).abs());
        scale = scale.max((*w as f64).abs());
    }
    (abs, if scale > 0.0 { abs / scale } else { abs })
}

#[test]
fn matches_torch() -> Result<()> {
    require_ieee_f32_matmul();
    let (_, metadata) = SafeTensors::read_metadata(FIXTURES).expect("lowrank fixtures");
    let cases: BTreeMap<String, Value> =
        serde_json::from_str(&metadata.metadata().as_ref().expect("metadata")["cases"])
            .expect("cases");
    let mut worst: BTreeMap<String, f64> = BTreeMap::new();
    for (name, c) in cases {
        let dtype = match c["dtype"].as_str().unwrap() {
            "f32" => Dtype::Float32,
            "bf16" => Dtype::Bfloat16,
            "f16" => Dtype::Float16,
            other => panic!("unknown dtype {other}"),
        };
        let rows: Vec<i32> = c["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let dim = |k: &str| c[k].as_i64().unwrap() as i32;
        let (inp, out, rank) = (dim("inp"), dim("out"), dim("rank"));
        let scale = c["scale"].as_f64().unwrap();
        let has_base = c["base"].as_bool().unwrap();
        let p = |part: &str| format!("{name}.{part}");
        let shaped = |part: &str, width: i32, dtype: Dtype| -> Result<Array> {
            let mut shape = rows.clone();
            shape.push(width);
            Array::from_slice(&recorded(&p(part)).unwrap(), &shape).as_dtype(dtype)
        };
        let x = shaped("x", inp, dtype)?;
        let a = Array::from_slice(&recorded(&p("a")).unwrap(), &[rank, inp]);
        let b = Array::from_slice(&recorded(&p("b")).unwrap(), &[out, rank]);
        let out_dtype = if has_base { dtype } else { Dtype::Float32 };
        let gy = shaped("gy", out, out_dtype)?;
        let options = LowRankOptions::default().with_scale(scale);
        let mut primals = vec![x, a, b];
        if has_base {
            primals.push(shaped("base", out, dtype)?);
        }
        let f = |args: &[Array]| -> Result<Vec<Array>> {
            // PEFT's factors as the product takes them: transposed views.
            Ok(vec![low_rank(
                &args[0],
                &args[1].t(),
                &args[2].t(),
                args.get(3),
                options,
            )?])
        };
        let (outputs, grads) = fallible_vjp(f, &primals, std::slice::from_ref(&gy))?;
        assert_eq!(outputs[0].dtype(), out_dtype, "{name}: the output's dtype");
        assert_eq!(grads[0].dtype(), dtype, "{name}: dx in x's dtype");
        let mut results = vec![
            ("y", values(&outputs[0]), out_dtype),
            ("dx", values(&grads[0]), dtype),
            ("da", values(&grads[1]), Dtype::Float32),
            ("db", values(&grads[2]), Dtype::Float32),
        ];
        if has_base {
            results.push(("dbase", values(&grads[3]), dtype));
        }
        for (what, got, dtype) in results {
            let want = match what {
                "dbase" => recorded(&p("gy")).unwrap(),
                _ => recorded(&p(what)).unwrap(),
            };
            let (abs, rel) = err(&got, &want);
            println!("{name:<15} {what:<6} max abs {abs:.3e}  rel {}", up3(rel));
            let limit = if what == "dbase" { 0.0 } else { bound(dtype) };
            assert!(
                rel <= limit,
                "{name} {what}: rel {rel:.3e} above {limit:.1e}"
            );
            let entry = worst.entry(format!("{dtype:?}")).or_insert(0.0);
            *entry = entry.max(rel);
        }
    }
    for (dtype, rel) in worst {
        println!("{dtype}: largest {}", up3(rel));
    }
    Ok(())
}

/// By default a run repeats bit for bit.
#[test]
fn repeats_bit_for_bit() -> Result<()> {
    let key = mlx_rs::random::key(3)?;
    let x =
        mlx_rs::random::normal::<f32>(&[64, 256], None, None, &key)?.as_dtype(Dtype::Bfloat16)?;
    let a = mlx_rs::random::normal::<f32>(&[16, 256], None, None, &mlx_rs::random::key(4)?)?;
    let b = mlx_rs::random::normal::<f32>(&[128, 16], None, None, &mlx_rs::random::key(5)?)?;
    let base = mlx_rs::random::normal::<f32>(&[64, 128], None, None, &mlx_rs::random::key(6)?)?
        .as_dtype(Dtype::Bfloat16)?;
    let gy = mlx_rs::random::normal::<f32>(&[64, 128], None, None, &mlx_rs::random::key(7)?)?
        .as_dtype(Dtype::Bfloat16)?;
    let run = || -> Result<Vec<Vec<u32>>> {
        let f = |args: &[Array]| -> Result<Vec<Array>> {
            Ok(vec![low_rank(
                &args[0],
                &args[1].t(),
                &args[2].t(),
                Some(&args[3]),
                LowRankOptions::default().with_scale(2.0),
            )?])
        };
        let (outputs, grads) = fallible_vjp(
            f,
            &[x.clone(), a.clone(), b.clone(), base.clone()],
            std::slice::from_ref(&gy),
        )?;
        Ok(outputs
            .iter()
            .chain(&grads)
            .map(|g| values(g).iter().map(|v| v.to_bits()).collect())
            .collect())
    };
    assert!(run()? == run()?, "two runs differ");
    Ok(())
}
