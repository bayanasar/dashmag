//! GPTQ's symmetric int4 in MLX's layout (`dash_mlx::quantized`). A seeded projection (`in` 256,
//! `out` 16, groups of 128) is packed as GPTQ packs it and repacked, and MLX's dequantize of the
//! result is compared with `(q - 8) · scale` computed on the host in f32:
//!
//! - f32 scales: bit for bit;
//! - f16 scales: bit for bit with that value rounded once to f16;
//! - bf16 scales: within one bf16 unit of that value rounded once to bf16. Each weight is `(q - 8)`
//!   times the scale rounded to bf16, rounded once, so the scale's rounding is the only difference.
//!
//! MLX's dequantize is called through the bindings, since `src/mlx` holds only what the operations
//! call.
//!
//!   cargo test -p dash-mlx --test quantized
#![cfg(target_os = "macos")]

use dash_mlx::mlx::{Array, Dtype, Stream, sys};
use dash_mlx::quantized::{BITS, GROUP_SIZE, gptq_sym4_to_mlx};
use half::{bf16, f16};
use std::ffi::CString;

const IN: usize = 256;
const OUT: usize = 16;

/// A seeded generator: xorshift64*, as dash-ops' tests have it.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// A projection as GPTQ stores it: the 4-bit values `q` `[in, out]`, `qweight` I32 `[in/8, out]`
/// with eight of them per word along `in`, low bits first, and F16 `scales` `[in/128, out]`.
struct Gptq {
    q: Vec<u8>,
    qweight: Vec<i32>,
    scales: Vec<f16>,
}

/// Every value of `q` from 0 to 15, and scales from 2^-16 (below f16's normal range) to 1.
fn seeded(seed: u64) -> Gptq {
    let mut rng = Rng::new(seed);
    let q: Vec<u8> = (0..IN * OUT).map(|_| (rng.next() >> 60) as u8).collect();
    let scales: Vec<f16> = (0..IN / GROUP_SIZE as usize * OUT)
        .map(|_| f16::from_f64((-16.0 * rng.uniform()).exp2()))
        .collect();
    let mut qweight = vec![0i32; IN / 8 * OUT];
    for j in 0..IN / 8 {
        for o in 0..OUT {
            let word = (0..8).fold(0u32, |w, k| {
                w | (q[(8 * j + k) * OUT + o] as u32) << (4 * k)
            });
            qweight[j * OUT + o] = word as i32;
        }
    }
    for v in 0..16 {
        assert!(q.contains(&v), "the seed leaves out q = {v}");
    }
    Gptq { q, qweight, scales }
}

impl Gptq {
    /// `(q - 8) · scale` in f32, `[out, in]`, the scale rounded first by `round`: exact in f32.
    fn dequantized(&self, round: impl Fn(f32) -> f32) -> Vec<f32> {
        let mut w = vec![0f32; OUT * IN];
        for o in 0..OUT {
            for i in 0..IN {
                let scale = round(self.scales[i / GROUP_SIZE as usize * OUT + o].to_f32());
                w[o * IN + i] = (self.q[i * OUT + o] as f32 - 8.0) * scale;
            }
        }
        w
    }

    fn arrays(&self) -> (Array, Array) {
        let qweight = Array::from_slice(&self.qweight, &[(IN / 8) as i32, OUT as i32]);
        let scales = Array::from_slice(&self.scales, &[IN as i32 / GROUP_SIZE, OUT as i32]);
        (qweight, scales)
    }
}

/// MLX's affine dequantize, through the bindings: `[out, in]` in the scales' dtype.
fn dequantize(w: &Array, scales: &Array, biases: &Array) -> Array {
    let mode = CString::new("affine").unwrap();
    let int = |value| sys::mlx_optional_int {
        value,
        has_value: true,
    };
    let no_dtype = sys::mlx_optional_dtype {
        value: sys::mlx_dtype__MLX_FLOAT32,
        has_value: false,
    };
    let no_global_scale = sys::mlx_array {
        ctx: std::ptr::null_mut(),
    };
    let stream = Stream::new().unwrap();
    // SAFETY: valid handles and a new result handle, which the array takes over.
    unsafe {
        let mut res = sys::mlx_array_new();
        let status = sys::mlx_dequantize(
            &mut res,
            w.as_ptr(),
            scales.as_ptr(),
            biases.as_ptr(),
            int(GROUP_SIZE),
            int(BITS),
            mode.as_ptr(),
            no_global_scale,
            no_dtype,
            stream.as_ptr(),
        );
        let out = Array::from_ptr(res);
        assert_eq!(status, 0, "mlx_dequantize failed");
        out
    }
}

fn repacked(gptq: &Gptq, dtype: Dtype) -> Array {
    let (qweight, scales) = gptq.arrays();
    let (w, s, b) = gptq_sym4_to_mlx(&qweight, &scales, dtype).unwrap();
    assert_eq!(
        (w.shape(), w.dtype()),
        (&[OUT as i32, IN as i32 / 8][..], Dtype::Uint32)
    );
    for a in [&s, &b] {
        assert_eq!(
            (a.shape(), a.dtype()),
            (&[OUT as i32, IN as i32 / GROUP_SIZE][..], dtype)
        );
    }
    let w = dequantize(&w, &s, &b);
    assert_eq!(
        (w.shape(), w.dtype()),
        (&[OUT as i32, IN as i32][..], dtype)
    );
    w
}

/// The distance in units of bf16 between two finite bf16 values.
fn bf16_units(a: bf16, b: bf16) -> u32 {
    let ordered = |x: bf16| {
        let bits = x.to_bits() as i32;
        if bits & 0x8000 != 0 {
            -(bits & 0x7fff)
        } else {
            bits
        }
    };
    ordered(a).abs_diff(ordered(b))
}

#[test]
fn the_repacked_weight_dequantizes_to_q_minus_8_times_the_scale() {
    for seed in [1, 2, 3] {
        let gptq = seeded(seed);
        let exact = gptq.dequantized(|s| s);

        let w = repacked(&gptq, Dtype::Float32);
        let got: Vec<u32> = w.as_slice::<f32>().iter().map(|x| x.to_bits()).collect();
        let want: Vec<u32> = exact.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, want, "f32 scales, seed {seed}");

        let w = repacked(&gptq, Dtype::Float16);
        let got: Vec<u16> = w.as_slice::<f16>().iter().map(|x| x.to_bits()).collect();
        let want: Vec<u16> = exact.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        assert_eq!(got, want, "f16 scales, seed {seed}");

        let w = repacked(&gptq, Dtype::Bfloat16);
        let got = w.as_slice::<bf16>();
        let rounded_scale = gptq.dequantized(|s| bf16::from_f32(s).to_f32());
        let (mut apart, mut most) = (0, 0);
        for (k, (&g, &x)) in got.iter().zip(&exact).enumerate() {
            assert_eq!(
                g.to_bits(),
                bf16::from_f32(rounded_scale[k]).to_bits(),
                "bf16 scales, seed {seed}: weight {k} is not (q - 8) times the rounded scale, rounded once"
            );
            let units = bf16_units(g, bf16::from_f32(x));
            apart += (units > 0) as usize;
            most = most.max(units);
        }
        assert!(
            most <= 1,
            "bf16 scales, seed {seed}: {most} units from (q - 8) · scale rounded once"
        );
        println!(
            "seed {seed}: with bf16 scales, {apart} of {} weights one bf16 unit from (q - 8) · scale rounded once",
            got.len()
        );
    }
}

#[test]
fn shapes_and_dtypes_that_are_not_gptq_s_are_refused() {
    let gptq = seeded(1);
    let (qweight, scales) = gptq.arrays();
    let refused = |q: &Array, s: &Array, dtype| gptq_sym4_to_mlx(q, s, dtype).is_err();

    assert!(refused(
        &qweight.as_dtype(Dtype::Uint32).unwrap(),
        &scales,
        Dtype::Float16
    ));
    assert!(refused(
        &qweight.reshape(&[-1]).unwrap(),
        &scales,
        Dtype::Float16
    ));
    assert!(refused(&scales.t(), &scales, Dtype::Float16));
    assert!(refused(&qweight, &scales.t(), Dtype::Float16));
    assert!(refused(
        &qweight,
        &scales.as_dtype(Dtype::Int32).unwrap(),
        Dtype::Float16
    ));
    assert!(refused(&qweight, &scales, Dtype::Int32));
    assert!(refused(&qweight, &scales, Dtype::Float64));

    // `in` = 64: half a group.
    let half = Array::from_slice(&gptq.qweight[..8 * OUT], &[8, OUT as i32]);
    let one = Array::from_slice(&gptq.scales[..OUT], &[1, OUT as i32]);
    assert!(refused(&half, &one, Dtype::Float16));

    assert!(!refused(&qweight, &scales, Dtype::Float16));
}
