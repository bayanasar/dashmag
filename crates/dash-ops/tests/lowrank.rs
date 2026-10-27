//! The low-rank product `base + scale · (x Aᵀ) Bᵀ` against torch, from one recording
//! (`tests/fixtures/lowrank/lowrank.safetensors`): ten cases of seeded inputs, f32, bf16 and f16,
//! ranks 1 to 64, with and without a base, rows given as a matrix and as a batch, and sums long
//! enough to be split into runs; for each, the output and the gradients of `x`, `A`, `B` and the
//! base for an upstream gradient, computed in float64 from the stored inputs, and (but for the long
//! cases) torch's own on the CPU, in PEFT's order (`lora_B(lora_A(x.float())) * scaling`, added to
//! the base and cast back), which the tests print beside.
//!
//! Errors are stated as the largest absolute difference and as that difference over the largest
//! magnitude of the reference, per output and gradient. The same checks run on Flex and on a
//! CubeCL device, any runtime (`DASH_OPS_DEVICE`); each asserts that the kernels ran exactly where
//! `kernels_take` says they take the case (always under CUDA), and that the backward pass written
//! out ran, and skips the cases in a dtype the device does not have.
//!
//!   cargo test --release -p dash-ops --test lowrank -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test lowrank
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test lowrank

mod common;

use burn::tensor::{DType, Device, Tensor, TensorData};
use dash_ops::lowrank::{LowRankOptions, kernels_take, low_rank, probe};
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// The tests read process-wide launch counters, so they run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const FIXTURES: &[u8] = include_bytes!("fixtures/lowrank/lowrank.safetensors");

/// A recorded tensor: its values and shape.
fn recorded(name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
    let tensors = SafeTensors::deserialize(FIXTURES).expect("lowrank fixtures");
    let view = tensors.tensor(name).ok()?;
    let values = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Some((values, view.shape().to_vec()))
}

fn data(name: &str) -> TensorData {
    let (values, shape) = recorded(name).unwrap_or_else(|| panic!("no fixture {name}"));
    TensorData::new(values, shape)
}

/// A recorded tensor as `[rows, width]` (a batch's rows flattened), in `dtype`.
fn load(name: &str, [rows, width]: [usize; 2], device: &Device, dtype: DType) -> Tensor<2> {
    let (values, _) = recorded(name).unwrap_or_else(|| panic!("no fixture {name}"));
    let data = TensorData::new(values, [rows, width]).convert_dtype(dtype);
    Tensor::from_data(data, (device, dtype))
}

/// One recorded case.
#[derive(Clone, Debug)]
struct Case {
    name: String,
    dtype: DType,
    rows: Vec<usize>,
    inp: usize,
    out: usize,
    rank: usize,
    scale: f64,
    base: bool,
}

fn cases() -> Vec<Case> {
    let (_, metadata) = SafeTensors::read_metadata(FIXTURES).expect("lowrank fixtures");
    let cases = metadata.metadata().as_ref().expect("metadata")["cases"].clone();
    let cases: BTreeMap<String, Value> = serde_json::from_str(&cases).expect("cases JSON");
    cases
        .into_iter()
        .map(|(name, c)| Case {
            dtype: match c["dtype"].as_str().unwrap() {
                "f32" => DType::F32,
                "bf16" => DType::BF16,
                "f16" => DType::F16,
                other => panic!("unknown dtype {other}"),
            },
            rows: c["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
            inp: c["inp"].as_u64().unwrap() as usize,
            out: c["out"].as_u64().unwrap() as usize,
            rank: c["rank"].as_u64().unwrap() as usize,
            scale: c["scale"].as_f64().unwrap(),
            base: c["base"].as_bool().unwrap(),
            name,
        })
        .collect()
}

/// `max |got - want|` and that over `max |want|`, over the elements in order (a batch's
/// shape and its flattened rows compare alike).
fn err(got: &TensorData, want: &TensorData) -> (f64, f64) {
    assert_eq!(got.num_elements(), want.num_elements());
    let got: Vec<f64> = got.iter::<f64>().collect();
    let want: Vec<f64> = want.iter::<f64>().collect();
    let mut abs = 0f64;
    let mut scale = 0f64;
    for (g, w) in got.iter().zip(&want) {
        assert!(g.is_finite(), "non-finite value {g}");
        abs = abs.max((g - w).abs());
        scale = scale.max(w.abs());
    }
    (abs, if scale > 0.0 { abs / scale } else { abs })
}

/// The largest relative error each output may have, by its dtype: f32 results within a few f32
/// roundings of the exact value; a result rounded to bf16 or f16, within half a unit of that dtype
/// of it, plus those roundings. Just above the largest measured over every build (Flex; an RTX
/// 5070 Ti through CUDA with Burn's default kernels, `fusion`, `autotune` and both; lavapipe and
/// CubeCL's CPU runtime): f32 6.7e-7 (the CPU runtime's reference; 3.6e-7 from the CUDA kernels,
/// 3.7e-7 on lavapipe, 5.2e-7 on Flex), bf16 3.30e-3, f16 2.78e-4.
fn bound(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => 8e-7,
        DType::BF16 => 3.5e-3,
        DType::F16 => 3e-4,
        other => panic!("no bound for {other:?}"),
    }
}

/// The case's output and gradients on `device` (an autodiff one), against the recording, and
/// whether the kernels and the written-out backward ran.
fn run(case: &Case, device: &Device, name: &str) {
    let p = |part: &str| format!("{}.{part}", case.name);
    let dtype = case.dtype;
    let rows: usize = case.rows.iter().product();
    let (inp, out) = (case.inp, case.out);
    let reshape_rows = |t: Tensor<2>, width: usize| -> Tensor<3> {
        match case.rows.len() {
            1 => t.reshape([1, rows, width]),
            _ => t.reshape([case.rows[0], case.rows[1], width]),
        }
    };
    let x: Tensor<3> = reshape_rows(load(&p("x"), [rows, inp], device, dtype), inp).require_grad();
    let a = load(&p("a"), [case.rank, inp], device, DType::F32).require_grad();
    let b = load(&p("b"), [out, case.rank], device, DType::F32).require_grad();
    let out_dtype = if case.base { dtype } else { DType::F32 };
    let base: Option<Tensor<3>> = case
        .base
        .then(|| reshape_rows(load(&p("base"), [rows, out], device, dtype), out).require_grad());
    let gy: Tensor<3> = reshape_rows(load(&p("gy"), [rows, out], device, out_dtype), out);

    let takes = kernels_take(device, dtype, case.rank, [rows, inp, out]);
    let (launches, runs) = (probe::kernel_launches(), probe::backward_runs());
    let (downs, ups, gathers) = (
        probe::down_launches(),
        probe::up_launches(),
        probe::gather_launches(),
    );
    // PEFT's factors, `A` `[r, in]` and `B` `[out, r]`, as the product takes them: transposed
    // views.
    let y = low_rank(
        x.clone(),
        a.clone().transpose(),
        b.clone().transpose(),
        base.clone(),
        LowRankOptions::default().with_scale(case.scale),
    );
    assert_eq!(
        y.dtype(),
        out_dtype,
        "{name} {}: the output's dtype",
        case.name
    );
    let grads = y.clone().mul(gy).sum().backward();
    let flat = |t: Tensor<3>, width: usize| -> TensorData {
        let t: Tensor<2> = t.reshape([rows, width]);
        t.into_data()
    };
    let mut results = vec![
        ("y", flat(y, out), out_dtype),
        ("dx", flat(x.grad(&grads).unwrap(), inp), dtype),
        ("da", a.grad(&grads).unwrap().into_data(), DType::F32),
        ("db", b.grad(&grads).unwrap().into_data(), DType::F32),
    ];
    if let Some(base) = &base {
        let dbase = flat(base.grad(&grads).unwrap(), out);
        results.push(("dbase", dbase, dtype));
    }
    for (what, got, dtype) in results {
        // The base's gradient is the upstream gradient as it is.
        let want = match what {
            "dbase" => data(&p("gy")),
            _ => data(&p(what)),
        };
        let (abs, rel) = err(&got, &want);
        let torch = recorded(&p(&format!("{what}.torch")))
            .map(|(values, shape)| err(&got, &TensorData::new(values, shape)));
        let torch = torch.map_or(String::new(), |(abs, rel)| {
            format!("   vs torch's: {abs:.3e} / {rel:.3e}")
        });
        println!(
            "{name:<7} {:<15} {what:<6} max abs {abs:.3e}  rel {rel:.3e}{torch}",
            case.name
        );
        let limit = if what == "dbase" { 0.0 } else { bound(dtype) };
        assert!(
            rel <= limit,
            "{name} {} {what}: rel {rel:.3e} above {limit:.1e}",
            case.name
        );
    }

    assert_eq!(
        probe::backward_runs(),
        runs + 1,
        "{name} {}: the written-out backward ran once",
        case.name
    );
    if takes {
        assert_eq!(
            (
                probe::down_launches() - downs,
                probe::up_launches() - ups,
                probe::gather_launches() - gathers
            ),
            (2, 2, 2),
            "{name} {}: down, up and gather each ran forward and backward",
            case.name
        );
    } else {
        assert_eq!(
            probe::kernel_launches(),
            launches,
            "{name} {}: the reference ran, no kernel",
            case.name
        );
    }
}

fn devices() -> Vec<(&'static str, Device)> {
    #[allow(unused_mut)]
    let mut devices = Vec::new();
    #[cfg(feature = "flex")]
    devices.push(("flex", Device::flex().autodiff()));
    #[cfg(feature = "cubecl")]
    {
        let device = common::cube_device();
        println!("{}", common::describe(&device));
        if common::on_cuda(&device) {
            assert!(
                kernels_take(&device, DType::BF16, 16, [2048, 1024, 3584]),
                "CUDA takes the kernels"
            );
        }
        devices.push(("cubecl", device.autodiff()));
    }
    devices
}

#[test]
fn matches_torch() {
    let _serial = serial();
    for (name, device) in devices() {
        for case in cases() {
            if !common::has(&device, case.dtype) {
                println!(
                    "skipped: {name} {} ({:?} not on the device)",
                    case.name, case.dtype
                );
                continue;
            }
            run(&case, &device, name);
        }
    }
}

/// Seeded host values `[rows, width]` from SplitMix64, uniform on `[-1, 1)`.
fn random(seed: u64, shape: [usize; 2]) -> TensorData {
    let mut state = seed;
    let values: Vec<f32> = (0..shape[0] * shape[1])
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            let unit = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64;
            (2.0 * unit - 1.0) as f32
        })
        .collect();
    TensorData::new(values, shape)
}

/// Gradients go only where they are asked for: `x` not tracked, no `dx` (one `up` launch, the
/// forward's); a frozen adapter with a tracked `x`, `dx` alone, and no `gather`.
#[cfg(feature = "cubecl")]
#[test]
fn only_the_gradients_asked_for_are_computed() {
    let _serial = serial();
    let device = common::cube_device().autodiff();
    let (rows, inp, out, rank) = (96, 128, 80, 8);
    if !kernels_take(&device, DType::F32, rank, [rows, inp, out]) {
        println!("skipped: the kernels do not take the case on {device:?}");
        return;
    }
    let t = |seed, shape| Tensor::<2>::from_data(random(seed, shape), (&device, DType::F32));
    let (x, a, b) = (t(1, [rows, inp]), t(2, [rank, inp]), t(3, [out, rank]));
    let options = LowRankOptions::default().with_scale(2.0);

    let (ups, gathers) = (probe::up_launches(), probe::gather_launches());
    let (a_t, b_t) = (a.clone().require_grad(), b.clone().require_grad());
    let y = low_rank(
        x.clone(),
        a_t.clone().transpose(),
        b_t.clone().transpose(),
        None,
        options,
    );
    let grads = y.sum().backward();
    // Read, so that the work Burn's fusion defers has run before the counters are.
    a_t.grad(&grads).expect("dA").into_data();
    b_t.grad(&grads).expect("dB").into_data();
    assert_eq!(probe::up_launches() - ups, 1, "x untracked: no dx");
    assert_eq!(probe::gather_launches() - gathers, 2, "dA and dB");

    let (ups, gathers) = (probe::up_launches(), probe::gather_launches());
    let x_t = x.require_grad();
    let y = low_rank(x_t.clone(), a.transpose(), b.transpose(), None, options);
    let grads = y.sum().backward();
    x_t.grad(&grads).expect("dx").into_data();
    assert_eq!(probe::up_launches() - ups, 2, "dx");
    assert_eq!(
        probe::gather_launches() - gathers,
        0,
        "the factors frozen: no dA, no dB"
    );
}

/// Outside autodiff the kernels give what they give inside it, to the bit; and at a layer's sizes
/// (2048 rows, `in` 1024, `out` 3584, r 16, bf16, sums split into runs), what Flex's reference
/// gives, within the bounds.
#[cfg(feature = "cubecl")]
#[test]
fn kernels_hold_to_the_reference_at_a_layer_s_size() {
    let _serial = serial();
    let device = common::cube_device();
    let (rows, inp, out, rank) = (2048, 1024, 3584, 16);
    if !common::has(&device, DType::BF16) {
        println!("skipped: no bf16 on {device:?}");
        return;
    }
    let flex = Device::flex().autodiff();
    let options = LowRankOptions::default().with_scale(2.0);
    let inputs = |device: &Device| {
        let x = Tensor::<2>::from_data(
            random(4, [rows, inp]).convert_dtype(DType::BF16),
            (device, DType::BF16),
        );
        let a = Tensor::<2>::from_data(random(5, [rank, inp]), (device, DType::F32));
        let b = Tensor::<2>::from_data(random(6, [out, rank]), (device, DType::F32));
        let base = Tensor::<2>::from_data(
            random(7, [rows, out]).convert_dtype(DType::BF16),
            (device, DType::BF16),
        );
        (x, a, b, base)
    };
    let (x, a, b, base) = inputs(&device);
    let plain = low_rank(x, a.transpose(), b.transpose(), Some(base), options);
    let ad = device.clone().autodiff();
    let (x_ad, a_ad, b_ad, base_ad) = inputs(&ad);
    let (x_ad, a_ad, b_ad) = (
        x_ad.require_grad(),
        a_ad.require_grad(),
        b_ad.require_grad(),
    );
    let tracked = low_rank(
        x_ad.clone(),
        a_ad.clone().transpose(),
        b_ad.clone().transpose(),
        Some(base_ad),
        options,
    );
    assert_eq!(
        err(&plain.clone().into_data(), &tracked.clone().into_data()),
        (0.0, 0.0),
        "the same kernels inside and outside autodiff"
    );
    let grads = tracked.clone().sum().backward();

    let (xf, af, bf, basef) = inputs(&flex);
    let (xf, af, bf) = (xf.require_grad(), af.require_grad(), bf.require_grad());
    let reference = low_rank(
        xf.clone(),
        af.clone().transpose(),
        bf.clone().transpose(),
        Some(basef),
        options,
    );
    let grads_f = reference.clone().sum().backward();
    for (what, got, want, dtype) in [
        ("y", tracked.into_data(), reference.into_data(), DType::BF16),
        (
            "dx",
            x_ad.grad(&grads).unwrap().into_data(),
            xf.grad(&grads_f).unwrap().into_data(),
            DType::BF16,
        ),
        (
            "da",
            a_ad.grad(&grads).unwrap().into_data(),
            af.grad(&grads_f).unwrap().into_data(),
            DType::F32,
        ),
        (
            "db",
            b_ad.grad(&grads).unwrap().into_data(),
            bf.grad(&grads_f).unwrap().into_data(),
            DType::F32,
        ),
    ] {
        let (abs, rel) = err(&got, &want);
        println!("layer   {what:<6} vs Flex's reference: max abs {abs:.3e}  rel {rel:.3e}");
        // Two correct bf16 results may round to neighbouring values: one unit, 2^-7 relative
        // (measured: 0 on CUDA, 3.7e-3 on the CPU runtime, which runs the reference); f32 sums
        // over 2048 rows in two orders, some f32 roundings of the largest term apart (measured:
        // 8.9e-7 from the CUDA kernels, 2.2e-6 from the CPU runtime's reference).
        let limit = match dtype {
            DType::BF16 => 2f64.powi(-7),
            _ => 3e-6,
        };
        assert!(
            rel <= limit,
            "layer {what}: rel {rel:.3e} above {limit:.1e}"
        );
    }
}
