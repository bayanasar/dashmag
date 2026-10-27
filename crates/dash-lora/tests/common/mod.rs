//! Helpers shared by the tests: fixtures recorded from PEFT 0.20 and torch 2.13 on the CPU,
//! the devices to run on, and error measurement.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Every tensor of a fixture file, at its stored dtype.
pub fn load(name: &str) -> HashMap<String, TensorData> {
    let bytes = std::fs::read(fixture(name)).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    st.tensors()
        .into_iter()
        .map(|(key, view)| {
            let dtype = match view.dtype() {
                Dtype::F32 => DType::F32,
                Dtype::BF16 => DType::BF16,
                Dtype::F64 => DType::F64,
                Dtype::I64 => DType::I64,
                other => panic!("{key}: {other:?}"),
            };
            let data =
                TensorData::from_bytes_vec(view.data().to_vec(), view.shape().to_vec(), dtype);
            (key, data)
        })
        .collect()
}

pub fn get(set: &HashMap<String, TensorData>, key: &str) -> TensorData {
    set.get(key)
        .unwrap_or_else(|| panic!("fixture has no {key}"))
        .clone()
}

/// A float tensor at the data's own dtype.
pub fn tensor<const D: usize>(data: TensorData, device: &Device) -> Tensor<D> {
    let dtype = data.dtype();
    Tensor::from_data(data, (device, dtype))
}

pub fn int_tensor(data: TensorData, device: &Device) -> Tensor<1, Int> {
    Tensor::from_data(data, device)
}

/// The scalar a one-element f64 fixture holds.
pub fn scalar(data: TensorData) -> f64 {
    data.iter::<f64>().next().unwrap()
}

/// A seeded generator on the host (SplitMix64) for test data. Burn's own random tensors
/// draw from one generator per process, which tests running in parallel threads share and
/// interleave, so their values would depend on what else runs; these depend on the seed alone.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform on [0, 1), in steps of 2^-53.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform on [lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u = 1.0 - self.unit();
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// A tensor of `shape` filled by `draw`, stored at `dtype` (rounded on the host).
    pub fn tensor<const D: usize>(
        &mut self,
        shape: [usize; D],
        dtype: DType,
        device: &Device,
        mut draw: impl FnMut(&mut Self) -> f64,
    ) -> Tensor<D> {
        let n = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|_| draw(self) as f32).collect();
        let data = TensorData::new(values, shape).convert_dtype(dtype);
        Tensor::from_data(data, (device, dtype))
    }
}

/// Errors of `actual` against `expected`, both read as f64: the largest absolute difference,
/// and that difference over the largest absolute expected value (an error relative to the
/// tensor's scale, which elements near zero do not inflate).
#[derive(Clone, Copy, Debug)]
pub struct Err {
    pub abs: f64,
    pub rel: f64,
}

pub fn err(actual: &TensorData, expected: &TensorData) -> Err {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shapes differ: {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );
    let a: Vec<f64> = actual.iter::<f64>().collect();
    let e: Vec<f64> = expected.iter::<f64>().collect();
    let mut abs = 0f64;
    let mut scale = 0f64;
    for (x, y) in a.iter().zip(&e) {
        assert!(x.is_finite(), "non-finite value {x}");
        abs = abs.max((x - y).abs());
        scale = scale.max(y.abs());
    }
    Err {
        abs,
        rel: if scale > 0.0 { abs / scale } else { abs },
    }
}

/// Asserts an error bound. With `DASH_LORA_MEASURE=1` an exceeded bound is printed instead
/// (`BOUND EXCEEDED: ...`), so that one run measures every error of a new configuration
/// (`--nocapture --test-threads 1`), from which the bounds are set.
pub fn bound(ok: bool, message: impl FnOnce() -> String) {
    if ok {
        return;
    }
    let message = message();
    if std::env::var("DASH_LORA_MEASURE").is_ok_and(|v| v == "1") {
        println!("BOUND EXCEEDED: {message}");
    } else {
        panic!("{message}");
    }
}

/// Measures, prints and bounds the error of `actual` against `expected`.
pub fn check(
    what: &str,
    actual: &TensorData,
    expected: &TensorData,
    max_abs: f64,
    max_rel: f64,
) -> Err {
    let e = err(actual, expected);
    println!("{what:<44} max abs {:.3e}  rel {:.3e}", e.abs, e.rel);
    bound(e.abs <= max_abs && e.rel <= max_rel, || {
        format!(
            "{what}: max abs {:.3e} (bound {max_abs:.1e}), rel {:.3e} (bound {max_rel:.1e})",
            e.abs, e.rel
        )
    });
    e
}

/// The devices the tests run on: Flex, and with a CubeCL runtime's feature the device of the
/// runtime `DASH_OPS_DEVICE` names, as dash-ops' tests take it (`cube_device`).
// Each device is pushed under its own feature.
#[allow(unused_mut, clippy::vec_init_then_push)]
pub fn devices() -> Vec<(&'static str, Device)> {
    let mut devices = Vec::new();
    #[cfg(feature = "flex")]
    devices.push(("flex", Device::flex().autodiff()));
    #[cfg(feature = "cubecl")]
    devices.push(("cubecl", cube_device().autodiff()));
    devices
}

/// The [`devices`] that have `dtype`; a test in that dtype skips the others, saying so.
pub fn devices_with(dtype: DType) -> Vec<(&'static str, Device)> {
    devices()
        .into_iter()
        .filter(|(name, device)| {
            let has = device.supports_dtype(dtype);
            if !has {
                println!("skipped: {name} ({device:?}) has no {dtype:?}");
            }
            has
        })
        .collect()
}

/// The CubeCL runtimes, in the order a device is chosen when `DASH_OPS_DEVICE` names none:
/// Burn's own order for its default device.
pub const RUNTIMES: [&str; 7] = ["cuda", "metal", "rocm", "vulkan", "webgpu", "wgpu", "cpu"];

/// The CubeCL device the tests run on beside Flex: that of the runtime `DASH_OPS_DEVICE` names,
/// one of [`RUNTIMES`] whose feature this build has (`wgpu:vulkan`, `wgpu:gl`, `wgpu:dx12` and
/// `wgpu:metal` pin wgpu's graphics API), else of the first of them it has.
///
/// # Panics
///
/// If `DASH_OPS_DEVICE` names a runtime this build does not have.
#[cfg(feature = "cubecl")]
pub fn cube_device() -> Device {
    match std::env::var("DASH_OPS_DEVICE") {
        Ok(name) => runtime(&name).unwrap_or_else(|| {
            panic!("DASH_OPS_DEVICE={name}: not one of {RUNTIMES:?} built in, with its feature")
        }),
        Err(_) => RUNTIMES
            .into_iter()
            .find_map(runtime)
            .expect("every runtime feature turns on `cubecl`, and only they do"),
    }
}

/// The device of the runtime `name`, if this build has its feature.
#[cfg(feature = "cubecl")]
fn runtime(name: &str) -> Option<Device> {
    #[allow(unused_imports)]
    use burn::tensor::DeviceKind;
    match name {
        #[cfg(feature = "cuda")]
        "cuda" => Some(Device::cuda(0)),
        #[cfg(feature = "metal")]
        "metal" => Some(Device::metal(DeviceKind::DefaultDevice)),
        #[cfg(feature = "rocm")]
        "rocm" => Some(Device::rocm(0)),
        #[cfg(feature = "vulkan")]
        "vulkan" => Some(Device::vulkan(DeviceKind::DefaultDevice)),
        #[cfg(feature = "webgpu")]
        "webgpu" => Some(Device::webgpu(DeviceKind::DefaultDevice)),
        #[cfg(feature = "wgpu")]
        "wgpu" => Some(Device::wgpu(DeviceKind::DefaultDevice)),
        #[cfg(feature = "cpu")]
        "cpu" => Some(Device::cpu()),
        #[cfg(feature = "wgpu")]
        pinned if pinned.starts_with("wgpu:") => {
            use burn::backend::{WgpuBackend, WgpuDevice, WgpuDeviceKind};
            let api = match &pinned["wgpu:".len()..] {
                "vulkan" => WgpuBackend::Vulkan,
                "gl" => WgpuBackend::Gl,
                "dx12" => WgpuBackend::Dx12,
                "metal" => WgpuBackend::Metal,
                _ => return None,
            };
            let device = WgpuDevice::new(WgpuDeviceKind::DefaultDevice).on(api);
            Some(Device::new(device))
        }
        _ => None,
    }
}
