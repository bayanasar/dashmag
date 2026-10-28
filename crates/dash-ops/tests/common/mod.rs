//! What the test files share: the CubeCL device the kernel tests run on.
//!
//! One test file serves every CubeCL runtime. Built with a runtime's feature, its kernel tests run
//! on the device of the runtime `DASH_OPS_DEVICE` names, and hold the kernels to the reference on
//! that device and to the recorded fixtures:
//!
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu

#![allow(dead_code)]

pub mod adapter;

use burn::tensor::Device;

/// The CubeCL runtimes, in the order a device is chosen when `DASH_OPS_DEVICE` names none:
/// Burn's own order for its default device.
pub const RUNTIMES: [&str; 7] = ["cuda", "metal", "rocm", "vulkan", "webgpu", "wgpu", "cpu"];

/// The device the kernel tests run on: that of the runtime `DASH_OPS_DEVICE` names, one of
/// [`RUNTIMES`] whose feature this build has, else of the first of them it has. `wgpu` takes the
/// graphics API wgpu settles on, `wgpu:vulkan`, `wgpu:gl`, `wgpu:dx12` or `wgpu:metal` pins one
/// with this build's compiler (WGSL with `wgpu` alone); wgpu's runtimes take the adapter wgpu
/// prefers, or the one `CUBECL_WGPU_DEFAULT_DEVICE` names (`Cpu`, `DiscreteGpu(0)`, ...).
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

/// Whether `device` stores `dtype`; tests in a dtype it does not have skip, saying so.
pub fn has(device: &Device, dtype: burn::tensor::DType) -> bool {
    let has = device.supports_dtype(dtype);
    if !has {
        println!("skipped: {} has no {dtype:?}", describe(device));
    }
    has
}

/// Whether `device` is CUDA's: every card of compute capability 8.0 or later has all the kernels
/// ask, so tests assert that the kernels ran there, not merely that the query was obeyed.
pub fn on_cuda(device: &Device) -> bool {
    runtime_of(device) == Some("cuda")
}

/// The name of the CubeCL runtime `device` is on (`cuda`, `hip`, `wgpu<wgsl>`, `wgpu<spirv>`,
/// `wgpu<msl>`, `cpu`), `None` for a device not CubeCL's.
pub fn runtime_of(device: &Device) -> Option<&'static str> {
    #[cfg(feature = "cubecl")]
    if let Some(client) = client(device.as_dispatch()) {
        return Some(client.name());
    }
    let _ = device;
    None
}

/// The device's runtime and adapter, and the limits the kernels ask about, for the tests' output.
pub fn describe(device: &Device) -> String {
    #[cfg(feature = "cubecl")]
    if let Some(client) = client(device.as_dispatch()) {
        let properties = client.properties();
        let hardware = &properties.hardware;
        return format!(
            "{} on {}: {} B of shared memory, {} units per cube, planes of {} to {} ({:?}), \
             {} bindings, {:?} cubes",
            client.name(),
            properties.identity.name,
            hardware.max_shared_memory_size,
            hardware.max_units_per_cube,
            hardware.plane_size_min,
            hardware.plane_size_max,
            properties.features.plane,
            hardware.max_bindings,
            hardware.max_cube_count,
        );
    }
    format!("{device:?}")
}

/// The compute client of a CubeCL device, with or without autodiff.
#[cfg(feature = "cubecl")]
fn client(device: &burn::backend::DispatchDevice) -> Option<cubecl::client::Client> {
    use burn::backend::DispatchDevice;
    match device {
        DispatchDevice::Cube(device) => Some(device.client()),
        DispatchDevice::Autodiff(device) => client(device),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}
