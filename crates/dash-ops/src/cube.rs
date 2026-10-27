//! What a CubeCL kernel asks of the device it launches on, held against what the device says it
//! has before the kernel launches.
//!
//! Burn 0.22 has one CubeCL backend type for every CubeCL runtime: CUDA, ROCm, wgpu (on Vulkan,
//! Metal, DirectX 12, OpenGL or WebGPU) and the CubeCL CPU runtime. A kernel written for it may
//! therefore meet any of their devices, and they differ in what a kernel may hold and do. An RTX
//! 5070 Ti under CUDA offers 99 KiB of shared memory per cube, 1024 units, planes of 32 and 1024
//! bindings; the same card under wgpu through OpenGL over D3D12 32 KiB, no plane operations and
//! 15 bindings; lavapipe (Vulkan on the CPU, through wgpu) 32 KiB, no plane operations (wgpu
//! offers none on a CPU adapter) and 47 bindings; wgpu at most 65535 cubes along each dimension;
//! the CubeCL CPU runtime as many units per cube as the machine has cores. Plane widths go from 8
//! to 64 across GPUs, and not every device stores bf16 or f16.
//!
//! A kernel states what it asks as [`Requirements`]; the operation launches it where
//! [`Requirements::met_on`] holds and runs its reference elsewhere, so that every device runs every
//! operation, the kernels where they fit.

use burn::backend::DispatchDevice;
use burn::backend::cubecl::dtype_to_storage_type;
use burn::tensor::DType;
use cubecl::client::Client;
use cubecl::features::{Plane, TypeUsage};
use cubecl::ir::{AddressType, DeviceProperties, ElemType, IntKind};

/// What one kernel launch asks of the device.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Requirements {
    /// Units per cube along x and y.
    pub cube_dim: (u32, u32),
    /// Cubes along x and y.
    pub cube_count: (usize, usize),
    /// Bytes of shared memory per cube.
    pub shared_bytes: usize,
    /// Tensors bound at once; the runtime binds one buffer more, for their shapes and strides.
    pub tensors: u32,
    /// The widest vector the kernel reads or writes, in elements.
    pub vector_size: usize,
    /// The plane width a kernel is written for, if it assumes one (a lane from `unit % 32`, a
    /// warp's fragment layout): the device's planes must be exactly that wide.
    pub plane_width: Option<u32>,
    /// For a kernel that uses plane operations at whatever width the device runs them, reading
    /// it from `PLANE_DIM`: the narrowest plane it works with. One that keeps a partial result per
    /// plane in shared memory has room for `units / width` of them. The planes must also fit in
    /// the cube.
    pub min_plane_width: Option<u32>,
    /// The type the kernel indexes its tensors with.
    pub address: AddressType,
    /// Whether the kernel computes with 64-bit integers, which not every device does (WGSL has
    /// them only where the device offers them, OpenGL never).
    pub int64: bool,
}

impl Requirements {
    /// The requirements of a launch of `tensors` tensors and `cube_count` cubes of `cube_dim`
    /// units, using no shared memory, vectors or plane operations, indexing with `u32`; the
    /// fields say more.
    pub(crate) fn new(cube_dim: (u32, u32), cube_count: (usize, usize), tensors: u32) -> Self {
        Self {
            cube_dim,
            cube_count,
            shared_bytes: 0,
            tensors,
            vector_size: 1,
            plane_width: None,
            min_plane_width: None,
            address: AddressType::U32,
            int64: false,
        }
    }

    /// The more demanding of two launches, field by field: what a sequence of kernels asks.
    pub(crate) fn max(self, other: Self) -> Self {
        let plane_width = match (self.plane_width, other.plane_width) {
            (Some(a), Some(b)) => {
                assert_eq!(
                    a, b,
                    "two kernels of one pass written for different plane widths"
                );
                Some(a)
            }
            (a, b) => a.or(b),
        };
        Self {
            cube_dim: (
                self.cube_dim.0.max(other.cube_dim.0),
                self.cube_dim.1.max(other.cube_dim.1),
            ),
            cube_count: (
                self.cube_count.0.max(other.cube_count.0),
                self.cube_count.1.max(other.cube_count.1),
            ),
            shared_bytes: self.shared_bytes.max(other.shared_bytes),
            tensors: self.tensors.max(other.tensors),
            vector_size: self.vector_size.max(other.vector_size),
            plane_width,
            min_plane_width: self.min_plane_width.max(other.min_plane_width),
            address: self.address.max(other.address),
            int64: self.int64 || other.int64,
        }
    }

    /// Whether the device `client` serves can launch the kernel on tensors of `dtypes`.
    pub(crate) fn met_on(&self, client: &Client, dtypes: &[DType]) -> bool {
        self.met_by(client.properties(), dtypes)
    }

    /// Whether a device of these `properties` can launch the kernel on tensors of `dtypes`: the
    /// cube's shape and the cube count within its limits, the shared memory, the bindings and the
    /// vector width within what it offers, plane operations on planes as wide as the kernel
    /// takes, the address type one it indexes with, 64-bit integers if the kernel computes with
    /// them, and every dtype one it stores and converts.
    pub(crate) fn met_by(&self, properties: &DeviceProperties, dtypes: &[DType]) -> bool {
        let hardware = &properties.hardware;
        let (x, y) = self.cube_dim;
        let cube_dim = x <= hardware.max_cube_dim.0
            && y <= hardware.max_cube_dim.1
            && x * y <= hardware.max_units_per_cube;
        let cube_count = self.cube_count.0 <= hardware.max_cube_count.0 as usize
            && self.cube_count.1 <= hardware.max_cube_count.1 as usize;
        let plane_ops = properties.features.plane.contains(Plane::Ops);
        let exact_planes = self.plane_width.is_none_or(|width| {
            plane_ops && hardware.plane_size_min == width && hardware.plane_size_max == width
        });
        let any_planes = self.min_plane_width.is_none_or(|width| {
            plane_ops && hardware.plane_size_min >= width && hardware.plane_size_max <= x * y
        });
        let int64 = !self.int64
            || properties
                .type_usage(ElemType::Int(IntKind::I64))
                .is_superset(TypeUsage::Conversion | TypeUsage::Arithmetic);
        let dtypes = dtypes.iter().all(|&dtype| {
            properties
                .type_usage(dtype_to_storage_type(dtype))
                .is_superset(TypeUsage::Buffer | TypeUsage::Conversion)
        });
        cube_dim
            && cube_count
            && self.shared_bytes <= hardware.max_shared_memory_size
            && self.tensors < hardware.max_bindings
            && self.vector_size <= hardware.max_vector_size
            && properties.supports_address(self.address)
            && exact_planes
            && any_planes
            && int64
            && dtypes
    }
}

/// The compute client of `device` when it is a CubeCL device, with or without autodiff.
pub(crate) fn client(device: &DispatchDevice) -> Option<Client> {
    match device {
        DispatchDevice::Cube(device) => Some(device.client()),
        DispatchDevice::Autodiff(device) => client(device),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::features::Features;
    use cubecl::ir::{DeviceIdentity, FloatKind, HardwareProperties, MemoryDeviceProperties};
    use cubecl::profile::TimingMethod;

    /// The properties the requirements read of an RTX 5070 Ti under CUDA, as measured.
    fn rtx_5070_ti() -> DeviceProperties {
        let hardware = HardwareProperties {
            load_width: 128,
            vector_register_count: None,
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 1024,
            max_shared_memory_size: 101376,
            max_cube_count: (2147483647, 65535, 65535),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 64),
            num_streaming_multiprocessors: Some(70),
            num_cpu_cores: None,
            last_level_cache_size: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: 4,
            cube_mma_reserved_shared_memory: 0,
        };
        let mut properties = DeviceProperties::new(
            Features::default(),
            MemoryDeviceProperties::new(1 << 30, 256),
            hardware,
            TimingMethod::Device,
            DeviceIdentity::default(),
        );
        properties.features.plane.insert(Plane::Ops);
        properties.register_address_type(AddressType::U32);
        let (f32, bf16) = (FloatKind::F32, FloatKind::BF16);
        properties.register_type_usage(ElemType::Float(f32), TypeUsage::all());
        let storage = TypeUsage::Buffer | TypeUsage::Conversion;
        properties.register_type_usage(ElemType::Float(bf16), storage);
        properties.register_type_usage(ElemType::Int(IntKind::I64), TypeUsage::all());
        properties
    }

    /// What the chunked gated delta rule's backward pass asks.
    fn backward() -> Requirements {
        Requirements {
            shared_bytes: 100352,
            vector_size: 4,
            plane_width: Some(32),
            ..Requirements::new((256, 1), (1536, 48), 19)
        }
    }

    #[test]
    fn a_device_with_all_it_asks_takes_the_kernel() {
        assert!(backward().met_by(&rtx_5070_ti(), &[DType::F32, DType::BF16]));
    }

    #[test]
    fn each_shortfall_turns_the_kernel_away() {
        type Shorten = fn(&mut DeviceProperties);
        let shortfalls: [(&str, Shorten); 10] = [
            ("shared memory", |p| {
                p.hardware.max_shared_memory_size = 100351
            }),
            ("units per cube", |p| p.hardware.max_units_per_cube = 255),
            ("cube width", |p| p.hardware.max_cube_dim.0 = 128),
            ("cubes along x", |p| p.hardware.max_cube_count.0 = 1535),
            ("cubes along y", |p| p.hardware.max_cube_count.1 = 47),
            ("bindings", |p| p.hardware.max_bindings = 19),
            ("vector width", |p| p.hardware.max_vector_size = 2),
            ("plane operations", |p| {
                p.features.plane.remove(Plane::Ops);
            }),
            ("planes of 16 too", |p| p.hardware.plane_size_min = 16),
            ("planes of 64", |p| {
                p.hardware.plane_size_min = 64;
                p.hardware.plane_size_max = 64;
            }),
        ];
        for (what, shorten) in shortfalls {
            let mut properties = rtx_5070_ti();
            shorten(&mut properties);
            assert!(!backward().met_by(&properties, &[DType::F32]), "{what}");
        }
    }

    #[test]
    fn a_dtype_the_device_does_not_store_turns_the_kernel_away() {
        assert!(!backward().met_by(&rtx_5070_ti(), &[DType::F32, DType::F16]));
    }

    #[test]
    fn wider_indices_than_the_device_has_turn_the_kernel_away() {
        let wide = Requirements {
            address: AddressType::U64,
            ..backward()
        };
        let mut properties = rtx_5070_ti();
        assert!(!wide.met_by(&properties, &[DType::F32]));
        properties.register_address_type(AddressType::U64);
        assert!(wide.met_by(&properties, &[DType::F32]));
    }

    #[test]
    fn a_kernel_without_plane_operations_takes_any_plane_width() {
        let plain = Requirements {
            plane_width: None,
            ..backward()
        };
        let mut properties = rtx_5070_ti();
        properties.features.plane.remove(Plane::Ops);
        properties.hardware.plane_size_min = 8;
        assert!(plain.met_by(&properties, &[DType::F32]));
    }

    /// What a kernel of 256 units asks that reduces over planes of any width, keeping one partial
    /// sum per plane in 32 slots.
    fn any_planes() -> Requirements {
        Requirements {
            min_plane_width: Some(8),
            ..Requirements::new((256, 1), (3072, 1), 6)
        }
    }

    #[test]
    fn plane_operations_of_any_width_take_every_width_from_the_narrowest() {
        let widths: [(&str, u32, u32, bool); 6] = [
            ("CUDA's warps", 32, 32, true),
            ("Intel's 8 to 32", 8, 32, true),
            ("AMD's waves of 64", 64, 64, true),
            ("planes of 4, too many for the slots", 4, 32, false),
            ("planes as wide as the cube", 256, 256, true),
            ("planes wider than the cube", 32, 512, false),
        ];
        for (what, min, max, takes) in widths {
            let mut properties = rtx_5070_ti();
            properties.hardware.plane_size_min = min;
            properties.hardware.plane_size_max = max;
            assert_eq!(
                any_planes().met_by(&properties, &[DType::F32]),
                takes,
                "{what}"
            );
        }
        let mut properties = rtx_5070_ti();
        properties.features.plane.remove(Plane::Ops);
        assert!(
            !any_planes().met_by(&properties, &[DType::F32]),
            "no plane operations"
        );
    }

    #[test]
    fn a_kernel_computing_in_64_bit_integers_takes_a_device_with_them() {
        let int64 = Requirements {
            int64: true,
            ..any_planes()
        };
        assert!(int64.met_by(&rtx_5070_ti(), &[DType::F32]));
        let mut properties = rtx_5070_ti();
        let elem = &mut properties.features.types.elem;
        elem.remove(&ElemType::Int(IntKind::I64));
        assert!(!int64.met_by(&properties, &[DType::F32]));
        assert!(any_planes().met_by(&properties, &[DType::F32]));
    }

    #[test]
    fn a_sequence_of_kernels_asks_the_most_of_each() {
        let a = Requirements {
            shared_bytes: 10,
            plane_width: Some(32),
            ..Requirements::new((256, 1), (100, 1), 4)
        };
        let b = Requirements {
            shared_bytes: 20,
            vector_size: 4,
            ..Requirements::new((32, 8), (3, 64), 9)
        };
        let both = a.max(b);
        assert_eq!(both.cube_dim, (256, 8));
        assert_eq!(both.cube_count, (100, 64));
        assert_eq!(both.shared_bytes, 20);
        assert_eq!(both.tensors, 9);
        assert_eq!(both.vector_size, 4);
        assert_eq!(both.plane_width, Some(32));
        assert_eq!(both.min_plane_width, None);
        assert_eq!(both.address, AddressType::U32);
        assert!(!both.int64);
        assert_eq!(any_planes().max(both).min_plane_width, Some(8));
    }
}
