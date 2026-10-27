//! The smallest complete backend extension, in the shape every operation in this crate takes:
//! `axpy(x, y, a) = a * x + y` with
//!
//! - a default body built from Burn's tensor operations: the reference, which Flex runs as is;
//! - a CubeCL kernel for Burn's CubeCL backend, built with any CubeCL runtime's feature (`cuda`,
//!   `rocm`, `wgpu`, `vulkan`, `metal`, `webgpu`, `cpu`) and run on whichever runtime the
//!   tensors' device names, after asking the device whether it can, the reference otherwise;
//! - an implementation for `Autodiff` whose backward pass is written out, so that training
//!   records one node for the operation instead of the reference's intermediate tensors;
//! - a `Tensor` wrapper going in and out of `Dispatch`.
//!
//!   cargo test -p dash-ops --test extension_pattern
//!   CUBECL_CUDA_TF32=0 cargo test -p dash-ops --features cuda --test extension_pattern
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test -p dash-ops --features wgpu --test extension_pattern

mod common;

#[cfg(feature = "flex")]
use burn::backend::Flex;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind, binary};
use burn::backend::{Autodiff, Backend, Dispatch, backend_extension, tensor::FloatTensor};
use burn::tensor::{Device, Tensor, TensorData, Tolerance};
use std::sync::atomic::Ordering;

/// Counters the tests read to tell which path ran, as every operation module's `probe`.
mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[cfg(feature = "cubecl")]
    pub(super) static KERNEL_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static BACKWARD_RUNS: AtomicUsize = AtomicUsize::new(0);

    /// Launches of the CubeCL kernel, so the CubeCL test can tell the kernel ran rather than
    /// the default body.
    #[cfg(feature = "cubecl")]
    pub fn kernel_launches() -> usize {
        KERNEL_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Runs of the written-out backward pass, so the tests can tell it ran rather than the
    /// autodiff of the default body, which gives the same gradients here.
    pub fn backward_runs() -> usize {
        BACKWARD_RUNS.load(Ordering::SeqCst)
    }
}

// Each backend's entry is compiled with the feature that brings it, `cubecl` with any CubeCL
// runtime's. A device whose entry is compiled out panics here rather than run the default body.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(feature = "cubecl"),
    Fusion: cfg(feature = "fusion"),
    Autodiff
)]
pub trait AxpyBackend: Backend {
    /// `a * x + y`, elementwise; `x` and `y` have one shape. Under Burn's fusion the output's
    /// dtype and shape are declared up front, here those of `x`.
    #[fusion(dtype = x, shape = x)]
    fn axpy(x: FloatTensor<Self>, y: FloatTensor<Self>, a: f32) -> FloatTensor<Self> {
        reference::<Self>(x, y, a)
    }
}

/// The reference, from Burn's tensor operations: the default body, and what a kernel runs instead
/// on a device that cannot launch it.
fn reference<B: Backend>(x: FloatTensor<B>, y: FloatTensor<B>, a: f32) -> FloatTensor<B> {
    B::float_add(B::float_mul_scalar(x, a.into()), y)
}

#[cfg(feature = "flex")]
impl AxpyBackend for Flex {}

/// One implementation for every CubeCL runtime: Burn 0.22 has a single CubeCL backend type, and
/// the device a tensor is on names the runtime its kernels compile for (CUDA, HIP, WGSL, SPIR-V,
/// MSL, or LLVM for the CPU).
#[cfg(feature = "cubecl")]
mod cube {
    use super::{AxpyBackend, probe, reference};
    use burn::backend::tensor::FloatTensor;
    use burn_cubecl::CubeBackend;
    use burn_cubecl::ops::numeric::empty_device_dtype;
    use cubecl::std::tensor::layout::linear::{LinearView, LinearViewMut};
    use cubecl::{calculate_cube_count_elemwise, prelude::*};
    use std::sync::atomic::Ordering;

    #[cube(launch_unchecked, address_type = "dynamic")]
    fn axpy_kernel<E: Float>(
        x: LinearView<'_, E>,
        y: LinearView<'_, E>,
        mut out: LinearViewMut<'_, E>,
        a: f32,
        #[define(E)] _dtype: ElemType,
    ) {
        let i = ABSOLUTE_POS;
        if !out.is_in_bounds(i) {
            terminate!();
        }
        out.write(i, x.read(i) * E::cast_from(a) + y.read(i));
    }

    impl AxpyBackend for CubeBackend {
        fn axpy(x: FloatTensor<Self>, y: FloatTensor<Self>, a: f32) -> FloatTensor<Self> {
            // What the kernel asks of the device, held against what the device says it has
            // before anything launches: here, indexing tensors this large (WGSL has no 64-bit
            // indices). A kernel holding shared memory, assuming a plane width or wanting more
            // units per cube than `CubeDim::new` picks asks for those too (the gated delta rule's
            // `Requirements`); where the device falls short, the reference runs.
            let address = [x.required_address_type(), y.required_address_type()]
                .into_iter()
                .max()
                .unwrap_or_default();
            if !x.client.properties().supports_address(address) {
                return reference::<Self>(x, y, a);
            }
            let shape = x.meta.shape().clone();
            let out =
                empty_device_dtype(x.client.clone(), x.device.clone(), shape.clone(), x.dtype);
            let n = shape.num_elements();
            // Sized by the device: within its units per cube and cubes per launch.
            let cube_dim = CubeDim::new(&x.client, n);
            let cube_count = calculate_cube_count_elemwise(&x.client, n, cube_dim);
            let address = address.max(out.required_address_type());
            let dtype = burn::backend::cubecl::dtype_to_storage_type(x.dtype);
            probe::KERNEL_LAUNCHES.fetch_add(1, Ordering::SeqCst);
            unsafe {
                axpy_kernel::launch_unchecked(
                    &out.client,
                    cube_count,
                    cube_dim,
                    address,
                    x.into_linear_view_like(&out),
                    y.into_linear_view_like(&out),
                    out.clone().into_linear_view(),
                    a,
                    dtype,
                );
            }
            out
        }
    }
}

impl<B: Backend + AxpyBackend, C: CheckpointStrategy> AxpyBackend for Autodiff<B, C> {
    fn axpy(x: FloatTensor<Self>, y: FloatTensor<Self>, a: f32) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Axpy;

        impl<B: Backend> Backward<B, 2> for Axpy {
            // What the backward pass needs that the gradient does not carry: here only `a`.
            // Inputs it needs go here too, or through `prep.checkpoint(&tensor)`.
            type State = f32;

            fn backward(self, ops: Ops<f32, 2>, grads: &mut Gradients, _: &mut Checkpointer) {
                probe::BACKWARD_RUNS.fetch_add(1, Ordering::SeqCst);
                let a = ops.state;
                binary::<B, _, _>(
                    ops.parents,
                    ops.node,
                    grads,
                    |grad| B::float_mul_scalar(grad, a.into()),
                    |grad| grad,
                );
            }
        }

        let (x_node, y_node) = (x.node(), y.node());
        let (x, y) = (x.into_primitive(), y.into_primitive());
        match Axpy
            .prepare::<C>([x_node, y_node])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(a, B::axpy(x, y, a)),
            OpsKind::UnTracked(prep) => prep.finish(B::axpy(x, y, a)),
        }
    }
}

/// The operation as users call it.
pub fn axpy<const D: usize>(x: Tensor<D>, y: Tensor<D>, a: f32) -> Tensor<D> {
    Tensor::from_dispatch(<Dispatch as AxpyBackend>::axpy(
        x.into_dispatch(),
        y.into_dispatch(),
        a,
    ))
}

fn check(device: &Device) {
    let x = Tensor::<2>::from_floats([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]], device).require_grad();
    let y = Tensor::<2>::from_floats([[0.25, 1.0, -3.0], [2.0, 0.0, 1.5]], device).require_grad();
    let out = axpy(x.clone(), y.clone(), 3.0);
    out.to_data().assert_approx_eq::<f32>(
        &TensorData::from([[3.25f32, -5.0, 6.0], [3.5, 12.0, -1.5]]),
        Tolerance::absolute(1e-6),
    );
    // d/dx sum(out * w) = 3 w and d/dy = w, written out by the backward pass above.
    let w = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device);
    let runs = probe::backward_runs();
    let grads = (out * w.clone()).sum().backward();
    assert!(
        probe::backward_runs() > runs,
        "the written-out backward did not run"
    );
    x.grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<f32>(&(w.clone() * 3.0).to_data(), Tolerance::absolute(1e-6));
    y.grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<f32>(&w.to_data(), Tolerance::absolute(1e-6));
}

#[cfg(feature = "flex")]
#[test]
fn axpy_on_flex_with_its_own_backward() {
    check(&Device::flex().autodiff());
}

/// On the CubeCL device `DASH_OPS_DEVICE` names (`tests/common`), whichever runtime it is.
#[cfg(feature = "cubecl")]
#[test]
fn axpy_on_cubecl_runs_the_kernel_with_its_own_backward() {
    let device = common::cube_device();
    println!("{}", common::describe(&device));
    let launches = probe::kernel_launches();
    check(&device.autodiff());
    assert!(
        probe::kernel_launches() > launches,
        "the CubeCL kernel did not run"
    );
}
