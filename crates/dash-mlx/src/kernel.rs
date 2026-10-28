//! Metal kernels built once per thread and kept, and how a call launches one. MLX compiles a
//! kernel's library once per process, but the kernel object a call applies is built from its
//! source each time it is made, which on a decoding step, every layer every token, is host time
//! spent again and again.

use mlx_rs::error::Result;
use mlx_rs::fast::{
    MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg,
};
use mlx_rs::{Array, Dtype};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local! {
    static KERNELS: RefCell<HashMap<&'static str, Rc<MetalKernel>>> = RefCell::new(HashMap::new());
}

/// The kernel named `name`, built from the rest the first time this thread asks for it; its
/// inputs are made row-contiguous. A name stands for one source.
pub(crate) fn kernel(
    name: &'static str,
    inputs: &[&str],
    outputs: &[&str],
    source: &str,
    header: &str,
) -> Result<Rc<MetalKernel>> {
    if let Some(kernel) = KERNELS.with(|kernels| kernels.borrow().get(name).cloned()) {
        return Ok(kernel);
    }
    let options = MetalKernelOptions {
        header: header.into(),
        ensure_row_contiguous: true,
        atomic_outputs: false,
    };
    let kernel = Rc::new(MetalKernel::new(name, inputs, outputs, source, options)?);
    KERNELS.with(|kernels| kernels.borrow_mut().insert(name, kernel.clone()));
    Ok(kernel)
}

/// How one call of a kernel runs: the grid and threadgroup in threads, each output's shape and
/// dtype, the template arguments by name, and a value every output starts from, if any.
#[derive(Clone, Debug)]
pub(crate) struct Launch {
    pub(crate) grid: (i32, i32, i32),
    pub(crate) thread_group: (i32, i32, i32),
    pub(crate) outputs: Vec<(Vec<i32>, Dtype)>,
    pub(crate) template: Vec<(String, TemplateArg)>,
    pub(crate) init_value: Option<f32>,
}

/// Runs `kernel` on `inputs` as `launch` says, on the surrounding scope's stream (the GPU's
/// outside one), and returns its outputs.
///
/// # Safety
///
/// [`MetalKernel::apply`]'s: for these inputs and this launch, the kernel reads and writes only
/// inside its buffers, writes no input, writes every output element unless `init_value` is set,
/// and has no conflicting accesses.
pub(crate) unsafe fn launch(
    kernel: &MetalKernel,
    inputs: &[Array],
    launch: &Launch,
) -> Result<Vec<Array>> {
    let outputs: Vec<MetalKernelOutput> = launch
        .outputs
        .iter()
        .map(|(shape, dtype)| MetalKernelOutput {
            shape: shape.clone(),
            dtype: *dtype,
        })
        .collect();
    let options = MetalKernelApplyOptions {
        template_args: launch.template.clone(),
        init_value: launch.init_value,
        verbose: false,
    };
    // SAFETY: the caller's.
    unsafe { kernel.apply(inputs, &outputs, launch.grid, launch.thread_group, options) }
}
