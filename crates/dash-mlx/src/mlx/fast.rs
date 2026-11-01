//! MLX's fused operations this crate calls, and Metal kernels from source.

use super::array::Dtype;
use super::array::{Array, op};
use super::device::Stream;
use super::error::{self, Exception, Result, check};
use super::sys;
use super::transforms::VectorArray;
use std::ffi::CString;

/// `x / sqrt(mean(x^2) + eps)` over the last axis, then times `weight` if there is one.
pub fn rms_norm(x: impl AsRef<Array>, weight: Option<&Array>, eps: f32) -> Result<Array> {
    // SAFETY: a new empty handle, freed once, meaning no weight.
    let empty = unsafe { Array::from_ptr(sys::mlx_array_new()) };
    let weight = weight.unwrap_or(&empty);
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_fast_rms_norm(res, x.as_ref().as_ptr(), weight.as_ptr(), eps, s)
    })
}

/// Rotary embeddings of `x` `[.., T, D]` over its first `dims` features, at positions `offset..`.
pub fn rope(
    x: impl AsRef<Array>,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
) -> Result<Array> {
    let base = sys::mlx_optional_float {
        value: base,
        has_value: true,
    };
    // SAFETY: a new empty handle, freed once, meaning no frequencies of its own.
    let no_freqs = unsafe { Array::from_ptr(sys::mlx_array_new()) };
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_fast_rope(
            res,
            x.as_ref().as_ptr(),
            dims,
            traditional,
            base,
            scale,
            offset,
            no_freqs.as_ptr(),
            s,
        )
    })
}

/// [`rope`] with each row at its own position, `offset` `[B]`.
pub fn rope_dynamic(
    x: impl AsRef<Array>,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: impl AsRef<Array>,
) -> Result<Array> {
    let base = sys::mlx_optional_float {
        value: base,
        has_value: true,
    };
    // SAFETY: a new empty handle, freed once, meaning no frequencies of its own.
    let no_freqs = unsafe { Array::from_ptr(sys::mlx_array_new()) };
    // SAFETY: valid handles, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_fast_rope_dynamic(
            res,
            x.as_ref().as_ptr(),
            dims,
            traditional,
            base,
            scale,
            offset.as_ref().as_ptr(),
            no_freqs.as_ptr(),
            s,
        )
    })
}

/// The mask of [`scaled_dot_product_attention`].
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    /// Boolean: true where a query may see a key. Otherwise added to the scores.
    Array(&'a Array),
    /// Each query sees the keys up to its own position.
    Causal,
}

/// MLX's fused attention forward pass, `[B, heads, T, D]`.
pub fn scaled_dot_product_attention<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    mask: impl Into<Option<ScaledDotProductAttentionMask<'a>>>,
    sinks: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    // SAFETY: new empty handles, each freed once, meaning no mask and no sinks.
    let (no_mask, no_sinks) = unsafe {
        (
            Array::from_ptr(sys::mlx_array_new()),
            Array::from_ptr(sys::mlx_array_new()),
        )
    };
    let (mode, mask) = match mask.into() {
        Some(ScaledDotProductAttentionMask::Array(mask)) => (c"", mask),
        Some(ScaledDotProductAttentionMask::Causal) => (c"causal", &no_mask),
        None => (c"", &no_mask),
    };
    let sinks = sinks.into().unwrap_or(&no_sinks);
    // SAFETY: valid handles and a C string, for the length of the call.
    op(|res, s| unsafe {
        sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mode.as_ptr(),
            mask.as_ptr(),
            sinks.as_ptr(),
            false,
            s,
        )
    })
}

/// A template argument of a [`MetalKernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateArg {
    /// A dtype, as its Metal type.
    Dtype(Dtype),
    /// An integer.
    Int(i32),
    /// A boolean.
    Bool(bool),
}

/// How a [`MetalKernel`] is built.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetalKernelOptions {
    /// Metal source before the kernel's function.
    pub header: String,
    /// Whether the kernel's inputs are made row-contiguous first.
    pub ensure_row_contiguous: bool,
    /// Whether the outputs are atomic.
    pub atomic_outputs: bool,
}

/// One output of a [`MetalKernel`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalKernelOutput {
    /// Its shape.
    pub shape: Vec<i32>,
    /// Its dtype.
    pub dtype: Dtype,
}

/// How one call of a [`MetalKernel`] runs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetalKernelApplyOptions {
    /// The template arguments by name.
    pub template_args: Vec<(String, TemplateArg)>,
    /// A value every output starts from.
    pub init_value: Option<f32>,
    /// Whether MLX prints the kernel's source.
    pub verbose: bool,
}

/// A Metal kernel from its source. MLX compiles a source once per process.
pub struct MetalKernel {
    raw: sys::mlx_fast_metal_kernel,
}

fn c_string(value: &str) -> Result<CString> {
    CString::new(value).map_err(|error| Exception::custom(error.to_string()))
}

/// mlx-c's vector of strings, for the length of a call.
struct StringVector(sys::mlx_vector_string);

impl StringVector {
    fn new(values: &[&str]) -> Result<Self> {
        let owned = values
            .iter()
            .map(|value| c_string(value))
            .collect::<Result<Vec<_>>>()?;
        let mut pointers: Vec<_> = owned.iter().map(|value| value.as_ptr()).collect();
        // SAFETY: mlx-c copies each string, so `owned` need only outlive this call.
        let raw = unsafe { sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len()) };
        Ok(Self(raw))
    }
}

impl Drop for StringVector {
    fn drop(&mut self) {
        // SAFETY: this handle owns its reference, and is freed once.
        unsafe { sys::mlx_vector_string_free(self.0) };
    }
}

impl MetalKernel {
    /// A kernel named `name` from `source`. A source that does not compile is an error when the
    /// kernel first runs.
    pub fn new(
        name: &str,
        inputs: &[&str],
        outputs: &[&str],
        source: &str,
        options: MetalKernelOptions,
    ) -> Result<Self> {
        error::init();
        let (name, source) = (c_string(name)?, c_string(source)?);
        let header = c_string(&options.header)?;
        let (inputs, outputs) = (StringVector::new(inputs)?, StringVector::new(outputs)?);
        // SAFETY: valid handles and C strings, for the length of the call.
        let raw = unsafe {
            sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                inputs.0,
                outputs.0,
                source.as_ptr(),
                header.as_ptr(),
                options.ensure_row_contiguous,
                options.atomic_outputs,
            )
        };
        if raw.ctx.is_null() {
            check(1)?;
        }
        Ok(Self { raw })
    }

    /// Runs the kernel on `inputs` over a grid and threadgroup in threads, on the GPU, and returns
    /// its outputs.
    ///
    /// # Safety
    ///
    /// For these inputs and this launch, through all of its deferred execution, the kernel reads
    /// and writes only inside its buffers, writes no input, writes every output element unless
    /// `init_value` is set, writes only values valid for the output dtype (a 2 in a `Bool` output
    /// makes [`Array::as_slice`] give an invalid `bool`), and has no conflicting accesses.
    pub unsafe fn apply(
        &self,
        inputs: &[impl AsRef<Array>],
        outputs: &[MetalKernelOutput],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        options: MetalKernelApplyOptions,
    ) -> Result<Vec<Array>> {
        error::init();
        let config = KernelConfig::new()?;
        for output in outputs {
            // SAFETY: a valid handle and `shape`'s elements, for the length of the call.
            check(unsafe {
                sys::mlx_fast_metal_kernel_config_add_output_arg(
                    config.0,
                    output.shape.as_ptr(),
                    output.shape.len(),
                    output.dtype as sys::mlx_dtype,
                )
            })?;
        }
        // SAFETY: a valid handle and values, for the length of each call.
        unsafe {
            check(sys::mlx_fast_metal_kernel_config_set_grid(
                config.0, grid.0, grid.1, grid.2,
            ))?;
            check(sys::mlx_fast_metal_kernel_config_set_thread_group(
                config.0,
                thread_group.0,
                thread_group.1,
                thread_group.2,
            ))?;
        }
        for (name, arg) in &options.template_args {
            let name = c_string(name)?;
            // SAFETY: a valid handle and a C string, for the length of the call.
            check(unsafe {
                match *arg {
                    TemplateArg::Dtype(dtype) => {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            config.0,
                            name.as_ptr(),
                            dtype as sys::mlx_dtype,
                        )
                    }
                    TemplateArg::Int(value) => {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                            config.0,
                            name.as_ptr(),
                            value,
                        )
                    }
                    TemplateArg::Bool(value) => {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            config.0,
                            name.as_ptr(),
                            value,
                        )
                    }
                }
            })?;
        }
        if let Some(value) = options.init_value {
            // SAFETY: a valid handle and a value, for the length of the call.
            check(unsafe { sys::mlx_fast_metal_kernel_config_set_init_value(config.0, value) })?;
        }
        // SAFETY: a valid handle and a value, for the length of the call.
        check(unsafe { sys::mlx_fast_metal_kernel_config_set_verbose(config.0, options.verbose) })?;

        let inputs = VectorArray::from_arrays(inputs)?;
        let stream = Stream::gpu();
        let mut res = VectorArray::empty();
        // SAFETY: valid handles, for the length of the call; the rest is the caller's.
        check(unsafe {
            sys::mlx_fast_metal_kernel_apply(
                res.as_mut_ptr(),
                self.raw,
                inputs.as_ptr(),
                config.0,
                stream.as_ptr(),
            )
        })?;
        res.to_arrays()
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        // SAFETY: this handle owns its reference, and is freed once.
        unsafe { sys::mlx_fast_metal_kernel_free(self.raw) };
    }
}

impl std::fmt::Debug for MetalKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalKernel").finish_non_exhaustive()
    }
}

/// mlx-c's configuration of one kernel call.
struct KernelConfig(sys::mlx_fast_metal_kernel_config);

impl KernelConfig {
    fn new() -> Result<Self> {
        // SAFETY: makes a new handle.
        let raw = unsafe { sys::mlx_fast_metal_kernel_config_new() };
        if raw.ctx.is_null() {
            check(1)?;
        }
        Ok(Self(raw))
    }
}

impl Drop for KernelConfig {
    fn drop(&mut self) {
        // SAFETY: this handle owns its reference, and is freed once.
        unsafe { sys::mlx_fast_metal_kernel_config_free(self.0) };
    }
}
