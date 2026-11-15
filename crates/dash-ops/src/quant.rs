//! A linear layer over weights quantized to 4 bits by group, read packed as GPTQ stores them:
//! `y = x W (+ bias)`, for inference.
//!
//! [`dash_contract::quant`] defines the operation: the packed layout, the zero points, the
//! definition in f32 and the dtypes. This module implements it on Burn.
//!
//! [`QuantizedWeight::from_gptq`] takes a checkpoint's four tensors as stored and checks them
//! once, when the weight is built: the shapes against each other, every stored zero against the
//! format's, and `g_idx` against groups in order. [`quantized_linear`] then reads the packed
//! weight on every call, through [`QuantBackend`]: its default body is the reference (`W` unpacked
//! and dequantized to f32 in Burn's integer and float operations, then multiplied in f32), and a
//! CubeCL device runs kernels where they take the call. It is a forward pass only: on `Autodiff`
//! an input that carries a gradient is refused.

use crate::lowrank::MaybeTensor;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Autodiff, AutodiffBackend, Backend, Dispatch, backend_extension};
use burn::tensor::{DType, Int, Tensor, TensorData};

#[cfg(feature = "flex")]
use burn::backend::Flex;

#[cfg(feature = "cubecl")]
mod cube;
mod reference;
#[cfg(feature = "cubecl")]
mod wmma;

pub use dash_contract::quant::{QuantFormat, QuantOptions};

/// Counters a test reads to tell which code ran. Not part of the API.
#[doc(hidden)]
pub mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(super) static GEMV_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static DEQUANT_LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static MATRIX_LAUNCHES: AtomicUsize = AtomicUsize::new(0);

    /// Launches of the decoding kernel (none where the reference runs).
    pub fn kernel_launches() -> usize {
        GEMV_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of the prefill path's dequantizing kernel.
    pub fn dequant_launches() -> usize {
        DEQUANT_LAUNCHES.load(Ordering::SeqCst)
    }

    /// Launches of the prefill kernel on RDNA4's matrix products.
    pub fn matrix_launches() -> usize {
        MATRIX_LAUNCHES.load(Ordering::SeqCst)
    }
}

/// Whether the decoding kernels, rather than the reference, take [`quantized_linear`] on `device`
/// for `x` `[rows, in]` in `dtype` over a weight of `out` outputs in groups of `group` inputs:
/// up to 16 rows, on a device with what the kernels ask, where the kernel on matrix products
/// ([`matrix_takes`]) does not (past 8 rows on such a device). `false` on any device not CubeCL's.
/// For tests, as the [`probe`] counters are.
#[doc(hidden)]
pub fn kernels_take(
    device: &burn::tensor::Device,
    dtype: DType,
    dims: [usize; 3],
    group: usize,
) -> bool {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return cube::kernels_take(&client, dtype, dims, group);
    }
    let _ = (device, dtype, dims, group);
    false
}

/// Whether the kernel on RDNA4's 16 x 16 x 16 matrix products takes [`quantized_linear`] on
/// `device` for these sizes: more than 8 rows of bf16 or f16, inputs a multiple of 32, groups a
/// multiple of 8 that divide the inputs, on a device with those products in RDNA4's layout. For
/// tests.
#[doc(hidden)]
pub fn matrix_takes(
    device: &burn::tensor::Device,
    dtype: DType,
    dims: [usize; 3],
    group: usize,
) -> bool {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return cube::matrix_takes(&client, dtype, dims, group);
    }
    let _ = (device, dtype, dims, group);
    false
}

/// The runs [`matrix_takes`]' kernel splits its inputs into on `device` for these sizes (1: not
/// split), where it takes them; 0 elsewhere. For tests.
#[doc(hidden)]
pub fn matrix_runs(
    device: &burn::tensor::Device,
    dtype: DType,
    dims: [usize; 3],
    group: usize,
) -> usize {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return cube::matrix_runs(&client, dtype, dims, group);
    }
    let _ = (device, dtype, dims, group);
    0
}

/// Whether the prefill path (the weight dequantized by a kernel, then Burn's matmul) takes
/// [`quantized_linear`] on `device` for these sizes: more than 16 rows, on a CubeCL device with
/// what the kernel asks, where the kernel on matrix products ([`matrix_takes`]) does not. For
/// tests.
#[doc(hidden)]
pub fn prefill_takes(
    device: &burn::tensor::Device,
    dtype: DType,
    dims: [usize; 3],
    group: usize,
) -> bool {
    #[cfg(feature = "cubecl")]
    if let Some(client) = crate::cube::client(device.as_dispatch()) {
        return cube::prefill_takes(&client, dtype, dims, group);
    }
    let _ = (device, dtype, dims, group);
    false
}

#[cfg(feature = "fusion")]
mod fusion_meta {
    use crate::lowrank::MaybeTensorMetadata;
    use burn::backend::fusion::custom::TensorSpec;
    use burn::tensor::Shape;

    pub(super) fn forward(
        x: &TensorSpec,
        qweight: &TensorSpec,
        _scales: &TensorSpec,
        _bias: &MaybeTensorMetadata,
        _group: &usize,
    ) -> TensorSpec {
        TensorSpec::new(Shape::new([x.shape[0], qweight.shape[1]]), x.dtype)
    }
}

/// The quantized linear as a backend extension, on rows: call it through [`quantized_linear`].
/// The default body is the reference; [`dash_contract::quant`] gives the definition and dtypes.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(feature = "cubecl"),
    Fusion: cfg(feature = "fusion"),
    Autodiff
)]
pub trait QuantBackend: Backend {
    /// `y` `[N, out]` in `x`'s dtype for `x` `[N, in]`, `qweight` `[in / 8, out]` (`i32`),
    /// `scales` `[in / group, out]` (f16) and the bias `[out]`.
    #[fusion(meta = fusion_meta::forward)]
    fn quantized_linear(
        x: FloatTensor<Self>,
        qweight: IntTensor<Self>,
        scales: FloatTensor<Self>,
        #[extension_type] bias: MaybeTensor<Self>,
        group: usize,
    ) -> FloatTensor<Self> {
        reference::forward::<Self>(x, qweight, scales, bias, group)
    }
}

#[cfg(feature = "flex")]
impl QuantBackend for Flex {}

impl<B: Backend + QuantBackend, C: CheckpointStrategy> QuantBackend for Autodiff<B, C> {
    fn quantized_linear(
        x: FloatTensor<Self>,
        qweight: IntTensor<Self>,
        scales: FloatTensor<Self>,
        bias: MaybeTensor<Self>,
        group: usize,
    ) -> FloatTensor<Self> {
        assert!(
            !x.is_tracked(),
            "quantized_linear is a forward pass only: x carries a gradient (detach it)"
        );
        let bias = match bias {
            MaybeTensor::None => MaybeTensor::None,
            MaybeTensor::Some(bias) => MaybeTensor::Some(bias.into_primitive()),
        };
        <Self as AutodiffBackend>::from_inner(B::quantized_linear(
            x.into_primitive(),
            qweight,
            scales.into_primitive(),
            bias,
            group,
        ))
    }
}

/// A weight `[in, out]` packed as GPTQ stores it, checked: `qweight` `[in / 8, out]` (`i32`) and
/// `scales` `[in / group, out]` (f16), on one device. The zero points and `g_idx` were checked
/// when it was built and are not kept.
#[derive(Clone, Debug)]
pub struct QuantizedWeight {
    qweight: Tensor<2, Int>,
    scales: Tensor<2>,
    inp: usize,
    out: usize,
    options: QuantOptions,
}

impl QuantizedWeight {
    /// The weight from a checkpoint's tensors as stored (`dash_contract::quant`'s layout):
    /// `qweight` `[in / 8, out]` and `qzeros` `[in / group, out / 8]`, `i32`; `scales`
    /// `[in / group, out]`, f16; `g_idx` `[in]`, `i32`, or none. Refused, with what does not
    /// hold: shapes that disagree, `in` not a multiple of the group or `out` of 8, `scales` not
    /// f16, a stored zero other than the format's ([`QuantOptions::SYMMETRIC_ZERO`]), or a
    /// `g_idx` other than `i / group`. Reads `qzeros` and `g_idx` back to the host once.
    pub fn from_gptq(
        qweight: Tensor<2, Int>,
        qzeros: Tensor<2, Int>,
        scales: Tensor<2>,
        g_idx: Option<Tensor<1, Int>>,
        options: QuantOptions,
    ) -> Result<Self, String> {
        let op = "QuantizedWeight::from_gptq";
        let group = options.group;
        let [packed_in, out] = qweight.dims();
        let inp = packed_in * 8;
        if group == 0 || inp % group != 0 || out % 8 != 0 {
            return Err(format!(
                "{op}: in = {inp} must be a multiple of the group ({group}), and out = {out} of 8"
            ));
        }
        let groups = inp / group;
        if scales.dims() != [groups, out] || qzeros.dims() != [groups, out / 8] {
            return Err(format!(
                "{op}: qweight [{packed_in}, {out}] wants scales [{groups}, {out}] and qzeros \
                 [{groups}, {}], got {:?} and {:?}",
                out / 8,
                scales.dims(),
                qzeros.dims()
            ));
        }
        if scales.dtype() != DType::F16 {
            return Err(format!("{op}: scales are f16, got {:?}", scales.dtype()));
        }
        let QuantFormat::Symmetric = options.format;
        let zero = QuantOptions::SYMMETRIC_ZERO as i64;
        let words = host_ints(qzeros);
        if let Some(i) = words
            .iter()
            .position(|&w| (0..8).any(|j| ((w >> (4 * j)) & 15) != zero))
        {
            return Err(format!(
                "{op}: a symmetric weight's stored zeros are all {zero}, but word {i} of qzeros \
                 is {:#010x} (GPTQ v1 stores zero - 1, 0x77777777)",
                words[i] as u32
            ));
        }
        if let Some(g_idx) = g_idx {
            if g_idx.dims() != [inp] {
                return Err(format!("{op}: g_idx is [{inp}], got {:?}", g_idx.dims()));
            }
            let g = host_ints(g_idx);
            if let Some(i) = (0..inp).find(|&i| g[i] != (i / group) as i64) {
                return Err(format!(
                    "{op}: g_idx[{i}] is {}, where groups in order give {} (reordered groups, \
                     GPTQ's desc_act, are not taken)",
                    g[i],
                    i / group
                ));
            }
        }
        Ok(Self {
            qweight,
            scales,
            inp,
            out,
            options,
        })
    }

    /// `in`, the inputs.
    pub fn in_features(&self) -> usize {
        self.inp
    }

    /// `out`, the outputs.
    pub fn out_features(&self) -> usize {
        self.out
    }

    /// The options it was built with.
    pub fn options(&self) -> QuantOptions {
        self.options
    }

    /// `W` `[in, out]` in f32, exactly: `(q - zero) · scale`.
    pub fn dequantize(&self) -> Tensor<2> {
        Tensor::from_dispatch(reference::dequantize::<Dispatch>(
            self.qweight.clone().into_dispatch(),
            self.scales.clone().into_dispatch(),
            self.options.group,
        ))
    }
}

fn host_ints<const D: usize>(t: Tensor<D, Int>) -> Vec<i64> {
    let data: TensorData = t.into_data();
    data.iter::<i64>().collect()
}

/// `y = x W (+ bias)` over `weight`, for `x` `[.., in]` in f32, bf16 or f16, the output in `x`'s
/// dtype: the definition is the product in f32, the bias added and the sum rounded once
/// (`dash_contract::quant`). The reference and the decoding kernels compute it so; the prefill
/// path (more than 16 rows on a CubeCL device) rounds `W`, and the product before the bias, to
/// `x`'s dtype, and the kernel on RDNA4's matrix products (in its place on such a device) rounds
/// `W` and adds the bias before its one rounding, within the bounds `tests/quant.rs` records. A
/// forward pass only.
pub fn quantized_linear<const D: usize>(
    x: Tensor<D>,
    weight: &QuantizedWeight,
    bias: Option<Tensor<1>>,
) -> Tensor<D> {
    let op = "quantized_linear";
    let dims = x.dims();
    assert!(
        dims[D - 1] == weight.inp,
        "{op}: x [.., {}] and a weight of {} inputs disagree",
        dims[D - 1],
        weight.inp
    );
    assert!(
        matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16),
        "{op}: x is f32, bf16 or f16, got {:?}",
        x.dtype()
    );
    if let Some(bias) = &bias {
        assert!(
            bias.dims() == [weight.out],
            "{op}: bias is [{}], got {:?}",
            weight.out,
            bias.dims()
        );
    }
    let rows: usize = dims[..D - 1].iter().product();
    let y = <Dispatch as QuantBackend>::quantized_linear(
        x.reshape([rows, weight.inp]).into_dispatch(),
        weight.qweight.clone().into_dispatch(),
        weight.scales.clone().into_dispatch(),
        MaybeTensor::from(bias.map(|bias| bias.into_dispatch())),
        weight.options.group,
    );
    let mut out_dims = dims;
    out_dims[D - 1] = weight.out;
    Tensor::<2>::from_dispatch(y).reshape(out_dims)
}
