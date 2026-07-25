//! `dash-core` — runtime-agnostic whole-model inference traits for Dashmag.
//!
//! The abstraction seam is **whole-model inference**: load a model, tensors in,
//! tensors out. Backends of very different shapes — interpret-a-graph-file
//! (tract) vs. run-a-compiled-artifact (IREE) vs. drive-a-closed-appliance
//! (Edge TPU / TensorRT) — all sit behind one trait. No backend dependencies
//! live here.
//!
//! `Capability` is deliberately *not* a promise that backends are
//! interchangeable; it is how we surface that they are not, so the [`Registry`]
//! can negotiate rather than assume. [`Capability::admits`] is where that
//! negotiation actually happens, and it reads every field it advertises.

use std::borrow::Cow;
use std::fmt;

/// Result type for all fallible `dash-core` operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors at the runtime-agnostic call boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The backend cannot run this artifact (wrong `BackendKind`, missing capability).
    Unsupported(String),
    /// The model artifact failed to load.
    Load(String),
    /// Inference failed at run time.
    Run(String),
    /// Shape or dtype mismatch at the call boundary.
    Shape(String),
    /// A numeric result was unusable (empty output, NaN) — not a crash condition.
    Numeric(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::Load(s) => write!(f, "load failed: {s}"),
            Error::Run(s) => write!(f, "run failed: {s}"),
            Error::Shape(s) => write!(f, "shape/dtype error: {s}"),
            Error::Numeric(s) => write!(f, "numeric error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

/// Which backend produced and can consume an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Iree,
    Tract,
    EdgeTpu,
    TensorRt,
}

/// Accelerator class a backend targets. Dispatch ranks against this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccelClass {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
    VendorNpu,
}

/// Scalar element type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    I64,
    I32,
    I8,
    U8,
}

impl DType {
    /// Size of one element, in bytes.
    pub fn size(self) -> usize {
        match self {
            DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
            DType::I8 | DType::U8 => 1,
        }
    }
}

/// The set of dtypes a backend accepts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DTypeSet(pub Vec<DType>);

impl DTypeSet {
    pub fn of(ds: &[DType]) -> Self {
        DTypeSet(ds.to_vec())
    }
    pub fn contains(&self, d: DType) -> bool {
        self.0.contains(&d)
    }
}

/// What a backend can actually do. This is read by dispatch; it is how we make
/// the fact that backends are *not* interchangeable explicit and machine-checkable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capability {
    pub accel: AccelClass,
    pub dtypes: DTypeSet,
    pub dynamic_shapes: bool,
    pub quantized: bool,
    /// True only for a backend that can build without `std`.
    pub no_std: bool,
}

impl Capability {
    /// Can this backend honour what the artifact declares it needs?
    ///
    /// Every advertised field is consulted. An artifact that declares nothing
    /// (an empty [`ArtifactMeta`]) is admitted — absence of a declaration is not
    /// evidence of incompatibility, and inventing a rejection would be dishonest.
    /// Returns the reason on rejection so dispatch can explain itself.
    pub fn admits(&self, meta: &ArtifactMeta) -> std::result::Result<(), String> {
        if meta.quantized && !self.quantized {
            return Err("artifact is quantized, backend is not".to_string());
        }
        if meta.dynamic_shapes && !self.dynamic_shapes {
            return Err("artifact needs dynamic shapes, backend is static-shape only".to_string());
        }
        for (label, specs) in [("input", &meta.inputs), ("output", &meta.outputs)] {
            for (i, s) in specs.iter().enumerate() {
                if !self.dtypes.contains(s.dtype) {
                    return Err(format!(
                        "{label} {i} dtype {:?} not supported (backend accepts {:?})",
                        s.dtype, self.dtypes.0
                    ));
                }
            }
        }
        Ok(())
    }
}

/// IO specification for one tensor slot of an artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub dtype: DType,
    /// Static dimensions. An empty shape means scalar or "not declared";
    /// dynamic extents are signalled by [`ArtifactMeta::dynamic_shapes`].
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn new(dtype: DType, shape: &[usize]) -> Self {
        TensorSpec {
            dtype,
            shape: shape.to_vec(),
        }
    }
    /// Element count implied by the shape.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// IO metadata carried alongside an artifact's bytes.
///
/// This is *load-bearing*, not documentation: backends whose native runner
/// erases output shape (a raw buffer from a subprocess) recover it from
/// [`ArtifactMeta::output_shape`], so the same model returns the same shape
/// through every backend.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArtifactMeta {
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    /// The artifact's weights/activations are quantized.
    pub quantized: bool,
    /// The artifact requires dynamic-shape support at run time.
    pub dynamic_shapes: bool,
}

impl ArtifactMeta {
    /// Declared shape of output slot `i`, if any.
    pub fn output_shape(&self, i: usize) -> Option<&[usize]> {
        self.outputs.get(i).map(|s| s.shape.as_slice())
    }
    /// Declared dtype of output slot `i`, if any.
    pub fn output_dtype(&self, i: usize) -> Option<DType> {
        self.outputs.get(i).map(|s| s.dtype)
    }
    /// Convenience: declare f32 IO with the given shapes.
    pub fn f32_io(inputs: &[&[usize]], outputs: &[&[usize]]) -> Self {
        ArtifactMeta {
            inputs: inputs.iter().map(|s| TensorSpec::new(DType::F32, s)).collect(),
            outputs: outputs.iter().map(|s| TensorSpec::new(DType::F32, s)).collect(),
            quantized: false,
            dynamic_shapes: false,
        }
    }
}

/// An opaque, backend-tagged model artifact produced offline by the build-time
/// pipeline (`.vmfb`, ONNX/NNEF graph, edgetpu `.tflite`, TensorRT `.engine`).
#[derive(Clone)]
pub struct Artifact {
    pub backend: BackendKind,
    pub bytes: Vec<u8>,
    pub meta: ArtifactMeta,
}

impl fmt::Debug for Artifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Artifact")
            .field("backend", &self.backend)
            .field("bytes", &format_args!("{} bytes", self.bytes.len()))
            .field("meta", &self.meta)
            .finish()
    }
}

/// A host-resident tensor: dtype + shape + raw little-endian bytes.
///
/// Bytes-backed (rather than typed) so a single representation carries f32,
/// i64 indices and quantized int8 alike; typed views are provided for the
/// common cases.
#[derive(Debug, Clone, PartialEq)]
pub struct HostTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl HostTensor {
    /// Build an f32 tensor from a slice.
    pub fn from_f32(shape: Vec<usize>, v: &[f32]) -> Self {
        let mut data = vec![0u8; v.len() * 4];
        for (dst, x) in data.chunks_exact_mut(4).zip(v) {
            dst.copy_from_slice(&x.to_le_bytes());
        }
        HostTensor {
            dtype: DType::F32,
            shape,
            data,
        }
    }

    /// Build an i64 tensor from a slice (detector index outputs).
    pub fn from_i64(shape: Vec<usize>, v: &[i64]) -> Self {
        let mut data = vec![0u8; v.len() * 8];
        for (dst, x) in data.chunks_exact_mut(8).zip(v) {
            dst.copy_from_slice(&x.to_le_bytes());
        }
        HostTensor {
            dtype: DType::I64,
            shape,
            data,
        }
    }

    /// Build an f32 tensor, adopting `declared` when its element count matches.
    ///
    /// This is how a backend whose runner returns a shape-erased buffer restores
    /// the model's real output shape, so callers never have to branch on which
    /// backend ran.
    pub fn f32_shaped(v: &[f32], declared: Option<&[usize]>) -> Self {
        match declared {
            Some(s) if !s.is_empty() && s.iter().product::<usize>() == v.len() => {
                HostTensor::from_f32(s.to_vec(), v)
            }
            _ => HostTensor::from_f32(vec![v.len()], v),
        }
    }

    /// Number of elements implied by the shape.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Decode the buffer as f32. Errors if the dtype is not f32 or the byte
    /// length is not a multiple of 4.
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(Error::Shape(format!("as_f32 called on {:?}", self.dtype)));
        }
        if self.data.len() % 4 != 0 {
            return Err(Error::Shape("byte length not a multiple of 4".into()));
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// A backend-owned handle to a tensor resident in accelerator memory.
///
/// The point of the variant is zero-copy residency across calls (no host
/// round-trip = no perf leak). It is a stub through Phase 0 (CPU-only); the
/// shape must not ossify around host-only tensors before Phase 2 lands it.
pub trait DeviceTensor: Send + Sync {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;
    /// Download to host memory.
    fn to_host(&self) -> Result<HostTensor>;
}

/// A tensor living on the host, or on opaque device memory.
pub enum Tensor {
    Host(HostTensor),
    Device(Box<dyn DeviceTensor>),
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::Host(h) => &h.shape,
            Tensor::Device(d) => d.shape(),
        }
    }
    pub fn dtype(&self) -> DType {
        match self {
            Tensor::Host(h) => h.dtype,
            Tensor::Device(d) => d.dtype(),
        }
    }
    /// Host view without copying when the tensor is already host-resident.
    /// Only a device tensor pays for a download.
    pub fn host(&self) -> Result<Cow<'_, HostTensor>> {
        match self {
            Tensor::Host(h) => Ok(Cow::Borrowed(h)),
            Tensor::Device(d) => Ok(Cow::Owned(d.to_host()?)),
        }
    }
    /// Owned host copy (clones a host tensor, downloads a device one).
    pub fn to_host(&self) -> Result<HostTensor> {
        match self {
            Tensor::Host(h) => Ok(h.clone()),
            Tensor::Device(d) => d.to_host(),
        }
    }
    /// Consume into a host tensor.
    pub fn into_host(self) -> Result<HostTensor> {
        match self {
            Tensor::Host(h) => Ok(h),
            Tensor::Device(d) => d.to_host(),
        }
    }
}

impl From<HostTensor> for Tensor {
    fn from(h: HostTensor) -> Self {
        Tensor::Host(h)
    }
}

/// Index and value of the largest element.
///
/// Errors rather than panics on an empty slice or NaN — a model emitting NaN
/// (a mis-quantized int8 graph, say) is a result to report, not a crash.
pub fn argmax(v: &[f32]) -> Result<(usize, f32)> {
    if v.is_empty() {
        return Err(Error::Numeric("empty output, no argmax".into()));
    }
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x.is_nan() {
            return Err(Error::Numeric(format!("output contains NaN at index {i}")));
        }
        if i == 0 || x > best_v {
            best_v = x;
            best = i;
        }
    }
    Ok((best, best_v))
}

/// A loaded, ready-to-run model. Cheap to call repeatedly.
pub trait Session: Send + Sync {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>>;
}

/// A backend: turns an [`Artifact`] into a [`Session`], and honestly reports
/// what it can and cannot run.
pub trait Runtime: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn capability(&self) -> &Capability;
    /// Honest gatekeeper: right backend kind *and* capabilities the artifact needs.
    fn can_run(&self, art: &Artifact) -> bool {
        art.backend == self.kind() && self.capability().admits(&art.meta).is_ok()
    }
    fn load(&self, art: &Artifact) -> Result<Box<dyn Session>>;
}

/// Description of the deployment target dispatch ranks backends against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetProfile {
    pub accel: AccelClass,
    /// True for an MCU / seL4 / bare-metal build that can only pull a `no_std` backend.
    pub no_std: bool,
}

/// The dispatch brain: given an artifact and a target, pick the best admissible
/// backend. Capability negotiation with graceful degrade — never an assumption
/// that any registered backend will do.
#[derive(Default)]
pub struct Registry {
    backends: Vec<Box<dyn Runtime>>,
}

impl Registry {
    pub fn new() -> Self {
        Registry {
            backends: Vec::new(),
        }
    }

    pub fn register(&mut self, backend: Box<dyn Runtime>) -> &mut Self {
        self.backends.push(backend);
        self
    }

    pub fn len(&self) -> usize {
        self.backends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }

    /// Rank by `can_run(artifact)` and target fit; prefer an accelerator-class
    /// match; disqualify a non-`no_std` backend for a `no_std` target. Returns
    /// the highest-scoring admissible backend, or `None` if nothing admits.
    pub fn select(&self, art: &Artifact, target: &TargetProfile) -> Option<&dyn Runtime> {
        let mut best: Option<(&dyn Runtime, i32)> = None;
        for b in &self.backends {
            let rt = b.as_ref();
            if !rt.can_run(art) {
                continue;
            }
            let cap = rt.capability();
            if target.no_std && !cap.no_std {
                continue;
            }
            let mut score = 1;
            if cap.accel == target.accel {
                score += 10;
            }
            match best {
                Some((_, bs)) if bs >= score => {}
                _ => best = Some((rt, score)),
            }
        }
        best.map(|(rt, _)| rt)
    }

    /// Why each registered backend was refused, for diagnostics when
    /// [`Registry::select`] returns `None`. Dispatch that cannot explain itself
    /// is indistinguishable from dispatch that is broken.
    pub fn explain(&self, art: &Artifact, target: &TargetProfile) -> Vec<(BackendKind, String)> {
        self.backends
            .iter()
            .map(|b| {
                let rt = b.as_ref();
                let cap = rt.capability();
                let why = if rt.kind() != art.backend {
                    format!("wants a {:?} artifact, got {:?}", rt.kind(), art.backend)
                } else if let Err(e) = cap.admits(&art.meta) {
                    e
                } else if target.no_std && !cap.no_std {
                    "target is no_std, backend requires std".to_string()
                } else {
                    "admissible".to_string()
                };
                (rt.kind(), why)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial backend that echoes its inputs, for exercising the trait shape
    /// and the registry without pulling a real runtime.
    struct EchoRuntime {
        cap: Capability,
        kind: BackendKind,
    }
    struct EchoSession;

    impl Runtime for EchoRuntime {
        fn kind(&self) -> BackendKind {
            self.kind
        }
        fn capability(&self) -> &Capability {
            &self.cap
        }
        fn load(&self, _art: &Artifact) -> Result<Box<dyn Session>> {
            Ok(Box::new(EchoSession))
        }
    }
    impl Session for EchoSession {
        fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
            inputs
                .iter()
                .map(|t| t.to_host().map(Tensor::Host))
                .collect()
        }
    }

    fn cap(accel: AccelClass, no_std: bool) -> Capability {
        Capability {
            accel,
            dtypes: DTypeSet::of(&[DType::F32]),
            dynamic_shapes: false,
            quantized: false,
            no_std,
        }
    }

    fn art(backend: BackendKind) -> Artifact {
        Artifact {
            backend,
            bytes: Vec::new(),
            meta: ArtifactMeta::default(),
        }
    }

    #[test]
    fn host_tensor_f32_roundtrip() {
        let t = HostTensor::from_f32(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.as_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.data.len(), 16);
    }

    #[test]
    fn as_f32_rejects_wrong_dtype() {
        let t = HostTensor {
            dtype: DType::U8,
            shape: vec![3],
            data: vec![1, 2, 3],
        };
        assert!(matches!(t.as_f32(), Err(Error::Shape(_))));
    }

    #[test]
    fn registry_selects_by_backend_and_accel() {
        let mut reg = Registry::new();
        reg.register(Box::new(EchoRuntime {
            cap: cap(AccelClass::Cpu, true),
            kind: BackendKind::Tract,
        }));
        reg.register(Box::new(EchoRuntime {
            cap: cap(AccelClass::Cuda, false),
            kind: BackendKind::Iree,
        }));

        let sel = reg
            .select(
                &art(BackendKind::Tract),
                &TargetProfile {
                    accel: AccelClass::Cpu,
                    no_std: false,
                },
            )
            .expect("a backend should admit");
        assert_eq!(sel.kind(), BackendKind::Tract);

        assert!(reg
            .select(
                &art(BackendKind::Iree),
                &TargetProfile {
                    accel: AccelClass::Cpu,
                    no_std: true,
                },
            )
            .is_none());

        assert!(reg
            .select(
                &art(BackendKind::EdgeTpu),
                &TargetProfile {
                    accel: AccelClass::VendorNpu,
                    no_std: false,
                },
            )
            .is_none());
    }

    #[test]
    fn end_to_end_through_the_trait() {
        let rt = EchoRuntime {
            cap: cap(AccelClass::Cpu, true),
            kind: BackendKind::Tract,
        };
        let sess = rt.load(&art(BackendKind::Tract)).unwrap();
        let out = sess
            .run(&[HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).into()])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].to_host().unwrap().as_f32().unwrap(),
            vec![1.0, 2.0, 3.0]
        );
    }

    // --- capability negotiation now reads every advertised field ---

    #[test]
    fn capability_rejects_unsupported_dtype() {
        let c = cap(AccelClass::Cpu, false); // f32 only
        let meta = ArtifactMeta {
            inputs: vec![TensorSpec::new(DType::U8, &[1, 3, 224, 224])],
            ..Default::default()
        };
        let err = c.admits(&meta).unwrap_err();
        assert!(err.contains("U8"), "reason should name the dtype: {err}");
    }

    #[test]
    fn capability_rejects_quantized_and_dynamic() {
        let c = cap(AccelClass::Cpu, false);
        assert!(c
            .admits(&ArtifactMeta {
                quantized: true,
                ..Default::default()
            })
            .is_err());
        assert!(c
            .admits(&ArtifactMeta {
                dynamic_shapes: true,
                ..Default::default()
            })
            .is_err());
    }

    #[test]
    fn empty_meta_is_admitted_not_guessed() {
        // No declaration is not evidence of incompatibility.
        assert!(cap(AccelClass::Cpu, false)
            .admits(&ArtifactMeta::default())
            .is_ok());
    }

    #[test]
    fn dispatch_refuses_artifact_needing_missing_capability() {
        let mut reg = Registry::new();
        reg.register(Box::new(EchoRuntime {
            cap: cap(AccelClass::Cpu, false), // not quantized-capable
            kind: BackendKind::Tract,
        }));
        let mut a = art(BackendKind::Tract);
        a.meta.quantized = true;
        let target = TargetProfile {
            accel: AccelClass::Cpu,
            no_std: false,
        };
        assert!(reg.select(&a, &target).is_none());
        let why = reg.explain(&a, &target);
        assert!(why[0].1.contains("quantized"), "explain: {:?}", why);
    }

    // --- output shape uniformity: the whole point of ArtifactMeta ---

    #[test]
    fn f32_shaped_adopts_declared_shape() {
        let flat: Vec<f32> = vec![0.0; 1000];
        let t = HostTensor::f32_shaped(&flat, Some(&[1, 1000]));
        assert_eq!(t.shape, vec![1, 1000]);
    }

    #[test]
    fn f32_shaped_falls_back_when_count_mismatches() {
        let flat: Vec<f32> = vec![0.0; 999];
        let t = HostTensor::f32_shaped(&flat, Some(&[1, 1000]));
        assert_eq!(t.shape, vec![999], "must not adopt a shape that does not fit");
    }

    // --- argmax: report NaN, never panic ---

    #[test]
    fn argmax_finds_max() {
        assert_eq!(argmax(&[1.0, 9.0, 3.0]).unwrap(), (1, 9.0));
    }

    #[test]
    fn argmax_reports_nan_instead_of_panicking() {
        assert!(matches!(
            argmax(&[1.0, f32::NAN, 3.0]),
            Err(Error::Numeric(_))
        ));
    }

    #[test]
    fn argmax_reports_empty() {
        assert!(matches!(argmax(&[]), Err(Error::Numeric(_))));
    }

    #[test]
    fn tensor_host_borrows_without_copying() {
        let t: Tensor = HostTensor::from_f32(vec![2], &[1.0, 2.0]).into();
        assert!(matches!(t.host().unwrap(), Cow::Borrowed(_)));
    }
}
