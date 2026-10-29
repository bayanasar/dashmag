//! The conformance tests' adapter for dash-ops: `dash_conformance`'s test bodies run their calls
//! here, on a Burn device, Flex or a CubeCL device. Inputs are built and outputs read back as the
//! operations' own tests built and read them before the bodies were shared, so that every
//! comparison they print is the number it was.
//!
//! What the bodies cannot ask of every implementation, which of dash-ops' kernels a call ran, is
//! checked here around the calls: a decoding step of conv1d always (its kernel ran exactly where
//! `kernels_take` says it takes the step), the chunked gated delta rule's written-out backward pass
//! always, the log-probability's chunk kernel and written-out backward pass always, attention's
//! kernels, its written-out backward pass, how it summed the query gradient and that it copied no
//! input always, decoding's kernels always, and conv1d over sequences where a test asks for it
//! ([`Watch`]).

use super::{on_cuda, runtime_of};
use burn::tensor::{Bool, DType, Device, Int, Tensor, TensorData};
use dash_conformance::{
    Adapter, Attention, AttentionDecode, AttentionGrads, AttentionOut, ChosenLogprob,
    ChosenLogprobOut, ChunkGatedDelta, ChunkGatedDeltaGrads, ChunkGatedDeltaOut, Conv1d,
    Conv1dGrads, Conv1dOut, Conv1dStep, Conv1dStepOut, GatedDeltaStep, GatedDeltaStepGrads,
    GatedDeltaStepOut, HostTensor, Unsupported,
};
use dash_contract::attention::AttentionMask as HostMask;
use dash_ops::attention::{self, AttentionMask, attention_decode, attention_with_lse};
use dash_ops::conv1d::{self, causal_conv1d, causal_conv1d_update};
use dash_ops::gated_delta::chunk::{self, chunk_gated_delta_rule};
use dash_ops::gated_delta::recurrent::gated_delta_step;
use dash_ops::logprob::{self, LogprobOptions, VocabChunk, chosen_logprob};
use std::cell::Cell;

/// What the adapter checks of dash-ops' kernels around each call of conv1d over sequences.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Watch {
    /// Nothing: the call may be one the kernels turn down for its size (an empty prompt).
    #[default]
    Nothing,
    /// That the forward and backward kernels ran exactly where `kernels_take` says they take the
    /// call, and that the written-out backward pass ran.
    Launches,
    /// That, and which of them ran: the kernels over sequences for states or packed sequences,
    /// those of states for initial or final states, and where the kernels take the call no read of
    /// `cu_seqlens` to the host, where they do not that read for packed sequences.
    Counters,
}

/// A Burn device, as the conformance bodies drive it. Calls with gradients run on the device's
/// autodiff form, the others on the device as given.
pub struct Burn {
    device: Device,
    /// The dtype of `cu_seqlens`: `None` for the device's default int dtype.
    int: Option<DType>,
    watch: Watch,
    /// Of the conv1d calls watched, how many ran, and how many the kernels took.
    taken: Cell<[usize; 2]>,
}

impl Burn {
    /// The adapter on `device`, watching nothing, `cu_seqlens` in the device's int dtype.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            int: None,
            watch: Watch::Nothing,
            taken: Cell::new([0, 0]),
        }
    }

    /// The same, `cu_seqlens` in `int`.
    pub fn with_int(self, int: DType) -> Self {
        Self {
            int: Some(int),
            ..self
        }
    }

    /// The same, watching conv1d over sequences as `watch` says.
    pub fn watching(self, watch: Watch) -> Self {
        Self { watch, ..self }
    }

    /// Of the conv1d calls watched so far, how many ran and how many the kernels took.
    pub fn taken(&self) -> [usize; 2] {
        self.taken.get()
    }

    /// The contract's dtypes the device does not have, whose cases the conformance bodies leave
    /// out there; Flex and CUDA have every one.
    pub fn lacking(&self) -> Vec<dash_conformance::DType> {
        use dash_conformance::DType::{BF16, F16, F32};
        let lacking: Vec<_> = [F32, BF16, F16]
            .into_iter()
            .filter(|&dtype| !self.device.supports_dtype(burn_dtype(dtype)))
            .collect();
        if on_cuda(&self.device) || runtime_of(&self.device).is_none() {
            assert!(lacking.is_empty(), "{:?} has no {lacking:?}", self.device);
        }
        lacking
    }

    /// `offsets` as `cu_seqlens` on `device`.
    fn offsets(&self, offsets: &[i64], device: &Device) -> Tensor<1, Int> {
        let data = TensorData::new(offsets.to_vec(), [offsets.len()]);
        match self.int {
            None => Tensor::from_data(data, device),
            Some(dtype) => Tensor::from_data(data.convert_dtype(dtype), (device, dtype)),
        }
    }

    fn device_for(&self, grad: bool) -> Device {
        match grad {
            true => self.device.clone().autodiff(),
            false => self.device.clone(),
        }
    }
}

/// Burn's dtype for the contract's.
pub fn burn_dtype(dtype: dash_conformance::DType) -> DType {
    match dtype {
        dash_conformance::DType::F32 => DType::F32,
        dash_conformance::DType::BF16 => DType::BF16,
        dash_conformance::DType::F16 => DType::F16,
    }
}

/// The contract's dtype for Burn's.
pub fn contract_dtype(dtype: DType) -> dash_conformance::DType {
    match dtype {
        DType::F32 => dash_conformance::DType::F32,
        DType::BF16 => dash_conformance::DType::BF16,
        DType::F16 => dash_conformance::DType::F16,
        other => panic!("an output in {other:?}"),
    }
}

/// `host` on `device` in its dtype, its values converted on the host, as conv1d's and the step's
/// tests made their inputs.
pub fn tensor<const D: usize>(host: &HostTensor, device: &Device) -> Tensor<D> {
    let dtype = burn_dtype(host.dtype);
    let data = TensorData::new(host.values.clone(), host.shape.clone()).convert_dtype(dtype);
    Tensor::from_data(data, (device, dtype))
}

/// `host` made on `device` in f32, then cast there to its dtype, as the chunked rule's tests made
/// their inputs (a cast to the dtype a tensor has is no operation).
pub fn cast_tensor<const D: usize>(host: &HostTensor, device: &Device) -> Tensor<D> {
    let data = TensorData::new(host.values.clone(), host.shape.clone());
    Tensor::<D>::from_data(data, device).cast(burn_dtype(host.dtype))
}

/// `tensor` read to the host.
pub fn host<const D: usize>(tensor: Tensor<D>) -> HostTensor {
    let (dtype, shape) = (contract_dtype(tensor.dtype()), tensor.dims().to_vec());
    let values = tensor
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap();
    HostTensor::new(dtype, shape, values)
}

/// Whether conv1d's kernels take a call of width `k` in `dtype` for `x` `[B, T, C]` on `device`,
/// with `packed` that many sequences of `cu_seqlens` in that dtype; under CUDA they take every
/// width they hold.
pub fn conv1d_kernels_take(
    device: &Device,
    dtype: DType,
    k: usize,
    shape: [usize; 3],
    packed: Option<(usize, DType)>,
) -> bool {
    let takes = conv1d::kernels_take(device, dtype, k, shape, packed);
    if on_cuda(device) && k <= conv1d::MAX_KERNEL_WIDTH {
        assert!(takes, "CUDA turned the kernels away for K = {k}, {shape:?}");
    }
    takes
}

/// Asserts that `count` grew past `before` when `kernels`, and did not otherwise.
pub fn launched(kernels: bool, count: fn() -> usize, before: usize, what: &str) {
    assert_eq!(
        count() > before,
        kernels,
        "{what}: the kernel ran {} times, expected {}",
        count() - before,
        if kernels { "some" } else { "none" }
    );
}

/// The log-probability's chunk kernel launches in one pass over a vocabulary of `v` tokens for
/// `n` rows: one per chunk of the vocabulary in each block of rows, the blocks of equal size but
/// for the last.
pub fn logprob_pass_launches(n: usize, v: usize, options: LogprobOptions) -> usize {
    let block = n.div_ceil(n.div_ceil(options.row_chunk).max(1)).max(1);
    (0..n)
        .step_by(block)
        .map(|r0| v.div_ceil(options.vocab_chunk_for(block.min(n - r0))))
        .sum()
}

/// Asserts that since `before` the chunk kernel ran one pass's launches for `n` rows over `v`
/// tokens in `dtypes` if `kernels_take` says the kernels take them on `device`, and never
/// otherwise; under CUDA they must take them.
pub fn assert_logprob_launches(
    device: &Device,
    dtypes: [DType; 3],
    [n, v]: [usize; 2],
    options: LogprobOptions,
    before: usize,
) {
    let takes = logprob::kernels_take(device, dtypes, n, options);
    if on_cuda(device) {
        assert!(takes, "CUDA turned the chunk kernel away");
    }
    let want = match takes && n > 0 {
        true => logprob_pass_launches(n, v, options),
        false => 0,
    };
    let ran = logprob::probe::kernel_launches() - before;
    assert_eq!(
        ran, want,
        "the chunk kernel ran {ran} times, expected {want}"
    );
}

/// Whether attention's kernels take a forward and a backward pass over `dims` (`[B, Tq, Tk, Hq,
/// Hkv, D]`) in `dtype` on `device`. Under CUDA they take every bf16 and f16 call with a head
/// dimension of 64, 128 or 256, whatever the mask: cards of compute capability 8.0 and later have
/// all they ask.
pub fn attention_kernels_take(device: &Device, dtype: DType, dims: [usize; 6]) -> [bool; 2] {
    let taken = attention::kernels_take(device, dtype, dims);
    if on_cuda(device) {
        let takes = matches!(dtype, DType::BF16 | DType::F16) && matches!(dims[5], 64 | 128 | 256);
        assert_eq!(
            taken,
            [takes, takes],
            "CUDA turned the kernels away for {dims:?} {dtype:?}"
        );
    }
    taken
}

/// Asserts that `f` launched exactly `expected` of attention's kernels, and one more for each
/// backward pass whose products split and whose query gradient is the fixed-order one (its key and
/// value gradients are rounded from accumulators by a kernel the atomic way runs anyway): under
/// fusion the kernels run when their results are read, so `f` reads them. The passes in one `f`
/// take their query gradient the same way.
pub fn attention_launches<R>(expected: usize, f: impl FnOnce() -> R) -> R {
    use attention::probe;
    let before = probe::kernel_launches();
    let (splits, atomics) = (probe::split_runs(), probe::atomic_runs());
    let result = f();
    let launched = probe::kernel_launches() - before;
    let split = probe::split_runs() - splits;
    let expected = expected
        + if probe::atomic_runs() == atomics {
            split
        } else {
            0
        };
    assert_eq!(
        launched, expected,
        "the kernels made {launched} launches, expected {expected}"
    );
    result
}

/// What conv1d's counters showed of one call: whether the kernels ran (forward, backward, those
/// over sequences and those of states), and whether `cu_seqlens` was read to the host.
struct Counters([usize; 6]);

impl Counters {
    fn now() -> Self {
        use conv1d::probe;
        Self([
            probe::forward_launches(),
            probe::backward_launches(),
            probe::sequence_launches(),
            probe::state_launches(),
            probe::host_reads(),
            probe::backward_runs(),
        ])
    }

    /// Asserts what ran since `self` for a call through `Autodiff`: the written-out backward
    /// always; where `kernels`, the forward and backward kernels, those over sequences for
    /// states or packed sequences, those of states for final or initial states, and no read of
    /// `cu_seqlens` to the host; elsewhere no kernel, and for packed sequences their read.
    fn check(&self, kernels: bool, [initial, packed, finals]: [bool; 3], what: &str) {
        let now = Counters::now().0;
        let grew = |i: usize| now[i] > self.0[i];
        assert!(grew(5), "{what}: the written-out backward did not run");
        let sequences = initial || packed || finals;
        let states = initial || finals;
        if kernels {
            assert!(grew(0) && grew(1), "{what}: the kernels did not run");
            assert_eq!(grew(2), sequences, "{what}: the kernels over sequences");
            assert_eq!(grew(3), states, "{what}: the kernels of states");
            assert!(!grew(4), "{what}: cu_seqlens read to the host");
        } else {
            assert!(!grew(0) && !grew(1) && !grew(3), "{what}: a kernel ran");
            assert_eq!(grew(4), packed, "{what}: host reads");
        }
    }
}

impl Adapter for Burn {
    fn describe(&self) -> String {
        super::describe(&self.device)
    }

    fn has(&self, dtype: dash_conformance::DType) -> bool {
        self.device.supports_dtype(burn_dtype(dtype))
    }

    fn causal_conv1d(&self, call: &Conv1d) -> Result<Conv1dOut, Unsupported> {
        let device = self.device_for(call.grad.is_some());
        let leaf = |tensor: Tensor<3>| match call.grad {
            Some(_) => tensor.require_grad(),
            None => tensor,
        };
        let x = leaf(tensor::<3>(&call.x, &device));
        let [b, t, c] = x.dims();
        let weight = tensor::<2>(&call.weight, &device);
        let k = weight.dims()[1];
        let (weight, bias) = match call.grad {
            Some(_) => (
                weight.require_grad(),
                call.bias
                    .as_ref()
                    .map(|bias| tensor::<1>(bias, &device).require_grad()),
            ),
            None => (
                weight,
                call.bias.as_ref().map(|bias| tensor::<1>(bias, &device)),
            ),
        };
        let initial = call
            .initial_state
            .as_ref()
            .map(|state| leaf(tensor::<3>(state, &device)));
        let cu = call.cu_seqlens.as_ref().map(|cu| self.offsets(cu, &device));
        let what = format!("causal_conv1d [{b}, {t}, {c}] K = {k}");
        let packed = cu.as_ref().map(|cu| (cu.dims()[0] - 1, cu.dtype()));
        let kernels = (self.watch != Watch::Nothing)
            .then(|| conv1d_kernels_take(&device, x.dtype(), k, [b, t, c], packed));
        let (before, counters) = (
            [
                conv1d::probe::forward_launches(),
                conv1d::probe::backward_launches(),
                conv1d::probe::backward_runs(),
            ],
            Counters::now(),
        );
        let (out, final_state) = causal_conv1d(
            x.clone(),
            weight.clone(),
            bias.clone(),
            initial.clone(),
            cu,
            call.options,
        );
        // The loss, in f32, which the final states' part is in: d loss / d out is the output's
        // cotangent in the output's dtype exactly.
        let mut loss = None;
        if let Some(grad) = &call.grad {
            if let Some(grad_out) = &grad.out {
                let grad_out = grad_out.clone().with_dtype(contract_dtype(out.dtype()));
                let grad_out = tensor::<3>(&grad_out, &device);
                loss = Some((out.clone() * grad_out).sum().cast(DType::F32));
            }
            if let (Some(final_state), Some(d_final)) = (&final_state, &grad.final_state) {
                let term = (final_state.clone() * tensor::<3>(d_final, &device)).sum();
                loss = Some(match loss {
                    Some(loss) => loss + term,
                    None => term,
                });
            }
        }
        let final_state = final_state.map(host);
        let out = host(out);
        let grads = loss.map(|loss| {
            let grads = loss.backward();
            Conv1dGrads {
                x: host(x.grad(&grads).expect("x gradient")),
                weight: host(weight.grad(&grads).expect("weight gradient")),
                bias: bias.map(|bias| host(bias.grad(&grads).expect("bias gradient"))),
                initial_state: initial
                    .map(|state| host(state.grad(&grads).expect("initial state gradient"))),
            }
        });
        // Read after the values: under fusion the kernels run when they are read.
        if let Some(kernels) = kernels {
            let [taken, ran] = self.taken.get();
            self.taken.set([taken + usize::from(kernels), ran + 1]);
            let finals = call.options.output_final_state;
            match self.watch {
                Watch::Launches => {
                    use conv1d::probe::{backward_launches, backward_runs, forward_launches};
                    launched(kernels, forward_launches, before[0], &what);
                    assert!(
                        backward_runs() > before[2],
                        "{what}: the written-out backward pass did not run"
                    );
                    launched(kernels, backward_launches, before[1], &what);
                }
                Watch::Counters => counters.check(
                    kernels,
                    [call.initial_state.is_some(), packed.is_some(), finals],
                    &what,
                ),
                Watch::Nothing => {}
            }
        }
        Ok(Conv1dOut {
            out,
            final_state,
            grads,
        })
    }

    fn causal_conv1d_update(&self, call: &Conv1dStep) -> Result<Conv1dStepOut, Unsupported> {
        let device = &self.device;
        let x = tensor::<2>(&call.x, device);
        let [b, c] = x.dims();
        let weight = tensor::<2>(&call.weight, device);
        let k = weight.dims()[1];
        let kernels = conv1d_kernels_take(device, x.dtype(), k, [b, 1, c], None);
        let launches = conv1d::probe::update_launches();
        let (out, next) = causal_conv1d_update(
            x,
            weight,
            call.bias.as_ref().map(|bias| tensor::<1>(bias, device)),
            tensor::<3>(&call.state, device),
            call.options,
        );
        let out = host(out);
        launched(
            kernels,
            conv1d::probe::update_launches,
            launches,
            "causal_conv1d_update",
        );
        Ok(Conv1dStepOut {
            out,
            state: host(next),
        })
    }

    fn chunk_gated_delta_rule(
        &self,
        call: &ChunkGatedDelta,
    ) -> Result<ChunkGatedDeltaOut, Unsupported> {
        let grad = call.grad.is_some();
        let device = self.device_for(grad);
        let leaf4 = |host: &HostTensor| {
            let tensor = cast_tensor::<4>(host, &device);
            if grad { tensor.require_grad() } else { tensor }
        };
        let leaf3 = |host: &HostTensor| {
            let tensor = cast_tensor::<3>(host, &device);
            if grad { tensor.require_grad() } else { tensor }
        };
        let (q, k, v) = (leaf4(&call.q), leaf4(&call.k), leaf4(&call.v));
        let (g, beta) = (leaf3(&call.g), leaf3(&call.beta));
        let h0 = call.initial_state.as_ref().map(leaf4);
        let cu = call.cu_seqlens.as_ref().map(|cu| self.offsets(cu, &device));
        let runs = chunk::probe::backward_runs();
        let (out, state) = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            h0.clone(),
            cu,
            call.options,
        );
        let grads = call.grad.as_ref().map(|grad| {
            let d_out = cast_tensor::<4>(&grad.out, &device);
            let mut loss = (out.clone().cast(DType::F32) * d_out).sum();
            if let (Some(state), Some(d_state)) = (&state, &grad.final_state) {
                loss = loss + (state.clone() * cast_tensor::<4>(d_state, &device)).sum();
            }
            let grads = loss.backward();
            assert!(
                chunk::probe::backward_runs() > runs,
                "the written-out backward did not run"
            );
            grads
        });
        let out_host = host(out);
        let state = state.map(host);
        let grads = grads.map(|grads| ChunkGatedDeltaGrads {
            q: host(q.grad(&grads).unwrap()),
            k: host(k.grad(&grads).unwrap()),
            v: host(v.grad(&grads).unwrap()),
            g: host(g.grad(&grads).unwrap()),
            beta: host(beta.grad(&grads).unwrap()),
            initial_state: h0.map(|h0| host(h0.grad(&grads).unwrap())),
        });
        Ok(ChunkGatedDeltaOut {
            out: out_host,
            final_state: state,
            grads,
        })
    }

    fn gated_delta_step(&self, call: &GatedDeltaStep) -> Result<GatedDeltaStepOut, Unsupported> {
        let grad = call.grad.is_some();
        let device = self.device_for(grad);
        let leaf3 = |host: &HostTensor| {
            let tensor = tensor::<3>(host, &device);
            if grad { tensor.require_grad() } else { tensor }
        };
        let leaf2 = |host: &HostTensor| {
            let tensor = tensor::<2>(host, &device);
            if grad { tensor.require_grad() } else { tensor }
        };
        let (q, k, v) = (leaf3(&call.q), leaf3(&call.k), leaf3(&call.v));
        let (g, beta) = (leaf2(&call.g), leaf2(&call.beta));
        let state = tensor::<4>(&call.state, &device);
        let state = if grad { state.require_grad() } else { state };
        let cu = call.cu_seqlens.as_ref().map(|cu| self.offsets(cu, &device));
        let (out, new_state) = gated_delta_step(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            state.clone(),
            cu,
            call.options,
        );
        let grads = call.grad.as_ref().map(|grad| {
            let loss = (out.clone().cast(DType::F32) * tensor::<3>(&grad.out, &device)).sum()
                + (new_state.clone() * tensor::<4>(&grad.state, &device)).sum();
            loss.backward()
        });
        let (out, new_state) = (host(out), host(new_state));
        let grads = grads.map(|grads| GatedDeltaStepGrads {
            q: host(q.grad(&grads).unwrap()),
            k: host(k.grad(&grads).unwrap()),
            v: host(v.grad(&grads).unwrap()),
            g: host(g.grad(&grads).unwrap()),
            beta: host(beta.grad(&grads).unwrap()),
            state: host(state.grad(&grads).unwrap()),
        });
        Ok(GatedDeltaStepOut {
            out,
            state: new_state,
            grads,
        })
    }

    fn chosen_logprob(&self, call: &ChosenLogprob) -> Result<ChosenLogprobOut, Unsupported> {
        let device = self.device_for(call.grad.is_some());
        let hidden = cast_tensor::<2>(&call.hidden, &device);
        let hidden = match call.grad {
            Some(_) => hidden.require_grad(),
            None => hidden,
        };
        let weight = cast_tensor::<2>(&call.weight, &device);
        let [v, _] = weight.dims();
        let n = call.targets.len();
        let targets =
            Tensor::<1, Int>::from_data(TensorData::new(call.targets.clone(), [n]), &device);
        let dtypes = [hidden.dtype(), weight.dtype(), targets.dtype()];
        let (launches, runs) = (
            logprob::probe::kernel_launches(),
            logprob::probe::backward_runs(),
        );
        let out = chosen_logprob(hidden.clone(), weight, targets, &call.masked, call.options);
        let (logprob, entropy) = (host(out.logprob.clone()), host(out.entropy));
        let grad_hidden = call.grad.as_ref().map(|upstream| {
            let grads = (out.logprob * tensor::<1>(upstream, &device))
                .sum()
                .backward();
            host(hidden.grad(&grads).expect("hidden has a gradient"))
        });
        // Read after the values: under fusion the kernels run when their results are read.
        assert_logprob_launches(&device, dtypes, [n, v], call.options, launches);
        assert_eq!(
            logprob::probe::backward_runs() > runs,
            call.grad.is_some(),
            "the written-out backward ran where a gradient was asked for, and only there"
        );
        Ok(ChosenLogprobOut {
            logprob,
            entropy,
            grad_hidden,
        })
    }

    fn attention(&self, call: &Attention) -> Result<AttentionOut, Unsupported> {
        use attention::probe;
        let grad = call.grad.is_some();
        let device = self.device_for(grad);
        let leaf = |host: &HostTensor| {
            let tensor = cast_tensor::<4>(host, &device);
            if grad {
                tensor.detach().require_grad()
            } else {
                tensor
            }
        };
        let (q, k, v) = (leaf(&call.q), leaf(&call.k), leaf(&call.v));
        let ([b, tq, hq, d], [_, tk, hkv, _]) = (q.dims(), k.dims());
        let dtype = q.dtype();
        let mask = match &call.mask {
            HostMask::None => AttentionMask::None,
            HostMask::KeyPadding(padding) => {
                AttentionMask::KeyPadding(Tensor::<2, Bool>::from_data(
                    TensorData::new(padding.values.clone(), padding.shape.clone()),
                    &device,
                ))
            }
            HostMask::Packed {
                cu_seqlens_q,
                cu_seqlens_k,
            } => AttentionMask::Packed {
                cu_seqlens_q: self.offsets(cu_seqlens_q, &device),
                cu_seqlens_k: self.offsets(cu_seqlens_k, &device),
            },
        };
        let [forward, backward] = attention_kernels_take(&device, dtype, [b, tq, tk, hq, hkv, d]);
        let expected = usize::from(forward) + 3 * usize::from(backward && grad);
        let (runs, copies, atomics) = (
            probe::backward_runs(),
            probe::input_copies(),
            probe::atomic_runs(),
        );
        let out = attention_launches(expected, || {
            let (o, lse) = attention_with_lse(q.clone(), k.clone(), v.clone(), mask, call.options);
            let grads = call.grad.as_ref().map(|grad| {
                let grad = grad.clone().with_dtype(contract_dtype(dtype));
                (o.clone() * cast_tensor::<4>(&grad, &device))
                    .sum()
                    .backward()
            });
            AttentionOut {
                out: host(o),
                lse: host(lse),
                grads: grads.map(|grads| AttentionGrads {
                    q: host(q.grad(&grads).expect("query gradient")),
                    k: host(k.grad(&grads).expect("key gradient")),
                    v: host(v.grad(&grads).expect("value gradient")),
                }),
            }
        });
        assert_eq!(
            probe::backward_runs() > runs,
            grad,
            "the written-out backward ran where gradients were asked for, and only there"
        );
        assert_eq!(
            probe::atomic_runs() - atomics,
            usize::from(grad && backward && !call.options.deterministic),
            "the query gradient was not summed the way asked"
        );
        assert_eq!(
            probe::input_copies(),
            copies,
            "contiguous inputs were copied"
        );
        Ok(out)
    }

    fn attention_decode(&self, call: &AttentionDecode) -> Result<HostTensor, Unsupported> {
        let device = &self.device;
        let q = cast_tensor::<3>(&call.q, device);
        let (k, v) = (
            cast_tensor::<4>(&call.k, device),
            cast_tensor::<4>(&call.v, device),
        );
        let ([b, hq, d], [_, s, hkv, _]) = (q.dims(), k.dims());
        let dtype = q.dtype();
        let takes = attention::decode_kernels_take(device, dtype, [b, s, hq, hkv, d]);
        if on_cuda(device) {
            let kernels = matches!(dtype, DType::BF16 | DType::F16)
                && matches!(d, 64 | 128 | 256)
                && hq / hkv <= 16;
            assert_eq!(
                takes, kernels,
                "CUDA's decoding kernels for {dtype:?}, D = {d}"
            );
        }
        let lengths = call
            .lengths
            .as_ref()
            .map(|lengths| self.offsets(lengths, device));
        Ok(attention_launches(2 * usize::from(takes), || {
            host(attention_decode(q, k, v, lengths, call.options))
        }))
    }
}
