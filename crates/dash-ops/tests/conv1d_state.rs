//! The causal depthwise conv1d from given states and over packed variable-length sequences:
//! `dash_conformance::conv1d_state`'s tests (against torch, a sequence split in two, packed
//! sequences against one call each, empty sequences, either output alone in the loss) through the
//! Burn adapter, on Flex and on a CubeCL device, any runtime (`DASH_OPS_DEVICE`). For every case
//! against torch the adapter asserts which path ran: the kernels where `kernels_take` says they
//! take the call (always under CUDA), and then without reading `cu_seqlens` to the host.
//!
//! What is dash-ops' own stays here: the split sequence through one autodiff graph, its parts
//! views of one tracked input and the first part's tracked final state the second's initial
//! state; and the kernels against Flex's reference at a size with many runs and boundaries.
//!
//!   cargo test --release -p dash-ops --test conv1d_state -- --nocapture
//!   CUBECL_CUDA_TF32=0 cargo test --release -p dash-ops --features cuda --test conv1d_state
//!   DASH_OPS_DEVICE=wgpu:vulkan cargo test --release -p dash-ops --features wgpu --test conv1d_state
//!   DASH_OPS_DEVICE=cpu cargo test --release -p dash-ops --features cpu --test conv1d_state

mod common;

use burn::tensor::{DType, Device, Tensor, TensorData};
use common::adapter::{Burn, Watch};
use dash_conformance::conv1d_state::{self, Inputs, assert_packed, assert_splits, random_inputs};
use dash_conformance::errors;
use dash_ops::conv1d::{Activation, CausalConv1dOptions, causal_conv1d};
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

/// The tests read process-wide counters, so they run one at a time.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Every case against torch through `adapter`, which watches the kernels, and how many of the
/// cases they took.
fn fixtures(adapter: &Burn, label: &str) {
    conv1d_state::fixtures(adapter, label).all_ran_without(&adapter.lacking());
    let [taken, ran] = adapter.taken();
    println!("{label}: the kernels took {taken} of {ran} cases");
}

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
}

/// A sequence of `T` steps split at `split`, the second part from the first's final state,
/// through one autodiff graph, against the whole sequence: outputs, the final state and every
/// gradient. The state between the parts holds `max(K - 1, W')` steps, all the second part's
/// final state can reach back to. Returns the largest relative difference per part.
fn split_against_whole(
    device: &Device,
    dtype: DType,
    whole: &Inputs,
    activation: Activation,
    split: usize,
) -> BTreeMap<&'static str, f64> {
    let device = device.clone().autodiff();
    let [b, t, c, k] = whole.shape;
    let tensor = |values: &[f32], shape: Vec<usize>, dtype: DType| {
        Tensor::<3>::from_data(
            TensorData::new(values.to_vec(), shape).convert_dtype(dtype),
            (&device, dtype),
        )
    };
    let x = tensor(&whole.x, vec![b, t, c], dtype).require_grad();
    let weight = tensor(&whole.weight, vec![1, c, k], dtype)
        .reshape([c, k])
        .detach()
        .require_grad();
    let bias = whole.bias.as_ref().map(|bias| {
        tensor(bias, vec![1, 1, c], dtype)
            .reshape([c])
            .detach()
            .require_grad()
    });
    let initial = whole
        .initial
        .as_ref()
        .map(|(w, values)| tensor(values, vec![b, *w, c], DType::F32).require_grad());
    let (w_out, grad_final) = whole.grad_final.clone().expect("a final state's gradient");
    let w = (k - 1).max(w_out);
    let grad_out = tensor(&whole.grad_out, vec![b, t, c], dtype);
    let grad_final = tensor(&grad_final, vec![b, w_out, c], DType::F32);
    let options = CausalConv1dOptions::default().with_activation(activation);

    let mut results: Vec<BTreeMap<&'static str, Vec<f32>>> = Vec::new();
    for split in [None, Some(split)] {
        let input = x.clone();
        let (out, final_state) = match split {
            None => causal_conv1d(
                input,
                weight.clone(),
                bias.clone(),
                initial.clone(),
                None,
                options.with_final_state_len(w_out),
            ),
            Some(p) => {
                let first = input.clone().slice([0..b, 0..p, 0..c]);
                let second = input.slice([0..b, p..t, 0..c]);
                let (out1, state) = causal_conv1d(
                    first,
                    weight.clone(),
                    bias.clone(),
                    initial.clone(),
                    None,
                    options.with_final_state_len(w),
                );
                let (out2, final_state) = causal_conv1d(
                    second,
                    weight.clone(),
                    bias.clone(),
                    state,
                    None,
                    options.with_final_state_len(w_out),
                );
                (Tensor::cat(vec![out1, out2], 1), final_state)
            }
        };
        let final_state = final_state.expect("the final state");
        let loss = (out.clone() * grad_out.clone()).sum().cast(DType::F32)
            + (final_state.clone() * grad_final.clone()).sum();
        let mut parts = BTreeMap::new();
        parts.insert("out", values(out));
        parts.insert("final_state", values(final_state));
        let grads = loss.backward();
        parts.insert("grad_x", values(x.grad(&grads).unwrap()));
        parts.insert("grad_weight", values(weight.grad(&grads).unwrap()));
        if let Some(bias) = &bias {
            parts.insert("grad_bias", values(bias.grad(&grads).unwrap()));
        }
        if let Some(initial) = &initial {
            parts.insert("grad_initial_state", values(initial.grad(&grads).unwrap()));
        }
        results.push(parts);
    }
    let (whole, split) = (&results[0], &results[1]);
    whole
        .iter()
        .map(|(part, expected)| (*part, errors(&split[part], expected).1))
        .collect()
}

/// [`conv1d_state::splits`]' splits through one autodiff graph: the parts are views of one
/// tracked input (`slice`), the first part's tracked final state is the second's initial state,
/// the outputs are joined with `cat` (whose backward hands each call a view of the output's
/// gradient), and one backward pass gives every gradient, the weight's and the bias's from both
/// calls. Where the conformance body composes the calls on the host, this holds dash-ops' autodiff
/// to passing a gradient back through a tracked initial state, and the kernels to reading other
/// strides than a shape implies.
fn splits(device: &Device, label: &str) -> BTreeMap<&'static str, f64> {
    let mut worst: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut seed = 0x51ed_2701_7a11_c0deu64;
    let silu = Activation::Silu;
    let configs = [
        (4, true, Some(3), 3, silu),
        (4, false, None, 3, silu),
        (5, true, Some(6), 6, Activation::None),
        (2, true, Some(1), 4, silu),
    ];
    for (k, bias, w, w_out, activation) in configs {
        let whole = random_inputs(&mut seed, [2, 200, 6, k], bias, w, Some(w_out), None);
        for split in [1, 2, 37, 64, 150, 199] {
            let differences = split_against_whole(device, DType::F32, &whole, activation, split);
            for (part, rel) in differences {
                println!(
                    "{label} split K = {k}, W = {w:?}, W' = {w_out}, at {split}: {part} rel {rel:.3e}"
                );
                let entry = worst.entry(part).or_insert(0.0);
                *entry = entry.max(rel);
            }
        }
    }
    for (part, rel) in &worst {
        println!("{label} split worst {part}: rel {rel:.3e}");
    }
    worst
}

/// Whether `device` is a CubeCL device but a CUDA one (lavapipe, OpenGL, the CPU runtime): the
/// reference sums in other orders on the CPU runtime, and those bounds are a little wider.
#[cfg(feature = "cubecl")]
fn other_runtime(device: &Device) -> bool {
    common::runtime_of(device).is_some() && !common::on_cuda(device)
}

#[cfg(feature = "flex")]
#[test]
fn fixtures_on_flex_match_torch() {
    let _serial = serial();
    let flex = Burn::new(Device::flex())
        .with_int(DType::I64)
        .watching(Watch::Counters);
    fixtures(&flex, "flex");
}

#[cfg(feature = "flex")]
#[test]
fn a_split_sequence_on_flex_equals_the_whole() {
    let _serial = serial();
    assert_splits(
        &conv1d_state::splits(&Burn::new(Device::flex()), "flex").all_ran(),
        false,
    );
}

#[cfg(feature = "flex")]
#[test]
fn a_split_sequence_in_one_graph_on_flex_equals_the_whole() {
    let _serial = serial();
    assert_splits(&splits(&Device::flex(), "flex"), false);
}

#[cfg(feature = "flex")]
#[test]
fn packed_sequences_on_flex_equal_one_call_each() {
    let _serial = serial();
    let flex = Burn::new(Device::flex());
    assert_packed(&conv1d_state::packed_against_calls(&flex, "flex").all_ran());
}

#[cfg(feature = "flex")]
#[test]
fn empty_sequences_on_flex_pass_their_states_through() {
    let _serial = serial();
    conv1d_state::empty_sequences(&Burn::new(Device::flex())).all_ran();
}

#[cfg(feature = "flex")]
#[test]
fn either_output_alone_on_flex_has_its_own_gradients() {
    let _serial = serial();
    conv1d_state::outputs_alone(&Burn::new(Device::flex())).all_ran();
}

/// The same checks on a CubeCL device, any runtime: the one `DASH_OPS_DEVICE` names.
#[cfg(feature = "cubecl")]
mod cube {
    use super::*;

    fn device() -> Device {
        common::cube_device()
    }

    #[test]
    fn fixtures_on_cubecl_match_torch() {
        let _serial = serial();
        println!("{}", common::describe(&device()));
        // The device's own int dtype, and 64-bit offsets (fla's) where the device has them.
        fixtures(&Burn::new(device()).watching(Watch::Counters), "cubecl");
        if device().supports_dtype(DType::I64) {
            let i64s = Burn::new(device())
                .with_int(DType::I64)
                .watching(Watch::Counters);
            fixtures(&i64s, "cubecl i64");
        } else {
            println!("skipped: 64-bit cu_seqlens, which the device does not have");
        }
    }

    #[test]
    fn empty_sequences_on_cubecl_pass_their_states_through() {
        let _serial = serial();
        conv1d_state::empty_sequences(&Burn::new(device())).all_ran();
    }

    #[test]
    fn either_output_alone_on_cubecl_has_its_own_gradients() {
        let _serial = serial();
        conv1d_state::outputs_alone(&Burn::new(device())).all_ran();
    }

    #[test]
    fn a_split_sequence_on_cubecl_equals_the_whole() {
        let _serial = serial();
        let worst = conv1d_state::splits(&Burn::new(device()), "cubecl").all_ran();
        assert_splits(&worst, other_runtime(&device()));
    }

    #[test]
    fn a_split_sequence_in_one_graph_on_cubecl_equals_the_whole() {
        let _serial = serial();
        assert_splits(&splits(&device(), "cubecl"), other_runtime(&device()));
    }

    #[test]
    fn packed_sequences_on_cubecl_equal_one_call_each() {
        let _serial = serial();
        assert_packed(
            &conv1d_state::packed_against_calls(&Burn::new(device()), "cubecl").all_ran(),
        );
    }

    /// The kernels against Flex's reference where the fixtures do not reach: 1000 steps packed
    /// into sequences whose boundaries fall anywhere, hundreds of channel vectors; for several
    /// widths, with and without a bias, SiLU, initial and final states; bf16 too. The adapter
    /// asserts which kernels ran.
    #[cfg(feature = "flex")]
    #[test]
    fn packed_kernels_match_the_reference_at_scale() {
        use dash_conformance::DType as Dtype;
        use dash_conformance::conv1d_state::{cumulative, run, uniform};

        let _serial = serial();
        let mut seed = 0x7a1e_5ca1_e000_0001u64;
        let c = 1544;
        let silu = Activation::Silu;
        let configs = [
            (4, true, Some(3), Some(3), silu, Dtype::F32),
            (2, false, None, Some(2), Activation::None, Dtype::F32),
            (5, true, Some(8), None, silu, Dtype::F32),
            (8, true, Some(7), Some(9), silu, Dtype::F32),
            (4, true, Some(3), Some(3), silu, Dtype::BF16),
        ];
        for (k, bias, w, w_out, activation, dtype) in configs {
            if !common::has(&device(), common::adapter::burn_dtype(dtype)) {
                continue;
            }
            // Lengths from 0 to 99, as many as fill 1000 steps.
            let mut lengths = Vec::new();
            let mut total = 0;
            while total < 1000 {
                let n = (uniform(&mut seed, 1, 1.0)[0].abs() * 100.0) as usize % 100;
                let n = n.min(1000 - total);
                lengths.push(n);
                total += n;
            }
            let cu = cumulative(&lengths);
            let inputs = random_inputs(&mut seed, [1, 1000, c, k], bias, w, w_out, Some(cu));
            let cube = Burn::new(device()).watching(Watch::Counters);
            let actual = run(&cube, dtype, &inputs, activation).expect("the kernels' call");
            let flex = Burn::new(Device::flex());
            let expected = run(&flex, dtype, &inputs, activation).expect("the reference's call");
            for (part, expected) in &expected {
                let (abs, rel) = errors(&actual[part], expected);
                // f32: rounding, the weight and bias gradients' long sums most; bf16, an ulp of
                // each side's own rounding (the initial states' gradient is f32). The largest
                // measured: on CUDA, then on lavapipe / OpenGL / the CPU runtime (whose reference
                // sums in other orders; neither of the others has bf16).
                let bound = match (dtype, *part, other_runtime(&device())) {
                    (_, "final_state", _) => 0.0,
                    (Dtype::F32, "out", _) => 2.6e-7, // 2.58e-7; 2.58e-7 / 2.58e-7 / 1.39e-7
                    (Dtype::F32, "grad_x", _) => 2.4e-7, // 2.35e-7; 1.56e-7 / 2.35e-7 / 1.56e-7
                    (Dtype::F32, "grad_weight", false) => 2.9e-7, // 2.81e-7
                    (Dtype::F32, "grad_weight", true) => 3.8e-7, // 2.81e-7 / 3.75e-7 / 2.11e-7
                    (Dtype::F32, "grad_bias", false) => 3.3e-7, // 3.27e-7
                    (Dtype::F32, "grad_bias", true) => 5.1e-7, // 5.03e-7 / 5.03e-7 / 2.01e-7
                    (_, "grad_initial_state", _) => 1.7e-7, // 1.69e-7 on each, f32 and bf16
                    (_, "out", false) => 1.5e-3,      // 1.49e-3
                    (_, "grad_x", false) => 1.1e-3,   // 1.07e-3
                    (_, "out", true) => 3.8e-4,       // 3.72e-4
                    (_, "grad_x", true) => 4.3e-3,    // 4.29e-3
                    (_, "grad_weight" | "grad_bias", false) => 7.0e-7, // 6.94e-7, 0
                    (_, "grad_weight" | "grad_bias", true) => 1.8e-4, // 1.78e-4, 0
                    _ => panic!("no bound for {dtype:?} {part}"),
                };
                println!(
                    "cubecl against flex, {} sequences, K = {k}, bias {bias}, W = {w:?}, \
                     W' = {w_out:?}, {dtype:?}, {part}: max abs {abs:.3e}, rel {rel:.3e}",
                    lengths.len()
                );
                assert!(rel <= bound, "{part}: {rel:.3e} exceeds {bound:.1e}");
            }
        }
    }
}
