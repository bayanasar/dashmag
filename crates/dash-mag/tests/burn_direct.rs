//! On the Burn side, every entry point is its dash-ops operation: outputs and gradients the same
//! bits as calling dash-ops directly with Burn's autodiff, in f32 and in bf16. So what dash-ops'
//! tests hold, against PyTorch, transformers and fla, the gateway holds too. On CUDA the kernels
//! are checked to have run.

#![cfg(not(target_os = "macos"))]

mod common;

use burn::tensor::{DType as BurnDType, Device, Gradients, Int, Tensor as Burn, TensorData};
use common::values;
use dash_mag::ops::{
    Activation, AttentionMask, AttentionOptions, CausalConv1dOptions, CausalConv1dStepOptions,
    ChunkGatedDeltaOptions, GatedDeltaStepOptions, LogprobOptions, LowRankOptions, SampleOptions,
    attention, attention_decode, causal_conv1d, causal_conv1d_update, chosen_logprob,
    chunk_gated_delta_rule, gated_delta_step, low_rank, sample,
};
use dash_mag::{DType, Tensor, vjp};
use dash_ops::attention as direct_attention;
use dash_ops::gated_delta::{chunk as direct_chunk, recurrent as direct_recurrent};

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::flex()
    }
}

fn dtypes() -> [(DType, BurnDType); 2] {
    [(DType::F32, BurnDType::F32), (DType::BF16, BurnDType::BF16)]
}

fn front(seed: u64, shape: &[usize], dtype: DType) -> Tensor {
    common::tensor(seed, shape, dtype)
}

/// The same values as [`front`], on Burn's device, recording.
fn leaf<const D: usize>(seed: u64, shape: [usize; D], dtype: BurnDType) -> Burn<D> {
    plain::<D>(seed, shape, dtype).autodiff().require_grad()
}

fn plain<const D: usize>(seed: u64, shape: [usize; D], dtype: BurnDType) -> Burn<D> {
    let n = shape.iter().product();
    Burn::<D>::from_data(TensorData::new(values(seed, n), shape), &device()).cast(dtype)
}

fn f32s<const D: usize>(t: Burn<D>) -> Vec<u32> {
    let data = t.inner().cast(BurnDType::F32).into_data();
    data.try_to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn same(what: &str, got: &Tensor, want: Vec<u32>) {
    assert_eq!(common::bits(got), want, "{what}");
}

/// `sum(out * cotangent)` as `vjp` takes it, its cotangent from `seed`.
fn term<const D: usize>(out: &Burn<D>, seed: u64) -> Burn<1> {
    let c = plain::<D>(seed, out.dims(), BurnDType::F32).cast(out.dtype());
    (out.clone() * c).sum().cast(BurnDType::F32)
}

fn grad<const D: usize>(t: &Burn<D>, grads: &Gradients) -> Vec<u32> {
    f32s(t.grad(grads).expect("a gradient"))
}

#[test]
fn conv1d() {
    for (dtype, burn) in dtypes() {
        let (b, t, c, k) = (2, 40, 8, 4);
        let options = CausalConv1dOptions::default()
            .with_activation(Activation::Silu)
            .with_final_state(true);
        let d_out = front(4, &[b, t, c], DType::F32);
        let d_state = front(5, &[b, k - 1, c], DType::F32);
        let (outs, grads) = vjp(
            |p| {
                let (y, s) = causal_conv1d(&p[0], &p[1], None, Some(&p[2]), None, options)?;
                Ok(vec![y, s.unwrap()])
            },
            &[
                front(1, &[b, t, c], dtype),
                front(2, &[c, k], dtype),
                front(3, &[b, k - 1, c], DType::F32),
            ],
            &[Some(d_out), Some(d_state)],
        )
        .unwrap();

        let (x, w, s) = (
            leaf(1, [b, t, c], burn),
            leaf(2, [c, k], burn),
            leaf(3, [b, k - 1, c], BurnDType::F32),
        );
        let (y, state) = dash_ops::conv1d::causal_conv1d(
            x.clone(),
            w.clone(),
            None,
            Some(s.clone()),
            None,
            options,
        );
        let state = state.unwrap();
        let g = (term(&y, 4) + term(&state, 5)).backward();
        same("conv1d output", &outs[0], f32s(y));
        same("conv1d state", &outs[1], f32s(state));
        same("conv1d x", &grads[0], grad(&x, &g));
        same("conv1d weight", &grads[1], grad(&w, &g));
        same("conv1d initial state", &grads[2], grad(&s, &g));

        // And a decoding step, without gradients.
        let (x, w, s) = (
            front(6, &[b, c], dtype),
            front(2, &[c, k], dtype),
            front(3, &[b, k - 1, c], DType::F32),
        );
        let step = CausalConv1dStepOptions::default().with_activation(Activation::Silu);
        let (y, next) = causal_conv1d_update(&x, &w, None, &s, step).unwrap();
        let (yd, nd) = dash_ops::conv1d::causal_conv1d_update(
            plain(6, [b, c], burn),
            plain(2, [c, k], burn),
            None,
            plain(3, [b, k - 1, c], BurnDType::F32),
            step,
        );
        same("conv1d step", &y, f32s(yd));
        same("conv1d next state", &next, f32s(nd));
    }
    #[cfg(feature = "cuda")]
    assert!(
        dash_ops::conv1d::probe::kernel_launches() > 0,
        "the conv1d kernels ran"
    );
}

#[test]
fn gated_delta_chunked_and_step() {
    for (dtype, burn) in dtypes() {
        let (b, t, h, d) = (1, 130, 4, 32);
        let options = ChunkGatedDeltaOptions::default().with_final_state(true);
        let primals = [
            front(11, &[b, t, h, d], dtype),
            front(12, &[b, t, h, d], dtype),
            front(13, &[b, t, h, d], dtype),
            front(14, &[b, t, h], DType::F32)
                .abs()
                .unwrap()
                .neg()
                .unwrap(),
            front(15, &[b, t, h], DType::F32).sigmoid().unwrap(),
        ];
        let (outs, grads) = vjp(
            |p| {
                let (y, s) =
                    chunk_gated_delta_rule(&p[0], &p[1], &p[2], &p[3], &p[4], None, None, options)?;
                Ok(vec![y, s.unwrap()])
            },
            &primals,
            &[
                Some(front(16, &[b, t, h, d], DType::F32)),
                Some(front(17, &[b, h, d, d], DType::F32)),
            ],
        )
        .unwrap();

        let (q, k, v) = (
            leaf(11, [b, t, h, d], burn),
            leaf(12, [b, t, h, d], burn),
            leaf(13, [b, t, h, d], burn),
        );
        let g0 = plain::<3>(14, [b, t, h], BurnDType::F32)
            .abs()
            .neg()
            .detach()
            .autodiff()
            .require_grad();
        let beta0 = burn::tensor::activation::sigmoid(plain::<3>(15, [b, t, h], BurnDType::F32))
            .detach()
            .autodiff()
            .require_grad();
        let (y, state) = direct_chunk::chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g0.clone(),
            beta0.clone(),
            None,
            None,
            options,
        );
        let state = state.unwrap();
        let g = (term(&y, 16) + term(&state, 17)).backward();
        same("chunk output", &outs[0], f32s(y));
        same("chunk state", &outs[1], f32s(state));
        for (i, x) in [&q, &k, &v].into_iter().enumerate() {
            same("chunk q, k, v", &grads[i], grad(x, &g));
        }
        same("chunk g", &grads[3], grad(&g0, &g));
        same("chunk beta", &grads[4], grad(&beta0, &g));

        // A step from that state's shape, without gradients.
        let step = GatedDeltaStepOptions::default();
        let state = front(18, &[b, h, d, d], DType::F32);
        let (y, next) = gated_delta_step(
            &front(21, &[b, h, d], dtype),
            &front(22, &[b, h, d], dtype),
            &front(23, &[b, h, d], dtype),
            &front(24, &[b, h], DType::F32).abs().unwrap().neg().unwrap(),
            &front(25, &[b, h], DType::F32).sigmoid().unwrap(),
            &state,
            None,
            step,
        )
        .unwrap();
        let (yd, nd) = direct_recurrent::gated_delta_step(
            plain(21, [b, h, d], burn),
            plain(22, [b, h, d], burn),
            plain(23, [b, h, d], burn),
            plain::<2>(24, [b, h], BurnDType::F32).abs().neg(),
            burn::tensor::activation::sigmoid(plain::<2>(25, [b, h], BurnDType::F32)),
            plain(18, [b, h, d, d], BurnDType::F32),
            None,
            step,
        );
        same("step output", &y, f32s(yd));
        same("step state", &next, f32s(nd));
    }
    #[cfg(feature = "cuda")]
    assert!(
        direct_chunk::probe::kernel_launches() > 0,
        "the delta rule's kernels ran"
    );
}

#[test]
fn attention_and_decoding() {
    for (dtype, burn) in dtypes() {
        let (b, t, hq, hkv, d) = (2, 48, 8, 2, 64);
        let options = AttentionOptions::default().with_causal(true);
        let (outs, grads) = vjp(
            |p| {
                Ok(vec![attention(
                    &p[0],
                    &p[1],
                    &p[2],
                    &AttentionMask::None,
                    options,
                )?])
            },
            &[
                front(31, &[b, t, hq, d], dtype),
                front(32, &[b, t, hkv, d], dtype),
                front(33, &[b, t, hkv, d], dtype),
            ],
            &[Some(front(34, &[b, t, hq, d], DType::F32))],
        )
        .unwrap();
        let (q, k, v) = (
            leaf(31, [b, t, hq, d], burn),
            leaf(32, [b, t, hkv, d], burn),
            leaf(33, [b, t, hkv, d], burn),
        );
        let y = direct_attention::attention(
            q.clone(),
            k.clone(),
            v.clone(),
            direct_attention::AttentionMask::None,
            options,
        );
        let g = term(&y, 34).backward();
        same("attention output", &outs[0], f32s(y));
        same("attention query", &grads[0], grad(&q, &g));
        same("attention key", &grads[1], grad(&k, &g));
        same("attention value", &grads[2], grad(&v, &g));

        // A key padding: a constant mask inside the product.
        let pad: Vec<bool> = (0..b * t).map(|i| i % 7 == 0).collect();
        let padding = Tensor::from_bool(&pad, &[b, t]).unwrap();
        let (outs, grads) = vjp(
            |p| {
                let mask = AttentionMask::KeyPadding(padding.clone());
                Ok(vec![attention(
                    &p[0],
                    &p[1],
                    &p[2],
                    &mask,
                    AttentionOptions::default(),
                )?])
            },
            &[
                front(31, &[b, t, hq, d], dtype),
                front(32, &[b, t, hkv, d], dtype),
                front(33, &[b, t, hkv, d], dtype),
            ],
            &[Some(front(34, &[b, t, hq, d], DType::F32))],
        )
        .unwrap();
        let (q, k, v) = (
            leaf(31, [b, t, hq, d], burn),
            leaf(32, [b, t, hkv, d], burn),
            leaf(33, [b, t, hkv, d], burn),
        );
        let mask =
            Burn::<2, burn::tensor::Bool>::from_data(TensorData::new(pad, [b, t]), &device())
                .autodiff();
        let y = direct_attention::attention(
            q.clone(),
            k.clone(),
            v.clone(),
            direct_attention::AttentionMask::KeyPadding(mask),
            AttentionOptions::default(),
        );
        let g = term(&y, 34).backward();
        same("masked attention output", &outs[0], f32s(y));
        same("masked attention query", &grads[0], grad(&q, &g));

        // Decoding against a cache, with lengths.
        let lengths = [5i32, 9];
        let got = attention_decode(
            &front(35, &[b, hq, d], dtype),
            &front(36, &[b, 9, hkv, d], dtype),
            &front(37, &[b, 9, hkv, d], dtype),
            Some(&Tensor::from_i32(&lengths, &[b]).unwrap()),
            AttentionOptions::default(),
        )
        .unwrap();
        let want = direct_attention::attention_decode(
            plain(35, [b, hq, d], burn),
            plain(36, [b, 9, hkv, d], burn),
            plain(37, [b, 9, hkv, d], burn),
            Some(Burn::<1, Int>::from_data(
                TensorData::new(lengths.to_vec(), [b]),
                &device(),
            )),
            AttentionOptions::default(),
        );
        same("attention decode", &got, f32s(want));
    }
    #[cfg(feature = "cuda")]
    assert!(
        direct_attention::probe::kernel_launches() > 0,
        "the attention kernels ran"
    );
}

#[test]
fn chosen_logprob_and_sampling() {
    for (dtype, burn) in dtypes() {
        let (n, d, v) = (12, 16, 50);
        let ids: Vec<i32> = (0..n as i32).map(|i| (i * 3 + 1) % v as i32).collect();
        let targets = Tensor::from_i32(&ids, &[n]).unwrap();
        let weight = front(42, &[v, d], dtype);
        let masked = [47u32];
        let (outs, grads) = vjp(
            |p| {
                let out =
                    chosen_logprob(&p[0], &weight, &targets, &masked, LogprobOptions::default())?;
                Ok(vec![out.logprob, out.entropy])
            },
            &[front(41, &[n, d], dtype)],
            &[Some(front(43, &[n], DType::F32)), None],
        )
        .unwrap();
        let h = leaf(41, [n, d], burn);
        let out = dash_ops::logprob::chosen_logprob(
            h.clone(),
            plain(42, [v, d], burn).autodiff(),
            Burn::<1, Int>::from_data(TensorData::new(ids.clone(), [n]), &device()).autodiff(),
            &masked,
            LogprobOptions::default(),
        );
        let g = term(&out.logprob, 43).backward();
        same("logprob", &outs[0], f32s(out.logprob));
        same("entropy", &outs[1], f32s(out.entropy));
        same("logprob gradient", &grads[0], grad(&h, &g));

        let options = SampleOptions {
            temperature: 0.8,
            top_k: 10,
            top_p: 0.95,
            seed: 7,
            offset: 3,
            masked: vec![47],
        };
        let got = sample(&front(44, &[4, v], dtype), &options).unwrap();
        let want = dash_ops::sample::sample(plain(44, [4, v], burn), &options);
        assert_eq!(
            got.tokens.to_i32().unwrap(),
            want.tokens
                .into_data()
                .convert::<i32>()
                .try_to_vec::<i32>()
                .unwrap(),
            "tokens"
        );
        same("sampled logprob", &got.logprob, f32s(want.logprob));
        same(
            "tempered logprob",
            &got.tempered_logprob,
            f32s(want.tempered_logprob),
        );
        same(
            "sampling logprob",
            &got.sampling_logprob,
            f32s(want.sampling_logprob),
        );
    }
    #[cfg(feature = "cuda")]
    assert!(
        dash_ops::logprob::probe::kernel_launches() > 0,
        "the logprob kernels ran"
    );
}

#[test]
fn low_rank_product() {
    for (dtype, burn) in dtypes() {
        let (n, i, r, o) = (6, 16, 4, 8);
        let options = LowRankOptions::default().with_scale(2.0);
        let (outs, grads) = vjp(
            |p| Ok(vec![low_rank(&p[0], &p[1], &p[2], Some(&p[3]), options)?]),
            &[
                front(51, &[n, i], dtype),
                front(52, &[i, r], DType::F32),
                front(53, &[r, o], DType::F32),
                front(54, &[n, o], dtype),
            ],
            &[Some(front(55, &[n, o], DType::F32))],
        )
        .unwrap();
        let (x, u, v, base) = (
            leaf(51, [n, i], burn),
            leaf(52, [i, r], BurnDType::F32),
            leaf(53, [r, o], BurnDType::F32),
            leaf(54, [n, o], burn),
        );
        let y = dash_ops::lowrank::low_rank(
            x.clone(),
            u.clone(),
            v.clone(),
            Some(base.clone()),
            options,
        );
        let g = term(&y, 55).backward();
        same("low rank output", &outs[0], f32s(y));
        same("low rank x", &grads[0], grad(&x, &g));
        same("low rank u", &grads[1], grad(&u, &g));
        same("low rank v", &grads[2], grad(&v, &g));
        same("low rank base", &grads[3], grad(&base, &g));
    }
}

#[test]
fn a_wrong_rank_is_refused_here_rather_than_a_panic_below() {
    let x = front(61, &[2, 3], DType::F32);
    assert!(
        attention(
            &x,
            &x,
            &x,
            &AttentionMask::None,
            AttentionOptions::default()
        )
        .is_err()
    );
    assert!(causal_conv1d(&x, &x, None, None, None, CausalConv1dOptions::default()).is_err());
    assert!(sample(&front(62, &[3], DType::F32), &SampleOptions::greedy()).is_err());
}
