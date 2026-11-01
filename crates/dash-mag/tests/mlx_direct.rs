//! On the MLX side, every entry point is its dash-mlx operation: outputs and gradients the same bits
//! as calling dash-mlx directly under MLX's vector-Jacobian product, in f32 and in bf16. So what
//! dash-mlx's tests hold, against dash-ops' fixtures, the gateway holds too. The kernels are checked
//! to have run.

#![cfg(target_os = "macos")]

mod common;

use common::values;
use dash_mag::ops::{
    Activation, AttentionMask, AttentionOptions, CausalConv1dOptions, CausalConv1dStepOptions,
    ChunkGatedDeltaOptions, GatedDeltaStepOptions, LogprobOptions, LowRankOptions, SampleOptions,
    attention, attention_decode, causal_conv1d, causal_conv1d_update, chosen_logprob,
    chunk_gated_delta_rule, gated_delta_step, low_rank, sample,
};
use dash_mag::{DType, Tensor, vjp};
use dash_mlx::mlx::error::Result;
use dash_mlx::mlx::ops::sigmoid;
use dash_mlx::mlx::transforms::fallible_vjp;
use dash_mlx::mlx::{Array, Dtype};

fn dtypes() -> [(DType, Dtype); 2] {
    [(DType::F32, Dtype::Float32), (DType::BF16, Dtype::Bfloat16)]
}

fn front(seed: u64, shape: &[usize], dtype: DType) -> Tensor {
    common::tensor(seed, shape, dtype)
}

/// The same values as [`front`], as MLX's array.
fn plain(seed: u64, shape: &[i32], dtype: Dtype) -> Array {
    let n = shape.iter().product::<i32>() as usize;
    Array::from_slice(&values(seed, n), shape)
        .as_dtype(dtype)
        .unwrap()
}

fn f32s(a: &Array) -> Vec<u32> {
    let a = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
    a.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
}

fn same(what: &str, got: &Tensor, want: &Array) {
    assert_eq!(common::bits(got), f32s(want), "{what}");
}

/// The cotangent [`vjp`] takes as f32, rounded to `dtype`, its output's.
fn cotangent(seed: u64, shape: &[i32], dtype: Dtype) -> Array {
    plain(seed, shape, Dtype::Float32).as_dtype(dtype).unwrap()
}

#[test]
fn conv1d() {
    for (dtype, mlx) in dtypes() {
        let (b, t, c, k) = (2usize, 40usize, 8usize, 4usize);
        let (bi, ti, ci, ki) = (b as i32, t as i32, c as i32, k as i32);
        let options = CausalConv1dOptions::default()
            .with_activation(Activation::Silu)
            .with_final_state(true);
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
            &[
                Some(front(4, &[b, t, c], DType::F32)),
                Some(front(5, &[b, k - 1, c], DType::F32)),
            ],
        )
        .unwrap();
        let (y, g) = fallible_vjp(
            |p: &[Array]| {
                let (y, s) = dash_mlx::conv1d::causal_conv1d(
                    &p[0],
                    &p[1],
                    None,
                    Some(&p[2]),
                    None,
                    options,
                )?;
                Ok(vec![y, s.unwrap()])
            },
            &[
                plain(1, &[bi, ti, ci], mlx),
                plain(2, &[ci, ki], mlx),
                plain(3, &[bi, ki - 1, ci], Dtype::Float32),
            ],
            &[
                cotangent(4, &[bi, ti, ci], mlx),
                cotangent(5, &[bi, ki - 1, ci], Dtype::Float32),
            ],
        )
        .unwrap();
        same("conv1d output", &outs[0], &y[0]);
        same("conv1d state", &outs[1], &y[1]);
        same("conv1d x", &grads[0], &g[0]);
        same("conv1d weight", &grads[1], &g[1]);
        same("conv1d initial state", &grads[2], &g[2]);

        // And a decoding step, without gradients.
        let step = CausalConv1dStepOptions::default().with_activation(Activation::Silu);
        let (y, next) = causal_conv1d_update(
            &front(6, &[b, c], dtype),
            &front(2, &[c, k], dtype),
            None,
            &front(3, &[b, k - 1, c], DType::F32),
            step,
        )
        .unwrap();
        let (yd, nd) = dash_mlx::conv1d::causal_conv1d_update(
            &plain(6, &[bi, ci], mlx),
            &plain(2, &[ci, ki], mlx),
            None,
            &plain(3, &[bi, ki - 1, ci], Dtype::Float32),
            step,
        )
        .unwrap();
        same("conv1d step", &y, &yd);
        same("conv1d next state", &next, &nd);
    }
    assert!(
        dash_mlx::conv1d::probe::kernel_launches() > 0,
        "the conv1d kernels ran"
    );
}

#[test]
fn gated_delta_chunked_and_step() {
    for (dtype, mlx) in dtypes() {
        let (b, t, h, d) = (1usize, 130usize, 4usize, 32usize);
        let (bi, ti, hi, di) = (b as i32, t as i32, h as i32, d as i32);
        let options = ChunkGatedDeltaOptions::default().with_final_state(true);
        let (outs, grads) = vjp(
            |p| {
                let (y, s) =
                    chunk_gated_delta_rule(&p[0], &p[1], &p[2], &p[3], &p[4], None, None, options)?;
                Ok(vec![y, s.unwrap()])
            },
            &[
                front(11, &[b, t, h, d], dtype),
                front(12, &[b, t, h, d], dtype),
                front(13, &[b, t, h, d], dtype),
                front(14, &[b, t, h], DType::F32)
                    .abs()
                    .unwrap()
                    .neg()
                    .unwrap(),
                front(15, &[b, t, h], DType::F32).sigmoid().unwrap(),
            ],
            &[
                Some(front(16, &[b, t, h, d], DType::F32)),
                Some(front(17, &[b, h, d, d], DType::F32)),
            ],
        )
        .unwrap();
        let (y, g) = fallible_vjp(
            |p: &[Array]| {
                let (y, s) = dash_mlx::gated_delta::chunk::chunk_gated_delta_rule(
                    &p[0], &p[1], &p[2], &p[3], &p[4], None, None, options,
                )?;
                Ok(vec![y, s.unwrap()])
            },
            &[
                plain(11, &[bi, ti, hi, di], mlx),
                plain(12, &[bi, ti, hi, di], mlx),
                plain(13, &[bi, ti, hi, di], mlx),
                plain(14, &[bi, ti, hi], Dtype::Float32)
                    .abs()
                    .unwrap()
                    .negative()
                    .unwrap(),
                sigmoid(plain(15, &[bi, ti, hi], Dtype::Float32)).unwrap(),
            ],
            &[
                cotangent(16, &[bi, ti, hi, di], mlx),
                cotangent(17, &[bi, hi, di, di], Dtype::Float32),
            ],
        )
        .unwrap();
        same("chunk output", &outs[0], &y[0]);
        same("chunk state", &outs[1], &y[1]);
        for (i, name) in ["q", "k", "v", "g", "beta"].iter().enumerate() {
            same(&format!("chunk {name}"), &grads[i], &g[i]);
        }

        // A step from that state's shape, without gradients.
        let step = GatedDeltaStepOptions::default();
        let (y, next) = gated_delta_step(
            &front(21, &[b, h, d], dtype),
            &front(22, &[b, h, d], dtype),
            &front(23, &[b, h, d], dtype),
            &front(24, &[b, h], DType::F32).abs().unwrap().neg().unwrap(),
            &front(25, &[b, h], DType::F32).sigmoid().unwrap(),
            &front(18, &[b, h, d, d], DType::F32),
            None,
            step,
        )
        .unwrap();
        let (yd, nd) = dash_mlx::gated_delta::recurrent::gated_delta_step(
            &plain(21, &[bi, hi, di], mlx),
            &plain(22, &[bi, hi, di], mlx),
            &plain(23, &[bi, hi, di], mlx),
            &plain(24, &[bi, hi], Dtype::Float32)
                .abs()
                .unwrap()
                .negative()
                .unwrap(),
            &sigmoid(plain(25, &[bi, hi], Dtype::Float32)).unwrap(),
            &plain(18, &[bi, hi, di, di], Dtype::Float32),
            None,
            step,
        )
        .unwrap();
        same("step output", &y, &yd);
        same("step state", &next, &nd);
    }
    assert!(
        dash_mlx::gated_delta::chunk::scan_kernels_take(32, 32),
        "the delta rule's kernels take this shape"
    );
}

#[test]
fn attention_and_decoding() {
    for (dtype, mlx) in dtypes() {
        let (b, t, hq, hkv, d) = (2usize, 48usize, 8usize, 2usize, 64usize);
        let (bi, ti, hqi, hkvi, di) = (b as i32, t as i32, hq as i32, hkv as i32, d as i32);
        let primals = || {
            [
                front(31, &[b, t, hq, d], dtype),
                front(32, &[b, t, hkv, d], dtype),
                front(33, &[b, t, hkv, d], dtype),
            ]
        };
        let direct = || {
            [
                plain(31, &[bi, ti, hqi, di], mlx),
                plain(32, &[bi, ti, hkvi, di], mlx),
                plain(33, &[bi, ti, hkvi, di], mlx),
            ]
        };
        let seed = || [cotangent(34, &[bi, ti, hqi, di], mlx)];
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
            &primals(),
            &[Some(front(34, &[b, t, hq, d], DType::F32))],
        )
        .unwrap();
        let (y, g) = fallible_vjp(
            |p: &[Array]| {
                Ok(vec![dash_mlx::attention::attention(
                    &p[0],
                    &p[1],
                    &p[2],
                    &dash_mlx::attention::AttentionMask::None,
                    options,
                )?])
            },
            &direct(),
            &seed(),
        )
        .unwrap();
        same("attention output", &outs[0], &y[0]);
        same("attention query", &grads[0], &g[0]);
        same("attention key", &grads[1], &g[1]);
        same("attention value", &grads[2], &g[2]);

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
            &primals(),
            &[Some(front(34, &[b, t, hq, d], DType::F32))],
        )
        .unwrap();
        let mask = Array::from_slice(&pad, &[bi, ti]);
        let (y, g) = fallible_vjp(
            |p: &[Array]| {
                Ok(vec![dash_mlx::attention::attention(
                    &p[0],
                    &p[1],
                    &p[2],
                    &dash_mlx::attention::AttentionMask::KeyPadding(mask.clone()),
                    AttentionOptions::default(),
                )?])
            },
            &direct(),
            &seed(),
        )
        .unwrap();
        same("masked attention output", &outs[0], &y[0]);
        same("masked attention query", &grads[0], &g[0]);

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
        let want = dash_mlx::attention::attention_decode(
            &plain(35, &[bi, hqi, di], mlx),
            &plain(36, &[bi, 9, hkvi, di], mlx),
            &plain(37, &[bi, 9, hkvi, di], mlx),
            Some(&Array::from_slice(&lengths, &[bi])),
            AttentionOptions::default(),
        )
        .unwrap();
        same("attention decode", &got, &want);
    }
    assert!(
        dash_mlx::attention::probe::tiles() > 0,
        "the attention kernels ran"
    );
}

#[test]
fn chosen_logprob_and_sampling() {
    for (dtype, mlx) in dtypes() {
        let (n, d, v) = (12usize, 16usize, 50usize);
        let (ni, di, vi) = (n as i32, d as i32, v as i32);
        let ids: Vec<i32> = (0..ni).map(|i| (i * 3 + 1) % vi).collect();
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
        let (w, ts) = (plain(42, &[vi, di], mlx), Array::from_slice(&ids, &[ni]));
        let mut entropy = None;
        let (y, g) = fallible_vjp(
            |p: &[Array]| -> Result<Vec<Array>> {
                let out = dash_mlx::logprob::chosen_logprob(
                    &p[0],
                    &w,
                    &ts,
                    &masked,
                    LogprobOptions::default(),
                )?;
                entropy = Some(out.entropy);
                Ok(vec![out.logprob])
            },
            &[plain(41, &[ni, di], mlx)],
            &[cotangent(43, &[ni], Dtype::Float32)],
        )
        .unwrap();
        same("logprob", &outs[0], &y[0]);
        same("entropy", &outs[1], &entropy.unwrap());
        same("logprob gradient", &grads[0], &g[0]);

        let options = SampleOptions {
            temperature: 0.8,
            top_k: 10,
            top_p: 0.95,
            seed: 7,
            offset: 3,
            masked: vec![47],
        };
        let got = sample(&front(44, &[4, v], dtype), &options).unwrap();
        assert_eq!(got.tokens.dtype(), DType::I32);
        let want = dash_mlx::sample::sample(&plain(44, &[4, vi], mlx), &options).unwrap();
        assert_eq!(
            got.tokens.to_i32().unwrap(),
            want.tokens.contiguous().unwrap().as_slice::<i32>(),
            "tokens"
        );
        same("sampled logprob", &got.logprob, &want.logprob);
        same(
            "tempered logprob",
            &got.tempered_logprob,
            &want.tempered_logprob,
        );
        same(
            "sampling logprob",
            &got.sampling_logprob,
            &want.sampling_logprob,
        );
    }
    assert!(
        dash_mlx::logprob::probe::forward_passes() > 0,
        "the logprob kernels ran"
    );
}

#[test]
fn low_rank_product() {
    for (dtype, mlx) in dtypes() {
        let (n, i, r, o) = (6usize, 16usize, 4usize, 8usize);
        let (ni, ii, ri, oi) = (n as i32, i as i32, r as i32, o as i32);
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
        let (y, g) = fallible_vjp(
            |p: &[Array]| {
                Ok(vec![dash_mlx::lowrank::low_rank(
                    &p[0],
                    &p[1],
                    &p[2],
                    Some(&p[3]),
                    options,
                )?])
            },
            &[
                plain(51, &[ni, ii], mlx),
                plain(52, &[ii, ri], Dtype::Float32),
                plain(53, &[ri, oi], Dtype::Float32),
                plain(54, &[ni, oi], mlx),
            ],
            &[cotangent(55, &[ni, oi], mlx)],
        )
        .unwrap();
        same("low rank output", &outs[0], &y[0]);
        for (i, name) in ["x", "u", "v", "base"].iter().enumerate() {
            same(&format!("low rank {name}"), &grads[i], &g[i]);
        }
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

/// The norm and the rotation are MLX's own: its `fast` functions, one op each, so that a model's
/// numbers are the ones mlx-lm's stack gives.
#[test]
fn rms_norm_and_rope_are_mlx_s_own() {
    use dash_mag::rope::Rope;
    use dash_mlx::mlx::fast;

    for (dtype, mlx) in dtypes() {
        let x = front(71, &[2, 5, 3, 8], dtype);
        let plain_x = plain(71, &[2, 5, 3, 8], mlx);
        let w = front(72, &[8], dtype);
        let plain_w = plain(72, &[8], mlx);
        same(
            "rms_norm with a weight",
            &x.rms_norm(Some(&w), 1e-6).unwrap(),
            &fast::rms_norm(&plain_x, Some(&plain_w), 1e-6).unwrap(),
        );
        same(
            "rms_norm without",
            &x.rms_norm(None, 1e-6).unwrap(),
            &fast::rms_norm(&plain_x, None, 1e-6).unwrap(),
        );
        // Each with the other as its weight: what they promote to, as MLX gives it.
        let (other, plain_other) = match dtype {
            DType::F32 => (DType::BF16, Dtype::Bfloat16),
            _ => (DType::F32, Dtype::Float32),
        };
        let got = x.rms_norm(Some(&front(72, &[8], other)), 1e-6).unwrap();
        let want = fast::rms_norm(&plain_x, Some(&plain(72, &[8], plain_other)), 1e-6).unwrap();
        assert_eq!(got.dtype(), DType::F32);
        same("rms_norm with a weight of the other", &got, &want);

        let rope = Rope::new(4, 10000.0, 64, dtype).unwrap();
        let turned = plain_x.swap_axes(1, 2).unwrap();
        let want = fast::rope(&turned, 4, false, 10000.0, 1.0, 7).unwrap();
        same(
            "rope over a range",
            &rope.apply_range(&x, 7..12).unwrap(),
            &want.swap_axes(1, 2).unwrap(),
        );
        let token = front(73, &[2, 3, 8], dtype);
        let positions = Tensor::from_i32(&[7, 40], &[2]).unwrap();
        let want = fast::rope_dynamic(
            plain(73, &[2, 3, 8], mlx).reshape(&[2, 3, 1, 8]).unwrap(),
            4,
            false,
            10000.0,
            1.0,
            Array::from_slice(&[7i32, 40], &[2]),
        )
        .unwrap();
        same(
            "rope at rows",
            &rope.apply_at(&token, &positions).unwrap(),
            &want.reshape(&[2, 3, 8]).unwrap(),
        );
    }
}
