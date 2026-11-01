//! The operations take dash-contract's dtypes on every side: f64 and integers are refused by the
//! gateway, where one side's kernels would refuse them and the other's reference might not.

use dash_mag::ops::{
    AttentionMask, AttentionOptions, CausalConv1dOptions, CausalConv1dStepOptions,
    ChunkGatedDeltaOptions, GatedDeltaStepOptions, LogprobOptions, LowRankOptions, SampleOptions,
    attention, attention_decode, causal_conv1d, causal_conv1d_update, chosen_logprob,
    chunk_gated_delta_rule, gated_delta_step, low_rank, sample,
};
use dash_mag::{DType, Tensor};

fn every_operation_refuses(dtype: DType) {
    let wrong = |shape: &[usize]| {
        let zeros = Tensor::zeros(shape, DType::F32).unwrap();
        zeros.as_dtype(dtype).unwrap()
    };
    let none = AttentionMask::None;
    let options = AttentionOptions::default();
    let (q4, q3, q2, q1) = (
        wrong(&[1, 2, 2, 4]),
        wrong(&[1, 2, 4]),
        wrong(&[2, 4]),
        wrong(&[2]),
    );
    assert!(attention(&q4, &q4, &q4, &none, options).is_err());
    assert!(attention_decode(&q3, &q4, &q4, None, options).is_err());
    assert!(causal_conv1d(&q3, &q2, None, None, None, CausalConv1dOptions::default()).is_err());
    assert!(causal_conv1d_update(&q2, &q2, None, &q3, CausalConv1dStepOptions::default()).is_err());
    let chunk = ChunkGatedDeltaOptions::default();
    assert!(chunk_gated_delta_rule(&q4, &q4, &q4, &q3, &q3, None, None, chunk).is_err());
    let step = GatedDeltaStepOptions::default();
    assert!(gated_delta_step(&q3, &q3, &q3, &q2, &q2, &q4, None, step).is_err());
    let targets = Tensor::from_i32(&[0, 1], &[2]).unwrap();
    assert!(chosen_logprob(&q2, &q2, &targets, &[], LogprobOptions::default()).is_err());
    assert!(sample(&q2, &SampleOptions::greedy()).is_err());
    assert!(low_rank(&q2, &q2, &q2, None, LowRankOptions::default()).is_err());
    let _ = q1;
}

#[test]
fn f64_is_refused_on_every_side() {
    every_operation_refuses(DType::F64);
}

#[test]
fn integers_are_refused_on_every_side() {
    every_operation_refuses(DType::I32);
}
