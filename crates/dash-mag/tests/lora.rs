//! LoRA's layer: on the Burn side the same bits as dash-lora's, output and gradients, with a bf16
//! and an f32 base; and on every side, PEFT's files through the layers and back byte for byte.

#![cfg(not(target_os = "macos"))]

mod common;

use common::{bits, close, tensor};
use dash_mag::lora::{
    AdapterLayer, ApplyOptions, DEFAULT_ADAPTER, HostTensor, Lora, LoraLinear, LoraWeights,
    PeftAdapter, PeftConfig,
};
use dash_mag::{DType, Tensor, vjp};
use std::path::PathBuf;

fn fixture(dir: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../dash-lora/tests/fixtures")
        .join(dir)
}

fn layer(dtype: DType) -> LoraLinear {
    let mut layer = LoraLinear::frozen(&tensor(1, &[8, 16], dtype)).unwrap();
    let lora = Lora::new(
        &tensor(2, &[4, 16], DType::F32),
        &tensor(3, &[8, 4], DType::F32),
        2.0,
    )
    .unwrap();
    layer.insert_adapter(DEFAULT_ADAPTER, lora).unwrap();
    layer
}

/// `f` over the chosen adapter's factors and `x`.
fn run(layer: &LoraLinear, x: &Tensor, cotangent: &Tensor) -> (Vec<Tensor>, Vec<Tensor>) {
    let lora = layer.lora().unwrap();
    let mut held = layer.clone();
    vjp(
        |p| {
            held.set_factors(p[1].clone(), p[2].clone())?;
            Ok(vec![held.forward(&p[0])?])
        },
        &[x.clone(), lora.a.clone(), lora.b.clone()],
        &[Some(cotangent.clone())],
    )
    .unwrap()
}

#[test]
fn with_no_adapter_it_is_the_base_alone() {
    let mut layer = layer(DType::BF16);
    let x = tensor(4, &[3, 16], DType::BF16);
    let base = x.matmul(&layer.weight().t().unwrap()).unwrap();
    let with = layer.forward(&x).unwrap();
    assert_ne!(bits(&with), bits(&base));
    layer.set_adapter(None);
    assert_eq!(bits(&layer.forward(&x).unwrap()), bits(&base));
    layer.set_adapter(Some("another"));
    assert_eq!(bits(&layer.forward(&x).unwrap()), bits(&base));
    layer.set_adapter(Some(DEFAULT_ADAPTER));
    assert_eq!(bits(&layer.forward(&x).unwrap()), bits(&with));
}

#[test]
fn it_is_the_base_plus_the_low_rank_term_at_x_s_dtype() {
    let layer = layer(DType::F32);
    let x = tensor(4, &[3, 16], DType::F32);
    let lora = layer.lora().unwrap();
    let base = x.matmul(&layer.weight().t().unwrap()).unwrap();
    let term = x
        .matmul(&lora.a.t().unwrap())
        .unwrap()
        .matmul(&lora.b.t().unwrap())
        .unwrap()
        .mul_scalar(2.0)
        .unwrap();
    let want = base.add(&term).unwrap();
    close(&layer.forward(&x).unwrap(), &want.to_f32().unwrap(), 1e-6);
    // Inputs of any rank, as rows.
    let cube = tensor(5, &[2, 3, 16], DType::F32);
    let out = layer.forward(&cube).unwrap();
    assert_eq!(out.shape(), [2, 3, 8]);
    let rows = layer.forward(&cube.reshape(&[6, 16]).unwrap()).unwrap();
    assert_eq!(bits(&out), bits(&rows));
}

#[test]
fn factors_are_refused_when_they_do_not_fit() {
    let mut layer = layer(DType::F32);
    let a = tensor(6, &[4, 15], DType::F32);
    let b = tensor(7, &[8, 4], DType::F32);
    assert!(
        layer
            .insert_adapter("x", Lora::new(&a, &b, 1.0).unwrap())
            .is_err()
    );
    assert!(Lora::new(&a, &tensor(8, &[8, 3], DType::F32), 1.0).is_err());
    assert!(layer.set_factors(a, b).is_err());
}

#[test]
fn peft_s_files_through_the_layers_and_back_are_byte_identical() {
    for dir in [
        "peft_tiny",
        "peft_patterns",
        "peft_multi",
        "peft_multi/reference",
    ] {
        let config = std::fs::read_to_string(fixture(dir).join("adapter_config.json")).unwrap();
        let weights = std::fs::read(fixture(dir).join("adapter_model.safetensors")).unwrap();
        let adapter = PeftAdapter::read(fixture(dir)).unwrap();
        let mut named = layers(&adapter);
        adapter
            .apply(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
            .unwrap();
        for (path, layer) in &named {
            let lora = layer.adapter(DEFAULT_ADAPTER).unwrap();
            assert_eq!(lora.scale, adapter.scale(path).unwrap(), "{dir} {path}");
            assert_eq!(lora.a.dtype(), DType::F32);
        }
        let back =
            PeftAdapter::from_layers(adapter.config.clone(), named.iter().map(|(p, l)| (p, l)))
                .unwrap();
        let (config2, weights2) = back.to_bytes().unwrap();
        assert!(weights2 == weights, "{dir}: the factors changed on the way");
        assert_eq!(config2, config, "{dir}");
    }
}

fn layers(adapter: &PeftAdapter) -> Vec<(String, LoraLinear)> {
    adapter
        .modules()
        .map(|(path, w)| {
            let weight = Tensor::zeros(&[w.b.shape()[0], w.a.shape()[1]], DType::BF16).unwrap();
            (path.to_string(), LoraLinear::frozen(&weight).unwrap())
        })
        .collect()
}

#[test]
fn a_frozen_adapter_beside_the_default_each_keeps_its_file_s_factors() {
    let adapter = PeftAdapter::read(fixture("peft_multi")).unwrap();
    let reference = PeftAdapter::read(fixture("peft_multi/reference")).unwrap();
    let mut named = layers(&adapter);
    for (path, layer) in layers(&reference) {
        if !named.iter().any(|(p, _)| *p == path) {
            named.push((path, layer));
        }
    }
    adapter
        .apply(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &())
        .unwrap();
    let frozen = ApplyOptions {
        name: "reference".into(),
        trainable: false,
    };
    reference
        .apply_with(named.iter_mut().map(|(p, l)| (p.as_str(), l)), &(), &frozen)
        .unwrap();
    for (path, layer) in &named {
        for (name, file) in [(DEFAULT_ADAPTER, &adapter), ("reference", &reference)] {
            let held = AdapterLayer::lora_weights(layer, name).map(|(w, _)| w);
            assert_eq!(held.as_ref(), file.module(path), "{path} {name}");
        }
    }
}

/// Silently leaving dropout out would train something else, so it is refused.
#[test]
fn dropout_is_refused_when_read() {
    let weights = LoraWeights {
        a: HostTensor::from_f32(vec![1, 4], &[1.0; 4]).unwrap(),
        b: HostTensor::from_f32(vec![3, 1], &[1.0; 3]).unwrap(),
    };
    assert!(<LoraLinear as AdapterLayer>::make_lora(&weights, 1.0, 0.1, true, &()).is_err());
    assert!(<LoraLinear as AdapterLayer>::make_lora(&weights, 1.0, 0.0, true, &()).is_ok());
    let _ = PeftConfig::new(1, 1.0, &["m"]);
}

/// The Burn side against dash-lora's layer.
#[test]
fn on_burn_it_is_dash_lora_s_layer_bit_for_bit() {
    use burn::tensor::{DType as BurnDType, Device, Tensor as Burn, TensorData};
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(not(feature = "cuda"))]
    let device = Device::flex();

    for (dtype, burn) in [(DType::BF16, BurnDType::BF16), (DType::F32, BurnDType::F32)] {
        let plain = |seed: u64, shape: [usize; 2], d: BurnDType| {
            let n = shape.iter().product();
            Burn::<2>::from_data(TensorData::new(common::values(seed, n), shape), &device).cast(d)
        };
        let f32s = |t: Burn<2>| -> Vec<u32> {
            let data = t.inner().cast(BurnDType::F32).into_data();
            data.try_to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let front = layer(dtype);
        let x = tensor(4, &[3, 16], dtype);
        let (outs, grads) = run(&front, &x, &tensor(5, &[3, 8], DType::F32));

        let ad = device.clone().autodiff();
        let a = plain(2, [4, 16], BurnDType::F32);
        let b = plain(3, [8, 4], BurnDType::F32);
        let direct = dash_lora::LoraLinear::frozen(plain(1, [8, 16], burn).autodiff())
            .with_lora(dash_lora::Lora::new(a.autodiff(), b.autodiff(), 2.0));
        let _ = ad;
        let xd = plain(4, [3, 16], burn).autodiff().require_grad();
        let y = direct.forward(xd.clone());
        let c = plain(5, [3, 8], BurnDType::F32).cast(y.dtype());
        let g = (y.clone() * c).sum().cast(BurnDType::F32).backward();
        let lora = direct.lora().unwrap();
        assert_eq!(bits(&outs[0]), f32s(y), "output, {dtype:?}");
        assert_eq!(bits(&grads[0]), f32s(xd.grad(&g).unwrap()), "x, {dtype:?}");
        assert_eq!(
            bits(&grads[1]),
            f32s(lora.a.val().grad(&g).unwrap()),
            "a, {dtype:?}"
        );
        assert_eq!(
            bits(&grads[2]),
            f32s(lora.b.val().grad(&g).unwrap()),
            "b, {dtype:?}"
        );
    }
}
