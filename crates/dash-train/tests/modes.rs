//! Validation during training is the inference the saved model will give.
//!
//! Burn switches batch norm and dropout by device: on an autodiff device they train, on
//! the plain device they infer, and `valid()` moves a module from one to the other. A
//! trainer that validated in the wrong mode would pass every shape check and only score
//! lower, and then keep the wrong epoch; this holds the trainer's validation path to the
//! saved weights, loaded fresh, bit for bit.

use burn::module::Module;
use burn::tensor::{Device, Distribution, Tensor};
use dash_models::{MobileNetV3Small, weights};

fn values(t: Vec<Tensor<2>>) -> Vec<f32> {
    t.into_iter()
        .flat_map(|t| t.into_data().try_to_vec::<f32>().unwrap())
        .collect()
}

#[test]
fn validation_logits_are_the_saved_models_logits() {
    let device = Device::flex().autodiff();
    device.seed(1);
    let model = MobileNetV3Small::new(&[3, 2], 0.2, &device);
    let x = Tensor::<4>::random([4, 3, 64, 64], Distribution::Normal(0.0, 1.0), &device);
    let infer = device.clone().without_autodiff();
    let xi = x.clone().without_autodiff();
    let untrained = values(model.valid().forward(xi.clone()));
    // A training forward uses batch statistics and dropout, and moves the running
    // statistics, as a step does.
    let trained = values(model.forward(x.clone()));

    let valid = model.valid();
    let validation = values(valid.forward(xi.clone()));

    let mut loaded = MobileNetV3Small::new(&[3, 2], 0.0, &infer);
    weights::load_safetensors(&mut loaded, weights::to_safetensors(&valid).unwrap()).unwrap();
    let saved = values(loaded.forward(xi));

    assert_eq!(
        validation, saved,
        "validation must be the saved model's inference"
    );
    assert_ne!(trained, validation, "a training forward is not inference");
    assert_ne!(
        untrained, validation,
        "validation sees the running statistics training moved"
    );
}
