//! Time one training step of MobileNetV3-Small with five heads at batch 128 on Flex.
use std::time::Instant;

use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::activation::log_softmax;
use burn::tensor::{Device, Distribution, Int, Tensor};
use dash_models::MobileNetV3Small;

fn main() {
    let batch: usize = std::env::args().nth(1).map_or(128, |s| s.parse().unwrap());
    let steps: usize = std::env::args().nth(2).map_or(3, |s| s.parse().unwrap());
    let device = Device::flex().autodiff();
    let mut model = MobileNetV3Small::new(&[3, 3, 3, 2, 2], 0.2, &device);
    let mut optim = AdamWConfig::new().with_weight_decay(0.05).init();
    let x = Tensor::<4>::random(
        [batch, 3, 224, 224],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let y = Tensor::<1, Int>::zeros([batch], &device);
    for step in 0..steps + 1 {
        let t0 = Instant::now();
        let logits = model.forward(x.clone());
        let t1 = Instant::now();
        let loss = logits
            .into_iter()
            .map(|l| {
                -log_softmax(l, 1)
                    .gather(1, y.clone().unsqueeze_dim(1))
                    .mean()
            })
            .reduce(|a, b| a + b)
            .unwrap();
        let grads = loss.backward();
        let t2 = Instant::now();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(1e-3, model, grads);
        let _ = model.heads[0].weight.val().into_data();
        let t3 = Instant::now();
        println!(
            "step {step}: forward {:.2} s, backward {:.2} s, optimiser {:.2} s, total {:.2} s",
            (t1 - t0).as_secs_f64(),
            (t2 - t1).as_secs_f64(),
            (t3 - t2).as_secs_f64(),
            (t3 - t0).as_secs_f64()
        );
    }
    let valid = model.valid();
    let xi = Tensor::<4>::random(
        [batch, 3, 224, 224],
        Distribution::Normal(0.0, 1.0),
        &Device::flex(),
    );
    let t0 = Instant::now();
    let out = valid.forward(xi);
    let _ = out[0].clone().into_data();
    println!("inference forward {:.2} s", t0.elapsed().as_secs_f64());
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines().filter(|l| l.starts_with("VmHWM")) {
            println!("{line}");
        }
    }
}
