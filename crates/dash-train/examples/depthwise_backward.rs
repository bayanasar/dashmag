//! Where a training step's backward time goes: each depthwise convolution of
//! MobileNetV3-Small at batch 128 and 224 x 224, timed forward, with the gradient of its
//! input only, and with the gradient of its weight only.
use std::time::Instant;

use burn::module::{Module, Param};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2dConfig;
use burn::tensor::{Device, Distribution, Tensor};

/// (channels, kernel, stride, input side) of each depthwise layer.
const LAYERS: [(usize, usize, usize, usize); 11] = [
    (16, 3, 2, 112),
    (72, 3, 2, 56),
    (88, 3, 1, 28),
    (96, 5, 2, 28),
    (240, 5, 1, 14),
    (240, 5, 1, 14),
    (120, 5, 1, 14),
    (144, 5, 1, 14),
    (288, 5, 2, 14),
    (576, 5, 1, 7),
    (576, 5, 1, 7),
];

fn time(f: impl FnOnce()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64()
}

fn main() {
    let batch: usize = std::env::args().nth(1).map_or(128, |s| s.parse().unwrap());
    let device = Device::flex().autodiff();
    let (mut fwd, mut dx, mut dw) = (0.0, 0.0, 0.0);
    println!("channels kernel stride side: forward, input grad, weight grad (s)");
    for (c, k, s, side) in LAYERS {
        let pad = (k - 1) / 2;
        let conv = Conv2dConfig::new([c, c], [k, k])
            .with_stride([s, s])
            .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
            .with_groups(c)
            .with_bias(false)
            .init(&device);
        let x = Tensor::<4>::random(
            [batch, c, side, side],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let f = time(|| {
            let _ = conv
                .valid()
                .forward(x.clone().without_autodiff())
                .into_data();
        });
        // Input gradient only: the weight is held constant.
        let frozen = Conv2dConfig::new([c, c], [k, k])
            .with_stride([s, s])
            .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
            .with_groups(c)
            .with_bias(false)
            .init(&device);
        let frozen = burn::nn::conv::Conv2d {
            weight: Param::from_tensor(frozen.weight.val().detach()).set_require_grad(false),
            ..frozen
        };
        let xg = x.clone().require_grad();
        let a = time(|| {
            let g = frozen.forward(xg.clone()).sum().backward();
            let _ = xg.grad(&g).unwrap().into_data();
        });
        // Weight gradient only: the input is a constant.
        let b = time(|| {
            let g = conv.forward(x.clone().detach()).sum().backward();
            let _ = conv.weight.grad(&g).unwrap().into_data();
        });
        println!("{c:4} {k} {s} {side:3}: {f:.3}  {a:.3}  {b:.3}");
        fwd += f;
        dx += a;
        dw += b;
    }
    println!(
        "all depthwise layers: forward {fwd:.2} s, input grad (incl. forward) {dx:.2} s, weight grad (incl. forward) {dw:.2} s"
    );
}
