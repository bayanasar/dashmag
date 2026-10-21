//! MobileNetV3-Small (Howard et al., 2019) as torchvision defines it, with any number of
//! linear heads on the pooled feature.
//!
//! torchvision's classifier is `Linear(576, 1024)`, Hardswish, Dropout, `Linear(1024, n)`.
//! Here the first linear layer is the shared `neck` and each output is one of `heads`: an
//! ImageNet classifier is one head of 1000; a multi-attribute model is several. Paths
//! below the classifier are torchvision's exactly; torchvision's `classifier.0` and
//! `classifier.3` are this module's `neck.0` and `heads.0`.
//!
//! Three constants pass every shape check and fail on numbers, so they are fixed here
//! rather than left to Burn's defaults:
//! - BatchNorm uses eps 1e-3 and momentum 0.01 (torchvision's `partial(nn.BatchNorm2d,
//!   eps=0.001, momentum=0.01)`), not 1e-5 and 0.1.
//! - Hardsigmoid is `relu6(x + 3) / 6`: slope 1/6, not the 0.2 of ONNX's HardSigmoid.
//! - Squeeze-excite widths are `make_divisible(c / 4, 8)`, not `c / 4`.

use burn::module::Module;
use burn::nn::activation::{Activation, HardSwish};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Identity, Linear, LinearConfig,
    PaddingConfig2d, Relu,
};
use burn::tensor::activation::{hard_sigmoid, hard_swish, relu};
use burn::tensor::module::adaptive_avg_pool2d;
use burn::tensor::{Device, Tensor};

const BN_EPS: f64 = 1e-3;
const BN_MOMENTUM: f64 = 0.01;
/// Width of the shared layer between the pooled feature and the heads.
pub const NECK: usize = 1024;
/// Channels of the pooled feature the neck reads.
pub const FEATURES: usize = 576;

/// Convolution, batch norm, activation — torchvision's `Conv2dNormActivation`. A tuple, so
/// the parameter paths are `.0.weight` and `.1.weight` as in torchvision's `Sequential`;
/// the activation (torchvision's index 2) has no parameters, and `Identity` stands in where
/// torchvision has no activation at all.
pub type ConvBnAct = (Conv2d, BatchNorm, Activation);

/// One entry of `features`: the stem and the last layer are a single `ConvBnAct`, the eleven
/// between are inverted residual blocks.
// Burn has no `Module` for `Box`, and the network holds a few dozen of these, built once.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum Feature {
    Conv(ConvBnAct),
    Block(InvertedResidual),
}

/// One layer inside an inverted residual block's `block`.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum Layer {
    Conv(ConvBnAct),
    Se(SqueezeExcitation),
}

#[derive(Module, Debug)]
pub struct InvertedResidual {
    pub block: Vec<Layer>,
    /// Stride 1 and as many channels out as in: the input is added to the output.
    #[module(skip)]
    pub residual: bool,
}

/// Squeeze-and-excitation: pool, 1x1 conv down, ReLU, 1x1 conv up, hardsigmoid, scale.
#[derive(Module, Debug)]
pub struct SqueezeExcitation {
    pub fc1: Conv2d,
    pub fc2: Conv2d,
}

#[derive(Module, Debug)]
pub struct MobileNetV3Small {
    pub features: Vec<Feature>,
    /// One linear layer, `FEATURES` to `NECK`, followed by Hardswish and dropout.
    pub neck: Vec<Linear>,
    pub dropout: Dropout,
    pub heads: Vec<Linear>,
}

#[derive(Clone, Copy)]
enum Act {
    Relu,
    HardSwish,
    None,
}

impl Act {
    fn module(self) -> Activation {
        match self {
            Act::Relu => Activation::Relu(Relu),
            Act::HardSwish => Activation::HardSwish(HardSwish),
            Act::None => Activation::Identity(Identity),
        }
    }
}

/// One row of torchvision's `_mobilenet_v3_conf("mobilenet_v3_small")`, at width 1.0.
struct Bneck {
    input: usize,
    kernel: usize,
    expanded: usize,
    out: usize,
    se: bool,
    act: Act,
    stride: usize,
}

const fn bneck(
    input: usize,
    kernel: usize,
    expanded: usize,
    out: usize,
    se: bool,
    act: Act,
    stride: usize,
) -> Bneck {
    Bneck {
        input,
        kernel,
        expanded,
        out,
        se,
        act,
        stride,
    }
}

const SMALL: [Bneck; 11] = [
    bneck(16, 3, 16, 16, true, Act::Relu, 2),
    bneck(16, 3, 72, 24, false, Act::Relu, 2),
    bneck(24, 3, 88, 24, false, Act::Relu, 1),
    bneck(24, 5, 96, 40, true, Act::HardSwish, 2),
    bneck(40, 5, 240, 40, true, Act::HardSwish, 1),
    bneck(40, 5, 240, 40, true, Act::HardSwish, 1),
    bneck(40, 5, 120, 48, true, Act::HardSwish, 1),
    bneck(48, 5, 144, 48, true, Act::HardSwish, 1),
    bneck(48, 5, 288, 96, true, Act::HardSwish, 2),
    bneck(96, 5, 576, 96, true, Act::HardSwish, 1),
    bneck(96, 5, 576, 96, true, Act::HardSwish, 1),
];

/// torchvision's `_make_divisible`: round to the nearest multiple of `divisor`, at least
/// `divisor`, and never more than 10% below `v`.
pub fn make_divisible(v: usize, divisor: usize) -> usize {
    let rounded = ((v + divisor / 2) / divisor * divisor).max(divisor);
    if (rounded as f64) < 0.9 * v as f64 {
        rounded + divisor
    } else {
        rounded
    }
}

fn conv_bn_act(
    channels: [usize; 2],
    kernel: usize,
    stride: usize,
    groups: usize,
    act: Act,
    device: &Device,
) -> ConvBnAct {
    let pad = (kernel - 1) / 2;
    let conv = Conv2dConfig::new(channels, [kernel, kernel])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
        .with_groups(groups)
        .with_bias(false)
        .init(device);
    let bn = BatchNormConfig::new(channels[1])
        .with_epsilon(BN_EPS)
        .with_momentum(BN_MOMENTUM)
        .init(device);
    (conv, bn, act.module())
}

fn forward_conv(m: &ConvBnAct, x: Tensor<4>) -> Tensor<4> {
    m.2.forward(m.1.forward(m.0.forward(x)))
}

impl SqueezeExcitation {
    fn new(channels: usize, squeeze: usize, device: &Device) -> Self {
        Self {
            fc1: Conv2dConfig::new([channels, squeeze], [1, 1]).init(device),
            fc2: Conv2dConfig::new([squeeze, channels], [1, 1]).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let scale = adaptive_avg_pool2d(x.clone(), [1, 1]);
        let scale = self.fc2.forward(relu(self.fc1.forward(scale)));
        x * hard_sigmoid(scale, 1.0 / 6.0, 0.5)
    }
}

impl InvertedResidual {
    fn new(c: &Bneck, device: &Device) -> Self {
        let mut block = Vec::new();
        if c.expanded != c.input {
            block.push(Layer::Conv(conv_bn_act(
                [c.input, c.expanded],
                1,
                1,
                1,
                c.act,
                device,
            )));
        }
        block.push(Layer::Conv(conv_bn_act(
            [c.expanded, c.expanded],
            c.kernel,
            c.stride,
            c.expanded,
            c.act,
            device,
        )));
        if c.se {
            let squeeze = make_divisible(c.expanded / 4, 8);
            block.push(Layer::Se(SqueezeExcitation::new(
                c.expanded, squeeze, device,
            )));
        }
        block.push(Layer::Conv(conv_bn_act(
            [c.expanded, c.out],
            1,
            1,
            1,
            Act::None,
            device,
        )));
        Self {
            block,
            residual: c.stride == 1 && c.input == c.out,
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let mut y = x.clone();
        for layer in &self.block {
            y = match layer {
                Layer::Conv(m) => forward_conv(m, y),
                Layer::Se(m) => m.forward(y),
            };
        }
        if self.residual { y + x } else { y }
    }
}

impl MobileNetV3Small {
    /// `heads` are the widths of the outputs, in output order; `dropout` is applied after
    /// the neck in training (torchvision: 0.2).
    pub fn new(heads: &[usize], dropout: f64, device: &Device) -> Self {
        let mut features = vec![Feature::Conv(conv_bn_act(
            [3, 16],
            3,
            2,
            1,
            Act::HardSwish,
            device,
        ))];
        features.extend(
            SMALL
                .iter()
                .map(|c| Feature::Block(InvertedResidual::new(c, device))),
        );
        features.push(Feature::Conv(conv_bn_act(
            [96, FEATURES],
            1,
            1,
            1,
            Act::HardSwish,
            device,
        )));
        Self {
            features,
            neck: vec![LinearConfig::new(FEATURES, NECK).init(device)],
            dropout: DropoutConfig::new(dropout).init(),
            heads: heads
                .iter()
                .map(|&n| LinearConfig::new(NECK, n).init(device))
                .collect(),
        }
    }

    /// NCHW float input, normalised as the weights were trained; one `[N, width]` tensor of
    /// logits per head, in the order the heads were given.
    pub fn forward(&self, x: Tensor<4>) -> Vec<Tensor<2>> {
        let mut x = x;
        for feature in &self.features {
            x = match feature {
                Feature::Conv(m) => forward_conv(m, x),
                Feature::Block(m) => m.forward(x),
            };
        }
        let z: Tensor<2> = adaptive_avg_pool2d(x, [1, 1]).flatten(1, 3);
        let z = self.dropout.forward(hard_swish(self.neck[0].forward(z)));
        self.heads.iter().map(|h| h.forward(z.clone())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// torchvision's squeeze widths for the small model's eleven blocks, from
    /// `_make_divisible(expanded // 4, 8)`; `c / 4` would give 4, 60, 30 and 36 where these
    /// have 8, 64, 32 and 40.
    #[test]
    fn squeeze_widths_are_torchvisions() {
        let got: Vec<usize> = SMALL
            .iter()
            .filter(|c| c.se)
            .map(|c| make_divisible(c.expanded / 4, 8))
            .collect();
        assert_eq!(got, [8, 24, 64, 64, 32, 40, 72, 144, 144]);
    }
}
