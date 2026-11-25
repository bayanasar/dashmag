//! 2-D convolution of images, torch's `F.conv2d`: any kernel size, stride, symmetric zero padding
//! and dilation, channels in groups, and an optional bias; with the gradients of the input, the
//! weight and the bias.
//!
//! With `x` `[N, C, H, W]`, `weight` `[O, C / G, KH, KW]` for `G` groups, and each output channel
//! `o` reading the input channels of its group, `g = o / (O / G)`:
//!
//! ```text
//! out[n, o, i, j] = bias[o] + sum_{c < C / G, p < KH, q < KW}
//!                   weight[o, c, p, q] * x[n, g (C / G) + c, i sh + p dh - ph, j sw + q dw - pw]
//! ```
//!
//! `x` read outside `0..H` by `0..W` is zero (the padding). `(sh, sw)` is the stride, `(ph, pw)`
//! the padding on each side, `(dh, dw)` the dilation; without a bias, `bias[o]` is zero. The output
//! has `HO = (H + 2 ph - dh (KH - 1) - 1) / sh + 1` rows and `WO` columns likewise, rounded down.
//! Gradients flow to `x`, the weight and the bias. This is torch's `F.conv2d(x, weight, bias,
//! stride, padding, dilation, groups)` with an integer pair for each option.
//!
//! # Layouts and dtypes
//!
//! - `x` and its gradient `[N, C, H, W]`, the output and its gradient `[N, O, HO, WO]`: torch's
//!   NCHW, the layout checkpoints store and name. Images are the one place the conventions'
//!   channels-last layout of sequences does not apply; an implementation whose framework convolves
//!   channels-last transposes inside the call. Inputs may have any strides; outputs are contiguous.
//! - `weight` and its gradient `[O, C / G, KH, KW]`, torch's OIHW; `bias` and its gradient `[O]`.
//! - `C` and `O` are multiples of `G`; depthwise convolution is `G = C`, its output channels
//!   `O = C m` for a multiplier `m`.
//!
//! `x`, `weight` and `bias` share one float dtype. The output and each gradient come back in it,
//! each element a sum of products accumulated in f32, IEEE f32 and not TF32, rounded once. Every
//! implementation takes f32. bf16 and f16 are the conventions' too, but no implementation is held
//! to them yet, so each refuses them, as a call with an input it does not have. For f16, Burn's
//! Flex sums f16 products in f16 (its native f16 matrix product), which the contract does not
//! allow; that is read in Burn's source, not measured. For bf16 there is no known reason against
//! it: no side has been measured in it, and it is refused until one is.
//!
//! # Gradients
//!
//! `x`'s gradient is the transposed convolution of the output's gradient with the weight, the
//! weight's the correlation of `x` with the output's gradient, the bias's the output's gradient
//! summed over `N`, `HO` and `WO`. An implementation takes its framework's own backward pass; each
//! element of each gradient is a sum of products like the output's, and is held to the same bound.
//!
//! # Bounds
//!
//! Each element of the output and of each gradient is a sum of `n` products (the output's `n =
//! (C / G) KH KW`, plus the bias), and its error is counted in roundings: in units of `2^-24` times
//! the sum of the absolute values of its terms. Computed in f32 in any order, with or without fused
//! multiply-adds, it is within `n + 1` of them; dash-conformance holds every implementation to that
//! per element and prints the largest count measured. A Winograd or FFT convolution, or TF32, can
//! exceed it, and is outside the contract.

/// The arguments of `conv2d` that have a usual default: stride 1, no padding, no dilation, one
/// group. Each pair is `(rows, columns)`, as torch's tuples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Conv2dOptions {
    /// The step between the windows of consecutive outputs. Default `[1, 1]`.
    pub stride: [usize; 2],
    /// The zeros added on each side of the input, before and after. Default `[0, 0]`.
    pub padding: [usize; 2],
    /// The step between the kernel's taps. Default `[1, 1]`.
    pub dilation: [usize; 2],
    /// The groups the channels are split into, each output channel reading its group's inputs.
    /// Default 1.
    pub groups: usize,
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Self {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        }
    }
}

impl Conv2dOptions {
    /// The options with this stride.
    pub fn with_stride(self, stride: [usize; 2]) -> Self {
        Self { stride, ..self }
    }

    /// The options with this padding on each side.
    pub fn with_padding(self, padding: [usize; 2]) -> Self {
        Self { padding, ..self }
    }

    /// The options with this dilation.
    pub fn with_dilation(self, dilation: [usize; 2]) -> Self {
        Self { dilation, ..self }
    }

    /// The options with the channels in this many groups.
    pub fn with_groups(self, groups: usize) -> Self {
        Self { groups, ..self }
    }

    /// The output's shape `[N, O, HO, WO]` for `x` of shape `x` and the weight of shape `weight`,
    /// with a bias of `bias` elements if there is one; or why the call is outside the contract,
    /// `conv2d: ...`, the same words on every implementation. Shapes only: the dtypes are the
    /// implementation's to check.
    pub fn output_shape(
        &self,
        x: &[usize],
        weight: &[usize],
        bias: Option<usize>,
    ) -> Result<[usize; 4], String> {
        let refuse = |why: String| Err(format!("conv2d: {why}"));
        let ([n, c, h, w], [o, cg, kh, kw]) = match (x, weight) {
            (&[n, c, h, w], &[o, cg, kh, kw]) => ([n, c, h, w], [o, cg, kh, kw]),
            _ => {
                return refuse(format!(
                    "x is [N, C, H, W] and the weight [O, C / G, KH, KW], got {x:?} and \
                     {weight:?}"
                ));
            }
        };
        let g = self.groups;
        if self.stride.contains(&0) || self.dilation.contains(&0) || g == 0 {
            return refuse(format!(
                "stride {:?}, dilation {:?} and groups {g} are at least 1",
                self.stride, self.dilation
            ));
        }
        if c % g != 0 || o % g != 0 || cg * g != c {
            return refuse(format!(
                "{g} groups of x's {c} channels and the weight's {o} outputs want a weight \
                 [{o}, C / G = {}, KH, KW] with C and O multiples of the groups, got {weight:?}",
                c / g
            ));
        }
        if kh == 0 || kw == 0 {
            return refuse(format!("a kernel of {kh} x {kw}"));
        }
        if let Some(len) = bias
            && len != o
        {
            return refuse(format!("a bias of {len} for {o} output channels"));
        }
        let span = |size: usize, pad: usize, dilation: usize, k: usize, stride: usize| {
            let padded = size + 2 * pad;
            let reach = dilation * (k - 1) + 1;
            (padded >= reach).then(|| (padded - reach) / stride + 1)
        };
        let (Some(ho), Some(wo)) = (
            span(h, self.padding[0], self.dilation[0], kh, self.stride[0]),
            span(w, self.padding[1], self.dilation[1], kw, self.stride[1]),
        ) else {
            return refuse(format!(
                "the kernel {kh} x {kw} at dilation {:?} reaches past x's {h} x {w} padded by \
                 {:?}",
                self.dilation, self.padding
            ));
        };
        Ok([n, o, ho, wo])
    }
}

#[cfg(test)]
mod tests {
    use super::Conv2dOptions;

    #[test]
    fn the_output_shape_is_torchs() {
        let o = Conv2dOptions::default();
        assert_eq!(
            o.output_shape(&[2, 3, 8, 7], &[4, 3, 3, 3], Some(4)),
            Ok([2, 4, 6, 5])
        );
        let o = o.with_stride([2, 1]).with_padding([1, 2]);
        assert_eq!(
            o.output_shape(&[1, 3, 10, 9], &[5, 3, 3, 5], None),
            Ok([1, 5, 5, 9])
        );
        let o = Conv2dOptions::default()
            .with_padding([2, 0])
            .with_dilation([2, 1])
            .with_stride([1, 2]);
        assert_eq!(
            o.output_shape(&[2, 2, 9, 12], &[3, 2, 3, 3], None),
            Ok([2, 3, 9, 5])
        );
        let depthwise = Conv2dOptions::default().with_groups(4).with_padding([2, 2]);
        assert_eq!(
            depthwise.output_shape(&[2, 4, 9, 8], &[8, 1, 5, 5], Some(8)),
            Ok([2, 8, 9, 8])
        );
        // Exactly one window.
        let o = Conv2dOptions::default().with_dilation([3, 1]);
        assert_eq!(
            o.output_shape(&[1, 1, 7, 3], &[1, 1, 3, 3], None),
            Ok([1, 1, 1, 1])
        );
    }

    #[test]
    fn what_is_outside_the_contract_is_refused() {
        let o = Conv2dOptions::default();
        for (options, x, w, bias) in [
            (o, &[2, 3, 8][..], &[4, 3, 3, 3][..], None),
            (o, &[2, 3, 8, 8], &[4, 2, 3, 3], None),
            (o.with_groups(2), &[2, 3, 8, 8], &[4, 1, 3, 3], None),
            (o.with_groups(2), &[2, 4, 8, 8], &[3, 2, 3, 3], None),
            (o.with_groups(0), &[2, 4, 8, 8], &[4, 4, 3, 3], None),
            (o.with_stride([0, 1]), &[2, 4, 8, 8], &[4, 4, 3, 3], None),
            (o.with_dilation([1, 0]), &[2, 4, 8, 8], &[4, 4, 3, 3], None),
            (o, &[2, 4, 8, 8], &[4, 4, 0, 3], None),
            (o, &[2, 4, 8, 8], &[4, 4, 3, 3], Some(3)),
            (o, &[2, 4, 2, 8], &[4, 4, 3, 3], None),
            (o.with_dilation([2, 1]), &[2, 4, 4, 8], &[4, 4, 3, 3], None),
        ] {
            let refused = options.output_shape(x, w, bias);
            assert!(
                refused
                    .as_ref()
                    .is_err_and(|why| why.starts_with("conv2d: ")),
                "{options:?} {x:?} {w:?} {bias:?}: {refused:?}"
            );
        }
    }
}
