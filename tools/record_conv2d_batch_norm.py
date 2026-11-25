#!/usr/bin/env python3
"""Record conv2d's and batch norm's values and gradients from torch, for the tests that hold
dash-ops, dash-mlx and dash-mag to them.

Requires torch and safetensors (the CPU build of torch is enough). Usage, from the repository's
root:

    python tools/record_conv2d_batch_norm.py

writes

- crates/dash-ops/tests/fixtures/conv2d/cases.safetensors, which dash-conformance's `conv2d` body
  reads: per case the f32 inputs (`x`, `weight`, `bias`, the output's cotangent `d_out`), and in
  float64 torch's `F.conv2d` output and the gradients of `x`, the weight and the bias, evaluated
  from those inputs in float64; and beside each of them its magnitude, the same evaluated on the
  inputs' absolute values (the sum of the absolute values of the terms each element adds up),
  which the tests count roundings against;
- crates/dash-mag/tests/fixtures/batch_norm_torch.json, which dash-mag's `tests/batch_norm.rs`
  reads: per case the f32 inputs, and torch's `F.batch_norm` output, new running statistics and
  gradients evaluated in float64, in training and in eval mode, with the magnitudes the tests count
  roundings against.

The inputs are drawn by torch's generator from fixed seeds, so a rerun writes the same files.
"""
import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONV2D = os.path.join(ROOT, "crates/dash-ops/tests/fixtures/conv2d/cases.safetensors")
BATCH_NORM = os.path.join(ROOT, "crates/dash-mag/tests/fixtures/batch_norm_torch.json")

# name: (N, C_in, H, W), (C_out, K_h, K_w), stride, padding, dilation, groups, bias
CONV2D_CASES = {
    "plain_3x3": ((2, 3, 8, 7), (4, 3, 3), (1, 1), (0, 0), (1, 1), 1, True),
    "padded_no_bias": ((2, 4, 9, 9), (6, 3, 3), (1, 1), (1, 1), (1, 1), 1, False),
    "stride_2": ((2, 4, 11, 10), (8, 3, 3), (2, 2), (1, 1), (1, 1), 1, True),
    "uneven": ((1, 3, 10, 9), (5, 3, 5), (2, 1), (1, 2), (1, 1), 1, True),
    "dilated": ((2, 3, 12, 11), (4, 3, 3), (1, 1), (2, 2), (2, 2), 1, False),
    "dilated_uneven": ((2, 2, 9, 12), (3, 2, 3), (1, 2), (2, 0), (2, 1), 1, True),
    "groups_2": ((2, 4, 8, 8), (6, 3, 3), (1, 1), (1, 1), (1, 1), 2, True),
    "depthwise_stride_2": ((2, 8, 9, 9), (8, 3, 3), (2, 2), (1, 1), (1, 1), 8, False),
    "depthwise_x2_5x5": ((2, 4, 9, 8), (8, 5, 5), (1, 1), (2, 2), (1, 1), 4, True),
    "pointwise": ((2, 16, 5, 6), (12, 1, 1), (1, 1), (0, 0), (1, 1), 1, True),
    "pointwise_stride_2": ((2, 6, 9, 7), (4, 1, 1), (2, 2), (0, 0), (1, 1), 1, False),
    "wide_sums": ((2, 32, 6, 6), (16, 3, 3), (1, 1), (1, 1), (1, 1), 1, True),
    "padding_past_kernel": ((1, 2, 5, 4), (3, 2, 2), (1, 1), (2, 3), (1, 1), 1, True),
    "stem_7x7": ((1, 3, 16, 16), (4, 7, 7), (2, 2), (3, 3), (1, 1), 1, False),
}


def conv2d_case(gen, name, shape, kernel, stride, padding, dilation, groups, bias):
    n, cin, h, w = shape
    cout, kh, kw = kernel
    x = torch.randn(shape, generator=gen, dtype=torch.float32)
    weight = torch.randn((cout, cin // groups, kh, kw), generator=gen, dtype=torch.float32)
    weight = weight / (cin // groups * kh * kw) ** 0.5
    b = torch.randn((cout,), generator=gen, dtype=torch.float32) if bias else None
    options = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    out_shape = F.conv2d(x, weight, b, **options).shape
    d_out = torch.randn(out_shape, generator=gen, dtype=torch.float32)

    def run(x, weight, b, d_out):
        leaves = [t.double().requires_grad_() for t in (x, weight) + ((b,) if bias else ())]
        out = F.conv2d(leaves[0], leaves[1], leaves[2] if bias else None, **options)
        grads = torch.autograd.grad(out, leaves, d_out.double())
        return [out.detach()] + list(grads)

    values = run(x, weight, b, d_out)
    magnitudes = run(x.abs(), weight.abs(), b.abs() if bias else None, d_out.abs())
    tensors = {f"{name}.x": x, f"{name}.weight": weight, f"{name}.d_out": d_out}
    if bias:
        tensors[f"{name}.bias"] = b
    parts = ["out", "grad_x", "grad_weight"] + (["grad_bias"] if bias else [])
    for part, value, magnitude in zip(parts, values, magnitudes):
        tensors[f"{name}.{part}"] = value
        tensors[f"{name}.{part}.magnitude"] = magnitude
    _, _, hout, wout = out_shape
    terms = {
        # The products each element adds up: an output's over its group's input channels and the
        # kernel, the input gradient's at most over the group's outputs and the kernel, the weight's
        # and the bias's over every output position of every batch element.
        "out": cin // groups * kh * kw,
        "grad_x": cout // groups * kh * kw,
        "grad_weight": n * hout * wout,
        "grad_bias": n * hout * wout,
    }
    case = dict(
        stride=list(stride),
        padding=list(padding),
        dilation=list(dilation),
        groups=groups,
        bias=bias,
        terms=terms,
    )
    return tensors, case


def record_conv2d():
    gen = torch.Generator().manual_seed(2026_11_25)
    tensors, cases = {}, {}
    for name, spec in CONV2D_CASES.items():
        t, case = conv2d_case(gen, name, *spec)
        tensors.update(t)
        cases[name] = case
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    metadata = {
        "cases": json.dumps(cases, sort_keys=True),
        "source": f"torch {torch.__version__}, CPU: F.conv2d evaluated in float64 from float32 "
        "inputs, gradients by torch.autograd.grad with the cotangent d_out",
    }
    os.makedirs(os.path.dirname(CONV2D), exist_ok=True)
    save_file(tensors, CONV2D, metadata=metadata)
    print(f"conv2d: {len(cases)} cases to {os.path.relpath(CONV2D, ROOT)}")


# name: shape, training, affine, momentum, eps, per-channel offsets of x
BATCH_NORM_CASES = {
    "train_2d": ((4, 3, 5, 6), True, True, 0.1, 1e-5, [0.0, 3.0, -20.0]),
    "train_2d_plain": ((2, 4, 3, 3), True, False, 0.3, 1e-3, [1.0, -1.0, 0.5, 0.0]),
    "train_1d": ((16, 5), True, True, 0.1, 1e-5, [0.0, 0.0, 2.0, -2.0, 8.0]),
    "train_3d_input": ((3, 2, 7), True, True, 0.05, 1e-5, [0.25, -4.0]),
    "train_two_values": ((2, 3), True, True, 0.1, 1e-5, [0.0, 1.0, -1.0]),
    "eval_2d": ((2, 3, 4, 4), False, True, 0.1, 1e-5, [0.0, 3.0, -20.0]),
    "eval_2d_plain": ((3, 2, 2, 3), False, False, 0.1, 1e-5, [0.5, -0.5]),
}


def per_channel(t, rank):
    return t.reshape([1, -1] + [1] * (rank - 2))


def batch_norm_case(gen, shape, training, affine, momentum, eps, offsets):
    rank, c = len(shape), shape[1]
    x = torch.randn(shape, generator=gen, dtype=torch.float32)
    x = x * (1.0 + torch.rand((c,), generator=gen)).reshape([1, -1] + [1] * (rank - 2))
    x = (x + torch.tensor(offsets).reshape([1, -1] + [1] * (rank - 2))).float()
    weight = (1.0 + 0.5 * torch.randn((c,), generator=gen)).float() if affine else None
    bias = (0.5 * torch.randn((c,), generator=gen)).float() if affine else None
    running_mean = (0.3 * torch.randn((c,), generator=gen)).float()
    running_var = (0.5 + torch.rand((c,), generator=gen)).float()
    d_out = torch.randn(shape, generator=gen, dtype=torch.float32)

    xd = x.double().requires_grad_()
    leaves = [xd]
    wd = bd = None
    if affine:
        wd = weight.double().requires_grad_()
        bd = bias.double().requires_grad_()
        leaves += [wd, bd]
    rm, rv = running_mean.double(), running_var.double()
    out = F.batch_norm(xd, rm, rv, wd, bd, training=training, momentum=momentum, eps=eps)
    grads = torch.autograd.grad(out, leaves, d_out.double())

    # What each element is held to: the magnitudes its roundings scale with (dash-mag's
    # tests/batch_norm.rs says how), from the same inputs in float64.
    axes = [0] + list(range(2, rank))
    m = x.numel() // c
    xa, dya = x.double(), d_out.double().abs()
    gamma = weight.double().abs() if affine else torch.ones(c, dtype=torch.float64)
    beta = bias.double().abs() if affine else torch.zeros(c, dtype=torch.float64)
    g, b = per_channel(gamma, rank), per_channel(beta, rank)
    if training:
        mean = xa.mean(axes, keepdim=True)
        spread = xa.abs().mean(axes, keepdim=True)
        var = ((xa - mean) ** 2).mean(axes, keepdim=True)
        sigma = (var + eps).sqrt()
        centred = (xa - mean).abs() + spread
        normal = centred / sigma
        magnitude = {
            "out": g * normal + b,
            "grad_x": g
            / sigma
            * (
                dya
                + dya.mean(axes, keepdim=True)
                + normal * (dya * normal).mean(axes, keepdim=True)
            ),
            "grad_weight": (dya * normal).sum(axes),
            "grad_bias": dya.sum(axes),
            "running_mean": (1 - momentum) * running_mean.double().abs()
            + momentum * spread.reshape(-1),
            "running_var": (1 - momentum) * running_var.double().abs()
            + momentum * m / (m - 1) * (centred**2).mean(axes).reshape(-1),
        }
        # torch keeps the unbiased batch variance in the running one,
        # and normalises by the biased one.
        unbiased = ((xa - mean) ** 2).sum(axes) / (m - 1)
        want = (1 - momentum) * running_var.double() + momentum * unbiased
        assert torch.allclose(rv, want, rtol=1e-14, atol=0), "running variance not unbiased"
    else:
        sigma = per_channel(running_var.double() + eps, rank).sqrt()
        shifted = (xa.abs() + per_channel(running_mean.double().abs(), rank)) / sigma
        magnitude = {
            "out": g * shifted + b,
            "grad_x": g * dya / sigma,
            "grad_weight": (dya * shifted).sum(axes),
            "grad_bias": dya.sum(axes),
            "running_mean": running_mean.double().abs(),
            "running_var": running_var.double().abs(),
        }

    def flat(t):
        return [float(v) for v in t.detach().reshape(-1)]

    case = dict(
        shape=list(shape),
        training=training,
        momentum=momentum,
        eps=eps,
        affine=affine,
        x=flat(x),
        running_mean=flat(running_mean),
        running_var=flat(running_var),
        d_out=flat(d_out),
        out=flat(out),
        new_running_mean=flat(rm),
        new_running_var=flat(rv),
        grad_x=flat(grads[0]),
        magnitude={k: flat(v) for k, v in magnitude.items()},
    )
    if affine:
        case.update(
            weight=flat(weight),
            bias=flat(bias),
            grad_weight=flat(grads[1]),
            grad_bias=flat(grads[2]),
        )
    return case


def record_batch_norm():
    gen = torch.Generator().manual_seed(2026_11_26)
    cases = {name: batch_norm_case(gen, *spec) for name, spec in BATCH_NORM_CASES.items()}
    record = {
        "what": "F.batch_norm in training and in eval mode: the output, the new running statistics "
        "and the gradients of x, the weight and the bias, with the magnitudes each element's "
        "roundings scale with",
        "source": f"torch {torch.__version__}, CPU: F.batch_norm evaluated in float64 from "
        "float32 inputs, gradients by torch.autograd.grad with the cotangent d_out; in training "
        "the running variance takes the unbiased batch variance, checked here",
        "cases": cases,
    }
    with open(BATCH_NORM, "w") as f:
        json.dump(record, f)
        f.write("\n")
    print(f"batch norm: {len(cases)} cases to {os.path.relpath(BATCH_NORM, ROOT)}")


if __name__ == "__main__":
    record_conv2d()
    record_batch_norm()
