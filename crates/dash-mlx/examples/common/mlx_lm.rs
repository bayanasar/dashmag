//! mlx-lm's gated delta kernel, the baseline the delta rule's benchmarks are held to: its Metal
//! source as mlx-lm 0.31.3 publishes it (`_make_gated_delta_kernel`, without a mask or
//! vectorised gates), run through MLX's `metal_kernel` with mlx-lm's grid and threadgroup. mlx-lm
//! runs it for decoding and for a prompt's prefill; it has no backward pass. Its layouts are
//! mlx-lm's: queries and keys `[B, T, Hk, Dk]`, values `[B, T, Hv, Dv]`, the decay `exp(g)` (not
//! its log) and `beta` `[B, T, Hv]`, the state `[B, Hv, Dv, Dk]`.
//!
//! The source is used under its license:
//!
//! MIT License
//!
//! Copyright © 2023 Apple Inc.
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy of this software
//! and associated documentation files (the "Software"), to deal in the Software without
//! restriction, including without limitation the rights to use, copy, modify, merge, publish,
//! distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
//! Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all copies or
//! substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
//! BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//! NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//! DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use mlx_rs::error::Result;
use mlx_rs::fast::{
    MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg,
};
use mlx_rs::{Array, Dtype};
use std::cell::OnceCell;

const SOURCE: &str = "
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (true) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          } else {
            y[dv_idx] = static_cast<InT>(0);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
";

thread_local! {
    static KERNEL: OnceCell<MetalKernel> = const { OnceCell::new() };
}

/// mlx-lm's `gated_delta_kernel`: the outputs `[B, T, Hv, Dv]` and the new state `[B, Hv, Dv,
/// Dk]`. `decay` is `exp(g)`.
pub fn gated_delta_kernel(
    q: &Array,
    k: &Array,
    v: &Array,
    decay: &Array,
    beta: &Array,
    state: &Array,
) -> Result<Vec<Array>> {
    let [b, t, hk, dk] = [q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]];
    let [hv, dv] = [v.shape()[2], v.shape()[3]];
    let outputs = [
        MetalKernelOutput {
            shape: vec![b, t, hv, dv],
            dtype: q.dtype(),
        },
        MetalKernelOutput {
            shape: state.shape().to_vec(),
            dtype: state.dtype(),
        },
    ];
    let options = MetalKernelApplyOptions {
        template_args: vec![
            ("InT".into(), TemplateArg::Dtype(q.dtype())),
            ("StT".into(), TemplateArg::Dtype(state.dtype())),
            ("Dk".into(), TemplateArg::Int(dk)),
            ("Dv".into(), TemplateArg::Int(dv)),
            ("Hk".into(), TemplateArg::Int(hk)),
            ("Hv".into(), TemplateArg::Int(hv)),
        ],
        ..Default::default()
    };
    assert!(
        dk % 32 == 0 && hv % hk == 0,
        "mlx-lm's kernel takes Dk a multiple of 32 and Hv a multiple of Hk"
    );
    let inputs = [
        q.clone(),
        k.clone(),
        v.clone(),
        decay.clone(),
        beta.clone(),
        state.clone(),
        Array::from_int(t),
    ];
    KERNEL.with(|kernel| {
        let kernel = match kernel.get() {
            Some(kernel) => kernel,
            None => {
                let built = MetalKernel::new(
                    "gated_delta_step",
                    &["q", "k", "v", "g", "beta", "state_in", "T"],
                    &["y", "state_out"],
                    SOURCE,
                    MetalKernelOptions {
                        ensure_row_contiguous: true,
                        ..Default::default()
                    },
                )?;
                let _ = kernel.set(built);
                kernel.get().expect("just set")
            }
        };
        // SAFETY: with Dk a multiple of 32 and Hv of Hk (asserted), thread (lane, column, b Hv + h)
        // reads its Dk / 32 rows of state column `column` of head h and of each token's query and
        // key, and writes them back to the new state; lane 0 writes each token's output at that
        // column. Every output element is written once.
        unsafe { kernel.apply(&inputs, &outputs, (32, dv, b * hv), (32, 4, 1), options) }
    })
}

/// mlx-lm's GatedDeltaNet's preparation of the kernel's inputs from the model's: queries and
/// keys normalised by MLX's `rms_norm` and scaled (the queries by `1 / Dk`, the keys by
/// `1 / sqrt(Dk)`, which leaves them at unit length and the queries at `1 / sqrt(Dk)` of it), and
/// the decay `exp(g)` from its log.
pub fn prepare(q: &Array, k: &Array, g: &Array) -> Result<[Array; 3]> {
    let dk = *q.shape().last().expect("a last axis") as f32;
    let inv_scale = 1.0 / dk.sqrt();
    // Python's scalars take the array's dtype; an f32 array here would promote bf16 to f32.
    let scalar = |x: f32| Array::from_f32(x).as_dtype(q.dtype());
    let q = mlx_rs::fast::rms_norm(q, None, 1e-6)?.multiply(scalar(inv_scale * inv_scale)?)?;
    let k = mlx_rs::fast::rms_norm(k, None, 1e-6)?.multiply(scalar(inv_scale)?)?;
    Ok([q, k, g.exp()?.as_dtype(Dtype::Float32)?])
}
