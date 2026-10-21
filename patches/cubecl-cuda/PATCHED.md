# cubecl-cuda 0.11.1, patched

The crate as published on crates.io (`cubecl-cuda-0.11.1.crate`, sha256
`7091c3152f53c364a140a9f846663561bc2ec1abdee03dcd51d397874293335f`), less
`Cargo.toml.orig`, `.cargo_vcs_info.json` and `Cargo.lock`, with one change in
`src/runtime.rs`:

```diff
         if arch_version >= 80 {
-            device_props
-                .register_type_usage(ElemType::Float(FloatKind::TF32), TypeUsage::Conversion);
+            // dashmag patch: with `CUBECL_CUDA_TF32=0`, TF32 is not registered, so cubek-matmul's
+            // `adjust_dtypes` does not promote f32 matmuls and convolutions to it and they run in
+            // IEEE f32. Any other value, or none, keeps the upstream behaviour.
+            if std::env::var("CUBECL_CUDA_TF32").as_deref() != Ok("0") {
+                device_props
+                    .register_type_usage(ElemType::Float(FloatKind::TF32), TypeUsage::Conversion);
+            }
             device_props.features.copy_async = true;
         }
```

## Why

On compute capability 8.0 and later (Ampere, Ada, Hopper, Blackwell, Jetson Orin),
cubecl-cuda registers TF32, and cubek-matmul's `adjust_dtypes` then gives every f32 matmul
and convolution TF32 inputs on the tensor cores. TF32 keeps 10 mantissa bits, so each input
is rounded with a relative error of up to 2^-11 (4.9e-4), against 2^-24 for f32; the checks
in this repository compare logits across backends to 1e-4. Burn 0.22.1 has no setting that
turns the promotion off.

## Use

Set `CUBECL_CUDA_TF32=0` in the environment of the process. It is read when a CUDA device
is first initialised, so it holds for the whole process. Unset, nothing changes.

With TF32 unregistered, the tensor-core kernels find no f32 instruction and report
themselves unavailable, and kernel selection falls back to kernels without tensor cores.
Expect f32 matmuls to be slower; the upstream report below measured 1.9-3.0x.

## When it goes

Upstream knows: https://github.com/tracel-ai/burn/issues/6140, where the maintainers
intend a setting in cubek's runtime config. Once a Burn release has it, upgrade to that
release, use its setting, and delete this directory and the `[patch.crates-io]` entry in
the workspace `Cargo.toml`.

## Checking this copy

```sh
curl -sL https://static.crates.io/crates/cubecl-cuda/cubecl-cuda-0.11.1.crate | tar xz
diff -r cubecl-cuda-0.11.1 patches/cubecl-cuda
```

prints the hunk above, this file, and the three files left out.
