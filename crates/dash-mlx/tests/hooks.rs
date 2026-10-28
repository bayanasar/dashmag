//! The three MLX functions the operations build on, `checkpoint` and `custom_vjp` from the
//! workspace's mlx-rs (bayanasar/mlx-rs's `dashmag` branch) and `MetalKernel` from mlx-rs 0.33.0,
//! each held to plain MLX operations: `checkpoint` gives the function's own outputs and gradients
//! and calls it again in the backward pass; `custom_vjp` runs the backward pass it is given, and
//! reports its errors and panics; `MetalKernel` computes what the operations it replaces do.
//!
//!   cargo test -p dash-mlx --test hooks
#![cfg(target_os = "macos")]

use mlx_rs::error::{Exception, Result};
use mlx_rs::fast::{
    MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg,
};
use mlx_rs::transforms::{checkpoint, custom_vjp, eval, fallible_vjp};
use mlx_rs::{Array, Dtype};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

/// One test at a time: the tests share the GPU's default stream and MLX's caches.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn normal(shape: &[i32], seed: u64) -> Array {
    let key = mlx_rs::random::key(seed).unwrap();
    mlx_rs::random::normal::<f32>(shape, None, None, &key).unwrap()
}

fn values(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// The largest absolute difference between two arrays of one shape.
fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    assert_eq!(a.shape(), b.shape());
    values(a)
        .iter()
        .zip(values(b))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// One layer of a stack: widen, sine, narrow.
fn layer(args: &[Array]) -> Result<Vec<Array>> {
    let (x, up, down) = (&args[0], &args[1], &args[2]);
    Ok(vec![x.matmul(up)?.sin()?.matmul(down)?])
}

#[test]
fn checkpoint_gives_the_function_outputs_and_gradients() -> Result<()> {
    let _serial = serial();
    let inputs = [
        normal(&[64, 32], 0),
        normal(&[32, 128], 1),
        normal(&[128, 32], 2),
    ];
    let cotangent = normal(&[64, 32], 3);
    let layer_checkpointed = checkpoint(layer)?;
    let (out, grads) = fallible_vjp(layer, &inputs, std::slice::from_ref(&cotangent))?;
    let (out_ckpt, grads_ckpt) = fallible_vjp(
        |args: &[Array]| layer_checkpointed.apply(args),
        &inputs,
        &[cotangent],
    )?;
    // The backward pass computes the same operations on the same inputs again: the same bits.
    assert_eq!(values(&out[0]), values(&out_ckpt[0]));
    for (g, g_ckpt) in grads.iter().zip(&grads_ckpt) {
        assert_eq!(values(g), values(g_ckpt));
    }
    Ok(())
}

/// What `checkpoint` changes: the backward pass calls the function again, from its inputs,
/// instead of reading what the forward pass computed. How much memory that saves is up to the
/// order MLX evaluates the graph in, not to the wrapper: on a stack of eight layers sharing their
/// weights, MLX 0.32.3's own checkpoint peaked at 1140 MiB against 1133 MiB without it
/// (measured), so a recompute that bounds memory evaluates one layer at a time.
#[test]
fn checkpoint_calls_the_function_again_in_the_backward_pass() -> Result<()> {
    let _serial = serial();
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let counted = checkpoint(|args: &[Array]| {
        CALLS.fetch_add(1, Ordering::SeqCst);
        layer(args)
    })?;
    let inputs = [normal(&[8, 4], 0), normal(&[4, 16], 1), normal(&[16, 4], 2)];
    let before = CALLS.load(Ordering::SeqCst);
    eval(counted.apply(&inputs)?.iter())?;
    assert_eq!(CALLS.load(Ordering::SeqCst) - before, 1, "the forward pass");
    let (_, grads) = fallible_vjp(
        |args: &[Array]| counted.apply(args),
        &inputs,
        &[normal(&[8, 4], 3)],
    )?;
    eval(grads.iter())?;
    assert_eq!(
        CALLS.load(Ordering::SeqCst) - before,
        3,
        "a second forward pass, and the backward pass's own"
    );
    Ok(())
}

/// `x * y`, with a backward pass that is not its derivative, so that the test can tell it ran:
/// `3 c` for `x`, and `c x + x y` for `y`, which reads the primals and the outputs it is handed.
fn product_with_a_made_up_backward() -> Result<mlx_rs::transforms::Function> {
    custom_vjp(
        |args: &[Array]| Ok(vec![args[0].multiply(&args[1])?]),
        |primals: &[Array], cotangents: &[Array], outputs: &[Array]| {
            Ok(vec![
                cotangents[0].multiply(Array::from_f32(3.0))?,
                cotangents[0].multiply(&primals[0])?.add(&outputs[0])?,
            ])
        },
    )
}

#[test]
fn custom_vjp_runs_the_backward_pass_it_is_given() -> Result<()> {
    let _serial = serial();
    let (x, y, c) = (normal(&[4, 5], 0), normal(&[4, 5], 1), normal(&[4, 5], 2));
    let product = product_with_a_made_up_backward()?;
    let (out, grads) = fallible_vjp(
        |args: &[Array]| product.apply(args),
        &[x.clone(), y.clone()],
        std::slice::from_ref(&c),
    )?;
    let xy = x.multiply(&y)?;
    assert_eq!(values(&out[0]), values(&xy));
    assert_eq!(
        values(&grads[0]),
        values(&c.multiply(Array::from_f32(3.0))?)
    );
    assert_eq!(values(&grads[1]), values(&c.multiply(&x)?.add(&xy)?));
    Ok(())
}

#[test]
fn custom_vjp_reports_an_error_from_the_backward_pass() -> Result<()> {
    let _serial = serial();
    let doubled = custom_vjp(
        |args: &[Array]| Ok(vec![args[0].multiply(Array::from_f32(2.0))?]),
        |_: &[Array], _: &[Array], _: &[Array]| Err(Exception::custom("no backward pass here")),
    )?;
    let err = fallible_vjp(
        |args: &[Array]| doubled.apply(args),
        &[normal(&[3], 0)],
        &[normal(&[3], 1)],
    )
    .unwrap_err();
    assert!(
        err.what().contains("no backward pass here"),
        "{}",
        err.what()
    );
    Ok(())
}

#[test]
fn custom_vjp_resumes_a_panic_from_the_backward_pass() -> Result<()> {
    let _serial = serial();
    let doubled = custom_vjp(
        |args: &[Array]| Ok(vec![args[0].multiply(Array::from_f32(2.0))?]),
        |_: &[Array], _: &[Array], _: &[Array]| panic!("the backward pass panicked"),
    )?;
    let (x, c) = (normal(&[3], 0), normal(&[3], 1));
    let caught = catch_unwind(AssertUnwindSafe(|| {
        fallible_vjp(|args: &[Array]| doubled.apply(args), &[x], &[c])
    }));
    let payload = caught.expect_err("the panic did not reach the caller");
    assert_eq!(
        payload.downcast_ref::<&str>(),
        Some(&"the backward pass panicked")
    );
    Ok(())
}

/// `out = A x + y`, elementwise, with the type and `A` as template arguments.
fn axpy_kernel() -> Result<MetalKernel> {
    MetalKernel::new(
        "dash_mlx_test_axpy",
        &["x", "y"],
        &["out"],
        "uint i = thread_position_in_grid.x;\nout[i] = T(A) * x[i] + y[i];\n",
        MetalKernelOptions {
            header: String::new(),
            ensure_row_contiguous: true,
            atomic_outputs: false,
        },
    )
}

/// Runs [`axpy_kernel`] over `x` and `y` of `n` elements.
fn axpy(kernel: &MetalKernel, x: &Array, y: &Array, n: i32) -> Result<Vec<Array>> {
    let dtype = x.dtype();
    let options = MetalKernelApplyOptions {
        template_args: vec![
            ("T".into(), TemplateArg::Dtype(dtype)),
            ("A".into(), TemplateArg::Int(3)),
        ],
        ..Default::default()
    };
    let outputs = [MetalKernelOutput {
        shape: vec![n],
        dtype,
    }];
    // SAFETY: one thread per element of `n`-element inputs and output, each writing its own
    // element of the output; the grid is exactly `n` threads.
    unsafe { kernel.apply(&[x, y], &outputs, (n, 1, 1), (256, 1, 1), options) }
}

#[test]
fn metal_kernel_computes_what_the_operations_do() -> Result<()> {
    let _serial = serial();
    let n = 1000;
    let kernel = axpy_kernel()?;
    for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
        let x = normal(&[n], 0).as_dtype(dtype)?;
        let y = normal(&[n], 1).as_dtype(dtype)?;
        let out = axpy(&kernel, &x, &y, n)?;
        let expected = x.multiply(Array::from_f32(3.0).as_dtype(dtype)?)?.add(&y)?;
        assert_eq!(out[0].dtype(), dtype);
        let diff = max_abs_diff(&out[0], &expected);
        println!("{dtype:?}: max abs difference {diff:e}");
        // Within one rounding of values below 16: the kernel's multiply-add may be fused into one
        // rounding where the operations round twice. f32 measured 9.5e-7, one ulp at 8 to 16.
        let bound = if dtype == Dtype::Float32 { 2e-6 } else { 0.07 };
        assert!(diff <= bound, "{dtype:?}: {diff}");
    }
    Ok(())
}

#[test]
fn metal_kernel_fills_its_outputs_with_the_init_value() -> Result<()> {
    let _serial = serial();
    let kernel = MetalKernel::new(
        "dash_mlx_test_even",
        &["x"],
        &["out"],
        "uint i = thread_position_in_grid.x;\nif (i % 2 == 0) { out[i] = x[i]; }\n",
        MetalKernelOptions {
            header: String::new(),
            ensure_row_contiguous: true,
            atomic_outputs: false,
        },
    )?;
    let x = normal(&[64], 0);
    let outputs = [MetalKernelOutput {
        shape: vec![64],
        dtype: Dtype::Float32,
    }];
    let options = MetalKernelApplyOptions {
        init_value: Some(-1.0),
        ..Default::default()
    };
    // SAFETY: 64 threads over 64-element f32 buffers, each touching only its own element; the
    // elements it leaves unwritten hold the init value.
    let out = unsafe { kernel.apply(&[&x], &outputs, (64, 1, 1), (32, 1, 1), options)? };
    let out = values(&out[0]);
    for (i, (o, x)) in out.iter().zip(values(&x)).enumerate() {
        assert_eq!(*o, if i % 2 == 0 { x } else { -1.0 }, "element {i}");
    }
    Ok(())
}

#[test]
fn metal_kernel_reports_a_source_that_does_not_compile() -> Result<()> {
    let _serial = serial();
    let kernel = MetalKernel::new(
        "dash_mlx_test_broken",
        &["x"],
        &["out"],
        "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + undeclared;\n",
        MetalKernelOptions {
            header: String::new(),
            ensure_row_contiguous: true,
            atomic_outputs: false,
        },
    )?;
    let outputs = [MetalKernelOutput {
        shape: vec![8],
        dtype: Dtype::Float32,
    }];
    // MLX compiles the kernel when the output is evaluated, and this one does not compile.
    // SAFETY: the kernel never runs; were it to, 8 threads would each write their own element.
    let out = unsafe {
        kernel.apply(
            &[normal(&[8], 0)],
            &outputs,
            (8, 1, 1),
            (8, 1, 1),
            MetalKernelApplyOptions::default(),
        )?
    };
    let err = out[0].eval().unwrap_err();
    assert!(err.what().contains("undeclared"), "{}", err.what());
    Ok(())
}
