//! MLX's function transforms over Rust closures: evaluation, the vector-Jacobian product, a function
//! with its own backward pass, and a function whose intermediates are computed again in its
//! backward pass.
//!
//! A closure MLX calls must not unwind through C++, so each trampoline catches a panic and returns
//! a failure; `check` then unwinds with the panic, or returns the closure's error, back in Rust.
//! A closure that MLX reaches again while it is running gets an error, not a second `&mut`.

use super::array::{Array, new_array};
use super::error::{self, Exception, Result, check, set_closure_error, set_closure_panic};
use super::sys;
use std::cell::RefCell;
use std::ffi::{c_int, c_void};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};

/// mlx-c's vector of arrays, owned here.
pub(crate) struct VectorArray(sys::mlx_vector_array);

impl VectorArray {
    pub(crate) fn empty() -> Self {
        error::init();
        // SAFETY: makes a new handle.
        Self(unsafe { sys::mlx_vector_array_new() })
    }

    /// A vector holding its own references to these arrays.
    pub(crate) fn from_arrays<A: AsRef<Array>>(
        arrays: impl IntoIterator<Item = A>,
    ) -> Result<Self> {
        let vector = Self::empty();
        for array in arrays {
            // SAFETY: valid handles, for the length of the call.
            check(unsafe {
                sys::mlx_vector_array_append_value(vector.0, array.as_ref().as_ptr())
            })?;
        }
        Ok(vector)
    }

    pub(crate) fn as_ptr(&self) -> sys::mlx_vector_array {
        self.0
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut sys::mlx_vector_array {
        &mut self.0
    }

    pub(crate) fn to_arrays(&self) -> Result<Vec<Array>> {
        arrays(self.0)
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        // SAFETY: this handle owns its reference, and is freed once.
        unsafe { sys::mlx_vector_array_free(self.0) };
    }
}

fn arrays(vector: sys::mlx_vector_array) -> Result<Vec<Array>> {
    // SAFETY: a valid handle, read within its bounds.
    let size = unsafe { sys::mlx_vector_array_size(vector) };
    (0..size)
        .map(|i| new_array(|res| unsafe { sys::mlx_vector_array_get(res, vector, i) }))
        .collect()
}

/// Fills the handle mlx-c hands a closure, which it frees after reading: in place, not replaced.
///
/// # Safety
///
/// `ret` is that handle.
unsafe fn fill(ret: *mut sys::mlx_vector_array, values: &[Array]) -> c_int {
    let handles: Vec<sys::mlx_array> = values.iter().map(Array::as_ptr).collect();
    // SAFETY: the caller's handle; valid handles, for the length of the call.
    unsafe { sys::mlx_vector_array_set_data(ret, handles.as_ptr(), handles.len()) }
}

/// Runs a trampoline's body: what it panics with waits for [`check`], and so does its error.
fn guarded(body: impl FnOnce() -> Result<c_int>) -> c_int {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            set_closure_error(error);
            1
        }
        Err(payload) => {
            set_closure_panic(payload);
            1
        }
    }
}

unsafe extern "C" fn drop_payload<F>(payload: *mut c_void) {
    // MLX frees a payload when the last array or function holding it goes, wherever that is, so a
    // panic in its destructor has no call to reach, and left waiting it would unwind out of an
    // unrelated one. The panic hook has reported it; it stops here.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the box `Closure::new` or `custom_vjp` made, which MLX frees once.
        drop(unsafe { Box::from_raw(payload as *mut F) });
    }));
}

/// A payload to call, or an error if it is running already.
fn borrow_payload<'a, F>(payload: &'a RefCell<F>) -> Result<std::cell::RefMut<'a, F>> {
    payload
        .try_borrow_mut()
        .map_err(|_| Exception::custom("a function given to MLX called itself"))
}

unsafe extern "C" fn trampoline<F>(
    ret: *mut sys::mlx_vector_array,
    inputs: sys::mlx_vector_array,
    payload: *mut c_void,
) -> c_int
where
    F: FnMut(&[Array]) -> Result<Vec<Array>>,
{
    guarded(|| {
        // SAFETY: the payload `Closure::new` boxed, alive while MLX can call this, and only ever
        // shared: the cell is what hands out the function.
        let mut f = borrow_payload(unsafe { &*(payload as *const RefCell<F>) })?;
        let outputs = (*f)(&arrays(inputs)?)?;
        // SAFETY: mlx-c's handle for the result.
        Ok(unsafe { fill(ret, &outputs) })
    })
}

unsafe extern "C" fn trampoline_custom<G>(
    ret: *mut sys::mlx_vector_array,
    primals: sys::mlx_vector_array,
    cotangents: sys::mlx_vector_array,
    outputs: sys::mlx_vector_array,
    payload: *mut c_void,
) -> c_int
where
    G: FnMut(&[Array], &[Array], &[Array]) -> Result<Vec<Array>>,
{
    guarded(|| {
        // SAFETY: the payload `custom_vjp` boxed, alive while MLX can call this, and only ever
        // shared: the cell is what hands out the function.
        let mut vjp = borrow_payload(unsafe { &*(payload as *const RefCell<G>) })?;
        let primals = arrays(primals)?;
        let grads = (*vjp)(&primals, &arrays(cotangents)?, &arrays(outputs)?)?;
        // MLX picks each gradient by its primal's position, past the end if there are too few.
        if grads.len() != primals.len() {
            return Err(Exception::custom(format!(
                "custom_vjp: the backward pass returned {} gradients for {} primals",
                grads.len(),
                primals.len()
            )));
        }
        // SAFETY: mlx-c's handle for the result.
        Ok(unsafe { fill(ret, &grads) })
    })
}

/// A function MLX calls back, alive for `'a`.
pub(crate) struct Closure<'a> {
    raw: sys::mlx_closure,
    lifetime: PhantomData<&'a ()>,
}

impl<'a> Closure<'a> {
    pub(crate) fn new<F>(f: F) -> Self
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
    {
        error::init();
        let payload = Box::into_raw(Box::new(RefCell::new(f))) as *mut c_void;
        // SAFETY: MLX keeps the payload until the last copy of the closure goes, then calls the
        // destructor; `'a` keeps what `F` borrows alive for this handle, and a transform given it
        // does not keep it past its own return unless `F` is `'static`.
        let raw = unsafe {
            sys::mlx_closure_new_func_payload(
                Some(trampoline::<F>),
                payload,
                Some(drop_payload::<RefCell<F>>),
            )
        };
        Self {
            raw,
            lifetime: PhantomData,
        }
    }

    pub(crate) fn as_ptr(&self) -> sys::mlx_closure {
        self.raw
    }

    /// Calls it on `inputs`.
    pub(crate) fn apply(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        let inputs = VectorArray::from_arrays(inputs)?;
        let mut outputs = VectorArray::empty();
        // SAFETY: valid handles, for the length of the call.
        check(unsafe { sys::mlx_closure_apply(outputs.as_mut_ptr(), self.raw, inputs.as_ptr()) })?;
        outputs.to_arrays()
    }
}

impl Drop for Closure<'_> {
    fn drop(&mut self) {
        // SAFETY: this handle owns its reference, and is freed once.
        unsafe { sys::mlx_closure_free(self.raw) };
    }
}

impl std::fmt::Debug for Closure<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Closure").finish_non_exhaustive()
    }
}

/// Evaluates the arrays, together.
pub fn eval<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    let outputs = VectorArray::from_arrays(outputs)?;
    // SAFETY: a valid handle, for the length of the call.
    check(unsafe { sys::mlx_eval(outputs.as_ptr()) })
}

/// `f`'s outputs at `primals`, and the product of `cotangents` with its Jacobian: one gradient per
/// primal. An error or a panic in `f` reaches the caller.
pub fn fallible_vjp<'a, F>(
    f: F,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new(f);
    let primals = VectorArray::from_arrays(primals)?;
    let cotangents = VectorArray::from_arrays(cotangents)?;
    let (mut outputs, mut grads) = (VectorArray::empty(), VectorArray::empty());
    // SAFETY: valid handles, for the length of the call.
    check(unsafe {
        sys::mlx_vjp(
            outputs.as_mut_ptr(),
            grads.as_mut_ptr(),
            closure.as_ptr(),
            primals.as_ptr(),
            cotangents.as_ptr(),
        )
    })?;
    Ok((outputs.to_arrays()?, grads.to_arrays()?))
}

/// `f`'s outputs on `inputs`, and the gradient of its first output, a scalar, to each input.
pub(crate) fn value_and_grad_all<'a, F>(f: F, inputs: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new(f);
    let argnums: Vec<c_int> = (0..inputs.len() as c_int).collect();
    // SAFETY: makes a new handle, which `ValueAndGrad` frees.
    let mut transformed = ValueAndGrad(unsafe { sys::mlx_closure_value_and_grad_new() });
    // SAFETY: valid handles and `argnums`' elements, for the length of the call.
    check(unsafe {
        sys::mlx_value_and_grad(
            &mut transformed.0,
            closure.as_ptr(),
            argnums.as_ptr(),
            argnums.len(),
        )
    })?;
    let inputs = VectorArray::from_arrays(inputs)?;
    let (mut values, mut grads) = (VectorArray::empty(), VectorArray::empty());
    // SAFETY: valid handles, for the length of the call.
    check(unsafe {
        sys::mlx_closure_value_and_grad_apply(
            values.as_mut_ptr(),
            grads.as_mut_ptr(),
            transformed.0,
            inputs.as_ptr(),
        )
    })?;
    Ok((values.to_arrays()?, grads.to_arrays()?))
}

/// mlx-c's closure for a value and its gradients, owned here.
struct ValueAndGrad(sys::mlx_closure_value_and_grad);

impl Drop for ValueAndGrad {
    fn drop(&mut self) {
        // SAFETY: this handle owns its reference, and is freed once.
        unsafe { sys::mlx_closure_value_and_grad_free(self.0) };
    }
}

/// A compiled function's id, whose tape leaves MLX's cache when this goes.
struct CompiledTape(usize);

impl Drop for CompiledTape {
    fn drop(&mut self) {
        // SAFETY: the cache handle is new, filled here, and freed once.
        unsafe {
            let mut cache = sys::mlx_compile_cache_new();
            if sys::mlx_detail_compile_cache(&mut cache) == 0 {
                sys::mlx_detail_compile_erase(cache, self.0);
            }
            sys::mlx_compile_cache_free(cache);
        }
    }
}

/// `f` on `inputs` once, compiled for inputs of any shape: traced, simplified, and run as one
/// graph, its tape dropped from MLX's cache afterwards.
pub(crate) fn compile_shapeless_once<F>(f: F, inputs: &[Array]) -> Result<Vec<Array>>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>>,
{
    static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
    let tape = CompiledTape(NEXT_ID.fetch_add(1, Ordering::Relaxed));
    let closure = Closure::new(f);
    // SAFETY: makes a new handle, which `Closure` frees.
    let mut raw = unsafe { sys::mlx_closure_new() };
    // SAFETY: valid handles, for the length of the call; no constants.
    let status = unsafe {
        sys::mlx_detail_compile(
            &mut raw,
            closure.as_ptr(),
            tape.0,
            true,
            std::ptr::null(),
            0,
        )
    };
    let compiled = Closure {
        raw,
        lifetime: PhantomData::<&()>,
    };
    check(status)?;
    compiled.apply(inputs)
}

/// A function MLX holds, callable and differentiable.
#[derive(Debug)]
pub struct Function {
    closure: Closure<'static>,
}

impl Function {
    /// Calls it. An error or a panic in the function, or in its backward pass if a transform is
    /// differentiating it, reaches the caller.
    pub fn apply(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        self.closure.apply(inputs)
    }
}

/// `f`, refusing to return nothing: MLX reads a checkpointed or custom-VJP function's first output.
fn with_outputs<F>(mut f: F) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'static
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'static,
{
    move |inputs| {
        let outputs = f(inputs)?;
        if outputs.is_empty() {
            return Err(Exception::custom(
                "a function given to MLX returned no outputs",
            ));
        }
        Ok(outputs)
    }
}

/// `f`, whose intermediates are not kept: its backward pass runs it again from its inputs.
pub fn checkpoint<F>(f: F) -> Result<Function>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'static,
{
    let f = Closure::new(with_outputs(f));
    // SAFETY: makes a new handle, which `Closure` frees.
    let mut raw = unsafe { sys::mlx_closure_new() };
    // SAFETY: valid handles, for the length of the call.
    let status = unsafe { sys::mlx_checkpoint(&mut raw, f.as_ptr()) };
    let closure = Closure {
        raw,
        lifetime: PhantomData,
    };
    check(status)?;
    Ok(Function { closure })
}

/// `forward`, with `vjp` as its backward pass in place of its derivative. `vjp` takes the primals,
/// the cotangents and the outputs, and returns one gradient per primal.
pub fn custom_vjp<F, G>(forward: F, vjp: G) -> Result<Function>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'static,
    G: FnMut(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + 'static,
{
    let forward = Closure::new(with_outputs(forward));
    let payload = Box::into_raw(Box::new(RefCell::new(vjp))) as *mut c_void;
    // SAFETY: MLX keeps the payload until the last copy goes, then calls the destructor.
    let backward = unsafe {
        sys::mlx_closure_custom_new_func_payload(
            Some(trampoline_custom::<G>),
            payload,
            Some(drop_payload::<RefCell<G>>),
        )
    };
    // SAFETY: makes a new handle, which `Closure` frees.
    let mut raw = unsafe { sys::mlx_closure_new() };
    // SAFETY: valid handles, for the length of the call; the result holds its own copies.
    let status = unsafe { sys::mlx_custom_vjp(&mut raw, forward.as_ptr(), backward) };
    // SAFETY: this handle owns its reference, and is freed once.
    unsafe { sys::mlx_closure_custom_free(backward) };
    let closure = Closure {
        raw,
        lifetime: PhantomData,
    };
    check(status)?;
    Ok(Function { closure })
}

#[cfg(test)]
mod tests {
    use super::{Function, checkpoint, custom_vjp, fallible_vjp, value_and_grad_all};
    use crate::mlx::Array;
    use crate::mlx::error::Result;
    use std::cell::RefCell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::rc::Rc;

    #[test]
    fn a_backward_pass_with_too_few_gradients_is_an_error() {
        let f = custom_vjp(
            |x: &[Array]| Ok(vec![x[0].multiply(&x[1])?]),
            |_: &[Array], c: &[Array], _: &[Array]| Ok(vec![c[0].clone()]),
        )
        .unwrap();
        let error = fallible_vjp(
            |x: &[Array]| f.apply(x),
            &[Array::from_f32(2.0), Array::from_f32(3.0)],
            &[Array::from_f32(1.0)],
        )
        .unwrap_err();
        assert!(
            error.what().contains("1 gradients for 2 primals"),
            "{error}"
        );
    }

    #[test]
    fn a_function_with_no_outputs_is_an_error() {
        let x = [Array::from_f32(1.0)];
        let nothing = |_: &[Array]| -> Result<Vec<Array>> { Ok(vec![]) };
        assert!(checkpoint(nothing).unwrap().apply(&x).is_err());
        let backward = |_: &[Array], _: &[Array], _: &[Array]| -> Result<Vec<Array>> { Ok(vec![]) };
        assert!(custom_vjp(nothing, backward).unwrap().apply(&x).is_err());
    }

    #[test]
    fn a_function_calling_itself_is_an_error() {
        let slot: Rc<RefCell<Option<Function>>> = Rc::new(RefCell::new(None));
        let inner = slot.clone();
        let f = checkpoint(move |x: &[Array]| match inner.borrow().as_ref() {
            Some(f) => f.apply(x),
            None => Ok(x.to_vec()),
        })
        .unwrap();
        *slot.borrow_mut() = Some(f);
        let result = slot
            .borrow()
            .as_ref()
            .unwrap()
            .apply(&[Array::from_f32(1.0)]);
        assert!(result.unwrap_err().what().contains("called itself"));
        // The function holds the slot and the slot the function.
        slot.borrow_mut().take();
    }

    #[test]
    fn a_panicking_destructor_stays_out_of_the_next_call() {
        struct Panics;
        impl Drop for Panics {
            fn drop(&mut self) {
                panic!("in the payload's destructor");
            }
        }
        let held = Panics;
        let f = checkpoint(move |x: &[Array]| {
            let _held = &held;
            Ok(x.to_vec())
        })
        .unwrap();
        drop(f);
        let sum = Array::from_f32(1.0).add(Array::from_f32(2.0)).unwrap();
        assert_eq!(sum.as_slice::<f32>(), [3.0]);
    }

    #[test]
    fn a_panic_under_value_and_grad_reaches_the_caller_and_frees_the_closure() {
        let held = Rc::new(());
        let in_closure = held.clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            value_and_grad_all(
                move |_: &[Array]| -> Result<Vec<Array>> {
                    let _held = &in_closure;
                    panic!("in the closure")
                },
                &[Array::from_f32(1.0)],
            )
        }));
        assert!(result.is_err());
        assert_eq!(Rc::strong_count(&held), 1);
    }
}
