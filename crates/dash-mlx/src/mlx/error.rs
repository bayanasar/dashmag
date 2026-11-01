//! MLX's errors in Rust. mlx-c reports an error through one handler per process and a nonzero
//! status; the handler keeps the message per thread, for the call that gets the status to read.
//! A Rust closure MLX calls cannot unwind through C++: what it returns as an error, or panics
//! with, waits here until control is back in Rust.

use super::sys;
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::panic::Location;
use std::sync::Once;

/// A `Result` with an [`Exception`].
pub type Result<T> = std::result::Result<T, Exception>;

/// An error, most from MLX.
#[derive(Debug, PartialEq)]
pub struct Exception {
    what: String,
    location: &'static Location<'static>,
}

impl Exception {
    /// The error's message.
    pub fn what(&self) -> &str {
        &self.what
    }

    /// An error with this message.
    #[track_caller]
    pub fn custom(what: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            location: Location::caller(),
        }
    }
}

impl std::fmt::Display for Exception {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} at {}", self.what, self.location)
    }
}

impl std::error::Error for Exception {}

enum ClosureFailure {
    Error(Exception),
    Panic(Box<dyn Any + Send>),
}

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    static CLOSURE_FAILURE: Cell<Option<ClosureFailure>> = const { Cell::new(None) };
}

unsafe extern "C" fn handler(message: *const c_char, _: *mut c_void) {
    // SAFETY: mlx-c passes the exception's message, a C string for the length of this call.
    let message = unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned();
    // MLX can report several errors before a caller reads one; the newest stands. During the
    // thread's exit there is nowhere to keep it and no caller to read it.
    let _ = LAST_ERROR.try_with(|last| *last.borrow_mut() = Some(message));
}

/// Registers the one handler, once per process; mlx-c's own exits the process.
pub(crate) fn init() {
    static INIT: Once = Once::new();
    // SAFETY: the handler is a function for the process's life and holds no payload.
    INIT.call_once(|| unsafe {
        sys::mlx_set_error_handler(Some(handler), std::ptr::null_mut(), None);
    });
}

pub(crate) fn set_closure_error(error: Exception) {
    CLOSURE_FAILURE.with(|failure| failure.set(Some(ClosureFailure::Error(error))));
}

pub(crate) fn set_closure_panic(payload: Box<dyn Any + Send>) {
    CLOSURE_FAILURE.with(|failure| failure.set(Some(ClosureFailure::Panic(payload))));
}

/// Unwinds with a panic a closure caught, its payload the original. An error stays for [`check`].
pub(crate) fn resume_closure_panic() {
    CLOSURE_FAILURE.with(|failure| match failure.take() {
        Some(ClosureFailure::Panic(payload)) => {
            // MLX recorded its own error when the closure reported failure; the panic is the cause.
            let _ = LAST_ERROR.with(|last| last.borrow_mut().take());
            std::panic::resume_unwind(payload)
        }
        other => failure.set(other),
    });
}

/// What a call of mlx-c's status says: a panic in a closure it ran unwinds on from here; a failure
/// is the closure's error if one failed, or MLX's.
#[track_caller]
pub(crate) fn check(status: c_int) -> Result<()> {
    resume_closure_panic();
    if status == 0 {
        return Ok(());
    }
    let last = LAST_ERROR.with(|last| last.borrow_mut().take());
    if let Some(ClosureFailure::Error(error)) = CLOSURE_FAILURE.with(Cell::take) {
        return Err(error);
    }
    Err(Exception::custom(last.unwrap_or_else(|| {
        format!("an MLX operation failed with status {status}")
    })))
}
