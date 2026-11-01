//! The C API of mlx-c, as `build.rs` generates it from the headers in `mlx-c/`.

#![allow(
    clippy::all,
    missing_docs,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unsafe_op_in_unsafe_fn
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
