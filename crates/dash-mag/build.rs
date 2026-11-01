//! On macOS, with a prebuilt MLX (`MLX_SYS_PREBUILT`, in dash-mlx's README), `libmlx.dylib` is found
//! through an rpath; this adds one, from dash-mlx's `DEP_MLX_LIB_DIR`, to this crate's tests, examples
//! and benchmarks. It also publishes the directory, which reaches only a direct dependent: an
//! application naming this crate reads it as `DEP_DASH_MAG_LIB_DIR` to add the rpath to its own binaries.
//! Without it, or on other targets, it does nothing.

fn main() {
    if let Ok(dir) = std::env::var("DEP_MLX_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        println!("cargo::metadata=lib_dir={dir}");
    }
}
