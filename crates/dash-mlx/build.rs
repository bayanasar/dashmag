//! With mlx-sys linking a prebuilt MLX (`MLX_SYS_PREBUILT`, in mlx-sys's README), `libmlx.dylib`
//! is found through an rpath; this adds one, from mlx-sys's `DEP_MLX_LIB_DIR`, to this crate's
//! tests, examples and benchmarks. Without it, or on other targets, it does nothing.

fn main() {
    if let Ok(dir) = std::env::var("DEP_MLX_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
}
