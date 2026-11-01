//! Builds Apple's mlx-c (`mlx-c/`, see its `VENDORED.md`) against MLX and generates the Rust
//! bindings from its headers. MLX is either a prebuilt copy (`MLX_SYS_PREBUILT`, the prefix of an
//! installed MLX) or built from source, which mlx-c's CMake configuration fetches and which
//! compiles the Metal kernels ahead of time with Xcode's Metal compiler.
//!
//! On other targets the crate is empty and this does nothing.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // `target_os` here is the machine building; the variable is the machine built for.
    #[cfg(target_os = "macos")]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        macos::build();
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use cmake::Config;
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    /// The commit of mlx-c in `mlx-c/`.
    const MLX_C_COMMIT: &str = "a26f2d584a14ec619ae8db9a09be03becd47598a";
    /// The MLX version that commit is written against, which its CMake configuration pins.
    const MLX_VERSION: &str = "0.32.4";

    pub fn build() {
        let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("mlx-c");
        println!("cargo:rerun-if-changed={}", root.display());
        println!("cargo:rerun-if-env-changed=MLX_SYS_PREBUILT");
        println!("cargo:rerun-if-env-changed=MLX_METAL_PATH");

        let mut config = Config::new(&root);
        config.define("CMAKE_INSTALL_PREFIX", ".");
        // Xcode's or the Command Line Tools' clang, for compatibility with the macOS SDK.
        config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
        config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");
        config.define(
            "CMAKE_BUILD_TYPE",
            if cfg!(debug_assertions) {
                "Debug"
            } else {
                "Release"
            },
        );
        config.define("MLX_C_BUILD_EXAMPLES", "OFF");

        // A build directory for each way of getting MLX: CMake's cache keeps a choice once made,
        // and one way's libraries would shadow the other's on the linker's path.
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        match env::var_os("MLX_SYS_PREBUILT") {
            Some(prefix) => link_prebuilt(config, Path::new(&prefix), &out),
            None => {
                config.out_dir(out.join("source"));
                link_from_source(config);
            }
        }
        generate_bindings(&root);
    }

    /// mlx-c against the installed MLX at `prefix`, and MLX's shared library. The prefix holds
    /// `lib/libmlx.dylib` with `lib/mlx.metallib` beside it, `include/` and `share/cmake/MLX/`, as
    /// the `mlx/` directory of the `mlx-metal` wheel does.
    fn link_prebuilt(mut config: Config, prefix: &Path, out: &Path) {
        let lib = prefix.join("lib");
        let cmake_dir = prefix.join("share/cmake/MLX");
        // What this reads from the prefix, so that an MLX replaced under the same path is noticed.
        // The directories, not their files: Cargo goes by modification times, and a wheel's files
        // keep the old ones they were packed with, where the directories they unpack into are new.
        for dir in [&lib, &prefix.join("include/mlx"), &cmake_dir] {
            println!("cargo:rerun-if-changed={}", dir.display());
        }
        for file in ["libmlx.dylib", "mlx.metallib"] {
            assert!(
                lib.join(file).is_file(),
                "MLX_SYS_PREBUILT: no {file} in {}",
                lib.display()
            );
        }
        let version = prebuilt_version(&cmake_dir);
        if let Some(found) = version.as_deref().filter(|found| *found != MLX_VERSION) {
            println!(
                "cargo:warning=MLX_SYS_PREBUILT holds MLX {found}, but mlx-c is written against MLX {MLX_VERSION}; mlx-c may not compile against it"
            );
        }
        // A build directory per MLX version: make goes by modification times too, and would keep
        // mlx-c's objects when another version's headers are older than they are.
        config.out_dir(out.join(format!(
            "prebuilt-{}",
            version.as_deref().unwrap_or("unknown")
        )));
        config.define("MLX_C_USE_SYSTEM_MLX", "ON");
        config.define("MLX_DIR", &cmake_dir);
        let dst = config.build();

        println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
        println!("cargo:rustc-link-search=native={}", lib.display());
        println!("cargo:rustc-link-lib=static=mlxc");
        println!("cargo:rustc-link-lib=dylib=mlx");
        link_system();
        // libmlx.dylib's install name is `@rpath/libmlx.dylib`, and MLX loads `mlx.metallib` from
        // beside it. This reaches this crate's own tests, examples and benchmarks; a build script
        // cannot add an rpath to another package's binaries, so a dependent reads the directory
        // as `DEP_MLX_LIB_DIR` and adds it to its own.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
        println!("cargo::metadata=lib_dir={}", lib.display());
    }

    /// mlx-c and MLX from source, as static libraries; the Metal library goes where a binary
    /// finds it after the build tree is gone.
    fn link_from_source(mut config: Config) {
        config.define("MLX_BUILD_METAL", "ON");
        config.define("MLX_BUILD_ACCELERATE", "ON");
        let metallib = metallib_dir();
        fs::create_dir_all(&metallib).expect("the directory of MLX's Metal library");
        config.define("MLX_METAL_PATH", &metallib);
        let dst = config.build();
        if !metallib.join("mlx.metallib").exists() {
            println!(
                "cargo:warning=mlx.metallib was not created at {}; Metal operations may fail at runtime",
                metallib.display()
            );
        }

        let build = dst.join("build");
        println!("cargo:rustc-link-search=native={}/lib", build.display());
        // MLX's GGUF reader is in a static library CMake leaves in MLX's build tree.
        println!(
            "cargo:rustc-link-search=native={}/_deps/mlx-build/mlx/io",
            build.display()
        );
        for library in ["mlx", "mlxc", "gguflib"] {
            println!("cargo:rustc-link-lib=static={library}");
        }
        if build.join("jaccl/libjaccl.a").is_file() {
            println!("cargo:rustc-link-search=native={}/jaccl", build.display());
            println!("cargo:rustc-link-lib=static=jaccl");
        }
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        link_system();
    }

    /// `MLX_METAL_PATH`, used as it is, or `~/.mlx/lib/<the mlx-c commit's first 12 digits>/`.
    fn metallib_dir() -> PathBuf {
        if let Some(path) = env::var_os("MLX_METAL_PATH") {
            return PathBuf::from(path);
        }
        let home = env::var_os("HOME").expect("HOME, or MLX_METAL_PATH");
        PathBuf::from(home)
            .join(".mlx/lib")
            .join(&MLX_C_COMMIT[..12])
    }

    /// What both ways of linking MLX need from the system.
    fn link_system() {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=dylib=objc");
        println!("cargo:rustc-link-lib=framework=Foundation");
        // MLX's `___isPlatformVersionAtLeast`, for a toolchain whose own runtime lacks it.
        if let Some(dir) = clang_rt_dir() {
            println!("cargo:rustc-link-search={}", dir.display());
            println!("cargo:rustc-link-lib=static=clang_rt.osx");
        }
    }

    /// The directory of Xcode's or the Command Line Tools' `libclang_rt.osx.a`, if one is there.
    fn clang_rt_dir() -> Option<PathBuf> {
        let output = Command::new("xcode-select")
            .arg("--print-path")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let developer = String::from_utf8_lossy(&output.stdout);
        let clang =
            Path::new(developer.trim()).join("Toolchains/XcodeDefault.xctoolchain/usr/lib/clang");
        fs::read_dir(clang).ok()?.flatten().find_map(|entry| {
            let dir = entry.path().join("lib/darwin");
            dir.join("libclang_rt.osx.a").exists().then_some(dir)
        })
    }

    /// The version of the MLX CMake package in `cmake_dir`, when it can be read.
    fn prebuilt_version(cmake_dir: &Path) -> Option<String> {
        let text = fs::read_to_string(cmake_dir.join("MLXConfigVersion.cmake")).ok()?;
        text.lines().find_map(|line| {
            let rest = line.trim().strip_prefix("set(PACKAGE_VERSION \"")?;
            rest.split('"').next().map(str::to_owned)
        })
    }

    /// `$OUT_DIR/bindings.rs`, from every header of mlx-c: its functions, types and constants, and
    /// nothing of the system headers' beyond what those name.
    fn generate_bindings(root: &Path) {
        let mut headers: Vec<PathBuf> = fs::read_dir(root.join("mlx/c"))
            .expect("mlx-c's headers")
            .map(|entry| entry.expect("mlx-c's headers").path())
            .filter(|path| path.extension().is_some_and(|e| e == "h"))
            .collect();
        headers.sort();
        let builder = bindgen::Builder::default()
            .rust_target("1.85.0".parse().expect("a Rust version"))
            .rust_edition(bindgen::RustEdition::Edition2024)
            .clang_arg(format!("-I{}", root.display()))
            .allowlist_function("_?mlx_.*")
            .allowlist_type("mlx_.*")
            .allowlist_var("mlx_.*");
        let bindings = headers
            .iter()
            .fold(builder, |builder, header| {
                builder.header(header.to_string_lossy())
            })
            .generate()
            .expect("the bindings to mlx-c");
        bindings
            .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
            .expect("the bindings' file");
    }
}
