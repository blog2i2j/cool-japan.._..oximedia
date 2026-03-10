// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Build script for oximedia-cv.
//!
//! Provides a glibc 2.38+ compatibility shim for ONNX Runtime.
//! The pre-compiled libonnxruntime.a references ISO C23 `__isoc23_*` symbols
//! introduced in glibc 2.38. On older systems (glibc 2.35), these don't exist.
//! We compile a thin C shim that wraps the glibc 2.35 equivalents.

fn main() {
    // Only apply compat shim on Linux (glibc), where the symbol mismatch occurs.
    // On macOS/Windows, ONNX Runtime uses different system libraries.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os == "linux" {
        // Detect if glibc version is older than 2.38 (missing __isoc23_* symbols).
        // We always compile the shim on Linux - if glibc >= 2.38 already defines
        // these symbols, the shim is a no-op (linker uses the first definition found).
        cc::Build::new()
            .file("compat/glibc_compat.c")
            // Compile as C99 (the shim code uses C99 restrict keyword).
            .flag("-std=c99")
            // Ensure symbols are visible for the linker.
            .flag("-fvisibility=default")
            // Optimize minimally - these are pure dispatch wrappers.
            .opt_level(2)
            .compile("glibc_compat");

        // cc::Build::compile() already emits:
        //   cargo:rustc-link-search=native=OUT_DIR
        //   cargo:rustc-link-lib=static=glibc_compat
        // But we emit them explicitly for clarity.
        println!("cargo:rerun-if-changed=compat/glibc_compat.c");
        println!("cargo:rerun-if-changed=build.rs");
    }
}
