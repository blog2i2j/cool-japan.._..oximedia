// Build script for oximedia-py.
//
// PyO3's `extension-module` feature suppresses linking against libpython so
// that the compiled `.so` extension module is loaded into an existing Python
// interpreter without duplicate Python state.  However, when Cargo builds
// test binaries (standalone executables based on the rlib), those Python
// symbols must be resolved at link time.
//
// We emit the libpython link directive here.  For the cdylib (the actual
// Python extension), this is harmless on Linux because the linker will
// resolve the symbols from the interpreter at dlopen time anyway.

fn main() {
    // Emit rerun guard so Cargo re-checks only when this script changes.
    println!("cargo:rerun-if-changed=build.rs");

    // Detect the target OS and emit the appropriate link directive.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Only needed on Linux/macOS; Windows uses generate-import-lib instead.
    if target_os == "linux" || target_os == "macos" {
        // On macOS, Python may be a framework build.  In that case LDLIBRARY
        // looks like "Python.framework/Versions/3.14/Python" and we must use
        // `-framework Python` with an `-F` search path instead of `-l`.
        if target_os == "macos" && is_framework_python() {
            if let Some(fw_dir) = find_python_framework_dir() {
                println!("cargo:rustc-link-search=framework={fw_dir}");
                println!("cargo:rustc-link-lib=framework=Python");
                return;
            }
        }

        // Non-framework case (Linux, or macOS non-framework installs).
        let python_lib = find_python_lib().unwrap_or_else(|| "python3.11".to_string());
        let python_lib_dir =
            find_python_lib_dir().unwrap_or_else(|| "/usr/lib/x86_64-linux-gnu".to_string());

        println!("cargo:rustc-link-search=native={python_lib_dir}");
        println!("cargo:rustc-link-lib={python_lib}");
    }
}

/// Check whether the installed Python is a framework build (macOS).
fn is_framework_python() -> bool {
    let out = match std::process::Command::new("python3")
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('PYTHONFRAMEWORK') or '')",
        ])
        .output()
    {
        Ok(o) => o,
        Err(_) => return false,
    };
    if !out.status.success() {
        return false;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    !s.is_empty()
}

/// Find the directory containing `Python.framework` (the `-F` search path).
///
/// We query `PYTHONFRAMEWORKPREFIX` which gives e.g.
/// `/opt/homebrew/opt/python@3.14/Frameworks`.  The linker needs the parent
/// directory that *contains* `Python.framework`, so if the prefix already
/// ends in the framework dir we go one level up.
fn find_python_framework_dir() -> Option<String> {
    let out = std::process::Command::new("python3")
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('PYTHONFRAMEWORKPREFIX') or '')",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let prefix = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if prefix.is_empty() {
        return None;
    }

    // PYTHONFRAMEWORKPREFIX is the directory that contains Python.framework.
    // Verify that Python.framework exists there.
    let candidate = std::path::Path::new(&prefix).join("Python.framework");
    if candidate.exists() {
        return Some(prefix);
    }

    // Some installs put it one level up (e.g. under Frameworks/).
    let parent = std::path::Path::new(&prefix).parent()?;
    let candidate2 = parent.join("Python.framework");
    if candidate2.exists() {
        return Some(parent.to_string_lossy().to_string());
    }

    // Last resort: just return the prefix and hope the linker finds it.
    Some(prefix)
}

/// Try to find the Python library name (e.g. "python3.11") via python3-config.
fn find_python_lib() -> Option<String> {
    let out = std::process::Command::new("python3")
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY') or '')",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    // On non-framework builds, s is like "libpython3.11.so" or "libpython3.11.a".
    // Strip "lib" prefix and extension suffix to get the -l argument.
    let name = s.strip_prefix("lib").unwrap_or(&s);
    let name = if let Some(pos) = name.find(".so") {
        &name[..pos]
    } else if let Some(pos) = name.find(".dylib") {
        &name[..pos]
    } else if let Some(pos) = name.find(".a") {
        &name[..pos]
    } else {
        name
    };
    let name = name.to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Try to find the directory that contains libpython.
fn find_python_lib_dir() -> Option<String> {
    let out = std::process::Command::new("python3")
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}
