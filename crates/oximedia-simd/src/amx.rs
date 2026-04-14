//! Apple Silicon AMX (Apple Matrix Extension) detection and acceleration paths.
//!
//! The Apple Matrix Extension (AMX) is a co-processor present on Apple M1 and
//! later SoCs that provides hardware matrix-multiply and accumulate operations
//! with tile registers up to 16×16 `f32` (or 32×32 `i8`).  It is used by
//! Apple's Accelerate framework and delivers significantly higher GEMM
//! throughput than NEON alone.
//!
//! # Scope of this module
//!
//! AMX is **not documented publicly** and its instruction encoding changes
//! between chip generations.  Apple does not expose a header for direct use;
//! only the Accelerate framework (Objective-C / Swift) is officially supported.
//!
//! This module therefore:
//!
//! 1. **Detects** whether the process is running on an Apple Silicon machine
//!    that is likely to have AMX hardware, using heuristic sysctl queries on
//!    macOS and `cpuinfo`-based detection on Linux (Asahi).
//! 2. **Exports** the [`AmxCapability`] type that downstream code can query.
//! 3. **Scaffolds** the future acceleration-path hooks: when AMX becomes
//!    accessible through a stable pure-Rust interface, implementations can be
//!    dropped in here without breaking the public API.
//! 4. **Provides** portable scalar fallbacks for every planned AMX operation,
//!    so the module is immediately useful as a quality reference even without
//!    real AMX hardware.
//!
//! # Usage
//!
//! ```
//! use oximedia_simd::amx::{AmxCapability, detect_amx};
//!
//! let caps = detect_amx();
//! println!("AMX present: {}", caps.has_amx);
//! println!("AMX generation: {:?}", caps.generation);
//! ```
//!
//! # Safety
//!
//! No `unsafe` code exists in this module.  All platform detection is done via
//! safe standard-library calls (`std::env`, `std::fs`, or
//! `std::process::Command` on macOS), with graceful fallback to
//! `AmxCapability::unavailable()` on any error.

#![allow(dead_code)]

use std::sync::OnceLock;

// ── Capability types ──────────────────────────────────────────────────────────

/// AMX hardware generation.
///
/// Each generation can have different tile register sizes and supported
/// precisions.  `Unknown` means detection succeeded but the exact revision
/// could not be determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmxGeneration {
    /// No AMX hardware detected or not an Apple Silicon platform.
    None,
    /// First-generation AMX (M1 family — A14, M1, M1 Pro/Max/Ultra).
    Gen1,
    /// Second-generation AMX (M2 family — A15/A16, M2, M2 Pro/Max/Ultra).
    Gen2,
    /// Third-generation AMX (M3 family and later).
    Gen3,
    /// AMX detected but generation could not be determined.
    Unknown,
}

impl std::fmt::Display for AmxGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Gen1 => write!(f, "gen1 (M1)"),
            Self::Gen2 => write!(f, "gen2 (M2)"),
            Self::Gen3 => write!(f, "gen3 (M3+)"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Detected AMX capabilities for the current process.
///
/// Obtain via [`detect_amx`] or [`detect_amx_cached`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmxCapability {
    /// Whether AMX hardware is present on this device.
    pub has_amx: bool,
    /// Hardware generation.
    pub generation: AmxGeneration,
    /// Whether the AMX can perform `f32` matrix-multiply (always true when
    /// `has_amx` is true — all known AMX generations support f32 GEMM).
    pub f32_matmul: bool,
    /// Whether the AMX can perform `f16` (half-precision) matrix-multiply.
    /// True on Gen2 (M2) and later.
    pub f16_matmul: bool,
    /// Whether the AMX can perform `i8` matrix-multiply (used for neural
    /// network inference).  True on Gen1 and later.
    pub i8_matmul: bool,
    /// Whether the AMX tile registers can be used for `bf16` accumulation.
    /// True on Gen2 and later.
    pub bf16_accumulate: bool,
}

impl AmxCapability {
    /// Construct a capability struct indicating no AMX hardware.
    #[must_use]
    pub const fn unavailable() -> Self {
        Self {
            has_amx: false,
            generation: AmxGeneration::None,
            f32_matmul: false,
            f16_matmul: false,
            i8_matmul: false,
            bf16_accumulate: false,
        }
    }

    /// Construct a capability struct for a given generation.
    #[must_use]
    pub fn for_generation(gen: AmxGeneration) -> Self {
        match gen {
            AmxGeneration::None => Self::unavailable(),
            AmxGeneration::Gen1 => Self {
                has_amx: true,
                generation: AmxGeneration::Gen1,
                f32_matmul: true,
                f16_matmul: false,
                i8_matmul: true,
                bf16_accumulate: false,
            },
            AmxGeneration::Gen2 => Self {
                has_amx: true,
                generation: AmxGeneration::Gen2,
                f32_matmul: true,
                f16_matmul: true,
                i8_matmul: true,
                bf16_accumulate: true,
            },
            AmxGeneration::Gen3 => Self {
                has_amx: true,
                generation: AmxGeneration::Gen3,
                f32_matmul: true,
                f16_matmul: true,
                i8_matmul: true,
                bf16_accumulate: true,
            },
            AmxGeneration::Unknown => Self {
                has_amx: true,
                generation: AmxGeneration::Unknown,
                f32_matmul: true, // Conservatively assume f32 only
                f16_matmul: false,
                i8_matmul: true,
                bf16_accumulate: false,
            },
        }
    }
}

// ── Detection logic ───────────────────────────────────────────────────────────

/// Detect AMX capabilities without caching.
///
/// This function performs OS-level probing on each call.  For hot paths use
/// [`detect_amx_cached`] instead.
///
/// # Platform notes
///
/// - **macOS (aarch64)**: queries `sysctl hw.optional.amx_version` (available
///   from macOS 12 Monterey onwards).  Falls back to chip-name matching via
///   `sysctl machdep.cpu.brand_string`.
/// - **Linux (aarch64)**: reads `/proc/cpuinfo` for `"Apple"` vendor and
///   matches `"cpu model"` lines.
/// - **All other platforms**: returns [`AmxCapability::unavailable()`].
#[must_use]
pub fn detect_amx() -> AmxCapability {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        return detect_amx_macos();
    }

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        return detect_amx_linux();
    }

    #[allow(unreachable_code)]
    AmxCapability::unavailable()
}

/// Cached AMX detection — probes once and returns the same result on all
/// subsequent calls.
#[must_use]
pub fn detect_amx_cached() -> AmxCapability {
    static CACHE: OnceLock<AmxCapability> = OnceLock::new();
    *CACHE.get_or_init(detect_amx)
}

// ── macOS detection ───────────────────────────────────────────────────────────

/// Run `sysctl -n <key>` and return trimmed stdout on success.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn sysctl_string(key: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl")
        .arg("-n")
        .arg(key)
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_owned())
    } else {
        None
    }
}

/// Infer AMX generation from the chip brand string (e.g. `"Apple M2 Pro"`).
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn generation_from_brand(brand: &str) -> AmxGeneration {
    let lower = brand.to_lowercase();
    if lower.contains("m3") || lower.contains("a17") || lower.contains("m4") {
        AmxGeneration::Gen3
    } else if lower.contains("m2") || lower.contains("a15") || lower.contains("a16") {
        AmxGeneration::Gen2
    } else if lower.contains("m1")
        || lower.contains("a14")
        || lower.contains("t8103")  // M1 internal identifier
        || lower.contains("t6000")
    // M1 Pro
    {
        AmxGeneration::Gen1
    } else if lower.contains("apple") {
        AmxGeneration::Unknown
    } else {
        AmxGeneration::None
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn detect_amx_macos() -> AmxCapability {
    // Try `sysctl hw.optional.amx_version` first (macOS 12+)
    if let Some(ver_str) = sysctl_string("hw.optional.amx_version") {
        let ver: u32 = ver_str.trim().parse().unwrap_or(0);
        let gen = match ver {
            1 => AmxGeneration::Gen1,
            2 => AmxGeneration::Gen2,
            v if v >= 3 => AmxGeneration::Gen3,
            _ => AmxGeneration::Unknown,
        };
        if gen != AmxGeneration::None {
            return AmxCapability::for_generation(gen);
        }
    }

    // Fall back to brand string heuristic
    if let Some(brand) = sysctl_string("machdep.cpu.brand_string") {
        let gen = generation_from_brand(&brand);
        if gen != AmxGeneration::None {
            return AmxCapability::for_generation(gen);
        }
    }

    AmxCapability::unavailable()
}

// ── Linux (Asahi) detection ───────────────────────────────────────────────────

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn detect_amx_linux() -> AmxCapability {
    use std::fs;
    use std::io::{BufRead, BufReader};

    let file = match fs::File::open("/proc/cpuinfo") {
        Ok(f) => f,
        Err(_) => return AmxCapability::unavailable(),
    };

    let reader = BufReader::new(file);
    let mut is_apple = false;
    let mut brand = String::new();

    for line in reader.lines().map_while(|l| l.ok()) {
        let lower = line.to_lowercase();
        if lower.starts_with("cpu implementer") && lower.contains("0x61") {
            // 0x61 == Apple
            is_apple = true;
        }
        if lower.starts_with("model name") || lower.starts_with("cpu model") {
            if let Some(val) = line.split(':').nth(1) {
                brand = val.trim().to_owned();
            }
        }
    }

    if !is_apple {
        return AmxCapability::unavailable();
    }

    let gen = if brand.is_empty() {
        AmxGeneration::Unknown
    } else {
        // Re-use the same heuristic as macOS
        let lower = brand.to_lowercase();
        if lower.contains("m3") || lower.contains("a17") {
            AmxGeneration::Gen3
        } else if lower.contains("m2") || lower.contains("a15") || lower.contains("a16") {
            AmxGeneration::Gen2
        } else if lower.contains("m1") || lower.contains("a14") {
            AmxGeneration::Gen1
        } else {
            AmxGeneration::Unknown
        }
    };

    AmxCapability::for_generation(gen)
}

// ── Scalar reference implementations ─────────────────────────────────────────
//
// When AMX becomes accessible through a pure-Rust interface, these scalar
// functions serve as correctness references and remain the fallback on
// non-AMX platforms.

/// General matrix-multiply: `C = alpha * A * B + beta * C`.
///
/// - `A` is `m × k`, row-major.
/// - `B` is `k × n`, row-major.
/// - `C` is `m × n`, row-major (in-out).
///
/// This scalar implementation is the portable fallback.  A future AMX-backed
/// version will carry the same signature.
///
/// # Errors
///
/// Returns `None` if the buffer sizes are inconsistent with `m`, `k`, `n`.
#[must_use]
pub fn amx_gemm_f32_fallback(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> bool {
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return false;
    }

    for i in 0..m {
        for j in 0..n {
            let mut dot = 0.0f32;
            for p in 0..k {
                dot += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = alpha * dot + beta * c[i * n + j];
        }
    }

    true
}

/// Compute `A * B` for two 4×4 `f32` matrices stored in row-major order.
///
/// Both `a` and `b` must have at least 16 elements.  Returns `None` if either
/// slice is too small.
///
/// On AMX hardware this would map to a single tile-load + FMADD instruction
/// sequence; here it is implemented as 64 multiply-accumulate operations.
#[must_use]
pub fn amx_matmul_4x4_f32(a: &[f32], b: &[f32]) -> Option<[f32; 16]> {
    if a.len() < 16 || b.len() < 16 {
        return None;
    }

    let mut out = [0.0f32; 16];
    for row in 0..4 {
        for col in 0..4 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                sum += a[row * 4 + k] * b[k * 4 + col];
            }
            out[row * 4 + col] = sum;
        }
    }

    Some(out)
}

/// Outer product accumulate: `C += A ⊗ B` for f32 vectors `a` (len m) and
/// `b` (len n), accumulating into matrix `c` (m × n).
///
/// On AMX this maps to the `FMAOUTER` tile instruction.
///
/// Returns `false` if `c.len() < m * n`.
#[must_use]
pub fn amx_outer_product_f32(a: &[f32], b: &[f32], c: &mut [f32]) -> bool {
    let m = a.len();
    let n = b.len();
    if c.len() < m * n {
        return false;
    }
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            c[i * n + j] += ai * bj;
        }
    }
    true
}

/// Dot product of two f32 slices (scalar reference for future AMX dot path).
///
/// Returns `0.0` if either slice is empty.  If the slices have different
/// lengths, only the shorter prefix is accumulated.
#[must_use]
pub fn amx_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AmxCapability ─────────────────────────────────────────────────────────

    #[test]
    fn unavailable_has_no_features() {
        let cap = AmxCapability::unavailable();
        assert!(!cap.has_amx);
        assert_eq!(cap.generation, AmxGeneration::None);
        assert!(!cap.f32_matmul);
        assert!(!cap.f16_matmul);
        assert!(!cap.i8_matmul);
        assert!(!cap.bf16_accumulate);
    }

    #[test]
    fn gen1_capability() {
        let cap = AmxCapability::for_generation(AmxGeneration::Gen1);
        assert!(cap.has_amx);
        assert!(cap.f32_matmul);
        assert!(cap.i8_matmul);
        assert!(!cap.f16_matmul, "Gen1 does not have f16 matmul");
        assert!(!cap.bf16_accumulate, "Gen1 does not have bf16");
    }

    #[test]
    fn gen2_capability() {
        let cap = AmxCapability::for_generation(AmxGeneration::Gen2);
        assert!(cap.has_amx);
        assert!(cap.f32_matmul);
        assert!(cap.f16_matmul, "Gen2 has f16 matmul");
        assert!(cap.bf16_accumulate, "Gen2 has bf16");
    }

    #[test]
    fn gen3_capability() {
        let cap = AmxCapability::for_generation(AmxGeneration::Gen3);
        assert!(cap.has_amx);
        assert!(cap.f16_matmul);
        assert!(cap.bf16_accumulate);
    }

    #[test]
    fn detect_amx_does_not_panic() {
        // Should always return without panicking regardless of platform
        let _ = detect_amx();
    }

    #[test]
    fn detect_amx_cached_consistent() {
        let a = detect_amx_cached();
        let b = detect_amx_cached();
        assert_eq!(a, b, "cached detection must be deterministic");
    }

    // ── Scalar GEMM ───────────────────────────────────────────────────────────

    #[test]
    fn amx_gemm_identity() {
        // C = 1.0 * I * A + 0.0 * C = A
        let identity = [1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let a = [2.0f32, 3.0, 4.0, 5.0]; // 2×2 matrix
        let mut c = [0.0f32; 4];
        let ok = amx_gemm_f32_fallback(&identity, &a, &mut c, 2, 2, 2, 1.0, 0.0);
        assert!(ok);
        assert_eq!(c, a, "I * A = A");
    }

    #[test]
    fn amx_gemm_buffer_too_small_returns_false() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut c = [0.0f32; 2]; // Too small for 2×2 output
        let ok = amx_gemm_f32_fallback(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
        assert!(!ok);
    }

    #[test]
    fn amx_matmul_4x4_identity() {
        let identity = [
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let result = amx_matmul_4x4_f32(&identity, &a).expect("4×4 matmul");
        for (i, (&r, &e)) in result.iter().zip(a.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "mismatch at {i}: got {r} expected {e}"
            );
        }
    }

    #[test]
    fn amx_matmul_4x4_too_small_returns_none() {
        let a = [1.0f32; 8]; // too short
        let b = [1.0f32; 16];
        assert!(amx_matmul_4x4_f32(&a, &b).is_none());
    }

    #[test]
    fn amx_outer_product_correctness() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let mut c = [0.0f32; 4];
        let ok = amx_outer_product_f32(&a, &b, &mut c);
        assert!(ok);
        // a ⊗ b = [[3, 4], [6, 8]]
        assert!((c[0] - 3.0).abs() < 1e-6);
        assert!((c[1] - 4.0).abs() < 1e-6);
        assert!((c[2] - 6.0).abs() < 1e-6);
        assert!((c[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn amx_outer_product_buffer_too_small_returns_false() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, 3.0];
        let mut c = [0.0f32; 4]; // needs 9
        assert!(!amx_outer_product_f32(&a, &b, &mut c));
    }

    #[test]
    fn amx_dot_f32_known_value() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let dot = amx_dot_f32(&a, &b);
        assert!((dot - 32.0).abs() < 1e-6, "1·4 + 2·5 + 3·6 = 32, got {dot}");
    }

    #[test]
    fn amx_dot_f32_empty_is_zero() {
        assert_eq!(amx_dot_f32(&[], &[]), 0.0);
    }

    // ── Generation display ────────────────────────────────────────────────────

    #[test]
    fn amx_generation_display() {
        assert_eq!(AmxGeneration::None.to_string(), "none");
        assert_eq!(AmxGeneration::Gen1.to_string(), "gen1 (M1)");
        assert_eq!(AmxGeneration::Gen2.to_string(), "gen2 (M2)");
        assert_eq!(AmxGeneration::Gen3.to_string(), "gen3 (M3+)");
        assert_eq!(AmxGeneration::Unknown.to_string(), "unknown");
    }
}
