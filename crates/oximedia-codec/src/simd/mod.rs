//! SIMD abstraction layer for video codec implementations.
//!
//! This module provides a unified interface for SIMD operations used in
//! video encoding and decoding. It abstracts over different SIMD instruction
//! sets (AVX2, AVX-512, NEON) while providing a scalar fallback for portability.
//!
//! # Architecture
//!
//! The SIMD abstraction consists of:
//!
//! - **Types** (`types.rs`): Vector types like `I16x8`, `I32x4`, `U8x16`
//! - **Traits** (`traits.rs`): `SimdOps` and `SimdOpsExt` for SIMD operations
//! - **Architecture-specific**: x86 (AVX2/AVX-512), ARM (NEON), scalar fallback
//! - **Codec-specific**: AV1 and VP9 optimized operations
//! - **Operations**: Domain-specific modules for codec operations
//!
//! # Usage
//!
//! ```ignore
//! use oximedia_codec::simd::{detect_simd, select_transform_impl};
//!
//! // Detect SIMD capabilities
//! let caps = detect_simd();
//! println!("Best SIMD: {}", caps.best_level());
//!
//! // Use codec-specific SIMD operations
//! use oximedia_codec::simd::av1::TransformSimd;
//! let transform = TransformSimd::new(select_transform_impl());
//! transform.forward_dct_8x8(&input, &mut output);
//! ```
//!
//! # Feature Detection and Dispatch
//!
//! The SIMD implementation is selected at runtime based on CPU capabilities:
//!
//! ```ignore
//! use oximedia_codec::simd::{SimdCapabilities, detect_simd};
//!
//! let caps = detect_simd();
//! if caps.avx512 {
//!     // Use AVX-512 optimized path
//! } else if caps.avx2 {
//!     // Use AVX2 path
//! } else if caps.neon {
//!     // Use ARM NEON path
//! } else {
//!     // Use scalar fallback
//! }
//! ```
//!
//! ## SIMD Dispatch Mechanism
//!
//! OxiMedia uses a two-tier dispatch strategy to guarantee correctness on every target
//! while achieving maximum throughput on modern hardware.
//!
//! **Tier 1: Compile-time `cfg` selection.**
//! Target-specific code paths are gated with `#[cfg(target_arch = "...")]`, so only the
//! code relevant to the current build target is compiled in:
//!
//! - `x86_64` — AVX-512 (`avx512f` + `avx512bw` + `avx512dq`), AVX2, SSE4.2 paths
//! - `aarch64` — ARM NEON path (always present on AArch64)
//! - `wasm32` — WASM SIMD128 path (`simd/wasm.rs`, `core::arch::wasm32` intrinsics)
//! - All other targets — scalar fallback only
//!
//! **Tier 2: Runtime [`SimdCapabilities`] detection.**
//! Even on `x86_64`, AVX-512 may not be available at runtime. [`detect_simd`] probes the
//! CPU at startup using `is_x86_feature_detected!` and fills a [`SimdCapabilities`] struct:
//!
//! ```ignore
//! use oximedia_codec::simd::{SimdCapabilities, detect_simd};
//!
//! let caps: SimdCapabilities = detect_simd();
//! if caps.avx512 {
//!     // 512-bit vector path — Ice Lake, Skylake-X, Zen 4+
//! } else if caps.avx2 {
//!     // 256-bit vector path — Haswell 2013+, Excavator 2015+
//! } else if caps.neon {
//!     // ARM NEON path — all ARMv8/AArch64
//! } else {
//!     // Pure scalar fallback
//! }
//! ```
//!
//! The `get_simd()` helper encapsulates the dispatch and returns a `&'static dyn SimdOps`:
//!
//! ```ignore
//! use oximedia_codec::simd::get_simd;
//!
//! let ops = get_simd();  // picks AVX-512 → AVX2 → NEON → scalar
//! ops.sad_8x8(&src, &ref_block); // calls fastest available path
//! ```
//!
//! **Tier 3: Scalar fallback.**
//! [`ScalarFallback`] provides a 100% pure-Rust implementation of every [`SimdOps`]
//! operation. It is always compiled in and always selected when no SIMD extension is
//! detected. This means OxiMedia:
//!
//! - compiles on any Rust target (including `wasm32`, `riscv64`, `mips`, etc.)
//! - runs correctly on any hardware, even without SIMD support
//! - achieves SIMD acceleration silently when the extension is available
//!
//! No unsafe dispatch tables or runtime dynamic linking are used; all dispatch paths are
//! statically allocated (`static AVX2_INSTANCE: Avx2Simd = Avx2Simd`) and accessed
//! via a single `&'static dyn SimdOps` fat pointer.

#![allow(unsafe_code)]

// Core modules
pub mod scalar;
pub mod traits;
pub mod types;

// Architecture-specific implementations
pub mod arm;
pub mod x86;

// Codec-specific SIMD operations
pub mod av1;
pub mod vp9;

// Legacy operation modules (preserved for compatibility)
pub mod blend;
pub mod dct;
pub mod filter;
pub mod sad;

// Pixel format conversion (YUV ↔ RGB, all subsampling modes)
pub mod pixel_convert;

// YUV subsampling format conversion (4:2:0 ↔ 4:2:2 ↔ 4:4:4, NV12 ↔ I420)
pub mod yuv_convert;

// Re-exports
pub use blend::{blend_ops, BlendOps};
pub use dct::{dct_ops, DctOps};
pub use filter::{filter_ops, FilterOps};
pub use sad::{sad_ops, SadOps};
pub use traits::{SimdOps, SimdOpsExt, SimdSelector};
pub use types::{I16x16, I16x8, I32x4, I32x8, U8x16, U8x32};

// Architecture-specific re-exports
pub use arm::NeonSimd;
pub use scalar::ScalarFallback;
pub use x86::{Avx2Simd, Avx512Simd};

// Codec-specific re-exports
pub use av1::{CdefSimd, IntraPredSimd, LoopFilterSimd, MotionCompSimd, TransformSimd};
pub use vp9::{Vp9DctSimd, Vp9InterpolateSimd, Vp9IntraPredSimd, Vp9LoopFilterSimd};

// ============================================================================
// CPU Feature Detection and Dispatch
// ============================================================================

/// CPU SIMD capabilities.
///
/// This structure represents the SIMD instruction sets available on the
/// current CPU, detected at runtime.
#[derive(Clone, Copy, Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct SimdCapabilities {
    /// x86 AVX2 support (Intel Haswell 2013+, AMD Excavator 2015+).
    pub avx2: bool,

    /// x86 AVX-512 support (Intel Skylake-X 2017+, Ice Lake 2019+).
    pub avx512: bool,

    /// ARM NEON support (all ARMv8/AArch64, ARMv7-A with NEON).
    pub neon: bool,
}

impl SimdCapabilities {
    /// Create with all features disabled.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            avx2: false,
            avx512: false,
            neon: false,
        }
    }

    /// Check if AVX2 is available.
    #[inline]
    #[must_use]
    pub const fn has_avx2(&self) -> bool {
        self.avx2
    }

    /// Check if AVX-512 is available.
    #[inline]
    #[must_use]
    pub const fn has_avx512(&self) -> bool {
        self.avx512
    }

    /// Check if NEON is available.
    #[inline]
    #[must_use]
    pub const fn has_neon(&self) -> bool {
        self.neon
    }

    /// Get the best available SIMD level name.
    #[must_use]
    pub const fn best_level(&self) -> &'static str {
        if self.avx512 {
            "avx512"
        } else if self.avx2 {
            "avx2"
        } else if self.neon {
            "neon"
        } else {
            "scalar"
        }
    }
}

/// Detect CPU SIMD capabilities at runtime.
///
/// This function uses CPU feature detection to determine which SIMD
/// instruction sets are available on the current processor.
///
/// # Returns
///
/// A `SimdCapabilities` struct indicating which SIMD features are available.
///
/// # Example
///
/// ```ignore
/// use oximedia_codec::simd::detect_simd;
///
/// let caps = detect_simd();
/// println!("Running on: {}", caps.best_level());
/// ```
#[must_use]
pub fn detect_simd() -> SimdCapabilities {
    #[cfg(target_arch = "x86_64")]
    {
        SimdCapabilities {
            avx2: is_x86_feature_detected!("avx2"),
            avx512: is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512dq"),
            neon: false,
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // On AArch64, NEON is always available
        SimdCapabilities {
            avx2: false,
            avx512: false,
            neon: true,
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdCapabilities::none()
    }
}

/// Transform implementation selection.
///
/// This enum represents the different SIMD implementations available
/// for transform operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformImpl {
    /// AVX-512 implementation.
    Avx512,
    /// AVX2 implementation.
    Avx2,
    /// ARM NEON implementation.
    Neon,
    /// Scalar fallback implementation.
    Scalar,
}

/// Select the best transform implementation for the current CPU.
///
/// This function detects CPU capabilities and returns the optimal
/// transform implementation.
///
/// # Returns
///
/// The best available `TransformImpl` for the current CPU.
#[must_use]
pub fn select_transform_impl() -> TransformImpl {
    let caps = detect_simd();

    if caps.has_avx512() {
        TransformImpl::Avx512
    } else if caps.has_avx2() {
        TransformImpl::Avx2
    } else if caps.has_neon() {
        TransformImpl::Neon
    } else {
        TransformImpl::Scalar
    }
}

// Static instances for each SIMD implementation
static SCALAR_INSTANCE: ScalarFallback = ScalarFallback;

#[cfg(target_arch = "x86_64")]
static AVX2_INSTANCE: Avx2Simd = Avx2Simd;

#[cfg(target_arch = "x86_64")]
static AVX512_INSTANCE: Avx512Simd = Avx512Simd;

#[cfg(target_arch = "aarch64")]
static NEON_INSTANCE: NeonSimd = NeonSimd;

/// Get the best SIMD implementation for the current CPU.
///
/// Returns a reference to the optimal SIMD implementation based on
/// detected CPU capabilities. This provides dynamic dispatch to the
/// fastest available implementation.
///
/// # Returns
///
/// A static reference to a `SimdOps` implementation.
#[must_use]
pub fn get_simd() -> &'static dyn SimdOps {
    #[cfg(target_arch = "x86_64")]
    {
        if Avx512Simd::is_available() {
            return &AVX512_INSTANCE;
        } else if Avx2Simd::is_available() {
            return &AVX2_INSTANCE;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if NeonSimd::is_available() {
            return &NEON_INSTANCE;
        }
    }

    &SCALAR_INSTANCE
}

/// Get the best extended SIMD implementation for the current CPU.
///
/// Returns a reference to the optimal extended SIMD implementation
/// (with additional operations like transpose and butterfly).
///
/// # Returns
///
/// A static reference to a `SimdOpsExt` implementation.
#[must_use]
pub fn get_simd_ext() -> &'static dyn SimdOpsExt {
    #[cfg(target_arch = "x86_64")]
    {
        if Avx512Simd::is_available() {
            return &AVX512_INSTANCE;
        } else if Avx2Simd::is_available() {
            return &AVX2_INSTANCE;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if NeonSimd::is_available() {
            return &NEON_INSTANCE;
        }
    }

    &SCALAR_INSTANCE
}

// ============================================================================
// Legacy Compatibility
// ============================================================================

/// Legacy scalar SIMD accessor (deprecated, use `ScalarFallback` directly).
#[deprecated(
    since = "0.1.0",
    note = "Use &SCALAR_INSTANCE or ScalarFallback directly"
)]
#[must_use]
pub fn scalar_simd() -> &'static ScalarFallback {
    &SCALAR_INSTANCE
}

/// Legacy capabilities detection (deprecated, use `detect_simd` instead).
#[deprecated(since = "0.1.0", note = "Use detect_simd() instead")]
#[must_use]
pub fn detect_capabilities() -> SimdCapabilities {
    detect_simd()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_simd() {
        let caps = detect_simd();

        // Should return valid capabilities
        let level = caps.best_level();
        assert!(!level.is_empty());

        // At least one implementation should be available
        assert!(get_simd().is_available());
    }

    #[test]
    fn test_simd_capabilities() {
        let caps = SimdCapabilities::none();
        assert!(!caps.has_avx2());
        assert!(!caps.has_avx512());
        assert!(!caps.has_neon());
        assert_eq!(caps.best_level(), "scalar");
    }

    #[test]
    fn test_get_simd() {
        let simd = get_simd();
        assert!(simd.is_available());

        // Check that the name matches expected values
        let name = simd.name();
        assert!(
            name == "scalar" || name == "avx2" || name == "avx512" || name == "neon",
            "Unexpected SIMD name: {}",
            name
        );
    }

    #[test]
    fn test_get_simd_ext() {
        let simd = get_simd_ext();
        assert!(simd.is_available());
    }

    #[test]
    fn test_select_transform_impl() {
        let impl_type = select_transform_impl();

        // Should select a valid implementation
        match impl_type {
            TransformImpl::Avx512
            | TransformImpl::Avx2
            | TransformImpl::Neon
            | TransformImpl::Scalar => {}
        }
    }

    #[test]
    fn test_module_reexports() {
        // Test that all reexports work
        let _v = I16x8::zero();
        let _v = I32x4::zero();
        let _v = U8x16::zero();

        let _ops = sad_ops();
        let _ops = blend_ops();
        let _ops = dct_ops();
        let _ops = filter_ops();
    }

    #[test]
    fn test_architecture_specific() {
        // Test that architecture-specific types are accessible
        let _scalar = ScalarFallback::new();

        #[cfg(target_arch = "x86_64")]
        {
            let _avx2 = Avx2Simd::new();
            let _avx512 = Avx512Simd::new();
        }

        #[cfg(target_arch = "aarch64")]
        {
            let _neon = NeonSimd::new();
        }
    }

    #[test]
    fn test_codec_specific_types() {
        // Verify codec-specific types are accessible
        use crate::simd::scalar::ScalarFallback;

        let simd = ScalarFallback::new();

        // AV1
        let _transform = TransformSimd::new(simd);
        let _loop_filter = LoopFilterSimd::new(simd);
        let _cdef = CdefSimd::new(simd);
        let _intra = IntraPredSimd::new(simd);
        let _motion = MotionCompSimd::new(simd);

        // VP9
        let _vp9_dct = Vp9DctSimd::new(simd);
        let _vp9_interp = Vp9InterpolateSimd::new(simd);
        let _vp9_intra = Vp9IntraPredSimd::new(simd);
        let _vp9_lf = Vp9LoopFilterSimd::new(simd);
    }

    #[test]
    fn test_integration_sad() {
        let sad = sad_ops();

        // Test basic SAD calculation
        let src = [128u8; 64];
        let ref_block = [128u8; 64];

        let result = sad.sad_8x8(&src, 8, &ref_block, 8);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_integration_blend() {
        let blend = blend_ops();

        // Test linear interpolation
        let result = blend.lerp_u8(0, 255, 128);
        assert!(result >= 126 && result <= 130);
    }

    #[test]
    fn test_integration_dct() {
        let dct = dct_ops();

        // Test DCT round-trip
        let input = [100i16; 16];
        let mut dct_out = [0i16; 16];
        let mut reconstructed = [0i16; 16];

        dct.forward_dct_4x4(&input, &mut dct_out);
        dct.inverse_dct_4x4(&dct_out, &mut reconstructed);

        // Should be close to original
        for i in 0..16 {
            let diff = (input[i] - reconstructed[i]).abs();
            assert!(
                diff <= 2,
                "DCT mismatch at {}: {} vs {}",
                i,
                input[i],
                reconstructed[i]
            );
        }
    }

    #[test]
    fn test_integration_filter() {
        let filter = filter_ops();

        // Test 2-tap filter on constant input
        let src = [128u8; 16];
        let mut dst = [0u8; 15];

        filter.filter_h_2tap(&src, &mut dst, 15);

        for &v in &dst {
            assert_eq!(v, 128);
        }
    }
}
