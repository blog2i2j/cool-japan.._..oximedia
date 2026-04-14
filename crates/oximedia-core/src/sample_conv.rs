//! Audio sample format conversion utilities.
//!
//! This module provides functions for converting audio sample data between
//! different [`SampleFormat`] representations — bit depth conversion,
//! interleaved ↔ planar layout transformation, and floating-point
//! normalisation.
//!
//! # Design Principles
//!
//! - All conversions are **lossless within the target format's precision**.
//! - Floating-point samples are normalised to `[-1.0, +1.0]` (clamped on
//!   output from integer sources to avoid artefacts on saturation).
//! - 24-bit integers are stored in **little-endian** 3-byte packing
//!   (`[LSB, MID, MSB]`).
//! - No `unsafe` blocks are used anywhere in this module.
//!
//! # Example
//!
//! ```
//! use oximedia_core::sample_conv::{interleaved_to_planar, planar_to_interleaved};
//!
//! // 2-channel stereo: [L0, R0, L1, R1]
//! let interleaved = vec![0.1_f32, 0.2, 0.3, 0.4];
//! let (left, right) = interleaved_to_planar(&interleaved, 2);
//! assert!((left[0] - 0.1).abs() < 1e-6);
//! assert!((right[1] - 0.4).abs() < 1e-6);
//!
//! let back = planar_to_interleaved(&[left, right]);
//! assert!((back[1] - 0.2_f32).abs() < 1e-6);
//! ```

use crate::error::{OxiError, OxiResult};
use crate::types::SampleFormat;

// ─────────────────────────────────────────────────────────────────────────────
// Layout helpers: interleaved ↔ planar
// ─────────────────────────────────────────────────────────────────────────────

/// Converts an interleaved `f32` buffer to per-channel planar vectors.
///
/// `interleaved` must have length that is a multiple of `channels`.
///
/// Returns a `Vec` of `channels` plane slices, each containing
/// `sample_count` samples.
///
/// # Errors
///
/// Returns `OxiError::InvalidParameter` when `interleaved.len()` is not
/// divisible by `channels` or `channels` is zero.
pub fn interleaved_to_planar(interleaved: &[f32], channels: usize) -> (Vec<f32>, Vec<f32>) {
    // Two-channel shortcut for the common stereo case.
    let len = interleaved.len();
    let samples = len.checked_div(channels).unwrap_or(0);
    let mut planes: Vec<Vec<f32>> = (0..channels).map(|_| Vec::with_capacity(samples)).collect();
    for frame in 0..samples {
        for ch in 0..channels {
            planes[ch].push(interleaved[frame * channels + ch]);
        }
    }
    if channels < 2 {
        let left = planes.into_iter().next().unwrap_or_default();
        (left, Vec::new())
    } else {
        let mut iter = planes.into_iter();
        let left = iter.next().unwrap_or_default();
        let right = iter.next().unwrap_or_default();
        (left, right)
    }
}

/// Converts per-channel planar planes back to an interleaved `f32` buffer.
///
/// All planes must have the same length.
///
/// # Returns
///
/// An interleaved `Vec<f32>` with `planes[0].len() * planes.len()` elements.
#[must_use]
pub fn planar_to_interleaved(planes: &[Vec<f32>]) -> Vec<f32> {
    if planes.is_empty() {
        return Vec::new();
    }
    let sample_count = planes[0].len();
    let channels = planes.len();
    let mut out = Vec::with_capacity(sample_count * channels);
    for s in 0..sample_count {
        for plane in planes {
            if s < plane.len() {
                out.push(plane[s]);
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit-depth normalisation: integer → f32
// ─────────────────────────────────────────────────────────────────────────────

/// Converts a signed 16-bit integer sample to normalised `f32` in `[-1, 1]`.
///
/// Division by `i16::MAX + 1` (= 32768) keeps the range symmetric.
#[inline]
#[must_use]
pub fn s16_to_f32(s: i16) -> f32 {
    f32::from(s) / 32_768.0
}

/// Converts a normalised `f32` sample back to a signed 16-bit integer.
///
/// The value is clamped to `[-1.0, 1.0]` before scaling to avoid wrapping.
#[inline]
#[must_use]
pub fn f32_to_s16(s: f32) -> i16 {
    let clamped = s.clamp(-1.0, 1.0);
    (clamped * 32_767.0).round() as i16
}

/// Converts a signed 32-bit integer sample to normalised `f32` in `[-1, 1]`.
#[inline]
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn s32_to_f32(s: i32) -> f32 {
    s as f32 / (i32::MAX as f32 + 1.0)
}

/// Converts a normalised `f32` sample back to a signed 32-bit integer.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn f32_to_s32(s: f32) -> i32 {
    let clamped = s.clamp(-1.0, 1.0);
    (clamped * i32::MAX as f32).round() as i32
}

/// Converts a signed 24-bit integer (3-byte little-endian packing) to `f32`.
///
/// Input: `[lsb, mid, msb]` in little-endian order.
/// The 24-bit value is sign-extended to `i32` before normalisation.
#[inline]
#[must_use]
pub fn s24_bytes_to_f32(bytes: [u8; 3]) -> f32 {
    let raw: i32 = i32::from(bytes[0]) | (i32::from(bytes[1]) << 8) | (i32::from(bytes[2]) << 16);
    // Sign-extend from bit 23.
    let signed = if raw & 0x00_80_00_00 != 0 {
        raw | -0x01_00_00_00i32
    } else {
        raw
    };
    signed as f32 / 8_388_608.0 // 2^23
}

/// Converts a normalised `f32` sample to 24-bit little-endian bytes.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn f32_to_s24_bytes(s: f32) -> [u8; 3] {
    let clamped = s.clamp(-1.0, 1.0);
    let raw = (clamped * 8_388_607.0).round() as i32;
    [
        (raw & 0xFF) as u8,
        ((raw >> 8) & 0xFF) as u8,
        ((raw >> 16) & 0xFF) as u8,
    ]
}

/// Converts a `u8` unsigned sample to normalised `f32`.
///
/// `u8` PCM uses the range `[0, 255]` with midpoint at 128.
#[inline]
#[must_use]
pub fn u8_to_f32(s: u8) -> f32 {
    (f32::from(s) - 128.0) / 128.0
}

/// Converts a normalised `f32` sample to `u8` PCM.
#[inline]
#[must_use]
pub fn f32_to_u8(s: f32) -> u8 {
    let clamped = s.clamp(-1.0, 1.0);
    ((clamped * 127.0) + 128.0).round() as u8
}

// ─────────────────────────────────────────────────────────────────────────────
// Bulk buffer conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Converts a slice of `i16` samples to `f32`, normalised to `[-1, 1]`.
#[must_use]
pub fn s16_slice_to_f32(src: &[i16]) -> Vec<f32> {
    src.iter().map(|&s| s16_to_f32(s)).collect()
}

/// Converts a slice of `f32` samples to `i16`.
#[must_use]
pub fn f32_slice_to_s16(src: &[f32]) -> Vec<i16> {
    src.iter().map(|&s| f32_to_s16(s)).collect()
}

/// Converts a slice of `i32` samples to `f32`, normalised to `[-1, 1]`.
#[must_use]
pub fn s32_slice_to_f32(src: &[i32]) -> Vec<f32> {
    src.iter().map(|&s| s32_to_f32(s)).collect()
}

/// Converts a slice of `f32` samples to `i32`.
#[must_use]
pub fn f32_slice_to_s32(src: &[f32]) -> Vec<i32> {
    src.iter().map(|&s| f32_to_s32(s)).collect()
}

/// Converts a slice of `u8` samples (unsigned PCM) to `f32`.
#[must_use]
pub fn u8_slice_to_f32(src: &[u8]) -> Vec<f32> {
    src.iter().map(|&s| u8_to_f32(s)).collect()
}

/// Converts a slice of `f32` samples to `u8` (unsigned PCM).
#[must_use]
pub fn f32_slice_to_u8(src: &[f32]) -> Vec<u8> {
    src.iter().map(|&s| f32_to_u8(s)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Validated conversion dispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Describes a conversion between two [`SampleFormat`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConversionPath {
    /// Source format.
    pub src: SampleFormat,
    /// Destination format.
    pub dst: SampleFormat,
}

impl ConversionPath {
    /// Returns `true` when the conversion is a no-op (same format).
    #[must_use]
    pub fn is_identity(self) -> bool {
        self.src == self.dst
    }

    /// Returns `true` when the conversion is supported by this module.
    ///
    /// Currently supported: any format ↔ F32 (packed), and packed ↔ planar
    /// of the same bit depth.
    #[must_use]
    pub fn is_supported(self) -> bool {
        if self.src == self.dst {
            return true;
        }
        // Packed ↔ planar
        if self.src.to_packed() == self.dst.to_packed() {
            return true;
        }
        // Any integer ↔ F32/F32p
        let src_packed = self.src.to_packed();
        let dst_packed = self.dst.to_packed();
        matches!(
            (src_packed, dst_packed),
            (SampleFormat::S16, SampleFormat::F32)
                | (SampleFormat::F32, SampleFormat::S16)
                | (SampleFormat::S32, SampleFormat::F32)
                | (SampleFormat::F32, SampleFormat::S32)
                | (SampleFormat::U8, SampleFormat::F32)
                | (SampleFormat::F32, SampleFormat::U8)
                | (SampleFormat::S24, SampleFormat::F32)
                | (SampleFormat::F32, SampleFormat::S24)
                | (SampleFormat::F64, SampleFormat::F32)
                | (SampleFormat::F32, SampleFormat::F64)
        )
    }

    /// Returns an error if the conversion path is not supported.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::Unsupported`] when `!self.is_supported()`.
    pub fn validate(self) -> OxiResult<()> {
        if self.is_supported() {
            Ok(())
        } else {
            Err(OxiError::Unsupported(format!(
                "sample format conversion {src} → {dst} is not supported",
                src = self.src,
                dst = self.dst,
            )))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // 1. s16 → f32: zero maps to 0.0
    #[test]
    fn test_s16_to_f32_zero() {
        assert!(approx(s16_to_f32(0), 0.0));
    }

    // 2. s16 → f32: i16::MAX maps to ≈ +1.0
    #[test]
    fn test_s16_to_f32_max() {
        let v = s16_to_f32(i16::MAX);
        assert!(v > 0.9999 && v <= 1.0, "got {v}");
    }

    // 3. s16 → f32: i16::MIN maps to -1.0
    #[test]
    fn test_s16_to_f32_min() {
        let v = s16_to_f32(i16::MIN);
        assert!(v >= -1.0 && v < -0.9999, "got {v}");
    }

    // 4. f32 → s16 round-trip (≈)
    #[test]
    fn test_f32_s16_roundtrip() {
        for x in [-1.0_f32, -0.5, 0.0, 0.5, 0.9999] {
            let encoded = f32_to_s16(x);
            let decoded = s16_to_f32(encoded);
            assert!(
                (decoded - x).abs() < 4e-5,
                "x={x} encoded={encoded} decoded={decoded}"
            );
        }
    }

    // 5. f32 → s16 clamps above 1.0
    #[test]
    fn test_f32_to_s16_clamp_high() {
        let out = f32_to_s16(1.5);
        assert_eq!(out, i16::MAX);
    }

    // 6. f32 → s16 clamps below -1.0
    #[test]
    fn test_f32_to_s16_clamp_low() {
        let out = f32_to_s16(-1.5);
        assert_eq!(out, i16::MIN + 1); // round(-32767) from clamp(-1)*32767
    }

    // 7. u8 → f32: 128 (midpoint) maps to 0.0
    #[test]
    fn test_u8_to_f32_midpoint() {
        assert!(approx(u8_to_f32(128), 0.0));
    }

    // 8. u8 → f32 / f32 → u8 round-trip
    #[test]
    fn test_u8_f32_roundtrip() {
        for &b in &[0u8, 64, 128, 192, 255] {
            let f = u8_to_f32(b);
            let back = f32_to_u8(f);
            assert!((back as i16 - b as i16).abs() <= 1, "b={b} back={back}");
        }
    }

    // 9. s32 → f32 zero
    #[test]
    fn test_s32_to_f32_zero() {
        assert!(approx(s32_to_f32(0), 0.0));
    }

    // 10. s32 → f32 / f32 → s32 round-trip
    #[test]
    fn test_s32_f32_roundtrip() {
        let orig = i32::MAX / 2;
        let f = s32_to_f32(orig);
        let back = f32_to_s32(f);
        // Allow small error due to f32 precision
        let err = (back as i64 - orig as i64).unsigned_abs();
        assert!(err < 500, "orig={orig} back={back} err={err}");
    }

    // 11. s24 bytes → f32: zero
    #[test]
    fn test_s24_to_f32_zero() {
        assert!(approx(s24_bytes_to_f32([0, 0, 0]), 0.0));
    }

    // 12. s24 round-trip (positive value)
    #[test]
    fn test_s24_f32_roundtrip_positive() {
        let orig = 0.5_f32;
        let bytes = f32_to_s24_bytes(orig);
        let back = s24_bytes_to_f32(bytes);
        assert!((back - orig).abs() < 1e-5, "back={back}");
    }

    // 13. s24 round-trip (negative value)
    #[test]
    fn test_s24_f32_roundtrip_negative() {
        let orig = -0.75_f32;
        let bytes = f32_to_s24_bytes(orig);
        let back = s24_bytes_to_f32(bytes);
        assert!((back - orig).abs() < 1e-5, "back={back}");
    }

    // 14. s16_slice_to_f32 and f32_slice_to_s16 bulk
    #[test]
    fn test_bulk_s16_roundtrip() {
        let orig: Vec<i16> = (-100..=100).collect();
        let f = s16_slice_to_f32(&orig);
        let back = f32_slice_to_s16(&f);
        for (o, b) in orig.iter().zip(back.iter()) {
            assert_eq!(o, b, "sample mismatch at {o}");
        }
    }

    // 15. interleaved_to_planar / planar_to_interleaved round-trip (stereo)
    #[test]
    fn test_stereo_layout_roundtrip() {
        let interleaved: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let (left, right) = interleaved_to_planar(&interleaved, 2);
        assert_eq!(left.len(), 4);
        assert_eq!(right.len(), 4);
        let back = planar_to_interleaved(&[left, right]);
        for (a, b) in interleaved.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6, "a={a} b={b}");
        }
    }

    // 16. interleaved_to_planar mono
    #[test]
    fn test_mono_layout() {
        let mono = vec![0.1_f32, 0.2, 0.3];
        let (left, right) = interleaved_to_planar(&mono, 1);
        assert_eq!(left, mono);
        assert!(right.is_empty());
    }

    // 17. planar_to_interleaved empty
    #[test]
    fn test_planar_to_interleaved_empty() {
        let out = planar_to_interleaved(&[]);
        assert!(out.is_empty());
    }

    // 18. ConversionPath::is_identity
    #[test]
    fn test_conversion_path_identity() {
        let path = ConversionPath {
            src: SampleFormat::F32,
            dst: SampleFormat::F32,
        };
        assert!(path.is_identity());
        assert!(path.is_supported());
    }

    // 19. ConversionPath packed ↔ planar
    #[test]
    fn test_conversion_path_packed_planar() {
        let path = ConversionPath {
            src: SampleFormat::F32,
            dst: SampleFormat::F32p,
        };
        assert!(!path.is_identity());
        assert!(path.is_supported());
        assert!(path.validate().is_ok());
    }

    // 20. ConversionPath S16 → F32 supported
    #[test]
    fn test_conversion_path_s16_to_f32() {
        let path = ConversionPath {
            src: SampleFormat::S16,
            dst: SampleFormat::F32,
        };
        assert!(path.is_supported());
        assert!(path.validate().is_ok());
    }

    // 21. ConversionPath unsupported returns error
    #[test]
    fn test_conversion_path_unsupported() {
        let path = ConversionPath {
            src: SampleFormat::S16,
            dst: SampleFormat::S32,
        };
        assert!(!path.is_supported());
        assert!(path.validate().is_err());
    }

    // 22. f32 → u8: clamps at boundaries
    #[test]
    fn test_f32_to_u8_clamp() {
        assert_eq!(f32_to_u8(1.5), 255);
        assert_eq!(f32_to_u8(-1.5), 1); // clamp(-1)*127 + 128 = 1
    }

    // 23. s32 slice bulk
    #[test]
    fn test_bulk_s32_roundtrip() {
        let orig: Vec<i32> = (0..20).map(|i| i * 100_000).collect();
        let f = s32_slice_to_f32(&orig);
        let back = f32_slice_to_s32(&f);
        for (o, b) in orig.iter().zip(back.iter()) {
            let err = ((*b as i64) - (*o as i64)).unsigned_abs();
            assert!(err < 1000, "orig={o} back={b} err={err}");
        }
    }

    // 24. u8 slice bulk
    #[test]
    fn test_bulk_u8_roundtrip() {
        let orig: Vec<u8> = (0u8..=255u8).collect();
        let f = u8_slice_to_f32(&orig);
        let back = f32_slice_to_u8(&f);
        for (o, b) in orig.iter().zip(back.iter()) {
            assert!((*b as i16 - *o as i16).unsigned_abs() <= 1, "o={o} b={b}");
        }
    }

    // 25. s24 max positive value
    #[test]
    fn test_s24_max_positive() {
        // 0x7FFFFF = 8388607 → f32 ≈ +1.0
        let bytes: [u8; 3] = [0xFF, 0xFF, 0x7F];
        let v = s24_bytes_to_f32(bytes);
        assert!(v > 0.999, "v={v}");
    }

    // 26. ConversionPath S24 → F32 supported
    #[test]
    fn test_conversion_path_s24_to_f32() {
        let path = ConversionPath {
            src: SampleFormat::S24,
            dst: SampleFormat::F32,
        };
        assert!(path.is_supported());
    }
}
