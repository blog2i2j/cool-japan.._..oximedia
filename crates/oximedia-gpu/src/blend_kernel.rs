//! GPU blend kernels (CPU simulation via Rayon).
//!
//! Provides parallel image compositing and blending operations that simulate
//! GPU compute-shader dispatch semantics.
//!
//! # Supported blend modes
//!
//! | Mode | Description |
//! |------|-------------|
//! | [`BlendMode::AlphaComposite`] | Porter-Duff "over" compositing |
//! | [`BlendMode::Additive`] | src + dst, clamped to 255 |
//! | [`BlendMode::Multiply`] | src * dst / 255 |
//! | [`BlendMode::Screen`] | 1 - (1-src)*(1-dst) |
//! | [`BlendMode::Overlay`] | Photoshop-style overlay |
//! | [`BlendMode::SoftLight`] | Pegtop soft light formula |
//! | [`BlendMode::Difference`] | abs(src - dst) |
//! | [`BlendMode::Dissolve`] | Random per-pixel src/dst selection by opacity |
//!
//! # Example
//!
//! ```rust
//! use oximedia_gpu::blend_kernel::{BlendKernel, BlendMode};
//!
//! let src = vec![200u8, 100, 50, 255];   // opaque orange
//! let mut dst = vec![0u8, 128, 255, 255]; // opaque blue
//! BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::AlphaComposite, 255)
//!     .expect("blend failed");
//! ```

use rayon::prelude::*;
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors returned by blend kernel operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum BlendError {
    /// Source or destination buffer has incorrect length.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },
    /// Image dimensions are zero or invalid.
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    /// Pixel count overflow.
    #[error("Pixel count overflow for {width}x{height}")]
    PixelCountOverflow { width: u32, height: u32 },
    /// Mask length doesn't match pixel count.
    #[error("Mask length mismatch: expected {expected}, got {actual}")]
    MaskLengthMismatch { expected: usize, actual: usize },
}

// ─── BlendMode ───────────────────────────────────────────────────────────────

/// Compositing / blend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Porter-Duff "over" — alpha-premultiplied compositing.
    AlphaComposite,
    /// Linear additive blend, clamped to 255.
    Additive,
    /// Photographic multiply (darkens).
    Multiply,
    /// Screen blend (lightens).
    Screen,
    /// Overlay (contrast boost).
    Overlay,
    /// Pegtop soft-light formula.
    SoftLight,
    /// Absolute difference (subtract with abs).
    Difference,
    /// Stochastic dissolve controlled by global opacity.
    Dissolve,
}

impl BlendMode {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::AlphaComposite => "alpha_composite",
            Self::Additive => "additive",
            Self::Multiply => "multiply",
            Self::Screen => "screen",
            Self::Overlay => "overlay",
            Self::SoftLight => "soft_light",
            Self::Difference => "difference",
            Self::Dissolve => "dissolve",
        }
    }
}

// ─── BlendStats ──────────────────────────────────────────────────────────────

/// Statistics from a blend operation.
#[derive(Debug, Clone, Default)]
pub struct BlendStats {
    /// Total destination pixels modified.
    pub pixels_blended: u64,
    /// Blend mode used.
    pub mode: Option<BlendMode>,
    /// Global opacity applied (0–255).
    pub opacity: u8,
}

// ─── BlendKernel ─────────────────────────────────────────────────────────────

/// GPU-style image blending kernel (CPU simulation via Rayon).
///
/// Operates on packed RGBA (4 bytes / pixel) buffers.
/// `dst` is modified in place (it acts as both the base and the output).
#[derive(Debug, Clone, Default)]
pub struct BlendKernel;

impl BlendKernel {
    // ── Validation helpers ────────────────────────────────────────────────────

    fn validate_rgba(buf: &[u8], width: u32, height: u32) -> Result<usize, BlendError> {
        if width == 0 || height == 0 {
            return Err(BlendError::InvalidDimensions { width, height });
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or(BlendError::PixelCountOverflow { width, height })?;
        let expected = pixels * 4;
        if buf.len() != expected {
            return Err(BlendError::BufferSizeMismatch {
                expected,
                actual: buf.len(),
            });
        }
        Ok(pixels)
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Blend `src` over `dst` using the specified blend mode and global opacity.
    ///
    /// `opacity` controls how much of `src` is mixed in (0 = invisible, 255 = fully opaque).
    /// The per-pixel alpha in `src` is also respected by all modes.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] if buffers or dimensions are invalid.
    pub fn blend(
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
        mode: BlendMode,
        opacity: u8,
    ) -> Result<BlendStats, BlendError> {
        Self::validate_rgba(src, width, height)?;
        let pixels = Self::validate_rgba(dst, width, height)?;
        let op = opacity as f32 / 255.0;

        src.par_chunks(4)
            .zip(dst.par_chunks_mut(4))
            .for_each(|(s, d)| {
                blend_pixel(s, d, mode, op);
            });

        Ok(BlendStats {
            pixels_blended: pixels as u64,
            mode: Some(mode),
            opacity,
        })
    }

    /// Blend with a per-pixel opacity mask (single channel, one byte per pixel).
    ///
    /// Each pixel's effective opacity is `global_opacity * mask[i] / 255`.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] if any buffer or mask length is invalid.
    pub fn blend_masked(
        src: &[u8],
        dst: &mut [u8],
        mask: &[u8],
        width: u32,
        height: u32,
        mode: BlendMode,
        global_opacity: u8,
    ) -> Result<BlendStats, BlendError> {
        Self::validate_rgba(src, width, height)?;
        let pixels = Self::validate_rgba(dst, width, height)?;
        if mask.len() != pixels {
            return Err(BlendError::MaskLengthMismatch {
                expected: pixels,
                actual: mask.len(),
            });
        }

        let go = global_opacity as f32 / 255.0;

        src.par_chunks(4)
            .zip(dst.par_chunks_mut(4))
            .zip(mask.par_iter())
            .for_each(|((s, d), &m)| {
                let op = go * (m as f32 / 255.0);
                blend_pixel(s, d, mode, op);
            });

        Ok(BlendStats {
            pixels_blended: pixels as u64,
            mode: Some(mode),
            opacity: global_opacity,
        })
    }

    /// Composite multiple layers using Porter-Duff "over" operator.
    ///
    /// Layers are composited from bottom to top (index 0 is the base).
    /// Each `(buffer, opacity)` entry is blended onto the accumulator.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] if any layer has incorrect dimensions.
    pub fn composite_layers(
        layers: &[(&[u8], u8)],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, BlendError> {
        if width == 0 || height == 0 {
            return Err(BlendError::InvalidDimensions { width, height });
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or(BlendError::PixelCountOverflow { width, height })?;
        let buf_size = pixels * 4;

        for (i, (layer, _)) in layers.iter().enumerate() {
            if layer.len() != buf_size {
                return Err(BlendError::BufferSizeMismatch {
                    expected: buf_size,
                    actual: layer.len(),
                });
            }
            let _ = i;
        }

        if layers.is_empty() {
            return Ok(vec![0u8; buf_size]);
        }

        // Start with a transparent black canvas.
        let mut acc = vec![0u8; buf_size];
        for (layer, opacity) in layers {
            let op = *opacity as f32 / 255.0;
            layer
                .par_chunks(4)
                .zip(acc.par_chunks_mut(4))
                .for_each(|(s, d)| {
                    blend_pixel(s, d, BlendMode::AlphaComposite, op);
                });
        }
        Ok(acc)
    }

    /// Apply a constant color tint (multiply by a solid RGBA color).
    ///
    /// Each pixel is multiplied component-wise by `tint / 255`.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] if buffer or dimensions are invalid.
    pub fn apply_tint(
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
        tint: [u8; 4],
    ) -> Result<(), BlendError> {
        Self::validate_rgba(src, width, height)?;
        Self::validate_rgba(dst, width, height)?;

        src.par_chunks(4)
            .zip(dst.par_chunks_mut(4))
            .for_each(|(s, d)| {
                for c in 0..4 {
                    let v = (s[c] as u32 * tint[c] as u32 + 127) / 255;
                    d[c] = v.min(255) as u8;
                }
            });
        Ok(())
    }

    /// Premultiply RGB channels by the alpha channel in place.
    ///
    /// Input: `[R, G, B, A]` straight alpha.
    /// Output: `[R*A/255, G*A/255, B*A/255, A]` premultiplied.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] on dimension or buffer mismatch.
    pub fn premultiply_alpha(buf: &mut [u8], width: u32, height: u32) -> Result<(), BlendError> {
        Self::validate_rgba(buf, width, height)?;
        buf.par_chunks_mut(4).for_each(|px| {
            let a = px[3] as u32;
            for c in 0..3 {
                px[c] = ((px[c] as u32 * a + 127) / 255) as u8;
            }
        });
        Ok(())
    }

    /// Un-premultiply alpha (divide RGB by alpha).
    ///
    /// Pixels with `A = 0` are left as `[0, 0, 0, 0]`.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] on dimension or buffer mismatch.
    pub fn unpremultiply_alpha(buf: &mut [u8], width: u32, height: u32) -> Result<(), BlendError> {
        Self::validate_rgba(buf, width, height)?;
        buf.par_chunks_mut(4).for_each(|px| {
            let a = px[3] as f32;
            if a > 0.0 {
                for c in 0..3 {
                    px[c] = (px[c] as f32 / a * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        });
        Ok(())
    }
}

// ─── Pixel-level blend functions ──────────────────────────────────────────────

/// Apply one blend operation to a single `[R,G,B,A]` pixel pair.
///
/// `s` is the source pixel, `d` is the destination (modified in place).
/// `opacity` is the global layer opacity in `[0.0, 1.0]`.
fn blend_pixel(s: &[u8], d: &mut [u8], mode: BlendMode, opacity: f32) {
    let sa = (s[3] as f32 / 255.0) * opacity;
    match mode {
        BlendMode::AlphaComposite => alpha_composite(s, d, sa),
        BlendMode::Additive => additive(s, d, sa),
        BlendMode::Multiply => multiply(s, d, sa),
        BlendMode::Screen => screen(s, d, sa),
        BlendMode::Overlay => overlay(s, d, sa),
        BlendMode::SoftLight => soft_light(s, d, sa),
        BlendMode::Difference => difference(s, d, sa),
        BlendMode::Dissolve => dissolve(s, d, sa),
    }
}

/// Porter-Duff "over": `Cout = Cs * As + Cd * Ad * (1 - As)`.
fn alpha_composite(s: &[u8], d: &mut [u8], sa: f32) {
    let da = d[3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a < 1e-9 {
        d[0] = 0;
        d[1] = 0;
        d[2] = 0;
        d[3] = 0;
        return;
    }
    for c in 0..3 {
        let sc = s[c] as f32 / 255.0;
        let dc = d[c] as f32 / 255.0;
        let out_c = (sc * sa + dc * da * (1.0 - sa)) / out_a;
        d[c] = (out_c * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    d[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

/// Additive blend: `Cout = clamp(Cd + Cs * Sa, 0, 255)`.
fn additive(s: &[u8], d: &mut [u8], sa: f32) {
    for c in 0..3 {
        let v = d[c] as f32 + s[c] as f32 * sa;
        d[c] = v.round().clamp(0.0, 255.0) as u8;
    }
    // Alpha: additive does not change destination alpha.
}

/// Multiply blend: `Cout = lerp(Cd, Cd * Cs / 255, Sa)`.
fn multiply(s: &[u8], d: &mut [u8], sa: f32) {
    for c in 0..3 {
        let dc = d[c] as f32;
        let sc = s[c] as f32;
        let blended = dc * sc / 255.0;
        d[c] = lerp_channel(dc, blended, sa);
    }
}

/// Screen blend: `Cout = lerp(Cd, 255 - (255-Cd)*(255-Cs)/255, Sa)`.
fn screen(s: &[u8], d: &mut [u8], sa: f32) {
    for c in 0..3 {
        let dc = d[c] as f32;
        let sc = s[c] as f32;
        let blended = 255.0 - (255.0 - dc) * (255.0 - sc) / 255.0;
        d[c] = lerp_channel(dc, blended, sa);
    }
}

/// Overlay blend: multiply if dst < 0.5, screen otherwise.
fn overlay(s: &[u8], d: &mut [u8], sa: f32) {
    for c in 0..3 {
        let dc = d[c] as f32 / 255.0;
        let sc = s[c] as f32 / 255.0;
        let blended = if dc < 0.5 {
            2.0 * dc * sc
        } else {
            1.0 - 2.0 * (1.0 - dc) * (1.0 - sc)
        };
        d[c] = lerp_channel(d[c] as f32, blended * 255.0, sa);
    }
}

/// Pegtop soft-light: `2*Cd*Cs + Cd²*(1-2*Cs)` (all in [0,1]).
fn soft_light(s: &[u8], d: &mut [u8], sa: f32) {
    for c in 0..3 {
        let dc = d[c] as f32 / 255.0;
        let sc = s[c] as f32 / 255.0;
        let blended = 2.0 * dc * sc + dc * dc * (1.0 - 2.0 * sc);
        d[c] = lerp_channel(d[c] as f32, blended * 255.0, sa);
    }
}

/// Difference blend: `Cout = lerp(Cd, abs(Cd - Cs), Sa)`.
fn difference(s: &[u8], d: &mut [u8], sa: f32) {
    for c in 0..3 {
        let dc = d[c] as f32;
        let sc = s[c] as f32;
        let blended = (dc - sc).abs();
        d[c] = lerp_channel(dc, blended, sa);
    }
}

/// Dissolve: use source pixel when opacity threshold is met.
///
/// Deterministic per-pixel: uses a hash of the channel index to simulate
/// stochastic per-pixel selection without an RNG.
fn dissolve(s: &[u8], d: &mut [u8], sa: f32) {
    // Deterministic "random" threshold using a simple fixed pattern.
    // In real GPU shaders this uses a noise texture; here we use a xor-shift
    // of the combined src/dst byte values to avoid needing a true RNG.
    let hash =
        xorshift32(s[0] as u32 ^ (s[1] as u32 * 17) ^ (d[0] as u32 * 31) ^ (d[1] as u32 * 7));
    let threshold = (hash & 0xFF) as f32 / 255.0;
    if sa > threshold {
        // Use source pixel
        for c in 0..3 {
            d[c] = s[c];
        }
        d[3] = s[3];
    }
    // else keep destination unchanged
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Linear interpolation: `a + (b - a) * t`, returns u8.
#[inline]
fn lerp_channel(a: f32, b: f32, t: f32) -> u8 {
    (a + (b - a) * t).round().clamp(0.0, 255.0) as u8
}

/// Simple xorshift32 for deterministic dissolve without RNG.
#[inline]
fn xorshift32(mut x: u32) -> u32 {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BlendMode ─────────────────────────────────────────────────────────────

    #[test]
    fn test_blend_mode_labels() {
        assert_eq!(BlendMode::AlphaComposite.label(), "alpha_composite");
        assert_eq!(BlendMode::Additive.label(), "additive");
        assert_eq!(BlendMode::Multiply.label(), "multiply");
        assert_eq!(BlendMode::Screen.label(), "screen");
        assert_eq!(BlendMode::Overlay.label(), "overlay");
        assert_eq!(BlendMode::SoftLight.label(), "soft_light");
        assert_eq!(BlendMode::Difference.label(), "difference");
        assert_eq!(BlendMode::Dissolve.label(), "dissolve");
    }

    // ── Error handling ────────────────────────────────────────────────────────

    #[test]
    fn test_blend_invalid_dims() {
        let src = vec![0u8; 4];
        let mut dst = vec![0u8; 4];
        let err = BlendKernel::blend(&src, &mut dst, 0, 1, BlendMode::Additive, 255);
        assert!(matches!(err, Err(BlendError::InvalidDimensions { .. })));
    }

    #[test]
    fn test_blend_buffer_mismatch() {
        let src = vec![0u8; 8]; // wrong: 1×1 = 4
        let mut dst = vec![0u8; 4];
        let err = BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::Additive, 255);
        assert!(matches!(err, Err(BlendError::BufferSizeMismatch { .. })));
    }

    #[test]
    fn test_blend_masked_mask_mismatch() {
        let src = vec![255u8; 4 * 4 * 4];
        let mut dst = vec![0u8; 4 * 4 * 4];
        let mask = vec![255u8; 10]; // wrong size
        let err = BlendKernel::blend_masked(&src, &mut dst, &mask, 4, 4, BlendMode::Multiply, 255);
        assert!(matches!(err, Err(BlendError::MaskLengthMismatch { .. })));
    }

    // ── Opacity=0 preserves destination ──────────────────────────────────────

    #[test]
    fn test_opacity_zero_preserves_dst() {
        let src: Vec<u8> = vec![255, 0, 0, 255]; // opaque red
        let original_dst = vec![0u8, 128, 255, 255]; // opaque blue
        let mut dst = original_dst.clone();
        BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::AlphaComposite, 0).unwrap();
        // With opacity=0 the src contributes nothing; dst should be almost unchanged.
        for (orig, &out) in original_dst.iter().zip(dst.iter()) {
            let diff = (*orig as i16 - out as i16).abs();
            assert!(diff <= 1, "channel diff={diff}");
        }
    }

    // ── Opacity=255 opaque source covers destination (AlphaComposite) ─────────

    #[test]
    fn test_alpha_composite_fully_opaque_src() {
        let src = vec![200u8, 100, 50, 255]; // fully opaque
        let mut dst = vec![0u8, 0, 0, 255];
        BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::AlphaComposite, 255).unwrap();
        assert_eq!(dst[0], 200);
        assert_eq!(dst[1], 100);
        assert_eq!(dst[2], 50);
        assert_eq!(dst[3], 255);
    }

    // ── Additive ──────────────────────────────────────────────────────────────

    #[test]
    fn test_additive_blend_clamps_to_255() {
        let src = vec![200u8, 200, 200, 255];
        let mut dst = vec![100u8, 100, 100, 255];
        BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::Additive, 255).unwrap();
        assert_eq!(dst[0], 255, "200+100=300 → clamp to 255");
    }

    #[test]
    fn test_additive_blend_zero_src() {
        let src = vec![0u8, 0, 0, 255];
        let original = vec![100u8, 150, 200, 255];
        let mut dst = original.clone();
        BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::Additive, 255).unwrap();
        assert_eq!(dst[..3], original[..3]);
    }

    // ── Multiply ──────────────────────────────────────────────────────────────

    #[test]
    fn test_multiply_with_white_src_unchanged() {
        let src = vec![255u8, 255, 255, 255]; // white multiplier
        let original = vec![100u8, 150, 200, 255];
        let mut dst = original.clone();
        BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::Multiply, 255).unwrap();
        // Multiply with white → dst * 255/255 = dst
        for c in 0..3 {
            let diff = (original[c] as i16 - dst[c] as i16).abs();
            assert!(diff <= 1, "channel {c}: diff={diff}");
        }
    }

    #[test]
    fn test_multiply_with_black_src_yields_zero() {
        let src = vec![0u8, 0, 0, 255]; // black multiplier
        let mut dst = vec![200u8, 150, 100, 255];
        BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::Multiply, 255).unwrap();
        // Multiply with black → 0
        for c in 0..3 {
            assert_eq!(dst[c], 0, "channel {c} should be 0");
        }
    }

    // ── Screen ────────────────────────────────────────────────────────────────

    #[test]
    fn test_screen_with_black_src_unchanged() {
        let src = vec![0u8, 0, 0, 255]; // black screen → no change
        let original = vec![100u8, 150, 200, 255];
        let mut dst = original.clone();
        BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::Screen, 255).unwrap();
        for c in 0..3 {
            let diff = (original[c] as i16 - dst[c] as i16).abs();
            assert!(diff <= 1, "channel {c}: diff={diff}");
        }
    }

    #[test]
    fn test_screen_with_white_src_yields_white() {
        let src = vec![255u8, 255, 255, 255]; // white screen → white
        let mut dst = vec![100u8, 150, 200, 255];
        BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::Screen, 255).unwrap();
        for c in 0..3 {
            assert_eq!(
                dst[c], 255,
                "channel {c} should be 255 after screen with white"
            );
        }
    }

    // ── Difference ────────────────────────────────────────────────────────────

    #[test]
    fn test_difference_with_same_src_dst_yields_black() {
        let src = vec![100u8, 150, 200, 255];
        let mut dst = vec![100u8, 150, 200, 255];
        BlendKernel::blend(&src, &mut dst, 1, 1, BlendMode::Difference, 255).unwrap();
        for c in 0..3 {
            assert_eq!(dst[c], 0, "difference of equal values should be 0");
        }
    }

    // ── Masked blend ──────────────────────────────────────────────────────────

    #[test]
    fn test_masked_blend_all_opaque() {
        let w = 2u32;
        let h = 2u32;
        let src = vec![255u8; (w * h * 4) as usize];
        let mut dst = vec![0u8; (w * h * 4) as usize];
        let mask = vec![255u8; (w * h) as usize]; // fully opaque mask
        BlendKernel::blend_masked(&src, &mut dst, &mask, w, h, BlendMode::AlphaComposite, 255)
            .unwrap();
        // All dst pixels should become white (src is all 255 opaque)
        for &v in &dst {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn test_masked_blend_all_transparent_preserves_dst() {
        let w = 2u32;
        let h = 2u32;
        let src = vec![255u8; (w * h * 4) as usize];
        let original_dst = vec![100u8; (w * h * 4) as usize];
        let mut dst = original_dst.clone();
        let mask = vec![0u8; (w * h) as usize]; // fully transparent mask
        BlendKernel::blend_masked(&src, &mut dst, &mask, w, h, BlendMode::AlphaComposite, 255)
            .unwrap();
        // With mask=0, alpha=0, dst should be unchanged
        assert_eq!(dst, original_dst);
    }

    // ── Layer compositing ─────────────────────────────────────────────────────

    #[test]
    fn test_composite_layers_empty_returns_transparent() {
        let result = BlendKernel::composite_layers(&[], 4, 4).unwrap();
        assert_eq!(result.len(), 4 * 4 * 4);
        assert!(result.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_composite_layers_single_opaque() {
        let layer = vec![200u8, 100, 50, 255]; // 1×1 opaque orange
        let result = BlendKernel::composite_layers(&[(&layer, 255)], 1, 1).unwrap();
        assert_eq!(result.len(), 4);
        // Should match the layer
        assert_eq!(result[0], 200);
        assert_eq!(result[1], 100);
        assert_eq!(result[2], 50);
    }

    // ── Tint ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_tint_white_tint_unchanged() {
        let src = vec![100u8, 150, 200, 255];
        let mut dst = vec![0u8; 4];
        BlendKernel::apply_tint(&src, &mut dst, 1, 1, [255, 255, 255, 255]).unwrap();
        for c in 0..3 {
            let diff = (src[c] as i16 - dst[c] as i16).abs();
            assert!(diff <= 1, "channel {c}: diff={diff}");
        }
    }

    #[test]
    fn test_apply_tint_black_tint_yields_black() {
        let src = vec![200u8, 150, 100, 255];
        let mut dst = vec![0u8; 4];
        BlendKernel::apply_tint(&src, &mut dst, 1, 1, [0, 0, 0, 0]).unwrap();
        assert_eq!(dst[0], 0);
        assert_eq!(dst[1], 0);
        assert_eq!(dst[2], 0);
    }

    // ── Premultiply / unpremultiply ───────────────────────────────────────────

    #[test]
    fn test_premultiply_alpha_full_opaque() {
        let mut buf = vec![200u8, 100, 50, 255];
        BlendKernel::premultiply_alpha(&mut buf, 1, 1).unwrap();
        // With alpha=255, channels stay the same
        assert_eq!(buf[0], 200);
        assert_eq!(buf[1], 100);
        assert_eq!(buf[2], 50);
    }

    #[test]
    fn test_premultiply_alpha_half_opacity() {
        let mut buf = vec![200u8, 200, 200, 128];
        BlendKernel::premultiply_alpha(&mut buf, 1, 1).unwrap();
        // ~200 * 128 / 255 ≈ 100
        let expected = (200u32 * 128 + 127) / 255;
        let diff = (buf[0] as i32 - expected as i32).abs();
        assert!(
            diff <= 1,
            "premultiplied R: got {}, expected ~{}",
            buf[0],
            expected
        );
    }

    #[test]
    fn test_unpremultiply_alpha_zero_alpha() {
        let mut buf = vec![100u8, 100, 100, 0]; // fully transparent
        BlendKernel::unpremultiply_alpha(&mut buf, 1, 1).unwrap();
        // alpha=0 → no change (stays 0)
        assert_eq!(buf[0], 100); // left unchanged when alpha=0
    }

    #[test]
    fn test_premultiply_unpremultiply_roundtrip() {
        let original = vec![200u8, 150, 100, 200];
        let mut buf = original.clone();
        BlendKernel::premultiply_alpha(&mut buf, 1, 1).unwrap();
        BlendKernel::unpremultiply_alpha(&mut buf, 1, 1).unwrap();
        for c in 0..3 {
            let diff = (original[c] as i16 - buf[c] as i16).abs();
            assert!(
                diff <= 2,
                "channel {c}: orig={} back={} diff={diff}",
                original[c],
                buf[c]
            );
        }
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_blend_stats_returned() {
        let src = vec![0u8; 4 * 4 * 4];
        let mut dst = vec![0u8; 4 * 4 * 4];
        let stats = BlendKernel::blend(&src, &mut dst, 4, 4, BlendMode::Screen, 200).unwrap();
        assert_eq!(stats.pixels_blended, 16);
        assert_eq!(stats.mode, Some(BlendMode::Screen));
        assert_eq!(stats.opacity, 200);
    }
}
