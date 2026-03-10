//! Multiple blend mode implementations for video frames.
//!
//! Provides Screen, Multiply, Overlay, Soft Light, Hard Light, Color Dodge,
//! Color Burn, and many more professional blending modes, together with the
//! high-level [`blend_frames`] helper.

#![allow(dead_code)]

use super::{validate_buffer, PixelFormat, VideoResult};

// ── Blend mode enum ────────────────────────────────────────────────────────────

/// Blend mode for compositing two RGBA frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Standard alpha compositing.
    Normal,
    /// Multiply – darkens by multiplying values.
    Multiply,
    /// Screen – lightens by inverting-multiply-inverting.
    Screen,
    /// Overlay – contrast-enhancing mix of Multiply and Screen.
    Overlay,
    /// Soft Light – gentle Overlay variant.
    SoftLight,
    /// Hard Light – strong Overlay variant.
    HardLight,
    /// Color Dodge – brightens the base.
    ColorDodge,
    /// Color Burn – darkens the base.
    ColorBurn,
    /// Difference – absolute difference of channels.
    Difference,
    /// Exclusion – softer Difference.
    Exclusion,
    /// Add (Linear Dodge) – sums channels, clamped.
    Add,
    /// Subtract – subtracts top from base, clamped to zero.
    Subtract,
    /// Darken – per-channel minimum.
    Darken,
    /// Lighten – per-channel maximum.
    Lighten,
}

// ── High-level API ─────────────────────────────────────────────────────────────

/// Blend `overlay` on top of `base` using `mode` at the given `opacity` [0, 1].
///
/// All three buffers must be RGBA (4 bytes per pixel) with identical dimensions.
///
/// # Errors
///
/// Returns [`crate::EffectError::BufferSizeMismatch`] if any buffer has the
/// wrong size.
pub fn blend_frames(
    base: &[u8],
    overlay: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    mode: BlendMode,
    opacity: f32,
) -> VideoResult<()> {
    validate_buffer(base, width, height, PixelFormat::Rgba)?;
    validate_buffer(overlay, width, height, PixelFormat::Rgba)?;
    validate_buffer(output, width, height, PixelFormat::Rgba)?;

    let opacity = opacity.clamp(0.0, 1.0);

    for i in 0..width * height {
        let idx = i * 4;
        let b = to_f32_4(&base[idx..idx + 4]);
        let s = to_f32_4(&overlay[idx..idx + 4]);

        let blended = blend_pixel(b, s, mode);

        // Mix blended with base using effective alpha = source_alpha * opacity
        let eff_alpha = s[3] * opacity;
        let inv = 1.0 - eff_alpha;

        output[idx] = f32_to_u8(blended[0] * eff_alpha + b[0] * inv);
        output[idx + 1] = f32_to_u8(blended[1] * eff_alpha + b[1] * inv);
        output[idx + 2] = f32_to_u8(blended[2] * eff_alpha + b[2] * inv);
        output[idx + 3] = f32_to_u8(eff_alpha + b[3] * inv);
    }

    Ok(())
}

// ── Per-pixel blending ─────────────────────────────────────────────────────────

/// Blend a single pixel pair in normalised [0, 1] RGBA space.
#[must_use]
pub fn blend_pixel(b: [f32; 4], s: [f32; 4], mode: BlendMode) -> [f32; 4] {
    let (r, g, bl) = match mode {
        BlendMode::Normal => (s[0], s[1], s[2]),
        BlendMode::Multiply => blend3(b, s, multiply),
        BlendMode::Screen => blend3(b, s, screen),
        BlendMode::Overlay => blend3(b, s, overlay),
        BlendMode::SoftLight => blend3(b, s, soft_light),
        BlendMode::HardLight => blend3(b, s, hard_light),
        BlendMode::ColorDodge => blend3(b, s, color_dodge),
        BlendMode::ColorBurn => blend3(b, s, color_burn),
        BlendMode::Difference => blend3(b, s, difference),
        BlendMode::Exclusion => blend3(b, s, exclusion),
        BlendMode::Add => blend3(b, s, add),
        BlendMode::Subtract => blend3(b, s, subtract),
        BlendMode::Darken => blend3(b, s, darken),
        BlendMode::Lighten => blend3(b, s, lighten),
    };
    [r, g, bl, s[3]]
}

// ── Channel-level blend functions ──────────────────────────────────────────────

#[inline]
fn multiply(b: f32, s: f32) -> f32 {
    b * s
}

#[inline]
fn screen(b: f32, s: f32) -> f32 {
    1.0 - (1.0 - b) * (1.0 - s)
}

#[inline]
fn overlay(b: f32, s: f32) -> f32 {
    if b < 0.5 {
        2.0 * b * s
    } else {
        1.0 - 2.0 * (1.0 - b) * (1.0 - s)
    }
}

#[inline]
fn soft_light(b: f32, s: f32) -> f32 {
    if s < 0.5 {
        b - (1.0 - 2.0 * s) * b * (1.0 - b)
    } else {
        b + (2.0 * s - 1.0) * (d_fn(b) - b)
    }
}

#[inline]
fn d_fn(x: f32) -> f32 {
    if x <= 0.25 {
        ((16.0 * x - 12.0) * x + 4.0) * x
    } else {
        x.sqrt()
    }
}

#[inline]
fn hard_light(b: f32, s: f32) -> f32 {
    overlay(s, b) // Hard light = Overlay with layers swapped
}

#[inline]
fn color_dodge(b: f32, s: f32) -> f32 {
    if s >= 1.0 {
        1.0
    } else {
        (b / (1.0 - s)).min(1.0)
    }
}

#[inline]
fn color_burn(b: f32, s: f32) -> f32 {
    if s <= 0.0 {
        0.0
    } else {
        1.0 - ((1.0 - b) / s).min(1.0)
    }
}

#[inline]
fn difference(b: f32, s: f32) -> f32 {
    (b - s).abs()
}

#[inline]
fn exclusion(b: f32, s: f32) -> f32 {
    b + s - 2.0 * b * s
}

#[inline]
fn add(b: f32, s: f32) -> f32 {
    (b + s).min(1.0)
}

#[inline]
fn subtract(b: f32, s: f32) -> f32 {
    (b - s).max(0.0)
}

#[inline]
fn darken(b: f32, s: f32) -> f32 {
    b.min(s)
}

#[inline]
fn lighten(b: f32, s: f32) -> f32 {
    b.max(s)
}

// ── Utilities ─────────────────────────────────────────────────────────────────

#[inline]
fn blend3<F>(b: [f32; 4], s: [f32; 4], f: F) -> (f32, f32, f32)
where
    F: Fn(f32, f32) -> f32,
{
    (f(b[0], s[0]), f(b[1], s[1]), f(b[2], s[2]))
}

#[inline]
fn to_f32_4(slice: &[u8]) -> [f32; 4] {
    [
        f32::from(slice[0]) / 255.0,
        f32::from(slice[1]) / 255.0,
        f32::from(slice[2]) / 255.0,
        f32::from(slice[3]) / 255.0,
    ]
}

#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn f32_to_u8(v: f32) -> u8 {
    (v * 255.0).clamp(0.0, 255.0) as u8
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: usize, h: usize, rgba: [u8; 4]) -> Vec<u8> {
        vec![rgba; w * h].into_iter().flatten().collect()
    }

    #[test]
    fn test_blend_frames_normal_full_opacity() {
        let base = solid(4, 4, [255, 0, 0, 255]);
        let overlay = solid(4, 4, [0, 0, 255, 255]);
        let mut out = vec![0u8; 4 * 4 * 4];
        blend_frames(&base, &overlay, &mut out, 4, 4, BlendMode::Normal, 1.0)
            .expect("test expectation failed");
        // Full opacity Normal → overlay replaces base
        assert!(out[2] > 200, "Blue channel should dominate");
    }

    #[test]
    fn test_blend_frames_normal_zero_opacity() {
        let base = solid(4, 4, [255, 0, 0, 255]);
        let overlay = solid(4, 4, [0, 0, 255, 255]);
        let mut out = vec![0u8; 4 * 4 * 4];
        blend_frames(&base, &overlay, &mut out, 4, 4, BlendMode::Normal, 0.0)
            .expect("test expectation failed");
        // Zero opacity → base unchanged
        assert_eq!(out[0], 255);
        assert_eq!(out[2], 0);
    }

    #[test]
    fn test_multiply_darkens() {
        let base = solid(2, 2, [200, 200, 200, 255]);
        let overlay = solid(2, 2, [128, 128, 128, 255]);
        let mut out = vec![0u8; 2 * 2 * 4];
        blend_frames(&base, &overlay, &mut out, 2, 2, BlendMode::Multiply, 1.0)
            .expect("test expectation failed");
        assert!(out[0] < 200, "Multiply should darken");
    }

    #[test]
    fn test_screen_lightens() {
        let base = solid(2, 2, [100, 100, 100, 255]);
        let overlay = solid(2, 2, [100, 100, 100, 255]);
        let mut out = vec![0u8; 2 * 2 * 4];
        blend_frames(&base, &overlay, &mut out, 2, 2, BlendMode::Screen, 1.0)
            .expect("test expectation failed");
        assert!(out[0] >= 100, "Screen should lighten or preserve");
    }

    #[test]
    fn test_add_clamps_to_255() {
        let base = solid(2, 2, [200, 200, 200, 255]);
        let overlay = solid(2, 2, [200, 200, 200, 255]);
        let mut out = vec![0u8; 2 * 2 * 4];
        blend_frames(&base, &overlay, &mut out, 2, 2, BlendMode::Add, 1.0)
            .expect("test expectation failed");
        assert_eq!(out[0], 255, "Add should clamp to 255");
    }

    #[test]
    fn test_difference_identical_is_black() {
        let frame = solid(2, 2, [120, 80, 40, 255]);
        let mut out = vec![0u8; 2 * 2 * 4];
        blend_frames(&frame, &frame, &mut out, 2, 2, BlendMode::Difference, 1.0)
            .expect("test expectation failed");
        assert!(out[0] < 10, "Difference of identical frames should be ~0");
    }

    #[test]
    fn test_buffer_mismatch_error() {
        let base = vec![0u8; 10];
        let overlay = solid(4, 4, [0, 0, 0, 255]);
        let mut out = vec![0u8; 4 * 4 * 4];
        let result = blend_frames(&base, &overlay, &mut out, 4, 4, BlendMode::Normal, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_color_dodge_bright_source_max() {
        // source channel = 1.0 → result should be 1.0
        let result = color_dodge(0.5, 1.0);
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_darken_and_lighten() {
        assert_eq!(darken(0.3, 0.7), 0.3);
        assert_eq!(lighten(0.3, 0.7), 0.7);
    }

    #[test]
    fn test_overlay_midpoint() {
        // b = 0.5 → boundary: 2*0.5*s or 1-2*(1-0.5)*(1-s)
        // both formulas give the same result at b==0.5
        let v = overlay(0.5, 0.5);
        assert!((v - 0.5).abs() < 1e-5);
    }
}
