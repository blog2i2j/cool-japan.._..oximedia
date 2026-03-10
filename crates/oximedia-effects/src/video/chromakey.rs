//! Chroma key / green screen removal with color spill suppression.
//!
//! Provides a complete chroma-key pipeline:
//! - Configurable key color (RGB), similarity threshold, smoothness, and spill factor
//! - `apply_chroma_key` – alpha-composites the keyed subject over a background frame
//! - `detect_key_color` – heuristically auto-detects the dominant background color

#![allow(dead_code)]

use super::{clamp_u8, validate_buffer, PixelFormat, VideoResult};
use crate::EffectError;

// ── Parameters ────────────────────────────────────────────────────────────────

/// Parameters for the chroma-key effect.
#[derive(Debug, Clone, Copy)]
pub struct ChromaKeyParams {
    /// Key color as (R, G, B) in 0–255.
    pub key_color: [u8; 3],
    /// Similarity threshold [0.0, 1.0]: pixels whose color distance from
    /// `key_color` is below this value are made fully transparent.
    pub similarity: f32,
    /// Smoothness [0.0, 1.0]: width of the semi-transparent transition zone.
    pub smoothness: f32,
    /// Spill suppression factor [0.0, 1.0]: 0 = no suppression, 1 = maximum.
    pub spill_factor: f32,
}

impl Default for ChromaKeyParams {
    fn default() -> Self {
        Self {
            key_color: [0, 255, 0], // pure green
            similarity: 0.30,
            smoothness: 0.10,
            spill_factor: 0.50,
        }
    }
}

impl ChromaKeyParams {
    /// Green-screen preset.
    #[must_use]
    pub const fn green_screen() -> Self {
        Self {
            key_color: [0, 255, 0],
            similarity: 0.30,
            smoothness: 0.10,
            spill_factor: 0.50,
        }
    }

    /// Blue-screen preset.
    #[must_use]
    pub const fn blue_screen() -> Self {
        Self {
            key_color: [0, 0, 255],
            similarity: 0.30,
            smoothness: 0.10,
            spill_factor: 0.50,
        }
    }
}

// ── Core functions ─────────────────────────────────────────────────────────────

/// Apply chroma-key to `foreground` and composite the result over `background`.
///
/// Both buffers must be RGBA (4 bytes per pixel).  The result is written into
/// `output`, which must have the same dimensions.
///
/// # Errors
///
/// Returns [`EffectError::BufferSizeMismatch`] if any buffer has the wrong size.
#[allow(clippy::too_many_arguments)]
pub fn apply_chroma_key(
    foreground: &[u8],
    background: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    params: &ChromaKeyParams,
) -> VideoResult<()> {
    validate_buffer(foreground, width, height, PixelFormat::Rgba)?;
    validate_buffer(background, width, height, PixelFormat::Rgba)?;
    validate_buffer(output, width, height, PixelFormat::Rgba)?;

    let similarity = params.similarity.clamp(0.0, 1.0);
    let smoothness = params.smoothness.clamp(0.0, 1.0);
    let spill = params.spill_factor.clamp(0.0, 1.0);

    let kr = f32::from(params.key_color[0]) / 255.0;
    let kg = f32::from(params.key_color[1]) / 255.0;
    let kb = f32::from(params.key_color[2]) / 255.0;

    for i in 0..width * height {
        let base = i * 4;

        // Foreground pixel (normalised)
        let fr = f32::from(foreground[base]) / 255.0;
        let fg = f32::from(foreground[base + 1]) / 255.0;
        let fb = f32::from(foreground[base + 2]) / 255.0;

        // Euclidean distance in RGB space
        let dist =
            ((fr - kr).powi(2) + (fg - kg).powi(2) + (fb - kb).powi(2)).sqrt() / 3.0_f32.sqrt();

        // Alpha: 0 inside key zone, ramps to 1 in the smoothness band
        let alpha = if dist < similarity {
            0.0_f32
        } else if dist < similarity + smoothness {
            (dist - similarity) / smoothness.max(1e-6)
        } else {
            1.0_f32
        };

        // Spill suppression on partially or fully transparent pixels
        let (mut or, mut og, mut ob) = (fr, fg, fb);
        if spill > 0.0 && alpha < 1.0 {
            // Identify dominant key channel and suppress it
            let max_key = kr.max(kg).max(kb);
            #[allow(clippy::float_cmp)]
            if max_key == kg {
                // Green-dominant key: cap green channel
                let neighbors_avg = (or + ob) / 2.0;
                og = og - (og - neighbors_avg).max(0.0) * spill * (1.0 - alpha);
            } else if max_key == kb {
                // Blue-dominant key
                let neighbors_avg = (or + og) / 2.0;
                ob = ob - (ob - neighbors_avg).max(0.0) * spill * (1.0 - alpha);
            } else {
                // Red-dominant key
                let neighbors_avg = (og + ob) / 2.0;
                or = or - (or - neighbors_avg).max(0.0) * spill * (1.0 - alpha);
            }
        }

        // Composite over background
        let bg_r = f32::from(background[base]) / 255.0;
        let bg_g = f32::from(background[base + 1]) / 255.0;
        let bg_b = f32::from(background[base + 2]) / 255.0;
        let bg_a = f32::from(background[base + 3]) / 255.0;

        let inv_alpha = 1.0 - alpha;
        let out_a = alpha + bg_a * inv_alpha;

        if out_a < 1e-6 {
            output[base] = 0;
            output[base + 1] = 0;
            output[base + 2] = 0;
            output[base + 3] = 0;
        } else {
            output[base] = clamp_u8((or * alpha + bg_r * bg_a * inv_alpha) / out_a * 255.0);
            output[base + 1] = clamp_u8((og * alpha + bg_g * bg_a * inv_alpha) / out_a * 255.0);
            output[base + 2] = clamp_u8((ob * alpha + bg_b * bg_a * inv_alpha) / out_a * 255.0);
            output[base + 3] = clamp_u8(out_a * 255.0);
        }
    }

    Ok(())
}

/// Heuristically detect the dominant background color in an RGBA frame.
///
/// Samples pixels around the frame border and returns the mean RGB value.
///
/// # Errors
///
/// Returns [`EffectError::InsufficientBuffer`] if the frame is too small (< 2×2).
pub fn detect_key_color(data: &[u8], width: usize, height: usize) -> VideoResult<[u8; 3]> {
    if width < 2 || height < 2 {
        return Err(EffectError::InsufficientBuffer {
            required: 16,
            actual: data.len(),
        });
    }
    validate_buffer(data, width, height, PixelFormat::Rgba)?;

    let mut sum_r = 0u64;
    let mut sum_g = 0u64;
    let mut sum_b = 0u64;
    let mut count = 0u64;

    // Top and bottom rows
    for x in 0..width {
        for &row in &[0usize, height - 1] {
            let base = (row * width + x) * 4;
            sum_r += u64::from(data[base]);
            sum_g += u64::from(data[base + 1]);
            sum_b += u64::from(data[base + 2]);
            count += 1;
        }
    }
    // Left and right columns (skip corners already counted)
    for y in 1..height - 1 {
        for &col in &[0usize, width - 1] {
            let base = (y * width + col) * 4;
            sum_r += u64::from(data[base]);
            sum_g += u64::from(data[base + 1]);
            sum_b += u64::from(data[base + 2]);
            count += 1;
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    Ok([
        (sum_r / count) as u8,
        (sum_g / count) as u8,
        (sum_b / count) as u8,
    ])
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: usize, h: usize, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        vec![[r, g, b, a]; w * h].into_iter().flatten().collect()
    }

    #[test]
    fn test_default_params_green_screen() {
        let p = ChromaKeyParams::default();
        assert_eq!(p.key_color, [0, 255, 0]);
        assert!(p.similarity > 0.0);
        assert!(p.smoothness > 0.0);
    }

    #[test]
    fn test_green_screen_preset() {
        let p = ChromaKeyParams::green_screen();
        assert_eq!(p.key_color, [0, 255, 0]);
    }

    #[test]
    fn test_blue_screen_preset() {
        let p = ChromaKeyParams::blue_screen();
        assert_eq!(p.key_color, [0, 0, 255]);
    }

    #[test]
    fn test_apply_chroma_key_keys_out_green() {
        let fg = solid_rgba(4, 4, 0, 255, 0, 255); // pure green fg
        let bg = solid_rgba(4, 4, 200, 0, 0, 255); // red bg
        let mut out = vec![0u8; 4 * 4 * 4];
        let params = ChromaKeyParams::green_screen();
        apply_chroma_key(&fg, &bg, &mut out, 4, 4, &params).expect("test expectation failed");
        // Alpha should be ~0: green was keyed → background shows through
        for px in out.chunks_exact(4) {
            // Background red channel should dominate
            assert!(px[0] > 100, "Expected red bg, got {}", px[0]);
        }
    }

    #[test]
    fn test_apply_chroma_key_keeps_non_key_color() {
        let fg = solid_rgba(4, 4, 255, 0, 0, 255); // red fg (not key color)
        let bg = solid_rgba(4, 4, 0, 0, 200, 255); // blue bg
        let mut out = vec![0u8; 4 * 4 * 4];
        let params = ChromaKeyParams::green_screen();
        apply_chroma_key(&fg, &bg, &mut out, 4, 4, &params).expect("test expectation failed");
        // Non-key pixels should be opaque (fg shows through)
        for px in out.chunks_exact(4) {
            assert_eq!(px[3], 255, "Alpha should be 255 for non-key pixel");
            assert!(px[0] > 200, "Red channel should dominate");
        }
    }

    #[test]
    fn test_apply_chroma_key_buffer_mismatch() {
        let fg = vec![0u8; 10]; // wrong size
        let bg = solid_rgba(4, 4, 0, 0, 0, 255);
        let mut out = vec![0u8; 4 * 4 * 4];
        let result = apply_chroma_key(&fg, &bg, &mut out, 4, 4, &ChromaKeyParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_key_color_uniform_green() {
        let data = solid_rgba(8, 8, 0, 200, 0, 255);
        let color = detect_key_color(&data, 8, 8).expect("color should be valid");
        // Should detect approximately green
        assert!(color[1] > color[0], "Green channel should dominate");
        assert!(
            color[1] > color[2],
            "Green channel should dominate over blue"
        );
    }

    #[test]
    fn test_detect_key_color_too_small() {
        let data = vec![0u8; 4]; // 1x1
        let result = detect_key_color(&data, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_key_color_blue_border() {
        // Border is blue, interior is white
        let w = 6usize;
        let h = 6usize;
        let mut data = vec![255u8; w * h * 4]; // white
                                               // Set alpha on all to 255
        for i in 0..w * h {
            data[i * 4 + 3] = 255;
        }
        // Overwrite border with blue
        for x in 0..w {
            for &y in &[0usize, h - 1] {
                let base = (y * w + x) * 4;
                data[base] = 0;
                data[base + 1] = 0;
                data[base + 2] = 200;
            }
        }
        for y in 1..h - 1 {
            for &x in &[0usize, w - 1] {
                let base = (y * w + x) * 4;
                data[base] = 0;
                data[base + 1] = 0;
                data[base + 2] = 200;
            }
        }
        let color = detect_key_color(&data, w, h).expect("color should be valid");
        assert!(color[2] > color[0], "Blue should dominate border");
    }
}
