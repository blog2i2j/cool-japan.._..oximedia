//! Color checker scope for analyzing captured color checker charts against reference values.
//!
//! This module provides tools for:
//!
//! - Defining reference color patches (e.g., X-Rite ColorChecker Classic 24-patch).
//! - Sampling corresponding patches from a captured RGB24 frame.
//! - Computing per-patch ΔE (CIE 2000 color difference) between captured and reference.
//! - Rendering a side-by-side comparison display showing captured vs. reference swatches
//!   with ΔE annotations.
//!
//! # Coordinate System
//!
//! Patch regions are specified as normalised rectangle `(x, y, w, h)` in `[0.0, 1.0]`
//! relative to the source frame dimensions.  This avoids hard-coding pixel addresses
//! and works for any input resolution.
//!
//! # ΔE 2000
//!
//! The CIE ΔE 2000 formula is implemented here with all five weighting terms
//! (`k_L = k_C = k_H = 1`).  Values below 1 are imperceptible, 1–3 are noticeable
//! to trained eyes, and above 5 indicate significant colour error.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Colour types
// ─────────────────────────────────────────────────────────────────────────────

/// An sRGB colour in linear [0.0, 1.0] per-channel form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearRgb {
    /// Red component [0.0, 1.0].
    pub r: f64,
    /// Green component [0.0, 1.0].
    pub g: f64,
    /// Blue component [0.0, 1.0].
    pub b: f64,
}

impl LinearRgb {
    /// Create from 8-bit sRGB values (applies sRGB gamma expansion).
    #[must_use]
    pub fn from_srgb8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: srgb_to_linear(r),
            g: srgb_to_linear(g),
            b: srgb_to_linear(b),
        }
    }

    /// Convert to CIE XYZ (D65 illuminant, sRGB primaries / BT.709 matrix).
    #[must_use]
    pub fn to_xyz(self) -> [f64; 3] {
        let r = self.r;
        let g = self.g;
        let b = self.b;
        [
            0.412_453 * r + 0.357_580 * g + 0.180_423 * b,
            0.212_671 * r + 0.715_160 * g + 0.072_169 * b,
            0.019_334 * r + 0.119_193 * g + 0.950_227 * b,
        ]
    }

    /// Convert to CIE L*a*b* (D65 illuminant).
    #[must_use]
    pub fn to_lab(self) -> [f64; 3] {
        let [x, y, z] = self.to_xyz();
        // D65 white point
        const WX: f64 = 0.950_456;
        const WY: f64 = 1.0;
        const WZ: f64 = 1.088_754;

        let fx = lab_f(x / WX);
        let fy = lab_f(y / WY);
        let fz = lab_f(z / WZ);

        let l = 116.0 * fy - 16.0;
        let a = 500.0 * (fx - fy);
        let b = 200.0 * (fy - fz);
        [l, a, b]
    }
}

/// sRGB gamma expansion: 8-bit → linear [0.0, 1.0].
fn srgb_to_linear(v: u8) -> f64 {
    let c = v as f64 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// CIE Lab `f(t)` helper.
fn lab_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA.powi(3) {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ΔE 2000
// ─────────────────────────────────────────────────────────────────────────────

/// Compute CIE ΔE 2000 between two Lab colours.
///
/// Reference: Sharma et al. (2005), "The CIEDE2000 Color-Difference Formula".
#[must_use]
pub fn delta_e2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;

    // Step 1: compute C' and h'
    let c_ab1 = (a1 * a1 + b1 * b1).sqrt();
    let c_ab2 = (a2 * a2 + b2 * b2).sqrt();
    let c_ab_bar = (c_ab1 + c_ab2) / 2.0;
    let c_ab_bar7 = c_ab_bar.powi(7);
    let g = 0.5 * (1.0 - (c_ab_bar7 / (c_ab_bar7 + 25_f64.powi(7))).sqrt());
    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();

    let h1p = b1.atan2(a1p).to_degrees().rem_euclid(360.0);
    let h2p = b2.atan2(a2p).to_degrees().rem_euclid(360.0);

    // Step 2: compute ΔL', ΔC', ΔH'
    let delta_l = l2 - l1;
    let delta_c = c2p - c1p;

    let delta_h_low = if c1p * c2p < 1e-10 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };
    let delta_h_big = 2.0 * (c1p * c2p).sqrt() * (delta_h_low.to_radians() / 2.0).sin();

    // Step 3: CIEDE2000 using weighting functions
    let l_bar = (l1 + l2) / 2.0;
    let c_bar = (c1p + c2p) / 2.0;

    let h_bar = if c1p * c2p < 1e-10 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };

    let t = 1.0
        - 0.17 * (h_bar - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar).to_radians().cos()
        + 0.32 * (3.0 * h_bar + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar - 63.0).to_radians().cos();

    let s_l = 1.0 + 0.015 * (l_bar - 50.0).powi(2) / (20.0 + (l_bar - 50.0).powi(2)).sqrt();
    let s_c = 1.0 + 0.045 * c_bar;
    let s_h = 1.0 + 0.015 * c_bar * t;

    let c_bar7 = c_bar.powi(7);
    let r_c = 2.0 * (c_bar7 / (c_bar7 + 25_f64.powi(7))).sqrt();
    let d_theta = 30.0 * (-((h_bar - 275.0) / 25.0).powi(2)).exp();
    let r_t = -r_c * (2.0 * d_theta.to_radians()).sin();

    let term_l = delta_l / s_l;
    let term_c = delta_c / s_c;
    let term_h = delta_h_big / s_h;

    (term_l * term_l + term_c * term_c + term_h * term_h + r_t * term_c * term_h).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Patch definition
// ─────────────────────────────────────────────────────────────────────────────

/// A single color checker patch with a label and reference sRGB value.
#[derive(Debug, Clone)]
pub struct CheckerPatch {
    /// Descriptive name for the patch (e.g., "Dark Skin", "Cyan").
    pub name: String,
    /// Reference colour in 8-bit sRGB.
    pub reference_srgb: [u8; 3],
    /// Normalised patch region `[x, y, width, height]` in the source frame
    /// (all values in `[0.0, 1.0]`).
    pub norm_rect: [f64; 4],
}

impl CheckerPatch {
    /// Create a new patch definition.
    #[must_use]
    pub fn new(name: impl Into<String>, reference_srgb: [u8; 3], norm_rect: [f64; 4]) -> Self {
        Self {
            name: name.into(),
            reference_srgb,
            norm_rect,
        }
    }
}

/// Returns the standard X-Rite ColorChecker Classic 24-patch layout for a
/// frame where the chart occupies `[x_offset, y_offset, chart_w, chart_h]`
/// (all in normalised `[0.0, 1.0]` frame coordinates).
///
/// The 24 patches are arranged in a 6×4 grid from left-to-right, top-to-bottom.
#[must_use]
pub fn colorchecker_classic_patches(
    x_offset: f64,
    y_offset: f64,
    chart_w: f64,
    chart_h: f64,
) -> Vec<CheckerPatch> {
    // Reference sRGB values for the X-Rite ColorChecker Classic 24 patches
    // (approximate; from published X-Rite data, linearised and re-gamma'd to sRGB 8-bit)
    const REFERENCE_SRGB: [[u8; 3]; 24] = [
        [115, 82, 68],   // 1 Dark Skin
        [194, 150, 130], // 2 Light Skin
        [98, 122, 157],  // 3 Blue Sky
        [87, 108, 67],   // 4 Foliage
        [133, 128, 177], // 5 Blue Flower
        [103, 189, 170], // 6 Bluish Green
        [214, 126, 44],  // 7 Orange
        [80, 91, 166],   // 8 Purplish Blue
        [193, 90, 99],   // 9 Moderate Red
        [94, 60, 108],   // 10 Purple
        [157, 188, 64],  // 11 Yellow Green
        [224, 163, 46],  // 12 Orange Yellow
        [56, 61, 150],   // 13 Blue
        [70, 148, 73],   // 14 Green
        [175, 54, 60],   // 15 Red
        [231, 199, 31],  // 16 Yellow
        [187, 86, 149],  // 17 Magenta
        [8, 133, 161],   // 18 Cyan
        [243, 243, 242], // 19 White
        [200, 200, 200], // 20 Neutral 8
        [160, 160, 160], // 21 Neutral 65
        [122, 122, 121], // 22 Neutral 5
        [85, 85, 85],    // 23 Neutral 35
        [52, 52, 52],    // 24 Black
    ];

    const NAMES: [&str; 24] = [
        "Dark Skin",
        "Light Skin",
        "Blue Sky",
        "Foliage",
        "Blue Flower",
        "Bluish Green",
        "Orange",
        "Purplish Blue",
        "Moderate Red",
        "Purple",
        "Yellow Green",
        "Orange Yellow",
        "Blue",
        "Green",
        "Red",
        "Yellow",
        "Magenta",
        "Cyan",
        "White",
        "Neutral 8",
        "Neutral 65",
        "Neutral 5",
        "Neutral 35",
        "Black",
    ];

    const COLS: usize = 6;
    const ROWS: usize = 4;

    let patch_w = chart_w / COLS as f64;
    let patch_h = chart_h / ROWS as f64;
    // Inner sampling rect: 50% of patch, centred
    let sample_w = patch_w * 0.5;
    let sample_h = patch_h * 0.5;

    let mut patches = Vec::with_capacity(24);
    for (i, (name, &srgb)) in NAMES.iter().zip(REFERENCE_SRGB.iter()).enumerate() {
        let col = i % COLS;
        let row = i / COLS;
        let patch_x = x_offset + col as f64 * patch_w;
        let patch_y = y_offset + row as f64 * patch_h;
        // Centre the sampling region within the patch
        let sample_x = patch_x + (patch_w - sample_w) / 2.0;
        let sample_y = patch_y + (patch_h - sample_h) / 2.0;

        patches.push(CheckerPatch::new(
            *name,
            srgb,
            [sample_x, sample_y, sample_w, sample_h],
        ));
    }
    patches
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Per-patch analysis result.
#[derive(Debug, Clone)]
pub struct PatchResult {
    /// Patch name.
    pub name: String,
    /// Mean sampled sRGB (8-bit) from the frame.
    pub sampled_srgb: [u8; 3],
    /// Reference sRGB (8-bit) from the patch definition.
    pub reference_srgb: [u8; 3],
    /// CIE ΔE 2000 colour difference between sampled and reference.
    pub delta_e: f64,
}

impl PatchResult {
    /// Returns `true` if the ΔE is below the given threshold.
    #[must_use]
    pub fn within_tolerance(&self, threshold: f64) -> bool {
        self.delta_e < threshold
    }
}

/// Full color checker analysis for one frame.
#[derive(Debug, Clone)]
pub struct ColorCheckerAnalysis {
    /// Per-patch results, in the same order as the input patches.
    pub patches: Vec<PatchResult>,
    /// Mean ΔE across all patches.
    pub mean_delta_e: f64,
    /// Maximum ΔE across all patches.
    pub max_delta_e: f64,
    /// Patch index with the worst ΔE.
    pub worst_patch_index: usize,
}

impl ColorCheckerAnalysis {
    /// Returns `true` if all patches have ΔE below `threshold`.
    #[must_use]
    pub fn all_within_tolerance(&self, threshold: f64) -> bool {
        self.patches.iter().all(|p| p.delta_e < threshold)
    }
}

/// Sample a frame and compute ΔE for each patch.
///
/// # Arguments
///
/// * `frame`  – RGB24 byte slice (`width * height * 3` bytes, row-major).
/// * `width`  – Frame width in pixels.
/// * `height` – Frame height in pixels.
/// * `patches` – Patch definitions with normalised sampling regions.
///
/// # Errors
///
/// Returns an error if the frame buffer is too small.
pub fn analyze_color_checker(
    frame: &[u8],
    width: u32,
    height: u32,
    patches: &[CheckerPatch],
) -> OxiResult<ColorCheckerAnalysis> {
    let total_pixels = (width as usize) * (height as usize);
    let expected = total_pixels * 3;
    if frame.len() < expected {
        return Err(OxiError::InvalidData(format!(
            "Frame too small: need {expected}, got {}",
            frame.len()
        )));
    }

    let mut patch_results: Vec<PatchResult> = Vec::with_capacity(patches.len());

    for patch in patches {
        let sampled = sample_patch(frame, width, height, &patch.norm_rect);
        let sampled_lab = LinearRgb::from_srgb8(sampled[0], sampled[1], sampled[2]).to_lab();
        let ref_lab =
            LinearRgb::from_srgb8(patch.reference_srgb[0], patch.reference_srgb[1], patch.reference_srgb[2])
                .to_lab();
        let de = delta_e2000(sampled_lab, ref_lab);

        patch_results.push(PatchResult {
            name: patch.name.clone(),
            sampled_srgb: sampled,
            reference_srgb: patch.reference_srgb,
            delta_e: de,
        });
    }

    let mean_delta_e = if patch_results.is_empty() {
        0.0
    } else {
        patch_results.iter().map(|p| p.delta_e).sum::<f64>() / patch_results.len() as f64
    };

    let (max_delta_e, worst_patch_index) = patch_results
        .iter()
        .enumerate()
        .fold((0.0f64, 0usize), |(max_de, max_idx), (i, p)| {
            if p.delta_e > max_de {
                (p.delta_e, i)
            } else {
                (max_de, max_idx)
            }
        });

    Ok(ColorCheckerAnalysis {
        patches: patch_results,
        mean_delta_e,
        max_delta_e,
        worst_patch_index,
    })
}

/// Sample the mean RGB colour from a normalised rectangular region of the frame.
fn sample_patch(frame: &[u8], width: u32, height: u32, norm_rect: &[f64; 4]) -> [u8; 3] {
    let w = width as usize;
    let h = height as usize;

    // Convert normalised rect to pixel coords, clamped to image bounds
    let x0 = (norm_rect[0] * w as f64).round() as usize;
    let y0 = (norm_rect[1] * h as f64).round() as usize;
    let rw = (norm_rect[2] * w as f64).round() as usize;
    let rh = (norm_rect[3] * h as f64).round() as usize;

    let x1 = (x0 + rw).min(w);
    let y1 = (y0 + rh).min(h);

    if x0 >= x1 || y0 >= y1 {
        return [0, 0, 0];
    }

    let mut sum_r = 0u64;
    let mut sum_g = 0u64;
    let mut sum_b = 0u64;
    let mut count = 0u64;

    for row in y0..y1 {
        for col in x0..x1 {
            let base = (row * w + col) * 3;
            if base + 2 < frame.len() {
                sum_r += u64::from(frame[base]);
                sum_g += u64::from(frame[base + 1]);
                sum_b += u64::from(frame[base + 2]);
                count += 1;
            }
        }
    }

    if count == 0 {
        return [0, 0, 0];
    }

    [
        (sum_r / count) as u8,
        (sum_g / count) as u8,
        (sum_b / count) as u8,
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the color checker scope renderer.
#[derive(Debug, Clone)]
pub struct ColorCheckerRenderConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Number of columns to arrange patches in.
    pub columns: usize,
    /// Whether to draw a ΔE indicator bar under each swatch pair.
    pub show_delta_bar: bool,
    /// ΔE threshold above which patches are highlighted as "out of tolerance".
    pub tolerance_threshold: f64,
}

impl Default for ColorCheckerRenderConfig {
    fn default() -> Self {
        Self {
            width: 720,
            height: 240,
            columns: 6,
            show_delta_bar: true,
            tolerance_threshold: 3.0,
        }
    }
}

/// Render a color checker analysis as an RGBA comparison image.
///
/// Each patch occupies two horizontal sub-swatches: left = reference colour,
/// right = sampled colour. A thin indicator below (if `show_delta_bar`)
/// transitions from green (low ΔE) to red (high ΔE).
///
/// # Errors
///
/// Returns an error if `analysis.patches` is empty.
pub fn render_color_checker_scope(
    analysis: &ColorCheckerAnalysis,
    config: &ColorCheckerRenderConfig,
) -> OxiResult<Vec<u8>> {
    if analysis.patches.is_empty() {
        return Err(OxiError::InvalidData(
            "No patches to render".into(),
        ));
    }

    let out_w = config.width as usize;
    let out_h = config.height as usize;
    let mut buf = vec![20u8; out_w * out_h * 4];
    // Set all alpha to 255
    for chunk in buf.chunks_exact_mut(4) {
        chunk[3] = 255;
    }

    let num_patches = analysis.patches.len();
    let cols = config.columns.max(1);
    let rows = (num_patches + cols - 1) / cols;

    let cell_w = out_w / cols;
    let cell_h = out_h.checked_div(rows).unwrap_or(out_h);
    let delta_bar_h = if config.show_delta_bar {
        (cell_h / 6).max(2)
    } else {
        0
    };
    let swatch_h = cell_h.saturating_sub(delta_bar_h);

    for (patch_idx, patch) in analysis.patches.iter().enumerate() {
        let col = patch_idx % cols;
        let row = patch_idx / cols;
        let cell_x = col * cell_w;
        let cell_y = row * cell_h;

        // Left half = reference, right half = sampled
        let half_w = cell_w / 2;

        // Draw reference swatch (left)
        fill_rect(
            &mut buf,
            out_w,
            cell_x,
            cell_y,
            half_w,
            swatch_h,
            [
                patch.reference_srgb[0],
                patch.reference_srgb[1],
                patch.reference_srgb[2],
                255,
            ],
        );

        // Draw sampled swatch (right)
        fill_rect(
            &mut buf,
            out_w,
            cell_x + half_w,
            cell_y,
            cell_w - half_w,
            swatch_h,
            [
                patch.sampled_srgb[0],
                patch.sampled_srgb[1],
                patch.sampled_srgb[2],
                255,
            ],
        );

        // ΔE indicator bar
        if config.show_delta_bar && delta_bar_h > 0 {
            let bar_y = cell_y + swatch_h;
            // Normalise ΔE to colour: 0 = green, tolerance_threshold = yellow, 2× threshold = red
            let t = (patch.delta_e / (config.tolerance_threshold * 2.0)).clamp(0.0, 1.0);
            let r = (t * 255.0) as u8;
            let g = ((1.0 - t) * 255.0) as u8;
            fill_rect(
                &mut buf,
                out_w,
                cell_x,
                bar_y,
                cell_w,
                delta_bar_h,
                [r, g, 0, 255],
            );
        }
    }

    Ok(buf)
}

/// Fill a rectangular region of an RGBA buffer with a solid colour.
fn fill_rect(buf: &mut [u8], buf_w: usize, x: usize, y: usize, w: usize, h: usize, color: [u8; 4]) {
    for row in y..(y + h) {
        for col in x..(x + w) {
            let idx = (row * buf_w + col) * 4;
            if idx + 3 < buf.len() {
                buf[idx] = color[0];
                buf[idx + 1] = color[1];
                buf[idx + 2] = color[2];
                buf[idx + 3] = color[3];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Solid RGB24 frame (all pixels the same colour).
    fn solid_frame(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        vec![[r, g, b]; (w * h) as usize]
            .into_iter()
            .flatten()
            .collect()
    }

    #[test]
    fn test_delta_e2000_identical_colours_is_zero() {
        let lab = [50.0, 10.0, -10.0];
        let de = delta_e2000(lab, lab);
        assert!(de < 1e-6, "ΔE for identical colours = {de}");
    }

    #[test]
    fn test_delta_e2000_black_vs_white_is_large() {
        let black = LinearRgb::from_srgb8(0, 0, 0).to_lab();
        let white = LinearRgb::from_srgb8(255, 255, 255).to_lab();
        let de = delta_e2000(black, white);
        assert!(de > 50.0, "Black vs white ΔE should be large, got {de}");
    }

    #[test]
    fn test_srgb_to_linear_black() {
        assert!(srgb_to_linear(0) < 1e-10);
    }

    #[test]
    fn test_srgb_to_linear_white() {
        assert!((srgb_to_linear(255) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_analyze_frame_too_small() {
        let frame = vec![0u8; 5];
        let patches = colorchecker_classic_patches(0.0, 0.0, 1.0, 1.0);
        let result = analyze_color_checker(&frame, 10, 10, &patches);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_color_checker_all_white() {
        // A white frame sampled against dark skin reference will have a large ΔE
        let frame = solid_frame(64, 64, 255, 255, 255);
        let patches = colorchecker_classic_patches(0.0, 0.0, 1.0, 1.0);
        let result = analyze_color_checker(&frame, 64, 64, &patches).expect("ok");
        assert_eq!(result.patches.len(), 24);
        // Mean ΔE should be significantly non-zero
        assert!(result.mean_delta_e > 1.0, "mean ΔE={}", result.mean_delta_e);
    }

    #[test]
    fn test_analyze_color_checker_matches_reference_exactly() {
        // Patch with 100% coverage of the reference colour of patch 0 (Dark Skin = 115,82,68)
        let ref_color = [115u8, 82, 68];
        let frame = solid_frame(32, 32, ref_color[0], ref_color[1], ref_color[2]);
        let patches = vec![CheckerPatch::new("Dark Skin", ref_color, [0.0, 0.0, 1.0, 1.0])];
        let result = analyze_color_checker(&frame, 32, 32, &patches).expect("ok");
        assert_eq!(result.patches.len(), 1);
        // ΔE should be very small (sampled colour ≈ reference colour)
        assert!(
            result.patches[0].delta_e < 2.0,
            "ΔE = {}",
            result.patches[0].delta_e
        );
    }

    #[test]
    fn test_colorchecker_classic_patches_count() {
        let patches = colorchecker_classic_patches(0.0, 0.0, 1.0, 1.0);
        assert_eq!(patches.len(), 24);
    }

    #[test]
    fn test_patch_within_tolerance() {
        let p = PatchResult {
            name: "test".into(),
            sampled_srgb: [128, 128, 128],
            reference_srgb: [128, 128, 128],
            delta_e: 1.5,
        };
        assert!(p.within_tolerance(2.0));
        assert!(!p.within_tolerance(1.0));
    }

    #[test]
    fn test_render_color_checker_scope_dimensions() {
        let frame = solid_frame(64, 64, 128, 128, 128);
        let patches = colorchecker_classic_patches(0.0, 0.0, 1.0, 1.0);
        let analysis = analyze_color_checker(&frame, 64, 64, &patches).expect("ok");
        let config = ColorCheckerRenderConfig {
            width: 720,
            height: 240,
            ..Default::default()
        };
        let result = render_color_checker_scope(&analysis, &config).expect("ok");
        assert_eq!(result.len(), 720 * 240 * 4);
    }

    #[test]
    fn test_render_color_checker_no_patches_error() {
        let analysis = ColorCheckerAnalysis {
            patches: vec![],
            mean_delta_e: 0.0,
            max_delta_e: 0.0,
            worst_patch_index: 0,
        };
        let config = ColorCheckerRenderConfig::default();
        assert!(render_color_checker_scope(&analysis, &config).is_err());
    }

    #[test]
    fn test_all_within_tolerance() {
        let patches = vec![
            PatchResult {
                name: "a".into(),
                sampled_srgb: [0, 0, 0],
                reference_srgb: [0, 0, 0],
                delta_e: 0.5,
            },
            PatchResult {
                name: "b".into(),
                sampled_srgb: [255, 255, 255],
                reference_srgb: [255, 255, 255],
                delta_e: 0.8,
            },
        ];
        let analysis = ColorCheckerAnalysis {
            mean_delta_e: 0.65,
            max_delta_e: 0.8,
            worst_patch_index: 1,
            patches,
        };
        assert!(analysis.all_within_tolerance(1.0));
        assert!(!analysis.all_within_tolerance(0.6));
    }
}
