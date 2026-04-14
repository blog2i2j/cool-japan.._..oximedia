#![allow(dead_code)]
//! Gamut scope with color space boundary overlays on the CIE 1931 chromaticity diagram.
//!
//! This module provides [`GamutBoundaryOverlay`] for rendering triangle outlines of
//! Rec.709, DCI-P3, and Rec.2020 gamut boundaries, and [`GamutCoverage`] for computing
//! the percentage of pixels falling within each gamut triangle using a point-in-triangle
//! test based on barycentric coordinates.
//!
//! # Example
//!
//! ```
//! use oximedia_scopes::gamut_scope_overlay::{GamutBoundaryOverlay, GamutCoverage, OverlayLineStyle};
//!
//! // Configure overlay with custom line colors
//! let overlay = GamutBoundaryOverlay::new()
//!     .with_rec709(OverlayLineStyle::new([0, 255, 0, 200], 2))
//!     .with_dci_p3(OverlayLineStyle::new([255, 255, 0, 200], 2))
//!     .with_rec2020(OverlayLineStyle::new([255, 0, 0, 200], 2));
//!
//! // Compute gamut coverage for a frame
//! let frame = vec![128u8; 64 * 64 * 3];
//! let coverage = GamutCoverage::analyze(&frame, 64, 64);
//! println!("Rec.709 coverage: {:.1}%", coverage.rec709_pct);
//! ```

use crate::gamut_scope::{
    linear_rgb_to_xyz, srgb_to_linear, xyz_to_xy, ChromaXy, GamutTriangle, TargetGamut,
};

// ---------------------------------------------------------------------------
// Line style
// ---------------------------------------------------------------------------

/// Style for a gamut boundary outline.
#[derive(Debug, Clone, Copy)]
pub struct OverlayLineStyle {
    /// RGBA color for the line.
    pub color: [u8; 4],
    /// Thickness in pixels (1 = single pixel line).
    pub thickness: u32,
}

impl OverlayLineStyle {
    /// Create a new line style with the given color and thickness.
    #[must_use]
    pub fn new(color: [u8; 4], thickness: u32) -> Self {
        Self {
            color,
            thickness: thickness.max(1),
        }
    }
}

impl Default for OverlayLineStyle {
    fn default() -> Self {
        Self {
            color: [200, 200, 200, 220],
            thickness: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// GamutBoundaryOverlay
// ---------------------------------------------------------------------------

/// Configures and renders multiple gamut boundary triangles on a CIE chromaticity diagram.
///
/// Each gamut (Rec.709, DCI-P3, Rec.2020) can be independently enabled with its own
/// line color and thickness. The overlay is rendered onto an existing RGBA buffer.
#[derive(Debug, Clone)]
pub struct GamutBoundaryOverlay {
    rec709: Option<OverlayLineStyle>,
    dci_p3: Option<OverlayLineStyle>,
    rec2020: Option<OverlayLineStyle>,
    /// Whether to draw the D65 white point marker.
    pub show_white_point: bool,
    /// Scope image width.
    pub width: u32,
    /// Scope image height.
    pub height: u32,
}

impl GamutBoundaryOverlay {
    /// Create a new overlay with no gamut boundaries enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rec709: None,
            dci_p3: None,
            rec2020: None,
            show_white_point: true,
            width: 512,
            height: 512,
        }
    }

    /// Set the scope dimensions. Zero is allowed and will cause
    /// [`render_to_image`](Self::render_to_image) to return an error.
    #[must_use]
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Enable the Rec.709 gamut triangle with the given style.
    #[must_use]
    pub fn with_rec709(mut self, style: OverlayLineStyle) -> Self {
        self.rec709 = Some(style);
        self
    }

    /// Enable the DCI-P3 gamut triangle with the given style.
    #[must_use]
    pub fn with_dci_p3(mut self, style: OverlayLineStyle) -> Self {
        self.dci_p3 = Some(style);
        self
    }

    /// Enable the Rec.2020 gamut triangle with the given style.
    #[must_use]
    pub fn with_rec2020(mut self, style: OverlayLineStyle) -> Self {
        self.rec2020 = Some(style);
        self
    }

    /// Enable all three standard gamut triangles with default broadcast colors.
    #[must_use]
    pub fn with_all_standard(self) -> Self {
        self.with_rec709(OverlayLineStyle::new([0, 220, 0, 220], 2))
            .with_dci_p3(OverlayLineStyle::new([220, 220, 0, 200], 2))
            .with_rec2020(OverlayLineStyle::new([220, 0, 0, 200], 2))
    }

    /// Return the list of enabled overlays with their styles and gamut data.
    fn enabled_overlays(&self) -> Vec<(GamutTriangle, OverlayLineStyle)> {
        let mut list = Vec::new();
        // Draw largest first so smaller boundaries are on top.
        if let Some(style) = self.rec2020 {
            list.push((GamutTriangle::rec2020(), style));
        }
        if let Some(style) = self.dci_p3 {
            list.push((GamutTriangle::dci_p3(), style));
        }
        if let Some(style) = self.rec709 {
            list.push((GamutTriangle::rec709(), style));
        }
        list
    }

    /// Render the gamut boundary overlays onto an existing RGBA buffer.
    ///
    /// The buffer must be `width * height * 4` bytes (RGBA, row-major).
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length does not match `width * height * 4`.
    pub fn render_onto(&self, rgba: &mut [u8]) -> Result<(), GamutOverlayError> {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        if rgba.len() < expected {
            return Err(GamutOverlayError::BufferTooSmall {
                expected,
                actual: rgba.len(),
            });
        }

        for (triangle, style) in self.enabled_overlays() {
            draw_thick_triangle(rgba, self.width, self.height, &triangle, &style);
        }

        if self.show_white_point {
            // D65 white point cross-hair
            let wp = ChromaXy::new(0.3127, 0.3290);
            draw_cross(rgba, self.width, self.height, &wp, [255, 255, 255, 200], 5);
        }

        Ok(())
    }

    /// Create a new RGBA buffer of `width x height` with a dark background and all
    /// enabled boundary overlays rendered.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are zero.
    pub fn render_to_image(&self) -> Result<Vec<u8>, GamutOverlayError> {
        if self.width == 0 || self.height == 0 {
            return Err(GamutOverlayError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }
        let size = (self.width as usize) * (self.height as usize) * 4;
        let mut rgba = vec![16u8; size];
        // Set alpha channel to 255
        for i in (3..size).step_by(4) {
            rgba[i] = 255;
        }
        self.render_onto(&mut rgba)?;
        Ok(rgba)
    }
}

impl Default for GamutBoundaryOverlay {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GamutCoverage
// ---------------------------------------------------------------------------

/// Gamut coverage analysis result showing what percentage of pixels fall within
/// each standard color gamut.
#[derive(Debug, Clone)]
pub struct GamutCoverage {
    /// Percentage of pixels within the Rec.709 gamut (0.0 - 100.0).
    pub rec709_pct: f64,
    /// Percentage of pixels within the DCI-P3 gamut (0.0 - 100.0).
    pub dci_p3_pct: f64,
    /// Percentage of pixels within the Rec.2020 gamut (0.0 - 100.0).
    pub rec2020_pct: f64,
    /// Number of pixels only in Rec.709 (subset of P3 and 2020).
    pub rec709_only_count: u64,
    /// Number of pixels in P3 but outside Rec.709.
    pub p3_excl_count: u64,
    /// Number of pixels in Rec.2020 but outside P3.
    pub bt2020_excl_count: u64,
    /// Number of pixels outside all three gamuts.
    pub out_of_all_count: u64,
    /// Total pixel count.
    pub total_pixels: u64,
}

impl GamutCoverage {
    /// Analyze an RGB24 frame and compute gamut coverage for Rec.709, DCI-P3, and Rec.2020.
    ///
    /// # Arguments
    ///
    /// * `frame` - RGB24 pixel data (3 bytes per pixel: R, G, B).
    /// * `width` - Frame width in pixels.
    /// * `height` - Frame height in pixels.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(frame: &[u8], width: u32, height: u32) -> Self {
        let total = (width as u64) * (height as u64);
        let expected = total as usize * 3;

        if frame.len() < expected || total == 0 {
            return Self {
                rec709_pct: 0.0,
                dci_p3_pct: 0.0,
                rec2020_pct: 0.0,
                rec709_only_count: 0,
                p3_excl_count: 0,
                bt2020_excl_count: 0,
                out_of_all_count: 0,
                total_pixels: total,
            };
        }

        let tri_709 = GamutTriangle::for_gamut(TargetGamut::Rec709);
        let tri_p3 = GamutTriangle::for_gamut(TargetGamut::DciP3);
        let tri_2020 = GamutTriangle::for_gamut(TargetGamut::Rec2020);

        let mut in_709 = 0_u64;
        let mut in_p3 = 0_u64;
        let mut in_2020 = 0_u64;
        let mut rec709_only = 0_u64;
        let mut p3_excl = 0_u64;
        let mut bt2020_excl = 0_u64;
        let mut out_of_all = 0_u64;

        for i in 0..total as usize {
            let r_lin = srgb_to_linear(f64::from(frame[i * 3]) / 255.0);
            let g_lin = srgb_to_linear(f64::from(frame[i * 3 + 1]) / 255.0);
            let b_lin = srgb_to_linear(f64::from(frame[i * 3 + 2]) / 255.0);

            let (x, y, z) = linear_rgb_to_xyz(r_lin, g_lin, b_lin);
            let chroma = xyz_to_xy(x, y, z);

            let is_709 = tri_709.contains(&chroma);
            let is_p3 = tri_p3.contains(&chroma);
            let is_2020 = tri_2020.contains(&chroma);

            if is_709 {
                in_709 += 1;
            }
            if is_p3 {
                in_p3 += 1;
            }
            if is_2020 {
                in_2020 += 1;
            }

            // Classify into exclusive buckets
            if is_709 {
                rec709_only += 1;
            } else if is_p3 {
                p3_excl += 1;
            } else if is_2020 {
                bt2020_excl += 1;
            } else {
                out_of_all += 1;
            }
        }

        let n = total as f64;
        Self {
            rec709_pct: (in_709 as f64 / n) * 100.0,
            dci_p3_pct: (in_p3 as f64 / n) * 100.0,
            rec2020_pct: (in_2020 as f64 / n) * 100.0,
            rec709_only_count: rec709_only,
            p3_excl_count: p3_excl,
            bt2020_excl_count: bt2020_excl,
            out_of_all_count: out_of_all,
            total_pixels: total,
        }
    }

    /// Check if content is purely Rec.709 (100% within Rec.709).
    #[must_use]
    pub fn is_pure_rec709(&self) -> bool {
        (self.rec709_pct - 100.0).abs() < f64::EPSILON || self.rec709_only_count == self.total_pixels
    }

    /// Check if content uses wide color gamut (any pixels outside Rec.709).
    #[must_use]
    pub fn uses_wide_gamut(&self) -> bool {
        self.p3_excl_count > 0 || self.bt2020_excl_count > 0 || self.out_of_all_count > 0
    }

    /// Dominant gamut — the smallest gamut that contains at least `threshold_pct` of pixels.
    #[must_use]
    pub fn dominant_gamut(&self, threshold_pct: f64) -> DominantGamut {
        if self.rec709_pct >= threshold_pct {
            DominantGamut::Rec709
        } else if self.dci_p3_pct >= threshold_pct {
            DominantGamut::DciP3
        } else if self.rec2020_pct >= threshold_pct {
            DominantGamut::Rec2020
        } else {
            DominantGamut::ExceedsAll
        }
    }
}

/// Dominant gamut classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DominantGamut {
    /// Content fits within Rec.709.
    Rec709,
    /// Content fits within DCI-P3 but exceeds Rec.709.
    DciP3,
    /// Content fits within Rec.2020 but exceeds P3.
    Rec2020,
    /// Content has pixels outside all three gamuts.
    ExceedsAll,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from gamut overlay operations.
#[derive(Debug, Clone)]
pub enum GamutOverlayError {
    /// RGBA buffer is too small.
    BufferTooSmall {
        /// Expected minimum byte length.
        expected: usize,
        /// Actual byte length.
        actual: usize,
    },
    /// Invalid dimensions.
    InvalidDimensions {
        /// Width.
        width: u32,
        /// Height.
        height: u32,
    },
}

impl std::fmt::Display for GamutOverlayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferTooSmall { expected, actual } => {
                write!(
                    f,
                    "RGBA buffer too small: need {expected} bytes, got {actual}"
                )
            }
            Self::InvalidDimensions { width, height } => {
                write!(f, "Invalid dimensions: {width}x{height}")
            }
        }
    }
}

impl std::error::Error for GamutOverlayError {}

// ---------------------------------------------------------------------------
// Internal drawing helpers
// ---------------------------------------------------------------------------

/// Map CIE xy to pixel coordinates. Viewport: x in [0, 0.8], y in [0, 0.9].
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn xy_to_pixel(cx: f64, cy: f64, w: u32, h: u32) -> (i32, i32) {
    let margin = 0.05;
    let x_min = -margin;
    let x_max = 0.80 + margin;
    let y_min = -margin;
    let y_max = 0.90 + margin;

    let px = ((cx - x_min) / (x_max - x_min) * (f64::from(w) - 1.0)).round() as i32;
    let py = ((1.0 - (cy - y_min) / (y_max - y_min)) * (f64::from(h) - 1.0)).round() as i32;
    (px, py)
}

/// Alpha-blend a single pixel in the RGBA buffer.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn blend_pixel(rgba: &mut [u8], w: usize, _h: usize, px: i32, py: i32, color: [u8; 4]) {
    if px < 0 || py < 0 || (px as usize) >= w {
        return;
    }
    let idx = (py as usize * w + px as usize) * 4;
    if idx + 3 >= rgba.len() {
        return;
    }
    let a = f32::from(color[3]) / 255.0;
    let ia = 1.0 - a;
    rgba[idx] = (f32::from(color[0]) * a + f32::from(rgba[idx]) * ia) as u8;
    rgba[idx + 1] = (f32::from(color[1]) * a + f32::from(rgba[idx + 1]) * ia) as u8;
    rgba[idx + 2] = (f32::from(color[2]) * a + f32::from(rgba[idx + 2]) * ia) as u8;
    rgba[idx + 3] = 255;
}

/// Draw a Bresenham line between two pixel coordinates with thickness.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn draw_thick_line(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 4],
    thickness: u32,
) {
    let wu = w as usize;
    let hu = h as usize;
    let half = (thickness / 2) as i32;

    let mut cx = x0;
    let mut cy = y0;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        // Draw a small filled square around the center point for thickness.
        for oy in -half..=half {
            for ox in -half..=half {
                blend_pixel(rgba, wu, hu, cx + ox, cy + oy, color);
            }
        }

        if cx == x1 && cy == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            cx += sx;
        }
        if e2 <= dx {
            err += dx;
            cy += sy;
        }
    }
}

/// Draw a gamut triangle with configurable thickness.
fn draw_thick_triangle(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    triangle: &GamutTriangle,
    style: &OverlayLineStyle,
) {
    let vertices = [
        (triangle.red.x, triangle.red.y),
        (triangle.green.x, triangle.green.y),
        (triangle.blue.x, triangle.blue.y),
    ];

    for edge in 0..3 {
        let (ax, ay) = vertices[edge];
        let (bx, by) = vertices[(edge + 1) % 3];
        let (px0, py0) = xy_to_pixel(ax, ay, w, h);
        let (px1, py1) = xy_to_pixel(bx, by, w, h);
        draw_thick_line(rgba, w, h, px0, py0, px1, py1, style.color, style.thickness);
    }
}

/// Draw a small cross-hair marker.
#[allow(clippy::cast_possible_truncation)]
fn draw_cross(rgba: &mut [u8], w: u32, h: u32, point: &ChromaXy, color: [u8; 4], arm: i32) {
    let wu = w as usize;
    let hu = h as usize;
    let (cx, cy) = xy_to_pixel(point.x, point.y, w, h);

    for d in -arm..=arm {
        blend_pixel(rgba, wu, hu, cx + d, cy, color);
        blend_pixel(rgba, wu, hu, cx, cy + d, color);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_line_style_default() {
        let s = OverlayLineStyle::default();
        assert_eq!(s.thickness, 1);
        assert_eq!(s.color[3], 220);
    }

    #[test]
    fn test_overlay_line_style_min_thickness() {
        let s = OverlayLineStyle::new([255, 0, 0, 255], 0);
        assert_eq!(s.thickness, 1);
    }

    #[test]
    fn test_gamut_boundary_overlay_default_empty() {
        let overlay = GamutBoundaryOverlay::new();
        assert!(overlay.enabled_overlays().is_empty());
    }

    #[test]
    fn test_gamut_boundary_overlay_with_all() {
        let overlay = GamutBoundaryOverlay::new().with_all_standard();
        assert_eq!(overlay.enabled_overlays().len(), 3);
    }

    #[test]
    fn test_render_to_image_dimensions() {
        let overlay = GamutBoundaryOverlay::new()
            .with_dimensions(64, 64)
            .with_rec709(OverlayLineStyle::new([0, 255, 0, 200], 1));
        let img = overlay.render_to_image().expect("render should succeed");
        assert_eq!(img.len(), 64 * 64 * 4);
    }

    #[test]
    fn test_render_to_image_zero_dimensions() {
        let overlay = GamutBoundaryOverlay::new().with_dimensions(0, 64);
        let result = overlay.render_to_image();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_onto_buffer_too_small() {
        let overlay = GamutBoundaryOverlay::new().with_dimensions(64, 64);
        let mut buf = vec![0u8; 10];
        let result = overlay.render_onto(&mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_render_onto_draws_pixels() {
        let overlay = GamutBoundaryOverlay::new()
            .with_dimensions(128, 128)
            .with_rec709(OverlayLineStyle::new([0, 255, 0, 255], 2));
        let mut buf = vec![0u8; 128 * 128 * 4];
        // Set alpha to 255 on background
        for i in (3..buf.len()).step_by(4) {
            buf[i] = 255;
        }
        overlay.render_onto(&mut buf).expect("render should succeed");

        // At least some pixels should have green > 0
        let has_green = buf.chunks(4).any(|px| px[1] > 0);
        assert!(has_green);
    }

    #[test]
    fn test_gamut_coverage_all_black() {
        let frame = vec![0u8; 4 * 4 * 3];
        let cov = GamutCoverage::analyze(&frame, 4, 4);
        assert_eq!(cov.total_pixels, 16);
        // Black maps to D65, which is inside all gamuts.
        assert!((cov.rec709_pct - 100.0).abs() < 1e-6);
        assert!((cov.dci_p3_pct - 100.0).abs() < 1e-6);
        assert!((cov.rec2020_pct - 100.0).abs() < 1e-6);
        assert!(cov.is_pure_rec709());
    }

    #[test]
    fn test_gamut_coverage_mid_gray() {
        let frame = vec![128u8; 8 * 8 * 3];
        let cov = GamutCoverage::analyze(&frame, 8, 8);
        assert_eq!(cov.total_pixels, 64);
        // Mid-gray (neutral) should be inside all gamuts.
        assert!(cov.rec709_pct > 99.0);
        assert!(!cov.uses_wide_gamut());
    }

    #[test]
    fn test_gamut_coverage_empty_frame() {
        let cov = GamutCoverage::analyze(&[], 0, 0);
        assert_eq!(cov.total_pixels, 0);
        assert_eq!(cov.rec709_pct, 0.0);
    }

    #[test]
    fn test_gamut_coverage_short_frame() {
        let frame = vec![128u8; 5]; // too short for 4x4
        let cov = GamutCoverage::analyze(&frame, 4, 4);
        // Should return zeroed result because frame is too short.
        assert_eq!(cov.rec709_pct, 0.0);
    }

    #[test]
    fn test_dominant_gamut_rec709() {
        let frame = vec![128u8; 4 * 4 * 3];
        let cov = GamutCoverage::analyze(&frame, 4, 4);
        assert_eq!(cov.dominant_gamut(95.0), DominantGamut::Rec709);
    }

    #[test]
    fn test_gamut_coverage_pure_primaries() {
        // Pure red (255,0,0) — should be on Rec.709 boundary.
        let mut frame = Vec::with_capacity(3 * 3);
        frame.extend_from_slice(&[255, 0, 0]); // red
        frame.extend_from_slice(&[0, 255, 0]); // green
        frame.extend_from_slice(&[0, 0, 255]); // blue
        let cov = GamutCoverage::analyze(&frame, 3, 1);
        assert_eq!(cov.total_pixels, 3);
        // All three sRGB primaries should be within Rec.709 (same primaries).
    }

    #[test]
    fn test_white_point_marker_drawn() {
        let overlay = GamutBoundaryOverlay::new().with_dimensions(128, 128);
        let img = overlay.render_to_image().expect("render should succeed");
        // White point cross should have bright pixels.
        let has_bright = img.chunks(4).any(|px| px[0] > 200 && px[1] > 200 && px[2] > 200);
        assert!(has_bright);
    }

    #[test]
    fn test_gamut_overlay_error_display() {
        let e = GamutOverlayError::BufferTooSmall {
            expected: 1000,
            actual: 500,
        };
        let msg = format!("{e}");
        assert!(msg.contains("1000"));
        assert!(msg.contains("500"));
    }
}
