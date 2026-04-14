//! 360° stitch quality assessment.
//!
//! This module provides algorithms for evaluating the quality of panoramic
//! stitching along the seams between adjacent camera images:
//!
//! * **Seam visibility scoring** — measures the photometric discontinuity at
//!   the vertical seam in an equirectangular image by comparing a narrow band
//!   on each side.
//! * **Colour mismatch detection** — computes mean absolute colour difference
//!   across a user-defined seam stripe, reporting per-channel and aggregate
//!   mismatch values.
//! * **Parallax artifact detection** — detects ghosting / double-edge artefacts
//!   by analysing the local gradient field across the seam and comparing the
//!   dominant edge directions on each side.
//! * [`StitchReport`] — a structured summary of all three metrics for a
//!   complete stitch quality assessment pass.
//!
//! ## Coordinate convention
//!
//! All pixel buffers are assumed to be **RGB, 3 bytes per pixel, row-major**.
//! The seam position is expressed as a column index within the equirectangular
//! image.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_360::stitching_quality::{
//!     SeamVisibilityAnalyser, ColourMismatch, StitchReport,
//! };
//!
//! // Create a 128×64 solid-colour image (for demo purposes)
//! let img = vec![128u8; 128 * 64 * 3];
//! let analyser = SeamVisibilityAnalyser::new(128, 64, 5).expect("valid");
//! let score = analyser.seam_score(&img, 64).expect("ok");
//! println!("seam score: {:.4}", score);
//! ```

use crate::VrError;

// ─── SeamVisibilityAnalyser ───────────────────────────────────────────────────

/// Analyses the photometric discontinuity along a vertical seam in an
/// equirectangular image.
///
/// The analyser compares a strip of `half_width` columns on each side of the
/// seam column and computes the mean absolute difference (MAD) normalised to
/// `[0, 1]`.  Higher scores indicate more visible seams.
#[derive(Debug, Clone)]
pub struct SeamVisibilityAnalyser {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Half-width of the comparison strip (columns on each side of the seam).
    pub half_width: u32,
}

impl SeamVisibilityAnalyser {
    /// Create a new analyser.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if `width`, `height`, or
    /// `half_width` is zero, or if `half_width` ≥ `width / 2`.
    pub fn new(width: u32, height: u32, half_width: u32) -> Result<Self, VrError> {
        if width == 0 || height == 0 {
            return Err(VrError::InvalidDimensions(
                "width and height must be > 0".into(),
            ));
        }
        if half_width == 0 {
            return Err(VrError::InvalidDimensions("half_width must be > 0".into()));
        }
        if half_width >= width / 2 {
            return Err(VrError::InvalidDimensions(
                "half_width must be < width/2".into(),
            ));
        }
        Ok(Self {
            width,
            height,
            half_width,
        })
    }

    /// Compute the seam visibility score in `[0, 1]` at the given seam column
    /// index.
    ///
    /// A score near 0 indicates an invisible seam (left and right sides match
    /// perfectly).  A score near 1 indicates a maximally visible seam.
    ///
    /// The score is the mean absolute pixel difference across the comparison
    /// strip, normalised by 255.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if `seam_col` is out of bounds.
    /// Returns [`VrError::BufferTooSmall`] if `pixels` is too small.
    pub fn seam_score(&self, pixels: &[u8], seam_col: u32) -> Result<f32, VrError> {
        self.validate_buffer(pixels)?;
        if seam_col == 0 || seam_col >= self.width {
            return Err(VrError::InvalidDimensions(format!(
                "seam_col {seam_col} out of range [1, {})",
                self.width
            )));
        }

        // Columns to compare: [seam_col - half_width .. seam_col)  vs
        //                      [seam_col .. seam_col + half_width)
        let left_start = seam_col.saturating_sub(self.half_width);
        let right_end = (seam_col + self.half_width).min(self.width);
        let strip_cols = (seam_col - left_start).min(right_end - seam_col);
        if strip_cols == 0 {
            return Ok(0.0);
        }

        let mut total_diff = 0u64;
        let mut count = 0u64;

        for row in 0..self.height {
            for offset in 0..strip_cols {
                let lc = seam_col - strip_cols + offset;
                let rc = seam_col + offset;
                for ch in 0..3usize {
                    let lv = self.get_pixel(pixels, row, lc, ch);
                    let rv = self.get_pixel(pixels, row, rc, ch);
                    total_diff += (lv as i32 - rv as i32).unsigned_abs() as u64;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return Ok(0.0);
        }
        Ok(total_diff as f32 / (count as f32 * 255.0))
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    fn validate_buffer(&self, pixels: &[u8]) -> Result<(), VrError> {
        let expected = self.width as usize * self.height as usize * 3;
        if pixels.len() < expected {
            return Err(VrError::BufferTooSmall {
                expected,
                got: pixels.len(),
            });
        }
        Ok(())
    }

    #[inline]
    fn get_pixel(&self, pixels: &[u8], row: u32, col: u32, channel: usize) -> u8 {
        let idx = (row as usize * self.width as usize + col as usize) * 3 + channel;
        pixels.get(idx).copied().unwrap_or(0)
    }
}

// ─── ColourMismatch ───────────────────────────────────────────────────────────

/// Per-channel and aggregate colour mismatch statistics across a seam stripe.
#[derive(Debug, Clone, PartialEq)]
pub struct ColourMismatch {
    /// Mean absolute difference for the red channel, normalised to `[0, 1]`.
    pub red_mad: f32,
    /// Mean absolute difference for the green channel, normalised to `[0, 1]`.
    pub green_mad: f32,
    /// Mean absolute difference for the blue channel, normalised to `[0, 1]`.
    pub blue_mad: f32,
    /// Aggregate (mean of R/G/B) MAD.
    pub aggregate_mad: f32,
}

impl ColourMismatch {
    /// Compute the colour mismatch across the seam in the given equirectangular
    /// image.
    ///
    /// `stripe_width` specifies how many columns on each side of `seam_col` to
    /// include in the comparison.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if dimensions or `seam_col` are
    /// invalid.  Returns [`VrError::BufferTooSmall`] if `pixels` is too small.
    pub fn compute(
        pixels: &[u8],
        width: u32,
        height: u32,
        seam_col: u32,
        stripe_width: u32,
    ) -> Result<Self, VrError> {
        if width == 0 || height == 0 {
            return Err(VrError::InvalidDimensions(
                "width and height must be > 0".into(),
            ));
        }
        if stripe_width == 0 {
            return Err(VrError::InvalidDimensions(
                "stripe_width must be > 0".into(),
            ));
        }
        if seam_col == 0 || seam_col >= width {
            return Err(VrError::InvalidDimensions(format!(
                "seam_col {seam_col} out of range [1, {width})"
            )));
        }
        let expected = width as usize * height as usize * 3;
        if pixels.len() < expected {
            return Err(VrError::BufferTooSmall {
                expected,
                got: pixels.len(),
            });
        }

        let strip = stripe_width.min(seam_col).min(width - seam_col);
        if strip == 0 {
            return Ok(Self {
                red_mad: 0.0,
                green_mad: 0.0,
                blue_mad: 0.0,
                aggregate_mad: 0.0,
            });
        }

        let mut sums = [0u64; 3];
        let mut count = 0u64;

        let w = width as usize;
        for row in 0..height as usize {
            for off in 0..strip as usize {
                let lc = (seam_col as usize - strip as usize) + off;
                let rc = seam_col as usize + off;
                for ch in 0..3usize {
                    let lv = pixels[row * w * 3 + lc * 3 + ch];
                    let rv = pixels[row * w * 3 + rc * 3 + ch];
                    sums[ch] += (lv as i32 - rv as i32).unsigned_abs() as u64;
                }
                count += 1;
            }
        }

        if count == 0 {
            return Ok(Self {
                red_mad: 0.0,
                green_mad: 0.0,
                blue_mad: 0.0,
                aggregate_mad: 0.0,
            });
        }

        let norm = count as f32 * 255.0;
        let red_mad = sums[0] as f32 / norm;
        let green_mad = sums[1] as f32 / norm;
        let blue_mad = sums[2] as f32 / norm;
        let aggregate_mad = (red_mad + green_mad + blue_mad) / 3.0;

        Ok(Self {
            red_mad,
            green_mad,
            blue_mad,
            aggregate_mad,
        })
    }
}

// ─── ParallaxDetector ─────────────────────────────────────────────────────────

/// Detects parallax artifacts across a stitching seam by analysing horizontal
/// gradient magnitude asymmetry and edge-direction divergence near the seam.
///
/// Parallax manifests as strong horizontal gradients of opposite sign on either
/// side of the seam, indicating content that is spatially misaligned (ghosting,
/// double-edges).
#[derive(Debug, Clone)]
pub struct ParallaxDetector {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of columns on each side of the seam to include in analysis.
    pub analysis_radius: u32,
}

/// Result of a parallax artifact analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct ParallaxReport {
    /// Mean absolute horizontal gradient magnitude on the **left** side of the
    /// seam, normalised to `[0, 1]`.
    pub left_grad_mag: f32,
    /// Mean absolute horizontal gradient magnitude on the **right** side of the
    /// seam, normalised to `[0, 1]`.
    pub right_grad_mag: f32,
    /// Asymmetry score `|left - right| / max(left, right, ε)` in `[0, 1]`.
    /// High asymmetry suggests parallax-induced misalignment.
    pub gradient_asymmetry: f32,
    /// Fraction of analysed rows where the horizontal gradient sign flips
    /// between the left and right sides — another indicator of parallax.
    pub sign_flip_fraction: f32,
}

impl ParallaxDetector {
    /// Create a new detector.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if any parameter is zero or
    /// `analysis_radius >= width / 2`.
    pub fn new(width: u32, height: u32, analysis_radius: u32) -> Result<Self, VrError> {
        if width == 0 || height == 0 || analysis_radius == 0 {
            return Err(VrError::InvalidDimensions(
                "dimensions and analysis_radius must be > 0".into(),
            ));
        }
        if analysis_radius >= width / 2 {
            return Err(VrError::InvalidDimensions(
                "analysis_radius must be < width/2".into(),
            ));
        }
        Ok(Self {
            width,
            height,
            analysis_radius,
        })
    }

    /// Analyse parallax artifacts at `seam_col`.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if `seam_col` is out of bounds.
    /// Returns [`VrError::BufferTooSmall`] if `pixels` is too small.
    pub fn analyse(&self, pixels: &[u8], seam_col: u32) -> Result<ParallaxReport, VrError> {
        let expected = self.width as usize * self.height as usize * 3;
        if pixels.len() < expected {
            return Err(VrError::BufferTooSmall {
                expected,
                got: pixels.len(),
            });
        }
        if seam_col < self.analysis_radius || seam_col + self.analysis_radius >= self.width {
            return Err(VrError::InvalidDimensions(format!(
                "seam_col {seam_col} too close to boundary for analysis_radius {}",
                self.analysis_radius
            )));
        }

        let mut left_sum = 0.0f64;
        let mut right_sum = 0.0f64;
        let mut sign_flips = 0u32;
        let mut total_rows = 0u32;

        let w = self.width as usize;
        for row in 0..self.height as usize {
            // Horizontal gradient at seam boundary: difference between the
            // pixel just right of the seam and just left (for each side).
            let lc = seam_col as usize - 1;
            let rc = seam_col as usize;

            // Left-side gradient: difference across the two pixels straddling seam from left
            let left_lum_left = luma(pixels, row, lc.saturating_sub(1), w);
            let left_lum_right = luma(pixels, row, lc, w);
            let left_grad = left_lum_right - left_lum_left;

            // Right-side gradient: difference across the two pixels straddling seam from right
            let right_lum_left = luma(pixels, row, rc, w);
            let right_lum_right = luma(pixels, row, (rc + 1).min(w - 1), w);
            let right_grad = right_lum_right - right_lum_left;

            left_sum += left_grad.abs() as f64;
            right_sum += right_grad.abs() as f64;

            // Sign flip: left and right gradients point in opposite directions
            if left_grad * right_grad < 0.0 {
                sign_flips += 1;
            }
            total_rows += 1;
        }

        if total_rows == 0 {
            return Ok(ParallaxReport {
                left_grad_mag: 0.0,
                right_grad_mag: 0.0,
                gradient_asymmetry: 0.0,
                sign_flip_fraction: 0.0,
            });
        }

        let norm = total_rows as f64 * 255.0;
        let left_grad_mag = (left_sum / norm).clamp(0.0, 1.0) as f32;
        let right_grad_mag = (right_sum / norm).clamp(0.0, 1.0) as f32;

        let max_grad = left_grad_mag.max(right_grad_mag).max(f32::EPSILON);
        let gradient_asymmetry = (left_grad_mag - right_grad_mag).abs() / max_grad;
        let sign_flip_fraction = sign_flips as f32 / total_rows as f32;

        Ok(ParallaxReport {
            left_grad_mag,
            right_grad_mag,
            gradient_asymmetry,
            sign_flip_fraction,
        })
    }
}

// ─── StitchReport ─────────────────────────────────────────────────────────────

/// Comprehensive stitch quality report for a single seam position.
#[derive(Debug, Clone)]
pub struct StitchReport {
    /// Seam column index in the equirectangular image.
    pub seam_col: u32,
    /// Overall seam visibility score in `[0, 1]`.
    pub visibility_score: f32,
    /// Per-channel and aggregate colour mismatch.
    pub colour_mismatch: ColourMismatch,
    /// Parallax artifact metrics.
    pub parallax: ParallaxReport,
    /// Composite quality score: lower = better stitch.
    ///
    /// Computed as:
    /// `0.4 * visibility_score + 0.35 * colour_mismatch.aggregate_mad
    ///  + 0.25 * parallax.gradient_asymmetry`
    pub composite_score: f32,
}

impl StitchReport {
    /// Run a full quality assessment on the given equirectangular image at the
    /// specified seam column.
    ///
    /// # Parameters
    ///
    /// * `pixels`       — RGB row-major pixel buffer.
    /// * `width`        — image width.
    /// * `height`       — image height.
    /// * `seam_col`     — column index of the stitching seam.
    /// * `strip_width`  — number of columns on each side to analyse.
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying analysers.
    pub fn assess(
        pixels: &[u8],
        width: u32,
        height: u32,
        seam_col: u32,
        strip_width: u32,
    ) -> Result<Self, VrError> {
        let strip_w = strip_width.max(1);

        let analyser =
            SeamVisibilityAnalyser::new(width, height, strip_w.min(width / 2 - 1).max(1))?;
        let visibility_score = analyser.seam_score(pixels, seam_col)?;

        let colour_mismatch = ColourMismatch::compute(pixels, width, height, seam_col, strip_w)?;

        // Parallax detector uses a potentially wider radius
        let par_radius = strip_w.min(width / 2 - 1).max(1);
        let parallax = if seam_col >= par_radius && seam_col + par_radius < width {
            let det = ParallaxDetector::new(width, height, par_radius)?;
            det.analyse(pixels, seam_col)?
        } else {
            ParallaxReport {
                left_grad_mag: 0.0,
                right_grad_mag: 0.0,
                gradient_asymmetry: 0.0,
                sign_flip_fraction: 0.0,
            }
        };

        let composite_score = 0.4 * visibility_score
            + 0.35 * colour_mismatch.aggregate_mad
            + 0.25 * parallax.gradient_asymmetry;

        Ok(Self {
            seam_col,
            visibility_score,
            colour_mismatch,
            parallax,
            composite_score,
        })
    }

    /// Return a human-readable quality label based on the composite score.
    #[must_use]
    pub fn quality_label(&self) -> &'static str {
        match self.composite_score {
            s if s < 0.05 => "Excellent",
            s if s < 0.15 => "Good",
            s if s < 0.30 => "Fair",
            s if s < 0.50 => "Poor",
            _ => "Unacceptable",
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Compute the luminance (BT.601) of a pixel at `(row, col)` in an RGB buffer.
#[inline]
fn luma(pixels: &[u8], row: usize, col: usize, width: usize) -> f32 {
    let base = (row * width + col) * 3;
    if base + 2 >= pixels.len() {
        return 0.0;
    }
    let r = pixels[base] as f32;
    let g = pixels[base + 1] as f32;
    let b = pixels[base + 2] as f32;
    0.299 * r + 0.587 * g + 0.114 * b
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a W×H solid-colour RGB image.
    fn solid(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = w as usize * h as usize;
        let mut v = Vec::with_capacity(n * 3);
        for _ in 0..n {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    /// Create an image with a vertical colour boundary at `seam_col`:
    /// left side filled with `(lr, lg, lb)` and right side with `(rr, rg, rb)`.
    fn split_image(
        w: u32,
        h: u32,
        seam_col: u32,
        lr: u8,
        lg: u8,
        lb: u8,
        rr: u8,
        rg: u8,
        rb: u8,
    ) -> Vec<u8> {
        let mut v = vec![0u8; w as usize * h as usize * 3];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let base = (row * w as usize + col) * 3;
                let (r, g, b) = if col < seam_col as usize {
                    (lr, lg, lb)
                } else {
                    (rr, rg, rb)
                };
                v[base] = r;
                v[base + 1] = g;
                v[base + 2] = b;
            }
        }
        v
    }

    // ── SeamVisibilityAnalyser ────────────────────────────────────────────────

    #[test]
    fn seam_analyser_rejects_zero_dimensions() {
        assert!(SeamVisibilityAnalyser::new(0, 32, 4).is_err());
        assert!(SeamVisibilityAnalyser::new(64, 0, 4).is_err());
        assert!(SeamVisibilityAnalyser::new(64, 32, 0).is_err());
    }

    #[test]
    fn seam_score_solid_image_is_zero() {
        let img = solid(128, 64, 100, 150, 200);
        let a = SeamVisibilityAnalyser::new(128, 64, 4).expect("valid");
        let score = a.seam_score(&img, 64).expect("ok");
        assert!(score < 1e-4, "solid image seam should be 0, got {score}");
    }

    #[test]
    fn seam_score_high_for_max_contrast_seam() {
        // Left half white, right half black
        let img = split_image(128, 64, 64, 255, 255, 255, 0, 0, 0);
        let a = SeamVisibilityAnalyser::new(128, 64, 4).expect("valid");
        let score = a.seam_score(&img, 64).expect("ok");
        assert!(
            score > 0.9,
            "high contrast seam should score near 1, got {score}"
        );
    }

    #[test]
    fn seam_score_rejects_buffer_too_small() {
        let a = SeamVisibilityAnalyser::new(128, 64, 4).expect("valid");
        assert!(a.seam_score(&[0u8; 10], 64).is_err());
    }

    // ── ColourMismatch ────────────────────────────────────────────────────────

    #[test]
    fn colour_mismatch_solid_is_zero() {
        let img = solid(128, 64, 200, 100, 50);
        let m = ColourMismatch::compute(&img, 128, 64, 64, 8).expect("ok");
        assert!(
            m.aggregate_mad < 1e-5,
            "solid image mismatch={}",
            m.aggregate_mad
        );
    }

    #[test]
    fn colour_mismatch_full_contrast_is_near_one() {
        let img = split_image(128, 64, 64, 255, 255, 255, 0, 0, 0);
        let m = ColourMismatch::compute(&img, 128, 64, 64, 8).expect("ok");
        assert!(
            m.aggregate_mad > 0.9,
            "full contrast mismatch={}",
            m.aggregate_mad
        );
    }

    #[test]
    fn colour_mismatch_rejects_invalid_seam_col() {
        let img = solid(128, 64, 100, 100, 100);
        assert!(ColourMismatch::compute(&img, 128, 64, 0, 8).is_err());
        assert!(ColourMismatch::compute(&img, 128, 64, 128, 8).is_err());
    }

    // ── ParallaxDetector ──────────────────────────────────────────────────────

    #[test]
    fn parallax_solid_image_low_gradient() {
        let img = solid(64, 32, 128, 128, 128);
        let det = ParallaxDetector::new(64, 32, 4).expect("valid");
        let report = det.analyse(&img, 32).expect("ok");
        assert!(
            report.left_grad_mag < 1e-4,
            "solid left_grad={}",
            report.left_grad_mag
        );
        assert!(
            report.right_grad_mag < 1e-4,
            "solid right_grad={}",
            report.right_grad_mag
        );
    }

    #[test]
    fn parallax_rejects_seam_too_close_to_edge() {
        let img = solid(64, 32, 100, 100, 100);
        let det = ParallaxDetector::new(64, 32, 10).expect("valid");
        // seam_col=5 is too close to the left edge for radius=10
        assert!(det.analyse(&img, 5).is_err());
    }

    #[test]
    fn parallax_rejects_zero_dimensions() {
        assert!(ParallaxDetector::new(0, 32, 4).is_err());
        assert!(ParallaxDetector::new(64, 0, 4).is_err());
        assert!(ParallaxDetector::new(64, 32, 0).is_err());
    }

    // ── StitchReport ─────────────────────────────────────────────────────────

    #[test]
    fn stitch_report_solid_image_excellent_quality() {
        let img = solid(128, 64, 180, 120, 60);
        let report = StitchReport::assess(&img, 128, 64, 64, 6).expect("ok");
        assert!(
            report.composite_score < 0.05,
            "score={}",
            report.composite_score
        );
        assert_eq!(report.quality_label(), "Excellent");
    }

    #[test]
    fn stitch_report_high_contrast_poor_quality() {
        let img = split_image(128, 64, 64, 255, 255, 255, 0, 0, 0);
        let report = StitchReport::assess(&img, 128, 64, 64, 6).expect("ok");
        assert!(
            report.composite_score > 0.3,
            "score={}",
            report.composite_score
        );
    }

    #[test]
    fn stitch_report_quality_labels_cover_range() {
        // Make a report with a known score by constructing one directly
        let dummy_colour = ColourMismatch {
            red_mad: 0.0,
            green_mad: 0.0,
            blue_mad: 0.0,
            aggregate_mad: 0.0,
        };
        let dummy_parallax = ParallaxReport {
            left_grad_mag: 0.0,
            right_grad_mag: 0.0,
            gradient_asymmetry: 0.0,
            sign_flip_fraction: 0.0,
        };

        let labels = ["Excellent", "Good", "Fair", "Poor", "Unacceptable"];
        let scores = [0.01f32, 0.10, 0.22, 0.40, 0.60];
        for (label, &score) in labels.iter().zip(scores.iter()) {
            let report = StitchReport {
                seam_col: 64,
                visibility_score: score,
                colour_mismatch: dummy_colour.clone(),
                parallax: dummy_parallax.clone(),
                composite_score: score,
            };
            assert_eq!(report.quality_label(), *label, "score={score}");
        }
    }
}
