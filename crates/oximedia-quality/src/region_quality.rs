//! Per-region quality assessment: compute SSIM and PSNR for specific frame
//! sub-regions or over a uniform grid.
//!
//! All quality values are computed in the spatial domain on the raw 8-bit
//! luma plane.  For SSIM the simplified single-scale formulation is used
//! (global mean/variance/covariance over the region rather than the windowed
//! approach) so that small regions remain well-defined even when they contain
//! fewer pixels than a typical 11×11 Gaussian window.
//!
//! Constants are chosen for 8-bit data:
//! * C1 = (K1 · L)² where K1 = 0.01, L = 255 → C1 ≈ 6.5025
//! * C2 = (K2 · L)² where K2 = 0.03, L = 255 → C2 ≈ 58.5225

#![allow(dead_code)]

use crate::psnr::extract_plane_roi;

// ── SSIM stabilisation constants for 8-bit data ────────────────────────────
const C1: f64 = 6.502_5; // (0.01 * 255)^2
const C2: f64 = 58.522_5; // (0.03 * 255)^2

// ── Public types ────────────────────────────────────────────────────────────

/// A rectangular region within a video frame (pixel coordinates, luma plane).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRegion {
    /// Left edge in pixels (inclusive).
    pub x: u32,
    /// Top edge in pixels (inclusive).
    pub y: u32,
    /// Width of the region in pixels.
    pub width: u32,
    /// Height of the region in pixels.
    pub height: u32,
}

impl FrameRegion {
    /// Creates a new `FrameRegion`.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Returns `true` when the region fits entirely within a frame of the given
    /// dimensions.
    #[must_use]
    pub fn is_within(&self, frame_width: u32, frame_height: u32) -> bool {
        self.width > 0
            && self.height > 0
            && self.x.saturating_add(self.width) <= frame_width
            && self.y.saturating_add(self.height) <= frame_height
    }
}

/// SSIM and PSNR scores for a specific [`FrameRegion`].
///
/// Either metric may be `None` when the region is out-of-bounds, degenerate,
/// or the pixel buffer is too small.
#[derive(Debug, Clone)]
pub struct RegionQuality {
    /// The spatial region these measurements correspond to.
    pub region: FrameRegion,
    /// PSNR in dB.  `None` if the region is invalid or the slice is too small.
    pub psnr: Option<f64>,
    /// SSIM in \[−1, 1\].  `None` if the region is invalid or degenerate.
    pub ssim: Option<f64>,
}

// ── Core helpers ─────────────────────────────────────────────────────────────

/// Compute PSNR over a packed 8-bit luma region.
///
/// Returns `None` when the pixel slice is empty.
fn psnr_from_pixels(reference: &[u8], distorted: &[u8]) -> Option<f64> {
    if reference.is_empty() || reference.len() != distorted.len() {
        return None;
    }

    let mse: f64 = reference
        .iter()
        .zip(distorted.iter())
        .map(|(&r, &d)| {
            let diff = i32::from(r) - i32::from(d);
            (diff * diff) as f64
        })
        .sum::<f64>()
        / reference.len() as f64;

    if mse < 1e-10 {
        // Identical pixels → infinite PSNR; cap at a finite sentinel so that
        // callers can perform numeric comparisons without special-casing.
        Some(100.0)
    } else {
        let max_val = 255.0_f64;
        Some(10.0 * (max_val * max_val / mse).log10())
    }
}

/// Compute simplified single-scale SSIM over a packed 8-bit luma region.
///
/// Returns `None` when the pixel slice is empty or has only one pixel
/// (variance is undefined for n < 2).
fn ssim_from_pixels(reference: &[u8], distorted: &[u8]) -> Option<f64> {
    let n = reference.len();
    if n < 2 || n != distorted.len() {
        return None;
    }

    let n_f = n as f64;

    // Means
    let mu_x: f64 = reference.iter().map(|&v| v as f64).sum::<f64>() / n_f;
    let mu_y: f64 = distorted.iter().map(|&v| v as f64).sum::<f64>() / n_f;

    // Variances and covariance (biased, consistent with ITU-R definition)
    let mut sigma_x_sq = 0.0_f64;
    let mut sigma_y_sq = 0.0_f64;
    let mut sigma_xy = 0.0_f64;

    for (&rx, &dx) in reference.iter().zip(distorted.iter()) {
        let dx_x = rx as f64 - mu_x;
        let dx_y = dx as f64 - mu_y;
        sigma_x_sq += dx_x * dx_x;
        sigma_y_sq += dx_y * dx_y;
        sigma_xy += dx_x * dx_y;
    }

    // Normalise by N (biased estimator matches the standard SSIM paper)
    sigma_x_sq /= n_f;
    sigma_y_sq /= n_f;
    sigma_xy /= n_f;

    let numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2);
    let denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x_sq + sigma_y_sq + C2);

    if denominator.abs() < 1e-12 {
        // Both regions are uniform and identical → SSIM = 1
        Some(1.0)
    } else {
        Some(numerator / denominator)
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Compute SSIM and PSNR for a specific region within a pair of frames.
///
/// `reference` and `distorted` are packed, row-major luma planes of size
/// `frame_width × frame_height` bytes.  Both slices must have at least
/// `frame_width * frame_height` bytes; if they are smaller, both metrics will
/// be `None`.
///
/// Either metric is `None` when the region extends beyond the frame boundaries
/// or is degenerate (zero area).
///
/// # Arguments
///
/// * `reference` – raw 8-bit luma pixels of the reference frame (row-major, no padding)
/// * `distorted` – raw 8-bit luma pixels of the distorted frame
/// * `frame_width` – frame width in pixels
/// * `frame_height` – frame height in pixels
/// * `region` – sub-region to evaluate
#[must_use]
pub fn assess_region(
    reference: &[u8],
    distorted: &[u8],
    frame_width: u32,
    frame_height: u32,
    region: FrameRegion,
) -> RegionQuality {
    // Validate that the slices are large enough for the declared frame size.
    let expected_len = (frame_width as usize).saturating_mul(frame_height as usize);
    if reference.len() < expected_len
        || distorted.len() < expected_len
        || !region.is_within(frame_width, frame_height)
    {
        return RegionQuality {
            region,
            psnr: None,
            ssim: None,
        };
    }

    // Extract the region from both planes using the existing ROI helper.
    let stride = frame_width as usize;
    let ref_roi = extract_plane_roi(
        reference,
        stride,
        region.x as usize,
        region.y as usize,
        region.width as usize,
        region.height as usize,
    );
    let dist_roi = extract_plane_roi(
        distorted,
        stride,
        region.x as usize,
        region.y as usize,
        region.width as usize,
        region.height as usize,
    );

    let psnr = psnr_from_pixels(&ref_roi, &dist_roi);
    let ssim = ssim_from_pixels(&ref_roi, &dist_roi);

    RegionQuality { region, psnr, ssim }
}

/// Divide the frame into a `grid_cols × grid_rows` uniform grid and compute
/// [`RegionQuality`] for every cell.
///
/// The grid is aligned to the frame's top-left corner.  Any pixels that fall
/// beyond the last cell boundary are included in the last cell so that the
/// entire frame is always covered.
///
/// Returns `grid_cols * grid_rows` items in row-major order (left-to-right,
/// top-to-bottom).  Cells that are degenerate (zero area) will have both
/// metrics set to `None`.
///
/// # Arguments
///
/// * `reference` / `distorted` – packed luma planes (see [`assess_region`])
/// * `frame_width` / `frame_height` – frame dimensions in pixels
/// * `grid_cols` – number of columns (must be ≥ 1; clamped otherwise)
/// * `grid_rows` – number of rows (must be ≥ 1; clamped otherwise)
#[must_use]
pub fn assess_grid(
    reference: &[u8],
    distorted: &[u8],
    frame_width: u32,
    frame_height: u32,
    grid_cols: u32,
    grid_rows: u32,
) -> Vec<RegionQuality> {
    let cols = grid_cols.max(1);
    let rows = grid_rows.max(1);

    let cell_w = frame_width / cols;
    let cell_h = frame_height / rows;

    let mut results = Vec::with_capacity((cols * rows) as usize);

    for row in 0..rows {
        for col in 0..cols {
            let x = col * cell_w;
            let y = row * cell_h;

            // The last column/row absorbs any remainder pixels.
            let w = if col == cols - 1 {
                frame_width - x
            } else {
                cell_w
            };
            let h = if row == rows - 1 {
                frame_height - y
            } else {
                cell_h
            };

            let region = FrameRegion::new(x, y, w, h);
            results.push(assess_region(
                reference,
                distorted,
                frame_width,
                frame_height,
                region,
            ));
        }
    }

    results
}

// ── RoiQualityScorer ─────────────────────────────────────────────────────────

/// Stateful scorer that computes a combined quality metric for a configured
/// Region of Interest (ROI) within a single distorted frame.
///
/// Unlike the functional [`assess_region`] API, `RoiQualityScorer` remembers
/// the frame dimensions and the current ROI so callers can call
/// [`score`](Self::score) repeatedly without passing those arguments each time.
///
/// The returned score is the arithmetic mean of the SSIM and PSNR values
/// normalised to `[0, 1]`:
///
/// ```text
/// score = 0.5 * ssim_clamp01 + 0.5 * (psnr / 100.0).min(1.0)
/// ```
///
/// When either metric is unavailable (e.g. the ROI is out of bounds) its
/// contribution is omitted and the available metric alone determines the score.
pub struct RoiQualityScorer {
    frame_width: u32,
    frame_height: u32,
    roi: Option<FrameRegion>,
}

impl RoiQualityScorer {
    /// Creates a scorer for frames of `w × h` pixels.
    ///
    /// No ROI is active until [`set_roi`](Self::set_roi) is called; before
    /// that, [`score`](Self::score) evaluates the full frame.
    #[must_use]
    pub fn new(w: u32, h: u32) -> Self {
        Self {
            frame_width: w,
            frame_height: h,
            roi: None,
        }
    }

    /// Sets (or replaces) the active ROI.
    ///
    /// The ROI is described by its top-left corner `(x, y)` and its
    /// dimensions `(rw, rh)` in pixels.  Out-of-bounds ROIs are stored as-is;
    /// [`score`](Self::score) will return `0.0` for them.
    pub fn set_roi(&mut self, x: u32, y: u32, rw: u32, rh: u32) {
        self.roi = Some(FrameRegion::new(x, y, rw, rh));
    }

    /// Clears the active ROI, reverting to full-frame scoring.
    pub fn clear_roi(&mut self) {
        self.roi = None;
    }

    /// Scores the distorted `frame` (packed 8-bit luma, row-major) against a
    /// uniform grey reference.
    ///
    /// This variant is useful for no-reference (single-frame) quality
    /// monitoring: it compares the frame to a mid-grey (128) reference to
    /// measure how far the content deviates from a flat field, which acts as a
    /// proxy for signal energy.
    ///
    /// For a reference-aware comparison see [`score_with_reference`](Self::score_with_reference).
    ///
    /// Returns a value in `[0, 1]` where `1.0` indicates the content
    /// perfectly matches the mid-grey reference (no distortion relative to
    /// that baseline).
    #[must_use]
    pub fn score(&self, frame: &[u8]) -> f32 {
        let reference: Vec<u8> = vec![128u8; frame.len()];
        self.score_with_reference(&reference, frame)
    }

    /// Scores the distorted `frame` against the provided `reference` luma
    /// plane.  Both slices must contain at least
    /// `frame_width * frame_height` bytes.
    ///
    /// Returns a value in `[0, 1]`.
    #[must_use]
    pub fn score_with_reference(&self, reference: &[u8], distorted: &[u8]) -> f32 {
        let region = self
            .roi
            .unwrap_or_else(|| FrameRegion::new(0, 0, self.frame_width, self.frame_height));

        let rq = assess_region(
            reference,
            distorted,
            self.frame_width,
            self.frame_height,
            region,
        );

        let ssim_contribution = rq.ssim.map(|s| ((s as f32 + 1.0) / 2.0).clamp(0.0, 1.0));
        let psnr_contribution = rq.psnr.map(|p| (p as f32 / 100.0).min(1.0).max(0.0));

        match (ssim_contribution, psnr_contribution) {
            (Some(s), Some(p)) => 0.5 * s + 0.5 * p,
            (Some(s), None) => s,
            (None, Some(p)) => p,
            (None, None) => 0.0,
        }
    }

    /// Returns the active ROI, or `None` when no ROI is set.
    #[must_use]
    pub fn roi(&self) -> Option<FrameRegion> {
        self.roi
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a packed luma plane filled with a single value.
    fn uniform_plane(w: u32, h: u32, value: u8) -> Vec<u8> {
        vec![value; (w * h) as usize]
    }

    // ── assess_region tests ──────────────────────────────────────────────────

    #[test]
    fn test_region_identical_full_frame_high_psnr() {
        let w = 32u32;
        let h = 32u32;
        let plane = uniform_plane(w, h, 128);
        let region = FrameRegion::new(0, 0, w, h);
        let rq = assess_region(&plane, &plane, w, h, region);
        let psnr = rq.psnr.expect("psnr should be Some for identical pixels");
        assert!(
            psnr > 99.0,
            "identical frame: expected psnr > 99, got {psnr:.2}"
        );
    }

    #[test]
    fn test_region_identical_full_frame_ssim_is_one() {
        let w = 64u32;
        let h = 64u32;
        let plane = uniform_plane(w, h, 200);
        let region = FrameRegion::new(0, 0, w, h);
        let rq = assess_region(&plane, &plane, w, h, region);
        let ssim = rq.ssim.expect("ssim should be Some for identical pixels");
        assert!(
            (ssim - 1.0).abs() < 1e-6,
            "identical frame: expected ssim ≈ 1, got {ssim}"
        );
    }

    #[test]
    fn test_region_out_of_bounds_returns_none() {
        let plane = uniform_plane(32, 32, 100);
        // Region extends beyond frame boundary
        let region = FrameRegion::new(24, 24, 16, 16); // (24+16=40) > 32
        let rq = assess_region(&plane, &plane, 32, 32, region);
        assert!(rq.psnr.is_none(), "out-of-bounds: psnr should be None");
        assert!(rq.ssim.is_none(), "out-of-bounds: ssim should be None");
    }

    #[test]
    fn test_region_zero_width_returns_none() {
        let plane = uniform_plane(32, 32, 100);
        let region = FrameRegion::new(0, 0, 0, 16);
        let rq = assess_region(&plane, &plane, 32, 32, region);
        assert!(rq.psnr.is_none(), "zero-width region: psnr should be None");
        assert!(rq.ssim.is_none(), "zero-width region: ssim should be None");
    }

    #[test]
    fn test_region_uniform_mse_zero_psnr_high() {
        let w = 16u32;
        let h = 16u32;
        let ref_plane = uniform_plane(w, h, 50);
        let dist_plane = uniform_plane(w, h, 50);
        let region = FrameRegion::new(0, 0, w, h);
        let rq = assess_region(&ref_plane, &dist_plane, w, h, region);
        let psnr = rq.psnr.expect("psnr must be Some");
        assert!(
            psnr > 99.0,
            "uniform identical: PSNR should cap at 100, got {psnr}"
        );
    }

    #[test]
    fn test_region_sub_region_at_origin() {
        let w = 64u32;
        let h = 64u32;
        // Left half identical, right half different
        let mut ref_plane = uniform_plane(w, h, 100);
        let mut dist_plane = uniform_plane(w, h, 100);
        for row in 0..h as usize {
            for col in (w / 2) as usize..w as usize {
                dist_plane[row * w as usize + col] = 200;
            }
        }
        let left_region = FrameRegion::new(0, 0, w / 2, h);
        let rq = assess_region(&ref_plane, &dist_plane, w, h, left_region);
        let psnr = rq.psnr.expect("left-half psnr must be Some");
        // Left half is identical → very high PSNR
        assert!(
            psnr > 99.0,
            "left identical half: expected high psnr, got {psnr}"
        );
        let _ = &mut ref_plane;
        let _ = &mut dist_plane;
    }

    #[test]
    fn test_region_different_pixels_lower_psnr() {
        let w = 16u32;
        let h = 16u32;
        let ref_plane = uniform_plane(w, h, 100);
        let dist_plane = uniform_plane(w, h, 150);
        let region = FrameRegion::new(0, 0, w, h);
        let rq = assess_region(&ref_plane, &dist_plane, w, h, region);
        let psnr = rq.psnr.expect("psnr must be Some");
        assert!(
            psnr < 99.0,
            "different pixels should yield psnr < 99, got {psnr}"
        );
        assert!(psnr > 0.0, "psnr must be positive, got {psnr}");
    }

    // ── assess_grid tests ────────────────────────────────────────────────────

    #[test]
    fn test_grid_2x2_returns_four_regions() {
        let w = 64u32;
        let h = 64u32;
        let plane = uniform_plane(w, h, 128);
        let results = assess_grid(&plane, &plane, w, h, 2, 2);
        assert_eq!(
            results.len(),
            4,
            "2×2 grid must return 4 RegionQuality entries"
        );
    }

    #[test]
    fn test_grid_cells_are_non_overlapping() {
        let w = 60u32;
        let h = 60u32;
        let plane = uniform_plane(w, h, 100);
        let results = assess_grid(&plane, &plane, w, h, 3, 3);
        assert_eq!(results.len(), 9);
        // Check that no two cells share the same (x, y) origin
        let origins: std::collections::HashSet<(u32, u32)> = results
            .iter()
            .map(|rq| (rq.region.x, rq.region.y))
            .collect();
        assert_eq!(origins.len(), 9, "all cells must have unique origins");
    }

    #[test]
    fn test_grid_1x1_equals_full_frame() {
        let w = 32u32;
        let h = 32u32;
        let ref_plane = uniform_plane(w, h, 80);
        let dist_plane = uniform_plane(w, h, 90);
        let grid_results = assess_grid(&ref_plane, &dist_plane, w, h, 1, 1);
        let full_region =
            assess_region(&ref_plane, &dist_plane, w, h, FrameRegion::new(0, 0, w, h));
        assert_eq!(grid_results.len(), 1);
        let grid_psnr = grid_results[0].psnr.expect("grid 1×1 psnr must be Some");
        let full_psnr = full_region.psnr.expect("full-frame psnr must be Some");
        assert!(
            (grid_psnr - full_psnr).abs() < 1e-6,
            "1×1 grid psnr {grid_psnr} must equal full-frame psnr {full_psnr}"
        );
    }
}
