//! Spatial quality heatmap — per-region quality assessment.
//!
//! Divides an image into a grid of equally-sized blocks and computes block-level
//! SSIM and PSNR scores.  The result is a [`QualityHeatmap`] that can be queried
//! for best/worst regions or regions below a quality threshold.

use serde::{Deserialize, Serialize};
use std::fmt;

// ─── error type ──────────────────────────────────────────────────────────────

/// Errors produced by heatmap generation.
#[derive(Debug, Clone, PartialEq)]
pub enum HeatmapError {
    /// Input pixel buffer is empty or has zero-area dimensions.
    EmptyImage,
    /// Reference and distorted buffers have different lengths.
    DimensionMismatch {
        /// Expected buffer length (reference length).
        expected: usize,
        /// Actual buffer length (distorted length).
        actual: usize,
    },
    /// `grid_size` is zero.
    InvalidGridSize,
    /// A grid block contains no pixels (image smaller than grid_size).
    BlockTooSmall,
}

impl fmt::Display for HeatmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => write!(f, "image buffer is empty"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "buffer length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidGridSize => write!(f, "grid_size must be greater than zero"),
            Self::BlockTooSmall => {
                write!(f, "image is smaller than the requested grid size")
            }
        }
    }
}

impl std::error::Error for HeatmapError {}

// ─── quality classification ───────────────────────────────────────────────────

/// Perceptual quality class derived from block SSIM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityClass {
    /// SSIM ≥ 0.95 — virtually indistinguishable from the reference.
    Excellent,
    /// 0.85 ≤ SSIM < 0.95 — good quality with minor artefacts.
    Good,
    /// 0.70 ≤ SSIM < 0.85 — noticeable but tolerable degradation.
    Fair,
    /// SSIM < 0.70 — significant degradation.
    Poor,
}

impl QualityClass {
    /// Derives the quality class from an SSIM value.
    #[must_use]
    pub fn from_ssim(ssim: f32) -> Self {
        if ssim >= 0.95 {
            Self::Excellent
        } else if ssim >= 0.85 {
            Self::Good
        } else if ssim >= 0.70 {
            Self::Fair
        } else {
            Self::Poor
        }
    }
}

// ─── region / heatmap ────────────────────────────────────────────────────────

/// Quality metrics for a single grid block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRegion {
    /// Left edge of the block (pixels from left).
    pub x: u32,
    /// Top edge of the block (pixels from top).
    pub y: u32,
    /// Width of the block in pixels.
    pub width: u32,
    /// Height of the block in pixels.
    pub height: u32,
    /// Block-level SSIM score [−1, 1] (1.0 = identical).
    pub ssim: f32,
    /// Block-level PSNR in dB (inf for identical blocks).
    pub psnr_db: f32,
    /// Perceptual quality class derived from `ssim`.
    pub quality_class: QualityClass,
}

/// Spatial quality heatmap: a grid of per-block quality scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityHeatmap {
    /// Individual block quality scores in row-major order.
    pub regions: Vec<QualityRegion>,
    /// Number of grid columns.
    pub grid_cols: u32,
    /// Number of grid rows.
    pub grid_rows: u32,
}

impl QualityHeatmap {
    /// Returns a reference to the region with the lowest SSIM score.
    ///
    /// Returns `None` if there are no regions.
    #[must_use]
    pub fn worst_region(&self) -> Option<&QualityRegion> {
        self.regions.iter().min_by(|a, b| {
            a.ssim
                .partial_cmp(&b.ssim)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns a reference to the region with the highest SSIM score.
    ///
    /// Returns `None` if there are no regions.
    #[must_use]
    pub fn best_region(&self) -> Option<&QualityRegion> {
        self.regions.iter().max_by(|a, b| {
            a.ssim
                .partial_cmp(&b.ssim)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Computes the arithmetic mean SSIM across all regions.
    ///
    /// Returns `0.0` if there are no regions.
    #[must_use]
    pub fn average_ssim(&self) -> f32 {
        if self.regions.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.regions.iter().map(|r| r.ssim).sum();
        sum / self.regions.len() as f32
    }

    /// Returns all regions whose SSIM is strictly below `ssim_threshold`.
    #[must_use]
    pub fn regions_below_threshold(&self, ssim_threshold: f32) -> Vec<&QualityRegion> {
        self.regions
            .iter()
            .filter(|r| r.ssim < ssim_threshold)
            .collect()
    }
}

// ─── generator ───────────────────────────────────────────────────────────────

/// Generates spatial quality heatmaps by dividing an image into a uniform grid
/// and computing per-block SSIM and PSNR.
pub struct HeatmapGenerator;

impl HeatmapGenerator {
    /// Creates a new `HeatmapGenerator`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Generates a quality heatmap by comparing reference and distorted images.
    ///
    /// # Parameters
    /// * `reference` — 8-bit grayscale (or luma-plane) pixel data for the reference.
    /// * `distorted` — matching pixel data for the distorted image.
    /// * `width` / `height` — image dimensions in pixels.
    /// * `grid_size` — the number of blocks along each axis (both horizontal and
    ///   vertical), resulting in `grid_size × grid_size` blocks in total.
    ///
    /// # Errors
    /// Returns a [`HeatmapError`] for empty images, mismatched buffers, zero
    /// grid sizes, or images too small to form even a single block.
    pub fn generate(
        &self,
        reference: &[u8],
        distorted: &[u8],
        width: u32,
        height: u32,
        grid_size: u32,
    ) -> Result<QualityHeatmap, HeatmapError> {
        generate_heatmap(reference, distorted, width, height, grid_size)
    }
}

impl Default for HeatmapGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── core computation ─────────────────────────────────────────────────────────

fn generate_heatmap(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    grid_size: u32,
) -> Result<QualityHeatmap, HeatmapError> {
    let expected = (width as usize) * (height as usize);
    if reference.is_empty() || expected == 0 {
        return Err(HeatmapError::EmptyImage);
    }
    if distorted.len() != expected {
        return Err(HeatmapError::DimensionMismatch {
            expected,
            actual: distorted.len(),
        });
    }
    if grid_size == 0 {
        return Err(HeatmapError::InvalidGridSize);
    }
    if width < grid_size || height < grid_size {
        return Err(HeatmapError::BlockTooSmall);
    }

    let block_w = width / grid_size;
    let block_h = height / grid_size;
    // Remainder pixels are absorbed into the last column / row.
    let mut regions = Vec::with_capacity((grid_size * grid_size) as usize);

    for row in 0..grid_size {
        let y_start = row * block_h;
        let y_end = if row == grid_size - 1 {
            height
        } else {
            y_start + block_h
        };

        for col in 0..grid_size {
            let x_start = col * block_w;
            let x_end = if col == grid_size - 1 {
                width
            } else {
                x_start + block_w
            };

            let bw = x_end - x_start;
            let bh = y_end - y_start;

            // Extract block pixels.
            let ref_block = extract_block(reference, width, x_start, y_start, bw, bh);
            let dist_block = extract_block(distorted, width, x_start, y_start, bw, bh);

            let ssim = block_ssim(&ref_block, &dist_block);
            let psnr_db = block_psnr(&ref_block, &dist_block);
            let quality_class = QualityClass::from_ssim(ssim);

            regions.push(QualityRegion {
                x: x_start,
                y: y_start,
                width: bw,
                height: bh,
                ssim,
                psnr_db,
                quality_class,
            });
        }
    }

    Ok(QualityHeatmap {
        regions,
        grid_cols: grid_size,
        grid_rows: grid_size,
    })
}

/// Extracts a rectangular block from a packed image buffer.
fn extract_block(pixels: &[u8], img_width: u32, x: u32, y: u32, bw: u32, bh: u32) -> Vec<f64> {
    let mut block = Vec::with_capacity((bw * bh) as usize);
    for row in 0..bh {
        let base = ((y + row) * img_width + x) as usize;
        for col in 0..bw as usize {
            block.push(pixels[base + col] as f64);
        }
    }
    block
}

/// Computes a block-level SSIM value between two flat f64 arrays.
///
/// Uses the simplified (global statistics) SSIM formula — appropriate for
/// moderate-sized blocks where a full sliding-window approach would be
/// computationally expensive and unnecessary.
fn block_ssim(x: &[f64], y: &[f64]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let n = x.len() as f64;
    if n == 0.0 {
        return 1.0;
    }

    let mu_x = x.iter().sum::<f64>() / n;
    let mu_y = y.iter().sum::<f64>() / n;

    let (var_x, var_y, cov_xy) =
        x.iter()
            .zip(y.iter())
            .fold((0.0_f64, 0.0_f64, 0.0_f64), |(vx, vy, cov), (&xi, &yi)| {
                let dx = xi - mu_x;
                let dy = yi - mu_y;
                (vx + dx * dx, vy + dy * dy, cov + dx * dy)
            });

    // Biased variance (same as reference SSIM implementations).
    let var_x = var_x / n;
    let var_y = var_y / n;
    let cov_xy = cov_xy / n;

    // SSIM constants for 8-bit images (L = 255).
    let c1 = (0.01 * 255.0_f64).powi(2);
    let c2 = (0.03 * 255.0_f64).powi(2);

    let numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2);
    let denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2);

    if denominator.abs() < 1e-12 {
        1.0_f32
    } else {
        (numerator / denominator).clamp(-1.0, 1.0) as f32
    }
}

/// Computes block-level PSNR in dB between two flat f64 arrays (8-bit images).
fn block_psnr(x: &[f64], y: &[f64]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let n = x.len() as f64;
    if n == 0.0 {
        return f32::INFINITY;
    }

    let mse = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi) * (xi - yi))
        .sum::<f64>()
        / n;

    if mse < 1e-12 {
        return f32::INFINITY;
    }

    (10.0 * (255.0_f64 * 255.0_f64 / mse).log10()) as f32
}

// ─── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(value: u8, w: u32, h: u32) -> Vec<u8> {
        vec![value; (w * h) as usize]
    }

    fn ramp_image(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h)).map(|i| (i % 256) as u8).collect()
    }

    // ── basic errors ──────────────────────────────────────────────────────────

    #[test]
    fn empty_image_returns_error() {
        let gen = HeatmapGenerator::new();
        let result = gen.generate(&[], &[], 0, 0, 4);
        assert!(matches!(result, Err(HeatmapError::EmptyImage)));
    }

    #[test]
    fn dimension_mismatch_error() {
        let gen = HeatmapGenerator::new();
        let r = solid_image(128, 4, 4);
        let d = solid_image(128, 3, 4); // wrong length
        let result = gen.generate(&r, &d, 4, 4, 2);
        assert!(matches!(
            result,
            Err(HeatmapError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn zero_grid_size_error() {
        let gen = HeatmapGenerator::new();
        let r = solid_image(128, 8, 8);
        let result = gen.generate(&r, &r, 8, 8, 0);
        assert!(matches!(result, Err(HeatmapError::InvalidGridSize)));
    }

    #[test]
    fn image_smaller_than_grid_error() {
        let gen = HeatmapGenerator::new();
        let r = solid_image(128, 2, 2);
        let result = gen.generate(&r, &r, 2, 2, 4);
        assert!(matches!(result, Err(HeatmapError::BlockTooSmall)));
    }

    // ── correctness ───────────────────────────────────────────────────────────

    #[test]
    fn identical_images_all_excellent() {
        let gen = HeatmapGenerator::new();
        let img = ramp_image(16, 16);
        let heatmap = gen.generate(&img, &img, 16, 16, 4).expect("should succeed");

        assert_eq!(heatmap.regions.len(), 16);
        for region in &heatmap.regions {
            assert_eq!(
                region.quality_class,
                QualityClass::Excellent,
                "expected Excellent, ssim={}",
                region.ssim
            );
            assert!(
                region.ssim > 0.999,
                "ssim should be ~1.0, got {}",
                region.ssim
            );
        }
        assert!(heatmap.average_ssim() > 0.999);
    }

    #[test]
    fn fully_distorted_image_all_poor() {
        let gen = HeatmapGenerator::new();
        let reference = solid_image(200, 16, 16);
        let distorted = solid_image(0, 16, 16);
        let heatmap = gen
            .generate(&reference, &distorted, 16, 16, 4)
            .expect("should succeed");

        for region in &heatmap.regions {
            assert_eq!(
                region.quality_class,
                QualityClass::Poor,
                "expected Poor, ssim={}",
                region.ssim
            );
        }
    }

    #[test]
    fn best_and_worst_region_detection() {
        let gen = HeatmapGenerator::new();
        // 8×8 image, 2×2 grid (4 blocks of 4×4 pixels each).
        let mut reference = solid_image(128, 8, 8);
        let mut distorted = solid_image(128, 8, 8);

        // Corrupt top-left block (rows 0-3, cols 0-3) heavily.
        for row in 0..4_usize {
            for col in 0..4_usize {
                distorted[row * 8 + col] = 0;
            }
        }
        // Make the bottom-right block perfect (already identical — reference 128).
        // Ensure reference top-left is very different from distorted.
        for row in 0..4_usize {
            for col in 0..4_usize {
                reference[row * 8 + col] = 255;
            }
        }

        let heatmap = gen
            .generate(&reference, &distorted, 8, 8, 2)
            .expect("should succeed");

        let worst = heatmap.worst_region().expect("should have a worst region");
        let best = heatmap.best_region().expect("should have a best region");

        assert!(worst.ssim < best.ssim, "worst must be worse than best");
    }

    #[test]
    fn regions_below_threshold_filters_correctly() {
        let gen = HeatmapGenerator::new();
        let reference = solid_image(200, 8, 8);
        let distorted = solid_image(0, 8, 8);
        let heatmap = gen
            .generate(&reference, &distorted, 8, 8, 2)
            .expect("should succeed");

        let poor_regions = heatmap.regions_below_threshold(0.70);
        assert!(!poor_regions.is_empty());
        for r in &poor_regions {
            assert!(r.ssim < 0.70);
        }
    }

    #[test]
    fn single_block_grid() {
        let gen = HeatmapGenerator::new();
        let img = ramp_image(8, 8);
        let heatmap = gen.generate(&img, &img, 8, 8, 1).expect("should succeed");

        assert_eq!(heatmap.regions.len(), 1);
        assert_eq!(heatmap.grid_cols, 1);
        assert_eq!(heatmap.grid_rows, 1);
        let region = &heatmap.regions[0];
        assert!(region.ssim > 0.999);
        assert!(region.psnr_db.is_infinite());
    }

    #[test]
    fn grid_has_correct_region_count() {
        let gen = HeatmapGenerator::new();
        let img = ramp_image(12, 12);
        let heatmap = gen.generate(&img, &img, 12, 12, 3).expect("should succeed");

        assert_eq!(heatmap.regions.len(), 9);
        assert_eq!(heatmap.grid_cols, 3);
        assert_eq!(heatmap.grid_rows, 3);
    }

    #[test]
    fn psnr_infinite_for_identical_blocks() {
        let gen = HeatmapGenerator::new();
        let img = ramp_image(8, 8);
        let heatmap = gen.generate(&img, &img, 8, 8, 2).expect("should succeed");

        for region in &heatmap.regions {
            assert!(
                region.psnr_db.is_infinite(),
                "expected infinite PSNR for identical blocks, got {}",
                region.psnr_db
            );
        }
    }
}
