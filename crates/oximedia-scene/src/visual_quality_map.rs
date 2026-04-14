//! Per-region visual quality mapping.
//!
//! This module divides a frame into a configurable grid of tiles and scores
//! each tile across four independent quality dimensions:
//!
//! | Dimension | Algorithm |
//! |-----------|-----------|
//! | **Sharpness** | Variance of Laplacian (Pertuz et al., 2013 — patent-free) |
//! | **Noise**     | Median-based noise estimation via MAD (Rousseeuw, 1987) |
//! | **Exposure**  | Mean luminance distance from mid-grey (0.5) |
//! | **Chroma uniformity** | Saturation standard deviation |
//!
//! # Output
//!
//! [`QualityMap`] contains one [`TileQuality`] per grid cell plus a
//! frame-level aggregate [`FrameQualitySummary`].  The summary includes a
//! broadcast-readiness flag based on configurable thresholds.
//!
//! # Example
//!
//! ```
//! use oximedia_scene::visual_quality_map::{VisualQualityAnalyzer, QualityMapConfig};
//!
//! let cfg = QualityMapConfig { grid_cols: 4, grid_rows: 4, ..Default::default() };
//! let analyzer = VisualQualityAnalyzer::new(cfg);
//! let width = 64usize;
//! let height = 64usize;
//! let rgb = vec![128u8; width * height * 3];
//! let map = analyzer.analyze(&rgb, width, height).unwrap();
//! assert_eq!(map.tiles.len(), 4 * 4);
//! ```

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the quality map analyzer.
#[derive(Debug, Clone)]
pub struct QualityMapConfig {
    /// Number of horizontal grid cells.
    pub grid_cols: usize,
    /// Number of vertical grid cells.
    pub grid_rows: usize,
    /// Minimum sharpness score considered acceptable (0.0–1.0).
    pub sharpness_threshold: f32,
    /// Maximum noise score considered acceptable (0.0–1.0).
    pub noise_threshold: f32,
    /// Maximum exposure deviation considered acceptable (0.0–1.0).
    pub exposure_threshold: f32,
    /// Whether to weight centre tiles more than corner tiles in the summary.
    pub centre_weighted: bool,
}

impl Default for QualityMapConfig {
    fn default() -> Self {
        Self {
            grid_cols: 6,
            grid_rows: 4,
            sharpness_threshold: 0.15,
            noise_threshold: 0.60,
            exposure_threshold: 0.40,
            centre_weighted: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Quality scores for a single grid tile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileQuality {
    /// Tile column index (0-based).
    pub col: usize,
    /// Tile row index (0-based).
    pub row: usize,
    /// Sharpness score (0.0 = very blurry, 1.0 = very sharp).
    pub sharpness: f32,
    /// Noise level (0.0 = clean, 1.0 = very noisy).
    pub noise: f32,
    /// Exposure score (0.0 = perfectly exposed, 1.0 = severely under/over-exposed).
    pub exposure_error: f32,
    /// Chroma uniformity (0.0 = highly uniform saturation, 1.0 = wildly varying).
    pub chroma_variance: f32,
    /// Composite quality score (0.0 = poor, 1.0 = excellent).
    pub composite: f32,
    /// Pixel-level defect flag.
    pub has_defect: bool,
}

impl TileQuality {
    /// Return the dominant quality issue as a string label.
    #[must_use]
    pub fn dominant_issue(&self) -> &'static str {
        // pick the worst metric
        let mut worst_val = self.composite;
        let mut issue = "none";

        // Low sharpness → issue
        if 1.0 - self.sharpness > 1.0 - worst_val {
            worst_val = self.sharpness;
            issue = "blur";
            let _ = worst_val; // suppress unused warning
        }
        if self.noise > 0.6 {
            issue = "noise";
        }
        if self.exposure_error > 0.4 {
            issue = "exposure";
        }
        issue
    }
}

/// Frame-level aggregate quality summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameQualitySummary {
    /// Mean sharpness across all tiles (optionally centre-weighted).
    pub mean_sharpness: f32,
    /// Mean noise level.
    pub mean_noise: f32,
    /// Mean exposure error.
    pub mean_exposure_error: f32,
    /// Mean chroma variance.
    pub mean_chroma_variance: f32,
    /// Overall composite quality (0.0–1.0).
    pub overall_quality: f32,
    /// Fraction of tiles flagged as defective (0.0–1.0).
    pub defect_ratio: f32,
    /// Whether the frame meets broadcast quality standards.
    pub broadcast_ready: bool,
    /// Index of the sharpest tile.
    pub sharpest_tile_index: usize,
    /// Index of the most defective tile.
    pub worst_tile_index: usize,
}

/// Complete quality map for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMap {
    /// Per-tile quality scores in row-major order.
    pub tiles: Vec<TileQuality>,
    /// Frame-level summary.
    pub summary: FrameQualitySummary,
    /// Grid columns.
    pub grid_cols: usize,
    /// Grid rows.
    pub grid_rows: usize,
}

impl QualityMap {
    /// Return the quality of a specific tile.
    #[must_use]
    pub fn tile(&self, col: usize, row: usize) -> Option<&TileQuality> {
        if col >= self.grid_cols || row >= self.grid_rows {
            return None;
        }
        self.tiles.get(row * self.grid_cols + col)
    }

    /// Return all tiles with a defect flag set.
    #[must_use]
    pub fn defective_tiles(&self) -> Vec<&TileQuality> {
        self.tiles.iter().filter(|t| t.has_defect).collect()
    }

    /// Return tiles sorted by composite quality (worst first).
    #[must_use]
    pub fn tiles_by_quality_asc(&self) -> Vec<&TileQuality> {
        let mut sorted: Vec<&TileQuality> = self.tiles.iter().collect();
        sorted.sort_by(|a, b| {
            a.composite
                .partial_cmp(&b.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Per-region visual quality analyzer.
#[derive(Debug, Clone)]
pub struct VisualQualityAnalyzer {
    /// Configuration.
    pub config: QualityMapConfig,
}

impl VisualQualityAnalyzer {
    /// Create a new analyzer.
    #[must_use]
    pub fn new(config: QualityMapConfig) -> Self {
        Self { config }
    }

    /// Analyze quality of an RGB frame.
    ///
    /// `rgb_data` must be packed `width × height × 3` bytes (R,G,B order).
    ///
    /// # Errors
    ///
    /// Returns [`SceneError::InvalidDimensions`] if the data size does not
    /// match the supplied dimensions, or if the grid configuration is invalid.
    pub fn analyze(&self, rgb_data: &[u8], width: usize, height: usize) -> SceneResult<QualityMap> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(format!(
                "Expected {} bytes, got {}",
                width * height * 3,
                rgb_data.len()
            )));
        }
        if width == 0 || height == 0 {
            return Err(SceneError::InvalidDimensions(
                "Width and height must be non-zero".to_string(),
            ));
        }
        let gc = self.config.grid_cols;
        let gr = self.config.grid_rows;
        if gc == 0 || gr == 0 {
            return Err(SceneError::InvalidParameter(
                "Grid dimensions must be non-zero".to_string(),
            ));
        }

        // Pre-compute luma plane
        let luma = luma_plane(rgb_data, width, height);

        let mut tiles = Vec::with_capacity(gc * gr);

        for row in 0..gr {
            for col in 0..gc {
                let (x0, x1) = tile_range(col, gc, width);
                let (y0, y1) = tile_range(row, gr, height);

                let sharpness = compute_sharpness(&luma, width, x0, x1, y0, y1);
                let noise = compute_noise(&luma, width, x0, x1, y0, y1);
                let (exposure_error, chroma_variance) =
                    compute_exposure_and_chroma(rgb_data, width, x0, x1, y0, y1);

                let composite =
                    compute_composite(sharpness, noise, exposure_error, chroma_variance);
                let has_defect = sharpness < self.config.sharpness_threshold
                    || noise > self.config.noise_threshold
                    || exposure_error > self.config.exposure_threshold;

                tiles.push(TileQuality {
                    col,
                    row,
                    sharpness,
                    noise,
                    exposure_error,
                    chroma_variance,
                    composite,
                    has_defect,
                });
            }
        }

        let summary = build_summary(&tiles, gc, gr, &self.config);
        Ok(QualityMap {
            tiles,
            summary,
            grid_cols: gc,
            grid_rows: gr,
        })
    }
}

// ---------------------------------------------------------------------------
// Quality metric implementations
// ---------------------------------------------------------------------------

/// Compute a sharpness score using variance of Laplacian.
///
/// Returns a value in [0, 1] where higher = sharper.
fn compute_sharpness(
    luma: &[f32],
    width: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
) -> f32 {
    let tw = x1 - x0;
    let th = y1 - y0;
    if tw < 3 || th < 3 {
        return 0.0;
    }

    let mut laplacian_vals = Vec::with_capacity(tw * th);

    for y in y0 + 1..y1 - 1 {
        for x in x0 + 1..x1 - 1 {
            let c = luma[y * width + x];
            let n = luma[(y - 1) * width + x];
            let s = luma[(y + 1) * width + x];
            let w = luma[y * width + x - 1];
            let e = luma[y * width + x + 1];
            let lap = (4.0 * c - n - s - w - e).abs();
            laplacian_vals.push(lap);
        }
    }

    if laplacian_vals.is_empty() {
        return 0.0;
    }

    let mean = laplacian_vals.iter().copied().sum::<f32>() / laplacian_vals.len() as f32;
    let variance = laplacian_vals
        .iter()
        .copied()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f32>()
        / laplacian_vals.len() as f32;

    // Normalise: variance of ~0.01 in [0,1] float range is already "sharp"
    let norm = (variance / 0.01).min(1.0);
    norm
}

/// Estimate noise using Median Absolute Deviation of the Laplacian.
///
/// Returns a value in [0, 1] where higher = noisier.
fn compute_noise(luma: &[f32], width: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> f32 {
    let tw = x1 - x0;
    let th = y1 - y0;
    if tw < 3 || th < 3 {
        return 0.0;
    }

    let mut lap_vals = Vec::with_capacity(tw * th);
    for y in y0 + 1..y1 - 1 {
        for x in x0 + 1..x1 - 1 {
            let c = luma[y * width + x];
            let n = luma[(y - 1) * width + x];
            let s = luma[(y + 1) * width + x];
            let w = luma[y * width + x - 1];
            let e = luma[y * width + x + 1];
            let lap = (4.0 * c - n - s - w - e).abs();
            lap_vals.push(lap);
        }
    }

    if lap_vals.is_empty() {
        return 0.0;
    }

    lap_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = lap_vals[lap_vals.len() / 2];
    let mad: f32 =
        lap_vals.iter().map(|&v| (v - median).abs()).sum::<f32>() / lap_vals.len() as f32;

    // Scale: noise sigma ~0.01 in float range → normalise
    (mad / 0.02).min(1.0)
}

/// Compute exposure error (distance from mid-grey) and chroma variance.
///
/// Returns (exposure_error ∈ [0,1], chroma_variance ∈ [0,1]).
fn compute_exposure_and_chroma(
    rgb: &[u8],
    width: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
) -> (f32, f32) {
    let mut sum_luma = 0.0_f32;
    let mut sat_vals = Vec::new();
    let mut count = 0usize;

    for y in y0..y1 {
        for x in x0..x1 {
            let base = (y * width + x) * 3;
            if base + 2 >= rgb.len() {
                continue;
            }
            let r = rgb[base] as f32 / 255.0;
            let g = rgb[base + 1] as f32 / 255.0;
            let b = rgb[base + 2] as f32 / 255.0;
            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            sum_luma += luma;

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let sat = if max < 1e-6 { 0.0 } else { (max - min) / max };
            sat_vals.push(sat);
            count += 1;
        }
    }

    if count == 0 {
        return (0.0, 0.0);
    }

    let mean_luma = sum_luma / count as f32;
    // Exposure error: 2× distance from 0.5 (0 = perfect, 1 = fully black/white)
    let exposure_error = ((mean_luma - 0.5).abs() * 2.0).min(1.0);

    // Chroma variance
    let mean_sat = sat_vals.iter().copied().sum::<f32>() / count as f32;
    let variance = sat_vals
        .iter()
        .copied()
        .map(|s| (s - mean_sat) * (s - mean_sat))
        .sum::<f32>()
        / count as f32;
    let chroma_variance = (variance.sqrt() / 0.3).min(1.0);

    (exposure_error, chroma_variance)
}

/// Composite quality score: high sharpness, low noise, low exposure error.
fn compute_composite(sharpness: f32, noise: f32, exposure_error: f32, chroma_variance: f32) -> f32 {
    let q = sharpness * 0.45
        + (1.0 - noise) * 0.30
        + (1.0 - exposure_error) * 0.20
        + (1.0 - chroma_variance) * 0.05;
    q.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Summary builder
// ---------------------------------------------------------------------------

fn build_summary(
    tiles: &[TileQuality],
    gc: usize,
    gr: usize,
    config: &QualityMapConfig,
) -> FrameQualitySummary {
    if tiles.is_empty() {
        return FrameQualitySummary {
            mean_sharpness: 0.0,
            mean_noise: 0.0,
            mean_exposure_error: 0.0,
            mean_chroma_variance: 0.0,
            overall_quality: 0.0,
            defect_ratio: 0.0,
            broadcast_ready: false,
            sharpest_tile_index: 0,
            worst_tile_index: 0,
        };
    }

    let cx = gc as f32 / 2.0;
    let cy = gr as f32 / 2.0;

    let mut w_sum = 0.0_f32;
    let mut wt_sharpness = 0.0_f32;
    let mut wt_noise = 0.0_f32;
    let mut wt_exposure = 0.0_f32;
    let mut wt_chroma = 0.0_f32;
    let mut defect_count = 0usize;
    let mut best_sharp = f32::NEG_INFINITY;
    let mut worst_comp = f32::INFINITY;
    let mut sharpest_idx = 0usize;
    let mut worst_idx = 0usize;

    for (i, t) in tiles.iter().enumerate() {
        let w = if config.centre_weighted {
            let dx = t.col as f32 + 0.5 - cx;
            let dy = t.row as f32 + 0.5 - cy;
            // Gaussian-like centre weight
            let d2 = (dx * dx + dy * dy) / ((cx * cx + cy * cy).max(1.0));
            1.0 + (-d2).exp()
        } else {
            1.0
        };
        wt_sharpness += t.sharpness * w;
        wt_noise += t.noise * w;
        wt_exposure += t.exposure_error * w;
        wt_chroma += t.chroma_variance * w;
        w_sum += w;

        if t.has_defect {
            defect_count += 1;
        }
        if t.sharpness > best_sharp {
            best_sharp = t.sharpness;
            sharpest_idx = i;
        }
        if t.composite < worst_comp {
            worst_comp = t.composite;
            worst_idx = i;
        }
    }

    let n = w_sum.max(1e-6);
    let mean_sharpness = wt_sharpness / n;
    let mean_noise = wt_noise / n;
    let mean_exposure_error = wt_exposure / n;
    let mean_chroma_variance = wt_chroma / n;

    let overall_quality = compute_composite(
        mean_sharpness,
        mean_noise,
        mean_exposure_error,
        mean_chroma_variance,
    );
    let defect_ratio = defect_count as f32 / tiles.len() as f32;

    let broadcast_ready = mean_sharpness >= config.sharpness_threshold
        && mean_noise <= config.noise_threshold
        && mean_exposure_error <= config.exposure_threshold
        && defect_ratio < 0.25;

    FrameQualitySummary {
        mean_sharpness,
        mean_noise,
        mean_exposure_error,
        mean_chroma_variance,
        overall_quality,
        defect_ratio,
        broadcast_ready,
        sharpest_tile_index: sharpest_idx,
        worst_tile_index: worst_idx,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert an RGB byte slice to a float luma plane.
fn luma_plane(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    let mut luma = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 3;
        if base + 2 < rgb.len() {
            let r = rgb[base] as f32 / 255.0;
            let g = rgb[base + 1] as f32 / 255.0;
            let b = rgb[base + 2] as f32 / 255.0;
            luma.push(0.299 * r + 0.587 * g + 0.114 * b);
        } else {
            luma.push(0.0);
        }
    }
    luma
}

/// Compute the pixel range [start, end) for a grid cell.
fn tile_range(cell: usize, cells: usize, total: usize) -> (usize, usize) {
    let start = cell * total / cells;
    let end = ((cell + 1) * total / cells).min(total);
    (start, end)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        (0..w * h).flat_map(|_| [r, g, b]).collect()
    }

    fn noisy_luma(w: usize, h: usize, base: u8, seed: u8) -> Vec<u8> {
        (0..w * h)
            .flat_map(|i| {
                let v = base.wrapping_add(((i as u8).wrapping_mul(seed)) % 80);
                [v, v, v]
            })
            .collect()
    }

    fn default_analyzer() -> VisualQualityAnalyzer {
        VisualQualityAnalyzer::new(QualityMapConfig::default())
    }

    #[test]
    fn test_tile_count() {
        let cfg = QualityMapConfig {
            grid_cols: 3,
            grid_rows: 2,
            ..Default::default()
        };
        let az = VisualQualityAnalyzer::new(cfg);
        let rgb = solid_rgb(128, 128, 128, 60, 40);
        let map = az.analyze(&rgb, 60, 40).unwrap();
        assert_eq!(map.tiles.len(), 6);
    }

    #[test]
    fn test_wrong_size_error() {
        let az = default_analyzer();
        let rgb = vec![0u8; 10];
        assert!(az.analyze(&rgb, 64, 64).is_err());
    }

    #[test]
    fn test_zero_dimensions_error() {
        let az = default_analyzer();
        let rgb = vec![];
        assert!(az.analyze(&rgb, 0, 0).is_err());
    }

    #[test]
    fn test_tile_accessor() {
        let cfg = QualityMapConfig {
            grid_cols: 3,
            grid_rows: 2,
            ..Default::default()
        };
        let az = VisualQualityAnalyzer::new(cfg);
        let rgb = solid_rgb(100, 100, 100, 60, 40);
        let map = az.analyze(&rgb, 60, 40).unwrap();
        assert!(map.tile(0, 0).is_some());
        assert!(map.tile(3, 0).is_none()); // out of bounds col
    }

    #[test]
    fn test_solid_frame_low_noise() {
        let az = default_analyzer();
        let rgb = solid_rgb(128, 128, 128, 64, 64);
        let map = az.analyze(&rgb, 64, 64).unwrap();
        // Solid colour → very low sharpness and very low noise
        assert!(
            map.summary.mean_noise < 0.3,
            "noise={}",
            map.summary.mean_noise
        );
    }

    #[test]
    fn test_noisy_frame_higher_noise() {
        let az = default_analyzer();
        let rgb = noisy_luma(64, 64, 128, 73);
        let map = az.analyze(&rgb, 64, 64).unwrap();
        // Noisy frame should score higher noise than solid
        let solid = {
            let rgb2 = solid_rgb(128, 128, 128, 64, 64);
            default_analyzer().analyze(&rgb2, 64, 64).unwrap()
        };
        assert!(
            map.summary.mean_noise >= solid.summary.mean_noise,
            "noisy={} solid={}",
            map.summary.mean_noise,
            solid.summary.mean_noise
        );
    }

    #[test]
    fn test_overexposed_frame() {
        let az = default_analyzer();
        let rgb = solid_rgb(255, 255, 255, 64, 64);
        let map = az.analyze(&rgb, 64, 64).unwrap();
        assert!(
            map.summary.mean_exposure_error > 0.5,
            "exposure_error={}",
            map.summary.mean_exposure_error
        );
    }

    #[test]
    fn test_defect_ratio_range() {
        let az = default_analyzer();
        let rgb = solid_rgb(10, 10, 10, 64, 64);
        let map = az.analyze(&rgb, 64, 64).unwrap();
        assert!((0.0..=1.0).contains(&map.summary.defect_ratio));
    }

    #[test]
    fn test_composite_range() {
        let az = default_analyzer();
        let rgb = solid_rgb(180, 90, 30, 64, 64);
        let map = az.analyze(&rgb, 64, 64).unwrap();
        for tile in &map.tiles {
            assert!(
                (0.0..=1.0).contains(&tile.composite),
                "composite={} out of range",
                tile.composite
            );
        }
        assert!((0.0..=1.0).contains(&map.summary.overall_quality));
    }

    #[test]
    fn test_tiles_sorted_ascending() {
        let az = default_analyzer();
        let rgb = noisy_luma(64, 64, 80, 101);
        let map = az.analyze(&rgb, 64, 64).unwrap();
        let sorted = map.tiles_by_quality_asc();
        for pair in sorted.windows(2) {
            assert!(pair[0].composite <= pair[1].composite);
        }
    }

    #[test]
    fn test_defective_tiles_all_flagged() {
        let az = default_analyzer();
        let rgb = solid_rgb(5, 5, 5, 64, 64); // very dark → exposure defect
        let map = az.analyze(&rgb, 64, 64).unwrap();
        for dt in map.defective_tiles() {
            assert!(dt.has_defect);
        }
    }
}
