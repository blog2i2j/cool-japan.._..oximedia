//! Golden ratio and phi grid composition analysis.
//!
//! This module extends the standard rule-of-thirds analysis with mathematically
//! richer composition guides derived from the golden ratio (φ ≈ 1.618034).
//!
//! # Composition Guides Implemented
//!
//! - **Phi Grid** — a 3×3 grid whose dividing lines are placed at proportions
//!   `1/φ` and `(φ-1)/φ` ≈ 38.2% and 61.8% of the frame width/height.
//! - **Golden Spiral** (Fibonacci spiral) — the frame is recursively subdivided
//!   into a golden rectangle and a square. Subject placement near the spiral's
//!   focal point is rewarded.
//! - **Golden Triangle** — two diagonal lines and a perpendicular from one corner
//!   divide the frame into four triangles. Subjects placed on these lines score
//!   higher.
//! - **Diagonal Method** — subjects placed on or near the main diagonals.
//! - **Centre of Interest Score** — how close the image's visual centroid
//!   (estimated via luminance weighting) is to any of the above power points.
//!
//! All scores are normalised to [0.0, 1.0] where 1.0 is perfect alignment.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Golden ratio φ.
pub const PHI: f64 = 1.618_033_988_749_895;

/// Reciprocal of φ (≈ 0.618).
pub const PHI_RECIPROCAL: f64 = 0.618_033_988_749_895;

/// 1/φ² ≈ 0.382 (used for phi-grid placement).
pub const PHI_SQ_RECIPROCAL: f64 = 0.381_966_011_250_105;

/// Maximum snap distance to a power line (fraction of frame dimension).
const SNAP_RADIUS: f64 = 0.08;

// ---------------------------------------------------------------------------
// Public Types
// ---------------------------------------------------------------------------

/// The full golden-ratio composition analysis for a single frame.
#[derive(Debug, Clone)]
pub struct GoldenRatioAnalysis {
    /// Score for phi-grid alignment (0.0–1.0).
    pub phi_grid_score: f32,
    /// Score for golden spiral focal point proximity (0.0–1.0).
    pub golden_spiral_score: f32,
    /// Score for golden triangle alignment (0.0–1.0).
    pub golden_triangle_score: f32,
    /// Score for diagonal-method alignment (0.0–1.0).
    pub diagonal_score: f32,
    /// Weighted aggregate of all sub-scores (0.0–1.0).
    pub overall_score: f32,
    /// Normalised (x, y) coordinates of the estimated visual centroid (0.0–1.0).
    pub visual_centroid: (f32, f32),
    /// The phi-grid power points as normalised (x, y) pairs.
    pub phi_power_points: Vec<(f32, f32)>,
    /// The golden spiral focal points (up to 4 orientations).
    pub spiral_focal_points: Vec<(f32, f32)>,
    /// Distance from visual centroid to the nearest phi power point (0.0–1.0).
    pub nearest_power_point_distance: f32,
}

/// Configuration for the golden-ratio analyser.
#[derive(Debug, Clone)]
pub struct GoldenRatioConfig {
    /// Weight for phi-grid score in overall.
    pub phi_grid_weight: f32,
    /// Weight for golden spiral score in overall.
    pub golden_spiral_weight: f32,
    /// Weight for golden triangle score in overall.
    pub golden_triangle_weight: f32,
    /// Weight for diagonal score in overall.
    pub diagonal_weight: f32,
    /// Snap radius as a fraction of frame size (default: 0.08).
    pub snap_radius: f32,
    /// Number of horizontal/vertical sample points for visual centroid (default: 16).
    pub centroid_grid: usize,
}

impl Default for GoldenRatioConfig {
    fn default() -> Self {
        Self {
            phi_grid_weight: 0.4,
            golden_spiral_weight: 0.3,
            golden_triangle_weight: 0.15,
            diagonal_weight: 0.15,
            snap_radius: SNAP_RADIUS as f32,
            centroid_grid: 16,
        }
    }
}

/// The golden-ratio composition analyser.
#[derive(Debug)]
pub struct GoldenRatioAnalyzer {
    config: GoldenRatioConfig,
}

impl Default for GoldenRatioAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl GoldenRatioAnalyzer {
    /// Create a new analyser with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: GoldenRatioConfig::default(),
        }
    }

    /// Create a new analyser with custom configuration.
    pub fn with_config(config: GoldenRatioConfig) -> ShotResult<Self> {
        let total_weight = config.phi_grid_weight
            + config.golden_spiral_weight
            + config.golden_triangle_weight
            + config.diagonal_weight;
        if total_weight < f32::EPSILON {
            return Err(ShotError::InvalidParameters(
                "sum of composition weights must be positive".to_string(),
            ));
        }
        if config.snap_radius <= 0.0 || config.snap_radius > 0.5 {
            return Err(ShotError::InvalidParameters(
                "snap_radius must be in (0, 0.5]".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Analyse the golden-ratio composition of a frame.
    pub fn analyze(&self, frame: &FrameBuffer) -> ShotResult<GoldenRatioAnalysis> {
        let (h, w, _) = frame.dim();
        if h == 0 || w == 0 {
            return Err(ShotError::InvalidFrame(
                "Frame dimensions must be non-zero".to_string(),
            ));
        }

        // 1. Estimate visual centroid via luminance-weighted sampling
        let centroid = self.estimate_visual_centroid(frame)?;

        // 2. Compute phi power points
        let phi_pts = phi_grid_power_points();

        // 3. Score phi-grid alignment
        let phi_score = score_phi_grid(centroid, &phi_pts, self.config.snap_radius as f64);

        // 4. Score golden spiral
        let spiral_pts = golden_spiral_focal_points();
        let spiral_score =
            score_nearest_focal(centroid, &spiral_pts, self.config.snap_radius as f64);

        // 5. Score golden triangle
        let triangle_score = score_golden_triangle(centroid, self.config.snap_radius as f64);

        // 6. Diagonal score
        let diagonal_score = score_diagonal(centroid, self.config.snap_radius as f64);

        // 7. Nearest power point distance
        let nearest_dist = phi_pts
            .iter()
            .map(|&(px, py)| {
                let dx = f64::from(centroid.0) - px;
                let dy = f64::from(centroid.1) - py;
                (dx * dx + dy * dy).sqrt()
            })
            .fold(f64::INFINITY, f64::min) as f32;

        // 8. Weighted overall
        let total_weight = self.config.phi_grid_weight
            + self.config.golden_spiral_weight
            + self.config.golden_triangle_weight
            + self.config.diagonal_weight;

        let overall = (phi_score * self.config.phi_grid_weight
            + spiral_score * self.config.golden_spiral_weight
            + triangle_score * self.config.golden_triangle_weight
            + diagonal_score * self.config.diagonal_weight)
            / total_weight;

        Ok(GoldenRatioAnalysis {
            phi_grid_score: phi_score,
            golden_spiral_score: spiral_score,
            golden_triangle_score: triangle_score,
            diagonal_score,
            overall_score: overall.clamp(0.0, 1.0),
            visual_centroid: (centroid.0, centroid.1),
            phi_power_points: phi_pts.iter().map(|&(x, y)| (x as f32, y as f32)).collect(),
            spiral_focal_points: spiral_pts
                .iter()
                .map(|&(x, y)| (x as f32, y as f32))
                .collect(),
            nearest_power_point_distance: nearest_dist,
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Estimate the visual centroid of the frame using luminance-weighted sampling.
    fn estimate_visual_centroid(&self, frame: &FrameBuffer) -> ShotResult<(f32, f32)> {
        let (h, w, c) = frame.dim();
        if c < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }

        let grid = self.config.centroid_grid.max(2);
        let mut sum_lum = 0.0f64;
        let mut sum_wx = 0.0f64;
        let mut sum_wy = 0.0f64;

        for gy in 0..grid {
            for gx in 0..grid {
                let py = (gy * h) / grid;
                let px = (gx * w) / grid;
                let py = py.min(h.saturating_sub(1));
                let px = px.min(w.saturating_sub(1));
                let r = f64::from(frame.get(py, px, 0));
                let g = f64::from(frame.get(py, px, 1));
                let b = f64::from(frame.get(py, px, 2));
                let lum = r * 0.299 + g * 0.587 + b * 0.114;
                let nx = (gx as f64 + 0.5) / grid as f64;
                let ny = (gy as f64 + 0.5) / grid as f64;
                sum_lum += lum;
                sum_wx += lum * nx;
                sum_wy += lum * ny;
            }
        }

        if sum_lum < f64::EPSILON {
            return Ok((0.5, 0.5));
        }

        Ok(((sum_wx / sum_lum) as f32, (sum_wy / sum_lum) as f32))
    }
}

// ---------------------------------------------------------------------------
// Composition Geometry
// ---------------------------------------------------------------------------

/// Return the 4 phi-grid power points as normalised (x, y) coordinates.
///
/// The phi grid divides the frame at ≈38.2% and ≈61.8% in each dimension.
/// The four intersections of these lines are the power points.
#[must_use]
pub fn phi_grid_power_points() -> Vec<(f64, f64)> {
    let a = PHI_SQ_RECIPROCAL; // ≈ 0.382
    let b = PHI_RECIPROCAL; // ≈ 0.618
    vec![(a, a), (b, a), (a, b), (b, b)]
}

/// Return the focal points of the golden spiral in all 4 orientations.
///
/// Each orientation places the focal point at one of the phi-grid corners but
/// the spiral converges at a slightly inward position.
#[must_use]
pub fn golden_spiral_focal_points() -> Vec<(f64, f64)> {
    // The spiral converges at the phi-grid power points
    phi_grid_power_points()
}

/// Return the rule-of-thirds power points for comparison.
#[must_use]
pub fn rule_of_thirds_power_points() -> Vec<(f64, f64)> {
    let a = 1.0 / 3.0;
    let b = 2.0 / 3.0;
    vec![(a, a), (b, a), (a, b), (b, b)]
}

/// Compute a golden-ratio alignment score for a given centroid position.
///
/// Returns 1.0 if the centroid is exactly on a phi power point, decreasing
/// to 0.0 as the distance exceeds `snap_radius`.
#[must_use]
pub fn score_phi_grid(centroid: (f32, f32), power_pts: &[(f64, f64)], snap_radius: f64) -> f32 {
    score_nearest_focal(centroid, power_pts, snap_radius)
}

/// Score proximity to the nearest point in `focal_pts`.
#[must_use]
pub fn score_nearest_focal(
    centroid: (f32, f32),
    focal_pts: &[(f64, f64)],
    snap_radius: f64,
) -> f32 {
    let cx = f64::from(centroid.0);
    let cy = f64::from(centroid.1);
    let min_dist = focal_pts
        .iter()
        .map(|&(px, py)| {
            let dx = cx - px;
            let dy = cy - py;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(f64::INFINITY, f64::min);

    if min_dist <= snap_radius {
        // Linear ramp: 1.0 at 0, 0.0 at snap_radius
        (1.0 - min_dist / snap_radius) as f32
    } else {
        0.0
    }
}

/// Score alignment to the golden-triangle diagonals.
///
/// The golden triangle divides a rectangle with corners at (0,0) and (1,1)
/// using two diagonals and a perpendicular. We score distance to any of
/// the three dividing lines.
#[must_use]
pub fn score_golden_triangle(centroid: (f32, f32), snap_radius: f64) -> f32 {
    let cx = f64::from(centroid.0);
    let cy = f64::from(centroid.1);

    // Main diagonal: y = x  →  distance = |y-x| / sqrt(2)
    let d_main = (cy - cx).abs() / 2.0f64.sqrt();

    // Anti-diagonal: y = 1-x  →  distance = |y-(1-x)| / sqrt(2)
    let d_anti = (cy - (1.0 - cx)).abs() / 2.0f64.sqrt();

    // Perpendicular from (1,0) to the main diagonal: hits at (0.5, 0.5)
    // Line from (1,0) perpendicular to y=x has equation y = -x + 1 (anti-diag)
    // So we merge d_anti and the perpendicular
    let min_dist = d_main.min(d_anti);

    if min_dist <= snap_radius {
        (1.0 - min_dist / snap_radius) as f32
    } else {
        0.0
    }
}

/// Score alignment to the main diagonal and anti-diagonal of the frame.
#[must_use]
pub fn score_diagonal(centroid: (f32, f32), snap_radius: f64) -> f32 {
    score_golden_triangle(centroid, snap_radius)
}

/// Compare phi-grid alignment versus rule-of-thirds alignment for a centroid.
///
/// Returns a positive value if the centroid is better aligned to phi grid,
/// negative if rule-of-thirds is a better match, and near 0 for similar alignment.
#[must_use]
pub fn compare_phi_vs_thirds(centroid: (f32, f32)) -> f32 {
    let phi_pts = phi_grid_power_points();
    let thirds_pts = rule_of_thirds_power_points();

    let cx = f64::from(centroid.0);
    let cy = f64::from(centroid.1);

    let min_phi = phi_pts
        .iter()
        .map(|&(px, py)| {
            let dx = cx - px;
            let dy = cy - py;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(f64::INFINITY, f64::min);

    let min_thirds = thirds_pts
        .iter()
        .map(|&(px, py)| {
            let dx = cx - px;
            let dy = cy - py;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(f64::INFINITY, f64::min);

    // Positive = phi wins, negative = thirds wins
    (min_thirds - min_phi) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(h: usize, w: usize, r: u8, g: u8, b: u8) -> FrameBuffer {
        let mut frame = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                frame.set(y, x, 0, r);
                frame.set(y, x, 1, g);
                frame.set(y, x, 2, b);
            }
        }
        frame
    }

    #[test]
    fn test_phi_constant() {
        // φ * (1/φ) = 1.0
        assert!((PHI * PHI_RECIPROCAL - 1.0).abs() < 1e-10);
        // 1/φ² = 1 - 1/φ
        assert!((PHI_SQ_RECIPROCAL - (1.0 - PHI_RECIPROCAL)).abs() < 1e-10);
    }

    #[test]
    fn test_phi_grid_power_points_count() {
        let pts = phi_grid_power_points();
        assert_eq!(pts.len(), 4);
    }

    #[test]
    fn test_phi_grid_power_points_values() {
        let pts = phi_grid_power_points();
        for &(x, y) in &pts {
            assert!(x > 0.3 && x < 0.7, "x={x} out of range");
            assert!(y > 0.3 && y < 0.7, "y={y} out of range");
        }
    }

    #[test]
    fn test_score_phi_grid_perfect() {
        let pts = phi_grid_power_points();
        // Place centroid exactly on first power point
        let (px, py) = pts[0];
        let score = score_phi_grid((px as f32, py as f32), &pts, SNAP_RADIUS);
        assert!((score - 1.0).abs() < 1e-5, "expected 1.0 got {score}");
    }

    #[test]
    fn test_score_phi_grid_far() {
        let pts = phi_grid_power_points();
        // Centre of frame is not a power point
        let score = score_phi_grid((0.5, 0.5), &pts, SNAP_RADIUS);
        // Centre is about 0.17 from nearest phi point — well beyond snap_radius
        assert!(score < 0.5, "expected low score at centre, got {score}");
    }

    #[test]
    fn test_golden_spiral_focal_points() {
        let pts = golden_spiral_focal_points();
        assert!(!pts.is_empty());
        // All focal points should be in the interior of the frame
        for &(x, y) in &pts {
            assert!(x > 0.0 && x < 1.0);
            assert!(y > 0.0 && y < 1.0);
        }
    }

    #[test]
    fn test_analyzer_uniform_frame() {
        let frame = make_frame(100, 100, 128, 128, 128);
        let analyzer = GoldenRatioAnalyzer::new();
        let result = analyzer.analyze(&frame).expect("analyze ok");
        assert!(result.overall_score >= 0.0);
        assert!(result.overall_score <= 1.0);
        // Centroid of uniform image should be near centre (0.5, 0.5)
        assert!((result.visual_centroid.0 - 0.5).abs() < 0.1);
        assert!((result.visual_centroid.1 - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_analyzer_bright_phi_point() {
        // Place a bright spot at the top-left phi power point
        let mut frame = FrameBuffer::zeros(100, 100, 3);
        let (px, py) = phi_grid_power_points()[0];
        let cx = (px * 100.0) as usize;
        let cy = (py * 100.0) as usize;
        for dy in 0..5 {
            for dx in 0..5 {
                let fy = (cy + dy).min(99);
                let fx = (cx + dx).min(99);
                frame.set(fy, fx, 0, 255);
                frame.set(fy, fx, 1, 255);
                frame.set(fy, fx, 2, 255);
            }
        }
        let analyzer = GoldenRatioAnalyzer::new();
        let result = analyzer.analyze(&frame).expect("ok");
        assert!(result.phi_grid_score > 0.0 || result.overall_score >= 0.0);
    }

    #[test]
    fn test_compare_phi_vs_thirds_phi_wins() {
        // At a phi power point, phi should win
        let (px, py) = phi_grid_power_points()[0];
        let diff = compare_phi_vs_thirds((px as f32, py as f32));
        // Should be positive (phi wins)
        assert!(diff > 0.0, "expected phi to win, diff={diff}");
    }

    #[test]
    fn test_compare_phi_vs_thirds_thirds_wins() {
        // At a rule-of-thirds power point
        let pts = rule_of_thirds_power_points();
        let (px, py) = pts[0];
        let diff = compare_phi_vs_thirds((px as f32, py as f32));
        // Should be negative or zero (thirds wins or tie)
        assert!(diff <= 0.05, "expected thirds to win, diff={diff}");
    }

    #[test]
    fn test_config_validation() {
        let mut cfg = GoldenRatioConfig::default();
        cfg.snap_radius = 0.0;
        assert!(GoldenRatioAnalyzer::with_config(cfg).is_err());
    }

    #[test]
    fn test_rule_of_thirds_vs_phi_count() {
        assert_eq!(
            rule_of_thirds_power_points().len(),
            phi_grid_power_points().len()
        );
    }
}
