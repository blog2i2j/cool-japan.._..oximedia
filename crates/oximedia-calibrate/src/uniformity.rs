//! Display uniformity measurement and analysis.
//!
//! Provides tools for measuring display uniformity across a grid of
//! measurement patches, computing luminance ratios, hot-spot detection,
//! and generating uniformity compensation maps.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// UniformityPatch
// ---------------------------------------------------------------------------

/// A single uniformity measurement patch at a screen grid position.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformityPatch {
    /// Column index (0-based, left-to-right).
    pub col: usize,
    /// Row index (0-based, top-to-bottom).
    pub row: usize,
    /// Measured luminance in cd/m².
    pub luminance: f64,
    /// Measured white-point x chromaticity.
    pub chroma_x: f64,
    /// Measured white-point y chromaticity.
    pub chroma_y: f64,
}

impl UniformityPatch {
    /// Create a new uniformity patch.
    #[must_use]
    pub fn new(col: usize, row: usize, luminance: f64, chroma_x: f64, chroma_y: f64) -> Self {
        Self {
            col,
            row,
            luminance,
            chroma_x,
            chroma_y,
        }
    }

    /// Luminance uniformity ratio relative to `reference_luminance`.
    ///
    /// Returns `luminance / reference_luminance`.  A perfect display returns
    /// `1.0` everywhere; values < 1.0 indicate dark corners/edges.
    #[must_use]
    pub fn luminance_ratio(&self, reference_luminance: f64) -> f64 {
        if reference_luminance.abs() < f64::EPSILON {
            return 0.0;
        }
        self.luminance / reference_luminance
    }
}

// ---------------------------------------------------------------------------
// UniformityGrid
// ---------------------------------------------------------------------------

/// A rectangular grid of uniformity measurements.
#[derive(Debug, Clone)]
pub struct UniformityGrid {
    /// Number of columns in the grid.
    pub cols: usize,
    /// Number of rows in the grid.
    pub rows: usize,
    /// All patches, stored in row-major order.
    patches: Vec<UniformityPatch>,
}

impl UniformityGrid {
    /// Create a uniformity grid from a list of patches.
    ///
    /// Patches need not be in any particular order; they will be sorted
    /// internally by `(row, col)`.
    #[must_use]
    pub fn new(cols: usize, rows: usize, mut patches: Vec<UniformityPatch>) -> Self {
        patches.sort_by_key(|p| (p.row, p.col));
        Self {
            cols,
            rows,
            patches,
        }
    }

    /// Return the patch at `(col, row)`, if it exists.
    #[must_use]
    pub fn get(&self, col: usize, row: usize) -> Option<&UniformityPatch> {
        self.patches.iter().find(|p| p.col == col && p.row == row)
    }

    /// Maximum luminance across all patches.
    #[must_use]
    pub fn max_luminance(&self) -> f64 {
        self.patches
            .iter()
            .map(|p| p.luminance)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum luminance across all patches.
    #[must_use]
    pub fn min_luminance(&self) -> f64 {
        self.patches
            .iter()
            .map(|p| p.luminance)
            .fold(f64::INFINITY, f64::min)
    }

    /// Mean luminance across all patches.
    #[must_use]
    pub fn mean_luminance(&self) -> f64 {
        if self.patches.is_empty() {
            return 0.0;
        }
        self.patches.iter().map(|p| p.luminance).sum::<f64>() / self.patches.len() as f64
    }

    /// Uniformity percentage: `(min / max) * 100`.
    ///
    /// A perfect display returns `100.0 %`.  A value of `80 %` means the
    /// darkest patch is 80 % of the brightest.
    #[must_use]
    pub fn uniformity_pct(&self) -> f64 {
        let max = self.max_luminance();
        if max <= 0.0 {
            return 0.0;
        }
        (self.min_luminance() / max) * 100.0
    }

    /// Identify hot-spot patches — those where luminance exceeds
    /// `mean_luminance * threshold_factor`.
    #[must_use]
    pub fn hot_spots(&self, threshold_factor: f64) -> Vec<&UniformityPatch> {
        let mean = self.mean_luminance();
        let threshold = mean * threshold_factor;
        self.patches
            .iter()
            .filter(|p| p.luminance > threshold)
            .collect()
    }

    /// Generate a compensation gain map.
    ///
    /// For each patch the gain is `target_luminance / patch.luminance`,
    /// clamped to `[0.1, 10.0]`.  Gains are returned in the same order as the
    /// internal (sorted) patch list.
    #[must_use]
    pub fn compensation_map(&self, target_luminance: f64) -> Vec<f64> {
        self.patches
            .iter()
            .map(|p| {
                if p.luminance <= 0.0 {
                    1.0
                } else {
                    (target_luminance / p.luminance).clamp(0.1, 10.0)
                }
            })
            .collect()
    }

    /// Chromaticity standard deviation across all patches (for x chromaticity).
    #[must_use]
    pub fn chroma_x_std_dev(&self) -> f64 {
        if self.patches.len() < 2 {
            return 0.0;
        }
        let n = self.patches.len() as f64;
        let mean = self.patches.iter().map(|p| p.chroma_x).sum::<f64>() / n;
        let variance = self
            .patches
            .iter()
            .map(|p| {
                let d = p.chroma_x - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        variance.sqrt()
    }
}

// ---------------------------------------------------------------------------
// UniformityReport
// ---------------------------------------------------------------------------

/// A summary report of display uniformity analysis.
#[derive(Debug, Clone)]
pub struct UniformityReport {
    /// Number of patches measured.
    pub patch_count: usize,
    /// Grid dimensions (cols × rows).
    pub grid_dims: (usize, usize),
    /// Uniformity percentage (min/max luminance ratio × 100).
    pub uniformity_pct: f64,
    /// Maximum luminance in cd/m².
    pub max_luminance: f64,
    /// Minimum luminance in cd/m².
    pub min_luminance: f64,
    /// Mean luminance in cd/m².
    pub mean_luminance: f64,
    /// Number of identified hot-spot patches.
    pub hot_spot_count: usize,
    /// Standard deviation of x chromaticity.
    pub chroma_x_std_dev: f64,
    /// Overall assessment.
    pub assessment: UniformityAssessment,
}

/// Qualitative assessment of display uniformity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniformityAssessment {
    /// Excellent uniformity (≥ 90 %).
    Excellent,
    /// Good uniformity (80 – 90 %).
    Good,
    /// Acceptable uniformity (70 – 80 %).
    Acceptable,
    /// Poor uniformity (< 70 %).
    Poor,
}

impl UniformityAssessment {
    /// Derive an assessment from the uniformity percentage.
    #[must_use]
    pub fn from_pct(pct: f64) -> Self {
        if pct >= 90.0 {
            Self::Excellent
        } else if pct >= 80.0 {
            Self::Good
        } else if pct >= 70.0 {
            Self::Acceptable
        } else {
            Self::Poor
        }
    }
}

impl UniformityReport {
    /// Generate a report from a `UniformityGrid`.
    #[must_use]
    pub fn from_grid(grid: &UniformityGrid) -> Self {
        let uniformity_pct = grid.uniformity_pct();
        Self {
            patch_count: grid.patches.len(),
            grid_dims: (grid.cols, grid.rows),
            uniformity_pct,
            max_luminance: grid.max_luminance(),
            min_luminance: grid.min_luminance(),
            mean_luminance: grid.mean_luminance(),
            hot_spot_count: grid.hot_spots(1.2).len(),
            chroma_x_std_dev: grid.chroma_x_std_dev(),
            assessment: UniformityAssessment::from_pct(uniformity_pct),
        }
    }

    /// Returns `true` if the display passes a minimum uniformity threshold.
    #[must_use]
    pub fn passes(&self, min_uniformity_pct: f64) -> bool {
        self.uniformity_pct >= min_uniformity_pct
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid_5x5(luminances: &[f64]) -> UniformityGrid {
        assert_eq!(luminances.len(), 25);
        let patches: Vec<UniformityPatch> = luminances
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let col = i % 5;
                let row = i / 5;
                UniformityPatch::new(col, row, l, 0.313, 0.329)
            })
            .collect();
        UniformityGrid::new(5, 5, patches)
    }

    fn perfect_grid() -> UniformityGrid {
        make_grid_5x5(&[100.0; 25])
    }

    fn imperfect_grid() -> UniformityGrid {
        let mut lums = [100.0_f64; 25];
        lums[0] = 75.0; // dark corner
        lums[24] = 120.0; // bright corner
        make_grid_5x5(&lums)
    }

    // ── UniformityPatch ──────────────────────────────────────────────────

    #[test]
    fn test_patch_luminance_ratio_unity() {
        let p = UniformityPatch::new(0, 0, 100.0, 0.313, 0.329);
        assert!((p.luminance_ratio(100.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_patch_luminance_ratio_dark() {
        let p = UniformityPatch::new(1, 1, 80.0, 0.313, 0.329);
        assert!((p.luminance_ratio(100.0) - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_patch_luminance_ratio_zero_reference() {
        let p = UniformityPatch::new(0, 0, 100.0, 0.313, 0.329);
        assert!((p.luminance_ratio(0.0)).abs() < 1e-9);
    }

    // ── UniformityGrid ───────────────────────────────────────────────────

    #[test]
    fn test_grid_max_luminance_perfect() {
        let grid = perfect_grid();
        assert!((grid.max_luminance() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_grid_min_luminance_imperfect() {
        let grid = imperfect_grid();
        assert!((grid.min_luminance() - 75.0).abs() < 1e-9);
    }

    #[test]
    fn test_grid_mean_luminance() {
        let grid = perfect_grid();
        assert!((grid.mean_luminance() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_grid_uniformity_pct_perfect() {
        let grid = perfect_grid();
        assert!((grid.uniformity_pct() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_grid_uniformity_pct_imperfect() {
        let grid = imperfect_grid();
        // min=75, max=120 → 62.5 %
        let pct = grid.uniformity_pct();
        assert!((pct - 62.5).abs() < 0.01, "pct={pct}");
    }

    #[test]
    fn test_grid_hot_spots_none_for_perfect() {
        let grid = perfect_grid();
        assert!(grid.hot_spots(1.2).is_empty());
    }

    #[test]
    fn test_grid_hot_spots_detected() {
        let grid = imperfect_grid();
        let hs = grid.hot_spots(1.1);
        assert!(!hs.is_empty(), "should detect hot spots");
    }

    #[test]
    fn test_grid_compensation_map_length() {
        let grid = perfect_grid();
        let map = grid.compensation_map(100.0);
        assert_eq!(map.len(), 25);
    }

    #[test]
    fn test_grid_compensation_map_uniform_gives_ones() {
        let grid = perfect_grid();
        let map = grid.compensation_map(100.0);
        for &g in &map {
            assert!((g - 1.0).abs() < 1e-9, "gain={g}");
        }
    }

    #[test]
    fn test_grid_get_patch() {
        let grid = perfect_grid();
        let p = grid.get(2, 2);
        assert!(p.is_some());
        assert_eq!(p.expect("expected p to be Some/Ok").col, 2);
        assert_eq!(p.expect("expected p to be Some/Ok").row, 2);
    }

    #[test]
    fn test_grid_chroma_x_std_dev_zero_for_perfect() {
        let grid = perfect_grid();
        assert!(grid.chroma_x_std_dev() < 1e-9);
    }

    // ── UniformityAssessment ─────────────────────────────────────────────

    #[test]
    fn test_assessment_excellent() {
        assert_eq!(
            UniformityAssessment::from_pct(95.0),
            UniformityAssessment::Excellent
        );
    }

    #[test]
    fn test_assessment_good() {
        assert_eq!(
            UniformityAssessment::from_pct(85.0),
            UniformityAssessment::Good
        );
    }

    #[test]
    fn test_assessment_acceptable() {
        assert_eq!(
            UniformityAssessment::from_pct(75.0),
            UniformityAssessment::Acceptable
        );
    }

    #[test]
    fn test_assessment_poor() {
        assert_eq!(
            UniformityAssessment::from_pct(60.0),
            UniformityAssessment::Poor
        );
    }

    // ── UniformityReport ─────────────────────────────────────────────────

    #[test]
    fn test_report_from_grid_perfect() {
        let grid = perfect_grid();
        let report = UniformityReport::from_grid(&grid);
        assert_eq!(report.assessment, UniformityAssessment::Excellent);
        assert!(report.passes(90.0));
    }

    #[test]
    fn test_report_from_grid_imperfect() {
        let grid = imperfect_grid();
        let report = UniformityReport::from_grid(&grid);
        assert_eq!(report.assessment, UniformityAssessment::Poor);
        assert!(!report.passes(90.0));
    }

    #[test]
    fn test_report_patch_count() {
        let grid = perfect_grid();
        let report = UniformityReport::from_grid(&grid);
        assert_eq!(report.patch_count, 25);
    }

    #[test]
    fn test_report_grid_dims() {
        let grid = perfect_grid();
        let report = UniformityReport::from_grid(&grid);
        assert_eq!(report.grid_dims, (5, 5));
    }
}
