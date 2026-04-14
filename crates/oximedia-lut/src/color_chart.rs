//! Macbeth/ColorChecker chart analysis and LUT generation from chart photos.
//!
//! This module provides tools for analysing colour reference charts (such as the
//! X-Rite/Gretag Macbeth ColorChecker Classic, ColorChecker Passport, and the
//! IT8.7 target) and generating correction LUTs that transform camera-captured
//! colours to standard reference values.
//!
//! # Workflow
//!
//! 1. Provide the measured `[R, G, B]` colours sampled from a chart photo.
//! 2. Provide the reference `[R, G, B]` or `[L*, a*, b*]` values for each patch.
//! 3. Fit a colour correction matrix (or polynomial) using the measured/reference pairs.
//! 4. Generate a 3-D LUT that applies the correction transform.
//!
//! # Example
//!
//! ```rust
//! use oximedia_lut::color_chart::{ChartPatch, ColorChart, ColorChartAnalyzer};
//!
//! // Minimal 4-patch example (full chart has 24 patches)
//! let measured = vec![
//!     [0.18_f64, 0.10, 0.08],  // dark skin
//!     [0.35, 0.22, 0.17],       // light skin
//!     [0.10, 0.17, 0.25],       // blue sky
//!     [0.18, 0.18, 0.18],       // neutral grey
//! ];
//! let reference = vec![
//!     [0.20_f64, 0.12, 0.09],
//!     [0.38, 0.25, 0.19],
//!     [0.11, 0.19, 0.28],
//!     [0.18, 0.18, 0.18],
//! ];
//!
//! let chart = ColorChart::from_patches(measured, reference).unwrap();
//! let analyzer = ColorChartAnalyzer::new(chart);
//! let matrix = analyzer.fit_correction_matrix();
//! assert_eq!(matrix.len(), 3);
//! ```

use crate::error::{LutError, LutResult};
use crate::Rgb;

// ============================================================================
// Chart patch
// ============================================================================

/// A single patch from a colour reference chart.
#[derive(Debug, Clone)]
pub struct ChartPatch {
    /// Name/label of this patch (e.g. `"Dark Skin"`, `"Neutral 5"`).
    pub name: String,
    /// Measured RGB value sampled from the camera image (normalised `[0, 1]`).
    pub measured: Rgb,
    /// Reference RGB value for this patch in the target colour space (`[0, 1]`).
    pub reference: Rgb,
}

impl ChartPatch {
    /// Create a new patch with a label.
    #[must_use]
    pub fn new(name: impl Into<String>, measured: Rgb, reference: Rgb) -> Self {
        Self {
            name: name.into(),
            measured,
            reference,
        }
    }

    /// Compute the per-channel absolute error between measured and reference.
    #[must_use]
    pub fn error(&self) -> Rgb {
        [
            (self.measured[0] - self.reference[0]).abs(),
            (self.measured[1] - self.reference[1]).abs(),
            (self.measured[2] - self.reference[2]).abs(),
        ]
    }

    /// Compute the Euclidean colour error (ΔE in a simplified RGB metric).
    #[must_use]
    pub fn delta_e_rgb(&self) -> f64 {
        let e = self.error();
        (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
    }
}

// ============================================================================
// Colour chart
// ============================================================================

/// A colour reference chart with measured and reference patch pairs.
#[derive(Debug, Clone)]
pub struct ColorChart {
    /// Ordered list of chart patches.
    pub patches: Vec<ChartPatch>,
}

impl ColorChart {
    /// Create a chart from two parallel vectors of measured and reference colours.
    ///
    /// # Errors
    ///
    /// Returns [`LutError::InvalidData`] if the vectors have different lengths
    /// or are empty.
    pub fn from_patches(measured: Vec<Rgb>, reference: Vec<Rgb>) -> LutResult<Self> {
        if measured.is_empty() {
            return Err(LutError::InvalidData(
                "Colour chart must have at least one patch".to_string(),
            ));
        }
        if measured.len() != reference.len() {
            return Err(LutError::InvalidData(format!(
                "Measured ({}) and reference ({}) patch counts differ",
                measured.len(),
                reference.len()
            )));
        }
        let patches = measured
            .into_iter()
            .zip(reference)
            .enumerate()
            .map(|(i, (m, r))| ChartPatch::new(format!("Patch {}", i + 1), m, r))
            .collect();
        Ok(Self { patches })
    }

    /// Create a chart from named patches.
    #[must_use]
    pub fn from_named_patches(patches: Vec<ChartPatch>) -> Self {
        Self { patches }
    }

    /// Return the Macbeth ColorChecker Classic reference values (sRGB, linearised).
    ///
    /// The 24-patch reference values are from the X-Rite specification.
    /// Values are in linear light (not gamma-encoded).
    #[must_use]
    pub fn macbeth_classic_reference() -> Vec<Rgb> {
        // Reference linear sRGB values derived from the ColorChecker specification
        // (D50 adapted, linearised from sRGB gamma).
        vec![
            // Row 1: Skin tones + naturals
            [0.117_7, 0.087_4, 0.058_2], // 1 Dark Skin
            [0.384_5, 0.304_2, 0.241_0], // 2 Light Skin
            [0.103_0, 0.149_5, 0.285_3], // 3 Blue Sky
            [0.098_4, 0.141_5, 0.066_4], // 4 Foliage
            [0.208_0, 0.205_5, 0.397_0], // 5 Blue Flower
            [0.123_0, 0.385_0, 0.366_0], // 6 Bluish Green
            // Row 2: Chromatic patches
            [0.466_0, 0.215_0, 0.048_5], // 7 Orange
            [0.085_5, 0.102_5, 0.374_0], // 8 Purplish Blue
            [0.331_0, 0.095_5, 0.107_0], // 9 Moderate Red
            [0.068_5, 0.035_5, 0.110_0], // 10 Purple
            [0.224_0, 0.368_0, 0.077_0], // 11 Yellow Green
            [0.479_0, 0.319_0, 0.032_0], // 12 Orange Yellow
            // Row 3: Saturated primaries
            [0.044_0, 0.047_0, 0.232_0], // 13 Blue
            [0.075_5, 0.229_0, 0.072_0], // 14 Green
            [0.267_0, 0.047_5, 0.049_0], // 15 Red
            [0.480_0, 0.440_0, 0.023_5], // 16 Yellow
            [0.289_0, 0.069_0, 0.210_0], // 17 Magenta
            [0.046_5, 0.200_0, 0.373_0], // 18 Cyan
            // Row 4: Neutral patches
            [0.914_0, 0.914_0, 0.914_0], // 19 White 9.5
            [0.591_0, 0.591_0, 0.591_0], // 20 Neutral 8
            [0.362_0, 0.362_0, 0.362_0], // 21 Neutral 6.5
            [0.195_0, 0.195_0, 0.195_0], // 22 Neutral 5
            [0.090_0, 0.090_0, 0.090_0], // 23 Neutral 3.5
            [0.031_0, 0.031_0, 0.031_0], // 24 Black 2
        ]
    }

    /// Compute the mean ΔE error across all patches.
    #[must_use]
    pub fn mean_delta_e(&self) -> f64 {
        if self.patches.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.patches.iter().map(|p| p.delta_e_rgb()).sum();
        sum / self.patches.len() as f64
    }

    /// Compute the maximum ΔE error across all patches.
    #[must_use]
    pub fn max_delta_e(&self) -> f64 {
        self.patches
            .iter()
            .map(|p| p.delta_e_rgb())
            .fold(0.0_f64, f64::max)
    }

    /// Return a list of the worst (highest ΔE) patches in descending order.
    #[must_use]
    pub fn worst_patches(&self, n: usize) -> Vec<&ChartPatch> {
        let mut indexed: Vec<(usize, f64)> = self
            .patches
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.delta_e_rgb()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .into_iter()
            .take(n)
            .map(|(i, _)| &self.patches[i])
            .collect()
    }
}

// ============================================================================
// Colour correction fitting
// ============================================================================

/// Least-squares colour correction matrix fitter and LUT generator.
#[derive(Debug, Clone)]
pub struct ColorChartAnalyzer {
    /// The chart data used for fitting.
    pub chart: ColorChart,
}

impl ColorChartAnalyzer {
    /// Create a new analyser for the given chart.
    #[must_use]
    pub fn new(chart: ColorChart) -> Self {
        Self { chart }
    }

    /// Fit a 3×3 colour correction matrix using least-squares regression.
    ///
    /// The matrix `M` satisfies `reference ≈ M * measured` in the least-squares sense.
    ///
    /// The returned matrix is in row-major order: `M[row][col]`.
    #[must_use]
    pub fn fit_correction_matrix(&self) -> [[f64; 3]; 3] {
        let patches = &self.chart.patches;
        if patches.is_empty() {
            return identity_matrix();
        }

        // Solve M for each output channel independently (3 separate LS problems).
        // For channel k: find row_k such that row_k · measured_i ≈ reference_i[k]
        let n = patches.len();
        let mut ata = [[0.0f64; 3]; 3]; // A^T A
        let mut atb = [[0.0f64; 3]; 3]; // A^T B  (one column per output channel)

        for p in patches {
            let m = p.measured;
            for i in 0..3 {
                for j in 0..3 {
                    ata[i][j] += m[i] * m[j];
                }
                for k in 0..3 {
                    atb[i][k] += m[i] * p.reference[k];
                }
            }
        }
        let _ = n; // used implicitly via loop

        // Solve A^T A * x = A^T b for each output channel
        match solve_3x3_system(&ata, &atb) {
            Some(solution) => solution,
            None => identity_matrix(),
        }
    }

    /// Fit a polynomial (degree-2) colour correction per channel.
    ///
    /// Uses the terms `[R, G, B, R*G, R*B, G*B, R², G², B²]` (9 terms) to
    /// model cross-channel interactions and saturation non-linearities.
    ///
    /// Returns a `3 × 9` coefficient matrix (one row per output channel).
    #[must_use]
    pub fn fit_polynomial_correction(&self) -> [[f64; 9]; 3] {
        let patches = &self.chart.patches;
        if patches.is_empty() {
            return [[0.0; 9]; 3];
        }

        let n = patches.len();
        let mut features: Vec<[f64; 9]> = Vec::with_capacity(n);
        for p in patches {
            let [r, g, b] = p.measured;
            features.push([r, g, b, r * g, r * b, g * b, r * r, g * g, b * b]);
        }

        // For each output channel solve a 9×9 normal-equation system
        let mut result = [[0.0f64; 9]; 3];
        for ch in 0..3 {
            let targets: Vec<f64> = patches.iter().map(|p| p.reference[ch]).collect();
            if let Some(coeffs) = least_squares_9(&features, &targets) {
                result[ch] = coeffs;
            }
        }
        result
    }

    /// Apply the fitted correction matrix to an RGB value.
    #[must_use]
    pub fn apply_matrix_correction(&self, rgb: &Rgb, matrix: &[[f64; 3]; 3]) -> Rgb {
        [
            (matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2]).clamp(0.0, 1.0),
            (matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2]).clamp(0.0, 1.0),
            (matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2]).clamp(0.0, 1.0),
        ]
    }

    /// Apply polynomial correction coefficients to an RGB value.
    #[must_use]
    pub fn apply_polynomial_correction(&self, rgb: &Rgb, coeffs: &[[f64; 9]; 3]) -> Rgb {
        let [r, g, b] = *rgb;
        let feat = [r, g, b, r * g, r * b, g * b, r * r, g * g, b * b];
        let mut out = [0.0f64; 3];
        for ch in 0..3 {
            for (i, &c) in coeffs[ch].iter().enumerate() {
                out[ch] += c * feat[i];
            }
            out[ch] = out[ch].clamp(0.0, 1.0);
        }
        out
    }

    /// Generate a 3-D correction LUT from the fitted matrix.
    ///
    /// Returns `size³` entries in `[r][g][b]` index order.
    #[must_use]
    pub fn generate_3d_lut(&self, size: usize) -> Vec<Rgb> {
        let size = size.max(2);
        let matrix = self.fit_correction_matrix();
        let scale = (size - 1) as f64;

        let mut lut = Vec::with_capacity(size * size * size);
        for ri in 0..size {
            for gi in 0..size {
                for bi in 0..size {
                    let rgb: Rgb = [ri as f64 / scale, gi as f64 / scale, bi as f64 / scale];
                    lut.push(self.apply_matrix_correction(&rgb, &matrix));
                }
            }
        }
        lut
    }

    /// Validate the fit by computing residual ΔE on the training patches.
    #[must_use]
    pub fn residual_delta_e(&self) -> f64 {
        let matrix = self.fit_correction_matrix();
        let mut total = 0.0f64;
        for p in &self.chart.patches {
            let corrected = self.apply_matrix_correction(&p.measured, &matrix);
            let de = (corrected[0] - p.reference[0]).powi(2)
                + (corrected[1] - p.reference[1]).powi(2)
                + (corrected[2] - p.reference[2]).powi(2);
            total += de.sqrt();
        }
        if self.chart.patches.is_empty() {
            0.0
        } else {
            total / self.chart.patches.len() as f64
        }
    }
}

// ============================================================================
// Math helpers
// ============================================================================

fn identity_matrix() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Solve `A * x = b` for a 3×3 system using Cramer's rule.
/// Returns `None` if the system is singular.
fn solve_3x3_system(ata: &[[f64; 3]; 3], atb: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = ata[0][0] * (ata[1][1] * ata[2][2] - ata[1][2] * ata[2][1])
        - ata[0][1] * (ata[1][0] * ata[2][2] - ata[1][2] * ata[2][0])
        + ata[0][2] * (ata[1][0] * ata[2][1] - ata[1][1] * ata[2][0]);

    if det.abs() < 1e-15 {
        return None;
    }

    // Invert A^T A
    let d = 1.0 / det;
    let inv = [
        [
            (ata[1][1] * ata[2][2] - ata[1][2] * ata[2][1]) * d,
            (ata[0][2] * ata[2][1] - ata[0][1] * ata[2][2]) * d,
            (ata[0][1] * ata[1][2] - ata[0][2] * ata[1][1]) * d,
        ],
        [
            (ata[1][2] * ata[2][0] - ata[1][0] * ata[2][2]) * d,
            (ata[0][0] * ata[2][2] - ata[0][2] * ata[2][0]) * d,
            (ata[0][2] * ata[1][0] - ata[0][0] * ata[1][2]) * d,
        ],
        [
            (ata[1][0] * ata[2][1] - ata[1][1] * ata[2][0]) * d,
            (ata[0][1] * ata[2][0] - ata[0][0] * ata[2][1]) * d,
            (ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]) * d,
        ],
    ];

    // x = inv(A^T A) * A^T b  — compute for each output channel (column of atb)
    let mut result = [[0.0f64; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for k in 0..3 {
                result[row][col] += inv[row][k] * atb[k][col];
            }
        }
    }
    Some(result)
}

/// Minimal 9-term least-squares solver via normal equations (9×9 system).
///
/// Returns `None` if singular.
fn least_squares_9(features: &[[f64; 9]], targets: &[f64]) -> Option<[f64; 9]> {
    let n_terms = 9;
    let n_samples = features.len();
    if n_samples < n_terms {
        return None;
    }

    // Accumulate A^T A (9×9) and A^T b (9)
    let mut ata = [[0.0f64; 9]; 9];
    let mut atb = [0.0f64; 9];

    for (feat, &target) in features.iter().zip(targets.iter()) {
        for i in 0..9 {
            atb[i] += feat[i] * target;
            for j in 0..9 {
                ata[i][j] += feat[i] * feat[j];
            }
        }
    }

    // Gaussian elimination with partial pivoting
    gaussian_elimination_9x9(&ata, &atb)
}

/// Gaussian elimination with partial pivoting for a 9×9 system.
fn gaussian_elimination_9x9(a: &[[f64; 9]; 9], b: &[f64; 9]) -> Option<[f64; 9]> {
    const N: usize = 9;
    let mut aug = [[0.0f64; 10]; 9];
    for i in 0..N {
        for j in 0..N {
            aug[i][j] = a[i][j];
        }
        aug[i][N] = b[i];
    }

    for col in 0..N {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..N {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None;
        }
        aug.swap(col, max_row);

        // Eliminate
        let pivot = aug[col][col];
        for row in (col + 1)..N {
            let factor = aug[row][col] / pivot;
            for k in col..=N {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }

    // Back-substitution
    let mut x = [0.0f64; N];
    for i in (0..N).rev() {
        x[i] = aug[i][N];
        for j in (i + 1)..N {
            x[i] -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() < 1e-15 {
            return None;
        }
        x[i] /= aug[i][i];
    }
    Some(x)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_chart(n: usize) -> ColorChart {
        let patches: Vec<ChartPatch> = (0..n)
            .map(|i| {
                let v = i as f64 / (n.max(2) - 1) as f64;
                ChartPatch::new(format!("P{i}"), [v, v, v], [v, v, v])
            })
            .collect();
        ColorChart::from_named_patches(patches)
    }

    #[test]
    fn test_chart_patch_error_zero() {
        let p = ChartPatch::new("test", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
        let e = p.error();
        assert!(e[0].abs() < 1e-10 && e[1].abs() < 1e-10 && e[2].abs() < 1e-10);
        assert!(p.delta_e_rgb().abs() < 1e-10);
    }

    #[test]
    fn test_chart_patch_delta_e() {
        let p = ChartPatch::new("test", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!((p.delta_e_rgb() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_color_chart_from_patches_length_mismatch() {
        let measured = vec![[0.5, 0.5, 0.5]];
        let reference = vec![[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]];
        let result = ColorChart::from_patches(measured, reference);
        assert!(result.is_err());
    }

    #[test]
    fn test_color_chart_from_patches_empty() {
        let result = ColorChart::from_patches(vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_color_chart_from_patches_ok() {
        let measured = vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let reference = vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let chart = ColorChart::from_patches(measured, reference).expect("should succeed");
        assert_eq!(chart.patches.len(), 2);
    }

    #[test]
    fn test_mean_delta_e_zero_for_identity() {
        let chart = make_identity_chart(5);
        assert!(chart.mean_delta_e().abs() < 1e-10);
    }

    #[test]
    fn test_max_delta_e() {
        let patches = vec![
            ChartPatch::new("a", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ChartPatch::new("b", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ];
        let chart = ColorChart::from_named_patches(patches);
        assert!((chart.max_delta_e() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_worst_patches() {
        let patches = vec![
            ChartPatch::new("worst", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ChartPatch::new("best", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ];
        let chart = ColorChart::from_named_patches(patches);
        let worst = chart.worst_patches(1);
        assert_eq!(worst[0].name, "worst");
    }

    #[test]
    fn test_macbeth_reference_count() {
        let refs = ColorChart::macbeth_classic_reference();
        assert_eq!(refs.len(), 24);
    }

    #[test]
    fn test_macbeth_reference_range() {
        for (i, rgb) in ColorChart::macbeth_classic_reference().iter().enumerate() {
            for ch in 0..3 {
                assert!(
                    rgb[ch] >= 0.0 && rgb[ch] <= 1.0,
                    "Patch {i} channel {ch} out of range: {}",
                    rgb[ch]
                );
            }
        }
    }

    #[test]
    fn test_fit_correction_matrix_identity() {
        // If measured == reference, matrix should be close to identity
        let chart = make_identity_chart(10);
        let analyzer = ColorChartAnalyzer::new(chart);
        let m = analyzer.fit_correction_matrix();
        // Diagonal should be ≈ 1, off-diagonal ≈ 0
        assert!((m[0][0] - 1.0).abs() < 0.05, "m[0][0] = {}", m[0][0]);
        assert!((m[1][1] - 1.0).abs() < 0.05, "m[1][1] = {}", m[1][1]);
        assert!((m[2][2] - 1.0).abs() < 0.05, "m[2][2] = {}", m[2][2]);
    }

    #[test]
    fn test_generate_3d_lut_size() {
        let chart = make_identity_chart(8);
        let analyzer = ColorChartAnalyzer::new(chart);
        let lut = analyzer.generate_3d_lut(5);
        assert_eq!(lut.len(), 5 * 5 * 5);
    }

    #[test]
    fn test_generate_3d_lut_range() {
        let chart = make_identity_chart(8);
        let analyzer = ColorChartAnalyzer::new(chart);
        let lut = analyzer.generate_3d_lut(4);
        for entry in &lut {
            assert!(entry[0] >= 0.0 && entry[0] <= 1.0);
            assert!(entry[1] >= 0.0 && entry[1] <= 1.0);
            assert!(entry[2] >= 0.0 && entry[2] <= 1.0);
        }
    }

    #[test]
    fn test_residual_delta_e_near_zero_for_identity() {
        let chart = make_identity_chart(12);
        let analyzer = ColorChartAnalyzer::new(chart);
        let residual = analyzer.residual_delta_e();
        assert!(residual < 0.05, "residual = {residual}");
    }

    #[test]
    fn test_apply_matrix_correction_clamps() {
        let chart = make_identity_chart(2);
        let analyzer = ColorChartAnalyzer::new(chart);
        let m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]; // scale > 1
        let out = analyzer.apply_matrix_correction(&[0.8, 0.8, 0.8], &m);
        assert!(out[0] <= 1.0 && out[1] <= 1.0 && out[2] <= 1.0);
    }

    #[test]
    fn test_fit_polynomial_correction_shape() {
        let chart = make_identity_chart(12);
        let analyzer = ColorChartAnalyzer::new(chart);
        let coeffs = analyzer.fit_polynomial_correction();
        assert_eq!(coeffs.len(), 3);
        assert_eq!(coeffs[0].len(), 9);
    }

    #[test]
    fn test_apply_polynomial_correction_range() {
        let chart = make_identity_chart(12);
        let analyzer = ColorChartAnalyzer::new(chart);
        let coeffs = analyzer.fit_polynomial_correction();
        let out = analyzer.apply_polynomial_correction(&[0.5, 0.3, 0.7], &coeffs);
        assert!(out[0] >= 0.0 && out[0] <= 1.0);
        assert!(out[1] >= 0.0 && out[1] <= 1.0);
        assert!(out[2] >= 0.0 && out[2] <= 1.0);
    }

    #[test]
    fn test_chart_named_patch_name() {
        let p = ChartPatch::new("Dark Skin", [0.18, 0.10, 0.08], [0.20, 0.12, 0.09]);
        assert_eq!(p.name, "Dark Skin");
    }

    #[test]
    fn test_gaussian_elimination_identity() {
        let a: [[f64; 9]; 9] = {
            let mut m = [[0.0f64; 9]; 9];
            for i in 0..9 {
                m[i][i] = 1.0;
            }
            m
        };
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x = gaussian_elimination_9x9(&a, &b).expect("should solve");
        for i in 0..9 {
            assert!((x[i] - b[i]).abs() < 1e-10, "x[{i}] = {}", x[i]);
        }
    }
}
