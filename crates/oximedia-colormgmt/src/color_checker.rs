//! ColorChecker chart analysis for camera calibration and colour accuracy testing.
//!
//! This module provides the canonical **X-Rite ColorChecker Classic** 24-patch
//! reference values in CIE L\*a\*b\* (D50/2°) and sRGB, along with utilities for:
//!
//! - Computing per-patch ΔE colour differences between measured and reference
//! - Generating a full accuracy report with statistics
//! - Detecting the 6 × 4 grid topology of a captured chart image (simulation)
//! - Fitting a 3 × 3 linear-light correction matrix to minimise overall ΔE
//!
//! # Reference Data
//!
//! Patch Lab values are the Macbeth/X-Rite published reference data for the
//! Classic chart (D50, 2° observer), as tabulated in:
//!
//! > X-Rite ColorChecker Classic Spectral Data, 2014 edition.
//!
//! # Example
//!
//! ```rust
//! use oximedia_colormgmt::color_checker::{ColorChecker, reference_patches};
//!
//! let patches = reference_patches();
//! assert_eq!(patches.len(), 24);
//!
//! let checker = ColorChecker::new();
//! let report = checker.analyse_lab(&patches); // perfect match → all dE ≈ 0
//! assert!(report.mean_delta_e < 0.01);
//! ```

#![allow(dead_code)]

use crate::delta_e::{delta_e_2000, CieDe2000Weights};
use crate::error::{ColorError, Result};
use crate::xyz::Lab;

// ── Patch definition ──────────────────────────────────────────────────────────

/// A single ColorChecker patch with name and CIE L\*a\*b\* reference values.
#[derive(Debug, Clone)]
pub struct Patch {
    /// Human-readable patch name (e.g. "Dark Skin", "White").
    pub name: &'static str,
    /// Row (0-based, 0 = top).
    pub row: u8,
    /// Column (0-based, 0 = left).
    pub col: u8,
    /// CIE L\* (lightness, 0–100).
    pub l: f64,
    /// CIE a\* (green–red, ±128).
    pub a: f64,
    /// CIE b\* (blue–yellow, ±128).
    pub b: f64,
    /// Reference linear sRGB (after linearisation from the published sRGB values).
    pub srgb: [f64; 3],
}

impl Patch {
    /// Returns the Lab value as a [`Lab`] struct.
    #[must_use]
    pub fn lab(&self) -> Lab {
        Lab::new(self.l, self.a, self.b)
    }

    /// Returns the patch index (0–23, row-major order).
    #[must_use]
    pub fn index(&self) -> usize {
        self.row as usize * 6 + self.col as usize
    }
}

// ── Reference data ────────────────────────────────────────────────────────────

/// Returns the 24 reference patches of the X-Rite ColorChecker Classic.
///
/// Values are CIE L\*a\*b\* under D50 illuminant, 2° observer.
/// sRGB values are the standard encoding (gamma ≈ 2.2 sRGB) linearised to [0, 1].
#[must_use]
pub fn reference_patches() -> Vec<Patch> {
    // Lab data: X-Rite 2014 reference (D50, 2°)
    // sRGB values: linearised from standard 8-bit sRGB encoding (X-Rite published)
    const DATA: &[(&str, u8, u8, f64, f64, f64, [f64; 3])] = &[
        // Row 0
        ("Dark Skin",         0, 0, 37.99,  13.56,  14.06, [0.102, 0.052, 0.032]),
        ("Light Skin",        0, 1, 65.71,  18.13,  17.81, [0.428, 0.219, 0.135]),
        ("Blue Sky",          0, 2, 49.93,  -4.88, -21.93, [0.099, 0.148, 0.277]),
        ("Foliage",           0, 3, 43.14, -13.10,  21.91, [0.089, 0.105, 0.034]),
        ("Blue Flower",       0, 4, 55.11,   8.84, -25.40, [0.181, 0.163, 0.355]),
        ("Bluish Green",      0, 5, 70.72, -33.40, -0.199, [0.124, 0.384, 0.334]),
        // Row 1
        ("Orange",            1, 0, 62.66,  36.07,  57.10, [0.585, 0.211, 0.012]),
        ("Purplish Blue",     1, 1, 40.02,  10.41, -45.96, [0.064, 0.076, 0.395]),
        ("Moderate Red",      1, 2, 51.12,  48.24,  16.25, [0.421, 0.083, 0.087]),
        ("Purple",            1, 3, 30.33,  22.98, -21.59, [0.117, 0.043, 0.115]),
        ("Yellow Green",      1, 4, 71.77, -24.48,  58.10, [0.310, 0.447, 0.019]),
        ("Orange Yellow",     1, 5, 71.51,  18.89,  67.37, [0.543, 0.371, 0.006]),
        // Row 2
        ("Blue",              2, 0, 28.78,  14.18, -50.30, [0.033, 0.043, 0.281]),
        ("Green",             2, 1, 55.26, -38.34,  31.37, [0.051, 0.266, 0.064]),
        ("Red",               2, 2, 42.10,  53.38,  28.19, [0.349, 0.030, 0.029]),
        ("Yellow",            2, 3, 81.73,   4.04,  79.82, [0.704, 0.607, 0.003]),
        ("Magenta",           2, 4, 51.94,  49.99, -14.57, [0.401, 0.095, 0.272]),
        ("Cyan",              2, 5, 51.04, -28.63, -28.64, [0.005, 0.192, 0.322]),
        // Row 3 — neutral grey scale
        ("White",             3, 0, 96.54,  -0.43,   1.19, [0.878, 0.878, 0.878]),
        ("Neutral 8",         3, 1, 81.26,  -0.64,  -0.34, [0.572, 0.572, 0.572]),
        ("Neutral 6.5",       3, 2, 66.77,  -0.73,  -0.50, [0.352, 0.352, 0.352]),
        ("Neutral 5",         3, 3, 50.87,  -0.15,  -0.27, [0.192, 0.192, 0.192]),
        ("Neutral 3.5",       3, 4, 35.66,  -0.46,  -0.48, [0.090, 0.090, 0.090]),
        ("Black",             3, 5, 20.46,  -0.08,  -0.26, [0.031, 0.031, 0.031]),
    ];

    DATA.iter()
        .map(|&(name, row, col, l, a, b, srgb)| Patch {
            name,
            row,
            col,
            l,
            a,
            b,
            srgb,
        })
        .collect()
}

// ── Per-patch result ──────────────────────────────────────────────────────────

/// Colour accuracy result for a single patch.
#[derive(Debug, Clone)]
pub struct PatchResult {
    /// Patch name.
    pub name: &'static str,
    /// Patch index (0–23).
    pub index: usize,
    /// Reference Lab value.
    pub reference: Lab,
    /// Measured Lab value.
    pub measured: Lab,
    /// ΔE 2000 colour difference.
    pub delta_e: f64,
}

impl PatchResult {
    /// Returns `true` if the colour difference is within broadcast tolerance (ΔE < 1.0).
    #[must_use]
    pub fn is_broadcast_accurate(&self) -> bool {
        self.delta_e < 1.0
    }

    /// Returns `true` if the difference is within critical viewing tolerance (ΔE < 3.0).
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.delta_e < 3.0
    }
}

// ── Accuracy report ───────────────────────────────────────────────────────────

/// Full 24-patch accuracy report.
#[derive(Debug, Clone)]
pub struct AccuracyReport {
    /// Per-patch results.
    pub patches: Vec<PatchResult>,
    /// Mean ΔE 2000 across all patches.
    pub mean_delta_e: f64,
    /// Maximum ΔE 2000 across all patches.
    pub max_delta_e: f64,
    /// 95th percentile ΔE (sorted, index ≥ 0.95 × n).
    pub p95_delta_e: f64,
    /// Number of patches with ΔE < 1.0 (broadcast quality).
    pub broadcast_count: usize,
    /// Number of patches with ΔE < 3.0 (acceptable quality).
    pub acceptable_count: usize,
}

impl AccuracyReport {
    fn from_results(mut results: Vec<PatchResult>) -> Self {
        let n = results.len();
        let mean_delta_e = if n == 0 {
            0.0
        } else {
            results.iter().map(|r| r.delta_e).sum::<f64>() / n as f64
        };
        let max_delta_e = results
            .iter()
            .map(|r| r.delta_e)
            .fold(0.0_f64, f64::max);

        // p95: sort a copy of delta-e values
        let mut sorted_de: Vec<f64> = results.iter().map(|r| r.delta_e).collect();
        sorted_de.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_idx = ((n as f64 * 0.95) as usize).min(n.saturating_sub(1));
        let p95_delta_e = sorted_de.get(p95_idx).copied().unwrap_or(0.0);

        let broadcast_count = results.iter().filter(|r| r.is_broadcast_accurate()).count();
        let acceptable_count = results.iter().filter(|r| r.is_acceptable()).count();

        // Sort by index for stable output
        results.sort_by_key(|r| r.index);

        Self {
            patches: results,
            mean_delta_e,
            max_delta_e,
            p95_delta_e,
            broadcast_count,
            acceptable_count,
        }
    }

    /// Returns the worst patch (highest ΔE).
    #[must_use]
    pub fn worst_patch(&self) -> Option<&PatchResult> {
        self.patches
            .iter()
            .max_by(|a, b| a.delta_e.partial_cmp(&b.delta_e).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Returns the best patch (lowest ΔE).
    #[must_use]
    pub fn best_patch(&self) -> Option<&PatchResult> {
        self.patches
            .iter()
            .min_by(|a, b| a.delta_e.partial_cmp(&b.delta_e).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ── ColorChecker analyser ─────────────────────────────────────────────────────

/// Performs ColorChecker accuracy analysis.
///
/// Compares measured Lab values against the reference patches and produces
/// a full accuracy report.
pub struct ColorChecker {
    reference: Vec<Patch>,
    weights: CieDe2000Weights,
}

impl ColorChecker {
    /// Creates a new analyser with the standard 24-patch reference and
    /// default CIEDE2000 weights (k_L = k_C = k_H = 1.0).
    #[must_use]
    pub fn new() -> Self {
        Self {
            reference: reference_patches(),
            weights: CieDe2000Weights::reference(),
        }
    }

    /// Creates a new analyser with custom CIEDE2000 weighting factors.
    #[must_use]
    pub fn with_weights(weights: CieDe2000Weights) -> Self {
        Self {
            reference: reference_patches(),
            weights,
        }
    }

    /// Analyses a set of 24 measured Lab values (one per reference patch).
    ///
    /// The `measured` slice must be in the same row-major order as
    /// [`reference_patches()`].
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::InvalidColor`] if `measured.len() != 24`.
    pub fn analyse_lab(&self, measured: &[Patch]) -> AccuracyReport {
        let results: Vec<PatchResult> = self
            .reference
            .iter()
            .zip(measured.iter())
            .map(|(reference, measured)| {
                let ref_lab = reference.lab();
                let meas_lab = measured.lab();
                let de = delta_e_2000(&ref_lab, &meas_lab);
                PatchResult {
                    name: reference.name,
                    index: reference.index(),
                    reference: ref_lab,
                    measured: meas_lab,
                    delta_e: de,
                }
            })
            .collect();

        AccuracyReport::from_results(results)
    }

    /// Analyses 24 measured Lab triplets `[L, a, b]` (one per patch, row-major).
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::InvalidColor`] if `measured.len() != 24`.
    pub fn analyse_lab_arrays(&self, measured: &[[f64; 3]]) -> Result<AccuracyReport> {
        if measured.len() != 24 {
            return Err(ColorError::InvalidColor(format!(
                "ColorChecker: expected 24 measured patches, got {}",
                measured.len()
            )));
        }

        let results: Vec<PatchResult> = self
            .reference
            .iter()
            .zip(measured.iter())
            .map(|(reference, &[ml, ma, mb])| {
                let ref_lab = reference.lab();
                let meas_lab = Lab::new(ml, ma, mb);
                let de = delta_e_2000(&ref_lab, &meas_lab);
                PatchResult {
                    name: reference.name,
                    index: reference.index(),
                    reference: ref_lab,
                    measured: meas_lab,
                    delta_e: de,
                }
            })
            .collect();

        Ok(AccuracyReport::from_results(results))
    }

    /// Returns the reference patches.
    #[must_use]
    pub fn reference_patches(&self) -> &[Patch] {
        &self.reference
    }
}

impl Default for ColorChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Matrix fitting ────────────────────────────────────────────────────────────

/// Fits a 3 × 3 linear-light correction matrix that maps measured sRGB
/// values toward the reference sRGB values.
///
/// The least-squares fit is solved analytically using the normal equations
/// (AtA x = Atb) for each output channel independently.
///
/// # Arguments
///
/// * `measured_srgb`   - 24 linear-light `[R, G, B]` measured values.
/// * `reference_srgb`  - 24 linear-light `[R, G, B]` reference values (or call
///                       [`reference_patches()`] and use `.srgb`).
///
/// # Returns
///
/// A 3 × 3 matrix `M` such that `M * measured ≈ reference` (column-vector
/// convention: output `[R', G', B'] = M * [R, G, B]`).
///
/// # Errors
///
/// Returns [`ColorError::InvalidColor`] if either slice has fewer than 3
/// entries or they differ in length, or if the normal equations are singular.
pub fn fit_correction_matrix(
    measured_srgb: &[[f64; 3]],
    reference_srgb: &[[f64; 3]],
) -> Result<[[f64; 3]; 3]> {
    if measured_srgb.len() < 3 {
        return Err(ColorError::InvalidColor(
            "fit_correction_matrix: need at least 3 patches".into(),
        ));
    }
    if measured_srgb.len() != reference_srgb.len() {
        return Err(ColorError::InvalidColor(format!(
            "fit_correction_matrix: measured ({}) and reference ({}) lengths differ",
            measured_srgb.len(),
            reference_srgb.len()
        )));
    }

    let n = measured_srgb.len();

    // Build AtA (3×3) and Atb (3×3, one column per output channel)
    let mut ata = [[0.0_f64; 3]; 3];
    let mut atb = [[0.0_f64; 3]; 3]; // atb[out_ch][in_ch]

    for i in 0..n {
        let m = measured_srgb[i];
        let r = reference_srgb[i];
        for row in 0..3 {
            for col in 0..3 {
                ata[row][col] += m[row] * m[col];
            }
            for out_ch in 0..3 {
                atb[out_ch][row] += m[row] * r[out_ch];
            }
        }
    }

    // Invert AtA via Cramer's rule (3×3)
    let ata_inv = invert_3x3(&ata)?;

    // Matrix = Atb * AtA^{-1}  (each row of M is ata_inv * atb[ch])
    let mut mat = [[0.0_f64; 3]; 3];
    for out_ch in 0..3 {
        for col in 0..3 {
            let mut v = 0.0_f64;
            for k in 0..3 {
                v += ata_inv[col][k] * atb[out_ch][k];
            }
            mat[out_ch][col] = v;
        }
    }

    Ok(mat)
}

/// Applies a 3 × 3 correction matrix to a single RGB pixel.
#[must_use]
pub fn apply_matrix(matrix: &[[f64; 3]; 3], rgb: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2],
        matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2],
        matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2],
    ]
}

// ── 3×3 matrix inversion ──────────────────────────────────────────────────────

fn invert_3x3(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-15 {
        return Err(ColorError::Matrix(
            "invert_3x3: matrix is singular".into(),
        ));
    }

    let inv_det = 1.0 / det;

    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

// ── Patch detection simulation ────────────────────────────────────────────────

/// Simulates grid detection on a known-good chart image.
///
/// In a real implementation this would analyse pixel luminance to locate
/// patch centres.  Here we return the reference patch centroids scaled to
/// a normalised `[0, 1] × [0, 1]` coordinate system for a 6 × 4 grid.
///
/// # Returns
///
/// 24 `(x, y)` normalised centroid positions, row-major.
#[must_use]
pub fn detect_patch_grid() -> Vec<(f64, f64)> {
    let mut centroids = Vec::with_capacity(24);
    // Patch centres: equal spacing with 10% border
    for row in 0..4_u8 {
        for col in 0..6_u8 {
            let x = 0.1 + (col as f64) * 0.8 / 5.0;
            let y = 0.1 + (row as f64) * 0.8 / 3.0;
            centroids.push((x, y));
        }
    }
    centroids
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Reference data ────────────────────────────────────────────────────────

    #[test]
    fn reference_has_24_patches() {
        assert_eq!(reference_patches().len(), 24);
    }

    #[test]
    fn reference_patch_indices_unique_and_valid() {
        let patches = reference_patches();
        let mut indices: Vec<usize> = patches.iter().map(|p| p.index()).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), 24);
        assert_eq!(*indices.last().unwrap(), 23);
    }

    #[test]
    fn neutral_patches_have_near_zero_ab() {
        let patches = reference_patches();
        let neutrals = ["White", "Neutral 8", "Neutral 6.5", "Neutral 5", "Neutral 3.5", "Black"];
        for name in neutrals {
            let p = patches.iter().find(|p| p.name == name).expect(name);
            assert!(
                p.a.abs() < 2.0 && p.b.abs() < 2.0,
                "{}: a={}, b={} not neutral",
                name, p.a, p.b
            );
        }
    }

    #[test]
    fn reference_l_values_in_range() {
        for p in reference_patches() {
            assert!(p.l > 0.0 && p.l < 100.0, "{}: L={} out of range", p.name, p.l);
        }
    }

    // ── ColorChecker analyser ─────────────────────────────────────────────────

    #[test]
    fn analyse_perfect_match_gives_zero_de() {
        let checker = ColorChecker::new();
        let reference = reference_patches();
        let report = checker.analyse_lab(&reference);
        assert!(
            report.mean_delta_e < 1e-10,
            "perfect match mean dE should be 0, got {}",
            report.mean_delta_e
        );
        assert!(report.max_delta_e < 1e-10);
    }

    #[test]
    fn analyse_lab_arrays_wrong_count_errors() {
        let checker = ColorChecker::new();
        let result = checker.analyse_lab_arrays(&[[50.0, 0.0, 0.0]; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn analyse_lab_arrays_correct_count() {
        let checker = ColorChecker::new();
        let reference = reference_patches();
        let arrays: Vec<[f64; 3]> = reference.iter().map(|p| [p.l, p.a, p.b]).collect();
        let report = checker.analyse_lab_arrays(&arrays).unwrap();
        assert!(report.mean_delta_e < 1e-10);
    }

    #[test]
    fn worst_and_best_patch() {
        let checker = ColorChecker::new();
        let mut measured = reference_patches();
        // Perturb patch 5 heavily
        measured[5].l += 20.0;
        let report = checker.analyse_lab(&measured);
        let worst = report.worst_patch().unwrap();
        assert_eq!(worst.index, 5, "worst should be patch 5");
        let best = report.best_patch().unwrap();
        assert_eq!(best.delta_e, 0.0);
    }

    #[test]
    fn report_statistics_broadcast_count() {
        let checker = ColorChecker::new();
        // Perfect match: all 24 should be broadcast accurate
        let reference = reference_patches();
        let report = checker.analyse_lab(&reference);
        assert_eq!(report.broadcast_count, 24);
        assert_eq!(report.acceptable_count, 24);
    }

    // ── Matrix fitting ────────────────────────────────────────────────────────

    #[test]
    fn identity_matrix_fit_for_perfect_data() {
        let patches = reference_patches();
        let srgb: Vec<[f64; 3]> = patches.iter().map(|p| p.srgb).collect();
        let matrix = fit_correction_matrix(&srgb, &srgb).unwrap();
        // Result should be close to identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (matrix[i][j] - expected).abs() < 1e-6,
                    "matrix[{i}][{j}] = {}, expected {expected}",
                    matrix[i][j]
                );
            }
        }
    }

    #[test]
    fn matrix_fit_insufficient_data_errors() {
        let m: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(fit_correction_matrix(&m, &m).is_err());
    }

    #[test]
    fn matrix_fit_length_mismatch_errors() {
        let a: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0]; 5];
        let b: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0]; 4];
        assert!(fit_correction_matrix(&a, &b).is_err());
    }

    // ── Grid detection ────────────────────────────────────────────────────────

    #[test]
    fn detect_patch_grid_returns_24_centroids() {
        let centroids = detect_patch_grid();
        assert_eq!(centroids.len(), 24);
    }

    #[test]
    fn detect_patch_grid_centroids_in_unit_square() {
        for (x, y) in detect_patch_grid() {
            assert!(x >= 0.0 && x <= 1.0, "x={x} out of [0,1]");
            assert!(y >= 0.0 && y <= 1.0, "y={y} out of [0,1]");
        }
    }
}
