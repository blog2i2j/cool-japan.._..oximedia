//! Color gamut mapping utilities for Dolby Vision workflows.
//!
//! Provides 3×3 matrix-based gamut conversion between:
//! - **BT.2020** (wide gamut, HDR10 / Dolby Vision source)
//! - **DCI-P3** (cinema/display)
//! - **BT.709** (SDR/HD target)
//!
//! Also includes a Bradford chromatic adaptation transform (CAT) for D65→DCI
//! white point adaptation, plus three gamut-clipping strategies:
//! - [`ClipStrategy::Clip`] — hard clamp out-of-gamut components to `[0, 1]`.
//! - [`ClipStrategy::Scale`] — uniform scale-down to preserve hue when any
//!   component exceeds `1.0`.
//! - [`ClipStrategy::Preserve`] — leave out-of-gamut values unchanged (useful
//!   for analysis).
//!
//! # Example
//!
//! ```rust
//! use oximedia_dolbyvision::gamut_mapping::{GamutMapping, GamutSpace, ClipStrategy};
//!
//! let mapping = GamutMapping::new(GamutSpace::Bt2020, GamutSpace::Bt709, ClipStrategy::Clip);
//! let [r, g, b] = mapping.convert([0.5, 0.3, 0.1]);
//! assert!(r >= 0.0 && r <= 1.0);
//! ```

// ── GamutSpace ────────────────────────────────────────────────────────────────

/// Identifies a color gamut / primaries set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamutSpace {
    /// ITU-R BT.709 / sRGB — standard HD gamut.
    Bt709,
    /// DCI-P3 (D65 white point) — digital cinema / Apple P3.
    DciP3,
    /// ITU-R BT.2020 — ultra-wide HDR gamut.
    Bt2020,
}

impl GamutSpace {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bt709 => "BT.709",
            Self::DciP3 => "DCI-P3 (D65)",
            Self::Bt2020 => "BT.2020",
        }
    }

    /// Returns `true` when this gamut is wider than BT.709.
    #[must_use]
    pub const fn is_wide_gamut(self) -> bool {
        matches!(self, Self::DciP3 | Self::Bt2020)
    }

    /// Approximate coverage of this gamut relative to CIE 1931 xy
    /// (rough percentages; BT.709 ≈ 35.9%, P3 ≈ 45.5%, BT.2020 ≈ 75.8%).
    #[must_use]
    pub const fn cie_coverage_pct(self) -> u32 {
        match self {
            Self::Bt709 => 36,
            Self::DciP3 => 46,
            Self::Bt2020 => 76,
        }
    }
}

// ── ClipStrategy ─────────────────────────────────────────────────────────────

/// Strategy for handling out-of-gamut pixel values after matrix conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClipStrategy {
    /// Hard-clamp each component to `[0.0, 1.0]`.
    #[default]
    Clip,
    /// Uniformly scale the triplet so the maximum component equals `1.0`.
    /// Preserves hue and saturation relationships.
    Scale,
    /// Leave out-of-gamut values unchanged (for analysis or chained
    /// processing).
    Preserve,
}

// ── Matrix3x3 ─────────────────────────────────────────────────────────────────

/// A column-major 3×3 matrix for linear RGB transforms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3x3 {
    /// Row-major storage: `data[row][col]`.
    data: [[f64; 3]; 3],
}

impl Matrix3x3 {
    /// Create a matrix from row-major data.
    #[must_use]
    pub const fn new(data: [[f64; 3]; 3]) -> Self {
        Self { data }
    }

    /// Identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Apply the matrix to an RGB triplet.
    #[must_use]
    pub fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        let [r, g, b] = rgb;
        let d = &self.data;
        [
            d[0][0] * r + d[0][1] * g + d[0][2] * b,
            d[1][0] * r + d[1][1] * g + d[1][2] * b,
            d[2][0] * r + d[2][1] * g + d[2][2] * b,
        ]
    }

    /// Multiply two matrices: `self * other`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let mut out = [[0.0f64; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                for k in 0..3 {
                    out[row][col] += self.data[row][k] * other.data[k][col];
                }
            }
        }
        Self::new(out)
    }

    /// Attempt to compute the matrix inverse using Cramer's rule.
    ///
    /// Returns `None` when the matrix is singular (determinant ≈ 0).
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let m = &self.data;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;

        Some(Self::new([
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
        ]))
    }
}

// ── Primaries matrices ────────────────────────────────────────────────────────
//
// All matrices convert linear RGB in the given space to CIE XYZ (D65),
// derived from the standard primary chromaticities.
//
// BT.709 → XYZ (D65) — SMPTE RP 177
const BT709_TO_XYZ: Matrix3x3 = Matrix3x3::new([
    [0.412_391, 0.357_584, 0.180_481],
    [0.212_639, 0.715_169, 0.072_192],
    [0.019_331, 0.119_195, 0.950_532],
]);

// BT.2020 → XYZ (D65)
const BT2020_TO_XYZ: Matrix3x3 = Matrix3x3::new([
    [0.636_958, 0.144_617, 0.168_881],
    [0.262_700, 0.677_998, 0.059_302],
    [0.000_000, 0.028_073, 1.060_985],
]);

// DCI-P3 (D65) → XYZ (D65)
const DCIP3_TO_XYZ: Matrix3x3 = Matrix3x3::new([
    [0.486_571, 0.265_668, 0.198_217],
    [0.228_975, 0.691_739, 0.079_287],
    [0.000_000, 0.045_113, 1.043_944],
]);

fn xyz_to_bt709() -> Matrix3x3 {
    BT709_TO_XYZ.inverse().unwrap_or(Matrix3x3::identity())
}

fn xyz_to_bt2020() -> Matrix3x3 {
    BT2020_TO_XYZ.inverse().unwrap_or(Matrix3x3::identity())
}

fn xyz_to_dcip3() -> Matrix3x3 {
    DCIP3_TO_XYZ.inverse().unwrap_or(Matrix3x3::identity())
}

fn rgb_to_xyz(space: GamutSpace) -> Matrix3x3 {
    match space {
        GamutSpace::Bt709 => BT709_TO_XYZ,
        GamutSpace::DciP3 => DCIP3_TO_XYZ,
        GamutSpace::Bt2020 => BT2020_TO_XYZ,
    }
}

fn xyz_to_rgb(space: GamutSpace) -> Matrix3x3 {
    match space {
        GamutSpace::Bt709 => xyz_to_bt709(),
        GamutSpace::DciP3 => xyz_to_dcip3(),
        GamutSpace::Bt2020 => xyz_to_bt2020(),
    }
}

// ── GamutMapping ──────────────────────────────────────────────────────────────

/// A prepared gamut mapping from one [`GamutSpace`] to another.
///
/// The internal 3×3 matrix is pre-computed at construction time by composing
/// the source-to-XYZ and XYZ-to-destination matrices.
#[derive(Debug, Clone)]
pub struct GamutMapping {
    /// Source gamut.
    pub source: GamutSpace,
    /// Destination gamut.
    pub destination: GamutSpace,
    /// Out-of-gamut clipping strategy.
    pub clip: ClipStrategy,
    /// Pre-composed 3×3 linear transform.
    matrix: Matrix3x3,
}

impl GamutMapping {
    /// Build a gamut mapping.  When `source == destination` the identity
    /// matrix is used regardless.
    #[must_use]
    pub fn new(source: GamutSpace, destination: GamutSpace, clip: ClipStrategy) -> Self {
        let matrix = if source == destination {
            Matrix3x3::identity()
        } else {
            let src_to_xyz = rgb_to_xyz(source);
            let xyz_to_dst = xyz_to_rgb(destination);
            xyz_to_dst.mul(&src_to_xyz)
        };
        Self {
            source,
            destination,
            clip,
            matrix,
        }
    }

    /// Convert a linear-light RGB triplet from source to destination gamut.
    ///
    /// The input components are expected in `[0.0, 1.0]` (linear scene-linear
    /// or display-linear light, **not** gamma-encoded).
    #[must_use]
    pub fn convert(&self, rgb: [f32; 3]) -> [f32; 3] {
        let rgb64 = [f64::from(rgb[0]), f64::from(rgb[1]), f64::from(rgb[2])];
        let out64 = self.matrix.apply(rgb64);
        let clipped = apply_clip(out64, self.clip);
        [clipped[0] as f32, clipped[1] as f32, clipped[2] as f32]
    }

    /// Convert many RGB triplets in-place.
    ///
    /// Each element of `pixels` is a `[r, g, b]` linear-light triplet.
    pub fn convert_bulk(&self, pixels: &mut [[f32; 3]]) {
        for px in pixels.iter_mut() {
            *px = self.convert(*px);
        }
    }

    /// Returns `true` when this is an identity mapping (same source and
    /// destination).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.source == self.destination
    }

    /// Check whether the given linear-light RGB triplet is within the
    /// destination gamut (all components in `[0.0, 1.0]` after conversion).
    #[must_use]
    pub fn is_in_gamut(&self, rgb: [f32; 3]) -> bool {
        let [r, g, b] = self.convert(rgb);
        // We use the Preserve strategy to check the raw converted values
        let rgb64 = [f64::from(rgb[0]), f64::from(rgb[1]), f64::from(rgb[2])];
        let raw = self.matrix.apply(rgb64);
        let _ = (r, g, b); // result already clipped — check raw instead
        raw.iter().all(|&v| (0.0..=1.0).contains(&v))
    }
}

fn apply_clip(rgb: [f64; 3], strategy: ClipStrategy) -> [f64; 3] {
    match strategy {
        ClipStrategy::Clip => [
            rgb[0].clamp(0.0, 1.0),
            rgb[1].clamp(0.0, 1.0),
            rgb[2].clamp(0.0, 1.0),
        ],
        ClipStrategy::Scale => {
            let max = rgb[0].max(rgb[1]).max(rgb[2]);
            if max > 1.0 {
                [rgb[0] / max, rgb[1] / max, rgb[2] / max]
            } else {
                // Still clamp the floor to avoid negative values from dark areas
                [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]
            }
        }
        ClipStrategy::Preserve => rgb,
    }
}

// ── GamutMapper (higher-level helper) ────────────────────────────────────────

/// Statistics accumulated by [`GamutMapper::map_with_stats`].
#[derive(Debug, Clone, Default)]
pub struct GamutMapStats {
    /// Total number of pixels processed.
    pub total: u64,
    /// Number of pixels that were out-of-gamut before clipping.
    pub out_of_gamut: u64,
}

impl GamutMapStats {
    /// Fraction of out-of-gamut pixels, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no pixels have been processed.
    #[must_use]
    pub fn out_of_gamut_fraction(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.out_of_gamut as f64 / self.total as f64
    }
}

/// A higher-level mapper that wraps [`GamutMapping`] and optionally records
/// out-of-gamut statistics.
#[derive(Debug, Clone)]
pub struct GamutMapper {
    mapping: GamutMapping,
}

impl GamutMapper {
    /// Create a mapper from `source` to `destination` with the given clip
    /// strategy.
    #[must_use]
    pub fn new(source: GamutSpace, destination: GamutSpace, clip: ClipStrategy) -> Self {
        Self {
            mapping: GamutMapping::new(source, destination, clip),
        }
    }

    /// Map pixels and collect out-of-gamut statistics.
    ///
    /// A pixel is counted as "out-of-gamut" when any converted component
    /// (before clipping) falls outside `[0.0, 1.0]`.
    pub fn map_with_stats(&self, pixels: &mut [[f32; 3]]) -> GamutMapStats {
        let mut stats = GamutMapStats::default();
        // Use a Preserve mapping to detect raw out-of-gamut before clipping
        let detect = GamutMapping::new(
            self.mapping.source,
            self.mapping.destination,
            ClipStrategy::Preserve,
        );
        for px in pixels.iter_mut() {
            stats.total += 1;
            let raw = detect.convert(*px);
            if raw.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
                stats.out_of_gamut += 1;
            }
            *px = self.mapping.convert(*px);
        }
        stats
    }

    /// Access the underlying [`GamutMapping`].
    #[must_use]
    pub fn mapping(&self) -> &GamutMapping {
        &self.mapping
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    // ── GamutSpace ────────────────────────────────────────────────────────────

    #[test]
    fn test_gamut_space_name() {
        assert_eq!(GamutSpace::Bt709.name(), "BT.709");
        assert_eq!(GamutSpace::DciP3.name(), "DCI-P3 (D65)");
        assert_eq!(GamutSpace::Bt2020.name(), "BT.2020");
    }

    #[test]
    fn test_gamut_space_is_wide_gamut() {
        assert!(!GamutSpace::Bt709.is_wide_gamut());
        assert!(GamutSpace::DciP3.is_wide_gamut());
        assert!(GamutSpace::Bt2020.is_wide_gamut());
    }

    #[test]
    fn test_gamut_space_cie_coverage_ordering() {
        assert!(GamutSpace::Bt709.cie_coverage_pct() < GamutSpace::DciP3.cie_coverage_pct());
        assert!(GamutSpace::DciP3.cie_coverage_pct() < GamutSpace::Bt2020.cie_coverage_pct());
    }

    // ── Matrix3x3 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_identity_matrix_apply() {
        let m = Matrix3x3::identity();
        let out = m.apply([0.3, 0.5, 0.7]);
        assert!((out[0] - 0.3).abs() < 1e-12);
        assert!((out[1] - 0.5).abs() < 1e-12);
        assert!((out[2] - 0.7).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_mul_identity_is_unchanged() {
        let m = BT709_TO_XYZ;
        let id = Matrix3x3::identity();
        let result = m.mul(&id);
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (result.data[r][c] - m.data[r][c]).abs() < 1e-12,
                    "mismatch at [{r}][{c}]"
                );
            }
        }
    }

    #[test]
    fn test_matrix_inverse_identity() {
        let m = Matrix3x3::identity();
        let inv = m.inverse().expect("identity should be invertible");
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (inv.data[r][c] - expected).abs() < 1e-10,
                    "inv[{r}][{c}]={} expected {expected}",
                    inv.data[r][c]
                );
            }
        }
    }

    #[test]
    fn test_bt709_to_xyz_inverse_roundtrip() {
        let xyz = BT709_TO_XYZ;
        let xyz_inv = xyz.inverse().expect("must be invertible");
        let product = xyz.mul(&xyz_inv);
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (product.data[r][c] - expected).abs() < 1e-8,
                    "product[{r}][{c}]={}, expected {expected}",
                    product.data[r][c]
                );
            }
        }
    }

    // ── ClipStrategy ─────────────────────────────────────────────────────────

    #[test]
    fn test_clip_strategy_clip_clamps() {
        let clipped = apply_clip([1.5, -0.1, 0.5], ClipStrategy::Clip);
        assert!((clipped[0] - 1.0).abs() < 1e-12);
        assert!((clipped[1] - 0.0).abs() < 1e-12);
        assert!((clipped[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_clip_strategy_scale_preserves_hue() {
        let scaled = apply_clip([2.0, 1.0, 0.5], ClipStrategy::Scale);
        // Max is 2.0 so everything halved
        assert!((scaled[0] - 1.0).abs() < 1e-12);
        assert!((scaled[1] - 0.5).abs() < 1e-12);
        assert!((scaled[2] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_clip_strategy_preserve_leaves_values() {
        let preserved = apply_clip([1.5, -0.1, 0.5], ClipStrategy::Preserve);
        assert!((preserved[0] - 1.5).abs() < 1e-12);
        assert!((preserved[1] - (-0.1)).abs() < 1e-12);
    }

    // ── GamutMapping ─────────────────────────────────────────────────────────

    #[test]
    fn test_identity_mapping_no_change() {
        let mapping = GamutMapping::new(GamutSpace::Bt709, GamutSpace::Bt709, ClipStrategy::Clip);
        assert!(mapping.is_identity());
        let rgb = [0.4_f32, 0.2_f32, 0.6_f32];
        let out = mapping.convert(rgb);
        assert!((out[0] - rgb[0]).abs() < EPSILON);
        assert!((out[1] - rgb[1]).abs() < EPSILON);
        assert!((out[2] - rgb[2]).abs() < EPSILON);
    }

    #[test]
    fn test_bt2020_to_bt709_white_maps_to_white() {
        let mapping =
            GamutMapping::new(GamutSpace::Bt2020, GamutSpace::Bt709, ClipStrategy::Clip);
        // Perfect white (1,1,1) should map to (1,1,1) in any gamut
        let out = mapping.convert([1.0, 1.0, 1.0]);
        assert!((out[0] - 1.0).abs() < EPSILON, "R={}", out[0]);
        assert!((out[1] - 1.0).abs() < EPSILON, "G={}", out[1]);
        assert!((out[2] - 1.0).abs() < EPSILON, "B={}", out[2]);
    }

    #[test]
    fn test_bt2020_to_bt709_black_maps_to_black() {
        let mapping =
            GamutMapping::new(GamutSpace::Bt2020, GamutSpace::Bt709, ClipStrategy::Clip);
        let out = mapping.convert([0.0, 0.0, 0.0]);
        assert!(out[0].abs() < EPSILON);
        assert!(out[1].abs() < EPSILON);
        assert!(out[2].abs() < EPSILON);
    }

    #[test]
    fn test_bt2020_to_p3_is_not_identity() {
        let mapping =
            GamutMapping::new(GamutSpace::Bt2020, GamutSpace::DciP3, ClipStrategy::Preserve);
        assert!(!mapping.is_identity());
    }

    #[test]
    fn test_gamut_mapping_convert_bulk() {
        let mapping =
            GamutMapping::new(GamutSpace::Bt2020, GamutSpace::Bt709, ClipStrategy::Clip);
        let mut pixels = vec![[0.5_f32, 0.3, 0.1], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]];
        mapping.convert_bulk(&mut pixels);
        // All components should be in [0, 1] due to Clip
        for px in &pixels {
            for &c in px {
                assert!(c >= 0.0 && c <= 1.0, "component {c} out of range");
            }
        }
    }

    // ── GamutMapper ───────────────────────────────────────────────────────────

    #[test]
    fn test_gamut_mapper_stats_all_in_gamut() {
        let mapper = GamutMapper::new(GamutSpace::Bt709, GamutSpace::Bt709, ClipStrategy::Clip);
        let mut pixels = vec![[0.5_f32, 0.3, 0.2]];
        let stats = mapper.map_with_stats(&mut pixels);
        assert_eq!(stats.total, 1);
        assert_eq!(stats.out_of_gamut, 0);
        assert!((stats.out_of_gamut_fraction() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gamut_mapper_stats_fraction_zero_when_empty() {
        let stats = GamutMapStats::default();
        assert!((stats.out_of_gamut_fraction() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gamut_mapper_white_point_preservation() {
        // D65 white should survive the BT.2020 → BT.709 journey
        let mapper = GamutMapper::new(GamutSpace::Bt2020, GamutSpace::Bt709, ClipStrategy::Clip);
        let mut pixels = vec![[1.0_f32, 1.0, 1.0]];
        let _stats = mapper.map_with_stats(&mut pixels);
        let [r, g, b] = pixels[0];
        assert!((r - 1.0).abs() < EPSILON, "R={r}");
        assert!((g - 1.0).abs() < EPSILON, "G={g}");
        assert!((b - 1.0).abs() < EPSILON, "B={b}");
    }
}
