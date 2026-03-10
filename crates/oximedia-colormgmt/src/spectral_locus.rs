#![allow(dead_code)]
//! CIE spectral locus utilities.
//!
//! Provides the CIE 1931 2-degree observer spectral locus data and helpers for
//! dominant-wavelength calculations, spectral purity, and Planckian (blackbody)
//! locus approximation used in white-point computations.

use std::fmt;

// ─── Spectral Locus Point ───────────────────────────────────────────────────

/// A single point on the spectral locus in CIE 1931 xy chromaticity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralPoint {
    /// Wavelength in nanometres.
    pub wavelength_nm: f64,
    /// CIE x chromaticity coordinate.
    pub x: f64,
    /// CIE y chromaticity coordinate.
    pub y: f64,
}

impl SpectralPoint {
    /// Create a new spectral locus point.
    #[must_use]
    pub const fn new(wavelength_nm: f64, x: f64, y: f64) -> Self {
        Self {
            wavelength_nm,
            x,
            y,
        }
    }
}

impl fmt::Display for SpectralPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.0} nm (x={:.4}, y={:.4})",
            self.wavelength_nm, self.x, self.y
        )
    }
}

// ─── Sampled spectral locus (CIE 1931, 10-nm steps, 380–700 nm) ────────────

/// CIE 1931 2-degree observer spectral locus sampled at 10-nm intervals.
///
/// Each entry is `(wavelength_nm, x, y)`.
pub const SPECTRAL_LOCUS_10NM: [(f64, f64, f64); 33] = [
    (380.0, 0.1741, 0.0050),
    (390.0, 0.1740, 0.0050),
    (400.0, 0.1733, 0.0048),
    (410.0, 0.1726, 0.0048),
    (420.0, 0.1714, 0.0051),
    (430.0, 0.1689, 0.0069),
    (440.0, 0.1644, 0.0109),
    (450.0, 0.1566, 0.0177),
    (460.0, 0.1440, 0.0297),
    (470.0, 0.1241, 0.0578),
    (480.0, 0.0913, 0.1327),
    (490.0, 0.0454, 0.2950),
    (500.0, 0.0082, 0.5384),
    (510.0, 0.0139, 0.7502),
    (520.0, 0.0743, 0.8338),
    (530.0, 0.1547, 0.8059),
    (540.0, 0.2296, 0.7543),
    (550.0, 0.3016, 0.6923),
    (560.0, 0.3731, 0.6245),
    (570.0, 0.4441, 0.5547),
    (580.0, 0.5125, 0.4866),
    (590.0, 0.5752, 0.4242),
    (600.0, 0.6270, 0.3725),
    (610.0, 0.6658, 0.3340),
    (620.0, 0.6915, 0.3083),
    (630.0, 0.7079, 0.2920),
    (640.0, 0.7190, 0.2809),
    (650.0, 0.7260, 0.2740),
    (660.0, 0.7300, 0.2700),
    (670.0, 0.7320, 0.2680),
    (680.0, 0.7334, 0.2666),
    (690.0, 0.7344, 0.2656),
    (700.0, 0.7347, 0.2653),
];

// ─── Functions ──────────────────────────────────────────────────────────────

/// Convert the compiled table into a vector of [`SpectralPoint`]s.
#[must_use]
pub fn spectral_locus_points() -> Vec<SpectralPoint> {
    SPECTRAL_LOCUS_10NM
        .iter()
        .map(|&(w, x, y)| SpectralPoint::new(w, x, y))
        .collect()
}

/// Find the nearest spectral locus point to a given chromaticity `(cx, cy)`.
///
/// Returns `None` only if the locus table is empty (it never is for the
/// built-in data).
#[must_use]
pub fn nearest_spectral_point(cx: f64, cy: f64) -> Option<SpectralPoint> {
    let pts = spectral_locus_points();
    pts.into_iter().min_by(|a, b| {
        let da = (a.x - cx).powi(2) + (a.y - cy).powi(2);
        let db = (b.x - cx).powi(2) + (b.y - cy).powi(2);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Estimate the dominant wavelength for a chromaticity `(cx, cy)` relative
/// to a white point `(wx, wy)`.
///
/// Dominant wavelength is found by extending a ray from the white point
/// through the sample chromaticity to the spectral locus boundary. This
/// simplified version returns the nearest spectral point along that
/// direction.
///
/// Returns `None` if the computation is degenerate (white point = sample).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn dominant_wavelength(cx: f64, cy: f64, wx: f64, wy: f64) -> Option<f64> {
    let dx = cx - wx;
    let dy = cy - wy;
    if dx.abs() < 1e-12 && dy.abs() < 1e-12 {
        return None; // sample is at white point
    }
    // Extend the ray and find the locus point closest to the ray
    let pts = spectral_locus_points();
    let mut best_wl = 0.0_f64;
    let mut best_proj = f64::NEG_INFINITY;
    let ray_len = (dx * dx + dy * dy).sqrt();
    let ux = dx / ray_len;
    let uy = dy / ray_len;
    for pt in &pts {
        let px = pt.x - wx;
        let py = pt.y - wy;
        let proj = px * ux + py * uy;
        if proj > best_proj {
            best_proj = proj;
            best_wl = pt.wavelength_nm;
        }
    }
    Some(best_wl)
}

/// Compute the spectral purity (excitation purity) of a chromaticity
/// `(cx, cy)` relative to white point `(wx, wy)`.
///
/// Purity ranges from 0.0 (at the white point) to 1.0 (on the spectral
/// locus boundary). Values > 1.0 can occur for out-of-locus purples.
#[must_use]
pub fn spectral_purity(cx: f64, cy: f64, wx: f64, wy: f64) -> f64 {
    let sample_dist = ((cx - wx).powi(2) + (cy - wy).powi(2)).sqrt();
    if sample_dist < 1e-12 {
        return 0.0;
    }
    // Find the locus point in the same direction
    let dx = cx - wx;
    let dy = cy - wy;
    let ux = dx / sample_dist;
    let uy = dy / sample_dist;
    let pts = spectral_locus_points();
    let mut best_dist = 0.0_f64;
    for pt in &pts {
        let px = pt.x - wx;
        let py = pt.y - wy;
        let proj = px * ux + py * uy;
        if proj > 0.0 {
            let d = (px * px + py * py).sqrt();
            if d > best_dist {
                best_dist = d;
            }
        }
    }
    if best_dist < 1e-12 {
        return 0.0;
    }
    (sample_dist / best_dist).min(2.0)
}

// ─── Planckian locus ────────────────────────────────────────────────────────

/// Approximate chromaticity `(x, y)` for a blackbody radiator at a given
/// Correlated Color Temperature (CCT) in Kelvin.
///
/// Uses the CIE approximation formulas valid for 1667 K to 25 000 K.
///
/// Returns `None` if `cct_k` is outside the valid range.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn planckian_xy(cct_k: f64) -> Option<(f64, f64)> {
    if cct_k < 1667.0 || cct_k > 25000.0 {
        return None;
    }
    let t = cct_k;
    let t2 = t * t;
    let t3 = t2 * t;
    let x = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    };
    let x2 = x * x;
    let x3 = x2 * x;
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };
    Some((x, y))
}

/// Standard illuminant D65 chromaticity.
#[must_use]
pub fn d65_chromaticity() -> (f64, f64) {
    (0.3127, 0.3290)
}

/// Standard illuminant D50 chromaticity.
#[must_use]
pub fn d50_chromaticity() -> (f64, f64) {
    (0.3457, 0.3585)
}

/// Correlated Color Temperature (CCT) estimate using McCamy's approximation.
///
/// Valid for CCTs roughly between 2000 K and 12 500 K.
#[must_use]
pub fn mccamy_cct(cx: f64, cy: f64) -> f64 {
    let n = (cx - 0.3320) / (0.1858 - cy);
    449.0 * n.powi(3) + 3525.0 * n.powi(2) + 6823.3 * n + 5520.33
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_locus_table_length() {
        assert_eq!(SPECTRAL_LOCUS_10NM.len(), 33);
    }

    #[test]
    fn test_spectral_locus_points_count() {
        let pts = spectral_locus_points();
        assert_eq!(pts.len(), 33);
    }

    #[test]
    fn test_spectral_point_display() {
        let p = SpectralPoint::new(550.0, 0.3016, 0.6923);
        let s = format!("{p}");
        assert!(s.contains("550 nm"));
    }

    #[test]
    fn test_nearest_spectral_point_exact() {
        let p =
            nearest_spectral_point(0.1741, 0.0050).expect("nearest spectral point should be found");
        assert!((p.wavelength_nm - 380.0).abs() < 1e-9);
    }

    #[test]
    fn test_nearest_spectral_point_close_to_red() {
        let p = nearest_spectral_point(0.73, 0.27).expect("nearest spectral point should be found");
        // Should be near the red end (660+ nm)
        assert!(p.wavelength_nm >= 650.0);
    }

    #[test]
    fn test_dominant_wavelength_red() {
        let (wx, wy) = d65_chromaticity();
        let wl = dominant_wavelength(0.65, 0.33, wx, wy)
            .expect("dominant wavelength should be computed");
        // A reddish chromaticity should yield a long wavelength
        assert!(wl >= 580.0, "expected red-ish wavelength, got {wl}");
    }

    #[test]
    fn test_dominant_wavelength_at_white() {
        let (wx, wy) = d65_chromaticity();
        let result = dominant_wavelength(wx, wy, wx, wy);
        assert!(result.is_none());
    }

    #[test]
    fn test_spectral_purity_at_white() {
        let (wx, wy) = d65_chromaticity();
        let p = spectral_purity(wx, wy, wx, wy);
        assert!(p < 1e-9);
    }

    #[test]
    fn test_spectral_purity_positive() {
        let (wx, wy) = d65_chromaticity();
        let p = spectral_purity(0.65, 0.33, wx, wy);
        assert!(p > 0.0, "purity should be > 0 away from white");
    }

    #[test]
    fn test_planckian_xy_d65() {
        let (x, y) = planckian_xy(6500.0).expect("planckian xy should be computed");
        // D65 ≈ (0.3127, 0.3290) — CIE approximation won't be exact
        assert!((x - 0.3127).abs() < 0.02, "x={x}");
        assert!((y - 0.3290).abs() < 0.02, "y={y}");
    }

    #[test]
    fn test_planckian_xy_out_of_range() {
        assert!(planckian_xy(500.0).is_none());
        assert!(planckian_xy(30000.0).is_none());
    }

    #[test]
    fn test_planckian_xy_tungsten() {
        let (x, y) = planckian_xy(3200.0).expect("planckian xy should be computed");
        // Tungsten should be warm (high x, low-ish y relative to D65)
        assert!(x > 0.4, "tungsten x should be warm, got {x}");
        assert!(y > 0.3, "tungsten y should be reasonable, got {y}");
    }

    #[test]
    fn test_d65_chromaticity() {
        let (x, y) = d65_chromaticity();
        assert!((x - 0.3127).abs() < 1e-9);
        assert!((y - 0.3290).abs() < 1e-9);
    }

    #[test]
    fn test_d50_chromaticity() {
        let (x, y) = d50_chromaticity();
        assert!((x - 0.3457).abs() < 1e-9);
        assert!((y - 0.3585).abs() < 1e-9);
    }

    #[test]
    fn test_mccamy_cct_d65() {
        let cct = mccamy_cct(0.3127, 0.3290);
        // Should be near 6500K (McCamy is approximate)
        assert!(
            (cct - 6500.0).abs() < 500.0,
            "D65 CCT should be ~6500K, got {cct}"
        );
    }

    #[test]
    fn test_spectral_locus_wavelength_range() {
        let pts = spectral_locus_points();
        let first = pts.first().expect("first element should exist");
        let last = pts.last().expect("last element should exist");
        assert!((first.wavelength_nm - 380.0).abs() < 1e-9);
        assert!((last.wavelength_nm - 700.0).abs() < 1e-9);
    }
}
