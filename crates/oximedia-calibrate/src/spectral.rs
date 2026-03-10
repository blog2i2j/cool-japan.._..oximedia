//! Spectral reflectance and color matching functions.
//!
//! Provides spectral power distribution (SPD) representation, CIE color
//! matching functions (CMF), tristimulus integration, and metameric index
//! calculations for professional color science workflows.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// SpectralSample
// ---------------------------------------------------------------------------

/// A single spectral sample at a given wavelength.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralSample {
    /// Wavelength in nanometers.
    pub wavelength_nm: f64,
    /// Spectral power / reflectance value (linear, arbitrary units).
    pub value: f64,
}

// ---------------------------------------------------------------------------
// SpectralPowerDistribution
// ---------------------------------------------------------------------------

/// A sampled spectral power distribution (SPD) or reflectance curve.
///
/// Samples must be stored in ascending wavelength order.  The helper
/// constructors enforce this.
#[derive(Debug, Clone)]
pub struct SpectralPowerDistribution {
    samples: Vec<SpectralSample>,
}

impl SpectralPowerDistribution {
    /// Create an SPD from a slice of `(wavelength_nm, value)` pairs.
    ///
    /// Samples are automatically sorted by wavelength.
    #[must_use]
    pub fn from_pairs(pairs: &[(f64, f64)]) -> Self {
        let mut samples: Vec<SpectralSample> = pairs
            .iter()
            .map(|&(w, v)| SpectralSample {
                wavelength_nm: w,
                value: v,
            })
            .collect();
        samples.sort_by(|a, b| {
            a.wavelength_nm
                .partial_cmp(&b.wavelength_nm)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { samples }
    }

    /// Create a flat (constant) SPD over `[start, end]` nm with `steps` samples.
    #[must_use]
    pub fn flat(start: f64, end: f64, steps: usize, value: f64) -> Self {
        assert!(steps >= 2, "at least 2 steps required");
        let step = (end - start) / (steps - 1) as f64;
        let samples = (0..steps)
            .map(|i| SpectralSample {
                wavelength_nm: start + i as f64 * step,
                value,
            })
            .collect();
        Self { samples }
    }

    /// Return the number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Return `true` if there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Linearly interpolate the SPD at `wavelength_nm`.
    ///
    /// Returns `0.0` for wavelengths outside the sampled range.
    #[must_use]
    pub fn evaluate(&self, wavelength_nm: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let first = self
            .samples
            .first()
            .expect("samples non-empty: is_empty check returned above");
        let last = self
            .samples
            .last()
            .expect("samples non-empty: is_empty check returned above");

        if wavelength_nm <= first.wavelength_nm {
            return first.value;
        }
        if wavelength_nm >= last.wavelength_nm {
            return last.value;
        }

        // Binary search for the surrounding interval
        let idx = self
            .samples
            .partition_point(|s| s.wavelength_nm <= wavelength_nm);
        let lo = &self.samples[idx - 1];
        let hi = &self.samples[idx];
        let t = (wavelength_nm - lo.wavelength_nm) / (hi.wavelength_nm - lo.wavelength_nm);
        lo.value + t * (hi.value - lo.value)
    }

    /// Compute the total power (area under the curve) via the trapezoidal rule.
    #[must_use]
    pub fn integrate(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        self.samples
            .windows(2)
            .map(|w| {
                let dw = w[1].wavelength_nm - w[0].wavelength_nm;
                0.5 * (w[0].value + w[1].value) * dw
            })
            .sum()
    }

    /// Normalise the SPD so that its maximum value equals `1.0`.
    ///
    /// Returns `self` unchanged if the maximum value is zero.
    #[must_use]
    pub fn normalise(&self) -> Self {
        let max_val = self
            .samples
            .iter()
            .map(|s| s.value)
            .fold(f64::NEG_INFINITY, f64::max);
        if max_val <= 0.0 {
            return self.clone();
        }
        let samples = self
            .samples
            .iter()
            .map(|s| SpectralSample {
                wavelength_nm: s.wavelength_nm,
                value: s.value / max_val,
            })
            .collect();
        Self { samples }
    }

    /// Multiply this SPD element-wise by another SPD, interpolating as needed.
    ///
    /// The output uses the wavelength grid of `self`.
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        let samples = self
            .samples
            .iter()
            .map(|s| SpectralSample {
                wavelength_nm: s.wavelength_nm,
                value: s.value * other.evaluate(s.wavelength_nm),
            })
            .collect();
        Self { samples }
    }
}

// ---------------------------------------------------------------------------
// CmfTable – CIE 1931 2° colour matching functions (sparse tabulation)
// ---------------------------------------------------------------------------

/// CIE 1931 2° observer colour matching function values at a single wavelength.
#[derive(Debug, Clone, Copy)]
pub struct CmfEntry {
    /// Wavelength in nm.
    pub wavelength_nm: f64,
    /// x̄ (red) matching function.
    pub x_bar: f64,
    /// ȳ (green / luminous efficiency) matching function.
    pub y_bar: f64,
    /// z̄ (blue) matching function.
    pub z_bar: f64,
}

/// A minimal tabulation of the CIE 1931 2° observer CMF (380–780 nm, 20 nm
/// steps) sufficient for demonstration and unit-testing purposes.
///
/// Values sourced from CIE publication 15:2004 Table 1.
#[must_use]
pub fn cie1931_cmf_table() -> Vec<CmfEntry> {
    // (wavelength, x̄, ȳ, z̄) — 20 nm steps 380–780 nm
    let data: &[(f64, f64, f64, f64)] = &[
        (380.0, 0.001_368, 0.000_039, 0.006_450),
        (400.0, 0.014_310, 0.000_396, 0.067_850),
        (420.0, 0.134_380, 0.004_000, 0.645_600),
        (440.0, 0.348_280, 0.023_000, 1.747_060),
        (460.0, 0.290_800, 0.060_000, 1.669_200),
        (480.0, 0.095_640, 0.139_020, 0.812_950),
        (500.0, 0.004_900, 0.323_000, 0.272_000),
        (520.0, 0.063_270, 0.710_000, 0.078_250),
        (540.0, 0.290_400, 0.954_000, 0.020_300),
        (560.0, 0.594_500, 0.995_000, 0.003_900),
        (580.0, 0.916_300, 0.870_000, 0.001_700),
        (600.0, 1.062_200, 0.631_000, 0.000_800),
        (620.0, 0.854_450, 0.381_000, 0.000_190),
        (640.0, 0.447_900, 0.175_000, 0.000_020),
        (660.0, 0.164_900, 0.061_000, 0.000_000),
        (680.0, 0.046_770, 0.017_000, 0.000_000),
        (700.0, 0.011_359, 0.004_102, 0.000_000),
        (720.0, 0.002_899, 0.001_047, 0.000_000),
        (740.0, 0.000_690, 0.000_249, 0.000_000),
        (760.0, 0.000_166, 0.000_060, 0.000_000),
        (780.0, 0.000_042, 0.000_015, 0.000_000),
    ];
    data.iter()
        .map(|&(w, x, y, z)| CmfEntry {
            wavelength_nm: w,
            x_bar: x,
            y_bar: y,
            z_bar: z,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tristimulus integration
// ---------------------------------------------------------------------------

/// Compute CIE XYZ tristimulus values by integrating an SPD against the
/// CIE 1931 2° CMF table using the trapezoidal rule.
///
/// Returns `[X, Y, Z]`.  The result is unnormalised (absolute units depend on
/// SPD units).
#[must_use]
pub fn spd_to_xyz(spd: &SpectralPowerDistribution, cmf: &[CmfEntry]) -> [f64; 3] {
    if cmf.len() < 2 {
        return [0.0; 3];
    }
    let mut x = 0.0_f64;
    let mut y = 0.0_f64;
    let mut z = 0.0_f64;

    for window in cmf.windows(2) {
        let lo = &window[0];
        let hi = &window[1];
        let dw = hi.wavelength_nm - lo.wavelength_nm;
        let mid_w = 0.5 * (lo.wavelength_nm + hi.wavelength_nm);
        let s = spd.evaluate(mid_w);
        let x_bar = 0.5 * (lo.x_bar + hi.x_bar);
        let y_bar = 0.5 * (lo.y_bar + hi.y_bar);
        let z_bar = 0.5 * (lo.z_bar + hi.z_bar);
        x += s * x_bar * dw;
        y += s * y_bar * dw;
        z += s * z_bar * dw;
    }
    [x, y, z]
}

/// Compute the xy chromaticity coordinates from XYZ.
///
/// Returns `(NaN, NaN)` if `X + Y + Z = 0`.
#[must_use]
pub fn xyz_to_xy(xyz: [f64; 3]) -> (f64, f64) {
    let sum = xyz[0] + xyz[1] + xyz[2];
    if sum.abs() < f64::EPSILON {
        return (f64::NAN, f64::NAN);
    }
    (xyz[0] / sum, xyz[1] / sum)
}

// ---------------------------------------------------------------------------
// MetamericIndex
// ---------------------------------------------------------------------------

/// Computes a simple metamerism index between two SPDs under two illuminants.
///
/// Two spectra are metamers if they produce the same XYZ under illuminant A
/// but differ under illuminant B.  The index is the Euclidean distance in XYZ
/// between the two spectra under illuminant B, after both are normalised to
/// Y = 1 under illuminant A.
#[must_use]
pub fn metameric_index(
    spd1: &SpectralPowerDistribution,
    spd2: &SpectralPowerDistribution,
    illuminant_a: &SpectralPowerDistribution,
    illuminant_b: &SpectralPowerDistribution,
    cmf: &[CmfEntry],
) -> f64 {
    let ref1_a = spd1.multiply(illuminant_a);
    let ref2_a = spd2.multiply(illuminant_a);
    let xyz1_a = spd_to_xyz(&ref1_a, cmf);
    let xyz2_a = spd_to_xyz(&ref2_a, cmf);

    // Normalise by Y under illuminant A
    let y1_a = xyz1_a[1].max(f64::EPSILON);
    let y2_a = xyz2_a[1].max(f64::EPSILON);

    let ref1_b = spd1.multiply(illuminant_b);
    let ref2_b = spd2.multiply(illuminant_b);
    let xyz1_b = spd_to_xyz(&ref1_b, cmf);
    let xyz2_b = spd_to_xyz(&ref2_b, cmf);

    // Normalised XYZ under B
    let n1 = [xyz1_b[0] / y1_a, xyz1_b[1] / y1_a, xyz1_b[2] / y1_a];
    let n2 = [xyz2_b[0] / y2_a, xyz2_b[1] / y2_a, xyz2_b[2] / y2_a];

    let dx = n1[0] - n2[0];
    let dy = n1[1] - n2[1];
    let dz = n1[2] - n2[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SpectralPowerDistribution ─────────────────────────────────────────

    #[test]
    fn test_from_pairs_sorted() {
        let spd =
            SpectralPowerDistribution::from_pairs(&[(600.0, 0.8), (400.0, 0.5), (500.0, 0.7)]);
        assert_eq!(spd.samples[0].wavelength_nm, 400.0);
        assert_eq!(spd.samples[2].wavelength_nm, 600.0);
    }

    #[test]
    fn test_flat_spd_length() {
        let spd = SpectralPowerDistribution::flat(380.0, 780.0, 41, 1.0);
        assert_eq!(spd.len(), 41);
    }

    #[test]
    fn test_flat_spd_is_not_empty() {
        let spd = SpectralPowerDistribution::flat(380.0, 780.0, 5, 0.5);
        assert!(!spd.is_empty());
    }

    #[test]
    fn test_evaluate_at_known_point() {
        let spd = SpectralPowerDistribution::from_pairs(&[(400.0, 0.0), (600.0, 1.0)]);
        // Midpoint should interpolate to 0.5
        let v = spd.evaluate(500.0);
        assert!((v - 0.5).abs() < 1e-9, "v={v}");
    }

    #[test]
    fn test_evaluate_clamps_below_range() {
        let spd = SpectralPowerDistribution::from_pairs(&[(500.0, 0.3), (600.0, 0.9)]);
        assert!((spd.evaluate(300.0) - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_clamps_above_range() {
        let spd = SpectralPowerDistribution::from_pairs(&[(500.0, 0.3), (600.0, 0.9)]);
        assert!((spd.evaluate(800.0) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_integrate_flat_spd() {
        // Flat SPD of value 1.0 over 100 nm → area = 100
        let spd = SpectralPowerDistribution::from_pairs(&[(400.0, 1.0), (500.0, 1.0)]);
        let area = spd.integrate();
        assert!((area - 100.0).abs() < 1e-9, "area={area}");
    }

    #[test]
    fn test_integrate_empty_spd() {
        let spd = SpectralPowerDistribution::from_pairs(&[]);
        assert!((spd.integrate()).abs() < 1e-12);
    }

    #[test]
    fn test_normalise_peak_is_one() {
        let spd =
            SpectralPowerDistribution::from_pairs(&[(400.0, 2.0), (500.0, 4.0), (600.0, 1.0)]);
        let norm = spd.normalise();
        let max_val = norm
            .samples
            .iter()
            .map(|s| s.value)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((max_val - 1.0).abs() < 1e-9, "max={max_val}");
    }

    #[test]
    fn test_normalise_zero_spd_unchanged() {
        let spd = SpectralPowerDistribution::from_pairs(&[(400.0, 0.0), (500.0, 0.0)]);
        let norm = spd.normalise();
        assert!((norm.samples[0].value).abs() < 1e-12);
    }

    #[test]
    fn test_multiply_flat_spds() {
        let a = SpectralPowerDistribution::flat(400.0, 600.0, 5, 2.0);
        let b = SpectralPowerDistribution::flat(400.0, 600.0, 5, 3.0);
        let prod = a.multiply(&b);
        for s in &prod.samples {
            assert!((s.value - 6.0).abs() < 1e-9, "value={}", s.value);
        }
    }

    // ── CMF ──────────────────────────────────────────────────────────────

    #[test]
    fn test_cmf_table_not_empty() {
        let cmf = cie1931_cmf_table();
        assert!(!cmf.is_empty());
    }

    #[test]
    fn test_cmf_wavelengths_ascending() {
        let cmf = cie1931_cmf_table();
        for w in cmf.windows(2) {
            assert!(w[1].wavelength_nm > w[0].wavelength_nm);
        }
    }

    #[test]
    fn test_cmf_y_bar_peaks_around_555nm() {
        let cmf = cie1931_cmf_table();
        // Find maximum y_bar entry
        let peak = cmf
            .iter()
            .max_by(|a, b| {
                a.y_bar
                    .partial_cmp(&b.y_bar)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("unexpected None/Err");
        // Peak of photopic luminous efficiency is around 555 nm; our sparse
        // table peaks at 560 nm which is the nearest 20-nm step.
        assert!(
            (peak.wavelength_nm - 560.0).abs() < 30.0,
            "peak at {}",
            peak.wavelength_nm
        );
    }

    // ── spd_to_xyz ───────────────────────────────────────────────────────

    #[test]
    fn test_spd_to_xyz_zero_spd_gives_zero() {
        let spd = SpectralPowerDistribution::flat(380.0, 780.0, 41, 0.0);
        let cmf = cie1931_cmf_table();
        let xyz = spd_to_xyz(&spd, &cmf);
        assert!(xyz[0].abs() < 1e-9 && xyz[1].abs() < 1e-9 && xyz[2].abs() < 1e-9);
    }

    #[test]
    fn test_spd_to_xyz_flat_gives_nonzero_y() {
        let spd = SpectralPowerDistribution::flat(380.0, 780.0, 41, 1.0);
        let cmf = cie1931_cmf_table();
        let xyz = spd_to_xyz(&spd, &cmf);
        assert!(xyz[1] > 0.0, "Y should be positive for non-zero flat SPD");
    }

    // ── xyz_to_xy ────────────────────────────────────────────────────────

    #[test]
    fn test_xyz_to_xy_known_point() {
        let xyz = [0.95047, 1.0, 1.08883]; // D65 white
        let (x, y) = xyz_to_xy(xyz);
        // D65 chromaticity: x≈0.3127, y≈0.3290
        assert!((x - 0.3127).abs() < 0.002, "x={x}");
        assert!((y - 0.3290).abs() < 0.002, "y={y}");
    }

    #[test]
    fn test_xyz_to_xy_zero_returns_nan() {
        let (x, y) = xyz_to_xy([0.0, 0.0, 0.0]);
        assert!(x.is_nan() && y.is_nan());
    }

    // ── metameric_index ──────────────────────────────────────────────────

    #[test]
    fn test_metameric_index_identical_spds_is_zero() {
        let cmf = cie1931_cmf_table();
        let spd = SpectralPowerDistribution::flat(380.0, 780.0, 41, 0.5);
        let illum = SpectralPowerDistribution::flat(380.0, 780.0, 41, 1.0);
        let idx = metameric_index(&spd, &spd, &illum, &illum, &cmf);
        assert!(
            idx.abs() < 1e-6,
            "index should be 0 for identical SPDs, got {idx}"
        );
    }
}
