#![allow(dead_code)]
//! Spectral power distribution (SPD) handling for physically-based color science.
//!
//! This module provides tools for working with spectral data — storing spectral
//! power distributions, converting them to CIE XYZ tri-stimulus values, computing
//! correlated color temperature (CCT), and performing spectral arithmetic.

use std::collections::BTreeMap;

/// A spectral power distribution represented as wavelength-power pairs.
///
/// Wavelengths are in nanometres (typically 380–780 nm for the visible spectrum).
/// Power values are arbitrary linear units.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralDistribution {
    /// Ordered map of wavelength (nm) to power.
    samples: BTreeMap<u16, f64>,
}

impl SpectralDistribution {
    /// Creates a new empty spectral distribution.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: BTreeMap::new(),
        }
    }

    /// Creates a distribution from an iterator of (wavelength_nm, power) pairs.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u16, f64)>) -> Self {
        Self {
            samples: pairs.into_iter().collect(),
        }
    }

    /// Inserts or replaces the power at a given wavelength.
    pub fn set(&mut self, wavelength_nm: u16, power: f64) {
        self.samples.insert(wavelength_nm, power);
    }

    /// Gets the power at a given wavelength, or `None` if not sampled.
    #[must_use]
    pub fn get(&self, wavelength_nm: u16) -> Option<f64> {
        self.samples.get(&wavelength_nm).copied()
    }

    /// Returns the number of sample points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` if no samples are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Returns the wavelength range as `(min, max)`, or `None` if empty.
    #[must_use]
    pub fn wavelength_range(&self) -> Option<(u16, u16)> {
        let min = *self.samples.keys().next()?;
        let max = *self.samples.keys().next_back()?;
        Some((min, max))
    }

    /// Linearly interpolates the power at a given wavelength.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn interpolate(&self, wavelength_nm: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let wl = wavelength_nm.round() as i64;

        // Exact match
        if let Some(&v) = self.samples.get(&(wl as u16)) {
            return v;
        }

        // Find surrounding samples
        let mut lower: Option<(u16, f64)> = None;
        let mut upper: Option<(u16, f64)> = None;

        for (&w, &p) in &self.samples {
            if (w as i64) <= wl {
                lower = Some((w, p));
            }
            if (w as i64) >= wl && upper.is_none() {
                upper = Some((w, p));
            }
        }

        match (lower, upper) {
            (Some((_, lp)), None) => lp,
            (None, Some((_, up))) => up,
            (Some((lw, lp)), Some((uw, up))) => {
                if lw == uw {
                    return lp;
                }
                let t = (wavelength_nm - lw as f64) / (uw as f64 - lw as f64);
                lp + t * (up - lp)
            }
            (None, None) => 0.0,
        }
    }

    /// Scales all power values by a constant factor.
    #[must_use]
    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            samples: self
                .samples
                .iter()
                .map(|(&w, &p)| (w, p * factor))
                .collect(),
        }
    }

    /// Adds another SPD point-wise (only at matching wavelengths).
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (&w, &p) in &other.samples {
            let entry = result.samples.entry(w).or_insert(0.0);
            *entry += p;
        }
        result
    }

    /// Returns the peak wavelength (wavelength with the highest power).
    #[must_use]
    pub fn peak_wavelength(&self) -> Option<u16> {
        self.samples
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&w, _)| w)
    }

    /// Returns the total integrated power (trapezoidal rule).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn total_power(&self) -> f64 {
        let keys: Vec<u16> = self.samples.keys().copied().collect();
        if keys.len() < 2 {
            return self.samples.values().sum();
        }
        let mut total = 0.0;
        for i in 0..keys.len() - 1 {
            let dw = (keys[i + 1] - keys[i]) as f64;
            let p0 = self.samples[&keys[i]];
            let p1 = self.samples[&keys[i + 1]];
            total += 0.5 * dw * (p0 + p1);
        }
        total
    }

    /// Normalizes the distribution so the peak power is 1.0.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let peak = self.samples.values().copied().fold(0.0_f64, f64::max);
        if peak <= f64::EPSILON {
            return self.clone();
        }
        self.scaled(1.0 / peak)
    }
}

impl Default for SpectralDistribution {
    fn default() -> Self {
        Self::new()
    }
}

/// CIE XYZ tri-stimulus values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CieXyz {
    /// X tri-stimulus value.
    pub x: f64,
    /// Y tri-stimulus value (luminance).
    pub y: f64,
    /// Z tri-stimulus value.
    pub z: f64,
}

impl CieXyz {
    /// Creates a new XYZ value.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Returns the CIE xy chromaticity coordinates.
    #[must_use]
    pub fn to_chromaticity(&self) -> (f64, f64) {
        let sum = self.x + self.y + self.z;
        if sum.abs() < f64::EPSILON {
            return (0.0, 0.0);
        }
        (self.x / sum, self.y / sum)
    }
}

/// CIE 1931 2-degree standard observer colour-matching functions (5 nm spacing).
///
/// Returns (x_bar, y_bar, z_bar) at a given wavelength. Returns zeros outside 380-780 nm.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn cie_1931_cmf(wavelength_nm: u16) -> (f64, f64, f64) {
    // Simplified Gaussian approximation of the CIE 1931 observer
    let w = wavelength_nm as f64;
    if !(380.0..=780.0).contains(&w) {
        return (0.0, 0.0, 0.0);
    }

    let gauss = |x: f64, mu: f64, sigma: f64| -> f64 { (-0.5 * ((x - mu) / sigma).powi(2)).exp() };

    let x_bar = 1.056 * gauss(w, 599.8, 37.9) + 0.362 * gauss(w, 442.0, 16.0)
        - 0.065 * gauss(w, 501.1, 20.4);
    let y_bar = 0.821 * gauss(w, 568.8, 46.9) + 0.286 * gauss(w, 530.9, 16.3);
    let z_bar = 1.217 * gauss(w, 437.0, 11.8) + 0.681 * gauss(w, 459.0, 26.0);

    (x_bar.max(0.0), y_bar.max(0.0), z_bar.max(0.0))
}

/// Converts a spectral distribution to CIE XYZ using numerical integration.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn spd_to_xyz(spd: &SpectralDistribution) -> CieXyz {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;

    // Integrate over 5 nm steps from 380 to 780
    for wl in (380..=780).step_by(5) {
        let power = spd.interpolate(wl as f64);
        let (xb, yb, zb) = cie_1931_cmf(wl);
        x += power * xb * 5.0;
        y += power * yb * 5.0;
        z += power * zb * 5.0;
    }

    // Normalize so that an equal-energy illuminant with Y=1
    let k = 1.0 / (y.max(f64::EPSILON));
    CieXyz::new(x * k, 1.0, z * k)
}

/// Estimates correlated color temperature (CCT) from CIE xy chromaticity.
///
/// Uses McCamy's approximation. Valid for approximately 2000K–12500K.
#[must_use]
pub fn estimate_cct(cx: f64, cy: f64) -> f64 {
    let n = (cx - 0.3320) / (0.1858 - cy);
    449.0 * n.powi(3) + 3525.0 * n.powi(2) + 6823.3 * n + 5520.33
}

/// Creates a standard illuminant D65 spectral approximation.
#[must_use]
pub fn d65_illuminant() -> SpectralDistribution {
    // Simplified D65 relative SPD at 20 nm intervals
    SpectralDistribution::from_pairs([
        (380, 49.98),
        (400, 82.75),
        (420, 93.43),
        (440, 104.86),
        (460, 117.01),
        (480, 115.10),
        (500, 109.35),
        (520, 104.79),
        (540, 104.41),
        (560, 100.0),
        (580, 95.79),
        (600, 90.01),
        (620, 87.38),
        (640, 83.70),
        (660, 80.21),
        (680, 78.28),
        (700, 71.61),
        (720, 68.26),
        (740, 61.60),
        (760, 56.51),
        (780, 51.98),
    ])
}

/// Creates a standard illuminant A (tungsten) spectral approximation.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn illuminant_a() -> SpectralDistribution {
    let mut spd = SpectralDistribution::new();
    // Planckian at 2856K
    for wl in (380..=780).step_by(5) {
        let w_m = wl as f64 * 1e-9;
        let power = 1.0 / (w_m.powi(5) * ((1.4388e-2 / (w_m * 2856.0)).exp() - 1.0));
        spd.set(wl, power);
    }
    spd.normalized()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spd_new_empty() {
        let spd = SpectralDistribution::new();
        assert!(spd.is_empty());
        assert_eq!(spd.len(), 0);
    }

    #[test]
    fn test_spd_from_pairs() {
        let spd = SpectralDistribution::from_pairs([(550, 1.0), (600, 0.8)]);
        assert_eq!(spd.len(), 2);
        assert!((spd.get(550).expect("spectral value should exist") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spd_set_get() {
        let mut spd = SpectralDistribution::new();
        spd.set(500, 0.5);
        assert!((spd.get(500).expect("spectral value should exist") - 0.5).abs() < f64::EPSILON);
        assert!(spd.get(501).is_none());
    }

    #[test]
    fn test_wavelength_range() {
        let spd = SpectralDistribution::from_pairs([(400, 0.1), (700, 0.9)]);
        assert_eq!(spd.wavelength_range(), Some((400, 700)));
    }

    #[test]
    fn test_wavelength_range_empty() {
        let spd = SpectralDistribution::new();
        assert_eq!(spd.wavelength_range(), None);
    }

    #[test]
    fn test_interpolate_exact() {
        let spd = SpectralDistribution::from_pairs([(500, 1.0)]);
        assert!((spd.interpolate(500.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_interpolate_between() {
        let spd = SpectralDistribution::from_pairs([(500, 0.0), (600, 1.0)]);
        let v = spd.interpolate(550.0);
        assert!((v - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_peak_wavelength() {
        let spd = SpectralDistribution::from_pairs([(400, 0.1), (550, 1.0), (700, 0.5)]);
        assert_eq!(spd.peak_wavelength(), Some(550));
    }

    #[test]
    fn test_total_power() {
        // Rectangle: constant 1.0 over 100nm → area = 100
        let spd = SpectralDistribution::from_pairs([(500, 1.0), (600, 1.0)]);
        assert!((spd.total_power() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scaled() {
        let spd = SpectralDistribution::from_pairs([(500, 2.0)]);
        let scaled = spd.scaled(0.5);
        assert!((scaled.get(500).expect("spectral value should exist") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalized() {
        let spd = SpectralDistribution::from_pairs([(500, 4.0), (600, 2.0)]);
        let norm = spd.normalized();
        assert!((norm.get(500).expect("spectral value should exist") - 1.0).abs() < f64::EPSILON);
        assert!((norm.get(600).expect("spectral value should exist") - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_add_spds() {
        let a = SpectralDistribution::from_pairs([(500, 1.0)]);
        let b = SpectralDistribution::from_pairs([(500, 0.5), (600, 0.3)]);
        let c = a.add(&b);
        assert!((c.get(500).expect("spectral value should exist") - 1.5).abs() < f64::EPSILON);
        assert!((c.get(600).expect("spectral value should exist") - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cie_cmf_out_of_range() {
        let (x, y, z) = cie_1931_cmf(300);
        assert!((x).abs() < f64::EPSILON);
        assert!((y).abs() < f64::EPSILON);
        assert!((z).abs() < f64::EPSILON);
    }

    #[test]
    fn test_d65_illuminant() {
        let d65 = d65_illuminant();
        assert!(!d65.is_empty());
        assert!(d65.get(560).expect("spectral value should exist") > 0.0);
    }

    #[test]
    fn test_estimate_cct_d65() {
        // D65 chromaticity is approximately x=0.3127, y=0.3290
        let cct = estimate_cct(0.3127, 0.3290);
        assert!((cct - 6500.0).abs() < 500.0); // Within 500K
    }
}
