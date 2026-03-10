#![allow(dead_code)]
//! Noise profiling and fingerprinting for audio restoration.
//!
//! Captures the spectral characteristics of noise from a reference segment
//! so it can be subtracted or gated from a full recording. Supports
//! stationary noise models, band-limited profiles, and profile interpolation.

use std::collections::HashMap;

/// Number of default spectral bands for profiling.
const DEFAULT_BANDS: usize = 256;

/// A single spectral band measurement.
#[derive(Debug, Clone, Copy)]
pub struct BandMeasurement {
    /// Centre frequency of the band in Hz.
    pub center_hz: f64,
    /// Average power level in dB.
    pub avg_power_db: f64,
    /// Peak power level in dB.
    pub peak_power_db: f64,
    /// Standard deviation of power in dB.
    pub std_dev_db: f64,
}

impl BandMeasurement {
    /// Create a new band measurement.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(center_hz: f64, avg_power_db: f64, peak_power_db: f64, std_dev_db: f64) -> Self {
        Self {
            center_hz,
            avg_power_db,
            peak_power_db,
            std_dev_db,
        }
    }

    /// Return the dynamic range (peak minus average) in dB.
    pub fn dynamic_range_db(&self) -> f64 {
        self.peak_power_db - self.avg_power_db
    }
}

/// Type of noise model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseType {
    /// Constant-level noise (e.g., hiss, hum).
    Stationary,
    /// Noise that varies slowly over time.
    SlowlyVarying,
    /// Impulsive noise (clicks, pops).
    Impulsive,
    /// Broadband noise with uniform spectral density.
    Broadband,
}

impl std::fmt::Display for NoiseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stationary => write!(f, "stationary"),
            Self::SlowlyVarying => write!(f, "slowly-varying"),
            Self::Impulsive => write!(f, "impulsive"),
            Self::Broadband => write!(f, "broadband"),
        }
    }
}

/// A complete noise profile captured from a reference segment.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Unique identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Sample rate the profile was captured at.
    pub sample_rate: u32,
    /// Type of noise modelled.
    pub noise_type: NoiseType,
    /// Per-band spectral measurements.
    pub bands: Vec<BandMeasurement>,
    /// Duration of the reference segment in seconds.
    pub reference_duration_secs: f64,
    /// Overall RMS level of the noise in dB.
    pub rms_level_db: f64,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

impl NoiseProfile {
    /// Create a new noise profile.
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        sample_rate: u32,
        noise_type: NoiseType,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            sample_rate,
            noise_type,
            bands: Vec::new(),
            reference_duration_secs: 0.0,
            rms_level_db: -96.0,
            metadata: HashMap::new(),
        }
    }

    /// Add a band measurement.
    pub fn add_band(&mut self, band: BandMeasurement) {
        self.bands.push(band);
    }

    /// Return the number of spectral bands.
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Return the frequency range covered by the profile.
    pub fn frequency_range(&self) -> Option<(f64, f64)> {
        if self.bands.is_empty() {
            return None;
        }
        let min = self
            .bands
            .iter()
            .map(|b| b.center_hz)
            .fold(f64::INFINITY, f64::min);
        let max = self
            .bands
            .iter()
            .map(|b| b.center_hz)
            .fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }

    /// Compute the average noise power across all bands.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_noise_db(&self) -> f64 {
        if self.bands.is_empty() {
            return -96.0;
        }
        let sum: f64 = self.bands.iter().map(|b| b.avg_power_db).sum();
        sum / self.bands.len() as f64
    }

    /// Find the band with the highest average noise.
    pub fn loudest_band(&self) -> Option<&BandMeasurement> {
        self.bands.iter().max_by(|a, b| {
            a.avg_power_db
                .partial_cmp(&b.avg_power_db)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the band with the lowest average noise.
    pub fn quietest_band(&self) -> Option<&BandMeasurement> {
        self.bands.iter().min_by(|a, b| {
            a.avg_power_db
                .partial_cmp(&b.avg_power_db)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Attach metadata.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Build a noise profile from raw spectral frames.
#[derive(Debug)]
pub struct NoiseProfileBuilder {
    /// Identifier for the profile being built.
    id: String,
    /// Label.
    label: String,
    /// Sample rate.
    sample_rate: u32,
    /// Noise type.
    noise_type: NoiseType,
    /// Accumulated power values per band (linear, not dB).
    band_powers: Vec<Vec<f64>>,
    /// Centre frequencies for each band.
    center_frequencies: Vec<f64>,
    /// Number of frames analysed.
    frame_count: u64,
}

impl NoiseProfileBuilder {
    /// Create a new builder.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        sample_rate: u32,
        num_bands: usize,
    ) -> Self {
        let bands = if num_bands == 0 {
            DEFAULT_BANDS
        } else {
            num_bands
        };
        let nyquist = f64::from(sample_rate) / 2.0;
        let center_frequencies: Vec<f64> = (0..bands)
            .map(|i| (i as f64 + 0.5) * nyquist / bands as f64)
            .collect();
        Self {
            id: id.into(),
            label: label.into(),
            sample_rate,
            noise_type: NoiseType::Stationary,
            band_powers: vec![Vec::new(); bands],
            center_frequencies,
            frame_count: 0,
        }
    }

    /// Set the noise type.
    pub fn with_noise_type(mut self, noise_type: NoiseType) -> Self {
        self.noise_type = noise_type;
        self
    }

    /// Feed a spectral frame (one power value per band) to the builder.
    ///
    /// The length of `powers` must match the number of bands.
    pub fn add_frame(&mut self, powers: &[f64]) -> bool {
        if powers.len() != self.band_powers.len() {
            return false;
        }
        for (i, &power) in powers.iter().enumerate() {
            self.band_powers[i].push(power);
        }
        self.frame_count += 1;
        true
    }

    /// Build the final noise profile from accumulated data.
    #[allow(clippy::cast_precision_loss)]
    pub fn build(self) -> NoiseProfile {
        let mut profile = NoiseProfile::new(self.id, self.label, self.sample_rate, self.noise_type);

        for (i, powers) in self.band_powers.iter().enumerate() {
            if powers.is_empty() {
                continue;
            }
            let n = powers.len() as f64;
            let avg = powers.iter().sum::<f64>() / n;
            let peak = powers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let variance = powers.iter().map(|&p| (p - avg).powi(2)).sum::<f64>() / n;
            let std_dev = variance.sqrt();

            let band = BandMeasurement::new(self.center_frequencies[i], avg, peak, std_dev);
            profile.add_band(band);
        }

        if !profile.bands.is_empty() {
            profile.rms_level_db = profile.average_noise_db();
        }
        profile.reference_duration_secs =
            self.frame_count as f64 * 1024.0 / f64::from(self.sample_rate);

        profile
    }
}

/// Interpolate between two noise profiles at a given ratio (0.0 = a, 1.0 = b).
#[allow(clippy::cast_precision_loss)]
pub fn interpolate_profiles(
    a: &NoiseProfile,
    b: &NoiseProfile,
    ratio: f64,
) -> Option<NoiseProfile> {
    if a.bands.len() != b.bands.len() {
        return None;
    }
    let ratio = ratio.clamp(0.0, 1.0);
    let inv = 1.0 - ratio;

    let mut result = NoiseProfile::new(
        format!("{}_x_{}", a.id, b.id),
        format!("interpolated({}, {})", a.label, b.label),
        a.sample_rate,
        a.noise_type,
    );

    for (ba, bb) in a.bands.iter().zip(b.bands.iter()) {
        let band = BandMeasurement::new(
            ba.center_hz * inv + bb.center_hz * ratio,
            ba.avg_power_db * inv + bb.avg_power_db * ratio,
            ba.peak_power_db * inv + bb.peak_power_db * ratio,
            ba.std_dev_db * inv + bb.std_dev_db * ratio,
        );
        result.add_band(band);
    }

    result.rms_level_db = a.rms_level_db * inv + b.rms_level_db * ratio;
    result.reference_duration_secs =
        a.reference_duration_secs * inv + b.reference_duration_secs * ratio;

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_band(center: f64, avg: f64) -> BandMeasurement {
        BandMeasurement::new(center, avg, avg + 6.0, 1.5)
    }

    fn sample_profile(id: &str, bands: usize) -> NoiseProfile {
        let mut p = NoiseProfile::new(id, id, 48000, NoiseType::Stationary);
        for i in 0..bands {
            #[allow(clippy::cast_precision_loss)]
            let center = (i as f64 + 1.0) * 100.0;
            p.add_band(sample_band(center, -60.0 + center * 0.01));
        }
        p
    }

    #[test]
    fn test_band_measurement_dynamic_range() {
        let b = BandMeasurement::new(1000.0, -40.0, -30.0, 2.0);
        assert!((b.dynamic_range_db() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_type_display() {
        assert_eq!(format!("{}", NoiseType::Stationary), "stationary");
        assert_eq!(format!("{}", NoiseType::SlowlyVarying), "slowly-varying");
        assert_eq!(format!("{}", NoiseType::Impulsive), "impulsive");
        assert_eq!(format!("{}", NoiseType::Broadband), "broadband");
    }

    #[test]
    fn test_noise_profile_new() {
        let p = NoiseProfile::new("np1", "Test Profile", 48000, NoiseType::Stationary);
        assert_eq!(p.id, "np1");
        assert_eq!(p.sample_rate, 48000);
        assert_eq!(p.band_count(), 0);
    }

    #[test]
    fn test_noise_profile_add_band() {
        let mut p = NoiseProfile::new("np1", "Test", 48000, NoiseType::Stationary);
        p.add_band(sample_band(1000.0, -40.0));
        p.add_band(sample_band(2000.0, -35.0));
        assert_eq!(p.band_count(), 2);
    }

    #[test]
    fn test_noise_profile_frequency_range() {
        let p = sample_profile("test", 10);
        let (lo, hi) = p.frequency_range().expect("frequency_range should succeed");
        assert!(lo < hi);
        assert!((lo - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_profile_empty_frequency_range() {
        let p = NoiseProfile::new("empty", "Empty", 48000, NoiseType::Stationary);
        assert!(p.frequency_range().is_none());
    }

    #[test]
    fn test_noise_profile_average_noise() {
        let mut p = NoiseProfile::new("test", "Test", 48000, NoiseType::Stationary);
        p.add_band(sample_band(1000.0, -40.0));
        p.add_band(sample_band(2000.0, -60.0));
        let avg = p.average_noise_db();
        assert!((avg - (-50.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_profile_loudest_quietest() {
        let mut p = NoiseProfile::new("test", "Test", 48000, NoiseType::Stationary);
        p.add_band(sample_band(1000.0, -40.0));
        p.add_band(sample_band(2000.0, -60.0));
        p.add_band(sample_band(3000.0, -30.0));
        assert!(
            (p.loudest_band()
                .expect("loudest_band should succeed")
                .center_hz
                - 3000.0)
                .abs()
                < f64::EPSILON
        );
        assert!(
            (p.quietest_band()
                .expect("quietest_band should succeed")
                .center_hz
                - 2000.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_noise_profile_metadata() {
        let mut p = NoiseProfile::new("test", "Test", 48000, NoiseType::Stationary);
        p.set_metadata("location", "studio-a");
        assert_eq!(
            p.metadata.get("location").expect("failed to get value"),
            "studio-a"
        );
    }

    #[test]
    fn test_builder_basic() {
        let mut builder = NoiseProfileBuilder::new("b1", "Build Test", 48000, 4);
        assert!(builder.add_frame(&[-40.0, -35.0, -50.0, -45.0]));
        assert!(builder.add_frame(&[-42.0, -33.0, -48.0, -44.0]));
        let profile = builder.build();
        assert_eq!(profile.band_count(), 4);
        assert!(profile.reference_duration_secs > 0.0);
    }

    #[test]
    fn test_builder_wrong_frame_size() {
        let mut builder = NoiseProfileBuilder::new("b1", "Build Test", 48000, 4);
        assert!(!builder.add_frame(&[-40.0, -35.0])); // wrong size
    }

    #[test]
    fn test_interpolate_profiles() {
        let a = sample_profile("a", 5);
        let b = sample_profile("b", 5);
        let result = interpolate_profiles(&a, &b, 0.5).expect("operation should succeed");
        assert_eq!(result.band_count(), 5);
    }

    #[test]
    fn test_interpolate_mismatched_bands() {
        let a = sample_profile("a", 3);
        let b = sample_profile("b", 5);
        assert!(interpolate_profiles(&a, &b, 0.5).is_none());
    }

    #[test]
    fn test_interpolate_at_zero() {
        let mut a = NoiseProfile::new("a", "A", 48000, NoiseType::Stationary);
        a.add_band(sample_band(1000.0, -40.0));
        a.rms_level_db = -40.0;

        let mut b = NoiseProfile::new("b", "B", 48000, NoiseType::Stationary);
        b.add_band(sample_band(2000.0, -20.0));
        b.rms_level_db = -20.0;

        let result = interpolate_profiles(&a, &b, 0.0).expect("operation should succeed");
        assert!((result.bands[0].avg_power_db - (-40.0)).abs() < f64::EPSILON);
        assert!((result.rms_level_db - (-40.0)).abs() < f64::EPSILON);
    }
}
