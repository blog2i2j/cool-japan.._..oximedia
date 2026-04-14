//! Audio-reactive effect modifier for driving VFX parameters from audio data.
//!
//! [`AudioReactiveModifier`] maps audio amplitude and frequency-band energy to
//! named parameter values, allowing effects to pulse, scale, or shift in sync
//! with a soundtrack.
//!
//! # Example
//!
//! ```
//! use oximedia_vfx::audio_reactive::{AudioAmplitudeData, AudioReactiveModifier, FrequencyBand};
//!
//! let mut data = AudioAmplitudeData::new(44100, 512);
//! data.set_band_energy(FrequencyBand::Bass, 0.8);
//! data.set_band_energy(FrequencyBand::Mid, 0.3);
//!
//! let mut modifier = AudioReactiveModifier::builder()
//!     .map_band_to_param(FrequencyBand::Bass, "intensity", 0.0, 1.0)
//!     .map_band_to_param(FrequencyBand::Mid, "blur_radius", 0.0, 20.0)
//!     .build();
//!
//! let params = modifier.evaluate(&data);
//! let intensity = params.get("intensity").copied().unwrap_or(0.0);
//! assert!(intensity >= 0.0 && intensity <= 1.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// FrequencyBand
// ─────────────────────────────────────────────────────────────────────────────

/// Standard frequency band classification for audio-reactive effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrequencyBand {
    /// Sub-bass: 20 – 60 Hz. Deep rumble, kick drum fundamentals.
    SubBass,
    /// Bass: 60 – 250 Hz. Warmth, rhythm, bass guitar.
    Bass,
    /// Low-mid: 250 – 500 Hz. Muddiness or fullness region.
    LowMid,
    /// Mid: 500 Hz – 2 kHz. Primary voice/instrument body.
    Mid,
    /// Upper-mid: 2 – 4 kHz. Presence, attack.
    UpperMid,
    /// Presence: 4 – 6 kHz. Brightness, sibilance onset.
    Presence,
    /// Brilliance: 6 – 20 kHz. Air, shimmer, high harmonics.
    Brilliance,
    /// Full-spectrum RMS: aggregate energy across all bands.
    FullSpectrum,
}

impl FrequencyBand {
    /// Returns the approximate frequency range `(low_hz, high_hz)` for this band.
    #[must_use]
    pub const fn frequency_range(&self) -> (f32, f32) {
        match self {
            Self::SubBass => (20.0, 60.0),
            Self::Bass => (60.0, 250.0),
            Self::LowMid => (250.0, 500.0),
            Self::Mid => (500.0, 2_000.0),
            Self::UpperMid => (2_000.0, 4_000.0),
            Self::Presence => (4_000.0, 6_000.0),
            Self::Brilliance => (6_000.0, 20_000.0),
            Self::FullSpectrum => (20.0, 20_000.0),
        }
    }

    /// Ordered slice of all discrete frequency bands (excluding `FullSpectrum`).
    #[must_use]
    pub fn all_bands() -> &'static [FrequencyBand] {
        &[
            FrequencyBand::SubBass,
            FrequencyBand::Bass,
            FrequencyBand::LowMid,
            FrequencyBand::Mid,
            FrequencyBand::UpperMid,
            FrequencyBand::Presence,
            FrequencyBand::Brilliance,
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AudioAmplitudeData
// ─────────────────────────────────────────────────────────────────────────────

/// Per-frame audio amplitude and frequency-band energy snapshot.
///
/// All energy values are normalised to `[0.0, 1.0]` where `0.0` is silence
/// and `1.0` is full-scale.  Values outside this range are clamped on input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioAmplitudeData {
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio samples represented by this snapshot (FFT window size).
    pub window_size: u32,
    /// RMS amplitude over the full window, `[0.0, 1.0]`.
    pub rms: f32,
    /// Peak amplitude over the full window, `[0.0, 1.0]`.
    pub peak: f32,
    /// Per-band energy values, `[0.0, 1.0]`.
    band_energy: HashMap<FrequencyBandKey, f32>,
    /// Optional raw per-bin magnitude spectrum (linear scale, length = window_size/2+1).
    pub spectrum: Option<Vec<f32>>,
}

/// A `Copy`-friendly key type for frequency bands (HashMap does not require Eq on band directly).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct FrequencyBandKey(u8);

impl From<FrequencyBand> for FrequencyBandKey {
    fn from(b: FrequencyBand) -> Self {
        let n: u8 = match b {
            FrequencyBand::SubBass => 0,
            FrequencyBand::Bass => 1,
            FrequencyBand::LowMid => 2,
            FrequencyBand::Mid => 3,
            FrequencyBand::UpperMid => 4,
            FrequencyBand::Presence => 5,
            FrequencyBand::Brilliance => 6,
            FrequencyBand::FullSpectrum => 7,
        };
        FrequencyBandKey(n)
    }
}

impl AudioAmplitudeData {
    /// Create a new snapshot with default zero energy.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Audio sample rate in Hz (e.g. 44100).
    /// * `window_size` - FFT window / analysis frame size in samples.
    #[must_use]
    pub fn new(sample_rate: u32, window_size: u32) -> Self {
        Self {
            sample_rate,
            window_size,
            rms: 0.0,
            peak: 0.0,
            band_energy: HashMap::new(),
            spectrum: None,
        }
    }

    /// Set the energy for a frequency band (clamped to `[0.0, 1.0]`).
    pub fn set_band_energy(&mut self, band: FrequencyBand, energy: f32) {
        self.band_energy
            .insert(FrequencyBandKey::from(band), energy.clamp(0.0, 1.0));
    }

    /// Get the energy for a frequency band, or `0.0` if not set.
    #[must_use]
    pub fn band_energy(&self, band: FrequencyBand) -> f32 {
        self.band_energy
            .get(&FrequencyBandKey::from(band))
            .copied()
            .unwrap_or(0.0)
    }

    /// Set the RMS and peak from raw sample data (automatically normalised).
    ///
    /// `max_amplitude` is the full-scale value; for `i16` samples use 32768.0,
    /// for normalised `f32` in `[-1, 1]` use 1.0.
    pub fn compute_from_samples(&mut self, samples: &[f32], max_amplitude: f32) {
        if samples.is_empty() || max_amplitude <= 0.0 {
            self.rms = 0.0;
            self.peak = 0.0;
            return;
        }
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        let rms_raw = (sum_sq / samples.len() as f32).sqrt();
        let peak_raw: f32 = samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        self.rms = (rms_raw / max_amplitude).clamp(0.0, 1.0);
        self.peak = (peak_raw / max_amplitude).clamp(0.0, 1.0);
    }

    /// Compute band energies from a linear-scale magnitude spectrum (bin values ≥ 0).
    ///
    /// Bins are averaged within each band's frequency range.  Bins outside any
    /// defined band are ignored.
    pub fn compute_bands_from_spectrum(&mut self, spectrum: &[f32], sample_rate: u32) {
        let n_bins = spectrum.len();
        if n_bins == 0 || sample_rate == 0 {
            return;
        }
        let bin_hz = sample_rate as f32 / (2.0 * n_bins as f32);

        for &band in FrequencyBand::all_bands() {
            let (lo, hi) = band.frequency_range();
            let lo_bin = ((lo / bin_hz) as usize).min(n_bins.saturating_sub(1));
            let hi_bin = ((hi / bin_hz) as usize).min(n_bins);
            if lo_bin >= hi_bin {
                continue;
            }
            let slice = &spectrum[lo_bin..hi_bin];
            let avg: f32 = slice.iter().copied().sum::<f32>() / slice.len() as f32;
            self.set_band_energy(band, avg.clamp(0.0, 1.0));
        }
        // Full-spectrum is just overall RMS of the spectrum
        let rms: f32 = {
            let sum_sq: f32 = spectrum.iter().map(|&v| v * v).sum();
            (sum_sq / n_bins as f32).sqrt()
        };
        self.set_band_energy(FrequencyBand::FullSpectrum, rms.clamp(0.0, 1.0));
        self.spectrum = Some(spectrum.to_vec());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ParameterMapping
// ─────────────────────────────────────────────────────────────────────────────

/// How amplitude data is mapped to a parameter value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MappingSource {
    /// Energy from a specific frequency band.
    Band(FrequencyBand),
    /// Overall RMS amplitude.
    Rms,
    /// Peak amplitude.
    Peak,
}

/// A single mapping from an audio source to a named effect parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterMapping {
    /// Source audio measurement.
    pub source: MappingSource,
    /// Target parameter name (arbitrary string key).
    pub param_name: String,
    /// Output value when source is `0.0`.
    pub min_value: f32,
    /// Output value when source is `1.0`.
    pub max_value: f32,
    /// Optional smooth-following coefficient in `[0, 1)`.
    /// `0.0` = instant; `0.9` = slow follow.
    pub smoothing: f32,
    /// Whether to apply a power curve (gamma > 1 = compresses highs).
    pub response_gamma: f32,
}

impl ParameterMapping {
    /// Create a linear mapping with no smoothing.
    #[must_use]
    pub fn linear(source: MappingSource, param_name: impl Into<String>, min: f32, max: f32) -> Self {
        Self {
            source,
            param_name: param_name.into(),
            min_value: min,
            max_value: max,
            smoothing: 0.0,
            response_gamma: 1.0,
        }
    }

    /// Evaluate the raw (non-smoothed) output value for the given audio data.
    #[must_use]
    pub fn evaluate_raw(&self, data: &AudioAmplitudeData) -> f32 {
        let raw = match &self.source {
            MappingSource::Band(b) => data.band_energy(*b),
            MappingSource::Rms => data.rms,
            MappingSource::Peak => data.peak,
        };
        let curved = if self.response_gamma != 1.0 && raw > 0.0 {
            raw.powf(self.response_gamma)
        } else {
            raw
        };
        self.min_value + (self.max_value - self.min_value) * curved.clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AudioReactiveModifier
// ─────────────────────────────────────────────────────────────────────────────

/// Drives effect parameters from audio amplitude and frequency data.
///
/// Call [`evaluate`](Self::evaluate) each frame to obtain a `HashMap` of
/// parameter name → current float value.  The modifier maintains internal
/// smoothed state per parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioReactiveModifier {
    mappings: Vec<ParameterMapping>,
    /// Smoothed previous values, keyed by mapping index.
    #[serde(default)]
    smoothed: Vec<f32>,
}

impl AudioReactiveModifier {
    /// Create an empty modifier (no mappings).
    #[must_use]
    pub fn new() -> Self {
        Self {
            mappings: Vec::new(),
            smoothed: Vec::new(),
        }
    }

    /// Add a mapping. Returns `&mut Self` for chaining.
    pub fn add_mapping(&mut self, mapping: ParameterMapping) -> &mut Self {
        self.smoothed.push(0.0);
        self.mappings.push(mapping);
        self
    }

    /// Evaluate all mappings against the current audio frame.
    ///
    /// Returns a map of parameter name → current value with smoothing applied.
    /// Values from multiple mappings targeting the same parameter name are summed
    /// (allowing additive layering of audio sources).
    pub fn evaluate(&mut self, data: &AudioAmplitudeData) -> HashMap<String, f32> {
        let mut out: HashMap<String, f32> = HashMap::new();

        // Ensure smoothed vec matches mappings length
        while self.smoothed.len() < self.mappings.len() {
            self.smoothed.push(0.0);
        }

        for (i, mapping) in self.mappings.iter().enumerate() {
            let raw = mapping.evaluate_raw(data);
            let prev = self.smoothed.get(i).copied().unwrap_or(0.0);
            let s = mapping.smoothing.clamp(0.0, 0.999);
            let current = prev * s + raw * (1.0 - s);
            if let Some(slot) = self.smoothed.get_mut(i) {
                *slot = current;
            }
            *out.entry(mapping.param_name.clone()).or_insert(0.0) += current;
        }

        out
    }

    /// Start a fluent builder.
    #[must_use]
    pub fn builder() -> AudioReactiveBuilder {
        AudioReactiveBuilder::new()
    }
}

impl Default for AudioReactiveModifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AudioReactiveBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Fluent builder for [`AudioReactiveModifier`].
#[derive(Debug, Default)]
pub struct AudioReactiveBuilder {
    mappings: Vec<ParameterMapping>,
}

impl AudioReactiveBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mappings: Vec::new(),
        }
    }

    /// Map a frequency-band energy linearly to `param_name` in `[min, max]`.
    #[must_use]
    pub fn map_band_to_param(
        mut self,
        band: FrequencyBand,
        param_name: impl Into<String>,
        min: f32,
        max: f32,
    ) -> Self {
        self.mappings.push(ParameterMapping::linear(
            MappingSource::Band(band),
            param_name,
            min,
            max,
        ));
        self
    }

    /// Map RMS amplitude linearly to `param_name`.
    #[must_use]
    pub fn map_rms_to_param(
        mut self,
        param_name: impl Into<String>,
        min: f32,
        max: f32,
    ) -> Self {
        self.mappings.push(ParameterMapping::linear(
            MappingSource::Rms,
            param_name,
            min,
            max,
        ));
        self
    }

    /// Add a mapping with custom smoothing and gamma.
    #[must_use]
    pub fn map_custom(mut self, mapping: ParameterMapping) -> Self {
        self.mappings.push(mapping);
        self
    }

    /// Finalise and produce an [`AudioReactiveModifier`].
    #[must_use]
    pub fn build(self) -> AudioReactiveModifier {
        let n = self.mappings.len();
        AudioReactiveModifier {
            mappings: self.mappings,
            smoothed: vec![0.0; n],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_band_ranges_ordered() {
        let bands = FrequencyBand::all_bands();
        let mut prev_hi = 0.0f32;
        for &b in bands {
            let (lo, hi) = b.frequency_range();
            assert!(lo < hi, "band {b:?} low >= high");
            assert!(lo >= prev_hi || prev_hi == 0.0, "bands not ordered at {b:?}");
            prev_hi = lo; // allow overlap at boundaries
        }
    }

    #[test]
    fn test_amplitude_data_set_get_band_energy() {
        let mut d = AudioAmplitudeData::new(44100, 512);
        d.set_band_energy(FrequencyBand::Bass, 0.7);
        assert!((d.band_energy(FrequencyBand::Bass) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_amplitude_data_clamp_energy() {
        let mut d = AudioAmplitudeData::new(44100, 512);
        d.set_band_energy(FrequencyBand::Mid, 2.0); // should clamp to 1.0
        assert_eq!(d.band_energy(FrequencyBand::Mid), 1.0);
        d.set_band_energy(FrequencyBand::Mid, -0.5); // should clamp to 0.0
        assert_eq!(d.band_energy(FrequencyBand::Mid), 0.0);
    }

    #[test]
    fn test_amplitude_data_unset_band_returns_zero() {
        let d = AudioAmplitudeData::new(44100, 512);
        assert_eq!(d.band_energy(FrequencyBand::Brilliance), 0.0);
    }

    #[test]
    fn test_compute_from_samples_sine() {
        let mut d = AudioAmplitudeData::new(44100, 512);
        let samples: Vec<f32> = (0..512)
            .map(|i| (i as f32 * std::f32::consts::TAU / 512.0).sin())
            .collect();
        d.compute_from_samples(&samples, 1.0);
        // RMS of a full-period sine at amplitude 1.0 is 1/sqrt(2) ≈ 0.707
        assert!((d.rms - 0.707).abs() < 0.01, "rms = {}", d.rms);
        assert!((d.peak - 1.0).abs() < 0.01, "peak = {}", d.peak);
    }

    #[test]
    fn test_compute_from_samples_silence() {
        let mut d = AudioAmplitudeData::new(44100, 512);
        d.compute_from_samples(&vec![0.0; 512], 1.0);
        assert_eq!(d.rms, 0.0);
        assert_eq!(d.peak, 0.0);
    }

    #[test]
    fn test_compute_from_samples_empty() {
        let mut d = AudioAmplitudeData::new(44100, 512);
        d.compute_from_samples(&[], 1.0);
        assert_eq!(d.rms, 0.0);
    }

    #[test]
    fn test_parameter_mapping_linear_min_max() {
        let m = ParameterMapping::linear(MappingSource::Rms, "x", 5.0, 15.0);
        let mut d = AudioAmplitudeData::new(44100, 512);
        d.rms = 0.0;
        assert!((m.evaluate_raw(&d) - 5.0).abs() < 1e-5);
        d.rms = 1.0;
        assert!((m.evaluate_raw(&d) - 15.0).abs() < 1e-5);
        d.rms = 0.5;
        assert!((m.evaluate_raw(&d) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_audio_reactive_modifier_evaluate() {
        let mut modifier = AudioReactiveModifier::builder()
            .map_band_to_param(FrequencyBand::Bass, "intensity", 0.0, 1.0)
            .map_band_to_param(FrequencyBand::Mid, "blur_radius", 0.0, 20.0)
            .build();

        let mut data = AudioAmplitudeData::new(44100, 512);
        data.set_band_energy(FrequencyBand::Bass, 0.8);
        data.set_band_energy(FrequencyBand::Mid, 0.5);

        let params = modifier.evaluate(&data);
        let intensity = params.get("intensity").copied().unwrap_or(0.0);
        let blur = params.get("blur_radius").copied().unwrap_or(0.0);

        assert!(intensity >= 0.0 && intensity <= 1.0, "intensity={intensity}");
        assert!(blur >= 0.0 && blur <= 20.0, "blur={blur}");
    }

    #[test]
    fn test_audio_reactive_smoothing_damps_change() {
        let mut modifier = AudioReactiveModifier::builder()
            .map_custom(ParameterMapping {
                source: MappingSource::Rms,
                param_name: "x".into(),
                min_value: 0.0,
                max_value: 1.0,
                smoothing: 0.9,
                response_gamma: 1.0,
            })
            .build();

        let mut data = AudioAmplitudeData::new(44100, 512);
        data.rms = 1.0;

        let v1 = modifier.evaluate(&data).get("x").copied().unwrap_or(0.0);
        // With smoothing=0.9 starting from 0, first step = 0.0*0.9 + 1.0*0.1 = 0.1
        assert!((v1 - 0.1).abs() < 1e-5, "v1={v1}");

        let v2 = modifier.evaluate(&data).get("x").copied().unwrap_or(0.0);
        // second step = 0.1*0.9 + 1.0*0.1 = 0.19
        assert!((v2 - 0.19).abs() < 1e-5, "v2={v2}");
    }

    #[test]
    fn test_audio_reactive_modifier_default_empty() {
        let mut m = AudioReactiveModifier::default();
        let data = AudioAmplitudeData::new(44100, 512);
        let params = m.evaluate(&data);
        assert!(params.is_empty());
    }

    #[test]
    fn test_compute_bands_from_spectrum() {
        // Flat spectrum at 0.5
        let spectrum = vec![0.5f32; 256];
        let mut d = AudioAmplitudeData::new(44100, 512);
        d.compute_bands_from_spectrum(&spectrum, 44100);
        // Bass band should have non-zero energy
        assert!(d.band_energy(FrequencyBand::Bass) > 0.0);
    }
}
