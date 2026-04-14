//! Pre-computed audio filter coefficients for the clarity enhancement pipeline.
//!
//! Instead of recomputing biquad filter coefficients on every audio frame,
//! this module pre-computes them for a set of standard sample rates and
//! band configurations, then caches the results for instant lookup.
//!
//! Supported filter types:
//!
//! - **Low-shelf** — boost low frequencies for bass enhancement
//! - **High-shelf** — boost high frequencies for sibilance / presence
//! - **Peaking EQ** — narrow boost/cut at a specific centre frequency
//! - **Low-pass** — remove high-frequency content above a cutoff
//! - **High-pass** — remove low-frequency content below a cutoff
//! - **Band-pass** — isolate a frequency band (speech enhancement)
//!
//! Coefficients follow the standard biquad transfer function:
//!
//! ```text
//! H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
//! ```
//!
//! The [`BiquadCoefficients`] struct stores `(b0, b1, b2, a1, a2)`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Biquad coefficient struct ────────────────────────────────────────────────

/// Normalised biquad filter coefficients.
///
/// Denominator coefficient `a0` is implicitly 1.0 (coefficients are normalised).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BiquadCoefficients {
    /// Feed-forward coefficient b0.
    pub b0: f64,
    /// Feed-forward coefficient b1.
    pub b1: f64,
    /// Feed-forward coefficient b2.
    pub b2: f64,
    /// Feedback coefficient a1 (sign convention: denominator is `1 + a1·z⁻¹ + a2·z⁻²`).
    pub a1: f64,
    /// Feedback coefficient a2.
    pub a2: f64,
}

impl BiquadCoefficients {
    /// All-pass identity (no filtering).
    pub const IDENTITY: Self = Self {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };

    /// Apply these coefficients to a single sample using Direct Form II Transposed.
    ///
    /// `state` must have at least two elements `[s1, s2]` and is mutated in-place.
    pub fn process_sample(&self, input: f64, state: &mut [f64; 2]) -> f64 {
        let output = self.b0 * input + state[0];
        state[0] = self.b1 * input - self.a1 * output + state[1];
        state[1] = self.b2 * input - self.a2 * output;
        output
    }

    /// Process an entire buffer of samples in-place.
    pub fn process_buffer(&self, buffer: &mut [f64], state: &mut [f64; 2]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample, state);
        }
    }

    /// Whether these are the identity (all-pass) coefficients.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (self.b0 - 1.0).abs() < 1e-12
            && self.b1.abs() < 1e-12
            && self.b2.abs() < 1e-12
            && self.a1.abs() < 1e-12
            && self.a2.abs() < 1e-12
    }
}

// ── Filter specification ─────────────────────────────────────────────────────

/// Type of biquad filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FilterType {
    /// Low-pass filter.
    LowPass,
    /// High-pass filter.
    HighPass,
    /// Band-pass filter (constant 0 dB peak gain).
    BandPass,
    /// Peaking EQ (parametric).
    PeakingEq,
    /// Low-shelf filter.
    LowShelf,
    /// High-shelf filter.
    HighShelf,
}

/// A complete filter specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterSpec {
    /// Type of filter.
    pub filter_type: FilterType,
    /// Centre / cutoff frequency in Hz.
    pub frequency_hz: f64,
    /// Quality factor (Q).  Typical range: 0.1 – 20.
    pub q: f64,
    /// Gain in dB (only used by PeakingEq, LowShelf, HighShelf).
    pub gain_db: f64,
}

impl FilterSpec {
    /// Create a band-pass spec for speech enhancement.
    #[must_use]
    pub fn speech_bandpass(center_hz: f64, q: f64) -> Self {
        Self {
            filter_type: FilterType::BandPass,
            frequency_hz: center_hz,
            q,
            gain_db: 0.0,
        }
    }

    /// Create a peaking EQ spec.
    #[must_use]
    pub fn peaking_eq(center_hz: f64, q: f64, gain_db: f64) -> Self {
        Self {
            filter_type: FilterType::PeakingEq,
            frequency_hz: center_hz,
            q,
            gain_db,
        }
    }

    /// Create a low-pass spec.
    #[must_use]
    pub fn low_pass(cutoff_hz: f64, q: f64) -> Self {
        Self {
            filter_type: FilterType::LowPass,
            frequency_hz: cutoff_hz,
            q,
            gain_db: 0.0,
        }
    }

    /// Create a high-pass spec.
    #[must_use]
    pub fn high_pass(cutoff_hz: f64, q: f64) -> Self {
        Self {
            filter_type: FilterType::HighPass,
            frequency_hz: cutoff_hz,
            q,
            gain_db: 0.0,
        }
    }

    /// Create a low-shelf spec.
    #[must_use]
    pub fn low_shelf(frequency_hz: f64, gain_db: f64) -> Self {
        Self {
            filter_type: FilterType::LowShelf,
            frequency_hz,
            q: 0.707,
            gain_db,
        }
    }

    /// Create a high-shelf spec.
    #[must_use]
    pub fn high_shelf(frequency_hz: f64, gain_db: f64) -> Self {
        Self {
            filter_type: FilterType::HighShelf,
            frequency_hz,
            q: 0.707,
            gain_db,
        }
    }
}

// ── Coefficient computation ──────────────────────────────────────────────────

/// Compute biquad coefficients from a [`FilterSpec`] and sample rate.
///
/// Uses the Audio EQ Cookbook formulas (Robert Bristow-Johnson).
#[must_use]
pub fn compute_coefficients(spec: &FilterSpec, sample_rate: u32) -> BiquadCoefficients {
    let fs = f64::from(sample_rate);
    let w0 = 2.0 * std::f64::consts::PI * spec.frequency_hz / fs;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * spec.q);

    match spec.filter_type {
        FilterType::LowPass => {
            let b1 = 1.0 - cos_w0;
            let b0 = b1 / 2.0;
            let b2 = b0;
            let a0 = 1.0 + alpha;
            let a1 = -2.0 * cos_w0;
            let a2 = 1.0 - alpha;
            normalise(b0, b1, b2, a0, a1, a2)
        }
        FilterType::HighPass => {
            let b1 = -(1.0 + cos_w0);
            let b0 = (1.0 + cos_w0) / 2.0;
            let b2 = b0;
            let a0 = 1.0 + alpha;
            let a1 = -2.0 * cos_w0;
            let a2 = 1.0 - alpha;
            normalise(b0, b1, b2, a0, a1, a2)
        }
        FilterType::BandPass => {
            let b0 = alpha;
            let b1 = 0.0;
            let b2 = -alpha;
            let a0 = 1.0 + alpha;
            let a1 = -2.0 * cos_w0;
            let a2 = 1.0 - alpha;
            normalise(b0, b1, b2, a0, a1, a2)
        }
        FilterType::PeakingEq => {
            let a_lin = 10.0_f64.powf(spec.gain_db / 40.0);
            let b0 = 1.0 + alpha * a_lin;
            let b1 = -2.0 * cos_w0;
            let b2 = 1.0 - alpha * a_lin;
            let a0 = 1.0 + alpha / a_lin;
            let a1 = -2.0 * cos_w0;
            let a2 = 1.0 - alpha / a_lin;
            normalise(b0, b1, b2, a0, a1, a2)
        }
        FilterType::LowShelf => {
            let a_lin = 10.0_f64.powf(spec.gain_db / 40.0);
            let sqrt_a = a_lin.sqrt();
            let b0 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
            let b1 = 2.0 * a_lin * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
            let b2 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
            let a0 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
            let a1 = -2.0 * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
            let a2 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
            normalise(b0, b1, b2, a0, a1, a2)
        }
        FilterType::HighShelf => {
            let a_lin = 10.0_f64.powf(spec.gain_db / 40.0);
            let sqrt_a = a_lin.sqrt();
            let b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
            let b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
            let b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
            let a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
            let a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
            let a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
            normalise(b0, b1, b2, a0, a1, a2)
        }
    }
}

/// Normalise coefficients by dividing by `a0`.
fn normalise(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> BiquadCoefficients {
    BiquadCoefficients {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

// ── Coefficient cache ────────────────────────────────────────────────────────

/// Cache key for pre-computed coefficients.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    filter_type: FilterType,
    // Frequencies and Q are stored as fixed-point millis to allow Eq/Hash.
    freq_mhz: u64,
    q_milli: u64,
    gain_milli_db: i64,
    sample_rate: u32,
}

impl CacheKey {
    fn from_spec(spec: &FilterSpec, sample_rate: u32) -> Self {
        Self {
            filter_type: spec.filter_type,
            freq_mhz: (spec.frequency_hz * 1000.0) as u64,
            q_milli: (spec.q * 1000.0) as u64,
            gain_milli_db: (spec.gain_db * 1000.0) as i64,
            sample_rate,
        }
    }
}

/// A cache of pre-computed biquad filter coefficients.
///
/// Call [`PrecomputedFilterCache::get_or_compute`] to retrieve coefficients;
/// they are computed once and then served from the cache on subsequent lookups.
pub struct PrecomputedFilterCache {
    cache: HashMap<CacheKey, BiquadCoefficients>,
}

impl PrecomputedFilterCache {
    /// Create an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Pre-populate the cache with a standard set of speech-clarity filters
    /// for the given sample rate.
    ///
    /// This populates: 200 Hz HP, 300 Hz HP, 1 kHz peaking (+3 dB),
    /// 2.5 kHz peaking (+4 dB), 5 kHz peaking (+2 dB), 8 kHz LP.
    pub fn populate_speech_clarity(&mut self, sample_rate: u32) {
        let specs = [
            FilterSpec::high_pass(200.0, 0.707),
            FilterSpec::high_pass(300.0, 0.707),
            FilterSpec::peaking_eq(1000.0, 1.5, 3.0),
            FilterSpec::peaking_eq(2500.0, 1.5, 4.0),
            FilterSpec::peaking_eq(5000.0, 1.0, 2.0),
            FilterSpec::low_pass(8000.0, 0.707),
        ];
        for spec in &specs {
            self.get_or_compute(spec, sample_rate);
        }
    }

    /// Retrieve cached coefficients or compute and cache them.
    pub fn get_or_compute(
        &mut self,
        spec: &FilterSpec,
        sample_rate: u32,
    ) -> BiquadCoefficients {
        let key = CacheKey::from_spec(spec, sample_rate);
        *self
            .cache
            .entry(key)
            .or_insert_with(|| compute_coefficients(spec, sample_rate))
    }

    /// Check whether a spec/sample-rate combination is already cached.
    #[must_use]
    pub fn contains(&self, spec: &FilterSpec, sample_rate: u32) -> bool {
        let key = CacheKey::from_spec(spec, sample_rate);
        self.cache.contains_key(&key)
    }

    /// Number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Pre-compute coefficients for every combination of specs × sample rates.
    pub fn populate_matrix(&mut self, specs: &[FilterSpec], sample_rates: &[u32]) {
        for spec in specs {
            for &sr in sample_rates {
                self.get_or_compute(spec, sr);
            }
        }
    }
}

impl Default for PrecomputedFilterCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── Chain of filters ─────────────────────────────────────────────────────────

/// A chain of biquad filters applied in series.
///
/// Maintains per-filter state so a chain can process streaming audio.
pub struct FilterChain {
    stages: Vec<(BiquadCoefficients, [f64; 2])>,
}

impl FilterChain {
    /// Create from a list of pre-computed coefficients.
    #[must_use]
    pub fn new(coefficients: Vec<BiquadCoefficients>) -> Self {
        let stages = coefficients.into_iter().map(|c| (c, [0.0; 2])).collect();
        Self { stages }
    }

    /// Process a single sample through the chain.
    pub fn process_sample(&mut self, mut sample: f64) -> f64 {
        for (coeff, state) in &mut self.stages {
            sample = coeff.process_sample(sample, state);
        }
        sample
    }

    /// Process an entire buffer in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset all filter states to zero.
    pub fn reset(&mut self) {
        for (_, state) in &mut self.stages {
            *state = [0.0; 2];
        }
    }

    /// Number of filter stages.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_coefficients() {
        let id = BiquadCoefficients::IDENTITY;
        assert!(id.is_identity());
        let mut state = [0.0; 2];
        let out = id.process_sample(0.5, &mut state);
        assert!((out - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_lowpass_coefficients_stable() {
        let spec = FilterSpec::low_pass(1000.0, 0.707);
        let coeff = compute_coefficients(&spec, 48000);
        // b0, b1, b2 should be positive for a low-pass
        assert!(coeff.b0 > 0.0);
        assert!(coeff.b1 > 0.0);
        assert!(coeff.b2 > 0.0);
        // Stability check: |a2| < 1
        assert!(coeff.a2.abs() < 1.0);
    }

    #[test]
    fn test_highpass_attenuates_dc() {
        let spec = FilterSpec::high_pass(500.0, 0.707);
        let coeff = compute_coefficients(&spec, 48000);
        // DC gain: H(z=1) = (b0 + b1 + b2) / (1 + a1 + a2)
        let dc_gain = (coeff.b0 + coeff.b1 + coeff.b2) / (1.0 + coeff.a1 + coeff.a2);
        assert!(
            dc_gain.abs() < 0.01,
            "high-pass DC gain should be near zero, got {dc_gain}"
        );
    }

    #[test]
    fn test_bandpass_zero_dc() {
        let spec = FilterSpec::speech_bandpass(2000.0, 2.0);
        let coeff = compute_coefficients(&spec, 44100);
        let dc_gain = (coeff.b0 + coeff.b1 + coeff.b2) / (1.0 + coeff.a1 + coeff.a2);
        assert!(
            dc_gain.abs() < 0.01,
            "band-pass DC gain should be near zero, got {dc_gain}"
        );
    }

    #[test]
    fn test_peaking_eq_unity_at_zero_gain() {
        let spec = FilterSpec::peaking_eq(1000.0, 1.0, 0.0);
        let coeff = compute_coefficients(&spec, 48000);
        // At 0 dB gain, peaking EQ should be identity-like
        let dc_gain = (coeff.b0 + coeff.b1 + coeff.b2) / (1.0 + coeff.a1 + coeff.a2);
        assert!(
            (dc_gain - 1.0).abs() < 0.01,
            "peaking EQ at 0 dB should have unity DC gain, got {dc_gain}"
        );
    }

    #[test]
    fn test_cache_miss_then_hit() {
        let mut cache = PrecomputedFilterCache::new();
        let spec = FilterSpec::low_pass(1000.0, 0.707);
        assert!(!cache.contains(&spec, 48000));
        let c1 = cache.get_or_compute(&spec, 48000);
        assert!(cache.contains(&spec, 48000));
        let c2 = cache.get_or_compute(&spec, 48000);
        assert_eq!(c1.b0, c2.b0);
    }

    #[test]
    fn test_cache_different_sample_rates() {
        let mut cache = PrecomputedFilterCache::new();
        let spec = FilterSpec::low_pass(1000.0, 0.707);
        let c48 = cache.get_or_compute(&spec, 48000);
        let c44 = cache.get_or_compute(&spec, 44100);
        // Coefficients should differ for different sample rates
        assert!((c48.b0 - c44.b0).abs() > 1e-6);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_populate_speech_clarity() {
        let mut cache = PrecomputedFilterCache::new();
        cache.populate_speech_clarity(48000);
        assert_eq!(cache.len(), 6, "should pre-populate 6 speech-clarity filters");
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = PrecomputedFilterCache::new();
        cache.populate_speech_clarity(48000);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_populate_matrix() {
        let mut cache = PrecomputedFilterCache::new();
        let specs = vec![
            FilterSpec::low_pass(1000.0, 0.707),
            FilterSpec::high_pass(200.0, 0.707),
        ];
        let rates = [44100, 48000, 96000];
        cache.populate_matrix(&specs, &rates);
        assert_eq!(cache.len(), 6); // 2 specs × 3 rates
    }

    #[test]
    fn test_filter_chain_passthrough() {
        let mut chain = FilterChain::new(vec![BiquadCoefficients::IDENTITY]);
        let result = chain.process_sample(0.75);
        assert!((result - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_filter_chain_multiple_stages() {
        let spec = FilterSpec::low_pass(4000.0, 0.707);
        let coeff = compute_coefficients(&spec, 48000);
        let mut chain = FilterChain::new(vec![coeff, coeff]);
        assert_eq!(chain.stage_count(), 2);
        // Process some samples — just verify it does not panic
        for i in 0..100 {
            let _ = chain.process_sample((i as f64 * 0.1).sin());
        }
    }

    #[test]
    fn test_filter_chain_reset() {
        let spec = FilterSpec::low_pass(1000.0, 0.707);
        let coeff = compute_coefficients(&spec, 48000);
        let mut chain = FilterChain::new(vec![coeff]);
        // Feed some signal
        for _ in 0..50 {
            chain.process_sample(1.0);
        }
        chain.reset();
        // After reset, state should be zero — first sample should behave fresh
        let out = chain.process_sample(0.0);
        assert!(
            out.abs() < 1e-10,
            "after reset, zero input should give zero output"
        );
    }

    #[test]
    fn test_process_buffer() {
        let spec = FilterSpec::low_pass(2000.0, 0.707);
        let coeff = compute_coefficients(&spec, 48000);
        let mut state = [0.0; 2];
        let mut buf = vec![1.0; 10];
        coeff.process_buffer(&mut buf, &mut state);
        // Low-pass with DC input (1.0) should converge towards 1.0
        // Last sample should be closer to 1.0 than first
        assert!(buf[9] > buf[0]);
    }

    #[test]
    fn test_low_shelf_boost() {
        let spec = FilterSpec::low_shelf(200.0, 6.0);
        let coeff = compute_coefficients(&spec, 48000);
        // DC gain should be boosted (> 1.0)
        let dc_gain = (coeff.b0 + coeff.b1 + coeff.b2) / (1.0 + coeff.a1 + coeff.a2);
        assert!(
            dc_gain > 1.0,
            "low-shelf +6 dB should boost DC, got {dc_gain}"
        );
    }

    #[test]
    fn test_high_shelf_boost() {
        let spec = FilterSpec::high_shelf(4000.0, 6.0);
        let coeff = compute_coefficients(&spec, 48000);
        // Nyquist gain: H(z=-1) = (b0 - b1 + b2) / (1 - a1 + a2)
        let nyquist_gain = (coeff.b0 - coeff.b1 + coeff.b2) / (1.0 - coeff.a1 + coeff.a2);
        assert!(
            nyquist_gain > 1.0,
            "high-shelf +6 dB should boost near Nyquist, got {nyquist_gain}"
        );
    }
}
