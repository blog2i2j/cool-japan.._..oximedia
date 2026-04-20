#![allow(dead_code)]
//! Noise profiling for audio restoration and de-noising.
//!
//! A [`NoiseProfiler`] analyses quiet sections of audio to build a
//! [`NoiseProfile`] that characterises the background noise floor.  The
//! profile can then be used by a downstream de-noising stage to subtract
//! noise-shaped spectral content from the audio signal.

// ─────────────────────────────────────────────────────────────────────────────
// NoiseType
// ─────────────────────────────────────────────────────────────────────────────

/// Noise classification taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoiseType {
    /// Broadband white noise (flat spectral density).
    White,
    /// Pink (1/f) noise — equal energy per octave.
    Pink,
    /// Mains hum at 50 Hz or 60 Hz (and harmonics).
    HumAc,
    /// Thermal noise floor of analogue electronics.
    Thermal,
    /// Tape hiss from analogue recording media.
    TapeHiss,
    /// HVAC / air-conditioning background rumble.
    Hvac,
    /// Impulse-type noise (clicks, pops, crackle).
    Impulse,
    /// Unknown or unclassified noise.
    Unknown,
}

impl NoiseType {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::White => "white noise",
            Self::Pink => "pink noise (1/f)",
            Self::HumAc => "AC mains hum",
            Self::Thermal => "thermal noise",
            Self::TapeHiss => "tape hiss",
            Self::Hvac => "HVAC/air-conditioning",
            Self::Impulse => "impulse noise",
            Self::Unknown => "unknown",
        }
    }

    /// Returns `true` for noise types that are primarily tonal (single or
    /// harmonic frequencies) rather than broadband.
    pub fn is_tonal(self) -> bool {
        matches!(self, Self::HumAc)
    }

    /// Returns `true` for broadband noise types.
    pub fn is_broadband(self) -> bool {
        matches!(
            self,
            Self::White | Self::Pink | Self::Thermal | Self::TapeHiss | Self::Hvac
        )
    }
}

impl std::fmt::Display for NoiseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NoiseProfile
// ─────────────────────────────────────────────────────────────────────────────

/// Characterisation of the background noise in an audio signal.
///
/// The profile stores per-band RMS energy (in dB FS) across a configurable
/// number of frequency bins, which a de-noising algorithm can use for
/// spectral subtraction.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Classified noise type.
    pub noise_type: NoiseType,
    /// Integrated noise floor in dB FS (full-scale).
    pub floor_dbfs: f64,
    /// Per-band noise energy in dB FS (one value per frequency bin).
    pub spectral_shape: Vec<f64>,
    /// Sample rate at which the profile was captured.
    pub sample_rate: f64,
    /// Duration of audio used to build the profile, in seconds.
    pub capture_duration_secs: f64,
    /// Signal-to-noise ratio estimate in dB.
    pub snr_db: f64,
}

impl NoiseProfile {
    /// Create a flat (white-noise) profile at the given noise floor.
    pub fn flat(floor_dbfs: f64, sample_rate: f64, num_bands: usize) -> Self {
        Self {
            noise_type: NoiseType::White,
            floor_dbfs,
            spectral_shape: vec![floor_dbfs; num_bands],
            sample_rate,
            capture_duration_secs: 0.0,
            snr_db: 0.0,
        }
    }

    /// Returns `true` if the noise floor is below the given threshold (in
    /// dB FS).  A lower (more negative) floor means quieter noise.
    pub fn is_below_threshold(&self, threshold_dbfs: f64) -> bool {
        self.floor_dbfs < threshold_dbfs
    }

    /// Peak noise energy across all spectral bands.
    pub fn peak_band_dbfs(&self) -> f64 {
        self.spectral_shape
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Number of frequency bands in the spectral shape.
    pub fn num_bands(&self) -> usize {
        self.spectral_shape.len()
    }

    /// Estimate the dominant noise type from the spectral shape.
    ///
    /// This heuristic checks the slope of the spectral envelope:
    /// - Flat → [`NoiseType::White`]
    /// - Decreasing (high → low frequency) → [`NoiseType::Pink`]
    /// - Otherwise → [`NoiseType::Unknown`]
    pub fn dominant_type(&self) -> NoiseType {
        if self.spectral_shape.len() < 2 {
            return self.noise_type;
        }
        let first = self.spectral_shape[0];
        // SAFETY: len >= 2 is guaranteed by the early-return guard above
        let last = self.spectral_shape[self.spectral_shape.len() - 1];
        let slope = last - first;
        if slope.abs() < 2.0 {
            NoiseType::White
        } else if slope < 0.0 {
            NoiseType::Pink
        } else {
            NoiseType::Unknown
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NoiseProfiler
// ─────────────────────────────────────────────────────────────────────────────

/// Analyses quiet audio sections to build a [`NoiseProfile`].
///
/// ```rust
/// use oximedia_normalize::noise_profile::{NoiseProfiler, NoiseType};
///
/// let profiler = NoiseProfiler::new(48_000.0, 32);
/// // Silence-only buffer ≈ noise floor of −96 dB FS.
/// let samples: Vec<f32> = vec![0.0; 4800];
/// let profile = profiler.analyze(&samples);
/// assert!(profile.floor_dbfs < -90.0);
/// ```
#[derive(Debug, Clone)]
pub struct NoiseProfiler {
    /// Audio sample rate.
    pub sample_rate: f64,
    /// Number of spectral bands in the output profile.
    pub num_bands: usize,
    /// Gate threshold: only process frames below this RMS level (dB FS).
    pub gate_threshold_dbfs: f64,
}

impl NoiseProfiler {
    /// Create a new profiler.
    pub fn new(sample_rate: f64, num_bands: usize) -> Self {
        Self {
            sample_rate,
            num_bands,
            gate_threshold_dbfs: -40.0,
        }
    }

    /// Set the gate threshold in dB FS (frames louder than this are skipped).
    pub fn with_gate_threshold(mut self, threshold_dbfs: f64) -> Self {
        self.gate_threshold_dbfs = threshold_dbfs;
        self
    }

    /// Analyse `samples` and return a [`NoiseProfile`].
    ///
    /// The implementation uses a simple frame-gated RMS approach: frames whose
    /// RMS exceeds the gate threshold are excluded, and the mean energy across
    /// the remaining frames is stored as the noise floor.  Spectral shape is
    /// approximated by splitting the signal into `num_bands` equal-width
    /// frequency buckets and computing per-bucket RMS.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, samples: &[f32]) -> NoiseProfile {
        let frame_size = (self.sample_rate * 0.025) as usize; // 25 ms frames
        let frame_size = frame_size.max(1);

        let gate_linear = 10_f64.powf(self.gate_threshold_dbfs / 20.0) as f32;

        // Collect gated frames.
        let mut gated_frames: Vec<&[f32]> = Vec::new();
        for chunk in samples.chunks(frame_size) {
            let rms = Self::rms(chunk);
            if rms <= gate_linear {
                gated_frames.push(chunk);
            }
        }

        // Compute overall noise floor from gated frames.
        let floor_dbfs = if gated_frames.is_empty() {
            // No quiet sections — use the overall RMS.
            let rms = Self::rms(samples);
            Self::to_dbfs(f64::from(rms))
        } else {
            let all_gated: Vec<f32> = gated_frames
                .iter()
                .flat_map(|f| f.iter().copied())
                .collect();
            let rms = Self::rms(&all_gated);
            Self::to_dbfs(f64::from(rms))
        };

        // Approximate spectral shape by splitting into equal-width buckets.
        let spectral_shape = self.estimate_spectral_shape(samples);

        // Approximate SNR: signal RMS vs. noise floor.
        let signal_dbfs = Self::to_dbfs(f64::from(Self::rms(samples)));
        let snr_db = (signal_dbfs - floor_dbfs).max(0.0);

        // Heuristic noise-type detection.
        let noise_type = Self::classify(floor_dbfs, &spectral_shape);

        let duration_secs = samples.len() as f64 / self.sample_rate;

        NoiseProfile {
            noise_type,
            floor_dbfs,
            spectral_shape,
            sample_rate: self.sample_rate,
            capture_duration_secs: duration_secs,
            snr_db,
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }

    fn to_dbfs(linear: f64) -> f64 {
        if linear <= 0.0 {
            return -200.0;
        }
        20.0 * linear.log10()
    }

    #[allow(clippy::cast_precision_loss)]
    fn estimate_spectral_shape(&self, samples: &[f32]) -> Vec<f64> {
        if samples.is_empty() || self.num_bands == 0 {
            return vec![-200.0; self.num_bands];
        }
        let band_size = (samples.len() / self.num_bands).max(1);
        (0..self.num_bands)
            .map(|b| {
                let start = b * band_size;
                let end = ((b + 1) * band_size).min(samples.len());
                if start >= end {
                    return -200.0;
                }
                let rms = Self::rms(&samples[start..end]);
                Self::to_dbfs(f64::from(rms))
            })
            .collect()
    }

    fn classify(floor_dbfs: f64, spectral_shape: &[f64]) -> NoiseType {
        if spectral_shape.len() < 2 {
            return NoiseType::Unknown;
        }
        let first = spectral_shape[0];
        // SAFETY: len >= 2 is guaranteed by the early-return guard above
        let last = spectral_shape[spectral_shape.len() - 1];
        let slope = last - first;

        if floor_dbfs < -80.0 {
            // Essentially silence — thermal / unknown.
            return NoiseType::Thermal;
        }
        if slope.abs() < 3.0 {
            NoiseType::White
        } else if slope < 0.0 {
            NoiseType::Pink
        } else {
            NoiseType::Unknown
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profiler() -> NoiseProfiler {
        NoiseProfiler::new(48_000.0, 16)
    }

    #[test]
    fn test_noise_type_label() {
        assert_eq!(NoiseType::White.label(), "white noise");
        assert_eq!(NoiseType::HumAc.label(), "AC mains hum");
    }

    #[test]
    fn test_noise_type_display() {
        assert_eq!(NoiseType::Pink.to_string(), "pink noise (1/f)");
    }

    #[test]
    fn test_noise_type_is_tonal() {
        assert!(NoiseType::HumAc.is_tonal());
        assert!(!NoiseType::White.is_tonal());
    }

    #[test]
    fn test_noise_type_is_broadband() {
        assert!(NoiseType::White.is_broadband());
        assert!(NoiseType::TapeHiss.is_broadband());
        assert!(!NoiseType::Impulse.is_broadband());
    }

    #[test]
    fn test_flat_profile_creation() {
        let profile = NoiseProfile::flat(-60.0, 48_000.0, 8);
        assert_eq!(profile.num_bands(), 8);
        assert!((profile.floor_dbfs - (-60.0)).abs() < 1e-9);
        assert!(matches!(profile.noise_type, NoiseType::White));
    }

    #[test]
    fn test_profile_is_below_threshold() {
        let profile = NoiseProfile::flat(-60.0, 48_000.0, 8);
        assert!(profile.is_below_threshold(-50.0));
        assert!(!profile.is_below_threshold(-70.0));
    }

    #[test]
    fn test_profile_peak_band() {
        let mut profile = NoiseProfile::flat(-60.0, 48_000.0, 4);
        profile.spectral_shape[2] = -40.0;
        assert!((profile.peak_band_dbfs() - (-40.0)).abs() < 1e-9);
    }

    #[test]
    fn test_profile_dominant_type_flat() {
        let profile = NoiseProfile::flat(-60.0, 48_000.0, 4);
        assert_eq!(profile.dominant_type(), NoiseType::White);
    }

    #[test]
    fn test_profile_dominant_type_pink() {
        let mut profile = NoiseProfile::flat(-60.0, 48_000.0, 4);
        // Decreasing shape → pink
        profile.spectral_shape = vec![-55.0, -58.0, -62.0, -65.0];
        assert_eq!(profile.dominant_type(), NoiseType::Pink);
    }

    #[test]
    fn test_analyze_silence_gives_low_floor() {
        let profiler = make_profiler();
        let silence: Vec<f32> = vec![0.0; 4800];
        let profile = profiler.analyze(&silence);
        assert!(
            profile.floor_dbfs < -90.0,
            "floor was {:.1} dB FS",
            profile.floor_dbfs
        );
    }

    #[test]
    fn test_analyze_full_scale_gives_high_floor() {
        let profiler = make_profiler();
        let signal: Vec<f32> = vec![1.0; 4800];
        let profile = profiler.analyze(&signal);
        assert!(
            profile.floor_dbfs > -10.0,
            "floor was {:.1}",
            profile.floor_dbfs
        );
    }

    #[test]
    fn test_analyze_returns_correct_num_bands() {
        let profiler = NoiseProfiler::new(48_000.0, 8);
        let samples: Vec<f32> = vec![0.01; 4800];
        let profile = profiler.analyze(&samples);
        assert_eq!(profile.num_bands(), 8);
    }

    #[test]
    fn test_analyze_sample_rate_stored() {
        let profiler = make_profiler();
        let samples: Vec<f32> = vec![0.0; 480];
        let profile = profiler.analyze(&samples);
        assert!((profile.sample_rate - 48_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_analyze_duration_correct() {
        let profiler = NoiseProfiler::new(48_000.0, 4);
        let samples: Vec<f32> = vec![0.0; 48_000]; // 1 second
        let profile = profiler.analyze(&samples);
        assert!((profile.capture_duration_secs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_with_gate_threshold() {
        let profiler = NoiseProfiler::new(48_000.0, 8).with_gate_threshold(-30.0);
        assert!((profiler.gate_threshold_dbfs - (-30.0)).abs() < 1e-9);
    }

    #[test]
    fn test_snr_nonnegative_for_silence() {
        let profiler = make_profiler();
        let silence: Vec<f32> = vec![0.0; 4800];
        let profile = profiler.analyze(&silence);
        assert!(profile.snr_db >= 0.0);
    }
}
