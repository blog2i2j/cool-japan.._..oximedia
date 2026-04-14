//! Automatic format detection and appropriate normalization standard selection.
//!
//! This module analyses audio signal characteristics to infer the most appropriate
//! loudness normalization standard without requiring the caller to specify one
//! explicitly. The heuristic combines:
//!
//! - **Spectral content** — high-frequency energy ratio distinguishes music from speech.
//! - **Dynamic range** — crest factor and LRA proxy discriminate programme types.
//! - **Channel count** — surround beds suggest broadcast standards.
//! - **Sample-rate hints** — 48 kHz / 96 kHz are common in broadcast and cinema.
//! - **Duration** — short clips may be podcast chapters; long files may be feature films.
//!
//! ## Format categories
//!
//! | [`DetectedFormat`]        | Recommended standard | Notes                           |
//! |--------------------------|----------------------|---------------------------------|
//! | `BroadcastFile`          | EBU R128 / ATSC A85  | ITU BS.1770 loudness gate       |
//! | `StreamingMusic`         | Spotify / YouTube    | −14 LUFS target                 |
//! | `Podcast`                | Apple Podcasts −16   | Speech-dominant content         |
//! | `CinemaFeature`          | Dolby −27 LUFS       | Long-form, high dynamic range   |
//! | `MusicAlbum`             | Spotify −14 LUFS     | High crest factor, 44.1 kHz     |
//! | `Unknown`                | EBU R128 (fallback)  |                                 |
//!
//! ## Example
//!
//! ```rust
//! use oximedia_normalize::format_detect::{FormatDetector, DetectedFormat};
//! use oximedia_metering::Standard;
//!
//! let detector = FormatDetector::new(48000.0, 2);
//!
//! // Feed a block of audio samples
//! let samples: Vec<f32> = (0..48000)
//!     .map(|i| {
//!         let t = i as f32 / 48000.0;
//!         0.3_f32 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
//!     })
//!     .collect();
//!
//! let result = detector.detect(&samples);
//! let standard = result.recommended_standard();
//! println!("Detected: {:?}, Standard: {:?}", result.format, standard);
//! ```

use crate::NormalizeResult;
use oximedia_metering::Standard;

// ─── Detected format ─────────────────────────────────────────────────────────

/// The inferred content format of an audio file or stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectedFormat {
    /// Broadcast programme (news, documentary, TV drama).
    BroadcastFile,

    /// Streaming music (pop, rock, electronic — high crest, musical structure).
    StreamingMusic,

    /// Podcast or audio book (speech dominant, moderate dynamics).
    Podcast,

    /// Cinema feature or trailer (very high dynamic range, long duration cues).
    CinemaFeature,

    /// Music album master (full dynamic range, CD/hi-res sample rates).
    MusicAlbum,

    /// Unknown / ambiguous content.
    Unknown,
}

impl DetectedFormat {
    /// Returns the recommended [`Standard`] for this content type.
    pub fn recommended_standard(self) -> Standard {
        match self {
            Self::BroadcastFile => Standard::EbuR128,
            Self::StreamingMusic => Standard::Spotify,
            Self::Podcast => Standard::AppleMusic, // Apple Podcasts −16 LUFS
            Self::CinemaFeature => Standard::Netflix, // Netflix −27 LUFS drama
            Self::MusicAlbum => Standard::Spotify, // −14 LUFS for album streaming
            Self::Unknown => Standard::EbuR128,
        }
    }

    /// Returns `true` if dialogue / speech normalization is the primary concern.
    pub fn is_speech_dominant(self) -> bool {
        matches!(self, Self::Podcast | Self::BroadcastFile)
    }

    /// Returns a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::BroadcastFile => "Broadcast file",
            Self::StreamingMusic => "Streaming music",
            Self::Podcast => "Podcast / audiobook",
            Self::CinemaFeature => "Cinema feature",
            Self::MusicAlbum => "Music album",
            Self::Unknown => "Unknown",
        }
    }
}

// ─── Detection result ────────────────────────────────────────────────────────

/// Result of automatic format detection.
#[derive(Clone, Debug)]
pub struct DetectionResult {
    /// The inferred format category.
    pub format: DetectedFormat,

    /// Confidence in the detection on a 0–1 scale.
    pub confidence: f64,

    /// Measured crest factor of the analysed block (dB).
    pub crest_factor_db: f64,

    /// Ratio of high-frequency energy to total energy (0–1).
    pub hf_energy_ratio: f64,

    /// RMS level of the analysed block (linear, 0–1).
    pub rms_level: f64,

    /// Channel count supplied to the detector.
    pub channels: usize,

    /// Sample rate supplied to the detector (Hz).
    pub sample_rate: f64,
}

impl DetectionResult {
    /// Shorthand to obtain the recommended [`Standard`].
    pub fn recommended_standard(&self) -> Standard {
        self.format.recommended_standard()
    }
}

// ─── Detector configuration ──────────────────────────────────────────────────

/// Tuning parameters for the format detector.
///
/// The defaults work well for typical broadcast and music content. You can
/// override individual thresholds to suit domain-specific material.
#[derive(Clone, Debug)]
pub struct FormatDetectorConfig {
    /// Crest factor threshold (dB) above which content is classified as
    /// high-dynamic-range (music album or cinema).
    pub high_crest_threshold_db: f64,

    /// Crest factor threshold (dB) below which content is classified as
    /// speech-dominant (podcast or broadcast).
    pub speech_crest_threshold_db: f64,

    /// High-frequency energy ratio above which content is considered
    /// music (as opposed to voice-only).
    pub music_hf_ratio: f64,

    /// High-frequency energy ratio below which content is considered
    /// speech-only.
    pub speech_hf_ratio: f64,

    /// Duration (in samples at the detected sample rate) above which a
    /// high-dynamic-range file is classified as CinemaFeature.
    pub cinema_min_samples: usize,
}

impl Default for FormatDetectorConfig {
    fn default() -> Self {
        Self {
            high_crest_threshold_db: 18.0,
            speech_crest_threshold_db: 12.0,
            music_hf_ratio: 0.08,
            speech_hf_ratio: 0.04,
            cinema_min_samples: 48000 * 60 * 10, // 10 minutes @ 48 kHz
        }
    }
}

// ─── Detector ────────────────────────────────────────────────────────────────

/// Analyses audio samples and returns a [`DetectionResult`].
#[derive(Clone, Debug)]
pub struct FormatDetector {
    sample_rate: f64,
    channels: usize,
    config: FormatDetectorConfig,
}

impl FormatDetector {
    /// Create a detector with default configuration.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            config: FormatDetectorConfig::default(),
        }
    }

    /// Create a detector with custom configuration.
    pub fn with_config(sample_rate: f64, channels: usize, config: FormatDetectorConfig) -> Self {
        Self {
            sample_rate,
            channels,
            config,
        }
    }

    /// Analyse a block of interleaved audio samples and return a [`DetectionResult`].
    ///
    /// The block must contain at least `channels` samples (one frame). Longer blocks
    /// produce more accurate results. A typical choice is 1–10 seconds of audio.
    pub fn detect(&self, samples: &[f32]) -> DetectionResult {
        let n = samples.len();
        let frames = n.checked_div(self.channels).unwrap_or(0);

        let rms_level = rms_f32(samples);
        let peak = samples.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max) as f64;

        let crest_factor_db = if rms_level < 1e-12 {
            0.0
        } else {
            20.0 * (peak / rms_level).log10()
        };

        let hf_energy_ratio = compute_hf_ratio(samples, self.sample_rate as f32);

        let format = self.classify(frames, crest_factor_db, hf_energy_ratio);
        let confidence = self.confidence_score(crest_factor_db, hf_energy_ratio, format);

        DetectionResult {
            format,
            confidence,
            crest_factor_db,
            hf_energy_ratio,
            rms_level,
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }

    /// Classify the format based on extracted features.
    fn classify(&self, frames: usize, crest_db: f64, hf_ratio: f64) -> DetectedFormat {
        let cfg = &self.config;
        let is_48k_family =
            (self.sample_rate - 48000.0).abs() < 1.0 || (self.sample_rate - 96000.0).abs() < 1.0;
        let is_cd_family =
            (self.sample_rate - 44100.0).abs() < 1.0 || (self.sample_rate - 88200.0).abs() < 1.0;
        let is_surround = self.channels > 2;

        // Speech-dominant heuristic
        let speech_dominant =
            crest_db < cfg.speech_crest_threshold_db || hf_ratio < cfg.speech_hf_ratio;

        // High-dynamic-range heuristic
        let high_dynamic_range = crest_db >= cfg.high_crest_threshold_db;

        if is_surround && is_48k_family {
            // Multichannel at broadcast sample rate → cinema or broadcast
            if high_dynamic_range && frames >= cfg.cinema_min_samples {
                return DetectedFormat::CinemaFeature;
            }
            return DetectedFormat::BroadcastFile;
        }

        if speech_dominant {
            // Short speech content → podcast; longer or 48 kHz → broadcast
            if is_48k_family {
                return DetectedFormat::BroadcastFile;
            }
            return DetectedFormat::Podcast;
        }

        if high_dynamic_range {
            // High crest + music HF content
            if is_cd_family && hf_ratio >= cfg.music_hf_ratio {
                return DetectedFormat::MusicAlbum;
            }
            if frames >= cfg.cinema_min_samples {
                return DetectedFormat::CinemaFeature;
            }
            return DetectedFormat::MusicAlbum;
        }

        // Mid-range crest + musical HF energy → streaming music
        if hf_ratio >= cfg.music_hf_ratio {
            return DetectedFormat::StreamingMusic;
        }

        // Default for 48 kHz mono/stereo with moderate dynamics
        if is_48k_family {
            return DetectedFormat::BroadcastFile;
        }

        DetectedFormat::Unknown
    }

    /// Compute a 0–1 confidence score for the classification decision.
    fn confidence_score(&self, crest_db: f64, hf_ratio: f64, format: DetectedFormat) -> f64 {
        let cfg = &self.config;
        match format {
            DetectedFormat::StreamingMusic => {
                // Confidence ↑ as HF ratio moves clearly above threshold
                let margin = hf_ratio - cfg.music_hf_ratio;
                (margin / cfg.music_hf_ratio).clamp(0.0, 1.0)
            }
            DetectedFormat::Podcast => {
                let margin = cfg.speech_hf_ratio - hf_ratio;
                (margin / cfg.speech_hf_ratio).clamp(0.0, 1.0)
            }
            DetectedFormat::BroadcastFile => {
                // Moderate confidence for the default 48 kHz case
                0.6
            }
            DetectedFormat::CinemaFeature | DetectedFormat::MusicAlbum => {
                let margin = crest_db - cfg.high_crest_threshold_db;
                (margin / 6.0).clamp(0.0, 1.0)
            }
            DetectedFormat::Unknown => 0.0,
        }
    }

    /// Returns the sample rate the detector was configured with.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Returns the channel count the detector was configured with.
    pub fn channels(&self) -> usize {
        self.channels
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Compute the RMS of a f32 slice.
fn rms_f32(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Estimate high-frequency energy ratio via a simple first-difference proxy.
///
/// The first difference `d[n] = x[n] - x[n-1]` emphasises high-frequency content.
/// The ratio `energy(d) / energy(x)` is ~0 for DC / low-frequency signals and ~1
/// for white noise. Typical music lies in [0.05, 0.25], speech in [0.02, 0.08].
///
/// This avoids FFT (and therefore no OxiFFT dependency) while being fast enough
/// for real-time use.
fn compute_hf_ratio(samples: &[f32], _sample_rate: f32) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let mut energy_signal = 0.0_f64;
    let mut energy_diff = 0.0_f64;

    for i in 1..samples.len() {
        let x = samples[i] as f64;
        let prev = samples[i - 1] as f64;
        let d = x - prev;
        energy_signal += x * x;
        energy_diff += d * d;
    }

    if energy_signal < 1e-15 {
        0.0
    } else {
        // Divide by 4 to normalise to [0,1] range (max diff energy for unit signal is 4)
        (energy_diff / energy_signal / 4.0).min(1.0)
    }
}

/// Select the appropriate [`Standard`] for the given audio content in one call.
///
/// This is a convenience wrapper around [`FormatDetector::detect`].
///
/// # Errors
///
/// Currently infallible; the error type is reserved for future validation.
pub fn select_standard(
    samples: &[f32],
    sample_rate: f64,
    channels: usize,
) -> NormalizeResult<(Standard, DetectionResult)> {
    let detector = FormatDetector::new(sample_rate, channels);
    let result = detector.detect(samples);
    let standard = result.recommended_standard();
    Ok((standard, result))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn pure_tone(freq: f32, sample_rate: u32, duration: f32, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amplitude * (2.0 * PI * freq * t).sin()
            })
            .collect()
    }

    fn white_noise_seeded(n: usize, seed: u64) -> Vec<f32> {
        // Deterministic LCG-based "noise" for reproducibility.
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let normalized = ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
                normalized * 0.3
            })
            .collect()
    }

    #[test]
    fn test_detector_creates_with_defaults() {
        let d = FormatDetector::new(48000.0, 2);
        assert_eq!(d.sample_rate(), 48000.0);
        assert_eq!(d.channels(), 2);
    }

    #[test]
    fn test_detect_returns_result() {
        let samples = pure_tone(1000.0, 48000, 1.0, 0.1);
        let d = FormatDetector::new(48000.0, 2);
        let r = d.detect(&samples);
        // Pure tone at 48 kHz stereo → broadcast or streaming
        assert!(
            r.format == DetectedFormat::BroadcastFile || r.format == DetectedFormat::StreamingMusic,
            "unexpected format: {:?}",
            r.format
        );
    }

    #[test]
    fn test_hf_ratio_low_for_pure_tone() {
        // A pure 100 Hz tone has minimal HF energy in first-difference proxy
        let samples = pure_tone(100.0, 44100, 0.5, 0.5);
        let hf = compute_hf_ratio(&samples, 44100.0);
        assert!(
            hf < 0.05,
            "pure low-freq tone should have low HF ratio, got {hf}"
        );
    }

    #[test]
    fn test_hf_ratio_high_for_white_noise() {
        let samples = white_noise_seeded(44100, 42);
        let hf = compute_hf_ratio(&samples, 44100.0);
        assert!(hf > 0.1, "white noise should have high HF ratio, got {hf}");
    }

    #[test]
    fn test_crest_factor_computed_correctly() {
        // Sine wave: crest factor = peak / RMS = 1 / (1/√2) = √2 ≈ 3.01 dB
        let samples = pure_tone(1000.0, 44100, 1.0, 1.0);
        let d = FormatDetector::new(44100.0, 1);
        let r = d.detect(&samples);
        let expected_crest_db = 20.0 * 2.0_f64.sqrt().log10();
        assert!(
            (r.crest_factor_db - expected_crest_db).abs() < 0.5,
            "crest factor {} differs from expected {expected_crest_db}",
            r.crest_factor_db
        );
    }

    #[test]
    fn test_broadcast_selected_for_48k_surround() {
        // 5.1 at 48 kHz → broadcast
        let samples = pure_tone(440.0, 48000, 0.5, 0.1)
            .into_iter()
            .flat_map(|s| [s; 6])
            .collect::<Vec<f32>>();
        let d = FormatDetector::new(48000.0, 6);
        let r = d.detect(&samples);
        assert_eq!(r.format, DetectedFormat::BroadcastFile);
        assert_eq!(r.recommended_standard(), Standard::EbuR128);
    }

    #[test]
    fn test_detected_format_labels() {
        assert!(!DetectedFormat::BroadcastFile.label().is_empty());
        assert!(!DetectedFormat::StreamingMusic.label().is_empty());
        assert!(!DetectedFormat::Podcast.label().is_empty());
        assert!(!DetectedFormat::CinemaFeature.label().is_empty());
        assert!(!DetectedFormat::MusicAlbum.label().is_empty());
        assert!(!DetectedFormat::Unknown.label().is_empty());
    }

    #[test]
    fn test_speech_dominant_flag() {
        assert!(DetectedFormat::Podcast.is_speech_dominant());
        assert!(DetectedFormat::BroadcastFile.is_speech_dominant());
        assert!(!DetectedFormat::StreamingMusic.is_speech_dominant());
        assert!(!DetectedFormat::MusicAlbum.is_speech_dominant());
    }

    #[test]
    fn test_select_standard_convenience_fn() {
        let samples = pure_tone(1000.0, 48000, 1.0, 0.2);
        let result = select_standard(&samples, 48000.0, 2);
        assert!(result.is_ok(), "select_standard should succeed");
        let (standard, detection) = result.expect("select_standard failed");
        assert_eq!(standard, detection.recommended_standard());
    }

    #[test]
    fn test_unknown_format_has_zero_confidence() {
        // Build a config where nothing matches to trigger Unknown
        let cfg = FormatDetectorConfig {
            high_crest_threshold_db: 100.0, // unreachably high
            speech_crest_threshold_db: 0.0, // no signal matches speech
            music_hf_ratio: 1.0,            // impossible to reach
            speech_hf_ratio: 0.0,
            cinema_min_samples: usize::MAX,
        };
        let samples = pure_tone(440.0, 44100, 0.1, 0.3);
        let d = FormatDetector::with_config(44100.0, 2, cfg);
        let r = d.detect(&samples);
        assert_eq!(r.format, DetectedFormat::Unknown);
        assert_eq!(r.confidence, 0.0);
    }

    #[test]
    fn test_rms_empty_returns_zero() {
        assert_eq!(rms_f32(&[]), 0.0);
    }

    #[test]
    fn test_hf_ratio_empty_returns_zero() {
        assert_eq!(compute_hf_ratio(&[], 44100.0), 0.0);
    }

    #[test]
    fn test_cinema_long_high_dynamic_range() {
        // Simulate a long 96-kHz stereo file with high dynamic range (crest > threshold).
        // We set the high_crest_threshold to 9 dB so that a 10:1 burst pattern (1 loud
        // frame in 10 near-zero frames) at 0.9 amplitude produces sufficient crest factor,
        // and speech_hf_ratio=0 prevents silence from being misclassified as speech.
        let cfg = FormatDetectorConfig {
            high_crest_threshold_db: 9.0,   // 10:1 burst gives ~10 dB crest
            speech_crest_threshold_db: 5.0, // only trigger speech for very compressed audio
            speech_hf_ratio: 0.0,           // disable silence→speech false positive
            music_hf_ratio: 0.5,            // require very high HF ratio for music classification
            cinema_min_samples: 96000 * 60, // 1 minute at 96 kHz
        };
        let frame_count = 96000 * 61; // just over 1 minute
        let sample_count = frame_count * 2; // stereo interleaved
                                            // 1-in-10 burst pattern: loud impulse every 10th frame, near-zero elsewhere.
        let mut samples = vec![0.001_f32; sample_count];
        for f in (0..frame_count).step_by(10) {
            samples[f * 2] = 0.9;
            samples[f * 2 + 1] = 0.9;
        }
        let d = FormatDetector::with_config(96000.0, 2, cfg);
        let r = d.detect(&samples);
        // With long duration and high crest, must be CinemaFeature.
        assert!(
            r.format == DetectedFormat::CinemaFeature,
            "expected CinemaFeature, got {:?} (crest={:.1} dB, hf={:.4}, frames={})",
            r.format,
            r.crest_factor_db,
            r.hf_energy_ratio,
            frame_count
        );
    }
}
