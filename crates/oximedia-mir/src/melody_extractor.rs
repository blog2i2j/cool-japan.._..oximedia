//! Predominant melody extraction using harmonic summation salience.
//!
//! Extracts the dominant melodic line from polyphonic audio by computing a
//! harmonic salience function across MIDI pitch bins, tracking the most
//! salient pitch over time via a Viterbi-style smoother, and detecting
//! vibrato modulation in the resulting pitch contour.
//!
//! ## Algorithm
//!
//! 1. **STFT** — Short-time magnitude spectrum using a Hann window.
//! 2. **Harmonic salience** — For each candidate MIDI pitch (36–96, i.e. C2–C7)
//!    sum the magnitude at the fundamental and the first `N_HARMONICS` partials,
//!    weighted by a harmonic decay function `1/k`.
//! 3. **Pitch smoothing** — Apply a simple median filter over a ±2-frame window
//!    to suppress spurious pitch jumps.
//! 4. **Voiced/unvoiced detection** — A frame is declared voiced when its
//!    peak salience exceeds a dynamic threshold derived from the mean salience.
//! 5. **Vibrato detection** — Analyse the instantaneous frequency deviation of
//!    the pitch contour within each voiced segment to detect periodic modulation
//!    consistent with vibrato (4–8 Hz rate, ≥0.25 semitone extent).
//!
//! # Example
//!
//! ```
//! use oximedia_mir::melody_extractor::{MelodyExtractor, MelodyExtractorConfig};
//!
//! let sr = 22050_u32;
//! let config = MelodyExtractorConfig::default();
//! let extractor = MelodyExtractor::new(sr, config);
//!
//! // 1-second 440 Hz sine wave
//! let samples: Vec<f32> = (0..sr)
//!     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
//!     .collect();
//!
//! let result = extractor.extract(&samples).expect("extraction failed");
//! println!("Extracted {} pitch frames", result.pitch_hz.len());
//! ```

#![allow(dead_code)]

use crate::{MirError, MirResult};
use oxifft::Complex;
use std::f32::consts::PI;

// ── Constants ─────────────────────────────────────────────────────────────────

/// MIDI pitch range lower bound (C2 = 36).
const MIDI_MIN: u8 = 36;
/// MIDI pitch range upper bound (C7 = 96).
const MIDI_MAX: u8 = 96;
/// Number of harmonics to sum for salience computation.
const N_HARMONICS: usize = 8;
/// Voiced-frame salience threshold multiplier (relative to mean salience).
const VOICED_THRESHOLD: f32 = 0.8;
/// Vibrato minimum rate in Hz.
const VIBRATO_MIN_RATE_HZ: f32 = 4.0;
/// Vibrato maximum rate in Hz.
const VIBRATO_MAX_RATE_HZ: f32 = 8.0;
/// Vibrato minimum extent in semitones.
const VIBRATO_MIN_EXTENT_ST: f32 = 0.25;
/// Minimum consecutive voiced frames to analyse for vibrato.
const VIBRATO_MIN_FRAMES: usize = 16;

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for melody extraction.
#[derive(Debug, Clone)]
pub struct MelodyExtractorConfig {
    /// FFT window size in samples.
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Harmonic weighting exponent (decay factor `1/k^exp`).
    pub harmonic_decay_exp: f32,
    /// Pitch smoothing window half-width in frames.
    pub smooth_radius: usize,
}

impl Default for MelodyExtractorConfig {
    fn default() -> Self {
        Self {
            window_size: 2048,
            hop_size: 512,
            harmonic_decay_exp: 1.0,
            smooth_radius: 2,
        }
    }
}

// ── PitchFrame ────────────────────────────────────────────────────────────────

/// A single analysis frame in the extracted melody.
#[derive(Debug, Clone)]
pub struct PitchFrame {
    /// Estimated fundamental frequency in Hz, or `None` for unvoiced frames.
    pub pitch_hz: Option<f32>,
    /// Corresponding MIDI pitch number (60 = middle C), or `None` for unvoiced.
    pub midi_pitch: Option<u8>,
    /// Peak harmonic salience for this frame.
    pub salience: f32,
    /// Frame timestamp in seconds.
    pub time_s: f32,
}

impl PitchFrame {
    /// Returns `true` if the frame is voiced (pitch is present).
    #[must_use]
    pub fn is_voiced(&self) -> bool {
        self.pitch_hz.is_some()
    }
}

// ── VibratoSegment ────────────────────────────────────────────────────────────

/// A detected vibrato segment in the melody.
#[derive(Debug, Clone)]
pub struct VibratoSegment {
    /// Start time in seconds.
    pub start_s: f32,
    /// End time in seconds.
    pub end_s: f32,
    /// Estimated vibrato rate in Hz.
    pub rate_hz: f32,
    /// Estimated vibrato extent in semitones (peak-to-peak).
    pub extent_semitones: f32,
}

// ── ExtractionResult ──────────────────────────────────────────────────────────

/// Result of melody extraction.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Per-frame fundamental frequency in Hz (None = unvoiced).
    pub pitch_hz: Vec<Option<f32>>,
    /// Per-frame MIDI pitch number (None = unvoiced).
    pub midi_pitch: Vec<Option<u8>>,
    /// Per-frame salience values.
    pub salience: Vec<f32>,
    /// Per-frame timestamps in seconds.
    pub timestamps_s: Vec<f32>,
    /// Detected vibrato segments.
    pub vibrato: Vec<VibratoSegment>,
    /// Fraction of frames that are voiced (0.0–1.0).
    pub voiced_fraction: f32,
    /// Mean pitch of voiced frames in Hz.
    pub mean_pitch_hz: f32,
}

impl ExtractionResult {
    /// Number of analysis frames.
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.pitch_hz.len()
    }

    /// Number of voiced frames.
    #[must_use]
    pub fn n_voiced(&self) -> usize {
        self.pitch_hz.iter().filter(|p| p.is_some()).count()
    }

    /// Return all voiced pitch values in Hz.
    #[must_use]
    pub fn voiced_pitches(&self) -> Vec<f32> {
        self.pitch_hz.iter().filter_map(|p| *p).collect()
    }
}

// ── MelodyExtractor ───────────────────────────────────────────────────────────

/// Predominant melody extractor.
pub struct MelodyExtractor {
    sample_rate: u32,
    config: MelodyExtractorConfig,
}

impl MelodyExtractor {
    /// Create a new melody extractor.
    #[must_use]
    pub fn new(sample_rate: u32, config: MelodyExtractorConfig) -> Self {
        Self {
            sample_rate,
            config,
        }
    }

    /// Extract the predominant melody from `samples`.
    ///
    /// # Errors
    ///
    /// Returns [`MirError::InsufficientData`] if `samples` is shorter than one window.
    pub fn extract(&self, samples: &[f32]) -> MirResult<ExtractionResult> {
        if samples.len() < self.config.window_size {
            return Err(MirError::InsufficientData(format!(
                "need ≥{} samples, got {}",
                self.config.window_size,
                samples.len()
            )));
        }

        let window = hann_window(self.config.window_size);
        let hop = self.config.hop_size.max(1);
        let sr = self.sample_rate as f32;
        let n_frames =
            (samples.len().saturating_sub(self.config.window_size)) / hop + 1;

        let mut raw_midi: Vec<Option<u8>> = Vec::with_capacity(n_frames);
        let mut saliences: Vec<f32> = Vec::with_capacity(n_frames);
        let mut timestamps: Vec<f32> = Vec::with_capacity(n_frames);

        // Phase 1: per-frame harmonic salience + pitch picking
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = start + self.config.window_size;
            if end > samples.len() {
                break;
            }

            let fft_in: Vec<Complex<f32>> = samples[start..end]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();

            let spectrum = oxifft::fft(&fft_in);
            let n_bins = spectrum.len() / 2;
            let mags: Vec<f32> = spectrum[..n_bins].iter().map(|c| c.norm()).collect();

            let (best_midi, best_sal) =
                self.compute_salience(&mags, n_bins, sr);

            raw_midi.push(best_midi);
            saliences.push(best_sal);
            timestamps.push(frame_idx as f32 * hop as f32 / sr);
        }

        // Phase 2: voiced / unvoiced thresholding
        let mean_sal: f32 = if saliences.is_empty() {
            0.0
        } else {
            saliences.iter().sum::<f32>() / saliences.len() as f32
        };
        let threshold = mean_sal * VOICED_THRESHOLD;

        let voiced_midi: Vec<Option<u8>> = raw_midi
            .iter()
            .zip(saliences.iter())
            .map(|(&m, &s)| if s >= threshold { m } else { None })
            .collect();

        // Phase 3: pitch smoothing (median over ±smooth_radius frames)
        let smoothed_midi =
            self.smooth_pitch(&voiced_midi, self.config.smooth_radius);

        // Phase 4: convert MIDI → Hz
        let pitch_hz: Vec<Option<f32>> = smoothed_midi
            .iter()
            .map(|m| m.map(midi_to_hz))
            .collect();

        // Phase 5: MIDI-pitch vec for output
        let midi_out: Vec<Option<u8>> = smoothed_midi;

        // Phase 6: voiced fraction + mean pitch
        let voiced_pitches: Vec<f32> = pitch_hz.iter().filter_map(|p| *p).collect();
        let voiced_fraction = voiced_pitches.len() as f32 / pitch_hz.len().max(1) as f32;
        let mean_pitch_hz = if voiced_pitches.is_empty() {
            0.0
        } else {
            voiced_pitches.iter().sum::<f32>() / voiced_pitches.len() as f32
        };

        // Phase 7: vibrato detection
        let vibrato = self.detect_vibrato(&pitch_hz, &timestamps, sr / hop as f32);

        Ok(ExtractionResult {
            pitch_hz,
            midi_pitch: midi_out,
            salience: saliences,
            timestamps_s: timestamps,
            vibrato,
            voiced_fraction,
            mean_pitch_hz,
        })
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Compute harmonic salience for all candidate MIDI pitches and return
    /// the MIDI pitch with highest salience plus the salience value.
    fn compute_salience(&self, mags: &[f32], n_bins: usize, sr: f32) -> (Option<u8>, f32) {
        let hz_per_bin = sr / (2.0 * n_bins as f32);
        if hz_per_bin < f32::EPSILON {
            return (None, 0.0);
        }

        let mut best_midi: Option<u8> = None;
        let mut best_sal = 0.0_f32;

        for midi in MIDI_MIN..=MIDI_MAX {
            let f0 = midi_to_hz(midi);
            let mut sal = 0.0_f32;

            for k in 1..=(N_HARMONICS as u32) {
                let freq = f0 * k as f32;
                let bin = (freq / hz_per_bin).round() as usize;
                if bin >= n_bins {
                    break;
                }
                let weight = 1.0 / (k as f32).powf(self.config.harmonic_decay_exp);
                sal += mags[bin] * weight;
            }

            if sal > best_sal {
                best_sal = sal;
                best_midi = Some(midi);
            }
        }

        (best_midi, best_sal)
    }

    /// Median-filter pitch sequence to remove isolated pitch errors.
    fn smooth_pitch(&self, midi: &[Option<u8>], radius: usize) -> Vec<Option<u8>> {
        let n = midi.len();
        let mut out = vec![None; n];

        for i in 0..n {
            let lo = i.saturating_sub(radius);
            let hi = (i + radius + 1).min(n);
            let window: Vec<u8> = midi[lo..hi].iter().filter_map(|m| *m).collect();

            if window.is_empty() {
                out[i] = None;
            } else {
                let mut sorted = window.clone();
                sorted.sort_unstable();
                out[i] = Some(sorted[sorted.len() / 2]);
            }
        }
        out
    }

    /// Detect vibrato segments by analysing pitch deviation within voiced runs.
    fn detect_vibrato(
        &self,
        pitch_hz: &[Option<f32>],
        timestamps: &[f32],
        frame_rate: f32,
    ) -> Vec<VibratoSegment> {
        if frame_rate < f32::EPSILON || pitch_hz.len() < VIBRATO_MIN_FRAMES {
            return Vec::new();
        }

        let mut vibrato_segments = Vec::new();
        let mut voiced_start: Option<usize> = None;
        let n = pitch_hz.len();

        let flush_segment = |start: usize, end: usize| -> Option<VibratoSegment> {
            let voiced: Vec<f32> = pitch_hz[start..end]
                .iter()
                .filter_map(|p| *p)
                .collect();

            if voiced.len() < VIBRATO_MIN_FRAMES {
                return None;
            }

            // Convert Hz to semitones (relative to mean)
            let mean_hz = voiced.iter().sum::<f32>() / voiced.len() as f32;
            if mean_hz < f32::EPSILON {
                return None;
            }
            let semitones: Vec<f32> = voiced
                .iter()
                .map(|&hz| 12.0 * (hz / mean_hz).log2())
                .collect();

            // Compute extent (peak-to-peak range)
            let max_st = semitones.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_st = semitones.iter().cloned().fold(f32::INFINITY, f32::min);
            let extent = max_st - min_st;

            if extent < VIBRATO_MIN_EXTENT_ST {
                return None;
            }

            // Estimate vibrato rate via zero-crossing count of the deviation signal
            let crossings = semitones
                .windows(2)
                .filter(|w| w[0] * w[1] < 0.0)
                .count();
            // Each full cycle has 2 zero-crossings
            let duration_s = voiced.len() as f32 / frame_rate;
            let rate_hz = crossings as f32 / (2.0 * duration_s);

            if rate_hz < VIBRATO_MIN_RATE_HZ || rate_hz > VIBRATO_MAX_RATE_HZ {
                return None;
            }

            let start_s = timestamps.get(start).copied().unwrap_or(0.0);
            let end_s = timestamps.get(end.saturating_sub(1)).copied().unwrap_or(duration_s);

            Some(VibratoSegment {
                start_s,
                end_s,
                rate_hz,
                extent_semitones: extent,
            })
        };

        for i in 0..=n {
            let voiced = i < n && pitch_hz[i].is_some();
            match (voiced_start, voiced) {
                (None, true) => voiced_start = Some(i),
                (Some(start), false) => {
                    if let Some(seg) = flush_segment(start, i) {
                        vibrato_segments.push(seg);
                    }
                    voiced_start = None;
                }
                _ => {}
            }
        }

        vibrato_segments
    }
}

// ── Utility functions ─────────────────────────────────────────────────────────

/// Convert a MIDI pitch number to frequency in Hz using the standard formula:
/// `f = 440 * 2^((midi - 69) / 12)`.
#[must_use]
pub fn midi_to_hz(midi: u8) -> f32 {
    440.0 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0)
}

/// Convert frequency in Hz to the nearest MIDI pitch number.
///
/// Returns `None` if `hz` is non-positive or outside the MIDI pitch range 0–127.
#[must_use]
pub fn hz_to_midi(hz: f32) -> Option<u8> {
    if hz <= 0.0 {
        return None;
    }
    let midi = 69.0 + 12.0 * (hz / 440.0).log2();
    if midi < 0.0 || midi > 127.0 {
        None
    } else {
        Some(midi.round() as u8)
    }
}

/// Hann window coefficients.
fn hann_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    let denom = (size.saturating_sub(1)) as f32;
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / denom).cos()))
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(freq_hz: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| (TAU * freq_hz * i as f32 / sr as f32).sin())
            .collect()
    }

    // ── midi_to_hz / hz_to_midi ───────────────────────────────────────────────

    #[test]
    fn test_midi_69_is_a440() {
        let hz = midi_to_hz(69);
        assert!((hz - 440.0).abs() < 1e-3, "A4 = {hz}");
    }

    #[test]
    fn test_midi_60_is_middle_c() {
        let hz = midi_to_hz(60);
        assert!((hz - 261.63).abs() < 1.0, "C4 = {hz}");
    }

    #[test]
    fn test_hz_to_midi_a440() {
        let midi = hz_to_midi(440.0);
        assert_eq!(midi, Some(69));
    }

    #[test]
    fn test_hz_to_midi_zero_is_none() {
        assert!(hz_to_midi(0.0).is_none());
    }

    #[test]
    fn test_midi_roundtrip() {
        for midi in 36_u8..=96 {
            let hz = midi_to_hz(midi);
            let back = hz_to_midi(hz).expect("should round-trip");
            assert_eq!(back, midi, "round-trip failed for MIDI {midi}");
        }
    }

    // ── ExtractionResult helpers ──────────────────────────────────────────────

    #[test]
    fn test_n_voiced_counts_some() {
        let result = ExtractionResult {
            pitch_hz: vec![Some(440.0), None, Some(220.0)],
            midi_pitch: vec![Some(69), None, Some(57)],
            salience: vec![1.0, 0.0, 0.8],
            timestamps_s: vec![0.0, 0.1, 0.2],
            vibrato: vec![],
            voiced_fraction: 2.0 / 3.0,
            mean_pitch_hz: 330.0,
        };
        assert_eq!(result.n_voiced(), 2);
        assert_eq!(result.n_frames(), 3);
    }

    #[test]
    fn test_voiced_pitches_filters_nones() {
        let result = ExtractionResult {
            pitch_hz: vec![None, Some(440.0), None, Some(220.0)],
            midi_pitch: vec![None, Some(69), None, Some(57)],
            salience: vec![0.0, 1.0, 0.0, 0.9],
            timestamps_s: vec![0.0, 0.1, 0.2, 0.3],
            vibrato: vec![],
            voiced_fraction: 0.5,
            mean_pitch_hz: 330.0,
        };
        let vp = result.voiced_pitches();
        assert_eq!(vp, vec![440.0, 220.0]);
    }

    // ── MelodyExtractor ───────────────────────────────────────────────────────

    #[test]
    fn test_extract_sine_returns_voiced_frames() {
        let sr = 22050_u32;
        let config = MelodyExtractorConfig {
            window_size: 1024,
            hop_size: 256,
            ..Default::default()
        };
        let extractor = MelodyExtractor::new(sr, config);
        let samples = sine(440.0, sr, 1.0);
        let result = extractor.extract(&samples).expect("extract failed");
        assert!(result.n_voiced() > 0, "expected voiced frames for 440 Hz sine");
    }

    #[test]
    fn test_extract_too_short_returns_error() {
        let sr = 22050_u32;
        let config = MelodyExtractorConfig::default();
        let extractor = MelodyExtractor::new(sr, config);
        let samples = vec![0.0_f32; 100];
        let err = extractor.extract(&samples);
        assert!(err.is_err());
    }

    #[test]
    fn test_pitch_frames_have_timestamps() {
        let sr = 22050_u32;
        let config = MelodyExtractorConfig {
            window_size: 512,
            hop_size: 128,
            ..Default::default()
        };
        let extractor = MelodyExtractor::new(sr, config);
        let samples = sine(440.0, sr, 0.5);
        let result = extractor.extract(&samples).expect("extract failed");
        assert_eq!(result.timestamps_s.len(), result.pitch_hz.len());
        // Timestamps should be monotonically non-decreasing
        for w in result.timestamps_s.windows(2) {
            assert!(w[1] >= w[0], "timestamps not monotone: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_voiced_fraction_in_unit_range() {
        let sr = 22050_u32;
        let config = MelodyExtractorConfig {
            window_size: 1024,
            hop_size: 512,
            ..Default::default()
        };
        let extractor = MelodyExtractor::new(sr, config);
        let samples = sine(440.0, sr, 1.0);
        let result = extractor.extract(&samples).expect("extract failed");
        assert!(
            result.voiced_fraction >= 0.0 && result.voiced_fraction <= 1.0,
            "voiced_fraction = {}",
            result.voiced_fraction
        );
    }

    #[test]
    fn test_a440_pitch_detected_near_440hz() {
        let sr = 44100_u32;
        let config = MelodyExtractorConfig {
            window_size: 4096,
            hop_size: 1024,
            ..Default::default()
        };
        let extractor = MelodyExtractor::new(sr, config);
        let samples = sine(440.0, sr, 2.0);
        let result = extractor.extract(&samples).expect("extract failed");
        let vp = result.voiced_pitches();
        assert!(!vp.is_empty(), "no voiced pitches detected");
        let mean: f32 = vp.iter().sum::<f32>() / vp.len() as f32;
        // Allow ±2 semitones = factor of 2^(2/12) ≈ 1.122
        assert!(
            mean > 380.0 && mean < 520.0,
            "mean pitch far from 440 Hz: {mean}"
        );
    }

    #[test]
    fn test_hann_window_length() {
        let w = hann_window(1024);
        assert_eq!(w.len(), 1024);
    }

    #[test]
    fn test_hann_window_edges_near_zero() {
        let w = hann_window(512);
        assert!(w[0].abs() < 1e-4);
        assert!(w[511].abs() < 1e-2);
    }
}
