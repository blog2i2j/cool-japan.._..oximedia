//! Pitch detection, analysis, and tracking.
//!
//! Provides autocorrelation-based pitch detection, MIDI note conversion, and
//! a smoothing tracker suitable for real-time use.

use std::f64::consts::LN_2;

/// The result of a single pitch detection run.
#[derive(Debug, Clone)]
pub struct PitchResult {
    /// Detected fundamental frequency in Hz.
    pub frequency_hz: f64,
    /// Confidence of the detection (0.0 – 1.0).
    pub confidence: f64,
    /// Nearest MIDI note number (0 – 127), or `None` if out of MIDI range.
    pub midi_note: Option<u8>,
    /// Deviation from the nearest MIDI note in cents (±50 cents range).
    pub cents_offset: f64,
}

/// Detect the fundamental frequency of `samples` using normalised
/// autocorrelation (AMDF-style normalisation).
///
/// Returns `None` when no confident pitch is found inside `[min_hz, max_hz]`.
pub fn autocorrelation_pitch(
    samples: &[f64],
    sample_rate: f64,
    min_hz: f64,
    max_hz: f64,
) -> Option<PitchResult> {
    if samples.is_empty() || sample_rate <= 0.0 || min_hz <= 0.0 || max_hz <= min_hz {
        return None;
    }

    let min_lag = (sample_rate / max_hz).ceil() as usize;
    let max_lag = (sample_rate / min_hz).floor() as usize;

    if max_lag >= samples.len() || min_lag == 0 {
        return None;
    }

    // Compute autocorrelation at lag 0 for normalisation.
    let r0: f64 = samples.iter().map(|s| s * s).sum();
    if r0 < 1e-12 {
        return None; // silence
    }

    let mut best_lag = 0usize;
    let mut best_r = f64::NEG_INFINITY;

    for lag in min_lag..=max_lag {
        let n = samples.len() - lag;
        let r: f64 = (0..n).map(|i| samples[i] * samples[i + lag]).sum();
        // Normalise by the power of each window.
        let r_left: f64 = samples[..n].iter().map(|s| s * s).sum();
        let r_right: f64 = samples[lag..lag + n].iter().map(|s| s * s).sum();
        let denom = (r_left * r_right).sqrt();
        let r_norm = if denom > 1e-12 { r / denom } else { 0.0 };
        if r_norm > best_r {
            best_r = r_norm;
            best_lag = lag;
        }
    }

    if best_lag == 0 || best_r < 0.1 {
        return None;
    }

    let frequency_hz = sample_rate / best_lag as f64;
    let confidence = best_r.clamp(0.0, 1.0);

    let fractional_midi = hz_to_midi(frequency_hz);
    let nearest = fractional_midi.round();
    let cents_offset = (fractional_midi - nearest) * 100.0;
    let midi_note = if nearest >= 0.0 && nearest <= 127.0 {
        Some(nearest as u8)
    } else {
        None
    };

    Some(PitchResult {
        frequency_hz,
        confidence,
        midi_note,
        cents_offset,
    })
}

/// Convert a frequency in Hz to a fractional MIDI note number.
///
/// MIDI note 69 = A4 = 440 Hz.
pub fn hz_to_midi(hz: f64) -> f64 {
    if hz <= 0.0 {
        return 0.0;
    }
    69.0 + 12.0 * (hz / 440.0).log2()
}

/// Convert a MIDI note number (may be fractional) to a frequency in Hz.
pub fn midi_to_hz(midi: f64) -> f64 {
    440.0 * 2.0_f64.powf((midi - 69.0) / 12.0)
}

/// Return the nearest MIDI note and the deviation in cents.
///
/// Returns `(midi_note, cents_deviation)` where cents_deviation is in the
/// range `(-50, +50]`.
pub fn nearest_note(hz: f64) -> (u8, f64) {
    let fractional = hz_to_midi(hz);
    let note = fractional.round().clamp(0.0, 127.0) as u8;
    let cents = (fractional - note as f64) * 100.0;
    (note, cents)
}

/// Convert a pitch deviation in cents to a linear frequency ratio.
///
/// A ratio of 1.0 means no change; 100 cents = one semitone ≈ 1.0595.
pub fn cents_to_ratio(cents: f64) -> f64 {
    2.0_f64.powf(cents / 1200.0)
}

// ---------------------------------------------------------------------------
// PitchTracker
// ---------------------------------------------------------------------------

/// A stateful tracker that smooths a sequence of [`PitchResult`]s using an
/// exponential moving average (EMA) on the detected frequency.
pub struct PitchTracker {
    /// History of raw pitch results.
    pub history: Vec<PitchResult>,
    /// EMA smoothing coefficient (0.0 = no smoothing, 1.0 = never update).
    pub smoothing: f64,
    /// Current smoothed frequency estimate in Hz.
    smoothed_hz: f64,
}

impl PitchTracker {
    /// Create a new tracker with the given smoothing coefficient.
    ///
    /// `smoothing` must be in `[0.0, 1.0)`.  A value of `0.0` passes each
    /// detection through unchanged; `0.9` applies heavy smoothing.
    pub fn new(smoothing: f64) -> Self {
        Self {
            history: Vec::new(),
            smoothing: smoothing.clamp(0.0, 0.999),
            smoothed_hz: 0.0,
        }
    }

    /// Feed a new [`PitchResult`] and return the current smoothed frequency.
    pub fn update(&mut self, result: PitchResult) -> f64 {
        let freq = result.frequency_hz;
        if self.smoothed_hz <= 0.0 {
            self.smoothed_hz = freq;
        } else {
            self.smoothed_hz = self.smoothing * self.smoothed_hz + (1.0 - self.smoothing) * freq;
        }
        self.history.push(result);
        self.smoothed_hz
    }

    /// Return `true` if the last few estimates are consistent (within ±1
    /// semitone of each other), indicating a stable pitch.
    pub fn is_stable(&self) -> bool {
        const WINDOW: usize = 4;
        const SEMITONE_THRESHOLD: f64 = 100.0; // cents
        if self.history.len() < WINDOW {
            return false;
        }
        let recent = &self.history[self.history.len() - WINDOW..];
        let freqs: Vec<f64> = recent.iter().map(|r| r.frequency_hz).collect();
        let mean = freqs.iter().sum::<f64>() / freqs.len() as f64;
        for &f in &freqs {
            let cents = (hz_to_midi(f) - hz_to_midi(mean)).abs() * 100.0;
            if cents > SEMITONE_THRESHOLD {
                return false;
            }
        }
        true
    }
}

// Suppress the unused import lint for LN_2 used in doctests / future use.
#[allow(dead_code)]
const _LN2: f64 = LN_2;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate a pure sine wave at the given frequency.
    fn sine(freq_hz: f64, sample_rate: f64, frames: usize) -> Vec<f64> {
        (0..frames)
            .map(|i| (2.0 * PI * freq_hz * i as f64 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_hz_to_midi_a4() {
        let m = hz_to_midi(440.0);
        assert!((m - 69.0).abs() < 1e-9);
    }

    #[test]
    fn test_hz_to_midi_a5() {
        let m = hz_to_midi(880.0);
        assert!((m - 81.0).abs() < 1e-9);
    }

    #[test]
    fn test_midi_to_hz_a4() {
        let hz = midi_to_hz(69.0);
        assert!((hz - 440.0).abs() < 1e-6);
    }

    #[test]
    fn test_midi_to_hz_c4() {
        // MIDI 60 = C4 ≈ 261.626 Hz
        let hz = midi_to_hz(60.0);
        assert!((hz - 261.626).abs() < 0.01);
    }

    #[test]
    fn test_hz_midi_roundtrip() {
        let original = 329.63; // E4
        let recovered = midi_to_hz(hz_to_midi(original));
        assert!((recovered - original).abs() < 0.01);
    }

    #[test]
    fn test_nearest_note_a4() {
        let (note, cents) = nearest_note(440.0);
        assert_eq!(note, 69);
        assert!(cents.abs() < 1e-6);
    }

    #[test]
    fn test_nearest_note_slightly_sharp() {
        // 442 Hz is slightly sharp of A4
        let (note, cents) = nearest_note(442.0);
        assert_eq!(note, 69);
        assert!(cents > 0.0);
    }

    #[test]
    fn test_cents_to_ratio_zero() {
        assert!((cents_to_ratio(0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cents_to_ratio_one_octave() {
        // 1200 cents = one octave = 2x frequency
        assert!((cents_to_ratio(1200.0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_autocorrelation_pitch_440hz() {
        let samples = sine(440.0, 44100.0, 4096);
        // Narrow the search range so only the 440 Hz period fits.
        // min_lag = ceil(44100 / 500) = 89, max_lag = floor(44100 / 400) = 110.
        let result = autocorrelation_pitch(&samples, 44100.0, 400.0, 500.0);
        assert!(result.is_some());
        let r = result.expect("should succeed");
        assert!(
            (r.frequency_hz - 440.0).abs() < 5.0,
            "freq={}",
            r.frequency_hz
        );
        assert!(r.confidence > 0.5);
    }

    #[test]
    fn test_autocorrelation_pitch_silence() {
        let samples = vec![0.0f64; 4096];
        let result = autocorrelation_pitch(&samples, 44100.0, 80.0, 1000.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_pitch_tracker_smoothing() {
        let mut tracker = PitchTracker::new(0.0);
        let r1 = PitchResult {
            frequency_hz: 440.0,
            confidence: 0.9,
            midi_note: Some(69),
            cents_offset: 0.0,
        };
        let freq = tracker.update(r1);
        assert!((freq - 440.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pitch_tracker_stable() {
        let mut tracker = PitchTracker::new(0.0);
        for _ in 0..5 {
            tracker.update(PitchResult {
                frequency_hz: 440.0,
                confidence: 0.95,
                midi_note: Some(69),
                cents_offset: 0.0,
            });
        }
        assert!(tracker.is_stable());
    }

    #[test]
    fn test_pitch_tracker_not_stable_few_samples() {
        let mut tracker = PitchTracker::new(0.0);
        tracker.update(PitchResult {
            frequency_hz: 440.0,
            confidence: 0.9,
            midi_note: Some(69),
            cents_offset: 0.0,
        });
        assert!(!tracker.is_stable());
    }
}
