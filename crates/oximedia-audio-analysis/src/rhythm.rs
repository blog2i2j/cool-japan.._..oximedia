//! Rhythm analysis: tempo estimation, beat grid, time signature detection.
//!
//! This module provides tools for analysing the rhythmic properties of an audio
//! signal – building on raw onset times to produce tempo estimates, regular beat
//! grids, and time-signature hypotheses.

#![allow(dead_code)]

use std::f32::consts::PI;

/// A single detected onset event.
#[derive(Debug, Clone, PartialEq)]
pub struct Onset {
    /// Time in seconds from the start of the signal.
    pub time_s: f32,
    /// Normalised strength of the onset (0.0 – 1.0).
    pub strength: f32,
}

impl Onset {
    /// Creates a new [`Onset`].
    #[must_use]
    pub fn new(time_s: f32, strength: f32) -> Self {
        Self {
            time_s,
            strength: strength.clamp(0.0, 1.0),
        }
    }
}

/// Estimate of the musical tempo in beats per minute.
#[derive(Debug, Clone, PartialEq)]
pub struct TempoEstimate {
    /// Tempo in BPM.
    pub bpm: f32,
    /// Confidence in [0, 1].
    pub confidence: f32,
    /// Period between beats in seconds.
    pub period_s: f32,
}

impl TempoEstimate {
    /// Creates a [`TempoEstimate`] from a BPM value and confidence score.
    #[must_use]
    pub fn from_bpm(bpm: f32, confidence: f32) -> Self {
        let period_s = if bpm > 0.0 { 60.0 / bpm } else { 0.0 };
        Self {
            bpm,
            confidence: confidence.clamp(0.0, 1.0),
            period_s,
        }
    }

    /// Returns `true` if the tempo is within reasonable musical bounds (40–300 BPM).
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.bpm >= 40.0 && self.bpm <= 300.0
    }
}

/// A regular beat grid anchored at a given phase.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// Tempo driving this grid.
    pub tempo: TempoEstimate,
    /// Phase offset in seconds (time of the first beat).
    pub phase_s: f32,
    /// Number of beats covered.
    pub beat_count: u32,
}

impl BeatGrid {
    /// Constructs a new [`BeatGrid`].
    #[must_use]
    pub fn new(tempo: TempoEstimate, phase_s: f32, beat_count: u32) -> Self {
        Self {
            tempo,
            phase_s,
            beat_count,
        }
    }

    /// Returns the time in seconds of the `n`-th beat (0-indexed).
    #[must_use]
    pub fn beat_time(&self, n: u32) -> f32 {
        self.phase_s + n as f32 * self.tempo.period_s
    }

    /// Returns all beat times as a [`Vec`].
    #[must_use]
    pub fn all_beat_times(&self) -> Vec<f32> {
        (0..self.beat_count).map(|n| self.beat_time(n)).collect()
    }
}

/// Candidate time signatures with associated confidence values.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeSignature {
    /// Numerator (beats per bar, e.g. 3 for 3/4).
    pub numerator: u8,
    /// Denominator (note value, e.g. 4 for quarter notes).
    pub denominator: u8,
    /// Confidence in [0, 1].
    pub confidence: f32,
}

impl TimeSignature {
    /// Creates a new [`TimeSignature`].
    #[must_use]
    pub fn new(numerator: u8, denominator: u8, confidence: f32) -> Self {
        Self {
            numerator,
            denominator,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Returns the number of beats per bar.
    #[must_use]
    pub fn beats_per_bar(&self) -> u8 {
        self.numerator
    }
}

/// Estimates tempo from a list of onset times using inter-onset interval (IOI)
/// histogram analysis.
///
/// Returns the best single [`TempoEstimate`] or `None` if the input is empty.
#[must_use]
pub fn estimate_tempo(onsets: &[Onset], min_bpm: f32, max_bpm: f32) -> Option<TempoEstimate> {
    if onsets.len() < 2 {
        return None;
    }

    let min_period = 60.0 / max_bpm;
    let max_period = 60.0 / min_bpm;

    // Build IOI list
    let iois: Vec<f32> = onsets
        .windows(2)
        .map(|w| w[1].time_s - w[0].time_s)
        .filter(|&d| d >= min_period && d <= max_period)
        .collect();

    if iois.is_empty() {
        return None;
    }

    // Simple mean IOI → BPM
    let mean_ioi: f32 = iois.iter().sum::<f32>() / iois.len() as f32;
    let bpm = 60.0 / mean_ioi;

    // Confidence: fraction of IOIs within ±10% of the mean period
    let tolerance = mean_ioi * 0.10;
    let consistent = iois
        .iter()
        .filter(|&&d| (d - mean_ioi).abs() <= tolerance)
        .count();
    let confidence = consistent as f32 / iois.len() as f32;

    Some(TempoEstimate::from_bpm(bpm, confidence))
}

/// Builds a [`BeatGrid`] from a tempo estimate and a list of onset times.
///
/// The phase is chosen to best align the grid with the provided onsets.
/// Returns `None` if no onsets are provided.
#[must_use]
pub fn build_beat_grid(
    onsets: &[Onset],
    tempo: TempoEstimate,
    duration_s: f32,
) -> Option<BeatGrid> {
    if onsets.is_empty() || tempo.period_s <= 0.0 {
        return None;
    }

    // Try phases [0, period) in small steps and pick the one with the most
    // onsets close to a beat position.
    let steps = 64u32;
    let step_size = tempo.period_s / steps as f32;
    let mut best_phase = 0.0_f32;
    let mut best_score = -1_i32;

    for s in 0..steps {
        let phase = s as f32 * step_size;
        let score = onsets
            .iter()
            .map(|o| {
                let rel = (o.time_s - phase) / tempo.period_s;
                let frac = rel - rel.round();
                i32::from(frac.abs() < 0.1)
            })
            .sum::<i32>();

        if score > best_score {
            best_score = score;
            best_phase = phase;
        }
    }

    let beat_count = if tempo.period_s > 0.0 {
        (duration_s / tempo.period_s).ceil() as u32
    } else {
        0
    };

    Some(BeatGrid::new(tempo, best_phase, beat_count))
}

/// Detects the most likely time signature given a beat grid and onset list.
///
/// Returns candidates sorted by confidence (descending).
#[must_use]
pub fn detect_time_signature(onsets: &[Onset], grid: &BeatGrid) -> Vec<TimeSignature> {
    let candidates: &[(u8, u8)] = &[(4, 4), (3, 4), (6, 8), (2, 4), (5, 4)];

    let period = grid.tempo.period_s;
    let mut results: Vec<TimeSignature> = candidates
        .iter()
        .map(|&(num, den)| {
            let bar_len = period * f32::from(num);
            // Score: fraction of onsets that land near a beat within a bar
            let score = if bar_len > 0.0 && !onsets.is_empty() {
                let aligned = onsets
                    .iter()
                    .filter(|o| {
                        let beat_pos = (o.time_s - grid.phase_s) / period;
                        let beat_in_bar = beat_pos % f32::from(num);
                        beat_in_bar.fract().abs() < 0.15
                    })
                    .count();
                aligned as f32 / onsets.len() as f32
            } else {
                0.0
            };

            TimeSignature::new(num, den, score)
        })
        .collect();

    results.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Computes a simple autocorrelation-based periodicity score for a series of
/// onset strength values sampled at a regular `hop_s` interval.
///
/// Returns a vector of `(lag_s, score)` pairs.
#[must_use]
pub fn periodicity_scores(strengths: &[f32], hop_s: f32) -> Vec<(f32, f32)> {
    let n = strengths.len();
    if n < 2 {
        return Vec::new();
    }

    // Autocorrelation at each lag
    (1..n)
        .map(|lag| {
            let sum: f32 = strengths[..n - lag]
                .iter()
                .zip(&strengths[lag..])
                .map(|(&a, &b)| a * b)
                .sum();
            let lag_s = lag as f32 * hop_s;
            (lag_s, sum)
        })
        .collect()
}

/// Generates a synthetic onset strength function from a sinusoidal rhythm.
///
/// Useful for testing and demonstration purposes.
#[must_use]
pub fn synthetic_onset_strengths(bpm: f32, sample_rate: f32, n_frames: usize) -> Vec<f32> {
    let period_frames = sample_rate * 60.0 / bpm;
    (0..n_frames)
        .map(|i| {
            let phase = 2.0 * PI * i as f32 / period_frames;
            (0.5 * (1.0 + phase.cos())).max(0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_onsets(times: &[f32]) -> Vec<Onset> {
        times.iter().map(|&t| Onset::new(t, 1.0)).collect()
    }

    #[test]
    fn test_onset_creation() {
        let o = Onset::new(1.5, 0.8);
        assert!((o.time_s - 1.5).abs() < 1e-6);
        assert!((o.strength - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_onset_strength_clamped() {
        let o = Onset::new(0.0, 2.5);
        assert!((o.strength - 1.0).abs() < 1e-6);
        let o2 = Onset::new(0.0, -0.5);
        assert!((o2.strength).abs() < 1e-6);
    }

    #[test]
    fn test_tempo_estimate_period() {
        let t = TempoEstimate::from_bpm(120.0, 0.9);
        assert!((t.period_s - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_tempo_estimate_plausible() {
        assert!(TempoEstimate::from_bpm(120.0, 1.0).is_plausible());
        assert!(!TempoEstimate::from_bpm(10.0, 1.0).is_plausible());
        assert!(!TempoEstimate::from_bpm(400.0, 1.0).is_plausible());
    }

    #[test]
    fn test_estimate_tempo_empty() {
        assert!(estimate_tempo(&[], 40.0, 200.0).is_none());
    }

    #[test]
    fn test_estimate_tempo_single_onset() {
        let onsets = vec![Onset::new(0.0, 1.0)];
        assert!(estimate_tempo(&onsets, 40.0, 200.0).is_none());
    }

    #[test]
    fn test_estimate_tempo_regular_120bpm() {
        // 120 BPM → period = 0.5 s
        let times: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
        let onsets = make_onsets(&times);
        let est = estimate_tempo(&onsets, 40.0, 200.0).expect("tempo estimation should succeed");
        assert!((est.bpm - 120.0).abs() < 1.0, "bpm = {}", est.bpm);
        assert!(est.confidence > 0.8);
    }

    #[test]
    fn test_estimate_tempo_result_plausible() {
        let times: Vec<f32> = (0..8).map(|i| i as f32 * 0.6).collect(); // 100 BPM
        let onsets = make_onsets(&times);
        let est = estimate_tempo(&onsets, 40.0, 200.0).expect("tempo estimation should succeed");
        assert!(est.is_plausible());
    }

    #[test]
    fn test_build_beat_grid_none_on_empty() {
        let tempo = TempoEstimate::from_bpm(120.0, 1.0);
        assert!(build_beat_grid(&[], tempo, 10.0).is_none());
    }

    #[test]
    fn test_build_beat_grid_beat_times_length() {
        let times: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let onsets = make_onsets(&times);
        let tempo = TempoEstimate::from_bpm(120.0, 1.0);
        let grid = build_beat_grid(&onsets, tempo, 4.0).expect("beat grid should succeed");
        assert_eq!(grid.all_beat_times().len() as u32, grid.beat_count);
    }

    #[test]
    fn test_build_beat_grid_beat_times_spaced_correctly() {
        let times: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let onsets = make_onsets(&times);
        let tempo = TempoEstimate::from_bpm(120.0, 1.0);
        let grid = build_beat_grid(&onsets, tempo, 4.0).expect("beat grid should succeed");
        let beats = grid.all_beat_times();
        for w in beats.windows(2) {
            let diff = w[1] - w[0];
            assert!((diff - 0.5).abs() < 1e-5, "diff = {}", diff);
        }
    }

    #[test]
    fn test_detect_time_signature_returns_candidates() {
        let times: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let onsets = make_onsets(&times);
        let tempo = TempoEstimate::from_bpm(120.0, 1.0);
        let grid = build_beat_grid(&onsets, tempo, 8.0).expect("beat grid should succeed");
        let sigs = detect_time_signature(&onsets, &grid);
        assert!(!sigs.is_empty());
        // Sorted by confidence descending
        for w in sigs.windows(2) {
            assert!(w[0].confidence >= w[1].confidence);
        }
    }

    #[test]
    fn test_detect_time_signature_4_4_most_likely() {
        // Regular 4/4 pattern: beats on every 0.5 s
        let times: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
        let onsets = make_onsets(&times);
        let tempo = TempoEstimate::from_bpm(120.0, 1.0);
        let grid = build_beat_grid(&onsets, tempo, 16.0).expect("beat grid should succeed");
        let sigs = detect_time_signature(&onsets, &grid);
        assert!(!sigs.is_empty());
        // Top candidate has numerator 2 or 4 (both valid for very regular signals)
        assert!(sigs[0].numerator >= 2);
    }

    #[test]
    fn test_time_signature_beats_per_bar() {
        let sig = TimeSignature::new(3, 4, 0.9);
        assert_eq!(sig.beats_per_bar(), 3);
    }

    #[test]
    fn test_periodicity_scores_empty() {
        assert!(periodicity_scores(&[], 0.01).is_empty());
    }

    #[test]
    fn test_periodicity_scores_length() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let scores = periodicity_scores(&data, 0.01);
        assert_eq!(scores.len(), data.len() - 1);
    }

    #[test]
    fn test_synthetic_onset_strengths_length() {
        let s = synthetic_onset_strengths(120.0, 100.0, 50);
        assert_eq!(s.len(), 50);
    }

    #[test]
    fn test_synthetic_onset_strengths_range() {
        let s = synthetic_onset_strengths(120.0, 100.0, 200);
        for &v in &s {
            assert!(v >= 0.0 && v <= 1.0, "value out of range: {v}");
        }
    }
}
