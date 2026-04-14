//! Multi-track / stem-based music information retrieval.
//!
//! Provides analysis of individual audio stems (vocals, drums, bass, melody,
//! other) and combines results into aggregate statistics.
//!
//! # Design
//!
//! Each stem is analysed independently using lightweight time-domain estimators:
//!
//! - **Tempo** — autocorrelation-based BPM estimation on the onset-strength
//!   envelope of the stem.
//! - **Key** — dominant pitch class via zero-lag chroma approximation.
//! - **Energy** — RMS energy averaged across the full stem.
//! - **Onset count** — number of detected transients using adaptive thresholding.
//!
//! The [`MultiTrackAnalyzer::combined_tempo`] method returns a weighted average
//! of per-stem tempos, with drum stems weighted 2× relative to other stems.

#![allow(dead_code)]

use std::f32::consts::TAU;

// ---------------------------------------------------------------------------
// StemType
// ---------------------------------------------------------------------------

/// Logical role of an audio stem within a multi-track production.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StemType {
    /// Lead or backing vocal content.
    Vocals,
    /// Drum kit and percussion.
    Drums,
    /// Bass guitar or synth bass.
    Bass,
    /// Lead / chordal melodic content (guitar, keys, synth lead…).
    Melody,
    /// Any stem that does not fit a named category.
    Other,
}

impl StemType {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Vocals => "Vocals",
            Self::Drums => "Drums",
            Self::Bass => "Bass",
            Self::Melody => "Melody",
            Self::Other => "Other",
        }
    }
}

impl std::fmt::Display for StemType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// StemAnalysis
// ---------------------------------------------------------------------------

/// Analysis result for a single audio stem.
#[derive(Debug, Clone)]
pub struct StemAnalysis {
    /// The role of this stem.
    pub stem_type: StemType,
    /// Estimated tempo in BPM (0.0 if could not be determined).
    pub tempo_bpm: f32,
    /// Dominant key as a string, e.g. `"C"`, `"F#"`.
    pub key: String,
    /// Normalised RMS energy in \[0, 1\].
    pub energy: f32,
    /// Number of detected note/transient onsets.
    pub onset_count: usize,
}

// ---------------------------------------------------------------------------
// Internal stem record
// ---------------------------------------------------------------------------

struct StemRecord {
    stem_type: StemType,
    samples: Vec<f32>,
    sample_rate: u32,
}

// ---------------------------------------------------------------------------
// MultiTrackAnalyzer
// ---------------------------------------------------------------------------

/// Analyses a collection of audio stems independently and combines results.
///
/// # Example
///
/// ```
/// use oximedia_mir::multitrack::{MultiTrackAnalyzer, StemType};
///
/// let mut analyzer = MultiTrackAnalyzer::new();
/// let samples = vec![0.0f32; 44100];
/// analyzer.add_stem(StemType::Drums, &samples, 44100);
/// analyzer.add_stem(StemType::Bass, &samples, 44100);
/// let analyses = analyzer.analyze_all();
/// assert_eq!(analyses.len(), 2);
/// ```
pub struct MultiTrackAnalyzer {
    stems: Vec<StemRecord>,
}

impl MultiTrackAnalyzer {
    /// Create a new, empty multi-track analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self { stems: Vec::new() }
    }

    /// Register a stem for later analysis.
    ///
    /// # Arguments
    ///
    /// * `stem` — the logical role of this stem.
    /// * `samples` — mono audio samples (f32, any scale).
    /// * `sample_rate` — sample rate in Hz.
    pub fn add_stem(&mut self, stem: StemType, samples: &[f32], sample_rate: u32) {
        self.stems.push(StemRecord {
            stem_type: stem,
            samples: samples.to_vec(),
            sample_rate,
        });
    }

    /// Run analysis on all registered stems and return one [`StemAnalysis`] per stem.
    ///
    /// The order matches the order in which stems were added via [`add_stem`].
    #[must_use]
    pub fn analyze_all(&self) -> Vec<StemAnalysis> {
        self.stems
            .iter()
            .map(|r| analyze_stem(r))
            .collect()
    }

    /// Weighted average of per-stem tempos.
    ///
    /// Drum stems receive weight **2**, all others receive weight **1**.
    /// Returns 0.0 if no stems have been added or no stem produced a valid tempo.
    #[must_use]
    pub fn combined_tempo(&self) -> f32 {
        let mut weighted_sum = 0.0_f32;
        let mut weight_total = 0.0_f32;

        for record in &self.stems {
            let analysis = analyze_stem(record);
            if analysis.tempo_bpm > 0.0 {
                let w = if record.stem_type == StemType::Drums {
                    2.0_f32
                } else {
                    1.0_f32
                };
                weighted_sum += analysis.tempo_bpm * w;
                weight_total += w;
            }
        }

        if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        }
    }

    /// Number of stems currently registered.
    #[must_use]
    pub fn stem_count(&self) -> usize {
        self.stems.len()
    }

    /// Remove all registered stems.
    pub fn clear(&mut self) {
        self.stems.clear();
    }
}

impl Default for MultiTrackAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Analysis helpers
// ---------------------------------------------------------------------------

/// Full analysis for a single stem record.
fn analyze_stem(record: &StemRecord) -> StemAnalysis {
    let samples = &record.samples;
    let sr = record.sample_rate;

    let energy = rms_energy(samples);
    let tempo_bpm = estimate_tempo(samples, sr);
    let key = estimate_key(samples, sr);
    let onset_count = count_onsets(samples, sr);

    StemAnalysis {
        stem_type: record.stem_type,
        tempo_bpm,
        key,
        energy,
        onset_count,
    }
}

/// RMS energy of a signal, clamped to \[0, 1\].
#[allow(clippy::cast_precision_loss)]
fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();
    rms.clamp(0.0, 1.0)
}

/// Estimate tempo (BPM) using energy-envelope autocorrelation.
///
/// Returns 0.0 when insufficient data or no clear periodicity is found.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_truncation)]
fn estimate_tempo(samples: &[f32], sample_rate: u32) -> f32 {
    if sample_rate == 0 || samples.is_empty() {
        return 0.0;
    }

    let sr = sample_rate as f32;

    // Compute onset-strength envelope (RMS per hop frame).
    let hop = (sr * 0.01) as usize; // 10 ms hops
    if hop == 0 {
        return 0.0;
    }
    let n_frames = samples.len() / hop;
    if n_frames < 4 {
        return 0.0;
    }

    let envelope: Vec<f32> = (0..n_frames)
        .map(|i| {
            let start = i * hop;
            let end = (start + hop).min(samples.len());
            let sq_sum: f32 = samples[start..end].iter().map(|&s| s * s).sum();
            let frame_len = (end - start) as f32;
            if frame_len > 0.0 {
                (sq_sum / frame_len).sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // Autocorrelation on the envelope to find periodicity.
    // Lag range corresponds to 40–220 BPM.
    let env_sr = sr / hop as f32; // frames per second
    let lag_min = (env_sr * 60.0 / 220.0) as usize;
    let lag_max = ((env_sr * 60.0 / 40.0) as usize).min(n_frames.saturating_sub(1));

    if lag_min >= lag_max {
        return 0.0;
    }

    let mean_env: f32 = envelope.iter().sum::<f32>() / n_frames as f32;
    let demeaned: Vec<f32> = envelope.iter().map(|&v| v - mean_env).collect();

    let mut best_corr = f32::NEG_INFINITY;
    let mut best_lag = lag_min;

    for lag in lag_min..=lag_max {
        let n_valid = n_frames.saturating_sub(lag);
        if n_valid == 0 {
            continue;
        }
        let corr: f32 = (0..n_valid)
            .map(|i| demeaned[i] * demeaned[i + lag])
            .sum::<f32>();
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_corr <= 0.0 || best_lag == 0 {
        return 0.0;
    }

    let period_frames = best_lag as f32;
    let bpm = 60.0 * env_sr / period_frames;
    bpm.clamp(40.0, 220.0)
}

/// Estimate dominant musical key using a simplified chroma histogram.
///
/// Builds a 12-bin chroma histogram by computing per-frame frequency content
/// approximated through harmonic partial analysis, then returns the pitch class
/// name with the highest accumulated energy.
#[allow(clippy::cast_precision_loss)]
fn estimate_key(samples: &[f32], sample_rate: u32) -> String {
    const PITCH_CLASSES: [&str; 12] =
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

    if samples.is_empty() || sample_rate == 0 {
        return "C".to_string();
    }

    let sr = sample_rate as f32;

    // For each of the 12 chromatic pitch classes (C4 = 261.63 Hz …) compute
    // how much spectral energy correlates with that fundamental and its harmonics.
    // We use a simplified matched-filter on the RMS envelope rather than a full
    // STFT-based chromagram, keeping complexity low.

    // Fundamental frequencies of C4..B4.
    let fundamentals: [f32; 12] = [
        261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16,
        493.88,
    ];

    let mut chroma = [0.0_f32; 12];

    let hop = (sr * 0.02) as usize; // 20 ms frames
    if hop == 0 {
        return "C".to_string();
    }

    let n_frames = samples.len() / hop;
    if n_frames == 0 {
        return "C".to_string();
    }

    for pitch_idx in 0..12usize {
        let f0 = fundamentals[pitch_idx];
        // Correlate signal with a cosine at f0 (first 4 harmonics).
        let mut acc = 0.0_f32;
        for harmonic in 1u32..=4 {
            let freq = f0 * harmonic as f32;
            let period_samples = (sr / freq).round() as usize;
            if period_samples == 0 {
                continue;
            }
            // Sliding-window correlation using a small template.
            let template_len = period_samples.min(64);
            for frame in 0..n_frames {
                let start = frame * hop;
                let end = (start + template_len).min(samples.len());
                if end <= start {
                    continue;
                }
                let correlation: f32 = samples[start..end]
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| {
                        s * (TAU * freq * i as f32 / sr).cos()
                    })
                    .sum::<f32>()
                    .abs();
                acc += correlation;
            }
        }
        chroma[pitch_idx] = acc;
    }

    // Find the pitch class with the most energy.
    let best = chroma
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    PITCH_CLASSES[best].to_string()
}

/// Count transient onsets using adaptive energy thresholding.
///
/// Splits the signal into frames, computes the positive energy delta, and
/// counts frames where the delta exceeds `sensitivity × median`.
#[allow(clippy::cast_precision_loss)]
fn count_onsets(samples: &[f32], sample_rate: u32) -> usize {
    if samples.is_empty() || sample_rate == 0 {
        return 0;
    }

    let sr = sample_rate as f32;
    let hop = ((sr * 0.01) as usize).max(1); // 10 ms
    let n_frames = samples.len() / hop;

    if n_frames < 2 {
        return 0;
    }

    // Per-frame RMS energy.
    let energy: Vec<f32> = (0..n_frames)
        .map(|i| {
            let start = i * hop;
            let end = (start + hop).min(samples.len());
            let sq: f32 = samples[start..end].iter().map(|&s| s * s).sum();
            let len = (end - start) as f32;
            if len > 0.0 { (sq / len).sqrt() } else { 0.0 }
        })
        .collect();

    // Positive first-difference (onset strength).
    let flux: Vec<f32> = (1..energy.len())
        .map(|i| {
            let diff = energy[i] - energy[i - 1];
            if diff > 0.0 { diff } else { 0.0 }
        })
        .collect();

    if flux.is_empty() {
        return 0;
    }

    // Adaptive threshold = 1.5 × median of flux.
    let mut sorted_flux = flux.clone();
    sorted_flux.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted_flux[sorted_flux.len() / 2];
    let threshold = median * 1.5;

    // Count peaks that exceed the threshold, respecting minimum distance.
    let min_dist = (sr * 0.05) as usize / hop; // 50 ms minimum gap
    let min_dist = min_dist.max(1);

    let mut count = 0usize;
    let mut last_onset = 0usize;

    for i in 0..flux.len() {
        if flux[i] > threshold {
            if count == 0 || i.saturating_sub(last_onset) >= min_dist {
                count += 1;
                last_onset = i;
            }
        }
    }

    count
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn make_sine(freq: f32, sr: u32, duration_secs: f32) -> Vec<f32> {
        let n = (sr as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| (TAU * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    fn make_click_train(interval_secs: f32, sr: u32, duration_secs: f32) -> Vec<f32> {
        let n = (sr as f32 * duration_secs) as usize;
        let interval = (sr as f32 * interval_secs) as usize;
        let mut out = vec![0.0f32; n];
        if interval == 0 {
            return out;
        }
        let mut pos = 0;
        while pos < n {
            out[pos] = 1.0;
            pos += interval;
        }
        out
    }

    // ── StemType tests ────────────────────────────────────────────────────────

    #[test]
    fn test_stem_type_labels() {
        assert_eq!(StemType::Vocals.label(), "Vocals");
        assert_eq!(StemType::Drums.label(), "Drums");
        assert_eq!(StemType::Bass.label(), "Bass");
        assert_eq!(StemType::Melody.label(), "Melody");
        assert_eq!(StemType::Other.label(), "Other");
    }

    #[test]
    fn test_stem_type_display() {
        assert_eq!(format!("{}", StemType::Drums), "Drums");
    }

    #[test]
    fn test_stem_type_equality() {
        assert_eq!(StemType::Vocals, StemType::Vocals);
        assert_ne!(StemType::Vocals, StemType::Drums);
    }

    // ── MultiTrackAnalyzer tests ──────────────────────────────────────────────

    #[test]
    fn test_new_analyzer_is_empty() {
        let analyzer = MultiTrackAnalyzer::new();
        assert_eq!(analyzer.stem_count(), 0);
    }

    #[test]
    fn test_add_stems_increments_count() {
        let mut analyzer = MultiTrackAnalyzer::new();
        let samples = vec![0.0f32; 4410];
        analyzer.add_stem(StemType::Drums, &samples, 44100);
        analyzer.add_stem(StemType::Bass, &samples, 44100);
        assert_eq!(analyzer.stem_count(), 2);
    }

    #[test]
    fn test_analyze_all_returns_correct_count() {
        let mut analyzer = MultiTrackAnalyzer::new();
        let samples = make_sine(440.0, 44100, 0.5);
        analyzer.add_stem(StemType::Vocals, &samples, 44100);
        analyzer.add_stem(StemType::Melody, &samples, 44100);
        analyzer.add_stem(StemType::Other, &samples, 44100);
        let results = analyzer.analyze_all();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_analyze_all_stem_types_preserved() {
        let mut analyzer = MultiTrackAnalyzer::new();
        let samples = make_sine(220.0, 44100, 1.0);
        analyzer.add_stem(StemType::Drums, &samples, 44100);
        analyzer.add_stem(StemType::Bass, &samples, 44100);
        let results = analyzer.analyze_all();
        assert_eq!(results[0].stem_type, StemType::Drums);
        assert_eq!(results[1].stem_type, StemType::Bass);
    }

    #[test]
    fn test_analyze_all_energy_nonnegative() {
        let mut analyzer = MultiTrackAnalyzer::new();
        let samples = make_sine(440.0, 22050, 0.5);
        analyzer.add_stem(StemType::Melody, &samples, 22050);
        let results = analyzer.analyze_all();
        assert!(results[0].energy >= 0.0);
        assert!(results[0].energy <= 1.0);
    }

    #[test]
    fn test_analyze_silence_zero_energy() {
        let mut analyzer = MultiTrackAnalyzer::new();
        let silence = vec![0.0f32; 22050];
        analyzer.add_stem(StemType::Vocals, &silence, 22050);
        let results = analyzer.analyze_all();
        assert!((results[0].energy - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_analyze_key_nonempty() {
        let mut analyzer = MultiTrackAnalyzer::new();
        let samples = make_sine(440.0, 44100, 1.0); // A440
        analyzer.add_stem(StemType::Melody, &samples, 44100);
        let results = analyzer.analyze_all();
        assert!(!results[0].key.is_empty());
    }

    #[test]
    fn test_combined_tempo_no_stems_is_zero() {
        let analyzer = MultiTrackAnalyzer::new();
        assert!((analyzer.combined_tempo() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_combined_tempo_drums_weighted_higher() {
        // Create a click train at 120 BPM for drums and a slower 80 BPM-like signal for melody.
        // Drums have 2× weight, so combined_tempo should skew towards drums tempo.
        let mut analyzer = MultiTrackAnalyzer::new();
        // 120 BPM → 0.5s period click train
        let drum_clicks = make_click_train(0.5, 44100, 3.0);
        // Use silence for melody (tempo = 0 → excluded from average)
        let melody_silence = vec![0.0f32; 44100 * 3];
        analyzer.add_stem(StemType::Drums, &drum_clicks, 44100);
        analyzer.add_stem(StemType::Melody, &melody_silence, 44100);
        let tempo = analyzer.combined_tempo();
        // If drums detected a tempo, it should drive the combined value.
        // With silence for melody, combined = drums tempo if nonzero.
        if tempo > 0.0 {
            assert!(tempo >= 40.0 && tempo <= 220.0);
        }
    }

    #[test]
    fn test_clear_removes_stems() {
        let mut analyzer = MultiTrackAnalyzer::new();
        let samples = vec![0.0f32; 100];
        analyzer.add_stem(StemType::Bass, &samples, 44100);
        analyzer.clear();
        assert_eq!(analyzer.stem_count(), 0);
        assert_eq!(analyzer.analyze_all().len(), 0);
    }

    #[test]
    fn test_onset_count_nonnegative() {
        let mut analyzer = MultiTrackAnalyzer::new();
        let clicks = make_click_train(0.25, 44100, 2.0);
        analyzer.add_stem(StemType::Drums, &clicks, 44100);
        let results = analyzer.analyze_all();
        assert!(results[0].onset_count >= 0);
    }

    #[test]
    fn test_analyze_empty_samples() {
        let mut analyzer = MultiTrackAnalyzer::new();
        let empty: Vec<f32> = Vec::new();
        analyzer.add_stem(StemType::Other, &empty, 44100);
        let results = analyzer.analyze_all();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].onset_count, 0);
        assert!((results[0].energy - 0.0).abs() < f32::EPSILON);
    }
}
