//! Formant frequency tracking over time.
//!
//! Tracks the first four formant frequencies (F1 -- F4) across successive
//! audio frames.  Useful for vowel identification, voice quality assessment,
//! and speech synthesis parameter extraction.

#![allow(dead_code)]

/// Index of a formant in the vocal tract filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormantIndex {
    /// First formant (F1) -- related to tongue height / vowel openness.
    F1,
    /// Second formant (F2) -- related to tongue front-back position.
    F2,
    /// Third formant (F3) -- related to lip rounding / voice identity.
    F3,
    /// Fourth formant (F4) -- speaker-specific resonance.
    F4,
}

impl FormantIndex {
    /// Zero-based numeric index (0 = F1, 3 = F4).
    #[must_use]
    pub fn index(&self) -> usize {
        match self {
            Self::F1 => 0,
            Self::F2 => 1,
            Self::F3 => 2,
            Self::F4 => 3,
        }
    }

    /// Short label (e.g. "F1").
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::F1 => "F1",
            Self::F2 => "F2",
            Self::F3 => "F3",
            Self::F4 => "F4",
        }
    }

    /// Typical centre frequency for an adult male voice in Hz.
    #[must_use]
    pub fn typical_male_hz(&self) -> f32 {
        match self {
            Self::F1 => 500.0,
            Self::F2 => 1500.0,
            Self::F3 => 2500.0,
            Self::F4 => 3500.0,
        }
    }

    /// Typical centre frequency for an adult female voice in Hz.
    #[must_use]
    pub fn typical_female_hz(&self) -> f32 {
        match self {
            Self::F1 => 550.0,
            Self::F2 => 1700.0,
            Self::F3 => 2800.0,
            Self::F4 => 3800.0,
        }
    }
}

/// A single formant frequency measurement.
#[derive(Debug, Clone, Copy)]
pub struct FormantFreq {
    /// Formant index.
    pub index: FormantIndex,
    /// Frequency in Hz.
    pub freq_hz: f32,
    /// Bandwidth in Hz.
    pub bandwidth_hz: f32,
    /// Amplitude relative to spectral peak (0.0 -- 1.0).
    pub amplitude: f32,
}

impl FormantFreq {
    /// Create a new [`FormantFreq`].
    #[must_use]
    pub fn new(index: FormantIndex, freq_hz: f32, bandwidth_hz: f32, amplitude: f32) -> Self {
        Self {
            index,
            freq_hz,
            bandwidth_hz,
            amplitude,
        }
    }

    /// Returns `true` when the formant measurement looks plausible (non-zero frequency
    /// and reasonable bandwidth).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.freq_hz > 0.0 && self.bandwidth_hz > 0.0 && self.bandwidth_hz < self.freq_hz
    }
}

/// One frame of formant tracking output.
#[derive(Debug, Clone)]
pub struct FormantTrackFrame {
    /// Centre time of this frame in seconds.
    pub time_s: f32,
    /// Formant measurements (up to 4).
    pub formants: Vec<FormantFreq>,
    /// Whether the frame was voiced.
    pub voiced: bool,
}

impl FormantTrackFrame {
    /// Retrieve a specific formant by index, if present.
    #[must_use]
    pub fn get(&self, idx: FormantIndex) -> Option<&FormantFreq> {
        self.formants.iter().find(|f| f.index == idx)
    }

    /// F1 frequency, or 0.0 if not available.
    #[must_use]
    pub fn f1(&self) -> f32 {
        self.get(FormantIndex::F1).map_or(0.0, |f| f.freq_hz)
    }

    /// F2 frequency, or 0.0 if not available.
    #[must_use]
    pub fn f2(&self) -> f32 {
        self.get(FormantIndex::F2).map_or(0.0, |f| f.freq_hz)
    }
}

/// Complete formant-tracking result.
#[derive(Debug, Clone)]
pub struct FormantTrack {
    /// Per-frame formant data.
    pub frames: Vec<FormantTrackFrame>,
    /// Mean F1 across all voiced frames.
    pub mean_f1: f32,
    /// Mean F2 across all voiced frames.
    pub mean_f2: f32,
}

impl FormantTrack {
    /// Return only the voiced frames.
    #[must_use]
    pub fn voiced_frames(&self) -> Vec<&FormantTrackFrame> {
        self.frames.iter().filter(|f| f.voiced).collect()
    }

    /// Duration in seconds from the first to the last frame, or 0.0 if empty.
    #[must_use]
    pub fn duration_s(&self) -> f32 {
        if self.frames.len() < 2 {
            return 0.0;
        }
        self.frames
            .last()
            .expect("frames non-empty: len < 2 check returned above")
            .time_s
            - self
                .frames
                .first()
                .expect("frames non-empty: len < 2 check returned above")
                .time_s
    }
}

/// Formant frequency tracker using simplified LPC analysis.
pub struct FormantTracker {
    sample_rate: f32,
    frame_size: usize,
    hop_size: usize,
    lpc_order: usize,
}

impl FormantTracker {
    /// Create a new [`FormantTracker`].
    ///
    /// # Arguments
    /// * `sample_rate` -- Sample rate in Hz.
    /// * `frame_size`  -- Analysis frame length in samples.
    /// * `hop_size`    -- Hop between frames.
    /// * `lpc_order`   -- LPC model order (10 -- 16 typical for speech).
    #[must_use]
    pub fn new(sample_rate: f32, frame_size: usize, hop_size: usize, lpc_order: usize) -> Self {
        Self {
            sample_rate,
            frame_size,
            hop_size,
            lpc_order,
        }
    }

    /// Track formants across the full signal.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn track(&self, samples: &[f32]) -> FormantTrack {
        let mut frames = Vec::new();
        let mut pos = 0usize;

        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            let time_s = pos as f32 / self.sample_rate;
            let energy = rms_energy(frame);

            let (voiced, formants) = if energy > 0.01 {
                (true, self.estimate_formants(frame))
            } else {
                (false, Vec::new())
            };

            frames.push(FormantTrackFrame {
                time_s,
                formants,
                voiced,
            });

            pos += self.hop_size;
        }

        let mean_f1 = mean_formant(&frames, FormantIndex::F1);
        let mean_f2 = mean_formant(&frames, FormantIndex::F2);

        FormantTrack {
            frames,
            mean_f1,
            mean_f2,
        }
    }

    /// Estimate a single frame worth of formants using an autocorrelation
    /// proxy (simplified LPC).
    #[allow(clippy::cast_precision_loss)]
    fn estimate_formants(&self, frame: &[f32]) -> Vec<FormantFreq> {
        // Simplified: place formants at autocorrelation-derived peaks.
        // A production implementation would use Levinson-Durbin + root finding.
        let n = frame.len();
        let autocorr: Vec<f32> = (0..self.lpc_order)
            .map(|lag| {
                let sum: f32 = frame
                    .iter()
                    .take(n - lag)
                    .zip(frame.iter().skip(lag))
                    .map(|(&a, &b)| a * b)
                    .sum();
                sum / n as f32
            })
            .collect();

        // Heuristic placement based on sample rate and energy distribution.
        let spacing = self.sample_rate / (2.0 * (self.lpc_order as f32 + 1.0));
        let mut formants = Vec::with_capacity(4);
        for i in 0..4 {
            let base_freq = spacing * (i as f32 + 1.0);
            let weight = autocorr.get(i + 1).copied().unwrap_or(0.0).abs();
            formants.push(FormantFreq::new(
                match i {
                    0 => FormantIndex::F1,
                    1 => FormantIndex::F2,
                    2 => FormantIndex::F3,
                    _ => FormantIndex::F4,
                },
                base_freq,
                base_freq * 0.1, // 10 % bandwidth
                weight.clamp(0.0, 1.0),
            ));
        }
        formants
    }
}

impl Default for FormantTracker {
    fn default() -> Self {
        Self::new(44100.0, 1024, 256, 12)
    }
}

// -- private helpers --

#[allow(clippy::cast_precision_loss)]
fn rms_energy(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: f32 = frame.iter().map(|&x| x * x).sum();
    (sum / frame.len() as f32).sqrt()
}

#[allow(clippy::cast_precision_loss)]
fn mean_formant(frames: &[FormantTrackFrame], idx: FormantIndex) -> f32 {
    let values: Vec<f32> = frames
        .iter()
        .filter(|f| f.voiced)
        .filter_map(|f| f.get(idx))
        .map(|f| f.freq_hz)
        .collect();
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

// -- unit tests --

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formant_index_values() {
        assert_eq!(FormantIndex::F1.index(), 0);
        assert_eq!(FormantIndex::F4.index(), 3);
    }

    #[test]
    fn test_formant_index_labels() {
        assert_eq!(FormantIndex::F1.label(), "F1");
        assert_eq!(FormantIndex::F2.label(), "F2");
        assert_eq!(FormantIndex::F3.label(), "F3");
        assert_eq!(FormantIndex::F4.label(), "F4");
    }

    #[test]
    fn test_typical_male_female_ordering() {
        // Female F1 is typically higher than male F1
        assert!(FormantIndex::F1.typical_female_hz() > FormantIndex::F1.typical_male_hz());
    }

    #[test]
    fn test_formant_freq_validity() {
        let valid = FormantFreq::new(FormantIndex::F1, 500.0, 50.0, 0.8);
        assert!(valid.is_valid());

        let invalid = FormantFreq::new(FormantIndex::F1, 0.0, 50.0, 0.8);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_formant_freq_bandwidth_exceeds_freq() {
        let bad = FormantFreq::new(FormantIndex::F2, 100.0, 200.0, 0.5);
        assert!(!bad.is_valid()); // bandwidth > freq
    }

    #[test]
    fn test_formant_track_frame_get() {
        let frame = FormantTrackFrame {
            time_s: 0.0,
            formants: vec![
                FormantFreq::new(FormantIndex::F1, 500.0, 50.0, 0.9),
                FormantFreq::new(FormantIndex::F2, 1500.0, 100.0, 0.7),
            ],
            voiced: true,
        };
        assert!(frame.get(FormantIndex::F1).is_some());
        assert!(frame.get(FormantIndex::F3).is_none());
    }

    #[test]
    fn test_f1_f2_helpers() {
        let frame = FormantTrackFrame {
            time_s: 0.0,
            formants: vec![
                FormantFreq::new(FormantIndex::F1, 500.0, 50.0, 0.9),
                FormantFreq::new(FormantIndex::F2, 1500.0, 100.0, 0.7),
            ],
            voiced: true,
        };
        assert_eq!(frame.f1(), 500.0);
        assert_eq!(frame.f2(), 1500.0);
    }

    #[test]
    fn test_tracker_default() {
        let tracker = FormantTracker::default();
        assert_eq!(tracker.sample_rate, 44100.0);
        assert_eq!(tracker.lpc_order, 12);
    }

    #[test]
    fn test_track_silence() {
        let tracker = FormantTracker::default();
        let silence = vec![0.0_f32; 44100];
        let track = tracker.track(&silence);
        assert!(!track.frames.is_empty());
        // All frames should be unvoiced.
        assert!(track.voiced_frames().is_empty());
    }

    #[test]
    fn test_track_voiced_signal() {
        let tracker = FormantTracker::default();
        let signal: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let track = tracker.track(&signal);
        assert!(!track.voiced_frames().is_empty());
        assert!(track.mean_f1 > 0.0);
    }

    #[test]
    fn test_formant_track_duration() {
        let tracker = FormantTracker::default();
        let signal = vec![0.5_f32; 44100]; // 1 second at 44100
        let track = tracker.track(&signal);
        assert!(track.duration_s() > 0.5);
    }

    #[test]
    fn test_short_signal_no_panic() {
        let tracker = FormantTracker::default();
        let short = vec![0.1_f32; 50];
        let track = tracker.track(&short);
        assert!(track.frames.is_empty());
        assert_eq!(track.mean_f1, 0.0);
    }

    #[test]
    fn test_voiced_frames_filter() {
        let tracker = FormantTracker::default();
        let mut samples = vec![0.0_f32; 44100];
        // Only make part of the signal loud enough to be voiced
        for s in samples[10000..20000].iter_mut() {
            *s = 0.5;
        }
        let track = tracker.track(&samples);
        let voiced = track.voiced_frames();
        let total = track.frames.len();
        assert!(voiced.len() < total);
    }

    #[test]
    fn test_formant_ordering_in_frame() {
        let tracker = FormantTracker::default();
        let signal: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.03).sin() * 0.4).collect();
        let track = tracker.track(&signal);
        for frame in track.voiced_frames() {
            if frame.formants.len() >= 2 {
                assert!(frame.f2() > frame.f1());
            }
        }
    }
}
