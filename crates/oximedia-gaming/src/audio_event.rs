//! Game event detection from audio cues.
//!
//! Detects game events by analyzing audio features such as amplitude spikes
//! and frequency signature matching. Configurable event patterns allow
//! mapping audio characteristics to game events.

use std::time::Duration;

// ---------------------------------------------------------------------------
// AudioEventType
// ---------------------------------------------------------------------------

/// Types of events that can be detected from audio.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioEventType {
    /// A sudden amplitude spike (e.g. explosion, gunshot).
    AmplitudeSpike,
    /// A frequency signature match (e.g. specific game sound effect).
    FrequencyMatch {
        /// Name of the matched pattern.
        pattern_name: String,
    },
    /// Sustained loud audio above a threshold.
    SustainedLoud,
    /// A sudden silence (e.g. dramatic moment, death screen).
    SuddenSilence,
    /// Custom event from pattern matching.
    Custom(String),
}

impl AudioEventType {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::AmplitudeSpike => "Amplitude Spike",
            Self::FrequencyMatch { .. } => "Frequency Match",
            Self::SustainedLoud => "Sustained Loud",
            Self::SuddenSilence => "Sudden Silence",
            Self::Custom(name) => name.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// AudioEventPattern
// ---------------------------------------------------------------------------

/// A pattern that defines how to detect a specific audio event.
#[derive(Debug, Clone)]
pub struct AudioEventPattern {
    /// Name of this pattern (used for identification).
    pub name: String,
    /// The type of detection to perform.
    pub detection: DetectionMode,
    /// Minimum interval between detections of this pattern.
    pub cooldown: Duration,
    /// Whether this pattern is enabled.
    pub enabled: bool,
}

/// The detection mode for an audio event pattern.
#[derive(Debug, Clone)]
pub enum DetectionMode {
    /// Detect when RMS amplitude exceeds a threshold.
    ///
    /// `threshold` is in the range 0.0-1.0 (normalized).
    AmplitudeThreshold {
        /// Amplitude threshold (0.0-1.0).
        threshold: f32,
    },
    /// Detect when amplitude drops below a threshold for a given duration.
    SilenceDetection {
        /// Silence threshold (0.0-1.0).
        threshold: f32,
        /// Minimum silence duration to trigger.
        min_duration: Duration,
    },
    /// Detect a specific frequency signature.
    ///
    /// The `center_frequency_hz` and `bandwidth_hz` define a band-pass region.
    /// An event fires when the energy in this band exceeds the `energy_threshold`.
    FrequencySignature {
        /// Center frequency in Hz.
        center_frequency_hz: f32,
        /// Bandwidth in Hz around the center.
        bandwidth_hz: f32,
        /// Energy threshold (0.0-1.0) for the band.
        energy_threshold: f32,
    },
    /// Detect when amplitude exceeds a threshold for a sustained period.
    SustainedAmplitude {
        /// Amplitude threshold (0.0-1.0).
        threshold: f32,
        /// Minimum duration of sustained amplitude.
        min_duration: Duration,
    },
}

impl AudioEventPattern {
    /// Create an amplitude spike detection pattern.
    #[must_use]
    pub fn amplitude_spike(name: &str, threshold: f32) -> Self {
        Self {
            name: name.to_string(),
            detection: DetectionMode::AmplitudeThreshold { threshold },
            cooldown: Duration::from_millis(500),
            enabled: true,
        }
    }

    /// Create a frequency signature detection pattern.
    #[must_use]
    pub fn frequency_signature(
        name: &str,
        center_hz: f32,
        bandwidth_hz: f32,
        energy_threshold: f32,
    ) -> Self {
        Self {
            name: name.to_string(),
            detection: DetectionMode::FrequencySignature {
                center_frequency_hz: center_hz,
                bandwidth_hz,
                energy_threshold,
            },
            cooldown: Duration::from_secs(1),
            enabled: true,
        }
    }

    /// Create a silence detection pattern.
    #[must_use]
    pub fn silence(name: &str, threshold: f32, min_duration: Duration) -> Self {
        Self {
            name: name.to_string(),
            detection: DetectionMode::SilenceDetection {
                threshold,
                min_duration,
            },
            cooldown: Duration::from_secs(2),
            enabled: true,
        }
    }

    /// Create a sustained amplitude pattern.
    #[must_use]
    pub fn sustained(name: &str, threshold: f32, min_duration: Duration) -> Self {
        Self {
            name: name.to_string(),
            detection: DetectionMode::SustainedAmplitude {
                threshold,
                min_duration,
            },
            cooldown: Duration::from_secs(1),
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// DetectedAudioEvent
// ---------------------------------------------------------------------------

/// An audio event that has been detected.
#[derive(Debug, Clone)]
pub struct DetectedAudioEvent {
    /// The type of event detected.
    pub event_type: AudioEventType,
    /// Timestamp within the audio stream when the event was detected.
    pub timestamp: Duration,
    /// Confidence score (0.0-1.0).
    pub confidence: f32,
    /// The measured amplitude / energy that triggered the detection.
    pub measured_value: f32,
    /// Name of the pattern that matched.
    pub pattern_name: String,
}

// ---------------------------------------------------------------------------
// AudioAnalysisFrame
// ---------------------------------------------------------------------------

/// Audio analysis data for a single analysis window.
#[derive(Debug, Clone)]
pub struct AudioAnalysisFrame {
    /// Timestamp of this analysis frame.
    pub timestamp: Duration,
    /// RMS amplitude (0.0-1.0, normalized).
    pub rms_amplitude: f32,
    /// Peak amplitude in this window.
    pub peak_amplitude: f32,
    /// Spectral energy per band (if computed).
    /// Each entry is (center_freq_hz, energy_0_to_1).
    pub spectral_bands: Vec<(f32, f32)>,
}

// ---------------------------------------------------------------------------
// AudioEventDetector
// ---------------------------------------------------------------------------

/// Detects game events from audio analysis frames by matching against
/// configured patterns.
pub struct AudioEventDetector {
    patterns: Vec<AudioEventPattern>,
    /// Last detection time per pattern (index-aligned with `patterns`).
    last_detection: Vec<Option<Duration>>,
    /// Accumulated silence duration per pattern index (for silence detection).
    silence_accum: Vec<Duration>,
    /// Accumulated sustained loud duration per pattern index.
    sustained_accum: Vec<Duration>,
    /// All detected events.
    events: Vec<DetectedAudioEvent>,
    /// Previous analysis frame timestamp.
    prev_timestamp: Option<Duration>,
}

impl AudioEventDetector {
    /// Create a new detector with no patterns.
    #[must_use]
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            last_detection: Vec::new(),
            silence_accum: Vec::new(),
            sustained_accum: Vec::new(),
            events: Vec::new(),
            prev_timestamp: None,
        }
    }

    /// Add a detection pattern.
    pub fn add_pattern(&mut self, pattern: AudioEventPattern) {
        self.patterns.push(pattern);
        self.last_detection.push(None);
        self.silence_accum.push(Duration::ZERO);
        self.sustained_accum.push(Duration::ZERO);
    }

    /// Number of configured patterns.
    #[must_use]
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Process an audio analysis frame and detect events.
    ///
    /// Returns newly detected events from this frame.
    pub fn process_frame(&mut self, frame: &AudioAnalysisFrame) -> Vec<DetectedAudioEvent> {
        let mut new_events = Vec::new();
        let dt = self
            .prev_timestamp
            .map(|prev| frame.timestamp.saturating_sub(prev))
            .unwrap_or(Duration::ZERO);
        self.prev_timestamp = Some(frame.timestamp);

        for i in 0..self.patterns.len() {
            if !self.patterns[i].enabled {
                continue;
            }

            // Check cooldown
            if let Some(last) = self.last_detection[i] {
                if frame.timestamp.saturating_sub(last) < self.patterns[i].cooldown {
                    continue;
                }
            }

            if let Some(event) = self.check_pattern(i, frame, dt) {
                self.last_detection[i] = Some(frame.timestamp);
                new_events.push(event);
            }
        }

        self.events.extend(new_events.clone());
        new_events
    }

    /// Check a single pattern against the frame.
    fn check_pattern(
        &mut self,
        index: usize,
        frame: &AudioAnalysisFrame,
        dt: Duration,
    ) -> Option<DetectedAudioEvent> {
        let pattern = &self.patterns[index];

        match &pattern.detection {
            DetectionMode::AmplitudeThreshold { threshold } => {
                if frame.rms_amplitude >= *threshold {
                    Some(DetectedAudioEvent {
                        event_type: AudioEventType::AmplitudeSpike,
                        timestamp: frame.timestamp,
                        confidence: (frame.rms_amplitude / threshold).min(1.0),
                        measured_value: frame.rms_amplitude,
                        pattern_name: pattern.name.clone(),
                    })
                } else {
                    None
                }
            }
            DetectionMode::SilenceDetection {
                threshold,
                min_duration,
            } => {
                if frame.rms_amplitude < *threshold {
                    self.silence_accum[index] += dt;
                    if self.silence_accum[index] >= *min_duration {
                        self.silence_accum[index] = Duration::ZERO;
                        return Some(DetectedAudioEvent {
                            event_type: AudioEventType::SuddenSilence,
                            timestamp: frame.timestamp,
                            confidence: 1.0 - (frame.rms_amplitude / threshold),
                            measured_value: frame.rms_amplitude,
                            pattern_name: pattern.name.clone(),
                        });
                    }
                } else {
                    self.silence_accum[index] = Duration::ZERO;
                }
                None
            }
            DetectionMode::FrequencySignature {
                center_frequency_hz,
                bandwidth_hz,
                energy_threshold,
            } => {
                let low = center_frequency_hz - bandwidth_hz / 2.0;
                let high = center_frequency_hz + bandwidth_hz / 2.0;

                let band_energy: f32 = frame
                    .spectral_bands
                    .iter()
                    .filter(|(freq, _)| *freq >= low && *freq <= high)
                    .map(|(_, energy)| *energy)
                    .sum::<f32>();

                let band_count = frame
                    .spectral_bands
                    .iter()
                    .filter(|(freq, _)| *freq >= low && *freq <= high)
                    .count();

                let avg_energy = if band_count > 0 {
                    band_energy / band_count as f32
                } else {
                    0.0
                };

                if avg_energy >= *energy_threshold {
                    Some(DetectedAudioEvent {
                        event_type: AudioEventType::FrequencyMatch {
                            pattern_name: pattern.name.clone(),
                        },
                        timestamp: frame.timestamp,
                        confidence: (avg_energy / energy_threshold).min(1.0),
                        measured_value: avg_energy,
                        pattern_name: pattern.name.clone(),
                    })
                } else {
                    None
                }
            }
            DetectionMode::SustainedAmplitude {
                threshold,
                min_duration,
            } => {
                if frame.rms_amplitude >= *threshold {
                    self.sustained_accum[index] += dt;
                    if self.sustained_accum[index] >= *min_duration {
                        self.sustained_accum[index] = Duration::ZERO;
                        return Some(DetectedAudioEvent {
                            event_type: AudioEventType::SustainedLoud,
                            timestamp: frame.timestamp,
                            confidence: (frame.rms_amplitude / threshold).min(1.0),
                            measured_value: frame.rms_amplitude,
                            pattern_name: pattern.name.clone(),
                        });
                    }
                } else {
                    self.sustained_accum[index] = Duration::ZERO;
                }
                None
            }
        }
    }

    /// All detected events so far.
    #[must_use]
    pub fn events(&self) -> &[DetectedAudioEvent] {
        &self.events
    }

    /// Number of detected events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Clear all detected events.
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Reset all detector state (cooldowns, accumulators, events).
    pub fn reset(&mut self) {
        self.events.clear();
        self.prev_timestamp = None;
        for ld in &mut self.last_detection {
            *ld = None;
        }
        for sa in &mut self.silence_accum {
            *sa = Duration::ZERO;
        }
        for su in &mut self.sustained_accum {
            *su = Duration::ZERO;
        }
    }
}

impl Default for AudioEventDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility: compute RMS from f32 samples
// ---------------------------------------------------------------------------

/// Compute the RMS (root mean square) amplitude from a slice of f32 samples.
///
/// Returns 0.0 for empty input.
#[must_use]
pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// Compute the peak absolute amplitude from a slice of f32 samples.
#[must_use]
pub fn compute_peak(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(timestamp_ms: u64, rms: f32, peak: f32) -> AudioAnalysisFrame {
        AudioAnalysisFrame {
            timestamp: Duration::from_millis(timestamp_ms),
            rms_amplitude: rms,
            peak_amplitude: peak,
            spectral_bands: Vec::new(),
        }
    }

    fn make_frame_with_bands(
        timestamp_ms: u64,
        rms: f32,
        bands: Vec<(f32, f32)>,
    ) -> AudioAnalysisFrame {
        AudioAnalysisFrame {
            timestamp: Duration::from_millis(timestamp_ms),
            rms_amplitude: rms,
            peak_amplitude: rms,
            spectral_bands: bands,
        }
    }

    // -- AudioEventType --

    #[test]
    fn test_event_type_labels() {
        assert_eq!(AudioEventType::AmplitudeSpike.label(), "Amplitude Spike");
        assert_eq!(
            AudioEventType::FrequencyMatch {
                pattern_name: "test".into()
            }
            .label(),
            "Frequency Match"
        );
        assert_eq!(AudioEventType::SustainedLoud.label(), "Sustained Loud");
        assert_eq!(AudioEventType::SuddenSilence.label(), "Sudden Silence");
        assert_eq!(AudioEventType::Custom("boom".into()).label(), "boom");
    }

    // -- AudioEventPattern --

    #[test]
    fn test_amplitude_spike_pattern() {
        let p = AudioEventPattern::amplitude_spike("gunshot", 0.8);
        assert_eq!(p.name, "gunshot");
        assert!(p.enabled);
        assert!(matches!(
            p.detection,
            DetectionMode::AmplitudeThreshold { .. }
        ));
    }

    #[test]
    fn test_frequency_signature_pattern() {
        let p = AudioEventPattern::frequency_signature("bell", 1000.0, 200.0, 0.5);
        assert_eq!(p.name, "bell");
        assert!(matches!(
            p.detection,
            DetectionMode::FrequencySignature { .. }
        ));
    }

    #[test]
    fn test_silence_pattern() {
        let p = AudioEventPattern::silence("death_screen", 0.05, Duration::from_secs(1));
        assert_eq!(p.name, "death_screen");
        assert!(matches!(
            p.detection,
            DetectionMode::SilenceDetection { .. }
        ));
    }

    #[test]
    fn test_sustained_pattern() {
        let p = AudioEventPattern::sustained("battle", 0.6, Duration::from_secs(2));
        assert_eq!(p.name, "battle");
        assert!(matches!(
            p.detection,
            DetectionMode::SustainedAmplitude { .. }
        ));
    }

    // -- AudioEventDetector --

    #[test]
    fn test_detector_creation() {
        let d = AudioEventDetector::new();
        assert_eq!(d.pattern_count(), 0);
        assert_eq!(d.event_count(), 0);
    }

    #[test]
    fn test_detector_default() {
        let d = AudioEventDetector::default();
        assert_eq!(d.pattern_count(), 0);
    }

    #[test]
    fn test_add_pattern() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::amplitude_spike("test", 0.5));
        assert_eq!(d.pattern_count(), 1);
    }

    #[test]
    fn test_amplitude_spike_detection() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::amplitude_spike("explosion", 0.7));

        // Below threshold
        let events = d.process_frame(&make_frame(0, 0.3, 0.5));
        assert!(events.is_empty());

        // Above threshold
        let events = d.process_frame(&make_frame(100, 0.9, 0.95));
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].event_type,
            AudioEventType::AmplitudeSpike
        ));
        assert_eq!(events[0].pattern_name, "explosion");
    }

    #[test]
    fn test_cooldown_prevents_rapid_fire() {
        let mut d = AudioEventDetector::new();
        let mut p = AudioEventPattern::amplitude_spike("gun", 0.5);
        p.cooldown = Duration::from_millis(500);
        d.add_pattern(p);

        // First detection
        let e1 = d.process_frame(&make_frame(0, 0.8, 0.9));
        assert_eq!(e1.len(), 1);

        // Within cooldown - should not fire
        let e2 = d.process_frame(&make_frame(200, 0.8, 0.9));
        assert!(e2.is_empty());

        // After cooldown
        let e3 = d.process_frame(&make_frame(600, 0.8, 0.9));
        assert_eq!(e3.len(), 1);
    }

    #[test]
    fn test_silence_detection() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::silence(
            "death",
            0.05,
            Duration::from_millis(200),
        ));

        // Start with silence
        let e1 = d.process_frame(&make_frame(0, 0.01, 0.02));
        assert!(e1.is_empty()); // dt is 0

        let e2 = d.process_frame(&make_frame(100, 0.01, 0.02));
        assert!(e2.is_empty()); // 100ms < 200ms

        let e3 = d.process_frame(&make_frame(250, 0.01, 0.02));
        assert_eq!(e3.len(), 1);
        assert!(matches!(e3[0].event_type, AudioEventType::SuddenSilence));
    }

    #[test]
    fn test_silence_reset_on_loud() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::silence(
            "death",
            0.05,
            Duration::from_millis(200),
        ));

        d.process_frame(&make_frame(0, 0.01, 0.02));
        d.process_frame(&make_frame(100, 0.01, 0.02));
        // Loud frame resets accumulator
        d.process_frame(&make_frame(150, 0.5, 0.6));
        // Continue silence - should need full 200ms again
        let e = d.process_frame(&make_frame(300, 0.01, 0.02));
        assert!(e.is_empty());
    }

    #[test]
    fn test_frequency_signature_detection() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::frequency_signature(
            "bell", 1000.0, 200.0, 0.6,
        ));

        // Bands within the target range
        let bands = vec![(900.0, 0.7), (1000.0, 0.8), (1100.0, 0.7)];
        let events = d.process_frame(&make_frame_with_bands(0, 0.5, bands));
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].event_type,
            AudioEventType::FrequencyMatch { .. }
        ));
    }

    #[test]
    fn test_frequency_no_match_below_threshold() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::frequency_signature(
            "bell", 1000.0, 200.0, 0.6,
        ));

        let bands = vec![(900.0, 0.2), (1000.0, 0.3), (1100.0, 0.2)];
        let events = d.process_frame(&make_frame_with_bands(0, 0.5, bands));
        assert!(events.is_empty());
    }

    #[test]
    fn test_sustained_amplitude_detection() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::sustained(
            "battle",
            0.5,
            Duration::from_millis(200),
        ));

        d.process_frame(&make_frame(0, 0.7, 0.8));
        let e1 = d.process_frame(&make_frame(100, 0.7, 0.8));
        assert!(e1.is_empty()); // only 100ms

        let e2 = d.process_frame(&make_frame(250, 0.7, 0.8));
        assert_eq!(e2.len(), 1);
        assert!(matches!(e2[0].event_type, AudioEventType::SustainedLoud));
    }

    #[test]
    fn test_sustained_reset_on_quiet() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::sustained(
            "battle",
            0.5,
            Duration::from_millis(300),
        ));

        d.process_frame(&make_frame(0, 0.7, 0.8));
        d.process_frame(&make_frame(100, 0.7, 0.8));
        d.process_frame(&make_frame(200, 0.1, 0.1)); // quiet resets accumulator
                                                     // Only 100ms of loud after reset -- not enough for 300ms threshold
        let e = d.process_frame(&make_frame(300, 0.7, 0.8));
        assert!(e.is_empty()); // not enough sustained time after reset
    }

    #[test]
    fn test_disabled_pattern_not_checked() {
        let mut d = AudioEventDetector::new();
        let mut p = AudioEventPattern::amplitude_spike("gun", 0.5);
        p.enabled = false;
        d.add_pattern(p);

        let events = d.process_frame(&make_frame(0, 0.9, 0.95));
        assert!(events.is_empty());
    }

    #[test]
    fn test_multiple_patterns() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::amplitude_spike("explosion", 0.8));
        d.add_pattern(AudioEventPattern::amplitude_spike("gunshot", 0.5));

        let events = d.process_frame(&make_frame(0, 0.9, 0.95));
        // Both should fire
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_clear_events() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::amplitude_spike("test", 0.5));
        d.process_frame(&make_frame(0, 0.8, 0.9));
        assert_eq!(d.event_count(), 1);

        d.clear_events();
        assert_eq!(d.event_count(), 0);
    }

    #[test]
    fn test_reset() {
        let mut d = AudioEventDetector::new();
        d.add_pattern(AudioEventPattern::amplitude_spike("test", 0.5));
        d.process_frame(&make_frame(0, 0.8, 0.9));
        d.reset();
        assert_eq!(d.event_count(), 0);

        // After reset, should be able to detect again immediately
        let events = d.process_frame(&make_frame(0, 0.8, 0.9));
        assert_eq!(events.len(), 1);
    }

    // -- Utility functions --

    #[test]
    fn test_compute_rms_empty() {
        assert!((compute_rms(&[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_rms_values() {
        let samples = [0.5, -0.5, 0.5, -0.5];
        let rms = compute_rms(&samples);
        assert!((rms - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_compute_rms_silence() {
        let samples = [0.0; 100];
        assert!((compute_rms(&samples) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_peak_empty() {
        assert!((compute_peak(&[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_peak_values() {
        let samples = [0.1, -0.8, 0.3, -0.2];
        assert!((compute_peak(&samples) - 0.8).abs() < 0.001);
    }
}
