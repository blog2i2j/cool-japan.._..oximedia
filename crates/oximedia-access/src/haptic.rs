//! Haptic feedback description generation for touch-enabled devices.
//!
//! This module translates media events, audio cues, and visual information
//! into haptic feedback patterns that can be rendered on touch-enabled devices.
//! It is designed to improve accessibility for deaf-blind users and enhance
//! immersion for hearing-impaired users who interact with media on tactile
//! devices.
//!
//! # Haptic Patterns
//!
//! Haptic feedback is described as a sequence of [`HapticPulse`] events,
//! each specifying intensity, duration, and optional waveform classification.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_access::haptic::{HapticDescriptionGenerator, MediaEvent};
//!
//! let gen = HapticDescriptionGenerator::new();
//! let pattern = gen.from_event(MediaEvent::BeatOnset { intensity: 0.8 });
//! assert!(!pattern.pulses.is_empty());
//! ```

use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Haptic pulse / waveform types
// ──────────────────────────────────────────────────────────────────────────────

/// Classification of a haptic waveform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HapticWaveform {
    /// Single sharp tap (very short transient, < 20 ms).
    Click,
    /// Double-tap (two clicks in rapid succession).
    DoubleTap,
    /// Continuous vibration ramp-up.
    Rise,
    /// Continuous vibration ramp-down.
    Fall,
    /// Long steady buzz.
    Buzz,
    /// Rhythmic on-off pulse train.
    Pulse,
    /// Soft thud (low-frequency transient).
    Thud,
    /// Custom (implementation-defined).
    Custom,
}

impl HapticWaveform {
    /// Typical duration in milliseconds for this waveform class.
    #[must_use]
    pub fn typical_duration_ms(&self) -> u32 {
        match self {
            Self::Click => 10,
            Self::DoubleTap => 30,
            Self::Rise => 200,
            Self::Fall => 200,
            Self::Buzz => 500,
            Self::Pulse => 100,
            Self::Thud => 50,
            Self::Custom => 100,
        }
    }
}

/// A single haptic pulse event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HapticPulse {
    /// Start time in milliseconds relative to the pattern's reference point.
    pub start_ms: u64,
    /// Duration in milliseconds.
    pub duration_ms: u32,
    /// Normalised intensity in [0.0, 1.0].
    pub intensity: f32,
    /// Waveform shape.
    pub waveform: HapticWaveform,
    /// Optional human-readable description.
    pub description: Option<String>,
}

impl HapticPulse {
    /// Create a new pulse at the given start time.
    #[must_use]
    pub fn new(start_ms: u64, duration_ms: u32, intensity: f32, waveform: HapticWaveform) -> Self {
        Self {
            start_ms,
            duration_ms,
            intensity: intensity.clamp(0.0, 1.0),
            waveform,
            description: None,
        }
    }

    /// Attach a human-readable description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// A sequence of haptic pulses constituting a haptic description for one event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HapticPattern {
    /// Ordered pulses (sorted by `start_ms`).
    pub pulses: Vec<HapticPulse>,
    /// Total pattern duration in milliseconds.
    pub total_duration_ms: u64,
    /// Human-readable name for this pattern.
    pub name: String,
}

impl HapticPattern {
    /// Create an empty pattern.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            pulses: Vec::new(),
            total_duration_ms: 0,
            name: name.into(),
        }
    }

    /// Add a pulse to the pattern, updating the total duration.
    pub fn add_pulse(&mut self, pulse: HapticPulse) {
        let end = pulse.start_ms + u64::from(pulse.duration_ms);
        if end > self.total_duration_ms {
            self.total_duration_ms = end;
        }
        self.pulses.push(pulse);
        self.pulses.sort_by_key(|p| p.start_ms);
    }

    /// Whether the pattern contains no pulses.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pulses.is_empty()
    }

    /// Number of pulses in the pattern.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pulses.len()
    }

    /// Scale all pulse intensities by a gain factor (clamped to [0,1]).
    pub fn scale_intensity(&mut self, gain: f32) {
        for pulse in &mut self.pulses {
            pulse.intensity = (pulse.intensity * gain).clamp(0.0, 1.0);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Media event types
// ──────────────────────────────────────────────────────────────────────────────

/// A media event that can be translated into a haptic pattern.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MediaEvent {
    /// Music beat onset.
    BeatOnset {
        /// Normalised beat intensity (0.0–1.0).
        intensity: f32,
    },
    /// A sharp audio transient (e.g. gunshot, door slam).
    AudioTransient {
        /// Transient intensity (0.0–1.0).
        intensity: f32,
        /// Approximate frequency band: "low", "mid", or "high".
        band: String,
    },
    /// Dialogue speech onset — someone starts speaking.
    SpeechOnset {
        /// Estimated loudness (0.0–1.0).
        loudness: f32,
    },
    /// A scene change in the video.
    SceneChange {
        /// Scene change confidence (0.0–1.0).
        confidence: f32,
    },
    /// An explosion or impact event.
    ImpactEvent {
        /// Impact magnitude (0.0–1.0).
        magnitude: f32,
    },
    /// Music bass line pulse.
    BassPulse {
        /// Bass intensity (0.0–1.0).
        intensity: f32,
    },
    /// Notification or alert in the media content.
    Alert,
    /// End of a sentence or phrase boundary.
    PhraseBoundary,
    /// Custom event with an explicit haptic pattern.
    Custom {
        /// Name of the event.
        name: String,
        /// Intensity hint (0.0–1.0).
        intensity: f32,
    },
}

// ──────────────────────────────────────────────────────────────────────────────
// Generator
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for haptic description generation.
#[derive(Debug, Clone)]
pub struct HapticDescriptionConfig {
    /// Global intensity multiplier (0.0–2.0).
    pub global_gain: f32,
    /// Whether to add a brief pre-event lead-in pulse.
    pub lead_in: bool,
    /// Minimum pulse intensity to emit (sub-threshold pulses are dropped).
    pub min_intensity: f32,
}

impl Default for HapticDescriptionConfig {
    fn default() -> Self {
        Self {
            global_gain: 1.0,
            lead_in: false,
            min_intensity: 0.05,
        }
    }
}

/// Generator that converts [`MediaEvent`]s into [`HapticPattern`]s.
///
/// Each media event type maps to a specific waveform strategy tuned for
/// perceptual clarity on mobile/wearable actuators.
pub struct HapticDescriptionGenerator {
    config: HapticDescriptionConfig,
}

impl HapticDescriptionGenerator {
    /// Create a generator with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: HapticDescriptionConfig::default(),
        }
    }

    /// Create a generator with custom configuration.
    #[must_use]
    pub fn with_config(config: HapticDescriptionConfig) -> Self {
        Self { config }
    }

    /// Translate a single media event into a haptic pattern.
    #[must_use]
    pub fn from_event(&self, event: MediaEvent) -> HapticPattern {
        let gain = self.config.global_gain.clamp(0.0, 2.0);
        match event {
            MediaEvent::BeatOnset { intensity } => {
                let eff = (intensity * gain).clamp(0.0, 1.0);
                if eff < self.config.min_intensity {
                    return HapticPattern::new("beat-onset-silent");
                }
                let mut pattern = HapticPattern::new("beat-onset");
                if self.config.lead_in {
                    pattern
                        .add_pulse(HapticPulse::new(0, 5, eff * 0.4, HapticWaveform::Click)
                            .with_description("lead-in"));
                    pattern.add_pulse(
                        HapticPulse::new(10, 20, eff, HapticWaveform::Thud)
                            .with_description("beat"),
                    );
                } else {
                    pattern.add_pulse(
                        HapticPulse::new(0, 20, eff, HapticWaveform::Thud)
                            .with_description("beat"),
                    );
                }
                pattern
            }
            MediaEvent::AudioTransient { intensity, band } => {
                let eff = (intensity * gain).clamp(0.0, 1.0);
                if eff < self.config.min_intensity {
                    return HapticPattern::new("transient-silent");
                }
                let waveform = if band == "low" {
                    HapticWaveform::Thud
                } else {
                    HapticWaveform::Click
                };
                let mut pattern = HapticPattern::new("audio-transient");
                pattern.add_pulse(
                    HapticPulse::new(0, 15, eff, waveform)
                        .with_description(format!("transient-{band}")),
                );
                pattern
            }
            MediaEvent::SpeechOnset { loudness } => {
                let eff = (loudness * gain * 0.6).clamp(0.0, 1.0);
                if eff < self.config.min_intensity {
                    return HapticPattern::new("speech-silent");
                }
                let mut pattern = HapticPattern::new("speech-onset");
                pattern.add_pulse(
                    HapticPulse::new(0, 30, eff, HapticWaveform::Rise)
                        .with_description("speech-start"),
                );
                pattern
            }
            MediaEvent::SceneChange { confidence } => {
                let eff = (confidence * gain).clamp(0.0, 1.0);
                if eff < self.config.min_intensity {
                    return HapticPattern::new("scene-change-silent");
                }
                let mut pattern = HapticPattern::new("scene-change");
                pattern.add_pulse(
                    HapticPulse::new(0, 10, eff * 0.7, HapticWaveform::Click)
                        .with_description("scene-flash-1"),
                );
                pattern.add_pulse(
                    HapticPulse::new(15, 10, eff, HapticWaveform::Click)
                        .with_description("scene-flash-2"),
                );
                pattern
            }
            MediaEvent::ImpactEvent { magnitude } => {
                let eff = (magnitude * gain).clamp(0.0, 1.0);
                if eff < self.config.min_intensity {
                    return HapticPattern::new("impact-silent");
                }
                let mut pattern = HapticPattern::new("impact");
                pattern.add_pulse(
                    HapticPulse::new(0, 50, eff, HapticWaveform::Thud)
                        .with_description("impact-main"),
                );
                if eff > 0.5 {
                    // Strong impacts get a rumble tail
                    pattern.add_pulse(
                        HapticPulse::new(55, 150, eff * 0.4, HapticWaveform::Buzz)
                            .with_description("impact-rumble"),
                    );
                }
                pattern
            }
            MediaEvent::BassPulse { intensity } => {
                let eff = (intensity * gain).clamp(0.0, 1.0);
                if eff < self.config.min_intensity {
                    return HapticPattern::new("bass-silent");
                }
                let mut pattern = HapticPattern::new("bass-pulse");
                pattern.add_pulse(
                    HapticPulse::new(0, 80, eff * 0.8, HapticWaveform::Buzz)
                        .with_description("bass-buzz"),
                );
                pattern
            }
            MediaEvent::Alert => {
                let eff = (0.8_f32 * gain).clamp(0.0, 1.0);
                let mut pattern = HapticPattern::new("alert");
                pattern.add_pulse(
                    HapticPulse::new(0, 10, eff, HapticWaveform::Click)
                        .with_description("alert-1"),
                );
                pattern.add_pulse(
                    HapticPulse::new(20, 10, eff, HapticWaveform::Click)
                        .with_description("alert-2"),
                );
                pattern.add_pulse(
                    HapticPulse::new(40, 10, eff, HapticWaveform::Click)
                        .with_description("alert-3"),
                );
                pattern
            }
            MediaEvent::PhraseBoundary => {
                let eff = (0.3_f32 * gain).clamp(0.0, 1.0);
                if eff < self.config.min_intensity {
                    return HapticPattern::new("phrase-silent");
                }
                let mut pattern = HapticPattern::new("phrase-boundary");
                pattern.add_pulse(
                    HapticPulse::new(0, 8, eff, HapticWaveform::Click)
                        .with_description("phrase-tick"),
                );
                pattern
            }
            MediaEvent::Custom { name, intensity } => {
                let eff = (intensity * gain).clamp(0.0, 1.0);
                let mut pattern = HapticPattern::new(format!("custom-{name}"));
                if eff >= self.config.min_intensity {
                    pattern.add_pulse(
                        HapticPulse::new(0, 40, eff, HapticWaveform::Custom)
                            .with_description(name),
                    );
                }
                pattern
            }
        }
    }

    /// Translate a sequence of timed media events into a merged haptic timeline.
    ///
    /// Each event is placed at its given timestamp (milliseconds from start).
    #[must_use]
    pub fn from_timed_events(&self, events: &[(u64, MediaEvent)]) -> HapticPattern {
        let mut combined = HapticPattern::new("timeline");
        for (ts_ms, event) in events {
            let sub = self.from_event(event.clone());
            for mut pulse in sub.pulses {
                pulse.start_ms += ts_ms;
                combined.add_pulse(pulse);
            }
        }
        combined
    }
}

impl Default for HapticDescriptionGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beat_onset_generates_pulse() {
        let gen = HapticDescriptionGenerator::new();
        let pattern = gen.from_event(MediaEvent::BeatOnset { intensity: 0.8 });
        assert!(!pattern.is_empty(), "beat-onset should generate at least one pulse");
        assert!(pattern.total_duration_ms > 0);
    }

    #[test]
    fn test_low_intensity_beat_produces_silent_pattern() {
        let gen = HapticDescriptionGenerator::new();
        let pattern = gen.from_event(MediaEvent::BeatOnset { intensity: 0.01 });
        assert!(
            pattern.is_empty(),
            "sub-threshold beat should yield no pulses"
        );
    }

    #[test]
    fn test_impact_with_strong_magnitude_has_rumble() {
        let gen = HapticDescriptionGenerator::new();
        let pattern = gen.from_event(MediaEvent::ImpactEvent { magnitude: 0.9 });
        assert!(pattern.len() >= 2, "strong impact should have main + rumble");
    }

    #[test]
    fn test_impact_with_weak_magnitude_no_rumble() {
        let gen = HapticDescriptionGenerator::new();
        let pattern = gen.from_event(MediaEvent::ImpactEvent { magnitude: 0.3 });
        assert_eq!(pattern.len(), 1, "weak impact should have only main pulse");
    }

    #[test]
    fn test_alert_has_three_clicks() {
        let gen = HapticDescriptionGenerator::new();
        let pattern = gen.from_event(MediaEvent::Alert);
        assert_eq!(pattern.len(), 3, "alert should have 3 click pulses");
    }

    #[test]
    fn test_scene_change_has_two_flashes() {
        let gen = HapticDescriptionGenerator::new();
        let pattern = gen.from_event(MediaEvent::SceneChange { confidence: 0.9 });
        assert_eq!(pattern.len(), 2, "scene change should have 2 click pulses");
    }

    #[test]
    fn test_lead_in_adds_extra_pulse() {
        let config = HapticDescriptionConfig {
            lead_in: true,
            ..Default::default()
        };
        let gen = HapticDescriptionGenerator::with_config(config);
        let pattern = gen.from_event(MediaEvent::BeatOnset { intensity: 0.8 });
        assert_eq!(pattern.len(), 2, "with lead-in: expect 2 pulses");
    }

    #[test]
    fn test_timed_events_timeline() {
        let gen = HapticDescriptionGenerator::new();
        let events = vec![
            (0, MediaEvent::BeatOnset { intensity: 0.9 }),
            (500, MediaEvent::BeatOnset { intensity: 0.7 }),
        ];
        let timeline = gen.from_timed_events(&events);
        // Should have pulses from both events
        assert!(timeline.len() >= 2);
        // Second event pulses should start at or after 500ms
        let late_pulses: Vec<_> = timeline.pulses.iter().filter(|p| p.start_ms >= 500).collect();
        assert!(!late_pulses.is_empty());
    }

    #[test]
    fn test_global_gain_zero_silences_all() {
        let config = HapticDescriptionConfig {
            global_gain: 0.0,
            min_intensity: 0.05,
            ..Default::default()
        };
        let gen = HapticDescriptionGenerator::with_config(config);
        let pattern = gen.from_event(MediaEvent::ImpactEvent { magnitude: 1.0 });
        assert!(pattern.is_empty(), "zero gain should silence all pulses");
    }

    #[test]
    fn test_pulse_intensity_clamped() {
        let pulse = HapticPulse::new(0, 10, 5.0, HapticWaveform::Click);
        assert!(pulse.intensity <= 1.0, "intensity must be clamped to 1.0");
    }

    #[test]
    fn test_pattern_scale_intensity() {
        let gen = HapticDescriptionGenerator::new();
        let mut pattern = gen.from_event(MediaEvent::Alert);
        let orig_intensity = pattern.pulses[0].intensity;
        pattern.scale_intensity(0.5);
        let scaled = pattern.pulses[0].intensity;
        assert!(
            (scaled - orig_intensity * 0.5).abs() < 1e-5,
            "scale_intensity should halve each pulse"
        );
    }

    #[test]
    fn test_waveform_typical_durations_positive() {
        for wf in &[
            HapticWaveform::Click,
            HapticWaveform::Buzz,
            HapticWaveform::Thud,
            HapticWaveform::Pulse,
        ] {
            assert!(wf.typical_duration_ms() > 0);
        }
    }

    #[test]
    fn test_phrase_boundary_is_short() {
        let gen = HapticDescriptionGenerator::new();
        let pattern = gen.from_event(MediaEvent::PhraseBoundary);
        assert!(
            pattern.total_duration_ms <= 20,
            "phrase boundary should be short"
        );
    }
}
