//! Playback speed adaptation for accessibility.
//!
//! Provides adaptive speed control that accounts for caption density,
//! dialogue pace, and cognitive pacing needs.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PlaybackSpeed newtype
// ---------------------------------------------------------------------------

/// A validated playback speed multiplier.
///
/// Valid range is 0.25× to 4.0× (inclusive).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct PlaybackSpeed(pub f32);

impl PlaybackSpeed {
    /// The minimum valid playback speed (0.25×).
    pub const MIN: f32 = 0.25;
    /// The maximum valid playback speed (4.0×).
    pub const MAX: f32 = 4.0;
    /// Normal playback speed (1.0×).
    pub const NORMAL: Self = Self(1.0);

    /// Create a new `PlaybackSpeed`, returning `None` if out of range.
    #[must_use]
    pub fn new(speed: f32) -> Option<Self> {
        if (Self::MIN..=Self::MAX).contains(&speed) {
            Some(Self(speed))
        } else {
            None
        }
    }

    /// Create a speed, clamping to the valid range.
    #[must_use]
    pub fn clamped(speed: f32) -> Self {
        Self(speed.clamp(Self::MIN, Self::MAX))
    }

    /// Whether this speed value is within the valid range.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.0 >= Self::MIN && self.0 <= Self::MAX
    }

    /// Speed expressed as an integer percentage (e.g. 1.5× → 150).
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_percent(&self) -> u32 {
        (self.0 * 100.0).round() as u32
    }

    /// The raw speed multiplier.
    #[must_use]
    pub const fn value(&self) -> f32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// SpeedAdaptation
// ---------------------------------------------------------------------------

/// A set of available playback speed options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedAdaptation {
    /// Available speed choices, always sorted ascending.
    pub speeds: Vec<PlaybackSpeed>,
}

impl SpeedAdaptation {
    /// Create a `SpeedAdaptation` from the given speed values.
    ///
    /// Values outside the valid range are silently filtered out.
    #[must_use]
    pub fn from_values(values: &[f32]) -> Self {
        let mut speeds: Vec<PlaybackSpeed> = values
            .iter()
            .filter_map(|&v| PlaybackSpeed::new(v))
            .collect();
        speeds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { speeds }
    }

    /// Standard set of accessibility-friendly speed options:
    /// 0.5×, 0.75×, 1.0×, 1.25×, 1.5×, 2.0×.
    #[must_use]
    pub fn standard() -> Self {
        Self::from_values(&[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    }

    /// Nearest available speed to the requested value.
    #[must_use]
    pub fn nearest(&self, requested: f32) -> Option<PlaybackSpeed> {
        self.speeds.iter().copied().min_by(|a, b| {
            let da = (a.0 - requested).abs();
            let db = (b.0 - requested).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ---------------------------------------------------------------------------
// PitchCompensation
// ---------------------------------------------------------------------------

/// Pitch-compensation (pitch-correction) helper.
pub struct PitchCompensation;

impl PitchCompensation {
    /// Whether pitch compensation should be enabled for the given speed.
    ///
    /// Returns `true` for any speed other than 1.0× (normal).
    #[must_use]
    pub fn enabled_for(speed: PlaybackSpeed) -> bool {
        (speed.0 - 1.0).abs() > 1e-4
    }
}

// ---------------------------------------------------------------------------
// ContentTypeAdaptation
// ---------------------------------------------------------------------------

/// Speed limits derived from content characteristics.
pub struct ContentTypeAdaptation;

impl ContentTypeAdaptation {
    /// Maximum comfortable speed given a caption character rate.
    ///
    /// A higher character-per-second rate means captions are already dense;
    /// increasing speed further would make them unreadable.
    ///
    /// # Arguments
    ///
    /// * `caption_rate_cps` — Caption character rate in characters per second.
    #[must_use]
    pub fn max_speed_for_captions(caption_rate_cps: f32) -> PlaybackSpeed {
        // At 20 cps assume comfortable; scale down for higher rates
        let max = if caption_rate_cps <= 0.0 {
            2.0_f32
        } else {
            // Each cps above 15 reduces max speed by 0.05
            (2.0 - (caption_rate_cps - 15.0).max(0.0) * 0.05).clamp(0.5, 2.0)
        };
        PlaybackSpeed::clamped(max)
    }

    /// Maximum comfortable speed given a speech rate.
    ///
    /// # Arguments
    ///
    /// * `speech_rate_wpm` — Speech rate in words per minute.
    #[must_use]
    pub fn max_speed_for_dialogue(speech_rate_wpm: f32) -> PlaybackSpeed {
        // Comfortable listening comprehension tops out around 300 wpm effective
        // Effective wpm = speech_rate * speed; cap at 250 wpm effective
        let max_effective_wpm = 250.0_f32;
        let max = if speech_rate_wpm <= 0.0 {
            2.0_f32
        } else {
            (max_effective_wpm / speech_rate_wpm).clamp(0.5, 2.0)
        };
        PlaybackSpeed::clamped(max)
    }
}

// ---------------------------------------------------------------------------
// CognitivePacing
// ---------------------------------------------------------------------------

/// Cognitive pacing configuration for learners or viewers who need
/// structured pauses and comprehension modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivePacing {
    /// Pause playback at each chapter marker.
    pub pause_at_chapter: bool,
    /// Duration of the automatic chapter pause in milliseconds.
    pub pause_duration_ms: u32,
    /// Enable comprehension mode: slower auto-advance, repeat prompts.
    pub comprehension_mode: bool,
}

impl Default for CognitivePacing {
    fn default() -> Self {
        Self {
            pause_at_chapter: false,
            pause_duration_ms: 3_000,
            comprehension_mode: false,
        }
    }
}

impl CognitivePacing {
    /// Create a new `CognitivePacing` configuration.
    #[must_use]
    pub const fn new(
        pause_at_chapter: bool,
        pause_duration_ms: u32,
        comprehension_mode: bool,
    ) -> Self {
        Self {
            pause_at_chapter,
            pause_duration_ms,
            comprehension_mode,
        }
    }

    /// Whether any pacing assistance is active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.pause_at_chapter || self.comprehension_mode
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playback_speed_valid_range() {
        assert!(PlaybackSpeed::new(0.25).is_some());
        assert!(PlaybackSpeed::new(4.0).is_some());
        assert!(PlaybackSpeed::new(1.0).is_some());
    }

    #[test]
    fn test_playback_speed_invalid_range() {
        assert!(PlaybackSpeed::new(0.1).is_none());
        assert!(PlaybackSpeed::new(5.0).is_none());
        assert!(PlaybackSpeed::new(-1.0).is_none());
    }

    #[test]
    fn test_playback_speed_is_valid() {
        assert!(PlaybackSpeed(1.5).is_valid());
        assert!(!PlaybackSpeed(0.1).is_valid());
        assert!(!PlaybackSpeed(5.0).is_valid());
    }

    #[test]
    fn test_to_percent() {
        assert_eq!(PlaybackSpeed(1.0).to_percent(), 100);
        assert_eq!(PlaybackSpeed(1.5).to_percent(), 150);
        assert_eq!(PlaybackSpeed(0.5).to_percent(), 50);
        assert_eq!(PlaybackSpeed(2.0).to_percent(), 200);
    }

    #[test]
    fn test_speed_adaptation_standard() {
        let adapt = SpeedAdaptation::standard();
        assert_eq!(adapt.speeds.len(), 6);
        let percents: Vec<u32> = adapt.speeds.iter().map(|s| s.to_percent()).collect();
        assert_eq!(percents, vec![50, 75, 100, 125, 150, 200]);
    }

    #[test]
    fn test_speed_adaptation_nearest() {
        let adapt = SpeedAdaptation::standard();
        let nearest = adapt.nearest(1.3).expect("nearest should be valid");
        assert_eq!(nearest.to_percent(), 125);
    }

    #[test]
    fn test_speed_adaptation_filters_invalid() {
        let adapt = SpeedAdaptation::from_values(&[0.0, 0.5, 1.0, 5.0]);
        // 0.0 and 5.0 are out of range
        assert_eq!(adapt.speeds.len(), 2);
    }

    #[test]
    fn test_pitch_compensation_enabled() {
        assert!(PitchCompensation::enabled_for(PlaybackSpeed(0.5)));
        assert!(PitchCompensation::enabled_for(PlaybackSpeed(1.5)));
        assert!(!PitchCompensation::enabled_for(PlaybackSpeed(1.0)));
    }

    #[test]
    fn test_max_speed_for_captions_low_rate() {
        // Low caption rate → should allow 2.0×
        let max = ContentTypeAdaptation::max_speed_for_captions(10.0);
        assert!((max.0 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_max_speed_for_captions_high_rate() {
        // Very high caption rate → reduced max speed
        let max_low = ContentTypeAdaptation::max_speed_for_captions(15.0);
        let max_high = ContentTypeAdaptation::max_speed_for_captions(30.0);
        assert!(max_high.0 < max_low.0);
    }

    #[test]
    fn test_max_speed_for_dialogue() {
        // Slow speaker (100 wpm) → allow higher speed
        let max_slow = ContentTypeAdaptation::max_speed_for_dialogue(100.0);
        // Fast speaker (200 wpm) → lower max speed
        let max_fast = ContentTypeAdaptation::max_speed_for_dialogue(200.0);
        assert!(max_slow.0 >= max_fast.0);
    }

    #[test]
    fn test_cognitive_pacing_default() {
        let pacing = CognitivePacing::default();
        assert!(!pacing.pause_at_chapter);
        assert!(!pacing.comprehension_mode);
        assert!(!pacing.is_active());
    }

    #[test]
    fn test_cognitive_pacing_active() {
        let pacing = CognitivePacing::new(true, 5_000, false);
        assert!(pacing.is_active());
        assert_eq!(pacing.pause_duration_ms, 5_000);
    }
}
