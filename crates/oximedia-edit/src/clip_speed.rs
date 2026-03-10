//! Clip speed and reverse control for timeline clips.
//!
//! Supports normal, reverse, and variable-speed playback with
//! keyframe-based speed envelopes.

#![allow(dead_code)]

/// The speed playback mode of a clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeedMode {
    /// Normal forward playback at the source frame rate.
    Normal,
    /// Reversed playback at the source frame rate.
    Reverse,
    /// Variable speed driven by keyframes.
    Variable,
}

impl SpeedMode {
    /// Returns `true` if this is a variable-speed mode.
    #[must_use]
    pub fn is_variable(self) -> bool {
        matches!(self, Self::Variable)
    }

    /// Returns `true` if this is reverse playback.
    #[must_use]
    pub fn is_reverse(self) -> bool {
        matches!(self, Self::Reverse)
    }
}

/// Speed configuration for a clip.
#[derive(Debug, Clone)]
pub struct ClipSpeed {
    /// The speed mode.
    pub mode: SpeedMode,
    /// Speed multiplier (e.g. 0.5 = half speed, 2.0 = double speed).
    /// Ignored when `mode` is `Variable`.
    pub multiplier: f64,
    /// Source duration in frames at 1× speed.
    pub source_frames: u64,
}

impl ClipSpeed {
    /// Create a new `ClipSpeed` with the given mode, multiplier, and source frame count.
    #[must_use]
    pub fn new(mode: SpeedMode, multiplier: f64, source_frames: u64) -> Self {
        Self {
            mode,
            multiplier: multiplier.max(0.001),
            source_frames,
        }
    }

    /// Compute the output duration in frames after speed adjustment.
    ///
    /// For normal/reverse modes: `ceil(source_frames / multiplier)`.
    /// For variable mode: returns `source_frames` unchanged (envelope not evaluated here).
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn output_duration_frames(&self) -> u64 {
        if self.mode == SpeedMode::Variable {
            self.source_frames
        } else {
            let out = (self.source_frames as f64 / self.multiplier).ceil();
            out as u64
        }
    }

    /// Returns `true` if the clip is playing slower than real-time.
    #[must_use]
    pub fn is_slow_motion(&self) -> bool {
        self.multiplier < 1.0 && !matches!(self.mode, SpeedMode::Variable)
    }

    /// Returns `true` if playing at exactly normal speed and not reversed.
    #[must_use]
    pub fn is_normal(&self) -> bool {
        matches!(self.mode, SpeedMode::Normal) && (self.multiplier - 1.0).abs() < f64::EPSILON
    }
}

/// A keyframe that defines speed at a specific output frame.
#[derive(Debug, Clone, Copy)]
pub struct SpeedKeyframe {
    /// Output frame at which this keyframe applies.
    pub frame: u64,
    /// Speed multiplier at this keyframe.
    pub speed: f64,
}

impl SpeedKeyframe {
    /// Create a new `SpeedKeyframe`.
    #[must_use]
    pub fn new(frame: u64, speed: f64) -> Self {
        Self { frame, speed }
    }

    /// Linearly interpolate speed toward `other` at `t` (0.0–1.0).
    ///
    /// Returns the interpolated speed value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn interpolate_to(&self, other: &Self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        self.speed + (other.speed - self.speed) * t
    }

    /// Compute `t` at a given output `frame` between `self` and `other`.
    ///
    /// Returns `None` if the keyframes are at the same frame.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn t_at_frame(&self, other: &Self, frame: u64) -> Option<f64> {
        let span = other.frame as f64 - self.frame as f64;
        if span.abs() < f64::EPSILON {
            return None;
        }
        Some(((frame as f64 - self.frame as f64) / span).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speed_mode_is_variable() {
        assert!(SpeedMode::Variable.is_variable());
        assert!(!SpeedMode::Normal.is_variable());
        assert!(!SpeedMode::Reverse.is_variable());
    }

    #[test]
    fn test_speed_mode_is_reverse() {
        assert!(SpeedMode::Reverse.is_reverse());
        assert!(!SpeedMode::Normal.is_reverse());
    }

    #[test]
    fn test_clip_speed_output_duration_normal() {
        let cs = ClipSpeed::new(SpeedMode::Normal, 2.0, 100);
        assert_eq!(cs.output_duration_frames(), 50);
    }

    #[test]
    fn test_clip_speed_output_duration_slow_motion() {
        let cs = ClipSpeed::new(SpeedMode::Normal, 0.5, 100);
        assert_eq!(cs.output_duration_frames(), 200);
    }

    #[test]
    fn test_clip_speed_output_duration_reverse() {
        let cs = ClipSpeed::new(SpeedMode::Reverse, 1.0, 60);
        assert_eq!(cs.output_duration_frames(), 60);
    }

    #[test]
    fn test_clip_speed_output_duration_variable() {
        let cs = ClipSpeed::new(SpeedMode::Variable, 2.0, 120);
        // Variable mode returns source_frames unchanged
        assert_eq!(cs.output_duration_frames(), 120);
    }

    #[test]
    fn test_clip_speed_is_slow_motion_true() {
        let cs = ClipSpeed::new(SpeedMode::Normal, 0.25, 100);
        assert!(cs.is_slow_motion());
    }

    #[test]
    fn test_clip_speed_is_slow_motion_false_fast() {
        let cs = ClipSpeed::new(SpeedMode::Normal, 2.0, 100);
        assert!(!cs.is_slow_motion());
    }

    #[test]
    fn test_clip_speed_is_slow_motion_false_variable() {
        let cs = ClipSpeed::new(SpeedMode::Variable, 0.1, 100);
        assert!(!cs.is_slow_motion());
    }

    #[test]
    fn test_clip_speed_is_normal() {
        let cs = ClipSpeed::new(SpeedMode::Normal, 1.0, 50);
        assert!(cs.is_normal());
    }

    #[test]
    fn test_clip_speed_not_normal_when_reverse() {
        let cs = ClipSpeed::new(SpeedMode::Reverse, 1.0, 50);
        assert!(!cs.is_normal());
    }

    #[test]
    fn test_speed_keyframe_interpolate_midpoint() {
        let a = SpeedKeyframe::new(0, 1.0);
        let b = SpeedKeyframe::new(100, 2.0);
        let mid = a.interpolate_to(&b, 0.5);
        assert!((mid - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_speed_keyframe_interpolate_start() {
        let a = SpeedKeyframe::new(0, 0.5);
        let b = SpeedKeyframe::new(100, 4.0);
        assert!((a.interpolate_to(&b, 0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_speed_keyframe_interpolate_end() {
        let a = SpeedKeyframe::new(0, 0.5);
        let b = SpeedKeyframe::new(100, 4.0);
        assert!((a.interpolate_to(&b, 1.0) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_speed_keyframe_t_at_frame() {
        let a = SpeedKeyframe::new(0, 1.0);
        let b = SpeedKeyframe::new(100, 2.0);
        let t = a.t_at_frame(&b, 50).expect("t should be valid");
        assert!((t - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_speed_keyframe_t_at_frame_same_frame_returns_none() {
        let a = SpeedKeyframe::new(10, 1.0);
        let b = SpeedKeyframe::new(10, 2.0);
        assert!(a.t_at_frame(&b, 10).is_none());
    }
}
