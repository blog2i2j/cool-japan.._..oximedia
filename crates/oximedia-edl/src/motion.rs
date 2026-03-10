//! Motion effects for EDL operations.
//!
//! This module provides support for motion effects including speed changes,
//! freeze frames, and reverse playback.

use crate::error::{EdlError, EdlResult};
use std::fmt;
use std::str::FromStr;

/// Motion effect applied to an event.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionEffect {
    /// Speed multiplier (1.0 = normal speed, 0.5 = half speed, 2.0 = double speed).
    pub speed: f64,
    /// Whether to reverse playback.
    pub reverse: bool,
    /// Freeze frame duration in frames (if any).
    pub freeze_frames: Option<u32>,
    /// Interpolation method for speed changes.
    pub interpolation: InterpolationMethod,
}

impl MotionEffect {
    /// Create a new motion effect with the given speed.
    #[must_use]
    pub fn new(speed: f64) -> Self {
        Self {
            speed,
            reverse: false,
            freeze_frames: None,
            interpolation: InterpolationMethod::Blend,
        }
    }

    /// Create a freeze frame effect.
    #[must_use]
    pub const fn freeze(frames: u32) -> Self {
        Self {
            speed: 0.0,
            reverse: false,
            freeze_frames: Some(frames),
            interpolation: InterpolationMethod::Nearest,
        }
    }

    /// Create a reverse playback effect.
    #[must_use]
    pub fn reverse() -> Self {
        Self {
            speed: -1.0,
            reverse: true,
            freeze_frames: None,
            interpolation: InterpolationMethod::Blend,
        }
    }

    /// Create a slow motion effect.
    #[must_use]
    pub fn slow_motion(factor: f64) -> Self {
        Self::new(factor)
    }

    /// Create a fast motion effect.
    #[must_use]
    pub fn fast_motion(factor: f64) -> Self {
        Self::new(factor)
    }

    /// Set the interpolation method.
    pub fn set_interpolation(&mut self, method: InterpolationMethod) {
        self.interpolation = method;
    }

    /// Check if this is a freeze frame effect.
    #[must_use]
    pub const fn is_freeze(&self) -> bool {
        self.freeze_frames.is_some()
    }

    /// Check if this is a reverse playback effect.
    #[must_use]
    pub const fn is_reverse(&self) -> bool {
        self.reverse
    }

    /// Check if this is a speed change effect.
    #[must_use]
    pub fn is_speed_change(&self) -> bool {
        (self.speed - 1.0).abs() > f64::EPSILON
    }

    /// Get the effective speed (accounting for reverse).
    #[must_use]
    pub fn effective_speed(&self) -> f64 {
        if self.reverse {
            -self.speed.abs()
        } else {
            self.speed
        }
    }

    /// Validate the motion effect.
    ///
    /// # Errors
    ///
    /// Returns an error if the motion effect has invalid parameters.
    pub fn validate(&self) -> EdlResult<()> {
        if self.speed < 0.0 && !self.reverse {
            return Err(EdlError::InvalidMotionEffect(
                "Negative speed requires reverse flag".to_string(),
            ));
        }

        if self.freeze_frames.is_some() && self.speed.abs() > f64::EPSILON {
            return Err(EdlError::InvalidMotionEffect(
                "Freeze frame must have speed 0.0".to_string(),
            ));
        }

        Ok(())
    }

    /// Parse motion effect from EDL M2 comment.
    ///
    /// Format: M2 {reel} {speed} {entry_frame}
    ///
    /// # Errors
    ///
    /// Returns an error if the M2 comment is invalid.
    pub fn from_m2_comment(comment: &str) -> EdlResult<Self> {
        let parts: Vec<&str> = comment.split_whitespace().collect();

        if parts.len() < 4 || parts[0] != "M2" {
            return Err(EdlError::InvalidMotionEffect(format!(
                "Invalid M2 comment: {comment}"
            )));
        }

        // parts[0] = "M2", parts[1] = reel, parts[2] = speed, parts[3] = entry_frame
        let speed = parts[2].parse::<f64>().map_err(|_| {
            EdlError::InvalidMotionEffect(format!("Invalid speed value: {}", parts[2]))
        })?;

        Ok(Self::new(speed))
    }

    /// Format as M2 comment for EDL export.
    #[must_use]
    pub fn to_m2_comment(&self, reel: &str, entry_frame: u32) -> String {
        format!("M2 {} {:.2} {}", reel, self.speed, entry_frame)
    }
}

impl Default for MotionEffect {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Interpolation method for motion effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InterpolationMethod {
    /// Nearest neighbor (no interpolation).
    Nearest,
    /// Linear interpolation.
    Linear,
    /// Blend/motion blur interpolation.
    Blend,
    /// Optical flow interpolation.
    OpticalFlow,
}

impl InterpolationMethod {
    /// Get the quality level of the interpolation method (0-3).
    #[must_use]
    pub const fn quality_level(&self) -> u8 {
        match self {
            Self::Nearest => 0,
            Self::Linear => 1,
            Self::Blend => 2,
            Self::OpticalFlow => 3,
        }
    }
}

impl FromStr for InterpolationMethod {
    type Err = EdlError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_uppercase().as_str() {
            "NEAREST" | "NONE" => Ok(Self::Nearest),
            "LINEAR" => Ok(Self::Linear),
            "BLEND" | "MOTION_BLUR" => Ok(Self::Blend),
            "OPTICAL_FLOW" | "FLOW" => Ok(Self::OpticalFlow),
            _ => Err(EdlError::InvalidMotionEffect(s.to_string())),
        }
    }
}

impl fmt::Display for InterpolationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Nearest => "Nearest",
            Self::Linear => "Linear",
            Self::Blend => "Blend",
            Self::OpticalFlow => "Optical Flow",
        };
        write!(f, "{s}")
    }
}

/// Motion effect builder for creating complex motion effects.
#[derive(Debug, Clone)]
pub struct MotionEffectBuilder {
    speed: f64,
    reverse: bool,
    freeze_frames: Option<u32>,
    interpolation: InterpolationMethod,
}

impl MotionEffectBuilder {
    /// Create a new motion effect builder with normal speed.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            speed: 1.0,
            reverse: false,
            freeze_frames: None,
            interpolation: InterpolationMethod::Blend,
        }
    }

    /// Set the speed multiplier.
    #[must_use]
    pub const fn speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }

    /// Enable reverse playback.
    #[must_use]
    pub const fn reverse(mut self) -> Self {
        self.reverse = true;
        self
    }

    /// Set freeze frame duration.
    #[must_use]
    pub const fn freeze(mut self, frames: u32) -> Self {
        self.freeze_frames = Some(frames);
        self.speed = 0.0;
        self
    }

    /// Set interpolation method.
    #[must_use]
    pub const fn interpolation(mut self, method: InterpolationMethod) -> Self {
        self.interpolation = method;
        self
    }

    /// Build the motion effect.
    #[must_use]
    pub const fn build(self) -> MotionEffect {
        MotionEffect {
            speed: self.speed,
            reverse: self.reverse,
            freeze_frames: self.freeze_frames,
            interpolation: self.interpolation,
        }
    }
}

impl Default for MotionEffectBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_speed() {
        let effect = MotionEffect::new(1.0);
        assert!((effect.speed - 1.0).abs() < f64::EPSILON);
        assert!(!effect.is_reverse());
        assert!(!effect.is_freeze());
        assert!(!effect.is_speed_change());
    }

    #[test]
    fn test_slow_motion() {
        let effect = MotionEffect::slow_motion(0.5);
        assert!((effect.speed - 0.5).abs() < f64::EPSILON);
        assert!(effect.is_speed_change());
    }

    #[test]
    fn test_fast_motion() {
        let effect = MotionEffect::fast_motion(2.0);
        assert!((effect.speed - 2.0).abs() < f64::EPSILON);
        assert!(effect.is_speed_change());
    }

    #[test]
    fn test_freeze_frame() {
        let effect = MotionEffect::freeze(100);
        assert!(effect.is_freeze());
        assert_eq!(effect.freeze_frames, Some(100));
        assert!((effect.speed).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reverse() {
        let effect = MotionEffect::reverse();
        assert!(effect.is_reverse());
        assert!((effect.speed + 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_effective_speed() {
        let effect = MotionEffect::reverse();
        assert!((effect.effective_speed() + 1.0).abs() < f64::EPSILON);

        let effect = MotionEffect::new(2.0);
        assert!((effect.effective_speed() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_builder() {
        let effect = MotionEffectBuilder::new()
            .speed(0.5)
            .interpolation(InterpolationMethod::Linear)
            .build();

        assert!((effect.speed - 0.5).abs() < f64::EPSILON);
        assert_eq!(effect.interpolation, InterpolationMethod::Linear);
    }

    #[test]
    fn test_m2_comment_parsing() {
        let effect =
            MotionEffect::from_m2_comment("M2 REEL1 0.50 100").expect("operation should succeed");
        assert!((effect.speed - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_m2_comment_formatting() {
        let effect = MotionEffect::new(0.5);
        let comment = effect.to_m2_comment("REEL1", 100);
        assert_eq!(comment, "M2 REEL1 0.50 100");
    }

    #[test]
    fn test_interpolation_parsing() {
        assert_eq!(
            "NEAREST"
                .parse::<InterpolationMethod>()
                .expect("operation should succeed"),
            InterpolationMethod::Nearest
        );
        assert_eq!(
            "BLEND"
                .parse::<InterpolationMethod>()
                .expect("operation should succeed"),
            InterpolationMethod::Blend
        );
    }

    #[test]
    fn test_validation() {
        let effect = MotionEffect::new(1.0);
        assert!(effect.validate().is_ok());

        let mut bad_effect = MotionEffect::freeze(100);
        bad_effect.speed = 1.0;
        assert!(bad_effect.validate().is_err());
    }
}
