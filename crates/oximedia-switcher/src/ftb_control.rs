#![allow(dead_code)]
//! Fade-to-black (FTB) control for live production switchers.
//!
//! This module implements a fade-to-black controller that smoothly transitions
//! the program output to black. FTB is a critical broadcast function used for
//! show opens/closes, emergency situations, and commercial breaks.

use std::fmt;

/// State of the FTB controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FtbState {
    /// Normal output (not fading).
    Normal,
    /// Fading to black.
    FadingToBlack,
    /// Fully black.
    Black,
    /// Fading from black back to normal.
    FadingFromBlack,
}

impl fmt::Display for FtbState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::FadingToBlack => write!(f, "Fading to Black"),
            Self::Black => write!(f, "Black"),
            Self::FadingFromBlack => write!(f, "Fading from Black"),
        }
    }
}

/// Fade curve for FTB transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FtbCurve {
    /// Linear fade.
    Linear,
    /// S-curve (smooth).
    SCurve,
    /// Fast start, slow end.
    EaseOut,
    /// Slow start, fast end.
    EaseIn,
}

impl FtbCurve {
    /// Evaluate the curve at position t (0.0 to 1.0).
    /// Returns a value from 0.0 (normal) to 1.0 (black).
    pub fn evaluate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::SCurve => 3.0 * t * t - 2.0 * t * t * t,
            Self::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            Self::EaseIn => t * t,
        }
    }
}

/// Configuration for the FTB controller.
#[derive(Debug, Clone)]
pub struct FtbConfig {
    /// Duration of the fade in frames.
    pub duration_frames: u32,
    /// Fade curve to use.
    pub curve: FtbCurve,
    /// Whether audio should also fade (audio-follow-video).
    pub fade_audio: bool,
    /// Whether downstream keyers should be affected.
    pub affect_dsk: bool,
    /// Whether FTB can be interrupted mid-fade.
    pub allow_interrupt: bool,
}

impl Default for FtbConfig {
    fn default() -> Self {
        Self {
            duration_frames: 25,
            curve: FtbCurve::Linear,
            fade_audio: true,
            affect_dsk: true,
            allow_interrupt: true,
        }
    }
}

impl FtbConfig {
    /// Create a new FTB configuration.
    pub fn new(duration_frames: u32, curve: FtbCurve) -> Self {
        Self {
            duration_frames,
            curve,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.duration_frames == 0 {
            return Err("FTB duration must be at least 1 frame".to_string());
        }
        if self.duration_frames > 300 {
            return Err("FTB duration exceeds 300 frames (10 seconds at 30fps)".to_string());
        }
        Ok(())
    }
}

/// Fade-to-black controller.
#[derive(Debug)]
pub struct FtbController {
    /// Configuration.
    config: FtbConfig,
    /// Current state.
    state: FtbState,
    /// Current frame position within the fade (0 to duration_frames).
    frame_position: u32,
    /// Current fade level (0.0 = normal, 1.0 = full black).
    fade_level: f64,
    /// Number of times FTB has been activated.
    activation_count: u64,
}

impl FtbController {
    /// Create a new FTB controller.
    pub fn new(config: FtbConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            state: FtbState::Normal,
            frame_position: 0,
            fade_level: 0.0,
            activation_count: 0,
        })
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self {
            config: FtbConfig::default(),
            state: FtbState::Normal,
            frame_position: 0,
            fade_level: 0.0,
            activation_count: 0,
        }
    }

    /// Get the current state.
    pub fn state(&self) -> FtbState {
        self.state
    }

    /// Get the current fade level (0.0 = normal, 1.0 = full black).
    pub fn fade_level(&self) -> f64 {
        self.fade_level
    }

    /// Get the video mix level (1.0 = full video, 0.0 = black).
    pub fn video_level(&self) -> f64 {
        1.0 - self.fade_level
    }

    /// Get the audio level if audio-follow is enabled.
    pub fn audio_level(&self) -> f64 {
        if self.config.fade_audio {
            1.0 - self.fade_level
        } else {
            1.0
        }
    }

    /// Get the current frame position within the fade.
    pub fn frame_position(&self) -> u32 {
        self.frame_position
    }

    /// Get the fade duration in frames.
    pub fn duration_frames(&self) -> u32 {
        self.config.duration_frames
    }

    /// Get the activation count.
    pub fn activation_count(&self) -> u64 {
        self.activation_count
    }

    /// Check if the output is currently black.
    pub fn is_black(&self) -> bool {
        self.state == FtbState::Black
    }

    /// Check if a fade is currently in progress.
    pub fn is_fading(&self) -> bool {
        matches!(
            self.state,
            FtbState::FadingToBlack | FtbState::FadingFromBlack
        )
    }

    /// Toggle FTB: if normal or fading from black, start fading to black.
    /// If black or fading to black, start fading from black.
    pub fn toggle(&mut self) -> FtbState {
        match self.state {
            FtbState::Normal => {
                self.start_fade_to_black();
            }
            FtbState::Black => {
                self.start_fade_from_black();
            }
            FtbState::FadingToBlack => {
                if self.config.allow_interrupt {
                    self.start_fade_from_black_at_current();
                }
            }
            FtbState::FadingFromBlack => {
                if self.config.allow_interrupt {
                    self.start_fade_to_black_at_current();
                }
            }
        }
        self.state
    }

    /// Start fading to black from normal.
    fn start_fade_to_black(&mut self) {
        self.state = FtbState::FadingToBlack;
        self.frame_position = 0;
        self.activation_count += 1;
    }

    /// Start fading from black.
    fn start_fade_from_black(&mut self) {
        self.state = FtbState::FadingFromBlack;
        self.frame_position = 0;
    }

    /// Start fading from black at the current level (interrupt).
    fn start_fade_from_black_at_current(&mut self) {
        self.state = FtbState::FadingFromBlack;
        // Convert current fade level to a frame position for the reverse
        let t = self.fade_level;
        self.frame_position = ((1.0 - t) * self.config.duration_frames as f64) as u32;
    }

    /// Start fading to black at the current level (interrupt).
    fn start_fade_to_black_at_current(&mut self) {
        self.state = FtbState::FadingToBlack;
        let t = self.fade_level;
        self.frame_position = (t * self.config.duration_frames as f64) as u32;
        self.activation_count += 1;
    }

    /// Advance the FTB by one frame. Returns the new state.
    pub fn advance_frame(&mut self) -> FtbState {
        match self.state {
            FtbState::FadingToBlack => {
                self.frame_position += 1;
                if self.frame_position >= self.config.duration_frames {
                    self.frame_position = self.config.duration_frames;
                    self.fade_level = 1.0;
                    self.state = FtbState::Black;
                } else {
                    let t = self.frame_position as f64 / self.config.duration_frames as f64;
                    self.fade_level = self.config.curve.evaluate(t);
                }
            }
            FtbState::FadingFromBlack => {
                self.frame_position += 1;
                if self.frame_position >= self.config.duration_frames {
                    self.frame_position = self.config.duration_frames;
                    self.fade_level = 0.0;
                    self.state = FtbState::Normal;
                } else {
                    let t = self.frame_position as f64 / self.config.duration_frames as f64;
                    self.fade_level = 1.0 - self.config.curve.evaluate(t);
                }
            }
            FtbState::Normal => {
                self.fade_level = 0.0;
            }
            FtbState::Black => {
                self.fade_level = 1.0;
            }
        }
        self.state
    }

    /// Apply the FTB level to a video frame buffer (multiply by video_level).
    /// Each pixel is assumed to be a sequence of `bytes_per_pixel` u8 values.
    pub fn apply_to_frame(&self, frame: &mut [u8]) {
        let level = self.video_level();
        if (level - 1.0).abs() < f64::EPSILON {
            return; // No change needed
        }
        if level.abs() < f64::EPSILON {
            // Full black
            for byte in frame.iter_mut() {
                *byte = 0;
            }
            return;
        }
        for byte in frame.iter_mut() {
            *byte = (*byte as f64 * level) as u8;
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &FtbConfig {
        &self.config
    }

    /// Update the configuration (only when not fading).
    pub fn set_config(&mut self, config: FtbConfig) -> Result<(), String> {
        if self.is_fading() {
            return Err("Cannot change config while fading".to_string());
        }
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Reset to normal state immediately.
    pub fn reset(&mut self) {
        self.state = FtbState::Normal;
        self.frame_position = 0;
        self.fade_level = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ftb_state_display() {
        assert_eq!(format!("{}", FtbState::Normal), "Normal");
        assert_eq!(format!("{}", FtbState::Black), "Black");
        assert_eq!(format!("{}", FtbState::FadingToBlack), "Fading to Black");
    }

    #[test]
    fn test_ftb_curve_linear() {
        let curve = FtbCurve::Linear;
        assert!((curve.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((curve.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ftb_curve_scurve() {
        let curve = FtbCurve::SCurve;
        assert!((curve.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((curve.evaluate(0.5) - 0.5).abs() < 1e-10);
        assert!((curve.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ftb_config_default() {
        let config = FtbConfig::default();
        assert_eq!(config.duration_frames, 25);
        assert!(config.fade_audio);
        assert!(config.affect_dsk);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ftb_config_validation_zero_frames() {
        let config = FtbConfig::new(0, FtbCurve::Linear);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ftb_config_validation_too_long() {
        let config = FtbConfig::new(500, FtbCurve::Linear);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_controller_creation() {
        let ctrl = FtbController::with_defaults();
        assert_eq!(ctrl.state(), FtbState::Normal);
        assert!((ctrl.fade_level()).abs() < f64::EPSILON);
        assert!((ctrl.video_level() - 1.0).abs() < f64::EPSILON);
        assert!(!ctrl.is_black());
        assert!(!ctrl.is_fading());
    }

    #[test]
    fn test_controller_toggle_to_black() {
        let mut ctrl = FtbController::with_defaults();
        let new_state = ctrl.toggle();
        assert_eq!(new_state, FtbState::FadingToBlack);
        assert!(ctrl.is_fading());
        assert_eq!(ctrl.activation_count(), 1);
    }

    #[test]
    fn test_controller_full_fade_to_black() {
        let config = FtbConfig::new(10, FtbCurve::Linear);
        let mut ctrl = FtbController::new(config).expect("should succeed in test");
        ctrl.toggle(); // Start fading

        // Advance through the fade
        for _ in 0..10 {
            ctrl.advance_frame();
        }

        assert_eq!(ctrl.state(), FtbState::Black);
        assert!((ctrl.fade_level() - 1.0).abs() < f64::EPSILON);
        assert!((ctrl.video_level()).abs() < f64::EPSILON);
        assert!(ctrl.is_black());
    }

    #[test]
    fn test_controller_full_roundtrip() {
        let config = FtbConfig::new(5, FtbCurve::Linear);
        let mut ctrl = FtbController::new(config).expect("should succeed in test");

        // Fade to black
        ctrl.toggle();
        for _ in 0..5 {
            ctrl.advance_frame();
        }
        assert_eq!(ctrl.state(), FtbState::Black);

        // Fade back from black
        ctrl.toggle();
        for _ in 0..5 {
            ctrl.advance_frame();
        }
        assert_eq!(ctrl.state(), FtbState::Normal);
        assert!((ctrl.fade_level()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_controller_audio_level() {
        let config = FtbConfig {
            fade_audio: true,
            duration_frames: 10,
            ..Default::default()
        };
        let mut ctrl = FtbController::new(config).expect("should succeed in test");
        ctrl.toggle();
        for _ in 0..10 {
            ctrl.advance_frame();
        }
        // Audio should be 0 when black
        assert!((ctrl.audio_level()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_controller_audio_no_follow() {
        let config = FtbConfig {
            fade_audio: false,
            duration_frames: 10,
            ..Default::default()
        };
        let mut ctrl = FtbController::new(config).expect("should succeed in test");
        ctrl.toggle();
        for _ in 0..10 {
            ctrl.advance_frame();
        }
        // Audio should still be 1.0 even when black
        assert!((ctrl.audio_level() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_controller_apply_to_frame_normal() {
        let ctrl = FtbController::with_defaults();
        let mut frame = vec![128u8; 16];
        ctrl.apply_to_frame(&mut frame);
        // No change - normal state
        assert!(frame.iter().all(|&b| b == 128));
    }

    #[test]
    fn test_controller_apply_to_frame_black() {
        let config = FtbConfig::new(1, FtbCurve::Linear);
        let mut ctrl = FtbController::new(config).expect("should succeed in test");
        ctrl.toggle();
        ctrl.advance_frame();
        assert_eq!(ctrl.state(), FtbState::Black);

        let mut frame = vec![200u8; 16];
        ctrl.apply_to_frame(&mut frame);
        assert!(frame.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_controller_interrupt_fade() {
        let config = FtbConfig::new(20, FtbCurve::Linear);
        let mut ctrl = FtbController::new(config).expect("should succeed in test");
        ctrl.toggle(); // Start fading to black

        for _ in 0..10 {
            ctrl.advance_frame();
        }
        assert!(ctrl.is_fading());
        let mid_level = ctrl.fade_level();
        assert!(mid_level > 0.0);
        assert!(mid_level < 1.0);

        // Interrupt: toggle back
        ctrl.toggle();
        assert_eq!(ctrl.state(), FtbState::FadingFromBlack);
    }

    #[test]
    fn test_controller_reset() {
        let config = FtbConfig::new(10, FtbCurve::Linear);
        let mut ctrl = FtbController::new(config).expect("should succeed in test");
        ctrl.toggle();
        for _ in 0..10 {
            ctrl.advance_frame();
        }
        assert_eq!(ctrl.state(), FtbState::Black);

        ctrl.reset();
        assert_eq!(ctrl.state(), FtbState::Normal);
        assert!((ctrl.fade_level()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_controller_set_config_while_fading() {
        let mut ctrl = FtbController::with_defaults();
        ctrl.toggle();
        let new_config = FtbConfig::new(50, FtbCurve::SCurve);
        let result = ctrl.set_config(new_config);
        assert!(result.is_err());
    }
}
