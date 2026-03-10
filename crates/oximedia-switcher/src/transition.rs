//! Transition engine for video switchers.
//!
//! Implements various transition types: cut, mix/dissolve, wipe, and DVE effects.

use oximedia_codec::VideoFrame;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during transitions.
#[derive(Error, Debug, Clone)]
pub enum TransitionError {
    #[error("Transition already in progress")]
    AlreadyInProgress,

    #[error("No transition in progress")]
    NoTransition,

    #[error("Invalid transition duration: {0}")]
    InvalidDuration(u32),

    #[error("Invalid position: {0}")]
    InvalidPosition(f32),

    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Transition type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TransitionType {
    /// Cut (instant)
    Cut,
    /// Mix/Dissolve (crossfade)
    Mix,
    /// Dip to color
    Dip,
    /// Wipe with pattern
    Wipe(WipePattern),
    /// DVE (digital video effect)
    Dve(DveType),
}

/// Wipe pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WipePattern {
    /// Horizontal wipe
    Horizontal,
    /// Vertical wipe
    Vertical,
    /// Diagonal from top-left
    DiagonalTopLeft,
    /// Diagonal from top-right
    DiagonalTopRight,
    /// Circle wipe
    Circle,
    /// Diamond wipe
    Diamond,
    /// Box wipe (from center)
    Box,
    /// Barn door (horizontal)
    BarnDoorHorizontal,
    /// Barn door (vertical)
    BarnDoorVertical,
    /// Iris
    Iris,
    /// Custom pattern (pattern ID)
    Custom(u32),
}

/// DVE transition types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DveType {
    /// Push (slide)
    Push,
    /// Squeeze
    Squeeze,
    /// Spin
    Spin,
    /// Fly
    Fly,
    /// Cube rotate
    Cube,
}

/// Transition direction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TransitionDirection {
    /// Left to right / Top to bottom
    Forward,
    /// Right to left / Bottom to top
    Reverse,
}

/// Wipe configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WipeConfig {
    /// Wipe pattern
    pub pattern: WipePattern,
    /// Border width (0.0 - 1.0)
    pub border_width: f32,
    /// Border softness (0.0 - 1.0)
    pub border_softness: f32,
    /// Direction
    pub direction: TransitionDirection,
}

impl WipeConfig {
    /// Create a new wipe configuration.
    pub fn new(pattern: WipePattern) -> Self {
        Self {
            pattern,
            border_width: 0.0,
            border_softness: 0.0,
            direction: TransitionDirection::Forward,
        }
    }
}

/// Transition configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionConfig {
    /// Transition type
    pub transition_type: TransitionType,
    /// Duration in frames
    pub duration_frames: u32,
    /// Wipe configuration (if applicable)
    pub wipe_config: Option<WipeConfig>,
}

impl TransitionConfig {
    /// Create a new transition configuration.
    pub fn new(transition_type: TransitionType, duration_frames: u32) -> Self {
        Self {
            transition_type,
            duration_frames,
            wipe_config: None,
        }
    }

    /// Create a cut transition (0 frames).
    pub fn cut() -> Self {
        Self::new(TransitionType::Cut, 0)
    }

    /// Create a mix transition.
    pub fn mix(duration_frames: u32) -> Self {
        Self::new(TransitionType::Mix, duration_frames)
    }

    /// Create a wipe transition.
    pub fn wipe(pattern: WipePattern, duration_frames: u32) -> Self {
        let mut config = Self::new(TransitionType::Wipe(pattern), duration_frames);
        config.wipe_config = Some(WipeConfig::new(pattern));
        config
    }
}

/// Transition state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TransitionState {
    /// No transition in progress
    Idle,
    /// Transition in progress
    InProgress,
    /// Transition paused
    Paused,
}

/// Transition engine manages transition execution.
pub struct TransitionEngine {
    /// Current configuration
    config: TransitionConfig,
    /// Current state
    state: TransitionState,
    /// Current position (0.0 - 1.0)
    position: f32,
    /// Current frame in transition
    current_frame: u32,
    /// Source A (usually program)
    source_a: usize,
    /// Source B (usually preview)
    source_b: usize,
}

impl TransitionEngine {
    /// Create a new transition engine.
    pub fn new() -> Self {
        Self {
            config: TransitionConfig::cut(),
            state: TransitionState::Idle,
            position: 0.0,
            current_frame: 0,
            source_a: 0,
            source_b: 0,
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &TransitionConfig {
        &self.config
    }

    /// Set the transition configuration.
    pub fn set_config(&mut self, config: TransitionConfig) {
        self.config = config;
    }

    /// Get the current state.
    pub fn state(&self) -> TransitionState {
        self.state
    }

    /// Get the current position (0.0 - 1.0).
    pub fn position(&self) -> f32 {
        self.position
    }

    /// Set the position manually (for cut bar control).
    pub fn set_position(&mut self, position: f32) -> Result<(), TransitionError> {
        if !(0.0..=1.0).contains(&position) {
            return Err(TransitionError::InvalidPosition(position));
        }
        self.position = position;
        self.current_frame = (position * self.config.duration_frames as f32) as u32;
        Ok(())
    }

    /// Start a transition.
    pub fn start(&mut self, source_a: usize, source_b: usize) -> Result<(), TransitionError> {
        if self.state == TransitionState::InProgress {
            return Err(TransitionError::AlreadyInProgress);
        }

        self.source_a = source_a;
        self.source_b = source_b;
        self.position = 0.0;
        self.current_frame = 0;
        self.state = TransitionState::InProgress;

        Ok(())
    }

    /// Take - start an auto transition.
    pub fn take(&mut self, source_a: usize, source_b: usize) -> Result<(), TransitionError> {
        self.start(source_a, source_b)
    }

    /// Advance the transition by one frame.
    pub fn advance(&mut self) -> Result<bool, TransitionError> {
        if self.state != TransitionState::InProgress {
            return Err(TransitionError::NoTransition);
        }

        if self.config.duration_frames == 0 {
            // Cut - instant transition
            self.position = 1.0;
            self.state = TransitionState::Idle;
            return Ok(true);
        }

        self.current_frame += 1;
        self.position = self.current_frame as f32 / self.config.duration_frames as f32;

        if self.position >= 1.0 {
            self.position = 1.0;
            self.state = TransitionState::Idle;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Pause the transition.
    pub fn pause(&mut self) -> Result<(), TransitionError> {
        if self.state != TransitionState::InProgress {
            return Err(TransitionError::NoTransition);
        }
        self.state = TransitionState::Paused;
        Ok(())
    }

    /// Resume the transition.
    pub fn resume(&mut self) -> Result<(), TransitionError> {
        if self.state != TransitionState::Paused {
            return Err(TransitionError::NoTransition);
        }
        self.state = TransitionState::InProgress;
        Ok(())
    }

    /// Cancel the transition.
    pub fn cancel(&mut self) {
        self.state = TransitionState::Idle;
        self.position = 0.0;
        self.current_frame = 0;
    }

    /// Check if a transition is in progress.
    pub fn is_in_progress(&self) -> bool {
        self.state == TransitionState::InProgress
    }

    /// Check if the transition is complete.
    pub fn is_complete(&self) -> bool {
        self.position >= 1.0 && self.state == TransitionState::Idle
    }

    /// Get the duration in frames.
    pub fn duration_frames(&self) -> u32 {
        self.config.duration_frames
    }

    /// Get the duration as a Duration.
    pub fn duration(&self, frame_rate: f64) -> Duration {
        let seconds = self.config.duration_frames as f64 / frame_rate;
        Duration::from_secs_f64(seconds)
    }

    /// Get the current frame number.
    pub fn current_frame(&self) -> u32 {
        self.current_frame
    }

    /// Get source A.
    pub fn source_a(&self) -> usize {
        self.source_a
    }

    /// Get source B.
    pub fn source_b(&self) -> usize {
        self.source_b
    }

    /// Process a transition frame by compositing `frame_a` and `frame_b`
    /// according to the current transition type and position.
    ///
    /// For *Cut* the output is `frame_b` once position >= 1.0.
    /// For *Mix / Dip* a linear crossfade is applied.
    /// For *Wipe* a spatial boundary separates the two sources.
    /// For *DVE* the mix ratio is used as a simple crossfade (the full DVE
    /// path is handled by `DveProcessor`).
    pub fn process(
        &self,
        frame_a: &VideoFrame,
        frame_b: &VideoFrame,
    ) -> Result<VideoFrame, TransitionError> {
        if frame_a.width != frame_b.width || frame_a.height != frame_b.height {
            return Err(TransitionError::ProcessingError(
                "Frame dimensions do not match".to_string(),
            ));
        }

        if frame_a.planes.is_empty() || frame_b.planes.is_empty() {
            return Err(TransitionError::ProcessingError(
                "Frames have no planes".to_string(),
            ));
        }

        // Cut: instant swap
        if matches!(self.config.transition_type, TransitionType::Cut) {
            return if self.position >= 1.0 {
                Ok(frame_b.clone())
            } else {
                Ok(frame_a.clone())
            };
        }

        let ratio = self.mix_ratio();

        // Create output frame
        let mut output = VideoFrame::new(frame_a.format, frame_a.width, frame_a.height);
        output.allocate();
        output.timestamp = frame_a.timestamp;
        output.frame_type = frame_a.frame_type;
        output.color_info = frame_a.color_info;

        let plane_count = output
            .planes
            .len()
            .min(frame_a.planes.len())
            .min(frame_b.planes.len());

        match self.config.transition_type {
            TransitionType::Cut => unreachable!(),
            TransitionType::Mix | TransitionType::Dip | TransitionType::Dve(_) => {
                // Linear crossfade on every plane
                for pi in 0..plane_count {
                    let pa = &frame_a.planes[pi];
                    let pb = &frame_b.planes[pi];
                    let po = &mut output.planes[pi];

                    let len = po.data.len().min(pa.data.len()).min(pb.data.len());
                    for i in 0..len {
                        let a_val = pa.data[i] as f32;
                        let b_val = pb.data[i] as f32;
                        po.data[i] = (a_val * (1.0 - ratio) + b_val * ratio) as u8;
                    }
                }
            }
            TransitionType::Wipe(pattern) => {
                // Spatial wipe: for each pixel determine whether it comes
                // from source A or B (with optional soft edge).
                for pi in 0..plane_count {
                    let pa = &frame_a.planes[pi];
                    let pb = &frame_b.planes[pi];
                    let po = &mut output.planes[pi];

                    let pw = po.width as usize;
                    let ph = po.height as usize;

                    let softness = self
                        .config
                        .wipe_config
                        .as_ref()
                        .map_or(0.02, |c| c.border_softness as f64)
                        as f32;
                    let half_soft = softness / 2.0;

                    for y in 0..ph {
                        for x in 0..pw {
                            let nx = x as f32 / pw.max(1) as f32;
                            let ny = y as f32 / ph.max(1) as f32;

                            // Compute wipe boundary value (0.0..1.0)
                            let boundary = match pattern {
                                WipePattern::Horizontal => nx,
                                WipePattern::Vertical => ny,
                                WipePattern::DiagonalTopLeft => (nx + ny) / 2.0,
                                WipePattern::DiagonalTopRight => ((1.0 - nx) + ny) / 2.0,
                                WipePattern::Circle => {
                                    let dx = nx - 0.5;
                                    let dy = ny - 0.5;
                                    ((dx * dx + dy * dy).sqrt() * 2.0).min(1.0)
                                }
                                WipePattern::Diamond => {
                                    ((nx - 0.5).abs() + (ny - 0.5).abs()).min(1.0)
                                }
                                WipePattern::Box => {
                                    let bx = (nx - 0.5).abs() * 2.0;
                                    let by = (ny - 0.5).abs() * 2.0;
                                    bx.max(by).min(1.0)
                                }
                                WipePattern::BarnDoorHorizontal => {
                                    ((nx - 0.5).abs() * 2.0).min(1.0)
                                }
                                WipePattern::BarnDoorVertical => ((ny - 0.5).abs() * 2.0).min(1.0),
                                WipePattern::Iris => {
                                    let dx = nx - 0.5;
                                    let dy = ny - 0.5;
                                    1.0 - ((dx * dx + dy * dy).sqrt() * 2.0).min(1.0)
                                }
                                WipePattern::Custom(_) => nx, // fallback to horizontal
                            };

                            // Soft edge: smoothly blend around the boundary
                            let local_ratio = if half_soft > f32::EPSILON {
                                ((ratio - boundary + half_soft) / (2.0 * half_soft)).clamp(0.0, 1.0)
                            } else if boundary <= ratio {
                                1.0
                            } else {
                                0.0
                            };

                            let idx = y * po.stride + x;
                            if idx < po.data.len() && idx < pa.data.len() && idx < pb.data.len() {
                                let a_val = pa.data[idx] as f32;
                                let b_val = pb.data[idx] as f32;
                                po.data[idx] =
                                    (a_val * (1.0 - local_ratio) + b_val * local_ratio) as u8;
                            }
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Calculate mix ratio for current position.
    pub fn mix_ratio(&self) -> f32 {
        match self.config.transition_type {
            TransitionType::Cut => {
                if self.position >= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            TransitionType::Mix | TransitionType::Dip => self.position,
            TransitionType::Wipe(_) => self.position,
            TransitionType::Dve(_) => self.position,
        }
    }
}

impl Default for TransitionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Transition preview - shows what the transition will look like.
pub struct TransitionPreview {
    engine: TransitionEngine,
    preview_position: f32,
}

impl TransitionPreview {
    /// Create a new transition preview.
    pub fn new(config: TransitionConfig) -> Self {
        let mut engine = TransitionEngine::new();
        engine.set_config(config);

        Self {
            engine,
            preview_position: 0.5,
        }
    }

    /// Set the preview position (0.0 - 1.0).
    pub fn set_position(&mut self, position: f32) -> Result<(), TransitionError> {
        self.preview_position = position.clamp(0.0, 1.0);
        self.engine.set_position(self.preview_position)
    }

    /// Get the preview position.
    pub fn position(&self) -> f32 {
        self.preview_position
    }

    /// Get the engine.
    pub fn engine(&self) -> &TransitionEngine {
        &self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_config_cut() {
        let config = TransitionConfig::cut();
        assert_eq!(config.transition_type, TransitionType::Cut);
        assert_eq!(config.duration_frames, 0);
    }

    #[test]
    fn test_transition_config_mix() {
        let config = TransitionConfig::mix(30);
        assert_eq!(config.transition_type, TransitionType::Mix);
        assert_eq!(config.duration_frames, 30);
    }

    #[test]
    fn test_transition_config_wipe() {
        let config = TransitionConfig::wipe(WipePattern::Horizontal, 45);
        assert!(matches!(config.transition_type, TransitionType::Wipe(_)));
        assert_eq!(config.duration_frames, 45);
        assert!(config.wipe_config.is_some());
    }

    #[test]
    fn test_transition_engine_creation() {
        let engine = TransitionEngine::new();
        assert_eq!(engine.state(), TransitionState::Idle);
        assert_eq!(engine.position(), 0.0);
        assert!(!engine.is_in_progress());
    }

    #[test]
    fn test_start_transition() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::mix(30));

        engine.start(1, 2).expect("should succeed in test");
        assert_eq!(engine.state(), TransitionState::InProgress);
        assert!(engine.is_in_progress());
        assert_eq!(engine.source_a(), 1);
        assert_eq!(engine.source_b(), 2);
    }

    #[test]
    fn test_advance_transition() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::mix(10));
        engine.start(1, 2).expect("should succeed in test");

        // Advance through transition
        for i in 1..=10 {
            let complete = engine.advance().expect("should succeed in test");
            if i < 10 {
                assert!(!complete);
                assert!(engine.is_in_progress());
            } else {
                assert!(complete);
                assert!(!engine.is_in_progress());
            }
        }

        assert_eq!(engine.position(), 1.0);
        assert!(engine.is_complete());
    }

    #[test]
    fn test_cut_transition() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::cut());
        engine.start(1, 2).expect("should succeed in test");

        let complete = engine.advance().expect("should succeed in test");
        assert!(complete);
        assert_eq!(engine.position(), 1.0);
    }

    #[test]
    fn test_pause_resume() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::mix(30));
        engine.start(1, 2).expect("should succeed in test");

        engine.advance().expect("should succeed in test");
        engine.pause().expect("should succeed in test");
        assert_eq!(engine.state(), TransitionState::Paused);

        engine.resume().expect("should succeed in test");
        assert_eq!(engine.state(), TransitionState::InProgress);
    }

    #[test]
    fn test_cancel_transition() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::mix(30));
        engine.start(1, 2).expect("should succeed in test");

        engine.advance().expect("should succeed in test");
        assert!(engine.position() > 0.0);

        engine.cancel();
        assert_eq!(engine.state(), TransitionState::Idle);
        assert_eq!(engine.position(), 0.0);
    }

    #[test]
    fn test_set_position() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::mix(100));

        engine.set_position(0.5).expect("should succeed in test");
        assert_eq!(engine.position(), 0.5);
        assert_eq!(engine.current_frame(), 50);

        assert!(engine.set_position(-0.1).is_err());
        assert!(engine.set_position(1.5).is_err());
    }

    #[test]
    fn test_mix_ratio() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::mix(100));

        engine.set_position(0.0).expect("should succeed in test");
        assert_eq!(engine.mix_ratio(), 0.0);

        engine.set_position(0.5).expect("should succeed in test");
        assert_eq!(engine.mix_ratio(), 0.5);

        engine.set_position(1.0).expect("should succeed in test");
        assert_eq!(engine.mix_ratio(), 1.0);
    }

    #[test]
    fn test_mix_ratio_cut() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::cut());

        engine.set_position(0.0).expect("should succeed in test");
        assert_eq!(engine.mix_ratio(), 0.0);

        engine.set_position(0.5).expect("should succeed in test");
        assert_eq!(engine.mix_ratio(), 0.0);

        engine.set_position(1.0).expect("should succeed in test");
        assert_eq!(engine.mix_ratio(), 1.0);
    }

    #[test]
    fn test_already_in_progress() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::mix(30));

        engine.start(1, 2).expect("should succeed in test");
        assert!(engine.start(3, 4).is_err());
    }

    #[test]
    fn test_wipe_config() {
        let mut config = WipeConfig::new(WipePattern::Horizontal);
        assert_eq!(config.pattern, WipePattern::Horizontal);
        assert_eq!(config.border_width, 0.0);
        assert_eq!(config.direction, TransitionDirection::Forward);

        config.border_width = 0.1;
        config.border_softness = 0.2;
        config.direction = TransitionDirection::Reverse;

        assert_eq!(config.border_width, 0.1);
        assert_eq!(config.border_softness, 0.2);
        assert_eq!(config.direction, TransitionDirection::Reverse);
    }

    #[test]
    fn test_wipe_patterns() {
        assert_eq!(WipePattern::Horizontal, WipePattern::Horizontal);
        assert_ne!(WipePattern::Horizontal, WipePattern::Vertical);
        assert!(matches!(WipePattern::Circle, WipePattern::Circle));
        assert!(matches!(WipePattern::Custom(1), WipePattern::Custom(_)));
    }

    #[test]
    fn test_dve_types() {
        assert_eq!(DveType::Push, DveType::Push);
        assert_ne!(DveType::Push, DveType::Squeeze);
        assert!(matches!(DveType::Spin, DveType::Spin));
    }

    #[test]
    fn test_transition_preview() {
        let config = TransitionConfig::mix(30);
        let mut preview = TransitionPreview::new(config);

        assert_eq!(preview.position(), 0.5);

        preview.set_position(0.75).expect("should succeed in test");
        assert_eq!(preview.position(), 0.75);
        assert_eq!(preview.engine().position(), 0.75);
    }

    #[test]
    fn test_duration_conversion() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::mix(25));

        let duration = engine.duration(25.0); // 25 fps
        assert_eq!(duration.as_secs(), 1);
    }

    #[test]
    fn test_take() {
        let mut engine = TransitionEngine::new();
        engine.set_config(TransitionConfig::mix(30));

        engine.take(1, 2).expect("should succeed in test");
        assert!(engine.is_in_progress());
        assert_eq!(engine.source_a(), 1);
        assert_eq!(engine.source_b(), 2);
    }
}
