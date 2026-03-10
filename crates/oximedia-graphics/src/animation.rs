//! Animation system with keyframe interpolation and easing functions

use crate::primitives::Point;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Animation timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    /// Current time position
    pub current_time: Duration,
    /// Total duration
    pub duration: Duration,
    /// Loop mode
    pub loop_mode: LoopMode,
    /// Playback state
    pub state: PlaybackState,
}

impl Timeline {
    /// Create a new timeline
    #[must_use]
    pub fn new(duration: Duration) -> Self {
        Self {
            current_time: Duration::ZERO,
            duration,
            loop_mode: LoopMode::Once,
            state: PlaybackState::Stopped,
        }
    }

    /// Update timeline
    pub fn update(&mut self, delta: Duration) {
        if self.state != PlaybackState::Playing {
            return;
        }

        self.current_time += delta;

        match self.loop_mode {
            LoopMode::Once => {
                if self.current_time >= self.duration {
                    self.current_time = self.duration;
                    self.state = PlaybackState::Stopped;
                }
            }
            LoopMode::Loop => {
                while self.current_time >= self.duration {
                    self.current_time -= self.duration;
                }
            }
            LoopMode::PingPong => {
                if self.current_time >= self.duration {
                    let overflow = self
                        .current_time
                        .checked_sub(self.duration)
                        .unwrap_or(Duration::ZERO);
                    self.current_time = self
                        .duration
                        .checked_sub(overflow)
                        .unwrap_or(Duration::ZERO);
                }
            }
        }
    }

    /// Get normalized time (0.0 to 1.0)
    #[must_use]
    pub fn normalized_time(&self) -> f32 {
        if self.duration.is_zero() {
            return 0.0;
        }
        (self.current_time.as_secs_f32() / self.duration.as_secs_f32()).clamp(0.0, 1.0)
    }

    /// Play
    pub fn play(&mut self) {
        self.state = PlaybackState::Playing;
    }

    /// Pause
    pub fn pause(&mut self) {
        self.state = PlaybackState::Paused;
    }

    /// Stop and reset
    pub fn stop(&mut self) {
        self.state = PlaybackState::Stopped;
        self.current_time = Duration::ZERO;
    }
}

/// Playback state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaybackState {
    /// Playing
    Playing,
    /// Paused
    Paused,
    /// Stopped
    Stopped,
}

/// Loop mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopMode {
    /// Play once
    Once,
    /// Loop continuously
    Loop,
    /// Ping-pong (forward then backward)
    PingPong,
}

/// Easing function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Easing {
    /// Linear (no easing)
    Linear,
    /// Ease in (slow start)
    EaseIn,
    /// Ease out (slow end)
    EaseOut,
    /// Ease in-out (slow start and end)
    EaseInOut,
    /// Quadratic ease in
    QuadIn,
    /// Quadratic ease out
    QuadOut,
    /// Quadratic ease in-out
    QuadInOut,
    /// Cubic ease in
    CubicIn,
    /// Cubic ease out
    CubicOut,
    /// Cubic ease in-out
    CubicInOut,
    /// Elastic ease in
    ElasticIn,
    /// Elastic ease out
    ElasticOut,
    /// Bounce ease in
    BounceIn,
    /// Bounce ease out
    BounceOut,
}

impl Easing {
    /// Apply easing to normalized time (0.0 to 1.0)
    #[must_use]
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => t * (2.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            Self::QuadIn => t * t,
            Self::QuadOut => t * (2.0 - t),
            Self::QuadInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            Self::CubicIn => t * t * t,
            Self::CubicOut => {
                let t = t - 1.0;
                t * t * t + 1.0
            }
            Self::CubicInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let t = 2.0 * t - 2.0;
                    1.0 + t * t * t / 2.0
                }
            }
            Self::ElasticIn => {
                if t == 0.0 || t == 1.0 {
                    return t;
                }
                let p = 0.3;
                let s = p / 4.0;
                let t = t - 1.0;
                -(2.0_f32.powf(10.0 * t) * ((t - s) * (2.0 * std::f32::consts::PI) / p).sin())
            }
            Self::ElasticOut => {
                if t == 0.0 || t == 1.0 {
                    return t;
                }
                let p = 0.3;
                let s = p / 4.0;
                2.0_f32.powf(-10.0 * t) * ((t - s) * (2.0 * std::f32::consts::PI) / p).sin() + 1.0
            }
            Self::BounceIn => 1.0 - Self::BounceOut.apply(1.0 - t),
            Self::BounceOut => {
                const N1: f32 = 7.5625;
                const D1: f32 = 2.75;

                if t < 1.0 / D1 {
                    N1 * t * t
                } else if t < 2.0 / D1 {
                    let t = t - 1.5 / D1;
                    N1 * t * t + 0.75
                } else if t < 2.5 / D1 {
                    let t = t - 2.25 / D1;
                    N1 * t * t + 0.9375
                } else {
                    let t = t - 2.625 / D1;
                    N1 * t * t + 0.984375
                }
            }
        }
    }
}

/// Keyframe for animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keyframe<T> {
    /// Time position (0.0 to 1.0)
    pub time: f32,
    /// Value at this keyframe
    pub value: T,
    /// Easing to next keyframe
    pub easing: Easing,
}

impl<T> Keyframe<T> {
    /// Create a new keyframe
    pub fn new(time: f32, value: T, easing: Easing) -> Self {
        Self {
            time,
            value,
            easing,
        }
    }
}

/// Trait for animatable values
pub trait Animatable: Clone {
    /// Interpolate between two values
    fn lerp(&self, other: &Self, t: f32) -> Self;
}

impl Animatable for f32 {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Animatable for Point {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Point::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )
    }
}

/// Animation track for a single property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationTrack<T> {
    /// Keyframes
    pub keyframes: Vec<Keyframe<T>>,
}

impl<T: Animatable> AnimationTrack<T> {
    /// Create a new animation track
    #[must_use]
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe
    pub fn add_keyframe(&mut self, keyframe: Keyframe<T>) -> &mut Self {
        self.keyframes.push(keyframe);
        self.keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self
    }

    /// Evaluate at time (0.0 to 1.0)
    #[must_use]
    pub fn evaluate(&self, time: f32) -> Option<T> {
        if self.keyframes.is_empty() {
            return None;
        }

        if self.keyframes.len() == 1 {
            return Some(self.keyframes[0].value.clone());
        }

        // Find the two keyframes to interpolate between
        let mut prev = &self.keyframes[0];
        for keyframe in self.keyframes.iter().skip(1) {
            if time <= keyframe.time {
                let local_time = (time - prev.time) / (keyframe.time - prev.time);
                let eased_time = prev.easing.apply(local_time);
                return Some(prev.value.lerp(&keyframe.value, eased_time));
            }
            prev = keyframe;
        }

        // SAFETY: the loop above always assigns `prev` to a keyframe and we have len >= 2,
        // so `last()` is always Some here
        Some(
            self.keyframes
                .last()
                .expect("keyframes non-empty after len >= 2 check")
                .value
                .clone(),
        )
    }
}

impl<T: Animatable> Default for AnimationTrack<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Transform animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    /// Position
    pub position: Point,
    /// Scale
    pub scale: Point,
    /// Rotation (radians)
    pub rotation: f32,
    /// Opacity (0.0 to 1.0)
    pub opacity: f32,
}

impl Transform {
    /// Create identity transform
    #[must_use]
    pub fn identity() -> Self {
        Self {
            position: Point::new(0.0, 0.0),
            scale: Point::new(1.0, 1.0),
            rotation: 0.0,
            opacity: 1.0,
        }
    }

    /// Create from position
    #[must_use]
    pub fn from_position(position: Point) -> Self {
        Self {
            position,
            ..Self::identity()
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

/// Animation clip
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationClip {
    /// Position track
    pub position: Option<AnimationTrack<Point>>,
    /// Scale track
    pub scale: Option<AnimationTrack<Point>>,
    /// Rotation track
    pub rotation: Option<AnimationTrack<f32>>,
    /// Opacity track
    pub opacity: Option<AnimationTrack<f32>>,
    /// Duration
    pub duration: Duration,
}

impl AnimationClip {
    /// Create a new animation clip
    #[must_use]
    pub fn new(duration: Duration) -> Self {
        Self {
            position: None,
            scale: None,
            rotation: None,
            opacity: None,
            duration,
        }
    }

    /// Evaluate at timeline
    #[must_use]
    pub fn evaluate(&self, timeline: &Timeline) -> Transform {
        let t = timeline.normalized_time();
        let mut transform = Transform::identity();

        if let Some(ref track) = self.position {
            if let Some(pos) = track.evaluate(t) {
                transform.position = pos;
            }
        }

        if let Some(ref track) = self.scale {
            if let Some(scale) = track.evaluate(t) {
                transform.scale = scale;
            }
        }

        if let Some(ref track) = self.rotation {
            if let Some(rot) = track.evaluate(t) {
                transform.rotation = rot;
            }
        }

        if let Some(ref track) = self.opacity {
            if let Some(opacity) = track.evaluate(t) {
                transform.opacity = opacity;
            }
        }

        transform
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline() {
        let mut timeline = Timeline::new(Duration::from_secs(1));
        timeline.play();
        assert_eq!(timeline.state, PlaybackState::Playing);

        timeline.update(Duration::from_millis(500));
        assert!((timeline.normalized_time() - 0.5).abs() < 0.01);

        timeline.update(Duration::from_millis(500));
        assert_eq!(timeline.state, PlaybackState::Stopped);
    }

    #[test]
    fn test_timeline_loop() {
        let mut timeline = Timeline::new(Duration::from_secs(1));
        timeline.loop_mode = LoopMode::Loop;
        timeline.play();

        timeline.update(Duration::from_millis(1500));
        assert!((timeline.normalized_time() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_easing_linear() {
        let easing = Easing::Linear;
        assert_eq!(easing.apply(0.0), 0.0);
        assert_eq!(easing.apply(0.5), 0.5);
        assert_eq!(easing.apply(1.0), 1.0);
    }

    #[test]
    fn test_easing_ease_in() {
        let easing = Easing::EaseIn;
        assert_eq!(easing.apply(0.0), 0.0);
        assert!(easing.apply(0.5) < 0.5);
        assert_eq!(easing.apply(1.0), 1.0);
    }

    #[test]
    fn test_easing_ease_out() {
        let easing = Easing::EaseOut;
        assert_eq!(easing.apply(0.0), 0.0);
        assert!(easing.apply(0.5) > 0.5);
        assert_eq!(easing.apply(1.0), 1.0);
    }

    #[test]
    fn test_animation_track() {
        let mut track = AnimationTrack::new();
        track.add_keyframe(Keyframe::new(0.0, 0.0_f32, Easing::Linear));
        track.add_keyframe(Keyframe::new(1.0, 100.0, Easing::Linear));

        assert_eq!(track.evaluate(0.0), Some(0.0));
        assert_eq!(track.evaluate(0.5), Some(50.0));
        assert_eq!(track.evaluate(1.0), Some(100.0));
    }

    #[test]
    fn test_animation_track_point() {
        let mut track = AnimationTrack::new();
        track.add_keyframe(Keyframe::new(0.0, Point::new(0.0, 0.0), Easing::Linear));
        track.add_keyframe(Keyframe::new(1.0, Point::new(100.0, 100.0), Easing::Linear));

        let pos = track.evaluate(0.5).expect("pos should be valid");
        assert_eq!(pos, Point::new(50.0, 50.0));
    }

    #[test]
    fn test_transform() {
        let transform = Transform::identity();
        assert_eq!(transform.position, Point::new(0.0, 0.0));
        assert_eq!(transform.scale, Point::new(1.0, 1.0));
        assert_eq!(transform.rotation, 0.0);
        assert_eq!(transform.opacity, 1.0);
    }

    #[test]
    fn test_animation_clip() {
        let mut clip = AnimationClip::new(Duration::from_secs(1));
        let mut position_track = AnimationTrack::new();
        position_track.add_keyframe(Keyframe::new(0.0, Point::new(0.0, 0.0), Easing::Linear));
        position_track.add_keyframe(Keyframe::new(1.0, Point::new(100.0, 100.0), Easing::Linear));
        clip.position = Some(position_track);

        let mut timeline = Timeline::new(Duration::from_secs(1));
        timeline.current_time = Duration::from_millis(500);

        let transform = clip.evaluate(&timeline);
        assert_eq!(transform.position, Point::new(50.0, 50.0));
    }
}
