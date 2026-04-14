//! Camera rig path recording and keyframe-based playback.
//!
//! This module provides facilities for:
//! - Recording live rig telemetry into a named path
//! - Storing keyframes with position, orientation, and lens metadata
//! - Interpolating between keyframes using Catmull-Rom splines (position)
//!   and spherical-linear interpolation (orientation)
//! - Velocity-profile shaping (ease-in / ease-out)
//! - Loop modes: clamp, ping-pong, wrap
//! - Serialisation to/from JSON for session persistence

use crate::camera_rig::{Orientation, Position, RigFrame};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Velocity profile
// ---------------------------------------------------------------------------

/// Velocity (easing) profile applied to path playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EasingMode {
    /// Constant velocity.
    Linear,
    /// Accelerate at start, constant in middle, decelerate at end.
    EaseInOut,
    /// Only decelerate at end.
    EaseOut,
    /// Only accelerate at start.
    EaseIn,
    /// Smooth step (cubic Hermite: 3t² - 2t³).
    SmoothStep,
}

impl EasingMode {
    /// Map a normalised time `t ∈ [0, 1]` through the easing curve.
    #[must_use]
    pub fn apply(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseInOut => {
                // Cubic ease-in-out
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                }
            }
            Self::EaseOut => {
                // Quadratic ease-out
                1.0 - (1.0 - t) * (1.0 - t)
            }
            Self::EaseIn => {
                // Quadratic ease-in
                t * t
            }
            Self::SmoothStep => t * t * (3.0 - 2.0 * t),
        }
    }
}

// ---------------------------------------------------------------------------
// Loop mode
// ---------------------------------------------------------------------------

/// Playback loop behaviour at path end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LoopMode {
    /// Stop at the last keyframe.
    Clamp,
    /// Reverse direction at each end (ping-pong).
    PingPong,
    /// Wrap back to the start seamlessly.
    Wrap,
}

// ---------------------------------------------------------------------------
// PathKeyframe
// ---------------------------------------------------------------------------

/// A single keyframe in a rig path.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PathKeyframe {
    /// Normalised time within the path segment `[0, 1]`.
    pub time: f64,
    /// World-space position.
    pub position: [f64; 3],
    /// Euler orientation (pan, tilt, roll) in degrees.
    pub orientation: [f64; 3],
    /// Lens focal length in mm (optional).
    pub focal_length_mm: Option<f64>,
    /// Focus distance in metres (optional).
    pub focus_distance_m: Option<f64>,
}

impl PathKeyframe {
    /// Construct from a [`RigFrame`] at the given normalised time.
    #[must_use]
    pub fn from_rig_frame(frame: &RigFrame, time: f64) -> Self {
        Self {
            time: time.clamp(0.0, 1.0),
            position: [frame.position.x, frame.position.y, frame.position.z],
            orientation: [
                frame.orientation.pan,
                frame.orientation.tilt,
                frame.orientation.roll,
            ],
            focal_length_mm: frame.focal_length_mm,
            focus_distance_m: frame.focus_distance_m,
        }
    }

    /// Position as a [`Position`].
    #[must_use]
    pub fn as_position(&self) -> Position {
        Position::new(self.position[0], self.position[1], self.position[2])
    }

    /// Orientation as an [`Orientation`].
    #[must_use]
    pub fn as_orientation(&self) -> Orientation {
        Orientation::new(
            self.orientation[0],
            self.orientation[1],
            self.orientation[2],
        )
    }
}

// ---------------------------------------------------------------------------
// Catmull-Rom spline helpers
// ---------------------------------------------------------------------------

/// Catmull-Rom interpolation for a single axis.
///
/// `p0`..`p3` are four control points; `t` is the parameter in `[0, 1]`
/// for the segment between `p1` and `p2`.
#[must_use]
fn catmull_rom(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Interpolate position using Catmull-Rom splines across four keyframes.
#[must_use]
fn interpolate_position_cr(
    k0: &PathKeyframe,
    k1: &PathKeyframe,
    k2: &PathKeyframe,
    k3: &PathKeyframe,
    t: f64,
) -> Position {
    let x = catmull_rom(
        k0.position[0],
        k1.position[0],
        k2.position[0],
        k3.position[0],
        t,
    );
    let y = catmull_rom(
        k0.position[1],
        k1.position[1],
        k2.position[1],
        k3.position[1],
        t,
    );
    let z = catmull_rom(
        k0.position[2],
        k1.position[2],
        k2.position[2],
        k3.position[2],
        t,
    );
    Position::new(x, y, z)
}

/// Interpolate orientation linearly (shortest-path per-component).
#[must_use]
fn interpolate_orientation(a: &Orientation, b: &Orientation, t: f64) -> Orientation {
    fn lerp_angle(a: f64, b: f64, t: f64) -> f64 {
        // shortest path on the circle
        let mut diff = b - a;
        while diff > 180.0 {
            diff -= 360.0;
        }
        while diff < -180.0 {
            diff += 360.0;
        }
        a + diff * t
    }
    Orientation::new(
        lerp_angle(a.pan, b.pan, t),
        lerp_angle(a.tilt, b.tilt, t),
        lerp_angle(a.roll, b.roll, t),
    )
}

// ---------------------------------------------------------------------------
// RigPath
// ---------------------------------------------------------------------------

/// A recorded or hand-authored camera path.
///
/// Keyframes are stored sorted by their `time` field.  Position is
/// interpolated using Catmull-Rom splines; orientation uses shortest-path
/// linear interpolation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RigPath {
    /// Human-readable name.
    pub name: String,
    /// Total real-world duration in seconds.
    pub duration_secs: f64,
    /// Sorted keyframes.
    keyframes: Vec<PathKeyframe>,
    /// Easing profile for playback.
    pub easing: EasingMode,
    /// Loop mode.
    pub loop_mode: LoopMode,
}

impl RigPath {
    /// Create an empty path.
    #[must_use]
    pub fn new(name: impl Into<String>, duration_secs: f64) -> Self {
        Self {
            name: name.into(),
            duration_secs: duration_secs.max(0.0),
            keyframes: Vec::new(),
            easing: EasingMode::Linear,
            loop_mode: LoopMode::Clamp,
        }
    }

    /// Builder: set easing mode.
    #[must_use]
    pub fn with_easing(mut self, easing: EasingMode) -> Self {
        self.easing = easing;
        self
    }

    /// Builder: set loop mode.
    #[must_use]
    pub fn with_loop_mode(mut self, mode: LoopMode) -> Self {
        self.loop_mode = mode;
        self
    }

    /// Add a keyframe, keeping the list sorted.
    pub fn add_keyframe(&mut self, kf: PathKeyframe) {
        let t = kf.time.clamp(0.0, 1.0);
        let idx = self
            .keyframes
            .partition_point(|k| k.time <= t);
        self.keyframes.insert(idx, kf);
    }

    /// Number of keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Read-only access to the keyframe list.
    #[must_use]
    pub fn keyframes(&self) -> &[PathKeyframe] {
        &self.keyframes
    }

    /// Evaluate the path at a normalised time `t ∈ [0, 1]` (before easing).
    ///
    /// Returns `None` if there are fewer than 2 keyframes.
    #[must_use]
    pub fn evaluate(&self, raw_t: f64) -> Option<PathKeyframe> {
        let n = self.keyframes.len();
        if n < 2 {
            return None;
        }

        // Apply loop mode
        let t = self.resolve_t(raw_t);
        // Apply easing
        let t = self.easing.apply(t);

        // Find segment: largest index where keyframe.time <= t
        let idx = self
            .keyframes
            .partition_point(|k| k.time <= t)
            .saturating_sub(1)
            .min(n - 2);

        let k1 = &self.keyframes[idx];
        let k2 = &self.keyframes[idx + 1];

        // Local t within this segment
        let seg_len = k2.time - k1.time;
        let local_t = if seg_len.abs() < f64::EPSILON {
            0.0
        } else {
            ((t - k1.time) / seg_len).clamp(0.0, 1.0)
        };

        // Ghost keyframes for Catmull-Rom (replicate endpoints)
        let k0 = if idx == 0 {
            k1
        } else {
            &self.keyframes[idx - 1]
        };
        let k3 = if idx + 2 >= n {
            k2
        } else {
            &self.keyframes[idx + 2]
        };

        let pos = interpolate_position_cr(k0, k1, k2, k3, local_t);
        let ori = interpolate_orientation(&k1.as_orientation(), &k2.as_orientation(), local_t);

        let focal = k1
            .focal_length_mm
            .and_then(|f1| k2.focal_length_mm.map(|f2| f1 + (f2 - f1) * local_t));
        let focus = k1
            .focus_distance_m
            .and_then(|d1| k2.focus_distance_m.map(|d2| d1 + (d2 - d1) * local_t));

        Some(PathKeyframe {
            time: t,
            position: [pos.x, pos.y, pos.z],
            orientation: [ori.pan, ori.tilt, ori.roll],
            focal_length_mm: focal,
            focus_distance_m: focus,
        })
    }

    /// Resolve raw time `t` according to the loop mode.
    fn resolve_t(&self, raw_t: f64) -> f64 {
        match self.loop_mode {
            LoopMode::Clamp => raw_t.clamp(0.0, 1.0),
            LoopMode::Wrap => {
                let t = raw_t.rem_euclid(1.0);
                if t < 0.0 {
                    0.0
                } else {
                    t
                }
            }
            LoopMode::PingPong => {
                let cycle = raw_t.rem_euclid(2.0);
                if cycle <= 1.0 {
                    cycle
                } else {
                    2.0 - cycle
                }
            }
        }
    }

    /// Bake `num_frames` evenly-spaced samples from the path.
    ///
    /// Useful for exporting to animation curves.  Returns `None` if the
    /// path has fewer than 2 keyframes.
    #[must_use]
    pub fn bake(&self, num_frames: usize) -> Option<Vec<PathKeyframe>> {
        if num_frames == 0 || self.keyframes.len() < 2 {
            return None;
        }
        let samples: Vec<PathKeyframe> = (0..num_frames)
            .filter_map(|i| {
                let t = i as f64 / (num_frames - 1).max(1) as f64;
                self.evaluate(t)
            })
            .collect();
        Some(samples)
    }

    /// Convert the path to a JSON string.
    ///
    /// # Errors
    /// Returns an error if serialisation fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Load a path from a JSON string.
    ///
    /// # Errors
    /// Returns an error if deserialisation fails.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ---------------------------------------------------------------------------
// PathLibrary
// ---------------------------------------------------------------------------

/// A named collection of [`RigPath`] objects.
#[derive(Debug, Default, Clone)]
pub struct PathLibrary {
    paths: HashMap<String, RigPath>,
}

impl PathLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a path. Overwrites any existing path with the same name.
    pub fn store(&mut self, path: RigPath) {
        self.paths.insert(path.name.clone(), path);
    }

    /// Retrieve a path by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&RigPath> {
        self.paths.get(name)
    }

    /// Remove a path.
    pub fn remove(&mut self, name: &str) -> Option<RigPath> {
        self.paths.remove(name)
    }

    /// Number of stored paths.
    #[must_use]
    pub fn count(&self) -> usize {
        self.paths.len()
    }

    /// Iterate over all stored paths.
    pub fn iter(&self) -> impl Iterator<Item = &RigPath> {
        self.paths.values()
    }
}

// ---------------------------------------------------------------------------
// PathRecorder
// ---------------------------------------------------------------------------

/// Records live rig telemetry into a [`RigPath`].
///
/// The recorder accumulates [`RigFrame`] data during a take and
/// converts them into sorted [`PathKeyframe`]s at normalised times.
pub struct PathRecorder {
    path_name: String,
    duration_secs: f64,
    easing: EasingMode,
    loop_mode: LoopMode,
    frames: Vec<(f64, RigFrame)>, // (elapsed_secs, frame)
    recording: bool,
}

impl PathRecorder {
    /// Create a recorder for a new take.
    #[must_use]
    pub fn new(path_name: impl Into<String>, duration_secs: f64) -> Self {
        Self {
            path_name: path_name.into(),
            duration_secs: duration_secs.max(0.001),
            easing: EasingMode::Linear,
            loop_mode: LoopMode::Clamp,
            frames: Vec::new(),
            recording: false,
        }
    }

    /// Builder: set easing mode for the finished path.
    #[must_use]
    pub fn with_easing(mut self, easing: EasingMode) -> Self {
        self.easing = easing;
        self
    }

    /// Builder: set loop mode for the finished path.
    #[must_use]
    pub fn with_loop_mode(mut self, mode: LoopMode) -> Self {
        self.loop_mode = mode;
        self
    }

    /// Start recording.
    pub fn start(&mut self) {
        self.frames.clear();
        self.recording = true;
    }

    /// Stop recording. Does not finalise; call [`finish`](Self::finish).
    pub fn stop(&mut self) {
        self.recording = false;
    }

    /// Whether currently recording.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Push a telemetry frame at `elapsed_secs` since recording start.
    ///
    /// Silently ignored if not currently recording.
    pub fn push_frame(&mut self, elapsed_secs: f64, frame: RigFrame) {
        if self.recording {
            self.frames.push((elapsed_secs, frame));
        }
    }

    /// Finalise the recording and return a [`RigPath`].
    ///
    /// Returns `None` if fewer than 2 frames were recorded.
    pub fn finish(self) -> Option<RigPath> {
        if self.frames.len() < 2 {
            return None;
        }
        let total = self.duration_secs;
        let mut path = RigPath::new(self.path_name, total)
            .with_easing(self.easing)
            .with_loop_mode(self.loop_mode);

        for (elapsed, frame) in &self.frames {
            let t = (elapsed / total).clamp(0.0, 1.0);
            let kf = PathKeyframe::from_rig_frame(frame, t);
            path.add_keyframe(kf);
        }
        Some(path)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_rig::{Orientation, Position, RigFrame};

    fn make_frame(frame: u64, x: f64, y: f64, z: f64) -> RigFrame {
        RigFrame {
            frame,
            position: Position::new(x, y, z),
            orientation: Orientation::new(0.0, 0.0, 0.0),
            focal_length_mm: Some(35.0),
            focus_distance_m: Some(2.0),
        }
    }

    fn two_keyframe_path() -> RigPath {
        let mut path = RigPath::new("test", 10.0);
        let kf0 = PathKeyframe {
            time: 0.0,
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0],
            focal_length_mm: Some(35.0),
            focus_distance_m: Some(2.0),
        };
        let kf1 = PathKeyframe {
            time: 1.0,
            position: [10.0, 0.0, 0.0],
            orientation: [90.0, 0.0, 0.0],
            focal_length_mm: Some(85.0),
            focus_distance_m: Some(4.0),
        };
        path.add_keyframe(kf0);
        path.add_keyframe(kf1);
        path
    }

    #[test]
    fn test_easing_linear_endpoints() {
        assert!((EasingMode::Linear.apply(0.0)).abs() < 1e-9);
        assert!((EasingMode::Linear.apply(1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_easing_smooth_step_midpoint() {
        let mid = EasingMode::SmoothStep.apply(0.5);
        assert!((mid - 0.5).abs() < 1e-9, "smoothstep(0.5) = {mid}");
    }

    #[test]
    fn test_easing_ease_in_out_monotone() {
        let vals: Vec<f64> = (0..=10)
            .map(|i| EasingMode::EaseInOut.apply(i as f64 / 10.0))
            .collect();
        for w in vals.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-12,
                "EaseInOut not monotone: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_path_evaluate_midpoint_position() {
        let path = two_keyframe_path();
        let result = path.evaluate(0.5).expect("should evaluate");
        // At t=0.5 with linear easing, position.x should be near 5
        assert!(
            (result.position[0] - 5.0).abs() < 0.5,
            "x={} (expected ~5)",
            result.position[0]
        );
    }

    #[test]
    fn test_path_evaluate_endpoints() {
        let path = two_keyframe_path();
        let start = path.evaluate(0.0).expect("start");
        let end = path.evaluate(1.0).expect("end");
        assert!(start.position[0] < 1.0);
        assert!(end.position[0] > 9.0);
    }

    #[test]
    fn test_path_loop_mode_wrap() {
        let mut path = two_keyframe_path();
        path.loop_mode = LoopMode::Wrap;
        // t=1.5 should wrap to 0.5
        let a = path.evaluate(0.5).expect("direct");
        let b = path.evaluate(1.5).expect("wrapped");
        assert!((a.position[0] - b.position[0]).abs() < 1e-6);
    }

    #[test]
    fn test_path_loop_mode_ping_pong() {
        let mut path = two_keyframe_path();
        path.loop_mode = LoopMode::PingPong;
        let a = path.evaluate(0.3).expect("forward");
        let b = path.evaluate(1.7).expect("reverse");
        // 1.7 in ping-pong => 2.0-1.7 = 0.3
        assert!((a.position[0] - b.position[0]).abs() < 0.5);
    }

    #[test]
    fn test_path_bake_count() {
        let path = two_keyframe_path();
        let baked = path.bake(11).expect("bake");
        assert_eq!(baked.len(), 11);
    }

    #[test]
    fn test_path_json_round_trip() {
        let path = two_keyframe_path();
        let json = path.to_json().expect("serialise");
        let restored = RigPath::from_json(&json).expect("deserialise");
        assert_eq!(restored.name, path.name);
        assert_eq!(restored.keyframe_count(), path.keyframe_count());
    }

    #[test]
    fn test_path_insufficient_keyframes() {
        let mut path = RigPath::new("empty", 5.0);
        path.add_keyframe(PathKeyframe {
            time: 0.0,
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0],
            focal_length_mm: None,
            focus_distance_m: None,
        });
        assert!(path.evaluate(0.5).is_none());
    }

    #[test]
    fn test_recorder_produces_path() {
        let mut rec = PathRecorder::new("rec", 1.0);
        rec.start();
        rec.push_frame(0.0, make_frame(0, 0.0, 0.0, 0.0));
        rec.push_frame(0.5, make_frame(30, 5.0, 0.0, 0.0));
        rec.push_frame(1.0, make_frame(60, 10.0, 0.0, 0.0));
        rec.stop();
        let path = rec.finish().expect("finish");
        assert_eq!(path.keyframe_count(), 3);
        let mid = path.evaluate(0.5).expect("eval");
        assert!((mid.position[0] - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_path_library_store_and_get() {
        let mut lib = PathLibrary::new();
        let path = two_keyframe_path();
        lib.store(path);
        assert!(lib.get("test").is_some());
        assert_eq!(lib.count(), 1);
        lib.remove("test");
        assert_eq!(lib.count(), 0);
    }
}
