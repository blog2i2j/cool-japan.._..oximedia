//! Motion-path animation system for broadcast graphics.
//!
//! Provides keyframed position / rotation / scale animation along arbitrary paths with:
//! - **Bezier splines**: cubic and quadratic Bézier segments with configurable control points
//! - **Easing functions**: full set from linear to spring physics (via [`EasingKind`])
//! - **Independent tracks**: separate [`MotionTrack`] for X, Y, rotation, scale-X, scale-Y
//! - **`MotionPath`**: assembled multi-track animator driving a [`Transform2D`]-equivalent
//!   [`MotionTransform`] output
//! - **Path following**: attach a node to a spline and query its world-space pose at any `t`
//!
//! # Example
//!
//! ```
//! use oximedia_graphics::motion_path::{MotionPath, MotionKeyframe, EasingKind};
//!
//! let mut path = MotionPath::new(2.0);
//! path.add_position_keyframe(0.0, 0.0, 0.0, EasingKind::Linear);
//! path.add_position_keyframe(1.0, 100.0, 50.0, EasingKind::EaseInOutCubic);
//! path.add_position_keyframe(2.0, 200.0, 0.0, EasingKind::EaseOutQuad);
//!
//! let pose = path.evaluate(1.0);
//! assert!(pose.x > 0.0 && pose.x < 200.0);
//! ```

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// EasingKind
// ─────────────────────────────────────────────────────────────────────────────

/// Easing function applied between two consecutive keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EasingKind {
    /// Constant velocity.
    Linear,
    /// Quadratic ease in.
    EaseInQuad,
    /// Quadratic ease out.
    EaseOutQuad,
    /// Quadratic ease in-out.
    EaseInOutQuad,
    /// Cubic ease in.
    EaseInCubic,
    /// Cubic ease out.
    EaseOutCubic,
    /// Cubic ease in-out.
    EaseInOutCubic,
    /// Quartic ease in.
    EaseInQuart,
    /// Quartic ease out.
    EaseOutQuart,
    /// Quartic ease in-out.
    EaseInOutQuart,
    /// Sine ease in.
    EaseInSine,
    /// Sine ease out.
    EaseOutSine,
    /// Sine ease in-out.
    EaseInOutSine,
    /// Back ease in — slight retraction before accelerating.
    EaseInBack,
    /// Back ease out — slight overshoot.
    EaseOutBack,
    /// Elastic ease out.
    EaseOutElastic,
    /// Bounce ease out.
    EaseOutBounce,
    /// Hold at the starting value until the next keyframe.
    Hold,
}

impl EasingKind {
    /// Apply this easing to a normalised `t ∈ [0, 1]`.
    #[must_use]
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            EasingKind::Linear => t,
            EasingKind::EaseInQuad => t * t,
            EasingKind::EaseOutQuad => t * (2.0 - t),
            EasingKind::EaseInOutQuad => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            EasingKind::EaseInCubic => t * t * t,
            EasingKind::EaseOutCubic => {
                let t1 = t - 1.0;
                t1 * t1 * t1 + 1.0
            }
            EasingKind::EaseInOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let t1 = t - 1.0;
                    (2.0 * t1) * (2.0 * t1) * (2.0 * t1) / 2.0 + 1.0
                }
            }
            EasingKind::EaseInQuart => t * t * t * t,
            EasingKind::EaseOutQuart => {
                let t1 = t - 1.0;
                1.0 - t1 * t1 * t1 * t1
            }
            EasingKind::EaseInOutQuart => {
                if t < 0.5 {
                    8.0 * t * t * t * t
                } else {
                    let t1 = t - 1.0;
                    1.0 - 8.0 * t1 * t1 * t1 * t1
                }
            }
            EasingKind::EaseInSine => 1.0 - (t * PI / 2.0).cos(),
            EasingKind::EaseOutSine => (t * PI / 2.0).sin(),
            EasingKind::EaseInOutSine => -(((PI * t).cos() - 1.0) / 2.0),
            EasingKind::EaseInBack => {
                const C1: f32 = 1.70158;
                const C3: f32 = C1 + 1.0;
                C3 * t * t * t - C1 * t * t
            }
            EasingKind::EaseOutBack => {
                const C1: f32 = 1.70158;
                const C3: f32 = C1 + 1.0;
                let t1 = t - 1.0;
                1.0 + C3 * t1 * t1 * t1 + C1 * t1 * t1
            }
            EasingKind::EaseOutElastic => {
                if t == 0.0 {
                    return 0.0;
                }
                if t == 1.0 {
                    return 1.0;
                }
                const C4: f32 = 2.0 * PI / 3.0;
                (2.0_f32).powf(-10.0 * t) * ((t * 10.0 - 0.75) * C4).sin() + 1.0
            }
            EasingKind::EaseOutBounce => bounce_out(t),
            EasingKind::Hold => 0.0,
        }
    }
}

fn bounce_out(t: f32) -> f32 {
    const N1: f32 = 7.5625;
    const D1: f32 = 2.75;
    if t < 1.0 / D1 {
        N1 * t * t
    } else if t < 2.0 / D1 {
        let t2 = t - 1.5 / D1;
        N1 * t2 * t2 + 0.75
    } else if t < 2.5 / D1 {
        let t2 = t - 2.25 / D1;
        N1 * t2 * t2 + 0.9375
    } else {
        let t2 = t - 2.625 / D1;
        N1 * t2 * t2 + 0.984_375
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BezierSegment — cubic Bézier in a single dimension
// ─────────────────────────────────────────────────────────────────────────────

/// A cubic Bézier segment from `p0` to `p3` with control points `p1` and `p2`.
///
/// All values are in the same dimension (e.g. pixels or normalised units).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BezierSegment {
    /// Start value.
    pub p0: f32,
    /// First control point.
    pub p1: f32,
    /// Second control point.
    pub p2: f32,
    /// End value.
    pub p3: f32,
}

impl BezierSegment {
    /// Evaluate the Bézier at local parameter `u ∈ [0, 1]`.
    #[must_use]
    pub fn evaluate(&self, u: f32) -> f32 {
        let u = u.clamp(0.0, 1.0);
        let u1 = 1.0 - u;
        u1 * u1 * u1 * self.p0
            + 3.0 * u1 * u1 * u * self.p1
            + 3.0 * u1 * u * u * self.p2
            + u * u * u * self.p3
    }

    /// Tangent (first derivative) at `u`.
    #[must_use]
    pub fn tangent(&self, u: f32) -> f32 {
        let u = u.clamp(0.0, 1.0);
        let u1 = 1.0 - u;
        3.0 * u1 * u1 * (self.p1 - self.p0)
            + 6.0 * u1 * u * (self.p2 - self.p1)
            + 3.0 * u * u * (self.p3 - self.p2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MotionKeyframe
// ─────────────────────────────────────────────────────────────────────────────

/// A single keyframe in a [`MotionTrack`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionKeyframe {
    /// Time in seconds from the start of the motion path.
    pub time: f32,
    /// Value at this keyframe.
    pub value: f32,
    /// Easing applied from this keyframe to the **next** one.
    pub easing: EasingKind,
    /// Optional Bézier control point (tangent out) for smooth curve interpolation.
    /// If `None`, linear/eased interpolation is used.
    pub control_out: Option<f32>,
    /// Optional Bézier control point (tangent in from the *previous* keyframe side).
    pub control_in: Option<f32>,
}

impl MotionKeyframe {
    /// Create a simple keyframe without Bézier control points.
    #[must_use]
    pub fn new(time: f32, value: f32, easing: EasingKind) -> Self {
        Self {
            time,
            value,
            easing,
            control_out: None,
            control_in: None,
        }
    }

    /// Create a keyframe with explicit Bézier tangent handles.
    #[must_use]
    pub fn with_handles(
        time: f32,
        value: f32,
        easing: EasingKind,
        control_in: f32,
        control_out: f32,
    ) -> Self {
        Self {
            time,
            value,
            easing,
            control_out: Some(control_out),
            control_in: Some(control_in),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MotionTrack — single-channel keyframe track
// ─────────────────────────────────────────────────────────────────────────────

/// A single-channel keyframe track (e.g. X position, rotation, opacity).
///
/// Keyframes must be added in ascending time order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MotionTrack {
    keyframes: Vec<MotionKeyframe>,
}

impl MotionTrack {
    /// Create an empty track.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a keyframe.  Keyframes must be pushed in chronological order.
    pub fn push(&mut self, kf: MotionKeyframe) {
        self.keyframes.push(kf);
    }

    /// Number of keyframes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns `true` if the track has no keyframes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Evaluate the track at absolute time `t` (seconds).
    ///
    /// - Before the first keyframe: returns the first value.
    /// - After the last keyframe: returns the last value.
    /// - Between keyframes: interpolates using the keyframe's easing and optional Bézier handles.
    #[must_use]
    pub fn evaluate(&self, t: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        let first = &self.keyframes[0];
        let last = &self.keyframes[self.keyframes.len() - 1];
        if t <= first.time {
            return first.value;
        }
        if t >= last.time {
            return last.value;
        }
        // Find the segment containing t
        let idx = self
            .keyframes
            .windows(2)
            .position(|w| t >= w[0].time && t < w[1].time)
            .unwrap_or(self.keyframes.len().saturating_sub(2));

        let kf0 = &self.keyframes[idx];
        let kf1 = &self.keyframes[idx + 1];
        let span = kf1.time - kf0.time;
        if span <= 0.0 {
            return kf1.value;
        }
        let local_t = (t - kf0.time) / span;

        // Bézier interpolation when control points are available
        if let (Some(cout), Some(cin)) = (kf0.control_out, kf1.control_in) {
            let seg = BezierSegment {
                p0: kf0.value,
                p1: cout,
                p2: cin,
                p3: kf1.value,
            };
            return seg.evaluate(local_t);
        }

        // Eased linear interpolation
        let et = kf0.easing.apply(local_t);
        kf0.value + (kf1.value - kf0.value) * et
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MotionTransform — output pose
// ─────────────────────────────────────────────────────────────────────────────

/// The evaluated pose at a single point in time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionTransform {
    /// X position (pixels or normalised units).
    pub x: f32,
    /// Y position.
    pub y: f32,
    /// Rotation in **degrees** (clockwise positive in screen space).
    pub rotation_deg: f32,
    /// Horizontal scale factor (`1.0` = original size).
    pub scale_x: f32,
    /// Vertical scale factor.
    pub scale_y: f32,
    /// Opacity (`1.0` = fully opaque).
    pub opacity: f32,
}

impl MotionTransform {
    /// Identity transform at origin.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            rotation_deg: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            opacity: 1.0,
        }
    }

    /// Build a column-major 3×3 affine matrix `[a, b, c, d, tx, ty]` from
    /// this transform for use with renderers that accept a 2D affine matrix.
    ///
    /// Rotation is applied around the **origin** of the transformed element.
    #[must_use]
    pub fn to_affine6(&self) -> [f32; 6] {
        let angle_rad = self.rotation_deg.to_radians();
        let (sin_a, cos_a) = angle_rad.sin_cos();
        let a = cos_a * self.scale_x;
        let b = sin_a * self.scale_x;
        let c = -sin_a * self.scale_y;
        let d = cos_a * self.scale_y;
        [a, b, c, d, self.x, self.y]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MotionPath — assembled multi-track animator
// ─────────────────────────────────────────────────────────────────────────────

/// A full motion-path animator combining independent tracks for position,
/// rotation, scale, and opacity.
///
/// # Example
///
/// ```
/// use oximedia_graphics::motion_path::{MotionPath, EasingKind};
///
/// let mut path = MotionPath::new(2.0);
/// path.add_position_keyframe(0.0, 0.0, 720.0, EasingKind::EaseOutCubic);
/// path.add_position_keyframe(2.0, 1280.0, 360.0, EasingKind::Linear);
///
/// let pose = path.evaluate(1.0);
/// assert!(pose.x > 0.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionPath {
    /// Total duration of the motion path in seconds.
    pub duration: f32,
    /// X-position track.
    pub track_x: MotionTrack,
    /// Y-position track.
    pub track_y: MotionTrack,
    /// Rotation track (degrees).
    pub track_rotation: MotionTrack,
    /// Horizontal scale track.
    pub track_scale_x: MotionTrack,
    /// Vertical scale track.
    pub track_scale_y: MotionTrack,
    /// Opacity track.
    pub track_opacity: MotionTrack,
}

impl MotionPath {
    /// Create a new motion path with a given total duration (seconds).
    #[must_use]
    pub fn new(duration: f32) -> Self {
        Self {
            duration: duration.max(0.0),
            track_x: MotionTrack::new(),
            track_y: MotionTrack::new(),
            track_rotation: MotionTrack::new(),
            track_scale_x: MotionTrack::new(),
            track_scale_y: MotionTrack::new(),
            track_opacity: MotionTrack::new(),
        }
    }

    /// Add a position keyframe for both X and Y axes simultaneously.
    pub fn add_position_keyframe(&mut self, time: f32, x: f32, y: f32, easing: EasingKind) {
        self.track_x.push(MotionKeyframe::new(time, x, easing));
        self.track_y.push(MotionKeyframe::new(time, y, easing));
    }

    /// Add a rotation keyframe (degrees).
    pub fn add_rotation_keyframe(&mut self, time: f32, degrees: f32, easing: EasingKind) {
        self.track_rotation
            .push(MotionKeyframe::new(time, degrees, easing));
    }

    /// Add uniform scale keyframe (same value for X and Y).
    pub fn add_scale_keyframe(&mut self, time: f32, scale: f32, easing: EasingKind) {
        self.track_scale_x
            .push(MotionKeyframe::new(time, scale, easing));
        self.track_scale_y
            .push(MotionKeyframe::new(time, scale, easing));
    }

    /// Add a non-uniform scale keyframe.
    pub fn add_scale_xy_keyframe(
        &mut self,
        time: f32,
        sx: f32,
        sy: f32,
        easing: EasingKind,
    ) {
        self.track_scale_x
            .push(MotionKeyframe::new(time, sx, easing));
        self.track_scale_y
            .push(MotionKeyframe::new(time, sy, easing));
    }

    /// Add an opacity keyframe (`0.0` = transparent, `1.0` = opaque).
    pub fn add_opacity_keyframe(&mut self, time: f32, opacity: f32, easing: EasingKind) {
        self.track_opacity
            .push(MotionKeyframe::new(time, opacity.clamp(0.0, 1.0), easing));
    }

    /// Evaluate the motion path at `t` seconds, returning a [`MotionTransform`].
    ///
    /// Default values (identity) are used for tracks with no keyframes.
    #[must_use]
    pub fn evaluate(&self, t: f32) -> MotionTransform {
        let t = t.clamp(0.0, self.duration.max(0.0));
        let x = if self.track_x.is_empty() {
            0.0
        } else {
            self.track_x.evaluate(t)
        };
        let y = if self.track_y.is_empty() {
            0.0
        } else {
            self.track_y.evaluate(t)
        };
        let rotation_deg = if self.track_rotation.is_empty() {
            0.0
        } else {
            self.track_rotation.evaluate(t)
        };
        let scale_x = if self.track_scale_x.is_empty() {
            1.0
        } else {
            self.track_scale_x.evaluate(t)
        };
        let scale_y = if self.track_scale_y.is_empty() {
            1.0
        } else {
            self.track_scale_y.evaluate(t)
        };
        let opacity = if self.track_opacity.is_empty() {
            1.0
        } else {
            self.track_opacity.evaluate(t)
        };
        MotionTransform {
            x,
            y,
            rotation_deg,
            scale_x,
            scale_y,
            opacity,
        }
    }

    /// Sample the path evenly at `count` frames over its total duration,
    /// returning a `Vec<MotionTransform>` suitable for pre-baking.
    #[must_use]
    pub fn bake_frames(&self, count: usize) -> Vec<MotionTransform> {
        if count == 0 {
            return Vec::new();
        }
        (0..count)
            .map(|i| {
                let t = if count == 1 {
                    0.0
                } else {
                    self.duration * (i as f32 / (count - 1) as f32)
                };
                self.evaluate(t)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SplinePathFollower — attach a node to a 2-D Bézier spline
// ─────────────────────────────────────────────────────────────────────────────

/// A 2-D control point on a Bézier spline path.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SplinePoint {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Tangent control point (outgoing) X offset from `(x, y)`.
    pub tangent_out_x: f32,
    /// Tangent control point (outgoing) Y offset.
    pub tangent_out_y: f32,
    /// Tangent control point (incoming) X offset from the **next** anchor.
    pub tangent_in_x: f32,
    /// Tangent control point (incoming) Y offset.
    pub tangent_in_y: f32,
}

impl SplinePoint {
    /// Create a smooth spline point with symmetric tangents.
    #[must_use]
    pub fn smooth(x: f32, y: f32, tx: f32, ty: f32) -> Self {
        Self {
            x,
            y,
            tangent_out_x: tx,
            tangent_out_y: ty,
            tangent_in_x: -tx,
            tangent_in_y: -ty,
        }
    }

    /// Create a corner spline point with no tangents (sharp corner).
    #[must_use]
    pub fn corner(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            tangent_out_x: 0.0,
            tangent_out_y: 0.0,
            tangent_in_x: 0.0,
            tangent_in_y: 0.0,
        }
    }
}

/// The pose of a node following a spline at a given normalised parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct SplinePose {
    /// World-space X position.
    pub x: f32,
    /// World-space Y position.
    pub y: f32,
    /// Tangent angle in degrees (rotation that aligns the node with the path direction).
    pub tangent_angle_deg: f32,
}

/// Attaches an object to a 2-D cubic Bézier spline and evaluates its position
/// and tangent orientation at any normalised parameter `u ∈ [0, 1]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplinePathFollower {
    /// Control points of the spline (minimum 2).
    pub points: Vec<SplinePoint>,
    /// Whether the spline should close (last point connects back to first).
    pub closed: bool,
}

impl SplinePathFollower {
    /// Create a new follower from a list of control points.
    ///
    /// Returns an error if fewer than 2 points are provided.
    pub fn new(points: Vec<SplinePoint>, closed: bool) -> Result<Self, PathError> {
        if points.len() < 2 {
            return Err(PathError::TooFewPoints);
        }
        Ok(Self { points, closed })
    }

    /// Number of Bézier segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        if self.closed {
            self.points.len()
        } else {
            self.points.len().saturating_sub(1)
        }
    }

    /// Evaluate the spline at normalised parameter `u ∈ [0, 1]`.
    #[must_use]
    pub fn evaluate(&self, u: f32) -> SplinePose {
        let u = u.clamp(0.0, 1.0);
        let n_segs = self.segment_count();
        if n_segs == 0 {
            let p = &self.points[0];
            return SplinePose {
                x: p.x,
                y: p.y,
                tangent_angle_deg: 0.0,
            };
        }
        // Map u to a specific segment and local parameter
        let scaled = u * n_segs as f32;
        let seg_idx = (scaled.floor() as usize).min(n_segs - 1);
        let local_u = scaled - seg_idx as f32;

        let p0 = &self.points[seg_idx];
        let p3_idx = if self.closed {
            (seg_idx + 1) % self.points.len()
        } else {
            seg_idx + 1
        };
        let p3 = &self.points[p3_idx];

        // Control points
        let cp1x = p0.x + p0.tangent_out_x;
        let cp1y = p0.y + p0.tangent_out_y;
        let cp2x = p3.x + p3.tangent_in_x;
        let cp2y = p3.y + p3.tangent_in_y;

        let bx = BezierSegment {
            p0: p0.x,
            p1: cp1x,
            p2: cp2x,
            p3: p3.x,
        };
        let by = BezierSegment {
            p0: p0.y,
            p1: cp1y,
            p2: cp2y,
            p3: p3.y,
        };

        let x = bx.evaluate(local_u);
        let y = by.evaluate(local_u);
        let dx = bx.tangent(local_u);
        let dy = by.tangent(local_u);
        let tangent_angle_deg = dy.atan2(dx).to_degrees();

        SplinePose {
            x,
            y,
            tangent_angle_deg,
        }
    }

    /// Sample the spline at `count` evenly-spaced parameter values and return
    /// their poses.  Useful for arc-length pre-computation or rendering previews.
    #[must_use]
    pub fn sample_uniform(&self, count: usize) -> Vec<SplinePose> {
        if count == 0 {
            return Vec::new();
        }
        (0..count)
            .map(|i| {
                let u = if count == 1 {
                    0.0
                } else {
                    i as f32 / (count - 1) as f32
                };
                self.evaluate(u)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PathError
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for path construction.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PathError {
    /// Fewer than 2 control points were supplied.
    #[error("a spline path requires at least 2 control points")]
    TooFewPoints,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easing_linear_identity() {
        assert!((EasingKind::Linear.apply(0.0)).abs() < 1e-6);
        assert!((EasingKind::Linear.apply(1.0) - 1.0).abs() < 1e-6);
        assert!((EasingKind::Linear.apply(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn easing_ease_in_out_cubic_symmetric() {
        let a = EasingKind::EaseInOutCubic.apply(0.25);
        let b = 1.0 - EasingKind::EaseInOutCubic.apply(0.75);
        assert!((a - b).abs() < 1e-5, "EaseInOutCubic not symmetric: {a} vs {b}");
    }

    #[test]
    fn bezier_segment_endpoints() {
        let seg = BezierSegment {
            p0: 0.0,
            p1: 33.0,
            p2: 66.0,
            p3: 100.0,
        };
        assert!((seg.evaluate(0.0)).abs() < 1e-5);
        assert!((seg.evaluate(1.0) - 100.0).abs() < 1e-5);
    }

    #[test]
    fn motion_track_constant_between_keyframes() {
        let mut track = MotionTrack::new();
        track.push(MotionKeyframe::new(0.0, 10.0, EasingKind::Linear));
        track.push(MotionKeyframe::new(2.0, 10.0, EasingKind::Linear));
        assert!((track.evaluate(1.0) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn motion_track_linear_midpoint() {
        let mut track = MotionTrack::new();
        track.push(MotionKeyframe::new(0.0, 0.0, EasingKind::Linear));
        track.push(MotionKeyframe::new(2.0, 200.0, EasingKind::Linear));
        let v = track.evaluate(1.0);
        assert!((v - 100.0).abs() < 1e-4, "Expected 100, got {v}");
    }

    #[test]
    fn motion_track_clamps_before_first_and_after_last() {
        let mut track = MotionTrack::new();
        track.push(MotionKeyframe::new(1.0, 5.0, EasingKind::Linear));
        track.push(MotionKeyframe::new(3.0, 15.0, EasingKind::Linear));
        assert!((track.evaluate(0.0) - 5.0).abs() < 1e-5);
        assert!((track.evaluate(10.0) - 15.0).abs() < 1e-5);
    }

    #[test]
    fn motion_path_position_interpolation() {
        let mut path = MotionPath::new(4.0);
        path.add_position_keyframe(0.0, 0.0, 0.0, EasingKind::Linear);
        path.add_position_keyframe(4.0, 400.0, 200.0, EasingKind::Linear);
        let pose = path.evaluate(2.0);
        assert!((pose.x - 200.0).abs() < 1e-3, "Expected x=200, got {}", pose.x);
        assert!((pose.y - 100.0).abs() < 1e-3, "Expected y=100, got {}", pose.y);
    }

    #[test]
    fn motion_path_default_scale_and_opacity() {
        let path = MotionPath::new(1.0);
        let pose = path.evaluate(0.5);
        assert!((pose.scale_x - 1.0).abs() < 1e-6);
        assert!((pose.scale_y - 1.0).abs() < 1e-6);
        assert!((pose.opacity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn motion_path_bake_frames_count() {
        let mut path = MotionPath::new(1.0);
        path.add_position_keyframe(0.0, 0.0, 0.0, EasingKind::Linear);
        path.add_position_keyframe(1.0, 100.0, 100.0, EasingKind::Linear);
        let frames = path.bake_frames(60);
        assert_eq!(frames.len(), 60);
    }

    #[test]
    fn spline_follower_two_point_corners() {
        let pts = vec![SplinePoint::corner(0.0, 0.0), SplinePoint::corner(100.0, 0.0)];
        let follower = SplinePathFollower::new(pts, false).expect("valid spline");
        let mid = follower.evaluate(0.5);
        assert!((mid.x - 50.0).abs() < 1e-4, "Expected x=50, got {}", mid.x);
        assert!((mid.y).abs() < 1e-4);
    }

    #[test]
    fn spline_follower_too_few_points_returns_error() {
        let result = SplinePathFollower::new(vec![SplinePoint::corner(0.0, 0.0)], false);
        assert!(matches!(result, Err(PathError::TooFewPoints)));
    }

    #[test]
    fn spline_sample_uniform_count() {
        let pts = vec![
            SplinePoint::smooth(0.0, 0.0, 50.0, 0.0),
            SplinePoint::smooth(200.0, 0.0, 50.0, 0.0),
        ];
        let follower = SplinePathFollower::new(pts, false).expect("valid spline");
        let samples = follower.sample_uniform(11);
        assert_eq!(samples.len(), 11);
    }

    #[test]
    fn transform_affine6_identity() {
        let tf = MotionTransform::identity();
        let m = tf.to_affine6();
        // identity: a=1, b=0, c=0, d=1, tx=0, ty=0
        assert!((m[0] - 1.0).abs() < 1e-6);
        assert!((m[1]).abs() < 1e-6);
        assert!((m[2]).abs() < 1e-6);
        assert!((m[3] - 1.0).abs() < 1e-6);
        assert!((m[4]).abs() < 1e-6);
        assert!((m[5]).abs() < 1e-6);
    }

    #[test]
    fn hold_easing_returns_zero_before_end() {
        // Hold easing should snap: t<1 returns 0, t=1 returns 0 (clamped)
        assert!((EasingKind::Hold.apply(0.5)).abs() < 1e-6);
        assert!((EasingKind::Hold.apply(0.0)).abs() < 1e-6);
    }
}
