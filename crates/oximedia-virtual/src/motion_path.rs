//! Camera motion path management for virtual production.
//!
//! Provides keyframe-based camera motion paths with interpolation, useful for
//! pre-visualisation and motion-control camera playback in LED volume shoots.

#![allow(dead_code)]

/// A single keyframe in a camera motion path.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Frame number (zero-based) at which this keyframe occurs.
    pub frame: u64,
    /// Camera position `(x, y, z)` in metres.
    pub position: (f32, f32, f32),
    /// Camera rotation `(pan_deg, tilt_deg, roll_deg)`.
    pub rotation: (f32, f32, f32),
    /// Lens focal length in millimetres.
    pub focal_mm: f32,
}

impl Keyframe {
    /// Create a new keyframe.
    #[must_use]
    pub fn new(
        frame: u64,
        position: (f32, f32, f32),
        rotation: (f32, f32, f32),
        focal_mm: f32,
    ) -> Self {
        Self {
            frame,
            position,
            rotation,
            focal_mm,
        }
    }
}

/// Linear interpolation between two scalar values.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Linearly interpolate between two keyframes at fractional parameter `t` ∈ [0, 1].
#[must_use]
pub fn interpolate_keyframes(a: &Keyframe, b: &Keyframe, t: f32) -> Keyframe {
    let t = t.clamp(0.0, 1.0);
    Keyframe {
        frame: a.frame + ((b.frame as f64 - a.frame as f64) * f64::from(t)) as u64,
        position: (
            lerp(a.position.0, b.position.0, t),
            lerp(a.position.1, b.position.1, t),
            lerp(a.position.2, b.position.2, t),
        ),
        rotation: (
            lerp(a.rotation.0, b.rotation.0, t),
            lerp(a.rotation.1, b.rotation.1, t),
            lerp(a.rotation.2, b.rotation.2, t),
        ),
        focal_mm: lerp(a.focal_mm, b.focal_mm, t),
    }
}

/// Type of easing applied to motion path interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EasingMode {
    /// Constant velocity between keyframes.
    #[default]
    Linear,
    /// Slow in, slow out.
    EaseInOut,
    /// Slow into each keyframe.
    EaseIn,
    /// Slow out of each keyframe.
    EaseOut,
}

impl EasingMode {
    /// Apply the easing curve to a linear parameter `t` ∈ [0, 1].
    #[must_use]
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            EasingMode::Linear => t,
            EasingMode::EaseInOut => t * t * (3.0 - 2.0 * t), // smoothstep
            EasingMode::EaseIn => t * t,
            EasingMode::EaseOut => t * (2.0 - t),
        }
    }
}

/// A camera motion path consisting of ordered keyframes.
#[derive(Debug, Clone, Default)]
pub struct MotionPath {
    keyframes: Vec<Keyframe>,
    /// Easing mode applied between keyframes.
    pub easing: EasingMode,
}

impl MotionPath {
    /// Create an empty motion path.
    #[must_use]
    pub fn new(easing: EasingMode) -> Self {
        Self {
            keyframes: Vec::new(),
            easing,
        }
    }

    /// Add a keyframe. Keyframes are kept sorted by frame number.
    pub fn add_keyframe(&mut self, kf: Keyframe) {
        let pos = self
            .keyframes
            .partition_point(|existing| existing.frame <= kf.frame);
        self.keyframes.insert(pos, kf);
    }

    /// Number of keyframes in the path.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns `true` if the path has no keyframes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Total duration of the path in frames (last frame − first frame).
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        if self.keyframes.len() < 2 {
            return 0;
        }
        let n = self.keyframes.len();
        self.keyframes[n - 1].frame - self.keyframes[0].frame
    }

    /// Evaluate the motion path at a given frame by interpolating between neighbours.
    ///
    /// Returns `None` if there are fewer than two keyframes or `frame` is outside range.
    #[must_use]
    pub fn evaluate(&self, frame: u64) -> Option<Keyframe> {
        if self.keyframes.len() < 2 {
            return None;
        }
        // Clamp to path range
        let first = self.keyframes[0].frame;
        let last = self.keyframes[self.keyframes.len() - 1].frame;
        if frame < first || frame > last {
            return None;
        }
        // Find bracketing keyframes
        let idx = self.keyframes.partition_point(|kf| kf.frame <= frame);
        let (a, b) = if idx == 0 {
            (&self.keyframes[0], &self.keyframes[0])
        } else if idx >= self.keyframes.len() {
            let last_idx = self.keyframes.len() - 1;
            (&self.keyframes[last_idx], &self.keyframes[last_idx])
        } else {
            (&self.keyframes[idx - 1], &self.keyframes[idx])
        };

        let span = b.frame.saturating_sub(a.frame) as f32;
        let t_linear = if span == 0.0 {
            0.0
        } else {
            (frame.saturating_sub(a.frame)) as f32 / span
        };
        let t = self.easing.apply(t_linear);
        Some(interpolate_keyframes(a, b, t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kf(frame: u64, x: f32) -> Keyframe {
        Keyframe::new(frame, (x, 0.0, 0.0), (0.0, 0.0, 0.0), 35.0)
    }

    #[test]
    fn test_lerp_midpoint() {
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_lerp_endpoints() {
        assert_eq!(lerp(2.0, 8.0, 0.0), 2.0);
        assert_eq!(lerp(2.0, 8.0, 1.0), 8.0);
    }

    #[test]
    fn test_keyframe_interpolate_midpoint() {
        let a = kf(0, 0.0);
        let b = kf(100, 10.0);
        let mid = interpolate_keyframes(&a, &b, 0.5);
        assert!((mid.position.0 - 5.0).abs() < 1e-5);
        assert_eq!(mid.frame, 50);
    }

    #[test]
    fn test_keyframe_interpolate_focal() {
        let a = Keyframe::new(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 24.0);
        let b = Keyframe::new(50, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 48.0);
        let mid = interpolate_keyframes(&a, &b, 0.5);
        assert!((mid.focal_mm - 36.0).abs() < 1e-5);
    }

    #[test]
    fn test_easing_linear() {
        assert!((EasingMode::Linear.apply(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_in_out_midpoint() {
        // smoothstep at 0.5 = 0.5
        let v = EasingMode::EaseInOut.apply(0.5);
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_in() {
        // ease-in: t² → 0.25 at t=0.5
        let v = EasingMode::EaseIn.apply(0.5);
        assert!((v - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_easing_ease_out() {
        // ease-out: t(2-t) → 0.75 at t=0.5
        let v = EasingMode::EaseOut.apply(0.5);
        assert!((v - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_easing_clamp() {
        assert_eq!(EasingMode::Linear.apply(-1.0), 0.0);
        assert_eq!(EasingMode::Linear.apply(2.0), 1.0);
    }

    #[test]
    fn test_motion_path_add_keeps_order() {
        let mut path = MotionPath::new(EasingMode::Linear);
        path.add_keyframe(kf(100, 10.0));
        path.add_keyframe(kf(0, 0.0));
        path.add_keyframe(kf(50, 5.0));
        assert_eq!(path.keyframes[0].frame, 0);
        assert_eq!(path.keyframes[1].frame, 50);
        assert_eq!(path.keyframes[2].frame, 100);
    }

    #[test]
    fn test_motion_path_duration() {
        let mut path = MotionPath::new(EasingMode::Linear);
        path.add_keyframe(kf(10, 0.0));
        path.add_keyframe(kf(110, 1.0));
        assert_eq!(path.duration_frames(), 100);
    }

    #[test]
    fn test_motion_path_evaluate_midpoint() {
        let mut path = MotionPath::new(EasingMode::Linear);
        path.add_keyframe(kf(0, 0.0));
        path.add_keyframe(kf(100, 100.0));
        let result = path.evaluate(50).expect("should succeed in test");
        assert!((result.position.0 - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_motion_path_evaluate_out_of_range() {
        let mut path = MotionPath::new(EasingMode::Linear);
        path.add_keyframe(kf(10, 0.0));
        path.add_keyframe(kf(20, 1.0));
        assert!(path.evaluate(5).is_none());
        assert!(path.evaluate(30).is_none());
    }

    #[test]
    fn test_motion_path_empty_evaluate() {
        let path = MotionPath::new(EasingMode::Linear);
        assert!(path.evaluate(0).is_none());
    }

    #[test]
    fn test_motion_path_is_empty() {
        let path = MotionPath::new(EasingMode::Linear);
        assert!(path.is_empty());
    }
}
