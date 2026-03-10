#![allow(dead_code)]
//! Animation curve and keyframe system for broadcast graphics.
//!
//! Provides easing functions, individual keyframes with linear interpolation,
//! and a full animation curve that maps a time position to an interpolated value.

/// Easing type applied between two keyframes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EasingType {
    /// Constant rate of change throughout the interval.
    Linear,
    /// Slow start, fast end.
    EaseIn,
    /// Fast start, slow end.
    EaseOut,
    /// Slow start and end, fast middle.
    EaseInOut,
    /// Instantaneous jump at the end of the interval.
    Step,
    /// Overshoot and bounce at the end.
    Bounce,
}

impl EasingType {
    /// Map normalised time `t ∈ [0.0, 1.0]` through this easing function.
    ///
    /// Returns a value in approximately `[0.0, 1.0]` (bounce may slightly exceed 1.0).
    pub fn ease_value(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            EasingType::Linear => t,
            EasingType::EaseIn => t * t,
            EasingType::EaseOut => t * (2.0 - t),
            EasingType::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            EasingType::Step => {
                if t < 1.0 {
                    0.0
                } else {
                    1.0
                }
            }
            EasingType::Bounce => {
                let t2 = 1.0 - t;
                1.0 - Self::bounce_out(t2)
            }
        }
    }

    fn bounce_out(t: f64) -> f64 {
        if t < 1.0 / 2.75 {
            7.5625 * t * t
        } else if t < 2.0 / 2.75 {
            let t = t - 1.5 / 2.75;
            7.5625 * t * t + 0.75
        } else if t < 2.5 / 2.75 {
            let t = t - 2.25 / 2.75;
            7.5625 * t * t + 0.9375
        } else {
            let t = t - 2.625 / 2.75;
            7.5625 * t * t + 0.984_375
        }
    }

    /// Returns `true` when this easing produces a smooth continuous transition.
    pub fn is_smooth(&self) -> bool {
        !matches!(self, EasingType::Step)
    }
}

/// A single keyframe in an animation curve.
#[derive(Debug, Clone)]
pub struct AnimationKeyframe {
    /// Time position in milliseconds from the start of the curve.
    pub time_ms: f64,
    /// Numeric value at this keyframe.
    pub value: f64,
    /// Easing applied from this keyframe to the next.
    pub easing: EasingType,
}

impl AnimationKeyframe {
    /// Create a new keyframe.
    pub fn new(time_ms: f64, value: f64, easing: EasingType) -> Self {
        Self {
            time_ms,
            value,
            easing,
        }
    }

    /// Linearly interpolate from this keyframe toward `next` at normalised time `t ∈ [0,1]`.
    ///
    /// Uses the easing type stored on *this* keyframe (the outgoing keyframe).
    pub fn lerp_to(&self, next: &AnimationKeyframe, t: f64) -> f64 {
        let eased_t = self.easing.ease_value(t);
        self.value + (next.value - self.value) * eased_t
    }

    /// Returns `true` when this keyframe sits at the curve origin.
    pub fn is_at_origin(&self) -> bool {
        self.time_ms == 0.0
    }
}

/// A multi-keyframe animation curve that returns interpolated values for any time position.
#[derive(Debug, Clone, Default)]
pub struct AnimationCurve {
    keyframes: Vec<AnimationKeyframe>,
}

impl AnimationCurve {
    /// Create an empty curve.
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe. The internal list is kept sorted by `time_ms`.
    pub fn add_keyframe(&mut self, kf: AnimationKeyframe) {
        self.keyframes.push(kf);
        self.keyframes.sort_by(|a, b| {
            a.time_ms
                .partial_cmp(&b.time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Evaluate the curve at `time_ms`.
    ///
    /// - Before the first keyframe: returns the first keyframe's value.
    /// - After the last keyframe: returns the last keyframe's value.
    /// - Between two keyframes: interpolates using the outgoing keyframe's easing.
    pub fn value_at(&self, time_ms: f64) -> f64 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if self.keyframes.len() == 1 || time_ms <= self.keyframes[0].time_ms {
            return self.keyframes[0].value;
        }
        let last = self
            .keyframes
            .last()
            .expect("keyframes non-empty: length check passed above");
        if time_ms >= last.time_ms {
            return last.value;
        }
        // Find the surrounding pair
        for i in 0..self.keyframes.len() - 1 {
            let a = &self.keyframes[i];
            let b = &self.keyframes[i + 1];
            if time_ms >= a.time_ms && time_ms <= b.time_ms {
                let span = b.time_ms - a.time_ms;
                let t = if span > 0.0 {
                    (time_ms - a.time_ms) / span
                } else {
                    0.0
                };
                return a.lerp_to(b, t);
            }
        }
        last.value
    }

    /// Total duration of the curve in milliseconds (last keyframe time).
    pub fn duration_ms(&self) -> f64 {
        self.keyframes.last().map_or(0.0, |k| k.time_ms)
    }

    /// Number of keyframes in this curve.
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns `true` when the curve has no keyframes.
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Remove all keyframes.
    pub fn clear(&mut self) {
        self.keyframes.clear();
    }

    /// Return the minimum value across all keyframes.
    pub fn min_value(&self) -> Option<f64> {
        self.keyframes.iter().map(|k| k.value).reduce(f64::min)
    }

    /// Return the maximum value across all keyframes.
    pub fn max_value(&self) -> Option<f64> {
        self.keyframes.iter().map(|k| k.value).reduce(f64::max)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_easing_linear_midpoint() {
        assert!((EasingType::Linear.ease_value(0.5) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_easing_linear_endpoints() {
        assert_eq!(EasingType::Linear.ease_value(0.0), 0.0);
        assert_eq!(EasingType::Linear.ease_value(1.0), 1.0);
    }

    #[test]
    fn test_easing_ease_in_slower_than_linear() {
        // EaseIn at 0.5 should be < linear (0.5)
        assert!(EasingType::EaseIn.ease_value(0.5) < 0.5);
    }

    #[test]
    fn test_easing_ease_out_faster_than_linear() {
        // EaseOut at 0.5 should be > linear (0.5)
        assert!(EasingType::EaseOut.ease_value(0.5) > 0.5);
    }

    #[test]
    fn test_easing_ease_in_out_midpoint() {
        // EaseInOut at 0.5 should equal 0.5 (symmetric)
        assert!((EasingType::EaseInOut.ease_value(0.5) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_easing_step_before_end() {
        assert_eq!(EasingType::Step.ease_value(0.5), 0.0);
    }

    #[test]
    fn test_easing_step_at_end() {
        assert_eq!(EasingType::Step.ease_value(1.0), 1.0);
    }

    #[test]
    fn test_easing_smooth_flag() {
        assert!(EasingType::Linear.is_smooth());
        assert!(!EasingType::Step.is_smooth());
    }

    #[test]
    fn test_keyframe_lerp_linear_half() {
        let a = AnimationKeyframe::new(0.0, 0.0, EasingType::Linear);
        let b = AnimationKeyframe::new(1000.0, 100.0, EasingType::Linear);
        let v = a.lerp_to(&b, 0.5);
        assert!((v - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_keyframe_lerp_at_start() {
        let a = AnimationKeyframe::new(0.0, 10.0, EasingType::Linear);
        let b = AnimationKeyframe::new(500.0, 20.0, EasingType::Linear);
        assert!((a.lerp_to(&b, 0.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_keyframe_lerp_at_end() {
        let a = AnimationKeyframe::new(0.0, 10.0, EasingType::Linear);
        let b = AnimationKeyframe::new(500.0, 20.0, EasingType::Linear);
        assert!((a.lerp_to(&b, 1.0) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_curve_empty_returns_zero() {
        let c = AnimationCurve::new();
        assert_eq!(c.value_at(0.0), 0.0);
    }

    #[test]
    fn test_curve_single_keyframe() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 42.0, EasingType::Linear));
        assert_eq!(c.value_at(0.0), 42.0);
        assert_eq!(c.value_at(9999.0), 42.0);
    }

    #[test]
    fn test_curve_two_keyframes_midpoint() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 0.0, EasingType::Linear));
        c.add_keyframe(AnimationKeyframe::new(1000.0, 200.0, EasingType::Linear));
        let v = c.value_at(500.0);
        assert!((v - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_curve_duration() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 0.0, EasingType::Linear));
        c.add_keyframe(AnimationKeyframe::new(2500.0, 1.0, EasingType::Linear));
        assert!((c.duration_ms() - 2500.0).abs() < 1e-9);
    }

    #[test]
    fn test_curve_keyframe_count() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 0.0, EasingType::Linear));
        c.add_keyframe(AnimationKeyframe::new(500.0, 1.0, EasingType::EaseOut));
        assert_eq!(c.keyframe_count(), 2);
    }

    #[test]
    fn test_curve_min_max_values() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 5.0, EasingType::Linear));
        c.add_keyframe(AnimationKeyframe::new(500.0, 95.0, EasingType::Linear));
        assert_eq!(c.min_value(), Some(5.0));
        assert_eq!(c.max_value(), Some(95.0));
    }

    #[test]
    fn test_curve_clear() {
        let mut c = AnimationCurve::new();
        c.add_keyframe(AnimationKeyframe::new(0.0, 1.0, EasingType::Linear));
        c.clear();
        assert!(c.is_empty());
    }
}
