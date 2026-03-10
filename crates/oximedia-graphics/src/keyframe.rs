//! Extended keyframe animation with advanced easing functions.
//!
//! Complements [`crate::animation`] with:
//! - A richer [`Easing`] enum (CSS cubic-bezier, spring physics, quartic / sine / back / bounce variants)
//! - A generic [`Keyframe<T>`] and [`AnimationTrack<T>`] that use absolute time in **seconds**
//! - A [`Lerp`] trait so custom value types can participate in interpolation
//!
//! # Example
//!
//! ```
//! use oximedia_graphics::keyframe::{AnimationTrack, Easing, Keyframe, Lerp};
//!
//! let mut track: AnimationTrack<f64> = AnimationTrack::new("opacity".to_string());
//! track.push(Keyframe { time: 0.0, value: 0.0_f64, easing: Easing::Linear });
//! track.push(Keyframe { time: 1.0, value: 1.0_f64, easing: Easing::EaseOutCubic });
//!
//! let v = track.evaluate(0.5);
//! assert!(v > 0.0 && v < 1.0);
//! ```

use serde::{Deserialize, Serialize};

/// Extended easing function supporting CSS-style cubic-bezier and spring physics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Easing {
    /// No easing — constant velocity.
    Linear,

    // ── Quadratic ─────────────────────────────────────────────────────────
    /// Accelerating from zero velocity.
    EaseInQuad,
    /// Decelerating to zero velocity.
    EaseOutQuad,
    /// Acceleration until halfway, then deceleration.
    EaseInOutQuad,

    // ── Cubic ─────────────────────────────────────────────────────────────
    /// Cubic ease in.
    EaseInCubic,
    /// Cubic ease out.
    EaseOutCubic,
    /// Cubic ease in-out.
    EaseInOutCubic,

    // ── Quartic ───────────────────────────────────────────────────────────
    /// Quartic ease in.
    EaseInQuart,
    /// Quartic ease out.
    EaseOutQuart,
    /// Quartic ease in-out.
    EaseInOutQuart,

    // ── Sine ──────────────────────────────────────────────────────────────
    /// Sine ease in.
    EaseInSine,
    /// Sine ease out.
    EaseOutSine,
    /// Sine ease in-out.
    EaseInOutSine,

    // ── Elastic ───────────────────────────────────────────────────────────
    /// Elastic ease in (overshoot at start).
    EaseInElastic,
    /// Elastic ease out (overshoot at end).
    EaseOutElastic,
    /// Elastic ease in-out.
    EaseInOutElastic,

    // ── Back ──────────────────────────────────────────────────────────────
    /// Back ease in — slight retraction before accelerating.
    EaseInBack,
    /// Back ease out — slight overshoot before settling.
    EaseOutBack,
    /// Back ease in-out.
    EaseInOutBack,

    // ── Bounce ────────────────────────────────────────────────────────────
    /// Bounce ease in.
    EaseInBounce,
    /// Bounce ease out (ball bouncing to a stop).
    EaseOutBounce,
    /// Bounce ease in-out.
    EaseInOutBounce,

    // ── Custom ────────────────────────────────────────────────────────────
    /// CSS-style cubic bezier defined by two control points P1 and P2
    /// where P0 = (0,0) and P3 = (1,1).
    /// `(x1, y1, x2, y2)` — x values must be in \[0,1\].
    CubicBezier(f64, f64, f64, f64),

    /// Damped spring simulation.
    /// - `mass`      — effective mass (kg, > 0)
    /// - `stiffness` — spring constant k (N/m, > 0)
    /// - `damping`   — damping coefficient c (N·s/m, ≥ 0)
    Spring {
        /// Spring mass.
        mass: f64,
        /// Spring stiffness.
        stiffness: f64,
        /// Damping coefficient.
        damping: f64,
    },
}

impl Easing {
    /// Apply the easing function to `t ∈ [0, 1]`.
    ///
    /// Returns a value approximately in `[0, 1]` (elastic / back variants may
    /// exceed this range momentarily).
    #[must_use]
    pub fn apply(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,

            // Quadratic
            Self::EaseInQuad => t * t,
            Self::EaseOutQuad => t * (2.0 - t),
            Self::EaseInOutQuad => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }

            // Cubic
            Self::EaseInCubic => t * t * t,
            Self::EaseOutCubic => {
                let u = t - 1.0;
                u * u * u + 1.0
            }
            Self::EaseInOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let u = 2.0 * t - 2.0;
                    0.5 * u * u * u + 1.0
                }
            }

            // Quartic
            Self::EaseInQuart => t * t * t * t,
            Self::EaseOutQuart => {
                let u = t - 1.0;
                1.0 - u * u * u * u
            }
            Self::EaseInOutQuart => {
                if t < 0.5 {
                    8.0 * t * t * t * t
                } else {
                    let u = t - 1.0;
                    1.0 - 8.0 * u * u * u * u
                }
            }

            // Sine
            Self::EaseInSine => 1.0 - (t * std::f64::consts::FRAC_PI_2).cos(),
            Self::EaseOutSine => (t * std::f64::consts::FRAC_PI_2).sin(),
            Self::EaseInOutSine => 0.5 * (1.0 - (std::f64::consts::PI * t).cos()),

            // Elastic
            Self::EaseInElastic => ease_in_elastic(t),
            Self::EaseOutElastic => ease_out_elastic(t),
            Self::EaseInOutElastic => ease_in_out_elastic(t),

            // Back
            Self::EaseInBack => ease_in_back(t),
            Self::EaseOutBack => ease_out_back(t),
            Self::EaseInOutBack => ease_in_out_back(t),

            // Bounce
            Self::EaseInBounce => 1.0 - ease_out_bounce(1.0 - t),
            Self::EaseOutBounce => ease_out_bounce(t),
            Self::EaseInOutBounce => {
                if t < 0.5 {
                    0.5 * (1.0 - ease_out_bounce(1.0 - 2.0 * t))
                } else {
                    0.5 * ease_out_bounce(2.0 * t - 1.0) + 0.5
                }
            }

            Self::CubicBezier(x1, y1, x2, y2) => cubic_bezier_solve(t, *x1, *y1, *x2, *y2),

            Self::Spring {
                mass,
                stiffness,
                damping,
            } => spring_solve(t, *mass, *stiffness, *damping),
        }
    }
}

// ── Elastic helpers ──────────────────────────────────────────────────────────

fn ease_in_elastic(t: f64) -> f64 {
    if t == 0.0 || t == 1.0 {
        return t;
    }
    let c4 = (2.0 * std::f64::consts::PI) / 3.0;
    -(2.0_f64.powf(10.0 * t - 10.0)) * ((10.0 * t - 10.75) * c4).sin()
}

fn ease_out_elastic(t: f64) -> f64 {
    if t == 0.0 || t == 1.0 {
        return t;
    }
    let c4 = (2.0 * std::f64::consts::PI) / 3.0;
    2.0_f64.powf(-10.0 * t) * ((10.0 * t - 0.75) * c4).sin() + 1.0
}

fn ease_in_out_elastic(t: f64) -> f64 {
    if t == 0.0 || t == 1.0 {
        return t;
    }
    let c5 = (2.0 * std::f64::consts::PI) / 4.5;
    if t < 0.5 {
        -(2.0_f64.powf(20.0 * t - 10.0) * ((20.0 * t - 11.125) * c5).sin()) / 2.0
    } else {
        (2.0_f64.powf(-20.0 * t + 10.0) * ((20.0 * t - 11.125) * c5).sin()) / 2.0 + 1.0
    }
}

// ── Back helpers ─────────────────────────────────────────────────────────────

const BACK_C1: f64 = 1.70158;
const BACK_C2: f64 = BACK_C1 * 1.525;
const BACK_C3: f64 = BACK_C1 + 1.0;

fn ease_in_back(t: f64) -> f64 {
    BACK_C3 * t * t * t - BACK_C1 * t * t
}

fn ease_out_back(t: f64) -> f64 {
    let u = t - 1.0;
    1.0 + BACK_C3 * u * u * u + BACK_C1 * u * u
}

fn ease_in_out_back(t: f64) -> f64 {
    if t < 0.5 {
        (2.0 * t).powi(2) * ((BACK_C2 + 1.0) * 2.0 * t - BACK_C2) / 2.0
    } else {
        let u = 2.0 * t - 2.0;
        (u.powi(2) * ((BACK_C2 + 1.0) * u + BACK_C2) + 2.0) / 2.0
    }
}

// ── Bounce helper ────────────────────────────────────────────────────────────

fn ease_out_bounce(t: f64) -> f64 {
    const N1: f64 = 7.5625;
    const D1: f64 = 2.75;

    if t < 1.0 / D1 {
        N1 * t * t
    } else if t < 2.0 / D1 {
        let u = t - 1.5 / D1;
        N1 * u * u + 0.75
    } else if t < 2.5 / D1 {
        let u = t - 2.25 / D1;
        N1 * u * u + 0.9375
    } else {
        let u = t - 2.625 / D1;
        N1 * u * u + 0.984_375
    }
}

// ── CSS cubic-bezier solver ──────────────────────────────────────────────────
//
// Standard algorithm: find the parametric `u` value such that B_x(u) ≈ t,
// then return B_y(u).  Uses Newton's method with binary-search fallback.

fn cubic_bezier_component(u: f64, p1: f64, p2: f64) -> f64 {
    // Bernstein form: 3*(1-u)²*u*p1 + 3*(1-u)*u²*p2 + u³
    let u2 = u * u;
    let u3 = u2 * u;
    let mu = 1.0 - u;
    3.0 * mu * mu * u * p1 + 3.0 * mu * u2 * p2 + u3
}

fn cubic_bezier_derivative(u: f64, p1: f64, p2: f64) -> f64 {
    let u2 = u * u;
    let mu = 1.0 - u;
    3.0 * mu * mu * p1 + 6.0 * mu * u * (p2 - p1) + 3.0 * u2 * (1.0 - p2)
}

fn cubic_bezier_solve(t: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    // Degenerate: linear bezier
    if (x1 - y1).abs() < 1e-12 && (x2 - y2).abs() < 1e-12 {
        return t;
    }

    // Newton iterations to find u such that B_x(u) = t
    let mut u = t;
    for _ in 0..8 {
        let bx = cubic_bezier_component(u, x1, x2) - t;
        let dx = cubic_bezier_derivative(u, x1, x2);
        if dx.abs() < 1e-12 {
            break;
        }
        u -= bx / dx;
        u = u.clamp(0.0, 1.0);
    }

    cubic_bezier_component(u, y1, y2).clamp(0.0, 1.0)
}

// ── Spring solver ────────────────────────────────────────────────────────────
//
// Models a critically-damped or under-damped spring with target = 1, start = 0.
// We sample the analytical solution at a virtual time proportional to t.

fn spring_solve(t: f64, mass: f64, stiffness: f64, damping: f64) -> f64 {
    let mass = mass.max(1e-6);
    let stiffness = stiffness.max(1e-6);
    let damping = damping.max(0.0);

    let omega0 = (stiffness / mass).sqrt(); // natural frequency
    let zeta = damping / (2.0 * (stiffness * mass).sqrt()); // damping ratio

    // Map t ∈ [0,1] to a virtual time in seconds.  We choose the scale so that
    // the spring is ~99 % settled at t = 1.
    let tau = 6.0 / (zeta * omega0).max(0.01); // approx settling time
    let time = t * tau;

    let pos = if zeta < 1.0 {
        // Under-damped
        let wd = omega0 * (1.0 - zeta * zeta).sqrt();
        let env = (-zeta * omega0 * time).exp();
        1.0 - env * (wd * time).cos()
            - env * (zeta / (1.0 - zeta * zeta).sqrt()) * (wd * time).sin()
    } else if (zeta - 1.0).abs() < 1e-6 {
        // Critically damped
        let env = (-omega0 * time).exp();
        1.0 - env * (1.0 + omega0 * time)
    } else {
        // Over-damped
        let r1 = -omega0 * (zeta + (zeta * zeta - 1.0).sqrt());
        let r2 = -omega0 * (zeta - (zeta * zeta - 1.0).sqrt());
        let c1 = r2 / (r2 - r1);
        let c2 = -r1 / (r2 - r1);
        1.0 - c1 * (r1 * time).exp() - c2 * (r2 * time).exp()
    };

    pos.clamp(0.0, 1.0)
}

// ── Lerp trait ───────────────────────────────────────────────────────────────

/// A value type that can be linearly interpolated.
pub trait Lerp: Clone {
    /// Linear interpolation: returns `a + (b - a) * t` for `t ∈ [0, 1]`.
    fn lerp(a: &Self, b: &Self, t: f64) -> Self;
}

impl Lerp for f64 {
    fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        a + (b - a) * t
    }
}

impl Lerp for f32 {
    fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        let t = t as f32;
        a + (b - a) * t
    }
}

impl Lerp for [f32; 2] {
    fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        let t = t as f32;
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    }
}

impl Lerp for [f32; 4] {
    fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        let t = t as f32;
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ]
    }
}

// ── Keyframe ─────────────────────────────────────────────────────────────────

/// A timed value in an animation track.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keyframe<T> {
    /// Absolute time in seconds.
    pub time: f64,
    /// Value at this moment.
    pub value: T,
    /// Easing applied **from** this keyframe to the next.
    pub easing: Easing,
}

// ── AnimationTrack ────────────────────────────────────────────────────────────

/// A named animation track holding keyframes for a single property.
///
/// Keyframes are stored in ascending time order.  Evaluating outside the
/// track's time range returns the nearest boundary value (no extrapolation).
#[derive(Debug, Clone)]
pub struct AnimationTrack<T: Lerp> {
    /// Human-readable name for this track (e.g. `"opacity"`, `"position.x"`).
    pub name: String,
    /// Sorted list of keyframes.
    pub keyframes: Vec<Keyframe<T>>,
}

impl<T: Lerp> AnimationTrack<T> {
    /// Create an empty track with the given name.
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            keyframes: Vec::new(),
        }
    }

    /// Insert a keyframe, maintaining ascending time order.
    pub fn push(&mut self, kf: Keyframe<T>) {
        let pos = self.keyframes.partition_point(|k| k.time <= kf.time);
        self.keyframes.insert(pos, kf);
    }

    /// Total duration of the track in seconds (time of last keyframe, or 0).
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.keyframes.last().map_or(0.0, |k| k.time)
    }

    /// Evaluate the track at `time` seconds, returning the interpolated value.
    ///
    /// - Before the first keyframe → first keyframe value.
    /// - After the last keyframe  → last keyframe value.
    /// - Between two keyframes    → interpolated with the first keyframe's easing.
    #[must_use]
    pub fn evaluate(&self, time: f64) -> T
    where
        T: Default,
    {
        if self.keyframes.is_empty() {
            return T::default();
        }
        if self.keyframes.len() == 1 || time <= self.keyframes[0].time {
            return self.keyframes[0].value.clone();
        }
        // SAFETY: is_empty() checked above and len >= 2, so last() is always Some
        let last_kf = self
            .keyframes
            .last()
            .expect("keyframes non-empty after len checks");
        if time >= last_kf.time {
            return last_kf.value.clone();
        }

        // Binary search for the preceding keyframe
        let next_idx = self.keyframes.partition_point(|k| k.time <= time);
        let prev_idx = next_idx - 1;

        let kf0 = &self.keyframes[prev_idx];
        let kf1 = &self.keyframes[next_idx];

        let span = kf1.time - kf0.time;
        let local_t = if span > 0.0 {
            ((time - kf0.time) / span).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let eased_t = kf0.easing.apply(local_t);
        Lerp::lerp(&kf0.value, &kf1.value, eased_t)
    }
}

impl AnimationTrack<f64> {
    /// Convenience: evaluate without requiring `T: Default`.
    #[must_use]
    pub fn evaluate_f64(&self, time: f64) -> f64 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if self.keyframes.len() == 1 || time <= self.keyframes[0].time {
            return self.keyframes[0].value;
        }
        // SAFETY: is_empty() checked above and len >= 2, so last() is always Some
        let last_kf = self
            .keyframes
            .last()
            .expect("keyframes non-empty after len checks");
        if time >= last_kf.time {
            return last_kf.value;
        }

        let next_idx = self.keyframes.partition_point(|k| k.time <= time);
        let prev_idx = next_idx - 1;

        let kf0 = &self.keyframes[prev_idx];
        let kf1 = &self.keyframes[next_idx];

        let span = kf1.time - kf0.time;
        let local_t = if span > 0.0 {
            ((time - kf0.time) / span).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let eased_t = kf0.easing.apply(local_t);
        f64::lerp(&kf0.value, &kf1.value, eased_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Easing tests ──────────────────────────────────────────────────────

    fn assert_boundary(easing: &Easing) {
        let lo = easing.apply(0.0);
        let hi = easing.apply(1.0);
        assert!(lo.abs() < 0.01, "{easing:?}.apply(0) = {lo}, expected ≈ 0");
        assert!(
            (hi - 1.0).abs() < 0.01,
            "{easing:?}.apply(1) = {hi}, expected ≈ 1"
        );
    }

    #[test]
    fn test_all_easing_boundaries() {
        let easings = [
            Easing::Linear,
            Easing::EaseInQuad,
            Easing::EaseOutQuad,
            Easing::EaseInOutQuad,
            Easing::EaseInCubic,
            Easing::EaseOutCubic,
            Easing::EaseInOutCubic,
            Easing::EaseInQuart,
            Easing::EaseOutQuart,
            Easing::EaseInOutQuart,
            Easing::EaseInSine,
            Easing::EaseOutSine,
            Easing::EaseInOutSine,
            Easing::EaseInElastic,
            Easing::EaseOutElastic,
            Easing::EaseInOutElastic,
            Easing::EaseInBack,
            Easing::EaseOutBack,
            Easing::EaseInOutBack,
            Easing::EaseInBounce,
            Easing::EaseOutBounce,
            Easing::EaseInOutBounce,
        ];
        for e in &easings {
            assert_boundary(e);
        }
    }

    #[test]
    fn test_cubic_bezier_linear() {
        // When x1=y1=0.33 and x2=y2=0.66, the curve should be approximately linear.
        let e = Easing::CubicBezier(0.33, 0.33, 0.66, 0.66);
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let v = e.apply(t);
            assert!(
                (v - t).abs() < 0.02,
                "CubicBezier(linear) at t={t}: expected ≈{t}, got {v}"
            );
        }
    }

    #[test]
    fn test_cubic_bezier_ease_in_out() {
        let e = Easing::CubicBezier(0.42, 0.0, 0.58, 1.0); // CSS ease-in-out
                                                           // Midpoint should be close to 0.5
        let mid = e.apply(0.5);
        assert!((mid - 0.5).abs() < 0.1, "ease-in-out mid: {mid}");
    }

    #[test]
    fn test_spring_boundary() {
        let spring = Easing::Spring {
            mass: 1.0,
            stiffness: 100.0,
            damping: 10.0,
        };
        assert_boundary(&spring);
    }

    #[test]
    fn test_spring_underdamped_rises() {
        let spring = Easing::Spring {
            mass: 1.0,
            stiffness: 200.0,
            damping: 5.0,
        };
        // Value at t=0.5 should be between 0 and 1
        let v = spring.apply(0.5);
        assert!(v >= 0.0 && v <= 1.0, "Spring at 0.5: {v}");
    }

    #[test]
    fn test_ease_in_is_slow_at_start() {
        // EaseIn should have a smaller value at t=0.3 than linear
        let val = Easing::EaseInCubic.apply(0.3);
        assert!(
            val < 0.3,
            "EaseInCubic at 0.3 should be below linear, got {val}"
        );
    }

    #[test]
    fn test_ease_out_is_fast_at_start() {
        let val = Easing::EaseOutCubic.apply(0.3);
        assert!(
            val > 0.3,
            "EaseOutCubic at 0.3 should be above linear, got {val}"
        );
    }

    // ── AnimationTrack tests ──────────────────────────────────────────────

    fn simple_track() -> AnimationTrack<f64> {
        let mut track = AnimationTrack::new("test".to_string());
        track.push(Keyframe {
            time: 0.0,
            value: 0.0,
            easing: Easing::Linear,
        });
        track.push(Keyframe {
            time: 1.0,
            value: 1.0,
            easing: Easing::Linear,
        });
        track
    }

    #[test]
    fn test_track_linear_midpoint() {
        let track = simple_track();
        let v = track.evaluate_f64(0.5);
        assert!((v - 0.5).abs() < 1e-9, "Linear midpoint: {v}");
    }

    #[test]
    fn test_track_clamps_before_start() {
        let track = simple_track();
        assert_eq!(track.evaluate_f64(-1.0), 0.0);
    }

    #[test]
    fn test_track_clamps_after_end() {
        let track = simple_track();
        assert_eq!(track.evaluate_f64(5.0), 1.0);
    }

    #[test]
    fn test_track_duration() {
        let track = simple_track();
        assert_eq!(track.duration(), 1.0);
    }

    #[test]
    fn test_track_empty_returns_default() {
        let track: AnimationTrack<f64> = AnimationTrack::new("empty".to_string());
        assert_eq!(track.evaluate_f64(0.5), 0.0);
    }

    #[test]
    fn test_track_single_keyframe() {
        let mut track = AnimationTrack::new("single".to_string());
        track.push(Keyframe {
            time: 0.5,
            value: 42.0_f64,
            easing: Easing::Linear,
        });
        assert_eq!(track.evaluate_f64(0.0), 42.0);
        assert_eq!(track.evaluate_f64(0.5), 42.0);
        assert_eq!(track.evaluate_f64(1.0), 42.0);
    }

    #[test]
    fn test_track_ease_out_cubic() {
        let mut track = AnimationTrack::new("eased".to_string());
        track.push(Keyframe {
            time: 0.0,
            value: 0.0_f64,
            easing: Easing::EaseOutCubic,
        });
        track.push(Keyframe {
            time: 2.0,
            value: 100.0,
            easing: Easing::Linear,
        });
        // At t=1.0 (halfway), EaseOutCubic should be ahead of linear (> 50)
        let v = track.evaluate_f64(1.0);
        assert!(
            v > 50.0,
            "EaseOutCubic at halfway should exceed 50, got {v}"
        );
    }

    #[test]
    fn test_track_push_sorted() {
        // Push keyframes out of order; track should sort them.
        let mut track = AnimationTrack::new("sorted".to_string());
        track.push(Keyframe {
            time: 1.0,
            value: 1.0_f64,
            easing: Easing::Linear,
        });
        track.push(Keyframe {
            time: 0.0,
            value: 0.0_f64,
            easing: Easing::Linear,
        });
        track.push(Keyframe {
            time: 0.5,
            value: 0.5_f64,
            easing: Easing::Linear,
        });
        assert_eq!(track.keyframes[0].time, 0.0);
        assert_eq!(track.keyframes[1].time, 0.5);
        assert_eq!(track.keyframes[2].time, 1.0);
    }

    #[test]
    fn test_lerp_f32_array4() {
        let a = [0.0_f32, 0.0, 0.0, 0.0];
        let b = [1.0_f32, 2.0, 3.0, 4.0];
        let mid = <[f32; 4] as Lerp>::lerp(&a, &b, 0.5);
        assert!((mid[0] - 0.5).abs() < 1e-5);
        assert!((mid[1] - 1.0).abs() < 1e-5);
    }
}
