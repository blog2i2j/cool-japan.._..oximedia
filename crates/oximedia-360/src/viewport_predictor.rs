//! Viewport prediction for adaptive 360° video streaming.
//!
//! In tile-based or quality-adaptive 360° streaming the client must request
//! higher-resolution segments for the region the user is currently looking at
//! *before* the segments are needed for playback.  This module implements a
//! lightweight predictor that estimates the viewport orientation the user will
//! be at after a configurable look-ahead window, based on a sliding history of
//! recent head-pose samples.
//!
//! ## Models
//!
//! | Model | Description |
//! |-------|-------------|
//! | [`ConstantPositionPredictor`] | Assumes the user stays still — zero-latency baseline. |
//! | [`LinearVelocityPredictor`]   | Extrapolates last-observed angular velocity linearly. |
//! | [`WeightedHistoryPredictor`]  | Exponential-weighted average of recent velocities. |
//!
//! ## Coordinate convention
//!
//! Orientations are represented as `(yaw_rad, pitch_rad)` pairs.  **Yaw** is
//! the azimuthal angle (rotation around the vertical Y axis), ranging over
//! `[−π, +π]`.  **Pitch** is the elevation angle (rotation around the
//! camera-local X axis), clamped to `[−π/2, +π/2]`.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::viewport_predictor::{
//!     HeadPoseSample, LinearVelocityPredictor, ViewportPredictor,
//! };
//!
//! let mut predictor = LinearVelocityPredictor::new(5);
//! predictor.push(HeadPoseSample { timestamp_s: 0.0, yaw_rad: 0.0, pitch_rad: 0.0 });
//! predictor.push(HeadPoseSample { timestamp_s: 0.1, yaw_rad: 0.05, pitch_rad: 0.0 });
//! let pred = predictor.predict(0.25);
//! println!("predicted yaw: {:.3}", pred.yaw_rad);
//! ```

use crate::VrError;
use std::collections::VecDeque;

// ─── HeadPoseSample ──────────────────────────────────────────────────────────

/// A single head-pose observation from an IMU / head-tracker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadPoseSample {
    /// Absolute time of this observation, in seconds (monotonically increasing).
    pub timestamp_s: f64,
    /// Yaw angle in radians (`[−π, +π]`).
    pub yaw_rad: f32,
    /// Pitch angle in radians (`[−π/2, +π/2]`).
    pub pitch_rad: f32,
}

// ─── PredictedOrientation ────────────────────────────────────────────────────

/// A predicted head orientation at a future point in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictedOrientation {
    /// Predicted yaw angle in radians, wrapped to `[−π, +π]`.
    pub yaw_rad: f32,
    /// Predicted pitch angle in radians, clamped to `[−π/2, +π/2]`.
    pub pitch_rad: f32,
    /// Confidence in `[0.0, 1.0]`.  Higher means the model has seen more
    /// samples and the prediction is likely more reliable.
    pub confidence: f32,
}

// ─── ViewportPredictor trait ─────────────────────────────────────────────────

/// Common interface for viewport prediction models.
pub trait ViewportPredictor {
    /// Record a new head-pose observation.
    fn push(&mut self, sample: HeadPoseSample);

    /// Predict the orientation at the given absolute timestamp (seconds).
    ///
    /// If no samples have been recorded yet, returns the zero orientation
    /// with zero confidence.
    fn predict(&self, target_time_s: f64) -> PredictedOrientation;

    /// Clear all recorded samples.
    fn reset(&mut self);

    /// Return the number of samples currently held.
    fn len(&self) -> usize;

    /// Return `true` if no samples have been recorded.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ─── ConstantPositionPredictor ────────────────────────────────────────────────

/// Predicts that the user will remain at the last observed orientation.
///
/// This is the simplest possible model; it is useful as a baseline and in
/// situations where head motion is very slow.
#[derive(Debug, Clone)]
pub struct ConstantPositionPredictor {
    last: Option<HeadPoseSample>,
}

impl ConstantPositionPredictor {
    /// Create a new constant-position predictor with no history.
    #[must_use]
    pub fn new() -> Self {
        Self { last: None }
    }
}

impl Default for ConstantPositionPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewportPredictor for ConstantPositionPredictor {
    fn push(&mut self, sample: HeadPoseSample) {
        self.last = Some(sample);
    }

    fn predict(&self, _target_time_s: f64) -> PredictedOrientation {
        match self.last {
            Some(s) => PredictedOrientation {
                yaw_rad: s.yaw_rad,
                pitch_rad: s.pitch_rad,
                confidence: 1.0,
            },
            None => PredictedOrientation {
                yaw_rad: 0.0,
                pitch_rad: 0.0,
                confidence: 0.0,
            },
        }
    }

    fn reset(&mut self) {
        self.last = None;
    }

    fn len(&self) -> usize {
        if self.last.is_some() {
            1
        } else {
            0
        }
    }
}

// ─── LinearVelocityPredictor ──────────────────────────────────────────────────

/// Extrapolates a linear angular velocity from the last two samples.
///
/// The angular velocity is estimated as `Δangle / Δtime` between the two most
/// recent observations and is applied for the entire prediction horizon.
///
/// `capacity` limits the ring-buffer size; only the `capacity` most recent
/// samples are retained.
#[derive(Debug, Clone)]
pub struct LinearVelocityPredictor {
    history: VecDeque<HeadPoseSample>,
    capacity: usize,
}

impl LinearVelocityPredictor {
    /// Create a predictor that retains up to `capacity` samples.
    ///
    /// `capacity` is clamped to at least 2.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(2);
        Self {
            history: VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Compute angular velocity `(Δyaw/Δt, Δpitch/Δt)` from the last two
    /// samples.  Returns `(0, 0)` if fewer than 2 samples are available or if
    /// the time delta is negligibly small.
    fn last_velocity(&self) -> (f32, f32) {
        if self.history.len() < 2 {
            return (0.0, 0.0);
        }
        let len = self.history.len();
        let prev = &self.history[len - 2];
        let curr = &self.history[len - 1];
        let dt = (curr.timestamp_s - prev.timestamp_s) as f32;
        if dt.abs() < f32::EPSILON {
            return (0.0, 0.0);
        }
        let dyaw = wrap_angle(curr.yaw_rad - prev.yaw_rad);
        let dpitch = curr.pitch_rad - prev.pitch_rad;
        (dyaw / dt, dpitch / dt)
    }
}

impl ViewportPredictor for LinearVelocityPredictor {
    fn push(&mut self, sample: HeadPoseSample) {
        if self.history.len() >= self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(sample);
    }

    fn predict(&self, target_time_s: f64) -> PredictedOrientation {
        let last = match self.history.back() {
            Some(s) => *s,
            None => {
                return PredictedOrientation {
                    yaw_rad: 0.0,
                    pitch_rad: 0.0,
                    confidence: 0.0,
                }
            }
        };

        let horizon = (target_time_s - last.timestamp_s) as f32;
        let (vy, vp) = self.last_velocity();

        let yaw = wrap_angle(last.yaw_rad + vy * horizon);
        let pitch = (last.pitch_rad + vp * horizon)
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);

        // Confidence grows with history depth, saturating at the capacity.
        let confidence = (self.history.len() as f32 / self.capacity as f32).min(1.0);

        PredictedOrientation {
            yaw_rad: yaw,
            pitch_rad: pitch,
            confidence,
        }
    }

    fn reset(&mut self) {
        self.history.clear();
    }

    fn len(&self) -> usize {
        self.history.len()
    }
}

// ─── WeightedHistoryPredictor ─────────────────────────────────────────────────

/// Estimates future orientation using an exponentially-weighted average of
/// recent per-sample angular velocities.
///
/// Each inter-sample velocity is weighted by `alpha^k` where `k` is the age
/// of the interval (0 = most recent).  Higher `alpha` gives more weight to
/// older samples (smoother but slower to respond to direction changes).
/// Lower `alpha` makes the predictor more reactive.
#[derive(Debug, Clone)]
pub struct WeightedHistoryPredictor {
    history: VecDeque<HeadPoseSample>,
    capacity: usize,
    /// Exponential decay factor in `(0, 1)`.
    alpha: f32,
}

impl WeightedHistoryPredictor {
    /// Create a new predictor.
    ///
    /// # Parameters
    ///
    /// * `capacity` — maximum number of samples to keep (≥ 2).
    /// * `alpha`    — decay factor in `(0, 1)`.  Values close to 1 produce
    ///   slow-changing, smoothed predictions; values close to 0 produce
    ///   reactive but noisy ones.  `0.7` is a sensible default.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidCoordinate`] if `alpha` is not in `(0, 1)`.
    pub fn new(capacity: usize, alpha: f32) -> Result<Self, VrError> {
        if !(0.0..1.0).contains(&alpha) || alpha <= 0.0 {
            return Err(VrError::InvalidCoordinate);
        }
        let cap = capacity.max(2);
        Ok(Self {
            history: VecDeque::with_capacity(cap),
            capacity: cap,
            alpha,
        })
    }

    /// Compute the exponentially-weighted average angular velocity over the
    /// stored inter-sample intervals.
    fn weighted_velocity(&self) -> (f32, f32) {
        let n = self.history.len();
        if n < 2 {
            return (0.0, 0.0);
        }

        let mut sum_vy = 0.0f32;
        let mut sum_vp = 0.0f32;
        let mut weight_total = 0.0f32;

        // Intervals are indexed from the most recent: age=0 is [n-2, n-1].
        for age in 0..(n - 1) {
            let idx_new = n - 1 - age;
            let idx_old = idx_new - 1;
            let curr = &self.history[idx_new];
            let prev = &self.history[idx_old];
            let dt = (curr.timestamp_s - prev.timestamp_s) as f32;
            if dt.abs() < f32::EPSILON {
                continue;
            }
            let vy = wrap_angle(curr.yaw_rad - prev.yaw_rad) / dt;
            let vp = (curr.pitch_rad - prev.pitch_rad) / dt;

            let w = self.alpha.powi(age as i32);
            sum_vy += w * vy;
            sum_vp += w * vp;
            weight_total += w;
        }

        if weight_total < f32::EPSILON {
            return (0.0, 0.0);
        }
        (sum_vy / weight_total, sum_vp / weight_total)
    }
}

impl ViewportPredictor for WeightedHistoryPredictor {
    fn push(&mut self, sample: HeadPoseSample) {
        if self.history.len() >= self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(sample);
    }

    fn predict(&self, target_time_s: f64) -> PredictedOrientation {
        let last = match self.history.back() {
            Some(s) => *s,
            None => {
                return PredictedOrientation {
                    yaw_rad: 0.0,
                    pitch_rad: 0.0,
                    confidence: 0.0,
                }
            }
        };

        let horizon = (target_time_s - last.timestamp_s) as f32;
        let (vy, vp) = self.weighted_velocity();

        let yaw = wrap_angle(last.yaw_rad + vy * horizon);
        let pitch = (last.pitch_rad + vp * horizon)
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);

        let confidence = (self.history.len() as f32 / self.capacity as f32).min(1.0);

        PredictedOrientation {
            yaw_rad: yaw,
            pitch_rad: pitch,
            confidence,
        }
    }

    fn reset(&mut self) {
        self.history.clear();
    }

    fn len(&self) -> usize {
        self.history.len()
    }
}

// ─── ViewportRegion ───────────────────────────────────────────────────────────

/// An axis-aligned bounding region on the sphere surface defined by angular
/// half-extents around a centre orientation.
///
/// This is used to express the spatial extent of a viewport (e.g. the field of
/// view cone) as a rectangular patch in spherical angular space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportRegion {
    /// Centre yaw in radians.
    pub yaw_rad: f32,
    /// Centre pitch in radians.
    pub pitch_rad: f32,
    /// Half-width of the region in the yaw direction (radians).
    pub half_yaw_rad: f32,
    /// Half-height of the region in the pitch direction (radians).
    pub half_pitch_rad: f32,
}

impl ViewportRegion {
    /// Construct a viewport region from an orientation and FOV angles.
    ///
    /// # Parameters
    ///
    /// * `yaw_rad`, `pitch_rad` — centre of the viewport on the sphere.
    /// * `fov_h_rad`, `fov_v_rad` — horizontal and vertical field of view angles.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if either FOV is ≤ 0.
    pub fn from_fov(
        yaw_rad: f32,
        pitch_rad: f32,
        fov_h_rad: f32,
        fov_v_rad: f32,
    ) -> Result<Self, VrError> {
        if fov_h_rad <= 0.0 || fov_v_rad <= 0.0 {
            return Err(VrError::InvalidDimensions(
                "FOV angles must be positive".into(),
            ));
        }
        Ok(Self {
            yaw_rad: wrap_angle(yaw_rad),
            pitch_rad: pitch_rad.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2),
            half_yaw_rad: fov_h_rad * 0.5,
            half_pitch_rad: fov_v_rad * 0.5,
        })
    }

    /// Test whether a spherical point `(yaw, pitch)` falls inside this region.
    #[must_use]
    pub fn contains(&self, yaw_rad: f32, pitch_rad: f32) -> bool {
        let dyaw = wrap_angle(yaw_rad - self.yaw_rad).abs();
        let dpitch = (pitch_rad - self.pitch_rad).abs();
        dyaw <= self.half_yaw_rad && dpitch <= self.half_pitch_rad
    }

    /// Compute the overlap fraction with another region (Intersection over Union
    /// in angular space).  Returns a value in `[0.0, 1.0]`.
    #[must_use]
    pub fn iou(&self, other: &ViewportRegion) -> f32 {
        let dyaw = wrap_angle(self.yaw_rad - other.yaw_rad).abs();
        let inter_yaw = (self.half_yaw_rad + other.half_yaw_rad - dyaw).max(0.0);

        let dpitch = (self.pitch_rad - other.pitch_rad).abs();
        let inter_pitch = (self.half_pitch_rad + other.half_pitch_rad - dpitch).max(0.0);

        let intersection = inter_yaw * inter_pitch;
        let area_a = 4.0 * self.half_yaw_rad * self.half_pitch_rad;
        let area_b = 4.0 * other.half_yaw_rad * other.half_pitch_rad;
        let union = area_a + area_b - intersection;

        if union < f32::EPSILON {
            return 1.0; // Both regions are degenerate points — treat as equal.
        }
        (intersection / union).clamp(0.0, 1.0)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Wrap an angle (radians) to `[−π, +π]`.
#[inline]
fn wrap_angle(a: f32) -> f32 {
    use std::f32::consts::PI;
    let mut a = a % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sample(t: f64, yaw: f32, pitch: f32) -> HeadPoseSample {
        HeadPoseSample {
            timestamp_s: t,
            yaw_rad: yaw,
            pitch_rad: pitch,
        }
    }

    // ── ConstantPositionPredictor ────────────────────────────────────────────

    #[test]
    fn constant_position_no_history_zero_confidence() {
        let p = ConstantPositionPredictor::new();
        let pred = p.predict(1.0);
        assert_eq!(pred.confidence, 0.0);
        assert_eq!(pred.yaw_rad, 0.0);
        assert_eq!(pred.pitch_rad, 0.0);
    }

    #[test]
    fn constant_position_returns_last_sample() {
        let mut p = ConstantPositionPredictor::new();
        p.push(sample(0.0, 1.0, 0.5));
        p.push(sample(0.1, 1.2, 0.6));
        let pred = p.predict(5.0); // Far future — should still return last
        assert!((pred.yaw_rad - 1.2).abs() < 1e-5, "yaw={}", pred.yaw_rad);
        assert!((pred.pitch_rad - 0.6).abs() < 1e-5);
        assert_eq!(pred.confidence, 1.0);
    }

    #[test]
    fn constant_position_reset_clears_state() {
        let mut p = ConstantPositionPredictor::new();
        p.push(sample(0.0, 0.5, 0.1));
        p.reset();
        assert!(p.is_empty());
        assert_eq!(p.predict(1.0).confidence, 0.0);
    }

    // ── LinearVelocityPredictor ──────────────────────────────────────────────

    #[test]
    fn linear_predictor_extrapolates_constant_velocity() {
        let mut p = LinearVelocityPredictor::new(10);
        // Yaw increases at 0.1 rad/s
        p.push(sample(0.0, 0.0, 0.0));
        p.push(sample(1.0, 0.1, 0.0));
        // Predict 1 s into the future from last sample
        let pred = p.predict(2.0);
        assert!((pred.yaw_rad - 0.2).abs() < 1e-4, "yaw={}", pred.yaw_rad);
    }

    #[test]
    fn linear_predictor_pitch_clamped_to_pi_over_2() {
        let mut p = LinearVelocityPredictor::new(4);
        p.push(sample(0.0, 0.0, 1.4)); // Near pole
        p.push(sample(1.0, 0.0, 1.5)); // Positive pitch velocity
        let pred = p.predict(100.0); // Far future — pitch must not exceed π/2
        assert!(pred.pitch_rad <= std::f32::consts::FRAC_PI_2 + 1e-5);
    }

    #[test]
    fn linear_predictor_wraps_yaw() {
        let mut p = LinearVelocityPredictor::new(4);
        // Yaw starting at π-0.05, moving at +0.1 rad/s → will cross π
        p.push(sample(0.0, PI - 0.05, 0.0));
        p.push(sample(1.0, PI + 0.05, 0.0)); // stored as wrapped
        let pred = p.predict(2.0);
        assert!(
            pred.yaw_rad.abs() <= PI + 1e-5,
            "yaw not wrapped: {}",
            pred.yaw_rad
        );
    }

    #[test]
    fn linear_predictor_capacity_evicts_old_samples() {
        let mut p = LinearVelocityPredictor::new(3);
        for i in 0..10 {
            p.push(sample(i as f64, i as f32 * 0.1, 0.0));
        }
        assert_eq!(p.len(), 3);
    }

    // ── WeightedHistoryPredictor ─────────────────────────────────────────────

    #[test]
    fn weighted_predictor_invalid_alpha_returns_error() {
        assert!(WeightedHistoryPredictor::new(5, 0.0).is_err());
        assert!(WeightedHistoryPredictor::new(5, 1.0).is_err());
        assert!(WeightedHistoryPredictor::new(5, -0.1).is_err());
    }

    #[test]
    fn weighted_predictor_constant_velocity_matches_linear() {
        let mut w = WeightedHistoryPredictor::new(8, 0.7).unwrap();
        let mut l = LinearVelocityPredictor::new(8);
        for i in 0..5 {
            let s = sample(i as f64, i as f32 * 0.2, 0.0);
            w.push(s);
            l.push(s);
        }
        let wp = w.predict(6.0);
        let lp = l.predict(6.0);
        // Both models should agree on direction for constant velocity
        assert!(
            (wp.yaw_rad - lp.yaw_rad).abs() < 0.2,
            "w={:.3}, l={:.3}",
            wp.yaw_rad,
            lp.yaw_rad
        );
    }

    #[test]
    fn weighted_predictor_confidence_grows_with_samples() {
        let mut p = WeightedHistoryPredictor::new(10, 0.7).unwrap();
        let c0 = p.predict(0.0).confidence;
        p.push(sample(0.0, 0.0, 0.0));
        p.push(sample(0.1, 0.0, 0.0));
        let c2 = p.predict(0.2).confidence;
        assert!(c2 > c0, "confidence should grow: c0={c0}, c2={c2}");
    }

    // ── ViewportRegion ───────────────────────────────────────────────────────

    #[test]
    fn viewport_region_contains_centre() {
        let r = ViewportRegion::from_fov(0.0, 0.0, PI / 2.0, PI / 3.0).unwrap();
        assert!(r.contains(0.0, 0.0));
    }

    #[test]
    fn viewport_region_does_not_contain_outside() {
        let r = ViewportRegion::from_fov(0.0, 0.0, PI / 4.0, PI / 4.0).unwrap();
        // A point 2 rad away in yaw should be outside a π/4 half-width region
        assert!(!r.contains(2.0, 0.0));
    }

    #[test]
    fn viewport_region_iou_identical_regions() {
        let r = ViewportRegion::from_fov(0.5, 0.2, 1.0, 0.8).unwrap();
        let iou = r.iou(&r);
        assert!(
            (iou - 1.0).abs() < 1e-5,
            "iou of identical regions should be 1: {iou}"
        );
    }

    #[test]
    fn viewport_region_iou_non_overlapping() {
        // Two regions separated far in yaw
        let a = ViewportRegion::from_fov(-1.5, 0.0, 0.5, 0.5).unwrap();
        let b = ViewportRegion::from_fov(1.5, 0.0, 0.5, 0.5).unwrap();
        let iou = a.iou(&b);
        assert!(iou < 1e-5, "non-overlapping iou should be ~0: {iou}");
    }

    #[test]
    fn viewport_region_from_fov_rejects_zero_fov() {
        assert!(ViewportRegion::from_fov(0.0, 0.0, 0.0, 1.0).is_err());
        assert!(ViewportRegion::from_fov(0.0, 0.0, 1.0, 0.0).is_err());
    }
}
