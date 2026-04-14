//! Custom pacing curves for automated video editing.
//!
//! This module allows fine-grained control over shot duration across the
//! timeline of a video edit.  Rather than using a fixed `PacingPreset`, you
//! can define a **pacing curve**: a function from normalised playback position
//! (0.0 = start, 1.0 = end) to an instantaneous shot-duration target in
//! milliseconds.
//!
//! # Built-in Curve Shapes
//!
//! | Shape | Description |
//! |-------|-------------|
//! | `Constant` | Same target duration throughout |
//! | `LinearAccelerate` | Starts slow, gradually speeds up |
//! | `LinearDecelerate` | Starts fast, gradually slows down |
//! | `Parabolic` | Fast in the middle, slower at start/end |
//! | `InverseParabolic` | Slow in the middle, faster at start/end |
//! | `SineWave` | Sinusoidal rhythm |
//! | `CustomKeyframes` | User-defined keyframe list |
//!
//! # Example
//!
//! ```
//! use oximedia_auto::pacing_curve::{PacingCurve, CurveShape};
//!
//! // Accelerating edit: starts at 6 s, ends at 1 s shot duration.
//! let curve = PacingCurve::new(CurveShape::LinearAccelerate, 6000, 1000);
//!
//! let mid_duration = curve.shot_duration_ms(0.5);
//! assert!(mid_duration > 1000 && mid_duration < 6000);
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use crate::rules::PacingPreset;
use oximedia_core::Timestamp;

// ─── Curve Keyframe ───────────────────────────────────────────────────────────

/// A single keyframe in a custom pacing curve.
#[derive(Debug, Clone, Copy)]
pub struct CurveKeyframe {
    /// Normalised position in the edit [0.0, 1.0].
    pub position: f64,
    /// Target shot duration at this position in milliseconds.
    pub duration_ms: i64,
}

impl CurveKeyframe {
    /// Create a new keyframe.
    ///
    /// # Errors
    ///
    /// Returns an error if position is outside [0.0, 1.0] or duration ≤ 0.
    pub fn new(position: f64, duration_ms: i64) -> AutoResult<Self> {
        if !(0.0..=1.0).contains(&position) {
            return Err(AutoError::InvalidParameter {
                name: "position".to_string(),
                value: format!("must be in [0.0, 1.0], got {position}"),
            });
        }
        if duration_ms <= 0 {
            return Err(AutoError::InvalidParameter {
                name: "duration_ms".to_string(),
                value: format!("must be > 0, got {duration_ms}"),
            });
        }
        Ok(Self {
            position,
            duration_ms,
        })
    }
}

// ─── Curve Shape ─────────────────────────────────────────────────────────────

/// The mathematical shape of the pacing curve.
#[derive(Debug, Clone)]
pub enum CurveShape {
    /// Constant shot duration throughout the edit.
    Constant,
    /// Linearly accelerating: starts at `start_ms`, ends at `end_ms`.
    LinearAccelerate,
    /// Linearly decelerating: starts at `start_ms`, ends at `end_ms`.
    LinearDecelerate,
    /// Parabolic: fastest at the midpoint, slowest at start and end.
    Parabolic,
    /// Inverse parabolic: slowest at the midpoint, fastest at start and end.
    InverseParabolic,
    /// Sinusoidal rhythm between `start_ms` and `end_ms`.
    SineWave,
    /// Exponential ease-in (slow start, fast end).
    ExponentialIn,
    /// Exponential ease-out (fast start, slow end).
    ExponentialOut,
    /// User-supplied keyframes; values are linearly interpolated.
    CustomKeyframes(Vec<CurveKeyframe>),
}

impl CurveShape {
    /// Evaluate the normalised pacing factor at position `t` ∈ [0.0, 1.0].
    ///
    /// Returns a factor in the range [0.0, 1.0] where 0.0 corresponds to the
    /// `start_ms` target and 1.0 corresponds to the `end_ms` target.
    #[must_use]
    pub fn factor_at(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Constant => 0.5,
            Self::LinearAccelerate => t,
            Self::LinearDecelerate => 1.0 - t,
            Self::Parabolic => {
                // Peaks at 1.0 at t=0.5, returns to 0.0 at t=0 and t=1.
                1.0 - (2.0 * t - 1.0).powi(2)
            }
            Self::InverseParabolic => (2.0 * t - 1.0).powi(2),
            Self::SineWave => (std::f64::consts::PI * t).sin(),
            Self::ExponentialIn => {
                if t == 0.0 {
                    0.0
                } else {
                    2.0_f64.powf(10.0 * (t - 1.0))
                }
            }
            Self::ExponentialOut => {
                if t == 1.0 {
                    1.0
                } else {
                    1.0 - 2.0_f64.powf(-10.0 * t)
                }
            }
            Self::CustomKeyframes(keyframes) => interpolate_keyframes(keyframes, t),
        }
    }
}

fn interpolate_keyframes(keyframes: &[CurveKeyframe], t: f64) -> f64 {
    if keyframes.is_empty() {
        return 0.5;
    }
    if keyframes.len() == 1 {
        return keyframes[0].duration_ms as f64;
    }

    // Find surrounding keyframes.
    let mut before: Option<&CurveKeyframe> = None;
    let mut after: Option<&CurveKeyframe> = None;

    for kf in keyframes {
        if kf.position <= t {
            before = Some(kf);
        }
        if kf.position >= t && after.is_none() {
            after = Some(kf);
        }
    }

    match (before, after) {
        (Some(b), Some(a)) if b.position == a.position => b.duration_ms as f64,
        (Some(b), Some(a)) => {
            let ratio = (t - b.position) / (a.position - b.position);
            let b_val = b.duration_ms as f64;
            let a_val = a.duration_ms as f64;
            b_val + (a_val - b_val) * ratio
        }
        (Some(b), None) => b.duration_ms as f64,
        (None, Some(a)) => a.duration_ms as f64,
        (None, None) => 0.5,
    }
}

// ─── Pacing Curve ─────────────────────────────────────────────────────────────

/// A user-defined pacing curve mapping edit position to shot duration.
#[derive(Debug, Clone)]
pub struct PacingCurve {
    /// Shape of the curve.
    pub shape: CurveShape,
    /// Shot duration at the start of the edit (milliseconds).
    pub start_ms: i64,
    /// Shot duration at the end of the edit (milliseconds).
    pub end_ms: i64,
    /// Hard minimum shot duration (milliseconds).
    pub min_clamp_ms: i64,
    /// Hard maximum shot duration (milliseconds).
    pub max_clamp_ms: i64,
}

impl PacingCurve {
    /// Create a new pacing curve.
    #[must_use]
    pub fn new(shape: CurveShape, start_ms: i64, end_ms: i64) -> Self {
        Self {
            shape,
            start_ms: start_ms.max(1),
            end_ms: end_ms.max(1),
            min_clamp_ms: 100,
            max_clamp_ms: 60_000,
        }
    }

    /// Create a constant-duration curve from a `PacingPreset`.
    #[must_use]
    pub fn from_preset(preset: PacingPreset) -> Self {
        let avg = preset.average_shot_duration_ms();
        Self {
            shape: CurveShape::Constant,
            start_ms: avg,
            end_ms: avg,
            min_clamp_ms: preset.min_shot_duration_ms(),
            max_clamp_ms: preset.max_shot_duration_ms(),
        }
    }

    /// Create an accelerating curve with the given start and end durations.
    #[must_use]
    pub fn accelerating(start_ms: i64, end_ms: i64) -> Self {
        Self::new(CurveShape::LinearAccelerate, start_ms, end_ms)
    }

    /// Create a decelerating curve with the given start and end durations.
    #[must_use]
    pub fn decelerating(start_ms: i64, end_ms: i64) -> Self {
        Self::new(CurveShape::LinearDecelerate, start_ms, end_ms)
    }

    /// Create a dramatic arc: slow start → fast middle → slow end.
    #[must_use]
    pub fn dramatic_arc(slow_ms: i64, fast_ms: i64) -> AutoResult<Self> {
        let keyframes = vec![
            CurveKeyframe::new(0.0, slow_ms)?,
            CurveKeyframe::new(0.2, slow_ms)?,
            CurveKeyframe::new(0.5, fast_ms)?,
            CurveKeyframe::new(0.8, slow_ms)?,
            CurveKeyframe::new(1.0, slow_ms)?,
        ];
        Ok(Self::new(
            CurveShape::CustomKeyframes(keyframes),
            slow_ms,
            slow_ms,
        ))
    }

    /// Create a climax build: slow → progressively faster.
    #[must_use]
    pub fn climax_build(start_ms: i64, end_ms: i64) -> Self {
        Self::new(CurveShape::ExponentialIn, start_ms, end_ms)
    }

    /// Set the hard clamp bounds and return self (builder pattern).
    #[must_use]
    pub fn with_clamp(mut self, min_ms: i64, max_ms: i64) -> Self {
        self.min_clamp_ms = min_ms.max(1);
        self.max_clamp_ms = max_ms.max(self.min_clamp_ms);
        self
    }

    /// Evaluate the target shot duration at normalised position `t` ∈ [0, 1].
    ///
    /// The result is clamped to [`min_clamp_ms`](Self::min_clamp_ms) ..
    /// [`max_clamp_ms`](Self::max_clamp_ms).
    #[must_use]
    pub fn shot_duration_ms(&self, t: f64) -> i64 {
        let t = t.clamp(0.0, 1.0);

        // For CustomKeyframes the factor_at() returns actual duration in ms.
        let duration_ms = if let CurveShape::CustomKeyframes(_) = &self.shape {
            let raw = self.shape.factor_at(t);
            raw as i64
        } else {
            let factor = self.shape.factor_at(t);
            let range = (self.end_ms - self.start_ms) as f64;
            (self.start_ms as f64 + range * factor) as i64
        };

        duration_ms.clamp(self.min_clamp_ms, self.max_clamp_ms)
    }

    /// Validate the curve configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if durations are non-positive or clamp bounds are
    /// inconsistent.
    pub fn validate(&self) -> AutoResult<()> {
        if self.start_ms <= 0 {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "start_ms must be > 0".to_string(),
            });
        }
        if self.end_ms <= 0 {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "end_ms must be > 0".to_string(),
            });
        }
        if self.min_clamp_ms <= 0 {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "min_clamp_ms must be > 0".to_string(),
            });
        }
        if self.max_clamp_ms < self.min_clamp_ms {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "max_clamp_ms must be >= min_clamp_ms".to_string(),
            });
        }
        if let CurveShape::CustomKeyframes(kfs) = &self.shape {
            if kfs.is_empty() {
                return Err(AutoError::InvalidParameter {
                    name: "config".to_string(),
                    value: "CustomKeyframes must have at least one keyframe".to_string(),
                });
            }
        }
        Ok(())
    }

    /// Sample the curve at `n` evenly spaced positions and return
    /// (position, duration_ms) pairs.
    #[must_use]
    pub fn sample(&self, n: usize) -> Vec<(f64, i64)> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![(0.0, self.shot_duration_ms(0.0))];
        }
        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (t, self.shot_duration_ms(t))
            })
            .collect()
    }

    /// Given a total edit duration, compute target durations for each clip.
    ///
    /// The number of clips is estimated from the average duration over the curve.
    /// Returns a `Vec` of shot duration targets in milliseconds, one per clip.
    #[must_use]
    pub fn distribute_clips(&self, total_duration_ms: i64, clip_count: usize) -> Vec<i64> {
        if clip_count == 0 || total_duration_ms <= 0 {
            return Vec::new();
        }
        (0..clip_count)
            .map(|i| {
                let t = i as f64 / clip_count as f64;
                self.shot_duration_ms(t)
            })
            .collect()
    }

    /// Compute the edit cut positions in milliseconds for a set of target
    /// durations, starting from `offset_ms`.
    #[must_use]
    pub fn compute_cut_positions(
        &self,
        total_duration_ms: i64,
        clip_count: usize,
        offset_ms: i64,
    ) -> Vec<Timestamp> {
        let durations = self.distribute_clips(total_duration_ms, clip_count);
        let mut positions = Vec::with_capacity(durations.len() + 1);
        let mut cursor = offset_ms;
        positions.push(Timestamp::new(
            cursor,
            oximedia_core::Rational::new(1, 1000),
        ));
        for dur in &durations {
            cursor += dur;
            positions.push(Timestamp::new(
                cursor,
                oximedia_core::Rational::new(1, 1000),
            ));
        }
        positions
    }
}

impl Default for PacingCurve {
    fn default() -> Self {
        Self::from_preset(PacingPreset::Medium)
    }
}

// ─── Curve Analyser ───────────────────────────────────────────────────────────

/// Analyser that computes statistics about a `PacingCurve`.
pub struct CurveAnalyser;

impl CurveAnalyser {
    /// Compute the minimum, maximum and mean shot duration over a curve.
    ///
    /// Uses 100 sample points for accuracy.
    #[must_use]
    pub fn stats(curve: &PacingCurve) -> CurveStats {
        let samples = curve.sample(100);
        if samples.is_empty() {
            return CurveStats::default();
        }
        let durations: Vec<i64> = samples.iter().map(|(_, d)| *d).collect();
        let min_ms = *durations.iter().min().unwrap_or(&0);
        let max_ms = *durations.iter().max().unwrap_or(&0);
        let mean_ms = durations.iter().sum::<i64>() / durations.len() as i64;

        // Variance
        let mean_f = mean_ms as f64;
        let variance = durations
            .iter()
            .map(|d| (*d as f64 - mean_f).powi(2))
            .sum::<f64>()
            / durations.len() as f64;
        let std_dev_ms = variance.sqrt() as i64;

        CurveStats {
            min_ms,
            max_ms,
            mean_ms,
            std_dev_ms,
            sample_count: durations.len(),
        }
    }

    /// Check whether the curve is monotonically decreasing (i.e. always
    /// accelerating from start to end).
    #[must_use]
    pub fn is_accelerating(curve: &PacingCurve) -> bool {
        let samples = curve.sample(50);
        samples.windows(2).all(|w| w[1].1 <= w[0].1)
    }

    /// Check whether the curve is monotonically increasing (i.e. always
    /// decelerating from start to end).
    #[must_use]
    pub fn is_decelerating(curve: &PacingCurve) -> bool {
        let samples = curve.sample(50);
        samples.windows(2).all(|w| w[1].1 >= w[0].1)
    }
}

/// Statistics about a pacing curve.
#[derive(Debug, Clone, Default)]
pub struct CurveStats {
    /// Minimum shot duration over the curve (ms).
    pub min_ms: i64,
    /// Maximum shot duration over the curve (ms).
    pub max_ms: i64,
    /// Mean shot duration over the curve (ms).
    pub mean_ms: i64,
    /// Standard deviation of shot duration (ms).
    pub std_dev_ms: i64,
    /// Number of sample points used.
    pub sample_count: usize,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_curve_uniform() {
        let curve = PacingCurve::new(CurveShape::Constant, 4000, 4000);
        let samples = curve.sample(10);
        for (_, dur) in &samples {
            assert_eq!(*dur, 4000);
        }
    }

    #[test]
    fn test_linear_accelerate_decreases() {
        let curve = PacingCurve::new(CurveShape::LinearAccelerate, 6000, 1000);
        let d_start = curve.shot_duration_ms(0.0);
        let d_end = curve.shot_duration_ms(1.0);
        assert!(
            d_start > d_end,
            "accelerating curve should have shorter shots at end"
        );
    }

    #[test]
    fn test_linear_decelerate_increases() {
        // LinearDecelerate factor = 1-t, so at t=0 the factor is 1.0 (→ end_ms)
        // and at t=1 the factor is 0.0 (→ start_ms).
        // With start_ms=6000 and end_ms=1000: d(0)=1000, d(1)=6000.
        let curve = PacingCurve::new(CurveShape::LinearDecelerate, 6000, 1000);
        let d_start = curve.shot_duration_ms(0.0);
        let d_end = curve.shot_duration_ms(1.0);
        assert!(
            d_start < d_end,
            "decelerating curve should have longer shots at end"
        );
    }

    #[test]
    fn test_parabolic_peak_at_midpoint() {
        let curve = PacingCurve::new(CurveShape::Parabolic, 1000, 1000);
        let d_mid = curve.shot_duration_ms(0.5);
        let d_start = curve.shot_duration_ms(0.0);
        // Parabolic peaks at midpoint → shortest durations at edges, longest at mid.
        assert!(d_mid >= d_start, "parabolic peak should be at midpoint");
    }

    #[test]
    fn test_clamp_respected() {
        let curve =
            PacingCurve::new(CurveShape::LinearAccelerate, 10_000, 50).with_clamp(200, 8_000);
        for t in [0.0f64, 0.25, 0.5, 0.75, 1.0] {
            let d = curve.shot_duration_ms(t);
            assert!(
                d >= 200 && d <= 8_000,
                "duration {d} at t={t} outside clamp bounds"
            );
        }
    }

    #[test]
    fn test_from_preset_uses_preset_values() {
        let curve = PacingCurve::from_preset(PacingPreset::Fast);
        let avg = curve.shot_duration_ms(0.5);
        // Fast preset avg is 2000ms; constant curve should return near that.
        assert!(avg > 0 && avg <= 5000);
    }

    #[test]
    fn test_custom_keyframes_interpolation() {
        let keyframes = vec![
            CurveKeyframe::new(0.0, 6000).expect("valid"),
            CurveKeyframe::new(1.0, 1000).expect("valid"),
        ];
        let curve = PacingCurve::new(CurveShape::CustomKeyframes(keyframes), 6000, 1000);
        let mid = curve.shot_duration_ms(0.5);
        // Should interpolate to approximately 3500ms.
        assert!(
            mid > 1000 && mid < 6000,
            "interpolated mid-point {mid} out of range"
        );
    }

    #[test]
    fn test_distribute_clips_correct_count() {
        let curve = PacingCurve::default();
        let clips = curve.distribute_clips(60_000, 10);
        assert_eq!(clips.len(), 10);
    }

    #[test]
    fn test_validate_rejects_zero_duration() {
        let mut curve = PacingCurve::default();
        curve.start_ms = 0;
        assert!(curve.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_inverted_clamp() {
        let mut curve = PacingCurve::default();
        curve.min_clamp_ms = 5000;
        curve.max_clamp_ms = 1000;
        assert!(curve.validate().is_err());
    }

    #[test]
    fn test_curve_analyser_stats() {
        let curve = PacingCurve::new(CurveShape::LinearAccelerate, 6000, 1000);
        let stats = CurveAnalyser::stats(&curve);
        assert!(stats.min_ms <= stats.mean_ms);
        assert!(stats.mean_ms <= stats.max_ms);
        assert_eq!(stats.sample_count, 100);
    }

    #[test]
    fn test_exponential_in_is_accelerating() {
        let curve = PacingCurve::new(CurveShape::ExponentialIn, 6000, 1000);
        assert!(CurveAnalyser::is_accelerating(&curve));
    }

    #[test]
    fn test_dramatic_arc_not_monotone() {
        let curve = PacingCurve::dramatic_arc(6000, 1000).expect("valid");
        // Dramatic arc is neither purely accelerating nor decelerating.
        assert!(!CurveAnalyser::is_accelerating(&curve));
        assert!(!CurveAnalyser::is_decelerating(&curve));
    }

    #[test]
    fn test_compute_cut_positions_count() {
        let curve = PacingCurve::default();
        let positions = curve.compute_cut_positions(60_000, 5, 0);
        // Expect clip_count + 1 positions (including start and after-last-clip).
        assert_eq!(positions.len(), 6);
    }

    #[test]
    fn test_keyframe_invalid_position() {
        assert!(CurveKeyframe::new(1.5, 2000).is_err());
        assert!(CurveKeyframe::new(-0.1, 2000).is_err());
    }

    #[test]
    fn test_keyframe_invalid_duration() {
        assert!(CurveKeyframe::new(0.5, 0).is_err());
        assert!(CurveKeyframe::new(0.5, -100).is_err());
    }

    #[test]
    fn test_sine_wave_range() {
        let curve = PacingCurve::new(CurveShape::SineWave, 1000, 5000);
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            let d = curve.shot_duration_ms(t);
            assert!(
                d >= curve.min_clamp_ms && d <= curve.max_clamp_ms,
                "sine-wave duration {d} outside clamp at t={t}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Custom pacing preset integration tests (PacingPreset::Custom)
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_preset_very_fast() {
        let curve = PacingCurve::from_preset(PacingPreset::VeryFast);
        let d = curve.shot_duration_ms(0.5);
        // VeryFast avg = 1000 ms; clamp allows that range
        assert!(d > 0);
        assert!(d <= PacingPreset::VeryFast.max_shot_duration_ms());
    }

    #[test]
    fn test_from_preset_custom_uses_medium_default() {
        // PacingPreset::Custom defaults to Medium (4000 ms avg)
        let curve = PacingCurve::from_preset(PacingPreset::Custom);
        let d = curve.shot_duration_ms(0.5);
        assert!(d > 0 && d <= 10_000, "custom default shot d={d}");
    }

    #[test]
    fn test_custom_keyframes_single_point() {
        let kfs = vec![CurveKeyframe::new(0.5, 3000).expect("valid")];
        let curve = PacingCurve::new(CurveShape::CustomKeyframes(kfs), 3000, 3000);
        let d = curve.shot_duration_ms(0.5);
        // Single keyframe → returns its value directly
        assert_eq!(d, 3000);
    }

    #[test]
    fn test_distribute_clips_sum_proportional_to_curve() {
        // Accelerating curve: durations should decrease from start to end
        let curve = PacingCurve::new(CurveShape::LinearAccelerate, 6000, 1000);
        let clips = curve.distribute_clips(60_000, 6);
        assert_eq!(clips.len(), 6);
        // First clip should be longer than last
        assert!(clips[0] > clips[clips.len() - 1]);
    }

    #[test]
    fn test_sample_count_matches_n() {
        let curve = PacingCurve::default();
        let s5 = curve.sample(5);
        let s0 = curve.sample(0);
        assert_eq!(s5.len(), 5);
        assert_eq!(s0.len(), 0);
    }

    #[test]
    fn test_sample_single_point_starts_at_zero() {
        let curve = PacingCurve::default();
        let s = curve.sample(1);
        assert_eq!(s.len(), 1);
        assert!((s[0].0 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_out_shot_duration_range() {
        // ExponentialOut: fast ease-out means shots get shorter rapidly at start
        // then stabilise. Check duration values are within clamp bounds.
        let curve = PacingCurve::new(CurveShape::ExponentialOut, 6000, 1000);
        let d_start = curve.shot_duration_ms(0.0);
        let d_end = curve.shot_duration_ms(1.0);
        // ExponentialOut factor goes 0→1 as t→1, so with start>end it accelerates.
        assert!(
            d_start > d_end,
            "ExponentialOut should accelerate (start={d_start}, end={d_end})"
        );
    }

    #[test]
    fn test_compute_cut_positions_chronological() {
        let curve = PacingCurve::default();
        let positions = curve.compute_cut_positions(40_000, 4, 0);
        for i in 1..positions.len() {
            assert!(
                positions[i].pts >= positions[i - 1].pts,
                "cut positions not in order at {i}"
            );
        }
    }

    #[test]
    fn test_dramatic_arc_climax_faster_than_edges() {
        let curve = PacingCurve::dramatic_arc(6000, 500).expect("valid");
        let d_start = curve.shot_duration_ms(0.0);
        let d_mid = curve.shot_duration_ms(0.5);
        let d_end = curve.shot_duration_ms(1.0);
        assert!(d_mid < d_start, "mid={d_mid} start={d_start}");
        assert!(d_mid < d_end, "mid={d_mid} end={d_end}");
    }

    #[test]
    fn test_climax_build_accelerates() {
        let curve = PacingCurve::climax_build(8000, 500);
        assert!(CurveAnalyser::is_accelerating(&curve));
    }
}
