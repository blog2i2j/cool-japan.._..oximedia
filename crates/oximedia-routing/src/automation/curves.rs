//! Curve-based gain automation (linear, S-curve, exponential fades).
//!
//! Provides [`GainCurveSegment`] types and [`GainAutomation`] for scheduling
//! smooth gain transitions over time alongside the discrete automation
//! events in [`AutomationTimeline`](super::AutomationTimeline).

use serde::{Deserialize, Serialize};

/// The shape of a gain transition curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CurveType {
    /// Linear interpolation from start to end.
    Linear,
    /// S-curve (smoothstep) for gentle transitions.
    SCurve,
    /// Exponential fade-in (slow start, fast end).
    ExponentialIn,
    /// Exponential fade-out (fast start, slow end).
    ExponentialOut,
    /// Logarithmic (fast start, slow end — perceptually linear).
    Logarithmic,
    /// Instant jump (no interpolation, step at start).
    Step,
}

impl CurveType {
    /// Evaluate the curve at a normalised position `t` in [0, 1].
    ///
    /// Returns a value in [0, 1] representing the blend factor.
    pub fn evaluate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::SCurve => {
                // Hermite smoothstep: 3t^2 - 2t^3
                t * t * (3.0 - 2.0 * t)
            }
            Self::ExponentialIn => {
                // t^3
                t * t * t
            }
            Self::ExponentialOut => {
                // 1 - (1-t)^3
                let inv = 1.0 - t;
                1.0 - inv * inv * inv
            }
            Self::Logarithmic => {
                // ln(1 + t * (e-1)) — approximately perceptually linear
                let e_minus_1 = std::f64::consts::E - 1.0;
                (1.0 + t * e_minus_1).ln()
            }
            Self::Step => {
                if t >= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// A single gain automation segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GainCurveSegment {
    /// Start time in frames.
    pub start_frame: u64,
    /// End time in frames.
    pub end_frame: u64,
    /// Gain at the start in dB.
    pub start_gain_db: f32,
    /// Gain at the end in dB.
    pub end_gain_db: f32,
    /// Curve shape.
    pub curve_type: CurveType,
    /// Channel this segment applies to (None = master).
    pub channel: Option<usize>,
}

impl GainCurveSegment {
    /// Creates a new segment.
    pub fn new(
        start_frame: u64,
        end_frame: u64,
        start_gain_db: f32,
        end_gain_db: f32,
        curve_type: CurveType,
    ) -> Self {
        Self {
            start_frame,
            end_frame,
            start_gain_db,
            end_gain_db,
            curve_type,
            channel: None,
        }
    }

    /// Sets the channel.
    pub fn with_channel(mut self, channel: usize) -> Self {
        self.channel = Some(channel);
        self
    }

    /// Duration in frames.
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Evaluates the gain at the given frame position.
    pub fn evaluate(&self, frame: u64) -> f32 {
        if frame <= self.start_frame {
            return self.start_gain_db;
        }
        if frame >= self.end_frame {
            return self.end_gain_db;
        }

        let duration = self.duration_frames();
        if duration == 0 {
            return self.end_gain_db;
        }

        let t = (frame - self.start_frame) as f64 / duration as f64;
        let blend = self.curve_type.evaluate(t) as f32;
        self.start_gain_db + blend * (self.end_gain_db - self.start_gain_db)
    }

    /// Returns `true` if the given frame is within this segment's range.
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.start_frame && frame <= self.end_frame
    }
}

/// Manages a collection of gain automation curve segments.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GainAutomation {
    /// All segments, sorted by start_frame.
    segments: Vec<GainCurveSegment>,
}

impl GainAutomation {
    /// Creates a new, empty gain automation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a segment, maintaining sort order by start_frame.
    pub fn add_segment(&mut self, segment: GainCurveSegment) {
        let pos = self
            .segments
            .binary_search_by_key(&segment.start_frame, |s| s.start_frame)
            .unwrap_or_else(|e| e);
        self.segments.insert(pos, segment);
    }

    /// Evaluates the gain at the given frame for the given channel.
    ///
    /// If multiple segments overlap, the last one wins. If no segment
    /// covers the frame, returns `None`.
    pub fn evaluate(&self, frame: u64, channel: Option<usize>) -> Option<f32> {
        let mut result = None;
        for seg in &self.segments {
            if seg.contains_frame(frame) {
                let ch_match = seg.channel == channel || seg.channel.is_none();
                if ch_match {
                    result = Some(seg.evaluate(frame));
                }
            }
        }
        result
    }

    /// Returns all segments.
    pub fn segments(&self) -> &[GainCurveSegment] {
        &self.segments
    }

    /// Number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Removes all segments that end before the given frame (cleanup).
    pub fn prune_before(&mut self, frame: u64) {
        self.segments.retain(|s| s.end_frame >= frame);
    }

    /// Clears all segments.
    pub fn clear(&mut self) {
        self.segments.clear();
    }

    /// Creates a linear fade-in.
    pub fn add_fade_in(&mut self, start_frame: u64, end_frame: u64, target_db: f32) {
        self.add_segment(GainCurveSegment::new(
            start_frame,
            end_frame,
            f32::NEG_INFINITY,
            target_db,
            CurveType::ExponentialOut,
        ));
    }

    /// Creates a linear fade-out.
    pub fn add_fade_out(&mut self, start_frame: u64, end_frame: u64, from_db: f32) {
        self.add_segment(GainCurveSegment::new(
            start_frame,
            end_frame,
            from_db,
            f32::NEG_INFINITY,
            CurveType::ExponentialIn,
        ));
    }

    /// Creates an S-curve crossfade between two gain levels.
    pub fn add_crossfade(&mut self, start_frame: u64, end_frame: u64, from_db: f32, to_db: f32) {
        self.add_segment(GainCurveSegment::new(
            start_frame,
            end_frame,
            from_db,
            to_db,
            CurveType::SCurve,
        ));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_curve_endpoints() {
        assert!((CurveType::Linear.evaluate(0.0)).abs() < 1e-10);
        assert!((CurveType::Linear.evaluate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_curve_midpoint() {
        assert!((CurveType::Linear.evaluate(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_scurve_endpoints() {
        assert!((CurveType::SCurve.evaluate(0.0)).abs() < 1e-10);
        assert!((CurveType::SCurve.evaluate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scurve_midpoint() {
        // smoothstep at 0.5 = 0.5
        assert!((CurveType::SCurve.evaluate(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_in_slow_start() {
        // ExponentialIn should be less than linear at 0.5
        let exp_val = CurveType::ExponentialIn.evaluate(0.5);
        assert!(exp_val < 0.5);
    }

    #[test]
    fn test_exponential_out_fast_start() {
        // ExponentialOut should be greater than linear at 0.5
        let exp_val = CurveType::ExponentialOut.evaluate(0.5);
        assert!(exp_val > 0.5);
    }

    #[test]
    fn test_logarithmic_endpoints() {
        assert!((CurveType::Logarithmic.evaluate(0.0)).abs() < 1e-10);
        assert!((CurveType::Logarithmic.evaluate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_curve() {
        assert!((CurveType::Step.evaluate(0.0)).abs() < 1e-10);
        assert!((CurveType::Step.evaluate(0.5)).abs() < 1e-10);
        assert!((CurveType::Step.evaluate(0.999)).abs() < 1e-10);
        assert!((CurveType::Step.evaluate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_curve_clamping() {
        // Values outside [0, 1] should be clamped
        assert!((CurveType::Linear.evaluate(-1.0)).abs() < 1e-10);
        assert!((CurveType::Linear.evaluate(2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_segment_evaluate_before_start() {
        let seg = GainCurveSegment::new(100, 200, 0.0, -20.0, CurveType::Linear);
        assert!((seg.evaluate(50) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_segment_evaluate_after_end() {
        let seg = GainCurveSegment::new(100, 200, 0.0, -20.0, CurveType::Linear);
        assert!((seg.evaluate(300) - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn test_segment_evaluate_midpoint() {
        let seg = GainCurveSegment::new(0, 100, 0.0, -20.0, CurveType::Linear);
        let val = seg.evaluate(50);
        assert!((val - (-10.0)).abs() < 1e-4);
    }

    #[test]
    fn test_segment_duration() {
        let seg = GainCurveSegment::new(100, 300, 0.0, -6.0, CurveType::Linear);
        assert_eq!(seg.duration_frames(), 200);
    }

    #[test]
    fn test_segment_contains_frame() {
        let seg = GainCurveSegment::new(100, 200, 0.0, -6.0, CurveType::Linear);
        assert!(seg.contains_frame(100));
        assert!(seg.contains_frame(150));
        assert!(seg.contains_frame(200));
        assert!(!seg.contains_frame(50));
        assert!(!seg.contains_frame(201));
    }

    #[test]
    fn test_segment_with_channel() {
        let seg = GainCurveSegment::new(0, 100, 0.0, -6.0, CurveType::Linear).with_channel(3);
        assert_eq!(seg.channel, Some(3));
    }

    #[test]
    fn test_gain_automation_add_segment() {
        let mut auto = GainAutomation::new();
        auto.add_segment(GainCurveSegment::new(
            100,
            200,
            0.0,
            -6.0,
            CurveType::Linear,
        ));
        auto.add_segment(GainCurveSegment::new(0, 50, -20.0, 0.0, CurveType::SCurve));
        assert_eq!(auto.segment_count(), 2);
        // Should be sorted: frame 0 first, then frame 100
        assert_eq!(auto.segments()[0].start_frame, 0);
        assert_eq!(auto.segments()[1].start_frame, 100);
    }

    #[test]
    fn test_gain_automation_evaluate() {
        let mut auto = GainAutomation::new();
        auto.add_segment(GainCurveSegment::new(0, 100, 0.0, -20.0, CurveType::Linear));
        let val = auto.evaluate(50, None);
        assert!(val.is_some());
        assert!((val.expect("should be Some") - (-10.0)).abs() < 0.1);
    }

    #[test]
    fn test_gain_automation_evaluate_no_match() {
        let mut auto = GainAutomation::new();
        auto.add_segment(GainCurveSegment::new(
            100,
            200,
            0.0,
            -6.0,
            CurveType::Linear,
        ));
        assert!(auto.evaluate(50, None).is_none());
    }

    #[test]
    fn test_gain_automation_channel_filter() {
        let mut auto = GainAutomation::new();
        auto.add_segment(
            GainCurveSegment::new(0, 100, 0.0, -6.0, CurveType::Linear).with_channel(0),
        );
        // Channel 0 should match
        assert!(auto.evaluate(50, Some(0)).is_some());
        // Channel 1 should not
        assert!(auto.evaluate(50, Some(1)).is_none());
    }

    #[test]
    fn test_gain_automation_prune() {
        let mut auto = GainAutomation::new();
        auto.add_segment(GainCurveSegment::new(0, 50, 0.0, -6.0, CurveType::Linear));
        auto.add_segment(GainCurveSegment::new(
            100,
            200,
            -6.0,
            -12.0,
            CurveType::SCurve,
        ));
        auto.prune_before(60);
        assert_eq!(auto.segment_count(), 1);
        assert_eq!(auto.segments()[0].start_frame, 100);
    }

    #[test]
    fn test_gain_automation_clear() {
        let mut auto = GainAutomation::new();
        auto.add_segment(GainCurveSegment::new(0, 100, 0.0, -6.0, CurveType::Linear));
        auto.clear();
        assert_eq!(auto.segment_count(), 0);
    }

    #[test]
    fn test_add_fade_in() {
        let mut auto = GainAutomation::new();
        auto.add_fade_in(0, 100, 0.0);
        assert_eq!(auto.segment_count(), 1);
        let seg = &auto.segments()[0];
        assert_eq!(seg.curve_type, CurveType::ExponentialOut);
    }

    #[test]
    fn test_add_fade_out() {
        let mut auto = GainAutomation::new();
        auto.add_fade_out(0, 100, 0.0);
        assert_eq!(auto.segment_count(), 1);
        let seg = &auto.segments()[0];
        assert_eq!(seg.curve_type, CurveType::ExponentialIn);
    }

    #[test]
    fn test_add_crossfade() {
        let mut auto = GainAutomation::new();
        auto.add_crossfade(0, 100, 0.0, -6.0);
        assert_eq!(auto.segment_count(), 1);
        let seg = &auto.segments()[0];
        assert_eq!(seg.curve_type, CurveType::SCurve);
    }

    #[test]
    fn test_scurve_fade_smooth() {
        let seg = GainCurveSegment::new(0, 1000, 0.0, -40.0, CurveType::SCurve);
        // At quarter point, S-curve should be less than linear quarter
        let v_250 = seg.evaluate(250);
        let v_750 = seg.evaluate(750);
        // S-curve is symmetric: v(0.25) + v(0.75) ≈ start + end
        let sum = v_250 + v_750;
        assert!((sum - (-40.0)).abs() < 0.1);
    }

    #[test]
    fn test_zero_duration_segment() {
        let seg = GainCurveSegment::new(100, 100, 0.0, -6.0, CurveType::Linear);
        assert_eq!(seg.duration_frames(), 0);
        // At start_frame when duration is 0, evaluate returns end_gain_db
        // But frame 100 <= start_frame 100, so the early return gives start_gain_db
        // This is correct: at the exact start of a zero-duration segment,
        // the segment hasn't started yet.
        assert!((seg.evaluate(100) - 0.0).abs() < 1e-6);
        // After the segment (frame 101+), we get end gain
        assert!((seg.evaluate(101) - (-6.0)).abs() < 1e-6);
    }
}
