#![allow(dead_code)]
//! Timecode interpolation between known reference points.
//!
//! Provides frame-accurate timecode interpolation for situations where
//! timecode values are only available at certain intervals (e.g., keyframes,
//! LTC sync points) and intermediate values must be derived.

use crate::{FrameRate, Timecode, TimecodeError};

/// A known timecode reference point at a specific sample or frame position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcRefPoint {
    /// The timecode value at this reference point.
    pub timecode: Timecode,
    /// The absolute sample or frame position in the media stream.
    pub position: u64,
}

impl TcRefPoint {
    /// Creates a new timecode reference point.
    pub fn new(timecode: Timecode, position: u64) -> Self {
        Self { timecode, position }
    }
}

/// Interpolation strategy for deriving intermediate timecodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Linear frame counting between reference points.
    Linear,
    /// Nearest reference point (snap to closest known value).
    Nearest,
    /// Forward-only: always use the preceding reference point and count forward.
    ForwardOnly,
}

/// Timecode interpolator that derives frame-accurate timecodes from sparse reference points.
#[derive(Debug, Clone)]
pub struct TcInterpolator {
    /// Sorted list of reference points (by position).
    refs: Vec<TcRefPoint>,
    /// Frame rate to use for interpolation.
    frame_rate: FrameRate,
    /// Interpolation mode.
    mode: InterpolationMode,
    /// Maximum allowable gap (in frames) before interpolation is considered unreliable.
    max_gap: u64,
}

impl TcInterpolator {
    /// Creates a new interpolator with the given frame rate and mode.
    pub fn new(frame_rate: FrameRate, mode: InterpolationMode) -> Self {
        Self {
            refs: Vec::new(),
            frame_rate,
            mode,
            max_gap: 300, // default: 10 seconds at 30fps
        }
    }

    /// Sets the maximum allowable gap in frames.
    pub fn with_max_gap(mut self, gap: u64) -> Self {
        self.max_gap = gap;
        self
    }

    /// Adds a reference point. Points are kept sorted by position.
    pub fn add_ref(&mut self, point: TcRefPoint) {
        let idx = self
            .refs
            .binary_search_by_key(&point.position, |r| r.position)
            .unwrap_or_else(|i| i);
        self.refs.insert(idx, point);
    }

    /// Returns the number of stored reference points.
    pub fn ref_count(&self) -> usize {
        self.refs.len()
    }

    /// Clears all reference points.
    pub fn clear(&mut self) {
        self.refs.clear();
    }

    /// Returns the frame rate used for interpolation.
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    /// Returns the interpolation mode.
    pub fn mode(&self) -> InterpolationMode {
        self.mode
    }

    /// Returns the maximum gap setting.
    pub fn max_gap(&self) -> u64 {
        self.max_gap
    }

    /// Interpolates the timecode at the given position.
    ///
    /// Returns `None` if there are no reference points or the position is out of range
    /// for the chosen interpolation mode.
    pub fn interpolate(&self, position: u64) -> Option<Result<Timecode, TimecodeError>> {
        if self.refs.is_empty() {
            return None;
        }

        match self.mode {
            InterpolationMode::Linear => self.interpolate_linear(position),
            InterpolationMode::Nearest => self.interpolate_nearest(position),
            InterpolationMode::ForwardOnly => self.interpolate_forward(position),
        }
    }

    /// Linear interpolation: find the bracketing reference points and count frames.
    fn interpolate_linear(&self, position: u64) -> Option<Result<Timecode, TimecodeError>> {
        // If exactly on a reference point, return it directly
        if let Ok(idx) = self.refs.binary_search_by_key(&position, |r| r.position) {
            return Some(Ok(self.refs[idx].timecode));
        }

        // Find the insertion point
        let idx = self
            .refs
            .binary_search_by_key(&position, |r| r.position)
            .unwrap_or_else(|i| i);

        if idx == 0 {
            // Before all reference points: extrapolate backward from first
            let first = &self.refs[0];
            let delta = first.position.saturating_sub(position);
            if delta > self.max_gap {
                return None;
            }
            let base_frames = first.timecode.to_frames();
            let target_frames = base_frames.saturating_sub(delta);
            Some(Timecode::from_frames(target_frames, self.frame_rate))
        } else if idx >= self.refs.len() {
            // After all reference points: extrapolate forward from last
            let last = &self.refs[self.refs.len() - 1];
            let delta = position.saturating_sub(last.position);
            if delta > self.max_gap {
                return None;
            }
            let base_frames = last.timecode.to_frames();
            Some(Timecode::from_frames(base_frames + delta, self.frame_rate))
        } else {
            // Between two points: use the earlier one and count forward
            let prev = &self.refs[idx - 1];
            let delta = position - prev.position;
            if delta > self.max_gap {
                return None;
            }
            let base_frames = prev.timecode.to_frames();
            Some(Timecode::from_frames(base_frames + delta, self.frame_rate))
        }
    }

    /// Nearest-neighbor: snap to the closest reference point.
    fn interpolate_nearest(&self, position: u64) -> Option<Result<Timecode, TimecodeError>> {
        let idx = self
            .refs
            .binary_search_by_key(&position, |r| r.position)
            .unwrap_or_else(|i| i);

        let candidate = if idx == 0 {
            &self.refs[0]
        } else if idx >= self.refs.len() {
            &self.refs[self.refs.len() - 1]
        } else {
            let dist_prev = position - self.refs[idx - 1].position;
            let dist_next = self.refs[idx].position - position;
            if dist_prev <= dist_next {
                &self.refs[idx - 1]
            } else {
                &self.refs[idx]
            }
        };

        let gap = position.abs_diff(candidate.position);

        if gap > self.max_gap {
            return None;
        }

        Some(Ok(candidate.timecode))
    }

    /// Forward-only: always use preceding reference and count frames forward.
    fn interpolate_forward(&self, position: u64) -> Option<Result<Timecode, TimecodeError>> {
        if let Ok(idx) = self.refs.binary_search_by_key(&position, |r| r.position) {
            return Some(Ok(self.refs[idx].timecode));
        }

        let idx = self
            .refs
            .binary_search_by_key(&position, |r| r.position)
            .unwrap_or_else(|i| i);

        if idx == 0 {
            return None; // no preceding reference
        }

        let prev = &self.refs[idx - 1];
        let delta = position - prev.position;
        if delta > self.max_gap {
            return None;
        }
        let base_frames = prev.timecode.to_frames();
        Some(Timecode::from_frames(base_frames + delta, self.frame_rate))
    }

    /// Checks whether the reference points are consistent
    /// (i.e., timecode increments match positional deltas).
    pub fn validate_consistency(&self) -> Vec<ConsistencyIssue> {
        let mut issues = Vec::new();
        for pair in self.refs.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            let pos_delta = b.position - a.position;
            let tc_delta = b
                .timecode
                .to_frames()
                .saturating_sub(a.timecode.to_frames());
            if pos_delta != tc_delta {
                issues.push(ConsistencyIssue {
                    position_a: a.position,
                    position_b: b.position,
                    expected_delta: pos_delta,
                    actual_tc_delta: tc_delta,
                });
            }
        }
        issues
    }
}

/// A consistency issue between two reference points.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsistencyIssue {
    /// Position of the first reference point.
    pub position_a: u64,
    /// Position of the second reference point.
    pub position_b: u64,
    /// Expected frame delta based on positional difference.
    pub expected_delta: u64,
    /// Actual timecode frame delta.
    pub actual_tc_delta: u64,
}

/// Batch interpolation result.
#[derive(Debug, Clone)]
pub struct BatchInterpolationResult {
    /// The position that was queried.
    pub position: u64,
    /// The interpolated timecode, or `None` if out of range.
    pub timecode: Option<Timecode>,
    /// Whether this result is considered reliable (within max_gap).
    pub reliable: bool,
}

/// Performs batch interpolation for a sorted list of positions.
pub fn batch_interpolate(
    interp: &TcInterpolator,
    positions: &[u64],
) -> Vec<BatchInterpolationResult> {
    positions
        .iter()
        .map(|&pos| {
            let result = interp.interpolate(pos);
            let (tc, reliable) = match result {
                Some(Ok(tc)) => (Some(tc), true),
                Some(Err(_)) => (None, false),
                None => (None, false),
            };
            BatchInterpolationResult {
                position: pos,
                timecode: tc,
                reliable,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tc(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_ref_point_creation() {
        let tc = make_tc(1, 0, 0, 0);
        let rp = TcRefPoint::new(tc, 90000);
        assert_eq!(rp.position, 90000);
        assert_eq!(rp.timecode, tc);
    }

    #[test]
    fn test_interpolator_creation() {
        let interp = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear);
        assert_eq!(interp.ref_count(), 0);
        assert_eq!(interp.mode(), InterpolationMode::Linear);
        assert_eq!(interp.max_gap(), 300);
    }

    #[test]
    fn test_add_ref_sorted() {
        let mut interp = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear);
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 1, 0), 25));
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 0, 0), 0));
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 2, 0), 50));
        assert_eq!(interp.ref_count(), 3);
        // Verify sorting
        assert_eq!(interp.refs[0].position, 0);
        assert_eq!(interp.refs[1].position, 25);
        assert_eq!(interp.refs[2].position, 50);
    }

    #[test]
    fn test_interpolate_exact_match() {
        let mut interp = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear);
        let tc = make_tc(0, 0, 1, 0);
        interp.add_ref(TcRefPoint::new(tc, 25));
        let result = interp
            .interpolate(25)
            .expect("interpolation should succeed")
            .expect("interpolation should succeed");
        assert_eq!(result, tc);
    }

    #[test]
    fn test_interpolate_linear_between() {
        let mut interp = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear);
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 0, 0), 0));
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 2, 0), 50));
        // Position 10 should be 10 frames from start => 00:00:00:10
        let result = interp
            .interpolate(10)
            .expect("interpolation should succeed")
            .expect("interpolation should succeed");
        assert_eq!(result.hours, 0);
        assert_eq!(result.minutes, 0);
        assert_eq!(result.seconds, 0);
        assert_eq!(result.frames, 10);
    }

    #[test]
    fn test_interpolate_forward_extrapolation() {
        let mut interp =
            TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear).with_max_gap(500);
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 0, 0), 0));
        // Position 30 => extrapolate forward => 00:00:01:05
        let result = interp
            .interpolate(30)
            .expect("interpolation should succeed")
            .expect("interpolation should succeed");
        assert_eq!(result.seconds, 1);
        assert_eq!(result.frames, 5);
    }

    #[test]
    fn test_interpolate_empty() {
        let interp = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear);
        assert!(interp.interpolate(10).is_none());
    }

    #[test]
    fn test_interpolate_nearest() {
        let mut interp = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Nearest);
        let tc0 = make_tc(0, 0, 0, 0);
        let tc1 = make_tc(0, 0, 2, 0);
        interp.add_ref(TcRefPoint::new(tc0, 0));
        interp.add_ref(TcRefPoint::new(tc1, 50));
        // Position 20 is closer to 0
        let result = interp
            .interpolate(20)
            .expect("interpolation should succeed")
            .expect("interpolation should succeed");
        assert_eq!(result, tc0);
        // Position 30 is closer to 50
        let result = interp
            .interpolate(30)
            .expect("interpolation should succeed")
            .expect("interpolation should succeed");
        assert_eq!(result, tc1);
    }

    #[test]
    fn test_interpolate_forward_only() {
        let mut interp = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::ForwardOnly);
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 0, 0), 10));
        // Before first ref => None
        assert!(interp.interpolate(5).is_none());
        // After first ref => count forward
        let result = interp
            .interpolate(15)
            .expect("interpolation should succeed")
            .expect("interpolation should succeed");
        assert_eq!(result.frames, 5);
    }

    #[test]
    fn test_max_gap_exceeded() {
        let mut interp =
            TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear).with_max_gap(10);
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 0, 0), 0));
        // Position 20 exceeds max_gap of 10
        assert!(interp.interpolate(20).is_none());
    }

    #[test]
    fn test_validate_consistency_ok() {
        let mut interp = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear);
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 0, 0), 0));
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 1, 0), 25));
        let issues = interp.validate_consistency();
        assert!(issues.is_empty());
    }

    #[test]
    fn test_validate_consistency_mismatch() {
        let mut interp = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear);
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 0, 0), 0));
        // Position delta = 30, but TC delta = 25 frames (1 second)
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 1, 0), 30));
        let issues = interp.validate_consistency();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].expected_delta, 30);
        assert_eq!(issues[0].actual_tc_delta, 25);
    }

    #[test]
    fn test_batch_interpolate() {
        let mut interp =
            TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear).with_max_gap(500);
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 0, 0), 0));
        let positions = vec![0, 5, 10, 25];
        let results = batch_interpolate(&interp, &positions);
        assert_eq!(results.len(), 4);
        assert!(results[0].reliable);
        assert_eq!(
            results[0].timecode.expect("timecode should exist").frames,
            0
        );
        assert_eq!(
            results[2].timecode.expect("timecode should exist").frames,
            10
        );
    }

    #[test]
    fn test_clear() {
        let mut interp = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear);
        interp.add_ref(TcRefPoint::new(make_tc(0, 0, 0, 0), 0));
        assert_eq!(interp.ref_count(), 1);
        interp.clear();
        assert_eq!(interp.ref_count(), 0);
    }
}
