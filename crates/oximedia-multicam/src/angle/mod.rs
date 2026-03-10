//! Camera angle management for multi-camera production.
//!
//! This module provides data structures and algorithms for tracking camera angles,
//! computing coverage metrics, selecting optimal angles, and logging switching history.

#![allow(dead_code)]

/// Describes a physical camera angle in 3D space.
#[derive(Debug, Clone)]
pub struct CameraAngle {
    /// Unique identifier for this angle.
    pub id: u32,
    /// Human-readable label (e.g., "Wide Shot", "Close-Up Left").
    pub label: String,
    /// 3D position of the camera (x, y, z) in metres.
    pub position: (f32, f32, f32),
    /// Direction vector the camera is pointing (x, y, z), should be unit length.
    pub direction: (f32, f32, f32),
    /// Focal length of the lens in millimetres.
    pub focal_length_mm: f32,
}

impl CameraAngle {
    /// Creates a new `CameraAngle`.
    pub fn new(
        id: u32,
        label: impl Into<String>,
        position: (f32, f32, f32),
        direction: (f32, f32, f32),
        focal_length_mm: f32,
    ) -> Self {
        Self {
            id,
            label: label.into(),
            position,
            direction,
            focal_length_mm,
        }
    }
}

/// Coverage metrics for a single camera angle at a given moment.
#[derive(Debug, Clone, Copy)]
pub struct AngleCoverage {
    /// The angle ID this coverage data refers to.
    pub angle_id: u32,
    /// Horizontal field of view in degrees.
    pub field_of_view_deg: f32,
    /// Percentage of the frame occupied by the primary subject (0.0–100.0).
    pub subject_size_pct: f32,
}

impl AngleCoverage {
    /// Creates a new `AngleCoverage`.
    #[must_use]
    pub fn new(angle_id: u32, field_of_view_deg: f32, subject_size_pct: f32) -> Self {
        Self {
            angle_id,
            field_of_view_deg,
            subject_size_pct,
        }
    }
}

/// Selects the best camera angle based on coverage metrics.
pub struct AngleSelector {
    /// Weighting factor for variety (penalises staying on the same angle). Range: [0.0, 1.0].
    variety_factor: f32,
}

impl AngleSelector {
    /// Creates a new `AngleSelector` with the given variety factor.
    ///
    /// A higher `variety_factor` increases the penalty for reusing the previous angle.
    #[must_use]
    pub fn new(variety_factor: f32) -> Self {
        Self {
            variety_factor: variety_factor.clamp(0.0, 1.0),
        }
    }

    /// Selects the best angle from the provided coverage data.
    ///
    /// Scoring criteria:
    /// - Prefer the angle with the largest `subject_size_pct`.
    /// - Apply a penalty proportional to `variety_factor` when the candidate is the same
    ///   as `previous`.
    ///
    /// Returns the `angle_id` of the selected angle.
    /// If `angles` is empty, returns `0`.
    #[must_use]
    pub fn select_best_angle(&self, angles: &[AngleCoverage], previous: Option<u32>) -> u32 {
        if angles.is_empty() {
            return 0;
        }

        let best = angles.iter().max_by(|a, b| {
            let score_a = self.score(a, previous);
            let score_b = self.score(b, previous);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        best.map_or(0, |b| b.angle_id)
    }

    /// Computes the selection score for a single coverage entry.
    fn score(&self, coverage: &AngleCoverage, previous: Option<u32>) -> f32 {
        let base_score = coverage.subject_size_pct;
        let same_as_previous = previous == Some(coverage.angle_id);
        if same_as_previous {
            base_score * (1.0 - self.variety_factor)
        } else {
            base_score
        }
    }
}

impl Default for AngleSelector {
    fn default() -> Self {
        Self::new(0.3)
    }
}

/// Reason for a camera angle switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchReason {
    /// Switch initiated manually by the operator.
    Manual,
    /// Automatic switch triggered by dialogue detection.
    AutoDialogue,
    /// Automatic switch triggered by action/movement detection.
    AutoAction,
    /// Automatic switch for editorial emphasis.
    AutoEmphasis,
    /// B-roll cutaway.
    Broll,
}

/// Record of a single camera angle switch event.
#[derive(Debug, Clone)]
pub struct AngleSwitch {
    /// Angle switched away from.
    pub from_angle: u32,
    /// Angle switched to.
    pub to_angle: u32,
    /// Frame index at which the switch occurred.
    pub frame_idx: u64,
    /// Reason for the switch.
    pub reason: SwitchReason,
}

impl AngleSwitch {
    /// Creates a new `AngleSwitch`.
    #[must_use]
    pub fn new(from_angle: u32, to_angle: u32, frame_idx: u64, reason: SwitchReason) -> Self {
        Self {
            from_angle,
            to_angle,
            frame_idx,
            reason,
        }
    }
}

/// A chronological log of all camera angle switch events.
#[derive(Debug, Default)]
pub struct AngleSwitchLog {
    /// Ordered list of switch events.
    pub switches: Vec<AngleSwitch>,
}

impl AngleSwitchLog {
    /// Creates a new empty `AngleSwitchLog`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a switch event to the log.
    pub fn record(&mut self, switch: AngleSwitch) {
        self.switches.push(switch);
    }

    /// Returns the total number of recorded switches.
    #[must_use]
    pub fn len(&self) -> usize {
        self.switches.len()
    }

    /// Returns `true` if no switches have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.switches.is_empty()
    }
}

/// Aggregate statistics for a single camera angle derived from the switch log.
#[derive(Debug, Clone, Default)]
pub struct AngleStatistics {
    /// The angle ID.
    pub angle_id: u32,
    /// Total number of frames this angle was active (on screen).
    pub screen_time_frames: u64,
    /// Number of times this angle was switched to.
    pub switch_count: u32,
}

impl AngleStatistics {
    /// Computes per-angle statistics from a switch log.
    ///
    /// `total_frames` is the total duration of the session in frames and is used to attribute
    /// screen time to the last active angle.
    #[must_use]
    pub fn compute(log: &AngleSwitchLog, total_frames: u64) -> Vec<AngleStatistics> {
        if log.switches.is_empty() {
            return Vec::new();
        }

        use std::collections::HashMap;
        let mut screen_time: HashMap<u32, u64> = HashMap::new();
        let mut switch_count: HashMap<u32, u32> = HashMap::new();

        for i in 0..log.switches.len() {
            let sw = &log.switches[i];
            let next_frame = if i + 1 < log.switches.len() {
                log.switches[i + 1].frame_idx
            } else {
                total_frames
            };

            let duration = next_frame.saturating_sub(sw.frame_idx);
            *screen_time.entry(sw.to_angle).or_insert(0) += duration;
            *switch_count.entry(sw.to_angle).or_insert(0) += 1;
        }

        let mut stats: Vec<AngleStatistics> = screen_time
            .into_iter()
            .map(|(angle_id, frames)| AngleStatistics {
                angle_id,
                screen_time_frames: frames,
                switch_count: *switch_count.get(&angle_id).unwrap_or(&0),
            })
            .collect();

        stats.sort_by_key(|s| s.angle_id);
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_angle_creation() {
        let angle = CameraAngle::new(1, "Wide", (0.0, 1.5, -5.0), (0.0, 0.0, 1.0), 35.0);
        assert_eq!(angle.id, 1);
        assert_eq!(angle.label, "Wide");
        assert_eq!(angle.focal_length_mm, 35.0);
    }

    #[test]
    fn test_angle_coverage_creation() {
        let cov = AngleCoverage::new(2, 60.0, 45.0);
        assert_eq!(cov.angle_id, 2);
        assert_eq!(cov.field_of_view_deg, 60.0);
        assert_eq!(cov.subject_size_pct, 45.0);
    }

    #[test]
    fn test_angle_selector_picks_largest_subject() {
        let selector = AngleSelector::new(0.0);
        let angles = vec![
            AngleCoverage::new(0, 70.0, 20.0),
            AngleCoverage::new(1, 45.0, 60.0),
            AngleCoverage::new(2, 30.0, 40.0),
        ];
        let best = selector.select_best_angle(&angles, None);
        assert_eq!(best, 1);
    }

    #[test]
    fn test_angle_selector_variety_avoids_previous() {
        let selector = AngleSelector::new(1.0); // max penalty for same angle
        let angles = vec![
            AngleCoverage::new(0, 70.0, 50.0),
            AngleCoverage::new(1, 45.0, 49.0),
        ];
        // Without variety, angle 0 wins; with full penalty on previous=0, angle 1 should win
        let best = selector.select_best_angle(&angles, Some(0));
        assert_eq!(best, 1);
    }

    #[test]
    fn test_angle_selector_empty() {
        let selector = AngleSelector::default();
        let result = selector.select_best_angle(&[], None);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_switch_reason_variants() {
        let _manual = SwitchReason::Manual;
        let _auto = SwitchReason::AutoDialogue;
        let _broll = SwitchReason::Broll;
    }

    #[test]
    fn test_angle_switch_log_record() {
        let mut log = AngleSwitchLog::new();
        log.record(AngleSwitch::new(0, 1, 100, SwitchReason::Manual));
        log.record(AngleSwitch::new(1, 2, 200, SwitchReason::AutoAction));
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_angle_switch_log_empty() {
        let log = AngleSwitchLog::new();
        assert!(log.is_empty());
    }

    #[test]
    fn test_angle_statistics_compute() {
        let mut log = AngleSwitchLog::new();
        log.record(AngleSwitch::new(0, 1, 0, SwitchReason::Manual));
        log.record(AngleSwitch::new(1, 2, 100, SwitchReason::Manual));

        let stats = AngleStatistics::compute(&log, 300);
        assert_eq!(stats.len(), 2);

        let angle1 = stats
            .iter()
            .find(|s| s.angle_id == 1)
            .expect("multicam test operation should succeed");
        assert_eq!(angle1.screen_time_frames, 100);
        assert_eq!(angle1.switch_count, 1);

        let angle2 = stats
            .iter()
            .find(|s| s.angle_id == 2)
            .expect("multicam test operation should succeed");
        assert_eq!(angle2.screen_time_frames, 200); // 300 - 100
    }

    #[test]
    fn test_angle_statistics_empty_log() {
        let log = AngleSwitchLog::new();
        let stats = AngleStatistics::compute(&log, 1000);
        assert!(stats.is_empty());
    }

    #[test]
    fn test_angle_selector_no_penalty_without_previous() {
        let selector = AngleSelector::new(0.9);
        let angles = vec![
            AngleCoverage::new(0, 60.0, 80.0),
            AngleCoverage::new(1, 45.0, 50.0),
        ];
        // No previous set, so no penalty; angle 0 should win
        let best = selector.select_best_angle(&angles, None);
        assert_eq!(best, 0);
    }
}
