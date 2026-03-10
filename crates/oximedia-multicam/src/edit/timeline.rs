//! Multi-angle timeline for multi-camera editing.

use super::{EditDecision, TransitionType};
use crate::sync::SyncResult;
use crate::{AngleId, FrameNumber, Result};

/// Multi-camera timeline
#[derive(Debug)]
pub struct MultiCamTimeline {
    /// Number of camera angles
    angle_count: usize,
    /// Edit decisions (angle switches)
    edit_decisions: Vec<EditDecision>,
    /// Synchronization result
    sync_result: Option<SyncResult>,
    /// Timeline duration (frames)
    duration: FrameNumber,
    /// Active angle at timeline start
    initial_angle: AngleId,
}

impl MultiCamTimeline {
    /// Create a new multi-camera timeline
    #[must_use]
    pub fn new(angle_count: usize) -> Self {
        Self {
            angle_count,
            edit_decisions: Vec::new(),
            sync_result: None,
            duration: 0,
            initial_angle: 0,
        }
    }

    /// Set synchronization result
    pub fn set_sync(&mut self, sync_result: SyncResult) {
        self.sync_result = Some(sync_result);
    }

    /// Add an edit decision (angle switch)
    ///
    /// # Errors
    ///
    /// Returns an error if the angle is invalid
    pub fn add_edit(&mut self, decision: EditDecision) -> Result<()> {
        if decision.angle >= self.angle_count {
            return Err(crate::MultiCamError::AngleNotFound(decision.angle));
        }

        // Insert decision in chronological order
        let insert_pos = self
            .edit_decisions
            .binary_search_by_key(&decision.frame, |d| d.frame)
            .unwrap_or_else(|pos| pos);

        self.edit_decisions.insert(insert_pos, decision);
        Ok(())
    }

    /// Add a cut to an angle
    ///
    /// # Errors
    ///
    /// Returns an error if the angle is invalid
    pub fn add_cut(&mut self, frame: FrameNumber, angle: AngleId) -> Result<()> {
        self.add_edit(EditDecision::cut(frame, angle))
    }

    /// Add a dissolve to an angle
    ///
    /// # Errors
    ///
    /// Returns an error if the angle is invalid
    pub fn add_dissolve(
        &mut self,
        frame: FrameNumber,
        angle: AngleId,
        duration: u32,
    ) -> Result<()> {
        self.add_edit(EditDecision::dissolve(frame, angle, duration))
    }

    /// Get active angle at specific frame
    #[must_use]
    pub fn get_angle_at_frame(&self, frame: FrameNumber) -> AngleId {
        let mut current_angle = self.initial_angle;

        for decision in &self.edit_decisions {
            if decision.frame > frame {
                break;
            }
            current_angle = decision.angle;
        }

        current_angle
    }

    /// Get all edit decisions
    #[must_use]
    pub fn edit_decisions(&self) -> &[EditDecision] {
        &self.edit_decisions
    }

    /// Remove edit decision at frame
    pub fn remove_edit(&mut self, frame: FrameNumber) -> bool {
        if let Some(pos) = self.edit_decisions.iter().position(|d| d.frame == frame) {
            self.edit_decisions.remove(pos);
            true
        } else {
            false
        }
    }

    /// Clear all edit decisions
    pub fn clear_edits(&mut self) {
        self.edit_decisions.clear();
    }

    /// Set timeline duration
    pub fn set_duration(&mut self, duration: FrameNumber) {
        self.duration = duration;
    }

    /// Get timeline duration
    #[must_use]
    pub fn duration(&self) -> FrameNumber {
        self.duration
    }

    /// Set initial angle
    pub fn set_initial_angle(&mut self, angle: AngleId) {
        self.initial_angle = angle;
    }

    /// Get number of angles
    #[must_use]
    pub fn angle_count(&self) -> usize {
        self.angle_count
    }

    /// Find next edit decision after frame
    #[must_use]
    pub fn next_edit(&self, frame: FrameNumber) -> Option<&EditDecision> {
        self.edit_decisions.iter().find(|d| d.frame > frame)
    }

    /// Find previous edit decision before frame
    #[must_use]
    pub fn previous_edit(&self, frame: FrameNumber) -> Option<&EditDecision> {
        self.edit_decisions.iter().rev().find(|d| d.frame < frame)
    }

    /// Get edit decision at specific frame
    #[must_use]
    pub fn edit_at_frame(&self, frame: FrameNumber) -> Option<&EditDecision> {
        self.edit_decisions.iter().find(|d| d.frame == frame)
    }

    /// Check if frame is within a transition
    #[must_use]
    pub fn is_in_transition(&self, frame: FrameNumber) -> bool {
        for decision in &self.edit_decisions {
            if decision.transition != TransitionType::Cut {
                let transition_end = decision.frame + u64::from(decision.duration);
                if frame >= decision.frame && frame < transition_end {
                    return true;
                }
            }
        }
        false
    }

    /// Get transition progress at frame (0.0 to 1.0)
    #[must_use]
    pub fn transition_progress(&self, frame: FrameNumber) -> Option<f32> {
        for decision in &self.edit_decisions {
            if decision.transition != TransitionType::Cut && decision.duration > 0 {
                let transition_end = decision.frame + u64::from(decision.duration);
                if frame >= decision.frame && frame < transition_end {
                    let progress = (frame - decision.frame) as f32 / decision.duration as f32;
                    return Some(progress.clamp(0.0, 1.0));
                }
            }
        }
        None
    }

    /// Get angle segments (continuous runs of same angle)
    #[must_use]
    pub fn get_segments(&self) -> Vec<AngleSegment> {
        let mut segments = Vec::new();
        let mut current_angle = self.initial_angle;
        let mut segment_start = 0;

        for decision in &self.edit_decisions {
            segments.push(AngleSegment {
                angle: current_angle,
                start_frame: segment_start,
                end_frame: decision.frame,
            });
            segment_start = decision.frame;
            current_angle = decision.angle;
        }

        // Add final segment
        if segment_start < self.duration {
            segments.push(AngleSegment {
                angle: current_angle,
                start_frame: segment_start,
                end_frame: self.duration,
            });
        }

        segments
    }

    /// Optimize edit decisions (remove redundant cuts)
    pub fn optimize(&mut self) {
        let mut optimized = Vec::new();
        let mut prev_angle = self.initial_angle;

        for decision in &self.edit_decisions {
            // Only keep decisions that actually change the angle
            if decision.angle != prev_angle {
                optimized.push(*decision);
                prev_angle = decision.angle;
            }
        }

        self.edit_decisions = optimized;
    }

    /// Ripple edit (move all edits after frame by offset)
    pub fn ripple(&mut self, from_frame: FrameNumber, offset: i64) {
        for decision in &mut self.edit_decisions {
            if decision.frame >= from_frame {
                decision.frame = (decision.frame as i64 + offset).max(0) as FrameNumber;
            }
        }
    }

    /// Validate timeline consistency
    #[must_use]
    pub fn validate(&self) -> bool {
        // Check that edits are in chronological order
        for i in 1..self.edit_decisions.len() {
            if self.edit_decisions[i].frame <= self.edit_decisions[i - 1].frame {
                return false;
            }
        }

        // Check that all angles are valid
        for decision in &self.edit_decisions {
            if decision.angle >= self.angle_count {
                return false;
            }
        }

        true
    }

    /// Export EDL (Edit Decision List)
    #[must_use]
    pub fn export_edl(&self) -> String {
        let mut edl = String::new();
        edl.push_str("TITLE: Multi-Camera Timeline\n");
        edl.push_str("FCM: NON-DROP FRAME\n\n");

        let mut edit_num = 1;
        let mut prev_frame = 0;

        for decision in &self.edit_decisions {
            let source_in = prev_frame;
            let source_out = decision.frame;
            let record_in = prev_frame;
            let record_out = decision.frame;

            edl.push_str(&format!(
                "{:03}  A{:03}  V     C        {:08} {:08} {:08} {:08}\n",
                edit_num,
                decision.angle + 1,
                source_in,
                source_out,
                record_in,
                record_out
            ));

            if decision.transition != TransitionType::Cut {
                edl.push_str(&format!(
                    "* TRANSITION: {:?} {} frames\n",
                    decision.transition, decision.duration
                ));
            }

            edit_num += 1;
            prev_frame = decision.frame;
        }

        edl
    }
}

/// Continuous segment of a single camera angle
#[derive(Debug, Clone, Copy)]
pub struct AngleSegment {
    /// Camera angle
    pub angle: AngleId,
    /// Start frame (inclusive)
    pub start_frame: FrameNumber,
    /// End frame (exclusive)
    pub end_frame: FrameNumber,
}

impl AngleSegment {
    /// Get duration in frames
    #[must_use]
    pub fn duration(&self) -> FrameNumber {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Check if frame is within segment
    #[must_use]
    pub fn contains(&self, frame: FrameNumber) -> bool {
        frame >= self.start_frame && frame < self.end_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_creation() {
        let timeline = MultiCamTimeline::new(3);
        assert_eq!(timeline.angle_count(), 3);
        assert_eq!(timeline.edit_decisions().len(), 0);
    }

    #[test]
    fn test_add_edit() {
        let mut timeline = MultiCamTimeline::new(3);
        assert!(timeline.add_cut(100, 1).is_ok());
        assert!(timeline.add_cut(200, 2).is_ok());
        assert_eq!(timeline.edit_decisions().len(), 2);
    }

    #[test]
    fn test_get_angle_at_frame() {
        let mut timeline = MultiCamTimeline::new(3);
        timeline
            .add_cut(100, 1)
            .expect("multicam test operation should succeed");
        timeline
            .add_cut(200, 2)
            .expect("multicam test operation should succeed");

        assert_eq!(timeline.get_angle_at_frame(50), 0);
        assert_eq!(timeline.get_angle_at_frame(150), 1);
        assert_eq!(timeline.get_angle_at_frame(250), 2);
    }

    #[test]
    fn test_transition_progress() {
        let mut timeline = MultiCamTimeline::new(2);
        timeline
            .add_dissolve(100, 1, 10)
            .expect("multicam test operation should succeed");

        assert_eq!(timeline.transition_progress(99), None);
        assert_eq!(timeline.transition_progress(100), Some(0.0));
        assert_eq!(timeline.transition_progress(105), Some(0.5));
        assert_eq!(timeline.transition_progress(109), Some(0.9));
        assert_eq!(timeline.transition_progress(110), None);
    }

    #[test]
    fn test_segments() {
        let mut timeline = MultiCamTimeline::new(3);
        timeline.set_duration(300);
        timeline
            .add_cut(100, 1)
            .expect("multicam test operation should succeed");
        timeline
            .add_cut(200, 2)
            .expect("multicam test operation should succeed");

        let segments = timeline.get_segments();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].angle, 0);
        assert_eq!(segments[1].angle, 1);
        assert_eq!(segments[2].angle, 2);
    }

    #[test]
    fn test_optimize() {
        let mut timeline = MultiCamTimeline::new(2);
        timeline
            .add_cut(100, 1)
            .expect("multicam test operation should succeed");
        timeline
            .add_cut(200, 1)
            .expect("multicam test operation should succeed"); // Redundant - same angle
        timeline
            .add_cut(300, 0)
            .expect("multicam test operation should succeed");

        timeline.optimize();
        assert_eq!(timeline.edit_decisions().len(), 2);
    }

    #[test]
    fn test_ripple() {
        let mut timeline = MultiCamTimeline::new(2);
        timeline
            .add_cut(100, 1)
            .expect("multicam test operation should succeed");
        timeline
            .add_cut(200, 0)
            .expect("multicam test operation should succeed");

        timeline.ripple(150, 50);
        assert_eq!(timeline.edit_decisions()[0].frame, 100);
        assert_eq!(timeline.edit_decisions()[1].frame, 250);
    }

    #[test]
    fn test_validate() {
        let mut timeline = MultiCamTimeline::new(2);
        timeline
            .add_cut(100, 1)
            .expect("multicam test operation should succeed");
        timeline
            .add_cut(200, 0)
            .expect("multicam test operation should succeed");
        assert!(timeline.validate());
    }
}
