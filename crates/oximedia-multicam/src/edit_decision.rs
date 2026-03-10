//! Edit Decision List (EDL) for multi-camera productions.

#![allow(dead_code)]

use std::collections::HashMap;

/// A single cut or edit decision: which camera is on-screen during a frame range.
#[derive(Debug, Clone)]
pub struct EditDecision {
    /// The camera angle (0-based index) that is active during this decision.
    pub camera_id: u32,
    /// First frame of this edit (inclusive).
    pub start_frame: u64,
    /// Last frame of this edit (inclusive).
    pub end_frame: u64,
}

impl EditDecision {
    /// Creates a new edit decision.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `end_frame < start_frame`.
    #[must_use]
    pub fn new(camera_id: u32, start_frame: u64, end_frame: u64) -> Self {
        debug_assert!(end_frame >= start_frame, "end_frame must be >= start_frame");
        Self {
            camera_id,
            start_frame,
            end_frame,
        }
    }

    /// Returns the duration of this edit in frames (inclusive range).
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame) + 1
    }

    /// Returns `true` if the given frame falls within this edit.
    #[must_use]
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.start_frame && frame <= self.end_frame
    }
}

/// Ordered list of edit decisions forming a complete cut sequence.
#[derive(Debug, Default)]
pub struct EditDecisionList {
    decisions: Vec<EditDecision>,
}

impl EditDecisionList {
    /// Creates an empty `EditDecisionList`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an edit decision to the end of the list.
    pub fn add(&mut self, decision: EditDecision) {
        self.decisions.push(decision);
    }

    /// Returns the edit decision that is active at `frame`, or `None` if
    /// no decision covers that frame.
    #[must_use]
    pub fn at_frame(&self, frame: u64) -> Option<&EditDecision> {
        self.decisions.iter().find(|d| d.contains_frame(frame))
    }

    /// Returns the total number of frames spanned by all decisions.
    ///
    /// This is the sum of individual decision durations, not the wall-clock
    /// span (decisions need not be contiguous).
    #[must_use]
    pub fn total_duration(&self) -> u64 {
        self.decisions
            .iter()
            .map(EditDecision::duration_frames)
            .sum()
    }

    /// Returns a map of `camera_id → percentage of total duration` (0.0–100.0).
    ///
    /// Returns an empty map if `total_duration()` is zero.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn camera_usage_pct(&self) -> HashMap<u32, f64> {
        let total = self.total_duration();
        if total == 0 {
            return HashMap::new();
        }
        let mut counts: HashMap<u32, u64> = HashMap::new();
        for d in &self.decisions {
            *counts.entry(d.camera_id).or_insert(0) += d.duration_frames();
        }
        counts
            .into_iter()
            .map(|(cam, frames)| (cam, (frames as f64 / total as f64) * 100.0))
            .collect()
    }

    /// Returns the number of edit decisions in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Returns `true` if the list contains no decisions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    /// Returns an iterator over all decisions.
    pub fn iter(&self) -> impl Iterator<Item = &EditDecision> {
        self.decisions.iter()
    }

    /// Returns all decisions that involve a specific camera.
    #[must_use]
    pub fn decisions_for_camera(&self, camera_id: u32) -> Vec<&EditDecision> {
        self.decisions
            .iter()
            .filter(|d| d.camera_id == camera_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_decision_duration_single_frame() {
        let ed = EditDecision::new(0, 10, 10);
        assert_eq!(ed.duration_frames(), 1);
    }

    #[test]
    fn test_edit_decision_duration_range() {
        let ed = EditDecision::new(1, 0, 49);
        assert_eq!(ed.duration_frames(), 50);
    }

    #[test]
    fn test_edit_decision_contains_frame() {
        let ed = EditDecision::new(0, 100, 200);
        assert!(ed.contains_frame(100));
        assert!(ed.contains_frame(150));
        assert!(ed.contains_frame(200));
        assert!(!ed.contains_frame(99));
        assert!(!ed.contains_frame(201));
    }

    #[test]
    fn test_edl_add_and_len() {
        let mut edl = EditDecisionList::new();
        edl.add(EditDecision::new(0, 0, 24));
        edl.add(EditDecision::new(1, 25, 49));
        assert_eq!(edl.len(), 2);
    }

    #[test]
    fn test_edl_is_empty() {
        let edl = EditDecisionList::new();
        assert!(edl.is_empty());
    }

    #[test]
    fn test_edl_at_frame_found() {
        let mut edl = EditDecisionList::new();
        edl.add(EditDecision::new(2, 0, 99));
        let found = edl.at_frame(50);
        assert!(found.is_some());
        assert_eq!(
            found
                .expect("multicam test operation should succeed")
                .camera_id,
            2
        );
    }

    #[test]
    fn test_edl_at_frame_not_found() {
        let mut edl = EditDecisionList::new();
        edl.add(EditDecision::new(0, 10, 20));
        assert!(edl.at_frame(5).is_none());
    }

    #[test]
    fn test_edl_total_duration() {
        let mut edl = EditDecisionList::new();
        edl.add(EditDecision::new(0, 0, 49)); // 50 frames
        edl.add(EditDecision::new(1, 50, 99)); // 50 frames
        assert_eq!(edl.total_duration(), 100);
    }

    #[test]
    fn test_edl_camera_usage_pct_even_split() {
        let mut edl = EditDecisionList::new();
        edl.add(EditDecision::new(0, 0, 49)); // 50 frames
        edl.add(EditDecision::new(1, 50, 99)); // 50 frames
        let pct = edl.camera_usage_pct();
        assert!((pct[&0] - 50.0).abs() < 0.001);
        assert!((pct[&1] - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_edl_camera_usage_pct_uneven() {
        let mut edl = EditDecisionList::new();
        edl.add(EditDecision::new(0, 0, 74)); // 75 frames
        edl.add(EditDecision::new(1, 75, 99)); // 25 frames
        let pct = edl.camera_usage_pct();
        assert!((pct[&0] - 75.0).abs() < 0.001);
        assert!((pct[&1] - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_edl_camera_usage_pct_empty() {
        let edl = EditDecisionList::new();
        assert!(edl.camera_usage_pct().is_empty());
    }

    #[test]
    fn test_edl_decisions_for_camera() {
        let mut edl = EditDecisionList::new();
        edl.add(EditDecision::new(0, 0, 24));
        edl.add(EditDecision::new(1, 25, 49));
        edl.add(EditDecision::new(0, 50, 74));
        let cam0 = edl.decisions_for_camera(0);
        assert_eq!(cam0.len(), 2);
        let cam1 = edl.decisions_for_camera(1);
        assert_eq!(cam1.len(), 1);
    }

    #[test]
    fn test_edl_iter() {
        let mut edl = EditDecisionList::new();
        edl.add(EditDecision::new(0, 0, 10));
        edl.add(EditDecision::new(1, 11, 20));
        let ids: Vec<u32> = edl.iter().map(|d| d.camera_id).collect();
        assert_eq!(ids, vec![0, 1]);
    }
}
