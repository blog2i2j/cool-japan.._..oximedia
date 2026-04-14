#![allow(dead_code)]
//! Multi-angle support for Matroska containers.
//!
//! In Matroska, multiple video tracks can represent different camera angles of
//! the same content.  This module provides abstractions for grouping tracks by
//! angle, selecting the active angle during playback, and managing angle
//! transitions at clean switch points (sync samples).
//!
//! # Matroska multi-angle encoding
//!
//! Matroska supports multi-angle via multiple video `TrackEntry` elements in
//! the same file.  Each angle track shares the same timescale and cluster
//! timestamps; the player picks which track to render at any given moment.
//!
//! This module does **not** parse Matroska directly — it operates on the
//! higher-level `StreamInfo` abstraction exposed by the container crate.
//!
//! # Example
//!
//! ```
//! use oximedia_container::multi_angle::{AngleDescriptor, AngleGroup, AngleManager};
//!
//! let mut manager = AngleManager::new();
//!
//! let group = AngleGroup::new("scene1")
//!     .with_angle(AngleDescriptor::new(0, "front"))
//!     .with_angle(AngleDescriptor::new(1, "side"))
//!     .with_angle(AngleDescriptor::new(2, "overhead"));
//!
//! manager.add_group(group);
//! assert_eq!(manager.group_count(), 1);
//! assert_eq!(manager.total_angles("scene1"), Some(3));
//!
//! manager.set_active_angle("scene1", 1).unwrap();
//! assert_eq!(manager.active_track("scene1"), Some(1));
//! ```

#![forbid(unsafe_code)]

use std::collections::HashMap;
use thiserror::Error;

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors for multi-angle operations.
#[derive(Debug, Error)]
pub enum MultiAngleError {
    /// The requested angle group was not found.
    #[error("angle group '{0}' not found")]
    GroupNotFound(String),

    /// The requested angle index is out of range.
    #[error("angle index {index} is out of range for group '{group}' (max {max})")]
    AngleOutOfRange {
        /// Group name.
        group: String,
        /// Requested angle index.
        index: usize,
        /// Maximum valid index.
        max: usize,
    },

    /// A duplicate group name was provided.
    #[error("duplicate angle group name: '{0}'")]
    DuplicateGroup(String),

    /// No angles are defined in the group.
    #[error("angle group '{0}' has no angles")]
    EmptyGroup(String),

    /// The switch point is not a sync sample.
    #[error("switch at sample {0} is not a sync sample")]
    NotSyncSample(u64),
}

// ─── Angle descriptor ───────────────────────────────────────────────────────

/// Describes a single camera angle within a multi-angle group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AngleDescriptor {
    /// Track index in the container (maps to `StreamInfo.index`).
    pub track_index: usize,
    /// Human-readable label (e.g. "Camera A", "Overhead", "Close-up").
    pub label: String,
    /// Optional ISO 639-2 language code for audio associated with this angle.
    pub language: Option<String>,
    /// Whether this angle is the default angle.
    pub is_default: bool,
    /// Optional metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl AngleDescriptor {
    /// Create a new angle descriptor.
    #[must_use]
    pub fn new(track_index: usize, label: impl Into<String>) -> Self {
        Self {
            track_index,
            label: label.into(),
            language: None,
            is_default: false,
            metadata: HashMap::new(),
        }
    }

    /// Set the language.
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Mark as the default angle.
    #[must_use]
    pub fn as_default(mut self) -> Self {
        self.is_default = true;
        self
    }

    /// Add a metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ─── Angle group ────────────────────────────────────────────────────────────

/// A group of alternative camera angles for the same scene or segment.
///
/// All angles in a group are expected to cover the same temporal range with
/// synchronised timestamps.
#[derive(Debug, Clone)]
pub struct AngleGroup {
    /// Unique name for this angle group.
    pub name: String,
    /// Camera angles in this group (index 0 is the primary/default).
    pub angles: Vec<AngleDescriptor>,
    /// Currently active angle index within `angles`.
    active_index: usize,
    /// Optional temporal range this group covers (start tick, end tick).
    pub time_range: Option<(u64, u64)>,
}

impl AngleGroup {
    /// Create a new angle group.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            angles: Vec::new(),
            active_index: 0,
            time_range: None,
        }
    }

    /// Add an angle to the group (builder pattern).
    #[must_use]
    pub fn with_angle(mut self, angle: AngleDescriptor) -> Self {
        self.angles.push(angle);
        self
    }

    /// Add an angle to the group.
    pub fn add_angle(&mut self, angle: AngleDescriptor) {
        self.angles.push(angle);
    }

    /// Set the temporal range (start tick, end tick).
    #[must_use]
    pub fn with_time_range(mut self, start: u64, end: u64) -> Self {
        self.time_range = Some((start, end));
        self
    }

    /// Number of angles in this group.
    #[must_use]
    pub fn angle_count(&self) -> usize {
        self.angles.len()
    }

    /// Get the currently active angle index.
    #[must_use]
    pub fn active_index(&self) -> usize {
        self.active_index
    }

    /// Get the currently active angle descriptor.
    #[must_use]
    pub fn active_angle(&self) -> Option<&AngleDescriptor> {
        self.angles.get(self.active_index)
    }

    /// Get the track index of the currently active angle.
    #[must_use]
    pub fn active_track_index(&self) -> Option<usize> {
        self.active_angle().map(|a| a.track_index)
    }

    /// Set the active angle by index.
    ///
    /// # Errors
    ///
    /// Returns [`MultiAngleError::AngleOutOfRange`] if the index is invalid.
    pub fn set_active(&mut self, index: usize) -> Result<(), MultiAngleError> {
        if index >= self.angles.len() {
            return Err(MultiAngleError::AngleOutOfRange {
                group: self.name.clone(),
                index,
                max: self.angles.len().saturating_sub(1),
            });
        }
        self.active_index = index;
        Ok(())
    }

    /// Find the default angle (the one marked `is_default`).
    ///
    /// If no angle is explicitly default, returns the first angle.
    #[must_use]
    pub fn default_angle_index(&self) -> Option<usize> {
        self.angles
            .iter()
            .position(|a| a.is_default)
            .or(if self.angles.is_empty() {
                None
            } else {
                Some(0)
            })
    }

    /// Get an angle by index.
    #[must_use]
    pub fn get_angle(&self, index: usize) -> Option<&AngleDescriptor> {
        self.angles.get(index)
    }

    /// Find an angle by track index.
    #[must_use]
    pub fn find_by_track(&self, track_index: usize) -> Option<(usize, &AngleDescriptor)> {
        self.angles
            .iter()
            .enumerate()
            .find(|(_, a)| a.track_index == track_index)
    }

    /// Find an angle by label (case-insensitive).
    #[must_use]
    pub fn find_by_label(&self, label: &str) -> Option<(usize, &AngleDescriptor)> {
        self.angles
            .iter()
            .enumerate()
            .find(|(_, a)| a.label.eq_ignore_ascii_case(label))
    }

    /// List all track indices in this group.
    #[must_use]
    pub fn all_track_indices(&self) -> Vec<usize> {
        self.angles.iter().map(|a| a.track_index).collect()
    }
}

// ─── Switch point ───────────────────────────────────────────────────────────

/// Describes a point where an angle switch can cleanly occur.
///
/// Angle switches should ideally happen at sync (key-frame) samples to avoid
/// visual artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AngleSwitchPoint {
    /// Timestamp in timescale ticks where the switch can occur.
    pub time_ticks: u64,
    /// Sample number (1-based) at the switch point.
    pub sample_number: u32,
    /// The angle index to switch from.
    pub from_angle: usize,
    /// The angle index to switch to.
    pub to_angle: usize,
}

// ─── Angle manager ──────────────────────────────────────────────────────────

/// Manages multiple angle groups and their active selections.
///
/// Provides high-level operations for multi-angle playback, including
/// angle enumeration, selection, and switch-point management.
#[derive(Debug, Clone, Default)]
pub struct AngleManager {
    /// Angle groups keyed by name.
    groups: HashMap<String, AngleGroup>,
    /// Ordered list of group names (insertion order).
    group_order: Vec<String>,
    /// Scheduled switch points.
    switch_points: Vec<AngleSwitchPoint>,
}

impl AngleManager {
    /// Create a new, empty angle manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an angle group.
    ///
    /// # Errors
    ///
    /// Returns [`MultiAngleError::DuplicateGroup`] if a group with the same
    /// name already exists.
    pub fn add_group(&mut self, group: AngleGroup) -> Result<(), MultiAngleError> {
        if self.groups.contains_key(&group.name) {
            return Err(MultiAngleError::DuplicateGroup(group.name));
        }
        self.group_order.push(group.name.clone());
        self.groups.insert(group.name.clone(), group);
        Ok(())
    }

    /// Get a group by name.
    #[must_use]
    pub fn get_group(&self, name: &str) -> Option<&AngleGroup> {
        self.groups.get(name)
    }

    /// Get a mutable reference to a group.
    #[must_use]
    pub fn get_group_mut(&mut self, name: &str) -> Option<&mut AngleGroup> {
        self.groups.get_mut(name)
    }

    /// Number of angle groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Total number of angles in a specific group.
    #[must_use]
    pub fn total_angles(&self, group_name: &str) -> Option<usize> {
        self.groups.get(group_name).map(|g| g.angle_count())
    }

    /// Set the active angle for a group.
    ///
    /// # Errors
    ///
    /// Returns an error if the group is not found or the angle index is
    /// out of range.
    pub fn set_active_angle(
        &mut self,
        group_name: &str,
        angle_index: usize,
    ) -> Result<(), MultiAngleError> {
        let group = self
            .groups
            .get_mut(group_name)
            .ok_or_else(|| MultiAngleError::GroupNotFound(group_name.to_string()))?;
        group.set_active(angle_index)
    }

    /// Get the active track index for a group.
    #[must_use]
    pub fn active_track(&self, group_name: &str) -> Option<usize> {
        self.groups
            .get(group_name)
            .and_then(|g| g.active_track_index())
    }

    /// Schedule an angle switch at a given point.
    pub fn schedule_switch(&mut self, point: AngleSwitchPoint) {
        self.switch_points.push(point);
        self.switch_points.sort_by_key(|p| p.time_ticks);
    }

    /// Get all scheduled switch points.
    #[must_use]
    pub fn switch_points(&self) -> &[AngleSwitchPoint] {
        &self.switch_points
    }

    /// Get the next switch point at or after `time_ticks`.
    #[must_use]
    pub fn next_switch_after(&self, time_ticks: u64) -> Option<&AngleSwitchPoint> {
        self.switch_points
            .iter()
            .find(|p| p.time_ticks >= time_ticks)
    }

    /// List group names in insertion order.
    #[must_use]
    pub fn group_names(&self) -> &[String] {
        &self.group_order
    }

    /// Collect all track indices across all groups (for demuxing all angles).
    #[must_use]
    pub fn all_track_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .groups
            .values()
            .flat_map(|g| g.all_track_indices())
            .collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    /// Find which group (if any) contains a given track index.
    #[must_use]
    pub fn group_for_track(&self, track_index: usize) -> Option<(&str, usize)> {
        for group in self.groups.values() {
            if let Some((angle_idx, _)) = group.find_by_track(track_index) {
                return Some((&group.name, angle_idx));
            }
        }
        None
    }

    /// Clear all scheduled switch points.
    pub fn clear_switch_points(&mut self) {
        self.switch_points.clear();
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_group() -> AngleGroup {
        AngleGroup::new("scene1")
            .with_angle(AngleDescriptor::new(0, "front").as_default())
            .with_angle(AngleDescriptor::new(1, "side"))
            .with_angle(AngleDescriptor::new(2, "overhead"))
    }

    #[test]
    fn test_angle_descriptor_creation() {
        let desc = AngleDescriptor::new(0, "Camera A")
            .with_language("eng")
            .as_default()
            .with_metadata("camera_model", "Sony A7");

        assert_eq!(desc.track_index, 0);
        assert_eq!(desc.label, "Camera A");
        assert_eq!(desc.language.as_deref(), Some("eng"));
        assert!(desc.is_default);
        assert_eq!(
            desc.metadata.get("camera_model").map(|s| s.as_str()),
            Some("Sony A7")
        );
    }

    #[test]
    fn test_angle_group_basics() {
        let group = sample_group();
        assert_eq!(group.angle_count(), 3);
        assert_eq!(group.active_index(), 0);
        assert_eq!(group.active_track_index(), Some(0));
        assert_eq!(group.default_angle_index(), Some(0));
    }

    #[test]
    fn test_angle_group_set_active() {
        let mut group = sample_group();
        assert!(group.set_active(1).is_ok());
        assert_eq!(group.active_index(), 1);
        assert_eq!(group.active_track_index(), Some(1));

        assert!(group.set_active(5).is_err());
    }

    #[test]
    fn test_angle_group_find_by_label() {
        let group = sample_group();
        let (idx, desc) = group.find_by_label("SIDE").expect("should find");
        assert_eq!(idx, 1);
        assert_eq!(desc.track_index, 1);
    }

    #[test]
    fn test_angle_group_find_by_track() {
        let group = sample_group();
        let (idx, desc) = group.find_by_track(2).expect("should find");
        assert_eq!(idx, 2);
        assert_eq!(desc.label, "overhead");
        assert!(group.find_by_track(99).is_none());
    }

    #[test]
    fn test_angle_group_all_tracks() {
        let group = sample_group();
        assert_eq!(group.all_track_indices(), vec![0, 1, 2]);
    }

    #[test]
    fn test_angle_group_time_range() {
        let group = AngleGroup::new("timed")
            .with_angle(AngleDescriptor::new(0, "a"))
            .with_time_range(1000, 5000);
        assert_eq!(group.time_range, Some((1000, 5000)));
    }

    #[test]
    fn test_angle_manager_add_and_query() {
        let mut manager = AngleManager::new();
        assert!(manager.add_group(sample_group()).is_ok());
        assert_eq!(manager.group_count(), 1);
        assert_eq!(manager.total_angles("scene1"), Some(3));
        assert_eq!(manager.active_track("scene1"), Some(0));

        // Duplicate should fail
        assert!(manager.add_group(AngleGroup::new("scene1")).is_err());
    }

    #[test]
    fn test_angle_manager_set_active() {
        let mut manager = AngleManager::new();
        manager.add_group(sample_group()).expect("should succeed");

        assert!(manager.set_active_angle("scene1", 2).is_ok());
        assert_eq!(manager.active_track("scene1"), Some(2));

        assert!(manager.set_active_angle("nonexistent", 0).is_err());
    }

    #[test]
    fn test_angle_manager_switch_points() {
        let mut manager = AngleManager::new();
        manager.add_group(sample_group()).expect("should succeed");

        manager.schedule_switch(AngleSwitchPoint {
            time_ticks: 5000,
            sample_number: 150,
            from_angle: 0,
            to_angle: 1,
        });
        manager.schedule_switch(AngleSwitchPoint {
            time_ticks: 2000,
            sample_number: 60,
            from_angle: 0,
            to_angle: 2,
        });

        // Should be sorted by time
        assert_eq!(manager.switch_points().len(), 2);
        assert_eq!(manager.switch_points()[0].time_ticks, 2000);
        assert_eq!(manager.switch_points()[1].time_ticks, 5000);

        let next = manager.next_switch_after(3000);
        assert!(next.is_some());
        assert_eq!(next.expect("should exist").time_ticks, 5000);

        manager.clear_switch_points();
        assert!(manager.switch_points().is_empty());
    }

    #[test]
    fn test_angle_manager_all_tracks() {
        let mut manager = AngleManager::new();
        manager.add_group(sample_group()).expect("should succeed");
        manager
            .add_group(
                AngleGroup::new("scene2")
                    .with_angle(AngleDescriptor::new(3, "wide"))
                    .with_angle(AngleDescriptor::new(1, "side_shared")),
            )
            .expect("should succeed");

        let tracks = manager.all_track_indices();
        // Deduplicated and sorted: 0, 1, 2, 3
        assert_eq!(tracks, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_angle_manager_group_for_track() {
        let mut manager = AngleManager::new();
        manager.add_group(sample_group()).expect("should succeed");

        let (group_name, angle_idx) = manager.group_for_track(2).expect("should find");
        assert_eq!(group_name, "scene1");
        assert_eq!(angle_idx, 2);

        assert!(manager.group_for_track(99).is_none());
    }

    #[test]
    fn test_angle_manager_group_names_order() {
        let mut manager = AngleManager::new();
        manager
            .add_group(AngleGroup::new("beta").with_angle(AngleDescriptor::new(0, "a")))
            .expect("ok");
        manager
            .add_group(AngleGroup::new("alpha").with_angle(AngleDescriptor::new(1, "b")))
            .expect("ok");

        assert_eq!(manager.group_names(), &["beta", "alpha"]);
    }

    #[test]
    fn test_angle_group_empty_default() {
        let group = AngleGroup::new("empty");
        assert_eq!(group.default_angle_index(), None);
        assert!(group.active_angle().is_none());
    }
}
