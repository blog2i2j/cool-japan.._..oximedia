//! Camera angle group management for multi-camera production.
//!
//! This module provides structures for grouping camera angles, tracking
//! coverage maps across groups, and maintaining switching history at the
//! group level.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── AngleGroupId ──────────────────────────────────────────────────────────────

/// Opaque identifier for an angle group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AngleGroupId(u32);

impl AngleGroupId {
    /// Create a new `AngleGroupId` from a raw value.
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Return the inner numeric value.
    #[must_use]
    pub fn value(self) -> u32 {
        self.0
    }
}

// ── AngleGroupKind ────────────────────────────────────────────────────────────

/// Semantic category of an angle group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleGroupKind {
    /// Wide establishing shots.
    Wide,
    /// Mid-range shots.
    Mid,
    /// Close-up shots.
    CloseUp,
    /// Cutaway / B-roll angles.
    Cutaway,
    /// Overhead / bird's-eye angles.
    Overhead,
    /// User-defined category.
    Custom,
}

// ── AngleGroup ────────────────────────────────────────────────────────────────

/// A named group of camera angles that can be managed as a unit.
#[derive(Debug, Clone)]
pub struct AngleGroup {
    /// Unique identifier.
    pub id: AngleGroupId,
    /// Human-readable name.
    pub name: String,
    /// Semantic category.
    pub kind: AngleGroupKind,
    /// Camera angle IDs belonging to this group (angle IDs from the rest of the crate).
    pub angle_ids: Vec<u32>,
    /// Optional priority when the auto-switcher considers this group.
    pub priority: u8,
}

impl AngleGroup {
    /// Create a new, empty `AngleGroup`.
    pub fn new(id: u32, name: impl Into<String>, kind: AngleGroupKind) -> Self {
        Self {
            id: AngleGroupId::new(id),
            name: name.into(),
            kind,
            angle_ids: Vec::new(),
            priority: 128,
        }
    }

    /// Add a camera angle to this group.
    pub fn add_angle(&mut self, angle_id: u32) {
        if !self.angle_ids.contains(&angle_id) {
            self.angle_ids.push(angle_id);
        }
    }

    /// Remove a camera angle from this group.
    pub fn remove_angle(&mut self, angle_id: u32) {
        self.angle_ids.retain(|&id| id != angle_id);
    }

    /// Return `true` if this group contains no angles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.angle_ids.is_empty()
    }

    /// Return the number of angles in this group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.angle_ids.len()
    }
}

// ── CoverageMap ───────────────────────────────────────────────────────────────

/// Per-angle coverage score (0.0 – 1.0).
#[derive(Debug, Clone, Copy)]
pub struct CoverageScore {
    /// Camera angle ID.
    pub angle_id: u32,
    /// Coverage score in the range \[0.0, 1.0\].
    pub score: f32,
}

impl CoverageScore {
    /// Create a new `CoverageScore`, clamping the score to \[0.0, 1.0\].
    #[must_use]
    pub fn new(angle_id: u32, score: f32) -> Self {
        Self {
            angle_id,
            score: score.clamp(0.0, 1.0),
        }
    }
}

/// Coverage map across all groups for a single frame instant.
#[derive(Debug, Default)]
pub struct CoverageMap {
    /// Map from `AngleGroupId` to per-angle coverage scores within that group.
    pub entries: HashMap<AngleGroupId, Vec<CoverageScore>>,
}

impl CoverageMap {
    /// Create an empty coverage map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace coverage scores for a group.
    pub fn set_group_coverage(&mut self, group_id: AngleGroupId, scores: Vec<CoverageScore>) {
        self.entries.insert(group_id, scores);
    }

    /// Return the average coverage score for a group.
    /// Returns `None` if the group has no entries or no angles.
    #[must_use]
    pub fn average_coverage(&self, group_id: AngleGroupId) -> Option<f32> {
        let scores = self.entries.get(&group_id)?;
        if scores.is_empty() {
            return None;
        }
        let sum: f32 = scores.iter().map(|s| s.score).sum();
        Some(sum / scores.len() as f32)
    }

    /// Return the ID of the angle with the highest coverage score across all groups.
    #[must_use]
    pub fn best_angle(&self) -> Option<u32> {
        self.entries
            .values()
            .flatten()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.angle_id)
    }
}

// ── GroupSwitchEvent ──────────────────────────────────────────────────────────

/// Record of a group-level switch during production.
#[derive(Debug, Clone)]
pub struct GroupSwitchEvent {
    /// Previous group.
    pub from_group: AngleGroupId,
    /// New active group.
    pub to_group: AngleGroupId,
    /// Frame index at which the switch happened.
    pub frame_idx: u64,
    /// Whether the switch was triggered automatically.
    pub automatic: bool,
}

impl GroupSwitchEvent {
    /// Create a new `GroupSwitchEvent`.
    #[must_use]
    pub fn new(from_group: u32, to_group: u32, frame_idx: u64, automatic: bool) -> Self {
        Self {
            from_group: AngleGroupId::new(from_group),
            to_group: AngleGroupId::new(to_group),
            frame_idx,
            automatic,
        }
    }
}

// ── GroupSwitchHistory ────────────────────────────────────────────────────────

/// Chronological log of group-level switch events.
#[derive(Debug, Default)]
pub struct GroupSwitchHistory {
    events: Vec<GroupSwitchEvent>,
}

impl GroupSwitchHistory {
    /// Create an empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new switch event.
    pub fn record(&mut self, event: GroupSwitchEvent) {
        self.events.push(event);
    }

    /// Return a reference to all recorded events.
    #[must_use]
    pub fn events(&self) -> &[GroupSwitchEvent] {
        &self.events
    }

    /// Return the number of recorded events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return `true` if no events have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Count how many times each group was switched *to*.
    #[must_use]
    pub fn switch_counts(&self) -> HashMap<AngleGroupId, u32> {
        let mut counts: HashMap<AngleGroupId, u32> = HashMap::new();
        for ev in &self.events {
            *counts.entry(ev.to_group).or_insert(0) += 1;
        }
        counts
    }

    /// Return the most-visited group, or `None` if the history is empty.
    #[must_use]
    pub fn most_visited(&self) -> Option<AngleGroupId> {
        let counts = self.switch_counts();
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(id, _)| id)
    }
}

// ── AngleGroupRegistry ────────────────────────────────────────────────────────

/// Registry that holds all angle groups for a session.
#[derive(Debug, Default)]
pub struct AngleGroupRegistry {
    groups: Vec<AngleGroup>,
}

impl AngleGroupRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a group. Returns an error string if a group with the same ID already exists.
    pub fn register(&mut self, group: AngleGroup) -> Result<(), String> {
        if self.groups.iter().any(|g| g.id == group.id) {
            return Err(format!("Group id {} already registered", group.id.value()));
        }
        self.groups.push(group);
        Ok(())
    }

    /// Look up a group by ID.
    #[must_use]
    pub fn get(&self, id: AngleGroupId) -> Option<&AngleGroup> {
        self.groups.iter().find(|g| g.id == id)
    }

    /// Look up a group mutably by ID.
    pub fn get_mut(&mut self, id: AngleGroupId) -> Option<&mut AngleGroup> {
        self.groups.iter_mut().find(|g| g.id == id)
    }

    /// Return the number of registered groups.
    #[must_use]
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Return `true` if no groups are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Return all groups sorted by priority (highest priority first).
    #[must_use]
    pub fn by_priority(&self) -> Vec<&AngleGroup> {
        let mut sorted: Vec<&AngleGroup> = self.groups.iter().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
        sorted
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_group_id_value() {
        let id = AngleGroupId::new(42);
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_angle_group_add_angle() {
        let mut group = AngleGroup::new(1, "Wide", AngleGroupKind::Wide);
        group.add_angle(10);
        group.add_angle(11);
        assert_eq!(group.len(), 2);
        assert!(!group.is_empty());
    }

    #[test]
    fn test_angle_group_no_duplicate_angles() {
        let mut group = AngleGroup::new(1, "Wide", AngleGroupKind::Wide);
        group.add_angle(10);
        group.add_angle(10);
        assert_eq!(group.len(), 1);
    }

    #[test]
    fn test_angle_group_remove_angle() {
        let mut group = AngleGroup::new(1, "Mid", AngleGroupKind::Mid);
        group.add_angle(5);
        group.add_angle(6);
        group.remove_angle(5);
        assert_eq!(group.len(), 1);
        assert!(!group.angle_ids.contains(&5));
    }

    #[test]
    fn test_coverage_score_clamped() {
        let high = CoverageScore::new(1, 1.5);
        assert_eq!(high.score, 1.0);
        let low = CoverageScore::new(2, -0.5);
        assert_eq!(low.score, 0.0);
    }

    #[test]
    fn test_coverage_map_average() {
        let gid = AngleGroupId::new(1);
        let mut map = CoverageMap::new();
        map.set_group_coverage(
            gid,
            vec![CoverageScore::new(0, 0.8), CoverageScore::new(1, 0.4)],
        );
        let avg = map
            .average_coverage(gid)
            .expect("multicam test operation should succeed");
        assert!((avg - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_coverage_map_average_missing_group() {
        let map = CoverageMap::new();
        assert!(map.average_coverage(AngleGroupId::new(99)).is_none());
    }

    #[test]
    fn test_coverage_map_best_angle() {
        let g1 = AngleGroupId::new(1);
        let g2 = AngleGroupId::new(2);
        let mut map = CoverageMap::new();
        map.set_group_coverage(g1, vec![CoverageScore::new(0, 0.3)]);
        map.set_group_coverage(g2, vec![CoverageScore::new(1, 0.9)]);
        assert_eq!(map.best_angle(), Some(1));
    }

    #[test]
    fn test_group_switch_history_record() {
        let mut history = GroupSwitchHistory::new();
        history.record(GroupSwitchEvent::new(0, 1, 100, false));
        history.record(GroupSwitchEvent::new(1, 2, 200, true));
        assert_eq!(history.len(), 2);
        assert!(!history.is_empty());
    }

    #[test]
    fn test_group_switch_history_empty() {
        let h = GroupSwitchHistory::new();
        assert!(h.is_empty());
    }

    #[test]
    fn test_group_switch_history_switch_counts() {
        let mut history = GroupSwitchHistory::new();
        history.record(GroupSwitchEvent::new(0, 1, 10, false));
        history.record(GroupSwitchEvent::new(1, 2, 20, true));
        history.record(GroupSwitchEvent::new(2, 1, 30, false));
        let counts = history.switch_counts();
        assert_eq!(
            *counts
                .get(&AngleGroupId::new(1))
                .expect("multicam test operation should succeed"),
            2
        );
        assert_eq!(
            *counts
                .get(&AngleGroupId::new(2))
                .expect("multicam test operation should succeed"),
            1
        );
    }

    #[test]
    fn test_group_switch_history_most_visited() {
        let mut history = GroupSwitchHistory::new();
        history.record(GroupSwitchEvent::new(0, 1, 0, false));
        history.record(GroupSwitchEvent::new(1, 2, 10, false));
        history.record(GroupSwitchEvent::new(2, 1, 20, false));
        assert_eq!(history.most_visited(), Some(AngleGroupId::new(1)));
    }

    #[test]
    fn test_angle_group_registry_register_and_get() {
        let mut reg = AngleGroupRegistry::new();
        let group = AngleGroup::new(1, "CloseUp", AngleGroupKind::CloseUp);
        reg.register(group)
            .expect("multicam test operation should succeed");
        let found = reg
            .get(AngleGroupId::new(1))
            .expect("multicam test operation should succeed");
        assert_eq!(found.name, "CloseUp");
    }

    #[test]
    fn test_angle_group_registry_duplicate_rejected() {
        let mut reg = AngleGroupRegistry::new();
        reg.register(AngleGroup::new(1, "A", AngleGroupKind::Wide))
            .expect("multicam test operation should succeed");
        let result = reg.register(AngleGroup::new(1, "B", AngleGroupKind::Mid));
        assert!(result.is_err());
    }

    #[test]
    fn test_angle_group_registry_by_priority() {
        let mut reg = AngleGroupRegistry::new();
        let mut g1 = AngleGroup::new(1, "Low", AngleGroupKind::Wide);
        g1.priority = 50;
        let mut g2 = AngleGroup::new(2, "High", AngleGroupKind::CloseUp);
        g2.priority = 200;
        reg.register(g1)
            .expect("multicam test operation should succeed");
        reg.register(g2)
            .expect("multicam test operation should succeed");
        let sorted = reg.by_priority();
        assert_eq!(sorted[0].name, "High");
        assert_eq!(sorted[1].name, "Low");
    }
}
