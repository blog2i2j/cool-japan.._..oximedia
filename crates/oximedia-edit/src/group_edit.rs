//! Group editing operations for multi-clip batch modifications.
//!
//! Provides the ability to create, manipulate, and apply editing operations
//! to groups of clips as a single atomic unit, enabling efficient batch
//! edits, synchronised moves, and linked transformations.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Group identity and membership
// ---------------------------------------------------------------------------

/// Unique identifier for a clip group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(pub u64);

impl fmt::Display for GroupId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "group-{}", self.0)
    }
}

/// Defines how clips within a group respond when one member is edited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupBehavior {
    /// All clips move together rigidly (translate in lock-step).
    Locked,
    /// Clips maintain relative timing but may be individually trimmed.
    Relative,
    /// Clips are loosely associated; edits only propagate on explicit request.
    Loose,
}

impl fmt::Display for GroupBehavior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Locked => write!(f, "locked"),
            Self::Relative => write!(f, "relative"),
            Self::Loose => write!(f, "loose"),
        }
    }
}

/// A group of clips that can be edited as a unit.
#[derive(Debug, Clone)]
pub struct EditGroup {
    /// Unique identifier for this group.
    pub id: GroupId,
    /// Human-readable name for the group.
    pub name: String,
    /// Clip IDs that belong to this group.
    pub members: HashSet<u64>,
    /// How edits propagate within the group.
    pub behavior: GroupBehavior,
    /// Whether the group is currently locked against edits.
    pub locked: bool,
    /// Optional color label for UI display (RGBA).
    pub color: Option<u32>,
}

impl EditGroup {
    /// Create a new group with the given id and name.
    pub fn new(id: GroupId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            members: HashSet::new(),
            behavior: GroupBehavior::Locked,
            locked: false,
            color: None,
        }
    }

    /// Builder: set the group behavior.
    #[must_use]
    pub fn with_behavior(mut self, behavior: GroupBehavior) -> Self {
        self.behavior = behavior;
        self
    }

    /// Builder: set color label.
    #[must_use]
    pub fn with_color(mut self, rgba: u32) -> Self {
        self.color = Some(rgba);
        self
    }

    /// Add a clip ID to this group.
    pub fn add_member(&mut self, clip_id: u64) -> bool {
        self.members.insert(clip_id)
    }

    /// Remove a clip ID from this group.
    pub fn remove_member(&mut self, clip_id: u64) -> bool {
        self.members.remove(&clip_id)
    }

    /// Returns `true` if the clip is in this group.
    #[must_use]
    pub fn contains(&self, clip_id: u64) -> bool {
        self.members.contains(&clip_id)
    }

    /// Returns the number of clips in the group.
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Returns `true` if the group has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Batch operation
// ---------------------------------------------------------------------------

/// An operation to apply to every clip in a group.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchOp {
    /// Move every clip by a signed offset (in timebase units).
    MoveBy(i64),
    /// Scale the duration of every clip by a factor.
    ScaleDuration(f64),
    /// Set the opacity of every clip (0.0 = transparent, 1.0 = opaque).
    SetOpacity(f64),
    /// Trim the in-point of every clip by a signed offset.
    TrimInBy(i64),
    /// Trim the out-point of every clip by a signed offset.
    TrimOutBy(i64),
    /// Mute or unmute every clip.
    SetMute(bool),
    /// Delete every clip in the group.
    Delete,
}

impl fmt::Display for BatchOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MoveBy(d) => write!(f, "move by {d}"),
            Self::ScaleDuration(s) => write!(f, "scale duration x{s}"),
            Self::SetOpacity(o) => write!(f, "opacity {o}"),
            Self::TrimInBy(d) => write!(f, "trim in by {d}"),
            Self::TrimOutBy(d) => write!(f, "trim out by {d}"),
            Self::SetMute(m) => write!(f, "mute={m}"),
            Self::Delete => write!(f, "delete"),
        }
    }
}

/// Result of applying a batch operation.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Number of clips affected.
    pub affected: usize,
    /// Number of clips that were skipped (e.g. locked).
    pub skipped: usize,
    /// Errors keyed by clip ID.
    pub errors: HashMap<u64, String>,
}

impl BatchResult {
    /// Create a new, empty result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            affected: 0,
            skipped: 0,
            errors: HashMap::new(),
        }
    }

    /// Returns `true` if no errors occurred.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns `true` if there were any errors.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

impl Default for BatchResult {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Group registry
// ---------------------------------------------------------------------------

/// Counter for assigning unique group IDs.
fn next_group_id() -> GroupId {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(1);
    GroupId(CTR.fetch_add(1, Ordering::Relaxed))
}

/// Manages all clip groups in a project.
#[derive(Debug, Clone)]
pub struct GroupEditRegistry {
    /// All known groups, keyed by their group-ID.
    groups: HashMap<GroupId, EditGroup>,
    /// Reverse index: clip-ID -> group-IDs it belongs to.
    clip_to_groups: HashMap<u64, HashSet<GroupId>>,
}

impl GroupEditRegistry {
    /// Create a new, empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            clip_to_groups: HashMap::new(),
        }
    }

    /// Create a new group and return its ID.
    pub fn create_group(&mut self, name: impl Into<String>) -> GroupId {
        let id = next_group_id();
        let group = EditGroup::new(id, name);
        self.groups.insert(id, group);
        id
    }

    /// Delete a group by its ID.
    pub fn delete_group(&mut self, id: GroupId) -> Option<EditGroup> {
        if let Some(group) = self.groups.remove(&id) {
            for &clip_id in &group.members {
                if let Some(set) = self.clip_to_groups.get_mut(&clip_id) {
                    set.remove(&id);
                }
            }
            Some(group)
        } else {
            None
        }
    }

    /// Get an immutable reference to a group.
    #[must_use]
    pub fn get(&self, id: GroupId) -> Option<&EditGroup> {
        self.groups.get(&id)
    }

    /// Get a mutable reference to a group.
    pub fn get_mut(&mut self, id: GroupId) -> Option<&mut EditGroup> {
        self.groups.get_mut(&id)
    }

    /// Add a clip to a group. Returns `true` if the clip was newly added.
    pub fn add_clip_to_group(&mut self, group_id: GroupId, clip_id: u64) -> bool {
        let added = self
            .groups
            .get_mut(&group_id)
            .is_some_and(|g| g.add_member(clip_id));
        if added {
            self.clip_to_groups
                .entry(clip_id)
                .or_default()
                .insert(group_id);
        }
        added
    }

    /// Remove a clip from a group. Returns `true` if the clip was removed.
    pub fn remove_clip_from_group(&mut self, group_id: GroupId, clip_id: u64) -> bool {
        let removed = self
            .groups
            .get_mut(&group_id)
            .is_some_and(|g| g.remove_member(clip_id));
        if removed {
            if let Some(set) = self.clip_to_groups.get_mut(&clip_id) {
                set.remove(&group_id);
            }
        }
        removed
    }

    /// Find all groups that a clip belongs to.
    pub fn groups_for_clip(&self, clip_id: u64) -> Vec<GroupId> {
        self.clip_to_groups
            .get(&clip_id)
            .map_or_else(Vec::new, |set| set.iter().copied().collect())
    }

    /// Returns the total number of groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Returns `true` if there are no groups.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Apply a batch operation to every member of a group.
    ///
    /// This is a dry-run: it returns a `BatchResult` describing what would
    /// happen, without actually modifying clip data (the caller is
    /// responsible for applying the changes to the timeline).
    #[must_use]
    pub fn plan_batch_op(&self, group_id: GroupId, _op: &BatchOp) -> BatchResult {
        let mut result = BatchResult::new();
        let group = match self.groups.get(&group_id) {
            Some(g) => g,
            None => return result,
        };
        if group.locked {
            result.skipped = group.member_count();
            return result;
        }
        result.affected = group.member_count();
        result
    }
}

impl Default for GroupEditRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_id_display() {
        assert_eq!(GroupId(42).to_string(), "group-42");
    }

    #[test]
    fn test_group_behavior_display() {
        assert_eq!(GroupBehavior::Locked.to_string(), "locked");
        assert_eq!(GroupBehavior::Relative.to_string(), "relative");
        assert_eq!(GroupBehavior::Loose.to_string(), "loose");
    }

    #[test]
    fn test_edit_group_new() {
        let g = EditGroup::new(GroupId(1), "My Group");
        assert_eq!(g.name, "My Group");
        assert!(g.is_empty());
        assert_eq!(g.behavior, GroupBehavior::Locked);
    }

    #[test]
    fn test_edit_group_add_remove_member() {
        let mut g = EditGroup::new(GroupId(1), "g");
        assert!(g.add_member(10));
        assert!(!g.add_member(10)); // duplicate
        assert_eq!(g.member_count(), 1);
        assert!(g.contains(10));
        assert!(g.remove_member(10));
        assert!(!g.remove_member(10)); // already removed
        assert!(g.is_empty());
    }

    #[test]
    fn test_edit_group_builders() {
        let g = EditGroup::new(GroupId(1), "g")
            .with_behavior(GroupBehavior::Loose)
            .with_color(0xFF0000FF);
        assert_eq!(g.behavior, GroupBehavior::Loose);
        assert_eq!(g.color, Some(0xFF0000FF));
    }

    #[test]
    fn test_batch_op_display() {
        assert_eq!(BatchOp::MoveBy(-100).to_string(), "move by -100");
        assert_eq!(BatchOp::Delete.to_string(), "delete");
        assert_eq!(BatchOp::SetMute(true).to_string(), "mute=true");
    }

    #[test]
    fn test_batch_result_default() {
        let r = BatchResult::default();
        assert!(r.is_ok());
        assert!(!r.has_errors());
        assert_eq!(r.affected, 0);
    }

    #[test]
    fn test_batch_result_with_errors() {
        let mut r = BatchResult::new();
        r.errors.insert(1, "locked".to_string());
        assert!(r.has_errors());
        assert!(!r.is_ok());
    }

    #[test]
    fn test_registry_create_delete() {
        let mut reg = GroupEditRegistry::new();
        let gid = reg.create_group("Test");
        assert_eq!(reg.group_count(), 1);
        assert!(reg.delete_group(gid).is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_add_clip_to_group() {
        let mut reg = GroupEditRegistry::new();
        let gid = reg.create_group("G1");
        assert!(reg.add_clip_to_group(gid, 100));
        assert!(reg.get(gid).expect("get should succeed").contains(100));
    }

    #[test]
    fn test_registry_remove_clip_from_group() {
        let mut reg = GroupEditRegistry::new();
        let gid = reg.create_group("G1");
        reg.add_clip_to_group(gid, 100);
        assert!(reg.remove_clip_from_group(gid, 100));
        assert!(!reg.get(gid).expect("get should succeed").contains(100));
    }

    #[test]
    fn test_registry_groups_for_clip() {
        let mut reg = GroupEditRegistry::new();
        let g1 = reg.create_group("A");
        let g2 = reg.create_group("B");
        reg.add_clip_to_group(g1, 5);
        reg.add_clip_to_group(g2, 5);
        let groups = reg.groups_for_clip(5);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_registry_plan_batch_op() {
        let mut reg = GroupEditRegistry::new();
        let gid = reg.create_group("G");
        reg.add_clip_to_group(gid, 1);
        reg.add_clip_to_group(gid, 2);
        let result = reg.plan_batch_op(gid, &BatchOp::MoveBy(50));
        assert_eq!(result.affected, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_registry_plan_batch_op_locked() {
        let mut reg = GroupEditRegistry::new();
        let gid = reg.create_group("Locked");
        reg.add_clip_to_group(gid, 1);
        reg.get_mut(gid).expect("get_mut should succeed").locked = true;
        let result = reg.plan_batch_op(gid, &BatchOp::Delete);
        assert_eq!(result.affected, 0);
        assert_eq!(result.skipped, 1);
    }

    #[test]
    fn test_registry_default() {
        let reg = GroupEditRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_delete_group_cleans_reverse_index() {
        let mut reg = GroupEditRegistry::new();
        let gid = reg.create_group("X");
        reg.add_clip_to_group(gid, 42);
        reg.delete_group(gid);
        assert!(reg.groups_for_clip(42).is_empty());
    }
}
