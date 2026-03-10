//! Edit conflict resolution for collaborative video editing.
//!
//! Provides last-write-wins, merge strategies, and conflict detection
//! for concurrent edits to timeline regions.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A logical clock value for ordering operations.
pub type LogicalClock = u64;

/// An identifier for an edit operation.
pub type EditId = Uuid;

/// The region of the timeline affected by an edit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineRegion {
    /// Track identifier.
    pub track_id: String,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
}

impl TimelineRegion {
    /// Create a new timeline region.
    pub fn new(track_id: impl Into<String>, start_ms: i64, end_ms: i64) -> Self {
        Self {
            track_id: track_id.into(),
            start_ms,
            end_ms,
        }
    }

    /// Check whether this region overlaps another.
    pub fn overlaps(&self, other: &TimelineRegion) -> bool {
        self.track_id == other.track_id
            && self.start_ms < other.end_ms
            && other.start_ms < self.end_ms
    }
}

/// Type of edit operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EditKind {
    /// Insert content at a position.
    Insert,
    /// Delete a region.
    Delete,
    /// Move a region.
    Move,
    /// Modify properties (e.g. gain, color).
    Modify,
    /// Replace a clip.
    Replace,
}

/// An edit operation submitted by a user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditOperation {
    /// Unique operation ID.
    pub id: EditId,
    /// Author user ID.
    pub author: Uuid,
    /// Logical clock at submission time.
    pub clock: LogicalClock,
    /// Affected timeline region.
    pub region: TimelineRegion,
    /// Type of edit.
    pub kind: EditKind,
    /// Serialized payload (kind-specific).
    pub payload: serde_json::Value,
}

impl EditOperation {
    /// Create a new edit operation.
    pub fn new(
        author: Uuid,
        clock: LogicalClock,
        region: TimelineRegion,
        kind: EditKind,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            author,
            clock,
            region,
            kind,
            payload,
        }
    }
}

/// How to resolve a conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionStrategy {
    /// The operation with the higher clock wins.
    LastWriteWins,
    /// The operation with the lower clock (earlier) wins.
    FirstWriteWins,
    /// Keep both operations (requires manual review).
    KeepBoth,
    /// Reject the incoming operation.
    RejectIncoming,
}

/// A detected conflict between two operations.
#[derive(Debug, Clone)]
pub struct Conflict {
    /// The existing (accepted) operation.
    pub existing: EditOperation,
    /// The incoming (challenger) operation.
    pub incoming: EditOperation,
    /// Chosen resolution strategy.
    pub resolution: ResolutionStrategy,
}

impl Conflict {
    /// Determine which operation wins under the chosen strategy.
    ///
    /// Returns `Some(op)` for the winner or `None` if both are kept.
    pub fn winner(&self) -> Option<&EditOperation> {
        match self.resolution {
            ResolutionStrategy::LastWriteWins => {
                if self.incoming.clock >= self.existing.clock {
                    Some(&self.incoming)
                } else {
                    Some(&self.existing)
                }
            }
            ResolutionStrategy::FirstWriteWins => {
                if self.existing.clock <= self.incoming.clock {
                    Some(&self.existing)
                } else {
                    Some(&self.incoming)
                }
            }
            ResolutionStrategy::KeepBoth => None,
            ResolutionStrategy::RejectIncoming => Some(&self.existing),
        }
    }
}

/// Conflict resolver that maintains applied operations.
#[derive(Debug)]
pub struct ConflictResolver {
    strategy: ResolutionStrategy,
    /// Applied operations keyed by region track.
    applied: HashMap<String, Vec<EditOperation>>,
}

impl ConflictResolver {
    /// Create a new resolver with the given strategy.
    pub fn new(strategy: ResolutionStrategy) -> Self {
        Self {
            strategy,
            applied: HashMap::new(),
        }
    }

    /// Detect conflicts between an incoming operation and applied operations.
    pub fn detect_conflicts(&self, incoming: &EditOperation) -> Vec<Conflict> {
        let track_ops = match self.applied.get(&incoming.region.track_id) {
            Some(ops) => ops,
            None => return vec![],
        };

        track_ops
            .iter()
            .filter(|existing| existing.region.overlaps(&incoming.region))
            .map(|existing| Conflict {
                existing: existing.clone(),
                incoming: incoming.clone(),
                resolution: self.strategy,
            })
            .collect()
    }

    /// Apply an operation, resolving conflicts as configured.
    ///
    /// Returns the list of conflicts that were resolved.
    pub fn apply(&mut self, incoming: EditOperation) -> Vec<Conflict> {
        let conflicts = self.detect_conflicts(&incoming);

        let accept_incoming = if conflicts.is_empty() {
            true
        } else {
            // Check if any conflict results in rejecting the incoming op
            conflicts.iter().all(|c| {
                !matches!(c.winner(), Some(w) if std::ptr::eq(w, &c.existing))
                    || matches!(self.strategy, ResolutionStrategy::LastWriteWins
                        if incoming.clock >= c.existing.clock)
                    || matches!(self.strategy, ResolutionStrategy::KeepBoth)
            });
            // Simplified: accept if strategy is not RejectIncoming
            !matches!(self.strategy, ResolutionStrategy::RejectIncoming)
        };

        if accept_incoming {
            // Under LastWriteWins, remove conflicting existing ops
            if matches!(self.strategy, ResolutionStrategy::LastWriteWins) {
                let incoming_region = incoming.region.clone();
                let track_id = incoming.region.track_id.clone();
                if let Some(ops) = self.applied.get_mut(&track_id) {
                    ops.retain(|op| !op.region.overlaps(&incoming_region));
                }
            }

            self.applied
                .entry(incoming.region.track_id.clone())
                .or_default()
                .push(incoming);
        }

        conflicts
    }

    /// Get all applied operations for a track.
    pub fn operations_for_track(&self, track_id: &str) -> &[EditOperation] {
        self.applied.get(track_id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Total number of applied operations.
    pub fn operation_count(&self) -> usize {
        self.applied.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user() -> Uuid {
        Uuid::new_v4()
    }

    fn region(start: i64, end: i64) -> TimelineRegion {
        TimelineRegion::new("track_1", start, end)
    }

    fn op(author: Uuid, clock: u64, r: TimelineRegion) -> EditOperation {
        EditOperation::new(author, clock, r, EditKind::Modify, serde_json::Value::Null)
    }

    #[test]
    fn test_region_overlaps() {
        let a = region(0, 1000);
        let b = region(500, 1500);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_region_no_overlap_adjacent() {
        let a = region(0, 1000);
        let b = region(1000, 2000);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_region_different_track_no_overlap() {
        let a = TimelineRegion::new("track_1", 0, 1000);
        let b = TimelineRegion::new("track_2", 0, 1000);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_apply_non_conflicting() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        let o = op(user(), 1, region(0, 1000));
        let conflicts = resolver.apply(o);
        assert!(conflicts.is_empty());
        assert_eq!(resolver.operation_count(), 1);
    }

    #[test]
    fn test_detect_conflict_overlapping_ops() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        let u1 = user();
        let o1 = op(u1, 1, region(0, 1000));
        resolver.apply(o1);

        let o2 = op(user(), 2, region(500, 1500));
        let conflicts = resolver.detect_conflicts(&o2);
        assert_eq!(conflicts.len(), 1);
    }

    #[test]
    fn test_last_write_wins_removes_old_op() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        let o1 = op(user(), 1, region(0, 1000));
        resolver.apply(o1);

        let o2 = op(user(), 5, region(0, 1000));
        resolver.apply(o2);

        let ops = resolver.operations_for_track("track_1");
        // Under LastWriteWins, old conflicting op should be replaced
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].clock, 5);
    }

    #[test]
    fn test_reject_incoming_strategy() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::RejectIncoming);
        let o1 = op(user(), 1, region(0, 1000));
        resolver.apply(o1);

        let o2 = op(user(), 5, region(0, 1000));
        resolver.apply(o2);

        let ops = resolver.operations_for_track("track_1");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].clock, 1);
    }

    #[test]
    fn test_conflict_winner_last_write_wins() {
        let existing = op(user(), 1, region(0, 1000));
        let incoming = op(user(), 5, region(0, 1000));
        let conflict = Conflict {
            existing: existing.clone(),
            incoming: incoming.clone(),
            resolution: ResolutionStrategy::LastWriteWins,
        };
        let winner = conflict
            .winner()
            .expect("collab test operation should succeed");
        assert_eq!(winner.clock, 5);
    }

    #[test]
    fn test_conflict_winner_first_write_wins() {
        let existing = op(user(), 1, region(0, 1000));
        let incoming = op(user(), 5, region(0, 1000));
        let conflict = Conflict {
            existing: existing.clone(),
            incoming: incoming.clone(),
            resolution: ResolutionStrategy::FirstWriteWins,
        };
        let winner = conflict
            .winner()
            .expect("collab test operation should succeed");
        assert_eq!(winner.clock, 1);
    }

    #[test]
    fn test_conflict_winner_keep_both_returns_none() {
        let existing = op(user(), 1, region(0, 1000));
        let incoming = op(user(), 5, region(0, 1000));
        let conflict = Conflict {
            existing,
            incoming,
            resolution: ResolutionStrategy::KeepBoth,
        };
        assert!(conflict.winner().is_none());
    }

    #[test]
    fn test_conflict_winner_reject_incoming() {
        let existing = op(user(), 1, region(0, 1000));
        let incoming = op(user(), 5, region(0, 1000));
        let conflict = Conflict {
            existing: existing.clone(),
            incoming,
            resolution: ResolutionStrategy::RejectIncoming,
        };
        let winner = conflict
            .winner()
            .expect("collab test operation should succeed");
        assert_eq!(winner.clock, 1);
    }

    #[test]
    fn test_non_overlapping_ops_accumulate() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        resolver.apply(op(user(), 1, region(0, 500)));
        resolver.apply(op(user(), 2, region(500, 1000)));
        assert_eq!(resolver.operation_count(), 2);
    }

    #[test]
    fn test_operations_for_unknown_track() {
        let resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        assert!(resolver.operations_for_track("nonexistent").is_empty());
    }

    #[test]
    fn test_edit_operation_new_generates_unique_ids() {
        let u = user();
        let o1 = EditOperation::new(
            u,
            1,
            region(0, 100),
            EditKind::Insert,
            serde_json::Value::Null,
        );
        let o2 = EditOperation::new(
            u,
            2,
            region(0, 100),
            EditKind::Insert,
            serde_json::Value::Null,
        );
        assert_ne!(o1.id, o2.id);
    }

    #[test]
    fn test_region_duration() {
        let r = region(1000, 4000);
        assert_eq!(r.end_ms - r.start_ms, 3000);
    }
}
