#![allow(dead_code)]
//! Review session history tracking and audit trail.
//!
//! This module records every action taken during a review session,
//! enabling full audit trails, undo/redo support, and compliance reporting.

use std::collections::VecDeque;
use std::fmt;

/// Maximum number of history entries retained by default.
const DEFAULT_MAX_ENTRIES: usize = 10_000;

/// The kind of action recorded in the history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistoryActionKind {
    /// A comment was added.
    CommentAdded,
    /// A comment was edited.
    CommentEdited,
    /// A comment was deleted.
    CommentDeleted,
    /// A comment was resolved.
    CommentResolved,
    /// An annotation/drawing was created.
    AnnotationCreated,
    /// An annotation/drawing was modified.
    AnnotationModified,
    /// An annotation/drawing was removed.
    AnnotationRemoved,
    /// The approval status changed.
    ApprovalChanged,
    /// A user was invited to the session.
    UserInvited,
    /// A user was removed from the session.
    UserRemoved,
    /// A version was uploaded.
    VersionUploaded,
    /// The session status changed.
    StatusChanged,
    /// A task was created.
    TaskCreated,
    /// A task was completed.
    TaskCompleted,
    /// Session metadata was modified.
    MetadataChanged,
}

impl fmt::Display for HistoryActionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::CommentAdded => "comment_added",
            Self::CommentEdited => "comment_edited",
            Self::CommentDeleted => "comment_deleted",
            Self::CommentResolved => "comment_resolved",
            Self::AnnotationCreated => "annotation_created",
            Self::AnnotationModified => "annotation_modified",
            Self::AnnotationRemoved => "annotation_removed",
            Self::ApprovalChanged => "approval_changed",
            Self::UserInvited => "user_invited",
            Self::UserRemoved => "user_removed",
            Self::VersionUploaded => "version_uploaded",
            Self::StatusChanged => "status_changed",
            Self::TaskCreated => "task_created",
            Self::TaskCompleted => "task_completed",
            Self::MetadataChanged => "metadata_changed",
        };
        write!(f, "{label}")
    }
}

/// A single entry in the review history.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Timestamp in milliseconds since session start.
    pub timestamp_ms: u64,
    /// Kind of action.
    pub kind: HistoryActionKind,
    /// ID of the user who performed the action.
    pub user_id: String,
    /// Target entity ID (comment id, annotation id, etc.).
    pub target_id: Option<String>,
    /// Human-readable description.
    pub description: String,
    /// Optional previous value for undo support.
    pub previous_value: Option<String>,
    /// Optional new value.
    pub new_value: Option<String>,
}

/// A filterable, bounded history log for a review session.
#[derive(Debug)]
pub struct ReviewHistory {
    /// All recorded entries.
    entries: VecDeque<HistoryEntry>,
    /// Next sequence number.
    next_seq: u64,
    /// Maximum capacity.
    max_entries: usize,
}

impl Default for ReviewHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl ReviewHistory {
    /// Create a new empty history with default capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            next_seq: 1,
            max_entries: DEFAULT_MAX_ENTRIES,
        }
    }

    /// Create a history with a custom maximum capacity.
    #[must_use]
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries.min(1024)),
            next_seq: 1,
            max_entries,
        }
    }

    /// Record a new action in the history.
    pub fn record(
        &mut self,
        timestamp_ms: u64,
        kind: HistoryActionKind,
        user_id: impl Into<String>,
        description: impl Into<String>,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;

        let entry = HistoryEntry {
            sequence: seq,
            timestamp_ms,
            kind,
            user_id: user_id.into(),
            target_id: None,
            description: description.into(),
            previous_value: None,
            new_value: None,
        };

        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        seq
    }

    /// Record a detailed action with target and value information.
    #[allow(clippy::too_many_arguments)]
    pub fn record_detailed(
        &mut self,
        timestamp_ms: u64,
        kind: HistoryActionKind,
        user_id: impl Into<String>,
        target_id: impl Into<String>,
        description: impl Into<String>,
        previous_value: Option<String>,
        new_value: Option<String>,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;

        let entry = HistoryEntry {
            sequence: seq,
            timestamp_ms,
            kind,
            user_id: user_id.into(),
            target_id: Some(target_id.into()),
            description: description.into(),
            previous_value,
            new_value,
        };

        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        seq
    }

    /// Get total number of recorded entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check whether the history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get an entry by its sequence number.
    #[must_use]
    pub fn get_by_sequence(&self, seq: u64) -> Option<&HistoryEntry> {
        self.entries.iter().find(|e| e.sequence == seq)
    }

    /// Get the most recent entry.
    #[must_use]
    pub fn latest(&self) -> Option<&HistoryEntry> {
        self.entries.back()
    }

    /// Filter entries by action kind.
    #[must_use]
    pub fn filter_by_kind(&self, kind: HistoryActionKind) -> Vec<&HistoryEntry> {
        self.entries.iter().filter(|e| e.kind == kind).collect()
    }

    /// Filter entries by user.
    #[must_use]
    pub fn filter_by_user(&self, user_id: &str) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.user_id == user_id)
            .collect()
    }

    /// Filter entries within a timestamp range (inclusive).
    #[must_use]
    pub fn filter_by_time_range(&self, start_ms: u64, end_ms: u64) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms <= end_ms)
            .collect()
    }

    /// Get all entries as a slice-like iterator.
    #[must_use]
    pub fn entries(&self) -> Vec<&HistoryEntry> {
        self.entries.iter().collect()
    }

    /// Clear all entries and reset the sequence counter.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_seq = 1;
    }

    /// Count entries by action kind.
    #[must_use]
    pub fn count_by_kind(&self, kind: HistoryActionKind) -> usize {
        self.entries.iter().filter(|e| e.kind == kind).count()
    }

    /// Get unique user IDs that have entries in the history.
    #[must_use]
    pub fn unique_users(&self) -> Vec<String> {
        let mut users: Vec<String> = self.entries.iter().map(|e| e.user_id.clone()).collect();
        users.sort();
        users.dedup();
        users
    }

    /// Generate a summary of actions per kind.
    #[must_use]
    pub fn summary(&self) -> Vec<(HistoryActionKind, usize)> {
        use std::collections::HashMap;
        let mut counts: HashMap<HistoryActionKind, usize> = HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.kind).or_insert(0) += 1;
        }
        let mut result: Vec<(HistoryActionKind, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_history_is_empty() {
        let history = ReviewHistory::new();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn test_record_single_entry() {
        let mut history = ReviewHistory::new();
        let seq = history.record(
            100,
            HistoryActionKind::CommentAdded,
            "user1",
            "Added comment",
        );
        assert_eq!(seq, 1);
        assert_eq!(history.len(), 1);
        assert!(!history.is_empty());
    }

    #[test]
    fn test_sequence_numbers_increment() {
        let mut history = ReviewHistory::new();
        let s1 = history.record(100, HistoryActionKind::CommentAdded, "user1", "First");
        let s2 = history.record(200, HistoryActionKind::CommentEdited, "user1", "Second");
        let s3 = history.record(300, HistoryActionKind::ApprovalChanged, "user2", "Third");
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
    }

    #[test]
    fn test_get_by_sequence() {
        let mut history = ReviewHistory::new();
        history.record(100, HistoryActionKind::CommentAdded, "user1", "Added");
        let seq = history.record(
            200,
            HistoryActionKind::VersionUploaded,
            "user2",
            "Uploaded v2",
        );
        let entry = history
            .get_by_sequence(seq)
            .expect("should succeed in test");
        assert_eq!(entry.kind, HistoryActionKind::VersionUploaded);
        assert_eq!(entry.user_id, "user2");
        assert!(history.get_by_sequence(999).is_none());
    }

    #[test]
    fn test_latest_entry() {
        let mut history = ReviewHistory::new();
        assert!(history.latest().is_none());
        history.record(100, HistoryActionKind::CommentAdded, "user1", "First");
        history.record(200, HistoryActionKind::ApprovalChanged, "user2", "Last");
        let latest = history.latest().expect("should succeed in test");
        assert_eq!(latest.kind, HistoryActionKind::ApprovalChanged);
    }

    #[test]
    fn test_filter_by_kind() {
        let mut history = ReviewHistory::new();
        history.record(100, HistoryActionKind::CommentAdded, "user1", "c1");
        history.record(200, HistoryActionKind::ApprovalChanged, "user2", "a1");
        history.record(300, HistoryActionKind::CommentAdded, "user3", "c2");
        let comments = history.filter_by_kind(HistoryActionKind::CommentAdded);
        assert_eq!(comments.len(), 2);
    }

    #[test]
    fn test_filter_by_user() {
        let mut history = ReviewHistory::new();
        history.record(100, HistoryActionKind::CommentAdded, "alice", "c1");
        history.record(200, HistoryActionKind::ApprovalChanged, "bob", "a1");
        history.record(300, HistoryActionKind::CommentEdited, "alice", "c2");
        let alice_entries = history.filter_by_user("alice");
        assert_eq!(alice_entries.len(), 2);
        let bob_entries = history.filter_by_user("bob");
        assert_eq!(bob_entries.len(), 1);
    }

    #[test]
    fn test_filter_by_time_range() {
        let mut history = ReviewHistory::new();
        history.record(100, HistoryActionKind::CommentAdded, "u1", "early");
        history.record(500, HistoryActionKind::CommentEdited, "u1", "mid");
        history.record(900, HistoryActionKind::ApprovalChanged, "u2", "late");
        let range = history.filter_by_time_range(200, 800);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].description, "mid");
    }

    #[test]
    fn test_capacity_eviction() {
        let mut history = ReviewHistory::with_capacity(3);
        history.record(100, HistoryActionKind::CommentAdded, "u1", "first");
        history.record(200, HistoryActionKind::CommentAdded, "u1", "second");
        history.record(300, HistoryActionKind::CommentAdded, "u1", "third");
        history.record(400, HistoryActionKind::CommentAdded, "u1", "fourth");
        assert_eq!(history.len(), 3);
        // first entry evicted
        assert!(history.get_by_sequence(1).is_none());
        assert!(history.get_by_sequence(2).is_some());
    }

    #[test]
    fn test_record_detailed() {
        let mut history = ReviewHistory::new();
        let seq = history.record_detailed(
            100,
            HistoryActionKind::CommentEdited,
            "user1",
            "comment-42",
            "Edited comment",
            Some("old text".to_string()),
            Some("new text".to_string()),
        );
        let entry = history
            .get_by_sequence(seq)
            .expect("should succeed in test");
        assert_eq!(entry.target_id.as_deref(), Some("comment-42"));
        assert_eq!(entry.previous_value.as_deref(), Some("old text"));
        assert_eq!(entry.new_value.as_deref(), Some("new text"));
    }

    #[test]
    fn test_clear() {
        let mut history = ReviewHistory::new();
        history.record(100, HistoryActionKind::CommentAdded, "u1", "c1");
        history.record(200, HistoryActionKind::CommentAdded, "u1", "c2");
        history.clear();
        assert!(history.is_empty());
        // sequence resets
        let seq = history.record(300, HistoryActionKind::CommentAdded, "u1", "c3");
        assert_eq!(seq, 1);
    }

    #[test]
    fn test_unique_users() {
        let mut history = ReviewHistory::new();
        history.record(100, HistoryActionKind::CommentAdded, "alice", "c1");
        history.record(200, HistoryActionKind::CommentAdded, "bob", "c2");
        history.record(300, HistoryActionKind::CommentAdded, "alice", "c3");
        let users = history.unique_users();
        assert_eq!(users.len(), 2);
        assert!(users.contains(&"alice".to_string()));
        assert!(users.contains(&"bob".to_string()));
    }

    #[test]
    fn test_summary() {
        let mut history = ReviewHistory::new();
        history.record(100, HistoryActionKind::CommentAdded, "u1", "c1");
        history.record(200, HistoryActionKind::CommentAdded, "u1", "c2");
        history.record(300, HistoryActionKind::ApprovalChanged, "u2", "a1");
        let summary = history.summary();
        assert_eq!(summary.len(), 2);
        // Most frequent first
        assert_eq!(summary[0].0, HistoryActionKind::CommentAdded);
        assert_eq!(summary[0].1, 2);
    }

    #[test]
    fn test_count_by_kind() {
        let mut history = ReviewHistory::new();
        history.record(100, HistoryActionKind::TaskCreated, "u1", "t1");
        history.record(200, HistoryActionKind::TaskCompleted, "u1", "t2");
        history.record(300, HistoryActionKind::TaskCreated, "u2", "t3");
        assert_eq!(history.count_by_kind(HistoryActionKind::TaskCreated), 2);
        assert_eq!(history.count_by_kind(HistoryActionKind::TaskCompleted), 1);
        assert_eq!(history.count_by_kind(HistoryActionKind::CommentAdded), 0);
    }

    #[test]
    fn test_action_kind_display() {
        assert_eq!(
            format!("{}", HistoryActionKind::CommentAdded),
            "comment_added"
        );
        assert_eq!(
            format!("{}", HistoryActionKind::ApprovalChanged),
            "approval_changed"
        );
        assert_eq!(
            format!("{}", HistoryActionKind::VersionUploaded),
            "version_uploaded"
        );
    }
}
