#![allow(dead_code)]
//! Clip audit trail and change logging.
//!
//! This module provides a comprehensive audit trail system for tracking all
//! changes made to clips, including metadata edits, trim adjustments, tag
//! assignments, rating changes, and organizational moves. Each change is
//! recorded with a timestamp, user identity, and before/after values for
//! undo support and compliance auditing.

use std::collections::HashMap;

/// Unique identifier for an audit entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AuditEntryId(pub u64);

/// Type of change recorded in the audit trail.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChangeType {
    /// Clip was created.
    Created,
    /// Clip was deleted.
    Deleted,
    /// Clip name was changed.
    NameChanged,
    /// Trim points were modified.
    TrimChanged,
    /// Rating was changed.
    RatingChanged,
    /// A keyword was added.
    KeywordAdded,
    /// A keyword was removed.
    KeywordRemoved,
    /// A tag was assigned.
    TagAssigned,
    /// A tag was removed.
    TagRemoved,
    /// Clip was moved to a different bin.
    BinMoved,
    /// Metadata field was updated.
    MetadataUpdated,
    /// Note was added or changed.
    NoteChanged,
    /// Color label was changed.
    ColorLabelChanged,
}

impl std::fmt::Display for ChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Created => write!(f, "Created"),
            Self::Deleted => write!(f, "Deleted"),
            Self::NameChanged => write!(f, "Name Changed"),
            Self::TrimChanged => write!(f, "Trim Changed"),
            Self::RatingChanged => write!(f, "Rating Changed"),
            Self::KeywordAdded => write!(f, "Keyword Added"),
            Self::KeywordRemoved => write!(f, "Keyword Removed"),
            Self::TagAssigned => write!(f, "Tag Assigned"),
            Self::TagRemoved => write!(f, "Tag Removed"),
            Self::BinMoved => write!(f, "Bin Moved"),
            Self::MetadataUpdated => write!(f, "Metadata Updated"),
            Self::NoteChanged => write!(f, "Note Changed"),
            Self::ColorLabelChanged => write!(f, "Color Label Changed"),
        }
    }
}

/// A single audit entry recording a change to a clip.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Unique entry identifier.
    pub id: AuditEntryId,
    /// Clip that was modified.
    pub clip_id: u64,
    /// Type of change.
    pub change_type: ChangeType,
    /// User who made the change.
    pub user: String,
    /// Timestamp as Unix epoch seconds.
    pub timestamp: u64,
    /// Previous value (serialized as string).
    pub old_value: Option<String>,
    /// New value (serialized as string).
    pub new_value: Option<String>,
    /// Optional description or reason for the change.
    pub description: String,
}

impl AuditEntry {
    /// Creates a new audit entry.
    #[must_use]
    pub fn new(
        id: AuditEntryId,
        clip_id: u64,
        change_type: ChangeType,
        user: impl Into<String>,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            clip_id,
            change_type,
            user: user.into(),
            timestamp,
            old_value: None,
            new_value: None,
            description: String::new(),
        }
    }

    /// Sets the old and new values.
    pub fn with_values(mut self, old: impl Into<String>, new: impl Into<String>) -> Self {
        self.old_value = Some(old.into());
        self.new_value = Some(new.into());
        self
    }

    /// Sets the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Returns true if this entry has before/after values (undoable).
    #[must_use]
    pub fn is_undoable(&self) -> bool {
        self.old_value.is_some() && self.new_value.is_some()
    }
}

/// Filter criteria for querying the audit trail.
#[derive(Debug, Clone, Default)]
pub struct AuditFilter {
    /// Filter by clip ID.
    pub clip_id: Option<u64>,
    /// Filter by change type.
    pub change_type: Option<ChangeType>,
    /// Filter by user.
    pub user: Option<String>,
    /// Filter entries after this timestamp.
    pub after_timestamp: Option<u64>,
    /// Filter entries before this timestamp.
    pub before_timestamp: Option<u64>,
    /// Maximum number of results.
    pub limit: Option<usize>,
}

impl AuditFilter {
    /// Creates a filter for a specific clip.
    #[must_use]
    pub fn for_clip(clip_id: u64) -> Self {
        Self {
            clip_id: Some(clip_id),
            ..Default::default()
        }
    }

    /// Creates a filter for a specific user.
    #[must_use]
    pub fn for_user(user: impl Into<String>) -> Self {
        Self {
            user: Some(user.into()),
            ..Default::default()
        }
    }

    /// Sets a time range.
    #[must_use]
    pub fn in_time_range(mut self, after: u64, before: u64) -> Self {
        self.after_timestamp = Some(after);
        self.before_timestamp = Some(before);
        self
    }

    /// Sets the result limit.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Returns true if an entry matches this filter.
    #[must_use]
    pub fn matches(&self, entry: &AuditEntry) -> bool {
        if let Some(cid) = self.clip_id {
            if entry.clip_id != cid {
                return false;
            }
        }
        if let Some(ref ct) = self.change_type {
            if entry.change_type != *ct {
                return false;
            }
        }
        if let Some(ref u) = self.user {
            if entry.user != *u {
                return false;
            }
        }
        if let Some(after) = self.after_timestamp {
            if entry.timestamp < after {
                return false;
            }
        }
        if let Some(before) = self.before_timestamp {
            if entry.timestamp > before {
                return false;
            }
        }
        true
    }
}

/// Summary of audit activity for reporting.
#[derive(Debug, Clone)]
pub struct AuditSummary {
    /// Total number of changes.
    pub total_changes: usize,
    /// Changes grouped by type.
    pub changes_by_type: HashMap<String, usize>,
    /// Changes grouped by user.
    pub changes_by_user: HashMap<String, usize>,
    /// Number of unique clips modified.
    pub unique_clips: usize,
}

/// Audit trail manager for tracking clip changes.
#[derive(Debug)]
pub struct AuditTrail {
    /// All audit entries.
    entries: Vec<AuditEntry>,
    /// Index by clip ID for fast lookup.
    clip_index: HashMap<u64, Vec<usize>>,
    /// Next entry ID.
    next_id: u64,
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditTrail {
    /// Creates a new empty audit trail.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            clip_index: HashMap::new(),
            next_id: 1,
        }
    }

    /// Records a change in the audit trail.
    pub fn record(
        &mut self,
        clip_id: u64,
        change_type: ChangeType,
        user: impl Into<String>,
        timestamp: u64,
    ) -> AuditEntryId {
        let id = AuditEntryId(self.next_id);
        self.next_id += 1;
        let entry = AuditEntry::new(id, clip_id, change_type, user, timestamp);
        let index = self.entries.len();
        self.entries.push(entry);
        self.clip_index.entry(clip_id).or_default().push(index);
        id
    }

    /// Records a change with before/after values.
    pub fn record_with_values(
        &mut self,
        clip_id: u64,
        change_type: ChangeType,
        user: impl Into<String>,
        timestamp: u64,
        old_value: impl Into<String>,
        new_value: impl Into<String>,
    ) -> AuditEntryId {
        let id = AuditEntryId(self.next_id);
        self.next_id += 1;
        let entry = AuditEntry::new(id, clip_id, change_type, user, timestamp)
            .with_values(old_value, new_value);
        let index = self.entries.len();
        self.entries.push(entry);
        self.clip_index.entry(clip_id).or_default().push(index);
        id
    }

    /// Returns a reference to an entry by ID.
    #[must_use]
    pub fn get_entry(&self, id: AuditEntryId) -> Option<&AuditEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Returns the total number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns all entries for a specific clip.
    #[must_use]
    pub fn history_for_clip(&self, clip_id: u64) -> Vec<&AuditEntry> {
        self.clip_index
            .get(&clip_id)
            .map(|indices| indices.iter().map(|&i| &self.entries[i]).collect())
            .unwrap_or_default()
    }

    /// Queries the audit trail with a filter.
    #[must_use]
    pub fn query(&self, filter: &AuditFilter) -> Vec<&AuditEntry> {
        let mut results: Vec<&AuditEntry> =
            self.entries.iter().filter(|e| filter.matches(e)).collect();
        // Sort by timestamp descending (most recent first)
        results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }
        results
    }

    /// Generates an audit summary for the given filter.
    #[must_use]
    pub fn summarize(&self, filter: &AuditFilter) -> AuditSummary {
        let matching: Vec<&AuditEntry> =
            self.entries.iter().filter(|e| filter.matches(e)).collect();

        let mut changes_by_type: HashMap<String, usize> = HashMap::new();
        let mut changes_by_user: HashMap<String, usize> = HashMap::new();
        let mut unique_clips = std::collections::HashSet::new();

        for entry in &matching {
            *changes_by_type
                .entry(entry.change_type.to_string())
                .or_insert(0) += 1;
            *changes_by_user.entry(entry.user.clone()).or_insert(0) += 1;
            unique_clips.insert(entry.clip_id);
        }

        AuditSummary {
            total_changes: matching.len(),
            changes_by_type,
            changes_by_user,
            unique_clips: unique_clips.len(),
        }
    }

    /// Returns the last N entries (most recent first).
    #[must_use]
    pub fn recent_entries(&self, count: usize) -> Vec<&AuditEntry> {
        let mut entries: Vec<&AuditEntry> = self.entries.iter().collect();
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(count);
        entries
    }

    /// Clears all entries older than the given timestamp.
    pub fn purge_before(&mut self, timestamp: u64) {
        self.entries.retain(|e| e.timestamp >= timestamp);
        // Rebuild clip index
        self.clip_index.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            self.clip_index.entry(entry.clip_id).or_default().push(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_type_display() {
        assert_eq!(ChangeType::Created.to_string(), "Created");
        assert_eq!(ChangeType::NameChanged.to_string(), "Name Changed");
        assert_eq!(ChangeType::TrimChanged.to_string(), "Trim Changed");
    }

    #[test]
    fn test_audit_entry_new() {
        let entry = AuditEntry::new(AuditEntryId(1), 100, ChangeType::Created, "admin", 1000);
        assert_eq!(entry.id, AuditEntryId(1));
        assert_eq!(entry.clip_id, 100);
        assert_eq!(entry.user, "admin");
    }

    #[test]
    fn test_audit_entry_with_values() {
        let entry = AuditEntry::new(
            AuditEntryId(1),
            100,
            ChangeType::NameChanged,
            "editor",
            1000,
        )
        .with_values("old-name", "new-name");
        assert_eq!(entry.old_value.as_deref(), Some("old-name"));
        assert_eq!(entry.new_value.as_deref(), Some("new-name"));
        assert!(entry.is_undoable());
    }

    #[test]
    fn test_audit_entry_not_undoable() {
        let entry = AuditEntry::new(AuditEntryId(1), 100, ChangeType::Created, "admin", 1000);
        assert!(!entry.is_undoable());
    }

    #[test]
    fn test_audit_entry_with_description() {
        let entry = AuditEntry::new(AuditEntryId(1), 100, ChangeType::Deleted, "admin", 1000)
            .with_description("Removed duplicate clip");
        assert_eq!(entry.description, "Removed duplicate clip");
    }

    #[test]
    fn test_audit_filter_matches() {
        let entry = AuditEntry::new(
            AuditEntryId(1),
            100,
            ChangeType::RatingChanged,
            "editor",
            5000,
        );
        let filter = AuditFilter::for_clip(100);
        assert!(filter.matches(&entry));
        let filter2 = AuditFilter::for_clip(999);
        assert!(!filter2.matches(&entry));
    }

    #[test]
    fn test_audit_filter_time_range() {
        let entry = AuditEntry::new(AuditEntryId(1), 100, ChangeType::Created, "admin", 5000);
        let filter = AuditFilter::default().in_time_range(4000, 6000);
        assert!(filter.matches(&entry));
        let filter2 = AuditFilter::default().in_time_range(6000, 7000);
        assert!(!filter2.matches(&entry));
    }

    #[test]
    fn test_audit_filter_user() {
        let entry = AuditEntry::new(AuditEntryId(1), 100, ChangeType::Created, "admin", 1000);
        let filter = AuditFilter::for_user("admin");
        assert!(filter.matches(&entry));
        let filter2 = AuditFilter::for_user("editor");
        assert!(!filter2.matches(&entry));
    }

    #[test]
    fn test_trail_record() {
        let mut trail = AuditTrail::new();
        let id = trail.record(100, ChangeType::Created, "admin", 1000);
        assert_eq!(trail.entry_count(), 1);
        let entry = trail.get_entry(id).expect("get_entry should succeed");
        assert_eq!(entry.clip_id, 100);
    }

    #[test]
    fn test_trail_record_with_values() {
        let mut trail = AuditTrail::new();
        let id = trail.record_with_values(
            100,
            ChangeType::NameChanged,
            "editor",
            2000,
            "clip-a",
            "clip-b",
        );
        let entry = trail.get_entry(id).expect("get_entry should succeed");
        assert!(entry.is_undoable());
    }

    #[test]
    fn test_trail_history_for_clip() {
        let mut trail = AuditTrail::new();
        trail.record(100, ChangeType::Created, "admin", 1000);
        trail.record(100, ChangeType::NameChanged, "editor", 2000);
        trail.record(200, ChangeType::Created, "admin", 1500);
        let history = trail.history_for_clip(100);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_trail_query_with_filter() {
        let mut trail = AuditTrail::new();
        trail.record(100, ChangeType::Created, "admin", 1000);
        trail.record(100, ChangeType::RatingChanged, "editor", 2000);
        trail.record(200, ChangeType::Created, "admin", 3000);
        let filter = AuditFilter::for_clip(100);
        let results = trail.query(&filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_trail_query_with_limit() {
        let mut trail = AuditTrail::new();
        for i in 0..10 {
            trail.record(100, ChangeType::MetadataUpdated, "bot", 1000 + i);
        }
        let filter = AuditFilter::for_clip(100).with_limit(3);
        let results = trail.query(&filter);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_trail_summarize() {
        let mut trail = AuditTrail::new();
        trail.record(100, ChangeType::Created, "admin", 1000);
        trail.record(100, ChangeType::RatingChanged, "editor", 2000);
        trail.record(200, ChangeType::Created, "admin", 1500);
        let summary = trail.summarize(&AuditFilter::default());
        assert_eq!(summary.total_changes, 3);
        assert_eq!(summary.unique_clips, 2);
        assert_eq!(
            *summary
                .changes_by_type
                .get("Created")
                .expect("get should succeed"),
            2
        );
        assert_eq!(
            *summary
                .changes_by_user
                .get("admin")
                .expect("get should succeed"),
            2
        );
    }

    #[test]
    fn test_trail_recent_entries() {
        let mut trail = AuditTrail::new();
        trail.record(100, ChangeType::Created, "admin", 1000);
        trail.record(200, ChangeType::Created, "admin", 3000);
        trail.record(300, ChangeType::Created, "admin", 2000);
        let recent = trail.recent_entries(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].clip_id, 200); // most recent
    }

    #[test]
    fn test_trail_purge_before() {
        let mut trail = AuditTrail::new();
        trail.record(100, ChangeType::Created, "admin", 1000);
        trail.record(200, ChangeType::Created, "admin", 2000);
        trail.record(300, ChangeType::Created, "admin", 3000);
        trail.purge_before(2000);
        assert_eq!(trail.entry_count(), 2);
        let history = trail.history_for_clip(100);
        assert!(history.is_empty());
    }
}
