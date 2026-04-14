//! Caption revision history and version control.
//!
//! Provides a lightweight append-only revision log for caption tracks.  Each
//! edit is recorded as an immutable [`CaptionRevision`] entry, enabling undo/redo,
//! audit trails, and collaborative conflict resolution.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ── Revision types ────────────────────────────────────────────────────────────

/// The kind of change captured in a [`CaptionRevision`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevisionKind {
    /// Caption text was edited (`(old_text, new_text)`).
    TextEdit,
    /// Timing was adjusted.
    TimingAdjust,
    /// A new caption was inserted.
    Insert,
    /// A caption was deleted.
    Delete,
    /// Multiple changes in a single user action (composite).
    Composite,
    /// Style or format change (color, font, position).
    StyleChange,
}

/// A single recorded change to a caption track.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionRevision {
    /// Monotonically increasing revision number (1-based).
    pub revision: u32,
    /// ID of the caption affected (may be empty for document-level changes).
    pub caption_id: String,
    /// The kind of change.
    pub kind: RevisionKind,
    /// Short description written by the author or generated automatically.
    pub summary: String,
    /// Optional free-form diff data (e.g. unified diff of the text).
    pub diff: Option<String>,
    /// Author identifier (username or UUID).
    pub author: String,
    /// Unix timestamp (seconds since epoch) when the revision was recorded.
    pub timestamp_secs: u64,
}

impl CaptionRevision {
    /// Create a new revision entry.
    #[must_use]
    pub fn new(
        revision: u32,
        caption_id: impl Into<String>,
        kind: RevisionKind,
        summary: impl Into<String>,
        author: impl Into<String>,
        timestamp_secs: u64,
    ) -> Self {
        Self {
            revision,
            caption_id: caption_id.into(),
            kind,
            summary: summary.into(),
            diff: None,
            author: author.into(),
            timestamp_secs,
        }
    }

    /// Attach diff data to this revision.
    #[must_use]
    pub fn with_diff(mut self, diff: impl Into<String>) -> Self {
        self.diff = Some(diff.into());
        self
    }
}

// ── VersionHistory ────────────────────────────────────────────────────────────

/// Append-only revision history for a caption track.
///
/// Revisions are numbered sequentially starting at 1.  The history supports
/// querying by author, by caption ID, and by time range.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VersionHistory {
    revisions: Vec<CaptionRevision>,
}

impl VersionHistory {
    /// Create an empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new revision.  The revision number is assigned automatically as
    /// `current_count + 1`.
    pub fn record(
        &mut self,
        caption_id: impl Into<String>,
        kind: RevisionKind,
        summary: impl Into<String>,
        author: impl Into<String>,
        timestamp_secs: u64,
    ) {
        let next = self.revisions.len() as u32 + 1;
        self.revisions.push(CaptionRevision::new(
            next,
            caption_id,
            kind,
            summary,
            author,
            timestamp_secs,
        ));
    }

    /// Return all revisions, oldest first.
    #[must_use]
    pub fn all(&self) -> &[CaptionRevision] {
        &self.revisions
    }

    /// Return all revisions for a specific caption ID.
    #[must_use]
    pub fn for_caption(&self, caption_id: &str) -> Vec<&CaptionRevision> {
        self.revisions
            .iter()
            .filter(|r| r.caption_id == caption_id)
            .collect()
    }

    /// Return all revisions authored by `author`.
    #[must_use]
    pub fn by_author(&self, author: &str) -> Vec<&CaptionRevision> {
        self.revisions
            .iter()
            .filter(|r| r.author == author)
            .collect()
    }

    /// Return revisions recorded in the half-open time interval
    /// `[start_secs, end_secs)`.
    #[must_use]
    pub fn in_range(&self, start_secs: u64, end_secs: u64) -> Vec<&CaptionRevision> {
        self.revisions
            .iter()
            .filter(|r| r.timestamp_secs >= start_secs && r.timestamp_secs < end_secs)
            .collect()
    }

    /// Total number of recorded revisions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.revisions.len()
    }

    /// Returns `true` when no revisions have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.revisions.is_empty()
    }

    /// Return the most recent revision, or `None` if the history is empty.
    #[must_use]
    pub fn latest(&self) -> Option<&CaptionRevision> {
        self.revisions.last()
    }
}

// ── Snapshot / restore ────────────────────────────────────────────────────────

/// A lightweight caption text snapshot used as a checkpoint.
///
/// Pairs a caption ID with the verbatim text at the time of snapshotting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionSnapshot {
    /// The caption identifier this snapshot was taken from.
    pub caption_id: String,
    /// Revision number at the time the snapshot was created.
    pub at_revision: u32,
    /// Verbatim caption text at `at_revision`.
    pub text: String,
}

impl CaptionSnapshot {
    /// Create a new snapshot.
    #[must_use]
    pub fn new(caption_id: impl Into<String>, at_revision: u32, text: impl Into<String>) -> Self {
        Self {
            caption_id: caption_id.into(),
            at_revision,
            text: text.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(n: u64) -> u64 {
        1_700_000_000 + n
    }

    #[test]
    fn test_empty_history() {
        let h = VersionHistory::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert!(h.latest().is_none());
    }

    #[test]
    fn test_record_and_count() {
        let mut h = VersionHistory::new();
        h.record("cap-1", RevisionKind::TextEdit, "Fixed typo", "alice", ts(0));
        h.record("cap-2", RevisionKind::Insert, "Added cue", "bob", ts(1));
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn test_revision_numbers_sequential() {
        let mut h = VersionHistory::new();
        h.record("cap-1", RevisionKind::TextEdit, "Edit 1", "alice", ts(0));
        h.record("cap-1", RevisionKind::TextEdit, "Edit 2", "alice", ts(1));
        let revs = h.all();
        assert_eq!(revs[0].revision, 1);
        assert_eq!(revs[1].revision, 2);
    }

    #[test]
    fn test_for_caption_filters_correctly() {
        let mut h = VersionHistory::new();
        h.record("cap-1", RevisionKind::TextEdit, "e", "alice", ts(0));
        h.record("cap-2", RevisionKind::Insert, "i", "alice", ts(1));
        h.record("cap-1", RevisionKind::TimingAdjust, "t", "alice", ts(2));
        let cap1_revs = h.for_caption("cap-1");
        assert_eq!(cap1_revs.len(), 2);
    }

    #[test]
    fn test_by_author() {
        let mut h = VersionHistory::new();
        h.record("cap-1", RevisionKind::TextEdit, "e", "alice", ts(0));
        h.record("cap-2", RevisionKind::TextEdit, "e", "bob", ts(1));
        assert_eq!(h.by_author("alice").len(), 1);
        assert_eq!(h.by_author("bob").len(), 1);
        assert_eq!(h.by_author("charlie").len(), 0);
    }

    #[test]
    fn test_in_range() {
        let mut h = VersionHistory::new();
        h.record("c", RevisionKind::TextEdit, "e", "u", ts(0)); // t=1_700_000_000
        h.record("c", RevisionKind::TextEdit, "e", "u", ts(5)); // t=1_700_000_005
        h.record("c", RevisionKind::TextEdit, "e", "u", ts(10)); // t=1_700_000_010
        let r = h.in_range(ts(0), ts(6));
        assert_eq!(r.len(), 2); // ts(0) and ts(5) are in [ts(0), ts(6))
    }

    #[test]
    fn test_latest_returns_last_revision() {
        let mut h = VersionHistory::new();
        h.record("cap-1", RevisionKind::Delete, "Removed", "alice", ts(0));
        h.record("cap-2", RevisionKind::Insert, "Added", "bob", ts(1));
        let latest = h.latest().expect("latest should exist");
        assert_eq!(latest.revision, 2);
        assert_eq!(latest.kind, RevisionKind::Insert);
    }

    #[test]
    fn test_snapshot_creation() {
        let snap = CaptionSnapshot::new("cap-3", 5, "Hello world.");
        assert_eq!(snap.caption_id, "cap-3");
        assert_eq!(snap.at_revision, 5);
        assert_eq!(snap.text, "Hello world.");
    }

    #[test]
    fn test_revision_with_diff() {
        let rev = CaptionRevision::new(1, "c1", RevisionKind::TextEdit, "fix", "u", ts(0))
            .with_diff("- old\n+ new\n");
        assert!(rev.diff.is_some());
    }
}
