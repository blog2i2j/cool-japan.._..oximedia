#![allow(dead_code)]

//! Conform session snapshot system for undo, comparison, and audit trails.
//!
//! Provides checkpoint and restore capabilities for conform sessions,
//! enabling users to compare states, roll back changes, and maintain
//! an audit trail of conform operations.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique identifier for a snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SnapshotId(String);

impl SnapshotId {
    /// Create a new snapshot ID from a string.
    #[must_use]
    pub fn new(id: String) -> Self {
        Self(id)
    }

    /// Generate a unique snapshot ID based on timestamp and sequence.
    #[must_use]
    pub fn generate(sequence: u64) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        Self(format!("snap-{ts}-{sequence}"))
    }

    /// Get the inner string value.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// State of a clip in a conform snapshot.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClipState {
    /// Clip name or identifier.
    pub name: String,
    /// Source reel or media file identifier.
    pub source_reel: String,
    /// Source in timecode as a string.
    pub source_in: String,
    /// Source out timecode as a string.
    pub source_out: String,
    /// Record in timecode.
    pub record_in: String,
    /// Record out timecode.
    pub record_out: String,
    /// Whether this clip has been matched to media.
    pub matched: bool,
    /// Path to matched media, if any.
    pub media_path: Option<String>,
}

impl ClipState {
    /// Create a new clip state.
    #[must_use]
    pub fn new(
        name: String,
        source_reel: String,
        source_in: String,
        source_out: String,
        record_in: String,
        record_out: String,
    ) -> Self {
        Self {
            name,
            source_reel,
            source_in,
            source_out,
            record_in,
            record_out,
            matched: false,
            media_path: None,
        }
    }

    /// Mark this clip as matched to a media file.
    pub fn set_matched(&mut self, path: String) {
        self.matched = true;
        self.media_path = Some(path);
    }
}

/// A complete snapshot of a conform session state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConformSnapshot {
    /// Unique identifier.
    pub id: SnapshotId,
    /// Human-readable label.
    pub label: String,
    /// Timestamp when created (Unix seconds).
    pub created_at: u64,
    /// All clip states at this point.
    pub clips: Vec<ClipState>,
    /// Key-value metadata.
    pub metadata: HashMap<String, String>,
    /// Number of matched clips at this point.
    pub matched_count: usize,
    /// Total number of clips.
    pub total_count: usize,
}

impl ConformSnapshot {
    /// Create a new snapshot.
    #[must_use]
    pub fn new(id: SnapshotId, label: String, clips: Vec<ClipState>) -> Self {
        let matched_count = clips.iter().filter(|c| c.matched).count();
        let total_count = clips.len();
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id,
            label,
            created_at,
            clips,
            metadata: HashMap::new(),
            matched_count,
            total_count,
        }
    }

    /// Match percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn match_pct(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        (self.matched_count as f64 / self.total_count as f64) * 100.0
    }

    /// Add metadata to the snapshot.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get unmatched clips.
    #[must_use]
    pub fn unmatched_clips(&self) -> Vec<&ClipState> {
        self.clips.iter().filter(|c| !c.matched).collect()
    }
}

/// A change between two snapshots.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SnapshotChange {
    /// A clip was added.
    ClipAdded(String),
    /// A clip was removed.
    ClipRemoved(String),
    /// A clip's match status changed.
    MatchChanged {
        /// Clip name.
        clip: String,
        /// Old matched state.
        was_matched: bool,
        /// New matched state.
        now_matched: bool,
    },
    /// A clip's media path changed.
    MediaChanged {
        /// Clip name.
        clip: String,
        /// Old media path.
        old_path: Option<String>,
        /// New media path.
        new_path: Option<String>,
    },
    /// A clip's timing changed.
    TimingChanged {
        /// Clip name.
        clip: String,
        /// Description of what changed.
        detail: String,
    },
}

impl fmt::Display for SnapshotChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClipAdded(name) => write!(f, "Added clip: {name}"),
            Self::ClipRemoved(name) => write!(f, "Removed clip: {name}"),
            Self::MatchChanged {
                clip,
                was_matched,
                now_matched,
            } => write!(
                f,
                "Match changed for {clip}: {was_matched} -> {now_matched}"
            ),
            Self::MediaChanged {
                clip,
                old_path,
                new_path,
            } => write!(
                f,
                "Media changed for {clip}: {:?} -> {:?}",
                old_path, new_path
            ),
            Self::TimingChanged { clip, detail } => {
                write!(f, "Timing changed for {clip}: {detail}")
            }
        }
    }
}

/// Result of comparing two snapshots.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotDiff {
    /// The older snapshot ID.
    pub from_id: SnapshotId,
    /// The newer snapshot ID.
    pub to_id: SnapshotId,
    /// All detected changes.
    pub changes: Vec<SnapshotChange>,
    /// Matched count difference (new - old).
    pub matched_delta: i64,
}

impl SnapshotDiff {
    /// Whether there are any changes.
    #[must_use]
    pub fn has_changes(&self) -> bool {
        !self.changes.is_empty()
    }

    /// Count of changes.
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }
}

/// Compare two snapshots to find differences.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn diff_snapshots(from: &ConformSnapshot, to: &ConformSnapshot) -> SnapshotDiff {
    let mut changes = Vec::new();

    let from_map: HashMap<&str, &ClipState> =
        from.clips.iter().map(|c| (c.name.as_str(), c)).collect();
    let to_map: HashMap<&str, &ClipState> =
        to.clips.iter().map(|c| (c.name.as_str(), c)).collect();

    // Find added and changed clips
    for (name, to_clip) in &to_map {
        match from_map.get(name) {
            None => {
                changes.push(SnapshotChange::ClipAdded(name.to_string()));
            }
            Some(from_clip) => {
                if from_clip.matched != to_clip.matched {
                    changes.push(SnapshotChange::MatchChanged {
                        clip: name.to_string(),
                        was_matched: from_clip.matched,
                        now_matched: to_clip.matched,
                    });
                }
                if from_clip.media_path != to_clip.media_path {
                    changes.push(SnapshotChange::MediaChanged {
                        clip: name.to_string(),
                        old_path: from_clip.media_path.clone(),
                        new_path: to_clip.media_path.clone(),
                    });
                }
                if from_clip.source_in != to_clip.source_in
                    || from_clip.source_out != to_clip.source_out
                {
                    changes.push(SnapshotChange::TimingChanged {
                        clip: name.to_string(),
                        detail: format!(
                            "src {}-{} -> {}-{}",
                            from_clip.source_in,
                            from_clip.source_out,
                            to_clip.source_in,
                            to_clip.source_out
                        ),
                    });
                }
            }
        }
    }

    // Find removed clips
    for name in from_map.keys() {
        if !to_map.contains_key(name) {
            changes.push(SnapshotChange::ClipRemoved(name.to_string()));
        }
    }

    let matched_delta = to.matched_count as i64 - from.matched_count as i64;

    SnapshotDiff {
        from_id: from.id.clone(),
        to_id: to.id.clone(),
        changes,
        matched_delta,
    }
}

/// Manages a sequence of snapshots for a session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotHistory {
    /// All snapshots in chronological order.
    snapshots: Vec<ConformSnapshot>,
    /// Maximum number of snapshots to retain.
    max_snapshots: usize,
    /// Sequence counter for ID generation.
    sequence: u64,
}

impl SnapshotHistory {
    /// Create a new snapshot history.
    #[must_use]
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots,
            sequence: 0,
        }
    }

    /// Take a new snapshot.
    pub fn take_snapshot(&mut self, label: String, clips: Vec<ClipState>) -> SnapshotId {
        self.sequence += 1;
        let id = SnapshotId::generate(self.sequence);
        let snapshot = ConformSnapshot::new(id.clone(), label, clips);
        self.snapshots.push(snapshot);

        // Evict oldest if over limit
        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.remove(0);
        }

        id
    }

    /// Get the latest snapshot.
    #[must_use]
    pub fn latest(&self) -> Option<&ConformSnapshot> {
        self.snapshots.last()
    }

    /// Get a snapshot by ID.
    #[must_use]
    pub fn get(&self, id: &SnapshotId) -> Option<&ConformSnapshot> {
        self.snapshots.iter().find(|s| &s.id == id)
    }

    /// Number of snapshots.
    #[must_use]
    pub fn count(&self) -> usize {
        self.snapshots.len()
    }

    /// List all snapshot IDs and labels.
    #[must_use]
    pub fn list(&self) -> Vec<(&SnapshotId, &str)> {
        self.snapshots
            .iter()
            .map(|s| (&s.id, s.label.as_str()))
            .collect()
    }

    /// Compare two snapshots by their indices in history.
    #[must_use]
    pub fn diff_by_index(&self, from_idx: usize, to_idx: usize) -> Option<SnapshotDiff> {
        let from = self.snapshots.get(from_idx)?;
        let to = self.snapshots.get(to_idx)?;
        Some(diff_snapshots(from, to))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_clip(name: &str, matched: bool) -> ClipState {
        let mut clip = ClipState::new(
            name.to_string(),
            "REEL01".to_string(),
            "01:00:00:00".to_string(),
            "01:00:10:00".to_string(),
            "00:00:00:00".to_string(),
            "00:00:10:00".to_string(),
        );
        if matched {
            clip.set_matched("/media/test.mxf".to_string());
        }
        clip
    }

    #[test]
    fn test_snapshot_id_generate() {
        let id1 = SnapshotId::generate(1);
        let id2 = SnapshotId::generate(2);
        assert_ne!(id1.as_str(), id2.as_str());
    }

    #[test]
    fn test_snapshot_id_display() {
        let id = SnapshotId::new("test-123".to_string());
        assert_eq!(id.to_string(), "test-123");
    }

    #[test]
    fn test_clip_state_set_matched() {
        let mut clip = sample_clip("A", false);
        assert!(!clip.matched);
        clip.set_matched("/path/to/media.mov".to_string());
        assert!(clip.matched);
        assert_eq!(clip.media_path.as_deref(), Some("/path/to/media.mov"));
    }

    #[test]
    fn test_snapshot_match_pct() {
        let clips = vec![sample_clip("A", true), sample_clip("B", false)];
        let snap = ConformSnapshot::new(SnapshotId::new("s1".to_string()), "test".to_string(), clips);
        assert!((snap.match_pct() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_snapshot_match_pct_empty() {
        let snap = ConformSnapshot::new(SnapshotId::new("s1".to_string()), "test".to_string(), vec![]);
        assert!(snap.match_pct().abs() < f64::EPSILON);
    }

    #[test]
    fn test_snapshot_unmatched_clips() {
        let clips = vec![sample_clip("A", true), sample_clip("B", false)];
        let snap = ConformSnapshot::new(SnapshotId::new("s1".to_string()), "test".to_string(), clips);
        let unmatched = snap.unmatched_clips();
        assert_eq!(unmatched.len(), 1);
        assert_eq!(unmatched[0].name, "B");
    }

    #[test]
    fn test_diff_no_changes() {
        let clips = vec![sample_clip("A", true)];
        let snap1 = ConformSnapshot::new(SnapshotId::new("s1".to_string()), "v1".to_string(), clips.clone());
        let snap2 = ConformSnapshot::new(SnapshotId::new("s2".to_string()), "v2".to_string(), clips);
        let diff = diff_snapshots(&snap1, &snap2);
        assert!(!diff.has_changes());
        assert_eq!(diff.matched_delta, 0);
    }

    #[test]
    fn test_diff_clip_added() {
        let clips1 = vec![sample_clip("A", true)];
        let clips2 = vec![sample_clip("A", true), sample_clip("B", false)];
        let snap1 = ConformSnapshot::new(SnapshotId::new("s1".to_string()), "v1".to_string(), clips1);
        let snap2 = ConformSnapshot::new(SnapshotId::new("s2".to_string()), "v2".to_string(), clips2);
        let diff = diff_snapshots(&snap1, &snap2);
        assert!(diff.has_changes());
        assert_eq!(diff.change_count(), 1);
    }

    #[test]
    fn test_diff_clip_removed() {
        let clips1 = vec![sample_clip("A", true), sample_clip("B", false)];
        let clips2 = vec![sample_clip("A", true)];
        let snap1 = ConformSnapshot::new(SnapshotId::new("s1".to_string()), "v1".to_string(), clips1);
        let snap2 = ConformSnapshot::new(SnapshotId::new("s2".to_string()), "v2".to_string(), clips2);
        let diff = diff_snapshots(&snap1, &snap2);
        assert!(diff.has_changes());
    }

    #[test]
    fn test_diff_match_changed() {
        let clips1 = vec![sample_clip("A", false)];
        let clips2 = vec![sample_clip("A", true)];
        let snap1 = ConformSnapshot::new(SnapshotId::new("s1".to_string()), "v1".to_string(), clips1);
        let snap2 = ConformSnapshot::new(SnapshotId::new("s2".to_string()), "v2".to_string(), clips2);
        let diff = diff_snapshots(&snap1, &snap2);
        assert_eq!(diff.matched_delta, 1);
    }

    #[test]
    fn test_history_take_snapshot() {
        let mut history = SnapshotHistory::new(10);
        let id = history.take_snapshot("first".to_string(), vec![sample_clip("A", true)]);
        assert_eq!(history.count(), 1);
        assert!(history.get(&id).is_some());
    }

    #[test]
    fn test_history_eviction() {
        let mut history = SnapshotHistory::new(3);
        for i in 0..5 {
            history.take_snapshot(format!("snap-{i}"), vec![]);
        }
        assert_eq!(history.count(), 3);
    }

    #[test]
    fn test_snapshot_change_display() {
        let change = SnapshotChange::ClipAdded("test_clip".to_string());
        assert!(change.to_string().contains("test_clip"));
    }

    #[test]
    fn test_history_diff_by_index() {
        let mut history = SnapshotHistory::new(10);
        history.take_snapshot("v1".to_string(), vec![sample_clip("A", false)]);
        history.take_snapshot("v2".to_string(), vec![sample_clip("A", true)]);
        let diff = history.diff_by_index(0, 1);
        assert!(diff.is_some());
        assert!(diff.expect("test expectation failed").has_changes());
    }
}
