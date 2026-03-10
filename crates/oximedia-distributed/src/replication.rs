//! Log replication and term tracking for distributed consensus support.
//!
//! Provides lightweight log entry management, term tracking, and
//! quorum calculation utilities used by the consensus module.

use std::collections::VecDeque;

/// A single entry in the replicated log
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    /// The term in which this entry was created
    pub term: u64,
    /// The log index (1-based)
    pub index: u64,
    /// Payload data
    pub data: Vec<u8>,
    /// Human-readable command tag
    pub command: String,
}

impl LogEntry {
    /// Create a new log entry
    #[allow(dead_code)]
    pub fn new(term: u64, index: u64, command: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            term,
            index,
            data,
            command: command.into(),
        }
    }
}

/// Replicated log with bounded capacity
#[allow(dead_code)]
pub struct ReplicatedLog {
    entries: VecDeque<LogEntry>,
    /// Maximum entries retained before compaction
    max_size: usize,
    /// Index of the highest entry applied to the state machine
    commit_index: u64,
    /// Index of the last entry known to be applied
    last_applied: u64,
}

impl ReplicatedLog {
    /// Create a new log with the given maximum capacity
    #[allow(dead_code)]
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_size,
            commit_index: 0,
            last_applied: 0,
        }
    }

    /// Append an entry and return its assigned index.
    ///
    /// If the log exceeds `max_size`, the oldest entry is dropped (compaction).
    #[allow(dead_code)]
    pub fn append(&mut self, term: u64, command: impl Into<String>, data: Vec<u8>) -> u64 {
        let index = self.last_index() + 1;
        let entry = LogEntry::new(term, index, command, data);
        self.entries.push_back(entry);
        if self.entries.len() > self.max_size {
            self.entries.pop_front();
        }
        index
    }

    /// Return the index of the last entry (0 if empty).
    #[allow(dead_code)]
    #[must_use]
    pub fn last_index(&self) -> u64 {
        self.entries.back().map_or(0, |e| e.index)
    }

    /// Return the term of the last entry (0 if empty).
    #[allow(dead_code)]
    #[must_use]
    pub fn last_term(&self) -> u64 {
        self.entries.back().map_or(0, |e| e.term)
    }

    /// Advance the commit index up to `index`.
    #[allow(dead_code)]
    pub fn commit_up_to(&mut self, index: u64) {
        if index > self.commit_index {
            self.commit_index = index.min(self.last_index());
        }
    }

    /// Advance `last_applied` to match `commit_index` and return applied entries.
    #[allow(dead_code)]
    pub fn apply_committed(&mut self) -> Vec<LogEntry> {
        let mut applied = Vec::new();
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            if let Some(entry) = self.get(self.last_applied) {
                applied.push(entry.clone());
            }
        }
        applied
    }

    /// Get entry by 1-based index (may not exist after compaction).
    #[allow(dead_code)]
    #[must_use]
    pub fn get(&self, index: u64) -> Option<&LogEntry> {
        self.entries.iter().find(|e| e.index == index)
    }

    /// Return current commit index.
    #[allow(dead_code)]
    #[must_use]
    pub fn commit_index(&self) -> u64 {
        self.commit_index
    }

    /// Return the number of retained entries.
    #[allow(dead_code)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if no entries have been appended (or all were compacted away).
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Tracks the current term and votes for leader election
#[allow(dead_code)]
pub struct TermTracker {
    current_term: u64,
    voted_for: Option<String>,
}

impl TermTracker {
    /// Create a new tracker starting at term 0.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_term: 0,
            voted_for: None,
        }
    }

    /// Return the current term.
    #[allow(dead_code)]
    #[must_use]
    pub fn current_term(&self) -> u64 {
        self.current_term
    }

    /// Advance to a new term, clearing the vote.
    ///
    /// Returns `true` if the term was actually advanced.
    #[allow(dead_code)]
    pub fn advance_term(&mut self, new_term: u64) -> bool {
        if new_term > self.current_term {
            self.current_term = new_term;
            self.voted_for = None;
            true
        } else {
            false
        }
    }

    /// Attempt to cast a vote for `candidate_id` in the current term.
    ///
    /// Returns `true` if the vote was granted (can only vote once per term).
    #[allow(dead_code)]
    pub fn grant_vote(&mut self, candidate_id: impl Into<String>) -> bool {
        if self.voted_for.is_none() {
            self.voted_for = Some(candidate_id.into());
            true
        } else {
            false
        }
    }

    /// The candidate voted for in the current term, if any.
    #[allow(dead_code)]
    #[must_use]
    pub fn voted_for(&self) -> Option<&str> {
        self.voted_for.as_deref()
    }
}

impl Default for TermTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Quorum calculation utilities
#[allow(dead_code)]
pub struct QuorumHelper;

impl QuorumHelper {
    /// Minimum votes needed for a quorum given `cluster_size` nodes.
    #[allow(dead_code)]
    #[must_use]
    pub fn majority(cluster_size: usize) -> usize {
        cluster_size / 2 + 1
    }

    /// True if `votes` constitutes a quorum for `cluster_size`.
    #[allow(dead_code)]
    #[must_use]
    pub fn has_quorum(votes: usize, cluster_size: usize) -> bool {
        votes >= Self::majority(cluster_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_entry_new() {
        let e = LogEntry::new(1, 1, "set", b"data".to_vec());
        assert_eq!(e.term, 1);
        assert_eq!(e.index, 1);
        assert_eq!(e.command, "set");
    }

    #[test]
    fn test_replicated_log_append_increments_index() {
        let mut log = ReplicatedLog::new(100);
        let i1 = log.append(1, "cmd1", vec![]);
        let i2 = log.append(1, "cmd2", vec![]);
        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
    }

    #[test]
    fn test_replicated_log_last_index_and_term() {
        let mut log = ReplicatedLog::new(100);
        assert_eq!(log.last_index(), 0);
        assert_eq!(log.last_term(), 0);
        log.append(2, "cmd", vec![]);
        assert_eq!(log.last_index(), 1);
        assert_eq!(log.last_term(), 2);
    }

    #[test]
    fn test_replicated_log_compaction() {
        let mut log = ReplicatedLog::new(3);
        for _ in 0..5 {
            log.append(1, "x", vec![]);
        }
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn test_replicated_log_get_entry() {
        let mut log = ReplicatedLog::new(10);
        log.append(1, "cmd", b"hello".to_vec());
        let entry = log.get(1).expect("get should return a value");
        assert_eq!(entry.data, b"hello");
    }

    #[test]
    fn test_replicated_log_get_missing() {
        let log = ReplicatedLog::new(10);
        assert!(log.get(99).is_none());
    }

    #[test]
    fn test_commit_and_apply() {
        let mut log = ReplicatedLog::new(100);
        log.append(1, "a", vec![]);
        log.append(1, "b", vec![]);
        log.append(1, "c", vec![]);
        log.commit_up_to(2);
        let applied = log.apply_committed();
        assert_eq!(applied.len(), 2);
        assert_eq!(applied[0].command, "a");
        assert_eq!(applied[1].command, "b");
    }

    #[test]
    fn test_commit_up_to_capped_at_last_index() {
        let mut log = ReplicatedLog::new(100);
        log.append(1, "x", vec![]);
        log.commit_up_to(999);
        assert_eq!(log.commit_index(), 1);
    }

    #[test]
    fn test_term_tracker_initial_state() {
        let t = TermTracker::new();
        assert_eq!(t.current_term(), 0);
        assert!(t.voted_for().is_none());
    }

    #[test]
    fn test_term_tracker_advance_term() {
        let mut t = TermTracker::new();
        assert!(t.advance_term(3));
        assert_eq!(t.current_term(), 3);
        assert!(!t.advance_term(2)); // going backwards fails
    }

    #[test]
    fn test_term_tracker_grant_vote_once() {
        let mut t = TermTracker::new();
        t.advance_term(1);
        assert!(t.grant_vote("node-a"));
        assert!(!t.grant_vote("node-b")); // already voted
        assert_eq!(t.voted_for(), Some("node-a"));
    }

    #[test]
    fn test_term_tracker_vote_cleared_on_term_advance() {
        let mut t = TermTracker::new();
        t.advance_term(1);
        t.grant_vote("node-a");
        t.advance_term(2);
        assert!(t.voted_for().is_none());
    }

    #[test]
    fn test_quorum_majority_odd_cluster() {
        assert_eq!(QuorumHelper::majority(5), 3);
        assert_eq!(QuorumHelper::majority(3), 2);
    }

    #[test]
    fn test_quorum_majority_even_cluster() {
        assert_eq!(QuorumHelper::majority(4), 3);
    }

    #[test]
    fn test_has_quorum() {
        assert!(QuorumHelper::has_quorum(3, 5));
        assert!(!QuorumHelper::has_quorum(2, 5));
        assert!(QuorumHelper::has_quorum(2, 3));
    }
}
