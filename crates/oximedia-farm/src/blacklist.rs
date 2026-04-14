//! Worker blacklisting for fault isolation.
//!
//! [`WorkerBlacklist`] maintains a set of worker IDs that are temporarily or
//! permanently banned from receiving new job assignments.  Reasons (e.g.
//! repeated failures, health-check timeouts, security violations) are stored
//! alongside each entry for operator inspection.
//!
//! # Example
//!
//! ```
//! use oximedia_farm::blacklist::WorkerBlacklist;
//!
//! let mut bl = WorkerBlacklist::new();
//! bl.add(42, "repeated transcode failures");
//!
//! assert!(bl.is_blocked(42));
//! assert!(!bl.is_blocked(7));
//!
//! bl.remove(42);
//! assert!(!bl.is_blocked(42));
//! ```

use std::collections::HashMap;

/// Blacklist entry for a single worker.
#[derive(Debug, Clone)]
pub struct BlacklistEntry {
    /// Worker identifier.
    pub worker_id: u64,
    /// Human-readable reason for blacklisting.
    pub reason: String,
}

/// Per-worker blacklist with add / query / remove operations.
///
/// The blacklist stores one entry per `worker_id`; calling [`add`](Self::add)
/// on an already-blacklisted worker updates the stored reason.
#[derive(Debug, Clone, Default)]
pub struct WorkerBlacklist {
    entries: HashMap<u64, BlacklistEntry>,
}

impl WorkerBlacklist {
    /// Create an empty blacklist.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or update) a worker to the blacklist with the given reason.
    ///
    /// If the worker is already blacklisted its reason is replaced.
    pub fn add(&mut self, worker_id: u64, reason: &str) {
        self.entries.insert(
            worker_id,
            BlacklistEntry {
                worker_id,
                reason: reason.to_string(),
            },
        );
    }

    /// Return `true` when `worker_id` is currently blacklisted.
    #[must_use]
    pub fn is_blocked(&self, worker_id: u64) -> bool {
        self.entries.contains_key(&worker_id)
    }

    /// Remove a worker from the blacklist.
    ///
    /// This is a no-op when the worker is not present.
    pub fn remove(&mut self, worker_id: u64) {
        self.entries.remove(&worker_id);
    }

    /// Return the reason a worker was blacklisted, or `None` if not blocked.
    #[must_use]
    pub fn reason(&self, worker_id: u64) -> Option<&str> {
        self.entries.get(&worker_id).map(|e| e.reason.as_str())
    }

    /// Return the number of blacklisted workers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no workers are blacklisted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all blacklisted entries (order unspecified).
    pub fn iter(&self) -> impl Iterator<Item = &BlacklistEntry> {
        self.entries.values()
    }

    /// Clear all entries (unblock all workers).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Filter a slice of worker IDs, returning only those that are **not**
    /// blacklisted.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_farm::blacklist::WorkerBlacklist;
    ///
    /// let mut bl = WorkerBlacklist::new();
    /// bl.add(2, "disk full");
    /// let available = bl.filter_available(&[1, 2, 3]);
    /// assert_eq!(available, vec![1, 3]);
    /// ```
    #[must_use]
    pub fn filter_available(&self, worker_ids: &[u64]) -> Vec<u64> {
        worker_ids
            .iter()
            .filter(|&&id| !self.is_blocked(id))
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let bl = WorkerBlacklist::new();
        assert!(bl.is_empty());
        assert_eq!(bl.len(), 0);
    }

    #[test]
    fn test_add_and_is_blocked() {
        let mut bl = WorkerBlacklist::new();
        bl.add(42, "too many errors");
        assert!(bl.is_blocked(42));
        assert!(!bl.is_blocked(7));
    }

    #[test]
    fn test_remove_unblocks() {
        let mut bl = WorkerBlacklist::new();
        bl.add(1, "test");
        bl.remove(1);
        assert!(!bl.is_blocked(1));
        assert_eq!(bl.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_noop() {
        let mut bl = WorkerBlacklist::new();
        bl.remove(999); // should not panic
        assert!(bl.is_empty());
    }

    #[test]
    fn test_reason_returned() {
        let mut bl = WorkerBlacklist::new();
        bl.add(5, "disk full");
        assert_eq!(bl.reason(5), Some("disk full"));
        assert!(bl.reason(99).is_none());
    }

    #[test]
    fn test_reason_updated_on_readd() {
        let mut bl = WorkerBlacklist::new();
        bl.add(3, "first reason");
        bl.add(3, "updated reason");
        assert_eq!(bl.reason(3), Some("updated reason"));
        assert_eq!(bl.len(), 1); // still only one entry
    }

    #[test]
    fn test_len_tracks_count() {
        let mut bl = WorkerBlacklist::new();
        bl.add(1, "a");
        bl.add(2, "b");
        bl.add(3, "c");
        assert_eq!(bl.len(), 3);
        bl.remove(2);
        assert_eq!(bl.len(), 2);
    }

    #[test]
    fn test_filter_available() {
        let mut bl = WorkerBlacklist::new();
        bl.add(2, "quarantined");
        let available = bl.filter_available(&[1, 2, 3, 4]);
        assert_eq!(available, vec![1, 3, 4]);
    }

    #[test]
    fn test_filter_available_all_blocked() {
        let mut bl = WorkerBlacklist::new();
        bl.add(1, "a");
        bl.add(2, "b");
        assert!(bl.filter_available(&[1, 2]).is_empty());
    }

    #[test]
    fn test_filter_available_none_blocked() {
        let bl = WorkerBlacklist::new();
        let result = bl.filter_available(&[10, 20, 30]);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn test_clear() {
        let mut bl = WorkerBlacklist::new();
        bl.add(1, "a");
        bl.add(2, "b");
        bl.clear();
        assert!(bl.is_empty());
    }

    #[test]
    fn test_iter_count() {
        let mut bl = WorkerBlacklist::new();
        bl.add(1, "x");
        bl.add(2, "y");
        assert_eq!(bl.iter().count(), 2);
    }
}
