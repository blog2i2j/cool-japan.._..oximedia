//! Content embargo and release scheduling.
//!
//! An embargo prevents a piece of content from being accessed in specified
//! territories until a configured release timestamp.  This module provides a
//! pure in-memory manager that is intentionally decoupled from the database
//! backend so it can be used in WASM and no-std-friendly contexts.
//!
//! # Design notes
//! * Embargoes are keyed by `content_id`.  Only one embargo entry per
//!   `content_id` is allowed; attempting to add a duplicate returns
//!   [`EmbargoError::AlreadyExists`].
//! * `territories` is an **allow-list** of affected territories.  An empty
//!   list means the embargo applies *worldwide*.
//! * A `notified` flag is maintained so that downstream notification systems
//!   (email, webhook, etc.) can mark an entry once they have sent their alert.

#![allow(missing_docs)]

use std::collections::HashMap;
use thiserror::Error;

// ── EmbargoError ──────────────────────────────────────────────────────────────

/// Errors returned by [`EmbargoManager`] operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EmbargoError {
    /// An embargo already exists for the given content identifier.
    #[error("embargo already exists for content: {0}")]
    AlreadyExists(String),

    /// No embargo was found for the given content identifier.
    #[error("embargo not found for content: {0}")]
    NotFound(String),

    /// The supplied release timestamp is invalid (e.g. zero or obviously wrong).
    #[error("invalid release time")]
    InvalidReleaseTime,
}

// ── EmbargoEntry ─────────────────────────────────────────────────────────────

/// A single embargo record for a piece of content.
#[derive(Debug, Clone)]
pub struct EmbargoEntry {
    /// Unique identifier of the embargoed content.
    pub content_id: String,
    /// Unix timestamp (seconds) at or after which the embargo is lifted.
    pub release_at_secs: u64,
    /// Territories where the embargo applies.  Empty = worldwide.
    pub territories: Vec<String>,
    /// Human-readable reason for the embargo.
    pub embargo_reason: String,
    /// Whether the rights-holder / downstream system has been notified.
    pub notified: bool,
}

// ── EmbargoStatus ─────────────────────────────────────────────────────────────

/// The current embargo status of a piece of content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbargoStatus {
    /// The content is still under embargo in the listed territories.
    Embargoed {
        /// Unix timestamp when the embargo lifts.
        release_at_secs: u64,
        /// Territories currently under embargo.
        territories: Vec<String>,
    },
    /// The embargo has been lifted globally (past the release timestamp).
    Released,
    /// The embargo has been lifted in the specified territory but may still
    /// apply elsewhere.  Returned when `territories` is non-empty and the
    /// requested territory is not in the list (i.e. it was never embargoed
    /// there).
    ReleasedInTerritory(String),
    /// The content has no embargo record (was never embargoed or was removed).
    Expired,
}

// ── EmbargoManager ───────────────────────────────────────────────────────────

/// In-memory manager for content embargoes and release scheduling.
#[derive(Debug, Default)]
pub struct EmbargoManager {
    /// Entries keyed by content_id.
    entries: HashMap<String, EmbargoEntry>,
}

impl EmbargoManager {
    /// Create a new, empty `EmbargoManager`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new embargo entry.
    ///
    /// # Errors
    /// * [`EmbargoError::AlreadyExists`] – an embargo already exists for the
    ///   content_id in `entry`.
    /// * [`EmbargoError::InvalidReleaseTime`] – `release_at_secs` is zero.
    pub fn add_embargo(&mut self, entry: EmbargoEntry) -> Result<(), EmbargoError> {
        if entry.release_at_secs == 0 {
            return Err(EmbargoError::InvalidReleaseTime);
        }
        if self.entries.contains_key(&entry.content_id) {
            return Err(EmbargoError::AlreadyExists(entry.content_id.clone()));
        }
        self.entries.insert(entry.content_id.clone(), entry);
        Ok(())
    }

    /// Remove an embargo entry.
    ///
    /// # Errors
    /// * [`EmbargoError::NotFound`] – no embargo exists for `content_id`.
    pub fn remove_embargo(&mut self, content_id: &str) -> Result<(), EmbargoError> {
        if self.entries.remove(content_id).is_none() {
            return Err(EmbargoError::NotFound(content_id.to_string()));
        }
        Ok(())
    }

    /// Return `true` if the content can be accessed in `territory` at
    /// `now_secs`.
    ///
    /// Access is **granted** when:
    /// * There is no embargo entry for the content, OR
    /// * The current time is at or after `release_at_secs`, OR
    /// * The embargo's territory list is non-empty and `territory` is not in
    ///   the list (the embargo only applies to listed territories).
    #[must_use]
    pub fn check_access(&self, content_id: &str, territory: &str, now_secs: u64) -> bool {
        match self.entries.get(content_id) {
            None => true,
            Some(entry) => {
                if now_secs >= entry.release_at_secs {
                    // Embargo time has passed → always grant access.
                    return true;
                }
                // If the territories list is non-empty and the requested
                // territory is NOT in it, the content is available there.
                !entry.territories.is_empty()
                    && !entry
                        .territories
                        .iter()
                        .any(|t| t.eq_ignore_ascii_case(territory))
            }
        }
    }

    /// Return the embargo status for `content_id` as observed at `now_secs`.
    #[must_use]
    pub fn status(&self, content_id: &str, now_secs: u64) -> EmbargoStatus {
        match self.entries.get(content_id) {
            None => EmbargoStatus::Expired,
            Some(entry) => {
                if now_secs >= entry.release_at_secs {
                    EmbargoStatus::Released
                } else {
                    EmbargoStatus::Embargoed {
                        release_at_secs: entry.release_at_secs,
                        territories: entry.territories.clone(),
                    }
                }
            }
        }
    }

    /// Return references to all embargo entries whose `release_at_secs` is
    /// at or before `now_secs` (i.e., content that should have been released
    /// by now but still has an active record).
    ///
    /// Callers should use this to drive release notifications or automatic
    /// record clean-up.
    #[must_use]
    pub fn due_for_release(&self, now_secs: u64) -> Vec<&EmbargoEntry> {
        self.entries
            .values()
            .filter(|e| e.release_at_secs <= now_secs)
            .collect()
    }

    /// Mark an embargo entry as notified.
    ///
    /// Silently does nothing if `content_id` is not found so that callers do
    /// not need to handle an error for a fire-and-forget operation.
    pub fn mark_notified(&mut self, content_id: &str) {
        if let Some(entry) = self.entries.get_mut(content_id) {
            entry.notified = true;
        }
    }

    /// Return a reference to an embargo entry, if it exists.
    #[must_use]
    pub fn get(&self, content_id: &str) -> Option<&EmbargoEntry> {
        self.entries.get(content_id)
    }

    /// Number of entries currently tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when there are no embargo entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_700_000_000;
    const FUTURE: u64 = 1_800_000_000;
    const PAST: u64 = 1_600_000_000;

    fn global_embargo(content_id: &str, release_at: u64) -> EmbargoEntry {
        EmbargoEntry {
            content_id: content_id.to_string(),
            release_at_secs: release_at,
            territories: vec![],
            embargo_reason: "pre-release".to_string(),
            notified: false,
        }
    }

    fn territorial_embargo(
        content_id: &str,
        release_at: u64,
        territories: Vec<&str>,
    ) -> EmbargoEntry {
        EmbargoEntry {
            content_id: content_id.to_string(),
            release_at_secs: release_at,
            territories: territories.iter().map(|t| t.to_string()).collect(),
            embargo_reason: "territorial restriction".to_string(),
            notified: false,
        }
    }

    // ── add / remove ──────────────────────────────────────────────────────────

    #[test]
    fn test_add_embargo_succeeds() {
        let mut mgr = EmbargoManager::new();
        let result = mgr.add_embargo(global_embargo("vid-1", FUTURE));
        assert!(result.is_ok());
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_add_embargo_duplicate_returns_error() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(global_embargo("vid-1", FUTURE)).unwrap();
        let err = mgr
            .add_embargo(global_embargo("vid-1", FUTURE))
            .unwrap_err();
        assert_eq!(err, EmbargoError::AlreadyExists("vid-1".to_string()));
    }

    #[test]
    fn test_add_embargo_zero_release_time_rejected() {
        let mut mgr = EmbargoManager::new();
        let err = mgr.add_embargo(global_embargo("vid-1", 0)).unwrap_err();
        assert_eq!(err, EmbargoError::InvalidReleaseTime);
    }

    #[test]
    fn test_remove_embargo_succeeds() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(global_embargo("vid-1", FUTURE)).unwrap();
        mgr.remove_embargo("vid-1").unwrap();
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_remove_embargo_not_found_returns_error() {
        let mut mgr = EmbargoManager::new();
        let err = mgr.remove_embargo("missing").unwrap_err();
        assert_eq!(err, EmbargoError::NotFound("missing".to_string()));
    }

    // ── check_access ──────────────────────────────────────────────────────────

    #[test]
    fn test_check_access_no_embargo_grants_access() {
        let mgr = EmbargoManager::new();
        assert!(mgr.check_access("vid-X", "US", NOW));
    }

    #[test]
    fn test_check_access_global_embargo_blocks_all_territories() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(global_embargo("vid-1", FUTURE)).unwrap();
        assert!(!mgr.check_access("vid-1", "US", NOW));
        assert!(!mgr.check_access("vid-1", "DE", NOW));
    }

    #[test]
    fn test_check_access_grants_after_release_time() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(global_embargo("vid-1", PAST)).unwrap();
        assert!(mgr.check_access("vid-1", "US", NOW));
    }

    #[test]
    fn test_check_access_territorial_embargo_blocks_listed_territory() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(territorial_embargo("vid-2", FUTURE, vec!["DE", "FR"]))
            .unwrap();
        assert!(!mgr.check_access("vid-2", "DE", NOW));
        assert!(!mgr.check_access("vid-2", "FR", NOW));
    }

    #[test]
    fn test_check_access_territorial_embargo_allows_unlisted_territory() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(territorial_embargo("vid-2", FUTURE, vec!["DE"]))
            .unwrap();
        // US is not in the embargo list → access allowed
        assert!(mgr.check_access("vid-2", "US", NOW));
    }

    // ── status ────────────────────────────────────────────────────────────────

    #[test]
    fn test_status_expired_when_no_entry() {
        let mgr = EmbargoManager::new();
        assert_eq!(mgr.status("missing", NOW), EmbargoStatus::Expired);
    }

    #[test]
    fn test_status_embargoed_before_release() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(territorial_embargo("vid-3", FUTURE, vec!["US"]))
            .unwrap();
        let s = mgr.status("vid-3", NOW);
        assert!(matches!(
            s,
            EmbargoStatus::Embargoed {
                release_at_secs: FUTURE,
                ..
            }
        ));
    }

    #[test]
    fn test_status_released_after_release_time() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(global_embargo("vid-4", PAST)).unwrap();
        assert_eq!(mgr.status("vid-4", NOW), EmbargoStatus::Released);
    }

    // ── due_for_release ───────────────────────────────────────────────────────

    #[test]
    fn test_due_for_release_finds_past_entries() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(global_embargo("vid-5", PAST)).unwrap();
        mgr.add_embargo(global_embargo("vid-6", FUTURE)).unwrap();
        let due = mgr.due_for_release(NOW);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].content_id, "vid-5");
    }

    #[test]
    fn test_due_for_release_empty_when_all_future() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(global_embargo("vid-7", FUTURE)).unwrap();
        let due = mgr.due_for_release(NOW);
        assert!(due.is_empty());
    }

    // ── mark_notified ─────────────────────────────────────────────────────────

    #[test]
    fn test_mark_notified_sets_flag() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(global_embargo("vid-8", FUTURE)).unwrap();
        assert!(!mgr.get("vid-8").expect("entry exists").notified);
        mgr.mark_notified("vid-8");
        assert!(mgr.get("vid-8").expect("entry exists").notified);
    }

    #[test]
    fn test_mark_notified_on_unknown_id_is_noop() {
        let mut mgr = EmbargoManager::new();
        // Should not panic or return an error.
        mgr.mark_notified("nonexistent");
        assert!(mgr.is_empty());
    }
}
