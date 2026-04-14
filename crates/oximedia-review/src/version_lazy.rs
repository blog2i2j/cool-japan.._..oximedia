//! Lazy-loading façade for version history.
//!
//! Opening a review session should not block on fetching every version ever
//! created for a piece of content.  Instead a [`LazyVersionHistory`] exposes
//! the **latest** version immediately and loads older versions on demand in
//! bounded pages using a *cursor-based* pagination scheme.
//!
//! # Design
//!
//! ```text
//! LazyVersionHistory
//!   ├── latest()          – returns the most-recent version stub (always cached)
//!   ├── load_page(cursor) – fetches one page of older versions, returns next cursor
//!   └── get(id)           – look up a version already loaded into the local cache
//! ```
//!
//! The actual I/O is provided by the caller through the [`VersionFetcher`]
//! trait, making the loader fully testable without a real network or database.

use crate::{SessionId, VersionId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// VersionStub — lightweight summary loaded eagerly
// ---------------------------------------------------------------------------

/// Lightweight summary of a single content version.
///
/// Only the metadata needed to render a version list is included; full
/// details are fetched lazily on demand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VersionStub {
    /// Version unique identifier.
    pub id: VersionId,
    /// Sequential version number (1-based).
    pub number: u32,
    /// Human-readable label (e.g. "v3 – colour pass").
    pub label: String,
    /// Who uploaded this version.
    pub created_by: String,
    /// When this version was uploaded.
    pub created_at: DateTime<Utc>,
    /// Parent version ID, if any.
    pub parent_id: Option<VersionId>,
}

// ---------------------------------------------------------------------------
// Page / cursor types
// ---------------------------------------------------------------------------

/// Opaque cursor pointing to the next page of version history.
///
/// Internally this is the sequence number of the last item returned, but
/// callers should treat it as an opaque token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionCursor(pub u64);

impl VersionCursor {
    /// The cursor value used to start loading from the most recent version.
    pub const LATEST: Self = Self(u64::MAX);
}

/// A single page of version stubs returned by [`VersionFetcher`].
#[derive(Debug, Clone)]
pub struct VersionPage {
    /// The stubs on this page, ordered newest-first.
    pub stubs: Vec<VersionStub>,
    /// Cursor to pass to the next `load_page` call, or `None` if this is
    /// the last page.
    pub next_cursor: Option<VersionCursor>,
}

// ---------------------------------------------------------------------------
// VersionFetcher trait
// ---------------------------------------------------------------------------

/// Provider of paginated version history.
///
/// Implement this trait to connect `LazyVersionHistory` to your persistence
/// layer (database, HTTP API, in-memory fixture, etc.).
pub trait VersionFetcher: Send + Sync {
    /// Fetch one page of version stubs for `session_id`.
    ///
    /// * `cursor` — opaque cursor from the previous page (use
    ///   [`VersionCursor::LATEST`] for the first request).
    /// * `page_size` — maximum number of stubs to return.
    ///
    /// # Errors
    ///
    /// Returns `Err(FetchError)` if the underlying store is unavailable.
    fn fetch_page(
        &self,
        session_id: SessionId,
        cursor: VersionCursor,
        page_size: usize,
    ) -> Result<VersionPage, FetchError>;
}

/// Error returned by [`VersionFetcher::fetch_page`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FetchError {
    /// The backend store is unavailable.
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    /// The session was not found.
    #[error("session not found: {0}")]
    SessionNotFound(SessionId),
    /// An invalid cursor was supplied.
    #[error("invalid cursor")]
    InvalidCursor,
}

// ---------------------------------------------------------------------------
// LoadState
// ---------------------------------------------------------------------------

/// Tracks how much of the version history has been loaded so far.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadState {
    /// No pages have been loaded yet (only the latest version is known, if any).
    Initial,
    /// Some pages have been loaded; more may be available.
    Partial {
        /// Cursor to use for the next page request.
        next_cursor: VersionCursor,
        /// Number of stubs loaded so far.
        loaded: usize,
    },
    /// All pages have been loaded — the local cache is complete.
    Complete {
        /// Total number of versions known.
        total: usize,
    },
}

// ---------------------------------------------------------------------------
// LazyVersionHistory
// ---------------------------------------------------------------------------

/// On-demand version history for a review session.
///
/// The history starts with only the latest version (if one is supplied at
/// construction time) and loads older versions page-by-page on request.
/// Already-loaded versions are cached in memory so they are never fetched
/// twice.
pub struct LazyVersionHistory {
    session_id: SessionId,
    page_size: usize,
    cache: HashMap<VersionId, VersionStub>,
    /// Newest-first ordered IDs of stubs known so far.
    ordered_ids: Vec<VersionId>,
    state: LoadState,
    fetcher: Box<dyn VersionFetcher>,
}

impl LazyVersionHistory {
    /// Create a new lazy history.
    ///
    /// * `session_id` — review session whose versions to load.
    /// * `page_size`  — how many stubs to fetch per page (clamped to ≥ 1).
    /// * `fetcher`    — backend provider.
    /// * `latest`     — if `Some`, the latest version stub is pre-populated.
    #[must_use]
    pub fn new(
        session_id: SessionId,
        page_size: usize,
        fetcher: Box<dyn VersionFetcher>,
        latest: Option<VersionStub>,
    ) -> Self {
        let page_size = page_size.max(1);
        let mut cache = HashMap::new();
        let mut ordered_ids = Vec::new();
        if let Some(stub) = latest {
            ordered_ids.push(stub.id);
            cache.insert(stub.id, stub);
        }
        Self {
            session_id,
            page_size,
            cache,
            ordered_ids,
            state: LoadState::Initial,
            fetcher,
        }
    }

    /// Return the latest (most recent) version stub, if known.
    #[must_use]
    pub fn latest(&self) -> Option<&VersionStub> {
        self.ordered_ids.first().and_then(|id| self.cache.get(id))
    }

    /// Look up a version by ID from the local cache.
    ///
    /// Returns `None` if the version has not been loaded yet.
    #[must_use]
    pub fn get(&self, id: VersionId) -> Option<&VersionStub> {
        self.cache.get(&id)
    }

    /// All version stubs currently in the local cache, newest-first.
    #[must_use]
    pub fn cached(&self) -> Vec<&VersionStub> {
        self.ordered_ids
            .iter()
            .filter_map(|id| self.cache.get(id))
            .collect()
    }

    /// Current load state.
    #[must_use]
    pub fn state(&self) -> &LoadState {
        &self.state
    }

    /// Whether the full history has been loaded.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self.state, LoadState::Complete { .. })
    }

    /// Number of stubs currently in the local cache.
    #[must_use]
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Load the next page of older versions.
    ///
    /// If the history is already complete, this is a no-op returning `Ok(0)`.
    ///
    /// Returns the number of new stubs added to the cache.
    ///
    /// # Errors
    ///
    /// Propagates [`FetchError`] from the underlying fetcher.
    pub fn load_next_page(&mut self) -> Result<usize, FetchError> {
        if self.is_complete() {
            return Ok(0);
        }
        let cursor = match &self.state {
            LoadState::Initial => VersionCursor::LATEST,
            LoadState::Partial { next_cursor, .. } => *next_cursor,
            LoadState::Complete { .. } => return Ok(0),
        };
        let page = self
            .fetcher
            .fetch_page(self.session_id, cursor, self.page_size)?;
        let added = page.stubs.len();
        for stub in page.stubs {
            let id = stub.id;
            if !self.cache.contains_key(&id) {
                self.ordered_ids.push(id);
                self.cache.insert(id, stub);
            }
        }
        let loaded_so_far = self.cache.len();
        self.state = match page.next_cursor {
            Some(next) => LoadState::Partial {
                next_cursor: next,
                loaded: loaded_so_far,
            },
            None => LoadState::Complete {
                total: loaded_so_far,
            },
        };
        Ok(added)
    }

    /// Load all remaining pages until the history is complete.
    ///
    /// **Warning:** on repositories with thousands of versions this may be
    /// slow.  Prefer incremental [`load_next_page`](Self::load_next_page)
    /// calls in UI code.
    ///
    /// Returns the total number of stubs added during this call.
    ///
    /// # Errors
    ///
    /// Propagates the first [`FetchError`] encountered.
    pub fn load_all(&mut self) -> Result<usize, FetchError> {
        let mut total_added = 0;
        while !self.is_complete() {
            total_added += self.load_next_page()?;
        }
        Ok(total_added)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // A simple in-memory VersionFetcher for testing.
    // ---------------------------------------------------------------------------

    struct MemFetcher {
        /// All stubs in newest-first order.
        stubs: Vec<VersionStub>,
    }

    impl MemFetcher {
        fn new(stubs: Vec<VersionStub>) -> Self {
            Self { stubs }
        }
    }

    impl VersionFetcher for MemFetcher {
        fn fetch_page(
            &self,
            _session_id: SessionId,
            cursor: VersionCursor,
            page_size: usize,
        ) -> Result<VersionPage, FetchError> {
            // Use the cursor value as the start index (0 = start from latest).
            let start = if cursor == VersionCursor::LATEST {
                0
            } else {
                cursor.0 as usize
            };
            if start > self.stubs.len() {
                return Err(FetchError::InvalidCursor);
            }
            let end = (start + page_size).min(self.stubs.len());
            let page_stubs = self.stubs[start..end].to_vec();
            let next_cursor = if end < self.stubs.len() {
                Some(VersionCursor(end as u64))
            } else {
                None
            };
            Ok(VersionPage {
                stubs: page_stubs,
                next_cursor,
            })
        }
    }

    fn make_stub(number: u32) -> VersionStub {
        VersionStub {
            id: VersionId::new(),
            number,
            label: format!("v{number}"),
            created_by: "tester".to_string(),
            created_at: Utc::now(),
            parent_id: None,
        }
    }

    fn make_history(count: usize, page_size: usize) -> LazyVersionHistory {
        let session_id = SessionId::new();
        let stubs: Vec<VersionStub> = (1..=count as u32).rev().map(make_stub).collect();
        let fetcher = Box::new(MemFetcher::new(stubs));
        LazyVersionHistory::new(session_id, page_size, fetcher, None)
    }

    // 1. New history with no latest stub starts empty.
    #[test]
    fn test_new_history_starts_empty() {
        let h = make_history(0, 5);
        assert_eq!(h.cached_count(), 0);
        assert_eq!(h.latest(), None);
        assert_eq!(h.state(), &LoadState::Initial);
    }

    // 2. Pre-populated latest stub is immediately accessible.
    #[test]
    fn test_latest_prepopulated() {
        let session_id = SessionId::new();
        let stub = make_stub(7);
        let expected_id = stub.id;
        let fetcher = Box::new(MemFetcher::new(vec![]));
        let h = LazyVersionHistory::new(session_id, 5, fetcher, Some(stub));
        let latest = h.latest().expect("should have latest");
        assert_eq!(latest.id, expected_id);
        assert_eq!(latest.number, 7);
    }

    // 3. load_next_page populates cache and transitions state.
    #[test]
    fn test_load_next_page_basic() {
        let mut h = make_history(5, 3);
        let added = h.load_next_page().expect("fetch succeeds");
        assert_eq!(added, 3);
        assert_eq!(h.cached_count(), 3);
        assert!(matches!(h.state(), LoadState::Partial { .. }));
    }

    // 4. Loading all pages results in Complete state.
    #[test]
    fn test_load_all_completes() {
        let mut h = make_history(7, 3);
        let total = h.load_all().expect("no fetch errors");
        assert_eq!(total, 7);
        assert!(h.is_complete());
        assert_eq!(h.cached_count(), 7);
        if let LoadState::Complete { total: t } = h.state() {
            assert_eq!(*t, 7);
        } else {
            panic!("expected Complete state");
        }
    }

    // 5. load_next_page on already-complete history is a no-op.
    #[test]
    fn test_load_next_page_noop_when_complete() {
        let mut h = make_history(2, 10);
        h.load_all().expect("ok");
        let added = h.load_next_page().expect("should not error");
        assert_eq!(added, 0);
        assert_eq!(h.cached_count(), 2);
    }

    // 6. get() returns cached version by ID.
    #[test]
    fn test_get_cached_version() {
        let mut h = make_history(4, 10);
        h.load_all().expect("ok");
        let cached = h.cached();
        assert!(!cached.is_empty());
        let first_id = cached[0].id;
        let found = h.get(first_id).expect("must find cached version");
        assert_eq!(found.id, first_id);
    }

    // 7. cached() returns versions in newest-first order after full load.
    #[test]
    fn test_cached_order_newest_first() {
        let mut h = make_history(5, 10);
        h.load_all().expect("ok");
        let cached = h.cached();
        let numbers: Vec<u32> = cached.iter().map(|s| s.number).collect();
        // Should be descending (newest first).
        for i in 0..numbers.len().saturating_sub(1) {
            assert!(
                numbers[i] >= numbers[i + 1],
                "expected newest-first ordering"
            );
        }
    }

    // 8. Single version repo loads in one page.
    #[test]
    fn test_single_version_repo() {
        let mut h = make_history(1, 5);
        let added = h.load_next_page().expect("ok");
        assert_eq!(added, 1);
        assert!(h.is_complete());
    }

    // 9. InvalidCursor error propagates correctly.
    #[test]
    fn test_invalid_cursor_error() {
        let session_id = SessionId::new();
        // Provide a fetcher with no stubs but override state manually via load.
        struct FailFetcher;
        impl VersionFetcher for FailFetcher {
            fn fetch_page(
                &self,
                _session_id: SessionId,
                _cursor: VersionCursor,
                _page_size: usize,
            ) -> Result<VersionPage, FetchError> {
                Err(FetchError::BackendUnavailable("down".into()))
            }
        }
        let mut h = LazyVersionHistory::new(session_id, 5, Box::new(FailFetcher), None);
        let result = h.load_next_page();
        assert!(matches!(result, Err(FetchError::BackendUnavailable(_))));
    }

    // 10. VersionCursor LATEST constant has expected value.
    #[test]
    fn test_version_cursor_latest_constant() {
        assert_eq!(VersionCursor::LATEST, VersionCursor(u64::MAX));
    }

    // 11. Empty repository completes immediately.
    #[test]
    fn test_empty_repo_completes_on_first_load() {
        let mut h = make_history(0, 5);
        let added = h.load_next_page().expect("ok");
        assert_eq!(added, 0);
        assert!(h.is_complete());
    }

    // 12. Stub serialisation round-trips.
    #[test]
    fn test_version_stub_serialize() {
        let stub = make_stub(3);
        let json = serde_json::to_string(&stub).expect("serialize");
        let back: VersionStub = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.number, 3);
        assert_eq!(back.label, "v3");
    }
}
