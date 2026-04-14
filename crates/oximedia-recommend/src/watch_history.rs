//! Watch history tracking and genre-preference analysis.
//!
//! [`WatchHistory`] maintains a compact per-user view record and derives
//! genre-level engagement from accumulated watch durations.  The module is
//! intentionally self-contained (no I/O, no async) so it can be embedded in
//! both online and offline recommendation pipelines.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ViewRecord
// ---------------------------------------------------------------------------

/// A single content view event.
#[derive(Debug, Clone)]
pub struct ViewRecord {
    /// Content identifier.
    pub content_id: u64,
    /// Total seconds watched in this session.
    pub duration_s: f64,
    /// Unix timestamp (seconds) when the view started.
    pub started_at: i64,
    /// Metadata attached at record time (e.g. `genre`, `title`).
    pub metadata: HashMap<String, String>,
}

impl ViewRecord {
    /// Creates a minimal view record with no metadata.
    #[must_use]
    pub fn new(content_id: u64, duration_s: f64) -> Self {
        Self {
            content_id,
            duration_s,
            started_at: 0,
            metadata: HashMap::new(),
        }
    }

    /// Attaches a key-value metadata entry (builder style).
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Attaches a start timestamp (builder style).
    #[must_use]
    pub fn with_started_at(mut self, ts: i64) -> Self {
        self.started_at = ts;
        self
    }
}

// ---------------------------------------------------------------------------
// WatchHistory
// ---------------------------------------------------------------------------

/// Accumulates view events and derives aggregated viewing statistics.
///
/// Genre information is extracted from the `"genre"` metadata key of each
/// [`ViewRecord`].  When no genre metadata is present, views are aggregated
/// under the genre label `"unknown"`.
pub struct WatchHistory {
    /// All recorded views in insertion order.
    views: Vec<ViewRecord>,
    /// Accumulated watch time (seconds) keyed by genre.
    genre_watch_time: HashMap<String, f64>,
}

impl WatchHistory {
    /// Creates a new, empty watch history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            views: Vec::new(),
            genre_watch_time: HashMap::new(),
        }
    }

    /// Appends a view event.
    ///
    /// Any `"genre"` metadata attached to the record is used to attribute the
    /// watch duration to that genre bucket.
    pub fn add_view(&mut self, content_id: u64, duration_s: f64) {
        let record = ViewRecord::new(content_id, duration_s.max(0.0));
        self.ingest_record(record);
    }

    /// Appends a fully-specified [`ViewRecord`].
    ///
    /// Use this variant when you want to include genre metadata or a
    /// timestamp.
    pub fn add_record(&mut self, record: ViewRecord) {
        self.ingest_record(record);
    }

    /// Returns the content IDs viewed, ordered by total accumulated watch time
    /// (descending).
    ///
    /// If a content item was viewed multiple times the durations are summed.
    #[must_use]
    pub fn top_content(&self, limit: usize) -> Vec<u64> {
        let mut by_content: HashMap<u64, f64> = HashMap::new();
        for v in &self.views {
            *by_content.entry(v.content_id).or_insert(0.0) += v.duration_s;
        }
        let mut entries: Vec<(u64, f64)> = by_content.into_iter().collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.into_iter().take(limit).map(|(id, _)| id).collect()
    }

    /// Returns genre labels sorted by accumulated watch time (descending).
    ///
    /// Genres are extracted from the `"genre"` metadata field of each record.
    /// Records without that field contribute to `"unknown"`.
    #[must_use]
    pub fn most_watched_genres(&self) -> Vec<String> {
        let mut entries: Vec<(String, f64)> = self.genre_watch_time.clone().into_iter().collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.into_iter().map(|(g, _)| g).collect()
    }

    /// Total number of view events recorded.
    #[must_use]
    pub fn view_count(&self) -> usize {
        self.views.len()
    }

    /// Total watch time across all views (seconds).
    #[must_use]
    pub fn total_watch_time_s(&self) -> f64 {
        self.views.iter().map(|v| v.duration_s).sum()
    }

    /// Accumulated watch time for a specific genre (seconds).
    ///
    /// Returns `0.0` when no views with that genre label have been recorded.
    #[must_use]
    pub fn genre_time_s(&self, genre: &str) -> f64 {
        self.genre_watch_time.get(genre).copied().unwrap_or(0.0)
    }

    /// Returns an iterator over all recorded view events.
    pub fn iter(&self) -> impl Iterator<Item = &ViewRecord> {
        self.views.iter()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn ingest_record(&mut self, record: ViewRecord) {
        let genre = record
            .metadata
            .get("genre")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        *self.genre_watch_time.entry(genre).or_insert(0.0) += record.duration_s;
        self.views.push(record);
    }
}

impl Default for WatchHistory {
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
    fn test_new_watch_history_empty() {
        let h = WatchHistory::new();
        assert_eq!(h.view_count(), 0);
        assert!((h.total_watch_time_s()).abs() < f64::EPSILON);
        assert!(h.most_watched_genres().is_empty());
    }

    #[test]
    fn test_add_view_increments_count() {
        let mut h = WatchHistory::new();
        h.add_view(1, 120.0);
        h.add_view(2, 60.0);
        assert_eq!(h.view_count(), 2);
    }

    #[test]
    fn test_total_watch_time() {
        let mut h = WatchHistory::new();
        h.add_view(1, 100.0);
        h.add_view(2, 200.0);
        assert!((h.total_watch_time_s() - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_most_watched_genres_ordered() {
        let mut h = WatchHistory::new();
        h.add_record(ViewRecord::new(1, 500.0).with_meta("genre", "sci-fi"));
        h.add_record(ViewRecord::new(2, 100.0).with_meta("genre", "comedy"));
        h.add_record(ViewRecord::new(3, 300.0).with_meta("genre", "action"));

        let genres = h.most_watched_genres();
        assert_eq!(genres[0], "sci-fi");
        assert_eq!(genres[1], "action");
        assert_eq!(genres[2], "comedy");
    }

    #[test]
    fn test_genre_accumulation_across_views() {
        let mut h = WatchHistory::new();
        h.add_record(ViewRecord::new(1, 100.0).with_meta("genre", "drama"));
        h.add_record(ViewRecord::new(2, 200.0).with_meta("genre", "drama"));
        assert!((h.genre_time_s("drama") - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_views_without_genre_go_to_unknown() {
        let mut h = WatchHistory::new();
        h.add_view(42, 90.0);
        let genres = h.most_watched_genres();
        assert_eq!(genres.len(), 1);
        assert_eq!(genres[0], "unknown");
    }

    #[test]
    fn test_top_content_ordering() {
        let mut h = WatchHistory::new();
        h.add_view(10, 300.0);
        h.add_view(20, 600.0);
        h.add_view(10, 100.0); // 2nd view of content 10 → total 400 s
        let top = h.top_content(2);
        assert_eq!(top[0], 20, "content 20 (600s) should be first");
        assert_eq!(top[1], 10, "content 10 (400s) should be second");
    }

    #[test]
    fn test_top_content_limit() {
        let mut h = WatchHistory::new();
        for i in 0..10u64 {
            h.add_view(i, (i + 1) as f64 * 60.0);
        }
        assert_eq!(h.top_content(3).len(), 3);
    }

    #[test]
    fn test_negative_duration_clamped_to_zero() {
        let mut h = WatchHistory::new();
        h.add_view(1, -50.0);
        assert!((h.total_watch_time_s()).abs() < f64::EPSILON);
    }
}
