#![allow(dead_code)]
//! Track content impressions and compute click-through / engagement rates.
//!
//! Every time a recommended item is shown to a user it counts as an
//! impression. If the user clicks or interacts with the item, that is a
//! click. This module records impressions and clicks, computes CTR
//! (click-through rate) per content item and per user, and exposes
//! aggregated metrics for recommendation quality evaluation.

use std::collections::HashMap;

/// Unique identifier for an impression event.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImpressionId(pub String);

impl std::fmt::Display for ImpressionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Position in the recommendation list where the item was shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position(pub u32);

/// A single impression event.
#[derive(Debug, Clone)]
pub struct Impression {
    /// Unique impression ID.
    pub id: ImpressionId,
    /// User who saw the impression.
    pub user_id: String,
    /// Content item that was shown.
    pub content_id: String,
    /// Position in the recommendation list (0-indexed).
    pub position: Position,
    /// Timestamp of the impression (epoch millis).
    pub timestamp_ms: i64,
    /// Whether the user clicked/interacted.
    pub clicked: bool,
    /// Dwell time in milliseconds (0 if not clicked).
    pub dwell_time_ms: u64,
}

/// Aggregated metrics for a single content item.
#[derive(Debug, Clone)]
pub struct ContentMetrics {
    /// Content ID.
    pub content_id: String,
    /// Total impressions.
    pub impressions: u64,
    /// Total clicks.
    pub clicks: u64,
    /// Average position when shown.
    pub avg_position: f64,
    /// Average dwell time in millis (among clicked impressions).
    pub avg_dwell_ms: f64,
}

impl ContentMetrics {
    /// Click-through rate (0.0-1.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn ctr(&self) -> f64 {
        if self.impressions == 0 {
            return 0.0;
        }
        self.clicks as f64 / self.impressions as f64
    }
}

/// Aggregated metrics for a single user.
#[derive(Debug, Clone)]
pub struct UserMetrics {
    /// User ID.
    pub user_id: String,
    /// Total impressions shown.
    pub impressions: u64,
    /// Total clicks.
    pub clicks: u64,
    /// Number of distinct content items shown.
    pub unique_items: usize,
}

impl UserMetrics {
    /// Click-through rate for this user.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn ctr(&self) -> f64 {
        if self.impressions == 0 {
            return 0.0;
        }
        self.clicks as f64 / self.impressions as f64
    }
}

/// Global impression statistics.
#[derive(Debug, Clone, Default)]
pub struct ImpressionStats {
    /// Total impressions recorded.
    pub total_impressions: u64,
    /// Total clicks recorded.
    pub total_clicks: u64,
    /// Number of distinct users.
    pub distinct_users: usize,
    /// Number of distinct content items.
    pub distinct_items: usize,
    /// Global CTR.
    pub global_ctr: f64,
}

/// Tracks impressions and computes engagement metrics.
#[derive(Debug)]
pub struct ImpressionTracker {
    /// All impressions indexed by impression ID.
    impressions: HashMap<String, Impression>,
    /// Per-content counters: (impressions, clicks, `sum_position`, `sum_dwell`).
    content_counters: HashMap<String, (u64, u64, u64, u64)>,
    /// Per-user counters: (impressions, clicks, `unique_items` set size tracking).
    user_counters: HashMap<String, (u64, u64, HashMap<String, bool>)>,
    /// Next auto-generated impression ID.
    next_id: u64,
}

impl ImpressionTracker {
    /// Create a new impression tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            impressions: HashMap::new(),
            content_counters: HashMap::new(),
            user_counters: HashMap::new(),
            next_id: 0,
        }
    }

    /// Record an impression (shown but not clicked).
    pub fn record_impression(
        &mut self,
        user_id: &str,
        content_id: &str,
        position: u32,
        timestamp_ms: i64,
    ) -> ImpressionId {
        let id = ImpressionId(format!("imp_{}", self.next_id));
        self.next_id += 1;

        let impression = Impression {
            id: id.clone(),
            user_id: user_id.to_string(),
            content_id: content_id.to_string(),
            position: Position(position),
            timestamp_ms,
            clicked: false,
            dwell_time_ms: 0,
        };

        // Update content counters.
        let entry = self
            .content_counters
            .entry(content_id.to_string())
            .or_insert((0, 0, 0, 0));
        entry.0 += 1;
        entry.2 += u64::from(position);

        // Update user counters.
        let user_entry = self
            .user_counters
            .entry(user_id.to_string())
            .or_insert_with(|| (0, 0, HashMap::new()));
        user_entry.0 += 1;
        user_entry.2.insert(content_id.to_string(), true);

        self.impressions.insert(id.0.clone(), impression);
        id
    }

    /// Record a click on an existing impression.
    pub fn record_click(&mut self, impression_id: &str, dwell_time_ms: u64) -> bool {
        let Some(imp) = self.impressions.get_mut(impression_id) else {
            return false;
        };
        if imp.clicked {
            return false; // Already clicked.
        }
        imp.clicked = true;
        imp.dwell_time_ms = dwell_time_ms;

        // Update content counters.
        if let Some(entry) = self.content_counters.get_mut(&imp.content_id) {
            entry.1 += 1;
            entry.3 += dwell_time_ms;
        }

        // Update user counters.
        if let Some(user_entry) = self.user_counters.get_mut(&imp.user_id) {
            user_entry.1 += 1;
        }

        true
    }

    /// Get metrics for a specific content item.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn content_metrics(&self, content_id: &str) -> Option<ContentMetrics> {
        let &(impressions, clicks, sum_pos, sum_dwell) = self.content_counters.get(content_id)?;
        let avg_position = if impressions > 0 {
            sum_pos as f64 / impressions as f64
        } else {
            0.0
        };
        let avg_dwell_ms = if clicks > 0 {
            sum_dwell as f64 / clicks as f64
        } else {
            0.0
        };
        Some(ContentMetrics {
            content_id: content_id.to_string(),
            impressions,
            clicks,
            avg_position,
            avg_dwell_ms,
        })
    }

    /// Get metrics for a specific user.
    #[must_use]
    pub fn user_metrics(&self, user_id: &str) -> Option<UserMetrics> {
        let (impressions, clicks, ref items) = *self.user_counters.get(user_id)?;
        Some(UserMetrics {
            user_id: user_id.to_string(),
            impressions,
            clicks,
            unique_items: items.len(),
        })
    }

    /// Get global statistics.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn global_stats(&self) -> ImpressionStats {
        let total_impressions: u64 = self.content_counters.values().map(|c| c.0).sum();
        let total_clicks: u64 = self.content_counters.values().map(|c| c.1).sum();
        let global_ctr = if total_impressions > 0 {
            total_clicks as f64 / total_impressions as f64
        } else {
            0.0
        };
        ImpressionStats {
            total_impressions,
            total_clicks,
            distinct_users: self.user_counters.len(),
            distinct_items: self.content_counters.len(),
            global_ctr,
        }
    }

    /// Total number of recorded impressions.
    #[must_use]
    pub fn total_impressions(&self) -> usize {
        self.impressions.len()
    }

    /// Top content items by CTR (minimum impression threshold).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn top_by_ctr(&self, min_impressions: u64, limit: usize) -> Vec<ContentMetrics> {
        let mut items: Vec<ContentMetrics> = self
            .content_counters
            .iter()
            .filter(|(_, &(imps, _, _, _))| imps >= min_impressions)
            .map(|(cid, &(imps, clicks, sum_pos, sum_dwell))| {
                let avg_position = if imps > 0 {
                    sum_pos as f64 / imps as f64
                } else {
                    0.0
                };
                let avg_dwell_ms = if clicks > 0 {
                    sum_dwell as f64 / clicks as f64
                } else {
                    0.0
                };
                ContentMetrics {
                    content_id: cid.clone(),
                    impressions: imps,
                    clicks,
                    avg_position,
                    avg_dwell_ms,
                }
            })
            .collect();
        items.sort_by(|a, b| {
            b.ctr()
                .partial_cmp(&a.ctr())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(limit);
        items
    }

    /// Clear all tracked data.
    pub fn clear(&mut self) {
        self.impressions.clear();
        self.content_counters.clear();
        self.user_counters.clear();
    }
}

impl Default for ImpressionTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Seen-content deduplication filter
// ---------------------------------------------------------------------------

/// Configuration for the seen-content filter.
#[derive(Debug, Clone)]
pub struct SeenContentConfig {
    /// Maximum number of seen items to remember per user.
    /// When this limit is reached, the oldest entries are evicted (FIFO).
    pub max_seen_per_user: usize,
    /// If `true`, only clicked impressions count as "seen".  If `false`,
    /// any impression (even without a click) counts.
    pub click_only: bool,
}

impl Default for SeenContentConfig {
    fn default() -> Self {
        Self {
            max_seen_per_user: 500,
            click_only: false,
        }
    }
}

/// Filter that tracks which content items each user has already seen and
/// provides deduplication for recommendation lists.
///
/// Internally, each user has an ordered ring buffer of content IDs so that
/// eviction is O(1) and look-up is O(1) via a `HashSet`.
#[derive(Debug)]
pub struct SeenContentFilter {
    /// Per-user: (ordered FIFO queue of content IDs, HashSet for fast lookup)
    seen: HashMap<
        String,
        (
            std::collections::VecDeque<String>,
            std::collections::HashSet<String>,
        ),
    >,
    /// Configuration.
    config: SeenContentConfig,
    /// Total items filtered out across all calls to `filter_unseen`.
    total_filtered: u64,
}

impl SeenContentFilter {
    /// Create a new filter with the given configuration.
    #[must_use]
    pub fn new(config: SeenContentConfig) -> Self {
        Self {
            seen: HashMap::new(),
            config,
            total_filtered: 0,
        }
    }

    /// Mark `content_id` as seen for `user_id`.
    ///
    /// If the user's seen buffer is at capacity, the oldest entry is evicted.
    pub fn mark_seen(&mut self, user_id: &str, content_id: &str) {
        let entry = self.seen.entry(user_id.to_string()).or_insert_with(|| {
            (
                std::collections::VecDeque::new(),
                std::collections::HashSet::new(),
            )
        });

        if !entry.1.contains(content_id) {
            // Evict oldest if at capacity
            if entry.0.len() >= self.config.max_seen_per_user {
                if let Some(oldest) = entry.0.pop_front() {
                    entry.1.remove(&oldest);
                }
            }
            entry.0.push_back(content_id.to_string());
            entry.1.insert(content_id.to_string());
        }
    }

    /// Returns `true` if `user_id` has already seen `content_id`.
    #[must_use]
    pub fn has_seen(&self, user_id: &str, content_id: &str) -> bool {
        self.seen
            .get(user_id)
            .map_or(false, |(_, set)| set.contains(content_id))
    }

    /// Filter a list of content IDs, returning only items the user has NOT seen.
    ///
    /// Updates the `total_filtered` counter by the number of removed items.
    pub fn filter_unseen(&mut self, user_id: &str, content_ids: Vec<String>) -> Vec<String> {
        let before = content_ids.len();
        let filtered: Vec<String> = content_ids
            .into_iter()
            .filter(|cid| !self.has_seen(user_id, cid))
            .collect();
        let removed = before - filtered.len();
        self.total_filtered += removed as u64;
        filtered
    }

    /// Ingest all impressions from an `ImpressionTracker` for a given user.
    ///
    /// Marks every item that appears in the tracker's impression log as seen
    /// for `user_id`.  If `config.click_only` is `true`, only clicked
    /// impressions are imported.
    pub fn ingest_from_tracker(&mut self, tracker: &ImpressionTracker, user_id: &str) {
        for imp in tracker.impressions.values() {
            if imp.user_id != user_id {
                continue;
            }
            if self.config.click_only && !imp.clicked {
                continue;
            }
            self.mark_seen(user_id, &imp.content_id);
        }
    }

    /// Clear all seen data for a single user.
    pub fn clear_user(&mut self, user_id: &str) {
        self.seen.remove(user_id);
    }

    /// Clear all seen data for all users.
    pub fn clear_all(&mut self) {
        self.seen.clear();
    }

    /// Number of seen items for a specific user.
    #[must_use]
    pub fn seen_count(&self, user_id: &str) -> usize {
        self.seen.get(user_id).map_or(0, |(q, _)| q.len())
    }

    /// Total number of users being tracked.
    #[must_use]
    pub fn tracked_users(&self) -> usize {
        self.seen.len()
    }

    /// Total items filtered out across all `filter_unseen` calls.
    #[must_use]
    pub fn total_filtered(&self) -> u64 {
        self.total_filtered
    }
}

impl Default for SeenContentFilter {
    fn default() -> Self {
        Self::new(SeenContentConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tracker_is_empty() {
        let tracker = ImpressionTracker::new();
        assert_eq!(tracker.total_impressions(), 0);
        let stats = tracker.global_stats();
        assert_eq!(stats.total_impressions, 0);
        assert_eq!(stats.global_ctr, 0.0);
    }

    #[test]
    fn test_record_impression() {
        let mut tracker = ImpressionTracker::new();
        let id = tracker.record_impression("user1", "video1", 0, 1000);
        assert_eq!(id.to_string(), "imp_0");
        assert_eq!(tracker.total_impressions(), 1);
    }

    #[test]
    fn test_record_click_success() {
        let mut tracker = ImpressionTracker::new();
        let id = tracker.record_impression("user1", "video1", 0, 1000);
        assert!(tracker.record_click(&id.0, 5000));
    }

    #[test]
    fn test_record_click_nonexistent() {
        let mut tracker = ImpressionTracker::new();
        assert!(!tracker.record_click("nonexistent", 5000));
    }

    #[test]
    fn test_double_click_rejected() {
        let mut tracker = ImpressionTracker::new();
        let id = tracker.record_impression("user1", "video1", 0, 1000);
        assert!(tracker.record_click(&id.0, 5000));
        assert!(!tracker.record_click(&id.0, 6000));
    }

    #[test]
    fn test_content_metrics_ctr() {
        let mut tracker = ImpressionTracker::new();
        let id1 = tracker.record_impression("u1", "vid", 0, 100);
        tracker.record_impression("u2", "vid", 1, 200);
        tracker.record_click(&id1.0, 3000);

        let metrics = tracker
            .content_metrics("vid")
            .expect("should succeed in test");
        assert_eq!(metrics.impressions, 2);
        assert_eq!(metrics.clicks, 1);
        assert!((metrics.ctr() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_content_metrics_avg_position() {
        let mut tracker = ImpressionTracker::new();
        tracker.record_impression("u1", "vid", 0, 100);
        tracker.record_impression("u2", "vid", 4, 200);
        let metrics = tracker
            .content_metrics("vid")
            .expect("should succeed in test");
        assert!((metrics.avg_position - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_user_metrics() {
        let mut tracker = ImpressionTracker::new();
        let id1 = tracker.record_impression("alice", "v1", 0, 100);
        tracker.record_impression("alice", "v2", 1, 200);
        tracker.record_click(&id1.0, 2000);

        let um = tracker
            .user_metrics("alice")
            .expect("should succeed in test");
        assert_eq!(um.impressions, 2);
        assert_eq!(um.clicks, 1);
        assert_eq!(um.unique_items, 2);
        assert!((um.ctr() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_global_stats() {
        let mut tracker = ImpressionTracker::new();
        let id1 = tracker.record_impression("u1", "v1", 0, 100);
        tracker.record_impression("u2", "v2", 1, 200);
        tracker.record_impression("u1", "v2", 2, 300);
        tracker.record_click(&id1.0, 1000);

        let stats = tracker.global_stats();
        assert_eq!(stats.total_impressions, 3);
        assert_eq!(stats.total_clicks, 1);
        assert_eq!(stats.distinct_users, 2);
        assert_eq!(stats.distinct_items, 2);
    }

    #[test]
    fn test_top_by_ctr() {
        let mut tracker = ImpressionTracker::new();
        // video_a: 2 impressions, 2 clicks => CTR 1.0
        let a1 = tracker.record_impression("u1", "video_a", 0, 100);
        let a2 = tracker.record_impression("u2", "video_a", 0, 200);
        tracker.record_click(&a1.0, 1000);
        tracker.record_click(&a2.0, 2000);
        // video_b: 2 impressions, 1 click => CTR 0.5
        let b1 = tracker.record_impression("u1", "video_b", 1, 300);
        tracker.record_impression("u2", "video_b", 1, 400);
        tracker.record_click(&b1.0, 500);

        let top = tracker.top_by_ctr(2, 10);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].content_id, "video_a");
    }

    #[test]
    fn test_clear() {
        let mut tracker = ImpressionTracker::new();
        tracker.record_impression("u1", "v1", 0, 100);
        tracker.clear();
        assert_eq!(tracker.total_impressions(), 0);
        assert_eq!(tracker.global_stats().distinct_items, 0);
    }

    #[test]
    fn test_content_metrics_none_for_unknown() {
        let tracker = ImpressionTracker::new();
        assert!(tracker.content_metrics("nonexistent").is_none());
    }

    #[test]
    fn test_impression_id_display() {
        let id = ImpressionId("imp_42".to_string());
        assert_eq!(id.to_string(), "imp_42");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SeenContentFilter tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_seen_filter_new_user_has_not_seen_anything() {
        let filter = SeenContentFilter::default();
        assert!(!filter.has_seen("alice", "video1"));
        assert_eq!(filter.seen_count("alice"), 0);
    }

    #[test]
    fn test_seen_filter_mark_and_check_seen() {
        let mut filter = SeenContentFilter::default();
        filter.mark_seen("alice", "video1");
        assert!(filter.has_seen("alice", "video1"));
        assert!(!filter.has_seen("alice", "video2"));
    }

    #[test]
    fn test_seen_filter_mark_multiple_users_independent() {
        let mut filter = SeenContentFilter::default();
        filter.mark_seen("alice", "video1");
        filter.mark_seen("bob", "video2");
        assert!(filter.has_seen("alice", "video1"));
        assert!(!filter.has_seen("alice", "video2"));
        assert!(filter.has_seen("bob", "video2"));
        assert!(!filter.has_seen("bob", "video1"));
    }

    #[test]
    fn test_seen_filter_evicts_oldest_on_capacity() {
        let config = SeenContentConfig {
            max_seen_per_user: 3,
            click_only: false,
        };
        let mut filter = SeenContentFilter::new(config);
        filter.mark_seen("alice", "v1");
        filter.mark_seen("alice", "v2");
        filter.mark_seen("alice", "v3");
        // Capacity is 3 — adding v4 evicts v1
        filter.mark_seen("alice", "v4");
        assert!(
            !filter.has_seen("alice", "v1"),
            "v1 should have been evicted"
        );
        assert!(filter.has_seen("alice", "v2"));
        assert!(filter.has_seen("alice", "v3"));
        assert!(filter.has_seen("alice", "v4"));
        assert_eq!(filter.seen_count("alice"), 3);
    }

    #[test]
    fn test_seen_filter_duplicate_mark_does_not_grow() {
        let mut filter = SeenContentFilter::default();
        filter.mark_seen("alice", "v1");
        filter.mark_seen("alice", "v1"); // duplicate
        assert_eq!(filter.seen_count("alice"), 1);
    }

    #[test]
    fn test_seen_filter_filter_unseen_removes_seen_items() {
        let mut filter = SeenContentFilter::default();
        filter.mark_seen("alice", "v1");
        filter.mark_seen("alice", "v3");
        let ids = vec![
            "v1".to_string(),
            "v2".to_string(),
            "v3".to_string(),
            "v4".to_string(),
        ];
        let result = filter.filter_unseen("alice", ids);
        assert_eq!(result, vec!["v2".to_string(), "v4".to_string()]);
    }

    #[test]
    fn test_seen_filter_filter_unseen_counts_filtered() {
        let mut filter = SeenContentFilter::default();
        filter.mark_seen("alice", "v1");
        filter.mark_seen("alice", "v2");
        let ids = vec!["v1".to_string(), "v2".to_string(), "v3".to_string()];
        filter.filter_unseen("alice", ids);
        assert_eq!(filter.total_filtered(), 2);
    }

    #[test]
    fn test_seen_filter_filter_unseen_empty_list() {
        let mut filter = SeenContentFilter::default();
        let result = filter.filter_unseen("alice", vec![]);
        assert!(result.is_empty());
        assert_eq!(filter.total_filtered(), 0);
    }

    #[test]
    fn test_seen_filter_ingest_from_tracker_all_impressions() {
        let mut tracker = ImpressionTracker::new();
        tracker.record_impression("alice", "v1", 0, 100);
        tracker.record_impression("alice", "v2", 1, 200);
        tracker.record_impression("bob", "v3", 2, 300);

        let config = SeenContentConfig {
            click_only: false,
            ..Default::default()
        };
        let mut filter = SeenContentFilter::new(config);
        filter.ingest_from_tracker(&tracker, "alice");

        assert!(filter.has_seen("alice", "v1"));
        assert!(filter.has_seen("alice", "v2"));
        assert!(
            !filter.has_seen("alice", "v3"),
            "bob's impression should not affect alice"
        );
        assert_eq!(filter.seen_count("alice"), 2);
    }

    #[test]
    fn test_seen_filter_ingest_click_only_skips_unclicked() {
        let mut tracker = ImpressionTracker::new();
        let id1 = tracker.record_impression("alice", "v1", 0, 100);
        tracker.record_impression("alice", "v2", 1, 200);
        tracker.record_click(&id1.0, 5000);

        let config = SeenContentConfig {
            click_only: true,
            ..Default::default()
        };
        let mut filter = SeenContentFilter::new(config);
        filter.ingest_from_tracker(&tracker, "alice");

        assert!(
            filter.has_seen("alice", "v1"),
            "clicked item should be seen"
        );
        assert!(
            !filter.has_seen("alice", "v2"),
            "unclicked item should not be seen"
        );
    }

    #[test]
    fn test_seen_filter_clear_user() {
        let mut filter = SeenContentFilter::default();
        filter.mark_seen("alice", "v1");
        filter.mark_seen("alice", "v2");
        filter.clear_user("alice");
        assert_eq!(filter.seen_count("alice"), 0);
        assert!(!filter.has_seen("alice", "v1"));
    }

    #[test]
    fn test_seen_filter_clear_all() {
        let mut filter = SeenContentFilter::default();
        filter.mark_seen("alice", "v1");
        filter.mark_seen("bob", "v2");
        filter.clear_all();
        assert_eq!(filter.tracked_users(), 0);
    }

    #[test]
    fn test_seen_filter_tracked_users() {
        let mut filter = SeenContentFilter::default();
        assert_eq!(filter.tracked_users(), 0);
        filter.mark_seen("alice", "v1");
        assert_eq!(filter.tracked_users(), 1);
        filter.mark_seen("bob", "v1");
        assert_eq!(filter.tracked_users(), 2);
    }
}
