//! Usage tracking for royalty reporting.
//!
//! This module provides fine-grained tracking of how content is consumed across
//! different usage types and territories.  Aggregates are computed on demand so
//! that individual events are available for auditing while reporting helpers
//! summarise them efficiently.

#![allow(missing_docs)]

use std::collections::{HashMap, HashSet};

// ── UsageEventType ────────────────────────────────────────────────────────────

/// The category of a recorded usage event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UsageEventType {
    /// Full on-demand playback.
    Play,
    /// File download.
    Download,
    /// Streaming playback (adaptive bitrate / live).
    Stream,
    /// Short preview (typically < 30 s).
    Preview,
    /// User-initiated share to another platform.
    Share,
    /// Embedded playback on a third-party site.
    Embed,
}

// ── UsageEvent ────────────────────────────────────────────────────────────────

/// A single usage event recorded against a piece of content.
#[derive(Debug, Clone)]
pub struct UsageEvent {
    /// Unique identifier for the piece of content.
    pub content_id: String,
    /// Identifier of the end-user who triggered the event.
    pub user_id: String,
    /// The kind of usage.
    pub event_type: UsageEventType,
    /// Unix timestamp (seconds) when the event occurred.
    pub timestamp_secs: u64,
    /// How long the content was consumed (seconds).  Zero for non-temporal
    /// events such as `Download` or `Share`.
    pub duration_secs: f32,
    /// ISO 3166-1 alpha-2 territory code (e.g. `"US"`, `"DE"`).
    pub territory: String,
}

// ── UsageReport ───────────────────────────────────────────────────────────────

/// Aggregated usage statistics for a single piece of content.
#[derive(Debug, Clone, PartialEq)]
pub struct UsageReport {
    /// The piece of content these statistics belong to.
    pub content_id: String,
    /// Total number of `Play` events.
    pub total_plays: u64,
    /// Total number of `Stream` events.
    pub total_streams: u64,
    /// Total number of `Download` events.
    pub total_downloads: u64,
    /// Total consumption time across all temporal event types (seconds).
    pub total_duration_secs: f64,
    /// Number of distinct user IDs that interacted with this content.
    pub unique_users: u64,
}

// ── UsageTracker ──────────────────────────────────────────────────────────────

/// Central store for usage events with aggregation helpers.
#[derive(Debug, Default)]
pub struct UsageTracker {
    /// All recorded events in insertion order.
    events: Vec<UsageEvent>,
}

impl UsageTracker {
    /// Create an empty `UsageTracker`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a usage event.
    pub fn record(&mut self, event: UsageEvent) {
        self.events.push(event);
    }

    /// Count occurrences of `event_type` for `content_id`.
    #[must_use]
    pub fn usage_count(&self, content_id: &str, event_type: UsageEventType) -> u64 {
        self.events
            .iter()
            .filter(|e| e.content_id == content_id && e.event_type == event_type)
            .count() as u64
    }

    /// Sum of `duration_secs` for all `Play`, `Stream`, and `Preview` events
    /// associated with `content_id`.
    ///
    /// Download / Share / Embed events are intentionally excluded because they
    /// do not represent consumption time.
    #[must_use]
    pub fn total_play_duration(&self, content_id: &str) -> f64 {
        self.events
            .iter()
            .filter(|e| {
                e.content_id == content_id
                    && matches!(
                        e.event_type,
                        UsageEventType::Play | UsageEventType::Stream | UsageEventType::Preview
                    )
            })
            .map(|e| e.duration_secs as f64)
            .sum()
    }

    /// Returns a map of `territory → play_count` for `content_id`.
    ///
    /// Only `Play` events are counted; `Stream` and other types are excluded so
    /// that the breakdown specifically represents discrete play royalty events.
    #[must_use]
    pub fn territory_breakdown(&self, content_id: &str) -> HashMap<String, u64> {
        let mut map: HashMap<String, u64> = HashMap::new();
        for event in self
            .events
            .iter()
            .filter(|e| e.content_id == content_id && e.event_type == UsageEventType::Play)
        {
            *map.entry(event.territory.clone()).or_insert(0) += 1;
        }
        map
    }

    /// Return the top-`n` pieces of content by count of `event_type`, sorted
    /// in descending order.  Ties are broken by content_id (lexicographic).
    ///
    /// Returns an empty `Vec` when `n` is 0.
    #[must_use]
    pub fn top_content(&self, n: usize, event_type: UsageEventType) -> Vec<(String, u64)> {
        if n == 0 {
            return Vec::new();
        }

        // Aggregate counts per content_id for the given event type.
        let mut counts: HashMap<String, u64> = HashMap::new();
        for event in self.events.iter().filter(|e| e.event_type == event_type) {
            *counts.entry(event.content_id.clone()).or_insert(0) += 1;
        }

        // Sort: primary key = count descending, secondary = content_id ascending.
        let mut sorted: Vec<(String, u64)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted.truncate(n);
        sorted
    }

    /// Generate a complete usage report for `content_id`.
    ///
    /// Returns `None` if no events have been recorded for that content.
    #[must_use]
    pub fn generate_report(&self, content_id: &str) -> Option<UsageReport> {
        let relevant: Vec<&UsageEvent> = self
            .events
            .iter()
            .filter(|e| e.content_id == content_id)
            .collect();

        if relevant.is_empty() {
            return None;
        }

        let mut total_plays = 0u64;
        let mut total_streams = 0u64;
        let mut total_downloads = 0u64;
        let mut total_duration_secs = 0.0_f64;
        let mut unique_users: HashSet<&str> = HashSet::new();

        for event in &relevant {
            unique_users.insert(event.user_id.as_str());
            match event.event_type {
                UsageEventType::Play => {
                    total_plays += 1;
                    total_duration_secs += event.duration_secs as f64;
                }
                UsageEventType::Stream => {
                    total_streams += 1;
                    total_duration_secs += event.duration_secs as f64;
                }
                UsageEventType::Download => {
                    total_downloads += 1;
                }
                UsageEventType::Preview => {
                    total_duration_secs += event.duration_secs as f64;
                }
                UsageEventType::Share | UsageEventType::Embed => {}
            }
        }

        Some(UsageReport {
            content_id: content_id.to_string(),
            total_plays,
            total_streams,
            total_downloads,
            total_duration_secs,
            unique_users: unique_users.len() as u64,
        })
    }

    /// Return a reference to all stored events.
    #[must_use]
    pub fn events(&self) -> &[UsageEvent] {
        &self.events
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn play(content_id: &str, user_id: &str, territory: &str, duration: f32) -> UsageEvent {
        UsageEvent {
            content_id: content_id.to_string(),
            user_id: user_id.to_string(),
            event_type: UsageEventType::Play,
            timestamp_secs: 1_000_000,
            duration_secs: duration,
            territory: territory.to_string(),
        }
    }

    fn stream_event(content_id: &str, user_id: &str, duration: f32) -> UsageEvent {
        UsageEvent {
            content_id: content_id.to_string(),
            user_id: user_id.to_string(),
            event_type: UsageEventType::Stream,
            timestamp_secs: 1_000_000,
            duration_secs: duration,
            territory: "US".to_string(),
        }
    }

    fn download(content_id: &str, user_id: &str) -> UsageEvent {
        UsageEvent {
            content_id: content_id.to_string(),
            user_id: user_id.to_string(),
            event_type: UsageEventType::Download,
            timestamp_secs: 1_000_000,
            duration_secs: 0.0,
            territory: "US".to_string(),
        }
    }

    // ── usage_count ───────────────────────────────────────────────────────────

    #[test]
    fn test_usage_count_zero_when_empty() {
        let tracker = UsageTracker::new();
        assert_eq!(tracker.usage_count("c1", UsageEventType::Play), 0);
    }

    #[test]
    fn test_usage_count_counts_correct_type_only() {
        let mut tracker = UsageTracker::new();
        tracker.record(play("c1", "u1", "US", 180.0));
        tracker.record(play("c1", "u2", "GB", 120.0));
        tracker.record(stream_event("c1", "u3", 60.0));
        // play count = 2, stream count = 1
        assert_eq!(tracker.usage_count("c1", UsageEventType::Play), 2);
        assert_eq!(tracker.usage_count("c1", UsageEventType::Stream), 1);
        assert_eq!(tracker.usage_count("c1", UsageEventType::Download), 0);
    }

    #[test]
    fn test_usage_count_isolated_per_content() {
        let mut tracker = UsageTracker::new();
        tracker.record(play("c1", "u1", "US", 100.0));
        tracker.record(play("c2", "u1", "US", 100.0));
        assert_eq!(tracker.usage_count("c1", UsageEventType::Play), 1);
        assert_eq!(tracker.usage_count("c2", UsageEventType::Play), 1);
    }

    // ── total_play_duration ───────────────────────────────────────────────────

    #[test]
    fn test_total_play_duration_accumulates_play_and_stream() {
        let mut tracker = UsageTracker::new();
        tracker.record(play("c1", "u1", "US", 200.0));
        tracker.record(stream_event("c1", "u2", 100.0));
        tracker.record(download("c1", "u3")); // should NOT add to duration
        let duration = tracker.total_play_duration("c1");
        assert!((duration - 300.0).abs() < 1e-6, "got {duration}");
    }

    #[test]
    fn test_total_play_duration_excludes_other_content() {
        let mut tracker = UsageTracker::new();
        tracker.record(play("c1", "u1", "US", 300.0));
        tracker.record(play("c2", "u1", "US", 999.0));
        let duration = tracker.total_play_duration("c1");
        assert!((duration - 300.0).abs() < 1e-6, "got {duration}");
    }

    // ── territory_breakdown ───────────────────────────────────────────────────

    #[test]
    fn test_territory_breakdown_correct_counts() {
        let mut tracker = UsageTracker::new();
        tracker.record(play("c1", "u1", "US", 100.0));
        tracker.record(play("c1", "u2", "US", 100.0));
        tracker.record(play("c1", "u3", "DE", 100.0));
        tracker.record(stream_event("c1", "u4", 100.0)); // stream should NOT appear
        let breakdown = tracker.territory_breakdown("c1");
        assert_eq!(breakdown.get("US"), Some(&2));
        assert_eq!(breakdown.get("DE"), Some(&1));
        assert!(!breakdown.contains_key("stream"), "stream entries leaked");
    }

    #[test]
    fn test_territory_breakdown_empty_when_no_plays() {
        let mut tracker = UsageTracker::new();
        tracker.record(download("c1", "u1"));
        let breakdown = tracker.territory_breakdown("c1");
        assert!(breakdown.is_empty());
    }

    // ── top_content ───────────────────────────────────────────────────────────

    #[test]
    fn test_top_content_ordering() {
        let mut tracker = UsageTracker::new();
        // c2 has most plays
        for i in 0..5u32 {
            tracker.record(play("c2", &format!("u{i}"), "US", 10.0));
        }
        for i in 0..3u32 {
            tracker.record(play("c1", &format!("u{i}"), "US", 10.0));
        }
        tracker.record(play("c3", "u0", "US", 10.0));

        let top = tracker.top_content(2, UsageEventType::Play);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "c2");
        assert_eq!(top[0].1, 5);
        assert_eq!(top[1].0, "c1");
        assert_eq!(top[1].1, 3);
    }

    #[test]
    fn test_top_content_n_zero_returns_empty() {
        let mut tracker = UsageTracker::new();
        tracker.record(play("c1", "u1", "US", 10.0));
        let top = tracker.top_content(0, UsageEventType::Play);
        assert!(top.is_empty());
    }

    #[test]
    fn test_top_content_respects_event_type_boundary() {
        let mut tracker = UsageTracker::new();
        tracker.record(play("c1", "u1", "US", 10.0));
        tracker.record(stream_event("c2", "u1", 10.0));
        let top = tracker.top_content(5, UsageEventType::Play);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "c1");
    }

    // ── generate_report ───────────────────────────────────────────────────────

    #[test]
    fn test_generate_report_none_for_unknown_content() {
        let tracker = UsageTracker::new();
        assert!(tracker.generate_report("unknown").is_none());
    }

    #[test]
    fn test_generate_report_aggregates_correctly() {
        let mut tracker = UsageTracker::new();
        tracker.record(play("c1", "u1", "US", 200.0));
        tracker.record(play("c1", "u2", "DE", 100.0));
        tracker.record(stream_event("c1", "u1", 50.0));
        tracker.record(download("c1", "u3"));
        tracker.record(UsageEvent {
            content_id: "c1".to_string(),
            user_id: "u4".to_string(),
            event_type: UsageEventType::Share,
            timestamp_secs: 1_000_001,
            duration_secs: 0.0,
            territory: "US".to_string(),
        });

        let report = tracker.generate_report("c1").expect("report should exist");
        assert_eq!(report.content_id, "c1");
        assert_eq!(report.total_plays, 2);
        assert_eq!(report.total_streams, 1);
        assert_eq!(report.total_downloads, 1);
        // 200 + 100 (plays) + 50 (stream) = 350
        assert!((report.total_duration_secs - 350.0).abs() < 1e-6);
        // u1, u2, u3, u4 → 4 unique users
        assert_eq!(report.unique_users, 4);
    }

    #[test]
    fn test_unique_users_counts_same_user_once() {
        let mut tracker = UsageTracker::new();
        tracker.record(play("c1", "alice", "US", 100.0));
        tracker.record(play("c1", "alice", "DE", 100.0)); // same user again
        tracker.record(play("c1", "bob", "US", 100.0));

        let report = tracker.generate_report("c1").expect("report should exist");
        assert_eq!(report.unique_users, 2);
    }
}
