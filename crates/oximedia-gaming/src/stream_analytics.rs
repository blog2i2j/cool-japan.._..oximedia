#![allow(dead_code)]
//! Stream analytics for `OxiMedia` gaming crate.
//!
//! Tracks viewer statistics, engagement metrics, and watch-time aggregates for
//! a live game stream session.

use std::collections::HashMap;

/// Broad classification of a viewer segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewerSegment {
    /// New viewer, joined within the last 5 minutes
    New,
    /// Returning viewer who has watched before
    Returning,
    /// Long-time subscriber or follower
    Loyal,
    /// Viewer who only clicked in from a recommendation
    Casual,
    /// Viewer who consistently interacts (chat, bits, etc.)
    Engaged,
}

impl ViewerSegment {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            ViewerSegment::New => "new",
            ViewerSegment::Returning => "returning",
            ViewerSegment::Loyal => "loyal",
            ViewerSegment::Casual => "casual",
            ViewerSegment::Engaged => "engaged",
        }
    }
}

/// Per-viewer statistics accumulated during a session.
#[derive(Debug, Clone)]
pub struct ViewerStats {
    /// Viewer identifier (opaque string from platform).
    pub viewer_id: String,
    /// Segment classification.
    pub segment: ViewerSegment,
    /// Total watch time in seconds.
    pub watch_time_secs: u64,
    /// Number of chat messages sent.
    pub chat_messages: u32,
    /// Number of subscription / donation events.
    pub support_actions: u32,
    /// Raids or shares initiated by this viewer.
    pub social_actions: u32,
}

impl ViewerStats {
    /// Create new zeroed stats for a viewer.
    #[must_use]
    pub fn new(viewer_id: impl Into<String>, segment: ViewerSegment) -> Self {
        Self {
            viewer_id: viewer_id.into(),
            segment,
            watch_time_secs: 0,
            chat_messages: 0,
            support_actions: 0,
            social_actions: 0,
        }
    }

    /// Composite engagement score in `[0.0, 1.0]`.
    ///
    /// Weights: watch time contributes up to 0.5, chat up to 0.3,
    /// support and social actions together up to 0.2.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn engagement_score(&self) -> f64 {
        let watch_cap = 3600.0_f64; // 1 hour cap for normalisation
        let watch_norm = (self.watch_time_secs as f64 / watch_cap).min(1.0);

        let chat_cap = 100.0_f64;
        let chat_norm = (f64::from(self.chat_messages) / chat_cap).min(1.0);

        let action_cap = 10.0_f64;
        let action_norm =
            (f64::from(self.support_actions + self.social_actions) / action_cap).min(1.0);

        (watch_norm * 0.5) + (chat_norm * 0.3) + (action_norm * 0.2)
    }

    /// Return `true` if the viewer is considered highly engaged (score >= 0.6).
    #[must_use]
    pub fn is_highly_engaged(&self) -> bool {
        self.engagement_score() >= 0.6
    }
}

/// A single viewer event recorded during the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewerEventKind {
    /// Viewer joined the stream
    Join,
    /// Viewer left the stream
    Leave,
    /// Viewer sent a chat message
    Chat,
    /// Viewer subscribed, donated, or similar
    Support,
    /// Viewer raided or shared
    Social,
    /// Heartbeat (viewer still present after N seconds)
    Heartbeat,
}

/// Aggregated analytics for a stream session.
#[derive(Debug, Default)]
pub struct StreamAnalytics {
    /// Per-viewer statistics, keyed by `viewer_id`.
    viewers: HashMap<String, ViewerStats>,
    /// Peak concurrent viewer count observed so far.
    peak_viewer_count: usize,
    /// Currently online viewer IDs.
    online: std::collections::HashSet<String>,
    /// Total watch-time seconds accumulated across all viewers.
    total_watch_secs: u64,
    /// Total heartbeat ticks used for watch-time increments (in seconds).
    heartbeat_interval_secs: u64,
}

impl StreamAnalytics {
    /// Create a new analytics tracker.
    ///
    /// `heartbeat_interval_secs` is added to a viewer's watch time on every
    /// `Heartbeat` event.
    #[must_use]
    pub fn new(heartbeat_interval_secs: u64) -> Self {
        Self {
            heartbeat_interval_secs,
            ..Default::default()
        }
    }

    /// Record a viewer event.
    pub fn record_viewer_event(
        &mut self,
        viewer_id: impl Into<String>,
        segment: ViewerSegment,
        kind: ViewerEventKind,
    ) {
        let id: String = viewer_id.into();

        let stats = self
            .viewers
            .entry(id.clone())
            .or_insert_with(|| ViewerStats::new(id.clone(), segment));

        match kind {
            ViewerEventKind::Join => {
                self.online.insert(id.clone());
                if self.online.len() > self.peak_viewer_count {
                    self.peak_viewer_count = self.online.len();
                }
            }
            ViewerEventKind::Leave => {
                self.online.remove(&id);
            }
            ViewerEventKind::Chat => {
                stats.chat_messages += 1;
            }
            ViewerEventKind::Support => {
                stats.support_actions += 1;
            }
            ViewerEventKind::Social => {
                stats.social_actions += 1;
            }
            ViewerEventKind::Heartbeat => {
                let delta = self.heartbeat_interval_secs;
                stats.watch_time_secs += delta;
                self.total_watch_secs += delta;
            }
        }
    }

    /// Peak concurrent viewer count for this session.
    #[must_use]
    pub fn peak_viewers(&self) -> usize {
        self.peak_viewer_count
    }

    /// Number of viewers currently online.
    #[must_use]
    pub fn current_viewers(&self) -> usize {
        self.online.len()
    }

    /// Average watch time in seconds across all viewers who have any watch time.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_watch_time_secs(&self) -> f64 {
        let with_time: Vec<_> = self
            .viewers
            .values()
            .filter(|v| v.watch_time_secs > 0)
            .collect();

        if with_time.is_empty() {
            return 0.0;
        }

        let total: u64 = with_time.iter().map(|v| v.watch_time_secs).sum();
        total as f64 / with_time.len() as f64
    }

    /// Total number of unique viewers tracked.
    #[must_use]
    pub fn unique_viewer_count(&self) -> usize {
        self.viewers.len()
    }

    /// Return stats for a specific viewer, if present.
    #[must_use]
    pub fn viewer_stats(&self, viewer_id: &str) -> Option<&ViewerStats> {
        self.viewers.get(viewer_id)
    }

    /// Viewers in a given segment.
    #[must_use]
    pub fn viewers_in_segment(&self, segment: ViewerSegment) -> Vec<&ViewerStats> {
        self.viewers
            .values()
            .filter(|v| v.segment == segment)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewer_segment_labels() {
        assert_eq!(ViewerSegment::New.label(), "new");
        assert_eq!(ViewerSegment::Returning.label(), "returning");
        assert_eq!(ViewerSegment::Loyal.label(), "loyal");
        assert_eq!(ViewerSegment::Casual.label(), "casual");
        assert_eq!(ViewerSegment::Engaged.label(), "engaged");
    }

    #[test]
    fn test_viewer_stats_new_zeroed() {
        let stats = ViewerStats::new("u1", ViewerSegment::New);
        assert_eq!(stats.viewer_id, "u1");
        assert_eq!(stats.watch_time_secs, 0);
        assert_eq!(stats.chat_messages, 0);
        assert_eq!(stats.support_actions, 0);
        assert_eq!(stats.social_actions, 0);
    }

    #[test]
    fn test_engagement_score_zero() {
        let stats = ViewerStats::new("u0", ViewerSegment::Casual);
        assert!((stats.engagement_score() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_engagement_score_watch_only() {
        let mut stats = ViewerStats::new("u1", ViewerSegment::New);
        stats.watch_time_secs = 3600; // max watch norm = 1.0 → 0.5 score
        let score = stats.engagement_score();
        assert!((score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_engagement_score_fully_engaged() {
        let mut stats = ViewerStats::new("u2", ViewerSegment::Loyal);
        stats.watch_time_secs = 3600;
        stats.chat_messages = 100;
        stats.support_actions = 5;
        stats.social_actions = 5;
        let score = stats.engagement_score();
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_highly_engaged() {
        let mut stats = ViewerStats::new("u3", ViewerSegment::Engaged);
        stats.watch_time_secs = 3600;
        stats.chat_messages = 80;
        assert!(stats.is_highly_engaged());
    }

    #[test]
    fn test_not_highly_engaged() {
        let stats = ViewerStats::new("u4", ViewerSegment::Casual);
        assert!(!stats.is_highly_engaged());
    }

    #[test]
    fn test_record_join_increments_peak() {
        let mut analytics = StreamAnalytics::new(30);
        analytics.record_viewer_event("v1", ViewerSegment::New, ViewerEventKind::Join);
        analytics.record_viewer_event("v2", ViewerSegment::New, ViewerEventKind::Join);
        assert_eq!(analytics.peak_viewers(), 2);
        assert_eq!(analytics.current_viewers(), 2);
    }

    #[test]
    fn test_record_leave_decrements_current() {
        let mut analytics = StreamAnalytics::new(30);
        analytics.record_viewer_event("v1", ViewerSegment::New, ViewerEventKind::Join);
        analytics.record_viewer_event("v2", ViewerSegment::New, ViewerEventKind::Join);
        analytics.record_viewer_event("v1", ViewerSegment::New, ViewerEventKind::Leave);
        assert_eq!(analytics.current_viewers(), 1);
        // peak should still be 2
        assert_eq!(analytics.peak_viewers(), 2);
    }

    #[test]
    fn test_heartbeat_accumulates_watch_time() {
        let mut analytics = StreamAnalytics::new(60);
        analytics.record_viewer_event("v1", ViewerSegment::Returning, ViewerEventKind::Heartbeat);
        analytics.record_viewer_event("v1", ViewerSegment::Returning, ViewerEventKind::Heartbeat);
        let stats = analytics
            .viewer_stats("v1")
            .expect("viewer stats should succeed");
        assert_eq!(stats.watch_time_secs, 120);
    }

    #[test]
    fn test_avg_watch_time_secs() {
        let mut analytics = StreamAnalytics::new(60);
        analytics.record_viewer_event("v1", ViewerSegment::Loyal, ViewerEventKind::Heartbeat);
        analytics.record_viewer_event("v2", ViewerSegment::Loyal, ViewerEventKind::Heartbeat);
        analytics.record_viewer_event("v2", ViewerSegment::Loyal, ViewerEventKind::Heartbeat);
        // v1: 60s, v2: 120s → avg = 90s
        assert!((analytics.avg_watch_time_secs() - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_watch_time_no_viewers() {
        let analytics = StreamAnalytics::new(30);
        assert!((analytics.avg_watch_time_secs() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chat_message_tracking() {
        let mut analytics = StreamAnalytics::new(30);
        analytics.record_viewer_event("v1", ViewerSegment::Engaged, ViewerEventKind::Chat);
        analytics.record_viewer_event("v1", ViewerSegment::Engaged, ViewerEventKind::Chat);
        let stats = analytics
            .viewer_stats("v1")
            .expect("viewer stats should succeed");
        assert_eq!(stats.chat_messages, 2);
    }

    #[test]
    fn test_viewers_in_segment() {
        let mut analytics = StreamAnalytics::new(30);
        analytics.record_viewer_event("v1", ViewerSegment::New, ViewerEventKind::Join);
        analytics.record_viewer_event("v2", ViewerSegment::New, ViewerEventKind::Join);
        analytics.record_viewer_event("v3", ViewerSegment::Loyal, ViewerEventKind::Join);
        assert_eq!(analytics.viewers_in_segment(ViewerSegment::New).len(), 2);
        assert_eq!(analytics.viewers_in_segment(ViewerSegment::Loyal).len(), 1);
    }

    #[test]
    fn test_unique_viewer_count() {
        let mut analytics = StreamAnalytics::new(30);
        analytics.record_viewer_event("v1", ViewerSegment::New, ViewerEventKind::Join);
        analytics.record_viewer_event("v2", ViewerSegment::Casual, ViewerEventKind::Join);
        analytics.record_viewer_event("v1", ViewerSegment::New, ViewerEventKind::Chat);
        assert_eq!(analytics.unique_viewer_count(), 2);
    }
}
