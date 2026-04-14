#![allow(dead_code)]

//! Real-time audience engagement analytics for gaming streams.
//!
//! Tracks viewer count, chat velocity, engagement score, and peak detection
//! over sliding windows to help streamers understand audience behavior.

use std::collections::VecDeque;

/// Default sliding window duration in seconds.
const DEFAULT_WINDOW_SECS: u64 = 300;

/// Number of buckets per window for histogram binning.
const HISTOGRAM_BUCKETS: usize = 60;

/// A timestamped viewer count sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewerSample {
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Concurrent viewer count.
    pub viewers: u32,
}

impl ViewerSample {
    /// Create a new viewer sample.
    #[must_use]
    pub fn new(timestamp: u64, viewers: u32) -> Self {
        Self { timestamp, viewers }
    }
}

/// A chat message event.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatEvent {
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Length of the message in characters.
    pub message_len: u32,
    /// Whether the message contained an emote.
    pub has_emote: bool,
}

impl ChatEvent {
    /// Create a new chat event.
    #[must_use]
    pub fn new(timestamp: u64, message_len: u32, has_emote: bool) -> Self {
        Self {
            timestamp,
            message_len,
            has_emote,
        }
    }
}

/// Engagement score computed from multiple signals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngagementScore {
    /// Overall engagement (0.0..1.0).
    pub overall: f64,
    /// Chat activity component (0.0..1.0).
    pub chat_activity: f64,
    /// Viewer retention component (0.0..1.0).
    pub viewer_retention: f64,
    /// Emote ratio component (0.0..1.0).
    pub emote_ratio: f64,
}

impl EngagementScore {
    /// Create a new engagement score.
    #[must_use]
    pub fn new(chat_activity: f64, viewer_retention: f64, emote_ratio: f64) -> Self {
        let overall = chat_activity * 0.4 + viewer_retention * 0.4 + emote_ratio * 0.2;
        Self {
            overall: overall.clamp(0.0, 1.0),
            chat_activity: chat_activity.clamp(0.0, 1.0),
            viewer_retention: viewer_retention.clamp(0.0, 1.0),
            emote_ratio: emote_ratio.clamp(0.0, 1.0),
        }
    }

    /// Check if the stream is in a "hype" state (engagement > 0.75).
    #[must_use]
    pub fn is_hype(&self) -> bool {
        self.overall > 0.75
    }
}

/// Peak detection result for audience metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakEvent {
    /// Timestamp of the peak.
    pub timestamp: u64,
    /// Value at peak.
    pub value: f64,
    /// Whether this is a local maximum (true) or minimum (false).
    pub is_maximum: bool,
}

impl PeakEvent {
    /// Create a new peak event.
    #[must_use]
    pub fn new(timestamp: u64, value: f64, is_maximum: bool) -> Self {
        Self {
            timestamp,
            value,
            is_maximum,
        }
    }
}

/// Sliding-window audience analytics tracker.
#[derive(Debug)]
pub struct AudienceTracker {
    /// Window duration in seconds.
    window_secs: u64,
    /// Viewer count samples.
    viewer_samples: VecDeque<ViewerSample>,
    /// Chat events.
    chat_events: VecDeque<ChatEvent>,
    /// Maximum samples to keep.
    max_samples: usize,
    /// Peak viewer count seen overall.
    peak_viewers: u32,
}

impl AudienceTracker {
    /// Create a new audience tracker with the given window duration.
    #[must_use]
    pub fn new(window_secs: u64) -> Self {
        let window = if window_secs == 0 {
            DEFAULT_WINDOW_SECS
        } else {
            window_secs
        };
        Self {
            window_secs: window,
            viewer_samples: VecDeque::with_capacity(1024),
            chat_events: VecDeque::with_capacity(4096),
            max_samples: 10_000,
            peak_viewers: 0,
        }
    }

    /// Record a viewer count sample.
    pub fn record_viewers(&mut self, sample: ViewerSample) {
        if sample.viewers > self.peak_viewers {
            self.peak_viewers = sample.viewers;
        }
        if self.viewer_samples.len() >= self.max_samples {
            self.viewer_samples.pop_front();
        }
        self.viewer_samples.push_back(sample);
    }

    /// Record a chat event.
    pub fn record_chat(&mut self, event: ChatEvent) {
        if self.chat_events.len() >= self.max_samples {
            self.chat_events.pop_front();
        }
        self.chat_events.push_back(event);
    }

    /// Get the current viewer count (latest sample).
    #[must_use]
    pub fn current_viewers(&self) -> u32 {
        self.viewer_samples.back().map_or(0, |s| s.viewers)
    }

    /// Get the peak viewer count.
    #[must_use]
    pub fn peak_viewers(&self) -> u32 {
        self.peak_viewers
    }

    /// Compute the average viewer count within the sliding window.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_viewers(&self, now: u64) -> f64 {
        let cutoff = now.saturating_sub(self.window_secs);
        let in_window: Vec<_> = self
            .viewer_samples
            .iter()
            .filter(|s| s.timestamp >= cutoff)
            .collect();
        if in_window.is_empty() {
            return 0.0;
        }
        let sum: u64 = in_window.iter().map(|s| u64::from(s.viewers)).sum();
        sum as f64 / in_window.len() as f64
    }

    /// Compute chat messages per minute within the sliding window.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn chat_velocity(&self, now: u64) -> f64 {
        let cutoff = now.saturating_sub(self.window_secs);
        let count = self
            .chat_events
            .iter()
            .filter(|e| e.timestamp >= cutoff)
            .count();
        if self.window_secs == 0 {
            return 0.0;
        }
        count as f64 / (self.window_secs as f64 / 60.0)
    }

    /// Compute the emote ratio within the sliding window.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn emote_ratio(&self, now: u64) -> f64 {
        let cutoff = now.saturating_sub(self.window_secs);
        let in_window: Vec<_> = self
            .chat_events
            .iter()
            .filter(|e| e.timestamp >= cutoff)
            .collect();
        if in_window.is_empty() {
            return 0.0;
        }
        let emote_count = in_window.iter().filter(|e| e.has_emote).count();
        emote_count as f64 / in_window.len() as f64
    }

    /// Compute viewer retention as a ratio of current to peak viewers.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn viewer_retention(&self) -> f64 {
        if self.peak_viewers == 0 {
            return 0.0;
        }
        let current = self.current_viewers();
        f64::from(current) / f64::from(self.peak_viewers)
    }

    /// Compute the overall engagement score at the given timestamp.
    #[must_use]
    pub fn engagement_score(&self, now: u64) -> EngagementScore {
        let velocity = self.chat_velocity(now);
        // Normalize chat velocity: 60 msgs/min = 1.0
        let chat_activity = (velocity / 60.0).min(1.0);
        let retention = self.viewer_retention();
        let emote = self.emote_ratio(now);
        EngagementScore::new(chat_activity, retention, emote)
    }

    /// Detect peaks in the viewer count history.
    ///
    /// A peak is a local maximum where the sample is higher than both neighbors.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_viewer_peaks(&self) -> Vec<PeakEvent> {
        let samples: Vec<_> = self.viewer_samples.iter().collect();
        let mut peaks = Vec::new();
        if samples.len() < 3 {
            return peaks;
        }
        for i in 1..samples.len() - 1 {
            if samples[i].viewers > samples[i - 1].viewers
                && samples[i].viewers > samples[i + 1].viewers
            {
                peaks.push(PeakEvent::new(
                    samples[i].timestamp,
                    f64::from(samples[i].viewers),
                    true,
                ));
            }
        }
        peaks
    }

    /// Return the number of viewer samples stored.
    #[must_use]
    pub fn viewer_sample_count(&self) -> usize {
        self.viewer_samples.len()
    }

    /// Return the number of chat events stored.
    #[must_use]
    pub fn chat_event_count(&self) -> usize {
        self.chat_events.len()
    }

    /// Clear all tracked data.
    pub fn reset(&mut self) {
        self.viewer_samples.clear();
        self.chat_events.clear();
        self.peak_viewers = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewer_sample_creation() {
        let s = ViewerSample::new(100, 500);
        assert_eq!(s.timestamp, 100);
        assert_eq!(s.viewers, 500);
    }

    #[test]
    fn test_chat_event_creation() {
        let e = ChatEvent::new(200, 42, true);
        assert_eq!(e.timestamp, 200);
        assert_eq!(e.message_len, 42);
        assert!(e.has_emote);
    }

    #[test]
    fn test_engagement_score_hype() {
        let score = EngagementScore::new(1.0, 1.0, 1.0);
        assert!(score.is_hype());
        assert!((score.overall - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_engagement_score_low() {
        let score = EngagementScore::new(0.1, 0.1, 0.1);
        assert!(!score.is_hype());
        assert!(score.overall < 0.2);
    }

    #[test]
    fn test_tracker_record_viewers() {
        let mut tracker = AudienceTracker::new(300);
        tracker.record_viewers(ViewerSample::new(10, 100));
        assert_eq!(tracker.current_viewers(), 100);
        assert_eq!(tracker.peak_viewers(), 100);
    }

    #[test]
    fn test_tracker_peak_tracking() {
        let mut tracker = AudienceTracker::new(300);
        tracker.record_viewers(ViewerSample::new(10, 100));
        tracker.record_viewers(ViewerSample::new(20, 500));
        tracker.record_viewers(ViewerSample::new(30, 200));
        assert_eq!(tracker.peak_viewers(), 500);
        assert_eq!(tracker.current_viewers(), 200);
    }

    #[test]
    fn test_average_viewers() {
        let mut tracker = AudienceTracker::new(300);
        tracker.record_viewers(ViewerSample::new(10, 100));
        tracker.record_viewers(ViewerSample::new(20, 200));
        tracker.record_viewers(ViewerSample::new(30, 300));
        let avg = tracker.average_viewers(100);
        assert!((avg - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chat_velocity() {
        let mut tracker = AudienceTracker::new(60); // 1 minute window
        for i in 0..30 {
            tracker.record_chat(ChatEvent::new(i, 10, false));
        }
        let velocity = tracker.chat_velocity(60);
        assert!((velocity - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_emote_ratio() {
        let mut tracker = AudienceTracker::new(300);
        tracker.record_chat(ChatEvent::new(10, 5, true));
        tracker.record_chat(ChatEvent::new(20, 5, false));
        tracker.record_chat(ChatEvent::new(30, 5, true));
        tracker.record_chat(ChatEvent::new(40, 5, false));
        let ratio = tracker.emote_ratio(100);
        assert!((ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_viewer_retention() {
        let mut tracker = AudienceTracker::new(300);
        tracker.record_viewers(ViewerSample::new(10, 1000));
        tracker.record_viewers(ViewerSample::new(20, 500));
        let retention = tracker.viewer_retention();
        assert!((retention - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_viewer_retention_no_data() {
        let tracker = AudienceTracker::new(300);
        assert!((tracker.viewer_retention() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_detect_viewer_peaks() {
        let mut tracker = AudienceTracker::new(300);
        tracker.record_viewers(ViewerSample::new(1, 100));
        tracker.record_viewers(ViewerSample::new(2, 500));
        tracker.record_viewers(ViewerSample::new(3, 200));
        tracker.record_viewers(ViewerSample::new(4, 800));
        tracker.record_viewers(ViewerSample::new(5, 300));
        let peaks = tracker.detect_viewer_peaks();
        assert_eq!(peaks.len(), 2);
        assert!(peaks[0].is_maximum);
        assert!((peaks[0].value - 500.0).abs() < f64::EPSILON);
        assert!((peaks[1].value - 800.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_engagement_score_computation() {
        let mut tracker = AudienceTracker::new(60);
        tracker.record_viewers(ViewerSample::new(10, 1000));
        tracker.record_viewers(ViewerSample::new(50, 900));
        for i in 0..60 {
            tracker.record_chat(ChatEvent::new(i, 10, i % 2 == 0));
        }
        let score = tracker.engagement_score(60);
        assert!(score.overall > 0.0);
        assert!(score.overall <= 1.0);
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = AudienceTracker::new(300);
        tracker.record_viewers(ViewerSample::new(1, 100));
        tracker.record_chat(ChatEvent::new(1, 5, false));
        tracker.reset();
        assert_eq!(tracker.viewer_sample_count(), 0);
        assert_eq!(tracker.chat_event_count(), 0);
        assert_eq!(tracker.peak_viewers(), 0);
    }

    #[test]
    fn test_peak_event_creation() {
        let peak = PeakEvent::new(100, 42.0, true);
        assert_eq!(peak.timestamp, 100);
        assert!((peak.value - 42.0).abs() < f64::EPSILON);
        assert!(peak.is_maximum);
    }
}
