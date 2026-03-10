//! Audit trail for DRM license and key events.
//!
//! Records every license acquisition, key request, policy decision, and
//! playback session event for compliance and forensics.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Category of DRM event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditCategory {
    /// License acquisition (request/grant/deny).
    License,
    /// Key operations (rotation, revocation, derivation).
    Key,
    /// Policy evaluation (allow/deny).
    Policy,
    /// Playback session (start/stop/heartbeat).
    Playback,
    /// Device registration and authentication.
    Device,
    /// Administrative action.
    Admin,
}

/// Severity level for an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AuditSeverity {
    /// Informational only.
    Info,
    /// Potentially noteworthy.
    Warning,
    /// Something went wrong.
    Error,
    /// Security-critical event.
    Critical,
}

/// A single audit event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Monotonically increasing sequence number.
    pub seq: u64,
    /// Epoch timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Category of the event.
    pub category: AuditCategory,
    /// Severity level.
    pub severity: AuditSeverity,
    /// Actor who triggered the event (device ID, user ID, system, etc.).
    pub actor: String,
    /// Content ID (if applicable).
    pub content_id: Option<String>,
    /// Short description of what happened.
    pub message: String,
    /// Optional detail payload (JSON-encodable string).
    pub detail: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event.
    #[must_use]
    pub fn new(
        seq: u64,
        timestamp_ms: u64,
        category: AuditCategory,
        severity: AuditSeverity,
        actor: &str,
        message: &str,
    ) -> Self {
        Self {
            seq,
            timestamp_ms,
            category,
            severity,
            actor: actor.to_string(),
            content_id: None,
            message: message.to_string(),
            detail: None,
        }
    }

    /// Set the content ID.
    #[must_use]
    pub fn with_content(mut self, content_id: &str) -> Self {
        self.content_id = Some(content_id.to_string());
        self
    }

    /// Set the detail payload.
    #[must_use]
    pub fn with_detail(mut self, detail: &str) -> Self {
        self.detail = Some(detail.to_string());
        self
    }

    /// Check if this event is at least the given severity.
    #[must_use]
    pub fn is_at_least(&self, severity: AuditSeverity) -> bool {
        self.severity >= severity
    }
}

/// Statistics about the audit trail.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditStats {
    /// Total events recorded.
    pub total: u64,
    /// Events by category.
    pub by_category: std::collections::HashMap<String, u64>,
    /// Events by severity.
    pub by_severity: std::collections::HashMap<String, u64>,
}

/// In-memory audit trail with configurable retention.
#[derive(Debug, Clone)]
pub struct AuditTrail {
    /// Maximum number of events to retain (ring-buffer semantics).
    capacity: usize,
    /// Next sequence number.
    next_seq: u64,
    /// Events stored in a ring buffer.
    events: VecDeque<AuditEvent>,
}

impl AuditTrail {
    /// Create a new audit trail with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            next_seq: 1,
            events: VecDeque::with_capacity(capacity.min(8192)),
        }
    }

    /// Record an event, assigning it the next sequence number.
    pub fn record(
        &mut self,
        timestamp_ms: u64,
        category: AuditCategory,
        severity: AuditSeverity,
        actor: &str,
        message: &str,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let event = AuditEvent::new(seq, timestamp_ms, category, severity, actor, message);
        self.push_event(event);
        seq
    }

    /// Record a fully constructed event (seq is overwritten).
    pub fn record_event(&mut self, mut event: AuditEvent) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        event.seq = seq;
        self.push_event(event);
        seq
    }

    /// Push an event, evicting the oldest if at capacity.
    fn push_event(&mut self, event: AuditEvent) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Number of events currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the trail is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get an event by sequence number.
    #[must_use]
    pub fn get_by_seq(&self, seq: u64) -> Option<&AuditEvent> {
        self.events.iter().find(|e| e.seq == seq)
    }

    /// Get the most recent N events (newest first).
    #[must_use]
    pub fn recent(&self, n: usize) -> Vec<&AuditEvent> {
        self.events.iter().rev().take(n).collect()
    }

    /// Query events by category.
    #[must_use]
    pub fn by_category(&self, category: AuditCategory) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Query events by severity (at least the given level).
    #[must_use]
    pub fn by_min_severity(&self, severity: AuditSeverity) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.is_at_least(severity))
            .collect()
    }

    /// Query events within a time range (inclusive).
    #[must_use]
    pub fn in_time_range(&self, from_ms: u64, to_ms: u64) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp_ms >= from_ms && e.timestamp_ms <= to_ms)
            .collect()
    }

    /// Query events by actor.
    #[must_use]
    pub fn by_actor(&self, actor: &str) -> Vec<&AuditEvent> {
        self.events.iter().filter(|e| e.actor == actor).collect()
    }

    /// Compute summary statistics.
    #[must_use]
    pub fn stats(&self) -> AuditStats {
        let mut s = AuditStats {
            total: self.events.len() as u64,
            ..Default::default()
        };
        for e in &self.events {
            *s.by_category
                .entry(format!("{:?}", e.category))
                .or_insert(0) += 1;
            *s.by_severity
                .entry(format!("{:?}", e.severity))
                .or_insert(0) += 1;
        }
        s
    }

    /// Clear all events (resets sequence counter too).
    pub fn clear(&mut self) {
        self.events.clear();
        self.next_seq = 1;
    }

    /// Capacity of the trail.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_new() {
        let e = AuditEvent::new(
            1,
            1000,
            AuditCategory::License,
            AuditSeverity::Info,
            "dev1",
            "acquired",
        );
        assert_eq!(e.seq, 1);
        assert_eq!(e.timestamp_ms, 1000);
        assert_eq!(e.actor, "dev1");
        assert!(e.content_id.is_none());
    }

    #[test]
    fn test_event_builder() {
        let e = AuditEvent::new(
            1,
            100,
            AuditCategory::Key,
            AuditSeverity::Warning,
            "sys",
            "rotated",
        )
        .with_content("movie42")
        .with_detail("{\"old_key\":\"abc\"}");
        assert_eq!(e.content_id.as_deref(), Some("movie42"));
        assert!(e.detail.is_some());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(AuditSeverity::Info < AuditSeverity::Warning);
        assert!(AuditSeverity::Warning < AuditSeverity::Error);
        assert!(AuditSeverity::Error < AuditSeverity::Critical);
    }

    #[test]
    fn test_is_at_least() {
        let e = AuditEvent::new(1, 0, AuditCategory::Policy, AuditSeverity::Error, "a", "b");
        assert!(e.is_at_least(AuditSeverity::Info));
        assert!(e.is_at_least(AuditSeverity::Error));
        assert!(!e.is_at_least(AuditSeverity::Critical));
    }

    #[test]
    fn test_trail_record() {
        let mut trail = AuditTrail::new(100);
        let seq = trail.record(
            500,
            AuditCategory::License,
            AuditSeverity::Info,
            "dev",
            "ok",
        );
        assert_eq!(seq, 1);
        assert_eq!(trail.len(), 1);
    }

    #[test]
    fn test_trail_capacity() {
        let mut trail = AuditTrail::new(3);
        trail.record(1, AuditCategory::Key, AuditSeverity::Info, "a", "1");
        trail.record(2, AuditCategory::Key, AuditSeverity::Info, "a", "2");
        trail.record(3, AuditCategory::Key, AuditSeverity::Info, "a", "3");
        trail.record(4, AuditCategory::Key, AuditSeverity::Info, "a", "4");
        // Oldest (seq=1) should have been evicted.
        assert_eq!(trail.len(), 3);
        assert!(trail.get_by_seq(1).is_none());
        assert!(trail.get_by_seq(2).is_some());
    }

    #[test]
    fn test_trail_recent() {
        let mut trail = AuditTrail::new(100);
        trail.record(1, AuditCategory::Playback, AuditSeverity::Info, "x", "a");
        trail.record(2, AuditCategory::Playback, AuditSeverity::Info, "x", "b");
        trail.record(3, AuditCategory::Playback, AuditSeverity::Info, "x", "c");
        let recent = trail.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].message, "c");
        assert_eq!(recent[1].message, "b");
    }

    #[test]
    fn test_trail_by_category() {
        let mut trail = AuditTrail::new(100);
        trail.record(1, AuditCategory::License, AuditSeverity::Info, "a", "lic");
        trail.record(2, AuditCategory::Key, AuditSeverity::Info, "a", "key");
        trail.record(3, AuditCategory::License, AuditSeverity::Error, "a", "lic2");
        let lic = trail.by_category(AuditCategory::License);
        assert_eq!(lic.len(), 2);
    }

    #[test]
    fn test_trail_by_severity() {
        let mut trail = AuditTrail::new(100);
        trail.record(1, AuditCategory::Policy, AuditSeverity::Info, "a", "i");
        trail.record(2, AuditCategory::Policy, AuditSeverity::Error, "a", "e");
        trail.record(3, AuditCategory::Policy, AuditSeverity::Critical, "a", "c");
        let errors = trail.by_min_severity(AuditSeverity::Error);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_trail_time_range() {
        let mut trail = AuditTrail::new(100);
        trail.record(100, AuditCategory::Device, AuditSeverity::Info, "a", "x");
        trail.record(200, AuditCategory::Device, AuditSeverity::Info, "a", "y");
        trail.record(300, AuditCategory::Device, AuditSeverity::Info, "a", "z");
        let range = trail.in_time_range(150, 250);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].message, "y");
    }

    #[test]
    fn test_trail_by_actor() {
        let mut trail = AuditTrail::new(100);
        trail.record(1, AuditCategory::Admin, AuditSeverity::Info, "alice", "a");
        trail.record(2, AuditCategory::Admin, AuditSeverity::Info, "bob", "b");
        assert_eq!(trail.by_actor("alice").len(), 1);
        assert_eq!(trail.by_actor("bob").len(), 1);
        assert_eq!(trail.by_actor("eve").len(), 0);
    }

    #[test]
    fn test_trail_stats() {
        let mut trail = AuditTrail::new(100);
        trail.record(1, AuditCategory::License, AuditSeverity::Info, "a", "x");
        trail.record(2, AuditCategory::License, AuditSeverity::Error, "a", "y");
        trail.record(3, AuditCategory::Key, AuditSeverity::Info, "a", "z");
        let stats = trail.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(
            *stats
                .by_category
                .get("License")
                .expect("entry should exist"),
            2
        );
        assert_eq!(
            *stats.by_category.get("Key").expect("entry should exist"),
            1
        );
    }

    #[test]
    fn test_trail_clear() {
        let mut trail = AuditTrail::new(100);
        trail.record(1, AuditCategory::Playback, AuditSeverity::Info, "a", "x");
        assert!(!trail.is_empty());
        trail.clear();
        assert!(trail.is_empty());
        // Sequence resets.
        let seq = trail.record(10, AuditCategory::Playback, AuditSeverity::Info, "a", "y");
        assert_eq!(seq, 1);
    }

    #[test]
    fn test_record_event() {
        let mut trail = AuditTrail::new(100);
        let event = AuditEvent::new(
            999,
            50,
            AuditCategory::Admin,
            AuditSeverity::Critical,
            "root",
            "wipe",
        )
        .with_content("all");
        let seq = trail.record_event(event);
        assert_eq!(seq, 1); // Seq is overwritten.
        let stored = trail.get_by_seq(1).expect("entry should exist");
        assert_eq!(stored.message, "wipe");
        assert_eq!(stored.content_id.as_deref(), Some("all"));
    }

    #[test]
    fn test_trail_capacity_getter() {
        let trail = AuditTrail::new(42);
        assert_eq!(trail.capacity(), 42);
    }
}
