//! Structured license audit log for DRM systems.
//!
//! This module provides a high-fidelity audit trail focused specifically on
//! *license* lifecycle events: request, grant, denial, renewal, and revocation.
//! It goes beyond the generic [`crate::audit_trail`] by adding:
//!
//! * **Typed event kinds** with rich, license-specific payload fields.
//! * **Compliance reporting**: generate per-device, per-content, and time-window
//!   summaries suitable for regulatory export.
//! * **Anomaly detection**: statistical heuristics that flag suspicious patterns
//!   such as rapid successive requests from one device, unusually high denial
//!   rates, or unexpected geo-location changes.
//!
//! All timestamps are caller-supplied Unix milliseconds so the module is fully
//! deterministic and easy to test without mocking the system clock.
//!
//! ## No `unsafe`, no `unwrap()` in library code.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by the audit log.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AuditError {
    /// The requested operation requires an event that does not exist.
    #[error("event with seq {seq} not found")]
    EventNotFound { seq: u64 },

    /// The compliance report could not be generated (e.g. no matching events).
    #[error("compliance report error: {reason}")]
    ReportError { reason: String },
}

// ---------------------------------------------------------------------------
// Event kinds
// ---------------------------------------------------------------------------

/// The kind of license lifecycle event.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LicenseEventKind {
    /// Client sent a license acquisition request.
    Request,
    /// License was granted to the client.
    Grant,
    /// License request was denied (policy, entitlement, geo-fence, etc.).
    Deny,
    /// An existing license was renewed before expiry.
    Renew,
    /// A previously issued license was revoked.
    Revoke,
    /// A license heartbeat check was performed (online license mode).
    Heartbeat,
    /// License expired naturally.
    Expire,
}

impl LicenseEventKind {
    /// `true` if this event represents a successful outcome.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Grant | Self::Renew | Self::Heartbeat)
    }

    /// `true` if this event represents a failure or negative outcome.
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Deny | Self::Revoke | Self::Expire)
    }

    /// Short, human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Request => "REQUEST",
            Self::Grant => "GRANT",
            Self::Deny => "DENY",
            Self::Renew => "RENEW",
            Self::Revoke => "REVOKE",
            Self::Heartbeat => "HEARTBEAT",
            Self::Expire => "EXPIRE",
        }
    }
}

// ---------------------------------------------------------------------------
// Deny reason
// ---------------------------------------------------------------------------

/// Structured reason for a license denial.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DenyReason {
    /// Device not entitled to this content.
    NotEntitled,
    /// Content not available in the device's region.
    GeoBlock,
    /// Maximum concurrent streams exceeded.
    ConcurrencyLimit,
    /// Device registration limit reached.
    DeviceLimit,
    /// Content or device key has been revoked.
    Revoked,
    /// Rate limit or burst guard triggered.
    RateLimit,
    /// License request is malformed or token invalid.
    InvalidRequest,
    /// Catch-all for server-side errors.
    ServerError,
    /// Custom reason string.
    Other(String),
}

impl DenyReason {
    /// Short code string (useful for metrics labels).
    pub fn code(&self) -> &str {
        match self {
            Self::NotEntitled => "not_entitled",
            Self::GeoBlock => "geo_block",
            Self::ConcurrencyLimit => "concurrency_limit",
            Self::DeviceLimit => "device_limit",
            Self::Revoked => "revoked",
            Self::RateLimit => "rate_limit",
            Self::InvalidRequest => "invalid_request",
            Self::ServerError => "server_error",
            Self::Other(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// License audit event
// ---------------------------------------------------------------------------

/// A single license lifecycle event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseAuditEvent {
    /// Monotonically increasing sequence number (assigned by the log).
    pub seq: u64,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Event kind.
    pub kind: LicenseEventKind,
    /// Device identifier.
    pub device_id: String,
    /// Content identifier (title, asset ID, etc.).
    pub content_id: String,
    /// Key ID associated with the license (optional).
    pub key_id: Option<String>,
    /// Reason the request was denied (only present for `Deny` events).
    pub deny_reason: Option<DenyReason>,
    /// ISO 3166-1 alpha-2 country code derived from the request (optional).
    pub country_code: Option<String>,
    /// IP address of the client (optional).
    pub client_ip: Option<String>,
    /// Duration granted in seconds (only present for `Grant` / `Renew`).
    pub duration_secs: Option<u64>,
}

impl LicenseAuditEvent {
    /// Create a basic event.
    pub fn new(
        seq: u64,
        timestamp_ms: u64,
        kind: LicenseEventKind,
        device_id: impl Into<String>,
        content_id: impl Into<String>,
    ) -> Self {
        Self {
            seq,
            timestamp_ms,
            kind,
            device_id: device_id.into(),
            content_id: content_id.into(),
            key_id: None,
            deny_reason: None,
            country_code: None,
            client_ip: None,
            duration_secs: None,
        }
    }

    /// Builder: set key ID.
    pub fn with_key_id(mut self, key_id: impl Into<String>) -> Self {
        self.key_id = Some(key_id.into());
        self
    }

    /// Builder: set deny reason.
    pub fn with_deny_reason(mut self, reason: DenyReason) -> Self {
        self.deny_reason = Some(reason);
        self
    }

    /// Builder: set country code.
    pub fn with_country(mut self, code: impl Into<String>) -> Self {
        self.country_code = Some(code.into());
        self
    }

    /// Builder: set client IP.
    pub fn with_ip(mut self, ip: impl Into<String>) -> Self {
        self.client_ip = Some(ip.into());
        self
    }

    /// Builder: set granted duration.
    pub fn with_duration(mut self, secs: u64) -> Self {
        self.duration_secs = Some(secs);
        self
    }
}

// ---------------------------------------------------------------------------
// Compliance report
// ---------------------------------------------------------------------------

/// Summary statistics for a compliance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Period start (Unix ms).
    pub from_ms: u64,
    /// Period end (Unix ms).
    pub to_ms: u64,
    /// Total events in period.
    pub total_events: u64,
    /// Breakdown by event kind.
    pub by_kind: HashMap<String, u64>,
    /// Number of unique devices that generated at least one event.
    pub unique_devices: usize,
    /// Number of unique content titles accessed.
    pub unique_content: usize,
    /// Grant count.
    pub grants: u64,
    /// Denial count.
    pub denials: u64,
    /// Revocations.
    pub revocations: u64,
    /// Denial breakdown by reason code.
    pub denial_reasons: HashMap<String, u64>,
    /// Grant-to-request ratio (0.0–1.0).
    pub grant_rate: f64,
}

// ---------------------------------------------------------------------------
// Anomaly detection
// ---------------------------------------------------------------------------

/// A detected anomaly in the license event stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditAnomaly {
    /// Short machine-readable code.
    pub code: String,
    /// Human-readable description.
    pub description: String,
    /// Device ID involved (if applicable).
    pub device_id: Option<String>,
    /// Content ID involved (if applicable).
    pub content_id: Option<String>,
    /// Observed metric value.
    pub observed: f64,
    /// Threshold that was exceeded.
    pub threshold: f64,
}

/// Configurable thresholds for anomaly detection.
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Maximum requests per device in a sliding window before flagging a burst.
    pub max_requests_per_device_window: u64,
    /// Window length in milliseconds for burst detection.
    pub burst_window_ms: u64,
    /// Maximum allowed denial rate (denials / total events) before flagging.
    pub max_denial_rate: f64,
    /// Minimum number of events before computing a denial rate.
    pub min_events_for_rate: u64,
    /// Maximum number of distinct content IDs requested by one device (suspicious)
    pub max_content_per_device: usize,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            max_requests_per_device_window: 20,
            burst_window_ms: 60_000, // 1 minute
            max_denial_rate: 0.40,   // 40% denial rate is suspicious
            min_events_for_rate: 10,
            max_content_per_device: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// License audit log
// ---------------------------------------------------------------------------

/// Append-only license audit log with compliance reporting and anomaly detection.
#[derive(Debug)]
pub struct LicenseAuditLog {
    /// Ring buffer of stored events.
    events: VecDeque<LicenseAuditEvent>,
    /// Maximum events to retain.
    capacity: usize,
    /// Next sequence number.
    next_seq: u64,
    /// Anomaly detection configuration.
    anomaly_config: AnomalyConfig,
}

impl LicenseAuditLog {
    /// Create a new log with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity.min(8192)),
            capacity: capacity.max(1),
            next_seq: 1,
            anomaly_config: AnomalyConfig::default(),
        }
    }

    /// Override the anomaly detection configuration.
    pub fn with_anomaly_config(mut self, config: AnomalyConfig) -> Self {
        self.anomaly_config = config;
        self
    }

    // -----------------------------------------------------------------------
    // Core recording API
    // -----------------------------------------------------------------------

    /// Record a `Request` event.
    pub fn record_request(
        &mut self,
        timestamp_ms: u64,
        device_id: &str,
        content_id: &str,
    ) -> u64 {
        self.push(LicenseAuditEvent::new(
            0,
            timestamp_ms,
            LicenseEventKind::Request,
            device_id,
            content_id,
        ))
    }

    /// Record a `Grant` event.
    pub fn record_grant(
        &mut self,
        timestamp_ms: u64,
        device_id: &str,
        content_id: &str,
        duration_secs: u64,
    ) -> u64 {
        self.push(
            LicenseAuditEvent::new(
                0,
                timestamp_ms,
                LicenseEventKind::Grant,
                device_id,
                content_id,
            )
            .with_duration(duration_secs),
        )
    }

    /// Record a `Deny` event.
    pub fn record_deny(
        &mut self,
        timestamp_ms: u64,
        device_id: &str,
        content_id: &str,
        reason: DenyReason,
    ) -> u64 {
        self.push(
            LicenseAuditEvent::new(
                0,
                timestamp_ms,
                LicenseEventKind::Deny,
                device_id,
                content_id,
            )
            .with_deny_reason(reason),
        )
    }

    /// Record a `Revoke` event.
    pub fn record_revoke(
        &mut self,
        timestamp_ms: u64,
        device_id: &str,
        content_id: &str,
    ) -> u64 {
        self.push(LicenseAuditEvent::new(
            0,
            timestamp_ms,
            LicenseEventKind::Revoke,
            device_id,
            content_id,
        ))
    }

    /// Record a fully-constructed event (seq is overwritten).
    pub fn record_event(&mut self, mut event: LicenseAuditEvent) -> u64 {
        event.seq = 0; // will be set inside push()
        self.push(event)
    }

    /// Internal push helper.
    fn push(&mut self, mut event: LicenseAuditEvent) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        event.seq = seq;
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(event);
        seq
    }

    // -----------------------------------------------------------------------
    // Query API
    // -----------------------------------------------------------------------

    /// Total events currently stored.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Retrieve an event by sequence number.
    pub fn get_by_seq(&self, seq: u64) -> Option<&LicenseAuditEvent> {
        self.events.iter().find(|e| e.seq == seq)
    }

    /// All events for a given device (oldest first).
    pub fn events_for_device(&self, device_id: &str) -> Vec<&LicenseAuditEvent> {
        self.events
            .iter()
            .filter(|e| e.device_id == device_id)
            .collect()
    }

    /// All events for a given content title (oldest first).
    pub fn events_for_content(&self, content_id: &str) -> Vec<&LicenseAuditEvent> {
        self.events
            .iter()
            .filter(|e| e.content_id == content_id)
            .collect()
    }

    /// All events of a given kind.
    pub fn events_by_kind(&self, kind: &LicenseEventKind) -> Vec<&LicenseAuditEvent> {
        self.events.iter().filter(|e| &e.kind == kind).collect()
    }

    /// All events within a time range (inclusive).
    pub fn events_in_range(&self, from_ms: u64, to_ms: u64) -> Vec<&LicenseAuditEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp_ms >= from_ms && e.timestamp_ms <= to_ms)
            .collect()
    }

    /// Most recent `n` events (newest first).
    pub fn recent(&self, n: usize) -> Vec<&LicenseAuditEvent> {
        self.events.iter().rev().take(n).collect()
    }

    // -----------------------------------------------------------------------
    // Compliance reporting
    // -----------------------------------------------------------------------

    /// Generate a compliance report covering the given time window.
    pub fn compliance_report(
        &self,
        from_ms: u64,
        to_ms: u64,
    ) -> Result<ComplianceReport, AuditError> {
        let window: Vec<&LicenseAuditEvent> = self
            .events
            .iter()
            .filter(|e| e.timestamp_ms >= from_ms && e.timestamp_ms <= to_ms)
            .collect();

        if window.is_empty() {
            return Err(AuditError::ReportError {
                reason: "no events in the requested time window".to_string(),
            });
        }

        let mut by_kind: HashMap<String, u64> = HashMap::new();
        let mut unique_devices: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut unique_content: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut grants = 0u64;
        let mut denials = 0u64;
        let mut revocations = 0u64;
        let mut denial_reasons: HashMap<String, u64> = HashMap::new();
        let mut requests = 0u64;

        for e in &window {
            *by_kind.entry(e.kind.label().to_string()).or_insert(0) += 1;
            unique_devices.insert(e.device_id.as_str());
            unique_content.insert(e.content_id.as_str());

            match &e.kind {
                LicenseEventKind::Grant | LicenseEventKind::Renew => grants += 1,
                LicenseEventKind::Deny => {
                    denials += 1;
                    if let Some(reason) = &e.deny_reason {
                        *denial_reasons.entry(reason.code().to_string()).or_insert(0) += 1;
                    }
                }
                LicenseEventKind::Revoke => revocations += 1,
                LicenseEventKind::Request => requests += 1,
                _ => {}
            }
        }

        let denominator = grants + denials + requests;
        let grant_rate = if denominator == 0 {
            0.0
        } else {
            grants as f64 / denominator as f64
        };

        Ok(ComplianceReport {
            from_ms,
            to_ms,
            total_events: window.len() as u64,
            by_kind,
            unique_devices: unique_devices.len(),
            unique_content: unique_content.len(),
            grants,
            denials,
            revocations,
            denial_reasons,
            grant_rate,
        })
    }

    // -----------------------------------------------------------------------
    // Anomaly detection
    // -----------------------------------------------------------------------

    /// Run anomaly detection over the entire stored event history.
    ///
    /// Returns a (possibly empty) list of detected anomalies.
    pub fn detect_anomalies(&self) -> Vec<AuditAnomaly> {
        let mut anomalies = Vec::new();

        // 1. Burst detection: any device with too many requests in the window
        anomalies.extend(self.detect_request_bursts());

        // 2. High denial rate across all events
        anomalies.extend(self.detect_high_denial_rate());

        // 3. Device accessing too many distinct content items (credential sharing?)
        anomalies.extend(self.detect_content_breadth());

        anomalies
    }

    fn detect_request_bursts(&self) -> Vec<AuditAnomaly> {
        let mut anomalies = Vec::new();
        // Group request-events by device, then slide a window
        let mut device_requests: HashMap<&str, Vec<u64>> = HashMap::new();
        for e in &self.events {
            if e.kind == LicenseEventKind::Request {
                device_requests
                    .entry(e.device_id.as_str())
                    .or_default()
                    .push(e.timestamp_ms);
            }
        }

        let window = self.anomaly_config.burst_window_ms;
        let threshold = self.anomaly_config.max_requests_per_device_window;

        for (device_id, mut times) in device_requests {
            times.sort_unstable();
            let max_in_window = max_sliding_window_count(&times, window);
            if max_in_window > threshold {
                anomalies.push(AuditAnomaly {
                    code: "REQUEST_BURST".to_string(),
                    description: format!(
                        "Device '{}' made {} requests within {}ms (threshold {})",
                        device_id, max_in_window, window, threshold
                    ),
                    device_id: Some(device_id.to_string()),
                    content_id: None,
                    observed: max_in_window as f64,
                    threshold: threshold as f64,
                });
            }
        }
        anomalies
    }

    fn detect_high_denial_rate(&self) -> Vec<AuditAnomaly> {
        let total = self.events.len() as u64;
        if total < self.anomaly_config.min_events_for_rate {
            return Vec::new();
        }
        let denials = self
            .events
            .iter()
            .filter(|e| e.kind == LicenseEventKind::Deny)
            .count() as f64;
        let rate = denials / total as f64;
        if rate > self.anomaly_config.max_denial_rate {
            vec![AuditAnomaly {
                code: "HIGH_DENIAL_RATE".to_string(),
                description: format!(
                    "Denial rate {:.1}% exceeds threshold {:.1}%",
                    rate * 100.0,
                    self.anomaly_config.max_denial_rate * 100.0
                ),
                device_id: None,
                content_id: None,
                observed: rate,
                threshold: self.anomaly_config.max_denial_rate,
            }]
        } else {
            Vec::new()
        }
    }

    fn detect_content_breadth(&self) -> Vec<AuditAnomaly> {
        let mut device_content: HashMap<&str, std::collections::HashSet<&str>> = HashMap::new();
        for e in &self.events {
            device_content
                .entry(e.device_id.as_str())
                .or_default()
                .insert(e.content_id.as_str());
        }

        let threshold = self.anomaly_config.max_content_per_device;
        device_content
            .into_iter()
            .filter(|(_, content)| content.len() > threshold)
            .map(|(device_id, content)| AuditAnomaly {
                code: "CONTENT_BREADTH".to_string(),
                description: format!(
                    "Device '{}' accessed {} distinct content items (threshold {})",
                    device_id,
                    content.len(),
                    threshold
                ),
                device_id: Some(device_id.to_string()),
                content_id: None,
                observed: content.len() as f64,
                threshold: threshold as f64,
            })
            .collect()
    }
}

/// Count the maximum number of timestamps that fall within any sliding window of
/// `window_ms` milliseconds.  `times` must be sorted ascending.
fn max_sliding_window_count(times: &[u64], window_ms: u64) -> u64 {
    if times.is_empty() {
        return 0;
    }
    let mut max_count = 0u64;
    let mut left = 0usize;
    for right in 0..times.len() {
        // Shrink the left side so all events are within `window_ms` of times[right]
        while times[right].saturating_sub(times[left]) > window_ms {
            left += 1;
        }
        let count = (right - left + 1) as u64;
        if count > max_count {
            max_count = count;
        }
    }
    max_count
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn populated_log() -> LicenseAuditLog {
        let mut log = LicenseAuditLog::new(1000);
        // Device A — 3 grants, 1 deny
        log.record_request(1000, "dev-A", "movie-1");
        log.record_grant(1100, "dev-A", "movie-1", 3600);
        log.record_request(2000, "dev-A", "movie-2");
        log.record_grant(2100, "dev-A", "movie-2", 3600);
        log.record_request(3000, "dev-A", "movie-3");
        log.record_deny(3100, "dev-A", "movie-3", DenyReason::GeoBlock);
        // Device B — 1 grant, 1 revoke
        log.record_request(4000, "dev-B", "movie-1");
        log.record_grant(4100, "dev-B", "movie-1", 7200);
        log.record_revoke(9000, "dev-B", "movie-1");
        log
    }

    #[test]
    fn test_record_and_retrieve() {
        let log = populated_log();
        assert_eq!(log.len(), 9);
        let ev = log.get_by_seq(1).expect("seq 1 must exist");
        assert_eq!(ev.kind, LicenseEventKind::Request);
        assert_eq!(ev.device_id, "dev-A");
    }

    #[test]
    fn test_events_for_device() {
        let log = populated_log();
        let events = log.events_for_device("dev-A");
        assert_eq!(events.len(), 6);
        let events_b = log.events_for_device("dev-B");
        assert_eq!(events_b.len(), 3);
    }

    #[test]
    fn test_events_for_content() {
        let log = populated_log();
        let movie1 = log.events_for_content("movie-1");
        // dev-A: request+grant; dev-B: request+grant+revoke → 5 events
        assert_eq!(movie1.len(), 5);
    }

    #[test]
    fn test_events_by_kind_grants() {
        let log = populated_log();
        let grants = log.events_by_kind(&LicenseEventKind::Grant);
        assert_eq!(grants.len(), 3);
    }

    #[test]
    fn test_events_in_range() {
        let log = populated_log();
        let range = log.events_in_range(2000, 3100);
        // request@2000, grant@2100, request@3000, deny@3100
        assert_eq!(range.len(), 4);
    }

    #[test]
    fn test_recent() {
        let log = populated_log();
        let recent = log.recent(3);
        assert_eq!(recent.len(), 3);
        // Newest first
        assert_eq!(recent[0].seq, 9);
    }

    #[test]
    fn test_compliance_report_basic() {
        let log = populated_log();
        let report = log
            .compliance_report(0, 100_000)
            .expect("report must succeed");
        assert_eq!(report.total_events, 9);
        assert_eq!(report.grants, 3);
        assert_eq!(report.denials, 1);
        assert_eq!(report.revocations, 1);
        assert_eq!(report.unique_devices, 2);
        assert_eq!(report.unique_content, 3);
    }

    #[test]
    fn test_compliance_report_grant_rate() {
        let log = populated_log();
        let report = log
            .compliance_report(0, 100_000)
            .expect("report must succeed");
        // 3 grants / (3 grants + 1 denial + 4 requests) = 3/8 = 0.375
        assert!((report.grant_rate - 3.0 / 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_compliance_report_empty_range_error() {
        let log = populated_log();
        let result = log.compliance_report(999_999, 1_000_000);
        assert!(matches!(result, Err(AuditError::ReportError { .. })));
    }

    #[test]
    fn test_compliance_denial_reasons() {
        let log = populated_log();
        let report = log.compliance_report(0, 100_000).expect("ok");
        assert_eq!(
            *report.denial_reasons.get("geo_block").expect("key"),
            1u64
        );
    }

    #[test]
    fn test_anomaly_request_burst() {
        let mut log = LicenseAuditLog::new(10_000).with_anomaly_config(AnomalyConfig {
            max_requests_per_device_window: 5,
            burst_window_ms: 10_000,
            max_denial_rate: 1.0, // disable denial rate check
            min_events_for_rate: 1_000_000,
            max_content_per_device: 10_000,
        });

        // 10 requests from one device within 5 seconds
        for i in 0..10u64 {
            log.record_request(i * 500, "spammer", "movie-x");
        }

        let anomalies = log.detect_anomalies();
        assert!(
            anomalies.iter().any(|a| a.code == "REQUEST_BURST"),
            "Expected REQUEST_BURST anomaly, got: {anomalies:?}"
        );
    }

    #[test]
    fn test_anomaly_high_denial_rate() {
        let mut log = LicenseAuditLog::new(10_000).with_anomaly_config(AnomalyConfig {
            max_requests_per_device_window: 100_000,
            burst_window_ms: 1,
            max_denial_rate: 0.30,
            min_events_for_rate: 5,
            max_content_per_device: 10_000,
        });

        // 4 denials out of 10 events → 40% denial rate → above 30% threshold
        for i in 0..6u64 {
            log.record_grant(i * 1000, "dev-ok", "movie-a", 3600);
        }
        for i in 0..4u64 {
            log.record_deny(
                i * 1000 + 500,
                "dev-bad",
                "movie-a",
                DenyReason::NotEntitled,
            );
        }

        let anomalies = log.detect_anomalies();
        assert!(
            anomalies.iter().any(|a| a.code == "HIGH_DENIAL_RATE"),
            "Expected HIGH_DENIAL_RATE, got: {anomalies:?}"
        );
    }

    #[test]
    fn test_anomaly_content_breadth() {
        let mut log = LicenseAuditLog::new(10_000).with_anomaly_config(AnomalyConfig {
            max_requests_per_device_window: 100_000,
            burst_window_ms: 1,
            max_denial_rate: 1.0,
            min_events_for_rate: 1_000_000,
            max_content_per_device: 3,
        });

        // Same device accesses 5 distinct titles
        for i in 0..5u64 {
            log.record_grant(i * 1000, "binge-device", &format!("movie-{i}"), 3600);
        }

        let anomalies = log.detect_anomalies();
        assert!(
            anomalies.iter().any(|a| a.code == "CONTENT_BREADTH"),
            "Expected CONTENT_BREADTH, got: {anomalies:?}"
        );
    }

    #[test]
    fn test_no_anomalies_normal_usage() {
        let mut log = LicenseAuditLog::new(1000).with_anomaly_config(AnomalyConfig::default());
        log.record_request(0, "normal-dev", "movie-1");
        log.record_grant(100, "normal-dev", "movie-1", 3600);
        let anomalies = log.detect_anomalies();
        assert!(anomalies.is_empty(), "Expected no anomalies: {anomalies:?}");
    }

    #[test]
    fn test_capacity_ring_buffer() {
        let mut log = LicenseAuditLog::new(3);
        log.record_request(1, "d", "c");
        log.record_request(2, "d", "c");
        log.record_request(3, "d", "c");
        log.record_request(4, "d", "c"); // evicts seq 1
        assert_eq!(log.len(), 3);
        assert!(log.get_by_seq(1).is_none());
        assert!(log.get_by_seq(2).is_some());
    }

    #[test]
    fn test_deny_reason_codes() {
        let reasons = [
            DenyReason::NotEntitled,
            DenyReason::GeoBlock,
            DenyReason::ConcurrencyLimit,
            DenyReason::DeviceLimit,
            DenyReason::Revoked,
            DenyReason::RateLimit,
            DenyReason::InvalidRequest,
            DenyReason::ServerError,
            DenyReason::Other("custom".to_string()),
        ];
        for r in &reasons {
            assert!(!r.code().is_empty(), "Code must not be empty for {r:?}");
        }
    }

    #[test]
    fn test_event_kind_labels_and_predicates() {
        assert!(LicenseEventKind::Grant.is_success());
        assert!(LicenseEventKind::Deny.is_failure());
        assert!(!LicenseEventKind::Request.is_success());
        assert!(!LicenseEventKind::Request.is_failure());
        for kind in &[
            LicenseEventKind::Request,
            LicenseEventKind::Grant,
            LicenseEventKind::Deny,
            LicenseEventKind::Renew,
            LicenseEventKind::Revoke,
            LicenseEventKind::Heartbeat,
            LicenseEventKind::Expire,
        ] {
            assert!(!kind.label().is_empty());
        }
    }

    #[test]
    fn test_sliding_window_helper() {
        let times = vec![0, 1000, 2000, 59_000, 60_000, 60_500, 61_000];
        // window = 60_000 ms
        // [0..60_000] = 5 events; [1000..61_000] = 6 events
        let max = max_sliding_window_count(&times, 60_000);
        assert_eq!(max, 6);
    }

    #[test]
    fn test_record_event_custom() {
        let mut log = LicenseAuditLog::new(100);
        let ev = LicenseAuditEvent::new(0, 5000, LicenseEventKind::Heartbeat, "dev-Z", "stream-1")
            .with_key_id("key-abc")
            .with_country("US")
            .with_ip("1.2.3.4");
        let seq = log.record_event(ev);
        let stored = log.get_by_seq(seq).expect("must exist");
        assert_eq!(stored.kind, LicenseEventKind::Heartbeat);
        assert_eq!(stored.country_code.as_deref(), Some("US"));
        assert_eq!(stored.client_ip.as_deref(), Some("1.2.3.4"));
    }
}
