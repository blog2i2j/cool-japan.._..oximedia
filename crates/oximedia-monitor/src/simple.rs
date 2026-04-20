//! Simplified monitoring primitives: `CodecMetrics`, `SimpleMetricsCollector`,
//! `SimpleAlertRule`, `SimpleAlertManager`, and the `HealthCheck` trait.
//!
//! These types complement the full-featured monitoring infrastructure with
//! lightweight, self-contained implementations suitable for embedding directly
//! in tests and small binaries.

#![allow(dead_code)]

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// CodecMetrics
// ---------------------------------------------------------------------------

/// Per-codec performance and quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodecMetrics {
    /// Encoding / decoding frame rate (frames per second).
    pub fps: f64,
    /// Total frames encoded or decoded.
    pub frames_encoded: u64,
    /// Current output bitrate in kilobits per second.
    pub bitrate_kbps: f64,
    /// Composite quality score \[0.0, 100.0\].
    pub quality_score: f64,
    /// Peak signal-to-noise ratio (dB). `None` if not computed.
    pub psnr_db: Option<f64>,
    /// Structural similarity index \[0.0, 1.0\]. `None` if not computed.
    pub ssim: Option<f64>,
    /// Codec name / identifier (e.g. `"h264"`, `"av1"`).
    pub codec_name: String,
    /// Timestamp of the last update.
    pub timestamp: DateTime<Utc>,
}

impl CodecMetrics {
    /// Create a zeroed-out `CodecMetrics` for the given codec.
    #[must_use]
    pub fn new(codec_name: impl Into<String>) -> Self {
        Self {
            codec_name: codec_name.into(),
            timestamp: Utc::now(),
            ..Default::default()
        }
    }

    /// Returns `true` when the quality score is above `threshold`.
    #[must_use]
    pub fn quality_ok(&self, threshold: f64) -> bool {
        self.quality_score >= threshold
    }

    /// Returns `true` when the FPS is at or above `min_fps`.
    #[must_use]
    pub fn fps_ok(&self, min_fps: f64) -> bool {
        self.fps >= min_fps
    }
}

// ---------------------------------------------------------------------------
// SimpleMetricsSnapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot produced by `SimpleMetricsCollector::snapshot`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleMetricsSnapshot {
    /// Snapshot timestamp.
    pub timestamp: DateTime<Utc>,
    /// CPU usage percentage \[0.0, 100.0\].
    pub cpu_percent: f64,
    /// Memory used in megabytes.
    pub memory_mb: f64,
    /// Disk I/O throughput in MB/s.
    pub disk_io_mbps: f64,
    /// Per-codec metrics indexed by codec name.
    pub codecs: HashMap<String, CodecMetrics>,
}

impl SimpleMetricsSnapshot {
    /// Returns `true` when all per-codec quality scores exceed `threshold`.
    #[must_use]
    pub fn all_codecs_ok(&self, threshold: f64) -> bool {
        self.codecs.values().all(|c| c.quality_ok(threshold))
    }
}

// ---------------------------------------------------------------------------
// SimpleMetricsCollector
// ---------------------------------------------------------------------------

/// A lightweight metrics collector that accumulates system and codec metrics.
///
/// Call [`poll`][Self::poll] to record a new sample, then
/// [`snapshot`][Self::snapshot] to retrieve the latest view.
pub struct SimpleMetricsCollector {
    inner: Arc<Mutex<CollectorState>>,
}

struct CollectorState {
    cpu_percent: f64,
    memory_mb: f64,
    disk_io_mbps: f64,
    codecs: HashMap<String, CodecMetrics>,
    poll_count: u64,
}

impl SimpleMetricsCollector {
    /// Create a new collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CollectorState {
                cpu_percent: 0.0,
                memory_mb: 0.0,
                disk_io_mbps: 0.0,
                codecs: HashMap::new(),
                poll_count: 0,
            })),
        }
    }

    /// Record a poll sample.
    ///
    /// In a production implementation this would query `sysinfo` or similar;
    /// here callers supply values directly (useful in tests and simulations).
    pub fn poll(&self, cpu_percent: f64, memory_mb: f64, disk_io_mbps: f64) {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        state.cpu_percent = cpu_percent;
        state.memory_mb = memory_mb;
        state.disk_io_mbps = disk_io_mbps;
        state.poll_count += 1;
    }

    /// Record metrics for a specific codec.
    pub fn poll_codec(&self, metrics: CodecMetrics) {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        state.codecs.insert(metrics.codec_name.clone(), metrics);
    }

    /// Return a snapshot of the current metrics.
    #[must_use]
    pub fn snapshot(&self) -> SimpleMetricsSnapshot {
        let state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        SimpleMetricsSnapshot {
            timestamp: Utc::now(),
            cpu_percent: state.cpu_percent,
            memory_mb: state.memory_mb,
            disk_io_mbps: state.disk_io_mbps,
            codecs: state.codecs.clone(),
        }
    }

    /// Total number of `poll` calls since creation.
    #[must_use]
    pub fn poll_count(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .poll_count
    }
}

impl Default for SimpleMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SimpleAlertRule
// ---------------------------------------------------------------------------

/// Comparison operator used in `SimpleAlertRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Comparison {
    /// Fires when value > threshold.
    GreaterThan,
    /// Fires when value >= threshold.
    GreaterThanOrEqual,
    /// Fires when value < threshold.
    LessThan,
    /// Fires when value <= threshold.
    LessThanOrEqual,
    /// Fires when value == threshold.
    Equal,
}

impl Comparison {
    /// Evaluate the comparison for `value` against `threshold`.
    #[must_use]
    pub fn evaluate(self, value: f64, threshold: f64) -> bool {
        match self {
            Self::GreaterThan => value > threshold,
            Self::GreaterThanOrEqual => value >= threshold,
            Self::LessThan => value < threshold,
            Self::LessThanOrEqual => value <= threshold,
            Self::Equal => (value - threshold).abs() < f64::EPSILON,
        }
    }
}

/// Notification action triggered when an alert fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationAction {
    /// Log to stderr.
    Log(String),
    /// Record a metric name.
    RecordMetric(String),
    /// Custom free-form payload.
    Custom(String),
}

/// A self-contained alert rule: threshold + comparison + notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleAlertRule {
    /// Rule name.
    pub name: String,
    /// Metric key this rule monitors (free-form tag).
    pub metric_key: String,
    /// Comparison to perform.
    pub comparison: Comparison,
    /// Threshold value.
    pub threshold: f64,
    /// Notification action.
    pub notification: NotificationAction,
    /// Whether the rule is enabled.
    pub enabled: bool,
}

impl SimpleAlertRule {
    /// Create a new rule.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        metric_key: impl Into<String>,
        comparison: Comparison,
        threshold: f64,
        notification: NotificationAction,
    ) -> Self {
        Self {
            name: name.into(),
            metric_key: metric_key.into(),
            comparison,
            threshold,
            notification,
            enabled: true,
        }
    }

    /// Evaluate whether the rule fires for `value`.
    #[must_use]
    pub fn evaluate(&self, value: f64) -> bool {
        self.enabled && self.comparison.evaluate(value, self.threshold)
    }
}

// ---------------------------------------------------------------------------
// SimpleAlertManager
// ---------------------------------------------------------------------------

/// A record of a fired alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiredAlert {
    /// Rule that produced this alert.
    pub rule_name: String,
    /// The metric value that triggered the rule.
    pub metric_value: f64,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
}

/// Manages a set of `SimpleAlertRule`s and evaluates them against snapshots.
pub struct SimpleAlertManager {
    rules: Arc<Mutex<Vec<SimpleAlertRule>>>,
    history: Arc<Mutex<Vec<FiredAlert>>>,
}

impl SimpleAlertManager {
    /// Create an empty alert manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: Arc::new(Mutex::new(Vec::new())),
            history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Register a rule.
    pub fn add_rule(&self, rule: SimpleAlertRule) {
        self.rules
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(rule);
    }

    /// Evaluate all rules against a named metric value.
    ///
    /// Fired alerts are appended to the internal history.
    /// Returns the number of rules that fired.
    #[must_use]
    pub fn evaluate(&self, metric_key: &str, value: f64) -> usize {
        let rules = self.rules.lock().unwrap_or_else(|e| e.into_inner());
        let mut history = self.history.lock().unwrap_or_else(|e| e.into_inner());
        let mut fired = 0usize;

        for rule in rules.iter() {
            if rule.metric_key == metric_key && rule.evaluate(value) {
                history.push(FiredAlert {
                    rule_name: rule.name.clone(),
                    metric_value: value,
                    timestamp: Utc::now(),
                });
                fired += 1;
            }
        }
        fired
    }

    /// Evaluate all rules whose `metric_key` matches a key in `metrics_map`.
    #[must_use]
    pub fn evaluate_snapshot(&self, metrics_map: &HashMap<String, f64>) -> usize {
        let mut total = 0usize;
        for (key, &value) in metrics_map {
            total += self.evaluate(key, value);
        }
        total
    }

    /// Return the full firing history.
    #[must_use]
    pub fn history(&self) -> Vec<FiredAlert> {
        self.history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Clear the history.
    pub fn clear_history(&self) {
        self.history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Number of registered rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

impl Default for SimpleAlertManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HealthCheck trait
// ---------------------------------------------------------------------------

/// Status returned by a `HealthCheck`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Component is operating normally.
    Healthy,
    /// Component is degraded but functional.
    Degraded(String),
    /// Component is not functioning.
    Unhealthy(String),
}

impl HealthStatus {
    /// Returns `true` for `Healthy`.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Returns `true` for `Unhealthy`.
    #[must_use]
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, Self::Unhealthy(_))
    }

    /// Short string summary.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded(_) => "degraded",
            Self::Unhealthy(_) => "unhealthy",
        }
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded(msg) => write!(f, "degraded: {msg}"),
            Self::Unhealthy(msg) => write!(f, "unhealthy: {msg}"),
        }
    }
}

/// Trait for components that can report their health status.
#[async_trait]
pub trait HealthCheck: Send + Sync {
    /// Component name.
    fn name(&self) -> &str;

    /// Perform the health check and return the current status.
    async fn check(&self) -> HealthStatus;
}

/// Aggregates multiple `HealthCheck` implementors and reports the worst status.
pub struct HealthCheckAggregator {
    checks: Vec<Arc<dyn HealthCheck>>,
}

impl HealthCheckAggregator {
    /// Create an empty aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// Register a health check.
    pub fn register(&mut self, check: Arc<dyn HealthCheck>) {
        self.checks.push(check);
    }

    /// Run all checks and return a map of `name -> status`.
    pub async fn check_all(&self) -> HashMap<String, HealthStatus> {
        let mut results = HashMap::new();
        for check in &self.checks {
            let status = check.check().await;
            results.insert(check.name().to_string(), status);
        }
        results
    }

    /// Returns the worst status across all checks.
    pub async fn overall_status(&self) -> HealthStatus {
        let results = self.check_all().await;
        let mut worst = HealthStatus::Healthy;
        for status in results.values() {
            match (&worst, status) {
                (_, HealthStatus::Unhealthy(msg)) => {
                    worst = HealthStatus::Unhealthy(msg.clone());
                    break; // cannot get worse
                }
                (HealthStatus::Healthy, HealthStatus::Degraded(msg)) => {
                    worst = HealthStatus::Degraded(msg.clone());
                }
                _ => {}
            }
        }
        worst
    }
}

impl Default for HealthCheckAggregator {
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

    // --- CodecMetrics ---

    #[test]
    fn test_codec_metrics_new() {
        let m = CodecMetrics::new("h264");
        assert_eq!(m.codec_name, "h264");
        assert_eq!(m.fps, 0.0);
        assert_eq!(m.frames_encoded, 0);
    }

    #[test]
    fn test_codec_metrics_quality_ok() {
        let mut m = CodecMetrics::new("av1");
        m.quality_score = 85.0;
        assert!(m.quality_ok(80.0));
        assert!(!m.quality_ok(90.0));
    }

    #[test]
    fn test_codec_metrics_fps_ok() {
        let mut m = CodecMetrics::new("vp9");
        m.fps = 30.0;
        assert!(m.fps_ok(25.0));
        assert!(!m.fps_ok(60.0));
    }

    // --- SimpleMetricsCollector ---

    #[test]
    fn test_collector_poll_and_snapshot() {
        let collector = SimpleMetricsCollector::new();
        collector.poll(55.0, 4096.0, 120.0);
        let snap = collector.snapshot();
        assert!((snap.cpu_percent - 55.0).abs() < f64::EPSILON);
        assert!((snap.memory_mb - 4096.0).abs() < f64::EPSILON);
        assert!((snap.disk_io_mbps - 120.0).abs() < f64::EPSILON);
        assert_eq!(collector.poll_count(), 1);
    }

    #[test]
    fn test_collector_poll_codec() {
        let collector = SimpleMetricsCollector::new();
        let mut m = CodecMetrics::new("h265");
        m.fps = 60.0;
        m.quality_score = 90.0;
        collector.poll_codec(m);

        let snap = collector.snapshot();
        assert!(snap.codecs.contains_key("h265"));
        assert!((snap.codecs["h265"].fps - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_collector_poll_count_increments() {
        let collector = SimpleMetricsCollector::new();
        for _ in 0..5 {
            collector.poll(10.0, 1024.0, 10.0);
        }
        assert_eq!(collector.poll_count(), 5);
    }

    #[test]
    fn test_collector_snapshot_codecs_ok() {
        let collector = SimpleMetricsCollector::new();
        let mut m = CodecMetrics::new("h264");
        m.quality_score = 75.0;
        collector.poll_codec(m);

        let snap = collector.snapshot();
        assert!(snap.all_codecs_ok(70.0));
        assert!(!snap.all_codecs_ok(80.0));
    }

    // --- Comparison ---

    #[test]
    fn test_comparison_variants() {
        assert!(Comparison::GreaterThan.evaluate(10.0, 9.0));
        assert!(!Comparison::GreaterThan.evaluate(9.0, 10.0));
        assert!(Comparison::GreaterThanOrEqual.evaluate(10.0, 10.0));
        assert!(Comparison::LessThan.evaluate(5.0, 10.0));
        assert!(Comparison::LessThanOrEqual.evaluate(10.0, 10.0));
        assert!(Comparison::Equal.evaluate(5.0, 5.0));
    }

    // --- SimpleAlertRule ---

    #[test]
    fn test_alert_rule_fires() {
        let rule = SimpleAlertRule::new(
            "cpu_high",
            "cpu",
            Comparison::GreaterThan,
            90.0,
            NotificationAction::Log("CPU is high!".to_string()),
        );
        assert!(rule.evaluate(95.0));
        assert!(!rule.evaluate(80.0));
    }

    #[test]
    fn test_alert_rule_disabled() {
        let mut rule = SimpleAlertRule::new(
            "mem_low",
            "memory",
            Comparison::LessThan,
            100.0,
            NotificationAction::Log("Low memory".to_string()),
        );
        rule.enabled = false;
        // Should never fire when disabled.
        assert!(!rule.evaluate(50.0));
    }

    // --- SimpleAlertManager ---

    #[test]
    fn test_alert_manager_evaluate() {
        let mgr = SimpleAlertManager::new();
        mgr.add_rule(SimpleAlertRule::new(
            "cpu_high",
            "cpu",
            Comparison::GreaterThan,
            90.0,
            NotificationAction::Log("cpu alert".to_string()),
        ));

        assert_eq!(mgr.evaluate("cpu", 95.0), 1);
        assert_eq!(mgr.evaluate("cpu", 80.0), 0);
        assert_eq!(mgr.history().len(), 1);
    }

    #[test]
    fn test_alert_manager_evaluate_snapshot() {
        let mgr = SimpleAlertManager::new();
        mgr.add_rule(SimpleAlertRule::new(
            "cpu_high",
            "cpu",
            Comparison::GreaterThan,
            90.0,
            NotificationAction::RecordMetric("cpu.alert".to_string()),
        ));
        mgr.add_rule(SimpleAlertRule::new(
            "mem_high",
            "memory",
            Comparison::GreaterThan,
            8192.0,
            NotificationAction::RecordMetric("mem.alert".to_string()),
        ));

        let mut snapshot = HashMap::new();
        snapshot.insert("cpu".to_string(), 95.0);
        snapshot.insert("memory".to_string(), 10000.0);

        let fired = mgr.evaluate_snapshot(&snapshot);
        assert_eq!(fired, 2);
    }

    #[test]
    fn test_alert_manager_clear_history() {
        let mgr = SimpleAlertManager::new();
        mgr.add_rule(SimpleAlertRule::new(
            "r",
            "x",
            Comparison::GreaterThan,
            1.0,
            NotificationAction::Log("x".to_string()),
        ));
        let _ = mgr.evaluate("x", 5.0);
        assert!(!mgr.history().is_empty());
        mgr.clear_history();
        assert!(mgr.history().is_empty());
    }

    // --- HealthStatus ---

    #[test]
    fn test_health_status_predicates() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(!HealthStatus::Healthy.is_unhealthy());
        assert!(HealthStatus::Unhealthy("down".to_string()).is_unhealthy());
        assert!(!HealthStatus::Degraded("slow".to_string()).is_healthy());
    }

    #[test]
    fn test_health_status_as_str() {
        assert_eq!(HealthStatus::Healthy.as_str(), "healthy");
        assert_eq!(HealthStatus::Degraded("x".to_string()).as_str(), "degraded");
        assert_eq!(
            HealthStatus::Unhealthy("x".to_string()).as_str(),
            "unhealthy"
        );
    }

    // --- HealthCheck trait ---

    struct AlwaysHealthy;

    #[async_trait]
    impl HealthCheck for AlwaysHealthy {
        fn name(&self) -> &str {
            "always_healthy"
        }
        async fn check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }
    }

    struct AlwaysUnhealthy;

    #[async_trait]
    impl HealthCheck for AlwaysUnhealthy {
        fn name(&self) -> &str {
            "always_unhealthy"
        }
        async fn check(&self) -> HealthStatus {
            HealthStatus::Unhealthy("component is down".to_string())
        }
    }

    struct SometimesDegraded;

    #[async_trait]
    impl HealthCheck for SometimesDegraded {
        fn name(&self) -> &str {
            "sometimes_degraded"
        }
        async fn check(&self) -> HealthStatus {
            HealthStatus::Degraded("high latency".to_string())
        }
    }

    #[tokio::test]
    async fn test_health_check_trait_healthy() {
        let check = AlwaysHealthy;
        assert_eq!(check.check().await, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_health_check_trait_unhealthy() {
        let check = AlwaysUnhealthy;
        assert!(check.check().await.is_unhealthy());
    }

    #[tokio::test]
    async fn test_aggregator_all_healthy() {
        let mut agg = HealthCheckAggregator::new();
        agg.register(Arc::new(AlwaysHealthy));

        let results = agg.check_all().await;
        assert_eq!(results["always_healthy"], HealthStatus::Healthy);
        assert_eq!(agg.overall_status().await, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_aggregator_worst_status() {
        let mut agg = HealthCheckAggregator::new();
        agg.register(Arc::new(AlwaysHealthy));
        agg.register(Arc::new(AlwaysUnhealthy));

        let overall = agg.overall_status().await;
        assert!(overall.is_unhealthy());
    }

    #[tokio::test]
    async fn test_aggregator_degraded_beats_healthy() {
        let mut agg = HealthCheckAggregator::new();
        agg.register(Arc::new(AlwaysHealthy));
        agg.register(Arc::new(SometimesDegraded));

        let overall = agg.overall_status().await;
        assert_eq!(overall.as_str(), "degraded");
    }
}
