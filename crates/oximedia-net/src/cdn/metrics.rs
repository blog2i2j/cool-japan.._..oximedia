//! Performance metrics and monitoring for CDN providers.
//!
//! This module provides comprehensive metrics collection including request/response
//! logging, bandwidth tracking, cache hit rates, error categorization, and SLA monitoring.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// Error category for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Connection errors.
    Connection,
    /// Timeout errors.
    Timeout,
    /// HTTP 4xx errors.
    ClientError,
    /// HTTP 5xx errors.
    ServerError,
    /// DNS resolution errors.
    Dns,
    /// SSL/TLS errors.
    Tls,
    /// Rate limiting errors.
    RateLimit,
    /// Other errors.
    Other,
}

impl ErrorCategory {
    /// Returns the category name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Connection => "Connection",
            Self::Timeout => "Timeout",
            Self::ClientError => "Client Error",
            Self::ServerError => "Server Error",
            Self::Dns => "DNS",
            Self::Tls => "TLS",
            Self::RateLimit => "Rate Limit",
            Self::Other => "Other",
        }
    }

    /// Categorizes an HTTP status code.
    #[must_use]
    pub const fn from_status_code(status: u16) -> Self {
        match status {
            400..=499 => {
                if status == 429 {
                    Self::RateLimit
                } else {
                    Self::ClientError
                }
            }
            500..=599 => Self::ServerError,
            _ => Self::Other,
        }
    }
}

/// Request log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLog {
    /// Request timestamp.
    pub timestamp: DateTime<Utc>,
    /// Provider ID.
    pub provider_id: String,
    /// Request path.
    pub path: String,
    /// Response status code.
    pub status_code: Option<u16>,
    /// Request latency.
    pub latency: Duration,
    /// Bytes transferred.
    pub bytes: u64,
    /// Whether request succeeded.
    pub success: bool,
    /// Error category if failed.
    pub error_category: Option<ErrorCategory>,
}

/// Cache hit rate statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total requests.
    pub total_requests: u64,
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
    /// Hit rate (0.0-1.0).
    pub hit_rate: f64,
}

impl CacheStats {
    /// Records a cache hit.
    pub fn record_hit(&mut self) {
        self.total_requests += 1;
        self.hits += 1;
        self.update_rate();
    }

    /// Records a cache miss.
    pub fn record_miss(&mut self) {
        self.total_requests += 1;
        self.misses += 1;
        self.update_rate();
    }

    /// Updates the hit rate.
    #[allow(clippy::manual_checked_ops)]
    fn update_rate(&mut self) {
        if self.total_requests > 0 {
            self.hit_rate = self.hits as f64 / self.total_requests as f64;
        }
    }
}

/// Bandwidth statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BandwidthStats {
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// Bytes in last second.
    pub bytes_per_second: u64,
    /// Peak bandwidth (bytes/sec).
    pub peak_bandwidth: u64,
    /// Average bandwidth (bytes/sec).
    pub avg_bandwidth: u64,
    /// Sample count for averaging.
    sample_count: u64,
}

impl BandwidthStats {
    /// Records bytes transferred.
    pub fn record_bytes(&mut self, bytes: u64) {
        self.total_bytes += bytes;
        self.bytes_per_second += bytes;
        self.sample_count += 1;

        // Update peak
        if self.bytes_per_second > self.peak_bandwidth {
            self.peak_bandwidth = self.bytes_per_second;
        }

        // Update average
        self.avg_bandwidth = self.total_bytes.checked_div(self.sample_count).unwrap_or(0);
    }

    /// Resets the per-second counter.
    pub fn reset_second(&mut self) {
        self.bytes_per_second = 0;
    }

    /// Gets bandwidth in Mbps.
    #[must_use]
    pub fn mbps(&self) -> f64 {
        (self.bytes_per_second as f64 * 8.0) / 1_000_000.0
    }
}

/// SLA (Service Level Agreement) metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaMetrics {
    /// Target availability (0.0-1.0, e.g., 0.999 for 99.9%).
    pub target_availability: f64,
    /// Actual availability.
    pub actual_availability: f64,
    /// Target latency (P95).
    pub target_latency_p95: Duration,
    /// Actual latency (P95).
    pub actual_latency_p95: Duration,
    /// Target error rate (0.0-1.0).
    pub target_error_rate: f64,
    /// Actual error rate.
    pub actual_error_rate: f64,
    /// SLA compliance (true if meeting all targets).
    pub compliant: bool,
}

impl Default for SlaMetrics {
    fn default() -> Self {
        Self {
            target_availability: 0.999,
            actual_availability: 1.0,
            target_latency_p95: Duration::from_millis(500),
            actual_latency_p95: Duration::ZERO,
            target_error_rate: 0.01,
            actual_error_rate: 0.0,
            compliant: true,
        }
    }
}

impl SlaMetrics {
    /// Updates SLA compliance status.
    pub fn update_compliance(&mut self) {
        self.compliant = self.actual_availability >= self.target_availability
            && self.actual_latency_p95 <= self.target_latency_p95
            && self.actual_error_rate <= self.target_error_rate;
    }

    /// Gets SLA breach severity (0.0-1.0, higher is worse).
    #[must_use]
    pub fn breach_severity(&self) -> f64 {
        if self.compliant {
            return 0.0;
        }

        let mut severity = 0.0;

        // Availability breach
        if self.actual_availability < self.target_availability {
            let diff = self.target_availability - self.actual_availability;
            severity += diff / self.target_availability;
        }

        // Latency breach
        if self.actual_latency_p95 > self.target_latency_p95 {
            let diff = self
                .actual_latency_p95
                .saturating_sub(self.target_latency_p95);
            let ratio = diff.as_secs_f64() / self.target_latency_p95.as_secs_f64();
            severity += ratio;
        }

        // Error rate breach
        if self.actual_error_rate > self.target_error_rate {
            let diff = self.actual_error_rate - self.target_error_rate;
            severity += diff / self.target_error_rate;
        }

        severity.min(1.0)
    }
}

/// Performance metrics for a CDN provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Provider ID.
    pub provider_id: String,
    /// Total requests.
    pub total_requests: u64,
    /// Successful requests.
    pub successful_requests: u64,
    /// Failed requests.
    pub failed_requests: u64,
    /// Error rate (0.0-1.0).
    pub error_rate: f64,
    /// Average latency.
    pub avg_latency: Duration,
    /// P50 latency.
    pub p50_latency: Duration,
    /// P95 latency.
    pub p95_latency: Duration,
    /// P99 latency.
    pub p99_latency: Duration,
    /// Bandwidth statistics.
    pub bandwidth: BandwidthStats,
    /// Cache statistics.
    pub cache: CacheStats,
    /// Error breakdown by category.
    pub errors_by_category: HashMap<ErrorCategory, u64>,
    /// SLA metrics.
    pub sla: SlaMetrics,
    /// First request timestamp.
    pub first_request_at: Option<DateTime<Utc>>,
    /// Last request timestamp.
    pub last_request_at: Option<DateTime<Utc>>,
}

impl PerformanceMetrics {
    /// Creates new performance metrics for a provider.
    #[must_use]
    pub fn new(provider_id: String) -> Self {
        Self {
            provider_id,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            error_rate: 0.0,
            avg_latency: Duration::ZERO,
            p50_latency: Duration::ZERO,
            p95_latency: Duration::ZERO,
            p99_latency: Duration::ZERO,
            bandwidth: BandwidthStats::default(),
            cache: CacheStats::default(),
            errors_by_category: HashMap::new(),
            sla: SlaMetrics::default(),
            first_request_at: None,
            last_request_at: None,
        }
    }

    /// Updates error rate.
    #[allow(clippy::manual_checked_ops)]
    fn update_error_rate(&mut self) {
        if self.total_requests > 0 {
            self.error_rate = self.failed_requests as f64 / self.total_requests as f64;
            self.sla.actual_error_rate = self.error_rate;
            self.sla.actual_availability = 1.0 - self.error_rate;
            self.sla.update_compliance();
        }
    }

    /// Records an error by category.
    pub fn record_error(&mut self, category: ErrorCategory) {
        *self.errors_by_category.entry(category).or_insert(0) += 1;
    }

    /// Gets success rate (0.0-1.0).
    #[must_use]
    #[allow(clippy::manual_checked_ops)]
    pub fn success_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.successful_requests as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }

    /// Gets requests per second.
    #[must_use]
    #[allow(clippy::manual_checked_ops)]
    pub fn requests_per_second(&self) -> f64 {
        if let (Some(first), Some(last)) = (self.first_request_at, self.last_request_at) {
            let duration = last.signed_duration_since(first);
            let seconds = duration.num_seconds() as f64;
            if seconds > 0.0 {
                return self.total_requests as f64 / seconds;
            }
        }
        0.0
    }
}

/// CDN metrics aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnMetrics {
    /// Metrics per provider.
    pub providers: HashMap<String, PerformanceMetrics>,
    /// Global statistics.
    pub global_total_requests: u64,
    /// Global total bytes.
    pub global_total_bytes: u64,
    /// Collection start time.
    pub collection_start: DateTime<Utc>,
}

impl CdnMetrics {
    /// Creates new CDN metrics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            global_total_requests: 0,
            global_total_bytes: 0,
            collection_start: Utc::now(),
        }
    }

    /// Gets metrics for a provider.
    #[must_use]
    pub fn get_provider(&self, provider_id: &str) -> Option<&PerformanceMetrics> {
        self.providers.get(provider_id)
    }

    /// Gets global success rate.
    #[must_use]
    #[allow(clippy::manual_checked_ops)]
    pub fn global_success_rate(&self) -> f64 {
        let total_success: u64 = self.providers.values().map(|m| m.successful_requests).sum();

        if self.global_total_requests > 0 {
            total_success as f64 / self.global_total_requests as f64
        } else {
            0.0
        }
    }
}

impl Default for CdnMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics collector state.
struct MetricsState {
    /// Performance metrics per provider.
    metrics: HashMap<String, PerformanceMetrics>,
    /// Request logs (ring buffer).
    request_logs: VecDeque<RequestLog>,
    /// Latency samples per provider.
    latency_samples: HashMap<String, VecDeque<Duration>>,
}

/// Metrics collector for CDN providers.
pub struct MetricsCollector {
    /// Collection interval.
    interval: Duration,
    /// Internal state.
    state: Arc<RwLock<MetricsState>>,
    /// Maximum log entries.
    max_log_entries: usize,
    /// Maximum latency samples.
    max_latency_samples: usize,
}

impl MetricsCollector {
    /// Creates a new metrics collector.
    #[must_use]
    pub fn new(interval: Duration) -> Self {
        let state = MetricsState {
            metrics: HashMap::new(),
            request_logs: VecDeque::new(),
            latency_samples: HashMap::new(),
        };

        Self {
            interval,
            state: Arc::new(RwLock::new(state)),
            max_log_entries: 10000,
            max_latency_samples: 1000,
        }
    }

    /// Records a successful request.
    pub fn record_success(&self, provider_id: &str, latency: Duration, bytes: u64) {
        let mut state = self.state.write();

        let metrics = state
            .metrics
            .entry(provider_id.to_string())
            .or_insert_with(|| PerformanceMetrics::new(provider_id.to_string()));

        metrics.total_requests += 1;
        metrics.successful_requests += 1;
        metrics.bandwidth.record_bytes(bytes);

        let now = Utc::now();
        if metrics.first_request_at.is_none() {
            metrics.first_request_at = Some(now);
        }
        metrics.last_request_at = Some(now);

        metrics.update_error_rate();

        // Get latency samples (need to borrow separately)
        let provider_id_string = provider_id.to_string();

        // Add latency sample
        {
            let samples = state
                .latency_samples
                .entry(provider_id_string.clone())
                .or_insert_with(VecDeque::new);
            samples.push_back(latency);
            while samples.len() > self.max_latency_samples {
                samples.pop_front();
            }
        }

        // Update latency metrics (compute values first to avoid borrow issue)
        let (avg, p50, p95, p99) = {
            if let Some(samples) = state.latency_samples.get(&provider_id_string) {
                if samples.is_empty() {
                    (
                        Duration::ZERO,
                        Duration::ZERO,
                        Duration::ZERO,
                        Duration::ZERO,
                    )
                } else {
                    let mut sorted: Vec<_> = samples.iter().copied().collect();
                    sorted.sort();
                    let sum: Duration = sorted.iter().sum();
                    let avg = sum / sorted.len() as u32;
                    let p50 = sorted[sorted.len() / 2];
                    let p95 = sorted[sorted.len() * 95 / 100];
                    let p99 = sorted[sorted.len() * 99 / 100];
                    (avg, p50, p95, p99)
                }
            } else {
                (
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                )
            }
        };

        // Now update metrics with computed values
        if let Some(metrics) = state.metrics.get_mut(&provider_id_string) {
            metrics.avg_latency = avg;
            metrics.p50_latency = p50;
            metrics.p95_latency = p95;
            metrics.p99_latency = p99;
        }

        // Add request log
        state.request_logs.push_back(RequestLog {
            timestamp: now,
            provider_id: provider_id.to_string(),
            path: String::new(),
            status_code: Some(200),
            latency,
            bytes,
            success: true,
            error_category: None,
        });

        // Trim logs
        while state.request_logs.len() > self.max_log_entries {
            state.request_logs.pop_front();
        }
    }

    /// Records a failed request.
    pub fn record_failure(&self, provider_id: &str) {
        let mut state = self.state.write();

        let metrics = state
            .metrics
            .entry(provider_id.to_string())
            .or_insert_with(|| PerformanceMetrics::new(provider_id.to_string()));

        metrics.total_requests += 1;
        metrics.failed_requests += 1;

        let now = Utc::now();
        if metrics.first_request_at.is_none() {
            metrics.first_request_at = Some(now);
        }
        metrics.last_request_at = Some(now);

        metrics.update_error_rate();

        // Add request log
        state.request_logs.push_back(RequestLog {
            timestamp: now,
            provider_id: provider_id.to_string(),
            path: String::new(),
            status_code: None,
            latency: Duration::ZERO,
            bytes: 0,
            success: false,
            error_category: Some(ErrorCategory::Other),
        });

        // Trim logs
        while state.request_logs.len() > self.max_log_entries {
            state.request_logs.pop_front();
        }
    }

    /// Records a request.
    pub fn record_request(&self, provider_id: &str) {
        let mut state = self.state.write();

        let metrics = state
            .metrics
            .entry(provider_id.to_string())
            .or_insert_with(|| PerformanceMetrics::new(provider_id.to_string()));

        metrics.total_requests += 1;
    }

    /// Updates latency metrics from samples.
    fn update_latency_metrics(
        &self,
        metrics: &mut PerformanceMetrics,
        samples: &VecDeque<Duration>,
    ) {
        if samples.is_empty() {
            return;
        }

        let mut sorted: Vec<_> = samples.iter().copied().collect();
        sorted.sort();

        let len = sorted.len();
        let sum: Duration = sorted.iter().sum();

        metrics.avg_latency = sum / len as u32;
        metrics.p50_latency = sorted[len / 2];
        metrics.p95_latency = sorted[(len as f64 * 0.95) as usize];
        metrics.p99_latency = sorted[(len as f64 * 0.99) as usize];

        metrics.sla.actual_latency_p95 = metrics.p95_latency;
        metrics.sla.update_compliance();
    }

    /// Gets metrics for a provider.
    #[must_use]
    pub fn get_metrics(&self, provider_id: &str) -> Option<PerformanceMetrics> {
        self.state.read().metrics.get(provider_id).cloned()
    }

    /// Gets all metrics.
    #[must_use]
    pub fn get_all_metrics(&self) -> HashMap<String, PerformanceMetrics> {
        self.state.read().metrics.clone()
    }

    /// Gets recent request logs.
    #[must_use]
    pub fn get_recent_logs(&self, limit: usize) -> Vec<RequestLog> {
        let state = self.state.read();
        state
            .request_logs
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Exports metrics in Prometheus format.
    #[must_use]
    pub fn export_prometheus(&self) -> String {
        let state = self.state.read();
        let mut output = String::new();

        // Total requests metric
        output.push_str("# HELP cdn_requests_total Total number of requests per provider\n");
        output.push_str("# TYPE cdn_requests_total counter\n");
        for (provider_id, metrics) in &state.metrics {
            output.push_str(&format!(
                "cdn_requests_total{{provider=\"{}\"}} {}\n",
                provider_id, metrics.total_requests
            ));
        }

        // Successful requests metric
        output.push_str("# HELP cdn_requests_success Successful requests per provider\n");
        output.push_str("# TYPE cdn_requests_success counter\n");
        for (provider_id, metrics) in &state.metrics {
            output.push_str(&format!(
                "cdn_requests_success{{provider=\"{}\"}} {}\n",
                provider_id, metrics.successful_requests
            ));
        }

        // Failed requests metric
        output.push_str("# HELP cdn_requests_failed Failed requests per provider\n");
        output.push_str("# TYPE cdn_requests_failed counter\n");
        for (provider_id, metrics) in &state.metrics {
            output.push_str(&format!(
                "cdn_requests_failed{{provider=\"{}\"}} {}\n",
                provider_id, metrics.failed_requests
            ));
        }

        // Error rate metric
        output.push_str("# HELP cdn_error_rate Error rate per provider\n");
        output.push_str("# TYPE cdn_error_rate gauge\n");
        for (provider_id, metrics) in &state.metrics {
            output.push_str(&format!(
                "cdn_error_rate{{provider=\"{}\"}} {:.6}\n",
                provider_id, metrics.error_rate
            ));
        }

        // Latency metrics
        output.push_str("# HELP cdn_latency_seconds Request latency in seconds\n");
        output.push_str("# TYPE cdn_latency_seconds gauge\n");
        for (provider_id, metrics) in &state.metrics {
            output.push_str(&format!(
                "cdn_latency_seconds{{provider=\"{}\",quantile=\"0.5\"}} {:.6}\n",
                provider_id,
                metrics.p50_latency.as_secs_f64()
            ));
            output.push_str(&format!(
                "cdn_latency_seconds{{provider=\"{}\",quantile=\"0.95\"}} {:.6}\n",
                provider_id,
                metrics.p95_latency.as_secs_f64()
            ));
            output.push_str(&format!(
                "cdn_latency_seconds{{provider=\"{}\",quantile=\"0.99\"}} {:.6}\n",
                provider_id,
                metrics.p99_latency.as_secs_f64()
            ));
        }

        // Bandwidth metrics
        output.push_str("# HELP cdn_bytes_total Total bytes transferred\n");
        output.push_str("# TYPE cdn_bytes_total counter\n");
        for (provider_id, metrics) in &state.metrics {
            output.push_str(&format!(
                "cdn_bytes_total{{provider=\"{}\"}} {}\n",
                provider_id, metrics.bandwidth.total_bytes
            ));
        }

        // Cache hit rate
        output.push_str("# HELP cdn_cache_hit_rate Cache hit rate\n");
        output.push_str("# TYPE cdn_cache_hit_rate gauge\n");
        for (provider_id, metrics) in &state.metrics {
            output.push_str(&format!(
                "cdn_cache_hit_rate{{provider=\"{}\"}} {:.6}\n",
                provider_id, metrics.cache.hit_rate
            ));
        }

        // SLA compliance
        output.push_str("# HELP cdn_sla_compliant SLA compliance status\n");
        output.push_str("# TYPE cdn_sla_compliant gauge\n");
        for (provider_id, metrics) in &state.metrics {
            output.push_str(&format!(
                "cdn_sla_compliant{{provider=\"{}\"}} {}\n",
                provider_id,
                if metrics.sla.compliant { 1 } else { 0 }
            ));
        }

        output
    }

    /// Resets all metrics.
    pub fn reset(&self) {
        let mut state = self.state.write();
        state.metrics.clear();
        state.request_logs.clear();
        state.latency_samples.clear();
    }
}

impl fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Provider: {}\nTotal Requests: {}\nSuccess Rate: {:.2}%\nError Rate: {:.2}%\nP95 Latency: {:?}\n",
            self.provider_id,
            self.total_requests,
            self.success_rate() * 100.0,
            self.error_rate * 100.0,
            self.p95_latency
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_category() {
        assert_eq!(
            ErrorCategory::from_status_code(404),
            ErrorCategory::ClientError
        );
        assert_eq!(
            ErrorCategory::from_status_code(429),
            ErrorCategory::RateLimit
        );
        assert_eq!(
            ErrorCategory::from_status_code(500),
            ErrorCategory::ServerError
        );
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::default();
        stats.record_hit();
        stats.record_hit();
        stats.record_miss();

        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_bandwidth_stats() {
        let mut stats = BandwidthStats::default();
        stats.record_bytes(1000);
        stats.record_bytes(2000);

        assert_eq!(stats.total_bytes, 3000);
        assert_eq!(stats.bytes_per_second, 3000);
    }

    #[test]
    fn test_sla_metrics() {
        let mut sla = SlaMetrics::default();
        sla.actual_availability = 0.9995;
        sla.actual_latency_p95 = Duration::from_millis(400);
        sla.actual_error_rate = 0.005;
        sla.update_compliance();

        assert!(sla.compliant);
        assert_eq!(sla.breach_severity(), 0.0);
    }

    #[test]
    fn test_sla_breach() {
        let mut sla = SlaMetrics::default();
        sla.actual_availability = 0.99; // Below 99.9%
        sla.update_compliance();

        assert!(!sla.compliant);
        assert!(sla.breach_severity() > 0.0);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new("provider-1".to_string());
        assert_eq!(metrics.provider_id, "provider-1");
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.success_rate(), 0.0);

        metrics.total_requests = 10;
        metrics.successful_requests = 9;
        metrics.failed_requests = 1;
        metrics.update_error_rate();

        assert_eq!(metrics.success_rate(), 0.9);
        assert_eq!(metrics.error_rate, 0.1);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new(Duration::from_secs(10));

        collector.record_success("provider-1", Duration::from_millis(50), 1000);
        collector.record_success("provider-1", Duration::from_millis(60), 2000);
        collector.record_failure("provider-1");

        let metrics = collector.get_metrics("provider-1").expect("Metrics exist");
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.successful_requests, 2);
        assert_eq!(metrics.failed_requests, 1);
    }

    #[test]
    fn test_prometheus_export() {
        let collector = MetricsCollector::new(Duration::from_secs(10));
        collector.record_success("provider-1", Duration::from_millis(50), 1000);

        let prometheus = collector.export_prometheus();
        assert!(prometheus.contains("cdn_requests_total"));
        assert!(prometheus.contains("provider-1"));
    }

    #[test]
    fn test_request_logs() {
        let collector = MetricsCollector::new(Duration::from_secs(10));
        collector.record_success("provider-1", Duration::from_millis(50), 1000);

        let logs = collector.get_recent_logs(10);
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].provider_id, "provider-1");
        assert!(logs[0].success);
    }
}
