//! Health monitoring system for CDN providers.
//!
//! This module provides comprehensive health checking and monitoring for CDN providers,
//! including latency tracking, error rate monitoring, and availability scoring.

use super::{CdnProvider, Region};
use crate::error::{NetError, NetResult};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

/// Health status of a CDN provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Provider is healthy and operational.
    Healthy,
    /// Provider is degraded but operational.
    Degraded,
    /// Provider is unhealthy and should not receive traffic.
    Unhealthy,
    /// Provider health is unknown (not yet checked).
    Unknown,
}

impl HealthStatus {
    /// Returns true if the provider can receive traffic.
    #[must_use]
    pub const fn is_available(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }

    /// Returns the numeric score (0-100).
    #[must_use]
    pub const fn score(&self) -> u8 {
        match self {
            Self::Healthy => 100,
            Self::Degraded => 50,
            Self::Unhealthy => 0,
            Self::Unknown => 0,
        }
    }
}

/// Latency percentile metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// P50 latency (median).
    pub p50: Duration,
    /// P95 latency.
    pub p95: Duration,
    /// P99 latency.
    pub p99: Duration,
    /// Minimum latency.
    pub min: Duration,
    /// Maximum latency.
    pub max: Duration,
    /// Average latency.
    pub avg: Duration,
}

impl LatencyMetrics {
    /// Calculates metrics from latency samples.
    #[must_use]
    pub fn from_samples(samples: &[Duration]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let mut sorted = samples.to_vec();
        sorted.sort();

        let len = sorted.len();
        let p50_idx = (len as f64 * 0.5) as usize;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;

        let sum: Duration = sorted.iter().sum();
        let avg = sum / len as u32;

        Self {
            p50: sorted.get(p50_idx).copied().unwrap_or_default(),
            p95: sorted.get(p95_idx).copied().unwrap_or_default(),
            p99: sorted.get(p99_idx).copied().unwrap_or_default(),
            min: sorted.first().copied().unwrap_or_default(),
            max: sorted.last().copied().unwrap_or_default(),
            avg,
        }
    }
}

/// Geographic latency information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeoLatency {
    /// Latency by region.
    pub by_region: HashMap<Region, LatencyMetrics>,
}

impl GeoLatency {
    /// Adds a latency sample for a region.
    pub fn add_sample(&mut self, region: Region, latency: Duration, window_size: usize) {
        // In a real implementation, we'd track samples and recalculate
        // For now, we update the average using the entry API directly.
        let metrics = self
            .by_region
            .entry(region)
            .or_insert_with(LatencyMetrics::default);
        let new_avg = (metrics.avg + latency) / 2;
        metrics.avg = new_avg;
        metrics.min = metrics.min.min(latency);
        metrics.max = metrics.max.max(latency);

        // Simple percentile approximation
        if latency < metrics.p50 {
            metrics.p50 = (metrics.p50 + latency) / 2;
        }
        if latency > metrics.p95 {
            metrics.p95 = (metrics.p95 + latency) / 2;
        }
        if latency > metrics.p99 {
            metrics.p99 = (metrics.p99 + latency) / 2;
        }

        // Trim to window size (simplified)
        let _window = window_size;
    }

    /// Gets latency metrics for a region.
    #[must_use]
    pub fn get_region_metrics(&self, region: &Region) -> Option<&LatencyMetrics> {
        self.by_region.get(region)
    }
}

/// Health metrics for a CDN provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHealth {
    /// Provider ID.
    pub provider_id: String,
    /// Current health status.
    pub status: HealthStatus,
    /// Availability score (0-100).
    pub availability_score: f64,
    /// Error rate (0.0-1.0).
    pub error_rate: f64,
    /// Total requests.
    pub total_requests: u64,
    /// Failed requests.
    pub failed_requests: u64,
    /// Latency metrics.
    pub latency: LatencyMetrics,
    /// Geographic latency.
    pub geo_latency: GeoLatency,
    /// Last check time.
    pub last_check: DateTime<Utc>,
    /// Last status change.
    pub last_status_change: DateTime<Utc>,
    /// Consecutive failures.
    pub consecutive_failures: u32,
    /// Consecutive successes.
    pub consecutive_successes: u32,
}

impl ProviderHealth {
    /// Creates new health metrics for a provider.
    #[must_use]
    pub fn new(provider_id: String) -> Self {
        let now = Utc::now();
        Self {
            provider_id,
            status: HealthStatus::Unknown,
            availability_score: 0.0,
            error_rate: 0.0,
            total_requests: 0,
            failed_requests: 0,
            latency: LatencyMetrics::default(),
            geo_latency: GeoLatency::default(),
            last_check: now,
            last_status_change: now,
            consecutive_failures: 0,
            consecutive_successes: 0,
        }
    }

    /// Updates health status.
    pub fn update_status(&mut self, new_status: HealthStatus) {
        if self.status != new_status {
            self.last_status_change = Utc::now();
        }
        self.status = new_status;
        self.last_check = Utc::now();
    }

    /// Records a successful request.
    pub fn record_success(&mut self, latency: Duration) {
        self.total_requests += 1;
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;
        self.update_metrics();
        self.update_latency(latency);
    }

    /// Records a failed request.
    pub fn record_failure(&mut self) {
        self.total_requests += 1;
        self.failed_requests += 1;
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
        self.update_metrics();
    }

    /// Updates calculated metrics.
    fn update_metrics(&mut self) {
        if self.total_requests > 0 {
            self.error_rate = self.failed_requests as f64 / self.total_requests as f64;
            self.availability_score = (1.0 - self.error_rate) * 100.0;
        }

        // Update status based on metrics
        let new_status = if self.consecutive_failures >= 5 {
            HealthStatus::Unhealthy
        } else if self.error_rate > 0.1 || self.latency.p95 > Duration::from_secs(2) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        self.update_status(new_status);
    }

    /// Updates latency metrics.
    fn update_latency(&mut self, latency: Duration) {
        // Simple running average for demonstration
        let old_avg = self.latency.avg;
        let count = self.total_requests.saturating_sub(self.failed_requests);

        if count > 0 {
            let new_avg = (old_avg * (count - 1) as u32 + latency) / count as u32;
            self.latency.avg = new_avg;
        } else {
            self.latency.avg = latency;
        }

        self.latency.min = if self.latency.min == Duration::ZERO {
            latency
        } else {
            self.latency.min.min(latency)
        };
        self.latency.max = self.latency.max.max(latency);

        // Simple percentile approximation
        if count <= 1 {
            self.latency.p50 = latency;
            self.latency.p95 = latency;
            self.latency.p99 = latency;
        } else {
            // Update P50 (median approximation)
            if latency < self.latency.p50 {
                self.latency.p50 = (self.latency.p50 * 9 + latency) / 10;
            } else {
                self.latency.p50 = (self.latency.p50 * 9 + latency) / 10;
            }

            // Update P95
            if latency > self.latency.p95 {
                self.latency.p95 = (self.latency.p95 * 19 + latency) / 20;
            }

            // Update P99
            if latency > self.latency.p99 {
                self.latency.p99 = (self.latency.p99 * 99 + latency) / 100;
            }
        }
    }

    /// Checks if the provider is healthy.
    #[must_use]
    pub const fn is_healthy(&self) -> bool {
        matches!(self.status, HealthStatus::Healthy)
    }

    /// Gets the health score (0-100).
    #[must_use]
    pub fn health_score(&self) -> f64 {
        self.availability_score
    }
}

/// Latency sample with timestamp.
#[derive(Debug, Clone)]
struct LatencySample {
    /// Sample value.
    value: Duration,
    /// Sample timestamp.
    timestamp: DateTime<Utc>,
    /// Optional region.
    region: Option<Region>,
}

/// Health check configuration.
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Check interval.
    pub interval: Duration,
    /// Check timeout.
    pub timeout: Duration,
    /// Sample window size.
    pub sample_window_size: usize,
    /// Failure threshold.
    pub failure_threshold: u32,
    /// Recovery threshold.
    pub recovery_threshold: u32,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(5),
            timeout: Duration::from_secs(3),
            sample_window_size: 100,
            failure_threshold: 3,
            recovery_threshold: 5,
        }
    }
}

/// Internal state for health tracking.
struct HealthState {
    /// Provider health metrics.
    health: HashMap<String, ProviderHealth>,
    /// Latency samples.
    latency_samples: HashMap<String, VecDeque<LatencySample>>,
    /// Provider configurations.
    providers: HashMap<String, CdnProvider>,
}

/// Health checker for CDN providers.
pub struct HealthChecker {
    /// Configuration.
    config: HealthCheckConfig,
    /// Internal state.
    state: Arc<RwLock<HealthState>>,
}

impl HealthChecker {
    /// Creates a new health checker.
    #[must_use]
    pub fn new(interval: Duration, timeout: Duration) -> Self {
        let config = HealthCheckConfig {
            interval,
            timeout,
            ..Default::default()
        };

        let state = HealthState {
            health: HashMap::new(),
            latency_samples: HashMap::new(),
            providers: HashMap::new(),
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
        }
    }

    /// Creates a health checker with custom configuration.
    #[must_use]
    pub fn with_config(config: HealthCheckConfig) -> Self {
        let state = HealthState {
            health: HashMap::new(),
            latency_samples: HashMap::new(),
            providers: HashMap::new(),
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
        }
    }

    /// Adds a provider to monitor.
    pub fn add_provider(&self, provider_id: String, provider: CdnProvider) {
        let mut state = self.state.write();
        state.health.insert(
            provider_id.clone(),
            ProviderHealth::new(provider_id.clone()),
        );
        state
            .latency_samples
            .insert(provider_id.clone(), VecDeque::new());
        state.providers.insert(provider_id, provider);
    }

    /// Removes a provider from monitoring.
    pub fn remove_provider(&self, provider_id: &str) {
        let mut state = self.state.write();
        state.health.remove(provider_id);
        state.latency_samples.remove(provider_id);
        state.providers.remove(provider_id);
    }

    /// Gets health status for a provider.
    #[must_use]
    pub fn get_health(&self, provider_id: &str) -> Option<ProviderHealth> {
        self.state.read().health.get(provider_id).cloned()
    }

    /// Gets health status for all providers.
    #[must_use]
    pub fn get_all_health(&self) -> HashMap<String, ProviderHealth> {
        self.state.read().health.clone()
    }

    /// Records a latency measurement.
    pub fn record_latency(&self, provider_id: &str, latency: Duration) {
        let mut state = self.state.write();

        // Add sample to window
        if let Some(samples) = state.latency_samples.get_mut(provider_id) {
            samples.push_back(LatencySample {
                value: latency,
                timestamp: Utc::now(),
                region: None,
            });

            // Maintain window size
            while samples.len() > self.config.sample_window_size {
                samples.pop_front();
            }
        }

        // Update health metrics
        if let Some(health) = state.health.get_mut(provider_id) {
            health.record_success(latency);
        }
    }

    /// Records a latency measurement with region.
    pub fn record_latency_with_region(&self, provider_id: &str, latency: Duration, region: Region) {
        let mut state = self.state.write();

        // Add sample to window
        if let Some(samples) = state.latency_samples.get_mut(provider_id) {
            samples.push_back(LatencySample {
                value: latency,
                timestamp: Utc::now(),
                region: Some(region),
            });

            // Maintain window size
            while samples.len() > self.config.sample_window_size {
                samples.pop_front();
            }
        }

        // Update health metrics
        if let Some(health) = state.health.get_mut(provider_id) {
            health.record_success(latency);
            health
                .geo_latency
                .add_sample(region, latency, self.config.sample_window_size);
        }
    }

    /// Records a health check failure.
    pub fn record_failure(&self, provider_id: &str) {
        let mut state = self.state.write();
        if let Some(health) = state.health.get_mut(provider_id) {
            health.record_failure();
        }
    }

    /// Performs a health check for a provider.
    pub async fn check_provider(&self, provider_id: &str) -> NetResult<ProviderHealth> {
        let provider = {
            let state = self.state.read();
            state
                .providers
                .get(provider_id)
                .cloned()
                .ok_or_else(|| NetError::not_found("Provider not found"))?
        };

        // Perform HTTP health check
        let start = std::time::Instant::now();
        let result = self.perform_http_check(&provider).await;
        let latency = start.elapsed();

        match result {
            Ok(()) => {
                self.record_latency(provider_id, latency);
                self.get_health(provider_id)
                    .ok_or_else(|| NetError::invalid_state("Health not found"))
            }
            Err(_) => {
                self.record_failure(provider_id);
                self.get_health(provider_id)
                    .ok_or_else(|| NetError::invalid_state("Health not found"))
            }
        }
    }

    /// Performs HTTP health check.
    async fn perform_http_check(&self, provider: &CdnProvider) -> NetResult<()> {
        let client = reqwest::Client::builder()
            .timeout(self.config.timeout)
            .build()
            .map_err(|e| NetError::connection(e.to_string()))?;

        // Use a simple HEAD request for health check
        let health_url = provider.build_url("/health");

        let response = client
            .head(&health_url)
            .send()
            .await
            .map_err(|e| NetError::connection(e.to_string()))?;

        if response.status().is_success() || response.status().as_u16() == 404 {
            // 404 is acceptable - endpoint exists but no health endpoint
            Ok(())
        } else {
            Err(NetError::http(
                response.status().as_u16(),
                "Health check failed",
            ))
        }
    }

    /// Calculates latency percentiles for a provider.
    #[must_use]
    pub fn calculate_latency_percentiles(&self, provider_id: &str) -> Option<LatencyMetrics> {
        let state = self.state.read();
        let samples = state.latency_samples.get(provider_id)?;

        if samples.is_empty() {
            return None;
        }

        let values: Vec<Duration> = samples.iter().map(|s| s.value).collect();
        Some(LatencyMetrics::from_samples(&values))
    }

    /// Gets geographic latency for a provider.
    #[must_use]
    pub fn get_geo_latency(&self, provider_id: &str) -> Option<GeoLatency> {
        self.state
            .read()
            .health
            .get(provider_id)
            .map(|h| h.geo_latency.clone())
    }

    /// Gets the best provider for a region based on latency.
    #[must_use]
    pub fn get_best_for_region(&self, region: Region, provider_ids: &[String]) -> Option<String> {
        let state = self.state.read();

        provider_ids
            .iter()
            .filter_map(|id| {
                state.health.get(id).and_then(|health| {
                    health
                        .geo_latency
                        .get_region_metrics(&region)
                        .map(|metrics| (id.clone(), metrics.avg))
                })
            })
            .min_by_key(|(_, latency)| *latency)
            .map(|(id, _)| id)
    }

    /// Resets health metrics for a provider.
    pub fn reset_health(&self, provider_id: &str) {
        let mut state = self.state.write();
        if state.health.contains_key(provider_id) {
            state.health.insert(
                provider_id.to_string(),
                ProviderHealth::new(provider_id.to_string()),
            );
        }
    }

    /// Gets availability score for a provider.
    #[must_use]
    pub fn get_availability_score(&self, provider_id: &str) -> Option<f64> {
        self.state
            .read()
            .health
            .get(provider_id)
            .map(|h| h.availability_score)
    }

    /// Gets error rate for a provider.
    #[must_use]
    pub fn get_error_rate(&self, provider_id: &str) -> Option<f64> {
        self.state
            .read()
            .health
            .get(provider_id)
            .map(|h| h.error_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.is_available());
        assert!(HealthStatus::Degraded.is_available());
        assert!(!HealthStatus::Unhealthy.is_available());
        assert_eq!(HealthStatus::Healthy.score(), 100);
        assert_eq!(HealthStatus::Degraded.score(), 50);
        assert_eq!(HealthStatus::Unhealthy.score(), 0);
    }

    #[test]
    fn test_latency_metrics() {
        let samples = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(100),
        ];

        let metrics = LatencyMetrics::from_samples(&samples);
        assert_eq!(metrics.min, Duration::from_millis(10));
        assert_eq!(metrics.max, Duration::from_millis(100));
        assert_eq!(metrics.p50, Duration::from_millis(30));
    }

    #[test]
    fn test_provider_health_creation() {
        let health = ProviderHealth::new("provider-1".to_string());
        assert_eq!(health.provider_id, "provider-1");
        assert_eq!(health.status, HealthStatus::Unknown);
        assert_eq!(health.total_requests, 0);
        assert_eq!(health.failed_requests, 0);
    }

    #[test]
    fn test_provider_health_record_success() {
        let mut health = ProviderHealth::new("provider-1".to_string());
        health.record_success(Duration::from_millis(50));

        assert_eq!(health.total_requests, 1);
        assert_eq!(health.failed_requests, 0);
        assert_eq!(health.consecutive_successes, 1);
        assert_eq!(health.consecutive_failures, 0);
    }

    #[test]
    fn test_provider_health_record_failure() {
        let mut health = ProviderHealth::new("provider-1".to_string());
        health.record_failure();

        assert_eq!(health.total_requests, 1);
        assert_eq!(health.failed_requests, 1);
        assert_eq!(health.consecutive_failures, 1);
        assert_eq!(health.consecutive_successes, 0);
    }

    #[test]
    fn test_health_checker_creation() {
        let checker = HealthChecker::new(Duration::from_secs(5), Duration::from_secs(3));
        assert_eq!(checker.config.interval, Duration::from_secs(5));
        assert_eq!(checker.config.timeout, Duration::from_secs(3));
    }

    #[test]
    fn test_health_checker_add_provider() {
        let checker = HealthChecker::new(Duration::from_secs(5), Duration::from_secs(3));
        let provider = CdnProvider::cloudflare("https://cdn.example.com", 100);
        let id = provider.id.clone();

        checker.add_provider(id.clone(), provider);
        assert!(checker.get_health(&id).is_some());
    }

    #[test]
    fn test_health_checker_record_latency() {
        let checker = HealthChecker::new(Duration::from_secs(5), Duration::from_secs(3));
        let provider = CdnProvider::cloudflare("https://cdn.example.com", 100);
        let id = provider.id.clone();

        checker.add_provider(id.clone(), provider);
        checker.record_latency(&id, Duration::from_millis(50));

        let health = checker.get_health(&id).expect("Health exists");
        assert!(health.latency.avg > Duration::ZERO);
    }

    #[test]
    fn test_geo_latency() {
        let mut geo = GeoLatency::default();
        geo.add_sample(Region::Europe, Duration::from_millis(50), 100);
        geo.add_sample(Region::AsiaPacific, Duration::from_millis(100), 100);

        assert!(geo.get_region_metrics(&Region::Europe).is_some());
        assert!(geo.get_region_metrics(&Region::AsiaPacific).is_some());
    }
}
