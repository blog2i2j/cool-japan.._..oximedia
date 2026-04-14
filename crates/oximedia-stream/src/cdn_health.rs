//! CDN health check with configurable probe interval and failure threshold.
//!
//! `CdnHealthChecker` tracks the health of individual CDN providers by
//! recording probe outcomes and computing a rolling failure rate.  When a
//! provider's failure rate exceeds the configured threshold it is marked as
//! unhealthy and excluded from routing until it recovers.
//!
//! # Design
//!
//! - Probes are recorded as `Ok` / `Err` outcomes with a latency measurement
//!   (in milliseconds).
//! - A sliding window of the last `window_size` probes is maintained per
//!   provider.
//! - A provider is *unhealthy* when its failure ratio exceeds
//!   `failure_threshold` (a value in `[0.0, 1.0]`).
//! - Recovery requires the failure ratio to drop *below* `recovery_threshold`
//!   (defaults to `failure_threshold / 2`).
//! - [`CdnHealthRegistry`] aggregates multiple providers and exposes methods to
//!   query which providers are currently healthy.
//!
//! # Example
//!
//! ```rust
//! use oximedia_stream::cdn_health::{CdnHealthRegistry, HealthCheckConfig, ProbeOutcome};
//!
//! let config = HealthCheckConfig::default();
//! let mut registry = CdnHealthRegistry::new(config);
//!
//! registry.register("cdn-a");
//! registry.register("cdn-b");
//!
//! // Simulate probes
//! registry.record_probe("cdn-a", ProbeOutcome::Success { latency_ms: 25 });
//! registry.record_probe("cdn-b", ProbeOutcome::Failure { reason: "timeout".into() });
//!
//! let healthy: Vec<&str> = registry.healthy_providers().collect();
//! assert!(healthy.contains(&"cdn-a"));
//! ```

use std::collections::{HashMap, VecDeque};

// ─── ProbeOutcome ─────────────────────────────────────────────────────────────

/// The outcome of a single CDN health probe.
#[derive(Debug, Clone, PartialEq)]
pub enum ProbeOutcome {
    /// The probe succeeded within the allowed latency budget.
    Success {
        /// Round-trip latency measured during the probe, in milliseconds.
        latency_ms: u64,
    },
    /// The probe failed (timeout, HTTP error, connection refused, etc.).
    Failure {
        /// A human-readable reason string for diagnostics.
        reason: String,
    },
}

impl ProbeOutcome {
    /// Returns `true` if this is a [`ProbeOutcome::Success`].
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Returns the latency in milliseconds for a successful probe, or `None` for
    /// a failure.
    #[must_use]
    pub fn latency_ms(&self) -> Option<u64> {
        match self {
            Self::Success { latency_ms } => Some(*latency_ms),
            Self::Failure { .. } => None,
        }
    }
}

// ─── HealthCheckConfig ────────────────────────────────────────────────────────

/// Configuration for the CDN health-check subsystem.
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Number of probe outcomes retained in the sliding window per provider.
    ///
    /// Older probes are evicted when the window is full (FIFO).
    /// Clamped to `[1, 1000]` at construction time.
    pub window_size: usize,

    /// Fraction of failures in the window above which a provider is marked
    /// unhealthy.  In the range `[0.0, 1.0]`.  Default: `0.5`.
    pub failure_threshold: f64,

    /// Fraction of failures below which an unhealthy provider is considered
    /// recovered.  Default: `failure_threshold / 2`.
    ///
    /// `None` means use `failure_threshold / 2` automatically.
    pub recovery_threshold: Option<f64>,

    /// Maximum acceptable probe latency in milliseconds.  Probes that succeed
    /// but exceed this value are counted as *latency failures* for health
    /// computation.  `0` disables latency-based failure (only connectivity
    /// failures are counted).
    pub max_latency_ms: u64,

    /// Minimum number of probes in the window before health decisions are made.
    ///
    /// When fewer than `min_probes` outcomes have been recorded the provider is
    /// considered healthy (benefit of the doubt).
    pub min_probes: usize,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            window_size: 20,
            failure_threshold: 0.5,
            recovery_threshold: None,
            max_latency_ms: 0, // disabled by default
            min_probes: 3,
        }
    }
}

impl HealthCheckConfig {
    /// Effective recovery threshold — either the explicit value or
    /// `failure_threshold / 2`.
    #[must_use]
    pub fn effective_recovery_threshold(&self) -> f64 {
        self.recovery_threshold
            .unwrap_or_else(|| self.failure_threshold / 2.0)
            .clamp(0.0, 1.0)
    }
}

// ─── ProviderHealth ───────────────────────────────────────────────────────────

/// Internal per-provider state maintained by [`CdnHealthRegistry`].
#[derive(Debug)]
struct ProviderHealth {
    /// Provider name (for diagnostics only; identity is keyed by map key).
    name: String,
    /// Ring buffer of recent probe outcomes, oldest at front.
    window: VecDeque<bool>, // `true` = success, `false` = failure
    /// Sliding-window latency samples for successful probes (ms).
    latencies: VecDeque<u64>,
    /// Whether the provider is currently considered healthy.
    is_healthy: bool,
    /// Total successful probes ever recorded (not just in window).
    total_successes: u64,
    /// Total failed probes ever recorded (not just in window).
    total_failures: u64,
}

impl ProviderHealth {
    fn new(name: impl Into<String>, window_size: usize) -> Self {
        let cap = window_size.clamp(1, 1000);
        Self {
            name: name.into(),
            window: VecDeque::with_capacity(cap),
            latencies: VecDeque::with_capacity(cap),
            is_healthy: true,
            total_successes: 0,
            total_failures: 0,
        }
    }

    /// Record one probe outcome, evicting the oldest if the window is full.
    fn record(&mut self, outcome: &ProbeOutcome, config: &HealthCheckConfig) {
        let window_size = config.window_size.clamp(1, 1000);

        // Determine whether this probe counts as a failure.
        let is_failure = match outcome {
            ProbeOutcome::Failure { .. } => true,
            ProbeOutcome::Success { latency_ms } => {
                config.max_latency_ms > 0 && *latency_ms > config.max_latency_ms
            }
        };

        // Evict oldest if at capacity.
        if self.window.len() >= window_size {
            self.window.pop_front();
        }
        self.window.push_back(!is_failure);

        // Track latencies for successful probes.
        if let ProbeOutcome::Success { latency_ms } = outcome {
            if self.latencies.len() >= window_size {
                self.latencies.pop_front();
            }
            self.latencies.push_back(*latency_ms);
        }

        // Update totals.
        if is_failure {
            self.total_failures = self.total_failures.saturating_add(1);
        } else {
            self.total_successes = self.total_successes.saturating_add(1);
        }

        // Recompute health state.
        let n = self.window.len();
        if n < config.min_probes {
            // Not enough data yet — default to healthy.
            self.is_healthy = true;
            return;
        }

        let failures = self.window.iter().filter(|&&ok| !ok).count();
        let failure_ratio = failures as f64 / n as f64;

        if self.is_healthy {
            // Transition to unhealthy if failure ratio exceeds the threshold.
            if failure_ratio > config.failure_threshold {
                self.is_healthy = false;
            }
        } else {
            // Recover only when failure ratio drops below recovery threshold.
            if failure_ratio <= config.effective_recovery_threshold() {
                self.is_healthy = true;
            }
        }
    }

    /// Failure ratio of the current window (0.0 when no probes recorded).
    fn failure_ratio(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let failures = self.window.iter().filter(|&&ok| !ok).count();
        failures as f64 / self.window.len() as f64
    }

    /// Mean latency across all successful probes in the current window, or
    /// `None` when no successful probes have been recorded.
    fn mean_latency_ms(&self) -> Option<f64> {
        if self.latencies.is_empty() {
            return None;
        }
        let sum: u64 = self.latencies.iter().sum();
        Some(sum as f64 / self.latencies.len() as f64)
    }
}

// ─── ProviderStatus ───────────────────────────────────────────────────────────

/// A snapshot of the health state for a single provider.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderStatus {
    /// Provider name.
    pub name: String,
    /// Whether the provider is currently considered healthy.
    pub is_healthy: bool,
    /// Failure ratio in the current window (`[0.0, 1.0]`).
    pub failure_ratio: f64,
    /// Mean probe latency in the current window (milliseconds), or `None` if
    /// no successful probes are present in the window.
    pub mean_latency_ms: Option<f64>,
    /// Total successful probes since registration.
    pub total_successes: u64,
    /// Total failed probes since registration.
    pub total_failures: u64,
    /// Number of probes currently in the sliding window.
    pub window_probe_count: usize,
}

// ─── CdnHealthRegistry ────────────────────────────────────────────────────────

/// Aggregates health state across multiple CDN providers.
///
/// Providers are identified by string name and registered before probes are
/// recorded.  Unknown provider names passed to [`record_probe`] are silently
/// ignored so that callers do not need to synchronise provider registration
/// with probe collection.
///
/// [`record_probe`]: CdnHealthRegistry::record_probe
pub struct CdnHealthRegistry {
    config: HealthCheckConfig,
    providers: HashMap<String, ProviderHealth>,
}

impl CdnHealthRegistry {
    /// Create a new registry with the given configuration.
    pub fn new(config: HealthCheckConfig) -> Self {
        Self {
            config,
            providers: HashMap::new(),
        }
    }

    /// Register a provider by name.
    ///
    /// If a provider with the same name already exists this is a no-op.
    pub fn register(&mut self, name: impl Into<String>) {
        let n = name.into();
        self.providers
            .entry(n.clone())
            .or_insert_with(|| ProviderHealth::new(n, self.config.window_size));
    }

    /// Remove a provider from the registry.
    ///
    /// Returns `true` if the provider existed and was removed.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.providers.remove(name).is_some()
    }

    /// Record a probe outcome for the named provider.
    ///
    /// Silently ignored when `name` has not been registered.
    pub fn record_probe(&mut self, name: &str, outcome: ProbeOutcome) {
        if let Some(ph) = self.providers.get_mut(name) {
            ph.record(&outcome, &self.config.clone());
        }
    }

    /// Returns `true` if the named provider is currently healthy.
    ///
    /// Unregistered providers return `false`.
    #[must_use]
    pub fn is_healthy(&self, name: &str) -> bool {
        self.providers
            .get(name)
            .map(|ph| ph.is_healthy)
            .unwrap_or(false)
    }

    /// Iterate over the names of all currently healthy providers.
    pub fn healthy_providers(&self) -> impl Iterator<Item = &str> {
        self.providers
            .iter()
            .filter(|(_, ph)| ph.is_healthy)
            .map(|(name, _)| name.as_str())
    }

    /// Iterate over the names of all currently unhealthy providers.
    pub fn unhealthy_providers(&self) -> impl Iterator<Item = &str> {
        self.providers
            .iter()
            .filter(|(_, ph)| !ph.is_healthy)
            .map(|(name, _)| name.as_str())
    }

    /// Return a [`ProviderStatus`] snapshot for the named provider, or `None`
    /// if the provider has not been registered.
    #[must_use]
    pub fn status(&self, name: &str) -> Option<ProviderStatus> {
        self.providers.get(name).map(|ph| ProviderStatus {
            name: ph.name.clone(),
            is_healthy: ph.is_healthy,
            failure_ratio: ph.failure_ratio(),
            mean_latency_ms: ph.mean_latency_ms(),
            total_successes: ph.total_successes,
            total_failures: ph.total_failures,
            window_probe_count: ph.window.len(),
        })
    }

    /// Return status snapshots for all registered providers, sorted by name.
    #[must_use]
    pub fn all_statuses(&self) -> Vec<ProviderStatus> {
        let mut statuses: Vec<ProviderStatus> = self
            .providers
            .values()
            .map(|ph| ProviderStatus {
                name: ph.name.clone(),
                is_healthy: ph.is_healthy,
                failure_ratio: ph.failure_ratio(),
                mean_latency_ms: ph.mean_latency_ms(),
                total_successes: ph.total_successes,
                total_failures: ph.total_failures,
                window_probe_count: ph.window.len(),
            })
            .collect();
        statuses.sort_by(|a, b| a.name.cmp(&b.name));
        statuses
    }

    /// Number of registered providers.
    #[must_use]
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Number of currently healthy providers.
    #[must_use]
    pub fn healthy_count(&self) -> usize {
        self.providers.values().filter(|ph| ph.is_healthy).count()
    }

    /// Number of currently unhealthy providers.
    #[must_use]
    pub fn unhealthy_count(&self) -> usize {
        self.providers.values().filter(|ph| !ph.is_healthy).count()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a config with the given failure threshold and a tight window.
    fn cfg(failure_threshold: f64) -> HealthCheckConfig {
        HealthCheckConfig {
            window_size: 10,
            failure_threshold,
            recovery_threshold: None,
            max_latency_ms: 0,
            min_probes: 3,
        }
    }

    fn ok(lat: u64) -> ProbeOutcome {
        ProbeOutcome::Success { latency_ms: lat }
    }

    fn fail(reason: &str) -> ProbeOutcome {
        ProbeOutcome::Failure {
            reason: reason.to_owned(),
        }
    }

    // 1. Newly registered provider is healthy before any probes.
    #[test]
    fn test_new_provider_is_healthy() {
        let mut reg = CdnHealthRegistry::new(cfg(0.5));
        reg.register("cdn-a");
        assert!(reg.is_healthy("cdn-a"));
    }

    // 2. Unregistered provider returns false from is_healthy.
    #[test]
    fn test_unregistered_is_not_healthy() {
        let reg = CdnHealthRegistry::new(cfg(0.5));
        assert!(!reg.is_healthy("ghost"));
    }

    // 3. Provider with all successes remains healthy.
    #[test]
    fn test_all_successes_remain_healthy() {
        let mut reg = CdnHealthRegistry::new(cfg(0.5));
        reg.register("cdn-a");
        for _ in 0..10 {
            reg.record_probe("cdn-a", ok(10));
        }
        assert!(reg.is_healthy("cdn-a"));
    }

    // 4. Provider transitions to unhealthy above the failure threshold.
    #[test]
    fn test_transitions_unhealthy_above_threshold() {
        let mut reg = CdnHealthRegistry::new(cfg(0.5));
        reg.register("cdn-a");
        // 6 failures + 4 successes = 60% failure ratio > 50% threshold.
        for _ in 0..6 {
            reg.record_probe("cdn-a", fail("err"));
        }
        for _ in 0..4 {
            reg.record_probe("cdn-a", ok(10));
        }
        assert!(!reg.is_healthy("cdn-a"));
    }

    // 5. Provider recovers after failure ratio drops below recovery threshold.
    #[test]
    fn test_recovery_after_failures_clear() {
        let mut reg = CdnHealthRegistry::new(cfg(0.5));
        reg.register("cdn-a");
        // Drive into unhealthy state.
        for _ in 0..10 {
            reg.record_probe("cdn-a", fail("down"));
        }
        assert!(!reg.is_healthy("cdn-a"));
        // Now flush window with successes (window_size=10).
        for _ in 0..10 {
            reg.record_probe("cdn-a", ok(5));
        }
        // failure_ratio = 0% < recovery_threshold (25%) → healthy.
        assert!(reg.is_healthy("cdn-a"));
    }

    // 6. Probes on unknown provider are silently ignored.
    #[test]
    fn test_unknown_provider_probe_is_ignored() {
        let mut reg = CdnHealthRegistry::new(cfg(0.5));
        // No panic, no provider registered.
        reg.record_probe("ghost", ok(10));
        assert_eq!(reg.provider_count(), 0);
    }

    // 7. healthy_providers yields only healthy providers.
    #[test]
    fn test_healthy_providers_subset() {
        let mut reg = CdnHealthRegistry::new(cfg(0.5));
        reg.register("cdn-a");
        reg.register("cdn-b");
        // Drive cdn-b unhealthy.
        for _ in 0..10 {
            reg.record_probe("cdn-b", fail("timeout"));
        }
        let healthy: Vec<&str> = reg.healthy_providers().collect();
        assert!(healthy.contains(&"cdn-a"), "cdn-a should be healthy");
        assert!(!healthy.contains(&"cdn-b"), "cdn-b should be unhealthy");
    }

    // 8. status() returns correct snapshot for a named provider.
    #[test]
    fn test_status_snapshot() {
        let mut reg = CdnHealthRegistry::new(cfg(0.5));
        reg.register("cdn-a");
        reg.record_probe("cdn-a", ok(20));
        reg.record_probe("cdn-a", ok(40));
        reg.record_probe("cdn-a", fail("err"));
        let status = reg.status("cdn-a").expect("status");
        assert_eq!(status.name, "cdn-a");
        assert_eq!(status.total_successes, 2);
        assert_eq!(status.total_failures, 1);
        assert_eq!(status.window_probe_count, 3);
        // 1 failure / 3 probes = 33.3% < 50% threshold → healthy.
        assert!(status.is_healthy);
        // Mean latency: (20 + 40) / 2 = 30 ms.
        let lat = status.mean_latency_ms.expect("mean latency");
        assert!(
            (lat - 30.0).abs() < 0.001,
            "mean latency should be 30 ms, got {lat}"
        );
    }

    // 9. status() returns None for unregistered provider.
    #[test]
    fn test_status_none_for_unknown() {
        let reg = CdnHealthRegistry::new(cfg(0.5));
        assert!(reg.status("ghost").is_none());
    }

    // 10. Latency-based failure marks probe as failure when latency exceeds max.
    #[test]
    fn test_latency_failure_threshold() {
        let config = HealthCheckConfig {
            window_size: 10,
            failure_threshold: 0.5,
            recovery_threshold: None,
            max_latency_ms: 100, // probes > 100 ms count as failures
            min_probes: 3,
        };
        let mut reg = CdnHealthRegistry::new(config);
        reg.register("cdn-slow");
        // All probes succeed but with high latency.
        for _ in 0..10 {
            reg.record_probe("cdn-slow", ProbeOutcome::Success { latency_ms: 500 });
        }
        // All 10 should be latency-failures → ratio = 100% > 50% → unhealthy.
        assert!(!reg.is_healthy("cdn-slow"));
    }

    // 11. Provider with insufficient probes defaults to healthy (benefit of doubt).
    #[test]
    fn test_below_min_probes_defaults_healthy() {
        let config = HealthCheckConfig {
            window_size: 10,
            failure_threshold: 0.5,
            recovery_threshold: None,
            max_latency_ms: 0,
            min_probes: 5, // need at least 5 probes before health decisions
        };
        let mut reg = CdnHealthRegistry::new(config);
        reg.register("cdn-new");
        // Only 2 failures — below min_probes=5 so no decision yet.
        reg.record_probe("cdn-new", fail("err"));
        reg.record_probe("cdn-new", fail("err"));
        // Should still be healthy (insufficient data).
        assert!(reg.is_healthy("cdn-new"));
    }

    // 12. healthy_count and unhealthy_count aggregate correctly.
    #[test]
    fn test_count_helpers() {
        let mut reg = CdnHealthRegistry::new(cfg(0.5));
        reg.register("a");
        reg.register("b");
        reg.register("c");
        // Drive "b" and "c" unhealthy.
        for name in ["b", "c"] {
            for _ in 0..10 {
                reg.record_probe(name, fail("down"));
            }
        }
        assert_eq!(reg.healthy_count(), 1);
        assert_eq!(reg.unhealthy_count(), 2);
    }

    // 13. unregister removes provider.
    #[test]
    fn test_unregister_removes_provider() {
        let mut reg = CdnHealthRegistry::new(cfg(0.5));
        reg.register("cdn-a");
        assert_eq!(reg.provider_count(), 1);
        let removed = reg.unregister("cdn-a");
        assert!(removed);
        assert_eq!(reg.provider_count(), 0);
        assert!(!reg.is_healthy("cdn-a"));
    }

    // 14. Window evicts oldest probes and uses only the latest `window_size`.
    #[test]
    fn test_window_eviction_replaces_old_probes() {
        let config = HealthCheckConfig {
            window_size: 5,
            failure_threshold: 0.5,
            recovery_threshold: None,
            max_latency_ms: 0,
            min_probes: 3,
        };
        let mut reg = CdnHealthRegistry::new(config);
        reg.register("cdn-a");
        // Fill with 5 failures → unhealthy.
        for _ in 0..5 {
            reg.record_probe("cdn-a", fail("err"));
        }
        assert!(!reg.is_healthy("cdn-a"));
        // Now push 5 successes — the old failures are evicted.
        for _ in 0..5 {
            reg.record_probe("cdn-a", ok(10));
        }
        // Window is now 100% successes → should recover.
        assert!(reg.is_healthy("cdn-a"));
    }

    // 15. all_statuses returns sorted list.
    #[test]
    fn test_all_statuses_sorted() {
        let mut reg = CdnHealthRegistry::new(cfg(0.5));
        reg.register("zzz");
        reg.register("aaa");
        reg.register("mmm");
        let statuses = reg.all_statuses();
        let names: Vec<&str> = statuses.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["aaa", "mmm", "zzz"]);
    }

    // 16. ProbeOutcome helpers work correctly.
    #[test]
    fn test_probe_outcome_helpers() {
        let success = ProbeOutcome::Success { latency_ms: 42 };
        let failure = ProbeOutcome::Failure {
            reason: "timeout".into(),
        };
        assert!(success.is_success());
        assert!(!failure.is_success());
        assert_eq!(success.latency_ms(), Some(42));
        assert_eq!(failure.latency_ms(), None);
    }
}
