//! Plugin health monitoring — periodic liveness probes and health reporting.
//!
//! This module provides a [`PluginHealthMonitor`] that tracks the liveness of
//! registered plugins by running periodic health checks and maintaining a history
//! of check results.  It is intentionally dependency-free with respect to OS
//! threads; callers decide *when* to advance the probe cycle (pull model).
//!
//! # Design
//!
//! - [`HealthProbe`] is a user-supplied callback that returns a [`ProbeOutcome`].
//! - [`HealthRecord`] stores the last N probe results for a plugin.
//! - [`PluginHealthMonitor`] manages all registered probes and exposes
//!   `run_due_probes` to execute any probes whose next scheduled time has passed.
//! - Healthy / degraded / failing states are derived from recent outcome history.
//!
//! # Health Status Derivation
//!
//! A plugin's [`HealthStatus`] is computed from the last `window` probe results:
//! - [`HealthStatus::Healthy`]   — all recent outcomes are `Ok`.
//! - [`HealthStatus::Degraded`]  — some outcomes are `Ok`, some are failures.
//! - [`HealthStatus::Failing`]   — all recent outcomes (≥ 1) are failures.
//! - [`HealthStatus::Unknown`]   — no probes have been run yet.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── ProbeOutcome ──────────────────────────────────────────────────────────────

/// The result of a single liveness probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// The plugin responded within the deadline and is considered live.
    Ok,
    /// The plugin did not respond in time.
    Timeout {
        /// Elapsed milliseconds before the probe gave up.
        elapsed_ms: u64,
    },
    /// The probe call returned an error.
    Error {
        /// Human-readable description of the error.
        message: String,
    },
}

impl ProbeOutcome {
    /// Return `true` when the outcome is [`ProbeOutcome::Ok`].
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Return `true` when the outcome indicates a failure (timeout or error).
    pub fn is_failure(&self) -> bool {
        !self.is_ok()
    }
}

impl std::fmt::Display for ProbeOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => write!(f, "ok"),
            Self::Timeout { elapsed_ms } => write!(f, "timeout ({elapsed_ms} ms)"),
            Self::Error { message } => write!(f, "error: {message}"),
        }
    }
}

// ── HealthStatus ──────────────────────────────────────────────────────────────

/// The derived health status of a plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// No probes have been executed yet.
    Unknown,
    /// All recent probes succeeded.
    Healthy,
    /// Some recent probes failed but at least one succeeded.
    Degraded,
    /// All recent probes (at least one) failed.
    Failing,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => write!(f, "unknown"),
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Failing => write!(f, "failing"),
        }
    }
}

// ── ProbeResult ───────────────────────────────────────────────────────────────

/// A timestamped probe result kept in the health record.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// When the probe was executed.
    pub timestamp: Instant,
    /// The outcome reported by the probe.
    pub outcome: ProbeOutcome,
}

// ── HealthRecord ──────────────────────────────────────────────────────────────

/// A rolling window of probe results for a single plugin.
///
/// The window size is set at construction time and limits memory usage.
#[derive(Debug)]
pub struct HealthRecord {
    /// Plugin identifier.
    pub plugin_id: String,
    /// Rolling window of the most-recent `window_size` probe results.
    results: Vec<ProbeResult>,
    /// Maximum number of results to keep.
    window_size: usize,
}

impl HealthRecord {
    /// Create a new empty record.
    pub fn new(plugin_id: impl Into<String>, window_size: usize) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            results: Vec::with_capacity(window_size),
            window_size: window_size.max(1),
        }
    }

    /// Append a probe result, evicting the oldest entry if the window is full.
    pub fn push(&mut self, result: ProbeResult) {
        if self.results.len() >= self.window_size {
            self.results.remove(0);
        }
        self.results.push(result);
    }

    /// Return all stored probe results (oldest first).
    pub fn results(&self) -> &[ProbeResult] {
        &self.results
    }

    /// Return the most-recent probe result, if any.
    pub fn last(&self) -> Option<&ProbeResult> {
        self.results.last()
    }

    /// Derive the overall health status from the current result window.
    pub fn status(&self) -> HealthStatus {
        if self.results.is_empty() {
            return HealthStatus::Unknown;
        }

        let ok_count = self.results.iter().filter(|r| r.outcome.is_ok()).count();
        let total = self.results.len();

        if ok_count == total {
            HealthStatus::Healthy
        } else if ok_count == 0 {
            HealthStatus::Failing
        } else {
            HealthStatus::Degraded
        }
    }

    /// Return the fraction of successful probes in the current window (0.0 – 1.0).
    pub fn success_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let ok = self.results.iter().filter(|r| r.outcome.is_ok()).count();
        ok as f64 / self.results.len() as f64
    }

    /// Return the number of consecutive failures at the tail of the window.
    pub fn consecutive_failures(&self) -> usize {
        self.results
            .iter()
            .rev()
            .take_while(|r| r.outcome.is_failure())
            .count()
    }
}

// ── ProbeConfig ───────────────────────────────────────────────────────────────

/// Configuration for a registered health probe.
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    /// How often the probe should run.
    pub interval: Duration,
    /// After this many consecutive failures the plugin is considered failing.
    pub failure_threshold: usize,
    /// How many probe results to keep in the rolling window.
    pub window_size: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            failure_threshold: 3,
            window_size: 10,
        }
    }
}

// ── RegisteredProbe ───────────────────────────────────────────────────────────

/// Internal bookkeeping for a single registered probe.
struct RegisteredProbe {
    config: ProbeConfig,
    /// The user-supplied probe function.
    probe_fn: Box<dyn Fn() -> ProbeOutcome + Send + Sync>,
    /// When the probe is next eligible to run.
    next_due: Instant,
    /// Rolling result window.
    record: HealthRecord,
}

// ── PluginHealthMonitor ───────────────────────────────────────────────────────

/// Manages health probes for all registered plugins.
///
/// Call [`run_due_probes`](Self::run_due_probes) periodically (e.g. from a
/// background thread or an async task) to advance the probe cycle.  The
/// monitor is designed to be wrapped in `Arc<Mutex<…>>` when shared between
/// threads.
///
/// # Example
///
/// ```rust
/// use oximedia_plugin::health_check::{PluginHealthMonitor, ProbeConfig, ProbeOutcome};
///
/// let mut monitor = PluginHealthMonitor::new();
/// monitor.register("my-plugin", ProbeConfig::default(), || ProbeOutcome::Ok);
///
/// // Force-run all probes regardless of schedule.
/// let ran = monitor.probe_all();
/// assert_eq!(ran, 1);
/// assert!(monitor.is_healthy("my-plugin"));
/// ```
pub struct PluginHealthMonitor {
    probes: HashMap<String, RegisteredProbe>,
}

impl PluginHealthMonitor {
    /// Create an empty monitor.
    pub fn new() -> Self {
        Self {
            probes: HashMap::new(),
        }
    }

    /// Register a probe for the given plugin.
    ///
    /// If a probe is already registered under `plugin_id` it is replaced.
    ///
    /// `probe_fn` will be called on each scheduled tick; it must return a
    /// [`ProbeOutcome`].
    pub fn register<F>(&mut self, plugin_id: &str, config: ProbeConfig, probe_fn: F)
    where
        F: Fn() -> ProbeOutcome + Send + Sync + 'static,
    {
        let window_size = config.window_size;
        let registered = RegisteredProbe {
            record: HealthRecord::new(plugin_id, window_size),
            config,
            probe_fn: Box::new(probe_fn),
            next_due: Instant::now(),
        };
        self.probes.insert(plugin_id.to_string(), registered);
    }

    /// Unregister a probe.  Returns `true` if there was a probe to remove.
    pub fn unregister(&mut self, plugin_id: &str) -> bool {
        self.probes.remove(plugin_id).is_some()
    }

    /// Run any probes whose next-due time has elapsed.
    ///
    /// Returns the number of probes that were executed.
    pub fn run_due_probes(&mut self) -> usize {
        let now = Instant::now();
        let mut count = 0;

        for registered in self.probes.values_mut() {
            if now >= registered.next_due {
                let outcome = (registered.probe_fn)();
                registered.record.push(ProbeResult {
                    timestamp: Instant::now(),
                    outcome,
                });
                registered.next_due = Instant::now() + registered.config.interval;
                count += 1;
            }
        }

        count
    }

    /// Execute all probes immediately, ignoring the schedule.
    ///
    /// Returns the number of probes that were executed.
    pub fn probe_all(&mut self) -> usize {
        let count = self.probes.len();
        for registered in self.probes.values_mut() {
            let outcome = (registered.probe_fn)();
            registered.record.push(ProbeResult {
                timestamp: Instant::now(),
                outcome,
            });
            registered.next_due = Instant::now() + registered.config.interval;
        }
        count
    }

    /// Force-run a single probe by plugin ID, regardless of schedule.
    ///
    /// Returns `true` if the probe exists and was executed.
    pub fn probe_one(&mut self, plugin_id: &str) -> bool {
        let Some(registered) = self.probes.get_mut(plugin_id) else {
            return false;
        };
        let outcome = (registered.probe_fn)();
        registered.record.push(ProbeResult {
            timestamp: Instant::now(),
            outcome,
        });
        registered.next_due = Instant::now() + registered.config.interval;
        true
    }

    /// Get the current health status of a plugin.
    ///
    /// Returns `None` if no probe is registered for the given plugin.
    pub fn status(&self, plugin_id: &str) -> Option<HealthStatus> {
        self.probes
            .get(plugin_id)
            .map(|r| r.record.status())
    }

    /// Return `true` if the plugin's current status is [`HealthStatus::Healthy`].
    ///
    /// Returns `false` if no probe is registered.
    pub fn is_healthy(&self, plugin_id: &str) -> bool {
        self.status(plugin_id) == Some(HealthStatus::Healthy)
    }

    /// Return a reference to the [`HealthRecord`] for the given plugin.
    pub fn record(&self, plugin_id: &str) -> Option<&HealthRecord> {
        self.probes.get(plugin_id).map(|r| &r.record)
    }

    /// Return the IDs of all plugins whose status is [`HealthStatus::Failing`].
    pub fn failing_plugins(&self) -> Vec<&str> {
        self.probes
            .iter()
            .filter(|(_, r)| r.record.status() == HealthStatus::Failing)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Return the IDs of all monitored plugins.
    pub fn monitored_plugins(&self) -> Vec<&str> {
        self.probes.keys().map(String::as_str).collect()
    }

    /// Return the number of registered probes.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Determine whether a plugin should be considered for graceful shutdown
    /// based on its consecutive failure count versus the configured threshold.
    pub fn should_deactivate(&self, plugin_id: &str) -> bool {
        let Some(registered) = self.probes.get(plugin_id) else {
            return false;
        };
        let consecutive = registered.record.consecutive_failures();
        consecutive >= registered.config.failure_threshold
    }
}

impl Default for PluginHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // 1. ProbeOutcome helpers
    #[test]
    fn test_probe_outcome_helpers() {
        assert!(ProbeOutcome::Ok.is_ok());
        assert!(!ProbeOutcome::Ok.is_failure());

        let t = ProbeOutcome::Timeout { elapsed_ms: 100 };
        assert!(!t.is_ok());
        assert!(t.is_failure());

        let e = ProbeOutcome::Error {
            message: "oops".to_string(),
        };
        assert!(e.is_failure());
    }

    // 2. ProbeOutcome display
    #[test]
    fn test_probe_outcome_display() {
        assert_eq!(ProbeOutcome::Ok.to_string(), "ok");
        assert!(ProbeOutcome::Timeout { elapsed_ms: 50 }
            .to_string()
            .contains("50"));
        assert!(ProbeOutcome::Error {
            message: "bad".to_string()
        }
        .to_string()
        .contains("bad"));
    }

    // 3. HealthStatus display
    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Unknown.to_string(), "unknown");
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Failing.to_string(), "failing");
    }

    // 4. HealthRecord starts as Unknown
    #[test]
    fn test_health_record_initial_unknown() {
        let record = HealthRecord::new("p", 5);
        assert_eq!(record.status(), HealthStatus::Unknown);
        assert_eq!(record.success_rate(), 0.0);
        assert_eq!(record.consecutive_failures(), 0);
    }

    // 5. HealthRecord derives Healthy from all-ok results
    #[test]
    fn test_health_record_healthy() {
        let mut record = HealthRecord::new("p", 5);
        for _ in 0..3 {
            record.push(ProbeResult {
                timestamp: Instant::now(),
                outcome: ProbeOutcome::Ok,
            });
        }
        assert_eq!(record.status(), HealthStatus::Healthy);
        assert!((record.success_rate() - 1.0).abs() < 1e-10);
    }

    // 6. HealthRecord derives Failing from all-error results
    #[test]
    fn test_health_record_failing() {
        let mut record = HealthRecord::new("p", 5);
        for _ in 0..3 {
            record.push(ProbeResult {
                timestamp: Instant::now(),
                outcome: ProbeOutcome::Error {
                    message: "dead".to_string(),
                },
            });
        }
        assert_eq!(record.status(), HealthStatus::Failing);
        assert_eq!(record.success_rate(), 0.0);
        assert_eq!(record.consecutive_failures(), 3);
    }

    // 7. HealthRecord derives Degraded from mixed results
    #[test]
    fn test_health_record_degraded() {
        let mut record = HealthRecord::new("p", 5);
        record.push(ProbeResult {
            timestamp: Instant::now(),
            outcome: ProbeOutcome::Ok,
        });
        record.push(ProbeResult {
            timestamp: Instant::now(),
            outcome: ProbeOutcome::Error {
                message: "bad".to_string(),
            },
        });
        assert_eq!(record.status(), HealthStatus::Degraded);
    }

    // 8. Rolling window eviction
    #[test]
    fn test_health_record_window_eviction() {
        let mut record = HealthRecord::new("p", 3);
        for _ in 0..3 {
            record.push(ProbeResult {
                timestamp: Instant::now(),
                outcome: ProbeOutcome::Ok,
            });
        }
        // Window is now full (3 ok). Push one failure.
        record.push(ProbeResult {
            timestamp: Instant::now(),
            outcome: ProbeOutcome::Error {
                message: "fail".to_string(),
            },
        });
        // Window should still be size 3: [ok, ok, fail]
        assert_eq!(record.results().len(), 3);
        // Now degraded (2 ok, 1 fail)
        assert_eq!(record.status(), HealthStatus::Degraded);
    }

    // 9. Monitor registers and can probe_all
    #[test]
    fn test_monitor_probe_all() {
        let mut monitor = PluginHealthMonitor::new();
        monitor.register("plugin-a", ProbeConfig::default(), || ProbeOutcome::Ok);
        monitor.register(
            "plugin-b",
            ProbeConfig::default(),
            || ProbeOutcome::Error {
                message: "down".to_string(),
            },
        );
        let ran = monitor.probe_all();
        assert_eq!(ran, 2);
        assert!(monitor.is_healthy("plugin-a"));
        assert_eq!(monitor.status("plugin-b"), Some(HealthStatus::Failing));
    }

    // 10. probe_one runs single probe
    #[test]
    fn test_probe_one() {
        let mut monitor = PluginHealthMonitor::new();
        monitor.register("p", ProbeConfig::default(), || ProbeOutcome::Ok);
        assert!(monitor.probe_one("p"));
        assert_eq!(monitor.status("p"), Some(HealthStatus::Healthy));
        assert!(!monitor.probe_one("does-not-exist"));
    }

    // 11. unregister removes a probe
    #[test]
    fn test_unregister() {
        let mut monitor = PluginHealthMonitor::new();
        monitor.register("p", ProbeConfig::default(), || ProbeOutcome::Ok);
        assert!(monitor.unregister("p"));
        assert!(!monitor.unregister("p")); // already gone
        assert_eq!(monitor.probe_count(), 0);
    }

    // 12. failing_plugins returns only failing ones
    #[test]
    fn test_failing_plugins() {
        let mut monitor = PluginHealthMonitor::new();
        monitor.register("good", ProbeConfig::default(), || ProbeOutcome::Ok);
        monitor.register(
            "bad",
            ProbeConfig::default(),
            || ProbeOutcome::Error {
                message: "x".to_string(),
            },
        );
        monitor.probe_all();
        let failing = monitor.failing_plugins();
        assert_eq!(failing, vec!["bad"]);
        assert!(!monitor.is_healthy("bad"));
    }

    // 13. should_deactivate uses failure_threshold
    #[test]
    fn test_should_deactivate() {
        let config = ProbeConfig {
            failure_threshold: 2,
            interval: Duration::from_secs(1),
            window_size: 10,
        };
        let mut monitor = PluginHealthMonitor::new();
        monitor.register(
            "p",
            config,
            || ProbeOutcome::Error {
                message: "x".to_string(),
            },
        );

        monitor.probe_one("p");
        // Only 1 consecutive failure, threshold is 2 → not yet
        assert!(!monitor.should_deactivate("p"));

        monitor.probe_one("p");
        // Now 2 consecutive failures → deactivate
        assert!(monitor.should_deactivate("p"));
    }

    // 14. probe invocation count via shared atomic
    #[test]
    fn test_probe_invocation_count() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let mut monitor = PluginHealthMonitor::new();
        monitor.register("p", ProbeConfig::default(), move || {
            c.fetch_add(1, Ordering::SeqCst);
            ProbeOutcome::Ok
        });

        monitor.probe_one("p");
        monitor.probe_one("p");
        monitor.probe_one("p");
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    // 15. run_due_probes runs only probes whose next_due has elapsed
    #[test]
    fn test_run_due_probes() {
        let mut monitor = PluginHealthMonitor::new();
        // Interval far in the future — should NOT be run by run_due_probes on the first call
        // unless forced.  Use probe_all first to seed the schedule, then verify run_due_probes
        // won't re-run them immediately.
        let long_interval = ProbeConfig {
            interval: Duration::from_secs(3600),
            ..ProbeConfig::default()
        };
        monitor.register("p", long_interval, || ProbeOutcome::Ok);

        // On fresh registration, next_due is Instant::now() so the probe IS due immediately.
        let ran = monitor.run_due_probes();
        assert_eq!(ran, 1);

        // After running, next_due is 1 hour away. Run again — should be 0.
        let ran2 = monitor.run_due_probes();
        assert_eq!(ran2, 0);
    }

    // 16. monitored_plugins returns all IDs
    #[test]
    fn test_monitored_plugins() {
        let mut monitor = PluginHealthMonitor::new();
        monitor.register("a", ProbeConfig::default(), || ProbeOutcome::Ok);
        monitor.register("b", ProbeConfig::default(), || ProbeOutcome::Ok);
        let mut ids = monitor.monitored_plugins();
        ids.sort_unstable();
        assert_eq!(ids, vec!["a", "b"]);
    }
}
