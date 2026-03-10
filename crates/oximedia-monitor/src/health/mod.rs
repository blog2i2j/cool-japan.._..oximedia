//! Health check system for component status monitoring.
//!
//! Provides a registry of named health checks, concurrent execution of all
//! checks, overall status aggregation, and JSON serialisation suitable for
//! use as an HTTP `/health` endpoint response.

use crate::error::MonitorResult;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Health status of a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All systems nominal.
    Healthy,
    /// Degraded – still functional but impaired.
    Degraded,
    /// Unhealthy – component is not functioning.
    Unhealthy,
}

impl HealthStatus {
    /// Return the string representation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
        }
    }

    /// Return `true` if the status is `Healthy`.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Merge two statuses, returning the worse of the two.
    #[must_use]
    pub fn worst(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Health information for a single component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name.
    pub name: String,
    /// Current status.
    pub status: HealthStatus,
    /// Human-readable status message or error description.
    pub message: String,
    /// Time of the last check.
    pub last_checked: DateTime<Utc>,
    /// Optional latency of the check in milliseconds.
    pub latency_ms: Option<u64>,
}

impl ComponentHealth {
    /// Create a healthy component result.
    #[must_use]
    pub fn healthy(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Healthy,
            message: "OK".to_string(),
            last_checked: Utc::now(),
            latency_ms: None,
        }
    }

    /// Create a degraded component result.
    #[must_use]
    pub fn degraded(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Degraded,
            message: message.into(),
            last_checked: Utc::now(),
            latency_ms: None,
        }
    }

    /// Create an unhealthy component result.
    #[must_use]
    pub fn unhealthy(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Unhealthy,
            message: message.into(),
            last_checked: Utc::now(),
            latency_ms: None,
        }
    }

    /// Attach a latency measurement (milliseconds).
    #[must_use]
    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = Some(ms);
        self
    }
}

/// Type alias for a boxed, thread-safe health-check function.
///
/// The function receives the component name and returns a `ComponentHealth`.
type CheckFn = Arc<dyn Fn(&str) -> ComponentHealth + Send + Sync>;

/// Registry of named health checks.
///
/// Register check functions with [`HealthChecker::register`], then call
/// [`HealthChecker::check_all`] to run them all and get a snapshot of every
/// component's health.
pub struct HealthChecker {
    checks: RwLock<HashMap<String, CheckFn>>,
}

impl HealthChecker {
    /// Create an empty health checker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            checks: RwLock::new(HashMap::new()),
        }
    }

    /// Register a health check for a named component.
    ///
    /// If a check with the same name already exists it is replaced.
    pub fn register<F>(&self, name: impl Into<String>, check_fn: F)
    where
        F: Fn(&str) -> ComponentHealth + Send + Sync + 'static,
    {
        self.checks.write().insert(name.into(), Arc::new(check_fn));
    }

    /// Run all registered health checks and return their results.
    ///
    /// Checks are executed synchronously in the calling thread.  For very
    /// slow I/O-bound checks consider wrapping in a `tokio::spawn` before
    /// calling this.
    #[must_use]
    pub fn check_all(&self) -> HashMap<String, ComponentHealth> {
        let checks = self.checks.read();
        let mut results = HashMap::with_capacity(checks.len());
        for (name, check_fn) in checks.iter() {
            let start = std::time::Instant::now();
            let mut result = check_fn(name.as_str());
            let elapsed = start.elapsed().as_millis() as u64;
            result.last_checked = Utc::now();
            result.latency_ms = Some(result.latency_ms.unwrap_or(elapsed));
            results.insert(name.clone(), result);
        }
        results
    }

    /// Return the worst (most severe) status across all components.
    ///
    /// Returns `Healthy` if no checks are registered.
    #[must_use]
    pub fn overall_status(&self) -> HealthStatus {
        let results = self.check_all();
        results
            .values()
            .map(|c| c.status)
            .fold(HealthStatus::Healthy, HealthStatus::worst)
    }

    /// Serialize the full health report to a JSON object suitable for an HTTP
    /// health endpoint.
    ///
    /// The returned object has the shape:
    /// ```json
    /// {
    ///   "status": "healthy",
    ///   "timestamp": "...",
    ///   "components": { "db": { ... }, "redis": { ... } }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization fails.
    pub fn to_json(&self) -> MonitorResult<String> {
        let components = self.check_all();
        let overall = components
            .values()
            .map(|c| c.status)
            .fold(HealthStatus::Healthy, HealthStatus::worst);

        let payload = serde_json::json!({
            "status": overall.as_str(),
            "timestamp": Utc::now().to_rfc3339(),
            "components": components,
        });

        Ok(serde_json::to_string_pretty(&payload)?)
    }

    /// Return the number of registered checks.
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.checks.read().len()
    }

    /// Remove a health check by name. Returns `true` if it existed.
    pub fn deregister(&self, name: &str) -> bool {
        self.checks.write().remove(name).is_some()
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_ordering() {
        assert!(HealthStatus::Healthy < HealthStatus::Degraded);
        assert!(HealthStatus::Degraded < HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_status_worst() {
        assert_eq!(
            HealthStatus::Healthy.worst(HealthStatus::Degraded),
            HealthStatus::Degraded
        );
        assert_eq!(
            HealthStatus::Unhealthy.worst(HealthStatus::Healthy),
            HealthStatus::Unhealthy
        );
        assert_eq!(
            HealthStatus::Degraded.worst(HealthStatus::Degraded),
            HealthStatus::Degraded
        );
    }

    #[test]
    fn test_register_and_check_all() {
        let checker = HealthChecker::new();

        checker.register("api", |name| ComponentHealth::healthy(name));
        checker.register("db", |name| ComponentHealth::degraded(name, "slow queries"));

        let results = checker.check_all();
        assert_eq!(results.len(), 2);
        assert_eq!(results["api"].status, HealthStatus::Healthy);
        assert_eq!(results["db"].status, HealthStatus::Degraded);
    }

    #[test]
    fn test_overall_status_all_healthy() {
        let checker = HealthChecker::new();
        checker.register("a", |n| ComponentHealth::healthy(n));
        checker.register("b", |n| ComponentHealth::healthy(n));

        assert_eq!(checker.overall_status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_overall_status_one_unhealthy() {
        let checker = HealthChecker::new();
        checker.register("a", |n| ComponentHealth::healthy(n));
        checker.register("b", |n| ComponentHealth::unhealthy(n, "crashed"));

        assert_eq!(checker.overall_status(), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_overall_status_empty() {
        let checker = HealthChecker::new();
        assert_eq!(checker.overall_status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_to_json() {
        let checker = HealthChecker::new();
        checker.register("encoder", |n| ComponentHealth::healthy(n));

        let json = checker.to_json().expect("to_json should succeed");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("failed to deserialize from JSON");

        assert_eq!(parsed["status"], "healthy");
        assert!(parsed["components"]["encoder"].is_object());
    }

    #[test]
    fn test_deregister() {
        let checker = HealthChecker::new();
        checker.register("service", |n| ComponentHealth::healthy(n));
        assert_eq!(checker.check_count(), 1);

        assert!(checker.deregister("service"));
        assert!(!checker.deregister("service")); // already removed
        assert_eq!(checker.check_count(), 0);
    }

    #[test]
    fn test_component_health_constructors() {
        let h = ComponentHealth::healthy("svc");
        assert_eq!(h.status, HealthStatus::Healthy);
        assert_eq!(h.message, "OK");

        let d = ComponentHealth::degraded("svc", "slow");
        assert_eq!(d.status, HealthStatus::Degraded);
        assert_eq!(d.message, "slow");

        let u = ComponentHealth::unhealthy("svc", "down");
        assert_eq!(u.status, HealthStatus::Unhealthy);
        assert_eq!(u.message, "down");
    }

    #[test]
    fn test_latency_attached() {
        let h = ComponentHealth::healthy("svc").with_latency(42);
        assert_eq!(h.latency_ms, Some(42));
    }
}
