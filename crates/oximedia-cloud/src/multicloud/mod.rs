//! Multi-cloud orchestration: endpoint routing, replication, and health checks.
//!
//! Provides abstractions for routing requests across multiple cloud providers
//! with priority-based selection, replication targets, and health tracking.

#![allow(dead_code)]

/// A single cloud endpoint with routing priority.
#[derive(Debug, Clone)]
pub struct CloudEndpoint {
    /// Cloud provider name (e.g., `"aws"`, `"azure"`, `"gcp"`).
    pub provider: String,
    /// Cloud region (e.g., `"us-east-1"`).
    pub region: String,
    /// Base URL or connection string.
    pub endpoint_url: String,
    /// Routing priority – lower value is higher priority (1 = primary).
    pub priority: u32,
}

impl CloudEndpoint {
    /// Create a new `CloudEndpoint`.
    #[must_use]
    pub fn new(
        provider: impl Into<String>,
        region: impl Into<String>,
        endpoint_url: impl Into<String>,
        priority: u32,
    ) -> Self {
        Self {
            provider: provider.into(),
            region: region.into(),
            endpoint_url: endpoint_url.into(),
            priority,
        }
    }

    /// Returns `true` when this endpoint has the highest priority (`priority == 1`).
    #[must_use]
    pub fn is_primary(&self) -> bool {
        self.priority == 1
    }
}

/// Policy controlling multi-cloud routing and replication behaviour.
#[derive(Debug, Clone)]
pub struct MultiCloudPolicy {
    /// Available cloud endpoints.
    pub endpoints: Vec<CloudEndpoint>,
    /// Whether automatic failover to secondary endpoints is enabled.
    pub failover_enabled: bool,
    /// Whether geo-proximity routing is applied.
    pub geo_routing: bool,
}

impl MultiCloudPolicy {
    /// Create a new policy.
    #[must_use]
    pub fn new(failover_enabled: bool, geo_routing: bool) -> Self {
        Self {
            endpoints: Vec::new(),
            failover_enabled,
            geo_routing,
        }
    }

    /// Add an endpoint.
    pub fn add_endpoint(&mut self, ep: CloudEndpoint) {
        self.endpoints.push(ep);
    }

    /// Returns a reference to the primary endpoint (lowest priority number),
    /// or `None` when no endpoints are registered.
    #[must_use]
    pub fn primary(&self) -> Option<&CloudEndpoint> {
        self.endpoints.iter().min_by_key(|ep| ep.priority)
    }

    /// Returns all endpoints sorted by priority ascending (lowest number first).
    #[must_use]
    pub fn sorted_by_priority(&self) -> Vec<&CloudEndpoint> {
        let mut sorted: Vec<&CloudEndpoint> = self.endpoints.iter().collect();
        sorted.sort_by_key(|ep| ep.priority);
        sorted
    }
}

/// Identifies a cloud storage bucket on a specific provider/region.
#[derive(Debug, Clone)]
pub struct ReplicationTarget {
    /// Cloud provider name.
    pub provider: String,
    /// Cloud region.
    pub region: String,
    /// Bucket name.
    pub bucket: String,
}

impl ReplicationTarget {
    /// Create a new `ReplicationTarget`.
    #[must_use]
    pub fn new(
        provider: impl Into<String>,
        region: impl Into<String>,
        bucket: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            region: region.into(),
            bucket: bucket.into(),
        }
    }

    /// Unique identifier combining provider, region, and bucket.
    #[must_use]
    pub fn identifier(&self) -> String {
        format!("{}:{}:{}", self.provider, self.region, self.bucket)
    }
}

/// Configuration for replicating data from one source to multiple targets.
#[derive(Debug, Clone)]
pub struct MultiCloudReplication {
    /// Source location.
    pub source: ReplicationTarget,
    /// Replication targets.
    targets: Vec<ReplicationTarget>,
}

impl MultiCloudReplication {
    /// Create a new replication configuration with a source and no targets.
    #[must_use]
    pub fn new(source: ReplicationTarget) -> Self {
        Self {
            source,
            targets: Vec::new(),
        }
    }

    /// Add a replication target.
    pub fn add_target(&mut self, target: ReplicationTarget) {
        self.targets.push(target);
    }

    /// Number of replication targets.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// References to all replication targets.
    #[must_use]
    pub fn all_targets(&self) -> Vec<&ReplicationTarget> {
        self.targets.iter().collect()
    }
}

/// Health state tracker for a single cloud endpoint.
#[derive(Debug, Clone)]
pub struct CloudHealthCheck {
    /// Endpoint URL being monitored.
    pub endpoint: String,
    /// Millisecond timestamp of the last successful health check.
    pub last_ok_ms: u64,
    /// Number of consecutive health-check failures.
    pub consecutive_failures: u32,
}

impl CloudHealthCheck {
    /// Create a new `CloudHealthCheck`.
    #[must_use]
    pub fn new(endpoint: impl Into<String>, last_ok_ms: u64) -> Self {
        Self {
            endpoint: endpoint.into(),
            last_ok_ms,
            consecutive_failures: 0,
        }
    }

    /// Record a successful health check at `now_ms`.
    pub fn record_success(&mut self, now_ms: u64) {
        self.last_ok_ms = now_ms;
        self.consecutive_failures = 0;
    }

    /// Record a failed health check.
    pub fn record_failure(&mut self) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
    }

    /// Returns `true` when the endpoint is considered healthy.
    ///
    /// An endpoint is healthy when there are no consecutive failures *and*
    /// the last successful check was within `timeout_ms` of `now_ms`.
    #[must_use]
    pub fn is_healthy(&self, now_ms: u64, timeout_ms: u64) -> bool {
        self.consecutive_failures == 0 && now_ms.saturating_sub(self.last_ok_ms) <= timeout_ms
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ep(priority: u32) -> CloudEndpoint {
        CloudEndpoint::new("aws", "us-east-1", "https://example.com", priority)
    }

    // 1. CloudEndpoint::is_primary – priority 1
    #[test]
    fn test_cloud_endpoint_is_primary() {
        assert!(make_ep(1).is_primary());
        assert!(!make_ep(2).is_primary());
        assert!(!make_ep(10).is_primary());
    }

    // 2. CloudEndpoint fields set correctly
    #[test]
    fn test_cloud_endpoint_fields() {
        let ep = CloudEndpoint::new("gcp", "eu-west", "https://gcp.example.com", 3);
        assert_eq!(ep.provider, "gcp");
        assert_eq!(ep.region, "eu-west");
        assert_eq!(ep.endpoint_url, "https://gcp.example.com");
        assert_eq!(ep.priority, 3);
    }

    // 3. MultiCloudPolicy::primary – empty returns None
    #[test]
    fn test_policy_primary_empty() {
        let policy = MultiCloudPolicy::new(true, false);
        assert!(policy.primary().is_none());
    }

    // 4. MultiCloudPolicy::primary – returns lowest priority number
    #[test]
    fn test_policy_primary_selection() {
        let mut policy = MultiCloudPolicy::new(true, false);
        policy.add_endpoint(make_ep(3));
        policy.add_endpoint(make_ep(1));
        policy.add_endpoint(make_ep(2));
        let primary = policy.primary().expect("primary should be valid");
        assert_eq!(primary.priority, 1);
    }

    // 5. MultiCloudPolicy::sorted_by_priority
    #[test]
    fn test_policy_sorted_by_priority() {
        let mut policy = MultiCloudPolicy::new(false, true);
        policy.add_endpoint(make_ep(5));
        policy.add_endpoint(make_ep(1));
        policy.add_endpoint(make_ep(3));
        let sorted = policy.sorted_by_priority();
        assert_eq!(sorted[0].priority, 1);
        assert_eq!(sorted[1].priority, 3);
        assert_eq!(sorted[2].priority, 5);
    }

    // 6. ReplicationTarget::identifier
    #[test]
    fn test_replication_target_identifier() {
        let t = ReplicationTarget::new("aws", "us-east-1", "my-bucket");
        assert_eq!(t.identifier(), "aws:us-east-1:my-bucket");
    }

    // 7. MultiCloudReplication::target_count – empty
    #[test]
    fn test_replication_target_count_empty() {
        let src = ReplicationTarget::new("aws", "us-east-1", "src-bucket");
        let rep = MultiCloudReplication::new(src);
        assert_eq!(rep.target_count(), 0);
    }

    // 8. MultiCloudReplication::add_target and target_count
    #[test]
    fn test_replication_add_targets() {
        let src = ReplicationTarget::new("aws", "us-east-1", "src");
        let mut rep = MultiCloudReplication::new(src);
        rep.add_target(ReplicationTarget::new("azure", "eastus", "replica-1"));
        rep.add_target(ReplicationTarget::new("gcp", "us-central1", "replica-2"));
        assert_eq!(rep.target_count(), 2);
    }

    // 9. MultiCloudReplication::all_targets
    #[test]
    fn test_replication_all_targets() {
        let src = ReplicationTarget::new("aws", "us-east-1", "src");
        let mut rep = MultiCloudReplication::new(src);
        rep.add_target(ReplicationTarget::new("azure", "eastus", "rep-a"));
        rep.add_target(ReplicationTarget::new("gcp", "us-central1", "rep-b"));
        let targets = rep.all_targets();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].bucket, "rep-a");
        assert_eq!(targets[1].bucket, "rep-b");
    }

    // 10. CloudHealthCheck::is_healthy – within timeout
    #[test]
    fn test_health_check_healthy() {
        let hc = CloudHealthCheck::new("https://endpoint.com", 1_000_000);
        // now is only 5_000ms after last_ok, well within 30_000ms timeout
        assert!(hc.is_healthy(1_005_000, 30_000));
    }

    // 11. CloudHealthCheck::is_healthy – timed out
    #[test]
    fn test_health_check_timed_out() {
        let hc = CloudHealthCheck::new("https://endpoint.com", 0);
        // 60 seconds later, timeout is 30 seconds
        assert!(!hc.is_healthy(60_000, 30_000));
    }

    // 12. CloudHealthCheck::record_failure sets consecutive_failures
    #[test]
    fn test_health_check_record_failure() {
        let mut hc = CloudHealthCheck::new("https://endpoint.com", 1_000_000);
        hc.record_failure();
        hc.record_failure();
        assert_eq!(hc.consecutive_failures, 2);
        // is_healthy returns false when consecutive_failures > 0
        assert!(!hc.is_healthy(1_000_001, 100_000));
    }

    // 13. CloudHealthCheck::record_success resets failures
    #[test]
    fn test_health_check_record_success() {
        let mut hc = CloudHealthCheck::new("https://endpoint.com", 0);
        hc.record_failure();
        hc.record_failure();
        hc.record_success(1_000_000);
        assert_eq!(hc.consecutive_failures, 0);
        assert!(hc.is_healthy(1_000_000, 1_000));
    }

    // 14. MultiCloudPolicy::sorted_by_priority – single endpoint
    #[test]
    fn test_policy_sorted_single() {
        let mut policy = MultiCloudPolicy::new(true, true);
        policy.add_endpoint(make_ep(7));
        let sorted = policy.sorted_by_priority();
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0].priority, 7);
    }

    // 15. ReplicationTarget – source is accessible via field
    #[test]
    fn test_replication_source_accessible() {
        let src = ReplicationTarget::new("alibaba", "cn-hangzhou", "my-src");
        let rep = MultiCloudReplication::new(src);
        assert_eq!(rep.source.provider, "alibaba");
        assert_eq!(rep.source.region, "cn-hangzhou");
        assert_eq!(rep.source.bucket, "my-src");
    }
}
