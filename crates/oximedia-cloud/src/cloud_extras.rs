//! Additional cloud utility APIs implementing TODO items.
//!
//! This module provides:
//! - `BandwidthThrottler` — token-bucket rate limiting for transfer bandwidth
//! - `CostEstimator` (simple) — monthly storage + egress cost by provider
//! - `LifecyclePolicy` / `LifecyclePolicyEngine` — apply lifecycle transitions
//! - `CloudEventBridge` — subscribe/emit event handlers
//! - `MultiRegionReplicator` — replicate object key to multiple target regions
//! - `AutoScaler` — compute desired capacity from queue depth
//! - `CloudBackupPolicy` — retention + frequency, backup-due check
//! - `MulticloudTransfer` — sync objects across cloud providers (stub)

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// BandwidthThrottler
// ---------------------------------------------------------------------------

/// Simple token-bucket bandwidth throttler.
///
/// Callers report how many bytes were transferred and how long the transfer
/// took; `throttle` returns the number of milliseconds they should sleep to
/// stay within the configured rate limit.
#[derive(Debug, Clone)]
pub struct BandwidthThrottler {
    /// Maximum bytes per second allowed.
    max_bytes_per_sec: u64,
    /// Accumulated "debt" in bytes that needs to be paid off by sleeping.
    debt_bytes: i64,
}

impl BandwidthThrottler {
    /// Creates a new throttler limited to `max_bytes_per_sec`.
    #[must_use]
    pub fn new(max_bytes_per_sec: u64) -> Self {
        Self {
            max_bytes_per_sec,
            debt_bytes: 0,
        }
    }

    /// Maximum bytes per second.
    #[must_use]
    pub fn max_bytes_per_sec(&self) -> u64 {
        self.max_bytes_per_sec
    }

    /// Report `bytes_transferred` over `elapsed_ms` milliseconds.
    ///
    /// Returns the number of milliseconds the caller should sleep before
    /// proceeding with the next chunk, so the long-term average rate stays
    /// at or below `max_bytes_per_sec`.  Returns 0 if no sleep is required.
    pub fn throttle(&mut self, bytes_transferred: u64, elapsed_ms: u64) -> u64 {
        if self.max_bytes_per_sec == 0 {
            return 0;
        }
        // Bytes allowed during the elapsed window
        let allowed = self.max_bytes_per_sec.saturating_mul(elapsed_ms) / 1_000;
        let excess = bytes_transferred as i64 - allowed as i64;
        self.debt_bytes = self.debt_bytes.saturating_add(excess);
        if self.debt_bytes <= 0 {
            return 0;
        }
        // How many ms to sleep to drain the debt
        let sleep_ms = (self.debt_bytes as u64).saturating_mul(1_000) / self.max_bytes_per_sec;
        // Drain the debt we are sleeping off
        let drained = self.max_bytes_per_sec.saturating_mul(sleep_ms) / 1_000;
        self.debt_bytes -= drained as i64;
        if self.debt_bytes < 0 {
            self.debt_bytes = 0;
        }
        sleep_ms
    }

    /// Resets the accumulated debt (e.g. after a period of inactivity).
    pub fn reset(&mut self) {
        self.debt_bytes = 0;
    }
}

// ---------------------------------------------------------------------------
// CostEstimator (simple per-provider)
// ---------------------------------------------------------------------------

/// Cloud provider for cost estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostProvider {
    /// Amazon S3.
    S3,
    /// Microsoft Azure Blob Storage.
    Azure,
    /// Google Cloud Storage.
    Gcs,
}

impl CostProvider {
    /// Storage cost in USD per GB per month.
    #[must_use]
    pub fn storage_usd_per_gb_month(self) -> f64 {
        match self {
            Self::S3 => 0.023,
            Self::Azure => 0.018,
            Self::Gcs => 0.020,
        }
    }

    /// Egress cost in USD per GB.
    #[must_use]
    pub fn egress_usd_per_gb(self) -> f64 {
        match self {
            Self::S3 => 0.09,
            Self::Azure => 0.087,
            Self::Gcs => 0.08,
        }
    }
}

/// Simple cloud storage cost estimator.
#[derive(Debug, Clone)]
pub struct SimpleCloudCostEstimator;

impl SimpleCloudCostEstimator {
    /// Estimate the total monthly cost for `storage_gb` stored and `egress_gb`
    /// egressed from `provider`.
    ///
    /// Returns the total USD cost for the month.
    #[must_use]
    pub fn estimate_storage_monthly(provider: CostProvider, storage_gb: f64, egress_gb: f64) -> f64 {
        let storage_cost = storage_gb * provider.storage_usd_per_gb_month();
        let egress_cost = egress_gb * provider.egress_usd_per_gb();
        storage_cost + egress_cost
    }
}

// ---------------------------------------------------------------------------
// LifecyclePolicy / LifecyclePolicyEngine
// ---------------------------------------------------------------------------

/// Target storage class after lifecycle transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleStorageClass {
    /// Infrequent-access / cool tier.
    InfrequentAccess,
    /// Archival / glacier tier.
    Archive,
    /// Deep archive / cold tier.
    DeepArchive,
    /// Deleted.
    Deleted,
}

impl std::fmt::Display for LifecycleStorageClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InfrequentAccess => write!(f, "infrequent-access"),
            Self::Archive => write!(f, "archive"),
            Self::DeepArchive => write!(f, "deep-archive"),
            Self::Deleted => write!(f, "deleted"),
        }
    }
}

/// A single lifecycle policy rule.
#[derive(Debug, Clone)]
pub struct LifecyclePolicy {
    /// Transition the object to `storage_class` after this many days.
    pub transition_days: u32,
    /// Delete the object after this many days (0 = no deletion).
    pub delete_days: u32,
    /// The storage class to transition into.
    pub storage_class: LifecycleStorageClass,
}

impl LifecyclePolicy {
    /// Creates a new lifecycle policy.
    #[must_use]
    pub fn new(
        transition_days: u32,
        delete_days: u32,
        storage_class: LifecycleStorageClass,
    ) -> Self {
        Self {
            transition_days,
            delete_days,
            storage_class,
        }
    }
}

/// Outcome of applying a lifecycle policy to an object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleAction {
    /// No change required.
    NoChange,
    /// Object should be moved to this storage class.
    Transition(LifecycleStorageClass),
    /// Object should be deleted.
    Delete,
}

/// Engine that applies lifecycle policies to objects based on their age.
#[derive(Debug, Default, Clone)]
pub struct LifecyclePolicyEngine {
    policies: Vec<LifecyclePolicy>,
}

impl LifecyclePolicyEngine {
    /// Creates a new engine with no policies.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a policy to the engine.
    pub fn add_policy(&mut self, policy: LifecyclePolicy) {
        self.policies.push(policy);
    }

    /// Applies all registered policies to an object that is `age_days` old.
    ///
    /// Returns the highest-priority action (`Delete` > `Transition` > `NoChange`).
    #[must_use]
    pub fn apply(&self, age_days: u32) -> LifecycleAction {
        let mut action = LifecycleAction::NoChange;
        for policy in &self.policies {
            if policy.delete_days > 0 && age_days >= policy.delete_days {
                return LifecycleAction::Delete;
            }
            if age_days >= policy.transition_days {
                action = LifecycleAction::Transition(policy.storage_class);
            }
        }
        action
    }
}

// ---------------------------------------------------------------------------
// CloudEventBridge (subscribe / emit)
// ---------------------------------------------------------------------------

/// A cloud event type string (e.g. "object.created", "transcode.completed").
pub type CloudEventType = String;

/// A cloud event payload.
#[derive(Debug, Clone)]
pub struct SimpleCloudEvent {
    /// Event type identifier.
    pub event_type: CloudEventType,
    /// Free-form JSON-like payload as key-value pairs.
    pub payload: std::collections::HashMap<String, String>,
}

impl SimpleCloudEvent {
    /// Creates a new event with an empty payload.
    #[must_use]
    pub fn new(event_type: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            payload: std::collections::HashMap::new(),
        }
    }

    /// Adds a key-value pair to the payload.
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.payload.insert(key.into(), value.into());
        self
    }
}

/// A boxed event handler function.
type EventHandler = Box<dyn Fn(&SimpleCloudEvent) + Send + Sync>;

/// Simple in-process event bridge that dispatches events to registered handlers.
pub struct CloudEventBridge {
    /// Handlers keyed by event type.
    handlers: std::collections::HashMap<CloudEventType, Vec<EventHandler>>,
    /// Events that have been emitted (for inspection in tests).
    emitted: Vec<SimpleCloudEvent>,
}

impl CloudEventBridge {
    /// Creates a new empty event bridge.
    #[must_use]
    pub fn new() -> Self {
        Self {
            handlers: std::collections::HashMap::new(),
            emitted: Vec::new(),
        }
    }

    /// Subscribes `handler` to events whose type equals `event_type`.
    pub fn subscribe<F>(&mut self, event_type: impl Into<String>, handler: F)
    where
        F: Fn(&SimpleCloudEvent) + Send + Sync + 'static,
    {
        self.handlers
            .entry(event_type.into())
            .or_default()
            .push(Box::new(handler));
    }

    /// Emits `event`, invoking all handlers registered for its type.
    pub fn emit(&mut self, event: SimpleCloudEvent) {
        if let Some(handlers) = self.handlers.get(&event.event_type) {
            for handler in handlers {
                handler(&event);
            }
        }
        self.emitted.push(event);
    }

    /// Returns the list of all emitted events (oldest first).
    #[must_use]
    pub fn emitted_events(&self) -> &[SimpleCloudEvent] {
        &self.emitted
    }

    /// Returns the count of events emitted for a specific type.
    #[must_use]
    pub fn count_emitted(&self, event_type: &str) -> usize {
        self.emitted
            .iter()
            .filter(|e| e.event_type == event_type)
            .count()
    }
}

impl Default for CloudEventBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MultiRegionReplicator
// ---------------------------------------------------------------------------

/// Outcome of a single replication attempt.
#[derive(Debug, Clone)]
pub struct ReplicationResult {
    /// Target region identifier.
    pub region: String,
    /// Whether the replication succeeded.
    pub success: bool,
    /// Error message on failure.
    pub error: Option<String>,
}

impl ReplicationResult {
    /// Constructs a successful replication result.
    #[must_use]
    pub fn ok(region: impl Into<String>) -> Self {
        Self {
            region: region.into(),
            success: true,
            error: None,
        }
    }

    /// Constructs a failed replication result.
    #[must_use]
    pub fn err(region: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            region: region.into(),
            success: false,
            error: Some(error.into()),
        }
    }
}

/// Multi-region object replicator.
///
/// This is a logical/planning layer; the actual I/O is expected to be
/// performed by callers using the `CloudStorage` trait.  The replicator
/// determines *which* regions to target and records the outcomes.
#[derive(Debug, Default, Clone)]
pub struct MultiRegionReplicator {
    /// Simulated latency overrides for testing (region_id → latency_ms).
    latency_overrides: std::collections::HashMap<String, u32>,
    /// Simulated failure regions (for testing).
    failure_regions: std::collections::HashSet<String>,
}

impl MultiRegionReplicator {
    /// Creates a new replicator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks a region as unavailable (for testing / simulation).
    pub fn mark_region_unavailable(&mut self, region: impl Into<String>) {
        self.failure_regions.insert(region.into());
    }

    /// Clears all simulated failures.
    pub fn clear_failures(&mut self) {
        self.failure_regions.clear();
    }

    /// Replicates `object_key` from `source_region` to all `target_regions`.
    ///
    /// Returns one `ReplicationResult` per target region.  In production this
    /// would trigger actual storage copy operations; here we model the logic
    /// (success/failure determination) without performing I/O.
    #[must_use]
    pub fn replicate(
        &self,
        object_key: &str,
        source_region: &str,
        target_regions: &[&str],
    ) -> Vec<ReplicationResult> {
        target_regions
            .iter()
            .map(|&region| {
                if self.failure_regions.contains(region) {
                    ReplicationResult::err(
                        region,
                        format!("region {region} is unavailable"),
                    )
                } else if region == source_region {
                    ReplicationResult::err(
                        region,
                        "target region is the same as source region".to_string(),
                    )
                } else {
                    ReplicationResult::ok(format!("{region}/{object_key}"))
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// AutoScaler — compute desired capacity
// ---------------------------------------------------------------------------

/// Stateless auto-scaler helper.
pub struct SimpleAutoScaler;

impl SimpleAutoScaler {
    /// Computes the desired number of workers based on the current queue depth.
    ///
    /// The target is `ceil(queue_depth / target_depth_per_worker)`, clamped to
    /// `[min_workers, max_workers]`.
    #[must_use]
    pub fn compute_desired_capacity(
        queue_depth: u32,
        target_depth_per_worker: u32,
        current: u32,
        min_workers: u32,
        max_workers: u32,
    ) -> u32 {
        let target = queue_depth
            .checked_div(target_depth_per_worker)
            .map(|q| {
                // Ceiling division: add 1 when there's a remainder.
                let ceiling = if queue_depth % target_depth_per_worker == 0 {
                    q
                } else {
                    q.saturating_add(1)
                };
                ceiling.max(1)
            })
            .unwrap_or(current);
        target.clamp(min_workers, max_workers)
    }
}

// ---------------------------------------------------------------------------
// CloudBackupPolicy
// ---------------------------------------------------------------------------

/// Cloud backup policy: retention window + backup frequency.
#[derive(Debug, Clone)]
pub struct CloudBackupPolicy {
    /// How many days to retain backups.
    pub retention_days: u32,
    /// How frequently to create backups, in hours.
    pub backup_frequency_hours: u32,
}

impl CloudBackupPolicy {
    /// Creates a new backup policy.
    #[must_use]
    pub fn new(retention_days: u32, backup_frequency_hours: u32) -> Self {
        Self {
            retention_days,
            backup_frequency_hours,
        }
    }

    /// Returns `true` if a backup is due.
    ///
    /// `last_backup_ts` and `now_ts` are Unix epoch seconds.
    /// Returns `true` when `now_ts - last_backup_ts >= backup_frequency_hours * 3600`
    /// or when no backup has been performed yet (`last_backup_ts == 0`).
    #[must_use]
    pub fn is_backup_due(&self, last_backup_ts: u64, now_ts: u64) -> bool {
        if last_backup_ts == 0 {
            return true;
        }
        let interval_secs = u64::from(self.backup_frequency_hours) * 3600;
        now_ts.saturating_sub(last_backup_ts) >= interval_secs
    }

    /// Returns `true` if a backup taken at `backup_ts` has expired and should
    /// be purged (`now_ts - backup_ts > retention_days * 86400`).
    #[must_use]
    pub fn is_expired(&self, backup_ts: u64, now_ts: u64) -> bool {
        let retention_secs = u64::from(self.retention_days) * 86_400;
        now_ts.saturating_sub(backup_ts) > retention_secs
    }
}

// ---------------------------------------------------------------------------
// MulticloudTransfer (sync stub)
// ---------------------------------------------------------------------------

/// Source/destination descriptor for a multicloud transfer.
#[derive(Debug, Clone)]
pub struct CloudEndpoint {
    /// Provider name (e.g. "s3", "azure", "gcs").
    pub provider: String,
    /// Bucket or container name.
    pub bucket: String,
    /// Optional key prefix to restrict the sync.
    pub prefix: Option<String>,
}

impl CloudEndpoint {
    /// Creates a new endpoint.
    #[must_use]
    pub fn new(provider: impl Into<String>, bucket: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            bucket: bucket.into(),
            prefix: None,
        }
    }

    /// Sets the key prefix.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }
}

/// Result of a multicloud sync operation.
#[derive(Debug, Clone, Default)]
pub struct SyncResult {
    /// Number of objects successfully transferred.
    pub transferred: u64,
    /// Number of objects skipped (already identical at the destination).
    pub skipped: u64,
    /// Number of objects that failed to transfer.
    pub failed: u64,
    /// Total bytes transferred.
    pub bytes_transferred: u64,
}

/// Multicloud object transfer / sync utility.
pub struct MulticloudTransfer;

impl MulticloudTransfer {
    /// Synchronises objects from `source` to `destination`.
    ///
    /// This is a planning/accounting stub — callers supply the list of
    /// `(key, size_bytes, already_at_dest)` tuples; the method returns a
    /// `SyncResult` reflecting what would happen.
    #[must_use]
    pub fn sync(
        _source: &CloudEndpoint,
        _destination: &CloudEndpoint,
        objects: &[(&str, u64, bool)],
    ) -> SyncResult {
        let mut result = SyncResult::default();
        for (_key, size, already_at_dest) in objects {
            if *already_at_dest {
                result.skipped += 1;
            } else {
                result.transferred += 1;
                result.bytes_transferred += size;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- BandwidthThrottler ---

    #[test]
    fn test_throttler_no_sleep_within_limit() {
        let mut t = BandwidthThrottler::new(1_000_000); // 1 MB/s
        // Transfer 500 KB in 1 000 ms — well within budget.
        let sleep = t.throttle(500_000, 1_000);
        assert_eq!(sleep, 0, "should not sleep when within limit");
    }

    #[test]
    fn test_throttler_sleep_when_over_limit() {
        let mut t = BandwidthThrottler::new(1_000); // 1 KB/s
        // Transfer 2 KB in 0 ms — double the allowed rate.
        let sleep = t.throttle(2_000, 0);
        assert!(sleep > 0, "should sleep when over limit");
    }

    #[test]
    fn test_throttler_zero_limit_no_sleep() {
        let mut t = BandwidthThrottler::new(0);
        let sleep = t.throttle(1_000_000, 0);
        assert_eq!(sleep, 0);
    }

    #[test]
    fn test_throttler_reset() {
        let mut t = BandwidthThrottler::new(100);
        t.throttle(10_000, 0); // build up debt
        t.reset();
        let sleep = t.throttle(50, 500); // well within limit after reset
        assert_eq!(sleep, 0);
    }

    #[test]
    fn test_throttler_max_bytes_per_sec() {
        let t = BandwidthThrottler::new(12_345);
        assert_eq!(t.max_bytes_per_sec(), 12_345);
    }

    // --- SimpleCloudCostEstimator ---

    #[test]
    fn test_cost_s3_storage_only() {
        let cost = SimpleCloudCostEstimator::estimate_storage_monthly(
            CostProvider::S3,
            100.0,
            0.0,
        );
        // 100 GB × $0.023 = $2.30
        assert!((cost - 2.30).abs() < 1e-6);
    }

    #[test]
    fn test_cost_azure_egress() {
        let cost = SimpleCloudCostEstimator::estimate_storage_monthly(
            CostProvider::Azure,
            0.0,
            10.0,
        );
        // 10 GB × $0.087 = $0.87
        assert!((cost - 0.87).abs() < 1e-6);
    }

    #[test]
    fn test_cost_gcs_combined() {
        let cost = SimpleCloudCostEstimator::estimate_storage_monthly(
            CostProvider::Gcs,
            50.0,
            5.0,
        );
        // 50 × 0.020 + 5 × 0.08 = 1.00 + 0.40 = 1.40
        assert!((cost - 1.40).abs() < 1e-6);
    }

    #[test]
    fn test_cost_provider_rates() {
        assert!((CostProvider::S3.storage_usd_per_gb_month() - 0.023).abs() < 1e-9);
        assert!((CostProvider::Azure.egress_usd_per_gb() - 0.087).abs() < 1e-9);
        assert!((CostProvider::Gcs.egress_usd_per_gb() - 0.08).abs() < 1e-9);
    }

    // --- LifecyclePolicyEngine ---

    #[test]
    fn test_lifecycle_no_policies_no_change() {
        let engine = LifecyclePolicyEngine::new();
        assert_eq!(engine.apply(365), LifecycleAction::NoChange);
    }

    #[test]
    fn test_lifecycle_transition() {
        let mut engine = LifecyclePolicyEngine::new();
        engine.add_policy(LifecyclePolicy::new(30, 0, LifecycleStorageClass::InfrequentAccess));
        assert_eq!(engine.apply(15), LifecycleAction::NoChange);
        assert_eq!(
            engine.apply(30),
            LifecycleAction::Transition(LifecycleStorageClass::InfrequentAccess)
        );
    }

    #[test]
    fn test_lifecycle_delete_takes_priority() {
        let mut engine = LifecyclePolicyEngine::new();
        engine.add_policy(LifecyclePolicy::new(30, 365, LifecycleStorageClass::Archive));
        assert_eq!(engine.apply(365), LifecycleAction::Delete);
    }

    #[test]
    fn test_lifecycle_storage_class_display() {
        assert_eq!(LifecycleStorageClass::Archive.to_string(), "archive");
        assert_eq!(LifecycleStorageClass::Deleted.to_string(), "deleted");
    }

    // --- CloudEventBridge ---

    #[test]
    fn test_event_bridge_subscribe_emit() {
        use std::sync::{Arc, Mutex};
        let counter = Arc::new(Mutex::new(0u32));
        let counter_clone = Arc::clone(&counter);

        let mut bridge = CloudEventBridge::new();
        bridge.subscribe("object.created", move |_event| {
            let mut c = counter_clone.lock().unwrap_or_else(|e| e.into_inner());
            *c += 1;
        });

        bridge.emit(SimpleCloudEvent::new("object.created"));
        bridge.emit(SimpleCloudEvent::new("object.created"));
        bridge.emit(SimpleCloudEvent::new("object.deleted"));

        let count = *counter.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(count, 2);
    }

    #[test]
    fn test_event_bridge_count_emitted() {
        let mut bridge = CloudEventBridge::new();
        bridge.emit(SimpleCloudEvent::new("a"));
        bridge.emit(SimpleCloudEvent::new("a"));
        bridge.emit(SimpleCloudEvent::new("b"));
        assert_eq!(bridge.count_emitted("a"), 2);
        assert_eq!(bridge.count_emitted("b"), 1);
        assert_eq!(bridge.count_emitted("c"), 0);
    }

    #[test]
    fn test_event_bridge_no_handler_does_not_panic() {
        let mut bridge = CloudEventBridge::new();
        bridge.emit(SimpleCloudEvent::new("unknown.event"));
        assert_eq!(bridge.emitted_events().len(), 1);
    }

    #[test]
    fn test_event_payload() {
        let event = SimpleCloudEvent::new("test")
            .with_field("key", "value")
            .with_field("size", "1024");
        assert_eq!(event.payload.get("key").map(|s| s.as_str()), Some("value"));
        assert_eq!(event.payload.get("size").map(|s| s.as_str()), Some("1024"));
    }

    // --- MultiRegionReplicator ---

    #[test]
    fn test_replicator_success() {
        let r = MultiRegionReplicator::new();
        let results = r.replicate("video.mp4", "us-east-1", &["eu-west-1", "ap-southeast-1"]);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.success));
    }

    #[test]
    fn test_replicator_same_region_fails() {
        let r = MultiRegionReplicator::new();
        let results = r.replicate("video.mp4", "us-east-1", &["us-east-1"]);
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
    }

    #[test]
    fn test_replicator_failure_region() {
        let mut r = MultiRegionReplicator::new();
        r.mark_region_unavailable("eu-west-1");
        let results = r.replicate("video.mp4", "us-east-1", &["eu-west-1", "us-west-2"]);
        assert!(!results[0].success);
        assert!(results[1].success);
        r.clear_failures();
        let results2 = r.replicate("video.mp4", "us-east-1", &["eu-west-1"]);
        assert!(results2[0].success);
    }

    // --- SimpleAutoScaler ---

    #[test]
    fn test_autoscaler_scale_up() {
        let desired = SimpleAutoScaler::compute_desired_capacity(100, 10, 2, 1, 20);
        assert_eq!(desired, 10);
    }

    #[test]
    fn test_autoscaler_clamp_min() {
        let desired = SimpleAutoScaler::compute_desired_capacity(0, 10, 5, 2, 20);
        assert_eq!(desired, 2); // clamped to min
    }

    #[test]
    fn test_autoscaler_clamp_max() {
        let desired = SimpleAutoScaler::compute_desired_capacity(1000, 10, 5, 1, 20);
        assert_eq!(desired, 20); // clamped to max
    }

    #[test]
    fn test_autoscaler_zero_target_depth() {
        let desired = SimpleAutoScaler::compute_desired_capacity(50, 0, 3, 1, 10);
        assert_eq!(desired, 3); // returns current when divisor is 0
    }

    // --- CloudBackupPolicy ---

    #[test]
    fn test_backup_policy_due_no_prior_backup() {
        let policy = CloudBackupPolicy::new(30, 24);
        assert!(policy.is_backup_due(0, 1_000));
    }

    #[test]
    fn test_backup_policy_due_elapsed() {
        let policy = CloudBackupPolicy::new(30, 24);
        let last = 1_000_000u64;
        let now = last + 24 * 3600; // exactly one interval later
        assert!(policy.is_backup_due(last, now));
    }

    #[test]
    fn test_backup_policy_not_due() {
        let policy = CloudBackupPolicy::new(30, 24);
        let last = 1_000_000u64;
        let now = last + 12 * 3600; // half an interval
        assert!(!policy.is_backup_due(last, now));
    }

    #[test]
    fn test_backup_policy_expired() {
        let policy = CloudBackupPolicy::new(7, 24);
        let backup_ts = 1_000_000u64;
        let now = backup_ts + 8 * 86_400; // 8 days > 7 days retention
        assert!(policy.is_expired(backup_ts, now));
    }

    #[test]
    fn test_backup_policy_not_expired() {
        let policy = CloudBackupPolicy::new(30, 24);
        let backup_ts = 1_000_000u64;
        let now = backup_ts + 5 * 86_400;
        assert!(!policy.is_expired(backup_ts, now));
    }

    // --- MulticloudTransfer ---

    #[test]
    fn test_multicloud_sync_all_new() {
        let src = CloudEndpoint::new("s3", "src-bucket");
        let dst = CloudEndpoint::new("gcs", "dst-bucket");
        let objects = vec![("a.mp4", 1024u64, false), ("b.mp4", 2048, false)];
        let result = MulticloudTransfer::sync(&src, &dst, &objects);
        assert_eq!(result.transferred, 2);
        assert_eq!(result.skipped, 0);
        assert_eq!(result.bytes_transferred, 3072);
    }

    #[test]
    fn test_multicloud_sync_some_skipped() {
        let src = CloudEndpoint::new("azure", "src");
        let dst = CloudEndpoint::new("s3", "dst");
        let objects = vec![("a.mp4", 1000u64, true), ("b.mp4", 2000, false)];
        let result = MulticloudTransfer::sync(&src, &dst, &objects);
        assert_eq!(result.transferred, 1);
        assert_eq!(result.skipped, 1);
        assert_eq!(result.bytes_transferred, 2000);
    }

    #[test]
    fn test_multicloud_endpoint_prefix() {
        let ep = CloudEndpoint::new("gcs", "bucket").with_prefix("media/");
        assert_eq!(ep.prefix.as_deref(), Some("media/"));
    }
}
