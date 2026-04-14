//! Advanced replication policy management for OxiMedia storage.
//!
//! Provides sync policy configuration, replication lag tracking,
//! consistency level enforcement, and multi-region replication state.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Replication consistency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsistencyLevel {
    /// Write succeeds as soon as the primary acknowledges
    Eventual,
    /// A quorum of replicas must acknowledge before success
    Quorum,
    /// All replicas must acknowledge before success
    All,
    /// At least one remote replica must acknowledge
    OneRemote,
}

impl ConsistencyLevel {
    /// Returns a human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            ConsistencyLevel::Eventual => "eventual",
            ConsistencyLevel::Quorum => "quorum",
            ConsistencyLevel::All => "all",
            ConsistencyLevel::OneRemote => "one_remote",
        }
    }

    /// Returns the minimum number of acknowledgements required for n replicas
    pub fn required_acks(&self, total_replicas: usize) -> usize {
        match self {
            ConsistencyLevel::Eventual => 1,
            ConsistencyLevel::OneRemote => 2.min(total_replicas),
            ConsistencyLevel::Quorum => (total_replicas / 2) + 1,
            ConsistencyLevel::All => total_replicas,
        }
    }
}

/// Replication sync strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// Replicate immediately, blocking until consistency is satisfied
    Synchronous,
    /// Replicate in the background after primary ack
    Asynchronous,
    /// Replicate on a fixed schedule
    Periodic,
    /// Replicate only when explicitly triggered
    OnDemand,
}

/// Represents a single replication site
#[derive(Debug, Clone)]
pub struct ReplicationSite {
    /// Unique site identifier
    pub site_id: String,
    /// Human-readable region name
    pub region: String,
    /// Site priority (lower = higher priority)
    pub priority: u8,
    /// Sync strategy for this site
    pub strategy: SyncStrategy,
    /// Consistency level required before reporting success
    pub consistency: ConsistencyLevel,
    /// Whether this site is currently active
    pub active: bool,
}

impl ReplicationSite {
    /// Creates a new active replication site
    pub fn new(
        site_id: impl Into<String>,
        region: impl Into<String>,
        priority: u8,
        strategy: SyncStrategy,
        consistency: ConsistencyLevel,
    ) -> Self {
        Self {
            site_id: site_id.into(),
            region: region.into(),
            priority,
            strategy,
            consistency,
            active: true,
        }
    }
}

/// Replication lag sample for a single object on a single site
#[derive(Debug, Clone)]
pub struct LagSample {
    /// Object key
    pub key: String,
    /// Site ID
    pub site_id: String,
    /// Measured replication lag
    pub lag: Duration,
    /// Timestamp when the sample was recorded
    pub recorded_at: Instant,
}

impl LagSample {
    /// Creates a new lag sample
    pub fn new(key: impl Into<String>, site_id: impl Into<String>, lag: Duration) -> Self {
        Self {
            key: key.into(),
            site_id: site_id.into(),
            lag,
            recorded_at: Instant::now(),
        }
    }

    /// Returns true if the lag exceeds the given threshold
    pub fn exceeds_threshold(&self, threshold: Duration) -> bool {
        self.lag > threshold
    }
}

/// State of object replication across sites
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationState {
    /// Object has not been replicated yet
    Pending,
    /// Replication is in progress
    InProgress,
    /// Object is fully replicated on all configured sites
    Complete,
    /// Replication failed on one or more sites
    Failed,
    /// Partial replication (some sites succeeded)
    Partial,
}

impl ReplicationState {
    /// Returns a human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            ReplicationState::Pending => "pending",
            ReplicationState::InProgress => "in_progress",
            ReplicationState::Complete => "complete",
            ReplicationState::Failed => "failed",
            ReplicationState::Partial => "partial",
        }
    }

    /// Returns true if the object is accessible on at least one replica
    pub fn is_available(&self) -> bool {
        matches!(self, ReplicationState::Complete | ReplicationState::Partial)
    }
}

/// Per-site replication status for a single object
#[derive(Debug, Clone)]
pub struct SiteReplicationStatus {
    /// Site ID
    pub site_id: String,
    /// Replication state on this site
    pub state: ReplicationState,
    /// When replication was last attempted
    pub last_attempt: Option<Instant>,
    /// When replication last succeeded
    pub last_success: Option<Instant>,
    /// Number of consecutive failures
    pub failure_count: u32,
    /// Last error message, if any
    pub last_error: Option<String>,
}

impl SiteReplicationStatus {
    /// Creates a new status in Pending state
    pub fn new(site_id: impl Into<String>) -> Self {
        Self {
            site_id: site_id.into(),
            state: ReplicationState::Pending,
            last_attempt: None,
            last_success: None,
            failure_count: 0,
            last_error: None,
        }
    }

    /// Records a successful replication
    pub fn record_success(&mut self) {
        self.state = ReplicationState::Complete;
        let now = Instant::now();
        self.last_attempt = Some(now);
        self.last_success = Some(now);
        self.failure_count = 0;
        self.last_error = None;
    }

    /// Records a replication failure
    pub fn record_failure(&mut self, error: impl Into<String>) {
        self.state = ReplicationState::Failed;
        self.last_attempt = Some(Instant::now());
        self.failure_count += 1;
        self.last_error = Some(error.into());
    }

    /// Returns true if this site has successfully replicated the object
    pub fn is_replicated(&self) -> bool {
        self.state == ReplicationState::Complete
    }
}

/// Replication policy for an object or prefix
#[derive(Debug, Clone)]
pub struct ReplicationPolicy {
    /// Policy ID
    pub id: String,
    /// Optional key prefix this policy applies to
    pub prefix: Option<String>,
    /// Replication sites
    pub sites: Vec<ReplicationSite>,
    /// Default consistency level for writes
    pub default_consistency: ConsistencyLevel,
    /// Maximum acceptable replication lag before alerting
    pub max_lag: Duration,
    /// Whether this policy is enabled
    pub enabled: bool,
}

impl ReplicationPolicy {
    /// Creates a new enabled replication policy
    pub fn new(id: impl Into<String>, consistency: ConsistencyLevel) -> Self {
        Self {
            id: id.into(),
            prefix: None,
            sites: Vec::new(),
            default_consistency: consistency,
            max_lag: Duration::from_mins(1),
            enabled: true,
        }
    }

    /// Adds a site to this policy
    pub fn add_site(mut self, site: ReplicationSite) -> Self {
        self.sites.push(site);
        self
    }

    /// Sets a key prefix filter
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Sets the maximum acceptable lag duration
    pub fn with_max_lag(mut self, max_lag: Duration) -> Self {
        self.max_lag = max_lag;
        self
    }

    /// Returns active sites only
    pub fn active_sites(&self) -> Vec<&ReplicationSite> {
        self.sites.iter().filter(|s| s.active).collect()
    }

    /// Returns the required number of acknowledgements based on consistency level
    pub fn required_acks(&self) -> usize {
        let n = self.active_sites().len().max(1);
        self.default_consistency.required_acks(n)
    }

    /// Returns true if the policy matches the given key
    pub fn matches_key(&self, key: &str) -> bool {
        match &self.prefix {
            Some(p) => key.starts_with(p.as_str()),
            None => true,
        }
    }
}

/// Tracks replication state across multiple sites for many objects
pub struct ReplicationTracker {
    /// policy ID → policy
    policies: Vec<ReplicationPolicy>,
    /// object key → site_id → status
    status: HashMap<String, HashMap<String, SiteReplicationStatus>>,
    /// lag samples
    lag_samples: Vec<LagSample>,
}

impl ReplicationTracker {
    /// Creates a new empty tracker
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
            status: HashMap::new(),
            lag_samples: Vec::new(),
        }
    }

    /// Registers a replication policy
    pub fn register_policy(&mut self, policy: ReplicationPolicy) {
        self.policies.push(policy);
    }

    /// Initialises tracking for the given object across all sites in matching policies
    pub fn track_object(&mut self, key: &str) {
        let site_ids: Vec<String> = self
            .policies
            .iter()
            .filter(|p| p.enabled && p.matches_key(key))
            .flat_map(|p| p.active_sites().into_iter().map(|s| s.site_id.clone()))
            .collect();

        let entry = self.status.entry(key.to_string()).or_default();
        for site_id in site_ids {
            entry
                .entry(site_id.clone())
                .or_insert_with(|| SiteReplicationStatus::new(site_id));
        }
    }

    /// Records a successful replication for the given object on the given site
    pub fn record_success(&mut self, key: &str, site_id: &str) {
        if let Some(sites) = self.status.get_mut(key) {
            if let Some(s) = sites.get_mut(site_id) {
                s.record_success();
            }
        }
    }

    /// Records a replication failure
    pub fn record_failure(&mut self, key: &str, site_id: &str, error: &str) {
        if let Some(sites) = self.status.get_mut(key) {
            if let Some(s) = sites.get_mut(site_id) {
                s.record_failure(error);
            }
        }
    }

    /// Adds a lag sample
    pub fn add_lag_sample(&mut self, sample: LagSample) {
        self.lag_samples.push(sample);
    }

    /// Returns the aggregate replication state for the given object
    pub fn aggregate_state(&self, key: &str) -> ReplicationState {
        let sites = match self.status.get(key) {
            Some(s) => s,
            None => return ReplicationState::Pending,
        };
        if sites.is_empty() {
            return ReplicationState::Pending;
        }
        let total = sites.len();
        let done = sites.values().filter(|s| s.is_replicated()).count();
        let failed = sites
            .values()
            .filter(|s| s.state == ReplicationState::Failed)
            .count();

        if done == total {
            ReplicationState::Complete
        } else if failed == total {
            ReplicationState::Failed
        } else if done > 0 {
            ReplicationState::Partial
        } else {
            ReplicationState::Pending
        }
    }

    /// Returns lag samples that exceed the given threshold
    pub fn lagging_objects(&self, threshold: Duration) -> Vec<&LagSample> {
        self.lag_samples
            .iter()
            .filter(|s| s.exceeds_threshold(threshold))
            .collect()
    }

    /// Returns the number of tracked objects
    pub fn tracked_object_count(&self) -> usize {
        self.status.len()
    }
}

impl Default for ReplicationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_site(id: &str, strategy: SyncStrategy) -> ReplicationSite {
        ReplicationSite::new(id, "us-east-1", 1, strategy, ConsistencyLevel::Eventual)
    }

    #[test]
    fn test_consistency_level_required_acks_eventual() {
        assert_eq!(ConsistencyLevel::Eventual.required_acks(3), 1);
    }

    #[test]
    fn test_consistency_level_required_acks_quorum() {
        assert_eq!(ConsistencyLevel::Quorum.required_acks(3), 2);
        assert_eq!(ConsistencyLevel::Quorum.required_acks(5), 3);
    }

    #[test]
    fn test_consistency_level_required_acks_all() {
        assert_eq!(ConsistencyLevel::All.required_acks(4), 4);
    }

    #[test]
    fn test_consistency_level_required_acks_one_remote() {
        assert_eq!(ConsistencyLevel::OneRemote.required_acks(1), 1);
        assert_eq!(ConsistencyLevel::OneRemote.required_acks(5), 2);
    }

    #[test]
    fn test_consistency_level_label() {
        assert_eq!(ConsistencyLevel::Quorum.label(), "quorum");
        assert_eq!(ConsistencyLevel::All.label(), "all");
    }

    #[test]
    fn test_lag_sample_threshold() {
        let sample = LagSample::new("file.mp4", "site-1", Duration::from_secs(30));
        assert!(sample.exceeds_threshold(Duration::from_secs(10)));
        assert!(!sample.exceeds_threshold(Duration::from_mins(1)));
    }

    #[test]
    fn test_replication_state_is_available() {
        assert!(ReplicationState::Complete.is_available());
        assert!(ReplicationState::Partial.is_available());
        assert!(!ReplicationState::Pending.is_available());
        assert!(!ReplicationState::Failed.is_available());
    }

    #[test]
    fn test_replication_state_label() {
        assert_eq!(ReplicationState::InProgress.label(), "in_progress");
        assert_eq!(ReplicationState::Failed.label(), "failed");
    }

    #[test]
    fn test_site_replication_status_success() {
        let mut s = SiteReplicationStatus::new("site-1");
        assert_eq!(s.state, ReplicationState::Pending);
        s.record_success();
        assert_eq!(s.state, ReplicationState::Complete);
        assert_eq!(s.failure_count, 0);
        assert!(s.is_replicated());
    }

    #[test]
    fn test_site_replication_status_failure() {
        let mut s = SiteReplicationStatus::new("site-2");
        s.record_failure("connection timeout");
        assert_eq!(s.state, ReplicationState::Failed);
        assert_eq!(s.failure_count, 1);
        assert_eq!(s.last_error.as_deref(), Some("connection timeout"));
    }

    #[test]
    fn test_replication_policy_required_acks() {
        let policy = ReplicationPolicy::new("p1", ConsistencyLevel::Quorum)
            .add_site(make_site("s1", SyncStrategy::Synchronous))
            .add_site(make_site("s2", SyncStrategy::Asynchronous))
            .add_site(make_site("s3", SyncStrategy::Asynchronous));
        assert_eq!(policy.required_acks(), 2); // quorum of 3
    }

    #[test]
    fn test_replication_policy_prefix_filter() {
        let policy = ReplicationPolicy::new("p1", ConsistencyLevel::Eventual).with_prefix("media/");
        assert!(policy.matches_key("media/video.mp4"));
        assert!(!policy.matches_key("logs/app.log"));
    }

    #[test]
    fn test_replication_tracker_track_and_aggregate() {
        let mut tracker = ReplicationTracker::new();
        let policy = ReplicationPolicy::new("p1", ConsistencyLevel::All)
            .add_site(make_site("s1", SyncStrategy::Synchronous))
            .add_site(make_site("s2", SyncStrategy::Asynchronous));
        tracker.register_policy(policy);

        tracker.track_object("file.mp4");
        assert_eq!(
            tracker.aggregate_state("file.mp4"),
            ReplicationState::Pending
        );

        tracker.record_success("file.mp4", "s1");
        assert_eq!(
            tracker.aggregate_state("file.mp4"),
            ReplicationState::Partial
        );

        tracker.record_success("file.mp4", "s2");
        assert_eq!(
            tracker.aggregate_state("file.mp4"),
            ReplicationState::Complete
        );
    }

    #[test]
    fn test_replication_tracker_failure_aggregate() {
        let mut tracker = ReplicationTracker::new();
        let policy = ReplicationPolicy::new("p1", ConsistencyLevel::All)
            .add_site(make_site("s1", SyncStrategy::Synchronous));
        tracker.register_policy(policy);

        tracker.track_object("obj.mp4");
        tracker.record_failure("obj.mp4", "s1", "network error");
        assert_eq!(tracker.aggregate_state("obj.mp4"), ReplicationState::Failed);
    }

    #[test]
    fn test_replication_tracker_lag_samples() {
        let mut tracker = ReplicationTracker::new();
        tracker.add_lag_sample(LagSample::new("a.mp4", "s1", Duration::from_secs(5)));
        tracker.add_lag_sample(LagSample::new("b.mp4", "s2", Duration::from_mins(2)));
        let lagging = tracker.lagging_objects(Duration::from_mins(1));
        assert_eq!(lagging.len(), 1);
        assert_eq!(lagging[0].key, "b.mp4");
    }

    #[test]
    fn test_replication_tracker_object_count() {
        let mut tracker = ReplicationTracker::new();
        let policy = ReplicationPolicy::new("p1", ConsistencyLevel::Eventual)
            .add_site(make_site("s1", SyncStrategy::Asynchronous));
        tracker.register_policy(policy);

        tracker.track_object("x.mp4");
        tracker.track_object("y.mp4");
        assert_eq!(tracker.tracked_object_count(), 2);
    }
}

// ─── Cross-provider replication ────────────────────────────────────────────────

use serde::{Deserialize, Serialize};

/// Identifies a cloud storage provider endpoint for cross-provider replication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderTarget {
    /// Amazon S3 bucket
    S3 {
        /// Bucket name
        bucket: String,
        /// AWS region
        region: String,
    },
    /// Google Cloud Storage bucket
    Gcs {
        /// Bucket name
        bucket: String,
    },
    /// Azure Blob Storage container
    Azure {
        /// Container name
        container: String,
    },
    /// Local filesystem path
    Local {
        /// Filesystem path
        path: String,
    },
}

impl ProviderTarget {
    /// Returns a human-readable description of this target.
    pub fn description(&self) -> String {
        match self {
            Self::S3 { bucket, region } => format!("s3://{bucket} ({region})"),
            Self::Gcs { bucket } => format!("gs://{bucket}"),
            Self::Azure { container } => format!("azure://{container}"),
            Self::Local { path } => format!("local:{path}"),
        }
    }

    /// Returns the provider name.
    pub fn provider_name(&self) -> &'static str {
        match self {
            Self::S3 { .. } => "S3",
            Self::Gcs { .. } => "GCS",
            Self::Azure { .. } => "Azure",
            Self::Local { .. } => "Local",
        }
    }
}

/// Result of replicating a single object to a single target.
#[derive(Debug, Clone)]
pub struct ReplicaResult {
    /// The target that was replicated to.
    pub target: ProviderTarget,
    /// Whether replication succeeded.
    pub success: bool,
    /// Error description if replication failed.
    pub error: Option<String>,
}

/// Summary report of a cross-provider replication operation.
#[derive(Debug, Clone)]
pub struct ReplicationReport {
    /// Object key that was replicated.
    pub key: String,
    /// Whether primary write succeeded.
    pub primary_ok: bool,
    /// Total number of replica targets.
    pub replica_count: usize,
    /// Number of replicas that succeeded.
    pub successful_replicas: usize,
    /// Names of failed replicas.
    pub failed_replicas: Vec<String>,
    /// SHA-256 checksum of the object data (hex-encoded).
    pub checksum_hex: String,
}

impl ReplicationReport {
    /// Returns true if all replicas succeeded.
    pub fn all_replicated(&self) -> bool {
        self.primary_ok && self.successful_replicas == self.replica_count
    }

    /// Returns the number of failed replicas.
    pub fn failure_count(&self) -> usize {
        self.failed_replicas.len()
    }
}

/// Policy for replicating objects across multiple cloud providers.
///
/// Simulates cross-provider replication (no real network calls).
/// Computes SHA-256 checksums and produces a `ReplicationReport` for each
/// replicate operation.
#[derive(Debug, Clone)]
pub struct CrossProviderReplicationPolicy {
    /// Primary storage target.
    pub primary: ProviderTarget,
    /// Replica targets (each receives a copy).
    pub replicas: Vec<ProviderTarget>,
    /// Whether to verify SHA-256 checksums after replication.
    pub verify_checksum: bool,
    /// Maximum simultaneous replications (for future async use).
    pub max_concurrent: usize,
}

impl CrossProviderReplicationPolicy {
    /// Create a new cross-provider replication policy.
    pub fn new(primary: ProviderTarget, replicas: Vec<ProviderTarget>) -> Self {
        Self {
            primary,
            replicas,
            verify_checksum: true,
            max_concurrent: 4,
        }
    }

    /// Builder: disable checksum verification.
    pub fn without_checksum_verification(mut self) -> Self {
        self.verify_checksum = false;
        self
    }

    /// Builder: set the maximum concurrent replication tasks.
    pub fn with_max_concurrent(mut self, n: usize) -> Self {
        self.max_concurrent = n;
        self
    }

    /// Simulate replicating `data` under `key` to all configured targets.
    ///
    /// No real network calls are made.  A SHA-256 checksum is computed for
    /// `data` and reported.  All replica results are simulated as successful
    /// unless a `Local` target with an invalid (empty) path is specified, in
    /// which case replication to that target is reported as failed.
    pub fn simulate_replicate(&self, key: &str, data: &[u8]) -> ReplicationReport {
        use sha2::{Digest, Sha256};
        let checksum: [u8; 32] = Sha256::digest(data).into();
        let checksum_hex = checksum
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();

        let mut successful_replicas = 0usize;
        let mut failed_replicas = Vec::new();

        for target in &self.replicas {
            // Simulate failure only for Local targets with an empty path.
            let ok = match target {
                ProviderTarget::Local { path } if path.is_empty() => false,
                _ => true,
            };

            if ok {
                successful_replicas += 1;
            } else {
                failed_replicas.push(target.description());
            }
        }

        ReplicationReport {
            key: key.to_string(),
            primary_ok: true,
            replica_count: self.replicas.len(),
            successful_replicas,
            failed_replicas,
            checksum_hex,
        }
    }

    /// Verify that `data` matches the expected SHA-256 `expected` digest.
    ///
    /// Returns `true` when the data's SHA-256 hash equals `expected`.
    pub fn verify_checksum(data: &[u8], expected: &[u8; 32]) -> bool {
        use sha2::{Digest, Sha256};
        let actual: [u8; 32] = Sha256::digest(data).into();
        actual == *expected
    }
}

#[cfg(test)]
mod cross_provider_tests {
    use super::*;

    fn s3_target() -> ProviderTarget {
        ProviderTarget::S3 {
            bucket: "my-bucket".to_string(),
            region: "us-east-1".to_string(),
        }
    }

    fn gcs_target() -> ProviderTarget {
        ProviderTarget::Gcs {
            bucket: "gcs-bucket".to_string(),
        }
    }

    fn azure_target() -> ProviderTarget {
        ProviderTarget::Azure {
            container: "az-container".to_string(),
        }
    }

    fn local_target() -> ProviderTarget {
        ProviderTarget::Local {
            path: "/tmp/replica".to_string(),
        }
    }

    #[test]
    fn test_provider_target_description_s3() {
        let t = s3_target();
        let desc = t.description();
        assert!(desc.contains("my-bucket"));
        assert!(desc.contains("us-east-1"));
    }

    #[test]
    fn test_provider_target_description_gcs() {
        assert!(gcs_target().description().contains("gcs-bucket"));
    }

    #[test]
    fn test_provider_target_description_azure() {
        assert!(azure_target().description().contains("az-container"));
    }

    #[test]
    fn test_provider_target_description_local() {
        assert!(local_target().description().contains("/tmp/replica"));
    }

    #[test]
    fn test_provider_target_provider_name() {
        assert_eq!(s3_target().provider_name(), "S3");
        assert_eq!(gcs_target().provider_name(), "GCS");
        assert_eq!(azure_target().provider_name(), "Azure");
        assert_eq!(local_target().provider_name(), "Local");
    }

    #[test]
    fn test_provider_target_serde_roundtrip() {
        let original = s3_target();
        let json = serde_json::to_string(&original).expect("serialize ProviderTarget");
        let back: ProviderTarget = serde_json::from_str(&json).expect("deserialize ProviderTarget");
        assert_eq!(original, back);
    }

    #[test]
    fn test_cross_provider_policy_new() {
        let policy =
            CrossProviderReplicationPolicy::new(s3_target(), vec![gcs_target(), azure_target()]);
        assert_eq!(policy.replicas.len(), 2);
        assert!(policy.verify_checksum);
        assert_eq!(policy.max_concurrent, 4);
    }

    #[test]
    fn test_cross_provider_policy_builders() {
        let policy = CrossProviderReplicationPolicy::new(s3_target(), vec![gcs_target()])
            .without_checksum_verification()
            .with_max_concurrent(8);
        assert!(!policy.verify_checksum);
        assert_eq!(policy.max_concurrent, 8);
    }

    #[test]
    fn test_simulate_replicate_all_success() {
        let policy = CrossProviderReplicationPolicy::new(
            s3_target(),
            vec![gcs_target(), azure_target(), local_target()],
        );
        let data = b"hello, oximedia cross-provider replication!";
        let report = policy.simulate_replicate("media/test.mp4", data);
        assert_eq!(report.key, "media/test.mp4");
        assert!(report.primary_ok);
        assert_eq!(report.replica_count, 3);
        assert_eq!(report.successful_replicas, 3);
        assert!(report.failed_replicas.is_empty());
        assert!(report.all_replicated());
        assert_eq!(report.failure_count(), 0);
        // Checksum should be 64 hex chars (SHA-256)
        assert_eq!(report.checksum_hex.len(), 64);
    }

    #[test]
    fn test_simulate_replicate_empty_local_path_fails() {
        let policy = CrossProviderReplicationPolicy::new(
            s3_target(),
            vec![
                gcs_target(),
                ProviderTarget::Local {
                    path: String::new(), // empty path → simulated failure
                },
            ],
        );
        let report = policy.simulate_replicate("file.bin", b"data");
        assert_eq!(report.replica_count, 2);
        assert_eq!(report.successful_replicas, 1);
        assert_eq!(report.failed_replicas.len(), 1);
        assert!(!report.all_replicated());
        assert_eq!(report.failure_count(), 1);
    }

    #[test]
    fn test_verify_checksum_match() {
        use sha2::{Digest, Sha256};
        let data = b"oximedia replication checksum verification";
        let expected: [u8; 32] = Sha256::digest(data).into();
        assert!(CrossProviderReplicationPolicy::verify_checksum(
            data, &expected
        ));
    }

    #[test]
    fn test_verify_checksum_mismatch() {
        let data = b"original data";
        let wrong_expected = [0u8; 32];
        assert!(!CrossProviderReplicationPolicy::verify_checksum(
            data,
            &wrong_expected
        ));
    }

    #[test]
    fn test_replication_report_checksum_consistency() {
        use sha2::{Digest, Sha256};
        let data = b"deterministic checksum test";
        let policy = CrossProviderReplicationPolicy::new(s3_target(), vec![gcs_target()]);
        let report = policy.simulate_replicate("obj", data);
        let expected: [u8; 32] = Sha256::digest(data).into();
        let expected_hex: String = expected.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(report.checksum_hex, expected_hex);
    }
}
