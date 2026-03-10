#![allow(dead_code)]
//! Storage tiering and lifecycle management for archived media.
//!
//! Provides multi-tier storage management where media assets are automatically
//! moved between tiers (hot, warm, cold, deep archive) based on access patterns,
//! age, and configurable policies.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Represents a storage tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    /// Frequently accessed, low-latency storage (SSD/NVMe).
    Hot,
    /// Moderately accessed storage (HDD/SAN).
    Warm,
    /// Infrequently accessed storage (tape library, nearline).
    Cold,
    /// Rarely accessed, highest latency (deep archive, glacier).
    DeepArchive,
}

impl fmt::Display for StorageTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hot => write!(f, "Hot"),
            Self::Warm => write!(f, "Warm"),
            Self::Cold => write!(f, "Cold"),
            Self::DeepArchive => write!(f, "DeepArchive"),
        }
    }
}

impl StorageTier {
    /// Returns the retrieval latency estimate for this tier.
    pub fn estimated_retrieval_latency(&self) -> Duration {
        match self {
            Self::Hot => Duration::from_millis(10),
            Self::Warm => Duration::from_secs(1),
            Self::Cold => Duration::from_secs(60),
            Self::DeepArchive => Duration::from_secs(3600),
        }
    }

    /// Returns the relative cost factor (1.0 = baseline).
    #[allow(clippy::cast_precision_loss)]
    pub fn cost_factor(&self) -> f64 {
        match self {
            Self::Hot => 10.0,
            Self::Warm => 3.0,
            Self::Cold => 1.0,
            Self::DeepArchive => 0.3,
        }
    }

    /// Returns the tier priority (lower = hotter).
    pub fn priority(&self) -> u8 {
        match self {
            Self::Hot => 0,
            Self::Warm => 1,
            Self::Cold => 2,
            Self::DeepArchive => 3,
        }
    }

    /// Returns the next colder tier, if any.
    pub fn colder(&self) -> Option<Self> {
        match self {
            Self::Hot => Some(Self::Warm),
            Self::Warm => Some(Self::Cold),
            Self::Cold => Some(Self::DeepArchive),
            Self::DeepArchive => None,
        }
    }

    /// Returns the next warmer tier, if any.
    pub fn warmer(&self) -> Option<Self> {
        match self {
            Self::Hot => None,
            Self::Warm => Some(Self::Hot),
            Self::Cold => Some(Self::Warm),
            Self::DeepArchive => Some(Self::Cold),
        }
    }
}

/// Policy rule for automatic tier migration.
#[derive(Debug, Clone)]
pub struct TierPolicy {
    /// Source tier this policy applies to.
    pub source_tier: StorageTier,
    /// Destination tier after migration.
    pub destination_tier: StorageTier,
    /// Minimum age before migration is considered.
    pub min_age: Duration,
    /// Maximum access count within the evaluation window.
    pub max_access_count: u64,
    /// Evaluation window for access count measurement.
    pub evaluation_window: Duration,
    /// Whether the policy is currently active.
    pub enabled: bool,
}

impl TierPolicy {
    /// Creates a new tier migration policy.
    pub fn new(
        source_tier: StorageTier,
        destination_tier: StorageTier,
        min_age: Duration,
        max_access_count: u64,
    ) -> Self {
        Self {
            source_tier,
            destination_tier,
            min_age,
            max_access_count,
            evaluation_window: Duration::from_secs(30 * 24 * 3600),
            enabled: true,
        }
    }

    /// Sets the evaluation window for access count.
    pub fn with_evaluation_window(mut self, window: Duration) -> Self {
        self.evaluation_window = window;
        self
    }

    /// Enables or disables the policy.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Tracks access patterns for a single archived asset.
#[derive(Debug, Clone)]
pub struct AssetAccessRecord {
    /// Path to the archived asset.
    pub path: PathBuf,
    /// Current storage tier.
    pub current_tier: StorageTier,
    /// When the asset was first ingested.
    pub ingested_at: SystemTime,
    /// When the asset was last accessed.
    pub last_accessed: SystemTime,
    /// Total number of accesses.
    pub total_accesses: u64,
    /// Access timestamps within the current evaluation window.
    pub recent_access_times: Vec<SystemTime>,
    /// Size of the asset in bytes.
    pub size_bytes: u64,
}

impl AssetAccessRecord {
    /// Creates a new access record for a freshly ingested asset.
    pub fn new(path: PathBuf, size_bytes: u64) -> Self {
        let now = SystemTime::now();
        Self {
            path,
            current_tier: StorageTier::Hot,
            ingested_at: now,
            last_accessed: now,
            total_accesses: 0,
            recent_access_times: Vec::new(),
            size_bytes,
        }
    }

    /// Records an access event.
    pub fn record_access(&mut self) {
        let now = SystemTime::now();
        self.last_accessed = now;
        self.total_accesses += 1;
        self.recent_access_times.push(now);
    }

    /// Prunes access times older than the given window.
    pub fn prune_access_times(&mut self, window: Duration) {
        let cutoff = SystemTime::now()
            .checked_sub(window)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.recent_access_times.retain(|t| *t >= cutoff);
    }

    /// Returns the number of accesses within the given window.
    pub fn accesses_within(&self, window: Duration) -> u64 {
        let cutoff = SystemTime::now()
            .checked_sub(window)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.recent_access_times
            .iter()
            .filter(|t| **t >= cutoff)
            .count() as u64
    }

    /// Returns the age of the asset since ingestion.
    pub fn age(&self) -> Duration {
        self.ingested_at
            .elapsed()
            .unwrap_or(Duration::ZERO)
    }
}

/// Proposed migration action.
#[derive(Debug, Clone)]
pub struct MigrationAction {
    /// Path to the asset to migrate.
    pub asset_path: PathBuf,
    /// Current tier.
    pub from_tier: StorageTier,
    /// Destination tier.
    pub to_tier: StorageTier,
    /// Reason for migration.
    pub reason: String,
    /// Estimated cost savings (negative means increased cost).
    pub estimated_cost_delta: f64,
}

impl fmt::Display for MigrationAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Migrate '{}' from {} to {} (reason: {}, cost delta: {:.2})",
            self.asset_path.display(),
            self.from_tier,
            self.to_tier,
            self.reason,
            self.estimated_cost_delta
        )
    }
}

/// Storage tier manager that evaluates policies and proposes migrations.
#[derive(Debug)]
pub struct TierManager {
    /// Configured policies.
    policies: Vec<TierPolicy>,
    /// Asset access records keyed by path string.
    assets: HashMap<String, AssetAccessRecord>,
}

impl TierManager {
    /// Creates a new tier manager with no policies.
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
            assets: HashMap::new(),
        }
    }

    /// Creates a tier manager with default policies.
    pub fn with_default_policies() -> Self {
        let mut mgr = Self::new();
        // Hot -> Warm after 7 days if fewer than 5 accesses in 30 days
        mgr.add_policy(TierPolicy::new(
            StorageTier::Hot,
            StorageTier::Warm,
            Duration::from_secs(7 * 24 * 3600),
            5,
        ));
        // Warm -> Cold after 30 days if fewer than 2 accesses in 30 days
        mgr.add_policy(TierPolicy::new(
            StorageTier::Warm,
            StorageTier::Cold,
            Duration::from_secs(30 * 24 * 3600),
            2,
        ));
        // Cold -> DeepArchive after 180 days if fewer than 1 access in 30 days
        mgr.add_policy(TierPolicy::new(
            StorageTier::Cold,
            StorageTier::DeepArchive,
            Duration::from_secs(180 * 24 * 3600),
            1,
        ));
        mgr
    }

    /// Adds a policy to the manager.
    pub fn add_policy(&mut self, policy: TierPolicy) {
        self.policies.push(policy);
    }

    /// Returns the number of configured policies.
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }

    /// Registers a new asset.
    pub fn register_asset(&mut self, record: AssetAccessRecord) {
        let key = record.path.to_string_lossy().to_string();
        self.assets.insert(key, record);
    }

    /// Records an access for an asset.
    pub fn record_access(&mut self, path: &str) -> bool {
        if let Some(record) = self.assets.get_mut(path) {
            record.record_access();
            true
        } else {
            false
        }
    }

    /// Returns the number of tracked assets.
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Returns the asset record for a path.
    pub fn get_asset(&self, path: &str) -> Option<&AssetAccessRecord> {
        self.assets.get(path)
    }

    /// Evaluates all policies and returns proposed migration actions.
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate_policies(&self) -> Vec<MigrationAction> {
        let mut actions = Vec::new();

        for (_, record) in &self.assets {
            for policy in &self.policies {
                if !policy.enabled {
                    continue;
                }
                if record.current_tier != policy.source_tier {
                    continue;
                }
                let age = record.age();
                if age < policy.min_age {
                    continue;
                }
                let recent = record.accesses_within(policy.evaluation_window);
                if recent < policy.max_access_count {
                    let cost_delta = record.size_bytes as f64
                        * (policy.destination_tier.cost_factor()
                            - policy.source_tier.cost_factor());
                    actions.push(MigrationAction {
                        asset_path: record.path.clone(),
                        from_tier: record.current_tier,
                        to_tier: policy.destination_tier,
                        reason: format!(
                            "Age: {:.1} days, accesses in window: {}",
                            age.as_secs_f64() / 86400.0,
                            recent
                        ),
                        estimated_cost_delta: cost_delta,
                    });
                }
            }
        }

        actions
    }

    /// Applies a migration action (updates the asset record tier).
    pub fn apply_migration(&mut self, action: &MigrationAction) -> bool {
        let key = action.asset_path.to_string_lossy().to_string();
        if let Some(record) = self.assets.get_mut(&key) {
            if record.current_tier == action.from_tier {
                record.current_tier = action.to_tier;
                return true;
            }
        }
        false
    }

    /// Returns a summary of assets per tier.
    pub fn tier_summary(&self) -> HashMap<StorageTier, TierSummary> {
        let mut summary: HashMap<StorageTier, TierSummary> = HashMap::new();
        for (_, record) in &self.assets {
            let entry = summary.entry(record.current_tier).or_insert(TierSummary {
                tier: record.current_tier,
                asset_count: 0,
                total_bytes: 0,
            });
            entry.asset_count += 1;
            entry.total_bytes += record.size_bytes;
        }
        summary
    }
}

impl Default for TierManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for a single storage tier.
#[derive(Debug, Clone)]
pub struct TierSummary {
    /// The tier.
    pub tier: StorageTier,
    /// Number of assets in this tier.
    pub asset_count: u64,
    /// Total bytes in this tier.
    pub total_bytes: u64,
}

impl TierSummary {
    /// Returns the estimated monthly cost for this tier.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_monthly_cost(&self) -> f64 {
        self.total_bytes as f64 * self.tier.cost_factor() / 1_073_741_824.0 * 0.023
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_tier_display() {
        assert_eq!(StorageTier::Hot.to_string(), "Hot");
        assert_eq!(StorageTier::Warm.to_string(), "Warm");
        assert_eq!(StorageTier::Cold.to_string(), "Cold");
        assert_eq!(StorageTier::DeepArchive.to_string(), "DeepArchive");
    }

    #[test]
    fn test_tier_cost_factor_ordering() {
        assert!(StorageTier::Hot.cost_factor() > StorageTier::Warm.cost_factor());
        assert!(StorageTier::Warm.cost_factor() > StorageTier::Cold.cost_factor());
        assert!(StorageTier::Cold.cost_factor() > StorageTier::DeepArchive.cost_factor());
    }

    #[test]
    fn test_tier_priority() {
        assert_eq!(StorageTier::Hot.priority(), 0);
        assert_eq!(StorageTier::Warm.priority(), 1);
        assert_eq!(StorageTier::Cold.priority(), 2);
        assert_eq!(StorageTier::DeepArchive.priority(), 3);
    }

    #[test]
    fn test_tier_colder_warmer() {
        assert_eq!(StorageTier::Hot.colder(), Some(StorageTier::Warm));
        assert_eq!(StorageTier::Warm.colder(), Some(StorageTier::Cold));
        assert_eq!(StorageTier::Cold.colder(), Some(StorageTier::DeepArchive));
        assert_eq!(StorageTier::DeepArchive.colder(), None);

        assert_eq!(StorageTier::Hot.warmer(), None);
        assert_eq!(StorageTier::Warm.warmer(), Some(StorageTier::Hot));
        assert_eq!(StorageTier::Cold.warmer(), Some(StorageTier::Warm));
        assert_eq!(StorageTier::DeepArchive.warmer(), Some(StorageTier::Cold));
    }

    #[test]
    fn test_tier_retrieval_latency_ordering() {
        assert!(
            StorageTier::Hot.estimated_retrieval_latency()
                < StorageTier::Warm.estimated_retrieval_latency()
        );
        assert!(
            StorageTier::Warm.estimated_retrieval_latency()
                < StorageTier::Cold.estimated_retrieval_latency()
        );
        assert!(
            StorageTier::Cold.estimated_retrieval_latency()
                < StorageTier::DeepArchive.estimated_retrieval_latency()
        );
    }

    #[test]
    fn test_asset_access_record_new() {
        let record = AssetAccessRecord::new(PathBuf::from("/archive/video.mxf"), 1_000_000);
        assert_eq!(record.current_tier, StorageTier::Hot);
        assert_eq!(record.total_accesses, 0);
        assert_eq!(record.size_bytes, 1_000_000);
    }

    #[test]
    fn test_asset_record_access() {
        let mut record = AssetAccessRecord::new(PathBuf::from("/archive/video.mxf"), 500);
        record.record_access();
        record.record_access();
        record.record_access();
        assert_eq!(record.total_accesses, 3);
        assert_eq!(record.recent_access_times.len(), 3);
    }

    #[test]
    fn test_accesses_within_window() {
        let mut record = AssetAccessRecord::new(PathBuf::from("/test.mxf"), 100);
        record.record_access();
        record.record_access();
        let count = record.accesses_within(Duration::from_secs(3600));
        assert_eq!(count, 2);
    }

    #[test]
    fn test_tier_manager_default_policies() {
        let mgr = TierManager::with_default_policies();
        assert_eq!(mgr.policy_count(), 3);
    }

    #[test]
    fn test_tier_manager_register_and_access() {
        let mut mgr = TierManager::new();
        let record = AssetAccessRecord::new(PathBuf::from("/archive/clip.mov"), 2048);
        mgr.register_asset(record);
        assert_eq!(mgr.asset_count(), 1);
        assert!(mgr.record_access("/archive/clip.mov"));
        assert!(!mgr.record_access("/nonexistent"));
        let asset = mgr.get_asset("/archive/clip.mov").expect("asset should be valid");
        assert_eq!(asset.total_accesses, 1);
    }

    #[test]
    fn test_tier_manager_summary() {
        let mut mgr = TierManager::new();
        mgr.register_asset(AssetAccessRecord::new(PathBuf::from("/a"), 1000));
        mgr.register_asset(AssetAccessRecord::new(PathBuf::from("/b"), 2000));
        let summary = mgr.tier_summary();
        let hot = summary.get(&StorageTier::Hot).expect("hot should be valid");
        assert_eq!(hot.asset_count, 2);
        assert_eq!(hot.total_bytes, 3000);
    }

    #[test]
    fn test_apply_migration() {
        let mut mgr = TierManager::new();
        mgr.register_asset(AssetAccessRecord::new(PathBuf::from("/vid.mxf"), 500));
        let action = MigrationAction {
            asset_path: PathBuf::from("/vid.mxf"),
            from_tier: StorageTier::Hot,
            to_tier: StorageTier::Warm,
            reason: "test".to_string(),
            estimated_cost_delta: -100.0,
        };
        assert!(mgr.apply_migration(&action));
        let asset = mgr.get_asset("/vid.mxf").expect("asset should be valid");
        assert_eq!(asset.current_tier, StorageTier::Warm);
    }

    #[test]
    fn test_migration_action_display() {
        let action = MigrationAction {
            asset_path: PathBuf::from("/archive/test.mxf"),
            from_tier: StorageTier::Hot,
            to_tier: StorageTier::Cold,
            reason: "low access".to_string(),
            estimated_cost_delta: -50.0,
        };
        let display = format!("{}", action);
        assert!(display.contains("Hot"));
        assert!(display.contains("Cold"));
        assert!(display.contains("low access"));
    }

    #[test]
    fn test_tier_summary_cost() {
        let summary = TierSummary {
            tier: StorageTier::Hot,
            asset_count: 1,
            total_bytes: 1_073_741_824, // 1 GiB
        };
        let cost = summary.estimated_monthly_cost();
        assert!(cost > 0.0);
    }
}
