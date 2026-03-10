#![allow(dead_code)]
//! Storage migration planning between providers and tiers.
//!
//! Provides cost estimation, bandwidth scheduling, validation, and
//! progress tracking for bulk object migrations across storage backends.

use std::collections::HashMap;

/// Storage tier levels for migration planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    /// Hot / standard access tier.
    Hot,
    /// Warm / infrequent access tier.
    Warm,
    /// Cold / archive tier.
    Cold,
    /// Deep archive / glacier tier.
    DeepArchive,
}

/// Source or destination backend for migration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MigrationEndpoint {
    /// Provider name (e.g. "s3", "gcs", "azure", "local").
    pub provider: String,
    /// Bucket or container name.
    pub bucket: String,
    /// Key prefix filter.
    pub prefix: String,
    /// Storage tier.
    pub tier: StorageTier,
}

/// A single object scheduled for migration.
#[derive(Debug, Clone)]
pub struct MigrationItem {
    /// Object key.
    pub key: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Current state.
    pub state: MigrationItemState,
    /// Error message if failed.
    pub error: Option<String>,
}

/// State of a single migration item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationItemState {
    /// Pending transfer.
    Pending,
    /// Currently transferring.
    InProgress,
    /// Transfer completed.
    Completed,
    /// Transfer failed.
    Failed,
    /// Skipped (e.g. already exists at destination).
    Skipped,
}

/// Cost breakdown for a migration plan.
#[derive(Debug, Clone, Default)]
pub struct MigrationCost {
    /// Estimated egress cost in USD.
    pub egress_usd: f64,
    /// Estimated ingress cost in USD.
    pub ingress_usd: f64,
    /// Estimated API call cost in USD.
    pub api_calls_usd: f64,
    /// Estimated storage cost delta per month in USD.
    pub storage_delta_usd: f64,
}

impl MigrationCost {
    /// Total estimated one-time cost.
    pub fn total_onetime(&self) -> f64 {
        self.egress_usd + self.ingress_usd + self.api_calls_usd
    }

    /// Total monthly delta (positive = more expensive destination).
    pub fn monthly_delta(&self) -> f64 {
        self.storage_delta_usd
    }
}

/// Bandwidth configuration for migration scheduling.
#[derive(Debug, Clone)]
pub struct BandwidthConfig {
    /// Maximum bytes per second for transfer.
    pub max_bytes_per_sec: u64,
    /// Maximum concurrent transfers.
    pub max_concurrent: u32,
    /// Retry count on failure.
    pub max_retries: u32,
}

impl Default for BandwidthConfig {
    fn default() -> Self {
        Self {
            max_bytes_per_sec: 100 * 1024 * 1024, // 100 MB/s
            max_concurrent: 8,
            max_retries: 3,
        }
    }
}

/// Overall migration plan state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanState {
    /// Plan is being built.
    Draft,
    /// Plan has been validated and is ready.
    Ready,
    /// Migration is in progress.
    Running,
    /// Migration completed.
    Completed,
    /// Migration failed (some items may have succeeded).
    Failed,
    /// Migration was cancelled.
    Cancelled,
}

/// Progress snapshot for a running migration.
#[derive(Debug, Clone, Default)]
pub struct MigrationProgress {
    /// Items completed.
    pub completed: u64,
    /// Items failed.
    pub failed: u64,
    /// Items skipped.
    pub skipped: u64,
    /// Items remaining.
    pub remaining: u64,
    /// Bytes transferred so far.
    pub bytes_transferred: u64,
    /// Total bytes to transfer.
    pub bytes_total: u64,
}

impl MigrationProgress {
    /// Percentage complete by item count.
    #[allow(clippy::cast_precision_loss)]
    pub fn percent_by_items(&self) -> f64 {
        let total = self.completed + self.failed + self.skipped + self.remaining;
        if total == 0 {
            return 0.0;
        }
        (self.completed + self.skipped) as f64 / total as f64 * 100.0
    }

    /// Percentage complete by bytes.
    #[allow(clippy::cast_precision_loss)]
    pub fn percent_by_bytes(&self) -> f64 {
        if self.bytes_total == 0 {
            return 0.0;
        }
        self.bytes_transferred as f64 / self.bytes_total as f64 * 100.0
    }
}

/// A complete migration plan.
#[derive(Debug)]
pub struct MigrationPlan {
    /// Plan identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Source endpoint.
    pub source: MigrationEndpoint,
    /// Destination endpoint.
    pub destination: MigrationEndpoint,
    /// Items to migrate.
    items: Vec<MigrationItem>,
    /// Bandwidth configuration.
    pub bandwidth: BandwidthConfig,
    /// Estimated cost.
    pub cost: MigrationCost,
    /// Current plan state.
    pub state: PlanState,
    /// Validation errors (empty if valid).
    validation_errors: Vec<String>,
}

impl MigrationPlan {
    /// Create a new migration plan in draft state.
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        source: MigrationEndpoint,
        destination: MigrationEndpoint,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            source,
            destination,
            items: Vec::new(),
            bandwidth: BandwidthConfig::default(),
            cost: MigrationCost::default(),
            state: PlanState::Draft,
            validation_errors: Vec::new(),
        }
    }

    /// Add an item to the migration plan.
    pub fn add_item(&mut self, key: impl Into<String>, size_bytes: u64) {
        self.items.push(MigrationItem {
            key: key.into(),
            size_bytes,
            state: MigrationItemState::Pending,
            error: None,
        });
    }

    /// Number of items in the plan.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Total bytes to migrate.
    pub fn total_bytes(&self) -> u64 {
        self.items.iter().map(|i| i.size_bytes).sum()
    }

    /// Estimate transfer duration in seconds given current bandwidth config.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_duration_secs(&self) -> f64 {
        if self.bandwidth.max_bytes_per_sec == 0 {
            return f64::INFINITY;
        }
        self.total_bytes() as f64 / self.bandwidth.max_bytes_per_sec as f64
    }

    /// Validate the plan and return whether it is valid.
    pub fn validate(&mut self) -> bool {
        self.validation_errors.clear();
        if self.items.is_empty() {
            self.validation_errors
                .push("plan has no items to migrate".into());
        }
        if self.source == self.destination {
            self.validation_errors
                .push("source and destination are identical".into());
        }
        if self.bandwidth.max_bytes_per_sec == 0 {
            self.validation_errors
                .push("bandwidth limit is zero".into());
        }
        if self.validation_errors.is_empty() {
            self.state = PlanState::Ready;
            true
        } else {
            false
        }
    }

    /// Return validation errors.
    pub fn validation_errors(&self) -> &[String] {
        &self.validation_errors
    }

    /// Compute current progress.
    pub fn progress(&self) -> MigrationProgress {
        let mut p = MigrationProgress::default();
        for item in &self.items {
            match item.state {
                MigrationItemState::Pending | MigrationItemState::InProgress => {
                    p.remaining += 1;
                }
                MigrationItemState::Completed => {
                    p.completed += 1;
                    p.bytes_transferred += item.size_bytes;
                }
                MigrationItemState::Failed => {
                    p.failed += 1;
                }
                MigrationItemState::Skipped => {
                    p.skipped += 1;
                }
            }
        }
        p.bytes_total = self.total_bytes();
        p
    }

    /// Mark an item by index as completed.
    pub fn complete_item(&mut self, index: usize) -> Result<(), MigrationError> {
        let item = self
            .items
            .get_mut(index)
            .ok_or(MigrationError::ItemNotFound(index))?;
        item.state = MigrationItemState::Completed;
        Ok(())
    }

    /// Mark an item by index as failed.
    pub fn fail_item(
        &mut self,
        index: usize,
        error: impl Into<String>,
    ) -> Result<(), MigrationError> {
        let item = self
            .items
            .get_mut(index)
            .ok_or(MigrationError::ItemNotFound(index))?;
        item.state = MigrationItemState::Failed;
        item.error = Some(error.into());
        Ok(())
    }

    /// Get items grouped by state.
    pub fn items_by_state(&self) -> HashMap<MigrationItemState, usize> {
        let mut map = HashMap::new();
        for item in &self.items {
            *map.entry(item.state).or_insert(0) += 1;
        }
        map
    }
}

/// Errors from migration operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MigrationError {
    /// Item index out of bounds.
    ItemNotFound(usize),
    /// Plan is not in a valid state for the operation.
    InvalidState(String),
    /// Validation failed.
    ValidationFailed(Vec<String>),
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ItemNotFound(idx) => write!(f, "item at index {idx} not found"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            Self::ValidationFailed(errs) => write!(f, "validation failed: {}", errs.join("; ")),
        }
    }
}

impl std::error::Error for MigrationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_endpoint(provider: &str, bucket: &str) -> MigrationEndpoint {
        MigrationEndpoint {
            provider: provider.into(),
            bucket: bucket.into(),
            prefix: String::new(),
            tier: StorageTier::Hot,
        }
    }

    #[test]
    fn test_create_plan() {
        let plan = MigrationPlan::new(
            "m1",
            "test migration",
            sample_endpoint("s3", "src"),
            sample_endpoint("gcs", "dst"),
        );
        assert_eq!(plan.state, PlanState::Draft);
        assert_eq!(plan.item_count(), 0);
    }

    #[test]
    fn test_add_items() {
        let mut plan = MigrationPlan::new(
            "m2",
            "items",
            sample_endpoint("s3", "src"),
            sample_endpoint("gcs", "dst"),
        );
        plan.add_item("file1.mp4", 1_000_000);
        plan.add_item("file2.mkv", 2_000_000);
        assert_eq!(plan.item_count(), 2);
        assert_eq!(plan.total_bytes(), 3_000_000);
    }

    #[test]
    fn test_validate_valid() {
        let mut plan = MigrationPlan::new(
            "m3",
            "valid",
            sample_endpoint("s3", "src"),
            sample_endpoint("gcs", "dst"),
        );
        plan.add_item("f.mp4", 100);
        assert!(plan.validate());
        assert_eq!(plan.state, PlanState::Ready);
    }

    #[test]
    fn test_validate_empty() {
        let mut plan = MigrationPlan::new(
            "m4",
            "empty",
            sample_endpoint("s3", "src"),
            sample_endpoint("gcs", "dst"),
        );
        assert!(!plan.validate());
        assert!(!plan.validation_errors().is_empty());
    }

    #[test]
    fn test_validate_same_endpoint() {
        let ep = sample_endpoint("s3", "same");
        let mut plan = MigrationPlan::new("m5", "same", ep.clone(), ep);
        plan.add_item("f.mp4", 100);
        assert!(!plan.validate());
    }

    #[test]
    fn test_estimated_duration() {
        let mut plan = MigrationPlan::new(
            "m6",
            "dur",
            sample_endpoint("s3", "a"),
            sample_endpoint("gcs", "b"),
        );
        plan.bandwidth.max_bytes_per_sec = 1_000_000;
        plan.add_item("f.mp4", 5_000_000);
        let dur = plan.estimated_duration_secs();
        assert!((dur - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_complete_item() {
        let mut plan = MigrationPlan::new(
            "m7",
            "complete",
            sample_endpoint("s3", "a"),
            sample_endpoint("gcs", "b"),
        );
        plan.add_item("f.mp4", 1000);
        plan.complete_item(0).expect("complete item should succeed");
        let prog = plan.progress();
        assert_eq!(prog.completed, 1);
        assert_eq!(prog.remaining, 0);
    }

    #[test]
    fn test_fail_item() {
        let mut plan = MigrationPlan::new(
            "m8",
            "fail",
            sample_endpoint("s3", "a"),
            sample_endpoint("gcs", "b"),
        );
        plan.add_item("f.mp4", 1000);
        plan.fail_item(0, "network timeout").expect("fail item should succeed");
        let prog = plan.progress();
        assert_eq!(prog.failed, 1);
    }

    #[test]
    fn test_item_not_found() {
        let mut plan = MigrationPlan::new(
            "m9",
            "oob",
            sample_endpoint("s3", "a"),
            sample_endpoint("gcs", "b"),
        );
        let err = plan.complete_item(0).unwrap_err();
        assert!(matches!(err, MigrationError::ItemNotFound(0)));
    }

    #[test]
    fn test_progress_percent_items() {
        let p = MigrationProgress {
            completed: 3,
            failed: 0,
            skipped: 1,
            remaining: 6,
            bytes_transferred: 0,
            bytes_total: 0,
        };
        assert!((p.percent_by_items() - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_progress_percent_bytes() {
        let p = MigrationProgress {
            completed: 0,
            failed: 0,
            skipped: 0,
            remaining: 0,
            bytes_transferred: 500,
            bytes_total: 1000,
        };
        assert!((p.percent_by_bytes() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_total() {
        let c = MigrationCost {
            egress_usd: 1.0,
            ingress_usd: 0.5,
            api_calls_usd: 0.1,
            storage_delta_usd: 2.0,
        };
        assert!((c.total_onetime() - 1.6).abs() < 0.01);
        assert!((c.monthly_delta() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_items_by_state() {
        let mut plan = MigrationPlan::new(
            "m10",
            "states",
            sample_endpoint("s3", "a"),
            sample_endpoint("gcs", "b"),
        );
        plan.add_item("a.mp4", 100);
        plan.add_item("b.mp4", 200);
        plan.add_item("c.mp4", 300);
        plan.complete_item(0).expect("complete item should succeed");
        plan.fail_item(1, "err").expect("fail item should succeed");
        let m = plan.items_by_state();
        assert_eq!(m.get(&MigrationItemState::Completed), Some(&1));
        assert_eq!(m.get(&MigrationItemState::Failed), Some(&1));
        assert_eq!(m.get(&MigrationItemState::Pending), Some(&1));
    }

    #[test]
    fn test_migration_error_display() {
        let e = MigrationError::ItemNotFound(5);
        assert!(e.to_string().contains("5"));
        let e2 = MigrationError::InvalidState("bad".into());
        assert!(e2.to_string().contains("bad"));
    }
}
