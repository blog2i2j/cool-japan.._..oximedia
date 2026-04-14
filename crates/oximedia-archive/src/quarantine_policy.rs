//! Configurable quarantine policies.
//!
//! Provides rules for:
//! - Maximum quarantine size (auto-evict oldest entries when limit exceeded)
//! - Auto-cleanup after N days
//! - Size-based eviction strategies
//! - Policy evaluation against a set of quarantine records

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// QuarantinePolicy
// ---------------------------------------------------------------------------

/// Configurable policy governing how quarantined files are managed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantinePolicy {
    /// Maximum total size of quarantine directory in bytes.
    /// `None` means unlimited.
    pub max_total_bytes: Option<u64>,
    /// Maximum number of quarantined files.
    /// `None` means unlimited.
    pub max_file_count: Option<usize>,
    /// Auto-delete quarantined files older than this many days.
    /// `None` means never auto-delete.
    pub auto_cleanup_after_days: Option<u32>,
    /// Maximum size of a single file that may be quarantined in bytes.
    /// Files larger than this are rejected from quarantine.
    pub max_single_file_bytes: Option<u64>,
    /// Strategy for evicting entries when the quota is exceeded.
    pub eviction_strategy: EvictionStrategy,
    /// Whether to allow restoring quarantined files.
    pub allow_restore: bool,
    /// Whether quarantined files should be zero-filled instead of moved.
    /// Useful for secure erasure workflows.
    pub secure_delete: bool,
}

impl Default for QuarantinePolicy {
    fn default() -> Self {
        Self {
            max_total_bytes: Some(10 * 1024 * 1024 * 1024), // 10 GiB
            max_file_count: Some(10_000),
            auto_cleanup_after_days: Some(90),
            max_single_file_bytes: None,
            eviction_strategy: EvictionStrategy::OldestFirst,
            allow_restore: true,
            secure_delete: false,
        }
    }
}

impl QuarantinePolicy {
    /// Create an unrestricted policy (no limits).
    #[must_use]
    pub fn unlimited() -> Self {
        Self {
            max_total_bytes: None,
            max_file_count: None,
            auto_cleanup_after_days: None,
            max_single_file_bytes: None,
            eviction_strategy: EvictionStrategy::OldestFirst,
            allow_restore: true,
            secure_delete: false,
        }
    }

    /// Create a strict policy suitable for high-security environments.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            max_total_bytes: Some(1024 * 1024 * 1024), // 1 GiB
            max_file_count: Some(100),
            auto_cleanup_after_days: Some(30),
            max_single_file_bytes: Some(512 * 1024 * 1024), // 512 MiB
            eviction_strategy: EvictionStrategy::OldestFirst,
            allow_restore: false,
            secure_delete: true,
        }
    }

    /// Check whether a file of `size_bytes` may be quarantined given the
    /// current state of the quarantine directory.
    pub fn check_admission(
        &self,
        file_size_bytes: u64,
        current_total_bytes: u64,
        current_file_count: usize,
    ) -> AdmissionDecision {
        // Single-file size check
        if let Some(max_single) = self.max_single_file_bytes {
            if file_size_bytes > max_single {
                return AdmissionDecision::Rejected(format!(
                    "file size {} bytes exceeds per-file limit {} bytes",
                    file_size_bytes, max_single
                ));
            }
        }

        // File count check
        if let Some(max_count) = self.max_file_count {
            if current_file_count >= max_count {
                return match self.eviction_strategy {
                    EvictionStrategy::OldestFirst => {
                        AdmissionDecision::AdmitAfterEviction(EvictionRequest {
                            reason: format!(
                                "file count {current_file_count} at limit {max_count}"
                            ),
                            strategy: self.eviction_strategy,
                            bytes_to_free: 0,
                            files_to_evict: 1,
                        })
                    }
                    EvictionStrategy::LargestFirst => {
                        AdmissionDecision::AdmitAfterEviction(EvictionRequest {
                            reason: format!(
                                "file count {current_file_count} at limit {max_count}"
                            ),
                            strategy: self.eviction_strategy,
                            bytes_to_free: 0,
                            files_to_evict: 1,
                        })
                    }
                    EvictionStrategy::None => {
                        AdmissionDecision::Rejected(format!(
                            "file count {current_file_count} at limit {max_count} and eviction is disabled"
                        ))
                    }
                };
            }
        }

        // Total size check
        if let Some(max_bytes) = self.max_total_bytes {
            let after = current_total_bytes.saturating_add(file_size_bytes);
            if after > max_bytes {
                let bytes_to_free = after - max_bytes;
                return match self.eviction_strategy {
                    EvictionStrategy::OldestFirst | EvictionStrategy::LargestFirst => {
                        AdmissionDecision::AdmitAfterEviction(EvictionRequest {
                            reason: format!(
                                "total size {after} bytes would exceed limit {max_bytes}"
                            ),
                            strategy: self.eviction_strategy,
                            bytes_to_free,
                            files_to_evict: 0,
                        })
                    }
                    EvictionStrategy::None => {
                        AdmissionDecision::Rejected(format!(
                            "total size {after} bytes would exceed limit {max_bytes} and eviction is disabled"
                        ))
                    }
                };
            }
        }

        AdmissionDecision::Admitted
    }

    /// Determine which quarantine records should be cleaned up based on age.
    ///
    /// `records` is a slice of `(record_id, quarantine_date_epoch_secs, size_bytes)`.
    /// Returns the IDs of records eligible for cleanup.
    #[must_use]
    pub fn eligible_for_cleanup(
        &self,
        records: &[(u64, u64, u64)],
        now_epoch_secs: u64,
    ) -> Vec<u64> {
        let Some(max_days) = self.auto_cleanup_after_days else {
            return Vec::new();
        };
        let threshold_secs = u64::from(max_days) * 86_400;

        records
            .iter()
            .filter_map(|(id, quarantine_secs, _size)| {
                let age = now_epoch_secs.saturating_sub(*quarantine_secs);
                if age >= threshold_secs {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Determine which records to evict to satisfy the given eviction request.
    ///
    /// `records` is sorted from oldest (index 0) to newest.
    /// Each record: `(record_id, quarantine_date_epoch_secs, size_bytes)`.
    /// Returns the IDs of records that should be evicted.
    #[must_use]
    pub fn select_for_eviction(
        &self,
        records: &[(u64, u64, u64)],
        request: &EvictionRequest,
    ) -> Vec<u64> {
        let mut sorted = records.to_vec();
        match request.strategy {
            EvictionStrategy::OldestFirst => {
                // Already sorted oldest-first
                sorted.sort_by_key(|(_, ts, _)| *ts);
            }
            EvictionStrategy::LargestFirst => {
                sorted.sort_by(|a, b| b.2.cmp(&a.2));
            }
            EvictionStrategy::None => return Vec::new(),
        }

        let mut evicted = Vec::new();
        let mut freed_bytes = 0u64;
        let mut freed_files = 0usize;

        for (id, _ts, size) in &sorted {
            if freed_files >= request.files_to_evict.max(1) && freed_bytes >= request.bytes_to_free
            {
                break;
            }
            evicted.push(*id);
            freed_bytes = freed_bytes.saturating_add(*size);
            freed_files += 1;
        }

        evicted
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Strategy for selecting which quarantined files to evict when space is low.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Evict the oldest quarantined files first.
    OldestFirst,
    /// Evict the largest files first to reclaim the most space quickly.
    LargestFirst,
    /// Do not automatically evict anything; reject new quarantine requests.
    None,
}

/// Decision returned by `QuarantinePolicy::check_admission`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// The file may be quarantined immediately.
    Admitted,
    /// The file may be quarantined after evicting the specified records.
    AdmitAfterEviction(EvictionRequest),
    /// The file cannot be quarantined under current policy.
    Rejected(String),
}

impl AdmissionDecision {
    /// Returns `true` if admission was granted (immediately or after eviction).
    #[must_use]
    pub fn is_admitted(&self) -> bool {
        !matches!(self, Self::Rejected(_))
    }

    /// Returns `true` if the file was outright rejected.
    #[must_use]
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected(_))
    }
}

/// Specification for an eviction operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionRequest {
    /// Human-readable reason for the eviction.
    pub reason: String,
    /// Strategy to use when selecting candidates.
    pub strategy: EvictionStrategy,
    /// Minimum bytes that must be freed.
    pub bytes_to_free: u64,
    /// Minimum number of files to evict.
    pub files_to_evict: usize,
}

// ---------------------------------------------------------------------------
// QuarantineInventory — in-memory snapshot for policy evaluation
// ---------------------------------------------------------------------------

/// Lightweight snapshot of the quarantine directory for policy evaluation.
#[derive(Debug, Clone, Default)]
pub struct QuarantineInventory {
    /// Records: `(record_id, quarantine_date_epoch_secs, size_bytes, path)`.
    records: Vec<(u64, u64, u64, PathBuf)>,
}

impl QuarantineInventory {
    /// Create an empty inventory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a record to the inventory.
    pub fn add(&mut self, id: u64, quarantine_epoch_secs: u64, size_bytes: u64, path: PathBuf) {
        self.records
            .push((id, quarantine_epoch_secs, size_bytes, path));
    }

    /// Total number of records.
    #[must_use]
    pub fn count(&self) -> usize {
        self.records.len()
    }

    /// Total bytes of all quarantined files.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.records.iter().map(|(_, _, s, _)| s).sum()
    }

    /// Records as `(id, timestamp, size)` tuples for policy evaluation.
    #[must_use]
    pub fn as_tuples(&self) -> Vec<(u64, u64, u64)> {
        self.records
            .iter()
            .map(|(id, ts, size, _)| (*id, *ts, *size))
            .collect()
    }

    /// Get the path for a given record ID.
    #[must_use]
    pub fn path_for(&self, id: u64) -> Option<&Path> {
        self.records
            .iter()
            .find(|(rid, _, _, _)| *rid == id)
            .map(|(_, _, _, p)| p.as_path())
    }

    /// Apply cleanup: return IDs of records that should be deleted based on policy.
    #[must_use]
    pub fn cleanup_candidates(&self, policy: &QuarantinePolicy, now_epoch_secs: u64) -> Vec<u64> {
        policy.eligible_for_cleanup(&self.as_tuples(), now_epoch_secs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    // --- QuarantinePolicy::check_admission ---

    #[test]
    fn test_admission_no_limits_always_admitted() {
        let policy = QuarantinePolicy::unlimited();
        let decision = policy.check_admission(999_999_999, 0, 0);
        assert_eq!(decision, AdmissionDecision::Admitted);
    }

    #[test]
    fn test_admission_within_limits() {
        let policy = QuarantinePolicy {
            max_total_bytes: Some(1_000_000),
            max_file_count: Some(100),
            max_single_file_bytes: Some(500_000),
            ..QuarantinePolicy::default()
        };
        let decision = policy.check_admission(1000, 100_000, 5);
        assert_eq!(decision, AdmissionDecision::Admitted);
    }

    #[test]
    fn test_admission_single_file_too_large() {
        let policy = QuarantinePolicy {
            max_single_file_bytes: Some(1024),
            ..QuarantinePolicy::unlimited()
        };
        let decision = policy.check_admission(2048, 0, 0);
        assert!(decision.is_rejected());
    }

    #[test]
    fn test_admission_file_count_exceeded_triggers_eviction() {
        let policy = QuarantinePolicy {
            max_file_count: Some(5),
            eviction_strategy: EvictionStrategy::OldestFirst,
            ..QuarantinePolicy::unlimited()
        };
        let decision = policy.check_admission(100, 0, 5);
        assert!(decision.is_admitted());
        assert!(matches!(decision, AdmissionDecision::AdmitAfterEviction(_)));
    }

    #[test]
    fn test_admission_file_count_exceeded_no_eviction_rejected() {
        let policy = QuarantinePolicy {
            max_file_count: Some(5),
            eviction_strategy: EvictionStrategy::None,
            ..QuarantinePolicy::unlimited()
        };
        let decision = policy.check_admission(100, 0, 5);
        assert!(decision.is_rejected());
    }

    #[test]
    fn test_admission_total_bytes_exceeded_triggers_eviction() {
        let policy = QuarantinePolicy {
            max_total_bytes: Some(1000),
            eviction_strategy: EvictionStrategy::OldestFirst,
            ..QuarantinePolicy::unlimited()
        };
        let decision = policy.check_admission(500, 800, 0);
        assert!(decision.is_admitted());
        assert!(matches!(decision, AdmissionDecision::AdmitAfterEviction(_)));
    }

    #[test]
    fn test_admission_total_bytes_exceeded_no_eviction_rejected() {
        let policy = QuarantinePolicy {
            max_total_bytes: Some(1000),
            eviction_strategy: EvictionStrategy::None,
            ..QuarantinePolicy::unlimited()
        };
        let decision = policy.check_admission(500, 800, 0);
        assert!(decision.is_rejected());
    }

    // --- QuarantinePolicy::eligible_for_cleanup ---

    #[test]
    fn test_cleanup_no_policy_none() {
        let policy = QuarantinePolicy {
            auto_cleanup_after_days: None,
            ..QuarantinePolicy::default()
        };
        let now = now_secs();
        let old_ts = now - 200 * 86400;
        let records = vec![(1, old_ts, 100)];
        let eligible = policy.eligible_for_cleanup(&records, now);
        assert!(eligible.is_empty());
    }

    #[test]
    fn test_cleanup_old_record_eligible() {
        let policy = QuarantinePolicy {
            auto_cleanup_after_days: Some(30),
            ..QuarantinePolicy::default()
        };
        let now = now_secs();
        let old_ts = now - 31 * 86400; // 31 days ago
        let records = vec![(42, old_ts, 500)];
        let eligible = policy.eligible_for_cleanup(&records, now);
        assert_eq!(eligible, vec![42]);
    }

    #[test]
    fn test_cleanup_recent_record_not_eligible() {
        let policy = QuarantinePolicy {
            auto_cleanup_after_days: Some(30),
            ..QuarantinePolicy::default()
        };
        let now = now_secs();
        let recent_ts = now - 5 * 86400; // 5 days ago
        let records = vec![(99, recent_ts, 100)];
        let eligible = policy.eligible_for_cleanup(&records, now);
        assert!(eligible.is_empty());
    }

    #[test]
    fn test_cleanup_mixed_records() {
        let policy = QuarantinePolicy {
            auto_cleanup_after_days: Some(30),
            ..QuarantinePolicy::default()
        };
        let now = now_secs();
        let records = vec![
            (1, now - 60 * 86400, 100), // 60 days old → eligible
            (2, now - 10 * 86400, 200), // 10 days old → not eligible
            (3, now - 31 * 86400, 300), // 31 days old → eligible
        ];
        let mut eligible = policy.eligible_for_cleanup(&records, now);
        eligible.sort();
        assert_eq!(eligible, vec![1, 3]);
    }

    // --- QuarantinePolicy::select_for_eviction ---

    #[test]
    fn test_eviction_oldest_first() {
        let policy = QuarantinePolicy {
            eviction_strategy: EvictionStrategy::OldestFirst,
            ..QuarantinePolicy::default()
        };
        let now = now_secs();
        let records = vec![
            (1, now - 10, 100),
            (2, now - 100, 200), // oldest
            (3, now - 5, 300),
        ];
        let req = EvictionRequest {
            reason: "test".into(),
            strategy: EvictionStrategy::OldestFirst,
            bytes_to_free: 0,
            files_to_evict: 1,
        };
        let evicted = policy.select_for_eviction(&records, &req);
        assert_eq!(evicted, vec![2]); // oldest evicted first
    }

    #[test]
    fn test_eviction_largest_first() {
        let policy = QuarantinePolicy {
            eviction_strategy: EvictionStrategy::LargestFirst,
            ..QuarantinePolicy::default()
        };
        let now = now_secs();
        let records = vec![
            (1, now - 10, 100),
            (2, now - 20, 500), // largest
            (3, now - 30, 200),
        ];
        let req = EvictionRequest {
            reason: "test".into(),
            strategy: EvictionStrategy::LargestFirst,
            bytes_to_free: 0,
            files_to_evict: 1,
        };
        let evicted = policy.select_for_eviction(&records, &req);
        assert_eq!(evicted, vec![2]); // largest evicted first
    }

    #[test]
    fn test_eviction_none_strategy() {
        let policy = QuarantinePolicy {
            eviction_strategy: EvictionStrategy::None,
            ..QuarantinePolicy::default()
        };
        let records = vec![(1, 1000, 100), (2, 2000, 200)];
        let req = EvictionRequest {
            reason: "test".into(),
            strategy: EvictionStrategy::None,
            bytes_to_free: 100,
            files_to_evict: 1,
        };
        let evicted = policy.select_for_eviction(&records, &req);
        assert!(evicted.is_empty());
    }

    #[test]
    fn test_eviction_frees_enough_bytes() {
        let policy = QuarantinePolicy::default();
        let now = now_secs();
        let records = vec![
            (1, now - 300, 100),
            (2, now - 200, 200),
            (3, now - 100, 500),
        ];
        let req = EvictionRequest {
            reason: "test".into(),
            strategy: EvictionStrategy::OldestFirst,
            bytes_to_free: 250,
            files_to_evict: 0,
        };
        let evicted = policy.select_for_eviction(&records, &req);
        let freed: u64 = evicted
            .iter()
            .map(|id| {
                records
                    .iter()
                    .find(|(r, _, _)| r == id)
                    .map(|(_, _, s)| *s)
                    .unwrap_or(0)
            })
            .sum();
        assert!(freed >= 250, "freed {freed} bytes, expected at least 250");
    }

    // --- QuarantineInventory ---

    #[test]
    fn test_inventory_empty() {
        let inv = QuarantineInventory::new();
        assert_eq!(inv.count(), 0);
        assert_eq!(inv.total_bytes(), 0);
    }

    #[test]
    fn test_inventory_add_and_query() {
        let mut inv = QuarantineInventory::new();
        inv.add(1, 1000, 500, PathBuf::from("/q/file1.bin"));
        inv.add(2, 2000, 1500, PathBuf::from("/q/file2.bin"));
        assert_eq!(inv.count(), 2);
        assert_eq!(inv.total_bytes(), 2000);
    }

    #[test]
    fn test_inventory_path_for() {
        let mut inv = QuarantineInventory::new();
        inv.add(42, 1000, 100, PathBuf::from("/q/answer.bin"));
        let path = inv.path_for(42);
        assert_eq!(path, Some(Path::new("/q/answer.bin")));
        assert!(inv.path_for(999).is_none());
    }

    #[test]
    fn test_inventory_cleanup_candidates() {
        let mut inv = QuarantineInventory::new();
        let now = now_secs();
        inv.add(1, now - 100 * 86400, 100, PathBuf::from("/q/old.bin"));
        inv.add(2, now - 5 * 86400, 200, PathBuf::from("/q/recent.bin"));

        let policy = QuarantinePolicy {
            auto_cleanup_after_days: Some(30),
            ..QuarantinePolicy::default()
        };
        let candidates = inv.cleanup_candidates(&policy, now);
        assert_eq!(candidates, vec![1]);
    }

    // --- Policy presets ---

    #[test]
    fn test_default_policy_has_limits() {
        let p = QuarantinePolicy::default();
        assert!(p.max_total_bytes.is_some());
        assert!(p.max_file_count.is_some());
        assert!(p.auto_cleanup_after_days.is_some());
        assert!(p.allow_restore);
    }

    #[test]
    fn test_strict_policy() {
        let p = QuarantinePolicy::strict();
        assert_eq!(p.allow_restore, false);
        assert_eq!(p.secure_delete, true);
        assert!(p.max_single_file_bytes.is_some());
        assert!(
            p.auto_cleanup_after_days
                .expect("strict policy should have auto_cleanup_after_days")
                <= 30
        );
    }

    #[test]
    fn test_unlimited_policy_admits_any_size() {
        let p = QuarantinePolicy::unlimited();
        let decision = p.check_admission(u64::MAX, u64::MAX / 2, usize::MAX / 2);
        // unlimited should always admit (no limits)
        assert_eq!(decision, AdmissionDecision::Admitted);
    }

    // --- AdmissionDecision helpers ---

    #[test]
    fn test_admission_decision_is_admitted_true_for_admitted() {
        assert!(AdmissionDecision::Admitted.is_admitted());
    }

    #[test]
    fn test_admission_decision_is_rejected_true() {
        let d = AdmissionDecision::Rejected("too big".into());
        assert!(d.is_rejected());
        assert!(!d.is_admitted());
    }

    #[test]
    fn test_admission_decision_eviction_is_admitted() {
        let d = AdmissionDecision::AdmitAfterEviction(EvictionRequest {
            reason: "r".into(),
            strategy: EvictionStrategy::OldestFirst,
            bytes_to_free: 0,
            files_to_evict: 1,
        });
        assert!(d.is_admitted());
        assert!(!d.is_rejected());
    }
}
