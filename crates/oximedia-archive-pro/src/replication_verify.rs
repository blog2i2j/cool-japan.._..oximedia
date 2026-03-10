#![allow(dead_code)]
//! Replication verification for archive copies.
//!
//! Ensures that replicated archive copies across multiple storage locations
//! remain consistent and intact. Supports multi-site verification, divergence
//! detection, and repair recommendations.

use std::collections::HashMap;

/// Identifies a storage location for replication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReplicaLocation {
    /// Unique name for this location.
    pub name: String,
    /// URI or path to the storage root.
    pub uri: String,
    /// Whether this location is the primary copy.
    pub is_primary: bool,
}

impl ReplicaLocation {
    /// Creates a new replica location.
    #[must_use]
    pub fn new(name: impl Into<String>, uri: impl Into<String>, is_primary: bool) -> Self {
        Self {
            name: name.into(),
            uri: uri.into(),
            is_primary,
        }
    }

    /// Creates a primary location.
    #[must_use]
    pub fn primary(name: impl Into<String>, uri: impl Into<String>) -> Self {
        Self::new(name, uri, true)
    }

    /// Creates a secondary location.
    #[must_use]
    pub fn secondary(name: impl Into<String>, uri: impl Into<String>) -> Self {
        Self::new(name, uri, false)
    }
}

/// Status of a single file across replicas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileReplicaStatus {
    /// File is identical across all checked replicas.
    Consistent,
    /// File checksum differs between replicas.
    Diverged,
    /// File is missing from one or more replicas.
    Missing,
    /// File has not been checked yet.
    Unchecked,
    /// File verification encountered an error.
    Error,
}

/// Details about a single file's replica state.
#[derive(Debug, Clone)]
pub struct FileReplicaInfo {
    /// Relative path of the file within the archive.
    pub path: String,
    /// Overall status.
    pub status: FileReplicaStatus,
    /// Checksum values per location name.
    pub checksums: HashMap<String, String>,
    /// Size in bytes per location name.
    pub sizes: HashMap<String, u64>,
}

impl FileReplicaInfo {
    /// Creates a new file replica info entry.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            status: FileReplicaStatus::Unchecked,
            checksums: HashMap::new(),
            sizes: HashMap::new(),
        }
    }

    /// Records the checksum and size for a specific location.
    pub fn record_location(
        &mut self,
        location_name: impl Into<String>,
        checksum: impl Into<String>,
        size: u64,
    ) {
        let name = location_name.into();
        self.checksums.insert(name.clone(), checksum.into());
        self.sizes.insert(name, size);
    }

    /// Evaluates the consistency status across all recorded locations.
    pub fn evaluate(&mut self) {
        if self.checksums.is_empty() {
            self.status = FileReplicaStatus::Unchecked;
            return;
        }
        let values: Vec<&String> = self.checksums.values().collect();
        let first = values[0];
        if values.iter().all(|v| *v == first) {
            self.status = FileReplicaStatus::Consistent;
        } else {
            self.status = FileReplicaStatus::Diverged;
        }
    }

    /// Returns the number of locations where this file has been verified.
    #[must_use]
    pub fn location_count(&self) -> usize {
        self.checksums.len()
    }

    /// Returns whether the file is consistent across all locations.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.status == FileReplicaStatus::Consistent
    }
}

/// Recommended repair action for a diverged file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairAction {
    /// Copy the file from the primary to the named location.
    CopyFromPrimary {
        /// Target location to repair.
        target: String,
    },
    /// Re-verify before taking action.
    ReVerify {
        /// File path to re-verify.
        path: String,
    },
    /// Manual intervention required.
    ManualReview {
        /// Reason for manual review.
        reason: String,
    },
}

/// Summary of a replication verification run.
#[derive(Debug, Clone, Default)]
pub struct VerificationSummary {
    /// Total files checked.
    pub total_files: usize,
    /// Files that are consistent.
    pub consistent: usize,
    /// Files that have diverged.
    pub diverged: usize,
    /// Files that are missing from one or more locations.
    pub missing: usize,
    /// Files that encountered errors.
    pub errors: usize,
    /// Total bytes verified.
    pub total_bytes: u64,
}

impl VerificationSummary {
    /// Returns whether all files are consistent.
    #[must_use]
    pub fn all_consistent(&self) -> bool {
        self.diverged == 0 && self.missing == 0 && self.errors == 0
    }

    /// Returns the consistency ratio as a percentage (0.0 to 100.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn consistency_percent(&self) -> f64 {
        if self.total_files == 0 {
            return 100.0;
        }
        (self.consistent as f64 / self.total_files as f64) * 100.0
    }
}

/// Manages replication verification across multiple locations.
#[derive(Debug, Default)]
pub struct ReplicationVerifier {
    /// Registered replica locations.
    locations: Vec<ReplicaLocation>,
    /// File replica information.
    files: Vec<FileReplicaInfo>,
}

impl ReplicationVerifier {
    /// Creates a new verifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a replica location.
    pub fn add_location(&mut self, location: ReplicaLocation) {
        self.locations.push(location);
    }

    /// Adds a file entry for verification.
    pub fn add_file(&mut self, info: FileReplicaInfo) {
        self.files.push(info);
    }

    /// Returns the number of registered locations.
    #[must_use]
    pub fn location_count(&self) -> usize {
        self.locations.len()
    }

    /// Returns the number of tracked files.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Returns the primary location, if any.
    #[must_use]
    pub fn primary_location(&self) -> Option<&ReplicaLocation> {
        self.locations.iter().find(|l| l.is_primary)
    }

    /// Evaluates all files and returns a summary.
    pub fn verify_all(&mut self) -> VerificationSummary {
        let mut summary = VerificationSummary::default();
        for file in &mut self.files {
            file.evaluate();
            summary.total_files += 1;
            summary.total_bytes += file.sizes.values().next().copied().unwrap_or(0);
            match file.status {
                FileReplicaStatus::Consistent => summary.consistent += 1,
                FileReplicaStatus::Diverged => summary.diverged += 1,
                FileReplicaStatus::Missing => summary.missing += 1,
                FileReplicaStatus::Error => summary.errors += 1,
                FileReplicaStatus::Unchecked => {}
            }
        }
        summary
    }

    /// Returns repair actions for all diverged files.
    #[must_use]
    pub fn recommend_repairs(&self) -> Vec<RepairAction> {
        let primary_name = self
            .primary_location()
            .map(|l| l.name.clone())
            .unwrap_or_default();

        let mut actions = Vec::new();
        for file in &self.files {
            if file.status != FileReplicaStatus::Diverged {
                continue;
            }
            let primary_checksum = file.checksums.get(&primary_name);
            for (loc_name, checksum) in &file.checksums {
                if loc_name == &primary_name {
                    continue;
                }
                if primary_checksum.is_some() && Some(checksum) != primary_checksum {
                    actions.push(RepairAction::CopyFromPrimary {
                        target: loc_name.clone(),
                    });
                }
            }
        }
        actions
    }

    /// Returns all diverged file paths.
    #[must_use]
    pub fn diverged_files(&self) -> Vec<&str> {
        self.files
            .iter()
            .filter(|f| f.status == FileReplicaStatus::Diverged)
            .map(|f| f.path.as_str())
            .collect()
    }

    /// Returns all consistent file paths.
    #[must_use]
    pub fn consistent_files(&self) -> Vec<&str> {
        self.files
            .iter()
            .filter(|f| f.status == FileReplicaStatus::Consistent)
            .map(|f| f.path.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replica_location_primary() {
        let loc = ReplicaLocation::primary("dc1", "s3://archive-dc1");
        assert!(loc.is_primary);
        assert_eq!(loc.name, "dc1");
    }

    #[test]
    fn test_replica_location_secondary() {
        let loc = ReplicaLocation::secondary("dc2", "s3://archive-dc2");
        assert!(!loc.is_primary);
    }

    #[test]
    fn test_file_replica_info_consistent() {
        let mut info = FileReplicaInfo::new("video/clip001.mkv");
        info.record_location("dc1", "abc123", 1000);
        info.record_location("dc2", "abc123", 1000);
        info.evaluate();
        assert!(info.is_consistent());
        assert_eq!(info.location_count(), 2);
    }

    #[test]
    fn test_file_replica_info_diverged() {
        let mut info = FileReplicaInfo::new("video/clip002.mkv");
        info.record_location("dc1", "abc123", 1000);
        info.record_location("dc2", "def456", 1000);
        info.evaluate();
        assert_eq!(info.status, FileReplicaStatus::Diverged);
        assert!(!info.is_consistent());
    }

    #[test]
    fn test_file_replica_info_unchecked() {
        let mut info = FileReplicaInfo::new("video/clip003.mkv");
        info.evaluate();
        assert_eq!(info.status, FileReplicaStatus::Unchecked);
    }

    #[test]
    fn test_verification_summary_all_consistent() {
        let summary = VerificationSummary {
            total_files: 10,
            consistent: 10,
            ..Default::default()
        };
        assert!(summary.all_consistent());
        assert!((summary.consistency_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_verification_summary_with_issues() {
        let summary = VerificationSummary {
            total_files: 10,
            consistent: 7,
            diverged: 2,
            missing: 1,
            ..Default::default()
        };
        assert!(!summary.all_consistent());
        assert!((summary.consistency_percent() - 70.0).abs() < 0.01);
    }

    #[test]
    fn test_verification_summary_empty() {
        let summary = VerificationSummary::default();
        assert!(summary.all_consistent());
        assert!((summary.consistency_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_verifier_add_locations() {
        let mut v = ReplicationVerifier::new();
        v.add_location(ReplicaLocation::primary("dc1", "/mnt/dc1"));
        v.add_location(ReplicaLocation::secondary("dc2", "/mnt/dc2"));
        assert_eq!(v.location_count(), 2);
        assert_eq!(
            v.primary_location().expect("operation should succeed").name,
            "dc1"
        );
    }

    #[test]
    fn test_verifier_verify_all() {
        let mut v = ReplicationVerifier::new();
        v.add_location(ReplicaLocation::primary("dc1", "/mnt/dc1"));
        v.add_location(ReplicaLocation::secondary("dc2", "/mnt/dc2"));

        let mut f1 = FileReplicaInfo::new("a.mkv");
        f1.record_location("dc1", "aaa", 100);
        f1.record_location("dc2", "aaa", 100);
        v.add_file(f1);

        let mut f2 = FileReplicaInfo::new("b.mkv");
        f2.record_location("dc1", "bbb", 200);
        f2.record_location("dc2", "ccc", 200);
        v.add_file(f2);

        let summary = v.verify_all();
        assert_eq!(summary.total_files, 2);
        assert_eq!(summary.consistent, 1);
        assert_eq!(summary.diverged, 1);
    }

    #[test]
    fn test_verifier_recommend_repairs() {
        let mut v = ReplicationVerifier::new();
        v.add_location(ReplicaLocation::primary("dc1", "/mnt/dc1"));
        v.add_location(ReplicaLocation::secondary("dc2", "/mnt/dc2"));

        let mut f = FileReplicaInfo::new("c.mkv");
        f.record_location("dc1", "good", 100);
        f.record_location("dc2", "bad", 100);
        f.evaluate();
        v.add_file(f);

        let repairs = v.recommend_repairs();
        assert_eq!(repairs.len(), 1);
        assert!(matches!(&repairs[0], RepairAction::CopyFromPrimary { target } if target == "dc2"));
    }

    #[test]
    fn test_verifier_diverged_consistent_files() {
        let mut v = ReplicationVerifier::new();
        v.add_location(ReplicaLocation::primary("p", "/p"));

        let mut f1 = FileReplicaInfo::new("ok.mkv");
        f1.record_location("p", "x", 10);
        f1.evaluate();
        v.add_file(f1);

        let mut f2 = FileReplicaInfo::new("bad.mkv");
        f2.record_location("p", "x", 10);
        f2.record_location("s", "y", 10);
        f2.evaluate();
        v.add_file(f2);

        assert_eq!(v.consistent_files(), vec!["ok.mkv"]);
        assert_eq!(v.diverged_files(), vec!["bad.mkv"]);
    }

    #[test]
    fn test_repair_action_variants() {
        let copy = RepairAction::CopyFromPrimary {
            target: "dc2".to_string(),
        };
        let verify = RepairAction::ReVerify {
            path: "test.mkv".to_string(),
        };
        let manual = RepairAction::ManualReview {
            reason: "unknown".to_string(),
        };
        assert_ne!(copy, verify);
        assert_ne!(verify, manual);
    }
}
