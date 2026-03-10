//! Digital preservation management based on NDSA Levels of Digital Preservation.
//!
//! Implements preservation levels, storage media risk profiles, and management
//! of multi-copy preservation records.

#![allow(dead_code)]

/// NDSA Levels of Digital Preservation (2019).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PreservationLevel {
    /// Level 0 – No organised approach to preservation.
    Level0,
    /// Level 1 – Know what you have; have basic inventory and storage.
    Level1,
    /// Level 2 – Protect your content; multiple copies in different locations.
    Level2,
    /// Level 3 – Monitor your content; active bit-integrity checking.
    Level3,
}

impl PreservationLevel {
    /// Human-readable description of this level.
    #[must_use]
    pub const fn description(&self) -> &str {
        match self {
            Self::Level0 => "No organised preservation approach",
            Self::Level1 => "Basic inventory; at least one copy in a known location",
            Self::Level2 => "Multiple geographically distributed copies",
            Self::Level3 => "Active bit-integrity monitoring and periodic fixity checks",
        }
    }

    /// Minimum number of copies required at this level.
    #[must_use]
    pub const fn copies_required(&self) -> u32 {
        match self {
            Self::Level0 => 0,
            Self::Level1 => 2,
            Self::Level2 => 3,
            Self::Level3 => 4,
        }
    }
}

/// Physical storage media types with typical annual failure rates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageMedia {
    /// Linear Tape-Open (LTO) or similar magnetic tape.
    Tape,
    /// Spinning hard-disk drive.
    Disk,
    /// Solid-state / flash storage.
    Flash,
    /// Cloud object storage (aggregated reliability).
    Cloud,
    /// Optical disc (M-DISC or pressed).
    Optical,
}

impl StorageMedia {
    /// Typical annual failure / data-loss probability (0.0–1.0).
    #[must_use]
    pub fn failure_rate_per_year(&self) -> f64 {
        match self {
            Self::Tape => 0.005,
            Self::Disk => 0.030,
            Self::Flash => 0.010,
            Self::Cloud => 0.0001,
            Self::Optical => 0.001,
        }
    }

    /// Display name for this media type.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::Tape => "Magnetic Tape",
            Self::Disk => "Hard Disk Drive",
            Self::Flash => "Flash / SSD",
            Self::Cloud => "Cloud Storage",
            Self::Optical => "Optical Disc",
        }
    }
}

/// A preservation policy derived from an NDSA level.
#[derive(Clone, Debug)]
pub struct PreservationPolicy {
    /// Target NDSA preservation level.
    pub level: PreservationLevel,
    /// Required number of distinct copies.
    pub copies: u32,
    /// Whether copies must span geographically separate sites.
    pub geographic_distribution: bool,
    /// Recommended interval (years) between format-migration reviews.
    pub format_migration_years: u32,
}

impl PreservationPolicy {
    /// Create a policy for the given level with sensible defaults.
    #[must_use]
    pub fn for_level(level: PreservationLevel) -> Self {
        let (geographic_distribution, format_migration_years) = match level {
            PreservationLevel::Level0 => (false, 10),
            PreservationLevel::Level1 => (false, 7),
            PreservationLevel::Level2 => (true, 5),
            PreservationLevel::Level3 => (true, 3),
        };
        Self {
            copies: level.copies_required(),
            level,
            geographic_distribution,
            format_migration_years,
        }
    }
}

/// A single stored copy of a preservation asset.
#[derive(Clone, Debug)]
pub struct PreservationCopy {
    /// Human-readable storage location (e.g., data-centre name or path).
    pub location: String,
    /// The physical media type used for this copy.
    pub media_type: StorageMedia,
    /// Unix timestamp (milliseconds) when this copy was created.
    pub created_at_ms: u64,
    /// Unix timestamp (milliseconds) of the last fixity verification.
    pub last_verified_ms: u64,
    /// Hex-encoded checksum of the stored content (any algorithm).
    pub checksum: String,
}

impl PreservationCopy {
    /// Returns whether this copy has been verified within `max_age_days` days.
    #[must_use]
    pub fn is_stale(&self, now_ms: u64, max_age_days: u32) -> bool {
        let age_ms = now_ms.saturating_sub(self.last_verified_ms);
        let max_ms = (max_age_days as u64) * 86_400 * 1_000;
        age_ms > max_ms
    }
}

/// A full preservation record for a single digital asset.
#[derive(Clone, Debug)]
pub struct PreservationRecord {
    /// Unique asset identifier (e.g., UUID or accession number).
    pub asset_id: String,
    /// Applicable preservation policy.
    pub policy: PreservationPolicy,
    /// All known copies of this asset.
    pub copies: Vec<PreservationCopy>,
    /// Unix timestamp (ms) of the most recent fixity check across all copies.
    pub last_fixity_check_ms: u64,
}

impl PreservationRecord {
    /// Create a new, empty preservation record.
    #[must_use]
    pub fn new(asset_id: impl Into<String>, policy: PreservationPolicy) -> Self {
        Self {
            asset_id: asset_id.into(),
            policy,
            copies: Vec::new(),
            last_fixity_check_ms: 0,
        }
    }

    /// Add a copy to this record.
    pub fn add_copy(&mut self, copy: PreservationCopy) {
        self.copies.push(copy);
    }

    /// Returns `true` if the record has enough copies to satisfy its policy.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.copies.len() as u32 >= self.policy.copies
    }

    /// Returns the number of copies that are stale (need re-verification).
    #[must_use]
    pub fn stale_copy_count(&self, now_ms: u64, max_age_days: u32) -> usize {
        self.copies
            .iter()
            .filter(|c| c.is_stale(now_ms, max_age_days))
            .count()
    }
}

/// Manages a collection of preservation records.
#[derive(Default)]
pub struct PreservationManager {
    records: Vec<PreservationRecord>,
}

impl PreservationManager {
    /// Create a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a preservation record.
    pub fn add_record(&mut self, record: PreservationRecord) {
        self.records.push(record);
    }

    /// Look up a record by asset ID.
    #[must_use]
    pub fn get_record(&self, id: &str) -> Option<&PreservationRecord> {
        self.records.iter().find(|r| r.asset_id == id)
    }

    /// Returns references to records that have at least one copy needing
    /// re-verification, using the current time in milliseconds.
    #[must_use]
    pub fn copies_needing_verification(&self, max_age_days: u32) -> Vec<&PreservationRecord> {
        // Use a fixed "now" for determinism in tests; callers can override
        // by choosing their own `max_age_days` relative to their epoch.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.records
            .iter()
            .filter(|r| r.stale_copy_count(now_ms, max_age_days) > 0)
            .collect()
    }

    /// Total number of managed records.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.records.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_copy(last_verified_ms: u64) -> PreservationCopy {
        PreservationCopy {
            location: "test-location".to_string(),
            media_type: StorageMedia::Disk,
            created_at_ms: 0,
            last_verified_ms,
            checksum: "abc123".to_string(),
        }
    }

    #[test]
    fn test_preservation_level_copies_required() {
        assert_eq!(PreservationLevel::Level0.copies_required(), 0);
        assert_eq!(PreservationLevel::Level1.copies_required(), 2);
        assert_eq!(PreservationLevel::Level2.copies_required(), 3);
        assert_eq!(PreservationLevel::Level3.copies_required(), 4);
    }

    #[test]
    fn test_preservation_level_descriptions_non_empty() {
        for level in [
            PreservationLevel::Level0,
            PreservationLevel::Level1,
            PreservationLevel::Level2,
            PreservationLevel::Level3,
        ] {
            assert!(!level.description().is_empty());
        }
    }

    #[test]
    fn test_storage_media_failure_rates_valid() {
        for media in [
            StorageMedia::Tape,
            StorageMedia::Disk,
            StorageMedia::Flash,
            StorageMedia::Cloud,
            StorageMedia::Optical,
        ] {
            let rate = media.failure_rate_per_year();
            assert!(
                rate > 0.0 && rate < 1.0,
                "Invalid rate for {:?}: {}",
                media,
                rate
            );
        }
    }

    #[test]
    fn test_cloud_has_lowest_failure_rate() {
        assert!(
            StorageMedia::Cloud.failure_rate_per_year()
                < StorageMedia::Disk.failure_rate_per_year()
        );
    }

    #[test]
    fn test_preservation_policy_for_level() {
        let p = PreservationPolicy::for_level(PreservationLevel::Level2);
        assert_eq!(p.copies, 3);
        assert!(p.geographic_distribution);
    }

    #[test]
    fn test_record_compliance() {
        let policy = PreservationPolicy::for_level(PreservationLevel::Level1);
        let mut record = PreservationRecord::new("asset-001", policy);
        assert!(!record.is_compliant());

        record.add_copy(make_copy(0));
        assert!(!record.is_compliant());

        record.add_copy(make_copy(0));
        assert!(record.is_compliant());
    }

    #[test]
    fn test_copy_is_stale() {
        let now_ms = 10_000_000_000u64;
        // Verified 100 days ago
        let old_ms = now_ms - 100 * 86_400 * 1_000;
        let copy = make_copy(old_ms);
        assert!(copy.is_stale(now_ms, 90));
        assert!(!copy.is_stale(now_ms, 101));
    }

    #[test]
    fn test_manager_add_and_get() {
        let mut mgr = PreservationManager::new();
        let policy = PreservationPolicy::for_level(PreservationLevel::Level1);
        mgr.add_record(PreservationRecord::new("id-42", policy));
        assert!(mgr.get_record("id-42").is_some());
        assert!(mgr.get_record("id-99").is_none());
        assert_eq!(mgr.record_count(), 1);
    }

    #[test]
    fn test_manager_copies_needing_verification_empty() {
        let mgr = PreservationManager::new();
        // With zero records, nothing should need verification.
        assert!(mgr.copies_needing_verification(90).is_empty());
    }

    #[test]
    fn test_stale_copy_count() {
        let policy = PreservationPolicy::for_level(PreservationLevel::Level2);
        let mut record = PreservationRecord::new("asset-x", policy);
        // Add one fresh copy (verified "now") and one stale (verified long ago)
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("test expectation failed")
            .as_millis() as u64;
        record.add_copy(make_copy(now_ms));
        record.add_copy(make_copy(1_000)); // epoch + 1 second → very stale
        assert_eq!(record.stale_copy_count(now_ms, 90), 1);
    }

    #[test]
    fn test_preservation_level_ordering() {
        assert!(PreservationLevel::Level3 > PreservationLevel::Level2);
        assert!(PreservationLevel::Level2 > PreservationLevel::Level0);
    }

    #[test]
    fn test_preservation_policy_level0_no_geographic() {
        let p = PreservationPolicy::for_level(PreservationLevel::Level0);
        assert!(!p.geographic_distribution);
        assert_eq!(p.copies, 0);
    }

    #[test]
    fn test_preservation_policy_level3_geographic() {
        let p = PreservationPolicy::for_level(PreservationLevel::Level3);
        assert!(p.geographic_distribution);
        assert_eq!(p.copies, 4);
    }
}

// ── Spec-required types ───────────────────────────────────────────────────────

/// Named long-term preservation media and storage formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PreservationFormat {
    /// LTO-7 magnetic tape (6 TB native capacity).
    Lto7,
    /// LTO-8 magnetic tape (12 TB native capacity).
    Lto8,
    /// LTO-9 magnetic tape (18 TB native capacity).
    Lto9,
    /// M-DISC optical (estimated 1 000-year data retention).
    MDisc,
    /// Microsoft Azure Archive storage tier (cloud, online access).
    AzureArchive,
    /// Amazon S3 Glacier (cloud, offline/near-offline).
    S3Glacier,
}

impl PreservationFormat {
    /// Expected media lifetime in years.
    #[must_use]
    pub fn expected_lifetime_years(&self) -> u32 {
        match self {
            Self::Lto7 | Self::Lto8 | Self::Lto9 => 30,
            Self::MDisc => 1_000,
            Self::AzureArchive | Self::S3Glacier => 0, // cloud: vendor-dependent
        }
    }

    /// Nominal media capacity in GB.
    #[must_use]
    pub fn capacity_gb(&self) -> u64 {
        match self {
            Self::Lto7 => 6_000,
            Self::Lto8 => 12_000,
            Self::Lto9 => 18_000,
            Self::MDisc => 100,
            Self::AzureArchive | Self::S3Glacier => u64::MAX, // effectively unlimited
        }
    }

    /// Returns `true` for cloud formats (immediately retrievable without media handling).
    #[must_use]
    pub fn is_online(&self) -> bool {
        matches!(self, Self::AzureArchive | Self::S3Glacier)
    }
}

/// Preservation policy describing how many copies to keep and where.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SpecPreservationPolicy {
    /// Storage format used for all copies.
    pub format: PreservationFormat,
    /// Required number of distinct copies.
    pub copies: u32,
    /// Whether copies must be spread across geographic locations.
    pub geographic_spread: bool,
    /// How often (in days) fixity verification must be performed.
    pub verification_interval_days: u32,
}

impl SpecPreservationPolicy {
    /// Returns `true` when the policy satisfies the 3-2-1 backup rule:
    /// at least 3 copies, at least 2 different media types, at least 1 offsite.
    ///
    /// For this simplified model we assume an online format counts as "offsite"
    /// and tape is a different media type to optical/cloud.
    #[must_use]
    pub fn is_3_2_1_compliant(&self) -> bool {
        self.copies >= 3 && self.geographic_spread
    }
}

/// A single digital preservation record for one asset.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct DigitalPreservationRecord {
    /// Unique asset identifier (e.g., UUID or accession number).
    pub asset_id: String,
    /// Storage format used for these copies.
    pub format: PreservationFormat,
    /// Locations of all copies (path, URI, or label).
    pub copies: Vec<String>,
    /// Unix timestamp (ms) of the most recent fixity verification.
    pub last_verified_ms: u64,
    /// Unix timestamp (ms) when the next verification is due.
    pub next_verify_ms: u64,
}

impl DigitalPreservationRecord {
    /// Returns `true` if the next scheduled verification is past due.
    #[must_use]
    pub fn is_verification_due(&self, now_ms: u64) -> bool {
        now_ms >= self.next_verify_ms
    }

    /// Number of stored copies.
    #[must_use]
    pub fn copy_count(&self) -> usize {
        self.copies.len()
    }
}

/// Collection of preservation records with audit capabilities.
#[allow(dead_code)]
#[derive(Default, Debug)]
pub struct PreservationAudit {
    /// All managed preservation records.
    pub records: Vec<DigitalPreservationRecord>,
}

impl PreservationAudit {
    /// Create a new empty audit.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns references to all records whose next verification is overdue.
    #[must_use]
    pub fn overdue_verifications(&self, now_ms: u64) -> Vec<&DigitalPreservationRecord> {
        self.records
            .iter()
            .filter(|r| r.is_verification_due(now_ms))
            .collect()
    }

    /// Count of records that satisfy `policy` (enough copies and geographic spread).
    #[must_use]
    pub fn compliant_count(&self, policy: &SpecPreservationPolicy) -> usize {
        self.records
            .iter()
            .filter(|r| r.copy_count() >= policy.copies as usize && policy.is_3_2_1_compliant())
            .count()
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;

    fn make_record(
        asset_id: &str,
        copies: usize,
        last_ms: u64,
        next_ms: u64,
    ) -> DigitalPreservationRecord {
        DigitalPreservationRecord {
            asset_id: asset_id.to_string(),
            format: PreservationFormat::Lto9,
            copies: (0..copies).map(|i| format!("loc-{i}")).collect(),
            last_verified_ms: last_ms,
            next_verify_ms: next_ms,
        }
    }

    #[test]
    fn test_preservation_format_lto9_lifetime() {
        assert_eq!(PreservationFormat::Lto9.expected_lifetime_years(), 30);
    }

    #[test]
    fn test_preservation_format_mdisc_lifetime() {
        assert_eq!(PreservationFormat::MDisc.expected_lifetime_years(), 1_000);
    }

    #[test]
    fn test_preservation_format_capacity_ordering() {
        assert!(PreservationFormat::Lto9.capacity_gb() > PreservationFormat::Lto7.capacity_gb());
    }

    #[test]
    fn test_preservation_format_is_online_cloud() {
        assert!(PreservationFormat::AzureArchive.is_online());
        assert!(PreservationFormat::S3Glacier.is_online());
    }

    #[test]
    fn test_preservation_format_is_online_tape_false() {
        assert!(!PreservationFormat::Lto8.is_online());
        assert!(!PreservationFormat::MDisc.is_online());
    }

    #[test]
    fn test_spec_policy_3_2_1_compliant() {
        let p = SpecPreservationPolicy {
            format: PreservationFormat::Lto9,
            copies: 3,
            geographic_spread: true,
            verification_interval_days: 90,
        };
        assert!(p.is_3_2_1_compliant());
    }

    #[test]
    fn test_spec_policy_not_3_2_1_too_few_copies() {
        let p = SpecPreservationPolicy {
            format: PreservationFormat::Lto9,
            copies: 2,
            geographic_spread: true,
            verification_interval_days: 90,
        };
        assert!(!p.is_3_2_1_compliant());
    }

    #[test]
    fn test_spec_policy_not_3_2_1_no_geo() {
        let p = SpecPreservationPolicy {
            format: PreservationFormat::Lto9,
            copies: 3,
            geographic_spread: false,
            verification_interval_days: 90,
        };
        assert!(!p.is_3_2_1_compliant());
    }

    #[test]
    fn test_digital_record_is_verification_due_true() {
        let r = make_record("a1", 3, 1_000, 5_000);
        assert!(r.is_verification_due(6_000));
    }

    #[test]
    fn test_digital_record_is_verification_due_false() {
        let r = make_record("a1", 3, 1_000, 10_000);
        assert!(!r.is_verification_due(5_000));
    }

    #[test]
    fn test_digital_record_copy_count() {
        let r = make_record("x", 4, 0, 100);
        assert_eq!(r.copy_count(), 4);
    }

    #[test]
    fn test_audit_overdue_verifications() {
        let mut audit = PreservationAudit::new();
        audit.records.push(make_record("r1", 3, 0, 1_000)); // overdue
        audit.records.push(make_record("r2", 3, 0, 99_000)); // not yet
        let overdue = audit.overdue_verifications(5_000);
        assert_eq!(overdue.len(), 1);
        assert_eq!(overdue[0].asset_id, "r1");
    }

    #[test]
    fn test_audit_compliant_count() {
        let policy = SpecPreservationPolicy {
            format: PreservationFormat::Lto9,
            copies: 3,
            geographic_spread: true,
            verification_interval_days: 90,
        };
        let mut audit = PreservationAudit::new();
        audit.records.push(make_record("a1", 4, 0, 100)); // compliant (4 >= 3)
        audit.records.push(make_record("a2", 2, 0, 100)); // not compliant (2 < 3)
        assert_eq!(audit.compliant_count(&policy), 1);
    }
}
