//! Cloud storage provider abstraction.
//!
//! Provides provider-agnostic types for object metadata, storage class
//! classification, and lifecycle rule evaluation.

#![allow(dead_code)]

/// The storage class (durability/access tier) of an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// Frequently accessed data; highest cost, lowest latency.
    Standard,
    /// Infrequently accessed data; lower cost, slightly higher retrieval latency.
    InfrequentAccess,
    /// Rarely accessed data; low cost, minutes-to-hours retrieval.
    Archive,
    /// Long-term archival; lowest cost, hours retrieval time.
    DeepArchive,
}

impl StorageClass {
    /// Human-readable description of retrieval time for this class.
    #[must_use]
    pub fn retrieval_time_description(&self) -> &str {
        match self {
            StorageClass::Standard => "Immediate",
            StorageClass::InfrequentAccess => "Milliseconds",
            StorageClass::Archive => "Minutes to hours",
            StorageClass::DeepArchive => "12 or more hours",
        }
    }

    /// Minimum number of days an object must stay in this storage class
    /// before it can be deleted or transitioned without incurring fees.
    #[must_use]
    pub fn min_storage_days(&self) -> u32 {
        match self {
            StorageClass::Standard => 0,
            StorageClass::InfrequentAccess => 30,
            StorageClass::Archive => 90,
            StorageClass::DeepArchive => 180,
        }
    }

    /// Returns `true` when this class is archival (Archive or DeepArchive).
    #[must_use]
    pub fn is_archival(&self) -> bool {
        matches!(self, StorageClass::Archive | StorageClass::DeepArchive)
    }
}

/// Metadata describing a single object in cloud object storage.
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    /// Object key (path within the bucket).
    pub key: String,
    /// Size of the object in bytes.
    pub size_bytes: u64,
    /// MIME content type.
    pub content_type: String,
    /// ETag / content hash (typically an MD5 hex string).
    pub etag: String,
    /// Unix epoch milliseconds of the last modification.
    pub last_modified_ms: u64,
    /// Storage class assigned to this object.
    pub storage_class: StorageClass,
}

impl ObjectMetadata {
    /// Create a new `ObjectMetadata`.
    #[must_use]
    pub fn new(
        key: impl Into<String>,
        size_bytes: u64,
        content_type: impl Into<String>,
        etag: impl Into<String>,
        last_modified_ms: u64,
        storage_class: StorageClass,
    ) -> Self {
        Self {
            key: key.into(),
            size_bytes,
            content_type: content_type.into(),
            etag: etag.into(),
            last_modified_ms,
            storage_class,
        }
    }

    /// Number of whole days since `last_modified_ms` as of `now_ms`.
    ///
    /// Returns 0 when `now_ms < last_modified_ms`.
    #[must_use]
    pub fn age_days(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.last_modified_ms) / (24 * 60 * 60 * 1_000)
    }

    /// Returns `true` when the object is in an archival storage class.
    #[must_use]
    pub fn is_archival(&self) -> bool {
        self.storage_class.is_archival()
    }
}

/// A lifecycle rule that transitions objects matching a prefix after a given age.
#[derive(Debug, Clone)]
pub struct LifecycleRule {
    /// Key prefix that this rule applies to.
    pub prefix: String,
    /// Target storage class after transition.
    pub transition_to: StorageClass,
    /// Minimum object age in days before the transition occurs.
    pub after_days: u32,
}

impl LifecycleRule {
    /// Create a new `LifecycleRule`.
    #[must_use]
    pub fn new(prefix: impl Into<String>, transition_to: StorageClass, after_days: u32) -> Self {
        Self {
            prefix: prefix.into(),
            transition_to,
            after_days,
        }
    }

    /// Returns `true` when this rule applies to the given object key at `age_days`.
    #[must_use]
    pub fn applies_to(&self, key: &str, age_days: u32) -> bool {
        key.starts_with(&self.prefix) && age_days >= self.after_days
    }
}

/// Storage configuration for a bucket, including its lifecycle rules.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Bucket name.
    pub bucket: String,
    /// Cloud region.
    pub region: String,
    /// Lifecycle rules evaluated in order.
    pub lifecycle_rules: Vec<LifecycleRule>,
}

impl StorageConfig {
    /// Create a new `StorageConfig` with no lifecycle rules.
    #[must_use]
    pub fn new(bucket: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            region: region.into(),
            lifecycle_rules: Vec::new(),
        }
    }

    /// Add a lifecycle rule.
    pub fn add_rule(&mut self, rule: LifecycleRule) {
        self.lifecycle_rules.push(rule);
    }

    /// Find the first lifecycle rule that applies to `key` at `age_days`.
    #[must_use]
    pub fn find_rule_for_object(&self, key: &str, age_days: u32) -> Option<&LifecycleRule> {
        self.lifecycle_rules
            .iter()
            .find(|r| r.applies_to(key, age_days))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. StorageClass::retrieval_time_description
    #[test]
    fn test_retrieval_time_descriptions() {
        assert_eq!(
            StorageClass::Standard.retrieval_time_description(),
            "Immediate"
        );
        assert_eq!(
            StorageClass::InfrequentAccess.retrieval_time_description(),
            "Milliseconds"
        );
        assert!(StorageClass::Archive
            .retrieval_time_description()
            .contains("hours"));
        assert!(StorageClass::DeepArchive
            .retrieval_time_description()
            .contains("hours"));
    }

    // 2. StorageClass::min_storage_days
    #[test]
    fn test_min_storage_days() {
        assert_eq!(StorageClass::Standard.min_storage_days(), 0);
        assert_eq!(StorageClass::InfrequentAccess.min_storage_days(), 30);
        assert_eq!(StorageClass::Archive.min_storage_days(), 90);
        assert_eq!(StorageClass::DeepArchive.min_storage_days(), 180);
    }

    // 3. StorageClass::is_archival
    #[test]
    fn test_storage_class_is_archival() {
        assert!(!StorageClass::Standard.is_archival());
        assert!(!StorageClass::InfrequentAccess.is_archival());
        assert!(StorageClass::Archive.is_archival());
        assert!(StorageClass::DeepArchive.is_archival());
    }

    // 4. ObjectMetadata::age_days – basic
    #[test]
    fn test_object_age_days_basic() {
        let one_day_ms: u64 = 24 * 60 * 60 * 1_000;
        let obj = ObjectMetadata::new("key", 100, "video/mp4", "abc", 0, StorageClass::Standard);
        assert_eq!(obj.age_days(one_day_ms * 5), 5);
    }

    // 5. ObjectMetadata::age_days – saturates to 0
    #[test]
    fn test_object_age_days_past() {
        let obj = ObjectMetadata::new(
            "key",
            100,
            "video/mp4",
            "abc",
            10_000,
            StorageClass::Standard,
        );
        assert_eq!(obj.age_days(5_000), 0);
    }

    // 6. ObjectMetadata::is_archival – standard returns false
    #[test]
    fn test_object_is_archival_standard() {
        let obj = ObjectMetadata::new("key", 0, "", "", 0, StorageClass::Standard);
        assert!(!obj.is_archival());
    }

    // 7. ObjectMetadata::is_archival – archive returns true
    #[test]
    fn test_object_is_archival_archive() {
        let obj = ObjectMetadata::new("key", 0, "", "", 0, StorageClass::Archive);
        assert!(obj.is_archival());
    }

    // 8. ObjectMetadata::is_archival – deep archive returns true
    #[test]
    fn test_object_is_archival_deep() {
        let obj = ObjectMetadata::new("key", 0, "", "", 0, StorageClass::DeepArchive);
        assert!(obj.is_archival());
    }

    // 9. LifecycleRule::applies_to – matches prefix and age
    #[test]
    fn test_lifecycle_rule_applies() {
        let rule = LifecycleRule::new("videos/", StorageClass::Archive, 30);
        assert!(rule.applies_to("videos/movie.mp4", 30));
        assert!(rule.applies_to("videos/movie.mp4", 60));
    }

    // 10. LifecycleRule::applies_to – wrong prefix
    #[test]
    fn test_lifecycle_rule_wrong_prefix() {
        let rule = LifecycleRule::new("videos/", StorageClass::Archive, 30);
        assert!(!rule.applies_to("images/photo.jpg", 60));
    }

    // 11. LifecycleRule::applies_to – too young
    #[test]
    fn test_lifecycle_rule_too_young() {
        let rule = LifecycleRule::new("videos/", StorageClass::Archive, 30);
        assert!(!rule.applies_to("videos/movie.mp4", 10));
    }

    // 12. StorageConfig::find_rule_for_object – finds first matching rule
    #[test]
    fn test_storage_config_find_rule() {
        let mut cfg = StorageConfig::new("my-bucket", "us-east-1");
        cfg.add_rule(LifecycleRule::new(
            "logs/",
            StorageClass::InfrequentAccess,
            7,
        ));
        cfg.add_rule(LifecycleRule::new("videos/", StorageClass::Archive, 90));
        let rule = cfg.find_rule_for_object("videos/clip.mp4", 91);
        assert!(rule.is_some());
        assert_eq!(
            rule.expect("test expectation failed").transition_to,
            StorageClass::Archive
        );
    }

    // 13. StorageConfig::find_rule_for_object – no match returns None
    #[test]
    fn test_storage_config_find_rule_none() {
        let mut cfg = StorageConfig::new("my-bucket", "us-east-1");
        cfg.add_rule(LifecycleRule::new("videos/", StorageClass::Archive, 90));
        assert!(cfg.find_rule_for_object("videos/clip.mp4", 30).is_none());
    }

    // 14. StorageConfig::find_rule_for_object – empty rules returns None
    #[test]
    fn test_storage_config_no_rules() {
        let cfg = StorageConfig::new("bucket", "eu-west-1");
        assert!(cfg.find_rule_for_object("any/key", 365).is_none());
    }

    // 15. StorageConfig::find_rule_for_object – first matching rule wins
    #[test]
    fn test_storage_config_first_rule_wins() {
        let mut cfg = StorageConfig::new("b", "r");
        cfg.add_rule(LifecycleRule::new("a/", StorageClass::InfrequentAccess, 10));
        cfg.add_rule(LifecycleRule::new("a/", StorageClass::Archive, 30));
        let rule = cfg.find_rule_for_object("a/file.bin", 40);
        assert!(rule.is_some());
        assert_eq!(
            rule.expect("test expectation failed").transition_to,
            StorageClass::InfrequentAccess
        );
    }
}
