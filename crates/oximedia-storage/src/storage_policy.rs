#![allow(dead_code)]
//! Storage policy management — access classes, retention rules, and policy evaluation.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Storage class representing access frequency tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// Frequently accessed data (low latency, higher cost).
    Hot,
    /// Infrequently accessed data (moderate latency, lower cost).
    Warm,
    /// Rarely accessed data (high latency, lowest cost).
    Cold,
    /// Long-term archival with rare retrieval.
    Archive,
    /// Instantly-accessible deep archive.
    DeepArchive,
}

impl StorageClass {
    /// Returns the expected access frequency in accesses per month (approximate).
    pub fn access_frequency(&self) -> u32 {
        match self {
            StorageClass::Hot => 1000,
            StorageClass::Warm => 100,
            StorageClass::Cold => 10,
            StorageClass::Archive => 1,
            StorageClass::DeepArchive => 0,
        }
    }

    /// Returns the relative cost multiplier compared to Cold storage (1.0 baseline).
    #[allow(clippy::cast_precision_loss)]
    pub fn cost_multiplier(&self) -> f64 {
        match self {
            StorageClass::Hot => 4.0,
            StorageClass::Warm => 2.0,
            StorageClass::Cold => 1.0,
            StorageClass::Archive => 0.3,
            StorageClass::DeepArchive => 0.1,
        }
    }

    /// Returns true if this class has near-instant retrieval (< 1 second).
    pub fn is_instant_retrieval(&self) -> bool {
        matches!(self, StorageClass::Hot | StorageClass::Warm)
    }

    /// Display name for the storage class.
    pub fn name(&self) -> &'static str {
        match self {
            StorageClass::Hot => "hot",
            StorageClass::Warm => "warm",
            StorageClass::Cold => "cold",
            StorageClass::Archive => "archive",
            StorageClass::DeepArchive => "deep_archive",
        }
    }
}

/// A retention rule that determines when objects should expire or transition.
#[derive(Debug, Clone)]
pub struct RetentionRule {
    /// Unique identifier for this rule.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// How long to retain data (None = retain forever).
    pub retention_period: Option<Duration>,
    /// Storage class to transition to after the retention period (None = delete).
    pub transition_to: Option<StorageClass>,
    /// Tags that must match for this rule to apply.
    pub match_tags: HashMap<String, String>,
    /// Whether this rule is currently active.
    pub enabled: bool,
}

impl RetentionRule {
    /// Create a new retention rule.
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            retention_period: None,
            transition_to: None,
            match_tags: HashMap::new(),
            enabled: true,
        }
    }

    /// Set the retention period.
    pub fn with_period(mut self, period: Duration) -> Self {
        self.retention_period = Some(period);
        self
    }

    /// Set the target storage class for transition (instead of deletion).
    pub fn with_transition(mut self, target: StorageClass) -> Self {
        self.transition_to = Some(target);
        self
    }

    /// Add a tag that objects must have for this rule to apply.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.match_tags.insert(key.into(), value.into());
        self
    }

    /// Returns true if the given creation time means the object is expired at `now`.
    pub fn is_expired_at(&self, created_at: SystemTime, now: SystemTime) -> bool {
        let Some(period) = self.retention_period else {
            return false; // no period = retain forever
        };
        match now.duration_since(created_at) {
            Ok(age) => age >= period,
            Err(_) => false, // clock skew: treat as not expired
        }
    }

    /// Returns true if all required tags are present in `object_tags`.
    pub fn matches_tags(&self, object_tags: &HashMap<String, String>) -> bool {
        self.match_tags
            .iter()
            .all(|(k, v)| object_tags.get(k) == Some(v))
    }
}

/// A storage policy combining a target class and a set of retention rules.
#[derive(Debug, Clone)]
pub struct StoragePolicy {
    /// Policy identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Default storage class for new objects under this policy.
    pub default_class: StorageClass,
    /// Ordered list of retention rules (evaluated first-match wins).
    pub rules: Vec<RetentionRule>,
    /// Tags that objects must have for this policy to apply.
    pub applies_to_tags: HashMap<String, String>,
    /// Prefix pattern that objects must match (empty = all objects).
    pub prefix_pattern: String,
}

impl StoragePolicy {
    /// Create a new storage policy.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        default_class: StorageClass,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            default_class,
            rules: Vec::new(),
            applies_to_tags: HashMap::new(),
            prefix_pattern: String::new(),
        }
    }

    /// Add a retention rule.
    pub fn with_rule(mut self, rule: RetentionRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Set the prefix pattern.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix_pattern = prefix.into();
        self
    }

    /// Add a required tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.applies_to_tags.insert(key.into(), value.into());
        self
    }

    /// Returns true if this policy applies to the given object key and tags.
    pub fn applies_to(&self, key: &str, object_tags: &HashMap<String, String>) -> bool {
        let prefix_ok = self.prefix_pattern.is_empty() || key.starts_with(&self.prefix_pattern);
        let tags_ok = self
            .applies_to_tags
            .iter()
            .all(|(k, v)| object_tags.get(k) == Some(v));
        prefix_ok && tags_ok
    }

    /// Find the first active rule that matches the object tags, or None.
    pub fn matching_rule<'a>(
        &'a self,
        object_tags: &HashMap<String, String>,
    ) -> Option<&'a RetentionRule> {
        self.rules
            .iter()
            .find(|r| r.enabled && r.matches_tags(object_tags))
    }
}

/// A set of policies evaluated against storage objects.
#[derive(Debug, Default, Clone)]
pub struct StoragePolicySet {
    policies: Vec<StoragePolicy>,
}

impl StoragePolicySet {
    /// Create a new empty policy set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a policy.
    pub fn add_policy(&mut self, policy: StoragePolicy) {
        self.policies.push(policy);
    }

    /// How many policies are registered.
    pub fn len(&self) -> usize {
        self.policies.len()
    }

    /// Returns true if no policies are registered.
    pub fn is_empty(&self) -> bool {
        self.policies.is_empty()
    }

    /// Evaluate all policies against an object and return the first matching policy
    /// together with the matching retention rule (if any).
    pub fn evaluate<'a>(
        &'a self,
        key: &str,
        object_tags: &HashMap<String, String>,
    ) -> Option<(&'a StoragePolicy, Option<&'a RetentionRule>)> {
        for policy in &self.policies {
            if policy.applies_to(key, object_tags) {
                let rule = policy.matching_rule(object_tags);
                return Some((policy, rule));
            }
        }
        None
    }

    /// Collect all policies that match the given object.
    pub fn all_matching<'a>(
        &'a self,
        key: &str,
        object_tags: &HashMap<String, String>,
    ) -> Vec<&'a StoragePolicy> {
        self.policies
            .iter()
            .filter(|p| p.applies_to(key, object_tags))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn test_storage_class_access_frequency() {
        assert!(StorageClass::Hot.access_frequency() > StorageClass::Cold.access_frequency());
        assert!(StorageClass::Cold.access_frequency() > StorageClass::Archive.access_frequency());
        assert_eq!(StorageClass::DeepArchive.access_frequency(), 0);
    }

    #[test]
    fn test_storage_class_cost_multiplier() {
        assert!(StorageClass::Hot.cost_multiplier() > StorageClass::Warm.cost_multiplier());
        assert!(StorageClass::Warm.cost_multiplier() > StorageClass::Cold.cost_multiplier());
        assert!(StorageClass::Cold.cost_multiplier() > StorageClass::Archive.cost_multiplier());
    }

    #[test]
    fn test_storage_class_instant_retrieval() {
        assert!(StorageClass::Hot.is_instant_retrieval());
        assert!(StorageClass::Warm.is_instant_retrieval());
        assert!(!StorageClass::Cold.is_instant_retrieval());
        assert!(!StorageClass::Archive.is_instant_retrieval());
    }

    #[test]
    fn test_storage_class_name() {
        assert_eq!(StorageClass::Hot.name(), "hot");
        assert_eq!(StorageClass::DeepArchive.name(), "deep_archive");
    }

    #[test]
    fn test_retention_rule_is_expired() {
        let rule = RetentionRule::new("r1", "expire after 30 days")
            .with_period(Duration::from_secs(30 * 24 * 3600));

        let now = SystemTime::now();
        let old = now - Duration::from_secs(31 * 24 * 3600);
        let recent = now - Duration::from_secs(1 * 24 * 3600);

        assert!(rule.is_expired_at(old, now));
        assert!(!rule.is_expired_at(recent, now));
    }

    #[test]
    fn test_retention_rule_no_period_never_expires() {
        let rule = RetentionRule::new("r2", "never expire");
        let now = SystemTime::now();
        let ancient = now - Duration::from_secs(365 * 24 * 3600 * 100);
        assert!(!rule.is_expired_at(ancient, now));
    }

    #[test]
    fn test_retention_rule_matches_tags() {
        let rule = RetentionRule::new("r3", "tag match")
            .with_tag("env", "prod")
            .with_tag("type", "video");

        let mut tags = HashMap::new();
        tags.insert("env".to_string(), "prod".to_string());
        tags.insert("type".to_string(), "video".to_string());
        tags.insert("extra".to_string(), "ignored".to_string());

        assert!(rule.matches_tags(&tags));

        tags.insert("env".to_string(), "dev".to_string());
        assert!(!rule.matches_tags(&tags));
    }

    #[test]
    fn test_storage_policy_applies_to_prefix() {
        let policy =
            StoragePolicy::new("p1", "video policy", StorageClass::Hot).with_prefix("videos/");

        let tags = HashMap::new();
        assert!(policy.applies_to("videos/clip.mp4", &tags));
        assert!(!policy.applies_to("audio/track.wav", &tags));
    }

    #[test]
    fn test_storage_policy_applies_to_tags() {
        let policy =
            StoragePolicy::new("p2", "prod only", StorageClass::Warm).with_tag("env", "prod");

        let mut tags = HashMap::new();
        tags.insert("env".to_string(), "prod".to_string());
        assert!(policy.applies_to("any/key", &tags));

        tags.insert("env".to_string(), "staging".to_string());
        assert!(!policy.applies_to("any/key", &tags));
    }

    #[test]
    fn test_storage_policy_matching_rule() {
        let rule = RetentionRule::new("r1", "expire old")
            .with_period(Duration::from_secs(90 * 24 * 3600))
            .with_tag("archive", "true");

        let policy = StoragePolicy::new("p3", "archive policy", StorageClass::Cold).with_rule(rule);

        let mut tags = HashMap::new();
        tags.insert("archive".to_string(), "true".to_string());
        assert!(policy.matching_rule(&tags).is_some());

        let empty_tags = HashMap::new();
        assert!(policy.matching_rule(&empty_tags).is_none());
    }

    #[test]
    fn test_storage_policy_set_evaluate() {
        let p1 = StoragePolicy::new("p1", "video", StorageClass::Hot).with_prefix("video/");
        let p2 = StoragePolicy::new("p2", "audio", StorageClass::Warm).with_prefix("audio/");

        let mut set = StoragePolicySet::new();
        set.add_policy(p1);
        set.add_policy(p2);

        let tags = HashMap::new();
        let result = set.evaluate("video/clip.mp4", &tags);
        assert!(result.is_some());
        assert_eq!(result.expect("result should be ok").0.id, "p1");

        let result2 = set.evaluate("audio/track.mp3", &tags);
        assert_eq!(result2.expect("result should be ok").0.id, "p2");
    }

    #[test]
    fn test_storage_policy_set_no_match() {
        let p1 = StoragePolicy::new("p1", "video only", StorageClass::Hot).with_prefix("video/");
        let mut set = StoragePolicySet::new();
        set.add_policy(p1);

        let tags = HashMap::new();
        assert!(set.evaluate("docs/readme.txt", &tags).is_none());
    }

    #[test]
    fn test_storage_policy_set_all_matching() {
        let p1 = StoragePolicy::new("p1", "all objects", StorageClass::Cold);
        let p2 = StoragePolicy::new("p2", "also all", StorageClass::Warm);
        let mut set = StoragePolicySet::new();
        set.add_policy(p1);
        set.add_policy(p2);

        let tags = HashMap::new();
        let matches = set.all_matching("any/key", &tags);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_policy_set_len_and_empty() {
        let mut set = StoragePolicySet::new();
        assert!(set.is_empty());
        set.add_policy(StoragePolicy::new("p1", "p1", StorageClass::Hot));
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());
    }
}
