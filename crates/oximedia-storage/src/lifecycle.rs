//! Storage lifecycle policies for OxiMedia.
//!
//! Supports age-based transitions between cost tiers, automatic expiration
//! rules, and tag-based policy matching.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::time::Duration;

/// Storage cost tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageTier {
    /// Frequently accessed data — highest cost
    Hot,
    /// Infrequently accessed data — moderate cost
    Warm,
    /// Rarely accessed archive data — low cost, higher retrieval latency
    Cold,
    /// Deep archive — lowest cost, long retrieval time
    Archive,
}

impl StorageTier {
    /// Returns a human-readable label for the tier
    pub fn label(&self) -> &'static str {
        match self {
            StorageTier::Hot => "hot",
            StorageTier::Warm => "warm",
            StorageTier::Cold => "cold",
            StorageTier::Archive => "archive",
        }
    }

    /// Returns the relative cost factor (Hot = 1.0 baseline)
    pub fn cost_factor(&self) -> f64 {
        match self {
            StorageTier::Hot => 1.0,
            StorageTier::Warm => 0.5,
            StorageTier::Cold => 0.1,
            StorageTier::Archive => 0.02,
        }
    }

    /// Returns the approximate retrieval latency for this tier
    pub fn retrieval_latency(&self) -> Duration {
        match self {
            StorageTier::Hot => Duration::from_millis(50),
            StorageTier::Warm => Duration::from_secs(1),
            StorageTier::Cold => Duration::from_mins(1),
            StorageTier::Archive => Duration::from_hours(4),
        }
    }
}

/// A lifecycle transition rule: move objects to a new tier after a given age
#[derive(Debug, Clone)]
pub struct TransitionRule {
    /// Minimum object age before the transition applies
    pub min_age: Duration,
    /// Target storage tier
    pub target_tier: StorageTier,
    /// Optional tag filter — only apply to objects with this tag key/value
    pub tag_filter: Option<(String, String)>,
}

impl TransitionRule {
    /// Creates a transition rule with no tag filter
    pub fn new(min_age: Duration, target_tier: StorageTier) -> Self {
        Self {
            min_age,
            target_tier,
            tag_filter: None,
        }
    }

    /// Adds a tag filter
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tag_filter = Some((key.into(), value.into()));
        self
    }

    /// Returns true if this rule applies to an object given its age and tags
    pub fn applies(&self, age: Duration, tags: &HashMap<String, String>) -> bool {
        if age < self.min_age {
            return false;
        }
        if let Some((ref k, ref v)) = self.tag_filter {
            return tags.get(k).map(String::as_str) == Some(v.as_str());
        }
        true
    }
}

/// An expiration rule: delete objects after a given age
#[derive(Debug, Clone)]
pub struct ExpirationRule {
    /// Minimum object age before deletion applies
    pub min_age: Duration,
    /// Optional tag filter
    pub tag_filter: Option<(String, String)>,
    /// Whether to permanently delete or move to a "deleted" soft-delete zone
    pub permanent: bool,
}

impl ExpirationRule {
    /// Creates a permanent expiration rule
    pub fn permanent(min_age: Duration) -> Self {
        Self {
            min_age,
            tag_filter: None,
            permanent: true,
        }
    }

    /// Creates a soft-delete expiration rule
    pub fn soft_delete(min_age: Duration) -> Self {
        Self {
            min_age,
            tag_filter: None,
            permanent: false,
        }
    }

    /// Adds a tag filter
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tag_filter = Some((key.into(), value.into()));
        self
    }

    /// Returns true if this expiration rule applies to an object
    pub fn applies(&self, age: Duration, tags: &HashMap<String, String>) -> bool {
        if age < self.min_age {
            return false;
        }
        if let Some((ref k, ref v)) = self.tag_filter {
            return tags.get(k).map(String::as_str) == Some(v.as_str());
        }
        true
    }
}

/// A complete lifecycle policy combining transition and expiration rules
#[derive(Debug, Clone)]
pub struct LifecyclePolicy {
    /// Policy identifier
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Prefix filter — only apply to objects whose key starts with this
    pub prefix: Option<String>,
    /// Ordered list of transition rules
    pub transitions: Vec<TransitionRule>,
    /// Optional expiration rule
    pub expiration: Option<ExpirationRule>,
    /// Whether this policy is enabled
    pub enabled: bool,
}

impl LifecyclePolicy {
    /// Creates a new enabled lifecycle policy
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            prefix: None,
            transitions: Vec::new(),
            expiration: None,
            enabled: true,
        }
    }

    /// Adds a prefix filter
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Adds a transition rule
    pub fn add_transition(mut self, rule: TransitionRule) -> Self {
        self.transitions.push(rule);
        self
    }

    /// Sets the expiration rule
    pub fn with_expiration(mut self, rule: ExpirationRule) -> Self {
        self.expiration = Some(rule);
        self
    }

    /// Disables this policy
    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Returns true if the policy applies to the given object key
    pub fn matches_key(&self, key: &str) -> bool {
        match &self.prefix {
            Some(p) => key.starts_with(p.as_str()),
            None => true,
        }
    }

    /// Evaluates the policy for an object and returns the recommended action
    pub fn evaluate(
        &self,
        key: &str,
        age: Duration,
        current_tier: StorageTier,
        tags: &HashMap<String, String>,
    ) -> PolicyAction {
        if !self.enabled || !self.matches_key(key) {
            return PolicyAction::NoOp;
        }

        // Check expiration first
        if let Some(ref exp) = self.expiration {
            if exp.applies(age, tags) {
                return PolicyAction::Delete {
                    permanent: exp.permanent,
                };
            }
        }

        // Find the most-advanced applicable transition
        let mut target = None;
        for rule in &self.transitions {
            if rule.applies(age, tags) && rule.target_tier > current_tier {
                target = Some(rule.target_tier);
            }
        }

        if let Some(tier) = target {
            PolicyAction::Transition { to: tier }
        } else {
            PolicyAction::NoOp
        }
    }
}

/// The action recommended by a lifecycle policy evaluation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyAction {
    /// No action required
    NoOp,
    /// Transition object to a new storage tier
    Transition {
        /// Target tier
        to: StorageTier,
    },
    /// Delete the object
    Delete {
        /// Whether the delete is permanent
        permanent: bool,
    },
}

/// Lifecycle policy manager — holds and evaluates multiple policies
#[derive(Debug, Clone)]
pub struct LifecyclePolicyManager {
    policies: Vec<LifecyclePolicy>,
}

impl LifecyclePolicyManager {
    /// Creates a new empty policy manager
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
        }
    }

    /// Registers a policy
    pub fn register(mut self, policy: LifecyclePolicy) -> Self {
        self.policies.push(policy);
        self
    }

    /// Returns all enabled policies
    pub fn enabled_policies(&self) -> Vec<&LifecyclePolicy> {
        self.policies.iter().filter(|p| p.enabled).collect()
    }

    /// Evaluates all enabled policies for the given object.
    /// Returns the first non-NoOp action, or NoOp if none applies.
    pub fn evaluate_all(
        &self,
        key: &str,
        age: Duration,
        current_tier: StorageTier,
        tags: &HashMap<String, String>,
    ) -> PolicyAction {
        for policy in self.enabled_policies() {
            let action = policy.evaluate(key, age, current_tier, tags);
            if action != PolicyAction::NoOp {
                return action;
            }
        }
        PolicyAction::NoOp
    }

    /// Returns the count of registered policies
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }
}

impl Default for LifecyclePolicyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_tags() -> HashMap<String, String> {
        HashMap::new()
    }

    fn tags(key: &str, value: &str) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(key.to_string(), value.to_string());
        m
    }

    #[test]
    fn test_storage_tier_ordering() {
        assert!(StorageTier::Hot < StorageTier::Warm);
        assert!(StorageTier::Warm < StorageTier::Cold);
        assert!(StorageTier::Cold < StorageTier::Archive);
    }

    #[test]
    fn test_storage_tier_cost_factor() {
        assert_eq!(StorageTier::Hot.cost_factor(), 1.0);
        assert!(StorageTier::Warm.cost_factor() < StorageTier::Hot.cost_factor());
        assert!(StorageTier::Archive.cost_factor() < StorageTier::Cold.cost_factor());
    }

    #[test]
    fn test_storage_tier_label() {
        assert_eq!(StorageTier::Hot.label(), "hot");
        assert_eq!(StorageTier::Archive.label(), "archive");
    }

    #[test]
    fn test_storage_tier_retrieval_latency() {
        assert!(StorageTier::Archive.retrieval_latency() > StorageTier::Hot.retrieval_latency());
    }

    #[test]
    fn test_transition_rule_applies_by_age() {
        let rule = TransitionRule::new(Duration::from_hours(720), StorageTier::Warm);
        assert!(!rule.applies(Duration::from_hours(240), &no_tags()));
        assert!(rule.applies(Duration::from_hours(720), &no_tags()));
        assert!(rule.applies(Duration::from_hours(1440), &no_tags()));
    }

    #[test]
    fn test_transition_rule_tag_filter() {
        let rule = TransitionRule::new(Duration::from_hours(24), StorageTier::Cold)
            .with_tag("env", "staging");
        assert!(!rule.applies(Duration::from_hours(48), &no_tags()));
        assert!(rule.applies(Duration::from_hours(48), &tags("env", "staging")));
        assert!(!rule.applies(Duration::from_hours(48), &tags("env", "prod")));
    }

    #[test]
    fn test_expiration_rule_permanent() {
        let rule = ExpirationRule::permanent(Duration::from_hours(8760));
        assert!(rule.permanent);
        assert!(!rule.applies(Duration::from_hours(2400), &no_tags()));
        assert!(rule.applies(Duration::from_hours(9600), &no_tags()));
    }

    #[test]
    fn test_expiration_rule_soft_delete() {
        let rule = ExpirationRule::soft_delete(Duration::from_hours(2160));
        assert!(!rule.permanent);
        assert!(rule.applies(Duration::from_hours(2400), &no_tags()));
    }

    #[test]
    fn test_lifecycle_policy_matches_key() {
        let policy = LifecyclePolicy::new("p1", "test").with_prefix("media/");
        assert!(policy.matches_key("media/video.mp4"));
        assert!(!policy.matches_key("logs/app.log"));
    }

    #[test]
    fn test_lifecycle_policy_evaluate_transition() {
        let policy = LifecyclePolicy::new("p1", "test")
            .add_transition(TransitionRule::new(
                Duration::from_hours(720),
                StorageTier::Warm,
            ))
            .add_transition(TransitionRule::new(
                Duration::from_hours(2160),
                StorageTier::Cold,
            ));

        let action = policy.evaluate(
            "file.mp4",
            Duration::from_hours(1440),
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(
            action,
            PolicyAction::Transition {
                to: StorageTier::Warm
            }
        );
    }

    #[test]
    fn test_lifecycle_policy_evaluate_expiration() {
        let policy = LifecyclePolicy::new("p1", "expire old")
            .with_expiration(ExpirationRule::permanent(Duration::from_hours(8760)));

        let action = policy.evaluate(
            "file.mp4",
            Duration::from_hours(9600),
            StorageTier::Cold,
            &no_tags(),
        );
        assert_eq!(action, PolicyAction::Delete { permanent: true });
    }

    #[test]
    fn test_lifecycle_policy_evaluate_no_op() {
        let policy = LifecyclePolicy::new("p1", "test").add_transition(TransitionRule::new(
            Duration::from_hours(2160),
            StorageTier::Warm,
        ));

        let action = policy.evaluate(
            "file.mp4",
            Duration::from_hours(240),
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(action, PolicyAction::NoOp);
    }

    #[test]
    fn test_lifecycle_policy_disabled() {
        let policy = LifecyclePolicy::new("p1", "test")
            .add_transition(TransitionRule::new(
                Duration::from_secs(1),
                StorageTier::Warm,
            ))
            .disable();

        let action = policy.evaluate(
            "file.mp4",
            Duration::from_secs(1000),
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(action, PolicyAction::NoOp);
    }

    #[test]
    fn test_policy_manager_evaluate_all() {
        let mgr =
            LifecyclePolicyManager::new()
                .register(LifecyclePolicy::new("p1", "transition").add_transition(
                    TransitionRule::new(Duration::from_hours(720), StorageTier::Warm),
                ))
                .register(
                    LifecyclePolicy::new("p2", "expire")
                        .with_expiration(ExpirationRule::permanent(Duration::from_hours(8760))),
                );

        let action = mgr.evaluate_all(
            "x.mp4",
            Duration::from_hours(1440),
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(
            action,
            PolicyAction::Transition {
                to: StorageTier::Warm
            }
        );
    }

    #[test]
    fn test_policy_manager_count() {
        let mgr = LifecyclePolicyManager::new()
            .register(LifecyclePolicy::new("a", "a"))
            .register(LifecyclePolicy::new("b", "b"));
        assert_eq!(mgr.policy_count(), 2);
    }

    #[test]
    fn test_policy_manager_enabled_policies() {
        let mgr = LifecyclePolicyManager::new()
            .register(LifecyclePolicy::new("active", "x"))
            .register(LifecyclePolicy::new("disabled", "x").disable());
        assert_eq!(mgr.enabled_policies().len(), 1);
    }

    // ── Additional lifecycle policy tests for tier transitions ────────────────

    #[test]
    fn test_lifecycle_new_object_no_action() {
        // Objects younger than the first transition threshold get NoOp
        let policy =
            LifecyclePolicy::new("p-new", "new object test").add_transition(TransitionRule::new(
                Duration::from_hours(720), // 30 days
                StorageTier::Warm,
            ));
        let action = policy.evaluate(
            "new-object.mp4",
            Duration::from_hours(24), // 1 day old
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(
            action,
            PolicyAction::NoOp,
            "new objects should not trigger transitions"
        );
    }

    #[test]
    fn test_lifecycle_transition_to_warm_at_30_days() {
        let policy = LifecyclePolicy::new("p-tiers", "standard tiers")
            .add_transition(TransitionRule::new(
                Duration::from_hours(720),
                StorageTier::Warm,
            ))
            .add_transition(TransitionRule::new(
                Duration::from_hours(2160),
                StorageTier::Cold,
            ));
        let action = policy.evaluate(
            "file.mp4",
            Duration::from_hours(720), // exactly 30 days
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(
            action,
            PolicyAction::Transition {
                to: StorageTier::Warm
            },
            "30 day old object should transition to Warm"
        );
    }

    #[test]
    fn test_lifecycle_transition_to_cold_at_90_days() {
        let policy = LifecyclePolicy::new("p-tiers", "standard tiers")
            .add_transition(TransitionRule::new(
                Duration::from_hours(720),
                StorageTier::Warm,
            ))
            .add_transition(TransitionRule::new(
                Duration::from_hours(2160),
                StorageTier::Cold,
            ));
        let action = policy.evaluate(
            "file.mp4",
            Duration::from_hours(2160), // 90 days
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(
            action,
            PolicyAction::Transition {
                to: StorageTier::Cold
            },
            "90 day old object on Hot should transition to Cold (most advanced applicable)"
        );
    }

    #[test]
    fn test_lifecycle_transition_to_archive_at_365_days() {
        let policy = LifecyclePolicy::new("p-all", "full tier ladder")
            .add_transition(TransitionRule::new(
                Duration::from_hours(720),
                StorageTier::Warm,
            ))
            .add_transition(TransitionRule::new(
                Duration::from_hours(2160),
                StorageTier::Cold,
            ))
            .add_transition(TransitionRule::new(
                Duration::from_hours(8760),
                StorageTier::Archive,
            ));
        let action = policy.evaluate(
            "archive.mp4",
            Duration::from_hours(9600), // > 365 days
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(
            action,
            PolicyAction::Transition {
                to: StorageTier::Archive
            }
        );
    }

    #[test]
    fn test_lifecycle_no_transition_if_already_at_target_tier() {
        // Object already at Warm — policy only transitions Hot→Warm
        let policy = LifecyclePolicy::new("p-warm-only", "transition to warm").add_transition(
            TransitionRule::new(Duration::from_hours(720), StorageTier::Warm),
        );
        let action = policy.evaluate(
            "file.mp4",
            Duration::from_hours(1440), // 60 days
            StorageTier::Warm,          // already at Warm — rule is > current_tier only
            &no_tags(),
        );
        // Warm is not > Warm, so the transition is suppressed
        assert_eq!(action, PolicyAction::NoOp);
    }

    #[test]
    fn test_lifecycle_retain_forever_tag_prevents_expiration() {
        let policy = LifecyclePolicy::new("p-expire", "expire old objects").with_expiration(
            ExpirationRule::permanent(Duration::from_hours(8760)).with_tag("lifecycle", "expire"),
        );
        // Object is old enough but does NOT have the "lifecycle=expire" tag
        let action = policy.evaluate(
            "retained.mp4",
            Duration::from_hours(9600),
            StorageTier::Archive,
            &no_tags(), // no matching tag
        );
        assert_eq!(
            action,
            PolicyAction::NoOp,
            "tag filter prevents expiration without matching tag"
        );
    }

    #[test]
    fn test_lifecycle_expire_with_matching_tag() {
        let policy = LifecyclePolicy::new("p-tagged-expire", "tag-driven expiration")
            .with_expiration(
                ExpirationRule::permanent(Duration::from_hours(24)).with_tag("lifecycle", "expire"),
            );
        let action = policy.evaluate(
            "old.mp4",
            Duration::from_hours(168),
            StorageTier::Cold,
            &tags("lifecycle", "expire"),
        );
        assert_eq!(action, PolicyAction::Delete { permanent: true });
    }

    #[test]
    fn test_lifecycle_soft_delete_vs_permanent() {
        let soft = LifecyclePolicy::new("p-soft", "soft delete")
            .with_expiration(ExpirationRule::soft_delete(Duration::from_hours(24)));
        let action = soft.evaluate(
            "file.mp4",
            Duration::from_hours(48),
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(action, PolicyAction::Delete { permanent: false });
    }

    #[test]
    fn test_lifecycle_custom_threshold_transitions() {
        // Custom: transition to Cold after just 1 hour
        let policy =
            LifecyclePolicy::new("p-fast", "fast cooling").add_transition(TransitionRule::new(
                Duration::from_hours(1), // 1 hour
                StorageTier::Cold,
            ));
        let new_action = policy.evaluate(
            "stream.ts",
            Duration::from_mins(30), // 30 minutes — too young
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(new_action, PolicyAction::NoOp);
        let old_action = policy.evaluate(
            "stream.ts",
            Duration::from_hours(2), // 2 hours — past threshold
            StorageTier::Hot,
            &no_tags(),
        );
        assert_eq!(
            old_action,
            PolicyAction::Transition {
                to: StorageTier::Cold
            }
        );
    }

    #[test]
    fn test_lifecycle_policy_manager_first_match_wins() {
        // Two policies both match; first non-NoOp action is returned
        let mgr = LifecyclePolicyManager::new()
            .register(LifecyclePolicy::new("p1", "first policy").add_transition(
                TransitionRule::new(Duration::from_hours(720), StorageTier::Warm),
            ))
            .register(LifecyclePolicy::new("p2", "second policy").add_transition(
                TransitionRule::new(Duration::from_hours(720), StorageTier::Cold),
            ));
        let action = mgr.evaluate_all(
            "obj.mp4",
            Duration::from_hours(840),
            StorageTier::Hot,
            &no_tags(),
        );
        // p1 fires first (Warm), p2 would say Cold — but we return the first
        assert_eq!(
            action,
            PolicyAction::Transition {
                to: StorageTier::Warm
            }
        );
    }
}
