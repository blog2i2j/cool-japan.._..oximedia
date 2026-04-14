//! Content embargo and scheduling for time-gated recommendations.
//!
//! Allows content to be withheld from recommendation surfaces until a
//! specified **release timestamp**, and optionally hidden again after an
//! **expiry timestamp**.  Regions can have independent release gates, and
//! content can be soft-embargoed (demoted in score) rather than hard-blocked.
//!
//! # Example flow
//!
//! 1. A studio registers content with `EmbargoRegistry::add`.
//! 2. Before surfacing recommendations, call `EmbargoRegistry::filter` to
//!    remove not-yet-released items and expired items.
//! 3. Optionally call `EmbargoRegistry::soft_filter` to retain embargoed
//!    items but apply a score penalty.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Embargo rule
// ---------------------------------------------------------------------------

/// Visibility status of a content item at a given instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbargoStatus {
    /// The item is available for recommendation.
    Available,
    /// The item is embargoed (not yet released).
    Embargoed,
    /// The item has passed its expiry and should no longer be recommended.
    Expired,
}

impl EmbargoStatus {
    /// Returns `true` if the item may be recommended.
    #[must_use]
    pub fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }
}

/// Region-specific visibility window for a content item.
#[derive(Debug, Clone)]
pub struct RegionWindow {
    /// Region code (e.g., "US", "GB", "JP", "global").
    pub region: String,
    /// Unix timestamp (seconds) before which the item is embargoed.
    pub release_at: i64,
    /// Unix timestamp (seconds) after which the item is expired (`None` = never expires).
    pub expires_at: Option<i64>,
}

impl RegionWindow {
    /// Create a region window.
    #[must_use]
    pub fn new(region: impl Into<String>, release_at: i64, expires_at: Option<i64>) -> Self {
        Self {
            region: region.into(),
            release_at,
            expires_at,
        }
    }

    /// Compute the embargo status for this region at `now`.
    #[must_use]
    pub fn status(&self, now: i64) -> EmbargoStatus {
        if now < self.release_at {
            return EmbargoStatus::Embargoed;
        }
        if let Some(exp) = self.expires_at {
            if now >= exp {
                return EmbargoStatus::Expired;
            }
        }
        EmbargoStatus::Available
    }
}

/// An embargo rule for a single content item.
#[derive(Debug, Clone)]
pub struct EmbargoRule {
    /// Content identifier.
    pub content_id: String,
    /// Global release timestamp (applies when no region-specific rule matches).
    pub global_release_at: i64,
    /// Global expiry timestamp (`None` = no global expiry).
    pub global_expires_at: Option<i64>,
    /// Region-specific overrides.
    pub regions: Vec<RegionWindow>,
    /// Score multiplier applied during soft embargo (after global release but before full rollout).
    /// Values in (0.0, 1.0) demote the item; 1.0 = no penalty.
    pub soft_penalty: f64,
}

impl EmbargoRule {
    /// Create a rule with only a global release timestamp.
    #[must_use]
    pub fn new(content_id: impl Into<String>, global_release_at: i64) -> Self {
        Self {
            content_id: content_id.into(),
            global_release_at,
            global_expires_at: None,
            regions: Vec::new(),
            soft_penalty: 1.0,
        }
    }

    /// Set a global expiry timestamp.
    #[must_use]
    pub fn with_expiry(mut self, expires_at: i64) -> Self {
        self.global_expires_at = Some(expires_at);
        self
    }

    /// Add a region-specific window.
    #[must_use]
    pub fn with_region(mut self, window: RegionWindow) -> Self {
        self.regions.push(window);
        self
    }

    /// Set the soft-embargo score penalty (0.0–1.0).
    #[must_use]
    pub fn with_soft_penalty(mut self, penalty: f64) -> Self {
        self.soft_penalty = penalty.clamp(0.0, 1.0);
        self
    }

    /// Determine the embargo status for a given region and timestamp.
    ///
    /// If a region-specific window exists for `region`, it is used; otherwise
    /// the global timestamps govern.
    #[must_use]
    pub fn status_for(&self, region: &str, now: i64) -> EmbargoStatus {
        // Check region-specific window first
        if let Some(rw) = self.regions.iter().find(|r| r.region == region) {
            return rw.status(now);
        }

        // Fall back to global timestamps
        if now < self.global_release_at {
            return EmbargoStatus::Embargoed;
        }
        if let Some(exp) = self.global_expires_at {
            if now >= exp {
                return EmbargoStatus::Expired;
            }
        }
        EmbargoStatus::Available
    }

    /// Seconds until release for the given region, or 0 if already released.
    #[must_use]
    pub fn seconds_until_release(&self, region: &str, now: i64) -> i64 {
        let release = self
            .regions
            .iter()
            .find(|r| r.region == region)
            .map(|r| r.release_at)
            .unwrap_or(self.global_release_at);
        (release - now).max(0)
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// A candidate item for filtering (id + current score).
#[derive(Debug, Clone)]
pub struct ScoredItem {
    /// Content identifier.
    pub content_id: String,
    /// Current recommendation score.
    pub score: f64,
}

impl ScoredItem {
    /// Create a new scored item.
    #[must_use]
    pub fn new(content_id: impl Into<String>, score: f64) -> Self {
        Self {
            content_id: content_id.into(),
            score,
        }
    }
}

/// Registry of content embargo rules.
///
/// Used to filter or demote time-gated content before surfacing
/// recommendations to users.
#[derive(Debug, Default)]
pub struct EmbargoRegistry {
    /// Rules keyed by content ID.
    rules: HashMap<String, EmbargoRule>,
}

impl EmbargoRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or replace an embargo rule for a content item.
    pub fn add(&mut self, rule: EmbargoRule) {
        self.rules.insert(rule.content_id.clone(), rule);
    }

    /// Remove an embargo rule.
    pub fn remove(&mut self, content_id: &str) {
        self.rules.remove(content_id);
    }

    /// Get the embargo rule for a content item.
    #[must_use]
    pub fn get(&self, content_id: &str) -> Option<&EmbargoRule> {
        self.rules.get(content_id)
    }

    /// Check the status of a content item.
    ///
    /// Items without a rule are considered always `Available`.
    #[must_use]
    pub fn status(&self, content_id: &str, region: &str, now: i64) -> EmbargoStatus {
        match self.rules.get(content_id) {
            Some(rule) => rule.status_for(region, now),
            None => EmbargoStatus::Available,
        }
    }

    /// Hard-filter a list of items, removing embargoed and expired content.
    ///
    /// Items not in the registry are kept unchanged.
    #[must_use]
    pub fn filter(&self, items: Vec<ScoredItem>, region: &str, now: i64) -> Vec<ScoredItem> {
        items
            .into_iter()
            .filter(|item| self.status(&item.content_id, region, now) == EmbargoStatus::Available)
            .collect()
    }

    /// Soft-filter: retain all items but apply the `soft_penalty` multiplier to
    /// embargoed ones and drop expired ones.
    #[must_use]
    pub fn soft_filter(&self, items: Vec<ScoredItem>, region: &str, now: i64) -> Vec<ScoredItem> {
        items
            .into_iter()
            .filter_map(|mut item| {
                let status = self.status(&item.content_id, region, now);
                match status {
                    EmbargoStatus::Expired => None,
                    EmbargoStatus::Embargoed => {
                        // Apply soft penalty from the rule
                        let penalty = self
                            .rules
                            .get(&item.content_id)
                            .map(|r| r.soft_penalty)
                            .unwrap_or(0.0);
                        item.score *= penalty;
                        Some(item)
                    }
                    EmbargoStatus::Available => Some(item),
                }
            })
            .collect()
    }

    /// Count of registered rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// List content IDs available for a region at `now`.
    #[must_use]
    pub fn available_content_ids(&self, region: &str, now: i64) -> Vec<&str> {
        self.rules
            .iter()
            .filter(|(_, rule)| rule.status_for(region, now) == EmbargoStatus::Available)
            .map(|(id, _)| id.as_str())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn base_rule(id: &str, release: i64) -> EmbargoRule {
        EmbargoRule::new(id, release)
    }

    // ---- EmbargoStatus ----

    #[test]
    fn test_status_is_available() {
        assert!(EmbargoStatus::Available.is_available());
        assert!(!EmbargoStatus::Embargoed.is_available());
        assert!(!EmbargoStatus::Expired.is_available());
    }

    // ---- RegionWindow ----

    #[test]
    fn test_region_window_embargoed() {
        let rw = RegionWindow::new("US", 1000, None);
        assert_eq!(rw.status(500), EmbargoStatus::Embargoed);
    }

    #[test]
    fn test_region_window_available() {
        let rw = RegionWindow::new("US", 1000, None);
        assert_eq!(rw.status(1001), EmbargoStatus::Available);
    }

    #[test]
    fn test_region_window_expired() {
        let rw = RegionWindow::new("US", 1000, Some(2000));
        assert_eq!(rw.status(2001), EmbargoStatus::Expired);
    }

    // ---- EmbargoRule ----

    #[test]
    fn test_rule_global_embargoed() {
        let rule = base_rule("c1", 5000);
        assert_eq!(rule.status_for("global", 4999), EmbargoStatus::Embargoed);
    }

    #[test]
    fn test_rule_global_available() {
        let rule = base_rule("c1", 5000);
        assert_eq!(rule.status_for("global", 5001), EmbargoStatus::Available);
    }

    #[test]
    fn test_rule_global_expiry() {
        let rule = base_rule("c1", 1000).with_expiry(3000);
        assert_eq!(rule.status_for("global", 3001), EmbargoStatus::Expired);
    }

    #[test]
    fn test_rule_region_override() {
        let rule = base_rule("c1", 10_000).with_region(RegionWindow::new("JP", 5000, None));
        // JP is released earlier than global
        assert_eq!(rule.status_for("JP", 6000), EmbargoStatus::Available);
        // Global still embargoed
        assert_eq!(rule.status_for("US", 6000), EmbargoStatus::Embargoed);
    }

    #[test]
    fn test_seconds_until_release() {
        let rule = base_rule("c1", 1000);
        assert_eq!(rule.seconds_until_release("global", 700), 300);
        assert_eq!(rule.seconds_until_release("global", 1200), 0);
    }

    #[test]
    fn test_soft_penalty_clamped() {
        let rule = base_rule("c1", 1000).with_soft_penalty(1.5);
        assert!((rule.soft_penalty - 1.0).abs() < f64::EPSILON);
    }

    // ---- EmbargoRegistry ----

    #[test]
    fn test_registry_empty_status_is_available() {
        let registry = EmbargoRegistry::new();
        assert_eq!(
            registry.status("any_content", "US", 999_999),
            EmbargoStatus::Available
        );
    }

    #[test]
    fn test_registry_add_and_status() {
        let mut reg = EmbargoRegistry::new();
        reg.add(base_rule("movie1", 5000));
        assert_eq!(
            reg.status("movie1", "global", 4000),
            EmbargoStatus::Embargoed
        );
        assert_eq!(
            reg.status("movie1", "global", 6000),
            EmbargoStatus::Available
        );
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = EmbargoRegistry::new();
        reg.add(base_rule("movie1", 5000));
        reg.remove("movie1");
        // After removal, defaults to available
        assert_eq!(
            reg.status("movie1", "global", 1000),
            EmbargoStatus::Available
        );
    }

    #[test]
    fn test_registry_hard_filter_removes_embargoed() {
        let mut reg = EmbargoRegistry::new();
        reg.add(base_rule("embargoed", 9999));
        let items = vec![
            ScoredItem::new("embargoed", 0.9),
            ScoredItem::new("available", 0.5),
        ];
        let filtered = reg.filter(items, "global", 1000);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].content_id, "available");
    }

    #[test]
    fn test_registry_hard_filter_removes_expired() {
        let mut reg = EmbargoRegistry::new();
        reg.add(base_rule("old", 0).with_expiry(500));
        let items = vec![ScoredItem::new("old", 0.9), ScoredItem::new("new", 0.5)];
        let filtered = reg.filter(items, "global", 1000);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].content_id, "new");
    }

    #[test]
    fn test_registry_soft_filter_keeps_embargoed_with_penalty() {
        let mut reg = EmbargoRegistry::new();
        reg.add(base_rule("embargoed", 9999).with_soft_penalty(0.5));
        let items = vec![ScoredItem::new("embargoed", 1.0)];
        let filtered = reg.soft_filter(items, "global", 1000);
        assert_eq!(filtered.len(), 1);
        assert!((filtered[0].score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_registry_soft_filter_drops_expired() {
        let mut reg = EmbargoRegistry::new();
        reg.add(base_rule("old", 0).with_expiry(500));
        let items = vec![ScoredItem::new("old", 0.9)];
        let filtered = reg.soft_filter(items, "global", 1000);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_registry_rule_count() {
        let mut reg = EmbargoRegistry::new();
        assert_eq!(reg.rule_count(), 0);
        reg.add(base_rule("a", 0));
        reg.add(base_rule("b", 0));
        assert_eq!(reg.rule_count(), 2);
    }

    #[test]
    fn test_registry_available_content_ids() {
        let mut reg = EmbargoRegistry::new();
        reg.add(base_rule("released", 100));
        reg.add(base_rule("unreleased", 5000));
        let available = reg.available_content_ids("global", 1000);
        assert_eq!(available.len(), 1);
        assert_eq!(available[0], "released");
    }

    #[test]
    fn test_region_window_with_region_specific_release() {
        let rule = base_rule("c1", 10_000).with_region(RegionWindow::new("EU", 2000, Some(8000)));
        assert_eq!(rule.status_for("EU", 1000), EmbargoStatus::Embargoed);
        assert_eq!(rule.status_for("EU", 5000), EmbargoStatus::Available);
        assert_eq!(rule.status_for("EU", 9000), EmbargoStatus::Expired);
        // Global still embargoed at 5000
        assert_eq!(rule.status_for("US", 5000), EmbargoStatus::Embargoed);
    }

    #[test]
    fn test_registry_get_rule() {
        let mut reg = EmbargoRegistry::new();
        reg.add(base_rule("x", 1000));
        assert!(reg.get("x").is_some());
        assert!(reg.get("y").is_none());
    }
}
