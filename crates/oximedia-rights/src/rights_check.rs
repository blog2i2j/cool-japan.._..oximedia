//! Rights-checking engine.
//!
//! Validates whether a given action is permitted under the rights associated
//! with a content asset. Checks include territory, time-window, usage-type,
//! and platform restrictions.

#![allow(dead_code)]

use std::collections::HashSet;

// ── ActionKind ─────────────────────────────────────────────────────────────

/// The kind of action a user wants to perform on content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionKind {
    /// Stream content for viewing.
    Stream,
    /// Download a local copy.
    Download,
    /// Embed content in another work.
    Embed,
    /// Broadcast over terrestrial/satellite/cable.
    Broadcast,
    /// Use in advertising / promotional material.
    Advertising,
    /// Create a derivative work.
    Derivative,
    /// Distribute to third parties.
    Distribute,
    /// Archive for long-term preservation.
    Archive,
}

// ── RightsGrant ────────────────────────────────────────────────────────────

/// A single rights grant attached to a content asset.
///
/// A grant authorises one or more [`ActionKind`]s within a set of
/// territory codes and a validity window expressed as Unix-epoch seconds.
#[derive(Debug, Clone)]
pub struct RightsGrant {
    /// Unique identifier for this grant.
    pub id: String,
    /// The content asset this grant applies to.
    pub asset_id: String,
    /// Actions permitted under this grant.
    pub allowed_actions: HashSet<ActionKind>,
    /// ISO-3166-1 alpha-2 territory codes where the grant applies.
    /// An empty set means "worldwide".
    pub territories: HashSet<String>,
    /// Platforms on which the grant is valid (empty = all).
    pub platforms: HashSet<String>,
    /// Validity start (Unix seconds, inclusive).
    pub valid_from: u64,
    /// Validity end (Unix seconds, exclusive). `u64::MAX` = no expiry.
    pub valid_until: u64,
    /// Whether the grant has been explicitly revoked.
    pub revoked: bool,
}

impl RightsGrant {
    /// Create a new, non-revoked grant.
    #[must_use]
    pub fn new(id: &str, asset_id: &str) -> Self {
        Self {
            id: id.to_string(),
            asset_id: asset_id.to_string(),
            allowed_actions: HashSet::new(),
            territories: HashSet::new(),
            platforms: HashSet::new(),
            valid_from: 0,
            valid_until: u64::MAX,
            revoked: false,
        }
    }

    /// Builder: allow an action.
    #[must_use]
    pub fn with_action(mut self, action: ActionKind) -> Self {
        self.allowed_actions.insert(action);
        self
    }

    /// Builder: restrict to a territory.
    #[must_use]
    pub fn with_territory(mut self, code: &str) -> Self {
        self.territories.insert(code.to_uppercase());
        self
    }

    /// Builder: restrict to a platform.
    #[must_use]
    pub fn with_platform(mut self, platform: &str) -> Self {
        self.platforms.insert(platform.to_lowercase());
        self
    }

    /// Builder: set validity window.
    #[must_use]
    pub fn with_window(mut self, from: u64, until: u64) -> Self {
        self.valid_from = from;
        self.valid_until = until;
        self
    }

    /// Check whether the grant covers a specific action.
    #[must_use]
    pub fn permits_action(&self, action: ActionKind) -> bool {
        self.allowed_actions.contains(&action)
    }

    /// Check whether the grant is valid at a given point in time.
    #[must_use]
    pub fn is_valid_at(&self, now: u64) -> bool {
        !self.revoked && now >= self.valid_from && now < self.valid_until
    }

    /// Check whether the grant covers a territory.
    /// An empty territory set means worldwide.
    #[must_use]
    pub fn covers_territory(&self, code: &str) -> bool {
        self.territories.is_empty() || self.territories.contains(&code.to_uppercase())
    }

    /// Check whether the grant covers a platform.
    /// An empty platform set means all platforms.
    #[must_use]
    pub fn covers_platform(&self, platform: &str) -> bool {
        self.platforms.is_empty() || self.platforms.contains(&platform.to_lowercase())
    }
}

// ── CheckRequest ───────────────────────────────────────────────────────────

/// Parameters for a rights check.
#[derive(Debug, Clone)]
pub struct CheckRequest {
    /// Asset being accessed.
    pub asset_id: String,
    /// Desired action.
    pub action: ActionKind,
    /// Territory code of the requester.
    pub territory: String,
    /// Platform where the action takes place.
    pub platform: String,
    /// Current time (Unix seconds).
    pub now: u64,
}

impl CheckRequest {
    /// Create a new check request.
    #[must_use]
    pub fn new(
        asset_id: &str,
        action: ActionKind,
        territory: &str,
        platform: &str,
        now: u64,
    ) -> Self {
        Self {
            asset_id: asset_id.to_string(),
            action,
            territory: territory.to_uppercase(),
            platform: platform.to_lowercase(),
            now,
        }
    }
}

// ── CheckResult ────────────────────────────────────────────────────────────

/// Result of a rights check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckResult {
    /// The action is permitted; includes the grant ID that authorised it.
    Allowed(String),
    /// The action is denied with a reason string.
    Denied(String),
}

impl CheckResult {
    /// Whether the result is `Allowed`.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed(_))
    }

    /// Whether the result is `Denied`.
    #[must_use]
    pub fn is_denied(&self) -> bool {
        matches!(self, Self::Denied(_))
    }

    /// Return the denial reason, if any.
    #[must_use]
    pub fn denial_reason(&self) -> Option<&str> {
        match self {
            Self::Denied(r) => Some(r),
            _ => None,
        }
    }
}

// ── RightsChecker ──────────────────────────────────────────────────────────

/// The rights-checking engine that evaluates a [`CheckRequest`] against a
/// set of [`RightsGrant`]s.
#[derive(Debug, Clone, Default)]
pub struct RightsChecker {
    /// All registered grants.
    grants: Vec<RightsGrant>,
}

impl RightsChecker {
    /// Create a new, empty checker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a grant.
    pub fn add_grant(&mut self, grant: RightsGrant) {
        self.grants.push(grant);
    }

    /// Number of registered grants.
    #[must_use]
    pub fn grant_count(&self) -> usize {
        self.grants.len()
    }

    /// Evaluate a check request.
    ///
    /// Returns [`CheckResult::Allowed`] if at least one non-revoked grant
    /// matches all criteria, otherwise [`CheckResult::Denied`].
    #[must_use]
    pub fn check(&self, req: &CheckRequest) -> CheckResult {
        for grant in &self.grants {
            if grant.asset_id != req.asset_id {
                continue;
            }
            if grant.revoked {
                continue;
            }
            if !grant.permits_action(req.action) {
                continue;
            }
            if !grant.is_valid_at(req.now) {
                continue;
            }
            if !grant.covers_territory(&req.territory) {
                continue;
            }
            if !grant.covers_platform(&req.platform) {
                continue;
            }
            return CheckResult::Allowed(grant.id.clone());
        }
        CheckResult::Denied(format!(
            "No grant found for asset={} action={:?} territory={} platform={}",
            req.asset_id, req.action, req.territory, req.platform,
        ))
    }

    /// Convenience: check whether an action is allowed.
    #[must_use]
    pub fn is_allowed(&self, req: &CheckRequest) -> bool {
        self.check(req).is_allowed()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn worldwide_stream_grant() -> RightsGrant {
        RightsGrant::new("g1", "asset-001")
            .with_action(ActionKind::Stream)
            .with_action(ActionKind::Download)
    }

    fn us_only_broadcast_grant() -> RightsGrant {
        RightsGrant::new("g2", "asset-001")
            .with_action(ActionKind::Broadcast)
            .with_territory("US")
            .with_window(1000, 2000)
    }

    fn platform_grant() -> RightsGrant {
        RightsGrant::new("g3", "asset-002")
            .with_action(ActionKind::Stream)
            .with_platform("web")
    }

    #[test]
    fn test_worldwide_stream_allowed() {
        let mut checker = RightsChecker::new();
        checker.add_grant(worldwide_stream_grant());
        let req = CheckRequest::new("asset-001", ActionKind::Stream, "GB", "web", 500);
        assert!(checker.is_allowed(&req));
    }

    #[test]
    fn test_worldwide_download_allowed() {
        let mut checker = RightsChecker::new();
        checker.add_grant(worldwide_stream_grant());
        let req = CheckRequest::new("asset-001", ActionKind::Download, "JP", "mobile", 500);
        assert!(checker.is_allowed(&req));
    }

    #[test]
    fn test_action_not_granted() {
        let mut checker = RightsChecker::new();
        checker.add_grant(worldwide_stream_grant());
        let req = CheckRequest::new("asset-001", ActionKind::Broadcast, "US", "tv", 500);
        let result = checker.check(&req);
        assert!(result.is_denied());
    }

    #[test]
    fn test_territory_restriction() {
        let mut checker = RightsChecker::new();
        checker.add_grant(us_only_broadcast_grant());
        // Allowed in US within the time window
        let req = CheckRequest::new("asset-001", ActionKind::Broadcast, "US", "tv", 1500);
        assert!(checker.is_allowed(&req));
        // Denied in GB
        let req2 = CheckRequest::new("asset-001", ActionKind::Broadcast, "GB", "tv", 1500);
        assert!(!checker.is_allowed(&req2));
    }

    #[test]
    fn test_time_window_before() {
        let mut checker = RightsChecker::new();
        checker.add_grant(us_only_broadcast_grant());
        let req = CheckRequest::new("asset-001", ActionKind::Broadcast, "US", "tv", 999);
        assert!(!checker.is_allowed(&req));
    }

    #[test]
    fn test_time_window_after() {
        let mut checker = RightsChecker::new();
        checker.add_grant(us_only_broadcast_grant());
        let req = CheckRequest::new("asset-001", ActionKind::Broadcast, "US", "tv", 2000);
        assert!(!checker.is_allowed(&req));
    }

    #[test]
    fn test_platform_restriction_allowed() {
        let mut checker = RightsChecker::new();
        checker.add_grant(platform_grant());
        let req = CheckRequest::new("asset-002", ActionKind::Stream, "US", "web", 100);
        assert!(checker.is_allowed(&req));
    }

    #[test]
    fn test_platform_restriction_denied() {
        let mut checker = RightsChecker::new();
        checker.add_grant(platform_grant());
        let req = CheckRequest::new("asset-002", ActionKind::Stream, "US", "mobile", 100);
        assert!(!checker.is_allowed(&req));
    }

    #[test]
    fn test_revoked_grant_denied() {
        let mut checker = RightsChecker::new();
        let mut grant = worldwide_stream_grant();
        grant.revoked = true;
        checker.add_grant(grant);
        let req = CheckRequest::new("asset-001", ActionKind::Stream, "US", "web", 100);
        assert!(!checker.is_allowed(&req));
    }

    #[test]
    fn test_wrong_asset_denied() {
        let mut checker = RightsChecker::new();
        checker.add_grant(worldwide_stream_grant());
        let req = CheckRequest::new("asset-999", ActionKind::Stream, "US", "web", 100);
        assert!(!checker.is_allowed(&req));
    }

    #[test]
    fn test_empty_checker_denied() {
        let checker = RightsChecker::new();
        let req = CheckRequest::new("asset-001", ActionKind::Stream, "US", "web", 100);
        assert!(checker.check(&req).is_denied());
    }

    #[test]
    fn test_check_result_denial_reason() {
        let checker = RightsChecker::new();
        let req = CheckRequest::new("asset-001", ActionKind::Stream, "US", "web", 100);
        let result = checker.check(&req);
        assert!(result.denial_reason().is_some());
        assert!(result
            .denial_reason()
            .expect("rights test operation should succeed")
            .contains("asset-001"));
    }

    #[test]
    fn test_grant_count() {
        let mut checker = RightsChecker::new();
        checker.add_grant(worldwide_stream_grant());
        checker.add_grant(us_only_broadcast_grant());
        assert_eq!(checker.grant_count(), 2);
    }

    #[test]
    fn test_check_result_is_allowed_flag() {
        let r = CheckResult::Allowed("g1".into());
        assert!(r.is_allowed());
        assert!(!r.is_denied());
    }
}
