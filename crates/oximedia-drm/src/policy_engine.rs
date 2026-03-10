//! DRM policy engine: play rules, offline permissions, geographic restrictions.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A country code in ISO 3166-1 alpha-2 format.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CountryCode(pub String);

impl CountryCode {
    /// Create a new country code (uppercased).
    #[must_use]
    pub fn new(code: &str) -> Self {
        Self(code.to_uppercase())
    }
}

/// Specifies which countries are allowed or denied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeoRestriction {
    /// Allow playback only in these countries.
    Allowlist(HashSet<CountryCode>),
    /// Deny playback in these countries, allow everywhere else.
    Denylist(HashSet<CountryCode>),
    /// No geographic restrictions.
    Unrestricted,
}

impl GeoRestriction {
    /// Returns `true` if playback is permitted in the given country.
    #[must_use]
    pub fn is_permitted(&self, country: &CountryCode) -> bool {
        match self {
            Self::Allowlist(allowed) => allowed.contains(country),
            Self::Denylist(denied) => !denied.contains(country),
            Self::Unrestricted => true,
        }
    }
}

/// Time window during which playback is permitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayWindow {
    /// Unix timestamp (seconds) when the window opens. `None` = no lower bound.
    pub not_before: Option<i64>,
    /// Unix timestamp (seconds) when the window closes. `None` = no upper bound.
    pub not_after: Option<i64>,
}

impl PlayWindow {
    /// Create an open-ended window.
    #[must_use]
    pub fn open() -> Self {
        Self {
            not_before: None,
            not_after: None,
        }
    }

    /// Create a bounded window.
    #[must_use]
    pub fn bounded(not_before: i64, not_after: i64) -> Self {
        Self {
            not_before: Some(not_before),
            not_after: Some(not_after),
        }
    }

    /// Returns `true` if the given Unix timestamp falls within the window.
    #[must_use]
    pub fn contains(&self, ts: i64) -> bool {
        if let Some(nb) = self.not_before {
            if ts < nb {
                return false;
            }
        }
        if let Some(na) = self.not_after {
            if ts > na {
                return false;
            }
        }
        true
    }
}

/// Offline playback permission settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflinePermission {
    /// Whether offline playback is permitted at all.
    pub allowed: bool,
    /// Maximum duration (seconds) for an offline licence to remain valid.
    pub max_offline_duration_secs: Option<u64>,
    /// Maximum number of times the content may be played offline.
    pub max_offline_plays: Option<u32>,
}

impl OfflinePermission {
    /// Offline playback fully disabled.
    #[must_use]
    pub fn denied() -> Self {
        Self {
            allowed: false,
            max_offline_duration_secs: None,
            max_offline_plays: None,
        }
    }

    /// Offline playback allowed with a duration limit.
    #[must_use]
    pub fn with_duration(duration_secs: u64) -> Self {
        Self {
            allowed: true,
            max_offline_duration_secs: Some(duration_secs),
            max_offline_plays: None,
        }
    }

    /// Offline playback allowed with a play-count limit.
    #[must_use]
    pub fn with_play_count(count: u32) -> Self {
        Self {
            allowed: true,
            max_offline_duration_secs: None,
            max_offline_plays: Some(count),
        }
    }
}

/// Resolution cap applied by the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ResolutionCap {
    /// Cap at 480p (SD).
    Sd480p,
    /// Cap at 720p (HD).
    Hd720p,
    /// Cap at 1080p (Full HD).
    Fhd1080p,
    /// Cap at 4K (UHD).
    Uhd4k,
    /// No restriction.
    Unlimited,
}

/// A complete DRM play policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayPolicy {
    /// Content identifier this policy applies to.
    pub content_id: String,
    /// Geographic restrictions.
    pub geo: GeoRestriction,
    /// Time window for playback.
    pub play_window: PlayWindow,
    /// Offline permission settings.
    pub offline: OfflinePermission,
    /// Whether rental (time-limited) mode is in effect.
    pub is_rental: bool,
    /// Rental expiry (Unix timestamp). Only relevant if `is_rental` is `true`.
    pub rental_expiry: Option<i64>,
    /// Maximum resolution allowed.
    pub resolution_cap: ResolutionCap,
    /// Whether screen capture is blocked.
    pub block_screen_capture: bool,
}

impl PlayPolicy {
    /// Create a permissive default policy for the given content ID.
    #[must_use]
    pub fn permissive(content_id: impl Into<String>) -> Self {
        Self {
            content_id: content_id.into(),
            geo: GeoRestriction::Unrestricted,
            play_window: PlayWindow::open(),
            offline: OfflinePermission::denied(),
            is_rental: false,
            rental_expiry: None,
            resolution_cap: ResolutionCap::Unlimited,
            block_screen_capture: false,
        }
    }

    /// Evaluate whether playback is permitted given the context.
    #[must_use]
    pub fn evaluate(&self, ctx: &PlayContext) -> PolicyDecision {
        // Geographic check.
        if let Some(ref country) = ctx.country {
            if !self.geo.is_permitted(country) {
                return PolicyDecision::denied("Geographic restriction");
            }
        }
        // Time window check.
        if !self.play_window.contains(ctx.current_time) {
            return PolicyDecision::denied("Outside play window");
        }
        // Rental expiry check.
        if self.is_rental {
            if let Some(expiry) = self.rental_expiry {
                if ctx.current_time > expiry {
                    return PolicyDecision::denied("Rental expired");
                }
            }
        }
        // Offline check.
        if ctx.is_offline && !self.offline.allowed {
            return PolicyDecision::denied("Offline playback not permitted");
        }
        PolicyDecision::allowed(self.resolution_cap)
    }
}

/// Context provided when evaluating a policy.
#[derive(Debug, Clone)]
pub struct PlayContext {
    /// Country of the requesting device. `None` = unknown.
    pub country: Option<CountryCode>,
    /// Current Unix timestamp.
    pub current_time: i64,
    /// Whether the device is offline.
    pub is_offline: bool,
}

impl PlayContext {
    /// Create a simple online context.
    #[must_use]
    pub fn online(current_time: i64, country: Option<CountryCode>) -> Self {
        Self {
            country,
            current_time,
            is_offline: false,
        }
    }
}

/// Result of a policy evaluation.
#[derive(Debug, Clone)]
pub struct PolicyDecision {
    /// Whether playback is permitted.
    pub permitted: bool,
    /// Reason if denied.
    pub reason: Option<String>,
    /// Maximum resolution permitted (only meaningful if `permitted` is `true`).
    pub resolution_cap: ResolutionCap,
}

impl PolicyDecision {
    /// Construct an allowed decision.
    #[must_use]
    pub fn allowed(resolution_cap: ResolutionCap) -> Self {
        Self {
            permitted: true,
            reason: None,
            resolution_cap,
        }
    }

    /// Construct a denied decision.
    #[must_use]
    pub fn denied(reason: &str) -> Self {
        Self {
            permitted: false,
            reason: Some(reason.to_string()),
            resolution_cap: ResolutionCap::Sd480p,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn us() -> CountryCode {
        CountryCode::new("us")
    }

    fn de() -> CountryCode {
        CountryCode::new("de")
    }

    #[test]
    fn test_country_code_uppercase() {
        assert_eq!(CountryCode::new("us").0, "US");
    }

    #[test]
    fn test_geo_allowlist_permits() {
        let mut set = HashSet::new();
        set.insert(us());
        let geo = GeoRestriction::Allowlist(set);
        assert!(geo.is_permitted(&us()));
        assert!(!geo.is_permitted(&de()));
    }

    #[test]
    fn test_geo_denylist_permits() {
        let mut set = HashSet::new();
        set.insert(de());
        let geo = GeoRestriction::Denylist(set);
        assert!(geo.is_permitted(&us()));
        assert!(!geo.is_permitted(&de()));
    }

    #[test]
    fn test_geo_unrestricted() {
        let geo = GeoRestriction::Unrestricted;
        assert!(geo.is_permitted(&us()));
        assert!(geo.is_permitted(&de()));
    }

    #[test]
    fn test_play_window_open_always_true() {
        let w = PlayWindow::open();
        assert!(w.contains(0));
        assert!(w.contains(i64::MAX));
        assert!(w.contains(i64::MIN));
    }

    #[test]
    fn test_play_window_bounded_inside() {
        let w = PlayWindow::bounded(1000, 2000);
        assert!(w.contains(1500));
        assert!(!w.contains(999));
        assert!(!w.contains(2001));
    }

    #[test]
    fn test_offline_denied() {
        let o = OfflinePermission::denied();
        assert!(!o.allowed);
    }

    #[test]
    fn test_offline_with_duration() {
        let o = OfflinePermission::with_duration(86400);
        assert!(o.allowed);
        assert_eq!(o.max_offline_duration_secs, Some(86400));
    }

    #[test]
    fn test_offline_with_play_count() {
        let o = OfflinePermission::with_play_count(5);
        assert!(o.allowed);
        assert_eq!(o.max_offline_plays, Some(5));
    }

    #[test]
    fn test_policy_permissive_allows_all() {
        let policy = PlayPolicy::permissive("content-001");
        let ctx = PlayContext::online(1_000_000, Some(us()));
        let decision = policy.evaluate(&ctx);
        assert!(decision.permitted);
    }

    #[test]
    fn test_policy_geo_block() {
        let mut denied_set = HashSet::new();
        denied_set.insert(de());
        let mut policy = PlayPolicy::permissive("content-002");
        policy.geo = GeoRestriction::Denylist(denied_set);
        let ctx = PlayContext::online(1_000_000, Some(de()));
        let decision = policy.evaluate(&ctx);
        assert!(!decision.permitted);
        assert!(decision
            .reason
            .as_deref()
            .expect("value should exist")
            .contains("Geographic"));
    }

    #[test]
    fn test_policy_time_window_block() {
        let mut policy = PlayPolicy::permissive("content-003");
        policy.play_window = PlayWindow::bounded(2000, 3000);
        let ctx = PlayContext::online(1000, None); // before window
        let decision = policy.evaluate(&ctx);
        assert!(!decision.permitted);
    }

    #[test]
    fn test_policy_rental_expired() {
        let mut policy = PlayPolicy::permissive("content-004");
        policy.is_rental = true;
        policy.rental_expiry = Some(500);
        let ctx = PlayContext::online(1000, None); // after expiry
        let decision = policy.evaluate(&ctx);
        assert!(!decision.permitted);
        assert!(decision
            .reason
            .as_deref()
            .expect("value should exist")
            .contains("Rental"));
    }

    #[test]
    fn test_policy_offline_block() {
        let policy = PlayPolicy::permissive("content-005");
        // offline.allowed is false by default in permissive()
        let ctx = PlayContext {
            country: None,
            current_time: 1000,
            is_offline: true,
        };
        let decision = policy.evaluate(&ctx);
        assert!(!decision.permitted);
    }

    #[test]
    fn test_resolution_cap_ordering() {
        assert!(ResolutionCap::Sd480p < ResolutionCap::Hd720p);
        assert!(ResolutionCap::Hd720p < ResolutionCap::Fhd1080p);
        assert!(ResolutionCap::Fhd1080p < ResolutionCap::Uhd4k);
        assert!(ResolutionCap::Uhd4k < ResolutionCap::Unlimited);
    }

    #[test]
    fn test_policy_decision_allowed_has_cap() {
        let d = PolicyDecision::allowed(ResolutionCap::Hd720p);
        assert!(d.permitted);
        assert_eq!(d.resolution_cap, ResolutionCap::Hd720p);
        assert!(d.reason.is_none());
    }
}
