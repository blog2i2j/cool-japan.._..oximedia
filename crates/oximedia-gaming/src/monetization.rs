//! Stream monetization models, revenue splits, and configuration.
//!
//! Provides [`MonetizationType`] to classify income sources,
//! [`RevenueSplit`] to define how earnings are divided among parties,
//! and [`MonetizationConfig`] to aggregate all monetization rules for a
//! streaming session.

#![allow(dead_code)]

use std::collections::HashMap;

/// Classification of stream revenue sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MonetizationType {
    /// Recurring subscriber-based income.
    Subscription,
    /// One-time donations / tips from viewers.
    Donation,
    /// Revenue from ads shown during the stream.
    AdRevenue,
    /// Sponsored segment or product placement.
    Sponsorship,
    /// Revenue from selling branded merchandise.
    Merchandise,
    /// Platform-specific virtual gifts or tokens (e.g. bits, stars).
    VirtualGifts,
    /// Paid access to premium/exclusive content.
    PaidContent,
}

impl MonetizationType {
    /// Human-readable label suitable for dashboards and overlays.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Subscription => "Subscription",
            Self::Donation => "Donation",
            Self::AdRevenue => "Ad Revenue",
            Self::Sponsorship => "Sponsorship",
            Self::Merchandise => "Merchandise",
            Self::VirtualGifts => "Virtual Gifts",
            Self::PaidContent => "Paid Content",
        }
    }

    /// Whether this type is typically recurring (vs. one-off).
    #[must_use]
    pub fn is_recurring(self) -> bool {
        matches!(self, Self::Subscription | Self::Sponsorship)
    }

    /// Whether this type is directly viewer-initiated.
    #[must_use]
    pub fn is_viewer_initiated(self) -> bool {
        matches!(
            self,
            Self::Donation | Self::VirtualGifts | Self::Subscription
        )
    }
}

/// Describes how revenue is split among multiple parties.
///
/// Each party has a name and a share expressed as a fraction in `[0.0, 1.0]`.
/// The total of all shares should equal 1.0 (100 %).
#[derive(Debug, Clone, PartialEq)]
pub struct RevenueSplit {
    /// Ordered list of (`party_name`, `share_fraction`) pairs.
    shares: Vec<(String, f64)>,
}

impl RevenueSplit {
    /// Create an empty split (no parties).
    #[must_use]
    pub fn new() -> Self {
        Self { shares: Vec::new() }
    }

    /// Add a party to the split.
    ///
    /// `share` is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn add_party(mut self, name: impl Into<String>, share: f64) -> Self {
        self.shares.push((name.into(), share.clamp(0.0, 1.0)));
        self
    }

    /// Total of all share fractions.
    #[must_use]
    pub fn total_share(&self) -> f64 {
        self.shares.iter().map(|(_, s)| s).sum()
    }

    /// Whether shares sum to approximately 1.0 (within epsilon).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        (self.total_share() - 1.0).abs() < 1e-6
    }

    /// Number of parties in the split.
    #[must_use]
    pub fn party_count(&self) -> usize {
        self.shares.len()
    }

    /// Look up the share for a specific party by name.
    #[must_use]
    pub fn share_for(&self, name: &str) -> Option<f64> {
        self.shares.iter().find(|(n, _)| n == name).map(|(_, s)| *s)
    }

    /// Compute absolute amounts for each party given a total revenue amount.
    #[must_use]
    pub fn distribute(&self, total: f64) -> Vec<(String, f64)> {
        self.shares
            .iter()
            .map(|(name, share)| (name.clone(), total * share))
            .collect()
    }

    /// All parties and their shares.
    #[must_use]
    pub fn parties(&self) -> &[(String, f64)] {
        &self.shares
    }
}

impl Default for RevenueSplit {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated monetization configuration for a streaming session.
///
/// Maps each [`MonetizationType`] to a [`RevenueSplit`] and optional settings.
#[derive(Debug, Clone)]
pub struct MonetizationConfig {
    /// Revenue splits keyed by monetization type.
    splits: HashMap<MonetizationType, RevenueSplit>,
    /// Minimum donation amount (in smallest currency unit, e.g. cents).
    pub min_donation_cents: u64,
    /// Whether ads are enabled.
    pub ads_enabled: bool,
    /// Maximum number of ad breaks per hour.
    pub max_ad_breaks_per_hour: u32,
    /// Currency code (e.g. "USD", "EUR").
    pub currency: String,
}

impl MonetizationConfig {
    /// Create a new configuration with sensible defaults.
    #[must_use]
    pub fn new(currency: impl Into<String>) -> Self {
        Self {
            splits: HashMap::new(),
            min_donation_cents: 100, // $1.00
            ads_enabled: true,
            max_ad_breaks_per_hour: 3,
            currency: currency.into(),
        }
    }

    /// Set the revenue split for a given monetization type.
    #[must_use]
    pub fn with_split(mut self, typ: MonetizationType, split: RevenueSplit) -> Self {
        self.splits.insert(typ, split);
        self
    }

    /// Set the minimum donation amount (in cents).
    #[must_use]
    pub fn with_min_donation(mut self, cents: u64) -> Self {
        self.min_donation_cents = cents;
        self
    }

    /// Enable or disable ads.
    #[must_use]
    pub fn with_ads(mut self, enabled: bool) -> Self {
        self.ads_enabled = enabled;
        self
    }

    /// Set maximum ad breaks per hour.
    #[must_use]
    pub fn with_max_ad_breaks(mut self, count: u32) -> Self {
        self.max_ad_breaks_per_hour = count;
        self
    }

    /// Retrieve the revenue split for a specific type.
    #[must_use]
    pub fn split_for(&self, typ: &MonetizationType) -> Option<&RevenueSplit> {
        self.splits.get(typ)
    }

    /// Number of configured monetization types.
    #[must_use]
    pub fn configured_types(&self) -> usize {
        self.splits.len()
    }

    /// Validate that all configured splits sum to 1.0.
    #[must_use]
    pub fn validate(&self) -> Vec<(MonetizationType, f64)> {
        self.splits
            .iter()
            .filter(|(_, split)| !split.is_valid())
            .map(|(typ, split)| (*typ, split.total_share()))
            .collect()
    }

    /// Calculate total payout across all types given per-type gross revenue.
    #[must_use]
    pub fn calculate_payouts(
        &self,
        revenues: &HashMap<MonetizationType, f64>,
    ) -> HashMap<String, f64> {
        let mut payouts: HashMap<String, f64> = HashMap::new();
        for (typ, gross) in revenues {
            if let Some(split) = self.splits.get(typ) {
                for (party, amount) in split.distribute(*gross) {
                    *payouts.entry(party).or_default() += amount;
                }
            }
        }
        payouts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- MonetizationType ---

    #[test]
    fn test_monetization_labels() {
        assert_eq!(MonetizationType::Subscription.label(), "Subscription");
        assert_eq!(MonetizationType::AdRevenue.label(), "Ad Revenue");
        assert_eq!(MonetizationType::VirtualGifts.label(), "Virtual Gifts");
    }

    #[test]
    fn test_is_recurring() {
        assert!(MonetizationType::Subscription.is_recurring());
        assert!(MonetizationType::Sponsorship.is_recurring());
        assert!(!MonetizationType::Donation.is_recurring());
    }

    #[test]
    fn test_is_viewer_initiated() {
        assert!(MonetizationType::Donation.is_viewer_initiated());
        assert!(MonetizationType::VirtualGifts.is_viewer_initiated());
        assert!(!MonetizationType::AdRevenue.is_viewer_initiated());
    }

    // --- RevenueSplit ---

    #[test]
    fn test_revenue_split_creation() {
        let split = RevenueSplit::new()
            .add_party("Streamer", 0.7)
            .add_party("Platform", 0.3);
        assert_eq!(split.party_count(), 2);
        assert!(split.is_valid());
    }

    #[test]
    fn test_revenue_split_invalid_total() {
        let split = RevenueSplit::new().add_party("A", 0.5).add_party("B", 0.3);
        assert!(!split.is_valid());
    }

    #[test]
    fn test_share_for() {
        let split = RevenueSplit::new()
            .add_party("Creator", 0.6)
            .add_party("Network", 0.4);
        assert!((split.share_for("Creator").expect("share should exist") - 0.6).abs() < 1e-9);
        assert!(split.share_for("Nobody").is_none());
    }

    #[test]
    fn test_distribute() {
        let split = RevenueSplit::new()
            .add_party("Creator", 0.7)
            .add_party("Platform", 0.3);
        let dist = split.distribute(1000.0);
        assert_eq!(dist.len(), 2);
        assert!((dist[0].1 - 700.0).abs() < 1e-6);
        assert!((dist[1].1 - 300.0).abs() < 1e-6);
    }

    #[test]
    fn test_share_clamping() {
        let split = RevenueSplit::new().add_party("Over", 1.5);
        assert!((split.share_for("Over").expect("share should exist") - 1.0).abs() < 1e-9);
    }

    // --- MonetizationConfig ---

    #[test]
    fn test_config_defaults() {
        let cfg = MonetizationConfig::new("USD");
        assert_eq!(cfg.currency, "USD");
        assert_eq!(cfg.min_donation_cents, 100);
        assert!(cfg.ads_enabled);
        assert_eq!(cfg.max_ad_breaks_per_hour, 3);
    }

    #[test]
    fn test_config_with_split() {
        let split = RevenueSplit::new()
            .add_party("Streamer", 0.5)
            .add_party("Platform", 0.5);
        let cfg = MonetizationConfig::new("EUR").with_split(MonetizationType::Subscription, split);
        assert_eq!(cfg.configured_types(), 1);
        assert!(cfg.split_for(&MonetizationType::Subscription).is_some());
    }

    #[test]
    fn test_config_validate_detects_bad_split() {
        let bad = RevenueSplit::new().add_party("Solo", 0.8);
        let cfg = MonetizationConfig::new("USD").with_split(MonetizationType::Donation, bad);
        let issues = cfg.validate();
        assert_eq!(issues.len(), 1);
    }

    #[test]
    fn test_config_calculate_payouts() {
        let sub_split = RevenueSplit::new()
            .add_party("Creator", 0.7)
            .add_party("Platform", 0.3);
        let ad_split = RevenueSplit::new()
            .add_party("Creator", 0.55)
            .add_party("Platform", 0.45);
        let cfg = MonetizationConfig::new("USD")
            .with_split(MonetizationType::Subscription, sub_split)
            .with_split(MonetizationType::AdRevenue, ad_split);

        let mut revenues = HashMap::new();
        revenues.insert(MonetizationType::Subscription, 1000.0);
        revenues.insert(MonetizationType::AdRevenue, 500.0);

        let payouts = cfg.calculate_payouts(&revenues);
        // Creator: 700 + 275 = 975
        assert!((payouts["Creator"] - 975.0).abs() < 1e-6);
        // Platform: 300 + 225 = 525
        assert!((payouts["Platform"] - 525.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = MonetizationConfig::new("JPY")
            .with_min_donation(500)
            .with_ads(false)
            .with_max_ad_breaks(0);
        assert_eq!(cfg.min_donation_cents, 500);
        assert!(!cfg.ads_enabled);
        assert_eq!(cfg.max_ad_breaks_per_hour, 0);
    }

    #[test]
    fn test_empty_config_validate() {
        let cfg = MonetizationConfig::new("GBP");
        let issues = cfg.validate();
        assert!(issues.is_empty());
    }
}
